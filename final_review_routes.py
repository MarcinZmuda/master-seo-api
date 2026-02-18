# ================================================================
# 🔍 FINAL REVIEW ROUTES v40.1 - ORIGINAL_MAX FIX
# ================================================================
# ZMIANY v40.1:
# - FIX: Używa original_max zamiast target_max dla zredukowanych fraz
# - TYLKO stuffing (>max) blokuje
# - Brak frazy (0×) = warning, Claude uzupełni
# - underused (<target) = OK
#
# ZMIANY v29.2:
# - TYLKO stuffing (>max) blokuje
# - Brak frazy (0×) = warning, Claude uzupełni
# ================================================================

import os
import re
import json
import traceback
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import google.generativeai as genai

# v24.2: Unified keyword counting
try:
    from keyword_counter import count_single_keyword, count_keywords_for_state, get_keyword_details
    UNIFIED_COUNTER = True
    print("[FINAL_REVIEW] ✅ Unified keyword counter loaded")
except ImportError:
    UNIFIED_COUNTER = False
    print("[FINAL_REVIEW] ⚠️ keyword_counter not available, using legacy regex")

# 🆕 v40.1: Advanced Semantic Features (source effort, topic completeness)
try:
    from advanced_semantic_features import (
        calculate_source_effort_v2,
        analyze_topic_completeness,
        perform_advanced_semantic_analysis
    )
    ADVANCED_SEMANTIC_ENABLED = True
    print("[FINAL_REVIEW] ✅ Advanced semantic features loaded")
except ImportError as e:
    ADVANCED_SEMANTIC_ENABLED = False
    print(f"[FINAL_REVIEW] ⚠️ Advanced semantic features not available: {e}")

# 🆕 v40.2: Entity Scoring for Semantic SEO
try:
    from entity_scoring import (
        calculate_entity_score,
        calculate_entity_coverage,
        calculate_entity_density
    )
    ENTITY_SCORING_ENABLED = True
    print("[FINAL_REVIEW] ✅ Entity scoring loaded")
except ImportError as e:
    ENTITY_SCORING_ENABLED = False
    print(f"[FINAL_REVIEW] ⚠️ Entity scoring not available: {e}")

final_review_routes = Blueprint("final_review_routes", __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[FINAL_REVIEW] ✅ Gemini API configured")
else:
    print("[FINAL_REVIEW] ⚠️ GEMINI_API_KEY not set")

GEMINI_MODEL = os.getenv("FINAL_REVIEW_MODEL", "gemini-2.5-flash")

# 🆕 v40.1: Tolerancja dla przekroczenia limitu
# 🆕 v42.1: Różne tolerancje dla BASIC i EXTENDED
BASIC_TOLERANCE_PERCENT = 30    # 30% tolerancji dla BASIC/MAIN
EXTENDED_TOLERANCE_PERCENT = 50  # 50% tolerancji dla EXTENDED
# Backwards compatibility - używane w niektórych miejscach jako default
TOLERANCE_PERCENT = BASIC_TOLERANCE_PERCENT


# ================================================================
# 1. MISSING KEYWORDS DETECTOR - v40.1 (ORIGINAL_MAX FIX)
# ================================================================
def detect_missing_keywords(text, keywords_state):
    """
    v40.1: Wykrywa brakujące frazy - używa ORIGINAL_MAX dla zredukowanych fraz!
    
    WAŻNE:
    - Frazy mogą mieć zredukowany target_max (np. 24→2) przez AUTO-REDUKCJA
    - original_max zawiera ORYGINALNY limit podany przez użytkownika
    - Final review MUSI używać original_max, nie zredukowanego target_max
    
    Logika:
    - missing (0×) → WARNING (Claude uzupełni)
    - underused (< target) → OK
    - stuffing (> original_max) → CRITICAL (JEDYNY BLOKER!)
    
    needs_correction = True TYLKO gdy stuffing > 0
    """
    try:
        text_lower = text.lower()
        
        missing_basic = []      # 0 wystąpień - WARNING
        missing_extended = []   # 0 wystąpień - WARNING
        underused_basic = []    # < target - OK
        underused_extended = [] # < target - OK
        stuffing = []           # > max - CRITICAL (JEDYNY BLOKER!)
        within_tolerance = []   # > max ale w tolerancji 30% - WARNING
        
        # v24.2: Unified counting
        if UNIFIED_COUNTER:
            counts = count_keywords_for_state(text, keywords_state, use_exclusive_for_nested=True)
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            
            kw_type = meta.get("type", "BASIC").upper()
            target_min = meta.get("target_min", 1)
            
            # ================================================================
            # 🆕 v40.1: CRITICAL FIX - użyj ORIGINAL_MAX!
            # ================================================================
            # target_max może być zredukowany przez AUTO-REDUKCJA (np. 24→2)
            # original_max to limit który użytkownik FAKTYCZNIE podał
            original_max = meta.get("original_max")  # Zapisany przed redukcją
            target_max_current = meta.get("target_max", target_min * 3)
            
            # Jeśli jest original_max → użyj go (fraza była zredukowana)
            # Jeśli nie ma → użyj target_max (fraza nie była zredukowana)
            target_max = original_max if original_max else target_max_current
            
            # Oblicz tolerancję (30% dla BASIC/MAIN, 50% dla EXTENDED)
            # 🆕 v42.1: Różne tolerancje!
            tolerance_percent = EXTENDED_TOLERANCE_PERCENT if kw_type == "EXTENDED" else BASIC_TOLERANCE_PERCENT
            tolerance_max = int(target_max * (1 + tolerance_percent / 100))
            
            # Debug log dla zredukowanych fraz
            if original_max and original_max != target_max_current:
                print(f"[FINAL_REVIEW] ℹ️ '{keyword}': using original_max={original_max} (was reduced to {target_max_current})")
            
            # v24.2: Unified vs legacy counting
            if UNIFIED_COUNTER:
                actual = counts.get(rid, 0)
            else:
                try:
                    actual = len(re.findall(rf"\b{re.escape(keyword.lower())}\b", text_lower))
                except:
                    actual = text_lower.count(keyword.lower())
            
            info = {
                "keyword": keyword,
                "type": kw_type,
                "actual": actual,
                "target_min": target_min,
                "target_max": target_max,
                "target_max_current": target_max_current,  # Zredukowany (dla info)
                "original_max": original_max,  # Oryginalny (jeśli był)
                "tolerance_max": tolerance_max,
                "tolerance_percent": tolerance_percent,  # 🆕 v42.1
                "missing": max(0, target_min - actual),
                "was_reduced": original_max is not None and original_max != target_max_current
            }
            
            # 🆕 v42.1: Skonwertowane keywords - pomiń jeśli do_not_use
            if meta.get("do_not_use") and actual == 0:
                print(f"[FINAL_REVIEW] ⏭️ '{keyword}': skipped (converted, do_not_use)")
                continue
            
            # v40.1 + v42.1: NOWA LOGIKA
            # EXTENDED NIGDY nie blokuje - tylko WARNING!
            if actual > tolerance_max:
                if kw_type == "EXTENDED":
                    # 🆕 v42.1: EXTENDED poza tolerancją = tylko WARNING
                    info["severity"] = "stuffing_warning"
                    info["exceeded_by"] = actual - target_max
                    within_tolerance.append(info)  # Traktuj jako warning
                    print(f"[FINAL_REVIEW] ⚠️ '{keyword}' (EXTENDED): {actual}/{target_max} - over tolerance but not blocking")
                else:
                    # BASIC/MAIN poza tolerancją = CRITICAL (blokuje)
                    info["severity"] = "stuffing_critical"
                    info["exceeded_by"] = actual - target_max
                    info["exceeded_tolerance_by"] = actual - tolerance_max
                    stuffing.append(info)
                    print(f"[FINAL_REVIEW] ❌ '{keyword}': {actual}/{target_max} (tolerance {tolerance_max}) - EXCEEDED!")
            elif actual > target_max:
                # WARNING: przekroczone ale W tolerancji
                info["severity"] = "stuffing_warning"
                info["exceeded_by"] = actual - target_max
                within_tolerance.append(info)
                print(f"[FINAL_REVIEW] ⚠️ '{keyword}': {actual}/{target_max} (tolerance {tolerance_percent}%) - within tolerance")
            elif actual == 0:
                # WARNING: fraza brakuje - Claude uzupełni
                if kw_type in ["BASIC", "MAIN", "ENTITY"]:
                    missing_basic.append(info)
                else:
                    missing_extended.append(info)
            elif actual < target_min:
                # OK: mogłoby być więcej
                if kw_type in ["BASIC", "MAIN", "ENTITY"]:
                    underused_basic.append(info)
                else:
                    underused_extended.append(info)
        
        missing_basic.sort(key=lambda x: x["missing"], reverse=True)
        
        # v40.1: needs_correction TYLKO dla stuffing POZA tolerancją!
        has_critical_stuffing = len(stuffing) > 0
        
        return {
            "missing": {"basic": missing_basic, "extended": missing_extended},
            "underused": {"basic": underused_basic, "extended": underused_extended},
            "stuffing": stuffing,  # POZA tolerancją - CRITICAL
            "within_tolerance": within_tolerance,  # W tolerancji - WARNING
            "priority_to_add": {
                "critical": stuffing,                          # TYLKO stuffing poza tolerancją jest critical
                "warning": within_tolerance,                   # W tolerancji 30%
                "to_add_by_claude": missing_basic + missing_extended,  # Claude uzupełni
                "ok": underused_basic + underused_extended     # OK, nie trzeba nic robić
            },
            "needs_correction": has_critical_stuffing,  # v40.1: TYLKO stuffing POZA tolerancją blokuje!
            "needs_claude_help": len(missing_basic) + len(missing_extended) > 0,
            "summary": {
                "missing_count": len(missing_basic) + len(missing_extended),
                "underused_count": len(underused_basic) + len(underused_extended),
                "stuffing_critical_count": len(stuffing),
                "stuffing_warning_count": len(within_tolerance),
                "tolerance_percent": {"BASIC": BASIC_TOLERANCE_PERCENT, "EXTENDED": EXTENDED_TOLERANCE_PERCENT}
            },
            "version": "v40.1"
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ detect_missing_keywords error: {e}")
        traceback.print_exc()
        return {
            "missing": {"basic": [], "extended": []},
            "underused": {"basic": [], "extended": []},
            "stuffing": [],
            "within_tolerance": [],
            "priority_to_add": {"critical": [], "warning": [], "to_add_by_claude": [], "ok": []},
            "needs_correction": False,
            "error": str(e)
        }


# ================================================================
# 2. MAIN vs SYNONYMS
# ================================================================
def check_main_vs_synonyms(text, main_keyword, keywords_state):
    """Sprawdza proporcję frazy głównej vs synonimy."""
    try:
        if not main_keyword:
            return {
                "main_keyword": "",
                "main_count": 0,
                "synonym_total": 0,
                "main_ratio": 1.0,
                "valid": True,
                "overused_synonyms": [],
                "warning": None
            }
        
        text_lower = text.lower()
        
        # v24.2: Unified counting
        if UNIFIED_COUNTER:
            main_count = count_single_keyword(text, main_keyword)
        else:
            try:
                main_count = len(re.findall(rf"\b{re.escape(main_keyword.lower())}\b", text_lower))
            except:
                main_count = text_lower.count(main_keyword.lower())
        
        synonym_counts = {}
        synonym_total = 0
        
        for rid, meta in keywords_state.items():
            if meta.get("is_synonym_of_main"):
                kw = meta.get("keyword", "").lower()
                if kw:
                    if UNIFIED_COUNTER:
                        count = count_single_keyword(text, kw)
                    else:
                        try:
                            count = len(re.findall(rf"\b{re.escape(kw)}\b", text_lower))
                        except:
                            count = text_lower.count(kw)
                    if count > 0:
                        synonym_counts[meta.get("keyword")] = count
                        synonym_total += count
        
        total = main_count + synonym_total
        main_ratio = main_count / total if total > 0 else 1.0
        
        overused = []
        for syn, count in synonym_counts.items():
            if count > main_count:
                overused.append({
                    "synonym": syn,
                    "count": count,
                    "action": f"Zamień {count - main_count}x '{syn}' na '{main_keyword}'"
                })
        
        return {
            "main_keyword": main_keyword,
            "main_count": main_count,
            "synonym_total": synonym_total,
            "main_ratio": round(main_ratio, 2),
            "valid": main_ratio >= 0.3,
            "overused_synonyms": overused,
            "warning": f"'{main_keyword}' ma tylko {main_ratio:.0%} użyć!" if main_ratio < 0.3 else None
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ check_main_vs_synonyms error: {e}")
        return {
            "main_keyword": main_keyword or "",
            "main_count": 0,
            "synonym_total": 0,
            "main_ratio": 1.0,
            "valid": True,
            "overused_synonyms": [],
            "warning": None,
            "error": str(e)
        }


# ================================================================
# 3. H2 VALIDATION
# ================================================================
def validate_h2_keywords(text, main_keyword):
    """Sprawdza czy H2 zawierają frazę główną."""
    try:
        if not main_keyword:
            return {"valid": True, "h2_count": 0, "coverage": 1.0, "issues": []}
        
        h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>|^##\s+(.+)$)'
        h2_matches = re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE)
        h2_list = [(m[0] or m[1] or m[2]).strip() for m in h2_matches if m[0] or m[1] or m[2]]
        
        if not h2_list:
            return {"valid": True, "h2_count": 0, "coverage": 1.0, "issues": []}
        
        main_lower = main_keyword.lower()
        h2_with_main = sum(1 for h2 in h2_list if main_lower in h2.lower())
        
        # v26.1: Max 1 H2 z frazą główną (unikamy przeoptymalizowania)
        overoptimized = h2_with_main > 1
        
        issues = []
        if overoptimized:
            issues.append({
                "issue": f"Za dużo H2 z frazą główną ({h2_with_main})",
                "suggestion": f"Max 1 H2 powinno zawierać '{main_keyword}'. Reszta: synonimy lub naturalne tytuły."
            })
        
        return {
            "valid": not overoptimized,
            "h2_count": len(h2_list),
            "h2_with_main": h2_with_main,
            "max_recommended": 1,
            "overoptimized": overoptimized,
            "issues": issues
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ validate_h2_keywords error: {e}")
        return {"valid": True, "h2_count": 0, "coverage": 1.0, "issues": [], "error": str(e)}


# ================================================================
# 4. H3 LENGTH
# ================================================================
def validate_h3_length(text, min_words=80):
    """Sprawdza czy sekcje H3 mają minimalną długość."""
    try:
        h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>|^###\s+(.+)$)'
        h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if not h3_matches:
            return {"valid": True, "issues": [], "total_h3": 0}
        
        issues = []
        
        for i, match in enumerate(h3_matches):
            h3_title = (match.group(1) or match.group(2) or match.group(3) or "").strip()
            start = match.end()
            end = len(text)
            
            next_h = re.search(r'^h[23]:|<h[23]|^##', text[start:], re.MULTILINE | re.IGNORECASE)
            if next_h:
                end = start + next_h.start()
            
            section = re.sub(r'<[^>]+>', '', text[start:end]).strip()
            words = len(section.split())
            
            if words < min_words:
                issues.append({
                    "h3": h3_title,
                    "word_count": words,
                    "deficit": min_words - words
                })
        
        return {"valid": len(issues) == 0, "issues": issues, "total_h3": len(h3_matches)}
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ validate_h3_length error: {e}")
        return {"valid": True, "issues": [], "total_h3": 0, "error": str(e)}


# ================================================================
# 5. LIST COUNT
# ================================================================
def count_lists(text):
    """Liczy bloki list wypunktowanych."""
    try:
        lines = text.split('\n')
        list_blocks = 0
        in_list = False
        
        for line in lines:
            is_bullet = bool(re.match(r'^\s*[-•*]\s+|^\s*\d+\.\s+', line.strip()))
            if is_bullet and not in_list:
                list_blocks += 1
                in_list = True
            elif not is_bullet and line.strip():
                in_list = False
        
        list_blocks += len(re.findall(r'<ul>|<ol>', text, re.IGNORECASE))
        
        return {
            "count": list_blocks,
            "valid": list_blocks <= 1,
            "action": f"Zamień {list_blocks - 1} list na tekst" if list_blocks > 1 else None
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ count_lists error: {e}")
        return {"count": 0, "valid": True, "action": None, "error": str(e)}


# ================================================================
# 6. N-GRAM COVERAGE
# ================================================================
def check_ngrams(text, s1_data):
    """Sprawdza pokrycie n-gramów z S1."""
    try:
        if not s1_data:
            return {"coverage": 1.0, "missing": [], "used_count": 0, "valid": True}
        
        text_lower = text.lower()
        ngrams = s1_data.get("ngrams", [])
        
        if not ngrams:
            return {"coverage": 1.0, "missing": [], "used_count": 0, "valid": True}
        
        top = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.4][:15]
        
        if not top:
            return {"coverage": 1.0, "missing": [], "used_count": 0, "valid": True}
        
        used = [ng for ng in top if ng and ng.lower() in text_lower]
        missing = [ng for ng in top if ng and ng.lower() not in text_lower]
        
        coverage = len(used) / len(top) if top else 1.0
        
        return {
            "coverage": round(coverage, 2),
            "used_count": len(used),
            "missing": missing[:5],
            "valid": coverage >= 0.6
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ check_ngrams error: {e}")
        return {"coverage": 1.0, "missing": [], "used_count": 0, "valid": True, "error": str(e)}


# ================================================================
# 7. 🆕 v40.1: KEYWORDS SUMMARY WITH TOLERANCE
# ================================================================
def get_keywords_validation_summary(missing_kw):
    """
    v40.1: Generuje podsumowanie walidacji keywords z tolerancją 30%.
    
    Returns:
        {
            "pass": [...],      # actual <= target_max
            "warning": [...],   # target_max < actual <= tolerance_max
            "fail": [...],      # actual > tolerance_max
            "overall_status": "PASS" | "WARNING" | "FAIL"
        }
    """
    pass_list = []
    warning_list = []
    fail_list = []
    
    # Stuffing poza tolerancją = FAIL
    for kw in missing_kw.get("stuffing", []):
        fail_list.append({
            "keyword": kw["keyword"],
            "actual": kw["actual"],
            "target_max": kw["target_max"],
            "tolerance_max": kw["tolerance_max"],
            "status": "FAIL",
            "message": f"Przekroczono tolerancję 30%: {kw['actual']}/{kw['target_max']} (max {kw['tolerance_max']})"
        })
    
    # Stuffing w tolerancji = WARNING
    for kw in missing_kw.get("within_tolerance", []):
        warning_list.append({
            "keyword": kw["keyword"],
            "actual": kw["actual"],
            "target_max": kw["target_max"],
            "tolerance_max": kw["tolerance_max"],
            "status": "WARNING",
            "message": f"W tolerancji 30%: {kw['actual']}/{kw['target_max']} (max {kw['tolerance_max']})"
        })
    
    # Missing = WARNING (Claude uzupełni)
    for kw in missing_kw.get("missing", {}).get("basic", []):
        warning_list.append({
            "keyword": kw["keyword"],
            "actual": kw["actual"],
            "target_min": kw["target_min"],
            "status": "WARNING",
            "message": f"Brak frazy - Claude uzupełni"
        })
    
    # Określ overall status
    if fail_list:
        overall_status = "FAIL"
    elif warning_list:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"
    
    return {
        "pass": pass_list,
        "warning": warning_list,
        "fail": fail_list,
        "overall_status": overall_status,
        "tolerance_percent": TOLERANCE_PERCENT
    }


# ================================================================
# MAIN ENDPOINT: performFinalReview
# ================================================================
@final_review_routes.get("/api/project/<project_id>/final_review")
def get_final_review(project_id):
    """v27.1: GET endpoint - zwraca istniejący final_review lub wykonuje nowy."""
    print(f"[FINAL_REVIEW] 🔍 GET request for project: {project_id}")
    
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found", "project_id": project_id}), 404
        
        data = doc.to_dict()
        
        existing_review = data.get("final_review")
        if existing_review:
            print(f"[FINAL_REVIEW] ✅ Returning existing review")
            return jsonify({
                "status": "EXISTS",
                "final_review": existing_review,
                "hint": "Use POST to run new review"
            }), 200
        
        print(f"[FINAL_REVIEW] ℹ️ No existing review, running new one...")
        return perform_final_review(project_id)
        
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    """
    v40.1: Kompleksowy audyt końcowy artykułu.
    
    ZMIANY v40.1:
    - Używa original_max zamiast target_max dla zredukowanych fraz
    - Tolerancja 30% dla przekroczenia
    - Szczegółowe podsumowanie keywords_validation
    """
    print(f"[FINAL_REVIEW] 🔍 Starting review v40.1 for project: {project_id}")
    
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            print(f"[FINAL_REVIEW] ❌ Project not found: {project_id}")
            return jsonify({"error": "Project not found", "project_id": project_id}), 404
        
        data = doc.to_dict()
        batches = data.get("batches", [])
        keywords_state = data.get("keywords_state", {})
        s1_data = data.get("s1_data", {})
        main_keyword = data.get("main_keyword", data.get("topic", ""))
        
        print(f"[FINAL_REVIEW] 📊 Project data: {len(batches)} batches, {len(keywords_state)} keywords, main: '{main_keyword}'")
        
        # Scal tekst
        full_text = "\n\n".join([b.get("text", "") for b in batches if b.get("text")])
        
        if not full_text.strip():
            print(f"[FINAL_REVIEW] ❌ No content in batches")
            return jsonify({
                "error": "No content to review",
                "project_id": project_id,
                "batches_count": len(batches)
            }), 400
        
        word_count = len(full_text.split())
        print(f"[FINAL_REVIEW] 📝 Text length: {word_count} words")
        
        # ================================================================
        # Wykonaj analizy
        # ================================================================
        print("[FINAL_REVIEW] 🔍 Running validations...")
        
        missing_kw = detect_missing_keywords(full_text, keywords_state)
        print(f"[FINAL_REVIEW] ✅ Missing keywords check done (v40.1 with original_max)")
        
        main_syn = check_main_vs_synonyms(full_text, main_keyword, keywords_state)
        print(f"[FINAL_REVIEW] ✅ Main vs synonyms check done")
        
        h2_val = validate_h2_keywords(full_text, main_keyword)
        print(f"[FINAL_REVIEW] ✅ H2 validation done")
        
        h3_val = validate_h3_length(full_text)
        print(f"[FINAL_REVIEW] ✅ H3 validation done")
        
        list_val = count_lists(full_text)
        print(f"[FINAL_REVIEW] ✅ List count done")
        
        ngram_val = check_ngrams(full_text, s1_data)
        print(f"[FINAL_REVIEW] ✅ N-gram check done")
        
        # 🆕 v40.1: Keywords validation summary
        keywords_validation = get_keywords_validation_summary(missing_kw)
        
        # ================================================================
        # Zbierz issues
        # ================================================================
        all_issues = []
        
        # 🆕 v40.1: Stuffing poza tolerancją = ERROR
        if missing_kw.get("stuffing"):
            for kw in missing_kw["stuffing"]:
                all_issues.append({
                    "type": "STUFFING_CRITICAL",
                    "severity": "ERROR",
                    "keyword": kw["keyword"],
                    "actual": kw["actual"],
                    "target_max": kw["target_max"],
                    "tolerance_max": kw["tolerance_max"],
                    "message": f"'{kw['keyword']}': {kw['actual']}/{kw['target_max']} przekracza tolerancję {TOLERANCE_PERCENT}%"
                })
        
        # 🆕 v40.1: Stuffing w tolerancji = WARNING
        if missing_kw.get("within_tolerance"):
            for kw in missing_kw["within_tolerance"]:
                all_issues.append({
                    "type": "STUFFING_WARNING",
                    "severity": "WARNING",
                    "keyword": kw["keyword"],
                    "actual": kw["actual"],
                    "target_max": kw["target_max"],
                    "tolerance_max": kw["tolerance_max"],
                    "message": f"'{kw['keyword']}': {kw['actual']}/{kw['target_max']} w tolerancji {TOLERANCE_PERCENT}%"
                })
        
        if missing_kw.get("missing", {}).get("basic"):
            all_issues.append({
                "type": "MISSING_BASIC",
                "severity": "WARNING",  # v40.1: Downgrade z ERROR do WARNING
                "keywords": [k["keyword"] for k in missing_kw["missing"]["basic"][:5]],
                "message": "Brakujące frazy - Claude uzupełni"
            })
        
        if not main_syn.get("valid", True):
            all_issues.append({
                "type": "SYNONYM_OVERUSE",
                "severity": "WARNING",
                "ratio": main_syn.get("main_ratio", 1.0),
                "warning": main_syn.get("warning")
            })
        
        if not h2_val.get("valid", True):
            all_issues.append({
                "type": "H2_OVEROPTIMIZED",
                "severity": "WARNING",
                "h2_with_main": h2_val.get("h2_with_main", 0)
            })
        
        if not h3_val.get("valid", True):
            all_issues.append({
                "type": "H3_TOO_SHORT",
                "severity": "WARNING",
                "issues": h3_val.get("issues", [])
            })
        
        if not list_val.get("valid", True):
            all_issues.append({
                "type": "TOO_MANY_LISTS",
                "severity": "WARNING",
                "count": list_val.get("count", 0)
            })
        
        if not ngram_val.get("valid", True):
            all_issues.append({
                "type": "LOW_NGRAM_COVERAGE",
                "severity": "WARNING",
                "coverage": ngram_val.get("coverage", 1.0)
            })
        
        # ================================================================
        # Status
        # ================================================================
        errors = sum(1 for i in all_issues if i.get("severity") == "ERROR")
        warnings = sum(1 for i in all_issues if i.get("severity") == "WARNING")
        
        # v40.1: TYLKO stuffing poza tolerancją blokuje
        status = "WYMAGA_POPRAWEK" if errors > 0 else ("WARN" if warnings > 2 else "OK")
        
        # ================================================================
        # Recommendations
        # ================================================================
        recommendations = []
        
        # Stuffing poza tolerancją - MUSI być naprawione
        for kw in missing_kw.get("stuffing", [])[:3]:
            excess = kw["actual"] - kw["target_max"]
            recommendations.append(f"🔴 USUŃ {excess}x '{kw['keyword']}' (aktualnie {kw['actual']}, max {kw['target_max']})")
        
        # Stuffing w tolerancji - zalecane
        for kw in missing_kw.get("within_tolerance", [])[:2]:
            excess = kw["actual"] - kw["target_max"]
            recommendations.append(f"🟡 Rozważ usunięcie {excess}x '{kw['keyword']}' (w tolerancji, ale przekroczone)")
        
        for ov in main_syn.get("overused_synonyms", [])[:2]:
            recommendations.append(ov.get("action", ""))
        
        for kw in missing_kw.get("priority_to_add", {}).get("to_add_by_claude", [])[:3]:
            recommendations.append(f"Wpleć '{kw.get('keyword', '')}' min. {kw.get('target_min', 1)}x")
        
        for issue in h3_val.get("issues", [])[:2]:
            recommendations.append(f"Rozbuduj H3 '{issue.get('h3', '')}' o {issue.get('deficit', 0)} słów")
        
        if list_val.get("action"):
            recommendations.append(list_val["action"])
        
        if ngram_val.get("missing"):
            recommendations.append(f"Wpleć: {', '.join(ngram_val.get('missing', [])[:3])}")
        
        # ================================================================
        # Score
        # ================================================================
        score = 100
        # Stuffing poza tolerancją = -15 za każde
        score -= len(missing_kw.get("stuffing", [])) * 15
        # Stuffing w tolerancji = -5 za każde
        score -= len(missing_kw.get("within_tolerance", [])) * 5
        # Missing = -3 za każde (mniejsza kara, Claude uzupełni)
        score -= len(missing_kw.get("missing", {}).get("basic", [])) * 3
        score -= len(missing_kw.get("underused", {}).get("basic", [])) * 1
        if main_syn.get("main_ratio", 1.0) < 0.3:
            score -= 15
        if h2_val.get("overoptimized"):
            score -= 5
        score -= len(h3_val.get("issues", [])) * 3
        if list_val.get("count", 0) > 1:
            score -= (list_val["count"] - 1) * 5
        if ngram_val.get("coverage", 1.0) < 0.6:
            score -= 10
        score = max(0, min(100, score))
        
        # ================================================================
        # 🆕 v40.1: ADVANCED SEMANTIC ANALYSIS
        # ================================================================
        advanced_semantic = None
        if ADVANCED_SEMANTIC_ENABLED:
            try:
                # Source Effort Score - czy artykuł ma sygnały ekspertyzji
                source_effort = calculate_source_effort_v2(full_text)
                
                # Topic Completeness - czy pokrywa temat kompleksowo
                expected_entities = []
                entity_seo = s1_data.get("entity_seo", {})
                if not isinstance(entity_seo, dict):
                    entity_seo = {}
                for ent in entity_seo.get("entities", [])[:20]:
                    expected_entities.append(ent.get("name", "") if isinstance(ent, dict) else str(ent))
                
                topic_analysis = analyze_topic_completeness(
                    text=full_text,
                    expected_topics=expected_entities,
                    main_keyword=main_keyword
                )
                
                advanced_semantic = {
                    "source_effort": {
                        "score": source_effort.get("score", 0),
                        "status": source_effort.get("status", "UNKNOWN"),
                        "signals_found": source_effort.get("signals_found", [])[:5],
                        "recommendations": source_effort.get("recommendations", [])[:3]
                    },
                    "topic_completeness": {
                        "score": topic_analysis.get("score", 0),
                        "status": topic_analysis.get("status", "UNKNOWN"),
                        "missing_topics": topic_analysis.get("missing_topics", [])[:5]
                    }
                }
                
                # Dodaj do score
                if source_effort.get("score", 50) < 40:
                    score -= 5
                    recommendations.append("🔵 Dodaj źródła (badania, ekspertów, orzecznictwo)")
                
                if topic_analysis.get("score", 50) < 60:
                    score -= 5
                    for topic in topic_analysis.get("missing_topics", [])[:2]:
                        recommendations.append(f"🔵 Rozwiń temat: {topic}")
                
                score = max(0, min(100, score))
                print(f"[FINAL_REVIEW] ✅ Advanced semantic done: source={source_effort.get('score', 0)}, completeness={topic_analysis.get('score', 0)}")
                
            except Exception as adv_error:
                print(f"[FINAL_REVIEW] ⚠️ Advanced semantic error: {adv_error}")
                advanced_semantic = {"error": str(adv_error)}
        
        # 🆕 v40.2: ENTITY SCORING (Semantic SEO 2025)
        # ================================================================
        entity_scoring_result = None
        if ENTITY_SCORING_ENABLED:
            try:
                # Pobierz encje i relacje z S1
                s1_entities_raw = s1_data.get("entity_seo", {}).get("entities", [])
                # v49: Normalize entity dicts to strings for scoring
                s1_entities = []
                for ent in s1_entities_raw:
                    if isinstance(ent, str):
                        s1_entities.append(ent)
                    elif isinstance(ent, dict):
                        name = ent.get("entity") or ent.get("name") or ent.get("text") or ""
                        if name:
                            s1_entities.append(name)
                s1_relationships = s1_data.get("relationships", [])
                
                # Kompleksowy scoring encji
                entity_score = calculate_entity_score(
                    text=full_text,
                    s1_entities=s1_entities,
                    main_keyword=main_keyword,
                    s1_relationships=s1_relationships
                )
                
                entity_scoring_result = {
                    "score": entity_score.get("score", 0),
                    "grade": entity_score.get("grade", "N/A"),
                    "status": entity_score.get("status", "UNKNOWN"),
                    "components": {
                        "coverage": entity_score.get("components", {}).get("coverage", {}).get("score", 0),
                        "density": entity_score.get("components", {}).get("density", {}).get("score", 0),
                        "relationships": entity_score.get("components", {}).get("relationships", {}).get("score", 0),
                        "salience": entity_score.get("components", {}).get("salience", {}).get("score", 0)
                    },
                    "summary": entity_score.get("summary", {}),
                    "recommendations": entity_score.get("recommendations", [])[:3]
                }
                
                # Impact na score
                if entity_score.get("score", 50) < 40:
                    score -= 5
                    for rec in entity_score.get("recommendations", [])[:1]:
                        recommendations.append(f"🟣 {rec}")
                
                score = max(0, min(100, score))
                print(f"[FINAL_REVIEW] ✅ Entity scoring done: {entity_score.get('score', 0)}/100 (grade: {entity_score.get('grade', 'N/A')})")
                
            except Exception as ent_error:
                print(f"[FINAL_REVIEW] ⚠️ Entity scoring error: {ent_error}")
                entity_scoring_result = {"error": str(ent_error)}
        
        # ================================================================
        # Result
        # ================================================================
        result = {
            "status": status,
            "project_id": project_id,
            "word_count": word_count,
            "score": score,
            "version": "v40.2",
            
            # 🆕 v40.1: Keywords validation z tolerancją
            "keywords_validation": keywords_validation,
            
            # 🆕 v40.1: Advanced semantic analysis
            "advanced_semantic": advanced_semantic,
            
            # 🆕 v40.2: Entity scoring (Semantic SEO)
            "entity_scoring": entity_scoring_result,
            
            "validations": {
                "missing_keywords": missing_kw,
                "main_vs_synonyms": main_syn,
                "h2_keywords": h2_val,
                "h3_length": h3_val,
                "list_count": list_val,
                "ngram_coverage": ngram_val
            },
            "all_issues": all_issues,
            "issues_summary": {
                "errors": errors,
                "warnings": warnings,
                "stuffing_critical": len(missing_kw.get("stuffing", [])),
                "stuffing_warning": len(missing_kw.get("within_tolerance", []))
            },
            "recommendations": [r for r in recommendations if r],
            "tolerance_info": {
                "percent": TOLERANCE_PERCENT,
                "explanation": f"Przekroczenie target_max o max {TOLERANCE_PERCENT}% = WARNING, powyżej = ERROR"
            }
        }
        
        # ================================================================
        # Save to Firestore
        # ================================================================
        try:
            doc_ref = db.collection("seo_projects").document(project_id)
            doc_ref.update({
                "final_review": result,
                "final_review_timestamp": firestore.SERVER_TIMESTAMP
            })
            print(f"[FINAL_REVIEW] ✅ Saved to Firestore")
        except Exception as save_error:
            print(f"[FINAL_REVIEW] ⚠️ Could not save to Firestore: {save_error}")
            result["firestore_save_error"] = str(save_error)
        
        print(f"[FINAL_REVIEW] ✅ Review complete. Status: {status}, Score: {score}")
        return jsonify(result), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[FINAL_REVIEW] ❌ Critical error: {e}")
        print(f"[FINAL_REVIEW] Traceback: {error_trace}")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "project_id": project_id,
            "traceback": error_trace
        }), 500


# ================================================================
# ENDPOINT: applyFinalCorrections
# ================================================================
@final_review_routes.post("/api/project/<project_id>/apply_corrections")
def apply_final_corrections(project_id):
    """Automatycznie aplikuje poprawki używając Gemini."""
    print(f"[FINAL_REVIEW] 🔧 Applying corrections for: {project_id}")
    
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        batches = data.get("batches", [])
        final_review = data.get("final_review", {})
        main_keyword = data.get("main_keyword", data.get("topic", ""))
        
        if not final_review:
            return jsonify({"error": "Run final_review first"}), 400
        
        full_text = "\n\n".join([b.get("text", "") for b in batches if b.get("text")])
        
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini not configured"}), 500
        
        recommendations = final_review.get("recommendations", [])
        if not recommendations:
            return jsonify({"status": "NO_CORRECTIONS_NEEDED"}), 200
        
        corrections = [r for r in recommendations[:10] if r]
        
        keywords_to_add = []
        missing_kw = final_review.get("validations", {}).get("missing_keywords", {})
        for kw in missing_kw.get("priority_to_add", {}).get("to_add_by_claude", [])[:5]:
            if kw.get("keyword"):
                keywords_to_add.append({"keyword": kw["keyword"], "times": kw.get("target_min", 1)})
        
        # 🆕 v40.1: Keywords to remove (stuffing)
        keywords_to_remove = []
        for kw in missing_kw.get("stuffing", [])[:3]:
            if kw.get("keyword"):
                excess = kw["actual"] - kw["target_max"]
                keywords_to_remove.append({"keyword": kw["keyword"], "remove_count": excess})
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        kw_section = ""
        if keywords_to_add:
            kw_list = "\n".join([f"  - '{k['keyword']}': {k['times']}x" for k in keywords_to_add])
            kw_section += f"FRAZY DO WPLECENIA:\n{kw_list}\n\n"
        
        if keywords_to_remove:
            remove_list = "\n".join([f"  - '{k['keyword']}': usuń {k['remove_count']}x" for k in keywords_to_remove])
            kw_section += f"FRAZY DO USUNIĘCIA (stuffing):\n{remove_list}\n\n"
        
        prompt = f"""Popraw artykuł:

{kw_section}INSTRUKCJE:
{chr(10).join(f"- {c}" for c in corrections)}

ZASADY:
1. Zachowaj h2:/h3: lub ## / ###
2. Frazy wplataj naturalnie
3. "{main_keyword}" częściej niż synonimy
4. Usuń nadmiarowe wystąpienia zaznaczonych fraz

ARTYKUŁ:
{full_text[:14000]}

Zwróć TYLKO poprawiony artykuł."""
        
        response = model.generate_content(prompt)
        corrected = response.text.strip()
        corrected = re.sub(r'^```(?:html|markdown)?\n?', '', corrected)
        corrected = re.sub(r'\n?```$', '', corrected)
        
        # Verify
        verification = {}
        for k in keywords_to_add[:5]:
            kw = k["keyword"]
            if UNIFIED_COUNTER:
                before = count_single_keyword(full_text, kw)
                after = count_single_keyword(corrected, kw)
            else:
                kw_lower = kw.lower()
                try:
                    before = len(re.findall(rf"\b{re.escape(kw_lower)}\b", full_text.lower()))
                    after = len(re.findall(rf"\b{re.escape(kw_lower)}\b", corrected.lower()))
                except:
                    before = full_text.lower().count(kw_lower)
                    after = corrected.lower().count(kw_lower)
            verification[kw] = {"before": before, "after": after, "added": after - before}
        
        # Verify removed
        for k in keywords_to_remove[:3]:
            kw = k["keyword"]
            if UNIFIED_COUNTER:
                before = count_single_keyword(full_text, kw)
                after = count_single_keyword(corrected, kw)
            else:
                kw_lower = kw.lower()
                try:
                    before = len(re.findall(rf"\b{re.escape(kw_lower)}\b", full_text.lower()))
                    after = len(re.findall(rf"\b{re.escape(kw_lower)}\b", corrected.lower()))
                except:
                    before = full_text.lower().count(kw_lower)
                    after = corrected.lower().count(kw_lower)
            verification[kw] = {"before": before, "after": after, "removed": before - after}
        
        doc_ref = db.collection("seo_projects").document(project_id)
        doc_ref.update({
            "corrected_article": corrected,
            "corrections_applied": corrections,
            "correction_timestamp": firestore.SERVER_TIMESTAMP
        })
        
        # Version tracking
        try:
            from version_manager import VersionManager, VersionSource, create_version_manager_for_project
            
            vm_data = data.get("version_manager")
            if vm_data:
                vm = VersionManager.from_dict(vm_data)
            else:
                vm = create_version_manager_for_project(project_id, batches)
            
            if not vm_data:
                vm.create_version(
                    batch_number=0,
                    text=full_text,
                    source=VersionSource.MANUAL,
                    metadata={"type": "original_before_corrections"}
                )
            
            new_version = vm.create_version(
                batch_number=0,
                text=corrected,
                source=VersionSource.FINAL_CORRECTIONS,
                metadata={
                    "corrections_count": len(corrections),
                    "keywords_added": list(verification.keys()),
                    "word_count_change": len(corrected.split()) - len(full_text.split())
                }
            )
            
            doc_ref.update({
                "version_manager": vm.to_dict()
            })
            
            print(f"[FINAL_REVIEW] 📚 Version saved: v{new_version.version_number}")
        except ImportError:
            print("[FINAL_REVIEW] ⚠️ version_manager not available")
        except Exception as e:
            print(f"[FINAL_REVIEW] ⚠️ Version tracking error: {e}")
        
        print(f"[FINAL_REVIEW] ✅ Corrections applied")
        
        return jsonify({
            "status": "CORRECTED",
            "corrections": corrections,
            "verification": verification,
            "word_count_before": len(full_text.split()),
            "word_count_after": len(corrected.split())
        }), 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[FINAL_REVIEW] ❌ Correction error: {e}")
        print(f"[FINAL_REVIEW] Traceback: {error_trace}")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }), 500


# ================================================================
# HEALTH CHECK
# ================================================================
@final_review_routes.get("/api/final_review/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "OK",
        "version": "40.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "tolerance_percent": TOLERANCE_PERCENT
    }), 200


# ================================================================
# ALIAS dla kompatybilności
# ================================================================
@final_review_routes.post("/api/project/<project_id>/apply_final_corrections")
def apply_final_corrections_alias(project_id):
    """Alias dla apply_corrections."""
    return apply_final_corrections(project_id)


# ================================================================
# VERSION MANAGEMENT ENDPOINTS
# ================================================================

@final_review_routes.get("/api/project/<project_id>/versions")
def get_version_history(project_id):
    """Pobiera historię wersji artykułu."""
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        vm_data = data.get("version_manager")
        
        if not vm_data:
            return jsonify({
                "status": "NO_VERSIONS",
                "message": "Brak zapisanych wersji dla tego projektu",
                "versions": []
            }), 200
        
        try:
            from version_manager import VersionManager
            vm = VersionManager.from_dict(vm_data)
            
            history = vm.get_batch_history(0)
            
            if history:
                versions_summary = []
                for v in history.versions:
                    versions_summary.append({
                        "version_id": v.version_id,
                        "version_number": v.version_number,
                        "source": v.source.value if hasattr(v.source, 'value') else str(v.source),
                        "created_at": v.created_at,
                        "word_count": v.word_count,
                        "is_current": v.is_current,
                        "metadata": v.metadata
                    })
                
                return jsonify({
                    "status": "OK",
                    "project_id": project_id,
                    "total_versions": len(versions_summary),
                    "current_version_id": history.current_version_id,
                    "versions": versions_summary
                }), 200
            else:
                return jsonify({
                    "status": "NO_VERSIONS",
                    "message": "Brak zapisanych wersji",
                    "versions": []
                }), 200
                
        except ImportError:
            return jsonify({"error": "version_manager not available"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@final_review_routes.post("/api/project/<project_id>/rollback/<version_id>")
def rollback_to_version(project_id, version_id):
    """Przywraca artykuł do wybranej wersji."""
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        vm_data = data.get("version_manager")
        
        if not vm_data:
            return jsonify({"error": "Brak historii wersji"}), 400
        
        try:
            from version_manager import VersionManager, VersionSource
            vm = VersionManager.from_dict(vm_data)
            
            result = vm.rollback_to_version(batch_number=0, version_id=version_id)
            
            if result.get("status") != "OK":
                return jsonify(result), 400
            
            restored_text = result.get("restored_text", "")
            
            doc_ref = db.collection("seo_projects").document(project_id)
            doc_ref.update({
                "corrected_article": restored_text,
                "version_manager": vm.to_dict(),
                "rollback_timestamp": firestore.SERVER_TIMESTAMP
            })
            
            print(f"[FINAL_REVIEW] 🔄 Rollback to version {version_id} successful")
            
            return jsonify({
                "status": "ROLLED_BACK",
                "project_id": project_id,
                "restored_version_id": version_id,
                "word_count": len(restored_text.split()),
                "message": f"Przywrócono wersję {result.get('restored_version_number', '?')}"
            }), 200
            
        except ImportError:
            return jsonify({"error": "version_manager not available"}), 500
            
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ Rollback error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================================================================
# MERGE BATCHES
# ================================================================
@final_review_routes.post("/api/project/<project_id>/merge_batches")
def merge_batches(project_id):
    """Scala wszystkie batche w jeden spójny artykuł."""
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        batches = data.get("batches", [])
        
        if not batches:
            return jsonify({"error": "No batches to merge"}), 400
        
        main_keyword = data.get("main_keyword", data.get("topic", ""))
        detected_category = data.get("detected_category", "general")
        
        merged_parts = []
        
        for i, batch in enumerate(batches):
            batch_text = batch.get("text", "")
            if not batch_text:
                continue
            
            clean_text = batch_text.strip()
            clean_text = re.sub(r'^h2:\s*(.+)$', r'## \1', clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r'^h3:\s*(.+)$', r'### \1', clean_text, flags=re.MULTILINE)
            
            merged_parts.append(clean_text)
        
        merged_article = "\n\n".join(merged_parts)
        
        if detected_category == "prawo":
            legal_disclaimer = data.get("legal_disclaimer", "")
            if not legal_disclaimer:
                legal_disclaimer = (
                    "\n\n---\n\n"
                    "**Zastrzeżenie prawne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
                    "i nie stanowi porady prawnej. W indywidualnych sprawach zalecamy konsultację "
                    "z wykwalifikowanym prawnikiem."
                )
            
            if "Zastrzeżenie prawne" not in merged_article and "zastrzeżenie" not in merged_article.lower():
                merged_article += legal_disclaimer
        
        # 🆕 v44.5: Medical disclaimer
        if detected_category == "medycyna":
            medical_disclaimer = data.get("medical_disclaimer", "")
            if not medical_disclaimer:
                medical_disclaimer = (
                    "\n\n---\n\n"
                    "**Zastrzeżenie medyczne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
                    "i edukacyjny. Nie stanowi porady medycznej ani nie zastępuje konsultacji "
                    "z lekarzem lub innym wykwalifikowanym specjalistą. W przypadku problemów "
                    "zdrowotnych skonsultuj się z lekarzem."
                )
            
            if "Zastrzeżenie medyczne" not in merged_article and "zastrzeżenie" not in merged_article.lower():
                merged_article += medical_disclaimer
        
        word_count = len(merged_article.split())
        h2_matches = re.findall(r'^##\s+.+$|^h2:\s*.+$|<h2[^>]*>.+</h2>', merged_article, re.MULTILINE | re.IGNORECASE)
        h2_count = len(h2_matches)
        
        doc_ref = db.collection("seo_projects").document(project_id)
        doc_ref.update({
            "corrected_article": merged_article,
            "merged_at": firestore.SERVER_TIMESTAMP,
            "merge_stats": {
                "batches_merged": len(batches),
                "word_count": word_count,
                "h2_count": h2_count,
                "has_disclaimer": detected_category in ("prawo", "medycyna"),
                "disclaimer_type": detected_category if detected_category in ("prawo", "medycyna") else None
            }
        })
        
        print(f"[FINAL_REVIEW] ✅ Merged {len(batches)} batches → {word_count} words, {h2_count} H2")
        
        return jsonify({
            "status": "MERGED",
            "project_id": project_id,
            "article": merged_article,
            "word_count": word_count,
            "h2_count": h2_count,
            "batches_merged": len(batches),
            "has_disclaimer": detected_category in ("prawo", "medycyna"),
            "disclaimer_type": detected_category if detected_category in ("prawo", "medycyna") else None,
            "message": f"Scalono {len(batches)} batchy w artykuł ({word_count} słów)"
        }), 200
        
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ Merge error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@final_review_routes.get("/api/project/<project_id>/merged_article")
def get_merged_article(project_id):
    """Pobiera scalony artykuł."""
    try:
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        data = doc.to_dict()
        corrected_article = data.get("corrected_article", "")
        merge_stats = data.get("merge_stats", {})
        
        if not corrected_article:
            return jsonify({
                "status": "NOT_MERGED",
                "message": "Artykuł nie został jeszcze scalony. Użyj POST /merge_batches"
            }), 200
        
        return jsonify({
            "status": "MERGED",
            "article": corrected_article,
            "word_count": len(corrected_article.split()),
            "stats": merge_stats
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
