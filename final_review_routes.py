# ================================================================
# 🔍 FINAL REVIEW ROUTES v24.2 - UNIFIED KEYWORD COUNTING
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

final_review_routes = Blueprint("final_review_routes", __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[FINAL_REVIEW] ✅ Gemini API configured")
else:
    print("[FINAL_REVIEW] ⚠️ GEMINI_API_KEY not set")

GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# 1. MISSING KEYWORDS DETECTOR
# ================================================================
def detect_missing_keywords(text, keywords_state):
    """Wykrywa brakujące frazy BASIC/EXTENDED."""
    try:
        text_lower = text.lower()
        
        missing_basic = []
        missing_extended = []
        underused_basic = []
        underused_extended = []
        
        # v24.2: Unified counting
        if UNIFIED_COUNTER:
            # Używa hybrydowego liczenia z keyword_counter
            counts = count_keywords_for_state(text, keywords_state)
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            
            kw_type = meta.get("type", "BASIC").upper()
            target_min = meta.get("target_min", 1)
            
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
                "missing": max(0, target_min - actual)
            }
            
            if actual == 0:
                if kw_type in ["BASIC", "MAIN"]:
                    missing_basic.append(info)
                else:
                    missing_extended.append(info)
            elif actual < target_min:
                if kw_type in ["BASIC", "MAIN"]:
                    underused_basic.append(info)
                else:
                    underused_extended.append(info)
        
        missing_basic.sort(key=lambda x: x["missing"], reverse=True)
        
        return {
            "missing": {"basic": missing_basic, "extended": missing_extended},
            "underused": {"basic": underused_basic, "extended": underused_extended},
            "priority_to_add": {
                "critical": missing_basic[:5],
                "high": underused_basic[:5] + missing_extended[:5],
                "medium": underused_extended[:5]
            },
            "needs_correction": len(missing_basic) > 0 or len(underused_basic) > 0
        }
    except Exception as e:
        print(f"[FINAL_REVIEW] ❌ detect_missing_keywords error: {e}")
        return {
            "missing": {"basic": [], "extended": []},
            "underused": {"basic": [], "extended": []},
            "priority_to_add": {"critical": [], "high": [], "medium": []},
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
                    # v24.2: Unified counting
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
        
        h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)'
        h2_matches = re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE)
        h2_list = [(m[0] or m[1]).strip() for m in h2_matches if m[0] or m[1]]
        
        if not h2_list:
            return {"valid": True, "h2_count": 0, "coverage": 1.0, "issues": []}
        
        main_lower = main_keyword.lower()
        h2_with_main = sum(1 for h2 in h2_list if main_lower in h2.lower())
        coverage = h2_with_main / len(h2_list) if h2_list else 1.0
        
        issues = [{"h2": h2, "suggestion": f"Dodaj '{main_keyword}'"} 
                  for h2 in h2_list if main_lower not in h2.lower()]
        
        return {
            "valid": coverage >= 0.2,
            "h2_count": len(h2_list),
            "h2_with_main": h2_with_main,
            "coverage": round(coverage, 2),
            "issues": issues[:3]
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
        h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
        h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if not h3_matches:
            return {"valid": True, "issues": [], "total_h3": 0}
        
        issues = []
        
        for i, match in enumerate(h3_matches):
            h3_title = (match.group(1) or match.group(2) or "").strip()
            start = match.end()
            end = len(text)
            
            next_h = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
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
# MAIN ENDPOINT: performFinalReview
# ================================================================
@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    """Kompleksowy audyt końcowy artykułu."""
    print(f"[FINAL_REVIEW] 🔍 Starting review for project: {project_id}")
    
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
        
        # Wykonaj analizy (każda z własną obsługą błędów)
        print("[FINAL_REVIEW] 🔍 Running validations...")
        
        missing_kw = detect_missing_keywords(full_text, keywords_state)
        print(f"[FINAL_REVIEW] ✅ Missing keywords check done")
        
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
        
        # Zbierz issues
        all_issues = []
        
        if missing_kw.get("missing", {}).get("basic"):
            all_issues.append({
                "type": "MISSING_BASIC",
                "severity": "ERROR",
                "keywords": [k["keyword"] for k in missing_kw["missing"]["basic"][:5]]
            })
        
        if not main_syn.get("valid", True):
            all_issues.append({
                "type": "SYNONYM_OVERUSE",
                "severity": "ERROR",
                "ratio": main_syn.get("main_ratio", 1.0),
                "warning": main_syn.get("warning")
            })
        
        if not h2_val.get("valid", True):
            all_issues.append({
                "type": "H2_NO_KEYWORDS",
                "severity": "WARNING",
                "coverage": h2_val.get("coverage", 1.0)
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
        
        # Status
        errors = sum(1 for i in all_issues if i.get("severity") == "ERROR")
        warnings = sum(1 for i in all_issues if i.get("severity") == "WARNING")
        
        status = "WYMAGA_POPRAWEK" if errors > 0 else ("WARN" if warnings > 2 else "OK")
        
        # Recommendations
        recommendations = []
        
        for ov in main_syn.get("overused_synonyms", [])[:2]:
            recommendations.append(ov.get("action", ""))
        
        for kw in missing_kw.get("priority_to_add", {}).get("critical", [])[:3]:
            recommendations.append(f"DODAJ '{kw.get('keyword', '')}' min. {kw.get('target_min', 1)}x")
        
        for issue in h3_val.get("issues", [])[:2]:
            recommendations.append(f"Rozbuduj H3 '{issue.get('h3', '')}' o {issue.get('deficit', 0)} słów")
        
        if list_val.get("action"):
            recommendations.append(list_val["action"])
        
        if ngram_val.get("missing"):
            recommendations.append(f"Wpleć: {', '.join(ngram_val.get('missing', [])[:3])}")
        
        # Score
        score = 100
        score -= len(missing_kw.get("missing", {}).get("basic", [])) * 5
        score -= len(missing_kw.get("underused", {}).get("basic", [])) * 2
        if main_syn.get("main_ratio", 1.0) < 0.3:
            score -= 15
        if h2_val.get("coverage", 1.0) < 0.2:
            score -= 10
        score -= len(h3_val.get("issues", [])) * 3
        if list_val.get("count", 0) > 1:
            score -= (list_val["count"] - 1) * 5
        if ngram_val.get("coverage", 1.0) < 0.6:
            score -= 10
        score = max(0, min(100, score))
        
        result = {
            "status": status,
            "project_id": project_id,
            "word_count": word_count,
            "score": score,
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
                "warnings": warnings
            },
            "recommendations": [r for r in recommendations if r]  # Filter empty
        }
        
        # Save to Firestore
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
        
        # Build corrections
        corrections = [r for r in recommendations[:10] if r]
        
        keywords_to_add = []
        missing_kw = final_review.get("validations", {}).get("missing_keywords", {})
        for kw in missing_kw.get("priority_to_add", {}).get("critical", [])[:5]:
            if kw.get("keyword"):
                keywords_to_add.append({"keyword": kw["keyword"], "times": kw.get("target_min", 1)})
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        kw_section = ""
        if keywords_to_add:
            kw_list = "\n".join([f"  - '{k['keyword']}': {k['times']}x" for k in keywords_to_add])
            kw_section = f"FRAZY DO WPLECENIA:\n{kw_list}\n\n"
        
        prompt = f"""Popraw artykuł:

{kw_section}INSTRUKCJE:
{chr(10).join(f"- {c}" for c in corrections)}

ZASADY:
1. Zachowaj h2:/h3:
2. Frazy wplataj naturalnie
3. "{main_keyword}" częściej niż synonimy

ARTYKUŁ:
{full_text[:14000]}

Zwróć TYLKO poprawiony artykuł."""
        
        response = model.generate_content(prompt)
        corrected = response.text.strip()
        corrected = re.sub(r'^```(?:html|markdown)?\n?', '', corrected)
        corrected = re.sub(r'\n?```$', '', corrected)
        
        # Verify - v24.2: Unified counting
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
        
        doc_ref = db.collection("seo_projects").document(project_id)
        doc_ref.update({
            "corrected_article": corrected,
            "corrections_applied": corrections,
            "correction_timestamp": firestore.SERVER_TIMESTAMP
        })
        
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
        "version": "22.5",
        "gemini_configured": bool(GEMINI_API_KEY)
    }), 200


# ================================================================
# ALIAS dla kompatybilności
# ================================================================
@final_review_routes.post("/api/project/<project_id>/apply_final_corrections")
def apply_final_corrections_alias(project_id):
    """Alias dla apply_corrections."""
    return apply_final_corrections(project_id)
