"""
SEO Content Tracker Routes - v25.0 BRAJEN SEO Engine
+ NEW: Density ranges (0.5-1.5% optimal, 2% acceptable, 3% max)
+ NEW: Coverage-first logic (BASIC min 1x, EXTENDED exactly 1x)
+ Minimal approve_batch response (~500B instead of 220KB)
+ Morfeusz2 lemmatization (with spaCy fallback)
+ Burstiness validation (3.2-3.8)
+ Transition words validation (25-50%)
+ Per-batch keyword validation
+ Rozróżnienie EXCEEDED TOTAL vs per-batch warnings
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
from rapidfuzz import fuzz
from seo_optimizer import unified_prevalidation
from google.api_core.exceptions import InvalidArgument
import google.generativeai as genai
import os

# v24.0: Współdzielony model spaCy (oszczędność RAM)
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
except ImportError:
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")

# v23.8: Import polish_lemmatizer (Morfeusz2 + spaCy fallback)
try:
    from polish_lemmatizer import count_phrase_occurrences, get_backend_info, init_backend
    LEMMATIZER_ENABLED = True
    LEMMATIZER_BACKEND = init_backend()
    print(f"[TRACKER] Lemmatizer loaded: {LEMMATIZER_BACKEND}")
except ImportError as e:
    LEMMATIZER_ENABLED = False
    LEMMATIZER_BACKEND = "PREFIX"
    print(f"[TRACKER] Lemmatizer not available, using prefix matching: {e}")

# v24.0: Hierarchical keyword deduplication
try:
    from hierarchical_keyword_dedup import deduplicate_keyword_counts
    DEDUP_ENABLED = True
    print("[TRACKER] Hierarchical keyword dedup loaded")
except ImportError as e:
    DEDUP_ENABLED = False
    print(f"[TRACKER] Hierarchical dedup not available: {e}")

# v24.1: Semantic analyzer - wykrywa semantyczne pokrycie fraz
try:
    from semantic_analyzer import semantic_validation, find_semantic_gaps
    SEMANTIC_ENABLED = True
    print("[TRACKER] Semantic Analyzer loaded")
except ImportError as e:
    SEMANTIC_ENABLED = False
    print(f"[TRACKER] Semantic Analyzer not available: {e}")

tracker_routes = Blueprint("tracker_routes", __name__)

# ============================================================================
# v25.0: NEW DENSITY CONFIGURATION
# ============================================================================
DENSITY_OPTIMAL_MIN = 0.5   # Poniżej = za mało fraz
DENSITY_OPTIMAL_MAX = 1.5   # Optymalny zakres
DENSITY_ACCEPTABLE_MAX = 2.0  # Akceptowalne
DENSITY_WARNING_MAX = 2.5   # Ostrzeżenie
DENSITY_MAX = 3.0           # Hard limit - powyżej = keyword stuffing

def get_density_status(density: float) -> tuple:
    """
    v25.0: Zwraca status density z kolorowym oznaczeniem.
    Returns: (status_code, message)
    """
    if density < DENSITY_OPTIMAL_MIN:
        return "LOW", f"⚪ Za nisko ({density:.1f}%) - dodaj więcej fraz kluczowych"
    elif density <= DENSITY_OPTIMAL_MAX:
        return "OPTIMAL", f"✅ Optymalne ({density:.1f}%)"
    elif density <= DENSITY_ACCEPTABLE_MAX:
        return "ACCEPTABLE", f"🟢 OK ({density:.1f}%)"
    elif density <= DENSITY_WARNING_MAX:
        return "WARNING", f"🟡 Wysoko ({density:.1f}%) - uważaj"
    elif density <= DENSITY_MAX:
        return "HIGH", f"🟠 Za wysoko ({density:.1f}%) - ogranicz frazy"
    else:
        return "STUFFING", f"🔴 KEYWORD STUFFING ({density:.1f}%) - przepisz!"


# --- Gemini Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[TRACKER] ✅ Gemini API aktywny")
else:
    print("[TRACKER] ⚠️ Brak GEMINI_API_KEY")


# ============================================================================
# 1. COUNTING FUNCTIONS
# ============================================================================
def count_all_forms(text: str, keyword: str) -> int:
    """Liczy WSZYSTKIE odmiany słowa/frazy w tekście."""
    if not text or not keyword:
        return 0
    
    if LEMMATIZER_ENABLED:
        result = count_phrase_occurrences(text, keyword)
        return result.get("count", 0)
    
    # Fallback: prefix matching
    text_lower = text.lower()
    keyword_lower = keyword.lower().strip()
    words = keyword_lower.split()
    
    if len(words) == 1:
        word = words[0]
        stem = word[:6] if len(word) > 6 else word[:len(word)-1] if len(word) > 4 else word
        pattern = rf'\b{re.escape(stem)}\w*\b'
        return len(re.findall(pattern, text_lower))
    else:
        stems = []
        for word in words:
            if len(word) <= 3:
                stems.append(re.escape(word))
            elif len(word) <= 5:
                stems.append(re.escape(word[:len(word)-1]) + r'\w*')
            else:
                stems.append(re.escape(word[:5]) + r'\w*')
        pattern = r'\b' + r'\s+(?:\w+\s+){0,2}'.join(stems) + r'\b'
        return len(re.findall(pattern, text_lower))


# ============================================================================
# 2. VALIDATIONS
# ============================================================================
def validate_structure(text):
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Markdown (##) zabroniony — użyj h2:"}
    banned = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie"]
    for h2 in re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL):
        if any(b in h2.lower() for b in banned):
            return {"valid": False, "error": f"❌ Niedozwolony nagłówek: '{h2.strip()}'"}
    return {"valid": True}


def calculate_burstiness(text):
    """Target: 3.2-3.8"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    if not mean:
        return 0.0
    raw_score = math.sqrt(variance) / mean
    return round(raw_score * 5, 2)


TRANSITION_WORDS_PL = [
    "również", "także", "ponadto", "dodatkowo", "co więcej",
    "jednak", "jednakże", "natomiast", "ale", "z drugiej strony",
    "mimo to", "niemniej", "pomimo", "choć", "chociaż",
    "dlatego", "w związku z tym", "w rezultacie", "ponieważ",
    "zatem", "więc", "stąd", "w konsekwencji",
    "na przykład", "przykładowo", "między innymi", "np.",
    "po pierwsze", "po drugie", "następnie", "potem", "na koniec",
]


def calculate_transition_score(text: str) -> dict:
    """Target: 25-50% zdań z transition words"""
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences), "warnings": []}
    
    transition_count = sum(1 for s in sentences if any(tw in s.lower()[:100] for tw in TRANSITION_WORDS_PL))
    ratio = transition_count / len(sentences)
    
    warnings = []
    if ratio < 0.20:
        warnings.append(f"⚠️ Za mało transition words: {ratio:.0%} (min 25%)")
    elif ratio > 0.55:
        warnings.append(f"⚠️ Za dużo transition words: {ratio:.0%} (max 50%)")
    
    return {"ratio": round(ratio, 3), "count": transition_count, "total": len(sentences), "warnings": warnings}


def validate_metrics(burstiness: float, transition_data: dict, density: float) -> list:
    """
    v25.0: Waliduje metryki z nowymi zakresami density.
    """
    warnings = []
    
    if burstiness < 3.2:
        warnings.append(f"⚠️ Burstiness za niski: {burstiness} (min 3.2)")
    elif burstiness > 3.8:
        warnings.append(f"⚠️ Burstiness za wysoki: {burstiness} (max 3.8)")
    
    # v25.0: Nowa logika density z zakresami
    density_status, density_msg = get_density_status(density)
    
    if density_status == "STUFFING":
        warnings.append(f"🔴 KEYWORD STUFFING: {density:.1f}% (max {DENSITY_MAX}%)")
    elif density_status == "HIGH":
        warnings.append(f"🟠 Density za wysoka: {density:.1f}% (zalecane < {DENSITY_ACCEPTABLE_MAX}%)")
    elif density_status == "WARNING":
        warnings.append(f"🟡 Density wysoka: {density:.1f}% (optymalne: {DENSITY_OPTIMAL_MIN}-{DENSITY_OPTIMAL_MAX}%)")
    elif density_status == "LOW":
        warnings.append(f"⚪ Density niska: {density:.1f}% (min {DENSITY_OPTIMAL_MIN}%)")
    
    warnings.extend(transition_data.get("warnings", []))
    return warnings


# ============================================================================
# v25.0: COVERAGE VALIDATION
# ============================================================================
def validate_coverage(keywords_state: dict) -> dict:
    """
    v25.0: Sprawdza coverage dla BASIC i EXTENDED keywords.
    BASIC: każda min 1x (hard requirement), target z inputu
    EXTENDED: każda dokładnie 1x
    """
    basic_total = 0
    basic_covered = 0
    basic_missing = []
    basic_target_met = 0
    
    extended_total = 0
    extended_covered = 0
    extended_missing = []
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        keyword = meta.get("keyword", "")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        
        if kw_type in ["BASIC", "MAIN"]:
            basic_total += 1
            if actual >= 1:  # Hard requirement: min 1x
                basic_covered += 1
            else:
                basic_missing.append(keyword)
            
            if actual >= target_min:  # Target met
                basic_target_met += 1
                
        elif kw_type == "EXTENDED":
            extended_total += 1
            if actual >= 1:
                extended_covered += 1
            else:
                extended_missing.append(keyword)
    
    basic_coverage = (basic_covered / basic_total * 100) if basic_total > 0 else 100
    extended_coverage = (extended_covered / extended_total * 100) if extended_total > 0 else 100
    
    return {
        "basic": {
            "total": basic_total,
            "covered": basic_covered,
            "coverage_percent": round(basic_coverage, 1),
            "target_met": basic_target_met,
            "missing": basic_missing[:5],  # Max 5 dla czytelności
            "status": "OK" if basic_coverage == 100 else "INCOMPLETE"
        },
        "extended": {
            "total": extended_total,
            "covered": extended_covered,
            "coverage_percent": round(extended_coverage, 1),
            "missing": extended_missing[:5],
            "status": "OK" if extended_coverage == 100 else "INCOMPLETE"
        },
        "overall_coverage": round((basic_coverage + extended_coverage) / 2, 1) if extended_total > 0 else basic_coverage
    }


# ============================================================================
# 3. FIRESTORE PROCESSOR
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None, forced=False):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # v24.2: UNIFIED COUNTING - jedna funkcja dla całego systemu
    try:
        from keyword_counter import count_keywords_for_state, get_stuffing_warnings, count_keywords
        
        # Zbierz keywordy do analizy szczegółowej
        keywords = [meta.get("keyword", "").strip() for meta in keywords_state.values() if meta.get("keyword")]
        
        # Policz z longest-match-first (automatyczna deduplikacja zagnieżdżonych)
        batch_counts = count_keywords_for_state(batch_text, keywords_state)
        
        # Stuffing warnings (zintegrowane z tym samym licznikiem)
        stuffing_warnings = get_stuffing_warnings(batch_text, keywords_state)
        
        # Szczegóły do diagnostyki
        full_result = count_keywords(batch_text, keywords)
        in_headers = full_result.get("in_headers", {})
        in_intro = full_result.get("in_intro", {})
        
        UNIFIED_COUNTING = True
        
    except ImportError:
        # Fallback do starej metody
        UNIFIED_COUNTING = False
        batch_counts = {}
        stuffing_warnings = []
        in_headers = {}
        in_intro = {}
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if keyword:
                batch_counts[keyword] = count_all_forms(batch_text, keyword)

    # Update keywords_state with new counts
    exceeded_keywords = []
    per_batch_warnings = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        
        if not keyword:
            continue
        
        batch_count = batch_counts.get(keyword, 0)
        old_actual = meta.get("actual_uses", 0)
        new_actual = old_actual + batch_count
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        
        # v25.0: EXTENDED = dokładnie 1x, więcej to warning
        if kw_type == "EXTENDED" and new_actual > 1:
            per_batch_warnings.append(f"ℹ️ EXTENDED '{keyword}' użyta {new_actual}x (cel: 1x)")
        
        # Update state
        keywords_state[rid]["actual_uses"] = new_actual
        keywords_state[rid]["remaining_max"] = max(0, target_max - new_actual)
        
        # Status update
        if new_actual > target_max:
            keywords_state[rid]["status"] = "EXCEEDED"
            exceeded_keywords.append({
                "keyword": keyword,
                "actual": new_actual,
                "target_max": target_max,
                "would_be": new_actual,
                "excess": new_actual - target_max
            })
        elif new_actual >= target_min:
            keywords_state[rid]["status"] = "OK"
        else:
            keywords_state[rid]["status"] = "UNDER"
    
    # Calculate metrics
    burstiness = calculate_burstiness(batch_text)
    transition_data = calculate_transition_score(batch_text)
    
    # v25.0: Density z unified_prevalidation
    prevalidation = unified_prevalidation(batch_text, keywords_state)
    density = prevalidation.get("density", 0)
    semantic_score = prevalidation.get("semantic_score", 0)
    
    # v25.0: Coverage validation
    coverage = validate_coverage(keywords_state)
    
    # v24.1: Semantic gaps
    semantic_gaps = []
    if SEMANTIC_ENABLED:
        try:
            gaps_result = find_semantic_gaps(batch_text, keywords_state)
            semantic_gaps = gaps_result.get("gaps", [])
        except Exception as e:
            print(f"[TRACKER] Semantic gaps error: {e}")
    
    # Validate all metrics
    warnings = validate_metrics(burstiness, transition_data, density)
    warnings.extend(stuffing_warnings)
    
    # Structure validation
    struct_check = validate_structure(batch_text)
    if not struct_check["valid"]:
        warnings.append(struct_check["error"])
    valid_struct = struct_check["valid"]

    status = "APPROVED"
    if warnings or not valid_struct:
        status = "WARN"
    if forced:
        status = "FORCED"
    
    # v25.0: Sprawdź czy są krytyczne problemy
    has_critical = any("EXCEEDED TOTAL" in w for w in warnings)
    has_stuffing = any("STUFFING" in w for w in warnings)
    has_density_issue = density > DENSITY_MAX
    
    if not has_critical and not has_stuffing and not has_density_issue and not exceeded_keywords:
        status = "APPROVED"

    # Save batch
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness,
        "transition_ratio": transition_data.get("ratio", 0),
        "batch_counts": batch_counts,
        "per_batch_info": per_batch_warnings,
        "semantic_gaps": semantic_gaps,
        "coverage": coverage,  # v25.0: coverage info
        "language_audit": {
            "semantic_score": semantic_score,
            "density": density,
            "density_status": get_density_status(density)[0],  # v25.0
            "burstiness": burstiness
        },
        "warnings": warnings,
        "status": status
    }

    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    try:
        doc_ref.set(project_data)
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Błąd zapisu: {e}")

    return {
        "status": status,
        "semantic_score": semantic_score,
        "density": density,
        "density_status": get_density_status(density),  # v25.0: (status_code, message)
        "burstiness": burstiness,
        "warnings": warnings,
        "per_batch_warnings": per_batch_warnings,
        "semantic_gaps": semantic_gaps,
        "coverage": coverage,  # v25.0
        "exceeded_keywords": exceeded_keywords,
        "batch_counts": batch_counts,
        "unified_counting": UNIFIED_COUNTING if 'UNIFIED_COUNTING' in dir() else False,
        "in_headers": in_headers if 'in_headers' in dir() else {},
        "in_intro": in_intro if 'in_intro' in dir() else {},
        "status_code": 200
    }


# ============================================================================
# 4. MINIMAL RESPONSE (v25.0 - z coverage info)
# ============================================================================
def _minimal_batch_response(result: dict, project_data: dict = None) -> dict:
    """
    v25.0: Rozróżnia EXCEEDED TOTAL vs per-batch warnings + coverage info.
    """
    problems = []  # Krytyczne - wymagają reakcji
    info = []  # Informacyjne - można zignorować
    
    # EXCEEDED TOTAL - KRYTYCZNE
    exceeded = result.get("exceeded_keywords", [])
    for ex in exceeded:
        problems.append(f"❌ '{ex['keyword']}' PRZEKROCZYŁA CAŁKOWITY LIMIT ({ex['would_be']}/{ex['target_max']})")
    
    # v25.0: Density stuffing - KRYTYCZNE
    density_status = result.get("density_status", ("OK", ""))
    if density_status[0] == "STUFFING":
        problems.append(density_status[1])
    elif density_status[0] in ["HIGH", "WARNING"]:
        info.append(density_status[1])
    
    # Per-batch warnings - tylko INFO
    for w in result.get("per_batch_warnings", []):
        info.append(w)
    
    # Inne ważne warnings
    for w in result.get("warnings", []):
        if "EXCEEDED TOTAL" in str(w) or "STUFFING" in str(w):
            if w not in problems:
                problems.append(w)
        elif "density" in str(w).lower() and "🔴" in str(w):
            problems.append(w)
    
    # Status
    status = "OK"
    if problems:
        status = "WARN"
    if result.get("status") == "FORCED":
        status = "FORCED"
    if result.get("status") == "APPROVED":
        status = "OK"
    
    # Batch info
    batch_number = 1
    remaining_batches = 0
    if project_data:
        batches_done = len(project_data.get("batches", []))
        batches_planned = len(project_data.get("batches_plan", [])) or project_data.get("total_planned_batches", 4)
        batch_number = batches_done
        remaining_batches = max(0, batches_planned - batches_done)
    
    # v25.0: Coverage info
    coverage = result.get("coverage", {})
    coverage_summary = None
    if coverage:
        basic_cov = coverage.get("basic", {}).get("coverage_percent", 100)
        ext_cov = coverage.get("extended", {}).get("coverage_percent", 100)
        if basic_cov < 100 or ext_cov < 100:
            coverage_summary = f"📊 Coverage: BASIC {basic_cov:.0f}%, EXTENDED {ext_cov:.0f}%"
            if remaining_batches == 0 and (basic_cov < 100 or ext_cov < 100):
                problems.append(f"⚠️ Ostatni batch! Brakuje coverage: BASIC {100-basic_cov:.0f}%, EXTENDED {100-ext_cov:.0f}%")
    
    # Next action
    if exceeded:
        next_action = {
            "action": "ask_user",
            "question": f"Przekroczono CAŁKOWITY limit dla {len(exceeded)} fraz. A) Przepisać batch B) Kontynuować (forced)?"
        }
    elif remaining_batches > 0:
        next_action = {
            "action": "continue",
            "call": "GET /pre_batch_info → pisz kolejny batch"
        }
    else:
        next_action = {
            "action": "review",
            "call": "POST /editorial_review → oceń całość"
        }
    
    response = {
        "saved": True,
        "batch": batch_number,
        "status": status,
        "next": next_action,
        "remaining_batches": remaining_batches
    }
    
    # v25.0: Density status
    response["density"] = {
        "value": result.get("density", 0),
        "status": density_status[0],
        "message": density_status[1]
    }
    
    # v25.0: Coverage summary
    if coverage_summary:
        response["coverage"] = coverage_summary
    
    if problems:
        response["problems"] = problems
    if info:
        response["info"] = info
    
    return response


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    v25.0: MINIMALNA ODPOWIEDŹ z coverage i density status
    """
    data = request.get_json(force=True)
    text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

    result = process_batch_in_firestore(project_id, text, meta_trace, forced)

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    project_data = doc.to_dict() if doc.exists else None
    
    return jsonify(_minimal_batch_response(result, project_data)), 200


@tracker_routes.get("/api/debug/<project_id>")
def debug_keywords(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    
    # v25.0: Coverage info
    coverage = validate_coverage(keywords)
    
    stats = []
    for rid, meta in keywords.items():
        stats.append({
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": meta.get("actual_uses", 0),
            "target": f"{meta.get('target_min', 0)}-{meta.get('target_max', 999)}",
            "status": meta.get("status"),
            "remaining": meta.get("remaining_max", 0)
        })
    
    return jsonify({
        "project_id": project_id,
        "keywords": stats,
        "batches": len(batches),
        "coverage": coverage,  # v25.0
        "density_config": {  # v25.0: show current config
            "optimal": f"{DENSITY_OPTIMAL_MIN}-{DENSITY_OPTIMAL_MAX}%",
            "acceptable_max": f"{DENSITY_ACCEPTABLE_MAX}%",
            "hard_max": f"{DENSITY_MAX}%"
        }
    }), 200


@tracker_routes.delete("/api/project/<project_id>")
def delete_project(project_id):
    """Usuwa projekt z Firestore."""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    doc_ref.delete()
    return jsonify({"status": "DELETED", "project_id": project_id}), 200


@tracker_routes.post("/api/project/<project_id>/reset")
def reset_project(project_id):
    """Resetuje projekt - usuwa batche, zeruje keywords."""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    doc_ref.update({
        "batches": [],
        "final_review": None,
        "keywords_state": {
            rid: {**meta, "actual_uses": 0, "status": "UNDER", "remaining_max": meta.get("target_max", 999)}
            for rid, meta in project_data.get("keywords_state", {}).items()
        }
    })
    
    return jsonify({"status": "RESET", "project_id": project_id}), 200
