"""
SEO Content Tracker Routes - v23.9 BRAJEN SEO Engine
+ Minimal approve_batch response (~500B instead of 220KB)
+ Morfeusz2 lemmatization (with spaCy fallback)
+ Burstiness validation (3.2-3.8)
+ Transition words validation (25-50%)
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

# v23.9: Współdzielony model spaCy (oszczędność RAM)
from shared_nlp import get_nlp
nlp = get_nlp()

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

# Semantic analyzer
try:
    from semantic_analyzer import semantic_validation, find_semantic_gaps
    SEMANTIC_ANALYZER_ENABLED = True
    print("[TRACKER] Semantic Analyzer loaded")
except ImportError as e:
    SEMANTIC_ANALYZER_ENABLED = False
    print(f"[TRACKER] Semantic Analyzer not available: {e}")

# Keyword limiter
try:
    from keyword_limiter import validate_keyword_limits, check_header_variation
    KEYWORD_LIMITER_ENABLED = True
    print("[TRACKER] Keyword Limiter loaded")
except ImportError as e:
    KEYWORD_LIMITER_ENABLED = False
    print(f"[TRACKER] Keyword Limiter not available: {e}")

tracker_routes = Blueprint("tracker_routes", __name__)

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
    """Waliduje metryki"""
    warnings = []
    
    if burstiness < 3.2:
        warnings.append(f"⚠️ Burstiness za niski: {burstiness} (min 3.2)")
    elif burstiness > 3.8:
        warnings.append(f"⚠️ Burstiness za wysoki: {burstiness} (max 3.8)")
    
    if density > 1.5:
        warnings.append(f"⚠️ Keyword density za wysoka: {density}% (max 1.5%)")
    
    warnings.extend(transition_data.get("warnings", []))
    return warnings


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
    
    # Prepare text
    clean_text_original = re.sub(r"<[^>]+>", " ", batch_text)
    clean_text_original = re.sub(r"\s+", " ", clean_text_original)
    
    # Count keywords in batch
    batch_counts = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").strip()
        if not keyword:
            batch_counts[rid] = 0
            continue
        batch_counts[rid] = count_all_forms(clean_text_original, keyword)
    
    # Check exceeded
    exceeded_keywords = []
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        if meta.get("type", "BASIC").upper() != "BASIC":
            continue
        current = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        new_total = current + batch_count
        if new_total > target_max:
            exceeded_keywords.append({
                "keyword": meta.get("keyword"),
                "current": current,
                "batch_uses": batch_count,
                "would_be": new_total,
                "target_max": target_max,
                "exceeded_by": new_total - target_max
            })
    
    # Update keywords state
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        meta["actual_uses"] = meta.get("actual_uses", 0) + batch_count
        
        min_t = meta.get("target_min", 0)
        max_t = meta.get("target_max", 999)
        actual = meta["actual_uses"]
        
        if actual < min_t:
            meta["status"] = "UNDER"
        elif actual == max_t:
            meta["status"] = "OPTIMAL"
        elif min_t <= actual < max_t:
            meta["status"] = "OK"
        else:
            meta["status"] = "OVER"
        
        if meta.get("type", "BASIC").upper() == "BASIC":
            meta["remaining_max"] = max(0, max_t - actual)
        
        keywords_state[rid] = meta

    # Prevalidation
    precheck = unified_prevalidation(batch_text, keywords_state)
    warnings = precheck.get("warnings", [])
    semantic_score = precheck.get("semantic_score", 1.0)
    density = precheck.get("density", 0.0)
    
    # Exceeded warnings
    for ek in exceeded_keywords:
        warnings.append(f"⚠️ EXCEEDED: '{ek['keyword']}' będzie {ek['would_be']}x (max {ek['target_max']}x)")
    
    # Metrics
    burstiness = calculate_burstiness(batch_text)
    transition_data = calculate_transition_score(batch_text)
    metrics_warnings = validate_metrics(burstiness, transition_data, density)
    warnings.extend(metrics_warnings)

    struct_check = validate_structure(batch_text)
    valid_struct = struct_check["valid"]

    status = "APPROVED"
    if warnings or not valid_struct:
        status = "WARN"
    if forced:
        status = "FORCED"

    # Save batch
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness,
        "transition_ratio": transition_data.get("ratio", 0),
        "language_audit": {
            "semantic_score": semantic_score,
            "density": density,
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
        "burstiness": burstiness,
        "warnings": warnings,
        "exceeded_keywords": exceeded_keywords,
        "batch_counts": batch_counts,
        "status_code": 200
    }


# ============================================================================
# 4. MINIMAL RESPONSE (v23.9 - ~500B instead of 220KB)
# ============================================================================
def _minimal_batch_response(result: dict, project_data: dict = None) -> dict:
    """
    Minimalna odpowiedź po zapisie batcha.
    GPT potrzebuje tylko: saved, status, problems, next
    Pełne dane są w pre_batch_info i editorial_review.
    """
    problems = []
    
    # Exceeded keywords - critical
    exceeded = result.get("exceeded_keywords", [])
    for ex in exceeded:
        problems.append(f"⚠️ '{ex['keyword']}' przekroczyła limit ({ex['would_be']}/{ex['target_max']})")
    
    # Important warnings
    for w in result.get("warnings", []):
        if "EXCEEDED" in str(w) or "density" in str(w).lower():
            if w not in problems:
                problems.append(w)
    
    # Status
    status = "OK"
    if problems:
        status = "WARN"
    if result.get("status") == "FORCED":
        status = "FORCED"
    
    # Batch info
    batch_number = 1
    remaining_batches = 0
    if project_data:
        batches_done = len(project_data.get("batches", []))
        batches_planned = len(project_data.get("batches_plan", [])) or project_data.get("total_planned_batches", 4)
        batch_number = batches_done
        remaining_batches = max(0, batches_planned - batches_done)
    
    # Next action
    if exceeded:
        next_action = {
            "action": "ask_user",
            "question": "Przekroczono limit fraz. A) Przepisać batch B) Kontynuować?"
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
    
    return {
        "saved": True,
        "batch": batch_number,
        "status": status,
        "problems": problems if problems else None,
        "next": next_action,
        "remaining_batches": remaining_batches
    }


# ============================================================================
# 5. ROUTES
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True)
    text = data.get("batch_text", "")
    result = process_batch_in_firestore(project_id, text, forced=False)
    result["mode"] = "PREVIEW_ONLY"
    return jsonify(result), 200


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    v23.9: MINIMALNA ODPOWIEDŹ (~500B zamiast 220KB)
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
        "batches": len(batches)
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
