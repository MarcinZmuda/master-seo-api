"""
SEO Content Tracker Routes - v39 Brajen Semantic
Features:
- Robust NLP Counting (Exact + Lemma + Fuzzy Lemma)
- Firestore Persistent Update (Preview + Approve)
- Keyword Status: UNDER / OK / OVER / LOCKED
- Diagnostic Route (/api/debug/<project_id>)
- Burstiness Monitor
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
import spacy
from rapidfuzz import fuzz

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INIT SPACY (LG for Polish) ---
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[TRACKER] ✅ Załadowano model pl_core_news_lg")
except OSError:
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")


# ============================================================================
# 1. ROBUST COUNTING (EXACT + LEMMA + FUZZY)
# ============================================================================
def count_robust(doc, keyword_meta):
    """Zlicza max(exact, lemma_exact, lemma_fuzzy)."""
    if not doc:
        return 0
    
    kw_exact = keyword_meta.get("keyword", "").lower().strip()
    
    # Lemma base
    kw_lemma = keyword_meta.get("search_lemma", "")
    if not kw_lemma:
        kw_doc = nlp(kw_exact)
        kw_lemma = " ".join([t.lemma_.lower() for t in kw_doc if t.is_alpha])
    
    # 1️⃣ Exact Match
    raw_text = getattr(doc, "text", "").lower()
    clean_text = re.sub(r"<[^>]+>", " ", raw_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    exact_hits = len(re.findall(rf"\b{re.escape(kw_exact)}\b", clean_text))
    
    # 2️⃣ Lemma Exact Match
    doc_lemmas = " ".join([t.lemma_.lower() for t in doc if t.is_alpha])
    lemma_exact_hits = len(re.findall(rf"\b{re.escape(kw_lemma)}\b", doc_lemmas))
    
    # 3️⃣ Fuzzy Lemma Match
    lemma_tokens = kw_lemma.split()
    lemma_fuzzy_hits = 0
    tokens = doc_lemmas.split()
    for i in range(len(tokens) - len(lemma_tokens) + 1):
        window = " ".join(tokens[i:i + len(lemma_tokens)])
        if fuzz.token_set_ratio(window, kw_lemma) >= 90:
            lemma_fuzzy_hits += 1
    
    total_hits = max(exact_hits, lemma_exact_hits + lemma_fuzzy_hits)
    return total_hits


# ============================================================================
# 2. VALIDATIONS
# ============================================================================
def validate_structure(text, expected_h2_count=2):
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Użyto Markdown (##). Wymagany czysty HTML (<h2>)."}
    
    h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL)
    banned = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie", "wnioski", "konkluzja", "introduction"]
    for h2 in h2_matches:
        clean = h2.lower().strip()
        for ban in banned:
            if ban == clean or clean.startswith(f"{ban} ") or clean.endswith(f" {ban}"):
                return {"valid": False, "error": f"❌ ZAKAZANY NAGŁÓWEK: '{h2}'."}
    return {"valid": True}


def calculate_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    if mean == 0:
        return 0.0
    return round((math.sqrt(variance) / mean) * 10, 1)


def validate_quality_and_limits(text, keywords_state, current_doc):
    burst = calculate_burstiness(text)
    word_count = len(text.split())
    if word_count > 0:
        for row_id, meta in keywords_state.items():
            count = count_robust(current_doc, meta)
            density = (count / word_count) * 100
            if density > 4.5:
                return {"valid": False, "error": f"❌ Keyword Stuffing: '{meta['keyword']}' {density:.1f}%."}
    return {"valid": True}


# ============================================================================
# 3. GPT INSTRUCTION GENERATOR
# ============================================================================
def generate_gpt_instruction(current_batch, total_batches, keywords_state, batch_len_target):
    remaining_batches = max(1, total_batches - current_batch)
    instruction = f"""✅ BATCH {current_batch} ZATWIERDZONY.
⏩ INSTRUKCJA NA BATCH {current_batch + 1}:

1. CEL: ok. {batch_len_target} słów.
2. STRUKTURA: LONG (250w+) / SHORT (150w).
3. SŁOWA KLUCZOWE (Priorytety):
"""
    targets = []
    for meta in keywords_state.values():
        if meta.get("type") == "BASIC":
            total_needed = meta.get("target_max", 0)
            current = meta.get("actual_uses", 0)
            remaining_uses = max(0, total_needed - current)
            if remaining_uses > 0:
                per_batch = math.ceil(remaining_uses / remaining_batches)
                targets.append(f"   - '{meta['keyword']}': ok. {per_batch}x")
    instruction += "\n".join(targets[:6])
    return instruction


# ============================================================================
# 4. PROCESS + FIRESTORE SAVE
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    doc_nlp = nlp(batch_text)
    
    for row_id, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        
        min_t, max_t = meta.get("target_min", 0), meta.get("target_max", 999)
        if meta["actual_uses"] < min_t:
            meta["status"] = "UNDER"
        elif min_t <= meta["actual_uses"] <= max_t:
            meta["status"] = "OK"
        elif meta["actual_uses"] > max_t:
            meta["status"] = "OVER"
        else:
            meta["status"] = "LOCKED"
        keywords_state[row_id] = meta
    
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": calculate_burstiness(batch_text)
    }
    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state

    try:
        doc_ref.set(project_data)
        print(f"[FIRESTORE] ✅ Zaktualizowano projekt {project_id}, fraz: {len(keywords_state)}")
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Błąd zapisu: {e}")
    
    return {"status": "BATCH_SAVED", "status_code": 200}


# ============================================================================
# ROUTES
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True) or {}
    text = data.get("batch_text", "")
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    doc_nlp = nlp(text)
    
    updated = {}
    for rid, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        current_total = meta.get("actual_uses", 0)
        updated[rid] = {
            **meta,
            "actual_uses": current_total + count,
            "used_in_current_batch": count
        }

    try:
        db.collection("seo_projects").document(project_id).update({"keywords_state": updated})
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Update error: {e}")
    
    return jsonify({
        "status": "PREVIEW",
        "corrected_text": text,
        "keywords_state": updated
    }), 200


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    data = request.get_json(force=True) or {}
    text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    current_batch = len(project_data.get("batches", [])) + 1
    plan = project_data.get("batches_plan", []) or [{"word_count": 500}] * 5
    doc_nlp = nlp(text)
    keywords_state = project_data.get("keywords_state", {})

    for rid, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        keywords_state[rid] = meta

    struct_check = validate_structure(text, meta_trace.get("h2_count", 2))
    if not struct_check["valid"]:
        return jsonify(struct_check), 400
    qual_check = validate_quality_and_limits(text, keywords_state, doc_nlp)
    if not qual_check["valid"]:
        return jsonify(qual_check), 400

    for rid, meta in keywords_state.items():
        if meta.get("type") == "BASIC" and meta["actual_uses"] > meta.get("target_max", 999):
            return jsonify({
                "error": f"❌ LIMIT: '{meta['keyword']}' użyto {meta['actual_uses']}x (limit {meta['target_max']})."
            }), 400

    batch_entry = {
        "batch_number": current_batch,
        "text": text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "validations": "OK"
    }
    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)

    is_complete = current_batch >= len(plan)
    next_len = plan[current_batch]["word_count"] if not is_complete else 0
    instruction = generate_gpt_instruction(current_batch, len(plan), keywords_state, next_len) if not is_complete else "ARTYKUŁ UKOŃCZONY."

    return jsonify({
        "status": "BATCH_SAVED",
        "batch_number": current_batch,
        "article_complete": is_complete,
        "gpt_instruction": instruction
    }), 200


# ============================================================================
# 5. VALIDATE + EXPORT
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    db = firestore.client()
    data = db.collection("seo_projects").document(project_id).get().to_dict()
    kw = data.get("keywords_state", {})
    ext_missing = [m['keyword'] for m in kw.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) == 0]
    if ext_missing:
        return jsonify({"error": f"❌ Brakuje fraz EXTENDED ({len(ext_missing)}).", "missing": ext_missing}), 400
    return jsonify({"status": "ARTICLE_READY"}), 200


@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    data = firestore.client().collection("seo_projects").document(project_id).get().to_dict()
    full_text = "\n\n".join([b.get("text", "") for b in data.get("batches", [])])
    return jsonify({"status": "EXPORTED", "full_article": full_text}), 200


# ============================================================================
# 6. DEBUG ROUTE — STATUS + STATS
# ============================================================================
@tracker_routes.get("/api/debug/<project_id>")
def debug_keywords(project_id):
    """Zwraca pełne dane słów kluczowych + burstiness + status."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    last_burst = batches[-1]["burstiness"] if batches else 0
    last_time = batches[-1]["timestamp"].isoformat() if batches else None
    
    result = []
    total_words = sum(len(b["text"].split()) for b in batches) or 1
    for rid, meta in keywords.items():
        density = round((meta.get("actual_uses", 0) / total_words) * 100, 2)
        result.append({
            "keyword": meta.get("keyword"),
            "type": meta.get("type"),
            "actual_uses": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min"),
            "target_max": meta.get("target_max"),
            "status": meta.get("status", "UNDER"),
            "density": f"{density}%",
        })
    
    return jsonify({
        "project_id": project_id,
        "keywords_count": len(result),
        "burstiness_last": last_burst,
        "last_update": last_time,
        "keywords": result
    }), 200
