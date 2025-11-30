import uuid
import re
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore

# Global spaCy
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)

# --- PARSER (Bez zmian) ---
def parse_brief_text_uuid(brief_text: str):
    lines = brief_text.split("\n")
    parsed_dict = {}
    for line in lines:
        line = line.strip()
        if not line: continue
        kw_type = "BASIC"
        upper_line = line.upper()
        if "[EXTENDED]" in upper_line:
            kw_type = "EXTENDED"
            line = re.sub(r"\[EXTENDED\]", "", line, flags=re.IGNORECASE).strip()
        elif "[BASIC]" in upper_line:
            kw_type = "BASIC"
            line = re.sub(r"\[BASIC\]", "", line, flags=re.IGNORECASE).strip()
        if ":" not in line: continue
        try:
            parts = line.rsplit(":", 1)
            original_keyword = parts[0].strip()
            counts_part = parts[1].strip().lower()
            numbers = re.findall(r"\d+", counts_part)
            if not numbers: continue
            if len(numbers) >= 2: min_val, max_val = int(numbers[0]), int(numbers[1])
            else: min_val, max_val = int(numbers[0]), int(numbers[0])
            doc = nlp(original_keyword)
            search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
            row_id = str(uuid.uuid4())
            parsed_dict[row_id] = {
                "keyword": original_keyword,
                "search_term_exact": original_keyword.lower(),
                "search_lemma": search_lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER",
                "type": kw_type
            }
        except Exception: continue
    return parsed_dict

# --- S2 CREATE (Bez zmian) ---
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()
    if not data or "topic" not in data or "brief_text" not in data:
        return jsonify({"error": "Required fields: topic, brief_text"}), 400
    topic = data["topic"]
    brief_text = data["brief_text"]
    firestore_keywords = parse_brief_text_uuid(brief_text)
    if not firestore_keywords: return jsonify({"error": "No keywords parsed."}), 400
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_data = {
        "topic": topic, "brief_raw": brief_text, "keywords_state": firestore_keywords,
        "counting_mode": "uuid_hybrid", "continuous_counting": True,
        "created_at": firestore.SERVER_TIMESTAMP, "batches": [], "total_batches": 0
    }
    doc_ref.set(project_data)
    return jsonify({"status": "CREATED", "project_id": doc_ref.id, "topic": topic, "keywords": len(firestore_keywords)}), 201

# --- S3 ADD BATCH (Bez zmian) ---
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data or "text" not in data: return jsonify({"error": "Field 'text' is required"}), 400
    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})
    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    status_code = result.get("status", 400)
    if not isinstance(status_code, int):
        status_code = 200 if "ACCEPTED" in str(result.get("status")) else 400
    result["batch_text"] = batch_text
    return jsonify(result), status_code

# ================================================================
# 🆕 S4 — EXPORT (Pobierz tekst bez usuwania)
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    
    # 1. Zszywanie tekstu
    batches = data.get("batches", [])
    full_text_parts = [b.get("text", "") for b in batches]
    full_article_text = "\n\n".join(full_text_parts)

    # 2. Statystyki SEO
    keywords_state = data.get("keywords_state", {})
    under = sum(1 for k in keywords_state.values() if k["status"] == "UNDER")
    over = sum(1 for k in keywords_state.values() if k["status"] == "OVER")
    ok = sum(1 for k in keywords_state.values() if k["status"] == "OK")
    locked = 1 if over >= 4 else 0

    # 3. Statystyki Jakości (Średnie)
    scores = [b.get("gemini_audit", {}).get("quality_score", 0) for b in batches if b.get("gemini_audit")]
    bursts = [b.get("language_audit", {}).get("burstiness", 0) for b in batches if b.get("language_audit")]
    fluffs = [b.get("language_audit", {}).get("fluff_ratio", 0) for b in batches if b.get("language_audit")]
    
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    avg_burst = round(sum(bursts) / len(bursts), 2) if bursts else 0
    avg_fluff = round(sum(fluffs) / len(fluffs), 3) if fluffs else 0

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "full_article_text": full_article_text,
        "final_stats": {"UNDER": under, "OVER": over, "LOCKED": locked, "OK": ok},
        "quality_metrics": {
            "avg_score": avg_score,
            "avg_burstiness": avg_burst,
            "avg_fluff": avg_fluff
        }
    }), 200

# ================================================================
# 🗑️ S4 — DELETE ONLY (Tylko usuwa)
# ================================================================
@project_routes.delete("/api/project/<project_id>")
def delete_project_final(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    if not doc_ref.get().exists: return jsonify({"error": "Not found"}), 404
    doc_ref.delete()
    return jsonify({"status": "DELETED", "message": "Project removed from Firestore."}), 200
