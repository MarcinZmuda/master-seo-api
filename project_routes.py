import uuid  # <--- DODANE
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import spacy
from firestore_tracker_routes import process_batch_in_firestore

nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)

# -------------------------------------------------------------
# 🔧 Parser UUID Hybrid (Każdy wiersz to osobny ID)
# -------------------------------------------------------------
def parse_brief_text_uuid(brief_text: str):
    """
    Parsuje brief linia po linii.
    Każda linia dostaje UNIKALNE ID (UUID).
    Dzięki temu nawet identyczne frazy są traktowane jako osobne liczniki.
    """
    lines = brief_text.split("\n")
    # Zmieniamy strukturę: to będzie lista obiektów, nie słownik po kluczu frazy
    # Ale Firestore wymaga mapy lub kolekcji. Zrobimy mapę po UUID.
    parsed_dict = {}
    
    range_pattern = re.compile(r"(.+?):\s*(\d+)[–-](\d+)x")

    for line in lines:
        line = line.strip()
        if not line: continue

        match = range_pattern.match(line)
        if match:
            original_keyword = match.group(1).strip()
            min_val = int(match.group(2))
            max_val = int(match.group(3))

            doc = nlp(original_keyword)
            search_lemma = " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

            # GENERUJEMY UNIKALNE ID DLA TEGO WIERSZA
            row_id = str(uuid.uuid4())

            parsed_dict[row_id] = {
                "keyword": original_keyword,    # Wyświetlana nazwa
                "search_term_exact": original_keyword.lower(),
                "search_lemma": search_lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER"
            }

    return parsed_dict


# -------------------------------------------------------------
# S2 — Tworzenie projektu
# -------------------------------------------------------------
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()

    if not data or "topic" not in data or "brief_text" not in data:
        return jsonify({"error": "Required fields: topic, brief_text"}), 400

    topic = data["topic"]
    brief_text = data["brief_text"]

    # 🔥 Używamy parsera UUID
    firestore_keywords = parse_brief_text_uuid(brief_text)

    if not firestore_keywords:
        return jsonify({"error": "Could not parse any keywords"}), 400

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    project_data = {
        "topic": topic,
        "brief_raw": brief_text,
        "keywords_state": firestore_keywords,
        "counting_mode": "uuid_hybrid", # Nowy tryb
        "continuous_counting": True,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "total_batches": 0
    }

    doc_ref.set(project_data)

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "counting_mode": "uuid_hybrid",
        "keywords": len(firestore_keywords), # Pokaże dokładną liczbę linii
        "info": "Tryb UUID Hybrid: Każdy wiersz to unikalny rekord."
    }), 201

# (Reszta endpointów DELETE i ADD BATCH bez zmian logicznych - tylko copy-paste standardowe)
@project_routes.delete("/api/project/<project_id>")
def delete_project_final(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})

    under = sum(1 for k in keywords_state.values() if k["status"] == "UNDER")
    over = sum(1 for k in keywords_state.values() if k["status"] == "OVER")
    ok = sum(1 for k in keywords_state.values() if k["status"] == "OK")
    locked = 1 if over >= 4 else 0

    summary = {
        "topic": data.get("topic"),
        "total_batches": data.get("total_batches", 0),
        "under_terms_count": under,
        "over_terms_count": over,
        "locked_terms_count": locked,
        "ok_terms_count": ok,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    doc_ref.delete()
    return jsonify({"status": "DELETED", "summary": summary}), 200

@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})
    result = process_batch_in_firestore(project_id, batch_text, meta_trace)

    status_code = result.get("status", 400)
    if not isinstance(status_code, int): status_code = 200 if "ACCEPTED" in str(result.get("status")) else 400

    result["batch_text"] = batch_text
    return jsonify(result), status_code
