import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import spacy
# Importujemy logikę trackera
from firestore_tracker_routes import process_batch_in_firestore

nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)

# -------------------------------------------------------------
# 🔧 Parser UUID Hybrid (NAPRAWIONY REGEX DLA "1x")
# -------------------------------------------------------------
def parse_brief_text_uuid(brief_text: str):
    """
    Parsuje brief linia po linii.
    Obsługuje formaty: "fraza: 1-5x" ORAZ "fraza: 1x".
    Każda linia dostaje UNIKALNE ID (UUID).
    """
    lines = brief_text.split("\n")
    parsed_dict = {}
    
    for line in lines:
        line = line.strip()
        if not line: continue

        # Musi być dwukropek oddzielający frazę od liczb
        if ":" not in line:
            continue

        try:
            # Rozdzielamy po ostatnim dwukropku (na wypadek dwukropka w frazie)
            parts = line.rsplit(":", 1)
            original_keyword = parts[0].strip()
            counts_part = parts[1].strip().lower()

            # Wyciągamy WSZYSTKIE liczby z prawej strony
            numbers = re.findall(r'\d+', counts_part)

            if not numbers:
                continue # Pomijamy linie bez liczb

            if len(numbers) >= 2:
                min_val = int(numbers[0])
                max_val = int(numbers[1])
            else:
                # Jeśli jest tylko jedna liczba (np. "1x"), to min=max
                min_val = int(numbers[0])
                max_val = int(numbers[0])

            # Generujemy Lemat
            doc = nlp(original_keyword)
            search_lemma = " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

            # Generujemy unikalne ID dla wiersza
            row_id = str(uuid.uuid4())

            parsed_dict[row_id] = {
                "keyword": original_keyword, # To wyświetlimy w raporcie
                "search_term_exact": original_keyword.lower(),
                "search_lemma": search_lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER"
            }
        except Exception as e:
            print(f"⚠️ Błąd parsowania linii: '{line}' -> {e}")
            continue

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

    # 🔥 Używamy naprawionego parsera
    firestore_keywords = parse_brief_text_uuid(brief_text)

    keyword_count = len(firestore_keywords)

    if keyword_count == 0:
        return jsonify({
            "error": "No keywords parsed. Check format (e.g., 'phrase: 1-5x' or 'phrase: 1x')"
        }), 400

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    project_data = {
        "topic": topic,
        "brief_raw": brief_text,
        "keywords_state": firestore_keywords,
        "counting_mode": "uuid_hybrid",
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
        "keywords": keyword_count, # Powinno być tyle, ile linii w briefie
        "info": "Projekt utworzony. Parser obsługuje '1x' i '1-5x'."
    }), 201


# -------------------------------------------------------------
# S4 — Usunięcie projektu
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# S3 — Dodawanie batcha
# -------------------------------------------------------------
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
