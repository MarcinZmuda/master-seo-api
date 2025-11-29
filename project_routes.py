import uuid
import re
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore

# Importujemy logikę batcha, żeby endpoint add_batch działał
from firestore_tracker_routes import process_batch_in_firestore

# Global spaCy model (raz w całej aplikacji)
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)


# ================================================================
# 🔧 Parser UUID Hybrid v3.2 — obsługa tagów [BASIC]/[EXTENDED]
# ================================================================
def parse_brief_text_uuid(brief_text: str):
    """
    Parsuje brief linia po linii i buduje strukturę Firestore.
    Obsługuje formaty:
    - "[BASIC] fraza: 1-5x"
    - "[EXTENDED] fraza długiego ogona: 1x"
    - "zwykła fraza: 1x" (domyślnie BASIC)
    """
    lines = brief_text.split("\n")
    parsed_dict = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Wykrywanie typu frazy (BASIC / EXTENDED)
        kw_type = "BASIC" # Domyślnie
        
        upper_line = line.upper()
        if "[EXTENDED]" in upper_line:
            kw_type = "EXTENDED"
            # Usuwamy tag z tekstu, żeby nie psuł parsowania
            line = re.sub(r"\[EXTENDED\]", "", line, flags=re.IGNORECASE).strip()
        elif "[BASIC]" in upper_line:
            kw_type = "BASIC"
            line = re.sub(r"\[BASIC\]", "", line, flags=re.IGNORECASE).strip()

        if ":" not in line:
            continue

        try:
            # Bierzemy część po ostatnim dwukropku (liczby)
            parts = line.rsplit(":", 1)
            original_keyword = parts[0].strip()
            counts_part = parts[1].strip().lower()

            # Wszystkie liczby z prawej strony
            numbers = re.findall(r"\d+", counts_part)

            if not numbers:
                continue

            # Zakres lub pojedyncza liczba
            if len(numbers) >= 2:
                min_val = int(numbers[0])
                max_val = int(numbers[1])
            else:
                min_val = int(numbers[0])
                max_val = int(numbers[0])

            # Lematy frazy (dla search_lemma)
            doc = nlp(original_keyword)
            search_lemma = " ".join(
                t.lemma_.lower() for t in doc if t.is_alpha
            )

            row_id = str(uuid.uuid4())

            parsed_dict[row_id] = {
                "keyword": original_keyword,
                "search_term_exact": original_keyword.lower(),
                "search_lemma": search_lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER",
                "type": kw_type  # <--- Zapisujemy typ rozpoznany przez parser
            }

        except Exception as e:
            print(f"⚠️ Błąd parsowania linii '{line}': {e}")
            continue

    return parsed_dict


# ================================================================
# 🟦 S2 — Tworzenie projektu
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()

    if not data or "topic" not in data or "brief_text" not in data:
        return jsonify({"error": "Required fields: topic, brief_text"}), 400

    topic = data["topic"]
    brief_text = data["brief_text"]

    # Używamy nowego parsera v3.2
    firestore_keywords = parse_brief_text_uuid(brief_text)
    keyword_count = len(firestore_keywords)

    if keyword_count == 0:
        return jsonify({
            "error": "No keywords parsed. Use format '[BASIC] keyword: 1-5x'."
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
        "keywords": keyword_count,
        "info": "Projekt utworzony. Obsługa tagów [BASIC]/[EXTENDED] aktywna."
    }), 201


# ================================================================
# 🟩 S3 — Dodawanie batcha (wykorzystuje logikę trackera)
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})

    # Wywołujemy logikę z firestore_tracker_routes.py
    result = process_batch_in_firestore(project_id, batch_text, meta_trace)

    # Ustalanie kodu HTTP na podstawie statusu biznesowego
    status_code = result.get("status", 400)
    if not isinstance(status_code, int):
        status_code = 200 if "ACCEPTED" in str(result.get("status")) else 400

    result["batch_text"] = batch_text
    return jsonify(result), status_code


# ================================================================
# 🟥 S4 — Usunięcie projektu (z podsumowaniem)
# ================================================================
@project_routes.delete("/api/project/<project_id>")
def delete_project_final(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

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

    return jsonify({
        "status": "DELETED",
        "summary": summary
    }), 200
