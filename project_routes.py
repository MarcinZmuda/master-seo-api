from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import spacy

# Ładowanie modelu spaCy raz w skali procesu
nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)


# -------------------------------------------------------------
# 🔧 Narzędzia pomocnicze
# -------------------------------------------------------------

def parse_brief_text(brief_text: str):
    """
    Parsuje brief BASIC/EXTENDED w formacie:
        fraza: 4–8x
    Zwraca listę słowników { keyword, min_val, max_val, lemma }
    """

    lines = brief_text.split("\n")
    parsed = []

    range_pattern = re.compile(r"(.+?):\s*(\d+)[–-](\d+)x")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = range_pattern.match(line)
        if match:
            keyword = match.group(1).strip()
            min_val = int(match.group(2))
            max_val = int(match.group(3))

            # Lematyzacja frazy
            doc = nlp(keyword)
            lemma = " ".join([token.lemma_ for token in doc])

            parsed.append({
                "keyword": keyword,
                "lemma": lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER"
            })

    return parsed


def convert_to_firestore_keywords(parsed_keywords):
    """
    Konwertuje listę fraz do struktury Firestore:
    {
        "fraza": {
            "lemma": "...",
            "target_min": ...,
            "target_max": ...,
            "actual_uses": 0,
            "status": "UNDER"
        }
    }
    """
    fs_dict = {}
    for item in parsed_keywords:
        fs_dict[item["keyword"]] = {
            "lemma": item["lemma"],
            "target_min": item["target_min"],
            "target_max": item["target_max"],
            "actual_uses": 0,
            "status": "UNDER"
        }
    return fs_dict


# -------------------------------------------------------------
# 🆕 S2 — Tworzenie projektu
# -------------------------------------------------------------
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()

    if not data or "topic" not in data or "brief_text" not in data:
        return jsonify({"error": "Required fields: topic, brief_text"}), 400

    topic = data["topic"]
    brief_text = data["brief_text"]

    # 🔍 Parsowanie fraz
    parsed_keywords = parse_brief_text(brief_text)
    firestore_keywords = convert_to_firestore_keywords(parsed_keywords)

    # 🔥 Tworzenie projektu w Firestore
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_id = doc_ref.id

    project_data = {
        "topic": topic,
        "brief_raw": brief_text,
        "keywords_state": firestore_keywords,
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "total_batches": 0
    }

    doc_ref.set(project_data)

    return jsonify({
        "status": "CREATED",
        "project_id": project_id,
        "topic": topic,
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "keywords": len(firestore_keywords),
        "headers": 0
    }), 201


# -------------------------------------------------------------
# 🧨 S4 — Usunięcie projektu i raport końcowy
# -------------------------------------------------------------
@project_routes.delete("/api/project/<project_id>")
def delete_project_final(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()

    # Zbudowanie raportu przed usunięciem
    keywords_state = data.get("keywords_state", {})

    under = sum(1 for k in keywords_state.values() if k["status"] == "UNDER")
    over = sum(1 for k in keywords_state.values() if k["status"] == "OVER")
    locked = sum(1 for k in keywords_state.values() if k["status"] == "LOCKED")
    ok = sum(1 for k in keywords_state.values() if k["status"] == "OK")

    summary = {
        "topic": data.get("topic"),
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "total_batches": data.get("total_batches", 0),
        "under_terms_count": under,
        "over_terms_count": over,
        "locked_terms_count": locked,
        "ok_terms_count": ok,
        "timestamp": firestore.SERVER_TIMESTAMP
    }

    # 🔥 Usunięcie projektu
    doc_ref.delete()

    return jsonify({
        "status": "DELETED",
        "summary": summary
    }), 200
