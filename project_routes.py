# ================================================================
# project_routes.py — Project Management Layer
# v7.3.0-firestore-continuous-lemma (Firestore + Lemmatyczny Tracker)
# ================================================================

import os
import re
import requests
from flask import Blueprint, request, jsonify
from datetime import datetime
import spacy

# ---------------------------------------------------------------
# 🔧 Inicjalizacja
# ---------------------------------------------------------------
project_bp = Blueprint("project_routes", __name__)
db = None

# ---------------------------------------------------------------
# 🧠 Załaduj model spaCy (lemmatyzacja)
# ---------------------------------------------------------------
try:
    NLP = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie (lemmaMode=ON).")
except OSError:
    NLP = None
    print("❌ BŁĄD: Nie można załadować modelu spaCy 'pl_core_news_sm'.")


# ---------------------------------------------------------------
# 🧩 Lematyzacja fraz
# ---------------------------------------------------------------
def lemmatize_phrase(phrase):
    """Zwraca listę lematów dla frazy (do trackera Firestore)."""
    if not NLP:
        # awaryjnie: split po białych znakach + lower
        return phrase.lower().split()
    doc = NLP(phrase.lower())
    return [token.lemma_ for token in doc if token.is_alpha]


# ---------------------------------------------------------------
# 🧾 Parser briefu SEO (BASIC / EXTENDED)
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """
    Parsuje brief SEO w formacie:

    BASIC TEXT TERMS:
    fraza 1: 8-12x
    fraza 2: 3–7x

    EXTENDED TEXT TERMS:
    fraza 3: 2-4x
    ...

    Zwraca:
      - keywords_state: dict do zapisania w Firestore
      - headers_list: lista samych fraz (np. do sugestii nagłówków)
    """
    lines = [line.strip() for line in brief_text.splitlines() if line.strip()]
    keywords_state = {}
    headers_list = []
    current_section = None

    # Obsługa "–" i "-" oraz opcjonalnego "x" na końcu
    pattern = re.compile(r"^(.*?)\s*:\s*(\d+)[–-](\d+)x?$")

    for line in lines:
        upper = line.upper()
        if "BASIC TEXT TERMS" in upper:
            current_section = "basic"
            continue
        elif "EXTENDED TEXT TERMS" in upper:
            current_section = "extended"
            continue
        elif line.startswith("="):
            # linie typu "====" pomijamy
            continue

        match = pattern.match(line)
        if match:
            keyword = match.group(1).strip()
            min_count = int(match.group(2))
            max_count = int(match.group(3))

            # EXTENDED → zakres x0.5 (łagodniejsze limity)
            if current_section == "extended":
                min_count = max(1, round(min_count * 0.5))
                max_count = max(1, round(max_count * 0.5))

            keywords_state[keyword] = {
                "target_min": min_count,
                "target_max": max_count,
                "actual": 0,
                "status": "UNDER",
                "locked": False,
                "lemmas": lemmatize_phrase(keyword)
            }
            headers_list.append(keyword)

    print(f"🧠 parse_brief_to_keywords → {len(keywords_state)} fraz sparsowanych.")
    return keywords_state, headers_list


# ---------------------------------------------------------------
# ✅ /api/project/create — Tworzy projekt (bez S1 w backendzie)
# ---------------------------------------------------------------
@project_bp.route("/project/create", methods=["POST"])
def create_project():
    """
    Tworzy projekt Firestore z briefem SEO i strukturą lemmaMode.

    WAŻNE:
    - Ten endpoint NIE wywołuje już /api/s1_analysis.
    - Analiza S1 jest wykonywana osobno przez GPT (POST /api/s1_analysis),
      a wynik może być opcjonalnie przekazany w polu "s1_data" w request body.
    """
    try:
        global db
        if not db:
            return jsonify({"error": "Firestore nie jest połączony"}), 503
        if not NLP:
            return jsonify({"error": "Model spaCy nie jest załadowany"}), 500

        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "").strip()
        brief_text = data.get("brief_text", "")
        s1_data_from_client = data.get("s1_data")  # opcjonalne

        if not topic:
            return jsonify({"error": "Brak 'topic'"}), 400

        print(f"[DEBUG] Tworzenie projektu Firestore: {topic}")
        keywords_state, headers_list = parse_brief_to_keywords(brief_text)

        # Jeśli GPT kiedyś zacznie przekazywać wynik S1 w body → zapisz
        if s1_data_from_client is None:
            s1_data = {"status": "not_provided", "note": "S1 wykonywane po stronie GPT / osobny krok."}
        else:
            s1_data = s1_data_from_client

        doc_ref = db.collection("seo_projects").document()
        doc_ref.set({
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "brief_text": brief_text[:8000],
            "keywords_state": keywords_state,
            "headers_suggestions": headers_list,
            "s1_data": s1_data,
            "batches": [],
            "counting_mode": "firestore_remote_lemma",
            "continuous_counting": True,
            "status": "created"
        })

        print(f"✅ Projekt {doc_ref.id} utworzony ({len(keywords_state)} fraz).")

        return jsonify({
            "status": "✅ Projekt utworzony",
            "project_id": doc_ref.id,
            "topic": topic,
            "keywords": len(keywords_state),
            "counting_mode": "firestore_remote_lemma"
        }), 201

    except Exception as e:
        print(f"❌ Błąd /project/create: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# ❌ USUNIĘTO: /api/project/<project_id>/add_batch (delegacja HTTP)
# ---------------------------------------------------------------
# Ten endpoint wcześniej robił:
#   /api/project/<id>/add_batch  →  HTTP POST na http://127.0.0.1:10000/api/project/<id>/add_batch
# co powodowało nieskończoną pętlę i WORKER TIMEOUT.
#
# Teraz:
#   - endpoint /api/project/<project_id>/add_batch obsługuje wyłącznie
#     moduł firestore_tracker_routes.py (Tracker Lemmatyczny),
#     bez żadnych wewnętrznych requestów HTTP.
#
# Dzięki temu:
#   - brak pętli HTTP w tym samym serwerze,
#   - brak WORKER TIMEOUT przy /add_batch,
#   - pełna logika zliczania pozostaje w trackerze.


# ---------------------------------------------------------------
# ✅ /api/project/<project_id>/delete_final — Usuwa projekt (lokalny)
# ---------------------------------------------------------------
@project_bp.route("/project/<project_id>/delete_final", methods=["DELETE"])
def delete_project_final(project_id):
    """
    Usuwa projekt Firestore i zwraca końcowe statystyki.
    Uwaga: dla workflow GPT głównym endpointem "final summary" jest
    DELETE /api/project/<project_id> z firestore_batch_summary_routes.py
    (ten tylko fizycznie usuwa dokument).
    """
    try:
        global db
        if not db:
            return jsonify({"error": "Firestore nie jest połączony"}), 503

        doc_ref = db.collection("seo_projects").document(project_id)
        snapshot = doc_ref.get()

        if not snapshot.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        data = snapshot.to_dict()
        keywords_state = data.get("keywords_state", {})

        under = sum(1 for k in keywords_state.values() if k.get("status") == "UNDER")
        over = sum(1 for k in keywords_state.values() if k.get("status") == "OVER")
        locked = sum(1 for k in keywords_state.values() if k.get("locked"))
        ok = sum(1 for k in keywords_state.values() if k.get("status") == "OK")

        summary = {
            "topic": data.get("topic"),
            "counting_mode": data.get("counting_mode"),
            "continuous_counting": data.get("continuous_counting", True),
            "total_batches": len(data.get("batches", [])),
            "under_terms_count": under,
            "over_terms_count": over,
            "locked_terms_count": locked,
            "ok_terms_count": ok,
            "timestamp": datetime.utcnow().isoformat()
        }

        doc_ref.delete()
        print(f"🗑️ Projekt {project_id} usunięty z Firestore.")
        return jsonify({"status": "deleted", "summary": summary}), 200

    except Exception as e:
        print(f"❌ Błąd delete_final: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# ❤️ Health-check blueprinta
# ---------------------------------------------------------------
@project_bp.route("/project/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "module": "project_routes",
        "version": "v7.3.0-firestore-continuous-lemma"
    }), 200


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    """Rejestruje blueprint project_routes."""
    global db
    db = _db
    app.register_blueprint(project_bp, url_prefix="/api")
    print("✅ [INIT] project_routes zarejestrowany pod prefixem /api (v7.3.0-firestore-continuous-lemma).")
