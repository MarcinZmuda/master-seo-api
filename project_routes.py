# ================================================================
# project_routes.py — Project Management Layer (v7.2.9-firestore-primary-remote)
# ================================================================

import os
import re
import json
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
    """Zwraca listę lematów dla frazy (np. 'adwokat rozwodowy' -> ['adwokat', 'rozwodowy'])."""
    if not NLP:
        return phrase.lower().split()
    doc = NLP(phrase.lower())
    return [token.lemma_ for token in doc if token.is_alpha]


# ---------------------------------------------------------------
# 🧾 Parser briefu SEO (BASIC / EXTENDED)
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """Parsuje brief SEO (sekcje BASIC TEXT TERMS i EXTENDED TEXT TERMS)."""
    lines = [line.strip() for line in brief_text.splitlines() if line.strip()]
    keywords_state = {}
    headers_list = []
    current_section = None

    pattern = re.compile(r"^(.*?)\s*:\s*(\d+)[–-](\d+)x?$")

    for line in lines:
        # wykrywanie sekcji
        if "BASIC TEXT TERMS" in line.upper():
            current_section = "basic"
            continue
        elif "EXTENDED TEXT TERMS" in line.upper():
            current_section = "extended"
            continue
        elif line.startswith("="):
            continue

        match = pattern.match(line)
        if match:
            keyword = match.group(1).strip()
            min_count = int(match.group(2))
            max_count = int(match.group(3))

            # EXTENDED → redukcja zakresu o 50%
            if current_section == "extended":
                min_count = max(1, round(min_count * 0.5))
                max_count = max(1, round(max_count * 0.5))

            keywords_state[keyword] = {
                "target_min": min_count,
                "target_max": max_count,
                "actual": 0,
                "status": "UNDER",
                "locked": False
            }
            headers_list.append(keyword)

    print(f"🧠 parse_brief_to_keywords → {len(keywords_state)} fraz sparsowanych.")
    return keywords_state, headers_list


# ---------------------------------------------------------------
# 🔧 Firestore + API pomocnicze
# ---------------------------------------------------------------
def call_s1_analysis(topic):
    """Wywołuje endpoint /api/s1_analysis dla tematu."""
    try:
        base_url = os.getenv("API_BASE_URL", "https://master-seo-api.onrender.com")
        url = f"{base_url}/api/s1_analysis"
        r = requests.post(url, json={"topic": topic}, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Błąd S1 Analysis: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------
# ✅ /api/project/create — tworzy nowy projekt Firestore
# ---------------------------------------------------------------
@project_bp.route("/api/project/create", methods=["POST"])
def create_project():
    """Tworzy nowy projekt Firestore z briefem SEO i strukturą lemmaMode."""
    try:
        global db
        if not db:
            return jsonify({"error": "Firestore nie jest połączony"}), 503
        if not NLP:
            return jsonify({"error": "Model spaCy nie jest załadowany"}), 500

        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "").strip()
        brief_text = data.get("brief_text", "")

        if not topic:
            return jsonify({"error": "Brak 'topic'"}), 400

        print(f"[DEBUG] Tworzenie projektu Firestore: {topic}")
        keywords_state, headers_list = parse_brief_to_keywords(brief_text)
        s1_data = call_s1_analysis(topic)

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
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# ✅ /api/project/<project_id>/add_batch — wysyła batch do Firestore Tracker
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    """Przekazuje batch do Firestore Tracker API (pełne liczenie lematyczne)."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Brak tekstu batcha"}), 400

        base_url = os.getenv("API_BASE_URL", "https://master-seo-api.onrender.com")
        tracker_url = f"{base_url}/api/project/{project_id}/add_batch"

        print(f"[INFO] 🔄 Deleguję batch do Firestore Tracker → {tracker_url}")

        try:
            r = requests.post(tracker_url, json={"text": text}, timeout=120)
        except requests.exceptions.ConnectionError:
            print("⚠️ Firestore Tracker niedostępny — zapisuję batch lokalnie (mirror mode).")
            os.makedirs("./offline_batches", exist_ok=True)
            file_path = f"./offline_batches/{project_id}_mirror.txt"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n[BATCH {datetime.utcnow().isoformat()}]\n{text}\n")
            return jsonify({
                "status": "⚠️ Firestore Tracker offline – batch zapisany lokalnie",
                "project_id": project_id,
                "mirror_file": file_path
            }), 503

        if r.status_code != 200:
            print(f"[WARN] Tracker zwrócił błąd: {r.status_code} – {r.text}")
            return jsonify({"error": f"Firestore Tracker error: {r.text}"}), r.status_code

        response_data = r.json()
        print(f"[OK] 🔢 Raport Firestore Tracker: {response_data.get('meta_prompt_summary', {})}")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Błąd delegacji batcha do Firestore Tracker: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    """Rejestruje blueprint project_routes."""
    global db
    db = _db
    app.register_blueprint(project_bp, url_prefix="/api")
    print("✅ [INIT] project_routes zarejestrowany pod prefixem /api (v7.2.9-firestore-primary-remote).")
