# ================================================================
# project_routes.py — Warstwa Project Management (v7.2.2-full + Forced Regeneration & Emergency Exit)
# ================================================================

import os
import json
import re
from flask import Blueprint, request, jsonify
from datetime import datetime
import requests
from firebase_admin import firestore
import spacy
from statistics import mean
from collections import Counter

# ---------------------------------------------------------------
# 🔐 Inicjalizacja Firebase i spaCy
# ---------------------------------------------------------------
db = None
project_bp = Blueprint("project_routes", __name__)

try:
    NLP = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie.")
except OSError:
    NLP = None
    print("❌ BŁĄD: Nie można załadować modelu spaCy 'pl_core_news_sm'.")


# ---------------------------------------------------------------
# 🧩 parse_brief_to_keywords — parser briefu BASIC / EXTENDED
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """
    Parsuje brief SEO (sekcje BASIC / EXTENDED) i zwraca:
    - keywords_state: słownik fraz z zakresem i statusem
    - headers_list: lista fraz bazowa dla dalszych nagłówków (opcjonalna)
    """
    import re
    lines = [line.strip() for line in brief_text.splitlines() if line.strip()]
    keywords_state = {}
    headers_list = []

    pattern = re.compile(r"^(.*?)\s*:\s*(\d+)[–-](\d+)x?$")

    for line in lines:
        match = pattern.match(line)
        if match:
            keyword = match.group(1).strip()
            min_count = int(match.group(2))
            max_count = int(match.group(3))
            keywords_state[keyword] = {
                "target_min": min_count,
                "target_max": max_count,
                "actual": 0,
                "status": "UNDER",
                "locked": False,
                "lemmas": [keyword.lower()]  # uproszczony lemat
            }
            headers_list.append(keyword)

    print(f"🧠 parse_brief_to_keywords → {len(keywords_state)} fraz zbriefowanych.")
    return keywords_state, headers_list


# ---------------------------------------------------------------
# 🧠 Funkcje językowe
# ---------------------------------------------------------------
def lemmatize_text(text):
    if not NLP:
        return re.findall(r'\b\w+\b', text.lower())
    doc = NLP(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]


def get_root_prefix(word):
    vowels = 'aeiouyąęó'
    root_len = 6
    for i, ch in enumerate(word):
        if ch in vowels:
            root_len = i + 3
            break
    return word[:max(6, root_len)]


def extract_context_matches(text, root_prefix, related_terms=None):
    if not NLP:
        return 0
    doc = NLP(text.lower())
    related_terms = related_terms or []
    contextual_matches = 0

    for token in doc:
        if token.lemma_.startswith(root_prefix):
            context = " ".join([t.text for t in token.subtree])
            if any(term in context for term in related_terms):
                contextual_matches += 1
    return contextual_matches


# ---------------------------------------------------------------
# 🔧 Pomocnicze funkcje Firestore + API
# ---------------------------------------------------------------
def call_s1_analysis(topic):
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
        url = f"{base_url}/api/s1_analysis"
        r = requests.post(url, json={"topic": topic}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Błąd S1 Analysis: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------
# 🧩 Analiza batcha + liczenie fraz
# ---------------------------------------------------------------
def analyze_batch_text(project_id, text):
    """Analizuje batch tekstu: zlicza frazy, ustala statusy UNDER/OVER/LOCKED."""
    doc_ref = db.collection("seo_projects").document(project_id)
    project_data = doc_ref.get().to_dict()
    if not project_data:
        raise ValueError("Projekt nie istnieje")

    keywords_state = project_data.get("keywords_state", {})
    text_lower = text.lower()
    lemmatized_text = lemmatize_text(text_lower)
    lemma_counts = Counter(lemmatized_text)

    over_terms_count = 0
    locked_terms_count = 0
    under_terms_count = 0
    ok_terms_count = 0
    updated_keywords = 0
    keywords_report = []

    for keyword, meta in keywords_state.items():
        lemmas = meta.get("lemmas", [])
        actual_count = sum(lemma_counts.get(l, 0) for l in lemmas)
        keywords_state[keyword]["actual"] += actual_count
        updated_keywords += 1

        # status logic
        if keywords_state[keyword]["actual"] > meta["target_max"] + 10:
            status = "OVER"
            over_terms_count += 1
        elif keywords_state[keyword]["actual"] < meta["target_min"]:
            status = "UNDER"
            under_terms_count += 1
        elif keywords_state[keyword]["locked"]:
            status = "LOCKED"
            locked_terms_count += 1
        else:
            status = "OK"
            ok_terms_count += 1

        keywords_report.append({
            "keyword": keyword,
            "actual_uses": keywords_state[keyword]["actual"],
            "target_range": f"{meta['target_min']}–{meta['target_max']}x",
            "status": status,
            "priority_instruction": "Zablokuj nadmiarowe wystąpienia." if status == "OVER" else ""
        })

    meta_prompt_summary = (
        f"BATCH – UNDER: {under_terms_count}, OVER: {over_terms_count}, LOCKED: {locked_terms_count}, OK: {ok_terms_count}"
    )

    batch_data = {
        "text": text[:10000],
        "created_at": datetime.utcnow().isoformat(),
        "summary": meta_prompt_summary
    }

    # Zapisz batch do Firestore
    doc_ref.update({
        "batches": firestore.ArrayUnion([batch_data]),
        "keywords_state": keywords_state
    })

    return {
        "keywords_report": keywords_report,
        "over_terms_count": over_terms_count,
        "locked_terms_count": locked_terms_count,
        "under_terms_count": under_terms_count,
        "ok_terms_count": ok_terms_count,
        "meta_prompt_summary": meta_prompt_summary
    }


# ---------------------------------------------------------------
# ⚙️ Forced Regeneration / Emergency Exit
# ---------------------------------------------------------------
def trigger_forced_regeneration(project_id, result):
    """Uruchamia automatyczną regenerację batcha, gdy OVER ≥10."""
    print(f"⚠️ [Forced Regeneration] Projekt {project_id}: OVER={result['over_terms_count']}")
    doc_ref = db.collection("seo_projects").document(project_id)
    doc_ref.update({
        "status": "regenerating",
        "regeneration_triggered_at": datetime.utcnow().isoformat(),
        "regeneration_reason": "OVER ≥ 10"
    })
    return True


def trigger_emergency_exit(project_id, result):
    """Zatrzymuje generację, gdy LOCKED ≥4."""
    print(f"⛔ [Emergency Exit] Projekt {project_id}: LOCKED={result['locked_terms_count']}")
    doc_ref = db.collection("seo_projects").document(project_id)
    doc_ref.update({
        "status": "halted",
        "emergency_exit_triggered_at": datetime.utcnow().isoformat(),
        "emergency_exit_reason": "LOCKED ≥ 4"
    })
    return True


# ---------------------------------------------------------------
# ✅ /api/project/{project_id}/add_batch
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    try:
        global db
        if not db:
            return jsonify({"error": "Firestore nie jest połączony"}), 503

        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Brak tekstu batcha"}), 400

        print(f"[INFO] 🧠 Analiza batcha dla projektu: {project_id}")
        result = analyze_batch_text(project_id, text)

        regeneration_triggered = False
        emergency_exit_triggered = False

        if result["over_terms_count"] >= 10:
            regeneration_triggered = trigger_forced_regeneration(project_id, result)

        if result["locked_terms_count"] >= 4:
            emergency_exit_triggered = trigger_emergency_exit(project_id, result)

        return jsonify({
            "status": "OK",
            "regeneration_triggered": regeneration_triggered,
            "emergency_exit_triggered": emergency_exit_triggered,
            "keywords_report": result["keywords_report"],
            "meta_prompt_summary": result["meta_prompt_summary"]
        }), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id}/add_batch: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# ✅ /api/project/create
# ---------------------------------------------------------------
@project_bp.route("/api/project/create", methods=["POST"])
def create_project():
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
            "status": "created"
        })

        print(f"[INFO] ✅ Projekt {doc_ref.id} utworzony ({len(keywords_state)} fraz).")
        return jsonify({
            "status": "✅ Projekt utworzony",
            "project_id": doc_ref.id,
            "topic": topic,
            "keywords": len(keywords_state)
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    global db
    db = _db
    app.register_blueprint(project_bp)
    print("✅ [INIT] project_routes zarejestrowany (v7.2.2-full + ForcedReg/EmergencyExit).")
