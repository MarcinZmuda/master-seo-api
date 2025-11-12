# ================================================================
# project_routes.py — Warstwa Project Management (v7.2.1 - Fixed + Blueprint Register)
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
# 🧠 Funkcje językowe
# ---------------------------------------------------------------
def lemmatize_text(text):
    """Zwraca listę lematów z tekstu (bez interpunkcji)."""
    if not NLP:
        return re.findall(r'\b\w+\b', text.lower())
    doc = NLP(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]


def get_root_prefix(word):
    """Wyznacza dynamiczny rdzeń semantyczny słowa."""
    if len(word) <= 6:
        return word
    vowels = 'aeiouyąęó'
    root_len = 6
    for i, ch in enumerate(word):
        if ch in vowels:
            root_len = i + 3
            break
    return word[:max(6, root_len)]


def extract_context_matches(text, root_prefix, related_terms=None):
    """Wykrywa kontekstowe wystąpienia rdzenia w otoczeniu powiązanych terminów."""
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
    """Bezpieczne wywołanie analizy S1 z dynamicznym adresem API."""
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
# 🔢 Hierarchiczne liczenie
# ---------------------------------------------------------------
def apply_hierarchical_counting(raw_counts):
    if not isinstance(raw_counts, dict):
        return raw_counts

    keywords = sorted(raw_counts.keys(), key=len, reverse=True)
    hierarchical_counts = raw_counts.copy()

    for i, long_kw in enumerate(keywords):
        for short_kw in keywords[i + 1:]:
            if short_kw in long_kw and re.search(r'\b' + re.escape(short_kw) + r'\b', long_kw):
                hierarchical_counts[short_kw] += raw_counts.get(long_kw, 0)
    return hierarchical_counts


# ---------------------------------------------------------------
# 🧩 Parser briefu z obsługą BASIC / EXTENDED
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """Parsuje brief i wyciąga frazy kluczowe BASIC / EXTENDED."""
    keywords_dict = {}
    headers_list = []
    cleaned_text = "\n".join([s.strip() for s in brief_text.splitlines() if s.strip()])

    section_regex = r"((?:BASIC|EXTENDED|H2)\s+(?:TEXT|HEADERS)\s+TERMS)\s*:\s*=*\s*([\s\S]*?)(?=\n[A-Z\s]+(?:TEXT|HEADERS)\s+TERMS|$)"
    keyword_regex = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$", re.UNICODE)
    keyword_regex_single = re.compile(r"^\s*(.*?)\s*:\s*(\d+)x\s*$", re.UNICODE)

    for match in re.finditer(section_regex, cleaned_text, re.IGNORECASE):
        section_name_raw = match.group(1).upper()
        section_content = match.group(2)

        section_type = "BASIC"
        if "EXTENDED" in section_name_raw:
            section_type = "EXTENDED"

        if "H2" in section_name_raw:
            for line in section_content.splitlines():
                if line.strip():
                    headers_list.append(line.strip())
            continue

        for line in section_content.splitlines():
            line = line.strip()
            if not line:
                continue

            kw_match = keyword_regex.match(line)
            if kw_match:
                keyword = kw_match.group(1).strip()
                min_val = int(kw_match.group(2))
                max_val = int(kw_match.group(3))
            else:
                kw_match_single = keyword_regex_single.match(line)
                if kw_match_single:
                    keyword = kw_match_single.group(1).strip()
                    min_val = max_val = int(kw_match_single.group(2))
                else:
                    continue

            keyword_lemmas = lemmatize_text(keyword)
            root_prefix = get_root_prefix(keyword_lemmas[0]) if keyword_lemmas else get_root_prefix(keyword)

            if section_type == "EXTENDED":
                min_val = max(1, round(min_val * 0.5))
                max_val = max(1, round(max_val * 0.5))

            keywords_dict[keyword] = {
                "type": section_type,
                "target_min": min_val,
                "target_max": max_val,
                "actual": 0,
                "contextual_hits": 0,
                "semantic_coverage": 0.0,
                "locked": False,
                "lemmas": keyword_lemmas,
                "lemma_len": len(keyword_lemmas),
                "root_prefix": root_prefix
            }

    return keywords_dict, headers_list


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
# 🔧 Rejestracja blueprinta (zgodna z master_api.py)
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    """Rejestruje blueprinta project_routes w aplikacji Flask."""
    global db
    db = _db
    app.register_blueprint(project_bp)
    print("✅ [INIT] project_routes zarejestrowany (v7.2.1-fixed).")
