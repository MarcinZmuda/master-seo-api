# ================================================================
# project_routes.py — Warstwa Project Management (v6.3.0)
# Obsługa: Firestore + integracja z Master SEO API (S1 Analysis)
# ================================================================

import json
import base64
import re
from flask import Blueprint, request, jsonify
from collections import Counter
from datetime import datetime
import requests

# --- Blueprint dla modularności ---
project_bp = Blueprint("project_routes", __name__)

# ---------------------------------------------------------------
# 🔧 Funkcje pomocnicze
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """Parsuje tekst briefu i wyciąga słowa kluczowe + nagłówki H2."""
    keywords_dict = {}
    headers_list = []

    cleaned_text = "\n".join([s.strip() for s in brief_text.splitlines() if s.strip()])
    section_regex = r"((?:BASIC|EXTENDED|H2)\s+TEXT\s+TERMS)\s*:\s*=*\s*([\s\S]*?)(?=\n[A-Z\s]+TEXT\s+TERMS|$)"
    keyword_regex = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$", re.UNICODE)
    keyword_regex_single = re.compile(r"^\s*(.*?)\s*:\s*(\d+)x\s*$", re.UNICODE)

    for match in re.finditer(section_regex, cleaned_text, re.IGNORECASE):
        section_name = match.group(1).upper()
        section_content = match.group(2)
        if section_name.startswith("H2"):
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

            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "remaining_min": min_val,
                "remaining_max": max_val,
                "actual": 0,
                "locked": False,
            }

    return keywords_dict, headers_list


def call_s1_analysis(topic):
    """Wywołuje wewnętrznie endpoint /api/s1_analysis (lokalnie lub zewnętrznie)."""
    try:
        r = requests.post("http://localhost:8080/api/s1_analysis", json={"topic": topic}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Błąd wywołania S1 Analysis: {str(e)}"}


# ---------------------------------------------------------------
# ✅ /api/project/create — tworzy nowy projekt SEO
# ---------------------------------------------------------------
@project_bp.route("/api/project/create", methods=["POST"])
def create_project():
    from firebase_admin import firestore
    db = project_bp.db

    try:
        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "").strip()
        brief_text = ""

        if not topic:
            return jsonify({"error": "Brak 'topic' (frazy kluczowej)"}), 400

        # Obsługa briefu (tekst lub base64)
        if "brief_base64" in data:
            brief_text = base64.b64decode(data["brief_base64"]).decode("utf-8")
        elif "brief_text" in data:
            brief_text = data["brief_text"]

        keywords_state, headers_list = parse_brief_to_keywords(brief_text) if brief_text else ({}, [])
        s1_data = call_s1_analysis(topic)

        doc_ref = db.collection("seo_projects").document()
        project_data = {
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "brief_text": brief_text[:5000],
            "keywords_state": keywords_state,
            "headers_suggestions": headers_list,
            "s1_data": s1_data,
            "batches": [],
            "status": "created",
        }
        doc_ref.set(project_data)

        return jsonify({
            "status": "✅ Projekt utworzony",
            "project_id": doc_ref.id,
            "topic": topic,
            "keywords": len(keywords_state),
            "headers": len(headers_list),
            "s1_summary": s1_data.get("competitive_metrics", {}),
        }), 201

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project/create: {str(e)}"}), 500


# ---------------------------------------------------------------
# 📄 /api/project/<id> — pobiera dane projektu
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>", methods=["GET"])
def get_project(project_id):
    db = project_bp.db
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Projekt nie istnieje"}), 404
    return jsonify(doc.to_dict()), 200


# ---------------------------------------------------------------
# ✏️ /api/project/<id>/update — aktualizuje projekt
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/update", methods=["POST"])
def update_project(project_id):
    db = project_bp.db
    data = request.get_json(silent=True) or {}
    doc_ref = db.collection("seo_projects").document(project_id)

    if not doc_ref.get().exists:
        return jsonify({"error": "Projekt nie istnieje"}), 404

    doc_ref.update(data)
    return jsonify({"status": f"✅ Projekt {project_id} zaktualizowany"}), 200


# ---------------------------------------------------------------
# 🗑️ /api/project/<id>/delete — usuwa projekt
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/delete", methods=["DELETE"])
def delete_project(project_id):
    db = project_bp.db
    doc_ref = db.collection("seo_projects").document(project_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "Projekt nie istnieje"}), 404
    doc_ref.delete()
    return jsonify({"status": f"🗑️ Projekt {project_id} usunięty"}), 200


# ---------------------------------------------------------------
# 🔧 Funkcja rejestrująca blueprint
# ---------------------------------------------------------------
def register_project_routes(app, db):
    project_bp.db = db
    app.register_blueprint(project_bp)
    print("✅ [DEBUG] Zarejestrowano project_routes (Firestore mode).")
