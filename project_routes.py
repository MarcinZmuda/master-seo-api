# ================================================================
# project_routes.py — Warstwa Project Management (v6.3.0)
# Obsługa: Firestore + integracja z Master SEO API (S1–S4)
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
# ✅ /api/project/create — tworzy nowy projekt SEO (S2)
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
    # automatyczna konwersja, jeśli brief zbyt długi
    if len(brief_text) > 2000:
        data["brief_base64"] = base64.b64encode(brief_text.encode("utf-8")).decode("utf-8")
        brief_text = base64.b64decode(data["brief_base64"]).decode("utf-8")
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
# 🧠 /api/project/<id>/add_batch — dodaje batch treści (S3)
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    from firebase_admin import firestore
    db = project_bp.db

    if not db:
        return jsonify({"error": "Brak połączenia z Firestore"}), 503

    try:
        # 🔹 Pobierz dokument projektu
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        batches = project_data.get("batches", [])

        # 🔹 Odczytaj dane z body (obsługa JSON i text/plain)
        text_input = ""
        if request.is_json:
            text_input = (request.get_json() or {}).get("text", "")
        else:
            text_input = request.data.decode("utf-8", errors="ignore")

        if not text_input.strip():
            return jsonify({"error": "Brak treści w żądaniu"}), 400

        text_clean = text_input.lower()
        text_clean = re.sub(r"[^\w\sąćęłńóśźż]", " ", text_clean)

        # 🔹 Liczenie wystąpień
        counts = {}
        for kw, meta in keywords_state.items():
            pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
            matches = re.findall(pattern, text_clean, flags=re.UNICODE)
            count = len(matches)
            meta["actual"] += count
            counts[kw] = count

            # 🔸 Logika blokowania (LOCKED / OVER / UNDER)
            if meta["actual"] > meta["target_max"] + 3:
                meta["locked"] = True
                meta["status"] = "LOCKED"
            elif meta["actual"] > meta["target_max"]:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"

        # 🔹 Aktualizacja Firestore
        batch_entry = {
            "created_at": datetime.utcnow().isoformat(),
            "length": len(text_input),
            "counts": counts,
            "text": text_input[:5000]  # limit zapisu dla Firestore
        }
        batches.append(batch_entry)

        doc_ref.update({
            "batches": firestore.ArrayUnion([batch_entry]),
            "keywords_state": keywords_state,
            "updated_at": datetime.utcnow().isoformat()
        })

        # 🔹 Raport
        report_lines = []
        for kw, meta in keywords_state.items():
            report_lines.append(
                f"{kw}: {meta['actual']} użyć / cel {meta['target_min']}-{meta['target_max']} / {meta.get('status', 'OK')}"
            )

        locked_terms = [kw for kw, meta in keywords_state.items() if meta.get("locked")]

        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "counts": counts,
            "report": report_lines,
            "locked_terms": locked_terms,
            "updated_keywords": len(keywords_state)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project/add_batch: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧹 /api/project/<id> — finalne usunięcie projektu + raport podsumowujący (S4)
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project_final(project_id):
    from firebase_admin import firestore
    db = project_bp.db

    if not db:
        return jsonify({"error": "Brak połączenia z Firestore"}), 503

    try:
        # 🔹 Pobierz projekt
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()

        # 🔹 Przygotuj raport końcowy
        topic = project_data.get("topic", "nieznany temat")
        keywords_state = project_data.get("keywords_state", {})
        batches = project_data.get("batches", [])

        total_batches = len(batches)
        total_length = sum(b.get("length", 0) for b in batches)
        locked_terms = [kw for kw, meta in keywords_state.items() if meta.get("locked")]
        over_terms = [kw for kw, meta in keywords_state.items() if meta.get("status") == "OVER"]
        under_terms = [kw for kw, meta in keywords_state.items() if meta.get("status") == "UNDER"]
        ok_terms = [kw for kw, meta in keywords_state.items() if meta.get("status") == "OK"]

        summary_report = {
            "topic": topic,
            "total_batches": total_batches,
            "total_length": total_length,
            "locked_terms_count": len(locked_terms),
            "over_terms_count": len(over_terms),
            "under_terms_count": len(under_terms),
            "ok_terms_count": len(ok_terms),
            "locked_terms": locked_terms,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # 🔹 Zapisz kopię raportu do kolekcji archiwalnej
        db.collection("seo_projects_archive").document(project_id).set(summary_report)

        # 🔹 Usuń oryginalny projekt
        doc_ref.delete()

        # 🔹 Zwróć raport końcowy
        return jsonify({
            "status": f"✅ Projekt {project_id} został usunięty z Firestore.",
            "summary": summary_report
        }), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project DELETE: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🔧 Funkcja rejestrująca blueprint
# ---------------------------------------------------------------
def register_project_routes(app, db):
    project_bp.db = db
    app.register_blueprint(project_bp)
    print("✅ [DEBUG] Zarejestrowano project_routes (Firestore mode).")
