# ================================================================
# project_routes.py — Warstwa Project Management (v6.3.3 - z raportowaniem)
# ================================================================

import json
import base64
import re
import os
from flask import Blueprint, request, jsonify
from collections import Counter
from datetime import datetime
import requests

# --- 🔐 Inicjalizacja Firebase (została usunięta) ---
from firebase_admin import firestore
import firebase_admin

db = None

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
                "remaining_min": min_val, # Te 'remaining' nie są obecnie używane, ale OK
                "remaining_max": max_val, # Te 'remaining' nie są obecnie używane, ale OK
                "actual": 0,
                "locked": False,
            }

    return keywords_dict, headers_list


def call_s1_analysis(topic):
    """Wywołuje wewnętrznie endpoint /api/s1_analysis (lokalnie lub zewnętrznie)."""
    try:
        r = requests.post("http://localhost:10000/api/s1_analysis", json={"topic": topic}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Błąd wywołania S1 Analysis: {str(e)}"}


# ---------------------------------------------------------------
# ✅ /api/project/create — tworzy nowy projekt SEO (S2)
# ---------------------------------------------------------------
@project_bp.route("/api/project/create", methods=["POST"])
def create_project():
    try:
        if not db:
            return jsonify({"error": "Firestore nie jest połączony (instancja db jest None)"}), 503

        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "").strip()
        brief_text = ""

        if not topic:
            return jsonify({"error": "Brak 'topic' (frazy kluczowej)"}), 400

        if "brief_base64" in data:
            brief_text = base64.b64decode(data["brief_base64"]).decode("utf-8")
        elif "brief_text" in data:
            brief_text = data["brief_text"]
            if len(brief_text) > 2000:
                data["brief_base64"] = base64.b64encode(brief_text.encode("utf-8")).decode("utf-8")
                brief_text = base64.b64decode(data["brief_base64"]).decode("utf-8")

        keywords_state, headers_list = parse_brief_to_keywords(brief_text) if brief_text else ({}, [])
        
        s1_data = call_s1_analysis(topic)

        if "error" in s1_data:
            return jsonify({"error": "Błąd podrzędny podczas analizy S1", "details": s1_data["error"]}), 500

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
    if not db:
        return jsonify({"error": "Brak połączenia z Firestore"}), 503

    try:
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        batches = project_data.get("batches", [])

        text_input = ""
        if request.is_json:
            text_input = (request.get_json() or {}).get("text", "")
        else:
            text_input = request.data.decode("utf-8", errors="ignore")

        if not text_input.strip():
            return jsonify({"error": "Brak treści w żądaniu"}), 400

        text_clean = text_input.lower()
        text_clean = re.sub(r"[^\w\sąćęłńóśźż]", " ", text_clean)

        counts = {} # Licznik dla bieżącego batcha
        
        # --- Zaktualizowana logika liczenia i raportowania ---
        
        for kw, meta in keywords_state.items():
            pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
            matches = re.findall(pattern, text_clean, flags=re.UNICODE)
            count_in_batch = len(matches)
            
            meta["actual"] += count_in_batch # Aktualizuj łączną liczbę
            counts[kw] = count_in_batch     # Zapisz liczbę z tego batcha

            # Ustal status na podstawie ŁĄCZNEJ liczby
            if meta["actual"] > meta["target_max"] + 3:
                meta["locked"] = True
                meta["status"] = "LOCKED"
            elif meta["actual"] > meta["target_max"]:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"

        batch_entry = {
            "created_at": datetime.utcnow().isoformat(),
            "length": len(text_input),
            "counts": counts, # Zapisujemy w batchu tylko liczniki z tego batcha
            "text": text_input[:5000]
        }
        batches.append(batch_entry)

        doc_ref.update({
            "batches": firestore.ArrayUnion([batch_entry]),
            "keywords_state": keywords_state, # Zapisujemy zaktualizowany stan (z nowymi 'actual' i 'status')
            "updated_at": datetime.utcnow().isoformat()
        })

        # === 🔽 TUTAJ JEST NOWY KOD 🔽 ===
        
        # 1. Stwórz listę zablokowanych terminów
        locked_terms = [kw for kw, meta in keywords_state.items() if meta.get("locked")]

        # 2. Wygeneruj pełny raport stanu (to, czego brakowało)
        report_list = []
        for kw, meta in keywords_state.items():
            report_str = f"{kw}: {meta['actual']} użyć / cel {meta['target_min']}-{meta['target_max']} / {meta.get('status', 'OK')}"
            report_list.append(report_str)

        # === KONIEC NOWEGO KODU ===

        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "counts": counts,           # Ile zliczono w TYM BATCHU
            "report": report_list,      # Pełny raport stanu (ŁĄCZNIE)
            "locked_terms": locked_terms, # Lista zablokowanych (ŁĄCZNIE)
            "updated_keywords": len(keywords_state)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project/add_batch: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧹 /api/project/<id> — finalne usunięcie projektu (S4)
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project_final(project_id):
    if not db:
        return jsonify({"error": "Brak połączenia z Firestore"}), 503

    try:
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        batches = project_data.get("batches", [])
        
        # --- Zliczanie statusów do raportu końcowego ---
        status_counts = {"LOCKED": 0, "OVER": 0, "UNDER": 0, "OK": 0}
        locked_terms_final = []
        for kw, meta in keywords_state.items():
            status = meta.get("status", "OK")
            status_counts[status] += 1
            if meta.get("locked"):
                locked_terms_final.append(kw)

        summary_report = {
            "topic": project_data.get("topic", "nieznany temat"),
            "total_batches": len(batches),
            "total_length": sum(b.get("length", 0) for b in batches),
            "locked_terms_count": status_counts["LOCKED"],
            "over_terms_count": status_counts["OVER"],
            "under_terms_count": status_counts["UNDER"],
            "ok_terms_count": status_counts["OK"],
            "locked_terms": locked_terms_final,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Archiwizuj raport przed usunięciem (opcjonalne, ale dobra praktyka)
        db.collection("seo_projects_archive").document(project_id).set(summary_report)
        doc_ref.delete()

        return jsonify({
            "status": f"✅ Projekt {project_id} został usunięty z Firestore.",
            "summary": summary_report
        }), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project DELETE: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🔧 Funkcja rejestrująca blueprint
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    global db
    db = _db
    
    app.register_blueprint(project_bp)
    
    if db:
        print("✅ [DEBUG] Zarejestrowano project_routes (przekazano instancję 'db').")
    else:
        print("⚠️ [DEBUG] Zarejestrowano project_routes (instancja 'db' jest None!).")
