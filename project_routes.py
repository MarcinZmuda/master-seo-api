# ================================================================
# project_routes.py — Warstwa Project Management (v6.5.0 - Lematyzacja + Priorytety)
# ================================================================

import json
import base64
import re
import os
from flask import Blueprint, request, jsonify
from collections import Counter
from datetime import datetime
import requests

# --- 🔐 Inicjalizacja Firebase ---
from firebase_admin import firestore
import firebase_admin

db = None

project_bp = Blueprint("project_routes", __name__)

# === 🔽 Inicjalizacja spaCy (do liczenia odmian) 🔽 ===
import spacy
try:
    # Ładujemy model polski
    NLP = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie.")
except OSError:
    print("❌ BŁĄD: Nie można załadować modelu spaCy 'pl_core_news_sm'.")
    print("Upewnij się, że jest dodany do requirements.txt (link .tar.gz).")
    NLP = None

def lemmatize_text(text):
    """Zwraca listę lematów z tekstu (bez interpunkcji, małe litery)."""
    if not NLP:
        # Fallback, jeśli spaCy nie działa - zwróci zwykłe słowa
        return re.findall(r'\b\w+\b', text.lower())
    doc = NLP(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]
# === 🔼 KONIEC SEKCJI spaCy 🔼 ===


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
            
            # Pre-lematyzacja fraz kluczowych
            keyword_lemmas = lemmatize_text(keyword)
            
            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "actual": 0,
                "locked": False,
                "lemmas": keyword_lemmas, # Zapisujemy lematy frazy
                "lemma_len": len(keyword_lemmas) # Zapisujemy długość
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
        if not NLP:
             return jsonify({"error": "Krytyczny błąd serwera: Model spaCy nie jest załadowany."}), 500

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

        # Parser od razu przeprowadzi lematyzację fraz
        keywords_state, headers_list = parse_brief_to_keywords(brief_text) if brief_text else ({}, [])
        
        s1_data = call_s1_analysis(topic)

        if "error" in s1_data:
            return jsonify({"error": "Błąd podrzędny podczas analizy S1", "details": s1_data["error"]}), 500

        doc_ref = db.collection("seo_projects").document()
        project_data = {
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "brief_text": brief_text[:5000],
            "keywords_state": keywords_state, # Zawiera już lematy
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
    if not NLP:
         return jsonify({"error": "Krytyczny błąd serwera: Model spaCy nie jest załadowany."}), 500

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

        # === Logika Lematyzacji ===
        text_lemmas = lemmatize_text(text_input)
        text_lemmas_len = len(text_lemmas)
        counts_in_batch = {}
        
        # Sortujemy frazy od najdłuższej do najkrótszej (wg liczby lematów)
        sorted_keywords = sorted(keywords_state.items(), key=lambda item: item[1].get('lemma_len', 0), reverse=True)

        for kw, meta in sorted_keywords:
            keyword_lemmas = meta.get("lemmas", [])
            kw_len = meta.get("lemma_len", 0)
            
            if kw_len == 0:
                counts_in_batch[kw] = 0
                continue

            count_in_batch = 0
            # Algorytm okna przesuwnego do szukania sekwencji lematów
            for i in range(text_lemmas_len - kw_len + 1):
                if text_lemmas[i:i + kw_len] == keyword_lemmas:
                    count_in_batch += 1
            
            # Aktualizujemy stan (łączną liczbę)
            meta["actual"] = meta.get("actual", 0) + count_in_batch
            counts_in_batch[kw] = count_in_batch

            # Ustalanie statusu na podstawie łącznej liczby
            if meta["actual"] > meta["target_max"] + 3:
                meta["locked"] = True
                meta["status"] = "LOCKED"
            elif meta["actual"] > meta["target_max"]:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"
            
            # Musimy zaktualizować słownik, aby zmiany były widoczne dla kolejnych iteracji
            keywords_state[kw] = meta 
        # === Koniec logiki Lematyzacji ===

        batch_entry = {
            "created_at": datetime.utcnow().isoformat(),
            "length": len(text_input),
            "counts": counts_in_batch, # Licznik tylko dla tego batcha
            "text": text_input[:5000]
        }
        batches.append(batch_entry)

        # Zapisujemy zaktualizowany stan do bazy
        doc_ref.update({
            "batches": firestore.ArrayUnion([batch_entry]),
            "keywords_state": keywords_state,
            "updated_at": datetime.utcnow().isoformat()
        })

        # === 🔽 NOWY, STRUKTURALNY RAPORT 🔽 ===
        locked_terms = []
        keywords_report = []
        
        # Generujemy raport na podstawie świeżo zaktualizowanego keywords_state
        for kw, meta in keywords_state.items():
            status = meta.get('status', 'OK')
            priority_instruction = "USE_AS_NEEDED" # Domyślna dla OK i OVER
            
            if status == "LOCKED":
                priority_instruction = "DO_NOT_USE" # Totalnie zablokowana
                locked_terms.append(kw)
            elif status == "UNDER":
                priority_instruction = "PRIORITY_USE" # Priorytet
            
            keywords_report.append({
                "keyword": kw,
                "actual_uses": meta.get('actual', 0),
                "target_range": f"{meta.get('target_min', 0)}-{meta.get('target_max', 0)}",
                "status": status,
                "priority_instruction": priority_instruction
            })
        # === 🔼 KONIEC NOWEGO RAPORTU 🔼 ===

        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "counts_in_batch": counts_in_batch,   # Ile zliczono w TYM BATCHU
            "keywords_report": keywords_report,   # NOWY, strukturalny raport dla GPT
            "locked_terms": locked_terms,         # Wciąż przydatne jako szybka lista
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
    """Rejestruje trasy i przekazuje instancję bazy danych z master_api.py."""
    global db
    db = _db
    
    app.register_blueprint(project_bp)
    
    if db:
        print("✅ [DEBUG] Zarejestrowano project_routes (przekazano instancję 'db').")
    else:
        print("⚠️ [DEBUG] Zarejestrowano project_routes (instancja 'db' jest None!).")
