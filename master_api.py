# master_api.py — Master SEO API (v5.7-PAA hybrid)
# Obsługa: Firestore, S1 (SerpApi + LangExtract + Ngram API), Brief Base64, Batch Counting, PAA Integration

import os
import re
import json
import base64
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# -------------------------------------------------------------------
# 🔧 Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
    if not FIREBASE_CREDS_JSON:
        print("❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON.")
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise ValueError("Brak FIREBASE_CREDS_JSON i pliku serviceAccountKey.json")
    else:
        creds_dict = json.loads(FIREBASE_CREDS_JSON)
        cred = credentials.Certificate(creds_dict)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Połączono z Firestore.")
except Exception as e:
    print(f"❌ Błąd inicjalizacji Firebase: {e}")
    db = None

# -------------------------------------------------------------------
# 🌐 API Zewnętrzne (S1)
# -------------------------------------------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"


def call_api_with_json(url, payload, name):
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd API {name}: {e}")
        return {"error": f"Błąd połączenia z {name}", "details": str(e)}


def call_serpapi(topic):
    params = {"api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"}
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd SerpAPI: {e}")
        return None


def call_langextract(url):
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract")

# -------------------------------------------------------------------
# 🧩 Parser briefu (BASIC / EXTENDED / H2 HEADERS TERMS)
# -------------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    keywords_dict = {}
    headers_list = []

    cleaned_text = os.linesep.join([s.strip() for s in brief_text.splitlines() if s.strip()])

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

# -------------------------------------------------------------------
# 🔢 Liczenie fraz (hierarchiczne, nienakładające się)
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
    text_lower = full_text.lower()
    sorted_keywords = sorted(keywords_dict.keys(), key=len, reverse=True)
    counts = {k: 0 for k in keywords_dict}
    masked_text = text_lower

    for kw in sorted_keywords:
        kw_lower = kw.lower()
        try:
            matches = re.findall(r"\b" + re.escape(kw_lower) + r"\b", masked_text)
            count = len(matches)
            counts[kw] = count
            if count > 0:
                masked_text = re.sub(r"\b" + re.escape(kw_lower) + r"\b", "X" * len(kw), masked_text, count=count)
        except re.error as e:
            print(f"Błąd regex dla '{kw}': {e}")
            continue

    return counts

# -------------------------------------------------------------------
# ✅ /api/project/create — tworzy projekt
# -------------------------------------------------------------------
@app.route("/api/project/create", methods=["POST"])
def create_project_hybrid():
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        data_json = request.get_json(silent=True)
        keywords_state, headers_list = {}, []

        # 🧩 Odczyt briefu (Base64 lub tekst)
        if data_json:
            brief_text = ""
            if "brief_base64" in data_json:
                try:
                    brief_text = base64.b64decode(data_json["brief_base64"]).decode("utf-8")
                except Exception as e:
                    return jsonify({"error": f"Błąd dekodowania Base64: {e}"}), 400
            elif "brief_text" in data_json:
                brief_text = data_json["brief_text"]
            else:
                return jsonify({"error": "Brak brief_text lub brief_base64."}), 400
        else:
            brief_text = request.data.decode("utf-8")
            if not brief_text:
                return jsonify({"error": "Brak danych w body (JSON lub text/plain)."}), 400

        # 🧠 Parsowanie briefu
        keywords_state, headers_list = parse_brief_to_keywords(brief_text)
        if not isinstance(keywords_state, dict):
            keywords_state = {}
        if not isinstance(headers_list, list):
            headers_list = []

        if not keywords_state:
            print("⚠️ Brak słów kluczowych – projekt zostanie oznaczony jako testowy, ale licznik będzie działał.")
            keywords_state = {}

        # 🗃️ Tworzenie dokumentu w Firestore
        doc_ref = db.collection("seo_projects").document()
        project_data = {
            "keywords_state": keywords_state,
            "headers_suggestions": headers_list,
            "s1_data": None,
            "full_text": "",
            "batches": [],
        }
        doc_ref.set(project_data)

        return jsonify({
            "status": "✅ Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state),
            "headers_count": len(headers_list),
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({"error": f"Błąd serwera: {e}"}), 500

# -------------------------------------------------------------------
# ✍️ /api/project/<id>/add_batch — dodaje batch i liczy frazy
# -------------------------------------------------------------------
@app.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt o podanym ID nie istnieje."}), 404

        project_data = doc.to_dict()
        current_keywords = project_data.get("keywords_state", {})
        current_full_text = project_data.get("full_text", "")
        batch_text = request.data.decode("utf-8")

        if not batch_text:
            return jsonify({"error": "Brak tekstu batcha."}), 400

        new_full_text = current_full_text + "\n\n" + batch_text
        new_counts = calculate_hierarchical_counts(new_full_text, current_keywords)
        report = []

        for kw, state in current_keywords.items():
            if state.get("locked"):
                report.append(f"{kw}: LOCKED (użyto max +3)")
                continue

            actual = new_counts.get(kw, 0)
            state["actual"] = actual
            state["remaining_min"] = max(0, state["target_min"] - actual)
            state["remaining_max"] = max(0, state["target_max"] - actual)

            if actual >= state["target_max"] + 3:
                state["locked"] = True
                status = f"LOCKED ({actual}/{state['target_max']} +3)"
            elif actual > state["target_max"]:
                status = f"OVER ({actual}/{state['target_max']})"
            elif actual < state["target_min"]:
                status = f"UNDER ({actual}/{state['target_min']})"
            else:
                status = "OK"

            report.append(f"{kw}: {actual} użyć / cel {state['target_min']}-{state['target_max']} / {status}")

        doc_ref.update({
            "keywords_state": current_keywords,
            "full_text": new_full_text,
            "batches": firestore.ArrayUnion([batch_text]),
        })

        return jsonify(report), 200

    except Exception as e:
        print(f"❌ Błąd add_batch: {e}")
        return jsonify({"error": f"Błąd serwera: {e}"}), 500

# -------------------------------------------------------------------
# 🧠 /api/s1_analysis — obowiązkowa analiza konkurencji + PAA
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    try:
        data = request.get_json()
        if not data or "topic" not in data:
            return jsonify({"error": "Brak 'topic' w body."}), 400

        topic = data["topic"]
        serp_data = call_serpapi(topic)

        if not serp_data:
            return jsonify({"error": "Nie udało się pobrać danych z SerpAPI."}), 502

        ai_overview_status = serp_data.get("ai_overview", {}).get("status", "not_available")
        ai_mode_results = len(serp_data.get("ai_results", [])) if "ai_results" in serp_data else 0
        people_also_ask = [q.get("question") for q in serp_data.get("related_questions", []) if q.get("question")]
        autocomplete_suggestions = [r.get("query") for r in serp_data.get("related_searches", []) if r.get("query")]

        top_urls = [r.get("link") for r in serp_data.get("organic_results", [])[:5]]
        combined_text, h2_counts, all_headings, source_log = "", [], [], []
        successful = 0

        for url in top_urls:
            if successful >= 3:
                break
            content = call_langextract(url)
            if content and content.get("content"):
                successful += 1
                h2s = content.get("h2", [])
                all_headings.extend([h.strip().lower() for h in h2s])
                h2_counts.append(len(h2s))
                combined_text += content.get("content", "") + "\n\n"
                source_log.append({"url": url, "status": "Success", "h2_count": len(h2s)})
            else:
                source_log.append({"url": url, "status": "Failure"})

        heading_counts = Counter(all_headings)
        top_headings = [h for h, _ in heading_counts.most_common(10)]

        ngram_data = call_api_with_json(
            NGRAM_API_URL,
            {"text": combined_text, "main_keyword": topic},
            "Ngram API"
        )

        competitive_metrics = {
            "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
            "min_h2": min(h2_counts) if h2_counts else 0,
            "max_h2": max(h2_counts) if h2_counts else 0
        }

        return jsonify({
            "identified_urls": top_urls,
            "processing_report": source_log,
            "competitive_metrics": competitive_metrics,
            "ai_overview_status": ai_overview_status,
            "ai_mode_results": ai_mode_results,
            "people_also_ask": people_also_ask,
            "autocomplete_suggestions": autocomplete_suggestions,
            "top_competitor_headings": top_headings,
            "s1_enrichment": ngram_data
        }), 200

    except Exception as e:
        print(f"❌ Błąd /api/s1_analysis: {e}")
        return jsonify({"error": f"Błąd serwera: {e}"}), 500

# -------------------------------------------------------------------
# 🧹 DELETE /api/project/<id>
# -------------------------------------------------------------------
@app.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    try:
        doc_ref = db.collection("seo_projects").document(project_id)
        if not doc_ref.get().exists:
            return jsonify({"error": "Projekt nie istnieje."}), 404
        doc_ref.delete()
        return jsonify({"status": f"Projekt {project_id} usunięty."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# ❤️ Health Check
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v5.7-paa-hybrid",
        "message": "Master SEO API działa poprawnie (S1 z PAA, Firestore aktywny)."
    }), 200

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
