import os
import re
import json
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
# KROK 1: Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
    if not FIREBASE_CREDS_JSON:
        print("❌ KRYTYCZNY BŁĄD: Brak zmiennej środowiskowej FIREBASE_CREDS_JSON.")
        if os.path.exists('serviceAccountKey.json'):
            print("🔧 Znaleziono lokalny plik 'serviceAccountKey.json'. Używam go...")
            cred = credentials.Certificate('serviceAccountKey.json')
        else:
            raise ValueError("Brak FIREBASE_CREDS_JSON i serviceAccountKey.json")
    else:
        creds_dict = json.loads(FIREBASE_CREDS_JSON)
        cred = credentials.Certificate(creds_dict)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Pomyślnie połączono z Firestore.")

except Exception as e:
    print(f"❌ KRYTYCZNY BŁĄD: Nie można zainicjować Firebase: {e}")
    db = None

# -------------------------------------------------------------------
# Konfiguracja API i funkcje pomocnicze (dla S1)
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
# ✅ KROK 2A: Parser Tekstu (fallback)
# -------------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    keywords_dict = {}
    section_regex = r'((?:BASIC|EXTENDED)\s*.*?\s*TERMS):\s*={10,}\s*([\s\S]*?)(?=\n[A-Z\s]+ TERMS:|$)'
    keyword_regex = re.compile(r'^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$', re.UNICODE)
    keyword_regex_single = re.compile(r'^\s*(.*?)\s*:\s*(\d+)x\s*$', re.UNICODE)

    cleaned_brief_text = os.linesep.join([s for s in brief_text.splitlines() if s.strip()])

    for match in re.finditer(section_regex, cleaned_brief_text, re.IGNORECASE):
        section_name = match.group(1).upper()
        section_content = match.group(2)

        if not (section_name.startswith("BASIC") or section_name.startswith("EXTENDED")):
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
                    min_val = int(kw_match_single.group(2))
                    max_val = int(kw_match_single.group(2))
                else:
                    continue

            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "remaining_min": min_val,
                "remaining_max": max_val,
                "actual": 0,
                "locked": False
            }

    return keywords_dict

# -------------------------------------------------------------------
# ✅ KROK 2B: Parser JSON (preferowany)
# -------------------------------------------------------------------
def parse_brief_from_json(data_json):
    keywords_state = {}
    terms_sources = [data_json.get('basic_terms', {}), data_json.get('extended_terms', {})]

    for terms_dict in terms_sources:
        if not isinstance(terms_dict, dict):
            continue

        for keyword, limits in terms_dict.items():
            try:
                limits_clean = str(limits).strip().lower().replace(' ', '')
                if '-' in limits_clean:
                    min_val_str, max_val_str = limits_clean.replace('x', '').split('-')
                    min_val = int(min_val_str)
                    max_val = int(max_val_str)
                else:
                    min_val = max_val = int(limits_clean.replace('x', ''))

                keywords_state[keyword.strip()] = {
                    "target_min": min_val,
                    "target_max": max_val,
                    "remaining_min": min_val,
                    "remaining_max": max_val,
                    "actual": 0,
                    "locked": False
                }
            except Exception as e:
                print(f"Błąd parsowania limitu JSON '{limits}' dla frazy '{keyword}': {e}")
                continue

    return keywords_state

# -------------------------------------------------------------------
# KROK 3: Liczenie hierarchiczne (bez zmian)
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
    text_lower = full_text.lower()
    sorted_keywords = sorted(keywords_dict.keys(), key=len, reverse=True)
    counts = {k: 0 for k in keywords_dict}
    masked_text = text_lower

    for kw in sorted_keywords:
        kw_lower = kw.lower()
        try:
            matches = re.findall(r'\b' + re.escape(kw_lower) + r'\b', masked_text)
            count = len(matches)
            counts[kw] = count
            if count > 0:
                masked_text = re.sub(r'\b' + re.escape(kw_lower) + r'\b', "X" * len(kw), masked_text, count=count)
        except re.error as e:
            print(f"Błąd regex dla frazy '{kw}': {e}")
            continue

    return counts

# -------------------------------------------------------------------
# ✅ KROK 4: Endpoint /api/project/create (HYBRYDOWY)
# -------------------------------------------------------------------
@app.route("/api/project/create", methods=["POST"])
def create_project_hybrid():
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    keywords_state = {}
    data_json = request.get_json(silent=True)

    if data_json:
        print("INFO: Otrzymano JSON – parsowanie...")
        keywords_state = parse_brief_from_json(data_json)
    else:
        print("WARN: JSON nie wykryty – próba parsowania jako text/plain")
        try:
            brief_text = request.data.decode('utf-8')
            if not brief_text:
                return jsonify({"error": "Brak danych (JSON lub text) w body."}), 400
            keywords_state = parse_brief_to_keywords(brief_text)
        except Exception as e:
            print(f"CRITICAL: Błąd parsowania briefu: {e}")
            return jsonify({"error": "Brak danych lub nieobsługiwany format kodowania."}), 400

    if not keywords_state:
        return jsonify({"error": "Nie udało się sparsować słów kluczowych."}), 400

    try:
        doc_ref = db.collection('seo_projects').document()
        project_data = {
            "keywords_state": keywords_state,
            "full_text": "",
            "batches": []
        }
        doc_ref.set(project_data)

        return jsonify({
            "status": "Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state)
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({"error": f"Błąd zapisu do Firestore: {e}"}), 500

# -------------------------------------------------------------------
# Endpoint: /api/project/<id>/add_batch
# -------------------------------------------------------------------
@app.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt o podanym ID nie istnieje."}), 404

        project_data = doc.to_dict()
        current_keywords_state = project_data.get('keywords_state', {})
        current_full_text = project_data.get('full_text', "")
        batch_text = request.data.decode('utf-8')

        if not batch_text:
            return jsonify({"error": "Brak tekstu w body żądania."}), 400

        new_full_text = current_full_text + "\n\n" + batch_text
        new_counts = calculate_hierarchical_counts(new_full_text, current_keywords_state)
        report_for_gpt = []

        for keyword, state in current_keywords_state.items():
            if state.get('locked', False):
                report_for_gpt.append(f"{keyword}: LOCKED (Użyto max + 3)")
                continue

            state['actual'] = new_counts.get(keyword, 0)
            state['remaining_min'] = max(0, state['target_min'] - state['actual'])
            state['remaining_max'] = max(0, state['target_max'] - state['actual'])

            status = "OK"
            if state['actual'] >= state['target_max'] + 3:
                state['locked'] = True
                status = f"LOCKED (Użyto {state['actual']} / Cel: {state['target_max']}. Przekroczono o 3+)"
            elif state['actual'] > state['target_max']:
                status = f"OVER ({state['actual']} / Cel: {state['target_max']})"
            elif state['actual'] < state['target_min']:
                status = f"UNDER ({state['actual']} / Cel: {state['target_min']})"

            report_for_gpt.append(f"{keyword}: {state['actual']} użyto / Cel: {state['target_min']}-{state['target_max']} / Status: {status}")

        doc_ref.update({
            "keywords_state": current_keywords_state,
            "full_text": new_full_text,
            "batches": firestore.ArrayUnion([batch_text])
        })

        return jsonify(report_for_gpt), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id}/add_batch: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500

# -------------------------------------------------------------------
# Endpoint: /api/project/<id> (DELETE)
# -------------------------------------------------------------------
@app.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt o podanym ID nie istnieje."}), 404

        doc_ref.delete()
        return jsonify({"status": f"Projekt {project_id} został pomyślnie usunięty."}), 200
    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id} [DELETE]: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500

# -------------------------------------------------------------------
# Endpoint: /api/health
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v5.3-hybrid-parser",
        "message": "Master SEO API (Firestore Edition) działa poprawnie."
    }), 200

# -------------------------------------------------------------------
# Endpoint: /api/s1_analysis
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Brak danych JSON"}), 400
        topic = data.get("topic")
        if not topic:
            return jsonify({"error": "Brak 'topic'"}), 400
    except Exception:
        return jsonify({"error": "Nieprawidłowy format JSON"}), 400

    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Błąd pobierania danych z SerpApi"}), 502

    top_urls = [res.get("link") for res in serp_data.get("organic_results", [])[:5]]
    source_log, h2_counts, all_headings, combined_text = [], [], [], ""
    successful_sources = 0

    for url in top_urls:
        if successful_sources >= 3:
            break
        content = call_langextract(url)
        if content and content.get("content"):
            successful_sources += 1
            h2s = content.get("h2", [])
            all_headings.extend([h.strip().lower() for h in h2s])
            h2_counts.append(len(h2s))
            combined_text += content.get("content", "") + "\n\n"
            source_log.append({"url": url, "status": "Success", "h2_count": len(h2s)})
        else:
            source_log.append({"url": url, "status": "Failure", "error": content.get("error")})

    if not combined_text:
        return jsonify({
            "identified_urls": top_urls,
            "processing_report": source_log,
            "error": "Nie udało się pobrać treści z żadnego z top 5 URL-i."
        }), 502

    heading_counts = Counter(all_headings)
    top_10_headings = [heading for heading, count in heading_counts.most_common(10)]

    ngram_data = call_api_with_json(
        NGRAM_API_URL,
        {"text": combined_text, "main_keyword": topic},
        "Ngram API (S1)"
    )

    return jsonify({
        "identified_urls": top_urls,
        "processing_report": source_log,
        "competitive_metrics": {
            "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
            "min_h2": min(h2_counts) if h2_counts else 0,
            "max_h2": max(h2_counts) if h2_counts else 0,
        },
        "top_competitor_headings": top_10_headings,
        "s1_enrichment": ngram_data
    })

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
