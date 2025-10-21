import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ======================================================
# 🌍 Konfiguracja aplikacji Flask + zmienne środowiskowe
# ======================================================
load_dotenv()
app = Flask(__name__)

# ======================================================
# 🔑 Klucze i adresy zewnętrznych serwisów
# ======================================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"

# 🔗 Adres Twojego projektu Render (gpt-ngram-api)
BASE_NGRAM_API_URL = "https://gpt-ngram-api.onrender.com"

NGRAM_API_URL = f"{BASE_NGRAM_API_URL}/api/ngram_entity_analysis"
HEADINGS_API_URL = f"{BASE_NGRAM_API_URL}/api/analyze_headings"
SYNTHESIZE_API_URL = f"{BASE_NGRAM_API_URL}/api/synthesize_topics"

# ======================================================
# 🧩 Funkcje pomocnicze
# ======================================================
def call_api_with_json(url, json_payload, service_name):
    """Uniwersalne wywołanie API z payloadem JSON."""
    try:
        response = requests.post(url, json=json_payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[BŁĄD] {service_name}: {e}")
        return {"error": f"Nie udało się połączyć z {service_name}", "details": str(e)}

def call_serpapi(topic):
    """Wywołuje SerpApi i zwraca dane SERP."""
    params = {
        "api_key": SERPAPI_KEY,
        "q": topic,
        "gl": "pl",
        "hl": "pl",
        "engine": "google"
    }
    try:
        response = requests.get(SERPAPI_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Błąd SerpApi]: {e}")
        return None

def call_langextract(url):
    """Wywołuje API do ekstrakcji treści stron."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

def call_ngram_api(text, topic):
    """Wywołuje API do analizy n-gramów i encji."""
    payload = {"text": text, "main_keyword": topic}
    return call_api_with_json(NGRAM_API_URL, payload, "N-gram API")

def call_headings_api(headings_list):
    """Wywołuje API do analizy nagłówków (opcjonalne)."""
    return call_api_with_json(HEADINGS_API_URL, {"headings": headings_list}, "Headings API")

def call_synthesize_api(text):
    """Wywołuje API do syntezy tematów."""
    return call_api_with_json(SYNTHESIZE_API_URL, {"text": text}, "Synthesize API")

# ======================================================
# 🚀 Endpoint: S1 – pełna analiza tematu
# ======================================================
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    1️⃣ Pobiera SERP z Google
    2️⃣ Ekstrahuje treść z 3–5 URL
    3️⃣ Wysyła tekst do analiz:
         - n-gramy i encje
         - synteza tematów
    4️⃣ Zwraca JSON dla GPT
    """
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Brak parametru 'topic'"}), 400
    if not SERPAPI_KEY:
        return jsonify({"error": "Brak klucza SERPAPI_KEY"}), 500

    # 1️⃣ Pobranie wyników z Google
    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Nie udało się pobrać danych z SerpApi"}), 502

    organic_results = serp_data.get("organic_results", [])
    top_5_urls = [res.get("link") for res in organic_results[:5]]

    # 2️⃣ Ekstrakcja treści
    successful_sources, log = [], []
    combined_text, combined_h2s = "", []

    for url in top_5_urls:
        if len(successful_sources) >= 4:
            break
        content_data = call_langextract(url)
        if content_data and not content_data.get("error") and content_data.get("content"):
            successful_sources.append(content_data)
            log.append({"url": url, "status": "✅ Success"})
            combined_text += content_data["content"] + "\n\n"
            combined_h2s.extend(content_data.get("h2", []))
        else:
            log.append({
                "url": url,
                "status": "❌ Failure",
                "reason": content_data.get("error", "Brak treści")
            })

    # 3️⃣ Uruchom analizy na połączonej treści
    ngrams_result = call_ngram_api(combined_text, topic)
    synthesis_result = call_synthesize_api(combined_text)
    headings_result = call_headings_api(combined_h2s)

    # 4️⃣ Odpowiedź dla GPT
    final_response = {
        "topic": topic,
        "identified_urls": top_5_urls,
        "processing_report": log,
        "successful_sources_count": len(successful_sources),
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "analysis_results": {
            "ngrams": ngrams_result,
            "synthesis": synthesis_result,
            "headings_analysis": headings_result
        }
    }

    return jsonify(final_response)

# ======================================================
# 🧠 Endpoint: S3 – weryfikacja keywordów
# ======================================================
def parse_keyword_string(keyword_data):
    """Parsuje string 'fraza (min-max)' na słownik."""
    if isinstance(keyword_data, dict):
        return keyword_data

    keyword_dict = {}
    pattern = re.compile(r"^\s*(.+?)\s*(?:\((\d+)\s*-\s*(\d+)\))?\s*$")

    for line in keyword_data.splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if match:
            phrase, min_val, max_val = match.groups()
            phrase = phrase.strip()
            min_allowed = int(min_val) if min_val else 1
            max_allowed = int(max_val) if max_val else 5
            keyword_dict[phrase.lower()] = {
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "allowed_range": f"{min_allowed}-{max_allowed}"
            }
    return keyword_dict

@app.route("/api/s3_verify_keywords", methods=["POST"])
def verify_s3_keywords():
    """Sprawdza zgodność tekstu z limitem wystąpień fraz."""
    data = request.get_json()
    text = data.get("text")
    keywords_with_ranges = data.get("keywords_with_ranges")

    if not isinstance(text, str) or not keywords_with_ranges:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges'"}), 400

    try:
        keywords_to_check = parse_keyword_string(keywords_with_ranges)
    except Exception as e:
        return jsonify({"error": f"Błąd parsowania keywords_with_ranges: {e}"}), 400

    text_lower = text.lower()
    keyword_report = {}

    for phrase, ranges in keywords_to_check.items():
        count = text_lower.count(phrase)
        if count < ranges["min_allowed"]:
            status = "UNDER"
        elif count > ranges["max_allowed"]:
            status = "OVER"
        else:
            status = "OK"

        keyword_report[phrase] = {
            "used": count,
            "min_allowed": ranges["min_allowed"],
            "max_allowed": ranges["max_allowed"],
            "allowed_range": ranges["allowed_range"],
            "status": status
        }

    return jsonify({"keyword_report": keyword_report})

# ======================================================
# 🩺 Health Check
# ======================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "✅ master-seo-api działa poprawnie",
        "version": "v2.2.0",
        "connected_services": {
            "serpapi": bool(SERPAPI_KEY),
            "langextract": LANGEXTRACT_API_URL,
            "ngram_api": BASE_NGRAM_API_URL
        }
    }), 200

# ======================================================
# 🚀 Uruchomienie lokalne / produkcyjne
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
