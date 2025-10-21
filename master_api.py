import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- Konfiguracja i inicjalizacja ---
load_dotenv()
app = Flask(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"
SYNTHESIZE_API_URL = "https://gpt-ngram-api.onrender.com/api/synthesize_topics"
COMPLIANCE_API_URL = "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"

# --- Funkcje pomocnicze ---

def call_api_with_json(url, json_payload, service_name):
    """Bezpieczne wywołanie API z JSON-em."""
    try:
        response = requests.post(url, json=json_payload, timeout=40)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Błąd w {service_name}: {e}")
        return {"error": f"Nie udało się połączyć z {service_name}", "details": str(e)}

def call_serpapi(topic):
    """Pobiera dane SERP z SerpAPI."""
    params = {
        "api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"
    }
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ Błąd SerpAPI:", e)
        return None

def call_langextract(url):
    """Wywołuje API do ekstrakcji treści."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

# --- Etap S1: analiza SERP + ekstrakcja nagłówków ---
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Brak parametru 'topic'"}), 400
    if not SERPAPI_KEY:
        return jsonify({"error": "Brak klucza SERPAPI_KEY"}), 500

    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Nie udało się pobrać danych z SerpApi"}), 502

    organic_results = serp_data.get("organic_results", [])
    top_5_urls = [res.get("link") for res in organic_results[:5]]

    successful_sources, source_processing_log = [], []
    total_text_length = 0
    headings_list = []

    for url in top_5_urls:
        if len(successful_sources) >= 4:
            break
        content_data = call_langextract(url)
        if content_data and not content_data.get("error") and content_data.get("content"):
            text_len = len(content_data.get("content", ""))
            total_text_length += text_len
            headings_list.extend(content_data.get("h2", []))
            successful_sources.append(url)
            source_processing_log.append({"url": url, "status": "Success", "length": text_len})
        else:
            source_processing_log.append({
                "url": url,
                "status": "Failure",
                "reason": content_data.get("error", "Brak treści")
            })

    headings_result = call_api_with_json(
        SYNTHESIZE_API_URL, {"headings": headings_list}, "Headings API"
    )

    return jsonify({
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "successful_sources_count": len(successful_sources),
        "total_text_length": total_text_length,
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "analysis_results": {
            "headings_analysis": headings_result
        }
    })

# --- Etap S3: weryfikacja słów kluczowych ---
def parse_keyword_string(keyword_data):
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
            phrase = phrase.strip().lower()
            min_allowed = int(min_val) if min_val else 1
            max_allowed = int(max_val) if max_val else 5
            keyword_dict[phrase] = {
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "allowed_range": f"{min_allowed}-{max_allowed}"
            }
    return keyword_dict

@app.route("/api/s3_verify_keywords", methods=["POST"])
def verify_s3_keywords():
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
        status = "OK"
        if count < ranges["min_allowed"]:
            status = "UNDER"
        elif count > ranges["max_allowed"]:
            status = "OVER"
        keyword_report[phrase] = {
            "used": count,
            "min_allowed": ranges["min_allowed"],
            "max_allowed": ranges["max_allowed"],
            "allowed_range": ranges["allowed_range"],
            "status": status
        }

    return jsonify({"keyword_report": keyword_report})

# --- Health check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "✅ OK", "version": "3.1-Lite", "message": "master-seo-api działa poprawnie"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 3000), debug=True)
