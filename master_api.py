import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ======================================================
# 🌍 Konfiguracja aplikacji Flask
# ======================================================
load_dotenv()
app = Flask(__name__)

# ======================================================
# 🔑 Konfiguracja kluczy i adresów
# ======================================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"

# Adres drugiego API (Render)
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api"

# ======================================================
# 🧩 Funkcje pomocnicze
# ======================================================
def call_api_with_json(url, json_payload, service_name):
    """Uniwersalna funkcja do wykonywania POST-ów z JSON-em."""
    try:
        response = requests.post(url, json=json_payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[❌] Błąd podczas wywołania {service_name}: {e}")
        return {"error": f"Nie udało się połączyć z {service_name}", "details": str(e)}

def call_serpapi(topic):
    """Pobiera dane z Google SERP przez SerpAPI."""
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
        print(f"[❌] SerpAPI error: {e}")
        return None

def call_langextract(url):
    """Wywołuje LangExtract API."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

# ======================================================
# 🧠 Etap S1 — Analiza SERP + ekstrakcja treści
# ======================================================
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
    combined_text, combined_h2s = "", []

    for url in top_5_urls:
        if len(successful_sources) >= 4:
            break
        content_data = call_langextract(url)
        if content_data and not content_data.get("error") and content_data.get("content"):
            successful_sources.append(content_data)
            source_processing_log.append({"url": url, "status": "✅ Success"})
            combined_text += content_data.get("content", "") + "\n\n"
            combined_h2s.extend(content_data.get("h2", []))
        else:
            source_processing_log.append({
                "url": url,
                "status": "❌ Failure",
                "reason": content_data.get("error", "Brak treści")
            })

    # Wynik końcowy
    return jsonify({
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "successful_sources_count": len(successful_sources),
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "combined_text": combined_text,
        "headings": combined_h2s
    })

# ======================================================
# 🧩 PROXY — przekierowanie do gpt-ngram-api
# ======================================================
@app.route("/api/ngram_entity_analysis", methods=["POST"])
def proxy_ngram_analysis():
    """Proxy do analizy n-gramów i encji."""
    payload = request.get_json()
    return jsonify(call_api_with_json(f"{NGRAM_API_URL}/ngram_entity_analysis", payload, "N-gram API"))

@app.route("/api/synthesize_topics", methods=["POST"])
def proxy_synthesize_topics():
    """Proxy do syntezy tematów."""
    payload = request.get_json()
    return jsonify(call_api_with_json(f"{NGRAM_API_URL}/synthesize_topics", payload, "Synthesize Topics API"))

@app.route("/api/generate_compliance_report", methods=["POST"])
def proxy_generate_compliance_report():
    """Proxy do końcowej walidacji SEO."""
    payload = request.get_json()
    return jsonify(call_api_with_json(f"{NGRAM_API_URL}/generate_compliance_report", payload, "Compliance Report API"))

# ======================================================
# 🧩 Etap S3 — Weryfikacja fraz z zakresami
# ======================================================
def parse_keyword_string(keyword_data):
    """Parsuje format 'fraza (min-max)' na strukturę JSON."""
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
    """Sprawdza użycie słów kluczowych i zakresów."""
    data = request.get_json()
    text = data.get("text", "").lower()
    keywords_input = data.get("keywords_with_ranges")

    if not text or not keywords_input:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges'"}), 400

    keywords = parse_keyword_string(keywords_input)
    report = {}

    for phrase, ranges in keywords.items():
        count = text.count(phrase)
        status = "OK"
        if count < ranges["min_allowed"]:
            status = "UNDER"
        elif count > ranges["max_allowed"]:
            status = "OVER"
        report[phrase] = {
            "used": count,
            "min_allowed": ranges["min_allowed"],
            "max_allowed": ranges["max_allowed"],
            "allowed_range": ranges["allowed_range"],
            "status": status
        }

    return jsonify({"keyword_report": report})

# ======================================================
# 🩺 Health check
# ======================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "✅ master-seo-api działa poprawnie",
        "version": "v3.0.0",
        "proxy_connected": True,
        "ngram_proxy_url": NGRAM_API_URL
    })

# ======================================================
# 🚀 Start
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
