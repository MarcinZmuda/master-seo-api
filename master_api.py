import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# --- Konfiguracja ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"
SYNTHESIZE_API_URL = "https://gpt-ngram-api.onrender.com/api/synthesize_topics"
COMPLIANCE_API_URL = "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"

# --- Helper: API call ---
def call_api_with_json(url, payload, name):
    """Pomocnik do wywoływania innych API z obsługą błędów."""
    try:
        r = requests.post(url, json=payload, timeout=40)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return {"error": f"Nie udało się połączyć z {name}", "details": str(e)}

# --- SerpAPI ---
def call_serpapi(topic):
    """Wywołuje SerpApi dla zadanego tematu."""
    params = {"api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"}
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ Błąd SerpAPI:", e)
        return None

# --- LangExtract ---
def call_langextract(url):
    """Wywołuje LangExtract API, aby pobrać treść ze strony."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

# --- Endpoint: S1 ANALYSIS ("Czarna Skrzynka") ---
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    Endpoint "Czarnej Skrzynki" dla S1:
    1. Pobiera dane SERP.
    2. Pobiera treść (LangExtract).
    3. Analizuje metryki H2.
    4. Wywołuje Ngram API, aby uzyskać encje i n-gramy.
    5. Zwraca połączony raport.
    """
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
    headings_list, h2_count_list = [], []
    
    combined_text_content = ""

    for url in top_5_urls:
        if len(successful_sources) >= 3: # Wystarczą 3 źródła do analizy n-gramów
            break
        
        content = call_langextract(url)
        
        if content and not content.get("error") and content.get("content"):
            current_text = content.get("content", "")
            combined_text_content += current_text + "\n\n"
            
            text_len = len(current_text)
            h2s = content.get("h2", [])
            total_text_length += text_len
            h2_count_list.append(len(h2s))
            headings_list.extend(h2s)
            successful_sources.append(url)
            source_processing_log.append({
                "url": url,
                "status": "Success",
                "length": text_len,
                "h2_count": len(h2s)
            })
        else:
            source_processing_log.append({
                "url": url,
                "status": "Failure",
                "reason": content.get("error", "Brak treści")
            })

    # Obliczanie metryk H2
    avg_h2 = sum(h2_count_list) / len(h2_count_list) if h2_count_list else 0
    min_h2 = min(h2_count_list) if h2_count_list else 0
    max_h2 = max(h2_count_list) if h2_count_list else 0

    # Wywołanie Ngram API
    ngram_payload = {
        "text": combined_text_content,
        "main_keyword": topic
    }
    ngram_data = call_api_with_json(NGRAM_API_URL, ngram_payload, "Ngram API")
    
    # Zwracanie połączonego raportu
    return jsonify({
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "successful_sources_count": len(successful_sources),
        "total_text_length": total_text_length,
        "competitive_metrics": {
            "avg_h2_per_article": round(avg_h2, 1),
            "min_h2": min_h2,
            "max_h2": max_h2,
            "h2_distribution": h2_count_list
        },
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "s1_enrichment": {
            "entities": ngram_data.get("entities"),
            "ngrams": ngram_data.get("ngrams"),
            "error": ngram_data.get("error") 
        }
    })

# --- Endpoint: H2 DISTRIBUTION (Bez zmian) ---
@app.route("/api/h2_distribution", methods=["POST"])
def h2_distribution():
    data = request.get_json()
    h2_counts = data.get("h2_counts")
    if not h2_counts or not isinstance(h2_counts, list):
        return jsonify({"error": "Brak danych h2_counts (lista liczb)"}), 400
    try:
        min_h2 = min(h2_counts)
        max_h2 = max(h2_counts)
        avg_h2 = sum(h2_counts) / len(h2_counts)
        histogram = {count: h2_counts.count(count) for count in set(h2_counts)}
        return jsonify({
            "min_h2": min_h2,
            "max_h2": max_h2,
            "avg_h2": round(avg_h2, 2),
            "distribution": histogram,
            "recommendation": f"Optymalna liczba H2: {round(avg_h2)} ±1"
        })
    except Exception as e:
        return jsonify({"error": f"Błąd przetwarzania: {e}"}), 500

# --- Keyword parsing (Bez zmian) ---
def parse_keyword_string(keyword_data):
    if isinstance(keyword_data, dict):
        return keyword_data
    result = {}
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
            result[phrase] = {
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "allowed_range": f"{min_allowed}-{max_allowed}"
            }
    return result

# --- Endpoint: S3 VERIFY KEYWORDS (Bez zmian) ---
@app.route("/api/s3_verify_keywords", methods=["POST"])
def verify_s3_keywords():
    data = request.get_json()
    text = data.get("text")
    keywords_with_ranges = data.get("keywords_with_ranges")
    if not isinstance(text, str) or not keywords_with_ranges:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges'"}), 400
    try:
        keywords = parse_keyword_string(keywords_with_ranges)
    except Exception as e:
        return jsonify({"error": f"Błąd parsowania: {e}"}), 400
    text_lower = text.lower()
    report = {}
    for phrase, rng in keywords.items():
        count = text_lower.count(phrase)
        status = "OK"
        if count < rng["min_allowed"]:
            status = "UNDER"
        elif count > rng["max_allowed"]:
            status = "OVER"
        report[phrase] = {
            "used": count,
            "min_allowed": rng["min_allowed"],
            "max_allowed": rng["max_allowed"],
            "allowed_range": rng["allowed_range"],
            "status": status
        }
    return jsonify({"keyword_report": report})

# --- Health check (Bez zmian) ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "✅ OK", "version": "3.4-integrated", "message": "master_api działa poprawnie"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 3000), debug=True)

