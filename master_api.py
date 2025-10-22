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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Potrzebne dla endpointu generate_article
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"

# --- Funkcje Pomocnicze ---
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

# --- Endpoint S1 (ZMODYFIKOWANY) ---
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")
    if not topic: return jsonify({"error": "Brak 'topic'"}), 400

    serp_data = call_serpapi(topic)
    if not serp_data: return jsonify({"error": "Błąd pobierania danych z SerpApi"}), 502

    top_urls = [res.get("link") for res in serp_data.get("organic_results", [])[:5]]
    
    source_log, h2_counts, all_headings, combined_text = [], [], [], ""
    successful_sources = 0

    for url in top_urls:
        if successful_sources >= 3: break
        content = call_langextract(url)
        if content and content.get("content"):
            successful_sources += 1
            h2s = content.get("h2", [])
            all_headings.extend(h2s) # Zbieramy H2
            h2_counts.append(len(h2s))
            combined_text += content.get("content", "") + "\n\n"
            source_log.append({"url": url, "status": "Success", "h2_count": len(h2s)})
        else:
            source_log.append({"url": url, "status": "Failure"})

    ngram_data = call_api_with_json(NGRAM_API_URL, {"text": combined_text, "main_keyword": topic}, "Ngram API")

    return jsonify({
        "identified_urls": top_urls,
        "processing_report": source_log,
        "competitive_metrics": {
            "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
            "min_h2": min(h2_counts) if h2_counts else 0,
            "max_h2": max(h2_counts) if h2_counts else 0,
        },
        "all_competitor_headings": all_headings, # <-- NOWE POLE Z LISTĄ H2
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "s1_enrichment": ngram_data
    })

# --- Pozostałe endpointy i kod (bez zmian) ---

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 3000), debug=True)
