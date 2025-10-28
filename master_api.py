import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from collections import Counter

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# --- Konfiguracja ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"

# -------------------------------------------------------------------
# ✅ POPRAWKA 1: Poprawiony URL docelowego API
# -------------------------------------------------------------------
# Wskazujemy na JEDYNY poprawny endpoint, który teraz używa
# Twojego idealnego, połączonego walidatora.
KEYWORD_API_URL = os.getenv(
    "KEYWORD_URL",
    "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"
)
# -------------------------------------------------------------------


# --- Funkcje pomocnicze ---
def call_api_with_json(url, payload, name):
    """Uniwersalna funkcja POST JSON z obsługą błędów."""
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd API {name}: {e}")
        return {"error": f"Błąd połączenia z {name}", "details": str(e)}

def call_serpapi(topic):
    """Pobiera wyniki z SerpAPI."""
    params = {"api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"}
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd SerpAPI: {e}")
        return None

def call_langextract(url):
    """Pobiera tekst z URL przy użyciu LangExtract."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract")


# --- Endpoint S1: Analiza konkurencji ---
# (Bez zmian, działa poprawnie)
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")
    if not topic:
        return jsonify({"error": "Brak 'topic'"}), 400

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
            source_log.append({"url": url, "status": "Failure"})

    if all_headings:
        heading_counts = Counter(all_headings)
        top_10_headings = [heading for heading, count in heading_counts.most_common(10)]
    else:
        top_10_headings = []

    ngram_data = call_api_with_json(
        NGRAM_API_URL,
        {"text": combined_text, "main_keyword": topic},
        "Ngram API"
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
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "s1_enrichment": ngram_data
    })


# -------------------------------------------------------------------
# ✅ POPRAWKA 2: Poprawiona funkcja walidacji (tłumaczenie payloadu)
# -------------------------------------------------------------------
@app.route("/api/s3_verify_keywords", methods=["POST"])
def s3_verify_keywords():
    """
    Tłumaczy payload z GPT i przekazuje go do
    docelowego endpointu /api/generate_compliance_report.
    """
    # 1. Odbierz payload z GPT (zgodny z promptem v3.8)
    # Spodziewany format: {"text": "...", "keywords_with_ranges": "..."}
    gpt_payload = request.get_json(force=True)
    text = gpt_payload.get("text")
    keyword_string = gpt_payload.get("keywords_with_ranges")

    if not text or not keyword_string:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges' w payloadzie"}), 400

    # 2. Przygotuj NOWY payload dla gpt-ngram-api
    # Wymagany format: {"text": "...", "keyword_list_string": "..."}
    # (Zakładając, że zaktualizowałeś swój index.py, by tego oczekiwał)
    target_payload = {
        "text": text,
        "keyword_list_string": keyword_string
    }

    try:
        # 3. Wywołaj docelowe API (zdefiniowane w KEYWORD_API_URL)
        r = requests.post(
            KEYWORD_API_URL,
            json=target_payload, # <-- Wysyła poprawnie sformatowany payload
            headers={"Content-Type": "application/json"},
            timeout=240
        )
        r.raise_for_status()
        return jsonify(r.json()), 200
    except Exception as e:
        print(f"❌ Błąd S3 Verify Keywords: {e}")
        return jsonify({"error": "Nie udało się połączyć z KEYWORD_API", "details": str(e)}), 500
# -------------------------------------------------------------------


# --- Health Check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        # ✅ POPRAWKA 3: Zaktualizowana wersja i komunikat
        "version": "v3.8-fixed",
        "message": "Master SEO API działa poprawnie (połączony z /api/generate_compliance_report)"
    }), 200


# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
