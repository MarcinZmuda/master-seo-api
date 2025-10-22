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
GPT_NGRAM_INDEX_URL = "https://gpt-ngram-api.onrender.com/api/index.py"  # 🔄 integracja z Twoim modułem
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

# --- Endpoint: S1 ANALYSIS (Z AKTYWNYM GPT-NGRAM INDEX) ---
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    Etap S1 – pełna integracja:
    1. Pobiera dane z SERP.
    2. Pobiera treść (LangExtract API).
    3. Oblicza metryki H2.
    4. Wysyła tekst do GPT-NGRAM-API (/api/index.py) w celu scoringu.
    5. Zwraca ujednolicony raport S1.
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
    h2_count_list, headings_list = [], []
    combined_text_content = ""

    for url in top_5_urls:
        if len(successful_sources) >= 3:
            break
        content = call_langextract(url)
        if content and not content.get("error") and content.get("content"):
            current_text = content.get("content", "")
            combined_text_content += current_text + "\n\n"
            h2s = content.get("h2", [])
            h2_count_list.append(len(h2s))
            headings_list.extend(h2s)
            successful_sources.append(url)
            source_processing_log.append({
                "url": url,
                "status": "Success",
                "h2_count": len(h2s),
                "length": len(current_text)
            })
        else:
            source_processing_log.append({
                "url": url,
                "status": "Failure",
                "reason": content.get("error", "Brak treści")
            })

    # --- Obliczanie metryk H2 ---
    avg_h2 = sum(h2_count_list) / len(h2_count_list) if h2_count_list else 0
    min_h2 = min(h2_count_list) if h2_count_list else 0
    max_h2 = max(h2_count_list) if h2_count_list else 0

    # --- Wywołanie Twojego GPT-NGRAM-API (/api/index.py) ---
    gpt_ngram_payload = {
        "text": combined_text_content,
        "topic": topic
    }
    gpt_ngram_data = call_api_with_json(GPT_NGRAM_INDEX_URL, gpt_ngram_payload, "GPT-NGRAM INDEX API")

    # --- Log bezpieczeństwa i scoring ---
    if not gpt_ngram_data or gpt_ngram_data.get("error"):
        print("⚠️ GPT-NGRAM-API zwróciło błąd lub pustą odpowiedź.")
        gpt_ngram_data = {"ngrams": [], "entities": [], "error": gpt_ngram_data.get("error", "brak danych")}
    else:
        print("\n🧠 Wynik z GPT-NGRAM-API:")
        if "score" in gpt_ngram_data:
            print(f"➡️ NGRAM SCORE: {gpt_ngram_data['score']}")
        if gpt_ngram_data.get("ngrams"):
            print(f"🔸 Top n-gramy ({len(gpt_ngram_data['ngrams'])}):")
            for ng in gpt_ngram_data["ngrams"][:5]:
                print(f"   - {ng.get('text', '---')} → {ng.get('score', 'N/A')}")

    # --- Zwracanie raportu ---
    return jsonify({
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "successful_sources_count": len(successful_sources),
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
            "entities": gpt_ngram_data.get("entities"),
            "ngrams": gpt_ngram_data.get("ngrams"),
            "error": gpt_ngram_data.get("error"),
            "ngram_score": gpt_ngram_data.get("score")
        }
    })


# --- Health check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "✅ OK", "version": "3.5-ngram", "message": "Master SEO API + GPT-NGRAM działa poprawnie"}), 200


# --- Sekcja testowa ---
if __name__ == "__main__":
    test_topic = "rozwód z orzeczeniem o winie"
    print(f"\n🚀 TEST LOKALNY: {test_topic}")
    
    try:
        payload = {"topic": test_topic}
        response = requests.post("http://127.0.0.1:3000/api/s1_analysis", json=payload, timeout=180)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Raport testowy:")
            print(f"   URL-e: {len(data.get('identified_urls', []))}")
            print(f"   Średnia H2: {data['competitive_metrics']['avg_h2_per_article']}")
            print(f"   N-gram score: {data['s1_enrichment'].get('ngram_score')}")
            print(f"   Top n-gramy:")
            for ng in (data['s1_enrichment'].get('ngrams') or [])[:5]:
                print(f"     - {ng.get('text', '')} ({ng.get('score', 'brak')})")
        else:
            print(f"❌ Błąd testu: {response.status_code} – {response.text}")
    except Exception as e:
        print(f"❌ Test lokalny nie powiódł się: {e}")

    # Uruchom serwer
    app.run(host="0.0.0.0", port=os.getenv("PORT", 3000), debug=True)
