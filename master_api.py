import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from collections import Counter
import json

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
# ✅ POPRAWKA 1: Poprawiony URL docelowego API (v4.1)
# -------------------------------------------------------------------
KEYWORD_API_URL = os.getenv(
    "KEYWORD_URL",
    "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"
)
# -------------------------------------------------------------------


# --- Funkcje pomocnicze ---
def call_api_with_json(url, payload, name, timeout=60):
    """Uniwersalna funkcja POST JSON z obsługą błędów."""
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers={"Content-Type": "application/json"})
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
    if not data:
        return jsonify({"error": "Brak danych JSON"}), 400
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
# ✅ POPRAWKA 3: "Inteligentna" funkcja walidacji (WERSJA v4.2-stateful-fixer)
# -------------------------------------------------------------------
@app.route("/api/s3_verify_keywords", methods=["POST"])
def s3_verify_keywords():
    """
    Tłumaczy payload z GPT i przekazuje go do
    docelowego endpointu /api/generate_compliance_report (w trybie STANOWYM).
    
    Posiada logikę "naprawczą":
    - Jeśli klient przyśle brief (string) i tekst (string),
      najpierw zainicjuje stan, a potem go użyje do walidacji.
    """
    try:
        gpt_payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Niepoprawny format JSON", "details": str(e)}), 400

    text = gpt_payload.get("text")
    keyword_state_from_gpt = gpt_payload.get("keyword_state")

    # 1. Walidacja podstawowa (to jest błąd, który powinien zwracać Twój serwer)
    # Jeśli text jest None LUB keyword_state jest None
    if text is None or keyword_state_from_gpt is None:
        print("❌ Błąd S3: Brak 'text' lub 'keyword_state'.")
        return jsonify({"error": "Brak 'text' lub 'keyword_state' w payloadzie"}), 400

    
    # 2. "Inteligentna" logika naprawcza (to, o co prosiłeś)
    
    # Przypadek A: Klient wysłał brief (string) ORAZ tekst do analizy (string).
    # To jest niepoprawne wywołanie, które musimy "naprawić".
    if isinstance(keyword_state_from_gpt, str) and text != "":
        print("🔧 S3 Fixer: Wykryto jednoczesną inicjalizację i walidację. Naprawiam...")
        
        # Krok 1: Inicjalizacja (wysyłamy pusty tekst i brief)
        init_payload = {"text": "", "keyword_state": keyword_state_from_gpt}
        print(f"🔧 S3 Fixer: Krok 1 - Inicjalizacja stanu...")
        init_response = call_api_with_json(KEYWORD_API_URL, init_payload, "Keyword API (Init-Fix)", timeout=240)
        
        if "error" in init_response:
            print("❌ S3 Fixer: Błąd podczas próby inicjalizacji.")
            return jsonify(init_response), 500
            
        new_state_object = init_response.get("new_keyword_state")
        if not new_state_object:
            print("❌ S3 Fixer: Inicjalizacja nie zwróciła 'new_keyword_state'.")
            return jsonify({"error": "Logika naprawcza nie uzyskała stanu z API"}), 500

        # Krok 2: Walidacja (wysyłamy tekst i nowy stan-obiekt)
        print(f"🔧 S3 Fixer: Krok 2 - Walidacja tekstu z nowym stanem...")
        validation_payload = {"text": text, "keyword_state": new_state_object}
        final_response = call_api_with_json(KEYWORD_API_URL, validation_payload, "Keyword API (Validate-Fix)", timeout=240)
        
        return jsonify(final_response), 200

    # Przypadek B: Klient wysłał poprawne dane (inicjalizacja LUB walidacja)
    # Albo (text: "", state: "brief")
    # Albo (text: "...", state: {...})
    else:
        print("✅ S3: Wywołanie poprawne (bezpośrednie przekazanie)...")
        target_payload = {
            "text": text,
            "keyword_state": keyword_state_from_gpt
        }
        
        try:
            r = requests.post(
                KEYWORD_API_URL, # Wskazuje na /api/generate_compliance_report
                json=target_payload,
                headers={"Content-Type": "application/json"},
                timeout=240
            )
            r.raise_for_status()
            
            # 4. Zwróć odpowiedź 1:1 do GPT
            return jsonify(r.json()), 200
        except Exception as e:
            print(f"❌ Błąd S3 (Przekazanie): {e}")
            return jsonify({"error": "Nie udało się połączyć z KEYWORD_API", "details": str(e)}), 500
# -------------------------------------------------------------------


# --- Health Check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v4.2-stateful-fixer",
        "message": "Master SEO API działa poprawnie (połączony z /api/generate_compliance_report i logiką 'naprawczą')"
    }), 200


# --- Uruchomienie ---
if __name__ == "__main__":
    # Poprawione wcięcie
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

