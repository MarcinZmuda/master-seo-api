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

KEYWORD_API_URL = os.getenv(
    "KEYWORD_URL",
    "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"
)
# -------------------------------------------------------------------


# --- Funkcje pomocnicze ---
def call_api_with_json(url, payload, name):
    """Uniwersalna funkcja POST JSON z obsługą błędów."""
    try:
        # Zwiększamy globalny timeout, bo nowa logika może wymagać 2 wywołań
        r = requests.post(url, json=payload, timeout=300, headers={"Content-Type": "application/json"})
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
# ✅ NOWA LOGIKA: Inteligentny pośrednik v4.2 (State-Fixer)
# -------------------------------------------------------------------
@app.route("/api/s3_verify_keywords", methods=["POST"])
def s3_verify_keywords():
    """
    Inteligentny pośrednik (v4.2):
    1. Odbiera dane od klienta (GPT).
    2. Sprawdza, czy 'keyword_state' to obiekt JSON (poprawnie)
       czy string z briefem (błędnie).
    3. Jeśli błędnie (string), automatycznie naprawia stan,
       wykonując dodatkowe wywołanie inicjalizujące.
    4. Przekazuje poprawny ładunek do docelowego API.
    """
    try:
        gpt_payload = request.get_json(force=True)
    except Exception as e:
        print(f"❌ Błąd S3: Nie można sparsować JSON. Treść: {request.data}")
        return jsonify({"error": "Błędny format JSON", "details": str(e)}), 400

    text_to_validate = gpt_payload.get("text")
    keyword_state_from_gpt = gpt_payload.get("keyword_state")

    if text_to_validate is None or keyword_state_from_gpt is None:
        return jsonify({"error": "Brak 'text' lub 'keyword_state' w payloadzie"}), 400

    try:
        # --- Inteligentna logika naprawcza ---
        
        final_payload = None

        # Przypadek A: Poprawny stan (klient wysłał obiekt JSON)
        if isinstance(keyword_state_from_gpt, dict):
            print("✅ S3 Info: Otrzymano poprawny obiekt stanu. Przekazuję dalej.")
            final_payload = {
                "text": text_to_validate,
                "keyword_state": keyword_state_from_gpt
            }

        # Przypadek B: Błędny stan (klient wysłał string z briefem)
        elif isinstance(keyword_state_from_gpt, str):
            print("⚠️ S3 Info: Otrzymano string (brief) zamiast obiektu stanu. Rozpoczynam naprawę...")
            
            # 1. Wykonaj wywołanie "rozgrzewkowe" (inicjalizujące), aby pobrać stan
            print("...Krok 1: Wywołanie inicjalizujące (text: \"\")...")
            pre_payload = {
                "text": "", # Pusty tekst, aby zasygnalizować inicjalizację
                "keyword_state": keyword_state_from_gpt # Brief w stringu
            }
            
            initial_state_data = call_api_with_json(
                KEYWORD_API_URL, 
                pre_payload, 
                "Keyword API (Init-Fix)"
            )
            
            if "error" in initial_state_data:
                print("❌ S3 Błąd: Nie udało się naprawić stanu. Błąd inicjalizacji.")
                return jsonify(initial_state_data), 500

            new_state_object = initial_state_data.get("new_keyword_state")

            if not new_state_object or not isinstance(new_state_object, dict):
                print("❌ S3 Błąd: Inicjalizacja nie zwróciła obiektu 'new_keyword_state'.")
                return jsonify({"error": "Błąd logiki naprawczej: API inicjalizujące nie zwróciło obiektu stanu."}), 500
            
            print("...Krok 2: Stan naprawiony. Wykonuję właściwe wywołanie walidacyjne...")
            
            # 2. Przygotuj właściwy payload z tekstem usera i naprawionym stanem
            final_payload = {
                "text": text_to_validate,      # Oryginalny tekst do analizy
                "keyword_state": new_state_object # Naprawiony obiekt stanu
            }
        
        else:
            return jsonify({"error": "Niepoprawny typ danych dla 'keyword_state'. Oczekiwano obiektu lub stringa."}), 400

        # --- Wykonanie właściwego wywołania ---
        
        if not final_payload:
             return jsonify({"error": "Wewnętrzny błąd serwera: Nie udało się utworzyć final_payload."}), 500

        print(f"✅ S3 Info: Wysyłanie do {KEYWORD_API_URL}...")
        
        # Wywołujemy docelowe API (już na pewno z poprawnym payloadem)
        response_data = call_api_with_json(
            KEYWORD_API_URL, 
            final_payload, 
            "Keyword API (Main)"
        )

        if "error" in response_data:
            return jsonify(response_data), 502 # 502 Bad Gateway (problem z API docelowym)

        # 4. Zwróć odpowiedź 1:1 do GPT
        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Błąd S3 Verify Keywords (logika wewnętrzna): {e}")
        return jsonify({"error": "Wewnętrzny błąd serwera w S3", "details": str(e)}), 500
# -------------------------------------------------------------------


# --- Health Check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v4.2-stateful-fixer",
        "message": "Master SEO API działa poprawnie (z inteligentną logiką naprawczą)"
    }), 200


# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
