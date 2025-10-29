import os
import re
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter # ✅ DODANO IMPORT

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# -------------------------------------------------------------------
# ✅ KROK 1: Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
# WAŻNE: W Render.com musisz utworzyć zmienną środowiskową o nazwie
# "FIREBASE_CREDS_JSON" i wkleić do niej CAŁĄ ZAWARTOŚĆ
# pliku serviceAccountKey.json, który pobierzesz z Firebase.
# 
# UWAGA: Musisz włączyć "Firestore Database" w swoim projekcie Firebase.
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
# ... (bez zmian) ...
    db = firestore.client()
    print("✅ Pomyślnie połączono z Firestore.")
except Exception as e:
    print(f"❌ KRYTYCZNY BŁĄD: Nie można zainicjować Firebase: {e}")
    db = None

# --- Konfiguracja SerpAPI (dla S1, jeśli nadal potrzebne) ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
# ✅ DODANO NGRAM API POTRZEBNE DLA S1
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"


# -------------------------------------------------------------------
# ✅ DODANO FUNKCJE POMOCNICZE DLA S1
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# ✅ KROK 2: Logika Parsowania Briefu
# -------------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
# ... (bez zmian) ...
            
    return keywords_dict

# -------------------------------------------------------------------
# ✅ KROK 3: Logika Hierarchicznego Liczenia (Kluczowy element)
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
# ... (bez zmian) ...

    return counts

# -------------------------------------------------------------------
# ✅ KROK 4: Nowe Endpointy (Architektura v5)
# -------------------------------------------------------------------

@app.route("/api/project/create", methods=["POST"])
def create_project():
# ... (bez zmian) ...
        return jsonify({
            "status": "Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state)
        }), 201
# ... (bez zmian) ...


@app.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
# ... (bez zmian) ...
        
        # Zwracamy raport tekstowy dla GPT
        return jsonify(report_for_gpt), 200

    except Exception as e:
# ... (bez zmian) ...
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
# ... (bez zmian) ...
        "message": "Master SEO API (Firestore Edition) działa poprawnie."
    }), 200

# -------------------------------------------------------------------
# ✅ PRZYWRÓCONO PEŁNY KOD ENDPOINTU S1
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    Wykonuje pełną analizę konkurencji (S1), niezależną od logiki S3.
    """
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
        # Zabezpieczenie, gdy żaden URL nie zwrócił treści
        print("❌ Błąd S1: Nie udało się pobrać treści z żadnego URL-a.")
        return jsonify({
            "identified_urls": top_urls,
            "processing_report": source_log,
            "error": "Nie udało się pobrać treści z żadnego z top 5 URL-i.",
            "serp_features": {
                "ai_overview": serp_data.get("ai_overview"),
                "people_also_ask": serp_data.get("related_questions"),
                "featured_snippets": serp_data.get("answer_box")
            }
        }), 502

    if all_headings:
        heading_counts = Counter(all_headings)
        top_10_headings = [heading for heading, count in heading_counts.most_common(10)]
    else:
        top_10_headings = []

    # Wywołanie API Ngram do wzbogacenia S1
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
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "s1_enrichment": ngram_data
    })
# -------------------------------------------------------------------

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

