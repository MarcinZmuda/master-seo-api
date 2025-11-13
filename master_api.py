# ================================================================
# master_api.py — Master SEO API (v7.2.1-fixed + Firestore Tracker)
# ================================================================

import os
import re
import json
import base64
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# -------------------------------------------------------------------
# 🔧 Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
    if not FIREBASE_CREDS_JSON:
        print("⚠️ Brak zmiennej FIREBASE_CREDS_JSON — próba użycia pliku lokalnego.")
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise ValueError("❌ Brak FIREBASE_CREDS_JSON i pliku serviceAccountKey.json.")
    else:
        creds_dict = json.loads(FIREBASE_CREDS_JSON)
        cred = credentials.Certificate(creds_dict)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firestore połączony poprawnie.")
except Exception as e:
    print(f"❌ Błąd inicjalizacji Firebase: {e}")
    db = None

# -------------------------------------------------------------------
# 🌐 API Zewnętrzne (S1)
# -------------------------------------------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"


def call_api_with_json(url, payload, name):
    try:
        r = requests.post(url, json=payload, timeout=120)
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


# -------------------------------------------------------------------
# 🧠 /api/s1_analysis — analiza konkurencji + n-gramy
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    try:
        data = request.get_json()
        if not data or "topic" not in data:
            return jsonify({"error": "Brak 'topic'"}), 400

        topic = data["topic"]
        serp_data = call_serpapi(topic)
        if not serp_data:
            return jsonify({"error": "Brak danych z SerpAPI"}), 502

        ai_overview_status = serp_data.get("ai_overview", {}).get("status", "not_available")
        people_also_ask = [q.get("question") for q in serp_data.get("related_questions", []) if q.get("question")]
        autocomplete_suggestions = [r.get("query") for r in serp_data.get("related_searches", []) if r.get("query")]
        top_urls = [r.get("link") for r in serp_data.get("organic_results", [])[:7]]

        print(f"🔍 [DEBUG] Analiza {len(top_urls)} wyników SERP dla: {topic}")

        sources_payload, h2_counts, text_lengths, all_headings = [], [], [], []

        for url in top_urls[:5]:
            content = call_langextract(url)
            if content and content.get("content"):
                text = content.get("content", "")
                h2s = content.get("h2", [])
                h2_counts.append(len(h2s))
                all_headings.extend([h.strip().lower() for h in h2s])
                word_count = len(re.findall(r'\w+', text))
                text_lengths.append(word_count)
                sources_payload.append({"url": url, "content": text})
            else:
                print(f"⚠️ Brak treści dla {url}")

        ngram_data = call_api_with_json(
            NGRAM_API_URL,
            {"sources": sources_payload, "main_keyword": topic, "serp_context": {
                "people_also_ask": people_also_ask,
                "related_searches": autocomplete_suggestions
            }},
            "Ngram API"
        )

        heading_counts = Counter(all_headings)
        top_headings = [h for h, _ in heading_counts.most_common(10)]

        competitive_metrics = {
            "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
            "min_h2": min(h2_counts) if h2_counts else 0,
            "max_h2": max(h2_counts) if h2_counts else 0,
            "avg_text_length_words": round(sum(text_lengths) / len(text_lengths)) if text_lengths else 0,
            "min_text_length_words": min(text_lengths) if text_lengths else 0,
            "max_text_length_words": max(text_lengths) if text_lengths else 0,
        }

        return jsonify({
            "identified_urls": top_urls,
            "competitive_metrics": competitive_metrics,
            "ai_overview_status": ai_overview_status,
            "people_also_ask": people_also_ask,
            "autocomplete_suggestions": autocomplete_suggestions,
            "top_competitor_headings": top_headings,
            "s1_enrichment": ngram_data
        }), 200

    except Exception as e:
        print(f"❌ Błąd /api/s1_analysis: {e}")
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------
# ❤️ Health Check
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v7.2.1-fixed + tracker",
        "message": "Master SEO API działa poprawnie (Render + Firestore OK)."
    }), 200


# -------------------------------------------------------------------
# 🔗 Integracja: Project Management Layer + Firestore Tracker
# -------------------------------------------------------------------
try:
    from project_routes import register_project_routes
    register_project_routes(app, db)
    print("✅ Zarejestrowano project_routes (Blueprint działa).")
except Exception as e:
    print(f"❌ Nie udało się załadować project_routes: {e}")

try:
    from firestore_tracker_routes import register_tracker_routes
    register_tracker_routes(app, db)
    print("✅ Zarejestrowano firestore_tracker_routes (Tracker Layer działa).")
except Exception as e:
    print(f"❌ Nie udało się załadować firestore_tracker_routes: {e}")

# -------------------------------------------------------------------
# 🚀 Uruchomienie (Render-compatible)
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Render wymaga portu z ENV
    print(f"🌐 Uruchamiam Master SEO API (port={port}, Firestore={'OK' if db else 'BRAK'})")
    app.run(host="0.0.0.0", port=port)
