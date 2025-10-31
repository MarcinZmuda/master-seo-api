# master_api.py — Master SEO API (v6.3.0 hybrid JSON mode)
# Obsługa: Firestore, S1 (SerpApi + LangExtract + Ngram API), Brief Base64, Batch Counting, PAA Integration

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
        print("❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON.")
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise ValueError("Brak FIREBASE_CREDS_JSON i pliku serviceAccountKey.json")
    else:
        creds_dict = json.loads(FIREBASE_CREDS_JSON)
        cred = credentials.Certificate(creds_dict)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Połączono z Firestore.")
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
# 🧩 Parser briefu (BASIC / EXTENDED / H2 HEADERS TERMS)
# -------------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    keywords_dict = {}
    headers_list = []

    cleaned_text = os.linesep.join([s.strip() for s in brief_text.splitlines() if s.strip()])

    section_regex = r"((?:BASIC|EXTENDED|H2)\s+TEXT\s+TERMS)\s*:\s*=*\s*([\s\S]*?)(?=\n[A-Z\s]+TEXT\s+TERMS|$)"
    keyword_regex = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$", re.UNICODE)
    keyword_regex_single = re.compile(r"^\s*(.*?)\s*:\s*(\d+)x\s*$", re.UNICODE)

    for match in re.finditer(section_regex, cleaned_text, re.IGNORECASE):
        section_name = match.group(1).upper()
        section_content = match.group(2)

        if section_name.startswith("H2"):
            for line in section_content.splitlines():
                if line.strip():
                    headers_list.append(line.strip())
            continue

        for line in section_content.splitlines():
            line = line.strip()
            if not line:
                continue

            kw_match = keyword_regex.match(line)
            if kw_match:
                keyword = kw_match.group(1).strip()
                min_val = int(kw_match.group(2))
                max_val = int(kw_match.group(3))
            else:
                kw_match_single = keyword_regex_single.match(line)
                if kw_match_single:
                    keyword = kw_match_single.group(1).strip()
                    min_val = max_val = int(kw_match_single.group(2))
                else:
                    continue

            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "remaining_min": min_val,
                "remaining_max": max_val,
                "actual": 0,
                "locked": False,
            }

    return keywords_dict, headers_list


# -------------------------------------------------------------------
# 🔢 Liczenie fraz (v6.3.0 – JSON-safe + Unicode)
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
    text = full_text.lower()
    text = re.sub(r"[^\w\sąćęłńóśźż]", " ", text)
    counts = {}
    for kw in sorted(keywords_dict.keys(), key=len, reverse=True):
        kw_clean = kw.lower().strip()
        if not kw_clean:
            counts[kw] = 0
            continue
        pattern = r"(?<!\w)" + re.escape(kw_clean) + r"(?!\w)"
        matches = re.findall(pattern, text, flags=re.UNICODE)
        counts[kw] = len(matches)
    return counts


# -------------------------------------------------------------------
# 🧠 /api/s1_analysis — analiza konkurencji + n-gramy (pełna integracja)
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
                word_count = len(re.findall(r"\w+", text))
                text_lengths.append(word_count)
                sources_payload.append({"url": url, "content": text})
            else:
                print(f"⚠️ [WARN] Brak treści dla {url}")

        # --- Wywołanie analizy n-gramów z wieloma źródłami ---
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
        "version": "v6.3.0-hybrid-json",
        "message": "Master SEO API działa poprawnie (pełna integracja z n-gram sources)."
    }), 200


# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
