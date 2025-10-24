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
KEYWORD_API_URL = os.getenv("KEYWORD_URL", "https://seo-keyword-api.onrender.com/api/verify_keywords")

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

# --- Endpoint S1 ---
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

    ngram_data = call_api_with_json(NGRAM_API_URL, {"text": combined_text, "main_keyword": topic}, "Ngram API")

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

# --- Endpoint: /api/update_keywords ---
@app.route("/api/update_keywords", methods=["POST"])
def update_keywords():
    data = request.get_json()
    text = data.get("text", "")
    keywords_state = data.get("keywords_state", {})

    if not text or not keywords_state:
        return jsonify({"error": "Missing text or keywords_state"}), 400

    limits_raw = keywords_state.get("keywords_with_limits", "")
    limits = {}
    for line in limits_raw.strip().split("\n"):
        match = re.match(r"^(.*?):\s*(\d+)-(\d+)x", line.strip())
        if match:
            kw, min_l, max_l = match.groups()
            limits[kw.strip()] = {"min": int(min_l), "max": int(max_l), "used": 0}

    text_lower = text.lower()
    for kw in limits.keys():
        count = len(re.findall(rf"\b{re.escape(kw.lower())}\b", text_lower))
        limits[kw]["used"] = count

    usage_summary = {}
    validation_report = []
    updated_lines = []

    for kw, vals in limits.items():
        used = vals["used"]
        remaining = max(0, vals["max"] - used)
        if used < vals["min"]:
            status = "❕ Niedobór"
        elif used > vals["max"]:
            status = "❌ Przekroczono"
        else:
            status = "✅ OK"

        usage_summary[kw] = f"{used}/{vals['max']}x ({status})"
        validation_report.append({"keyword": kw, "used": used, "min": vals["min"], "max": vals["max"], "status": status})
        updated_lines.append(f"{kw}: {vals['min']}-{vals['max']}x (pozostało: {remaining})")

    updated_keywords_state = {
        **keywords_state,
        "keywords_with_limits": "\n".join(updated_lines),
        "usage": usage_summary
    }

    return jsonify({
        "updated_keywords_state": updated_keywords_state,
        "summary": usage_summary,
        "validation_report": validation_report
    })

# --- Endpoint: /api/s3_verify_keywords ---
@app.route("/api/s3_verify_keywords", methods=["POST"])
def s3_verify_keywords():
    payload = request.get_json(force=True)
    try:
        r = requests.post(KEYWORD_API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=240)
        r.raise_for_status()
        return jsonify(r.json()), 200
    except Exception as e:
        print(f"❌ Błąd S3 Verify Keywords: {e}")
        return jsonify({"error": "Nie udało się połączyć z KEYWORD_API", "details": str(e)}), 500

# --- Health Check ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v3.6", "message": "Master SEO API działa poprawnie"}), 200

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
