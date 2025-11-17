# ================================================================
# s1_analysis_routes.py — Turbo S1 (SERP + LangExtract + N-gramy + spaCy NER)
# v7.3.2-ultra
# ================================================================

import os
import re
import json
import requests
from collections import Counter
from flask import Blueprint, request, jsonify
import spacy

# ------------------------------------------------
# 🌐 Konfiguracja z ENV (Render)
# ------------------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

LANGEXTRACT_API_URL = os.getenv(
    "LANGEXTRACT_API_URL",
    "https://langextract-api.onrender.com/extract"
)

NGRAM_API_URL = os.getenv(
    "NGRAM_API_URL",
    "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"
)

# ------------------------------------------------
# 🧠 spaCy ładowane RAZ w całym procesie
# ------------------------------------------------

nlp = spacy.load("pl_core_news_sm")

# Blueprint
s1_routes = Blueprint("s1_routes", __name__)


# ------------------------------------------------
# 🔧 Helper: POST z JSON-em z obsługą błędów
# ------------------------------------------------
def call_api_with_json(url, payload, name, timeout=180):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd API {name}: {e}")
        return {"error": str(e), "location": name}


# ------------------------------------------------
# 🔧 SerpAPI
# ------------------------------------------------
def call_serpapi(topic: str):
    if not SERPAPI_KEY:
        print("❌ Brak SERPAPI_KEY w ENV.")
        return None

    params = {
        "api_key": SERPAPI_KEY,
        "q": topic,
        "gl": "pl",
        "hl": "pl",
        "engine": "google",
    }

    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=35)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd SerpAPI: {e}")
        return None


# ------------------------------------------------
# 🔧 LangExtract — pełny tekst + H2 z URL
# ------------------------------------------------
def call_langextract(url: str):
    payload = {"url": url}
    return call_api_with_json(LANGEXTRACT_API_URL, payload, "LangExtract", timeout=120)


# ------------------------------------------------
# 🔧 N-gram API
# ------------------------------------------------
def call_ngram_api(sources_payload, topic, serp_context):
    if not NGRAM_API_URL:
        print("❌ Brak NGRAM_API_URL w ENV.")
        return {"error": "Missing NGRAM_API_URL"}

    payload = {
        "sources": sources_payload,
        "main_keyword": topic,
        "serp_context": serp_context,
    }

    return call_api_with_json(NGRAM_API_URL, payload, "GPT-Ngram", timeout=240)


# ------------------------------------------------
# 🔧 spaCy entity extractor (NER + noun chunks)
# ------------------------------------------------
def extract_spacy_entities(text: str):
    """
    Zwraca:
      - named entities
      - noun chunks
    """
    doc = nlp(text)

    entities = []
    noun_chunks = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)

    return {
        "entities": entities,
        "noun_chunks": noun_chunks
    }


# ------------------------------------------------
# 🔧 Statystyki konkurencji (H2 + długość tekstu)
# ------------------------------------------------
def build_competitive_metrics(h2_counts, text_lengths):
    if not h2_counts:
        h2_counts = []
    if not text_lengths:
        text_lengths = []

    competitive_metrics = {
        "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
        "min_h2": min(h2_counts) if h2_counts else 0,
        "max_h2": max(h2_counts) if h2_counts else 0,
        "avg_text_length_words": round(sum(text_lengths) / len(text_lengths)) if text_lengths else 0,
        "min_text_length_words": min(text_lengths) if text_lengths else 0,
        "max_text_length_words": max(text_lengths) if text_lengths else 0,
    }
    return competitive_metrics


def extract_top_headings(all_headings, limit=10):
    if not all_headings:
        return []

    heading_counts = Counter(all_headings)
    top_headings = [h for h, _ in heading_counts.most_common(limit)]
    return top_headings


# ------------------------------------------------
# 🧠 /api/s1_analysis — główny endpoint Turbo S1
# ------------------------------------------------
@s1_routes.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    FULL FLOW:
      1. SerpAPI → organic_results, PAA, related_searches, AI Overview status
      2. LangExtract (top 5 URL) → content + H2
      3. spaCy → NER + noun chunks
      4. Metryki konkurencji
      5. GPT-Ngram API
      6. Finalny raport S1 (100% kompatybilny z Twoim GPT)
    """
    try:
        data = request.get_json() or {}
        topic = data.get("topic")

        if not topic:
            return jsonify({"error": "Brak 'topic' w body requestu"}), 400

        # 1. SerpAPI
        serp_data = call_serpapi(topic)
        if not serp_data:
            return jsonify({"error": "Brak danych z SerpAPI"}), 502

        ai_overview_status = (
            serp_data.get("ai_overview", {}).get("status", "not_available")
        )

        people_also_ask = [
            q.get("question")
            for q in serp_data.get("related_questions", [])
            if q.get("question")
        ]

        autocomplete_suggestions = [
            r.get("query")
            for r in serp_data.get("related_searches", [])
            if r.get("query")
        ]

        organic_results = serp_data.get("organic_results", []) or []
        top_urls = [r.get("link") for r in organic_results[:7] if r.get("link")]

        print(f"🔍 S1 Analysis — temat: {topic}, URL-i: {len(top_urls)}")

        # 2. LangExtract + spaCy dla TOP 5
        sources_payload = []
        h2_counts = []
        text_lengths = []
        all_headings = []

        for url in top_urls[:5]:
            content = call_langextract(url)

            if not content or content.get("error"):
                print(f"⚠️ Brak treści z LangExtract dla {url} → {content}")
                continue

            text = (
                content.get("content")
                or content.get("clean_text")
                or ""
            )
            h2s = content.get("h2", []) or []

            # Statystyki
            h2_counts.append(len(h2s))
            all_headings.extend([h.strip().lower() for h in h2s if isinstance(h, str)])

            word_count = len(re.findall(r"\w+", text))
            text_lengths.append(word_count)

            # spaCy NER + chunks
            spacy_data = extract_spacy_entities(text)

            # Payload do NGRAM API
            sources_payload.append({
                "url": url,
                "content": text,
                "spacy_entities": spacy_data["entities"],
                "spacy_noun_chunks": spacy_data["noun_chunks"]
            })

        # 3. Metryki konkurencji
        competitive_metrics = build_competitive_metrics(h2_counts, text_lengths)
        top_headings = extract_top_headings(all_headings, limit=10)

        # 4. GPT-Ngram API
        serp_context = {
            "people_also_ask": people_also_ask,
            "related_searches": autocomplete_suggestions,
        }
        ngram_data = call_ngram_api(sources_payload, topic, serp_context)

        # 5. Top n-gram summary
        top_ngrams_summary = []
        try:
            ngrams_list = None
            if isinstance(ngram_data, dict):
                if isinstance(ngram_data.get("ngrams"), list):
                    ngrams_list = ngram_data.get("ngrams")
                elif isinstance(ngram_data.get("data"), list):
                    ngrams_list = ngram_data.get("data")

            if ngrams_list:
                def ngram_sort_key(item):
                    if not isinstance(item, dict):
                        return 0
                    return (
                        item.get("weight")
                        or item.get("score")
                        or item.get("freq_norm")
                        or 0
                    )

                sorted_ngrams = sorted(
                    ngrams_list,
                    key=ngram_sort_key,
                    reverse=True
                )
                for item in sorted_ngrams[:30]:
                    if not isinstance(item, dict):
                        continue
                    top_ngrams_summary.append({
                        "ngram": item.get("ngram") or item.get("text"),
                        "weight": item.get("weight"),
                        "freq_norm": item.get("freq_norm"),
                        "site_distribution_score": item.get("site_distribution_score"),
                        "position_score": item.get("position_score"),
                    })
        except Exception as e:
            print(f"⚠️ Nie udało się zbudować top_ngrams_summary: {e}")
            top_ngrams_summary = []

        # 6. Finalny raport S1 (100% kompatybilny)
        response = {
            "topic": topic,
            "identified_urls": top_urls,
            "competitive_metrics": competitive_metrics,
            "ai_overview_status": ai_overview_status,
            "people_also_ask": people_also_ask,
            "autocomplete_suggestions": autocomplete_suggestions,
            "top_competitor_headings": top_headings,
            "s1_enrichment": ngram_data,   # legacy
            "ngram_summary": {
                "top_ngrams": top_ngrams_summary,
                "sources_count": len(sources_payload),
            },
            "spacy_entities_summary": {
                "total_urls": len(sources_payload),
                "entities_per_url": [
                    {
                        "url": s["url"],
                        "entities": s["spacy_entities"],
                        "noun_chunks": s["spacy_noun_chunks"]
                    }
                    for s in sources_payload
                ]
            }
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Błąd /api/s1_analysis: {e}")
        return jsonify({"error": str(e)}), 500
