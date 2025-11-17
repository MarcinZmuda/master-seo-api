# ================================================================
# s1_analysis_routes.py — Turbo S1 (SERP + LangExtract + N-gramy)
# v7.3.0-firestore-continuous-lemma
# ================================================================

import os
import re
import json
import requests
from collections import Counter
from flask import Blueprint, request, jsonify

# ------------------------------------------------
# 🌐 Konfiguracja z ENV (Render)
# ------------------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# Możesz też wrzucić do ENV: LANGEXTRACT_API_URL,
# ale zostawiam sensowne domyślne jak w starym kodzie:
LANGEXTRACT_API_URL = os.getenv(
    "LANGEXTRACT_API_URL",
    "https://langextract-api.onrender.com/extract"
)

# NGRAM_API_URL masz już w ENV, ale daję fallback:
NGRAM_API_URL = os.getenv(
    "NGRAM_API_URL",
    "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"
)

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
    """
    Wywołuje LangExtract API w formacie:
    POST { "url": "https://..." }
    Oczekuje odpowiedzi z kluczami:
        - "content" (pełny tekst)
        - "h2" (lista nagłówków H2)
    """
    payload = {"url": url}
    return call_api_with_json(LANGEXTRACT_API_URL, payload, "LangExtract", timeout=120)


# ------------------------------------------------
# 🔧 N-gram API
# ------------------------------------------------
def call_ngram_api(sources_payload, topic, serp_context):
    """
    Wywołuje zewnętrzne GPT-Ngram API z Twoją logiką:
      - freq_norm
      - site_distribution_score
      - position_score
      - context boosts (PAA + related)
    """
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
    """
    Normalizuje H2 do lower() i liczy najczęściej powtarzane.
    """
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
      3. Obliczenie metryk konkurencji (H2 + długość tekstu)
      4. N-gram API (Twoja logika) → freq_norm / site_distribution / position / context boosts
      5. Zwrócenie pełnego raportu S1 do GPT (S1 full report in chat)
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

        # AI Overview status (jeśli jest)
        ai_overview_status = (
            serp_data.get("ai_overview", {}).get("status", "not_available")
        )

        # PAA (People Also Ask)
        people_also_ask = [
            q.get("question")
            for q in serp_data.get("related_questions", [])
            if q.get("question")
        ]

        # Autocomplete / Related searches
        autocomplete_suggestions = [
            r.get("query")
            for r in serp_data.get("related_searches", [])
            if r.get("query")
        ]

        # Top organic results — bierzemy do 7, ale analizujemy treść TOP 5
        organic_results = serp_data.get("organic_results", []) or []
        top_urls = [r.get("link") for r in organic_results[:7] if r.get("link")]

        print(f"🔍 S1 Analysis — temat: {topic}, URL-i: {len(top_urls)}")

        # 2. LangExtract dla TOP 5
        sources_payload = []   # do NGRAM API
        h2_counts = []         # do metryk
        text_lengths = []      # do metryk
        all_headings = []      # globalna lista H2

        for url in top_urls[:5]:
            content = call_langextract(url)

            if not content or content.get("error"):
                print(f"⚠️ Brak treści z LangExtract dla {url} → {content}")
                continue

            # LangExtract może zwracać "content" lub np. "clean_text"
            text = (
                content.get("content")
                or content.get("clean_text")
                or ""
            )
            h2s = content.get("h2", []) or []

            # proste statystyki
            h2_counts.append(len(h2s))
            all_headings.extend([h.strip().lower() for h in h2s if isinstance(h, str)])

            # liczenie słów w tekście
            word_count = len(re.findall(r"\w+", text))
            text_lengths.append(word_count)

            # payload dla NGRAM API
            sources_payload.append({
                "url": url,
                "content": text
            })

        # 3. Metryki konkurencji
        competitive_metrics = build_competitive_metrics(h2_counts, text_lengths)
        top_headings = extract_top_headings(all_headings, limit=10)

        # 4. N-gram API z Twoją logiką (freq_norm, site_distribution, position_score)
        serp_context = {
            "people_also_ask": people_also_ask,
            "related_searches": autocomplete_suggestions,
        }
        ngram_data = call_ngram_api(sources_payload, topic, serp_context)

        # Możemy spróbować wyciągnąć "top_ngrams", ale robimy to defensywnie,
        # żeby nie wywalić się, jeśli format JSON się zmieni.
        top_ngrams_summary = []
        try:
            # Oczekiwany format: {"ngrams": [{ "ngram": "...", "weight": 0.87, ... }, ...]}
            ngrams_list = None
            if isinstance(ngram_data, dict):
                if isinstance(ngram_data.get("ngrams"), list):
                    ngrams_list = ngram_data.get("ngrams")
                elif isinstance(ngram_data.get("data"), list):
                    # alternatywne pole
                    ngrams_list = ngram_data.get("data")

            if ngrams_list:
                # sortujemy po "weight" / "score" / "freq_norm"
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

        # 5. Finalny raport S1 — ZACHOWUJEMY POLA ZE STAREJ WERSJI,
        #    tak żeby Twój custom GPT dalej działał bez zmian:
        response = {
            "topic": topic,
            "identified_urls": top_urls,
            "competitive_metrics": competitive_metrics,
            "ai_overview_status": ai_overview_status,
            "people_also_ask": people_also_ask,
            "autocomplete_suggestions": autocomplete_suggestions,
            "top_competitor_headings": top_headings,
            # To pole miało wcześniej "gołego" JSON-a z NGRAM API:
            "s1_enrichment": ngram_data,
            # DODATKOWE pole z wyciągniętym TOP 30 n-gramów (bez psucia starego formatu):
            "ngram_summary": {
                "top_ngrams": top_ngrams_summary,
                "sources_count": len(sources_payload),
            },
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Błąd /api/s1_analysis: {e}")
        return jsonify({"error": str(e)}), 500
