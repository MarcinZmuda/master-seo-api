# ================================================================
# s1_analysis_routes.py — Turbo S1 (Fixed noun_chunks for PL)
# v12.25.1 - Added FIX #6 (Competitive Gap Analysis)
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

try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    # Fallback: pobieramy w locie jeśli brakuje
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

# Blueprint
s1_routes = Blueprint("s1_routes", __name__)


# ================================================
# ⭐ FIX #6: COMPETITIVE GAP ANALYSIS
# ================================================

def analyze_competitive_gaps(competitor_h2_list, user_h2_plan):
    """
    Porównuje H2 użytkownika z top competitor headings.
    Zwraca: co mają konkurenci, czego użytkownik nie ma.
    """
    if not competitor_h2_list or not user_h2_plan:
        return []
    
    # Normalizacja
    competitor_h2 = [h.lower().strip() for h in competitor_h2_list if h]
    your_h2 = [h.lower().strip() for h in user_h2_plan if h]
    
    # Extract core topics (usuwamy question words)
    def extract_core_topic(heading):
        # Remove question words
        heading = re.sub(
            r'^(co to jest|jak|dlaczego|kiedy|gdzie|czym jest|po co)\s+', 
            '', 
            heading, 
            flags=re.IGNORECASE
        )
        
        # Get main nouns (spaCy)
        try:
            doc = nlp(heading)
            nouns = [t.text.lower() for t in doc if t.pos_ == "NOUN"]
            return " ".join(nouns[:2]) if nouns else heading
        except:
            return heading
    
    competitor_topics = [extract_core_topic(h) for h in competitor_h2]
    your_topics = [extract_core_topic(h) for h in your_h2]
    
    # Count frequency of each topic in competitors
    topic_counter = Counter(competitor_topics)
    total_competitors = len(competitor_h2)
    
    # Find gaps
    gaps = []
    for topic, count in topic_counter.most_common():
        if not topic or len(topic) < 3:  # Skip very short topics
            continue
        
        # Check if user has this topic
        is_missing = not any(topic in your_topic for your_topic in your_topics)
        
        if is_missing:
            coverage = count / total_competitors if total_competitors > 0 else 0
            
            # Only report if >30% of competitors have it
            if coverage >= 0.3:
                priority = "HIGH" if coverage >= 0.6 else "MEDIUM"
                
                gaps.append({
                    "topic": topic,
                    "coverage": f"{coverage*100:.0f}%",
                    "priority": priority,
                    "competitor_count": count
                })
    
    # Sort by coverage (descending)
    gaps.sort(key=lambda x: x["competitor_count"], reverse=True)
    
    return gaps[:10]  # Top 10 gaps


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
# 🔧 spaCy entity extractor (NER + noun chunks FIX)
# ------------------------------------------------
def extract_spacy_entities(text: str):
    """
    Zwraca:
      - named entities
      - noun chunks (bezpiecznie dla PL)
    """
    doc = nlp(text)

    entities = []
    noun_chunks = []

    # 1. Wyciąganie Encji (Działa OK)
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    # 2. Wyciąganie Noun Chunks (ZABEZPIECZONE)
    # Model PL rzuca tutaj NotImplementedError, więc musimy to obsłużyć
    try:
        for chunk in doc.noun_chunks:
            noun_chunks.append(chunk.text)
    except (NotImplementedError, AttributeError):
        # Ignorujemy błąd dla języka polskiego
        pass

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
      1. SerpAPI
      2. LangExtract (top 5 URL)
      3. spaCy (SAFE MODE)
      4. Metryki
      5. GPT-Ngram API
      6. Final Report
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

            # spaCy NER + chunks (TERAZ ZABEZPIECZONE)
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
                    if not isinstance(item, dict): return 0
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
                    if not isinstance(item, dict): continue
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
            "entities_summary": {
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


# ================================================
# ⭐ NEW ENDPOINT: /api/gap_analysis
# FIX #6: Competitive Gap Analysis
# ================================================

@s1_routes.route("/api/gap_analysis", methods=["POST"])
def perform_gap_analysis():
    """
    Porównuje plan H2 użytkownika z competitive headings.
    Zwraca: Missing topics (gaps).
    """
    try:
        data = request.get_json() or {}
        
        competitor_headings = data.get("competitor_headings", [])
        user_h2_plan = data.get("user_h2_plan", [])
        
        if not competitor_headings:
            return jsonify({"error": "Brak 'competitor_headings'"}), 400
        
        if not user_h2_plan:
            return jsonify({"error": "Brak 'user_h2_plan'"}), 400
        
        gaps = analyze_competitive_gaps(competitor_headings, user_h2_plan)
        
        return jsonify({
            "status": "success",
            "gaps_found": len(gaps),
            "gaps": gaps,
            "message": f"Znaleziono {len(gaps)} tematów, których brakuje w Twoim planie" if gaps else "Brak braków - dobry coverage!"
        }), 200
        
    except Exception as e:
        print(f"❌ Błąd /api/gap_analysis: {e}")
        return jsonify({"error": str(e)}), 500
