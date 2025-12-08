"""
s1_analysis_routes.py — v18.0 Brajen Semantic Engine
Integracja z Firestore + analiza SERP i konkurencji
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import requests
import re
import statistics
from collections import Counter
import spacy

s1_routes = Blueprint("s1_routes", __name__)

# --- Load spaCy (Polish model) ---
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[S1] ✅ Załadowano model pl_core_news_lg")
except OSError:
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

# ================================================================
# 🔍 Helper: Extract H2, H3, keywords
# ================================================================
def extract_headings_and_terms(html_text):
    """Prosty parser nagłówków i popularnych terminów"""
    h2_list = re.findall(r"<h2[^>]*>(.*?)</h2>", html_text, re.IGNORECASE)
    h3_list = re.findall(r"<h3[^>]*>(.*?)</h3>", html_text, re.IGNORECASE)

    doc = nlp(re.sub(r"<[^>]+>", " ", html_text))
    words = [t.lemma_.lower() for t in doc if t.is_alpha and len(t) > 3]
    common_terms = Counter(words).most_common(30)

    return {
        "h2_count": len(h2_list),
        "h2_titles": h2_list,
        "h3_count": len(h3_list),
        "top_terms": common_terms
    }

# ================================================================
# 🧠 Core Function — SERP Analysis
# ================================================================
@s1_routes.post("/api/s1_analysis")
def perform_s1_analysis():
    data = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    serp_urls = data.get("urls", [])

    if not topic:
        return jsonify({"error": "Missing topic"}), 400
    if not serp_urls:
        return jsonify({"error": "Missing 'urls' list"}), 400

    print(f"[S1] 🔍 Analiza SERP dla tematu: {topic}, {len(serp_urls)} wyników")

    h2_counts = []
    article_lengths = []
    global_terms = Counter()
    entities_summary = []

    for url in serp_urls:
        try:
            html = requests.get(url, timeout=8).text
            extracted = extract_headings_and_terms(html)
            h2_counts.append(extracted["h2_count"])
            article_lengths.append(len(html.split()))
            global_terms.update([w for w, _ in extracted["top_terms"]])
            entities_summary.append({
                "url": url,
                "h2_count": extracted["h2_count"],
                "h2_titles": extracted["h2_titles"],
                "top_terms": extracted["top_terms"][:10]
            })
        except Exception as e:
            print(f"[S1] ⚠️ Błąd przy analizie {url}: {e}")

    if not h2_counts:
        return jsonify({"error": "No valid data from provided URLs"}), 400

    avg_h2_count = round(statistics.mean(h2_counts), 1)
    avg_article_length = round(statistics.mean(article_lengths), 1)
    top_ngrams = global_terms.most_common(20)

    result = {
        "topic": topic,
        "competitors_analyzed": len(serp_urls),
        "avg_h2_count": avg_h2_count,
        "avg_article_length": avg_article_length,
        "h2_counts": h2_counts,
        "h2_topics": [{"topic": h, "frequency": h2_counts.count(len(h))} for h in serp_urls],
        "top_ngrams": [{"ngram": w, "frequency": c} for w, c in top_ngrams],
        "entities_summary": entities_summary
    }

    return jsonify({"status": "S1_ANALYSIS_DONE", "analysis": result}), 200

# ================================================================
# 🔗 Firestore Integration
# ================================================================
@s1_routes.post("/api/s1_analysis/<project_id>")
def perform_s1_and_save(project_id):
    """Wykonuje analizę i zapisuje wynik S1 bezpośrednio do projektu Firestore."""
    data = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    serp_urls = data.get("urls", [])

    if not topic or not serp_urls:
        return jsonify({"error": "Missing topic or URLs"}), 400

    # 1️⃣ Analiza SERP
    analysis_response = perform_s1_analysis().json
    if "error" in analysis_response:
        return jsonify(analysis_response), 400
    s1_data = analysis_response["analysis"]

    # 2️⃣ Zapis do Firestore
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "Project not found"}), 404

    doc_ref.update({
        "s1_data": s1_data,
        "avg_competitor_length": s1_data["avg_article_length"],
        "lsi_enrichment": {"enabled": True, "source": "S1"},
        "updated_at": firestore.SERVER_TIMESTAMP
    })

    print(f"[S1] ✅ Wyniki analizy zapisane do projektu {project_id}")

    return jsonify({
        "status": "S1_SAVED",
        "project_id": project_id,
        "topic": topic,
        "avg_h2_count": s1_data["avg_h2_count"],
        "avg_length": s1_data["avg_article_length"],
        "lsi_ready": True
    }), 200
