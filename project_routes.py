import uuid
import re
import os
import json
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Global spaCy
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

project_routes = Blueprint("project_routes", __name__)


# ================================================================
# ⭐ FIX #5: LSI KEYWORD ENRICHMENT
# ================================================================

def extract_lsi_candidates(s1_data, main_keywords):
    """
    Z danych S1 (n-gramy + entities) wybiera LSI candidates,
    które NIE są w main keywords, ale są semantycznie związane.
    """
    if not s1_data or not isinstance(s1_data, dict):
        return []
    
    # 1. Zbierz wszystkie n-gramy i encje
    all_ngrams = []
    ngram_summary = s1_data.get("ngram_summary", {})
    if isinstance(ngram_summary, dict):
        top_ngrams = ngram_summary.get("top_ngrams", [])
        if isinstance(top_ngrams, list):
            all_ngrams = [ng.get("ngram", "") for ng in top_ngrams if isinstance(ng, dict)]
    
    all_entities = []
    entities_summary = s1_data.get("entities_summary", {})
    if isinstance(entities_summary, dict):
        entities_per_url = entities_summary.get("entities_per_url", [])
        if isinstance(entities_per_url, list):
            for url_data in entities_per_url:
                if isinstance(url_data, dict):
                    entities = url_data.get("entities", [])
                    if isinstance(entities, list):
                        for ent in entities:
                            if isinstance(ent, dict):
                                all_entities.append(ent.get("text", "").lower())
    
    # 2. Combine and filter
    candidates = set(all_ngrams + all_entities)
    candidates = {c for c in candidates if c and len(c.split()) <= 4}  # Max 4-gram
    
    # Remove exact matches z main keywords
    main_kw_lower = {kw.lower() for kw in main_keywords if kw}
    candidates = candidates - main_kw_lower
    
    if not candidates:
        return []
    
    # 3. Semantic filtering przez Gemini (jeśli dostępny)
    if not GEMINI_API_KEY:
        # Fallback: zwróć top 15 najczęstszych
        return list(candidates)[:15]
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        main_kw_sample = ', '.join(list(main_kw_lower)[:5])
        candidates_sample = ', '.join(list(candidates)[:50])  # Limit 50 dla API
        
        prompt = f"""
        Główne słowa kluczowe: {main_kw_sample}
        
        Kandydaci na LSI keywords: {candidates_sample}
        
        Wybierz 15 najbardziej semantycznie powiązanych terminów.
        Zwróć TYLKO JSON list: ["term1", "term2", ...]
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON (usuń markdown)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        lsi_terms = json.loads(response_text)
        
        if isinstance(lsi_terms, list):
            return lsi_terms[:15]
        else:
            return list(candidates)[:15]
            
    except Exception as e:
        print(f"LSI extraction error: {e}")
        return list(candidates)[:15]


# --- PARSER (Zachowany ze starego kodu dla wstecznej kompatybilności) ---
def parse_brief_text_uuid(brief_text: str):
    lines = brief_text.split("\n")
    parsed_dict = {}
    for line in lines:
        line = line.strip()
        if not line: continue
        kw_type = "BASIC"
        upper_line = line.upper()
        if "[EXTENDED]" in upper_line:
            kw_type = "EXTENDED"
            line = re.sub(r"\[EXTENDED\]", "", line, flags=re.IGNORECASE).strip()
        elif "[BASIC]" in upper_line:
            kw_type = "BASIC"
            line = re.sub(r"\[BASIC\]", "", line, flags=re.IGNORECASE).strip()
        if ":" not in line: continue
        try:
            parts = line.rsplit(":", 1)
            original_keyword = parts[0].strip()
            counts_part = parts[1].strip().lower()
            numbers = re.findall(r"\d+", counts_part)
            if not numbers: continue
            if len(numbers) >= 2: min_val, max_val = int(numbers[0]), int(numbers[1])
            else: min_val, max_val = int(numbers[0]), int(numbers[0])
            doc = nlp(original_keyword)
            search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
            row_id = str(uuid.uuid4())
            parsed_dict[row_id] = {
                "keyword": original_keyword,
                "search_term_exact": original_keyword.lower(),
                "search_lemma": search_lemma,
                "target_min": min_val,
                "target_max": max_val,
                "actual_uses": 0,
                "status": "UNDER",
                "type": kw_type
            }
        except Exception: continue
    return parsed_dict

# --- S2 CREATE (V12.25.1: JSON First + LSI Enrichment) ---
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Required field: topic"}), 400
    
    topic = data["topic"]
    
    # 1. Próba użycia nowego formatu JSON (V12)
    raw_keywords = data.get("keywords_list", [])
    firestore_keywords = {}

    if raw_keywords:
        # Nowy tryb JSON
        for item in raw_keywords:
            term = item.get("term", "").strip()
            if not term: continue
            
            doc = nlp(term)
            search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
            row_id = str(uuid.uuid4())
            
            firestore_keywords[row_id] = {
                "keyword": term,
                "search_term_exact": term.lower(),
                "search_lemma": search_lemma,
                "target_min": item.get("min", 1),
                "target_max": item.get("max", 5),
                "actual_uses": 0,
                "status": "UNDER",
                "type": item.get("type", "BASIC").upper()
            }
        brief_raw = "JSON_MODE_V12"
    
    # 2. Fallback do starego trybu tekstowego (V11)
    elif "brief_text" in data:
        brief_raw = data["brief_text"]
        firestore_keywords = parse_brief_text_uuid(brief_raw)
    
    else:
        return jsonify({"error": "Missing 'keywords_list' (JSON) or 'brief_text' (Legacy)."}), 400

    if not firestore_keywords:
        return jsonify({"error": "No valid keywords parsed."}), 400

    # ⭐ FIX #5: LSI AUTO-ENRICHMENT
    # Jeśli user przesłał dane S1, automatycznie dodaj LSI keywords
    s1_data = data.get("s1_data", None)  # Opcjonalne: dane z poprzedniego /api/s1_analysis
    lsi_added_count = 0
    
    if s1_data and isinstance(s1_data, dict):
        main_keywords = [meta["keyword"] for meta in firestore_keywords.values()]
        lsi_keywords = extract_lsi_candidates(s1_data, main_keywords)
        
        if lsi_keywords:
            for lsi_term in lsi_keywords:
                # Sprawdź czy już nie istnieje
                already_exists = any(
                    meta["keyword"].lower() == lsi_term.lower() 
                    for meta in firestore_keywords.values()
                )
                
                if not already_exists:
                    doc = nlp(lsi_term)
                    search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
                    row_id = str(uuid.uuid4())
                    
                    firestore_keywords[row_id] = {
                        "keyword": lsi_term,
                        "search_term_exact": lsi_term.lower(),
                        "search_lemma": search_lemma,
                        "target_min": 1,
                        "target_max": 3,  # Soft target dla LSI
                        "actual_uses": 0,
                        "status": "UNDER",
                        "type": "LSI_AUTO"  # Nowy typ: automatycznie dodany LSI
                    }
                    lsi_added_count += 1

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    project_data = {
        "topic": topic,
        "brief_raw": brief_raw,
        "keywords_state": firestore_keywords,
        "counting_mode": "uuid_hybrid",
        "continuous_counting": True,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "total_batches": 0,
        "version": "v12.25.1",  # ⭐ Updated version
        "lsi_enrichment": {
            "enabled": lsi_added_count > 0,
            "count": lsi_added_count
        }
    }
    
    doc_ref.set(project_data)
    
    response = {
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "keywords": len(firestore_keywords),
        "version": "v12.25.1"
    }
    
    # ⭐ Informuj o LSI enrichment
    if lsi_added_count > 0:
        response["lsi_enrichment"] = f"Dodano {lsi_added_count} LSI keywords automatycznie (typ: LSI_AUTO)"
    
    return jsonify(response), 201

# --- S3 ADD BATCH (Z obsługą next_action) ---
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400
        
    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})
    
    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    # Obsługa status code (zachowujemy logikę ze starego pliku)
    status_code = result.get("status_code", 200)
    if "status" in result and isinstance(result["status"], str):
         status_code = 200
    
    # Dodajemy tekst batcha do odpowiedzi (czasami przydatne dla debugu)
    result["batch_text_snippet"] = batch_text[:50] + "..."
    
    return jsonify(result), status_code

# ================================================================
# 🆕 S4 — EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    
    # 1. Zszywanie tekstu
    batches = data.get("batches", [])
    full_text_parts = [b.get("text", "") for b in batches]
    full_article_text = "\n\n".join(full_text_parts)

    # 2. Statystyki SEO (Ogólne)
    keywords_state = data.get("keywords_state", {})
    under = sum(1 for k in keywords_state.values() if k["status"] == "UNDER")
    over = sum(1 for k in keywords_state.values() if k["status"] == "OVER")
    ok = sum(1 for k in keywords_state.values() if k["status"] == "OK")
    locked = 1 if over >= 4 else 0

    # 3. LISTA SZCZEGÓŁOWA
    keyword_details = []
    for row_id, meta in keywords_state.items():
        keyword_details.append({
            "keyword": meta.get("keyword", "Unknown"),
            "type": meta.get("type", "BASIC"),
            "target": f"{meta.get('target_min')}-{meta.get('target_max')}",
            "actual": meta.get("actual_uses", 0),
            "status": meta.get("status", "UNKNOWN"),
            "position_score": meta.get("position_score", 0),  # ⭐ NEW (FIX #4)
            "position_quality": meta.get("position_quality", "NONE")  # ⭐ NEW (FIX #4)
        })
    keyword_details.sort(key=lambda x: (x['type'], x['keyword']))

    # 4. Statystyki Jakości (Średnie)
    scores = [b.get("gemini_audit", {}).get("quality_score", 0) for b in batches if b.get("gemini_audit")]
    bursts = [b.get("language_audit", {}).get("burstiness", 0) for b in batches if b.get("language_audit")]
    
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    avg_burst = round(sum(bursts) / len(bursts), 2) if bursts else 0

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "full_article_text": full_article_text,
        "final_stats": {"UNDER": under, "OVER": over, "LOCKED": locked, "OK": ok},
        "quality_metrics": {
            "avg_score": avg_score,
            "avg_burstiness": avg_burst
        },
        "keyword_details": keyword_details,
        "version": data.get("version", "v12.0")
    }), 200

# ================================================================
# 🗑️ S4 — DELETE
# ================================================================
@project_routes.delete("/api/project/<project_id>")
def delete_project_final(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    if not doc_ref.get().exists: return jsonify({"error": "Not found"}), 404
    doc_ref.delete()
    return jsonify({"status": "DELETED", "message": "Project permanently removed."}), 200
