"""
SEO Content Tracker Routes - v19.5 Brajen Semantic Engine
+ Interactive Final Review (Gemini)
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
import spacy
from rapidfuzz import fuzz
from seo_optimizer import unified_prevalidation
from google.api_core.exceptions import InvalidArgument
import google.generativeai as genai
import os

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INIT SPACY (MD for Polish - fix for Docker timeout) ---
try:
    nlp = spacy.load("pl_core_news_md")
    print("[TRACKER] ✅ Załadowano model pl_core_news_md")
except OSError:
    from spacy.cli import download
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

# --- Gemini Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[TRACKER] ✅ Gemini API aktywny (Final Review Mode)")
else:
    print("[TRACKER] ⚠️ Brak GEMINI_API_KEY — Final Review nieaktywny")

# ============================================================================
# 1. ROBUST COUNTING
# ============================================================================
def count_robust(doc, keyword_meta):
    if not doc:
        return 0
    kw_exact = keyword_meta.get("keyword", "").lower().strip()
    if not kw_exact:
        return 0
    kw_doc = nlp(kw_exact)
    kw_lemma = " ".join([t.lemma_.lower() for t in kw_doc if t.is_alpha]) or kw_exact
    raw_text = getattr(doc, "text", "").lower()
    clean_text = re.sub(r"<[^>]+>", " ", raw_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    exact_hits = len(re.findall(rf"\b{re.escape(kw_exact)}\b", clean_text))
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
    lemma_str = " ".join(lemmas)
    lemma_exact_hits = len(re.findall(rf"\b{re.escape(kw_lemma)}\b", lemma_str))
    lemma_fuzzy_hits = sum(
        lemma_str.count(token)
        for token in set(lemmas)
        if fuzz.token_set_ratio(kw_lemma, token) >= 90
    )
    kw_stem = kw_lemma[: min(5, len(kw_lemma))]
    stem_hits = sum(1 for token in lemmas if token.startswith(kw_stem) and len(token) > 4)
    fuzzy_stem_hits = sum(
        lemma_str.count(token)
        for token in set(lemmas)
        if fuzz.partial_ratio(kw_stem, token) >= 85
    )
    total_hits = max(exact_hits, lemma_exact_hits + lemma_fuzzy_hits + stem_hits + fuzzy_stem_hits)
    return total_hits

# ============================================================================
# 2. VALIDATIONS
# ============================================================================
def validate_structure(text):
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Markdown (##) zabroniony — użyj <h2>."}
    banned = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie", "wnioski", "konkluzja"]
    for h2 in re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL):
        if any(b in h2.lower() for b in banned):
            return {"valid": False, "error": f"❌ Niedozwolony nagłówek: '{h2.strip()}'"}
    return {"valid": True}

def calculate_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return round((math.sqrt(variance) / mean) * 10, 2) if mean else 0.0

# ============================================================================
# 3. FIRESTORE PROCESSOR
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None, forced=False):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    doc_nlp = nlp(batch_text)

    for rid, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        min_t, max_t = meta.get("target_min", 0), meta.get("target_max", 999)
        if meta["actual_uses"] < min_t:
            meta["status"] = "UNDER"
        elif min_t <= meta["actual_uses"] <= max_t:
            meta["status"] = "OK"
        elif meta["actual_uses"] > max_t:
            meta["status"] = "OVER"
        else:
            meta["status"] = "LOCKED"
        keywords_state[rid] = meta

    # 🔹 POPRAWKA: Teraz przekazujemy keywords_state zamiast project_id
    precheck = unified_prevalidation(batch_text, keywords_state)
    warnings = precheck.get("warnings", [])
    semantic_score = precheck.get("semantic_score", 1.0)
    density = precheck.get("density", 0.0)
    smog = precheck.get("smog", 0.0)
    readability = precheck.get("readability", 0.0)
    burstiness = calculate_burstiness(batch_text)

    struct_check = validate_structure(batch_text)
    valid_struct = struct_check["valid"]
    warning_text = struct_check.get("error") if not valid_struct else None

    status = "APPROVED"
    if warnings or not valid_struct:
        status = "WARN"
    if forced:
        status = "FORCED"

    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness,
        "language_audit": {
            "semantic_score": semantic_score,
            "density": density,
            "smog": smog,
            "readability": readability
        },
        "warnings": warnings,
        "validation_error": warning_text,
        "status": status
    }

    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state
    try:
        doc_ref.set(project_data)
    except InvalidArgument:
        doc_ref.set({k: v for k, v in project_data.items() if v is not None})
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Błąd zapisu: {e}")

    # 🔹 POPRAWKA: Dodajemy meta do zwracanego dict
    return {
        "status": status,
        "semantic_score": semantic_score,
        "density": density,
        "burstiness": burstiness,
        "warnings": warnings,
        "meta": precheck.get("meta", {}),  # ✅ Dodane meta z paragraph_rhythm
        "status_code": 200
    }

# ============================================================================
# 4. ROUTES — PREVIEW, APPROVE, DEBUG
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True)
    text = data.get("batch_text", "")
    forced = data.get("forced", False)
    result = process_batch_in_firestore(project_id, text, forced=forced)
    result["mode"] = "PREVIEW_ONLY"
    return jsonify(result), 200


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    Zapisuje batch i automatycznie uruchamia końcowy audyt (Gemini),
    jeśli to był ostatni batch.
    """
    data = request.get_json(force=True)
    text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

    # Zapisz batch
    result = process_batch_in_firestore(project_id, text, meta_trace, forced)
    result["mode"] = "APPROVE"

    # 🔹 Sprawdź, czy to ostatni batch
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify(result), 200

    project_data = doc.to_dict()
    total_planned = len(project_data.get("batches_plan", []))
    total_current = len(project_data.get("batches", []))

    if total_planned and total_current >= total_planned and GEMINI_API_KEY:
        try:
            print(f"[TRACKER] 🧠 Final batch detected → uruchamiam Gemini review dla {project_id}")
            model = genai.GenerativeModel("gemini-1.5-pro")
            full_article = "\n\n".join([b.get("text", "") for b in project_data.get("batches", [])])[:15000]
            review_prompt = (
                "Podaj w punktach szczegółową ocenę przesłanego artykułu pod kątem:\n"
                "1. merytorycznym,\n2. redakcyjnym,\n3. językowym.\n"
                "Dla każdego punktu zaproponuj, jak to poprawić.\n\n"
                f"---\n{full_article}"
            )
            review_response = model.generate_content(review_prompt)
            review_text = review_response.text.strip()
            doc_ref.update({
                "final_review": {
                    "review_text": review_text,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "model": "gemini-1.5-pro",
                    "status": "REVIEW_READY"
                }
            })
            result["final_review"] = review_text
            result["next_action"] = "Czy chcesz wprowadzić poprawki automatycznie? (POST /api/project/<id>/apply_final_corrections)"
            result["final_review_status"] = "READY"
            print(f"[TRACKER] ✅ Raport Gemini zapisany w Firestore → {project_id}")
        except Exception as e:
            print(f"[TRACKER] ⚠️ Błąd Gemini review: {e}")
            result["final_review_error"] = str(e)

    return jsonify(result), 200


@tracker_routes.get("/api/debug/<project_id>")
def debug_keywords(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    stats = []
    for rid, meta in keywords.items():
        stats.append({
            "keyword": meta.get("keyword"),
            "type": meta.get("type"),
            "actual_uses": meta.get("actual_uses", 0),
            "status": meta.get("status"),
            "target": f"{meta.get('target_min')}-{meta.get('target_max')}"
        })
    return jsonify({
        "project_id": project_id,
        "keywords": stats,
        "batches": len(batches),
        "last_burst": batches[-1].get("burstiness") if batches else None
    }), 200
