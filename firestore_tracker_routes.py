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
# 1. SIMPLE LEMMA COUNTING (for BASIC keywords - exact lemma match only)
# ============================================================================
def count_lemma_only(text_lemmas, keyword_lemmas):
    """
    Proste zliczanie lematów - szuka dokładnej sekwencji lematów frazy w tekście.
    Używane dla BASIC keywords.
    
    Args:
        text_lemmas: lista lematów z tekstu batcha
        keyword_lemmas: lista lematów frazy kluczowej
    
    Returns:
        int: liczba wystąpień frazy
    """
    if not text_lemmas or not keyword_lemmas:
        return 0
    
    kw_len = len(keyword_lemmas)
    text_len = len(text_lemmas)
    
    if kw_len > text_len:
        return 0
    
    count = 0
    # Przesuwamy okno po tekście i szukamy dokładnych dopasowań sekwencji lematów
    for i in range(text_len - kw_len + 1):
        if text_lemmas[i:i + kw_len] == keyword_lemmas:
            count += 1
    
    return count


# ============================================================================
# 2. ROBUST COUNTING (for EXTENDED keywords - keeps fuzzy matching)
# ============================================================================
def count_robust(doc, keyword_meta):
    """
    Zaawansowane zliczanie z fuzzy matching - używane dla EXTENDED keywords.
    """
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
# 3. FIRESTORE PROCESSOR (v19.7 - Simple Lemma Count for BASIC + HARD BLOCK)
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None, forced=False):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # ⭐ KROK 1: Przygotuj lematy tekstu (raz dla całego batcha)
    # Usuń tagi HTML przed lematyzacją
    clean_text = re.sub(r"<[^>]+>", " ", batch_text)
    clean_text = re.sub(r"\s+", " ", clean_text).lower()
    doc_nlp = nlp(clean_text)
    text_lemmas = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    # ⭐ KROK 2: Policz wystąpienia w batchu BEZ aktualizacji stanu
    batch_counts = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower().strip()
        kw_type = meta.get("type", "BASIC").upper()
        
        if not keyword:
            batch_counts[rid] = 0
            continue
        
        # Lematyzuj frazę kluczową
        kw_doc = nlp(keyword)
        keyword_lemmas = [t.lemma_.lower() for t in kw_doc if t.is_alpha]
        
        if kw_type == "BASIC":
            # ⭐ BASIC: Proste zliczanie lematów (dokładne dopasowanie sekwencji)
            count = count_lemma_only(text_lemmas, keyword_lemmas)
        else:
            # EXTENDED: Używa starej metody (fuzzy matching)
            count = count_robust(doc_nlp, meta)
        
        batch_counts[rid] = count
    
    # ⭐ KROK 3: Sprawdź czy BASIC keywords nie przekroczą target_max (HARD BLOCK)
    blocked_keywords = []
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        kw_type = meta.get("type", "BASIC").upper()
        
        # HARD BLOCK tylko dla BASIC!
        if kw_type != "BASIC":
            continue
            
        current = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        new_total = current + batch_count
        
        if new_total > target_max:
            blocked_keywords.append({
                "keyword": meta.get("keyword"),
                "current": current,
                "batch_uses": batch_count,
                "would_be": new_total,
                "target_max": target_max,
                "exceeded_by": new_total - target_max
            })
    
    # ⭐ KROK 4: Jeśli BASIC przekracza → BLOCK (chyba że forced)
    if blocked_keywords and not forced:
        return {
            "error": "KEYWORDS_EXCEEDED_MAX",
            "status": "BLOCKED",
            "blocked_keywords": blocked_keywords,
            "message": f"❌ {len(blocked_keywords)} BASIC keyword(s) would exceed target_max. Use synonyms!",
            "hint": "Użyj synonimów dla zablokowanych fraz lub wyślij z forced=true",
            "status_code": 400
        }
    
    # ⭐ KROK 5: Aktualizuj stan keywords
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        kw_type = meta.get("type", "BASIC").upper()
        
        meta["actual_uses"] = meta.get("actual_uses", 0) + batch_count
        
        min_t = meta.get("target_min", 0)
        max_t = meta.get("target_max", 999)
        actual = meta["actual_uses"]
        
        # Status calculation
        if actual < min_t:
            meta["status"] = "UNDER"
        elif actual == max_t:
            meta["status"] = "OPTIMAL"
        elif min_t <= actual < max_t:
            meta["status"] = "OK"
        elif actual > max_t:
            meta["status"] = "OVER"
        
        # ⭐ remaining_max - tylko dla BASIC
        if kw_type == "BASIC":
            meta["remaining_max"] = max(0, max_t - actual)
            meta["optimal_target"] = max_t
        
        keywords_state[rid] = meta

    # 🔹 Prewalidacja
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

    # ⭐ KROK 6: Przygotuj keyword_targets dla response (z remaining_max dla BASIC!)
    keyword_targets = []
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        
        target_info = {
            "keyword": meta.get("keyword"),
            "current": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min", 0),
            "target_max": meta.get("target_max", 999),
            "status": meta.get("status"),
            "type": kw_type,
            "batch_count": batch_counts.get(rid, 0)  # ⭐ Ile było w tym batchu
        }
        
        # ⭐ remaining_max tylko dla BASIC
        if kw_type == "BASIC":
            target_info["optimal_target"] = meta.get("optimal_target", meta.get("target_max", 999))
            target_info["remaining_max"] = meta.get("remaining_max", 0)
            target_info["remaining_to_optimal"] = max(0, target_info["optimal_target"] - target_info["current"])
        
        keyword_targets.append(target_info)

    return {
        "status": status,
        "semantic_score": semantic_score,
        "density": density,
        "burstiness": burstiness,
        "warnings": warnings,
        "keyword_targets": keyword_targets,  # ⭐ CRITICAL FOR GPT!
        "meta": precheck.get("meta", {}),
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
    
    # ⭐ Jeśli BLOCKED, zwróć 400
    if result.get("status") == "BLOCKED":
        return jsonify(result), 400
    
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
    
    # ⭐ Jeśli BLOCKED, zwróć 400
    if result.get("status") == "BLOCKED":
        return jsonify(result), 400
    
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
            # ✅ POPRAWKA: Usunięto [:15000] - teraz analizuje CAŁY artykuł
            full_article = "\n\n".join([b.get("text", "") for b in project_data.get("batches", [])])
            print(f"[TRACKER] 🔍 Analiza CAŁEGO artykułu ({len(full_article)} znaków)...")
            review_prompt = (
                "Podaj w punktach szczegółową ocenę przesłanego artykułu pod kątem:\n"
                "1. merytorycznym (zgodność faktów, aktualność, błędy logiczne),\n"
                "2. redakcyjnym (struktura, powtórzenia, styl),\n"
                "3. językowym (poprawność gramatyczna, płynność),\n"
                "a także zaproponuj konkretne poprawki dla każdego problemu.\n\n"
                f"---\n{full_article}"
            )
            review_response = model.generate_content(review_prompt)
            review_text = review_response.text.strip()
            doc_ref.update({
                "final_review": {
                    "review_text": review_text,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "model": "gemini-1.5-pro",
                    "status": "REVIEW_READY",
                    "article_length": len(full_article)  # ⭐ DODANO tracking długości
                }
            })
            result["final_review"] = review_text
            result["article_length"] = len(full_article)  # ⭐ DODANO info o długości
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
        kw_type = meta.get("type", "BASIC").upper()
        stat_entry = {
            "keyword": meta.get("keyword"),
            "type": kw_type,
            "actual_uses": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min", 0),
            "target_max": meta.get("target_max", 999),
            "status": meta.get("status"),
            "target": f"{meta.get('target_min')}-{meta.get('target_max')}"
        }
        
        # ⭐ remaining_max tylko dla BASIC
        if kw_type == "BASIC":
            stat_entry["optimal_target"] = meta.get("optimal_target", meta.get("target_max", 999))
            stat_entry["remaining_max"] = meta.get("remaining_max", max(0, meta.get("target_max", 999) - meta.get("actual_uses", 0)))
        
        stats.append(stat_entry)
        
    return jsonify({
        "project_id": project_id,
        "keywords": stats,
        "batches": len(batches),
        "last_burst": batches[-1].get("burstiness") if batches else None
    }), 200
