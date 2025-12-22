"""
SEO Content Tracker Routes - v22.1 Brajen Semantic Engine
+ Interactive Final Review (Gemini)
+ Burstiness validation (3.2-3.8)
+ Transition words validation (25-50%)
+ Metrics object in response
+ FIX: approve_batch accepts corrected_text/text/batch_text
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
    """
    if not text_lemmas or not keyword_lemmas:
        return 0
    
    kw_len = len(keyword_lemmas)
    text_len = len(text_lemmas)
    
    if kw_len > text_len:
        return 0
    
    count = 0
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
# 3. VALIDATIONS
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
    """
    Oblicza burstiness - zróżnicowanie długości zdań.
    Target: 3.2-3.8 (optymalnie 3.5)
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    
    if not mean:
        return 0.0
    
    raw_score = math.sqrt(variance) / mean
    normalized = raw_score * 5
    return round(normalized, 2)


# ============================================================================
# 🆕 v22.0: TRANSITION WORDS VALIDATION
# ============================================================================
TRANSITION_WORDS_PL = [
    "również", "także", "ponadto", "dodatkowo", "co więcej",
    "oprócz tego", "poza tym", "w dodatku", "nie tylko", "ale także",
    "jednak", "jednakże", "natomiast", "ale", "z drugiej strony",
    "mimo to", "niemniej", "pomimo", "choć", "chociaż", "wprawdzie",
    "dlatego", "w związku z tym", "w rezultacie", "wskutek", "ponieważ",
    "zatem", "więc", "stąd", "w konsekwencji", "przez co",
    "na przykład", "przykładowo", "między innymi", "m.in.", "np.",
    "podsumowując", "reasumując", "w skrócie", "ogólnie rzecz biorąc",
    "po pierwsze", "po drugie", "następnie", "potem", "w końcu", "na koniec",
    "efekt?", "rezultat?", "co się dzieje?", "i co dalej?", "co to oznacza?",
    "dlaczego?", "jak to działa?", "mechanizm?", "przyczyna?"
]

BANNED_SECTION_OPENERS = [
    "dlatego", "ponadto", "dodatkowo", "w związku z tym", "tym samym", "warto", "należy"
]


def calculate_transition_score(text: str) -> dict:
    """
    Oblicza jakość przejść między zdaniami.
    Target: 25-50% zdań z transition words
    """
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences), "warnings": []}
    
    transition_count = 0
    warnings = []
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()[:100]
        has_transition = any(tw in sentence_lower for tw in TRANSITION_WORDS_PL)
        if has_transition:
            transition_count += 1
    
    ratio = transition_count / len(sentences) if sentences else 0
    
    if ratio < 0.20:
        warnings.append(f"⚠️ Za mało transition words: {ratio:.0%} (min 25%)")
    elif ratio > 0.55:
        warnings.append(f"⚠️ Za dużo transition words: {ratio:.0%} (max 50%)")
    
    return {
        "ratio": round(ratio, 3),
        "count": transition_count,
        "total": len(sentences),
        "warnings": warnings
    }


def check_banned_openers(text: str) -> list:
    """
    Sprawdza czy sekcje zaczynają się od zakazanych słów.
    """
    warnings = []
    
    h2_pattern = re.compile(r'</h2>\s*<p>([^.!?]+[.!?])', re.IGNORECASE)
    h3_pattern = re.compile(r'</h3>\s*<p>([^.!?]+[.!?])', re.IGNORECASE)
    
    for match in h2_pattern.findall(text):
        first_word = match.strip().split()[0].lower() if match.strip() else ""
        for banned in BANNED_SECTION_OPENERS:
            if first_word == banned or match.strip().lower().startswith(banned):
                warnings.append(f"⚠️ Sekcja H2 zaczyna się od '{banned}' - przenieś dalej w akapicie")
                break
    
    for match in h3_pattern.findall(text):
        first_word = match.strip().split()[0].lower() if match.strip() else ""
        for banned in BANNED_SECTION_OPENERS:
            if first_word == banned or match.strip().lower().startswith(banned):
                warnings.append(f"⚠️ Sekcja H3 zaczyna się od '{banned}' - przenieś dalej w akapicie")
                break
    
    return warnings


def validate_metrics(burstiness: float, transition_data: dict, density: float) -> list:
    """
    Waliduje wszystkie metryki i zwraca listę warnings.
    """
    warnings = []
    
    if burstiness < 3.2:
        warnings.append(f"⚠️ Burstiness za niski: {burstiness} (min 3.2)")
    elif burstiness > 3.8:
        warnings.append(f"⚠️ Burstiness za wysoki: {burstiness} (max 3.8)")
    
    if density > 3.0:
        warnings.append(f"⚠️ Keyword density za wysoka: {density}% (max 3.0%)")
    elif density > 2.5:
        warnings.append(f"⚡ Keyword density blisko limitu: {density}%")
    
    warnings.extend(transition_data.get("warnings", []))
    
    return warnings


# ============================================================================
# 4. FIRESTORE PROCESSOR (v22.0 - with metrics validation)
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None, forced=False):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    clean_text = re.sub(r"<[^>]+>", " ", batch_text)
    clean_text = re.sub(r"\s+", " ", clean_text).lower()
    doc_nlp = nlp(clean_text)
    text_lemmas = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    batch_counts = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower().strip()
        kw_type = meta.get("type", "BASIC").upper()
        
        if not keyword:
            batch_counts[rid] = 0
            continue
        
        kw_doc = nlp(keyword)
        keyword_lemmas = [t.lemma_.lower() for t in kw_doc if t.is_alpha]
        
        if kw_type == "BASIC":
            count = count_lemma_only(text_lemmas, keyword_lemmas)
        else:
            count = count_robust(doc_nlp, meta)
        
        batch_counts[rid] = count
    
    exceeded_keywords = []
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        kw_type = meta.get("type", "BASIC").upper()
        
        if kw_type != "BASIC":
            continue
            
        current = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        new_total = current + batch_count
        
        if new_total > target_max:
            exceeded_keywords.append({
                "keyword": meta.get("keyword"),
                "current": current,
                "batch_uses": batch_count,
                "would_be": new_total,
                "target_max": target_max,
                "exceeded_by": new_total - target_max
            })
    
    exceeded_warnings = []
    if exceeded_keywords:
        for ek in exceeded_keywords:
            exceeded_warnings.append(
                f"⚠️ EXCEEDED: '{ek['keyword']}' będzie {ek['would_be']}x (max {ek['target_max']}x)"
            )
    
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        kw_type = meta.get("type", "BASIC").upper()
        
        meta["actual_uses"] = meta.get("actual_uses", 0) + batch_count
        
        min_t = meta.get("target_min", 0)
        max_t = meta.get("target_max", 999)
        actual = meta["actual_uses"]
        
        if actual < min_t:
            meta["status"] = "UNDER"
        elif actual == max_t:
            meta["status"] = "OPTIMAL"
        elif min_t <= actual < max_t:
            meta["status"] = "OK"
        elif actual > max_t:
            meta["status"] = "OVER"
        
        if kw_type == "BASIC":
            meta["remaining_max"] = max(0, max_t - actual)
            meta["optimal_target"] = max_t
        
        keywords_state[rid] = meta

    precheck = unified_prevalidation(batch_text, keywords_state)
    warnings = precheck.get("warnings", [])
    semantic_score = precheck.get("semantic_score", 1.0)
    transition_score = precheck.get("transition_score", 0.8)
    density = precheck.get("density", 0.0)
    smog = precheck.get("smog", 0.0)
    readability = precheck.get("readability", 0.0)
    
    warnings.extend(exceeded_warnings)
    
    burstiness = calculate_burstiness(batch_text)
    transition_data = calculate_transition_score(batch_text)
    banned_opener_warnings = check_banned_openers(batch_text)
    
    metrics_warnings = validate_metrics(burstiness, transition_data, density)
    warnings.extend(metrics_warnings)
    warnings.extend(banned_opener_warnings)

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
        "transition_ratio": transition_data.get("ratio", 0),
        "language_audit": {
            "semantic_score": semantic_score,
            "transition_score": transition_score,
            "transition_ratio": transition_data.get("ratio", 0),
            "density": density,
            "smog": smog,
            "readability": readability,
            "burstiness": burstiness
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
            "batch_count": batch_counts.get(rid, 0)
        }
        
        if kw_type == "BASIC":
            target_info["optimal_target"] = meta.get("optimal_target", meta.get("target_max", 999))
            target_info["remaining_max"] = meta.get("remaining_max", 0)
            target_info["remaining_to_optimal"] = max(0, target_info["optimal_target"] - target_info["current"])
        
        keyword_targets.append(target_info)

    return {
        "status": status,
        "semantic_score": semantic_score,
        "transition_score": transition_score,
        "transition_ratio": transition_data.get("ratio", 0),
        "density": density,
        "burstiness": burstiness,
        "warnings": warnings,
        "exceeded_keywords": exceeded_keywords,
        "keyword_targets": keyword_targets,
        "metrics": {
            "burstiness": {
                "value": burstiness,
                "target": "3.2-3.8",
                "status": "OK" if 3.2 <= burstiness <= 3.8 else "WARN"
            },
            "transition_ratio": {
                "value": transition_data.get("ratio", 0),
                "target": "0.25-0.50",
                "status": "OK" if 0.20 <= transition_data.get("ratio", 0) <= 0.55 else "WARN"
            },
            "density": {
                "value": density,
                "target": "<3.0%",
                "status": "OK" if density <= 3.0 else "WARN"
            },
            "semantic_score": {
                "value": semantic_score,
                "target": ">0.6",
                "status": "OK" if semantic_score >= 0.6 else "WARN"
            }
        },
        "meta": precheck.get("meta", {}),
        "status_code": 200
    }


# ============================================================================
# 5. ROUTES — PREVIEW, APPROVE, DEBUG
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True)
    text = data.get("batch_text") or data.get("text") or data.get("corrected_text") or ""
    forced = data.get("forced", False)
    
    if not text.strip():
        return jsonify({
            "error": "No text provided",
            "hint": "Send 'batch_text', 'text', or 'corrected_text' field"
        }), 400
    
    result = process_batch_in_firestore(project_id, text, forced=forced)
    result["mode"] = "PREVIEW_ONLY"
    return jsonify(result), 200


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    Zapisuje batch i automatycznie uruchamia końcowy audyt (Gemini),
    jeśli to był ostatni batch.
    
    ⭐ FIX v22.1: Akceptuje corrected_text, text, lub batch_text
    """
    data = request.get_json(force=True)
    
    # ⭐ FIX: Akceptuj różne nazwy pól
    text = data.get("corrected_text") or data.get("text") or data.get("batch_text") or ""
    
    if not text.strip():
        return jsonify({
            "error": "No text provided",
            "hint": "Send 'corrected_text', 'text', or 'batch_text' field with content",
            "received_fields": list(data.keys())
        }), 400
    
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
        existing_fr = project_data.get("final_review") if isinstance(project_data, dict) else None
        if existing_fr and not forced:
            result["final_review"] = existing_fr.get("review_text") if isinstance(existing_fr, dict) else existing_fr
            result["final_review_status"] = existing_fr.get("status") if isinstance(existing_fr, dict) else "REVIEW_READY"
            result["next_action"] = "Final review już istnieje. Jeśli chcesz przeliczyć, wyślij approve_batch z forced=true."
            return jsonify(result), 200
        try:
            print(f"[TRACKER] 🧠 Final batch detected → uruchamiam Gemini review dla {project_id}")
            model_name = os.getenv("FINAL_REVIEW_MODEL", "gemini-2.5-flash")
            model = genai.GenerativeModel(model_name)
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
                    "model": model_name,
                    "status": "REVIEW_READY",
                    "article_length": len(full_article)
                }
            })
            result["final_review"] = review_text
            result["article_length"] = len(full_article)
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
        
        if kw_type == "BASIC":
            stat_entry["optimal_target"] = meta.get("optimal_target", meta.get("target_max", 999))
            stat_entry["remaining_max"] = meta.get("remaining_max", max(0, meta.get("target_max", 999) - meta.get("actual_uses", 0)))
        
        stats.append(stat_entry)
    
    avg_burstiness = 0
    avg_transition = 0
    if batches:
        burst_values = [b.get("burstiness", 0) for b in batches if b.get("burstiness")]
        trans_values = [b.get("transition_ratio", 0) for b in batches if b.get("transition_ratio")]
        avg_burstiness = round(sum(burst_values) / len(burst_values), 2) if burst_values else 0
        avg_transition = round(sum(trans_values) / len(trans_values), 3) if trans_values else 0
        
    return jsonify({
        "project_id": project_id,
        "keywords": stats,
        "batches": len(batches),
        "last_burst": batches[-1].get("burstiness") if batches else None,
        "avg_burstiness": avg_burstiness,
        "avg_transition_ratio": avg_transition
    }), 200
