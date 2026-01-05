"""
SEO Content Tracker Routes - v27.0 BRAJEN SEO Engine
+ v27.0: approve_batch fallback z last_preview
+ Minimal approve_batch response (~500B instead of 220KB)
+ Morfeusz2 lemmatization (with spaCy fallback)
+ Burstiness validation (3.2-3.8)
+ Transition words validation (25-50%)
+ v24.0: Per-batch keyword validation
+ v24.0: Fixed density limit (3.0% from seo_rules.json)
+ v24.0: Rozróżnienie EXCEEDED TOTAL vs per-batch warnings
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
from rapidfuzz import fuzz
from seo_optimizer import unified_prevalidation
from google.api_core.exceptions import InvalidArgument
import google.generativeai as genai
import os

# v24.0: Współdzielony model spaCy (oszczędność RAM)
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
except ImportError:
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")

# v23.8: Import polish_lemmatizer (Morfeusz2 + spaCy fallback)
try:
    from polish_lemmatizer import count_phrase_occurrences, get_backend_info, init_backend
    LEMMATIZER_ENABLED = True
    LEMMATIZER_BACKEND = init_backend()
    print(f"[TRACKER] Lemmatizer loaded: {LEMMATIZER_BACKEND}")
except ImportError as e:
    LEMMATIZER_ENABLED = False
    LEMMATIZER_BACKEND = "PREFIX"
    print(f"[TRACKER] Lemmatizer not available, using prefix matching: {e}")

# v24.0: Hierarchical keyword deduplication
try:
    from hierarchical_keyword_dedup import deduplicate_keyword_counts
    DEDUP_ENABLED = True
    print("[TRACKER] Hierarchical keyword dedup loaded")
except ImportError as e:
    DEDUP_ENABLED = False
    print(f"[TRACKER] Hierarchical dedup not available: {e}")

# v24.1: Semantic analyzer - wykrywa semantyczne pokrycie fraz
try:
    from semantic_analyzer import semantic_validation, find_semantic_gaps
    SEMANTIC_ENABLED = True
    print("[TRACKER] Semantic Analyzer loaded")
except ImportError as e:
    SEMANTIC_ENABLED = False
    print(f"[TRACKER] Semantic Analyzer not available: {e}")

tracker_routes = Blueprint("tracker_routes", __name__)

# v24.0: Density limit from seo_rules.json (was hardcoded 1.5%)
DENSITY_MAX = 3.0

# --- Gemini Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[TRACKER] ✅ Gemini API aktywny")
else:
    print("[TRACKER] ⚠️ Brak GEMINI_API_KEY")


# ============================================================================
# 1. COUNTING FUNCTIONS
# ============================================================================
def count_all_forms(text: str, keyword: str) -> int:
    """Liczy WSZYSTKIE odmiany słowa/frazy w tekście."""
    if not text or not keyword:
        return 0
    
    if LEMMATIZER_ENABLED:
        result = count_phrase_occurrences(text, keyword)
        return result.get("count", 0)
    
    # Fallback: prefix matching
    text_lower = text.lower()
    keyword_lower = keyword.lower().strip()
    words = keyword_lower.split()
    
    if len(words) == 1:
        word = words[0]
        stem = word[:6] if len(word) > 6 else word[:len(word)-1] if len(word) > 4 else word
        pattern = rf'\b{re.escape(stem)}\w*\b'
        return len(re.findall(pattern, text_lower))
    else:
        stems = []
        for word in words:
            if len(word) <= 3:
                stems.append(re.escape(word))
            elif len(word) <= 5:
                stems.append(re.escape(word[:len(word)-1]) + r'\w*')
            else:
                stems.append(re.escape(word[:5]) + r'\w*')
        pattern = r'\b' + r'\s+(?:\w+\s+){0,2}'.join(stems) + r'\b'
        return len(re.findall(pattern, text_lower))


# ============================================================================
# 2. VALIDATIONS
# ============================================================================
def validate_structure(text):
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Markdown (##) zabroniony — użyj h2:"}
    banned = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie"]
    for h2 in re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL):
        if any(b in h2.lower() for b in banned):
            return {"valid": False, "error": f"❌ Niedozwolony nagłówek: '{h2.strip()}'"}
    return {"valid": True}


def calculate_burstiness(text):
    """Target: 3.2-3.8"""
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
    return round(raw_score * 5, 2)


TRANSITION_WORDS_PL = [
    "również", "także", "ponadto", "dodatkowo", "co więcej",
    "jednak", "jednakże", "natomiast", "ale", "z drugiej strony",
    "mimo to", "niemniej", "pomimo", "choć", "chociaż",
    "dlatego", "w związku z tym", "w rezultacie", "ponieważ",
    "zatem", "więc", "stąd", "w konsekwencji",
    "na przykład", "przykładowo", "między innymi", "np.",
    "po pierwsze", "po drugie", "następnie", "potem", "na koniec",
]


def calculate_transition_score(text: str) -> dict:
    """Target: 25-50% zdań z transition words"""
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences), "warnings": []}
    
    transition_count = sum(1 for s in sentences if any(tw in s.lower()[:100] for tw in TRANSITION_WORDS_PL))
    ratio = transition_count / len(sentences)
    
    warnings = []
    if ratio < 0.20:
        warnings.append(f"⚠️ Za mało transition words: {ratio:.0%} (min 25%)")
    elif ratio > 0.55:
        warnings.append(f"⚠️ Za dużo transition words: {ratio:.0%} (max 50%)")
    
    return {"ratio": round(ratio, 3), "count": transition_count, "total": len(sentences), "warnings": warnings}


def validate_metrics(burstiness: float, transition_data: dict, density: float) -> list:
    """Waliduje metryki"""
    warnings = []
    
    if burstiness < 3.2:
        warnings.append(f"⚠️ Burstiness za niski: {burstiness} (min 3.2)")
    elif burstiness > 3.8:
        warnings.append(f"⚠️ Burstiness za wysoki: {burstiness} (max 3.8)")
    
    # v24.0: Fixed density limit (was 1.5%, now 3.0% from seo_rules.json)
    if density > DENSITY_MAX:
        warnings.append(f"⚠️ Keyword density za wysoka: {density}% (max {DENSITY_MAX}%)")
    
    warnings.extend(transition_data.get("warnings", []))
    return warnings


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
    
    # v24.2: UNIFIED COUNTING - jedna funkcja dla całego systemu
    # Zastępuje: count_all_forms + deduplicate_keyword_counts + stuffing detection
    try:
        from keyword_counter import count_keywords_for_state, get_stuffing_warnings, count_keywords
        
        # Zbierz keywordy do analizy szczegółowej
        keywords = [meta.get("keyword", "").strip() for meta in keywords_state.values() if meta.get("keyword")]
        
        # v26.1: Policz z OVERLAPPING dla actual_uses (każda fraza osobno)
        # "uzależnienie od narkotyków" liczy się jako:
        #   +1 dla "uzależnienie od narkotyków"
        #   +1 dla "uzależnienie" (bo jest zawarte)
        # To jest zgodne z tym jak Google/Surfer/Neuron liczą frazy
        batch_counts = count_keywords_for_state(batch_text, keywords_state, use_exclusive_for_nested=False)
        
        # Stuffing warnings (zintegrowane z tym samym licznikiem)
        stuffing_warnings = get_stuffing_warnings(batch_text, keywords_state)
        
        # Szczegóły do diagnostyki
        full_result = count_keywords(batch_text, keywords)
        in_headers = full_result.get("in_headers", {})
        in_intro = full_result.get("in_intro", {})
        
        UNIFIED_COUNTING = True
    except ImportError as e:
        print(f"[TRACKER] keyword_counter not available, using legacy: {e}")
        UNIFIED_COUNTING = False
        
        # LEGACY FALLBACK - stara metoda
        clean_text_original = re.sub(r"<[^>]+>", " ", batch_text)
        clean_text_original = re.sub(r"\s+", " ", clean_text_original)
        
        batch_counts = {}
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "").strip()
            if not keyword:
                batch_counts[rid] = 0
                continue
            batch_counts[rid] = count_all_forms(clean_text_original, keyword)
        
        # Legacy deduplikacja
        if DEDUP_ENABLED:
            raw_counts = {meta.get("keyword", ""): batch_counts.get(rid, 0) 
                          for rid, meta in keywords_state.items() if meta.get("keyword")}
            adjusted = deduplicate_keyword_counts(raw_counts)
            for rid, meta in keywords_state.items():
                kw = meta.get("keyword", "")
                if kw in adjusted:
                    batch_counts[rid] = adjusted[kw]
        
        # Legacy stuffing
        stuffing_warnings = []
        paragraphs = batch_text.split('\n\n')
        for rid, meta in keywords_state.items():
            if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
                continue
            keyword = meta.get("keyword", "").lower()
            if not keyword:
                continue
            for para in paragraphs:
                if para.lower().count(keyword) > 3:
                    stuffing_warnings.append(f"⚠️ '{meta.get('keyword')}' występuje >3x w jednym akapicie")
                    break
        
        in_headers = {}
        in_intro = {}
    
    # v24.0: Walidacja pierwszego zdania (dla INTRO batcha)
    batches_done = len(project_data.get("batches", []))
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    first_sentence_warning = None
    
    if batches_done == 0 and main_keyword:  # To jest INTRO
        first_sentence = batch_text.split('.')[0] if batch_text else ""
        if main_keyword.lower() not in first_sentence.lower():
            first_sentence_warning = f"⚠️ Pierwsze zdanie nie zawiera głównej frazy '{main_keyword}' - kluczowe dla featured snippet!"
    
    # v24.0: Pobierz info o batchach do walidacji per-batch
    total_batches = project_data.get("total_planned_batches", 4)
    remaining_batches = max(1, total_batches - batches_done)
    
    # v24.0: Per-batch warnings (informacyjne, nie blokują)
    per_batch_warnings = []
    for rid, batch_count in batch_counts.items():
        if batch_count == 0:
            continue
        meta = keywords_state[rid]
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        keyword = meta.get("keyword", "")
        target_max = meta.get("target_max", 999)
        actual = meta.get("actual_uses", 0)
        remaining_to_max = max(0, target_max - actual)
        
        # Oblicz suggested per batch
        if remaining_to_max > 0 and remaining_batches > 0:
            suggested = math.ceil(remaining_to_max / remaining_batches)
        else:
            suggested = 0
        
        # Warning jeśli batch_count > suggested * 1.5 (ale nie blokuje)
        if suggested > 0 and batch_count > suggested * 1.5:
            per_batch_warnings.append(
                f"ℹ️ '{keyword}': użyto {batch_count}x w batchu (sugerowano ~{suggested}x). "
                f"Zostało {max(0, remaining_to_max - batch_count)}/{target_max} dla artykułu."
            )
    
    # Check EXCEEDED TOTAL (całkowity limit artykułu - to jest KRYTYCZNE)
    exceeded_keywords = []
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
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
    
    # Update keywords state
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
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
        else:
            meta["status"] = "OVER"
        
        if meta.get("type", "BASIC").upper() == "BASIC":
            meta["remaining_max"] = max(0, max_t - actual)
        
        keywords_state[rid] = meta

    # Prevalidation
    precheck = unified_prevalidation(batch_text, keywords_state)
    warnings = precheck.get("warnings", [])
    semantic_score = precheck.get("semantic_score", 1.0)
    density = precheck.get("density", 0.0)
    
    # v24.1: Semantic validation - czy frazy są semantycznie pokryte
    semantic_gaps = []
    if SEMANTIC_ENABLED:
        try:
            sem_result = semantic_validation(batch_text, keywords_state, min_coverage=0.4)
            if sem_result.get("semantic_enabled"):
                semantic_gaps = sem_result.get("gaps", [])
                overall_coverage = sem_result.get("overall_coverage", 1.0)
                if overall_coverage < 0.4:
                    warnings.append(f"⚠️ Semantyczne pokrycie {overall_coverage:.0%} < 40% - rozwiń tematy: {', '.join(semantic_gaps[:3])}")
                elif semantic_gaps:
                    # Info, nie warning - są luki ale ogólne pokrycie OK
                    pass
        except Exception as e:
            print(f"[TRACKER] Semantic validation error: {e}")
    
    # v24.0: Walidacja pierwszego zdania (WAŻNE dla SEO)
    if first_sentence_warning:
        warnings.insert(0, first_sentence_warning)  # Na początku - ważne!
    
    # v24.0: Keyword stuffing warnings
    warnings.extend(stuffing_warnings)
    
    # v24.0: EXCEEDED TOTAL warnings (KRYTYCZNE - przekroczono limit całkowity)
    for ek in exceeded_keywords:
        warnings.append(f"❌ EXCEEDED TOTAL: '{ek['keyword']}' = {ek['would_be']}x (limit {ek['target_max']}x dla CAŁEGO artykułu)")
    
    # Metrics
    burstiness = calculate_burstiness(batch_text)
    transition_data = calculate_transition_score(batch_text)
    metrics_warnings = validate_metrics(burstiness, transition_data, density)
    warnings.extend(metrics_warnings)

    struct_check = validate_structure(batch_text)
    valid_struct = struct_check["valid"]

    status = "APPROVED"
    if warnings or not valid_struct:
        status = "WARN"
    if forced:
        status = "FORCED"
    
    # v24.0: Jeśli tylko per_batch warnings (nie EXCEEDED TOTAL) - status APPROVED
    has_critical = any("EXCEEDED TOTAL" in w for w in warnings)
    has_density_issue = any("density" in w.lower() and density > DENSITY_MAX for w in warnings)
    if not has_critical and not has_density_issue and not exceeded_keywords:
        status = "APPROVED"

    # Save batch
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness,
        "transition_ratio": transition_data.get("ratio", 0),
        "batch_counts": batch_counts,  # v24.0: zapisuj counts dla debug
        "per_batch_info": per_batch_warnings,  # v24.0: info per batch
        "semantic_gaps": semantic_gaps,  # v24.1: luki semantyczne
        "language_audit": {
            "semantic_score": semantic_score,
            "density": density,
            "burstiness": burstiness
        },
        "warnings": warnings,
        "status": status
    }

    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    try:
        doc_ref.set(project_data)
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Błąd zapisu: {e}")

    return {
        "status": status,
        "semantic_score": semantic_score,
        "density": density,
        "burstiness": burstiness,
        "warnings": warnings,
        "per_batch_warnings": per_batch_warnings,  # v24.0: osobno
        "semantic_gaps": semantic_gaps,  # v24.1: frazy bez pokrycia
        "exceeded_keywords": exceeded_keywords,
        "batch_counts": batch_counts,
        "unified_counting": UNIFIED_COUNTING if 'UNIFIED_COUNTING' in dir() else False,  # v24.2
        "in_headers": in_headers if 'in_headers' in dir() else {},  # v24.2: frazy w H2/H3
        "in_intro": in_intro if 'in_intro' in dir() else {},  # v24.2: frazy w intro
        "status_code": 200
    }


# ============================================================================
# 4. MINIMAL RESPONSE (v24.0 - rozróżnia EXCEEDED TOTAL vs per-batch)
# ============================================================================
def _minimal_batch_response(result: dict, project_data: dict = None) -> dict:
    """
    v24.0: Rozróżnia EXCEEDED TOTAL (blokuje) vs per-batch warnings (info).
    """
    problems = []  # Krytyczne - wymagają reakcji
    info = []  # Informacyjne - można zignorować
    
    # EXCEEDED TOTAL - KRYTYCZNE
    exceeded = result.get("exceeded_keywords", [])
    for ex in exceeded:
        problems.append(f"❌ '{ex['keyword']}' PRZEKROCZYŁA CAŁKOWITY LIMIT ({ex['would_be']}/{ex['target_max']})")
    
    # Per-batch warnings - tylko INFO
    for w in result.get("per_batch_warnings", []):
        info.append(w)
    
    # Inne ważne warnings (density)
    for w in result.get("warnings", []):
        if "EXCEEDED TOTAL" in str(w):
            if w not in problems:
                problems.append(w)
        elif "density" in str(w).lower():
            problems.append(w)
    
    # Status
    status = "OK"
    if problems:
        status = "WARN"
    if result.get("status") == "FORCED":
        status = "FORCED"
    # v24.0: Jeśli status APPROVED z process_batch - zachowaj
    if result.get("status") == "APPROVED":
        status = "OK"
    
    # Batch info
    batch_number = 1
    remaining_batches = 0
    if project_data:
        batches_done = len(project_data.get("batches", []))
        batches_planned = len(project_data.get("batches_plan", [])) or project_data.get("total_planned_batches", 4)
        batch_number = batches_done
        remaining_batches = max(0, batches_planned - batches_done)
    
    # Next action
    if exceeded:
        next_action = {
            "action": "ask_user",
            "question": f"Przekroczono CAŁKOWITY limit dla {len(exceeded)} fraz. A) Przepisać batch B) Kontynuować (forced)?"
        }
    elif remaining_batches > 0:
        next_action = {
            "action": "continue",
            "call": "GET /pre_batch_info → pisz kolejny batch"
        }
    else:
        next_action = {
            "action": "review",
            "call": "POST /editorial_review → oceń całość"
        }
    
    response = {
        "saved": True,
        "batch": batch_number,
        "status": status,
        "next": next_action,
        "remaining_batches": remaining_batches
    }
    
    # v24.0: Osobno problems (krytyczne) i info (per-batch)
    if problems:
        response["problems"] = problems
    if info:
        response["info"] = info  # Per-batch to tylko info
    
    return response

@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    v27.0: NAPRAWIONE - akceptuje różne nazwy pól tekstu + fallback z ostatniego preview.
    Obsługuje: corrected_text, text, content, batch_text
    Fallback: Pobiera tekst z ostatniego preview jeśli nie wysłano w body.
    """
    data = request.get_json(force=True) if request.is_json else {}
    
    # v27.0: Próbuj różne nazwy pól
    text = None
    source = None
    for field in ["corrected_text", "text", "content", "batch_text"]:
        if field in data and data[field]:
            text = data[field].strip()
            source = f"body.{field}"
            print(f"[APPROVE_BATCH] Znaleziono tekst w polu '{field}' ({len(text)} znaków)")
            break
    
    # v27.0: FALLBACK - pobierz z ostatniego preview jeśli brak tekstu
    if not text:
        print(f"[APPROVE_BATCH] ⚠️ Brak tekstu w body, próbuję fallback z last_preview...")
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        
        if doc.exists:
            project_data = doc.to_dict()
            last_preview = project_data.get("last_preview", {})
            preview_text = last_preview.get("text", "")
            
            if preview_text:
                text = preview_text.strip()
                source = "fallback.last_preview"
                print(f"[APPROVE_BATCH] ✅ Fallback OK - użyto tekstu z last_preview ({len(text)} znaków)")
            else:
                # Próbuj też z ostatniego batcha w trybie "approve again"
                batches = project_data.get("batches", [])
                if batches:
                    last_batch_text = batches[-1].get("text", "")
                    if last_batch_text:
                        text = last_batch_text.strip()
                        source = "fallback.last_batch"
                        print(f"[APPROVE_BATCH] ✅ Fallback OK - użyto tekstu z ostatniego batcha ({len(text)} znaków)")
    
    if not text:
        return jsonify({
            "error": "No text provided",
            "hint": "Wyślij tekst w polu 'corrected_text' lub 'text'. Możesz też najpierw wywołać preview_batch.",
            "received_fields": list(data.keys()),
            "fallback_tried": True,
            "fallback_failed": "Brak last_preview w projekcie"
        }), 400
    
    meta_trace = data.get("meta_trace", {})
    if source:
        meta_trace["text_source"] = source
    forced = data.get("forced", False)

    result = process_batch_in_firestore(project_id, text, meta_trace, forced)

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    project_data = doc.to_dict() if doc.exists else None
    
    response = _minimal_batch_response(result, project_data)
    response["text_source"] = source
    
    return jsonify(response), 200


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
            "type": meta.get("type", "BASIC"),
            "actual": meta.get("actual_uses", 0),
            "target": f"{meta.get('target_min', 0)}-{meta.get('target_max', 999)}",
            "status": meta.get("status"),
            "remaining": meta.get("remaining_max", 0)
        })
    
    return jsonify({
        "project_id": project_id,
        "keywords": stats,
        "batches": len(batches)
    }), 200


@tracker_routes.delete("/api/project/<project_id>")
def delete_project(project_id):
    """Usuwa projekt z Firestore."""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    doc_ref.delete()
    return jsonify({"status": "DELETED", "project_id": project_id}), 200


@tracker_routes.post("/api/project/<project_id>/reset")
def reset_project(project_id):
    """Resetuje projekt - usuwa batche, zeruje keywords."""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    doc_ref.update({
        "batches": [],
        "final_review": None,
        "keywords_state": {
            rid: {**meta, "actual_uses": 0, "status": "UNDER", "remaining_max": meta.get("target_max", 999)}
            for rid, meta in project_data.get("keywords_state", {}).items()
        }
    })
    
    return jsonify({"status": "RESET", "project_id": project_id}), 200
