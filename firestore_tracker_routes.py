"""
SEO Content Tracker Routes - v36.3 BRAJEN SEO Engine

ZMIANY v36.3:
- 🆕 FIRESTORE FIX: sanitize_for_firestore() dla kluczy ze znakami . / [ ]
- 🆕 Naprawiono ValueError: One or more components is not a string or is empty

v27.0 Original:
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

# ================================================================
# 🆕 v36.3: FIRESTORE KEY SANITIZATION
# ================================================================
def sanitize_for_firestore(data, depth=0, max_depth=50):
    """
    Recursively sanitize dictionary keys for Firestore compatibility.
    """
    if depth > max_depth:
        return data
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key is None:
                continue
            str_key = str(key).strip()
            if not str_key:
                continue
            safe_key = (str_key
                .replace('.', '_')
                .replace('/', '_')
                .replace('[', '(')
                .replace(']', ')')
                .replace('\\', '_')
                .replace('"', '')
                .replace("'", '')
            )
            if not safe_key:
                safe_key = f"_sanitized_key_{depth}"
            sanitized[safe_key] = sanitize_for_firestore(value, depth + 1, max_depth)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_firestore(item, depth + 1, max_depth) for item in data]
    else:
        return data

# 🆕 v36.2: Anti-Frankenstein System
try:
    from anti_frankenstein_integration import (
        update_project_after_batch,
        validate_batch_with_soft_caps
    )
    ANTI_FRANKENSTEIN_ENABLED = True
except ImportError:
    ANTI_FRANKENSTEIN_ENABLED = False
    print("[TRACKER] ⚠️ Anti-Frankenstein modules not available")

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
# v33.3: DELTA-S2 - Mierzy przyrost pokrycia encji po każdym batchu
# ============================================================================
def calculate_delta_s2(
    batch_text: str,
    accumulated_text: str,
    s1_entities: list,
    batch_number: int,
    total_batches: int = 7
) -> dict:
    """
    v33.3: Mierzy ile nowych encji z S1 zostało pokrytych w tym batchu.
    
    Pozwala śledzić czy "przyrost wiedzy" jest równomierny.
    
    Args:
        batch_text: Tekst bieżącego batcha
        accumulated_text: Tekst wszystkich poprzednich batchów
        s1_entities: Lista encji z S1 analysis
        batch_number: Numer bieżącego batcha (1-7)
        total_batches: Planowana liczba batchów
    
    Returns:
        Dict z: delta_entities, coverage_percent, on_track
    """
    if not s1_entities:
        return {
            "enabled": False,
            "message": "Brak encji z S1"
        }
    
    # Normalizuj encje do listy stringów
    entity_names = []
    for e in s1_entities:
        if isinstance(e, dict):
            name = e.get("entity", e.get("name", ""))
        else:
            name = str(e)
        if name:
            entity_names.append(name.lower())
    
    if not entity_names:
        return {
            "enabled": False,
            "message": "Brak nazw encji"
        }
    
    # Funkcja pomocnicza - które encje są pokryte w tekście
    def get_covered_entities(text: str) -> set:
        text_lower = text.lower()
        covered = set()
        for entity in entity_names:
            if entity in text_lower:
                covered.add(entity)
        return covered
    
    # Encje pokryte przed tym batchem
    covered_before = get_covered_entities(accumulated_text) if accumulated_text else set()
    
    # Encje pokryte po tym batchu
    full_text = (accumulated_text + "\n\n" + batch_text) if accumulated_text else batch_text
    covered_after = get_covered_entities(full_text)
    
    # Delta - nowe encje w tym batchu
    new_entities = covered_after - covered_before
    
    # Oblicz oczekiwany przyrost
    total_entities = len(entity_names)
    expected_per_batch = total_entities / total_batches
    expected_by_now = expected_per_batch * batch_number
    
    # Status
    coverage_percent = len(covered_after) / total_entities * 100 if total_entities > 0 else 0
    on_track = len(covered_after) >= (expected_by_now * 0.8)  # 80% oczekiwanego = OK
    
    # Pozostałe do pokrycia
    remaining = set(entity_names) - covered_after
    
    return {
        "enabled": True,
        "batch_number": batch_number,
        "delta_entities": list(new_entities),
        "delta_count": len(new_entities),
        "total_covered": len(covered_after),
        "total_entities": total_entities,
        "coverage_percent": round(coverage_percent, 1),
        "expected_by_now": round(expected_by_now, 1),
        "on_track": on_track,
        "remaining_entities": list(remaining)[:10],  # Max 10
        "remaining_count": len(remaining),
        "status": "OK" if on_track else "BEHIND",
        "message": f"Pokryto {len(covered_after)}/{total_entities} encji ({coverage_percent:.0f}%)" + 
                   ("" if on_track else f" - poniżej oczekiwań ({expected_by_now:.0f})")
    }


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
        
        # v27.2: Policz z EXCLUSIVE dla actual_uses (jak NeuronWriter!)
        # "spadek po rodzicach" liczy się TYLKO jako:
        #   +1 dla "spadek po rodzicach"
        # NIE liczy się jako +1 dla "spadek" (to byłoby OVERLAPPING)
        # 
        # EXCLUSIVE = longest-match-first, konsumuje tokeny
        # To jest zgodne z tym jak NeuronWriter liczy frazy
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
    
    # ================================================================
    # 🆕 v36.0: RESERVED KEYWORDS VALIDATION
    # Sprawdź czy batch używa fraz zarezerwowanych dla innych sekcji
    # ================================================================
    reserved_keyword_warnings = []
    semantic_plan = project_data.get("semantic_keyword_plan", {})
    if semantic_plan:
        current_batch_num = batches_done + 1  # Bo to jest batch który właśnie dodajemy
        batch_plans = semantic_plan.get("batch_plans", [])
        
        # Znajdź plan dla bieżącego batcha
        current_batch_plan = None
        for bp in batch_plans:
            if bp.get("batch_number") == current_batch_num:
                current_batch_plan = bp
                break
        
        if current_batch_plan:
            # Pobierz reserved_keywords dla tego batcha
            reserved_kws = current_batch_plan.get("reserved_keywords", [])
            assigned_kws = set([k.lower() for k in current_batch_plan.get("assigned_keywords", [])])
            universal_kws = set([k.lower() for k in current_batch_plan.get("universal_keywords", [])])
            
            # Sprawdź czy batch używa reserved keywords
            batch_text_lower = batch_text.lower()
            for reserved_info in reserved_kws:
                if isinstance(reserved_info, dict):
                    reserved_kw = reserved_info.get("keyword", "")
                    reserved_for_batch = reserved_info.get("reserved_for_batch", 0)
                    reserved_for_h2 = reserved_info.get("reserved_for_h2", "")
                else:
                    reserved_kw = str(reserved_info)
                    reserved_for_batch = 0
                    reserved_for_h2 = ""
                
                reserved_kw_lower = reserved_kw.lower()
                
                # Pomiń jeśli jest też w assigned lub universal
                if reserved_kw_lower in assigned_kws or reserved_kw_lower in universal_kws:
                    continue
                
                # Sprawdź czy fraza jest użyta w batchu
                if reserved_kw_lower and reserved_kw_lower in batch_text_lower:
                    reserved_keyword_warnings.append(
                        f"⚠️ RESERVED: '{reserved_kw}' jest zarezerwowana dla batcha {reserved_for_batch}"
                        + (f" ({reserved_for_h2})" if reserved_for_h2 else "")
                        + " - użyj jej tam gdzie pasuje tematycznie"
                    )
    
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
    
    # ================================================================
    # 🆕 v36.1: SEMANTIC PROXIMITY VALIDATION
    # Sprawdza czy frazy są otoczone kontekstem (nie w "próżni")
    # Używa danych z S1 zamiast concept_map
    # ================================================================
    isolated_keywords = []
    proximity_score = 100
    try:
        from semantic_proximity_validator import full_semantic_validation
        
        # Pobierz dane z S1
        s1_data_for_proximity = project_data.get("s1_data", {})
        entity_seo = s1_data_for_proximity.get("entity_seo", {})
        entities = entity_seo.get("entities", [])
        relationships = entity_seo.get("entity_relationships", [])
        
        # Zbuduj proximity_clusters z entity_relationships
        proximity_clusters = []
        for rel in relationships[:10]:
            if isinstance(rel, dict):
                subject = rel.get("subject", "")
                obj = rel.get("object", "")
                if subject and obj:
                    proximity_clusters.append({
                        "anchor": subject,
                        "must_have_nearby": [obj],
                        "max_distance": 30
                    })
        
        # Zbuduj supporting_entities z entities
        entity_names = []
        for e in entities[:20]:
            if isinstance(e, dict):
                entity_names.append(e.get("name", ""))
            else:
                entity_names.append(str(e))
        
        supporting_entities = {"all": [n for n in entity_names if n]}
        
        # Pobierz keywords do walidacji
        keywords_to_check = [
            meta.get("keyword", "") 
            for meta in keywords_state.values() 
            if meta.get("keyword") and meta.get("type", "BASIC").upper() in ["BASIC", "MAIN"]
        ][:15]  # Max 15 keywords
        
        if proximity_clusters or supporting_entities.get("all"):
            proximity_result = full_semantic_validation(
                text=batch_text,
                keywords=keywords_to_check,
                proximity_clusters=proximity_clusters,
                supporting_entities=supporting_entities
            )
            
            isolated_keywords = proximity_result.get("isolated_keywords", [])
            proximity_score = proximity_result.get("overall_score", 100)
            
            # Dodaj warnings dla izolowanych fraz (max 3)
            for isolated in isolated_keywords[:3]:
                keyword_name = isolated.get("keyword", isolated) if isinstance(isolated, dict) else isolated
                warnings.append(f"⚠️ ISOLATED: '{keyword_name}' - fraza bez kontekstu semantycznego")
            
            if proximity_score < 60:
                warnings.append(f"⚠️ Semantic proximity score: {proximity_score}/100 - dodaj więcej kontekstu")
                
            print(f"[TRACKER] 🔗 Semantic proximity: score={proximity_score}, isolated={len(isolated_keywords)}")
    except ImportError:
        print("[TRACKER] ⚠️ semantic_proximity_validator not available")
    except Exception as e:
        print(f"[TRACKER] Semantic proximity error: {e}")
    
    # v24.0: Walidacja pierwszego zdania (WAŻNE dla SEO)
    if first_sentence_warning:
        warnings.insert(0, first_sentence_warning)  # Na początku - ważne!
    
    # v24.0: Keyword stuffing warnings
    warnings.extend(stuffing_warnings)
    
    # 🆕 v36.0: Reserved keyword warnings (informacyjne)
    warnings.extend(reserved_keyword_warnings)
    
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
    
    # ================================================================
    # 🆕 v36.2: ANTI-FRANKENSTEIN - Update Memory & Style
    # ================================================================
    if ANTI_FRANKENSTEIN_ENABLED:
        try:
            batch_number = len(project_data.get("batches", []))
            h2_structure = project_data.get("h2_structure", [])
            
            # Wykryj H2 w tym batchu
            h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
            h2_sections = [(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]]
            if not h2_sections and h2_structure:
                # Fallback - użyj następnego nieużytego H2
                used_h2 = []
                for b in project_data.get("batches", [])[:-1]:
                    h2_match = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', b.get("text", ""), re.MULTILINE | re.IGNORECASE)
                    used_h2.extend([(m[0] or m[1]).strip() for m in h2_match if m[0] or m[1]])
                remaining = [h for h in h2_structure if h not in used_h2]
                h2_sections = remaining[:1] if remaining else []
            
            # Zbierz użyte encje
            entity_seo = project_data.get("s1_data", {}).get("entity_seo", {})
            all_entities = [e.get("name", "") for e in entity_seo.get("entities", [])]
            entities_used = [e for e in all_entities if e.lower() in batch_text.lower()]
            
            project_data = update_project_after_batch(
                project_data=project_data,
                batch_text=batch_text,
                batch_number=batch_number,
                h2_sections=h2_sections,
                entities_used=entities_used,
                humanness_score=100.0  # TODO: pobierz z ai_detection jeśli dostępne
            )
            print(f"[TRACKER] 🧟 Anti-Frankenstein updated: batch {batch_number}, memory claims: {len(project_data.get('article_memory', {}).get('key_claims', []))}")
        except Exception as e:
            print(f"[TRACKER] ⚠️ Anti-Frankenstein update error: {e}")
    
    # 🆕 v36.3: Sanitize before Firestore save
    try:
        project_data = sanitize_for_firestore(project_data)
    except Exception as e:
        print(f"[TRACKER] ⚠️ Sanitization warning: {e}")
    
    try:
        doc_ref.set(project_data)
    except Exception as e:
        print(f"[FIRESTORE] ⚠️ Błąd zapisu: {e}")

    # =========================================================================
    # v33.3: Przygotuj keywords_state_after do response
    # GPT od razu widzi aktualny stan keywords bez dodatkowego GET /status
    # =========================================================================
    keywords_state_after = {}
    for rid, meta in keywords_state.items():
        keywords_state_after[rid] = {
            "keyword": meta.get("keyword", ""),
            "type": meta.get("type", "BASIC"),
            "actual_uses": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min", 1),
            "target_max": meta.get("target_max", 999),
            "remaining_max": meta.get("remaining_max", 0),
            "status": meta.get("status", "UNDER")
        }

    # =========================================================================
    # v33.3: Delta-S2 - mierz przyrost pokrycia encji
    # =========================================================================
    delta_s2 = None
    try:
        s1_data = project_data.get("s1_data", {})
        entity_seo = s1_data.get("entity_seo", {})
        s1_entities = entity_seo.get("entities", [])
        
        # Zbuduj accumulated_text z poprzednich batchów (bez bieżącego)
        previous_batches = project_data.get("batches", [])[:-1]  # Bez właśnie dodanego
        accumulated_text = "\n\n".join([b.get("text", "") for b in previous_batches])
        
        # Oblicz batch_number
        batch_number = len(project_data.get("batches", []))
        total_batches = project_data.get("total_planned_batches", 7)
        
        if s1_entities:
            delta_s2 = calculate_delta_s2(
                batch_text=batch_text,
                accumulated_text=accumulated_text,
                s1_entities=s1_entities,
                batch_number=batch_number,
                total_batches=total_batches
            )
            print(f"[TRACKER] 📊 Delta-S2: +{delta_s2.get('delta_count', 0)} encji, total {delta_s2.get('coverage_percent', 0)}%")
    except Exception as e:
        print(f"[TRACKER] Delta-S2 error: {e}")
        delta_s2 = {"enabled": False, "error": str(e)}

    # ================================================================
    # 🆕 v36.2: SOFT CAP VALIDATION
    # ================================================================
    soft_cap_result = None
    if ANTI_FRANKENSTEIN_ENABLED:
        try:
            # Przekształć batch_counts na format {keyword: count}
            keyword_counts = {}
            for rid, count in batch_counts.items():
                meta = keywords_state.get(rid, {})
                keyword = meta.get("keyword", "")
                if keyword:
                    keyword_counts[keyword] = count
            
            soft_cap_result = validate_batch_with_soft_caps(
                batch_counts=keyword_counts,
                keywords_state=keywords_state,
                humanness_score=100.0,  # TODO: użyj rzeczywistego humanness jeśli dostępny
                total_batches=project_data.get("total_planned_batches", 7)
            )
            
            if soft_cap_result.get("available"):
                soft_exceeded = soft_cap_result.get("soft_exceeded", [])
                if soft_exceeded:
                    accepted = [e["keyword"] for e in soft_exceeded if e.get("accepted")]
                    rejected = [e["keyword"] for e in soft_exceeded if not e.get("accepted")]
                    if accepted:
                        print(f"[TRACKER] 🎚️ Soft cap: {len(accepted)} exceeded but accepted (natural text)")
                    if rejected:
                        print(f"[TRACKER] ⚠️ Soft cap: {len(rejected)} exceeded and rejected")
        except Exception as e:
            print(f"[TRACKER] ⚠️ Soft cap validation error: {e}")

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
        "keywords_state_after": keywords_state_after,  # v33.3: Aktualny stan keywords po zapisie batcha
        "delta_s2": delta_s2,  # v33.3: Przyrost pokrycia encji
        # 🆕 v36.1: Semantic proximity validation
        "semantic_proximity": {
            "score": proximity_score,
            "isolated_keywords": isolated_keywords[:5]
        },
        # 🆕 v36.2: Soft cap validation
        "soft_cap_validation": soft_cap_result,
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
    v28.1: Grammar validation before save + fallback z ostatniego preview.
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
    
    # v28.1: GRAMMAR VALIDATION - sprawdź przed zapisem (chyba że forced=true)
    if not forced:
        try:
            from grammar_middleware import validate_batch_full
            grammar_check = validate_batch_full(text)
            
            if not grammar_check["is_valid"]:
                print(f"[APPROVE_BATCH] ⚠️ Grammar issues found, returning for correction")
                return jsonify({
                    "saved": False,
                    "status": "NEEDS_CORRECTION",
                    "needs_correction": True,
                    "grammar": grammar_check["grammar"],
                    "banned_phrases": grammar_check["banned_phrases"],
                    "correction_prompt": grammar_check["correction_prompt"],
                    "instruction": "Popraw błędy i wyślij ponownie. Użyj forced=true aby zapisać mimo błędów.",
                    "hint": "Możesz też wywołać z 'forced': true aby wymusić zapis"
                }), 200  # 200, nie 400 - to nie jest błąd, to walidacja
                
        except ImportError:
            print(f"[APPROVE_BATCH] ⚠️ grammar_middleware not available, skipping validation")
        except Exception as e:
            print(f"[APPROVE_BATCH] ⚠️ Grammar check error: {e}, proceeding with save")

    result = process_batch_in_firestore(project_id, text, meta_trace, forced)

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    project_data = doc.to_dict() if doc.exists else None
    
    response = _minimal_batch_response(result, project_data)
    response["text_source"] = source
    response["grammar_validated"] = not forced  # v28.1
    
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
