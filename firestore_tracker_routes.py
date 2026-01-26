"""
SEO Content Tracker Routes - v37.2 BRAJEN SEO Engine

ZMIANY v37.2:
- 🆕 CLAUDE SMART-FIX: inteligentne poprawki z pre_batch_info
- 🆕 POST /claude_smart_fix - Claude poprawia batch z kontekstem
- 🆕 POST /review_and_fix - auto-fix + opcjonalnie Claude
- 🆕 Claude wie: które frazy UNDER (dodaj), EXCEEDED (zamień), H2, spójność

ZMIANY v37.1:
- 🆕 MOE BATCH VALIDATOR: 4 ekspertów walidujących każdy batch
  - STRUCTURE: różna liczba akapitów, anty-monotonność
  - SEO: BASIC/EXTENDED keywords, encje, n-gramy
  - LANGUAGE: gramatyka polska (LanguageTool)
  - AI DETECTION: burstiness, TTR, rozkład zdań
- 🆕 moe_validation w response i batch_entry
- 🆕 moe_fix_instructions z konkretnymi poprawkami

ZMIANY v37.0:
- 🆕 EXCEEDED KEYWORDS: podział na WARNING (1-49%) i CRITICAL (50%+)
- 🆕 BLOKADA BATCHA przy exceeded 50%+ (chyba że forced=True)
- 🆕 SYNONIMY: automatyczne pobieranie synonimów dla exceeded keywords
- 🆕 FIX INSTRUCTIONS: konkretne synonimy do przepisania batcha
- 🆕 Tylko BASIC/MAIN są pilnowane - EXTENDED mogą być przekroczone

ZMIANY v36.3:
- 🆕 FIRESTORE FIX: sanitize_for_firestore() dla kluczy ze znakami . / [ ]

v27.0 Original:
+ Morfeusz2 lemmatization (with spaCy fallback)
+ Burstiness validation (3.2-3.8)
+ Transition words validation (25-50%)
+ v24.0: Per-batch keyword validation
+ v24.0: Fixed density limit (3.0% from seo_rules.json)
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

# 🆕 v37.0: Keyword Synonyms dla exceeded keywords
try:
    from keyword_synonyms import get_synonyms, generate_synonyms_prompt_section
    SYNONYMS_ENABLED = True
    print("[TRACKER] ✅ Keyword Synonyms loaded")
except ImportError as e:
    SYNONYMS_ENABLED = False
    print(f"[TRACKER] ⚠️ Keyword Synonyms not available: {e}")
    def get_synonyms(keyword, max_synonyms=4):
        return []

# 🆕 v37.1: Batch Review System z auto-poprawkami
try:
    from batch_review_system import (
        review_batch_comprehensive,
        get_review_summary,
        generate_claude_fix_prompt,
        claude_smart_fix,
        should_use_claude_smart_fix,
        build_smart_fix_prompt,
        get_pre_batch_info_for_claude,
        SmartFixResult,
        ReviewResult
    )
    BATCH_REVIEW_ENABLED = True
    print("[TRACKER] ✅ Batch Review System loaded")
except ImportError as e:
    BATCH_REVIEW_ENABLED = False
    print(f"[TRACKER] ⚠️ Batch Review System not available: {e}")

# 🆕 v37.1: MoE Batch Validator
try:
    from moe_batch_validator import validate_batch_moe, format_validation_for_gpt, ValidationMode
    MOE_VALIDATOR_ENABLED = True
    print("[TRACKER] ✅ MoE Batch Validator loaded")
except ImportError as e:
    MOE_VALIDATOR_ENABLED = False
    print(f"[TRACKER] ⚠️ MoE Validator not available: {e}")

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
# 🆕 v36.9: AUTO-MERGE FULL ARTICLE
# ============================================================================
def auto_merge_full_article(project_id: str, project_data: dict) -> dict:
    """
    🆕 v36.9: Automatycznie scala wszystkie batche w full_article.
    Wywoływane po zapisaniu ostatniego batcha.
    """
    try:
        batches = project_data.get("batches", [])
        if not batches:
            return {"merged": False, "reason": "No batches to merge"}
        
        # Scala wszystkie batche
        full_content_parts = []
        total_words = 0
        h2_count = 0
        
        for batch in batches:
            text = batch.get("text", "")
            if text:
                full_content_parts.append(text.strip())
                total_words += len(text.split())
                h2_count += len(re.findall(r'(?:^h2:|<h2)', text, re.MULTILINE | re.IGNORECASE))
        
        full_content = "\n\n".join(full_content_parts)
        
        # Zapisz do projektu
        project_data["full_article"] = {
            "content": full_content,
            "word_count": total_words,
            "h2_count": h2_count,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "auto_merged": True,
            "batch_count": len(batches)
        }
        
        # Zapisz do Firestore
        db = firestore.client()
        doc_ref = db.collection("seo_projects").document(project_id)
        doc_ref.set(project_data, merge=True)
        
        print(f"[AUTO-MERGE] ✅ Full article merged: {total_words} words, {h2_count} H2, {len(batches)} batches")
        
        return {
            "merged": True,
            "word_count": total_words,
            "h2_count": h2_count,
            "batch_count": len(batches)
        }
        
    except Exception as e:
        print(f"[AUTO-MERGE] ❌ Error: {e}")
        return {"merged": False, "error": str(e)}


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
    
    # ================================================================
    # 🆕 v37.0: EXCEEDED KEYWORDS z podziałem na WARNING i CRITICAL
    # WARNING: exceeded 1-49% → batch zapisuje się + ostrzeżenie + synonimy
    # CRITICAL: exceeded 50%+ → batch ZABLOKOWANY + synonimy do przepisania
    # ================================================================
    exceeded_warning = []   # 1-49% over max
    exceeded_critical = []  # 50%+ over max
    
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        
        # Tylko BASIC i MAIN - EXTENDED pomijamy (mogą być przekroczone)
        if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
            continue
        
        keyword = meta.get("keyword", "")
        current = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        new_total = current + batch_count
        
        if new_total > target_max:
            exceeded_by = new_total - target_max
            exceed_percent = (exceeded_by / target_max * 100) if target_max > 0 else 100
            
            # Pobierz synonimy
            synonyms = get_synonyms(keyword) if SYNONYMS_ENABLED else []
            
            exceeded_info = {
                "keyword": keyword,
                "current": current,
                "batch_uses": batch_count,
                "would_be": new_total,
                "target_max": target_max,
                "exceeded_by": exceeded_by,
                "exceed_percent": round(exceed_percent),
                "synonyms": synonyms[:3]
            }
            
            if exceed_percent >= 50:
                # 50%+ przekroczenia → CRITICAL (blokada)
                exceeded_critical.append(exceeded_info)
                print(f"[TRACKER] ❌ CRITICAL: '{keyword}' exceeded by {exceed_percent:.0f}% (50%+ = BLOCK)")
            else:
                # 1-49% przekroczenia → WARNING (ostrzeżenie)
                exceeded_warning.append(exceeded_info)
                print(f"[TRACKER] ⚠️ WARNING: '{keyword}' exceeded by {exceed_percent:.0f}% (<50% = warn only)")
    
    # Backward compatibility: exceeded_keywords = wszystkie exceeded
    exceeded_keywords = exceeded_warning + exceeded_critical
    
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

    # ================================================================
    # 🆕 v37.0: BLOKADA przy exceeded_critical (50%+) - NIE zapisuj batcha!
    # ================================================================
    if exceeded_critical and not forced:
        # Generuj mapę synonimów do przepisania
        synonyms_map = {}
        details = []
        for ek in exceeded_critical:
            kw = ek["keyword"]
            syns = ek.get("synonyms", [])
            synonyms_map[kw] = syns
            syn_text = f" → użyj: {', '.join(syns[:2])}" if syns else " → pomiń tę frazę"
            details.append(
                f"'{kw}': {ek['would_be']}/{ek['target_max']} "
                f"(+{ek['exceed_percent']}%){syn_text}"
            )
        
        print(f"[TRACKER] ❌ BATCH REJECTED: {len(exceeded_critical)} fraz BASIC exceeded 50%+")
        
        return {
            "status": "REJECTED",
            "saved": False,
            "reason": "BASIC_EXCEEDED_50_PERCENT",
            "message": f"❌ Batch odrzucony: {len(exceeded_critical)} fraz BASIC przekroczyło max o 50%+",
            "exceeded_critical": exceeded_critical,
            "exceeded_warning": exceeded_warning,
            "fix_instruction": "Przepisz batch UŻYWAJĄC SYNONIMÓW zamiast przekroczonych fraz:",
            "synonyms_map": synonyms_map,
            "details": details,
            "tip": "Możesz też wysłać z forced=True aby zaakceptować mimo przekroczenia"
        }
    
    # 🆕 v37.0: Dodaj warnings dla exceeded_warning (1-49%)
    for ew in exceeded_warning:
        syn_hint = f" → rozważ synonimy: {', '.join(ew['synonyms'][:2])}" if ew.get('synonyms') else ""
        warnings.append(
            f"⚠️ EXCEEDED: '{ew['keyword']}' = {ew['would_be']}/{ew['target_max']} "
            f"(+{ew['exceed_percent']}%){syn_hint}"
        )

    # ================================================================
    # 🆕 v37.1: MoE BATCH VALIDATOR - Kompleksowa walidacja
    # ================================================================
    moe_validation_result = None
    moe_fix_instructions = []
    
    if MOE_VALIDATOR_ENABLED:
        try:
            batches_done = len(project_data.get("batches", []))
            moe_validation_result = validate_batch_moe(
                batch_text=batch_text,
                project_data=project_data,
                batch_number=batches_done + 1,
                mode=ValidationMode.SOFT  # SOFT = warnings nie blokują
            )
            
            # Zbierz fix instructions
            moe_fix_instructions = moe_validation_result.fix_instructions
            
            # Dodaj MoE warnings do ogólnych warnings
            for issue in moe_validation_result.issues:
                if issue.severity == "critical":
                    warnings.append(f"🔍 MOE/{issue.expert}: {issue.message}")
                elif issue.severity == "warning":
                    warnings.append(f"ℹ️ MOE/{issue.expert}: {issue.message}")
            
            # Log
            critical_count = len([i for i in moe_validation_result.issues if i.severity == "critical"])
            warning_count = len([i for i in moe_validation_result.issues if i.severity == "warning"])
            print(f"[TRACKER] 🔍 MoE Validation: {moe_validation_result.status} "
                  f"({critical_count} critical, {warning_count} warnings)")
            
            # Sprawdź strukturę (różna liczba akapitów)
            structure_summary = moe_validation_result.experts_summary.get("structure", {})
            if structure_summary:
                para_count = structure_summary.get("paragraph_count", 0)
                prev_counts = structure_summary.get("previous_counts", [])
                print(f"[TRACKER] 📊 Structure: {para_count} paragraphs (previous: {prev_counts[-3:] if prev_counts else 'none'})")
                
        except Exception as e:
            print(f"[TRACKER] ⚠️ MoE Validation error: {e}")

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
        "status": status,
        # 🆕 v37.1: MoE Validation results
        "moe_validation": moe_validation_result.to_dict() if moe_validation_result else None,
        "moe_fix_instructions": moe_fix_instructions
    }

    project_data.setdefault("batches", []).append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    # ================================================================
    # 🆕 v36.2: ANTI-FRANKENSTEIN - Update Memory & Style
    # ================================================================
    # 🆕 v36.4: Calculate real humanness_score for soft caps
    real_humanness_score = 100.0
    try:
        from ai_detection_metrics import calculate_humanness_score
        humanness_result = calculate_humanness_score(batch_text)
        real_humanness_score = float(humanness_result.get("humanness_score", 100.0))
        print(f"[TRACKER] 🎯 Humanness score calculated: {real_humanness_score}")
    except ImportError:
        print("[TRACKER] ⚠️ ai_detection_metrics not available, using default humanness=100")
    except Exception as e:
        print(f"[TRACKER] ⚠️ Humanness calculation error: {e}, using default=100")
    
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
                humanness_score=real_humanness_score  # 🆕 v36.4: Real humanness!
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
        print(f"[FIRESTORE] ✅ Batch saved successfully, total batches: {len(project_data.get('batches', []))}")
    except Exception as e:
        print(f"[FIRESTORE] ❌ CRITICAL: Błąd zapisu: {e}")
        # 🆕 v36.3: Return error instead of silently continuing!
        return {
            "status": "ERROR",
            "error": f"Firestore save failed: {str(e)}",
            "message": "Batch was NOT saved. Please retry."
        }

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
                humanness_score=real_humanness_score,  # 🆕 v36.4: Real humanness!
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

    # ================================================================
    # 🆕 v37.1: BATCH REVIEW + AUTO-FIX
    # ================================================================
    batch_review_result = None
    auto_fixed_text = None
    
    if BATCH_REVIEW_ENABLED:
        try:
            # Pobierz poprzedni batch dla sprawdzenia spójności
            previous_batches = project_data.get("batches", [])
            previous_batch_text = previous_batches[-2]["text"] if len(previous_batches) > 1 else None
            
            # Uruchom comprehensive review
            batch_review_result = review_batch_comprehensive(
                batch_text=batch_text,
                keywords_state=keywords_state,
                batch_counts=batch_counts,
                previous_batch_text=previous_batch_text,
                auto_fix=True,  # Włącz auto-poprawki
                use_claude_for_complex=False  # Na razie bez Claude
            )
            
            # Jeśli były auto-poprawki, zaktualizuj batch w Firestore
            if batch_review_result.fixed_text and batch_review_result.fixed_text != batch_text:
                auto_fixed_text = batch_review_result.fixed_text
                
                # Zaktualizuj ostatni batch w project_data
                if project_data.get("batches"):
                    project_data["batches"][-1]["text"] = auto_fixed_text
                    project_data["batches"][-1]["auto_fixed"] = True
                    project_data["batches"][-1]["auto_fixes"] = batch_review_result.auto_fixes_applied
                    
                    # Zapisz do Firestore
                    doc_ref.update({
                        "batches": sanitize_for_firestore(project_data["batches"])
                    })
                
                print(f"[TRACKER] ✅ Auto-fixed {len(batch_review_result.auto_fixes_applied)} issues")
                for fix in batch_review_result.auto_fixes_applied[:3]:
                    print(f"[TRACKER]    • {fix}")
            
            # Loguj review summary
            print(f"[TRACKER] 📋 Review: {batch_review_result.status}, issues: {len(batch_review_result.issues)}")
            
        except Exception as e:
            print(f"[TRACKER] ⚠️ Batch review error: {e}")

    # ================================================================
    # 🆕 v36.9: AUTO-MERGE po ostatnim batchu
    # ================================================================
    batches_done = len(project_data.get("batches", []))
    total_planned = project_data.get("total_planned_batches", 4)
    remaining_batches = max(0, total_planned - batches_done)
    is_last_batch = (remaining_batches == 0)
    
    auto_merge_result = None
    if is_last_batch:
        print(f"[TRACKER] 🏁 Last batch detected! Auto-merging full article...")
        auto_merge_result = auto_merge_full_article(project_id, project_data)

    return {
        "status": status,
        "semantic_score": semantic_score,
        "density": density,
        "burstiness": burstiness,
        "warnings": warnings,
        "per_batch_warnings": per_batch_warnings,
        "semantic_gaps": semantic_gaps,
        "exceeded_keywords": exceeded_keywords,
        # 🆕 v37.0: Rozdzielone exceeded na warning i critical
        "exceeded_warning": exceeded_warning,
        "exceeded_critical": exceeded_critical,
        "batch_counts": batch_counts,
        "unified_counting": UNIFIED_COUNTING if 'UNIFIED_COUNTING' in dir() else False,
        "in_headers": in_headers if 'in_headers' in dir() else {},
        "in_intro": in_intro if 'in_intro' in dir() else {},
        "keywords_state_after": keywords_state_after,
        "delta_s2": delta_s2,
        "semantic_proximity": {
            "score": proximity_score,
            "isolated_keywords": isolated_keywords[:5]
        },
        "soft_cap_validation": soft_cap_result,
        # 🆕 v37.1: MoE Validation
        "moe_validation": moe_validation_result.to_dict() if moe_validation_result else None,
        "moe_fix_instructions": moe_fix_instructions,
        # 🆕 v37.1: Batch Review + Auto-Fix
        "batch_review": batch_review_result.to_dict() if batch_review_result else None,
        "auto_fixed": auto_fixed_text is not None,
        "auto_fixed_text": auto_fixed_text,
        "claude_fixes_needed": batch_review_result.claude_fixes_needed if batch_review_result else [],
        # 🆕 v36.9: Info o postępie i auto-merge
        "batch_number": batches_done,
        "total_planned_batches": total_planned,
        "remaining_batches": remaining_batches,
        "is_last_batch": is_last_batch,
        "auto_merge": auto_merge_result,
        "article_complete": is_last_batch and auto_merge_result and auto_merge_result.get("merged", False),
        "export_ready": is_last_batch,
        "next_step": "GET /api/project/{id}/export/docx" if is_last_batch else f"Continue with batch {batches_done + 1}",
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


# ============================================================================
# 🆕 v37.1: CLAUDE SMART-FIX ENDPOINT
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/claude_smart_fix")
def claude_smart_fix_endpoint(project_id):
    """
    🤖 Claude poprawia batch INTELIGENTNIE z pełnym kontekstem pre_batch_info.
    
    Nie przepisuje całości - tylko:
    1. DODAJE brakujące frazy (UNDER) w naturalnych miejscach
    2. ZAMIENIA exceeded na synonimy
    3. POPRAWIA spójność z poprzednim batchem
    4. NAPRAWIA wzorce AI
    
    Request body:
    {
        "text": "tekst batcha do poprawy",
        "batch_number": 3,  // opcjonalne, domyślnie ostatni batch
        "auto_save": true   // czy zapisać poprawiony tekst
    }
    
    Response:
    {
        "success": true,
        "fixed_text": "poprawiony tekst",
        "changes_made": ["Dodano frazę: 'xyz'", ...],
        "keywords_added": ["xyz", ...],
        "keywords_replaced": [{"original": "sąd", "replacement": "organ sądowy"}],
        "auto_saved": true/false
    }
    """
    data = request.get_json(force=True) if request.is_json else {}
    
    # Pobierz tekst
    text = data.get("text", "").strip()
    auto_save = data.get("auto_save", False)
    batch_number = data.get("batch_number")
    
    # Pobierz projekt
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    keywords_state = project_data.get("keywords_state", {})
    
    # Jeśli nie podano tekstu, użyj ostatniego batcha
    if not text and batches:
        text = batches[-1].get("text", "")
        batch_number = len(batches)
        print(f"[CLAUDE_FIX] Używam tekstu z ostatniego batcha #{batch_number}")
    
    if not text:
        return jsonify({
            "error": "No text to fix",
            "hint": "Podaj tekst w polu 'text' lub najpierw dodaj batch"
        }), 400
    
    # Określ numer batcha
    if not batch_number:
        batch_number = len(batches) if batches else 1
    
    # Sprawdź czy mamy batch_review_system
    if not BATCH_REVIEW_ENABLED:
        return jsonify({
            "error": "Batch Review System not available",
            "hint": "Moduł batch_review_system.py nie jest dostępny"
        }), 500
    
    try:
        # Przygotuj pre_batch_info
        pre_batch_info = get_pre_batch_info_for_claude(project_data, batch_number)
        
        # Policz batch_counts dla review
        from keyword_counter import count_keywords_for_state
        batch_counts = count_keywords_for_state(text, keywords_state, use_exclusive_for_nested=False)
        
        # Najpierw zrób review żeby wykryć problemy
        previous_batch_text = batches[batch_number - 2]["text"] if batch_number > 1 and len(batches) >= batch_number - 1 else None
        
        review_result = review_batch_comprehensive(
            batch_text=text,
            keywords_state=keywords_state,
            batch_counts=batch_counts,
            previous_batch_text=previous_batch_text,
            auto_fix=False,  # Nie auto-fix, Claude to zrobi
            use_claude_for_complex=False
        )
        
        # Wywołaj Claude Smart-Fix
        result = claude_smart_fix(
            batch_text=text,
            pre_batch_info=pre_batch_info,
            review_result=review_result,
            keywords_state=keywords_state
        )
        
        if not result.success:
            return jsonify({
                "success": False,
                "error": result.error or "Unknown error",
                "review_issues": [i.message for i in review_result.issues] if review_result else [],
                "prompt_preview": result.prompt_used[:1000] + "..." if result.prompt_used else None
            }), 500
        
        fixed_text = result.fixed_text
        
        # Auto-save jeśli requested
        saved = False
        if auto_save and fixed_text and batches and batch_number <= len(batches):
            # Zaktualizuj odpowiedni batch
            batch_idx = batch_number - 1
            batches[batch_idx]["text"] = fixed_text
            batches[batch_idx]["claude_fixed"] = True
            batches[batch_idx]["claude_changes"] = result.changes_made
            
            doc_ref.update({
                "batches": sanitize_for_firestore(batches)
            })
            saved = True
            print(f"[CLAUDE_FIX] ✅ Auto-saved fixed batch #{batch_number}")
        
        return jsonify({
            "success": True,
            "fixed_text": fixed_text,
            "changes_made": result.changes_made,
            "keywords_added": result.keywords_added,
            "keywords_replaced": result.keywords_replaced,
            "auto_saved": saved,
            "batch_number": batch_number,
            "review_before": {
                "issues_count": len(review_result.issues) if review_result else 0,
                "status": review_result.status if review_result else "UNKNOWN",
                "issues": [
                    {"type": i.type.value, "message": i.message}
                    for i in (review_result.issues if review_result else [])[:10]
                ]
            }
        })
        
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"Missing dependency: {e}",
            "hint": "Zainstaluj keyword_counter"
        }), 500
    except Exception as e:
        import traceback
        print(f"[CLAUDE_FIX] ❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@tracker_routes.post("/api/project/<project_id>/review_and_fix")
def review_and_fix_batch(project_id):
    """
    🔍 Review batcha + automatyczne poprawki + opcjonalnie Claude Smart-Fix.
    
    Workflow:
    1. Auto-fix (synonimy, burstiness) - bez Claude
    2. Jeśli są problemy wymagające Claude → wywołaj Claude Smart-Fix
    3. Zapisz poprawiony batch
    
    Request body:
    {
        "text": "tekst batcha",
        "use_claude": true,  // czy użyć Claude dla złożonych problemów
        "auto_save": true    // czy zapisać poprawiony batch
    }
    """
    data = request.get_json(force=True) if request.is_json else {}
    
    text = data.get("text", "").strip()
    use_claude = data.get("use_claude", True)
    auto_save = data.get("auto_save", True)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Pobierz projekt
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    keywords_state = project_data.get("keywords_state", {})
    batch_number = len(batches) + 1
    
    if not BATCH_REVIEW_ENABLED:
        return jsonify({
            "error": "Batch Review System not available"
        }), 500
    
    try:
        from batch_review_system import (
            review_batch_comprehensive,
            claude_smart_fix,
            get_pre_batch_info_for_claude,
            get_review_summary
        )
        
        # KROK 1: Review + Auto-Fix
        previous_text = batches[-1]["text"] if batches else None
        
        review_result = review_batch_comprehensive(
            batch_text=text,
            keywords_state=keywords_state,
            batch_counts={},
            previous_batch_text=previous_text,
            auto_fix=True,
            use_claude_for_complex=False
        )
        
        current_text = review_result.fixed_text or text
        
        # KROK 2: Claude Smart-Fix jeśli potrzebny
        claude_result = None
        if use_claude and review_result.claude_fixes_needed:
            pre_batch_info = get_pre_batch_info_for_claude(project_data, batch_number)
            
            claude_result = claude_smart_fix(
                batch_text=current_text,
                pre_batch_info=pre_batch_info,
                review_result=review_result,
                keywords_state=keywords_state
            )
            
            if claude_result.get("success"):
                current_text = claude_result.get("fixed_text", current_text)
        
        # KROK 3: Zapisz jeśli requested
        saved = False
        if auto_save:
            # Użyj process_batch do zapisania z pełną walidacją
            save_result = process_batch_in_firestore(
                project_id=project_id,
                batch_text=current_text,
                meta_trace={"source": "review_and_fix", "claude_used": use_claude}
            )
            saved = save_result.get("status") != "REJECTED"
        
        return jsonify({
            "success": True,
            "original_text": text,
            "fixed_text": current_text,
            "review_summary": get_review_summary(review_result),
            "auto_fixes_applied": review_result.auto_fixes_applied,
            "claude_used": claude_result is not None and claude_result.get("success", False),
            "claude_changes": claude_result.get("changes_made", []) if claude_result else [],
            "saved": saved,
            "batch_number": batch_number,
            "final_status": review_result.status
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
    
    # 🆕 v36.3: Check if Firestore save failed
    if isinstance(result, dict) and result.get("status") == "ERROR":
        return jsonify({
            "status": "ERROR",
            "error": result.get("error", "Unknown Firestore error"),
            "message": result.get("message", "Batch was NOT saved"),
            "retry": True
        }), 500

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
