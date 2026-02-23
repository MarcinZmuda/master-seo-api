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

# v40.2: Import from core_metrics (Single Source of Truth)
try:
    from core_metrics import (
        calculate_burstiness_simple as _calculate_burstiness_core,
        calculate_transition_score as _calculate_transition_score_core,
        split_into_sentences,
        TRANSITION_WORDS_PL as TRANSITION_WORDS_CORE
    )
    CORE_METRICS_AVAILABLE = True
except ImportError:
    CORE_METRICS_AVAILABLE = False
    print("[FIRESTORE_TRACKER] ⚠️ core_metrics not available, using local functions")

# ================================================================
# 🆕 v44.1: FIRESTORE KEY SANITIZATION - import z firestore_utils
# ================================================================
from firestore_utils import sanitize_for_firestore

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

# v24.2: Unified keyword counter — count_keyword_occurrences alias
try:
    from keyword_counter import count_single_keyword as count_keyword_occurrences
except ImportError:
    def count_keyword_occurrences(text, keyword):
        """Fallback: simple case-insensitive count."""
        return text.lower().count(keyword.lower()) if text and keyword else 0

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
# ⚠️ FIX v1.2: Lazy import - unikamy circular import
# Import przeniesiony do funkcji gdzie jest używany
MOE_VALIDATOR_ENABLED = True  # Zakładamy że jest, sprawdzimy w runtime
print("[TRACKER] ✅ MoE Batch Validator (lazy load)")

# 🆕 v37.4: Quality Score Module
try:
    from quality_score_module import (
        calculate_global_quality_score,
        create_simplified_response,
        validate_fast_mode,
        get_gpt_action_response,
        QualityConfig,
        CONFIG as QUALITY_CONFIG
    )
    QUALITY_SCORE_ENABLED = True
    print("[TRACKER] ✅ Quality Score Module loaded")
except ImportError as e:
    QUALITY_SCORE_ENABLED = False
    print(f"[TRACKER] ⚠️ Quality Score Module not available: {e}")

# 🆕 v38: Entity Coverage Validator
try:
    from entity_validator import (
        EntityCoverageExpert,
        initialize_entity_state,
        generate_entity_requirements,
        EntityValidationResult,
        EntityCoverageResult,
        detect_entity_drift  # 🆕 v38.2
    )
    ENTITY_VALIDATOR_ENABLED = True
    print("[TRACKER] ✅ Entity Coverage Validator loaded")
except ImportError as e:
    ENTITY_VALIDATOR_ENABLED = False
    print(f"[TRACKER] ⚠️ Entity Coverage Validator not available: {e}")

# 🆕 v38: Legal Hard-Lock Validator
try:
    from legal_validator import (
        LegalHardLockValidator,
        create_legal_whitelist,
        add_common_legal_articles,
        LegalValidationResult
    )
    LEGAL_VALIDATOR_ENABLED = True
    print("[TRACKER] ✅ Legal Hard-Lock Validator loaded")
except ImportError as e:
    LEGAL_VALIDATOR_ENABLED = False
    print(f"[TRACKER] ⚠️ Legal Hard-Lock Validator not available: {e}")

# 🆕 v38: Helpful Reflex Detector (z batch_review_system)
try:
    from batch_review_system import detect_helpful_reflex, auto_remove_fillers
    HELPFUL_REFLEX_ENABLED = True
except ImportError:
    HELPFUL_REFLEX_ENABLED = False

# 🆕 v42.2: Keyword Limiter Integration (dynamiczne limity stuffingu)
try:
    from keyword_limiter_integration import (
        get_enhanced_stuffing_check,
        validate_headers_and_structure,
        validate_batch_with_keyword_limiter,
        get_limit_for_keyword,
        KEYWORD_LIMITER_ENABLED
    )
    print("[TRACKER] ✅ Keyword Limiter Integration v42.2 loaded")
except ImportError as e:
    KEYWORD_LIMITER_ENABLED = False
    print(f"[TRACKER] ⚠️ Keyword Limiter Integration not available: {e}")

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


# 🆕 v44.1: calculate_burstiness z core_metrics (Single Source of Truth)
# Poprzednio: warunkowy fallback - USUNIĘTY (core_metrics zawsze dostępny)
from core_metrics import calculate_burstiness_simple as calculate_burstiness


# 🆕 v44.1: Używamy TRANSITION_WORDS z core_metrics (Single Source of Truth)
# Poprzednio: lokalna kopia TRANSITION_WORDS_PL - USUNIĘTA


def calculate_transition_score(text: str) -> dict:
    """Target: 25-50% zdań z transition words"""
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences), "warnings": []}
    
    # v44.1: Używamy TRANSITION_WORDS_CORE z core_metrics
    transition_words = list(TRANSITION_WORDS_CORE) if CORE_METRICS_AVAILABLE else [
        "również", "także", "ponadto", "jednak", "natomiast", "dlatego", "ponieważ"
    ]
    transition_count = sum(1 for s in sentences if any(tw in s.lower()[:100] for tw in transition_words))
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
                # v55.1: Strip markdown code fences (GPT-4.1 wraps HTML in ```html...```)
                _bt = text.strip()
                if _bt.startswith("```html"):
                    _bt = _bt[7:]
                elif _bt.startswith("```"):
                    _bt = _bt[3:]
                if _bt.endswith("```"):
                    _bt = _bt[:-3]
                _bt = _bt.strip()
                # v56 FIX 2C: Normalize malformed HTML tags (same as _clean_batch_text)
                _bt = re.sub(r'<p[.,;:]+>', '<p>', _bt, flags=re.IGNORECASE)
                _bt = re.sub(r'</p[.,;:]+>', '</p>', _bt, flags=re.IGNORECASE)
                _bt = re.sub(r'<(h[2-6])[.,;:]+>', r'<\1>', _bt, flags=re.IGNORECASE)
                _bt = re.sub(r'</(h[2-6])[.,;:]+>', r'</\1>', _bt, flags=re.IGNORECASE)
                def _lower_tag(m):
                    return m.group(0).lower()
                _bt = re.sub(r'</?[A-Z][A-Z0-9]*(?:\s[^>]*)?\s*/?>', _lower_tag, _bt)
                full_content_parts.append(_bt)
                total_words += len(_bt.split())
                h2_count += len(re.findall(r'(?:^h2:|<h2)', _bt, re.MULTILINE | re.IGNORECASE))

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


# v57.1: Helper to detect if batch text has a specific H2
def _batch_matches_h2(text: str, target_h2_lower: str) -> bool:
    """Check if batch text starts with the same H2 header."""
    if not text:
        return False
    m = re.match(r'(?:h2:\s*(.+)|<h2[^>]*>([^<]+)</h2>)', text.strip(), re.IGNORECASE)
    if m:
        h2 = (m.group(1) or m.group(2)).strip().lower()
        return h2 == target_h2_lower
    return False


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
    
    # ================================================================
    # 🆕 v38: INICJALIZACJA ENTITY_STATE (jeśli brak)
    # ================================================================
    if ENTITY_VALIDATOR_ENABLED and "entity_state" not in project_data:
        s1_data = project_data.get("s1_data", {})
        batch_plan = {
            "total_batches": project_data.get("total_planned_batches", 6)
        }
        if s1_data.get("entities"):
            project_data["entity_state"] = initialize_entity_state(s1_data, batch_plan)
            print(f"[TRACKER] 🎯 Entity state initialized: {len(project_data['entity_state'])} entities")
    
    # ================================================================
    # 🆕 v38: INICJALIZACJA LEGAL_WHITELIST (jeśli YMYL i brak)
    # ================================================================
    if LEGAL_VALIDATOR_ENABLED and project_data.get("is_legal") and "legal_whitelist" not in project_data:
        detected_articles = project_data.get("detected_articles", [])
        legal_judgments = project_data.get("legal_judgments", [])
        legal_category = project_data.get("legal_category", "")
        
        whitelist = create_legal_whitelist(detected_articles, legal_judgments)
        
        # Dodaj popularne artykuły dla kategorii
        if legal_category:
            whitelist = add_common_legal_articles(whitelist, legal_category)
        
        project_data["legal_whitelist"] = whitelist
        project_data["legal_hardlock_enabled"] = True
        print(f"[TRACKER] ⚖️ Legal whitelist created: {len(whitelist.get('articles', []))} articles, {len(whitelist.get('judgments', []))} judgments")
    
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
        
        # 🆕 v42.2: ENHANCED STUFFING CHECK z dynamicznymi limitami
        # Jeśli keyword_limiter_integration dostępny - używa dynamicznych limitów
        # W przeciwnym razie - fallback do starej metody
        if KEYWORD_LIMITER_ENABLED:
            stuffing_result = get_enhanced_stuffing_check(batch_text, keywords_state)
            stuffing_warnings = stuffing_result.get("warnings", [])
            
            # Dodaj walidację nagłówków i struktury
            main_kw = project_data.get("main_keyword", project_data.get("topic", ""))
            header_result = validate_headers_and_structure(
                batch_text,
                main_keyword=main_kw,
                title=project_data.get("title", "")
            )
            stuffing_warnings.extend(header_result.get("warnings", []))
            
            # Log info o dynamicznych limitach
            if stuffing_result.get("dynamic_limits_used"):
                print(f"[TRACKER] ✅ Dynamic stuffing limits used (word_count={stuffing_result.get('word_count', 0)})")
        else:
            # Fallback do standardowej metody
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
        
        # 🆕 v42.2: Legacy stuffing z keyword_limiter fallback
        if KEYWORD_LIMITER_ENABLED:
            # Użyj dynamicznych limitów nawet w legacy mode
            stuffing_result = get_enhanced_stuffing_check(batch_text, keywords_state)
            stuffing_warnings = stuffing_result.get("warnings", [])
            
            main_kw = project_data.get("main_keyword", project_data.get("topic", ""))
            header_result = validate_headers_and_structure(
                batch_text, main_keyword=main_kw, title=project_data.get("title", "")
            )
            stuffing_warnings.extend(header_result.get("warnings", []))
        else:
            # Original legacy stuffing
            stuffing_warnings = []
            paragraphs = batch_text.split('\n\n')
            for rid, meta in keywords_state.items():
                if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN", "ENTITY"]:
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
        if kw_type not in ["BASIC", "MAIN", "ENTITY"]:
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
    # 🆕 v38.3: STRUCTURAL keywords (w MAIN/H2) - POMIJANE, tylko global check
    # ================================================================
    exceeded_warning = []   # 1-49% over max
    exceeded_critical = []  # 50%+ over max
    
    # 🆕 v38.3: Wykryj STRUCTURAL keywords (w MAIN lub H2)
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    h2_structure = project_data.get("h2_structure", [])
    main_lower = main_keyword.lower().strip()
    h2_lower = [h.lower().strip() for h in h2_structure]
    
    def is_structural_keyword(kw: str) -> bool:
        """Sprawdza czy fraza jest STRUCTURAL (w MAIN lub H2)."""
        kw_lower = kw.lower().strip()
        # W MAIN
        if kw_lower in main_lower or kw_lower == main_lower:
            return True
        # W H2
        for h2 in h2_lower:
            if kw_lower in h2 or kw_lower == h2 or h2 in kw_lower:
                return True
        return False
    
    structural_exceeded = []  # 🆕 v38.3: exceeded ale STRUCTURAL - tylko info
    
    for rid, batch_count in batch_counts.items():
        meta = keywords_state[rid]
        
        # Tylko BASIC, MAIN i ENTITY - EXTENDED pomijamy (mogą być przekroczone)
        if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN", "ENTITY"]:
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
                "type": meta.get("type", "BASIC").upper(),
                "current": current,
                "batch_uses": batch_count,
                "would_be": new_total,
                "target_max": target_max,
                "exceeded_by": exceeded_by,
                "exceed_percent": round(exceed_percent),
                "synonyms": synonyms[:3]
            }
            
            # 🆕 v38.3: STRUCTURAL keywords - NIE blokują batcha!
            if is_structural_keyword(keyword):
                exceeded_info["is_structural"] = True
                structural_exceeded.append(exceeded_info)
                print(f"[TRACKER] 🔵 STRUCTURAL: '{keyword}' exceeded but STRUCTURAL (global check only)")
                continue  # Pomijamy - nie dodajemy do critical/warning
            
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
    
    # 🆕 v38.3: Info o structural exceeded (nie blokują, ale warto wiedzieć)
    if structural_exceeded:
        print(f"[TRACKER] 🔵 {len(structural_exceeded)} STRUCTURAL keywords exceeded (global check in final_review)")
    
    # ================================================================
    # 🆕 v36.0: RESERVED KEYWORDS VALIDATION
    # Sprawdź czy batch używa fraz zarezerwowanych dla innych sekcji
    # ================================================================
    reserved_keyword_warnings = []
    current_batch_num = batches_done + 1  # Bo to jest batch który właśnie dodajemy
    semantic_plan = project_data.get("semantic_keyword_plan", {})
    if semantic_plan:
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
            if meta.get("keyword") and meta.get("type", "BASIC").upper() in ["BASIC", "MAIN", "ENTITY"]
        ][:15]  # Max 15 keywords
        
        if proximity_clusters or supporting_entities.get("all"):
            concept_map = {
                "proximity_clusters": proximity_clusters,
                "supporting_entities": supporting_entities,
            }
            proximity_result = full_semantic_validation(
                text=batch_text,
                keywords=keywords_to_check,
                concept_map=concept_map,
                batch_number=current_batch_num
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

    # v24.0: Jeśli tylko per_batch warnings (nie EXCEEDED TOTAL) - status APPROVED
    has_critical = any("EXCEEDED TOTAL" in w for w in warnings)
    has_density_issue = any("density" in w.lower() and density > DENSITY_MAX for w in warnings)
    if not has_critical and not has_density_issue and not exceeded_keywords:
        status = "APPROVED"

    # 🆕 v45.4: FORCED status MUST be the last override - prevents being overwritten by APPROVED
    if forced:
        status = "FORCED"

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
            # 🆕 FIX v1.2: Lazy import - unikamy circular import
            from moe_batch_validator import validate_batch_moe, format_validation_for_gpt, ValidationMode
            
            batches_done = len(project_data.get("batches", []))
            moe_validation_result = validate_batch_moe(
                batch_text=batch_text,
                project_data=project_data,
                batch_number=batches_done + 1,
                mode=ValidationMode.AUTO_FIX  # 🆕 v42.1: AUTO_FIX = Content Surgeon aktywny!
            )
            
            # 🆕 v42.1: Jeśli Content Surgeon naprawił tekst, użyj poprawionego
            if moe_validation_result.surgery_applied and moe_validation_result.corrected_text:
                batch_text = moe_validation_result.corrected_text
                surgery_stats = getattr(moe_validation_result, 'surgery_stats', {})
                print(f"[TRACKER] 🔬 Content Surgery applied: {surgery_stats.get('injected', 0)} phrases injected")
            
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

    # ================================================================
    # 🆕 v38: ENTITY COVERAGE VALIDATION
    # ================================================================
    entity_validation_result = None
    entity_fix_instructions = []
    
    if ENTITY_VALIDATOR_ENABLED and project_data.get("entity_state"):
        try:
            entity_expert = EntityCoverageExpert()
            batches_done = len(project_data.get("batches", []))
            batch_number = batches_done + 1
            
            # Generuj wymagania encji dla tego batcha
            entity_requirements = generate_entity_requirements(
                entity_state=project_data.get("entity_state", {}),
                batch_number=batch_number,
                semantic_plan=project_data.get("semantic_keyword_plan", {}),
                total_batches=project_data.get("total_planned_batches", 6)
            )
            
            # Waliduj pokrycie encji
            entity_validation_result = entity_expert.validate_batch(
                batch_text=batch_text,
                entity_requirements=entity_requirements.get("required_entities", []),
                entity_state=project_data.get("entity_state", {}),
                relationships_to_check=entity_requirements.get("relationships_to_establish", [])
            )
            
            # Dodaj warnings
            if entity_validation_result.must_missing:
                for entity in entity_validation_result.must_missing:
                    warnings.append(f"🎯 ENTITY MISSING: '{entity}' - wymagana encja nie występuje!")
                    entity_fix_instructions.append(f"Dodaj encję '{entity}' do treści")
            
            if entity_validation_result.should_missing:
                for entity in entity_validation_result.should_missing:
                    warnings.append(f"ℹ️ ENTITY: '{entity}' - zalecana encja nie występuje")
            
            # Aktualizuj entity_state
            if entity_validation_result.status != "FAIL":
                project_data["entity_state"] = entity_expert.update_entity_state(
                    entity_state=project_data.get("entity_state", {}),
                    batch_text=batch_text,
                    batch_number=batch_number,
                    validation_results=entity_validation_result.results
                )
                
                # Aktualizuj relacje
                if entity_validation_result.relationships_established:
                    project_data["entity_state"] = entity_expert.update_relationships(
                        entity_state=project_data["entity_state"],
                        relationships_established=entity_validation_result.relationships_established
                    )
            
            print(f"[TRACKER] 🎯 Entity Coverage: {entity_validation_result.status} "
                  f"(score: {entity_validation_result.score}, missing: {len(entity_validation_result.must_missing)})")
                
        except Exception as e:
            print(f"[TRACKER] ⚠️ Entity Coverage Validation error: {e}")

    # ================================================================
    # 🆕 v38: LEGAL HARD-LOCK VALIDATION (YMYL)
    # ================================================================
    legal_validation_result = None
    legal_removals = []
    processed_batch_text = batch_text
    
    if LEGAL_VALIDATOR_ENABLED and project_data.get("legal_hardlock_enabled"):
        try:
            legal_validator = LegalHardLockValidator()
            
            legal_validation_result = legal_validator.validate_batch(
                batch_text=batch_text,
                legal_whitelist=project_data.get("legal_whitelist", {}),
                auto_remove=True,  # Automatycznie usuwaj nielegalne przepisy
                strict_mode=True
            )
            
            # Użyj przetworzonego tekstu (bez nielegalnych przepisów)
            if legal_validation_result.removals:
                processed_batch_text = legal_validation_result.processed_text
                batch_text = processed_batch_text
                
                for removal in legal_validation_result.removals:
                    warnings.append(f"⚖️ LEGAL: Usunięto '{removal.original}' - {removal.reason}")
                    legal_removals.append(removal.to_dict())
            
            if legal_validation_result.violations:
                for violation in legal_validation_result.violations:
                    if violation.severity.value == "CRITICAL":
                        warnings.append(f"⚖️ LEGAL CRITICAL: {violation.found_text} - potencjalna halucynacja!")
            
            print(f"[TRACKER] ⚖️ Legal Validation: valid={legal_validation_result.is_valid}, "
                  f"violations={len(legal_validation_result.violations)}, "
                  f"auto_removed={len(legal_validation_result.removals)}")
                
        except Exception as e:
            print(f"[TRACKER] ⚠️ Legal Validation error: {e}")

    # ================================================================
    # 🆕 v45.4: METRICS RECOMPUTE AFTER SURGERY
    # Jeśli tekst został zmodyfikowany (surgery lub legal removals)
    # - Oblicz nowe metryki (burstiness, transition, density)
    # - Przebuduj warnings bez starych metryk
    # - Re-evaluate exceeded keywords z nowym tekstem
    # - Określ status SURGERY_WARN jeśli były zmian
    # ================================================================
    text_was_modified = (moe_validation_result and moe_validation_result.surgery_applied) or (len(legal_removals) > 0)

    if text_was_modified:
        # Snapshot original exceeded state
        original_exceeded = exceeded_keywords.copy() if exceeded_keywords else []

        # Recompute metrics with potentially modified text
        burstiness = calculate_burstiness(batch_text)
        transition_data = calculate_transition_score(batch_text)
        precheck = unified_prevalidation(batch_text, keywords_state)
        density = precheck.get("density", 0.0)

        # Rebuild warnings - remove old metric-based warnings, keep structure/validity
        old_warnings = warnings.copy()
        warnings = [w for w in old_warnings if not any(
            pattern in w.lower() for pattern in
            ["burstiness", "transition", "density", "exceeded:"]
        )]

        # Revalidate metrics
        metrics_warnings = validate_metrics(burstiness, transition_data, density)
        warnings.extend(metrics_warnings)

        # Re-evaluate exceeded keywords with modified text
        exceeded_warning_new = []
        exceeded_critical_new = []

        for rid, meta in keywords_state.items():
            if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN", "ENTITY"]:
                continue

            keyword = meta.get("keyword", "")
            if not keyword:
                continue

            # Count uses in modified text
            batch_count_new = count_keyword_occurrences(batch_text, keyword)
            current = meta.get("actual_uses", 0)
            target_max = meta.get("target_max", 999)
            new_total = current + batch_count_new

            if new_total > target_max:
                exceeded_by = new_total - target_max
                exceed_percent = (exceeded_by / target_max * 100) if target_max > 0 else 100

                synonyms = get_synonyms(keyword) if SYNONYMS_ENABLED else []

                exceeded_info = {
                    "keyword": keyword,
                    "current": current,
                    "batch_uses": batch_count_new,
                    "would_be": new_total,
                    "target_max": target_max,
                    "exceeded_by": exceeded_by,
                    "exceed_percent": round(exceed_percent),
                    "synonyms": synonyms[:3]
                }

                if is_structural_keyword(keyword):
                    exceeded_info["is_structural"] = True
                    continue

                if exceed_percent >= 50:
                    exceeded_critical_new.append(exceeded_info)
                else:
                    exceeded_warning_new.append(exceeded_info)

        # Update exceeded lists
        exceeded_warning = exceeded_warning_new
        exceeded_critical = exceeded_critical_new
        exceeded_keywords = exceeded_warning + exceeded_critical

        # Add warnings for re-evaluated exceeded keywords
        for ew in exceeded_warning:
            syn_hint = f" → rozważ synonimy: {', '.join(ew['synonyms'][:2])}" if ew.get('synonyms') else ""
            warnings.append(
                f"⚠️ EXCEEDED: '{ew['keyword']}' = {ew['would_be']}/{ew['target_max']} "
                f"(+{ew['exceed_percent']}%){syn_hint}"
            )

        # 🆕 Fix #5 v4.2: SURGERY_WARN — detect regressions introduced by surgery
        pre_surgery_exceeded = set(e['keyword'] for e in original_exceeded) if original_exceeded else set()
        post_surgery_exceeded = set(e['keyword'] for e in exceeded_keywords) if exceeded_keywords else set()
        new_exceeded = post_surgery_exceeded - pre_surgery_exceeded

        # Re-evaluate status after surgery
        status = 'APPROVED'
        if warnings or not valid_struct:
            status = 'WARN'
        if new_exceeded:
            status = 'SURGERY_WARN'
            warnings.append(f'SURGERY_WARN: nowe exceeded po surgery: {new_exceeded}')
        has_critical = any('EXCEEDED TOTAL' in w for w in warnings)
        has_density_issue = any('density' in w.lower() and density > DENSITY_MAX for w in warnings)
        if not has_critical and not has_density_issue and not exceeded_keywords:
            status = 'APPROVED'
        if forced:
            status = 'FORCED'

        print(f"[TRACKER] 🔬 Surgery metrics recomputed: burstiness={burstiness:.2f}, "
              f"transition={transition_data.get('ratio', 0):.1%}, density={density:.1%}, status={status}")

    # Save batch
    # 🆕 v41.3: Bezpieczne wywołanie to_dict() - chroni przed błędami
    def safe_to_dict(obj):
        """Bezpiecznie konwertuje obiekt na dict."""
        if obj is None:
            return None
        if hasattr(obj, 'to_dict') and callable(obj.to_dict):
            try:
                return obj.to_dict()
            except Exception as e:
                print(f"[TRACKER] ⚠️ safe_to_dict error: {e}")
                return {"error": str(e), "type": type(obj).__name__}
        if isinstance(obj, dict):
            return obj
        # Fallback - spróbuj __dict__
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return {"raw": str(obj)}
    
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
        # 🆕 v37.1: MoE Validation results - używamy safe_to_dict
        "moe_validation": safe_to_dict(moe_validation_result),
        "moe_fix_instructions": moe_fix_instructions,
        # 🆕 v38: Entity Coverage results
        "entity_validation": safe_to_dict(entity_validation_result),
        "entity_fix_instructions": entity_fix_instructions,
        # 🆕 v38: Legal Validation results
        "legal_validation": safe_to_dict(legal_validation_result),
        "legal_removals": legal_removals
    }

    # v57.1: Replace previous batch with same H2 instead of appending.
    # When batch is retried (5 attempts), each call appended a new entry,
    # resulting in 5x duplicated H2 sections in the final article.
    # Fix: detect H2 in batch text and replace existing batch with same H2.
    _batch_h2_match = re.match(r'(?:h2:\s*(.+)|<h2[^>]*>([^<]+)</h2>)', batch_text.strip(), re.IGNORECASE)
    _batch_h2 = (_batch_h2_match.group(1) or _batch_h2_match.group(2)).strip().lower() if _batch_h2_match else None

    existing_batches = project_data.setdefault("batches", [])
    if _batch_h2:
        # Remove previous batch(es) with same H2 header
        _before_count = len(existing_batches)
        existing_batches[:] = [
            b for b in existing_batches
            if not _batch_matches_h2(b.get("text", ""), _batch_h2)
        ]
        _removed = _before_count - len(existing_batches)
        if _removed:
            print(f"[TRACKER] 🔄 Replaced {_removed} previous batch(es) for H2: {_batch_h2[:60]}")

    existing_batches.append(batch_entry)
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
            
            # 🆕 v41.3: Bezpieczny dostęp do atrybutów
            fixed_text_val = getattr(batch_review_result, 'fixed_text', None) or getattr(batch_review_result, 'corrected_text', None)
            auto_fixes_val = getattr(batch_review_result, 'auto_fixes_applied', []) or []
            
            # Jeśli były auto-poprawki, zaktualizuj batch w Firestore
            if fixed_text_val and fixed_text_val != batch_text:
                auto_fixed_text = fixed_text_val
                
                # Zaktualizuj ostatni batch w project_data
                if project_data.get("batches"):
                    project_data["batches"][-1]["text"] = auto_fixed_text
                    project_data["batches"][-1]["auto_fixed"] = True
                    project_data["batches"][-1]["auto_fixes"] = auto_fixes_val
                    
                    # Zapisz do Firestore
                    doc_ref.update({
                        "batches": sanitize_for_firestore(project_data["batches"])
                    })
                
                print(f"[TRACKER] ✅ Auto-fixed {len(auto_fixes_val)} issues")
                for fix in auto_fixes_val[:3]:
                    print(f"[TRACKER]    • {fix}")
                
                # ================================================================
                # 🆕 v37.5: RECOUNT KEYWORDS AFTER AUTO-FIX
                # ================================================================
                print(f"[TRACKER] 🔄 Recounting keywords after auto-fix...")
                
                # 1. Przelicz frazy w POPRAWIONYM tekście
                new_batch_counts = count_keywords_for_state(
                    auto_fixed_text, 
                    keywords_state, 
                    use_exclusive_for_nested=False
                )
                
                # 2. Oblicz różnicę i zaktualizuj keywords_state
                recount_changes = []
                for rid, meta in keywords_state.items():
                    old_count = batch_counts.get(rid, 0)
                    new_count = new_batch_counts.get(rid, 0)
                    
                    if old_count != new_count:
                        keyword = meta.get("keyword", rid)
                        diff = new_count - old_count
                        recount_changes.append({
                            "keyword": keyword,
                            "before": old_count,
                            "after": new_count,
                            "diff": diff
                        })
                        print(f"[TRACKER]    • '{keyword}': {old_count} → {new_count} ({diff:+d})")
                        
                        # Cofnij stare liczenie, dodaj nowe
                        current_actual = meta.get("actual_uses", 0)
                        corrected_actual = current_actual - old_count + new_count
                        meta["actual_uses"] = max(0, corrected_actual)
                        
                        # Przelicz status
                        target_min = meta.get("target_min", 0)
                        target_max = meta.get("target_max", 999)
                        actual = meta["actual_uses"]
                        
                        if actual < target_min:
                            meta["status"] = "UNDER"
                        elif actual == target_max:
                            meta["status"] = "OPTIMAL"
                        elif target_min <= actual < target_max:
                            meta["status"] = "OK"
                        elif actual > target_max:
                            meta["status"] = "OVER"
                        
                        meta["remaining_max"] = max(0, target_max - actual)
                
                # 3. Użyj nowych batch_counts
                batch_counts = new_batch_counts
                
                # 4. Zapisz poprawiony keywords_state do Firestore
                doc_ref.update({
                    "keywords_state": sanitize_for_firestore(keywords_state)
                })
                
                print(f"[TRACKER] ✅ Keywords recounted: {len(recount_changes)} changes, state saved to Firestore")
            
            # Loguj review summary
            review_status = getattr(batch_review_result, 'status', 'UNKNOWN')
            review_issues = getattr(batch_review_result, 'issues', []) or []
            print(f"[TRACKER] 📋 Review: {review_status}, issues: {len(review_issues)}")
            
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
    final_review_result = None
    
    if is_last_batch:
        print(f"[TRACKER] 🏁 Last batch detected! Running final review...")
        
        # ================================================================
        # 🆕 v38.4: FINAL REVIEW - weryfikacja globalna z tolerancją 30%
        # ================================================================
        final_review_warnings = []
        final_review_exceeded = []
        
        TOLERANCE_PERCENT = 30  # 30% tolerancji powyżej max
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            
            kw_type = meta.get("type", "BASIC").upper()
            actual = meta.get("actual_uses", 0)
            target_max = meta.get("target_max", 999)
            target_min = meta.get("target_min", 1)
            
            # Oblicz tolerancję (30% powyżej max)
            tolerance_max = int(target_max * (1 + TOLERANCE_PERCENT / 100))
            
            # Sprawdź przekroczenie
            if actual > tolerance_max:
                # Przekroczono nawet z tolerancją 30% → ERROR
                exceeded_by = actual - target_max
                over_tolerance_by = actual - tolerance_max
                final_review_exceeded.append({
                    "keyword": keyword,
                    "type": kw_type,
                    "actual": actual,
                    "target_max": target_max,
                    "tolerance_max": tolerance_max,
                    "exceeded_by": exceeded_by,
                    "over_tolerance_by": over_tolerance_by,
                    "severity": "ERROR",
                    "message": f"'{keyword}' przekroczyła limit nawet z tolerancją 30%: {actual}/{target_max} (max z tolerancją: {tolerance_max})"
                })
                print(f"[FINAL REVIEW] ❌ '{keyword}': {actual}/{target_max} (tolerance {tolerance_max}) - EXCEEDED!")
            
            elif actual > target_max:
                # Przekroczono max ale w granicach tolerancji → WARNING (OK)
                final_review_warnings.append({
                    "keyword": keyword,
                    "type": kw_type,
                    "actual": actual,
                    "target_max": target_max,
                    "tolerance_max": tolerance_max,
                    "exceeded_by": actual - target_max,
                    "severity": "WARNING",
                    "message": f"'{keyword}' lekko przekroczona ale w tolerancji 30%: {actual}/{target_max} (OK)"
                })
                print(f"[FINAL REVIEW] ⚠️ '{keyword}': {actual}/{target_max} (within tolerance) - OK")
            
            elif actual < target_min:
                # Poniżej minimum → WARNING
                final_review_warnings.append({
                    "keyword": keyword,
                    "type": kw_type,
                    "actual": actual,
                    "target_min": target_min,
                    "target_max": target_max,
                    "severity": "WARNING",
                    "message": f"'{keyword}' poniżej minimum: {actual}/{target_min}"
                })
                print(f"[FINAL REVIEW] ⚠️ '{keyword}': {actual} < {target_min} (under min)")
        
        final_review_result = {
            "status": "PASS" if not final_review_exceeded else "NEEDS_CORRECTION",
            "tolerance_percent": TOLERANCE_PERCENT,
            "warnings": final_review_warnings,
            "exceeded": final_review_exceeded,
            "warnings_count": len(final_review_warnings),
            "exceeded_count": len(final_review_exceeded),
            "message": (
                f"✅ Final review PASS: {len(final_review_warnings)} warnings (within tolerance)"
                if not final_review_exceeded
                else f"❌ Final review NEEDS_CORRECTION: {len(final_review_exceeded)} keywords exceeded tolerance"
            )
        }
        
        print(f"[TRACKER] 📊 Final review: {final_review_result['status']} ({len(final_review_exceeded)} exceeded, {len(final_review_warnings)} warnings)")
        
        # Auto-merge tylko jeśli final review OK
        if not final_review_exceeded:
            print(f"[TRACKER] 🏁 Final review OK! Auto-merging full article...")
            auto_merge_result = auto_merge_full_article(project_id, project_data)
        else:
            print(f"[TRACKER] ⚠️ Final review has errors - skipping auto-merge")

    # v56: Re-merge if previous auto_merge is stale (fewer batches than current)
    # This handles FIX_AND_RETRY adding batches beyond the original plan.
    existing_fa = project_data.get("full_article", {})
    if isinstance(existing_fa, dict) and existing_fa.get("auto_merged"):
        merged_batch_count = existing_fa.get("batch_count", 0)
        if batches_done > merged_batch_count:
            print(f"[TRACKER] 🔄 Re-merging: full_article has {merged_batch_count} batches but {batches_done} exist")
            auto_merge_result = auto_merge_full_article(project_id, project_data)

    # ================================================================
    # 🆕 v37.4: GLOBAL QUALITY SCORE
    # ================================================================
    quality_result = None
    gpt_action = None
    
    if QUALITY_SCORE_ENABLED:
        try:
            # Przygotuj dane do quality score
            validation_data = {
                "exceeded_critical": exceeded_critical,
                "exceeded_warning": exceeded_warning,
                "burstiness_cv": burstiness / 5.0 if burstiness and burstiness > 0 else 0.4,
                "burstiness": burstiness,
                "humanness_score": real_humanness_score if 'real_humanness_score' in dir() else 50,
                "semantic_score": semantic_score,
                "structure_valid": valid_struct if 'valid_struct' in dir() else True,
                "has_h2": True,  # Już sprawdzone wcześniej
                "batch_text": batch_text,
                "batch_role": "INTRO" if batches_done == 1 else ("FINAL" if is_last_batch else "CONTENT"),
                "moe_validation": safe_to_dict(moe_validation_result) if 'safe_to_dict' in dir() else (moe_validation_result.to_dict() if moe_validation_result and hasattr(moe_validation_result, 'to_dict') else {}),
                "batch_review": safe_to_dict(batch_review_result) if 'safe_to_dict' in dir() else (batch_review_result.to_dict() if batch_review_result and hasattr(batch_review_result, 'to_dict') else {})
            }
            
            # Oblicz quality score
            quality_result = calculate_global_quality_score(
                validation_data, 
                project_data
            )
            
            # Pobierz GPT action
            gpt_action = get_gpt_action_response(
                validation_data,
                quality_result,
                project_data,
                batches_done
            )
            
            print(f"[TRACKER] 🎯 Quality: {quality_result['score']}/100 ({quality_result['grade']}) "
                  f"→ {gpt_action['action']} [{gpt_action['confidence']}]")
            
            if quality_result.get("decision_trace"):
                for trace in quality_result["decision_trace"][:3]:
                    print(f"[TRACKER]    • {trace}")
                    
        except Exception as e:
            print(f"[TRACKER] ⚠️ Quality score error: {e}")
            import traceback
            traceback.print_exc()

    # ================================================================
    # 🆕 v38.1: DECISION ENGINE - ENTITY/LEGAL/HELPFUL → ACTION
    # ================================================================
    # To jest JEDYNE miejsce gdzie entity/legal/helpful WPŁYWAJĄ na action
    # Wcześniejsze walidacje tylko zbierały dane - tu podejmujemy decyzje
    # ================================================================
    
    v38_overrides = []
    v38_fixes_needed = []
    v38_action_override = None
    
    # 1. ENTITY COVERAGE - MUST missing → REWRITE
    if entity_validation_result and entity_validation_result.status == "FAIL":
        v38_action_override = "REWRITE"
        for entity in entity_validation_result.must_missing:
            v38_fixes_needed.append(f"[ENTITY MUST] Dodaj encję '{entity}' do treści")
        v38_overrides.append(f"entity_coverage: FAIL → REWRITE (missing: {entity_validation_result.must_missing})")
        print(f"[TRACKER] 🔴 v38.1 OVERRIDE: Entity FAIL → action=REWRITE")
    
    # 2. LEGAL VIOLATIONS - critical pozostałe po auto-remove → FIX_AND_RETRY
    if legal_validation_result and legal_validation_result.violations:
        # Sprawdź czy są CRITICAL violations które nie zostały auto-usunięte
        critical_violations = [v for v in legal_validation_result.violations 
                              if v.severity.value == "CRITICAL"]
        
        if critical_violations and v38_action_override != "REWRITE":
            v38_action_override = "FIX_AND_RETRY"
            for v in critical_violations[:3]:
                v38_fixes_needed.append(f"[LEGAL] Usuń nielegalny przepis: '{v.found_text}'")
            v38_overrides.append(f"legal_hardlock: {len(critical_violations)} CRITICAL → FIX_AND_RETRY")
            print(f"[TRACKER] 🔴 v38.1 OVERRIDE: Legal CRITICAL → action=FIX_AND_RETRY")
    
    # 3. HELPFUL REFLEX (tylko dla YMYL) - soft_advice → FIX_AND_RETRY
    is_ymyl = project_data.get("is_ymyl", False) or project_data.get("is_legal", False)
    
    if is_ymyl and HELPFUL_REFLEX_ENABLED:
        try:
            helpful_issues = detect_helpful_reflex(batch_text, is_ymyl=True)
            soft_advice_issues = [i for i in helpful_issues 
                                 if i.type.value in ["SOFT_ADVICE", "UNNECESSARY_EXPLANATION"] 
                                 and i.severity.value in ["ERROR", "WARNING"]]
            
            if soft_advice_issues and v38_action_override not in ["REWRITE", "FIX_AND_RETRY"]:
                v38_action_override = "FIX_AND_RETRY"
                for issue in soft_advice_issues[:2]:
                    v38_fixes_needed.append(f"[YMYL] Usuń: '{issue.location[:50]}...'")
                v38_overrides.append(f"helpful_reflex: {len(soft_advice_issues)} soft_advice in YMYL → FIX_AND_RETRY")
                print(f"[TRACKER] 🔴 v38.1 OVERRIDE: YMYL soft_advice → action=FIX_AND_RETRY")
        except Exception as e:
            print(f"[TRACKER] ⚠️ Helpful reflex check error: {e}")
    
    # 4. 🆕 v38.2: RELATION COMPLETION - brakujące relacje MUST → FIX_AND_RETRY
    if entity_validation_result and entity_validation_result.relationships_missing:
        # Sprawdź ile relacji MUST brakuje
        must_relations_missing = [r for r in entity_validation_result.relationships_missing 
                                  if r.get("priority") == "MUST"]
        
        if must_relations_missing and v38_action_override not in ["REWRITE"]:
            # Relacje to mniejszy problem niż brak encji - tylko FIX_AND_RETRY
            if v38_action_override != "FIX_AND_RETRY":
                v38_action_override = "FIX_AND_RETRY"
            
            for rel in must_relations_missing[:3]:
                v38_fixes_needed.append(
                    f"[RELATION] Ustanów relację: '{rel.get('subject')}' → {rel.get('relation')} → '{rel.get('object')}'"
                )
            v38_overrides.append(f"relations: {len(must_relations_missing)} MUST missing → FIX_AND_RETRY")
            print(f"[TRACKER] 🔴 v38.2 OVERRIDE: Relations MUST missing → action=FIX_AND_RETRY")
    
    # 5. 🆕 v38.2: PROXIMITY CLUSTERS - soft validation (tylko warning, nie blokuje)
    # proximity_score and isolated_keywords are already computed above (lines ~919-973)
    
    if proximity_score < 60 and isolated_keywords:
        # Nie blokujemy, ale dodajemy do fixes_needed jako sugestię
        v38_fixes_needed.append(
            f"[PROXIMITY] Słabe powiązanie fraz: {', '.join(isolated_keywords[:3])} - rozważ lepszą integrację"
        )
        v38_overrides.append(f"proximity_clusters: score={proximity_score}, isolated={len(isolated_keywords)} (warning only)")
        warnings.append(f"⚠️ Proximity clusters: {len(isolated_keywords)} izolowanych fraz")
        print(f"[TRACKER] ⚠️ v38.2: Proximity clusters weak (score={proximity_score})")
    
    # 6. 🆕 v38.2: ENTITY DRIFT DETECTION - wykrywa zmiany definicji encji
    entity_drifts = []
    if ENTITY_VALIDATOR_ENABLED and project_data.get("entity_state"):
        try:
            batches_done = len(project_data.get("batches", []))
            entity_drifts = detect_entity_drift(
                batch_text=batch_text,
                entity_state=project_data.get("entity_state", {}),
                batch_number=batches_done + 1
            )
            
            if entity_drifts:
                critical_drifts = [d for d in entity_drifts if d.get("severity") == "CRITICAL"]
                warning_drifts = [d for d in entity_drifts if d.get("severity") == "WARNING"]
                
                # CRITICAL drift → FIX_AND_RETRY
                if critical_drifts and v38_action_override not in ["REWRITE"]:
                    v38_action_override = "FIX_AND_RETRY"
                    for drift in critical_drifts[:2]:
                        v38_fixes_needed.append(
                            f"[DRIFT CRITICAL] '{drift['entity']}' zmienia definicję z '{drift['old_category']}' na '{drift['new_category']}' - NAPRAW!"
                        )
                    v38_overrides.append(f"entity_drift: {len(critical_drifts)} CRITICAL → FIX_AND_RETRY")
                    print(f"[TRACKER] 🔴 v38.2 OVERRIDE: Entity drift CRITICAL → action=FIX_AND_RETRY")
                
                # Warning drifts → tylko info
                for drift in warning_drifts[:2]:
                    warnings.append(f"⚠️ Entity drift: '{drift['entity']}' - {drift['message']}")
                
                print(f"[TRACKER] 🔍 Entity drift detection: {len(critical_drifts)} critical, {len(warning_drifts)} warnings")
        except Exception as e:
            print(f"[TRACKER] ⚠️ Entity drift detection error: {e}")
    
    # 7. ZASTOSUJ OVERRIDE do gpt_action
    if v38_action_override and gpt_action:
        original_action = gpt_action.get("action", "CONTINUE")
        
        # Override tylko jeśli nowy action jest "gorszy"
        action_priority = {"CONTINUE": 0, "FIX_AND_RETRY": 1, "REWRITE": 2}
        
        if action_priority.get(v38_action_override, 0) > action_priority.get(original_action, 0):
            gpt_action["action"] = v38_action_override
            gpt_action["accepted"] = False
            gpt_action["confidence"] = "HIGH"
            gpt_action["message"] = f"❌ v38.1 Override: {v38_action_override} (was: {original_action})"
            
            # Dodaj fixes do istniejących
            existing_fixes = gpt_action.get("fixes_needed", [])
            gpt_action["fixes_needed"] = v38_fixes_needed + existing_fixes
            
            print(f"[TRACKER] ✅ v38.1 Applied override: {original_action} → {v38_action_override}")
    
    # Jeśli nie ma gpt_action ale mamy override - stwórz
    elif v38_action_override and not gpt_action:
        gpt_action = {
            "accepted": False,
            "action": v38_action_override,
            "confidence": "HIGH",
            "message": f"❌ v38.1: {v38_action_override}",
            "fixes_needed": v38_fixes_needed,
            "next_task": None
        }
        print(f"[TRACKER] ✅ v38.1 Created action: {v38_action_override}")
    
    # Dodaj v38 overrides do decision_trace
    if quality_result and v38_overrides:
        if "decision_trace" not in quality_result:
            quality_result["decision_trace"] = []
        quality_result["decision_trace"].extend(v38_overrides)

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
        # 🆕 v38.3: STRUCTURAL exceeded (nie blokują, global check)
        "structural_exceeded": structural_exceeded if 'structural_exceeded' in dir() else [],
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
        # 🆕 v37.1: MoE Validation - używamy safe_to_dict
        "moe_validation": safe_to_dict(moe_validation_result) if 'safe_to_dict' in dir() else (moe_validation_result.to_dict() if moe_validation_result and hasattr(moe_validation_result, 'to_dict') else None),
        "moe_fix_instructions": moe_fix_instructions,
        # 🆕 v42.1: Content Surgery info
        "surgery_applied": moe_validation_result.surgery_applied if moe_validation_result and hasattr(moe_validation_result, 'surgery_applied') else False,
        "surgery_stats": getattr(moe_validation_result, 'surgery_stats', {}) if moe_validation_result else {},
        # 🆕 v37.1: Batch Review + Auto-Fix - używamy safe_to_dict
        "batch_review": safe_to_dict(batch_review_result) if 'safe_to_dict' in dir() else (batch_review_result.to_dict() if batch_review_result and hasattr(batch_review_result, 'to_dict') else None),
        "auto_fixed": auto_fixed_text is not None,
        "auto_fixed_text": auto_fixed_text,
        "claude_fixes_needed": getattr(batch_review_result, 'claude_fixes_needed', []) if batch_review_result else [],
        # 🆕 v36.9: Info o postępie i auto-merge
        "batch_number": batches_done,
        "total_planned_batches": total_planned,
        "remaining_batches": remaining_batches,
        "is_last_batch": is_last_batch,
        "auto_merge": auto_merge_result,
        # 🆕 v38.4: Final review z tolerancją 30%
        "final_review": final_review_result if 'final_review_result' in dir() else None,
        "article_complete": is_last_batch and auto_merge_result and auto_merge_result.get("merged", False),
        "export_ready": is_last_batch and (not final_review_result or final_review_result.get("status") == "PASS"),
        "next_step": (
            "GET /api/project/{id}/export/docx" 
            if is_last_batch and (not final_review_result or final_review_result.get("status") == "PASS")
            else f"Continue with batch {batches_done + 1}" if not is_last_batch
            else "FIX exceeded keywords (see final_review.exceeded)"
        ),
        # 🆕 v37.4: Global Quality Score
        "quality": {
            "score": quality_result["score"] if quality_result else None,
            "grade": quality_result["grade"] if quality_result else None,
            "status": quality_result["status"] if quality_result else None,
            "components": quality_result.get("components") if quality_result else None,
            "max_grade_cap": quality_result.get("max_grade_cap") if quality_result else None
        } if quality_result else None,
        # 🆕 v37.4: GPT Action (uproszczona decyzja)
        "gpt_action": {
            "accepted": gpt_action["accepted"] if gpt_action else None,
            "action": gpt_action["action"] if gpt_action else None,
            "confidence": gpt_action["confidence"] if gpt_action else None,
            "message": gpt_action["message"] if gpt_action else None,
            "fixes_needed": gpt_action.get("fixes_needed", []) if gpt_action else [],
            "next_task": gpt_action.get("next_task") if gpt_action else None
        } if gpt_action else None,
        # 🆕 v37.4: Decision trace (audit)
        "decision_trace": quality_result.get("decision_trace", []) if quality_result else [],
        # 🆕 v38.1: Entity Coverage Validation
        "entity_validation": {
            "status": entity_validation_result.status if entity_validation_result else None,
            "score": entity_validation_result.score if entity_validation_result else None,
            "must_missing": entity_validation_result.must_missing if entity_validation_result else [],
            "should_missing": entity_validation_result.should_missing if entity_validation_result else [],
            "relationships_established": entity_validation_result.relationships_established if entity_validation_result else []
        } if entity_validation_result else None,
        # 🆕 v38.1: Legal Hard-Lock Validation
        "legal_validation": {
            "is_valid": legal_validation_result.is_valid if legal_validation_result else True,
            "violations_count": len(legal_validation_result.violations) if legal_validation_result else 0,
            "auto_removed_count": len(legal_validation_result.removals) if legal_validation_result else 0,
            "removals": [safe_to_dict(r) if 'safe_to_dict' in dir() else (r.to_dict() if hasattr(r, 'to_dict') else str(r)) for r in legal_validation_result.removals] if legal_validation_result and hasattr(legal_validation_result, 'removals') else []
        } if legal_validation_result else None,
        # 🆕 v38.1: Override info
        "v38_overrides": v38_overrides if 'v38_overrides' in dir() and v38_overrides else None,
        # 🆕 v38.2: Entity drift detection
        "entity_drifts": entity_drifts if 'entity_drifts' in dir() and entity_drifts else None,
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


# ============================================================================
# 🆕 v37.4: FAST VALIDATE - szybka walidacja (~100ms)
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/fast_validate")
def fast_validate_batch(project_id):
    """
    Szybka walidacja batcha (~100ms zamiast ~1200ms).
    
    Sprawdza TYLKO krytyczne rzeczy:
    1. Exceeded keywords (CRITICAL)
    2. Struktura (H2)
    3. Burstiness (AI detection)
    
    Używaj do szybkiego feedbacku przed pełnym POST /batch.
    
    Request:
        {"text": "h2: Tytuł\n\nTreść batcha..."}
    
    Response:
        {
            "status": "OK" | "WARNING" | "REJECTED",
            "action": "CONTINUE" | "REWRITE",
            "message": "...",
            "exceeded_critical": [...],
            "exceeded_warning": [...],
            "decision_trace": [...]
        }
    """
    if not QUALITY_SCORE_ENABLED:
        return jsonify({
            "error": "Quality Score module not available",
            "hint": "Install quality_score_module.py"
        }), 503
    
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Pobierz projekt
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Policz keywords w tekście (uproszczone)
    batch_counts = {}
    text_lower = text.lower()
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if keyword and keyword in text_lower:
            # Proste liczenie (fast mode)
            count = text_lower.count(keyword)
            if count > 0:
                batch_counts[rid] = count
    
    # Fast validate
    result = validate_fast_mode(text, keywords_state, batch_counts)
    
    return jsonify(result), 200


# ============================================================================
# 🆕 v37.4: GET QUALITY - pobierz jakość ostatniego batcha
# ============================================================================
@tracker_routes.get("/api/project/<project_id>/quality")
def get_project_quality(project_id):
    """
    Pobiera quality score dla projektu (ostatni batch + ogólnie).
    
    Response:
        {
            "last_batch_quality": {...},
            "overall_progress": {...},
            "recommendations": [...]
        }
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    if not batches:
        return jsonify({
            "error": "No batches yet",
            "hint": "Submit first batch with POST /batch"
        }), 200
    
    # Ostatni batch
    last_batch = batches[-1]
    last_quality = last_batch.get("quality", {})
    
    # Statystyki ogólne
    total_batches = len(batches)
    planned_batches = project_data.get("total_planned_batches", 7)
    
    # Zbierz quality scores ze wszystkich batchów
    quality_scores = []
    for b in batches:
        q = b.get("quality", {})
        if q and q.get("score"):
            quality_scores.append(q["score"])
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
    
    # Rekomendacje
    recommendations = []
    keywords_state = project_data.get("keywords_state", {})
    
    # Frazy UNDER
    under_keywords = [
        meta.get("keyword") for rid, meta in keywords_state.items()
        if meta.get("status") == "UNDER" and meta.get("type", "").upper() in ["BASIC", "MAIN", "ENTITY"]
    ]
    if under_keywords:
        recommendations.append({
            "type": "UNDER_KEYWORDS",
            "message": f"{len(under_keywords)} keywords still need more uses",
            "keywords": under_keywords[:5]
        })
    
    # Frazy OVER
    over_keywords = [
        meta.get("keyword") for rid, meta in keywords_state.items()
        if meta.get("status") in ["OVER", "LOCKED"]
    ]
    if over_keywords:
        recommendations.append({
            "type": "OVER_KEYWORDS",
            "message": f"{len(over_keywords)} keywords exceeded - use synonyms",
            "keywords": over_keywords[:5]
        })
    
    return jsonify({
        "project_id": project_id,
        "last_batch_quality": last_quality,
        "overall_progress": {
            "batches_done": total_batches,
            "batches_planned": planned_batches,
            "progress_percent": round(total_batches / planned_batches * 100) if planned_batches else 0,
            "avg_quality_score": round(avg_quality) if avg_quality else None
        },
        "recommendations": recommendations
    }), 200


# ============================================================================
# 🆕 v37.4: SIMPLIFIED RESPONSE ENDPOINT
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/batch_simple")
def submit_batch_simple(project_id):
    """
    Alternatywny endpoint z uproszczonym response dla GPT.

    Zamiast 50+ pól, zwraca tylko to co potrzebne:
    - accepted: true/false
    - action: CONTINUE/FIX_AND_RETRY/REWRITE
    - quality: {score, grade, status}
    - issues: [top 4 issues]
    - fixes_needed: [top 3 fixes]
    - next_task: {batch_number, h2, action}

    Request:
        {"text": "h2: Tytuł\n\nTreść batcha..."}
    """
    if not QUALITY_SCORE_ENABLED:
        # Graceful degradation — zaakceptuj batch bez quality check
        data = request.get_json() or {}
        if not data.get("text", "").strip():
            return jsonify({"error": "No text provided"}), 400
        return jsonify({
            "accepted": True,
            "action": "CONTINUE",
            "confidence": 0.5,
            "message": "Quality Score unavailable — batch accepted without scoring",
            "quality": {"score": None, "grade": "N/A", "status": "quality_module_offline"},
            "issues": [],
            "fixes_needed": [],
            "fixes_applied": [],
            "depth_score": None,
            "depth_shallow_sections": None,
            "next_task": {"action": "CONTINUE"}
        }), 200

    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        forced = data.get("forced", False)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Standardowe przetwarzanie
        result = process_batch_in_firestore(project_id, text, {}, forced)

        if isinstance(result, dict) and result.get("status_code") == 404:
            return jsonify({"error": "Project not found"}), 404

        # Pobierz project_data dla simplified response
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        project_data = doc.to_dict() if doc.exists else {}

        batch_number = len(project_data.get("batches", []))

        # Utwórz simplified response
        simplified = create_simplified_response(result, project_data, batch_number, forced=forced)

        return jsonify(simplified), 200

    except Exception as e:
        # Fix #36: Catch-all aby batch_simple nigdy nie zwracał 500 bez info
        import traceback
        tb = traceback.format_exc()
        print(f"[BATCH_SIMPLE] ❌ EXCEPTION for {project_id}: {str(e)[:300]}")
        print(f"[BATCH_SIMPLE] Traceback:\n{tb[-500:]}")
        return jsonify({
            "accepted": True,
            "action": "CONTINUE",
            "confidence": 0.3,
            "message": f"Internal error caught — batch force-accepted: {str(e)[:150]}",
            "quality": {"score": None, "grade": "N/A", "status": "error_recovery"},
            "issues": [f"Server error: {str(e)[:100]}"],
            "fixes_needed": [],
            "fixes_applied": [],
            "depth_score": None,
            "depth_shallow_sections": None,
            "exceeded_keywords": [],
            "next_task": {"action": "CONTINUE"},
            "_error": str(e)[:200],
            "_traceback": tb[-300:]
        }), 200
