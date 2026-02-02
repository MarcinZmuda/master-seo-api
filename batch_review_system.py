"""
BATCH REVIEW SYSTEM v44.2 - HELPER MODULE
==========================================
Wydzielony moduł pomocniczy z funkcjami review.

v44.2 ZMIANY:
- Usunięto ~2300 linii martwego kodu (nieużywany Blueprint, zduplikowane endpointy)
- Zachowano TYLKO funkcje importowane przez firestore_tracker_routes.py
- Plik służy jako wrapper dla claude_reviewer.py + stub funkcje

EKSPORTY:
- review_batch_comprehensive() - wrapper dla claude_reviewer
- detect_helpful_reflex() - detekcja "helpful reflex" w YMYL
- auto_remove_fillers() - usuwanie fillerów
- SmartFixResult, ReviewResult - klasy pomocnicze

UŻYCIE:
    from batch_review_system import (
        review_batch_comprehensive,
        detect_helpful_reflex,
        auto_remove_fillers,
        SmartFixResult,
        ReviewResult
    )
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
    print("[BATCH_REVIEW] ⚠️ core_metrics not available, using local functions")

# 🆕 v40.2: Style Analyzer integration
try:
    from style_analyzer import StyleAnalyzer, StyleFingerprint
    STYLE_ANALYZER_AVAILABLE = True
    _style_analyzer = StyleAnalyzer()
    print("[BATCH_REVIEW] ✅ style_analyzer loaded")
except ImportError:
    STYLE_ANALYZER_AVAILABLE = False
    _style_analyzer = None
    print("[BATCH_REVIEW] ⚠️ style_analyzer not available")

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

# 🆕 v37.1: Batch Review System z auto-poprawkami
# ⚠️ FIX v1.2: Usunięto circular import (plik importował sam siebie)
# ⚠️ FIX v1.3: Naprawiono niezgodność sygnatur review_batch
# Te funkcje powinny być zaimportowane z claude_reviewer.py lub zdefiniowane tutaj
try:
    from claude_reviewer import (
        ReviewResult,
        review_batch as _review_batch_original,  # oryginalna funkcja
    )
    
    # 🆕 FIX v1.3: Wrapper dostosowujący sygnatury
    def review_batch_comprehensive(
        batch_text: str = None,
        text: str = None,  # alternatywna nazwa
        keywords_state: dict = None,
        batch_counts: dict = None,
        previous_batch_text: str = None,
        auto_fix: bool = True,
        use_claude_for_complex: bool = False,
        context: dict = None,  # dla kompatybilności wstecznej
        **kwargs
    ):
        """
        Wrapper dla review_batch z claude_reviewer.py
        Tłumaczy stare argumenty na nowy format.
        """
        # Użyj text lub batch_text
        actual_text = batch_text or text or ""
        
        # Buduj context jeśli nie podany
        if context is None:
            context = {}
        
        # Dodaj argumenty do context
        if keywords_state:
            context["keywords_state"] = keywords_state
            # Buduj required_keywords z keywords_state
            required = []
            for kw_id, meta in keywords_state.items():
                if isinstance(meta, dict):
                    required.append({
                        "phrase": meta.get("keyword", kw_id),
                        "min": meta.get("target_min", 1),
                        "max": meta.get("target_max", 5),
                        "priority": meta.get("priority", "EXTENDED")
                    })
            context["required_keywords"] = required
        
        if batch_counts:
            context["batch_counts"] = batch_counts
        
        if previous_batch_text:
            context["previous_batch_text"] = previous_batch_text
        
        # skip_claude = not use_claude_for_complex
        skip_claude = not use_claude_for_complex
        
        try:
            return _review_batch_original(actual_text, context, skip_claude)
        except Exception as e:
            print(f"[BATCH_REVIEW] ⚠️ review_batch error: {e}")
            # Zwróć pusty wynik zamiast None
            return ReviewResult(
                status="ERROR",
                original_text=actual_text,
                corrected_text=None,
                issues=[],
                summary=f"Review error: {e}",
                word_count=len(actual_text.split()),
                paragraph_count=len(actual_text.split('\n\n'))
            )
    
    # Stub dla brakujących funkcji
    def get_review_summary(result): return result.__dict__ if result else {}
    def generate_claude_fix_prompt(text, issues): return f"Fix: {issues}"
    def claude_smart_fix(text, context): return text
    def should_use_claude_smart_fix(issues): return False
    def build_smart_fix_prompt(text, pre_batch): return text
    def get_pre_batch_info_for_claude(project): return {}
    class SmartFixResult:
        def __init__(self, text="", fixes=None):
            self.fixed_text = text
            self.auto_fixes_applied = fixes or []
    BATCH_REVIEW_ENABLED = True
    print("[TRACKER] ✅ Batch Review System loaded (from claude_reviewer)")
except ImportError as e:
    BATCH_REVIEW_ENABLED = False
    print(f"[TRACKER] ⚠️ Batch Review System not available: {e}")
    # Fallback stubs
    class ReviewResult:
        status: str = "ERROR"
        original_text: str = ""
        corrected_text: str = None
        issues: list = None
        summary: str = ""
        word_count: int = 0
        paragraph_count: int = 0
        fixed_text: str = None
        auto_fixes_applied: list = None
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.issues = self.issues or []
            self.auto_fixes_applied = self.auto_fixes_applied or []
        
        def to_dict(self):
            """Konwertuje ReviewResult na dict."""
            return {
                "status": getattr(self, "status", "ERROR"),
                "original_text": getattr(self, "original_text", ""),
                "corrected_text": getattr(self, "corrected_text", None),
                "issues": getattr(self, "issues", []),
                "summary": getattr(self, "summary", ""),
                "word_count": getattr(self, "word_count", 0),
                "paragraph_count": getattr(self, "paragraph_count", 0),
                "fixed_text": getattr(self, "fixed_text", None),
                "auto_fixes_applied": getattr(self, "auto_fixes_applied", [])
            }
    class SmartFixResult:
        def __init__(self, text="", fixes=None):
            self.fixed_text = text
            self.auto_fixes_applied = fixes or []
    def review_batch_comprehensive(*args, **kwargs): return None
    def get_review_summary(result): return {}
    def generate_claude_fix_prompt(text, issues): return ""
    def claude_smart_fix(text, context): return text
    def should_use_claude_smart_fix(issues): return False
    def build_smart_fix_prompt(text, pre_batch): return text
    def get_pre_batch_info_for_claude(project): return {}

# 🆕 v37.1: MoE Batch Validator
# ⚠️ FIX v1.2: Lazy import - unikamy circular import
# Import przeniesiony do funkcji process_batch_in_firestore() gdzie jest używany
MOE_VALIDATOR_ENABLED = True  # Zakładamy że jest dostępny, sprawdzimy w runtime

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

# 🆕 v40.1: Legal Post-Validator (walidacja poprawności prawnej po generacji)
try:
    from legal_post_validator import validate_legal_content, validate_batch_legal_content
    LEGAL_POST_VALIDATOR_ENABLED = True
    print("[TRACKER] ✅ Legal Post-Validator loaded")
except ImportError as e:
    LEGAL_POST_VALIDATOR_ENABLED = False
    print(f"[TRACKER] ⚠️ Legal Post-Validator not available: {e}")

# 🆕 v38: Helpful Reflex Detector
# ⚠️ FIX v1.2: Usunięto circular import - definiujemy stub funkcje lokalnie
HELPFUL_REFLEX_ENABLED = False  # Wyłączone - brak implementacji

def detect_helpful_reflex(text: str, is_ymyl: bool = False) -> list:
    """Stub - wykrywa 'helpful reflex' patterns w tekście YMYL."""
    return []

def auto_remove_fillers(text: str) -> str:
    """Stub - usuwa fillery z tekstu."""
    return text

# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    # Klasy
    'ReviewResult',
    'SmartFixResult',
    
    # Funkcje review
    'review_batch_comprehensive',
    'get_review_summary',
    'generate_claude_fix_prompt',
    'claude_smart_fix',
    'should_use_claude_smart_fix',
    'build_smart_fix_prompt',
    'get_pre_batch_info_for_claude',
    
    # Helpful reflex
    'detect_helpful_reflex',
    'auto_remove_fillers',
    
    # Flagi
    'BATCH_REVIEW_ENABLED',
    'HELPFUL_REFLEX_ENABLED'
]
