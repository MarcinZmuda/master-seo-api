"""
FEATURE FLAGS v1.0
==================
Centralne zarządzanie dostępnością modułów.

ZASTĘPUJE wzorzec powtarzany ~200 razy w projekcie:
    try:
        from module import function
        MODULE_ENABLED = True
    except ImportError:
        MODULE_ENABLED = False

UŻYCIE:
    from feature_flags import FEATURES, get_module
    
    if FEATURES.phrase_hierarchy:
        analyze = get_module('phrase_hierarchy', 'analyze_phrase_hierarchy')
        result = analyze(...)

    # Lub bezpośrednio:
    from feature_flags import safe_import
    
    content_surgeon = safe_import('content_surgeon')
    if content_surgeon:
        content_surgeon.perform_surgery(...)
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Callable
import sys


@dataclass
class FeatureFlags:
    """Stan dostępności wszystkich opcjonalnych modułów."""
    
    # Core modules (should always be available)
    core_metrics: bool = False
    firestore_utils: bool = False
    
    # SEO modules
    phrase_hierarchy: bool = False
    content_surgeon: bool = False
    semantic_triplet_validator: bool = False
    keyword_counter: bool = False
    keyword_conflict_validator: bool = False
    
    # Validation modules
    unified_validator: bool = False
    moe_batch_validator: bool = False
    ai_detection_metrics: bool = False
    polish_language_quality: bool = False
    grammar_middleware: bool = False
    
    # Planning modules
    batch_planner: bool = False
    enhanced_pre_batch: bool = False
    h2_generator: bool = False
    smart_batch_instructions: bool = False
    
    # Review modules
    claude_reviewer: bool = False
    batch_best_of_n: bool = False
    
    # Anti-patterns modules
    anti_frankenstein_integration: bool = False
    overflow_buffer: bool = False
    dynamic_sub_batch: bool = False
    
    # Legal modules
    legal_routes_v3: bool = False
    
    # Export modules
    export_routes: bool = False
    
    # Synonym modules
    keyword_synonyms: bool = False
    
    # Analysis modules
    entity_ngram_analyzer: bool = False
    style_analyzer: bool = False
    text_analyzer: bool = False
    
    # Loaded module references
    _modules: Dict[str, Any] = field(default_factory=dict)


def _try_import(module_name: str) -> tuple[bool, Optional[Any]]:
    """
    Próbuje zaimportować moduł.
    
    Returns:
        (success: bool, module: Optional[module])
    """
    try:
        # Check if already imported
        if module_name in sys.modules:
            return True, sys.modules[module_name]
        
        module = importlib.import_module(module_name)
        return True, module
    except ImportError as e:
        print(f"[FEATURES] ⚠️ {module_name} not available: {e}")
        return False, None
    except Exception as e:
        print(f"[FEATURES] ❌ {module_name} error: {e}")
        return False, None


def init_features() -> FeatureFlags:
    """
    Inicjalizuje feature flags sprawdzając dostępność modułów.
    Wywoływane raz przy starcie aplikacji.
    """
    features = FeatureFlags()
    
    # Lista modułów do sprawdzenia
    modules_to_check = [
        # Core
        ('core_metrics', 'core_metrics'),
        ('firestore_utils', 'firestore_utils'),
        
        # SEO
        ('phrase_hierarchy', 'phrase_hierarchy'),
        ('content_surgeon', 'content_surgeon'),
        ('semantic_triplet_validator', 'semantic_triplet_validator'),
        ('keyword_counter', 'keyword_counter'),
        ('keyword_conflict_validator', 'keyword_conflict_validator'),
        
        # Validation
        ('unified_validator', 'unified_validator'),
        ('moe_batch_validator', 'moe_batch_validator'),
        ('ai_detection_metrics', 'ai_detection_metrics'),
        ('polish_language_quality', 'polish_language_quality'),
        ('grammar_middleware', 'grammar_middleware'),
        
        # Planning
        ('batch_planner', 'batch_planner'),
        ('enhanced_pre_batch', 'enhanced_pre_batch'),
        ('h2_generator', 'h2_generator'),
        ('smart_batch_instructions', 'smart_batch_instructions'),
        
        # Review
        ('claude_reviewer', 'claude_reviewer'),
        ('batch_best_of_n', 'batch_best_of_n'),
        
        # Anti-patterns
        ('anti_frankenstein_integration', 'anti_frankenstein_integration'),
        ('overflow_buffer', 'overflow_buffer'),
        ('dynamic_sub_batch', 'dynamic_sub_batch'),
        
        # Legal
        ('legal_routes_v3', 'legal_routes_v3'),
        
        # Export
        ('export_routes', 'export_routes'),
        
        # Synonyms
        ('keyword_synonyms', 'keyword_synonyms'),
        
        # Analysis
        ('entity_ngram_analyzer', 'entity_ngram_analyzer'),
        ('style_analyzer', 'style_analyzer'),
        ('text_analyzer', 'text_analyzer'),
    ]
    
    enabled_count = 0
    disabled_count = 0
    
    for attr_name, module_name in modules_to_check:
        success, module = _try_import(module_name)
        setattr(features, attr_name, success)
        
        if success:
            features._modules[module_name] = module
            enabled_count += 1
        else:
            disabled_count += 1
    
    print(f"[FEATURES] ✅ Initialized: {enabled_count} enabled, {disabled_count} disabled")
    
    return features


def get_module(module_name: str) -> Optional[Any]:
    """
    Pobiera zaimportowany moduł.
    
    Args:
        module_name: Nazwa modułu
        
    Returns:
        Moduł lub None jeśli niedostępny
    """
    return FEATURES._modules.get(module_name)


def get_function(module_name: str, function_name: str) -> Optional[Callable]:
    """
    Pobiera funkcję z modułu.
    
    Args:
        module_name: Nazwa modułu
        function_name: Nazwa funkcji
        
    Returns:
        Funkcja lub None
    """
    module = get_module(module_name)
    if module is None:
        return None
    
    return getattr(module, function_name, None)


def safe_import(module_name: str) -> Optional[Any]:
    """
    Bezpieczny import modułu (dla użycia poza inicjalizacją).
    
    Jeśli moduł już zainicjalizowany, zwraca z cache.
    W przeciwnym razie próbuje zaimportować.
    
    Args:
        module_name: Nazwa modułu
        
    Returns:
        Moduł lub None
    """
    # Check cache first
    if module_name in FEATURES._modules:
        return FEATURES._modules[module_name]
    
    # Try to import
    success, module = _try_import(module_name)
    
    if success and module:
        FEATURES._modules[module_name] = module
        return module
    
    return None


def require_feature(feature_name: str) -> Callable:
    """
    Dekorator sprawdzający czy feature jest dostępny.
    
    Usage:
        @require_feature('phrase_hierarchy')
        def analyze_phrases():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if not getattr(FEATURES, feature_name, False):
                raise RuntimeError(f"Feature '{feature_name}' is not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_enabled(feature_name: str) -> bool:
    """
    Sprawdza czy feature jest włączony.
    
    Args:
        feature_name: Nazwa feature'a
        
    Returns:
        True jeśli włączony
    """
    return getattr(FEATURES, feature_name, False)


# ============================================================
# LAZY IMPORTS HELPERS
# Używaj tych funkcji zamiast bezpośrednich importów
# ============================================================

def get_phrase_hierarchy():
    """Lazy import phrase_hierarchy module."""
    return safe_import('phrase_hierarchy')


def get_content_surgeon():
    """Lazy import content_surgeon module."""
    return safe_import('content_surgeon')


def get_unified_validator():
    """Lazy import unified_validator module."""
    return safe_import('unified_validator')


def get_keyword_counter():
    """Lazy import keyword_counter module."""
    return safe_import('keyword_counter')


def get_batch_planner():
    """Lazy import batch_planner module."""
    return safe_import('batch_planner')


def get_enhanced_pre_batch():
    """Lazy import enhanced_pre_batch module."""
    return safe_import('enhanced_pre_batch')


def get_claude_reviewer():
    """Lazy import claude_reviewer module."""
    return safe_import('claude_reviewer')


def get_core_metrics():
    """Lazy import core_metrics module."""
    return safe_import('core_metrics')


# ============================================================
# SINGLETON INITIALIZATION
# ============================================================

# Initialize on module load
FEATURES = init_features()


# ============================================================
# VERSION INFO
# ============================================================

__version__ = "1.0"
__all__ = [
    "FEATURES",
    "FeatureFlags",
    "get_module",
    "get_function",
    "safe_import",
    "require_feature",
    "is_enabled",
    # Lazy imports
    "get_phrase_hierarchy",
    "get_content_surgeon",
    "get_unified_validator",
    "get_keyword_counter",
    "get_batch_planner",
    "get_enhanced_pre_batch",
    "get_claude_reviewer",
    "get_core_metrics",
]
