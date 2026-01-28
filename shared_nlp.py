"""
===============================================================================
🧠 SHARED NLP v40.2 - Współdzielony model spaCy
===============================================================================
Rozwiązuje problem wielokrotnego ładowania modelu spaCy.

v40.2 UPDATE:
- Używa nlp_config.py jako backendu (lepsze modele, auto-upgrade)
- Zachowana kompatybilność wsteczna
- Automatyczny fallback do starszej implementacji

OSZCZĘDNOŚĆ: ~450MB RAM (jeden model dla całej aplikacji)
===============================================================================
"""

import os

# ================================================================
# 🔧 KONFIGURACJA
# ================================================================
SPACY_MODEL = os.getenv("SPACY_MODEL", "pl_core_news_md")

# ================================================================
# 🆕 v40.2: Try to use nlp_config.py (better models, auto-upgrade)
# ================================================================
_nlp_instance = None
_backend = "legacy"

try:
    from nlp_config import get_nlp as _get_nlp_v2, get_nlp_info
    _backend = "nlp_config"
    print("[SHARED_NLP] ✅ v40.2: Using nlp_config backend (better models)")
except ImportError:
    print("[SHARED_NLP] ⚠️ nlp_config not available, using legacy spacy.load")
    import spacy


def get_nlp():
    """
    Zwraca współdzieloną instancję modelu spaCy.
    Ładuje model tylko przy pierwszym wywołaniu (lazy loading).
    
    v40.2: Preferuje nlp_config.py (lepsze modele), fallback do legacy
    """
    global _nlp_instance
    
    if _nlp_instance is not None:
        return _nlp_instance
    
    # v40.2: Preferuj nlp_config
    if _backend == "nlp_config":
        try:
            _nlp_instance = _get_nlp_v2()
            info = get_nlp_info()
            print(f"[SHARED_NLP] ✅ Model loaded via nlp_config: {info.get('loaded_model', 'unknown')}")
            return _nlp_instance
        except Exception as e:
            print(f"[SHARED_NLP] ⚠️ nlp_config failed: {e}, falling back to legacy")
    
    # Legacy fallback
    try:
        _nlp_instance = spacy.load(SPACY_MODEL)
        print(f"[SHARED_NLP] ✅ Legacy model loaded: {SPACY_MODEL}")
    except OSError:
        print(f"[SHARED_NLP] ⚠️ Model {SPACY_MODEL} not found, downloading...")
        from spacy.cli import download
        download(SPACY_MODEL)
        _nlp_instance = spacy.load(SPACY_MODEL)
        print(f"[SHARED_NLP] ✅ Downloaded and loaded: {SPACY_MODEL}")
    
    return _nlp_instance


# ================================================================
# 🔗 EKSPORT - dla kompatybilności wstecznej
# ================================================================
nlp = None

def __getattr__(name):
    """Lazy loading przy pierwszym użyciu `nlp`."""
    if name == "nlp":
        global nlp
        if nlp is None:
            nlp = get_nlp()
        return nlp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================
def reload_model(model_name: str = None):
    """
    Przeładowuje model spaCy.
    Użyteczne przy zmianie modelu w runtime.
    """
    global _nlp_instance, nlp, SPACY_MODEL
    
    if model_name:
        SPACY_MODEL = model_name
    
    _nlp_instance = None
    nlp = None
    return get_nlp()


def get_model_info() -> dict:
    """Zwraca informacje o załadowanym modelu."""
    model = get_nlp()
    return {
        "model_name": SPACY_MODEL,
        "pipeline": model.pipe_names,
        "vocab_size": len(model.vocab),
        "vectors": model.vocab.vectors.shape if model.vocab.vectors else None
    }


# ================================================================
# 📊 PRE-LOAD przy starcie (opcjonalne)
# ================================================================
if os.getenv("PRELOAD_SPACY", "false").lower() == "true":
    print("[SHARED_NLP] Pre-loading spaCy model...")
    get_nlp()
