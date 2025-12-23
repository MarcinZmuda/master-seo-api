"""
===============================================================================
🧠 SHARED NLP v23.0 - Współdzielony model spaCy
===============================================================================
Rozwiązuje problem wielokrotnego ładowania modelu spaCy.

Zamiast:
    unified_validator.py    → nlp = spacy.load()  # 150MB
    project_routes_v23.py   → nlp = spacy.load()  # 150MB
    entity_ngram_analyzer.py → nlp = spacy.load() # 150MB
    polish_language_quality.py → nlp = spacy.load() # 150MB
    RAZEM: ~600MB RAM

Teraz:
    shared_nlp.py → nlp = spacy.load()  # 150MB (raz!)
    Wszystkie moduły: from shared_nlp import nlp

OSZCZĘDNOŚĆ: ~450MB RAM
===============================================================================
"""

import spacy
import os

# ================================================================
# 🔧 KONFIGURACJA
# ================================================================
# Domyślny model - można zmienić przez env
SPACY_MODEL = os.getenv("SPACY_MODEL", "pl_core_news_md")

# Alternatywy:
# - pl_core_news_sm  (15MB)  - szybki, mniej dokładny
# - pl_core_news_md  (50MB)  - balans (DOMYŚLNY)
# - pl_core_news_lg  (150MB) - najdokładniejszy, więcej RAM

# ================================================================
# 🧠 SINGLETON - JEDEN MODEL DLA CAŁEJ APLIKACJI
# ================================================================
_nlp_instance = None

def get_nlp():
    """
    Zwraca współdzieloną instancję modelu spaCy.
    Ładuje model tylko przy pierwszym wywołaniu (lazy loading).
    """
    global _nlp_instance
    
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load(SPACY_MODEL)
            print(f"[SHARED_NLP] ✅ Załadowano model: {SPACY_MODEL}")
        except OSError:
            print(f"[SHARED_NLP] ⚠️ Model {SPACY_MODEL} nie znaleziony, pobieram...")
            from spacy.cli import download
            download(SPACY_MODEL)
            _nlp_instance = spacy.load(SPACY_MODEL)
            print(f"[SHARED_NLP] ✅ Pobrano i załadowano: {SPACY_MODEL}")
    
    return _nlp_instance


# ================================================================
# 🔗 EKSPORT - dla kompatybilności wstecznej
# ================================================================
# Moduły mogą używać:
#   from shared_nlp import nlp
# lub:
#   from shared_nlp import get_nlp
#   nlp = get_nlp()

# Lazy loading przy imporcie
nlp = None

def __getattr__(name):
    """Lazy loading przy pierwszym użyciu `nlp`."""
    if name == "nlp":
        global nlp
        if nlp is None:
            nlp = get_nlp()
        return nlp
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
