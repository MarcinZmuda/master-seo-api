"""
===============================================================================
NLP CONFIG v40.2
===============================================================================
Zunifikowana konfiguracja NLP dla języka polskiego.

UPGRADE PATH:
1. pl_spacy_model_morfeusz (IPI PAN + Morfeusz2) - BEST
2. pl_core_news_lg (spaCy large) - GOOD  
3. pl_core_news_md (spaCy medium) - OK (current)
4. pl_core_news_sm (spaCy small) - FALLBACK

UŻYCIE:
    from nlp_config import get_nlp, get_nlp_light, NLP_INFO
    
    nlp = get_nlp()  # Best available model
    doc = nlp("Tekst do analizy")
    
INSTALACJA MORFEUSZ2 (opcjonalne, dla najlepszej jakości):
    pip install morfeusz2
    pip install pl_spacy_model_morfeusz
===============================================================================
"""

import os
import warnings
from typing import Optional, Dict, Any
from functools import lru_cache

# ============================================================
# KONFIGURACJA
# ============================================================

# Tryby NLP
NLP_MODES = {
    "best": {
        "models": [
            "pl_spacy_model_morfeusz",  # IPI PAN + Morfeusz2 (najlepsza lemmatyzacja)
            "pl_core_news_lg",           # spaCy large
            "pl_core_news_md",           # spaCy medium
        ],
        "description": "Najlepsza jakość, wolniejsze"
    },
    "balanced": {
        "models": [
            "pl_core_news_lg",
            "pl_core_news_md",
        ],
        "description": "Dobra jakość, średnia prędkość"
    },
    "fast": {
        "models": [
            "pl_core_news_md",
            "pl_core_news_sm",
        ],
        "description": "Szybkie, podstawowa jakość"
    },
    "minimal": {
        "models": [
            "pl_core_news_sm",
        ],
        "description": "Minimalne wymagania"
    }
}

# Domyślny tryb (można zmienić przez env var)
DEFAULT_MODE = os.environ.get("NLP_MODE", "balanced")

# Global cache dla załadowanych modeli
_nlp_cache: Dict[str, Any] = {}
_nlp_info: Dict[str, Any] = {
    "loaded_model": None,
    "mode": None,
    "has_morfeusz": False,
    "available_models": []
}


# ============================================================
# MODEL LOADING
# ============================================================

def _check_available_models() -> list:
    """Sprawdza które modele są dostępne."""
    available = []
    
    try:
        import spacy
    except ImportError:
        warnings.warn("[NLP_CONFIG] spaCy not installed!")
        return available
    
    models_to_check = [
        "pl_spacy_model_morfeusz",
        "pl_core_news_lg",
        "pl_core_news_md",
        "pl_core_news_sm",
    ]
    
    for model_name in models_to_check:
        try:
            spacy.load(model_name)
            available.append(model_name)
        except OSError:
            pass
    
    return available


def _check_morfeusz() -> bool:
    """Sprawdza czy Morfeusz2 jest dostępny."""
    try:
        import morfeusz2
        return True
    except ImportError:
        return False


def _load_model(model_name: str):
    """Ładuje model spaCy."""
    import spacy
    
    # Sprawdź cache
    if model_name in _nlp_cache:
        return _nlp_cache[model_name]
    
    try:
        nlp = spacy.load(model_name)
        _nlp_cache[model_name] = nlp
        print(f"[NLP_CONFIG] ✅ Loaded: {model_name}")
        return nlp
    except OSError as e:
        print(f"[NLP_CONFIG] ⚠️ Failed to load {model_name}: {e}")
        return None


def _auto_download_model(model_name: str) -> bool:
    """Próbuje pobrać brakujący model."""
    try:
        from spacy.cli import download
        print(f"[NLP_CONFIG] 📦 Downloading {model_name}...")
        download(model_name)
        return True
    except Exception as e:
        print(f"[NLP_CONFIG] ❌ Download failed: {e}")
        return False


# ============================================================
# PUBLIC API
# ============================================================

@lru_cache(maxsize=1)
def get_nlp(mode: str = None, auto_download: bool = True):
    """
    Zwraca najlepszy dostępny model NLP.
    
    Args:
        mode: "best", "balanced", "fast", "minimal" (default: env NLP_MODE or "balanced")
        auto_download: Czy automatycznie pobierać brakujące modele
        
    Returns:
        spaCy Language model
        
    Example:
        nlp = get_nlp()
        doc = nlp("Ala ma kota.")
        for token in doc:
            print(token.text, token.lemma_, token.pos_)
    """
    global _nlp_info
    
    if mode is None:
        mode = DEFAULT_MODE
    
    if mode not in NLP_MODES:
        warnings.warn(f"[NLP_CONFIG] Unknown mode '{mode}', using 'balanced'")
        mode = "balanced"
    
    config = NLP_MODES[mode]
    models_to_try = config["models"]
    
    # Sprawdź dostępność Morfeusz
    _nlp_info["has_morfeusz"] = _check_morfeusz()
    _nlp_info["mode"] = mode
    
    # Próbuj załadować modele w kolejności preferencji
    for model_name in models_to_try:
        # Skip Morfeusz model jeśli Morfeusz nie jest zainstalowany
        if "morfeusz" in model_name and not _nlp_info["has_morfeusz"]:
            print(f"[NLP_CONFIG] ⏭️ Skipping {model_name} (Morfeusz2 not installed)")
            continue
        
        nlp = _load_model(model_name)
        
        if nlp is not None:
            _nlp_info["loaded_model"] = model_name
            return nlp
        
        # Próba auto-download
        if auto_download and model_name.startswith("pl_core_news"):
            if _auto_download_model(model_name):
                nlp = _load_model(model_name)
                if nlp is not None:
                    _nlp_info["loaded_model"] = model_name
                    return nlp
    
    # Ultimate fallback - pl_core_news_sm
    if auto_download:
        if _auto_download_model("pl_core_news_sm"):
            nlp = _load_model("pl_core_news_sm")
            if nlp is not None:
                _nlp_info["loaded_model"] = "pl_core_news_sm"
                return nlp
    
    raise RuntimeError("[NLP_CONFIG] No Polish NLP model available! Run: python -m spacy download pl_core_news_md")


def get_nlp_light():
    """
    Zwraca lekki model NLP (dla szybkich operacji).
    
    Używa trybu "fast".
    """
    return get_nlp(mode="fast")


def get_nlp_info() -> Dict[str, Any]:
    """
    Zwraca informacje o załadowanym modelu NLP.
    
    Returns:
        Dict z: loaded_model, mode, has_morfeusz, available_models
    """
    if _nlp_info["available_models"] == []:
        _nlp_info["available_models"] = _check_available_models()
    
    return _nlp_info.copy()


def reload_nlp(mode: str = None):
    """
    Wymusza przeładowanie modelu NLP.
    
    Użyteczne po instalacji nowych modeli.
    """
    global _nlp_cache
    _nlp_cache.clear()
    get_nlp.cache_clear()
    return get_nlp(mode=mode)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lemmatize(text: str, nlp=None) -> str:
    """
    Lemmatyzuje tekst.
    
    Args:
        text: Tekst do lemmatyzacji
        nlp: Model NLP (opcjonalnie)
        
    Returns:
        str: Tekst z lemmatami
    """
    if nlp is None:
        nlp = get_nlp()
    
    doc = nlp(text[:50000])  # Limit dla wydajności
    return " ".join([token.lemma_ for token in doc])


def get_pos_tags(text: str, nlp=None) -> list:
    """
    Zwraca tagi POS dla tekstu.
    
    Args:
        text: Tekst do analizy
        nlp: Model NLP (opcjonalnie)
        
    Returns:
        List of (token, lemma, pos) tuples
    """
    if nlp is None:
        nlp = get_nlp()
    
    doc = nlp(text[:50000])
    return [(token.text, token.lemma_, token.pos_) for token in doc]


def extract_entities(text: str, nlp=None) -> list:
    """
    Ekstrahuje named entities z tekstu.
    
    Args:
        text: Tekst do analizy
        nlp: Model NLP (opcjonalnie)
        
    Returns:
        List of (text, label) tuples
    """
    if nlp is None:
        nlp = get_nlp()
    
    doc = nlp(text[:50000])
    return [(ent.text, ent.label_) for ent in doc.ents]


# ============================================================
# DIAGNOSTICS
# ============================================================

def run_diagnostics() -> Dict[str, Any]:
    """
    Uruchamia diagnostykę NLP.
    
    Returns:
        Dict z wynikami testów
    """
    results = {
        "spacy_installed": False,
        "morfeusz_installed": False,
        "available_models": [],
        "recommended_action": None,
        "test_results": {}
    }
    
    # Sprawdź spaCy
    try:
        import spacy
        results["spacy_installed"] = True
        results["spacy_version"] = spacy.__version__
    except ImportError:
        results["recommended_action"] = "pip install spacy"
        return results
    
    # Sprawdź Morfeusz
    results["morfeusz_installed"] = _check_morfeusz()
    
    # Sprawdź modele
    results["available_models"] = _check_available_models()
    
    # Rekomendacja
    if not results["available_models"]:
        results["recommended_action"] = "python -m spacy download pl_core_news_md"
    elif "pl_spacy_model_morfeusz" not in results["available_models"]:
        if results["morfeusz_installed"]:
            results["recommended_action"] = "pip install pl_spacy_model_morfeusz"
        else:
            results["recommended_action"] = "pip install morfeusz2 && pip install pl_spacy_model_morfeusz"
    
    # Test podstawowy
    if results["available_models"]:
        try:
            nlp = get_nlp()
            doc = nlp("Ala ma kota.")
            results["test_results"]["tokenization"] = [t.text for t in doc]
            results["test_results"]["lemmatization"] = [t.lemma_ for t in doc]
            results["test_results"]["pos_tags"] = [t.pos_ for t in doc]
            results["test_results"]["status"] = "OK"
        except Exception as e:
            results["test_results"]["status"] = f"ERROR: {e}"
    
    return results


# ============================================================
# VERSION INFO
# ============================================================

__version__ = "40.2"
__all__ = [
    "get_nlp",
    "get_nlp_light",
    "get_nlp_info",
    "reload_nlp",
    "lemmatize",
    "get_pos_tags",
    "extract_entities",
    "run_diagnostics",
    "NLP_MODES",
    "DEFAULT_MODE",
]


# ============================================================
# AUTO-INIT INFO
# ============================================================

if __name__ != "__main__":
    # Print info przy imporcie
    print(f"[NLP_CONFIG] v{__version__} loaded (mode: {DEFAULT_MODE})")
