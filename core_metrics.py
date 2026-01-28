"""
===============================================================================
CORE METRICS v40.2
===============================================================================
Single Source of Truth dla wszystkich metryk tekstowych.

KONSOLIDACJA z plików:
- ai_detection_metrics.py (calculate_burstiness)
- batch_review_system.py (calculate_burstiness, calculate_transition_score)
- firestore_tracker_routes.py (calculate_burstiness)
- text_analyzer.py (calculate_burstiness)
- unified_validator.py (calculate_burstiness)

UŻYCIE:
    from core_metrics import (
        calculate_burstiness,
        calculate_burstiness_simple,
        calculate_cv,
        calculate_transition_score,
        split_into_sentences
    )
===============================================================================
"""

import re
import math
import statistics
from typing import Dict, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass


# ============================================================
# KONFIGURACJA
# ============================================================

@dataclass
class BurstinessConfig:
    """Konfiguracja progów burstiness/CV."""
    # Progi CV (Coefficient of Variation)
    CV_CRITICAL_LOW: float = 0.26    # AI pattern
    CV_WARNING_LOW: float = 0.36     # Needs improvement
    CV_OK_MIN: float = 0.44          # Acceptable human-like
    CV_OK_MAX: float = 0.86          # Good variance
    CV_WARNING_HIGH: float = 0.96    # Too chaotic
    CV_CRITICAL_HIGH: float = 1.10   # Unreadable
    
    # Progi burstiness (CV * 5)
    BURSTINESS_CRITICAL_LOW: float = 1.3
    BURSTINESS_WARNING_LOW: float = 1.8
    BURSTINESS_OK_MIN: float = 2.2
    BURSTINESS_OK_MAX: float = 4.3
    BURSTINESS_WARNING_HIGH: float = 4.8
    BURSTINESS_CRITICAL_HIGH: float = 5.5
    
    # Minimalna liczba zdań do analizy
    MIN_SENTENCES: int = 5


class Severity(Enum):
    """Poziomy ważności."""
    OK = "OK"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Domyślna konfiguracja
DEFAULT_CONFIG = BurstinessConfig()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text: str) -> List[str]:
    """
    Inteligentne dzielenie tekstu na zdania.
    
    Obsługuje:
    - Skróty (np., dr., art., itp.)
    - Inicjały (A. Kowalski)
    - Liczby z kropkami (art. 13)
    - Wielokropki (...)
    """
    if not text or not text.strip():
        return []
    
    # Ochrona skrótów
    abbreviations = [
        'np', 'dr', 'prof', 'mgr', 'inż', 'art', 'ust', 'pkt', 'lit',
        'tj', 'tzn', 'itd', 'itp', 'etc', 'vs', 'ok', 'ww', 'jw',
        'r', 'w', 'z', 's', 'k', 'm', 'n',  # pojedyncze litery
        'sp', 'ul', 'al', 'pl', 'os',  # adresy
        'tel', 'fax', 'e-mail',
        'godz', 'min', 'sek',
        'tys', 'mln', 'mld',
        'zł', 'gr', 'USD', 'EUR',
    ]
    
    protected_text = text
    placeholders = {}
    
    # Ochrona skrótów
    for i, abbr in enumerate(abbreviations):
        pattern = rf'\b{abbr}\.'
        placeholder = f'__ABBR{i}__'
        protected_text = re.sub(pattern, placeholder, protected_text, flags=re.IGNORECASE)
        placeholders[placeholder] = f'{abbr}.'
    
    # Ochrona inicjałów (A. B. Kowalski)
    protected_text = re.sub(r'([A-ZĄĆĘŁŃÓŚŹŻ])\.\s*(?=[A-ZĄĆĘŁŃÓŚŹŻ])', r'\1__DOT__ ', protected_text)
    
    # Ochrona numerów (art. 13, § 2)
    protected_text = re.sub(r'(\d)\.\s*(\d)', r'\1__DOT__\2', protected_text)
    
    # Ochrona wielokropków
    protected_text = re.sub(r'\.{3,}', '__ELLIPSIS__', protected_text)
    
    # Podział na zdania
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])', protected_text)
    
    # Przywracanie placeholderów
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        
        # Przywróć skróty
        for placeholder, original in placeholders.items():
            s = s.replace(placeholder, original)
        
        s = s.replace('__DOT__', '.')
        s = s.replace('__ELLIPSIS__', '...')
        
        if s:
            result.append(s)
    
    return result


# ============================================================
# BURSTINESS / CV CALCULATIONS
# ============================================================

def calculate_cv(text: str) -> float:
    """
    Oblicza Coefficient of Variation (CV) dla długości zdań.
    
    CV = std_dev / mean
    
    Returns:
        float: CV value (0.0 - 1.5+)
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) < 3:
        return 0.5  # Neutralna wartość dla za mało danych
    
    lengths = [len(s.split()) for s in sentences]
    
    if not lengths:
        return 0.5
    
    mean_len = statistics.mean(lengths)
    
    if mean_len == 0:
        return 0.5
    
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
    cv = std_len / mean_len
    
    return round(cv, 3)


def calculate_burstiness_simple(text: str) -> float:
    """
    Prosta wersja - zwraca tylko wartość burstiness (float).
    
    Kompatybilność wsteczna z:
    - batch_review_system.py
    - firestore_tracker_routes.py
    - text_analyzer.py
    - unified_validator.py
    
    Returns:
        float: Burstiness value (CV * 5), range ~1.0 - 5.5
    """
    cv = calculate_cv(text)
    burstiness = cv * 5
    
    # Clamp do sensownego zakresu
    burstiness = max(1.0, min(burstiness, 6.0))
    
    return round(burstiness, 2)


def calculate_burstiness(text: str, config: BurstinessConfig = None) -> Dict[str, Any]:
    """
    Pełna analiza burstiness z diagnostyką.
    
    Formuła: burstiness = CV * 5
    
    Progi (zgodne z badaniami NKJP):
    - CV < 0.26 (burstiness < 1.3) = CRITICAL (AI pattern)
    - CV 0.26-0.36 (burstiness 1.3-1.8) = WARNING
    - CV 0.36-0.44 (burstiness 1.8-2.2) = INFO (suboptimal)
    - CV 0.44-0.86 (burstiness 2.2-4.3) = OK (human-like)
    - CV > 0.86 (burstiness > 4.3) = WARNING (too chaotic)
    
    Args:
        text: Tekst do analizy
        config: Opcjonalna konfiguracja progów
        
    Returns:
        Dict z: value, cv, status, severity, message, sentence_count, 
                sentence_lengths, recommendations
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    sentences = split_into_sentences(text)
    sentence_count = len(sentences)
    
    # Za mało zdań
    if sentence_count < config.MIN_SENTENCES:
        return {
            "value": 0.0,
            "cv": 0.0,
            "status": "INSUFFICIENT_DATA",
            "severity": Severity.WARNING.value,
            "message": f"Za mało zdań do analizy (masz {sentence_count}, min {config.MIN_SENTENCES})",
            "sentence_count": sentence_count,
            "sentence_lengths": [],
            "recommendations": ["Dodaj więcej treści przed analizą burstiness"]
        }
    
    # Obliczenia
    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
    
    cv = std_len / mean_len if mean_len > 0 else 0
    burstiness = round(cv * 5, 2)
    cv_rounded = round(cv, 3)
    
    # Określenie statusu i severity
    status, severity, message, recommendations = _evaluate_burstiness(
        burstiness, cv_rounded, config, lengths
    )
    
    return {
        "value": burstiness,
        "cv": cv_rounded,
        "status": status,
        "severity": severity,
        "message": message,
        "sentence_count": sentence_count,
        "sentence_lengths": lengths,
        "mean_length": round(mean_len, 1),
        "std_length": round(std_len, 1),
        "recommendations": recommendations,
        "distribution": _analyze_length_distribution(lengths)
    }


def _evaluate_burstiness(
    burstiness: float, 
    cv: float, 
    config: BurstinessConfig,
    lengths: List[int]
) -> Tuple[str, str, str, List[str]]:
    """Ocenia burstiness i zwraca status, severity, message, recommendations."""
    
    recommendations = []
    
    # CRITICAL LOW - AI pattern
    if burstiness < config.BURSTINESS_CRITICAL_LOW:
        status = "CRITICAL_LOW"
        severity = Severity.CRITICAL.value
        message = f"⚠️ SYGNAŁ AI: burstiness {burstiness} (CV {cv:.2f} < 0.26). Wszystkie zdania mają podobną długość."
        recommendations = [
            "Dodaj krótkie zdania (3-8 słów): 'Sąd orzeka.', 'To ważne.', 'Co dalej?'",
            "Dodaj długie zdania (25-35 słów) z rozbudowanymi wyjaśnieniami",
            "Unikaj zdań o podobnej długości (15-20 słów) - to wzorzec AI"
        ]
    
    # WARNING LOW
    elif burstiness < config.BURSTINESS_WARNING_LOW:
        status = "WARNING_LOW"
        severity = Severity.WARNING.value
        message = f"Niska zmienność zdań: burstiness {burstiness} (CV {cv:.2f}). Dodaj więcej różnorodności."
        recommendations = [
            "Dodaj 2-3 bardzo krótkie zdania (3-6 słów)",
            "Dodaj 1-2 długie zdania opisowe (25+ słów)"
        ]
    
    # SUBOPTIMAL (INFO - nie blokuje)
    elif burstiness < config.BURSTINESS_OK_MIN:
        status = "SUBOPTIMAL"
        severity = Severity.INFO.value
        message = f"Akceptowalna zmienność: burstiness {burstiness} (CV {cv:.2f}). Można poprawić."
        recommendations = [
            "Opcjonalnie: dodaj więcej krótkich zdań dla naturalności"
        ]
    
    # OK - optimal range
    elif burstiness <= config.BURSTINESS_OK_MAX:
        status = "OK"
        severity = Severity.OK.value
        message = f"✅ Dobra zmienność zdań: burstiness {burstiness} (CV {cv:.2f})"
        recommendations = []
    
    # WARNING HIGH
    elif burstiness < config.BURSTINESS_CRITICAL_HIGH:
        status = "WARNING_HIGH"
        severity = Severity.WARNING.value
        message = f"Zbyt chaotyczna struktura: burstiness {burstiness} (CV {cv:.2f})"
        recommendations = [
            "Wyrównaj niektóre zdania - tekst może być trudny do czytania",
            "Sprawdź czy nie ma błędnie podzielonych zdań"
        ]
    
    # CRITICAL HIGH
    else:
        status = "CRITICAL_HIGH"
        severity = Severity.CRITICAL.value
        message = f"Ekstremalnie chaotyczna struktura: burstiness {burstiness} (CV {cv:.2f})"
        recommendations = [
            "Tekst prawdopodobnie ma błędy segmentacji",
            "Sprawdź interpunkcję i podział na zdania"
        ]
    
    return status, severity, message, recommendations


def _analyze_length_distribution(lengths: List[int]) -> Dict[str, Any]:
    """Analizuje rozkład długości zdań."""
    if not lengths:
        return {}
    
    short = sum(1 for l in lengths if l <= 8)      # 3-8 słów
    medium = sum(1 for l in lengths if 9 <= l <= 18)  # 9-18 słów
    long = sum(1 for l in lengths if l >= 19)      # 19+ słów
    
    total = len(lengths)
    
    return {
        "short_pct": round(short / total * 100, 1),   # Target: 20-25%
        "medium_pct": round(medium / total * 100, 1), # Target: 50-60%
        "long_pct": round(long / total * 100, 1),     # Target: 15-25%
        "short_count": short,
        "medium_count": medium,
        "long_count": long,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "is_balanced": (15 <= short/total*100 <= 30 and 
                       45 <= medium/total*100 <= 65 and 
                       10 <= long/total*100 <= 30)
    }


# ============================================================
# TRANSITION WORDS
# ============================================================

TRANSITION_WORDS_PL = {
    # Dodawanie
    "również", "także", "ponadto", "dodatkowo", "co więcej", "oprócz tego",
    "poza tym", "przy tym", "nadto", "zarazem",
    
    # Kontrast
    "jednak", "jednakże", "natomiast", "ale", "lecz", "z drugiej strony",
    "mimo to", "niemniej", "pomimo", "choć", "chociaż", "aczkolwiek",
    "wprawdzie", "tymczasem", "przeciwnie", "w przeciwieństwie",
    
    # Przyczyna/skutek
    "dlatego", "w związku z tym", "w rezultacie", "ponieważ", "gdyż",
    "zatem", "więc", "stąd", "w konsekwencji", "wobec tego", "toteż",
    "skutkiem tego", "w efekcie", "przez to",
    
    # Przykłady
    "na przykład", "przykładowo", "między innymi", "np.", "m.in.",
    "w szczególności", "zwłaszcza", "przede wszystkim",
    
    # Sekwencja
    "po pierwsze", "po drugie", "po trzecie", "następnie", "potem",
    "na koniec", "wreszcie", "najpierw", "w końcu", "finalnie",
    
    # Podsumowanie
    "podsumowując", "reasumując", "w skrócie", "ogólnie rzecz biorąc",
    "innymi słowy", "to znaczy", "mianowicie", "czyli", "w sumie",
    
    # Podkreślenie
    "co ważne", "warto zauważyć", "należy podkreślić", "istotne jest",
    "kluczowe jest", "szczególnie", "znacząco",
}


def calculate_transition_score(text: str) -> Dict[str, Any]:
    """
    Oblicza wynik transition words (słowa łączące).
    
    Target: 25-50% zdań zaczyna się lub zawiera transition words
    
    Returns:
        Dict z: score, ratio, status, sentences_with_transitions, total_sentences
    """
    if not text:
        return {
            "score": 0,
            "ratio": 0.0,
            "status": "NO_DATA",
            "sentences_with_transitions": 0,
            "total_sentences": 0
        }
    
    text_lower = text.lower()
    sentences = split_into_sentences(text)
    
    if not sentences:
        return {
            "score": 0,
            "ratio": 0.0,
            "status": "NO_SENTENCES",
            "sentences_with_transitions": 0,
            "total_sentences": 0
        }
    
    sentences_with_transitions = 0
    found_transitions = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for tw in TRANSITION_WORDS_PL:
            # Sprawdź czy zdanie zawiera transition word
            if tw in sentence_lower:
                sentences_with_transitions += 1
                found_transitions.append(tw)
                break
    
    ratio = sentences_with_transitions / len(sentences)
    
    # Scoring
    if ratio < 0.15:
        status = "LOW"
        score = int(ratio * 200)  # 0-30
    elif ratio < 0.25:
        status = "SUBOPTIMAL"
        score = 30 + int((ratio - 0.15) * 200)  # 30-50
    elif ratio <= 0.50:
        status = "OK"
        score = 50 + int((ratio - 0.25) * 200)  # 50-100
    else:
        status = "HIGH"
        score = max(70, 100 - int((ratio - 0.50) * 100))  # może być za dużo
    
    return {
        "score": min(100, score),
        "ratio": round(ratio, 3),
        "percentage": round(ratio * 100, 1),
        "status": status,
        "sentences_with_transitions": sentences_with_transitions,
        "total_sentences": len(sentences),
        "found_transitions": list(set(found_transitions))[:10],
        "recommendation": _get_transition_recommendation(ratio)
    }


def _get_transition_recommendation(ratio: float) -> str:
    """Zwraca rekomendację dla transition ratio."""
    if ratio < 0.15:
        return "Dodaj więcej słów łączących (jednak, dlatego, ponadto) - tekst jest chaotyczny"
    elif ratio < 0.25:
        return "Rozważ dodanie transition words na początku niektórych zdań"
    elif ratio <= 0.50:
        return ""  # OK
    else:
        return "Zbyt wiele transition words - tekst może brzmieć nienaturalnie"


# ============================================================
# KEYWORD DENSITY
# ============================================================

def calculate_keyword_density(text: str, keyword: str) -> Dict[str, Any]:
    """
    Oblicza gęstość słowa kluczowego w tekście.
    
    Args:
        text: Tekst do analizy
        keyword: Słowo kluczowe
        
    Returns:
        Dict z: count, density, status
    """
    if not text or not keyword:
        return {"count": 0, "density": 0.0, "status": "NO_DATA"}
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Liczenie wystąpień
    count = len(re.findall(rf'\b{re.escape(keyword_lower)}\b', text_lower))
    
    # Liczba słów
    words = text_lower.split()
    word_count = len(words)
    
    if word_count == 0:
        return {"count": 0, "density": 0.0, "status": "NO_TEXT"}
    
    density = (count / word_count) * 100
    
    # Status
    if density == 0:
        status = "MISSING"
    elif density < 0.5:
        status = "LOW"
    elif density <= 2.5:
        status = "OK"
    elif density <= 3.5:
        status = "HIGH"
    else:
        status = "STUFFING"
    
    return {
        "count": count,
        "density": round(density, 2),
        "word_count": word_count,
        "status": status
    }


# ============================================================
# HELPER EXPORTS
# ============================================================

def get_burstiness_status(burstiness: float, config: BurstinessConfig = None) -> str:
    """Szybkie sprawdzenie statusu burstiness bez pełnej analizy."""
    if config is None:
        config = DEFAULT_CONFIG
    
    if burstiness < config.BURSTINESS_CRITICAL_LOW:
        return "CRITICAL"
    elif burstiness < config.BURSTINESS_WARNING_LOW:
        return "WARNING"
    elif burstiness < config.BURSTINESS_OK_MIN:
        return "SUBOPTIMAL"
    elif burstiness <= config.BURSTINESS_OK_MAX:
        return "OK"
    elif burstiness < config.BURSTINESS_CRITICAL_HIGH:
        return "WARNING"
    else:
        return "CRITICAL"


# ============================================================
# VERSION INFO
# ============================================================

__version__ = "40.2"
__all__ = [
    # Main functions
    "calculate_burstiness",
    "calculate_burstiness_simple",
    "calculate_cv",
    "calculate_transition_score",
    "calculate_keyword_density",
    
    # Helpers
    "split_into_sentences",
    "get_burstiness_status",
    
    # Config & Constants
    "BurstinessConfig",
    "DEFAULT_CONFIG",
    "TRANSITION_WORDS_PL",
    "Severity",
]
