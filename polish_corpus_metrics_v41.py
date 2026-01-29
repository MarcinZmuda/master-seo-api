"""
===============================================================================
📊 POLISH CORPUS METRICS v41.2 - Metryki oparte na NKJP
===============================================================================
WARSTWA INFORMACYJNA - NIE BLOKUJE WALIDACJI!

Moduł dostarcza dodatkowe insights o naturalności tekstu polskiego
na podstawie statystyk z Narodowego Korpusu Języka Polskiego (NKJP).

ZASADY BEZPIECZEŃSTWA:
1. Wszystkie metryki zwracają severity="info" lub "suggestion"
2. NIGDY nie zwracamy severity="critical" lub "warning" blokującego
3. Wyniki są opcjonalne - brak ich nie wpływa na główną walidację
4. Moduł może być wyłączony przez flagę ENABLE_CORPUS_INSIGHTS=false

Źródła:
- NKJP (1,8 mld segmentów)
- IPI PAN (25 mln wyrazów)
- Badania Moździerza (2020)
- Practical Cryptography (90 mln znaków)

Użycie:
    from polish_corpus_metrics_v41 import analyze_corpus_metrics
    result = analyze_corpus_metrics(text)
    
    # Integracja z MOE:
    from polish_corpus_metrics_v41 import get_corpus_insights_for_moe
    insights = get_corpus_insights_for_moe(batch_text)

===============================================================================
"""

import re
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter
from enum import Enum


# ============================================================================
# KONFIGURACJA - FLAGI BEZPIECZEŃSTWA
# ============================================================================

# Główna flaga włączająca/wyłączająca moduł
ENABLE_CORPUS_INSIGHTS = os.environ.get("ENABLE_CORPUS_INSIGHTS", "true").lower() == "true"

# Flagi poszczególnych metryk
INCLUDE_DIACRITICS = os.environ.get("CORPUS_INCLUDE_DIACRITICS", "true").lower() == "true"
INCLUDE_WORD_LENGTH = os.environ.get("CORPUS_INCLUDE_WORD_LENGTH", "true").lower() == "true"
INCLUDE_FOG = os.environ.get("CORPUS_INCLUDE_FOG", "true").lower() == "true"
INCLUDE_PUNCTUATION = os.environ.get("CORPUS_INCLUDE_PUNCTUATION", "true").lower() == "true"
INCLUDE_VOWELS = os.environ.get("CORPUS_INCLUDE_VOWELS", "true").lower() == "true"

# Minima dla wiarygodnej analizy
MIN_LETTERS_FOR_ANALYSIS = 50
MIN_WORDS_FOR_ANALYSIS = 20
MIN_SENTENCES_FOR_FOG = 3
MIN_CHARS_FOR_PUNCTUATION = 100


# ============================================================================
# STAŁE KORPUSOWE (z dokumentu "Statystyczne cechy naturalnego tekstu")
# ============================================================================

# Diakrytyki polskie
POLISH_DIACRITICS: Set[str] = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')

# Samogłoski polskie (z diakrytykami)
POLISH_VOWELS: Set[str] = set('aąeęioóuyAĄEĘIOÓUY')

# Spójniki wymagające przecinka przed sobą
SUBORDINATE_CONJUNCTIONS = [
    'że', 'który', 'która', 'które', 'którzy', 'których', 'którym',
    'ponieważ', 'gdyż', 'bowiem', 'albowiem',
    'aby', 'żeby', 'ażeby',
    'jeśli', 'jeżeli', 'gdyby',
    'chociaż', 'choć', 'mimo że', 'pomimo że',
    'dopóki', 'zanim', 'odkąd', 'skoro'
]

# Wartości referencyjne z NKJP
CORPUS_REFERENCE = {
    "diacritic_ratio": {
        "target": 0.069,      # 6.9%
        "tolerance": 0.01,    # ±1%
        "min_natural": 0.05,  # <5% = nienaturalne
        "max_natural": 0.09,  # >9% = nienaturalne
    },
    "word_length": {
        "target": 6.0,        # 6 znaków
        "tolerance": 0.5,
        "by_style": {
            "literatura": 5.32,
            "publicystyka": 6.00,
            "urzędowy": 6.14,
            "naukowy": 6.43,
        }
    },
    "vowel_ratio": {
        "target": 0.365,      # 36.5%
        "min": 0.35,
        "max": 0.38,
    },
    "punctuation": {
        "comma_min": 0.0147,  # Przecinek > 1.47% (częstszy niż litera "b")
    },
    "fog_pl": {
        "optimal_min": 8,
        "optimal_max": 9,
        "ranges": {
            (1, 6): "szkoła podstawowa",
            (7, 9): "gimnazjum/liceum (optymalne)",
            (10, 12): "studia licencjackie",
            (13, 15): "studia magisterskie",
            (16, 21): "specjalistyczne",
        }
    },
    "sentence_length": {
        "target": 10,         # ~10 słów w naturalnym zdaniu
        "by_style": {
            "kolokwialny": 8,
            "publicystyka": 10,
            "prawniczy": 17,
            "naukowy": 20,
        }
    }
}


# ============================================================================
# TYPY I KLASY
# ============================================================================

class InsightSeverity(Enum):
    """
    BEZPIECZNE poziomy severity - nigdy nie blokują walidacji!
    
    WAŻNE: Celowo brak CRITICAL i WARNING (blokujących).
    """
    INFO = "info"               # Neutralna informacja
    SUGGESTION = "suggestion"   # Sugestia poprawy (nie wymóg)
    OBSERVATION = "observation" # Obserwacja bez oceny


@dataclass
class CorpusInsight:
    """
    Pojedynczy insight z analizy korpusowej.
    
    GWARANCJA: severity NIGDY nie może być "critical" lub "warning"!
    """
    metric: str               # np. "diacritic_ratio"
    value: float              # Zmierzona wartość
    target: float             # Wartość docelowa z NKJP
    deviation: float          # Odchylenie od celu (może być ujemne)
    severity: InsightSeverity # Tylko INFO/SUGGESTION/OBSERVATION
    message: str              # Opis dla użytkownika
    suggestion: str = ""      # Opcjonalna sugestia poprawy
    details: Dict = field(default_factory=dict)  # Dodatkowe dane
    
    def __post_init__(self):
        """Walidacja że severity jest bezpieczne."""
        if self.severity not in [InsightSeverity.INFO, 
                                  InsightSeverity.SUGGESTION, 
                                  InsightSeverity.OBSERVATION]:
            # Fallback do INFO zamiast rzucania wyjątku
            self.severity = InsightSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": round(self.value, 4) if isinstance(self.value, float) else self.value,
            "target": self.target,
            "deviation": round(self.deviation, 4) if isinstance(self.deviation, float) else self.deviation,
            "deviation_pct": f"{abs(self.deviation/self.target)*100:.1f}%" if self.target else "N/A",
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
            # WAŻNE: Jawne oznaczenie że NIE blokuje
            "blocks_validation": False,
        }


@dataclass
class CorpusAnalysisResult:
    """
    Wynik pełnej analizy korpusowej.
    """
    insights: List[CorpusInsight]
    overall_naturalness: float  # 0-100 (orientacyjne, NIE do blokowania!)
    style_detected: str         # "publicystyka", "naukowy", etc.
    summary: str
    word_count: int = 0
    
    # FLAGA BEZPIECZEŃSTWA - ZAWSZE False!
    blocks_validation: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Wymuszenie blocks_validation=False."""
        object.__setattr__(self, 'blocks_validation', False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insights": [i.to_dict() for i in self.insights],
            "overall_naturalness": round(self.overall_naturalness, 1),
            "style_detected": self.style_detected,
            "summary": self.summary,
            "word_count": self.word_count,
            "insights_count": len(self.insights),
            "suggestions_count": sum(1 for i in self.insights 
                                     if i.severity == InsightSeverity.SUGGESTION),
            # WYMUSZONY False - nigdy nie zmieniać!
            "blocks_validation": False,
            "is_informational_only": True,
        }


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def _extract_words(text: str) -> List[str]:
    """Wyodrębnia słowa z tekstu (tylko litery polskie)."""
    return re.findall(r'\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b', text)


def _extract_sentences(text: str) -> List[str]:
    """Dzieli tekst na zdania."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def _count_syllables_pl(word: str) -> int:
    """
    Przybliżone liczenie sylab w polskim słowie.
    
    Polski ma średnio 3 sylaby na słowo (vs 2 w angielskim).
    """
    word = word.lower()
    vowels = 'aeiouyąęó'
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    return max(1, count)


def _detect_style_from_word_length(avg_length: float) -> str:
    """Wykrywa styl tekstu na podstawie średniej długości słów."""
    styles = CORPUS_REFERENCE["word_length"]["by_style"]
    
    closest_style = "publicystyka"
    min_diff = float('inf')
    
    for style, style_avg in styles.items():
        diff = abs(avg_length - style_avg)
        if diff < min_diff:
            min_diff = diff
            closest_style = style
    
    return closest_style


# ============================================================================
# METRYKI KORPUSOWE
# ============================================================================

def calculate_diacritic_ratio(text: str) -> CorpusInsight:
    """
    Oblicza udział znaków diakrytycznych w tekście.
    
    Wartość referencyjna: 6.9% ± 1% (z NKJP)
    
    Tekst z udziałem diakrytyków <5% lub >9% jest statystycznie nienaturalny
    i może być rozpoznany jako sztuczny przez rodzimych użytkowników.
    
    Returns:
        CorpusInsight z severity INFO lub SUGGESTION (nigdy nie blokuje!)
    """
    if not text:
        return CorpusInsight(
            metric="diacritic_ratio",
            value=0,
            target=CORPUS_REFERENCE["diacritic_ratio"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Brak tekstu do analizy"
        )
    
    # Policz tylko litery (bez spacji, cyfr, interpunkcji)
    letters_only = [c for c in text if c.isalpha()]
    total_letters = len(letters_only)
    
    if total_letters < MIN_LETTERS_FOR_ANALYSIS:
        return CorpusInsight(
            metric="diacritic_ratio",
            value=0,
            target=CORPUS_REFERENCE["diacritic_ratio"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Za mało liter do wiarygodnej analizy (<{MIN_LETTERS_FOR_ANALYSIS})"
        )
    
    diacritic_count = sum(1 for c in letters_only if c in POLISH_DIACRITICS)
    ratio = diacritic_count / total_letters
    
    ref = CORPUS_REFERENCE["diacritic_ratio"]
    target = ref["target"]
    deviation = ratio - target
    
    # Szczegóły do debug
    details = {
        "total_letters": total_letters,
        "diacritic_count": diacritic_count,
        "ratio_pct": f"{ratio*100:.2f}%",
        "reference_range": "5-9%",
    }
    
    # Określ severity (tylko INFO lub SUGGESTION, nigdy nie blokuje!)
    if ref["min_natural"] <= ratio <= ref["max_natural"]:
        severity = InsightSeverity.INFO
        message = f"Udział diakrytyków ({ratio*100:.1f}%) w normie NKJP (5-9%)"
        suggestion = ""
    elif ratio < ref["min_natural"]:
        severity = InsightSeverity.SUGGESTION
        message = f"Niski udział diakrytyków ({ratio*100:.1f}%) - poniżej normy 5%"
        suggestion = ("Rozważ użycie bardziej polskich form: "
                     "'że' zamiast 'ze', 'są' zamiast 'sa', "
                     "'będzie' zamiast 'bedzie', 'możliwość' zamiast 'mozliwosc'")
    else:
        severity = InsightSeverity.SUGGESTION
        message = f"Wysoki udział diakrytyków ({ratio*100:.1f}%) - powyżej normy 9%"
        suggestion = ("Tekst może być postrzegany jako nienaturalnie 'polski'. "
                     "Sprawdź czy nie ma nadmiaru słów z wieloma diakrytykami.")
    
    return CorpusInsight(
        metric="diacritic_ratio",
        value=ratio,
        target=target,
        deviation=deviation,
        severity=severity,
        message=message,
        suggestion=suggestion,
        details=details
    )


def calculate_word_length_stats(text: str) -> CorpusInsight:
    """
    Oblicza średnią długość słowa.
    
    Wartość referencyjna: 6 znaków ± 0.5 (z NKJP)
    Polski ma dłuższe słowa niż angielski (4.6 znaków).
    
    Różne style mają różne średnie:
    - Literatura piękna: 5.32 zn.
    - Publicystyka: 6.00 zn.
    - Teksty urzędowe: 6.14 zn.
    - Teksty naukowe: 6.43 zn.
    
    Returns:
        CorpusInsight z wykrytym stylem tekstu
    """
    if not text:
        return CorpusInsight(
            metric="word_length_avg",
            value=0,
            target=CORPUS_REFERENCE["word_length"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Brak tekstu"
        )
    
    words = _extract_words(text)
    
    if len(words) < MIN_WORDS_FOR_ANALYSIS:
        return CorpusInsight(
            metric="word_length_avg",
            value=0,
            target=CORPUS_REFERENCE["word_length"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Za mało słów do analizy (<{MIN_WORDS_FOR_ANALYSIS})"
        )
    
    word_lengths = [len(w) for w in words]
    avg_length = sum(word_lengths) / len(word_lengths)
    
    ref = CORPUS_REFERENCE["word_length"]
    target = ref["target"]
    tolerance = ref["tolerance"]
    deviation = avg_length - target
    
    # Wykryj styl na podstawie długości słów
    style_detected = _detect_style_from_word_length(avg_length)
    
    # Statystyki dodatkowe
    details = {
        "word_count": len(words),
        "avg_length": round(avg_length, 2),
        "min_length": min(word_lengths),
        "max_length": max(word_lengths),
        "style_detected": style_detected,
        "style_reference": ref["by_style"].get(style_detected, 6.0),
    }
    
    if abs(deviation) <= tolerance:
        severity = InsightSeverity.INFO
        message = f"Średnia długość słowa ({avg_length:.2f} zn.) zgodna z normą NKJP"
        suggestion = ""
    else:
        severity = InsightSeverity.SUGGESTION
        if deviation > 0:
            message = f"Długie słowa ({avg_length:.2f} zn.) - styl: {style_detected}"
            suggestion = ("Tekst może być trudniejszy w odbiorze. "
                         "Rozważ użycie prostszych synonimów dla długich słów.")
        else:
            message = f"Krótkie słowa ({avg_length:.2f} zn.) - styl: {style_detected}"
            suggestion = ""
    
    return CorpusInsight(
        metric="word_length_avg",
        value=avg_length,
        target=target,
        deviation=deviation,
        severity=severity,
        message=message,
        suggestion=suggestion,
        details=details
    )


def calculate_fog_pl_index(text: str) -> CorpusInsight:
    """
    Oblicza zmodyfikowany indeks FOG dla polskiego.
    
    FOG-PL = 0.4 × (słowa/zdania + 100 × słowa_trudne/słowa)
    
    WAŻNE: Słowa trudne w polskim to ≥4 sylaby (nie 3 jak w angielskim!)
    
    Interpretacja:
    - 1-6: szkoła podstawowa
    - 7-9: gimnazjum/liceum (OPTYMALNE dla ogółu)
    - 10-12: studia licencjackie
    - 13-15: studia magisterskie
    - 16-21: specjalistyczne
    
    Returns:
        CorpusInsight z indeksem FOG-PL i poziomem trudności
    """
    if not text:
        return CorpusInsight(
            metric="fog_pl",
            value=0,
            target=8.5,
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Brak tekstu"
        )
    
    sentences = _extract_sentences(text)
    
    if len(sentences) < MIN_SENTENCES_FOR_FOG:
        return CorpusInsight(
            metric="fog_pl",
            value=0,
            target=8.5,
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Za mało zdań do analizy (<{MIN_SENTENCES_FOR_FOG})"
        )
    
    words = _extract_words(text)
    total_words = len(words)
    
    if total_words < 30:
        return CorpusInsight(
            metric="fog_pl",
            value=0,
            target=8.5,
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Za mało słów (<30)"
        )
    
    # Średnia długość zdania
    avg_sentence_length = total_words / len(sentences)
    
    # Policz słowa trudne (≥4 sylaby dla polskiego!)
    difficult_words = [w for w in words if _count_syllables_pl(w) >= 4]
    difficult_count = len(difficult_words)
    difficult_ratio = difficult_count / total_words
    
    # FOG-PL
    fog = 0.4 * (avg_sentence_length + 100 * difficult_ratio)
    
    ref = CORPUS_REFERENCE["fog_pl"]
    optimal_mid = (ref["optimal_min"] + ref["optimal_max"]) / 2
    deviation = fog - optimal_mid
    
    # Znajdź opis poziomu
    level_desc = "nieznany"
    for (low, high), desc in ref["ranges"].items():
        if low <= fog <= high:
            level_desc = desc
            break
    if fog > 21:
        level_desc = "bardzo specjalistyczny"
    
    details = {
        "fog_value": round(fog, 2),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "difficult_words_count": difficult_count,
        "difficult_words_pct": f"{difficult_ratio*100:.1f}%",
        "level_description": level_desc,
        "difficult_examples": difficult_words[:5],  # Przykłady trudnych słów
    }
    
    if ref["optimal_min"] <= fog <= ref["optimal_max"]:
        severity = InsightSeverity.INFO
        message = f"FOG-PL = {fog:.1f} (optymalne dla ogółu odbiorców)"
        suggestion = ""
    elif fog < ref["optimal_min"]:
        severity = InsightSeverity.INFO
        message = f"FOG-PL = {fog:.1f} - tekst łatwy ({level_desc})"
        suggestion = ""
    else:
        severity = InsightSeverity.SUGGESTION
        message = f"FOG-PL = {fog:.1f} - tekst trudny ({level_desc})"
        suggestion = ("Rozważ skrócenie zdań lub użycie prostszych słów "
                     "dla lepszej przystępności. Optymalne FOG-PL to 8-9.")
    
    return CorpusInsight(
        metric="fog_pl",
        value=fog,
        target=optimal_mid,
        deviation=deviation,
        severity=severity,
        message=message,
        suggestion=suggestion,
        details=details
    )


def calculate_punctuation_density(text: str) -> CorpusInsight:
    """
    Oblicza gęstość interpunkcji (szczególnie przecinków).
    
    W polskim przecinek występuje częściej niż litera "b" (>1.47%).
    
    Obligatoryjny przecinek przed:
    - "że", "który/a/e" - zdania względne
    - "ponieważ", "gdyż" - przyczynowe
    - "aby", "żeby" - celowe
    - "jednak", "lecz", "ale" - przeciwstawne
    
    Returns:
        CorpusInsight z analizą interpunkcji i brakującymi przecinkami
    """
    if not text:
        return CorpusInsight(
            metric="punctuation_density",
            value=0,
            target=CORPUS_REFERENCE["punctuation"]["comma_min"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Brak tekstu"
        )
    
    total_chars = len(text)
    if total_chars < MIN_CHARS_FOR_PUNCTUATION:
        return CorpusInsight(
            metric="punctuation_density",
            value=0,
            target=CORPUS_REFERENCE["punctuation"]["comma_min"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Za mało znaków (<{MIN_CHARS_FOR_PUNCTUATION})"
        )
    
    # Policz przecinki
    comma_count = text.count(',')
    comma_ratio = comma_count / total_chars
    
    target = CORPUS_REFERENCE["punctuation"]["comma_min"]
    deviation = comma_ratio - target
    
    # Sprawdź brakujące przecinki przed spójnikami podrzędnymi
    missing_commas = []
    text_lower = text.lower()
    
    for conj in SUBORDINATE_CONJUNCTIONS:
        # Szukaj spójnika bez przecinka przed nim (słowo + spacja + spójnik)
        # Ale nie na początku zdania
        pattern = rf'[a-ząćęłńóśźż]\s+{conj}\b'
        matches = re.findall(pattern, text_lower)
        
        for match in matches[:2]:  # Max 2 przykłady per spójnik
            # Sprawdź czy przed spójnikiem jest przecinek
            idx = text_lower.find(match)
            if idx > 0:
                before = text[max(0, idx-5):idx+len(match)]
                if ',' not in before:
                    missing_commas.append(f"przed '{conj}'")
    
    # Usuń duplikaty
    missing_commas = list(dict.fromkeys(missing_commas))[:5]
    
    details = {
        "comma_count": comma_count,
        "comma_ratio_pct": f"{comma_ratio*100:.2f}%",
        "target_pct": f"{target*100:.2f}%",
        "missing_commas": missing_commas,
    }
    
    if comma_ratio >= target and not missing_commas:
        severity = InsightSeverity.INFO
        message = f"Gęstość przecinków ({comma_ratio*100:.2f}%) zgodna z normą polską"
        suggestion = ""
    elif missing_commas:
        severity = InsightSeverity.SUGGESTION
        message = f"Wykryto potencjalnie brakujące przecinki: {', '.join(missing_commas[:3])}"
        suggestion = ("Polski wymaga przecinka przed spójnikami podrzędnymi: "
                     "że, który, ponieważ, aby, żeby. Brak przecinka przed 'że' "
                     "jest natychmiast rozpoznawalny jako błąd.")
    else:
        severity = InsightSeverity.INFO
        message = f"Gęstość przecinków ({comma_ratio*100:.2f}%) - sprawdź interpunkcję"
        suggestion = ""
    
    return CorpusInsight(
        metric="punctuation_density",
        value=comma_ratio,
        target=target,
        deviation=deviation,
        severity=severity,
        message=message,
        suggestion=suggestion,
        details=details
    )


def analyze_vowel_ratio(text: str) -> CorpusInsight:
    """
    Oblicza udział samogłosek w tekście.
    
    Wartość referencyjna: 35-38% (z NKJP)
    
    Dominują samogłoski A (8.91%), I (8.21%), O (7.75%), E (7.66%),
    które łącznie stanowią około 35-38% tekstu.
    
    Returns:
        CorpusInsight z udziałem samogłosek
    """
    if not text:
        return CorpusInsight(
            metric="vowel_ratio",
            value=0,
            target=CORPUS_REFERENCE["vowel_ratio"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message="Brak tekstu"
        )
    
    letters = [c for c in text.lower() if c.isalpha()]
    total = len(letters)
    
    if total < MIN_LETTERS_FOR_ANALYSIS:
        return CorpusInsight(
            metric="vowel_ratio",
            value=0,
            target=CORPUS_REFERENCE["vowel_ratio"]["target"],
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Za mało liter (<{MIN_LETTERS_FOR_ANALYSIS})"
        )
    
    vowel_count = sum(1 for c in letters if c in POLISH_VOWELS)
    ratio = vowel_count / total
    
    ref = CORPUS_REFERENCE["vowel_ratio"]
    target = ref["target"]
    deviation = ratio - target
    
    # Rozkład samogłosek
    vowel_distribution = Counter(c for c in letters if c in POLISH_VOWELS)
    top_vowels = vowel_distribution.most_common(5)
    
    details = {
        "total_letters": total,
        "vowel_count": vowel_count,
        "ratio_pct": f"{ratio*100:.1f}%",
        "reference_range": "35-38%",
        "top_vowels": {v: f"{c/total*100:.1f}%" for v, c in top_vowels},
    }
    
    if ref["min"] <= ratio <= ref["max"]:
        severity = InsightSeverity.INFO
        message = f"Udział samogłosek ({ratio*100:.1f}%) w normie NKJP (35-38%)"
    else:
        severity = InsightSeverity.INFO  # Tylko INFO, nie blokuje
        if ratio < ref["min"]:
            message = f"Niski udział samogłosek ({ratio*100:.1f}%) - poniżej normy 35%"
        else:
            message = f"Wysoki udział samogłosek ({ratio*100:.1f}%) - powyżej normy 38%"
    
    return CorpusInsight(
        metric="vowel_ratio",
        value=ratio,
        target=target,
        deviation=deviation,
        severity=severity,
        message=message,
        details=details
    )


# ============================================================================
# GŁÓWNA FUNKCJA ANALIZY
# ============================================================================

def analyze_corpus_metrics(
    text: str,
    include_fog: bool = True,
    include_punctuation: bool = True,
    include_diacritics: bool = True,
    include_word_length: bool = True,
    include_vowels: bool = True,
) -> CorpusAnalysisResult:
    """
    Przeprowadza pełną analizę korpusową tekstu.
    
    GWARANCJA BEZPIECZEŃSTWA:
    - Nigdy nie zwraca blocks_validation=True
    - Wszystkie severity to INFO lub SUGGESTION
    - Może być bezpiecznie wywołana bez wpływu na walidację MOE
    - W razie błędu zwraca pusty wynik, nie rzuca wyjątku
    
    Args:
        text: Tekst do analizy
        include_fog: Czy obliczać FOG-PL (default: True)
        include_punctuation: Czy analizować interpunkcję (default: True)
        include_diacritics: Czy analizować diakrytyki (default: True)
        include_word_length: Czy analizować długość słów (default: True)
        include_vowels: Czy analizować samogłoski (default: True)
    
    Returns:
        CorpusAnalysisResult z insights (NIGDY nie blokuje walidacji!)
    """
    # Sprawdź czy moduł jest włączony
    if not ENABLE_CORPUS_INSIGHTS:
        return CorpusAnalysisResult(
            insights=[],
            overall_naturalness=100.0,
            style_detected="unknown",
            summary="Corpus insights disabled (ENABLE_CORPUS_INSIGHTS=false)",
            word_count=0
        )
    
    # Bezpieczne przetwarzanie
    if not text or not isinstance(text, str):
        return CorpusAnalysisResult(
            insights=[],
            overall_naturalness=100.0,
            style_detected="unknown",
            summary="Brak tekstu do analizy",
            word_count=0
        )
    
    insights = []
    word_count = len(_extract_words(text))
    
    try:
        # 1. Diakrytyki (WYSOKI PRIORYTET - kluczowy marker autentyczności)
        if include_diacritics and INCLUDE_DIACRITICS:
            insights.append(calculate_diacritic_ratio(text))
        
        # 2. Długość słów
        if include_word_length and INCLUDE_WORD_LENGTH:
            word_insight = calculate_word_length_stats(text)
            insights.append(word_insight)
        
        # 3. Samogłoski
        if include_vowels and INCLUDE_VOWELS:
            insights.append(analyze_vowel_ratio(text))
        
        # 4. FOG-PL
        if include_fog and INCLUDE_FOG:
            insights.append(calculate_fog_pl_index(text))
        
        # 5. Interpunkcja
        if include_punctuation and INCLUDE_PUNCTUATION:
            insights.append(calculate_punctuation_density(text))
        
    except Exception as e:
        # W razie błędu - nie blokuj, zwróć częściowe wyniki
        insights.append(CorpusInsight(
            metric="error",
            value=0,
            target=0,
            deviation=0,
            severity=InsightSeverity.OBSERVATION,
            message=f"Błąd podczas analizy: {str(e)[:100]}"
        ))
    
    # Oblicz orientacyjną "naturalność" (tylko informacyjnie, NIE do blokowania!)
    valid_insights = [i for i in insights if i.value > 0 and i.target > 0]
    if valid_insights:
        deviations = []
        for i in valid_insights:
            normalized_dev = abs(i.deviation / i.target)
            deviations.append(min(normalized_dev, 1.0))  # Cap at 100%
        
        if deviations:
            avg_deviation = sum(deviations) / len(deviations)
            naturalness = max(0, (1 - avg_deviation) * 100)
        else:
            naturalness = 100.0
    else:
        naturalness = 100.0
    
    # Wykryj styl
    style = "publicystyka"
    for insight in insights:
        if insight.metric == "word_length_avg" and insight.details:
            style = insight.details.get("style_detected", "publicystyka")
            break
    
    # Generuj podsumowanie
    suggestions = [i for i in insights if i.severity == InsightSeverity.SUGGESTION]
    if suggestions:
        summary = f"Wykryto {len(suggestions)} sugestii poprawy naturalności tekstu."
    else:
        summary = "Tekst zgodny z normami korpusu NKJP."
    
    return CorpusAnalysisResult(
        insights=insights,
        overall_naturalness=naturalness,
        style_detected=style,
        summary=summary,
        word_count=word_count
    )


# ============================================================================
# INTEGRACJA Z MOE VALIDATOR
# ============================================================================

def get_corpus_insights_for_moe(
    batch_text: str,
    include_all: bool = True
) -> Dict[str, Any]:
    """
    Funkcja pomocnicza do integracji z MOE Validator.
    
    Zwraca insights jako dodatkowe pole w response,
    NIE jako część issues (które mogą blokować).
    
    GWARANCJA: Nigdy nie wpływa na action (CONTINUE/FIX/REWRITE)!
    
    Użycie w moe_batch_validator.py:
    
        from polish_corpus_metrics_v41 import get_corpus_insights_for_moe
        
        # Po walidacji MoE (na końcu funkcji):
        result = validate_batch_moe(...)
        
        # Dodaj insights jako osobne pole:
        corpus = get_corpus_insights_for_moe(batch_text)
        result_dict = result.to_dict()
        result_dict["corpus_insights"] = corpus
    
    Args:
        batch_text: Tekst batcha do analizy
        include_all: Czy włączyć wszystkie metryki (default: True)
    
    Returns:
        Dict z insights do dodania do response (nigdy nie blokuje!)
    """
    try:
        analysis = analyze_corpus_metrics(
            batch_text,
            include_fog=include_all,
            include_punctuation=include_all,
            include_diacritics=True,  # Zawsze włączone - kluczowe
            include_word_length=include_all,
            include_vowels=include_all
        )
        
        # Wyodrębnij tylko sugestie (do pola naturalness_hints)
        suggestions = [
            {
                "metric": i.metric,
                "message": i.message,
                "suggestion": i.suggestion,
            }
            for i in analysis.insights
            if i.severity == InsightSeverity.SUGGESTION and i.suggestion
        ]
        
        return {
            "enabled": True,
            "insights": [i.to_dict() for i in analysis.insights],
            "naturalness_score": round(analysis.overall_naturalness, 1),
            "style_detected": analysis.style_detected,
            "word_count": analysis.word_count,
            "summary": analysis.summary,
            "suggestions": suggestions,
            "suggestions_count": len(suggestions),
            
            # WAŻNE: Jawne oznaczenie że to tylko informacja
            "affects_validation": False,
            "is_blocking": False,
            "blocks_action": False,
        }
        
    except Exception as e:
        # W razie błędu - zwróć pusty wynik, NIGDY nie blokuj!
        return {
            "enabled": False,
            "error": str(e)[:200],
            "insights": [],
            "suggestions": [],
            
            # WAŻNE: Nawet przy błędzie - nie blokuje
            "affects_validation": False,
            "is_blocking": False,
            "blocks_action": False,
        }


def get_naturalness_hints(text: str) -> List[Dict[str, str]]:
    """
    Zwraca tylko listę sugestii poprawy naturalności.
    
    Uproszczona funkcja do użycia w pre_batch_info.
    
    Returns:
        Lista słowników z polami: metric, hint
    """
    try:
        analysis = analyze_corpus_metrics(text)
        
        return [
            {
                "metric": i.metric,
                "hint": i.suggestion
            }
            for i in analysis.insights
            if i.severity == InsightSeverity.SUGGESTION and i.suggestion
        ]
        
    except Exception:
        return []


# ============================================================================
# TESTY WBUDOWANE
# ============================================================================

def _run_self_tests() -> Dict[str, Any]:
    """
    Uruchamia testy bezpieczeństwa modułu.
    
    Returns:
        Dict z wynikami testów
    """
    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    # Test 1: Nigdy nie blokuje walidacji
    test_texts = [
        "",
        "Test",
        "Zażółć gęślą jaźń" * 100,
        "test test test" * 100,
        "Konstantynopolitańczykowianeczka" * 20,
    ]
    
    for text in test_texts:
        result = analyze_corpus_metrics(text)
        if result.blocks_validation == False:
            results["passed"] += 1
            results["tests"].append({"name": f"blocks_validation=False for '{text[:20]}...'", "passed": True})
        else:
            results["failed"] += 1
            results["tests"].append({"name": f"blocks_validation=False for '{text[:20]}...'", "passed": False})
    
    # Test 2: Severity nigdy nie jest critical
    result = analyze_corpus_metrics("Test tekstu polskiego z różnymi znakami." * 10)
    all_safe = all(
        i.severity in [InsightSeverity.INFO, InsightSeverity.SUGGESTION, InsightSeverity.OBSERVATION]
        for i in result.insights
    )
    
    if all_safe:
        results["passed"] += 1
        results["tests"].append({"name": "severity_never_critical", "passed": True})
    else:
        results["failed"] += 1
        results["tests"].append({"name": "severity_never_critical", "passed": False})
    
    # Test 3: Graceful handling of edge cases
    edge_cases = [None, "", " ", ".", "12345", "\n\n\n"]
    edge_ok = True
    
    for ec in edge_cases:
        try:
            result = analyze_corpus_metrics(ec or "")
            if result.blocks_validation != False:
                edge_ok = False
        except Exception:
            edge_ok = False
    
    if edge_ok:
        results["passed"] += 1
        results["tests"].append({"name": "edge_cases_graceful", "passed": True})
    else:
        results["failed"] += 1
        results["tests"].append({"name": "edge_cases_graceful", "passed": False})
    
    results["all_passed"] = results["failed"] == 0
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📊 POLISH CORPUS METRICS v41.2")
    print("=" * 70)
    
    # Uruchom testy bezpieczeństwa
    print("\n🔒 Testy bezpieczeństwa:")
    test_results = _run_self_tests()
    
    for test in test_results["tests"]:
        icon = "✅" if test["passed"] else "❌"
        print(f"   {icon} {test['name']}")
    
    print(f"\n   Passed: {test_results['passed']}, Failed: {test_results['failed']}")
    
    if test_results["all_passed"]:
        print("   ✅ Wszystkie testy przeszły - moduł jest bezpieczny")
    else:
        print("   ❌ UWAGA: Niektóre testy nie przeszły!")
    
    # Przykład użycia
    print("\n" + "=" * 70)
    print("📝 Przykład analizy:")
    print("=" * 70)
    
    sample_text = """
    Ubezwłasnowolnienie to instytucja prawa cywilnego, która pozwala na 
    ograniczenie zdolności do czynności prawnych osoby, która z powodu 
    choroby psychicznej, niedorozwoju umysłowego lub innego rodzaju 
    zaburzeń psychicznych nie jest w stanie kierować swoim postępowaniem.
    
    Sąd okręgowy rozpatruje sprawy o ubezwłasnowolnienie. Wniosek może 
    złożyć małżonek, krewny w linii prostej, rodzeństwo lub prokurator.
    """
    
    result = analyze_corpus_metrics(sample_text)
    
    print(f"\n📊 Wyniki dla tekstu ({result.word_count} słów):")
    print(f"   Styl wykryty: {result.style_detected}")
    print(f"   Naturalność: {result.overall_naturalness:.1f}/100")
    print(f"   Podsumowanie: {result.summary}")
    
    print("\n📋 Insights:")
    for insight in result.insights:
        icon = "ℹ️" if insight.severity == InsightSeverity.INFO else "💡"
        print(f"   {icon} [{insight.metric}] {insight.message}")
        if insight.suggestion:
            print(f"      → {insight.suggestion}")
    
    print(f"\n   blocks_validation: {result.blocks_validation} (ZAWSZE False)")
