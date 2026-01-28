"""
===============================================================================
📊 MATTR CALCULATOR v41.0 - Moving Average Type-Token Ratio
===============================================================================

PROBLEM Z PROSTYM TTR:
TTR (unique_words / total_words) spada z długością tekstu.
Tekst 500 słów może mieć TTR 0.60, ale tekst 5000 słów tylko TTR 0.30.
To błąd wielkości próbki - nie można porównywać tekstów różnej długości.

ROZWIĄZANIE - MATTR:
Moving Average Type-Token Ratio oblicza TTR w "oknach" (np. 500 słów)
i uśrednia wyniki. Dzięki temu:
- Wynik jest porównywalny między tekstami różnej długości
- Lokalnie mierzy różnorodność słownictwa
- Bardziej wiarygodny dla detekcji AI

PROGI (oparte na empirycznych obserwacjach):
- MATTR < 0.35 = CRITICAL (niskie zróżnicowanie, sygnał AI)
- MATTR 0.35-0.42 = WARNING (strefa podejrzana)
- MATTR > 0.42 = OK (dobra różnorodność)

===============================================================================
"""

import re
import statistics
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# POLSKIE STOP WORDS (do wykluczenia z analizy)
# ============================================================================

POLISH_STOP_WORDS: Set[str] = {
    # Spójniki
    "i", "oraz", "a", "ale", "lub", "albo", "jednak", "więc", "bo", "że",
    "czy", "jeśli", "gdy", "kiedy", "choć", "chociaż", "ponieważ", "gdyż",
    
    # Przyimki
    "w", "we", "na", "z", "ze", "do", "od", "przy", "po", "za", "przed",
    "nad", "pod", "między", "przez", "dla", "bez", "o", "u", "ku",
    
    # Zaimki
    "ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one",
    "mnie", "mi", "cię", "ci", "go", "mu", "jej", "nas", "was", "im", "ich",
    "ten", "ta", "to", "ci", "te", "tamten", "tamta", "tamto",
    "który", "która", "które", "którzy", "których", "której", "któremu",
    "co", "kto", "jaki", "jaka", "jakie", "jakiego", "jakiej",
    "sam", "sama", "samo", "sami", "same",
    "się", "siebie", "sobie", "sobą",
    
    # Czasowniki posiłkowe
    "być", "jest", "są", "był", "była", "było", "byli", "były",
    "będzie", "będą", "będę", "będziesz", "będziemy", "będziecie",
    "mieć", "ma", "mają", "miał", "miała", "miało", "mieli", "miały",
    "zostać", "zostanie", "zostaną", "został", "została", "zostało",
    
    # Partykuły i przysłówki
    "nie", "tak", "też", "także", "również", "tylko", "już", "jeszcze",
    "bardzo", "bardziej", "najbardziej", "dość", "dosyć", "zbyt",
    "tu", "tutaj", "tam", "teraz", "wtedy", "zawsze", "nigdy",
    "może", "można", "trzeba", "należy", "warto",
    
    # Liczebniki
    "jeden", "jedna", "jedno", "dwa", "dwie", "trzy", "cztery", "pięć",
    "pierwszy", "druga", "trzeci",
    
    # Inne częste
    "jako", "jak", "gdzie", "czym", "tym", "tego", "tej", "tych",
    "ile", "tyle", "kilka", "wiele", "wielu", "wszystko", "wszystkie",
    "każdy", "każda", "każde", "żaden", "żadna", "żadne",
    "inny", "inna", "inne", "innych",
}


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class MATTRConfig:
    """Konfiguracja kalkulatora MATTR."""
    
    # Rozmiar okna (w słowach)
    WINDOW_SIZE: int = 500
    
    # Minimalna liczba słów do analizy MATTR
    MIN_WORDS_FOR_MATTR: int = 500
    
    # Progi MATTR
    MATTR_CRITICAL_LOW: float = 0.35
    MATTR_WARNING_LOW: float = 0.42
    MATTR_OK_MIN: float = 0.42
    MATTR_OK_MAX: float = 0.65
    MATTR_WARNING_HIGH: float = 0.70  # Zbyt wysokie może oznaczać nadmiar żargonu
    
    # Czy wykluczać stop words
    EXCLUDE_STOP_WORDS: bool = True


CONFIG = MATTRConfig()


# ============================================================================
# TOKENIZACJA
# ============================================================================

def tokenize_text(text: str, exclude_stop_words: bool = True) -> List[str]:
    """
    Tokenizuje tekst do listy słów (lowercase).
    
    Args:
        text: Tekst do tokenizacji
        exclude_stop_words: Czy wykluczać stop words
        
    Returns:
        Lista słów (lowercase)
    """
    # Wyciągnij słowa (alfanumeryczne + polskie znaki)
    words = re.findall(r'\b[a-ząćęłńóśźż]+\b', text.lower())
    
    # Filtruj krótkie (1-2 znaki) i stop words
    if exclude_stop_words:
        words = [w for w in words if len(w) > 2 and w not in POLISH_STOP_WORDS]
    else:
        words = [w for w in words if len(w) > 2]
    
    return words


# ============================================================================
# GŁÓWNA FUNKCJA MATTR
# ============================================================================

def calculate_mattr(
    text: str,
    window_size: int = None,
    config: MATTRConfig = None
) -> Dict[str, Any]:
    """
    Oblicza Moving Average Type-Token Ratio.
    
    MATTR oblicza TTR w przesuwających się oknach i uśrednia wyniki.
    Bardziej wiarygodna miara niż prosty TTR dla dłuższych tekstów.
    
    Args:
        text: Tekst do analizy
        window_size: Rozmiar okna (domyślnie 500)
        config: Konfiguracja
        
    Returns:
        Dict z wynikami:
        - value: wartość MATTR (0-1)
        - std: odchylenie standardowe TTR między oknami
        - status: OK/WARNING/CRITICAL
        - message: komunikat diagnostyczny
        - method: "mattr" lub "standard_ttr" (dla krótkich tekstów)
        - window_size: użyty rozmiar okna
        - windows_count: liczba okien
        - word_count: całkowita liczba słów
        - score: znormalizowany score (0-100)
    """
    if config is None:
        config = CONFIG
    
    if window_size is None:
        window_size = config.WINDOW_SIZE
    
    # Tokenizuj
    words = tokenize_text(text, exclude_stop_words=config.EXCLUDE_STOP_WORDS)
    word_count = len(words)
    
    # Za mało słów - użyj standardowego TTR
    if word_count < config.MIN_WORDS_FOR_MATTR:
        return _calculate_simple_ttr(words, word_count, config)
    
    # Oblicz TTR dla każdego okna
    ttr_values = []
    
    for i in range(word_count - window_size + 1):
        window = words[i:i + window_size]
        unique = len(set(window))
        ttr = unique / window_size
        ttr_values.append(ttr)
    
    # Oblicz MATTR (średnia TTR z wszystkich okien)
    mattr = statistics.mean(ttr_values)
    mattr_std = statistics.stdev(ttr_values) if len(ttr_values) > 1 else 0
    
    # Określ status i score
    status, score, message = _evaluate_mattr(mattr, config)
    
    return {
        "value": round(mattr, 3),
        "std": round(mattr_std, 4),
        "status": status.value,
        "message": message,
        "method": "mattr",
        "window_size": window_size,
        "windows_count": len(ttr_values),
        "word_count": word_count,
        "score": score,
        "min_ttr_window": round(min(ttr_values), 3) if ttr_values else 0,
        "max_ttr_window": round(max(ttr_values), 3) if ttr_values else 0,
    }


def _calculate_simple_ttr(
    words: List[str],
    word_count: int,
    config: MATTRConfig
) -> Dict[str, Any]:
    """
    Fallback do prostego TTR dla krótkich tekstów.
    """
    if word_count == 0:
        return {
            "value": 0.0,
            "status": "INSUFFICIENT_DATA",
            "message": "Brak słów do analizy",
            "method": "none",
            "word_count": 0,
            "score": 50
        }
    
    unique = len(set(words))
    ttr = unique / word_count
    
    # Użyj tych samych progów co MATTR (w przybliżeniu)
    status, score, message = _evaluate_mattr(ttr, config)
    
    return {
        "value": round(ttr, 3),
        "std": 0.0,
        "status": status.value,
        "message": f"(Simple TTR - tekst < {config.MIN_WORDS_FOR_MATTR} słów) {message}",
        "method": "standard_ttr",
        "window_size": word_count,
        "windows_count": 1,
        "word_count": word_count,
        "unique_words": unique,
        "score": score
    }


def _evaluate_mattr(mattr: float, config: MATTRConfig) -> tuple:
    """
    Ocenia wartość MATTR i zwraca (status, score, message).
    """
    if mattr < config.MATTR_CRITICAL_LOW:
        status = Severity.CRITICAL
        score = max(10, int(mattr / config.MATTR_CRITICAL_LOW * 40))
        message = f"⚠️ MATTR {mattr:.3f} < {config.MATTR_CRITICAL_LOW} = niskie zróżnicowanie słownictwa"
        
    elif mattr < config.MATTR_WARNING_LOW:
        status = Severity.WARNING
        score = 40 + int((mattr - config.MATTR_CRITICAL_LOW) / 
                        (config.MATTR_WARNING_LOW - config.MATTR_CRITICAL_LOW) * 30)
        message = f"⚠ MATTR {mattr:.3f} < {config.MATTR_WARNING_LOW} = strefa podejrzana"
        
    elif mattr <= config.MATTR_OK_MAX:
        status = Severity.OK
        # Score 70-100 w zależności od tego jak blisko optimum (0.50)
        optimal = 0.50
        deviation = abs(mattr - optimal)
        score = 85 + int((1 - deviation / 0.15) * 15)
        score = min(100, max(70, score))
        message = f"✅ MATTR {mattr:.3f} = dobre zróżnicowanie słownictwa"
        
    elif mattr <= config.MATTR_WARNING_HIGH:
        status = Severity.WARNING
        score = 65
        message = f"⚠ MATTR {mattr:.3f} > {config.MATTR_OK_MAX} = bardzo wysokie (sprawdź nadmiar żargonu)"
        
    else:
        status = Severity.WARNING
        score = 55
        message = f"⚠ MATTR {mattr:.3f} > {config.MATTR_WARNING_HIGH} = nienaturalnie wysokie"
    
    return status, score, message


# ============================================================================
# PORÓWNANIE TTR vs MATTR
# ============================================================================

def compare_ttr_mattr(text: str) -> Dict[str, Any]:
    """
    Porównuje prosty TTR z MATTR dla tego samego tekstu.
    Pokazuje dlaczego MATTR jest lepszą miarą.
    """
    words = tokenize_text(text, exclude_stop_words=True)
    word_count = len(words)
    
    # Simple TTR
    unique = len(set(words))
    simple_ttr = unique / word_count if word_count > 0 else 0
    
    # MATTR
    mattr_result = calculate_mattr(text)
    
    return {
        "word_count": word_count,
        "simple_ttr": round(simple_ttr, 3),
        "mattr": mattr_result["value"],
        "difference": round(abs(simple_ttr - mattr_result["value"]), 3),
        "mattr_method": mattr_result["method"],
        "recommendation": "MATTR" if word_count >= CONFIG.MIN_WORDS_FOR_MATTR else "TTR (tekst krótki)"
    }


# ============================================================================
# INTEGRACJA Z AI_DETECTION_METRICS
# ============================================================================

def get_vocabulary_richness_v41(text: str) -> Dict[str, Any]:
    """
    Zastępuje calculate_vocabulary_richness() z ai_detection_metrics.py.
    
    Używa MATTR dla tekstów >= 500 słów, TTR dla krótszych.
    """
    mattr_result = calculate_mattr(text)
    
    return {
        "value": mattr_result["value"],
        "score": mattr_result["score"],
        "status": mattr_result["status"],
        "message": mattr_result["message"],
        "method": mattr_result["method"],
        "word_count": mattr_result["word_count"],
        # Kompatybilność wsteczna
        "ttr": mattr_result["value"],  # alias
        "normalized_score": mattr_result["score"] / 100  # 0-1 dla wag
    }


# ============================================================================
# INSTRUKCJA INTEGRACJI
# ============================================================================

"""
INTEGRACJA Z BRAJEN:

1. W ai_detection_metrics.py:

   # Zamień import/funkcję calculate_vocabulary_richness na:
   from mattr_calculator_v41 import get_vocabulary_richness_v41 as calculate_vocabulary_richness

   # Lub jeśli chcesz zachować starą funkcję jako fallback:
   from mattr_calculator_v41 import get_vocabulary_richness_v41
   
   def calculate_vocabulary_richness(text: str) -> Dict[str, Any]:
       # Użyj MATTR v41
       return get_vocabulary_richness_v41(text)

2. Progi pozostają podobne, ale MATTR daje bardziej stabilne wyniki.

3. Waga w WEIGHTS może pozostać 0.18 lub zmniejszyć do 0.14 
   (jeśli dodajesz paragraph_cv z wagą 0.15).
"""


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test z krótkim tekstem
    short_text = """
    Ubezwłasnowolnienie to poważna decyzja prawna. Sąd musi zbadać 
    wszystkie okoliczności sprawy. Procedura trwa kilka miesięcy.
    """
    
    # Test z długim tekstem (powtarzalny - symulacja AI)
    ai_like_text = """
    Ubezwłasnowolnienie jest instytucją prawną uregulowaną w Kodeksie cywilnym.
    Procedura ubezwłasnowolnienia wymaga złożenia wniosku do sądu okręgowego.
    Sąd okręgowy przeprowadza postępowanie z udziałem biegłych sądowych.
    Biegli sądowi wydają opinię psychiatryczną i psychologiczną w sprawie.
    Opinia biegłych jest podstawą do wydania orzeczenia przez sąd.
    Orzeczenie sądu określa zakres ubezwłasnowolnienia osoby fizycznej.
    Osoba fizyczna może być ubezwłasnowolniona całkowicie lub częściowo.
    Ubezwłasnowolnienie całkowite pozbawia zdolności do czynności prawnych.
    Zdolność do czynności prawnych jest niezbędna do zawierania umów.
    Umowy zawarte przez osobę ubezwłasnowolnioną są nieważne z mocy prawa.
    """ * 3  # Powtórzenie żeby mieć > 500 słów
    
    # Test z naturalnym tekstem
    human_like_text = """
    Ubezwłasnowolnienie to jedna z najtrudniejszych decyzji, jakie może podjąć sąd.
    Dlaczego? Bo dotyka sfery najbardziej intymnej - autonomii człowieka.
    
    Wyobraź sobie sytuację: Twoja babcia ma 85 lat. Przez całe życie była niezależna,
    prowadziła własny biznes, wychowała troje dzieci. Teraz demencja postępuje.
    Zaczyna zapominać twarze, gubiła się w drodze do sklepu, a ostatnio dała
    "miłemu panu z telefonu" numer karty kredytowej.
    
    Co robić? Rodzina stoi przed dylematem. Z jednej strony trzeba chronić babcię
    przed oszustami i jej własnymi, niestety, błędnymi decyzjami. Z drugiej -
    nikt nie chce jej odbierać godności, traktować jak dziecko.
    
    Prawo daje narzędzie: ubezwłasnowolnienie. Ale to narzędzie obosieczne.
    Sąd nie podejmie takiej decyzji pochopnie. Wymaga dowodów, opinii biegłych,
    przesłuchań. Procedura może trwać rok albo dłużej. I dobrze - bo chodzi o coś
    więcej niż formalności. Chodzi o człowieka.
    
    Czy warto? To zależy od konkretnej sytuacji. Czasem tak. Czasem lepsze są
    inne rozwiązania: pełnomocnictwo, pomoc społeczna, opieka rodziny bez
    formalnego pozbawienia praw. Każdy przypadek jest inny.
    """ * 2  # Powtórzenie żeby mieć > 500 słów
    
    print("=" * 60)
    print("TEST 1: Krótki tekst (TTR fallback)")
    print("=" * 60)
    result1 = calculate_mattr(short_text)
    print(f"Method: {result1['method']}")
    print(f"Value: {result1['value']}")
    print(f"Status: {result1['status']}")
    print(f"Score: {result1['score']}")
    print(f"Word count: {result1['word_count']}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Długi tekst AI-like (powtarzalne słownictwo)")
    print("=" * 60)
    result2 = calculate_mattr(ai_like_text)
    print(f"Method: {result2['method']}")
    print(f"MATTR: {result2['value']}")
    print(f"Std: {result2['std']}")
    print(f"Status: {result2['status']}")
    print(f"Score: {result2['score']}")
    print(f"Word count: {result2['word_count']}")
    print(f"Windows: {result2['windows_count']}")
    print(f"Message: {result2['message']}")
    
    print("\n" + "=" * 60)
    print("TEST 3: Długi tekst human-like (zróżnicowane słownictwo)")
    print("=" * 60)
    result3 = calculate_mattr(human_like_text)
    print(f"Method: {result3['method']}")
    print(f"MATTR: {result3['value']}")
    print(f"Std: {result3['std']}")
    print(f"Status: {result3['status']}")
    print(f"Score: {result3['score']}")
    print(f"Word count: {result3['word_count']}")
    print(f"Windows: {result3['windows_count']}")
    print(f"Message: {result3['message']}")
    
    print("\n" + "=" * 60)
    print("PORÓWNANIE TTR vs MATTR")
    print("=" * 60)
    comp = compare_ttr_mattr(human_like_text)
    print(f"Simple TTR: {comp['simple_ttr']}")
    print(f"MATTR: {comp['mattr']}")
    print(f"Difference: {comp['difference']}")
    print(f"Recommendation: {comp['recommendation']}")
