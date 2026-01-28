"""
===============================================================================
📊 PARAGRAPH CV ANALYZER v41.0 - Analiza zmienności długości akapitów
===============================================================================

Badania MDPI 2024 pokazują, że CV (Coefficient of Variation) długości akapitów
to #2 cecha do wykrywania tekstu AI (zaraz po CV zdań).

ZASADA:
- Tekst ludzki: WYŻSZY CV (większa zmienność - akapity różnej długości)
- Tekst AI: NIŻSZY CV (monotonne akapity podobnej długości)

PROGI (empiryczne, oparte na analizie):
- CV < 0.25 = CRITICAL (silny sygnał AI)
- CV 0.25-0.35 = WARNING (strefa podejrzana)
- CV > 0.35 = OK (naturalna zmienność)

FORMUŁA:
CV = (odchylenie standardowe długości akapitów) / (średnia długość akapitów)

===============================================================================
"""

import re
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class ParagraphCVConfig:
    """Konfiguracja analizy CV akapitów."""
    
    # Progi CV (oparte na empirycznych obserwacjach)
    CV_CRITICAL_LOW: float = 0.25    # Poniżej = silny sygnał AI
    CV_WARNING_LOW: float = 0.35     # Poniżej = strefa podejrzana
    CV_OK_MIN: float = 0.35          # Powyżej = naturalna zmienność
    CV_WARNING_HIGH: float = 0.80    # Powyżej = zbyt chaotyczne
    CV_CRITICAL_HIGH: float = 1.00   # Powyżej = prawdopodobnie błąd formatowania
    
    # Minimalna liczba akapitów do analizy
    MIN_PARAGRAPHS: int = 3
    
    # Minimalna liczba słów w akapicie (żeby liczyć)
    MIN_WORDS_IN_PARAGRAPH: int = 10
    
    # Target dla optymalizacji
    OPTIMAL_CV_MIN: float = 0.40
    OPTIMAL_CV_MAX: float = 0.60
    
    # Optymalne długości akapitów (w słowach)
    PARAGRAPH_LENGTH_MIN: int = 30
    PARAGRAPH_LENGTH_MAX: int = 150
    PARAGRAPH_LENGTH_OPTIMAL: int = 70


CONFIG = ParagraphCVConfig()


# ============================================================================
# GŁÓWNA FUNKCJA ANALIZY
# ============================================================================

def calculate_paragraph_cv(text: str, config: ParagraphCVConfig = None) -> Dict[str, Any]:
    """
    Oblicza Coefficient of Variation (CV) długości akapitów.
    
    Args:
        text: Tekst do analizy
        config: Opcjonalna konfiguracja
        
    Returns:
        Dict z wynikami analizy:
        - cv: wartość CV (0-1+)
        - status: OK/WARNING/CRITICAL
        - message: komunikat diagnostyczny
        - paragraph_count: liczba akapitów
        - lengths: lista długości
        - mean_length: średnia długość
        - std_length: odchylenie standardowe
        - score: znormalizowany score (0-100)
        - recommendations: lista rekomendacji
        - prebatch_instruction: instrukcja dla GPT
    """
    if config is None:
        config = CONFIG
    
    # Podziel na akapity (podwójny newline lub więcej)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filtruj i oblicz długości
    lengths = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Usuń nagłówki (H2:, H3:, <h2>, etc.)
        if re.match(r'^(h[23]:|\s*<h[23])', para, re.IGNORECASE):
            continue
        
        word_count = len(para.split())
        
        # Tylko akapity z min. słowami
        if word_count >= config.MIN_WORDS_IN_PARAGRAPH:
            lengths.append(word_count)
    
    # Za mało akapitów
    if len(lengths) < config.MIN_PARAGRAPHS:
        return {
            "cv": 0.0,
            "status": "INSUFFICIENT_DATA",
            "message": f"Za mało akapitów do analizy ({len(lengths)}/{config.MIN_PARAGRAPHS})",
            "paragraph_count": len(lengths),
            "lengths": lengths,
            "score": 50,  # Neutral
            "recommendations": [],
            "prebatch_instruction": None
        }
    
    # Oblicz statystyki
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths)
    cv = std_len / mean_len if mean_len > 0 else 0
    
    # Określ status
    recommendations = []
    prebatch_instruction = None
    
    if cv < config.CV_CRITICAL_LOW:
        status = Severity.CRITICAL
        score = max(10, int(cv / config.CV_CRITICAL_LOW * 40))
        message = f"⚠️ PARAGRAPH CV {cv:.2f} < {config.CV_CRITICAL_LOW} = SILNY SYGNAŁ AI"
        recommendations = [
            f"CV akapitów {cv:.2f} jest zbyt niskie - akapity są zbyt jednolite",
            "Zróżnicuj długości akapitów: mieszaj krótkie (30-50 słów), średnie (60-90 słów) i długie (100-150 słów)",
            "Dodaj 1-2 bardzo krótkie akapity (1-2 zdania) po długich blokach tekstu",
            "Unikaj wzorca: wszystkie akapity ~70-80 słów"
        ]
        prebatch_instruction = _generate_prebatch_instruction(cv, lengths, "CRITICAL")
        
    elif cv < config.CV_WARNING_LOW:
        status = Severity.WARNING
        score = 40 + int((cv - config.CV_CRITICAL_LOW) / (config.CV_WARNING_LOW - config.CV_CRITICAL_LOW) * 30)
        message = f"⚠ PARAGRAPH CV {cv:.2f} < {config.CV_WARNING_LOW} = strefa podejrzana"
        recommendations = [
            f"CV akapitów {cv:.2f} jest w strefie podejrzanej",
            "Dodaj więcej zróżnicowania: 1-2 krótkie akapity (2-3 zdania) i 1 dłuższy (120+ słów)",
            "Naturalne teksty mają CV > 0.40"
        ]
        prebatch_instruction = _generate_prebatch_instruction(cv, lengths, "WARNING")
        
    elif cv <= config.CV_WARNING_HIGH:
        status = Severity.OK
        # Score zależy od tego jak blisko optimum
        if config.OPTIMAL_CV_MIN <= cv <= config.OPTIMAL_CV_MAX:
            score = 85 + int((1 - abs(cv - 0.50) / 0.10) * 15)  # 85-100
        else:
            score = 70 + int((1 - abs(cv - 0.50) / 0.30) * 15)  # 70-85
        message = f"✅ PARAGRAPH CV {cv:.2f} = naturalna zmienność"
        prebatch_instruction = None  # Nie potrzeba instrukcji
        
    elif cv <= config.CV_CRITICAL_HIGH:
        status = Severity.WARNING
        score = 60
        message = f"⚠ PARAGRAPH CV {cv:.2f} > {config.CV_WARNING_HIGH} = zbyt chaotyczne"
        recommendations = [
            f"CV akapitów {cv:.2f} jest bardzo wysokie - tekst może wyglądać chaotycznie",
            "Wyrównaj niektóre skrajnie krótkie lub długie akapity"
        ]
        prebatch_instruction = _generate_prebatch_instruction(cv, lengths, "HIGH")
        
    else:
        status = Severity.CRITICAL
        score = 40
        message = f"⚠️ PARAGRAPH CV {cv:.2f} > {config.CV_CRITICAL_HIGH} = prawdopodobny błąd formatowania"
        recommendations = [
            "Sprawdź formatowanie tekstu - mogą być błędne podziały akapitów"
        ]
        prebatch_instruction = None
    
    return {
        "cv": round(cv, 3),
        "status": status.value,
        "message": message,
        "paragraph_count": len(lengths),
        "lengths": lengths,
        "mean_length": round(mean_len, 1),
        "std_length": round(std_len, 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "score": min(100, max(0, score)),
        "recommendations": recommendations,
        "prebatch_instruction": prebatch_instruction
    }


# ============================================================================
# GENERATOR INSTRUKCJI PRE-BATCH
# ============================================================================

def _generate_prebatch_instruction(cv: float, lengths: List[int], severity: str) -> str:
    """
    Generuje konkretną instrukcję dla GPT do włączenia w pre-batch info.
    """
    mean_len = statistics.mean(lengths)
    
    if severity == "CRITICAL":
        return f"""
🚨 PARAGRAPH DIVERSITY ALERT (CV={cv:.2f})

PROBLEM: Akapity są zbyt jednolite (średnio {mean_len:.0f} słów każdy).
To silny marker tekstu AI - musisz ZRÓŻNICOWAĆ długości!

WYMAGANIA DLA TEGO BATCHA:
1. KRÓTKI akapit (2-3 zdania, 20-40 słów) - użyj po skomplikowanym wyjaśnieniu
2. ŚREDNI akapit (4-5 zdań, 60-90 słów) - standardowy
3. DŁUŻSZY akapit (6-8 zdań, 100-140 słów) - dla głównej treści

NIE PISZ wszystkich akapitów podobnej długości!
Naturalny tekst ma akapity od 25 do 150 słów z CV > 0.40.
"""
    
    elif severity == "WARNING":
        return f"""
⚠️ PARAGRAPH DIVERSITY (CV={cv:.2f})

Akapity są dość jednolite (średnio {mean_len:.0f} słów).
Dodaj więcej zróżnicowania w tym batchu:

- Napisz minimum 1 krótki akapit (25-40 słów) 
- Napisz minimum 1 dłuższy akapit (100+ słów)
- Nie wszystkie akapity powinny mieć 60-80 słów

Target: CV > 0.40
"""
    
    elif severity == "HIGH":
        return f"""
⚠️ PARAGRAPH STRUCTURE (CV={cv:.2f})

Akapity są zbyt różnorodne - tekst może wyglądać chaotycznie.
W tym batchu:
- Unikaj bardzo krótkich akapitów (1 zdanie)
- Unikaj bardzo długich akapitów (200+ słów)
- Targetuj zakres 40-120 słów dla większości akapitów
"""
    
    return None


# ============================================================================
# ANALIZA Z SUGESTIAMI FIX
# ============================================================================

def analyze_paragraph_structure(text: str) -> Dict[str, Any]:
    """
    Rozszerzona analiza struktury akapitów z konkretnymi sugestiami naprawy.
    """
    result = calculate_paragraph_cv(text)
    
    if result["status"] == "INSUFFICIENT_DATA":
        return result
    
    lengths = result["lengths"]
    
    # Analiza rozkładu
    short_count = sum(1 for l in lengths if l < 50)
    medium_count = sum(1 for l in lengths if 50 <= l <= 100)
    long_count = sum(1 for l in lengths if l > 100)
    
    total = len(lengths)
    
    distribution = {
        "short": {"count": short_count, "pct": round(short_count / total * 100, 1)},
        "medium": {"count": medium_count, "pct": round(medium_count / total * 100, 1)},
        "long": {"count": long_count, "pct": round(long_count / total * 100, 1)}
    }
    
    # Idealna dystrybucja: ~25% krótkich, ~50% średnich, ~25% długich
    distribution_score = 100
    
    # Penalty za brak krótkich
    if short_count == 0:
        distribution_score -= 20
    elif distribution["short"]["pct"] < 15:
        distribution_score -= 10
    
    # Penalty za brak długich
    if long_count == 0:
        distribution_score -= 15
    elif distribution["long"]["pct"] < 10:
        distribution_score -= 8
    
    # Penalty za zbyt dużo średnich (monotonia)
    if distribution["medium"]["pct"] > 70:
        distribution_score -= 15
    
    result["distribution"] = distribution
    result["distribution_score"] = max(0, distribution_score)
    
    # Konkretne sugestie
    fix_suggestions = []
    
    if short_count == 0:
        fix_suggestions.append(
            "BRAK KRÓTKICH AKAPITÓW: Dodaj 1-2 krótkie akapity (20-40 słów) "
            "po złożonych wyjaśnieniach lub przed zmianą tematu"
        )
    
    if long_count == 0:
        fix_suggestions.append(
            "BRAK DŁUGICH AKAPITÓW: Rozwiń 1-2 akapity do 100-140 słów "
            "dla głównych punktów artykułu"
        )
    
    if distribution["medium"]["pct"] > 70:
        fix_suggestions.append(
            f"MONOTONIA: {distribution['medium']['pct']:.0f}% akapitów ma 50-100 słów. "
            "Skróć niektóre do 25-40 słów, rozwiń inne do 110-140 słów"
        )
    
    result["fix_suggestions"] = fix_suggestions
    
    return result


# ============================================================================
# INTEGRACJA Z PRE-BATCH INFO
# ============================================================================

def get_paragraph_cv_for_prebatch(
    accumulated_text: str,
    batch_number: int
) -> Optional[Dict[str, Any]]:
    """
    Funkcja do wywołania w enhanced_pre_batch.py.
    
    Zwraca instrukcję dla GPT tylko jeśli CV jest problematyczne.
    
    Args:
        accumulated_text: Dotychczas napisany tekst
        batch_number: Numer batcha (instrukcje tylko od batch 2+)
        
    Returns:
        Dict z instrukcją lub None jeśli OK
    """
    # Nie analizuj pierwszego batcha (za mało danych)
    if batch_number < 2:
        return None
    
    result = calculate_paragraph_cv(accumulated_text)
    
    if result["status"] in ["CRITICAL", "WARNING"]:
        return {
            "alert_type": "PARAGRAPH_CV",
            "severity": result["status"],
            "cv": result["cv"],
            "instruction": result["prebatch_instruction"],
            "recommendations": result["recommendations"]
        }
    
    return None


# ============================================================================
# INSTRUKCJA INTEGRACJI
# ============================================================================

"""
INTEGRACJA Z BRAJEN:

1. W enhanced_pre_batch.py, w funkcji generate_pre_batch_info():

   from paragraph_cv_analyzer_v41 import get_paragraph_cv_for_prebatch
   
   # Po sekcji z burstiness/humanness
   paragraph_cv_alert = get_paragraph_cv_for_prebatch(
       accumulated_text=accumulated_content,
       batch_number=batch_number
   )
   
   if paragraph_cv_alert:
       style_warnings.append(paragraph_cv_alert["instruction"])

2. W ai_detection_metrics.py, w calculate_humanness_score():

   from paragraph_cv_analyzer_v41 import calculate_paragraph_cv
   
   # Dodaj do components
   paragraph_cv = calculate_paragraph_cv(text)
   
   # Dodaj do wag
   scores["paragraph_cv"] = paragraph_cv["score"] / 100  # normalize to 0-1
   
   # Zaktualizuj WEIGHTS
   WEIGHTS["paragraph_cv"] = 0.15

3. W batch_review_system.py:

   from paragraph_cv_analyzer_v41 import analyze_paragraph_structure
   
   # W walidacji batcha
   para_analysis = analyze_paragraph_structure(batch_content)
   if para_analysis["status"] == "CRITICAL":
       issues.append(f"Paragraph CV: {para_analysis['message']}")
"""


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test z tekstem o niskim CV (AI-like)
    ai_text = """
    Ubezwłasnowolnienie to instytucja prawna regulowana przez Kodeks cywilny. 
    Procedura wymaga złożenia wniosku do sądu okręgowego właściwego miejscowo.
    Sąd przeprowadza postępowanie z udziałem biegłych psychiatrów i psychologów.
    
    Przesłanki ubezwłasnowolnienia obejmują chorobę psychiczną i niedorozwój.
    Postępowanie może trwać od kilku miesięcy do ponad roku w zależności od sprawy.
    Kurator sprawuje opiekę nad majątkiem osoby ubezwłasnowolnionej całkowicie.
    
    Skutki prawne ubezwłasnowolnienia są bardzo poważne dla osoby której dotyczy.
    Osoba traci zdolność do czynności prawnych i nie może samodzielnie decydować.
    Wszystkie ważne decyzje musi podejmować kurator lub opiekun prawny osoby.
    """
    
    # Test z tekstem o wysokim CV (human-like)
    human_text = """
    Ubezwłasnowolnienie to poważna decyzja. Sąd nie podejmuje jej lekko.
    
    Procedura zaczyna się od wniosku. Kto może go złożyć? Przede wszystkim najbliższa 
    rodzina - małżonek, rodzice, dzieci, rodzeństwo. Prokurator również ma takie 
    uprawnienie, choć korzysta z niego rzadziej. Sam zainteresowany nie może złożyć 
    wniosku o własne ubezwłasnowolnienie - to jedna z ciekawszych cech tej instytucji 
    prawnej, która budzi czasem kontrowersje wśród prawników zajmujących się prawami 
    człowieka i autonomią jednostki.
    
    Co dalej?
    
    Sąd wyznacza biegłych. Psychiatra i psycholog badają osobę, której dotyczy wniosek. 
    To kluczowy etap - od ich opinii zależy bardzo wiele. Biegli muszą odpowiedzieć na 
    konkretne pytania: czy występuje choroba psychiczna? Niedorozwój umysłowy? Inne 
    zaburzenia? I najważniejsze: czy stan ten uniemożliwia samodzielne kierowanie 
    swoim postępowaniem?
    
    Skutki? Daleko idące.
    """
    
    print("=" * 60)
    print("TEST 1: Tekst AI-like (niski CV)")
    print("=" * 60)
    result1 = analyze_paragraph_structure(ai_text)
    print(f"CV: {result1['cv']}")
    print(f"Status: {result1['status']}")
    print(f"Score: {result1['score']}")
    print(f"Distribution: {result1.get('distribution', {})}")
    print(f"Message: {result1['message']}")
    if result1.get('prebatch_instruction'):
        print(f"\nPRE-BATCH INSTRUCTION:\n{result1['prebatch_instruction']}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Tekst human-like (wysoki CV)")
    print("=" * 60)
    result2 = analyze_paragraph_structure(human_text)
    print(f"CV: {result2['cv']}")
    print(f"Status: {result2['status']}")
    print(f"Score: {result2['score']}")
    print(f"Distribution: {result2.get('distribution', {})}")
    print(f"Message: {result2['message']}")
