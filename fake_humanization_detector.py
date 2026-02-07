"""
===============================================================================
FAKE HUMANIZATION DETECTOR v41.1
===============================================================================
Wykrywa "sztuczną humanizację" - gdy agent dodaje krótkie zdania na końcu
akapitów zamiast naturalnie mieszać długości.

PROBLEMY KTÓRE WYKRYWA:
1. Wszystkie krótkie zdania na końcach akapitów
2. Powtarzające się fillery ("To ważne.", "Sprawdź to.")
3. Zdania w "AI zone" (20-25 słów) dominują
4. Brak krótkich zdań w środku akapitów

v41.1: Nowy moduł
===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class FakeHumanizationConfig:
    """Konfiguracja wykrywania fake humanization."""
    
    # Znane fillery
    # v45.0: Rozszerzono o zdania z usuniętej statycznej biblioteki
    # (smart_batch_instructions.py + semantic_phrase_assignment.py)
    # Jeśli GPT nadal je generuje mimo braku w bibliotece → to filler
    KNOWN_FILLERS = [
        # Oryginalne fillery
        "to ważne",
        "to istotne", 
        "sprawdź to",
        "warto wiedzieć",
        "pamiętaj o tym",
        "zapamiętaj to",
        "to kluczowe",
        "oto szczegóły",
        "co dalej",
        "ale uwaga",
        "to proste",
        "to jasne",
        # v45.0: Usunięte z biblioteki, ale GPT może je generować z przyzwyczajenia
        "sąd orzeka",
        "termin biegnie",
        "dowody decydują",
        "prawo wymaga",
        "procedura trwa",
        "wyrok zapada",
        "sprawa się toczy",
        "kara grozi",
        "przepis obowiązuje",
        "lekarz decyduje",
        "badanie wykaże",
        "leczenie trwa",
        "diagnoza potwierdzona",
        "zysk rośnie",
        "ryzyko istnieje",
        "rynek reaguje",
        "warto rozważyć",
        "szczegóły poniżej",
        "praktyka pokazuje",
        "sytuacja jest złożona",
        "definicja jest kluczowa",
        "znaczenie jest jasne",
        "relacja ma znaczenie",
        "kontakt jest ważny",
        "opieka trwa",
        "dobro dziecka",
        "to ważne pojęcie",
        "sankcja jest surowa",
        "odpowiedzialność istnieje",
        "wyrok jest prawomocny",
    ]
    
    # AI zone - zdania które AI typowo produkuje
    AI_ZONE_MIN_WORDS = 18
    AI_ZONE_MAX_WORDS = 26
    AI_ZONE_MAX_RATIO = 0.40  # Max 40% zdań w AI zone
    
    # Minimalne wymagania
    MIN_SHORT_SENTENCES_IN_MIDDLE = 0.10  # 10% krótkich w środku akapitów
    MIN_SHORT_TOTAL = 0.15  # 15% krótkich ogółem (bloker)
    
    # Progi severity
    CRITICAL_THRESHOLD = 0.70  # >70% fillerów na końcach = CRITICAL
    WARNING_THRESHOLD = 0.50   # >50% fillerów na końcach = WARNING


CONFIG = FakeHumanizationConfig()


def split_into_sentences(text: str) -> List[str]:
    """Dzieli tekst na zdania."""
    if not text:
        return []
    
    # Ochrona skrótów
    protected = text
    abbreviations = ['art', 'ust', 'pkt', 'np', 'dr', 'prof', 'mgr', 'inż', 'tj', 'tzn']
    for abbr in abbreviations:
        protected = re.sub(rf'\b{abbr}\.', f'{abbr}@@DOT@@', protected, flags=re.IGNORECASE)
    
    # Split na zdania
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    
    # Przywróć kropki
    sentences = [s.replace('@@DOT@@', '.').strip() for s in sentences if s.strip()]
    
    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """Dzieli tekst na akapity."""
    if not text:
        return []
    
    paragraphs = re.split(r'\n\n+', text.strip())
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


def is_known_filler(sentence: str) -> bool:
    """Sprawdza czy zdanie to znany filler."""
    sentence_lower = sentence.lower().strip().rstrip('.!?')
    return any(filler in sentence_lower for filler in CONFIG.KNOWN_FILLERS)


def is_short_sentence(sentence: str, max_words: int = 8) -> bool:
    """Sprawdza czy zdanie jest krótkie (3-8 słów)."""
    words = len(sentence.split())
    return 2 <= words <= max_words


def is_in_ai_zone(sentence: str) -> bool:
    """Sprawdza czy zdanie jest w typowym zakresie AI (18-26 słów)."""
    words = len(sentence.split())
    return CONFIG.AI_ZONE_MIN_WORDS <= words <= CONFIG.AI_ZONE_MAX_WORDS


def analyze_paragraph_structure(paragraph: str) -> Dict[str, Any]:
    """
    Analizuje strukturę pojedynczego akapitu.
    
    Zwraca:
    - Pozycje krótkich zdań (początek, środek, koniec)
    - Czy kończy się fillerem
    - Rozkład długości
    """
    sentences = split_into_sentences(paragraph)
    
    if len(sentences) < 2:
        return {
            "sentence_count": len(sentences),
            "short_at_start": False,
            "short_in_middle": False,
            "short_at_end": False,
            "ends_with_filler": False,
            "ai_zone_ratio": 0
        }
    
    # Klasyfikuj pozycje
    short_positions = []
    filler_positions = []
    ai_zone_count = 0
    
    for i, sent in enumerate(sentences):
        if is_short_sentence(sent):
            if i == 0:
                short_positions.append("start")
            elif i == len(sentences) - 1:
                short_positions.append("end")
            else:
                short_positions.append("middle")
                
            if is_known_filler(sent):
                filler_positions.append(i)
        
        if is_in_ai_zone(sent):
            ai_zone_count += 1
    
    return {
        "sentence_count": len(sentences),
        "short_at_start": "start" in short_positions,
        "short_in_middle": "middle" in short_positions,
        "short_at_end": "end" in short_positions,
        "ends_with_filler": len(sentences) - 1 in filler_positions if filler_positions else False,
        "filler_count": len(filler_positions),
        "ai_zone_ratio": ai_zone_count / len(sentences) if sentences else 0,
        "short_positions": short_positions
    }


def detect_fake_humanization(text: str) -> Dict[str, Any]:
    """
    Główna funkcja wykrywająca fake humanization.
    
    Zwraca:
    - is_fake: bool - czy wykryto sztuczną humanizację
    - severity: CRITICAL/WARNING/OK
    - score: 0-100 (0 = bardzo fake, 100 = natural)
    - issues: lista problemów
    - recommendations: rekomendacje naprawy
    """
    if not text or len(text) < 100:
        return {
            "is_fake": False,
            "severity": "OK",
            "score": 100,
            "issues": [],
            "recommendations": []
        }
    
    paragraphs = split_into_paragraphs(text)
    all_sentences = split_into_sentences(text)
    
    if len(paragraphs) < 2 or len(all_sentences) < 5:
        return {
            "is_fake": False,
            "severity": "OK", 
            "score": 100,
            "issues": ["Za mało tekstu do analizy"],
            "recommendations": []
        }
    
    # Analizuj każdy akapit
    paragraph_analyses = [analyze_paragraph_structure(p) for p in paragraphs]
    
    # === METRYKI ===
    
    # 1. Ile akapitów kończy się fillerem?
    ends_with_filler_count = sum(1 for pa in paragraph_analyses if pa["ends_with_filler"])
    filler_at_end_ratio = ends_with_filler_count / len(paragraphs)
    
    # 2. Ile krótkich zdań jest w środku vs na końcach?
    total_short_middle = sum(1 for pa in paragraph_analyses if pa["short_in_middle"])
    total_short_end = sum(1 for pa in paragraph_analyses if pa["short_at_end"])
    
    short_position_ratio = 0
    if total_short_middle + total_short_end > 0:
        short_position_ratio = total_short_middle / (total_short_middle + total_short_end)
    
    # 3. Ratio zdań w AI zone
    total_ai_zone_ratio = sum(pa["ai_zone_ratio"] for pa in paragraph_analyses) / len(paragraph_analyses)
    
    # 4. Ogólny % krótkich zdań
    all_short = sum(1 for s in all_sentences if is_short_sentence(s))
    short_total_ratio = all_short / len(all_sentences) if all_sentences else 0
    
    # 5. Powtarzające się fillery
    short_sentences = [s.lower().strip().rstrip('.!?') for s in all_sentences if is_short_sentence(s)]
    filler_counter = Counter(short_sentences)
    repeated_fillers = [f for f, count in filler_counter.items() if count > 1]
    
    # === OCENA ===
    issues = []
    recommendations = []
    
    # Problem 1: Fillery na końcach
    if filler_at_end_ratio > CONFIG.CRITICAL_THRESHOLD:
        issues.append(f"CRITICAL: {filler_at_end_ratio*100:.0f}% akapitów kończy się sztucznym fillerem")
        recommendations.append("Usuń fillery z końców akapitów. Zamiast 'To ważne.' napisz pełne zdanie rozwijające myśl.")
    elif filler_at_end_ratio > CONFIG.WARNING_THRESHOLD:
        issues.append(f"WARNING: {filler_at_end_ratio*100:.0f}% akapitów kończy się fillerem")
        recommendations.append("Zmniejsz liczbę krótkich zdań na końcach akapitów.")
    
    # Problem 2: Brak krótkich w środku
    if short_position_ratio < 0.3 and total_short_middle + total_short_end > 0:
        issues.append(f"WARNING: Tylko {short_position_ratio*100:.0f}% krótkich zdań jest w środku akapitów")
        recommendations.append("Dodaj krótkie zdania W ŚRODKU akapitów, nie tylko na końcach.")
    
    # Problem 3: Za dużo w AI zone
    if total_ai_zone_ratio > CONFIG.AI_ZONE_MAX_RATIO:
        issues.append(f"WARNING: {total_ai_zone_ratio*100:.0f}% zdań ma 18-26 słów (typowa długość AI)")
        recommendations.append("Mieszaj długości zdań bardziej naturalnie. Dodaj zdania 10-15 słów i 28-35 słów.")
    
    # Problem 4: Za mało krótkich ogółem
    if short_total_ratio < CONFIG.MIN_SHORT_TOTAL:
        issues.append(f"WARNING: Tylko {short_total_ratio*100:.0f}% krótkich zdań (cel: 15-25%)")
        recommendations.append("Dodaj więcej krótkich zdań (3-8 słów) naturalnie w tekście.")
    
    # Problem 5: Powtarzające się fillery
    if repeated_fillers:
        issues.append(f"WARNING: Powtarzające się fillery: {', '.join(repeated_fillers[:3])}")
        recommendations.append("Unikaj powtarzania tych samych krótkich zdań.")
    
    # === SCORING ===
    score = 100
    
    # Kary za problemy
    score -= filler_at_end_ratio * 30  # Max -30 za fillery na końcach
    score -= (1 - short_position_ratio) * 20 if total_short_middle + total_short_end > 0 else 0  # Max -20 za brak w środku
    score -= max(0, total_ai_zone_ratio - 0.40) * 50  # Kara za AI zone > 40%
    score -= len(repeated_fillers) * 5  # -5 za każdy powtórzony filler
    
    score = max(0, min(100, score))
    
    # === SEVERITY ===
    if score < 50 or any("CRITICAL" in i for i in issues):
        severity = "CRITICAL"
        is_fake = True
    elif score < 70 or len(issues) >= 2:
        severity = "WARNING"
        is_fake = True
    else:
        severity = "OK"
        is_fake = False
    
    return {
        "is_fake": is_fake,
        "severity": severity,
        "score": round(score, 1),
        "issues": issues,
        "recommendations": recommendations,
        "metrics": {
            "filler_at_end_ratio": round(filler_at_end_ratio, 3),
            "short_position_ratio": round(short_position_ratio, 3),
            "ai_zone_ratio": round(total_ai_zone_ratio, 3),
            "short_total_ratio": round(short_total_ratio, 3),
            "repeated_fillers": repeated_fillers[:5],
            "paragraph_count": len(paragraphs),
            "sentence_count": len(all_sentences)
        }
    }


def generate_natural_humanization_tips(analysis: Dict[str, Any]) -> List[str]:
    """
    Generuje konkretne wskazówki dla naturalnej humanizacji.
    """
    tips = []
    
    if analysis.get("metrics", {}).get("ai_zone_ratio", 0) > 0.35:
        tips.append("🎯 Przeplataj długości: zamiast '15-20-18-22-19' napisz '8-25-12-30-6-18'")
    
    if analysis.get("metrics", {}).get("filler_at_end_ratio", 0) > 0.3:
        tips.append("🚫 Nie kończ akapitów fillerami jak 'To ważne.' - rozwiń myśl w pełne zdanie")
        tips.append("✅ Zamiast: '...wymaga uwagi. To ważne.' napisz: '...wymaga szczególnej uwagi ze strony prawnika.'")
    
    if analysis.get("metrics", {}).get("short_position_ratio", 0) < 0.4:
        tips.append("💡 Krótkie zdania w ŚRODKU akapitu: 'Sąd orzekł. To zmieniło wszystko.' - nie na końcu!")
        tips.append("✅ Przykład: 'Procedura jest złożona. Wymaga trzech etapów. Pierwszy to...'")
    
    tips.append("📊 Cel rozkładu: 20% krótkich (3-8), 55% średnich (10-18), 25% długich (20-30)")
    
    return tips


# === INTEGRATION HELPER ===

def validate_humanization_quality(text: str) -> Dict[str, Any]:
    """
    Główna funkcja do integracji z walidatorem.
    
    Returns:
        {
            "passed": bool,
            "severity": "CRITICAL" | "WARNING" | "OK",
            "score": 0-100,
            "issues": [...],
            "action": "CONTINUE" | "FIX_AND_RETRY" | "REWRITE"
        }
    """
    analysis = detect_fake_humanization(text)
    tips = generate_natural_humanization_tips(analysis)
    
    # Determine action
    if analysis["severity"] == "CRITICAL":
        action = "REWRITE"
        passed = False
    elif analysis["severity"] == "WARNING":
        action = "FIX_AND_RETRY"
        passed = False
    else:
        action = "CONTINUE"
        passed = True
    
    return {
        "passed": passed,
        "severity": analysis["severity"],
        "score": analysis["score"],
        "issues": analysis["issues"],
        "recommendations": analysis["recommendations"],
        "tips": tips,
        "action": action,
        "metrics": analysis["metrics"]
    }


if __name__ == "__main__":
    # Test z tekstem ze screenshota
    test_text = """
Porwanie rodzicielskie to sytuacja, w której jeden z rodziców samowolnie zabiera lub zatrzymuje dziecko, mimo że drugi rodzic również posiada prawa do sprawowania opieki. Najczęściej dotyczy to przypadków, gdy oboje rodzice mają pełnię praw rodzicielskich, a mimo to jeden z nich jednostronnie decyduje o zmianie miejsca pobytu dziecka. To ważne.

W praktyce nie chodzi o klasyczne porwanie przez osobę trzecią. Sprawcą jest rodzic, który działa bez porozumienia i bez zgody drugiego rodzica, naruszając ustalony porządek prawny. Organy rozstrzygające takie sprawy nie koncentrują się na konflikcie między dorosłymi, lecz na tym, czy zachowanie jednego z nich pozostaje zgodne z dobrem dziecka i zapewnia mu stabilne warunki rozwoju.

Kluczowe znaczenie ma odróżnienie porwania rodzicielskiego od uprowadzenia dziecka w rozumieniu prawa karnego. W pierwszym przypadku sprawcą jest rodzic posiadający formalne uprawnienia, który działa jednostronnie, lecz niekoniecznie łamie przepisy karne. Taka sytuacja jest najczęściej oceniana na gruncie prawa rodzinnego.

Inaczej wygląda to przy uprowadzeniu, o którym mowa w art. 211 kodeksu karnego. Do odpowiedzialności karnej może dojść wtedy, gdy osoba nieuprawniona albo rodzic pozbawiony lub ograniczony we władzy zatrzymuje dziecko wbrew orzeczeniu organu sądowego. W orzecznictwie podkreśla się, że decydujące znaczenie ma naruszenie obowiązującego rozstrzygnięcia oraz faktyczne pozbawienie drugiego rodzica możliwości wykonywania jego praw. Sprawdź to.

Odpowiedź na to pytanie nie jest jednoznaczna. Sam fakt, że rodzic zabiera dziecko bez zgody drugiego rodzica, nie zawsze oznacza popełnienie przestępstwa. W polskim systemie prawnym kluczowe jest to, czy doszło do naruszenia konkretnego orzeczenia lub czy władza rodzicielska została wcześniej ograniczona.

Jeżeli jednak porwanie prowadzi do trwałego zerwania relacji, ukrywania miejsca pobytu albo odbywa się wbrew wiążącemu rozstrzygnięciu, może zostać uznane za działanie bezprawne. W takich sprawach wymiar sprawiedliwości analizuje okoliczności indywidualnie, biorąc pod uwagę wpływ zdarzenia na dziecko oraz to, czy drugi rodzic został realnie pozbawiony kontaktu. To ważne.
"""
    
    result = validate_humanization_quality(test_text)
    
    print("=== FAKE HUMANIZATION DETECTOR ===\n")
    print(f"Passed: {result['passed']}")
    print(f"Severity: {result['severity']}")
    print(f"Score: {result['score']}")
    print(f"Action: {result['action']}")
    print(f"\nIssues:")
    for issue in result['issues']:
        print(f"  - {issue}")
    print(f"\nMetrics: {result['metrics']}")
    print(f"\nTips:")
    for tip in result['tips']:
        print(f"  {tip}")
