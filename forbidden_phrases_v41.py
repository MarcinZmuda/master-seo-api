"""
===============================================================================
🚫 FORBIDDEN PHRASES v41.0 - Rozszerzona lista polskich markerów AI
===============================================================================

Rozszerzenie istniejącej listy FORBIDDEN_PATTERNS z ai_detection_metrics.py.

ŹRÓDŁA MARKERÓW (tylko zweryfikowane):
1. Obecna lista BRAJEN v40.2 (20 wzorców) - zachowane
2. Analiza wyjścia ChatGPT/Claude w języku polskim - obserwacje empiryczne
3. Wzorce powtarzalne w masowo generowanych treściach SEO

ZASADY:
- Każdy wzorzec musi być MIERZALNY (regex match)
- Każdy wzorzec musi mieć ZAMIENNIK lub [USUŃ]
- Brak spekulacji - tylko wzorce zaobserwowane w praktyce

===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple
from enum import Enum


class Severity(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# ROZSZERZONA LISTA FORBIDDEN_PATTERNS v41
# ============================================================================

FORBIDDEN_PATTERNS_V41 = [
    # ========================================================================
    # ISTNIEJĄCE Z BRAJEN v40.2 (zachowane bez zmian)
    # ========================================================================
    (r'\bwarto wiedzieć\b', "warto wiedzieć"),
    (r'\bnależy pamiętać\b', "należy pamiętać"),
    (r'\bnależy podkreślić\b', "należy podkreślić"),
    (r'\bkluczowy aspekt\b', "kluczowy aspekt"),
    (r'\bkompleksowe rozwiązanie\b', "kompleksowe rozwiązanie"),
    (r'\bholistyczne podejście\b', "holistyczne podejście"),
    (r'\bw dzisiejszych czasach\b', "w dzisiejszych czasach"),
    (r'\bnie ulega wątpliwości\b', "nie ulega wątpliwości"),
    (r'\bcoraz więcej osób\b', "coraz więcej osób"),
    (r'\bw tym artykule\b', "w tym artykule"),
    (r'\bpodsumowując\b', "podsumowując"),
    (r'\bjak już wspomniano\b', "jak już wspomniano"),
    (r'\bkażdy z nas\b', "każdy z nas"),
    (r'\bnie jest tajemnicą\b', "nie jest tajemnicą"),
    (r'\bpowszechnie wiadomo\b', "powszechnie wiadomo"),
    (r'\btrudno przecenić\b', "trudno przecenić"),
    (r'\bw erze\s+\w+\b', "w erze..."),
    (r'\bw dobie\s+\w+\b', "w dobie..."),
    (r'\bw obliczu\b', "w obliczu"),
    (r'\bna przestrzeni lat\b', "na przestrzeni lat"),
    
    # ========================================================================
    # NOWE v41 - Zaobserwowane markery ChatGPT (polski output)
    # ========================================================================
    (r'\bna podstawie dostępnych danych\b', "na podstawie dostępnych danych"),
    (r'\bogólnie rzecz biorąc\b', "ogólnie rzecz biorąc"),
    (r'\bto prowadzi nas do wniosku\b', "to prowadzi nas do wniosku"),
    (r'\bco prowadzi nas do\b', "co prowadzi nas do"),
    (r'\brozważmy\b', "rozważmy"),
    (r'\bzagłębmy się\b', "zagłębmy się"),
    (r'\bprzejdźmy do\b', "przejdźmy do"),
    (r'\bw świetle powyższego\b', "w świetle powyższego"),
    (r'\bw kontekście powyższego\b', "w kontekście powyższego"),
    (r'\bpodsumowując powyższe\b', "podsumowując powyższe"),
    (r'\bwarto również wspomnieć\b', "warto również wspomnieć"),
    (r'\bnie sposób pominąć\b', "nie sposób pominąć"),
    (r'\bjednak warto zauważyć\b', "jednak warto zauważyć"),
    (r'\bniezwykle istotne jest\b', "niezwykle istotne jest"),
    (r'\bwarto mieć na uwadze\b', "warto mieć na uwadze"),
    (r'\bw pierwszej kolejności\b', "w pierwszej kolejności"),
    (r'\bna samym początku\b', "na samym początku"),
    (r'\bna koniec warto\b', "na koniec warto"),
    (r'\bna zakończenie\b', "na zakończenie"),
    
    # ========================================================================
    # NOWE v41 - Wzorce "filler phrases" (puste słowa)
    # ========================================================================
    (r'\bjest to niezwykle\b', "jest to niezwykle"),
    (r'\bz całą pewnością\b', "z całą pewnością"),
    (r'\bbez wątpienia\b', "bez wątpienia"),
    (r'\bbezsprzecznie\b', "bezsprzecznie"),
    (r'\bniepodważalnie\b', "niepodważalnie"),
    (r'\bbezdyskusyjnie\b', "bezdyskusyjnie"),
    
    # ========================================================================
    # NOWE v41 - Meta-komentarze (AI mówi o sobie/tekście)
    # ========================================================================
    (r'\bw niniejszym artykule\b', "w niniejszym artykule"),
    (r'\bw poniższym tekście\b', "w poniższym tekście"),
    (r'\bponiżej przedstawiamy\b', "poniżej przedstawiamy"),
    (r'\bomówimy\s+\w+\s+aspekty\b', "omówimy ... aspekty"),
    (r'\bprzedstawimy\s+\w+\s+kwestie\b', "przedstawimy ... kwestie"),
    
    # ========================================================================
    # NOWE v41 - Nadmierne uogólnienia
    # ========================================================================
    (r'\bw dzisiejszym świecie\b', "w dzisiejszym świecie"),
    (r'\bw obecnych czasach\b', "w obecnych czasach"),
    (r'\bw nowoczesnym społeczeństwie\b', "w nowoczesnym społeczeństwie"),
    (r'\bw dynamicznie zmieniającym się\b', "w dynamicznie zmieniającym się"),
    
    # ========================================================================
    # v50 - Szablonowe pytania retoryczne (nadużywane przez AI)
    # ========================================================================
    (r'\bjak to wygląda w praktyce\b', "jak to wygląda w praktyce"),
    (r'\bco to (dokładnie )?oznacza\b', "co to oznacza"),
    (r'\bczy zawsze tak jest\b', "czy zawsze tak jest"),
    (r'\bczy to takie proste\b', "czy to takie proste"),
    (r'\bjakie są (zatem |więc )?wyjątki\b', "jakie są wyjątki"),
    (r'\bale czy to wystarczy\b', "ale czy to wystarczy"),
    (r'\bi tu zaczyna się\b', "i tu zaczyna się"),
    (r'\bczas na konkrety\b', "czas na konkrety"),
]

# ============================================================================
# ROZSZERZONA LISTA FORBIDDEN_WORDS v41
# ============================================================================

FORBIDDEN_WORDS_V41 = [
    # ISTNIEJĄCE Z BRAJEN v40.2
    "kluczowy", "kompleksowy", "innowacyjny", "holistyczny", 
    "transformacyjny", "fundamentalny", "niewątpliwie", "wieloaspektowy",
    "przełomowy", "bezsprzecznie", "rewolucyjny", "optymalizować",
    
    # NOWE v41 - często nadużywane przez AI
    "bezprecedensowy",
    "synergiczny",
    "paradygmat",
    "transparentny",    # kalka z angielskiego, w polskim lepiej: przejrzysty
    "implikacje",       # AI nadużywa, lepiej: skutki, konsekwencje
    "implementować",    # AI nadużywa, lepiej: wdrożyć, wprowadzić
    "ewaluować",        # AI nadużywa, lepiej: oceniać, sprawdzać
    "dedykowany",       # AI nadużywa, lepiej: przeznaczony, specjalny
    "generować",        # AI nadużywa w kontekście nie-technicznym
    "optymalizacja",    # rzeczownik od optymalizować
    "wielopłaszczyznowy",
    "multidyscyplinarny",
]

# ============================================================================
# ROZSZERZONE REPLACEMENTS v41
# ============================================================================

FORBIDDEN_REPLACEMENTS_V41 = {
    # ISTNIEJĄCE
    "coraz więcej osób": "wiele osób",
    "w dzisiejszych czasach": "[USUŃ - niepotrzebne]",
    "warto wiedzieć": "[USUŃ - zacznij od konkretu]",
    "należy podkreślić": "[USUŃ - po prostu podkreśl]",
    "podsumowując": "[zamień na konkretne zakończenie]",
    "w tym artykule": "[NIGDY - czytelnik wie że czyta artykuł]",
    "kluczowy": "istotny / ważny / główny",
    "kompleksowy": "pełny / całościowy / obszerny",
    "innowacyjny": "nowoczesny / nowatorski / nowy",
    "holistyczny": "całościowy / pełny",
    
    # NOWE v41
    "na podstawie dostępnych danych": "[USUŃ lub podaj konkretne źródło]",
    "ogólnie rzecz biorąc": "[USUŃ - bądź konkretny]",
    "rozważmy": "[USUŃ - po prostu rozważ]",
    "zagłębmy się": "[USUŃ - zacznij od tematu]",
    "przejdźmy do": "[USUŃ - po prostu przejdź]",
    "bezprecedensowy": "niespotykany / wyjątkowy / niezwykły",
    "transparentny": "przejrzysty / jawny / otwarty",
    "implikacje": "skutki / konsekwencje / następstwa",
    "implementować": "wdrożyć / wprowadzić / zastosować",
    "dedykowany": "przeznaczony / specjalny / przygotowany dla",
    "w dzisiejszym świecie": "[USUŃ - oczywiste]",
    "w niniejszym artykule": "[USUŃ - czytelnik wie]",
    "synergiczny": "współdziałający / wzajemnie wspierający się",
    "paradygmat": "model / wzorzec / schemat",
    "wieloaspektowy": "różnorodny / złożony",
    "fundamentalny": "podstawowy / zasadniczy",
    "transformacyjny": "zmieniający / przekształcający",
}


# ============================================================================
# GŁÓWNA FUNKCJA CHECK (kompatybilna z istniejącym API)
# ============================================================================

def check_forbidden_phrases_v41(text: str) -> Dict[str, Any]:
    """
    Sprawdza zakazane frazy i słowa (rozszerzona wersja v41).
    
    Zwraca:
        Dict z kluczami:
        - status: OK/CRITICAL
        - forbidden_found: lista znalezionych
        - phrases: znalezione frazy
        - words: znalezione słowa
        - count: liczba znalezionych
        - message: komunikat
        - replacements: sugestie zamienników
        - should_block: bool (czy blokować batch)
    """
    text_lower = text.lower()
    found_phrases = []
    found_words = []
    replacements = []
    
    # Sprawdź frazy
    for pattern, name in FORBIDDEN_PATTERNS_V41:
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_phrases.append(name)
            if name in FORBIDDEN_REPLACEMENTS_V41:
                replacements.append(f"'{name}' → {FORBIDDEN_REPLACEMENTS_V41[name]}")
            else:
                replacements.append(f"'{name}' → [znajdź alternatywę]")
    
    # Sprawdź pojedyncze słowa
    for word in FORBIDDEN_WORDS_V41:
        if re.search(rf'\b{re.escape(word)}\b', text_lower, re.IGNORECASE):
            found_words.append(word)
            if word in FORBIDDEN_REPLACEMENTS_V41:
                replacements.append(f"'{word}' → {FORBIDDEN_REPLACEMENTS_V41[word]}")
    
    all_found = found_phrases + found_words
    
    if all_found:
        status = Severity.CRITICAL
        message = f"🚫 ZAKAZANE FRAZY ({len(all_found)}×): {', '.join(all_found[:5])}"
        if len(all_found) > 5:
            message += f" ...i {len(all_found) - 5} więcej"
        should_block = True
    else:
        status = Severity.OK
        message = "✅ Brak zakazanych fraz"
        should_block = False
    
    return {
        "status": status.value,
        "forbidden_found": all_found,
        "phrases": found_phrases,
        "words": found_words,
        "count": len(all_found),
        "message": message,
        "replacements": replacements[:10],  # max 10 sugestii
        "should_block": should_block
    }


# ============================================================================
# STATYSTYKI MODUŁU
# ============================================================================

def get_forbidden_stats() -> Dict[str, int]:
    """Zwraca statystyki rozszerzenia."""
    return {
        "patterns_count": len(FORBIDDEN_PATTERNS_V41),
        "words_count": len(FORBIDDEN_WORDS_V41),
        "replacements_count": len(FORBIDDEN_REPLACEMENTS_V41),
        "total": len(FORBIDDEN_PATTERNS_V41) + len(FORBIDDEN_WORDS_V41),
        "version": "41.0"
    }


# ============================================================================
# INTEGRACJA Z ISTNIEJĄCYM KODEM
# ============================================================================

def integrate_with_ai_detection_metrics():
    """
    Instrukcja integracji z ai_detection_metrics.py:
    
    1. Dodaj import na początku pliku:
       from forbidden_phrases_v41 import (
           FORBIDDEN_PATTERNS_V41 as FORBIDDEN_PATTERNS,
           FORBIDDEN_WORDS_V41 as FORBIDDEN_WORDS,
           FORBIDDEN_REPLACEMENTS_V41 as FORBIDDEN_REPLACEMENTS,
           check_forbidden_phrases_v41 as check_forbidden_phrases
       )
    
    2. Usuń stare definicje FORBIDDEN_PATTERNS, FORBIDDEN_WORDS, FORBIDDEN_REPLACEMENTS
    
    3. Funkcja check_forbidden_phrases zostanie nadpisana nową wersją
    """
    pass


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    stats = get_forbidden_stats()
    print(f"📊 FORBIDDEN PHRASES v41 Statistics:")
    print(f"   Patterns: {stats['patterns_count']}")
    print(f"   Words: {stats['words_count']}")
    print(f"   Replacements: {stats['replacements_count']}")
    print(f"   TOTAL: {stats['total']}")
    
    # Test
    test_text = """
    W dzisiejszych czasach warto wiedzieć, że kompleksowe rozwiązania są kluczowe.
    Holistyczne podejście pozwala na transformacyjne zmiany.
    Ogólnie rzecz biorąc, implementacja jest bezprecedensowa.
    """
    
    result = check_forbidden_phrases_v41(test_text)
    print(f"\n🧪 Test result:")
    print(f"   Status: {result['status']}")
    print(f"   Found: {result['count']}")
    print(f"   Phrases: {result['phrases']}")
    print(f"   Words: {result['words']}")
    print(f"\n📝 Replacements:")
    for r in result['replacements']:
        print(f"   {r}")
