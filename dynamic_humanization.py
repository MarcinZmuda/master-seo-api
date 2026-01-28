"""
===============================================================================
🎯 DYNAMIC HUMANIZATION MODULE v40.1
===============================================================================
Zastępuje słabe SHORT_INSERTS_LIBRARY dynamicznym systemem.

PROBLEMY ZE STARYM SYSTEMEM:
1. Tylko 9 fraz statycznych
2. Generyczne - nie pasują do tematu
3. Sztuczne - "Efekt? Natychmiastowy." brzmi jak reklama

NOWE PODEJŚCIE:
1. Dynamiczne krótkie zdania generowane na podstawie TEMATU
2. Wzorce gramatyczne zamiast gotowych fraz
3. Tematyczne biblioteki (prawo, medycyna, tech, etc.)

ZMIANY v40.1:
- Integracja z synonym_service.py (plWordNet + Firestore cache + LLM fallback)
- CONTEXTUAL_SYNONYMS jako pierwsza warstwa, synonym_service jako fallback
- Wsparcie dla get_synonyms_batch() dla wielu słów

Autor: BRAJEN SEO Master API v40.1
===============================================================================
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# ============================================================================
# 🆕 v40.1: INTEGRACJA Z SYNONYM_SERVICE
# ============================================================================

SYNONYM_SERVICE_AVAILABLE = False

try:
    from synonym_service import (
        get_synonyms as _get_synonyms_external,
        get_synonyms_batch as _get_synonyms_batch_external,
        suggest_synonym_for_repetition
    )
    SYNONYM_SERVICE_AVAILABLE = True
    print("[DYNAMIC_HUMANIZATION] ✅ synonym_service loaded (plWordNet + cache)")
except ImportError as e:
    print(f"[DYNAMIC_HUMANIZATION] ⚠️ synonym_service not available: {e}")
    print("[DYNAMIC_HUMANIZATION] ℹ️ Using local CONTEXTUAL_SYNONYMS only")
    
    # Fallback funkcje
    def _get_synonyms_external(word: str, context: str = "", use_cache: bool = True) -> Dict:
        return {"word": word, "synonyms": [], "source": "none", "count": 0}
    
    def _get_synonyms_batch_external(words: List[str], context: str = "") -> Dict[str, List[str]]:
        return {}
    
    def suggest_synonym_for_repetition(word: str, count: int, context: str = "") -> Dict:
        return {"word": word, "suggestions": [], "source": "none"}


# ============================================================================
# WZORCE KRÓTKICH ZDAŃ (3-8 SŁÓW) - UNIWERSALNE
# ============================================================================

SHORT_SENTENCE_PATTERNS = {
    # Wzorce z placeholderem {TEMAT}
    "statement": [
        "To {kluczowe}.",
        "{Podmiot} decyduje.",
        "Procedura trwa.",
        "To wymaga {czego}.",
        "Prawo to reguluje.",
        "Warto to wiedzieć.",
    ],
    "question": [
        "Co dalej?",
        "A jak to wygląda?",
        "Dlaczego to ważne?",
        "Kiedy to następuje?",
        "Jak to działa?",
    ],
    "transition": [
        "Ale uwaga.",
        "Jest wyjątek.",
        "To nie koniec.",
        "Idźmy dalej.",
        "Wróćmy do tematu.",
    ],
    "emphasis": [
        "To kluczowe.",
        "Zapamiętaj to.",
        "Ważna uwaga.",
        "Kluczowy punkt.",
    ]
}


# ============================================================================
# BIBLIOTEKI TEMATYCZNE - KRÓTKIE ZDANIA DOPASOWANE DO TEMATU
# ============================================================================

TOPIC_SHORT_SENTENCES = {
    # PRAWO / LEGAL
    "prawo": {
        "patterns": [
            "Sąd orzeka.",
            "Prawo to reguluje.",
            "Ustawa wymaga.",
            "Termin mija.",
            "Dowody decydują.",
            "Procedura trwa.",
            "To wymaga dowodów.",
            "Apelacja możliwa.",
            "Koszty rosną.",
            "Wyrok zapadł.",
        ],
        "keywords": ["sąd", "ustawa", "kodeks", "prawo", "wyrok", "pozew", 
                     "ubezwłasnowolnienie", "kuratela", "opiekun", "prawny"]
    },
    
    # MEDYCYNA / ZDROWIE
    "medycyna": {
        "patterns": [
            "Lekarz decyduje.",
            "Badanie wykaże.",
            "Objawy mogą się różnić.",
            "To wymaga diagnostyki.",
            "Leczenie trwa.",
            "Rokowania dobre.",
            "Konsultacja konieczna.",
            "Efekty widoczne.",
        ],
        "keywords": ["lekarz", "choroba", "leczenie", "diagnoza", "objawy",
                     "terapia", "pacjent", "zdrowie", "psychiczny", "psychiatra"]
    },
    
    # FINANSE
    "finanse": {
        "patterns": [
            "Koszty rosną.",
            "Podatek obowiązuje.",
            "Termin płatności.",
            "Opłaty stałe.",
            "Budżet ograniczony.",
            "Zwrot możliwy.",
        ],
        "keywords": ["podatek", "opłata", "koszt", "budżet", "finanse", 
                     "pieniądze", "kredyt", "rata"]
    },
    
    # TECHNOLOGIA
    "technologia": {
        "patterns": [
            "System działa.",
            "Aktualizacja konieczna.",
            "Dane bezpieczne.",
            "Proces automatyczny.",
            "Integracja prosta.",
        ],
        "keywords": ["system", "aplikacja", "software", "kod", "program",
                     "technologia", "digital", "online"]
    },
    
    # EDUKACJA / DZIECI
    "edukacja": {
        "patterns": [
            "Dziecko się uczy.",
            "Postępy widoczne.",
            "Ćwiczenia pomagają.",
            "Efekty przyjdą.",
            "Cierpliwość kluczowa.",
        ],
        "keywords": ["dziecko", "nauka", "szkoła", "rozwój", "edukacja",
                     "terapia", "ćwiczenia", "przedszkole"]
    },
    
    # UNIWERSALNE (fallback)
    "universal": {
        "patterns": [
            "To ważne.",
            "Warto wiedzieć.",
            "Sprawdź to.",
            "Pamiętaj.",
            "Uwaga na to.",
            "To istotne.",
            "Czas na decyzję.",
        ],
        "keywords": []
    }
}


# ============================================================================
# GŁÓWNA FUNKCJA - GENEROWANIE KRÓTKICH ZDAŃ
# ============================================================================

def detect_topic_domain(main_keyword: str, h2_titles: List[str] = None) -> str:
    """
    Wykrywa domenę tematyczną na podstawie słów kluczowych.
    
    Returns:
        Nazwa domeny: "prawo", "medycyna", "finanse", "technologia", "edukacja", "universal"
    """
    text_to_check = main_keyword.lower()
    if h2_titles:
        text_to_check += " " + " ".join(h2_titles).lower()
    
    # Sprawdź każdą domenę
    domain_scores = {}
    for domain, config in TOPIC_SHORT_SENTENCES.items():
        if domain == "universal":
            continue
        score = 0
        for keyword in config["keywords"]:
            if keyword in text_to_check:
                score += 1
        domain_scores[domain] = score
    
    # Zwróć domenę z najwyższym score (lub universal)
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[best_domain] > 0:
            return best_domain
    
    return "universal"


def get_dynamic_short_sentences(
    main_keyword: str,
    h2_titles: List[str] = None,
    count: int = 8,
    include_questions: bool = True
) -> Dict[str, any]:
    """
    Generuje dynamiczne krótkie zdania dopasowane do tematu.
    
    Args:
        main_keyword: Główna fraza kluczowa
        h2_titles: Lista tytułów H2 (opcjonalnie)
        count: Ile zdań zwrócić
        include_questions: Czy dołączyć pytania retoryczne
        
    Returns:
        Dict z:
        - domain: wykryta domena
        - sentences: lista krótkich zdań
        - patterns: wzorce do użycia
        - instruction: instrukcja dla GPT
    """
    # Wykryj domenę
    domain = detect_topic_domain(main_keyword, h2_titles)
    
    # Pobierz zdania z domeny
    domain_sentences = TOPIC_SHORT_SENTENCES.get(domain, {}).get("patterns", [])
    universal_sentences = TOPIC_SHORT_SENTENCES["universal"]["patterns"]
    
    # Połącz (priorytet dla domenowych)
    all_sentences = domain_sentences.copy()
    
    # Dodaj pytania jeśli włączone
    if include_questions:
        all_sentences.extend(SHORT_SENTENCE_PATTERNS["question"])
    
    # Dodaj tranzycje
    all_sentences.extend(SHORT_SENTENCE_PATTERNS["transition"][:3])
    
    # Uzupełnij uniwersalnymi jeśli za mało
    if len(all_sentences) < count:
        all_sentences.extend(universal_sentences)
    
    # Ogranicz do żądanej liczby
    selected_sentences = all_sentences[:count]
    
    return {
        "domain": domain,
        "sentences": selected_sentences,
        "patterns": SHORT_SENTENCE_PATTERNS,
        "instruction": f"""
🎯 KRÓTKIE ZDANIA ({domain.upper()}) - użyj 2-4 w batchu:

PRZYKŁADY:
{chr(10).join(f"• {s}" for s in selected_sentences[:6])}

ZASADY:
1. Wstaw po długim zdaniu (>25 słów)
2. Używaj przed zmianą tematu
3. NIE POWTARZAJ tych samych fraz!
4. Możesz tworzyć WŁASNE krótkie zdania (3-8 słów)
"""
    }


# ============================================================================
# SYNONIMY DYNAMICZNE - zamiast słabego SYNONYM_MAP
# ============================================================================

# Synonimy kontekstowe - używane gdy fraza jest nadużywana
CONTEXTUAL_SYNONYMS = {
    # Czasowniki - najczęściej powtarzane
    "można": ["da się", "istnieje możliwość", "jest opcja"],
    "należy": ["trzeba", "wymaga się", "konieczne jest"],
    "wymaga": ["potrzebuje", "niezbędne jest", "konieczne"],
    "pozwala": ["umożliwia", "daje możliwość", "otwiera drogę do"],
    "dotyczy": ["odnosi się do", "obejmuje", "tyczy się"],
    "stanowi": ["jest", "reprezentuje", "tworzy"],
    
    # Przymiotniki - łatwe do nadużycia
    "ważny": ["istotny", "znaczący", "kluczowy", "zasadniczy"],
    "dobry": ["skuteczny", "wartościowy", "odpowiedni", "właściwy"],
    "główny": ["podstawowy", "kluczowy", "centralny", "nadrzędny"],
    "odpowiedni": ["właściwy", "stosowny", "adekwatny"],
    
    # Rzeczowniki - kontekstowe
    "osoba": ["człowiek", "jednostka", "indywiduum"],
    "sprawa": ["kwestia", "zagadnienie", "przypadek"],
    "sposób": ["metoda", "forma", "droga"],
    "proces": ["procedura", "przebieg", "tok"],
    "warunek": ["wymóg", "kryterium", "przesłanka"],
    
    # Frazy do zamiany
    "w przypadku": ["gdy", "jeśli", "kiedy"],
    "w celu": ["aby", "żeby", "dla"],
    "ze względu na": ["z powodu", "przez", "wskutek"],
    "w kontekście": ["przy", "podczas", "w ramach"],
}


def get_synonyms_for_word(word: str, context: str = "") -> List[str]:
    """
    Zwraca synonimy dla słowa.
    
    v40.1: Hierarchia źródeł:
    1. CONTEXTUAL_SYNONYMS (lokalna mapa - najszybsze)
    2. synonym_service (plWordNet API + Firestore cache + LLM fallback)
    
    Args:
        word: Słowo do znalezienia synonimów
        context: Opcjonalny kontekst (np. "artykuł prawniczy")
        
    Returns:
        Lista synonimów (max 5)
    """
    word_lower = word.lower().strip()
    
    # 1. NAJPIERW: lokalna mapa CONTEXTUAL_SYNONYMS (najszybsze)
    local_synonyms = CONTEXTUAL_SYNONYMS.get(word_lower, [])
    if local_synonyms:
        return local_synonyms[:5]
    
    # 2. FALLBACK: synonym_service (plWordNet + cache + LLM)
    if SYNONYM_SERVICE_AVAILABLE:
        try:
            result = _get_synonyms_external(word_lower, context=context, use_cache=True)
            external_synonyms = result.get("synonyms", [])
            if external_synonyms:
                source = result.get("source", "unknown")
                print(f"[DYNAMIC_HUMANIZATION] 📚 Synonyms for '{word}' from {source}: {external_synonyms[:3]}")
                return external_synonyms[:5]
        except Exception as e:
            print(f"[DYNAMIC_HUMANIZATION] ⚠️ synonym_service error for '{word}': {e}")
    
    return []


def get_synonyms_batch(words: List[str], context: str = "") -> Dict[str, List[str]]:
    """
    🆕 v40.1: Pobiera synonimy dla wielu słów naraz.
    
    Optymalizacja - jedno zapytanie zamiast wielu.
    
    Args:
        words: Lista słów
        context: Kontekst artykułu
        
    Returns:
        Dict {słowo: [synonimy]}
    """
    result = {}
    words_to_fetch_external = []
    
    # 1. Sprawdź lokalną mapę
    for word in words:
        word_lower = word.lower().strip()
        local = CONTEXTUAL_SYNONYMS.get(word_lower, [])
        if local:
            result[word] = local[:5]
        else:
            words_to_fetch_external.append(word)
    
    # 2. Pobierz brakujące z synonym_service
    if words_to_fetch_external and SYNONYM_SERVICE_AVAILABLE:
        try:
            external_results = _get_synonyms_batch_external(words_to_fetch_external, context)
            for word, synonyms in external_results.items():
                if synonyms:
                    result[word] = synonyms[:5]
        except Exception as e:
            print(f"[DYNAMIC_HUMANIZATION] ⚠️ Batch synonym fetch error: {e}")
    
    return result


def get_synonym_instructions(overused_words: List[str] = None, context: str = "") -> Dict[str, any]:
    """
    Generuje instrukcje synonimów dla GPT.
    
    v40.1: Używa batch fetch dla wydajności + kontekst dla lepszych wyników.
    
    Args:
        overused_words: Lista słów które są nadużywane w artykule
        context: Kontekst artykułu (np. "prawo", "medycyna")
        
    Returns:
        Dict z instrukcjami i mapą synonimów
    """
    # Jeśli podano nadużywane słowa, priorytetyzuj je
    if overused_words:
        # v40.1: Użyj batch fetch dla wydajności
        all_synonyms = get_synonyms_batch(overused_words, context=context)
        
        priority_synonyms = {}
        for word in overused_words:
            syns = all_synonyms.get(word, [])
            if not syns:
                # Fallback do pojedynczego zapytania
                syns = get_synonyms_for_word(word, context=context)
            if syns:
                priority_synonyms[word] = syns[:3]
        
        if priority_synonyms:
            # Informacja o źródle
            source_info = "plWordNet + cache" if SYNONYM_SERVICE_AVAILABLE else "local"
            
            return {
                "priority": "HIGH",
                "instruction": "⚠️ TE SŁOWA SĄ NADUŻYWANE - użyj synonimów:",
                "synonyms": priority_synonyms,
                "warning": "Nie powtarzaj tego samego słowa >3x w batchu!",
                "source": source_info
            }
    
    # Domyślne - ogólne wskazówki
    return {
        "priority": "NORMAL",
        "instruction": "Unikaj powtórzeń - używaj synonimów:",
        "synonyms": {
            "można/należy": ["trzeba", "warto", "da się"],
            "ważny/istotny": ["kluczowy", "znaczący", "zasadniczy"],
            "w przypadku": ["gdy", "jeśli", "kiedy"],
        },
        "tip": "Sprawdź czy nie powtarzasz słów >3x",
        "source": "defaults"
    }


# ============================================================================
# BURSTINESS - SPRAWDZANIE I INSTRUKCJE
# ============================================================================

@dataclass
class BurstinessMetrics:
    """Metryki burstiness (zróżnicowania długości zdań)."""
    cv: float  # Współczynnik zmienności (target > 0.40)
    short_pct: float  # % krótkich zdań (3-8 słów) - target 20-25%
    medium_pct: float  # % średnich (10-18 słów) - target 50-60%
    long_pct: float  # % długich (22-35 słów) - target 15-25%
    ai_pattern_pct: float  # % zdań 15-22 słów (AI pattern) - target <30%
    is_healthy: bool
    issues: List[str]


def analyze_burstiness(text: str) -> BurstinessMetrics:
    """
    Analizuje burstiness tekstu.
    """
    # Podziel na zdania
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return BurstinessMetrics(
            cv=0.0, short_pct=0, medium_pct=0, long_pct=0,
            ai_pattern_pct=0, is_healthy=False, 
            issues=["Za mało zdań do analizy"]
        )
    
    # Policz słowa w każdym zdaniu
    lengths = [len(s.split()) for s in sentences]
    
    # Oblicz metryki
    import statistics
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
    cv = std_len / mean_len if mean_len > 0 else 0
    
    total = len(lengths)
    short_count = sum(1 for l in lengths if 3 <= l <= 8)
    medium_count = sum(1 for l in lengths if 10 <= l <= 18)
    long_count = sum(1 for l in lengths if 22 <= l <= 35)
    ai_pattern_count = sum(1 for l in lengths if 15 <= l <= 22)
    
    short_pct = (short_count / total) * 100
    medium_pct = (medium_count / total) * 100
    long_pct = (long_count / total) * 100
    ai_pattern_pct = (ai_pattern_count / total) * 100
    
    # Sprawdź problemy
    issues = []
    if cv < 0.35:
        issues.append(f"CV={cv:.2f} za niskie (target >0.40) - zdania za podobne!")
    if short_pct < 15:
        issues.append(f"Za mało krótkich zdań: {short_pct:.0f}% (target 20-25%)")
    if ai_pattern_pct > 40:
        issues.append(f"Za dużo zdań 15-22 słów: {ai_pattern_pct:.0f}% (AI pattern!)")
    
    is_healthy = len(issues) == 0
    
    return BurstinessMetrics(
        cv=round(cv, 3),
        short_pct=round(short_pct, 1),
        medium_pct=round(medium_pct, 1),
        long_pct=round(long_pct, 1),
        ai_pattern_pct=round(ai_pattern_pct, 1),
        is_healthy=is_healthy,
        issues=issues
    )


def get_burstiness_instructions(previous_batch_text: str = None) -> Dict[str, any]:
    """
    Generuje instrukcje burstiness dla GPT.
    
    Args:
        previous_batch_text: Tekst poprzedniego batcha (opcjonalnie, do analizy)
        
    Returns:
        Dict z instrukcjami i metrykami
    """
    base_instruction = {
        "critical": True,
        "what": "BURSTINESS = zróżnicowanie długości zdań",
        "why": "Monotonne zdania 15-20 słów = wykrycie AI!",
        "target_cv": ">0.40",
        "distribution": {
            "short_3_8_words": "20-25%",
            "medium_10_18_words": "50-60%",
            "long_22_35_words": "15-25%"
        },
        "example_sequence": "5, 18, 8, 25, 12, 6, 30, 14 słów",
        "avoid": "❌ NIE PISZ wszystkich zdań 15-22 słów!"
    }
    
    # Jeśli mamy poprzedni batch, analizuj go
    if previous_batch_text:
        metrics = analyze_burstiness(previous_batch_text)
        
        if not metrics.is_healthy:
            base_instruction["previous_batch_analysis"] = {
                "cv": metrics.cv,
                "short_pct": metrics.short_pct,
                "issues": metrics.issues,
                "fix_instruction": "⚠️ Poprzedni batch ma problemy z burstiness - POPRAW!"
            }
    
    return base_instruction


# ============================================================================
# GŁÓWNA FUNKCJA - PEŁNE INSTRUKCJE HUMANIZACJI
# ============================================================================

def get_humanization_instructions(
    main_keyword: str,
    h2_titles: List[str] = None,
    previous_batch_text: str = None,
    overused_words: List[str] = None
) -> Dict[str, any]:
    """
    Generuje kompletne instrukcje humanizacji dla GPT.
    
    Args:
        main_keyword: Główna fraza kluczowa
        h2_titles: Lista H2 (opcjonalnie)
        previous_batch_text: Poprzedni batch do analizy (opcjonalnie)
        overused_words: Nadużywane słowa (opcjonalnie)
        
    Returns:
        Dict z pełnymi instrukcjami humanizacji
    """
    return {
        "version": "v40.0",
        
        # Krótkie zdania
        "short_sentences": get_dynamic_short_sentences(
            main_keyword, h2_titles, count=8
        ),
        
        # Burstiness
        "burstiness": get_burstiness_instructions(previous_batch_text),
        
        # Synonimy
        "synonyms": get_synonym_instructions(overused_words),
        
        # AI patterns do unikania
        "avoid_ai_patterns": {
            "instruction": "❌ UNIKAJ tych fraz (typowe AI):",
            "patterns": {
                "warto podkreślić": "→ usuń lub 'Zwróć uwagę:'",
                "należy pamiętać": "→ 'Pamiętaj:' lub usuń",
                "w kontekście": "→ 'przy', 'podczas'",
                "istotne jest": "→ 'Ważne:'",
                "kluczowym aspektem jest": "→ usuń całość",
                "warto zauważyć": "→ usuń",
                "nie bez znaczenia jest": "→ 'Ważne:'",
            }
        },
        
        # Styl
        "style_tips": {
            "instruction": "Pisz jak ekspert rozmawiający ze znajomym",
            "tips": [
                "Używaj pytań retorycznych",
                "Nie każde zdanie musi być 'mądre'",
                "Dodawaj krótkie reakcje (To ważne. Uwaga na to.)",
                "Mieszaj zdania proste ze złożonymi"
            ]
        }
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Główne funkcje
    'get_dynamic_short_sentences',
    'get_synonym_instructions',
    'get_burstiness_instructions',
    'get_humanization_instructions',
    
    # Funkcje pomocnicze
    'detect_topic_domain',
    'analyze_burstiness',
    'get_synonyms_for_word',
    'get_synonyms_batch',  # 🆕 v40.1
    
    # Klasy
    'BurstinessMetrics',
    
    # Stałe
    'CONTEXTUAL_SYNONYMS',
    'TOPIC_SHORT_SENTENCES',
    'SHORT_SENTENCE_PATTERNS',
    
    # Status integracji
    'SYNONYM_SERVICE_AVAILABLE',  # 🆕 v40.1
]


# ============================================================================
# TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    # Test detekcji domeny
    print("=" * 60)
    print("TEST: Detekcja domeny")
    print("=" * 60)
    
    test_cases = [
        ("ubezwłasnowolnienie częściowe", ["Przesłanki prawne", "Procedura sądowa"]),
        ("terapia integracji sensorycznej", ["Ćwiczenia dla dzieci"]),
        ("rozliczenie podatku PIT", ["Ulgi podatkowe"]),
        ("programowanie w Python", ["Podstawy kodu"]),
    ]
    
    for main_kw, h2s in test_cases:
        domain = detect_topic_domain(main_kw, h2s)
        print(f"'{main_kw}' → {domain}")
    
    print("\n" + "=" * 60)
    print("TEST: Krótkie zdania dla tematu prawnego")
    print("=" * 60)
    
    result = get_dynamic_short_sentences(
        "ubezwłasnowolnienie całkowite",
        ["Procedura sądowa", "Skutki prawne"]
    )
    print(f"Domena: {result['domain']}")
    print("Zdania:")
    for s in result['sentences']:
        print(f"  • {s}")
    
    print("\n" + "=" * 60)
    print("TEST: Analiza burstiness")
    print("=" * 60)
    
    test_text = """
    Ubezwłasnowolnienie to poważna decyzja. Sąd orzeka. Wymaga to odpowiednich 
    przesłanek prawnych określonych w kodeksie cywilnym. To ważne. Procedura 
    jest skomplikowana i wymaga udziału biegłych psychiatrów oraz psychologów 
    w celu oceny stanu zdrowia osoby, która ma być ubezwłasnowolniona. 
    Termin mija. Należy pamiętać o terminach.
    """
    
    metrics = analyze_burstiness(test_text)
    print(f"CV: {metrics.cv}")
    print(f"Krótkie: {metrics.short_pct}%")
    print(f"Średnie: {metrics.medium_pct}%")
    print(f"Długie: {metrics.long_pct}%")
    print(f"AI pattern: {metrics.ai_pattern_pct}%")
    print(f"Zdrowe: {metrics.is_healthy}")
    if metrics.issues:
        print(f"Problemy: {metrics.issues}")
    
    # 🆕 v40.1: Test integracji z synonym_service
    print("\n" + "=" * 60)
    print("TEST: Synonimy (v40.1 - z integracją synonym_service)")
    print("=" * 60)
    print(f"SYNONYM_SERVICE_AVAILABLE: {SYNONYM_SERVICE_AVAILABLE}")
    
    test_words = ["można", "ważny", "procedura", "ubezwłasnowolnienie"]
    for word in test_words:
        syns = get_synonyms_for_word(word, context="prawo")
        source = "local" if word in CONTEXTUAL_SYNONYMS else ("external" if syns else "none")
        print(f"'{word}' → {syns[:3] if syns else '(brak)'} [source: {source}]")
    
    print("\n" + "=" * 60)
    print("TEST: Batch synonym fetch")
    print("=" * 60)
    
    batch_result = get_synonyms_batch(["można", "należy", "sąd"], context="prawo")
    for word, syns in batch_result.items():
        print(f"'{word}' → {syns[:3]}")

