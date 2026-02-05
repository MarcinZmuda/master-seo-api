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

# 🆕 v41.0: Import rozszerzonej mapy synonimów (105 słów zamiast 25)
from contextual_synonyms_v41 import (
    CONTEXTUAL_SYNONYMS_V41,
    get_synonyms_v41,
    get_synonyms_batch_v41,
    get_stats_v41 as get_synonyms_stats_v41
)

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
# KONTEKSTOWE KRÓTKIE ZDANIA v41.0
# ============================================================================
# Zamiast statycznych zdań ("Lekarz decyduje.", "Sąd orzeka.") system
# generuje WZORCE GRAMATYCZNE + INSTRUKCJE dla GPT, żeby sam tworzył
# krótkie zdania pasujące do kontekstu aktualnej sekcji H2.
#
# DLACZEGO:
# - Statyczne zdania brzmiały sztucznie i oderwanie od tematu
# - "Rokowania dobre." wstawione losowo w akapicie o diagnostyce = cringe
# - GPT potrafi stworzyć dobre krótkie zdania, jeśli dostanie wzorce
# ============================================================================

# Wzorce gramatyczne (3-8 słów) — GPT wypełnia kontekstem z aktualnej sekcji
SHORT_SENTENCE_GRAMMAR_PATTERNS = {
    # Wzorce stwierdzające — GPT wstawia podmiot/dopełnienie z tematu sekcji
    "stwierdzenie": [
        "[Podmiot z akapitu] + orzeczenie (3-5 słów)",
        "To + przymiotnik kontekstowy (np. 'To częste.', 'To ryzykowne.')",
        "Krótkie podsumowanie ostatniego zdania (max 5 słów)",
        "Zdanie nominalne — sam rzeczownik + przymiotnik (np. 'Częsty problem.', 'Ważna różnica.')",
    ],
    # Pytania retoryczne — nawiązują do tego co jest DALEJ w akapicie
    "pytanie": [
        "Pytanie zaczynające nowy wątek (np. 'A co z dawkowaniem?')",
        "'Dlaczego/Jak/Kiedy + nawiązanie do następnego zdania'",
        "Pytanie potwierdzające (np. 'Brzmi skomplikowanie?')",
    ],
    # Tranzycje — łączą myśli
    "tranzycja": [
        "Ale/Jednak + krótka uwaga (np. 'Ale jest wyjątek.')",
        "Kontrast do poprzedniego zdania (3-6 słów)",
        "Zapowiedź zwrotu (np. 'Tu robi się ciekawie.')",
    ],
}

# Domeny tematyczne — słowa kluczowe do detekcji + KONTEKSTOWE PODPOWIEDZI
# (nie gotowe zdania, a wskazówki jakie krótkie zdania pasują do domeny)
TOPIC_DOMAIN_CONFIG = {
    "prawo": {
        "keywords": ["sąd", "ustawa", "kodeks", "prawo", "wyrok", "pozew",
                     "ubezwłasnowolnienie", "kuratela", "opiekun", "prawny",
                     "notariusz", "akt", "przepis", "roszczenie", "apelacja"],
        "context_hints": [
            "Krótkie zdania prawnicze: odniesienie do terminu, procedury lub konsekwencji",
            "Np. po opisie procedury: 'Termin jest sztywny.' / 'Tu nie ma wyjątków.'",
            "Np. po opisie ryzyka: 'Warto to sprawdzić wcześniej.'",
        ],
    },
    "medycyna": {
        "keywords": ["lekarz", "choroba", "leczenie", "diagnoza", "objawy",
                     "terapia", "pacjent", "zdrowie", "psychiczny", "psychiatra",
                     "badanie", "lek", "dawka", "zabieg", "profilaktyka"],
        "context_hints": [
            "Krótkie zdania medyczne: odniesienie do objawu, leczenia lub rokowania",
            "Np. po opisie objawów: 'Nie u każdego.' / 'Zależy od pacjenta.'",
            "Np. po opisie leczenia: 'Efekty nie są natychmiastowe.'",
        ],
    },
    "finanse": {
        "keywords": ["podatek", "opłata", "koszt", "budżet", "finanse",
                     "pieniądze", "kredyt", "rata", "faktura", "rozliczenie"],
        "context_hints": [
            "Krótkie zdania finansowe: odniesienie do kwoty, terminu lub ryzyka",
            "Np. po opisie kosztów: 'To sporo.' / 'Zależy od umowy.'",
            "Np. po opisie procedury: 'Warto policzyć wcześniej.'",
        ],
    },
    "technologia": {
        "keywords": ["system", "aplikacja", "software", "kod", "program",
                     "technologia", "digital", "online", "algorytm", "serwer"],
        "context_hints": [
            "Krótkie zdania tech: odniesienie do działania, wymagań lub ograniczeń",
            "Np. po opisie funkcji: 'Działa automatycznie.' / 'Nie zawsze.'",
            "Np. po opisie problemu: 'Łatwa poprawka.' / 'To znany problem.'",
        ],
    },
    "edukacja": {
        "keywords": ["dziecko", "nauka", "szkoła", "rozwój", "edukacja",
                     "terapia", "ćwiczenia", "przedszkole", "uczeń", "nauczyciel"],
        "context_hints": [
            "Krótkie zdania edukacyjne: odniesienie do postępów, metod lub efektów",
            "Np. po opisie metody: 'Wymaga cierpliwości.' / 'Efekty przyjdą.'",
            "Np. po opisie problemu: 'To normalne na tym etapie.'",
        ],
    },
    "universal": {
        "keywords": [],
        "context_hints": [
            "Krótkie zdania odnoszące się do treści poprzedniego lub następnego zdania",
            "Unikaj ogólników — zdanie musi wynikać z kontekstu akapitu",
        ],
    },
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
    for domain, config in TOPIC_DOMAIN_CONFIG.items():
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
    include_questions: bool = True,
    current_h2: str = None,
    batch_num: int = None,
) -> Dict[str, any]:
    """
    Generuje KONTEKSTOWE instrukcje krótkich zdań — GPT tworzy je sam
    na podstawie wzorców gramatycznych i podpowiedzi domenowych.
    
    ZMIANA v41.0: Zamiast statycznych zdań ("Lekarz decyduje.") system
    daje GPT wzorce + kontekst, żeby tworzył zdania pasujące do
    aktualnej sekcji H2.
    
    Args:
        main_keyword: Główna fraza kluczowa
        h2_titles: Lista tytułów H2 (opcjonalnie)
        count: Ile wzorców zwrócić (nieużywane, zachowane dla kompatybilności)
        include_questions: Czy dołączyć wzorce pytań retorycznych
        current_h2: Aktualny tytuł H2 (dla lepszego kontekstu)
        batch_num: Numer aktualnego batcha
        
    Returns:
        Dict z:
        - domain: wykryta domena
        - grammar_patterns: wzorce gramatyczne do zastosowania
        - context_hints: podpowiedzi domenowe
        - instruction: pełna instrukcja dla GPT
    """
    # Wykryj domenę
    domain = detect_topic_domain(main_keyword, h2_titles)
    
    # Pobierz podpowiedzi domenowe
    domain_config = TOPIC_DOMAIN_CONFIG.get(domain, TOPIC_DOMAIN_CONFIG["universal"])
    context_hints = domain_config.get("context_hints", [])
    
    # Zbierz wzorce gramatyczne
    patterns_to_use = []
    patterns_to_use.extend(SHORT_SENTENCE_GRAMMAR_PATTERNS["stwierdzenie"])
    if include_questions:
        patterns_to_use.extend(SHORT_SENTENCE_GRAMMAR_PATTERNS["pytanie"])
    patterns_to_use.extend(SHORT_SENTENCE_GRAMMAR_PATTERNS["tranzycja"])
    
    # Kontekst sekcji — jeśli znamy aktualne H2
    section_context = ""
    if current_h2:
        section_context = f"\nAktualna sekcja: \"{current_h2}\" — krótkie zdania MUSZĄ dotyczyć tego tematu."
    
    instruction = f"""✂️ KRÓTKIE ZDANIA (3-8 słów) — twórz WŁASNE, pasujące do kontekstu:

TEMAT ARTYKUŁU: {main_keyword}
DOMENA: {domain}{section_context}

ZASADY TWORZENIA (2-4 krótkie zdania na batch):
1. Każde krótkie zdanie MUSI wynikać z poprzedniego lub następnego zdania
2. NIE wstawiaj ogólników oderwanych od treści
3. Wstaw po długim zdaniu (>25 słów) jako "oddech" dla czytelnika
4. Przed zmianą wątku w akapicie

WZORCE GRAMATYCZNE (wypełnij treścią z akapitu):
• Stwierdzenie: podmiot z akapitu + krótkie orzeczenie (np. "Termin jest sztywny.", "To zależy od dawki.")
• Zdanie nominalne: rzeczownik + przymiotnik z kontekstu (np. "Częsty błąd.", "Ważna różnica.")
• Pytanie retoryczne: nawiązanie do następnego zdania (np. "A co z kosztami?", "Jak to wygląda w praktyce?")
• Kontrast/tranzycja: krótki zwrot akcji (np. "Ale jest wyjątek.", "Nie zawsze.")
• Podsumowanie: esencja ostatniego zdania w 3-5 słów

PODPOWIEDZI DLA TEJ DOMENY ({domain}):
{chr(10).join(f"• {h}" for h in context_hints)}

❌ NIE RÓB TAK (oderwane od kontekstu):
• "To ważne." (ogólnik)
• "Warto wiedzieć." (nic nie mówi)
• "Pamiętaj." (pusty rozkaz)

✅ RÓB TAK (wynika z kontekstu):
• Po akapicie o skutkach ubocznych leku: "Nie u każdego pacjenta."
• Po akapicie o terminach sądowych: "Termin jest nieprzekraczalny."
• Po opisie skomplikowanej procedury: "Brzmi skomplikowanie? Niekoniecznie."

💡 DZIELENIE DŁUGICH ZDAŃ:
Jeśli zdanie ma >25 słów, podziel je na dwa krótsze w naturalnym punkcie:
• Przed "ale", "jednak", "natomiast" → kropka, usuń spójnik, capitalize resztę
• Przy średniku → zamień na kropkę
• Przed "ponieważ", "gdyż" → przebuduj na samodzielne zdanie przyczynowe
Przykład: "Leczenie trwa kilka tygodni, ale efekty są widoczne już po pierwszym cyklu."
→ "Leczenie trwa kilka tygodni. Efekty są widoczne już po pierwszym cyklu."
"""
    
    return {
        "domain": domain,
        "grammar_patterns": patterns_to_use,
        "context_hints": context_hints,
        "instruction": instruction,
        # Zachowane dla kompatybilności wstecznej — puste, bo GPT ma tworzyć własne
        "sentences": [],
        "patterns": SHORT_SENTENCE_GRAMMAR_PATTERNS,
    }


# ============================================================================
# 🆕 v41.0: DZIELENIE DŁUGICH ZDAŃ (SENTENCE SPLITTER)
# ============================================================================
# Zamiast wstawiać sztuczne krótkie zdania, dzielimy istniejące długie
# zdania w naturalnych punktach gramatycznych polszczyzny.
#
# Efekt:
# - Burstiness rośnie organicznie (więcej krótkich zdań)
# - Treść pozostaje kontekstowa (bo pochodzi z oryginalnego zdania)
# - Czytelność się poprawia (krótsze zdania = łatwiejszy odbiór)
# ============================================================================

import re as _re

# Punkty podziału zdań — posortowane wg bezpieczeństwa (od najbezpieczniejszych)

# TIER 1: Bardzo bezpieczne — prawie zawsze dają poprawne dwa zdania
SPLIT_POINTS_TIER1 = [
    # Średnik → kropka (zawsze bezpieczne)
    (r';\s+', '. ', 'semicolon'),
    # Myślnik em-dash z spacjami — często oddziela niezależne myśli
    (r'\s+–\s+', '. ', 'em_dash'),
]

# TIER 2: Bezpieczne — spójniki współrzędne (niezależne zdania składowe)
# Po podziale spójnik jest USUWANY — kropka pełni jego funkcję
SPLIT_POINTS_TIER2 = [
    (r',\s+ale\s+', '. ', 'ale'),
    (r',\s+jednak\s+', '. ', 'jednak'),
    (r',\s+natomiast\s+', '. ', 'natomiast'),
    (r',\s+lecz\s+', '. ', 'lecz'),
    (r',\s+więc\s+', '. ', 'wiec'),
    (r',\s+dlatego\s+', '. ', 'dlatego'),
    (r',\s+zatem\s+', '. ', 'zatem'),
    (r',\s+tymczasem\s+', '. ', 'tymczasem'),
    (r',\s+z kolei\s+', '. ', 'z_kolei'),
    (r',\s+a także\s+', '. ', 'a_takze'),
    (r',\s+a jednocześnie\s+', '. ', 'a_jednoczesnie'),
]

# TIER 3: Ostrożne — spójniki przyczynowe/wynikowe
# Usuwamy spójnik i przebudowujemy początek na samodzielne zdanie
SPLIT_POINTS_TIER3 = [
    (r',\s+ponieważ\s+', '. Wynika to z tego, że ', 'poniewaz'),
    (r',\s+gdyż\s+', '. Powodem jest to, że ', 'gdyz'),
    (r',\s+bowiem\s+', '. ', 'bowiem'),
    (r',\s+przy czym\s+', '. Warto dodać, że ', 'przy_czym'),
    (r',\s+co oznacza,?\s+że\s+', '. Oznacza to, że ', 'co_oznacza'),
    (r',\s+co powoduje,?\s+że\s+', '. Skutkuje to tym, że ', 'co_powoduje'),
    (r',\s+co sprawia,?\s+że\s+', '. W rezultacie ', 'co_sprawia'),
]

# Minimalna długość każdej z dwóch części po podziale (w słowach)
MIN_HALF_WORDS = 5

# Próg długości zdania, powyżej którego próbujemy dzielić
LONG_SENTENCE_THRESHOLD = 28  # słów


@dataclass
class SplitResult:
    """Wynik podziału jednego zdania."""
    original: str
    part1: str
    part2: str
    split_type: str  # np. 'ale', 'semicolon', 'em_dash'
    tier: int  # 1, 2 lub 3


def _count_words(text: str) -> int:
    """Liczy słowa w tekście."""
    return len(text.split())


def _find_best_split(sentence: str, threshold: int = LONG_SENTENCE_THRESHOLD) -> Optional[SplitResult]:
    """
    Znajduje najlepszy punkt podziału dla długiego zdania.
    
    Strategia:
    1. Sprawdź TIER 1 (średniki, myślniki) — zawsze bezpieczne
    2. Sprawdź TIER 2 (ale, jednak, natomiast) — bezpieczne
    3. Sprawdź TIER 3 (ponieważ, gdyż) — ostrożnie
    4. Jeśli wiele opcji — wybierz tę, która daje najbardziej równy podział
    
    Returns:
        SplitResult lub None jeśli nie znaleziono bezpiecznego podziału
    """
    word_count = _count_words(sentence)
    if word_count < threshold:
        return None
    
    candidates = []
    
    all_tiers = [
        (1, SPLIT_POINTS_TIER1),
        (2, SPLIT_POINTS_TIER2),
        (3, SPLIT_POINTS_TIER3),
    ]
    
    for tier_num, tier_points in all_tiers:
        for pattern, replacement, split_type in tier_points:
            # Znajdź WSZYSTKIE wystąpienia wzorca w zdaniu
            for match in _re.finditer(pattern, sentence):
                start, end = match.start(), match.end()
                part1 = sentence[:start].strip()
                # Replacement zawiera nowy początek part2 (np. ". Ale ")
                # Weź tylko to co po replacement (kapitalizacja jest w replacement)
                part2_raw = sentence[end:].strip()
                
                # Zbuduj part2 z odpowiednim początkiem
                # replacement = '. Ale ' → part1 kończy się kropką, part2 zaczyna od 'Ale ...'
                rep_parts = replacement.split('. ', 1)
                if len(rep_parts) == 2 and rep_parts[1]:
                    # np. replacement = '. Ale ' → prefix = 'Ale '
                    prefix = rep_parts[1]
                    part2 = prefix + part2_raw
                else:
                    # np. replacement = '. ' → po prostu capitalize
                    part2 = part2_raw[0].upper() + part2_raw[1:] if part2_raw else part2_raw
                
                # Zakończ part1 kropką jeśli nie ma
                if part1 and part1[-1] not in '.!?':
                    part1 = part1 + '.'
                
                # Sprawdź czy obie części mają minimalną długość
                if _count_words(part1) >= MIN_HALF_WORDS and _count_words(part2) >= MIN_HALF_WORDS:
                    # Oblicz balans (im bliżej 0.5, tym lepiej)
                    total = _count_words(part1) + _count_words(part2)
                    balance = min(_count_words(part1), _count_words(part2)) / total
                    
                    candidates.append({
                        'result': SplitResult(
                            original=sentence,
                            part1=part1,
                            part2=part2,
                            split_type=split_type,
                            tier=tier_num
                        ),
                        'balance': balance,
                        'tier': tier_num,
                    })
    
    if not candidates:
        return None
    
    # Wybierz: priorytet tier (niższy = lepszy), potem balans (wyższy = lepszy)
    candidates.sort(key=lambda c: (c['tier'], -c['balance']))
    return candidates[0]['result']


def split_long_sentences(
    text: str,
    threshold: int = LONG_SENTENCE_THRESHOLD,
    max_splits: int = 4,
    min_tier: int = 3,
) -> Dict[str, any]:
    """
    Dzieli długie zdania w tekście na krótsze w naturalnych punktach gramatycznych.
    
    Args:
        text: Tekst do przetworzenia (batch_content)
        threshold: Min. liczba słów w zdaniu, żeby próbować dzielić (default 28)
        max_splits: Max liczba zdań do podzielenia w jednym batchu (default 4)
        min_tier: Najniższy akceptowalny tier (1=tylko bezpieczne, 3=wszystkie)
        
    Returns:
        Dict z:
        - modified_text: tekst po podziale
        - splits: lista SplitResult (co zostało podzielone)
        - stats: statystyki (ile zdań było długich, ile podzielono)
        - before_after: lista par (before, after) do prezentacji GPT
    """
    # Podziel na akapity (zachowaj strukturę)
    paragraphs = text.split('\n')
    
    splits_done = []
    modified_paragraphs = []
    long_count = 0
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            modified_paragraphs.append(paragraph)
            continue
        
        # Podziel akapit na zdania (regex z ai_detection_metrics)
        sentences = _re.split(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])', paragraph)
        modified_sentences = []
        
        for sentence in sentences:
            word_count = _count_words(sentence)
            
            if word_count >= threshold and len(splits_done) < max_splits:
                long_count += 1
                split = _find_best_split(sentence, threshold)
                
                if split and split.tier <= min_tier:
                    splits_done.append(split)
                    modified_sentences.append(split.part1)
                    modified_sentences.append(split.part2)
                else:
                    modified_sentences.append(sentence)
            else:
                modified_sentences.append(sentence)
        
        modified_paragraphs.append(' '.join(modified_sentences))
    
    modified_text = '\n'.join(modified_paragraphs)
    
    # Wygeneruj before/after do prezentacji
    before_after = []
    for s in splits_done:
        before_after.append({
            "before": s.original,
            "after": f"{s.part1} {s.part2}",
            "split_type": s.split_type,
            "tier": s.tier,
        })
    
    return {
        "modified_text": modified_text,
        "splits": splits_done,
        "split_count": len(splits_done),
        "stats": {
            "long_sentences_found": long_count,
            "sentences_split": len(splits_done),
            "threshold": threshold,
            "max_tier_used": max(s.tier for s in splits_done) if splits_done else 0,
        },
        "before_after": before_after,
    }


def suggest_sentence_splits(
    text: str,
    threshold: int = LONG_SENTENCE_THRESHOLD,
    max_suggestions: int = 4,
) -> List[Dict[str, str]]:
    """
    Zwraca SUGESTIE podziału długich zdań (bez modyfikacji tekstu).
    Używane w fix_instructions dla GPT — pokazuje co i jak podzielić.
    
    Returns:
        Lista dict z: original, suggested_part1, suggested_part2, split_type
    """
    result = split_long_sentences(text, threshold=threshold, max_splits=max_suggestions)
    
    suggestions = []
    for ba in result["before_after"]:
        suggestions.append({
            "original": ba["before"][:120] + ("..." if len(ba["before"]) > 120 else ""),
            "suggested": ba["after"][:140] + ("..." if len(ba["after"]) > 140 else ""),
            "split_type": ba["split_type"],
        })
    
    return suggestions


# ============================================================================
# SYNONIMY DYNAMICZNE - zamiast słabego SYNONYM_MAP
# ============================================================================

# 🆕 v41.0: Synonimy kontekstowe - importowane z contextual_synonyms_v41.py
# 105 słów w 7 kategoriach (było 25 słów)
CONTEXTUAL_SYNONYMS = CONTEXTUAL_SYNONYMS_V41


def get_synonyms_for_word(word: str, context: str = "") -> List[str]:
    """
    🆕 v41.0: Zwraca synonimy dla słowa - najpierw rozszerzona mapa v41.
    
    Hierarchia źródeł:
    1. contextual_synonyms_v41 (105 słów - najszybsze)
    2. synonym_service (plWordNet API + Firestore cache + LLM fallback)
    
    Args:
        word: Słowo do znalezienia synonimów
        context: Opcjonalny kontekst (np. "artykuł prawniczy")
        
    Returns:
        Lista synonimów (max 5)
    """
    word_lower = word.lower().strip()
    
    # 1. NAJPIERW: rozszerzona mapa v41 (105 słów)
    v41_synonyms = get_synonyms_v41(word_lower, max_count=5)
    if v41_synonyms:
        return v41_synonyms
    
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
                "Krótkie zdania twórz SAM z kontekstu akapitu — nie kopiuj gotowych fraz",
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
    
    # 🆕 v41.0: Sentence Splitter
    'split_long_sentences',
    'suggest_sentence_splits',
    
    # Klasy
    'BurstinessMetrics',
    
    # Stałe
    'CONTEXTUAL_SYNONYMS',
    'TOPIC_DOMAIN_CONFIG',
    'SHORT_SENTENCE_GRAMMAR_PATTERNS',
    
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
    print("Instrukcja (fragment):")
    print(result['instruction'][:300])
    print("...")
    print(f"Wzorce gramatyczne: {len(result['grammar_patterns'])}")
    print(f"Podpowiedzi domenowe: {len(result['context_hints'])}")
    
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

    # 🆕 v41.0: Test Sentence Splitter
    print("\n" + "=" * 60)
    print("TEST: Sentence Splitter v41.0")
    print("=" * 60)
    
    test_long_text = """Ubezwłasnowolnienie całkowite jest instytucją prawa cywilnego, która ma na celu ochronę osób niezdolnych do samodzielnego kierowania swoim postępowaniem, jednak jej zastosowanie wymaga spełnienia ściśle określonych przesłanek ustawowych. Sąd okręgowy rozpatruje wniosek o ubezwłasnowolnienie w postępowaniu nieprocesowym, ale przed wydaniem postanowienia konieczne jest przeprowadzenie badania przez biegłych psychiatrów oraz psychologów klinicznych. Procedura ta trwa zazwyczaj od kilku miesięcy do nawet roku, ponieważ wymaga zgromadzenia obszernej dokumentacji medycznej oraz przeprowadzenia szczegółowych badań stanu zdrowia psychicznego osoby, której dotyczy wniosek. Krótkie zdanie. Kolejne długie zdanie o prawie rodzinnym i opiekuńczym, które reguluje kwestie kurateli nad osobą ubezwłasnowolnioną częściowo; opiekun prawny natomiast jest powoływany w przypadku ubezwłasnowolnienia całkowitego."""
    
    result = split_long_sentences(test_long_text, threshold=25, max_splits=4)
    print(f"Znaleziono długich zdań: {result['stats']['long_sentences_found']}")
    print(f"Podzielono: {result['stats']['sentences_split']}")
    print()
    for ba in result['before_after']:
        print(f"  TYP: {ba['split_type']} (tier {ba['tier']})")
        print(f"  PRZED: {ba['before'][:100]}...")
        print(f"  PO:    {ba['after'][:100]}...")
        print()
