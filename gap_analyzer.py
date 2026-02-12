"""
===============================================================================
📊 GAP ANALYZER v1.0 — Information Gain dla S1
===============================================================================
Identyfikuje tematy, których NIKT w top 10 nie pokrywa,
ale użytkownicy o nie pytają (PAA, related searches).

3 typy gapów:
1. PAA_UNANSWERED — pytania PAA, na które nikt nie odpowiada
2. SUBTOPIC_MISSING — related searches nieobecne w H2 konkurencji
3. DEPTH_MISSING — sekcje u wszystkich, ale płytkie (<120 słów)

Integracja: index.py → po entity_seo → dodaje "content_gaps" do response

Autor: BRAJEN Team
Data: 2025
===============================================================================
"""

import re
from typing import List, Dict, Set, Optional
from collections import Counter
from dataclasses import dataclass, asdict


# ================================================================
# 📦 STRUKTURY DANYCH
# ================================================================

@dataclass
class ContentGap:
    """Pojedynczy zidentyfikowany gap w treści konkurencji."""
    topic: str
    gap_type: str           # "paa_unanswered", "subtopic_missing", "depth_missing"
    evidence: str           # dlaczego to gap
    priority: str           # "high", "medium", "low"
    suggested_h2: str       # propozycja H2

    def to_dict(self) -> Dict:
        return asdict(self)


# ================================================================
# 🧹 HELPERY
# ================================================================

# Polskie stop words (rozszerzone)
_STOP_WORDS_PL = {
    "i", "w", "na", "z", "do", "że", "się", "nie", "to", "jest", "za", "po",
    "od", "o", "jak", "ale", "co", "ten", "tym", "być", "może", "już", "tak",
    "gdy", "lub", "czy", "tego", "tej", "są", "dla", "ich", "przez", "jako",
    "te", "ze", "tych", "było", "ma", "przy", "tym", "które", "który", "która",
    "których", "jego", "jej", "tego", "także", "więc", "tylko", "też", "sobie",
    "bardzo", "jeszcze", "wszystko", "przed", "między", "pod", "nad", "bez",
    "oraz", "gdzie", "kiedy", "ile", "jeśli", "jaki", "jaka", "jakie",
}


def _extract_content_words(text: str, min_len: int = 4) -> Set[str]:
    """Wyciąga znaczące słowa z tekstu (bez stop words)."""
    words = set(re.findall(r'\b[a-ząćęłńóśźż]{' + str(min_len) + r',}\b', text.lower()))
    return words - _STOP_WORDS_PL


def _normalize_h2(h2: str) -> str:
    """Normalizuje H2 do porównań."""
    return re.sub(r'[^a-ząćęłńóśźż0-9\s]', '', h2.lower()).strip()


def _words_overlap(words_a: Set[str], words_b: Set[str], threshold: int = 2) -> bool:
    """Sprawdza czy dwa zestawy słów mają wystarczający overlap."""
    return len(words_a & words_b) >= threshold


# ================================================================
# 📊 GŁÓWNA ANALIZA
# ================================================================

def analyze_content_gaps(
    competitor_texts: List[str],
    competitor_h2s: List[str],
    paa_questions: List[Dict],
    related_searches: List[str],
    main_keyword: str,
    max_gaps: int = 15
) -> Dict:
    """
    Identyfikuje 3 typy gapów w treści konkurencji.

    Args:
        competitor_texts: Pełne treści stron z top 10
        competitor_h2s: Wszystkie H2 ze stron konkurencji
        paa_questions: Pytania People Also Ask (list of dicts z "question")
        related_searches: Related searches z Google
        main_keyword: Główna fraza kluczowa
        max_gaps: Max liczba gapów do zwrócenia

    Returns:
        Dict z content_gaps + agent_instruction
    """
    # Przygotuj dane
    combined_competitor = " ".join(t.lower() for t in competitor_texts if t)[:500000]
    
    # Normalizuj H2 konkurencji
    competitor_h2_normalized = set()
    competitor_h2_words = []
    for h2 in competitor_h2s:
        norm = _normalize_h2(h2)
        if norm:
            competitor_h2_normalized.add(norm)
            competitor_h2_words.append(_extract_content_words(h2))

    gaps = []

    # ═══════════════════════════════════════
    # 1. PAA UNANSWERED
    # ═══════════════════════════════════════
    for paa in paa_questions:
        question = paa.get("question", "") if isinstance(paa, dict) else str(paa)
        if not question or len(question) < 10:
            continue

        question_words = _extract_content_words(question)
        if len(question_words) < 2:
            continue

        # Czy jakiś H2 konkurencji pokrywa to pytanie?
        covered_by_h2 = any(
            _words_overlap(question_words, h2w)
            for h2w in competitor_h2_words
        )

        # Czy treść konkurencji odpowiada na pytanie?
        # (sprawdzamy ile kluczowych słów z pytania pojawia się w treściach)
        words_in_content = sum(1 for w in question_words if w in combined_competitor)
        content_coverage = words_in_content / max(1, len(question_words))

        if not covered_by_h2 and content_coverage < 0.6:
            # Nikt nie pokrywa tego pytania
            gaps.append(ContentGap(
                topic=question,
                gap_type="paa_unanswered",
                evidence=f"PAA pytanie niepokryte: {words_in_content}/{len(question_words)} "
                         f"słów kluczowych w treściach, brak H2",
                priority="high",
                suggested_h2=_question_to_h2(question)
            ))
        elif not covered_by_h2 and content_coverage < 0.8:
            # Treść częściowo pokrywa, ale brak dedykowanego H2
            gaps.append(ContentGap(
                topic=question,
                gap_type="paa_unanswered",
                evidence=f"PAA pytanie bez dedykowanego H2 ({content_coverage:.0%} pokrycia w treści)",
                priority="medium",
                suggested_h2=_question_to_h2(question)
            ))

    # ═══════════════════════════════════════
    # 2. SUBTOPIC MISSING
    # ═══════════════════════════════════════
    for search in related_searches:
        if not search or len(search) < 5:
            continue
            
        search_words = _extract_content_words(search)
        if len(search_words) < 2:
            continue

        # Czy jakiś H2 pokrywa ten related search?
        covered_by_h2 = any(
            _words_overlap(search_words, h2w)
            for h2w in competitor_h2_words
        )

        if not covered_by_h2:
            # Sprawdź czy to nie duplikat main_keyword
            main_kw_words = _extract_content_words(main_keyword)
            if search_words == main_kw_words:
                continue

            gaps.append(ContentGap(
                topic=search,
                gap_type="subtopic_missing",
                evidence=f"Related search '{search}' nieobecny w H2 konkurencji",
                priority="medium",
                suggested_h2=search.strip().capitalize()
            ))

    # ═══════════════════════════════════════
    # 3. DEPTH MISSING — płytkie sekcje
    # ═══════════════════════════════════════
    # Policz H2 pojawiające się u wielu konkurentów
    h2_counter = Counter(_normalize_h2(h) for h in competitor_h2s if h.strip())
    common_h2s = [h2 for h2, count in h2_counter.items() if count >= 3 and h2]

    for h2_norm in common_h2s:
        # Szacuj średnią długość sekcji pod tym H2 u konkurencji
        section_lengths = _estimate_section_lengths(competitor_texts, h2_norm)

        if section_lengths:
            avg_words = sum(section_lengths) / len(section_lengths)
            if avg_words < 120:
                # Znaleziono H2 w oryginale (nienormalizowany)
                original_h2 = _find_original_h2(competitor_h2s, h2_norm)
                
                gaps.append(ContentGap(
                    topic=original_h2 or h2_norm,
                    gap_type="depth_missing",
                    evidence=f"Sekcja '{original_h2 or h2_norm}' u konkurencji ma śr. "
                             f"{int(avg_words)} słów — można rozbudować",
                    priority="medium" if avg_words > 80 else "high",
                    suggested_h2=original_h2 or h2_norm.capitalize()
                ))

    # ═══════════════════════════════════════
    # DEDUPLIKACJA I SORTOWANIE
    # ═══════════════════════════════════════
    gaps = _deduplicate_gaps(gaps)
    
    # Sortuj: high > medium > low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: priority_order.get(g.priority, 2))
    gaps = gaps[:max_gaps]

    # ═══════════════════════════════════════
    # FORMATOWANIE DLA AGENTA
    # ═══════════════════════════════════════
    agent_instruction = _format_gaps_for_agent(gaps, main_keyword)

    return {
        "total_gaps": len(gaps),
        "paa_unanswered": [g.to_dict() for g in gaps if g.gap_type == "paa_unanswered"],
        "subtopic_missing": [g.to_dict() for g in gaps if g.gap_type == "subtopic_missing"],
        "depth_missing": [g.to_dict() for g in gaps if g.gap_type == "depth_missing"],
        "all_gaps": [g.to_dict() for g in gaps],
        "suggested_new_h2s": [
            g.suggested_h2 for g in gaps 
            if g.priority == "high" and g.gap_type in ("paa_unanswered", "subtopic_missing")
        ][:3],
        "agent_instruction": agent_instruction,
        "status": "OK"
    }


# ================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================

def _question_to_h2(question: str) -> str:
    """Konwertuje pytanie PAA na propozycję H2."""
    # Usuń znaki zapytania i capitalize
    h2 = question.rstrip("?").strip()
    # Jeśli zaczyna się od "Czy ", zmień na stwierdzenie
    if h2.lower().startswith("czy "):
        h2 = h2[4:]
    # Capitalize first letter
    if h2:
        h2 = h2[0].upper() + h2[1:]
    return h2


def _estimate_section_lengths(
    competitor_texts: List[str],
    h2_normalized: str
) -> List[int]:
    """
    Szacuje długość sekcji pod danym H2 u konkurencji.
    Proste podejście: znajdź H2 → policz słowa do następnego H2.
    """
    lengths = []
    h2_words = set(h2_normalized.split())
    
    for text in competitor_texts:
        if not text:
            continue
        text_lower = text.lower()
        
        # Znajdź pozycję H2 (przybliżone — szukamy słów z H2)
        # Szukamy linii, która wygląda jak H2 i zawiera nasze słowa
        lines = text_lower.split('\n')
        found_pos = -1
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if len(line_stripped) < 100:  # H2 raczej krótkie
                line_words = set(re.findall(r'\b[a-ząćęłńóśźż]{3,}\b', line_stripped))
                if len(h2_words & line_words) >= max(2, len(h2_words) - 1):
                    found_pos = i
                    break
        
        if found_pos == -1:
            continue
        
        # Policz słowa do następnego "H2-like" nagłówka
        section_text = []
        for j in range(found_pos + 1, min(found_pos + 50, len(lines))):
            line = lines[j].strip()
            # Heurystyka: krótka linia (<80 znaków) po pustej linii = prawdopodobnie H2
            if (len(line) < 80 and len(line) > 5 and 
                j > found_pos + 2 and not lines[j-1].strip()):
                break
            section_text.append(line)
        
        word_count = len(" ".join(section_text).split())
        if word_count > 10:  # Ignoruj bardzo krótkie (prawdopodobnie błędne matche)
            lengths.append(word_count)
    
    return lengths


def _find_original_h2(h2_list: List[str], normalized: str) -> Optional[str]:
    """Znajduje oryginalny (nienormalizowany) H2."""
    for h2 in h2_list:
        if _normalize_h2(h2) == normalized:
            return h2
    return None


def _deduplicate_gaps(gaps: List[ContentGap]) -> List[ContentGap]:
    """Usuwa duplikaty na podstawie podobieństwa tematu."""
    seen_topics = set()
    unique = []
    
    for gap in gaps:
        # Klucz dedup: pierwsze 3 znaczące słowa
        topic_words = sorted(_extract_content_words(gap.topic))[:3]
        key = "|".join(topic_words)
        
        if key not in seen_topics and key:
            seen_topics.add(key)
            unique.append(gap)
    
    return unique


def _format_gaps_for_agent(gaps: List[ContentGap], main_keyword: str) -> str:
    """Formatuje gapy jako instrukcję dla agenta GPT — v45.1 aggressive Information Gain."""
    if not gaps:
        return ""

    lines = [
        f"🏆 TWOJA PRZEWAGA NAD KONKURENCJĄ — \"{main_keyword}\":",
        "=" * 50,
        "Żaden z TOP 10 wyników w Google NIE pokrywa tych tematów.",
        "To Twoja UNIKALNA szansa na Information Gain — Google nagradza",
        "artykuły wnoszące NOWĄ informację vs to co już jest w SERP.",
        ""
    ]

    high_gaps = [g for g in gaps if g.priority == "high"]
    medium_gaps = [g for g in gaps if g.priority == "medium"]

    if high_gaps:
        lines.append("🔴 OBOWIĄZKOWE — napisz min 2-3 zdania o KAŻDYM:")
        for g in high_gaps[:5]:
            gap_type_label = {
                "paa_unanswered": "PAA — użytkownicy pytają, nikt nie odpowiada",
                "subtopic_missing": "TEMAT — konkurencja go pomija",
                "depth_missing": "GŁĘBIA — wszyscy piszą ogólnikowo"
            }.get(g.gap_type, g.gap_type)
            lines.append(f"  ▶ {g.topic}")
            lines.append(f"    [{gap_type_label}]")
            if g.suggested_h2:
                lines.append(f"    → Proponowany H2: \"{g.suggested_h2}\"")
        lines.append("")

    if medium_gaps:
        lines.append("🟡 WPLEĆ w istniejące sekcje (min 1 zdanie każdy):")
        for g in medium_gaps[:5]:
            lines.append(f"  ▶ {g.topic}")
            if g.evidence:
                lines.append(f"    (dowód: {g.evidence[:80]})")
        lines.append("")

    lines.append("⚠️ ZASADA: Skup się na tych gapach BARDZIEJ niż na powtarzaniu")
    lines.append("tego co wszyscy piszą. Powtarzanie = zero wartości dla Google.")
    lines.append("Unikalne treści = wyższy ranking. To proste.")
    lines.append("=" * 50)

    return "\n".join(lines)


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'analyze_content_gaps',
    'ContentGap',
    'analyze_batch_depth',
    'get_depth_hints',
]


# ================================================================
# 📏 DEPTH SCORER v1.0 — 10 sygnałów głębi treści
# ================================================================
# Importowane przez moe_batch_validator.py (Expert #11)
# Docs: Sekcja 5.3 — Depth Scorer
# ================================================================

# --- Signal definitions: (name, weight, patterns, expected_count) ---
_DEPTH_SIGNALS = [
    {
        "name": "legal_reference",
        "weight": 2.5,
        "expected": 2,
        "patterns": [
            r'\bart\.\s*\d+',                           # art. 58
            r'\b§\s*\d+',                               # § 3
            r'\bk\.c\.\b',                              # k.c.
            r'\bk\.p\.c\.\b',                           # k.p.c.
            r'\bk\.r\.o\.\b',                           # k.r.o.
            r'\bk\.k\.\b',                              # k.k.
            r'\bk\.p\.\b',                              # k.p.
            r'\bDz\.\s*U\.',                            # Dz. U.
            r'\bSN\b.*\bwyrok',                         # wyrok SN
            r'\bwyrok\b.*\bSN\b',                       # wyrok ... SN
            r'\b(?:I|II|III|IV|V)\s+C[SK]?\s+\d+/\d+',  # sygnatura I CSK 123/20
            r'\bustaw[ayę]\b',                          # ustawy/ustawę
            r'\brozporządzen',                          # rozporządzeni-e/a
        ],
    },
    {
        "name": "scientific_reference",
        "weight": 2.5,
        "expected": 1,
        "patterns": [
            r'\bPMID\s*:?\s*\d+',                       # PMID: 12345678
            r'\bDOI\s*:?\s*10\.\d+',                    # DOI: 10.xxx
            r'\bbadani[ae]\b.*\b(?:wykazał|potwierdził|dowodzą)',
            r'\bmeta[\-\s]?analiz',                     # meta-analiza
            r'\bprzegląd\s+systematyczny',
            r'\brandomizowane\b',
            r'\bpublikacj[iaę]\b.*\b\d{4}\b',           # publikacja ... 2023
            r'\bwg\s+(?:[A-Z][a-z]+)',                  # wg Kowalski
            r'\bwedług\s+(?:[A-Z][a-z]+)',              # według Nowaka
        ],
    },
    {
        "name": "specific_number",
        "weight": 2.0,
        "expected": 3,
        "patterns": [
            r'\d+\s*(?:zł|PLN|EUR|USD)',                # kwoty
            r'\d+\s*%',                                 # procenty
            r'\d+\s*(?:dni|miesięc|lat|godzin|tygodni)',  # okresy
            r'\d+\s*(?:mm|cm|m|km|kg|g|mg|ml|l)',       # miary
            r'(?:od|do|około|ok\.)\s+\d+',              # od/do X
            r'\d+(?:\s*-\s*|\s*–\s*)\d+',              # zakresy 10-20
            r'(?:co najmniej|minimum|maksimum)\s+\d+',
        ],
    },
    {
        "name": "named_institution",
        "weight": 1.8,
        "expected": 2,
        "patterns": [
            r'\b(?:Sąd\s+(?:Najwyższy|Okręgowy|Rejonowy|Apelacyjny))',
            r'\b(?:ZUS|NFZ|GUS|NBP|KNF|UOKiK|PIP|GIODO|UODO)\b',
            r'\b(?:WHO|FDA|EMA|CDC|EFSA)\b',
            r'\b(?:Ministerstwo|Minister)\s+\w+',
            r'\b(?:Urząd|Izba|Komisja|Rada|Instytut)\s+\w+',
            r'\b(?:Uniwersytet|Politechnika|Akademia)\s+\w+',
            r'\b(?:Rzecznik|Prokurator|Komornik)\b',
        ],
    },
    {
        "name": "practical_advice",
        "weight": 1.8,
        "expected": 2,
        "patterns": [
            r'\b(?:warto|zaleca się|rekomenduje|sugeruje)\b',
            r'\b(?:najlepiej|optymalnie|idealnie)\b',
            r'\b(?:pamiętaj|uwaga|ważne|istotne)\b',
            r'\b(?:w praktyce|w rzeczywistości)\b',
            r'\b(?:tip|wskazówka|rada|porada)\b',
            r'\b(?:unikaj|nie rób|nie stosuj)\b',
            r'\b(?:krok\s+\d+|po\s+pierwsze|po\s+drugie)\b',
        ],
    },
    {
        "name": "causal_explanation",
        "weight": 1.5,
        "expected": 2,
        "patterns": [
            r'\b(?:ponieważ|dlatego(?:\s+że)?|gdyż|bowiem|albowiem)\b',
            r'\b(?:w\s+rezultacie|w\s+konsekwencji|w\s+efekcie|skutkiem)\b',
            r'\b(?:powoduje|prowadzi\s+do|skutkuje|wynika\s+z)\b',
            r'\b(?:przyczyn[ayą]|źródł[eo]m?)\b',
            r'\b(?:mechanizm|proces)\b.*\b(?:polega|działa)\b',
        ],
    },
    {
        "name": "exception_case",
        "weight": 1.5,
        "expected": 1,
        "patterns": [
            r'\b(?:wyjąt(?:ek|kiem|kowo)|z\s+wyjątkiem)\b',
            r'\b(?:jednakże|niemniej|aczkolwiek)\b',
            r'\b(?:chyba\s+że|o\s+ile|pod\s+warunkiem)\b',
            r'\b(?:nie\s+dotyczy|nie\s+obejmuje|wyłącz(?:ając|enie))\b',
            r'\b(?:szczególn(?:ym|ie|ych)\s+przypadk)\b',
            r'\b(?:odstępstwo|wyjątek\s+od\s+zasady)\b',
        ],
    },
    {
        "name": "date_reference",
        "weight": 1.5,
        "expected": 2,
        "patterns": [
            r'\b(?:20[12]\d)\s*(?:r\.?|roku)\b',        # 2024 r.
            r'\b(?:od|do|w)\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca'
            r'|lipca|sierpnia|września|października|listopada|grudnia)\b',
            r'\b(?:od|do|w)\s+\d{4}\b',                 # od 2023
            r'\b(?:nowelizacj[aię]|zmiana)\b.*\b\d{4}\b',
            r'\b(?:obowiązuje|wchodzi\s+w\s+życie)\b',
        ],
    },
    {
        "name": "comparison",
        "weight": 1.2,
        "expected": 1,
        "patterns": [
            r'\b(?:w\s+porównaniu|w\s+odróżnieniu|w\s+przeciwieństwie)\b',
            r'\b(?:z\s+kolei|natomiast|tymczasem)\b',
            r'\b(?:więcej|mniej|szybciej|wolniej|drożej|taniej)\s+niż\b',
            r'\b(?:podobnie|analogicznie|tak\s+jak)\b',
            r'\b(?:z\s+jednej\s+strony|z\s+drugiej\s+strony)\b',
        ],
    },
    {
        "name": "process_steps",
        "weight": 1.0,
        "expected": 1,
        "patterns": [
            r'\b(?:krok\s+\d+|etap\s+\d+|faz[aę]\s+\d+)\b',
            r'\b(?:następnie|kolejno|w\s+dalszej\s+kolejności)\b',
            r'\b(?:najpierw|na\s+początku|na\s+końcu|na\s+koniec)\b',
            r'\b(?:procedura|proces)\b.*\b(?:obejmuje|polega|składa\s+się)\b',
            r'\bpo\s+(?:pierwsze|drugie|trzecie|czwarte)\b',
        ],
    },
]

_MAX_RAW_SCORE = sum(s["weight"] for s in _DEPTH_SIGNALS)  # 17.3


def _count_signal(text: str, patterns: List[str]) -> int:
    """Count unique matches for a set of regex patterns in text."""
    count = 0
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        except re.error:
            continue
    return count


def _score_signal(count: int, expected: int, weight: float) -> float:
    """Score a single signal: capped at weight, proportional to expected count."""
    if expected <= 0:
        return weight if count > 0 else 0.0
    ratio = min(1.0, count / expected)
    return ratio * weight


def analyze_batch_depth(
    batch_text: str,
    h2_list: List[str] = None,
    is_ymyl: bool = False
) -> Dict:
    """
    Analyze batch text for 10 depth signals.
    
    Returns:
        Dict with overall_score (0-100), sections[], shallow_count,
        shallow_sections[], fix_instructions[]
    
    Used by: moe_batch_validator.py (Expert #11)
    Docs: Section 5.3 Depth Scorer
    """
    if not batch_text or len(batch_text.strip()) < 50:
        return {
            "overall_score": 0,
            "raw_score": 0.0,
            "sections": [],
            "shallow_count": 1 if h2_list else 0,
            "shallow_sections": h2_list or [],
            "fix_instructions": ["Zbyt krótki tekst — brak sygnałów głębi"],
            "signals": {},
        }
    
    threshold = 40 if is_ymyl else 30
    text = batch_text
    
    # --- Score the full batch text ---
    signals_detail = {}
    raw_score = 0.0
    
    for signal_def in _DEPTH_SIGNALS:
        name = signal_def["name"]
        count = _count_signal(text, signal_def["patterns"])
        score = _score_signal(count, signal_def["expected"], signal_def["weight"])
        raw_score += score
        signals_detail[name] = {
            "count": count,
            "score": round(score, 2),
            "weight": signal_def["weight"],
            "max_score": signal_def["weight"],
        }
    
    overall_score = round((raw_score / _MAX_RAW_SCORE) * 100) if _MAX_RAW_SCORE > 0 else 0
    overall_score = min(100, max(0, overall_score))
    
    # --- Per-section analysis (split by H2) ---
    sections = []
    section_texts = []
    
    if h2_list:
        # Split text by H2 markers
        h2_pattern = r'(?:^|\n)\s*h2:\s*(.+?)(?:\n|$)'
        parts = re.split(h2_pattern, text, flags=re.IGNORECASE)
        
        # parts = [before_first_h2, h2_title_1, content_1, h2_title_2, content_2, ...]
        for i in range(1, len(parts) - 1, 2):
            h2_title = parts[i].strip()
            h2_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            section_texts.append((h2_title, h2_content))
        
        # If no splits found, treat whole text as one section
        if not section_texts and h2_list:
            section_texts = [(h2_list[0], text)]
    else:
        section_texts = [("batch", text)]
    
    shallow_sections = []
    
    for h2_title, section_text in section_texts:
        section_raw = 0.0
        section_signals = {}
        
        for signal_def in _DEPTH_SIGNALS:
            name = signal_def["name"]
            count = _count_signal(section_text, signal_def["patterns"])
            score = _score_signal(count, signal_def["expected"], signal_def["weight"])
            section_raw += score
            if count > 0:
                section_signals[name] = count
        
        section_score = round((section_raw / _MAX_RAW_SCORE) * 100) if _MAX_RAW_SCORE > 0 else 0
        is_shallow = section_score < threshold
        
        sections.append({
            "h2": h2_title,
            "score": section_score,
            "is_shallow": is_shallow,
            "signals_found": section_signals,
            "word_count": len(section_text.split()),
        })
        
        if is_shallow:
            shallow_sections.append(h2_title)
    
    # --- Fix instructions ---
    fix_instructions = []
    
    # Find missing signals (score = 0)
    missing_signals = [
        name for name, detail in signals_detail.items()
        if detail["count"] == 0
    ]
    
    signal_hints = {
        "legal_reference": "Dodaj referencje prawne (art., wyroki, ustawy)",
        "scientific_reference": "Dodaj cytowania naukowe (badania, PMID, DOI)",
        "specific_number": "Dodaj konkretne liczby (kwoty, %, okresy, wymiary)",
        "named_institution": "Wymień konkretne instytucje (sądy, urzędy, organizacje)",
        "practical_advice": "Dodaj praktyczne porady i wskazówki",
        "causal_explanation": "Wyjaśnij przyczyny i skutki (dlaczego/dlatego)",
        "exception_case": "Opisz wyjątki i przypadki szczególne",
        "date_reference": "Dodaj daty i odniesienia czasowe",
        "comparison": "Dodaj porównania (więcej niż, w odróżnieniu od)",
        "process_steps": "Opisz kroki procedury (krok 1, następnie, po pierwsze)",
    }
    
    # Top 3 missing high-weight signals
    missing_sorted = sorted(
        missing_signals,
        key=lambda n: next(
            (s["weight"] for s in _DEPTH_SIGNALS if s["name"] == n), 0
        ),
        reverse=True
    )
    
    for sig_name in missing_sorted[:3]:
        if sig_name in signal_hints:
            fix_instructions.append(signal_hints[sig_name])
    
    if overall_score < threshold:
        fix_instructions.insert(0,
            f"Depth score {overall_score}/100 < próg {threshold} "
            f"({'YMYL' if is_ymyl else 'non-YMYL'}). Wzbogać treść o konkretne dane."
        )
    
    return {
        "overall_score": overall_score,
        "raw_score": round(raw_score, 2),
        "max_raw_score": round(_MAX_RAW_SCORE, 2),
        "threshold": threshold,
        "is_ymyl": is_ymyl,
        "sections": sections,
        "shallow_count": len(shallow_sections),
        "shallow_sections": shallow_sections,
        "fix_instructions": fix_instructions,
        "signals": signals_detail,
    }


def get_depth_hints(
    s1_data: Dict,
    is_ymyl: bool = False
) -> str:
    """
    Generate depth hints for GPT prompt (pre_batch_info).
    
    Returns formatted string with depth guidance based on S1 analysis.
    Used by: enhanced_pre_batch.py
    """
    hints = []
    threshold = 40 if is_ymyl else 30
    
    hints.append(f"=== DEPTH SIGNALS (próg: {threshold}/100) ===")
    
    if is_ymyl:
        hints.append("⚠️ YMYL: Wymagane referencje prawne/naukowe i cytowania instytucji.")
        hints.append("Każda sekcja MUSI zawierać: konkretne liczby, daty, nazwy instytucji.")
    
    # Extract competitor depth signals from S1
    content_gaps = s1_data.get("content_gaps", {})
    depth_missing = content_gaps.get("depth_missing", [])
    
    if depth_missing:
        hints.append(f"\nBrakujące sygnały głębi u konkurencji ({len(depth_missing)}):")
        for gap in depth_missing[:5]:
            topic = gap.get("topic", "") if isinstance(gap, dict) else str(gap)
            hints.append(f"  → {topic}")
    
    hints.append("\n10 sygnałów głębi (wagi):")
    hints.append("  2.5: referencje prawne (art., wyroki) + naukowe (badania, PMID)")
    hints.append("  2.0: konkretne liczby (kwoty, %, okresy)")
    hints.append("  1.8: instytucje + praktyczne porady")
    hints.append("  1.5: wyjaśnienia przyczynowe + wyjątki + daty")
    hints.append("  1.2: porównania | 1.0: kroki procedur")
    
    return "\n".join(hints)
