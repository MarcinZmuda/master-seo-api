"""
===============================================================================
📏 DEPTH SCORER v1.0 — Section Quality Measurement
===============================================================================
Mierzy czy sekcja H2 wnosi GŁĘBIĘ merytoryczną, czy jest powierzchowna.

Sygnały głębi (więcej = lepiej):
- Konkretne liczby, daty, kwoty
- Nazwane instytucje (nie "właściwy sąd" tylko "Sąd Okręgowy w Warszawie")
- Cytowania prawne, naukowe
- Wyjaśnienia przyczynowe (dlaczego, ponieważ)
- Porównania z alternatywami
- Wyjątki od reguły
- Praktyczne porady

Integracja:
1. Standalone: score_section_depth() per sekcja H2
2. MoE Expert #11: DepthExpert w moe_batch_validator.py
3. Pre-batch hint: get_depth_hints() → instrukcja dla agenta

Autor: BRAJEN Team
Data: 2025
===============================================================================
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ================================================================
# 📊 KONFIGURACJA
# ================================================================

@dataclass
class DepthSignal:
    """Definicja jednego sygnału głębi."""
    name: str
    description: str
    weight: float
    patterns: List[str]     # regex patterns do wykrycia


# Sygnały głębi — posortowane po wadze (najważniejsze pierwsze)
DEPTH_SIGNALS: List[DepthSignal] = [
    # ═══ TWARDE DANE — najwyższa waga ═══
    DepthSignal(
        name="legal_reference",
        description="Cytowanie artykułu ustawy, wyroku, rozporządzenia",
        weight=2.5,
        patterns=[
            r'art\.\s*\d+\s*(?:§\s*\d+)?\s*(?:k\.c\.|k\.p\.c\.|k\.r\.o\.|k\.k\.|k\.p\.|k\.s\.h\.|k\.w\.|u\.s\.p\.)',
            r'(?:Dz\.?\s*U\.?\s*(?:z\s*)?\d{4})',
            r'(?:wyrok|uchwała|postanowienie)\s+(?:SN|SA|SO|SR|NSA|WSA|TK)',
            r'(?:rozporządzeni[eua])\s+(?:Ministra|Prezesa|Rady)',
        ]
    ),
    DepthSignal(
        name="scientific_reference",
        description="Cytowanie badania, publikacji, danych statystycznych",
        weight=2.5,
        patterns=[
            r'(?:PMID|DOI|NCT)\s*:?\s*[\d/]+',
            r'(?:badanie|metaanaliza|przegląd systematyczny|metaanalizę)\s+(?:[A-Z][a-ząćęłńóśźż]+)',
            r'(?:wg|według)\s+(?:badań|danych|raportu|publikacji|statystyk)',
            r'(?:opublikowan[aoey]\s+w|w\s+czasopiśmie|w\s+journalu)',
        ]
    ),
    DepthSignal(
        name="specific_number",
        description="Konkretna liczba/kwota/procent (nie 'około')",
        weight=2.0,
        patterns=[
            r'\b\d+[\s,.]\d*\s*(?:zł|złotych|PLN|EUR|USD|%|procent)',
            r'\b\d+\s*(?:tygodni|miesięcy|dni|lat|godzin|minut)',
            r'(?:od|do|między)\s+\d+\s+(?:a|do|i)\s+\d+',
            r'\b\d{2,}\s*(?:m²|m2|km|ha|cm|mm|mg|ml|kg)',
        ]
    ),
    DepthSignal(
        name="named_institution",
        description="Nazwana instytucja (nie 'właściwy sąd' — konkretna nazwa)",
        weight=1.8,
        patterns=[
            r'(?:Sąd\s+(?:Okręgowy|Rejonowy|Najwyższy|Apelacyjny)\s+(?:w\s+)?[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)',
            r'(?:(?:Ministerstwo|Urząd|Zakład|Agencja|Instytut|Centrum|Szpital|Klinika)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\s]{3,})',
            r'(?:ZUS|NFZ|GUS|PZH|AOTMiT|PARP|UOKiK|KRS|CEIDG|GIF|PIP|UODO)',
        ]
    ),
    DepthSignal(
        name="date_reference",
        description="Konkretna data/rok/okres",
        weight=1.5,
        patterns=[
            r'\b(?:20[12]\d|19\d{2})\s*(?:r\.|roku)',
            r'\b\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)',
            r'(?:od\s+(?:20[12]\d|19\d{2})\s+(?:r\.|roku))',
            r'(?:nowelizacj[aię]\s+z\s+\d{1,2})',
        ]
    ),

    # ═══ GŁĘBIA WYJAŚNIENIOWA ═══
    DepthSignal(
        name="causal_explanation",
        description="Wyjaśnienie przyczynowe (dlaczego, ponieważ, w wyniku)",
        weight=1.5,
        patterns=[
            r'(?:ponieważ|dlatego\s+że|gdyż|bowiem|albowiem)',
            r'(?:w wyniku|na skutek|wskutek|w rezultacie|co prowadzi do)',
            r'(?:przyczyną|powodem|skutkiem)\s+(?:jest|może być|bywa)',
            r'(?:wynika\s+to\s+z|tłumaczy\s+to)',
        ]
    ),
    DepthSignal(
        name="exception_case",
        description="Wyjątek od reguły (chyba że, z wyjątkiem, jednak)",
        weight=1.5,
        patterns=[
            r'(?:z wyjątkiem|wyjątkowo|chyba\s+że|o ile nie)',
            r'(?:jednakże|niemniej jednak|mimo to|aczkolwiek|pomimo)',
            r'(?:uwaga:|zastrzeżenie:|wyjątek:)',
            r'(?:nie dotyczy to|nie stosuje się do|wyłączeni[ea])',
        ]
    ),
    DepthSignal(
        name="comparison",
        description="Porównanie z alternatywą (w odróżnieniu od, zamiast)",
        weight=1.2,
        patterns=[
            r'(?:w odróżnieniu od|w przeciwieństwie do|w porównaniu z)',
            r'(?:z jednej strony|z drugiej strony)',
            r'(?:zamiast|lepsze niż|gorsze niż|szybsze niż|skuteczniejsz)',
            r'(?:w odróżnieniu|w porównaniu)',
        ]
    ),

    # ═══ SYGNAŁY DOŚWIADCZENIA ═══
    DepthSignal(
        name="practical_advice",
        description="Praktyczna porada (w praktyce, z doświadczenia)",
        weight=1.8,
        patterns=[
            r'(?:w praktyce|z doświadczenia|z naszego doświadczenia)',
            r'(?:typowo|najczęściej|statystycznie)',
            r'(?:klienci|pacjenci)\s+(?:często|najczęściej|zwykle)\s+(?:pytają|zgłaszają|nie wiedzą)',
            r'(?:częsty błąd|częstym błędem|często popełnianym)',
        ]
    ),
    DepthSignal(
        name="process_steps",
        description="Kroki procedury (krok 1, etap, najpierw/potem)",
        weight=1.0,
        patterns=[
            r'(?:krok\s+\d|etap\s+\d|faza\s+\d)',
            r'(?:najpierw|następnie|potem|w kolejnym kroku|na końcu)',
            r'(?:procedura\s+(?:obejmuje|składa się|wygląda))',
        ]
    ),
]


# ================================================================
# 📊 SCORING
# ================================================================

def score_section_depth(
    section_text: str,
    h2_title: str,
    is_ymyl: bool = False
) -> Dict:
    """
    Ocenia głębię merytoryczną sekcji H2.

    Args:
        section_text: Tekst sekcji (pod jednym H2)
        h2_title: Tytuł H2
        is_ymyl: Czy artykuł YMYL (wyższe progi)

    Returns:
        {
            "depth_score": 0-100,
            "signals_found": {...},
            "signals_missing": [...],
            "is_shallow": bool,
            "word_count": int,
            "recommendation": str
        }
    """
    if not section_text or not section_text.strip():
        return {
            "depth_score": 0,
            "signals_found": {},
            "signals_missing": [{"signal": s.name, "description": s.description} for s in DEPTH_SIGNALS[:5]],
            "is_shallow": True,
            "word_count": 0,
            "recommendation": f"Sekcja '{h2_title}' jest pusta."
        }

    found_signals = {}
    total_weight = 0.0
    max_weight = sum(s.weight for s in DEPTH_SIGNALS)

    for signal in DEPTH_SIGNALS:
        for pattern in signal.patterns:
            try:
                if re.search(pattern, section_text, re.IGNORECASE):
                    found_signals[signal.name] = {
                        "description": signal.description,
                        "weight": signal.weight
                    }
                    total_weight += signal.weight
                    break  # Jeden match per signal wystarczy
            except re.error:
                continue

    # Bonus za długość (sekcja >200 słów = dodatkowe punkty, max 15%)
    word_count = len(section_text.split())
    length_bonus = min(0.15, word_count / 2000)

    # Oblicz score
    raw_score = (total_weight / max_weight) + length_bonus
    depth_score = min(100, int(raw_score * 100))

    # Progi — YMYL wymaga wyższych standardów
    threshold = 40 if is_ymyl else 30

    # Brakujące sygnały (posortowane po wadze, najważniejsze pierwsze)
    missing = [
        {"signal": s.name, "description": s.description, "weight": s.weight}
        for s in DEPTH_SIGNALS
        if s.name not in found_signals
    ]
    missing.sort(key=lambda m: -m["weight"])

    # Rekomendacja
    recommendation = ""
    if depth_score < threshold:
        top_missing = missing[:3]
        recommendation = (
            f"Sekcja '{h2_title}' jest płytka (score: {depth_score}/{threshold}). "
            f"Dodaj: {', '.join(m['description'] for m in top_missing)}"
        )

    return {
        "depth_score": depth_score,
        "signals_found": found_signals,
        "signals_missing": missing[:5],
        "is_shallow": depth_score < threshold,
        "word_count": word_count,
        "threshold": threshold,
        "recommendation": recommendation
    }


# ================================================================
# 📋 BATCH-LEVEL DEPTH ANALYSIS
# ================================================================

def analyze_batch_depth(
    batch_text: str,
    h2_list: List[str],
    is_ymyl: bool = False
) -> Dict:
    """
    Analizuje głębię wszystkich sekcji w batchu.

    Args:
        batch_text: Pełny tekst batcha
        h2_list: Lista H2 w tym batchu
        is_ymyl: Czy YMYL

    Returns:
        {
            "overall_score": 0-100,
            "sections": [{"h2": ..., "score": ..., "is_shallow": ...}],
            "shallow_sections": [...],
            "fix_instructions": [...]
        }
    """
    if not batch_text or not h2_list:
        return {
            "overall_score": 0,
            "sections": [],
            "shallow_sections": [],
            "fix_instructions": []
        }

    # Podziel tekst na sekcje po H2
    sections = _split_by_h2(batch_text, h2_list)

    section_results = []
    shallow = []
    fixes = []

    for h2, text in sections.items():
        result = score_section_depth(text, h2, is_ymyl)
        section_results.append({
            "h2": h2,
            "depth_score": result["depth_score"],
            "is_shallow": result["is_shallow"],
            "word_count": result["word_count"],
            "signals_found": list(result["signals_found"].keys()),
            "top_missing": [m["description"] for m in result["signals_missing"][:2]]
        })

        if result["is_shallow"]:
            shallow.append(h2)
            if result["recommendation"]:
                fixes.append(result["recommendation"])

    # Overall score = średnia ważona (dłuższe sekcje ważą więcej)
    if section_results:
        total_words = sum(s["word_count"] for s in section_results)
        if total_words > 0:
            overall = sum(
                s["depth_score"] * s["word_count"] / total_words
                for s in section_results
            )
        else:
            overall = sum(s["depth_score"] for s in section_results) / len(section_results)
    else:
        overall = 0

    return {
        "overall_score": int(overall),
        "sections": section_results,
        "shallow_sections": shallow,
        "shallow_count": len(shallow),
        "fix_instructions": fixes[:5]
    }


def _split_by_h2(text: str, h2_list: List[str]) -> Dict[str, str]:
    """Dzieli tekst na sekcje po H2."""
    sections = {}
    text_lower = text.lower()

    # Znajdź pozycje H2 w tekście
    h2_positions = []
    for h2 in h2_list:
        h2_lower = h2.lower().strip()
        # Szukaj H2 z różnymi formatami (markdown, plain)
        for prefix in ['## ', '### ', '']:
            pos = text_lower.find(prefix + h2_lower)
            if pos != -1:
                h2_positions.append((pos, h2))
                break

    # Posortuj po pozycji
    h2_positions.sort(key=lambda x: x[0])

    # Wyciągnij tekst między H2
    for i, (pos, h2) in enumerate(h2_positions):
        start = pos + len(h2) + 5  # +5 na prefix i newline
        end = h2_positions[i + 1][0] if i + 1 < len(h2_positions) else len(text)
        section_text = text[start:end].strip()
        sections[h2] = section_text

    # Jeśli nie znaleziono żadnego H2, traktuj cały tekst jako jedną sekcję
    if not sections and h2_list:
        sections[h2_list[0]] = text

    return sections


# ================================================================
# 📝 PRE-BATCH HINTS
# ================================================================

def get_depth_hints(
    h2_title: str,
    domain: str = "prawo",
    is_ymyl: bool = False
) -> str:
    """
    Generuje hint dla agenta GPT — jakie sygnały głębi dodać.

    Args:
        h2_title: Tytuł aktualnego H2
        domain: Domena artykułu
        is_ymyl: Czy YMYL

    Returns:
        Instrukcja tekstowa dla agenta
    """
    hints = []

    if domain == "prawo" or is_ymyl:
        hints.extend([
            "Cytuj konkretny artykuł ustawy (art. X k.c./k.r.o./k.p.c.)",
            "Podaj nazwę sądu (Sąd Okręgowy w..., nie 'właściwy sąd')",
            "Dodaj konkretny termin lub kwotę (np. '14 dni', '300 zł')",
        ])
    elif domain == "medycyna":
        hints.extend([
            "Podaj konkretną dawkę, czas trwania lub skuteczność (%)",
            "Nazwij badanie lub wytyczne (np. 'wg wytycznych PTG z 2023')",
            "Dodaj wyjątek lub przeciwwskazanie",
        ])
    else:
        hints.extend([
            "Podaj konkretną liczbę lub statystykę",
            "Wymień nazwaną instytucję lub źródło",
            "Dodaj porównanie z alternatywą",
        ])

    # Uniwersalne
    hints.extend([
        "Wyjaśnij DLACZEGO (przyczyna), nie tylko CO (fakt)",
        "Dodaj wyjątek od opisanej reguły",
    ])

    return (
        f"📏 GŁĘBIA SEKCJI \"{h2_title}\": "
        + " | ".join(hints[:4])
    )


# ================================================================
# 🔌 MOE EXPERT INTERFACE
# ================================================================
# Gotowy interface do użycia jako Expert #11 w moe_batch_validator.py
#
# W moe_batch_validator.py dodaj:
#
# try:
#     from depth_scorer import analyze_batch_depth, get_depth_hints
#     DEPTH_SCORER_AVAILABLE = True
# except ImportError:
#     DEPTH_SCORER_AVAILABLE = False
#
# Następnie w validate_batch_moe(), po PERPLEXITY EXPERT:
#
#     if DEPTH_SCORER_AVAILABLE:
#         try:
#             h2_list = [current_h2] if current_h2 else []
#             is_ymyl = project_data.get("is_ymyl", False)
#             depth_result = analyze_batch_depth(
#                 corrected_text or batch_text, h2_list, is_ymyl
#             )
#             experts_summary["depth"] = {
#                 "enabled": True,
#                 "overall_score": depth_result["overall_score"],
#                 "shallow_count": depth_result["shallow_count"],
#             }
#             for fix in depth_result["fix_instructions"][:3]:
#                 fix_instructions.append(f"[DEPTH] {fix}")
#         except Exception as e:
#             experts_summary["depth"] = {"enabled": False, "error": str(e)[:100]}
# ================================================================


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'score_section_depth',
    'analyze_batch_depth',
    'get_depth_hints',
    'DEPTH_SIGNALS',
]
