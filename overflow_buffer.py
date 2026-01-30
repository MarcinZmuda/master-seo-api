"""
===============================================================================
📦 OVERFLOW BUFFER v1.0
===============================================================================
Automatyczne FAQ/sekcja dla "sierocych" fraz które nie pasują do żadnego H2.

PROBLEM:
- semantic_phrase_assignment przypisuje frazy do H2
- Frazy z niskim relevance score (< 0.3) nie pasują nigdzie
- Obecnie: przypisane na siłę (psuje narrację) lub pominięte (spadek coverage)

ROZWIĄZANIE:
- Wszystkie "sieroce" frazy trafiają do automatycznej sekcji FAQ
- Sekcja "Często zadawane pytania" lub "Warto wiedzieć"
- Frazy wplatane naturalnie w format Q&A

ZYSK:
- 85%+ pokrycia bez psucia głównej narracji
- "Trudne" frazy mają dedykowane miejsce
- Naturalny format Q&A

===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class OrphanPhrase:
    """Fraza-sierota (nie pasuje do żadnego H2)."""
    keyword: str
    kw_type: str  # BASIC, EXTENDED
    actual_uses: int
    target_min: int
    target_max: int
    best_relevance: float  # Najwyższy score do jakiegokolwiek H2
    reason: str  # Dlaczego jest sierotą


@dataclass
class FAQItem:
    """Element FAQ generowany z frazy."""
    question: str
    answer_template: str
    target_phrase: str
    phrase_type: str


@dataclass
class OverflowBuffer:
    """Bufor dla sierocych fraz."""
    orphan_phrases: List[OrphanPhrase] = field(default_factory=list)
    faq_items: List[FAQItem] = field(default_factory=list)
    section_title: str = "Często zadawane pytania"
    section_type: str = "FAQ"  # FAQ, WORTH_KNOWING, SUMMARY


# ============================================================================
# KONFIGURACJA
# ============================================================================

class OverflowConfig:
    # Próg relevance poniżej którego fraza jest "sierotą"
    ORPHAN_RELEVANCE_THRESHOLD: float = 0.30
    
    # Max fraz w sekcji FAQ
    MAX_FAQ_ITEMS: int = 5
    
    # Typy sekcji
    SECTION_TYPES = {
        "FAQ": "Często zadawane pytania",
        "WORTH_KNOWING": "Warto wiedzieć",
        "SUMMARY": "Podsumowanie",
        "ADDITIONAL": "Dodatkowe informacje"
    }
    
    # Szablony pytań dla różnych typów fraz
    QUESTION_TEMPLATES = {
        "prawo": [
            "Czym jest {phrase}?",
            "Jak działa {phrase}?",
            "Kiedy stosuje się {phrase}?",
            "Jakie są konsekwencje {phrase}?",
            "Kto zajmuje się {phrase}?",
        ],
        "medycyna": [
            "Co to jest {phrase}?",
            "Jakie są objawy {phrase}?",
            "Jak leczyć {phrase}?",
        ],
        "general": [
            "Co warto wiedzieć o {phrase}?",
            "Dlaczego {phrase} jest ważne?",
            "Jak rozumieć {phrase}?",
        ]
    }


CONFIG = OverflowConfig()


# ============================================================================
# IDENTYFIKACJA SIEROT
# ============================================================================

def identify_orphan_phrases(
    keywords_state: Dict,
    phrase_assignments: Dict[str, List[Dict]],
    h2_structure: List[str]
) -> List[OrphanPhrase]:
    """
    Identyfikuje frazy które nie pasują dobrze do żadnego H2.
    
    Args:
        keywords_state: Stan wszystkich fraz
        phrase_assignments: Wynik semantic_phrase_assignment
        h2_structure: Lista H2
    
    Returns:
        Lista OrphanPhrase
    """
    orphans = []
    
    # Zbierz wszystkie przypisania
    all_assigned = {}
    for h2, phrases in phrase_assignments.items():
        for p in phrases:
            keyword = p.get("keyword", "")
            relevance = p.get("relevance", 0)
            
            if keyword not in all_assigned or all_assigned[keyword] < relevance:
                all_assigned[keyword] = relevance
    
    # Sprawdź każdą frazę
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "EXTENDED").upper()
        actual = meta.get("actual_uses", 0)
        
        # Pomiń już użyte frazy
        if actual >= meta.get("target_min", 1):
            continue
        
        best_relevance = all_assigned.get(keyword, 0)
        
        # Czy to sierota?
        is_orphan = False
        reason = ""
        
        if best_relevance < CONFIG.ORPHAN_RELEVANCE_THRESHOLD:
            is_orphan = True
            reason = f"Low relevance to all H2 ({best_relevance:.2f})"
        elif best_relevance < 0.4 and kw_type == "BASIC":
            # BASIC z niskim relevance też może być problematyczny
            is_orphan = True
            reason = f"BASIC phrase with weak H2 match ({best_relevance:.2f})"
        
        if is_orphan:
            orphans.append(OrphanPhrase(
                keyword=keyword,
                kw_type=kw_type,
                actual_uses=actual,
                target_min=meta.get("target_min", 1),
                target_max=meta.get("target_max", 10),
                best_relevance=best_relevance,
                reason=reason
            ))
    
    # Sortuj: BASIC first, potem po relevance (najgorsze pierwsze)
    orphans.sort(key=lambda x: (0 if x.kw_type == "BASIC" else 1, x.best_relevance))
    
    return orphans


# ============================================================================
# GENEROWANIE FAQ
# ============================================================================

def generate_faq_for_phrase(
    phrase: OrphanPhrase,
    domain: str = "prawo",
    main_keyword: str = ""
) -> FAQItem:
    """
    Generuje pytanie i szablon odpowiedzi dla frazy.
    """
    keyword = phrase.keyword
    keyword_lower = keyword.lower()
    
    # Wybierz szablon pytania
    templates = CONFIG.QUESTION_TEMPLATES.get(domain, CONFIG.QUESTION_TEMPLATES["general"])
    
    # Dopasuj najlepszy szablon do frazy
    question = select_best_question_template(keyword, templates, domain)
    
    # Wygeneruj szablon odpowiedzi
    answer_template = generate_answer_template(keyword, domain, main_keyword)
    
    return FAQItem(
        question=question,
        answer_template=answer_template,
        target_phrase=keyword,
        phrase_type=phrase.kw_type
    )


def select_best_question_template(keyword: str, templates: List[str], domain: str) -> str:
    """Wybiera najlepszy szablon pytania dla frazy."""
    keyword_lower = keyword.lower()
    
    # Heurystyki dla prawa
    if domain == "prawo":
        if any(w in keyword_lower for w in ["sąd", "procedur", "postępowan"]):
            return f"Jak przebiega {keyword}?"
        if any(w in keyword_lower for w in ["kar", "odpowiedzialn"]):
            return f"Jakie są konsekwencje {keyword}?"
        if any(w in keyword_lower for w in ["prawa", "obowiązk"]):
            return f"Czym jest {keyword}?"
        if any(w in keyword_lower for w in ["art.", "kodeks", "ustaw"]):
            return f"Co reguluje {keyword}?"
    
    # Domyślnie: pierwszy pasujący szablon
    return templates[0].format(phrase=keyword)


def generate_answer_template(keyword: str, domain: str, main_keyword: str) -> str:
    """Generuje szablon odpowiedzi zawierający frazę."""
    keyword_lower = keyword.lower()
    
    if domain == "prawo":
        if "sąd" in keyword_lower:
            return f"{keyword.capitalize()} to organ właściwy w sprawach dotyczących {main_keyword}. {{context}}"
        if "art." in keyword_lower or "kodeks" in keyword_lower:
            return f"{keyword.capitalize()} określa zasady postępowania w omawianej materii. {{context}}"
        if any(w in keyword_lower for w in ["procedur", "postępowan"]):
            return f"{keyword.capitalize()} obejmuje szereg czynności procesowych. {{context}}"
    
    return f"{keyword.capitalize()} jest istotnym elementem w kontekście omawianego tematu. {{context}}"


# ============================================================================
# TWORZENIE SEKCJI FAQ
# ============================================================================

def create_overflow_buffer(
    keywords_state: Dict,
    phrase_assignments: Dict[str, List[Dict]],
    h2_structure: List[str],
    main_keyword: str,
    domain: str = "prawo"
) -> OverflowBuffer:
    """
    Tworzy bufor overflow z sierocymi frazami i FAQ.
    
    Returns:
        OverflowBuffer gotowy do użycia jako dodatkowa sekcja
    """
    # 1. Identyfikuj sieroty
    orphans = identify_orphan_phrases(
        keywords_state=keywords_state,
        phrase_assignments=phrase_assignments,
        h2_structure=h2_structure
    )
    
    if not orphans:
        return OverflowBuffer(
            orphan_phrases=[],
            faq_items=[],
            section_title="",
            section_type="NONE"
        )
    
    # 2. Generuj FAQ dla top sierot
    faq_items = []
    for orphan in orphans[:CONFIG.MAX_FAQ_ITEMS]:
        faq = generate_faq_for_phrase(
            phrase=orphan,
            domain=domain,
            main_keyword=main_keyword
        )
        faq_items.append(faq)
    
    # 3. Wybierz typ sekcji
    if len(faq_items) >= 3:
        section_type = "FAQ"
        section_title = CONFIG.SECTION_TYPES["FAQ"]
    else:
        section_type = "WORTH_KNOWING"
        section_title = CONFIG.SECTION_TYPES["WORTH_KNOWING"]
    
    return OverflowBuffer(
        orphan_phrases=orphans,
        faq_items=faq_items,
        section_title=section_title,
        section_type=section_type
    )


# ============================================================================
# FORMATOWANIE SEKCJI
# ============================================================================

def format_faq_section(buffer: OverflowBuffer) -> str:
    """
    Formatuje sekcję FAQ jako tekst/HTML do wstawienia na końcu artykułu.
    """
    if not buffer.faq_items:
        return ""
    
    lines = []
    lines.append(f"\n## {buffer.section_title}\n")
    
    for faq in buffer.faq_items:
        lines.append(f"### {faq.question}\n")
        lines.append(f"{faq.answer_template}\n")
    
    return "\n".join(lines)


def format_faq_instructions(buffer: OverflowBuffer) -> str:
    """
    Formatuje instrukcje dla agenta do napisania sekcji FAQ.
    """
    if not buffer.faq_items:
        return ""
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"📦 SEKCJA DODATKOWA: {buffer.section_title}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Napisz sekcję FAQ zawierającą poniższe pytania i odpowiedzi.")
    lines.append("W odpowiedziach NATURALNIE wpleć podane frazy kluczowe.")
    lines.append("")
    
    for i, faq in enumerate(buffer.faq_items, 1):
        lines.append(f"❓ PYTANIE {i}: {faq.question}")
        lines.append(f"   📍 FRAZA MUST: \"{faq.target_phrase}\" ({faq.phrase_type})")
        lines.append(f"   💡 SZABLON: {faq.answer_template}")
        lines.append("")
    
    lines.append("✨ WSKAZÓWKI:")
    lines.append("   • Każda odpowiedź: 2-4 zdania")
    lines.append("   • Fraza MUSI pojawić się w odpowiedzi")
    lines.append("   • Naturalny język, bez stuffingu")
    
    return "\n".join(lines)


# ============================================================================
# INTEGRACJA Z BATCH PLANNER
# ============================================================================

def add_faq_batch_if_needed(
    batch_plan: List[Dict],
    overflow_buffer: OverflowBuffer,
    domain: str = "prawo"
) -> List[Dict]:
    """
    Dodaje batch FAQ na końcu planu jeśli są sieroce frazy.
    
    Args:
        batch_plan: Oryginalny plan batchów
        overflow_buffer: Bufor z sierocymi frazami
        domain: Domena
    
    Returns:
        Zaktualizowany plan batchów
    """
    if not overflow_buffer.faq_items:
        return batch_plan
    
    # Znajdź ostatni numer batcha
    last_batch_num = 0
    for batch in batch_plan:
        batch_num = batch.get("batch_number", 0)
        if isinstance(batch_num, int) and batch_num > last_batch_num:
            last_batch_num = batch_num
        elif isinstance(batch_num, str) and batch_num[0].isdigit():
            num = int(re.match(r'\d+', batch_num).group())
            if num > last_batch_num:
                last_batch_num = num
    
    # Utwórz batch FAQ
    faq_batch = {
        "batch_number": last_batch_num + 1,
        "batch_type": "FAQ",
        "h2_sections": [overflow_buffer.section_title],
        "is_faq_batch": True,
        "faq_items": [
            {
                "question": faq.question,
                "target_phrase": faq.target_phrase,
                "phrase_type": faq.phrase_type,
                "answer_template": faq.answer_template
            }
            for faq in overflow_buffer.faq_items
        ],
        "words_min": 150,
        "words_max": 300,
        "overflow_stats": {
            "total_orphans": len(overflow_buffer.orphan_phrases),
            "included_in_faq": len(overflow_buffer.faq_items)
        }
    }
    
    return batch_plan + [faq_batch]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: OVERFLOW BUFFER")
    print("=" * 60)
    
    # Symulacja danych
    keywords_state = {
        "k1": {"keyword": "sąd rodzinny", "type": "BASIC", "actual_uses": 0, "target_min": 3},
        "k2": {"keyword": "władza rodzicielska", "type": "BASIC", "actual_uses": 2, "target_min": 2},
        "k3": {"keyword": "Konwencja haska", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
        "k4": {"keyword": "kurator sądowy", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
        "k5": {"keyword": "mediacja rodzinna", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
    }
    
    # Symulacja przypisań (niektóre frazy z niskim relevance)
    phrase_assignments = {
        "Czym jest porwanie": [
            {"keyword": "sąd rodzinny", "relevance": 0.45},  # OK
        ],
        "Procedura sądowa": [
            {"keyword": "władza rodzicielska", "relevance": 0.55},  # OK
        ],
    }
    # Konwencja haska, kurator sądowy, mediacja rodzinna - nie przypisane (sieroty)
    
    h2_structure = ["Czym jest porwanie", "Procedura sądowa"]
    
    # Test
    buffer = create_overflow_buffer(
        keywords_state=keywords_state,
        phrase_assignments=phrase_assignments,
        h2_structure=h2_structure,
        main_keyword="porwanie rodzicielskie",
        domain="prawo"
    )
    
    print(f"\n📊 Wynik:")
    print(f"   Sieroty znalezione: {len(buffer.orphan_phrases)}")
    print(f"   FAQ items: {len(buffer.faq_items)}")
    print(f"   Typ sekcji: {buffer.section_type}")
    
    if buffer.orphan_phrases:
        print(f"\n⚠️ Sieroce frazy:")
        for orphan in buffer.orphan_phrases:
            print(f"   • \"{orphan.keyword}\" ({orphan.kw_type}) - {orphan.reason}")
    
    if buffer.faq_items:
        print(f"\n📝 Wygenerowane FAQ:")
        for faq in buffer.faq_items:
            print(f"   ❓ {faq.question}")
            print(f"      Fraza: {faq.target_phrase}")
    
    print("\n" + "=" * 60)
    print("INSTRUKCJE DLA AGENTA:")
    print("=" * 60)
    print(format_faq_instructions(buffer))
