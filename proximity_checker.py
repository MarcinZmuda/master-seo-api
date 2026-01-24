"""
🆕 v36.8: PROXIMITY CHECKER - Wymuszanie bliskości encji

Sprawdza i wymusza:
- Entity proximity (encje w tym samym zdaniu/akapicie)
- Keyword clustering (powiązane słowa kluczowe blisko siebie)
- Context windows dla fraz

Autor: Claude
Wersja: 36.8
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

# ================================================================
# KONFIGURACJA
# ================================================================

@dataclass
class ProximityConfig:
    """Konfiguracja proximity checker."""
    
    # Progi proximity (w słowach)
    SAME_SENTENCE_THRESHOLD: int = 0       # W tym samym zdaniu
    CLOSE_PROXIMITY_THRESHOLD: int = 30    # Blisko (max 30 słów)
    MEDIUM_PROXIMITY_THRESHOLD: int = 75   # Średnia odległość
    FAR_THRESHOLD: int = 150               # Daleko (>150 słów = słabe powiązanie)
    
    # Wymagania
    REQUIRE_ENTITY_PAIRS_SAME_SENTENCE: bool = True  # Wymuszaj pary encji w tym samym zdaniu
    REQUIRE_KEYWORD_CONTEXT: bool = True   # Wymuszaj kontekst dla keywords
    
    # Wagi dla proximity score
    SAME_SENTENCE_SCORE: float = 1.0
    CLOSE_SCORE: float = 0.7
    MEDIUM_SCORE: float = 0.4
    FAR_SCORE: float = 0.1

CONFIG = ProximityConfig()

# ================================================================
# TEXT ANALYSIS HELPERS
# ================================================================

def split_into_sentences(text: str) -> List[str]:
    """Dzieli tekst na zdania."""
    # Pattern dla polskich zdań
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])', text)
    return [s.strip() for s in sentences if s.strip()]

def split_into_paragraphs(text: str) -> List[str]:
    """Dzieli tekst na akapity."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def find_positions(text: str, phrase: str) -> List[int]:
    """
    Znajduje pozycje (w słowach) wszystkich wystąpień frazy.
    
    Returns:
        Lista pozycji (indeks słowa gdzie zaczyna się fraza)
    """
    if not text or not phrase:
        return []
    
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    words = text_lower.split()
    phrase_words = phrase_lower.split()
    
    positions = []
    phrase_len = len(phrase_words)
    
    for i in range(len(words) - phrase_len + 1):
        if words[i:i + phrase_len] == phrase_words:
            positions.append(i)
    
    # Fallback: szukaj jako substring i konwertuj na pozycję słowa
    if not positions:
        start = 0
        while True:
            pos = text_lower.find(phrase_lower, start)
            if pos == -1:
                break
            # Konwertuj pozycję znaku na pozycję słowa
            word_pos = len(text_lower[:pos].split())
            positions.append(word_pos)
            start = pos + len(phrase_lower)
    
    return positions

def get_sentence_containing_position(text: str, word_position: int) -> Tuple[int, str]:
    """
    Znajduje zdanie zawierające daną pozycję słowa.
    
    Returns:
        (sentence_index, sentence_text)
    """
    sentences = split_into_sentences(text)
    words = text.split()
    
    current_word_idx = 0
    for sent_idx, sentence in enumerate(sentences):
        sent_words = len(sentence.split())
        if current_word_idx <= word_position < current_word_idx + sent_words:
            return sent_idx, sentence
        current_word_idx += sent_words
    
    return -1, ""

# ================================================================
# PROXIMITY CALCULATION
# ================================================================

@dataclass
class ProximityResult:
    """Wynik analizy proximity."""
    entity1: str
    entity2: str
    distance_words: int
    same_sentence: bool
    same_paragraph: bool
    proximity_score: float  # 0.0 - 1.0
    status: str  # EXCELLENT, GOOD, FAIR, POOR
    positions: Dict[str, List[int]]

def calculate_proximity(
    text: str,
    phrase1: str,
    phrase2: str
) -> ProximityResult:
    """
    Oblicza proximity między dwoma frazami.
    
    Args:
        text: Tekst do analizy
        phrase1: Pierwsza fraza
        phrase2: Druga fraza
        
    Returns:
        ProximityResult
    """
    pos1 = find_positions(text, phrase1)
    pos2 = find_positions(text, phrase2)
    
    if not pos1 or not pos2:
        return ProximityResult(
            entity1=phrase1,
            entity2=phrase2,
            distance_words=-1,
            same_sentence=False,
            same_paragraph=False,
            proximity_score=0.0,
            status="NOT_FOUND",
            positions={"phrase1": pos1, "phrase2": pos2}
        )
    
    # Znajdź minimalną odległość
    min_distance = float('inf')
    best_pos1 = pos1[0]
    best_pos2 = pos2[0]
    
    for p1 in pos1:
        for p2 in pos2:
            dist = abs(p1 - p2)
            if dist < min_distance:
                min_distance = dist
                best_pos1 = p1
                best_pos2 = p2
    
    # Sprawdź czy w tym samym zdaniu
    sent1_idx, sent1 = get_sentence_containing_position(text, best_pos1)
    sent2_idx, sent2 = get_sentence_containing_position(text, best_pos2)
    same_sentence = (sent1_idx == sent2_idx and sent1_idx >= 0)
    
    # Sprawdź czy w tym samym akapicie
    paragraphs = split_into_paragraphs(text)
    para1_idx = -1
    para2_idx = -1
    word_count = 0
    
    for para_idx, para in enumerate(paragraphs):
        para_words = len(para.split())
        if word_count <= best_pos1 < word_count + para_words:
            para1_idx = para_idx
        if word_count <= best_pos2 < word_count + para_words:
            para2_idx = para_idx
        word_count += para_words
    
    same_paragraph = (para1_idx == para2_idx and para1_idx >= 0)
    
    # Oblicz proximity score
    if same_sentence:
        score = CONFIG.SAME_SENTENCE_SCORE
        status = "EXCELLENT"
    elif min_distance <= CONFIG.CLOSE_PROXIMITY_THRESHOLD:
        score = CONFIG.CLOSE_SCORE
        status = "GOOD"
    elif min_distance <= CONFIG.MEDIUM_PROXIMITY_THRESHOLD:
        score = CONFIG.MEDIUM_SCORE
        status = "FAIR"
    else:
        score = CONFIG.FAR_SCORE
        status = "POOR"
    
    return ProximityResult(
        entity1=phrase1,
        entity2=phrase2,
        distance_words=min_distance,
        same_sentence=same_sentence,
        same_paragraph=same_paragraph,
        proximity_score=score,
        status=status,
        positions={"phrase1": pos1, "phrase2": pos2}
    )

# ================================================================
# ENTITY PAIRS ANALYSIS
# ================================================================

# Pary encji które powinny występować blisko siebie
REQUIRED_ENTITY_PAIRS = {
    # Prawo
    ("sąd okręgowy", "wydział cywilny"): "legal_court",
    ("sąd rejonowy", "wydział rodzinny"): "legal_court",
    ("kodeks cywilny", "art."): "legal_reference",
    ("kodeks karny", "art."): "legal_reference",
    ("wniosek", "sąd"): "legal_procedure",
    
    # Medycyna
    ("choroba psychiczna", "biegły"): "medical_expert",
    ("opinia", "psychiatra"): "medical_expert",
    ("badanie", "lekarz"): "medical_exam",
    
    # Finanse
    ("podatek", "urząd skarbowy"): "tax_authority",
    ("pit", "zeznanie"): "tax_form",
}

def analyze_entity_pairs(
    text: str,
    entities: List[str],
    custom_pairs: Optional[Dict[Tuple[str, str], str]] = None
) -> Dict[str, Any]:
    """
    Analizuje proximity dla par encji.
    
    Args:
        text: Tekst do analizy
        entities: Lista encji do sprawdzenia
        custom_pairs: Dodatkowe wymagane pary
        
    Returns:
        Analiza par encji
    """
    pairs_to_check = REQUIRED_ENTITY_PAIRS.copy()
    if custom_pairs:
        pairs_to_check.update(custom_pairs)
    
    results = []
    issues = []
    
    text_lower = text.lower()
    entities_lower = [e.lower() for e in entities]
    
    for (e1, e2), pair_type in pairs_to_check.items():
        e1_lower = e1.lower()
        e2_lower = e2.lower()
        
        # Sprawdź czy obie encje są w tekście
        e1_present = e1_lower in text_lower or any(e1_lower in ent for ent in entities_lower)
        e2_present = e2_lower in text_lower or any(e2_lower in ent for ent in entities_lower)
        
        if e1_present and e2_present:
            proximity = calculate_proximity(text, e1, e2)
            results.append({
                "pair": (e1, e2),
                "type": pair_type,
                "proximity": proximity.proximity_score,
                "distance": proximity.distance_words,
                "same_sentence": proximity.same_sentence,
                "status": proximity.status
            })
            
            # Dodaj issue jeśli proximity jest słabe
            if proximity.status in ["FAIR", "POOR"]:
                issues.append({
                    "type": "WEAK_ENTITY_PROXIMITY",
                    "entity1": e1,
                    "entity2": e2,
                    "distance": proximity.distance_words,
                    "recommendation": f"Umieść '{e1}' i '{e2}' bliżej siebie (najlepiej w tym samym zdaniu)"
                })
    
    return {
        "pairs_checked": len(results),
        "pairs_found": results,
        "issues": issues,
        "avg_proximity_score": sum(r["proximity"] for r in results) / len(results) if results else 0
    }

# ================================================================
# KEYWORD CONTEXT VALIDATION
# ================================================================

def validate_keyword_context(
    text: str,
    keyword: str,
    required_context_words: List[str],
    context_window: int = 50
) -> Dict[str, Any]:
    """
    Sprawdza czy keyword występuje w odpowiednim kontekście.
    
    Args:
        text: Tekst do analizy
        keyword: Słowo kluczowe
        required_context_words: Słowa które powinny być w pobliżu
        context_window: Okno kontekstowe (w słowach)
        
    Returns:
        Wynik walidacji kontekstu
    """
    keyword_positions = find_positions(text, keyword)
    
    if not keyword_positions:
        return {
            "keyword": keyword,
            "found": False,
            "context_valid": False,
            "missing_context": required_context_words
        }
    
    words = text.lower().split()
    found_context = set()
    missing_context = set(w.lower() for w in required_context_words)
    
    for kw_pos in keyword_positions:
        # Sprawdź okno kontekstowe
        start = max(0, kw_pos - context_window)
        end = min(len(words), kw_pos + context_window)
        context_words = set(words[start:end])
        
        for ctx_word in required_context_words:
            if ctx_word.lower() in context_words:
                found_context.add(ctx_word.lower())
                missing_context.discard(ctx_word.lower())
    
    return {
        "keyword": keyword,
        "found": True,
        "occurrences": len(keyword_positions),
        "context_valid": len(missing_context) == 0,
        "found_context": list(found_context),
        "missing_context": list(missing_context),
        "context_window": context_window
    }

# ================================================================
# PROXIMITY ENFORCEMENT FOR BATCHES
# ================================================================

def enforce_proximity_requirements(
    batch_text: str,
    entities: List[str],
    keywords: List[str],
    detected_category: str = "general"
) -> Dict[str, Any]:
    """
    Wymusza wymagania proximity dla batcha.
    
    Args:
        batch_text: Tekst batcha
        entities: Encje w batchu
        keywords: Keywords w batchu
        detected_category: Kategoria tematyczna
        
    Returns:
        Wyniki enforcement z issues i recommendations
    """
    results = {
        "entity_pairs": analyze_entity_pairs(batch_text, entities),
        "proximity_issues": [],
        "recommendations": [],
        "overall_score": 0.0
    }
    
    # Zbierz issues z entity pairs
    results["proximity_issues"].extend(results["entity_pairs"]["issues"])
    
    # Dodatkowe sprawdzenia dla kategorii prawnej
    if detected_category == "prawo":
        # Sprawdź czy artykuły kodeksu są blisko nazwy kodeksu
        legal_context = validate_keyword_context(
            batch_text,
            "art.",
            ["kodeks", "k.c.", "k.k.", "k.p.c.", "ustawa"],
            context_window=20
        )
        
        if legal_context["found"] and not legal_context["context_valid"]:
            results["proximity_issues"].append({
                "type": "LEGAL_CONTEXT_MISSING",
                "keyword": "art.",
                "missing": legal_context["missing_context"],
                "recommendation": "Artykuły powinny mieć odniesienie do konkretnego kodeksu/ustawy w tym samym zdaniu"
            })
    
    # Generuj recommendations
    for issue in results["proximity_issues"]:
        if "recommendation" in issue:
            results["recommendations"].append(issue["recommendation"])
    
    # Oblicz overall score
    entity_score = results["entity_pairs"]["avg_proximity_score"]
    issues_penalty = len(results["proximity_issues"]) * 0.1
    results["overall_score"] = max(0, min(1.0, entity_score - issues_penalty))
    
    return results

# ================================================================
# PROXIMITY SUGGESTIONS FOR GPT PROMPT
# ================================================================

def generate_proximity_instructions(
    entities: List[str],
    keywords: List[str],
    detected_category: str = "general"
) -> List[str]:
    """
    Generuje instrukcje proximity dla GPT.
    
    Args:
        entities: Encje do użycia
        keywords: Keywords do użycia
        detected_category: Kategoria tematyczna
        
    Returns:
        Lista instrukcji dla GPT
    """
    instructions = []
    
    # Instrukcje ogólne
    instructions.append("📍 PROXIMITY - Bliskość fraz:")
    
    # Znajdź pary encji które powinny być blisko
    entities_lower = [e.lower() for e in entities]
    
    pairs_found = []
    for (e1, e2), pair_type in REQUIRED_ENTITY_PAIRS.items():
        e1_match = any(e1.lower() in ent for ent in entities_lower)
        e2_match = any(e2.lower() in ent for ent in entities_lower)
        
        if e1_match or e2_match:
            # Znajdź pełne nazwy encji
            e1_full = next((e for e in entities if e1.lower() in e.lower()), e1)
            e2_full = next((e for e in entities if e2.lower() in e.lower()), e2)
            pairs_found.append((e1_full, e2_full, pair_type))
    
    if pairs_found:
        for e1, e2, ptype in pairs_found[:5]:  # Max 5 par
            instructions.append(f"   • Umieść '{e1}' i '{e2}' w tym samym zdaniu")
    
    # Instrukcje dla kategorii prawnej
    if detected_category == "prawo":
        instructions.append("   • Cytując artykuł, ZAWSZE podaj źródło (np. 'art. 13 k.c.')")
        instructions.append("   • Nazwy sądów pisz pełne (np. 'Sąd Okręgowy w Warszawie')")
    
    # Instrukcje dla kategorii medycznej
    if detected_category == "medycyna":
        instructions.append("   • Terminy medyczne wyjaśniaj w nawiasie przy pierwszym użyciu")
        instructions.append("   • Opinie biegłych łącz z ich specjalizacją")
    
    return instructions

# ================================================================
# TESTING
# ================================================================

def test_proximity_checker():
    """Test proximity checker."""
    print("="*60)
    print("PROXIMITY CHECKER TEST")
    print("="*60)
    
    test_text = """
    Wniosek o ubezwłasnowolnienie składa się do Sądu Okręgowego.
    Wydział Cywilny rozpatruje takie sprawy w trybie nieprocesowym.
    
    Sąd powołuje biegłego psychiatrę do wydania opinii.
    Choroba psychiczna musi być potwierdzona badaniem.
    
    Zgodnie z art. 13 Kodeksu cywilnego, osoba ubezwłasnowolniona całkowicie
    nie ma zdolności do czynności prawnych.
    """
    
    print("\n1. Proximity between phrases:")
    result = calculate_proximity(test_text, "Sąd Okręgowy", "Wydział Cywilny")
    print(f"   'Sąd Okręgowy' <-> 'Wydział Cywilny':")
    print(f"      Distance: {result.distance_words} words")
    print(f"      Same sentence: {result.same_sentence}")
    print(f"      Score: {result.proximity_score}")
    print(f"      Status: {result.status}")
    
    print("\n2. Entity pairs analysis:")
    entities = ["Sąd Okręgowy", "Wydział Cywilny", "biegły psychiatra", "choroba psychiczna"]
    analysis = analyze_entity_pairs(test_text, entities)
    print(f"   Pairs found: {analysis['pairs_checked']}")
    print(f"   Issues: {len(analysis['issues'])}")
    for issue in analysis["issues"]:
        print(f"      - {issue['type']}: {issue['entity1']} <-> {issue['entity2']}")
    
    print("\n3. Legal context validation:")
    ctx = validate_keyword_context(test_text, "art.", ["kodeks", "k.c."], context_window=10)
    print(f"   'art.' context valid: {ctx['context_valid']}")
    print(f"   Found context: {ctx['found_context']}")
    
    print("\n4. GPT instructions:")
    instructions = generate_proximity_instructions(entities, [], "prawo")
    for instr in instructions:
        print(f"   {instr}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_proximity_checker()
