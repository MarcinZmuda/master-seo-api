"""
===============================================================================
ENTITY SCORING v40.2
===============================================================================
Kompleksowy scoring encji zgodny z trendami Semantic SEO 2025.

METRYKI:
1. Entity Coverage - % encji z S1 obecnych w tekście
2. Entity Density - encje / 100 słów (optimal: 3-5)
3. Entity Relationships - czy encje są połączone (S-V-O)
4. Entity Salience - prominencja głównej encji
5. Entity Definitions - czy encje są wyjaśnione

RESEARCH BASIS:
- Google Knowledge Graph: 8B entities (2024)
- 87% SERP results contain entity-linked rich snippets
- Case study: 1400% visibility increase with entity optimization

UŻYCIE:
    from entity_scoring import (
        calculate_entity_score,
        calculate_entity_coverage,
        calculate_entity_density,
        analyze_entity_relationships
    )
    
    score = calculate_entity_score(text, s1_entities, main_keyword)
===============================================================================
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

# 🆕 v41.0: Import TRIPLET PRIORITY SYSTEM
from triplet_priority_v41 import (
    analyze_triplets_with_priority,
    prioritize_triplets,
    get_triplet_instructions_for_prebatch
)


# ============================================================
# KONFIGURACJA
# ============================================================

@dataclass
class EntityScoringConfig:
    """Konfiguracja scoringu encji."""
    
    # Wagi komponentów
    WEIGHT_COVERAGE: float = 0.30      # Pokrycie encji z S1
    WEIGHT_DENSITY: float = 0.25       # Gęstość encji
    WEIGHT_RELATIONSHIPS: float = 0.25 # Relacje między encjami
    WEIGHT_SALIENCE: float = 0.20      # Prominencja głównej encji
    
    # Progi gęstości (encje / 100 słów)
    DENSITY_MIN: float = 2.0           # Minimum akceptowalne
    DENSITY_OPTIMAL_MIN: float = 3.0   # Początek optymalnego zakresu
    DENSITY_OPTIMAL_MAX: float = 5.0   # Koniec optymalnego zakresu
    DENSITY_MAX: float = 7.0           # Maximum (powyżej = stuffing)
    
    # Progi pokrycia
    COVERAGE_EXCELLENT: float = 0.80   # 80%+ encji z S1
    COVERAGE_GOOD: float = 0.60        # 60%+ encji
    COVERAGE_ACCEPTABLE: float = 0.40  # 40%+ encji
    
    # Minimum encji do analizy
    MIN_S1_ENTITIES: int = 5


DEFAULT_CONFIG = EntityScoringConfig()


# ============================================================
# ENTITY EXTRACTION (lightweight)
# ============================================================

def extract_entities_simple(text: str, known_entities: List[str] = None) -> List[Dict]:
    """
    Prosta ekstrakcja encji bez NLP (pattern matching).
    
    Args:
        text: Tekst do analizy
        known_entities: Lista znanych encji do szukania
        
    Returns:
        List of {"name": str, "count": int, "positions": list}
    """
    if not text:
        return []
    
    text_lower = text.lower()
    entities = []
    
    if known_entities:
        for entity in known_entities:
            entity_lower = entity.lower()
            # Znajdź wszystkie wystąpienia
            pattern = rf'\b{re.escape(entity_lower)}\b'
            matches = list(re.finditer(pattern, text_lower))
            
            if matches:
                entities.append({
                    "name": entity,
                    "count": len(matches),
                    "positions": [m.start() for m in matches],
                    "first_position": matches[0].start(),
                    "density": len(matches) / (len(text_lower.split()) / 100)
                })
    
    return entities


def extract_capitalized_phrases(text: str) -> List[str]:
    """
    Ekstrahuje frazy z wielkimi literami (potencjalne named entities).
    """
    # Pattern dla fraz z wielkimi literami (2-4 słowa)
    pattern = r'\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+){0,3})\b'
    matches = re.findall(pattern, text)
    
    # Filtruj rozpoczynające zdanie (po . ! ?)
    filtered = []
    for match in matches:
        # Sprawdź czy nie jest na początku zdania
        idx = text.find(match)
        if idx > 0 and text[idx-2:idx].strip() not in '.!?':
            filtered.append(match)
    
    return list(set(filtered))


# ============================================================
# v50.5 FIX 32: POLISH INFLECTION-AWARE ENTITY MATCHING
# ============================================================
# Polish has rich morphology — entity names appear in many forms:
#   "prąd elektryczny" → "prądu elektrycznego", "prądem elektrycznym"
#   "prawo Ohma" → "prawem Ohma", "prawa Ohma"
#   "obwód elektryczny" → "obwodzie elektrycznym", "obwodu elektrycznego"
#
# Simple regex with [aąeęioóuy]? suffix fails for these cases.
# Solution: stem-based matching — take 60-75% of each word as stem,
# then check if all stems appear close together in text.

def _polish_normalize(text: str) -> str:
    """Normalize Polish vowel alternations for matching.
    
    Polish has systematic vowel changes in inflection:
      ó → o (opór → oporu, wzór → wzoru, obwód → obwodu)
      ą → ę (not systematic but helps matching)
    This normalizes both forms to the same representation.
    """
    return text.replace("ó", "o").replace("Ó", "O")


def _polish_stem(word: str, min_len: int = 3) -> str:
    """Create a crude Polish stem by removing likely inflection suffix.
    
    Takes ~60-75% of the word. For short words (≤4 chars), takes all but last char.
    This is NOT a proper stemmer but sufficient for entity matching.
    
    Examples:
        elektrycznego → elektrycz (9/13)
        elektryczny → elektrycz (9/11)
        prądu → prą (3/5)
        prądem → prąd (4/6)
        Ohma → Ohm (3/4)
        transformator → transformat (10/13)
        izolator → izolat (6/8)
    """
    if not word or len(word) <= min_len:
        return word
    
    # Remove common Polish suffixes explicitly first
    _SUFFIXES = [
        # Adjective endings (longest first)
        "ycznego", "icznego", "ycznej", "icznej",
        "ycznym", "icznym", "ycznych", "icznych",
        "owego", "owej", "owym", "owych", "ową",
        "iego", "iej", "imi", "iem",
        "nego", "nej", "nym", "nych", "ną",
        "ego", "emu", "ej",
        # Noun endings
        "owi", "ami", "ach",
        "ów", "om",
        "em", "ie",
        "ą", "ę", "u", "y", "i", "a", "e", "o",
    ]
    
    w_lower = word.lower()
    for suffix in _SUFFIXES:
        if w_lower.endswith(suffix) and len(w_lower) - len(suffix) >= min_len:
            return word[:len(word) - len(suffix)]
    
    # Fallback: take ~70% of the word
    stem_len = max(min_len, int(len(word) * 0.7))
    return word[:stem_len]


def _polish_entity_match(entity_name: str, text: str) -> bool:
    """Check if a Polish entity appears in text, accounting for inflection.
    
    For single-word entities: check if stem appears in text.
    For multi-word entities: check if ALL word stems appear within 60 chars of each other.
    
    Args:
        entity_name: Entity name in lowercase (e.g. "prąd elektryczny")
        text: Full article text in lowercase
        
    Returns:
        True if entity is found in text (any inflected form)
    """
    # 1. Try exact match first (fastest)
    if entity_name in text:
        return True
    
    # 2. Normalize ó→o for alternation matching
    norm_entity = _polish_normalize(entity_name)
    norm_text = _polish_normalize(text)
    if norm_entity in norm_text:
        return True
    
    words = entity_name.split()
    
    if len(words) == 1:
        # Single word: check if normalized stem appears
        stem = _polish_normalize(_polish_stem(words[0]).lower())
        if len(stem) < 3:
            return entity_name in text
        return stem in norm_text
    
    # Multi-word entity: check if all stems appear close together
    stems = []
    for w in words:
        if len(w) <= 2:
            continue  # Skip very short words (prepositions etc.)
        stem = _polish_normalize(_polish_stem(w).lower())
        if len(stem) >= 3:
            stems.append(stem)
    
    if not stems:
        return entity_name in text
    
    # All stems must exist in normalized text
    for stem in stems:
        if stem not in norm_text:
            return False
    
    # Check proximity: find all positions of first stem, check if others are nearby
    first_stem = stems[0]
    start = 0
    while True:
        pos = norm_text.find(first_stem, start)
        if pos == -1:
            break
        
        # Check 80-char window around this position for all other stems
        window_start = max(0, pos - 10)
        window_end = min(len(norm_text), pos + 80)
        window = norm_text[window_start:window_end]
        
        if all(s in window for s in stems):
            return True
        
        start = pos + 1
    
    return False


def _polish_entity_find_all(entity_name: str, text: str) -> list:
    """Find ALL occurrences of a Polish entity in text, accounting for inflection.
    
    Returns list of match positions (start index in text).
    Handles Polish morphology: "jazda po alkoholu" matches
    "jazdę po alkoholu", "jazdy po alkoholu", "jazdą po alkoholu" etc.
    
    Args:
        entity_name: Entity name in lowercase
        text: Full text in lowercase
        
    Returns:
        List of start positions where entity was found
    """
    positions = []
    
    # 1. Exact matches first
    start = 0
    while True:
        pos = text.find(entity_name, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    # 2. Normalized (ó→o) matches
    norm_entity = _polish_normalize(entity_name)
    norm_text = _polish_normalize(text)
    if norm_entity != entity_name:
        start = 0
        while True:
            pos = norm_text.find(norm_entity, start)
            if pos == -1:
                break
            if pos not in positions:
                positions.append(pos)
            start = pos + 1
    
    # 3. Stem-based matching for inflected forms
    words = entity_name.split()
    stems = []
    for w in words:
        if len(w) <= 2:
            continue  # Skip prepositions
        stem = _polish_normalize(_polish_stem(w).lower())
        if len(stem) >= 3:
            stems.append(stem)
    
    if stems:
        # Find all positions of the first stem
        first_stem = stems[0]
        start = 0
        while True:
            pos = norm_text.find(first_stem, start)
            if pos == -1:
                break
            
            # Check if all other stems are within 80 chars
            if len(stems) == 1:
                # Single-stem entity
                if pos not in positions:
                    positions.append(pos)
            else:
                window_start = max(0, pos - 10)
                window_end = min(len(norm_text), pos + 80)
                window = norm_text[window_start:window_end]
                
                if all(s in window for s in stems):
                    if pos not in positions:
                        positions.append(pos)
            
            start = pos + 1
    
    positions.sort()
    return positions


# ============================================================
# COVERAGE CALCULATION
# ============================================================

def calculate_entity_coverage(
    text: str, 
    s1_entities: List[Dict],
    config: EntityScoringConfig = None
) -> Dict[str, Any]:
    """
    Oblicza pokrycie encji z analizy S1.
    
    Args:
        text: Tekst do analizy
        s1_entities: Encje z S1 [{"name": str, "importance": float}, ...]
        config: Konfiguracja
        
    Returns:
        Dict z: score, coverage_ratio, found, missing, by_importance
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not text or not s1_entities:
        return {
            "score": 0,
            "coverage_ratio": 0.0,
            "found": [],
            "missing": [],
            "status": "NO_DATA"
        }
    
    text_lower = text.lower()
    
    # Przygotuj encje do szukania
    entities_to_find = []
    for e in s1_entities:
        if isinstance(e, dict):
            name = e.get("name", e.get("entity", ""))
            importance = e.get("importance", e.get("weight", 0.5))
        else:
            name = str(e)
            importance = 0.5
        
        if name:
            entities_to_find.append({
                "name": name,
                "importance": importance
            })
    
    if not entities_to_find:
        return {
            "score": 0,
            "coverage_ratio": 0.0,
            "found": [],
            "missing": [],
            "status": "NO_ENTITIES"
        }
    
    # Szukaj encji
    found = []
    missing = []
    
    for entity in entities_to_find:
        name_lower = entity["name"].lower()
        
        # v50.5 FIX 32: Polish inflection-aware entity matching
        # Polish has rich morphology — "prąd elektryczny" appears as
        # "prądu elektrycznego", "prądem elektrycznym", etc.
        # Simple suffix matching fails. Use stem-based approach instead.
        entity_found = _polish_entity_match(name_lower, text_lower)
        
        if entity_found:
            found.append(entity)
        else:
            missing.append(entity)
    
    # Oblicz coverage
    total = len(entities_to_find)
    found_count = len(found)
    coverage_ratio = found_count / total if total > 0 else 0
    
    # Ważone pokrycie (uwzględnia importance)
    weighted_found = sum(e["importance"] for e in found)
    weighted_total = sum(e["importance"] for e in entities_to_find)
    weighted_coverage = weighted_found / weighted_total if weighted_total > 0 else 0
    
    # Score (0-100)
    if weighted_coverage >= config.COVERAGE_EXCELLENT:
        score = 85 + (weighted_coverage - config.COVERAGE_EXCELLENT) * 75
    elif weighted_coverage >= config.COVERAGE_GOOD:
        score = 65 + (weighted_coverage - config.COVERAGE_GOOD) * 100
    elif weighted_coverage >= config.COVERAGE_ACCEPTABLE:
        score = 40 + (weighted_coverage - config.COVERAGE_ACCEPTABLE) * 125
    else:
        score = weighted_coverage * 100
    
    score = min(100, max(0, score))
    
    # Status
    if weighted_coverage >= config.COVERAGE_EXCELLENT:
        status = "EXCELLENT"
    elif weighted_coverage >= config.COVERAGE_GOOD:
        status = "GOOD"
    elif weighted_coverage >= config.COVERAGE_ACCEPTABLE:
        status = "ACCEPTABLE"
    else:
        status = "LOW"
    
    return {
        "score": round(score),
        "coverage_ratio": round(coverage_ratio, 3),
        "weighted_coverage": round(weighted_coverage, 3),
        "found_count": found_count,
        "total_count": total,
        "found": [e["name"] for e in found],
        "missing": [e["name"] for e in missing],
        "missing_important": [e["name"] for e in missing if e["importance"] >= 0.7],
        "status": status
    }


# ============================================================
# DENSITY CALCULATION
# ============================================================

def calculate_entity_density(
    text: str,
    s1_entities: List[Dict] = None,
    config: EntityScoringConfig = None
) -> Dict[str, Any]:
    """
    Oblicza gęstość encji w tekście.
    
    Optimal: 3-5 encji na 100 słów
    
    Args:
        text: Tekst do analizy
        s1_entities: Opcjonalna lista encji do szukania
        config: Konfiguracja
        
    Returns:
        Dict z: density, score, status, entity_mentions
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not text:
        return {
            "density": 0.0,
            "score": 0,
            "status": "NO_DATA"
        }
    
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            "density": 0.0,
            "score": 0,
            "status": "NO_TEXT"
        }
    
    # Liczenie encji
    entity_mentions = 0
    entities_found = []
    
    if s1_entities:
        # Szukaj znanych encji — v59.1: use Polish inflection-aware matching
        text_lower = text.lower()
        for entity in s1_entities:
            name = entity.get("name", entity) if isinstance(entity, dict) else str(entity)
            name_lower = name.lower()
            
            # v59.1 FIX: Use inflection-aware matching instead of exact regex
            match_positions = _polish_entity_find_all(name_lower, text_lower)
            
            if match_positions:
                entity_mentions += len(match_positions)
                entities_found.append({
                    "name": name,
                    "count": len(match_positions)
                })
    else:
        # Fallback: szukaj kapitalizowanych fraz
        capitalized = extract_capitalized_phrases(text)
        entity_mentions = len(capitalized)
        entities_found = [{"name": c, "count": 1} for c in capitalized[:20]]
    
    # Oblicz gęstość (na 100 słów)
    density = (entity_mentions / word_count) * 100
    
    # Score (0-100)
    if density < config.DENSITY_MIN:
        score = int(density / config.DENSITY_MIN * 40)
        status = "LOW"
    elif density < config.DENSITY_OPTIMAL_MIN:
        score = 40 + int((density - config.DENSITY_MIN) / (config.DENSITY_OPTIMAL_MIN - config.DENSITY_MIN) * 20)
        status = "SUBOPTIMAL"
    elif density <= config.DENSITY_OPTIMAL_MAX:
        # Optimal range
        optimal_mid = (config.DENSITY_OPTIMAL_MIN + config.DENSITY_OPTIMAL_MAX) / 2
        if density <= optimal_mid:
            score = 70 + int((density - config.DENSITY_OPTIMAL_MIN) / (optimal_mid - config.DENSITY_OPTIMAL_MIN) * 15)
        else:
            score = 85 + int((config.DENSITY_OPTIMAL_MAX - density) / (config.DENSITY_OPTIMAL_MAX - optimal_mid) * 15)
        status = "OPTIMAL"
    elif density <= config.DENSITY_MAX:
        score = 60 + int((config.DENSITY_MAX - density) / (config.DENSITY_MAX - config.DENSITY_OPTIMAL_MAX) * 20)
        status = "HIGH"
    else:
        score = max(20, 60 - int((density - config.DENSITY_MAX) * 5))
        status = "STUFFING"
    
    score = min(100, max(0, score))
    
    return {
        "density": round(density, 2),
        "entity_mentions": entity_mentions,
        "word_count": word_count,
        "score": score,
        "status": status,
        "entities_found": entities_found[:15],
        "recommendation": _get_density_recommendation(density, config)
    }


def _get_density_recommendation(density: float, config: EntityScoringConfig) -> str:
    """Zwraca rekomendację dla gęstości encji."""
    if density < config.DENSITY_MIN:
        return f"Za mało encji ({density:.1f}/100 słów). Dodaj definicje i wyjaśnienia kluczowych pojęć."
    elif density < config.DENSITY_OPTIMAL_MIN:
        return f"Gęstość encji poniżej optymalnej ({density:.1f}/100 słów). Rozważ dodanie więcej encji."
    elif density <= config.DENSITY_OPTIMAL_MAX:
        return ""  # OK
    elif density <= config.DENSITY_MAX:
        return f"Wysoka gęstość encji ({density:.1f}/100 słów). Może utrudniać czytanie."
    else:
        return f"Zbyt wysoka gęstość encji ({density:.1f}/100 słów) - tekst może być trudny w odbiorze."


# ============================================================
# RELATIONSHIP ANALYSIS
# ============================================================

def analyze_entity_relationships(
    text: str,
    s1_relationships: List[Dict] = None,
    s1_entities: List[Dict] = None
) -> Dict[str, Any]:
    """
    🆕 v41.0: Analizuje relacje między encjami z priorytetyzacją MUST/SHOULD/NICE.
    
    Args:
        text: Tekst do analizy
        s1_relationships: Relacje z S1 [{"subject": str, "verb": str, "object": str}, ...]
        s1_entities: Encje z S1 [{"name", "importance", "sources_count"}, ...]
        
    Returns:
        Dict z: score, found_relationships, missing_relationships, prioritized, prebatch_instruction
    """
    if not text:
        return {
            "score": 0,
            "status": "NO_DATA"
        }
    
    # 🆕 v41: Użyj nowego systemu priorytetyzacji tripletów
    if s1_relationships:
        result = analyze_triplets_with_priority(
            text=text,
            s1_relationships=s1_relationships,
            s1_entities=s1_entities or []
        )
        return result
    
    # Fallback jeśli brak relacji z S1
    return {
        "score": 50,
        "status": "NO_DATA",
        "found_relationships": [],
        "missing_relationships": [],
        "prioritized": None,
        "prebatch_instruction": None
    }


# ============================================================
# SALIENCE CALCULATION
# ============================================================

def calculate_entity_salience(
    text: str,
    main_entity: str,
    config: EntityScoringConfig = None
) -> Dict[str, Any]:
    """
    Oblicza prominencję głównej encji.
    
    Sprawdza:
    - Czy główna encja jest w pierwszym zdaniu
    - Pozycja pierwszego wystąpienia
    - Częstotliwość w stosunku do innych encji
    - Czy jest w nagłówkach (H2/H3)
    
    Args:
        text: Tekst do analizy
        main_entity: Główna encja/słowo kluczowe
        
    Returns:
        Dict z: score, first_position_pct, frequency, is_in_intro
    """
    if not text or not main_entity:
        return {
            "score": 0,
            "status": "NO_DATA"
        }
    
    text_lower = text.lower()
    main_lower = main_entity.lower()
    
    # v59.1 FIX: Use Polish inflection-aware matching instead of exact regex
    # Old: pattern = rf'\b{re.escape(main_lower)}\b' → misses ALL inflected forms
    # "jazda po alkoholu" never matched "jazdę po alkoholu", "jazdy po alkoholu"
    match_positions = _polish_entity_find_all(main_lower, text_lower)
    
    if not match_positions:
        return {
            "score": 0,
            "frequency": 0,
            "is_in_intro": False,
            "first_position_pct": 100,
            "status": "MISSING"
        }
    
    # Pozycja pierwszego wystąpienia (jako % tekstu)
    first_pos = match_positions[0]
    text_len = len(text_lower)
    first_pos_pct = (first_pos / text_len) * 100 if text_len > 0 else 100
    
    # Częstotliwość
    frequency = len(match_positions)
    words = text_lower.split()
    frequency_per_100 = (frequency / len(words)) * 100 if words else 0
    
    # Czy w intro (pierwszych 200 znaków)
    is_in_intro = first_pos < 200
    
    # Czy w pierwszym zdaniu
    first_sentence_end = text_lower.find('.')
    is_in_first_sentence = first_pos < first_sentence_end if first_sentence_end > 0 else is_in_intro
    
    # Score
    score = 0
    
    # +40 za intro/pierwsze zdanie
    if is_in_first_sentence:
        score += 40
    elif is_in_intro:
        score += 25
    elif first_pos_pct < 10:
        score += 15
    
    # +30 za dobrą częstotliwość (0.5-2.0 na 100 słów)
    if 0.5 <= frequency_per_100 <= 2.0:
        score += 30
    elif 0.3 <= frequency_per_100 <= 3.0:
        score += 20
    elif frequency_per_100 > 0:
        score += 10
    
    # +30 za ogólną prominencję (częste wystąpienia rozproszone)
    if frequency >= 5:
        # Sprawdź rozproszenie
        positions_pct = [p / text_len for p in match_positions] if text_len > 0 else []
        spread = max(positions_pct) - min(positions_pct) if len(positions_pct) > 1 else 0
        
        if spread > 0.7:  # Encja przez cały tekst
            score += 30
        elif spread > 0.4:
            score += 20
        else:
            score += 10
    
    score = min(100, score)
    
    # Status
    if score >= 80:
        status = "HIGH"
    elif score >= 50:
        status = "MEDIUM"
    else:
        status = "LOW"
    
    return {
        "score": score,
        "frequency": frequency,
        "frequency_per_100": round(frequency_per_100, 2),
        "first_position_pct": round(first_pos_pct, 1),
        "is_in_intro": is_in_intro,
        "is_in_first_sentence": is_in_first_sentence,
        "status": status
    }


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def calculate_entity_score(
    text: str,
    s1_entities: List[Dict] = None,
    main_keyword: str = None,
    s1_relationships: List[Dict] = None,
    config: EntityScoringConfig = None
) -> Dict[str, Any]:
    """
    Kompleksowy scoring encji.
    
    Komponenty:
    1. Coverage (30%): % encji z S1 obecnych w tekście
    2. Density (25%): encje / 100 słów
    3. Relationships (25%): relacje S-V-O między encjami
    4. Salience (20%): prominencja głównej encji
    
    Args:
        text: Tekst do analizy
        s1_entities: Encje z analizy S1
        main_keyword: Główne słowo kluczowe
        s1_relationships: Relacje z S1
        config: Konfiguracja
        
    Returns:
        Dict z: score, grade, components, recommendations
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not text:
        return {
            "score": 0,
            "grade": "F",
            "status": "NO_DATA",
            "recommendations": ["Brak tekstu do analizy"]
        }
    
    # 1. Coverage
    coverage = calculate_entity_coverage(text, s1_entities or [], config)
    
    # 2. Density
    density = calculate_entity_density(text, s1_entities, config)
    
    # 3. Relationships
    relationships = analyze_entity_relationships(text, s1_relationships, s1_entities)
    
    # 4. Salience
    salience = calculate_entity_salience(text, main_keyword or "", config)
    
    # Weighted score
    weighted_score = (
        coverage["score"] * config.WEIGHT_COVERAGE +
        density["score"] * config.WEIGHT_DENSITY +
        relationships["score"] * config.WEIGHT_RELATIONSHIPS +
        salience["score"] * config.WEIGHT_SALIENCE
    )
    
    final_score = round(weighted_score)
    
    # Grade
    if final_score >= 85:
        grade = "A"
    elif final_score >= 70:
        grade = "B"
    elif final_score >= 55:
        grade = "C"
    elif final_score >= 40:
        grade = "D"
    else:
        grade = "F"
    
    # Recommendations
    recommendations = []
    
    if coverage["score"] < 60 and coverage.get("missing_important"):
        recommendations.append(f"Brakuje ważnych encji: {', '.join(coverage['missing_important'][:3])}")
    
    if density["status"] == "LOW":
        recommendations.append(density.get("recommendation", "Dodaj więcej encji/definicji"))
    elif density["status"] == "STUFFING":
        recommendations.append("Zbyt wysoka gęstość encji - uprość tekst")
    
    if relationships["score"] < 40:
        recommendations.append("Połącz encje relacjami (kto/co + robi + co/komu)")
    
    if salience["score"] < 50:
        if not salience.get("is_in_intro"):
            recommendations.append(f"Umieść '{main_keyword}' w pierwszym zdaniu")
    
    return {
        "score": final_score,
        "grade": grade,
        "status": "EXCELLENT" if grade == "A" else "GOOD" if grade == "B" else "ACCEPTABLE" if grade == "C" else "NEEDS_WORK",
        "components": {
            "coverage": {
                "score": coverage["score"],
                "weight": config.WEIGHT_COVERAGE,
                "details": coverage
            },
            "density": {
                "score": density["score"],
                "weight": config.WEIGHT_DENSITY,
                "details": density
            },
            "relationships": {
                "score": relationships["score"],
                "weight": config.WEIGHT_RELATIONSHIPS,
                "details": relationships
            },
            "salience": {
                "score": salience["score"],
                "weight": config.WEIGHT_SALIENCE,
                "details": salience
            }
        },
        "recommendations": recommendations,
        "summary": {
            "entities_found": coverage.get("found_count", 0),
            "entities_missing": len(coverage.get("missing", [])),
            "density_per_100": density.get("density", 0),
            "main_entity_prominent": salience.get("is_in_first_sentence", False)
        }
    }


# ============================================================
# VERSION INFO
# ============================================================

__version__ = "40.2"
__all__ = [
    "calculate_entity_score",
    "calculate_entity_coverage",
    "calculate_entity_density",
    "analyze_entity_relationships",
    "calculate_entity_salience",
    "extract_entities_simple",
    "EntityScoringConfig",
    "DEFAULT_CONFIG",
]
