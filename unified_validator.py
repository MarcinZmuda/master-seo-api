"""
===============================================================================
🔧 UNIFIED VALIDATOR PATCH v31.0 - Semantic Enhancement Integration
===============================================================================
Rozszerza unified_validator.py v23.0 o:
1. Entity Density validation
2. Topic Completeness validation  
3. Entity Gap detection (automatyczne z S1)
4. Source Effort scoring

INSTALACJA:
1. Skopiuj ten plik do katalogu z unified_validator.py
2. Dodaj import na końcu unified_validator.py:
   from unified_validator_patch import validate_semantic_enhancement, extend_validation_result
3. W full_validate_complete() dodaj wywołanie semantic validation

===============================================================================
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# ================================================================
# 📊 KONFIGURACJA
# ================================================================

@dataclass
class SemanticConfig:
    """Progi walidacji semantycznej."""
    # Entity Density
    ENTITY_DENSITY_MIN = 2.5
    ENTITY_DENSITY_MAX = 7.0
    HARD_ENTITY_RATIO_MIN = 0.15
    
    # Topic Completeness
    TOPIC_COMPLETENESS_MIN = 0.60
    
    # Entity Gap
    ENTITY_GAP_MIN = 0.40
    
    # Source Effort
    SOURCE_EFFORT_MIN = 0.35

# Wagi typów encji
ENTITY_TYPE_WEIGHTS = {
    "PERSON": 1.5, "PER": 1.5,
    "ORGANIZATION": 1.3, "ORG": 1.3,
    "LEGAL_ACT": 1.4,
    "PUBLICATION": 1.2,
    "STANDARD": 1.1,
    "PRODUCT": 1.0,
    "LOCATION": 0.8, "LOC": 0.8, "GPE": 0.8,
    "DATE": 0.6,
}

HARD_ENTITY_TYPES = {"PERSON", "PER", "ORGANIZATION", "ORG", "LEGAL_ACT", "PUBLICATION", "STANDARD"}

# ================================================================
# 📝 WZORCE REGEX - Source Effort
# ================================================================

SOURCE_EFFORT_PATTERNS = {
    "COURT_RULING": {
        "weight": 2.0,
        "patterns": [
            r'(?:wyrok|uchwała|postanowienie)\s+(?:SN|SA|SO|TK|NSA|WSA|TSUE)',
            r'sygn\.\s*akt\s*[A-Z]{1,4}\s*\d+/\d+',
        ]
    },
    "LEGAL_ACT": {
        "weight": 1.5,
        "patterns": [
            r'art\.\s*\d+(?:\s*ust\.\s*\d+)?',
            r'§\s*\d+',
            r'Dz\.?\s*U\.?\s*\d{4}',
            r'RODO|GDPR|NIS2',
        ]
    },
    "SCIENTIFIC": {
        "weight": 1.8,
        "patterns": [
            r'et\s+al\.?',
            r'\(\d{4}\)\s*(?:wykazał|pokazał)',
            r'p\s*[<>=]\s*0[,\.]\d+',
            r'n\s*=\s*\d{2,}',
        ]
    },
    "OFFICIAL_DATA": {
        "weight": 1.4,
        "patterns": [
            r'(?:dane|raport)\s+(?:GUS|Eurostat|OECD|WHO|NBP)',
            r'stan\s+na\s+(?:dzień\s+)?\d+[./]\d+[./]\d+',
        ]
    },
    "EXPERT": {
        "weight": 1.3,
        "patterns": [
            r'(?:prof\.|dr\.?|mgr)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ]',
        ]
    }
}

NEGATIVE_PATTERNS = [
    (r'niektórzy\s+(?:eksperci|badacze)', -0.3),
    (r'powszechnie\s+wiadomo', -0.3),
    (r'(?:ostatnio|niedawno)\b', -0.2),
]

# Ogólniki do wykrycia
GENERIC_PHRASES = [
    "według ekspertów", "badania pokazują", "statystyki wskazują",
    "zgodnie z prawem", "algorytm google", "w ostatnich latach",
    "odpowiedni urząd", "wiele osób uważa", "powszechnie wiadomo",
]

ZERO_VALUE_PHRASES = [
    "warto wiedzieć", "nie ulega wątpliwości", "jak wszyscy wiedzą",
    "w dzisiejszych czasach", "coraz więcej osób", "każdy wie że",
]

# ================================================================
# 🔧 FUNKCJE ANALIZY
# ================================================================

def calculate_entity_density(text: str, entities: List[Dict] = None) -> Dict[str, Any]:
    """Oblicza gęstość encji w tekście."""
    words = text.split()
    word_count = len(words)
    
    if word_count < 50:
        return {"status": "TOO_SHORT", "density": 0}
    
    entity_count = len(entities) if entities else 0
    density = (entity_count / word_count) * 100
    
    # Hard entities
    hard_count = sum(1 for e in (entities or []) if e.get("type") in HARD_ENTITY_TYPES)
    hard_ratio = hard_count / entity_count if entity_count > 0 else 0
    
    # Generics found
    text_lower = text.lower()
    generics_found = [p for p in GENERIC_PHRASES if p in text_lower]
    zero_value_found = [p for p in ZERO_VALUE_PHRASES if p in text_lower]
    
    config = SemanticConfig()
    status = "GOOD" if density >= config.ENTITY_DENSITY_MIN else "NEEDS_IMPROVEMENT"
    if density > config.ENTITY_DENSITY_MAX:
        status = "OVERSTUFFED"
    
    return {
        "status": status,
        "density_per_100": round(density, 2),
        "entity_count": entity_count,
        "hard_entity_ratio": round(hard_ratio, 2),
        "generics_found": generics_found[:5],
        "zero_value_found": zero_value_found[:3],
        "action_required": status != "GOOD" or len(generics_found) > 2
    }


def calculate_topic_completeness(content: str, s1_topics: List[Dict]) -> Dict[str, Any]:
    """Sprawdza pokrycie tematów z S1 topical_coverage."""
    if not s1_topics:
        return {"status": "NO_DATA", "score": 0.5}
    
    content_lower = content.lower()
    must_high = [t for t in s1_topics if t.get("priority") in ["MUST", "HIGH"]]
    
    covered = []
    missing = []
    
    for topic in must_high:
        subtopic = topic.get("subtopic", "").lower()
        words = subtopic.split()
        matches = sum(1 for w in words if w in content_lower)
        is_covered = matches / len(words) >= 0.5 if words else False
        
        if is_covered:
            covered.append(topic)
        else:
            missing.append({
                "topic": subtopic,
                "priority": topic.get("priority"),
                "sample_h2": topic.get("sample_h2", "")
            })
    
    score = len(covered) / len(must_high) if must_high else 1.0
    config = SemanticConfig()
    
    return {
        "status": "GOOD" if score >= config.TOPIC_COMPLETENESS_MIN else "NEEDS_IMPROVEMENT",
        "score": round(score, 2),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "missing_topics": missing[:5],
        "action_required": score < config.TOPIC_COMPLETENESS_MIN
    }


def calculate_entity_gap(content: str, s1_entities: List[Dict]) -> Dict[str, Any]:
    """Wykrywa brakujące encje z S1 - AUTOMATYCZNIE."""
    if not s1_entities:
        return {"status": "NO_DATA", "score": 0.5}
    
    content_lower = content.lower()
    important = [e for e in s1_entities if e.get("importance", 0) >= 0.3]
    
    found = []
    missing = []
    
    for entity in important:
        text = entity.get("text", "").lower()
        importance = entity.get("importance", 0.5)
        sources = entity.get("sources_count", 1)
        
        if text in content_lower:
            found.append(entity)
        else:
            # Określ priorytet
            if importance >= 0.7 and sources >= 4:
                priority = "CRITICAL"
            elif importance >= 0.5 and sources >= 2:
                priority = "HIGH"
            else:
                priority = "MEDIUM"
            
            missing.append({
                "entity": entity.get("text"),
                "type": entity.get("type"),
                "priority": priority,
                "importance": importance
            })
    
    # Sortuj missing po priorytecie
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    missing.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    score = len(found) / len(important) if important else 1.0
    config = SemanticConfig()
    
    return {
        "status": "GOOD" if score >= config.ENTITY_GAP_MIN else "WEAK",
        "score": round(score, 2),
        "found_count": len(found),
        "missing_count": len(missing),
        "critical_missing": [m for m in missing if m["priority"] == "CRITICAL"][:5],
        "high_missing": [m for m in missing if m["priority"] == "HIGH"][:5],
        "action_required": score < config.ENTITY_GAP_MIN
    }


def calculate_source_effort(text: str) -> Dict[str, Any]:
    """Mierzy sygnały wysiłku badawczego."""
    text_lower = text.lower()
    word_count = len(text.split())
    
    if word_count < 100:
        return {"status": "TOO_SHORT", "score": 0}
    
    # Positive signals
    signals_found = defaultdict(list)
    total_weight = 0
    
    for category, data in SOURCE_EFFORT_PATTERNS.items():
        for pattern in data["patterns"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                signals_found[category].extend(matches[:3])
                total_weight += data["weight"] * len(matches)
    
    # Negative signals
    penalty = 0
    negatives = []
    for pattern, pen in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            penalty += pen
            negatives.append(pattern)
    
    # Normalize
    normalized = (total_weight / (word_count / 500)) if word_count > 0 else 0
    score = max(0, min(1.0, (normalized / 10) + penalty))
    
    # Diversity bonus
    diversity = min(0.15, len(signals_found) * 0.03)
    score = min(1.0, score + diversity)
    
    config = SemanticConfig()
    
    return {
        "status": "GOOD" if score >= config.SOURCE_EFFORT_MIN else "NEEDS_IMPROVEMENT",
        "score": round(score, 2),
        "signals_found": {k: len(v) for k, v in signals_found.items()},
        "negative_patterns": negatives[:3],
        "categories_covered": len(signals_found),
        "action_required": score < config.SOURCE_EFFORT_MIN
    }


# ================================================================
# 🎯 GŁÓWNA FUNKCJA
# ================================================================

def validate_semantic_enhancement(
    content: str,
    s1_data: Dict = None,
    detected_entities: List[Dict] = None
) -> Dict[str, Any]:
    """
    Główna funkcja walidacji semantycznej.
    
    Args:
        content: Treść do walidacji
        s1_data: Dane z S1 (topical_coverage, entities)
        detected_entities: Encje wykryte w treści (z NER)
    """
    s1_data = s1_data or {}
    
    # 1. Entity Density
    density = calculate_entity_density(content, detected_entities)
    
    # 2. Topic Completeness
    topics = s1_data.get("topical_coverage", [])
    completeness = calculate_topic_completeness(content, topics)
    
    # 3. Entity Gap
    entities = s1_data.get("entities", s1_data.get("entity_seo", {}).get("entities", []))
    gap = calculate_entity_gap(content, entities)
    
    # 4. Source Effort
    effort = calculate_source_effort(content)
    
    # Final score
    scores = {
        "entity_density": 0.7 if density["status"] == "GOOD" else 0.3,
        "topic_completeness": completeness.get("score", 0.5),
        "entity_gap": gap.get("score", 0.5),
        "source_effort": effort.get("score", 0.5)
    }
    
    final_score = sum(scores.values()) / 4
    
    # Issues
    issues = []
    if density.get("action_required"):
        issues.append({"code": "LOW_ENTITY_DENSITY", "severity": "WARNING"})
    if completeness.get("action_required"):
        issues.append({"code": "LOW_TOPIC_COMPLETENESS", "severity": "WARNING"})
    if gap.get("action_required"):
        issues.append({"code": "HIGH_ENTITY_GAP", "severity": "WARNING"})
    if effort.get("action_required"):
        issues.append({"code": "LOW_SOURCE_EFFORT", "severity": "WARNING"})
    
    status = "APPROVED" if len(issues) <= 1 else "WARN" if len(issues) <= 3 else "REJECTED"
    
    return {
        "status": status,
        "semantic_score": round(final_score, 2),
        "component_scores": scores,
        "analyses": {
            "entity_density": density,
            "topic_completeness": completeness,
            "entity_gap": gap,
            "source_effort": effort
        },
        "issues": issues,
        "quick_wins": _get_quick_wins(density, completeness, gap, effort)
    }


def _get_quick_wins(density, completeness, gap, effort) -> List[str]:
    """Generuje listę szybkich poprawek."""
    wins = []
    
    if density.get("generics_found"):
        wins.append(f"Zamień ogólniki: {', '.join(density['generics_found'][:2])}")
    
    if completeness.get("missing_topics"):
        topic = completeness["missing_topics"][0]
        wins.append(f"Dodaj H2 o: {topic.get('sample_h2', topic.get('topic'))}")
    
    if gap.get("critical_missing"):
        entity = gap["critical_missing"][0]
        wins.append(f"Dodaj encję: {entity['entity']} ({entity['type']})")
    
    if effort.get("score", 1) < 0.4:
        wins.append("Dodaj źródła: orzecznictwo, akty prawne lub dane statystyczne")
    
    return wins[:5]


def extend_validation_result(base_result: Dict, semantic_result: Dict) -> Dict:
    """Rozszerza wynik walidacji o dane semantyczne."""
    base_result["semantic_enhancement"] = {
        "score": semantic_result.get("semantic_score"),
        "status": semantic_result.get("status"),
        "component_scores": semantic_result.get("component_scores", {})
    }
    
    # Add issues
    for issue in semantic_result.get("issues", []):
        base_result.setdefault("issues", []).append(issue)
    
    # Add recommendations
    base_result["semantic_quick_wins"] = semantic_result.get("quick_wins", [])
    
    return base_result
