"""
===============================================================================
🔍 SEMANTIC TRIPLET VALIDATOR v1.0
===============================================================================
Walidacja semantyczna tripletów zamiast dosłownego porównania.

PROBLEM: "Sąd rodzinny ustala miejsce pobytu" x3 brzmi jak robot

ROZWIĄZANIE: Akceptuj warianty: "Miejsce pobytu jest ustalane przez sąd"

===============================================================================
"""

import re
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class TripletMatch:
    triplet: Dict
    matched_sentence: str
    similarity_score: float
    match_type: str  # "exact", "semantic", "partial", "none"


# Synonimy czasowników
VERB_SYNONYMS = {
    "ustala": ["decyduje o", "określa", "wyznacza", "rozstrzyga"],
    "reguluje": ["normuje", "określa", "stanowi o"],
    "narusza": ["łamie", "przekracza", "nie respektuje"],
    "wymaga": ["zobowiązuje do", "nakazuje"],
    "rozpatruje": ["bada", "analizuje", "zajmuje się"],
    "orzeka": ["decyduje", "postanawia"],
}

# Formy bierne
PASSIVE_FORMS = {
    "ustala": "jest ustalane przez",
    "reguluje": "jest regulowane przez",
    "rozpatruje": "jest rozpatrywane przez",
}


def normalize(text: str) -> str:
    return re.sub(r'[^\w\s]', ' ', text.lower()).strip()


def match_component(target: str, sentence: str) -> Tuple[float, str]:
    """Sprawdza czy komponent jest w zdaniu."""
    target_norm = normalize(target)
    sentence_norm = normalize(sentence)
    
    # Exact
    if target_norm in sentence_norm:
        return 1.0, "exact"
    
    # Word overlap
    target_words = set(target_norm.split())
    sentence_words = set(sentence_norm.split())
    overlap = len(target_words & sentence_words) / len(target_words) if target_words else 0
    
    if overlap >= 0.6:
        return overlap, "partial"
    
    # Main word
    main_words = [w for w in target_words if len(w) > 4]
    if main_words and main_words[0] in sentence_norm:
        return 0.5, "main_word"
    
    return 0.0, "none"


def match_verb(verb: str, sentence: str) -> Tuple[float, str]:
    """Sprawdza czasownik z synonimami i formą bierną."""
    verb_norm = normalize(verb)
    sentence_norm = normalize(sentence)
    
    if verb_norm in sentence_norm:
        return 1.0, "exact"
    
    for syn in VERB_SYNONYMS.get(verb_norm, []):
        if syn in sentence_norm:
            return 0.9, "synonym"
    
    passive = PASSIVE_FORMS.get(verb_norm)
    if passive and passive in sentence_norm:
        return 0.85, "passive"
    
    return 0.0, "none"


def validate_triplet_in_sentence(triplet: Dict, sentence: str) -> TripletMatch:
    """Sprawdza czy triplet jest semantycznie w zdaniu."""
    subject_score, _ = match_component(triplet.get("subject", ""), sentence)
    verb_score, _ = match_verb(triplet.get("verb", ""), sentence)
    object_score, _ = match_component(triplet.get("object", ""), sentence)
    
    total = subject_score * 0.35 + verb_score * 0.30 + object_score * 0.35
    
    if total >= 0.55:
        match_type = "semantic"
    elif total >= 0.35:
        match_type = "partial"
    else:
        match_type = "none"
    
    return TripletMatch(triplet, sentence, total, match_type)


def validate_triplets_in_text(text: str, triplets: List[Dict]) -> Dict:
    """Waliduje wszystkie triplety w tekście."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    results = []
    missing = []
    
    for triplet in triplets:
        best_match = None
        best_score = 0
        
        for sentence in sentences:
            match = validate_triplet_in_sentence(triplet, sentence)
            if match.similarity_score > best_score:
                best_score = match.similarity_score
                best_match = match
        
        if best_match and best_match.match_type in ["semantic", "exact"]:
            results.append(best_match)
        else:
            missing.append(triplet)
    
    return {
        "passed": len(missing) == 0,
        "matched": len(results),
        "total": len(triplets),
        "missing": missing,
        "score": len(results) / len(triplets) if triplets else 1.0
    }


def generate_semantic_instruction(triplet: Dict) -> str:
    """Generuje instrukcję w nowym formacie."""
    s, v, o = triplet.get("subject", ""), triplet.get("verb", ""), triplet.get("object", "")
    passive = PASSIVE_FORMS.get(v.lower(), "")
    
    return f"""
🔗 RELACJA: {s} → {v} → {o}

✅ AKCEPTOWANE FORMY:
   • "{s.capitalize()} {v} {o}."
   • "{o.capitalize()} {passive} {s}." (bierna)
   • Dowolna inna forma zachowująca SENS relacji

❌ UNIKAJ powtarzania tej samej formy!
"""


if __name__ == "__main__":
    triplet = {"subject": "sąd rodzinny", "verb": "ustala", "object": "miejsce pobytu dziecka"}
    
    tests = [
        "Sąd rodzinny ustala miejsce pobytu dziecka.",  # exact
        "Miejsce pobytu dziecka jest ustalane przez sąd rodzinny.",  # passive
        "Sąd rodzinny decyduje o miejscu pobytu dziecka.",  # synonym
        "Rodzice ustalają wspólnie.",  # none
    ]
    
    print("TEST: SEMANTIC TRIPLET VALIDATOR\n")
    for t in tests:
        m = validate_triplet_in_sentence(triplet, t)
        status = "✅" if m.match_type in ["semantic", "exact"] else "❌"
        print(f"{status} {m.similarity_score:.2f} | {m.match_type} | {t[:50]}")
