"""
===============================================================================
🎯 SEMANTIC PHRASE ASSIGNMENT v1.0
===============================================================================
Przypisuje frazy, encje i triplety do konkretnych H2 na podstawie 
podobieństwa semantycznego.

PROBLEM KTÓRY ROZWIĄZUJE:
- Agent dostaje 40 fraz bez kontekstu → wybiera "łatwe"
- Frazy nie pasują do aktualnego H2 → nienaturalne wplecenie

ROZWIĄZANIE:
- Analiza semantyczna: która fraza pasuje do którego H2
- Agent wie GDZIE użyć frazy (nie "gdziekolwiek")
- Przykłady zdań dopasowane do kontekstu H2

===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class AssignmentConfig:
    """Konfiguracja przypisywania elementów do H2."""
    
    # Minimalne podobieństwo do przypisania
    MIN_RELEVANCE_THRESHOLD: float = 0.25
    
    # Max elementów MUST per batch
    MAX_MUST_PHRASES_PER_BATCH: int = 5
    MAX_MUST_ENTITIES_PER_BATCH: int = 3
    MAX_MUST_TRIPLETS_PER_BATCH: int = 2
    
    # Wagi dla obliczeń similarity
    JACCARD_WEIGHT: float = 0.4
    KEYWORD_OVERLAP_WEIGHT: float = 0.3
    DOMAIN_HINT_WEIGHT: float = 0.3


CONFIG = AssignmentConfig()


# ============================================================================
# DOMAIN HINTS - wskazówki dla różnych domen
# ============================================================================

DOMAIN_HINTS = {
    "prawo": {
        "sąd": ["procedura", "postępowanie", "orzeczenie", "rozprawa", "wyrok", "sprawa"],
        "kodeks": ["kara", "odpowiedzialność", "przestępstwo", "przepis", "artykuł"],
        "konwencja": ["międzynarodowy", "granica", "zagraniczny", "haska"],
        "ustawa": ["prawo", "regulacja", "przepis", "obowiązek"],
        "wniosek": ["procedura", "złożyć", "sąd", "podanie"],
        "orzeczenie": ["sąd", "wyrok", "decyzja", "rozstrzygnięcie"],
        "władza": ["rodzicielska", "ograniczenie", "pozbawienie", "sąd"],
        "miejsce pobytu": ["dziecko", "ustalenie", "sąd", "rodzic"],
        "uprowadzenie": ["dziecko", "porwanie", "karne", "przestępstwo"],
        "kontakt": ["dziecko", "rodzic", "prawo", "regulacja"],
    },
    "medycyna": {
        "lekarz": ["diagnoza", "leczenie", "badanie", "konsultacja"],
        "pacjent": ["choroba", "leczenie", "objawy", "terapia"],
        "lek": ["dawka", "skutki", "działanie", "recepta"],
    },
    "finanse": {
        "kredyt": ["bank", "rata", "oprocentowanie", "spłata"],
        "inwestycja": ["zysk", "ryzyko", "portfel", "stopa"],
    }
}


# ============================================================================
# SEMANTIC SIMILARITY
# ============================================================================

def calculate_semantic_similarity(
    phrase: str, 
    h2_title: str,
    domain: str = "prawo"
) -> float:
    """
    Oblicza podobieństwo semantyczne frazy do H2.
    
    Składowe:
    1. Jaccard similarity (wspólne słowa)
    2. Keyword overlap (słowa kluczowe domeny)
    3. Domain hints (mapowanie typowe dla domeny)
    
    Returns:
        float: 0.0 - 1.0
    """
    phrase_lower = phrase.lower().strip()
    h2_lower = h2_title.lower().strip()
    
    # Tokenizacja
    phrase_words = set(re.findall(r'\b\w{3,}\b', phrase_lower))
    h2_words = set(re.findall(r'\b\w{3,}\b', h2_lower))
    
    # 1. Jaccard similarity
    intersection = phrase_words & h2_words
    union = phrase_words | h2_words
    jaccard = len(intersection) / len(union) if union else 0
    
    # 2. Keyword overlap - wspólne słowa o długości > 4
    significant_intersection = set(w for w in intersection if len(w) > 4)
    keyword_overlap = len(significant_intersection) * 0.2
    
    # 3. Domain hints
    domain_score = 0
    domain_hints = DOMAIN_HINTS.get(domain, {})
    
    for hint_word, related_words in domain_hints.items():
        # Czy fraza zawiera hint?
        if hint_word in phrase_lower:
            # Czy H2 zawiera powiązane słowa?
            for related in related_words:
                if related in h2_lower:
                    domain_score += 0.15
                    break
        
        # Odwrotnie: czy H2 zawiera hint, a fraza related?
        if hint_word in h2_lower:
            for related in related_words:
                if related in phrase_lower:
                    domain_score += 0.1
                    break
    
    # Suma ważona
    total = (
        jaccard * CONFIG.JACCARD_WEIGHT +
        min(keyword_overlap, 0.4) * CONFIG.KEYWORD_OVERLAP_WEIGHT +
        min(domain_score, 0.5) * CONFIG.DOMAIN_HINT_WEIGHT
    )
    
    return min(1.0, total)


# ============================================================================
# PHRASE ASSIGNMENT
# ============================================================================

def assign_phrases_to_h2(
    keywords_state: Dict,
    h2_structure: List[str],
    main_keyword: str,
    domain: str = "prawo"
) -> Dict[str, List[Dict]]:
    """
    Przypisuje frazy do konkretnych H2 na podstawie podobieństwa semantycznego.
    
    Args:
        keywords_state: Stan fraz {rid: {keyword, type, actual_uses, ...}}
        h2_structure: Lista tytułów H2
        main_keyword: Główne słowo kluczowe
        domain: Domena (prawo, medycyna, finanse, ...)
    
    Returns:
        Dict mapping H2 → lista fraz z relevance score
        
    Example:
        {
            "Czym jest porwanie rodzicielskie": [
                {"keyword": "porwanie rodzicielskie", "type": "MAIN", "relevance": 0.95},
                {"keyword": "definicja porwania", "type": "BASIC", "relevance": 0.72}
            ],
            "Procedura sądowa": [
                {"keyword": "sąd rodzinny", "type": "BASIC", "relevance": 0.88},
                {"keyword": "wniosek do sądu", "type": "EXTENDED", "relevance": 0.65}
            ]
        }
    """
    assignments = {h2: [] for h2 in h2_structure}
    assigned_keywords = set()  # Śledź już przypisane
    
    # Zbierz wszystkie frazy
    all_phrases = []
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        all_phrases.append({
            "rid": rid,
            "keyword": keyword,
            "type": meta.get("type", "BASIC").upper(),
            "actual_uses": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min", 1),
            "target_max": meta.get("target_max", 10),
            "is_main": meta.get("is_main_keyword", False)
        })
    
    # Sortuj: MAIN > BASIC nieużyte > BASIC użyte > EXTENDED
    def phrase_priority(p):
        if p["is_main"]:
            return (0, p["actual_uses"])
        if p["type"] == "BASIC" and p["actual_uses"] == 0:
            return (1, 0)
        if p["type"] == "BASIC":
            return (2, p["actual_uses"])
        return (3, p["actual_uses"])
    
    all_phrases.sort(key=phrase_priority)
    
    # Przypisz każdą frazę do najlepszego H2
    for phrase_data in all_phrases:
        keyword = phrase_data["keyword"]
        
        if keyword in assigned_keywords:
            continue
        
        best_h2 = None
        best_score = CONFIG.MIN_RELEVANCE_THRESHOLD
        
        for h2 in h2_structure:
            score = calculate_semantic_similarity(keyword, h2, domain)
            
            # Bonus dla main keyword - przypisz do pierwszego H2 (definicja)
            if phrase_data["is_main"] and h2 == h2_structure[0]:
                score += 0.3
            
            # Bonus jeśli H2 ma mało przypisanych fraz
            current_count = len(assignments[h2])
            if current_count < 3:
                score += 0.05
            
            if score > best_score:
                best_score = score
                best_h2 = h2
        
        if best_h2:
            assignments[best_h2].append({
                **phrase_data,
                "relevance": round(best_score, 3)
            })
            assigned_keywords.add(keyword)
    
    # Sortuj frazy w każdym H2 po relevance
    for h2 in assignments:
        assignments[h2].sort(key=lambda x: (-x["relevance"], x["actual_uses"]))
    
    # Rozprowadź nieprzypisane frazy równomiernie
    unassigned = [p for p in all_phrases if p["keyword"] not in assigned_keywords]
    if unassigned:
        # Przypisz do H2 z najmniejszą liczbą fraz
        for phrase_data in unassigned:
            min_h2 = min(h2_structure, key=lambda h: len(assignments[h]))
            assignments[min_h2].append({
                **phrase_data,
                "relevance": 0.1,
                "fallback": True
            })
    
    return assignments


# ============================================================================
# ENTITY ASSIGNMENT
# ============================================================================

def assign_entities_to_h2(
    entities: List[Dict],
    h2_structure: List[str],
    domain: str = "prawo"
) -> Dict[str, List[Dict]]:
    """
    Przypisuje encje do H2 na podstawie kontekstu.
    
    Args:
        entities: Lista encji z S1 [{name, importance, sources_count, ...}]
        h2_structure: Lista tytułów H2
        domain: Domena
    
    Returns:
        Dict mapping H2 → lista encji
    """
    assignments = {h2: [] for h2 in h2_structure}
    
    for entity in entities:
        name = entity.get("name", "")
        if not name:
            continue
        
        best_h2 = None
        best_score = 0
        
        for h2 in h2_structure:
            score = calculate_semantic_similarity(name, h2, domain)
            
            # Bonus dla ważnych encji - przypisz do wczesnych H2
            importance = entity.get("importance", 0.5)
            if importance >= 0.7:
                h2_idx = h2_structure.index(h2)
                early_bonus = max(0, 0.1 - h2_idx * 0.02)
                score += early_bonus
            
            if score > best_score:
                best_score = score
                best_h2 = h2
        
        if best_h2 and best_score > CONFIG.MIN_RELEVANCE_THRESHOLD:
            assignments[best_h2].append({
                **entity,
                "h2_relevance": round(best_score, 3)
            })
        else:
            # Fallback: przypisz do H2 z najmniejszą liczbą encji
            min_h2 = min(h2_structure, key=lambda h: len(assignments[h]))
            assignments[min_h2].append({
                **entity,
                "h2_relevance": 0.1,
                "fallback": True
            })
    
    # Sortuj encje w każdym H2 po importance
    for h2 in assignments:
        assignments[h2].sort(key=lambda x: (-x.get("importance", 0), -x.get("h2_relevance", 0)))
    
    return assignments


# ============================================================================
# TRIPLET ASSIGNMENT
# ============================================================================

def assign_triplets_to_h2(
    triplets: List[Dict],
    h2_structure: List[str],
    entity_assignments: Dict[str, List[Dict]],
    domain: str = "prawo"
) -> Dict[str, List[Dict]]:
    """
    Przypisuje triplety do H2 na podstawie encji podmiotu.
    
    Logika: Triplet idzie tam gdzie jest jego SUBJECT entity.
    
    Args:
        triplets: Lista tripletów [{subject, verb, object}]
        h2_structure: Lista tytułów H2
        entity_assignments: Wynik assign_entities_to_h2
        domain: Domena
    
    Returns:
        Dict mapping H2 → lista tripletów
    """
    assignments = {h2: [] for h2 in h2_structure}
    
    # Zbuduj mapę: encja → H2
    entity_to_h2 = {}
    for h2, entities in entity_assignments.items():
        for ent in entities:
            ent_name = ent.get("name", "").lower()
            if ent_name:
                entity_to_h2[ent_name] = h2
    
    for triplet in triplets:
        subject = triplet.get("subject", "").lower()
        obj = triplet.get("object", "").lower()
        
        # Szukaj H2 dla podmiotu
        target_h2 = entity_to_h2.get(subject)
        
        if not target_h2:
            # Szukaj częściowego dopasowania
            for ent_name, h2 in entity_to_h2.items():
                if subject in ent_name or ent_name in subject:
                    target_h2 = h2
                    break
        
        if not target_h2:
            # Spróbuj po obiekcie
            target_h2 = entity_to_h2.get(obj)
            if not target_h2:
                for ent_name, h2 in entity_to_h2.items():
                    if obj in ent_name or ent_name in obj:
                        target_h2 = h2
                        break
        
        if not target_h2:
            # Semantic similarity jako ostateczność
            best_score = 0
            combined = f"{subject} {triplet.get('verb', '')} {obj}"
            for h2 in h2_structure:
                score = calculate_semantic_similarity(combined, h2, domain)
                if score > best_score:
                    best_score = score
                    target_h2 = h2
        
        if target_h2:
            assignments[target_h2].append({
                **triplet,
                "assigned_by": "subject_entity" if entity_to_h2.get(subject) else "semantic_fallback"
            })
        else:
            # Ostateczny fallback: pierwszy H2
            assignments[h2_structure[0]].append({
                **triplet,
                "assigned_by": "default_fallback"
            })
    
    return assignments


# ============================================================================
# CONTEXT-AWARE EXAMPLE GENERATOR
# ============================================================================

def generate_contextual_example(
    phrase: str,
    h2_title: str,
    assigned_triplets: List[Dict],
    domain: str = "prawo"
) -> str:
    """
    Generuje przykładowe zdanie DOPASOWANE do kontekstu H2.
    
    Strategia:
    1. Jeśli fraza jest w triplecie → użyj tripletu
    2. Jeśli H2 ma charakterystyczne słowa → dopasuj styl
    3. Fallback: generyczne zdanie z frazą
    """
    phrase_lower = phrase.lower()
    h2_lower = h2_title.lower()
    
    # 1. Sprawdź czy fraza jest w którymś triplecie
    for triplet in assigned_triplets:
        subj = triplet.get("subject", "").lower()
        obj = triplet.get("object", "").lower()
        verb = triplet.get("verb", "")
        
        if phrase_lower in subj or phrase_lower in obj:
            # Mamy match! Użyj tripletu jako przykładu
            return f"{triplet['subject'].capitalize()} {verb} {triplet['object']}."
    
    # 2. Dopasuj do kontekstu H2
    if domain == "prawo":
        if any(w in h2_lower for w in ["procedur", "sąd", "postępowan"]):
            return f"W toku postępowania, {phrase} wymaga szczegółowej analizy przez sąd."
        
        if any(w in h2_lower for w in ["kar", "przestępst", "odpowiedzialn"]):
            return f"Z perspektywy prawa karnego, {phrase} może prowadzić do odpowiedzialności."
        
        if any(w in h2_lower for w in ["defin", "czym jest", "co to"]):
            return f"{phrase.capitalize()} to termin oznaczający określoną sytuację prawną."
        
        if any(w in h2_lower for w in ["różnic", "porównan"]):
            return f"W odróżnieniu od innych pojęć, {phrase} ma specyficzne znaczenie."
        
        if any(w in h2_lower for w in ["kiedy", "warunek", "przesłank"]):
            return f"O {phrase} mówimy wtedy, gdy spełnione są określone przesłanki."
    
    # 3. Fallback
    return f"{phrase.capitalize()} odgrywa istotną rolę w omawianym kontekście."


def generate_contextual_short_sentences(
    h2_title: str,
    domain: str = "prawo"
) -> List[str]:
    """
    Generuje REGUŁY tworzenia krótkich zdań (3-8 słów) dopasowane do kontekstu H2.
    
    ⚠️ v45.0: Usunięto statyczne zdania ("Sąd orzeka.", "Termin biegnie.").
    GPT kopiował je verbatim → powtarzalny pattern w setkach artykułów.
    
    Teraz zwraca REGUŁY, nie gotowe zdania. GPT tworzy własne z materiału sekcji.
    """
    h2_lower = h2_title.lower()
    
    # Bazowa reguła — zawsze
    rules = [
        f"Krótkie zdanie MUSI zawierać termin z sekcji \"{h2_title}\"",
    ]
    
    if domain == "prawo":
        if any(w in h2_lower for w in ["sąd", "procedur", "postępowan"]):
            rules.append("Skondensuj kluczowy wymóg proceduralny lub termin do 3-5 słów")
            rules.append("Użyj nazwy sądu, terminu lub wymogu z TEGO akapitu")
        
        elif any(w in h2_lower for w in ["kar", "przestępst"]):
            rules.append("Skondensuj konsekwencję prawną lub wymiar kary do 3-5 słów")
            rules.append("Użyj artykułu ustawy lub nazwy przestępstwa z TEGO akapitu")
        
        elif any(w in h2_lower for w in ["dziec", "rodzic", "opiek"]):
            rules.append("Skondensuj kluczowy obowiązek lub prawo do 3-5 słów")
            rules.append("Użyj terminu rodzinno-prawnego z TEGO akapitu")
        
        elif any(w in h2_lower for w in ["defin", "czym", "co to"]):
            rules.append("Skondensuj kluczowy element definicji do 3-5 słów")
            rules.append("Użyj terminu definiowanego w TEJ sekcji")
        
        else:
            rules.append("Wyciągnij kluczowy fakt prawny z poprzedniego zdania")
    
    elif domain == "medycyna":
        rules.append("Użyj nazwy leku, objawu lub parametru medycznego z TEGO akapitu")
        rules.append("Skondensuj kluczowe zalecenie lub wynik do 3-5 słów")
    
    else:
        rules.append("Wyciągnij kluczowy fakt z poprzedniego zdania i skondensuj do 3-5 słów")
    
    # Uniwersalna reguła końcowa
    rules.append("TEST: czy to zdanie pasowałoby do innego artykułu? Jeśli tak → przepisz")
    
    return rules


# ============================================================================
# MAIN: GET ASSIGNMENTS FOR BATCH
# ============================================================================

def get_assignments_for_batch(
    keywords_state: Dict,
    s1_data: Dict,
    h2_structure: List[str],
    current_h2: str,
    main_keyword: str,
    domain: str = "prawo"
) -> Dict[str, Any]:
    """
    Główna funkcja - zwraca wszystkie przypisania dla konkretnego batcha.
    
    Returns:
        {
            "must_phrases": [...],      # Max 5 fraz MUST dla tego H2
            "should_phrases": [...],    # Opcjonalne frazy
            "must_entities": [...],     # Max 3 encje MUST
            "must_triplets": [...],     # Max 2 triplety MUST
            "short_sentences": [...],   # Krótkie zdania do H2
            "phrase_examples": {...},   # Przykłady użycia fraz
        }
    """
    # 1. Pobierz encje i triplety z S1
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    triplets = entity_seo.get("entity_relationships", [])
    
    # 2. Przypisz wszystko do H2
    phrase_assignments = assign_phrases_to_h2(
        keywords_state, h2_structure, main_keyword, domain
    )
    
    entity_assignments = assign_entities_to_h2(
        entities, h2_structure, domain
    )
    
    triplet_assignments = assign_triplets_to_h2(
        triplets, h2_structure, entity_assignments, domain
    )
    
    # 3. Weź elementy dla aktualnego H2
    h2_phrases = phrase_assignments.get(current_h2, [])
    h2_entities = entity_assignments.get(current_h2, [])
    h2_triplets = triplet_assignments.get(current_h2, [])
    
    # 4. Podziel na MUST i SHOULD
    # MUST phrases: nieużyte BASIC (top 5)
    must_phrases = [
        p for p in h2_phrases 
        if p["type"] == "BASIC" and p["actual_uses"] == 0
    ][:CONFIG.MAX_MUST_PHRASES_PER_BATCH]
    
    # Dodaj MAIN jeśli jest przypisany tu i nieużyty wystarczająco
    main_phrases = [p for p in h2_phrases if p.get("is_main")]
    for mp in main_phrases:
        if mp["actual_uses"] < mp["target_min"] and mp not in must_phrases:
            must_phrases.insert(0, mp)
    
    must_phrases = must_phrases[:CONFIG.MAX_MUST_PHRASES_PER_BATCH]
    
    # SHOULD phrases: reszta
    should_phrases = [p for p in h2_phrases if p not in must_phrases][:5]
    
    # MUST entities: importance >= 0.7 (top 3)
    must_entities = [
        e for e in h2_entities 
        if e.get("importance", 0) >= 0.7
    ][:CONFIG.MAX_MUST_ENTITIES_PER_BATCH]
    
    # MUST triplets: top 2
    must_triplets = h2_triplets[:CONFIG.MAX_MUST_TRIPLETS_PER_BATCH]
    
    # 5. Generuj przykłady
    phrase_examples = {}
    for p in must_phrases + should_phrases[:3]:
        phrase_examples[p["keyword"]] = generate_contextual_example(
            p["keyword"], current_h2, h2_triplets, domain
        )
    
    # 6. Generuj krótkie zdania
    short_sentences = generate_contextual_short_sentences(current_h2, domain)
    
    return {
        "current_h2": current_h2,
        "must_phrases": must_phrases,
        "should_phrases": should_phrases,
        "must_entities": must_entities,
        "must_triplets": must_triplets,
        "short_sentences": short_sentences,
        "phrase_examples": phrase_examples,
        "stats": {
            "total_phrases_for_h2": len(h2_phrases),
            "total_entities_for_h2": len(h2_entities),
            "total_triplets_for_h2": len(h2_triplets),
            "must_count": len(must_phrases) + len(must_entities) + len(must_triplets)
        }
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test data
    keywords_state = {
        "k1": {"keyword": "porwanie rodzicielskie", "type": "MAIN", "actual_uses": 3, "target_min": 10, "is_main_keyword": True},
        "k2": {"keyword": "sąd rodzinny", "type": "BASIC", "actual_uses": 0, "target_min": 3},
        "k3": {"keyword": "ustalenie miejsca pobytu dziecka", "type": "BASIC", "actual_uses": 0, "target_min": 2},
        "k4": {"keyword": "uprowadzenie dziecka", "type": "BASIC", "actual_uses": 0, "target_min": 1},
        "k5": {"keyword": "art. 211 kodeksu karnego", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
        "k6": {"keyword": "władza rodzicielska", "type": "BASIC", "actual_uses": 1, "target_min": 3},
        "k7": {"keyword": "odpowiedzialność karna", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
        "k8": {"keyword": "Konwencja haska", "type": "EXTENDED", "actual_uses": 0, "target_min": 1},
    }
    
    h2_structure = [
        "Czym jest porwanie rodzicielskie – definicja",
        "Różnica między porwaniem rodzicielskim a uprowadzeniem dziecka",
        "Procedura sądowa w sprawach o miejsce pobytu dziecka",
        "Kiedy porwanie rodzicielskie jest przestępstwem"
    ]
    
    s1_data = {
        "entity_seo": {
            "entities": [
                {"name": "sąd rodzinny", "importance": 0.85, "sources_count": 6},
                {"name": "Kodeks karny", "importance": 0.75, "sources_count": 4},
                {"name": "Konwencja haska", "importance": 0.70, "sources_count": 3},
            ],
            "entity_relationships": [
                {"subject": "sąd rodzinny", "verb": "ustala", "object": "miejsce pobytu dziecka"},
                {"subject": "rodzic", "verb": "narusza", "object": "prawa drugiego rodzica"},
            ]
        }
    }
    
    print("=" * 60)
    print("TEST: SEMANTIC PHRASE ASSIGNMENT")
    print("=" * 60)
    
    # Test assign_phrases_to_h2
    phrase_assignments = assign_phrases_to_h2(keywords_state, h2_structure, "porwanie rodzicielskie")
    
    print("\n📝 PHRASE ASSIGNMENTS:")
    for h2, phrases in phrase_assignments.items():
        print(f"\n  H2: {h2}")
        for p in phrases[:3]:
            print(f"    • {p['keyword']} ({p['type']}) - relevance: {p['relevance']}")
    
    # Test full batch assignment
    print("\n" + "=" * 60)
    print("TEST: FULL BATCH ASSIGNMENT")
    print("=" * 60)
    
    for h2 in h2_structure[:2]:
        result = get_assignments_for_batch(
            keywords_state=keywords_state,
            s1_data=s1_data,
            h2_structure=h2_structure,
            current_h2=h2,
            main_keyword="porwanie rodzicielskie"
        )
        
        print(f"\n📌 H2: {h2}")
        print(f"\n  MUST PHRASES ({len(result['must_phrases'])}):")
        for p in result['must_phrases']:
            print(f"    • {p['keyword']}")
            if p['keyword'] in result['phrase_examples']:
                print(f"      Przykład: {result['phrase_examples'][p['keyword']]}")
        
        print(f"\n  MUST ENTITIES ({len(result['must_entities'])}):")
        for e in result['must_entities']:
            print(f"    • {e['name']} (importance: {e.get('importance', 'N/A')})")
        
        print(f"\n  MUST TRIPLETS ({len(result['must_triplets'])}):")
        for t in result['must_triplets']:
            print(f"    • {t['subject']} → {t['verb']} → {t['object']}")
        
        print(f"\n  SHORT SENTENCES: {', '.join(result['short_sentences'][:3])}")
        print(f"\n  STATS: {result['stats']}")
