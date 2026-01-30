"""
===============================================================================
🔀 DYNAMIC SUB-BATCH SPLITTER v1.0
===============================================================================
Automatyczny podział H2 na sub-batche gdy za dużo elementów.

PROBLEM:
- H2 "Procedura sądowa" ma przypisane 15 fraz kluczowych
- Upchnięcie ich w jednym batchu (nawet LONG) = stuffing
- Max 5 MUST phrases per instrukcja = reszta pominięta

ROZWIĄZANIE:
- Przed generowaniem sprawdź "gęstość" elementów
- Jeśli > 8 fraz LUB > 3 encje → podziel H2 na sub-batche
- Batch 3 → Batch 3A (first half) + Batch 3B (second half)
- Automatycznie generuj H3 jako pod-wątki

ZYSK:
- Agent ma 2x więcej miejsca na naturalne użycie fraz
- Drastycznie podnosi Human Score
- Pozwala użyć WSZYSTKIE frazy EXTENDED

===============================================================================
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SubBatchConfig:
    """Konfiguracja sub-batchów."""
    # Progi do podziału
    MAX_PHRASES_PER_BATCH: int = 8
    MAX_ENTITIES_PER_BATCH: int = 3
    MAX_TRIPLETS_PER_BATCH: int = 2
    
    # Minimalna liczba elementów do podziału
    MIN_ELEMENTS_TO_SPLIT: int = 6
    
    # Max sub-batchów per H2
    MAX_SUB_BATCHES: int = 3


@dataclass
class SubBatch:
    """Pojedynczy sub-batch."""
    parent_h2: str
    sub_batch_id: str  # "3A", "3B", etc.
    h3_title: str
    assigned_phrases: List[Dict]
    assigned_entities: List[Dict]
    assigned_triplets: List[Dict]
    word_target: Tuple[int, int]  # (min, max)
    
    @property
    def total_elements(self) -> int:
        return len(self.assigned_phrases) + len(self.assigned_entities) + len(self.assigned_triplets)


@dataclass 
class SplitResult:
    """Wynik podziału H2."""
    original_h2: str
    needs_split: bool
    sub_batches: List[SubBatch] = field(default_factory=list)
    reason: str = ""
    stats: Dict = field(default_factory=dict)


CONFIG = SubBatchConfig()


# ============================================================================
# ANALIZA GĘSTOŚCI
# ============================================================================

def analyze_h2_density(
    h2_title: str,
    assigned_phrases: List[Dict],
    assigned_entities: List[Dict],
    assigned_triplets: List[Dict]
) -> Dict:
    """
    Analizuje "gęstość" elementów przypisanych do H2.
    
    Returns:
        {
            "needs_split": True/False,
            "density_score": float,  # 0-1, >0.7 = needs split
            "total_elements": int,
            "recommended_sub_batches": int,
            "bottleneck": "phrases" | "entities" | "triplets" | None
        }
    """
    total_phrases = len(assigned_phrases)
    total_entities = len(assigned_entities)
    total_triplets = len(assigned_triplets)
    total_elements = total_phrases + total_entities + total_triplets
    
    # Oblicz "przepełnienie" dla każdej kategorii
    phrase_overflow = max(0, total_phrases - CONFIG.MAX_PHRASES_PER_BATCH) / CONFIG.MAX_PHRASES_PER_BATCH
    entity_overflow = max(0, total_entities - CONFIG.MAX_ENTITIES_PER_BATCH) / CONFIG.MAX_ENTITIES_PER_BATCH
    triplet_overflow = max(0, total_triplets - CONFIG.MAX_TRIPLETS_PER_BATCH) / CONFIG.MAX_TRIPLETS_PER_BATCH
    
    # Density score: średnia ważona przepełnień
    density_score = (
        phrase_overflow * 0.5 +   # Frazy najważniejsze
        entity_overflow * 0.3 +   # Encje drugie
        triplet_overflow * 0.2    # Triplety trzecie
    )
    
    # Znajdź bottleneck
    bottleneck = None
    if phrase_overflow > 0:
        bottleneck = "phrases"
    elif entity_overflow > 0:
        bottleneck = "entities"
    elif triplet_overflow > 0:
        bottleneck = "triplets"
    
    # Ile sub-batchów potrzeba?
    recommended = 1
    if total_phrases > CONFIG.MAX_PHRASES_PER_BATCH:
        recommended = max(recommended, math.ceil(total_phrases / CONFIG.MAX_PHRASES_PER_BATCH))
    if total_entities > CONFIG.MAX_ENTITIES_PER_BATCH:
        recommended = max(recommended, math.ceil(total_entities / CONFIG.MAX_ENTITIES_PER_BATCH))
    
    recommended = min(recommended, CONFIG.MAX_SUB_BATCHES)
    
    # Decyzja o podziale
    needs_split = (
        density_score > 0.3 or
        total_elements > CONFIG.MIN_ELEMENTS_TO_SPLIT * 1.5 or
        total_phrases > CONFIG.MAX_PHRASES_PER_BATCH
    )
    
    return {
        "needs_split": needs_split,
        "density_score": round(density_score, 2),
        "total_elements": total_elements,
        "element_counts": {
            "phrases": total_phrases,
            "entities": total_entities,
            "triplets": total_triplets
        },
        "recommended_sub_batches": recommended,
        "bottleneck": bottleneck
    }


# ============================================================================
# GENEROWANIE H3 (POD-WĄTKÓW)
# ============================================================================

def generate_h3_titles(h2_title: str, num_sub_batches: int, domain: str = "prawo") -> List[str]:
    """
    Generuje tytuły H3 na podstawie H2.
    
    Strategie:
    1. Rozdziel na aspekty (teoretyczny vs praktyczny)
    2. Rozdziel chronologicznie (przed, w trakcie, po)
    3. Rozdziel na elementy (co, jak, kiedy)
    """
    h2_lower = h2_title.lower()
    
    # Strategia 1: Aspekty (dla definicji, pojęć)
    if any(w in h2_lower for w in ["czym jest", "definicja", "pojęcie", "co to"]):
        return [
            f"Definicja i podstawy prawne",
            f"Praktyczne zastosowanie"
        ][:num_sub_batches]
    
    # Strategia 2: Chronologiczna (dla procedur)
    if any(w in h2_lower for w in ["procedur", "postępowan", "jak", "krok"]):
        chronological = [
            f"Przygotowanie i pierwsze kroki",
            f"Przebieg postępowania",
            f"Zakończenie i skutki"
        ]
        return chronological[:num_sub_batches]
    
    # Strategia 3: Elementy (dla odpowiedzialności, konsekwencji)
    if any(w in h2_lower for w in ["odpowiedzialn", "kar", "konsekwen", "skutk"]):
        return [
            f"Rodzaje i zakres odpowiedzialności",
            f"Przesłanki i okoliczności"
        ][:num_sub_batches]
    
    # Strategia 4: Podmioty (dla spraw rodzinnych)
    if any(w in h2_lower for w in ["rodzic", "dziec", "opiek"]):
        return [
            f"Perspektywa prawna",
            f"Aspekty praktyczne i psychologiczne"
        ][:num_sub_batches]
    
    # Domyślna strategia: numeracja
    return [
        f"{h2_title} - część {i+1}"
        for i in range(num_sub_batches)
    ]


# ============================================================================
# PODZIAŁ ELEMENTÓW NA SUB-BATCHE
# ============================================================================

def distribute_elements_to_sub_batches(
    phrases: List[Dict],
    entities: List[Dict],
    triplets: List[Dict],
    num_sub_batches: int
) -> List[Dict]:
    """
    Rozdziela elementy równomiernie na sub-batche.
    
    Strategia:
    1. Sortuj frazy po importance/relevance
    2. Rozdziel round-robin z priorytetem BASIC > EXTENDED
    3. Encje i triplety dopasuj do fraz (semantic matching)
    """
    # Sortuj frazy: BASIC nieużyte > BASIC użyte > EXTENDED
    def phrase_priority(p):
        ptype = p.get("type", "EXTENDED").upper()
        actual = p.get("actual_uses", 0)
        relevance = p.get("relevance", 0)
        
        if ptype == "BASIC" and actual == 0:
            return (0, -relevance)
        if ptype == "BASIC":
            return (1, -relevance)
        return (2, -relevance)
    
    sorted_phrases = sorted(phrases, key=phrase_priority)
    
    # Rozdziel frazy round-robin
    phrase_buckets = [[] for _ in range(num_sub_batches)]
    for i, phrase in enumerate(sorted_phrases):
        bucket_idx = i % num_sub_batches
        phrase_buckets[bucket_idx].append(phrase)
    
    # Rozdziel encje równomiernie
    entity_buckets = [[] for _ in range(num_sub_batches)]
    for i, entity in enumerate(entities):
        bucket_idx = i % num_sub_batches
        entity_buckets[bucket_idx].append(entity)
    
    # Rozdziel triplety równomiernie
    triplet_buckets = [[] for _ in range(num_sub_batches)]
    for i, triplet in enumerate(triplets):
        bucket_idx = i % num_sub_batches
        triplet_buckets[bucket_idx].append(triplet)
    
    return [
        {
            "phrases": phrase_buckets[i],
            "entities": entity_buckets[i],
            "triplets": triplet_buckets[i]
        }
        for i in range(num_sub_batches)
    ]


# ============================================================================
# GŁÓWNA FUNKCJA: SPLIT H2
# ============================================================================

def split_h2_if_needed(
    h2_title: str,
    batch_number: int,
    assigned_phrases: List[Dict],
    assigned_entities: List[Dict],
    assigned_triplets: List[Dict],
    target_words_per_batch: Tuple[int, int] = (400, 600),
    domain: str = "prawo"
) -> SplitResult:
    """
    Analizuje H2 i dzieli na sub-batche jeśli potrzeba.
    
    Args:
        h2_title: Tytuł H2
        batch_number: Numer oryginalnego batcha
        assigned_phrases: Frazy przypisane do tego H2
        assigned_entities: Encje przypisane do tego H2
        assigned_triplets: Triplety przypisane do tego H2
        target_words_per_batch: Cel długości batcha
        domain: Domena
    
    Returns:
        SplitResult z listą SubBatch (1 jeśli bez podziału, >1 jeśli podzielony)
    """
    # 1. Analiza gęstości
    density = analyze_h2_density(
        h2_title=h2_title,
        assigned_phrases=assigned_phrases,
        assigned_entities=assigned_entities,
        assigned_triplets=assigned_triplets
    )
    
    # 2. Sprawdź czy potrzebny podział
    if not density["needs_split"]:
        # Bez podziału - zwróć jeden "sub-batch" = oryginalny batch
        return SplitResult(
            original_h2=h2_title,
            needs_split=False,
            sub_batches=[
                SubBatch(
                    parent_h2=h2_title,
                    sub_batch_id=str(batch_number),
                    h3_title="",  # Brak H3 bo nie ma podziału
                    assigned_phrases=assigned_phrases,
                    assigned_entities=assigned_entities,
                    assigned_triplets=assigned_triplets,
                    word_target=target_words_per_batch
                )
            ],
            reason="Density OK, no split needed",
            stats=density
        )
    
    # 3. Określ liczbę sub-batchów
    num_sub_batches = density["recommended_sub_batches"]
    
    # 4. Wygeneruj H3
    h3_titles = generate_h3_titles(h2_title, num_sub_batches, domain)
    
    # 5. Rozdziel elementy
    element_distribution = distribute_elements_to_sub_batches(
        phrases=assigned_phrases,
        entities=assigned_entities,
        triplets=assigned_triplets,
        num_sub_batches=num_sub_batches
    )
    
    # 6. Oblicz target words per sub-batch (proporcjonalnie mniejszy)
    sub_batch_words = (
        target_words_per_batch[0] // num_sub_batches + 50,
        target_words_per_batch[1] // num_sub_batches + 100
    )
    # Minimum 200 słów
    sub_batch_words = (max(200, sub_batch_words[0]), max(300, sub_batch_words[1]))
    
    # 7. Utwórz sub-batche
    sub_batches = []
    sub_batch_letters = "ABCDEFGHIJ"
    
    for i in range(num_sub_batches):
        sub_batch_id = f"{batch_number}{sub_batch_letters[i]}"
        
        sub_batches.append(SubBatch(
            parent_h2=h2_title,
            sub_batch_id=sub_batch_id,
            h3_title=h3_titles[i] if i < len(h3_titles) else f"Część {i+1}",
            assigned_phrases=element_distribution[i]["phrases"],
            assigned_entities=element_distribution[i]["entities"],
            assigned_triplets=element_distribution[i]["triplets"],
            word_target=sub_batch_words
        ))
    
    return SplitResult(
        original_h2=h2_title,
        needs_split=True,
        sub_batches=sub_batches,
        reason=f"High density ({density['density_score']}), bottleneck: {density['bottleneck']}",
        stats=density
    )


# ============================================================================
# INTEGRACJA Z BATCH PLANNER
# ============================================================================

def process_batch_plan_with_splitting(
    batch_plan: List[Dict],
    phrase_assignments: Dict[str, List[Dict]],
    entity_assignments: Dict[str, List[Dict]],
    triplet_assignments: Dict[str, List[Dict]],
    domain: str = "prawo"
) -> List[Dict]:
    """
    Przetwarza plan batchów i dzieli H2 gdzie potrzeba.
    
    Args:
        batch_plan: Oryginalny plan batchów
        phrase_assignments: Przypisanie fraz do H2
        entity_assignments: Przypisanie encji do H2
        triplet_assignments: Przypisanie tripletów do H2
        domain: Domena
    
    Returns:
        Nowy plan batchów z sub-batchami
    """
    new_plan = []
    
    for batch in batch_plan:
        batch_number = batch.get("batch_number", len(new_plan) + 1)
        h2_sections = batch.get("h2_sections", [])
        
        # Batch bez H2 (intro) - przepuść bez zmian
        if not h2_sections:
            new_plan.append(batch)
            continue
        
        # Dla każdego H2 w batchu
        for h2_title in h2_sections:
            phrases = phrase_assignments.get(h2_title, [])
            entities = entity_assignments.get(h2_title, [])
            triplets = triplet_assignments.get(h2_title, [])
            
            # Sprawdź czy potrzebny podział
            split_result = split_h2_if_needed(
                h2_title=h2_title,
                batch_number=batch_number,
                assigned_phrases=phrases,
                assigned_entities=entities,
                assigned_triplets=triplets,
                target_words_per_batch=(
                    batch.get("words_min", 400),
                    batch.get("words_max", 600)
                ),
                domain=domain
            )
            
            # Dodaj sub-batche do planu
            for sub_batch in split_result.sub_batches:
                new_plan.append({
                    "batch_number": sub_batch.sub_batch_id,
                    "batch_type": "CONTENT",
                    "h2_sections": [sub_batch.parent_h2],
                    "h3_title": sub_batch.h3_title if split_result.needs_split else None,
                    "is_sub_batch": split_result.needs_split,
                    "parent_batch": batch_number if split_result.needs_split else None,
                    "assigned_phrases": sub_batch.assigned_phrases,
                    "assigned_entities": sub_batch.assigned_entities,
                    "assigned_triplets": sub_batch.assigned_triplets,
                    "words_min": sub_batch.word_target[0],
                    "words_max": sub_batch.word_target[1],
                    "split_stats": split_result.stats if split_result.needs_split else None
                })
    
    return new_plan


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: DYNAMIC SUB-BATCH SPLITTER")
    print("=" * 60)
    
    # Test 1: H2 z małą gęstością (bez podziału)
    print("\n📝 Test 1: Mała gęstość (5 fraz)")
    result1 = split_h2_if_needed(
        h2_title="Czym jest porwanie rodzicielskie",
        batch_number=2,
        assigned_phrases=[
            {"keyword": "porwanie rodzicielskie", "type": "BASIC"},
            {"keyword": "definicja", "type": "EXTENDED"},
            {"keyword": "rodzic", "type": "EXTENDED"},
            {"keyword": "dziecko", "type": "EXTENDED"},
            {"keyword": "prawo", "type": "EXTENDED"},
        ],
        assigned_entities=[{"name": "sąd rodzinny"}],
        assigned_triplets=[{"subject": "rodzic", "verb": "zabiera", "object": "dziecko"}]
    )
    
    print(f"   Needs split: {result1.needs_split}")
    print(f"   Sub-batches: {len(result1.sub_batches)}")
    print(f"   Reason: {result1.reason}")
    
    # Test 2: H2 z dużą gęstością (wymaga podziału)
    print("\n📝 Test 2: Duża gęstość (12 fraz, 4 encje)")
    result2 = split_h2_if_needed(
        h2_title="Procedura sądowa w sprawach o miejsce pobytu dziecka",
        batch_number=3,
        assigned_phrases=[
            {"keyword": f"fraza_{i}", "type": "BASIC" if i < 5 else "EXTENDED"}
            for i in range(12)
        ],
        assigned_entities=[
            {"name": "sąd rodzinny"},
            {"name": "kurator"},
            {"name": "biegły"},
            {"name": "pełnomocnik"},
        ],
        assigned_triplets=[
            {"subject": "sąd", "verb": "ustala", "object": "miejsce pobytu"},
            {"subject": "kurator", "verb": "bada", "object": "sytuację"},
        ]
    )
    
    print(f"   Needs split: {result2.needs_split}")
    print(f"   Sub-batches: {len(result2.sub_batches)}")
    print(f"   Reason: {result2.reason}")
    print(f"   Density score: {result2.stats.get('density_score')}")
    
    if result2.needs_split:
        print(f"\n   Sub-batch details:")
        for sb in result2.sub_batches:
            print(f"     • {sb.sub_batch_id}: H3='{sb.h3_title}'")
            print(f"       Phrases: {len(sb.assigned_phrases)}, Entities: {len(sb.assigned_entities)}")
