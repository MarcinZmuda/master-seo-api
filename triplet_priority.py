"""
===============================================================================
🔺 TRIPLET PRIORITY SYSTEM v41.0 - Hierarchia ważności relacji S-V-O
===============================================================================

System priorytetyzacji tripletów oparty na MIERZALNYCH danych z S1:
- importance (0-1) - ważność encji według S1
- sources_count - liczba źródeł gdzie encja występuje

HIERARCHIA:
1. MUST (wymagane) - triplety z encjami importance >= 0.7 AND sources >= 4
   → Brak = BLOKUJE batch
   
2. SHOULD (powinny być) - triplety z encjami importance >= 0.5 OR sources >= 3
   → Brak = WARNING + rekomendacja
   
3. NICE (opcjonalne) - pozostałe triplety
   → Brak = scoring bonus jeśli obecne

ZASADA KLUCZOWA:
Nie wymyślamy tripletów! Priorytetyzujemy tylko te, które przyszły z S1.

===============================================================================
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TripletPriority(Enum):
    MUST = "MUST"           # Wymagane - blokuje jeśli brak
    SHOULD = "SHOULD"       # Powinny być - warning jeśli brak
    NICE = "NICE"           # Opcjonalne - bonus jeśli obecne


class ValidationStatus(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class TripletPriorityConfig:
    """Konfiguracja systemu priorytetów tripletów."""
    
    # Progi dla MUST (wymagane)
    MUST_MIN_IMPORTANCE: float = 0.7
    MUST_MIN_SOURCES: int = 4
    
    # Progi dla SHOULD (powinny być)
    SHOULD_MIN_IMPORTANCE: float = 0.5
    SHOULD_MIN_SOURCES: int = 3
    
    # Ile MUST tripletów może brakować (domyślnie 0)
    MUST_MAX_MISSING: int = 0
    
    # Ile SHOULD tripletów może brakować
    SHOULD_MAX_MISSING: int = 2
    
    # Wagi dla scoringu
    MUST_WEIGHT: float = 1.0
    SHOULD_WEIGHT: float = 0.6
    NICE_WEIGHT: float = 0.3
    
    # Max tripletów do sprawdzenia (performance)
    MAX_TRIPLETS_TO_CHECK: int = 20


CONFIG = TripletPriorityConfig()


# ============================================================================
# STRUKTURY DANYCH
# ============================================================================

@dataclass
class PrioritizedTriplet:
    """Triplet z przypisanym priorytetem."""
    subject: str
    verb: str
    object: str
    priority: TripletPriority
    subject_importance: float = 0.0
    subject_sources: int = 0
    object_importance: float = 0.0
    object_sources: int = 0
    found_in_text: bool = False
    context_sentence: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "priority": self.priority.value,
            "found": self.found_in_text,
            "subject_importance": self.subject_importance,
            "object_importance": self.object_importance,
            "context": self.context_sentence
        }


# ============================================================================
# GŁÓWNE FUNKCJE
# ============================================================================

def prioritize_triplets(
    s1_relationships: List[Dict],
    s1_entities: List[Dict],
    config: TripletPriorityConfig = None
) -> Dict[str, List[PrioritizedTriplet]]:
    """
    Przypisuje priorytety tripletom na podstawie danych z S1.
    
    Args:
        s1_relationships: Lista relacji z S1 [{"subject": str, "verb": str, "object": str}]
        s1_entities: Lista encji z S1 [{"name": str, "importance": float, "sources_count": int}]
        config: Konfiguracja
        
    Returns:
        Dict z kluczami "MUST", "SHOULD", "NICE" i listami tripletów
    """
    if config is None:
        config = CONFIG
    
    # Zbuduj mapę entity -> (importance, sources)
    entity_map = _build_entity_map(s1_entities)
    
    # Kategoryzuj triplety
    categorized = {
        "MUST": [],
        "SHOULD": [],
        "NICE": []
    }
    
    for rel in s1_relationships[:config.MAX_TRIPLETS_TO_CHECK]:
        subject = rel.get("subject", "").strip()
        verb = rel.get("verb", "").strip()
        obj = rel.get("object", "").strip()
        
        if not subject or not obj:
            continue
        
        # Pobierz importance/sources dla subject i object
        subj_importance, subj_sources = entity_map.get(subject.lower(), (0.0, 0))
        obj_importance, obj_sources = entity_map.get(obj.lower(), (0.0, 0))
        
        # Określ priorytet na podstawie WYŻSZEJ wartości (subject lub object)
        max_importance = max(subj_importance, obj_importance)
        max_sources = max(subj_sources, obj_sources)
        
        priority = _determine_priority(max_importance, max_sources, config)
        
        triplet = PrioritizedTriplet(
            subject=subject,
            verb=verb,
            object=obj,
            priority=priority,
            subject_importance=subj_importance,
            subject_sources=subj_sources,
            object_importance=obj_importance,
            object_sources=obj_sources
        )
        
        categorized[priority.value].append(triplet)
    
    return categorized


def _build_entity_map(s1_entities: List[Dict]) -> Dict[str, Tuple[float, int]]:
    """
    Buduje mapę entity_name_lowercase -> (importance, sources_count).
    """
    entity_map = {}
    
    for entity in s1_entities:
        # Obsłuż różne formaty z S1
        name = entity.get("name") or entity.get("text") or str(entity)
        if isinstance(name, dict):
            name = name.get("name", "")
        
        name_lower = name.lower().strip()
        if not name_lower:
            continue
        
        importance = float(entity.get("importance", 0) or 0)
        sources = int(entity.get("sources_count", 0) or entity.get("sources", 0) or 0)
        
        # Zachowaj wyższe wartości jeśli duplikat
        if name_lower in entity_map:
            old_imp, old_src = entity_map[name_lower]
            importance = max(importance, old_imp)
            sources = max(sources, old_src)
        
        entity_map[name_lower] = (importance, sources)
    
    return entity_map


def _determine_priority(
    importance: float,
    sources: int,
    config: TripletPriorityConfig
) -> TripletPriority:
    """
    Określa priorytet na podstawie importance i sources.
    
    MUST: importance >= 0.7 AND sources >= 4
    SHOULD: importance >= 0.5 OR sources >= 3
    NICE: pozostałe
    """
    # MUST: Oba warunki muszą być spełnione
    if importance >= config.MUST_MIN_IMPORTANCE and sources >= config.MUST_MIN_SOURCES:
        return TripletPriority.MUST
    
    # SHOULD: Jeden z warunków
    if importance >= config.SHOULD_MIN_IMPORTANCE or sources >= config.SHOULD_MIN_SOURCES:
        return TripletPriority.SHOULD
    
    return TripletPriority.NICE


# ============================================================================
# WALIDACJA TRIPLETÓW W TEKŚCIE
# ============================================================================

def validate_triplets_in_text(
    text: str,
    prioritized_triplets: Dict[str, List[PrioritizedTriplet]],
    config: TripletPriorityConfig = None
) -> Dict[str, Any]:
    """
    Sprawdza które triplety występują w tekście.
    
    Args:
        text: Tekst do walidacji
        prioritized_triplets: Wynik z prioritize_triplets()
        config: Konfiguracja
        
    Returns:
        Dict z wynikami walidacji i statusem
    """
    if config is None:
        config = CONFIG
    
    text_lower = text.lower()
    
    # Waliduj każdą kategorię
    results = {
        "MUST": {"found": [], "missing": [], "count": 0, "found_count": 0},
        "SHOULD": {"found": [], "missing": [], "count": 0, "found_count": 0},
        "NICE": {"found": [], "missing": [], "count": 0, "found_count": 0}
    }
    
    for priority_name, triplets in prioritized_triplets.items():
        for triplet in triplets:
            results[priority_name]["count"] += 1
            
            # Sprawdź czy subject i object są w tekście w odległości < 200 znaków
            found, context = _check_triplet_presence(
                text_lower, 
                triplet.subject, 
                triplet.object,
                text  # Oryginalny tekst dla kontekstu
            )
            
            triplet.found_in_text = found
            triplet.context_sentence = context
            
            if found:
                results[priority_name]["found"].append(triplet)
                results[priority_name]["found_count"] += 1
            else:
                results[priority_name]["missing"].append(triplet)
    
    # Określ status
    status, message, issues = _evaluate_validation(results, config)
    
    # Oblicz score
    score = _calculate_triplet_score(results, config)
    
    return {
        "status": status.value,
        "message": message,
        "score": score,
        "results": {
            k: {
                "count": v["count"],
                "found_count": v["found_count"],
                "missing_count": len(v["missing"]),
                "found": [t.to_dict() for t in v["found"]],
                "missing": [t.to_dict() for t in v["missing"]]
            }
            for k, v in results.items()
        },
        "issues": issues,
        "summary": {
            "must_coverage": f"{results['MUST']['found_count']}/{results['MUST']['count']}",
            "should_coverage": f"{results['SHOULD']['found_count']}/{results['SHOULD']['count']}",
            "nice_coverage": f"{results['NICE']['found_count']}/{results['NICE']['count']}"
        }
    }


def _check_triplet_presence(
    text_lower: str,
    subject: str,
    obj: str,
    original_text: str,
    max_distance: int = 200
) -> Tuple[bool, Optional[str]]:
    """
    Sprawdza czy subject i object występują blisko siebie w tekście.
    
    Returns:
        (found: bool, context_sentence: str lub None)
    """
    subject_lower = subject.lower()
    object_lower = obj.lower()
    
    # Znajdź wszystkie wystąpienia subject
    subject_positions = [m.start() for m in re.finditer(
        rf'\b{re.escape(subject_lower)}\b', text_lower
    )]
    
    # Znajdź wszystkie wystąpienia object
    object_positions = [m.start() for m in re.finditer(
        rf'\b{re.escape(object_lower)}\b', text_lower
    )]
    
    if not subject_positions or not object_positions:
        return False, None
    
    # Sprawdź czy są blisko siebie
    for subj_pos in subject_positions:
        for obj_pos in object_positions:
            if abs(subj_pos - obj_pos) < max_distance:
                # Wyciągnij kontekst (zdanie)
                start = max(0, min(subj_pos, obj_pos) - 50)
                end = min(len(original_text), max(subj_pos, obj_pos) + 100)
                context = original_text[start:end].strip()
                # Przytnij do pełnych słów
                if start > 0:
                    context = "..." + context
                if end < len(original_text):
                    context = context + "..."
                return True, context
    
    return False, None


def _evaluate_validation(
    results: Dict,
    config: TripletPriorityConfig
) -> Tuple[ValidationStatus, str, List[str]]:
    """
    Ocenia wyniki walidacji i zwraca status.
    """
    issues = []
    
    must_missing = len(results["MUST"]["missing"])
    should_missing = len(results["SHOULD"]["missing"])
    
    # CRITICAL jeśli brakuje MUST tripletów
    if must_missing > config.MUST_MAX_MISSING:
        status = ValidationStatus.CRITICAL
        missing_names = [f"{t.subject}→{t.object}" for t in results["MUST"]["missing"][:3]]
        message = f"🚫 BRAKUJE {must_missing} WYMAGANYCH tripletów: {', '.join(missing_names)}"
        issues.append(f"MUST triplets missing: {must_missing}")
        
    # WARNING jeśli brakuje zbyt wielu SHOULD
    elif should_missing > config.SHOULD_MAX_MISSING:
        status = ValidationStatus.WARNING
        missing_names = [f"{t.subject}→{t.object}" for t in results["SHOULD"]["missing"][:3]]
        message = f"⚠️ Brakuje {should_missing} zalecanych tripletów: {', '.join(missing_names)}"
        issues.append(f"SHOULD triplets missing: {should_missing}")
        
    else:
        status = ValidationStatus.OK
        total_found = sum(r["found_count"] for r in results.values())
        total_count = sum(r["count"] for r in results.values())
        message = f"✅ Triplety OK ({total_found}/{total_count})"
    
    return status, message, issues


def _calculate_triplet_score(
    results: Dict,
    config: TripletPriorityConfig
) -> int:
    """
    Oblicza znormalizowany score (0-100).
    """
    total_weighted = 0
    total_max = 0
    
    for priority_name, data in results.items():
        weight = getattr(config, f"{priority_name}_WEIGHT")
        count = data["count"]
        found = data["found_count"]
        
        if count > 0:
            coverage = found / count
            total_weighted += coverage * weight * count
            total_max += weight * count
    
    if total_max == 0:
        return 50  # Brak tripletów do sprawdzenia
    
    score = (total_weighted / total_max) * 100
    return min(100, max(0, int(score)))


# ============================================================================
# GENEROWANIE INSTRUKCJI PRE-BATCH
# ============================================================================

def get_triplet_instructions_for_prebatch(
    validation_result: Dict[str, Any],
    batch_number: int
) -> Optional[str]:
    """
    Generuje instrukcję dla GPT jeśli brakuje ważnych tripletów.
    
    Args:
        validation_result: Wynik z validate_triplets_in_text()
        batch_number: Numer batcha
        
    Returns:
        Instrukcja dla GPT lub None jeśli OK
    """
    if validation_result["status"] == "OK":
        return None
    
    instructions = []
    
    # MUST missing
    must_missing = validation_result["results"]["MUST"]["missing"]
    if must_missing:
        instructions.append("🚨 WYMAGANE RELACJE (MUSISZ użyć w tym batchu):")
        for triplet_dict in must_missing[:3]:
            instructions.append(
                f"   • {triplet_dict['subject']} → {triplet_dict['verb']} → {triplet_dict['object']}"
            )
    
    # SHOULD missing (tylko jeśli > 2)
    should_missing = validation_result["results"]["SHOULD"]["missing"]
    if len(should_missing) > 2:
        instructions.append("\n⚠️ ZALECANE RELACJE (użyj min. 1):")
        for triplet_dict in should_missing[:2]:
            instructions.append(
                f"   • {triplet_dict['subject']} → {triplet_dict['verb']} → {triplet_dict['object']}"
            )
    
    if not instructions:
        return None
    
    return "\n".join(instructions)


# ============================================================================
# GŁÓWNA FUNKCJA DO INTEGRACJI
# ============================================================================

def analyze_triplets_with_priority(
    text: str,
    s1_relationships: List[Dict],
    s1_entities: List[Dict],
    config: TripletPriorityConfig = None
) -> Dict[str, Any]:
    """
    Główna funkcja - priorytetyzuje i waliduje triplety.
    
    To jest funkcja do wywołania z entity_scoring.py zamiast
    analyze_entity_relationships().
    
    Args:
        text: Tekst do walidacji
        s1_relationships: Relacje z S1
        s1_entities: Encje z S1
        config: Konfiguracja
        
    Returns:
        Dict kompatybilny z istniejącym API + nowe pola
    """
    if config is None:
        config = CONFIG
    
    if not s1_relationships:
        return {
            "score": 50,
            "status": "NO_DATA",
            "message": "Brak relacji z S1 do walidacji",
            "found_relationships": [],
            "missing_relationships": [],
            "priority_breakdown": None
        }
    
    # Priorytetyzuj
    prioritized = prioritize_triplets(s1_relationships, s1_entities, config)
    
    # Waliduj
    validation = validate_triplets_in_text(text, prioritized, config)
    
    # Mapuj na format kompatybilny z istniejącym API
    all_found = []
    all_missing = []
    
    for priority in ["MUST", "SHOULD", "NICE"]:
        for triplet in validation["results"][priority]["found"]:
            all_found.append({
                "subject": triplet["subject"],
                "verb": triplet["verb"],
                "object": triplet["object"],
                "priority": priority
            })
        for triplet in validation["results"][priority]["missing"]:
            all_missing.append({
                "subject": triplet["subject"],
                "verb": triplet["verb"],
                "object": triplet["object"],
                "priority": priority
            })
    
    return {
        # Kompatybilność wsteczna
        "score": validation["score"],
        "status": validation["status"],
        "message": validation["message"],
        "found_relationships": all_found,
        "missing_relationships": all_missing,
        
        # Nowe pola v41
        "priority_breakdown": validation["summary"],
        "issues": validation["issues"],
        "prebatch_instruction": get_triplet_instructions_for_prebatch(
            validation, batch_number=1
        )
    }


# ============================================================================
# INSTRUKCJA INTEGRACJI
# ============================================================================

"""
INTEGRACJA Z BRAJEN:

1. W entity_scoring.py, zamień analyze_entity_relationships() na:

   from triplet_priority_v41 import analyze_triplets_with_priority
   
   # W funkcji calculate_entity_score():
   relationships_result = analyze_triplets_with_priority(
       text=text,
       s1_relationships=s1_data.get("entity_seo", {}).get("entity_relationships", []),
       s1_entities=s1_data.get("entity_seo", {}).get("entities", [])
   )

2. W enhanced_pre_batch.py, dodaj instrukcje o tripletach:

   from triplet_priority_v41 import (
       prioritize_triplets, 
       validate_triplets_in_text,
       get_triplet_instructions_for_prebatch
   )
   
   # W generate_pre_batch_info():
   if accumulated_content and batch_number >= 2:
       triplet_result = analyze_triplets_with_priority(
           accumulated_content, s1_relationships, s1_entities
       )
       if triplet_result.get("prebatch_instruction"):
           entity_instructions.append(triplet_result["prebatch_instruction"])

3. Score pozostaje kompatybilny z istniejącymi wagami.
"""


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Przykładowe dane z S1
    s1_entities = [
        {"name": "ubezwłasnowolnienie", "importance": 0.9, "sources_count": 8},
        {"name": "sąd okręgowy", "importance": 0.75, "sources_count": 6},
        {"name": "kurator", "importance": 0.65, "sources_count": 4},
        {"name": "biegły sądowy", "importance": 0.55, "sources_count": 3},
        {"name": "zdolność do czynności prawnych", "importance": 0.7, "sources_count": 5},
        {"name": "opinia psychiatryczna", "importance": 0.45, "sources_count": 2},
    ]
    
    s1_relationships = [
        {"subject": "sąd okręgowy", "verb": "orzeka", "object": "ubezwłasnowolnienie"},
        {"subject": "ubezwłasnowolnienie", "verb": "pozbawia", "object": "zdolność do czynności prawnych"},
        {"subject": "kurator", "verb": "sprawuje opiekę nad", "object": "ubezwłasnowolnienie"},
        {"subject": "biegły sądowy", "verb": "wydaje", "object": "opinia psychiatryczna"},
    ]
    
    # Tekst do walidacji
    test_text = """
    Sąd okręgowy orzeka ubezwłasnowolnienie po przeprowadzeniu postępowania.
    Procedura wymaga opinii biegłych psychiatrów. Kurator sprawuje opiekę
    nad osobą ubezwłasnowolnioną. Ubezwłasnowolnienie pozbawia zdolności
    do czynności prawnych w zakresie określonym przez sąd.
    """
    
    print("=" * 60)
    print("TEST: TRIPLET PRIORITY SYSTEM v41")
    print("=" * 60)
    
    # 1. Priorytetyzacja
    print("\n1. PRIORYTETYZACJA TRIPLETÓW:")
    prioritized = prioritize_triplets(s1_relationships, s1_entities)
    for priority in ["MUST", "SHOULD", "NICE"]:
        print(f"\n   {priority}:")
        for t in prioritized[priority]:
            print(f"      {t.subject} → {t.verb} → {t.object}")
            print(f"         (imp={t.subject_importance:.2f}, src={t.subject_sources})")
    
    # 2. Walidacja
    print("\n2. WALIDACJA W TEKŚCIE:")
    validation = validate_triplets_in_text(test_text, prioritized)
    print(f"   Status: {validation['status']}")
    print(f"   Score: {validation['score']}")
    print(f"   Summary: {validation['summary']}")
    print(f"   Message: {validation['message']}")
    
    # 3. Główna funkcja
    print("\n3. GŁÓWNA FUNKCJA (kompatybilna z API):")
    result = analyze_triplets_with_priority(test_text, s1_relationships, s1_entities)
    print(f"   Score: {result['score']}")
    print(f"   Found: {len(result['found_relationships'])}")
    print(f"   Missing: {len(result['missing_relationships'])}")
    
    if result.get("prebatch_instruction"):
        print(f"\n   PRE-BATCH INSTRUCTION:")
        print(result["prebatch_instruction"])
