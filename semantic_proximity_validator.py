# semantic_proximity_validator.py
# BRAJEN v34.0 - Semantic Entity SEO
# Walidator Bliskości Semantycznej (Semantic Proximity Validator)

"""
===============================================================================
🔗 SEMANTIC PROXIMITY VALIDATOR v34.0
===============================================================================

Sprawdza czy frazy kluczowe są "otoczone" odpowiednimi encjami wspierającymi.
Waliduje "gęstość semantyczną" tekstu.

Zasada: Fraza kluczowa NIE MOŻE występować "w próżni" - musi być otoczona
słowami potwierdzającymi kontekst merytoryczny.

Przykład:
❌ "Przeprowadzki są ważne. Każdy potrzebuje przeprowadzki."
   → Fraza "przeprowadzki" jest IZOLOWANA (brak kontekstu)

✅ "Profesjonalne przeprowadzki wymagają windy meblowej i pasów transportowych.
    Ubezpieczenie OCP chroni mienie podczas transportu."
   → Fraza "przeprowadzki" jest OTOCZONA encjami wspierającymi

===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# KONFIGURACJA
# ============================================================================

class SemanticStatus(Enum):
    """Status kontekstu semantycznego."""
    OK = "OK"                    # Fraza otoczona encjami (≥2)
    WARNING = "WARNING"          # Fraza z 1 encją
    ISOLATED = "ISOLATED"        # Fraza bez encji wspierających


@dataclass
class ProximityConfig:
    """Konfiguracja walidatora bliskości semantycznej."""
    DEFAULT_MAX_DISTANCE: int = 25      # Domyślny promień w słowach
    MIN_SUPPORTING_ENTITIES: int = 2    # Min encji dla statusu OK
    ISOLATED_PENALTY: float = 0.15      # Kara za izolowaną frazę
    LEAD_PARAGRAPH_WORDS: int = 100     # Słów w "Złotym Akapicie"
    
    # Progi dla semantic_score
    SCORE_OK_THRESHOLD: int = 75
    SCORE_WARNING_THRESHOLD: int = 50


CONFIG = ProximityConfig()


# ============================================================================
# TOKENIZACJA
# ============================================================================

def tokenize_with_positions(text: str) -> List[Tuple[str, int]]:
    """
    Tokenizuje tekst zachowując pozycje słów.
    
    Args:
        text: Tekst do tokenizacji
        
    Returns:
        Lista krotek (słowo, pozycja)
    """
    # Znajdź wszystkie słowa (alfanumeryczne)
    words = re.findall(r'\b\w+\b', text.lower())
    return [(word, i) for i, word in enumerate(words)]


def find_phrase_positions(tokens: List[Tuple[str, int]], phrase: str) -> List[int]:
    """
    Znajduje wszystkie pozycje frazy wielowyrazowej w tekście.
    
    Args:
        tokens: Lista (słowo, pozycja)
        phrase: Fraza do znalezienia (może być wielowyrazowa)
        
    Returns:
        Lista pozycji początkowych frazy
    """
    phrase_words = phrase.lower().split()
    if not phrase_words:
        return []
    
    positions = []
    
    for i in range(len(tokens) - len(phrase_words) + 1):
        match = True
        for j, phrase_word in enumerate(phrase_words):
            if tokens[i + j][0] != phrase_word:
                match = False
                break
        if match:
            positions.append(i)
    
    return positions


# ============================================================================
# WALIDACJA PROXIMITY DLA POJEDYNCZEJ FRAZY
# ============================================================================

def check_proximity_for_keyword(
    tokens: List[Tuple[str, int]],
    keyword_position: int,
    supporting_entities: List[str],
    max_distance: int = 25
) -> Dict[str, Any]:
    """
    Sprawdza czy w promieniu max_distance słów od frazy kluczowej
    znajdują się encje wspierające.
    
    Args:
        tokens: Lista tokenów z pozycjami
        keyword_position: Pozycja frazy kluczowej
        supporting_entities: Lista encji wspierających do sprawdzenia
        max_distance: Maksymalna odległość w słowach
        
    Returns:
        Dict z wynikami sprawdzenia
    """
    # Określ zakres do sprawdzenia (promień wokół frazy)
    start_pos = max(0, keyword_position - max_distance)
    end_pos = min(len(tokens), keyword_position + max_distance + 1)
    
    # Słowa w proximity (jako set dla szybkiego lookup)
    nearby_words = set(tokens[i][0] for i in range(start_pos, end_pos))
    
    # Sprawdź które encje wspierające są obecne
    found_entities = []
    for entity in supporting_entities:
        # Encja może być wielowyrazowa - sprawdź czy wszystkie słowa są w pobliżu
        entity_words = set(entity.lower().split())
        if entity_words.issubset(nearby_words):
            found_entities.append(entity)
        elif entity_words & nearby_words:  # częściowe dopasowanie
            # Sprawdź czy przynajmniej główne słowo encji jest obecne
            main_word = max(entity_words, key=len)  # najdłuższe słowo
            if main_word in nearby_words:
                found_entities.append(entity)
    
    # Oceń status
    found_count = len(found_entities)
    if found_count >= CONFIG.MIN_SUPPORTING_ENTITIES:
        status = SemanticStatus.OK
    elif found_count == 1:
        status = SemanticStatus.WARNING
    else:
        status = SemanticStatus.ISOLATED
    
    return {
        "position": keyword_position,
        "found_entities": found_entities,
        "found_count": found_count,
        "status": status.value,
        "max_distance_used": max_distance,
        "context_words": list(nearby_words)[:20]  # próbka słów w pobliżu
    }


# ============================================================================
# GŁÓWNA WALIDACJA SEMANTIC PROXIMITY
# ============================================================================

def validate_semantic_proximity(
    text: str,
    keywords: List[str],
    proximity_clusters: List[Dict],
    supporting_entities: List[str]
) -> Dict[str, Any]:
    """
    Główna funkcja walidacji bliskości semantycznej.
    
    Args:
        text: Tekst batcha do walidacji
        keywords: Lista fraz kluczowych do sprawdzenia
        proximity_clusters: Lista reguł proximity z concept_map
        supporting_entities: Płaska lista wszystkich encji wspierających
        
    Returns:
        Dict z wynikami walidacji
    """
    tokens = tokenize_with_positions(text)
    
    results = {
        "keywords_analysis": [],
        "proximity_clusters_analysis": [],
        "overall_status": SemanticStatus.OK.value,
        "semantic_score": 100,
        "isolated_keywords": [],
        "warnings": [],
        "suggestions": []
    }
    
    total_checks = 0
    ok_checks = 0
    warning_checks = 0
    isolated_checks = 0
    
    # -------------------------------------------------------------------------
    # 1. Sprawdź każdą frazę kluczową
    # -------------------------------------------------------------------------
    for keyword in keywords:
        positions = find_phrase_positions(tokens, keyword)
        
        if not positions:
            continue  # fraza nie występuje w tekście
        
        for pos in positions:
            total_checks += 1
            check_result = check_proximity_for_keyword(
                tokens=tokens,
                keyword_position=pos,
                supporting_entities=supporting_entities,
                max_distance=CONFIG.DEFAULT_MAX_DISTANCE
            )
            
            check_result["keyword"] = keyword
            results["keywords_analysis"].append(check_result)
            
            # Zlicz statusy
            if check_result["status"] == SemanticStatus.OK.value:
                ok_checks += 1
            elif check_result["status"] == SemanticStatus.WARNING.value:
                warning_checks += 1
            else:  # ISOLATED
                isolated_checks += 1
                results["isolated_keywords"].append({
                    "keyword": keyword,
                    "position": pos,
                    "found_entities": check_result["found_entities"],
                    "suggestion": f"Dodaj encje wspierające blisko '{keyword}': "
                                 f"{supporting_entities[:5]}"
                })
    
    # -------------------------------------------------------------------------
    # 2. Sprawdź specjalne reguły proximity_clusters
    # -------------------------------------------------------------------------
    for cluster in proximity_clusters:
        anchor = cluster.get("anchor", "")
        required_nearby = cluster.get("must_have_nearby", [])
        max_distance = cluster.get("max_distance", CONFIG.DEFAULT_MAX_DISTANCE)
        
        if not anchor or not required_nearby:
            continue
        
        anchor_positions = find_phrase_positions(tokens, anchor)
        
        for pos in anchor_positions:
            total_checks += 1
            check_result = check_proximity_for_keyword(
                tokens=tokens,
                keyword_position=pos,
                supporting_entities=required_nearby,
                max_distance=max_distance
            )
            
            check_result["anchor"] = anchor
            check_result["required"] = required_nearby
            check_result["is_cluster_rule"] = True
            results["proximity_clusters_analysis"].append(check_result)
            
            # Zlicz statusy
            if check_result["status"] == SemanticStatus.OK.value:
                ok_checks += 1
            elif check_result["status"] == SemanticStatus.WARNING.value:
                warning_checks += 1
            else:  # ISOLATED
                isolated_checks += 1
                results["warnings"].append(
                    f"PROXIMITY VIOLATION: '{anchor}' użyte bez wymaganych encji: "
                    f"{required_nearby}"
                )
    
    # -------------------------------------------------------------------------
    # 3. Oblicz semantic_score
    # -------------------------------------------------------------------------
    if total_checks > 0:
        # Base score = procent OK checks
        base_score = (ok_checks / total_checks) * 100
        
        # Kara za izolowane frazy
        penalty = isolated_checks * CONFIG.ISOLATED_PENALTY * 100
        
        # Warning daje mniejszą karę
        warning_penalty = warning_checks * (CONFIG.ISOLATED_PENALTY / 2) * 100
        
        results["semantic_score"] = max(0, round(base_score - penalty - warning_penalty, 1))
    
    # -------------------------------------------------------------------------
    # 4. Określ overall_status
    # -------------------------------------------------------------------------
    if results["semantic_score"] >= CONFIG.SCORE_OK_THRESHOLD:
        results["overall_status"] = SemanticStatus.OK.value
    elif results["semantic_score"] >= CONFIG.SCORE_WARNING_THRESHOLD:
        results["overall_status"] = SemanticStatus.WARNING.value
    else:
        results["overall_status"] = SemanticStatus.ISOLATED.value
    
    # -------------------------------------------------------------------------
    # 5. Generuj sugestie
    # -------------------------------------------------------------------------
    if isolated_checks > 0:
        results["suggestions"].append(
            f"Masz {isolated_checks} izolowanych wystąpień fraz. "
            f"Dodaj encje wspierające w ich pobliżu (w promieniu ~25 słów)."
        )
    
    if warning_checks > 0:
        results["suggestions"].append(
            f"{warning_checks} wystąpień ma tylko 1 encję wspierającą. "
            f"Dodaj więcej kontekstu semantycznego."
        )
    
    # Statystyki
    results["stats"] = {
        "total_checks": total_checks,
        "ok_checks": ok_checks,
        "warning_checks": warning_checks,
        "isolated_checks": isolated_checks
    }
    
    return results


# ============================================================================
# WALIDACJA ZŁOTEGO AKAPITU (Lead Paragraph)
# ============================================================================

def validate_lead_paragraph(
    text: str,
    classification_triplet: Dict[str, str],
    word_limit: int = None
) -> Dict[str, Any]:
    """
    Waliduje Złoty Akapit (pierwsze N słów).
    
    Sprawdza czy pierwsze 100 słów zawiera Trójkę Klasyfikacyjną:
    [Typ usługi] + [Kontekst/Lokalizacja] + [Główny atrybut]
    
    Args:
        text: Pełny tekst batcha
        classification_triplet: {"service_type": "...", "context": "...", "main_attribute": "..."}
        word_limit: Liczba słów do sprawdzenia (domyślnie z CONFIG)
        
    Returns:
        Dict z wynikami walidacji
    """
    if word_limit is None:
        word_limit = CONFIG.LEAD_PARAGRAPH_WORDS
    
    # Pobierz pierwsze N słów
    words = text.lower().split()[:word_limit]
    first_n_words = " ".join(words)
    
    # Elementy trójki
    triplet_elements = {
        "service_type": classification_triplet.get("service_type", "").lower(),
        "context": classification_triplet.get("context", "").lower(),
        "main_attribute": classification_triplet.get("main_attribute", "").lower()
    }
    
    found = []
    missing = []
    
    for key, value in triplet_elements.items():
        if value and value in first_n_words:
            found.append(value)
        elif value:
            missing.append(value)
    
    # Określ status
    found_count = len(found)
    if found_count >= 3:
        status = "OK"
    elif found_count == 2:
        status = "WARNING"
    else:
        status = "CRITICAL"
    
    return {
        "status": status,
        "triplet": classification_triplet,
        "found_in_first_100": found,
        "missing_in_first_100": missing,
        "coverage": f"{found_count}/3",
        "word_limit": word_limit,
        "message": (
            f"Złoty Akapit: znaleziono {found_count}/3 elementów Trójki Klasyfikacyjnej"
            + (f". Brakuje: {missing}" if missing else " ✓")
        )
    }


# ============================================================================
# PEŁNA WALIDACJA SEMANTYCZNA (dla approve_batch)
# ============================================================================

def full_semantic_validation(
    text: str,
    keywords: List[str],
    concept_map: Dict,
    batch_number: int = 1
) -> Dict[str, Any]:
    """
    Pełna walidacja semantyczna dla endpointu approve_batch.
    
    Args:
        text: Tekst batcha
        keywords: Lista fraz kluczowych (z keywords_list)
        concept_map: Mapa pojęć z S1
        batch_number: Numer batcha (1 = wymaga walidacji Lead Paragraph)
        
    Returns:
        Dict z kompletnymi wynikami walidacji
    """
    # Pobierz dane z concept_map
    proximity_clusters = concept_map.get("proximity_clusters", [])
    supporting_entities_dict = concept_map.get("supporting_entities", {})
    classification_triplet = concept_map.get("classification_triplet", {})
    
    # Spłaszcz encje wspierające do listy
    supporting_entities = []
    for category, entities in supporting_entities_dict.items():
        if isinstance(entities, list):
            supporting_entities.extend(entities)
    supporting_entities = list(set(supporting_entities))  # usuń duplikaty
    
    # -------------------------------------------------------------------------
    # 1. Walidacja Proximity
    # -------------------------------------------------------------------------
    proximity_result = validate_semantic_proximity(
        text=text,
        keywords=keywords,
        proximity_clusters=proximity_clusters,
        supporting_entities=supporting_entities
    )
    
    # -------------------------------------------------------------------------
    # 2. Walidacja Lead Paragraph (tylko dla Batch 1)
    # -------------------------------------------------------------------------
    lead_result = None
    if batch_number == 1:
        lead_result = validate_lead_paragraph(
            text=text,
            classification_triplet=classification_triplet
        )
    
    # -------------------------------------------------------------------------
    # 3. Oblicz łączny semantic_score
    # -------------------------------------------------------------------------
    proximity_score = proximity_result["semantic_score"]
    lead_score = 100
    
    if lead_result:
        lead_status = lead_result["status"]
        if lead_status == "OK":
            lead_score = 100
        elif lead_status == "WARNING":
            lead_score = 70
        else:  # CRITICAL
            lead_score = 40
    
    # Średnia ważona
    if batch_number == 1:
        # Dla batch 1: proximity 60%, lead paragraph 40%
        final_score = round(proximity_score * 0.6 + lead_score * 0.4, 1)
    else:
        final_score = proximity_score
    
    # -------------------------------------------------------------------------
    # 4. Określ overall status
    # -------------------------------------------------------------------------
    if final_score >= CONFIG.SCORE_OK_THRESHOLD:
        overall_status = "OK"
    elif final_score >= CONFIG.SCORE_WARNING_THRESHOLD:
        overall_status = "WARNING"
    else:
        overall_status = "LOW_SEMANTIC_CONTEXT"
    
    # -------------------------------------------------------------------------
    # 5. Zbierz wszystkie warnings
    # -------------------------------------------------------------------------
    all_warnings = proximity_result.get("warnings", [])
    
    if lead_result and lead_result["status"] != "OK":
        all_warnings.append(lead_result["message"])
    
    return {
        "proximity_validation": proximity_result,
        "lead_paragraph_validation": lead_result,
        "overall_semantic_status": overall_status,
        "semantic_score": final_score,
        "isolated_keywords": proximity_result.get("isolated_keywords", []),
        "warnings": all_warnings,
        "suggestions": proximity_result.get("suggestions", [])
    }


# ============================================================================
# AKTUALIZACJA context_status DLA KEYWORDS
# ============================================================================

def get_keyword_context_statuses(
    keywords_analysis: List[Dict]
) -> Dict[str, str]:
    """
    Generuje słownik context_status dla każdej frazy kluczowej.
    
    Args:
        keywords_analysis: Lista wyników z validate_semantic_proximity
        
    Returns:
        Dict {keyword: "OK"|"WARNING"|"ISOLATED"}
    """
    # Grupuj wyniki po keyword
    keyword_statuses = {}
    
    for analysis in keywords_analysis:
        keyword = analysis.get("keyword", "")
        status = analysis.get("status", "ISOLATED")
        
        if keyword not in keyword_statuses:
            keyword_statuses[keyword] = []
        keyword_statuses[keyword].append(status)
    
    # Dla każdej frazy: weź najgorszy status (pesymistyczne podejście)
    result = {}
    status_priority = {"ISOLATED": 0, "WARNING": 1, "OK": 2}
    
    for keyword, statuses in keyword_statuses.items():
        # Najniższy priorytet = najgorszy status
        worst = min(statuses, key=lambda s: status_priority.get(s, 0))
        result[keyword] = worst
    
    return result


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test z przykładowymi danymi
    test_text = """
    Profesjonalne przeprowadzki w Warszawie to usługa wymagająca doświadczenia 
    i odpowiedniego sprzętu. Nasza firma oferuje kompleksowy transport mebli 
    z wykorzystaniem windy meblowej i pasów transportowych.
    
    Transport fortepianu wymaga szczególnej ostrożności. Używamy specjalnych 
    pasów z grzechotką i foli ochronnej do zabezpieczenia instrumentu.
    
    Ubezpieczenie OCP do 100 000 zł chroni Państwa mienie podczas całej 
    przeprowadzki. Gwarantujemy terminowość i profesjonalizm.
    """
    
    test_keywords = ["przeprowadzki", "transport mebli", "fortepian"]
    
    test_concept_map = {
        "supporting_entities": {
            "tools": ["winda meblowa", "pasy transportowe", "folia ochronna"],
            "attributes": ["ubezpieczenie OCP", "doświadczenie", "terminowość"],
            "processes": ["pakowanie", "transport"]
        },
        "proximity_clusters": [
            {
                "anchor": "fortepian",
                "must_have_nearby": ["pasy", "ochronna", "instrument"],
                "max_distance": 20
            }
        ],
        "classification_triplet": {
            "service_type": "przeprowadzki",
            "context": "warszawa",
            "main_attribute": "profesjonalne"
        }
    }
    
    # Uruchom walidację
    result = full_semantic_validation(
        text=test_text,
        keywords=test_keywords,
        concept_map=test_concept_map,
        batch_number=1
    )
    
    print("="*60)
    print("SEMANTIC VALIDATION RESULT")
    print("="*60)
    print(f"Overall Status: {result['overall_semantic_status']}")
    print(f"Semantic Score: {result['semantic_score']}")
    print(f"\nProximity Stats: {result['proximity_validation']['stats']}")
    print(f"\nLead Paragraph: {result['lead_paragraph_validation']}")
    print(f"\nIsolated Keywords: {result['isolated_keywords']}")
    print(f"\nWarnings: {result['warnings']}")
    print(f"\nSuggestions: {result['suggestions']}")
    
    # Test context_statuses
    context_statuses = get_keyword_context_statuses(
        result['proximity_validation']['keywords_analysis']
    )
    print(f"\nContext Statuses: {context_statuses}")
