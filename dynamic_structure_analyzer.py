"""
===============================================================================
🏗️ DYNAMIC STRUCTURE ANALYZER v1.0 - Elastyczna struktura H3/List
===============================================================================

Moduł analizujący strukturę konkurencji z S1 Analysis i dynamicznie określający
wymagania dla H3/list w artykule.

STARY SYSTEM (sztywne reguły):
    - Cały artykuł = DOKŁADNIE 1 lista + DOKŁADNIE 1 sekcja z H3
    - Brak elastyczności

NOWY SYSTEM (dynamiczny):
    - Analiza competitor_h2_patterns z S1
    - Jeśli >50% konkurentów używa H3 → wymagaj H3
    - Jeśli >50% konkurentów używa list → wymagaj listy
    - Skalowanie limitów na podstawie długości artykułu
    - Wsparcie dla różnych typów treści (guides, reviews, how-to, etc.)

UŻYCIE:
    from dynamic_structure_analyzer import analyze_structure_requirements
    
    requirements = analyze_structure_requirements(
        s1_data=project["s1_data"],
        target_length=2500,
        content_type="informational"  # opcjonalne
    )
    
    # Wynik:
    # {
    #     "h3": {"required": True, "max_sections": 2, "source": "competitor_analysis"},
    #     "list": {"required": False, "max_lists": 1, "source": "competitor_analysis"},
    #     ...
    # }

===============================================================================
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


# ============================================================================
# KONFIGURACJA
# ============================================================================

class ContentType(Enum):
    """Typy treści z różnymi domyślnymi wymaganiami struktury."""
    INFORMATIONAL = "informational"      # Artykuły informacyjne
    HOW_TO = "how_to"                    # Poradniki krok-po-kroku
    LISTICLE = "listicle"                # Listy (top 10, ranking)
    REVIEW = "review"                    # Recenzje
    COMPARISON = "comparison"            # Porównania
    NEWS = "news"                        # Newsy
    LEGAL = "legal"                      # Treści prawne
    MEDICAL = "medical"                  # Treści medyczne
    PRODUCT = "product"                  # Opisy produktów


@dataclass
class StructureDefaults:
    """Domyślne wymagania struktury per typ treści (gdy brak danych S1)."""
    h3_required: bool
    h3_max_sections: int
    list_required: bool
    list_max_count: int
    list_types_allowed: List[str]  # "bullet", "numbered", "checklist"


# Domyślne ustawienia per typ treści
CONTENT_TYPE_DEFAULTS: Dict[ContentType, StructureDefaults] = {
    ContentType.INFORMATIONAL: StructureDefaults(
        h3_required=False,
        h3_max_sections=2,
        list_required=False,
        list_max_count=1,
        list_types_allowed=["bullet"]
    ),
    ContentType.HOW_TO: StructureDefaults(
        h3_required=True,
        h3_max_sections=3,
        list_required=True,
        list_max_count=2,
        list_types_allowed=["numbered", "checklist"]
    ),
    ContentType.LISTICLE: StructureDefaults(
        h3_required=True,
        h3_max_sections=5,
        list_required=True,
        list_max_count=3,
        list_types_allowed=["bullet", "numbered"]
    ),
    ContentType.REVIEW: StructureDefaults(
        h3_required=True,
        h3_max_sections=2,
        list_required=True,
        list_max_count=2,
        list_types_allowed=["bullet"]
    ),
    ContentType.COMPARISON: StructureDefaults(
        h3_required=True,
        h3_max_sections=3,
        list_required=True,
        list_max_count=2,
        list_types_allowed=["bullet"]
    ),
    ContentType.NEWS: StructureDefaults(
        h3_required=False,
        h3_max_sections=1,
        list_required=False,
        list_max_count=1,
        list_types_allowed=["bullet"]
    ),
    ContentType.LEGAL: StructureDefaults(
        h3_required=True,
        h3_max_sections=2,
        list_required=True,  # Często wyliczenia przepisów
        list_max_count=2,
        list_types_allowed=["numbered", "bullet"]
    ),
    ContentType.MEDICAL: StructureDefaults(
        h3_required=True,
        h3_max_sections=2,
        list_required=True,  # Objawy, leczenie
        list_max_count=2,
        list_types_allowed=["bullet"]
    ),
    ContentType.PRODUCT: StructureDefaults(
        h3_required=True,
        h3_max_sections=3,
        list_required=True,  # Specyfikacje
        list_max_count=2,
        list_types_allowed=["bullet"]
    ),
}


# ============================================================================
# PROGI DECYZYJNE
# ============================================================================

@dataclass
class AnalysisThresholds:
    """Progi dla analizy konkurencji."""
    
    # Minimum konkurentów do wiarygodnej analizy
    min_competitors_for_analysis: int = 3
    
    # Procent konkurentów wymagany do wymuszenia elementu
    h3_adoption_threshold: float = 0.50      # >50% używa H3 → wymagaj
    list_adoption_threshold: float = 0.50    # >50% używa list → wymagaj
    
    # Progi długości artykułu dla skalowania limitów
    short_article_max_words: int = 1500
    medium_article_max_words: int = 3000
    long_article_max_words: int = 5000
    
    # Mnożniki limitów dla długich artykułów
    long_article_h3_multiplier: float = 1.5
    long_article_list_multiplier: float = 1.5


THRESHOLDS = AnalysisThresholds()


# ============================================================================
# STRUKTURY WYNIKOWE
# ============================================================================

@dataclass
class H3Requirements:
    """Wymagania dla nagłówków H3."""
    required: bool
    max_sections: int
    min_sections: int
    recommended_sections: List[int]  # np. [1, 2] = "1 lub 2 sekcje"
    source: str  # "competitor_analysis" | "content_type_default" | "length_based"
    confidence: float  # 0.0-1.0
    rationale: str


@dataclass
class ListRequirements:
    """Wymagania dla list."""
    required: bool
    max_count: int
    min_count: int
    types_allowed: List[str]
    recommended_placement: List[str]  # np. ["middle", "end"]
    source: str
    confidence: float
    rationale: str


@dataclass
class StructureRequirements:
    """Pełne wymagania struktury artykułu."""
    h3: H3Requirements
    list: ListRequirements
    target_length: int
    content_type: str
    analysis_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "h3": asdict(self.h3),
            "list": asdict(self.list),
            "target_length": self.target_length,
            "content_type": self.content_type,
            "analysis_summary": self.analysis_summary
        }
    
    def get_structure_instructions(self) -> Dict[str, Any]:
        """
        Zwraca instrukcje struktury w formacie kompatybilnym z pre_batch_info.
        """
        return {
            "h3_required": self.h3.required,
            "h3_max_sections": self.h3.max_sections,
            "h3_min_sections": self.h3.min_sections,
            "list_required": self.list.required,
            "list_max_count": self.list.max_count,
            "list_min_count": self.list.min_count,
            "list_types_allowed": self.list.types_allowed,
            "source": self.h3.source,
            "confidence": min(self.h3.confidence, self.list.confidence)
        }


# ============================================================================
# ANALIZA KONKURENCJI
# ============================================================================

def _analyze_competitor_patterns(s1_data: Dict) -> Dict[str, Any]:
    """
    Analizuje wzorce struktury konkurencji z S1.
    
    Args:
        s1_data: Dane z S1 Analysis (zawiera competitor_h2_patterns)
        
    Returns:
        Dict z analizą:
        {
            "competitors_count": int,
            "h3_usage_count": int,
            "h3_usage_ratio": float,
            "list_usage_count": int,
            "list_usage_ratio": float,
            "avg_h3_per_article": float,
            "avg_lists_per_article": float,
            "sufficient_data": bool
        }
    """
    competitor_patterns = s1_data.get("competitor_h2_patterns", [])
    
    if not competitor_patterns:
        # Próbuj alternatywne lokalizacje danych
        competitor_patterns = s1_data.get("competitors", [])
        if not competitor_patterns:
            competitor_patterns = s1_data.get("serp_analysis", {}).get("competitors", [])
    
    total = len(competitor_patterns)
    
    if total == 0:
        return {
            "competitors_count": 0,
            "h3_usage_count": 0,
            "h3_usage_ratio": 0.0,
            "list_usage_count": 0,
            "list_usage_ratio": 0.0,
            "avg_h3_per_article": 0.0,
            "avg_lists_per_article": 0.0,
            "sufficient_data": False,
            "raw_data": []
        }
    
    # Zlicz użycie H3 i list
    h3_count = 0
    list_count = 0
    total_h3_sections = 0
    total_lists = 0
    
    for pattern in competitor_patterns:
        # Obsługa różnych formatów danych
        has_h3 = (
            pattern.get("has_h3", False) or 
            pattern.get("h3_count", 0) > 0 or
            len(pattern.get("h3_headers", [])) > 0
        )
        has_list = (
            pattern.get("has_list", False) or 
            pattern.get("list_count", 0) > 0 or
            pattern.get("has_bullets", False)
        )
        
        if has_h3:
            h3_count += 1
            total_h3_sections += pattern.get("h3_count", 1)
        
        if has_list:
            list_count += 1
            total_lists += pattern.get("list_count", 1)
    
    return {
        "competitors_count": total,
        "h3_usage_count": h3_count,
        "h3_usage_ratio": h3_count / total if total > 0 else 0.0,
        "list_usage_count": list_count,
        "list_usage_ratio": list_count / total if total > 0 else 0.0,
        "avg_h3_per_article": total_h3_sections / total if total > 0 else 0.0,
        "avg_lists_per_article": total_lists / total if total > 0 else 0.0,
        "sufficient_data": total >= THRESHOLDS.min_competitors_for_analysis,
        "raw_data": competitor_patterns
    }


def _detect_content_type(s1_data: Dict) -> ContentType:
    """
    Wykrywa typ treści na podstawie S1 Analysis.
    
    Args:
        s1_data: Dane z S1 Analysis
        
    Returns:
        ContentType enum
    """
    search_intent = s1_data.get("search_intent", "").lower()
    topic = s1_data.get("topic", "").lower()
    main_keyword = s1_data.get("main_keyword", "").lower()
    
    # Wykrywanie na podstawie intencji
    intent_mapping = {
        "transactional": ContentType.PRODUCT,
        "commercial": ContentType.COMPARISON,
        "navigational": ContentType.INFORMATIONAL,
        "informational": ContentType.INFORMATIONAL,
    }
    
    # Wykrywanie na podstawie słów kluczowych w temacie
    keyword_patterns = {
        ContentType.HOW_TO: ["jak", "how to", "poradnik", "tutorial", "krok po kroku", "instrukcja"],
        ContentType.LISTICLE: ["top", "najlepsze", "ranking", "lista", "best", "zestawienie"],
        ContentType.REVIEW: ["recenzja", "review", "opinia", "test", "ocena"],
        ContentType.COMPARISON: ["porównanie", "vs", "versus", "czy", "różnica między"],
        ContentType.LEGAL: ["prawo", "ustawa", "kodeks", "sąd", "prawny", "przepis", "artykuł"],
        ContentType.MEDICAL: ["leczenie", "objawy", "choroba", "zdrowie", "lekarz", "diagnoza"],
        ContentType.NEWS: ["news", "aktualności", "wiadomości", "wydarzenie"],
        ContentType.PRODUCT: ["cena", "sklep", "kup", "produkt", "specyfikacja"],
    }
    
    combined_text = f"{topic} {main_keyword}".lower()
    
    for content_type, patterns in keyword_patterns.items():
        for pattern in patterns:
            if pattern in combined_text:
                return content_type
    
    # Fallback na podstawie intencji
    return intent_mapping.get(search_intent, ContentType.INFORMATIONAL)


def _scale_limits_by_length(
    base_h3_max: int,
    base_list_max: int,
    target_length: int
) -> tuple:
    """
    Skaluje limity H3/list na podstawie długości artykułu.
    
    Args:
        base_h3_max: Bazowy limit H3
        base_list_max: Bazowy limit list
        target_length: Docelowa długość artykułu (słowa)
        
    Returns:
        (scaled_h3_max, scaled_list_max)
    """
    if target_length <= THRESHOLDS.short_article_max_words:
        # Krótkie artykuły - zachowaj lub zmniejsz limity
        return (max(1, base_h3_max - 1), max(1, base_list_max))
    
    elif target_length <= THRESHOLDS.medium_article_max_words:
        # Średnie artykuły - standardowe limity
        return (base_h3_max, base_list_max)
    
    elif target_length <= THRESHOLDS.long_article_max_words:
        # Długie artykuły - zwiększ limity
        return (
            int(base_h3_max * THRESHOLDS.long_article_h3_multiplier),
            int(base_list_max * THRESHOLDS.long_article_list_multiplier)
        )
    
    else:
        # Bardzo długie artykuły - jeszcze większe limity
        return (
            int(base_h3_max * THRESHOLDS.long_article_h3_multiplier * 1.5),
            int(base_list_max * THRESHOLDS.long_article_list_multiplier * 1.5)
        )


# ============================================================================
# GŁÓWNA FUNKCJA ANALIZY
# ============================================================================

def analyze_structure_requirements(
    s1_data: Dict,
    target_length: int = 2500,
    content_type: Optional[str] = None,
    force_h3: Optional[bool] = None,
    force_list: Optional[bool] = None
) -> StructureRequirements:
    """
    Analizuje wymagania struktury na podstawie S1 i typu treści.
    
    Args:
        s1_data: Dane z S1 Analysis
        target_length: Docelowa długość artykułu (słowa)
        content_type: Opcjonalny typ treści (jeśli None, wykrywany automatycznie)
        force_h3: Opcjonalne wymuszenie H3 (True/False/None=auto)
        force_list: Opcjonalne wymuszenie list (True/False/None=auto)
        
    Returns:
        StructureRequirements z pełnymi wymaganiami
        
    Example:
        >>> req = analyze_structure_requirements(s1_data, target_length=3000)
        >>> req.h3.required  # True/False
        >>> req.list.max_count  # 1, 2, 3...
        >>> req.get_structure_instructions()  # Dict dla pre_batch_info
    """
    # 1. Analiza konkurencji
    competitor_analysis = _analyze_competitor_patterns(s1_data)
    
    # 2. Wykryj typ treści
    if content_type:
        try:
            detected_type = ContentType(content_type)
        except ValueError:
            detected_type = ContentType.INFORMATIONAL
    else:
        detected_type = _detect_content_type(s1_data)
    
    # 3. Pobierz domyślne ustawienia dla typu treści
    defaults = CONTENT_TYPE_DEFAULTS.get(detected_type, CONTENT_TYPE_DEFAULTS[ContentType.INFORMATIONAL])
    
    # 4. Określ wymagania H3
    if force_h3 is not None:
        # Użytkownik wymusił
        h3_required = force_h3
        h3_source = "user_override"
        h3_confidence = 1.0
        h3_rationale = "Wymuszone przez użytkownika"
    elif competitor_analysis["sufficient_data"]:
        # Na podstawie konkurencji
        h3_required = competitor_analysis["h3_usage_ratio"] >= THRESHOLDS.h3_adoption_threshold
        h3_source = "competitor_analysis"
        h3_confidence = min(1.0, competitor_analysis["competitors_count"] / 6)  # Max confidence przy 6+ konkurentach
        h3_rationale = (
            f"{competitor_analysis['h3_usage_count']}/{competitor_analysis['competitors_count']} "
            f"konkurentów ({competitor_analysis['h3_usage_ratio']:.0%}) używa H3. "
            f"Próg: {THRESHOLDS.h3_adoption_threshold:.0%}"
        )
    else:
        # Fallback na domyślne dla typu treści
        h3_required = defaults.h3_required
        h3_source = "content_type_default"
        h3_confidence = 0.5
        h3_rationale = f"Brak danych konkurencji. Domyślne dla typu '{detected_type.value}': {defaults.h3_required}"
    
    # 5. Określ wymagania list
    if force_list is not None:
        list_required = force_list
        list_source = "user_override"
        list_confidence = 1.0
        list_rationale = "Wymuszone przez użytkownika"
    elif competitor_analysis["sufficient_data"]:
        list_required = competitor_analysis["list_usage_ratio"] >= THRESHOLDS.list_adoption_threshold
        list_source = "competitor_analysis"
        list_confidence = min(1.0, competitor_analysis["competitors_count"] / 6)
        list_rationale = (
            f"{competitor_analysis['list_usage_count']}/{competitor_analysis['competitors_count']} "
            f"konkurentów ({competitor_analysis['list_usage_ratio']:.0%}) używa list. "
            f"Próg: {THRESHOLDS.list_adoption_threshold:.0%}"
        )
    else:
        list_required = defaults.list_required
        list_source = "content_type_default"
        list_confidence = 0.5
        list_rationale = f"Brak danych konkurencji. Domyślne dla typu '{detected_type.value}': {defaults.list_required}"
    
    # 6. Skaluj limity na podstawie długości
    base_h3_max = defaults.h3_max_sections
    base_list_max = defaults.list_max_count
    
    if competitor_analysis["sufficient_data"]:
        # Użyj średniej z konkurencji jako bazy
        base_h3_max = max(1, int(competitor_analysis["avg_h3_per_article"] + 0.5))
        base_list_max = max(1, int(competitor_analysis["avg_lists_per_article"] + 0.5))
    
    scaled_h3_max, scaled_list_max = _scale_limits_by_length(
        base_h3_max, base_list_max, target_length
    )
    
    # 7. Zbuduj wynik
    h3_requirements = H3Requirements(
        required=h3_required,
        max_sections=scaled_h3_max,
        min_sections=1 if h3_required else 0,
        recommended_sections=list(range(1, scaled_h3_max + 1)) if h3_required else [0, 1],
        source=h3_source,
        confidence=h3_confidence,
        rationale=h3_rationale
    )
    
    list_requirements = ListRequirements(
        required=list_required,
        max_count=scaled_list_max,
        min_count=1 if list_required else 0,
        types_allowed=defaults.list_types_allowed,
        recommended_placement=["middle"] if list_required else [],
        source=list_source,
        confidence=list_confidence,
        rationale=list_rationale
    )
    
    analysis_summary = {
        "competitor_analysis": competitor_analysis,
        "detected_content_type": detected_type.value,
        "target_length": target_length,
        "scaling_applied": target_length > THRESHOLDS.medium_article_max_words,
        "data_quality": "high" if competitor_analysis["sufficient_data"] else "low"
    }
    
    return StructureRequirements(
        h3=h3_requirements,
        list=list_requirements,
        target_length=target_length,
        content_type=detected_type.value,
        analysis_summary=analysis_summary
    )


# ============================================================================
# FUNKCJE POMOCNICZE DLA INTEGRACJI
# ============================================================================

def should_batch_have_h3(
    batch_number: int,
    total_batches: int,
    requirements: StructureRequirements
) -> bool:
    """
    Określa czy dany batch powinien mieć H3.
    
    Logika:
    - Jeśli H3 wymagane i max_sections >= 1: przydziel do najdłuższych batchy
    - Rozłóż równomiernie jeśli więcej niż 1 sekcja dozwolona
    """
    if not requirements.h3.required and requirements.h3.max_sections == 0:
        return False
    
    if requirements.h3.max_sections >= total_batches:
        # Każdy batch może mieć H3
        return True
    
    # Przydziel do środkowych/dłuższych batchy (nie INTRO, nie FINAL)
    if batch_number == 1 or batch_number == total_batches:
        return False
    
    # Dla pozostałych - przydziel proporcjonalnie
    content_batches = total_batches - 2  # Bez INTRO i FINAL
    if content_batches <= requirements.h3.max_sections:
        return True
    
    # Wybierz co N-ty batch
    step = content_batches // requirements.h3.max_sections
    adjusted_batch = batch_number - 1  # Pomijamy INTRO
    return adjusted_batch % step == 0


def should_batch_have_list(
    batch_number: int,
    total_batches: int,
    requirements: StructureRequirements,
    lists_used_so_far: int = 0
) -> bool:
    """
    Określa czy dany batch powinien mieć listę.
    
    Args:
        batch_number: Numer bieżącego batcha
        total_batches: Całkowita liczba batchy
        requirements: Wymagania struktury
        lists_used_so_far: Ile list już użyto w poprzednich batchach
    """
    if not requirements.list.required and requirements.list.max_count == 0:
        return False
    
    if lists_used_so_far >= requirements.list.max_count:
        return False
    
    # Nie dodawaj list w INTRO
    if batch_number == 1:
        return False
    
    # Preferuj środkowe batche dla list
    middle_batch = total_batches // 2
    return abs(batch_number - middle_batch) <= 1


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🏗️ DYNAMIC STRUCTURE ANALYZER v1.0")
    print("=" * 60)
    
    # Test 1: S1 z danymi konkurencji (większość używa H3 i list)
    s1_with_competitors = {
        "search_intent": "informational",
        "main_keyword": "ubezwłasnowolnienie",
        "topic": "Ubezwłasnowolnienie - procedura i skutki",
        "competitor_h2_patterns": [
            {"has_h3": True, "h3_count": 2, "has_list": True, "list_count": 1},
            {"has_h3": True, "h3_count": 1, "has_list": True, "list_count": 2},
            {"has_h3": True, "h3_count": 2, "has_list": False, "list_count": 0},
            {"has_h3": False, "h3_count": 0, "has_list": True, "list_count": 1},
            {"has_h3": True, "h3_count": 1, "has_list": True, "list_count": 1},
        ]
    }
    
    print("\n📊 Test 1: S1 z danymi konkurencji (5 konkurentów)")
    req1 = analyze_structure_requirements(s1_with_competitors, target_length=2500)
    print(f"   Content type: {req1.content_type}")
    print(f"   H3 required: {req1.h3.required} (confidence: {req1.h3.confidence:.0%})")
    print(f"   H3 max sections: {req1.h3.max_sections}")
    print(f"   H3 rationale: {req1.h3.rationale}")
    print(f"   List required: {req1.list.required} (confidence: {req1.list.confidence:.0%})")
    print(f"   List max count: {req1.list.max_count}")
    print(f"   List rationale: {req1.list.rationale}")
    
    # Test 2: S1 bez danych konkurencji (fallback)
    s1_no_competitors = {
        "search_intent": "informational",
        "main_keyword": "jak napisać CV",
        "topic": "Jak napisać CV - poradnik krok po kroku",
        "competitor_h2_patterns": []
    }
    
    print("\n📊 Test 2: S1 bez konkurencji (fallback na content type)")
    req2 = analyze_structure_requirements(s1_no_competitors, target_length=2000)
    print(f"   Content type: {req2.content_type}")
    print(f"   H3 required: {req2.h3.required}")
    print(f"   H3 rationale: {req2.h3.rationale}")
    print(f"   List required: {req2.list.required}")
    
    # Test 3: Długi artykuł (skalowanie limitów)
    print("\n📊 Test 3: Długi artykuł (5000 słów) - skalowanie limitów")
    req3 = analyze_structure_requirements(s1_with_competitors, target_length=5000)
    print(f"   H3 max sections: {req3.h3.max_sections} (vs {req1.h3.max_sections} dla 2500 słów)")
    print(f"   List max count: {req3.list.max_count} (vs {req1.list.max_count} dla 2500 słów)")
    
    # Test 4: Force override
    print("\n📊 Test 4: Force override (wymuś brak H3)")
    req4 = analyze_structure_requirements(s1_with_competitors, target_length=2500, force_h3=False)
    print(f"   H3 required: {req4.h3.required}")
    print(f"   H3 source: {req4.h3.source}")
    
    # Test 5: Instrukcje dla pre_batch_info
    print("\n📊 Test 5: Instrukcje dla pre_batch_info")
    instructions = req1.get_structure_instructions()
    for key, value in instructions.items():
        print(f"   {key}: {value}")
