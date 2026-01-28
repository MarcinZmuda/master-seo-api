"""
===============================================================================
🧠 ADVANCED SEMANTIC FEATURES v1.1 - Rozszerzenie dla BRAJEN SEO Writer
===============================================================================
Nowe mechanizmy zgodne z Google 2024+ dla lepszego rankingu:

1. ENTITY DENSITY TRANSFORMER
   - Automatyczna transformacja ogólników → konkretne encje
   - Cel: 3+ encje na akapit, Knowledge Graph alignment
   
2. TOPIC COMPLETENESS ANALYZER
   - Porównanie pokrycia tematycznego vs TOP 10 konkurencji
   - Score kompletności: 0-100%

3. ENTITY GAP DETECTOR  
   - Wykrywa brakujące "twarde" encje (osoby, organizacje, akty prawne)
   - Priorytetyzuje encje z wysokim "authority weight"

4. SOURCE EFFORT SCORER v2.0
   - Mierzy sygnały wysiłku badawczego
   - Premiuje: orzecznictwo, badania naukowe, oficjalne dane

Autor: BRAJEN Team
Data: 2025-01
v1.1: Naprawiono sygnaturę detect_entity_gap (dodano alias 'text')
===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum


# ================================================================
# 📊 KONFIGURACJA
# ================================================================

class EntityPriority(Enum):
    """Priorytet encji dla SEO."""
    CRITICAL = 4      # Osoby ekspertów, kluczowe organizacje
    HIGH = 3          # Akty prawne, badania
    MEDIUM = 2        # Miejsca, daty
    LOW = 1           # Ogólne terminy


@dataclass
class AdvancedSemanticConfig:
    """Konfiguracja zaawansowanych funkcji semantycznych."""
    
    # Entity Density
    MIN_ENTITIES_PER_100_WORDS = 3.0      # Cel: 3 encje na 100 słów
    OPTIMAL_ENTITIES_PER_100_WORDS = 4.5  # Optymalnie: 4.5
    MAX_ENTITIES_PER_100_WORDS = 7.0      # Maks: unikamy stuffingu
    
    # Entity types weights (dla scoring)
    ENTITY_TYPE_WEIGHTS = {
        "PERSON": 1.5,         # Eksperci, autorzy = najwyższy weight
        "ORGANIZATION": 1.3,   # Instytucje, firmy
        "LEGAL_ACT": 1.4,      # Ustawy, rozporządzenia
        "PUBLICATION": 1.2,    # Badania, raporty
        "LOCATION": 0.8,       # Miejsca
        "DATE": 0.6,           # Daty
        "PRODUCT": 1.0,        # Produkty, narzędzia
        "CONCEPT": 0.7,        # Pojęcia ogólne
    }
    
    # Topic Completeness thresholds
    TOPIC_COMPLETENESS_MIN = 0.70        # Minimum 70% pokrycia
    TOPIC_COMPLETENESS_OPTIMAL = 0.85    # Optymalnie 85%
    
    # Entity Gap
    ENTITY_GAP_MAX_SUGGESTIONS = 10      # Max sugestii brakujących encji
    HARD_ENTITY_MIN_RATIO = 0.20         # Min 20% "twardych" encji
    
    # Source Effort
    SOURCE_EFFORT_MIN = 0.40             # Minimum score
    SOURCE_EFFORT_OPTIMAL = 0.70         # Optymalny score


# ================================================================
# 1️⃣ ENTITY DENSITY TRANSFORMER
# ================================================================

# Słownik transformacji: ogólnik → konkretne encje (uniwersalne, branżowo-agnostyczne)
UNIVERSAL_ENTITY_TRANSFORMATIONS = {
    # ===== ŹRÓDŁA / AUTORYTETY =====
    "według ekspertów": {
        "transform_to": "według {EXPERT_NAME}, {EXPERT_TITLE} z {INSTITUTION}",
        "entity_slots": ["PERSON", "TITLE", "ORGANIZATION"],
        "example": "według dr. Jana Kowalskiego, specjalisty SEO z Uniwersytetu Warszawskiego",
        "entities_gained": 3
    },
    "badania pokazują": {
        "transform_to": "badanie {STUDY_AUTHOR} ({YEAR}) opublikowane w {JOURNAL} wykazało",
        "entity_slots": ["PERSON", "DATE", "PUBLICATION"],
        "example": "badanie Smith et al. (2024) opublikowane w Journal of Marketing wykazało",
        "entities_gained": 3
    },
    "statystyki wskazują": {
        "transform_to": "dane {SOURCE_ORG} z {DATE} ({REPORT_NAME}) wskazują",
        "entity_slots": ["ORGANIZATION", "DATE", "PUBLICATION"],
        "example": "dane GUS z 2024 roku (Rocznik Statystyczny) wskazują",
        "entities_gained": 3
    },
    "eksperci twierdzą": {
        "transform_to": "{EXPERT_NAME} ({CREDENTIAL}) twierdzi w {PUBLICATION}",
        "entity_slots": ["PERSON", "CREDENTIAL", "PUBLICATION"],
        "example": "prof. Maria Nowak (Harvard Business Review) twierdzi",
        "entities_gained": 3
    },
    
    # ===== PRAWO / REGULACJE =====
    "zgodnie z prawem": {
        "transform_to": "zgodnie z {LEGAL_ACT} (Dz.U. {YEAR} poz. {NUMBER})",
        "entity_slots": ["LEGAL_ACT", "DATE", "IDENTIFIER"],
        "example": "zgodnie z Ustawą o ochronie danych (Dz.U. 2018 poz. 1000)",
        "entities_gained": 3
    },
    "przepisy wymagają": {
        "transform_to": "art. {NUMBER} {LEGAL_ACT} wymaga",
        "entity_slots": ["IDENTIFIER", "LEGAL_ACT"],
        "example": "art. 15 RODO wymaga",
        "entities_gained": 2
    },
    "regulacje nakazują": {
        "transform_to": "{REGULATION_NAME} ({ISSUING_BODY}) nakazuje",
        "entity_slots": ["LEGAL_ACT", "ORGANIZATION"],
        "example": "Rozporządzenie MF z 15.01.2024 (Ministerstwo Finansów) nakazuje",
        "entities_gained": 2
    },
    
    # ===== TECHNOLOGIA =====
    "algorytm ocenia": {
        "transform_to": "systemy {ALGORITHM_NAMES} oceniają na podstawie {CRITERIA}",
        "entity_slots": ["PRODUCT", "CONCEPT"],
        "example": "systemy SpamBrain i Helpful Content System oceniają na podstawie E-E-A-T",
        "entities_gained": 3
    },
    "wyszukiwarka analizuje": {
        "transform_to": "{SEARCH_ENGINE} poprzez {TECHNOLOGY} analizuje",
        "entity_slots": ["ORGANIZATION", "PRODUCT"],
        "example": "Google poprzez RankBrain i BERT analizuje",
        "entities_gained": 3
    },
    "sztuczna inteligencja": {
        "transform_to": "modele AI takie jak {MODEL_NAMES}",
        "entity_slots": ["PRODUCT"],
        "example": "modele AI takie jak GPT-4, Claude 3, Gemini Pro",
        "entities_gained": 3
    },
    
    # ===== INSTYTUCJE =====
    "odpowiedni urząd": {
        "transform_to": "{OFFICE_NAME} ({LOCATION})",
        "entity_slots": ["ORGANIZATION", "LOCATION"],
        "example": "Urząd Skarbowy w Warszawie-Mokotów",
        "entities_gained": 2
    },
    "właściwa instytucja": {
        "transform_to": "{INSTITUTION_NAME} – {INSTITUTION_ROLE}",
        "entity_slots": ["ORGANIZATION", "CONCEPT"],
        "example": "ZUS – Zakład Ubezpieczeń Społecznych odpowiedzialny za emerytury",
        "entities_gained": 2
    },
    "organizacja branżowa": {
        "transform_to": "{ORG_NAME}, {ORG_DESCRIPTION}",
        "entity_slots": ["ORGANIZATION", "CONCEPT"],
        "example": "PARP (Polska Agencja Rozwoju Przedsiębiorczości), wspierająca MŚP",
        "entities_gained": 2
    },
    
    # ===== CZASOWE =====
    "w ostatnich latach": {
        "transform_to": "od {YEAR} roku, po {EVENT}",
        "entity_slots": ["DATE", "CONCEPT"],
        "example": "od 2022 roku, po wprowadzeniu Google Helpful Content Update",
        "entities_gained": 2
    },
    "obecnie": {
        "transform_to": "w {MONTH} {YEAR}, według {SOURCE}",
        "entity_slots": ["DATE", "ORGANIZATION"],
        "example": "w styczniu 2025, według raportu Deloitte",
        "entities_gained": 2
    },
    "niedawno": {
        "transform_to": "w {SPECIFIC_DATE}, {EVENT_DESCRIPTION}",
        "entity_slots": ["DATE", "CONCEPT"],
        "example": "12 grudnia 2024, podczas konferencji Google I/O",
        "entities_gained": 2
    },
    
    # ===== MIEJSCA =====
    "w kraju": {
        "transform_to": "w Polsce, szczególnie w {REGION}",
        "entity_slots": ["LOCATION", "LOCATION"],
        "example": "w Polsce, szczególnie w województwie mazowieckim",
        "entities_gained": 2
    },
    "na świecie": {
        "transform_to": "globalnie, z największym udziałem {COUNTRIES}",
        "entity_slots": ["LOCATION"],
        "example": "globalnie, z największym udziałem USA, Chin i Niemiec",
        "entities_gained": 3
    },
    
    # ===== OGÓLNE UNIKANIE =====
    "wiele czynników": {
        "transform_to": "czynniki takie jak {FACTOR_1}, {FACTOR_2} i {FACTOR_3}",
        "entity_slots": ["CONCEPT", "CONCEPT", "CONCEPT"],
        "example": "czynniki takie jak Core Web Vitals, E-E-A-T i topical authority",
        "entities_gained": 3
    },
    "różne metody": {
        "transform_to": "metody {METHOD_1}, {METHOD_2} oraz {METHOD_3}",
        "entity_slots": ["CONCEPT", "CONCEPT", "CONCEPT"],
        "example": "metody on-page SEO, link building oraz content marketing",
        "entities_gained": 3
    },
    "odpowiednie narzędzia": {
        "transform_to": "narzędzia takie jak {TOOL_1}, {TOOL_2} i {TOOL_3}",
        "entity_slots": ["PRODUCT", "PRODUCT", "PRODUCT"],
        "example": "narzędzia takie jak Ahrefs, SEMrush i Screaming Frog",
        "entities_gained": 3
    },
}

# Frazy "puste" - zero wartości informacyjnej
ZERO_VALUE_PHRASES = [
    "warto wiedzieć że",
    "należy pamiętać że",
    "powszechnie wiadomo",
    "jak wszyscy wiedzą",
    "nie ulega wątpliwości",
    "oczywiste jest",
    "każdy wie że",
    "z doświadczenia wiemy",
    "w dzisiejszych czasach",
    "we współczesnym świecie",
    "na przestrzeni lat",
    "z biegiem czasu",
    "coraz częściej",
    "coraz więcej osób",
]


def analyze_entity_density_advanced(
    text: str,
    detected_entities: List[Dict] = None
) -> Dict[str, Any]:
    """
    Zaawansowana analiza gęstości encji z transformacjami.
    
    Args:
        text: Tekst do analizy
        detected_entities: Lista encji już wykrytych przez NER
        
    Returns:
        Dict z analizą i sugestiami transformacji
    """
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)
    
    if word_count < 50:
        return {
            "status": "TOO_SHORT",
            "word_count": word_count,
            "message": "Tekst zbyt krótki do analizy"
        }
    
    config = AdvancedSemanticConfig()
    
    # 1. Znajdź ogólniki do transformacji
    transformations_needed = []
    total_entities_gainable = 0
    
    for generic, transform_data in UNIVERSAL_ENTITY_TRANSFORMATIONS.items():
        pattern = r'\b' + re.escape(generic) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            # Kontekst: 50 znaków przed i po
            ctx_start = max(0, match.start() - 50)
            ctx_end = min(len(text), match.end() + 50)
            context = text[ctx_start:ctx_end]
            
            transformations_needed.append({
                "generic_phrase": generic,
                "position": match.start(),
                "context": f"...{context}...",
                "transform_template": transform_data["transform_to"],
                "example": transform_data["example"],
                "entities_gained": transform_data["entities_gained"],
                "entity_slots": transform_data["entity_slots"]
            })
            total_entities_gainable += transform_data["entities_gained"]
    
    # 2. Znajdź frazy o zerowej wartości
    zero_value_found = []
    for phrase in ZERO_VALUE_PHRASES:
        if phrase in text_lower:
            # Znajdź pozycję
            idx = text_lower.find(phrase)
            ctx_start = max(0, idx - 30)
            ctx_end = min(len(text), idx + len(phrase) + 30)
            context = text[ctx_start:ctx_end]
            
            zero_value_found.append({
                "phrase": phrase,
                "context": f"...{context}...",
                "action": "USUŃ lub zamień na konkretny fakt"
            })
    
    # 3. Oblicz obecną gęstość encji
    existing_entity_count = len(detected_entities) if detected_entities else 0
    current_density = (existing_entity_count / word_count) * 100 if word_count > 0 else 0
    
    # 4. Oblicz potencjalną gęstość po transformacjach
    potential_density = ((existing_entity_count + total_entities_gainable) / word_count) * 100
    
    # 5. Kategoryzuj encje po typach (jeśli dostępne)
    entity_type_distribution = defaultdict(int)
    weighted_entity_score = 0
    
    if detected_entities:
        for entity in detected_entities:
            ent_type = entity.get("type", "UNKNOWN")
            entity_type_distribution[ent_type] += 1
            weighted_entity_score += config.ENTITY_TYPE_WEIGHTS.get(ent_type, 0.5)
    
    # 6. Oblicz "hard entity ratio" - procent twardych encji
    hard_entity_types = {"PERSON", "ORGANIZATION", "LEGAL_ACT", "PUBLICATION"}
    hard_count = sum(
        entity_type_distribution.get(t, 0) for t in hard_entity_types
    )
    hard_ratio = hard_count / existing_entity_count if existing_entity_count > 0 else 0
    
    # 7. Określ status
    if current_density >= config.OPTIMAL_ENTITIES_PER_100_WORDS:
        density_status = "EXCELLENT"
    elif current_density >= config.MIN_ENTITIES_PER_100_WORDS:
        density_status = "GOOD"
    elif current_density >= config.MIN_ENTITIES_PER_100_WORDS * 0.7:
        density_status = "NEEDS_IMPROVEMENT"
    else:
        density_status = "POOR"
    
    # 8. Hard entity status
    if hard_ratio >= config.HARD_ENTITY_MIN_RATIO:
        hard_status = "OK"
    else:
        hard_status = "INSUFFICIENT"
    
    return {
        "status": density_status,
        "hard_entity_status": hard_status,
        "metrics": {
            "word_count": word_count,
            "existing_entities": existing_entity_count,
            "current_density_per_100": round(current_density, 2),
            "potential_density_per_100": round(potential_density, 2),
            "entities_gainable": total_entities_gainable,
            "hard_entity_ratio": round(hard_ratio, 2),
            "weighted_entity_score": round(weighted_entity_score, 2)
        },
        "entity_type_distribution": dict(entity_type_distribution),
        "transformations": {
            "count": len(transformations_needed),
            "items": transformations_needed[:10]  # Top 10
        },
        "zero_value_phrases": {
            "count": len(zero_value_found),
            "items": zero_value_found[:5]  # Top 5
        },
        "thresholds": {
            "min_density": config.MIN_ENTITIES_PER_100_WORDS,
            "optimal_density": config.OPTIMAL_ENTITIES_PER_100_WORDS,
            "min_hard_ratio": config.HARD_ENTITY_MIN_RATIO
        },
        "action_required": density_status in ["NEEDS_IMPROVEMENT", "POOR"] or hard_status == "INSUFFICIENT"
    }


# ================================================================
# 2️⃣ TOPIC COMPLETENESS ANALYZER
# ================================================================

@dataclass
class TopicCluster:
    """Klaster tematyczny z konkurencji."""
    topic_name: str
    h2_variants: List[str]
    coverage_count: int
    total_sources: int
    priority: str  # MUST, HIGH, MEDIUM, LOW
    sample_h2: str
    key_entities: List[str] = field(default_factory=list)
    
    @property
    def coverage_percent(self) -> float:
        return (self.coverage_count / self.total_sources * 100) if self.total_sources > 0 else 0


def analyze_topic_completeness(
    content: str = None,
    competitor_topics: List[Dict] = None,
    competitor_entities: List[Dict] = None,
    main_keyword: str = "",
    *,
    text: str = None  # ✅ ALIAS dla kompatybilności
) -> Dict[str, Any]:
    """
    Analizuje kompletność tematyczną vs konkurencja.
    
    Args:
        content: Treść artykułu do oceny (lub użyj 'text' jako alias)
        competitor_topics: Lista tematów z analyze_topical_coverage()
        competitor_entities: Lista encji z extract_entities()
        main_keyword: Główna fraza kluczowa
        text: Alias dla 'content' (dla kompatybilności wstecznej)
        
    Returns:
        Dict z oceną kompletności i brakami
    """
    # ✅ OBSŁUGA ALIASU
    if content is None and text is not None:
        content = text
    
    if content is None:
        return {
            "status": "NO_DATA",
            "message": "Brak tekstu do analizy"
        }
    
    if competitor_topics is None:
        competitor_topics = []
    
    if competitor_entities is None:
        competitor_entities = []
    
    content_lower = content.lower()
    config = AdvancedSemanticConfig()
    
    # 1. Analizuj pokrycie tematów MUST i HIGH
    must_topics = [t for t in competitor_topics if t.get("priority") in ["MUST", "HIGH"]]
    
    covered_topics = []
    missing_topics = []
    
    for topic in must_topics:
        subtopic = topic.get("subtopic", "").lower()
        sample_h2 = topic.get("sample_h2", "")
        
        # Sprawdź czy temat jest pokryty w treści
        # Szukamy słów kluczowych z nazwy tematu
        topic_words = subtopic.split()
        matches = sum(1 for word in topic_words if word in content_lower)
        coverage = matches / len(topic_words) if topic_words else 0
        
        topic_data = {
            "topic": subtopic,
            "sample_h2": sample_h2,
            "priority": topic.get("priority"),
            "coverage_in_competitors": topic.get("coverage_percent", 0),
            "found_in_content": coverage >= 0.5
        }
        
        if coverage >= 0.5:
            covered_topics.append(topic_data)
        else:
            missing_topics.append(topic_data)
    
    # 2. Oblicz Topic Completeness Score
    total_must_topics = len(must_topics)
    covered_count = len(covered_topics)
    
    if total_must_topics > 0:
        completeness_score = covered_count / total_must_topics
    else:
        completeness_score = 1.0  # Brak danych = OK
    
    # 3. Analizuj "Topic Gap" - tematy które ma konkurencja a my nie
    topic_gap = []
    for missing in missing_topics:
        if missing["priority"] == "MUST":
            gap_priority = "CRITICAL"
        elif missing["priority"] == "HIGH":
            gap_priority = "HIGH"
        else:
            gap_priority = "MEDIUM"
        
        topic_gap.append({
            "topic": missing["topic"],
            "sample_h2": missing["sample_h2"],
            "gap_priority": gap_priority,
            "competitors_coverage": f"{missing['coverage_in_competitors']}%",
            "recommendation": f"Dodaj sekcję H2 o: {missing['sample_h2']}"
        })
    
    # 4. Status
    if completeness_score >= config.TOPIC_COMPLETENESS_OPTIMAL:
        status = "EXCELLENT"
    elif completeness_score >= config.TOPIC_COMPLETENESS_MIN:
        status = "GOOD"
    elif completeness_score >= 0.5:
        status = "NEEDS_IMPROVEMENT"
    else:
        status = "POOR"
    
    return {
        "status": status,
        "completeness_score": round(completeness_score, 2),
        "completeness_percent": round(completeness_score * 100),
        "metrics": {
            "total_must_topics": total_must_topics,
            "covered_topics": covered_count,
            "missing_topics": len(missing_topics)
        },
        "covered_topics": covered_topics,
        "topic_gap": topic_gap[:config.ENTITY_GAP_MAX_SUGGESTIONS],
        "thresholds": {
            "min_completeness": config.TOPIC_COMPLETENESS_MIN,
            "optimal_completeness": config.TOPIC_COMPLETENESS_OPTIMAL
        },
        "action_required": status in ["NEEDS_IMPROVEMENT", "POOR"],
        "summary": _generate_completeness_summary(status, completeness_score, len(topic_gap))
    }


def _generate_completeness_summary(status: str, score: float, gap_count: int) -> str:
    """Generuje podsumowanie kompletności."""
    if status == "EXCELLENT":
        return f"Artykuł pokrywa {score*100:.0f}% kluczowych tematów konkurencji. Doskonała kompletność."
    elif status == "GOOD":
        return f"Artykuł pokrywa {score*100:.0f}% tematów. Dobra kompletność, {gap_count} mniejszych braków."
    elif status == "NEEDS_IMPROVEMENT":
        return f"Artykuł pokrywa tylko {score*100:.0f}% tematów. Brakuje {gap_count} kluczowych sekcji."
    else:
        return f"Niska kompletność ({score*100:.0f}%). Pilnie dodaj {gap_count} brakujących tematów."


# ================================================================
# 3️⃣ ENTITY GAP DETECTOR (Automatyczny - bazuje na S1)
# ================================================================

# Wagi dla typów encji - używane do scoringu, NIE do wykrywania
# Wykrywanie jest AUTOMATYCZNE z S1 + NER
ENTITY_TYPE_WEIGHTS = {
    "PERSON": 1.5,
    "PER": 1.5,
    "ORGANIZATION": 1.3,
    "ORG": 1.3,
    "LEGAL_ACT": 1.4,
    "PUBLICATION": 1.2,
    "STANDARD": 1.1,
    "PRODUCT": 1.0,
    "LOCATION": 0.8,
    "LOC": 0.8,
    "GPE": 0.8,
    "DATE": 0.6,
    "EVENT": 0.9,
    "METHOD": 0.9,
}

# Typy uznawane za "twarde" encje (wysokiej jakości dla SEO)
HARD_ENTITY_TYPE_NAMES = {"PERSON", "PER", "ORGANIZATION", "ORG", "LEGAL_ACT", "PUBLICATION", "STANDARD"}


def detect_entity_gap(
    content: str = None,
    competitor_entities: List[Dict] = None,
    detected_content_entities: List[Dict] = None,
    *,
    text: str = None  # ✅ ALIAS dla kompatybilności
) -> Dict[str, Any]:
    """
    Wykrywa brakujące encje vs konkurencja - AUTOMATYCZNIE z S1.
    
    NIE używa predefiniowanych słowników!
    Wszystkie encje pochodzą z S1 (analiza konkurencji).
    
    Args:
        content: Treść artykułu (lub użyj 'text' jako alias)
        competitor_entities: Encje z S1 entity_seo.entities (z konkurencji)
        detected_content_entities: Encje wykryte w naszej treści (opcjonalne)
        text: Alias dla 'content' (dla kompatybilności wstecznej)
        
    Returns:
        Dict z analizą braków i rekomendacjami
    """
    # ✅ OBSŁUGA ALIASU - przyjmij 'text' jeśli 'content' nie podano
    if content is None and text is not None:
        content = text
    
    if content is None:
        return {
            "status": "NO_DATA",
            "message": "Brak tekstu do analizy",
            "coverage_score": 0.5
        }
    
    content_lower = content.lower()
    config = AdvancedSemanticConfig()
    
    if not competitor_entities:
        return {
            "status": "NO_DATA",
            "message": "Brak danych o encjach konkurencji z S1",
            "coverage_score": 0.5
        }
    
    # 1. Zbierz nasze encje (te które już mamy w treści)
    our_entities = set()
    
    # Z NER detection (jeśli dostępne)
    if detected_content_entities:
        for e in detected_content_entities:
            if isinstance(e, dict):
                text_val = e.get("text", "").lower().strip()
            else:
                text_val = str(e).lower().strip()
            if text_val and len(text_val) > 2:
                our_entities.add(text_val)
    
    # Dodatkowo: sprawdź które encje z S1 występują w treści
    for entity in competitor_entities:
        if isinstance(entity, dict):
            text_val = entity.get("text", "").lower().strip()
        else:
            text_val = str(entity).lower().strip()
        if text_val and text_val in content_lower:
            our_entities.add(text_val)
    
    # 2. Analizuj encje z S1 - które mamy, których brakuje
    entity_gap = []
    found_entities = []
    
    found_by_type = defaultdict(int)
    missing_by_type = defaultdict(list)
    
    for entity in competitor_entities:
        # ✅ Defensywna obsługa różnych formatów
        if isinstance(entity, dict):
            text_val = entity.get("text", "")
            ent_type = entity.get("type", "UNKNOWN")
            importance = entity.get("importance", 0.5)
            sources_count = entity.get("sources_count", 1)
            context = entity.get("sample_context", entity.get("context", ""))
        elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
            text_val = str(entity[0])
            ent_type = str(entity[1]) if len(entity) > 1 else "UNKNOWN"
            importance = float(entity[2]) if len(entity) > 2 else 0.5
            sources_count = int(entity[3]) if len(entity) > 3 else 1
            context = ""
        else:
            text_val = str(entity)
            ent_type = "UNKNOWN"
            importance = 0.5
            sources_count = 1
            context = ""
        
        text_lower_val = text_val.lower().strip()
        
        # Czy encja jest w naszej treści?
        is_found = text_lower_val in our_entities or text_lower_val in content_lower
        
        if is_found:
            found_entities.append({
                "entity": text_val,
                "type": ent_type,
                "importance": importance
            })
            found_by_type[ent_type] += 1
        else:
            # Brakuje! Oceń priorytet
            if importance >= 0.7 and sources_count >= 4:
                priority = "CRITICAL"
            elif importance >= 0.5 and sources_count >= 2:
                priority = "HIGH"
            elif importance >= 0.3:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            # Waga dla tego typu encji
            weight = ENTITY_TYPE_WEIGHTS.get(ent_type, 0.8)
            
            gap_entry = {
                "entity": text_val,
                "type": ent_type,
                "priority": priority,
                "importance": importance,
                "sources_in_competitors": sources_count,
                "weight": weight,
                "context_hint": context[:150] if context else "",
                "recommendation": f"Dodaj wzmiankę o: {text_val}"
            }
            
            entity_gap.append(gap_entry)
            missing_by_type[ent_type].append(gap_entry)
    
    # 3. Sortuj gaps po priorytecie i wadze
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    entity_gap.sort(
        key=lambda x: (priority_order.get(x["priority"], 3), -x["weight"], -x["importance"]),
    )
    
    # 4. Oblicz coverage score
    # Liczymy tylko encje z importance >= 0.3 (istotne)
    important_competitor_entities = []
    for e in competitor_entities:
        if isinstance(e, dict):
            imp = e.get("importance", 0)
        else:
            imp = 0.5
        if imp >= 0.3:
            important_competitor_entities.append(e)
    
    total_important = len(important_competitor_entities)
    total_found = len(found_entities)
    
    if total_important > 0:
        coverage_score = total_found / total_important
        coverage_score = min(1.0, coverage_score)
    else:
        coverage_score = 1.0  # Brak danych = OK
    
    # 5. Oblicz hard entity coverage
    hard_found = sum(1 for e in found_entities if e["type"] in HARD_ENTITY_TYPE_NAMES)
    hard_missing = sum(1 for e in entity_gap if e["type"] in HARD_ENTITY_TYPE_NAMES and e["priority"] in ["CRITICAL", "HIGH"])
    
    # 6. Status
    if coverage_score >= 0.7:
        status = "GOOD"
    elif coverage_score >= 0.4:
        status = "MODERATE"
    else:
        status = "WEAK"
    
    return {
        "status": status,
        "coverage_score": round(coverage_score, 2),
        "metrics": {
            "total_competitor_entities": len(competitor_entities),
            "important_entities": total_important,
            "found_in_content": total_found,
            "missing": len(entity_gap),
            "hard_entities_found": hard_found,
            "hard_entities_missing": hard_missing,
            "by_type_found": dict(found_by_type),
            "by_type_missing": {k: len(v) for k, v in missing_by_type.items()}
        },
        "found_entities": found_entities[:10],
        "entity_gap": entity_gap[:config.ENTITY_GAP_MAX_SUGGESTIONS],
        "critical_gaps": [g for g in entity_gap if g["priority"] == "CRITICAL"][:5],
        "high_gaps": [g for g in entity_gap if g["priority"] == "HIGH"][:5],
        "action_required": status == "WEAK" or hard_missing > 3,
        "summary": _generate_entity_gap_summary(status, total_found, len(entity_gap), hard_missing)
    }


def _generate_entity_gap_summary(status: str, found: int, missing: int, hard_missing: int) -> str:
    """Generuje podsumowanie gap analysis."""
    if status == "GOOD":
        return f"Znaleziono {found} encji z konkurencji. Dobra reprezentacja."
    elif status == "MODERATE":
        return f"Znaleziono {found} encji, brakuje {missing}. Dodaj kluczowe encje z S1."
    else:
        return f"Tylko {found} encji z konkurencji. Brakuje {missing} (w tym {hard_missing} twardych). Pilnie uzupełnij!"


# ================================================================
# 4️⃣ SOURCE EFFORT SCORER v2.0
# ================================================================

# Wzorce sygnałów wysiłku badawczego - rozszerzone
SOURCE_EFFORT_SIGNALS = {
    # ===== ORZECZNICTWO (najwyższy weight) =====
    "legal_rulings": {
        "patterns": [
            r'wyrok\s+(?:SN|SA|SO|TK|NSA|WSA)\s+z\s+dnia',     # wyrok SN z dnia
            r'sygn\.\s*akt\s*[A-Z]{1,3}\s*\d+/\d+',            # sygn. akt II CSK 123/22
            r'uchwała\s+(?:SN|TK|NSA)',                         # uchwała SN
            r'orzeczenie\s+(?:TSUE|ETPCz)',                     # orzeczenie TSUE
            r'postanowienie\s+(?:SN|SA)',                       # postanowienie SN
        ],
        "weight": 2.0,
        "category": "LEGAL_RULINGS",
        "description": "Cytowanie orzeczeń sądowych"
    },
    
    # ===== AKTY PRAWNE =====
    "legal_acts": {
        "patterns": [
            r'art\.\s*\d+\s*(?:ust\.\s*\d+)?',                  # art. 15 ust. 1
            r'§\s*\d+\s*(?:ust\.\s*\d+)?',                      # § 45 ust. 2
            r'Dz\.?\s*U\.?\s*(?:z\s*)?\d{4}',                   # Dz.U. 2024
            r'poz\.\s*\d{1,4}',                                 # poz. 1234
            r'(?:ustawa|rozporządzenie)\s+z\s+dnia\s+\d+',      # ustawa z dnia 15
            r'dyrektywa\s+(?:\d+/\d+)?[A-Z]{2,}',               # dyrektywa RODO
        ],
        "weight": 1.5,
        "category": "LEGAL_ACTS",
        "description": "Odniesienia do aktów prawnych"
    },
    
    # ===== BADANIA NAUKOWE =====
    "scientific_research": {
        "patterns": [
            r'(?:badanie|badania)\s+(?:\w+\s+){0,3}(?:et\s+al\.?|i\s+(?:wsp|in)\.|współpracowników)',
            r'\(\d{4}\)\s*(?:wykazał[oaiy]?|pokazał[oaiy]?|stwierdził[oaiy]?)',  # (2024) wykazało
            r'opublikowane?\s+w\s+(?:czasopiśmie|journalu?)',
            r'peer[\s-]?review(?:ed)?',
            r'meta[\s-]?analiza',
            r'randomizowane\s+(?:badanie|próba)',
            r'n\s*=\s*\d{2,}',                                   # n = 150 (sample size)
            r'p\s*[<>=]\s*0[,\.]\d+',                            # p < 0.05
        ],
        "weight": 1.8,
        "category": "SCIENTIFIC_RESEARCH",
        "description": "Cytowanie badań naukowych"
    },
    
    # ===== OFICJALNE DANE =====
    "official_data": {
        "patterns": [
            r'(?:dane|statystyki|raport)\s+(?:GUS|Eurostat|OECD|WHO|NBP|KNF)',
            r'według\s+(?:GUS|Eurostatu?|OECD|WHO|NBP)',
            r'(?:rocznik|biuletyn)\s+statystyczny',
            r'(?:kwartalny|roczny)\s+raport',
            r'stan\s+na\s+(?:dzień\s+)?\d+[./]\d+[./]\d+',       # stan na 15.01.2025
            r'\d{1,2}[./]\d{1,2}[./]\d{4}\s*r?\.?',             # 15.01.2025 r.
        ],
        "weight": 1.4,
        "category": "OFFICIAL_DATA",
        "description": "Oficjalne dane i statystyki"
    },
    
    # ===== EKSPERCI / AUTORYTETY =====
    "expert_citations": {
        "patterns": [
            r'(?:dr|prof\.|mgr|inż\.)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+',
            r'według\s+(?:dr|prof\.)',
            r'(?:ekspert|specjalista)\s+[A-ZĄĆĘŁŃÓŚŹŻ]',
            r'(?:zdaniem|opinii)\s+(?:eksperta|specjalisty)',
        ],
        "weight": 1.3,
        "category": "EXPERT_CITATIONS",
        "description": "Cytowanie ekspertów"
    },
    
    # ===== METODOLOGIA =====
    "methodology": {
        "patterns": [
            r'metodologia\s+(?:badania|analizy)',
            r'próba\s+(?:badawcza|reprezentatywna)',
            r'(?:analiza|badanie)\s+(?:jakościowe|ilościowe)',
            r'(?:wywiad|ankieta)\s+(?:pogłębion|strukturaln)',
            r'(?:obserwacja|eksperyment)\s+(?:kontrolowan|naturaln)',
        ],
        "weight": 1.2,
        "category": "METHODOLOGY",
        "description": "Opis metodologii badawczej"
    },
    
    # ===== ŹRÓDŁA BRANŻOWE =====
    "industry_sources": {
        "patterns": [
            r'raport\s+(?:Deloitte|McKinsey|PwC|EY|KPMG|Gartner)',
            r'(?:analiza|badanie)\s+(?:Kantar|Nielsen|GfK)',
            r'(?:indeks|ranking)\s+(?:Doing\s+Business|Global)',
        ],
        "weight": 1.3,
        "category": "INDUSTRY_SOURCES",
        "description": "Raporty branżowe i analizy rynkowe"
    },
}

# Negatywne sygnały - obniżają score
NEGATIVE_EFFORT_SIGNALS = {
    "vague_sources": {
        "patterns": [
            r'(?:niektórzy|wielu)\s+(?:ekspert|badacz)',        # "niektórzy eksperci"
            r'(?:badania|statystyki)\s+(?:pokazują|wskazują)',  # bez konkretów
            r'(?:powszechnie|ogólnie)\s+(?:wiadomo|przyjęte)',
            r'(?:według|jak)\s+(?:wielu|niektórych)',
        ],
        "penalty": -0.3,
        "reason": "Niejasne odniesienia do źródeł"
    },
    "no_dates": {
        "patterns": [
            r'(?:ostatnio|niedawno|w\s+ostatnim\s+czasie)',
            r'(?:kiedyś|dawniej|wcześniej)',
        ],
        "penalty": -0.2,
        "reason": "Brak konkretnych dat"
    }
}


def calculate_source_effort_v2(text: str) -> Dict[str, Any]:
    """
    Zaawansowany Source Effort Scorer v2.0.
    
    Mierzy sygnały wysiłku badawczego w tekście.
    Premiuje orzecznictwo, badania naukowe, oficjalne dane.
    
    Args:
        text: Tekst do analizy
        
    Returns:
        Dict ze score i szczegółami
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
    if word_count < 100:
        return {
            "status": "TOO_SHORT",
            "score": 0,
            "message": "Tekst zbyt krótki do analizy source effort"
        }
    
    config = AdvancedSemanticConfig()
    
    # 1. Znajdź pozytywne sygnały
    positive_signals_found = defaultdict(list)
    total_weighted_score = 0
    
    for signal_name, signal_data in SOURCE_EFFORT_SIGNALS.items():
        patterns = signal_data["patterns"]
        weight = signal_data["weight"]
        category = signal_data["category"]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                # Kontekst
                ctx_start = max(0, match.start() - 40)
                ctx_end = min(len(text), match.end() + 40)
                context = text[ctx_start:ctx_end]
                
                positive_signals_found[category].append({
                    "match": match.group(),
                    "context": f"...{context}...",
                    "weight": weight
                })
                total_weighted_score += weight
    
    # 2. Znajdź negatywne sygnały
    negative_signals_found = []
    total_penalty = 0
    
    for signal_name, signal_data in NEGATIVE_EFFORT_SIGNALS.items():
        patterns = signal_data["patterns"]
        penalty = signal_data["penalty"]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                negative_signals_found.append({
                    "signal": signal_name,
                    "count": len(matches),
                    "penalty": penalty,
                    "reason": signal_data["reason"]
                })
                total_penalty += penalty * len(matches)
    
    # 3. Oblicz score
    # Normalizuj do długości tekstu (per 500 słów)
    normalized_score = (total_weighted_score / (word_count / 500)) if word_count > 0 else 0
    
    # Dodaj penalty
    final_score = max(0, min(1.0, (normalized_score / 10) + total_penalty))
    
    # 4. Zlicz unikalne kategorie
    categories_with_evidence = len([c for c in positive_signals_found if positive_signals_found[c]])
    
    # Bonus za diversity (różnorodność źródeł)
    diversity_bonus = min(0.15, categories_with_evidence * 0.03)
    final_score = min(1.0, final_score + diversity_bonus)
    
    # 5. Status
    if final_score >= config.SOURCE_EFFORT_OPTIMAL:
        status = "EXCELLENT"
    elif final_score >= config.SOURCE_EFFORT_MIN:
        status = "GOOD"
    elif final_score >= 0.25:
        status = "NEEDS_IMPROVEMENT"
    else:
        status = "POOR"
    
    # 6. Rekomendacje
    recommendations = []
    missing_categories = set(SOURCE_EFFORT_SIGNALS.keys()) - set(positive_signals_found.keys())
    
    priority_categories = ["legal_rulings", "scientific_research", "official_data"]
    for cat in priority_categories:
        if cat in missing_categories:
            signal_info = SOURCE_EFFORT_SIGNALS[cat]
            recommendations.append({
                "category": cat,
                "action": f"Dodaj: {signal_info['description']}",
                "impact": signal_info["weight"]
            })
    
    # Jeśli są negatywne sygnały
    for neg in negative_signals_found:
        recommendations.append({
            "category": "FIX",
            "action": f"Popraw: {neg['reason']} (znaleziono {neg['count']}x)",
            "impact": abs(neg["penalty"])
        })
    
    return {
        "status": status,
        "score": round(final_score, 2),
        "metrics": {
            "raw_weighted_score": round(total_weighted_score, 2),
            "normalized_score": round(normalized_score, 2),
            "penalty": round(total_penalty, 2),
            "diversity_bonus": round(diversity_bonus, 2),
            "categories_with_evidence": categories_with_evidence
        },
        "positive_signals": {
            k: {
                "count": len(v),
                "samples": [s["context"] for s in v[:2]]
            }
            for k, v in positive_signals_found.items()
        },
        "negative_signals": negative_signals_found,
        "recommendations": recommendations[:5],
        "thresholds": {
            "min_score": config.SOURCE_EFFORT_MIN,
            "optimal_score": config.SOURCE_EFFORT_OPTIMAL
        },
        "action_required": status in ["NEEDS_IMPROVEMENT", "POOR"]
    }


# ================================================================
# 🎯 GŁÓWNA FUNKCJA - UNIFIED ANALYSIS
# ================================================================

def perform_advanced_semantic_analysis(
    content: str,
    main_keyword: str = "",
    competitor_topics: List[Dict] = None,
    competitor_entities: List[Dict] = None,
    detected_content_entities: List[Dict] = None
) -> Dict[str, Any]:
    """
    Wykonuje pełną zaawansowaną analizę semantyczną.
    
    Łączy cztery analizy:
    1. Entity Density (transformacje ogólników)
    2. Topic Completeness (vs konkurencja)
    3. Entity Gap (brakujące twarde encje)
    4. Source Effort (wysiłek badawczy)
    
    Args:
        content: Treść artykułu
        main_keyword: Główna fraza kluczowa
        competitor_topics: Tematy z konkurencji (z S1)
        competitor_entities: Encje z konkurencji (z S1)
        detected_content_entities: Encje wykryte w naszej treści
        
    Returns:
        Dict z pełną analizą i priorytetyzowanymi rekomendacjami
    """
    # 1. Entity Density Analysis
    density_result = analyze_entity_density_advanced(content, detected_content_entities)
    
    # 2. Topic Completeness Analysis
    if competitor_topics:
        completeness_result = analyze_topic_completeness(
            content, competitor_topics, competitor_entities or [], main_keyword
        )
    else:
        completeness_result = {
            "status": "NO_DATA",
            "message": "Brak danych o tematach konkurencji"
        }
    
    # 3. Entity Gap Analysis
    if competitor_entities:
        gap_result = detect_entity_gap(content=content, competitor_entities=competitor_entities, detected_content_entities=detected_content_entities)
    else:
        gap_result = {
            "status": "NO_DATA",
            "message": "Brak danych o encjach konkurencji"
        }
    
    # 4. Source Effort Analysis
    effort_result = calculate_source_effort_v2(content)
    
    # 5. Oblicz Final Score
    scores = {
        "entity_density": _status_to_score(density_result.get("status")),
        "topic_completeness": completeness_result.get("completeness_score", 0.5) if competitor_topics else 0.5,
        "entity_gap": gap_result.get("coverage_score", 0.5) if competitor_entities else 0.5,
        "source_effort": effort_result.get("score", 0)
    }
    
    # Wagi
    weights = {
        "entity_density": 0.25,
        "topic_completeness": 0.25,
        "entity_gap": 0.25,
        "source_effort": 0.25
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    
    # 6. Final Status
    if final_score >= 0.75:
        final_status = "EXCELLENT"
    elif final_score >= 0.55:
        final_status = "GOOD"
    elif final_score >= 0.35:
        final_status = "NEEDS_IMPROVEMENT"
    else:
        final_status = "POOR"
    
    # 7. Zbierz priorytetyzowane rekomendacje
    all_recommendations = []
    
    # Z Entity Density
    if density_result.get("action_required"):
        for trans in density_result.get("transformations", {}).get("items", [])[:3]:
            all_recommendations.append({
                "type": "ENTITY_DENSITY",
                "priority": "HIGH",
                "action": f"Zamień '{trans['generic_phrase']}' → {trans['example']}",
                "entities_gained": trans.get("entities_gained", 2)
            })
    
    # Z Topic Completeness
    if completeness_result.get("action_required"):
        for gap in completeness_result.get("topic_gap", [])[:2]:
            all_recommendations.append({
                "type": "TOPIC_COMPLETENESS",
                "priority": "HIGH" if gap.get("gap_priority") == "CRITICAL" else "MEDIUM",
                "action": gap.get("recommendation", ""),
                "coverage_info": gap.get("competitors_coverage", "")
            })
    
    # Z Entity Gap
    if gap_result.get("action_required"):
        for entity in gap_result.get("critical_gaps", [])[:3]:
            all_recommendations.append({
                "type": "ENTITY_GAP",
                "priority": "HIGH",
                "action": f"Dodaj wzmiankę o: {entity['entity']} ({entity.get('type', 'UNKNOWN')})",
                "importance": entity.get("importance", 0.5)
            })
    
    # Z Source Effort
    if effort_result.get("action_required"):
        for rec in effort_result.get("recommendations", [])[:2]:
            all_recommendations.append({
                "type": "SOURCE_EFFORT",
                "priority": "MEDIUM",
                "action": rec.get("action", ""),
                "impact": rec.get("impact", 1.0)
            })
    
    # Sortuj po priorytecie
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "LOW"), 2))
    
    return {
        "status": final_status,
        "final_score": round(final_score, 2),
        "component_scores": {k: round(v, 2) for k, v in scores.items()},
        "analyses": {
            "entity_density": density_result,
            "topic_completeness": completeness_result,
            "entity_gap": gap_result,
            "source_effort": effort_result
        },
        "prioritized_recommendations": all_recommendations[:10],
        "quick_wins": [r for r in all_recommendations if r.get("priority") == "HIGH"][:5],
        "action_required": final_status in ["NEEDS_IMPROVEMENT", "POOR"],
        "summary": _generate_final_summary(final_status, scores)
    }


def _status_to_score(status: str) -> float:
    """Konwertuje status na score numeryczny."""
    mapping = {
        "EXCELLENT": 1.0,
        "GOOD": 0.75,
        "MODERATE": 0.5,
        "NEEDS_IMPROVEMENT": 0.35,
        "POOR": 0.15,
        "WEAK": 0.25,
        "TOO_SHORT": 0.0,
        "NO_DATA": 0.5
    }
    return mapping.get(status, 0.5)


def _generate_final_summary(status: str, scores: Dict[str, float]) -> str:
    """Generuje końcowe podsumowanie."""
    parts = []
    
    if scores["entity_density"] < 0.5:
        parts.append("niska gęstość encji")
    if scores["topic_completeness"] < 0.5:
        parts.append("niepełne pokrycie tematyczne")
    if scores["entity_gap"] < 0.5:
        parts.append("brak twardych encji")
    if scores["source_effort"] < 0.4:
        parts.append("niski wysiłek badawczy")
    
    if not parts:
        return "Tekst wykazuje wysoką jakość semantyczną. Wszystkie metryki na dobrym poziomie."
    
    if status in ["EXCELLENT", "GOOD"]:
        return "Tekst jest dobry. Drobne poprawki: " + ", ".join(parts)
    else:
        return "WYMAGANE POPRAWKI: " + ", ".join(parts)


# ================================================================
# 🔧 HELPER: Generowanie instrukcji dla GPT
# ================================================================

def generate_advanced_prompt_instructions(analysis: Dict) -> str:
    """
    Generuje sekcję instrukcji dla Custom GPT na podstawie analizy.
    
    Używane w BRAJEN prompt engineering.
    """
    if analysis.get("status") in ["EXCELLENT", "GOOD"]:
        return ""  # Nie potrzeba dodatkowych instrukcji
    
    lines = ["\n⚡ SEMANTIC ENHANCEMENT REQUIRED:\n"]
    
    quick_wins = analysis.get("quick_wins", [])
    
    for i, rec in enumerate(quick_wins[:5], 1):
        rec_type = rec.get("type", "")
        action = rec.get("action", "")
        
        icon = {
            "ENTITY_DENSITY": "🎯",
            "TOPIC_COMPLETENESS": "📋",
            "ENTITY_GAP": "🔍",
            "SOURCE_EFFORT": "📚"
        }.get(rec_type, "•")
        
        lines.append(f"{i}. {icon} {action}")
    
    # Dodaj konkretne przykłady transformacji
    density = analysis.get("analyses", {}).get("entity_density", {})
    transforms = density.get("transformations", {}).get("items", [])
    
    if transforms:
        lines.append("\n📝 PRZYKŁADY TRANSFORMACJI:")
        for trans in transforms[:3]:
            lines.append(f"   ✗ \"{trans['generic_phrase']}\"")
            lines.append(f"   ✓ \"{trans['example']}\"")
    
    lines.append("")  # Empty line at end
    
    return "\n".join(lines)
