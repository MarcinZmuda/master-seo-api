"""
===============================================================================
🎯 ENHANCED PRE-BATCH INSTRUCTIONS v40.1
===============================================================================
Moduł generujący KONKRETNE instrukcje dla GPT zamiast surowych danych.

ROZWIĄZUJE PROBLEMY:
1. Encje/Triplety - zamiast listy → konkretne "jak zdefiniować"
2. Keywords - tracking w tle, nie blokowanie per-batch
3. Humanizacja - konkretne instrukcje stylu
4. Kontynuacja - pełny kontekst poprzedniego batcha

ZMIANY v40.1:
- USUNIĘTO słabe SHORT_INSERTS_LIBRARY (9 generycznych fraz)
- USUNIĘTO słabe SYNONYM_MAP (4 słowa)
- DODANO integrację z dynamic_humanization.py (tematyczne biblioteki)

ZMIANY v40.1:
- Synonimy z kontekstem domeny (przekazuje domenę do synonym_service)
- Integracja z plWordNet API + Firestore cache

Autor: BRAJEN SEO Master API v40.1
===============================================================================
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# 🆕 v41.0: IMPORTS Z MODUŁÓW OPTYMALIZACYJNYCH
from paragraph_cv_analyzer_v41 import (
    get_paragraph_cv_for_prebatch,
    calculate_paragraph_cv
)
from triplet_priority_v41 import get_triplet_instructions_for_prebatch


# ============================================================================
# 🆕 v40.2: IMPORT CONCEPT MAP EXTRACTOR (Semantic Entity SEO)
# ============================================================================

try:
    from concept_map_extractor import (
        extract_concept_map,
        get_fallback_concept_map,
        validate_concept_map
    )
    CONCEPT_MAP_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ concept_map_extractor loaded")
except ImportError as e:
    CONCEPT_MAP_AVAILABLE = False
    print(f"[ENHANCED_PRE_BATCH] ⚠️ concept_map_extractor not available: {e}")
    
    def extract_concept_map(main_kw, competitor_texts, **kwargs):
        return {"status": "UNAVAILABLE", "concept_map": {}}
    
    def get_fallback_concept_map(main_kw):
        return {"main_entity": {"name": main_kw, "type": "Thing"}}


# ============================================================================
# 🆕 v40.2: IMPORT STYLE ANALYZER (Persona Fingerprint)
# ============================================================================

try:
    from style_analyzer import (
        StyleAnalyzer,
        StyleFingerprint,
        FormalityLevel,
        PersonalPronouns
    )
    STYLE_ANALYZER_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ style_analyzer loaded")
except ImportError as e:
    STYLE_ANALYZER_AVAILABLE = False
    print(f"[ENHANCED_PRE_BATCH] ⚠️ style_analyzer not available: {e}")


# ============================================================================
# 🆕 v40.1: IMPORT DYNAMIC HUMANIZATION (zastępuje słabe biblioteki)
# ============================================================================

try:
    from dynamic_humanization import (
        get_dynamic_short_sentences,
        get_synonym_instructions,
        get_burstiness_instructions,
        get_humanization_instructions,
        analyze_burstiness,
        detect_topic_domain,
        CONTEXTUAL_SYNONYMS,
        SYNONYM_SERVICE_AVAILABLE  # 🆕 v40.1
    )
    DYNAMIC_HUMANIZATION_AVAILABLE = True
    synonym_source = "plWordNet + cache" if SYNONYM_SERVICE_AVAILABLE else "local only"
    print(f"[ENHANCED_PRE_BATCH] ✅ dynamic_humanization v40.1 loaded (synonyms: {synonym_source})")
except ImportError as e:
    DYNAMIC_HUMANIZATION_AVAILABLE = False
    SYNONYM_SERVICE_AVAILABLE = False
    print(f"[ENHANCED_PRE_BATCH] ⚠️ dynamic_humanization not available: {e}")
    
    # Fallback - minimalne funkcje
    CONTEXTUAL_SYNONYMS = {
        "można": ["da się", "istnieje możliwość"],
        "należy": ["trzeba", "wymaga się"],
        "ważny": ["istotny", "kluczowy", "zasadniczy"],
    }
    
    def get_dynamic_short_sentences(main_kw, h2s=None, count=8, include_q=True):
        return {
            "domain": "universal",
            "sentences": ["To ważne.", "Co dalej?", "Warto wiedzieć.", "Ale uwaga."],
            "instruction": "Wstaw 2-4 krótkie zdania (3-8 słów)"
        }
    
    def get_synonym_instructions(overused=None):
        return {
            "instruction": "Unikaj powtórzeń - używaj synonimów",
            "synonyms": CONTEXTUAL_SYNONYMS
        }
    
    def get_burstiness_instructions(prev_text=None):
        return {
            "critical": True,
            "target_cv": ">0.40",
            "example_sequence": "5, 18, 8, 25, 12 słów"
        }
    
    def detect_topic_domain(main_kw, h2s=None):
        return "universal"


# ============================================================================
# IMPORT POLISH LANGUAGE QUALITY (słowa łączące)
# ============================================================================

try:
    from polish_language_quality import TRANSITION_WORDS_CATEGORIZED
    POLISH_QUALITY_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ polish_language_quality loaded")
except ImportError:
    POLISH_QUALITY_AVAILABLE = False
    TRANSITION_WORDS_CATEGORIZED = {
        "kontrast": ["jednak", "natomiast", "ale", "mimo to", "z drugiej strony"],
        "przyczyna": ["ponieważ", "bowiem", "dlatego że", "ze względu na"],
        "skutek": ["dlatego", "zatem", "więc", "w efekcie", "w rezultacie"],
        "sekwencja": ["najpierw", "następnie", "potem", "na koniec"]
    }
    print("[ENHANCED_PRE_BATCH] ⚠️ polish_language_quality not available, using fallback")


# ============================================================================
# 🆕 v40.1: ADVANCED SEMANTIC FEATURES (entity density, topic completeness)
# ============================================================================

try:
    from advanced_semantic_features import (
        perform_advanced_semantic_analysis,
        generate_advanced_prompt_instructions,
        detect_entity_gap
    )
    ADVANCED_SEMANTIC_ENABLED = True
    print("[ENHANCED_PRE_BATCH] ✅ advanced_semantic_features loaded")
except ImportError as e:
    ADVANCED_SEMANTIC_ENABLED = False
    print(f"[ENHANCED_PRE_BATCH] ⚠️ advanced_semantic_features not available: {e}")


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class EnhancedPreBatchConfig:
    """Konfiguracja dla enhanced pre_batch_info."""
    
    # Ile encji max pokazać w instrukcjach
    MAX_ENTITIES_TO_DEFINE: int = 5
    MAX_RELATIONS_TO_ESTABLISH: int = 4
    MAX_NGRAMS_PER_BATCH: int = 6
    
    # Semantic context
    MIN_CONTEXT_TERMS: int = 3
    MAX_CONTEXT_TERMS: int = 6
    
    # Style
    TARGET_SENTENCE_CV: float = 0.40  # Współczynnik zmienności zdań
    MIN_SHORT_SENTENCES_PCT: int = 20
    MAX_AI_PATTERN_SENTENCES: int = 30  # % zdań w przedziale 15-22 słów
    
    # Kontynuacja
    LAST_PARAGRAPH_WORDS: int = 150  # Ile słów z ostatniego akapitu


CONFIG = EnhancedPreBatchConfig()


# ============================================================================
# 1. ENTITIES TO DEFINE - konkretne instrukcje jak zdefiniować
# ============================================================================

# Wzorce definicji dla różnych typów encji
DEFINITION_TEMPLATES = {
    "legal_concept": [
        "Wyjaśnij że {entity} to instytucja prawna polegająca na...",
        "Zdefiniuj {entity} jako procedurę/mechanizm służący do...",
        "Napisz że {entity} w rozumieniu prawa oznacza..."
    ],
    "person_role": [
        "Przedstaw {entity} jako osobę/organ odpowiedzialną za...",
        "Wyjaśnij rolę {entity} w kontekście..."
    ],
    "process": [
        "Opisz {entity} jako proces składający się z etapów...",
        "Wyjaśnij że {entity} przebiega w następujący sposób..."
    ],
    "document": [
        "Wyjaśnij że {entity} to dokument zawierający...",
        "Zdefiniuj {entity} jako pismo/wniosek służący do..."
    ],
    "institution": [
        "Przedstaw {entity} jako organ/instytucję właściwą do...",
        "Wyjaśnij kompetencje {entity} w zakresie..."
    ],
    "default": [
        "Zdefiniuj {entity} wyjaśniając czym jest i do czego służy",
        "Wprowadź pojęcie {entity} w kontekście tematu"
    ]
}

# Słowa kluczowe do klasyfikacji typu encji
ENTITY_TYPE_KEYWORDS = {
    "legal_concept": ["ubezwłasnowolnienie", "prawo", "przepis", "ustawa", "kodeks", 
                      "zdolność", "legitymacja", "postępowanie"],
    "person_role": ["sędzia", "kurator", "opiekun", "biegły", "prokurator", 
                    "adwokat", "radca", "przedstawiciel"],
    "process": ["procedura", "postępowanie", "proces", "tryb", "etap"],
    "document": ["wniosek", "pismo", "pozew", "apelacja", "orzeczenie", "wyrok"],
    "institution": ["sąd", "organ", "urząd", "ministerstwo", "instytut"]
}


def classify_entity_type(entity: str) -> str:
    """Klasyfikuje typ encji na podstawie słów kluczowych."""
    entity_lower = entity.lower()
    
    for entity_type, keywords in ENTITY_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in entity_lower:
                return entity_type
    
    return "default"


def generate_definition_instruction(entity: str, context: str = "", h2: str = "") -> Dict[str, Any]:
    """
    Generuje konkretną instrukcję jak zdefiniować encję.
    """
    entity_type = classify_entity_type(entity)
    templates = DEFINITION_TEMPLATES.get(entity_type, DEFINITION_TEMPLATES["default"])
    
    template = templates[0]
    how = template.format(entity=entity)
    
    if context:
        how += f" W kontekście: {context}"
    
    return {
        "entity": entity,
        "action": "DEFINE",
        "type": entity_type,
        "how": how,
        "example_pattern": f'"{entity} to/jest/oznacza..."',
        "h2_context": h2
    }


def get_entities_to_define(
    s1_data: Dict,
    current_batch_num: int,
    entity_state: Dict,
    current_h2: str,
    total_batches: int
) -> List[Dict]:
    """
    Zwraca listę encji do zdefiniowania w tym batchu z konkretnymi instrukcjami.
    """
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    already_defined = set(entity_state.get("defined", []))
    
    result = []
    
    # 1. Najpierw encje MUST z topical_coverage
    for topic in topical_coverage:
        if topic.get("priority") == "MUST":
            entity = topic.get("topic", "")
            if entity and entity.lower() not in {e.lower() for e in already_defined}:
                instruction = generate_definition_instruction(
                    entity=entity,
                    context=topic.get("context", ""),
                    h2=current_h2
                )
                instruction["priority"] = "MUST"
                result.append(instruction)
    
    # 2. Encje HIGH importance
    for ent in entities:
        if ent.get("importance", 0) >= 0.7:
            entity = ent.get("text", ent.get("entity", ""))
            if entity and entity.lower() not in {e.lower() for e in already_defined}:
                if entity.lower() not in {r["entity"].lower() for r in result}:
                    instruction = generate_definition_instruction(
                        entity=entity,
                        context=ent.get("context", ""),
                        h2=current_h2
                    )
                    instruction["priority"] = "SHOULD"
                    result.append(instruction)
    
    # 3. Rozłóż równomiernie na batche
    entities_per_batch = max(2, len(result) // total_batches)
    start_idx = (current_batch_num - 1) * entities_per_batch
    end_idx = min(start_idx + entities_per_batch + 1, len(result))
    
    if current_batch_num == 1:
        result.sort(key=lambda x: 0 if x.get("priority") == "MUST" else 1)
        return result[:CONFIG.MAX_ENTITIES_TO_DEFINE]
    
    return result[start_idx:end_idx][:CONFIG.MAX_ENTITIES_TO_DEFINE]


# ============================================================================
# 2. RELATIONS TO ESTABLISH - relacje z wzorcami zdań
# ============================================================================

RELATION_TEMPLATES = {
    "orzeka": [
        "{subject} orzeka o {object}",
        "{subject} wydaje orzeczenie w sprawie {object}",
        "to {subject} rozstrzyga o {object}"
    ],
    "prowadzi_do": [
        "{subject} może prowadzić do {object}",
        "{subject} skutkuje {object}",
        "konsekwencją {subject} jest {object}"
    ],
    "wymaga": [
        "{subject} wymaga {object}",
        "do {subject} niezbędne jest {object}",
        "{subject} nie może nastąpić bez {object}"
    ],
    "reprezentuje": [
        "{subject} reprezentuje interesy {object}",
        "{subject} działa w imieniu {object}"
    ],
    "chroni": [
        "{subject} służy ochronie {object}",
        "celem {subject} jest zabezpieczenie {object}"
    ],
    "default": [
        "{subject} jest powiązane z {object}",
        "{subject} ma związek z {object}"
    ]
}


def generate_relation_instruction(
    from_entity: str,
    relation: str,
    to_entity: str
) -> Dict[str, Any]:
    """Generuje instrukcję ustanowienia relacji między encjami."""
    relation_lower = relation.lower().replace(" ", "_")
    templates = RELATION_TEMPLATES.get(relation_lower, RELATION_TEMPLATES["default"])
    
    example_sentences = [
        t.format(subject=from_entity, object=to_entity)
        for t in templates[:2]
    ]
    
    return {
        "from": from_entity,
        "relation": relation,
        "to": to_entity,
        "example_sentences": example_sentences,
        "instruction": f"Ustanów relację: {from_entity} → {relation} → {to_entity}",
        "how": f"Napisz zdanie łączące '{from_entity}' z '{to_entity}' przez '{relation}'"
    }


def get_relations_to_establish(
    s1_data: Dict,
    current_batch_num: int,
    entity_state: Dict,
    total_batches: int
) -> List[Dict]:
    """Zwraca relacje do ustanowienia w tym batchu."""
    entity_seo = s1_data.get("entity_seo", {})
    relationships = entity_seo.get("entity_relationships", [])
    
    established = set(entity_state.get("relations_established", []))
    
    result = []
    for rel in relationships:
        from_ent = rel.get("from", rel.get("subject", ""))
        to_ent = rel.get("to", rel.get("object", ""))
        relation = rel.get("relation", rel.get("predicate", ""))
        
        if not from_ent or not to_ent or not relation:
            continue
        
        rel_key = f"{from_ent}|{relation}|{to_ent}".lower()
        if rel_key not in established:
            instruction = generate_relation_instruction(from_ent, relation, to_ent)
            instruction["priority"] = rel.get("priority", "SHOULD")
            result.append(instruction)
    
    rels_per_batch = max(1, len(result) // total_batches)
    start_idx = (current_batch_num - 1) * rels_per_batch
    end_idx = min(start_idx + rels_per_batch + 1, len(result))
    
    return result[start_idx:end_idx][:CONFIG.MAX_RELATIONS_TO_ESTABLISH]


# ============================================================================
# 3. SEMANTIC CONTEXT - terminy które MUSZĄ być użyte
# ============================================================================

def get_semantic_context(
    s1_data: Dict,
    current_batch_num: int,
    current_h2: str,
    keywords_state: Dict
) -> Dict[str, Any]:
    """Generuje semantic context z terminami do użycia."""
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.4]
    
    semantic_keyphrases = s1_data.get("semantic_keyphrases", [])
    lsi_keywords = [kp.get("phrase", "") for kp in semantic_keyphrases if kp.get("score", 0) > 0.6]
    
    serp = s1_data.get("serp_analysis", {})
    related = serp.get("related_searches", [])[:5]
    
    all_terms = top_ngrams + lsi_keywords
    terms_per_batch = max(CONFIG.MIN_CONTEXT_TERMS, len(all_terms) // 8)
    
    start_idx = (current_batch_num - 1) * terms_per_batch
    end_idx = min(start_idx + terms_per_batch + 2, len(all_terms))
    
    batch_terms = all_terms[start_idx:end_idx][:CONFIG.MAX_CONTEXT_TERMS]
    
    keyword_set = {meta.get("keyword", "").lower() for meta in keywords_state.values()}
    batch_terms = [t for t in batch_terms if t.lower() not in keyword_set]
    
    return {
        "context_terms": batch_terms,
        "instruction": f"Użyj NATURALNIE w tekście (nie stuffing!): {', '.join(batch_terms[:4])}",
        "supporting_phrases": lsi_keywords[:3],
        "related_topics": related[:3],
        "semantic_density_target": "min 2 terminy na 100 słów"
    }


# ============================================================================
# 4. STYLE INSTRUCTIONS - humanizacja tekstu (v40.1 - DYNAMIC)
# ============================================================================

# Frazy typowe dla AI do unikania
AI_PATTERNS_TO_AVOID = [
    "warto podkreślić",
    "należy pamiętać",
    "w kontekście",
    "istotne jest",
    "kluczowym aspektem",
    "nie można pominąć",
    "szczególnie ważne",
    "fundamentalne znaczenie",
    "z perspektywy",
    "w odniesieniu do",
    "mając na uwadze",
    "biorąc pod uwagę",
    "co więcej",
    "ponadto",
    "dodatkowo",
    "warto zaznaczyć",
    "należy podkreślić",
    "trzeba wspomnieć",
    "nie bez znaczenia",
    "warto zauważyć"
]

# Naturalne alternatywy
NATURAL_ALTERNATIVES = {
    "warto podkreślić": ["", "Zwróć uwagę:", "Ważne:"],
    "należy pamiętać": ["Pamiętaj,", "Nie zapomnij,", ""],
    "w kontekście": ["przy", "jeśli chodzi o", "w sprawie"],
    "co więcej": ["Poza tym", "Również", "A co ważne"],
    "ponadto": ["Oprócz tego", "Też", "Również"],
    "warto zauważyć": ["", "Ważne:", "Zwróć uwagę:"],
    "nie bez znaczenia": ["Ważne:", "Istotne:"],
}


def get_style_instructions(
    style_fingerprint: Dict,
    current_batch_num: int,
    is_ymyl: bool = False,
    main_keyword: str = "",
    h2_titles: List[str] = None,
    previous_batch_text: str = None,
    overused_words: List[str] = None
) -> Dict[str, Any]:
    """
    Generuje konkretne instrukcje stylistyczne dla GPT.
    
    v40.1: DYNAMICZNA HUMANIZACJA
    - Krótkie zdania dopasowane do TEMATU (prawo/medycyna/finanse/etc.)
    - Synonimy kontekstowe
    - Analiza burstiness poprzedniego batcha
    """
    
    # 🆕 v40.1: DYNAMICZNE KRÓTKIE ZDANIA (zastępuje SHORT_INSERTS_LIBRARY)
    short_sentences_data = get_dynamic_short_sentences(
        main_keyword=main_keyword or style_fingerprint.get("main_keyword", ""),
        h2_titles=h2_titles or style_fingerprint.get("h2_titles", []),
        count=8,
        include_questions=True
    )
    
    # 🆕 v40.1: DYNAMICZNE SYNONIMY z kontekstem domeny (plWordNet + cache)
    domain_context = short_sentences_data.get("domain", "universal")
    synonyms_data = get_synonym_instructions(
        overused_words=overused_words or style_fingerprint.get("overused_words", []),
        context=domain_context  # 🆕 v40.1: Przekaż domenę dla lepszych synonimów
    )
    
    # 🆕 v40.1: BURSTINESS Z ANALIZĄ POPRZEDNIEGO BATCHA
    burstiness_data = get_burstiness_instructions(previous_batch_text)
    
    instructions = {
        # ================================================================
        # 🆕 v40.1: BURSTINESS - kluczowa metryka humanizacji
        # ================================================================
        "burstiness_critical": {
            "instruction": "⚠️ ZRÓŻNICUJ długości zdań! CV musi być > 0.40",
            "why": "Monotonne zdania 15-20 słów = wykrycie AI!",
            "example_lengths": burstiness_data.get("example_sequence", "5, 18, 8, 25, 12, 6, 30, 14 słów"),
            "target_distribution": {
                "short_3_8_words": "20-25% (np. 'To ważne.', 'Sąd orzeka.')",
                "medium_10_18_words": "50-60%",
                "long_22_35_words": "15-25%"
            },
            "avoid": "❌ NIE PISZ wszystkich zdań 15-22 słów!",
            "previous_batch_analysis": burstiness_data.get("previous_batch_analysis")
        },
        
        # ================================================================
        # 🆕 v40.1: DYNAMICZNE KRÓTKIE ZDANIA (tematyczne)
        # ================================================================
        "short_sentences_dynamic": {
            "instruction": short_sentences_data.get("instruction", "Wstaw 2-4 krótkie zdania"),
            "domain": short_sentences_data.get("domain", "universal"),
            "examples": short_sentences_data.get("sentences", [])[:8],
            "usage": "Wstaw 2-4 takie zdania w każdym batchu. NIE POWTARZAJ!",
            "tip": "Możesz tworzyć WŁASNE krótkie zdania (3-8 słów) pasujące do tematu"
        },
        
        # ================================================================
        # 🆕 v40.1: SYNONIMY DYNAMICZNE (kontekstowe)
        # ================================================================
        "synonyms_dynamic": {
            "instruction": synonyms_data.get("instruction", "NIE POWTARZAJ tych samych słów!"),
            "priority": synonyms_data.get("priority", "NORMAL"),
            "map": synonyms_data.get("synonyms", {}),
            "warning": "Nie powtarzaj tego samego słowa >3× w batchu!"
        },
        
        # ================================================================
        # SŁOWA ŁĄCZĄCE z polish_language_quality
        # ================================================================
        "transition_words_pl": {
            "instruction": "Używaj polskich słów łączących:",
            "kontrast": TRANSITION_WORDS_CATEGORIZED.get("kontrast", ["jednak", "natomiast"])[:5],
            "przyczyna": TRANSITION_WORDS_CATEGORIZED.get("przyczyna", ["ponieważ", "bowiem"])[:5],
            "skutek": TRANSITION_WORDS_CATEGORIZED.get("skutek", ["dlatego", "zatem"])[:5],
            "sekwencja": TRANSITION_WORDS_CATEGORIZED.get("czas_sekwencja", ["najpierw", "następnie"])[:5]
        },
        
        # ================================================================
        # AI PATTERNS DO UNIKANIA
        # ================================================================
        "avoid_ai_patterns": {
            "instruction": "UNIKAJ tych fraz (typowe dla AI):",
            "patterns": AI_PATTERNS_TO_AVOID[:10],
            "patterns_with_fixes": {
                "warto podkreślić": "usuń lub 'Zwróć uwagę:'",
                "należy pamiętać": "'Pamiętaj:' lub usuń",
                "w kontekście": "'przy', 'podczas'",
                "istotne jest": "'Ważne:'",
                "kluczowym aspektem": "usuń",
                "kompleksowe omówienie": "'omówienie'",
                "warto zauważyć": "usuń",
                "nie bez znaczenia": "'Ważne:'"
            }
        },
        
        "pronouns_consistency": {
            "instruction": "Wybierz JEDEN styl i TRZYMAJ SIĘ go!",
            "options": ["bezosobowo (można, należy)", "per 'ty' (możesz, powinieneś)"],
            "warning": "❌ NIE mieszaj stylów w jednym artykule!"
        },
        
        "natural_flow": {
            "instruction": "Pisz jak ekspert tłumaczący znajomemu",
            "tips": [
                "Używaj pytań retorycznych (Co dalej? Dlaczego to ważne?)",
                "Dodaj krótkie zdania dla naturalności",
                "Nie każde zdanie musi być 'mądrą' definicją"
            ]
        }
    }
    
    # Jeśli mamy fingerprint z poprzednich batchów
    if style_fingerprint and style_fingerprint.get("analyzed_batches", 0) > 0:
        instructions["match_established_style"] = {
            "instruction": "ZACHOWAJ styl z poprzednich batchów:",
            "formality": style_fingerprint.get("formality_level", "semi_formal"),
            "pronouns": style_fingerprint.get("personal_pronouns", "bezosobowo"),
            "avg_sentence_length": style_fingerprint.get("sentence_length_avg", 16),
            "example_sentences": style_fingerprint.get("example_sentences", [])[:2]
        }
    
    # YMYL - dodatkowe wymagania
    if is_ymyl:
        instructions["ymyl_precision"] = {
            "instruction": "Treść YMYL - wymagana precyzja!",
            "requirements": [
                "Cytuj przepisy: 'art. X § Y k.c.'",
                "Nie używaj 'zaleca się', 'warto' - pisz konkretnie",
                "Dodaj disclaimer na końcu artykułu"
            ]
        }
    
    return instructions


# ============================================================================
# 5. CONTINUATION CONTEXT - połączenie między batchami
# ============================================================================

def get_continuation_context(
    batches: List[Dict],
    keywords_state: Dict,
    style_fingerprint: Dict,
    entity_state: Dict
) -> Dict[str, Any]:
    """Generuje pełny kontekst kontynuacji dla GPT."""
    if not batches:
        return {
            "is_first_batch": True,
            "instruction": "To jest PIERWSZY batch - wprowadź temat"
        }
    
    last_batch = batches[-1]
    last_text = last_batch.get("text", "")
    
    paragraphs = re.split(r'\n\n+', last_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and not p.startswith("h2:")]
    
    last_paragraph = ""
    if paragraphs:
        last_paragraph = paragraphs[-1]
        words = last_paragraph.split()
        if len(words) > CONFIG.LAST_PARAGRAPH_WORDS:
            last_paragraph = " ".join(words[-CONFIG.LAST_PARAGRAPH_WORDS:])
    
    defined_entities = {}
    for ent, batch_num in entity_state.get("introduced_entities", {}).items():
        definition = entity_state.get("defined_terms", {}).get(ent, "wprowadzone")
        defined_entities[ent] = {
            "status": "zdefiniowane" if definition != "wprowadzone" else "wspomniane",
            "in_batch": batch_num
        }
    
    last_h2 = ""
    h2_match = re.search(r'^h2:\s*(.+)$', last_text, re.MULTILINE | re.IGNORECASE)
    if h2_match:
        last_h2 = h2_match.group(1).strip()
    
    return {
        "is_first_batch": False,
        "last_paragraph": last_paragraph,
        "last_h2": last_h2,
        "batches_completed": len(batches),
        
        "established_entities": defined_entities,
        "instruction_entities": "NIE powtarzaj definicji tych pojęć - są już wyjaśnione",
        
        "style_fingerprint": {
            "tone": style_fingerprint.get("formality_level", "semi_formal"),
            "pronouns": style_fingerprint.get("personal_pronouns", "bezosobowo"),
            "avg_sentence_length": round(style_fingerprint.get("sentence_length_avg", 16))
        },
        "instruction_style": "ZACHOWAJ ten sam styl pisania!",
        
        "continuation_instruction": "KONTYNUUJ narrację płynnie. Pierwsze zdanie powinno nawiązywać do poprzedniej sekcji.",
        
        "example_transitions": [
            "Kolejnym aspektem jest...",
            "Omawiając [temat H2], należy...",
            "W kontekście [poprzedniego tematu], warto teraz...",
            "Przechodząc do [nowy temat]..."
        ]
    }


# ============================================================================
# 6. STRUCTURE INSTRUCTIONS - H3, listy, długość sekcji
# ============================================================================

def get_structure_instructions(
    current_batch_num: int,
    total_batches: int,
    h2_structure: List[str],
    current_h2: str,
    batch_type: str
) -> Dict[str, Any]:
    """Generuje instrukcje strukturalne dla batcha."""
    h2_index = 0
    for i, h2 in enumerate(h2_structure):
        if h2.lower() == current_h2.lower() or current_h2.lower() in h2.lower():
            h2_index = i
            break
    
    num_h2 = len(h2_structure)
    longest_section_index = num_h2 // 2
    
    if batch_type == "INTRO":
        paragraphs_target = 2
        length_profile = "SHORT"
    elif batch_type == "FINAL":
        paragraphs_target = 3
        length_profile = "MEDIUM"
    elif h2_index == longest_section_index:
        paragraphs_target = 5
        length_profile = "LONG"
    elif h2_index < longest_section_index:
        paragraphs_target = 2 + h2_index
        length_profile = "MEDIUM" if paragraphs_target >= 3 else "SHORT"
    else:
        paragraphs_target = 4 - (h2_index - longest_section_index)
        paragraphs_target = max(2, paragraphs_target)
        length_profile = "MEDIUM" if paragraphs_target >= 3 else "SHORT"
    
    has_h3 = (h2_index == longest_section_index and length_profile == "LONG")
    
    has_list = (
        h2_index > longest_section_index and 
        batch_type != "FINAL" and 
        not has_h3
    )
    
    if not has_list and h2_index == num_h2 - 2:
        has_list = True
    
    return {
        "paragraphs_target": paragraphs_target,
        "length_profile": length_profile,
        "has_h3": has_h3,
        "h3_instruction": "Ta sekcja MUSI mieć H3 (np. podział na kroki/etapy)" if has_h3 else None,
        "has_list": has_list,
        "list_instruction": "Ta sekcja MUSI zawierać listę wypunktowaną" if has_list else None,
        "is_longest_section": h2_index == longest_section_index,
        "section_index": h2_index,
        "total_sections": num_h2,
        "summary": _get_structure_summary(paragraphs_target, has_h3, has_list, length_profile)
    }


def _get_structure_summary(paragraphs: int, has_h3: bool, has_list: bool, profile: str) -> str:
    """Generuje podsumowanie struktury dla GPT."""
    parts = [f"{paragraphs} paragrafów ({profile})"]
    if has_h3:
        parts.append("+ H3 (NAJDŁUŻSZA sekcja)")
    if has_list:
        parts.append("+ lista wypunktowana")
    return ", ".join(parts)


# ============================================================================
# 7. KEYWORD TRACKING MODE - tracking zamiast blokowania
# ============================================================================

def get_keyword_tracking_info(
    keywords_state: Dict,
    current_batch_num: int,
    total_batches: int,
    remaining_batches: int
) -> Dict[str, Any]:
    """Generuje informacje o keywords w trybie TRACKING (nie blokującym)."""
    tracking = {
        "mode": "TRACKING",
        "explanation": "Frazy są ŚLEDZONE w tle. Per-batch nie blokuje. Weryfikacja globalna w final_review.",
        
        "use_naturally": [],
        "available": [],
        "near_limit": [],
        "structural": [],
    }
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        is_structural = meta.get("is_structural", False) or is_main
        
        remaining_needed = max(0, target_min - actual)
        remaining_allowed = max(0, target_max - actual)
        
        if remaining_batches > 0:
            suggested = math.ceil(remaining_needed / remaining_batches) if remaining_needed > 0 else 0
            max_here = math.ceil(remaining_allowed / remaining_batches)
        else:
            suggested = remaining_needed
            max_here = remaining_allowed
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "actual_total": actual,
            "target": f"{target_min}-{target_max}",
            "remaining_needed": remaining_needed,
            "remaining_allowed": remaining_allowed,
            "suggested_this_batch": min(suggested, 3),
            "max_this_batch": min(max_here, 5)
        }
        
        if is_structural:
            kw_info["note"] = "🔵 STRUCTURAL - użyj naturalnie, limit globalny"
            tracking["structural"].append(kw_info)
        elif remaining_allowed <= 2:
            kw_info["note"] = "⚠️ Blisko limitu - max 1× tu"
            tracking["near_limit"].append(kw_info)
        elif remaining_needed > 0:
            kw_info["note"] = f"Użyj ~{suggested}× w tym batchu"
            tracking["use_naturally"].append(kw_info)
        else:
            kw_info["note"] = "✓ W normie, opcjonalnie 1×"
            tracking["available"].append(kw_info)
    
    tracking["summary"] = {
        "total_keywords": len(keywords_state),
        "need_usage": len(tracking["use_naturally"]),
        "near_limit": len(tracking["near_limit"]),
        "structural": len(tracking["structural"]),
        "instruction": "Użyj fraz NATURALNIE. System śledzi ilości automatycznie. Nie rób stuffingu!"
    }
    
    return tracking


# ============================================================================
# 8. DYNAMIC BATCH COUNT - na podstawie S1
# ============================================================================

def calculate_optimal_batch_count(
    s1_data: Dict,
    keywords_count: int,
    h2_count: int,
    target_length: int,
    is_ymyl: bool = False
) -> Dict[str, Any]:
    """Oblicza optymalną liczbę batchów na podstawie analizy S1."""
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    relationships = entity_seo.get("entity_relationships", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    high_entities = len([e for e in entities if e.get("importance", 0) >= 0.7])
    must_topics = len([t for t in topical_coverage if t.get("priority") == "MUST"])
    
    base_batches = h2_count + 1
    complexity_batches = 0
    
    if high_entities > 8:
        complexity_batches += 2
    elif high_entities > 5:
        complexity_batches += 1
    
    if len(relationships) > 6:
        complexity_batches += 1
    
    if keywords_count > 25:
        complexity_batches += 2
    elif keywords_count > 15:
        complexity_batches += 1
    
    if is_ymyl:
        complexity_batches += 1
    
    if target_length > 3500:
        complexity_batches += 2
    elif target_length > 2500:
        complexity_batches += 1
    
    optimal = base_batches + complexity_batches
    
    min_batches = max(4, h2_count)
    max_batches = 15
    
    optimal = max(min_batches, min(optimal, max_batches))
    
    return {
        "recommended_batches": optimal,
        "min_batches": min_batches,
        "max_batches": max_batches,
        "factors": {
            "h2_count": h2_count,
            "high_entities": high_entities,
            "must_topics": must_topics,
            "relationships": len(relationships),
            "keywords": keywords_count,
            "target_length": target_length,
            "is_ymyl": is_ymyl
        },
        "explanation": f"Zalecane {optimal} batchów: {h2_count} H2 + intro + {complexity_batches} dla złożoności"
    }


# ============================================================================
# 9. MAIN FUNCTION - generuje kompletne enhanced pre_batch_info
# ============================================================================

def generate_enhanced_pre_batch_info(
    s1_data: Dict,
    keywords_state: Dict,
    batches: List[Dict],
    h2_structure: List[str],
    current_batch_num: int,
    total_batches: int,
    main_keyword: str,
    entity_state: Dict = None,
    style_fingerprint: Dict = None,
    is_ymyl: bool = False,
    is_legal: bool = False,
    batch_plan: Dict = None  # 🆕 v40.1: Plan batcha z h2_sections
) -> Dict[str, Any]:
    """Generuje KOMPLETNE enhanced pre_batch_info z konkretnymi instrukcjami."""
    if entity_state is None:
        entity_state = {}
    if style_fingerprint is None:
        style_fingerprint = {}
    if batch_plan is None:
        batch_plan = {}
    
    remaining_batches = max(1, total_batches - len(batches))
    
    # ================================================================
    # 🆕 v40.1: NAPRAWIONA LOGIKA H2 PER BATCH
    # Batch może mieć WIELE H2 (zgodnie z batch_planner.py)
    # ================================================================
    
    # Najpierw sprawdź czy mamy h2_sections z batch_plan
    batch_h2_from_plan = None
    if batch_plan:
        # batch_plan może być dict z "batches" lub bezpośrednio info o batchu
        batches_list = batch_plan.get("batches", [])
        if batches_list and current_batch_num <= len(batches_list):
            current_batch_plan = batches_list[current_batch_num - 1]
            batch_h2_from_plan = current_batch_plan.get("h2_sections", [])
    
    # Fallback: oblicz remaining_h2 jak wcześniej
    used_h2 = []
    for batch in batches:
        # Szukaj wszystkich H2 w tekście batcha
        h2_matches = re.findall(r'^h2:\s*(.+)$', batch.get("text", ""), re.MULTILINE | re.IGNORECASE)
        used_h2.extend([h.strip() for h in h2_matches])
        # Szukaj też w formatcie HTML
        h2_html = re.findall(r'<h2[^>]*>([^<]+)</h2>', batch.get("text", ""), re.IGNORECASE)
        used_h2.extend([h.strip() for h in h2_html])
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    # Określ H2 dla tego batcha
    if batch_h2_from_plan:
        # Użyj h2_sections z batch_plan (może być lista!)
        current_h2_list = batch_h2_from_plan
        current_h2 = current_h2_list[0] if current_h2_list else main_keyword
    else:
        # Fallback: oblicz ile H2 przypada na batch
        h2_per_batch = max(1, len(h2_structure) // total_batches) if total_batches > 0 else 1
        if current_batch_num == 1:
            # INTRO - bez H2
            current_h2_list = []
            current_h2 = main_keyword
        elif current_batch_num >= total_batches:
            # FINAL - wszystkie pozostałe H2
            current_h2_list = remaining_h2
            current_h2 = remaining_h2[0] if remaining_h2 else main_keyword
        else:
            # CONTENT - weź odpowiednią liczbę H2
            current_h2_list = remaining_h2[:h2_per_batch]
            current_h2 = current_h2_list[0] if current_h2_list else main_keyword
    
    # Batch type
    if current_batch_num == 1:
        batch_type = "INTRO"
    elif current_batch_num >= total_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # 🆕 v40.1: Pobierz tekst poprzedniego batcha do analizy burstiness
    previous_batch_text = None
    if batches:
        previous_batch_text = batches[-1].get("text", "")
    
    # 🆕 v40.1: Wykryj nadużywane słowa
    overused_words = style_fingerprint.get("overused_words", [])
    
    # 🆕 v40.1: Oblicz remaining_h2 po użyciu current_h2_list
    remaining_h2_after_current = [h2 for h2 in remaining_h2 if h2 not in current_h2_list]
    
    # ================================================================
    # GENERUJ WSZYSTKIE SEKCJE
    # ================================================================
    
    enhanced = {
        "batch_number": current_batch_num,
        "total_batches": total_batches,
        "batch_type": batch_type,
        "current_h2": current_h2,  # Pierwszy H2 (dla kompatybilności)
        "current_h2_list": current_h2_list,  # 🆕 v40.1: WSZYSTKIE H2 dla tego batcha
        "h2_count_in_batch": len(current_h2_list),  # 🆕 v40.1: Ile H2 w tym batchu
        "remaining_h2": remaining_h2_after_current[:4],  # Następne H2 (po tym batchu)
        
        # 1. ENCJE DO ZDEFINIOWANIA
        "entities_to_define": get_entities_to_define(
            s1_data=s1_data,
            current_batch_num=current_batch_num,
            entity_state=entity_state,
            current_h2=current_h2,
            total_batches=total_batches
        ),
        
        # 2. RELACJE DO USTANOWIENIA
        "relations_to_establish": get_relations_to_establish(
            s1_data=s1_data,
            current_batch_num=current_batch_num,
            entity_state=entity_state,
            total_batches=total_batches
        ),
        
        # 3. KONTEKST SEMANTYCZNY
        "semantic_context": get_semantic_context(
            s1_data=s1_data,
            current_batch_num=current_batch_num,
            current_h2=current_h2,
            keywords_state=keywords_state
        ),
        
        # 4. INSTRUKCJE STYLU (v40.1 - DYNAMIC)
        "style_instructions": get_style_instructions(
            style_fingerprint=style_fingerprint,
            current_batch_num=current_batch_num,
            is_ymyl=is_ymyl,
            main_keyword=main_keyword,
            h2_titles=h2_structure,
            previous_batch_text=previous_batch_text,
            overused_words=overused_words
        ),
        
        # 5. KONTEKST KONTYNUACJI
        "continuation": get_continuation_context(
            batches=batches,
            keywords_state=keywords_state,
            style_fingerprint=style_fingerprint,
            entity_state=entity_state
        ),
        
        # 6. KEYWORD TRACKING
        "keyword_tracking": get_keyword_tracking_info(
            keywords_state=keywords_state,
            current_batch_num=current_batch_num,
            total_batches=total_batches,
            remaining_batches=remaining_batches
        ),
        
        # 7. STRUCTURE INSTRUCTIONS - H3, listy, długość
        "structure_instructions": get_structure_instructions(
            current_batch_num=current_batch_num,
            total_batches=total_batches,
            h2_structure=h2_structure,
            current_h2=current_h2,
            batch_type=batch_type
        )
    }
    
    # 🆕 v40.1: ADVANCED SEMANTIC - entity density, topic completeness, entity gap
    if ADVANCED_SEMANTIC_ENABLED:
        try:
            # Zbierz tekst wszystkich poprzednich batchów
            all_previous_text = "\n".join([b.get("text", "") for b in batches]) if batches else ""
            
            # Pobierz oczekiwane encje z S1
            expected_entities = []
            entity_seo = s1_data.get("entity_seo", {})
            for ent in entity_seo.get("entities", [])[:20]:
                expected_entities.append(ent.get("name", "") if isinstance(ent, dict) else str(ent))
            
            # Wykryj brakujące encje
            entity_gap = detect_entity_gap(
                text=all_previous_text,
                expected_entities=expected_entities,
                topic=main_keyword
            )
            
            enhanced["advanced_semantic"] = {
                "entity_gap": {
                    "missing_entities": entity_gap.get("missing_hard_entities", [])[:5],
                    "missing_count": entity_gap.get("hard_missing_count", 0),
                    "status": entity_gap.get("status", "OK")
                },
                "instructions": generate_advanced_prompt_instructions(entity_gap) if entity_gap.get("status") != "OK" else None
            }
            
            # Dodaj do fixes_needed jeśli brakuje wielu encji
            if entity_gap.get("hard_missing_count", 0) > 3:
                enhanced["advanced_semantic"]["priority_entities"] = entity_gap.get("missing_hard_entities", [])[:3]
                
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Advanced semantic error: {e}")
            enhanced["advanced_semantic"] = {"error": str(e)}
    
    # GPT PROMPT SECTION
    enhanced["gpt_instructions"] = _generate_gpt_prompt_section(enhanced, is_legal)
    
    # 🆕 v40.2: CONCEPT MAP (Semantic Entity SEO)
    if CONCEPT_MAP_AVAILABLE and current_batch_num == 1:
        try:
            # Pobierz teksty konkurencji z S1
            competitor_texts = []
            competitor_h2 = s1_data.get("competitor_h2", [])
            for comp in competitor_h2[:5]:
                if isinstance(comp, dict):
                    # Zbierz teksty z H2 konkurencji
                    h2_texts = comp.get("h2_content", [])
                    if h2_texts:
                        competitor_texts.append(" ".join(h2_texts[:3]))
            
            # Fallback: użyj PAA jako źródła semantycznego
            if not competitor_texts:
                paa_data = s1_data.get("paa", [])
                for paa in paa_data[:5]:
                    q = paa.get("question", "") if isinstance(paa, dict) else str(paa)
                    competitor_texts.append(q)
            
            if competitor_texts:
                result = extract_concept_map(
                    main_keyword=main_keyword,
                    competitor_texts=competitor_texts
                )
                
                if result.get("status") == "OK":
                    enhanced["concept_map"] = result.get("concept_map", {})
                    print(f"[ENHANCED_PRE_BATCH] ✅ Concept map extracted for '{main_keyword}'")
                else:
                    enhanced["concept_map"] = result.get("concept_map", {})
                    print(f"[ENHANCED_PRE_BATCH] ⚠️ Concept map fallback used")
            else:
                enhanced["concept_map"] = get_fallback_concept_map(main_keyword)
                
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Concept map error: {e}")
            enhanced["concept_map"] = {"error": str(e)}
    
    # 🆕 v40.2: STYLE FINGERPRINT INSTRUCTIONS
    if STYLE_ANALYZER_AVAILABLE and style_fingerprint:
        try:
            # Konwertuj dict na StyleFingerprint jeśli to dict
            if isinstance(style_fingerprint, dict) and style_fingerprint:
                fp = StyleFingerprint(
                    formality_score=style_fingerprint.get("formality_score", 0.5),
                    sentence_length_avg=style_fingerprint.get("sentence_length_avg", 18.0),
                    passive_voice_ratio=style_fingerprint.get("passive_voice_ratio", 0.15),
                    example_sentences=style_fingerprint.get("example_sentences", []),
                    preferred_transitions=style_fingerprint.get("preferred_transitions", [])
                )
                enhanced["style_fingerprint_instructions"] = fp.generate_style_instructions()
            elif isinstance(style_fingerprint, StyleFingerprint):
                enhanced["style_fingerprint_instructions"] = style_fingerprint.generate_style_instructions()
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Style fingerprint error: {e}")
    
    # ================================================================
    # 🆕 v41.0: PARAGRAPH CV INSTRUCTIONS
    # ================================================================
    if current_batch_num >= 2 and batches:
        try:
            # Zbierz tekst wszystkich poprzednich batchów
            accumulated_text = "\n".join([b.get("text", "") for b in batches])
            
            para_cv_alert = get_paragraph_cv_for_prebatch(
                accumulated_text=accumulated_text,
                batch_number=current_batch_num
            )
            
            if para_cv_alert:
                enhanced["paragraph_cv_alert"] = para_cv_alert
                # Dodaj do gpt_instructions
                if "style_instructions" not in enhanced:
                    enhanced["style_instructions"] = {}
                style_inst = enhanced.get("style_instructions", {})
                warnings = style_inst.get("warnings", [])
                warnings.append(para_cv_alert["instruction"])
                enhanced["style_instructions"]["warnings"] = warnings
                print(f"[ENHANCED_PRE_BATCH] ⚠️ Paragraph CV {para_cv_alert['severity']}: CV={para_cv_alert.get('cv', 'N/A')}")
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Paragraph CV error: {e}")
    
    # ================================================================
    # 🆕 v41.0: TRIPLET PRIORITY INSTRUCTIONS
    # ================================================================
    if "relations_to_establish" in enhanced:
        try:
            relations = enhanced.get("relations_to_establish", {})
            if relations.get("prebatch_instruction"):
                # Dodaj instrukcję tripletów do encji
                if "entities_to_define" not in enhanced:
                    enhanced["entities_to_define"] = {}
                ent_def = enhanced.get("entities_to_define", {})
                instructions = ent_def.get("instructions", [])
                instructions.append(relations["prebatch_instruction"])
                enhanced["entities_to_define"]["instructions"] = instructions
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Triplet priority error: {e}")
    
    return enhanced


def _generate_gpt_prompt_section(enhanced: Dict, is_legal: bool = False) -> str:
    """Generuje gotową sekcję promptu dla GPT."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"📋 BATCH #{enhanced['batch_number']} - {enhanced['batch_type']}")
    lines.append("=" * 60)
    lines.append("")
    
    # 🆕 v40.1: Pokaż WSZYSTKIE H2 dla tego batcha
    h2_list = enhanced.get("current_h2_list", [])
    h2_count = enhanced.get("h2_count_in_batch", 0)
    
    # 🆕 v41.2: Różna liczba akapitów dla każdej H2 (2-4)
    paragraph_options = [2, 3, 4, 3, 2, 4]
    
    if h2_count > 1:
        lines.append(f"📌 H2 W TYM BATCHU ({h2_count} sekcje):")
        for i, h2 in enumerate(h2_list, 1):
            para_count = paragraph_options[(i - 1) % len(paragraph_options)]
            lines.append(f"   {i}. \"{h2}\" → {para_count} akapity")
        lines.append("")
        lines.append("⚠️ WYMAGANE: Napisz WSZYSTKIE powyższe sekcje H2 w tym batchu!")
        lines.append("⚠️ WAŻNE: Każda sekcja H2 MUSI mieć INNĄ liczbę akapitów (2-4)!")
    elif h2_count == 1:
        lines.append(f"📌 H2: \"{enhanced['current_h2']}\" → 3 akapity")
    else:
        lines.append(f"📌 SEKCJA: {enhanced.get('batch_type', 'CONTENT')}")
    lines.append("")
    
    # Encje do zdefiniowania
    entities = enhanced.get("entities_to_define", [])
    if entities:
        lines.append("🧠 ENCJE DO ZDEFINIOWANIA:")
        for ent in entities[:4]:
            priority_icon = "🔴" if ent.get("priority") == "MUST" else "🟡"
            lines.append(f"   {priority_icon} {ent['entity']}")
            lines.append(f"      → {ent['how']}")
        lines.append("")
    
    # Relacje
    relations = enhanced.get("relations_to_establish", [])
    if relations:
        lines.append("🔗 RELACJE DO USTANOWIENIA:")
        for rel in relations[:3]:
            lines.append(f"   • {rel['from']} → {rel['relation']} → {rel['to']}")
            if rel.get("example_sentences"):
                lines.append(f"     Przykład: \"{rel['example_sentences'][0]}\"")
        lines.append("")
    
    # Kontekst semantyczny
    semantic = enhanced.get("semantic_context", {})
    context_terms = semantic.get("context_terms", [])
    if context_terms:
        lines.append("📚 TERMINY KONTEKSTOWE (użyj naturalnie):")
        lines.append(f"   {', '.join(context_terms[:5])}")
        lines.append("")
    
    # 🆕 v40.1: Krótkie zdania (dynamiczne)
    style = enhanced.get("style_instructions", {})
    short_sentences = style.get("short_sentences_dynamic", {})
    if short_sentences.get("examples"):
        domain = short_sentences.get("domain", "universal")
        lines.append(f"✂️ KRÓTKIE ZDANIA ({domain.upper()}):")
        lines.append(f"   {' | '.join(short_sentences['examples'][:5])}")
        lines.append("")
    
    # AI patterns do unikania
    if style.get("avoid_ai_patterns"):
        patterns = style["avoid_ai_patterns"].get("patterns", [])[:5]
        lines.append("🚫 UNIKAJ (typowe dla AI):")
        lines.append(f"   {', '.join(patterns)}")
        lines.append("")
    
    # Kontynuacja
    continuation = enhanced.get("continuation", {})
    if not continuation.get("is_first_batch"):
        lines.append("🔄 KONTYNUACJA:")
        if continuation.get("last_paragraph"):
            last_p = continuation["last_paragraph"][:200]
            lines.append(f"   Ostatni akapit: \"{last_p}...\"")
        
        established = continuation.get("established_entities", {})
        if established:
            defined = [k for k, v in established.items() if v.get("status") == "zdefiniowane"][:5]
            if defined:
                lines.append(f"   ✓ Już zdefiniowane: {', '.join(defined)}")
        lines.append("")
    
    # Keywords summary
    tracking = enhanced.get("keyword_tracking", {})
    summary = tracking.get("summary", {})
    if summary:
        lines.append(f"📊 KEYWORDS: {summary.get('need_usage', 0)} do użycia | {summary.get('near_limit', 0)} blisko limitu")
        lines.append("   💡 Użyj NATURALNIE - system śledzi automatycznie")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'generate_enhanced_pre_batch_info',
    'get_entities_to_define',
    'get_relations_to_establish',
    'get_semantic_context',
    'get_style_instructions',
    'get_continuation_context',
    'get_keyword_tracking_info',
    'get_structure_instructions',
    'calculate_optimal_batch_count',
    'CONFIG',
    'AI_PATTERNS_TO_AVOID',
    'NATURAL_ALTERNATIVES',
    'DYNAMIC_HUMANIZATION_AVAILABLE'
]
