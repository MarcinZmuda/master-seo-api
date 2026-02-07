"""
===============================================================================
🎯 ENHANCED PRE-BATCH INSTRUCTIONS v43.1
===============================================================================
Moduł generujący KONKRETNE instrukcje dla GPT zamiast surowych danych.

🆕 v43.1 ZMIANY:
- PEŁNE instrukcje humanizacji w GPT prompt (burstiness, struktura, AI patterns)
- Instrukcje o różnej liczbie akapitów między sekcjami
- Rozkład długości zdań (20-25% krótkich, 50-60% średnich, 15-25% długich)
- Instrukcje o różnej długości akapitów (40-150 słów)
- 📋 LISTY I WYPUNKTOWANIA - dynamiczne instrukcje kiedy używać list
- 📑 H3 SUBHEADINGI - instrukcje o dzieleniu długich sekcji

🆕 v43.0 ZMIANY:
- Integracja z phrase_hierarchy.py (hierarchia fraz - zapobiega stuffing)
- Nowy parametr phrase_hierarchy_data w generate_enhanced_pre_batch_info()
- Sekcja 🌳 HIERARCHIA FRAZ w gpt_instructions
- Informacje o rdzeniach, rozszerzeniach i strategiach

🆕 v42.0 ZMIANY:
- Integracja z overflow_buffer.py (FAQ dla sierocych fraz)
- Integracja z semantic_triplet_validator.py (instrukcje semantyczne dla tripletów)
- Integracja z dynamic_sub_batch.py (info o sub-batchach)
- Nowy format instrukcji tripletów (akceptuje warianty językowe)

===============================================================================
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# IMPORTS Z MODUŁÓW OPTYMALIZACYJNYCH
try:
    from paragraph_cv_analyzer_v41 import (
        get_paragraph_cv_for_prebatch,
        calculate_paragraph_cv
    )
    PARAGRAPH_CV_AVAILABLE = True
except ImportError:
    PARAGRAPH_CV_AVAILABLE = False
    def get_paragraph_cv_for_prebatch(*args, **kwargs): return None

try:
    from triplet_priority_v41 import get_triplet_instructions_for_prebatch
    TRIPLET_PRIORITY_AVAILABLE = True
except ImportError:
    TRIPLET_PRIORITY_AVAILABLE = False

# DYNAMIC STRUCTURE ANALYZER
try:
    from dynamic_structure_analyzer import (
        analyze_structure_requirements,
        should_batch_have_h3,
        should_batch_have_list,
        StructureRequirements
    )
    DYNAMIC_STRUCTURE_AVAILABLE = True
except ImportError:
    DYNAMIC_STRUCTURE_AVAILABLE = False

# SMART BATCH INSTRUCTIONS
try:
    from smart_batch_instructions import (
        generate_smart_batch_instructions,
        format_instructions_for_gpt
    )
    SMART_INSTRUCTIONS_AVAILABLE = True
except ImportError:
    SMART_INSTRUCTIONS_AVAILABLE = False

# ================================================================
# 🆕 v42.0: OVERFLOW BUFFER (FAQ dla sierocych fraz)
# ================================================================
try:
    from overflow_buffer import (
        create_overflow_buffer,
        format_faq_instructions,
        identify_orphan_phrases,
        OverflowBuffer
    )
    OVERFLOW_BUFFER_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ overflow_buffer v1.0 loaded (auto FAQ)")
except ImportError:
    OVERFLOW_BUFFER_AVAILABLE = False
    print("[ENHANCED_PRE_BATCH] ⚠️ overflow_buffer not available")
    
    def create_overflow_buffer(**kwargs):
        class EmptyBuffer:
            orphan_phrases = []
            faq_items = []
            section_title = ""
        return EmptyBuffer()
    
    def format_faq_instructions(buffer):
        return ""

# ================================================================
# 🆕 v42.0: SEMANTIC TRIPLET VALIDATOR (instrukcje semantyczne)
# ================================================================
try:
    from semantic_triplet_validator import (
        generate_semantic_instruction,
        validate_triplets_in_text
    )
    SEMANTIC_TRIPLET_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ semantic_triplet_validator v1.0 loaded")
except ImportError:
    SEMANTIC_TRIPLET_AVAILABLE = False
    print("[ENHANCED_PRE_BATCH] ⚠️ semantic_triplet_validator not available")
    
    def generate_semantic_instruction(triplet):
        s = triplet.get("subject", "")
        v = triplet.get("verb", "")
        o = triplet.get("object", "")
        return f"Napisz zdanie: {s} {v} {o}"

# CONCEPT MAP EXTRACTOR
try:
    from concept_map_extractor import (
        extract_concept_map,
        get_fallback_concept_map,
        validate_concept_map
    )
    CONCEPT_MAP_AVAILABLE = True
except ImportError:
    CONCEPT_MAP_AVAILABLE = False
    def extract_concept_map(main_kw, competitor_texts, **kwargs):
        return {"status": "UNAVAILABLE", "concept_map": {}}
    def get_fallback_concept_map(main_kw):
        return {"main_entity": {"name": main_kw, "type": "Thing"}}

# STYLE ANALYZER
try:
    from style_analyzer import StyleAnalyzer, StyleFingerprint
    STYLE_ANALYZER_AVAILABLE = True
except ImportError:
    STYLE_ANALYZER_AVAILABLE = False

# DYNAMIC HUMANIZATION
try:
    from dynamic_humanization import (
        get_dynamic_short_sentences,
        get_synonym_instructions,
        get_burstiness_instructions,
        get_humanization_instructions,
        detect_topic_domain,
        CONTEXTUAL_SYNONYMS
    )
    DYNAMIC_HUMANIZATION_AVAILABLE = True
except ImportError:
    DYNAMIC_HUMANIZATION_AVAILABLE = False
    CONTEXTUAL_SYNONYMS = {}
    def get_dynamic_short_sentences(main_kw, h2s=None, count=8, include_q=True, current_h2=None, batch_num=None):
        return {"domain": "universal", "sentences": [], "grammar_patterns": [], "context_hints": [], "instruction": "Twórz krótkie zdania (3-8 słów) pasujące do kontekstu akapitu. Wstaw 2-4 na batch."}
    def get_synonym_instructions(overused=None, context=None):
        return {"instruction": "Unikaj powtórzeń", "synonyms": {}}
    def get_burstiness_instructions(prev_text=None):
        return {"critical": True, "target_cv": ">0.40"}
    def detect_topic_domain(main_kw, h2s=None):
        return "universal"

# POLISH LANGUAGE QUALITY
try:
    from polish_language_quality import TRANSITION_WORDS_CATEGORIZED
    POLISH_QUALITY_AVAILABLE = True
except ImportError:
    POLISH_QUALITY_AVAILABLE = False
    TRANSITION_WORDS_CATEGORIZED = {
        "kontrast": ["jednak", "natomiast", "ale"],
        "przyczyna": ["ponieważ", "bowiem"],
        "skutek": ["dlatego", "zatem"],
        "sekwencja": ["najpierw", "następnie", "potem"]
    }

# ADVANCED SEMANTIC FEATURES
try:
    from advanced_semantic_features import (
        perform_advanced_semantic_analysis,
        generate_advanced_prompt_instructions,
        detect_entity_gap
    )
    ADVANCED_SEMANTIC_ENABLED = True
except ImportError:
    ADVANCED_SEMANTIC_ENABLED = False

# ================================================================
# 🆕 v43.0: PHRASE HIERARCHY INTEGRATION
# Dodaje kontekst hierarchii fraz do pre_batch_info
# ================================================================
try:
    from phrase_hierarchy import (
        dict_to_hierarchy,
        get_batch_hierarchy_context,
        format_hierarchy_summary_short,
        PhraseHierarchy
    )
    PHRASE_HIERARCHY_AVAILABLE = True
    print("[ENHANCED_PRE_BATCH] ✅ phrase_hierarchy v1.0 loaded")
except ImportError:
    PHRASE_HIERARCHY_AVAILABLE = False
    print("[ENHANCED_PRE_BATCH] ⚠️ phrase_hierarchy not available")
    
    # Fallback functions
    def dict_to_hierarchy(data):
        return None
    def get_batch_hierarchy_context(hierarchy, batch_phrases, current_counts=None):
        return {}
    def format_hierarchy_summary_short(hierarchy):
        return ""


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class EnhancedPreBatchConfig:
    MAX_ENTITIES_TO_DEFINE: int = 5
    MAX_RELATIONS_TO_ESTABLISH: int = 4
    MAX_NGRAMS_PER_BATCH: int = 6
    MIN_CONTEXT_TERMS: int = 3
    MAX_CONTEXT_TERMS: int = 6
    TARGET_SENTENCE_CV: float = 0.40
    MIN_SHORT_SENTENCES_PCT: int = 20
    MAX_AI_PATTERN_SENTENCES: int = 30
    LAST_PARAGRAPH_WORDS: int = 150


CONFIG = EnhancedPreBatchConfig()


# ============================================================================
# ENTITY DEFINITIONS
# ============================================================================

DEFINITION_TEMPLATES = {
    "legal_concept": [
        "Wyjaśnij że {entity} to instytucja prawna polegająca na...",
        "Zdefiniuj {entity} jako procedurę/mechanizm służący do...",
    ],
    "person_role": [
        "Przedstaw {entity} jako osobę/organ odpowiedzialną za...",
    ],
    "process": [
        "Opisz {entity} jako proces składający się z etapów...",
    ],
    "default": [
        "Zdefiniuj {entity} wyjaśniając czym jest i do czego służy",
    ]
}

ENTITY_TYPE_KEYWORDS = {
    "legal_concept": ["ubezwłasnowolnienie", "prawo", "przepis", "ustawa", "kodeks"],
    "person_role": ["sędzia", "kurator", "opiekun", "biegły", "prokurator"],
    "process": ["procedura", "postępowanie", "proces", "tryb"],
}


def classify_entity_type(entity: str) -> str:
    entity_lower = entity.lower()
    for entity_type, keywords in ENTITY_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in entity_lower:
                return entity_type
    return "default"


def generate_definition_instruction(entity: str, context: str = "", h2: str = "") -> Dict[str, Any]:
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
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    already_defined = set(entity_state.get("defined", []))
    result = []
    
    for topic in topical_coverage:
        if topic.get("priority") == "MUST":
            entity = topic.get("topic", "")
            if entity and entity.lower() not in {e.lower() for e in already_defined}:
                instruction = generate_definition_instruction(entity, topic.get("context", ""), current_h2)
                instruction["priority"] = "MUST"
                result.append(instruction)
    
    for ent in entities:
        if ent.get("importance", 0) >= 0.7:
            entity = ent.get("text", ent.get("entity", ""))
            if entity and entity.lower() not in {e.lower() for e in already_defined}:
                if entity.lower() not in {r["entity"].lower() for r in result}:
                    instruction = generate_definition_instruction(entity, ent.get("context", ""), current_h2)
                    instruction["priority"] = "SHOULD"
                    result.append(instruction)
    
    entities_per_batch = max(2, len(result) // total_batches)
    start_idx = (current_batch_num - 1) * entities_per_batch
    end_idx = min(start_idx + entities_per_batch + 1, len(result))
    
    if current_batch_num == 1:
        result.sort(key=lambda x: 0 if x.get("priority") == "MUST" else 1)
        return result[:CONFIG.MAX_ENTITIES_TO_DEFINE]
    
    return result[start_idx:end_idx][:CONFIG.MAX_ENTITIES_TO_DEFINE]


# ============================================================================
# 🆕 v42.0: RELATIONS WITH SEMANTIC INSTRUCTIONS
# ============================================================================

RELATION_TEMPLATES = {
    "orzeka": ["{subject} orzeka o {object}", "{subject} wydaje orzeczenie w sprawie {object}"],
    "prowadzi_do": ["{subject} może prowadzić do {object}", "{subject} skutkuje {object}"],
    "wymaga": ["{subject} wymaga {object}", "do {subject} niezbędne jest {object}"],
    "default": ["{subject} jest powiązane z {object}"]
}


def generate_relation_instruction(from_entity: str, relation: str, to_entity: str) -> Dict[str, Any]:
    """
    🆕 v42.0: Generuje instrukcję relacji w formacie SEMANTYCZNYM.
    Zamiast "napisz dosłownie" → "zachowaj sens relacji".
    """
    relation_lower = relation.lower().replace(" ", "_")
    templates = RELATION_TEMPLATES.get(relation_lower, RELATION_TEMPLATES["default"])
    
    example_sentences = [
        t.format(subject=from_entity, object=to_entity)
        for t in templates[:2]
    ]
    
    # 🆕 v42.0: Użyj semantic_triplet_validator dla lepszych instrukcji
    triplet = {"subject": from_entity, "verb": relation, "object": to_entity}
    
    if SEMANTIC_TRIPLET_AVAILABLE:
        semantic_instruction = generate_semantic_instruction(triplet)
    else:
        semantic_instruction = f"Napisz zdanie łączące '{from_entity}' z '{to_entity}' przez '{relation}'"
    
    return {
        "from": from_entity,
        "relation": relation,
        "to": to_entity,
        "example_sentences": example_sentences,
        "instruction": f"Ustanów relację: {from_entity} → {relation} → {to_entity}",
        "how": semantic_instruction,  # 🆕 v42.0: Semantyczna instrukcja
        "semantic_mode": SEMANTIC_TRIPLET_AVAILABLE,  # 🆕 v42.0: Info czy semantyczny
        "triplet": triplet  # 🆕 v42.0: Dla walidatora
    }


def get_relations_to_establish(
    s1_data: Dict,
    current_batch_num: int,
    entity_state: Dict,
    total_batches: int
) -> List[Dict]:
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
# SEMANTIC CONTEXT
# ============================================================================

def get_semantic_context(
    s1_data: Dict,
    current_batch_num: int,
    current_h2: str,
    keywords_state: Dict
) -> Dict[str, Any]:
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
# STYLE INSTRUCTIONS
# ============================================================================

AI_PATTERNS_TO_AVOID = [
    "warto podkreślić", "należy pamiętać", "w kontekście", "istotne jest",
    "kluczowym aspektem", "nie można pominąć", "szczególnie ważne",
    "fundamentalne znaczenie", "z perspektywy", "w odniesieniu do"
]


def get_style_instructions(
    style_fingerprint: Dict,
    current_batch_num: int,
    is_ymyl: bool = False,
    main_keyword: str = "",
    h2_titles: List[str] = None,
    previous_batch_text: str = None,
    overused_words: List[str] = None
) -> Dict[str, Any]:
    
    if DYNAMIC_HUMANIZATION_AVAILABLE:
        short_sentences_data = get_dynamic_short_sentences(
            main_keyword=main_keyword or style_fingerprint.get("main_keyword", ""),
            h2_titles=h2_titles or style_fingerprint.get("h2_titles", []),
            count=8,
            include_questions=True
        )
        domain_context = short_sentences_data.get("domain", "universal")
        synonyms_data = get_synonym_instructions(
            overused_words=overused_words or style_fingerprint.get("overused_words", []),
            context=domain_context
        )
        burstiness_data = get_burstiness_instructions(previous_batch_text)
    else:
        short_sentences_data = {"domain": "universal", "sentences": [], "instruction": "Twórz krótkie zdania (3-8 słów) pasujące do kontekstu akapitu."}
        synonyms_data = {"instruction": "Unikaj powtórzeń", "synonyms": {}}
        burstiness_data = {"target_cv": ">0.40"}
    
    instructions = {
        "burstiness_critical": {
            "instruction": "⚠️ ZRÓŻNICUJ długości zdań! CV musi być > 0.40",
            "why": "Monotonne zdania 15-20 słów = wykrycie AI!",
            "example_lengths": burstiness_data.get("example_sequence", "5, 18, 8, 25, 12, 6, 30, 14 słów"),
            "target_distribution": {
                "short_3_8_words": "20-25%",
                "medium_10_18_words": "50-60%",
                "long_22_35_words": "15-25%"
            }
        },
        "short_sentences_dynamic": {
            "instruction": short_sentences_data.get("instruction", "Wstaw krótkie zdania"),
            "domain": short_sentences_data.get("domain", "universal"),
            "usage": "Twórz 2-3 krótkie zdania z materiału akapitu — NIE kopiuj gotowych fraz"
        },
        "synonyms_dynamic": {
            "instruction": synonyms_data.get("instruction", "NIE POWTARZAJ tych samych słów!"),
            "map": synonyms_data.get("synonyms", {}),
            "warning": "Nie powtarzaj tego samego słowa >3× w batchu!"
        },
        "transition_words_pl": {
            "instruction": "Używaj polskich słów łączących:",
            "kontrast": TRANSITION_WORDS_CATEGORIZED.get("kontrast", [])[:5],
            "przyczyna": TRANSITION_WORDS_CATEGORIZED.get("przyczyna", [])[:5],
            "skutek": TRANSITION_WORDS_CATEGORIZED.get("skutek", [])[:5]
        },
        "avoid_ai_patterns": {
            "instruction": "UNIKAJ tych fraz (typowe dla AI):",
            "patterns": AI_PATTERNS_TO_AVOID[:10]
        }
    }
    
    if is_ymyl:
        instructions["ymyl_precision"] = {
            "instruction": "Treść YMYL - wymagana precyzja!",
            "requirements": ["Cytuj przepisy", "Pisz konkretnie", "Dodaj disclaimer"]
        }
    
    return instructions


# ============================================================================
# CONTINUATION CONTEXT
# ============================================================================

def get_continuation_context(
    batches: List[Dict],
    keywords_state: Dict,
    style_fingerprint: Dict,
    entity_state: Dict
) -> Dict[str, Any]:
    if not batches:
        return {"is_first_batch": True, "instruction": "To jest PIERWSZY batch - wprowadź temat"}
    
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
        defined_entities[ent] = {"status": "zdefiniowane" if definition != "wprowadzone" else "wspomniane", "in_batch": batch_num}
    
    return {
        "is_first_batch": False,
        "last_paragraph": last_paragraph,
        "batches_completed": len(batches),
        "established_entities": defined_entities,
        "instruction_entities": "NIE powtarzaj definicji tych pojęć",
        "continuation_instruction": "KONTYNUUJ narrację płynnie"
    }


# ============================================================================
# KEYWORD TRACKING
# ============================================================================

def get_keyword_tracking_info(
    keywords_state: Dict,
    current_batch_num: int,
    total_batches: int,
    remaining_batches: int
) -> Dict[str, Any]:
    tracking = {
        "mode": "TRACKING",
        "explanation": "Frazy są ŚLEDZONE w tle. Per-batch nie blokuje.",
        "use_naturally": [],
        "available": [],
        "near_limit": [],
        "structural": []
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
            "suggested_this_batch": min(suggested, 3)
        }
        
        if is_structural:
            tracking["structural"].append(kw_info)
        elif remaining_allowed <= 2:
            tracking["near_limit"].append(kw_info)
        elif remaining_needed > 0:
            tracking["use_naturally"].append(kw_info)
        else:
            tracking["available"].append(kw_info)
    
    tracking["summary"] = {
        "total_keywords": len(keywords_state),
        "need_usage": len(tracking["use_naturally"]),
        "near_limit": len(tracking["near_limit"]),
        "instruction": "Użyj fraz NATURALNIE. System śledzi automatycznie."
    }
    
    return tracking


# ============================================================================
# STRUCTURE INSTRUCTIONS
# ============================================================================

def get_structure_instructions(
    current_batch_num: int,
    total_batches: int,
    h2_structure: List[str],
    current_h2: str,
    batch_type: str,
    s1_data: Dict = None,
    target_length: int = None
) -> Dict[str, Any]:
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
    else:
        paragraphs_target = 3
        length_profile = "MEDIUM"
    
    has_h3 = (h2_index == longest_section_index and length_profile == "LONG")
    has_list = (h2_index > longest_section_index and batch_type != "FINAL" and not has_h3)
    
    return {
        "paragraphs_target": paragraphs_target,
        "length_profile": length_profile,
        "has_h3": has_h3,
        "has_list": has_list,
        "section_index": h2_index,
        "total_sections": num_h2
    }


# ============================================================================
# 🆕 v42.0: FAQ INSTRUCTIONS (from overflow_buffer)
# ============================================================================

def get_faq_instructions(
    overflow_buffer: Any,
    batch_type: str
) -> Optional[Dict]:
    """
    🆕 v42.0: Generuje instrukcje dla sekcji FAQ z sieroców.
    """
    if batch_type != "FAQ":
        return None
    
    if not OVERFLOW_BUFFER_AVAILABLE:
        return None
    
    if not overflow_buffer or not overflow_buffer.faq_items:
        return None
    
    return {
        "section_title": overflow_buffer.section_title,
        "faq_count": len(overflow_buffer.faq_items),
        "faq_items": [
            {
                "question": faq.question,
                "target_phrase": faq.target_phrase,
                "phrase_type": faq.phrase_type,
                "answer_hint": faq.answer_template
            }
            for faq in overflow_buffer.faq_items
        ],
        "instruction": format_faq_instructions(overflow_buffer),
        "note": "Każda odpowiedź 2-4 zdania. Fraza MUSI pojawić się w odpowiedzi naturalnie."
    }


# ============================================================================
# 🎯 MAIN FUNCTION
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
    is_medical: bool = False,
    batch_plan: Dict = None,
    # 🆕 v42.0: Dodatkowe parametry
    overflow_buffer: Any = None,
    required_triplets: List[Dict] = None,
    is_sub_batch: bool = False,
    h3_title: str = None,
    assigned_phrases: List[Dict] = None,
    assigned_entities: List[Dict] = None,
    # 🆕 v43.0: Phrase Hierarchy
    phrase_hierarchy_data: Dict = None,
    # 🆕 v44.5: YMYL citation instructions
    legal_instruction: str = "",
    medical_instruction: str = ""
) -> Dict[str, Any]:
    """
    Generuje KOMPLETNE enhanced pre_batch_info z konkretnymi instrukcjami.
    
    🆕 v42.0 ZMIANY:
    - overflow_buffer: FAQ dla sierocych fraz
    - required_triplets: Z semantycznymi instrukcjami
    - is_sub_batch/h3_title: Dla dynamic_sub_batch
    - assigned_phrases/entities: Elementy przypisane do sub-batcha
    """
    if entity_state is None:
        entity_state = {}
    if style_fingerprint is None:
        style_fingerprint = {}
    if batch_plan is None:
        batch_plan = {}
    
    remaining_batches = max(1, total_batches - len(batches))
    
    # H2 dla tego batcha
    batch_h2_from_plan = None
    if batch_plan:
        batches_list = batch_plan.get("batches", [])
        if batches_list and current_batch_num <= len(batches_list):
            current_batch_plan = batches_list[current_batch_num - 1]
            batch_h2_from_plan = current_batch_plan.get("h2_sections", [])
    
    used_h2 = []
    for batch in batches:
        h2_matches = re.findall(r'^h2:\s*(.+)$', batch.get("text", ""), re.MULTILINE | re.IGNORECASE)
        used_h2.extend([h.strip() for h in h2_matches])
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    if batch_h2_from_plan:
        current_h2_list = batch_h2_from_plan
        current_h2 = current_h2_list[0] if current_h2_list else main_keyword
    else:
        if current_batch_num == 1:
            current_h2_list = []
            current_h2 = main_keyword
        else:
            current_h2_list = remaining_h2[:1]
            current_h2 = current_h2_list[0] if current_h2_list else main_keyword
    
    # Batch type
    if current_batch_num == 1:
        batch_type = "INTRO"
    elif current_batch_num >= total_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # 🆕 v42.0: Sprawdź czy to FAQ batch
    if overflow_buffer and overflow_buffer.faq_items and current_batch_num == total_batches:
        batch_type = "FAQ"
    
    previous_batch_text = batches[-1].get("text", "") if batches else None
    overused_words = style_fingerprint.get("overused_words", [])
    remaining_h2_after_current = [h2 for h2 in remaining_h2 if h2 not in current_h2_list]
    
    # ================================================================
    # GENERUJ WSZYSTKIE SEKCJE
    # ================================================================
    
    enhanced = {
        "batch_number": current_batch_num,
        "total_batches": total_batches,
        "batch_type": batch_type,
        "current_h2": current_h2,
        "current_h2_list": current_h2_list,
        "h2_count_in_batch": len(current_h2_list),
        "remaining_h2": remaining_h2_after_current[:4],
        
        # 🆕 v42.0: Sub-batch info
        "is_sub_batch": is_sub_batch,
        "h3_title": h3_title,
        "assigned_phrases": assigned_phrases or [],
        "assigned_entities": assigned_entities or [],
        
        # 1. ENCJE
        "entities_to_define": get_entities_to_define(
            s1_data=s1_data,
            current_batch_num=current_batch_num,
            entity_state=entity_state,
            current_h2=current_h2,
            total_batches=total_batches
        ),
        
        # 2. RELACJE (z semantycznymi instrukcjami)
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
        
        # 4. INSTRUKCJE STYLU
        "style_instructions": get_style_instructions(
            style_fingerprint=style_fingerprint,
            current_batch_num=current_batch_num,
            is_ymyl=is_ymyl,
            main_keyword=main_keyword,
            h2_titles=h2_structure,
            previous_batch_text=previous_batch_text,
            overused_words=overused_words
        ),
        
        # 5. KONTYNUACJA
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
        
        # 7. STRUKTURA
        "structure_instructions": get_structure_instructions(
            current_batch_num=current_batch_num,
            total_batches=total_batches,
            h2_structure=h2_structure,
            current_h2=current_h2,
            batch_type=batch_type,
            s1_data=s1_data
        )
    }
    
    # ================================================================
    # 🆕 v42.0: REQUIRED TRIPLETS (z semantycznymi instrukcjami)
    # ================================================================
    if required_triplets:
        triplets_instructions = []
        for triplet in required_triplets:
            if SEMANTIC_TRIPLET_AVAILABLE:
                instruction = generate_semantic_instruction(triplet)
            else:
                s = triplet.get("subject", "")
                v = triplet.get("verb", "")
                o = triplet.get("object", "")
                instruction = f"Napisz zdanie: {s} {v} {o}"
            
            triplets_instructions.append({
                "triplet": triplet,
                "instruction": instruction,
                "semantic_mode": SEMANTIC_TRIPLET_AVAILABLE
            })
        
        enhanced["required_triplets"] = {
            "count": len(required_triplets),
            "triplets": triplets_instructions,
            "note": "🆕 Triplety walidowane SEMANTYCZNIE - akceptowane są warianty językowe!"
        }
    
    # ================================================================
    # 🆕 v42.0: FAQ INSTRUCTIONS (from overflow_buffer)
    # ================================================================
    if batch_type == "FAQ" and overflow_buffer:
        enhanced["faq_instructions"] = get_faq_instructions(overflow_buffer, batch_type)
    
    # ================================================================
    # 🆕 v42.0: SUB-BATCH SPECIFIC INSTRUCTIONS
    # ================================================================
    if is_sub_batch:
        enhanced["sub_batch_instructions"] = {
            "note": "To jest SUB-BATCH - mniejsza część większej sekcji",
            "h3_title": h3_title,
            "focus": "Skup się TYLKO na przypisanych elementach",
            "assigned_phrases_count": len(assigned_phrases or []),
            "assigned_entities_count": len(assigned_entities or [])
        }
    
    # ================================================================
    # 🆕 v43.0: PHRASE HIERARCHY CONTEXT
    # Dodaje informacje o hierarchii fraz do kontekstu batcha
    # ================================================================
    hierarchy_context = None
    
    if phrase_hierarchy_data and PHRASE_HIERARCHY_AVAILABLE:
        try:
            hierarchy = dict_to_hierarchy(phrase_hierarchy_data)
            
            if hierarchy:
                # Zbierz frazy przypisane do tego batcha
                batch_phrases = []
                
                # Z assigned_phrases (jeśli sub-batch)
                if assigned_phrases:
                    batch_phrases = [p.get("keyword", "") for p in assigned_phrases if p.get("keyword")]
                
                # Z keywords_state - frazy które jeszcze nie osiągnęły limitu
                if not batch_phrases:
                    for rid, meta in keywords_state.items():
                        actual = meta.get("actual_uses", 0)
                        target_max = meta.get("target_max", 999)
                        if actual < target_max:
                            kw = meta.get("keyword", "")
                            if kw:
                                batch_phrases.append(kw)
                
                # Pobierz aktualne zliczenia
                current_counts = {
                    meta.get("keyword", ""): meta.get("actual_uses", 0)
                    for rid, meta in keywords_state.items()
                    if meta.get("keyword")
                }
                
                # Pobierz kontekst hierarchii dla tego batcha
                hierarchy_context = get_batch_hierarchy_context(
                    hierarchy=hierarchy,
                    batch_phrases=batch_phrases[:30],  # Max 30 fraz
                    current_counts=current_counts
                )
                
                print(f"[ENHANCED_PRE_BATCH] ✅ Hierarchy context: "
                      f"{len(hierarchy_context.get('roots_covered', []))} roots, "
                      f"{len(hierarchy_context.get('entity_phrases', []))} entity phrases")
                
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Hierarchy context error: {e}")
            hierarchy_context = None
    
    # Dodaj do enhanced
    enhanced["phrase_hierarchy"] = hierarchy_context
    
    # ================================================================
    # 🆕 v45.0: CAUSAL TRIPLETS + CONTENT GAPS CONTEXT
    # ================================================================
    # Przekaż agent_instruction z S1 do agenta GPT
    causal_triplets = s1_data.get("causal_triplets", {}) if s1_data else {}
    if causal_triplets and isinstance(causal_triplets, dict) and causal_triplets.get("agent_instruction"):
        enhanced["causal_context"] = causal_triplets["agent_instruction"]
    
    content_gaps = s1_data.get("content_gaps", {}) if s1_data else {}
    if content_gaps and isinstance(content_gaps, dict) and content_gaps.get("agent_instruction"):
        enhanced["information_gain"] = content_gaps["agent_instruction"]
    
    # GPT PROMPT SECTION
    enhanced["gpt_instructions"] = _generate_gpt_prompt_section(
        enhanced, is_legal=is_legal, is_medical=is_medical,
        legal_instruction=legal_instruction, medical_instruction=medical_instruction
    )
    
    # SMART INSTRUCTIONS
    if SMART_INSTRUCTIONS_AVAILABLE:
        try:
            already_covered = [meta.get("keyword", "") for rid, meta in keywords_state.items() if meta.get("actual_uses", 0) >= meta.get("target_min", 1)]
            domain = "prawo" if is_legal else "general"
            
            smart = generate_smart_batch_instructions(
                keywords_state=keywords_state,
                s1_data=s1_data,
                current_batch_num=current_batch_num,
                total_batches=total_batches,
                current_h2=enhanced.get("current_h2_list", []),
                batch_type=batch_type,
                already_well_covered=already_covered,
                domain=domain
            )
            
            enhanced["smart_instructions"] = smart
            enhanced["smart_instructions_formatted"] = format_instructions_for_gpt(smart)
        except Exception as e:
            print(f"[ENHANCED_PRE_BATCH] ⚠️ Smart instructions error: {e}")
    
    return enhanced


def _generate_gpt_prompt_section(
    enhanced: Dict, 
    is_legal: bool = False, 
    is_medical: bool = False,
    legal_instruction: str = "",
    medical_instruction: str = ""
) -> str:
    """Generuje gotową sekcję promptu dla GPT."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"📋 BATCH #{enhanced['batch_number']} - {enhanced['batch_type']}")
    lines.append("=" * 60)
    lines.append("")
    
    # ================================================================
    # 🆕 v44.6: YMYL CITATION INSTRUCTIONS - kompaktowe
    # ================================================================
    if is_legal and legal_instruction:
        lines.append(legal_instruction[:800])
        lines.append("Wplataj cytowania NATURALNIE w tekst, min 1-2 na sekcję H2.")
        lines.append("")
    
    if is_medical and medical_instruction:
        lines.append(medical_instruction[:800])
        lines.append("Min 1-2 cytowania na sekcję H2. NIE wymyślaj statystyk!")
        lines.append("")
    
    # 🆕 v42.0: Info o sub-batch
    if enhanced.get("is_sub_batch"):
        lines.append(f"🔀 SUB-BATCH: {enhanced.get('h3_title', 'Część')}")
        lines.append(f"   Przypisane frazy: {len(enhanced.get('assigned_phrases', []))}")
        lines.append("")
    
    h2_list = enhanced.get("current_h2_list", [])
    h2_count = enhanced.get("h2_count_in_batch", 0)
    
    paragraph_options = [2, 3, 4, 3, 2, 4]
    
    if h2_count > 1:
        lines.append(f"📌 H2 W TYM BATCHU ({h2_count} sekcje):")
        for i, h2 in enumerate(h2_list, 1):
            para_count = paragraph_options[(i - 1) % len(paragraph_options)]
            lines.append(f"   {i}. \"{h2}\" → {para_count} akapity")
        lines.append("")
    elif h2_count == 1:
        lines.append(f"📌 H2: \"{enhanced['current_h2']}\"")
    
    # ================================================================
    # 🆕 v43.0: PHRASE HIERARCHY TIPS
    # ================================================================
    hierarchy = enhanced.get("phrase_hierarchy")
    if hierarchy:
        lines.append("")
        lines.append("🌳 HIERARCHIA FRAZ - KLUCZOWE!")
        lines.append("=" * 40)
        
        # Writing tips
        tips = hierarchy.get("writing_tips", [])
        for tip in tips[:5]:
            lines.append(f"   {tip}")
        
        # Roots covered info
        roots_covered = hierarchy.get("roots_covered", [])
        if roots_covered:
            lines.append("")
            lines.append("   📊 RDZENIE W TYM BATCHU:")
            for root_info in roots_covered[:4]:
                root = root_info.get("root", "")
                strategy = root_info.get("strategy", "")
                eff_target = root_info.get("effective_target", "")
                extensions_in_batch = root_info.get("extensions_in_batch", 0)
                
                if strategy == "extensions_sufficient":
                    lines.append(f"   ✅ \"{root}\" → NIE POWTARZAJ osobno!")
                    lines.append(f"      (masz {extensions_in_batch} rozszerzeń w tym batchu)")
                elif strategy == "mixed":
                    lines.append(f"   ⚖️ \"{root}\" → użyj {eff_target} samodzielnie")
                    lines.append(f"      + {extensions_in_batch} rozszerzeń")
                else:
                    lines.append(f"   📌 \"{root}\" → użyj {eff_target}")
        
        # Entity phrases
        entity_phrases = hierarchy.get("entity_phrases", [])
        if entity_phrases:
            lines.append("")
            lines.append("   🔵 FRAZY = ENCJE (PRIORYTET!):")
            for ep in entity_phrases[:5]:
                lines.append(f"   • \"{ep}\" ← buduje topical authority")
        
        # Triplet phrases
        triplet_phrases = hierarchy.get("triplet_phrases", [])
        if triplet_phrases:
            lines.append("")
            lines.append("   🔺 FRAZY Z TRIPLETÓW (bonus SEO):")
            for tp in triplet_phrases[:4]:
                lines.append(f"   • \"{tp}\"")
        
        lines.append("")
        lines.append("=" * 40)
    
    # 🆕 v42.0: FAQ instructions
    if enhanced.get("batch_type") == "FAQ" and enhanced.get("faq_instructions"):
        faq = enhanced["faq_instructions"]
        lines.append("📦 SEKCJA FAQ:")
        for item in faq.get("faq_items", [])[:5]:
            lines.append(f"   ❓ {item['question']}")
            lines.append(f"      → Fraza MUST: \"{item['target_phrase']}\"")
        lines.append("")
    
    # 🆕 v42.0: Required triplets with semantic instructions
    if enhanced.get("required_triplets"):
        triplets_info = enhanced["required_triplets"]
        lines.append("🔗 TRIPLETY DO WYRAŻENIA:")
        lines.append(f"   ℹ️ {triplets_info.get('note', '')}")
        for t in triplets_info.get("triplets", [])[:3]:
            triplet = t["triplet"]
            lines.append(f"   • {triplet.get('subject', '')} → {triplet.get('verb', '')} → {triplet.get('object', '')}")
        lines.append("")
    
    # Entities
    entities = enhanced.get("entities_to_define", [])
    if entities:
        lines.append("🧠 ENCJE:")
        for ent in entities[:3]:
            priority_icon = "🔴" if ent.get("priority") == "MUST" else "🟡"
            lines.append(f"   {priority_icon} {ent['entity']}: {ent['how'][:50]}...")
        lines.append("")
    
    # ================================================================
    # 🆕 v43.1: PEŁNE INSTRUKCJE HUMANIZACJI
    # ================================================================
    style = enhanced.get("style_instructions", {})
    
    # BURSTINESS - KLUCZOWE!
    burstiness = style.get("burstiness_critical", {})
    if burstiness:
        lines.append("⚡ BURSTINESS (KLUCZOWE!):")
        lines.append("=" * 40)
        lines.append("   ❌ MONOTONNE ZDANIA = WYKRYCIE AI!")
        lines.append("   ✅ ZRÓŻNICUJ długości zdań!")
        lines.append("")
        lines.append("   📊 WYMAGANY ROZKŁAD:")
        target_dist = burstiness.get("target_distribution", {})
        lines.append(f"   • Krótkie (3-8 słów):  20-25%")
        lines.append(f"   • Średnie (10-18 słów): 50-60%")
        lines.append(f"   • Długie (22-35 słów):  15-25%")
        lines.append("")
        example = burstiness.get("example_lengths", "5, 18, 8, 25, 12, 6, 30, 14 słów")
        lines.append(f"   📝 Przykład sekwencji: {example}")
        lines.append("")
    
    # ================================================================
    # 🆕 v43.1: STRUKTURA - DYNAMICZNA LICZBA AKAPITÓW
    # ================================================================
    structure = enhanced.get("structure_instructions", {})
    batch_num = enhanced.get("batch_number", 1)
    batch_type = enhanced.get("batch_type", "CONTENT")
    
    # Różna liczba akapitów dla różnych batchy (cyklicznie)
    paragraph_patterns = [2, 4, 3, 5, 2, 4, 3, 2, 4]
    target_paragraphs = paragraph_patterns[(batch_num - 1) % len(paragraph_patterns)]
    
    # Dla INTRO zawsze 2-3, dla FINAL 2-3
    if batch_type == "INTRO":
        target_paragraphs = 2
    elif batch_type == "FINAL":
        target_paragraphs = 3
    elif batch_type == "FAQ":
        target_paragraphs = 0  # FAQ ma inną strukturę
    
    # Różna długość akapitów (pattern dla tego batcha)
    length_patterns = [
        ["krótki (50 słów)", "długi (120 słów)"],  # 2 akapity
        ["średni (80 słów)", "krótki (50 słów)", "długi (130 słów)"],  # 3 akapity
        ["krótki (50 słów)", "średni (90 słów)", "długi (120 słów)", "krótki (60 słów)"],  # 4 akapity
        ["średni (70 słów)", "krótki (40 słów)", "długi (140 słów)", "średni (80 słów)", "krótki (50 słów)"],  # 5 akapitów
    ]
    
    if batch_type not in ["FAQ"]:
        lines.append("📐 STRUKTURA TEJ SEKCJI (KONKRETNE!):")
        lines.append("=" * 40)
        lines.append(f"   🎯 LICZBA AKAPITÓW: **{target_paragraphs}**")
        lines.append("")
        
        if target_paragraphs >= 2 and target_paragraphs <= 5:
            pattern_idx = target_paragraphs - 2
            if pattern_idx < len(length_patterns):
                lengths = length_patterns[pattern_idx]
                lines.append("   📏 DŁUGOŚĆ KAŻDEGO AKAPITU:")
                for i, length in enumerate(lengths, 1):
                    lines.append(f"      Akapit {i}: {length}")
        lines.append("")
        lines.append("   ⚠️ NIE PISZ wszystkich akapitów tej samej długości!")
        lines.append("   ⚠️ NIE PISZ zawsze 2 lub 3 akapitów w każdej sekcji!")
        lines.append("")
    
    # 🆕 v43.1: LISTY I WYPUNKTOWANIA
    has_list = structure.get("has_list", False)
    has_h3 = structure.get("has_h3", False)
    
    if batch_type not in ["INTRO", "FAQ"]:
        lines.append("📋 LISTY I WYPUNKTOWANIA:")
        if has_list:
            lines.append("   ✅ W TEJ SEKCJI UŻYJ LISTY WYPUNKTOWANEJ!")
            lines.append("   • Lista 3-7 punktów")
            lines.append("   • Każdy punkt 1-2 zdania")
            lines.append("   • Poprzedź listę zdaniem wprowadzającym")
            lines.append("")
            lines.append("   📝 Przykład:")
            lines.append("   \"Główne objawy to:\"")
            lines.append("   • punkt pierwszy,")
            lines.append("   • punkt drugi,")
            lines.append("   • punkt trzeci.")
        else:
            lines.append("   ℹ️ Ta sekcja BEZ listy (użyj w innych sekcjach)")
            lines.append("   💡 W artykule powinno być 1-2 listy wypunktowane")
            lines.append("   💡 Najlepiej w sekcjach: objawy, przyczyny, zalecenia")
        lines.append("")
    
    # H3 subheadingi - TYLKO gdy has_h3=true!
    if has_h3:
        lines.append("📑 H3 SUBHEADINGI:")
        lines.append("   ✅ W TEJ sekcji DODAJ 1-2 nagłówki H3!")
        lines.append("   • H3 dzieli długą sekcję na części")
        lines.append("   • H3 powinien zawierać słowo kluczowe")
        lines.append("   • Format: h3: Tytuł podsekcji")
        lines.append("")
    else:
        # WYRAŹNY ZAKAZ H3 gdy nie jest wymagany
        lines.append("📑 H3 SUBHEADINGI:")
        lines.append("   ❌ W TEJ sekcji NIE UŻYWAJ H3!")
        lines.append("   ℹ️ H3 tylko w najdłuższej sekcji artykułu")
        lines.append("   ℹ️ Ta sekcja: same akapity, bez subheadingów")
        lines.append("")
    
    # KRÓTKIE ZDANIA - kontekstowa instrukcja (v41.0)
    short_sentences = style.get("short_sentences_dynamic", {})
    short_instruction = short_sentences.get("instruction", "")
    if short_instruction:
        lines.append(short_instruction)
        lines.append("")
    
    # AI PATTERNS DO UNIKANIA
    avoid_patterns = style.get("avoid_ai_patterns", {})
    patterns = avoid_patterns.get("patterns", [])
    if patterns:
        lines.append("🚫 UNIKAJ FRAZ AI:")
        if isinstance(patterns, dict):
            for pattern, replacement in list(patterns.items())[:5]:
                lines.append(f"   ❌ \"{pattern}\" {replacement}")
        elif isinstance(patterns, list):
            for pattern in patterns[:5]:
                lines.append(f"   ❌ \"{pattern}\"")
        lines.append("")
    
    # SŁOWA ŁĄCZĄCE
    transitions = style.get("transition_words_pl", {})
    if transitions:
        lines.append("🔗 SŁOWA ŁĄCZĄCE (używaj!):")
        if transitions.get("kontrast"):
            lines.append(f"   Kontrast: {', '.join(transitions['kontrast'][:4])}")
        if transitions.get("przyczyna"):
            lines.append(f"   Przyczyna: {', '.join(transitions['przyczyna'][:4])}")
        if transitions.get("skutek"):
            lines.append(f"   Skutek: {', '.join(transitions['skutek'][:4])}")
        lines.append("")
    
    # 🆕 v45.0: KAUZALNE RELACJE (z S1)
    causal_context = enhanced.get("causal_context", "")
    if causal_context:
        lines.append(causal_context)
        lines.append("")
    
    # 🆕 v45.0: INFORMATION GAIN / CONTENT GAPS (z S1)
    gain_context = enhanced.get("information_gain", "")
    if gain_context:
        lines.append(gain_context)
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
    'get_faq_instructions',
    'CONFIG',
    'OVERFLOW_BUFFER_AVAILABLE',
    'SEMANTIC_TRIPLET_AVAILABLE',
    'PHRASE_HIERARCHY_AVAILABLE'
]
