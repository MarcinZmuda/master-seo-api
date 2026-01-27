"""
===============================================================================
🎯 ENHANCED PRE-BATCH INSTRUCTIONS v39.0
===============================================================================
Moduł generujący KONKRETNE instrukcje dla GPT zamiast surowych danych.

ROZWIĄZUJE PROBLEMY:
1. Encje/Triplety - zamiast listy → konkretne "jak zdefiniować"
2. Keywords - tracking w tle, nie blokowanie per-batch
3. Humanizacja - konkretne instrukcje stylu
4. Kontynuacja - pełny kontekst poprzedniego batcha

Autor: BRAJEN SEO Master API v39.0
===============================================================================
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


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
    
    Args:
        entity: Nazwa encji
        context: Kontekst z S1 (jeśli dostępny)
        h2: Nagłówek H2 tej sekcji
        
    Returns:
        Dict z instrukcją definicji
    """
    entity_type = classify_entity_type(entity)
    templates = DEFINITION_TEMPLATES.get(entity_type, DEFINITION_TEMPLATES["default"])
    
    # Wybierz template
    template = templates[0]
    how = template.format(entity=entity)
    
    # Dodaj kontekst jeśli dostępny
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
    
    Args:
        s1_data: Dane z analizy S1
        current_batch_num: Numer aktualnego batcha
        entity_state: Stan encji (które już zdefiniowane)
        current_h2: Nagłówek H2 tego batcha
        total_batches: Łączna liczba batchów
        
    Returns:
        Lista encji z instrukcjami definicji
    """
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    # Filtruj encje które jeszcze nie zostały zdefiniowane
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
                # Sprawdź czy nie ma już w wynikach
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
    
    # Batch 1 = INTRO, weź więcej encji fundamentalnych
    if current_batch_num == 1:
        # Sortuj by MUST były pierwsze
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
    """
    Generuje instrukcję ustanowienia relacji między encjami.
    """
    # Znajdź odpowiedni template
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
    """
    Zwraca relacje do ustanowienia w tym batchu.
    """
    entity_seo = s1_data.get("entity_seo", {})
    relationships = entity_seo.get("entity_relationships", [])
    
    # Filtruj już ustanowione
    established = set(entity_state.get("relations_established", []))
    
    result = []
    for rel in relationships:
        from_ent = rel.get("from", rel.get("subject", ""))
        to_ent = rel.get("to", rel.get("object", ""))
        relation = rel.get("relation", rel.get("predicate", ""))
        
        if not from_ent or not to_ent or not relation:
            continue
        
        # Utwórz klucz relacji
        rel_key = f"{from_ent}|{relation}|{to_ent}".lower()
        if rel_key not in established:
            instruction = generate_relation_instruction(from_ent, relation, to_ent)
            instruction["priority"] = rel.get("priority", "SHOULD")
            result.append(instruction)
    
    # Rozłóż na batche
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
    """
    Generuje semantic context z terminami do użycia.
    
    Zawiera:
    - context_terms: terminy które MUSZĄ pojawić się w tekście
    - supporting_phrases: frazy wzbogacające
    - semantic_field: pole semantyczne tematu
    """
    # N-gramy
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.4]
    
    # LSI keywords
    semantic_keyphrases = s1_data.get("semantic_keyphrases", [])
    lsi_keywords = [kp.get("phrase", "") for kp in semantic_keyphrases if kp.get("score", 0) > 0.6]
    
    # Related searches
    serp = s1_data.get("serp_analysis", {})
    related = serp.get("related_searches", [])[:5]
    
    # Wybierz terminy dla tego batcha
    all_terms = top_ngrams + lsi_keywords
    terms_per_batch = max(CONFIG.MIN_CONTEXT_TERMS, len(all_terms) // 8)
    
    start_idx = (current_batch_num - 1) * terms_per_batch
    end_idx = min(start_idx + terms_per_batch + 2, len(all_terms))
    
    batch_terms = all_terms[start_idx:end_idx][:CONFIG.MAX_CONTEXT_TERMS]
    
    # Filtruj - usuń terminy które są w keywords_state (będą osobno trackowane)
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
# 4. STYLE INSTRUCTIONS - humanizacja tekstu
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
    "dodatkowo",  # Tylko na początku zdania
    "warto zaznaczyć",
    "należy podkreślić",
    "trzeba wspomnieć"
]

# Naturalne alternatywy
NATURAL_ALTERNATIVES = {
    "warto podkreślić": ["", "Zwróć uwagę:", "Ważne:"],
    "należy pamiętać": ["Pamiętaj,", "Nie zapomnij,", ""],
    "w kontekście": ["przy", "jeśli chodzi o", "w sprawie"],
    "co więcej": ["Poza tym", "Również", "A co ważne"],
    "ponadto": ["Oprócz tego", "Też", "Również"],
}


def get_style_instructions(
    style_fingerprint: Dict,
    current_batch_num: int,
    is_ymyl: bool = False
) -> Dict[str, Any]:
    """
    Generuje konkretne instrukcje stylistyczne dla GPT.
    """
    # Bazowe instrukcje
    instructions = {
        "vary_sentence_length": {
            "instruction": "Mieszaj długości zdań: 5-40 słów",
            "target_distribution": {
                "short_2_10_words": "20-25%",
                "medium_12_18_words": "50-60%",
                "long_20_35_words": "15-25%"
            },
            "avoid": "Nie pisz wszystkich zdań 15-22 słów (wzorzec AI)"
        },
        
        "avoid_ai_patterns": {
            "instruction": "UNIKAJ tych fraz (typowe dla AI):",
            "patterns": AI_PATTERNS_TO_AVOID[:10],
            "alternatives": NATURAL_ALTERNATIVES
        },
        
        "use_active_voice": {
            "instruction": "Preferuj stronę czynną",
            "examples": {
                "bad": "Wniosek jest składany do sądu",
                "good": "Wniosek składa się do sądu"
            }
        },
        
        "pronouns_consistency": {
            "instruction": "Wybierz JEDEN styl i trzymaj się go",
            "options": ["bezosobowo (można, należy)", "per 'ty' (możesz, powinieneś)"],
            "warning": "NIE mieszaj stylów w jednym tekście!"
        },
        
        "natural_flow": {
            "instruction": "Pisz jak ekspert tłumaczący znajomemu, nie jak encyklopedia",
            "tips": [
                "Używaj pytań retorycznych",
                "Dodaj przykłady z życia",
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
    """
    Generuje pełny kontekst kontynuacji dla GPT.
    """
    if not batches:
        return {
            "is_first_batch": True,
            "instruction": "To jest PIERWSZY batch - wprowadź temat"
        }
    
    last_batch = batches[-1]
    last_text = last_batch.get("text", "")
    
    # Wyciągnij ostatni pełny akapit (nie tylko 2 zdania)
    paragraphs = re.split(r'\n\n+', last_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and not p.startswith("h2:")]
    
    last_paragraph = ""
    if paragraphs:
        last_paragraph = paragraphs[-1]
        # Ogranicz długość
        words = last_paragraph.split()
        if len(words) > CONFIG.LAST_PARAGRAPH_WORDS:
            last_paragraph = " ".join(words[-CONFIG.LAST_PARAGRAPH_WORDS:])
    
    # Zbierz zdefiniowane encje
    defined_entities = {}
    for ent, batch_num in entity_state.get("introduced_entities", {}).items():
        definition = entity_state.get("defined_terms", {}).get(ent, "wprowadzone")
        defined_entities[ent] = {
            "status": "zdefiniowane" if definition != "wprowadzone" else "wspomniane",
            "in_batch": batch_num
        }
    
    # Ostatnie H2
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
# 6. KEYWORD TRACKING MODE - tracking zamiast blokowania
# ============================================================================

def get_keyword_tracking_info(
    keywords_state: Dict,
    current_batch_num: int,
    total_batches: int,
    remaining_batches: int
) -> Dict[str, Any]:
    """
    Generuje informacje o keywords w trybie TRACKING (nie blokującym).
    
    Per-batch: tylko INFO/WARNING, nigdy STOP
    Final review: weryfikacja globalna
    """
    tracking = {
        "mode": "TRACKING",
        "explanation": "Frazy są ŚLEDZONE w tle. Per-batch nie blokuje. Weryfikacja globalna w final_review.",
        
        "use_naturally": [],      # Użyj naturalnie
        "available": [],          # Dostępne, ale nie wymagane
        "near_limit": [],         # Blisko limitu - uważaj
        "structural": [],         # STRUCTURAL - bez limitu per-batch
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
        
        # Oblicz ile jeszcze potrzeba
        remaining_needed = max(0, target_min - actual)
        remaining_allowed = max(0, target_max - actual)
        
        # Suggested per batch
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
        
        # Kategoryzuj
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
    
    # Podsumowanie dla GPT
    tracking["summary"] = {
        "total_keywords": len(keywords_state),
        "need_usage": len(tracking["use_naturally"]),
        "near_limit": len(tracking["near_limit"]),
        "structural": len(tracking["structural"]),
        "instruction": "Użyj fraz NATURALNIE. System śledzi ilości automatycznie. Nie rób stuffingu!"
    }
    
    return tracking


# ============================================================================
# 7. DYNAMIC BATCH COUNT - na podstawie S1
# ============================================================================

def calculate_optimal_batch_count(
    s1_data: Dict,
    keywords_count: int,
    h2_count: int,
    target_length: int,
    is_ymyl: bool = False
) -> Dict[str, Any]:
    """
    Oblicza optymalną liczbę batchów na podstawie analizy S1.
    
    Faktory:
    - Liczba encji do zdefiniowania
    - Liczba relacji do ustanowienia
    - Liczba keywords
    - Długość docelowa
    - YMYL wymaga więcej szczegółów
    """
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    relationships = entity_seo.get("entity_relationships", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    # Policz encje HIGH importance
    high_entities = len([e for e in entities if e.get("importance", 0) >= 0.7])
    must_topics = len([t for t in topical_coverage if t.get("priority") == "MUST"])
    
    # Bazowa liczba batchów
    base_batches = h2_count + 1  # H2 + intro
    
    # Dodatkowe batche na podstawie złożoności
    complexity_batches = 0
    
    # Dużo encji = więcej batchów
    if high_entities > 8:
        complexity_batches += 2
    elif high_entities > 5:
        complexity_batches += 1
    
    # Dużo relacji = więcej batchów
    if len(relationships) > 6:
        complexity_batches += 1
    
    # Dużo keywords = więcej batchów
    if keywords_count > 25:
        complexity_batches += 2
    elif keywords_count > 15:
        complexity_batches += 1
    
    # YMYL = więcej szczegółów
    if is_ymyl:
        complexity_batches += 1
    
    # Długi artykuł = więcej batchów
    if target_length > 3500:
        complexity_batches += 2
    elif target_length > 2500:
        complexity_batches += 1
    
    optimal = base_batches + complexity_batches
    
    # Limity
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
# 8. MAIN FUNCTION - generuje kompletne enhanced pre_batch_info
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
    is_legal: bool = False
) -> Dict[str, Any]:
    """
    Generuje KOMPLETNE enhanced pre_batch_info z konkretnymi instrukcjami.
    
    Returns:
        Dict gotowy do wysłania do GPT
    """
    if entity_state is None:
        entity_state = {}
    if style_fingerprint is None:
        style_fingerprint = {}
    
    remaining_batches = max(1, total_batches - len(batches))
    
    # Określ H2 dla tego batcha
    used_h2 = []
    for batch in batches:
        h2_match = re.search(r'^h2:\s*(.+)$', batch.get("text", ""), re.MULTILINE | re.IGNORECASE)
        if h2_match:
            used_h2.append(h2_match.group(1).strip())
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    current_h2 = remaining_h2[0] if remaining_h2 else main_keyword
    
    # Batch type
    if current_batch_num == 1:
        batch_type = "INTRO"
    elif current_batch_num >= total_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # ================================================================
    # GENERUJ WSZYSTKIE SEKCJE
    # ================================================================
    
    enhanced = {
        "batch_number": current_batch_num,
        "total_batches": total_batches,
        "batch_type": batch_type,
        "current_h2": current_h2,
        "remaining_h2": remaining_h2[1:4],  # Następne 3 H2
        
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
        
        # 4. INSTRUKCJE STYLU
        "style_instructions": get_style_instructions(
            style_fingerprint=style_fingerprint,
            current_batch_num=current_batch_num,
            is_ymyl=is_ymyl
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
        )
    }
    
    # ================================================================
    # GPT PROMPT SECTION - gotowy do wklejenia
    # ================================================================
    
    enhanced["gpt_instructions"] = _generate_gpt_prompt_section(enhanced, is_legal)
    
    return enhanced


def _generate_gpt_prompt_section(enhanced: Dict, is_legal: bool = False) -> str:
    """
    Generuje gotową sekcję promptu dla GPT.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"📋 BATCH #{enhanced['batch_number']} - {enhanced['batch_type']}")
    lines.append("=" * 60)
    lines.append("")
    
    # H2
    lines.append(f"📌 H2: \"{enhanced['current_h2']}\"")
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
    
    # Styl
    style = enhanced.get("style_instructions", {})
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
    'calculate_optimal_batch_count',
    'CONFIG',
    'AI_PATTERNS_TO_AVOID',
    'NATURAL_ALTERNATIVES'
]
