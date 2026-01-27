"""
===============================================================================
🎯 ENTITY COVERAGE VALIDATOR v38.3
===============================================================================
Waliduje pokrycie encji z S1 w batchach.

ZMIANY v38.3:
- 🆕 SEMANTIC FALLBACK: gdy regex nie znajdzie, próbuje keyword proximity
- 🔧 DRIFT: procedural ↔ institutional dozwolone (nie jest driftem)
- 🔧 DRIFT: tylko positive ↔ negative blokuje (CRITICAL)

FUNKCJE:
- Sprawdza czy encje HIGH importance zostały wprowadzone/zdefiniowane
- Wykrywa brak definicji kluczowych encji (regex + semantic fallback)
- Waliduje relacje między encjami
- Wykrywa entity drift (zmiana definicji między batchami)
- Aktualizuje entity_state po każdym batchu

INTEGRACJA:
- Wywoływany w batch_simple po MoE validation
- Aktualizuje entity_state w Firestore
- Może blokować batch jeśli brak MUST entity
===============================================================================
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class EntityAction(Enum):
    """Wymagane akcje dla encji."""
    INTRODUCE = "INTRODUCE"  # Pierwsza wzmianka
    DEFINE = "DEFINE"        # Definicja (X to/jest...)
    EXPLAIN = "EXPLAIN"      # Wyjaśnienie (cel, skutki)
    MENTION = "MENTION"      # Tylko wzmianka
    ESTABLISH_RELATION = "ESTABLISH_RELATION"  # Ustanów relację


class EntityPriority(Enum):
    """Priorytet wymagania encji."""
    MUST = "MUST"      # Blokuje jeśli brak
    SHOULD = "SHOULD"  # Warning jeśli brak
    CAN = "CAN"        # Opcjonalne


@dataclass
class EntityValidationResult:
    """Wynik walidacji pojedynczej encji w batchu."""
    entity: str
    expected_action: str
    priority: str
    actual_status: str  # FOUND, NOT_FOUND, PARTIALLY_FOUND
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "expected_action": self.expected_action,
            "priority": self.priority,
            "actual_status": self.actual_status,
            "is_valid": self.is_valid,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "patterns_matched": self.patterns_matched
        }


@dataclass 
class EntityCoverageResult:
    """Wynik walidacji pokrycia encji w batchu."""
    status: str  # PASS, WARNING, FAIL
    score: int   # 0-100
    results: List[EntityValidationResult]
    must_missing: List[str]
    should_missing: List[str]
    relationships_established: List[dict]
    relationships_missing: List[dict]
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": self.score,
            "results": [r.to_dict() for r in self.results],
            "must_missing": self.must_missing,
            "should_missing": self.should_missing,
            "relationships_established": self.relationships_established,
            "relationships_missing": self.relationships_missing
        }


class EntityCoverageExpert:
    """
    Expert MoE do walidacji pokrycia encji w batchu.
    
    Sprawdza czy:
    - Encje HIGH importance są wprowadzane we właściwych batchach
    - Definicje encji są kompletne
    - Relacje między encjami są ustanawiane
    """
    
    # ================================================================
    # WZORCE DEFINICJI
    # ================================================================
    DEFINITION_PATTERNS = [
        # "X jest/to Y"
        r"{entity}\s+(?:jest|to|stanowi|oznacza)\s+[^.]+",
        # "jest X" (odwrócone)
        r"(?:jest|to|stanowi)\s+[^.]*{entity}",
        # "przez X rozumie się"
        r"(?:przez|pod pojęciem)\s+{entity}\s+(?:rozumie się|należy rozumieć)",
        # "X - to/czyli"
        r"{entity}\s*[-–—]\s*(?:to|czyli|tj\.?)",
        # "X polega na"
        r"{entity}\s+polega\s+na\s+[^.]+",
        # "istota X"
        r"istot[aąę]\s+{entity}\s+[^.]+",
        # "definicja X"
        r"definicj[aąę]\s+{entity}",
    ]
    
    # ================================================================
    # WZORCE WYJAŚNIENIA (cel, skutki, konsekwencje)
    # ================================================================
    EXPLANATION_PATTERNS = [
        # "X służy/ma na celu"
        r"{entity}\s+(?:służy|ma\s+na\s+celu|umożliwia|pozwala)",
        # "celem X jest"
        r"cel(?:em)?\s+{entity}\s+(?:jest|było)",
        # "X prowadzi do/skutkuje"
        r"{entity}\s+(?:prowadzi\s+do|skutkuje|powoduje|wywołuje)",
        # "skutkiem X"
        r"(?:skutkiem|konsekwencją|następstwem)\s+{entity}",
        # "dzięki X"
        r"(?:dzięki|poprzez|w\s+wyniku|na\s+skutek)\s+{entity}",
        # "X chroni/zabezpiecza"
        r"{entity}\s+(?:chroni|zabezpiecza|gwarantuje|zapewnia)",
    ]
    
    # ================================================================
    # WZORCE WPROWADZENIA (pierwsza wzmianka w kontekście)
    # ================================================================
    INTRODUCTION_PATTERNS = [
        # "istnieje X"
        r"(?:istnieje|występuje|funkcjonuje)\s+[^.]*{entity}",
        # "X może/powinien"
        r"{entity}\s+(?:może|powinien|musi|wymaga)",
        # "w przypadku X"
        r"(?:w\s+przypadku|w\s+sytuacji|przy)\s+{entity}",
        # "instytucja X"
        r"instytucj[aąę]\s+{entity}",
        # "procedura X"
        r"procedur[aąę]\s+{entity}",
        # "kwestia X"
        r"kwesti[aąę]\s+{entity}",
    ]
    
    # ================================================================
    # WZORCE RELACJI
    # ================================================================
    RELATION_PATTERNS = {
        "orzeka": [
            r"{subject}\s+(?:orzeka|wydaje\s+orzeczenie|rozstrzyga)\s+[^.]*{object}",
            r"{object}\s+(?:jest\s+)?(?:orzekane|wydawane)\s+przez\s+{subject}",
        ],
        "może_prowadzić_do": [
            r"{subject}\s+(?:może\s+)?(?:prowadzi|doprowadza)\s+do\s+{object}",
            r"{object}\s+(?:wynika|następuje)\s+[^.]*{subject}",
        ],
        "reprezentuje": [
            r"{subject}\s+(?:reprezentuje|działa\s+w\s+imieniu)\s+[^.]*{object}",
            r"{object}\s+(?:jest\s+)?reprezentowan[yaei]\s+przez\s+{subject}",
        ],
        "wymaga": [
            r"{subject}\s+wymaga\s+{object}",
            r"do\s+{subject}\s+(?:potrzebne|konieczne|niezbędne)\s+[^.]*{object}",
        ],
        "skutkuje": [
            r"{subject}\s+skutkuje\s+{object}",
            r"{object}\s+(?:jest\s+)?(?:skutkiem|konsekwencją)\s+{subject}",
        ],
    }
    
    def __init__(self):
        self.compiled_patterns = {}
    
    def validate_batch(
        self,
        batch_text: str,
        entity_requirements: List[dict],
        entity_state: dict,
        relationships_to_check: List[dict] = None
    ) -> EntityCoverageResult:
        """
        Waliduje czy batch spełnia wymagania encji.
        
        Args:
            batch_text: Tekst batcha do sprawdzenia
            entity_requirements: Lista wymaganych encji z pre_batch_info
            entity_state: Aktualny stan encji z Firestore
            relationships_to_check: Relacje do sprawdzenia
            
        Returns:
            EntityCoverageResult z wynikiem walidacji
        """
        results = []
        must_missing = []
        should_missing = []
        relationships_established = []
        relationships_missing = []
        
        # Waliduj każdą wymaganą encję
        for req in entity_requirements:
            entity = req.get("entity", "")
            action = req.get("action", "MENTION")
            priority = req.get("priority", "SHOULD")
            
            result = self._validate_entity(
                batch_text=batch_text,
                entity=entity,
                action=action,
                entity_data=entity_state.get(entity, {})
            )
            result.priority = priority
            results.append(result)
            
            # Śledź brakujące
            if not result.is_valid:
                if priority == "MUST":
                    must_missing.append(entity)
                elif priority == "SHOULD":
                    should_missing.append(entity)
        
        # Waliduj relacje
        if relationships_to_check:
            for rel in relationships_to_check:
                established = self._check_relationship(
                    batch_text=batch_text,
                    subject=rel.get("subject", ""),
                    relation=rel.get("relation", ""),
                    obj=rel.get("object", "")
                )
                
                rel_info = {
                    "subject": rel.get("subject"),
                    "relation": rel.get("relation"),
                    "object": rel.get("object")
                }
                
                if established:
                    relationships_established.append(rel_info)
                else:
                    relationships_missing.append(rel_info)
        
        # Oblicz score
        total = len(results) if results else 1
        valid_count = sum(1 for r in results if r.is_valid)
        score = int((valid_count / total) * 100)
        
        # Określ status
        if must_missing:
            status = "FAIL"
        elif should_missing or relationships_missing:
            status = "WARNING"
        else:
            status = "PASS"
        
        return EntityCoverageResult(
            status=status,
            score=score,
            results=results,
            must_missing=must_missing,
            should_missing=should_missing,
            relationships_established=relationships_established,
            relationships_missing=relationships_missing
        )
    
    def _validate_entity(
        self,
        batch_text: str,
        entity: str,
        action: str,
        entity_data: dict
    ) -> EntityValidationResult:
        """Waliduje pojedynczą encję."""
        
        text_lower = batch_text.lower()
        entity_lower = entity.lower()
        issues = []
        suggestions = []
        patterns_matched = []
        
        # 1. Sprawdź czy encja w ogóle występuje
        entity_found = entity_lower in text_lower
        
        if not entity_found:
            return EntityValidationResult(
                entity=entity,
                expected_action=action,
                priority="",
                actual_status="NOT_FOUND",
                is_valid=False,
                issues=[f"Encja '{entity}' nie występuje w tekście"],
                suggestions=[f"Dodaj wzmiankę o '{entity}' zgodnie z wymaganiem: {action}"]
            )
        
        # 2. Sprawdź czy wymagana akcja została wykonana
        if action == "DEFINE":
            matched = self._check_patterns(batch_text, entity, self.DEFINITION_PATTERNS)
            
            # v38.3: Semantic fallback jeśli regex nie znalazł
            if not matched:
                matched = self._semantic_fallback_definition(batch_text, entity)
            
            if not matched:
                issues.append(f"Encja '{entity}' występuje, ale nie została zdefiniowana")
                suggestions.append(f"Dodaj definicję, np.: '{entity} to...' lub '{entity} jest...'")
                return EntityValidationResult(
                    entity=entity,
                    expected_action=action,
                    priority="",
                    actual_status="PARTIALLY_FOUND",
                    is_valid=False,
                    issues=issues,
                    suggestions=suggestions
                )
            patterns_matched.extend(matched)
        
        elif action == "EXPLAIN":
            matched = self._check_patterns(batch_text, entity, self.EXPLANATION_PATTERNS)
            
            # v38.3: Semantic fallback jeśli regex nie znalazł
            if not matched:
                matched = self._semantic_fallback_explanation(batch_text, entity)
            
            if not matched:
                issues.append(f"Encja '{entity}' nie została wyjaśniona (brak opisu celu/skutków)")
                suggestions.append(f"Dodaj wyjaśnienie, np.: 'celem {entity} jest...' lub '{entity} służy...'")
                return EntityValidationResult(
                    entity=entity,
                    expected_action=action,
                    priority="",
                    actual_status="PARTIALLY_FOUND",
                    is_valid=False,
                    issues=issues,
                    suggestions=suggestions
                )
            patterns_matched.extend(matched)
        
        elif action == "INTRODUCE":
            # Dla INTRODUCE sprawdź czy jest w sensownym kontekście
            matched_intro = self._check_patterns(batch_text, entity, self.INTRODUCTION_PATTERNS)
            matched_def = self._check_patterns(batch_text, entity, self.DEFINITION_PATTERNS)
            
            if not matched_intro and not matched_def:
                # Soft warning - encja jest, ale nie w kontekście
                issues.append(f"Encja '{entity}' wymieniona, ale nie wprowadzona w kontekst")
                suggestions.append(f"Wprowadź encję w kontekst, np.: 'instytucja {entity}' lub 'w przypadku {entity}'")
                # Nie blokujemy, tylko warning
                return EntityValidationResult(
                    entity=entity,
                    expected_action=action,
                    priority="",
                    actual_status="PARTIALLY_FOUND",
                    is_valid=True,  # Soft - akceptujemy
                    issues=issues,
                    suggestions=suggestions,
                    patterns_matched=matched_intro + matched_def
                )
            patterns_matched.extend(matched_intro + matched_def)
        
        # MENTION - wystarczy że jest
        return EntityValidationResult(
            entity=entity,
            expected_action=action,
            priority="",
            actual_status="FOUND",
            is_valid=True,
            issues=[],
            suggestions=[],
            patterns_matched=patterns_matched
        )
    
    def _check_patterns(
        self, 
        text: str, 
        entity: str, 
        patterns: List[str]
    ) -> List[str]:
        """Sprawdza które wzorce pasują."""
        text_lower = text.lower()
        entity_escaped = re.escape(entity.lower())
        matched = []
        
        for pattern in patterns:
            regex = pattern.format(entity=entity_escaped)
            try:
                if re.search(regex, text_lower):
                    matched.append(pattern)
            except re.error:
                continue
        
        return matched
    
    # ================================================================
    # v38.3: SEMANTIC FALLBACK - gdy regex nie znalazł
    # ================================================================
    
    # Słowa kluczowe wskazujące na definicję
    DEFINITION_KEYWORDS = [
        "to", "jest", "stanowi", "oznacza", "definiuje", "określa",
        "polega", "nazywa", "rozumie", "traktuje", "uznaje"
    ]
    
    # Słowa kluczowe wskazujące na wyjaśnienie
    EXPLANATION_KEYWORDS = [
        "służy", "celem", "pozwala", "umożliwia", "prowadzi", "skutkuje",
        "zapewnia", "chroni", "zabezpiecza", "gwarantuje", "daje",
        "ma na celu", "w celu", "po to", "dzięki", "przez co"
    ]
    
    def _semantic_fallback_definition(self, text: str, entity: str) -> List[str]:
        """
        v38.3: Semantic fallback dla definicji.
        Sprawdza czy zdanie z encją zawiera słowa kluczowe definicji.
        """
        text_lower = text.lower()
        entity_lower = entity.lower()
        matched = []
        
        # Podziel na zdania
        sentences = re.split(r'[.!?]+', text_lower)
        
        for sentence in sentences:
            if entity_lower not in sentence:
                continue
            
            # Sprawdź czy encja jest blisko słowa kluczowego definicji
            for keyword in self.DEFINITION_KEYWORDS:
                # Encja + keyword w tym samym zdaniu (max 50 znaków odległości)
                entity_pos = sentence.find(entity_lower)
                keyword_pos = sentence.find(keyword)
                
                if entity_pos >= 0 and keyword_pos >= 0:
                    distance = abs(entity_pos - keyword_pos)
                    if distance < 50:  # Blisko siebie
                        matched.append(f"[SEMANTIC_FALLBACK] {entity} + {keyword}")
                        return matched  # Wystarczy jeden match
        
        return matched
    
    def _semantic_fallback_explanation(self, text: str, entity: str) -> List[str]:
        """
        v38.3: Semantic fallback dla wyjaśnienia.
        Sprawdza czy zdanie z encją zawiera słowa kluczowe wyjaśnienia.
        """
        text_lower = text.lower()
        entity_lower = entity.lower()
        matched = []
        
        # Podziel na zdania
        sentences = re.split(r'[.!?]+', text_lower)
        
        for sentence in sentences:
            if entity_lower not in sentence:
                continue
            
            # Sprawdź czy encja jest blisko słowa kluczowego wyjaśnienia
            for keyword in self.EXPLANATION_KEYWORDS:
                if keyword in sentence:
                    entity_pos = sentence.find(entity_lower)
                    keyword_pos = sentence.find(keyword)
                    
                    if entity_pos >= 0 and keyword_pos >= 0:
                        distance = abs(entity_pos - keyword_pos)
                        if distance < 60:  # Trochę większy zakres dla wyjaśnień
                            matched.append(f"[SEMANTIC_FALLBACK] {entity} + {keyword}")
                            return matched
        
        return matched
    
    def _check_relationship(
        self,
        batch_text: str,
        subject: str,
        relation: str,
        obj: str
    ) -> bool:
        """Sprawdza czy relacja została ustanowiona."""
        
        text_lower = batch_text.lower()
        subject_lower = subject.lower()
        obj_lower = obj.lower()
        
        # Sprawdź czy oba byty są w tekście
        if subject_lower not in text_lower or obj_lower not in text_lower:
            return False
        
        # Sprawdź wzorce relacji
        patterns = self.RELATION_PATTERNS.get(relation, [])
        
        for pattern in patterns:
            regex = pattern.format(
                subject=re.escape(subject_lower),
                object=re.escape(obj_lower)
            )
            try:
                if re.search(regex, text_lower):
                    return True
            except re.error:
                continue
        
        # Fallback: sprawdź czy są blisko siebie (w tym samym zdaniu)
        sentences = re.split(r'[.!?]+', text_lower)
        for sentence in sentences:
            if subject_lower in sentence and obj_lower in sentence:
                return True
        
        return False
    
    def update_entity_state(
        self,
        entity_state: dict,
        batch_text: str,
        batch_number: int,
        validation_results: List[EntityValidationResult]
    ) -> dict:
        """
        Aktualizuje entity_state na podstawie wyników walidacji.
        """
        text_lower = batch_text.lower()
        
        for result in validation_results:
            entity = result.entity
            if entity not in entity_state:
                continue
            
            state = entity_state[entity]
            entity_lower = entity.lower()
            
            # Aktualizuj tracking
            if result.actual_status in ["FOUND", "PARTIALLY_FOUND"]:
                # Licz wystąpienia
                mentions = text_lower.count(entity_lower)
                state["tracking"]["mentions"] = state["tracking"].get("mentions", 0) + mentions
                
                # Dodaj batch do listy
                if batch_number not in state["tracking"].get("batches_used", []):
                    if "batches_used" not in state["tracking"]:
                        state["tracking"]["batches_used"] = []
                    state["tracking"]["batches_used"].append(batch_number)
                
                # Pierwszy batch z wzmianką
                if not state["status"].get("first_mentioned_batch"):
                    state["status"]["first_mentioned_batch"] = batch_number
            
            # Aktualizuj status na podstawie akcji
            if result.is_valid:
                if result.expected_action == "INTRODUCE":
                    state["status"]["introduced"] = True
                
                elif result.expected_action == "DEFINE":
                    state["status"]["defined"] = True
                    state["status"]["introduced"] = True
                    state["tracking"]["definition_batch"] = batch_number
                
                elif result.expected_action == "EXPLAIN":
                    state["status"]["explained"] = True
                    state["tracking"]["explanation_batch"] = batch_number
        
        return entity_state
    
    def update_relationships(
        self,
        entity_state: dict,
        relationships_established: List[dict]
    ) -> dict:
        """Aktualizuje status relacji w entity_state."""
        
        for rel in relationships_established:
            subject = rel.get("subject", "")
            relation = rel.get("relation", "")
            obj = rel.get("object", "")
            
            if subject in entity_state:
                for r in entity_state[subject].get("relationships", []):
                    if r.get("relation") == relation and r.get("target") == obj:
                        r["established"] = True
        
        return entity_state


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def initialize_entity_state(s1_data: dict, batch_plan: dict) -> dict:
    """
    Tworzy entity_state na podstawie S1.
    
    Args:
        s1_data: Dane z analizy S1
        batch_plan: Plan batchów
        
    Returns:
        Zainicjalizowany entity_state
    """
    entity_state = {}
    total_batches = batch_plan.get("total_batches", 6)
    
    for entity in s1_data.get("entities", []):
        entity_name = entity.get("name", "")
        if not entity_name:
            continue
            
        importance = entity.get("importance", "MEDIUM")
        entity_type = entity.get("type", "concept")
        
        # Określ wymagania na podstawie importance
        if importance == "HIGH":
            must_define = True
            must_explain_by = min(2, total_batches // 2)
            define_before_use = True
        elif importance == "MEDIUM":
            must_define = False
            must_explain_by = total_batches - 1
            define_before_use = False
        else:
            must_define = False
            must_explain_by = None
            define_before_use = False
        
        # Pobierz relacje z S1
        relationships = []
        for rel in s1_data.get("entity_relationships", []):
            if rel.get("subject") == entity_name:
                relationships.append({
                    "relation": rel.get("relation", ""),
                    "target": rel.get("object", ""),
                    "established": False
                })
        
        entity_state[entity_name] = {
            "entity": entity_name,
            "type": entity_type,
            "importance": importance,
            "source": "S1",
            "status": {
                "introduced": False,
                "defined": False,
                "explained": False,
                "first_mentioned_batch": None
            },
            "requirements": {
                "must_define": must_define,
                "must_explain_by_batch": must_explain_by,
                "define_before_use": define_before_use
            },
            "tracking": {
                "mentions": 0,
                "batches_used": [],
                "definition_batch": None,
                "explanation_batch": None
            },
            "relationships": relationships
        }
    
    return entity_state


def generate_entity_requirements(
    entity_state: dict,
    batch_number: int,
    semantic_plan: dict,
    total_batches: int
) -> dict:
    """
    Generuje wymagania encji dla danego batcha.
    
    Returns:
        {
            "required_entities": [...],
            "relationships_to_establish": [...],
            "entity_warnings": [...]
        }
    """
    required = []
    warnings = []
    relationships = []
    
    batch_key = f"batch_{batch_number}"
    assigned_entities = semantic_plan.get(batch_key, {}).get("assigned_entities", [])
    reserved_keywords = semantic_plan.get("reserved_keywords", [])
    
    for entity_name, entity_data in entity_state.items():
        status = entity_data.get("status", {})
        reqs = entity_data.get("requirements", {})
        importance = entity_data.get("importance", "MEDIUM")
        
        # Czy encja jest przypisana do tego batcha?
        assigned_to_batch = entity_name in assigned_entities
        
        # Czy deadline się zbliża?
        must_explain_by = reqs.get("must_explain_by_batch")
        deadline_approaching = must_explain_by and batch_number >= must_explain_by - 1
        
        # Sprawdź czy zarezerwowana na później
        reserved_for_later = False
        reserved_batch = None
        for r in reserved_keywords:
            if r.get("keyword") == entity_name:
                reserved_batch = r.get("reserved_for_batch", 0)
                if reserved_batch > batch_number:
                    reserved_for_later = True
                    warnings.append({
                        "entity": entity_name,
                        "warning": "RESERVED_FOR_LATER",
                        "reserved_for_batch": reserved_batch,
                        "reason": f"Encja zarezerwowana na batch {reserved_batch}"
                    })
                break
        
        if reserved_for_later:
            continue
        
        # Określ wymaganą akcję
        action = None
        priority = None
        reason = None
        guidance = None
        
        if not status.get("introduced") and (assigned_to_batch or deadline_approaching):
            action = "INTRODUCE"
            priority = "MUST" if importance == "HIGH" else "SHOULD"
            reason = f"{'Assigned to this batch' if assigned_to_batch else 'Deadline approaching'}"
            guidance = f"Wprowadź encję '{entity_name}' w kontekst tematu"
        
        elif status.get("introduced") and not status.get("defined") and reqs.get("must_define"):
            action = "DEFINE"
            priority = "MUST"
            reason = "HIGH importance entity requires definition"
            guidance = f"Zdefiniuj: '{entity_name} to/jest...'"
        
        elif status.get("introduced") and status.get("defined") and not status.get("explained") and deadline_approaching:
            action = "EXPLAIN"
            priority = "MUST" if importance == "HIGH" else "SHOULD"
            reason = f"Must explain by batch {must_explain_by}"
            guidance = f"Wyjaśnij cel/skutki: 'celem {entity_name} jest...' lub '{entity_name} służy...'"
        
        elif assigned_to_batch:
            action = "MENTION"
            priority = "SHOULD"
            reason = "Assigned to this batch"
            guidance = f"Użyj encji '{entity_name}' w treści"
        
        if action:
            required.append({
                "entity": entity_name,
                "current_status": {
                    "introduced": status.get("introduced", False),
                    "defined": status.get("defined", False),
                    "explained": status.get("explained", False)
                },
                "action": action,
                "priority": priority,
                "reason": reason,
                "guidance": guidance
            })
        
        # Sprawdź relacje do ustalenia
        if status.get("introduced"):
            for rel in entity_data.get("relationships", []):
                if rel.get("established"):
                    continue
                    
                target = rel.get("target", "")
                target_state = entity_state.get(target, {}).get("status", {})
                
                if target_state.get("introduced"):
                    relationships.append({
                        "subject": entity_name,
                        "relation": rel.get("relation", ""),
                        "object": target,
                        "priority": "SHOULD",
                        "status": "NOT_ESTABLISHED"
                    })
    
    return {
        "required_entities": required,
        "relationships_to_establish": relationships,
        "entity_warnings": warnings
    }


# ================================================================
# 🆕 v38.2: ENTITY DRIFT DETECTION
# ================================================================

class EntityDriftDetector:
    """
    Wykrywa zmiany w definicji/roli encji między batchami.
    
    Drift = encja jest definiowana inaczej niż wcześniej
    Przykład: 
    - Batch 1: "ubezwłasnowolnienie to instytucja ochronna"
    - Batch 3: "ubezwłasnowolnienie to forma kary"
    → DRIFT DETECTED!
    """
    
    # Wzorce wyciągające definicję
    DEFINITION_EXTRACT_PATTERNS = [
        # "X to Y"
        r"{entity}\s+to\s+([^.]+)",
        # "X jest Y"
        r"{entity}\s+jest\s+([^.]+)",
        # "X stanowi Y"
        r"{entity}\s+stanowi\s+([^.]+)",
        # "X oznacza Y"  
        r"{entity}\s+oznacza\s+([^.]+)",
        # "przez X rozumie się Y"
        r"przez\s+{entity}\s+rozumie\s+się\s+([^.]+)",
    ]
    
    # Słowa kluczowe charakteryzujące definicję
    DEFINITION_KEYWORDS = {
        "positive": ["ochrona", "ochronny", "ochronna", "pomoc", "wsparcie", "zabezpieczenie"],
        "negative": ["kara", "karny", "sankcja", "ograniczenie", "pozbawienie"],
        "procedural": ["procedura", "postępowanie", "proces", "tryb"],
        "institutional": ["instytucja", "organ", "urząd", "sąd"],
    }
    
    def __init__(self):
        self.initial_definitions = {}  # {entity: {text, category, batch}}
    
    def extract_definition(self, text: str, entity: str) -> Optional[dict]:
        """Wyciąga definicję encji z tekstu."""
        text_lower = text.lower()
        entity_lower = entity.lower()
        entity_escaped = re.escape(entity_lower)
        
        for pattern in self.DEFINITION_EXTRACT_PATTERNS:
            regex = pattern.format(entity=entity_escaped)
            match = re.search(regex, text_lower)
            if match:
                definition_text = match.group(1).strip()
                category = self._categorize_definition(definition_text)
                return {
                    "text": definition_text[:200],  # Max 200 chars
                    "category": category,
                    "full_match": match.group(0)
                }
        
        return None
    
    def _categorize_definition(self, definition_text: str) -> str:
        """Kategoryzuje definicję na podstawie słów kluczowych."""
        text_lower = definition_text.lower()
        
        scores = {}
        for category, keywords in self.DEFINITION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return "neutral"
        
        return max(scores, key=scores.get)
    
    def check_drift(
        self,
        entity: str,
        new_definition: dict,
        stored_definition: dict
    ) -> Optional[dict]:
        """
        Sprawdza czy nowa definicja różni się od zapisanej.
        
        Returns:
            dict z informacją o drift lub None jeśli brak
        """
        if not stored_definition or not new_definition:
            return None
        
        old_category = stored_definition.get("category", "neutral")
        new_category = new_definition.get("category", "neutral")
        
        # v38.3: Dozwolone przejścia (nie są driftem)
        allowed_transitions = {
            ("procedural", "institutional"),
            ("institutional", "procedural"),
            # Neutral może przejść w cokolwiek
            ("neutral", "positive"),
            ("neutral", "negative"),
            ("neutral", "procedural"),
            ("neutral", "institutional"),
        }
        
        # Jeśli przejście jest dozwolone → brak driftu
        if (old_category, new_category) in allowed_transitions:
            return None
        
        # Krytyczny drift: positive ↔ negative (sprzeczność semantyczna!)
        critical_drift = (
            (old_category == "positive" and new_category == "negative") or
            (old_category == "negative" and new_category == "positive")
        )
        
        if critical_drift:
            return {
                "entity": entity,
                "severity": "CRITICAL",
                "old_definition": stored_definition.get("text", ""),
                "old_category": old_category,
                "new_definition": new_definition.get("text", ""),
                "new_category": new_category,
                "message": f"Entity drift: '{entity}' zmienia charakter z {old_category} na {new_category}"
            }
        
        # Warning drift: inna zmiana kategorii (nie critical, nie dozwolona)
        if old_category != new_category and old_category != "neutral" and new_category != "neutral":
            return {
                "entity": entity,
                "severity": "WARNING",
                "old_definition": stored_definition.get("text", ""),
                "old_category": old_category,
                "new_definition": new_definition.get("text", ""),
                "new_category": new_category,
                "message": f"Entity drift: '{entity}' zmienia kategorię z {old_category} na {new_category}"
            }
        
        return None
    
    def detect_drift_in_batch(
        self,
        batch_text: str,
        entity_state: dict,
        batch_number: int
    ) -> List[dict]:
        """
        Sprawdza wszystkie encje w batchu pod kątem drift.
        
        Returns:
            Lista wykrytych driftów
        """
        drifts = []
        
        for entity_name, entity_data in entity_state.items():
            # Wyciągnij definicję z nowego batcha
            new_def = self.extract_definition(batch_text, entity_name)
            
            if not new_def:
                continue
            
            # Pobierz zapisaną definicję
            stored_def = entity_data.get("initial_definition")
            
            if stored_def:
                # Sprawdź drift
                drift = self.check_drift(entity_name, new_def, stored_def)
                if drift:
                    drift["batch_number"] = batch_number
                    drifts.append(drift)
            else:
                # Zapisz jako initial definition
                entity_data["initial_definition"] = {
                    "text": new_def.get("text", ""),
                    "category": new_def.get("category", "neutral"),
                    "batch": batch_number
                }
        
        return drifts


# Global instance
_drift_detector = EntityDriftDetector()


def detect_entity_drift(
    batch_text: str,
    entity_state: dict,
    batch_number: int
) -> List[dict]:
    """Convenience function do wykrywania entity drift."""
    return _drift_detector.detect_drift_in_batch(batch_text, entity_state, batch_number)
