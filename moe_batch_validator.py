"""
===============================================================================
🔍 MOE BATCH VALIDATOR v1.3 - Mixture of Experts Post-Batch Validation
===============================================================================
Kompleksowa walidacja batcha po wygenerowaniu przez GPT.

EKSPERCI (MoE):
1. STRUCTURE EXPERT - różna liczba akapitów, anty-monotonność
2. SEO EXPERT - BASIC/EXTENDED keywords, encje, n-gramy
3. LANGUAGE EXPERT - gramatyka polska (LanguageTool), styl
4. AI DETECTION EXPERT - burstiness, TTR, rozkład zdań
5. UNIFIED BRIDGE EXPERT - mostek do unified_validator (optional)
6. CORPUS INSIGHTS - metryki NKJP (informacyjne, nie blokuje!)

🆕 v1.3 ZMIANY:
- Integracja z content_surgeon.py (chirurgiczne wstawianie fraz)
- Integracja z semantic_triplet_validator.py (semantyczna walidacja tripletów)
- AUTO_FIX mode używa Content Surgeon zamiast pełnego retry
- Triplety walidowane semantycznie (akceptuje warianty językowe)

TRYBY:
- SOFT: tylko warnings, batch zapisuje się
- STRICT: critical errors blokują batch
- AUTO_FIX: 🆕 używa Content Surgeon do naprawy

===============================================================================
"""

import re
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
from enum import Enum

# ================================================================
# IMPORTY ZEWNĘTRZNE
# ================================================================
try:
    from grammar_middleware import validate_batch_grammar
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ LanguageTool not available")

try:
    from ai_detection_metrics import (
        calculate_burstiness, 
        calculate_burstiness_dynamic,
        calculate_vocabulary_richness,
        analyze_sentence_distribution,
        check_word_repetition_detailed,
        AIDetectionConfig
    )
    AI_METRICS_AVAILABLE = True
except ImportError:
    AI_METRICS_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ AI Detection Metrics not available")

try:
    from keyword_synonyms import get_synonyms
    SYNONYMS_AVAILABLE = True
except ImportError:
    SYNONYMS_AVAILABLE = False
    def get_synonyms(kw): return []

try:
    from keyword_counter import count_keywords_for_state
    KEYWORD_COUNTER_AVAILABLE = True
except ImportError:
    KEYWORD_COUNTER_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ Keyword Counter not available")

# UNIFIED VALIDATOR BRIDGE (optional)
try:
    from unified_validator import (
        validate_content,
        quick_validate,
        calculate_entity_density,
        validate_semantic_enhancement,
        check_template_patterns,
        ValidationConfig as UnifiedValidationConfig
    )
    UNIFIED_VALIDATOR_AVAILABLE = True
    print("[MOE_VALIDATOR] ✅ Unified Validator bridge enabled")
except (ImportError, NameError, Exception) as e:
    UNIFIED_VALIDATOR_AVAILABLE = False
    print(f"[MOE_VALIDATOR] ⚠️ Unified Validator not available: {e}")

# DYNAMIC CV THRESHOLDS
try:
    from humanness_weights_v41 import (
        get_dynamic_cv_thresholds,
        evaluate_cv_dynamic
    )
    DYNAMIC_CV_AVAILABLE = True
except ImportError:
    DYNAMIC_CV_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ Dynamic CV thresholds not available")

# POLISH CORPUS INSIGHTS (optional, NEVER blocks!)
try:
    from polish_corpus_metrics_v41 import (
        get_corpus_insights_for_moe,
        get_naturalness_hints,
        analyze_corpus_metrics,
        ENABLE_CORPUS_INSIGHTS
    )
    CORPUS_INSIGHTS_AVAILABLE = True
    print("[MOE_VALIDATOR] ✅ Polish Corpus Insights enabled")
except ImportError:
    CORPUS_INSIGHTS_AVAILABLE = False
    ENABLE_CORPUS_INSIGHTS = False
    print("[MOE_VALIDATOR] ℹ️ Polish Corpus Insights not available (optional)")
    
    def get_corpus_insights_for_moe(text: str, **kwargs) -> dict:
        return {"enabled": False, "affects_validation": False}
    
    def get_naturalness_hints(text: str) -> list:
        return []

# ================================================================
# 🆕 v1.3: CONTENT SURGEON (chirurgiczne wstawianie fraz)
# ================================================================
try:
    from content_surgeon import (
        perform_surgery,
        find_injection_point,
        generate_injection_sentence
    )
    CONTENT_SURGEON_AVAILABLE = True
    print("[MOE_VALIDATOR] ✅ Content Surgeon v1.0 enabled (auto-fix mode)")
except ImportError:
    CONTENT_SURGEON_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ Content Surgeon not available")
    
    def perform_surgery(text, phrases, h2, domain="prawo"):
        return {"success": False, "modified_text": text, "stats": {"injected": 0}}

# ================================================================
# 🆕 v1.3: SEMANTIC TRIPLET VALIDATOR (semantyczna walidacja)
# ================================================================
try:
    from semantic_triplet_validator import (
        validate_triplets_in_text,
        validate_triplet_in_sentence,
        generate_semantic_instruction
    )
    SEMANTIC_TRIPLET_AVAILABLE = True
    print("[MOE_VALIDATOR] ✅ Semantic Triplet Validator v1.0 enabled")
except ImportError:
    SEMANTIC_TRIPLET_AVAILABLE = False
    print("[MOE_VALIDATOR] ⚠️ Semantic Triplet Validator not available")
    
    def validate_triplets_in_text(text, triplets):
        return {"passed": True, "matched": len(triplets), "missing": []}


# ================================================================
# KONFIGURACJA
# ================================================================
class ValidationMode(Enum):
    SOFT = "soft"
    STRICT = "strict"
    AUTO_FIX = "auto_fix"  # 🆕 v1.3: Używa Content Surgeon


@dataclass
class ValidationConfig:
    """Konfiguracja progów walidacji."""
    min_paragraph_count: int = 3
    max_paragraph_count: int = 8
    paragraph_variance_min: float = 0.3
    basic_coverage_min: float = 0.7
    entity_coverage_min: float = 0.5
    max_grammar_errors: int = 3
    max_critical_grammar: int = 1
    burstiness_min: float = 2.2
    burstiness_max: float = 4.5
    ttr_min: float = 0.42
    ttr_max: float = 0.65
    short_sentence_pct_min: int = 15
    short_sentence_pct_max: int = 30
    max_word_repetition: int = 6
    include_corpus_insights: bool = True
    # 🆕 v1.3: Content Surgeon config
    enable_auto_fix: bool = True
    max_auto_fix_phrases: int = 3
    # 🆕 v1.3: Semantic triplet config
    triplet_similarity_threshold: float = 0.55


@dataclass
class ValidationIssue:
    """Pojedynczy problem znaleziony przez walidator."""
    expert: str
    severity: str
    code: str
    message: str
    fix_instruction: str = ""
    auto_fixable: bool = False
    context: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Wynik walidacji MoE."""
    passed: bool
    status: str
    issues: List[ValidationIssue]
    experts_summary: Dict[str, Dict]
    fix_instructions: List[str]
    auto_fixes_applied: List[str] = field(default_factory=list)
    corrected_text: Optional[str] = None
    corpus_insights: Optional[Dict] = None
    naturalness_hints: List[Dict] = field(default_factory=list)
    # 🆕 v1.3: Surgery stats
    surgery_applied: bool = False
    surgery_stats: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        result = {
            "passed": self.passed,
            "status": self.status,
            "issues": [asdict(i) for i in self.issues],
            "experts_summary": self.experts_summary,
            "fix_instructions": self.fix_instructions,
            "auto_fixes_applied": self.auto_fixes_applied,
            "has_corrected_text": self.corrected_text is not None,
            "surgery_applied": self.surgery_applied
        }
        if self.corpus_insights:
            result["corpus_insights"] = self.corpus_insights
        if self.naturalness_hints:
            result["naturalness_hints"] = self.naturalness_hints
        if self.surgery_stats:
            result["surgery_stats"] = self.surgery_stats
        return result


# ================================================================
# 1️⃣ STRUCTURE EXPERT
# ================================================================
class StructureExpert:
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, batch_text: str, previous_batches: List[Dict], current_h2: str = "") -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        paragraphs = self._split_paragraphs(batch_text)
        current_para_count = len(paragraphs)
        
        previous_para_counts = []
        for batch in previous_batches:
            text = batch.get("text", "")
            prev_paras = self._split_paragraphs(text)
            previous_para_counts.append(len(prev_paras))
        
        if previous_para_counts:
            last_para_count = previous_para_counts[-1]
            if current_para_count == last_para_count and current_para_count > 2:
                issues.append(ValidationIssue(
                    expert="structure",
                    severity="warning",
                    code="PARAGRAPH_MONOTONY",
                    message=f"Ta sekcja ma {current_para_count} akapitów - IDENTYCZNIE jak poprzednia!",
                    fix_instruction=f"Zmień liczbę akapitów z {current_para_count} na {current_para_count + 1} lub {max(2, current_para_count - 1)}.",
                    context={"current_count": current_para_count, "previous_count": last_para_count}
                ))
            
            if len(previous_para_counts) >= 2:
                last_three = previous_para_counts[-2:] + [current_para_count]
                if len(set(last_three)) == 1:
                    issues.append(ValidationIssue(
                        expert="structure",
                        severity="critical",
                        code="STRUCTURE_PATTERN_DETECTED",
                        message=f"Ostatnie 3 sekcje mają identyczną strukturę ({last_three[0]} akapitów). AI pattern!",
                        fix_instruction=f"MUSISZ zmienić liczbę akapitów. Użyj {last_three[0] + 2} lub {max(2, last_three[0] - 1)}.",
                        context={"pattern": last_three}
                    ))
        
        para_lengths = [len(p.split()) for p in paragraphs]
        if para_lengths and len(para_lengths) > 1:
            mean_len = statistics.mean(para_lengths)
            std_len = statistics.stdev(para_lengths) if len(para_lengths) > 1 else 0
            cv = std_len / mean_len if mean_len > 0 else 0
            
            if cv < self.config.paragraph_variance_min:
                issues.append(ValidationIssue(
                    expert="structure",
                    severity="warning",
                    code="PARAGRAPH_LENGTH_UNIFORM",
                    message=f"Akapity zbyt podobnej długości (CV={cv:.2f}). Długości: {para_lengths}",
                    fix_instruction="Zróżnicuj długość akapitów. Niektóre krótkie (2-3 zdania), inne dłuższe (5-6 zdań).",
                    context={"lengths": para_lengths, "cv": cv}
                ))
        
        summary = {
            "paragraph_count": current_para_count,
            "paragraph_lengths": para_lengths,
            "previous_counts": previous_para_counts,
            "issues_count": len(issues)
        }
        return issues, summary
    
    def _split_paragraphs(self, text: str) -> List[str]:
        clean = re.sub(r'<[^>]+>', '\n', text)
        paragraphs = re.split(r'\n\s*\n', clean)
        return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


# ================================================================
# 2️⃣ SEO EXPERT (z integracją Content Surgeon)
# ================================================================
class SEOExpert:
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(
        self,
        batch_text: str,
        keywords_state: Dict,
        batch_counts: Dict,
        s1_entities: List = None,
        assigned_keywords: List[str] = None,
        batch_number: int = 1,
        current_h2: str = ""
    ) -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        basic_missing = []
        basic_used = []
        
        for rid, meta in keywords_state.items():
            if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
                continue
            
            keyword = meta.get("keyword", "")
            target_min = meta.get("target_min", 1)
            target_max = meta.get("target_max", 999)
            actual = meta.get("actual_uses", 0)
            batch_use = batch_counts.get(rid, 0)
            
            if assigned_keywords and keyword.lower() in [k.lower() for k in assigned_keywords]:
                if batch_use == 0:
                    synonyms = get_synonyms(keyword) if SYNONYMS_AVAILABLE else []
                    issues.append(ValidationIssue(
                        expert="seo",
                        severity="warning",
                        code="ASSIGNED_KEYWORD_MISSING",
                        message=f"Fraza '{keyword}' była PRZYPISANA do tej sekcji, ale nie została użyta!",
                        fix_instruction=f"Dodaj frazę '{keyword}' do tekstu (min 1x).",
                        auto_fixable=CONTENT_SURGEON_AVAILABLE,  # 🆕 v1.3
                        context={"keyword": keyword, "assigned": True, "synonyms": synonyms}
                    ))
                    basic_missing.append(keyword)
            
            if actual < target_min and batch_use == 0:
                remaining_batches = 7 - batch_number
                if remaining_batches <= 2 and actual == 0:
                    issues.append(ValidationIssue(
                        expert="seo",
                        severity="critical" if meta.get("type") == "MAIN" else "warning",
                        code="BASIC_STILL_ZERO",
                        message=f"BASIC '{keyword}' ma 0 użyć! Zostały {remaining_batches} batche.",
                        fix_instruction=f"MUSISZ użyć '{keyword}' w tym lub następnym batchu.",
                        auto_fixable=CONTENT_SURGEON_AVAILABLE,  # 🆕 v1.3
                        context={"keyword": keyword, "remaining_batches": remaining_batches}
                    ))
                    basic_missing.append(keyword)
            
            if batch_use > 0:
                basic_used.append({"keyword": keyword, "count": batch_use})
        
        coverage = 0
        entities_missing = []
        if s1_entities:
            text_lower = batch_text.lower()
            entities_in_batch = []
            for entity in s1_entities[:15]:
                name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                if name.lower() in text_lower:
                    entities_in_batch.append(name)
                else:
                    entities_missing.append(name)
            coverage = len(entities_in_batch) / len(s1_entities[:15]) if s1_entities else 0
        
        summary = {
            "basic_used": basic_used,
            "basic_missing": basic_missing,
            "entity_coverage": coverage if s1_entities else None,
            "issues_count": len(issues),
            "auto_fixable_count": len([i for i in issues if i.auto_fixable])  # 🆕 v1.3
        }
        return issues, summary


# ================================================================
# 3️⃣ LANGUAGE EXPERT
# ================================================================
class LanguageExpert:
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, batch_text: str) -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        grammar_errors = []
        
        if LANGUAGETOOL_AVAILABLE:
            try:
                lt_result = validate_batch_grammar(batch_text)
                grammar_errors = lt_result.errors if hasattr(lt_result, 'errors') else []
                critical_errors = [e for e in grammar_errors if e.get("rule", {}).get("category", {}).get("id") in ["GRAMMAR", "TYPOS"]] if grammar_errors else []
                
                for err in critical_errors[:self.config.max_critical_grammar + 1]:
                    issues.append(ValidationIssue(
                        expert="language",
                        severity="critical",
                        code="GRAMMAR_CRITICAL",
                        message=f"Błąd gramatyczny: {err.get('message', 'nieznany')}",
                        fix_instruction=f"Popraw: '{err.get('context', '')}' → {err.get('suggestions', ['?'])[0] if err.get('suggestions') else '?'}",
                        auto_fixable=bool(err.get("suggestions")),
                        context=err
                    ))
            except Exception as e:
                print(f"[MOE_VALIDATOR] LanguageTool error: {e}")
        
        polish_issues = self._check_polish_rules(batch_text)
        issues.extend(polish_issues)
        
        summary = {
            "grammar_errors": len(grammar_errors),
            "critical_errors": len([i for i in issues if i.severity == "critical"]),
            "languagetool_available": LANGUAGETOOL_AVAILABLE
        }
        return issues, summary
    
    def _check_polish_rules(self, text: str) -> List[ValidationIssue]:
        issues = []
        common_errors = [
            (r'\bw między\b', 'w między → wśród/między', 'Błędna konstrukcja'),
            (r'\bw skutek\b', 'w skutek → wskutek', 'Pisownia łączna'),
            (r'\bz pod\b', 'z pod → spod', 'Pisownia łączna'),
        ]
        text_lower = text.lower()
        for pattern, fix, msg in common_errors:
            if re.search(pattern, text_lower):
                issues.append(ValidationIssue(
                    expert="language", severity="warning", code="POLISH_SPELLING",
                    message=msg, fix_instruction=fix, auto_fixable=True
                ))
        return issues


# ================================================================
# 4️⃣ AI DETECTION EXPERT
# ================================================================
class AIDetectionExpert:
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, batch_text: str) -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        metrics = {}
        
        if not AI_METRICS_AVAILABLE:
            return issues, {"available": False}
        
        try:
            burstiness_result = calculate_burstiness(batch_text)
            burstiness = burstiness_result.get("value", 0)
            cv = burstiness_result.get("cv", 0)
            metrics["burstiness"] = burstiness
            metrics["cv"] = cv
            
            if burstiness < self.config.burstiness_min:
                issues.append(ValidationIssue(
                    expert="ai_detection",
                    severity="critical" if burstiness < 1.5 else "warning",
                    code="LOW_BURSTINESS",
                    message=f"Niski burstiness: {burstiness} (CV={cv:.2f}). AI pattern!",
                    fix_instruction="Dodaj więcej KRÓTKICH zdań (2-8 słów).",
                    context=burstiness_result
                ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Burstiness error: {e}")
        
        try:
            dist_result = analyze_sentence_distribution(batch_text)
            distribution = dist_result.get("distribution", [0, 0, 0])
            metrics["sentence_distribution"] = distribution
            
            short_pct = distribution[0] if distribution else 0
            if short_pct < self.config.short_sentence_pct_min:
                issues.append(ValidationIssue(
                    expert="ai_detection",
                    severity="warning",
                    code="TOO_FEW_SHORT_SENTENCES",
                    message=f"Za mało krótkich zdań: {short_pct}%",
                    fix_instruction="Dodaj zdania 2-10 słów.",
                    context=dist_result
                ))
            
            ai_concentration = dist_result.get("ai_concentration", 0)
            if ai_concentration > 60:
                issues.append(ValidationIssue(
                    expert="ai_detection",
                    severity="critical",
                    code="AI_SENTENCE_PATTERN",
                    message=f"AI pattern: {ai_concentration:.0f}% zdań ma 15-22 słów",
                    fix_instruction="Zróżnicuj długość zdań! Dodaj krótkie i długie.",
                    context={"ai_concentration": ai_concentration}
                ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Sentence distribution error: {e}")
        
        try:
            repetition_result = check_word_repetition_detailed(batch_text)
            repeated_words = repetition_result.get("repeated_words", [])
            metrics["repeated_words"] = repeated_words[:5]
            
            for word_info in repeated_words[:3]:
                word = word_info.get("word", "")
                count = word_info.get("count", 0)
                if count > self.config.max_word_repetition:
                    issues.append(ValidationIssue(
                        expert="ai_detection",
                        severity="warning",
                        code="WORD_REPETITION",
                        message=f"Słowo '{word}' powtórzone {count}x",
                        fix_instruction=f"Użyj synonimów",
                        auto_fixable=True,
                        context=word_info
                    ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Repetition check error: {e}")
        
        return issues, {"metrics": metrics, "issues_count": len(issues)}


# ================================================================
# 🆕 v1.3: 5️⃣ TRIPLET EXPERT (Semantyczna walidacja)
# ================================================================
class TripletExpert:
    """
    Waliduje triplety SEMANTYCZNIE zamiast dosłownie.
    Akceptuje warianty językowe: forma bierna, synonimy, etc.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(
        self,
        batch_text: str,
        required_triplets: List[Dict],
        h2_title: str = ""
    ) -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        
        if not required_triplets:
            return issues, {"triplets_checked": 0, "passed": True}
        
        if not SEMANTIC_TRIPLET_AVAILABLE:
            # Fallback: proste sprawdzenie string matching
            return self._validate_literal(batch_text, required_triplets)
        
        # Semantyczna walidacja
        result = validate_triplets_in_text(batch_text, required_triplets)
        
        matched = result.get("matched", 0)
        missing = result.get("missing", [])
        score = result.get("score", 0)
        
        for triplet in missing:
            s = triplet.get("subject", "")
            v = triplet.get("verb", "")
            o = triplet.get("object", "")
            
            # Wygeneruj semantyczną instrukcję (nie dosłowną!)
            instruction = generate_semantic_instruction(triplet) if SEMANTIC_TRIPLET_AVAILABLE else f"Napisz zdanie: {s} {v} {o}"
            
            issues.append(ValidationIssue(
                expert="triplet",
                severity="warning",
                code="TRIPLET_MISSING",
                message=f"Brak relacji: {s} → {v} → {o}",
                fix_instruction=instruction,
                auto_fixable=False,  # Triplety wymagają ludzkiej interwencji
                context={"triplet": triplet, "h2": h2_title}
            ))
        
        summary = {
            "triplets_checked": len(required_triplets),
            "matched": matched,
            "missing_count": len(missing),
            "score": score,
            "passed": len(missing) == 0,
            "validation_type": "semantic" if SEMANTIC_TRIPLET_AVAILABLE else "literal"
        }
        
        return issues, summary
    
    def _validate_literal(self, text: str, triplets: List[Dict]) -> Tuple[List[ValidationIssue], Dict]:
        """Fallback: dosłowne sprawdzenie (stara logika)."""
        issues = []
        text_lower = text.lower()
        matched = 0
        
        for triplet in triplets:
            s = triplet.get("subject", "").lower()
            v = triplet.get("verb", "").lower()
            o = triplet.get("object", "").lower()
            
            if s in text_lower and v in text_lower and o in text_lower:
                matched += 1
            else:
                issues.append(ValidationIssue(
                    expert="triplet",
                    severity="warning",
                    code="TRIPLET_MISSING",
                    message=f"Brak relacji: {s} → {v} → {o}",
                    fix_instruction=f"Napisz zdanie zawierające: {s}, {v}, {o}",
                    context={"triplet": triplet}
                ))
        
        return issues, {"triplets_checked": len(triplets), "matched": matched, "validation_type": "literal"}


# ================================================================
# 6️⃣ UNIFIED BRIDGE EXPERT (bez zmian)
# ================================================================
class UnifiedBridgeExpert:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.enabled = UNIFIED_VALIDATOR_AVAILABLE
    
    def validate(self, batch_text: str, keywords_state: Dict, s1_data: Dict = None, main_keyword: str = "") -> Tuple[List[ValidationIssue], Dict]:
        issues = []
        metrics = {"enabled": self.enabled, "checks_performed": []}
        
        if not self.enabled:
            return issues, {"enabled": False}
        
        try:
            quick_result = quick_validate(batch_text, keywords_state)
            metrics["quick_score"] = quick_result.get("score", 0)
            metrics["checks_performed"].append("quick_validate")
        except Exception as e:
            metrics["error"] = str(e)
        
        return issues, {"enabled": True, "metrics": metrics}


# ================================================================
# 🆕 v1.3: CONTENT SURGEON INTEGRATION
# ================================================================
def apply_content_surgery(
    batch_text: str,
    missing_keywords: List[str],
    h2_title: str,
    config: ValidationConfig
) -> Tuple[str, Dict]:
    """
    Wykonuje chirurgiczne wstawianie brakujących fraz.
    
    Returns:
        (corrected_text, surgery_stats)
    """
    if not CONTENT_SURGEON_AVAILABLE:
        return batch_text, {"success": False, "reason": "Content Surgeon not available"}
    
    if not missing_keywords:
        return batch_text, {"success": True, "injected": 0}
    
    # Ogranicz do max_auto_fix_phrases
    keywords_to_fix = missing_keywords[:config.max_auto_fix_phrases]
    
    result = perform_surgery(
        text=batch_text,
        missing_phrases=keywords_to_fix,
        h2_title=h2_title,
        domain="prawo"  # TODO: wykryć domenę
    )
    
    if result["success"]:
        print(f"[MOE_VALIDATOR] ✅ Content Surgery: {result['stats']['injected']}/{len(keywords_to_fix)} phrases injected")
        return result["modified_text"], result["stats"]
    else:
        print(f"[MOE_VALIDATOR] ⚠️ Content Surgery failed: {result.get('failed_phrases', [])}")
        return batch_text, result.get("stats", {"success": False})


# ================================================================
# 🎯 GŁÓWNA FUNKCJA WALIDACJI MOE
# ================================================================
def validate_batch_moe(
    batch_text: str,
    project_data: Dict,
    batch_number: int = 1,
    mode: ValidationMode = ValidationMode.SOFT,
    config: Optional[ValidationConfig] = None,
    include_corpus_insights: bool = True,
    current_h2: str = "",
    required_triplets: List[Dict] = None  # 🆕 v1.3
) -> ValidationResult:
    """
    Główna funkcja walidacji MoE.
    
    🆕 v1.3 ZMIANY:
    - AUTO_FIX mode używa Content Surgeon do naprawy MISSING_KEYWORD
    - Triplety walidowane przez TripletExpert (semantycznie)
    - Dodano surgery_applied i surgery_stats do wyniku
    """
    if config is None:
        config = ValidationConfig()
    
    all_issues = []
    experts_summary = {}
    fix_instructions = []
    auto_fixes_applied = []
    corrected_text = None
    surgery_applied = False
    surgery_stats = None
    
    keywords_state = project_data.get("keywords_state", {})
    previous_batches = project_data.get("batches", [])
    s1_data = project_data.get("s1_data", {})
    s1_entities = s1_data.get("entity_seo", {}).get("entities", [])
    semantic_plan = project_data.get("semantic_keyword_plan", {})
    
    assigned_keywords = []
    for bp in semantic_plan.get("batch_plans", []):
        if bp.get("batch_number") == batch_number:
            assigned_keywords = bp.get("assigned_keywords", [])
            break
    
    batch_counts = {}
    if KEYWORD_COUNTER_AVAILABLE:
        try:
            batch_counts = count_keywords_for_state(batch_text, keywords_state, use_exclusive_for_nested=False)
        except Exception as e:
            print(f"[MOE_VALIDATOR] Keyword counting error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 1️⃣ STRUCTURE EXPERT
    # ═══════════════════════════════════════════════════════════════
    structure_expert = StructureExpert(config)
    structure_issues, structure_summary = structure_expert.validate(batch_text, previous_batches, current_h2)
    all_issues.extend(structure_issues)
    experts_summary["structure"] = structure_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 2️⃣ SEO EXPERT
    # ═══════════════════════════════════════════════════════════════
    seo_expert = SEOExpert(config)
    seo_issues, seo_summary = seo_expert.validate(
        batch_text, keywords_state, batch_counts, s1_entities, assigned_keywords, batch_number, current_h2
    )
    all_issues.extend(seo_issues)
    experts_summary["seo"] = seo_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 🆕 v1.3: AUTO_FIX MODE - Content Surgery
    # ═══════════════════════════════════════════════════════════════
    if mode == ValidationMode.AUTO_FIX and config.enable_auto_fix:
        missing_keywords = seo_summary.get("basic_missing", [])
        
        if missing_keywords and CONTENT_SURGEON_AVAILABLE:
            print(f"[MOE_VALIDATOR] 🔬 Attempting Content Surgery for {len(missing_keywords)} missing keywords...")
            
            corrected_text, surgery_stats = apply_content_surgery(
                batch_text=batch_text,
                missing_keywords=missing_keywords,
                h2_title=current_h2,
                config=config
            )
            
            if surgery_stats.get("injected", 0) > 0:
                surgery_applied = True
                auto_fixes_applied.append(f"Content Surgery: {surgery_stats['injected']} phrases injected")
                
                # Oznacz naprawione issues jako "auto_fixed"
                for issue in all_issues:
                    if issue.code in ["ASSIGNED_KEYWORD_MISSING", "BASIC_STILL_ZERO"]:
                        keyword = issue.context.get("keyword", "")
                        if keyword in missing_keywords[:config.max_auto_fix_phrases]:
                            issue.severity = "info"
                            issue.message += " [AUTO-FIXED by Content Surgeon]"
    
    # ═══════════════════════════════════════════════════════════════
    # 3️⃣ LANGUAGE EXPERT
    # ═══════════════════════════════════════════════════════════════
    language_expert = LanguageExpert(config)
    language_issues, language_summary = language_expert.validate(corrected_text or batch_text)
    all_issues.extend(language_issues)
    experts_summary["language"] = language_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 4️⃣ AI DETECTION EXPERT
    # ═══════════════════════════════════════════════════════════════
    ai_expert = AIDetectionExpert(config)
    ai_issues, ai_summary = ai_expert.validate(corrected_text or batch_text)
    all_issues.extend(ai_issues)
    experts_summary["ai_detection"] = ai_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 🆕 v1.3: 5️⃣ TRIPLET EXPERT (Semantyczna walidacja)
    # ═══════════════════════════════════════════════════════════════
    if required_triplets:
        triplet_expert = TripletExpert(config)
        triplet_issues, triplet_summary = triplet_expert.validate(
            corrected_text or batch_text, required_triplets, current_h2
        )
        all_issues.extend(triplet_issues)
        experts_summary["triplet"] = triplet_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 6️⃣ UNIFIED BRIDGE EXPERT (optional)
    # ═══════════════════════════════════════════════════════════════
    if UNIFIED_VALIDATOR_AVAILABLE:
        try:
            unified_expert = UnifiedBridgeExpert(config)
            main_keyword = project_data.get("main_keyword", "")
            unified_issues, unified_summary = unified_expert.validate(
                batch_text=corrected_text or batch_text,
                keywords_state=keywords_state,
                s1_data=s1_data,
                main_keyword=main_keyword
            )
            existing_codes = {i.code for i in all_issues}
            for issue in unified_issues:
                if issue.code not in existing_codes:
                    all_issues.append(issue)
            experts_summary["unified_bridge"] = unified_summary
        except Exception as e:
            experts_summary["unified_bridge"] = {"enabled": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════
    # 7️⃣ CORPUS INSIGHTS (NIGDY nie blokuje!)
    # ═══════════════════════════════════════════════════════════════
    corpus_insights = None
    naturalness_hints = []
    
    if include_corpus_insights and CORPUS_INSIGHTS_AVAILABLE:
        try:
            corpus_insights = get_corpus_insights_for_moe(corrected_text or batch_text)
            if corpus_insights.get("enabled"):
                naturalness_hints = corpus_insights.get("suggestions", [])
                experts_summary["corpus_insights"] = {
                    "enabled": True,
                    "naturalness_score": corpus_insights.get("naturalness_score", 100),
                    "affects_validation": False,
                }
        except Exception as e:
            corpus_insights = {"enabled": False, "error": str(e)[:100]}
    
    # ═══════════════════════════════════════════════════════════════
    # AGREGACJA WYNIKÓW
    # ═══════════════════════════════════════════════════════════════
    critical_issues = [i for i in all_issues if i.severity == "critical"]
    warning_issues = [i for i in all_issues if i.severity == "warning"]
    
    for issue in all_issues:
        if issue.fix_instruction and issue.severity in ["critical", "warning"]:
            fix_instructions.append(f"[{issue.expert.upper()}] {issue.fix_instruction}")
    
    # Ustal status
    if surgery_applied:
        status = "AUTO_FIXED"
        passed = True
    elif mode == ValidationMode.STRICT and critical_issues:
        status = "REJECTED"
        passed = False
    elif critical_issues:
        status = "WARNING"
        passed = True
    elif warning_issues:
        status = "WARNING"
        passed = True
    else:
        status = "APPROVED"
        passed = True
    
    return ValidationResult(
        passed=passed,
        status=status,
        issues=all_issues,
        experts_summary=experts_summary,
        fix_instructions=fix_instructions[:10],
        auto_fixes_applied=auto_fixes_applied,
        corrected_text=corrected_text,
        corpus_insights=corpus_insights,
        naturalness_hints=naturalness_hints,
        surgery_applied=surgery_applied,
        surgery_stats=surgery_stats
    )


# ================================================================
# HELPER: Format dla GPT
# ================================================================
def format_validation_for_gpt(result: ValidationResult) -> str:
    """Formatuje wynik walidacji jako instrukcje dla GPT."""
    if result.status == "APPROVED":
        return ""
    
    lines = []
    lines.append("=" * 60)
    
    if result.surgery_applied:
        lines.append("✅ AUTO-FIX APPLIED (Content Surgeon)")
        lines.append(f"   Injected: {result.surgery_stats.get('injected', 0)} phrases")
        lines.append("")
    
    lines.append("⚠️ WALIDACJA MOE - WYMAGANE POPRAWKI")
    lines.append("=" * 60)
    
    by_expert = {}
    for issue in result.issues:
        if issue.severity in ["critical", "warning"]:
            by_expert.setdefault(issue.expert, []).append(issue)
    
    for expert, issues in by_expert.items():
        lines.append(f"\n🔍 {expert.upper()}:")
        for issue in issues[:3]:
            severity_icon = "❌" if issue.severity == "critical" else "⚠️"
            lines.append(f"  {severity_icon} {issue.message}")
            if issue.fix_instruction:
                lines.append(f"     → FIX: {issue.fix_instruction}")
    
    if result.naturalness_hints:
        lines.append(f"\n💡 SUGESTIE NATURALNOŚCI:")
        for hint in result.naturalness_hints[:3]:
            lines.append(f"  ℹ️ {hint.get('suggestion', hint.get('message', ''))}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ================================================================
# EXPORTS
# ================================================================
__all__ = [
    'validate_batch_moe',
    'format_validation_for_gpt',
    'ValidationMode',
    'ValidationConfig',
    'ValidationResult',
    'ValidationIssue',
    'CONTENT_SURGEON_AVAILABLE',
    'SEMANTIC_TRIPLET_AVAILABLE'
]
