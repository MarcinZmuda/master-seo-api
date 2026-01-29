"""
===============================================================================
🔍 MOE BATCH VALIDATOR v1.2 - Mixture of Experts Post-Batch Validation
===============================================================================
Kompleksowa walidacja batcha po wygenerowaniu przez GPT.

EKSPERCI (MoE):
1. STRUCTURE EXPERT - różna liczba akapitów, anty-monotonność
2. SEO EXPERT - BASIC/EXTENDED keywords, encje, n-gramy
3. LANGUAGE EXPERT - gramatyka polska (LanguageTool), styl
4. AI DETECTION EXPERT - burstiness, TTR, rozkład zdań
5. UNIFIED BRIDGE EXPERT - mostek do unified_validator (optional)
6. 🆕 CORPUS INSIGHTS - metryki NKJP (informacyjne, nie blokuje!)

🆕 v1.2 ZMIANY:
- Dodano integrację z polish_corpus_metrics_v41 (NKJP insights)
- Corpus insights NIE blokują walidacji - tylko informacyjne
- Dodano pole corpus_insights do ValidationResult
- Dodano naturalness_hints do response

TRYBY:
- SOFT: tylko warnings, batch zapisuje się
- STRICT: critical errors blokują batch
- AUTO_FIX: próbuje naprawić automatycznie

Użycie:
    from moe_batch_validator import validate_batch_moe
    result = validate_batch_moe(batch_text, project_data, batch_number)
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

# 🆕 v1.2: POLISH CORPUS INSIGHTS (optional, NEVER blocks!)
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
    
    # Fallback functions - zwracają puste wyniki, NIGDY nie blokują
    def get_corpus_insights_for_moe(text: str, **kwargs) -> dict:
        return {
            "enabled": False, 
            "affects_validation": False,
            "is_blocking": False,
            "blocks_action": False
        }
    
    def get_naturalness_hints(text: str) -> list:
        return []
    
    def analyze_corpus_metrics(text: str, **kwargs):
        return None


# ================================================================
# KONFIGURACJA
# ================================================================
class ValidationMode(Enum):
    SOFT = "soft"       # Tylko warnings
    STRICT = "strict"   # Critical errors blokują
    AUTO_FIX = "auto_fix"  # Próbuje naprawić


@dataclass
class ValidationConfig:
    """Konfiguracja progów walidacji."""
    
    # STRUKTURA - różnorodność akapitów
    min_paragraph_count: int = 3
    max_paragraph_count: int = 8
    paragraph_variance_min: float = 0.3  # Min różnorodność między sekcjami
    
    # SEO - keywords
    basic_coverage_min: float = 0.7      # 70% BASIC musi być użyte
    entity_coverage_min: float = 0.5     # 50% encji
    
    # JĘZYK - gramatyka
    max_grammar_errors: int = 3          # Max błędów gramatycznych
    max_critical_grammar: int = 1        # Max błędów krytycznych
    
    # AI DETECTION
    burstiness_min: float = 2.2          # CV * 5 >= 2.2 (CV >= 0.44)
    burstiness_max: float = 4.5
    ttr_min: float = 0.42
    ttr_max: float = 0.65
    short_sentence_pct_min: int = 15     # % krótkich zdań
    short_sentence_pct_max: int = 30
    
    # POWTÓRZENIA
    max_word_repetition: int = 6         # Max powtórzeń jednego słowa
    
    # 🆕 v1.2: CORPUS INSIGHTS
    include_corpus_insights: bool = True  # Czy dodawać corpus insights


@dataclass
class ValidationIssue:
    """Pojedynczy problem znaleziony przez walidator."""
    expert: str           # structure, seo, language, ai_detection
    severity: str         # critical, warning, info
    code: str            # np. PARAGRAPH_MONOTONY, MISSING_BASIC
    message: str
    fix_instruction: str = ""
    auto_fixable: bool = False
    context: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Wynik walidacji MoE."""
    passed: bool
    status: str          # APPROVED, WARNING, REJECTED, AUTO_FIXED
    issues: List[ValidationIssue]
    experts_summary: Dict[str, Dict]
    fix_instructions: List[str]
    auto_fixes_applied: List[str] = field(default_factory=list)
    corrected_text: Optional[str] = None
    # 🆕 v1.2: Corpus insights (NIGDY nie wpływa na passed/status!)
    corpus_insights: Optional[Dict] = None
    naturalness_hints: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {
            "passed": self.passed,
            "status": self.status,
            "issues": [asdict(i) for i in self.issues],
            "experts_summary": self.experts_summary,
            "fix_instructions": self.fix_instructions,
            "auto_fixes_applied": self.auto_fixes_applied,
            "has_corrected_text": self.corrected_text is not None
        }
        
        # 🆕 v1.2: Dodaj corpus insights (jeśli dostępne)
        if self.corpus_insights:
            result["corpus_insights"] = self.corpus_insights
        if self.naturalness_hints:
            result["naturalness_hints"] = self.naturalness_hints
            
        return result


# ================================================================
# 1️⃣ STRUCTURE EXPERT - Różnorodność struktury
# ================================================================
class StructureExpert:
    """
    Sprawdza strukturę tekstu:
    - Różna liczba akapitów między sekcjami H2
    - Różna długość akapitów
    - Anty-monotonność
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(
        self, 
        batch_text: str, 
        previous_batches: List[Dict],
        current_h2: str = ""
    ) -> Tuple[List[ValidationIssue], Dict]:
        """Walidacja struktury batcha."""
        issues = []
        
        # Policz akapity w bieżącym batchu
        paragraphs = self._split_paragraphs(batch_text)
        current_para_count = len(paragraphs)
        
        # Policz akapity w poprzednich batchach
        previous_para_counts = []
        for batch in previous_batches:
            text = batch.get("text", "")
            prev_paras = self._split_paragraphs(text)
            previous_para_counts.append(len(prev_paras))
        
        # Sprawdź czy liczba akapitów jest inna niż w poprzednim batchu
        if previous_para_counts:
            last_para_count = previous_para_counts[-1]
            
            # CRITICAL: Identyczna liczba akapitów jak poprzedni batch
            if current_para_count == last_para_count and current_para_count > 2:
                issues.append(ValidationIssue(
                    expert="structure",
                    severity="warning",
                    code="PARAGRAPH_MONOTONY",
                    message=f"Ta sekcja ma {current_para_count} akapitów - IDENTYCZNIE jak poprzednia! "
                            f"Zmień strukturę dla naturalności.",
                    fix_instruction=f"Zmień liczbę akapitów z {current_para_count} na {current_para_count + 1} lub {max(2, current_para_count - 1)}. "
                                   f"Możesz: połączyć 2 krótkie akapity w 1 dłuższy LUB podzielić 1 długi na 2 krótsze.",
                    context={
                        "current_count": current_para_count,
                        "previous_count": last_para_count
                    }
                ))
            
            # Sprawdź ostatnie 3 batche - czy nie są monotonne
            if len(previous_para_counts) >= 2:
                last_three = previous_para_counts[-2:] + [current_para_count]
                if len(set(last_three)) == 1:  # Wszystkie identyczne
                    issues.append(ValidationIssue(
                        expert="structure",
                        severity="critical",
                        code="STRUCTURE_PATTERN_DETECTED",
                        message=f"Ostatnie 3 sekcje mają identyczną strukturę ({last_three[0]} akapitów). "
                                f"AI pattern detected!",
                        fix_instruction="MUSISZ zmienić liczbę akapitów w tej sekcji. "
                                       f"Użyj {last_three[0] + 2} lub {max(2, last_three[0] - 1)} akapitów.",
                        context={"pattern": last_three}
                    ))
        
        # Sprawdź długości akapitów (różnorodność)
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
                    message=f"Akapity są zbyt podobnej długości (CV={cv:.2f}). "
                            f"Długości: {para_lengths}",
                    fix_instruction="Zróżnicuj długość akapitów. Niektóre powinny być krótkie (2-3 zdania), "
                                   "inne dłuższe (5-6 zdań).",
                    context={"lengths": para_lengths, "cv": cv}
                ))
        
        # Sprawdź minimalną i maksymalną liczbę akapitów
        if current_para_count < self.config.min_paragraph_count:
            issues.append(ValidationIssue(
                expert="structure",
                severity="warning",
                code="TOO_FEW_PARAGRAPHS",
                message=f"Za mało akapitów: {current_para_count} (min: {self.config.min_paragraph_count})",
                fix_instruction=f"Podziel tekst na więcej akapitów (min {self.config.min_paragraph_count})."
            ))
        
        if current_para_count > self.config.max_paragraph_count:
            issues.append(ValidationIssue(
                expert="structure",
                severity="info",
                code="MANY_PARAGRAPHS",
                message=f"Dużo akapitów: {current_para_count} (sugerowane max: {self.config.max_paragraph_count})",
                fix_instruction="Rozważ połączenie niektórych krótkich akapitów."
            ))
        
        summary = {
            "paragraph_count": current_para_count,
            "paragraph_lengths": para_lengths,
            "previous_counts": previous_para_counts,
            "issues_count": len(issues)
        }
        
        return issues, summary
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Dzieli tekst na akapity."""
        # Usuń HTML
        clean = re.sub(r'<[^>]+>', '\n', text)
        # Podziel na podwójne newline
        paragraphs = re.split(r'\n\s*\n', clean)
        # Filtruj puste
        return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


# ================================================================
# 2️⃣ SEO EXPERT - Keywords i encje
# ================================================================
class SEOExpert:
    """
    Sprawdza SEO:
    - BASIC keywords coverage
    - EXTENDED keywords (info)
    - Encje z S1
    - N-gramy
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(
        self,
        batch_text: str,
        keywords_state: Dict,
        batch_counts: Dict,
        s1_entities: List = None,
        assigned_keywords: List[str] = None,
        batch_number: int = 1
    ) -> Tuple[List[ValidationIssue], Dict]:
        """Walidacja SEO batcha."""
        issues = []
        
        # === BASIC KEYWORDS ===
        basic_missing = []
        basic_used = []
        basic_exceeded = []
        
        for rid, meta in keywords_state.items():
            if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
                continue
            
            keyword = meta.get("keyword", "")
            target_min = meta.get("target_min", 1)
            target_max = meta.get("target_max", 999)
            actual = meta.get("actual_uses", 0)
            batch_use = batch_counts.get(rid, 0)
            
            # Frazy PRZYPISANE do tego batcha które nie zostały użyte
            if assigned_keywords and keyword.lower() in [k.lower() for k in assigned_keywords]:
                if batch_use == 0:
                    synonyms = get_synonyms(keyword) if SYNONYMS_AVAILABLE else []
                    issues.append(ValidationIssue(
                        expert="seo",
                        severity="warning",
                        code="ASSIGNED_KEYWORD_MISSING",
                        message=f"Fraza '{keyword}' była PRZYPISANA do tej sekcji, ale nie została użyta!",
                        fix_instruction=f"Dodaj frazę '{keyword}' do tekstu (min 1x). "
                                       f"Synonimy: {', '.join(synonyms[:2]) if synonyms else 'brak'}",
                        context={"keyword": keyword, "assigned": True, "synonyms": synonyms}
                    ))
                    basic_missing.append(keyword)
            
            # Frazy które są UNDER i powinny być użyte
            if actual < target_min and batch_use == 0:
                remaining_batches = 7 - batch_number  # Zakładamy 7 batchów
                if remaining_batches <= 2 and actual == 0:
                    issues.append(ValidationIssue(
                        expert="seo",
                        severity="critical" if meta.get("type") == "MAIN" else "warning",
                        code="BASIC_STILL_ZERO",
                        message=f"BASIC '{keyword}' ma 0 użyć! Zostały {remaining_batches} batche.",
                        fix_instruction=f"MUSISZ użyć '{keyword}' w tym lub następnym batchu (cel: {target_min}-{target_max}x)."
                    ))
                    basic_missing.append(keyword)
            
            if batch_use > 0:
                basic_used.append({"keyword": keyword, "count": batch_use})
        
        # === ENCJE ===
        coverage = 0
        entities_missing = []
        if s1_entities:
            text_lower = batch_text.lower()
            entities_in_batch = []
            
            for entity in s1_entities[:15]:  # Top 15 encji
                name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                if name.lower() in text_lower:
                    entities_in_batch.append(name)
                else:
                    entities_missing.append(name)
            
            coverage = len(entities_in_batch) / len(s1_entities[:15]) if s1_entities else 0
            
            if coverage < 0.2 and batch_number <= 3:  # Pierwsze 3 batche powinny mieć więcej encji
                issues.append(ValidationIssue(
                    expert="seo",
                    severity="warning",
                    code="LOW_ENTITY_COVERAGE",
                    message=f"Niskie pokrycie encji: {coverage:.0%}. Użyte: {len(entities_in_batch)}/{len(s1_entities[:15])}",
                    fix_instruction=f"Dodaj wzmianki o: {', '.join(entities_missing[:3])}",
                    context={"entities_missing": entities_missing[:5]}
                ))
        
        summary = {
            "basic_used": basic_used,
            "basic_missing": basic_missing,
            "entity_coverage": coverage if s1_entities else None,
            "issues_count": len(issues)
        }
        
        return issues, summary


# ================================================================
# 3️⃣ LANGUAGE EXPERT - Gramatyka polska
# ================================================================
class LanguageExpert:
    """
    Sprawdza język polski:
    - LanguageTool (błędy gramatyczne)
    - Zgodność przypadków
    - Interpunkcja
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, batch_text: str) -> Tuple[List[ValidationIssue], Dict]:
        """Walidacja języka polskiego."""
        issues = []
        grammar_errors = []
        
        if LANGUAGETOOL_AVAILABLE:
            try:
                lt_result = validate_batch_grammar(batch_text)
                grammar_errors = lt_result.errors if hasattr(lt_result, 'errors') else []
                critical_errors = [e for e in grammar_errors if e.get("rule", {}).get("category", {}).get("id") in ["GRAMMAR", "TYPOS"]] if grammar_errors else []
                
                # Critical grammar errors
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
                
                # Warning grammar errors
                for err in grammar_errors[len(critical_errors):self.config.max_grammar_errors + 1]:
                    issues.append(ValidationIssue(
                        expert="language",
                        severity="warning",
                        code="GRAMMAR_WARNING",
                        message=f"Sugestia: {err.get('message', 'nieznany')}",
                        fix_instruction=f"Rozważ zmianę: '{err.get('context', '')}'",
                        context=err
                    ))
                    
            except Exception as e:
                print(f"[MOE_VALIDATOR] LanguageTool error: {e}")
        
        # Dodatkowe sprawdzenia polskiego
        polish_issues = self._check_polish_rules(batch_text)
        issues.extend(polish_issues)
        
        summary = {
            "grammar_errors": len(grammar_errors),
            "critical_errors": len([i for i in issues if i.severity == "critical"]),
            "languagetool_available": LANGUAGETOOL_AVAILABLE
        }
        
        return issues, summary
    
    def _check_polish_rules(self, text: str) -> List[ValidationIssue]:
        """Dodatkowe reguły polskie."""
        issues = []
        
        # Sprawdź częste błędy
        common_errors = [
            (r'\bw między\b', 'w między → wśród/między', 'Błędna konstrukcja "w między"'),
            (r'\bw skutek\b', 'w skutek → wskutek', 'Pisownia łączna "wskutek"'),
            (r'\bz pod\b', 'z pod → spod', 'Pisownia łączna "spod"'),
            (r'\bz\s+nad\b', 'z nad → znad', 'Pisownia łączna "znad"'),
            (r'\bpo mimo\b', 'po mimo → pomimo', 'Pisownia łączna "pomimo"'),
        ]
        
        text_lower = text.lower()
        for pattern, fix, msg in common_errors:
            if re.search(pattern, text_lower):
                issues.append(ValidationIssue(
                    expert="language",
                    severity="warning",
                    code="POLISH_SPELLING",
                    message=msg,
                    fix_instruction=fix,
                    auto_fixable=True
                ))
        
        return issues


# ================================================================
# 4️⃣ AI DETECTION EXPERT - Wykrywanie AI patterns
# ================================================================
class AIDetectionExpert:
    """
    Sprawdza metryki anty-AI:
    - Burstiness (zmienność zdań)
    - TTR (bogactwo słownictwa)
    - Rozkład długości zdań
    - Powtórzenia słów
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, batch_text: str) -> Tuple[List[ValidationIssue], Dict]:
        """Walidacja anty-AI."""
        issues = []
        metrics = {}
        
        if not AI_METRICS_AVAILABLE:
            return issues, {"available": False}
        
        # === BURSTINESS ===
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
                    fix_instruction="Dodaj więcej KRÓTKICH zdań (2-8 słów). "
                                   "Np. 'To ważne.', 'Warto wiedzieć.', 'Co to oznacza?'",
                    context=burstiness_result
                ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Burstiness error: {e}")
        
        # === ROZKŁAD ZDAŃ ===
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
                    message=f"Za mało krótkich zdań: {short_pct}% (cel: {self.config.short_sentence_pct_min}-{self.config.short_sentence_pct_max}%)",
                    fix_instruction="Dodaj zdania 2-10 słów. Przykłady: 'To kluczowe.', 'Warto pamiętać.', 'Jak to działa?'",
                    context=dist_result
                ))
            
            # Wykryj AI concentration (60%+ w przedziale 15-22)
            ai_concentration = dist_result.get("ai_concentration", 0)
            if ai_concentration > 60:
                issues.append(ValidationIssue(
                    expert="ai_detection",
                    severity="critical",
                    code="AI_SENTENCE_PATTERN",
                    message=f"AI pattern: {ai_concentration:.0f}% zdań ma 15-22 słów (monotonna długość)",
                    fix_instruction="Zróżnicuj długość zdań! Dodaj krótkie (5-8 słów) i długie (25-30 słów).",
                    context={"ai_concentration": ai_concentration}
                ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Sentence distribution error: {e}")
        
        # === POWTÓRZENIA SŁÓW ===
        try:
            repetition_result = check_word_repetition_detailed(batch_text)
            repeated_words = repetition_result.get("repeated_words", [])
            metrics["repeated_words"] = repeated_words[:5]
            
            for word_info in repeated_words[:3]:
                word = word_info.get("word", "")
                count = word_info.get("count", 0)
                if count > self.config.max_word_repetition:
                    synonyms = word_info.get("synonyms", [])
                    issues.append(ValidationIssue(
                        expert="ai_detection",
                        severity="warning",
                        code="WORD_REPETITION",
                        message=f"Słowo '{word}' powtórzone {count}x",
                        fix_instruction=f"Zamień niektóre '{word}' na: {', '.join(synonyms[:3]) if synonyms else 'synonimy'}",
                        auto_fixable=bool(synonyms),
                        context=word_info
                    ))
        except Exception as e:
            print(f"[MOE_VALIDATOR] Repetition check error: {e}")
        
        summary = {
            "metrics": metrics,
            "issues_count": len(issues)
        }
        
        return issues, summary


# ================================================================
# 5️⃣ UNIFIED BRIDGE EXPERT (optional)
# ================================================================
class UnifiedBridgeExpert:
    """
    Ekspert mostkowy do unified_validator.
    
    Nie duplikuje logiki - WYWOŁUJE funkcje z unified_validator
    i tłumaczy wyniki na format MoE.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.enabled = UNIFIED_VALIDATOR_AVAILABLE
    
    def validate(
        self, 
        batch_text: str,
        keywords_state: Dict,
        s1_data: Dict = None,
        main_keyword: str = ""
    ) -> Tuple[List[ValidationIssue], Dict]:
        """Waliduje batch używając unified_validator."""
        issues = []
        metrics = {
            "enabled": self.enabled,
            "checks_performed": []
        }
        
        if not self.enabled:
            return issues, {"enabled": False, "reason": "unified_validator not available"}
        
        try:
            # 1. Quick validate
            quick_result = quick_validate(batch_text, keywords_state)
            metrics["quick_score"] = quick_result.get("score", 0)
            metrics["checks_performed"].append("quick_validate")
            
            # Konwertuj issues z quick_validate
            for issue in quick_result.get("issues", []):
                if isinstance(issue, dict):
                    severity = issue.get("severity", "WARNING")
                else:
                    severity = getattr(issue, "severity", "WARNING")
                    if hasattr(severity, "value"):
                        severity = severity.value
                
                issues.append(ValidationIssue(
                    expert="unified_bridge",
                    severity=severity.lower() if severity != "CRITICAL" else "critical",
                    code=issue.get("code", "UNIFIED_ISSUE") if isinstance(issue, dict) else getattr(issue, "code", "UNIFIED_ISSUE"),
                    message=issue.get("message", str(issue)) if isinstance(issue, dict) else getattr(issue, "message", str(issue)),
                    fix_instruction="",
                    context={"source": "unified_validator.quick_validate"}
                ))
            
            # 2. Entity density
            if s1_data:
                entities = s1_data.get("entity_seo", {}).get("entities", [])
                density_result = calculate_entity_density(batch_text, entities)
                metrics["entity_density"] = density_result.get("density", 0)
                metrics["checks_performed"].append("entity_density")
                
                if density_result.get("status") == "CRITICAL":
                    issues.append(ValidationIssue(
                        expert="unified_bridge",
                        severity="warning",
                        code="ENTITY_DENSITY_LOW",
                        message=f"Niska gęstość encji: {density_result.get('density', 0):.2f}",
                        fix_instruction="Dodaj więcej encji z S1 Analysis",
                        context=density_result
                    ))
            
            # 3. Template patterns
            template_issues = check_template_patterns(batch_text)
            metrics["template_patterns_found"] = len(template_issues)
            metrics["checks_performed"].append("template_patterns")
            
            for t_issue in template_issues[:2]:
                issues.append(ValidationIssue(
                    expert="unified_bridge",
                    severity="warning",
                    code="TEMPLATE_PATTERN",
                    message=t_issue.message if hasattr(t_issue, "message") else str(t_issue),
                    fix_instruction="Przepisz fragment unikając powtarzalnych struktur",
                    context={"source": "unified_validator.check_template_patterns"}
                ))
            
            # 4. Semantic enhancement
            if s1_data:
                semantic_result = validate_semantic_enhancement(batch_text, s1_data)
                metrics["semantic_score"] = semantic_result.get("score", 0)
                metrics["checks_performed"].append("semantic_enhancement")
                
                if semantic_result.get("score", 100) < 50:
                    issues.append(ValidationIssue(
                        expert="unified_bridge",
                        severity="info",
                        code="SEMANTIC_LOW",
                        message=f"Niski semantic score: {semantic_result.get('score', 0)}",
                        fix_instruction="Wzmocnij powiązania semantyczne z encjami S1",
                        context=semantic_result
                    ))
            
        except Exception as e:
            print(f"[MOE_VALIDATOR] UnifiedBridgeExpert error: {e}")
            metrics["error"] = str(e)
        
        summary = {
            "enabled": True,
            "metrics": metrics,
            "issues_count": len(issues)
        }
        
        return issues, summary


# ================================================================
# 🎯 GŁÓWNA FUNKCJA WALIDACJI MOE
# ================================================================
def validate_batch_moe(
    batch_text: str,
    project_data: Dict,
    batch_number: int = 1,
    mode: ValidationMode = ValidationMode.SOFT,
    config: Optional[ValidationConfig] = None,
    include_corpus_insights: bool = True  # 🆕 v1.2
) -> ValidationResult:
    """
    Główna funkcja walidacji MoE.
    
    Args:
        batch_text: Tekst batcha do walidacji
        project_data: Dane projektu (keywords_state, batches, s1_data, etc.)
        batch_number: Numer batcha (1-7)
        mode: Tryb walidacji (SOFT, STRICT, AUTO_FIX)
        config: Opcjonalna konfiguracja (domyślna jeśli None)
        include_corpus_insights: Czy dodać corpus insights (default: True)
    
    Returns:
        ValidationResult z wynikami wszystkich ekspertów
    """
    if config is None:
        config = ValidationConfig()
    
    all_issues = []
    experts_summary = {}
    fix_instructions = []
    
    # Pobierz dane z projektu
    keywords_state = project_data.get("keywords_state", {})
    previous_batches = project_data.get("batches", [])
    s1_data = project_data.get("s1_data", {})
    s1_entities = s1_data.get("entity_seo", {}).get("entities", [])
    semantic_plan = project_data.get("semantic_keyword_plan", {})
    
    # Pobierz assigned keywords dla tego batcha
    assigned_keywords = []
    for bp in semantic_plan.get("batch_plans", []):
        if bp.get("batch_number") == batch_number:
            assigned_keywords = bp.get("assigned_keywords", [])
            break
    
    # Policz keywords w batchu
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
    current_h2 = ""
    structure_issues, structure_summary = structure_expert.validate(
        batch_text, previous_batches, current_h2
    )
    all_issues.extend(structure_issues)
    experts_summary["structure"] = structure_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 2️⃣ SEO EXPERT
    # ═══════════════════════════════════════════════════════════════
    seo_expert = SEOExpert(config)
    seo_issues, seo_summary = seo_expert.validate(
        batch_text, keywords_state, batch_counts, s1_entities, assigned_keywords, batch_number
    )
    all_issues.extend(seo_issues)
    experts_summary["seo"] = seo_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 3️⃣ LANGUAGE EXPERT
    # ═══════════════════════════════════════════════════════════════
    language_expert = LanguageExpert(config)
    language_issues, language_summary = language_expert.validate(batch_text)
    all_issues.extend(language_issues)
    experts_summary["language"] = language_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 4️⃣ AI DETECTION EXPERT
    # ═══════════════════════════════════════════════════════════════
    ai_expert = AIDetectionExpert(config)
    ai_issues, ai_summary = ai_expert.validate(batch_text)
    all_issues.extend(ai_issues)
    experts_summary["ai_detection"] = ai_summary
    
    # ═══════════════════════════════════════════════════════════════
    # 5️⃣ UNIFIED BRIDGE EXPERT (optional)
    # ═══════════════════════════════════════════════════════════════
    if UNIFIED_VALIDATOR_AVAILABLE:
        try:
            unified_expert = UnifiedBridgeExpert(config)
            main_keyword = project_data.get("main_keyword", "")
            unified_issues, unified_summary = unified_expert.validate(
                batch_text=batch_text,
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
            print(f"[MOE_VALIDATOR] UnifiedBridgeExpert skipped: {e}")
            experts_summary["unified_bridge"] = {"enabled": False, "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════
    # 🆕 v1.2: 6️⃣ CORPUS INSIGHTS (NIGDY nie blokuje!)
    # ═══════════════════════════════════════════════════════════════
    corpus_insights = None
    naturalness_hints = []
    
    if include_corpus_insights and CORPUS_INSIGHTS_AVAILABLE:
        try:
            corpus_insights = get_corpus_insights_for_moe(batch_text)
            
            # Wyodrębnij hints do osobnego pola
            if corpus_insights.get("enabled"):
                naturalness_hints = corpus_insights.get("suggestions", [])
                
                # Dodaj do summary (ale NIE do issues!)
                experts_summary["corpus_insights"] = {
                    "enabled": True,
                    "naturalness_score": corpus_insights.get("naturalness_score", 100),
                    "style_detected": corpus_insights.get("style_detected", "unknown"),
                    "suggestions_count": len(naturalness_hints),
                    # WAŻNE: Jawne oznaczenie że NIE wpływa na walidację
                    "affects_validation": False,
                }
        except Exception as e:
            print(f"[MOE_VALIDATOR] Corpus insights error (non-blocking): {e}")
            corpus_insights = {
                "enabled": False,
                "error": str(e)[:100],
                "affects_validation": False
            }
    
    # ═══════════════════════════════════════════════════════════════
    # AGREGACJA WYNIKÓW
    # ═══════════════════════════════════════════════════════════════
    critical_issues = [i for i in all_issues if i.severity == "critical"]
    warning_issues = [i for i in all_issues if i.severity == "warning"]
    
    # Zbierz fix instructions
    for issue in all_issues:
        if issue.fix_instruction and issue.severity in ["critical", "warning"]:
            fix_instructions.append(f"[{issue.expert.upper()}] {issue.fix_instruction}")
    
    # Ustal status (corpus insights NIGDY nie wpływa na status!)
    if mode == ValidationMode.STRICT and critical_issues:
        status = "REJECTED"
        passed = False
    elif critical_issues:
        status = "WARNING"
        passed = True  # W SOFT mode przepuszczamy z warning
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
        auto_fixes_applied=[],
        corrected_text=None,
        corpus_insights=corpus_insights,
        naturalness_hints=naturalness_hints
    )


# ================================================================
# HELPER: Format dla GPT
# ================================================================
def format_validation_for_gpt(result: ValidationResult) -> str:
    """
    Formatuje wynik walidacji jako instrukcje dla GPT do przepisania batcha.
    """
    if result.status == "APPROVED":
        return ""
    
    lines = []
    lines.append("=" * 60)
    lines.append("⚠️ WALIDACJA MOE - WYMAGANE POPRAWKI")
    lines.append("=" * 60)
    
    # Pogrupuj po ekspertach
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
    
    # 🆕 v1.2: Dodaj naturalness hints (informacyjnie)
    if result.naturalness_hints:
        lines.append(f"\n💡 SUGESTIE NATURALNOŚCI (informacyjne):")
        for hint in result.naturalness_hints[:3]:
            lines.append(f"  ℹ️ [{hint.get('metric', '?')}] {hint.get('suggestion', hint.get('message', ''))}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


# ================================================================
# 🆕 v1.2: HELPER - Extract naturalness suggestions
# ================================================================
def extract_naturalness_suggestions(result: ValidationResult) -> List[Dict]:
    """
    Wyciąga sugestie naturalności z corpus_insights.
    
    Użyteczne do dodania do response API.
    """
    if not result.corpus_insights:
        return []
    
    if not result.corpus_insights.get("enabled"):
        return []
    
    return result.corpus_insights.get("suggestions", [])


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    test_text = """
    Ubezwłasnowolnienie to ważna instytucja prawa cywilnego. Sąd może orzec ubezwłasnowolnienie całkowite lub częściowe.
    
    Wniosek o ubezwłasnowolnienie składa się do sądu okręgowego. Postępowanie wymaga opinii biegłego psychiatry.
    
    Skutki ubezwłasnowolnienia są poważne. Osoba ubezwłasnowolniona traci zdolność do czynności prawnych.
    """
    
    # Symulacja project_data
    project_data = {
        "keywords_state": {
            "k1": {"keyword": "ubezwłasnowolnienie", "type": "MAIN", "target_min": 5, "target_max": 15, "actual_uses": 2},
            "k2": {"keyword": "sąd", "type": "BASIC", "target_min": 3, "target_max": 10, "actual_uses": 1},
        },
        "batches": [],
        "s1_data": {"entity_seo": {"entities": [{"name": "Kodeks cywilny"}, {"name": "sąd okręgowy"}]}}
    }
    
    result = validate_batch_moe(test_text, project_data, batch_number=1)
    
    print(f"Status: {result.status}")
    print(f"Passed: {result.passed}")
    print(f"Issues: {len(result.issues)}")
    for issue in result.issues:
        print(f"  [{issue.expert}] {issue.severity}: {issue.message}")
    
    # 🆕 v1.2: Pokaż corpus insights
    if result.corpus_insights and result.corpus_insights.get("enabled"):
        print(f"\n📊 Corpus Insights:")
        print(f"   Naturalness: {result.corpus_insights.get('naturalness_score', 'N/A')}")
        print(f"   Style: {result.corpus_insights.get('style_detected', 'N/A')}")
        print(f"   Suggestions: {len(result.naturalness_hints)}")
        for hint in result.naturalness_hints:
            print(f"     💡 {hint.get('metric')}: {hint.get('suggestion', hint.get('message', ''))}")
    
    print("\n" + format_validation_for_gpt(result))
