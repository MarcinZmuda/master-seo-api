"""
===============================================================================
🎯 QUALITY SCORE MODULE v37.4 - BRAJEN SEO Engine
===============================================================================

ZMIANY v37.4:
- 🔧 Grammar YMYL: ≥2 błędy → CRITICAL + cap grade C
- 🔧 Semantic: konfigurowalne progi (SEMANTIC_THRESHOLDS)
- 🆕 Decision trace: audit trail dla decyzji
- 🔧 Config-driven thresholds (przygotowanie pod v38)

ZMIANY v37.3:
- Poprawiona logika QualityGrade (sorted descending)
- Keywords: rozdzielenie score od decyzji
- Humanness: usunięte magiczne /5.0
- Structure: word count per batch_role
- Action response: confidence
- Fast mode: division by zero fix

Autor: BRAJEN SEO Engine
===============================================================================
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ================================================================
# CONFIGURATION - Config-driven thresholds (v38 ready)
# ================================================================

@dataclass
class QualityConfig:
    """Konfigurowalne progi jakości."""
    
    # Semantic thresholds
    semantic_cap_B: float = 0.25       # Poniżej → max grade B
    semantic_cap_B_plus: float = 0.35  # Poniżej → max grade B+
    
    # Humanness (burstiness CV)
    cv_critical: float = 0.26   # Poniżej → AI detected
    cv_warning: float = 0.36    # Poniżej → monotonny
    cv_good: float = 0.44       # Powyżej → naturalny
    
    # AI patterns
    ai_patterns_critical: int = 5
    ai_patterns_warning: int = 3
    
    # Grammar
    grammar_errors_critical_ymyl: int = 2   # YMYL: ≥2 → CRITICAL
    grammar_errors_warning_ymyl: int = 1    # YMYL: ≥1 → WARNING
    grammar_errors_critical_normal: int = 5  # non-YMYL: ≥5 → WARNING
    grammar_errors_warning_normal: int = 2   # non-YMYL: ≥2 → minor
    
    # Keywords exceeded
    exceeded_critical_threshold: int = 50  # % powyżej którego CRITICAL
    
    # Grade boundaries
    grade_A_plus: int = 95
    grade_A: int = 90
    grade_B_plus: int = 80
    grade_B: int = 70
    grade_C: int = 60
    grade_D: int = 40


# Domyślna konfiguracja
CONFIG = QualityConfig()


# ================================================================
# QUALITY GRADES
# ================================================================

def get_grade_thresholds(config: QualityConfig = CONFIG) -> List[Tuple[str, int, str]]:
    """Zwraca progi grade'ów posortowane malejąco."""
    return [
        ("A+", config.grade_A_plus, "EXCELLENT"),
        ("A", config.grade_A, "EXCELLENT"),
        ("B+", config.grade_B_plus, "GOOD"),
        ("B", config.grade_B, "OK"),
        ("C", config.grade_C, "NEEDS_WORK"),
        ("D", config.grade_D, "POOR"),
        ("F", 0, "REJECTED"),
    ]


def get_grade(score: int, max_grade: Optional[str] = None, config: QualityConfig = CONFIG) -> Tuple[str, str]:
    """
    Zwraca (grade_letter, status) na podstawie score.
    
    Args:
        score: 0-100
        max_grade: Opcjonalny cap na grade
        config: Konfiguracja progów
    """
    thresholds = get_grade_thresholds(config)
    grade_letter, grade_status = "F", "REJECTED"
    
    for letter, min_score, status in thresholds:
        if score >= min_score:
            grade_letter = letter
            grade_status = status
            break
    
    # Zastosuj cap
    if max_grade:
        cap_order = [g[0] for g in thresholds]
        
        if grade_letter in cap_order and max_grade in cap_order:
            current_idx = cap_order.index(grade_letter)
            cap_idx = cap_order.index(max_grade)
            
            if current_idx < cap_idx:
                grade_letter = max_grade
                for letter, _, status in thresholds:
                    if letter == max_grade:
                        grade_status = status
                        break
    
    return grade_letter, grade_status


# ================================================================
# WAGI KOMPONENTÓW
# ================================================================

QUALITY_WEIGHTS = {
    "keywords": 30,
    "humanness": 25,
    "grammar": 15,
    "structure": 10,
    "semantic": 10,
    "coherence": 10
}

WORD_COUNT_RANGES = {
    "INTRO": (200, 400),
    "DEFINITION": (250, 450),
    "CONTENT": (400, 650),
    "PROCEDURE": (400, 700),
    "FINAL": (300, 500),
    "DEFAULT": (350, 600)
}


# ================================================================
# GLOBAL QUALITY SCORE
# ================================================================

def calculate_global_quality_score(
    validation_results: Dict,
    project_data: Optional[Dict] = None,
    config: QualityConfig = CONFIG
) -> Dict:
    """
    Oblicza globalny score 0-100 z wagami.
    
    v37.4:
    - Grammar YMYL: ≥2 → CRITICAL + cap C
    - Semantic: wielopoziomowe capy (0.25 → B, 0.35 → B+)
    - Decision trace
    """
    project_data = project_data or {}
    scores = {}
    issues = []
    decision_trace = []  # 🆕 Audit trail
    max_grade = None
    
    is_ymyl = project_data.get("is_ymyl", False) or project_data.get("is_legal", False)
    batch_role = validation_results.get("batch_role", "CONTENT").upper()
    
    if is_ymyl:
        decision_trace.append("context: YMYL content detected")
    
    # ================================================================
    # 1. KEYWORDS (30%)
    # ================================================================
    exceeded_critical = validation_results.get("exceeded_critical", [])
    exceeded_warning = validation_results.get("exceeded_warning", [])
    
    if exceeded_critical:
        penalty = min(90, 50 + len(exceeded_critical) * 10)
        scores["keywords"] = max(10, 100 - penalty)
        
        for exc in exceeded_critical[:3]:
            kw = exc.get('keyword', '?')
            pct = exc.get('exceeded_percent', 50)
            decision_trace.append(f"exceeded_critical: '{kw}' +{pct}%")
            issues.append({
                "severity": "CRITICAL",
                "component": "keywords",
                "message": f"'{kw}' exceeded {pct}%",
                "fix": f"Use synonyms: {', '.join(exc.get('synonyms', [])[:3]) or 'rephrase'}"
            })
    elif exceeded_warning:
        penalty = min(40, len(exceeded_warning) * 10)
        scores["keywords"] = 100 - penalty
        decision_trace.append(f"exceeded_warning: {len(exceeded_warning)} keyword(s)")
        
        for exc in exceeded_warning[:2]:
            issues.append({
                "severity": "WARNING",
                "component": "keywords",
                "message": f"'{exc.get('keyword')}' slightly exceeded",
                "fix": f"Use synonyms: {', '.join(exc.get('synonyms', [])[:3]) or 'rephrase'}"
            })
    else:
        scores["keywords"] = 100
    
    # ================================================================
    # 2. HUMANNESS (25%)
    # ================================================================
    cv = validation_results.get("burstiness_cv")
    if cv is None:
        burstiness_raw = validation_results.get("burstiness", 0)
        if isinstance(burstiness_raw, dict):
            cv = burstiness_raw.get("cv", 0.4)
        else:
            cv = 0.4
    
    moe = validation_results.get("moe_validation", {})
    style_expert = moe.get("experts_summary", {}).get("style", {}) if isinstance(moe, dict) else {}
    ai_patterns_count = style_expert.get("ai_patterns_count", 0)
    
    if cv < config.cv_critical or ai_patterns_count >= config.ai_patterns_critical:
        scores["humanness"] = 20
        decision_trace.append(f"humanness: CRITICAL (CV={cv:.2f}, patterns={ai_patterns_count})")
        issues.append({
            "severity": "CRITICAL",
            "component": "humanness",
            "message": f"AI-like text (CV={cv:.2f}, {ai_patterns_count} patterns)",
            "fix": "Add short sentences (2-10 words), remove 'Warto zauważyć' etc."
        })
    elif cv < config.cv_warning or ai_patterns_count >= config.ai_patterns_warning:
        scores["humanness"] = 50
        decision_trace.append(f"humanness: WARNING (CV={cv:.2f})")
        issues.append({
            "severity": "WARNING",
            "component": "humanness",
            "message": f"Text could be more natural (CV={cv:.2f})",
            "fix": "Vary sentence length, remove formulaic phrases"
        })
    elif cv < config.cv_good:
        scores["humanness"] = 75
    else:
        humanness_raw = validation_results.get("humanness_score", 80)
        scores["humanness"] = min(100, max(80, humanness_raw))
    
    # ================================================================
    # 3. GRAMMAR (15%)
    # v37.4: YMYL ≥2 błędy → CRITICAL + cap C
    # ================================================================
    grammar_expert = moe.get("experts_summary", {}).get("grammar", {}) if isinstance(moe, dict) else {}
    grammar_errors = grammar_expert.get("error_count", 0)
    
    if is_ymyl:
        if grammar_errors >= config.grammar_errors_critical_ymyl:
            scores["grammar"] = 30
            # 🆕 v37.4: Cap grade na C przy ≥2 błędach YMYL
            if max_grade is None or max_grade in ["A+", "A", "B+", "B"]:
                max_grade = "C"
                decision_trace.append(f"grammar: CRITICAL YMYL ({grammar_errors} errors) → cap grade C")
            issues.append({
                "severity": "CRITICAL",
                "component": "grammar",
                "message": f"{grammar_errors} grammar errors in YMYL content",
                "fix": "Fix ALL grammar errors - critical for legal/medical credibility"
            })
        elif grammar_errors >= config.grammar_errors_warning_ymyl:
            scores["grammar"] = 60
            decision_trace.append(f"grammar: WARNING YMYL ({grammar_errors} error)")
            issues.append({
                "severity": "WARNING",
                "component": "grammar",
                "message": f"{grammar_errors} grammar error(s) in YMYL",
                "fix": "Fix grammar errors for credibility"
            })
        else:
            scores["grammar"] = 100
    else:
        if grammar_errors >= config.grammar_errors_critical_normal:
            scores["grammar"] = 40
            issues.append({
                "severity": "WARNING",
                "component": "grammar",
                "message": f"{grammar_errors} grammar errors",
                "fix": "Fix grammar and punctuation"
            })
        elif grammar_errors >= config.grammar_errors_warning_normal:
            scores["grammar"] = 70
        else:
            scores["grammar"] = 100
    
    # ================================================================
    # 4. STRUCTURE (10%)
    # ================================================================
    struct_valid = validation_results.get("structure_valid", True)
    has_h2 = validation_results.get("has_h2", True)
    batch_text = validation_results.get("batch_text", "")
    word_count = len(batch_text.split()) if batch_text else 0
    
    min_words, max_words = WORD_COUNT_RANGES.get(batch_role, WORD_COUNT_RANGES["DEFAULT"])
    
    if not struct_valid or not has_h2:
        scores["structure"] = 40
        decision_trace.append("structure: missing H2")
        issues.append({
            "severity": "WARNING",
            "component": "structure",
            "message": "Missing or invalid H2",
            "fix": "Start batch with 'h2: Title'"
        })
    elif word_count < min_words:
        scores["structure"] = 60
        issues.append({
            "severity": "INFO",
            "component": "structure",
            "message": f"Batch short ({word_count} words, min {min_words})",
            "fix": f"Aim for {min_words}-{max_words} words"
        })
    elif word_count > max_words:
        scores["structure"] = 75
        issues.append({
            "severity": "INFO",
            "component": "structure",
            "message": f"Batch long ({word_count} words, max {max_words})",
            "fix": f"Aim for {min_words}-{max_words} words"
        })
    else:
        scores["structure"] = 100
    
    # ================================================================
    # 5. SEMANTIC (10%)
    # v37.4: Wielopoziomowe capy
    # ================================================================
    semantic_score = validation_results.get("semantic_score", 0.5)
    scores["semantic"] = int(semantic_score * 100)
    
    if semantic_score < config.semantic_cap_B:
        # Bardzo słaba → cap B
        if max_grade is None or max_grade in ["A+", "A", "B+"]:
            max_grade = "B"
            decision_trace.append(f"semantic: {semantic_score:.0%} < {config.semantic_cap_B} → cap grade B")
        issues.append({
            "severity": "WARNING",
            "component": "semantic",
            "message": f"Low semantic coverage ({semantic_score:.0%})",
            "fix": "Add more topic-related terms and entities"
        })
    elif semantic_score < config.semantic_cap_B_plus:
        # Średnia → cap B+
        if max_grade is None or max_grade in ["A+", "A"]:
            max_grade = "B+"
            decision_trace.append(f"semantic: {semantic_score:.0%} < {config.semantic_cap_B_plus} → cap grade B+")
        issues.append({
            "severity": "INFO",
            "component": "semantic",
            "message": f"Moderate semantic coverage ({semantic_score:.0%})",
            "fix": "Consider adding more related terms"
        })
    
    # ================================================================
    # 6. COHERENCE (10%)
    # ================================================================
    batch_review = validation_results.get("batch_review", {})
    coherence_issues = []
    if isinstance(batch_review, dict):
        coherence_issues = [
            i for i in batch_review.get("issues", [])
            if isinstance(i, dict) and i.get("type") == "COHERENCE"
        ]
    
    if coherence_issues:
        scores["coherence"] = 60
        issues.append({
            "severity": "INFO",
            "component": "coherence",
            "message": "Could flow better from previous batch",
            "fix": "Add transition sentence at the beginning"
        })
    else:
        scores["coherence"] = 100
    
    # ================================================================
    # OBLICZ GLOBAL SCORE
    # ================================================================
    global_score = sum(
        scores.get(k, 50) * QUALITY_WEIGHTS[k] / 100
        for k in QUALITY_WEIGHTS
    )
    global_score = round(global_score)
    
    # ================================================================
    # OKREŚL GRADE (z capem)
    # ================================================================
    grade_letter, grade_status = get_grade(global_score, max_grade, config)
    
    if max_grade and grade_letter == max_grade:
        decision_trace.append(f"grade: {global_score} pts → capped to {grade_letter}")
    else:
        decision_trace.append(f"grade: {global_score} pts → {grade_letter}")
    
    # ================================================================
    # SORTUJ ISSUES
    # ================================================================
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    issues.sort(key=lambda x: severity_order.get(x.get("severity", "INFO"), 99))
    
    # FIX PRIORITY
    fix_priority = []
    seen = set()
    for issue in issues:
        fix = issue.get("fix")
        if fix and fix not in seen:
            fix_priority.append(fix)
            seen.add(fix)
    
    return {
        "score": global_score,
        "grade": grade_letter,
        "status": grade_status,
        "components": scores,
        "max_grade_cap": max_grade,
        "top_issues": [f"[{i['severity']}] {i['message']}" for i in issues[:5]],
        "fix_priority": fix_priority[:3],
        "issue_count": {
            "critical": len([i for i in issues if i["severity"] == "CRITICAL"]),
            "warning": len([i for i in issues if i["severity"] == "WARNING"]),
            "info": len([i for i in issues if i["severity"] == "INFO"])
        },
        "decision_trace": decision_trace  # 🆕 Audit trail
    }


# ================================================================
# CONFIDENCE LEVEL
# ================================================================

def calculate_confidence(quality_result: Dict, config: QualityConfig = CONFIG) -> str:
    """Oblicza poziom pewności decyzji."""
    score = quality_result.get("score", 0)
    issue_count = quality_result.get("issue_count", {})
    
    critical_count = issue_count.get("critical", 0)
    warning_count = issue_count.get("warning", 0)
    
    grade_boundaries = [
        config.grade_A_plus, config.grade_A, config.grade_B_plus,
        config.grade_B, config.grade_C, config.grade_D
    ]
    near_boundary = any(abs(score - b) <= 3 for b in grade_boundaries)
    
    if critical_count == 0 and warning_count <= 1 and not near_boundary:
        return "HIGH"
    if critical_count >= 2 or warning_count >= 4 or near_boundary:
        return "LOW"
    return "MEDIUM"


# ================================================================
# GPT ACTION RESPONSE
# ================================================================

def get_gpt_action_response(
    validation_results: Dict,
    quality_result: Dict,
    project_data: Dict,
    batch_number: int
) -> Dict:
    """Generuje response dla GPT z akcją i decision_trace."""
    score = quality_result.get("score", 0)
    grade = quality_result.get("grade", "F")
    status = quality_result.get("status", "REJECTED")
    decision_trace = quality_result.get("decision_trace", [])
    
    confidence = calculate_confidence(quality_result)
    
    exceeded_critical = validation_results.get("exceeded_critical", [])
    
    # DECYZJA (osobna od score!)
    if exceeded_critical:
        accepted = False
        action = "REWRITE"
        message = f"❌ Rejected - {len(exceeded_critical)} keyword(s) exceeded 50%+. Rewrite with synonyms."
        decision_trace.append(f"action: REWRITE (exceeded_critical)")
    elif grade == "F" or status == "REJECTED":
        accepted = False
        action = "REWRITE"
        message = f"❌ Rejected (score: {score}/100). Rewrite with fixes."
        decision_trace.append(f"action: REWRITE (grade F)")
    elif grade in ["C", "D"]:
        accepted = False
        action = "FIX_AND_RETRY"
        message = f"⚠️ Needs fixes (score: {score}/100, grade: {grade}). Fix and retry."
        decision_trace.append(f"action: FIX_AND_RETRY (grade {grade})")
    else:
        accepted = True
        action = "CONTINUE"
        message = f"✅ Accepted (score: {score}/100, grade: {grade}). Continue."
        decision_trace.append(f"action: CONTINUE")
    
    # Auto-fixy
    batch_review = validation_results.get("batch_review", {})
    fixes_applied = batch_review.get("auto_fixes_applied", []) if isinstance(batch_review, dict) else []
    fixes_needed = quality_result.get("fix_priority", [])
    
    # Next task
    next_task = None
    if accepted:
        total_batches = project_data.get("total_planned_batches", 7)
        if batch_number < total_batches:
            h2_structure = project_data.get("h2_structure", [])
            next_h2 = h2_structure[batch_number] if batch_number < len(h2_structure) else "Podsumowanie"
            next_task = {
                "batch_number": batch_number + 1,
                "h2": next_h2,
                "action": "GET /pre_batch_info"
            }
        else:
            next_task = {
                "batch_number": None,
                "h2": None,
                "action": "GET /full_article (complete)"
            }
    
    return {
        "accepted": accepted,
        "action": action,
        "confidence": confidence,
        "score": score,
        "grade": grade,
        "message": message,
        "fixes_applied": fixes_applied[:5],
        "fixes_needed": fixes_needed[:3],
        "next_task": next_task,
        "decision_trace": decision_trace  # 🆕
    }


# ================================================================
# SIMPLIFIED RESPONSE
# ================================================================

def create_simplified_response(
    full_results: Dict,
    project_data: Dict,
    batch_number: int
) -> Dict:
    """Tworzy uproszczony response dla GPT."""
    quality = calculate_global_quality_score(full_results, project_data)
    gpt_action = get_gpt_action_response(full_results, quality, project_data, batch_number)
    
    # Exceeded summary
    exceeded_critical = full_results.get("exceeded_critical", [])
    exceeded_warning = full_results.get("exceeded_warning", [])
    
    exceeded_summary = []
    for exc in exceeded_critical:
        exceeded_summary.append({
            "keyword": exc.get("keyword"),
            "severity": "CRITICAL",
            "use_instead": exc.get("synonyms", [])[:3]
        })
    for exc in exceeded_warning[:3]:
        exceeded_summary.append({
            "keyword": exc.get("keyword"),
            "severity": "WARNING",
            "use_instead": exc.get("synonyms", [])[:3]
        })
    
    return {
        "accepted": gpt_action["accepted"],
        "action": gpt_action["action"],
        "confidence": gpt_action["confidence"],
        "message": gpt_action["message"],
        
        "quality": {
            "score": quality["score"],
            "grade": quality["grade"],
            "status": quality["status"]
        },
        
        "issues": quality["top_issues"][:4],
        "fixes_needed": gpt_action["fixes_needed"],
        "fixes_applied": gpt_action["fixes_applied"],
        
        "exceeded_keywords": exceeded_summary,
        "next_task": gpt_action["next_task"],
        
        "decision_trace": gpt_action["decision_trace"]  # 🆕 Audit
    }


# ================================================================
# FAST MODE VALIDATION
# ================================================================

def validate_fast_mode(
    batch_text: str,
    keywords_state: Dict,
    batch_counts: Dict,
    config: QualityConfig = CONFIG
) -> Dict:
    """Szybka walidacja (~100ms)."""
    import re
    import statistics
    
    results = {
        "mode": "FAST",
        "checks_performed": [],
        "decision_trace": []
    }
    
    # 1. EXCEEDED CHECK
    results["checks_performed"].append("exceeded_keywords")
    exceeded_critical = []
    exceeded_warning = []
    
    for rid, count in batch_counts.items():
        if rid not in keywords_state:
            continue
        
        meta = keywords_state[rid]
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        target_max = meta.get("target_max", 999)
        if target_max <= 0:
            continue
        
        actual = meta.get("actual_uses", 0)
        new_total = actual + count
        
        if new_total > target_max:
            exceeded_by = new_total - target_max
            exceeded_percent = round((exceeded_by / target_max) * 100)
            
            exc_info = {
                "keyword": keyword,
                "actual": new_total,
                "target_max": target_max,
                "exceeded_percent": exceeded_percent,
                "synonyms": meta.get("synonyms", [])
            }
            
            if exceeded_percent >= config.exceeded_critical_threshold:
                exceeded_critical.append(exc_info)
                results["decision_trace"].append(f"exceeded_critical: '{keyword}' +{exceeded_percent}%")
            else:
                exceeded_warning.append(exc_info)
    
    results["exceeded_critical"] = exceeded_critical
    results["exceeded_warning"] = exceeded_warning
    
    if exceeded_critical:
        results["status"] = "REJECTED"
        results["action"] = "REWRITE"
        results["message"] = f"❌ {len(exceeded_critical)} keyword(s) exceeded 50%+. Use synonyms."
        results["decision_trace"].append("action: REWRITE")
        return results
    
    # 2. STRUCTURE CHECK
    results["checks_performed"].append("basic_structure")
    has_h2 = bool(re.search(r'^h2:\s*.+', batch_text, re.MULTILINE | re.IGNORECASE))
    word_count = len(batch_text.split())
    
    results["has_h2"] = has_h2
    results["word_count"] = word_count
    
    if not has_h2:
        results["status"] = "REJECTED"
        results["action"] = "REWRITE"
        results["message"] = "❌ Missing H2. Start with 'h2: Title'"
        results["decision_trace"].append("action: REWRITE (no H2)")
        return results
    
    # 3. BURSTINESS CHECK
    results["checks_performed"].append("burstiness_quick")
    
    sentences = re.split(r'[.!?]+', batch_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    cv = 0.4
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        
        if mean_len > 0:
            try:
                std_dev = statistics.stdev(lengths)
                cv = std_dev / mean_len
            except statistics.StatisticsError:
                cv = 0.4
        
        results["burstiness_cv"] = round(cv, 3)
    
    if cv < config.cv_critical:
        results["ai_warning"] = True
        results["decision_trace"].append(f"burstiness: CV={cv:.2f} < {config.cv_critical} (AI warning)")
    
    # FINAL
    if exceeded_warning:
        results["status"] = "WARNING"
        results["action"] = "CONTINUE"
        results["message"] = f"⚠️ {len(exceeded_warning)} keyword(s) slightly exceeded."
    else:
        results["status"] = "OK"
        results["action"] = "CONTINUE"
        results["message"] = "✅ Fast check passed."
    
    results["decision_trace"].append(f"action: {results['action']}")
    return results
