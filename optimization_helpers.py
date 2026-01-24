"""
===============================================================================
🔄 OPTIMIZATION HELPERS v36.9 - Moduł anty-pętlowy
===============================================================================

Rozwiązuje problem "over-optimization" gdzie AI wpada w pętlę poprawek.

FUNKCJE:
1. prioritize_issues() - Zwraca max 2 najważniejsze problemy
2. get_actionable_feedback() - Konkretne instrukcje naprawy
3. auto_fix_burstiness() - Automatyczna naprawa monotonii
4. enhance_response() - Wzbogaca response o priorytyzowane problemy

UŻYCIE:
    from optimization_helpers import enhance_response, try_auto_fix
    
    # W approve_batch lub podobnym:
    result = process_batch_in_firestore(...)
    enhanced = enhance_response(result, attempt=2)
    
    # Auto-fix przed zapisem:
    fixed_text, fixes, stats = try_auto_fix(text, result)

KOMPATYBILNOŚĆ:
- Wszystkie funkcje są opcjonalne (opt-in)
- Nie modyfikują istniejących struktur
- Dodają nowe pola do response (nie usuwają starych)
===============================================================================
"""

from typing import Dict, List, Any, Tuple, Optional


# ================================================================
# IMPORT Z UNIFIED_VALIDATOR (z fallbackiem)
# ================================================================
try:
    from unified_validator import (
        prioritize_issues,
        get_actionable_feedback,
        auto_fix_burstiness,
        ValidationIssue,
        Severity
    )
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    print("[OPTIMIZATION_HELPERS] ⚠️ unified_validator not available, using fallback")


# ================================================================
# FALLBACK IMPLEMENTATIONS (jeśli unified_validator niedostępny)
# ================================================================
if not VALIDATOR_AVAILABLE:
    def prioritize_issues(issues: List, max_issues: int = 2) -> List:
        """Fallback - zwraca pierwsze N issues"""
        return issues[:max_issues] if issues else []
    
    def get_actionable_feedback(issues: List, attempt: int = 1, previous_issues: List = None) -> Dict:
        """Fallback - podstawowy feedback"""
        return {
            "attempt": attempt,
            "total_issues": len(issues) if issues else 0,
            "instructions": [],
            "focus_message": "Sprawdź problemy w liście warnings"
        }
    
    def auto_fix_burstiness(text: str, target_cv: float = 0.44) -> Tuple[str, List[str], Dict]:
        """Fallback - bez zmian"""
        return text, [], {"status": "UNAVAILABLE", "reason": "unified_validator not available"}


# ================================================================
# MAIN HELPER: ENHANCE RESPONSE
# ================================================================
def enhance_response(
    result: Dict[str, Any],
    attempt: int = 1,
    previous_issues: List[Dict] = None,
    include_actionable: bool = True
) -> Dict[str, Any]:
    """
    🆕 v36.9: Wzbogaca response o priorytyzowane instrukcje.
    
    NIE MODYFIKUJE oryginalnego result - zwraca kopię z dodatkowymi polami.
    
    Args:
        result: Oryginalny result z process_batch_in_firestore
        attempt: Numer próby (1, 2, 3...)
        previous_issues: Issues z poprzedniej próby (opcjonalnie)
        include_actionable: Czy dodać actionable feedback
        
    Returns:
        Wzbogacony result z nowymi polami:
        - prioritized_issues: Max 2 najważniejsze problemy
        - actionable_feedback: Konkretne instrukcje naprawy
        - optimization_hints: Dodatkowe wskazówki
    """
    # Kopia żeby nie modyfikować oryginału
    enhanced = dict(result)
    
    # Zbierz wszystkie issues z różnych źródeł
    all_issues = []
    
    # Z warnings (stary format - stringi)
    for warning in result.get("warnings", []):
        if isinstance(warning, str):
            # Konwertuj string warning na pseudo-issue
            issue_type = _extract_issue_type(warning)
            all_issues.append({
                "type": issue_type,
                "message": warning,
                "severity": "WARNING",
                "details": {}
            })
    
    # Z exceeded_keywords
    for exceeded in result.get("exceeded_keywords", []):
        all_issues.append({
            "type": "EXCEEDED_TOTAL",
            "message": f"Przekroczono limit dla '{exceeded.get('keyword')}'",
            "severity": "CRITICAL",
            "details": exceeded
        })
    
    # Z soft_cap_validation
    soft_cap = result.get("soft_cap_validation", {})
    for hard_exc in soft_cap.get("hard_exceeded", []):
        all_issues.append({
            "type": "HARD_EXCEEDED",
            "message": f"Hard limit dla '{hard_exc.get('keyword')}'",
            "severity": "CRITICAL",
            "details": hard_exc
        })
    
    # Priorytyzuj
    if VALIDATOR_AVAILABLE and all_issues:
        # Konwertuj na ValidationIssue jeśli możliwe
        try:
            validation_issues = [
                ValidationIssue(
                    type=i["type"],
                    message=i["message"],
                    severity=Severity[i["severity"]] if isinstance(i["severity"], str) else i["severity"],
                    details=i.get("details", {})
                )
                for i in all_issues
            ]
            prioritized = prioritize_issues(validation_issues, max_issues=2)
            enhanced["prioritized_issues"] = [
                {"type": p.type, "message": p.message, "severity": p.severity.value}
                for p in prioritized
            ]
        except Exception as e:
            # Fallback - użyj surowej listy
            enhanced["prioritized_issues"] = all_issues[:2]
            enhanced["_prioritization_error"] = str(e)
    else:
        enhanced["prioritized_issues"] = all_issues[:2]
    
    # Actionable feedback
    if include_actionable and VALIDATOR_AVAILABLE:
        try:
            if all_issues:
                validation_issues = [
                    ValidationIssue(
                        type=i["type"],
                        message=i["message"],
                        severity=Severity.WARNING,
                        details=i.get("details", {})
                    )
                    for i in all_issues
                ]
                feedback = get_actionable_feedback(validation_issues, attempt, previous_issues)
                enhanced["actionable_feedback"] = feedback
        except Exception as e:
            enhanced["actionable_feedback"] = {
                "error": str(e),
                "fallback_message": "Napraw problemy z listy prioritized_issues"
            }
    
    # Dodaj metadane
    enhanced["optimization_meta"] = {
        "attempt": attempt,
        "total_issues_found": len(all_issues),
        "prioritized_count": len(enhanced.get("prioritized_issues", [])),
        "validator_available": VALIDATOR_AVAILABLE
    }
    
    return enhanced


def _extract_issue_type(warning: str) -> str:
    """Ekstrahuje typ issue z tekstu warning."""
    warning_lower = warning.lower()
    
    if "exceeded total" in warning_lower:
        return "EXCEEDED_TOTAL"
    elif "density" in warning_lower:
        return "HIGH_DENSITY"
    elif "burstiness" in warning_lower:
        if "low" in warning_lower or "niski" in warning_lower:
            return "LOW_BURSTINESS"
        else:
            return "HIGH_BURSTINESS"
    elif "stuffing" in warning_lower:
        return "STUFFING"
    elif "coverage" in warning_lower or "pokrycie" in warning_lower:
        return "LOW_COVERAGE"
    else:
        return "UNKNOWN"


# ================================================================
# AUTO-FIX HELPER
# ================================================================
def try_auto_fix(
    text: str,
    result: Dict[str, Any],
    fix_burstiness: bool = True,
    fix_stuffing: bool = True
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    🆕 v36.9: Próbuje automatycznie naprawić problemy w tekście.
    
    BEZPIECZNY: Zwraca oryginalny tekst jeśli naprawa się nie powiedzie.
    
    Args:
        text: Tekst do naprawy
        result: Wynik walidacji (do wykrycia problemów)
        fix_burstiness: Czy naprawiać burstiness
        fix_stuffing: Czy naprawiać stuffing
        
    Returns:
        Tuple[fixed_text, all_fixes_applied, stats]
    """
    fixed_text = text
    all_fixes = []
    stats = {
        "burstiness_fixed": False,
        "stuffing_fixed": False,
        "total_changes": 0
    }
    
    # 1. Auto-fix burstiness
    if fix_burstiness and VALIDATOR_AVAILABLE:
        # Sprawdź czy jest problem z burstiness
        burstiness_issue = False
        for warning in result.get("warnings", []):
            if "burstiness" in str(warning).lower():
                burstiness_issue = True
                break
        
        if burstiness_issue:
            try:
                fixed_text, burstiness_fixes, burstiness_stats = auto_fix_burstiness(fixed_text)
                if burstiness_fixes:
                    all_fixes.extend([f"[BURSTINESS] {f}" for f in burstiness_fixes])
                    stats["burstiness_fixed"] = True
                    stats["burstiness_stats"] = burstiness_stats
            except Exception as e:
                stats["burstiness_error"] = str(e)
    
    # 2. Auto-fix stuffing (używa istniejącej funkcji z claude_reviewer)
    if fix_stuffing:
        try:
            from claude_reviewer import auto_fix_stuffing
            
            # Zbierz stuffed keywords
            stuffed_keywords = []
            for warning in result.get("warnings", []):
                if "stuffing" in str(warning).lower() or "za dużo" in str(warning).lower():
                    # Spróbuj wyciągnąć keyword z warning
                    # Format: "STUFFING: 'keyword' (X×) ..."
                    import re
                    match = re.search(r"'([^']+)'.*?(\d+)×", str(warning))
                    if match:
                        stuffed_keywords.append({
                            "keyword": match.group(1),
                            "count": int(match.group(2)),
                            "limit": 3  # Domyślny limit
                        })
            
            if stuffed_keywords:
                fixed_text, stuffing_fixes = auto_fix_stuffing(fixed_text, stuffed_keywords)
                if stuffing_fixes:
                    all_fixes.extend([f"[STUFFING] {f}" for f in stuffing_fixes])
                    stats["stuffing_fixed"] = True
                    
        except ImportError:
            stats["stuffing_error"] = "claude_reviewer not available"
        except Exception as e:
            stats["stuffing_error"] = str(e)
    
    stats["total_changes"] = len(all_fixes)
    
    return fixed_text, all_fixes, stats


# ================================================================
# CONVENIENCE FUNCTION: SHOULD RETRY?
# ================================================================
def should_retry(
    result: Dict[str, Any],
    attempt: int,
    max_attempts: int = 3
) -> Tuple[bool, str]:
    """
    🆕 v36.9: Decyduje czy AI powinno ponowić próbę.
    
    Returns:
        Tuple[should_retry, reason]
    """
    # Sprawdź czy są CRITICAL issues
    has_critical = False
    
    # Z soft_cap_validation
    soft_cap = result.get("soft_cap_validation", {})
    if soft_cap.get("hard_exceeded"):
        has_critical = True
    
    # Z warnings
    for warning in result.get("warnings", []):
        if "CRITICAL" in str(warning).upper() or "EXCEEDED TOTAL" in str(warning).upper():
            has_critical = True
            break
    
    # Logika decyzji
    if attempt >= max_attempts:
        return False, f"Osiągnięto max prób ({max_attempts})"
    
    if has_critical:
        return True, "Są problemy CRITICAL do naprawy"
    
    # Jeśli brak critical i attempt >= 2, nie ponawiaj
    if attempt >= 2:
        return False, "Brak CRITICAL issues, auto-approve po 2 próbach"
    
    # Sprawdź czy są jakiekolwiek problemy
    warnings = result.get("warnings", [])
    exceeded = result.get("exceeded_keywords", [])
    
    if not warnings and not exceeded:
        return False, "Brak problemów"
    
    return True, "Są problemy do naprawy (próba 1)"


# ================================================================
# GENERATE RETRY CONTEXT
# ================================================================
def generate_retry_context(
    result: Dict[str, Any],
    attempt: int,
    previous_attempts: List[Dict] = None
) -> Dict[str, Any]:
    """
    🆕 v36.9: Generuje kontekst dla kolejnej próby AI.
    
    Zawiera:
    - Które problemy zostały naprawione
    - Które problemy się powtarzają
    - Konkretne instrukcje co robić inaczej
    """
    enhanced = enhance_response(result, attempt, previous_attempts)
    
    context = {
        "attempt": attempt + 1,  # Następna próba
        "previous_attempt": attempt,
        "prioritized_issues": enhanced.get("prioritized_issues", []),
        "actionable_feedback": enhanced.get("actionable_feedback", {}),
        "recurring_issues": [],
        "fixed_issues": [],
        "specific_instructions": []
    }
    
    # Porównaj z poprzednimi próbami
    if previous_attempts:
        prev_types = set()
        for prev in previous_attempts:
            for issue in prev.get("issues", []):
                prev_types.add(issue.get("type"))
        
        current_types = set(i.get("type") for i in enhanced.get("prioritized_issues", []))
        
        context["recurring_issues"] = list(prev_types & current_types)
        context["fixed_issues"] = list(prev_types - current_types)
        
        # Specjalne instrukcje dla powtarzających się problemów
        for recurring in context["recurring_issues"]:
            if recurring == "LOW_BURSTINESS":
                context["specific_instructions"].append(
                    "BURSTINESS się powtarza - spróbuj ZUPEŁNIE innego podejścia: "
                    "zacznij od bardzo krótkiego zdania (3-5 słów), "
                    "potem długie (20+ słów), potem średnie."
                )
            elif recurring in ("STUFFING", "EXCEEDED_TOTAL", "HARD_EXCEEDED"):
                context["specific_instructions"].append(
                    f"{recurring} się powtarza - UŻYJ SYNONIMÓW zamiast powtarzać frazę. "
                    "Spróbuj: 'ta procedura', 'omawiany proces', 'wspomniana instytucja'."
                )
    
    return context


# ================================================================
# EXPORT ALL
# ================================================================
__all__ = [
    "enhance_response",
    "try_auto_fix", 
    "should_retry",
    "generate_retry_context",
    "VALIDATOR_AVAILABLE"
]
