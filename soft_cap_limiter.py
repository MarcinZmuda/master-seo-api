"""
Soft Cap Limiter v1.0
=====================
Elastyczne limity dla anti-frankenstein integration.
Fix #23 v4.2: Wymagane przez anti_frankenstein_integration.py:46-48

Eksportuje:
  - create_soft_cap_validator(base_limits, tolerance)
  - validate_with_soft_caps(text, keywords_state, validator)
  - get_flexible_limits(keywords_state, tolerance)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import re


@dataclass
class SoftCapResult:
    """Wynik walidacji z elastycznymi limitami."""
    passed: bool
    violations: List[Dict] = field(default_factory=list)
    adjusted_limits: Dict[str, int] = field(default_factory=dict)
    summary: str = ""


def create_soft_cap_validator(base_limits: Dict, tolerance: float = 0.15) -> Dict:
    """
    Tworzy walidator z elastycznymi limitami.

    Args:
        base_limits: dict {keyword: max_count}
        tolerance: procent tolerancji (0.15 = 15%)

    Returns:
        dict z limits i tolerance do uzycia w validate_with_soft_caps
    """
    adjusted = {}
    for key, limit in base_limits.items():
        adjusted[key] = int(limit * (1 + tolerance))
    return {
        'limits': adjusted,
        'base_limits': base_limits,
        'tolerance': tolerance
    }


def validate_with_soft_caps(
    text: str,
    keywords_state: Dict,
    validator: Dict
) -> SoftCapResult:
    """
    Waliduj tekst z elastycznymi limitami.

    Args:
        text: tekst do walidacji
        keywords_state: dict keywords z target_max
        validator: wynik z create_soft_cap_validator

    Returns:
        SoftCapResult z passed, violations, adjusted_limits
    """
    limits = validator.get('limits', {})
    tolerance = validator.get('tolerance', 0.15)
    violations = []
    text_lower = text.lower()

    for kw, meta in keywords_state.items():
        keyword = meta.get('keyword', kw) if isinstance(meta, dict) else str(kw)
        keyword_lower = keyword.lower()

        # Count occurrences
        count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_lower))

        # Get soft max (with tolerance)
        base_max = meta.get('target_max', 999) if isinstance(meta, dict) else 999
        soft_max = limits.get(kw, int(base_max * (1 + tolerance)))

        if count > soft_max:
            violations.append({
                'keyword': keyword,
                'count': count,
                'base_max': base_max,
                'soft_max': soft_max,
                'exceeded_by': count - soft_max,
                'exceed_percent': round((count - soft_max) / soft_max * 100) if soft_max > 0 else 100
            })

    passed = len(violations) == 0
    summary = f"OK: all within soft caps" if passed else f"FAIL: {len(violations)} keywords exceed soft caps"

    return SoftCapResult(
        passed=passed,
        violations=violations,
        adjusted_limits=limits,
        summary=summary
    )


def get_flexible_limits(
    keywords_state: Dict,
    tolerance: float = 0.15
) -> Dict[str, int]:
    """
    Zwroc elastyczne limity (base * (1 + tolerance)).

    Args:
        keywords_state: dict keywords z target_max
        tolerance: procent tolerancji (0.15 = 15%)

    Returns:
        dict {keyword_id: flexible_max}
    """
    result = {}
    for kw, meta in keywords_state.items():
        if isinstance(meta, dict):
            base_max = meta.get('target_max', 10)
        else:
            base_max = 10
        result[kw] = int(base_max * (1 + tolerance))
    return result
