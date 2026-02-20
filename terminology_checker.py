"""
terminology_checker.py - Advanced terminology verification for YMYL content
===========================================================================

Provides terminology verification using Claude Sonnet as primary, with static fallback
from content_editorial.YMYL_EXPERT_ROLES for robustness.

USAGE:
    from terminology_checker import verify_terminology

    result = verify_terminology(
        text="artykuł zawierający terminy",
        category="prawo",
        api_key="sk-..."
    )
    if result["status"] == "ERROR":
        print(result["issues"])

Features:
- Claude Sonnet primary model for advanced analysis
- Static fallback using content_editorial YMYL_EXPERT_ROLES
- Lazy import for optional dependency
- JSON parse with markdown stripping
- Terminology issue detection and suggestion
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model config
TERMINOLOGY_MODEL = os.getenv("TERMINOLOGY_MODEL", "claude-sonnet-4-6")
TERMINOLOGY_MAX_TOKENS = 2000


def _static_fallback(text: str, category: str) -> Dict[str, Any]:
    """
    Static fallback using lazy import from content_editorial YMYL_EXPERT_ROLES.

    Falls back to forbidden_terms and critical_checks from content_editorial
    when API is unavailable.

    Args:
        text: Text to check
        category: YMYL category (prawo, medycyna, finanse)

    Returns:
        Dict with issues, suggestions, and error status
    """
    try:
        # Lazy import to avoid circular dependencies
        from content_editorial import YMYL_EXPERT_ROLES
    except ImportError:
        logger.warning("[TERMINOLOGY] content_editorial not available, using empty fallback")
        return {
            "status": "ERROR",
            "method": "static_fallback",
            "issues": ["Static fallback: content_editorial not available"],
            "suggestions": [],
            "error": "content_editorial import failed"
        }

    try:
        config = YMYL_EXPERT_ROLES.get(category, {})
        forbidden_terms = config.get("forbidden_terms", [])

        issues = []
        suggestions = []

        # Check each forbidden term
        for wrong_term, correct_term in forbidden_terms:
            if wrong_term.lower() in text.lower():
                # Count occurrences
                pattern = re.compile(re.escape(wrong_term), re.IGNORECASE)
                matches = pattern.findall(text)
                count = len(matches)

                if count > 0:
                    issues.append({
                        "type": "TERMINOLOGY_ERROR",
                        "found": wrong_term,
                        "replacement": correct_term,
                        "occurrences": count,
                        "category": category
                    })
                    suggestions.append(f"Replace '{wrong_term}' ({count}x) with '{correct_term}'")

        return {
            "status": "OK" if not issues else "ISSUES_FOUND",
            "method": "static_fallback",
            "issues": issues,
            "suggestions": suggestions,
            "text_length": len(text),
            "checked_category": category
        }

    except Exception as e:
        logger.error(f"[TERMINOLOGY] Static fallback error: {e}")
        return {
            "status": "ERROR",
            "method": "static_fallback",
            "issues": [f"Static fallback error: {str(e)}"],
            "suggestions": [],
            "error": str(e)
        }


def _parse_terminology_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse Claude's terminology response with markdown stripping.

    Handles:
    - JSON wrapped in markdown code blocks
    - Raw JSON
    - Partial/malformed JSON

    Args:
        raw_text: Raw response from Claude

    Returns:
        Parsed dict or fallback dict
    """
    try:
        # Strip markdown fences
        clean = raw_text.strip()
        clean = re.sub(r'^```json\s*', '', clean)
        clean = re.sub(r'^```\s*', '', clean)
        clean = re.sub(r'```\s*$', '', clean).strip()

        # Find JSON boundaries
        first_brace = clean.find('{')
        last_brace = clean.rfind('}')

        if first_brace == -1 or last_brace <= first_brace:
            logger.warning("[TERMINOLOGY] No JSON found in response")
            return {"status": "PARSE_ERROR", "raw_response": raw_text[:500]}

        json_str = clean[first_brace:last_brace + 1]

        # Try direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try with strict=False
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                logger.warning("[TERMINOLOGY] JSON parse failed")
                return {"status": "PARSE_ERROR", "raw_response": raw_text[:500]}

    except Exception as e:
        logger.error(f"[TERMINOLOGY] Parse error: {e}")
        return {"status": "PARSE_ERROR", "error": str(e), "raw_response": raw_text[:500]}


def verify_terminology(
    text: str,
    category: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify terminology in text using Claude Sonnet as primary, fallback to static.

    This is the main public API for terminology verification.

    Args:
        text: Text to verify
        category: YMYL category (prawo, medycyna, finanse, or 'general')
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Dict with:
        - status: OK | ISSUES_FOUND | ERROR
        - method: api | static_fallback
        - issues: List of terminology issues found
        - suggestions: List of suggested fixes
        - model: Which model was used
    """
    if not text or len(text.strip()) < 10:
        return {
            "status": "ERROR",
            "method": "validation",
            "error": "Text too short or empty",
            "issues": [],
            "suggestions": []
        }

    # Ensure we have an API key
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    # Try Claude Sonnet first
    if ANTHROPIC_AVAILABLE and api_key:
        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Build prompt for terminology verification
            prompt = f"""Jesteś ekspertem w terminologii {category}.
Przeanalizuj poniższy tekst i zidentyfikuj błędy terminologiczne, halucynacje i niespójności.

KATEGORIA: {category}
TEKST ({len(text)} znaków):
{text}

Zwróć TYLKO JSON bez markdown:
{{
  "issues": [
    {{
      "type": "TERMINOLOGY_ERROR|HALLUCINATION|INCONSISTENCY",
      "found": "błędny tekst",
      "replacement": "poprawna wersja",
      "position": "gdzie w tekście",
      "severity": "CRITICAL|WARNING|INFO"
    }}
  ],
  "suggestions": ["konkretna sugestia naprawy"],
  "summary": "krótko co poprawić"
}}"""

            response = client.messages.create(
                model=TERMINOLOGY_MODEL,
                max_tokens=TERMINOLOGY_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_response = response.content[0].text.strip()
            parsed = _parse_terminology_response(raw_response)

            # Add metadata
            parsed["status"] = parsed.get("status", "OK")
            parsed["method"] = "api"
            parsed["model"] = TERMINOLOGY_MODEL

            logger.info(f"[TERMINOLOGY] Verified {len(text)} chars via {TERMINOLOGY_MODEL}")
            return parsed

        except Exception as e:
            logger.warning(f"[TERMINOLOGY] Claude API failed: {e}, falling back to static")

    # Fallback to static method
    logger.info("[TERMINOLOGY] Using static fallback")
    return _static_fallback(text, category)


# ================================================================
# EXPORT
# ================================================================

__all__ = [
    "verify_terminology",
    "_parse_terminology_response",
    "_static_fallback",
]
