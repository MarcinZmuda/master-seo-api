"""
grammar_checker.py — Full grammar correction for BRAJEN SEO editorial pipeline
================================================================================
v53: Created to provide actual grammar correction (not just diagnostics).

Uses LanguageTool REST API to detect and auto-fix grammar errors in Polish text.
Called BEFORE burstiness rollback check so fixes are not lost to rollback.

Exported functions:
    full_correction(text: str) -> dict
"""

import re
import logging
import os

logger = logging.getLogger(__name__)

_LT_API_URL = os.environ.get("LANGUAGETOOL_URL", "https://api.languagetool.org/v2/check")

# Rules to SKIP (stylistic, not errors)
_SKIP_RULES = {
    "WHITESPACE_RULE",
    "COMMA_PARENTHESIS_WHITESPACE",
    "UPPERCASE_SENTENCE_START",
    "PUNCTUATION_PARAGRAPH_END",
    "PL_WORD_REPEAT",
}

# Rules that are safe to auto-apply (grammar, spelling, case agreement)
_AUTO_FIX_CATEGORIES = {
    "GRAMMAR", "TYPOS", "CONFUSED_WORDS",
    "AGREEMENT", "CASE", "VERB", "MORPHOLOGY",
}

# Banned AI-generated filler phrases to remove
_BANNED_PHRASES = [
    "warto zauważyć, że",
    "warto wiedzieć, że",
    "warto wspomnieć, że",
    "warto podkreślić, że",
    "należy podkreślić, że",
    "należy zaznaczyć, że",
    "co istotne,",
    "co ważne,",
    "co kluczowe,",
    "nie ulega wątpliwości, że",
    "w dzisiejszych czasach",
    "jak wiadomo,",
    "nie jest tajemnicą, że",
]


def _lt_check(text: str) -> list:
    """Call LanguageTool REST API. Returns list of match dicts."""
    try:
        import requests
        payload = {
            "text": text[:8000],
            "language": "pl-PL",
            "disabledCategories": "TYPOGRAPHY",
        }
        resp = requests.post(_LT_API_URL, data=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("matches", [])
        else:
            logger.warning(f"[GRAMMAR_CHECKER] LT API {resp.status_code}")
            return []
    except Exception as e:
        logger.warning(f"[GRAMMAR_CHECKER] LT API error: {e}")
        return []


def _is_safe_to_autofix(match: dict) -> bool:
    """Check if a LanguageTool match is safe to auto-apply."""
    rule = match.get("rule", {})
    rule_id = rule.get("id", "")
    cat_id = rule.get("category", {}).get("id", "")

    if rule_id in _SKIP_RULES:
        return False

    # Must have at least one replacement
    replacements = match.get("replacements", [])
    if not replacements:
        return False

    # Auto-fix grammar, typos, agreement
    for cat in _AUTO_FIX_CATEGORIES:
        if cat in cat_id.upper() or cat in rule_id.upper():
            return True

    return False


def _apply_lt_fixes(text: str, matches: list) -> tuple:
    """
    Apply LanguageTool fixes to text, working backwards to preserve offsets.
    Returns (fixed_text, fix_count, fixes_list).
    """
    # Filter safe-to-fix matches
    safe_matches = [m for m in matches if _is_safe_to_autofix(m)]

    # Sort by offset descending (apply from end to preserve positions)
    safe_matches.sort(key=lambda m: m.get("offset", 0), reverse=True)

    fixes = []
    fixed_text = text

    for m in safe_matches:
        offset = m.get("offset", 0)
        length = m.get("length", 0)
        replacements = m.get("replacements", [])

        if not replacements or offset < 0 or length <= 0:
            continue

        # Use first suggestion
        replacement = replacements[0]
        new_value = replacement.get("value", "") if isinstance(replacement, dict) else str(replacement)

        if not new_value:
            continue

        # Get original fragment
        original_fragment = fixed_text[offset:offset + length]

        # Skip if replacement is same as original
        if original_fragment == new_value:
            continue

        # Apply fix
        fixed_text = fixed_text[:offset] + new_value + fixed_text[offset + length:]
        fixes.append({
            "original": original_fragment,
            "replacement": new_value,
            "rule": m.get("rule", {}).get("id", ""),
            "message": m.get("message", ""),
        })

    return fixed_text, len(fixes), fixes


def _remove_banned_phrases(text: str) -> tuple:
    """Remove banned AI-filler phrases. Returns (cleaned_text, removed_list)."""
    removed = []
    cleaned = text

    for phrase in _BANNED_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(cleaned):
            # Remove the phrase (and trim surrounding whitespace)
            cleaned = pattern.sub("", cleaned)
            removed.append(phrase)

    # Clean up double spaces left after removal
    cleaned = re.sub(r'  +', ' ', cleaned)
    # Clean up empty sentence starts (capitalization after removal)
    cleaned = re.sub(r'\.\s+([a-ząćęłńóśźż])', lambda m: '. ' + m.group(1).upper(), cleaned)

    return cleaned, removed


def full_correction(text: str) -> dict:
    """
    Full grammar correction pipeline:
    1. LanguageTool auto-fix (safe rules only)
    2. Banned phrase removal

    Returns:
        {
            "corrected": str,
            "grammar_fixes": int,
            "grammar_details": list,
            "phrases_removed": list,
            "backend": str,
        }
    """
    if not text or len(text.strip()) < 50:
        return {
            "corrected": text,
            "grammar_fixes": 0,
            "grammar_details": [],
            "phrases_removed": [],
            "backend": "skipped",
        }

    # Step 1: LanguageTool fixes
    matches = _lt_check(text)
    corrected, fix_count, fix_details = _apply_lt_fixes(text, matches)

    if fix_count > 0:
        logger.info(f"[GRAMMAR_CHECKER] Applied {fix_count} grammar fixes")

    # Step 2: Banned phrases
    corrected, removed_phrases = _remove_banned_phrases(corrected)

    if removed_phrases:
        logger.info(f"[GRAMMAR_CHECKER] Removed {len(removed_phrases)} banned phrases")

    return {
        "corrected": corrected,
        "grammar_fixes": fix_count,
        "grammar_details": fix_details[:20],
        "phrases_removed": removed_phrases,
        "backend": "languagetool_api",
    }
