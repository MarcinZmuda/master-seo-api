# grammar_middleware.py
# v1.0 - Middleware sprawdzający gramatykę PRZED zapisem batcha
#
# Flow:
# 1. GPT wysyła batch do preview_batch
# 2. Middleware sprawdza LanguageTool
# 3. Jeśli błędy → zwraca instrukcję do poprawy
# 4. Jeśli OK → approve_batch zapisuje
#
# Integracja: w preview_batch i approve_batch

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================
# LANGUAGETOOL
# ============================================================

_TOOL = None
_BACKEND = "none"

def init_languagetool():
    """Inicjalizuje LanguageTool (lazy loading)."""
    global _TOOL, _BACKEND
    
    if _TOOL is not None:
        return _TOOL
    
    # Próba 1: Lokalna instalacja
    try:
        import language_tool_python
        _TOOL = language_tool_python.LanguageTool('pl')
        _BACKEND = "local"
        print("[GRAMMAR_MW] ✅ LanguageTool (local) initialized")
        return _TOOL
    except ImportError:
        pass
    except Exception as e:
        print(f"[GRAMMAR_MW] ⚠️ Local LT failed: {e}")
    
    # Próba 2: API
    _BACKEND = "api"
    print("[GRAMMAR_MW] Using LanguageTool API fallback")
    return None


def check_grammar_api(text: str) -> List[Dict]:
    """Sprawdza gramatykę przez API."""
    try:
        import requests
        response = requests.post(
            "https://api.languagetool.org/v2/check",
            data={"text": text, "language": "pl"},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("matches", [])
    except Exception as e:
        print(f"[GRAMMAR_MW] API error: {e}")
        return []


# ============================================================
# FILTROWANIE BŁĘDÓW - TYLKO KRYTYCZNE
# ============================================================

# Reguły do ignorowania (sugestie stylistyczne, nie błędy)
IGNORED_RULES = {
    "WHITESPACE_RULE",           # podwójne spacje
    "COMMA_PARENTHESIS_WHITESPACE",
    "MORFOLOGIK_RULE_PL_PL",     # nieznane słowa (nazwy własne)
    "PL_WORD_REPEAT",            # powtórzenia (czasem celowe)
    "UPPERCASE_SENTENCE_START",  # wielka litera na początku
    "PUNCTUATION_PARAGRAPH_END", # interpunkcja na końcu
}

# Reguły KRYTYCZNE - błędy gramatyczne
CRITICAL_RULES = {
    "AGREEMENT",                 # zgodność przypadków ← TO ŁAPIE "olejem" vs "oleju"!
    "PL_CASE_AGREEMENT",         # zgodność przypadków PL
    "PL_GENDER_AGREEMENT",       # zgodność rodzajów
    "PL_NUMBER_AGREEMENT",       # zgodność liczby
    "VERB_FORM",                 # forma czasownika
    "PREP_CASE",                 # przypadek po przyimku
}


def filter_critical_errors(matches: List[Dict]) -> List[Dict]:
    """Filtruje tylko krytyczne błędy gramatyczne."""
    critical = []
    
    for m in matches:
        rule_id = m.get("rule", {}).get("id", "") if isinstance(m.get("rule"), dict) else m.get("ruleId", "")
        
        # Ignoruj niektóre reguły
        if rule_id in IGNORED_RULES:
            continue
        
        # Zawsze przepuszczaj krytyczne
        if any(crit in rule_id for crit in CRITICAL_RULES):
            critical.append(m)
            continue
        
        # Dla pozostałych - tylko jeśli są sugestie
        replacements = m.get("replacements", [])
        if replacements and len(replacements) > 0:
            # Sprawdź czy to prawdziwy błąd (nie tylko sugestia)
            category = m.get("rule", {}).get("category", {}).get("id", "")
            if category in ["GRAMMAR", "TYPOS", "CONFUSED_WORDS"]:
                critical.append(m)
    
    return critical


# ============================================================
# GŁÓWNA FUNKCJA WALIDACJI
# ============================================================

@dataclass
class GrammarValidation:
    """Wynik walidacji gramatycznej."""
    is_valid: bool
    errors: List[Dict]
    error_count: int
    correction_prompt: Optional[str]
    backend: str


def validate_batch_grammar(text: str, max_errors: int = 3) -> GrammarValidation:
    """
    Waliduje gramatykę batcha.
    
    Args:
        text: Tekst batcha do sprawdzenia
        max_errors: Max błędów do pokazania w prompcie (żeby nie przytłoczyć)
    
    Returns:
        GrammarValidation z is_valid=True jeśli OK, lub correction_prompt jeśli błędy
    """
    if not text or len(text.strip()) < 50:
        return GrammarValidation(
            is_valid=True,
            errors=[],
            error_count=0,
            correction_prompt=None,
            backend="skipped"
        )
    
    # Inicjalizuj LT
    tool = init_languagetool()
    
    # Sprawdź gramatykę
    if tool and _BACKEND == "local":
        raw_matches = tool.check(text)
        matches = [{
            "ruleId": m.ruleId,
            "message": m.message,
            "context": m.context,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": m.replacements[:3],
            "rule": {"id": m.ruleId, "category": {"id": m.category}}
        } for m in raw_matches]
    else:
        matches = check_grammar_api(text)
    
    # Filtruj tylko krytyczne błędy
    critical_errors = filter_critical_errors(matches)
    
    if not critical_errors:
        return GrammarValidation(
            is_valid=True,
            errors=[],
            error_count=0,
            correction_prompt=None,
            backend=_BACKEND
        )
    
    # Buduj prompt do poprawy
    error_descriptions = []
    for i, err in enumerate(critical_errors[:max_errors], 1):
        msg = err.get("message", "błąd gramatyczny")
        context = err.get("context", {})
        if isinstance(context, dict):
            context_text = context.get("text", "")
        else:
            context_text = str(context)[:50]
        
        replacements = err.get("replacements", [])
        if isinstance(replacements, list) and replacements:
            if isinstance(replacements[0], dict):
                suggestion = replacements[0].get("value", "")
            else:
                suggestion = str(replacements[0])
        else:
            suggestion = ""
        
        error_descriptions.append(
            f"{i}. {msg}\n   Kontekst: ...{context_text}...\n   Sugestia: {suggestion}"
        )
    
    more_errors = len(critical_errors) - max_errors
    if more_errors > 0:
        error_descriptions.append(f"... i {more_errors} więcej błędów")
    
    correction_prompt = f"""⚠️ WYKRYTO {len(critical_errors)} BŁĘDÓW GRAMATYCZNYCH!

Popraw poniższe błędy, zachowując WSZYSTKIE frazy kluczowe:

{chr(10).join(error_descriptions)}

INSTRUKCJE:
1. Popraw TYLKO wskazane błędy
2. NIE zmieniaj fraz kluczowych
3. NIE dodawaj nowej treści
4. Zwróć CAŁY poprawiony tekst batcha"""

    return GrammarValidation(
        is_valid=False,
        errors=critical_errors,
        error_count=len(critical_errors),
        correction_prompt=correction_prompt,
        backend=_BACKEND
    )


# ============================================================
# BANNED PHRASES CHECK
# ============================================================

BANNED_PATTERNS = [
    (r"(?i)przykład(?:owo)?:\s*[^.!?]+[.!?]", "Przykład: ..."),
    (r"(?i)na przykład\s+[^.!?]+[.!?]", "Na przykład ..."),
    (r"(?i)dla przykładu[,:]\s*[^.!?]+[.!?]", "Dla przykładu ..."),
    (r"(?i)warto (?:wiedzieć|zauważyć|wspomnieć),?\s*że", "Warto wiedzieć, że"),
    (r"(?i)w dzisiejszych czasach", "W dzisiejszych czasach"),
    (r"(?i)nie jest tajemnicą,?\s*że", "Nie jest tajemnicą, że"),
    (r"(?i)jak wiadomo", "Jak wiadomo"),
]


def check_banned_phrases(text: str) -> Tuple[bool, List[str]]:
    """
    Sprawdza czy tekst zawiera zabronione frazy.
    
    Returns:
        (is_clean, list_of_found_phrases)
    """
    found = []
    for pattern, name in BANNED_PATTERNS:
        if re.search(pattern, text):
            found.append(name)
    
    return len(found) == 0, found


# ============================================================
# PEŁNA WALIDACJA BATCHA
# ============================================================

def validate_batch_full(text: str) -> Dict:
    """
    Pełna walidacja batcha: gramatyka + banned phrases.
    
    Użycie w preview_batch:
        from grammar_middleware import validate_batch_full
        validation = validate_batch_full(batch_text)
        if not validation["is_valid"]:
            return jsonify({"needs_correction": True, ...})
    
    Returns:
        {
            "is_valid": bool,
            "grammar": GrammarValidation dict,
            "banned_phrases": {"is_clean": bool, "found": [...]},
            "correction_needed": bool,
            "correction_prompt": str or None
        }
    """
    # Gramatyka
    grammar = validate_batch_grammar(text)
    
    # Banned phrases
    is_clean, found_banned = check_banned_phrases(text)
    
    # Łączny wynik
    is_valid = grammar.is_valid and is_clean
    
    # Buduj łączny prompt jeśli potrzeba
    correction_prompt = None
    if not is_valid:
        prompts = []
        if grammar.correction_prompt:
            prompts.append(grammar.correction_prompt)
        if found_banned:
            prompts.append(f"""⚠️ WYKRYTO ZABRONIONE FRAZY!

Usuń lub przepisz następujące fragmenty:
{chr(10).join(f'- "{phrase}"' for phrase in found_banned)}

INSTRUKCJE:
1. Usuń całe zdania zawierające te frazy
2. LUB przepisz bez użycia tych zwrotów
3. NIE usuwaj fraz kluczowych""")
        
        correction_prompt = "\n\n---\n\n".join(prompts)
    
    return {
        "is_valid": is_valid,
        "grammar": {
            "is_valid": grammar.is_valid,
            "error_count": grammar.error_count,
            "errors": grammar.errors[:5],  # max 5 do response
            "backend": grammar.backend
        },
        "banned_phrases": {
            "is_clean": is_clean,
            "found": found_banned
        },
        "correction_needed": not is_valid,
        "correction_prompt": correction_prompt
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    test_text = """
Czyścić ostrze i zapobiec dalszej korozji, należy używać delikatnych środków oraz oleju spożywczego lub olejem do konserwacji noży.

Przykład: nóż ze stali 1095 z lekką rdzą odzyskał połysk po delikatnym czyszczeniu. W ten sposób można dbać o nóż.

Warto wiedzieć, że konserwacja noży jest ważna.
"""
    
    print("=" * 60)
    print("WALIDACJA BATCHA")
    print("=" * 60)
    
    result = validate_batch_full(test_text)
    
    print(f"\n✅ Valid: {result['is_valid']}")
    print(f"📝 Grammar errors: {result['grammar']['error_count']}")
    print(f"🚫 Banned phrases: {result['banned_phrases']['found']}")
    
    if result['correction_prompt']:
        print("\n" + "=" * 60)
        print("PROMPT DO POPRAWY:")
        print("=" * 60)
        print(result['correction_prompt'])
