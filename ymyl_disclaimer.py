"""
═══════════════════════════════════════════════════════════
YMYL DISCLAIMER — jedno źródło prawdy
═══════════════════════════════════════════════════════════

Centralizuje tekst i logikę disclaimerów YMYL.
Używany przez: prompt_builder.py, content_editorial.py, export_routes.py

Przed tym modułem disclaimer był hardcodowany w 4 miejscach.
Teraz: jeden import, jeden tekst, jedna funkcja.

v1.0 — 2025-02: Initial release
═══════════════════════════════════════════════════════════
"""

from typing import Optional


# ─────────────────────────────────────────────────────
# TREŚCI DISCLAIMERÓW — jedyne źródło prawdy
# ─────────────────────────────────────────────────────

DISCLAIMERS = {
    "prawo": {
        "heading": "Zastrzeżenie prawne",
        "body": (
            "Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i nie stanowi porady prawnej. W indywidualnych sprawach "
            "zalecamy konsultację z wykwalifikowanym prawnikiem."
        ),
    },
    "medycyna": {
        "heading": "Zastrzeżenie medyczne",
        "body": (
            "Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i edukacyjny. Nie stanowi porady medycznej ani nie zastępuje "
            "konsultacji z lekarzem lub innym wykwalifikowanym specjalistą."
        ),
    },
    "finanse": {
        "heading": "Zastrzeżenie finansowe",
        "body": (
            "Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i nie stanowi porady finansowej ani rekomendacji inwestycyjnej. "
            "Przed podjęciem decyzji finansowych zalecamy konsultację "
            "z licencjonowanym doradcą."
        ),
    },
}

# Słowo-klucz do wykrywania istniejącego disclaimera
DISCLAIMER_DETECT_KEYWORD = "zastrzeżenie"

# Aliasy kategorii — app.py używa "zdrowie" i "medycyna" zamiennie
_CATEGORY_ALIASES = {
    "zdrowie": "medycyna",
    "finance": "finanse",
}


# ─────────────────────────────────────────────────────
# FUNKCJE PUBLICZNE
# ─────────────────────────────────────────────────────

def _resolve(category: str) -> str:
    """Resolve category aliases (zdrowie→medycyna, finance→finanse)."""
    return _CATEGORY_ALIASES.get(category, category)


def get_disclaimer(category: str) -> Optional[dict]:
    """
    Zwraca dict {"heading": ..., "body": ...} lub None.
    Obsługuje aliasy: "zdrowie" → "medycyna", "finance" → "finanse".
    
    Użycie:
        d = get_disclaimer("prawo")
        if d:
            print(d["heading"])  # "Zastrzeżenie prawne"
            print(d["body"])     # "Niniejszy artykuł..."
    """
    return DISCLAIMERS.get(_resolve(category))


def has_disclaimer(text: str) -> bool:
    """Sprawdza czy tekst już zawiera disclaimer."""
    return DISCLAIMER_DETECT_KEYWORD in text.lower()


def format_disclaimer_markdown(category: str) -> str:
    """
    Zwraca disclaimer jako markdown (do export_routes).
    
    Użycie w export_routes.py:
        from ymyl_disclaimer import needs_disclaimer, format_disclaimer_markdown
        if needs_disclaimer(full_text, detected_category):
            full_text += format_disclaimer_markdown(detected_category)
    """
    d = get_disclaimer(category)
    if not d:
        return ""
    return f"\n\n---\n\n**{d['heading']}:** {d['body']}"


def format_disclaimer_html(category: str) -> str:
    """
    Zwraca disclaimer jako HTML z <hr> (do eksportu HTML w export_routes).
    """
    d = get_disclaimer(category)
    if not d:
        return ""
    return (
        f'\n\n<hr>\n<p><strong>{d["heading"]}:</strong> '
        f'{d["body"]}</p>'
    )


def format_disclaimer_html_inline(category: str) -> str:
    """
    Zwraca disclaimer jako <p> bez <hr> (do wstrzyknięcia w app.py
    PRZED scoringiem YMYL — format musi pasować do article_text).
    """
    d = get_disclaimer(category)
    if not d:
        return ""
    return (
        f'\n\n<p><strong>{d["heading"]}:</strong> '
        f'{d["body"]}</p>'
    )


def format_disclaimer_plain(category: str) -> str:
    """
    Zwraca disclaimer jako plain text (do promptu/artykułu).
    """
    d = get_disclaimer(category)
    if not d:
        return ""
    return f"\n\n{d['heading']}\n{d['body']}"


def needs_disclaimer(text: str, category: str) -> bool:
    """
    Czy tekst potrzebuje disclaimera?
    True jeśli: kategoria YMYL + brak istniejącego disclaimera.
    """
    return _resolve(category) in DISCLAIMERS and not has_disclaimer(text)


def ensure_disclaimer(text: str, category: str, fmt: str = "markdown") -> str:
    """
    Dodaje disclaimer jeśli brakuje. Zwraca tekst (z lub bez zmian).
    
    Args:
        text: treść artykułu
        category: "prawo" / "medycyna" / "finanse" / inne
        fmt: "markdown" | "html" | "plain"
    
    Returns:
        Tekst z disclaimerem (lub bez zmian jeśli niepotrzebny)
    
    Użycie (zastępuje 4 bloki if/elif w export_routes.py):
        full_text = ensure_disclaimer(full_text, detected_category, fmt="markdown")
    """
    if not needs_disclaimer(text, category):
        return text

    formatters = {
        "markdown": format_disclaimer_markdown,
        "html": format_disclaimer_html,
        "html_inline": format_disclaimer_html_inline,
        "plain": format_disclaimer_plain,
    }
    formatter = formatters.get(fmt, format_disclaimer_markdown)
    return text.rstrip() + formatter(category)


def get_prompt_instruction(category: str) -> str:
    """
    Zwraca instrukcję dla Claude do dodania disclaimera.
    
    Użycie w prompt_builder.py:
        from ymyl_disclaimer import get_prompt_instruction
        instr = get_prompt_instruction(detected_category)
        if instr:
            parts.append(instr)
    """
    d = get_disclaimer(category)
    if not d:
        return ""
    return (
        f"Na końcu artykułu dodaj zastrzeżenie (2-3 zdania):\n"
        f'  Nagłówek: "{d["heading"]}"\n'
        f'  Treść: "{d["body"]}"'
    )
