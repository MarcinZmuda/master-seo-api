"""
content_editorial.py — Merytoryczny editorial (v1.0)
=======================================================
NOWY ETAP w pipeline, uruchamiany PO merge batchów, PRZED final_review i editorial językowym.

Kolejność pipeline:
  1. Batch review        (claude_reviewer.py)     — podczas generacji
  2. Merge batchów
  3. ► CONTENT EDITORIAL (ten moduł)              — ekspert domenowy
  4. Final review        (final_review_routes.py) — keywords na czystym tekście
  5. Editorial językowy  (export_routes.py)       — polerowanie języka

Logika:
  - YMYL (prawo/medycyna/finanse): prompt "ekspert domenowy z 20-letnim stażem"
    → blokuje artykuł jeśli błąd CRITICAL
  - Non-YMYL: prompt "redaktor ze zdrowym rozsądkiem"
    → tylko ostrzeżenie, nie blokuje

Model: claude (przez ANTHROPIC_API_KEY)
"""

import os
import re
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

import anthropic

try:
    from llm_retry import llm_call_with_retry
except ImportError:
    def llm_call_with_retry(fn, *a, **kw):
        return fn(*a, **kw)

try:
    from prompt_logger import log_prompt as _log_prompt
except ImportError:
    def _log_prompt(*a, **kw): pass

logger = logging.getLogger(__name__)

# ================================================================
# KONFIGURACJA
# ================================================================

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Kategorie YMYL i ich "rola eksperta"
YMYL_EXPERT_ROLES = {
    "prawo": {
        "role": "adwokat i redaktor naczelny pisma prawniczego z 20-letnim stażem",
        "domain": "prawo",
        "critical_checks": [
            "Czy artykuł zawiera placeholder 'odpowiednich przepisów prawa' lub podobny? → BŁĄD KRYTYCZNY",
            "Czy używa błędnej terminologii: 'aresztowanie' zamiast 'pozbawienie wolności', 'uwięzienie', 'alkohol w/z natury', 'alkohol z urodzenia', 'promile z natury/urodzenia', 'opilstwo' zamiast 'stan nietrzeźwości', 'obsługiwał pojazd' zamiast 'prowadził pojazd'? → BŁĄD KRYTYCZNY — KAŻDE WYSTĄPIENIE POPRAW",
            "Czy podane wymiary kar są aktualne? (art. 178a § 1 KK: do 3 lat od 1.10.2023, recydywa: do 5 lat) → BŁĄD KRYTYCZNY jeśli nieaktualne",
            "Czy brakuje kluczowych instytucji prawnych istotnych dla tematu? (np. konfiskata pojazdu art. 44b KK dla jazdy po alkoholu) → BŁĄD POWAŻNY",
            "Czy artykuł zawiera wymyślone sygnatury orzeczeń, daty rozporządzeń, numery ustaw? → HALUCYNACJA KRYTYCZNA",
        ],
        "forbidden_terms": [
            # --- błędna terminologia karna ---
            ("aresztowanie", "pozbawienie wolności"),
            ("bezwzględne aresztowanie", "bezwzględne pozbawienie wolności"),
            ("uwięzienie", "pozbawienie wolności"),
            # --- halucynacje alkohol ---
            ("alkohol w naturze", "stężenie alkoholu we krwi"),
            ("alkohol z natury", "stężenie alkoholu we krwi"),
            ("alkohol z urodzenia", "stężenie alkoholu we krwi"),
            ("promile z natury", "zawartość alkoholu we krwi"),
            ("promile z urodzenia", "zawartość alkoholu we krwi"),
            ("promile alkoholu z", "stężenie alkoholu"),
            ("stężenie alkoholu z natury", "stężenie alkoholu we krwi"),
            ("stężenie alkoholu z urodzenia", "stężenie alkoholu we krwi"),
            # --- archaizmy i błędna terminologia ---
            ("opilstwo", "stan nietrzeźwości"),
            ("pijaństwo", "stan nietrzeźwości"),  # w kontekście prawnym
            ("obsługiwał pojazd", "prowadził pojazd"),
            ("obsługi pojazdu", "prowadzenia pojazdu"),
            ("zakaz obsługi pojazdu", "zakaz prowadzenia pojazdu"),
            ("zakaz obsługi pojazdów", "zakaz prowadzenia pojazdów"),
            # --- phantom przepisy ---
            ("odpowiednich przepisów prawa", "[konkretny artykuł ustawy]"),
            ("właściwych regulacji prawnych", "[konkretny artykuł ustawy]"),
            ("stosownych przepisów", "[konkretny artykuł ustawy]"),
        ],
    },
    "medycyna": {
        "role": "lekarz i redaktor naczelny pisma medycznego z 20-letnim stażem",
        "domain": "medycyna",
        "critical_checks": [
            "Czy artykuł zawiera wymyślone badania, statystyki, nazwy leków, dawki? → HALUCYNACJA KRYTYCZNA",
            "Czy podane informacje o leczeniu/dawkach są bezpieczne i zgodne z aktualną wiedzą medyczną?",
            "Czy używa poprawnej terminologii medycznej (nie potocznej)? → BŁĄD POWAŻNY jeśli nie",
            "Czy zawiera niebezpieczne rady zdrowotne bez zastrzeżenia 'skonsultuj z lekarzem'?",
        ],
        "forbidden_terms": [],
    },
    "finanse": {
        "role": "doradca finansowy i redaktor naczelny pisma finansowego z 20-letnim stażem",
        "domain": "finanse",
        "critical_checks": [
            "Czy artykuł zawiera wymyślone stopy procentowe, kursy, dane giełdowe? → HALUCYNACJA KRYTYCZNA",
            "Czy zawiera porady inwestycyjne bez zastrzeżenia 'nie stanowi porady finansowej'?",
            "Czy podane przepisy podatkowe/prawne są aktualne?",
        ],
        "forbidden_terms": [],
    },
}

# Domyślna rola dla non-YMYL
NON_YMYL_ROLE = {
    "role": "starszy redaktor merytoryczny z 20-letnim stażem w prasie branżowej",
    "domain": "general",
    "critical_checks": [
        "Czy artykuł zawiera twierdzenia które brzmią technicznie/faktograficznie absurdalnie?",
        "Czy podane ceny, daty, dane techniczne wyglądają na realistyczne?",
        "Czy nie ma nonsensów semantycznych (np. odpowiednik 'promile z natury' dla innej branży)?",
        "Czy tekst jest spójny wewnętrznie (brak sprzecznych twierdzeń)?",
    ],
    "forbidden_terms": [],
}


# ================================================================
# DATACLASSES
# ================================================================

@dataclass
class ContentIssue:
    type: str           # CRITICAL | WARNING | INFO
    category: str       # terminology | hallucination | outdated | missing_content | nonsense
    description: str
    found_text: str = ""
    suggestion: str = ""
    auto_fixed: bool = False


@dataclass
class ContentEditorialResult:
    status: str                        # OK | WARNING | BLOCKED
    issues: list = field(default_factory=list)
    corrected_text: Optional[str] = None
    summary: str = ""
    score: int = 100
    domain: str = ""
    is_ymyl: bool = False
    blocked_reason: str = ""


# ================================================================
# PROMPT BUILDER
# ================================================================

def _build_system_prompt(category: str, is_ymyl: bool) -> str:
    config = YMYL_EXPERT_ROLES.get(category, NON_YMYL_ROLE) if is_ymyl else NON_YMYL_ROLE

    role = config["role"]
    checks = config["critical_checks"]
    forbidden = config.get("forbidden_terms", [])

    checks_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(checks))

    forbidden_text = ""
    if forbidden:
        lines = "\n".join(f'  ❌ "{w}" → użyj: "{c}"' for w, c in forbidden)
        forbidden_text = f"\n\nZAKAZANA TERMINOLOGIA — ZAWSZE POPRAW:\n{lines}"

    return f"""Jesteś {role}.

Twoim zadaniem jest MERYTORYCZNY przegląd artykułu — NIE styl, NIE SEO.
Sprawdzasz wyłącznie: poprawność faktyczną, terminologię, aktualność, spójność.

CZEGO SZUKASZ:
{checks_text}{forbidden_text}

ZASADY:
• Jeśli znajdziesz błąd CRITICAL — oznacz status: BLOCKED
• Jeśli tylko WARNING — oznacz status: WARNING  
• Jeśli wszystko OK — oznacz status: OK
• Popraw błędy CRITICAL i TERMINOLOGY bezpośrednio w tekście (corrected_text)
• Dla WARNING tylko opisz w issues, nie poprawiaj samodzielnie

NIE OCENIASZ: stylu, długości zdań, SEO, struktury H2 — to robi osobny moduł.
"""


def _build_user_prompt(article_text: str, topic: str, category: str, is_ymyl: bool) -> str:
    label = f"YMYL/{category.upper()}" if is_ymyl else "STANDARD"
    return f"""Artykuł: "{topic}" [{label}]

Wykonaj merytoryczny przegląd. Odpowiedz TYLKO w formacie JSON:

{{
  "status": "OK|WARNING|BLOCKED",
  "blocked_reason": "<jeśli BLOCKED: krótkie wyjaśnienie>",
  "score": <0-100>,
  "issues": [
    {{
      "type": "CRITICAL|WARNING|INFO",
      "category": "terminology|hallucination|outdated|missing_content|nonsense",
      "description": "<opis błędu>",
      "found_text": "<cytat z artykułu>",
      "suggestion": "<jak poprawić>"
    }}
  ],
  "corrected_text": "<PEŁNY poprawiony artykuł — TYLKO jeśli były błędy CRITICAL lub TERMINOLOGY>",
  "summary": "<2 zdania: co znalazłeś>"
}}

ARTYKUŁ:
{article_text}"""


# ================================================================
# GŁÓWNA FUNKCJA
# ================================================================

def run_content_editorial(
    article_text: str,
    topic: str,
    category: str = "inne",
    is_ymyl: bool = False,
) -> ContentEditorialResult:
    """
    Uruchamia merytoryczny editorial artykułu.

    Parametry:
        article_text — pełny tekst artykułu po merge batchów
        topic        — tytuł/keyword artykułu
        category     — "prawo" | "medycyna" | "finanse" | "inne"
        is_ymyl      — True jeśli YMYL

    Zwraca:
        ContentEditorialResult z polami: status, issues, corrected_text, score
    """
    if not article_text or len(article_text.strip()) < 100:
        return ContentEditorialResult(
            status="OK",
            summary="Brak tekstu do analizy",
            domain=category,
            is_ymyl=is_ymyl,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("[CONTENT_EDITORIAL] Brak ANTHROPIC_API_KEY — pomijam")
        return ContentEditorialResult(
            status="OK",
            summary="Claude niedostępny — editorial pominięty",
            domain=category,
            is_ymyl=is_ymyl,
        )

    system_prompt = _build_system_prompt(category, is_ymyl)
    user_prompt = _build_user_prompt(article_text, topic, category, is_ymyl)

    # Log prompt
    _log_prompt(
        stage="content_editorial",
        system_prompt=system_prompt,
        user_prompt=user_prompt[:2000] + ("..." if len(user_prompt) > 2000 else ""),
        keyword=topic,
        engine="claude",
        extra={"category": category, "is_ymyl": is_ymyl},
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)

        def _call():
            return client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=6000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        response = llm_call_with_retry(_call)
        raw = response.content[0].text

        # Parse JSON
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            logger.error("[CONTENT_EDITORIAL] Brak JSON w odpowiedzi")
            return ContentEditorialResult(
                status="OK",
                summary="Błąd parsowania odpowiedzi",
                domain=category,
                is_ymyl=is_ymyl,
            )

        data = json.loads(json_match.group())

        issues = []
        for i in data.get("issues", []):
            issues.append(ContentIssue(
                type=i.get("type", "INFO"),
                category=i.get("category", ""),
                description=i.get("description", ""),
                found_text=i.get("found_text", ""),
                suggestion=i.get("suggestion", ""),
            ))

        status = data.get("status", "OK")
        corrected = data.get("corrected_text", "")

        # Dla non-YMYL: nigdy nie blokuj, tylko warn
        if not is_ymyl and status == "BLOCKED":
            status = "WARNING"

        # Jeśli corrected_text krótszy niż 50% oryginału — odrzuć (zabezpieczenie)
        if corrected and len(corrected) < len(article_text) * 0.5:
            logger.warning("[CONTENT_EDITORIAL] corrected_text za krótki — odrzucam")
            corrected = None

        return ContentEditorialResult(
            status=status,
            issues=issues,
            corrected_text=corrected if corrected else None,
            summary=data.get("summary", ""),
            score=data.get("score", 100),
            domain=category,
            is_ymyl=is_ymyl,
            blocked_reason=data.get("blocked_reason", ""),
        )

    except Exception as e:
        logger.error(f"[CONTENT_EDITORIAL] Błąd: {e}")
        return ContentEditorialResult(
            status="OK",
            summary=f"Błąd: {e}",
            domain=category,
            is_ymyl=is_ymyl,
        )


def result_to_dict(result: ContentEditorialResult) -> dict:
    """Serializuje wynik do dict (dla Firestore / JSON response)."""
    return {
        "status": result.status,
        "score": result.score,
        "domain": result.domain,
        "is_ymyl": result.is_ymyl,
        "blocked": result.status == "BLOCKED",
        "blocked_reason": result.blocked_reason,
        "issue_count": len(result.issues),
        "critical_count": sum(1 for i in result.issues if i.type == "CRITICAL"),
        "warning_count": sum(1 for i in result.issues if i.type == "WARNING"),
        "issues": [
            {
                "type": i.type,
                "category": i.category,
                "description": i.description,
                "found_text": i.found_text,
                "suggestion": i.suggestion,
            }
            for i in result.issues
        ],
        "has_corrected_text": bool(result.corrected_text),
        "summary": result.summary,
    }
