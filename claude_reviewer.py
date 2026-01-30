"""
===============================================================================
CLAUDE BATCH REVIEW PROMPT v2.0 - OPTIMIZED
===============================================================================

Zoptymalizowany prompt dla Claude do review batchy SEO.
Zawiera wszystkie kluczowe instrukcje w zwięzłej formie.

UŻYCIE:
    from claude_review_prompt_v2 import build_review_prompt_v2
    
    prompt = build_review_prompt_v2(
        text=batch_text,
        ctx={
            "topic": "ubezwłasnowolnienie",
            "keywords_required": [...],
            "missing_basic": [...],
            "is_ymyl": True,
            "batch_number": 3,
            "total_batches": 8
        }
    )

===============================================================================
"""

import json
from typing import Dict, List, Any, Optional


def build_review_prompt_v2(text: str, ctx: Dict) -> str:
    """
    Buduje zoptymalizowany prompt dla Claude review.
    
    Args:
        text: Tekst batcha do review
        ctx: Kontekst z pre_batch_info
        
    Returns:
        Prompt string
    """
    
    # Wyciągnij dane z kontekstu
    topic = ctx.get('topic', '')
    keywords_required = ctx.get('keywords_required', [])
    missing_basic = ctx.get('missing_basic', [])
    missing_extended = ctx.get('missing_extended', [])
    is_ymyl = ctx.get('is_ymyl', False)
    batch_number = ctx.get('batch_number', 1)
    total_batches = ctx.get('total_batches', 8)
    
    # Entities i triplets (jeśli dostępne)
    entities_must = ctx.get('entities_must', [])
    triplets = ctx.get('triplets', [])
    
    # Forbidden phrases
    forbidden_phrases = ctx.get('forbidden_phrases', [
        "warto podkreślić", "warto zauważyć", "warto wspomnieć",
        "należy pamiętać", "istotne jest", "kluczowe jest",
        "w kontekście", "ogólnie rzecz biorąc", "podsumowując",
        "bez wątpienia", "nie ulega wątpliwości"
    ])
    
    # Sekcja YMYL/Legal (warunkowa)
    ymyl_section = ""
    if is_ymyl:
        ymyl_section = """
═══════════════════════════════════════════════════════════════
⚖️ WYMOGI YMYL/LEGAL
═══════════════════════════════════════════════════════════════
CYTATY PRZEPISÓW:
✓ POPRAWNIE: "art. 13 § 1 k.c.", "art. 544 k.p.c."
✗ BŁĘDNIE: "artykuł 13", "Art. 13", "zgodnie z artykułem"

TERMINOLOGIA PRAWNA:
• kurator ≠ opiekun (TO RÓŻNE INSTYTUCJE!)
• orzeczenie ≠ wyrok (różne rodzaje rozstrzygnięć)
• ubezwłasnowolnienie częściowe ≠ całkowite

STYL:
• Używaj strony biernej dla obiektywizmu
• Cytuj przepisy z pełną sygnaturą
• Unikaj kategorycznych stwierdzeń bez podstawy prawnej
"""

    # Sekcja entities (warunkowa)
    entities_section = ""
    if entities_must:
        entities_list = ", ".join([e.get('entity', e) if isinstance(e, dict) else str(e) for e in entities_must[:5]])
        entities_section = f"""
ENCJE DO UŻYCIA (MUST):
{entities_list}
→ Każda encja powinna być WYJAŚNIONA, nie tylko wspomniana
"""

    # Sekcja triplets (warunkowa)
    triplets_section = ""
    if triplets:
        triplet_examples = []
        for t in triplets[:3]:
            if isinstance(t, dict):
                subj = t.get('subject', '')
                verb = t.get('verb', '')
                obj = t.get('object', '')
                triplet_examples.append(f"  • {subj} → {verb} → {obj}")
        if triplet_examples:
            triplets_section = f"""
RELACJE DO WYRAŻENIA (semantic OK):
{chr(10).join(triplet_examples)}
→ Akceptowane: aktywna/bierna/synonim formy
"""

    # Sekcja keywords
    keywords_section = ""
    if keywords_required:
        kw_list = []
        for kw in keywords_required[:8]:
            if isinstance(kw, dict):
                kw_list.append(f"  • \"{kw.get('keyword', '')}\" (×{kw.get('count', 1)})")
            else:
                kw_list.append(f"  • \"{kw}\"")
        keywords_section = f"""
FRAZY WYMAGANE W TYM BATCHU:
{chr(10).join(kw_list)}
"""

    # Sekcja missing
    missing_section = ""
    if missing_basic or missing_extended:
        missing_items = []
        for kw in missing_basic[:3]:
            missing_items.append(f"  🔴 BASIC: \"{kw}\" (MUSI być)")
        for kw in missing_extended[:2]:
            missing_items.append(f"  🟡 EXTENDED: \"{kw}\" (bonus)")
        if missing_items:
            missing_section = f"""
BRAKUJĄCE FRAZY - WPLEĆ NATURALNIE:
{chr(10).join(missing_items)}
"""

    # Główny prompt
    prompt = f"""Jesteś ekspertem SEO i redaktorem tekstów polskich. Przejrzyj batch artykułu.

═══════════════════════════════════════════════════════════════
📋 KONTEKST
═══════════════════════════════════════════════════════════════
Temat: {topic}
Batch: {batch_number}/{total_batches}
Typ: {"YMYL/Legal" if is_ymyl else "Standard"}
{keywords_section}{missing_section}{entities_section}{triplets_section}
═══════════════════════════════════════════════════════════════
📝 TEKST DO REVIEW
═══════════════════════════════════════════════════════════════
{text}

═══════════════════════════════════════════════════════════════
🔍 KRYTERIA OCENY
═══════════════════════════════════════════════════════════════

1️⃣ HUMANIZACJA (KRYTYCZNE!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZDANIA - zróżnicuj długość:
• KRÓTKIE (3-8 słów): 20-25% → "To ważne. Sąd decyduje."
• ŚREDNIE (10-18 słów): 50-60% → normalne zdania
• DŁUGIE (22-35 słów): 15-25% → złożone wyjaśnienia

AKAPITY - zróżnicuj liczbę zdań:
• NIE: 4, 4, 4, 4 zdania (monotonne = AI!)
• TAK: 2, 5, 3, 6 zdań (naturalne)

FORBIDDEN PHRASES (USUŃ!):
{', '.join(f'"{p}"' for p in forbidden_phrases[:8])}

ZAMIEŃ:
• "należy pamiętać" → "Pamiętaj:"
• "istotne jest" → "Ważne:"
• "warto zauważyć" → [usuń, napisz wprost]

2️⃣ POPRAWNOŚĆ JĘZYKOWA
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Sprawdź odmianę przypadków
• Sprawdź zgodność liczby/rodzaju
• Wykryj powtórzenia w sąsiednich zdaniach
• Wykryj tautologie ("ubezwłasnowolniony całkowicie w pełni")
{ymyl_section}
3️⃣ SEO & STRUKTURA
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Frazy wplataj NATURALNIE (nie na siłę)
• H2/H3 powinny zawierać frazę lub synonim
• Unikaj keyword stuffing (max density 2-3%)

═══════════════════════════════════════════════════════════════
📤 ODPOWIEDŹ (TYLKO JSON!)
═══════════════════════════════════════════════════════════════
{{
  "status": "APPROVED|CORRECTED|REJECTED",
  
  "issues": [
    {{
      "type": "FORBIDDEN_PHRASE|GRAMMAR|REPETITION|HUMANIZATION|KEYWORD|YMYL",
      "severity": "critical|warning|suggestion",
      "location": "akapit X / zdanie Y",
      "description": "opis problemu",
      "fix": "proponowana poprawka",
      "fix_applied": true|false
    }}
  ],
  
  "humanization_score": {{
    "sentence_variety": 0-100,
    "paragraph_variety": 0-100,
    "forbidden_phrases_found": ["lista znalezionych"],
    "ai_patterns_detected": true|false
  }},
  
  "corrected_text": "PEŁNY poprawiony tekst (jeśli status=CORRECTED)",
  
  "summary": "1-2 zdania podsumowania"
}}

═══════════════════════════════════════════════════════════════
⚠️ ZASADY DECYZJI
═══════════════════════════════════════════════════════════════
APPROVED = tekst OK, max 2 drobne sugestie
CORRECTED = naprawiłeś problemy, zwróć corrected_text
REJECTED = >3 critical issues LUB brak kluczowej frazy BASIC

PREFERUJ CORRECTED nad REJECTED!
→ Lepiej naprawić niż odrzucić (oszczędność tokenów)
→ REJECTED tylko gdy tekst wymaga przepisania od zera

Odpowiedz TYLKO poprawnym JSON (bez markdown, bez ```).
"""

    return prompt


def build_review_prompt_minimal(text: str, ctx: Dict) -> str:
    """
    Minimalistyczna wersja promptu (dla szybkości/kosztów).
    ~50% krótszy, skupia się na najważniejszym.
    """
    
    topic = ctx.get('topic', '')
    missing_basic = ctx.get('missing_basic', [])
    is_ymyl = ctx.get('is_ymyl', False)
    
    forbidden = ["warto podkreślić", "warto zauważyć", "należy pamiętać", "w kontekście"]
    
    ymyl_note = "⚖️ YMYL: cytaty jako 'art. X k.c.', kurator≠opiekun" if is_ymyl else ""
    
    missing_note = ""
    if missing_basic:
        missing_note = f"WPLEĆ: {', '.join(missing_basic[:3])}"
    
    return f"""Review tekstu SEO. Temat: {topic}

TEKST:
{text}

SPRAWDŹ:
1. Forbidden phrases: {', '.join(forbidden)} → USUŃ
2. Zdania: mix krótkich (3-8 słów) i długich (20+ słów)
3. Akapity: różna liczba zdań (nie 4,4,4,4)
4. Gramatyka polska
{ymyl_note}
{missing_note}

JSON:
{{"status":"APPROVED|CORRECTED|REJECTED","issues":[{{"type":"...","severity":"critical|warning","description":"...","fix_applied":bool}}],"corrected_text":"...jeśli CORRECTED","summary":"..."}}

CORRECTED > REJECTED (napraw zamiast odrzucać).
Tylko JSON."""


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================

if __name__ == "__main__":
    # Test
    test_text = """h2: Procedura ubezwłasnowolnienia

Warto podkreślić, że procedura ubezwłasnowolnienia jest złożona. Należy pamiętać o wielu aspektach. 
W kontekście prawa cywilnego istotne jest zachowanie wszystkich wymogów formalnych.

Sąd okręgowy rozpatruje wniosek. Sąd okręgowy powołuje biegłych. Sąd okręgowy wydaje orzeczenie.

Kurator sprawuje opiekę nad osobą ubezwłasnowolnioną i zarządza jej majątkiem w sposób odpowiedni."""

    test_ctx = {
        "topic": "ubezwłasnowolnienie całkowite",
        "keywords_required": [
            {"keyword": "ubezwłasnowolnienie", "count": 2},
            {"keyword": "sąd okręgowy", "count": 1}
        ],
        "missing_basic": ["choroba psychiczna"],
        "missing_extended": ["kurator sądowy"],
        "is_ymyl": True,
        "batch_number": 2,
        "total_batches": 6,
        "entities_must": [
            {"entity": "sąd okręgowy", "priority": "MUST"},
            {"entity": "kurator", "priority": "MUST"}
        ],
        "triplets": [
            {"subject": "sąd", "verb": "powołuje", "object": "biegłych"},
            {"subject": "kurator", "verb": "zarządza", "object": "majątkiem"}
        ]
    }
    
    print("=" * 60)
    print("PROMPT v2.0 (FULL)")
    print("=" * 60)
    prompt_full = build_review_prompt_v2(test_text, test_ctx)
    print(prompt_full)
    print(f"\nDługość: {len(prompt_full)} znaków")
    
    print("\n" + "=" * 60)
    print("PROMPT MINIMAL")
    print("=" * 60)
    prompt_min = build_review_prompt_minimal(test_text, test_ctx)
    print(prompt_min)
    print(f"\nDługość: {len(prompt_min)} znaków")
