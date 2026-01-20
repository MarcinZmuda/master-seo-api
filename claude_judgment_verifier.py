# claude_judgment_verifier.py
# BRAJEN Legal Module v3.2 - Weryfikacja orzeczeń przez Claude
# Claude wybiera najlepsze orzeczenia zamiast prostego scoringu

"""
===============================================================================
🤖 CLAUDE JUDGMENT VERIFIER v3.2
===============================================================================

Zamiast głupiego scoringu regex, Claude:
1. Analizuje czy orzeczenie PASUJE do tematu artykułu
2. Ocenia kierunek (za/przeciw)
3. Wybiera 2 najlepsze z 10-15 kandydatów

Koszt: ~300 input + ~200 output = ~500 tokenów = ~$0.0003 per artykuł (Haiku)

===============================================================================
"""

import os
import json
from typing import Dict, List, Any, Optional
from anthropic import Anthropic

# ============================================================================
# KONFIGURACJA
# ============================================================================

CLAUDE_MODEL = "claude-3-haiku-20240307"  # Najtańszy, wystarczy do klasyfikacji
MAX_JUDGMENTS_TO_VERIFY = 10
MAX_JUDGMENTS_TO_SELECT = 2

# ============================================================================
# KLIENT ANTHROPIC
# ============================================================================

_client = None

def get_anthropic_client() -> Optional[Anthropic]:
    """Zwraca singleton klienta Anthropic."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[CLAUDE_VERIFIER] ⚠️ Brak ANTHROPIC_API_KEY")
            return None
        _client = Anthropic(api_key=api_key)
    return _client


# ============================================================================
# WERYFIKACJA ORZECZEŃ
# ============================================================================

def verify_judgments_with_claude(
    article_topic: str,
    judgments: List[Dict],
    max_to_select: int = MAX_JUDGMENTS_TO_SELECT
) -> Dict[str, Any]:
    """
    Claude wybiera najlepsze orzeczenia dla tematu artykułu.
    
    Args:
        article_topic: Temat artykułu (np. "alimenty na dziecko")
        judgments: Lista orzeczeń z SAOS (max 10-15)
        max_to_select: Ile wybrać (default 2)
        
    Returns:
        Dict z wybranymi orzeczeniami i uzasadnieniem
    """
    client = get_anthropic_client()
    
    if not client:
        return {
            "status": "ERROR",
            "error": "Anthropic client not available",
            "selected": [],
            "fallback": True
        }
    
    if not judgments:
        return {
            "status": "NO_JUDGMENTS",
            "selected": [],
            "fallback": False
        }
    
    # Ogranicz liczbę do weryfikacji
    judgments_to_verify = judgments[:MAX_JUDGMENTS_TO_VERIFY]
    
    # Przygotuj prompt
    prompt = _build_verification_prompt(article_topic, judgments_to_verify, max_to_select)
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parsuj odpowiedź
        result = _parse_claude_response(response.content[0].text, judgments_to_verify)
        
        return {
            "status": "OK",
            "selected": result["selected"],
            "reasoning": result.get("reasoning", ""),
            "model": CLAUDE_MODEL,
            "fallback": False
        }
        
    except Exception as e:
        print(f"[CLAUDE_VERIFIER] ❌ Error: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "selected": [],
            "fallback": True
        }


def _build_verification_prompt(
    topic: str,
    judgments: List[Dict],
    max_to_select: int
) -> str:
    """Buduje prompt dla Claude'a - v3.2 z weryfikacją PRZEDMIOTU sprawy."""
    
    judgments_text = ""
    for i, j in enumerate(judgments, 1):
        signature = j.get('signature', '')
        excerpt = j.get('excerpt', '')[:400]
        judgments_text += f"""
[{i}] {j.get('citation', f'Orzeczenie {i}')}
Sygnatura: {signature}
Fragment: "{excerpt}..."
"""
    
    # Wykryj czy temat dotyczy przestępstwa
    criminal_keywords = ["przestępstwo", "art. 209", "niealimentacja", "niepłacenie alimentów", 
                         "kara", "wyrok karny", "skazany", "oskarżony"]
    is_criminal = any(kw in topic.lower() for kw in criminal_keywords)
    
    division_hint = ""
    if not is_criminal:
        division_hint = """
PRIORYTET WYDZIAŁÓW (KRYTYCZNE!):
- Sprawy rodzinne/cywilne → preferuj sygnatury: C, Ca, ACa, RC, CZP, CSK
- ⛔ ODRZUĆ sygnatury: K, Ka (Karne), U, Ua (Ubezpieczenia społeczne)
- Sygnatura "K" = sprawa KARNA → NIE PASUJE do tematu cywilnego!
- Sygnatura "U" = sprawa UBEZPIECZEŃ SPOŁECZNYCH → NIE PASUJE!
"""
    else:
        division_hint = """
PRIORYTET WYDZIAŁÓW:
- Temat dotyczy przestępstwa → preferuj sygnatury: K, Ka, AKa, KK, KZP
- Sygnatura "I KZP" = uchwała SN (Izba Karna) → bardzo wartościowe!
"""
    
    prompt = f"""Jesteś ekspertem prawnym. Analizujesz orzeczenia sądowe dla artykułu SEO.

TEMAT ARTYKUŁU: "{topic}"

KANDYDACI (orzeczenia z SAOS):
{judgments_text}

ZADANIE:
Wybierz {max_to_select} orzeczenia które NAJLEPIEJ pasują do tematu artykułu.

⛔ KRYTYCZNE KRYTERIUM - PRZEDMIOT SPRAWY vs KONTEKST UBOCZNY:

Orzeczenie PASUJE tylko jeśli temat artykułu jest PRZEDMIOTEM sprawy (głównym zagadnieniem).
Orzeczenie NIE PASUJE jeśli temat jest tylko KONTEKSTEM UBOCZNYM (wspomniany przy okazji).

Przykłady dla tematu "ubezwłasnowolnienie":
✅ PASUJE: Sprawa O ubezwłasnowolnienie (sygnatura C, Ca) - to jest przedmiot sprawy
❌ NIE PASUJE: Sprawa karna (sygnatura K) gdzie oskarżony "jest ubezwłasnowolniony" - to tylko kontekst
❌ NIE PASUJE: Sprawa o rentę (sygnatura U) gdzie wnioskodawca "został ubezwłasnowolniony" - to tylko kontekst

Przykłady dla tematu "alimenty":
✅ PASUJE: Sprawa O alimenty (art. 133 KRO) - przedmiot sprawy
❌ NIE PASUJE: Sprawa karna o niealimentację (art. 209 KK) - to inna kategoria!
❌ NIE PASUJE: Sprawa spadkowa gdzie wspomina "obowiązek alimentacyjny" - kontekst uboczny

KRYTERIA WYBORU:

1. SYGNATURA (NAJWAŻNIEJSZE!):
   - Sprawdź czy wydział pasuje do tematu
   - "K" = karny, "U" = ubezpieczenia, "C/Ca/ACa" = cywilny
{division_hint}

2. WERYFIKACJA PRZEPISU:
   Znasz treść polskich kodeksów (KK, KC, KRO, KPC, KPK, KP).
   Sprawdź czy przepis cytowany w orzeczeniu PASUJE do tematu artykułu.
   
   - Temat "ubezwłasnowolnienie" → art. 13, 16 KC (✅) | art. 178a KK (❌ to jazda po alkoholu!)
   - Temat "alimenty" → art. 133 KRO (✅) | art. 209 KK (❌ to sprawa karna!)

3. KONTEKST MERYTORYCZNY:
   Fragment orzeczenia musi zawierać wartościową tezę prawną związaną z tematem.
   NIE wybieraj orzeczeń gdzie temat jest tylko wspomniany "przy okazji".

ODPOWIEDZ W FORMACIE JSON:
{{
    "selected": [
        {{
            "index": 1,
            "direction": "za|przeciw|neutralny",
            "article_cited": "art. X ustawy Y",
            "is_main_subject": true,
            "division_ok": true,
            "reason": "krótkie uzasadnienie (max 20 słów)"
        }}
    ],
    "rejected_reason": "dlaczego pozostałe nie pasują (max 30 słów)"
}}

WAŻNE:
- "is_main_subject": true TYLKO jeśli temat jest PRZEDMIOTEM sprawy, nie kontekstem!
- "division_ok": true TYLKO jeśli wydział (z sygnatury) pasuje do tematu!
- Jeśli ŻADNE orzeczenie nie pasuje → zwróć pustą listę selected i wyjaśnij dlaczego

Odpowiedz TYLKO JSON, bez dodatkowego tekstu."""

    return prompt


def _parse_claude_response(
    response_text: str,
    original_judgments: List[Dict]
) -> Dict[str, Any]:
    """Parsuje odpowiedź Claude'a i mapuje na oryginalne orzeczenia. v3.2"""
    
    # Wyczyść response z markdown
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: weź pierwsze 2
        print("[CLAUDE_VERIFIER] ⚠️ JSON parse error, using fallback")
        return {
            "selected": original_judgments[:2],
            "reasoning": "Nie udało się sparsować odpowiedzi Claude'a"
        }
    
    selected = []
    for item in data.get("selected", []):
        idx = item.get("index", 0) - 1  # Claude zwraca 1-indexed
        if 0 <= idx < len(original_judgments):
            # 🆕 v3.2: Sprawdź czy to PRZEDMIOT sprawy i czy wydział pasuje
            is_main_subject = item.get("is_main_subject", True)
            division_ok = item.get("division_ok", True)
            article_matches = item.get("article_matches_topic", True)
            
            # Odrzuć jeśli którekolwiek kryterium nie jest spełnione
            if is_main_subject == False:
                print(f"[CLAUDE_VERIFIER] ⚠️ Skipping [{idx+1}] - temat to tylko kontekst uboczny")
                continue
            if division_ok == False:
                print(f"[CLAUDE_VERIFIER] ⚠️ Skipping [{idx+1}] - wydział nie pasuje do tematu")
                continue
            if article_matches == False:
                print(f"[CLAUDE_VERIFIER] ⚠️ Skipping [{idx+1}] - przepis nie pasuje do tematu")
                continue
                
            judgment = original_judgments[idx].copy()
            judgment["direction"] = item.get("direction", "neutralny")
            judgment["claude_reason"] = item.get("reason", "")
            judgment["article_cited"] = item.get("article_cited", "")
            judgment["verified_by_claude"] = True
            judgment["is_main_subject"] = is_main_subject
            judgment["division_ok"] = division_ok
            selected.append(judgment)
    
    return {
        "selected": selected,
        "reasoning": data.get("rejected_reason", "")
    }


# ============================================================================
# FALLBACK: Prosty scoring (gdy Claude niedostępny)
# ============================================================================

def simple_scoring_fallback(
    judgments: List[Dict],
    max_to_select: int = 2
) -> List[Dict]:
    """
    Prosty scoring jako fallback gdy Claude niedostępny.
    Używa tylko podstawowych heurystyk.
    """
    import re
    
    scored = []
    for j in judgments:
        text = (j.get("full_text", "") or j.get("excerpt", "")).lower()
        score = 0
        
        # +40: zawiera przepis
        if re.search(r'art\.\s*\d+', text):
            score += 40
        
        # +30: ma tezę
        if any(p in text for p in ["zdaniem sądu", "należy uznać", "sąd zważył"]):
            score += 30
        
        # +20: nie jest proceduralne
        if not any(p in text[:500] for p in ["umarza", "odrzuca", "zwraca sprawę"]):
            score += 20
        
        j_copy = j.copy()
        j_copy["score"] = score
        j_copy["direction"] = "neutralny"
        j_copy["verified_by_claude"] = False
        scored.append(j_copy)
    
    # Sortuj i weź najlepsze
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_to_select]


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def select_best_judgments(
    article_topic: str,
    judgments: List[Dict],
    max_to_select: int = 2,
    use_claude: bool = True
) -> Dict[str, Any]:
    """
    Główna funkcja - wybiera najlepsze orzeczenia.
    
    Próbuje Claude, fallback na prosty scoring.
    
    Args:
        article_topic: Temat artykułu
        judgments: Lista orzeczeń z SAOS
        max_to_select: Ile wybrać
        use_claude: Czy używać Claude (default True)
        
    Returns:
        Dict z wybranymi orzeczeniami
    """
    if not judgments:
        return {
            "status": "NO_JUDGMENTS",
            "selected": [],
            "method": "none"
        }
    
    # Próbuj Claude
    if use_claude:
        result = verify_judgments_with_claude(article_topic, judgments, max_to_select)
        
        if result["status"] == "OK" and result["selected"]:
            return {
                "status": "OK",
                "selected": result["selected"],
                "method": "claude",
                "reasoning": result.get("reasoning", "")
            }
    
    # Fallback na prosty scoring
    print("[JUDGMENT_VERIFIER] ⚠️ Using fallback scoring")
    selected = simple_scoring_fallback(judgments, max_to_select)
    
    return {
        "status": "OK",
        "selected": selected,
        "method": "fallback_scoring",
        "reasoning": "Claude niedostępny, użyto prostego scoringu"
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🤖 Claude Judgment Verifier Test\n")
    
    # Symulacja orzeczeń
    test_judgments = [
        {
            "citation": "wyrok SN z dnia 15.03.2023 (III CZP 12/23)",
            "excerpt": "Obowiązek alimentacyjny zgodnie z art. 133 KRO polega na dostarczaniu środków utrzymania odpowiadających usprawiedliwionym potrzebom uprawnionego."
        },
        {
            "citation": "wyrok SA Warszawa z dnia 10.01.2022 (I ACa 456/22)",
            "excerpt": "Sprzedaż alkoholu nieletnim stanowi naruszenie art. 43 ustawy o wychowaniu w trzeźwości."
        },
        {
            "citation": "wyrok SO Kraków z dnia 05.06.2022 (III Ca 789/22)",
            "excerpt": "Przy ustalaniu wysokości alimentów sąd bierze pod uwagę możliwości zarobkowe zobowiązanego zgodnie z art. 135 KRO."
        }
    ]
    
    result = select_best_judgments(
        article_topic="alimenty na dziecko",
        judgments=test_judgments,
        use_claude=True
    )
    
    print(f"Status: {result['status']}")
    print(f"Method: {result['method']}")
    print(f"Selected: {len(result['selected'])}")
    
    for j in result["selected"]:
        print(f"\n  📄 {j['citation']}")
        print(f"     Direction: {j.get('direction', '?')}")
        print(f"     Reason: {j.get('claude_reason', 'N/A')}")
