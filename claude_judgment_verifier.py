# claude_judgment_verifier.py
# BRAJEN Legal Module v3.3 - Weryfikacja orzeczeń przez Claude
# Claude wnioskuje NA ŻYWO czy artykuł ustawy pasuje do tematu

"""
===============================================================================
🤖 CLAUDE JUDGMENT VERIFIER v3.3
===============================================================================

Claude używa swojej WIEDZY O KODEKSACH do oceny orzeczeń:
1. Znajduje artykuł cytowany w orzeczeniu (np. "art. 13 KC")
2. WNIOSKUJE co ten artykuł reguluje (zna treść kodeksów!)
3. Ocenia czy PASUJE do tematu artykułu
4. Sprawdza czy temat to PRZEDMIOT sprawy czy tylko kontekst

Przykład wnioskowania:
- Temat: "ubezwłasnowolnienie"
- Orzeczenie cytuje: "art. 178a KK"
- Claude wie: art. 178a KK = jazda po alkoholu
- Wniosek: ❌ NIE PASUJE!

Koszt: ~500-700 tokenów = ~$0.0004 per artykuł (Haiku)

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
    """Buduje prompt dla Claude'a - v3.3 z wnioskowaniem o artykułach na żywo."""
    
    judgments_text = ""
    for i, j in enumerate(judgments, 1):
        signature = j.get('signature', '')
        excerpt = j.get('excerpt', '')[:500]  # Więcej tekstu dla lepszej analizy
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
⛔ FILTR WYDZIAŁÓW (sprawdź sygnaturę!):
- Sygnatury C, Ca, ACa, RC, CZP = cywilne/rodzinne → ✅ OK dla tematów cywilnych
- Sygnatury K, Ka, AKa = KARNE → ❌ ODRZUĆ dla tematów cywilnych!
- Sygnatury U, Ua = UBEZPIECZENIA SPOŁECZNE → ❌ ODRZUĆ dla tematów cywilnych!
"""
    else:
        division_hint = """
FILTR WYDZIAŁÓW:
- Temat dotyczy przestępstwa → sygnatury K, Ka, AKa = ✅ OK
"""
    
    prompt = f"""Jesteś ekspertem prawa polskiego. Znasz WSZYSTKIE polskie kodeksy:
- KC (Kodeks cywilny) - art. 1-1088
- KRO (Kodeks rodzinny) - art. 1-184  
- KK (Kodeks karny) - art. 1-363
- KPC (Kodeks postępowania cywilnego)
- KPK (Kodeks postępowania karnego)
- KP (Kodeks pracy)

TEMAT ARTYKUŁU: "{topic}"

KANDYDACI (orzeczenia z SAOS):
{judgments_text}

ZADANIE:
Wybierz {max_to_select} orzeczenia które NAJLEPIEJ pasują do tematu artykułu.

═══════════════════════════════════════════════════════════════
🔴 KLUCZOWE KRYTERIUM: WERYFIKACJA ARTYKUŁU USTAWY
═══════════════════════════════════════════════════════════════

Użyj swojej wiedzy o polskich kodeksach! Dla każdego orzeczenia:
1. Znajdź cytowany artykuł (np. "art. 13 KC", "art. 178a KK")
2. Przypomnij sobie CO TEN ARTYKUŁ REGULUJE
3. Oceń czy to PASUJE do tematu artykułu

PRZYKŁADY WNIOSKOWANIA:

Temat: "ubezwłasnowolnienie"
- art. 13 KC → "Osoba, która ukończyła lat trzynaście, może być ubezwłasnowolniona całkowicie..." → ✅ PASUJE!
- art. 16 KC → "Osoba pełnoletnia może być ubezwłasnowolniona częściowo..." → ✅ PASUJE!
- art. 178a KK → "Kto, znajdując się w stanie nietrzeźwości, prowadzi pojazd..." → ❌ NIE PASUJE (jazda po alkoholu!)
- art. 209 KK → "Kto uchyla się od obowiązku alimentacyjnego..." → ❌ NIE PASUJE (to przestępstwo!)

Temat: "alimenty" (prawo rodzinne)
- art. 133 KRO → "Rodzice obowiązani są do świadczeń alimentacyjnych..." → ✅ PASUJE!
- art. 135 KRO → "Zakres świadczeń alimentacyjnych zależy od..." → ✅ PASUJE!
- art. 209 KK → przestępstwo niealimentacji → ❌ INNA KATEGORIA (karna vs rodzinna)!

Temat: "rozwód"
- art. 56 KRO → "Jeżeli między małżonkami nastąpił zupełny rozkład pożycia..." → ✅ PASUJE!
- art. 57 KRO → "Orzekając rozwód sąd orzeka także..." → ✅ PASUJE!

═══════════════════════════════════════════════════════════════
🔴 DRUGIE KRYTERIUM: PRZEDMIOT SPRAWY vs KONTEKST UBOCZNY
═══════════════════════════════════════════════════════════════

Orzeczenie PASUJE tylko jeśli temat jest GŁÓWNYM PRZEDMIOTEM sprawy.
NIE PASUJE jeśli temat jest tylko WSPOMNIANY przy okazji innej sprawy.

Przykład dla "ubezwłasnowolnienie":
✅ "Sąd orzeka ubezwłasnowolnienie całkowite Jana Kowalskiego..." → przedmiot sprawy
❌ "Oskarżony, będący osobą ubezwłasnowolnioną, dopuścił się..." → tylko kontekst w sprawie karnej!
{division_hint}
═══════════════════════════════════════════════════════════════

ODPOWIEDZ W FORMACIE JSON:
{{
    "selected": [
        {{
            "index": 1,
            "article_found": "art. X ustawy",
            "article_meaning": "co ten artykuł reguluje (max 10 słów)",
            "matches_topic": true,
            "is_main_subject": true,
            "division_code": "C/K/U/P",
            "direction": "za|przeciw|neutralny",
            "reason": "dlaczego pasuje (max 15 słów)"
        }}
    ],
    "rejected": [
        {{
            "index": 2,
            "reason": "dlaczego nie pasuje (max 15 słów)"
        }}
    ]
}}

WAŻNE:
- "matches_topic": true TYLKO jeśli artykuł ustawy dotyczy tematu!
- "is_main_subject": true TYLKO jeśli temat to przedmiot sprawy, nie kontekst!
- Jeśli ŻADNE nie pasuje → zwróć pustą listę "selected" i wyjaśnij w "rejected"

Odpowiedz TYLKO JSON."""

    return prompt


def _parse_claude_response(
    response_text: str,
    original_judgments: List[Dict]
) -> Dict[str, Any]:
    """Parsuje odpowiedź Claude'a i mapuje na oryginalne orzeczenia. v3.3"""
    
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
            "reasoning": "Nie udało się sparsować odpowiedzi Claude'a",
            "rejected": []
        }
    
    selected = []
    for item in data.get("selected", []):
        idx = item.get("index", 0) - 1  # Claude zwraca 1-indexed
        if 0 <= idx < len(original_judgments):
            # 🆕 v3.3: Sprawdź wszystkie kryteria
            matches_topic = item.get("matches_topic", True)
            is_main_subject = item.get("is_main_subject", True)
            
            # Odrzuć jeśli którekolwiek kryterium nie jest spełnione
            if matches_topic == False:
                print(f"[CLAUDE_VERIFIER] ⚠️ Skipping [{idx+1}] - artykuł ustawy nie pasuje do tematu")
                continue
            if is_main_subject == False:
                print(f"[CLAUDE_VERIFIER] ⚠️ Skipping [{idx+1}] - temat to tylko kontekst uboczny")
                continue
                
            judgment = original_judgments[idx].copy()
            judgment["direction"] = item.get("direction", "neutralny")
            judgment["claude_reason"] = item.get("reason", "")
            judgment["article_cited"] = item.get("article_found", "")
            judgment["article_meaning"] = item.get("article_meaning", "")
            judgment["verified_by_claude"] = True
            judgment["matches_topic"] = matches_topic
            judgment["is_main_subject"] = is_main_subject
            judgment["division_code"] = item.get("division_code", "")
            selected.append(judgment)
    
    # 🆕 v3.3: Zbierz info o odrzuconych
    rejected_info = data.get("rejected", [])
    rejected_summary = "; ".join([f"[{r.get('index')}]: {r.get('reason', '')}" for r in rejected_info[:3]])
    
    return {
        "selected": selected,
        "reasoning": rejected_summary if rejected_summary else data.get("rejected_reason", ""),
        "rejected": rejected_info
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
