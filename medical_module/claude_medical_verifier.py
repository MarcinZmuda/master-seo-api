"""
===============================================================================
🤖 CLAUDE MEDICAL VERIFIER v1.0
===============================================================================
Weryfikacja i scoring publikacji medycznych przez Claude.

Funkcje:
1. Wybór najlepszych publikacji dla tematu artykułu
2. Ocena poziomu dowodów (Evidence Level)
3. Sprawdzenie zgodności z konsensusem naukowym
4. Filtrowanie nieodpowiednich źródeł

Hierarchia dowodów (Evidence-Based Medicine):
- Level 1: Meta-analizy, Systematic Reviews
- Level 2: RCT (Randomized Controlled Trials)
- Level 3: Cohort studies, Case-control studies
- Level 4: Case series, Case reports
- Level 5: Expert opinion

Koszt: ~$0.001 per weryfikację (Claude Haiku)
===============================================================================
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class VerifierConfig:
    """Konfiguracja weryfikatora."""
    
    # Model Claude
    MODEL: str = "claude-haiku-4-5-20251001"  # Najtańszy, wystarczy
    MAX_TOKENS: int = 1000
    
    # Limity
    MAX_PUBLICATIONS_TO_VERIFY: int = 15
    MAX_TO_SELECT: int = 3
    MIN_RELEVANCE_SCORE: int = 40
    
    # Poziomy dowodów
    EVIDENCE_LEVELS: Dict[str, int] = None
    
    def __post_init__(self):
        self.EVIDENCE_LEVELS = {
            "meta-analysis": 1,
            "systematic review": 1,
            "randomized controlled trial": 2,
            "rct": 2,
            "clinical trial": 2,
            "cohort study": 3,
            "case-control": 3,
            "case series": 4,
            "case report": 4,
            "review": 3,
            "guideline": 1,
            "practice guideline": 1,
            "expert opinion": 5,
            "editorial": 5,
            "comment": 5
        }


CONFIG = VerifierConfig()


# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

_client = None


def get_anthropic_client():
    """Lazy loading klienta Anthropic."""
    global _client
    
    if _client is not None:
        return _client
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[CLAUDE_MEDICAL] ⚠️ ANTHROPIC_API_KEY not set")
        return None
    
    try:
        from anthropic import Anthropic
        _client = Anthropic(api_key=api_key)
        print(f"[CLAUDE_MEDICAL] ✅ Claude client initialized ({CONFIG.MODEL})")
        return _client
    except ImportError:
        print("[CLAUDE_MEDICAL] ❌ anthropic package not installed")
        return None
    except Exception as e:
        print(f"[CLAUDE_MEDICAL] ❌ Error: {e}")
        return None


# ============================================================================
# WERYFIKACJA PUBLIKACJI
# ============================================================================

def verify_publications_with_claude(
    article_topic: str,
    publications: List[Dict],
    max_to_select: int = None
) -> Dict[str, Any]:
    """
    Claude wybiera najlepsze publikacje dla tematu artykułu.
    
    Args:
        article_topic: Temat artykułu (np. "leczenie cukrzycy typu 2")
        publications: Lista publikacji z PubMed/ClinicalTrials
        max_to_select: Ile wybrać (default: 3)
    
    Returns:
        {
            "status": "OK",
            "selected": [...],
            "reasoning": "...",
            "model": "claude-3-haiku-..."
        }
    
    Kryteria wyboru:
    1. Relevantność do tematu
    2. Poziom dowodów (EBM hierarchy)
    3. Aktualność (preferowane ostatnie 5 lat)
    4. Jakość czasopisma
    5. Zgodność z konsensusem naukowym
    """
    client = get_anthropic_client()
    max_to_select = max_to_select or CONFIG.MAX_TO_SELECT
    
    if not client:
        # Fallback bez Claude
        return _fallback_selection(publications, max_to_select)
    
    if not publications:
        return {
            "status": "NO_PUBLICATIONS",
            "selected": [],
            "reasoning": "Brak publikacji do weryfikacji"
        }
    
    # Ogranicz liczbę do weryfikacji
    publications_to_verify = publications[:CONFIG.MAX_PUBLICATIONS_TO_VERIFY]
    
    # Buduj prompt
    prompt = _build_verification_prompt(article_topic, publications_to_verify, max_to_select)
    
    try:
        response = client.messages.create(
            model=CONFIG.MODEL,
            max_tokens=CONFIG.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parsuj odpowiedź
        result = _parse_claude_response(
            response.content[0].text,
            publications_to_verify
        )
        
        return {
            "status": "OK",
            "selected": result["selected"],
            "rejected_reasons": result.get("rejected_reasons", []),
            "reasoning": result.get("reasoning", ""),
            "model": CONFIG.MODEL,
            "method": "claude"
        }
        
    except Exception as e:
        print(f"[CLAUDE_MEDICAL] ❌ Verification error: {e}")
        # Fallback
        return _fallback_selection(publications, max_to_select)


def _build_verification_prompt(
    topic: str,
    publications: List[Dict],
    max_to_select: int
) -> str:
    """Buduje prompt dla Claude."""
    
    # Formatuj publikacje
    pubs_text = ""
    for i, pub in enumerate(publications, 1):
        # Określ źródło
        source = pub.get("source", "Unknown")
        
        # Typy publikacji
        pub_types = pub.get("publication_types", [])
        types_str = ", ".join(pub_types[:3]) if pub_types else "Not specified"
        
        # Abstract lub summary
        abstract = pub.get("abstract") or pub.get("brief_summary", "")
        abstract = abstract[:400] + "..." if len(abstract) > 400 else abstract
        
        pubs_text += f"""
═══ [{i}] ═══
Tytuł: {pub.get('title', 'N/A')}
Autorzy: {pub.get('authors_short', 'N/A')}
Źródło: {pub.get('journal', pub.get('lead_sponsor', source))} ({pub.get('year', 'N/A')})
Typ: {types_str}
Streszczenie: {abstract}
"""
    
    return f"""Jesteś ekspertem medycznym i recenzentem naukowym. Twoim zadaniem jest wybrać 
{max_to_select} NAJLEPSZE publikacje do zacytowania w artykule o temacie:

📋 TEMAT ARTYKUŁU: "{topic}"

════════════════════════════════════════════════════════════════
📚 PUBLIKACJE DO OCENY:
════════════════════════════════════════════════════════════════
{pubs_text}
════════════════════════════════════════════════════════════════

🎯 KRYTERIA WYBORU (w kolejności ważności):

1. RELEVANTNOŚĆ (najważniejsze!)
   - Publikacja MUSI bezpośrednio dotyczyć tematu artykułu
   - Odrzuć jeśli temat jest tylko wspomniany przy okazji

2. HIERARCHIA DOWODÓW (Evidence-Based Medicine):
   ⭐⭐⭐⭐⭐ Meta-analizy, Systematic Reviews, Guidelines
   ⭐⭐⭐⭐ RCT (Randomized Controlled Trials)
   ⭐⭐⭐ Cohort studies, Reviews
   ⭐⭐ Case series
   ⭐ Case reports, Expert opinion
   
3. AKTUALNOŚĆ:
   - Preferuj publikacje z ostatnich 5 lat
   - Starsze OK tylko jeśli to "klasyka" lub brak nowszych

4. WIARYGODNOŚĆ:
   - Preferuj renomowane czasopisma
   - Unikaj predatory journals
   - Sprawdź czy autor ma afiliację akademicką/kliniczną

5. ZGODNOŚĆ Z KONSENSUSEM:
   - Unikaj kontrowersyjnych/obalonych tez
   - Preferuj badania potwierdzające ustalony konsensus

⛔ NIE WYBIERAJ jeśli:
- Publikacja tylko WSPOMINA temat
- Badanie na zwierzętach (chyba że temat tego wymaga)
- Case report gdy są lepsze dowody
- Artykuł z podejrzanego źródła
- Przestarzałe wytyczne (>10 lat)

════════════════════════════════════════════════════════════════
📝 ODPOWIEDZ W FORMACIE JSON:
════════════════════════════════════════════════════════════════

{{
    "selected": [
        {{
            "index": 1,
            "evidence_level": 1-5,
            "evidence_type": "np. RCT, Meta-analysis, Review",
            "relevance_score": 0-100,
            "reason": "Dlaczego wybrano (max 25 słów)"
        }}
    ],
    "rejected": [
        {{
            "index": 2,
            "reason": "Dlaczego odrzucono (max 15 słów)"
        }}
    ],
    "overall_quality": "Krótka ocena jakości dostępnych źródeł (max 20 słów)"
}}

WAŻNE:
- Wybierz DOKŁADNIE {max_to_select} publikacje (lub mniej jeśli nie ma dobrych)
- evidence_level: 1 = najwyższy (meta-analiza), 5 = najniższy (opinia)
- relevance_score: 100 = idealne dopasowanie, 0 = brak związku
- Jeśli ŻADNA publikacja nie pasuje → pusta lista "selected"

Odpowiedz TYLKO JSON, bez dodatkowego tekstu."""


def _parse_claude_response(
    response_text: str,
    original_publications: List[Dict]
) -> Dict[str, Any]:
    """Parsuje odpowiedź Claude i mapuje na oryginalne publikacje."""
    
    # Wyczyść z markdown
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[CLAUDE_MEDICAL] ⚠️ JSON parse error: {e}")
        # Fallback
        return {
            "selected": original_publications[:CONFIG.MAX_TO_SELECT],
            "reasoning": "JSON parse failed, using fallback"
        }
    
    # Mapuj wybrane na oryginalne publikacje
    selected = []
    for item in data.get("selected", []):
        idx = item.get("index", 0) - 1  # Claude zwraca 1-indexed
        
        if 0 <= idx < len(original_publications):
            # Sprawdź minimalny score
            relevance = item.get("relevance_score", 0)
            if relevance < CONFIG.MIN_RELEVANCE_SCORE:
                print(f"[CLAUDE_MEDICAL] ⚠️ Skipping [{idx+1}] - low relevance ({relevance})")
                continue
            
            pub = original_publications[idx].copy()
            
            # Dodaj metadane z Claude
            pub["evidence_level"] = item.get("evidence_level", 3)
            pub["evidence_type"] = item.get("evidence_type", "")
            pub["relevance_score"] = relevance
            pub["claude_reason"] = item.get("reason", "")
            pub["verified_by_claude"] = True
            
            selected.append(pub)
    
    # Zbierz powody odrzucenia
    rejected_reasons = []
    for item in data.get("rejected", [])[:5]:
        rejected_reasons.append({
            "index": item.get("index"),
            "reason": item.get("reason", "")
        })
    
    return {
        "selected": selected,
        "rejected_reasons": rejected_reasons,
        "reasoning": data.get("overall_quality", "")
    }


# ============================================================================
# FALLBACK (bez Claude)
# ============================================================================

def _fallback_selection(
    publications: List[Dict],
    max_to_select: int
) -> Dict[str, Any]:
    """
    Prosty scoring jako fallback gdy Claude niedostępny.
    Używa heurystyk opartych na typie publikacji i roku.
    """
    print("[CLAUDE_MEDICAL] ⚠️ Using fallback selection (no Claude)")
    
    if not publications:
        return {
            "status": "NO_PUBLICATIONS",
            "selected": [],
            "method": "fallback"
        }
    
    scored = []
    
    for pub in publications:
        score = 50  # Bazowy
        
        # Scoring na podstawie typu publikacji
        pub_types = [t.lower() for t in pub.get("publication_types", [])]
        
        best_level = 5
        for pt in pub_types:
            for keyword, level in CONFIG.EVIDENCE_LEVELS.items():
                if keyword in pt:
                    best_level = min(best_level, level)
        
        # Wyższy poziom = wyższy score (level 1 = +50, level 5 = +10)
        score += (6 - best_level) * 10
        
        # Bonus za aktualność
        try:
            year = int(pub.get("year", "2000"))
            if year >= 2023:
                score += 20
            elif year >= 2020:
                score += 15
            elif year >= 2015:
                score += 10
        except:
            pass
        
        # Bonus za obecność abstraktu
        if pub.get("abstract") or pub.get("brief_summary"):
            score += 10
        
        # Bonus za DOI
        if pub.get("doi"):
            score += 5
        
        pub_copy = pub.copy()
        pub_copy["relevance_score"] = min(100, score)
        pub_copy["evidence_level"] = best_level
        pub_copy["verified_by_claude"] = False
        
        scored.append(pub_copy)
    
    # Sortuj po score
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "status": "OK",
        "selected": scored[:max_to_select],
        "reasoning": "Automatic selection based on publication type and recency",
        "method": "fallback"
    }


# ============================================================================
# DODATKOWE FUNKCJE
# ============================================================================

def get_evidence_level(publication: Dict) -> int:
    """
    Określa poziom dowodów dla publikacji.
    
    Returns:
        1-5 (1 = najwyższy)
    """
    pub_types = [t.lower() for t in publication.get("publication_types", [])]
    
    best_level = 5
    for pt in pub_types:
        for keyword, level in CONFIG.EVIDENCE_LEVELS.items():
            if keyword in pt:
                best_level = min(best_level, level)
    
    return best_level


def get_evidence_label(level: int) -> str:
    """Zwraca etykietę dla poziomu dowodów."""
    labels = {
        1: "Bardzo wysoki (Meta-analiza/Systematic Review/Guidelines)",
        2: "Wysoki (RCT)",
        3: "Średni (Cohort/Review)",
        4: "Niski (Case series)",
        5: "Bardzo niski (Case report/Opinion)"
    }
    return labels.get(level, "Nieznany")


# ============================================================================
# EXPORT
# ============================================================================

CLAUDE_MEDICAL_VERIFIER_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))

__all__ = [
    "verify_publications_with_claude",
    "get_evidence_level",
    "get_evidence_label",
    "CONFIG",
    "CLAUDE_MEDICAL_VERIFIER_AVAILABLE"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 CLAUDE MEDICAL VERIFIER v1.0 TEST")
    print("=" * 60)
    
    print(f"\nClaude available: {'✅' if CLAUDE_MEDICAL_VERIFIER_AVAILABLE else '❌'}")
    
    # Test z przykładowymi publikacjami
    test_publications = [
        {
            "title": "Metformin versus placebo in type 2 diabetes: A systematic review",
            "authors_short": "Smith et al.",
            "journal": "Lancet Diabetes Endocrinol",
            "year": "2023",
            "publication_types": ["Systematic Review", "Meta-Analysis"],
            "abstract": "Background: Metformin is first-line treatment for type 2 diabetes..."
        },
        {
            "title": "Case report: Unusual presentation of diabetes",
            "authors_short": "Jones",
            "journal": "J Case Rep",
            "year": "2022",
            "publication_types": ["Case Report"],
            "abstract": "We present a case of a 45-year-old male..."
        },
        {
            "title": "Diabetes management guidelines 2023",
            "authors_short": "ADA Committee",
            "journal": "Diabetes Care",
            "year": "2023",
            "publication_types": ["Practice Guideline"],
            "abstract": "These guidelines provide evidence-based recommendations..."
        }
    ]
    
    print("\n📋 Test verification...")
    result = verify_publications_with_claude(
        article_topic="leczenie cukrzycy typu 2 metforminą",
        publications=test_publications,
        max_to_select=2
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Method: {result.get('method', 'claude')}")
    print(f"Selected: {len(result.get('selected', []))}")
    
    for pub in result.get("selected", []):
        print(f"\n✅ {pub['title'][:50]}...")
        print(f"   Evidence Level: {pub.get('evidence_level', '?')} ({get_evidence_label(pub.get('evidence_level', 5))})")
        print(f"   Relevance: {pub.get('relevance_score', '?')}/100")
        print(f"   Reason: {pub.get('claude_reason', 'N/A')}")
