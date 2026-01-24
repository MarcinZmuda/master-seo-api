"""
===============================================================================
⚖️ LEGAL MODULE v3.0 - BRAJEN SEO Engine
===============================================================================

Moduł wykrywania i integracji treści prawnych (YMYL) z 3 źródłami orzeczeń:
1. SAOS API (System Analizy Orzeczeń Sądowych)
2. Google Fallback (orzeczenia.*.gov.pl, sn.pl, nsa.gov.pl)
3. Claude Scoring (weryfikacja relevantności)

Funkcje eksportowane:
- detect_category: Wykrywa kategorię treści (prawo, finanse, zdrowie)
- get_legal_context_for_article: Główna funkcja - pobiera orzeczenia
- validate_article_citations: Waliduje cytaty w tekście
- score_judgment: Scoring orzeczenia (relevantność)
- SAOS_AVAILABLE: Czy SAOS API dostępne
- LEGAL_DISCLAIMER: Tekst disclaimera
- CONFIG: Konfiguracja modułu

Autor: BRAJEN SEO Engine v36.4
===============================================================================
"""

import re
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class LegalConfig:
    """Konfiguracja modułu prawnego."""
    
    # Limity
    MAX_CITATIONS_PER_ARTICLE: int = 2  # Max orzeczeń do cytowania
    MIN_SCORE_TO_USE: int = 40  # Minimalny score żeby użyć orzeczenia
    
    # SAOS
    SAOS_ENABLED: bool = True
    SAOS_TIMEOUT: int = 15
    
    # Google Fallback
    GOOGLE_FALLBACK_ENABLED: bool = True
    
    # Scoring
    SCORING_ENABLED: bool = True
    
    # Cache
    CACHE_TTL_HOURS: int = 24


CONFIG = LegalConfig()


# ================================================================
# IMPORT ŹRÓDEŁ ORZECZEŃ
# ================================================================

# Źródło 1: SAOS API
SAOS_AVAILABLE = False
try:
    from saos_client import (
        search_judgments as saos_search,
        SAOSConfig
    )
    SAOS_AVAILABLE = True
    print("[LEGAL_MODULE] ✅ SAOS Client loaded")
except ImportError as e:
    print(f"[LEGAL_MODULE] ⚠️ SAOS Client not available: {e}")

# Źródło 2: Google Fallback
GOOGLE_FALLBACK_AVAILABLE = False
try:
    from google_judgment_fallback import (
        search_google_fallback,
        GoogleFallbackConfig
    )
    GOOGLE_FALLBACK_AVAILABLE = True
    print("[LEGAL_MODULE] ✅ Google Fallback loaded")
except ImportError as e:
    print(f"[LEGAL_MODULE] ⚠️ Google Fallback not available: {e}")

# Źródło 3: Claude Scoring
SCORING_AVAILABLE = False
try:
    from claude_judgment_verifier import (
        simple_scoring_fallback as score_judgment_relevance,
        verify_judgments_with_claude
    )
    SCORING_AVAILABLE = True
    print("[LEGAL_MODULE] ✅ Claude Scoring loaded")
except ImportError as e:
    print(f"[LEGAL_MODULE] ⚠️ Claude Scoring not available: {e}")


# ================================================================
# LEGAL DISCLAIMER
# ================================================================

LEGAL_DISCLAIMER = """
ZASTRZEŻENIE PRAWNE: Niniejszy artykuł ma charakter wyłącznie informacyjny 
i nie stanowi porady prawnej. W przypadku wątpliwości zalecamy konsultację 
z wykwalifikowanym prawnikiem lub radcą prawnym.
""".strip()


# ================================================================
# CATEGORY DETECTION
# ================================================================

LEGAL_KEYWORDS = [
    # Instytucje prawne
    "ubezwłasnowolnienie", "ubezwłasnowolnić", "ubezwłasnowolnienia",
    "ubezwłasnowolniony", "ubezwłasnowolniona",
    "testament", "spadek", "dziedziczenie", "zachowek",
    "rozwód", "alimenty", "separacja", "władza rodzicielska",
    "umowa", "kontrakt", "zobowiązanie", "wierzytelność",
    "odszkodowanie", "zadośćuczynienie", "szkoda",
    "kredyt", "hipoteka", "zastaw", "poręczenie",
    "spółka", "firma", "działalność gospodarcza",
    "wykroczenie", "przestępstwo", "kara", "wyrok",
    
    # Procedury
    "postępowanie sądowe", "pozew", "apelacja", "kasacja",
    "wniosek", "skarga", "odwołanie", "zażalenie",
    
    # Podmioty prawne
    "sąd", "prokurator", "adwokat", "radca prawny",
    "notariusz", "komornik", "kurator", "biegły",
    "opiekun prawny", "przedstawiciel ustawowy",
    "osoba chora psychicznie", "choroba psychiczna",
    "zaburzenia psychiczne", "niedorozwój umysłowy",
    
    # Akty prawne
    "kodeks cywilny", "kodeks karny", "kodeks pracy",
    "ustawa", "rozporządzenie", "artykuł ustawy",
    "przepis prawny", "regulacja", "norma prawna",
    "k.c.", "k.p.c.", "k.r.o.", "k.k.",
    
    # Pojęcia prawne
    "zdolność prawna", "zdolność do czynności prawnych",
    "przedawnienie", "zasiedzenie", "służebność",
    "czynności prawne", "osoba prawna", "osoba fizyczna"
]

FINANCE_KEYWORDS = [
    "inwestycja", "inwestować", "portfel", "akcje", "obligacje",
    "giełda", "trading", "emerytura", "podatek", "pit", "vat",
    "księgowość", "bilans", "bank", "ubezpieczenie", "polisa"
]

HEALTH_KEYWORDS = [
    "choroba", "leczenie", "terapia", "lekarz", "szpital",
    "lek", "dawkowanie", "recepta", "diagnoza", "objawy",
    "operacja", "rehabilitacja", "psycholog", "psychiatra"
]


def detect_category(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Wykrywa kategorię treści na podstawie słów kluczowych.
    
    Args:
        main_keyword: Główna fraza kluczowa
        additional_keywords: Dodatkowe frazy
        
    Returns:
        {
            "category": "prawo" | "finanse" | "zdrowie" | "general",
            "confidence": float (0.0-1.0),
            "is_ymyl": bool,
            "detected_keywords": List[str],
            "legal_enabled": bool
        }
    """
    additional_keywords = additional_keywords or []
    all_keywords = [main_keyword] + additional_keywords
    combined_text = " ".join(all_keywords).lower()
    
    # Zlicz dopasowania
    legal_matches = [kw for kw in LEGAL_KEYWORDS if kw.lower() in combined_text]
    finance_matches = [kw for kw in FINANCE_KEYWORDS if kw.lower() in combined_text]
    health_matches = [kw for kw in HEALTH_KEYWORDS if kw.lower() in combined_text]
    
    scores = {
        "prawo": len(legal_matches),
        "finanse": len(finance_matches),
        "zdrowie": len(health_matches)
    }
    
    max_category = max(scores, key=scores.get)
    max_score = scores[max_category]
    
    if max_score == 0:
        return {
            "category": "general",
            "confidence": 0.0,
            "is_ymyl": False,
            "detected_keywords": [],
            "legal_enabled": False,
            "sources_available": {
                "saos": SAOS_AVAILABLE,
                "google": GOOGLE_FALLBACK_AVAILABLE,
                "scoring": SCORING_AVAILABLE
            }
        }
    
    confidence = min(1.0, max_score / 3)  # 3+ dopasowań = 100%
    category = max_category if max_score >= 1 else "general"  # 1+ wystarcza
    detected = {
        "prawo": legal_matches,
        "finanse": finance_matches,
        "zdrowie": health_matches
    }.get(category, [])
    
    is_ymyl = category in ["prawo", "finanse", "zdrowie"] and confidence >= 0.4
    legal_enabled = category == "prawo" and SAOS_AVAILABLE
    
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "is_ymyl": is_ymyl,
        "detected_keywords": detected[:10],
        "legal_enabled": legal_enabled,
        "sources_available": {
            "saos": SAOS_AVAILABLE,
            "google": GOOGLE_FALLBACK_AVAILABLE,
            "scoring": SCORING_AVAILABLE
        }
    }


# ================================================================
# GŁÓWNA FUNKCJA - POBIERANIE KONTEKSTU PRAWNEGO
# ================================================================

def get_legal_context_for_article(
    main_keyword: str,
    additional_keywords: List[str] = None,
    force_enable: bool = False,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Główna funkcja - pobiera kontekst prawny z 3 źródeł.
    
    Args:
        main_keyword: Główna fraza kluczowa
        additional_keywords: Dodatkowe frazy
        force_enable: Wymuś włączenie nawet dla nie-prawnych tematów
        max_results: Maksymalna liczba orzeczeń
        
    Returns:
        {
            "status": "OK" | "NOT_LEGAL" | "NO_RESULTS",
            "category": "prawo",
            "judgments": [...],
            "total_found": int,
            "sources_used": ["saos", "google"],
            "disclaimer": str,
            "instruction": str
        }
    """
    additional_keywords = additional_keywords or []
    
    # 1. Wykryj kategorię
    detection = detect_category(main_keyword, additional_keywords)
    
    if not force_enable and detection["category"] != "prawo":
        return {
            "status": "NOT_LEGAL",
            "category": detection["category"],
            "reason": f"Temat '{main_keyword}' nie jest kategorią prawną",
            "judgments": [],
            "total_found": 0,
            "sources_used": [],
            "disclaimer": "",
            "instruction": ""
        }
    
    print(f"[LEGAL_MODULE] 🔍 Szukam orzeczeń dla: '{main_keyword}'")
    
    judgments = []
    sources_used = []
    
    # 2. Źródło 1: SAOS API
    if SAOS_AVAILABLE:
        try:
            print(f"[LEGAL_MODULE] 📡 Zapytanie do SAOS...")
            saos_result = saos_search(
                query=main_keyword,
                max_results=max_results * 2  # Pobierz więcej do scoringu
            )
            
            if saos_result and saos_result.get("judgments"):
                judgments.extend(saos_result["judgments"])
                sources_used.append("saos")
                print(f"[LEGAL_MODULE] ✅ SAOS: {len(saos_result['judgments'])} wyników")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ SAOS error: {e}")
    
    # 3. Źródło 2: Google Fallback (jeśli SAOS nie zwrócił wyników)
    if not judgments and GOOGLE_FALLBACK_AVAILABLE:
        try:
            print(f"[LEGAL_MODULE] 📡 Fallback do Google...")
            google_result = search_google_fallback(
                articles=[],  # Szukaj po słowie kluczowym
                keyword=main_keyword,
                max_results=max_results
            )
            
            if google_result and google_result.get("judgments"):
                judgments.extend(google_result["judgments"])
                sources_used.append("google")
                print(f"[LEGAL_MODULE] ✅ Google: {len(google_result['judgments'])} wyników")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ Google error: {e}")
    
    # 4. Scoring i filtrowanie
    if judgments and SCORING_AVAILABLE:
        try:
            scored_judgments = []
            for j in judgments:
                score_result = score_judgment(
                    text=j.get("excerpt", j.get("text", "")),
                    keyword=main_keyword
                )
                j["relevance_score"] = score_result.get("score", 50)
                scored_judgments.append(j)
            
            # Sortuj po score i filtruj
            scored_judgments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            judgments = [j for j in scored_judgments if j.get("relevance_score", 0) >= CONFIG.MIN_SCORE_TO_USE]
            sources_used.append("scoring")
            print(f"[LEGAL_MODULE] ✅ Scoring: {len(judgments)} po filtracji")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ Scoring error: {e}")
    
    # 5. Ogranicz do max_results
    judgments = judgments[:max_results]
    
    # 6. Brak wyników
    if not judgments:
        return {
            "status": "NO_RESULTS",
            "category": "prawo",
            "reason": f"Nie znaleziono orzeczeń dla '{main_keyword}'",
            "judgments": [],
            "total_found": 0,
            "sources_used": sources_used,
            "disclaimer": LEGAL_DISCLAIMER,
            "instruction": "Brak orzeczeń do cytowania. Artykuł może być bez sygnatur."
        }
    
    # 7. Buduj instrukcję
    instruction = _build_citation_instruction(main_keyword, judgments)
    
    return {
        "status": "OK",
        "category": "prawo",
        "judgments": judgments,
        "total_found": len(judgments),
        "sources_used": sources_used,
        "disclaimer": LEGAL_DISCLAIMER,
        "instruction": instruction
    }


def _build_citation_instruction(keyword: str, judgments: List[Dict]) -> str:
    """Buduje instrukcję cytowania dla GPT."""
    
    lines = [
        f"ORZECZENIA DLA TEMATU: {keyword}",
        f"Znaleziono: {len(judgments)} orzeczeń",
        "",
        "ZASADY CYTOWANIA:",
        f"- Użyj MAKSYMALNIE {CONFIG.MAX_CITATIONS_PER_ARTICLE} orzeczeń",
        "- Cytuj sygnaturę, datę i sąd",
        "- Nie wymyślaj sygnatur!",
        "",
        "DOSTĘPNE ORZECZENIA:"
    ]
    
    for i, j in enumerate(judgments[:CONFIG.MAX_CITATIONS_PER_ARTICLE], 1):
        sig = j.get("signature", "brak")
        date = j.get("formatted_date", j.get("date", ""))
        court = j.get("court", "")
        score = j.get("relevance_score", "?")
        
        lines.append(f"{i}. {sig} z dnia {date}")
        lines.append(f"   Sąd: {court}")
        lines.append(f"   Relevantność: {score}/100")
        
        excerpt = j.get("excerpt", "")
        if excerpt:
            lines.append(f"   Fragment: \"{excerpt[:150]}...\"")
        lines.append("")
    
    return "\n".join(lines)


# ================================================================
# SCORING ORZECZENIA
# ================================================================

def score_judgment(text: str, keyword: str) -> Dict[str, Any]:
    """
    Ocenia relevantność orzeczenia dla danego słowa kluczowego.
    
    Args:
        text: Treść orzeczenia (fragment)
        keyword: Słowo kluczowe tematu
        
    Returns:
        {
            "score": int (0-100),
            "factors": {...},
            "recommendation": str
        }
    """
    if not text:
        return {"score": 0, "factors": {}, "recommendation": "Brak tekstu"}
    
    # Użyj Claude jeśli dostępny
    if SCORING_AVAILABLE:
        try:
            return score_judgment_relevance(text, keyword)
        except:
            pass
    
    # Fallback: prosty scoring
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    score = 0
    factors = {}
    
    # Czy keyword występuje?
    keyword_count = text_lower.count(keyword_lower)
    if keyword_count > 0:
        score += min(30, keyword_count * 10)
        factors["keyword_mentions"] = keyword_count
    
    # Czy są artykuły prawne?
    articles = re.findall(r'art\.\s*\d+', text_lower)
    if articles:
        score += min(20, len(articles) * 5)
        factors["legal_articles"] = len(articles)
    
    # Czy są sygnatury?
    signatures = re.findall(r'[IVX]+\s+[A-Za-z]+\s+\d+/\d+', text)
    if signatures:
        score += 20
        factors["signatures"] = len(signatures)
    
    # Długość tekstu
    if len(text) > 500:
        score += 10
        factors["sufficient_length"] = True
    
    # Normalizuj do 0-100
    score = min(100, max(0, score))
    
    recommendation = "OK" if score >= CONFIG.MIN_SCORE_TO_USE else "Za niska relevantność"
    
    return {
        "score": score,
        "factors": factors,
        "recommendation": recommendation
    }


# ================================================================
# WALIDACJA CYTATÓW
# ================================================================

def validate_article_citations(full_text: str) -> Dict[str, Any]:
    """
    Waliduje cytaty prawne w tekście artykułu.
    
    Args:
        full_text: Pełny tekst artykułu
        
    Returns:
        {
            "valid": bool,
            "signatures_found": List[str],
            "signatures_count": int,
            "has_disclaimer": bool,
            "warnings": List[str],
            "suggestions": List[str]
        }
    """
    warnings = []
    suggestions = []
    
    # Znajdź sygnatury
    signature_pattern = r'([IVX]+)\s+([A-Za-z]{1,4})\s+(\d+)/(\d{2,4})'
    signatures = re.findall(signature_pattern, full_text)
    signatures_formatted = [f"{m[0]} {m[1]} {m[2]}/{m[3]}" for m in signatures]
    
    # Sprawdź liczbę sygnatur
    if len(signatures) > CONFIG.MAX_CITATIONS_PER_ARTICLE:
        warnings.append(f"Za dużo sygnatur ({len(signatures)} > {CONFIG.MAX_CITATIONS_PER_ARTICLE})")
        suggestions.append(f"Ogranicz cytaty do {CONFIG.MAX_CITATIONS_PER_ARTICLE} najważniejszych")
    
    if len(signatures) == 0:
        suggestions.append("Rozważ dodanie 1-2 orzeczeń sądowych")
    
    # Sprawdź disclaimer
    disclaimer_keywords = ["zastrzeżenie", "porada prawna", "konsultacja z prawnikiem"]
    has_disclaimer = any(kw in full_text.lower() for kw in disclaimer_keywords)
    
    if not has_disclaimer:
        warnings.append("Brak zastrzeżenia prawnego")
        suggestions.append("Dodaj disclaimer na końcu artykułu")
    
    # Sprawdź artykuły prawne
    articles = re.findall(r'art\.\s*\d+', full_text.lower())
    
    valid = len(warnings) == 0
    
    return {
        "valid": valid,
        "signatures_found": signatures_formatted,
        "signatures_count": len(signatures),
        "articles_mentioned": len(articles),
        "has_disclaimer": has_disclaimer,
        "warnings": warnings,
        "suggestions": suggestions
    }


# ================================================================
# HELPER: KONTEKST DLA PROMPTU GPT
# ================================================================

def get_legal_context_for_prompt(topic: str) -> str:
    """
    Generuje sekcję promptu z kontekstem prawnym dla GPT.
    """
    detection = detect_category(topic)
    
    if detection["category"] != "prawo":
        return ""
    
    lines = [
        "",
        "=" * 60,
        "⚖️ KONTEKST PRAWNY (YMYL)",
        "=" * 60,
        "",
        "Ten artykuł dotyczy tematyki PRAWNEJ. Przestrzegaj zasad:",
        "",
        "1. TERMINOLOGIA:",
        "   - Używaj precyzyjnych terminów prawnych",
        "   - Odwołuj się do aktów prawnych (KC, KPC, KRO)",
        "",
        "2. ORZECZENIA:",
        f"   - Używaj TYLKO sygnatur z get_judgments",
        f"   - Max {CONFIG.MAX_CITATIONS_PER_ARTICLE} cytaty",
        "   - NIE wymyślaj sygnatur!",
        "",
        "3. DISCLAIMER:",
        "   - Dodaj zastrzeżenie prawne (raz, w outro)",
        "",
    ]
    
    return "\n".join(lines)


# ================================================================
# EXPORT
# ================================================================

__all__ = [
    "detect_category",
    "get_legal_context_for_article",
    "validate_article_citations",
    "score_judgment",
    "get_legal_context_for_prompt",
    "SAOS_AVAILABLE",
    "GOOGLE_FALLBACK_AVAILABLE",
    "SCORING_AVAILABLE",
    "LEGAL_DISCLAIMER",
    "CONFIG"
]


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("⚖️ LEGAL MODULE v3.0 TEST")
    print("=" * 60)
    
    print(f"\nŹródła:")
    print(f"  SAOS:    {'✅' if SAOS_AVAILABLE else '❌'}")
    print(f"  Google:  {'✅' if GOOGLE_FALLBACK_AVAILABLE else '❌'}")
    print(f"  Scoring: {'✅' if SCORING_AVAILABLE else '❌'}")
    
    # Test detekcji
    test_topics = [
        "Ubezwłasnowolnienie osoby chorej psychicznie",
        "Przepis na ciasto"
    ]
    
    print(f"\nDetekcja kategorii:")
    for topic in test_topics:
        result = detect_category(topic)
        print(f"  '{topic[:40]}...' → {result['category']} ({result['confidence']*100:.0f}%)")
    
    # Test pobierania kontekstu
    print(f"\nTest get_legal_context_for_article:")
    result = get_legal_context_for_article("ubezwłasnowolnienie", max_results=2)
    print(f"  Status: {result['status']}")
    print(f"  Znaleziono: {result['total_found']} orzeczeń")
    print(f"  Źródła: {result['sources_used']}")
