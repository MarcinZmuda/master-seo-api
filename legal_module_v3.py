"""
===============================================================================
⚖️ LEGAL MODULE v3.1 - BRAJEN SEO Engine
===============================================================================

Moduł wykrywania i integracji treści prawnych (YMYL) z 4 źródłami orzeczeń:
1. SAOS API (System Analizy Orzeczeń Sądowych)
2. Google Fallback (orzeczenia.*.gov.pl, sn.pl, nsa.gov.pl)
3. 🆕 Local Court Scraper (10 lokalnych portali sądów)
4. Claude Scoring (weryfikacja relevantności)

🆕 Zmiany w v3.1:
- Dodano local_court_scraper jako 3. źródło (fallback gdy SAOS < 2 wyniki)
- Deduplikacja po sygnaturze
- Graceful degradation (błąd scrapera = kontynuuj bez)
- source field w każdym orzeczeniu ("saos", "google", "local")

Funkcje eksportowane:
- detect_category: Wykrywa kategorię treści (prawo, finanse, zdrowie)
- get_legal_context_for_article: Główna funkcja - pobiera orzeczenia
- validate_article_citations: Waliduje cytaty w tekście
- score_judgment: Scoring orzeczenia (relevantność)
- SAOS_AVAILABLE: Czy SAOS API dostępne
- LEGAL_DISCLAIMER: Tekst disclaimera
- CONFIG: Konfiguracja modułu

Autor: BRAJEN SEO Engine v42.2
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
    MAX_CITATIONS_PER_ARTICLE: int = 4
    MIN_SCORE_TO_USE: int = 40
    
    # SAOS
    SAOS_ENABLED: bool = True
    SAOS_TIMEOUT: int = 15
    
    # Google Fallback
    GOOGLE_FALLBACK_ENABLED: bool = True
    
    # 🆕 v3.1: Local Court Scraper
    LOCAL_SCRAPER_ENABLED: bool = True
    LOCAL_SCRAPER_TIMEOUT: int = 10  # Krótszy timeout dla fallbacku
    LOCAL_SCRAPER_MIN_SAOS_RESULTS: int = 2  # Użyj local tylko gdy SAOS < 2
    
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

# 🆕 Źródło 3: Local Court Scraper (v3.1)
LOCAL_SCRAPER_AVAILABLE = False
try:
    from local_court_scraper import (
        search_local_courts,
        get_local_scraper,
        LocalCourtScraper
    )
    LOCAL_SCRAPER_AVAILABLE = True
    print("[LEGAL_MODULE] ✅ Local Court Scraper loaded (10 portals)")
except ImportError as e:
    print(f"[LEGAL_MODULE] ⚠️ Local Court Scraper not available: {e}")

# Źródło 4: Claude Scoring
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
    "inwestycj", "inwestowa", "portfel", "akcj", "obligacj",
    "giełd", "trading", "emerytur", "podatk", "pit", "vat",
    "księgowoś", "bilans", "bank", "ubezpiecz", "polis"
]

HEALTH_KEYWORDS = [
    "chorob", "leczeni", "terapi", "lekarz", "szpital",
    "dawkowani", "recept", "diagnoz", "objaw",
    "operacj", "rehabilitacj", "psycholog", "psychiatr",
    "cukrzyc", "nadciśnieni", "nowotwór", "szczepion",
    "antybiotyk", "insulin", "depresj", "astm",
    "zawał", "serc", "udar", "migren", "alergi",
    "zapaleni", "infekcj", "gryp", "covid",
    "rak piersi", "rak płuc", "rak jelita", "onkolog",
]


def detect_category(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Wykrywa kategorię treści na podstawie słów kluczowych.
    🆕 v44.6: Wagi dla terminów + lemmatyzacja spaCy.
    """
    additional_keywords = additional_keywords or []
    all_keywords = [main_keyword] + additional_keywords
    combined_text = " ".join(all_keywords).lower()
    
    # 🆕 Lemmatyzacja (jeśli spaCy dostępne)
    lemmatized_text = combined_text
    try:
        from shared_nlp import get_nlp
        nlp = get_nlp()
        if nlp:
            doc = nlp(combined_text)
            lemmatized_text = " ".join([t.lemma_.lower() for t in doc])
    except Exception:
        pass
    
    search_text = combined_text + " " + lemmatized_text
    
    # 🆕 Ważone keyword matching
    LEGAL_WEIGHTED = {
        # Silne sygnały (waga 3) — jednoznacznie prawne
        "ubezwłasnowolnienie": 3, "kodeks cywilny": 3, "kodeks karny": 3,
        "kodeks pracy": 3, "k.c.": 3, "k.p.c.": 3, "k.r.o.": 3, "k.k.": 3,
        "zdolność do czynności prawnych": 3, "postępowanie sądowe": 3,
        "zachowek": 3, "przedawnienie roszczenia": 3, "zasiedzenie": 3,
        "władza rodzicielska": 3, "opiekun prawny": 3, "kurator sądowy": 3,
        "radca prawny": 3, "komornik": 3, "notariusz": 3,
        # Średnie sygnały (waga 2) — prawne, mało dwuznaczne
        "testament": 2, "dziedziczenie": 2,
        "rozwód": 2, "alimenty": 2, "separacja": 2,
        "odszkodowanie": 2, "zadośćuczynienie": 2,
        "pozew": 2, "apelacja": 2, "kasacja": 2, "zażalenie": 2,
        "wyrok sądu": 2, "orzeczenie sądu": 2, "sąd okręgowy": 2,
        "przepis prawny": 2, "norma prawna": 2,
        "hipoteka": 2, "zastaw": 2, "poręczenie": 2,
        "przestępstwo": 2, "wykroczenie": 2,
        "adwokat": 2, "ustawa": 2, "rozporządzenie": 2,
        # Słabe sygnały (waga 1) — mogą być w innym kontekście
        "umowa": 1, "kontrakt": 1, "kara": 1,
        "skarga": 1, "odwołanie": 1,
        "firma": 1, "spółka": 1, "kredyt": 1,
        "spadek": 1, "sąd": 1, "wyrok": 1, "orzeczenie": 1,
    }
    
    legal_score = 0
    legal_matches = []
    for kw, weight in LEGAL_WEIGHTED.items():
        if kw in search_text:
            legal_score += weight
            legal_matches.append(kw)
    
    finance_matches = [kw for kw in FINANCE_KEYWORDS if kw.lower() in search_text]
    health_matches = [kw for kw in HEALTH_KEYWORDS if kw.lower() in search_text]
    
    scores = {
        "prawo": legal_score,
        "finanse": len(finance_matches) * 2,
        "zdrowie": len(health_matches) * 2
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
                "local": LOCAL_SCRAPER_AVAILABLE,
                "scoring": SCORING_AVAILABLE
            }
        }
    
    # Confidence: score 2 = 0.4, score 3 = 0.6, score 5+ = 1.0
    confidence = min(1.0, max_score / 5)
    # Prawo wymaga silniejszego sygnału (score >= 2), zdrowie/finanse mogą na 1 match (score >= 2)
    category = max_category if max_score >= 2 else "general"
    detected = {
        "prawo": legal_matches,
        "finanse": finance_matches,
        "zdrowie": health_matches
    }.get(category, [])
    
    is_ymyl = category in ["prawo", "finanse", "zdrowie"] and confidence >= 0.3
    legal_enabled = category == "prawo" and SAOS_AVAILABLE
    
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "is_ymyl": is_ymyl,
        "detected_keywords": detected[:10],
        "legal_enabled": legal_enabled,
        "weighted_score": round(max_score, 1),
        "sources_available": {
            "saos": SAOS_AVAILABLE,
            "google": GOOGLE_FALLBACK_AVAILABLE,
            "local": LOCAL_SCRAPER_AVAILABLE,
            "scoring": SCORING_AVAILABLE
        }
    }


# ================================================================
# 🆕 v3.1: DEDUPLIKACJA ORZECZEŃ
# ================================================================

def deduplicate_judgments(judgments: List[Dict]) -> List[Dict]:
    """
    Deduplikuje orzeczenia po sygnaturze.
    Preferuje SAOS > Google > Local.
    """
    seen_signatures = set()
    unique = []
    
    # Sortuj by preferować SAOS
    source_priority = {"saos": 0, "google": 1, "local": 2}
    sorted_judgments = sorted(
        judgments, 
        key=lambda j: source_priority.get(j.get("source", "local"), 99)
    )
    
    for j in sorted_judgments:
        sig = j.get("signature", "").strip()
        if not sig:
            # Bez sygnatury - dodaj (ale może być duplikat)
            unique.append(j)
            continue
        
        # Normalizuj sygnaturę (usuń spacje, małe litery)
        sig_normalized = re.sub(r'\s+', '', sig.upper())
        
        if sig_normalized not in seen_signatures:
            seen_signatures.add(sig_normalized)
            unique.append(j)
        else:
            print(f"[LEGAL_MODULE] ⏭️ Deduplicated: {sig} (already have from better source)")
    
    return unique


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
    Główna funkcja - pobiera kontekst prawny z 4 źródeł.
    
    🆕 v3.1: Dodano local_court_scraper jako 3. źródło (fallback)
    
    Kolejność:
    1. SAOS API (główne źródło)
    2. Google Fallback (gdy SAOS brak)
    3. Local Court Scraper (gdy SAOS < 2 wyniki)
    4. Scoring i filtrowanie
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
    
    # 🆕 v44.6: Cache check
    try:
        from ymyl_cache import ymyl_cache
        cache_key = f"{main_keyword}|{max_results}"
        cached = ymyl_cache.get("saos", cache_key)
        if cached:
            print(f"[LEGAL_MODULE] 📦 Cache HIT dla '{main_keyword}'")
            return cached
    except ImportError:
        ymyl_cache = None
        cache_key = None
    
    judgments = []
    sources_used = []
    
    # ═══════════════════════════════════════════════════════════════
    # 2. Źródło 1: SAOS API
    # ═══════════════════════════════════════════════════════════════
    saos_count = 0
    if SAOS_AVAILABLE and CONFIG.SAOS_ENABLED:
        try:
            print(f"[LEGAL_MODULE] 📡 Zapytanie do SAOS...")
            saos_result = saos_search(
                query=main_keyword,
                max_results=max_results * 2
            )
            
            if saos_result and saos_result.get("judgments"):
                for j in saos_result["judgments"]:
                    j["source"] = "saos"
                judgments.extend(saos_result["judgments"])
                sources_used.append("saos")
                saos_count = len(saos_result["judgments"])
                print(f"[LEGAL_MODULE] ✅ SAOS: {saos_count} wyników")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ SAOS error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. Źródło 2: Google Fallback (gdy SAOS brak)
    # ═══════════════════════════════════════════════════════════════
    if not judgments and GOOGLE_FALLBACK_AVAILABLE and CONFIG.GOOGLE_FALLBACK_ENABLED:
        try:
            print(f"[LEGAL_MODULE] 📡 Fallback do Google...")
            google_result = search_google_fallback(
                articles=[],
                keyword=main_keyword,
                max_results=max_results
            )
            
            if google_result and google_result.get("judgments"):
                for j in google_result["judgments"]:
                    j["source"] = "google"
                judgments.extend(google_result["judgments"])
                sources_used.append("google")
                print(f"[LEGAL_MODULE] ✅ Google: {len(google_result['judgments'])} wyników")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ Google error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 🆕 4. Źródło 3: Local Court Scraper (gdy SAOS < 2)
    # ═══════════════════════════════════════════════════════════════
    if (LOCAL_SCRAPER_AVAILABLE and 
        CONFIG.LOCAL_SCRAPER_ENABLED and 
        saos_count < CONFIG.LOCAL_SCRAPER_MIN_SAOS_RESULTS):
        
        try:
            print(f"[LEGAL_MODULE] 📡 Fallback do Local Courts (SAOS={saos_count} < {CONFIG.LOCAL_SCRAPER_MIN_SAOS_RESULTS})...")
            local_result = search_local_courts(
                keyword=main_keyword,
                max_results=max_results
            )
            
            if local_result and local_result.get("judgments"):
                for j in local_result["judgments"]:
                    j["source"] = "local"
                    # Dodaj official_portal jeśli brak
                    if "official_portal" not in j:
                        j["official_portal"] = j.get("source_url", "").replace("https://", "").replace("http://", "")
                
                judgments.extend(local_result["judgments"])
                sources_used.append("local")
                print(f"[LEGAL_MODULE] ✅ Local Courts: {len(local_result['judgments'])} wyników")
                
                # Log errors jeśli były
                if local_result.get("errors"):
                    for err in local_result["errors"][:3]:
                        print(f"[LEGAL_MODULE] ⚠️ Local error: {err}")
                        
        except Exception as e:
            # Graceful degradation - nie przerywaj
            print(f"[LEGAL_MODULE] ⚠️ Local Courts error (continuing): {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. Deduplikacja
    # ═══════════════════════════════════════════════════════════════
    if len(sources_used) > 1:
        before_dedup = len(judgments)
        judgments = deduplicate_judgments(judgments)
        print(f"[LEGAL_MODULE] 🔄 Deduplicated: {before_dedup} → {len(judgments)}")
    
    # ═══════════════════════════════════════════════════════════════
    # 6. Scoring i filtrowanie
    # ═══════════════════════════════════════════════════════════════
    if judgments and SCORING_AVAILABLE and CONFIG.SCORING_ENABLED:
        try:
            scored_judgments = []
            for j in judgments:
                score_result = score_judgment(
                    text=j.get("excerpt", j.get("text", "")),
                    keyword=main_keyword
                )
                j["relevance_score"] = score_result.get("score", 50)
                scored_judgments.append(j)
            
            scored_judgments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            judgments = [j for j in scored_judgments if j.get("relevance_score", 0) >= CONFIG.MIN_SCORE_TO_USE]
            sources_used.append("scoring")
            print(f"[LEGAL_MODULE] ✅ Scoring: {len(judgments)} po filtracji")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ Scoring error: {e}")
    
    # 7. Ogranicz do max_results
    judgments = judgments[:max_results]
    
    # 8. Brak wyników
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
    
    # 9. Buduj instrukcję
    instruction = _build_citation_instruction(main_keyword, judgments)
    
    result = {
        "status": "OK",
        "category": "prawo",
        "judgments": judgments,
        "total_found": len(judgments),
        "sources_used": sources_used,
        "disclaimer": LEGAL_DISCLAIMER,
        "instruction": instruction
    }
    
    # 🆕 v44.6: Cache SET
    try:
        if ymyl_cache and cache_key:
            ymyl_cache.set("saos", cache_key, result)
    except Exception:
        pass
    
    return result


def _build_citation_instruction(keyword: str, judgments: List[Dict]) -> str:
    """Buduje KOMPAKTOWĄ instrukcję cytowania (v44.6 — oszczędność tokenów)."""
    
    lines = [
        f"⚖️ ORZECZENIA: {keyword} ({len(judgments)} szt.)",
        f"Max {CONFIG.MAX_CITATIONS_PER_ARTICLE} w tekście. Format: sygnatura + data + sąd + portal.",
        ""
    ]
    
    for i, j in enumerate(judgments[:CONFIG.MAX_CITATIONS_PER_ARTICLE], 1):
        sig = j.get("signature", "brak")
        date = j.get("formatted_date", j.get("date", ""))
        court = j.get("court", "")
        source = j.get("source", "?")
        
        portal = j.get("official_portal", "")
        if portal:
            portal = portal.replace("https://", "").replace("http://", "").rstrip("/")
        else:
            portal = "orzeczenia.ms.gov.pl"
        
        lines.append(f"#{i} [{source}] {sig} | {date} | {court} | {portal}")
        
        excerpt = j.get("excerpt", "")
        if excerpt:
            lines.append(f"   → {excerpt[:180]}...")
        lines.append("")
    
    lines.append("Cytuj: \"...SĄD w wyroku z DD.MM.RRRR (sygn. XXX)... (dostępne na: portal).\"")
    lines.append("NIE wymyślaj sygnatur. NIE wklejaj URL.")
    
    return "\n".join(lines)


# ================================================================
# SCORING ORZECZENIA
# ================================================================

def score_judgment(text: str, keyword: str) -> Dict[str, Any]:
    """
    Ocenia relevantność orzeczenia dla danego słowa kluczowego.
    """
    if not text:
        return {"score": 0, "factors": {}, "recommendation": "Brak tekstu"}
    
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
    
    keyword_count = text_lower.count(keyword_lower)
    if keyword_count > 0:
        score += min(30, keyword_count * 10)
        factors["keyword_mentions"] = keyword_count
    
    articles = re.findall(r'art\.\s*\d+', text_lower)
    if articles:
        score += min(20, len(articles) * 5)
        factors["legal_articles"] = len(articles)
    
    signatures = re.findall(r'[IVX]+\s+[A-Za-z]+\s+\d+/\d+', text)
    if signatures:
        score += 20
        factors["signatures"] = len(signatures)
    
    if len(text) > 500:
        score += 10
        factors["sufficient_length"] = True
    
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

def validate_article_citations(full_text: str, provided_judgments: List[Dict] = None) -> Dict[str, Any]:
    """
    Waliduje cytaty prawne w tekście artykułu.
    🆕 v44.6: Sprawdza czy sygnatury istnieją w dostarczonych źródłach (anti-hallucination).
    """
    warnings = []
    suggestions = []
    
    signature_pattern = r'([IVX]+)\s+([A-Za-z]{1,4})\s+(\d+)/(\d{2,4})'
    signatures = re.findall(signature_pattern, full_text)
    signatures_formatted = [f"{m[0]} {m[1]} {m[2]}/{m[3]}" for m in signatures]
    
    if len(signatures) > CONFIG.MAX_CITATIONS_PER_ARTICLE:
        warnings.append(f"Za dużo sygnatur ({len(signatures)} > {CONFIG.MAX_CITATIONS_PER_ARTICLE})")
        suggestions.append(f"Ogranicz cytaty do {CONFIG.MAX_CITATIONS_PER_ARTICLE} najważniejszych")
    
    if len(signatures) == 0:
        suggestions.append("Rozważ dodanie 1-2 orzeczeń sądowych")
    
    # 🆕 v44.6: Anti-hallucination — sprawdź czy sygnatury pasują do dostarczonych
    hallucinated = []
    verified = []
    if provided_judgments and signatures_formatted:
        provided_sigs = set()
        for j in provided_judgments:
            sig = j.get("signature", "").strip()
            if sig:
                # Normalizuj (usuń spacje wewnętrzne, uppercase)
                sig_norm = re.sub(r'\s+', ' ', sig.upper().strip())
                provided_sigs.add(sig_norm)
        
        for found_sig in signatures_formatted:
            found_norm = re.sub(r'\s+', ' ', found_sig.upper().strip())
            # Szukaj częściowego dopasowania (numer sprawy + rok)
            matched = False
            for provided in provided_sigs:
                # Wystarczy match numeru i roku (np. "895/18")
                found_case = re.search(r'(\d+/\d+)', found_norm)
                prov_case = re.search(r'(\d+/\d+)', provided)
                if found_case and prov_case and found_case.group() == prov_case.group():
                    matched = True
                    break
            
            if matched:
                verified.append(found_sig)
            else:
                hallucinated.append(found_sig)
        
        if hallucinated:
            warnings.append(
                f"⚠️ HALLUCYNACJA: {len(hallucinated)} sygnatur NIE pochodzi z dostarczonych źródeł: "
                f"{', '.join(hallucinated[:3])}"
            )
            suggestions.append("Usuń wymyślone sygnatury i użyj TYLKO dostarczonych orzeczeń")
    
    disclaimer_keywords = ["zastrzeżenie", "porada prawna", "konsultacja z prawnikiem"]
    has_disclaimer = any(kw in full_text.lower() for kw in disclaimer_keywords)
    
    if not has_disclaimer:
        warnings.append("Brak zastrzeżenia prawnego")
        suggestions.append("Dodaj disclaimer na końcu artykułu")
    
    articles = re.findall(r'art\.\s*\d+', full_text.lower())
    
    valid = len(warnings) == 0
    
    return {
        "valid": valid,
        "signatures_found": signatures_formatted,
        "signatures_count": len(signatures),
        "signatures_verified": verified,
        "signatures_hallucinated": hallucinated,
        "articles_mentioned": len(articles),
        "has_disclaimer": has_disclaimer,
        "warnings": warnings,
        "suggestions": suggestions
    }


# ================================================================
# HELPER: KONTEKST DLA PROMPTU GPT
# ================================================================

def get_legal_context_for_prompt(topic: str) -> str:
    """Generuje sekcję promptu z kontekstem prawnym (v44.6 kompakt)."""
    detection = detect_category(topic)
    
    if detection["category"] != "prawo":
        return ""
    
    return (
        f"⚖️ YMYL PRAWNY: Używaj terminologii prawnej (KC, KPC, KRO). "
        f"Max {CONFIG.MAX_CITATIONS_PER_ARTICLE} sygnatur — TYLKO z dostarczonych. "
        f"NIE wymyślaj. Disclaimer w outro."
    )


# ================================================================
# EXPORT
# ================================================================

__all__ = [
    "detect_category",
    "get_legal_context_for_article",
    "validate_article_citations",
    "score_judgment",
    "get_legal_context_for_prompt",
    "deduplicate_judgments",
    "SAOS_AVAILABLE",
    "GOOGLE_FALLBACK_AVAILABLE",
    "LOCAL_SCRAPER_AVAILABLE",
    "SCORING_AVAILABLE",
    "LEGAL_DISCLAIMER",
    "CONFIG"
]


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("⚖️ LEGAL MODULE v3.1 TEST")
    print("=" * 60)
    
    print(f"\nŹródła:")
    print(f"  SAOS:    {'✅' if SAOS_AVAILABLE else '❌'}")
    print(f"  Google:  {'✅' if GOOGLE_FALLBACK_AVAILABLE else '❌'}")
    print(f"  Local:   {'✅' if LOCAL_SCRAPER_AVAILABLE else '❌'} (10 portals)")
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
    
    if result.get("judgments"):
        for j in result["judgments"][:2]:
            print(f"    📄 {j.get('signature', 'N/A')} [{j.get('source', 'N/A')}] - {j.get('official_portal', 'N/A')}")
