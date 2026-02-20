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


# ================================================================
# 🧠 v45.3: CLAUDE HAIKU SEMANTIC YMYL CLASSIFIER
# ================================================================

def _classify_ymyl_with_claude(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Klasyfikuje temat jako YMYL używając Claude Haiku.
    Rozumie kontekst semantyczny — nie wymaga listy keywords.
    
    Returns: {"category": "prawo"|"finanse"|"zdrowie"|"general",
              "confidence": 0.0-1.0, "reasoning": str, "reasoning_keywords": []}
    Or None if Claude unavailable.
    """
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[YMYL] ⚠️ ANTHROPIC_API_KEY not set — Claude classifier unavailable")
        return None
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except (ImportError, Exception) as e:
        print(f"[YMYL] ⚠️ anthropic client error: {e}")
        return None
    
    additional = ", ".join(additional_keywords) if additional_keywords else ""
    topic_str = f"{main_keyword}" + (f" (dodatkowe: {additional})" if additional else "")
    
    prompt = f"""Klasyfikuj poniższy temat artykułu SEO według kategorii Google YMYL (Your Money or Your Life).

TEMAT: "{topic_str}"

Kategorie:
- "prawo" — tematy wymagające wiedzy prawnej, dotyczące przepisów, kar, postępowań, praw i obowiązków (np. jazda po alkoholu, rozwód, umowa o pracę, mandat za prędkość, eksmisja)
- "finanse" — tematy dotyczące pieniędzy, inwestycji, podatków, kredytów, emerytur
- "zdrowie" — tematy dotyczące zdrowia, chorób, leków, terapii, diety zdrowotnej
- "general" — tematy niezwiązane z YMYL (np. ogrodnictwo, hobby, przepisy kulinarne)

Odpowiedz WYŁĄCZNIE w formacie JSON (bez markdown):
{{"category": "prawo|finanse|zdrowie|general", "confidence": 0.0-1.0, "reasoning": "krótkie uzasadnienie po polsku", "reasoning_keywords": ["słowo1", "słowo2"]}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        # Parse JSON — handle possible markdown fences
        import json
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        
        # Validate
        if result.get("category") not in ("prawo", "finanse", "zdrowie", "general"):
            print(f"[YMYL] ⚠️ Claude returned invalid category: {result.get('category')}")
            return None
        
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0))))
        print(f"[YMYL] 🧠 Claude: '{main_keyword}' → {result['category']} ({result['confidence']}) — {result.get('reasoning', '')[:80]}")
        return result
        
    except Exception as e:
        print(f"[YMYL] ⚠️ Claude classifier error: {e}")
        return None


def detect_category(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Wykrywa kategorię treści (prawo/finanse/zdrowie/general).
    
    🆕 v45.3: Dwustopniowy klasyfikator:
      1. Keyword pre-filter — szybki, dla oczywistych trafień (score >= 4)
      2. Claude Haiku semantic — dla niejednoznacznych przypadków (score 0-3)
      
    Dzięki temu system rozumie DOWOLNY temat prawny/medyczny/finansowy,
    nie tylko te z hardkodowanej listy.
    """
    additional_keywords = additional_keywords or []
    all_keywords = [main_keyword] + additional_keywords
    combined_text = " ".join(all_keywords).lower()
    
    # Lemmatyzacja (jeśli spaCy dostępne)
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
    
    # ──────────────────────────────────────────────────────
    # STAGE 1: Keyword pre-filter (szybki, zero kosztu)
    # ──────────────────────────────────────────────────────
    LEGAL_WEIGHTED = {
        # Silne sygnały (waga 3) — jednoznacznie prawne
        "ubezwłasnowolnienie": 3, "kodeks cywilny": 3, "kodeks karny": 3,
        "kodeks pracy": 3, "k.c.": 3, "k.p.c.": 3, "k.r.o.": 3, "k.k.": 3,
        "zdolność do czynności prawnych": 3, "postępowanie sądowe": 3,
        "zachowek": 3, "przedawnienie roszczenia": 3, "zasiedzenie": 3,
        "władza rodzicielska": 3, "opiekun prawny": 3, "kurator sądowy": 3,
        "radca prawny": 3, "komornik": 3, "notariusz": 3,
        "kodeks wykroczeń": 3, "k.w.": 3,
        # v47.2: Prawo drogowe / karne — jazda po alkoholu, narkotyki etc.
        "jazda po alkoholu": 3, "prowadzenie pod wpływem": 3,
        "stan nietrzeźwości": 3, "konfiskata pojazdu": 3,
        "prowadzenie pojazdu pod wpływem": 3,
        "art. 178a": 3, "art. 87": 3,
        "pozbawienie wolności": 3, "areszt": 3,
        "recydywa": 3, "warunkowe umorzenie": 3,
        # v47.2: Prawo karne — częste przestępstwa
        "kradzież": 3, "rozbój": 3, "oszustwo": 3,
        "groźby karalne": 3, "stalking": 3, "znęcanie": 3,
        "nękanie": 3, "przemoc domowa": 3,
        "posiadanie narkotyków": 3, "handel narkotykami": 3,
        # v47.2: Prawo administracyjne
        "pozwolenie na budowę": 3, "warunki zabudowy": 3,
        "decyzja administracyjna": 3, "odwołanie od decyzji": 3,
        # Średnie sygnały (waga 2)
        "testament": 2, "dziedziczenie": 2,
        "rozwód": 2, "alimenty": 2, "separacja": 2,
        "odszkodowanie": 2, "zadośćuczynienie": 2,
        "pozew": 2, "apelacja": 2, "kasacja": 2, "zażalenie": 2,
        "wyrok sądu": 2, "orzeczenie sądu": 2, "sąd okręgowy": 2,
        "przepis prawny": 2, "norma prawna": 2,
        "hipoteka": 2, "zastaw": 2, "poręczenie": 2,
        "przestępstwo": 2, "wykroczenie": 2,
        "adwokat": 2, "ustawa": 2, "rozporządzenie": 2,
        # v47.2: Prawo drogowe — średnie sygnały
        "promil": 2, "zatrzymanie prawa jazdy": 2,
        "zakaz prowadzenia": 2, "mandat": 2, "punkty karne": 2,
        "utrata prawa jazdy": 2, "alkohol za kierownicą": 2,
        "narkotyki za kierownicą": 2, "kontrola drogowa": 2,
        "alkomat": 2, "badanie trzeźwości": 2,
        "świadczenie pieniężne": 2, "fundusz pomocy pokrzywdzonym": 2,
        "dozór elektroniczny": 2, "wyrok w zawieszeniu": 2,
        "grzywna": 2, "kara ograniczenia wolności": 2,
        "kara pozbawienia wolności": 2, "zarzuty": 2,
        "prokuratura": 2, "prokurator": 2,
        # v47.2: Prawo pracy — rozszerzenie
        "zwolnienie dyscyplinarne": 2, "mobbing": 2,
        "wypowiedzenie umowy": 2, "odprawa": 2,
        # Słabe sygnały (waga 1)
        "umowa": 1, "kontrakt": 1, "kara": 1,
        "skarga": 1, "odwołanie": 1,
        "firma": 1, "spółka": 1, "kredyt": 1,
        "spadek": 1, "sąd": 1, "wyrok": 1, "orzeczenie": 1,
        # v47.2: Weak — mogą być prawne w kontekście
        "alkohol": 1, "narkotyki": 1, "policja": 1,
        "zatrzymanie": 1, "areszt": 1, "więzienie": 1,
        "prawo jazdy": 1, "kierowca": 1, "pojazd": 1,
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
    
    # High-confidence keyword match → return immediately (no Claude needed)
    # Score 3+ = at least one strong signal (weight 3) or multiple medium signals
    KEYWORD_CONFIDENCE_THRESHOLD = 3
    if max_score >= KEYWORD_CONFIDENCE_THRESHOLD:
        confidence = min(1.0, max_score / 5)
        category = max_category
        detected = {"prawo": legal_matches, "finanse": finance_matches, "zdrowie": health_matches}.get(category, [])
        is_ymyl = True
        legal_enabled = category == "prawo" and SAOS_AVAILABLE
        
        print(f"[YMYL] ✅ Keyword pre-filter: '{main_keyword}' → {category} (score={max_score}, high confidence)")
        return {
            "category": category,
            "confidence": round(confidence, 2),
            "is_ymyl": is_ymyl,
            "detected_keywords": detected[:10],
            "legal_enabled": legal_enabled,
            "weighted_score": round(max_score, 1),
            "detection_method": "keyword_prefilter",
            "sources_available": {
                "saos": SAOS_AVAILABLE,
                "google": GOOGLE_FALLBACK_AVAILABLE,
                "local": LOCAL_SCRAPER_AVAILABLE,
                "scoring": SCORING_AVAILABLE
            }
        }
    
    # ──────────────────────────────────────────────────────
    # STAGE 2: Claude Haiku semantic classifier
    # (for ambiguous cases: keyword score 0-3)
    # ──────────────────────────────────────────────────────
    claude_result = _classify_ymyl_with_claude(main_keyword, additional_keywords)
    
    if claude_result:
        category = claude_result["category"]
        confidence = claude_result["confidence"]
        is_ymyl = category in ["prawo", "finanse", "zdrowie"] and confidence >= 0.5
        legal_enabled = category == "prawo" and SAOS_AVAILABLE
        
        # Merge: keyword matches + Claude reasoning
        detected = claude_result.get("reasoning_keywords", [])
        if legal_matches and category == "prawo":
            detected = list(set(legal_matches + detected))
        
        print(f"[YMYL] 🧠 Claude classifier: '{main_keyword}' → {category} (conf={confidence})")
        return {
            "category": category,
            "confidence": round(confidence, 2),
            "is_ymyl": is_ymyl,
            "detected_keywords": detected[:10],
            "legal_enabled": legal_enabled,
            "weighted_score": round(max_score, 1),
            "detection_method": "claude_semantic",
            "claude_reasoning": claude_result.get("reasoning", ""),
            "sources_available": {
                "saos": SAOS_AVAILABLE,
                "google": GOOGLE_FALLBACK_AVAILABLE,
                "local": LOCAL_SCRAPER_AVAILABLE,
                "scoring": SCORING_AVAILABLE
            }
        }
    
    # ──────────────────────────────────────────────────────
    # FALLBACK: Claude unavailable, use keyword score as-is
    # ──────────────────────────────────────────────────────
    if max_score == 0:
        print(f"[YMYL] ℹ️ No YMYL signals: '{main_keyword}' → general")
        return {
            "category": "general",
            "confidence": 0.0,
            "is_ymyl": False,
            "detected_keywords": [],
            "legal_enabled": False,
            "weighted_score": 0,
            "detection_method": "keyword_fallback",
            "sources_available": {
                "saos": SAOS_AVAILABLE,
                "google": GOOGLE_FALLBACK_AVAILABLE,
                "local": LOCAL_SCRAPER_AVAILABLE,
                "scoring": SCORING_AVAILABLE
            }
        }
    
    confidence = min(1.0, max_score / 5)
    category = max_category if max_score >= 2 else "general"
    detected = {"prawo": legal_matches, "finanse": finance_matches, "zdrowie": health_matches}.get(category, [])
    is_ymyl = category in ["prawo", "finanse", "zdrowie"] and confidence >= 0.3
    legal_enabled = category == "prawo" and SAOS_AVAILABLE
    
    print(f"[YMYL] ⚠️ Keyword fallback (Claude unavailable): '{main_keyword}' → {category} (score={max_score})")
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "is_ymyl": is_ymyl,
        "detected_keywords": detected[:10],
        "legal_enabled": legal_enabled,
        "weighted_score": round(max_score, 1),
        "detection_method": "keyword_fallback",
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
# v52.5: FILTR TYPÓW SPRAW — civil vs criminal
# ================================================================

_CIVIL_SIG_PATTERNS = re.compile(
    r'^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|I\s+A|II\s+A)?'
    r'\s*(C|Ca|ACa|ACo|ACz|AGa|Cz|Co|Nc|GC|GCo|GCz)\b',
    re.IGNORECASE
)
_CRIMINAL_SIG_PATTERNS = re.compile(
    r'^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|I\s+A|II\s+A)?'
    r'\s*(K|Ka|AKa|AKo|AKz|AKp|AKzp|Kow|Kop|Ko|Kk|Kzd|Kzw|SDI)\b',
    re.IGNORECASE
)

def _is_civil_judgment(signature: str) -> bool:
    """Zwraca True jeśli sygnatura wskazuje na sprawę cywilną."""
    if not signature:
        return False
    sig = signature.strip().upper()
    # Usuń prefix sądu (np. "SO w Słupsku I C 245/21" → "I C 245/21")
    sig = re.sub(r'^(SO|SA|SR|SN|TK|NSA|WSA)\s+.*?\s+', '', sig)
    return bool(_CIVIL_SIG_PATTERNS.match(sig))

def _is_criminal_judgment(signature: str) -> bool:
    """Zwraca True jeśli sygnatura wskazuje na sprawę karną."""
    if not signature:
        return False
    sig = signature.strip().upper()
    sig = re.sub(r'^(SO|SA|SR|SN|TK|NSA|WSA)\s+.*?\s+', '', sig)
    return bool(_CRIMINAL_SIG_PATTERNS.match(sig))

def _infer_required_judgment_type(article_hints: list, keyword: str) -> str:
    """
    Na podstawie article_hints i słowa kluczowego określa jaki typ wyroku jest potrzebny.
    Zwraca: 'criminal', 'civil', lub 'any'
    """
    combined = ' '.join(article_hints + [keyword]).lower()
    criminal_markers = ['k.k.', 'kodeks karny', 'kk', 'art. 178', 'art. 177',
                        'art. 286', 'art. 297', 'art. 300', 'przestępstwo',
                        'przestępstw', 'kodeks karny', 'postępowanie karne',
                        'wykroczenie', 'k.w.', 'kodeks wykroczeń']
    civil_markers = ['k.c.', 'kodeks cywilny', 'odszkodowanie', 'umowa',
                     'odpowiedzialność cywilna', 'k.r.o.', 'kodeks rodzinny']
    
    criminal_score = sum(1 for m in criminal_markers if m in combined)
    civil_score = sum(1 for m in civil_markers if m in combined)
    
    if criminal_score > civil_score and criminal_score >= 1:
        return 'criminal'
    elif civil_score > criminal_score and civil_score >= 1:
        return 'civil'
    return 'any'

def filter_judgments_by_type(judgments: list, required_type: str) -> list:
    """
    Filtruje wyroki po typie (karne/cywilne).
    Jeśli required_type == 'criminal': odrzuca wyroki I C, II C, ACa itp.
    Jeśli required_type == 'civil': odrzuca wyroki II K, AKa itp.
    Jeśli required_type == 'any': nie filtruje.
    """
    if required_type == 'any':
        return judgments
    
    filtered = []
    rejected = []
    for j in judgments:
        sig = j.get('signature', j.get('caseNumber', ''))
        if required_type == 'criminal':
            if _is_civil_judgment(sig):
                rejected.append(sig)
                continue
        elif required_type == 'civil':
            if _is_criminal_judgment(sig):
                rejected.append(sig)
                continue
        filtered.append(j)
    
    if rejected:
        print(f"[LEGAL_MODULE] 🔍 Odrzucono {len(rejected)} wyroków niezgodnych z typem '{required_type}': {rejected[:3]}")
    return filtered


# ================================================================
# GŁÓWNA FUNKCJA - POBIERANIE KONTEKSTU PRAWNEGO
# ================================================================

def get_legal_context_for_article(
    main_keyword: str,
    additional_keywords: List[str] = None,
    force_enable: bool = False,
    max_results: int = 5,
    article_hints: List[str] = None,
    search_queries: List[str] = None,
) -> Dict[str, Any]:
    """
    Główna funkcja - pobiera kontekst prawny z 4 źródeł.
    
    🆕 v47.2: Accepts article_hints from Claude unified classifier.
    When provided, searches SAOS by specific articles (e.g. "art. 178a k.k.")
    in addition to keyword search → much more relevant judgments.
    
    Kolejność:
    1. SAOS API (główne źródło) — keyword + article-specific queries
    2. Google Fallback (gdy SAOS brak)
    3. Local Court Scraper (gdy SAOS < 2 wyniki)
    4. Scoring i filtrowanie
    """
    additional_keywords = additional_keywords or []
    article_hints = article_hints or []
    search_queries = search_queries or []
    
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
            # v47.2: Search by keyword AND by specific articles from Claude
            all_saos_queries = [main_keyword]
            if article_hints:
                # Normalize article references for SAOS search
                for art in article_hints[:4]:
                    # "art. 178a § 1 k.k." → search by article
                    all_saos_queries.append(art)
                print(f"[LEGAL_MODULE] 📜 Claude article hints: {', '.join(article_hints[:4])}")
            if search_queries:
                all_saos_queries.extend(search_queries[:3])
            
            for sq in all_saos_queries:
                print(f"[LEGAL_MODULE] 📡 SAOS query: '{sq}'")
                # Fix #30: parametr to 'topic', nie 'query'
                saos_result = saos_search(
                    topic=sq,
                    max_results=max_results * 2
                )
                
                if saos_result and saos_result.get("judgments"):
                    for j in saos_result["judgments"]:
                        j["source"] = "saos"
                        if sq != main_keyword:
                            j["matched_article"] = sq
                    judgments.extend(saos_result["judgments"])
                    if "saos" not in sources_used:
                        sources_used.append("saos")
                    saos_count += len(saos_result["judgments"])
                    print(f"[LEGAL_MODULE] ✅ SAOS '{sq}': {len(saos_result['judgments'])} wyników")
                    
                    # If we already have enough results, stop querying
                    if saos_count >= max_results * 3:
                        break
                        
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
    
    # 6b. v52.5: Filtr mechaniczny (karne/cywilne) — szybki pre-filter
    required_type = _infer_required_judgment_type(article_hints, main_keyword)
    if required_type != 'any':
        before_type_filter = len(judgments)
        judgments = filter_judgments_by_type(judgments, required_type)
        print(f"[LEGAL_MODULE] ⚖️ Filtr typu '{required_type}': {before_type_filter} → {len(judgments)} wyroków")

    # 6c. v52.5: Weryfikacja AI przez Claude Haiku — wybiera najlepsze, odrzuca niepasujące
    if judgments and CLAUDE_VERIFIER_AVAILABLE:
        try:
            print(f"[LEGAL_MODULE] 🤖 Claude Haiku weryfikuje {len(judgments)} orzeczeń dla: '{main_keyword}'")
            verified = verify_judgments_with_claude(
                article_topic=main_keyword,
                judgments=judgments,
                max_to_select=min(3, max_results)
            )
            if verified.get("status") == "OK" and verified.get("selected"):
                # Mapuj wybrane indeksy z powrotem na obiekty orzeczeń
                selected_indices = {s["index"] - 1 for s in verified["selected"]}
                judgments = [j for i, j in enumerate(judgments) if i in selected_indices]
                print(f"[LEGAL_MODULE] ✅ Claude wybrał {len(judgments)} orzeczeń (reasoning: {verified.get('reasoning', '')[:80]})")
            elif verified.get("status") == "OK" and not verified.get("selected"):
                print(f"[LEGAL_MODULE] ⚠️ Claude odrzucił WSZYSTKIE orzeczenia jako niepasujące do tematu")
                judgments = []
            else:
                print(f"[LEGAL_MODULE] ⚠️ Verifier fallback: {verified.get('error', 'unknown')}")
        except Exception as e:
            print(f"[LEGAL_MODULE] ⚠️ Verifier error (kontynuuję bez): {e}")

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
