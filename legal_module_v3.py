# legal_module_v3.py
# BRAJEN Legal Module v3.0 - Ze scoringiem orzeczeń
# Max 2 sygnatury na artykuł + weryfikacja jakości

"""
===============================================================================
🏛️ BRAJEN LEGAL MODULE v3.0
===============================================================================

Ulepszona wersja:
- Max 2 sygnatury na artykuł
- SCORING orzeczeń (wybór najlepszych)
- Weryfikacja: zawiera przepis? merytoryczny? ma tezę?

===============================================================================
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import re

# Import klienta SAOS
try:
    from saos_client import search_judgments, get_saos_client
    SAOS_AVAILABLE = True
except ImportError:
    SAOS_AVAILABLE = False
    print("[LEGAL_MODULE] ⚠️ SAOS Client not available")

# 🆕 v3.2: Import weryfikatora Claude
try:
    from claude_judgment_verifier import select_best_judgments, CLAUDE_MODEL
    CLAUDE_VERIFIER_AVAILABLE = True
    print(f"[LEGAL_MODULE] ✅ Claude Verifier loaded ({CLAUDE_MODEL})")
except ImportError:
    CLAUDE_VERIFIER_AVAILABLE = False
    print("[LEGAL_MODULE] ⚠️ Claude Verifier not available, using fallback scoring")


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class LegalConfig:
    """Konfiguracja modułu prawnego."""
    
    MAX_CITATIONS_PER_ARTICLE: int = 2
    MIN_SCORE_TO_USE: int = 40  # Minimalna jakość orzeczenia
    FETCH_COUNT: int = 15       # Pobierz więcej, wybierz najlepsze
    MIN_YEAR: int = 2022        # v3.1: Tylko ostatnie 3 lata
    
    # Priorytet sądów
    COURT_PRIORITY: Dict[str, int] = field(default_factory=lambda: {
        "SUPREME": 100,
        "CONSTITUTIONAL": 90,
        "ADMINISTRATIVE": 80,
        "COMMON": 50
    })
    
    # Słowa kluczowe kategorii PRAWO (do detekcji)
    LEGAL_KEYWORDS: List[str] = field(default_factory=lambda: [
        "alimenty", "rozwód", "separacja", "opieka nad dzieckiem",
        "władza rodzicielska", "spadek", "testament", "dziedziczenie",
        "zachowek", "umowa", "odszkodowanie", "zadośćuczynienie",
        "pozew", "roszczenie", "wyrok", "kara", "przestępstwo",
        "wypowiedzenie", "mobbing", "sąd", "adwokat", "komornik"
    ])
    
    # 🆕 v3.2: Mapowanie TEMAT → USTAWA (do weryfikacji kontekstu)
    TOPIC_TO_ACT: Dict[str, List[str]] = field(default_factory=lambda: {
        # Prawo rodzinne → KRO
        "alimenty": ["kro", "k.r.o", "kodeks rodzinny"],
        "rozwód": ["kro", "k.r.o", "kodeks rodzinny"],
        "separacja": ["kro", "k.r.o", "kodeks rodzinny"],
        "opieka nad dzieckiem": ["kro", "k.r.o", "kodeks rodzinny"],
        "władza rodzicielska": ["kro", "k.r.o", "kodeks rodzinny"],
        
        # Prawo spadkowe → KC (księga 4)
        "spadek": ["kc", "k.c", "kodeks cywilny"],
        "testament": ["kc", "k.c", "kodeks cywilny"],
        "dziedziczenie": ["kc", "k.c", "kodeks cywilny"],
        "zachowek": ["kc", "k.c", "kodeks cywilny"],
        
        # Prawo cywilne → KC
        "umowa": ["kc", "k.c", "kodeks cywilny"],
        "odszkodowanie": ["kc", "k.c", "kodeks cywilny"],
        "zadośćuczynienie": ["kc", "k.c", "kodeks cywilny"],
        
        # Prawo pracy → KP
        "wypowiedzenie": ["kp", "k.p", "kodeks pracy"],
        "mobbing": ["kp", "k.p", "kodeks pracy"],
        
        # Prawo karne → KK
        "przestępstwo": ["kk", "k.k", "kodeks karny"],
        "kara": ["kk", "k.k", "kodeks karny"],
    })


CONFIG = LegalConfig()


# ============================================================================
# SCORING ORZECZEŃ
# ============================================================================

def score_judgment(text: str, keyword: str) -> Dict[str, Any]:
    """
    Ocenia jakość orzeczenia.
    
    Kryteria v3.2:
    - Zawiera artykuł ustawy (art. X) → +40 pkt
    - Ma tezę/uzasadnienie prawne → +30 pkt
    - NIE jest czysto proceduralne → +20 pkt
    - Keyword występuje często → +10 pkt
    - 🆕 Przepisy z WŁAŚCIWEJ ustawy → +15 pkt bonus / -20 pkt kara
    
    Dodatkowo: wykrywa KIERUNEK wyroku (za/przeciw/neutralny)
    
    Args:
        text: Pełna treść orzeczenia
        keyword: Słowo kluczowe którego szukamy
        
    Returns:
        Dict ze score i szczegółami
    """
    text_lower = text.lower()
    first_500 = text_lower[:500]  # Początek = sentencja
    
    score = 0
    details = []
    
    # 1. Czy zawiera artykuł ustawy? (+40 pkt) - KLUCZOWE
    article_pattern = r'art\.\s*\d+[a-z]?\s*(?:§\s*\d+)?(?:\s*(?:k\.?[rcpk]\.?|kro|kpc|kpk|kc|kk|kp))?'
    articles_found = re.findall(article_pattern, text_lower, re.IGNORECASE)
    
    if articles_found:
        score += 40
        details.append(f"✓ Zawiera przepisy ({len(articles_found)}x)")
    else:
        details.append("✗ Brak przepisów")
    
    # 2. Czy ma tezę/uzasadnienie prawne? (+30 pkt)
    thesis_phrases = [
        "należy uznać", "zdaniem sądu", "sąd zważył", "w ocenie sądu",
        "nie ulega wątpliwości", "bezspornym jest", "jak słusznie",
        "trafnie wskazał", "prawidłowo ustalił", "słuszne jest stanowisko",
        "przyjąć należy", "sąd podziela", "zasadny jest pogląd"
    ]
    if any(phrase in text_lower for phrase in thesis_phrases):
        score += 30
        details.append("✓ Zawiera uzasadnienie/tezę")
    else:
        details.append("✗ Brak tezy")
    
    # 3. Czy NIE jest czysto proceduralne? (+20 pkt)
    # (umorzenie, odrzucenie z przyczyn formalnych - BEZ meritum)
    procedural_only = [
        "umarza postępowanie", "odrzuca pozew", "odrzuca apelację",
        "zwraca sprawę", "brak opłaty", "niedopuszczalny", "przekazuje sprawę"
    ]
    is_procedural = any(phrase in first_500 for phrase in procedural_only)
    
    if not is_procedural:
        score += 20
        details.append("✓ Nie jest czysto proceduralne")
    else:
        details.append("✗ Czysto proceduralne")
    
    # 4. BONUS: Keyword występuje często (+10 pkt)
    keyword_count = text_lower.count(keyword.lower())
    if keyword_count >= 5:
        score += 10
        details.append(f"✓ Keyword występuje {keyword_count}x")
    
    # 5. 🆕 v3.2: Czy przepisy są z WŁAŚCIWEJ ustawy dla tematu?
    # (+15 pkt bonus jeśli pasują, -20 pkt kara jeśli nie pasują)
    expected_acts = CONFIG.TOPIC_TO_ACT.get(keyword.lower(), [])
    
    if articles_found and expected_acts:
        # Sprawdź czy którykolwiek znaleziony przepis jest z oczekiwanej ustawy
        articles_text = " ".join(articles_found).lower()
        
        has_matching_act = any(act in articles_text for act in expected_acts)
        # Sprawdź też w całym tekście (czasem "kodeks rodzinny" jest osobno)
        has_matching_act = has_matching_act or any(act in text_lower for act in expected_acts)
        
        if has_matching_act:
            score += 15
            details.append(f"✓ Przepisy z właściwej ustawy ({expected_acts[0].upper()})")
        else:
            # Kara za przepisy z INNEJ ustawy (np. KK w artykule o alimentach)
            score -= 20
            details.append(f"✗ Przepisy z INNEJ ustawy (oczekiwano: {expected_acts[0].upper()})")
    
    # ================================================================
    # KIERUNEK WYROKU (za/przeciw/neutralny) - BEZ wpływu na score
    # GPT dostaje tę info żeby wiedzieć jak użyć
    # ================================================================
    direction = "neutralny"
    direction_details = ""
    
    # Wyroki "za" (uwzględniające roszczenie)
    positive_phrases = ["zasądza", "uwzględnia", "zobowiązuje", "nakazuje", "orzeka zgodnie"]
    # Wyroki "przeciw" (oddalające roszczenie, ale z uzasadnieniem!)
    negative_phrases = ["oddala powództwo", "oddala apelację", "nie uwzględnia", "odmawia"]
    
    if any(phrase in first_500 for phrase in positive_phrases):
        direction = "za"
        direction_details = "Sąd uwzględnił roszczenie"
    elif any(phrase in first_500 for phrase in negative_phrases):
        direction = "przeciw"
        direction_details = "Sąd oddalił roszczenie (ale uzasadnienie może być wartościowe!)"
    else:
        direction = "neutralny"
        direction_details = "Brak jasnego rozstrzygnięcia w sentencji"
    
    return {
        "score": score,
        "max_score": 115,  # 40+30+20+10+15
        "details": details,
        "articles_found": articles_found[:3] if articles_found else [],
        "is_usable": score >= CONFIG.MIN_SCORE_TO_USE,
        # 🆕 Kierunek wyroku
        "direction": direction,
        "direction_details": direction_details
    }


def extract_best_excerpt(text: str, keyword: str, context_chars: int = 300) -> str:
    """
    Wyciąga fragment zawierający keyword, starając się zachować PEŁNE ZDANIA.
    v3.2: Poprawione cięcie na granicach zdań + szukanie form pochodnych.
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Szukaj też form pochodnych (alimenty → alimentacyjny, alimentów)
    keyword_base = keyword_lower[:min(6, len(keyword_lower))]  # Pierwsze 6 liter
    
    # 1. Znajdź keyword (preferuj pozycję z przepisem w pobliżu)
    article_pattern = r'art\.\s*\d+'
    
    # Szukaj pełnego słowa lub bazy
    keyword_positions = []
    start_search = 0
    while True:
        # Najpierw szukaj pełnego słowa
        pos = text_lower.find(keyword_lower, start_search)
        if pos == -1:
            # Jeśli nie ma, szukaj bazy (np. "aliment" znajdzie "alimentacyjny")
            pos = text_lower.find(keyword_base, start_search)
        if pos == -1:
            break
        keyword_positions.append(pos)
        start_search = pos + 1
    
    if not keyword_positions:
        # Fallback: zwróć początek jeśli nie znaleziono
        end = text.find('.', 0, context_chars)
        if end != -1:
            return text[:end + 1].strip()
        return text[:context_chars].strip() + "..."
    
    # Preferuj pozycję z przepisem w pobliżu
    best_pos = keyword_positions[0]
    for pos in keyword_positions:
        context_start = max(0, pos - 150)
        context_end = min(len(text), pos + 150)
        context = text[context_start:context_end]
        
        if re.search(article_pattern, context, re.IGNORECASE):
            best_pos = pos
            break
    
    # 2. Ustal wstępny zakres
    start = max(0, best_pos - context_chars // 2)
    end = min(len(text), best_pos + context_chars // 2)
    
    # 3. Rozszerz do granic zdań (szukamy kropki)
    # Szukamy w lewo początku zdania
    sent_start = text.rfind('.', 0, start)
    if sent_start != -1:
        start = sent_start + 2  # +2 żeby pominąć kropkę i spację
    else:
        start = 0
    
    # Szukamy w prawo końca zdania
    sent_end = text.find('.', end)
    if sent_end != -1 and sent_end < end + 100:  # max 100 znaków dalej
        end = sent_end + 1
    
    # 4. Wyczyść i zwróć
    excerpt = text[start:end].strip()
    
    # Usuń ewentualne śmieci na początku (np. fragment numeracji)
    excerpt = re.sub(r'^\d+\.\s*', '', excerpt)
    excerpt = re.sub(r'^[a-z]\)\s*', '', excerpt)
    
    # Dodaj elipsy jeśli to nie początek/koniec
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text) - 1:
        excerpt = excerpt + "..."
    
    return excerpt


# ============================================================================
# DETEKCJA KATEGORII
# ============================================================================

def detect_category(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Dict[str, Any]:
    """Wykrywa czy artykuł dotyczy tematyki prawnej."""
    
    all_text = main_keyword.lower()
    if additional_keywords:
        all_text += " " + " ".join([kw.lower() for kw in additional_keywords])
    
    matched = []
    for keyword in CONFIG.LEGAL_KEYWORDS:
        if keyword.lower() in all_text:
            matched.append(keyword)
    
    is_legal = len(matched) >= 1
    confidence = "HIGH" if len(matched) >= 3 else "MEDIUM" if len(matched) >= 1 else "LOW"
    
    return {
        "detected_category": "prawo" if is_legal else "inne",
        "is_legal": is_legal,
        "confidence": confidence,
        "matched_keywords": matched[:5],
        "legal_module_active": is_legal and SAOS_AVAILABLE
    }


# ============================================================================
# POBIERANIE NAJLEPSZYCH ORZECZEŃ
# ============================================================================

def get_best_judgments_for_article(
    main_keyword: str,
    max_results: int = 2
) -> Dict[str, Any]:
    """
    Pobiera najlepsze orzeczenia dla artykułu.
    
    🆕 v3.2: Używa Claude do weryfikacji kontekstowej!
    
    Proces:
    1. Pobierz 15 orzeczeń z SAOS (full-text search)
    2. Claude weryfikuje i wybiera 2 najlepsze (kontekstowo!)
    3. Fallback na prosty scoring jeśli Claude niedostępny
    """
    if not SAOS_AVAILABLE:
        return {
            "status": "DISABLED",
            "message": "SAOS module not available",
            "judgments": []
        }
    
    # Wyciągnij keyword
    search_keyword = _extract_legal_keyword(main_keyword)
    
    if not search_keyword:
        return {
            "status": "NO_KEYWORD",
            "message": f"Nie znaleziono słowa prawnego w: {main_keyword}",
            "judgments": []
        }
    
    # Pobierz orzeczenia z SAOS
    results = search_judgments(
        keyword=search_keyword,
        max_results=CONFIG.FETCH_COUNT,
        min_year=CONFIG.MIN_YEAR
    )
    
    if results.get("status") != "OK":
        return results
    
    all_judgments = results.get("judgments", [])
    
    if not all_judgments:
        return {
            "status": "NO_RESULTS",
            "message": f"Brak orzeczeń dla: {search_keyword}",
            "judgments": []
        }
    
    # Przygotuj excerpty dla każdego orzeczenia
    for j in all_judgments:
        text = j.get("full_text", "") or j.get("excerpt", "")
        j["excerpt"] = extract_best_excerpt(text, search_keyword)
    
    # ================================================================
    # 🆕 v3.2: CLAUDE WERYFIKUJE ORZECZENIA
    # ================================================================
    if CLAUDE_VERIFIER_AVAILABLE:
        print(f"[LEGAL_MODULE] 🤖 Claude weryfikuje {len(all_judgments)} orzeczeń dla '{main_keyword}'")
        
        claude_result = select_best_judgments(
            article_topic=main_keyword,
            judgments=all_judgments,
            max_to_select=max_results,
            use_claude=True
        )
        
        if claude_result["status"] == "OK" and claude_result["selected"]:
            best_judgments = claude_result["selected"]
            method = claude_result["method"]
            reasoning = claude_result.get("reasoning", "")
            
            print(f"[LEGAL_MODULE] ✅ Claude wybrał {len(best_judgments)} orzeczeń (method: {method})")
            
            return {
                "status": "OK",
                "keyword_used": search_keyword,
                "total_found": results.get("total_found", 0),
                "analyzed": len(all_judgments),
                "selection_method": method,
                "claude_reasoning": reasoning,
                "judgments": best_judgments,
                "instruction": _build_article_instruction(best_judgments)
            }
    
    # ================================================================
    # FALLBACK: Prosty scoring (gdy Claude niedostępny)
    # ================================================================
    print(f"[LEGAL_MODULE] ⚠️ Fallback na prosty scoring")
    
    scored_judgments = []
    for j in all_judgments:
        text = j.get("full_text", "") or j.get("excerpt", "")
        scoring = score_judgment(text, search_keyword)
        
        if scoring["is_usable"]:
            scored_judgments.append({
                **j,
                "score": scoring["score"],
                "direction": scoring["direction"],
                "verified_by_claude": False
            })
    
    if not scored_judgments:
        scored_judgments = all_judgments[:max_results]
    
    # Sortuj i weź najlepsze
    sorted_judgments = sorted(
        scored_judgments,
        key=lambda x: (
            x.get("score", 0),
            CONFIG.COURT_PRIORITY.get(x.get("court_type", "COMMON"), 0),
            x.get("date", "2000-01-01")
        ),
        reverse=True
    )
    
    best_judgments = sorted_judgments[:max_results]
    
    return {
        "status": "OK",
        "keyword_used": search_keyword,
        "total_found": results.get("total_found", 0),
        "analyzed": len(all_judgments),
        "selection_method": "fallback_scoring",
        "judgments": best_judgments,
        "instruction": _build_article_instruction(best_judgments)
    }


def _extract_legal_keyword(text: str) -> Optional[str]:
    """Wyciąga słowo prawne do wyszukania."""
    text_lower = text.lower()
    
    for keyword in CONFIG.LEGAL_KEYWORDS:
        if keyword.lower() in text_lower:
            return keyword
    
    words = text_lower.split()[:2]
    return " ".join(words) if words else None


def _build_article_instruction(judgments: List[Dict]) -> str:
    """Buduje MINIMALNĄ instrukcję dla GPT."""
    
    if not judgments:
        return ""
    
    # Skondensowana instrukcja - minimum pól w prompcie
    lines = [
        f"⚖️ ORZECZENIA (max {CONFIG.MAX_CITATIONS_PER_ARTICLE}, skopiuj dokładnie sygnaturę):"
    ]
    
    for i, j in enumerate(judgments, 1):
        direction = j.get("direction", "")
        dir_marker = "✓" if direction == "za" else "✗" if direction == "przeciw" else "○"
        
        # Pokaż cytowany przepis jeśli dostępny
        article = j.get("article_cited", "")
        article_str = f" [{article}]" if article else ""
        
        lines.append(f"{i}. {j.get('citation', '')}{article_str} [{dir_marker}]")
        
        # Dodaj URL źródła
        url = j.get("url", "")
        if url:
            lines.append(f"   🔗 Źródło: {url}")
        
        # Dodaj uzasadnienie Claude'a jeśli dostępne
        claude_reason = j.get("claude_reason", "")
        if claude_reason:
            lines.append(f"   Pasuje: {claude_reason}")
        
        lines.append(f"   \"{j.get('excerpt', '')[:120]}...\"")
    
    lines.append("")
    lines.append("Wzór: \"Jak wskazał [Sąd] w wyroku z [data] (sygn. [X]), ...\"")
    lines.append("Jeśli [✗]: \"Warto zauważyć, że sądy oddalają gdy...\"")
    lines.append("⚠️ PODLINKUJ sygnaturę do źródła SAOS!")
    lines.append("Koniec: *Nie stanowi porady prawnej.*")
    
    return "\n".join(lines)


# ============================================================================
# WALIDACJA CAŁEGO ARTYKUŁU
# ============================================================================

def validate_article_citations(full_text: str) -> Dict[str, Any]:
    """Waliduje liczbę sygnatur w całym artykule."""
    
    patterns = [
        r'\b[IVX]+\s+[A-Z]+\s+\d+/\d+\b',
        r'\bsygn\.\s*[IVX\d]+\s*[A-Za-z]+\s*\d+/\d+',
        r'\b[IVX]?\s*(?:C|K|Ca|Ka|ACa|AKa|CZP)\s*\d+/\d+',
    ]
    
    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        found.update(matches)
    
    count = len(found)
    
    if count == 0:
        status = "INFO"
        message = "Brak sygnatur - rozważ dodanie 1-2 orzeczeń"
    elif count <= CONFIG.MAX_CITATIONS_PER_ARTICLE:
        status = "OK"
        message = f"Znaleziono {count} sygnatur ✓"
    else:
        status = "WARNING"
        message = f"Za dużo sygnatur ({count}), max {CONFIG.MAX_CITATIONS_PER_ARTICLE}"
    
    has_disclaimer = any(phrase in full_text.lower() for phrase in [
        "nie stanowi porady prawnej",
        "charakter informacyjny"
    ])
    
    return {
        "status": status,
        "message": message,
        "citations_found": count,
        "citations_limit": CONFIG.MAX_CITATIONS_PER_ARTICLE,
        "citations": list(found)[:5],
        "has_disclaimer": has_disclaimer,
        "disclaimer_reminder": None if has_disclaimer else "⚠️ Dodaj disclaimer!"
    }


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def get_legal_context_for_article(
    main_keyword: str,
    additional_keywords: List[str] = None,
    force_enable: bool = False
) -> Dict[str, Any]:
    """
    Główna funkcja - zwraca kontekst prawny dla artykułu.
    """
    category = detect_category(main_keyword, additional_keywords)
    
    if not category["is_legal"] and not force_enable:
        return {
            "legal_module_active": False,
            "category": category,
            "judgments": [],
            "instruction": None
        }
    
    judgments_result = get_best_judgments_for_article(main_keyword)
    
    return {
        "legal_module_active": True,
        "category": category,
        "keyword_used": judgments_result.get("keyword_used"),
        "stats": {
            "total_found": judgments_result.get("total_found", 0),
            "analyzed": judgments_result.get("analyzed", 0),
            "passed_scoring": judgments_result.get("passed_scoring", 0)
        },
        "judgments": judgments_result.get("judgments", []),
        "instruction": judgments_result.get("instruction", ""),
        "max_citations": CONFIG.MAX_CITATIONS_PER_ARTICLE,
        "disclaimer_required": True
    }


# ============================================================================
# DISCLAIMER
# ============================================================================

LEGAL_DISCLAIMER = "*Artykuł ma charakter informacyjny i nie stanowi porady prawnej.*"


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🏛️ BRAJEN Legal Module v3.0 Test\n")
    
    # Test scoringu
    print("=" * 50)
    print("TEST: Scoring orzeczenia")
    print("=" * 50)
    
    sample_text = """
    Sąd Najwyższy orzeka, że na podstawie art. 133 KRO obowiązek alimentacyjny 
    polega na dostarczaniu środków utrzymania. Zdaniem Sądu, przy ustalaniu 
    wysokości alimentów należy brać pod uwagę możliwości zarobkowe zobowiązanego
    zgodnie z art. 135 § 1 KRO. Powództwo zasługuje na uwzględnienie.
    """
    
    result = score_judgment(sample_text, "alimenty")
    print(f"Score: {result['score']}/{result['max_score']}")
    for detail in result['details']:
        print(f"  {detail}")
    print(f"Przepisy: {result['articles_found']}")
    print(f"Użyteczne: {result['is_usable']}")
    
    # Test złego orzeczenia
    print("\n" + "=" * 50)
    print("TEST: Słabe orzeczenie")
    print("=" * 50)
    
    bad_text = """
    Sąd oddala powództwo w całości. Apelacja nie zasługuje na uwzględnienie.
    Koszty postępowania ponosi powód.
    """
    
    result2 = score_judgment(bad_text, "alimenty")
    print(f"Score: {result2['score']}/{result2['max_score']}")
    for detail in result2['details']:
        print(f"  {detail}")
    print(f"Użyteczne: {result2['is_usable']}")
