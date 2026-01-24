"""
KEYWORD SYNONYMS MODULE v2.0
Wrapper dla synonym_service.py z dodatkowymi funkcjami dla SEO.

Funkcje:
- generate_exceeded_warning: Generuje ostrzeżenie gdy fraza przekroczyła limit
- generate_softcap_warning: Generuje ostrzeżenie gdy fraza zbliża się do soft cap
- generate_synonyms_prompt_section: Generuje sekcję promptu z synonimami
- get_synonyms: Pobiera synonimy (deleguje do synonym_service)

Backend: synonym_service.py (plWordNet API, cache Firestore, LLM fallback)
"""

from typing import List, Dict, Optional

# ================================================================
# IMPORT BACKEND (synonym_service.py)
# ================================================================
try:
    from synonym_service import (
        get_synonyms as _get_synonyms_backend,
        get_synonyms_batch,
        suggest_synonym_for_repetition,
        STATIC_SYNONYM_MAP
    )
    BACKEND_AVAILABLE = True
    print("[KEYWORD_SYNONYMS] ✅ Using synonym_service.py backend (plWordNet + cache)")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"[KEYWORD_SYNONYMS] ⚠️ synonym_service not available: {e}")
    STATIC_SYNONYM_MAP = {}

# ================================================================
# PREDEFINIOWANE SYNONIMY DLA FRAZ PRAWNYCH
# (rozszerzenie dla tematyki YMYL/prawo)
# ================================================================
LEGAL_SYNONYMS = {
    "ubezwłasnowolnienie": [
        "pozbawienie zdolności do czynności prawnych",
        "ograniczenie zdolności prawnej",
        "instytucja ubezwłasnowolnienia",
        "orzeczenie o ubezwłasnowolnieniu"
    ],
    "sąd": [
        "organ sądowy",
        "sąd orzekający",
        "wymiar sprawiedliwości",
        "instancja sądowa"
    ],
    "wniosek": [
        "podanie",
        "pismo procesowe",
        "żądanie",
        "petycja"
    ],
    "wniosek o ubezwłasnowolnienie": [
        "pismo o ubezwłasnowolnienie",
        "podanie o pozbawienie zdolności prawnej",
        "żądanie ubezwłasnowolnienia"
    ],
    "choroba psychiczna": [
        "zaburzenia psychiczne",
        "schorzenie psychiatryczne",
        "problemy zdrowia psychicznego",
        "dysfunkcje psychiczne"
    ],
    "opiekun prawny": [
        "przedstawiciel ustawowy",
        "kurator",
        "osoba reprezentująca",
        "pełnomocnik ustawowy"
    ],
    "zdolność do czynności prawnych": [
        "zdolność prawna",
        "możliwość dokonywania czynności prawnych",
        "kompetencja prawna",
        "zdolność działania w obrocie prawnym"
    ],
    "postępowanie sądowe": [
        "procedura sądowa",
        "proces",
        "sprawa sądowa",
        "postępowanie przed sądem"
    ],
    "postępowanie o ubezwłasnowolnienie": [
        "sprawa o ubezwłasnowolnienie",
        "procedura ubezwłasnowolnienia",
        "proces o ubezwłasnowolnienie"
    ],
    "biegły": [
        "ekspert sądowy",
        "specjalista",
        "rzeczoznawca",
        "biegły sądowy"
    ],
    "orzeczenie": [
        "wyrok",
        "postanowienie",
        "rozstrzygnięcie",
        "decyzja sądu"
    ],
    "przedstawiciel ustawowy": [
        "opiekun prawny",
        "kurator",
        "reprezentant",
        "pełnomocnik z mocy prawa"
    ]
}


# ================================================================
# CORE FUNCTIONS
# ================================================================

def get_synonyms(keyword: str, max_synonyms: int = 4) -> List[str]:
    """
    Pobierz synonimy dla frazy kluczowej.
    
    Kolejność źródeł:
    1. Predefiniowane synonimy prawne (LEGAL_SYNONYMS)
    2. synonym_service.py backend (plWordNet, cache, LLM)
    3. Fallback: pusta lista
    
    Args:
        keyword: Fraza kluczowa
        max_synonyms: Maksymalna liczba synonimów do zwrócenia
        
    Returns:
        Lista synonimów
    """
    keyword_lower = keyword.lower().strip()
    
    # 1. Sprawdź predefiniowane synonimy prawne
    for key, synonyms in LEGAL_SYNONYMS.items():
        if key in keyword_lower or keyword_lower in key:
            return synonyms[:max_synonyms]
    
    # 2. Użyj backendu synonym_service jeśli dostępny
    if BACKEND_AVAILABLE:
        try:
            result = _get_synonyms_backend(keyword)
            if result and result.get("synonyms"):
                return result["synonyms"][:max_synonyms]
        except Exception as e:
            print(f"[KEYWORD_SYNONYMS] Backend error: {e}")
    
    # 3. Fallback: pusta lista
    return []


def generate_exceeded_warning(keyword: str, actual: int, max_allowed: int) -> str:
    """
    Generuj ostrzeżenie gdy fraza przekroczyła limit.
    
    Args:
        keyword: Fraza kluczowa
        actual: Aktualna liczba użyć
        max_allowed: Maksymalna dozwolona liczba
        
    Returns:
        Tekst ostrzeżenia z sugestiami synonimów
    """
    synonyms = get_synonyms(keyword)
    
    warning = f"⛔ PRZEKROCZONO LIMIT dla '{keyword}' ({actual}/{max_allowed})\n"
    warning += "   NIE UŻYWAJ TEJ FRAZY! "
    
    if synonyms:
        warning += f"Zamiast tego użyj SYNONIMÓW:\n"
        for syn in synonyms[:3]:
            warning += f"   • {syn}\n"
    else:
        warning += "Pomiń tę frazę w tym batchu.\n"
    
    return warning


def generate_softcap_warning(keyword: str, actual: int, target_max: int, soft_max: int) -> str:
    """
    Generuj ostrzeżenie gdy fraza zbliża się do soft cap.
    
    Args:
        keyword: Fraza kluczowa
        actual: Aktualna liczba użyć
        target_max: Cel maksymalny
        soft_max: Miękki limit
        
    Returns:
        Tekst ostrzeżenia
    """
    synonyms = get_synonyms(keyword)
    
    remaining = soft_max - actual
    
    warning = f"⚠️ SOFT CAP dla '{keyword}' ({actual}/{target_max}, max={soft_max})\n"
    warning += f"   Zostało {remaining} użyć do limitu. "
    
    if synonyms and remaining <= 2:
        warning += f"Rozważ SYNONIMY:\n"
        for syn in synonyms[:2]:
            warning += f"   • {syn}\n"
    elif remaining > 2:
        warning += "Używaj oszczędnie.\n"
    
    return warning


def generate_synonyms_prompt_section(exceeded_keywords: List[Dict], softcap_keywords: List[Dict]) -> str:
    """
    Generuj sekcję promptu z synonimami dla GPT.
    
    Args:
        exceeded_keywords: Lista słowników z przekroczonymi frazami
            [{"keyword": "...", "actual": X, "max": Y}, ...]
        softcap_keywords: Lista słowników z frazami przy soft cap
            [{"keyword": "...", "actual": X, "target_max": Y, "soft_max": Z}, ...]
            
    Returns:
        Sekcja promptu do wstrzyknięcia
    """
    if not exceeded_keywords and not softcap_keywords:
        return ""
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("🔄 SYNONIMY I OSTRZEŻENIA FRAZ")
    lines.append("=" * 60)
    
    # Exceeded keywords
    if exceeded_keywords:
        lines.append("\n🛑 FRAZY ZABLOKOWANE (NIE UŻYWAJ!):")
        for kw in exceeded_keywords:
            keyword = kw.get("keyword", "")
            actual = kw.get("actual", 0)
            max_allowed = kw.get("max", 0)
            
            synonyms = get_synonyms(keyword)
            lines.append(f"\n   ❌ '{keyword}' ({actual}/{max_allowed})")
            if synonyms:
                lines.append(f"      → Użyj zamiast tego: {', '.join(synonyms[:3])}")
    
    # Soft cap keywords
    if softcap_keywords:
        lines.append("\n⚠️ FRAZY PRZY LIMICIE (UŻYWAJ OSTROŻNIE):")
        for kw in softcap_keywords:
            keyword = kw.get("keyword", "")
            actual = kw.get("actual", 0)
            soft_max = kw.get("soft_max", 0)
            remaining = soft_max - actual
            
            synonyms = get_synonyms(keyword)
            lines.append(f"\n   ⚠️ '{keyword}' (zostało {remaining}x)")
            if synonyms and remaining <= 2:
                lines.append(f"      → Alternatywy: {', '.join(synonyms[:2])}")
    
    lines.append("")
    return "\n".join(lines)


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def preload_synonyms(keywords: List[str]):
    """
    Preload synonimów dla listy fraz (optymalizacja).
    """
    for kw in keywords:
        get_synonyms(kw)


def get_all_synonyms_for_project(keywords_state: Dict) -> Dict[str, List[str]]:
    """
    Pobierz synonimy dla wszystkich fraz w projekcie.
    
    Args:
        keywords_state: Dict z frazami projektu
        
    Returns:
        Dict {keyword: [synonyms]}
    """
    result = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if keyword:
            result[keyword] = get_synonyms(keyword)
    return result


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=== KEYWORD SYNONYMS v2.0 TEST ===")
    print(f"Backend available: {BACKEND_AVAILABLE}")
    
    # Test podstawowy
    test_keywords = ["ubezwłasnowolnienie", "sąd", "choroba psychiczna", "skóra"]
    
    for kw in test_keywords:
        synonyms = get_synonyms(kw)
        print(f"\n'{kw}' → {synonyms}")
    
    # Test ostrzeżeń
    print("\n" + "=" * 40)
    warning = generate_exceeded_warning("ubezwłasnowolnienie", 28, 24)
    print(warning)
    
    # Test sekcji promptu
    exceeded = [{"keyword": "ubezwłasnowolnienie", "actual": 28, "max": 24}]
    softcap = [{"keyword": "sąd", "actual": 10, "soft_max": 12}]
    section = generate_synonyms_prompt_section(exceeded, softcap)
    print(section)
