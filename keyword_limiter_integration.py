"""
===============================================================================
🔧 KEYWORD LIMITER INTEGRATION v42.2
===============================================================================

Moduł integracyjny dla keyword_limiter.py → firestore_tracker_routes.py

INSTRUKCJA INTEGRACJI:
1. Skopiuj ten plik do katalogu projektu
2. W firestore_tracker_routes.py dodaj import na początku:
   
   try:
       from keyword_limiter_integration import (
           get_enhanced_stuffing_check,
           validate_headers_and_structure,
           KEYWORD_LIMITER_ENABLED
       )
   except ImportError:
       KEYWORD_LIMITER_ENABLED = False
       print("[TRACKER] ⚠️ Keyword Limiter Integration not available")

3. W sekcji stuffing_warnings (linia ~633) zamień:
   
   PRZED:
   stuffing_warnings = get_stuffing_warnings(batch_text, keywords_state)
   
   PO:
   if KEYWORD_LIMITER_ENABLED:
       stuffing_result = get_enhanced_stuffing_check(batch_text, keywords_state)
       stuffing_warnings = stuffing_result.get("warnings", [])
       header_warnings = validate_headers_and_structure(
           batch_text, 
           main_keyword=project_data.get("main_keyword", ""),
           title=project_data.get("title", "")
       )
       stuffing_warnings.extend(header_warnings.get("warnings", []))
   else:
       stuffing_warnings = get_stuffing_warnings(batch_text, keywords_state)

===============================================================================
"""

from typing import Dict, List, Any, Optional

# ================================================================
# IMPORT KEYWORD LIMITER
# ================================================================

KEYWORD_LIMITER_ENABLED = False
try:
    from keyword_limiter import (
        get_dynamic_stuffing_limit,
        check_batch_stuffing,
        check_header_variation,
        check_structural_density,
        check_value_proposition,
        check_readable_headers,
        check_keyword_density,
        validate_keyword_limits,
        StuffingConfig
    )
    KEYWORD_LIMITER_ENABLED = True
    print("[KEYWORD_LIMITER_INTEGRATION] ✅ Keyword Limiter v35.1 loaded")
except ImportError as e:
    print(f"[KEYWORD_LIMITER_INTEGRATION] ⚠️ Keyword Limiter not available: {e}")
    
    # Fallback functions
    def get_dynamic_stuffing_limit(kw, kw_type, word_count, batch_mode=True):
        """Fallback: stały limit."""
        limits = {"MAIN": 3, "BASIC": 2, "EXTENDED": 2}
        return limits.get(kw_type.upper(), 2)
    
    def check_batch_stuffing(text, keywords_state, batch_mode=True):
        """Fallback: podstawowe sprawdzenie."""
        return {"valid": True, "stuffed_keywords": [], "warnings": []}


# ================================================================
# ENHANCED STUFFING CHECK
# ================================================================

def get_enhanced_stuffing_check(
    batch_text: str,
    keywords_state: Dict,
    batch_mode: bool = True
) -> Dict[str, Any]:
    """
    🆕 v42.2: Enhanced stuffing check z dynamicznymi limitami.
    
    Zastępuje statyczne limity (DENSITY_MAX = 3.0) dynamicznymi
    zależnymi od długości tekstu i typu frazy.
    
    Args:
        batch_text: Tekst batcha
        keywords_state: Stan keywords z projektu
        batch_mode: True dla pojedynczego batcha
        
    Returns:
        {
            "valid": bool,
            "stuffed_keywords": [...],
            "warnings": [str],
            "dynamic_limits_used": bool,
            "word_count": int
        }
    """
    if not KEYWORD_LIMITER_ENABLED:
        # Fallback do podstawowego sprawdzenia
        return _legacy_stuffing_check(batch_text, keywords_state)
    
    # Użyj dynamicznych limitów z keyword_limiter
    result = check_batch_stuffing(batch_text, keywords_state, batch_mode)
    
    # Dodaj info że użyto dynamicznych limitów
    result["dynamic_limits_used"] = True
    
    # Format warnings dla zgodności
    formatted_warnings = []
    for stuffed in result.get("stuffed_keywords", []):
        kw = stuffed.get("keyword", "")
        count = stuffed.get("count", 0)
        limit = stuffed.get("limit", 0)
        kw_type = stuffed.get("type", "BASIC")
        
        formatted_warnings.append(
            f"⚠️ STUFFING: '{kw}' ({kw_type}) użyte {count}× "
            f"(dynamiczny limit: {limit}× dla {result.get('word_count', 0)} słów)"
        )
    
    result["warnings"] = formatted_warnings
    
    return result


def _legacy_stuffing_check(batch_text: str, keywords_state: Dict) -> Dict[str, Any]:
    """Legacy fallback gdy keyword_limiter niedostępny."""
    import re
    
    warnings = []
    stuffed = []
    paragraphs = batch_text.split('\n\n')
    
    for rid, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() not in ["BASIC", "MAIN"]:
            continue
        
        keyword = meta.get("keyword", "").lower()
        if not keyword:
            continue
        
        for para in paragraphs:
            if para.lower().count(keyword) > 3:
                warnings.append(f"⚠️ '{meta.get('keyword')}' występuje >3× w jednym akapicie")
                stuffed.append({
                    "keyword": meta.get("keyword"),
                    "count": para.lower().count(keyword),
                    "limit": 3,
                    "type": meta.get("type", "BASIC")
                })
                break
    
    return {
        "valid": len(stuffed) == 0,
        "stuffed_keywords": stuffed,
        "warnings": warnings,
        "dynamic_limits_used": False,
        "word_count": len(batch_text.split())
    }


# ================================================================
# HEADER & STRUCTURE VALIDATION
# ================================================================

def validate_headers_and_structure(
    batch_text: str,
    main_keyword: str = "",
    title: str = "",
    h1: str = "",
    meta_description: str = ""
) -> Dict[str, Any]:
    """
    🆕 v42.2: Walidacja nagłówków i struktury z keyword_limiter.
    
    Sprawdza:
    - Header variation (max 3 wystąpienia main keyword w nagłówkach)
    - Structural density (max 6 wystąpień w H1/H2)
    - Value proposition w tytule
    - Readable headers (min 3 słowa lub pytanie)
    
    Returns:
        {
            "valid": bool,
            "checks": {...},
            "warnings": [str],
            "recommendations": [str]
        }
    """
    if not KEYWORD_LIMITER_ENABLED:
        return {
            "valid": True,
            "checks": {},
            "warnings": [],
            "recommendations": [],
            "enabled": False
        }
    
    warnings = []
    recommendations = []
    checks = {}
    
    # 1. Header variation check
    header_check = check_header_variation(batch_text, main_keyword)
    checks["header_variation"] = header_check
    if not header_check.get("valid", True):
        warning = header_check.get("warning", "")
        if warning:
            warnings.append(f"⚠️ HEADER: {warning}")
            recommendations.append("Użyj synonimów w nagłówkach zamiast powtarzać frazę główną")
    
    # 2. Structural density check
    density_check = check_structural_density(batch_text, main_keyword)
    checks["structural_density"] = density_check
    if not density_check.get("valid", True):
        warning = density_check.get("warning", "")
        if warning:
            warnings.append(f"⚠️ STRUCTURE: {warning}")
            recommendations.append("Ogranicz keyword w nagłówkach - użyj w treści")
    
    # 3. Value proposition (tylko dla pierwszego batcha z tytułem)
    if title:
        value_check = check_value_proposition(title, main_keyword)
        checks["value_proposition"] = value_check
        if not value_check.get("has_value", True):
            recommendations.append("Tytuł powinien zawierać obietnicę wartości (np. 'kompletny przewodnik', 'krok po kroku')")
        if value_check.get("keyword_stuffing", False):
            warnings.append("⚠️ TITLE: Keyword stuffing w tytule!")
    
    # 4. Readable headers
    readable_check = check_readable_headers(batch_text)
    checks["readable_headers"] = readable_check
    if not readable_check.get("valid", True):
        problematic = readable_check.get("problematic", [])
        if problematic:
            warnings.append(f"⚠️ HEADERS: {len(problematic)} nagłówków za krótkich: {', '.join(problematic[:2])}")
            recommendations.append("Nagłówki powinny mieć min. 3 słowa lub być pytaniami")
    
    # 5. Keyword density (globalny check)
    density_result = check_keyword_density(batch_text, main_keyword, max_density=0.020)
    checks["keyword_density"] = density_result
    if not density_result.get("valid", True):
        warning = density_result.get("warning", "")
        if warning:
            warnings.append(f"⚠️ DENSITY: {warning}")
    
    all_valid = all(c.get("valid", True) for c in checks.values())
    
    return {
        "valid": all_valid,
        "checks": checks,
        "warnings": warnings,
        "recommendations": recommendations,
        "enabled": True
    }


# ================================================================
# GET DYNAMIC LIMIT (HELPER)
# ================================================================

def get_limit_for_keyword(
    keyword: str,
    keyword_type: str,
    text_word_count: int,
    batch_mode: bool = True
) -> int:
    """
    Helper: Pobiera dynamiczny limit dla pojedynczej frazy.
    
    Args:
        keyword: Fraza kluczowa
        keyword_type: "MAIN", "BASIC", "EXTENDED"
        text_word_count: Liczba słów w tekście
        batch_mode: True dla batcha, False dla całego artykułu
        
    Returns:
        int: Maksymalna dozwolona liczba wystąpień
    """
    if not KEYWORD_LIMITER_ENABLED:
        # Stałe limity jako fallback
        limits = {"MAIN": 3, "BASIC": 2, "EXTENDED": 2}
        return limits.get(keyword_type.upper(), 2)
    
    return get_dynamic_stuffing_limit(keyword, keyword_type, text_word_count, batch_mode)


# ================================================================
# FULL VALIDATION (KOMBINACJA WSZYSTKIEGO)
# ================================================================

def validate_batch_with_keyword_limiter(
    batch_text: str,
    keywords_state: Dict,
    main_keyword: str = "",
    title: str = "",
    h1: str = "",
    is_intro: bool = False
) -> Dict[str, Any]:
    """
    🆕 v42.2: Pełna walidacja batcha z keyword_limiter.
    
    Łączy:
    - Enhanced stuffing check (dynamiczne limity)
    - Header/structure validation
    - Optional: title/H1 validation dla INTRO
    
    Returns:
        {
            "valid": bool,
            "stuffing": {...},
            "structure": {...},
            "all_warnings": [str],
            "all_recommendations": [str],
            "summary": str
        }
    """
    all_warnings = []
    all_recommendations = []
    
    # 1. Stuffing check
    stuffing_result = get_enhanced_stuffing_check(batch_text, keywords_state)
    all_warnings.extend(stuffing_result.get("warnings", []))
    
    # 2. Structure check
    structure_result = validate_headers_and_structure(
        batch_text,
        main_keyword=main_keyword,
        title=title if is_intro else "",
        h1=h1 if is_intro else ""
    )
    all_warnings.extend(structure_result.get("warnings", []))
    all_recommendations.extend(structure_result.get("recommendations", []))
    
    # 3. Summary
    issues_count = len(all_warnings)
    if issues_count == 0:
        summary = "✅ Batch OK - dynamiczne limity spełnione"
    elif issues_count <= 2:
        summary = f"⚠️ {issues_count} ostrzeżeń - rozważ poprawki"
    else:
        summary = f"❌ {issues_count} problemów - wymaga poprawy"
    
    overall_valid = stuffing_result.get("valid", True) and structure_result.get("valid", True)
    
    return {
        "valid": overall_valid,
        "stuffing": stuffing_result,
        "structure": structure_result,
        "all_warnings": all_warnings,
        "all_recommendations": all_recommendations,
        "summary": summary,
        "keyword_limiter_enabled": KEYWORD_LIMITER_ENABLED
    }


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'get_enhanced_stuffing_check',
    'validate_headers_and_structure',
    'get_limit_for_keyword',
    'validate_batch_with_keyword_limiter',
    'KEYWORD_LIMITER_ENABLED'
]


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 KEYWORD LIMITER INTEGRATION v42.2 TEST")
    print("=" * 60)
    print(f"\nKeyword Limiter: {'✅ Enabled' if KEYWORD_LIMITER_ENABLED else '❌ Disabled'}")
    
    # Test text
    test_text = """
    h2: Jak ubiegać się o alimenty?
    
    Alimenty to świadczenie finansowe na rzecz dziecka. Alimenty są ważne dla rodziny.
    Wysokość alimentów zależy od wielu czynników. Sąd ustala alimenty na podstawie
    potrzeb dziecka i możliwości zarobkowych rodzica.
    
    h2: Ile wynoszą alimenty w Polsce?
    
    Alimenty w Polsce różnią się w zależności od regionu. Średnie alimenty to około
    500-1500 zł miesięcznie. Alimenty powinny pokrywać podstawowe potrzeby dziecka.
    """
    
    test_keywords_state = {
        "kw1": {"keyword": "alimenty", "type": "MAIN", "actual_uses": 0, "target_min": 8, "target_max": 15},
        "kw2": {"keyword": "świadczenie", "type": "BASIC", "actual_uses": 0, "target_min": 2, "target_max": 4},
    }
    
    # Test stuffing
    print("\n📊 Stuffing Check:")
    result = get_enhanced_stuffing_check(test_text, test_keywords_state)
    print(f"  Valid: {result['valid']}")
    print(f"  Dynamic limits: {result.get('dynamic_limits_used', False)}")
    print(f"  Word count: {result.get('word_count', 0)}")
    for w in result.get("warnings", []):
        print(f"  ⚠️ {w}")
    
    # Test structure
    print("\n📐 Structure Check:")
    struct_result = validate_headers_and_structure(test_text, "alimenty", "Alimenty - kompletny przewodnik")
    print(f"  Valid: {struct_result['valid']}")
    print(f"  Checks: {list(struct_result['checks'].keys())}")
    for w in struct_result.get("warnings", []):
        print(f"  ⚠️ {w}")
    for r in struct_result.get("recommendations", []):
        print(f"  💡 {r}")
    
    # Full validation
    print("\n🎯 Full Validation:")
    full_result = validate_batch_with_keyword_limiter(
        test_text, 
        test_keywords_state,
        main_keyword="alimenty",
        title="Alimenty - kompletny przewodnik 2024",
        is_intro=True
    )
    print(f"  {full_result['summary']}")
