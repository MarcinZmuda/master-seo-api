# hierarchical_keyword_dedup.py
"""
===============================================================================
HIERARCHICAL KEYWORD DEDUPLICATION v24.0
===============================================================================
Rozwiazuje problem podwojnego liczenia fraz zagniezdzonych.

PROBLEM:
  Frazy: "renta rodzinna" i "renta"
  Tekst: "Renta rodzinna przysluguje... Renta rodzinna to..."
  
  BEZ DEDUP: "renta rodzinna" = 2, "renta" = 2 (liczy te same!)
  Z DEDUP:   "renta rodzinna" = 2, "renta" = 0 (poprawnie!)

LOGIKA:
  Jesli krotsza fraza wystepuje WEWNATRZ dluzszej frazy,
  odejmujemy te wystapienia od krotszej.

===============================================================================
"""

import re
from typing import Dict, List, Tuple


def deduplicate_keyword_counts(
    raw_counts: Dict[str, int],
    keywords_list: List[str] = None
) -> Dict[str, int]:
    """
    Deduplikuje zliczenia fraz - odejmuje wystapienia zagniezdzonych.
    
    LOGIKA:
    - Dla kazdej frazy odejmujemy wszystkie BEZPOSREDNIE dzieci
    - Bezposrednie dziecko = fraza ktora zawiera nasza fraze, ale nie ma posrednika
    
    Args:
        raw_counts: Surowe zliczenia np. {"renta rodzinna": 5, "renta": 8}
        keywords_list: Lista wszystkich fraz (opcjonalna)
    
    Returns:
        Skorygowane zliczenia
    
    Przyklady:
        >>> deduplicate_keyword_counts({"renta rodzinna": 5, "renta": 8})
        {"renta rodzinna": 5, "renta": 3}
        
        >>> deduplicate_keyword_counts({
        ...     "renta rodzinna": 4,
        ...     "renta wdowia": 3,
        ...     "renta": 12
        ... })
        {"renta rodzinna": 4, "renta wdowia": 3, "renta": 5}
        # renta: 12 - 4 - 3 = 5 (odjete OBA bezposrednie dzieci)
    """
    if not isinstance(raw_counts, dict):
        return {}
    
    if not raw_counts:
        return {}
    
    keywords = keywords_list or list(raw_counts.keys())
    
    # Sortuj od najdluzszej do najkrotszej
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    # Kopia do modyfikacji
    adjusted_counts = raw_counts.copy()
    
    # Dla kazdej krotszej frazy znajdz WSZYSTKIE bezposrednie dzieci
    for i, short_kw in enumerate(sorted_keywords):
        short_kw_lower = short_kw.lower().strip()
        
        if not short_kw_lower:
            continue
        
        # Zbierz wszystkich bezposrednich rodzicow (frazy ktore zawieraja short_kw)
        direct_parents = []
        
        for long_kw in sorted_keywords[:i]:
            long_kw_lower = long_kw.lower().strip()
            
            if _is_phrase_inside(short_kw_lower, long_kw_lower):
                # To jest potencjalny rodzic
                # Sprawdz czy jest "bezposredni" (nie ma posrednika miedzy short a long)
                is_direct = True
                for middle_kw in sorted_keywords[:i]:
                    if middle_kw == long_kw:
                        continue
                    middle_lower = middle_kw.lower().strip()
                    # Czy jest fraza posrodku? (zawiera short i jest zawarta w long)
                    if (_is_phrase_inside(short_kw_lower, middle_lower) and 
                        _is_phrase_inside(middle_lower, long_kw_lower)):
                        is_direct = False
                        break
                
                if is_direct:
                    direct_parents.append(long_kw)
        
        # Odejmij WSZYSTKICH bezposrednich rodzicow
        total_to_subtract = sum(raw_counts.get(parent, 0) for parent in direct_parents)
        current_short = adjusted_counts.get(short_kw, 0)
        adjusted_counts[short_kw] = max(0, current_short - total_to_subtract)
    
    return adjusted_counts


def _is_phrase_inside(short_phrase: str, long_phrase: str) -> bool:
    """
    Sprawdza czy krotsza fraza wystepuje jako cale slowo wewnatrz dluzszej.
    
    "renta" in "renta rodzinna" -> True
    "rent" in "renta rodzinna" -> False (nie cale slowo)
    "rodzinna" in "renta rodzinna" -> True
    """
    if not short_phrase or not long_phrase:
        return False
    
    if len(short_phrase) >= len(long_phrase):
        return False
    
    # Sprawdz czy wystepuje jako cale slowo (word boundary)
    pattern = r'\b' + re.escape(short_phrase) + r'\b'
    return bool(re.search(pattern, long_phrase, re.IGNORECASE))


def get_nesting_map(keywords: List[str]) -> Dict[str, List[str]]:
    """
    Tworzy mape zagniezdzenia fraz.
    
    Args:
        keywords: Lista fraz
    
    Returns:
        Slownik gdzie klucz to fraza, wartosc to lista fraz ktore ja zawieraja
        
    Przyklad:
        >>> get_nesting_map(["renta", "renta rodzinna", "renta wdowia"])
        {
            "renta": ["renta rodzinna", "renta wdowia"],
            "renta rodzinna": [],
            "renta wdowia": []
        }
    """
    nesting = {kw: [] for kw in keywords}
    
    sorted_keywords = sorted(keywords, key=len)
    
    for i, short_kw in enumerate(sorted_keywords):
        for long_kw in sorted_keywords[i+1:]:
            if _is_phrase_inside(short_kw.lower(), long_kw.lower()):
                nesting[short_kw].append(long_kw)
    
    return nesting


def analyze_keyword_hierarchy(keywords: List[str]) -> Dict:
    """
    Analizuje hierarchie fraz i zwraca raport.
    
    Uzyteczne do debugowania i zrozumienia struktury fraz.
    """
    nesting = get_nesting_map(keywords)
    
    nested_phrases = {k: v for k, v in nesting.items() if v}
    independent_phrases = [k for k, v in nesting.items() if not v]
    
    return {
        "total_keywords": len(keywords),
        "nested_phrases": nested_phrases,
        "nested_count": len(nested_phrases),
        "independent_phrases": independent_phrases,
        "independent_count": len(independent_phrases),
        "warning": "Frazy zagniezdzene beda mialy skorygowane zliczenia" if nested_phrases else None
    }


# ============================================================================
# PRZYKLAD UZYCIA
# ============================================================================
if __name__ == "__main__":
    # Test
    raw = {
        "renta rodzinna": 5,
        "renta": 8,
        "rodzinna": 2
    }
    
    print("RAW:", raw)
    print("DEDUP:", deduplicate_keyword_counts(raw))
    # Oczekiwany wynik:
    # renta rodzinna: 5 (bez zmian - najdluzsza)
    # renta: 8 - 5 = 3 (odejmujemy "renta rodzinna")
    # rodzinna: 2 - 5 = 0 (odejmujemy "renta rodzinna", min 0)
    
    print("\nHIERARCHY:", analyze_keyword_hierarchy(list(raw.keys())))
