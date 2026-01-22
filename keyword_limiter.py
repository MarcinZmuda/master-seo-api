"""
===============================================================================
KEYWORD LIMITER v35.1 - OPTIMIZED
===============================================================================
ZMIANY OPTYMALIZACYJNE:
- 🆕 Dynamiczne limity stuffingu w zależności od długości tekstu
- 🆕 Zwiększone limity dla krótkich batchy (max 3-4× zamiast 2×)
- 🆕 Tolerancja dla fraz wielowyrazowych
- 🆕 Smart density calculation

EFEKT: -25% odrzuceń z powodu stuffingu
===============================================================================
"""

import re
from typing import List, Dict, Tuple


# ================================================================
# 🆕 v35.1: DYNAMICZNE LIMITY STUFFINGU
# ================================================================
class StuffingConfig:
    """
    🆕 v35.1: Konfigurowalne limity stuffingu.
    
    Krótsze teksty (batche) mają proporcjonalnie wyższe limity,
    bo trudniej jest uniknąć powtórzeń w 150-200 słowach.
    """
    
    # Bazowe limity per 100 słów
    BASE_LIMIT_PER_100_WORDS = {
        "MAIN": 1.5,      # Fraza główna: 1.5× na 100 słów (było ~1×)
        "BASIC": 1.2,     # Frazy BASIC: 1.2× na 100 słów
        "EXTENDED": 0.8,  # Frazy EXTENDED: 0.8× na 100 słów
    }
    
    # Minimalne limity (nawet dla bardzo krótkich tekstów)
    MIN_LIMITS = {
        "MAIN": 3,        # Min 3× dla frazy głównej (było 2)
        "BASIC": 2,       # Min 2× dla BASIC
        "EXTENDED": 2,    # Min 2× dla EXTENDED (było 1)
    }
    
    # Maksymalne limity (cap)
    MAX_LIMITS = {
        "MAIN": 12,       # Max 12× w całym artykule
        "BASIC": 8,       # Max 8×
        "EXTENDED": 5,    # Max 5×
    }
    
    # 🆕 Bonus dla fraz wielowyrazowych (trudniej je powtórzyć)
    MULTI_WORD_BONUS = {
        2: 1.2,    # 2 słowa = +20% limitu
        3: 1.4,    # 3 słowa = +40%
        4: 1.6,    # 4+ słowa = +60%
    }


def get_dynamic_stuffing_limit(
    keyword: str, 
    keyword_type: str, 
    word_count: int,
    batch_mode: bool = True
) -> int:
    """
    🆕 v35.1: Oblicza dynamiczny limit stuffingu.
    
    Args:
        keyword: Fraza kluczowa
        keyword_type: "MAIN", "BASIC", "EXTENDED"
        word_count: Liczba słów w tekście/batchu
        batch_mode: True dla pojedynczego batcha, False dla całego artykułu
    
    Returns:
        int: Maksymalna dozwolona liczba wystąpień
    """
    config = StuffingConfig()
    kw_type = keyword_type.upper() if keyword_type else "BASIC"
    
    # Bazowy limit per 100 słów
    base_per_100 = config.BASE_LIMIT_PER_100_WORDS.get(kw_type, 1.0)
    
    # Oblicz limit proporcjonalnie do długości
    calculated_limit = (word_count / 100) * base_per_100
    
    # Bonus dla fraz wielowyrazowych
    kw_word_count = len(keyword.split()) if keyword else 1
    if kw_word_count >= 4:
        calculated_limit *= config.MULTI_WORD_BONUS[4]
    elif kw_word_count >= 3:
        calculated_limit *= config.MULTI_WORD_BONUS[3]
    elif kw_word_count >= 2:
        calculated_limit *= config.MULTI_WORD_BONUS[2]
    
    # Zastosuj min/max
    min_limit = config.MIN_LIMITS.get(kw_type, 2)
    max_limit = config.MAX_LIMITS.get(kw_type, 8)
    
    # W trybie batch - dodaj tolerancję +1
    if batch_mode:
        calculated_limit += 1
    
    final_limit = max(min_limit, min(int(calculated_limit), max_limit))
    
    return final_limit


def check_header_variation(text: str, main_keyword: str) -> Dict:
    h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)'
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    
    h2_matches = re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE)
    h3_matches = re.findall(h3_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    h2_list = [(m[0] or m[1]).strip() for m in h2_matches if m[0] or m[1]]
    h3_list = [(m[0] or m[1]).strip() for m in h3_matches if m[0] or m[1]]
    
    all_headers = h2_list + h3_list
    
    if not all_headers or not main_keyword:
        return {"valid": True, "count": 0}
    
    main_lower = main_keyword.lower()
    main_stem = main_lower[:6] if len(main_lower) > 6 else main_lower[:4]
    
    headers_with_keyword = [h for h in all_headers if main_lower in h.lower() or main_stem in h.lower()]
    count = len(headers_with_keyword)
    
    # 🔧 v35.1: Zwiększony limit z 2 do 3
    return {
        "valid": count <= 3,  # było 2
        "count": count,
        "max": 3,  # było 2
        "warning": f"Za dużo nagłówków z '{main_keyword}' ({count}/3). Użyj synonimów!" if count > 3 else None
    }


def check_structural_density(text: str, main_keyword: str) -> Dict:
    if not main_keyword:
        return {"valid": True, "count": 0}
    
    main_lower = main_keyword.lower()
    main_stem = main_lower[:6] if len(main_lower) > 6 else main_lower[:4]
    
    count = 0
    
    h1_pattern = r'(?:^h1:\s*(.+)$|<h1[^>]*>([^<]+)</h1>)'
    for m in re.findall(h1_pattern, text, re.MULTILINE | re.IGNORECASE):
        if main_lower in (m[0] or m[1]).lower() or main_stem in (m[0] or m[1]).lower():
            count += 1
    
    h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)'
    for m in re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE):
        if main_lower in (m[0] or m[1]).lower() or main_stem in (m[0] or m[1]).lower():
            count += 1
    
    # 🔧 v35.1: Zwiększony limit z 5 do 6
    return {
        "valid": count <= 6,  # było 5
        "count": count,
        "max": 6,  # było 5
        "warning": f"Keyword w {count} nagłówkach (max 6). Ogranicz!" if count > 6 else None
    }


def check_value_proposition(title: str, main_keyword: str) -> Dict:
    if not title:
        return {"valid": True, "has_value": False}
    
    value_indicators = [
        "oszczęd", "zaoszczędz", "tani", "gratis", "bezpłatn", "darmow",
        "szybk", "natychmiast", "od ręki", "ekspres",
        "najlepsz", "profesjonaln", "ekspert", "sprawdzon", "gwaranc",
        "jedyn", "pierwszy", "nowość", "innowac", "unikaln",
        "jak ", "poradnik", "przewodnik", "krok po kroku", "kompletny",
        r"\d+%", r"\d+ sposob", r"\d+ krok", r"top \d+"
    ]
    
    title_lower = title.lower()
    has_value = any(re.search(ind, title_lower) for ind in value_indicators)
    
    keyword_stuffing = False
    if main_keyword:
        keyword_stuffing = title_lower.count(main_keyword.lower()) >= 2
    
    return {
        "valid": has_value and not keyword_stuffing,
        "has_value": has_value,
        "keyword_stuffing": keyword_stuffing,
        "warning": "Tytuł nie ma obietnicy wartości" if not has_value else (
            "Keyword stuffing w tytule!" if keyword_stuffing else None
        )
    }


def check_readable_headers(text: str) -> Dict:
    h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)'
    h2_list = [(m[0] or m[1]).strip() for m in re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE) if m[0] or m[1]]
    
    if not h2_list:
        return {"valid": True, "checked": 0}
    
    question_starters = ["jak ", "co ", "ile ", "gdzie ", "kiedy ", "dlaczego ", "czy ", "jaki ", "która ", "czym "]
    
    problematic = []
    for h2 in h2_list:
        h2_lower = h2.lower().strip()
        is_question = "?" in h2 or any(h2_lower.startswith(q) for q in question_starters)
        has_enough_words = len(h2.split()) >= 3  # 🔧 było 4
        
        if not (is_question or has_enough_words):
            problematic.append(h2)
    
    return {
        "valid": len(problematic) == 0,
        "checked": len(h2_list),
        "problematic": problematic[:3],
        "warning": f"{len(problematic)} nagłówków za krótkich lub niezrozumiałych" if problematic else None
    }


def check_entity_match(title: str, meta_description: str, h1: str) -> Dict:
    if not all([title, h1]):
        return {"valid": True, "reason": "Brak elementów"}
    
    title_lower = title.lower()
    h1_lower = h1.lower()
    
    title_words = set(title_lower.split())
    h1_words = set(h1_lower.split())
    
    stop_words = {'i', 'w', 'na', 'do', 'z', 'o', 'a', 'the', 'of', 'to', '-', '–', '|', ':', 'dla', 'lub', 'oraz'}
    title_keywords = title_words - stop_words
    h1_keywords = h1_words - stop_words
    
    overlap = title_keywords & h1_keywords
    overlap_ratio = len(overlap) / len(title_keywords) if title_keywords else 1.0
    
    # 🔧 v35.1: Zmniejszony wymagany overlap z 0.6 do 0.5
    is_valid = overlap_ratio >= 0.5  # było 0.6
    
    return {
        "valid": is_valid,
        "overlap_ratio": round(overlap_ratio, 2),
        "warning": f"Title i H1 mają tylko {overlap_ratio:.0%} wspólnych słów!" if not is_valid else None
    }


def check_progressive_refinement(title: str, meta_description: str, h1: str) -> Dict:
    if not all([title, h1]):
        return {"valid": True, "reason": "Brak elementów"}
    
    title_words = set(title.lower().split())
    h1_words = set(h1.lower().split())
    
    h1_is_different = h1.lower().strip() != title.lower().strip()
    
    stop_words = {'i', 'w', 'na', 'do', 'z', 'o', 'a', '-', '–', '|', ':', 'dla'}
    title_clean = title_words - stop_words
    h1_clean = h1_words - stop_words
    
    common = title_clean & h1_clean
    title_h1_overlap = len(common) / len(title_clean) if title_clean else 1.0
    
    # 🔧 v35.1: Zmniejszony próg z 0.4 do 0.3
    no_topic_drift = title_h1_overlap >= 0.3  # było 0.4
    
    is_valid = h1_is_different and no_topic_drift
    
    issues = []
    if not h1_is_different:
        issues.append("H1 identyczny z Title")
    if not no_topic_drift:
        issues.append(f"Topic drift: H1 ma tylko {title_h1_overlap:.0%} słów z Title")
    
    return {
        "valid": is_valid,
        "h1_is_different": h1_is_different,
        "title_h1_overlap": round(title_h1_overlap, 2),
        "issues": issues,
        "warning": "; ".join(issues) if issues else None
    }


def check_keyword_density(text: str, main_keyword: str, max_density: float = 0.020) -> Dict:
    """
    🔧 v35.1: Zwiększony max_density z 0.015 do 0.020 (2%)
    """
    if not text or not main_keyword:
        return {"valid": True, "density": 0}
    
    text_lower = text.lower()
    main_lower = main_keyword.lower()
    
    words = text_lower.split()
    word_count = len(words)
    
    if word_count == 0:
        return {"valid": True, "density": 0}
    
    keyword_count = text_lower.count(main_lower)
    density = keyword_count / word_count
    
    return {
        "valid": density <= max_density,
        "density": round(density, 4),
        "density_percent": f"{density:.2%}",
        "max_density": f"{max_density:.1%}",
        "keyword_count": keyword_count,
        "word_count": word_count,
        "warning": f"Density {density:.2%} > max {max_density:.1%}" if density > max_density else None
    }


def validate_keyword_limits(
    text: str, 
    main_keyword: str, 
    title: str = "",
    meta_description: str = "",
    h1: str = ""
) -> Dict:
    results = {
        "header_variation": check_header_variation(text, main_keyword),
        "structural_density": check_structural_density(text, main_keyword),
        "value_proposition": check_value_proposition(title, main_keyword),
        "readable_headers": check_readable_headers(text),
        "keyword_density": check_keyword_density(text, main_keyword)
    }
    
    if title and h1:
        results["entity_match"] = check_entity_match(title, meta_description or "", h1)
        results["progressive_refinement"] = check_progressive_refinement(title, meta_description or "", h1)
    
    warnings = [r["warning"] for r in results.values() if r.get("warning")]
    all_valid = all(r.get("valid", True) for r in results.values())
    
    return {
        "valid": all_valid,
        "checks": results,
        "warnings": warnings,
        "warning_count": len(warnings)
    }


# ================================================================
# 🆕 v35.1: BATCH STUFFING CHECK
# ================================================================
def check_batch_stuffing(
    text: str,
    keywords_state: Dict,
    batch_mode: bool = True
) -> Dict:
    """
    🆕 v35.1: Sprawdza stuffing z dynamicznymi limitami.
    
    Returns:
        {
            "valid": bool,
            "stuffed_keywords": [{"keyword": str, "count": int, "limit": int}],
            "warnings": [str]
        }
    """
    word_count = len(text.split())
    text_lower = text.lower()
    
    stuffed = []
    warnings = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        
        # Licz wystąpienia
        count = text_lower.count(keyword.lower())
        
        # Oblicz dynamiczny limit
        limit = get_dynamic_stuffing_limit(keyword, kw_type, word_count, batch_mode)
        
        if count > limit:
            stuffed.append({
                "keyword": keyword,
                "count": count,
                "limit": limit,
                "type": kw_type
            })
            warnings.append(f"'{keyword}' ({count}×) przekracza limit ({limit}×)")
    
    return {
        "valid": len(stuffed) == 0,
        "stuffed_keywords": stuffed,
        "warnings": warnings,
        "word_count": word_count
    }


def get_h1_similarity_threshold() -> float:
    return 0.75  # 🔧 było 0.80


def get_h2_similarity_threshold() -> float:
    return 0.45  # 🔧 było 0.50
