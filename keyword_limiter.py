import re
from typing import List, Dict, Tuple


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
    
    return {
        "valid": count <= 2,
        "count": count,
        "max": 2,
        "warning": f"Za dużo nagłówków z '{main_keyword}' ({count}/2). Użyj synonimów!" if count > 2 else None
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
    
    return {
        "valid": count <= 5,
        "count": count,
        "max": 5,
        "warning": f"Keyword w {count} nagłówkach (max 5). Ogranicz!" if count > 5 else None
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
        has_enough_words = len(h2.split()) >= 4
        
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
    
    is_valid = overlap_ratio >= 0.6
    
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
    
    no_topic_drift = title_h1_overlap >= 0.4
    
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


def check_keyword_density(text: str, main_keyword: str, max_density: float = 0.015) -> Dict:
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


def get_h1_similarity_threshold() -> float:
    return 0.80


def get_h2_similarity_threshold() -> float:
    return 0.50
