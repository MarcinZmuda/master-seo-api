"""
===============================================================================
🧠 SEMANTIC ANALYZER v23.9
===============================================================================
Wrapper dla funkcji semantycznych.
Główna implementacja jest w text_analyzer.py - ten moduł reeksportuje dla
zachowania kompatybilności wstecznej.

Użycie:
    from semantic_analyzer import analyze_semantic_coverage, semantic_validation
    
lub bezpośrednio:
    from text_analyzer import analyze_semantic_coverage
===============================================================================
"""

import re
from typing import List, Dict

# ================================================================
# v23.9: Reeksport z text_analyzer (jedna implementacja)
# ================================================================
try:
    from text_analyzer import (
        analyze_semantic_coverage,
        semantic_similarity,
        split_sentences,
        get_embeddings,
        clear_caches
    )
    SEMANTIC_ENABLED = True
    print("[SEMANTIC_ANALYZER] ✅ Używam implementacji z text_analyzer")
except ImportError as e:
    print(f"[SEMANTIC_ANALYZER] ⚠️ text_analyzer niedostępny: {e}")
    SEMANTIC_ENABLED = False
    
    # Fallback - pusta implementacja
    def analyze_semantic_coverage(text: str, keywords: List[str], **kwargs) -> Dict:
        return {"semantic_enabled": False, "error": "text_analyzer not available"}
    
    def semantic_similarity(text1: str, text2: str) -> float:
        return 0.0
    
    def split_sentences(text: str) -> List[str]:
        return []
    
    def get_embeddings(texts: List[str], use_cache: bool = True):
        return None
    
    def clear_caches():
        pass


def split_into_sentences(text: str) -> List[str]:
    """Alias dla split_sentences z text_analyzer."""
    try:
        return split_sentences(text)
    except:
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        return [s.strip() for s in sentences if len(s.split()) >= 5]


def sentence_keyword_similarity(sentence: str, keyword: str) -> float:
    """Similarity pojedynczego zdania z keywordem."""
    try:
        return semantic_similarity(sentence, keyword)
    except:
        return 0.0


def find_semantic_gaps(text: str, keywords: List[str], threshold: float = 0.50) -> List[Dict]:
    """Znajduje luki semantyczne w tekście."""
    result = analyze_semantic_coverage(text, keywords)
    
    if not result.get("enabled", result.get("semantic_enabled", False)):
        return []
    
    gaps = []
    for keyword, data in result.get("keywords", {}).items():
        if data.get("status") in ["GAP", "WEAK"]:
            gaps.append({
                "keyword": keyword,
                "status": data.get("status"),
                "best_similarity": data.get("best_similarity", 0)
            })
    
    return sorted(gaps, key=lambda x: x.get("best_similarity", 0))


def count_semantic_occurrences(text: str, keyword: str, threshold: float = 0.60) -> int:
    """Liczy semantyczne wystąpienia keywordu."""
    sentences = split_into_sentences(text)
    if not sentences:
        return 0
    
    count = 0
    for sent in sentences:
        sim = sentence_keyword_similarity(sent, keyword)
        if sim >= threshold:
            count += 1
    
    return count


def semantic_validation(text: str, keywords_state: Dict, min_coverage: float = 0.4) -> Dict:
    """Walidacja semantyczna tekstu względem keywords."""
    keywords = [
        meta.get("keyword", "") 
        for meta in keywords_state.values() 
        if meta.get("keyword")
    ]
    
    if not keywords:
        return {"valid": True, "semantic_enabled": False}
    
    result = analyze_semantic_coverage(text, keywords)
    
    if not result.get("enabled", result.get("semantic_enabled", False)):
        return {"valid": True, "semantic_enabled": False}
    
    # Pobierz coverage - różne nazwy pól w różnych wersjach
    overall = result.get("overall_coverage", result.get("coverage", 0))
    gaps = result.get("gaps", [])
    
    return {
        "valid": overall >= min_coverage,
        "semantic_enabled": True,
        "overall_coverage": overall,
        "min_required": min_coverage,
        "gaps": gaps,
        "gap_count": len(gaps),
        "summary": result.get("summary", {}),
        "warning": f"Pokrycie {overall:.0%} < {min_coverage:.0%}" if overall < min_coverage else None
    }
