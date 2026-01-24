"""
🆕 v36.8: KEYWORD DENSITY - Kontrola gęstości słów kluczowych

Implementuje:
- Global keyword density % (max 2-3% dla pojedynczego słowa)
- Per-section density
- Density decay recommendations
- Integration z soft caps

Autor: Claude
Wersja: 36.8
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# ================================================================
# KONFIGURACJA
# ================================================================

@dataclass
class DensityConfig:
    """Konfiguracja dla keyword density."""
    
    # Globalne limity (% tekstu)
    MAX_SINGLE_KEYWORD_DENSITY: float = 2.5  # Max 2.5% dla jednego keyword
    MAX_MAIN_KEYWORD_DENSITY: float = 3.0    # Main keyword może mieć więcej
    MAX_TOTAL_KEYWORDS_DENSITY: float = 8.0  # Suma wszystkich keywords
    
    # Optymalne wartości
    OPTIMAL_MAIN_DENSITY_MIN: float = 1.0    # Min dla main keyword
    OPTIMAL_MAIN_DENSITY_MAX: float = 2.0    # Optymalny max
    OPTIMAL_KEYWORD_DENSITY: float = 0.5     # Optymalne dla innych keywords
    
    # Per-paragraph limity
    MAX_KEYWORD_PER_100_WORDS: int = 3       # Max 3 wystąpienia na 100 słów
    
    # Progi ostrzeżeń
    WARNING_THRESHOLD: float = 0.8  # 80% limitu = warning
    CRITICAL_THRESHOLD: float = 1.0  # 100% = critical

CONFIG = DensityConfig()

# ================================================================
# DENSITY CALCULATION
# ================================================================

@dataclass
class DensityResult:
    """Wynik analizy density."""
    keyword: str
    count: int
    word_count: int
    density_percent: float
    limit_percent: float
    status: str  # OK, WARNING, EXCEEDED
    message: str
    usage_ratio: float  # count / limit

def count_keyword_occurrences(text: str, keyword: str, use_lemma: bool = True) -> int:
    """
    Liczy wystąpienia keyword w tekście.
    
    Args:
        text: Tekst do analizy
        keyword: Słowo kluczowe
        use_lemma: Czy używać lematyzacji (jeśli dostępna)
        
    Returns:
        Liczba wystąpień
    """
    if not text or not keyword:
        return 0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Podstawowe liczenie (case-insensitive)
    count = 0
    
    # Metoda 1: Regex word boundary
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    matches = re.findall(pattern, text_lower)
    count = len(matches)
    
    # Metoda 2: Jeśli mamy lematyzator, użyj go
    if use_lemma:
        try:
            from polish_lemmatizer import lemmatize_text, get_phrase_lemmas
            
            text_lemmas = lemmatize_text(text)
            keyword_lemmas = get_phrase_lemmas(keyword)
            
            if keyword_lemmas and text_lemmas:
                # Szukaj sekwencji lemm
                kw_len = len(keyword_lemmas)
                for i in range(len(text_lemmas) - kw_len + 1):
                    if text_lemmas[i:i+kw_len] == keyword_lemmas:
                        count = max(count, count)  # Użyj wyższej wartości
                        
        except ImportError:
            pass  # Brak lematyzatora - użyj podstawowego liczenia
    
    return count

def calculate_keyword_density(
    text: str,
    keyword: str,
    is_main: bool = False,
    use_lemma: bool = True
) -> DensityResult:
    """
    Oblicza density dla pojedynczego keyword.
    
    Args:
        text: Tekst do analizy
        keyword: Słowo kluczowe
        is_main: Czy to główne słowo kluczowe
        use_lemma: Czy używać lematyzacji
        
    Returns:
        DensityResult z pełną analizą
    """
    if not text:
        return DensityResult(
            keyword=keyword,
            count=0,
            word_count=0,
            density_percent=0.0,
            limit_percent=CONFIG.MAX_SINGLE_KEYWORD_DENSITY,
            status="OK",
            message="Brak tekstu",
            usage_ratio=0.0
        )
    
    word_count = len(text.split())
    count = count_keyword_occurrences(text, keyword, use_lemma)
    
    # Oblicz density %
    if word_count > 0:
        density = (count / word_count) * 100
    else:
        density = 0.0
    
    # Określ limit
    limit = CONFIG.MAX_MAIN_KEYWORD_DENSITY if is_main else CONFIG.MAX_SINGLE_KEYWORD_DENSITY
    
    # Określ status
    usage_ratio = density / limit if limit > 0 else 0
    
    if usage_ratio >= CONFIG.CRITICAL_THRESHOLD:
        status = "EXCEEDED"
        message = f"⚠️ PRZEKROCZONY LIMIT: {density:.2f}% (max {limit}%)"
    elif usage_ratio >= CONFIG.WARNING_THRESHOLD:
        status = "WARNING"
        message = f"⚠️ Blisko limitu: {density:.2f}% (max {limit}%)"
    else:
        status = "OK"
        message = f"✅ OK: {density:.2f}% (max {limit}%)"
    
    return DensityResult(
        keyword=keyword,
        count=count,
        word_count=word_count,
        density_percent=round(density, 3),
        limit_percent=limit,
        status=status,
        message=message,
        usage_ratio=round(usage_ratio, 3)
    )

def analyze_all_keywords_density(
    text: str,
    keywords_state: Dict[str, Any],
    main_keyword: str
) -> Dict[str, Any]:
    """
    Analizuje density dla wszystkich keywords.
    
    Args:
        text: Tekst do analizy
        keywords_state: Stan keywords z projektu
        main_keyword: Główne słowo kluczowe
        
    Returns:
        Pełna analiza density
    """
    results = {}
    total_keyword_words = 0
    word_count = len(text.split()) if text else 0
    
    exceeded = []
    warnings = []
    
    main_keyword_lower = main_keyword.lower()
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        is_main = (keyword.lower() == main_keyword_lower or 
                   meta.get("is_main_keyword", False))
        
        result = calculate_keyword_density(text, keyword, is_main)
        results[keyword] = {
            "count": result.count,
            "density_percent": result.density_percent,
            "limit_percent": result.limit_percent,
            "status": result.status,
            "message": result.message,
            "usage_ratio": result.usage_ratio
        }
        
        if result.status == "EXCEEDED":
            exceeded.append(keyword)
        elif result.status == "WARNING":
            warnings.append(keyword)
        
        # Policz total keyword words (keyword_length * count)
        total_keyword_words += len(keyword.split()) * result.count
    
    # Total keywords density
    total_density = (total_keyword_words / word_count * 100) if word_count > 0 else 0
    
    return {
        "word_count": word_count,
        "keywords_analyzed": len(results),
        "total_keywords_density": round(total_density, 2),
        "total_limit": CONFIG.MAX_TOTAL_KEYWORDS_DENSITY,
        "total_status": "EXCEEDED" if total_density > CONFIG.MAX_TOTAL_KEYWORDS_DENSITY else "OK",
        "exceeded_keywords": exceeded,
        "warning_keywords": warnings,
        "per_keyword": results,
        "summary": {
            "ok_count": len([r for r in results.values() if r["status"] == "OK"]),
            "warning_count": len(warnings),
            "exceeded_count": len(exceeded)
        }
    }

# ================================================================
# DENSITY-BASED SUGGESTIONS
# ================================================================

def get_density_based_suggestion(
    keyword: str,
    current_density: float,
    remaining_batches: int,
    is_main: bool = False,
    target_min: int = 1,
    target_max: int = 5,
    actual_uses: int = 0
) -> Dict[str, Any]:
    """
    Oblicza sugerowaną liczbę użyć na podstawie density.
    
    Args:
        keyword: Słowo kluczowe
        current_density: Aktualna density %
        remaining_batches: Ile batchy zostało
        is_main: Czy główne słowo
        target_min/max: Cele
        actual_uses: Aktualne użycia
        
    Returns:
        Sugestia użycia
    """
    limit = CONFIG.MAX_MAIN_KEYWORD_DENSITY if is_main else CONFIG.MAX_SINGLE_KEYWORD_DENSITY
    
    # Ile miejsca zostało do limitu
    density_headroom = limit - current_density
    
    # Pozostałe użycia do target_max
    remaining_to_max = max(0, target_max - actual_uses)
    remaining_to_min = max(0, target_min - actual_uses)
    
    if density_headroom <= 0:
        # Limit osiągnięty
        return {
            "suggested": 0,
            "hard_max": 0,
            "reason": "density_limit_reached",
            "message": f"❌ Limit density osiągnięty ({current_density:.2f}% >= {limit}%)"
        }
    
    if density_headroom < 0.5:
        # Blisko limitu - max 1
        suggested = min(1, remaining_to_max)
        return {
            "suggested": suggested,
            "hard_max": suggested,
            "reason": "density_near_limit",
            "message": f"⚠️ Blisko limitu - max {suggested}x"
        }
    
    # Normalny przypadek
    if remaining_batches > 0:
        per_batch = max(1, remaining_to_max // remaining_batches)
    else:
        per_batch = remaining_to_max
    
    return {
        "suggested": min(per_batch, remaining_to_max),
        "hard_max": remaining_to_max,
        "reason": "normal",
        "message": f"✅ Sugerowane: {per_batch}x (density: {current_density:.2f}%)"
    }

# ================================================================
# DECAY MECHANISM FOR UNIVERSAL KEYWORDS
# ================================================================

def calculate_universal_decay(
    keyword: str,
    batch_number: int,
    total_batches: int,
    base_suggested: int = 2,
    decay_rate: float = 0.15
) -> Dict[str, Any]:
    """
    Oblicza decay dla universal keywords.
    
    Universal keywords powinny być używane częściej na początku,
    rzadziej pod koniec (naturalny rozkład).
    
    Args:
        keyword: Słowo kluczowe
        batch_number: Numer aktualnego batcha (1-indexed)
        total_batches: Łączna liczba batchy
        base_suggested: Bazowa sugestia dla batch 1
        decay_rate: Współczynnik decay per batch (0.15 = 15% mniej per batch)
        
    Returns:
        Sugestia z decay
    """
    # Decay formula: suggested = base * (1 - decay_rate)^(batch - 1)
    decay_factor = (1 - decay_rate) ** (batch_number - 1)
    decayed = base_suggested * decay_factor
    
    # Zaokrąglij, ale minimum 1 dla pierwszych 2/3 batchy
    suggested = max(1, round(decayed))
    
    # W ostatnich batchach może być 0
    if batch_number > total_batches * 0.7:
        suggested = max(0, round(decayed))
    
    return {
        "suggested": suggested,
        "decay_factor": round(decay_factor, 3),
        "original": base_suggested,
        "batch": batch_number,
        "message": f"Decay {batch_number}/{total_batches}: {base_suggested} → {suggested}"
    }

# ================================================================
# INTEGRATION FUNCTIONS
# ================================================================

def validate_batch_density(
    batch_text: str,
    keywords_state: Dict[str, Any],
    main_keyword: str,
    batch_number: int
) -> Dict[str, Any]:
    """
    Waliduje density dla batcha przed zapisem.
    
    Args:
        batch_text: Tekst batcha
        keywords_state: Stan keywords
        main_keyword: Main keyword
        batch_number: Numer batcha
        
    Returns:
        Wynik walidacji z ewentualnymi ostrzeżeniami
    """
    analysis = analyze_all_keywords_density(batch_text, keywords_state, main_keyword)
    
    issues = []
    
    # Sprawdź przekroczone limity
    if analysis["exceeded_keywords"]:
        for kw in analysis["exceeded_keywords"]:
            info = analysis["per_keyword"].get(kw, {})
            issues.append({
                "type": "DENSITY_EXCEEDED",
                "keyword": kw,
                "density": info.get("density_percent", 0),
                "limit": info.get("limit_percent", 0),
                "severity": "HIGH"
            })
    
    # Sprawdź ostrzeżenia
    if analysis["warning_keywords"]:
        for kw in analysis["warning_keywords"]:
            info = analysis["per_keyword"].get(kw, {})
            issues.append({
                "type": "DENSITY_WARNING",
                "keyword": kw,
                "density": info.get("density_percent", 0),
                "limit": info.get("limit_percent", 0),
                "severity": "MEDIUM"
            })
    
    # Sprawdź total density
    if analysis["total_status"] == "EXCEEDED":
        issues.append({
            "type": "TOTAL_DENSITY_EXCEEDED",
            "density": analysis["total_keywords_density"],
            "limit": analysis["total_limit"],
            "severity": "HIGH"
        })
    
    return {
        "valid": len([i for i in issues if i["severity"] == "HIGH"]) == 0,
        "issues": issues,
        "analysis": analysis,
        "batch_number": batch_number
    }

# ================================================================
# TESTING
# ================================================================

def test_keyword_density():
    """Test keyword density functions."""
    print("="*60)
    print("KEYWORD DENSITY TEST")
    print("="*60)
    
    test_text = """
    Ubezwłasnowolnienie to instytucja prawna stosowana przez sąd.
    Wniosek o ubezwłasnowolnienie składa się do sądu okręgowego.
    Ubezwłasnowolnienie może być całkowite lub częściowe.
    Sąd rozpatruje każdy wniosek indywidualnie.
    Ubezwłasnowolnienie wymaga opinii biegłego psychiatry.
    """
    
    print(f"\nTekst: {len(test_text.split())} słów")
    
    # Test 1: Single keyword
    print("\n1. Single keyword density:")
    result = calculate_keyword_density(test_text, "ubezwłasnowolnienie", is_main=True)
    print(f"   '{result.keyword}': {result.count}x, {result.density_percent}%, status={result.status}")
    
    # Test 2: All keywords
    print("\n2. All keywords density:")
    keywords_state = {
        "1": {"keyword": "ubezwłasnowolnienie"},
        "2": {"keyword": "sąd"},
        "3": {"keyword": "wniosek"}
    }
    
    analysis = analyze_all_keywords_density(test_text, keywords_state, "ubezwłasnowolnienie")
    print(f"   Total density: {analysis['total_keywords_density']}%")
    print(f"   Exceeded: {analysis['exceeded_keywords']}")
    print(f"   Warnings: {analysis['warning_keywords']}")
    
    # Test 3: Decay
    print("\n3. Universal keyword decay:")
    for batch in range(1, 6):
        decay = calculate_universal_decay("ubezwłasnowolnienie", batch, 5, base_suggested=3)
        print(f"   Batch {batch}: {decay['suggested']}x (factor: {decay['decay_factor']})")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_keyword_density()
