"""
===============================================================================
BATCH BEST-OF-N SELECTOR v26.1
===============================================================================
Generuje N wersji batcha i wybiera najlepszą na podstawie scoringu.

Logika:
1. GPT generuje batch 3 razy (różne temperature/prompty)
2. Każda wersja jest walidowana przez unified_prevalidation
3. Wybierana jest wersja z najwyższym score
4. Jeśli żadna nie spełnia minimum - zwracana jest najlepsza z ostrzeżeniem

Autor: BRAJEN SEO Engine v26.1
===============================================================================
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Walidacja
try:
    from seo_optimizer import unified_prevalidation, calculate_keyword_density
    from unified_validator import validate_content
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class BestOfNConfig:
    """Konfiguracja dla Best-of-N selection."""
    n_candidates: int = 3                    # Ile wersji generować
    min_acceptable_score: float = 60.0       # Minimalny akceptowalny score
    temperature_base: float = 0.7            # Bazowa temperatura
    temperature_variance: float = 0.15       # Wariancja temperatury między wersjami
    timeout_per_generation: int = 60         # Timeout na generację (sekundy)
    
    # Wagi dla scoringu
    weights: Dict[str, float] = field(default_factory=lambda: {
        "density": 0.20,           # Density w zakresie 0.5-1.5%
        "coverage": 0.25,          # Pokrycie keywords
        "polish_quality": 0.15,    # Jakość języka polskiego
        "structure": 0.15,         # Struktura (długość akapitów, H3)
        "keywords_natural": 0.15,  # Naturalność wplecenia fraz
        "no_banned_phrases": 0.10  # Brak zakazanych fraz
    })


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_candidate_score(
    content: str,
    keywords_state: Dict,
    main_keyword: str,
    validation_result: Dict
) -> Dict[str, Any]:
    """
    Oblicza szczegółowy score dla kandydata.
    
    Returns:
        Dict z:
        - total_score: float (0-100)
        - component_scores: Dict z poszczególnymi scorami
        - issues: List problemów
        - warnings: List ostrzeżeń
    """
    scores = {}
    issues = []
    warnings = []
    
    # 1. DENSITY SCORE (0-100)
    density = validation_result.get("density", {}).get("value", 0)
    if 0.5 <= density <= 1.5:
        scores["density"] = 100
    elif 0.3 <= density < 0.5 or 1.5 < density <= 2.0:
        scores["density"] = 80
    elif 0.2 <= density < 0.3 or 2.0 < density <= 2.5:
        scores["density"] = 60
        warnings.append(f"Density {density:.1f}% - lekko poza zakresem")
    elif density > 2.5:
        scores["density"] = 30
        issues.append(f"Density {density:.1f}% - za wysokie!")
    else:
        scores["density"] = 50
        warnings.append(f"Density {density:.1f}% - za niskie")
    
    # 2. COVERAGE SCORE (0-100)
    coverage_data = validation_result.get("coverage", {})
    basic_coverage = coverage_data.get("basic_percent", 0)
    extended_coverage = coverage_data.get("extended_percent", 0)
    
    # Średnia ważona (BASIC ważniejsze)
    coverage_score = (basic_coverage * 0.7 + extended_coverage * 0.3)
    scores["coverage"] = coverage_score
    
    if basic_coverage < 100:
        missing = coverage_data.get("basic_missing", [])
        if missing:
            issues.append(f"Brak BASIC: {', '.join(missing[:3])}")
    
    # 3. POLISH QUALITY SCORE (0-100)
    polish_data = validation_result.get("polish_quality", {})
    scores["polish_quality"] = polish_data.get("score", 70)
    
    if polish_data.get("issues"):
        for issue in polish_data.get("issues", [])[:2]:
            warnings.append(f"Polski: {issue.get('message', '')[:50]}")
    
    # 4. STRUCTURE SCORE (0-100)
    structure_score = 100
    
    # Sprawdź długość akapitów
    paragraphs = content.split('\n\n')
    for p in paragraphs:
        words = len(p.split())
        if words > 180:
            structure_score -= 10
            warnings.append(f"Akapit za długi ({words} słów)")
        elif words < 30 and words > 5:
            structure_score -= 5
    
    scores["structure"] = max(0, structure_score)
    
    # 5. KEYWORDS NATURAL SCORE (0-100)
    # Sprawdzamy czy frazy nie są upchane w jednym miejscu
    natural_score = 100
    
    # Prosta heurystyka: czy frazy są rozłożone równomiernie
    text_parts = content.split('\n\n')
    if len(text_parts) > 1:
        keyword_positions = []
        for i, part in enumerate(text_parts):
            part_lower = part.lower()
            if main_keyword.lower() in part_lower:
                keyword_positions.append(i)
        
        if keyword_positions:
            # Sprawdź czy nie wszystkie w jednym miejscu
            if len(set(keyword_positions)) == 1 and len(keyword_positions) > 2:
                natural_score -= 20
                warnings.append("Frazy skupione w jednym akapicie")
    
    scores["keywords_natural"] = natural_score
    
    # 6. NO BANNED PHRASES SCORE (0-100)
    banned_count = len(polish_data.get("banned_phrases_found", []))
    if banned_count == 0:
        scores["no_banned_phrases"] = 100
    elif banned_count <= 2:
        scores["no_banned_phrases"] = 70
        warnings.append(f"Znaleziono {banned_count} zakazanych fraz")
    else:
        scores["no_banned_phrases"] = 40
        issues.append(f"Za dużo zakazanych fraz ({banned_count})")
    
    # TOTAL SCORE (średnia ważona)
    config = BestOfNConfig()
    total = 0
    for component, weight in config.weights.items():
        total += scores.get(component, 50) * weight
    
    return {
        "total_score": round(total, 1),
        "component_scores": scores,
        "issues": issues,
        "warnings": warnings,
        "density": density,
        "coverage": {
            "basic": basic_coverage,
            "extended": extended_coverage
        }
    }


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def generate_candidate(
    prompt: str,
    temperature: float,
    model_name: str = "gemini-2.0-flash"
) -> Optional[str]:
    """
    Generuje pojedynczego kandydata przez Gemini.
    """
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096
            )
        )
        
        if response and response.text:
            return response.text.strip()
        return None
        
    except Exception as e:
        print(f"[BEST-OF-N] Generation error: {e}")
        return None


def generate_n_candidates(
    base_prompt: str,
    n: int = 3,
    model_name: str = "gemini-2.0-flash"
) -> List[Dict[str, Any]]:
    """
    Generuje N kandydatów z różnymi temperaturami.
    
    Returns:
        List[Dict] z:
        - content: str
        - temperature: float
        - generation_time: float
    """
    config = BestOfNConfig()
    candidates = []
    
    temperatures = [
        config.temperature_base - config.temperature_variance,  # 0.55 - bardziej deterministyczny
        config.temperature_base,                                  # 0.70 - bazowy
        config.temperature_base + config.temperature_variance,  # 0.85 - bardziej kreatywny
    ]
    
    for i in range(n):
        temp = temperatures[i] if i < len(temperatures) else config.temperature_base
        
        # Lekka modyfikacja promptu dla różnorodności
        modified_prompt = base_prompt
        if i == 1:
            modified_prompt += "\n\nZadbaj szczególnie o naturalność języka i płynność tekstu."
        elif i == 2:
            modified_prompt += "\n\nSkup się na konkretach, liczbach i praktycznych przykładach."
        
        start_time = datetime.now()
        content = generate_candidate(modified_prompt, temp, model_name)
        gen_time = (datetime.now() - start_time).total_seconds()
        
        if content:
            candidates.append({
                "content": content,
                "temperature": temp,
                "generation_time": gen_time,
                "variant": i + 1
            })
    
    return candidates


# ============================================================================
# MAIN: SELECT BEST CANDIDATE
# ============================================================================

def select_best_batch(
    base_prompt: str,
    keywords_state: Dict,
    main_keyword: str,
    n_candidates: int = 3,
    model_name: str = "gemini-2.0-flash",
    validation_mode: str = "batch"
) -> Dict[str, Any]:
    """
    Główna funkcja: generuje N wersji batcha i wybiera najlepszą.
    
    Args:
        base_prompt: Prompt do generowania batcha
        keywords_state: Stan słów kluczowych
        main_keyword: Fraza główna
        n_candidates: Ile wersji generować (default 3)
        model_name: Model Gemini do użycia
        validation_mode: Tryb walidacji ('batch' lub 'final')
    
    Returns:
        Dict z:
        - selected_content: str - wybrana treść
        - selected_score: float - score wybranej wersji
        - all_candidates: List - wszystkie kandydaci ze scorami
        - selection_reason: str - dlaczego wybrano tę wersję
        - meets_minimum: bool - czy spełnia minimum
    """
    config = BestOfNConfig()
    
    # 1. Generuj kandydatów
    print(f"[BEST-OF-N] Generating {n_candidates} candidates...")
    candidates = generate_n_candidates(base_prompt, n_candidates, model_name)
    
    if not candidates:
        return {
            "error": "Nie udało się wygenerować żadnego kandydata",
            "selected_content": None,
            "selected_score": 0,
            "meets_minimum": False
        }
    
    # 2. Waliduj i scoruj każdego kandydata
    scored_candidates = []
    
    for candidate in candidates:
        content = candidate["content"]
        
        # Walidacja przez unified_prevalidation
        if VALIDATOR_AVAILABLE:
            validation = unified_prevalidation(
                text=content,
                keywords_state=keywords_state,
                main_keyword=main_keyword
            )
        else:
            # Fallback - podstawowa walidacja
            validation = {
                "density": {"value": 1.0},
                "coverage": {"basic_percent": 80, "extended_percent": 80},
                "polish_quality": {"score": 70}
            }
        
        # Oblicz score
        score_data = calculate_candidate_score(
            content=content,
            keywords_state=keywords_state,
            main_keyword=main_keyword,
            validation_result=validation
        )
        
        scored_candidates.append({
            **candidate,
            **score_data,
            "validation": validation
        })
        
        print(f"[BEST-OF-N] Candidate {candidate['variant']}: "
              f"score={score_data['total_score']:.1f}, "
              f"density={score_data['density']:.1f}%")
    
    # 3. Sortuj po score (malejąco)
    scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    
    # 4. Wybierz najlepszego
    best = scored_candidates[0]
    
    meets_minimum = best["total_score"] >= config.min_acceptable_score
    
    # Przygotuj reason
    if meets_minimum:
        selection_reason = f"Wybrano wariant {best['variant']} (score: {best['total_score']:.1f}/100)"
    else:
        selection_reason = (
            f"UWAGA: Żaden wariant nie osiągnął minimum {config.min_acceptable_score}. "
            f"Wybrano najlepszy: wariant {best['variant']} (score: {best['total_score']:.1f}/100)"
        )
    
    # 5. Zwróć wynik
    return {
        "selected_content": best["content"],
        "selected_score": best["total_score"],
        "selected_variant": best["variant"],
        "all_candidates": [
            {
                "variant": c["variant"],
                "score": c["total_score"],
                "density": c["density"],
                "issues": c["issues"],
                "warnings": c["warnings"]
            }
            for c in scored_candidates
        ],
        "selection_reason": selection_reason,
        "meets_minimum": meets_minimum,
        "component_scores": best["component_scores"],
        "issues": best["issues"],
        "warnings": best["warnings"]
    }


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def batch_with_best_of_n(
    project_id: str,
    batch_content_prompt: str,
    keywords_state: Dict,
    main_keyword: str,
    use_best_of_n: bool = True,
    n_candidates: int = 3
) -> Dict[str, Any]:
    """
    Helper do integracji z istniejącym workflow.
    
    Jeśli use_best_of_n=True, generuje N wersji i wybiera najlepszą.
    Jeśli False, działa jak dotychczas (1 generacja).
    """
    if not use_best_of_n or n_candidates <= 1:
        # Standardowa ścieżka - 1 generacja
        content = generate_candidate(batch_content_prompt, 0.7)
        return {
            "content": content,
            "method": "single",
            "score": None
        }
    
    # Best-of-N
    result = select_best_batch(
        base_prompt=batch_content_prompt,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        n_candidates=n_candidates
    )
    
    return {
        "content": result.get("selected_content"),
        "method": "best_of_n",
        "score": result.get("selected_score"),
        "details": result
    }


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    # Test
    test_prompt = """
    Napisz sekcję H2 o temacie "Dokumenty potrzebne do rozwodu".
    Użyj fraz: pozew o rozwód, dokumenty do rozwodu, akt małżeństwa.
    Długość: 200-300 słów.
    """
    
    test_keywords = {
        "1": {"keyword": "pozew o rozwód", "type": "BASIC", "actual_uses": 0, "target_min": 1, "target_max": 3},
        "2": {"keyword": "dokumenty do rozwodu", "type": "BASIC", "actual_uses": 0, "target_min": 1, "target_max": 2},
        "3": {"keyword": "akt małżeństwa", "type": "EXTENDED", "actual_uses": 0, "target_min": 1, "target_max": 1}
    }
    
    result = select_best_batch(
        base_prompt=test_prompt,
        keywords_state=test_keywords,
        main_keyword="pozew o rozwód",
        n_candidates=3
    )
    
    print("\n" + "="*60)
    print("WYNIK BEST-OF-N:")
    print("="*60)
    print(f"Wybrano: wariant {result.get('selected_variant')}")
    print(f"Score: {result.get('selected_score')}")
    print(f"Reason: {result.get('selection_reason')}")
    print(f"\nWszyscy kandydaci:")
    for c in result.get("all_candidates", []):
        print(f"  - Wariant {c['variant']}: {c['score']:.1f} (density: {c['density']:.1f}%)")
