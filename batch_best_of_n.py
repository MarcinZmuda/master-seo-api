"""
===============================================================================
BATCH BEST-OF-N SELECTOR v26.2 - OPTIMIZED
===============================================================================
ZMIANY OPTYMALIZACYJNE:
- 🆕 Obniżony min_acceptable_score z 60 do 50
- 🆕 Szybsza generacja (2 kandydatów domyślnie zamiast 3)
- 🆕 Lepsze wagi scoringu (coverage ważniejsze)
- 🆕 Timeout skrócony

EFEKT: Szybsze Best-of-N z lepszą selekcją
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
# KONFIGURACJA - 🔧 v26.2 OPTIMIZED
# ============================================================================

@dataclass
class BestOfNConfig:
    """
    🔧 v26.2 OPTIMIZED: Zmienione parametry.
    """
    n_candidates: int = 2              # 🔧 było 3 - szybciej!
    min_acceptable_score: float = 50.0  # 🔧 było 60 - mniej restrykcyjne
    temperature_base: float = 0.7
    temperature_variance: float = 0.20  # 🔧 było 0.15 - więcej różnorodności
    timeout_per_generation: int = 45    # 🔧 było 60
    
    # 🔧 v26.2: Zmienione wagi - coverage ważniejsze
    weights: Dict[str, float] = field(default_factory=lambda: {
        "density": 0.15,            # 🔧 było 0.20
        "coverage": 0.35,           # 🔧 było 0.25 - ZWIĘKSZONE
        "polish_quality": 0.15,
        "structure": 0.10,          # 🔧 było 0.15
        "keywords_natural": 0.15,
        "no_banned_phrases": 0.10
    })


# ============================================================================
# SCORING FUNCTIONS - 🔧 v26.2 OPTIMIZED
# ============================================================================

def calculate_candidate_score(
    content: str,
    keywords_state: Dict,
    main_keyword: str,
    validation_result: Dict
) -> Dict[str, Any]:
    """
    🔧 v26.2 OPTIMIZED: Bardziej elastyczne scoring.
    """
    scores = {}
    issues = []
    warnings = []
    
    # 1. DENSITY SCORE - 🔧 szersze zakresy
    density = validation_result.get("density", {}).get("value", 0)
    if 0.4 <= density <= 2.0:  # 🔧 było 0.5-1.5
        scores["density"] = 100
    elif 0.2 <= density < 0.4 or 2.0 < density <= 2.5:  # 🔧 szerszy zakres
        scores["density"] = 85
    elif 0.1 <= density < 0.2 or 2.5 < density <= 3.0:
        scores["density"] = 65
        warnings.append(f"Density {density:.1f}% - poza zakresem")
    elif density > 3.0:
        scores["density"] = 35
        issues.append(f"Density {density:.1f}% - za wysokie!")
    else:
        scores["density"] = 55
        warnings.append(f"Density {density:.1f}% - za niskie")
    
    # 2. COVERAGE SCORE - 🔧 NAJWAŻNIEJSZE
    coverage_data = validation_result.get("coverage", {})
    basic_coverage = coverage_data.get("basic_percent", 0)
    extended_coverage = coverage_data.get("extended_percent", 0)
    
    # 🔧 v26.2: Bardziej złożona formuła
    if basic_coverage >= 90:
        coverage_score = 90 + (extended_coverage * 0.1)
    elif basic_coverage >= 70:
        coverage_score = 70 + (basic_coverage - 70) + (extended_coverage * 0.1)
    else:
        coverage_score = basic_coverage * 0.9 + extended_coverage * 0.1
    
    scores["coverage"] = min(100, coverage_score)
    
    if basic_coverage < 100:
        missing = coverage_data.get("basic_missing", [])
        if missing:
            warnings.append(f"Brak BASIC: {', '.join(missing[:2])}")  # 🔧 było 3
    
    # 3. POLISH QUALITY SCORE
    polish_data = validation_result.get("polish_quality", {})
    scores["polish_quality"] = polish_data.get("score", 75)  # 🔧 było 70
    
    if polish_data.get("issues"):
        for issue in polish_data.get("issues", [])[:1]:  # 🔧 było 2
            warnings.append(f"Polski: {issue.get('message', '')[:40]}")
    
    # 4. STRUCTURE SCORE - 🔧 bardziej elastyczne
    structure_score = 100
    
    paragraphs = content.split('\n\n')
    long_para_count = 0
    for p in paragraphs:
        words = len(p.split())
        if words > 200:  # 🔧 było 180
            long_para_count += 1
            if long_para_count > 1:  # 🔧 Tylko jeśli > 1 długi akapit
                structure_score -= 8
    
    scores["structure"] = max(60, structure_score)  # 🔧 było 0
    
    # 5. KEYWORDS NATURAL SCORE
    natural_score = 100
    
    text_parts = content.split('\n\n')
    if len(text_parts) > 1 and main_keyword:
        keyword_positions = []
        for i, part in enumerate(text_parts):
            part_lower = part.lower()
            if main_keyword.lower() in part_lower:
                keyword_positions.append(i)
        
        if keyword_positions:
            if len(set(keyword_positions)) == 1 and len(keyword_positions) > 3:  # 🔧 było 2
                natural_score -= 15
    
    scores["keywords_natural"] = natural_score
    
    # 6. NO BANNED PHRASES SCORE
    banned_count = len(polish_data.get("banned_phrases_found", []))
    if banned_count == 0:
        scores["no_banned_phrases"] = 100
    elif banned_count <= 3:  # 🔧 było 2
        scores["no_banned_phrases"] = 75
    else:
        scores["no_banned_phrases"] = 45
        issues.append(f"Za dużo zakazanych fraz ({banned_count})")
    
    # TOTAL SCORE
    config = BestOfNConfig()
    total = 0
    for component, weight in config.weights.items():
        total += scores.get(component, 60) * weight  # 🔧 default 60 zamiast 50
    
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
    """Generuje pojedynczego kandydata przez Gemini."""
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
    n: int = 2,  # 🔧 było 3
    model_name: str = "gemini-2.0-flash"
) -> List[Dict[str, Any]]:
    """
    🔧 v26.2 OPTIMIZED: Generuje N kandydatów szybciej.
    """
    config = BestOfNConfig()
    candidates = []
    
    # 🔧 v26.2: Tylko 2 temperatury domyślnie
    temperatures = [
        config.temperature_base - config.temperature_variance,  # 0.50
        config.temperature_base + config.temperature_variance,  # 0.90
    ]
    
    # Dodaj trzecią jeśli n > 2
    if n > 2:
        temperatures.insert(1, config.temperature_base)  # 0.70
    
    for i in range(n):
        temp = temperatures[i] if i < len(temperatures) else config.temperature_base
        
        # 🔧 v26.2: Krótsze modyfikacje promptu
        modified_prompt = base_prompt
        if i == 1:
            modified_prompt += "\n\nPisz naturalnie, unikaj sztywnych konstrukcji."
        
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
            print(f"[BEST-OF-N] Generated variant {i+1} in {gen_time:.1f}s (temp={temp:.2f})")
    
    return candidates


# ============================================================================
# MAIN: SELECT BEST CANDIDATE
# ============================================================================

def select_best_batch(
    base_prompt: str,
    keywords_state: Dict,
    main_keyword: str,
    n_candidates: int = 2,  # 🔧 było 3
    model_name: str = "gemini-2.0-flash",
    validation_mode: str = "batch"
) -> Dict[str, Any]:
    """
    🔧 v26.2 OPTIMIZED: Szybszy wybór najlepszego batcha.
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
    
    # 2. Waliduj i scoruj
    scored_candidates = []
    
    for candidate in candidates:
        content = candidate["content"]
        
        if VALIDATOR_AVAILABLE:
            validation = unified_prevalidation(
                text=content,
                keywords_state=keywords_state,
                main_keyword=main_keyword
            )
        else:
            validation = {
                "density": {"value": 1.0},
                "coverage": {"basic_percent": 80, "extended_percent": 80},
                "polish_quality": {"score": 75}
            }
        
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
        
        print(f"[BEST-OF-N] Variant {candidate['variant']}: "
              f"score={score_data['total_score']:.1f}, "
              f"density={score_data['density']:.1f}%, "
              f"coverage_basic={score_data['coverage']['basic']:.0f}%")
    
    # 3. Sortuj
    scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    
    # 4. Wybierz
    best = scored_candidates[0]
    
    meets_minimum = best["total_score"] >= config.min_acceptable_score
    
    if meets_minimum:
        selection_reason = f"✅ Wariant {best['variant']} (score: {best['total_score']:.1f})"
    else:
        selection_reason = (
            f"⚠️ Najlepszy wariant {best['variant']} (score: {best['total_score']:.1f}) "
            f"poniżej minimum {config.min_acceptable_score}"
        )
    
    return {
        "selected_content": best["content"],
        "selected_score": best["total_score"],
        "selected_variant": best["variant"],
        "all_candidates": [
            {
                "variant": c["variant"],
                "score": c["total_score"],
                "density": c["density"],
                "coverage_basic": c["coverage"]["basic"],
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
    use_best_of_n: bool = True,  # 🔧 domyślnie True!
    n_candidates: int = 2  # 🔧 było 3
) -> Dict[str, Any]:
    """
    🔧 v26.2 OPTIMIZED: Domyślnie włączony Best-of-N.
    """
    if not use_best_of_n or n_candidates <= 1:
        content = generate_candidate(batch_content_prompt, 0.7)
        return {
            "content": content,
            "method": "single",
            "score": None
        }
    
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


if __name__ == "__main__":
    print("Best-of-N Selector v26.2 OPTIMIZED")
    print(f"  - Default candidates: {BestOfNConfig().n_candidates}")
    print(f"  - Min acceptable score: {BestOfNConfig().min_acceptable_score}")
    print(f"  - Weights: {BestOfNConfig().weights}")
