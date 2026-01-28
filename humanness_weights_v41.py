"""
===============================================================================
⚖️ HUMANNESS WEIGHTS v41.0 - Nowe wagi dla AI detection
===============================================================================

Aktualizacja wag w calculate_humanness_score() na podstawie:
1. Dodania nowej metryki: paragraph_cv
2. Badań MDPI 2024 (Random Forest 98.3% dokładności)
3. Praktycznych obserwacji w BRAJEN v40.2

ZMIANY WZGLĘDEM v36.5:
- DODANO: paragraph_cv (0.12) - #2 cecha wykrywania AI
- ZMNIEJSZONO: burstiness (0.18 → 0.16) - wciąż ważne ale mniej dominujące
- ZMNIEJSZONO: vocabulary (0.18 → 0.14) - MATTR jest bardziej stabilny
- ZMNIEJSZONO: entropy (0.15 → 0.12)
- ZMNIEJSZONO: repetition (0.12 → 0.10)
- ZMNIEJSZONO: sophistication (0.10 → 0.08)
- ZMNIEJSZONO: pos_diversity (0.07 → 0.05)
- ZWIĘKSZONO: template_diversity (0.15 → 0.16) - ważne dla wzorców AI

SUMA WAG: 1.00 (bez zmian)

===============================================================================
"""

from typing import Dict
from dataclasses import dataclass


# ============================================================================
# OBECNE WAGI (v36.5) - DLA PORÓWNANIA
# ============================================================================

WEIGHTS_V36_5 = {
    "burstiness": 0.18,
    "vocabulary": 0.18,
    "sophistication": 0.10,
    "entropy": 0.15,
    "repetition": 0.12,
    "pos_diversity": 0.07,
    "sentence_distribution": 0.05,
    "template_diversity": 0.15
}
# SUMA: 1.00


# ============================================================================
# NOWE WAGI v41.0
# ============================================================================

WEIGHTS_V41 = {
    # ===============================================
    # NAJWAŻNIEJSZE METRYKI (łącznie 0.44)
    # ===============================================
    
    # Burstiness (CV zdań) - nadal ważne, ale mniej dominujące
    # Badania: #1 cecha rozróżniająca AI od ludzi
    "burstiness": 0.16,  # było 0.18
    
    # 🆕 Paragraph CV - NOWA METRYKA
    # Badania MDPI 2024: #2 cecha po CV zdań
    # CV długości akapitów - AI produkuje jednolite akapity
    "paragraph_cv": 0.12,  # NOWE
    
    # Template Diversity - wzorce AI
    # Wykrywa powtarzalne struktury zdań
    "template_diversity": 0.16,  # było 0.15
    
    # ===============================================
    # WAŻNE METRYKI (łącznie 0.36)
    # ===============================================
    
    # Vocabulary Richness (MATTR zamiast TTR)
    # MATTR jest bardziej stabilny dla różnych długości
    "vocabulary": 0.14,  # było 0.18
    
    # Starter Entropy - różnorodność początków zdań
    # AI często zaczyna zdania podobnie
    "entropy": 0.12,  # było 0.15
    
    # Word Repetition - nadmierne powtórzenia
    "repetition": 0.10,  # było 0.12
    
    # ===============================================
    # DRUGORZĘDNE METRYKI (łącznie 0.20)
    # ===============================================
    
    # Lexical Sophistication (Zipf frequency)
    # Mniej wiarygodne dla polskiego
    "sophistication": 0.08,  # było 0.10
    
    # POS Diversity - różnorodność części mowy
    # Wymaga spaCy, nie zawsze dostępne
    "pos_diversity": 0.05,  # było 0.07
    
    # Sentence Distribution - rozkład długości
    # Częściowo pokrywa się z burstiness
    "sentence_distribution": 0.07,  # było 0.05
}

# Weryfikacja sumy
assert abs(sum(WEIGHTS_V41.values()) - 1.0) < 0.001, "Wagi muszą sumować się do 1.0!"


# ============================================================================
# PROGI DLA NOWYCH METRYK
# ============================================================================

@dataclass
class ThresholdsV41:
    """Progi dla metryk v41."""
    
    # Paragraph CV (NOWE)
    PARAGRAPH_CV_CRITICAL_LOW: float = 0.25
    PARAGRAPH_CV_WARNING_LOW: float = 0.35
    PARAGRAPH_CV_OK_MIN: float = 0.35
    PARAGRAPH_CV_OK_MAX: float = 0.70
    
    # MATTR (zastępuje proste TTR)
    MATTR_CRITICAL: float = 0.35
    MATTR_WARNING: float = 0.42
    MATTR_OK: float = 0.42
    
    # Pozostałe bez zmian (z AIDetectionConfig)


THRESHOLDS_V41 = ThresholdsV41()


# ============================================================================
# FUNKCJA AKTUALIZACJI
# ============================================================================

def get_weights_v41() -> Dict[str, float]:
    """Zwraca nowe wagi v41."""
    return WEIGHTS_V41.copy()


def get_weight_changes() -> Dict[str, Dict]:
    """Zwraca porównanie zmian wag."""
    changes = {}
    
    all_keys = set(WEIGHTS_V36_5.keys()) | set(WEIGHTS_V41.keys())
    
    for key in all_keys:
        old = WEIGHTS_V36_5.get(key, 0)
        new = WEIGHTS_V41.get(key, 0)
        diff = new - old
        
        if diff != 0 or key not in WEIGHTS_V36_5:
            changes[key] = {
                "old": old,
                "new": new,
                "diff": round(diff, 3),
                "change": "NEW" if old == 0 else ("↑" if diff > 0 else "↓")
            }
    
    return changes


# ============================================================================
# INSTRUKCJA INTEGRACJI
# ============================================================================

"""
INTEGRACJA Z ai_detection_metrics.py:

1. Zamień definicję WEIGHTS w AIDetectionConfig:

   from humanness_weights_v41 import WEIGHTS_V41
   
   class AIDetectionConfig:
       # ...
       WEIGHTS = WEIGHTS_V41

2. Dodaj import paragraph_cv w calculate_humanness_score():

   from paragraph_cv_analyzer_v41 import calculate_paragraph_cv
   
   # W funkcji calculate_humanness_score():
   paragraph_cv = calculate_paragraph_cv(text)
   scores["paragraph_cv"] = paragraph_cv["score"] / 100  # normalize 0-1

3. Dodaj import MATTR:

   from mattr_calculator_v41 import calculate_mattr
   
   # Zamień calculate_vocabulary_richness() na MATTR dla długich tekstów:
   if len(text.split()) >= 500:
       vocab_result = calculate_mattr(text)
   else:
       vocab_result = calculate_vocabulary_richness(text)  # fallback

4. Aktualizuj funkcję łączącą scores:

   # W calculate_humanness_score(), po obliczeniu wszystkich metryk:
   weighted_sum = 0
   for metric, weight in WEIGHTS_V41.items():
       if metric in scores:
           weighted_sum += scores[metric] * weight
   
   humanness_score = weighted_sum * 100

5. WAŻNE: Upewnij się że wszystkie metryki zwracają wartość 0-1 (normalized).
"""


# ============================================================================
# WALIDACJA KONFIGURACJI
# ============================================================================

def validate_weights_config() -> Dict[str, Any]:
    """
    Waliduje konfigurację wag.
    
    Returns:
        Dict z wynikami walidacji
    """
    issues = []
    warnings = []
    
    # Sprawdź sumę
    total = sum(WEIGHTS_V41.values())
    if abs(total - 1.0) > 0.001:
        issues.append(f"Suma wag = {total}, powinno być 1.0")
    
    # Sprawdź czy są wszystkie wymagane metryki
    required = ["burstiness", "vocabulary", "entropy", "repetition"]
    for metric in required:
        if metric not in WEIGHTS_V41:
            issues.append(f"Brak wymaganej metryki: {metric}")
    
    # Sprawdź czy paragraph_cv ma rozsądną wagę
    if WEIGHTS_V41.get("paragraph_cv", 0) < 0.05:
        warnings.append("paragraph_cv ma niską wagę - rozważ zwiększenie")
    
    # Sprawdź czy żadna waga nie dominuje
    max_weight = max(WEIGHTS_V41.values())
    if max_weight > 0.25:
        warnings.append(f"Jedna metryka ma wagę > 0.25 ({max_weight}) - może dominować")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_weight": round(total, 4),
        "metrics_count": len(WEIGHTS_V41)
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("⚖️ HUMANNESS WEIGHTS v41.0")
    print("=" * 50)
    
    print("\n📊 Nowe wagi:")
    for metric, weight in sorted(WEIGHTS_V41.items(), key=lambda x: -x[1]):
        print(f"   {metric}: {weight:.2f}")
    
    print(f"\n   SUMA: {sum(WEIGHTS_V41.values()):.2f}")
    
    print("\n🔄 Zmiany względem v36.5:")
    changes = get_weight_changes()
    for metric, change in sorted(changes.items(), key=lambda x: -abs(x[1]["diff"])):
        if change["change"] == "NEW":
            print(f"   {metric}: {change['change']} ({change['new']:.2f})")
        else:
            print(f"   {metric}: {change['old']:.2f} → {change['new']:.2f} ({change['change']})")
    
    print("\n✅ Walidacja:")
    validation = validate_weights_config()
    print(f"   Valid: {validation['valid']}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
