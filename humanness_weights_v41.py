"""
===============================================================================
⚖️ HUMANNESS WEIGHTS v41.1 - Nowe wagi + Dynamiczne progi CV
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

🆕 v41.1 ZMIANY:
- DODANO: Dynamiczne progi CV w zależności od długości tekstu
- DODANO: get_dynamic_cv_thresholds(word_count) - zwraca progi dla danej długości
- DODANO: evaluate_cv_dynamic(cv_value, word_count) - ocena CV z dynamicznymi progami
- DODANO: DYNAMIC_CV_THRESHOLDS - lista progów per zakres słów

UZASADNIENIE DYNAMICZNYCH PROGÓW:
- Dłuższe teksty AI mają tendencję do "wygładzania" wariancji
- SHORT (<200 słów): CV >= 0.35 (krótkie batche mają naturalnie mniejszą wariancję)
- MEDIUM (200-400 słów): CV >= 0.40 (standardowy batch)
- LONG (400-600 słów): CV >= 0.43 (większe wymagania)
- EXTENDED (>600 słów): CV >= 0.45 (najwyższe wymagania - AI się "wygładza")

===============================================================================
"""

from typing import Dict, Any  # ✅ POPRAWIONE - dodano Any
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
    """Progi dla metryk v41 (statyczne - legacy)."""
    
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
# 🆕 v41.1: DYNAMICZNE PROGI CV (w zależności od długości tekstu)
# ============================================================================
# Uzasadnienie: Dłuższe teksty AI mają tendencję do "wygładzania" wariancji.
# Im dłuższy tekst, tym wyższe wymagania dla naturalności (CV).
#
# Badania empiryczne (BRAJEN v40.2):
# - SHORT (<200 słów): CV 0.35 wystarcza (krótkie batche mają naturalnie mniejszą wariancję)
# - MEDIUM (200-400 słów): CV 0.40 wymagane (standardowy batch)
# - EXTENDED (>400 słów): CV 0.45 wymagane (długie teksty AI "wygładzają się")
# ============================================================================

@dataclass
class DynamicCVThresholds:
    """Dynamiczne progi CV dla danego zakresu słów."""
    word_count_min: int
    word_count_max: int
    cv_critical: float      # Poniżej = REWRITE
    cv_warning: float       # Poniżej = WARNING
    cv_ok_min: float        # Powyżej = OK
    cv_excellent: float     # Powyżej = EXCELLENT
    label: str


# Definicja progów per zakres długości
DYNAMIC_CV_THRESHOLDS = [
    DynamicCVThresholds(
        word_count_min=0,
        word_count_max=199,
        cv_critical=0.25,
        cv_warning=0.30,
        cv_ok_min=0.35,
        cv_excellent=0.50,
        label="SHORT"
    ),
    DynamicCVThresholds(
        word_count_min=200,
        word_count_max=399,
        cv_critical=0.26,
        cv_warning=0.33,
        cv_ok_min=0.40,
        cv_excellent=0.55,
        label="MEDIUM"
    ),
    DynamicCVThresholds(
        word_count_min=400,
        word_count_max=599,
        cv_critical=0.28,
        cv_warning=0.36,
        cv_ok_min=0.43,
        cv_excellent=0.58,
        label="LONG"
    ),
    DynamicCVThresholds(
        word_count_min=600,
        word_count_max=99999,
        cv_critical=0.30,
        cv_warning=0.38,
        cv_ok_min=0.45,
        cv_excellent=0.60,
        label="EXTENDED"
    ),
]


def get_dynamic_cv_thresholds(word_count: int) -> Dict[str, Any]:
    """
    Zwraca dynamiczne progi CV w zależności od długości tekstu.
    
    Args:
        word_count: Liczba słów w tekście/batchu
        
    Returns:
        Dict z progami i etykietą zakresu:
        {
            "critical": float,    # Poniżej = REWRITE required
            "warning": float,     # Poniżej = WARNING
            "ok_min": float,      # Powyżej = PASS
            "excellent": float,   # Powyżej = EXCELLENT
            "label": str,         # "SHORT" | "MEDIUM" | "LONG" | "EXTENDED"
            "word_count": int,
            "rationale": str
        }
    
    Example:
        >>> get_dynamic_cv_thresholds(150)
        {"critical": 0.25, "warning": 0.30, "ok_min": 0.35, "label": "SHORT", ...}
        
        >>> get_dynamic_cv_thresholds(450)
        {"critical": 0.28, "warning": 0.36, "ok_min": 0.43, "label": "LONG", ...}
    """
    for threshold in DYNAMIC_CV_THRESHOLDS:
        if threshold.word_count_min <= word_count <= threshold.word_count_max:
            return {
                "critical": threshold.cv_critical,
                "warning": threshold.cv_warning,
                "ok_min": threshold.cv_ok_min,
                "excellent": threshold.cv_excellent,
                "label": threshold.label,
                "word_count": word_count,
                "rationale": f"Batch {threshold.label} ({word_count} słów): "
                            f"CV >= {threshold.cv_ok_min} required for PASS"
            }
    
    # Fallback (nie powinno wystąpić)
    return {
        "critical": 0.26,
        "warning": 0.33,
        "ok_min": 0.40,
        "excellent": 0.55,
        "label": "MEDIUM",
        "word_count": word_count,
        "rationale": "Fallback to MEDIUM thresholds"
    }


def evaluate_cv_dynamic(cv_value: float, word_count: int) -> Dict[str, Any]:
    """
    Ocenia wartość CV względem dynamicznych progów.
    
    Args:
        cv_value: Obliczona wartość CV (Coefficient of Variation)
        word_count: Liczba słów w tekście
        
    Returns:
        Dict z oceną:
        {
            "status": "CRITICAL" | "WARNING" | "OK" | "EXCELLENT",
            "passed": bool,
            "cv_value": float,
            "threshold_used": float,
            "margin": float,        # Różnica od progu ok_min
            "action": str,          # "REWRITE" | "IMPROVE" | "CONTINUE"
            "details": str
        }
    """
    thresholds = get_dynamic_cv_thresholds(word_count)
    
    if cv_value < thresholds["critical"]:
        status = "CRITICAL"
        passed = False
        action = "REWRITE"
        details = (f"CV {cv_value:.3f} < {thresholds['critical']} (CRITICAL dla {thresholds['label']}). "
                   f"Tekst zbyt monotonny - wymaga przepisania z większą wariancją zdań.")
    elif cv_value < thresholds["warning"]:
        status = "WARNING"
        passed = False
        action = "IMPROVE"
        details = (f"CV {cv_value:.3f} < {thresholds['warning']} (WARNING dla {thresholds['label']}). "
                   f"Dodaj krótkie zdania (3-8 słów) i zróżnicuj długości.")
    elif cv_value < thresholds["ok_min"]:
        status = "OK_LOW"
        passed = True
        action = "CONTINUE"
        details = (f"CV {cv_value:.3f} - minimalnie akceptowalne dla {thresholds['label']}. "
                   f"Rozważ poprawę wariancji.")
    elif cv_value >= thresholds["excellent"]:
        status = "EXCELLENT"
        passed = True
        action = "CONTINUE"
        details = f"CV {cv_value:.3f} - doskonała wariancja dla {thresholds['label']}."
    else:
        status = "OK"
        passed = True
        action = "CONTINUE"
        details = f"CV {cv_value:.3f} - dobra wariancja dla {thresholds['label']}."
    
    return {
        "status": status,
        "passed": passed,
        "cv_value": round(cv_value, 4),
        "threshold_used": thresholds["ok_min"],
        "margin": round(cv_value - thresholds["ok_min"], 4),
        "action": action,
        "details": details,
        "thresholds": thresholds
    }


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
    print("⚖️ HUMANNESS WEIGHTS v41.1 (+ Dynamic CV Thresholds)")
    print("=" * 60)
    
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
    
    print("\n✅ Walidacja wag:")
    validation = validate_weights_config()
    print(f"   Valid: {validation['valid']}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    # ═══════════════════════════════════════════════════════════════
    # 🆕 TEST DYNAMICZNYCH PROGÓW CV
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🆕 DYNAMICZNE PROGI CV (v41.1)")
    print("=" * 60)
    
    print("\n📏 Progi per zakres długości:")
    print(f"   {'Zakres':<12} {'Label':<10} {'Critical':<10} {'Warning':<10} {'OK min':<10} {'Excellent':<10}")
    print("   " + "-" * 62)
    for t in DYNAMIC_CV_THRESHOLDS:
        range_str = f"{t.word_count_min}-{t.word_count_max}"
        print(f"   {range_str:<12} {t.label:<10} {t.cv_critical:<10.2f} {t.cv_warning:<10.2f} {t.cv_ok_min:<10.2f} {t.cv_excellent:<10.2f}")
    
    print("\n🧪 Test evaluate_cv_dynamic():")
    test_cases = [
        (0.22, 150),   # SHORT, CRITICAL
        (0.32, 150),   # SHORT, OK
        (0.35, 300),   # MEDIUM, WARNING
        (0.42, 300),   # MEDIUM, OK
        (0.38, 500),   # LONG, WARNING
        (0.48, 500),   # LONG, OK
        (0.40, 700),   # EXTENDED, WARNING
        (0.52, 700),   # EXTENDED, OK
    ]
    
    for cv, words in test_cases:
        result = evaluate_cv_dynamic(cv, words)
        icon = "✅" if result["passed"] else "❌"
        print(f"   {icon} CV={cv:.2f}, {words}w → {result['status']:<10} ({result['thresholds']['label']}) | {result['action']}")
