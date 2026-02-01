"""
===============================================================================
🎯 WIDE COVERAGE STRATEGY v1.0
===============================================================================
NOWA STRATEGIA: Szeroki coverage > głęboki coverage

STARA STRATEGIA (Deep):
  "zespół turnera" 8-15x, ale "choroba genetyczna" 0x
  → Kilka fraz dużo razy, reszta pominięta

NOWA STRATEGIA (Wide):
  "zespół turnera" 2-4x, "choroba genetyczna" 2x, "aberracja" 2x, itd.
  → WSZYSTKIE frazy min 2x, żadna nie pominięta

ZASADY:
1. KAŻDA fraza BASIC/EXTENDED musi mieć min 2 użycia
2. Target max jest OBNIŻONY aby agent nie skupiał się na jednej frazie
3. Priorytet w batchu: frazy z 0 użyć > frazy z 1 użyciem > reszta
4. "Szerokość" jest ważniejsza niż "głębokość"

===============================================================================
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ============================================================================
# KONFIGURACJA - WIDE COVERAGE
# ============================================================================

# Minimalne użycia - HARD REQUIREMENT
MINIMUM_USES = {
    "MAIN": 3,       # Main keyword min 3x
    "BASIC": 2,      # BASIC min 2x - HARD REQUIREMENT!
    "EXTENDED": 2,   # EXTENDED min 2x - priorytet niższy ale też wymagane
    "H2": 1,         # H2 headers min 1x
}

# Maksymalne użycia - OBNIŻONE aby wymuszać szerokość
MAX_USES_MULTIPLIER = {
    "MAIN": 2.5,     # Main: min * 2.5 (np. 3 * 2.5 = 7-8 max)
    "BASIC": 2.0,    # BASIC: min * 2 (np. 2 * 2 = 4 max)
    "EXTENDED": 2.0, # EXTENDED: min * 2
    "H2": 3.0,       # H2 headers: min * 3
}

# Absolutne maksimum (niezależnie od obliczeń)
ABSOLUTE_MAX = {
    "MAIN": 15,
    "BASIC": 6,      # Max 6 użyć BASIC - wymusza dystrybucję
    "EXTENDED": 4,   # Max 4 użycia EXTENDED
    "H2": 5,
}

# Progi coverage
COVERAGE_THRESHOLDS = {
    "EXCELLENT": 0.95,  # 95%+ fraz ma min 2 użycia
    "GOOD": 0.80,       # 80%+ fraz
    "ACCEPTABLE": 0.60, # 60%+ fraz
    "POOR": 0.40,       # Poniżej = CRITICAL
}


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class WideCoverageTarget:
    """Nowy target dla frazy w strategii wide coverage."""
    phrase: str
    phrase_type: str
    min_uses: int
    max_uses: int
    current_uses: int
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW, DONE
    priority_reason: str


@dataclass
class CoverageReport:
    """Raport pokrycia fraz."""
    total_phrases: int
    covered_phrases: int  # >= min_uses
    partially_covered: int  # > 0 ale < min_uses
    not_covered: int  # = 0
    coverage_ratio: float
    status: str  # EXCELLENT, GOOD, ACCEPTABLE, POOR
    critical_phrases: List[str]  # Frazy z 0 użyć
    needs_attention: List[str]  # Frazy z 1 użyciem (BASIC/EXTENDED)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def calculate_wide_coverage_targets(
    keywords_state: Dict[str, dict],
    article_length: int = 2000
) -> Dict[str, WideCoverageTarget]:
    """
    Oblicza nowe targety w strategii wide coverage.
    
    Args:
        keywords_state: Obecny stan fraz {rid: {keyword, type, actual_uses, ...}}
        article_length: Długość artykułu (słowa)
    
    Returns:
        Dict {rid: WideCoverageTarget}
    """
    targets = {}
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        current_uses = meta.get("actual_uses", 0)
        
        if not phrase:
            continue
        
        # Oblicz min/max
        min_uses = MINIMUM_USES.get(phrase_type, 2)
        
        # Max = min * multiplier, ale nie więcej niż ABSOLUTE_MAX
        multiplier = MAX_USES_MULTIPLIER.get(phrase_type, 2.0)
        calculated_max = int(min_uses * multiplier)
        absolute_max = ABSOLUTE_MAX.get(phrase_type, 6)
        max_uses = min(calculated_max, absolute_max)
        
        # Dostosuj do długości artykułu (dla bardzo krótkich)
        if article_length < 1000:
            max_uses = max(min_uses, max_uses - 1)
        
        # Określ priorytet
        if current_uses == 0:
            priority = "CRITICAL"
            priority_reason = f"0/{min_uses} użyć - WYMAGANE!"
        elif current_uses < min_uses:
            priority = "HIGH"
            priority_reason = f"{current_uses}/{min_uses} użyć - potrzeba jeszcze {min_uses - current_uses}x"
        elif current_uses == min_uses:
            priority = "MEDIUM"
            priority_reason = f"Minimum osiągnięte ({min_uses}x) - opcjonalnie więcej"
        elif current_uses < max_uses:
            priority = "LOW"
            priority_reason = f"OK ({current_uses}x) - można dodać do {max_uses}x"
        else:
            priority = "DONE"
            priority_reason = f"Max osiągnięty ({current_uses}/{max_uses}x) - STOP"
        
        targets[rid] = WideCoverageTarget(
            phrase=phrase,
            phrase_type=phrase_type,
            min_uses=min_uses,
            max_uses=max_uses,
            current_uses=current_uses,
            priority=priority,
            priority_reason=priority_reason
        )
    
    return targets


def get_batch_priorities(
    targets: Dict[str, WideCoverageTarget],
    batch_number: int,
    total_batches: int,
    max_phrases_per_batch: int = 5
) -> Dict:
    """
    Zwraca frazy do użycia w tym batchu, posortowane priorytetem.
    
    ZASADA: Szerokość > Głębokość
    - Najpierw frazy z 0 użyć (CRITICAL)
    - Potem frazy z 1 użyciem (HIGH)  
    - Potem reszta
    
    Returns:
        {
            "must_use": [...],      # CRITICAL - muszą być w tym batchu
            "should_use": [...],    # HIGH - powinny być
            "can_use": [...],       # MEDIUM/LOW - opcjonalne
            "stop": [...],          # DONE - NIE używać
            "coverage_pressure": str # Jak pilne jest pokrycie
        }
    """
    critical = []
    high = []
    medium = []
    low = []
    done = []
    
    for rid, target in targets.items():
        entry = {
            "rid": rid,
            "phrase": target.phrase,
            "type": target.phrase_type,
            "current": target.current_uses,
            "min": target.min_uses,
            "max": target.max_uses,
            "reason": target.priority_reason
        }
        
        if target.priority == "CRITICAL":
            critical.append(entry)
        elif target.priority == "HIGH":
            high.append(entry)
        elif target.priority == "MEDIUM":
            medium.append(entry)
        elif target.priority == "LOW":
            low.append(entry)
        else:
            done.append(entry)
    
    # Sortuj CRITICAL i HIGH według typu (BASIC przed EXTENDED)
    type_order = {"MAIN": 0, "BASIC": 1, "EXTENDED": 2, "H2": 3}
    critical.sort(key=lambda x: type_order.get(x["type"], 4))
    high.sort(key=lambda x: type_order.get(x["type"], 4))
    
    # Oblicz pressure
    remaining_batches = total_batches - batch_number + 1
    critical_count = len(critical)
    high_count = len(high)
    
    if critical_count > remaining_batches * 2:
        pressure = "CRITICAL"
    elif critical_count > 0 and remaining_batches <= 2:
        pressure = "HIGH"
    elif high_count > remaining_batches * 3:
        pressure = "MEDIUM"
    else:
        pressure = "LOW"
    
    # Wybierz frazy do batcha
    must_use = critical[:max_phrases_per_batch]
    remaining_slots = max_phrases_per_batch - len(must_use)
    
    should_use = high[:remaining_slots] if remaining_slots > 0 else []
    remaining_slots -= len(should_use)
    
    can_use = (medium + low)[:remaining_slots] if remaining_slots > 0 else []
    
    return {
        "must_use": must_use,
        "should_use": should_use,
        "can_use": can_use,
        "stop": done,
        "coverage_pressure": pressure,
        "stats": {
            "critical_total": len(critical),
            "high_total": len(high),
            "remaining_batches": remaining_batches
        }
    }


def calculate_coverage_report(targets: Dict[str, WideCoverageTarget]) -> CoverageReport:
    """
    Oblicza raport pokrycia fraz.
    
    Returns:
        CoverageReport z info o coverage
    """
    total = 0
    covered = 0
    partial = 0
    not_covered = 0
    critical_phrases = []
    needs_attention = []
    
    for rid, target in targets.items():
        # Pomiń MAIN (ma inne zasady)
        if target.phrase_type == "MAIN":
            continue
        
        total += 1
        
        if target.current_uses >= target.min_uses:
            covered += 1
        elif target.current_uses > 0:
            partial += 1
            needs_attention.append(target.phrase)
        else:
            not_covered += 1
            critical_phrases.append(target.phrase)
    
    coverage_ratio = covered / total if total > 0 else 0
    
    # Określ status
    if coverage_ratio >= COVERAGE_THRESHOLDS["EXCELLENT"]:
        status = "EXCELLENT"
    elif coverage_ratio >= COVERAGE_THRESHOLDS["GOOD"]:
        status = "GOOD"
    elif coverage_ratio >= COVERAGE_THRESHOLDS["ACCEPTABLE"]:
        status = "ACCEPTABLE"
    else:
        status = "POOR"
    
    return CoverageReport(
        total_phrases=total,
        covered_phrases=covered,
        partially_covered=partial,
        not_covered=not_covered,
        coverage_ratio=coverage_ratio,
        status=status,
        critical_phrases=critical_phrases,
        needs_attention=needs_attention
    )


# ============================================================================
# RECALCULATE EXISTING TARGETS
# ============================================================================

def recalculate_targets_for_wide_coverage(
    keywords_state: Dict[str, dict],
    article_length: int = 2000
) -> Dict[str, dict]:
    """
    Przelicza istniejące targety na strategię wide coverage.
    
    Użyj tej funkcji w project_routes.py przy tworzeniu projektu
    lub w pre_batch_info aby nadpisać targety.
    
    Returns:
        Zaktualizowany keywords_state z nowymi target_min/max
    """
    updated = {}
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            updated[rid] = meta
            continue
        
        # Oblicz nowe targety
        min_uses = MINIMUM_USES.get(phrase_type, 2)
        multiplier = MAX_USES_MULTIPLIER.get(phrase_type, 2.0)
        calculated_max = int(min_uses * multiplier)
        absolute_max = ABSOLUTE_MAX.get(phrase_type, 6)
        max_uses = min(calculated_max, absolute_max)
        
        # Dostosuj do długości artykułu
        if article_length < 1000:
            max_uses = max(min_uses, max_uses - 1)
        elif article_length > 3000:
            # Dłuższe artykuły mogą mieć więcej, ale nie za dużo
            max_uses = min(max_uses + 2, absolute_max)
        
        # Zaktualizuj meta
        new_meta = meta.copy()
        new_meta["target_min"] = min_uses
        new_meta["target_max"] = max_uses
        new_meta["wide_coverage"] = True  # Flag że używamy nowej strategii
        
        updated[rid] = new_meta
    
    return updated


# ============================================================================
# FORMAT FOR PROMPT
# ============================================================================

def format_wide_coverage_instructions(
    priorities: Dict,
    coverage_report: CoverageReport
) -> str:
    """
    Formatuje instrukcje wide coverage dla agenta.
    """
    lines = []
    
    lines.append("\n" + "=" * 60)
    lines.append("🎯 WIDE COVERAGE - Szerokość > Głębokość")
    lines.append("=" * 60)
    
    # Coverage status
    status_emoji = {
        "EXCELLENT": "🟢",
        "GOOD": "🟡",
        "ACCEPTABLE": "🟠",
        "POOR": "🔴"
    }
    
    lines.append(f"\n📊 COVERAGE: {status_emoji.get(coverage_report.status, '⚪')} {coverage_report.status}")
    lines.append(f"   {coverage_report.covered_phrases}/{coverage_report.total_phrases} fraz pokrytych ({round(coverage_report.coverage_ratio * 100)}%)")
    
    if coverage_report.not_covered > 0:
        lines.append(f"   ⚠️ {coverage_report.not_covered} fraz z 0 użyć!")
    
    # Pressure
    pressure = priorities.get("coverage_pressure", "LOW")
    if pressure == "CRITICAL":
        lines.append(f"\n🚨 PRESSURE: CRITICAL - dużo fraz bez pokrycia!")
    elif pressure == "HIGH":
        lines.append(f"\n⚠️ PRESSURE: HIGH - ostatnie batche, użyj brakujących fraz!")
    
    # Must use (CRITICAL)
    if priorities["must_use"]:
        lines.append("\n" + "─" * 40)
        lines.append("🔴 MUST USE (0 użyć - WYMAGANE):")
        for p in priorities["must_use"]:
            lines.append(f"   • \"{p['phrase']}\" ({p['type']}) - {p['reason']}")
    
    # Should use (HIGH)
    if priorities["should_use"]:
        lines.append("\n" + "─" * 40)
        lines.append("🟠 SHOULD USE (potrzeba więcej):")
        for p in priorities["should_use"]:
            lines.append(f"   • \"{p['phrase']}\" - {p['current']}/{p['min']} użyć")
    
    # Stop
    if priorities["stop"]:
        lines.append("\n" + "─" * 40)
        lines.append("⛔ STOP (max osiągnięty):")
        stop_phrases = [p['phrase'] for p in priorities["stop"][:5]]
        lines.append(f"   {', '.join(stop_phrases)}")
    
    # Reminder
    lines.append("\n" + "─" * 40)
    lines.append("💡 ZASADA: Lepiej użyć WSZYSTKIE frazy po 2x niż kilka po 8x!")
    lines.append("   → Najpierw frazy z 0 użyć, potem z 1 użyciem")
    
    return "\n".join(lines)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_wide_coverage_final(
    targets: Dict[str, WideCoverageTarget]
) -> Dict:
    """
    Walidacja końcowa - czy osiągnięto wide coverage.
    Użyj w final_review.
    
    Returns:
        {
            "passed": bool,
            "coverage_report": CoverageReport,
            "issues": [...],
            "suggestions": [...]
        }
    """
    report = calculate_coverage_report(targets)
    
    issues = []
    suggestions = []
    
    # Check critical
    if report.not_covered > 0:
        issues.append({
            "type": "UNCOVERED_PHRASES",
            "severity": "CRITICAL",
            "count": report.not_covered,
            "phrases": report.critical_phrases
        })
        suggestions.append(
            f"Dodaj {report.not_covered} brakujących fraz: {', '.join(report.critical_phrases[:3])}"
        )
    
    # Check partial
    if report.partially_covered > 0:
        issues.append({
            "type": "PARTIAL_COVERAGE",
            "severity": "WARNING",
            "count": report.partially_covered,
            "phrases": report.needs_attention
        })
        suggestions.append(
            f"{report.partially_covered} fraz ma tylko 1 użycie - dodaj drugie"
        )
    
    # Passed if coverage >= 80% AND no critical phrases
    passed = (
        report.coverage_ratio >= COVERAGE_THRESHOLDS["GOOD"] and
        report.not_covered == 0
    )
    
    return {
        "passed": passed,
        "coverage_report": report,
        "issues": issues,
        "suggestions": suggestions
    }


# ============================================================================
# MAIN - TEST
# ============================================================================

if __name__ == "__main__":
    # Test
    keywords_state = {
        "k1": {"keyword": "zespół turnera", "type": "MAIN", "actual_uses": 4},
        "k2": {"keyword": "choroba genetyczna", "type": "BASIC", "actual_uses": 0},
        "k3": {"keyword": "aberracja chromosomalna", "type": "BASIC", "actual_uses": 1},
        "k4": {"keyword": "niski wzrost", "type": "BASIC", "actual_uses": 2},
        "k5": {"keyword": "zaburzenia hormonalne", "type": "EXTENDED", "actual_uses": 0},
        "k6": {"keyword": "wady serca", "type": "EXTENDED", "actual_uses": 0},
        "k7": {"keyword": "diagnostyka", "type": "EXTENDED", "actual_uses": 3},
    }
    
    print("=" * 70)
    print("TEST WIDE COVERAGE STRATEGY")
    print("=" * 70)
    
    # 1. Calculate targets
    targets = calculate_wide_coverage_targets(keywords_state, article_length=2000)
    
    print("\n📋 TARGETS:")
    for rid, target in targets.items():
        print(f"  {target.phrase}: {target.current_uses}/{target.min_uses}-{target.max_uses} [{target.priority}]")
    
    # 2. Get batch priorities
    priorities = get_batch_priorities(targets, batch_number=3, total_batches=5)
    
    # 3. Coverage report
    report = calculate_coverage_report(targets)
    
    print(f"\n📊 COVERAGE REPORT:")
    print(f"  Status: {report.status}")
    print(f"  Covered: {report.covered_phrases}/{report.total_phrases} ({round(report.coverage_ratio * 100)}%)")
    print(f"  Not covered: {report.not_covered}")
    print(f"  Critical phrases: {report.critical_phrases}")
    
    # 4. Format instructions
    print(format_wide_coverage_instructions(priorities, report))
    
    # 5. Final validation
    print("\n" + "=" * 70)
    print("📝 FINAL VALIDATION:")
    print("=" * 70)
    
    validation = validate_wide_coverage_final(targets)
    print(f"\nPassed: {'✅ TAK' if validation['passed'] else '❌ NIE'}")
    
    if validation["issues"]:
        print("\nIssues:")
        for issue in validation["issues"]:
            print(f"  - [{issue['severity']}] {issue['type']}: {issue['count']} fraz")
    
    if validation["suggestions"]:
        print("\nSuggestions:")
        for s in validation["suggestions"]:
            print(f"  - {s}")
