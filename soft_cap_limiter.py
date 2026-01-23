"""
🎚️ SOFT CAP LIMITER v1.0
Elastyczne limity fraz - naturalność > sztywne liczby

Rozwiązuje problem "Over-constraint":
- Zamiast sztywnego hard_max=4 → tiered limits (target, soft_max, hard_max)
- Pozwala na "miękkie przekroczenie" jeśli humanness_score jest wysoki
- Penalizuje stuffing proporcjonalnie, nie binarnie

Autor: SEO Master API v36.2
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class LimitStatus(Enum):
    """Status użycia frazy względem limitów"""
    UNDER = "under"           # poniżej target_min
    ON_TARGET = "on_target"   # między target_min a target_max
    SOFT_EXCEEDED = "soft_exceeded"   # między soft_max a hard_max (OK jeśli naturalny)
    HARD_EXCEEDED = "hard_exceeded"   # powyżej hard_max (zawsze blokuj)


@dataclass
class TieredLimit:
    """
    Wielopoziomowy limit dla frazy.
    
    target: Idealny zakres użyć
    soft_max: Akceptowalne jeśli humanness > threshold
    hard_max: Absolutna granica (stuffing)
    """
    keyword: str
    keyword_type: str = "BASIC"  # BASIC, EXTENDED, MAIN
    
    # Cele
    target_min: int = 1
    target_max: int = 5
    
    # Miękki limit (OK przy wysokim humanness)
    soft_max: int = 7
    humanness_threshold: float = 65.0  # min humanness do akceptacji soft_max
    
    # Twardy limit (zawsze blokuj)
    hard_max: int = 10
    
    # Kary
    soft_overflow_penalty: int = -3    # punkty za każde użycie > target_max
    hard_overflow_penalty: int = -10   # punkty za każde użycie > soft_max
    
    # Stan
    actual_uses: int = 0
    
    @classmethod
    def from_keyword_meta(cls, keyword: str, meta: dict, total_batches: int = 7) -> "TieredLimit":
        """
        Utwórz TieredLimit z metadanych frazy.
        
        Automatycznie oblicza soft_max i hard_max na podstawie target.
        """
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        
        # Oblicz soft_max i hard_max
        if kw_type == "EXTENDED":
            # EXTENDED: sztywne limity (zazwyczaj 1-2)
            soft_max = target_max + 1
            hard_max = target_max + 2
        elif kw_type == "MAIN" or meta.get("is_main_keyword"):
            # MAIN: więcej swobody
            soft_max = int(target_max * 1.3)
            hard_max = int(target_max * 1.6)
        else:
            # BASIC: standardowy margines
            soft_max = int(target_max * 1.2) + 1
            hard_max = int(target_max * 1.5) + 2
        
        return cls(
            keyword=keyword,
            keyword_type=kw_type,
            target_min=target_min,
            target_max=target_max,
            soft_max=soft_max,
            hard_max=hard_max,
            actual_uses=actual
        )
    
    def get_status(self, batch_uses: int = 0) -> LimitStatus:
        """Określ status po dodaniu batch_uses"""
        total = self.actual_uses + batch_uses
        
        if total < self.target_min:
            return LimitStatus.UNDER
        elif total <= self.target_max:
            return LimitStatus.ON_TARGET
        elif total <= self.soft_max:
            return LimitStatus.SOFT_EXCEEDED
        else:
            return LimitStatus.HARD_EXCEEDED
    
    def can_use_more(self, humanness_score: float = 100.0) -> Tuple[bool, str]:
        """
        Sprawdź czy można użyć więcej tej frazy.
        
        Returns:
            Tuple (can_use, reason)
        """
        remaining_to_target = self.target_max - self.actual_uses
        remaining_to_soft = self.soft_max - self.actual_uses
        remaining_to_hard = self.hard_max - self.actual_uses
        
        if remaining_to_target > 0:
            return True, f"OK - pozostało {remaining_to_target} do celu"
        
        if remaining_to_soft > 0:
            if humanness_score >= self.humanness_threshold:
                return True, f"OK (soft) - humanness {humanness_score:.0f}% pozwala na {remaining_to_soft} więcej"
            else:
                return False, f"STOP - cel osiągnięty, humanness {humanness_score:.0f}% < {self.humanness_threshold:.0f}%"
        
        if remaining_to_hard > 0:
            return False, f"⚠️ SOFT EXCEEDED - użycie możliwe tylko przy bardzo wysokim humanness"
        
        return False, f"❌ HARD EXCEEDED - absolutny limit osiągnięty"
    
    def get_batch_recommendation(
        self, 
        remaining_batches: int, 
        humanness_score: float = 100.0
    ) -> dict:
        """
        Oblicz rekomendację dla tego batcha.
        
        Returns:
            Dict z suggested, hard_max_this_batch, instruction
        """
        remaining_to_target = max(0, self.target_min - self.actual_uses)
        remaining_to_max = max(0, self.target_max - self.actual_uses)
        remaining_to_soft = max(0, self.soft_max - self.actual_uses)
        remaining_to_hard = max(0, self.hard_max - self.actual_uses)
        
        # Ile potrzeba per batch żeby osiągnąć target_min
        if remaining_batches > 0 and remaining_to_target > 0:
            suggested = max(1, (remaining_to_target + remaining_batches - 1) // remaining_batches)
        else:
            suggested = 0 if self.actual_uses >= self.target_min else 1
        
        # Hard max dla tego batcha
        if humanness_score >= self.humanness_threshold:
            # Pozwól na więcej jeśli naturalny tekst
            hard_max_this = min(remaining_to_soft, max(2, remaining_batches + 1))
        else:
            hard_max_this = min(remaining_to_max, max(1, remaining_batches))
        
        # Instruction
        status = self.get_status()
        
        if status == LimitStatus.UNDER:
            instruction = f"📌 UŻYJ {suggested}-{hard_max_this}x (potrzeba: {remaining_to_target})"
            priority = "MUST"
        elif status == LimitStatus.ON_TARGET:
            instruction = f"✅ CEL OSIĄGNIĘTY - możesz użyć 0-{hard_max_this}x"
            priority = "OPTIONAL"
        elif status == LimitStatus.SOFT_EXCEEDED:
            instruction = f"⚠️ POWYŻEJ CELU - użyj tylko jeśli naturalnie pasuje (max {remaining_to_hard}x)"
            priority = "AVOID"
        else:
            instruction = f"🛑 LIMIT - NIE UŻYWAJ WIĘCEJ"
            priority = "STOP"
            hard_max_this = 0
        
        return {
            "keyword": self.keyword,
            "type": self.keyword_type,
            "actual": self.actual_uses,
            "target": f"{self.target_min}-{self.target_max}",
            "suggested": suggested,
            "hard_max_this_batch": hard_max_this,
            "instruction": instruction,
            "priority": priority,
            "limits": {
                "target_max": self.target_max,
                "soft_max": self.soft_max,
                "hard_max": self.hard_max
            },
            "status": status.value
        }
    
    def to_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "keyword_type": self.keyword_type,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "soft_max": self.soft_max,
            "hard_max": self.hard_max,
            "humanness_threshold": self.humanness_threshold,
            "actual_uses": self.actual_uses
        }


class SoftCapValidator:
    """
    Walidator z miękkimi limitami.
    
    Zastępuje binarne EXCEEDED → wielopoziomową ocenę.
    """
    
    def __init__(self, keywords_state: dict, total_batches: int = 7):
        """
        Args:
            keywords_state: Dict z metadanymi fraz
            total_batches: Łączna liczba batchy w artykule
        """
        self.total_batches = total_batches
        self.limits: Dict[str, TieredLimit] = {}
        
        # Utwórz TieredLimit dla każdej frazy
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "").strip()
            if keyword:
                self.limits[keyword] = TieredLimit.from_keyword_meta(
                    keyword=keyword,
                    meta=meta,
                    total_batches=total_batches
                )
    
    def validate_batch(
        self, 
        batch_counts: Dict[str, int],
        humanness_score: float = 100.0
    ) -> dict:
        """
        Waliduj batch z miękkimi limitami.
        
        Args:
            batch_counts: Dict {keyword: count} dla tego batcha
            humanness_score: Wynik detekcji AI (0-100)
            
        Returns:
            Dict z wynikami walidacji
        """
        results = {
            "valid": True,
            "hard_exceeded": [],      # Zawsze blokuj
            "soft_exceeded": [],      # OK jeśli humanness wysoki
            "on_target": [],
            "under_target": [],
            "total_penalty": 0,
            "recommendations": []
        }
        
        for keyword, batch_count in batch_counts.items():
            if keyword not in self.limits:
                continue
            
            limit = self.limits[keyword]
            status = limit.get_status(batch_count)
            
            if status == LimitStatus.HARD_EXCEEDED:
                results["hard_exceeded"].append({
                    "keyword": keyword,
                    "actual": limit.actual_uses,
                    "batch_uses": batch_count,
                    "total": limit.actual_uses + batch_count,
                    "hard_max": limit.hard_max,
                    "severity": "CRITICAL"
                })
                results["valid"] = False
                results["total_penalty"] += limit.hard_overflow_penalty * (
                    limit.actual_uses + batch_count - limit.soft_max
                )
                
            elif status == LimitStatus.SOFT_EXCEEDED:
                total = limit.actual_uses + batch_count
                overflow = total - limit.target_max
                
                if humanness_score >= limit.humanness_threshold:
                    # Akceptuj z ostrzeżeniem
                    results["soft_exceeded"].append({
                        "keyword": keyword,
                        "total": total,
                        "target_max": limit.target_max,
                        "soft_max": limit.soft_max,
                        "accepted": True,
                        "reason": f"Humanness {humanness_score:.0f}% >= {limit.humanness_threshold:.0f}%"
                    })
                    results["total_penalty"] += limit.soft_overflow_penalty * overflow
                else:
                    # Blokuj
                    results["soft_exceeded"].append({
                        "keyword": keyword,
                        "total": total,
                        "target_max": limit.target_max,
                        "accepted": False,
                        "reason": f"Humanness {humanness_score:.0f}% < {limit.humanness_threshold:.0f}%"
                    })
                    results["valid"] = False
                    
            elif status == LimitStatus.ON_TARGET:
                results["on_target"].append(keyword)
                
            else:  # UNDER
                results["under_target"].append({
                    "keyword": keyword,
                    "actual": limit.actual_uses,
                    "target_min": limit.target_min,
                    "remaining": limit.target_min - limit.actual_uses
                })
        
        # Generuj rekomendacje
        if results["hard_exceeded"]:
            results["recommendations"].append(
                "❌ USUŃ nadmiarowe użycia fraz: " + 
                ", ".join(e["keyword"] for e in results["hard_exceeded"])
            )
        
        if results["soft_exceeded"]:
            accepted = [e for e in results["soft_exceeded"] if e["accepted"]]
            rejected = [e for e in results["soft_exceeded"] if not e["accepted"]]
            
            if accepted:
                results["recommendations"].append(
                    f"⚠️ Akceptowano lekkie przekroczenie ({len(accepted)} fraz) - tekst naturalny"
                )
            if rejected:
                results["recommendations"].append(
                    "⚠️ ZREDUKUJ użycie fraz (tekst mało naturalny): " +
                    ", ".join(e["keyword"] for e in rejected)
                )
        
        return results
    
    def get_batch_recommendations(
        self, 
        remaining_batches: int,
        humanness_score: float = 100.0
    ) -> List[dict]:
        """
        Pobierz rekomendacje dla wszystkich fraz na nadchodzący batch.
        
        Returns:
            Lista rekomendacji dla każdej frazy
        """
        recommendations = []
        
        for keyword, limit in self.limits.items():
            rec = limit.get_batch_recommendation(
                remaining_batches=remaining_batches,
                humanness_score=humanness_score
            )
            recommendations.append(rec)
        
        # Sortuj: MUST > OPTIONAL > AVOID > STOP
        priority_order = {"MUST": 0, "OPTIONAL": 1, "AVOID": 2, "STOP": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return recommendations
    
    def update_actual_uses(self, batch_counts: Dict[str, int]):
        """Aktualizuj actual_uses po zatwierdzeniu batcha"""
        for keyword, count in batch_counts.items():
            if keyword in self.limits:
                self.limits[keyword].actual_uses += count


def create_soft_cap_validator(keywords_state: dict, total_batches: int = 7) -> SoftCapValidator:
    """Główna funkcja do tworzenia walidatora"""
    return SoftCapValidator(keywords_state, total_batches)


def validate_with_soft_caps(
    batch_counts: Dict[str, int],
    keywords_state: dict,
    humanness_score: float = 100.0,
    total_batches: int = 7
) -> dict:
    """
    Szybka walidacja batcha z miękkimi limitami.
    
    Args:
        batch_counts: {keyword: count} dla batcha
        keywords_state: Metadane fraz
        humanness_score: Wynik detekcji AI
        total_batches: Łączna liczba batchy
        
    Returns:
        Dict z wynikami walidacji
    """
    validator = create_soft_cap_validator(keywords_state, total_batches)
    return validator.validate_batch(batch_counts, humanness_score)


def get_flexible_limits(keyword_meta: dict, total_batches: int = 7) -> dict:
    """
    Oblicz elastyczne limity dla frazy.
    
    Używane w pre_batch_info zamiast sztywnego hard_max.
    
    Returns:
        Dict z target, soft_max, hard_max
    """
    keyword = keyword_meta.get("keyword", "")
    limit = TieredLimit.from_keyword_meta(keyword, keyword_meta, total_batches)
    
    return {
        "keyword": keyword,
        "target_min": limit.target_min,
        "target_max": limit.target_max,
        "soft_max": limit.soft_max,
        "hard_max": limit.hard_max,
        "humanness_threshold": limit.humanness_threshold,
        "flexibility": "HIGH" if limit.keyword_type == "MAIN" else "MEDIUM" if limit.keyword_type == "BASIC" else "LOW"
    }


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    # Przykładowe keywords_state
    keywords_state = {
        "1": {
            "keyword": "ubezwłasnowolnienie",
            "type": "BASIC",
            "target_min": 8,
            "target_max": 15,
            "actual_uses": 12,
            "is_main_keyword": True
        },
        "2": {
            "keyword": "opiekun prawny",
            "type": "BASIC",
            "target_min": 3,
            "target_max": 6,
            "actual_uses": 5
        },
        "3": {
            "keyword": "sąd okręgowy warszawa",
            "type": "EXTENDED",
            "target_min": 1,
            "target_max": 2,
            "actual_uses": 2
        }
    }
    
    # Utwórz walidator
    validator = create_soft_cap_validator(keywords_state, total_batches=7)
    
    # Symuluj batch
    batch_counts = {
        "ubezwłasnowolnienie": 4,  # 12 + 4 = 16 (> target_max=15, < soft_max)
        "opiekun prawny": 2,       # 5 + 2 = 7 (> target_max=6, na granicy soft_max)
        "sąd okręgowy warszawa": 1 # 2 + 1 = 3 (> hard_max dla EXTENDED)
    }
    
    print("=== WALIDACJA Z SOFT CAPS ===")
    print()
    
    # Wysoki humanness
    result_high = validator.validate_batch(batch_counts, humanness_score=75)
    print("Humanness 75%:")
    print(f"  Valid: {result_high['valid']}")
    print(f"  Hard exceeded: {result_high['hard_exceeded']}")
    print(f"  Soft exceeded: {result_high['soft_exceeded']}")
    print(f"  Recommendations: {result_high['recommendations']}")
    print()
    
    # Niski humanness
    validator2 = create_soft_cap_validator(keywords_state, total_batches=7)
    result_low = validator2.validate_batch(batch_counts, humanness_score=50)
    print("Humanness 50%:")
    print(f"  Valid: {result_low['valid']}")
    print(f"  Recommendations: {result_low['recommendations']}")
