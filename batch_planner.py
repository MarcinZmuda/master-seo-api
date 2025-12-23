"""
===============================================================================
📋 BATCH PLANNER v23.0 - Planowanie batchów Z GÓRY
===============================================================================
Rozwiązuje PROBLEM 1: Nadmierna Złożoność Orkiestracji
+ Should Have: Plan batchów z góry (nie reaktywnie)

Zamiast reagować na bieżący stan w pre_batch_info,
system PLANUJE z góry:
- Co będzie w każdym batchu
- Ile słów kluczowych przydzielić do każdego
- Jaka długość docelowa

===============================================================================
"""

import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class BatchPlan:
    """Plan pojedynczego batcha."""
    batch_number: int
    batch_type: str  # "INTRO", "CONTENT", "FINAL"
    h2_sections: List[str]
    target_words_min: int
    target_words_max: int
    keywords_budget: Dict[str, int]  # keyword -> suggested uses
    ngrams_to_use: List[str]
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class ArticlePlan:
    """Kompletny plan artykułu."""
    total_batches: int
    total_target_words: int
    batches: List[BatchPlan]
    keywords_distribution: Dict[str, Dict]  # keyword -> {total, per_batch: []}
    main_keyword: str
    h2_structure: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "total_batches": self.total_batches,
            "total_target_words": self.total_target_words,
            "batches": [b.to_dict() for b in self.batches],
            "keywords_distribution": self.keywords_distribution,
            "main_keyword": self.main_keyword,
            "h2_structure": self.h2_structure
        }


def create_article_plan(
    h2_structure: List[str],
    keywords_state: Dict,
    main_keyword: str,
    target_length: int = 3000,
    ngrams: List[str] = None,
    max_batches: int = 6
) -> ArticlePlan:
    """
    🎯 Tworzy kompletny plan artykułu Z GÓRY.
    
    Args:
        h2_structure: Lista nagłówków H2
        keywords_state: Stan słów kluczowych
        main_keyword: Fraza główna
        target_length: Docelowa długość artykułu
        ngrams: N-gramy z S1
        max_batches: Maksymalna liczba batchów
    
    Returns:
        ArticlePlan z kompletnym planem
    """
    ngrams = ngrams or []
    
    # ================================================================
    # 1. OBLICZ LICZBĘ BATCHÓW
    # ================================================================
    num_h2 = len(h2_structure)
    
    # Heurystyka: 2-3 H2 na batch, min 2 batche, max 6
    if num_h2 <= 3:
        total_batches = 2
    elif num_h2 <= 5:
        total_batches = 3
    elif num_h2 <= 8:
        total_batches = 4
    else:
        total_batches = min(max_batches, math.ceil(num_h2 / 2))
    
    # ================================================================
    # 2. ROZDZIEL H2 NA BATCHE
    # ================================================================
    h2_per_batch = distribute_items(h2_structure, total_batches)
    
    # ================================================================
    # 3. OBLICZ TARGET WORDS PER BATCH
    # ================================================================
    # Intro batch: ~15% słów
    # Content batches: równomiernie reszta
    intro_words = int(target_length * 0.12)  # ~360 słów dla 3000
    remaining_words = target_length - intro_words
    words_per_content_batch = remaining_words // (total_batches - 1) if total_batches > 1 else remaining_words
    
    # ================================================================
    # 4. ROZDZIEL KEYWORDS NA BATCHE
    # ================================================================
    keywords_distribution = distribute_keywords(keywords_state, total_batches, main_keyword)
    
    # ================================================================
    # 5. ROZDZIEL N-GRAMY NA BATCHE
    # ================================================================
    ngrams_per_batch = distribute_items(ngrams, total_batches)
    
    # ================================================================
    # 6. STWÓRZ PLANY BATCHÓW
    # ================================================================
    batches: List[BatchPlan] = []
    
    for i in range(total_batches):
        batch_num = i + 1
        
        # Typ batcha
        if i == 0:
            batch_type = "INTRO"
            target_min = intro_words - 50
            target_max = intro_words + 100
            notes = "INTRO: Direct answer 40-60 słów na początku, fraza główna w pierwszym zdaniu"
        elif i == total_batches - 1:
            batch_type = "FINAL"
            target_min = words_per_content_batch - 100
            target_max = words_per_content_batch + 150
            notes = "OSTATNI BATCH: Upewnij się, że wszystkie CRITICAL keywords są użyte"
        else:
            batch_type = "CONTENT"
            target_min = words_per_content_batch - 100
            target_max = words_per_content_batch + 100
            notes = ""
        
        # Keywords budget dla tego batcha
        keywords_budget = {}
        for kw, dist in keywords_distribution.items():
            if i < len(dist["per_batch"]):
                keywords_budget[kw] = dist["per_batch"][i]
        
        batch_plan = BatchPlan(
            batch_number=batch_num,
            batch_type=batch_type,
            h2_sections=h2_per_batch[i] if i < len(h2_per_batch) else [],
            target_words_min=target_min,
            target_words_max=target_max,
            keywords_budget=keywords_budget,
            ngrams_to_use=ngrams_per_batch[i] if i < len(ngrams_per_batch) else [],
            notes=notes
        )
        batches.append(batch_plan)
    
    return ArticlePlan(
        total_batches=total_batches,
        total_target_words=target_length,
        batches=batches,
        keywords_distribution=keywords_distribution,
        main_keyword=main_keyword,
        h2_structure=h2_structure
    )


def distribute_items(items: List, num_buckets: int) -> List[List]:
    """
    Rozdziela elementy równomiernie na buckety.
    """
    if not items or num_buckets <= 0:
        return [[] for _ in range(num_buckets)]
    
    result = [[] for _ in range(num_buckets)]
    for i, item in enumerate(items):
        bucket_idx = i % num_buckets
        result[bucket_idx].append(item)
    
    return result


def distribute_keywords(
    keywords_state: Dict,
    total_batches: int,
    main_keyword: str
) -> Dict[str, Dict]:
    """
    Rozdziela słowa kluczowe na batche Z GÓRY.
    
    Zasady:
    - MAIN keyword: równomiernie, min 2 na batch
    - BASIC: proporcjonalnie do target_max
    - EXTENDED: 1x w najbardziej pasującym batchu
    
    Returns:
        Dict[keyword] = {
            "total_target": int,
            "per_batch": [int, int, ...]
        }
    """
    distribution = {}
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        is_main = meta.get("is_main_keyword", False)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        per_batch = [0] * total_batches
        
        if is_main or keyword.lower() == main_keyword.lower():
            # MAIN KEYWORD: równomiernie rozłożony, min 2 na batch
            per_batch_target = max(2, math.ceil(target_max / total_batches))
            for i in range(total_batches):
                per_batch[i] = per_batch_target
            # Dostosuj ostatni batch żeby suma = target_max
            current_sum = sum(per_batch)
            if current_sum > target_max:
                diff = current_sum - target_max
                for i in range(total_batches - 1, -1, -1):
                    reduce = min(diff, per_batch[i] - 1)
                    per_batch[i] -= reduce
                    diff -= reduce
                    if diff <= 0:
                        break
        
        elif kw_type == "EXTENDED":
            # EXTENDED: 1x w środkowym batchu
            mid_batch = total_batches // 2
            per_batch[mid_batch] = 1
        
        else:
            # BASIC: proporcjonalnie rozłożone
            if target_max <= total_batches:
                # Mało użyć - rozłóż po 1
                for i in range(min(target_max, total_batches)):
                    per_batch[i] = 1
            else:
                # Więcej użyć - równomiernie
                base = target_max // total_batches
                remainder = target_max % total_batches
                for i in range(total_batches):
                    per_batch[i] = base + (1 if i < remainder else 0)
        
        distribution[keyword] = {
            "total_target": target_max,
            "target_min": target_min,
            "type": kw_type,
            "is_main": is_main,
            "per_batch": per_batch
        }
    
    return distribution


def get_batch_instructions(
    plan: ArticlePlan,
    batch_number: int,
    current_keywords_state: Dict = None
) -> Dict:
    """
    Generuje instrukcje dla konkretnego batcha.
    Uwzględnia aktualny stan (ile już użyto) jeśli podany.
    
    Returns:
        Dict z instrukcjami dla GPT
    """
    if batch_number < 1 or batch_number > plan.total_batches:
        return {"error": f"Invalid batch number: {batch_number}"}
    
    batch_plan = plan.batches[batch_number - 1]
    
    # Oblicz remaining jeśli mamy current state
    keywords_instructions = []
    critical_keywords = []
    high_priority = []
    
    for keyword, budget in batch_plan.keywords_budget.items():
        if budget <= 0:
            continue
        
        # Sprawdź aktualny stan
        actual_used = 0
        target_max = budget
        is_main = False
        
        if current_keywords_state:
            for rid, meta in current_keywords_state.items():
                if meta.get("keyword") == keyword:
                    actual_used = meta.get("actual_uses", 0)
                    target_max = meta.get("target_max", budget)
                    is_main = meta.get("is_main_keyword", False)
                    break
        
        remaining = max(0, target_max - actual_used)
        suggested = min(budget, remaining)
        
        if suggested <= 0:
            continue
        
        kw_info = {
            "keyword": keyword,
            "suggested": suggested,
            "remaining_total": remaining,
            "is_main": is_main
        }
        
        if is_main:
            critical_keywords.append(kw_info)
        elif suggested >= 2:
            high_priority.append(kw_info)
        else:
            keywords_instructions.append(kw_info)
    
    # Buduj prompt
    prompt_lines = []
    prompt_lines.append(f"📋 BATCH #{batch_number} z {plan.total_batches}")
    prompt_lines.append(f"📝 Typ: {batch_plan.batch_type}")
    prompt_lines.append(f"📏 Target: {batch_plan.target_words_min}-{batch_plan.target_words_max} słów")
    prompt_lines.append("")
    
    if batch_plan.notes:
        prompt_lines.append(f"⚠️ {batch_plan.notes}")
        prompt_lines.append("")
    
    # Main keyword
    if critical_keywords:
        prompt_lines.append("=" * 50)
        for kw in critical_keywords:
            prompt_lines.append(f"🔴 FRAZA GŁÓWNA: \"{kw['keyword']}\"")
            prompt_lines.append(f"   → użyj {kw['suggested']}x w tym batchu")
        prompt_lines.append("=" * 50)
        prompt_lines.append("")
    
    # High priority
    if high_priority:
        prompt_lines.append("🟠 WPLEĆ (priorytet):")
        for kw in high_priority:
            prompt_lines.append(f"   • {kw['keyword']}: {kw['suggested']}x")
        prompt_lines.append("")
    
    # H2 do napisania
    if batch_plan.h2_sections:
        prompt_lines.append("✏️ H2 DO NAPISANIA:")
        for h2 in batch_plan.h2_sections:
            prompt_lines.append(f"   • {h2}")
        prompt_lines.append("")
    
    # N-gramy
    if batch_plan.ngrams_to_use:
        prompt_lines.append("📝 N-GRAMY (wpleć naturalnie):")
        for ng in batch_plan.ngrams_to_use[:5]:
            prompt_lines.append(f"   • \"{ng}\"")
        prompt_lines.append("")
    
    return {
        "batch_number": batch_number,
        "batch_type": batch_plan.batch_type,
        "target_words": {
            "min": batch_plan.target_words_min,
            "max": batch_plan.target_words_max
        },
        "h2_sections": batch_plan.h2_sections,
        "keywords": {
            "critical": critical_keywords,
            "high_priority": high_priority,
            "normal": keywords_instructions
        },
        "ngrams": batch_plan.ngrams_to_use,
        "notes": batch_plan.notes,
        "gpt_prompt": "\n".join(prompt_lines)
    }


def update_plan_after_batch(
    plan: ArticlePlan,
    batch_number: int,
    actual_keywords_used: Dict[str, int]
) -> ArticlePlan:
    """
    Aktualizuje plan po napisaniu batcha.
    Przenosi nieużyte keywords do następnych batchów.
    
    Args:
        plan: Obecny plan
        batch_number: Numer ukończonego batcha
        actual_keywords_used: {keyword: actual_count}
    
    Returns:
        Zaktualizowany plan
    """
    if batch_number >= plan.total_batches:
        return plan  # Ostatni batch, nic do przeniesienia
    
    # Dla każdego keyword sprawdź czy użyto mniej niż planowano
    for keyword, dist in plan.keywords_distribution.items():
        planned = dist["per_batch"][batch_number - 1]
        actual = actual_keywords_used.get(keyword, 0)
        
        if actual < planned:
            # Przenieś brakujące do następnych batchów
            deficit = planned - actual
            remaining_batches = plan.total_batches - batch_number
            
            if remaining_batches > 0:
                extra_per_batch = math.ceil(deficit / remaining_batches)
                for i in range(batch_number, plan.total_batches):
                    dist["per_batch"][i] += extra_per_batch
                    deficit -= extra_per_batch
                    if deficit <= 0:
                        break
    
    # Zaktualizuj BatchPlany
    for i, batch in enumerate(plan.batches):
        if i >= batch_number:
            new_budget = {}
            for kw, dist in plan.keywords_distribution.items():
                if i < len(dist["per_batch"]):
                    new_budget[kw] = dist["per_batch"][i]
            batch.keywords_budget = new_budget
    
    return plan
