"""
===============================================================================
📋 BATCH PLANNER v28.1 - DYNAMICZNE DŁUGOŚCI OPARTE NA TREŚCI
===============================================================================
v28.1: Długość batcha zależy od WIELU czynników:

  1. Typ H2 (pytające/instrukcja/definicja) - bazowy profil
  2. N-gramy powiązane z sekcją - im więcej, tym dłużej
  3. Encje do zdefiniowania - każda wymaga miejsca
  4. Keywords do wplecenia - minimum dla zachowania density
  5. PAA match - bonus za snippet potential

Score 0-100 → profil: short/medium/long/extended

===============================================================================
"""

import math
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Import kalkulatora złożoności
try:
    from batch_complexity import (
        calculate_batch_complexity,
        calculate_complexity_for_batch_plan,
        PROFILES,
        ComplexityResult,
        classify_h2_type
    )
    COMPLEXITY_AVAILABLE = True
    print("[BATCH_PLANNER] ✅ batch_complexity module loaded")
except ImportError:
    COMPLEXITY_AVAILABLE = False
    print("[BATCH_PLANNER] ⚠️ batch_complexity not available, using fallback")


# ================================================================
# 📋 STRUKTURY DANYCH
# ================================================================

@dataclass
class BatchPlan:
    """Plan pojedynczego batcha."""
    batch_number: int
    batch_type: str  # "INTRO", "CONTENT", "FINAL"
    h2_sections: List[str]
    target_words_min: int
    target_words_max: int
    target_paragraphs_min: int
    target_paragraphs_max: int
    length_profile: str  # short/medium/long/extended
    complexity_score: int  # 0-100
    complexity_factors: Dict[str, Any]
    complexity_reasoning: List[str]
    keywords_budget: Dict[str, int]
    ngrams_to_use: List[str]
    snippet_required: bool
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class ArticlePlan:
    """Kompletny plan artykułu."""
    total_batches: int
    total_target_words: int
    batches: List[BatchPlan]
    keywords_distribution: Dict[str, Dict]
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


# ================================================================
# 🔧 FALLBACK - gdy batch_complexity nie jest dostępny
# ================================================================

H2_TYPE_FALLBACK = {
    "tutorial": {
        "patterns": ["krok po kroku", "poradnik", "instrukcja", "jak zrobić", "jak wykonać"],
        "profile": "long",
        "words": (500, 700),  # 🆕 v41.2: +100 (było 400-600)
        "paragraphs": (3, 4)
    },
    "definition": {
        "patterns": ["co to", "czym jest", "definicja", "co oznacza"],
        "profile": "short",
        "words": (300, 450),  # 🆕 v41.2: +100 (było 200-350)
        "paragraphs": (2, 3)
    },
    "yes_no": {
        "patterns": ["czy można", "czy warto", "czy trzeba", "czy należy"],
        "profile": "short",
        "words": (300, 450),  # 🆕 v41.2: +100 (było 200-350)
        "paragraphs": (2, 3)
    },
    "comparison": {
        "patterns": ["vs", "porównanie", "różnice", "co lepsze"],
        "profile": "long",
        "words": (500, 700),  # 🆕 v41.2: +100 (było 400-600)
        "paragraphs": (3, 4)
    },
    "list": {
        "patterns": ["najlepsze", "top", "ranking", "rodzaje", "typy"],
        "profile": "extended",
        "words": (600, 850),  # 🆕 v41.2: +100 (było 500-750)
        "paragraphs": (3, 4)  # 🆕 v41.2: max 4 (było 5-7)
    },
    "explanation": {
        "patterns": ["jak ", "dlaczego", "w jaki sposób"],
        "profile": "long",
        "words": (500, 700),  # 🆕 v41.2: +100 (było 400-600)
        "paragraphs": (3, 4)
    }
}

DEFAULT_FALLBACK = {
    "profile": "medium",
    "words": (400, 600),  # 🆕 v41.2: +100 (było 300-500)
    "paragraphs": (2, 4)
}


def detect_h2_type_fallback(h2_title: str) -> Dict:
    """Fallback wykrywania typu H2."""
    h2_lower = h2_title.lower().strip()
    
    for type_name, config in H2_TYPE_FALLBACK.items():
        for pattern in config["patterns"]:
            if pattern in h2_lower:
                return config
    
    return DEFAULT_FALLBACK


def calculate_length_fallback(
    h2_sections: List[str],
    keywords_count: int,
    is_intro: bool,
    is_final: bool
) -> Dict:
    """Fallback obliczania długości bez batch_complexity."""
    
    # 🆕 v41.2: MINIMUM 250 słów per H2! (było 150)
    MIN_WORDS_PER_H2 = 250
    h2_count = len(h2_sections) if h2_sections else 1
    
    if is_intro:
        # 🆕 v41.2: INTRO skaluje się z liczbą H2
        base_min = max(300, h2_count * MIN_WORDS_PER_H2)
        base_max = max(450, h2_count * 350)
        return {
            "profile": "intro",
            "words_min": base_min,
            "words_max": base_max,
            "paragraphs_min": max(2, h2_count * 2),
            "paragraphs_max": min(h2_count * 4, 12),  # Max 4 per H2, max 12 total
            "score": 30,
            "factors": {"type": "intro", "h2_count": h2_count},
            "reasoning": [f"INTRO: {h2_count} H2 × {MIN_WORDS_PER_H2} słów = {base_min}-{base_max} słów"]
        }
    
    if h2_sections:
        configs = [detect_h2_type_fallback(h2) for h2 in h2_sections]
        best_config = max(configs, key=lambda c: c["words"][1])
    else:
        best_config = DEFAULT_FALLBACK
    
    words_min, words_max = best_config["words"]
    para_min, para_max = best_config["paragraphs"]
    
    # 🆕 v41.2: MNOŻNIK PER H2 z max 4 akapitów per sekcja
    if h2_count > 1:
        # Każda dodatkowa H2 dodaje ~80% bazowej długości
        h2_multiplier = 1 + (h2_count - 1) * 0.8
        words_min = int(words_min * h2_multiplier)
        words_max = int(words_max * h2_multiplier)
        # 🆕 v41.2: Max 4 akapity per H2, różna liczba dla każdej
        para_min = h2_count * 2  # Min 2 akapity per H2
        para_max = h2_count * 4  # Max 4 akapity per H2
    
    # Enforce absolute minimum: 250 słów per H2 (było 150)
    words_min = max(words_min, h2_count * MIN_WORDS_PER_H2)
    
    if keywords_count > 10:
        words_max = int(words_max * 1.2)
    elif keywords_count < 5:
        words_min = int(words_min * 0.9)
    
    if is_final:
        words_max = int(words_max * 1.1)
    
    return {
        "profile": best_config["profile"],
        "words_min": words_min,
        "words_max": words_max,
        "paragraphs_min": para_min,
        "paragraphs_max": para_max,
        "score": 50,
        "factors": {"fallback": True, "h2_count": h2_count},
        "reasoning": [f"Batch z {h2_count} H2 × {MIN_WORDS_PER_H2} min = {words_min}-{words_max} słów, {para_min}-{para_max} akapitów"]
    }


# ================================================================
# 🔧 HELPERS
# ================================================================

def distribute_items(items: List, num_buckets: int) -> List[List]:
    """Rozdziela elementy równomiernie na buckety."""
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
    """Rozdziela słowa kluczowe na batche Z GÓRY."""
    distribution = {}
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        is_main = meta.get("is_main_keyword", False)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        per_batch = [0] * total_batches
        
        if is_main or keyword.lower() == main_keyword.lower():
            per_batch_target = max(2, math.ceil(target_max / total_batches))
            for i in range(total_batches):
                per_batch[i] = per_batch_target
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
            mid_batch = total_batches // 2
            per_batch[mid_batch] = 1
        
        else:
            if target_max <= total_batches:
                for i in range(min(target_max, total_batches)):
                    per_batch[i] = 1
            else:
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


# ================================================================
# 🎯 GŁÓWNA FUNKCJA - CREATE ARTICLE PLAN
# ================================================================

def convert_semantic_plan_to_distribution(
    semantic_plan: Dict,
    keywords_state: Dict,
    total_batches: int
) -> Dict[str, Dict]:
    """
    🆕 v36.0: Konwertuje semantic_keyword_plan na format keywords_distribution.
    
    semantic_plan.batch_plans[i].assigned_keywords → per_batch array
    """
    distribution = {}
    
    batch_plans = semantic_plan.get("batch_plans", [])
    universal_keywords = set(semantic_plan.get("universal_keywords", []))
    keyword_assignments = semantic_plan.get("keyword_assignments", {})
    
    # Zbierz wszystkie keywords z keywords_state
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
            
        kw_type = meta.get("type", "BASIC").upper()
        is_main = meta.get("is_main_keyword", False)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        per_batch = [0] * total_batches
        
        # Sprawdź czy keyword jest uniwersalny
        if keyword in universal_keywords or keyword.lower() in [u.lower() for u in universal_keywords]:
            # Uniwersalne: rozłóż równomiernie
            per_batch_target = max(1, math.ceil(target_max / total_batches))
            for i in range(total_batches):
                per_batch[i] = per_batch_target
            # Dostosuj do target_max
            current_sum = sum(per_batch)
            if current_sum > target_max:
                diff = current_sum - target_max
                for i in range(total_batches - 1, -1, -1):
                    reduce = min(diff, per_batch[i] - 1)
                    per_batch[i] -= reduce
                    diff -= reduce
                    if diff <= 0:
                        break
        else:
            # Sprawdź przypisanie z semantic_plan
            assignment = keyword_assignments.get(keyword) or keyword_assignments.get(keyword.lower())
            
            if assignment and assignment.get("batch"):
                # Przypisany do konkretnego batcha
                assigned_batch = assignment["batch"] - 1  # 0-indexed
                if 0 <= assigned_batch < total_batches:
                    # Daj wszystkie użycia w przypisanym batchu
                    per_batch[assigned_batch] = target_max
                else:
                    # Fallback: ostatni batch
                    per_batch[total_batches - 1] = target_max
            else:
                # Brak przypisania - szukaj w batch_plans
                found = False
                for bp in batch_plans:
                    batch_idx = bp.get("batch_number", 1) - 1
                    assigned_kws = bp.get("assigned_keywords", [])
                    if keyword in assigned_kws or keyword.lower() in [k.lower() for k in assigned_kws]:
                        if 0 <= batch_idx < total_batches:
                            per_batch[batch_idx] = target_max
                            found = True
                            break
                
                if not found:
                    # Fallback: hash distribution
                    batch_idx = hash(keyword) % total_batches
                    per_batch[batch_idx] = target_max
        
        distribution[keyword] = {
            "total_target": target_max,
            "target_min": target_min,
            "type": kw_type,
            "is_main": is_main,
            "per_batch": per_batch
        }
    
    return distribution


def create_article_plan(
    h2_structure: List[str],
    keywords_state: Dict,
    main_keyword: str,
    target_length: int = 3000,
    ngrams: List[str] = None,
    entities: List[Dict] = None,
    paa_questions: List[str] = None,
    max_batches: int = 6,
    semantic_keyword_plan: Dict = None  # 🆕 v36.0
) -> ArticlePlan:
    """
    🎯 Tworzy kompletny plan artykułu Z GÓRY.
    
    v28.1: DYNAMICZNE DŁUGOŚCI na podstawie:
    - Typu H2 (pytające/instrukcja/definicja)
    - N-gramów powiązanych z sekcją
    - Encji do zdefiniowania
    - Keywords do wplecenia
    - PAA match (snippet potential)
    
    🆕 v36.0: Jeśli semantic_keyword_plan jest dostępny, używa go
    zamiast mechanicznego distribute_keywords().
    """
    ngrams = ngrams or []
    entities = entities or []
    paa_questions = paa_questions or []
    
    # 1. OBLICZ LICZBĘ BATCHÓW
    num_h2 = len(h2_structure)
    
    if num_h2 <= 3:
        total_batches = 2
    elif num_h2 <= 5:
        total_batches = 3
    elif num_h2 <= 8:
        total_batches = 4
    else:
        total_batches = min(max_batches, math.ceil(num_h2 / 2))
    
    # 2. ROZDZIEL H2 NA BATCHE
    h2_per_batch = distribute_items(h2_structure, total_batches)
    
    # 3. ROZDZIEL KEYWORDS NA BATCHE
    # 🆕 v36.0: Użyj semantic_keyword_plan jeśli dostępny
    if semantic_keyword_plan and semantic_keyword_plan.get("batch_plans"):
        print(f"[BATCH_PLANNER] 🎯 Using semantic_keyword_plan for keyword distribution")
        keywords_distribution = convert_semantic_plan_to_distribution(
            semantic_plan=semantic_keyword_plan,
            keywords_state=keywords_state,
            total_batches=total_batches
        )
    else:
        keywords_distribution = distribute_keywords(keywords_state, total_batches, main_keyword)
    
    # 4. ROZDZIEL N-GRAMY NA BATCHE
    ngrams_per_batch = distribute_items(ngrams, total_batches)
    
    # 5. OBLICZ KEYWORDS COUNT PER BATCH
    keywords_per_batch_list = []
    for i in range(total_batches):
        batch_keywords = []
        for kw, dist in keywords_distribution.items():
            if i < len(dist["per_batch"]) and dist["per_batch"][i] > 0:
                batch_keywords.append({
                    "keyword": kw,
                    "uses_this_batch": dist["per_batch"][i]
                })
        keywords_per_batch_list.append(batch_keywords)
    
    # 6. OBLICZ DYNAMICZNE DŁUGOŚCI
    batches: List[BatchPlan] = []
    
    for i in range(total_batches):
        batch_num = i + 1
        is_intro = (i == 0)
        is_final = (i == total_batches - 1)
        
        h2_list = h2_per_batch[i] if i < len(h2_per_batch) else []
        batch_keywords = keywords_per_batch_list[i]
        batch_ngrams = ngrams_per_batch[i] if i < len(ngrams_per_batch) else []
        
        if is_intro:
            h2_title = "INTRO"
        elif h2_list:
            h2_title = " | ".join(h2_list)
        else:
            h2_title = f"Batch {batch_num}"
        
        # OBLICZ ZŁOŻONOŚĆ
        if COMPLEXITY_AVAILABLE:
            complexity = calculate_batch_complexity(
                h2_title=h2_title,
                ngrams=ngrams,
                entities=entities,
                keywords_for_batch=batch_keywords,
                paa_questions=paa_questions,
                is_intro=is_intro,
                is_final=is_final
            )
            
            length_info = {
                "profile": complexity.profile.name,
                "words_min": complexity.profile.words_min,
                "words_max": complexity.profile.words_max,
                "paragraphs_min": complexity.profile.paragraphs_min,
                "paragraphs_max": complexity.profile.paragraphs_max,
                "score": complexity.score,
                "factors": complexity.factors,
                "reasoning": complexity.reasoning,
                "snippet_required": complexity.profile.snippet_required
            }
        else:
            length_info = calculate_length_fallback(
                h2_sections=h2_list,
                keywords_count=len(batch_keywords),
                is_intro=is_intro,
                is_final=is_final
            )
            length_info["snippet_required"] = not is_intro
        
        # TYP BATCHA I NOTATKI
        if is_intro:
            batch_type = "INTRO"
            notes = "INTRO: Snippet 40-60 słów, fraza główna w 1. zdaniu"
        elif is_final:
            batch_type = "FINAL"
            notes = "OSTATNI: Użyj wszystkie pozostałe CRITICAL keywords"
        else:
            batch_type = "CONTENT"
            notes = " | ".join(length_info.get("reasoning", [])[:2])
        
        # KEYWORDS BUDGET
        keywords_budget = {}
        for kw, dist in keywords_distribution.items():
            if i < len(dist["per_batch"]):
                keywords_budget[kw] = dist["per_batch"][i]
        
        batch_plan = BatchPlan(
            batch_number=batch_num,
            batch_type=batch_type,
            h2_sections=h2_list,
            target_words_min=length_info["words_min"],
            target_words_max=length_info["words_max"],
            target_paragraphs_min=length_info["paragraphs_min"],
            target_paragraphs_max=length_info["paragraphs_max"],
            length_profile=length_info["profile"],
            complexity_score=length_info["score"],
            complexity_factors=length_info["factors"],
            complexity_reasoning=length_info.get("reasoning", []),
            keywords_budget=keywords_budget,
            ngrams_to_use=batch_ngrams,
            snippet_required=length_info.get("snippet_required", True),
            notes=notes
        )
        batches.append(batch_plan)
    
    actual_total = sum(
        (b.target_words_min + b.target_words_max) // 2 
        for b in batches
    )
    
    return ArticlePlan(
        total_batches=total_batches,
        total_target_words=actual_total,
        batches=batches,
        keywords_distribution=keywords_distribution,
        main_keyword=main_keyword,
        h2_structure=h2_structure
    )


# ================================================================
# 🔧 GET BATCH INSTRUCTIONS
# ================================================================

def get_batch_instructions(
    plan: ArticlePlan,
    batch_number: int,
    current_keywords_state: Dict = None
) -> Dict:
    """Generuje instrukcje dla konkretnego batcha."""
    if batch_number < 1 or batch_number > plan.total_batches:
        return {"error": f"Invalid batch number: {batch_number}"}
    
    batch_plan = plan.batches[batch_number - 1]
    
    keywords_instructions = []
    critical_keywords = []
    high_priority = []
    
    for keyword, budget in batch_plan.keywords_budget.items():
        if budget <= 0:
            continue
        
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
    
    prompt_lines = []
    prompt_lines.append(f"📋 BATCH #{batch_number} z {plan.total_batches}")
    prompt_lines.append(f"📝 Typ: {batch_plan.batch_type}")
    prompt_lines.append(f"📏 Słowa: {batch_plan.target_words_min}-{batch_plan.target_words_max}")
    prompt_lines.append(f"📄 Akapity: {batch_plan.target_paragraphs_min}-{batch_plan.target_paragraphs_max}")
    prompt_lines.append(f"🎯 Score: {batch_plan.complexity_score}/100 → {batch_plan.length_profile.upper()}")
    
    if batch_plan.snippet_required:
        prompt_lines.append(f"⚡ SNIPPET: Pierwszych 40-60 słów = bezpośrednia odpowiedź!")
    
    prompt_lines.append("")
    
    if batch_plan.complexity_reasoning:
        prompt_lines.append("📊 Dlaczego taka długość:")
        for reason in batch_plan.complexity_reasoning[:3]:
            prompt_lines.append(f"   • {reason}")
        prompt_lines.append("")
    
    if batch_plan.notes:
        prompt_lines.append(f"⚠️ {batch_plan.notes}")
        prompt_lines.append("")
    
    if critical_keywords:
        prompt_lines.append("=" * 50)
        for kw in critical_keywords:
            prompt_lines.append(f"🔴 FRAZA GŁÓWNA: \"{kw['keyword']}\"")
            prompt_lines.append(f"   → użyj {kw['suggested']}x w tym batchu")
        prompt_lines.append("=" * 50)
        prompt_lines.append("")
    
    if high_priority:
        prompt_lines.append("🟠 WPLEĆ (priorytet):")
        for kw in high_priority:
            prompt_lines.append(f"   • {kw['keyword']}: {kw['suggested']}x")
        prompt_lines.append("")
    
    if batch_plan.h2_sections:
        prompt_lines.append("✏️ H2 DO NAPISANIA:")
        # 🆕 v41.2: Różna liczba akapitów dla każdej H2 (2-4, różne)
        paragraph_options = [2, 3, 4, 3, 2, 4]  # Cykliczna lista dla różnorodności
        for i, h2 in enumerate(batch_plan.h2_sections):
            para_count = paragraph_options[i % len(paragraph_options)]
            prompt_lines.append(f"   • {h2}")
            prompt_lines.append(f"     → {para_count} akapity (różna liczba dla każdej sekcji!)")
        prompt_lines.append("")
        prompt_lines.append("   ⚠️ WAŻNE: Każda sekcja H2 MUSI mieć INNĄ liczbę akapitów!")
        prompt_lines.append("")
    
    if batch_plan.ngrams_to_use:
        prompt_lines.append("📝 N-GRAMY:")
        for ng in batch_plan.ngrams_to_use[:5]:
            prompt_lines.append(f"   • \"{ng}\"")
        prompt_lines.append("")
    
    # 🆕 v41.2: Wygeneruj różne liczby akapitów dla każdej H2
    paragraph_options = [2, 3, 4, 3, 2, 4]
    h2_paragraph_counts = {}
    for i, h2 in enumerate(batch_plan.h2_sections):
        h2_paragraph_counts[h2] = paragraph_options[i % len(paragraph_options)]
    
    return {
        "batch_number": batch_number,
        "batch_type": batch_plan.batch_type,
        "target_words": {"min": batch_plan.target_words_min, "max": batch_plan.target_words_max},
        "target_paragraphs": {"min": batch_plan.target_paragraphs_min, "max": batch_plan.target_paragraphs_max},
        "length_profile": batch_plan.length_profile,
        "complexity_score": batch_plan.complexity_score,
        "complexity_reasoning": batch_plan.complexity_reasoning,
        "snippet_required": batch_plan.snippet_required,
        "h2_sections": batch_plan.h2_sections,
        "h2_paragraph_counts": h2_paragraph_counts,  # 🆕 v41.2
        "keywords": {"critical": critical_keywords, "high_priority": high_priority, "normal": keywords_instructions},
        "ngrams": batch_plan.ngrams_to_use,
        "notes": batch_plan.notes,
        "gpt_prompt": "\n".join(prompt_lines)
    }


def update_plan_after_batch(
    plan: ArticlePlan,
    batch_number: int,
    actual_keywords_used: Dict[str, int]
) -> ArticlePlan:
    """Aktualizuje plan po napisaniu batcha."""
    if batch_number >= plan.total_batches:
        return plan
    
    for keyword, dist in plan.keywords_distribution.items():
        planned = dist["per_batch"][batch_number - 1]
        actual = actual_keywords_used.get(keyword, 0)
        
        if actual < planned:
            deficit = planned - actual
            remaining_batches = plan.total_batches - batch_number
            
            if remaining_batches > 0:
                extra_per_batch = math.ceil(deficit / remaining_batches)
                for i in range(batch_number, plan.total_batches):
                    dist["per_batch"][i] += extra_per_batch
                    deficit -= extra_per_batch
                    if deficit <= 0:
                        break
    
    for i, batch in enumerate(plan.batches):
        if i >= batch_number:
            new_budget = {}
            for kw, dist in plan.keywords_distribution.items():
                if i < len(dist["per_batch"]):
                    new_budget[kw] = dist["per_batch"][i]
            batch.keywords_budget = new_budget
    
    return plan
