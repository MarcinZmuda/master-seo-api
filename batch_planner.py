"""
===============================================================================
📋 BATCH PLANNER v42.0 - DYNAMICZNE DŁUGOŚCI + SMART SUB-BATCHE
===============================================================================
v42.0: NOWE FUNKCJE:

🆕 DYNAMIC SUB-BATCH SPLITTER:
  - Automatyczny podział H2 gdy za dużo elementów
  - Jeśli H2 ma >8 fraz lub >3 encje → dzieli na sub-batche
  - Batch 3 → Batch 3A + Batch 3B
  - Agent ma więcej miejsca na naturalne użycie fraz

🆕 OVERFLOW BUFFER INTEGRATION:
  - Automatyczna sekcja FAQ dla "sierocych" fraz
  - Frazy z niskim relevance trafiają do FAQ na końcu
  - 85%+ pokrycia bez psucia narracji

ZACHOWANE Z v41.2:
  - 1 H2 = 1 batch (divide and conquer)
  - Długość batcha zależy od: typu H2, n-gramów, encji, keywords

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
# 🆕 v42.0: DYNAMIC SUB-BATCH SPLITTER
# ================================================================
try:
    from dynamic_sub_batch import (
        split_h2_if_needed,
        analyze_h2_density,
        process_batch_plan_with_splitting,
        SubBatch,
        SplitResult
    )
    DYNAMIC_SUB_BATCH_AVAILABLE = True
    print("[BATCH_PLANNER] ✅ dynamic_sub_batch v1.0 loaded (auto-split)")
except ImportError:
    DYNAMIC_SUB_BATCH_AVAILABLE = False
    print("[BATCH_PLANNER] ⚠️ dynamic_sub_batch not available")
    
    def split_h2_if_needed(h2_title, batch_number, assigned_phrases, assigned_entities, assigned_triplets, **kwargs):
        """Fallback - bez podziału."""
        class FallbackResult:
            needs_split = False
            sub_batches = []
            stats = {}
        return FallbackResult()

# ================================================================
# 🆕 v42.0: OVERFLOW BUFFER (Auto FAQ)
# ================================================================
try:
    from overflow_buffer import (
        create_overflow_buffer,
        add_faq_batch_if_needed,
        identify_orphan_phrases,
        OverflowBuffer
    )
    OVERFLOW_BUFFER_AVAILABLE = True
    print("[BATCH_PLANNER] ✅ overflow_buffer v1.0 loaded (auto FAQ)")
except ImportError:
    OVERFLOW_BUFFER_AVAILABLE = False
    print("[BATCH_PLANNER] ⚠️ overflow_buffer not available")
    
    def create_overflow_buffer(**kwargs):
        class EmptyBuffer:
            orphan_phrases = []
            faq_items = []
        return EmptyBuffer()
    
    def add_faq_batch_if_needed(batch_plan, buffer, **kwargs):
        return batch_plan


# ================================================================
# 📋 STRUKTURY DANYCH
# ================================================================

@dataclass
class BatchPlan:
    """Plan pojedynczego batcha."""
    batch_number: int  # Może być "3A", "3B" dla sub-batchów
    batch_type: str  # "INTRO", "CONTENT", "FINAL", "FAQ"
    h2_sections: List[str]
    target_words_min: int
    target_words_max: int
    target_paragraphs_min: int
    target_paragraphs_max: int
    length_profile: str
    complexity_score: int
    complexity_factors: Dict[str, Any]
    complexity_reasoning: List[str]
    keywords_budget: Dict[str, int]
    ngrams_to_use: List[str]
    snippet_required: bool
    notes: str = ""
    # 🆕 v42.0: Sub-batch fields
    is_sub_batch: bool = False
    parent_batch: Optional[int] = None
    h3_title: Optional[str] = None
    assigned_phrases: List[Dict] = field(default_factory=list)
    assigned_entities: List[Dict] = field(default_factory=list)
    assigned_triplets: List[Dict] = field(default_factory=list)
    
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
    # 🆕 v42.0: Overflow buffer
    overflow_buffer: Optional[Any] = None
    has_faq_section: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "total_batches": self.total_batches,
            "total_target_words": self.total_target_words,
            "batches": [b.to_dict() for b in self.batches],
            "keywords_distribution": self.keywords_distribution,
            "main_keyword": self.main_keyword,
            "h2_structure": self.h2_structure,
            "has_faq_section": self.has_faq_section
        }


# ================================================================
# 🔧 FALLBACK - gdy batch_complexity nie jest dostępny
# ================================================================

H2_TYPE_FALLBACK = {
    "tutorial": {
        "patterns": ["krok po kroku", "poradnik", "instrukcja", "jak zrobić", "jak wykonać"],
        "profile": "long",
        "words": (500, 700),
        "paragraphs": (3, 4)
    },
    "definition": {
        "patterns": ["co to", "czym jest", "definicja", "co oznacza"],
        "profile": "short",
        "words": (300, 450),
        "paragraphs": (2, 3)
    },
    "yes_no": {
        "patterns": ["czy można", "czy warto", "czy trzeba", "czy należy"],
        "profile": "short",
        "words": (300, 450),
        "paragraphs": (2, 3)
    },
    "comparison": {
        "patterns": ["vs", "porównanie", "różnice", "co lepsze"],
        "profile": "long",
        "words": (500, 700),
        "paragraphs": (3, 4)
    },
    "list": {
        "patterns": ["najlepsze", "top", "ranking", "rodzaje", "typy"],
        "profile": "extended",
        "words": (600, 850),
        "paragraphs": (3, 4)
    },
    "explanation": {
        "patterns": ["jak ", "dlaczego", "w jaki sposób"],
        "profile": "long",
        "words": (500, 700),
        "paragraphs": (3, 4)
    }
}

DEFAULT_FALLBACK = {
    "profile": "medium",
    "words": (400, 600),
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
    MIN_WORDS_PER_H2 = 250
    h2_count = len(h2_sections) if h2_sections else 1
    
    if is_intro:
        base_min = max(300, h2_count * MIN_WORDS_PER_H2)
        base_max = max(450, h2_count * 350)
        return {
            "profile": "intro",
            "words_min": base_min,
            "words_max": base_max,
            "paragraphs_min": max(2, h2_count * 2),
            "paragraphs_max": min(h2_count * 4, 12),
            "score": 30,
            "factors": {"type": "intro", "h2_count": h2_count},
            "reasoning": [f"INTRO: {h2_count} H2 × {MIN_WORDS_PER_H2} słów"]
        }
    
    if h2_sections:
        configs = [detect_h2_type_fallback(h2) for h2 in h2_sections]
        best_config = max(configs, key=lambda c: c["words"][1])
    else:
        best_config = DEFAULT_FALLBACK
    
    words_min, words_max = best_config["words"]
    para_min, para_max = best_config["paragraphs"]
    
    if h2_count > 1:
        h2_multiplier = 1 + (h2_count - 1) * 0.8
        words_min = int(words_min * h2_multiplier)
        words_max = int(words_max * h2_multiplier)
        para_min = h2_count * 2
        para_max = h2_count * 4
    
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
        "reasoning": [f"Batch z {h2_count} H2 × {MIN_WORDS_PER_H2} min = {words_min}-{words_max} słów"]
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


def convert_semantic_plan_to_distribution(
    semantic_plan: Dict,
    keywords_state: Dict,
    total_batches: int
) -> Dict[str, Dict]:
    """Konwertuje semantic_keyword_plan na format keywords_distribution."""
    distribution = {}
    
    batch_plans = semantic_plan.get("batch_plans", [])
    universal_keywords = set(semantic_plan.get("universal_keywords", []))
    keyword_assignments = semantic_plan.get("keyword_assignments", {})
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
            
        kw_type = meta.get("type", "BASIC").upper()
        is_main = meta.get("is_main_keyword", False)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        per_batch = [0] * total_batches
        
        if keyword in universal_keywords or keyword.lower() in [u.lower() for u in universal_keywords]:
            per_batch_target = max(1, math.ceil(target_max / total_batches))
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
        else:
            assignment = keyword_assignments.get(keyword) or keyword_assignments.get(keyword.lower())
            
            if assignment and assignment.get("batch"):
                assigned_batch = assignment["batch"] - 1
                if 0 <= assigned_batch < total_batches:
                    per_batch[assigned_batch] = target_max
                else:
                    per_batch[total_batches - 1] = target_max
            else:
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


# ================================================================
# 🆕 v42.0: DISTRIBUTE H2 - 1 H2 = 1 BATCH + DYNAMIC SPLIT
# ================================================================

def _distribute_h2_one_per_batch(h2_structure: List[str], total_batches: int) -> List[List[str]]:
    """
    Przypisuje H2 do batchów: 1 H2 = 1 batch (gdzie możliwe).
    """
    result = []
    result.append([])  # Batch 0 = intro (bez H2)
    
    num_h2 = len(h2_structure)
    content_batches = total_batches - 1
    
    if num_h2 == 0:
        for _ in range(content_batches):
            result.append([])
        return result
    
    if num_h2 <= content_batches:
        for h2 in h2_structure:
            result.append([h2])
        while len(result) < total_batches:
            result.append([])
    else:
        extra_h2 = num_h2 - content_batches
        h2_idx = 0
        for i in range(content_batches):
            if i < extra_h2:
                result.append([h2_structure[h2_idx], h2_structure[h2_idx + 1]])
                h2_idx += 2
            else:
                if h2_idx < num_h2:
                    result.append([h2_structure[h2_idx]])
                    h2_idx += 1
                else:
                    result.append([])
    
    print(f"[BATCH_PLANNER] H2 distribution: {[len(b) for b in result]} H2 per batch")
    return result


# ================================================================
# 🎯 GŁÓWNA FUNKCJA - CREATE ARTICLE PLAN
# ================================================================

def create_article_plan(
    h2_structure: List[str],
    keywords_state: Dict,
    main_keyword: str,
    target_length: int = 3000,
    ngrams: List[str] = None,
    entities: List[Dict] = None,
    paa_questions: List[str] = None,
    max_batches: int = 6,
    semantic_keyword_plan: Dict = None,
    # 🆕 v42.0: Dodatkowe parametry dla dynamic sub-batch
    phrase_assignments: Dict[str, List[Dict]] = None,
    entity_assignments: Dict[str, List[Dict]] = None,
    triplet_assignments: Dict[str, List[Dict]] = None,
    enable_dynamic_split: bool = True,
    enable_overflow_buffer: bool = True
) -> ArticlePlan:
    """
    🎯 Tworzy kompletny plan artykułu Z GÓRY.
    
    🆕 v42.0 ZMIANY:
    - DYNAMIC SUB-BATCH: Automatyczny podział H2 gdy >8 fraz
    - OVERFLOW BUFFER: Auto FAQ dla sierocych fraz
    """
    ngrams = ngrams or []
    entities = entities or []
    paa_questions = paa_questions or []
    phrase_assignments = phrase_assignments or {}
    entity_assignments = entity_assignments or {}
    triplet_assignments = triplet_assignments or {}
    
    # ================================================================
    # 1. OBLICZ LICZBĘ BATCHÓW (1 H2 = 1 batch)
    # ================================================================
    num_h2 = len(h2_structure)
    total_batches = 1 + num_h2  # intro + H2s
    
    if total_batches < 3:
        total_batches = 3
    if total_batches > 8:
        total_batches = 8
    
    print(f"[BATCH_PLANNER] v42.0: {num_h2} H2 → {total_batches} batchów")
    
    # ================================================================
    # 2. ROZDZIEL H2 NA BATCHE
    # ================================================================
    h2_per_batch = _distribute_h2_one_per_batch(h2_structure, total_batches)
    
    # ================================================================
    # 3. ROZDZIEL KEYWORDS NA BATCHE
    # ================================================================
    if semantic_keyword_plan and semantic_keyword_plan.get("batch_plans"):
        print(f"[BATCH_PLANNER] Using semantic_keyword_plan for keyword distribution")
        keywords_distribution = convert_semantic_plan_to_distribution(
            semantic_plan=semantic_keyword_plan,
            keywords_state=keywords_state,
            total_batches=total_batches
        )
    else:
        keywords_distribution = distribute_keywords(keywords_state, total_batches, main_keyword)
    
    # ================================================================
    # 4. ROZDZIEL N-GRAMY NA BATCHE
    # ================================================================
    ngrams_per_batch = distribute_items(ngrams, total_batches)
    
    # ================================================================
    # 5. OBLICZ KEYWORDS COUNT PER BATCH
    # ================================================================
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
    
    # ================================================================
    # 6. TWÓRZ BATCHE (z dynamicznym podziałem)
    # ================================================================
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
        
        # ============================================================
        # 🆕 v42.0: DYNAMIC SUB-BATCH SPLITTING
        # ============================================================
        needs_split = False
        sub_batches = []
        
        if enable_dynamic_split and DYNAMIC_SUB_BATCH_AVAILABLE and h2_list:
            for h2 in h2_list:
                h2_phrases = phrase_assignments.get(h2, [])
                h2_entities = entity_assignments.get(h2, [])
                h2_triplets = triplet_assignments.get(h2, [])
                
                split_result = split_h2_if_needed(
                    h2_title=h2,
                    batch_number=batch_num,
                    assigned_phrases=h2_phrases,
                    assigned_entities=h2_entities,
                    assigned_triplets=h2_triplets,
                    target_words_per_batch=(400, 600)
                )
                
                if split_result.needs_split:
                    needs_split = True
                    sub_batches.extend(split_result.sub_batches)
                    print(f"[BATCH_PLANNER] 🔀 H2 '{h2[:30]}...' split into {len(split_result.sub_batches)} sub-batches")
        
        if needs_split and sub_batches:
            # Dodaj sub-batche zamiast jednego batcha
            for sb in sub_batches:
                sub_batch_plan = BatchPlan(
                    batch_number=sb.sub_batch_id,
                    batch_type="CONTENT",
                    h2_sections=[sb.parent_h2],
                    target_words_min=sb.word_target[0],
                    target_words_max=sb.word_target[1],
                    target_paragraphs_min=2,
                    target_paragraphs_max=3,
                    length_profile="sub_batch",
                    complexity_score=50,
                    complexity_factors={"sub_batch": True},
                    complexity_reasoning=[f"Sub-batch for H2: {sb.parent_h2[:30]}"],
                    keywords_budget={},
                    ngrams_to_use=batch_ngrams,
                    snippet_required=True,
                    notes=f"Sub-batch: {sb.h3_title or 'Part'}",
                    is_sub_batch=True,
                    parent_batch=batch_num,
                    h3_title=sb.h3_title,
                    assigned_phrases=sb.assigned_phrases,
                    assigned_entities=sb.assigned_entities,
                    assigned_triplets=sb.assigned_triplets
                )
                batches.append(sub_batch_plan)
        else:
            # Standardowy batch (bez podziału)
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
            
            if is_intro:
                batch_type = "INTRO"
                notes = "INTRO: Snippet 40-60 słów, fraza główna w 1. zdaniu"
            elif is_final:
                batch_type = "FINAL"
                notes = "OSTATNI: Użyj wszystkie pozostałe CRITICAL keywords"
            else:
                batch_type = "CONTENT"
                notes = " | ".join(length_info.get("reasoning", [])[:2])
            
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
    
    # ================================================================
    # 🆕 v42.0: OVERFLOW BUFFER (Auto FAQ)
    # ================================================================
    overflow_buffer = None
    has_faq_section = False
    
    if enable_overflow_buffer and OVERFLOW_BUFFER_AVAILABLE:
        overflow_buffer = create_overflow_buffer(
            keywords_state=keywords_state,
            phrase_assignments=phrase_assignments,
            h2_structure=h2_structure,
            main_keyword=main_keyword,
            domain="prawo"  # TODO: wykryć domenę
        )
        
        if overflow_buffer.faq_items:
            print(f"[BATCH_PLANNER] 📦 Overflow buffer: {len(overflow_buffer.orphan_phrases)} orphan phrases → {len(overflow_buffer.faq_items)} FAQ items")
            
            # Dodaj batch FAQ
            faq_batch = BatchPlan(
                batch_number=len(batches) + 1,
                batch_type="FAQ",
                h2_sections=[overflow_buffer.section_title],
                target_words_min=150,
                target_words_max=300,
                target_paragraphs_min=len(overflow_buffer.faq_items),
                target_paragraphs_max=len(overflow_buffer.faq_items) + 2,
                length_profile="faq",
                complexity_score=30,
                complexity_factors={"faq_items": len(overflow_buffer.faq_items)},
                complexity_reasoning=["Auto-generated FAQ for orphan phrases"],
                keywords_budget={},
                ngrams_to_use=[],
                snippet_required=False,
                notes="FAQ dla sierocych fraz"
            )
            batches.append(faq_batch)
            has_faq_section = True
    
    # ================================================================
    # 7. OBLICZ TOTAL WORDS
    # ================================================================
    actual_total = sum(
        (b.target_words_min + b.target_words_max) // 2 
        for b in batches
    )
    
    return ArticlePlan(
        total_batches=len(batches),
        total_target_words=actual_total,
        batches=batches,
        keywords_distribution=keywords_distribution,
        main_keyword=main_keyword,
        h2_structure=h2_structure,
        overflow_buffer=overflow_buffer,
        has_faq_section=has_faq_section
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
    # Znajdź batch (może być sub-batch jak "3A")
    batch_plan = None
    for b in plan.batches:
        if str(b.batch_number) == str(batch_number):
            batch_plan = b
            break
    
    if not batch_plan:
        return {"error": f"Invalid batch number: {batch_number}"}
    
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
    prompt_lines.append(f"📋 BATCH #{batch_plan.batch_number} z {plan.total_batches}")
    prompt_lines.append(f"📝 Typ: {batch_plan.batch_type}")
    prompt_lines.append(f"📏 Słowa: {batch_plan.target_words_min}-{batch_plan.target_words_max}")
    prompt_lines.append(f"📄 Akapity: {batch_plan.target_paragraphs_min}-{batch_plan.target_paragraphs_max}")
    prompt_lines.append(f"🎯 Score: {batch_plan.complexity_score}/100 → {batch_plan.length_profile.upper()}")
    
    # 🆕 v42.0: Info o sub-batch
    if batch_plan.is_sub_batch:
        prompt_lines.append(f"🔀 SUB-BATCH: część {batch_plan.batch_number[-1]} z batcha {batch_plan.parent_batch}")
        if batch_plan.h3_title:
            prompt_lines.append(f"   H3: {batch_plan.h3_title}")
    
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
    
    if batch_plan.h2_sections:
        prompt_lines.append("✏️ H2 DO NAPISANIA:")
        paragraph_options = [2, 3, 4, 3, 2, 4]
        for i, h2 in enumerate(batch_plan.h2_sections):
            para_count = paragraph_options[i % len(paragraph_options)]
            prompt_lines.append(f"   • {h2}")
            prompt_lines.append(f"     → {para_count} akapity")
        prompt_lines.append("")
    
    # 🆕 v42.0: Assigned elements for sub-batch
    if batch_plan.is_sub_batch:
        if batch_plan.assigned_phrases:
            prompt_lines.append("📍 FRAZY PRZYPISANE (MUST):")
            for p in batch_plan.assigned_phrases[:5]:
                prompt_lines.append(f"   • \"{p.get('keyword', p)}\"")
            prompt_lines.append("")
        
        if batch_plan.assigned_triplets:
            prompt_lines.append("🔗 TRIPLETY PRZYPISANE:")
            for t in batch_plan.assigned_triplets[:2]:
                s = t.get("subject", "")
                v = t.get("verb", "")
                o = t.get("object", "")
                prompt_lines.append(f"   • {s} → {v} → {o}")
            prompt_lines.append("")
    
    return {
        "batch_number": batch_plan.batch_number,
        "batch_type": batch_plan.batch_type,
        "is_sub_batch": batch_plan.is_sub_batch,
        "h3_title": batch_plan.h3_title,
        "target_words": {"min": batch_plan.target_words_min, "max": batch_plan.target_words_max},
        "target_paragraphs": {"min": batch_plan.target_paragraphs_min, "max": batch_plan.target_paragraphs_max},
        "length_profile": batch_plan.length_profile,
        "complexity_score": batch_plan.complexity_score,
        "h2_sections": batch_plan.h2_sections,
        "assigned_phrases": batch_plan.assigned_phrases,
        "assigned_triplets": batch_plan.assigned_triplets,
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
    # Znajdź indeks batcha
    batch_idx = None
    for i, b in enumerate(plan.batches):
        if str(b.batch_number) == str(batch_number):
            batch_idx = i
            break
    
    if batch_idx is None or batch_idx >= len(plan.batches) - 1:
        return plan
    
    for keyword, dist in plan.keywords_distribution.items():
        planned = dist["per_batch"][batch_idx] if batch_idx < len(dist["per_batch"]) else 0
        actual = actual_keywords_used.get(keyword, 0)
        
        if actual < planned:
            deficit = planned - actual
            remaining_batches = len(plan.batches) - batch_idx - 1
            
            if remaining_batches > 0:
                extra_per_batch = math.ceil(deficit / remaining_batches)
                for i in range(batch_idx + 1, len(plan.batches)):
                    if i < len(dist["per_batch"]):
                        dist["per_batch"][i] += extra_per_batch
                        deficit -= extra_per_batch
                        if deficit <= 0:
                            break
    
    return plan


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'create_article_plan',
    'get_batch_instructions',
    'update_plan_after_batch',
    'ArticlePlan',
    'BatchPlan',
    'DYNAMIC_SUB_BATCH_AVAILABLE',
    'OVERFLOW_BUFFER_AVAILABLE'
]
