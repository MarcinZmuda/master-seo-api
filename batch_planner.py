"""
BATCH PLANNER v2.0 - Wrapper
=============================
Deleguje do dynamic_batch_planner (token budgeting) + batch_complexity.
Interfejs: create_article_plan() / create_article_plan_fast()
Zgodnosc: project_routes.py:174

Fix #1 v4.2: Zastepuje stary batch_planner.py ktory zawieral tresc Dockerfile.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Fix #32: resilient import — nie lamie feature_flags jesli dynamic_batch_planner ma blad
PLANNER_AVAILABLE = False
try:
    from dynamic_batch_planner import (
        DynamicBatchPlanner, create_dynamic_batch_plan
    )
    PLANNER_AVAILABLE = True
except ImportError as e:
    print(f"[BATCH_PLANNER] ⚠️ dynamic_batch_planner not available: {e}")
    DynamicBatchPlanner = None
    create_dynamic_batch_plan = None

# Import scoringu
COMPLEXITY_AVAILABLE = False
try:
    from batch_complexity import calculate_batch_complexity
    COMPLEXITY_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ArticlePlan:
    """Wynik planowania - zgodny z project_routes.py."""
    total_batches: int
    total_target_words: int
    plan_dict: dict

    def to_dict(self):
        return self.plan_dict


def create_article_plan(
    h2_structure: List,
    keywords_state: Dict,
    main_keyword: str = '',
    target_length: int = 3500,
    ngrams: List[str] = None,
    entities: List[str] = None,
    paa_questions: List[str] = None,
    max_batches: int = 9,
    semantic_keyword_plan: dict = None,
    **kwargs
) -> ArticlePlan:
    """Planowanie batchow - deleguje do DynamicBatchPlanner."""
    # Normalizuj h2_structure do List[str]
    h2_list = [
        h.get('h2', h) if isinstance(h, dict) else str(h)
        for h in h2_structure
    ]

    # S1 data z ngrams/entities
    s1_data = {}
    if ngrams:
        s1_data['ngrams'] = ngrams
    if entities:
        s1_data['entities'] = entities
    if paa_questions:
        s1_data['paa'] = paa_questions

    # Fix #32: graceful fallback jesli planner niedostepny
    if not PLANNER_AVAILABLE or create_dynamic_batch_plan is None:
        # Minimalny plan: 1 batch per H2
        batches = [{"batch_idx": i+1, "sections": [{"h2": h, "target_words": target_length // len(h2_list)}]} for i, h in enumerate(h2_list)]
        plan_dict = {"total_batches": len(batches), "batches": batches}
        return ArticlePlan(total_batches=len(batches), total_target_words=target_length, plan_dict=plan_dict)

    # Deleguj do DynamicBatchPlanner (token budgeting)
    plan_dict = create_dynamic_batch_plan(
        h2_structure=h2_list,
        semantic_plan=semantic_keyword_plan,
        s1_data=s1_data,
        target_length=target_length
    )

    # Enrichment z batch_complexity (jesli dostepny)
    if COMPLEXITY_AVAILABLE:
        for batch in plan_dict.get('batches', []):
            for section in batch.get('sections', []):
                cr = calculate_batch_complexity(
                    h2_title=section.get('h2', ''),
                    ngrams=ngrams or [],
                    entities=entities or [],
                    keywords_for_batch=section.get('assigned_keywords', []),
                    paa_questions=paa_questions or [],
                    is_intro=(batch.get('batch_type') == 'INTRO'),
                    is_final=(batch.get('batch_type') == 'FINAL')
                )
                # ComplexityResult: .score, .profile, .factors, .reasoning
                section['complexity_score'] = cr.score
                section['length_profile'] = cr.profile.name

    total_words = sum(
        s.get('target_words', 400)
        for b in plan_dict.get('batches', [])
        for s in b.get('sections', [])
    )

    return ArticlePlan(
        total_batches=plan_dict.get('total_batches', len(h2_list)),
        total_target_words=total_words,
        plan_dict=plan_dict
    )


def create_article_plan_fast(
    h2_structure: List,
    keywords_state: Dict,
    target_length: int = 2000,
    **kwargs
) -> ArticlePlan:
    """Fast mode - max 3 batche, krotszy artykul."""
    return create_article_plan(
        h2_structure=h2_structure[:3],
        keywords_state=keywords_state,
        target_length=target_length,
        max_batches=3,
        **kwargs
    )
