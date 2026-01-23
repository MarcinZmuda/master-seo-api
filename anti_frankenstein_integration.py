"""
🧟 ANTI-FRANKENSTEIN INTEGRATION v1.0
Centralny moduł integrujący wszystkie rozwiązania spójności

Integruje:
1. dynamic_batch_planner - Token Budgeting
2. article_memory - Running Summary + Thesis
3. style_analyzer - Persona Fingerprint  
4. soft_cap_limiter - Elastyczne limity

Autor: SEO Master API v36.2
"""

import json
from typing import Dict, List, Optional, Tuple, Any


# ============================================
# IMPORTS Z NOWYCH MODUŁÓW
# ============================================

def _safe_import(module_name: str, func_name: str):
    """Bezpieczny import z fallbackiem"""
    try:
        module = __import__(module_name)
        return getattr(module, func_name, None)
    except ImportError:
        return None


# Dynamic Batch Planner
create_dynamic_batch_plan = _safe_import('dynamic_batch_planner', 'create_dynamic_batch_plan')
get_batch_sections_for_pre_batch = _safe_import('dynamic_batch_planner', 'get_batch_sections_for_pre_batch')

# Article Memory
create_article_memory = _safe_import('article_memory', 'create_article_memory')
update_article_memory = _safe_import('article_memory', 'update_article_memory')
ArticleMemory = _safe_import('article_memory', 'ArticleMemory')

# Style Analyzer
analyze_style = _safe_import('style_analyzer', 'analyze_style')
check_style_consistency = _safe_import('style_analyzer', 'check_style_consistency')
generate_style_prompt = _safe_import('style_analyzer', 'generate_style_prompt')

# Soft Cap Limiter
create_soft_cap_validator = _safe_import('soft_cap_limiter', 'create_soft_cap_validator')
validate_with_soft_caps = _safe_import('soft_cap_limiter', 'validate_with_soft_caps')
get_flexible_limits = _safe_import('soft_cap_limiter', 'get_flexible_limits')


# ============================================
# 1. INTEGRACJA W CREATE_PROJECT
# ============================================

def enhance_project_with_anti_frankenstein(
    project_data: dict,
    h2_structure: List[str],
    keywords_state: dict,
    s1_data: dict,
    main_keyword: str,
    target_length: int = 3500
) -> dict:
    """
    Rozszerz dane projektu o komponenty anti-Frankenstein.
    
    Wywoływane w create_project() po utworzeniu podstawowych danych.
    
    Dodaje:
    - dynamic_batch_plan (Token Budgeting)
    - article_memory (pusta, gotowa do aktualizacji)
    - style_fingerprint (pusta, zostanie wypełniona po batch 1)
    - soft_cap_limits (elastyczne limity dla fraz)
    
    Args:
        project_data: Istniejące dane projektu
        h2_structure: Lista nagłówków H2
        keywords_state: Metadane fraz
        s1_data: Dane z S1
        main_keyword: Główna fraza
        target_length: Docelowa długość artykułu
        
    Returns:
        Rozszerzone project_data
    """
    project_id = project_data.get("project_id", "unknown")
    
    # ================================================================
    # 1. DYNAMIC BATCH PLAN (Token Budgeting)
    # ================================================================
    if create_dynamic_batch_plan:
        try:
            semantic_plan = project_data.get("semantic_keyword_plan", {})
            
            dynamic_plan = create_dynamic_batch_plan(
                h2_structure=h2_structure,
                semantic_plan=semantic_plan,
                s1_data=s1_data,
                target_length=target_length
            )
            
            project_data["dynamic_batch_plan"] = dynamic_plan
            project_data["total_planned_batches"] = dynamic_plan.get("total_batches", len(h2_structure))
            
            print(f"[ANTI-FRANKENSTEIN] ✅ Dynamic batch plan: {dynamic_plan['total_batches']} batches")
            print(f"[ANTI-FRANKENSTEIN]    Section distribution: {dynamic_plan.get('section_distribution', {})}")
            
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Dynamic batch plan error: {e}")
    else:
        print("[ANTI-FRANKENSTEIN] ⚠️ dynamic_batch_planner not available")
    
    # ================================================================
    # 2. ARTICLE MEMORY (Running Summary)
    # ================================================================
    if create_article_memory:
        try:
            memory = create_article_memory(project_id, main_keyword)
            project_data["article_memory"] = memory.to_dict()
            print(f"[ANTI-FRANKENSTEIN] ✅ Article memory initialized")
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Article memory error: {e}")
    else:
        print("[ANTI-FRANKENSTEIN] ⚠️ article_memory not available")
    
    # ================================================================
    # 3. STYLE FINGERPRINT (pusta - wypełni się po batch 1)
    # ================================================================
    project_data["style_fingerprint"] = {
        "analyzed_batches": 0,
        "formality_score": 0.5,
        "formality_level": "semi_formal",
        "sentence_length_avg": 18.0,
        "personal_pronouns": "bezosobowo",
        "example_sentences": [],
        "preferred_transitions": []
    }
    print(f"[ANTI-FRANKENSTEIN] ✅ Style fingerprint placeholder created")
    
    # ================================================================
    # 4. SOFT CAP LIMITS
    # ================================================================
    if get_flexible_limits:
        try:
            total_batches = project_data.get("total_planned_batches", 7)
            soft_caps = {}
            
            for rid, meta in keywords_state.items():
                keyword = meta.get("keyword", "")
                if keyword:
                    soft_caps[keyword] = get_flexible_limits(meta, total_batches)
            
            project_data["soft_cap_limits"] = soft_caps
            print(f"[ANTI-FRANKENSTEIN] ✅ Soft cap limits for {len(soft_caps)} keywords")
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Soft cap limits error: {e}")
    else:
        print("[ANTI-FRANKENSTEIN] ⚠️ soft_cap_limiter not available")
    
    return project_data


# ============================================
# 2. INTEGRACJA W PRE_BATCH_INFO
# ============================================

def get_anti_frankenstein_context(
    project_data: dict,
    current_batch_num: int,
    current_h2: str = ""
) -> dict:
    """
    Pobierz kontekst anti-Frankenstein dla pre_batch_info.
    
    Returns:
        Dict z dodatkowymi sekcjami do wstrzyknięcia w gpt_prompt:
        - dynamic_sections (jeśli multi-section batch)
        - article_memory_context
        - style_instructions
        - soft_cap_recommendations
    """
    context = {
        "has_anti_frankenstein": False,
        "dynamic_sections": None,
        "article_memory_context": None,
        "style_instructions": None,
        "soft_cap_keywords": None
    }
    
    # ================================================================
    # 1. DYNAMIC SECTIONS (jeśli Token Budgeting aktywne)
    # ================================================================
    dynamic_plan = project_data.get("dynamic_batch_plan", {})
    if dynamic_plan and get_batch_sections_for_pre_batch:
        try:
            sections = get_batch_sections_for_pre_batch(dynamic_plan, current_batch_num)
            if sections and sections.get("sections"):
                context["dynamic_sections"] = sections
                context["has_anti_frankenstein"] = True
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Dynamic sections error: {e}")
    
    # ================================================================
    # 2. ARTICLE MEMORY CONTEXT
    # ================================================================
    memory_data = project_data.get("article_memory", {})
    if memory_data and ArticleMemory:
        try:
            memory = ArticleMemory.from_dict(memory_data)
            memory_context = memory.generate_context_for_gpt(current_batch_num, current_h2)
            
            if memory_context and len(memory_context) > 100:
                context["article_memory_context"] = memory_context
                context["has_anti_frankenstein"] = True
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Article memory context error: {e}")
    
    # ================================================================
    # 3. STYLE INSTRUCTIONS
    # ================================================================
    fingerprint = project_data.get("style_fingerprint", {})
    if fingerprint and fingerprint.get("analyzed_batches", 0) > 0 and generate_style_prompt:
        try:
            style_instructions = generate_style_prompt(fingerprint)
            if style_instructions:
                context["style_instructions"] = style_instructions
                context["has_anti_frankenstein"] = True
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Style instructions error: {e}")
    
    # ================================================================
    # 4. SOFT CAP RECOMMENDATIONS
    # ================================================================
    soft_caps = project_data.get("soft_cap_limits", {})
    if soft_caps and create_soft_cap_validator:
        try:
            keywords_state = project_data.get("keywords_state", {})
            total_batches = project_data.get("total_planned_batches", 7)
            batches_done = len(project_data.get("batches", []))
            remaining_batches = max(1, total_batches - batches_done)
            
            validator = create_soft_cap_validator(keywords_state, total_batches)
            recommendations = validator.get_batch_recommendations(remaining_batches)
            
            if recommendations:
                context["soft_cap_keywords"] = recommendations
                context["has_anti_frankenstein"] = True
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Soft cap recommendations error: {e}")
    
    return context


def format_anti_frankenstein_prompt(context: dict) -> str:
    """
    Sformatuj kontekst anti-Frankenstein jako tekst do wstrzyknięcia w gpt_prompt.
    
    Args:
        context: Dict z get_anti_frankenstein_context()
        
    Returns:
        String do dodania do gpt_prompt
    """
    if not context.get("has_anti_frankenstein"):
        return ""
    
    lines = []
    
    # Dynamic sections
    if context.get("dynamic_sections"):
        sections = context["dynamic_sections"]
        if sections.get("section_count", 0) > 1:
            lines.append("=" * 60)
            lines.append("📑 MULTI-SECTION BATCH (Token Budgeting)")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"Ten batch zawiera {sections['section_count']} sekcji H2.")
            lines.append(f"Łączna długość: {sections['total_target_words']} słów")
            lines.append("")
            
            for i, section in enumerate(sections.get("sections", []), 1):
                lines.append(f"📌 SEKCJA {i}: \"{section['h2']}\"")
                lines.append(f"   Kategoria: {section.get('category', 'content')}")
                lines.append(f"   Długość: {section.get('target_length', '300-400')} słów")
                if section.get("guidance"):
                    lines.append(f"   Wskazówka: {section['guidance']}")
                if section.get("assigned_keywords"):
                    lines.append(f"   Frazy: {', '.join(section['assigned_keywords'][:5])}")
                lines.append("")
            
            lines.append("💡 ŁĄCZ SEKCJE PŁYNNIE - używaj słów przejściowych między H2!")
            lines.append("")
    
    # Article memory
    if context.get("article_memory_context"):
        lines.append(context["article_memory_context"])
        lines.append("")
    
    # Style instructions
    if context.get("style_instructions"):
        lines.append(context["style_instructions"])
        lines.append("")
    
    return "\n".join(lines)


# ============================================
# 3. INTEGRACJA W APPROVE_BATCH
# ============================================

def update_project_after_batch(
    project_data: dict,
    batch_text: str,
    batch_number: int,
    h2_sections: List[str],
    entities_used: List[str] = None,
    humanness_score: float = 100.0
) -> dict:
    """
    Aktualizuj dane projektu po zatwierdzeniu batcha.
    
    Aktualizuje:
    - article_memory (claims, entities, summary)
    - style_fingerprint (analiza stylu)
    - soft_cap_limits (actual_uses)
    
    Args:
        project_data: Dane projektu
        batch_text: Tekst zatwierdzonego batcha
        batch_number: Numer batcha
        h2_sections: Nagłówki H2 w batchu
        entities_used: Użyte encje
        humanness_score: Wynik detekcji AI
        
    Returns:
        Zaktualizowane project_data
    """
    
    # ================================================================
    # 1. UPDATE ARTICLE MEMORY
    # ================================================================
    memory_data = project_data.get("article_memory", {})
    if memory_data and update_article_memory and ArticleMemory:
        try:
            memory = ArticleMemory.from_dict(memory_data)
            
            updated_memory = update_article_memory(
                memory=memory,
                batch_text=batch_text,
                batch_number=batch_number,
                h2_sections=h2_sections,
                entities_used=entities_used
            )
            
            project_data["article_memory"] = updated_memory.to_dict()
            print(f"[ANTI-FRANKENSTEIN] ✅ Article memory updated: {len(updated_memory.key_claims)} claims")
            
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Article memory update error: {e}")
    
    # ================================================================
    # 2. UPDATE STYLE FINGERPRINT
    # ================================================================
    fingerprint = project_data.get("style_fingerprint", {})
    if analyze_style:
        try:
            updated_fingerprint = analyze_style(batch_text, fingerprint)
            project_data["style_fingerprint"] = updated_fingerprint
            print(f"[ANTI-FRANKENSTEIN] ✅ Style fingerprint updated: "
                  f"formality={updated_fingerprint.get('formality_score', 0):.2f}, "
                  f"pronouns={updated_fingerprint.get('personal_pronouns', 'unknown')}")
            
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Style fingerprint update error: {e}")
    
    # ================================================================
    # 3. CHECK STYLE CONSISTENCY
    # ================================================================
    if check_style_consistency and fingerprint.get("analyzed_batches", 0) > 0:
        try:
            consistency = check_style_consistency(batch_text, fingerprint)
            
            if consistency.get("severity") in ["MEDIUM", "HIGH"]:
                deviations = consistency.get("deviations", [])
                print(f"[ANTI-FRANKENSTEIN] ⚠️ Style deviation detected ({consistency['severity']}):")
                for dev in deviations[:2]:
                    print(f"   - {dev.get('type')}: {dev.get('suggestion', '')[:60]}")
                
                # Zapisz ostrzeżenie do projektu
                if "style_warnings" not in project_data:
                    project_data["style_warnings"] = []
                project_data["style_warnings"].append({
                    "batch": batch_number,
                    "severity": consistency["severity"],
                    "deviations": deviations
                })
        except Exception as e:
            print(f"[ANTI-FRANKENSTEIN] ⚠️ Style consistency check error: {e}")
    
    return project_data


def validate_batch_with_soft_caps(
    batch_counts: Dict[str, int],
    keywords_state: dict,
    humanness_score: float = 100.0,
    total_batches: int = 7
) -> dict:
    """
    Waliduj batch z miękkimi limitami.
    
    Używane w process_batch_in_firestore jako uzupełnienie standardowej walidacji.
    
    Returns:
        Dict z wynikami walidacji soft cap
    """
    if not validate_with_soft_caps:
        return {"available": False}
    
    try:
        result = validate_with_soft_caps(
            batch_counts=batch_counts,
            keywords_state=keywords_state,
            humanness_score=humanness_score,
            total_batches=total_batches
        )
        result["available"] = True
        return result
    except Exception as e:
        print(f"[ANTI-FRANKENSTEIN] ⚠️ Soft cap validation error: {e}")
        return {"available": False, "error": str(e)}


# ============================================
# HELPER: GENERATE FULL ANTI-FRANKENSTEIN PROMPT
# ============================================

def generate_anti_frankenstein_gpt_section(
    project_data: dict,
    current_batch_num: int,
    current_h2: str = ""
) -> str:
    """
    Generuj pełną sekcję anti-Frankenstein do wstrzyknięcia w gpt_prompt.
    
    Convenience function łącząca get_context i format_prompt.
    """
    context = get_anti_frankenstein_context(project_data, current_batch_num, current_h2)
    return format_anti_frankenstein_prompt(context)


# ============================================
# STATUS CHECK
# ============================================

def get_anti_frankenstein_status() -> dict:
    """Sprawdź dostępność wszystkich modułów anti-Frankenstein"""
    return {
        "dynamic_batch_planner": create_dynamic_batch_plan is not None,
        "article_memory": create_article_memory is not None,
        "style_analyzer": analyze_style is not None,
        "soft_cap_limiter": create_soft_cap_validator is not None,
        "all_available": all([
            create_dynamic_batch_plan,
            create_article_memory,
            analyze_style,
            create_soft_cap_validator
        ])
    }


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    print("=== ANTI-FRANKENSTEIN STATUS ===")
    status = get_anti_frankenstein_status()
    for module, available in status.items():
        emoji = "✅" if available else "❌"
        print(f"  {emoji} {module}: {'available' if available else 'NOT AVAILABLE'}")
