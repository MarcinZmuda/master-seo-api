"""
PROJECT ROUTES - v44.0 BRAJEN SEO Engine - DYNAMIC BATCH CALCULATION

🆕 v44.0 ZMIANY:
- DYNAMIC BATCH CALCULATION: Zamiast sztywnych progów (100+ fraz → 9 batchy),
  teraz obliczamy dynamicznie na podstawie 3 constraintów:
  1. Słowa per batch (target 500) → czytelność
  2. Frazy per batch (max 12) → bez stuffingu
  3. H2 per batch (1-2) → struktura
- Wybieramy MAX z constraintów (najostrzejszy limit)
- Response zawiera breakdown i limiting_factor dla przejrzystości

🆕 v43.0 ZMIANY:
- PHRASE HIERARCHY: Analiza hierarchii fraz (rdzenie vs rozszerzenia)
- Nowy endpoint GET /api/project/{id}/phrase_hierarchy
- Nowy endpoint POST /api/project/{id}/analyze_hierarchy
- Integracja z enhanced_pre_batch.py (phrase_hierarchy_data)
- Zapobieganie stuffing przez świadomość rozszerzeń

ZMIANY v40.1:
- 🆕 FIRESTORE FIX: sanitize_for_firestore() dla kluczy ze znakami . / [ ]
- 🆕 Sanityzacja przed każdym doc_ref.set() i doc_ref.update()
- 🆕 Naprawiono ValueError: One or more components is not a string or is empty

ZMIANY v30.1 OPTIMIZED:
- 🆕 Best-of-N domyślnie WŁĄCZONE (use_best_of_n=True)
- 🆕 Auto-approve po 2 próbach (było 3)
- 🆕 Funkcja distribute_extended_keywords dla lepszego rozłożenia fraz
- 🆕 Timeout 45s (było 60s)

ZMIANY v30.0:
- 🆕 LEGAL MODULE: Auto-detekcja kategorii "prawo"
- 🆕 SAOS Integration: Pobieranie orzeczeń sądowych
- 🆕 Judgment Scoring: Wybór najlepszych orzeczeń (40+ pkt)
- 🆕 Max 2 citations per article
- 🆕 legal_instruction w response z project/create

ZMIANY v29.2:
- NOWY ENDPOINT: generateH2Plan - generuje H2 na podstawie Semantic HTML + Content Relevancy
- H2 Generator: Intent matching, Related subtopics, PAA integration
- Frazy użytkownika MUSZĄ być w H2 (ale naturalnie!)

ZMIANY v29.1:
- NOWE PRIORYTETY: Jakość tekstu > Encje > SEO
- Elastyczne podejście do fraz (min 1×, nie blokuje za "za mało")
- Lemmatyzacja fraz (ścieżka sensoryczna = ścieżką sensoryczną)
- Wykrywanie tautologii i pleonazmów
- Auto-approve po 3 próbach

ZMIANY v26.1:
- Best-of-N batch selection (generuje 3 wersje, wybiera najlepszą)
- Intro excluded from density calculation
- Polish quality validation integrated
- EXCLUSIVE counting dla actual_uses (nie overlapping)
- Soft cap + short keyword protection
- Synonimy przy przekroczeniu fraz

LOGIKA FRAZ v29.2:
- BASIC/MAIN: min 1× (MUSI być), zalecane ilości to CEL nie wymóg
- EXTENDED: min 1× (MUSI być), potem OK
- Stuffing (>MAX): JEDYNY BLOKER!
- Brak (0×): WARNING, Claude uzupełni
- Underused (<target ale >0): OK
"""

import uuid
import re
import os
import json
import math
import spacy
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# ================================================================
# 🆕 v44.1: CENTRALIZED IMPORTS via feature_flags
# Zastępuje ~20 bloków try/except jednym importem
# ================================================================
from firestore_utils import sanitize_for_firestore, batch_update
from feature_flags import FEATURES, get_module, safe_import, is_enabled

# Feature availability flags (for backward compatibility)
KEYWORD_CONFLICT_VALIDATOR_ENABLED = FEATURES.keyword_conflict_validator
PHRASE_HIERARCHY_ENABLED = FEATURES.phrase_hierarchy
ANTI_FRANKENSTEIN_ENABLED = FEATURES.anti_frankenstein_integration
ENHANCED_PRE_BATCH_ENABLED = FEATURES.enhanced_pre_batch
SYNONYMS_ENABLED = FEATURES.keyword_synonyms
BEST_OF_N_ENABLED = FEATURES.batch_best_of_n
BATCH_PLANNER_ENABLED = FEATURES.batch_planner
LEGAL_MODULE_ENABLED = FEATURES.legal_routes_v3
POLISH_QUALITY_ENABLED = FEATURES.polish_language_quality
H2_GENERATOR_ENABLED = FEATURES.h2_generator

# Lazy imports - funkcje pobierane gdy potrzebne
def _get_phrase_hierarchy_functions():
    """Lazy import phrase_hierarchy functions."""
    ph = get_module('phrase_hierarchy')
    if ph:
        return (
            ph.analyze_phrase_hierarchy,
            ph.hierarchy_to_dict,
            ph.dict_to_hierarchy,
            ph.format_hierarchy_for_agent,
            ph.format_hierarchy_summary_short,
            ph.get_batch_hierarchy_context
        )
    return (None,) * 6

def _get_anti_frankenstein_functions():
    """Lazy import anti_frankenstein functions."""
    af = get_module('anti_frankenstein_integration')
    if af:
        return (
            af.enhance_project_with_anti_frankenstein,
            af.get_anti_frankenstein_context,
            af.generate_anti_frankenstein_gpt_section,
            af.update_project_after_batch
        )
    return (None,) * 4

def _get_enhanced_pre_batch_functions():
    """Lazy import enhanced_pre_batch functions."""
    epb = get_module('enhanced_pre_batch')
    if epb:
        return {
            'generate_enhanced_pre_batch_info': epb.generate_enhanced_pre_batch_info,
            'get_entities_to_define': getattr(epb, 'get_entities_to_define', None),
            'get_relations_to_establish': getattr(epb, 'get_relations_to_establish', None),
            'get_semantic_context': getattr(epb, 'get_semantic_context', None),
            'get_style_instructions': getattr(epb, 'get_style_instructions', None),
            'get_continuation_context': getattr(epb, 'get_continuation_context', None),
            'get_keyword_tracking_info': getattr(epb, 'get_keyword_tracking_info', None),
            'calculate_optimal_batch_count': getattr(epb, 'calculate_optimal_batch_count', None),
            'AI_PATTERNS_TO_AVOID': getattr(epb, 'AI_PATTERNS_TO_AVOID', [])
        }
    return {}

# Compatibility layer for direct imports (używane w kodzie)
if PHRASE_HIERARCHY_ENABLED:
    (analyze_phrase_hierarchy, hierarchy_to_dict, dict_to_hierarchy,
     format_hierarchy_for_agent, format_hierarchy_summary_short,
     get_batch_hierarchy_context) = _get_phrase_hierarchy_functions()

if ANTI_FRANKENSTEIN_ENABLED:
    (enhance_project_with_anti_frankenstein, get_anti_frankenstein_context,
     generate_anti_frankenstein_gpt_section, update_project_after_batch) = _get_anti_frankenstein_functions()

if ENHANCED_PRE_BATCH_ENABLED:
    _epb_funcs = _get_enhanced_pre_batch_functions()
    generate_enhanced_pre_batch_info = _epb_funcs.get('generate_enhanced_pre_batch_info')
    AI_PATTERNS_TO_AVOID = _epb_funcs.get('AI_PATTERNS_TO_AVOID', [])

if SYNONYMS_ENABLED:
    ks = get_module('keyword_synonyms')
    if ks:
        generate_exceeded_warning = ks.generate_exceeded_warning
        generate_softcap_warning = ks.generate_softcap_warning
        generate_synonyms_prompt_section = ks.generate_synonyms_prompt_section
        get_synonyms = ks.get_synonyms

if BEST_OF_N_ENABLED:
    bon = get_module('batch_best_of_n')
    if bon:
        select_best_batch = bon.select_best_batch
        BestOfNConfig = bon.BestOfNConfig

if BATCH_PLANNER_ENABLED:
    bp = get_module('batch_planner')
    if bp:
        create_article_plan = bp.create_article_plan
        # 🆕 v44.2: FAST MODE
        create_article_plan_fast = getattr(bp, 'create_article_plan_fast', None)
        if create_article_plan_fast:
            print("[PROJECT_ROUTES] ✅ FAST MODE available (create_article_plan_fast)")
        else:
            create_article_plan_fast = None

if LEGAL_MODULE_ENABLED:
    lr = get_module('legal_routes_v3')
    if lr:
        enhance_project_with_legal = lr.enhance_project_with_legal
else:
    def enhance_project_with_legal(project_data, main_keyword, h2_list):
        return project_data

if POLISH_QUALITY_ENABLED:
    plq = get_module('polish_language_quality')
    if plq:
        quick_polish_check = plq.quick_polish_check
        check_collocations = plq.check_collocations
        check_banned_phrases = plq.check_banned_phrases
        INCORRECT_COLLOCATIONS = getattr(plq, 'INCORRECT_COLLOCATIONS', {})

if H2_GENERATOR_ENABLED:
    h2g = get_module('h2_generator')
    if h2g:
        generate_h2_plan = h2g.generate_h2_plan
        validate_h2_plan = h2g.validate_h2_plan

if KEYWORD_CONFLICT_VALIDATOR_ENABLED:
    kcv = get_module('keyword_conflict_validator')
    if kcv:
        validate_keywords_before_create = kcv.validate_keywords_before_create

print(f"[PROJECT_ROUTES] ✅ Loaded via feature_flags: "
      f"phrase_hierarchy={PHRASE_HIERARCHY_ENABLED}, "
      f"enhanced_pre_batch={ENHANCED_PRE_BATCH_ENABLED}, "
      f"batch_planner={BATCH_PLANNER_ENABLED}")

# ================================================================
# 🆕 v44.2: IMPORT Z PROJECT_HELPERS (wydzielone funkcje)
# Zastępuje ~730 linii lokalnych definicji
# ================================================================
from project_helpers.helpers import (
    # Constants
    DENSITY_OPTIMAL_MIN,
    DENSITY_OPTIMAL_MAX,
    DENSITY_ACCEPTABLE_MAX,
    DENSITY_WARNING_MAX,
    DENSITY_MAX,
    SOFT_CAP_THRESHOLD,
    SHORT_KEYWORD_MAX_WORDS,
    SHORT_KEYWORD_MAX_REDUCTION,
    SHORT_KEYWORD_ABSOLUTE_MAX,
    
    # Entity helpers
    get_entities_to_introduce,
    get_already_defined_entities,
    get_overused_phrases,
    get_synonyms_for_overused,
    
    # Keyword distribution
    distribute_extended_keywords,
    get_section_length_guidance,
    
    # Density & soft cap
    get_adjusted_target_max,
    check_soft_cap,
    get_density_status,
    
    # Coverage
    validate_coverage,
    
    # Synonyms
    detect_main_keyword_synonyms,
    
    # Suggested calculation
    calculate_suggested_v25,
)

from project_helpers.semantic_planning import (
    THEMATIC_RULES,
    create_semantic_keyword_plan,
)

print(f"[PROJECT_ROUTES] ✅ Imported {22} functions from project_helpers")

# ================================================================
# 🧠 H2 SUGGESTIONS (Claude primary, Gemini fallback) v27.0
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    """Generuje sugestie H2 używając Claude (primary) lub Gemini (fallback)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "")
    if not topic:
        return jsonify({"error": "Required: topic or main_keyword"}), 400
    
    serp_h2_patterns = data.get("serp_h2_patterns", [])
    target_keywords = data.get("target_keywords", [])
    target_count = min(data.get("target_count", 6), 6)
    
    # Build prompt (shared between Claude and Gemini)
    competitor_context = ""
    if serp_h2_patterns:
        competitor_context = f"""
WZORCE H2 Z KONKURENCJI (TOP 10 SERP):
{chr(10).join(f"- {h2}" for h2 in serp_h2_patterns[:20])}
"""
    
    keywords_context = ""
    if target_keywords:
        keywords_context = f"""
FRAZY KLUCZOWE DO WPLECENIA W H2:
{', '.join(target_keywords[:10])}
"""
    
    prompt = f"""Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu SEO o temacie: "{topic}"

{competitor_context}
{keywords_context}

KRYTYCZNE ZASADY:
1. MAX 1 H2 z frazą główną "{topic}"! Reszta: synonimy lub naturalne tytuły
2. NIE UŻYWAJ ogólników: "dokument", "wniosek", "sprawa", "proces"
3. Każdy H2 powinien mieć 5-8 słów (max 70 znaków)
4. Minimum 30% H2 w formie pytania (Jak...?, Ile...?, Gdzie...?)
5. NIE używaj: "Wstęp", "Podsumowanie", "Zakończenie", "FAQ"

FORMAT: Zwróć TYLKO listę {target_count} H2, każdy w nowej linii, bez numeracji."""
    
    suggestions = []
    model_used = "fallback"
    
    # === TRY CLAUDE FIRST ===
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            print(f"[H2_SUGGESTIONS] Trying Claude for: {topic}")
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = response.content[0].text.strip()
            raw_suggestions = raw_text.split('\n')
            suggestions = [
                h2.strip().lstrip('•-–—0123456789.). ')
                for h2 in raw_suggestions 
                if h2.strip() and len(h2.strip()) > 5
            ][:target_count]
            
            model_used = "claude-sonnet-4-20250514"
            print(f"[H2_SUGGESTIONS] ✅ Claude generated {len(suggestions)} H2")
            
        except Exception as e:
            print(f"[H2_SUGGESTIONS] ⚠️ Claude failed: {e}, trying Gemini...")
            suggestions = []
    
    # === FALLBACK TO GEMINI ===
    if not suggestions and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            print(f"[H2_SUGGESTIONS] Trying Gemini for: {topic}")
            response = model.generate_content(prompt)
            
            raw_suggestions = response.text.strip().split('\n')
            suggestions = [
                h2.strip().lstrip('•-–—0123456789.). ')
                for h2 in raw_suggestions 
                if h2.strip() and len(h2.strip()) > 5
            ][:target_count]
            
            model_used = GEMINI_MODEL
            print(f"[H2_SUGGESTIONS] ✅ Gemini generated {len(suggestions)} H2")
            
        except Exception as e:
            print(f"[H2_SUGGESTIONS] ⚠️ Gemini failed: {e}")
            suggestions = []
    
    # === STATIC FALLBACK ===
    if not suggestions:
        suggestions = [
            f"Czym jest {topic}?",
            f"Jak działa {topic}?",
            f"Korzyści z {topic}",
            f"Kiedy warto skorzystać z {topic}?",
            f"Ile kosztuje {topic}?",
            f"Najczęstsze pytania o {topic}"
        ][:target_count]
        model_used = "static_fallback"
        print(f"[H2_SUGGESTIONS] ⚠️ Using static fallback")
    
    # Analyze main keyword coverage
    topic_lower = topic.lower()
    h2_with_main = sum(1 for h2 in suggestions if topic_lower in h2.lower())
    
    if h2_with_main > 1:
        print(f"[H2_SUGGESTIONS] ⚠️ Za dużo H2 z frazą główną ({h2_with_main}). Zalecane: max 1")
    
    return jsonify({
        "status": "OK" if model_used != "static_fallback" else "FALLBACK",
        "suggestions": suggestions,
        "topic": topic,
        "model": model_used,
        "count": len(suggestions),
        "main_keyword_in_h2": {
            "count": h2_with_main,
            "max_recommended": 1,
            "overoptimized": h2_with_main > 1,
            "note": "Max 1 H2 z frazą główną. Reszta: synonimy lub naturalne tytuły."
        },
        "action_required": "USER_H2_INPUT_NEEDED"
    }), 200

# ================================================================
# FINALIZE H2
# ================================================================
@project_routes.post("/api/project/finalize_h2")
def finalize_h2():
    """Łączy sugestie H2 z frazami usera."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    suggested_h2 = data.get("suggested_h2", [])
    user_h2_phrases = data.get("user_h2_phrases", [])
    topic = data.get("topic", "")
    
    if not suggested_h2:
        return jsonify({"error": "Required: suggested_h2"}), 400
    
    if not GEMINI_API_KEY or not user_h2_phrases:
        return jsonify({
            "status": "OK",
            "final_h2": suggested_h2,
            "message": "No user phrases or Gemini unavailable"
        }), 200
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
Masz sugestie H2 dla artykulu o "{topic}":
{chr(10).join(f"- {h2}" for h2 in suggested_h2)}

User chce zeby w H2 byly frazy:
{chr(10).join(f"- {phrase}" for phrase in user_h2_phrases)}

Zmodyfikuj H2 zeby KAZDA fraza usera pojawila sie w przynajmniej jednym H2.
Zachowaj naturalnosc, 6-15 slow kazdy H2, min 30% w formie pytania.

Zwroc TYLKO liste H2, kazdy w nowej linii.
"""
        
        response = model.generate_content(prompt)
        
        final_h2 = [
            h2.strip().lstrip('-0123456789.). ')
            for h2 in response.text.strip().split('\n')
            if h2.strip() and len(h2.strip()) > 5
        ]
        
        covered = []
        uncovered = []
        for phrase in user_h2_phrases:
            if any(phrase.lower() in h2.lower() for h2 in final_h2):
                covered.append(phrase)
            else:
                uncovered.append(phrase)
        
        return jsonify({
            "status": "OK",
            "final_h2": final_h2,
            "coverage": {
                "covered_phrases": covered,
                "uncovered_phrases": uncovered,
                "coverage_percent": round(len(covered) / len(user_h2_phrases) * 100, 1) if user_h2_phrases else 100
            }
        }), 200
        
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e), "final_h2": suggested_h2}), 500


# ================================================================
# 🏗️ VALIDATE H2 PLAN v29.2 - Claude tworzy, API waliduje
# ================================================================
@project_routes.post("/api/project/<project_id>/validate_h2_plan")
def validate_h2_plan_endpoint(project_id):
    """
    Waliduje plan H2 stworzony przez Claude.
    
    CLAUDE TWORZY H2 według zasad → API WALIDUJE
    
    INPUT:
    {
        "main_keyword": "pomoce sensoryczne w przedszkolu",
        "h2_phrases": ["integracja sensoryczna", "ścieżka sensoryczna"],
        "h2_plan": [
            {"h2": "Czym są pomoce sensoryczne?", "phrase_used": "pomoce sensoryczne"},
            {"h2": "Integracja sensoryczna - dlaczego?", "phrase_used": "integracja sensoryczna"},
            ...
        ]
    }
    
    OUTPUT:
    {
        "valid": true/false,
        "coverage": {"all_phrases_covered": true, "missing": []},
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
    """
    if not H2_GENERATOR_ENABLED:
        return jsonify({
            "error": "H2 Validator module not available",
            "fallback": True
        }), 500
    
    data = request.get_json() or {}
    
    main_keyword = data.get("main_keyword", "")
    h2_phrases = data.get("h2_phrases", [])
    h2_plan = data.get("h2_plan", [])
    
    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required (list of H2 from Claude)"}), 400
    
    try:
        # Walidacja planu
        validation = validate_h2_plan(h2_plan, main_keyword)
        
        # Sprawdź coverage fraz
        coverage = check_phrase_coverage(h2_plan, h2_phrases, main_keyword)
        
        # Ogólna ocena
        is_valid = validation["valid"] and coverage["all_phrases_covered"]
        
        # Zapisz do projektu jeśli valid
        if is_valid:
            db = firestore.client()
            project_ref = db.collection("projects").document(project_id)
            if project_ref.get().exists:
                # Normalizuj h2_plan do listy dict
                normalized_plan = []
                for i, h2 in enumerate(h2_plan, 1):
                    if isinstance(h2, str):
                        normalized_plan.append({
                            "position": i,
                            "h2": h2,
                            "phrase_used": None
                        })
                    else:
                        h2["position"] = i
                        normalized_plan.append(h2)
                
                project_ref.update({
                    "h2_plan": normalized_plan,
                    "h2_coverage": coverage,
                    "h2_validated_at": firestore.SERVER_TIMESTAMP
                })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "valid": is_valid,
            "validation": validation,
            "coverage": coverage,
            "message": "Plan H2 zaakceptowany!" if is_valid else "Plan H2 wymaga poprawek"
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def check_phrase_coverage(h2_plan: list, h2_phrases: list, main_keyword: str) -> dict:
    """Sprawdza czy wszystkie frazy użytkownika są pokryte w H2."""
    
    # Zbierz wszystkie H2 jako tekst
    h2_texts = []
    for h2 in h2_plan:
        if isinstance(h2, str):
            h2_texts.append(h2.lower())
        elif isinstance(h2, dict):
            h2_texts.append(h2.get("h2", "").lower())
    
    all_h2_text = " ".join(h2_texts)
    
    # Sprawdź główną frazę
    main_keyword_covered = main_keyword.lower() in all_h2_text
    
    # Sprawdź frazy użytkownika
    covered = []
    missing = []
    
    for phrase in h2_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in all_h2_text:
            covered.append(phrase)
        else:
            # Sprawdź też odmiany (częściowe dopasowanie)
            phrase_words = set(phrase_lower.split())
            found_partial = False
            for h2_text in h2_texts:
                h2_words = set(h2_text.split())
                if phrase_words.issubset(h2_words) or len(phrase_words.intersection(h2_words)) >= len(phrase_words) * 0.7:
                    found_partial = True
                    break
            
            if found_partial:
                covered.append(phrase)
            else:
                missing.append(phrase)
    
    coverage_percent = (len(covered) / len(h2_phrases) * 100) if h2_phrases else 100
    
    return {
        "main_keyword_covered": main_keyword_covered,
        "phrases_covered": covered,
        "phrases_missing": missing,
        "coverage_percent": round(coverage_percent, 1),
        "all_phrases_covered": len(missing) == 0 and main_keyword_covered
    }


@project_routes.post("/api/project/<project_id>/save_h2_plan")
def save_h2_plan_endpoint(project_id):
    """
    Zapisuje plan H2 do projektu (bez walidacji - zaufaj Claude).
    
    INPUT:
    {
        "h2_plan": [
            "Czym są pomoce sensoryczne?",
            "Integracja sensoryczna - dlaczego?",
            ...
        ]
    }
    """
    data = request.get_json() or {}
    h2_plan = data.get("h2_plan", [])
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required"}), 400
    
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        
        if not project_ref.get().exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        # Normalizuj do listy dict
        normalized_plan = []
        for i, h2 in enumerate(h2_plan, 1):
            if isinstance(h2, str):
                normalized_plan.append({
                    "position": i,
                    "h2": h2
                })
            else:
                h2["position"] = i
                normalized_plan.append(h2)
        
        project_ref.update({
            "h2_plan": normalized_plan,
            "h2_saved_at": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": normalized_plan,
            "message": "Plan H2 zapisany!"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@project_routes.get("/api/project/<project_id>/h2_plan")
def get_h2_plan(project_id):
    """
    Pobiera zapisany plan H2 dla projektu.
    """
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        project = project_doc.to_dict()
        
        h2_plan = project.get("h2_plan", [])
        
        if not h2_plan:
            return jsonify({
                "status": "NOT_GENERATED",
                "message": "H2 plan not generated yet. Call POST /generate_h2_plan first."
            }), 200
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": h2_plan,
            "h3_suggestions": project.get("h2_h3_suggestions", {}),
            "coverage": project.get("h2_coverage", {}),
            "meta": project.get("h2_plan_meta", {})
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@project_routes.post("/api/project/<project_id>/update_h2_plan")
def update_h2_plan(project_id):
    """
    Aktualizuje plan H2 (po modyfikacjach użytkownika).
    
    INPUT:
    {
        "h2_plan": [
            {"position": 1, "h2": "...", ...},
            ...
        ]
    }
    """
    data = request.get_json() or {}
    h2_plan = data.get("h2_plan", [])
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required"}), 400
    
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        project = project_doc.to_dict()
        main_keyword = project.get("main_keyword", "")
        
        # Walidacja nowego planu
        if H2_GENERATOR_ENABLED:
            validation = validate_h2_plan(h2_plan, main_keyword)
        else:
            validation = {"valid": True, "issues": [], "warnings": []}
        
        # Zapisz zaktualizowany plan
        project_ref.update({
            "h2_plan": h2_plan,
            "h2_plan_updated_at": firestore.SERVER_TIMESTAMP,
            "h2_plan_validation": validation
        })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": h2_plan,
            "validation": validation
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
#  PROJECT CREATE - v25.0
# ================================================================

@project_routes.post("/api/project/create")
def create_project():
    """Tworzy nowy projekt SEO w Firestore."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "").strip()
    if not topic:
        return jsonify({"error": "Required field: topic or main_keyword"}), 400
    
    h2_structure = data.get("h2_structure", [])
    h2_terms = data.get("h2_terms", [])  # v39.0: Hasła do wygenerowania H2
    raw_keywords = data.get("keywords_list") or data.get("keywords", [])
    target_length = data.get("target_length", 3000)
    source = data.get("source", "unknown")
    s1_data = data.get("s1_data", {})  # v39.0: Dane z S1 do generowania H2
    
    # 🆕 v44.2: FAST MODE - artykuł w 3 batchach
    article_mode = data.get("mode", "standard").lower()
    if article_mode not in ["standard", "fast"]:
        article_mode = "standard"
    
    print(f"[PROJECT] 📋 Mode: {article_mode.upper()}")
    
    # ================================================================
    # v39.0: AUTO-GENEROWANIE H2 z h2_terms + S1
    # Jeśli user podał h2_terms (hasła) zamiast gotowych h2_structure,
    # generujemy pełne H2 używając h2_generator + danych z S1
    # ================================================================
    h2_generation_info = None
    if h2_terms and not h2_structure:
        print(f"[PROJECT] 🏗️ Generating H2 from h2_terms: {h2_terms}")
        
        if H2_GENERATOR_ENABLED:
            try:
                # Wyciągnij dane z S1
                search_intent = s1_data.get("search_intent", "informational")
                entities = s1_data.get("entity_seo", {}).get("entities", []) or s1_data.get("entities", [])
                paa_questions = [p.get("question", "") for p in s1_data.get("paa", []) or s1_data.get("paa_questions", [])]
                competitor_h2 = s1_data.get("serp_analysis", {}).get("competitor_h2", []) or s1_data.get("competitor_h2", [])
                
                # Generuj plan H2
                h2_result = generate_h2_plan(
                    main_keyword=topic,
                    h2_phrases=h2_terms,
                    search_intent=search_intent,
                    entities=entities[:15] if entities else [],
                    paa_questions=paa_questions[:10] if paa_questions else [],
                    competitor_h2=competitor_h2[:20] if competitor_h2 else []
                )
                
                # Wyciągnij wygenerowane H2
                h2_structure = [h.get("h2", "") for h in h2_result.get("h2_plan", []) if h.get("h2")]
                
                h2_generation_info = {
                    "source": "h2_generator",
                    "h2_terms_used": h2_terms,
                    "search_intent": search_intent,
                    "h2_plan_details": h2_result.get("h2_plan", []),
                    "coverage": h2_result.get("coverage", {}),
                    "h3_suggestions": h2_result.get("h3_suggestions", {})
                }
                
                print(f"[PROJECT] ✅ Generated {len(h2_structure)} H2: {h2_structure}")
                
            except Exception as e:
                print(f"[PROJECT] ⚠️ H2 generator failed: {e}, using h2_terms as-is")
                # Fallback: użyj h2_terms jako proste H2
                h2_structure = [f"Czym jest {h2_terms[0]}?" if i == 0 else term for i, term in enumerate(h2_terms)]
                h2_generation_info = {"source": "fallback", "error": str(e)}
        else:
            # Fallback bez modułu: proste H2 z terminów
            print(f"[PROJECT] ⚠️ H2 generator not available, using h2_terms as-is")
            h2_structure = [f"Czym jest {h2_terms[0]}?" if i == 0 else term for i, term in enumerate(h2_terms)]
            h2_generation_info = {"source": "fallback_no_module"}
    
    total_planned_batches = data.get("total_planned_batches")
    if not total_planned_batches:
        total_planned_batches = max(2, min(6, math.ceil(len(h2_structure) / 2))) if h2_structure else 4

    main_keyword_synonyms = detect_main_keyword_synonyms(topic)
    print(f"[PROJECT]  Main keyword synonyms for '{topic}': {main_keyword_synonyms}")

    # ================================================================
    # 🆕 v40.1: WALIDACJA KONFLIKTÓW FRAZ
    # Zapobiega tworzeniu projektów gdzie BASIC keyword ⊂ MAIN lub H2
    # (co prowadzi do nieskończonej pętli REWRITE)
    # ================================================================
    keyword_conflict_info = None
    if KEYWORD_CONFLICT_VALIDATOR_ENABLED and raw_keywords:
        try:
            conflict_result = validate_keywords_before_create(
                main_keyword=topic,
                h2_structure=h2_structure,
                keywords_list=raw_keywords,
                auto_fix=True  # Automatycznie napraw konflikty (degradacja BASIC → EXTENDED)
            )
            
            keyword_conflict_info = {
                "conflicts_found": len(conflict_result.get("conflicts", [])),
                "critical_fixed": conflict_result.get("critical_count", 0),
                "warnings": conflict_result.get("warning_count", 0),
                "message": conflict_result.get("message", "")
            }
            
            # Jeśli auto_fix naprawił konflikty, użyj poprawionej listy
            if conflict_result.get("fixed_keywords"):
                raw_keywords = conflict_result["fixed_keywords"]
                print(f"[PROJECT] 🔧 Keyword conflicts auto-fixed: {keyword_conflict_info}")
            
            if not conflict_result.get("can_create", True):
                return jsonify({
                    "error": "Keyword conflicts detected",
                    "conflicts": conflict_result.get("conflicts", []),
                    "message": conflict_result.get("message", "")
                }), 400
                
        except Exception as e:
            print(f"[PROJECT] ⚠️ Keyword conflict validation failed: {e}")
            keyword_conflict_info = {"error": str(e)}

    firestore_keywords = {}
    main_keyword_found = False
    
    for item in raw_keywords:
        term = item.get("term") or item.get("keyword", "")
        term = term.strip() if term else ""
        
        if not term:
            continue
        
        doc = nlp(term)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        min_val = item.get("min") or item.get("target_min", 1)
        max_val = item.get("max") or item.get("target_max", 5)
        
        row_id = item.get("id") or str(uuid.uuid4())
        
        is_main = term.lower() == topic.lower()
        if is_main:
            main_keyword_found = True
            min_val = max(min_val, max(6, target_length // 350))
            max_val = max(max_val, target_length // 150)
        
        is_synonym_of_main = term.lower() in [s.lower() for s in main_keyword_synonyms]
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": min_val,
            "target_max": max_val,
            "display_limit": min_val + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN" if is_main else item.get("type", "BASIC").upper(),
            "is_main_keyword": is_main,
            "is_synonym_of_main": is_synonym_of_main,
            "remaining_max": max_val,
            "optimal_target": max_val
        }
    
    if not main_keyword_found:
        main_min = max(6, target_length // 350)
        main_max = target_length // 150
        
        doc = nlp(topic)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        firestore_keywords["main_keyword_auto"] = {
            "keyword": topic,
            "search_term_exact": topic.lower(),
            "search_lemma": search_lemma,
            "target_min": main_min,
            "target_max": main_max,
            "display_limit": main_min + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN",
            "is_main_keyword": True,
            "is_synonym_of_main": False,
            "remaining_max": main_max,
            "optimal_target": main_max
        }
        print(f"[PROJECT]  Auto-added main keyword '{topic}' with min={main_min}, max={main_max}")

    # ================================================================
    # v26.1: AUTO-REDUKCJA target_max dla NESTED KEYWORDS (INCLUSIVE)
    # W stylu NeuronWriter: "radca prawny" liczy się jako:
    #   - "radca prawny" → 1
    #   - "radca" → 1 (bo słowo "radca" jest w środku)
    #   - "prawny" → 1 (bo słowo "prawny" jest w środku)
    # 
    # Musimy obniżyć target_max krótszej frazy proporcjonalnie do tego
    # ile razy będzie "dziedziczona" z dłuższych fraz.
    #
    # v39.0: SKIP dla fraz w H2 (STRUCTURAL) - nie redukujemy ich limitów!
    # ================================================================
    
    # v39.0: Zidentyfikuj frazy STRUCTURAL (w H2 lub MAIN)
    structural_keywords = set()
    for h2 in h2_structure:
        h2_lower = h2.lower()
        for rid, meta in firestore_keywords.items():
            kw = meta.get("keyword", "").lower()
            if kw and (kw in h2_lower or h2_lower in kw or meta.get("type", "").upper() == "MAIN"):
                structural_keywords.add(rid)
                firestore_keywords[rid]["is_structural"] = True
    
    all_keywords = [(rid, meta.get("keyword", "").lower(), meta.get("keyword", "").lower().split()) 
                    for rid, meta in firestore_keywords.items()]
    
    for rid, meta in firestore_keywords.items():
        # v39.0: SKIP redukcji dla STRUCTURAL keywords
        if rid in structural_keywords:
            print(f"[PROJECT] 🔵 STRUCTURAL: '{meta.get('keyword')}' - skipping auto-reduction")
            continue
            
        keyword_lower = meta.get("keyword", "").lower()
        keyword_words = set(keyword_lower.split())  # słowa z tej frazy
        original_max = meta.get("target_max", 5)
        
        # Znajdź dłuższe frazy które zawierają WSZYSTKIE słowa z tej frazy
        # (lub tę frazę jako substring)
        containing_keywords = []
        for other_rid, other_kw, other_words in all_keywords:
            if other_rid == rid:
                continue
            if len(other_words) <= len(keyword_words):
                continue  # Dłuższa fraza musi mieć więcej słów
            
            # Sprawdź czy wszystkie słowa z krótkiej frazy są w dłuższej
            # LUB czy krótka fraza jest substringiem dłuższej
            words_match = keyword_words.issubset(set(other_words))
            substring_match = keyword_lower in other_kw
            
            if words_match or substring_match:
                other_meta = firestore_keywords[other_rid]
                containing_keywords.append({
                    "keyword": other_kw,
                    "max": other_meta.get("target_max", 1),
                    "match_type": "words" if words_match else "substring"
                })
        
        if containing_keywords:
            # Oblicz ile razy ta fraza będzie liczona przez dłuższe frazy
            inherited_count = sum(kw["max"] for kw in containing_keywords)
            
            # Obniż target_max o inherited_count (ale min 2)
            adjusted_max = max(2, original_max - inherited_count)
            
            if adjusted_max < original_max:
                firestore_keywords[rid]["target_max"] = adjusted_max
                firestore_keywords[rid]["remaining_max"] = adjusted_max
                firestore_keywords[rid]["original_max"] = original_max
                firestore_keywords[rid]["nested_in"] = [kw["keyword"] for kw in containing_keywords]
                firestore_keywords[rid]["inherited_reduction"] = inherited_count
                
                print(f"[PROJECT] ⚠️ NESTED: '{meta.get('keyword')}' max {original_max}→{adjusted_max} "
                      f"(zawarta w: {[kw['keyword'] for kw in containing_keywords]})")

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    s1_data = data.get("s1_data", {})
    
    project_data = {
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "batches_plan": [],
        "total_batches": 0,
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        "source": source,
        "version": "v25.0",
        "manual_mode": False if source == "n8n-brajen-workflow" else True,
        "output_format": "clean_text_with_headers",
        "s1_data": s1_data,
        "article_mode": article_mode  # 🆕 v44.2: "standard" lub "fast"
    }
    
    # ================================================================
    # 🆕 v36.1: E-E-A-T ANALYSIS (entity_ngram_analyzer)
    # Wykrywa sygnały ekspertyzji, autorytetu i zaufania
    # ================================================================
    try:
        from entity_ngram_analyzer import analyze_eeat, analyze_content_semantics
        
        # Zbierz teksty konkurencji z S1
        competitor_analysis = s1_data.get("competitor_analysis", [])
        competitor_texts = []
        for comp in competitor_analysis:
            if isinstance(comp, dict):
                content = comp.get("content", "") or comp.get("text", "")
                if content:
                    competitor_texts.append(content)
        
        combined_text = "\n\n".join(competitor_texts[:5])  # Max 5 konkurentów
        
        if combined_text and len(combined_text) > 500:
            eeat_analysis = analyze_eeat(combined_text)
            
            # Zapisz wyniki
            if hasattr(eeat_analysis, 'to_dict'):
                project_data["eeat_analysis"] = eeat_analysis.to_dict()
            else:
                project_data["eeat_analysis"] = {
                    "expertise_score": getattr(eeat_analysis, 'expertise_score', 0),
                    "authority_score": getattr(eeat_analysis, 'authority_score', 0),
                    "trust_score": getattr(eeat_analysis, 'trust_score', 0),
                    "expertise_signals": getattr(eeat_analysis, 'expertise_signals', []),
                    "authority_signals": getattr(eeat_analysis, 'authority_signals', []),
                    "trust_signals": getattr(eeat_analysis, 'trust_signals', [])
                }
            
            print(f"[PROJECT] 🏆 E-E-A-T analysis: expertise={project_data['eeat_analysis'].get('expertise_score', 0)}, "
                  f"authority={project_data['eeat_analysis'].get('authority_score', 0)}, "
                  f"trust={project_data['eeat_analysis'].get('trust_score', 0)}")
        else:
            print(f"[PROJECT] ⚠️ E-E-A-T skipped: insufficient competitor text ({len(combined_text)} chars)")
    except ImportError:
        print("[PROJECT] ⚠️ entity_ngram_analyzer not available")
    except Exception as e:
        print(f"[PROJECT] ⚠️ E-E-A-T analysis error: {e}")
    
    batch_plan_dict = None
    semantic_plan = None
    
    # ================================================================
    # 🆕 v36.0: SEMANTIC KEYWORD PLAN (NAJPIERW!)
    # Musi być przed create_article_plan żeby przekazać mu plan
    # ================================================================
    if h2_structure:
        try:
            semantic_plan = create_semantic_keyword_plan(
                h2_structure=h2_structure,
                keywords_state=firestore_keywords,
                s1_data=s1_data,
                main_keyword=topic,
                total_batches=total_planned_batches
            )
            project_data["semantic_keyword_plan"] = semantic_plan
            
            stats = semantic_plan.get("stats", {})
            print(f"[PROJECT] 🎯 Semantic plan created: {stats.get('assigned_count', 0)} keywords assigned, "
                  f"{stats.get('universal_count', 0)} universal")
        except Exception as e:
            print(f"[PROJECT] ⚠️ Semantic keyword plan error: {e}")
            import traceback
            traceback.print_exc()
    
    if BATCH_PLANNER_ENABLED and h2_structure:
        try:
            # v28.1: Zbierz dane dla batch_complexity
            ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.3]
            
            # v28.1: Encje z S1
            entities = []
            for e in s1_data.get("entities", []):
                if isinstance(e, dict):
                    entities.append(e.get("name", str(e)))
                else:
                    entities.append(str(e))
            
            # v28.1: PAA z S1
            paa_questions = [p.get("question", "") for p in s1_data.get("paa", [])]
            
            # 🆕 v44.2: FAST MODE vs STANDARD MODE
            if article_mode == "fast" and create_article_plan_fast:
                print(f"[PROJECT] 🚀 Using FAST MODE (3 batches)")
                article_plan = create_article_plan_fast(
                    h2_structure=h2_structure,
                    keywords_state=firestore_keywords,
                    main_keyword=topic,
                    target_length=target_length,
                    ngrams=ngrams[:15],
                    entities=entities[:10],
                    paa_questions=paa_questions[:5],
                    semantic_keyword_plan=semantic_plan
                )
            else:
                # STANDARD MODE (6-9 batches)
                article_plan = create_article_plan(
                    h2_structure=h2_structure,
                    keywords_state=firestore_keywords,
                    main_keyword=topic,
                    target_length=target_length,
                    ngrams=ngrams[:20],
                    entities=entities[:15],
                    paa_questions=paa_questions[:10],
                    max_batches=6,
                    semantic_keyword_plan=semantic_plan
                )
            batch_plan_dict = article_plan.to_dict()
            project_data["batch_plan"] = batch_plan_dict
            project_data["total_planned_batches"] = article_plan.total_batches
            total_planned_batches = article_plan.total_batches
            print(f"[PROJECT] Generated batch_plan: {article_plan.total_batches} batches, ~{article_plan.total_target_words} words")
        except Exception as e:
            print(f"[PROJECT] batch_plan failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ================================================================
    # 🆕 v30.0: Legal Module - auto-detekcja i pobieranie orzeczeń
    # ================================================================
    if LEGAL_MODULE_ENABLED:
        try:
            project_data = enhance_project_with_legal(
                project_data=project_data,
                main_keyword=topic,
                h2_list=h2_structure
            )
            if project_data.get("detected_category") == "prawo":
                judgments_count = len(project_data.get("legal_judgments", []))
                print(f"[PROJECT] ⚖️ Legal module active: category=prawo, {judgments_count} judgments loaded")
        except Exception as e:
            print(f"[PROJECT] ⚠️ Legal module error: {e}")
    
    # ================================================================
    # 🆕 v36.2: ANTI-FRANKENSTEIN SYSTEM
    # Token Budgeting, Article Memory, Style Fingerprint, Soft Caps
    # ================================================================
    if ANTI_FRANKENSTEIN_ENABLED:
        try:
            project_data = enhance_project_with_anti_frankenstein(
                project_data=project_data,
                h2_structure=h2_structure,
                keywords_state=firestore_keywords,
                s1_data=s1_data,
                main_keyword=topic,
                target_length=target_length
            )
            print(f"[PROJECT] 🧟 Anti-Frankenstein: dynamic_plan={project_data.get('dynamic_batch_plan') is not None}, "
                  f"memory={project_data.get('article_memory') is not None}")
        except Exception as e:
            print(f"[PROJECT] ⚠️ Anti-Frankenstein error: {e}")
    
    # ================================================================
    # 🆕 v43.0: PHRASE HIERARCHY ANALYSIS
    # Analizuje hierarchię fraz PRZED zapisem do Firestore
    # ================================================================
    phrase_hierarchy_data = None
    
    if PHRASE_HIERARCHY_ENABLED:
        try:
            # Przygotuj encje z S1
            s1_entities = []
            entity_seo = s1_data.get("entity_seo", {})
            
            # Encje z entity_seo.entities
            for ent in entity_seo.get("entities", []):
                s1_entities.append({
                    "name": ent.get("text", ent.get("entity", ent.get("name", ""))),
                    "priority": ent.get("priority", "SHOULD"),
                    "importance": ent.get("importance", 0.5)
                })
            
            # MUST topics z topical_coverage
            for topic_item in entity_seo.get("topical_coverage", []):
                if topic_item.get("priority") == "MUST":
                    s1_entities.append({
                        "name": topic_item.get("topic", ""),
                        "priority": "MUST",
                        "importance": 0.9
                    })
            
            # Triplety z entity_relationships
            s1_triplets = entity_seo.get("entity_relationships", [])
            
            # Konwertuj keywords_state do listy
            keywords_list_for_hierarchy = []
            for rid, meta in firestore_keywords.items():
                keywords_list_for_hierarchy.append({
                    "keyword": meta.get("keyword", ""),
                    "type": meta.get("type", "BASIC"),
                    "target_min": meta.get("target_min", 1),
                    "target_max": meta.get("target_max", 5)
                })
            
            # Analizuj hierarchię
            hierarchy = analyze_phrase_hierarchy(
                keywords=keywords_list_for_hierarchy,
                entities=s1_entities,
                triplets=s1_triplets,
                h2_terms=h2_structure
            )
            
            # Konwertuj do dict
            phrase_hierarchy_data = hierarchy_to_dict(hierarchy)
            
            print(f"[PROJECT] 🌳 Phrase hierarchy: "
                  f"{hierarchy.stats.get('roots_count', 0)} roots, "
                  f"{hierarchy.stats.get('entity_phrases', 0)} entity phrases, "
                  f"{hierarchy.stats.get('triplet_phrases', 0)} triplet phrases")
            
        except Exception as e:
            print(f"[PROJECT] ⚠️ Phrase hierarchy error: {e}")
            import traceback
            traceback.print_exc()
            phrase_hierarchy_data = None
    
    # Dodaj do project_data przed sanityzacją
    if phrase_hierarchy_data:
        project_data["phrase_hierarchy"] = phrase_hierarchy_data
    
    # 🆕 v40.1: Sanitize keys before Firestore save
    try:
        project_data = sanitize_for_firestore(project_data)
        print(f"[PROJECT] ✅ Data sanitized for Firestore")
    except Exception as e:
        print(f"[PROJECT] ⚠️ Sanitization warning: {e}")
    
    doc_ref.set(project_data)
    
    # v27.2: Policz ile BASIC vs EXTENDED
    basic_count = sum(1 for k in firestore_keywords.values() if k.get("type", "BASIC").upper() in ["BASIC", "MAIN"])
    extended_count = sum(1 for k in firestore_keywords.values() if k.get("type", "").upper() == "EXTENDED")
    total_keywords = basic_count + extended_count
    
    # ================================================================
    # 🆕 v44.0: DYNAMIC BATCH CALCULATION
    # Zamiast sztywnych progów (100+ → 9 batchy), obliczamy dynamicznie
    # na podstawie 3 constraintów:
    # 1. Słowa per batch (min 400, max 600) → czytelność
    # 2. Frazy per batch (max 12-15) → bez stuffingu
    # 3. H2 per batch (1-2) → struktura
    # ================================================================
    
    auto_scaled = False
    original_batches = total_planned_batches
    original_length = target_length
    h2_count = len(h2_structure)
    
    def calculate_optimal_batches(keywords: int, words: int, h2s: int) -> dict:
        """
        Dynamiczne obliczanie optymalnej liczby batchy.
        
        Constrainty:
        - Min ~400-600 słów per batch (czytelność)
        - Max ~12-15 fraz per batch (bez stuffingu)  
        - Min 1 H2 per batch, max 2 H2 per batch
        
        Wybieramy MAX z constraintów (najostrzejszy limit).
        """
        # Constraint 1: Słowa per batch
        # Target: 450-550 słów/batch dla dobrej czytelności
        WORDS_PER_BATCH_TARGET = 500
        batches_by_words = math.ceil(words / WORDS_PER_BATCH_TARGET)
        
        # Constraint 2: Frazy per batch
        # Max 12 fraz/batch żeby uniknąć stuffingu
        MAX_KEYWORDS_PER_BATCH = 12
        batches_by_keywords = math.ceil(keywords / MAX_KEYWORDS_PER_BATCH)
        
        # Constraint 3: H2 per batch
        # Optymalnie 1-2 H2 na batch (1.5 średnio)
        H2_PER_BATCH_TARGET = 1.5
        batches_by_h2 = math.ceil(h2s / H2_PER_BATCH_TARGET) if h2s > 0 else 1
        
        # Wybierz MAX (najostrzejszy constraint)
        optimal = max(batches_by_words, batches_by_keywords, batches_by_h2)
        
        # Clamp do sensownego zakresu (3-12)
        optimal = max(3, min(12, optimal))
        
        # Określ limiting factor
        if batches_by_keywords >= batches_by_words and batches_by_keywords >= batches_by_h2:
            limiting_factor = "keywords"
            explanation = f"ceil({keywords} fraz / {MAX_KEYWORDS_PER_BATCH} per batch)"
        elif batches_by_words >= batches_by_h2:
            limiting_factor = "word_target"
            explanation = f"ceil({words} słów / {WORDS_PER_BATCH_TARGET} per batch)"
        else:
            limiting_factor = "h2_structure"
            explanation = f"ceil({h2s} H2 / {H2_PER_BATCH_TARGET} per batch)"
        
        # Oblicz odpowiedni target_length
        # Jeśli mamy dużo fraz, potrzebujemy więcej słów żeby je pomieścić
        # Formuła: min 30 słów na frazę (żeby density była ok)
        MIN_WORDS_PER_KEYWORD = 30
        min_words_for_keywords = keywords * MIN_WORDS_PER_KEYWORD
        
        # Nie mniej niż optimal_batches * 400 słów
        min_words_for_batches = optimal * 400
        
        suggested_length = max(words, min_words_for_keywords, min_words_for_batches)
        
        return {
            "optimal_batches": optimal,
            "suggested_length": suggested_length,
            "limiting_factor": limiting_factor,
            "explanation": explanation,
            "breakdown": {
                "by_words": batches_by_words,
                "by_keywords": batches_by_keywords,
                "by_h2": batches_by_h2
            }
        }
    
    # Oblicz optymalną liczbę batchy
    batch_calc = calculate_optimal_batches(total_keywords, target_length, h2_count)
    calculated_batches = batch_calc["optimal_batches"]
    calculated_length = batch_calc["suggested_length"]
    
    # Sprawdź czy trzeba skalować
    if calculated_batches > original_batches or calculated_length > original_length:
        auto_scaled = True
        total_planned_batches = max(original_batches, calculated_batches)
        target_length = max(original_length, calculated_length)
        
        # Zaktualizuj w Firestore
        project_data["total_planned_batches"] = total_planned_batches
        project_data["target_length"] = target_length
        project_data["auto_scaled"] = {
            "reason": f"{total_keywords} fraz, {h2_count} H2 → dynamicznie obliczono",
            "original_batches": original_batches,
            "scaled_batches": total_planned_batches,
            "original_length": original_length,
            "scaled_length": target_length,
            "limiting_factor": batch_calc["limiting_factor"],
            "explanation": batch_calc["explanation"],
            "breakdown": batch_calc["breakdown"]
        }
        doc_ref.update({
            "total_planned_batches": total_planned_batches,
            "target_length": target_length,
            "auto_scaled": project_data["auto_scaled"]
        })
        print(f"[PROJECT] 🔄 DYNAMIC SCALING: {total_keywords} fraz + {h2_count} H2 → {total_planned_batches} batchy "
              f"(limit: {batch_calc['limiting_factor']}, było: {original_batches})")
    else:
        print(f"[PROJECT] ✅ No scaling needed: {total_keywords} fraz → {original_batches} batchy OK")
    
    print(f"[PROJECT] Created project {doc_ref.id}: {topic} ({len(firestore_keywords)} keywords: {basic_count} BASIC, {extended_count} EXTENDED, {total_planned_batches} planned batches)")
    
    # v27.2: WARNING jeśli brak EXTENDED
    warning = None
    if extended_count == 0 and len(firestore_keywords) > 5:
        warning = "⚠️ BRAK FRAZ EXTENDED! Upewnij się że wysyłasz 'type': 'EXTENDED' w keywords_list"

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "keywords_count": len(firestore_keywords),
        "keywords_breakdown": {
            "basic": basic_count,
            "extended": extended_count,
            "total": total_keywords,
            "warning": warning
        },
        "h2_sections": len(h2_structure),
        "h2_structure": h2_structure,  # 🆕 v39.0: Zwracamy wygenerowane H2
        "h2_generation_info": h2_generation_info,  # 🆕 v39.0: Info o generowaniu
        "keyword_conflict_info": keyword_conflict_info,  # 🆕 v40.1: Info o konfliktach fraz
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        # 🆕 v44.0: Dynamic batch calculation info
        "batch_calculation": batch_calc if auto_scaled else {
            "optimal_batches": total_planned_batches,
            "limiting_factor": "user_specified",
            "explanation": "Użyto wartości podanej przez użytkownika",
            "breakdown": {
                "by_words": math.ceil(target_length / 500),
                "by_keywords": math.ceil(total_keywords / 12),
                "by_h2": math.ceil(h2_count / 1.5) if h2_count > 0 else 1
            }
        },
        # 🆕 v35.7: Auto-scaling info (legacy, zachowane dla kompatybilności)
        "auto_scaled": project_data.get("auto_scaled") if auto_scaled else None,
        "source": source,
        "batch_plan": batch_plan_dict,
        "has_featured_snippet": bool(s1_data.get("featured_snippet")),
        # 🆕 v30.0: Legal Module fields
        "detected_category": project_data.get("detected_category", "inne"),
        "legal_module_active": project_data.get("legal_context", {}).get("legal_module_active", False),
        "legal_instruction": project_data.get("legal_instruction"),
        "legal_judgments": project_data.get("legal_judgments", []),
        # 🆕 v43.0: Phrase Hierarchy
        "phrase_hierarchy_enabled": phrase_hierarchy_data is not None,
        "phrase_hierarchy_stats": phrase_hierarchy_data.get("stats", {}) if phrase_hierarchy_data else None,
        "version": "v44.0"
    }), 201


# ================================================================
#  CONVERT KEYWORDS TO EXTENDED (v27.3)
# ================================================================
@project_routes.post("/api/project/<project_id>/convert_to_extended")
def convert_to_extended(project_id):
    """
    v27.3: Konwertuje wybrane frazy BASIC na EXTENDED.
    
    Body:
    {
        "keywords": ["fraza1", "fraza2", ...]  // lista fraz do konwersji
    }
    
    lub
    
    {
        "all_with_target_1": true  // konwertuj wszystkie z target_max=1
    }
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    
    body = request.get_json() or {}
    keywords_to_convert = body.get("keywords", [])
    all_with_target_1 = body.get("all_with_target_1", False)
    
    converted = []
    skipped = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        
        # Pomiń już EXTENDED i MAIN
        if kw_type == "EXTENDED" or kw_type == "MAIN" or is_main:
            continue
        
        should_convert = False
        
        # Konwertuj jeśli na liście
        if keyword.lower() in [k.lower() for k in keywords_to_convert]:
            should_convert = True
        
        # Konwertuj jeśli target_max=1 i flaga ustawiona
        if all_with_target_1 and target_max == 1:
            should_convert = True
        
        if should_convert:
            keywords_state[rid]["type"] = "EXTENDED"
            keywords_state[rid]["target_min"] = 1
            keywords_state[rid]["target_max"] = max(1, target_max)
            converted.append(keyword)
        else:
            if keyword in keywords_to_convert:
                skipped.append({"keyword": keyword, "reason": "not found or already EXTENDED/MAIN"})
    
    # Zapisz do Firestore
    if converted:
        doc_ref.update({"keywords_state": sanitize_for_firestore(keywords_state)})
    
    # Policz nowe statystyki
    basic_count = sum(1 for k in keywords_state.values() if k.get("type", "BASIC").upper() in ["BASIC"])
    extended_count = sum(1 for k in keywords_state.values() if k.get("type", "").upper() == "EXTENDED")
    main_count = sum(1 for k in keywords_state.values() if k.get("type", "").upper() == "MAIN" or k.get("is_main_keyword"))
    
    return jsonify({
        "status": "OK",
        "converted": converted,
        "converted_count": len(converted),
        "skipped": skipped,
        "keywords_breakdown": {
            "main": main_count,
            "basic": basic_count,
            "extended": extended_count,
            "total": len(keywords_state)
        }
    }), 200


# ================================================================
#  GET PROJECT STATUS
# ================================================================
@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """Zwraca aktualny status projektu z coverage info."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    
    # v25.0: Coverage
    coverage = validate_coverage(keywords_state)
    
    keyword_summary = []
    locked_keywords = []
    near_limit_keywords = []
    
    main_keyword_uses = 0
    synonym_uses = 0
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        
        kw_info = {
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "status": meta.get("status"),
            "remaining_max": remaining,
            "is_main_keyword": meta.get("is_main_keyword", False),
            "is_synonym_of_main": meta.get("is_synonym_of_main", False)
        }
        keyword_summary.append(kw_info)
        
        if meta.get("is_main_keyword"):
            main_keyword_uses = actual
        elif meta.get("is_synonym_of_main"):
            synonym_uses += actual
        
        if remaining == 0:
            locked_keywords.append({
                "keyword": meta.get("keyword"),
                "message": f" LOCKED: '{meta.get('keyword')}' osiągnęło limit {target_max}x"
            })
        elif remaining <= 3:
            near_limit_keywords.append({
                "keyword": meta.get("keyword"),
                "remaining": remaining
            })
    
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 1.0
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "main_keyword": main_keyword,
        "batch_count": len(batches),
        "total_planned_batches": data.get("total_planned_batches", 4),
        "keywords_summary": keyword_summary,
        "locked_keywords": locked_keywords,
        "near_limit_keywords": near_limit_keywords,
        "coverage": coverage,
        "main_vs_synonyms": {
            "main_uses": main_keyword_uses,
            "synonym_uses": synonym_uses,
            "main_ratio": round(main_ratio, 2),
            "valid": main_ratio >= 0.3
        },
        "version": "v25.0"
    }), 200


# ================================================================
#  PRE-BATCH INFO - v25.0
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """v36.0: Używa semantic_keyword_plan dla inteligentnego rozmieszczenia fraz."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    
    # 🆕 v41.1 FIX: Safe batches extraction (fixes "unhashable type: 'slice'" error)
    raw_batches = data.get("batches", [])
    if raw_batches is None:
        batches = []
    elif isinstance(raw_batches, list):
        batches = raw_batches
    else:
        try:
            batches = list(raw_batches) if hasattr(raw_batches, '__iter__') else []
        except Exception as e:
            print(f"[PRE_BATCH] ⚠️ batches conversion error: {e}")
            batches = []
    
    # 🆕 v41.1 FIX: Safe batch texts extraction helper
    def safe_batch_texts(batch_list):
        """Safely extract text from batches, handling malformed data."""
        try:
            if not batch_list:
                return []
            texts = []
            for b in batch_list:
                if isinstance(b, dict):
                    texts.append(str(b.get("text", "")))
                elif isinstance(b, str):
                    texts.append(b)
            return texts
        except Exception as e:
            print(f"[PRE_BATCH] ⚠️ safe_batch_texts error: {e}")
            return []
    
    # Pre-compute batch texts for reuse
    batch_texts = safe_batch_texts(batches)
    
    h2_structure = data.get("h2_structure", [])
    total_planned_batches = data.get("total_planned_batches", 4)
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    main_keyword_synonyms = data.get("main_keyword_synonyms", [])
    s1_data = data.get("s1_data", {})
    
    # v28.1: Pobierz batch_plan
    batch_plan = data.get("batch_plan", {})
    
    # 🆕 v36.0: Pobierz semantic_keyword_plan
    semantic_plan = data.get("semantic_keyword_plan", {})
    
    current_batch_num = len(batches) + 1
    remaining_batches = max(1, total_planned_batches - len(batches))
    
    # Batch type
    if current_batch_num == 1:
        batch_type = "INTRO"
    elif current_batch_num >= total_planned_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # Intro guidance
    intro_guidance = None
    if batch_type == "INTRO":
        featured_snippet = s1_data.get("featured_snippet", {})
        ai_overview = s1_data.get("ai_overview", {})  # v27.1: Google SGE
        serp_analysis = s1_data.get("serp_analysis", {})
        
        # Fallback - ai_overview może być w serp_analysis
        if not ai_overview:
            ai_overview = serp_analysis.get("ai_overview", {})
        
        intro_guidance = {
            "direct_answer_required": True,
            "direct_answer_length": "40-60 slow",
            "first_sentence_must_contain": main_keyword,
            "featured_snippet": None,
            "ai_overview": None  # v27.1
        }
        
        # Featured Snippet
        if featured_snippet and featured_snippet.get("answer"):
            intro_guidance["featured_snippet"] = {
                "google_answer": featured_snippet.get("answer", "")[:500],
                "source_type": featured_snippet.get("type", "unknown"),
                "hint": "Napisz LEPSZA, pelniejsza wersje tej odpowiedzi. NIE kopiuj."
            }
        
        # v27.1: AI Overview (Google SGE)
        if ai_overview and ai_overview.get("text"):
            intro_guidance["ai_overview"] = {
                "google_sge_answer": ai_overview.get("text", "")[:800],
                "sources_count": len(ai_overview.get("sources", [])),
                "hint": "Google SGE pokazuje te informacje. Twój wstep powinien byc LEPSZY i bardziej szczegolowy."
            }
    
    # Coverage
    coverage = validate_coverage(keywords_state)
    
    # Density - v27.3: per keyword
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    current_density = 0
    density_details = {}
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        current_density = prevalidation.get("density", 0)
        density_details = prevalidation.get("density_details", {})
    
    density_status, density_msg = get_density_status(current_density)
    
    # Main vs synonyms
    main_keyword_uses = 0
    synonym_uses = 0
    main_keyword_meta = None
    
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            main_keyword_uses = meta.get("actual_uses", 0)
            main_keyword_meta = meta
        elif meta.get("is_synonym_of_main"):
            synonym_uses += meta.get("actual_uses", 0)
    
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 1.0
    
    ratio_warning = None
    if current_batch_num > 1 and main_ratio < 0.30:
        ratio_warning = f"⚠️ Main keyword ratio {main_ratio:.0%} < 30%. Użyj więcej '{main_keyword}'!"
    
    # v33.3: Wcześniejsze obliczenie remaining_h2 dla dopasowania n-gramów
    h2_structure = data.get("h2_structure", [])
    used_h2_early = []
    for batch in batches:
        batch_text = batch.get("text", "")
        h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
        used_h2_early.extend([(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]])
    remaining_h2_early = [h2 for h2 in h2_structure if h2 not in used_h2_early]
    
    # v33.3: N-gramy dopasowane do H2 (zamiast sekwencyjnych)
    ngrams = s1_data.get("ngrams", [])
    top_ngrams_objs = [n for n in ngrams if n.get("weight", 0) > 0.4][:15]
    top_ngrams = [n.get("ngram", "") for n in top_ngrams_objs]
    
    # Pobierz użyte n-gramy z poprzednich batchów
    batches_so_far = data.get("batches", [])
    used_ngrams = get_used_ngrams_from_batches(batches_so_far, top_ngrams)
    
    # Pobierz H2 dla tego batcha
    current_h2 = remaining_h2_early[0] if remaining_h2_early else main_keyword
    
    # v33.3: Dopasuj n-gramy do H2 zamiast sekwencyjnego przydzielania
    if current_batch_num == 1:
        # Batch 1 (intro) - użyj n-gramów związanych z main keyword
        batch_ngrams = get_ngrams_for_h2(main_keyword, top_ngrams_objs, used_ngrams, max_ngrams=4)
    else:
        # Pozostałe batche - dopasuj do H2
        batch_ngrams = get_ngrams_for_h2(current_h2, top_ngrams_objs, used_ngrams, max_ngrams=4)
    
    # Fallback na sekwencyjne jeśli brak dopasowań
    if not batch_ngrams and top_ngrams:
        ngrams_per_batch = max(3, len(top_ngrams) // total_planned_batches)
        start_idx = (current_batch_num - 1) * ngrams_per_batch
        end_idx = min(start_idx + ngrams_per_batch + 2, len(top_ngrams))
        batch_ngrams = top_ngrams[start_idx:end_idx]
    
    # v28.0: Entity SEO - wyciągnij encje z s1_data
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    entity_relationships = entity_seo.get("entity_relationships", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    # Top encje do wspomnienia (max 8)
    top_entities = [e for e in entities if e.get("importance", 0) > 0.5][:8]
    # Top relacje (max 5)
    top_relationships = entity_relationships[:5]
    # MUST topics
    must_topics = [t for t in topical_coverage if t.get("priority") == "MUST"][:5]
    
    # v28.0: Dodatkowe dane z S1
    serp_analysis = s1_data.get("serp_analysis", {})
    
    # PAA - pytania użytkowników
    paa_questions = serp_analysis.get("paa_questions", [])
    paa_for_batch = []
    if paa_questions and current_batch_num <= 3:  # PAA tylko w pierwszych 3 batchach
        paa_per_batch = max(1, len(paa_questions) // 3)
        start_paa = (current_batch_num - 1) * paa_per_batch
        paa_for_batch = paa_questions[start_paa:start_paa + paa_per_batch][:2]
    
    # Related searches - powiązane tematy
    related_searches = serp_analysis.get("related_searches", [])[:6]
    
    # Semantic keyphrases (LSI) - jeśli dostępne
    semantic_keyphrases = s1_data.get("semantic_keyphrases", [])
    lsi_keywords = [kp.get("phrase", "") for kp in semantic_keyphrases if kp.get("score", 0) > 0.7][:6]
    
    # ================================================================
    # 🆕 v36.6: UNIFIED KEYWORD DISTRIBUTION
    # Używa semantic_plan jako ŹRÓDŁO PRAWDY, z dynamicznym sprawdzaniem limitów
    # ================================================================
    semantic_batch_plan_early = None
    semantic_assigned = []      # Frazy PRZYPISANE tematycznie do tego batcha
    semantic_universal = []     # Frazy UNIWERSALNE (wysoki limit)
    semantic_reserved = []      # Frazy ZAREZERWOWANE na inne batche
    
    if semantic_plan and "batch_plans" in semantic_plan:
        for bp in semantic_plan.get("batch_plans", []):
            if bp.get("batch_number") == current_batch_num:
                semantic_batch_plan_early = bp
                semantic_assigned = bp.get("assigned_keywords", [])
                semantic_universal = bp.get("universal_keywords", [])
                semantic_reserved = bp.get("reserved_keywords", [])
                print(f"[PRE_BATCH] 🎯 Semantic plan: {len(semantic_assigned)} assigned + {len(semantic_universal)} universal")
                break
    
    # Zbuduj słownik keyword → meta dla szybkiego lookup
    keyword_meta_map = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if keyword:
            keyword_meta_map[keyword.lower()] = {
                "rid": rid,
                "meta": meta,
                "keyword": keyword
            }
    
    # ================================================================
    # KROK 1: Przetwórz ASSIGNED keywords z semantic_plan
    # To są frazy TEMATYCZNIE dopasowane do tego H2/batcha
    # ================================================================
    basic_must_use = []
    extended_this_batch = []
    main_keyword_info = None
    
    for kw in semantic_assigned:
        kw_lower = kw.lower()
        if kw_lower not in keyword_meta_map:
            continue
        
        data = keyword_meta_map[kw_lower]
        meta = data["meta"]
        keyword = data["keyword"]
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        is_synonym = meta.get("is_synonym_of_main", False)
        
        # Oblicz suggested dla tego batcha
        suggested_info = calculate_suggested_v25(
            keyword=keyword,
            kw_type=kw_type,
            actual=actual,
            target_min=target_min,
            target_max=target_max,
            remaining_batches=remaining_batches,
            total_batches=total_planned_batches,
            current_batch=current_batch_num,
            is_main=is_main
        )
        
        suggested_use = suggested_info["suggested"]
        hard_max = suggested_info["hard_max_this_batch"]
        
        # Pomiń jeśli limit już przekroczony
        if actual >= target_max:
            continue
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_total": f"{target_min}-{target_max}",
            "use_this_batch": f"{suggested_use}-{hard_max}" if suggested_use > 0 else "1",
            "suggested": max(1, suggested_use),  # Min 1 dla assigned
            "priority": "ASSIGNED",  # Tematycznie przypisane!
            "instruction": f"✅ Przypisana do tej sekcji - użyj 1×",
            "hard_max_this_batch": max(1, hard_max),
            "flexibility": "LOW",
            "is_main": is_main,
            "is_synonym": is_synonym,
            "source": "semantic_assigned"
        }
        
        if kw_type == "EXTENDED":
            extended_this_batch.append(kw_info)
        else:
            basic_must_use.append(kw_info)
    
    # ================================================================
    # KROK 2: Przetwórz UNIVERSAL keywords
    # To są frazy z wysokim limitem które mogą być wszędzie
    # ================================================================
    for kw in semantic_universal:
        kw_lower = kw.lower()
        if kw_lower not in keyword_meta_map:
            continue
        
        data = keyword_meta_map[kw_lower]
        meta = data["meta"]
        keyword = data["keyword"]
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        is_synonym = meta.get("is_synonym_of_main", False)
        
        suggested_info = calculate_suggested_v25(
            keyword=keyword,
            kw_type=kw_type,
            actual=actual,
            target_min=target_min,
            target_max=target_max,
            remaining_batches=remaining_batches,
            total_batches=total_planned_batches,
            current_batch=current_batch_num,
            is_main=is_main
        )
        
        suggested_use = suggested_info["suggested"]
        hard_max = suggested_info["hard_max_this_batch"]
        
        if actual >= target_max:
            continue
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_total": f"{target_min}-{target_max}",
            "use_this_batch": f"{suggested_use}-{hard_max}",
            "suggested": suggested_use,
            "priority": "UNIVERSAL" if is_main else "HIGH",
            "instruction": f"🔄 Uniwersalna - użyj {suggested_use}-{hard_max}×" if not is_main else f"🎯 GŁÓWNA - użyj {suggested_use}-{hard_max}×",
            "hard_max_this_batch": hard_max,
            "flexibility": "MEDIUM",
            "is_main": is_main,
            "is_synonym": is_synonym,
            "source": "semantic_universal"
        }
        
        if is_main:
            main_keyword_info = kw_info
        elif kw_type == "EXTENDED":
            # Universal EXTENDED - dodaj tylko jeśli nie ma jeszcze dużo
            if len(extended_this_batch) < 5:
                extended_this_batch.append(kw_info)
        else:
            # Universal BASIC - dodaj do must_use
            basic_must_use.append(kw_info)
    
    # ================================================================
    # KROK 3: FALLBACK - jeśli brak semantic_plan, użyj starej logiki
    # ================================================================
    basic_done = []
    basic_target = []
    extended_done = []
    extended_scheduled = []
    locked_exceeded = []
    
    if not semantic_batch_plan_early:
        print(f"[PRE_BATCH] ⚠️ Brak semantic_plan, używam fallback logic")
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            
            kw_type = meta.get("type", "BASIC").upper()
            actual = meta.get("actual_uses", 0)
            target_min = meta.get("target_min", 1)
            target_max = meta.get("target_max", 999)
            is_main = meta.get("is_main_keyword", False)
            is_synonym = meta.get("is_synonym_of_main", False)
            
            suggested_info = calculate_suggested_v25(
                keyword=keyword,
                kw_type=kw_type,
                actual=actual,
                target_min=target_min,
                target_max=target_max,
                remaining_batches=remaining_batches,
                total_batches=total_planned_batches,
                current_batch=current_batch_num,
                is_main=is_main
            )
            
            suggested_use = suggested_info["suggested"]
            hard_max = suggested_info["hard_max_this_batch"]
            
            kw_info = {
                "keyword": keyword,
                "type": kw_type,
                "actual": actual,
                "target_total": f"{target_min}-{target_max}",
                "use_this_batch": f"{suggested_use}-{hard_max}" if suggested_use > 0 else "0",
                "suggested": suggested_use,
                "priority": suggested_info["priority"],
                "instruction": suggested_info["instruction"],
                "hard_max_this_batch": hard_max,
                "flexibility": suggested_info["flexibility"],
                "is_main": is_main,
                "is_synonym": is_synonym,
                "source": "fallback"
            }
            
            if is_main:
                main_keyword_info = kw_info
            elif suggested_info["priority"] in ["EXCEEDED", "LOCKED"]:
                locked_exceeded.append(kw_info)
            elif kw_type == "EXTENDED":
                if suggested_info["priority"] == "DONE":
                    extended_done.append(keyword)
                else:
                    extended_this_batch.append(kw_info)
            else:
                if actual == 0 and hard_max > 0:
                    basic_must_use.append(kw_info)
                elif actual < target_min and hard_max > 0:
                    basic_target.append(kw_info)
                else:
                    basic_done.append(kw_info)
    
    # ================================================================
    # KROK 4: Zbierz info o RESERVED keywords (dla wyświetlenia w prompt)
    # ================================================================
    reserved_for_display = []
    for rk in semantic_reserved[:15]:
        if isinstance(rk, dict):
            reserved_for_display.append({
                "keyword": rk.get("keyword", ""),
                "reserved_for_h2": rk.get("reserved_for_h2", ""),
                "reserved_for_batch": rk.get("reserved_for_batch", 0)
            })
    
    # Policz statystyki
    total_unused_basic = len(basic_must_use)
    total_unused_extended = len(extended_this_batch)
    
    # Used H2
    used_h2 = []
    for batch in batches:
        batch_text = batch.get("text", "")
        h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
        used_h2.extend([(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]])
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    # Last sentences
    last_sentences = ""
    if batches:
        last_batch_text = batches[-1].get("text", "")
        clean_last = re.sub(r'<[^>]+>', '', last_batch_text)
        clean_last = re.sub(r'^h[23]:\s*.+$', '', clean_last, flags=re.MULTILINE)
        sentences = re.split(r'[.!?]+', clean_last)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        if len(sentences) >= 2:
            last_sentences = ". ".join(sentences[-2:]) + "."
        elif sentences:
            last_sentences = sentences[-1] + "."
    
    # GPT Prompt - v27.4: WYMUSZAJĄCY użycie fraz
    prompt_sections = []
    prompt_sections.append("="*60)
    prompt_sections.append("⚠️ KRYTYCZNE INSTRUKCJE - PRZECZYTAJ UWAŻNIE!")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append(f"📝 BATCH #{current_batch_num} z {total_planned_batches} ({batch_type})")
    prompt_sections.append("")
    
    basic_cov = coverage.get("basic", {}).get("coverage_percent", 100)
    ext_cov = coverage.get("extended", {}).get("coverage_percent", 100)
    prompt_sections.append(f"📊 COVERAGE: BASIC {basic_cov:.0f}% | EXTENDED {ext_cov:.0f}%")
    
    # v27.3: Density per keyword
    max_density = density_details.get("max_density", 0)
    prompt_sections.append(f"📈 DENSITY: main={current_density:.1f}% | max={max_density:.1f}% ({density_status})")
    
    # Pokaż warnings jeśli są
    density_warnings = density_details.get("warnings", [])
    if density_warnings:
        for warn in density_warnings[:3]:
            prompt_sections.append(f"   {warn}")
    prompt_sections.append("")
    
    if ratio_warning:
        prompt_sections.append(f"⚠️ {ratio_warning}")
        prompt_sections.append("")
    
    # ================================================================
    # 🆕 v36.0: SEMANTIC CONTEXT DLA TEGO BATCHA
    # 🆕 v36.5: Użyj już pobranego semantic_batch_plan_early
    # ================================================================
    semantic_batch_plan = semantic_batch_plan_early  # Już pobrane wcześniej
    
    if semantic_batch_plan:
        prompt_sections.append("=" * 60)
        prompt_sections.append("🎯 KONTEKST SEMANTYCZNY TEJ SEKCJI")
        prompt_sections.append("=" * 60)
        prompt_sections.append("")
        
        # H2 i kategoria
        batch_h2 = semantic_batch_plan.get("h2")
        batch_category = semantic_batch_plan.get("h2_category")
        if batch_h2:
            prompt_sections.append(f"📌 H2: \"{batch_h2}\"")
            if batch_category:
                prompt_sections.append(f"📂 KATEGORIA: {batch_category}")
            prompt_sections.append("")
        
        # Encje do wprowadzenia
        assigned_entities = semantic_batch_plan.get("assigned_entities", [])
        if assigned_entities:
            prompt_sections.append("🧠 ENCJE DO WPROWADZENIA W TEJ SEKCJI:")
            for entity in assigned_entities[:6]:
                prompt_sections.append(f"   • {entity}")
            prompt_sections.append("")
        
        # N-gramy powiązane
        assigned_ngrams = semantic_batch_plan.get("assigned_ngrams", [])
        if assigned_ngrams:
            prompt_sections.append("🔗 N-GRAMY/FRAZY POWIĄZANE:")
            for ngram in assigned_ngrams[:5]:
                prompt_sections.append(f"   • \"{ngram}\"")
            prompt_sections.append("")
        
        # Frazy PRZYPISANE do tej sekcji (główna lista)
        assigned_keywords = semantic_batch_plan.get("assigned_keywords", [])
        if assigned_keywords:
            prompt_sections.append("✅ FRAZY PRZYPISANE DO TEJ SEKCJI (użyj tu!):")
            for kw in assigned_keywords[:15]:
                # Pobierz info o limicie
                kw_meta = None
                for rid, meta in keywords_state.items():
                    if meta.get("keyword", "").lower() == kw.lower():
                        kw_meta = meta
                        break
                
                if kw_meta:
                    actual = kw_meta.get("actual_uses", 0)
                    target_max = kw_meta.get("target_max", 5)
                    remaining = max(0, target_max - actual)
                    if remaining > 0:
                        prompt_sections.append(f"   ✅ \"{kw}\" ({actual}/{target_max}) → użyj 1×")
                else:
                    prompt_sections.append(f"   ✅ \"{kw}\" → użyj 1×")
            prompt_sections.append("")
        
        # Frazy UNIWERSALNE
        universal_keywords = semantic_batch_plan.get("universal_keywords", [])
        if universal_keywords:
            prompt_sections.append("🔄 FRAZY UNIWERSALNE (możesz użyć w każdym batchu):")
            for kw in universal_keywords[:8]:
                # Pobierz info o limicie
                kw_meta = None
                for rid, meta in keywords_state.items():
                    if meta.get("keyword", "").lower() == kw.lower():
                        kw_meta = meta
                        break
                
                if kw_meta:
                    actual = kw_meta.get("actual_uses", 0)
                    target_max = kw_meta.get("target_max", 99)
                    remaining = max(0, target_max - actual)
                    max_here = min(3, math.ceil(remaining / remaining_batches)) if remaining_batches > 0 else remaining
                    if remaining > 0:
                        prompt_sections.append(f"   🔄 \"{kw}\" ({actual}/{target_max}) → max {max_here}× tu")
                    else:
                        prompt_sections.append(f"   🛑 \"{kw}\" ({actual}/{target_max}) → LIMIT PEŁNY!")
                else:
                    prompt_sections.append(f"   🔄 \"{kw}\"")
            prompt_sections.append("")
        
        # Frazy ZAREZERWOWANE na inne sekcje
        reserved_keywords = semantic_batch_plan.get("reserved_keywords", [])
        if reserved_keywords:
            prompt_sections.append("⏳ ZAREZERWOWANE NA INNE SEKCJE (nie używaj tu!):")
            for rk in reserved_keywords[:10]:
                kw = rk.get("keyword", "")
                reserved_h2 = rk.get("reserved_for_h2", "")
                if reserved_h2:
                    prompt_sections.append(f"   ⏳ \"{kw}\" → sekcja \"{reserved_h2[:40]}...\"")
                else:
                    prompt_sections.append(f"   ⏳ \"{kw}\" → inny batch")
            if len(reserved_keywords) > 10:
                prompt_sections.append(f"   ... i {len(reserved_keywords) - 10} więcej")
            prompt_sections.append("")
        
        prompt_sections.append("=" * 60)
        prompt_sections.append("")
    
    # v27.4: Oblicz ile fraz MUSI być użytych w tym batchu
    total_unused_basic = len(basic_must_use)
    total_unused_extended = len(extended_this_batch) + len(extended_scheduled)
    total_unused = total_unused_basic + total_unused_extended
    
    # 🆕 v35.8: UŻYJ keywords_budget z batch_plan jeśli dostępny!
    basic_this_batch_count = None
    extended_this_batch_count = None
    used_batch_plan_budget = False
    
    if batch_plan and "batches" in batch_plan:
        for bp in batch_plan.get("batches", []):
            if bp.get("batch_number") == current_batch_num:
                keywords_budget = bp.get("keywords_budget", {})
                if keywords_budget:
                    # Policz ile BASIC i EXTENDED w budżecie
                    basic_in_budget = 0
                    extended_in_budget = 0
                    
                    for kw, count in keywords_budget.items():
                        if count > 0:
                            # Sprawdź typ frazy
                            for rid, meta in keywords_state.items():
                                if meta.get("keyword", "").lower() == kw.lower():
                                    kw_type = meta.get("type", "BASIC").upper()
                                    if kw_type == "EXTENDED":
                                        extended_in_budget += 1
                                    else:
                                        basic_in_budget += 1
                                    break
                    
                    if basic_in_budget > 0 or extended_in_budget > 0:
                        basic_this_batch_count = basic_in_budget
                        extended_this_batch_count = extended_in_budget
                        used_batch_plan_budget = True
                        print(f"[PRE_BATCH] ✅ Używam keywords_budget z batch_plan: {basic_in_budget} BASIC, {extended_in_budget} EXTENDED")
                break
    
    # FALLBACK: jeśli brak keywords_budget, użyj równomiernego podziału
    if basic_this_batch_count is None:
        if remaining_batches > 0:
            basic_this_batch_count = max(3, math.ceil(total_unused_basic / remaining_batches))
            extended_this_batch_count = max(2, math.ceil(total_unused_extended / remaining_batches))
        else:
            basic_this_batch_count = total_unused_basic
            extended_this_batch_count = total_unused_extended
        
        if not used_batch_plan_budget:
            print(f"[PRE_BATCH] ⚠️ Brak keywords_budget, używam fallback: {basic_this_batch_count} BASIC, {extended_this_batch_count} EXTENDED")
    
    # 🆕 v35.8: DOSTOSUJ liczbę fraz do profilu długości (jeśli batch_plan dostępny)
    # Dłuższe batche = więcej fraz, krótsze = mniej
    if batch_plan and "batches" in batch_plan:
        for bp in batch_plan.get("batches", []):
            if bp.get("batch_number") == current_batch_num:
                lp = bp.get("length_profile", "medium")
                
                # Mnożniki dla profili
                PROFILE_MULTIPLIERS = {
                    "intro": 0.5,      # intro = mało fraz
                    "short": 0.7,      # krótki = mniej fraz
                    "medium": 1.0,     # medium = baseline
                    "long": 1.3,       # długi = więcej fraz
                    "extended": 1.6    # extended = dużo fraz
                }
                
                multiplier = PROFILE_MULTIPLIERS.get(lp, 1.0)
                
                if multiplier != 1.0:
                    original_basic = basic_this_batch_count
                    original_extended = extended_this_batch_count
                    
                    basic_this_batch_count = max(2, int(basic_this_batch_count * multiplier))
                    extended_this_batch_count = max(1, int(extended_this_batch_count * multiplier))
                    
                    # Nie więcej niż dostępne
                    basic_this_batch_count = min(basic_this_batch_count, total_unused_basic)
                    extended_this_batch_count = min(extended_this_batch_count, total_unused_extended)
                    
                    print(f"[PRE_BATCH] 📊 Profil '{lp}' (×{multiplier}): BASIC {original_basic}→{basic_this_batch_count}, EXTENDED {original_extended}→{extended_this_batch_count}")
                break
    
    # ================================================================
    # 🆕 v36.7: UNIFIED APPROACH
    # Jeśli mamy semantic_plan → użyj WSZYSTKIE assigned (nie tnij!)
    # keywords_budget kontroluje ILE RAZY, nie KTÓRE frazy
    # ================================================================
    if semantic_batch_plan_early:
        # Mamy semantic_plan → basic_must_use zawiera TYLKO assigned+universal
        # Użyj WSZYSTKIE (nie tnij przez basic_this_batch_count)
        basic_for_this_batch = basic_must_use  # Wszystkie!
        extended_for_this_batch = extended_this_batch  # Wszystkie!
        print(f"[PRE_BATCH] ✅ Semantic mode: {len(basic_for_this_batch)} BASIC, {len(extended_for_this_batch)} EXTENDED (all assigned)")
    else:
        # Brak semantic_plan → użyj starej logiki z count
        basic_for_this_batch = basic_must_use[:basic_this_batch_count]
        extended_for_this_batch = extended_this_batch[:extended_this_batch_count]
        print(f"[PRE_BATCH] ⚠️ Fallback mode: {len(basic_for_this_batch)} BASIC, {len(extended_for_this_batch)} EXTENDED (limited)")
    
    prompt_sections.append("="*60)
    prompt_sections.append("🔴🔴🔴 OBOWIĄZKOWE FRAZY DO UŻYCIA W TYM BATCHU 🔴🔴🔴")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append("❗ KAŻDA fraza z poniższej listy MUSI pojawić się w tekście!")
    prompt_sections.append("❗ Nie możesz pominąć ŻADNEJ frazy - to warunek konieczny!")
    prompt_sections.append("❗ Wpleć frazy naturalnie w zdania, nie zmieniaj ich formy!")
    prompt_sections.append("")
    
    if main_keyword_info:
        prompt_sections.append(f"🎯 FRAZA GŁÓWNA: \"{main_keyword}\"")
        prompt_sections.append(f"   → Użyj DOKŁADNIE {main_keyword_info['use_this_batch']}x w tym batchu")
        prompt_sections.append("")
    
    if basic_for_this_batch:
        prompt_sections.append(f"📋 BASIC - MUSISZ UŻYĆ WSZYSTKIE ({len(basic_for_this_batch)} fraz):")
        for i, kw in enumerate(basic_for_this_batch, 1):
            prompt_sections.append(f"   {i}. \"{kw['keyword']}\" ← OBOWIĄZKOWO 1x")
        prompt_sections.append("")
    
    if extended_for_this_batch:
        prompt_sections.append(f"📋 EXTENDED - MUSISZ UŻYĆ WSZYSTKIE ({len(extended_for_this_batch)} fraz):")
        for i, kw in enumerate(extended_for_this_batch, 1):
            prompt_sections.append(f"   {i}. \"{kw['keyword']}\" ← OBOWIĄZKOWO 1x")
        prompt_sections.append("")
    
    # ================================================================
    # 🆕 v35.9: WYRAŹNE LIMITY NA BATCH
    # ================================================================
    prompt_sections.append("")
    prompt_sections.append("=" * 60)
    prompt_sections.append("📊 LIMITY FRAZ - SPRAWDŹ ZANIM NAPISZESZ!")
    prompt_sections.append("=" * 60)
    prompt_sections.append("")
    
    # Zbierz wszystkie frazy z limitami
    all_keywords_limits = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
            
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        
        remaining_to_max = max(0, target_max - actual)
        
        # 🆕 v35.9: INTELIGENTNE PRZYPISANIE FRAZ DO BATCHY
        # Frazy z niskim limitem (remaining < remaining_batches) są przypisane do KONKRETNYCH batchy
        if remaining_batches > 0:
            if remaining_to_max == 0:
                # Limit wyczerpany
                max_this_batch = 0
            elif remaining_to_max < remaining_batches:
                # NISKI LIMIT: przypisz do konkretnych batchy
                # Używamy hash aby deterministycznie wybrać które batche
                keyword_hash = hash(keyword) % 1000
                
                # Oblicz które batche "należą" do tej frazy
                # np. remaining=2, batches=6 → fraza należy do batcha 2 i 5
                batch_spacing = remaining_batches // remaining_to_max if remaining_to_max > 0 else remaining_batches
                assigned_batches = []
                for i in range(remaining_to_max):
                    # Rozłóż równomiernie + offset z hash
                    batch_offset = (keyword_hash + i * batch_spacing) % remaining_batches
                    assigned_batch = current_batch_num + batch_offset
                    if assigned_batch <= total_planned_batches:
                        assigned_batches.append(assigned_batch)
                
                # Czy TEN batch jest przypisany do tej frazy?
                if current_batch_num in assigned_batches:
                    max_this_batch = 1
                else:
                    max_this_batch = 0
            else:
                # NORMALNY LIMIT: podziel równo
                max_this_batch = math.ceil(remaining_to_max / remaining_batches)
        else:
            max_this_batch = remaining_to_max
        
        # Status
        if actual >= target_max:
            status = "🛑 STOP"
            max_this_batch = 0
        elif remaining_to_max <= 2 and remaining_to_max > 0:
            status = "⚠️ OSTROŻNIE"
        elif actual >= target_min:
            status = "✅ OK"
        else:
            status = "📌 UŻYJ"
        
        all_keywords_limits.append({
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_max": target_max,
            "remaining": remaining_to_max,
            "max_this_batch": max_this_batch,
            "status": status
        })
    
    # Sortuj: STOP na górze, potem OSTROŻNIE
    priority_order = {"🛑 STOP": 0, "⚠️ OSTROŻNIE": 1, "📌 UŻYJ": 2, "✅ OK": 3}
    all_keywords_limits.sort(key=lambda x: (priority_order.get(x["status"], 99), -x["actual"]))
    
    # Pokaż STOP (limit wyczerpany)
    stop_keywords = [k for k in all_keywords_limits if k["status"] == "🛑 STOP"]
    caution_keywords = [k for k in all_keywords_limits if k["status"] == "⚠️ OSTROŻNIE"]
    
    if stop_keywords:
        prompt_sections.append("🛑🛑🛑 STOP! NIE UŻYWAJ TYCH FRAZ (limit wyczerpany):")
        for kw in stop_keywords[:15]:
            prompt_sections.append(f"   ❌ \"{kw['keyword']}\" = {kw['actual']}/{kw['target_max']} → NIE UŻYWAJ!")
        prompt_sections.append("")
    
    if caution_keywords:
        prompt_sections.append("⚠️ OSTROŻNIE - bliskie limitu:")
        for kw in caution_keywords[:10]:
            prompt_sections.append(f"   ⚠️ \"{kw['keyword']}\" = {kw['actual']}/{kw['target_max']} → max {kw['max_this_batch']}× więcej")
        prompt_sections.append("")
    
    # Top 10 fraz z zapasem
    available_keywords = [k for k in all_keywords_limits if k["max_this_batch"] > 0 and k["status"] not in ["🛑 STOP"]]
    available_keywords.sort(key=lambda x: -x["remaining"])
    
    if available_keywords[:10]:
        prompt_sections.append("✅ BEZPIECZNE DO UŻYCIA (duży zapas):")
        for kw in available_keywords[:10]:
            prompt_sections.append(f"   ✅ \"{kw['keyword']}\" = {kw['actual']}/{kw['target_max']} → max {kw['max_this_batch']}× w tym batchu")
        prompt_sections.append("")
    
    prompt_sections.append("=" * 60)
    
    # Pokaż pozostałe nieużyte (info)
    basic_remaining = basic_must_use[basic_this_batch_count:]
    extended_remaining = extended_this_batch[extended_this_batch_count:] + extended_scheduled
    
    if basic_remaining or extended_remaining:
        prompt_sections.append(f"📌 POZOSTAŁE NIEUŻYTE (do kolejnych batchy: {len(basic_remaining)} BASIC + {len(extended_remaining)} EXTENDED)")
        prompt_sections.append("")
    
    prompt_sections.append("="*60)
    prompt_sections.append("✅ CHECKLIST PRZED WYSŁANIEM:")
    prompt_sections.append(f"   [ ] Fraza główna użyta {main_keyword_info['use_this_batch'] if main_keyword_info else 1}x")
    prompt_sections.append(f"   [ ] Wszystkie {len(basic_for_this_batch)} fraz BASIC użyte")
    prompt_sections.append(f"   [ ] Wszystkie {len(extended_for_this_batch)} fraz EXTENDED użyte")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    
    if basic_target:
        prompt_sections.append("🟠 OPCJONALNE - DĄŻ DO TARGET (jeśli zmieścisz):")
        for kw in basic_target[:3]:
            prompt_sections.append(f"   • \"{kw['keyword']}\" → {kw['use_this_batch']}x")
        prompt_sections.append("")
    
    # 🆕 v35.9: FILTRUJ ngrams i entities które już są w keywords
    # Żeby GPT nie dostał podwójnej instrukcji używania tej samej frazy
    all_keyword_phrases = set()
    for rid, meta in keywords_state.items():
        kw = meta.get("keyword", "").lower()
        if kw:
            all_keyword_phrases.add(kw)
            # Dodaj też poszczególne słowa dla częściowego matchowania
            for word in kw.split():
                if len(word) > 4:  # Tylko znaczące słowa
                    all_keyword_phrases.add(word)
    
    # Filtruj n-gramy
    filtered_ngrams = []
    skipped_ngrams = []
    if batch_ngrams:
        for ngram in batch_ngrams:
            ngram_lower = ngram.lower()
            # Sprawdź czy n-gram pokrywa się z keyword
            is_keyword = ngram_lower in all_keyword_phrases
            # Sprawdź czy n-gram zawiera keyword lub odwrotnie
            overlaps_keyword = any(kw in ngram_lower or ngram_lower in kw for kw in all_keyword_phrases if len(kw) > 4)
            
            if is_keyword or overlaps_keyword:
                skipped_ngrams.append(ngram)
            else:
                filtered_ngrams.append(ngram)
    
    # Filtruj encje
    filtered_entities = []
    skipped_entities = []
    if top_entities:
        for ent in top_entities:
            ent_text = ent.get("text", "").lower()
            # Sprawdź czy encja pokrywa się z keyword
            is_keyword = ent_text in all_keyword_phrases
            overlaps_keyword = any(kw in ent_text or ent_text in kw for kw in all_keyword_phrases if len(kw) > 4)
            
            if is_keyword or overlaps_keyword:
                skipped_entities.append(ent.get("text", ""))
            else:
                filtered_entities.append(ent)
    
    if filtered_ngrams:
        prompt_sections.append("💡 N-GRAMY (wpleć naturalnie - NIE LICZONE w limitach):")
        for ngram in filtered_ngrams[:4]:
            prompt_sections.append(f"   • \"{ngram}\"")
        prompt_sections.append("")
    
    if skipped_ngrams:
        prompt_sections.append(f"ℹ️ N-gramy pominięte (już w keywords): {', '.join(skipped_ngrams[:5])}")
        prompt_sections.append("")
    
    # v28.0: Entity SEO - encje do wspomnienia
    if filtered_entities:
        prompt_sections.append("🏢 ENCJE DO WSPOMNIENIA (NIE LICZONE w limitach):")
        prompt_sections.append("   Wspomnij te nazwy własne naturalnie w tekście:")
        for ent in filtered_entities[:5]:
            ent_type = ent.get("type", "")
            type_label = {"ORGANIZATION": "firma/inst.", "PERSON": "osoba", "LOCATION": "miejsce"}.get(ent_type, "")
            if type_label:
                prompt_sections.append(f"   • {ent.get('text', '')} ({type_label})")
            else:
                prompt_sections.append(f"   • {ent.get('text', '')}")
        prompt_sections.append("")
    
    if skipped_entities:
        prompt_sections.append(f"ℹ️ Encje pominięte (już w keywords): {', '.join(skipped_entities[:5])}")
        prompt_sections.append("")
    
    if top_relationships:
        prompt_sections.append("🔗 RELACJE DO OPISANIA:")
        for rel in top_relationships[:3]:
            prompt_sections.append(f"   • {rel.get('subject', '')} → {rel.get('verb', '')} → {rel.get('object', '')}")
        prompt_sections.append("")
    
    # v28.0: PAA - pytania użytkowników (odpowiedz na nie w tekście)
    if paa_for_batch:
        prompt_sections.append("❓ PYTANIA UŻYTKOWNIKÓW (odpowiedz w tekście):")
        for paa in paa_for_batch:
            q = paa.get("question", "")
            if q:
                prompt_sections.append(f"   • {q}")
        prompt_sections.append("")
    
    # v28.0: LSI keywords (semantic keyphrases)
    if lsi_keywords:
        prompt_sections.append("🔤 LSI KEYWORDS (wpleć naturalnie):")
        prompt_sections.append(f"   {', '.join(lsi_keywords)}")
        prompt_sections.append("")
    
    # v28.0: Related searches (powiązane tematy - inspiracja)
    if related_searches and current_batch_num <= 2:  # tylko w pierwszych batchach
        prompt_sections.append("🔍 POWIĄZANE TEMATY (opcjonalnie nawiąż):")
        prompt_sections.append(f"   {', '.join(related_searches[:4])}")
        prompt_sections.append("")
    
    # v27.2: Sekcja ZABRONIONYCH fraz - rozdzielona na typy
    # Zbierz wszystkie zabronione: locked + exceeded + extended_done
    forbidden_basic = []
    forbidden_extended = []
    
    for kw in locked_exceeded:
        if kw.get('type', 'BASIC').upper() == 'EXTENDED':
            forbidden_extended.append(kw['keyword'])
        else:
            forbidden_basic.append(kw['keyword'])
    
    # EXTENDED DONE też są zabronione (już użyte 1x)
    forbidden_extended.extend(extended_done)
    
    # Wyświetl zabronione
    if forbidden_basic or forbidden_extended:
        prompt_sections.append("=" * 50)
        prompt_sections.append("🚫 ZABRONIONE FRAZY (NIE UŻYWAJ!):")
        
        if forbidden_basic:
            prompt_sections.append(f"   BASIC (limit osiągnięty): {', '.join(forbidden_basic[:10])}")
            if len(forbidden_basic) > 10:
                prompt_sections.append(f"   ... i {len(forbidden_basic) - 10} więcej BASIC")
        
        if forbidden_extended:
            prompt_sections.append(f"   EXTENDED (już użyte 1x): {', '.join(forbidden_extended[:10])}")
            if len(forbidden_extended) > 10:
                prompt_sections.append(f"   ... i {len(forbidden_extended) - 10} więcej EXTENDED")
        
        prompt_sections.append("=" * 50)
        prompt_sections.append("")
    
    if remaining_h2:
        prompt_sections.append("📋 H2 DO NAPISANIA:")
        for h2 in remaining_h2[:3]:
            prompt_sections.append(f"   • {h2}")
        prompt_sections.append("")
    
    if last_sentences:
        prompt_sections.append(f"🔗 KONTYNUUJ OD: \"{last_sentences[:80]}...\"")
        prompt_sections.append("")
    
    # ================================================================
    # v27.2: DYNAMIC BATCH LENGTH - oblicz minimalną długość na podstawie fraz
    # ================================================================
    # Formuła: 
    # 1. Policz WSZYSTKIE pozostałe użycia fraz (do końca artykułu)
    # 2. Podziel przez remaining_batches = ile użyć na TEN batch
    # 3. min_words = (uses_this_batch * avg_phrase_length) / target_density
    
    # Policz WSZYSTKIE pozostałe użycia (nie tylko ten batch)
    total_remaining_basic = 0
    total_remaining_extended = 0
    avg_phrase_words = 0
    phrase_count = 0
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        keyword = meta.get("keyword", "")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        if not keyword:
            continue
        
        phrase_count += 1
        avg_phrase_words += len(keyword.split())
        
        if kw_type == "EXTENDED":
            # EXTENDED: potrzebuje min 1x
            if actual < 1:
                total_remaining_extended += 1
        else:
            # BASIC: potrzebuje min target_min
            remaining = max(0, target_min - actual)
            total_remaining_basic += remaining
    
    # Średnia długość frazy
    if phrase_count > 0:
        avg_phrase_words = avg_phrase_words / phrase_count
    else:
        avg_phrase_words = 2.0  # domyślnie 2 słowa
    
    # PODZIEL przez remaining_batches = ile na TEN batch
    total_remaining_all = total_remaining_basic + total_remaining_extended
    
    if remaining_batches > 0:
        uses_this_batch = math.ceil(total_remaining_all / remaining_batches)
    else:
        uses_this_batch = total_remaining_all
    
    # v28.1: UŻYJ BATCH_PLAN jeśli dostępny
    suggested_min_words = None
    suggested_max_words = None
    suggested_paragraphs_min = 3  # v28.1: default
    suggested_paragraphs_max = 4  # v28.1: default
    length_profile = "medium"  # v28.1: default
    complexity_score = 50  # v28.1: default
    complexity_reasoning = []  # v28.1: dlaczego taka długość
    snippet_required = True  # v28.1
    batch_plan_used = False
    
    if batch_plan and "batches" in batch_plan:
        batch_plans_list = batch_plan.get("batches", [])
        # Znajdź plan dla current_batch_num
        for bp in batch_plans_list:
            if bp.get("batch_number") == current_batch_num:
                suggested_min_words = bp.get("target_words_min")
                suggested_max_words = bp.get("target_words_max")
                # v28.1: Pobierz wszystkie nowe pola
                suggested_paragraphs_min = bp.get("target_paragraphs_min", 3)
                suggested_paragraphs_max = bp.get("target_paragraphs_max", 4)
                length_profile = bp.get("length_profile", "medium")
                complexity_score = bp.get("complexity_score", 50)
                complexity_reasoning = bp.get("complexity_reasoning", [])
                snippet_required = bp.get("snippet_required", True)
                batch_plan_used = True
                print(f"[PRE_BATCH] batch_plan: batch {current_batch_num}, score={complexity_score}, profile={length_profile}, {suggested_min_words}-{suggested_max_words} words")
                break
    
    # FALLBACK: jeśli brak batch_plan, oblicz dynamicznie
    if not suggested_min_words:
        TARGET_DENSITY_FOR_CALC = 1.5
        
        if uses_this_batch > 0:
            min_words_for_density = int((uses_this_batch * avg_phrase_words) / (TARGET_DENSITY_FOR_CALC / 100))
        else:
            min_words_for_density = 200
        
        # Podstawowa długość zależy od typu batcha - ZMNIEJSZONE WARTOŚCI!
        if batch_type == "INTRO":
            base_min_words = 120
            base_max_words = 180
        elif batch_type == "FINAL":
            base_min_words = 250
            base_max_words = 400
        else:
            base_min_words = 280
            base_max_words = 450
        
        # Weź większą z: bazowej i obliczonej dla density
        suggested_min_words = max(base_min_words, min_words_for_density)
        suggested_max_words = max(base_max_words, suggested_min_words + 100)
        
        # Limit maksymalny
        suggested_min_words = min(suggested_min_words, 600)
        suggested_max_words = min(suggested_max_words, 800)
    
    batch_length_info = {
        "suggested_min": suggested_min_words,
        "suggested_max": suggested_max_words,
        "paragraphs_min": suggested_paragraphs_min,
        "paragraphs_max": suggested_paragraphs_max,
        "length_profile": length_profile,
        "complexity_score": complexity_score,  # v28.1
        "complexity_reasoning": complexity_reasoning,  # v28.1: DLACZEGO taka długość
        "snippet_required": snippet_required,  # v28.1
        "total_remaining": total_remaining_all,
        "uses_this_batch": uses_this_batch,
        "remaining_batches": remaining_batches,
        "from_batch_plan": batch_plan_used,
        "reason": f"Pozostało {total_remaining_all} użyć fraz / {remaining_batches} batchy = ~{uses_this_batch} na ten batch",
        "density_note": f"Przy {suggested_min_words} słowach utrzymasz density w normie"
    }
    
    prompt_sections.append("="*50)
    plan_note = " (z batch_plan)" if batch_plan_used else ""
    prompt_sections.append(f"📏 DŁUGOŚĆ BATCHA{plan_note}: {suggested_min_words}-{suggested_max_words} słów")
    prompt_sections.append(f"📄 AKAPITY: {suggested_paragraphs_min}-{suggested_paragraphs_max}")
    prompt_sections.append(f"🎯 SCORE ZŁOŻONOŚCI: {complexity_score}/100 → profil: {length_profile.upper()}")
    
    # v28.1: Pokaż DLACZEGO taka długość (max 2 powody)
    if complexity_reasoning:
        prompt_sections.append(f"💡 DLACZEGO TAKA DŁUGOŚĆ:")
        for reason in complexity_reasoning[:2]:
            prompt_sections.append(f"   • {reason}")
    
    if snippet_required:
        prompt_sections.append(f"⚡ SNIPPET WYMAGANY: Pierwszych 40-60 słów = bezpośrednia odpowiedź!")
    
    prompt_sections.append(f"   Pozostało {total_remaining_all} użyć fraz / {remaining_batches} batchy = ~{uses_this_batch} na ten batch")
    if uses_this_batch > 15:
        prompt_sections.append(f"   ⚠️ DUŻO FRAZ! Pisz dłuższe sekcje żeby zmieścić wszystkie.")
    prompt_sections.append("")
    
    # v27.4: FINALNE PODSUMOWANIE z konkretną listą
    prompt_sections.append("="*60)
    prompt_sections.append("🎯 FINALNE PODSUMOWANIE - CO MUSISZ ZROBIĆ:")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append(f"W tym batchu MUSISZ użyć DOKŁADNIE tych fraz:")
    prompt_sections.append("")
    
    all_required = []
    if main_keyword_info:
        all_required.append(f"• \"{main_keyword}\" × {main_keyword_info['use_this_batch']}")
    for kw in basic_for_this_batch:
        all_required.append(f"• \"{kw['keyword']}\" × 1")
    for kw in extended_for_this_batch:
        all_required.append(f"• \"{kw['keyword']}\" × 1")
    
    for req in all_required:
        prompt_sections.append(f"   {req}")
    
    prompt_sections.append("")
    prompt_sections.append(f"RAZEM: {len(all_required)} fraz do wplecenia")
    prompt_sections.append("")
    prompt_sections.append("❌ Jeśli pominiesz KTÓRĄKOLWIEK frazę - batch będzie ODRZUCONY!")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    
    prompt_sections.append("="*50)
    prompt_sections.append("✍️ STYL:")
    prompt_sections.append(f"   • Sekcje H2: różna długość (min {suggested_min_words // 2} słów na sekcję)")
    prompt_sections.append("   • Akapity: 40-150 słów")
    prompt_sections.append("   • H3: max 2-3 na artykuł")
    prompt_sections.append("   • Listy wypunktowane: dozwolone w miarę potrzeb")
    prompt_sections.append("   • Format: h2: / h3:")
    prompt_sections.append("="*50)
    
    # ================================================================
    # 🆕 v36.2: ANTI-FRANKENSTEIN CONTEXT
    # Token Budgeting sections, Article Memory, Style Instructions
    # ================================================================
    anti_frankenstein_section = ""
    dynamic_sections_data = None
    # 🆕 v36.9: Dodatkowe pola Anti-Frankenstein
    article_memory_data = None
    style_instructions_data = None
    soft_cap_data = None
    if ANTI_FRANKENSTEIN_ENABLED:
        try:
            anti_frankenstein_section = generate_anti_frankenstein_gpt_section(
                project_data=data,
                current_batch_num=current_batch_num,
                current_h2=remaining_h2[0] if remaining_h2 else ""
            )
            if anti_frankenstein_section:
                prompt_sections.insert(0, anti_frankenstein_section)  # Na początku promptu
                print(f"[PRE_BATCH] 🧟 Anti-Frankenstein context added ({len(anti_frankenstein_section)} chars)")
            
            # Pobierz dynamic sections dla response
            af_context = get_anti_frankenstein_context(
                project_data=data,
                current_batch_num=current_batch_num,
                current_h2=remaining_h2[0] if remaining_h2 else ""
            )
            dynamic_sections_data = af_context.get("dynamic_sections")
            # 🆕 v36.9: Article Memory i Style Instructions
            article_memory_data = af_context.get("article_memory_context")
            style_instructions_data = af_context.get("style_instructions")
            soft_cap_data = af_context.get("soft_cap_keywords")
        except Exception as e:
            print(f"[PRE_BATCH] ⚠️ Anti-Frankenstein context error: {e}")
    
    gpt_prompt = "\n".join(prompt_sections)
    
    # ================================================================
    # 🆕 v39.0: ENHANCED PRE-BATCH INSTRUCTIONS
    # Konkretne instrukcje zamiast surowych danych
    # ================================================================
    enhanced_info = None
    if ENHANCED_PRE_BATCH_ENABLED:
        try:
            # 🆕 v43.0: Pobierz phrase_hierarchy z projektu
            phrase_hierarchy_data = data.get("phrase_hierarchy")
            
            enhanced_info = generate_enhanced_pre_batch_info(
                s1_data=s1_data,
                keywords_state=keywords_state,
                batches=batches,
                h2_structure=h2_structure,
                current_batch_num=current_batch_num,
                total_batches=total_planned_batches,
                main_keyword=main_keyword,
                entity_state=data.get("entity_state", {}),
                style_fingerprint=data.get("style_fingerprint", {}),
                is_ymyl=data.get("is_ymyl", False),
                is_legal=data.get("is_legal", False) or data.get("detected_category") == "prawo",
                batch_plan=batch_plan,  # 🆕 v40.1: Przekaż batch_plan z h2_sections
                phrase_hierarchy_data=phrase_hierarchy_data  # 🆕 v43.0: Phrase Hierarchy
            )
            # 🆕 v40.1: Log ile H2 w batchu
            h2_in_batch = enhanced_info.get("h2_count_in_batch", 0)
            hierarchy_roots = len(enhanced_info.get("phrase_hierarchy", {}).get("roots_covered", [])) if enhanced_info.get("phrase_hierarchy") else 0
            print(f"[PRE_BATCH] 🎯 Enhanced: {len(enhanced_info.get('entities_to_define', []))} entities, {len(enhanced_info.get('relations_to_establish', []))} relations, {h2_in_batch} H2, {hierarchy_roots} roots")
        except Exception as e:
            print(f"[PRE_BATCH] ⚠️ Enhanced pre-batch error: {e}")
            import traceback
            traceback.print_exc()
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "batch_number": current_batch_num,
        "batch_type": batch_type,
        "intro_guidance": intro_guidance,
        "total_planned_batches": total_planned_batches,
        "remaining_batches": remaining_batches,
        
        # v27.2: Dynamic batch length
        "batch_length": batch_length_info,
        
        "coverage": {
            "basic": coverage.get("basic", {}),
            "extended": coverage.get("extended", {}),
            "overall": coverage.get("overall_coverage", 100)
        },
        
        "density": {
            "current": current_density,
            "max_density": density_details.get("max_density", 0),
            "avg_density": density_details.get("avg_density", 0),
            "status": density_status,
            "message": density_msg,
            "optimal_range": f"{DENSITY_OPTIMAL_MIN}-{DENSITY_OPTIMAL_MAX}%",
            "warnings": density_details.get("warnings", []),
            "per_keyword_top5": dict(list(density_details.get("per_keyword", {}).items())[:5])
        },
        
        "main_keyword": {
            "keyword": main_keyword,
            "info": main_keyword_info,
            "ratio": round(main_ratio, 2),
            "ratio_warning": ratio_warning
        },
        
        "keywords": {
            "basic_must_use": basic_must_use,
            "basic_target": basic_target,
            "basic_done": [kw["keyword"] for kw in basic_done],
            "extended_this_batch": extended_this_batch,
            "extended_done": extended_done,
            "extended_scheduled": extended_scheduled,
            "locked_exceeded": locked_exceeded
        },
        
        "ngrams_for_batch": batch_ngrams,
        
        # v28.0: Entity SEO
        "entity_seo": {
            "top_entities": top_entities,
            "relationships": top_relationships,
            "must_topics": must_topics,
            "total_entities": len(entities),
            "enabled": bool(entity_seo)
        },
        
        # v29.3: Entity guidance for batch
        "entities_for_batch": {
            "to_introduce": get_entities_to_introduce(
                top_entities, 
                current_batch_num, 
                total_planned_batches,
                batch_texts  # 🆕 v41.1 FIX: Use pre-computed safe batch_texts
            ),
            "already_defined": get_already_defined_entities(
                batch_texts  # 🆕 v41.1 FIX: Use pre-computed safe batch_texts
            ),
            "suggested_relationships": top_relationships[:2] if current_batch_num > 1 else []
        },
        
        # v29.3: N-gram diversity guidance
        "ngram_guidance": {
            "overused_phrases": get_overused_phrases(
                batch_texts,  # 🆕 v41.1 FIX: Use pre-computed safe batch_texts
                main_keyword
            ),
            "suggested_synonyms": get_synonyms_for_overused(
                batch_texts,  # 🆕 v41.1 FIX: Use pre-computed safe batch_texts
                main_keyword
            ),
            "lsi_to_include": lsi_keywords[:3] if lsi_keywords else batch_ngrams[:3]
        },
        
        # v29.3: Section length variety guidance
        "section_length_guidance": get_section_length_guidance(
            current_batch_num,
            total_planned_batches,
            batch_type
        ),
        
        # v28.0: Dodatkowe dane SERP
        "serp_enrichment": {
            "paa_for_batch": paa_for_batch,
            "lsi_keywords": lsi_keywords,
            "related_searches": related_searches
        },
        
        "h2_remaining": remaining_h2,
        "h2_used": used_h2,
        
        # v29.2: H2 Plan z generatora
        "h2_plan": data.get("h2_plan", []),
        "h2_plan_meta": data.get("h2_plan_meta", {}),
        
        # 🆕 v35.9: Wyraźne limity fraz na batch
        "keyword_limits": {
            "stop_keywords": [k for k in all_keywords_limits if k["status"] == "🛑 STOP"][:15],
            "caution_keywords": [k for k in all_keywords_limits if k["status"] == "⚠️ OSTROŻNIE"][:10],
            "safe_keywords": [k for k in all_keywords_limits if k["max_this_batch"] > 0 and k["status"] not in ["🛑 STOP"]][:10],
            "summary": {
                "total_stop": len([k for k in all_keywords_limits if k["status"] == "🛑 STOP"]),
                "total_caution": len([k for k in all_keywords_limits if k["status"] == "⚠️ OSTROŻNIE"]),
                "message": f"🛑 {len([k for k in all_keywords_limits if k['status'] == '🛑 STOP'])} fraz z wyczerpanym limitem - NIE UŻYWAJ!"
            }
        },
        
        # 🆕 v36.0: Semantic batch plan
        "semantic_batch_plan": semantic_batch_plan if semantic_batch_plan else None,
        
        # 🆕 v36.2: Anti-Frankenstein dynamic sections (Token Budgeting)
        "dynamic_sections": dynamic_sections_data,
        
        # 🆕 v36.9: Article Memory - kontekst z poprzednich batchów
        "article_memory": article_memory_data,
        
        # 🆕 v36.9: Style Instructions - utrzymanie spójnego tonu
        "style_instructions": style_instructions_data,
        
        # 🆕 v36.9: Soft Cap Recommendations - elastyczne limity
        "soft_cap_recommendations": soft_cap_data,
        
        # 🆕 v36.5: Legal context - przepisy prawne i instrukcje cytowania
        "legal_context": {
            "active": data.get("detected_category") == "prawo",
            "detected_articles": data.get("detected_articles", []),
            "legal_instruction": data.get("legal_instruction", ""),
            "top_judgments": [
                {
                    "signature": j.get("signature", j.get("caseNumber", "")),
                    "court": j.get("court", j.get("courtName", "")),
                    "date": j.get("date", j.get("judgmentDate", "")),
                    "relevance_score": j.get("relevance_score", 0)
                }
                for j in data.get("legal_judgments", [])[:3]
            ],
            "must_cite": len(data.get("detected_articles", [])) > 0,
            "citation_hint": f"Odwołaj się do: {', '.join(data.get('detected_articles', []))}" 
                if data.get("detected_articles") else None
        } if data.get("detected_category") == "prawo" else None,
        
        "gpt_prompt": gpt_prompt,
        
        # 🆕 v39.0: Enhanced Pre-Batch Instructions
        "enhanced": enhanced_info,
        
        # 🆕 v39.0: Konkretne instrukcje (wyodrębnione dla łatwości użycia)
        "entities_to_define": enhanced_info.get("entities_to_define", []) if enhanced_info else [],
        "relations_to_establish": enhanced_info.get("relations_to_establish", []) if enhanced_info else [],
        "semantic_context": enhanced_info.get("semantic_context", {}) if enhanced_info else {},
        "style_instructions_v39": enhanced_info.get("style_instructions", {}) if enhanced_info else {},
        "continuation_v39": enhanced_info.get("continuation", {}) if enhanced_info else {},
        "keyword_tracking": enhanced_info.get("keyword_tracking", {}) if enhanced_info else {},
        "gpt_instructions_v39": enhanced_info.get("gpt_instructions", "") if enhanced_info else "",
        
        "version": "v39.0"
    }), 200


# ================================================================
#  ADD BATCH
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    return jsonify(result), 200


# ================================================================
#  APPROVE BATCH - v28.2 z CLAUDE REVIEWER
# ================================================================
@project_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch_with_review(project_id):
    """
    v28.3: Approve batch z automatycznym review przez Claude.
    
    NOWOŚĆ v28.3:
    - Auto-approve po 2 próbach (attempt >= 2)
    - Lemmatyzacja fraz (ścieżka sensoryczna = ścieżką sensoryczną)
    - Wykrywanie tautologii przez Claude
    
    Flow:
    1. Quick checks (Python) - frazy, długość
    2. Claude review - pełna analiza semantyczna
    3. Jeśli CORRECTED → zwróć poprawiony tekst
    4. Jeśli APPROVED → zapisz do Firestore
    5. Jeśli REJECTED → zwróć do przepisania
    
    Request:
    {
        "text": "h2: Tytuł...",
        "skip_review": false,  // opcjonalne - pomiń Claude
        "force_save": false,   // opcjonalne - zapisz mimo warnings
        "attempt": 1           // NOWE! numer próby (1, 2, 3...)
    }
    
    Po attempt >= 2: automatyczne force_save=True (auto-approve)
    """
    from dataclasses import asdict
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    
    batch_text = data.get("corrected_text") or data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "No text provided"}), 400
    
    skip_review = data.get("skip_review", False)
    force_save = data.get("force_save", False)
    attempt = data.get("attempt", 1)  # v28.3: numer próby
    
    # v30.1 OPTIMIZED: AUTO-APPROVE po 2 próbach (było 3)
    if attempt >= 2 and not force_save:
        print(f"[APPROVE_BATCH] ⚡ Auto-approve: attempt={attempt} >= 3, force_save=True")
        force_save = True
    
    db = firestore.client()
    project_ref = db.collection("seo_projects").document(project_id)
    project_doc = project_ref.get()
    
    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = project_doc.to_dict()
    
    # ============================================
    # ZBUDUJ CONTEXT DLA REVIEWERA
    # ============================================
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    batch_plan = project_data.get("batch_plan", {})
    current_batch = project_data.get("current_batch_num", 1)
    
    # Znajdź wymagane frazy (nieużyte)
    keywords_required = []
    keywords_forbidden = []
    
    # Main keyword
    main_kw_count = 2
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            actual = meta.get("actual_uses", 0)
            target = meta.get("target_max", 10)
            remaining = max(0, target - actual)
            main_kw_count = min(3, max(1, remaining // max(1, project_data.get("total_planned_batches", 4) - current_batch + 1)))
            break
    
    keywords_required.append({"keyword": main_keyword, "count": main_kw_count})
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword or keyword.lower() == main_keyword.lower():
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        if kw_type == "EXTENDED":
            if actual >= 1:
                keywords_forbidden.append(keyword)
            else:
                keywords_required.append({"keyword": keyword, "count": 1})
        else:
            if actual >= target_max:
                keywords_forbidden.append(keyword)
            elif actual < target_min:
                keywords_required.append({"keyword": keyword, "count": 1})
    
    # Limit fraz na batch
    keywords_required = keywords_required[:12]
    
    # Długość z batch_plan
    target_words_min = 200
    target_words_max = 500
    target_para_min = 2
    target_para_max = 5
    snippet_required = True
    complexity_score = 50
    
    if batch_plan and "batches" in batch_plan:
        for bp in batch_plan.get("batches", []):
            if bp.get("batch_number") == current_batch:
                target_words_min = bp.get("target_words_min", 200)
                target_words_max = bp.get("target_words_max", 500)
                target_para_min = bp.get("target_paragraphs_min", 2)
                target_para_max = bp.get("target_paragraphs_max", 5)
                snippet_required = bp.get("snippet_required", True)
                complexity_score = bp.get("complexity_score", 50)
                break
    
    # Last sentences
    article_content = project_data.get("article_content", "")
    last_sentences = article_content[-200:] if len(article_content) > 200 else article_content
    
    review_context = {
        "topic": project_data.get("topic", ""),
        "h2_current": project_data.get("h2_remaining", [])[:2],
        "keywords_required": keywords_required,
        "keywords_forbidden": keywords_forbidden,
        "last_sentences": last_sentences,
        "target_words_min": target_words_min,
        "target_words_max": target_words_max,
        "target_paragraphs_min": target_para_min,
        "target_paragraphs_max": target_para_max,
        "main_keyword": main_keyword,
        "main_keyword_count": main_kw_count,
        "batch_number": current_batch,
        "snippet_required": snippet_required,
        "complexity_score": complexity_score
    }
    
    # 🆕 v36.9: Domyślne wartości
    auto_fixes = []
    has_optimization_helpers = False
    
    # ============================================
    # CLAUDE REVIEW
    # ============================================
    try:
        from claude_reviewer import review_batch, ReviewResult
        # 🆕 v36.9: Import optimization_helpers dla actionable_feedback
        try:
            from optimization_helpers import get_actionable_feedback
            has_optimization_helpers = True
        except ImportError:
            has_optimization_helpers = False
            print("[APPROVE_BATCH] ⚠️ optimization_helpers not available")
        
        result = review_batch(batch_text, review_context, skip_claude=skip_review)
        
        # 🆕 v36.9: Pobierz auto_fixes_applied jeśli dostępne
        auto_fixes = getattr(result, 'auto_fixes_applied', []) if result else []
        
        # QUICK_CHECK_FAILED - zwróć do poprawy
        if result.status == "QUICK_CHECK_FAILED":
            issues_list = [asdict(i) for i in result.issues]
            
            # 🆕 v36.9: Prioritized issues i actionable feedback
            prioritized_issues = []
            actionable_feedback = {}
            if has_optimization_helpers:
                try:
                    feedback = get_actionable_feedback(issues_list, attempt)
                    prioritized_issues = feedback.get("prioritized_issues", issues_list[:2])
                    actionable_feedback = {
                        "attempt": attempt,
                        "total_issues": len(issues_list),
                        "prioritized_count": len(prioritized_issues),
                        "focus_message": f"🎯 SKUP SIĘ NA: {prioritized_issues[0].get('type', 'UNKNOWN') if prioritized_issues else 'brak'}",
                        "instructions": feedback.get("instructions", [])
                    }
                except Exception as e:
                    print(f"[APPROVE_BATCH] ⚠️ actionable_feedback error: {e}")
                    prioritized_issues = issues_list[:2]
            else:
                prioritized_issues = issues_list[:2]
            
            return jsonify({
                "status": "NEEDS_CORRECTION",  # 🆕 v36.9: Zmiana z QUICK_CHECK_FAILED
                "needs_correction": True,
                "prioritized_issues": prioritized_issues,  # 🆕 v36.9
                "actionable_feedback": actionable_feedback,  # 🆕 v36.9
                "auto_fixes_applied": auto_fixes,  # 🆕 v36.9
                "issues": issues_list,  # zachowane dla kompatybilności
                "correction_prompt": build_correction_prompt(issues_list, batch_text),
                "message": result.summary,
                "word_count": result.word_count,
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "auto_approve_at": 2
            }), 200
        
        # REJECTED - wymaga przepisania
        if result.status == "REJECTED":
            issues_list = [asdict(i) for i in result.issues]
            
            # 🆕 v36.9: Prioritized issues i actionable feedback
            prioritized_issues = []
            actionable_feedback = {}
            if has_optimization_helpers:
                try:
                    feedback = get_actionable_feedback(issues_list, attempt)
                    prioritized_issues = feedback.get("prioritized_issues", issues_list[:2])
                    actionable_feedback = {
                        "attempt": attempt,
                        "total_issues": len(issues_list),
                        "prioritized_count": len(prioritized_issues),
                        "focus_message": f"🎯 SKUP SIĘ NA: {prioritized_issues[0].get('type', 'UNKNOWN') if prioritized_issues else 'przepisanie'}",
                        "instructions": feedback.get("instructions", [])
                    }
                except Exception as e:
                    print(f"[APPROVE_BATCH] ⚠️ actionable_feedback error: {e}")
                    prioritized_issues = issues_list[:2]
            else:
                prioritized_issues = issues_list[:2]
            
            return jsonify({
                "status": "NEEDS_CORRECTION",  # 🆕 v36.9: Zmiana z REJECTED
                "needs_correction": True,
                "prioritized_issues": prioritized_issues,  # 🆕 v36.9
                "actionable_feedback": actionable_feedback,  # 🆕 v36.9
                "auto_fixes_applied": auto_fixes,  # 🆕 v36.9
                "issues": issues_list,
                "correction_prompt": f"Tekst wymaga przepisania. {result.summary}",
                "message": result.summary,
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "auto_approve_at": 2
            }), 200
        
        # CORRECTED - Claude poprawił
        if result.status == "CORRECTED" and result.corrected_text:
            # Użyj poprawionego tekstu
            batch_text = result.corrected_text
            issues_list = [asdict(i) for i in result.issues]
            
            if not force_save:
                # Zwróć do akceptacji przez GPT
                return jsonify({
                    "status": "CORRECTED",
                    "needs_correction": False,
                    "corrected_text": batch_text,
                    "original_text": result.original_text,
                    "issues": issues_list,
                    "message": f"Claude poprawił tekst: {result.summary}",
                    "word_count": result.word_count,
                    "instruction": "Użyj corrected_text i wyślij ponownie z force_save=true",
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "auto_approve_at": 2
                }), 200
        
        # APPROVED lub force_save - zapisz
        print(f"[APPROVE_BATCH] ✅ Review passed: {result.status}, saving batch")
        
    except ImportError:
        print(f"[APPROVE_BATCH] ⚠️ claude_reviewer not available, saving without review")
    except ImportError:
        print(f"[APPROVE_BATCH] ⚠️ claude_reviewer not available, saving without review")
    except Exception as e:
        print(f"[APPROVE_BATCH] ⚠️ Review error: {e}, saving anyway")
    
    # ============================================
    # ZAPISZ DO FIRESTORE
    # ============================================
    meta_trace = data.get("meta_trace", {})
    save_result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    return jsonify({
        "status": "APPROVED",
        "needs_correction": False,
        "saved": True,
        "batch_number": save_result.get("batch_number"),
        "word_count": len(batch_text.split()),
        "message": "Batch zatwierdzony i zapisany",
        "auto_fixes_applied": auto_fixes,  # 🆕 v36.9
        **save_result
    }), 200


def build_correction_prompt(issues: list, original_text: str) -> str:
    """Buduje prompt do poprawienia tekstu."""
    lines = ["Popraw poniższy tekst:"]
    
    for issue in issues:
        severity = issue.get("severity", "warning")
        desc = issue.get("description", "")
        if severity == "critical":
            lines.append(f"❌ KRYTYCZNE: {desc}")
        else:
            lines.append(f"⚠️ {desc}")
    
    lines.append("\n--- TEKST DO POPRAWY ---")
    lines.append(original_text[:1000])
    if len(original_text) > 1000:
        lines.append("...")
    lines.append("\n--- WYŚLIJ POPRAWIONY TEKST ---")
    
    return "\n".join(lines)


# ================================================================
#  PREVIEW BATCH
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """Preview batch z walidacją."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    s1_data = project_data.get("s1_data", {})

    report = unified_prevalidation(batch_text, keywords_state)
    
    warnings = report.get("warnings", [])
    errors = []
    
    # v25.0: Density check
    density = report.get("density", 0)
    density_status, density_msg = get_density_status(density)
    if density_status in ["HIGH", "STUFFING"]:
        warnings.append({
            "type": "DENSITY_HIGH",
            "density": density,
            "status": density_status,
            "message": density_msg
        })
    
    # v28.1: Usunięto limit 1 listy - teraz max 3 (informacyjnie)
    list_count = count_bullet_lists(batch_text)
    # Tylko informacja, nie blokuje
    # if list_count > 3:
    #     warnings.append({
    #         "type": "TOO_MANY_LISTS",
    #         "count": list_count,
    #         "max": 3,
    #         "message": f"Dużo list ({list_count}). Rozważ ograniczenie."
    #     })
    
    h3_validation = validate_h3_length(batch_text, min_words=80)
    if h3_validation["issues"]:
        for issue in h3_validation["issues"]:
            warnings.append({
                "type": "H3_TOO_SHORT",
                "h3": issue["h3"],
                "word_count": issue["word_count"],
                "min": 80,
                "message": f"H3 '{issue['h3']}' za krótkie ({issue['word_count']} słów, min 80)"
            })
    
    main_synonym_check = check_main_vs_synonyms_in_text(batch_text, main_keyword, keywords_state)
    if not main_synonym_check["valid"]:
        warnings.append({
            "type": "SYNONYM_OVERUSE",
            "main_count": main_synonym_check["main_count"],
            "synonym_total": main_synonym_check["synonym_total"],
            "ratio": main_synonym_check["main_ratio"],
            "message": main_synonym_check["warning"]
        })
    
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.5][:10]
    ngram_check = check_ngram_coverage_in_text(batch_text, top_ngrams)
    if ngram_check["coverage"] < 0.5:
        warnings.append({
            "type": "LOW_NGRAM_COVERAGE",
            "coverage": ngram_check["coverage"],
            "missing": ngram_check["missing"][:3],
            "message": f"Niskie pokrycie n-gramów ({ngram_check['coverage']:.0%})"
        })
    
    # v27.4: Polish Language Quality Check
    polish_quality = {"status": "DISABLED", "issues": []}
    if POLISH_QUALITY_ENABLED:
        try:
            polish_quality = quick_polish_check(batch_text)
            
            # Dodaj szczegóły kolokacji i banned phrases
            collocations, _ = check_collocations(batch_text)
            banned, _ = check_banned_phrases(batch_text)
            
            polish_quality["collocations"] = collocations[:5]  # Max 5
            polish_quality["banned_phrases"] = banned[:5]  # Max 5
            
            # Dodaj warnings jeśli są problemy
            for coll in collocations[:3]:
                warnings.append({
                    "type": "COLLOCATION_ERROR",
                    "found": coll.get("found", ""),
                    "suggested": coll.get("suggested", ""),
                    "message": f"Błędna kolokacja: '{coll.get('found')}' → '{coll.get('suggested')}'"
                })
            
            for bp in banned[:3]:
                warnings.append({
                    "type": "BANNED_PHRASE",
                    "phrase": bp.get("phrase", ""),
                    "category": bp.get("category", ""),
                    "message": f"Fraza AI: '{bp.get('phrase')}' - usuń lub przeformułuj"
                })
                
        except Exception as e:
            print(f"[PREVIEW_BATCH] ⚠️ Polish quality check error: {e}")
            polish_quality = {"status": "ERROR", "error": str(e)}
    
    status = "OK"
    if errors:
        status = "ERROR"
    elif len(warnings) > 2:
        status = "WARN"
    
    # v28.1: GRAMMAR VALIDATION - sprawdź przed zapisem!
    grammar_validation = {"is_valid": True, "error_count": 0, "correction_needed": False}
    try:
        from grammar_middleware import validate_batch_full
        grammar_validation = validate_batch_full(batch_text)
        
        if not grammar_validation["is_valid"]:
            status = "NEEDS_CORRECTION"
            print(f"[PREVIEW_BATCH] ⚠️ Grammar issues: {grammar_validation['grammar']['error_count']} errors, banned: {grammar_validation['banned_phrases']['found']}")
            
            # Dodaj do warnings
            if grammar_validation["grammar"]["error_count"] > 0:
                warnings.append({
                    "type": "GRAMMAR_ERRORS",
                    "count": grammar_validation["grammar"]["error_count"],
                    "message": f"Wykryto {grammar_validation['grammar']['error_count']} błędów gramatycznych - popraw przed zapisem!"
                })
            
            for phrase in grammar_validation["banned_phrases"]["found"]:
                warnings.append({
                    "type": "BANNED_PHRASE_DETECTED",
                    "phrase": phrase,
                    "message": f"Usuń frazę: '{phrase}'"
                })
    except ImportError:
        print(f"[PREVIEW_BATCH] ⚠️ grammar_middleware not available")
    except Exception as e:
        print(f"[PREVIEW_BATCH] ⚠️ Grammar check error: {e}")
    
    # v27.0: Zapisz tekst do last_preview (fallback dla approve_batch)
    try:
        db.collection("seo_projects").document(project_id).update({
            "last_preview": {
                "text": batch_text,
                "status": status,
                "grammar_valid": grammar_validation.get("is_valid", True),
                "timestamp": firestore.SERVER_TIMESTAMP
            }
        })
        print(f"[PREVIEW_BATCH] ✅ Zapisano last_preview ({len(batch_text)} znaków)")
    except Exception as e:
        print(f"[PREVIEW_BATCH] ⚠️ Nie udało się zapisać last_preview: {e}")
    
    # v28.1: Jeśli błędy gramatyczne - zwróć prompt do poprawy
    response_data = {
        "status": status,
        "semantic_score": report.get("semantic_score", 0),
        "density": density,
        "density_status": density_status,
        "warnings": warnings,
        "errors": errors,
        "validations": {
            "lists": {"count": list_count, "valid": True},  # v28.1: brak limitu list
            "h3_length": h3_validation,
            "main_vs_synonyms": main_synonym_check,
            "ngram_coverage": ngram_check,
            "density": {"value": density, "status": density_status, "message": density_msg},
            "polish_quality": polish_quality,
            "grammar": grammar_validation  # v28.1
        },
        "last_preview_saved": True,
        "version": "v28.1"
    }
    
    # v28.1: Jeśli wymaga korekty - dodaj prompt
    if grammar_validation.get("correction_needed"):
        response_data["needs_correction"] = True
        response_data["correction_prompt"] = grammar_validation.get("correction_prompt", "")
        response_data["instruction"] = "POPRAW błędy i wyślij ponownie do preview_batch"
    
    return jsonify(response_data), 200


# ================================================================
# v26.1: BEST-OF-N BATCH GENERATION
# ================================================================
@project_routes.post("/api/project/<project_id>/generate_best_batch")
def generate_best_batch(project_id):
    """
    v26.1: Generuje N wersji batcha i zwraca najlepszą.
    
    Request body:
    {
        "prompt": "Treść promptu do generowania batcha",
        "n_candidates": 3,  // opcjonalne, default 3
        "min_score": 60     // opcjonalne, minimalny akceptowalny score
    }
    
    Response:
    {
        "status": "OK" | "WARN",
        "selected_content": "...",
        "selected_score": 85.2,
        "selected_variant": 2,
        "all_candidates": [...],
        "meets_minimum": true,
        "selection_reason": "..."
    }
    """
    if not BEST_OF_N_ENABLED:
        return jsonify({
            "error": "Best-of-N module not available",
            "fallback": "Use standard preview_batch endpoint"
        }), 501
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Field 'prompt' required"}), 400
    
    n_candidates = data.get("n_candidates", 3)
    min_score = data.get("min_score", 60)
    
    # Pobierz dane projektu
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    
    # Wywołaj Best-of-N selection
    try:
        result = select_best_batch(
            base_prompt=prompt,
            keywords_state=keywords_state,
            main_keyword=main_keyword,
            n_candidates=n_candidates
        )
    except Exception as e:
        return jsonify({
            "error": f"Generation failed: {str(e)}",
            "status": "ERROR"
        }), 500
    
    if result.get("error"):
        return jsonify({
            "error": result.get("error"),
            "status": "ERROR"
        }), 500
    
    # Określ status
    meets_minimum = result.get("meets_minimum", False)
    status = "OK" if meets_minimum else "WARN"
    
    return jsonify({
        "status": status,
        "selected_content": result.get("selected_content"),
        "selected_score": result.get("selected_score"),
        "selected_variant": result.get("selected_variant"),
        "all_candidates": result.get("all_candidates", []),
        "meets_minimum": meets_minimum,
        "selection_reason": result.get("selection_reason"),
        "component_scores": result.get("component_scores", {}),
        "issues": result.get("issues", []),
        "warnings": result.get("warnings", []),
        "version": "v26.1"
    }), 200


@project_routes.post("/api/project/<project_id>/preview_batch_v2")
def preview_batch_v2(project_id):
    """
    v26.1: Preview batch z opcjonalnym Best-of-N.
    
    Jeśli use_best_of_n=true i podano prompt, generuje 3 wersje.
    Jeśli podano text, waliduje jak dotychczas.
    
    Request body:
    {
        "text": "...",           // opcjonalne - do walidacji istniejącego tekstu
        "prompt": "...",         // opcjonalne - do generowania Best-of-N
        "use_best_of_n": true,   // opcjonalne, default false
        "n_candidates": 3        // opcjonalne
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    use_best_of_n = data.get("use_best_of_n", True)  # 🔧 v30.1: Domyślnie WŁĄCZONE
    prompt = data.get("prompt")
    batch_text = data.get("text") or data.get("batch_text")
    
    # Jeśli Best-of-N i mamy prompt
    if use_best_of_n and prompt and BEST_OF_N_ENABLED:
        # Przekieruj do generate_best_batch
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
        
        n_candidates = data.get("n_candidates", 3)
        
        try:
            result = select_best_batch(
                base_prompt=prompt,
                keywords_state=keywords_state,
                main_keyword=main_keyword,
                n_candidates=n_candidates
            )
            
            # Zwróć w formacie kompatybilnym z preview_batch
            return jsonify({
                "status": "OK" if result.get("meets_minimum") else "WARN",
                "method": "best_of_n",
                "selected_content": result.get("selected_content"),
                "selected_score": result.get("selected_score"),
                "all_candidates": result.get("all_candidates", []),
                "selection_reason": result.get("selection_reason"),
                "warnings": [{"type": "INFO", "message": result.get("selection_reason")}],
                "errors": [],
                "version": "v26.1"
            }), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Best-of-N failed: {str(e)}",
                "fallback": "Provide 'text' for standard validation"
            }), 500
    
    # Standardowa walidacja (jak w preview_batch)
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'prompt' with use_best_of_n required"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    s1_data = project_data.get("s1_data", {})

    report = unified_prevalidation(batch_text, keywords_state)
    
    warnings = report.get("warnings", [])
    errors = []
    
    density = report.get("density", 0)
    density_status, density_msg = get_density_status(density)
    if density_status in ["HIGH", "STUFFING"]:
        warnings.append({
            "type": "DENSITY_HIGH",
            "density": density,
            "status": density_status,
            "message": density_msg
        })
    
    # v28.1: Usunięto limit list
    list_count = count_bullet_lists(batch_text)
    
    status = "OK"
    if errors:
        status = "ERROR"
    elif len(warnings) > 2:
        status = "WARN"
    
    return jsonify({
        "status": status,
        "method": "standard",
        "density": density,
        "density_status": density_status,
        "warnings": warnings,
        "errors": errors,
        "version": "v26.1"
    }), 200


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def count_bullet_lists(text: str) -> int:
    """Liczy bloki list wypunktowanych."""
    lines = text.split('\n')
    list_blocks = 0
    in_list = False
    
    for line in lines:
        is_bullet = bool(re.match(r'^\s*[-•*]\s+|^\s*\d+\.\s+', line.strip()))
        
        if is_bullet and not in_list:
            list_blocks += 1
            in_list = True
        elif not is_bullet and line.strip():
            in_list = False
    
    html_lists = len(re.findall(r'<ul>|<ol>', text, re.IGNORECASE))
    
    return list_blocks + html_lists


def validate_h3_length(text: str, min_words: int = 80) -> dict:
    """Sprawdza czy sekcje H3 mają minimalną długość."""
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    issues = []
    sections = []
    
    for i, match in enumerate(h3_matches):
        h3_title = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = len(text)
        
        next_header = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
        if next_header:
            end = start + next_header.start()
        
        section_text = text[start:end].strip()
        section_text = re.sub(r'<[^>]+>', '', section_text)
        word_count = len(section_text.split())
        
        sections.append({"h3": h3_title, "word_count": word_count})
        
        if word_count < min_words:
            issues.append({
                "h3": h3_title,
                "word_count": word_count,
                "min_required": min_words,
                "deficit": min_words - word_count
            })
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "sections": sections,
        "total_h3": len(h3_matches)
    }


def check_main_vs_synonyms_in_text(text: str, main_keyword: str, keywords_state: dict) -> dict:
    """Sprawdza proporcję frazy głównej vs synonimy w tekście."""
    text_lower = text.lower()
    
    main_count = len(re.findall(rf"\b{re.escape(main_keyword.lower())}\b", text_lower))
    
    synonym_counts = {}
    synonym_total = 0
    
    for rid, meta in keywords_state.items():
        if meta.get("is_synonym_of_main"):
            keyword = meta.get("keyword", "").lower()
            count = len(re.findall(rf"\b{re.escape(keyword)}\b", text_lower))
            if count > 0:
                synonym_counts[meta.get("keyword")] = count
                synonym_total += count
    
    total = main_count + synonym_total
    main_ratio = main_count / total if total > 0 else 1.0
    
    return {
        "main_keyword": main_keyword,
        "main_count": main_count,
        "synonyms": synonym_counts,
        "synonym_total": synonym_total,
        "total": total,
        "main_ratio": round(main_ratio, 2),
        "valid": main_ratio >= 0.3,
        "warning": f"Za dużo synonimów! '{main_keyword}' ma tylko {main_ratio:.0%}. Zamień synonimy." if main_ratio < 0.3 else None
    }


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    v33.3: Oblicza podobieństwo między dwoma tekstami (0-1).
    Używa Jaccard similarity na słowach.
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Usuń stop words polskie
    stop_words = {'i', 'w', 'na', 'do', 'z', 'się', 'nie', 'to', 'że', 'o', 'jak', 'ale', 'po', 'co', 'tak', 'za', 'od', 'czy', 'tylko', 'są', 'jest', 'dla', 'oraz', 'przez', 'przy', 'już', 'być', 'ma', 'te', 'ten', 'ta', 'tym'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def get_ngrams_for_h2(
    h2_title: str,
    all_ngrams: List[dict],
    used_ngrams: List[str],
    max_ngrams: int = 4
) -> List[str]:
    """
    v33.3: Dopasowuje n-gramy do tematu H2 używając similarity.
    
    Zamiast przydzielać n-gramy sekwencyjnie, wybiera te które 
    są najbardziej semantycznie związane z nagłówkiem H2.
    
    Args:
        h2_title: Tytuł nagłówka H2 dla tego batcha
        all_ngrams: Lista wszystkich n-gramów z S1 (z weight)
        used_ngrams: Lista już użytych n-gramów w poprzednich batchach
        max_ngrams: Max liczba n-gramów do zwrócenia
    
    Returns:
        Lista n-gramów dopasowanych do H2
    """
    if not h2_title or not all_ngrams:
        return []
    
    # Filtruj już użyte
    available = [n for n in all_ngrams if n.get("ngram", "") not in used_ngrams]
    
    if not available:
        return []
    
    # Oblicz score dla każdego n-grama: similarity + weight
    scored = []
    for ngram_obj in available:
        ngram = ngram_obj.get("ngram", "")
        weight = ngram_obj.get("weight", 0.5)
        
        # Similarity do H2
        similarity = calculate_text_similarity(h2_title, ngram)
        
        # Bonus jeśli n-gram zawiera słowo z H2
        h2_words = set(h2_title.lower().split())
        ngram_words = set(ngram.lower().split())
        word_overlap_bonus = 0.3 if h2_words & ngram_words else 0
        
        # Final score
        score = similarity * 0.4 + weight * 0.4 + word_overlap_bonus * 0.2
        scored.append((ngram, score))
    
    # Sortuj po score malejąco
    scored.sort(key=lambda x: -x[1])
    
    # Zwróć top n-gramy
    return [s[0] for s in scored[:max_ngrams]]


def get_used_ngrams_from_batches(batches: List[dict], all_ngrams: List[str]) -> List[str]:
    """
    v33.3: Zbiera n-gramy które już zostały użyte w poprzednich batchach.
    """
    used = []
    all_text = " ".join([b.get("text", "") for b in batches]).lower()
    
    for ngram in all_ngrams:
        if ngram.lower() in all_text:
            used.append(ngram)
    
    return used


def check_ngram_coverage_in_text(text: str, required_ngrams: list) -> dict:
    """Sprawdza pokrycie n-gramów w tekście."""
    text_lower = text.lower()
    used = []
    missing = []
    
    for ngram in required_ngrams:
        if ngram and ngram.lower() in text_lower:
            used.append(ngram)
        elif ngram:
            missing.append(ngram)
    
    coverage = len(used) / len(required_ngrams) if required_ngrams else 1.0
    
    return {
        "coverage": round(coverage, 2),
        "used": used,
        "missing": missing,
        "valid": coverage >= 0.6
    }


# ================================================================
#  AUTO-CORRECT ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """Automatyczna korekta batcha."""
    data = request.get_json() or {}
    batch_text = data.get("text") or data.get("batch_text")
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    
    if not batch_text:
        batches = project_data.get("batches", [])
        if batches:
            for batch in reversed(batches):
                if batch.get("text"):
                    batch_text = batch.get("text")
                    break
    
    if not batch_text:
        batches = project_data.get("batches", [])
        all_texts = [b.get("text", "") for b in batches if b.get("text")]
        if all_texts:
            batch_text = "\n\n".join(all_texts)
    
    if not batch_text:
        return jsonify({
            "error": "No text provided",
            "hint": "Brak zapisanych batchy w projekcie lub wszystkie są puste",
            "batches_count": len(project_data.get("batches", []))
        }), 400
    
    keywords_state = project_data.get("keywords_state", {})
    
    under_keywords = []
    over_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        min_target = meta.get("target_min", 0)
        max_target = meta.get("target_max", 999)
        keyword = meta.get("keyword", "")
        
        if actual < min_target:
            under_keywords.append({
                "keyword": keyword,
                "missing": min_target - actual,
                "current": actual,
                "target_min": min_target
            })
        elif actual > max_target:
            over_keywords.append({
                "keyword": keyword,
                "excess": actual - max_target,
                "current": actual,
                "target_max": max_target
            })
    
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "corrected_text": batch_text
        }), 200
    
    if not GEMINI_API_KEY:
        return jsonify({"status": "ERROR", "error": "Gemini API not configured"}), 500
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        correction_instructions = []
        if under_keywords:
            under_list = "\n".join([f"  - '{kw['keyword']}': Dodaj {kw['missing']}x" for kw in under_keywords[:10]])
            correction_instructions.append(f"DODAJ te frazy:\n{under_list}")
        
        if over_keywords:
            over_list = "\n".join([f"  - '{kw['keyword']}': Usuń {kw['excess']}x" for kw in over_keywords[:5]])
            correction_instructions.append(f"USUŃ nadmiar:\n{over_list}")
        
        correction_prompt = f"""
Popraw tekst SEO:

{chr(10).join(correction_instructions)}

ZASADY:
1. Zachowaj h2: i h3:
2. Dodawaj frazy naturalnie
3. Zachowaj styl

TEKST:
{batch_text[:12000]}

Zwróć TYLKO poprawiony tekst.
"""
        
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        corrected_text = re.sub(r'^```(?:html)?\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        batches = project_data.get("batches", [])
        auto_saved = False
        new_metrics = {}
        
        if batches:
            batches[-1]["text"] = corrected_text
            batches[-1]["auto_corrected"] = True
            new_metrics = unified_prevalidation(corrected_text, keywords_state)
            batches[-1]["burstiness"] = new_metrics.get("burstiness", 0)
            batches[-1]["density"] = new_metrics.get("density", 0)
            doc_ref.update({"batches": batches})
            auto_saved = True
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "auto_saved": auto_saved,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords]
        }), 200
        
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ================================================================
# 📄 GET FULL ARTICLE (przed eksportem)
# ================================================================
@project_routes.get("/api/project/<project_id>/full_article")
def get_full_article(project_id):
    """
    v26.1: Zwraca pełną treść artykułu przed eksportem.
    GPT używa tego do przeglądu całości.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    keywords_state = data.get("keywords_state", {})
    
    # Złóż pełny tekst
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    # Walidacja końcowa
    coverage = validate_coverage(keywords_state)
    
    # Density
    density = 0
    density_status = "UNKNOWN"
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        density = prevalidation.get("density", 0)
        density_status, _ = get_density_status(density)
    
    # Statystyki
    word_count = len(full_text.split())
    h2_count = full_text.lower().count("h2:")
    h3_count = full_text.lower().count("h3:")
    
    return jsonify({
        "status": "OK",
        "full_article": full_text,
        "stats": {
            "word_count": word_count,
            "batch_count": len(batches),
            "h2_count": h2_count,
            "h3_count": h3_count
        },
        "coverage": coverage,
        "density": {
            "value": round(density, 2),
            "status": density_status
        },
        "topic": data.get("topic"),
        "main_keyword": data.get("main_keyword"),
        "version": "v26.1"
    }), 200


# ================================================================
# 🤖 GEMINI REVIEW (S5)
# ================================================================
@project_routes.post("/api/project/<project_id>/gemini_review")
def gemini_review(project_id):
    """
    v26.1: Wysyła artykuł do Gemini do analizy jakości.
    
    Request body (opcjonalne):
    {
        "focus": ["readability", "seo", "polish_quality"]  // na czym się skupić
    }
    
    Response:
    {
        "status": "APPROVED" | "NEEDS_REVISION",
        "score": 85,
        "recommendations": [...],
        "analysis": {...}
    }
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    topic = data.get("topic", "")
    main_keyword = data.get("main_keyword", topic)
    
    if not full_text or len(full_text) < 500:
        return jsonify({
            "error": "Article too short for review",
            "min_length": 500,
            "current_length": len(full_text)
        }), 400
    
    # Request body
    request_data = request.get_json() or {}
    focus_areas = request_data.get("focus", ["readability", "seo", "polish_quality"])
    
    # Prompt do Gemini
    review_prompt = f"""Przeanalizuj poniższy artykuł SEO i oceń jego jakość.

TEMAT: {topic}
GŁÓWNA FRAZA: {main_keyword}

ARTYKUŁ:
{full_text[:8000]}

OCEŃ (skala 1-100) i podaj rekomendacje dla:
1. CZYTELNOŚĆ - czy tekst jest płynny, zrozumiały, dobrze sformatowany?
2. SEO - czy struktura H2/H3 jest logiczna, czy frazy są naturalnie wplecione?
3. JAKOŚĆ JĘZYKA - czy nie ma błędów, sztucznych fraz AI, powtórzeń?
4. WARTOŚĆ MERYTORYCZNA - czy artykuł odpowiada na pytania użytkownika?

Odpowiedz w formacie JSON:
{{
    "overall_score": <1-100>,
    "scores": {{
        "readability": <1-100>,
        "seo": <1-100>,
        "polish_quality": <1-100>,
        "content_value": <1-100>
    }},
    "status": "APPROVED" lub "NEEDS_REVISION",
    "recommendations": [
        {{"area": "...", "issue": "...", "suggestion": "..."}}
    ],
    "strengths": ["...", "..."],
    "critical_issues": ["..."] 
}}
"""
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            review_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048
            )
        )
        
        if not response or not response.text:
            return jsonify({
                "error": "Empty response from Gemini",
                "status": "ERROR"
            }), 500
        
        # Parsuj JSON z odpowiedzi
        response_text = response.text.strip()
        
        # Usuń markdown code blocks jeśli są
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback - zwróć surową odpowiedź
            analysis = {
                "overall_score": 70,
                "status": "NEEDS_REVISION",
                "raw_response": response.text[:1000],
                "parse_error": True
            }
        
        # Zapisz wynik review do projektu
        doc.reference.update({
            "gemini_review": {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "analysis": analysis,
                "status": analysis.get("status", "UNKNOWN")
            }
        })
        
        return jsonify({
            "status": analysis.get("status", "UNKNOWN"),
            "overall_score": analysis.get("overall_score", 0),
            "scores": analysis.get("scores", {}),
            "recommendations": analysis.get("recommendations", []),
            "strengths": analysis.get("strengths", []),
            "critical_issues": analysis.get("critical_issues", []),
            "version": "v26.1"
        }), 200
        
    except Exception as e:
        print(f"[GEMINI_REVIEW] Error: {e}")
        return jsonify({
            "error": f"Gemini review failed: {str(e)}",
            "status": "ERROR"
        }), 500


# ================================================================
# 💾 SAVE FINAL ARTICLE (przed eksportem)
# ================================================================
@project_routes.post("/api/project/<project_id>/save_final")
def save_final_article(project_id):
    """
    v26.1: Zapisuje finalną wersję artykułu do bazy.
    Wywoływane po przejściu wszystkich review.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    keywords_state = data.get("keywords_state", {})
    
    # Złóż pełny tekst
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    # Konwertuj na HTML
    def convert_markers_to_html(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                title = stripped[3:].strip()
                result.append(f'<h2>{title}</h2>')
            elif stripped.lower().startswith('h3:'):
                title = stripped[3:].strip()
                result.append(f'<h3>{title}</h3>')
            elif stripped.startswith('- ') or stripped.startswith('• '):
                result.append(f'<li>{stripped[2:]}</li>')
            elif stripped:
                result.append(f'<p>{stripped}</p>')
        return '\n'.join(result)
    
    article_html = convert_markers_to_html(full_text)
    
    # Walidacja końcowa
    coverage = validate_coverage(keywords_state)
    
    # Density
    density = 0
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        density = prevalidation.get("density", 0)
    
    # Zapisz do bazy
    final_data = {
        "final_article": {
            "text": full_text,
            "html": article_html,
            "word_count": len(full_text.split()),
            "saved_at": firestore.SERVER_TIMESTAMP
        },
        "final_stats": {
            "coverage": coverage,
            "density": round(density, 2),
            "batch_count": len(batches)
        },
        "status": "FINAL_SAVED"
    }
    
    doc.reference.update(final_data)
    
    return jsonify({
        "status": "SAVED",
        "message": "Final article saved to database",
        "word_count": len(full_text.split()),
        "coverage": coverage,
        "density": round(density, 2),
        "ready_for_export": True,
        "version": "v26.1"
    }), 200


# ================================================================
# 📦 EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    def convert_markers_to_html(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                title = stripped[3:].strip()
                result.append(f'<h2>{title}</h2>')
            elif stripped.lower().startswith('h3:'):
                title = stripped[3:].strip()
                result.append(f'<h3>{title}</h3>')
            else:
                result.append(line)
        return '\n'.join(result)
    
    article_html = convert_markers_to_html(full_text)
    
    # v25.0: Coverage info
    keywords_state = data.get("keywords_state", {})
    coverage = validate_coverage(keywords_state)

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_text": full_text,
        "article_html": article_html,
        "batch_count": len(batches),
        "coverage": coverage,
        "version": "v25.0"
    }), 200


# ================================================================
# 🔄 ALIASES
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    return auto_correct_batch(project_id)


@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    return preview_batch(project_id)


# ================================================================
# v27.2: PHRASE ANALYSIS - dokładne sprawdzenie fraz w tekście
# ================================================================
@project_routes.post("/api/project/<project_id>/analyze_phrases")
def analyze_phrases(project_id):
    """
    Analizuje DOKŁADNE wystąpienia fraz BASIC i EXTENDED w tekście.
    Pokazuje:
    - Gdzie dokładnie fraza występuje (indeks znaków)
    - W jakiej formie (oryginalna vs zlemmatyzowana)
    - Porównanie: regex vs lemmatizer vs firestore
    
    Użycie: przed FAQ żeby sprawdzić które frazy trzeba jeszcze użyć.
    """
    import re
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Field 'text' required"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Import keyword_counter
    try:
        from keyword_counter import count_keywords, count_keywords_for_state
        COUNTER_AVAILABLE = True
    except ImportError:
        COUNTER_AVAILABLE = False
    
    # Zbierz frazy
    all_phrases = []
    for rid, meta in keywords_state.items():
        kw = meta.get("keyword", "").strip()
        kw_type = meta.get("type", "BASIC").upper()
        if not kw:
            continue
        
        all_phrases.append({
            "rid": rid,
            "keyword": kw,
            "type": kw_type,
            "target_min": meta.get("target_min", 1),
            "target_max": meta.get("target_max", 5),
            "actual_in_firestore": meta.get("actual_uses", 0)
        })
    
    # Funkcja szukania DOKŁADNYCH wystąpień (regex, bez lemmatyzacji)
    def find_exact_regex(text_to_search: str, phrase: str) -> list:
        """Znajduje wszystkie dokładne wystąpienia frazy (case-insensitive, regex)."""
        matches = []
        text_lower = text_to_search.lower()
        phrase_lower = phrase.lower()
        
        # Szukaj z word boundaries
        pattern = r'\b' + re.escape(phrase_lower) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start = match.start()
            end = match.end()
            original_form = text_to_search[start:end]
            
            ctx_start = max(0, start - 25)
            ctx_end = min(len(text_to_search), end + 25)
            context = text_to_search[ctx_start:ctx_end]
            
            matches.append({
                "pos": f"{start}-{end}",
                "found": original_form,
                "ctx": f"...{context}..."
            })
        
        return matches
    
    # Policz każdą frazę na 3 sposoby
    analysis = []
    
    # 1. Unified counter (jeśli dostępny)
    if COUNTER_AVAILABLE:
        keywords_list = [p["keyword"] for p in all_phrases]
        unified_result = count_keywords(text, keywords_list, return_per_segment=False, return_paragraph_stuffing=False)
        overlapping = unified_result.get("overlapping", {})
        exclusive = unified_result.get("exclusive", {})
    else:
        overlapping = {}
        exclusive = {}
    
    for phrase_info in all_phrases:
        kw = phrase_info["keyword"]
        
        # Regex count (dokładne dopasowanie, bez lemmatyzacji)
        regex_matches = find_exact_regex(text, kw)
        regex_count = len(regex_matches)
        
        # Unified counter counts
        overlap_count = overlapping.get(kw, 0)
        excl_count = exclusive.get(kw, 0)
        
        # Firestore value
        firestore_count = phrase_info["actual_in_firestore"]
        
        # Status
        target_min = phrase_info["target_min"] if phrase_info["type"] != "EXTENDED" else 1
        
        # Wykryj rozbieżności
        discrepancy = None
        if regex_count != overlap_count:
            discrepancy = f"REGEX({regex_count}) != LEMMA({overlap_count})"
        
        analysis.append({
            "keyword": kw,
            "type": phrase_info["type"],
            "rid": phrase_info["rid"],
            
            # 3 metody liczenia
            "count_regex": regex_count,           # Dokładne dopasowanie (bez odmian)
            "count_overlapping": overlap_count,   # Z lemmatyzacją (Google-style)
            "count_exclusive": excl_count,        # Bez zagnieżdżonych
            "count_firestore": firestore_count,   # Zapisane w Firestore
            
            # Targety
            "target_min": target_min,
            "target_max": phrase_info["target_max"],
            
            # Status
            "status": "✅" if overlap_count >= target_min else "❌",
            "discrepancy": discrepancy,
            
            # Przykłady (max 5)
            "examples": regex_matches[:5]
        })
    
    # Podsumowanie
    basic_analysis = [a for a in analysis if a["type"] == "BASIC"]
    extended_analysis = [a for a in analysis if a["type"] == "EXTENDED"]
    
    basic_missing = [a["keyword"] for a in basic_analysis if a["count_overlapping"] < a["target_min"]]
    extended_missing = [a["keyword"] for a in extended_analysis if a["count_overlapping"] < 1]
    
    # Wykryj problemy
    problems = []
    for a in analysis:
        if a["discrepancy"]:
            problems.append(f"{a['keyword']}: {a['discrepancy']}")
        if a["count_firestore"] != a["count_overlapping"]:
            problems.append(f"{a['keyword']}: Firestore({a['count_firestore']}) != Text({a['count_overlapping']})")
    
    return jsonify({
        "project_id": project_id,
        "text_length": len(text),
        "word_count": len(text.split()),
        "counter_type": "unified_lemmatizer" if COUNTER_AVAILABLE else "regex_only",
        
        "summary": {
            "basic_total": len(basic_analysis),
            "basic_covered": len(basic_analysis) - len(basic_missing),
            "basic_missing": len(basic_missing),
            "extended_total": len(extended_analysis),
            "extended_covered": len(extended_analysis) - len(extended_missing),
            "extended_missing": len(extended_missing)
        },
        
        "missing_basic": basic_missing,
        "missing_extended": extended_missing,
        
        "problems_detected": problems[:10],
        
        "analysis": analysis,
        
        "legend": {
            "count_regex": "Dokładne dopasowanie (bez odmian)",
            "count_overlapping": "Z lemmatyzacją + zagnieżdżone (Google-style)",
            "count_exclusive": "Z lemmatyzacją, BEZ zagnieżdżonych",
            "count_firestore": "Wartość zapisana w Firestore (może być nieaktualna)"
        }
    })


# ================================================================
# 🆕 v43.0: PHRASE HIERARCHY ENDPOINTS
# ================================================================

@project_routes.get("/api/project/<project_id>/phrase_hierarchy")
def get_phrase_hierarchy(project_id):
    """
    🆕 v43.0: Zwraca globalną hierarchię fraz dla projektu.
    
    Zawiera:
    - roots: Rdzenie fraz i ich rozszerzenia
    - effective_targets: Efektywne targety z uwzględnieniem rozszerzeń
    - entity_phrases: Frazy będące encjami
    - triplet_phrases: Frazy z tripletów
    - strategies: extensions_sufficient / mixed / need_standalone
    """
    try:
        db = firestore.client()
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        project = doc.to_dict()
        hierarchy_data = project.get("phrase_hierarchy")
        
        if not hierarchy_data:
            return jsonify({
                "error": "Phrase hierarchy not available",
                "message": "This project was created before v43.0. Use /analyze_hierarchy to generate.",
                "project_id": project_id
            }), 404
        
        # Format dla agenta
        formatted = ""
        summary = ""
        if PHRASE_HIERARCHY_ENABLED:
            try:
                hierarchy = dict_to_hierarchy(hierarchy_data)
                formatted = format_hierarchy_for_agent(hierarchy, include_full_list=True)
                summary = format_hierarchy_summary_short(hierarchy)
            except Exception as e:
                print(f"[PHRASE_HIERARCHY] Format error: {e}")
        
        # Podsumowanie strategii
        roots = hierarchy_data.get("roots", {})
        effective_targets = hierarchy_data.get("effective_targets", {})
        stats = hierarchy_data.get("stats", {})
        
        strategies_summary = {
            "extensions_sufficient": [],
            "mixed": [],
            "need_standalone": []
        }
        
        for root, target_info in effective_targets.items():
            strategy = target_info.get("strategy", "unknown")
            if strategy in strategies_summary:
                strategies_summary[strategy].append({
                    "root": root,
                    "original": f"{target_info.get('original_min', 0)}-{target_info.get('original_max', 0)}x",
                    "effective": f"{target_info.get('effective_min', 0)}-{target_info.get('effective_max', 0)}x",
                    "extensions": target_info.get("extensions_count", 0)
                })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "phrase_hierarchy": hierarchy_data,
            "stats": stats,
            "strategies": {
                "extensions_sufficient": {
                    "description": "Rozszerzenia wystarczą - NIE powtarzaj rdzenia osobno!",
                    "roots": strategies_summary["extensions_sufficient"]
                },
                "mixed": {
                    "description": "Częściowo pokryte - użyj kilka samodzielnie + rozszerzenia",
                    "roots": strategies_summary["mixed"]
                },
                "need_standalone": {
                    "description": "Mało rozszerzeń - użyj samodzielnie",
                    "roots": strategies_summary["need_standalone"]
                }
            },
            "formatted_for_agent": formatted,
            "summary": summary,
            "version": "v44.0"
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@project_routes.post("/api/project/<project_id>/analyze_hierarchy")
def analyze_hierarchy_for_project(project_id):
    """
    🆕 v43.0: Analizuje hierarchię fraz dla istniejącego projektu.
    
    Używaj dla projektów utworzonych przed v43.0.
    Generuje phrase_hierarchy i zapisuje w Firestore.
    """
    if not PHRASE_HIERARCHY_ENABLED:
        return jsonify({
            "error": "Phrase hierarchy module not available",
            "message": "Install phrase_hierarchy.py in API directory"
        }), 500
    
    try:
        db = firestore.client()
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        project = doc.to_dict()
        keywords_state = project.get("keywords_state", {})
        s1_data = project.get("s1_data", {})
        h2_structure = project.get("h2_structure", [])
        
        # Przygotuj encje z S1
        s1_entities = []
        entity_seo = s1_data.get("entity_seo", {})
        
        for ent in entity_seo.get("entities", []):
            s1_entities.append({
                "name": ent.get("text", ent.get("entity", ent.get("name", ""))),
                "priority": ent.get("priority", "SHOULD"),
                "importance": ent.get("importance", 0.5)
            })
        
        for topic_item in entity_seo.get("topical_coverage", []):
            if topic_item.get("priority") == "MUST":
                s1_entities.append({
                    "name": topic_item.get("topic", ""),
                    "priority": "MUST",
                    "importance": 0.9
                })
        
        s1_triplets = entity_seo.get("entity_relationships", [])
        
        # Konwertuj keywords_state do listy
        keywords_list = []
        for rid, meta in keywords_state.items():
            keywords_list.append({
                "keyword": meta.get("keyword", ""),
                "type": meta.get("type", "BASIC"),
                "target_min": meta.get("target_min", 1),
                "target_max": meta.get("target_max", 5)
            })
        
        # Analizuj hierarchię
        hierarchy = analyze_phrase_hierarchy(
            keywords=keywords_list,
            entities=s1_entities,
            triplets=s1_triplets,
            h2_terms=h2_structure
        )
        
        # Konwertuj do dict i zapisz
        phrase_hierarchy_data = hierarchy_to_dict(hierarchy)
        
        doc_ref.update({
            "phrase_hierarchy": sanitize_for_firestore(phrase_hierarchy_data),
            "phrase_hierarchy_analyzed_at": firestore.SERVER_TIMESTAMP
        })
        
        # Format dla agenta
        formatted = format_hierarchy_for_agent(hierarchy, include_full_list=True)
        
        return jsonify({
            "status": "OK",
            "message": "Phrase hierarchy analyzed and saved",
            "project_id": project_id,
            "stats": hierarchy.stats,
            "roots_count": hierarchy.stats.get("roots_count", 0),
            "entity_phrases": hierarchy.stats.get("entity_phrases", 0),
            "triplet_phrases": hierarchy.stats.get("triplet_phrases", 0),
            "formatted_for_agent": formatted,
            "version": "v44.0"
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
