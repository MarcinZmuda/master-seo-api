"""
===============================================================================
📦 PROJECT ROUTES v23.0 - Zrefaktoryzowany
===============================================================================
Zmiany:
1. Ujednolicone formaty JSON (konsystentne nazwy pól)
2. Integracja z BatchPlanner (planowanie z góry)
3. Integracja z VersionManager (wersjonowanie)
4. Integracja z UnifiedValidator (jedna walidacja)
5. Połączony endpoint S1 + H2 suggestions
===============================================================================
"""

import uuid
import re
import os
import json
import math
import spacy
import requests
from flask import Blueprint, request as flask_request, jsonify
from firebase_admin import firestore
import google.generativeai as genai
from datetime import datetime, timezone

# Import nowych modułów
from unified_validator import (
    validate_content, quick_validate, ValidationConfig,
    lemmatize_text, count_keyword_occurrences
)
from batch_planner import (
    create_article_plan, get_batch_instructions, update_plan_after_batch,
    ArticlePlan, BatchPlan
)
from version_manager import VersionManager, VersionSource, create_version_manager_for_project

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Współdzielony model spaCy
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
except ImportError:
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)
GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# 🔧 HELPER: Normalizacja nazw pól
# ================================================================
def normalize_input(data: dict) -> dict:
    """Normalizuje nazwy pól wejściowych do standardowego formatu."""
    normalized = {}
    
    normalized["main_keyword"] = (
        data.get("main_keyword") or 
        data.get("topic") or 
        data.get("keyword") or ""
    ).strip()
    
    normalized["text"] = (
        data.get("text") or 
        data.get("batch_text") or 
        data.get("content") or ""
    )
    
    raw_keywords = data.get("keywords_list") or data.get("keywords") or []
    normalized["keywords"] = []
    for kw in raw_keywords:
        if isinstance(kw, str):
            normalized["keywords"].append({"keyword": kw, "min": 1, "max": 5, "type": "BASIC"})
        elif isinstance(kw, dict):
            normalized["keywords"].append({
                "keyword": kw.get("term") or kw.get("keyword") or "",
                "min": kw.get("min") or kw.get("target_min") or 1,
                "max": kw.get("max") or kw.get("target_max") or 5,
                "type": (kw.get("type") or "BASIC").upper()
            })
    
    normalized["h2_structure"] = data.get("h2_structure") or data.get("h2_list") or []
    normalized["target_length"] = data.get("target_length") or data.get("length") or 3000
    normalized["total_batches"] = (
        data.get("total_planned_batches") or 
        data.get("total_batches") or None
    )
    
    return normalized


# ================================================================
# ⭐ v23.0: POŁĄCZONY ENDPOINT S1 + H2 SUGGESTIONS
# ================================================================
@project_routes.post("/api/article/analyze_and_plan")
def analyze_and_plan():
    """
    NOWY POŁĄCZONY ENDPOINT - wykonuje S1 + H2 suggestions + plan w jednym requeście.
    """
    data = flask_request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    normalized = normalize_input(data)
    main_keyword = normalized["main_keyword"]
    
    if not main_keyword:
        return jsonify({"error": "Required: main_keyword"}), 400
    
    target_length = normalized["target_length"]
    user_keywords = normalized["keywords"]
    user_h2 = normalized["h2_structure"]
    
    # 1. S1 ANALYSIS (proxy do N-gram API)
    NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com")
    
    # FIX: Sprawdź czy URL już zawiera endpoint (unikaj duplikacji)
    if "/api/ngram_entity_analysis" in NGRAM_API_URL:
        ngram_endpoint = NGRAM_API_URL
    else:
        ngram_endpoint = f"{NGRAM_API_URL}/api/ngram_entity_analysis"
    
    s1_data = {}
    try:
        s1_response = requests.post(
            ngram_endpoint,
            json={"main_keyword": main_keyword, "top_n": 30},
            timeout=60
        )
        if s1_response.status_code == 200:
            s1_data = s1_response.json()
    except Exception as e:
        s1_data = {"error": str(e)}
    
    # 2. H2 SUGGESTIONS (Gemini)
    h2_suggestions = []
    if not user_h2 and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            serp_h2 = s1_data.get("serp_analysis", {}).get("competitor_h2_patterns", [])
            competitor_context = ""
            if serp_h2:
                competitor_context = f"\nWZORCE H2 Z KONKURENCJI:\n" + "\n".join(f"- {h}" for h in serp_h2[:15])
            
            prompt = f"""Wygeneruj 5 nagłówków H2 dla artykułu o: "{main_keyword}"
{competitor_context}

ZASADY:
- Min 2 z 5 H2 musi zawierać "{main_keyword}"
- Min 3 w formie pytania
- NIE używaj: Wstęp, Podsumowanie, Zakończenie, FAQ
- 6-12 słów każdy

Zwróć TYLKO listę H2, każdy w nowej linii."""
            
            response = model.generate_content(prompt)
            h2_suggestions = [
                h.strip().lstrip('•-–—0123456789.). ')
                for h in response.text.strip().split('\n') 
                if h.strip() and len(h.strip()) > 5
            ][:6]
        except Exception as e:
            print(f"[ANALYZE] H2 error: {e}")
    
    final_h2 = user_h2 if user_h2 else h2_suggestions
    
    # 3. Wykryj synonimy
    synonyms = []
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(f"Podaj 2-4 synonimy dla: \"{main_keyword}\". Jeden na linię.")
            synonyms = [s.strip() for s in response.text.strip().split('\n') if s.strip()][:4]
        except:
            pass
    
    # 4. Przygotuj keywords
    suggested_keywords = []
    main_min = max(6, target_length // 350)
    main_max = target_length // 150
    
    suggested_keywords.append({
        "keyword": main_keyword, "min": main_min, "max": main_max, 
        "type": "MAIN", "is_main": True
    })
    
    for syn in synonyms:
        suggested_keywords.append({
            "keyword": syn, "min": 1, "max": 3, 
            "type": "BASIC", "is_synonym": True
        })
    
    for kp in s1_data.get("semantic_keyphrases", [])[:5]:
        phrase = kp.get("phrase", "") if isinstance(kp, dict) else kp
        if phrase and phrase.lower() != main_keyword.lower():
            suggested_keywords.append({"keyword": phrase, "min": 1, "max": 4, "type": "EXTENDED"})
    
    # 5. Walidacja H2
    main_lower = main_keyword.lower()
    h2_with_main = sum(1 for h in final_h2 if main_lower in h.lower())
    h2_coverage = h2_with_main / len(final_h2) if final_h2 else 0
    
    # 6. Entity + Hybrid N-gram Analysis (jeśli mamy content z S1)
    entity_analysis = {}
    hybrid_ngrams = []
    try:
        from entity_ngram_analyzer import analyze_content_semantics
        
        # Analizuj konkurencję z S1
        competitor_content = s1_data.get("full_text_sample", "") or s1_data.get("serp_content", "")
        if competitor_content:
            entity_analysis = analyze_content_semantics(
                text=competitor_content,
                s1_ngrams=s1_data.get("ngrams", []),
                main_keyword=main_keyword
            )
            hybrid_ngrams = entity_analysis.get("hybrid_ngrams", [])[:15]
    except ImportError:
        pass  # entity_ngram_analyzer niedostępny
    except Exception as e:
        entity_analysis = {"error": str(e)}
    
    # 7. Wstępny plan
    temp_kw_state = {f"kw_{i}": {
        "keyword": k["keyword"], "target_min": k["min"], "target_max": k["max"],
        "type": k["type"], "is_main_keyword": k.get("is_main", False),
        "is_synonym_of_main": k.get("is_synonym", False), "actual_uses": 0
    } for i, k in enumerate(suggested_keywords)}
    
    # Użyj hybrid_ngrams jeśli dostępne, otherwise standard ngrams
    if hybrid_ngrams:
        ngrams = [n.get("ngram", "") for n in hybrid_ngrams if n.get("weight", 0) > 0.4][:20]
    else:
        ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.4][:20]
    
    plan = create_article_plan(
        h2_structure=final_h2,
        keywords_state=temp_kw_state,
        main_keyword=main_keyword,
        target_length=target_length,
        ngrams=ngrams
    )
    
    # 8. E-E-A-T prompt enhancement
    eeat_prompt = ""
    try:
        from entity_ngram_analyzer import generate_eeat_prompt_enhancement
        eeat_prompt = generate_eeat_prompt_enhancement(main_keyword)
    except ImportError:
        eeat_prompt = """
⭐ E-E-A-T TIPS:
- Dodaj sygnały ekspertyzy (terminologia branżowa)
- Cytuj źródła (przepisy, dane, badania)
- Podawaj aktualne daty i konkretne liczby
"""
    
    return jsonify({
        "status": "ANALYSIS_COMPLETE",
        "main_keyword": main_keyword,
        "synonyms": synonyms,
        "h2_structure": {
            "suggestions": h2_suggestions,
            "final": final_h2,
            "coverage": round(h2_coverage, 2),
            "valid": h2_coverage >= 0.4
        },
        "keywords": {"suggested": suggested_keywords, "count": len(suggested_keywords)},
        "s1_summary": {
            "ngrams_count": len(s1_data.get("ngrams", [])),
            "paa_count": len(s1_data.get("serp_analysis", {}).get("paa_questions", [])),
            "has_featured_snippet": bool(s1_data.get("serp_analysis", {}).get("featured_snippet"))
        },
        "entity_analysis": {
            "entities_found": len(entity_analysis.get("entities", [])),
            "top_entities": entity_analysis.get("entity_summary", {}).get("top_entities", [])[:5],
            "hybrid_ngrams_count": len(hybrid_ngrams),
            "recommendations": entity_analysis.get("recommendations", [])[:3]
        },
        "s1_data": s1_data,
        "eeat_prompt": eeat_prompt,
        "preliminary_plan": {
            "total_batches": plan.total_batches,
            "target_words": plan.total_target_words,
            "batches": [{
                "batch": b.batch_number, "type": b.batch_type,
                "h2": b.h2_sections, "words": f"{b.target_words_min}-{b.target_words_max}"
            } for b in plan.batches]
        },
        "next_step": "POST /api/project/create"
    }), 200


# ================================================================
# 🧱 PROJECT CREATE v23.0
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    """Tworzy projekt z BatchPlanem i VersionManagerem."""
    data = flask_request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    normalized = normalize_input(data)
    main_keyword = normalized["main_keyword"]
    
    if not main_keyword:
        return jsonify({"error": "Required: main_keyword"}), 400
    
    h2_structure = normalized["h2_structure"]
    keywords_list = normalized["keywords"]
    target_length = normalized["target_length"]
    s1_data = data.get("s1_data", {})
    synonyms = data.get("synonyms", [])
    
    # Przygotuj keywords_state
    keywords_state = {}
    main_found = False
    
    for i, kw in enumerate(keywords_list):
        keyword = kw.get("keyword", "").strip()
        if not keyword:
            continue
        
        doc = nlp(keyword)
        lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        is_main = keyword.lower() == main_keyword.lower()
        is_syn = keyword.lower() in [s.lower() for s in synonyms]
        
        min_val = kw.get("min", 1)
        max_val = kw.get("max", 5)
        kw_type = kw.get("type", "BASIC").upper()
        
        if is_main:
            main_found = True
            kw_type = "MAIN"
            min_val = max(min_val, max(6, target_length // 350))
            max_val = max(max_val, target_length // 150)
        
        rid = f"kw_{i}_{uuid.uuid4().hex[:6]}"
        keywords_state[rid] = {
            "keyword": keyword,
            "search_lemma": lemma,
            "target_min": min_val,
            "target_max": max_val,
            "actual_uses": 0,
            "status": "UNDER",
            "type": kw_type,
            "is_main_keyword": is_main,
            "is_synonym_of_main": is_syn,
            "remaining_max": max_val
        }
    
    # Auto-dodaj main keyword
    if not main_found:
        doc = nlp(main_keyword)
        lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        keywords_state["main_auto"] = {
            "keyword": main_keyword,
            "search_lemma": lemma,
            "target_min": max(6, target_length // 350),
            "target_max": target_length // 150,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN",
            "is_main_keyword": True,
            "is_synonym_of_main": False
        }
    
    # Stwórz plan
    ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.4][:20]
    
    article_plan = create_article_plan(
        h2_structure=h2_structure,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        target_length=target_length,
        ngrams=ngrams
    )
    
    # Zapisz do Firestore
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    project_data = {
        "main_keyword": main_keyword,
        "topic": main_keyword,
        "synonyms": synonyms,
        "h2_structure": h2_structure,
        "keywords_state": keywords_state,
        "target_length": target_length,
        "article_plan": article_plan.to_dict(),
        "s1_data": s1_data,
        "batches": [],
        "version_history": {},
        "created_at": firestore.SERVER_TIMESTAMP,
        "version": "v23.0"
    }
    
    doc_ref.set(project_data)
    project_id = doc_ref.id
    
    return jsonify({
        "status": "CREATED",
        "project_id": project_id,
        "main_keyword": main_keyword,
        "keywords_count": len(keywords_state),
        "h2_count": len(h2_structure),
        "plan": {
            "total_batches": article_plan.total_batches,
            "target_words": article_plan.total_target_words
        },
        "first_batch_instructions": get_batch_instructions(article_plan, 1)
    }), 200  # Changed from 201 for OpenAPI compatibility


# ================================================================
# 📋 PRE-BATCH INFO v23.0 - używa BatchPlannera
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """Zwraca instrukcje dla następnego batcha z planu."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    article_plan_data = data.get("article_plan", {})
    main_keyword = data.get("main_keyword", "")
    
    current_batch_num = len(batches) + 1
    total_batches = 4  # Default
    
    # Rekonstruuj plan z danych
    article_plan = None
    if article_plan_data:
        # Rekonstruuj plan
        batch_plans = []
        for bp_data in article_plan_data.get("batches", []):
            batch_plans.append(BatchPlan(
                batch_number=bp_data.get("batch_number", 1),
                batch_type=bp_data.get("batch_type", "CONTENT"),
                h2_sections=bp_data.get("h2_sections", []),
                target_words_min=bp_data.get("target_words_min", 400),
                target_words_max=bp_data.get("target_words_max", 600),
                keywords_budget=bp_data.get("keywords_budget", {}),
                ngrams_to_use=bp_data.get("ngrams_to_use", []),
                notes=bp_data.get("notes", "")
            ))
        
        article_plan = ArticlePlan(
            total_batches=article_plan_data.get("total_batches", len(batch_plans)),
            total_target_words=article_plan_data.get("total_target_words", 3000),
            batches=batch_plans,
            keywords_distribution=article_plan_data.get("keywords_distribution", {}),
            main_keyword=main_keyword,
            h2_structure=article_plan_data.get("h2_structure", [])
        )
        
        total_batches = article_plan.total_batches
        instructions = get_batch_instructions(article_plan, current_batch_num, keywords_state)
    else:
        # Fallback - stary format
        instructions = {"error": "No article plan found", "batch_number": current_batch_num}
    
    # Dodaj info o main vs synonyms
    main_uses = 0
    synonym_uses = 0
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            main_uses = meta.get("actual_uses", 0)
        elif meta.get("is_synonym_of_main"):
            synonym_uses += meta.get("actual_uses", 0)
    
    total = main_uses + synonym_uses
    main_ratio = main_uses / total if total > 0 else 1.0
    
    return jsonify({
        "project_id": project_id,
        "batch_number": current_batch_num,
        "total_batches": total_batches,
        "main_keyword_status": {
            "main_keyword": main_keyword,
            "main_uses": main_uses,
            "synonym_uses": synonym_uses,
            "main_ratio": round(main_ratio, 2),
            "valid": main_ratio >= 0.5
        },
        "instructions": instructions,
        "gpt_prompt": instructions.get("gpt_prompt", "")
    }), 200


# ================================================================
# ✏️ PREVIEW BATCH v23.0 - używa UnifiedValidator
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """Waliduje batch używając UnifiedValidator."""
    data = flask_request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    
    normalized = normalize_input(data)
    text = normalized["text"]
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", "")
    batches = project_data.get("batches", [])
    s1_data = project_data.get("s1_data", {})
    
    # Pobierz ngrams
    ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.5][:10]
    
    # Policz istniejące listy
    existing_lists = sum(
        len(re.findall(r'<ul>|<ol>|^\s*[-•]\s', b.get("text", ""), re.MULTILINE))
        for b in batches
    )
    
    # Walidacja
    result = validate_content(
        text=text,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        required_ngrams=ngrams,
        is_intro_batch=(len(batches) == 0),
        existing_lists_count=existing_lists
    )
    
    return jsonify({
        "status": "OK" if result.is_valid else "WARN",
        "score": result.score,
        "is_valid": result.is_valid,
        "issues": [i.to_dict() for i in result.issues],
        "metrics": result.metrics,
        "keywords_analysis": result.keywords_analysis,
        "structure": result.structure_analysis,
        "mode": "PREVIEW"
    }), 200


# ================================================================
# ✅ APPROVE BATCH v23.0 - z wersjonowaniem
# ================================================================
@project_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """Zapisuje batch z wersjonowaniem."""
    data = flask_request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    
    normalized = normalize_input(data)
    text = normalized["text"]
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    batches = project_data.get("batches", [])
    main_keyword = project_data.get("main_keyword", "")
    version_history = project_data.get("version_history", {})
    
    batch_number = len(batches) + 1
    
    # Walidacja
    result = validate_content(text=text, keywords_state=keywords_state, main_keyword=main_keyword)
    
    # Aktualizuj keywords_state
    text_lemmas = lemmatize_text(text)
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        count = count_keyword_occurrences(text_lemmas, keyword)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        
        actual = meta["actual_uses"]
        min_t = meta.get("target_min", 0)
        max_t = meta.get("target_max", 999)
        
        if actual < min_t:
            meta["status"] = "UNDER"
        elif actual > max_t:
            meta["status"] = "OVER"
        else:
            meta["status"] = "OK"
        
        meta["remaining_max"] = max(0, max_t - actual)
    
    # Wersjonowanie
    vm = VersionManager(project_id)
    if version_history:
        vm = VersionManager.from_dict({"project_id": project_id, "batch_histories": version_history})
    
    version = vm.add_version(
        batch_number=batch_number,
        text=text,
        source=VersionSource.MANUAL,
        metadata={"validation_score": result.score}
    )
    
    # Zapisz batch
    batch_entry = {
        "text": text,
        "batch_number": batch_number,
        "timestamp": datetime.now(timezone.utc),
        "version_id": version.version_id,
        "validation": {
            "score": result.score,
            "is_valid": result.is_valid,
            "issues_count": len(result.issues)
        },
        "metrics": result.metrics
    }
    
    batches.append(batch_entry)
    
    # Update Firestore
    doc_ref.update({
        "batches": batches,
        "keywords_state": keywords_state,
        "version_history": vm.to_dict()["batch_histories"]
    })
    
    return jsonify({
        "status": "APPROVED",
        "batch_number": batch_number,
        "version_id": version.version_id,
        "validation": {
            "score": result.score,
            "is_valid": result.is_valid,
            "warnings": len(result.get_warnings()),
            "errors": len(result.get_errors())
        },
        "keywords_summary": {
            rid: {"keyword": m["keyword"], "actual": m["actual_uses"], "status": m["status"]}
            for rid, m in list(keywords_state.items())[:10]
        }
    }), 200


# ================================================================
# 🔄 ROLLBACK BATCH v23.0
# ================================================================
@project_routes.post("/api/project/<project_id>/rollback_batch")
def rollback_batch(project_id):
    """Przywraca poprzednią wersję batcha."""
    data = flask_request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    
    batch_number = data.get("batch_number")
    version_id = data.get("version_id")
    reason = data.get("reason", "Manual rollback")
    
    if not batch_number or not version_id:
        return jsonify({"error": "Required: batch_number, version_id"}), 400
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    version_history = project_data.get("version_history", {})
    batches = project_data.get("batches", [])
    
    # Odtwórz VersionManager
    vm = VersionManager.from_dict({"project_id": project_id, "batch_histories": version_history})
    
    # Rollback
    new_version = vm.rollback_to_version(batch_number, version_id, reason)
    
    if not new_version:
        return jsonify({"error": "Version not found"}), 404
    
    # Aktualizuj batch
    for batch in batches:
        if batch.get("batch_number") == batch_number:
            batch["text"] = new_version.text
            batch["version_id"] = new_version.version_id
            batch["rollback_from"] = version_id
            break
    
    # Zapisz
    doc_ref.update({
        "batches": batches,
        "version_history": vm.to_dict()["batch_histories"]
    })
    
    return jsonify({
        "status": "ROLLED_BACK",
        "batch_number": batch_number,
        "new_version_id": new_version.version_id,
        "rolled_back_to": version_id
    }), 200


# ================================================================
# 📜 VERSION HISTORY
# ================================================================
@project_routes.get("/api/project/<project_id>/versions/<int:batch_number>")
def get_batch_versions(project_id, batch_number):
    """Zwraca historię wersji batcha."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    version_history = project_data.get("version_history", {})
    
    vm = VersionManager.from_dict({"project_id": project_id, "batch_histories": version_history})
    history = vm.get_history(batch_number)
    
    if not history:
        return jsonify({"error": "No history for this batch"}), 404
    
    return jsonify({
        "batch_number": batch_number,
        "versions": [v.to_dict() for v in history.versions],
        "current_version_id": history.current_version_id
    }), 200


# ================================================================
# 📦 EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project(project_id):
    """Eksportuje artykuł."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    # Convert markers to HTML
    def convert_markers(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                result.append(f'<h2>{stripped[3:].strip()}</h2>')
            elif stripped.lower().startswith('h3:'):
                result.append(f'<h3>{stripped[3:].strip()}</h3>')
            else:
                result.append(line)
        return '\n'.join(result)
    
    html = convert_markers(full_text)
    
    # Pobierz PAA jeśli jest
    paa_section = data.get("paa_section", {})
    if paa_section:
        html += "\n\n" + paa_section.get("html_schema", "")
    
    return jsonify({
        "status": "EXPORT_READY",
        "main_keyword": data.get("main_keyword"),
        "article_text": full_text,
        "article_html": html,
        "batch_count": len(batches),
        "word_count": len(full_text.split()),
        "version": "v23.0"
    }), 200
