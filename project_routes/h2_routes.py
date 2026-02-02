"""
PROJECT ROUTES - H2 MANAGEMENT v44.1
====================================
Endpointy do zarządzania H2 (nagłówkami) w projekcie.

Zawiera:
- generate_h2_suggestions - sugestie H2 (Claude/Gemini)
- finalize_h2 - finalizacja H2 z frazami usera
- validate_h2_plan - walidacja planu H2
- save_h2_plan - zapis planu H2
- get_h2_plan - pobranie planu H2
- update_h2_plan - aktualizacja planu H2

Autor: BRAJEN SEO Engine v44.1
"""

import os
from flask import Blueprint, request, jsonify
from firebase_admin import firestore

# ================================================================
# BLUEPRINT
# ================================================================
h2_routes = Blueprint("h2_routes", __name__)


# ================================================================
# OPTIONAL IMPORTS
# ================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

GEMINI_MODEL = "gemini-2.5-flash"

# H2 Generator module
try:
    from h2_generator import validate_h2_plan
    H2_GENERATOR_ENABLED = True
except ImportError:
    H2_GENERATOR_ENABLED = False
    def validate_h2_plan(h2_plan, main_keyword):
        return {"valid": True, "issues": [], "warnings": []}


# ================================================================
# H2 SUGGESTIONS (Claude primary, Gemini fallback)
# ================================================================
@h2_routes.post("/api/project/s1_h2_suggestions")
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
    
    # Build prompt
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
    if not suggestions and GENAI_AVAILABLE and GEMINI_API_KEY:
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
@h2_routes.post("/api/project/finalize_h2")
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
    
    if not GENAI_AVAILABLE or not GEMINI_API_KEY or not user_h2_phrases:
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
# HELPER: CHECK PHRASE COVERAGE
# ================================================================
def check_phrase_coverage(h2_plan: list, h2_phrases: list, main_keyword: str) -> dict:
    """Sprawdza czy wszystkie frazy użytkownika są pokryte w H2."""
    
    h2_texts = []
    for h2 in h2_plan:
        if isinstance(h2, str):
            h2_texts.append(h2.lower())
        elif isinstance(h2, dict):
            h2_texts.append(h2.get("h2", "").lower())
    
    all_h2_text = " ".join(h2_texts)
    
    main_keyword_covered = main_keyword.lower() in all_h2_text
    
    covered = []
    missing = []
    
    for phrase in h2_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in all_h2_text:
            covered.append(phrase)
        else:
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


# ================================================================
# VALIDATE H2 PLAN
# ================================================================
@h2_routes.post("/api/project/<project_id>/validate_h2_plan")
def validate_h2_plan_endpoint(project_id):
    """
    Waliduje plan H2 stworzony przez Claude.
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
        validation = validate_h2_plan(h2_plan, main_keyword)
        coverage = check_phrase_coverage(h2_plan, h2_phrases, main_keyword)
        is_valid = validation["valid"] and coverage["all_phrases_covered"]
        
        if is_valid:
            db = firestore.client()
            project_ref = db.collection("projects").document(project_id)
            if project_ref.get().exists:
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


# ================================================================
# SAVE H2 PLAN
# ================================================================
@h2_routes.post("/api/project/<project_id>/save_h2_plan")
def save_h2_plan_endpoint(project_id):
    """Zapisuje plan H2 do projektu (bez walidacji)."""
    data = request.get_json() or {}
    h2_plan = data.get("h2_plan", [])
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required"}), 400
    
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        
        if not project_ref.get().exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
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


# ================================================================
# GET H2 PLAN
# ================================================================
@h2_routes.get("/api/project/<project_id>/h2_plan")
def get_h2_plan(project_id):
    """Pobiera zapisany plan H2 dla projektu."""
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


# ================================================================
# UPDATE H2 PLAN
# ================================================================
@h2_routes.post("/api/project/<project_id>/update_h2_plan")
def update_h2_plan(project_id):
    """Aktualizuje plan H2 (po modyfikacjach użytkownika)."""
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
        
        if H2_GENERATOR_ENABLED:
            validation = validate_h2_plan(h2_plan, main_keyword)
        else:
            validation = {"valid": True, "issues": [], "warnings": []}
        
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
# EXPORTS
# ================================================================
__all__ = [
    "h2_routes",
    "check_phrase_coverage",
]
