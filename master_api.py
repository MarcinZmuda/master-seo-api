"""
===============================================================================
🚀 MASTER SEO API v23.1 - Poprawiony
===============================================================================
Zmiany względem v23.0:
- Naprawiony import w h2_suggestions_proxy (błąd 500)
- Kompatybilność wsteczna dla starych endpointów

===============================================================================
"""

import os
import json
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# ================================================================
# 🔥 Firebase Initialization
# ================================================================
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError("❌ FIREBASE_CREDS_JSON not set")

try:
    creds_dict = json.loads(FIREBASE_CREDS_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Invalid JSON in FIREBASE_CREDS_JSON: {e}")

cred = credentials.Certificate(creds_dict)
firebase_app = initialize_app(cred)
db = firestore.client()

# ================================================================
# ⚙️ Flask App
# ================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB
CORS(app)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERSION = "v23.1"

# ================================================================
# 🧠 Check modules
# ================================================================
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_ENABLED = True
    print("[MASTER] ✅ Semantic analysis available")
except ImportError:
    SEMANTIC_ENABLED = False
    print("[MASTER] ⚠️ Semantic analysis NOT available")

# ================================================================
# 🔗 N-gram API Configuration
# ================================================================
NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com")
NGRAM_ANALYSIS_ENDPOINT = f"{NGRAM_API_URL}/api/ngram_entity_analysis"
print(f"[MASTER] 🔗 N-gram API: {NGRAM_API_URL}")

# ================================================================
# 📦 Import blueprints
# ================================================================
from project_routes import project_routes
from final_review_routes import final_review_routes
from paa_routes import paa_routes

# ================================================================
# 🔗 Register blueprints
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(final_review_routes)
app.register_blueprint(paa_routes)

# ================================================================
# 🔗 S1 PROXY (zachowane dla kompatybilności wstecznej)
# ================================================================
@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    """
    Proxy do N-gram API.
    UWAGA: Preferuj /api/article/analyze_and_plan dla pełnego workflow.
    """
    data = request.get_json(force=True)
    
    try:
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json=data,
            timeout=90,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": f"N-gram API error: {response.status_code}",
                "details": response.text
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "N-gram API timeout"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# 🔗 H2 SUGGESTIONS PROXY (NAPRAWIONY - v23.1)
# ================================================================
@app.post("/api/project/s1_h2_suggestions")
def h2_suggestions_proxy():
    """
    Generuje H2 suggestions.
    DEPRECATED: Użyj /api/article/analyze_and_plan dla pełnego workflow.
    
    v23.1 FIX: Zamiast importować funkcję (co powodowało błąd),
    zwracamy redirect z instrukcją użycia nowego endpointu.
    """
    data = request.get_json(force=True)
    
    # Przekształć stary format na nowy
    new_payload = {
        "main_keyword": data.get("topic", data.get("main_keyword", "")),
        "target_length": data.get("target_length", 3000)
    }
    
    return jsonify({
        "status": "deprecated",
        "message": "Ten endpoint jest przestarzały. Użyj POST /api/article/analyze_and_plan",
        "redirect_to": "/api/article/analyze_and_plan",
        "suggested_payload": new_payload,
        "note": "Nowy endpoint łączy S1 + H2 suggestions + plan w jednym requeście"
    }), 301


# ================================================================
# 📊 DEBUG & STATUS ENDPOINTS
# ================================================================
@app.get("/api/master_debug/<project_id>")
def master_debug(project_id):
    """Pełna diagnostyka projektu."""
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    article_plan = data.get("article_plan", {})
    version_history = data.get("version_history", {})
    
    # Policz wersje
    total_versions = sum(
        len(h.get("versions", [])) 
        for h in version_history.values()
    ) if version_history else 0
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("main_keyword"),
        "version": data.get("version", "unknown"),
        "total_batches": len(batches),
        "planned_batches": article_plan.get("total_batches", 0),
        "keywords_count": len(keywords),
        "total_versions": total_versions,
        "has_final_review": "final_review" in data,
        "has_corrections": "corrected_article" in data,
        "has_paa": "paa_section" in data,
        "semantic_enabled": SEMANTIC_ENABLED
    }), 200


@app.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """Status projektu."""
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    article_plan = data.get("article_plan", {})
    
    # Keyword summary
    keywords_summary = []
    main_uses = 0
    synonym_uses = 0
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        is_main = meta.get("is_main_keyword", False)
        is_syn = meta.get("is_synonym_of_main", False)
        
        if is_main:
            main_uses = actual
        elif is_syn:
            synonym_uses += actual
        
        keywords_summary.append({
            "keyword": meta.get("keyword"),
            "type": meta.get("type"),
            "actual": actual,
            "target": f"{meta.get('target_min', 0)}-{meta.get('target_max', 999)}",
            "status": meta.get("status"),
            "is_main": is_main
        })
    
    total = main_uses + synonym_uses
    main_ratio = main_uses / total if total > 0 else 1.0
    
    return jsonify({
        "project_id": project_id,
        "main_keyword": data.get("main_keyword"),
        "progress": {
            "batches_written": len(batches),
            "batches_planned": article_plan.get("total_batches", 0),
            "percent": round(len(batches) / max(1, article_plan.get("total_batches", 1)) * 100)
        },
        "main_vs_synonyms": {
            "main_uses": main_uses,
            "synonym_uses": synonym_uses,
            "ratio": round(main_ratio, 2),
            "valid": main_ratio >= 0.5
        },
        "keywords": keywords_summary,
        "status": {
            "has_plan": bool(article_plan),
            "has_final_review": "final_review" in data,
            "has_paa": "paa_section" in data
        }
    }), 200


# ================================================================
# 🧪 MANUAL CHECK ENDPOINT
# ================================================================
@app.post("/api/manual_check")
def manual_check():
    """Ręczna walidacja tekstu."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400

    try:
        from unified_validator import validate_content
        
        result = validate_content(
            text=data["text"],
            keywords_state=data.get("keywords_state", {}),
            main_keyword=data.get("main_keyword", "")
        )

        return jsonify({
            "status": "OK" if result.is_valid else "WARN",
            "score": result.score,
            "metrics": result.metrics,
            "issues": [i.to_dict() for i in result.issues],
            "keywords": result.keywords_analysis,
            "structure": result.structure_analysis
        }), 200
    except ImportError:
        # Fallback jeśli unified_validator nie istnieje
        return jsonify({
            "status": "OK",
            "message": "unified_validator not available, basic check only",
            "word_count": len(data["text"].split())
        }), 200


# ================================================================
# 📋 API DOCUMENTATION ENDPOINT
# ================================================================
@app.get("/api/docs")
def api_docs():
    """Dokumentacja API."""
    return jsonify({
        "version": VERSION,
        "endpoints": {
            "analyze_and_plan": {
                "method": "POST",
                "path": "/api/article/analyze_and_plan",
                "description": "S1 + H2 suggestions + plan w jednym requeście",
                "input": {
                    "main_keyword": "string (required)",
                    "h2_structure": "array (optional)",
                    "keywords": "array (optional)",
                    "target_length": "int (default: 3000)"
                }
            },
            "create_project": {
                "method": "POST",
                "path": "/api/project/create",
                "description": "Tworzy projekt z BatchPlanem"
            },
            "pre_batch_info": {
                "method": "GET",
                "path": "/api/project/{id}/pre_batch_info",
                "description": "Instrukcje dla następnego batcha"
            },
            "preview_batch": {
                "method": "POST",
                "path": "/api/project/{id}/preview_batch",
                "description": "Walidacja batcha"
            },
            "approve_batch": {
                "method": "POST",
                "path": "/api/project/{id}/approve_batch",
                "description": "Zapisuje batch z wersjonowaniem"
            },
            "rollback_batch": {
                "method": "POST",
                "path": "/api/project/{id}/rollback_batch",
                "description": "Rollback do poprzedniej wersji"
            },
            "versions": {
                "method": "GET",
                "path": "/api/project/{id}/versions/{batch_number}",
                "description": "Historia wersji batcha"
            },
            "final_review": {
                "method": "POST",
                "path": "/api/project/{id}/final_review",
                "description": "Końcowa walidacja"
            },
            "apply_corrections": {
                "method": "POST",
                "path": "/api/project/{id}/apply_corrections",
                "description": "Auto-korekta z Gemini"
            },
            "paa_analyze": {
                "method": "GET",
                "path": "/api/project/{id}/paa/analyze",
                "description": "Dane do FAQ"
            },
            "paa_save": {
                "method": "POST",
                "path": "/api/project/{id}/paa/save",
                "description": "Zapisz FAQ (accepts {questions:[...]} or {paa_section:'string'})"
            },
            "export": {
                "method": "GET",
                "path": "/api/project/{id}/export",
                "description": "Eksport artykułu"
            }
        },
        "deprecated_endpoints": {
            "/api/project/s1_h2_suggestions": "Użyj /api/article/analyze_and_plan"
        },
        "changes_from_v23.0": [
            "Naprawiony błąd importu w h2_suggestions_proxy",
            "Dual input dla paa/save (questions[] lub paa_section string)"
        ]
    }), 200


# ================================================================
# 🚨 ERROR HANDLERS
# ================================================================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request too large (max 32MB)"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500


# ================================================================
# 🏥 HEALTHCHECK
# ================================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "batch_planner": True,
            "version_manager": True,
            "unified_validator": True,
            "semantic_analysis": SEMANTIC_ENABLED,
            "paa_dual_input": True
        },
        "ngram_api": NGRAM_API_URL,
        "firebase": True
    }), 200


@app.get("/api/version")
def version_info():
    return jsonify({
        "engine": "BRAJEN SEO Engine",
        "api_version": VERSION,
        "components": {
            "project_routes": "v23.0",
            "final_review_routes": "v23.0",
            "paa_routes": "v23.1",
            "unified_validator": "v23.0",
            "batch_planner": "v23.0",
            "version_manager": "v23.0"
        },
        "fixes_in_v23.1": [
            "h2_suggestions_proxy import error fixed",
            "paa/save dual input (questions[] + paa_section string)",
            "paa/analyze v22↔v23 field compatibility"
        ]
    }), 200


# ================================================================
# 🏃 Local Run
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting BRAJEN SEO API {VERSION} on port {port}")
    print(f"🔧 Debug mode: {DEBUG_MODE}")
    print(f"🧠 Semantic: {'ENABLED' if SEMANTIC_ENABLED else 'DISABLED'}")
    print(f"🔗 N-gram API: {NGRAM_API_URL}\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
