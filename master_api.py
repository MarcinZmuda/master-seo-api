import os
import json
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# ================================================================
# 🔥 Firestore Initialization
# ================================================================
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError("❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON")

try:
    creds_dict = json.loads(FIREBASE_CREDS_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Niepoprawny JSON w FIREBASE_CREDS_JSON: {e}")

cred = credentials.Certificate(creds_dict)
firebase_app = initialize_app(cred)
db = firestore.client()

# ================================================================
# ⚙️ Flask App Initialization
# ================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
CORS(app)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERSION = "v32.0"
PROJECTS = {}

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

try:
    from unified_validator import (
        validate_content, validate_semantic_enhancement,
        full_validate_complete, quick_validate, calculate_entity_density
    )
    SEMANTIC_ENHANCEMENT_ENABLED = True
    print("[MASTER] ✅ Semantic Enhancement v31.0 loaded")
except ImportError as e:
    SEMANTIC_ENHANCEMENT_ENABLED = False
    print(f"[MASTER] ⚠️ Semantic Enhancement not available: {e}")

# 🆕 v32.0: AI Detection
try:
    from ai_detection_metrics import validate_ai_detection, create_ai_detection_response
    AI_DETECTION_ENABLED = True
    print("[MASTER] ✅ AI Detection Metrics v1.0 loaded")
except ImportError as e:
    AI_DETECTION_ENABLED = False
    print(f"[MASTER] ⚠️ AI Detection not available: {e}")

# ================================================================
# 🔗 N-gram API Configuration
# ================================================================
NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com")

if "/api/ngram_entity_analysis" in NGRAM_API_URL:
    NGRAM_BASE_URL = NGRAM_API_URL.replace("/api/ngram_entity_analysis", "")
    NGRAM_ANALYSIS_ENDPOINT = NGRAM_API_URL
else:
    NGRAM_BASE_URL = NGRAM_API_URL
    NGRAM_ANALYSIS_ENDPOINT = f"{NGRAM_API_URL}/api/ngram_entity_analysis"

print(f"[MASTER] 🎯 S1 Analysis endpoint: {NGRAM_ANALYSIS_ENDPOINT}")

# ================================================================
# 📦 Import blueprintów
# ================================================================
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from seo_optimizer import unified_prevalidation
from final_review_routes import final_review_routes
from paa_routes import paa_routes
from export_routes import export_routes

try:
    from entity_routes import entity_routes
    ENTITY_ROUTES_ENABLED = True
except ImportError:
    ENTITY_ROUTES_ENABLED = False
    entity_routes = None

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)
app.register_blueprint(paa_routes)
app.register_blueprint(export_routes)

if ENTITY_ROUTES_ENABLED and entity_routes:
    app.register_blueprint(entity_routes)
    # ================================================================
# 🆕 v32.0: AI DETECTION ENDPOINTS
# ================================================================
@app.post("/api/ai_detection")
def ai_detection_endpoint():
    """🆕 v32.0: Sprawdza tekst pod kątem wykrywalności przez detektory AI."""
    if not AI_DETECTION_ENABLED:
        return jsonify({"status": "ERROR", "message": "AI Detection not available. Install: pip install wordfreq"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    result, status_code = create_ai_detection_response(text)
    return jsonify(result), status_code


@app.post("/api/quick_ai_check")
def quick_ai_check_endpoint():
    """🆕 v32.0: Szybki check AI - tylko score i status."""
    if not AI_DETECTION_ENABLED:
        return jsonify({"score": 70, "status": "UNAVAILABLE"}), 200
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 200:
        return jsonify({"score": 70, "status": "INSUFFICIENT_DATA"}), 200
    
    try:
        result = validate_ai_detection(text)
        return jsonify({
            "score": result["humanness_score"],
            "status": result["status"],
            "warnings_count": len(result["warnings"]),
            "top_warning": result["warnings"][0] if result["warnings"] else None,
            "burstiness": result["components"]["burstiness"]["value"]
        }), 200
    except Exception as e:
        return jsonify({"score": 70, "status": "ERROR", "message": str(e)}), 200


# ================================================================
# 🔗 S1 PROXY ENDPOINT
# ================================================================
@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    """Proxy endpoint dla S1 analysis."""
    data = request.get_json(force=True)
    
    if "keyword" in data and "main_keyword" not in data:
        data["main_keyword"] = data["keyword"]
    if "main_keyword" not in data:
        return jsonify({"error": "Required: keyword or main_keyword"}), 400
    
    if "max_urls" not in data:
        data["max_urls"] = 6
    if "top_results" not in data:
        data["top_results"] = 6
    
    keyword = data.get("main_keyword", "")
    print(f"[S1_PROXY] 📡 Forwarding S1 analysis for '{keyword}'")
    
    try:
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT, json=data, timeout=90,
            headers={'Content-Type': 'application/json; charset=utf-8', 'Accept': 'application/json'}
        )
        
        if response.status_code != 200:
            return jsonify({"error": "N-gram API error", "status_code": response.status_code}), response.status_code
        
        result = response.json()
        
        # Length analysis
        word_counts = []
        serp_data = result.get("serp_analysis", {}) or {}
        competitors = serp_data.get("competitors", []) or result.get("competitors", []) or []
        
        for comp in competitors:
            if isinstance(comp, dict):
                wc = comp.get("word_count") or comp.get("wordCount") or comp.get("content_length", 0)
                if wc and wc > 100:
                    word_counts.append(wc)
        
        if not word_counts:
            ngrams_count = len(result.get("ngrams", []) or [])
            estimated = 2000 if ngrams_count < 30 else 3000 if ngrams_count < 50 else 4000
            word_counts = [estimated]
        
        word_counts.sort()
        n = len(word_counts)
        median = word_counts[n // 2] if n % 2 == 1 else (word_counts[n // 2 - 1] + word_counts[n // 2]) // 2
        recommended = max(1000, min(6000, round(int(median * 1.1) / 100) * 100))
        
        result["recommended_length"] = recommended
        result["length_analysis"] = {"median": median, "recommended": recommended, "analyzed_urls": len(word_counts)}
        
        return jsonify(result), 200
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "N-gram API timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to N-gram API"}), 503
    except Exception as e:
        return jsonify({"error": "S1 proxy error", "message": str(e)}), 500


# ================================================================
# 🆕 v31.0: SEMANTIC VALIDATION ENDPOINTS
# ================================================================
@app.post("/api/semantic_validate")
def semantic_validate():
    if not SEMANTIC_ENHANCEMENT_ENABLED:
        return jsonify({"error": "Semantic Enhancement not available"}), 503
    
    data = request.get_json(force=True)
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    try:
        result = validate_semantic_enhancement(content=data["text"], s1_data=data.get("s1_data", {}), detected_entities=data.get("entities", []))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/quick_semantic_check")
def quick_semantic_check():
    if not SEMANTIC_ENHANCEMENT_ENABLED:
        return jsonify({"status": "unavailable"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 100:
        return jsonify({"status": "text_too_short"}), 400
    
    try:
        result = validate_semantic_enhancement(content=text)
        density = result.get("analyses", {}).get("entity_density", {})
        effort = result.get("analyses", {}).get("source_effort", {})
        
        return jsonify({
            "status": result.get("status", "UNKNOWN"),
            "semantic_score": result.get("semantic_score", 0),
            "entity_density_ok": density.get("status") == "GOOD",
            "generics_found": density.get("generics_found", [])[:3],
            "source_effort_score": effort.get("score", 0),
            "quick_wins": result.get("quick_wins", [])[:3]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        # ================================================================
# 🆕 v32.0: APPROVE BATCH (z AI Detection)
# ================================================================
@app.post("/api/approveBatch")
def approve_batch():
    """🆕 v32.0: Zatwierdzenie batcha z walidacją + AI Detection."""
    data = request.get_json(force=True)
    
    project_id = data.get("project_id", "unknown")
    batch_number = data.get("batch_number", 1)
    batch_content = data.get("batch_content", "")
    accumulated_content = data.get("accumulated_content", batch_content)
    keywords_state = data.get("keywords_state", {})
    s1_data = data.get("s1_data", {})
    total_batches = data.get("total_batches", 7)
    
    # 1. Walidacja fraz
    keyword_warnings = []
    keyword_blockers = []
    basic_kw = keywords_state.get("basic", {})
    extended_kw = keywords_state.get("extended", {})
    
    for phrase, state in basic_kw.items():
        current = state.get("current", 0)
        target_min = state.get("target_min", state.get("target", 1))
        target_max = state.get("target_max", target_min * 4)
        
        if current > target_max:
            keyword_blockers.append({"type": "STUFFING", "phrase": phrase, "current": current, "max": target_max})
        elif current == 0:
            keyword_warnings.append({"type": "MISSING", "phrase": phrase, "target": target_min})
    
    extended_used = sum(1 for state in extended_kw.values() if state.get("current", 0) >= 1)
    extended_total = len(extended_kw)
    
    # 2. Semantic progress
    semantic_progress = {"entity_density_current": 0, "word_count_total": 0, "extended_progress": f"{extended_used}/{extended_total}"}
    
    if SEMANTIC_ENHANCEMENT_ENABLED and accumulated_content:
        try:
            density_result = calculate_entity_density(accumulated_content)
            semantic_progress["entity_density_current"] = density_result.get("density", 0)
            semantic_progress["word_count_total"] = density_result.get("word_count", 0)
        except:
            pass
    
    # 🆕 3. AI Detection
    ai_detection_result = None
    if AI_DETECTION_ENABLED and len(accumulated_content) > 500:
        try:
            ai_result = validate_ai_detection(accumulated_content)
            ai_detection_result = {
                "humanness_score": ai_result["humanness_score"],
                "status": ai_result["status"],
                "burstiness": ai_result["components"]["burstiness"]["value"],
                "warnings": ai_result["warnings"][:3]
            }
        except:
            pass
    
    # 4. Checkpoint alerts
    checkpoint_alerts = []
    if ai_detection_result and ai_detection_result["humanness_score"] < 50:
        checkpoint_alerts.append({
            "checkpoint": "AI_DETECTION",
            "status": "WARNING",
            "message": f"Humanness score {ai_detection_result['humanness_score']}/100 - tekst może być wykryty jako AI"
        })
    
    has_blockers = len(keyword_blockers) > 0
    
    return jsonify({
        "status": "BLOCKED" if has_blockers else "SAVED",
        "project_id": project_id,
        "batch_number": batch_number,
        "batch_total": total_batches,
        "keyword_validation": {"blockers": keyword_blockers, "warnings": keyword_warnings},
        "semantic_progress": semantic_progress,
        "ai_detection": ai_detection_result,
        "checkpoint_alerts": checkpoint_alerts,
        "summary": {"can_continue": not has_blockers, "next_step": f"Batch {batch_number + 1}/{total_batches}" if not has_blockers else "FIX_ISSUES"}
    }), 200


# ================================================================
# SAVE/GET FULL ARTICLE
# ================================================================
@app.post("/api/project/<project_id>/save_full_article")
def save_full_article(project_id):
    data = request.get_json(force=True)
    full_content = data.get("full_content", "")
    
    if len(full_content) < 500:
        return jsonify({"status": "ERROR", "message": "full_content too short"}), 400
    
    if project_id not in PROJECTS:
        PROJECTS[project_id] = {}
    
    PROJECTS[project_id]["full_article"] = {
        "content": full_content,
        "word_count": data.get("word_count", len(full_content.split())),
        "saved_at": datetime.utcnow().isoformat()
    }
    
    return jsonify({"status": "SAVED", "project_id": project_id}), 200


@app.get("/api/project/<project_id>/full_article")
def get_full_article(project_id):
    if project_id not in PROJECTS or "full_article" not in PROJECTS[project_id]:
        return jsonify({"status": "ERROR", "message": "No full article saved"}), 404
    
    return jsonify({"status": "OK", **PROJECTS[project_id]["full_article"]}), 200


# ================================================================
# HEALTH & VERSION
# ================================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "semantic_enabled": SEMANTIC_ENABLED,
        "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED,
        "ai_detection_enabled": AI_DETECTION_ENABLED
    }), 200


@app.get("/api/version")
def version_info():
    return jsonify({
        "engine": "BRAJEN SEO Engine",
        "api_version": VERSION,
        "ai_detection": {"enabled": AI_DETECTION_ENABLED, "features": ["burstiness_v2", "vocabulary_richness", "lexical_sophistication", "starter_entropy"]},
        "semantic_enhancement": {"enabled": SEMANTIC_ENHANCEMENT_ENABLED}
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request Entity Too Large"}), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500


# ================================================================
# 🏃 Local Run
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Master SEO API {VERSION} on port {port}")
    print(f"🆕 AI Detection v32.0: {'ENABLED ✅' if AI_DETECTION_ENABLED else 'DISABLED ⚠️'}")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
```

