import os
import json
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# ================================================================
# 🔥 Firestore Initialization – kompatybilne z Render
# ================================================================
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError(
        "❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON – "
        "wgraj JSON z Service Account jako string do ENV."
    )

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

# 🔧 FIX: Zwiększenie limitu payloadu do 32MB (dla dużych analiz SERP/S1)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

CORS(app)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERSION = "v19.6-semantic"

# ================================================================
# 🧠 Check if semantic analysis is available
# ================================================================
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_ENABLED = True
    print("[MASTER] ✅ Semantic analysis available")
except ImportError:
    SEMANTIC_ENABLED = False
    print("[MASTER] ⚠️ Semantic analysis NOT available (sentence-transformers not installed)")

# ================================================================
# 🔗 N-gram API Configuration (for S1 proxy)
# ================================================================
NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com")

# Sprawdź czy URL już zawiera endpoint
if "/api/ngram_entity_analysis" in NGRAM_API_URL:
    # URL już ma endpoint - użyj go bezpośrednio
    NGRAM_BASE_URL = NGRAM_API_URL.replace("/api/ngram_entity_analysis", "")
    NGRAM_ANALYSIS_ENDPOINT = NGRAM_API_URL
    print(f"[MASTER] 🔗 N-gram API URL (full endpoint detected): {NGRAM_ANALYSIS_ENDPOINT}")
else:
    # URL to tylko base - dodaj endpoint
    NGRAM_BASE_URL = NGRAM_API_URL
    NGRAM_ANALYSIS_ENDPOINT = f"{NGRAM_API_URL}/api/ngram_entity_analysis"
    print(f"[MASTER] 🔗 N-gram API URL (base URL): {NGRAM_BASE_URL}")

print(f"[MASTER] 🎯 S1 Analysis endpoint: {NGRAM_ANALYSIS_ENDPOINT}")

# ================================================================
# 📦 Import blueprintów (po inicjalizacji Firestore)
# ================================================================
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from seo_optimizer import unified_prevalidation
from final_review_routes import final_review_routes

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)

# ================================================================
# 🔗 S1 PROXY ENDPOINTS (przekierowanie do N-gram API)
# ================================================================
@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    """
    Proxy endpoint dla S1 analysis.
    Przekierowuje request do N-gram API service.
    """
    data = request.get_json(force=True)
    
    print(f"[S1_PROXY] 📡 Forwarding S1 analysis to {NGRAM_ANALYSIS_ENDPOINT}")
    
    try:
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json=data,
            timeout=90,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[S1_PROXY] ✅ S1 analysis completed successfully")
            
            # ⭐ NOWE: Dodaj semantic analysis do S1 wyniku
            if SEMANTIC_ENABLED:
                try:
                    from seo_optimizer import semantic_keyword_coverage
                    
                    # Jeśli S1 zwrócił jakieś keywords
                    if "keywords" in result:
                        # Przykładowy tekst z SERP (pierwszych 5000 znaków)
                        sample_text = result.get("serp_content", "")[:5000]
                        
                        # Dummy keywords_state dla semantic analysis
                        dummy_kw_state = {
                            str(i): {"keyword": kw, "actual_uses": 0}
                            for i, kw in enumerate(result.get("keywords", []))
                        }
                        
                        semantic_cov = semantic_keyword_coverage(sample_text, dummy_kw_state)
                        result["semantic_analysis"] = semantic_cov
                        print(f"[S1_PROXY] ✅ Added semantic analysis to S1 result")
                except Exception as e:
                    print(f"[S1_PROXY] ⚠️ Semantic analysis failed: {e}")
            
            return jsonify(result), 200
        else:
            print(f"[S1_PROXY] ❌ N-gram API error: {response.status_code}")
            return jsonify({
                "error": "N-gram API error",
                "status_code": response.status_code,
                "details": response.text[:500]
            }), response.status_code
            
    except requests.exceptions.Timeout:
        print(f"[S1_PROXY] ⏱️ Timeout after 90s")
        return jsonify({
            "error": "N-gram API timeout",
            "message": "SERP analysis took too long (>90s). Try with fewer sources."
        }), 504
        
    except requests.exceptions.ConnectionError:
        print(f"[S1_PROXY] ❌ Connection error to {NGRAM_ANALYSIS_ENDPOINT}")
        return jsonify({
            "error": "Cannot connect to N-gram API",
            "ngram_api_url": NGRAM_ANALYSIS_ENDPOINT,
            "message": "Check if N-gram API service is running"
        }), 503
        
    except Exception as e:
        print(f"[S1_PROXY] ❌ Unexpected error: {e}")
        return jsonify({
            "error": "S1 proxy error",
            "message": str(e)
        }), 500


@app.post("/api/synthesize_topics")
def synthesize_topics_proxy():
    """Proxy dla synthesize_topics."""
    data = request.get_json(force=True)
    
    try:
        response = requests.post(
            f"{NGRAM_BASE_URL}/api/synthesize_topics",
            json=data,
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/generate_compliance_report")
def compliance_report_proxy():
    """Proxy dla generate_compliance_report."""
    data = request.get_json(force=True)
    
    try:
        response = requests.post(
            f"{NGRAM_BASE_URL}/api/generate_compliance_report",
            json=data,
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/s1_health")
def s1_health_check():
    """Sprawdza czy N-gram API service jest dostępny."""
    try:
        response = requests.get(f"{NGRAM_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            ngram_status = response.json()
            return jsonify({
                "status": "ok",
                "ngram_api_status": ngram_status,
                "ngram_base_url": NGRAM_BASE_URL,
                "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
                "proxy_enabled": True,
                "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
            }), 200
        else:
            return jsonify({
                "status": "degraded",
                "ngram_api_status": "error",
                "ngram_base_url": NGRAM_BASE_URL,
                "proxy_enabled": True,
                "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
            }), 200
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e),
            "ngram_base_url": NGRAM_BASE_URL,
            "proxy_enabled": True,
            "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
        }), 503

# ================================================================
# 🧠 MASTER DEBUG ROUTES (diagnostyka)
# ================================================================
@app.get("/api/master_debug/<project_id>")
def master_debug(project_id):
    """Pełna diagnostyka projektu: frazy, batch count, semantyka, ostrzeżenia."""
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    total_batches = len(batches)

    all_warnings = []
    for b in batches:
        warns = b.get("warnings", [])
        if warns:
            all_warnings.extend(warns)

    semantic_scores = [
        b.get("language_audit", {}).get("semantic_score")
        for b in batches if b.get("language_audit")
    ]
    avg_semantic = (
        round(sum([s for s in semantic_scores if s]) / len(semantic_scores), 3)
        if semantic_scores else 0
    )

    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": total_batches,
        "keywords_count": len(keywords),
        "warnings_total": len(all_warnings),
        "avg_semantic_score": avg_semantic,
        "avg_density": round(
            sum([b.get("language_audit", {}).get("density", 0) for b in batches]) / max(1, total_batches), 2
        ),
        "burstiness_avg": round(
            sum([b.get("burstiness", 0) for b in batches]) / max(1, total_batches), 2
        ),
        "last_update": batches[-1]["timestamp"].isoformat() if batches else None,
        "lsi_keywords": data.get("lsi_enrichment", {}).get("count", 0),
        "has_final_review": "final_review" in data,
        "
