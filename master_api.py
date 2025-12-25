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
VERSION = "v22.1-semantic"

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
from paa_routes import paa_routes
from export_routes import export_routes  # v23.8: Eksport PDF/DOCX

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)
app.register_blueprint(paa_routes)
app.register_blueprint(export_routes)  # v23.8: Eksport PDF/DOCX

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
                        # Preferuj próbkę pełnego tekstu, jeśli backend ją zwraca (np. full_text_sample).
                        # W przeciwnym razie buduj próbkę z pól zwrotnych SERP (snippety/PAA/related/H2),
                        # aby uniknąć liczenia coverage na pustym tekście.
                        sample_text = ""
                        if isinstance(result, dict):
                            ft = result.get("full_text_sample") or result.get("full_text_content") or ""
                            if isinstance(ft, str) and ft.strip():
                                sample_text = ft
                            else:
                                serp_analysis = result.get("serp_analysis", {}) or {}
                                parts = []

                                fs = serp_analysis.get("featured_snippet")
                                if isinstance(fs, dict):
                                    for k in ("snippet", "text", "answer"):
                                        v = fs.get(k)
                                        if isinstance(v, str) and v.strip():
                                            parts.append(v.strip())
                                elif isinstance(fs, str) and fs.strip():
                                    parts.append(fs.strip())

                                paa = serp_analysis.get("paa_questions", [])
                                if isinstance(paa, list):
                                    for item in paa:
                                        if isinstance(item, dict):
                                            q = item.get("question") or item.get("q")
                                            a = item.get("answer") or item.get("snippet") or item.get("a")
                                            if isinstance(q, str) and q.strip():
                                                parts.append(q.strip())
                                            if isinstance(a, str) and a.strip():
                                                parts.append(a.strip())
                                        elif isinstance(item, str) and item.strip():
                                            parts.append(item.strip())

                                snips = serp_analysis.get("competitor_snippets", [])
                                if isinstance(snips, list):
                                    for s in snips:
                                        if isinstance(s, str) and s.strip():
                                            parts.append(s.strip())

                                related = serp_analysis.get("related_searches", [])
                                if isinstance(related, list):
                                    for r in related:
                                        if isinstance(r, str) and r.strip():
                                            parts.append(r.strip())

                                h2p = serp_analysis.get("competitor_h2_patterns", [])
                                if isinstance(h2p, list):
                                    for h in h2p:
                                        if isinstance(h, str) and h.strip():
                                            parts.append(h.strip())

                                sample_text = "\n".join(parts)

                        sample_text = (sample_text or "")[:5000]
                        
                        # Dummy keywords_state dla semantic analysis
                        dummy_kw_state = {
                            str(i): {"keyword": kw, "actual_uses": 0}
                            for i, kw in enumerate(result.get("keywords", []))
                        }
                        
                        if not sample_text.strip():
                            raise ValueError("Brak danych tekstowych do semantic coverage (serp_content/full_text_sample/pola serp_analysis puste).")
                        
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

    # Normalizacja: jeśli ngrams to lista dictów (np. {"ngram": "...", ...}),
    # przekształć ją do listy stringów dla kompatybilności z backendem.
    if isinstance(data, dict):
        ngrams = data.get("ngrams")
        if isinstance(ngrams, list) and ngrams and isinstance(ngrams[0], dict):
            data["ngrams"] = [x.get("ngram", "") for x in ngrams if isinstance(x, dict) and x.get("ngram")]
    
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
        "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
    }), 200

# ================================================================
# 🚨 ERROR HANDLERS (Globalne)
# ================================================================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request Entity Too Large", "message": "Payload przekracza 32MB"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500

# ================================================================
# 🏥 HEALTHCHECK
# ================================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Master SEO API działa",
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "modules": [
            "project_routes",
            "firestore_tracker_routes",
            "final_review_routes",
            "seo_optimizer",
            "s1_proxy (to N-gram API)"
        ],
        "debug_mode": DEBUG_MODE,
        "firebase_connected": True,
        "ngram_base_url": NGRAM_BASE_URL,
        "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
        "s1_proxy_enabled": True,
        "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
    }), 200

# ================================================================
# 🔎 VERSION CHECK
# ================================================================
@app.get("/api/version")
def version_info():
    return jsonify({
        "engine": "Brajen Semantic Engine",
        "api_version": VERSION,
        "components": {
            "project_routes": "v22.1-semantic",
            "firestore_tracker_routes": "v22.1-semantic",
            "seo_optimizer": "v22.1-semantic",
            "final_review_routes": "v22.1-gemini-2.5",
            "s1_proxy": "v22.1 (to N-gram API)"
        },
        "environment": {
            "debug_mode": DEBUG_MODE,
            "firebase_connected": True,
            "ngram_base_url": NGRAM_BASE_URL,
            "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
            "semantic_enabled": SEMANTIC_ENABLED  # ⭐ NOWE
        }
    }), 200

# ================================================================
# 🧩 MANUAL CHECK ENDPOINT (test unified_prevalidation)
# ================================================================
@app.post("/api/manual_check")
def manual_check():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400

    dummy_keywords = data.get("keywords_state", {})
    result = unified_prevalidation(data["text"], dummy_keywords)

    return jsonify({
        "status": "CHECK_OK",
        "semantic_score": result["semantic_score"],
        "density": result["density"],
        "smog": result["smog"],
        "readability": result["readability"],
        "warnings": result["warnings"],
        "semantic_coverage": result.get("semantic_coverage", {})  # ⭐ NOWE
    }), 200

# ================================================================
# 🧩 AUTO FINAL REVIEW TRIGGER (po eksporcie)
# ================================================================
@app.post("/api/auto_final_review/<project_id>")
def auto_final_review(project_id):
    from final_review_routes import perform_final_review
    try:
        response = perform_final_review(project_id)
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# 🏃 Local Run
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Master SEO API {VERSION} on port {port}")
    print(f"🔧 Debug mode: {DEBUG_MODE}")
    print(f"🔗 S1 Proxy enabled → {NGRAM_ANALYSIS_ENDPOINT}")
    print(f"🧠 Semantic analysis: {'ENABLED ✅' if SEMANTIC_ENABLED else 'DISABLED ⚠️'}\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
