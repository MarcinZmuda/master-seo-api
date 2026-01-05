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
VERSION = "v27.0"

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
    NGRAM_BASE_URL = NGRAM_API_URL.replace("/api/ngram_entity_analysis", "")
    NGRAM_ANALYSIS_ENDPOINT = NGRAM_API_URL
    print(f"[MASTER] 🔗 N-gram API URL (full endpoint detected): {NGRAM_ANALYSIS_ENDPOINT}")
else:
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
from export_routes import export_routes  # v23.9: Eksport DOCX/HTML/TXT + Editorial Review

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)
app.register_blueprint(paa_routes)
app.register_blueprint(export_routes)  # v23.9: Eksport + Editorial Review

# ================================================================
# 🔗 S1 PROXY ENDPOINTS (przekierowanie do N-gram API)
# ================================================================
@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    """
    Proxy endpoint dla S1 analysis.
    v27.0: UTF-8 encoding fix + keyword normalization + recommended_length
    """
    data = request.get_json(force=True)
    
    # v27.0: Normalizuj nazwę parametru - N-gram API oczekuje "main_keyword"
    if "keyword" in data and "main_keyword" not in data:
        data["main_keyword"] = data["keyword"]
    if "main_keyword" not in data:
        return jsonify({"error": "Required: keyword or main_keyword"}), 400
    
    # v23.9: Domyślnie 6 stron zamiast 30
    if "max_urls" not in data:
        data["max_urls"] = 6
    if "top_results" not in data:
        data["top_results"] = 6
    
    keyword = data.get("main_keyword", "")
    print(f"[S1_PROXY] 📡 Forwarding S1 analysis for '{keyword}' to {NGRAM_ANALYSIS_ENDPOINT}")
    print(f"[S1_PROXY] 📦 Request body: main_keyword='{keyword}', keys={list(data.keys())}")
    
    try:
        # v27.0: Explicit UTF-8 encoding
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json=data,
            timeout=90,
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            }
        )
        
        print(f"[S1_PROXY] 📬 Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "No error message"
            print(f"[S1_PROXY] ❌ N-gram API error {response.status_code}: {error_text}")
            return jsonify({
                "error": "N-gram API error",
                "status_code": response.status_code,
                "details": error_text,
                "sent_keyword": keyword,
                "sent_keys": list(data.keys())
            }), response.status_code
        
        result = response.json()
        print(f"[S1_PROXY] ✅ S1 analysis completed successfully")
        
        # =============================================================
        # v27.0: AUTOMATYCZNE OBLICZANIE recommended_length
        # =============================================================
        word_counts = []
        
        # Szukaj word_count w danych konkurencji
        serp_data = result.get("serp_analysis", {}) or {}
        competitors = serp_data.get("competitors", []) or result.get("competitors", []) or []
        
        for comp in competitors:
            if isinstance(comp, dict):
                wc = comp.get("word_count") or comp.get("wordCount") or comp.get("content_length", 0)
                if wc and wc > 100:
                    word_counts.append(wc)
        
        # Heurystyka jeśli brak danych word_count
        if not word_counts:
            ngrams_count = len(result.get("ngrams", []) or result.get("hybrid_ngrams", []) or [])
            h2_count = len(serp_data.get("competitor_h2_patterns", []) or [])
            
            if ngrams_count > 50 or h2_count > 15:
                estimated = 4000
            elif ngrams_count > 30 or h2_count > 10:
                estimated = 3000
            elif ngrams_count > 15 or h2_count > 5:
                estimated = 2000
            else:
                estimated = 1500
            
            word_counts = [estimated]
            print(f"[S1_PROXY] ℹ️ No word_count data, estimated: {estimated} (ngrams={ngrams_count}, h2={h2_count})")
        
        # Oblicz statystyki
        word_counts.sort()
        n = len(word_counts)
        
        if n > 0:
            median = word_counts[n // 2] if n % 2 == 1 else (word_counts[n // 2 - 1] + word_counts[n // 2]) // 2
            avg = sum(word_counts) // n
            recommended = int(median * 1.1)
            recommended = round(recommended / 100) * 100
            recommended = max(1000, min(6000, recommended))
        else:
            median = 3000
            avg = 3000
            recommended = 3000
        
        result["recommended_length"] = recommended
        result["length_analysis"] = {
            "word_counts": word_counts,
            "median": median,
            "average": avg,
            "recommended": recommended,
            "analyzed_urls": len(word_counts),
            "note": "Rekomendacja = mediana + 10%, zaokrąglone do 100"
        }
        
        print(f"[S1_PROXY] 📏 Length analysis: median={median}, recommended={recommended} words")
        
        # Semantic analysis
        if SEMANTIC_ENABLED:
            try:
                from seo_optimizer import semantic_keyword_coverage
                
                if "keywords" in result:
                    sample_text = ""
                    ft = result.get("full_text_sample") or result.get("full_text_content") or ""
                    if isinstance(ft, str) and ft.strip():
                        sample_text = ft
                    else:
                        parts = []
                        fs = serp_data.get("featured_snippet")
                        if isinstance(fs, dict):
                            for k in ("snippet", "text", "answer"):
                                v = fs.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                        elif isinstance(fs, str) and fs.strip():
                            parts.append(fs.strip())
                        
                        paa = serp_data.get("paa_questions", [])
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
                        
                        sample_text = "\n".join(parts)
                    
                    sample_text = (sample_text or "")[:5000]
                    
                    if sample_text.strip():
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


# ============================================================================
# 🆕 v23.9: ANALYZE SERP LENGTH - mediana długości z konkurencji
# ============================================================================
@app.post("/api/analyze_serp_length")
def analyze_serp_length():
    """
    Analizuje długość artykułów konkurencji i zwraca rekomendowaną długość.
    
    Zamiast stałych 3000 słów - pobiera medianę z konkurencji.
    """
    data = request.get_json(force=True)
    keyword = data.get("keyword") or data.get("main_keyword", "")
    
    if not keyword:
        return jsonify({"error": "Missing keyword"}), 400
    
    try:
        # Pobierz analizę SERP (max 6 stron)
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json={"keyword": keyword, "max_urls": 6, "top_results": 6},
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            return jsonify({
                "keyword": keyword,
                "recommended_length": 3000,
                "source": "default",
                "message": "Could not analyze SERP, using default 3000 words"
            }), 200
        
        result = response.json()
        
        # Szukaj word_count w danych konkurencji
        word_counts = []
        
        serp_data = result.get("serp_analysis", {}) or {}
        competitors = serp_data.get("competitors", []) or result.get("competitors", []) or []
        
        for comp in competitors:
            if isinstance(comp, dict):
                wc = comp.get("word_count") or comp.get("wordCount") or comp.get("content_length", 0)
                if wc and wc > 100:
                    word_counts.append(wc)
        
        # Heurystyka jeśli brak danych
        if not word_counts:
            ngrams_count = len(result.get("ngrams", []) or result.get("hybrid_ngrams", []) or [])
            estimated = max(2000, min(5000, 2000 + ngrams_count * 20))
            word_counts = [estimated]
        
        # Oblicz statystyki
        word_counts.sort()
        n = len(word_counts)
        
        if n == 0:
            median = 3000
        elif n % 2 == 0:
            median = (word_counts[n//2 - 1] + word_counts[n//2]) // 2
        else:
            median = word_counts[n//2]
        
        avg = sum(word_counts) // n if n > 0 else 3000
        min_wc = min(word_counts) if word_counts else 2000
        max_wc = max(word_counts) if word_counts else 4000
        
        # Rekomendacja: mediana + 10%
        recommended = int(median * 1.1)
        recommended = max(1500, min(6000, recommended))
        
        return jsonify({
            "keyword": keyword,
            "analyzed_competitors": n,
            "word_counts": word_counts,
            "statistics": {
                "median": median,
                "average": avg,
                "min": min_wc,
                "max": max_wc
            },
            "recommended_length": recommended,
            "source": "serp_analysis",
            "note": f"Rekomendowana długość {recommended} słów (mediana konkurencji + 10%)"
        }), 200
        
    except Exception as e:
        print(f"[SERP_LENGTH] ❌ Error: {e}")
        return jsonify({
            "keyword": keyword,
            "recommended_length": 3000,
            "source": "fallback",
            "error": str(e)
        }), 200


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
                "semantic_enabled": SEMANTIC_ENABLED
            }), 200
        else:
            return jsonify({
                "status": "degraded",
                "ngram_api_status": "error",
                "ngram_base_url": NGRAM_BASE_URL,
                "proxy_enabled": True,
                "semantic_enabled": SEMANTIC_ENABLED
            }), 200
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e),
            "ngram_base_url": NGRAM_BASE_URL,
            "proxy_enabled": True,
            "semantic_enabled": SEMANTIC_ENABLED
        }), 503


# ================================================================
# 🧠 MASTER DEBUG ROUTES (diagnostyka)
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
        "semantic_enabled": SEMANTIC_ENABLED
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
            "paa_routes",
            "export_routes",
            "seo_optimizer",
            "s1_proxy (to N-gram API)"
        ],
        "debug_mode": DEBUG_MODE,
        "firebase_connected": True,
        "ngram_base_url": NGRAM_BASE_URL,
        "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
        "s1_proxy_enabled": True,
        "semantic_enabled": SEMANTIC_ENABLED,
        "features_v23_9": [
            "approve_batch: minimal response (~500B)",
            "pre_batch_info: BASIC full, EXTENDED names only",
            "paa/analyze: ALL unused keywords",
            "analyze_serp_length: dynamic article length",
            "editorial_review: Gemini as editor",
            "export: DOCX/HTML/TXT direct download"
        ]
    }), 200


# ================================================================
# 🔎 VERSION CHECK
# ================================================================
@app.get("/api/version")
def version_info():
    return jsonify({
        "engine": "BRAJEN SEO Engine",
        "api_version": VERSION,
        "components": {
            "project_routes": "v23.9-optimized",
            "firestore_tracker_routes": "v23.9-minimal-response",
            "paa_routes": "v23.9-all-unused",
            "export_routes": "v23.9-editorial-review",
            "seo_optimizer": "v23.9",
            "final_review_routes": "v23.9",
            "s1_proxy": "v23.9 (6 URLs default)"
        },
        "optimizations": {
            "approve_batch_response": "~500B (was 220KB)",
            "pre_batch_info_response": "~5-10KB (was 30-60KB)",
            "s1_default_urls": 6
        },
        "environment": {
            "debug_mode": DEBUG_MODE,
            "firebase_connected": True,
            "ngram_base_url": NGRAM_BASE_URL,
            "semantic_enabled": SEMANTIC_ENABLED
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
        "semantic_coverage": result.get("semantic_coverage", {})
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
    print(f"🧠 Semantic analysis: {'ENABLED ✅' if SEMANTIC_ENABLED else 'DISABLED ⚠️'}")
    print(f"📦 Features v23.9:")
    print(f"   - approve_batch: minimal response (~500B)")
    print(f"   - pre_batch_info: BASIC full, EXTENDED names")
    print(f"   - analyze_serp_length: dynamic article length")
    print(f"   - editorial_review: Gemini as editor\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
