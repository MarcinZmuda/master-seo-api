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
        "Firebase jest wymagany zawsze."
    )

cred_data = json.loads(FIREBASE_CREDS_JSON)
cred = credentials.Certificate(cred_data)
initialize_app(cred)
db = firestore.client()
print("✅ Firebase connected successfully (Render compatible)")

# ================================================================
# 🌍 N-gram API Service Configuration
# ================================================================
NGRAM_BASE_URL = os.getenv("NGRAM_API_URL", "http://localhost:5000")
NGRAM_ANALYSIS_ENDPOINT = f"{NGRAM_BASE_URL}/api/ngram_entity_analysis"
SEMANTIC_ENABLED = os.getenv("SEMANTIC_ENABLED", "true").lower() == "true"

# ================================================================
# 🧠 Flask App
# ================================================================
app = Flask(__name__)
CORS(app)

# ================================================================
# 🧩 Healthcheck
# ================================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "seo-master-api",
        "time": datetime.utcnow().isoformat() + "Z"
    }), 200

# ================================================================
# ✅ PROJECT ROUTES (importowane moduły)
# ================================================================
try:
    from project_routes import project_routes
    app.register_blueprint(project_routes)
except Exception as e:
    print(f"⚠️ project_routes import failed: {e}")

try:
    from firestore_tracker_routes import firestore_tracker_routes
    app.register_blueprint(firestore_tracker_routes)
except Exception as e:
    print(f"⚠️ firestore_tracker_routes import failed: {e}")

try:
    from final_review_routes import final_review_routes
    app.register_blueprint(final_review_routes)
except Exception as e:
    print(f"⚠️ final_review_routes import failed: {e}")

# ================================================================
# 🛰️ S1 PROXY: /api/s1_analysis → N-gram service
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
            # UWAGA: N-gram backend już liczy semantykę na pełnych treściach (full_text_sample),
            # natomiast tutaj liczymy „lekki” coverage tylko na podstawie pól zwrotnych.
            if SEMANTIC_ENABLED:
                try:
                    from seo_optimizer import semantic_keyword_coverage

                    # Jeśli S1 zwrócił jakieś keywords
                    if "keywords" in result:
                        # Zbuduj próbkę tekstu z pól, które realnie wracają z N-gram API.
                        serp_analysis = result.get("serp_analysis", {}) if isinstance(result, dict) else {}
                        parts = []
                        # Featured snippet (jeśli dostępny)
                        fs = serp_analysis.get("featured_snippet")
                        if isinstance(fs, dict):
                            for k in ("snippet", "text", "answer"):
                                v = fs.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                        elif isinstance(fs, str) and fs.strip():
                            parts.append(fs.strip())

                        # PAA (People Also Ask)
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

                        # Snippety konkurencji
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

                        sample_text = "\n".join(parts)[:5000]
                        # Jeśli nadal pusto, nie próbuj liczyć semantyki (unikamy „cichej degradacji”).
                        if not sample_text.strip():
                            raise ValueError("No usable text fields in S1 response for semantic analysis.")

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

    # Normalizacja: n-gram backend może zwracać listę dictów (np. {"ngram": "...", ...})
    # a synthesize_topics oczekuje listy stringów.
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
