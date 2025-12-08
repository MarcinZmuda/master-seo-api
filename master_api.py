import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# ================================================================
# 🔥 Firestore Initialization — kompatybilne z Render
# ================================================================
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError(
        "❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON — "
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
VERSION = "v19.1-hotfix-large-payloads"

# ================================================================
# 📦 Import blueprintów (po inicjalizacji Firestore)
# ================================================================
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from seo_optimizer import unified_prevalidation
from final_review_routes import final_review_routes

# 🔄 Nowy zintegrowany moduł S1 (ngram_entity_analysis)
try:
    from api.index import app as s1_app
    print("[MASTER] ✅ Zarejestrowano nowy moduł S1: api/index.py")
except ImportError:
    print("[MASTER] ⚠️ Nie znaleziono modułu api/index.py — sprawdź ścieżkę")

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)

if "s1_app" in locals():
    app.register_blueprint(s1_app, url_prefix="/api")

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
# 🔍 HEALTHCHECK
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
            "api/index (S1 consolidated)",
            "seo_optimizer",
            "final_review_routes"
        ],
        "debug_mode": DEBUG_MODE,
        "firebase_connected": True
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
            "project_routes": "v19.1",
            "firestore_tracker_routes": "v19.1",
            "seo_optimizer": "v19.1-safe-gemini",
            "api/index": "v19.1-semantic-firestore",
            "final_review_routes": "v19.1-gemini"
        },
        "environment": {
            "debug_mode": DEBUG_MODE,
            "firebase_connected": True
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
        "warnings": result["warnings"]
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
# 🏁 Local Run
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Master SEO API {VERSION} on port {port}")
    print(f"🔧 Debug mode: {DEBUG_MODE}\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
