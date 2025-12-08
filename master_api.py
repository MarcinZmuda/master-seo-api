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
CORS(app)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERSION = "v18.0-brajen-semantic"

# ================================================================
# 📦 Import blueprintów (po inicjalizacji Firestore)
# ================================================================
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from s1_analysis_routes import s1_routes
from seo_optimizer import unified_prevalidation  # ✅ globalne wywołania

# Rejestracja blueprintów
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(s1_routes)

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

    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": total_batches,
        "keywords_count": len(keywords),
        "warnings_total": len(all_warnings),
        "semantic_scores": [b.get("language_audit", {}).get("semantic_score") for b in batches if b.get("language_audit")],
        "avg_density": round(sum([b.get("language_audit", {}).get("density", 0) for b in batches]) / max(1, total_batches), 2),
        "burstiness_avg": round(sum([b.get("burstiness", 0) for b in batches]) / max(1, total_batches), 2),
        "last_update": batches[-1]["timestamp"].isoformat() if batches else None
    }), 200


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
        "modules": ["project_routes", "firestore_tracker_routes", "s1_analysis_routes", "seo_optimizer"],
        "debug_mode": DEBUG_MODE
    }), 200


# ================================================================
# 🔎 VERSION CHECK
# ================================================================
@app.get("/api/version")
def version_info():
    """Zwraca aktualną wersję wszystkich komponentów."""
    return jsonify({
        "engine": "Brajen Semantic Engine",
        "api_version": VERSION,
        "components": {
            "project_routes": "v18.0",
            "firestore_tracker_routes": "v18.0",
            "seo_optimizer": "v18.0",
            "s1_analysis_routes": "v17.x-compatible"
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
    """Szybki endpoint do testowania unified_prevalidation z dowolnym tekstem."""
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
# 🏁 Local Run (Render używa Gunicorna)
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Master SEO API {VERSION} on port {port}")
    print(f"🔧 Debug mode: {DEBUG_MODE}\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
