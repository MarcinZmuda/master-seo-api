import os
import json

from flask import Flask, jsonify
from flask_cors import CORS

from firebase_admin import credentials, initialize_app, firestore

# ---------------------------------------------------------
# 🔥 Firestore initialization — kompatybilne z Render
# ---------------------------------------------------------
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")

if not FIREBASE_CREDS_JSON:
    raise RuntimeError(
        "Brak zmiennej środowiskowej FIREBASE_CREDS_JSON — "
        "wgraj JSON z Service Account jako string do ENV."
    )

try:
    creds_dict = json.loads(FIREBASE_CREDS_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Niepoprawny JSON w FIREBASE_CREDS_JSON: {e}")

cred = credentials.Certificate(creds_dict)
firebase_app = initialize_app(cred)
db = firestore.client()

# ---------------------------------------------------------
# 🔥 Flask App Initialization
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 🔥 Import blueprintów (po inicjalizacji Firestore)
# ---------------------------------------------------------
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from s1_analysis_routes import s1_routes  # Turbo S1 (SerpAPI + n-gramy)

# ---------------------------------------------------------
# 🔥 Rejestracja blueprintów
# ---------------------------------------------------------
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(s1_routes)

# ---------------------------------------------------------
# 🔍 Healthcheck
# ---------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Master SEO API 7.5.0-hybrid-fuzzy-polars-lt działa — Firestore OK, Hybrid Row-Level + Language QA ON"
    }), 200


# ---------------------------------------------------------
# 🏁 Local Run (Render używa Gunicorna)
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
