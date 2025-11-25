import os
import json
from flask import Flask, jsonify
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore

# ---------------------------------------------------------
# 🔥 Firestore initialization — Compatible with Render
# ---------------------------------------------------------
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")

if not FIREBASE_CREDS_JSON:
    raise RuntimeError("Brak FIREBASE_CREDS_JSON — nie mogę połączyć z Firestore!")

# Render przechowuje Service Account jako string → trzeba zdekodować
creds_dict = json.loads(FIREBASE_CREDS_JSON)
cred = credentials.Certificate(creds_dict)

firebase_app = initialize_app(cred)
db = firestore.client()

# ---------------------------------------------------------
# 🔥 Flask App Initialization
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 🔥 Importujemy blueprinty (muszą być po inicjalizacji Firestore)
# ---------------------------------------------------------
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from s1_analysis_routes import s1_routes   # 🔥 Turbo S1

# ---------------------------------------------------------
# 🔥 Rejestracja blueprintów
# ---------------------------------------------------------
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(s1_routes)

# ---------------------------------------------------------
# 🔥 Healthcheck
# ---------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Master SEO API 7.5.0-hybrid-fuzzy-polars-lt działa — Firestore OK, Hybrid Row-Level + Language QA ON"
    }), 200

# ---------------------------------------------------------
# 🔥 Local Run (Render uses Gunicorn)
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
