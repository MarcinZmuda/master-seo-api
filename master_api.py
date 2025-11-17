import os
from flask import Flask, jsonify
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore

# -----------------------------------------
# 🔥 Inicjalizacja Firestore
# -----------------------------------------
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GOOGLE_APPLICATION_CREDENTIALS:
    raise RuntimeError("Brak zmiennej środowiskowej GOOGLE_APPLICATION_CREDENTIALS!")

cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
firebase_app = initialize_app(cred)
db = firestore.client()

# -----------------------------------------
# 🔥 Inicjalizacja aplikacji Flask
# -----------------------------------------
app = Flask(__name__)
CORS(app)


# -----------------------------------------
# 🔥 Import blueprintów
# -----------------------------------------
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes


# -----------------------------------------
# 🔥 Rejestracja blueprintów
# -----------------------------------------
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)


# -----------------------------------------
# 🔥 Healthcheck
# -----------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Master SEO API 7.3.0 — Firestore Continuous Lemma Running"
    }), 200


# -----------------------------------------
# 🔥 Uruchamianie lokalne
# Render używa Gunicorn, więc tego nie dotyka
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
