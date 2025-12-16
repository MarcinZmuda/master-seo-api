import os
import json
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore

# Init Firebase
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError("❌ Missing FIREBASE_CREDS_JSON")

creds_dict = json.loads(FIREBASE_CREDS_JSON)
cred = credentials.Certificate(creds_dict)
firebase_app = initialize_app(cred)
db = firestore.client()

# Init Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
CORS(app)

# Import Routes
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from final_review_routes import final_review_routes

app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)

# S1 Proxy
NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis")

@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    data = request.get_json(force=True)
    try:
        response = requests.post(NGRAM_API_URL, json=data, timeout=90)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": "S1 Proxy Error", "details": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "v22.1-AntiStuffing",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
