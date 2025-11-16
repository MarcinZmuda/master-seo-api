# ================================================================
# firestore_batch_summary_routes.py — Project Final Summary Layer
# v7.3.0-firestore-continuous-lemma
# ================================================================

from flask import Blueprint, jsonify
from datetime import datetime

batch_summary_bp = Blueprint("firestore_batch_summary_routes", __name__)
db = None


# ================================================================
# 🧠 /api/project/<project_id> [DELETE] — FINAL SUMMARY
# ================================================================
# ❗ POPRAWIONA ŚCIEŻKA — usunięto /api
@batch_summary_bp.route("/project/<project_id>", methods=["DELETE"])
def delete_project_final(project_id):
    """
    Usuwa projekt z Firestore i zwraca końcowe statystyki:
    - liczba batchy
    - liczba fraz UNDER / OVER / LOCKED / OK
    """
    global db
    if not db:
        return jsonify({"error": "Firestore nie jest połączony"}), 503

    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": f"Projekt {project_id} nie istnieje"}), 404

    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = list(doc_ref.collection("batches").stream())

    # 🔥 Poprawione: locked_terms to frazy z flagą "locked"
    under_terms = [k for k, v in keywords_state.items() if v.get("status") == "UNDER"]
    over_terms = [k for k, v in keywords_state.items() if v.get("status") == "OVER"]
    locked_terms = [k for k, v in keywords_state.items() if v.get("locked") is True]
    ok_terms = [k for k, v in keywords_state.items() if v.get("status") == "OK"]

    summary = {
        "topic": data.get("topic"),
        "counting_mode": data.get("counting_mode", "firestore_remote_lemma"),
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "total_batches": len(batches),
        "under_terms_count": len(under_terms),
        "over_terms_count": len(over_terms),
        "locked_terms_count": len(locked_terms),
        "ok_terms_count": len(ok_terms),
        "timestamp": datetime.utcnow().isoformat()
    }

    # ❗Nie usuwamy projektu — archiwizacja
    doc_ref.update({
        "status": "archived",
        "archived_at": datetime.utcnow().isoformat()
    })

    print(f"🧾 Projekt {project_id} zamknięty. UNDER={len(under_terms)}, OVER={len(over_terms)}, LOCKED={len(locked_terms)}")

    return jsonify({"status": "✅ Projekt zakończony", "summary": summary}), 200


# ================================================================
# 🔧 Rejestracja blueprinta
# ================================================================
def register_batch_summary_routes(app, _db=None):
    global db
    db = _db
    app.register_blueprint(batch_summary_bp, url_prefix="/api")
    print("✅ [INIT] firestore_batch_summary_routes zarejestrowany pod prefixem /api (v7.3.0-firestore-continuous-lemma).")
