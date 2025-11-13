# ================================================================
# firestore_batch_summary_routes.py — Batch Aggregator (v7.2.8-firestore-summarizer)
# ================================================================

from flask import Blueprint, request, jsonify
from datetime import datetime

batch_summary_bp = Blueprint("firestore_batch_summary_routes", __name__)
db = None


# ---------------------------------------------------------------
# 🧩 Funkcja pomocnicza: agregacja raportów partów
# ---------------------------------------------------------------
def aggregate_batch_parts(parts_snapshots):
    """
    Łączy raporty z poszczególnych części batcha w jeden scalony raport.
    """
    aggregated = {
        "under": 0,
        "over": 0,
        "locked": 0,
        "ok": 0,
        "updated_keywords": 0
    }
    combined_keywords = {}

    for part_doc in parts_snapshots:
        part_data = part_doc.to_dict() or {}
        keywords_report = part_data.get("keywords_report", [])

        for item in keywords_report:
            kw = item["keyword"]
            combined_keywords.setdefault(kw, {
                "actual_uses": 0,
                "target_range": item["target_range"],
                "status": item["status"]
            })
            combined_keywords[kw]["actual_uses"] += item["actual_uses"]

        # próbujemy też zsumować summary (jeśli jest)
        summary = part_data.get("summary", {})
        for key in ["under", "over", "locked", "ok", "updated_keywords"]:
            aggregated[key] += summary.get(key, 0)

    # zliczamy ostateczne statusy
    for kw, data in combined_keywords.items():
        if data["status"] == "UNDER":
            aggregated["under"] += 1
        elif data["status"] == "OVER":
            aggregated["over"] += 1
        elif data["status"] == "OK":
            aggregated["ok"] += 1
        else:
            aggregated["locked"] += 1

    return aggregated, list(combined_keywords.values())


# ---------------------------------------------------------------
# ✅ /api/project/<project_id>/summarize_batches
# ---------------------------------------------------------------
@batch_summary_bp.route("/api/project/<project_id>/summarize_batches", methods=["POST"])
def summarize_batches(project_id):
    """
    Agreguje wszystkie części (parts) dla najnowszego batcha i tworzy
    scalony raport meta_prompt_summary w dokumencie Firestore.
    """
    global db
    if not db:
        return jsonify({"error": "Firestore nie jest połączony"}), 503

    try:
        data = request.get_json(silent=True) or {}
        batch_id = data.get("batch_id")

        if not batch_id:
            return jsonify({"error": "Brak batch_id"}), 400

        print(f"📊 Agreguję batch {batch_id} projektu {project_id}")

        batch_ref = (
            db.collection("seo_projects")
            .document(project_id)
            .collection("batches")
            .document(batch_id)
        )
        parts_ref = batch_ref.collection("parts").stream()
        parts = list(parts_ref)

        if not parts:
            return jsonify({"error": "Brak części batcha do agregacji"}), 404

        aggregated_summary, combined_keywords = aggregate_batch_parts(parts)

        # zapis scalonego raportu
        batch_ref.update({
            "summary_full": {
                **aggregated_summary,
                "parts_count": len(parts),
                "aggregated_at": datetime.utcnow().isoformat()
            },
            "keywords_combined": combined_keywords
        })

        print(f"✅ Batch {batch_id} scalony ({len(parts)} parts).")

        return jsonify({
            "status": "✅ Batch zsumowany",
            "project_id": project_id,
            "batch_id": batch_id,
            "summary": aggregated_summary,
            "total_parts": len(parts)
        }), 200

    except Exception as e:
        print(f"❌ Błąd podczas agregacji batchy: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_batch_summary_routes(app, _db=None):
    """Rejestruje blueprint firestore_batch_summary_routes."""
    global db
    db = _db
    app.register_blueprint(batch_summary_bp, url_prefix="/api")  # ✅ poprawny blueprint i prefix
    print("✅ [INIT] firestore_batch_summary_routes zarejestrowany pod prefixem /api (v7.2.8-firestore-summarizer).")
