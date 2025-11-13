# ================================================================
# firestore_tracker_routes.py — Warstwa Batch + Keyword Tracker (v7.2.1-fixed)
# ================================================================

from flask import Blueprint, request, jsonify
from datetime import datetime
import re

tracker_bp = Blueprint("firestore_tracker_routes", __name__)

db = None


def count_keyword_occurrences(text, keyword):
    """Liczy semantyczne wystąpienia frazy w tekście (bez rozróżniania wielkości liter)."""
    pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.UNICODE)
    return len(pattern.findall(text.lower()))


@tracker_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch(project_id):
    """Dodaje batch treści do projektu Firestore i aktualizuje statystyki słów kluczowych."""
    global db
    if not db:
        return jsonify({"error": "Firestore nie jest połączony"}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Brak tekstu do zapisu"}), 400

    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": f"Projekt {project_id} nie istnieje"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})

    # 🔢 Liczenie wystąpień fraz w batchu
    updated_keywords = 0
    keywords_report = []

    for keyword, info in keywords_state.items():
        count = count_keyword_occurrences(text, keyword)
        if count > 0:
            info["actual"] = info.get("actual", 0) + count
            updated_keywords += 1
        # status logic
        if info["actual"] < info["target_min"]:
            status = "UNDER"
        elif info["actual"] > info["target_max"]:
            status = "OVER"
        else:
            status = "OK"
        info["status"] = status
        keywords_report.append({
            "keyword": keyword,
            "actual_uses": info["actual"],
            "target_range": f"{info['target_min']}-{info['target_max']}",
            "status": status
        })
        keywords_state[keyword] = info

    # 🔐 Blokowanie LOCKED
    locked_terms = [k for k, v in keywords_state.items() if v.get("status") == "OVER"]

    # 🔄 Zapis batcha
    existing_batches = project_data.get("batches", [])
    new_batch = {
        "created_at": datetime.utcnow().isoformat(),
        "length": len(text),
        "text": text
    }
    existing_batches.append(new_batch)
    doc_ref.update({
        "batches": existing_batches,
        "keywords_state": keywords_state,
        "status": "updated"
    })

    # 🧠 Meta Prompt Summary (HEAR 2.0 ready)
    meta_prompt_summary = {
        "updated_keywords": updated_keywords,
        "locked_terms_count": len(locked_terms),
        "under_terms_count": len([k for k in keywords_state.values() if k["status"] == "UNDER"]),
        "over_terms_count": len([k for k in keywords_state.values() if k["status"] == "OVER"]),
        "ok_terms_count": len([k for k in keywords_state.values() if k["status"] == "OK"]),
    }

    print(f"[INFO] ✅ Batch dodany do projektu {project_id} ({updated_keywords} fraz zaktualizowano).")

    return jsonify({
        "status": "✅ Batch zapisany i przetworzony",
        "project_id": project_id,
        "batch_length": len(text),
        "keywords_report": keywords_report,
        "locked_terms": locked_terms,
        "updated_keywords": updated_keywords,
        "meta_prompt_summary": meta_prompt_summary
    }), 200


def register_tracker_routes(app, _db=None):
    """Rejestruje blueprint firestore_tracker_routes."""
    global db
    db = _db
    app.register_blueprint(tracker_bp)
    print("✅ [INIT] firestore_tracker_routes zarejestrowany (v7.2.1-fixed).")
