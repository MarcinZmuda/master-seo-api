from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import spacy

batch_routes = Blueprint("batch_routes", __name__)

# spaCy ładowany raz na start kontenera
nlp = spacy.load("pl_core_news_sm")

# ---------------------------------------------------------
# 🔧 Lematyzacja
# ---------------------------------------------------------
def lemmatize_text(text: str):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


# ---------------------------------------------------------
# 🔧 Aktualizacja statusów: UNDER / OK / OVER / LOCKED
# ---------------------------------------------------------
def update_keyword_status(actual, target_min, target_max):
    if actual < target_min:
        return "UNDER"
    if target_min <= actual <= target_max:
        return "OK"
    if actual > target_max:
        return "OVER"
    return "OK"


# ---------------------------------------------------------
# 🔧 Liczenie OVER/LOCKED po aktualizacji
# ---------------------------------------------------------
def compute_global_stats(keywords_state):
    under = sum(1 for k in keywords_state.values() if k["status"] == "UNDER")
    over = sum(1 for k in keywords_state.values() if k["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for k in keywords_state.values() if k["status"] == "OK")

    return under, over, locked, ok


# ---------------------------------------------------------
# 🆕 S3 – Dodanie batcha + globalne liczenie lemma
# ---------------------------------------------------------
@batch_routes.post("/api/project/<project_id>/add_batch")
def add_batch(project_id):

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' required"}), 400

    batch_text = data["text"]

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data["keywords_state"]

    # ---------------------------------------------------------
    # 1. 🔠 Lematyzacja batcha
    # ---------------------------------------------------------
    lemmas = lemmatize_text(batch_text)

    # ---------------------------------------------------------
    # 2. 🔢 Aktualizacja liczników w Firestore (global)
    # ---------------------------------------------------------
    for kw, meta in keywords_state.items():
        lemma = meta["lemma"]

        # Liczymy wystąpienia lematów
        occurrences = sum(1 for word in lemmas if word == lemma.lower())

        # Zwiększamy globalny stan
        meta["actual_uses"] += occurrences

        # Aktualizacja statusu
        meta["status"] = update_keyword_status(
            meta["actual_uses"], meta["target_min"], meta["target_max"]
        )

    # ---------------------------------------------------------
    # 3. 🚨 Forced Regeneration / Emergency Exit
    # ---------------------------------------------------------
    under_count, over_count, locked_count, ok_count = compute_global_stats(keywords_state)

    forced_regen = False
    emergency_exit = False

    # Emergency exit → LOCKED (≥4 OVER)
    if locked_count >= 1:
        emergency_exit = True

    # Forced regeneration → OVER ≥ 10
    if over_count >= 10:
        forced_regen = True

    # ---------------------------------------------------------
    # 4. 🔥 Zapis batcha + aktualizacja projektu
    # ---------------------------------------------------------
    batch_entry = {
        "text": batch_text,
        "lemmas": lemmas,
        "forced_regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "summary": {
            "under": under_count,
            "over": over_count,
            "locked": locked_count,
            "ok": ok_count,
        }
    }

    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state

    doc_ref.set(project_data)

    # ---------------------------------------------------------
    # 5. 📦 Meta Prompt Summary — wraca do GPT
    # ---------------------------------------------------------
    meta_prompt_summary = (
        f"UNDER={under_count}, "
        f"OVER={over_count}, "
        f"LOCKED={locked_count}, "
        f"OK={ok_count} – global lemma"
    )

    return jsonify({
        "status": "BATCH_RECORDED",
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "keywords_report": [
            {
                "keyword": kw,
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": (
                    "INCREASE" if meta["status"] == "UNDER" else
                    "DECREASE" if meta["status"] == "OVER" else
                    "IGNORE"
                )
            }
            for kw, meta in keywords_state.items()
        ],
        "meta_prompt_summary": meta_prompt_summary
    }), 200
