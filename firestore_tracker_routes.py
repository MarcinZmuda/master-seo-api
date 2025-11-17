from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy

tracker_routes = Blueprint("tracker_routes", __name__)

# -------------------------------------------------------------------
# SPAcy ładowany 1 raz — optymalne dla Render + Gunicorn
# -------------------------------------------------------------------
nlp = spacy.load("pl_core_news_sm")


# ===========================================================
# 🔠 Lematization helper
# ===========================================================
def lemmatize_text(text: str):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc if t.is_alpha]


# ===========================================================
# 🔧 Status calculation (UNDER / OK / OVER / LOCKED)
# ===========================================================
def compute_status(actual, target_min, target_max):
    if actual < target_min:
        return "UNDER"
    if actual > target_max:
        return "OVER"
    return "OK"


# ===========================================================
# 🔄 Global Stats Summary (UNDER / OVER / LOCKED / OK)
# LOCKED = jeśli ≥4 frazy są w stanie OVER
# ===========================================================
def global_keyword_stats(keywords_state):
    under = sum(1 for v in keywords_state.values() if v["status"] == "UNDER")
    over = sum(1 for v in keywords_state.values() if v["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for v in keywords_state.values() if v["status"] == "OK")
    return under, over, locked, ok


# ===========================================================
# 🧠 MAIN FUNCTION (the one project_routes will call)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str):
    """
    Najważniejsza funkcja — wywoływana BEZPOŚREDNIO z project_routes.py
    Nie ma żadnych requestów HTTP — wszystko odbywa się lokalnie i szybko.
    """

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return {"error": "Project not found", "status": 404}

    data = doc.to_dict()
    keywords_state = data["keywords_state"]

    # ------------------------------------------
    # 1) Lematyzacja batcha
    # ------------------------------------------
    lemmas = lemmatize_text(batch_text)

    # ------------------------------------------
    # 2) Aktualizacja liczników globalnych
    # ------------------------------------------
    for kw, meta in keywords_state.items():
        lemma = meta["lemma"].lower()

        occ = sum(1 for w in lemmas if w == lemma)
        meta["actual_uses"] += occ

        meta["status"] = compute_status(
            meta["actual_uses"],
            meta["target_min"],
            meta["target_max"]
        )

    # ------------------------------------------
    # 3) Obliczenie globalnych statusów
    # ------------------------------------------
    under, over, locked, ok = global_keyword_stats(keywords_state)

    forced_regen = over >= 10
    emergency_exit = locked >= 1

    # ------------------------------------------
    # 4) Dodanie batcha do historii projektu
    # ------------------------------------------
    batch_entry = {
        "text": batch_text,
        "lemmas": lemmas,
        "forced_regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "summary": {
            "under": under,
            "over": over,
            "locked": locked,
            "ok": ok
        }
    }

    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state

    # ------------------------------------------
    # 5) Zapis zmian
    # ------------------------------------------
    doc_ref.set(data)

    # ------------------------------------------
    # 6) Meta summary (wraca do GPT)
    # ------------------------------------------
    meta_prompt_summary = (
        f"UNDER={under}, OVER={over}, LOCKED={locked}, OK={ok} – global lemma"
    )

    # ------------------------------------------
    # 7) Finalny JSON — identyczny format jak /add_batch
    # + zwraca batch_text dla GPT
    # ------------------------------------------
    return {
        "status": "BATCH_PROCESSED",
        "batch_text": batch_text,   # <<< DODANE — TREŚĆ BATCHA WRACA DO GPT
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "keywords_report": [
            {
                "keyword": kw,
                "lemma": meta["lemma"],
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
    }


# ===========================================================
# 📌 ENDPOINT 1: Podgląd projektu
# ===========================================================
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    return jsonify(doc.to_dict()), 200


# ===========================================================
# 📌 ENDPOINT 2: Podgląd keywordów
# ===========================================================
@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()
    return jsonify(data.get("keywords_state", {})), 200
