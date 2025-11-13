# ================================================================
# firestore_tracker_routes.py — Warstwa Batch + Keyword Tracker (v7.2.9-firestore-continuous-lemma)
# ================================================================

from flask import Blueprint, request, jsonify
from datetime import datetime
import re
import spacy
import uuid

tracker_bp = Blueprint("firestore_tracker_routes", __name__)
db = None

# ---------------------------------------------------------------
# ✅ Inicjalizacja modelu spaCy
# ---------------------------------------------------------------
try:
    nlp_pl = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie.")
except OSError:
    import os
    os.system("python -m spacy download pl_core_news_sm")
    nlp_pl = spacy.load("pl_core_news_sm")
    print("📦 Model spaCy 'pl_core_news_sm' został pobrany i załadowany.")


# ---------------------------------------------------------------
# 🔠 Funkcje lematyzacji i zliczania
# ---------------------------------------------------------------
def lemmatize_text(text: str):
    """Zwraca listę lematów z tekstu (tylko tokeny alfabetyczne, lowercase)."""
    doc = nlp_pl(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


def count_keyword_occurrences(text: str, keyword: str) -> int:
    """
    Liczy wystąpienia frazy w tekście na podstawie lematów.
    'adwokata rozwodowego' == 'adwokaci rozwodowi' (lematy: adwokat, rozwodowy)
    """
    text_lemmas = lemmatize_text(text)
    keyword_lemmas = lemmatize_text(keyword)
    keyword_len = len(keyword_lemmas)
    count = 0

    for i in range(len(text_lemmas) - keyword_len + 1):
        if text_lemmas[i:i + keyword_len] == keyword_lemmas:
            count += 1
    return count


# ---------------------------------------------------------------
# ⚙️ System progów i reakcji
# ---------------------------------------------------------------
def trigger_forced_regeneration(doc_ref, project_id, over_count):
    print(f"⚠️ [Forced Regeneration] Projekt {project_id} – OVER={over_count}")
    doc_ref.update({
        "status": "regenerating",
        "regeneration_triggered_at": datetime.utcnow().isoformat(),
        "regeneration_reason": "OVER ≥10"
    })
    return True


def trigger_emergency_exit(doc_ref, project_id, locked_count):
    print(f"⛔ [Emergency Exit] Projekt {project_id} – LOCKED={locked_count}")
    doc_ref.update({
        "status": "halted",
        "emergency_exit_triggered_at": datetime.utcnow().isoformat(),
        "emergency_exit_reason": "LOCKED ≥4"
    })
    return True


# ---------------------------------------------------------------
# ✅ /api/project/<project_id>/add_batch — CIĄGŁE LICZENIE
# ---------------------------------------------------------------
@tracker_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch(project_id):
    """
    Dodaje batch treści do projektu Firestore.
    Zlicza frazy (lematycznie) i aktualizuje globalne wartości w keywords_state.
    Zwraca zawsze pełny raport z całego projektu.
    """
    global db
    if not db:
        return jsonify({"error": "Firestore nie jest połączony"}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    batch_id = data.get("batch_id") or f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    if not text:
        return jsonify({"error": "Brak tekstu do zapisu"}), 400

    # 🔍 Pobranie projektu
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": f"Projekt {project_id} nie istnieje"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})

    if not keywords_state:
        return jsonify({"error": "Brak fraz w projekcie (keywords_state pusty)"}), 400

    print(f"[INFO] 🧮 Zliczanie batcha {batch_id} ({len(keywords_state)} fraz) dla projektu {project_id}")

    # 🔢 Liczenie
    updated_keywords = 0
    keywords_report = []
    under_terms_count = over_terms_count = ok_terms_count = 0

    for keyword, info in keywords_state.items():
        count = count_keyword_occurrences(text, keyword)

        if count > 0:
            info["actual"] = info.get("actual", 0) + count
            updated_keywords += 1

        if info["actual"] < info["target_min"]:
            status = "UNDER"
            under_terms_count += 1
        elif info["actual"] > info["target_max"] + 10:
            status = "OVER"
            over_terms_count += 1
        else:
            status = "OK"
            ok_terms_count += 1

        info["status"] = status
        keywords_report.append({
            "keyword": keyword,
            "actual_uses": info["actual"],
            "target_range": f"{info['target_min']}-{info['target_max']}",
            "status": status
        })
        keywords_state[keyword] = info

    # 🔐 LOCKED = OVER
    locked_terms = [k for k, v in keywords_state.items() if v["status"] == "OVER"]
    locked_terms_count = len(locked_terms)

    regeneration_triggered = over_terms_count >= 10
    emergency_exit_triggered = locked_terms_count >= 4

    if regeneration_triggered:
        trigger_forced_regeneration(doc_ref, project_id, over_terms_count)
    if emergency_exit_triggered:
        trigger_emergency_exit(doc_ref, project_id, locked_terms_count)

    # 💾 Zapis batcha w subkolekcji
    batch_ref = db.collection("seo_projects").document(project_id).collection("batches").document(batch_id)
    batch_ref.set({
        "created_at": datetime.utcnow().isoformat(),
        "text_length": len(text),
        "summary": {
            "under": under_terms_count,
            "over": over_terms_count,
            "locked": locked_terms_count,
            "ok": ok_terms_count,
            "updated_keywords": updated_keywords,
        },
        "text": text
    })

    # 🔄 Aktualizacja głównego dokumentu projektu
    doc_ref.update({
        "keywords_state": keywords_state,
        "status": "updated",
        "last_batch_id": batch_id,
        "last_update_at": datetime.utcnow().isoformat()
    })

    # 🧠 Raport meta (sumaryczny, z całego projektu)
    meta_prompt_summary = {
        "updated_keywords": updated_keywords,
        "locked_terms_count": locked_terms_count,
        "under_terms_count": under_terms_count,
        "over_terms_count": over_terms_count,
        "ok_terms_count": ok_terms_count,
        "summary_text": f"UNDER={under_terms_count}, OVER={over_terms_count}, LOCKED={locked_terms_count}, OK={ok_terms_count}"
    }

    print(f"[INFO] ✅ Batch {batch_id} przetworzony ({updated_keywords} fraz). "
          f"OVER={over_terms_count}, LOCKED={locked_terms_count}")

    return jsonify({
        "status": "✅ Batch zapisany i przetworzony",
        "project_id": project_id,
        "batch_id": batch_id,
        "batch_length": len(text),
        "keywords_report": sorted(keywords_report, key=lambda x: x['keyword']),
        "meta_prompt_summary": meta_prompt_summary,
        "regeneration_triggered": regeneration_triggered,
        "emergency_exit_triggered": emergency_exit_triggered
    }), 200


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_tracker_routes(app, _db=None):
    """Rejestruje blueprint firestore_tracker_routes."""
    global db
    db = _db
    app.register_blueprint(tracker_bp)
    print("✅ [INIT] firestore_tracker_routes zarejestrowany (v7.2.9-firestore-continuous-lemma).")
