from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy

tracker_routes = Blueprint("tracker_routes", __name__)
nlp = spacy.load("pl_core_news_sm")

# ===========================================================
# 🧠 Algorytm liczenia sekwencji lematów
# ===========================================================
def count_phrase_in_text_lemmas(text_lemma_list, phrase_lemma_str):
    """
    Sprawdza, ile razy sekwencja lematów (phrase_lemma_str) występuje
    w liście lematów tekstu (text_lemma_list).
    """
    target_tokens = phrase_lemma_str.split()
    if not target_tokens:
        return 0
    
    target_len = len(target_tokens)
    text_len = len(text_lemma_list)
    count = 0

    # Przesuwamy okno po tekście
    for i in range(text_len - target_len + 1):
        # Wycinamy okno o długości szukanej frazy
        window = text_lemma_list[i : i + target_len]
        
        # Porównujemy listy
        if window == target_tokens:
            count += 1

    return count

# ===========================================================
# 🔧 Status calc
# ===========================================================
def compute_status(actual, target_min, target_max):
    if actual < target_min: return "UNDER"
    if actual > target_max: return "OVER"
    return "OK"

def global_keyword_stats(keywords_state):
    under = sum(1 for v in keywords_state.values() if v["status"] == "UNDER")
    over = sum(1 for v in keywords_state.values() if v["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for v in keywords_state.values() if v["status"] == "OK")
    return under, over, locked, ok

# ===========================================================
# 🧠 MAIN PROCESS
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return {"error": "Project not found", "status": 404}

    data = doc.to_dict()
    keywords_state = data["keywords_state"]

    # 1. Lematyzacja CAŁEGO tekstu batcha do listy
    # "Kupiłem aparaty" -> ['kupić', 'aparat']
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]

    # 2. Iteracja po wszystkich frazach z briefu
    for original_keyword, meta in keywords_state.items():
        # Pobieramy wzorzec lematyczny (np. "aparat słuchowy")
        search_lemma = meta.get("search_lemma", "")
        
        # Jeśli to stary projekt bez search_lemma, robimy fallback
        if not search_lemma:
            doc_tmp = nlp(original_keyword)
            search_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        # Liczymy wystąpienia WZORCA w LEMATACH TEKSTU
        occurrences = count_phrase_in_text_lemmas(text_lemma_list, search_lemma)

        # Aktualizujemy stan (Continuous counting)
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(
            meta["actual_uses"], 
            meta["target_min"], 
            meta["target_max"]
        )

    # 3. Statystyki globalne
    under, over, locked, ok = global_keyword_stats(keywords_state)
    forced_regen = over >= 10
    emergency_exit = locked >= 1

    # 4. Zapis batcha
    batch_entry = {
        "text": batch_text,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok}
    }
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state

    doc_ref.set(data)

    # 5. Wynik
    meta_prompt_summary = f"UNDER={under}, OVER={over}, LOCKED={locked}, OK={ok} – Row-Level Lemma Mode"

    return {
        "status": "BATCH_PROCESSED",
        "counting_mode": "row_lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "keywords_report": [
            {
                "keyword": kw, # Oryginalna fraza z briefu
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": (
                    "INCREASE" if meta["status"] == "UNDER" else
                    "DECREASE" if meta["status"] == "OVER" else "IGNORE"
                )
            }
            for kw, meta in keywords_state.items()
        ],
        "meta_prompt_summary": meta_prompt_summary
    }

# Endpointy GET bez zmian (pominięte dla zwięzłości, są w poprzednich plikach)
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    # ... (standardowy kod)
    pass 

@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    # ... (standardowy kod)
    pass
