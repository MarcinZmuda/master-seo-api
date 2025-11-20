import os
import json
from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy
import google.generativeai as genai

tracker_routes = Blueprint("tracker_routes", __name__)
nlp = spacy.load("pl_core_news_sm")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# (Funkcja evaluate_with_gemini BEZ ZMIAN - skopiuj z poprzedniej wersji)
# ... [TUTAJ WKLEJ evaluate_with_gemini] ...
# Skracam dla czytelności, ale Ty wklej pełną funkcję.

def evaluate_with_gemini(text, meta_trace):
    if not GEMINI_API_KEY: return {"pass": True, "quality_score": 100, "feedback": "No key"}
    model = genai.GenerativeModel('gemini-1.5-flash')
    # ... (reszta logiki sędziego) ...
    return {"pass": True, "quality_score": 100} # Placeholder

def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    target_exact_lower = target_exact.lower()
    exact_hits = text_lower.count(target_exact_lower)
    lemma_hits = 0
    target_tokens = target_lemma.split()
    if target_tokens:
        target_len = len(target_tokens)
        text_len = len(text_lemma_list)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i + target_len] == target_tokens:
                lemma_hits += 1
    return max(exact_hits, lemma_hits)

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
# 🧠 GŁÓWNA FUNKCJA (UUID SUPPORT)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists: return {"error": "Project not found", "status": 404}

    # 1. QUALITY GATE
    gemini_verdict = {"pass": True, "quality_score": 100}
    if meta_trace:
        # Tu wywołaj pełną funkcję evaluate_with_gemini
        pass 
    
    # (Logika odrzucania jak wcześniej)

    # 2. ZLICZANIE (UUID MODE)
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})

    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]

    # Iterujemy po UUID (klucz = ID, meta = dane frazy)
    for row_id, meta in keywords_state.items():
        # Teraz nazwa frazy jest wewnątrz meta['keyword']
        original_keyword = meta.get("keyword", "")
        
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        
        if not target_lemma: # Fallback
             doc_tmp = nlp(original_keyword)
             target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    under, over, locked, ok = global_keyword_stats(keywords_state)

    # 3. ZAPIS
    batch_entry = {
        "text": batch_text,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok}
    }
    if "batches" not in data: data["batches"] = []
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state
    doc_ref.set(data)

    meta_prompt_summary = f"UNDER={under}, OVER={over}, LOCKED={locked} (UUID Mode)"

    return {
        "status": "BATCH_ACCEPTED",
        "keywords_report": [
            {
                "keyword": meta.get("keyword", "Unknown"), # Wyciągamy nazwę z obiektu
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": ("INCREASE" if meta["status"] == "UNDER" else "DECREASE" if meta["status"] == "OVER" else "IGNORE")
            }
            # Sortujemy raport alfabetycznie po nazwie frazy, żeby był czytelny
            for row_id, meta in sorted(keywords_state.items(), key=lambda item: item[1].get("keyword", ""))
        ],
        "meta_prompt_summary": meta_prompt_summary
    }
