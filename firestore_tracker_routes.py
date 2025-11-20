import os
import json
from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy
import google.generativeai as genai

tracker_routes = Blueprint("tracker_routes", __name__)

# Ładowanie spaCy
nlp = spacy.load("pl_core_news_sm")

# Konfiguracja Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ===========================================================
# ⚖️ GEMINI JUDGE (Bez zmian)
# ===========================================================
def evaluate_with_gemini(text, meta_trace):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "Brak klucza Gemini - skip check"}

    try:
        model = genai.GenerativeModel('gemini-pro')
    except:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": "Model Init Error"}
    
    intent = meta_trace.get("execution_intent", "Brak")
    rhythm = meta_trace.get("rhythm_pattern_used", "Brak")
    
    banned_phrases_list = """
    1. WYPEŁNIACZE: "W dzisiejszych czasach", "W dobie...", "Od zarania dziejów".
    2. ŁĄCZNIKI: "Warto zauważyć", "Należy wspomnieć", "Warto dodać".
    3. ZAKOŃCZENIA: "Podsumowując", "Reasumując", "W ostatecznym rozrachunku".
    4. IDIOMY: "Gra warta świeczki", "Strzał w dziesiątkę".
    5. ASEKURANCTWO: "Wszystko zależy od...", "Nie ma jednoznacznej odpowiedzi".
    6. WZMOCNIENIA: "Nie ma wątpliwości", "Bez wątpienia".
    7. ZNAKI: "—" (Długi myślnik/Pauza).
    """

    prompt = f"""
    Jesteś Sędzią Jakości SEO.
    Oceniasz fragment tekstu pod kątem naturalności i braku "AI-izmów".
    
    INTENCJA: {intent} | RYTM: {rhythm}
    BANNED LIST: {banned_phrases_list}
    
    Zwróć JSON:
    {{
        "pass": true/false, (false jeśli zakazane frazy LUB score < 70)
        "quality_score": (0-100),
        "feedback_for_writer": "Instrukcja co poprawić"
    }}
    TEKST: "{text}"
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"pass": True, "quality_score": 80, "feedback_for_writer": f"Gemini Error: {str(e)}"}

# ===========================================================
# 🧠 HYBRID COUNTING
# ===========================================================
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
# 🧠 MAIN PROCESS (HARD SEO VETO)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists: return {"error": "Project not found", "status": 404}

    # -------------------------------------------------------
    # 1. QUALITY GATE (GEMINI)
    # -------------------------------------------------------
    gemini_verdict = {"pass": True, "quality_score": 100}
    if meta_trace:
        gemini_verdict = evaluate_with_gemini(batch_text, meta_trace)
    
    QUALITY_THRESHOLD = 70

    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < QUALITY_THRESHOLD:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "quality_alert": True,
            "info": "Tekst odrzucony przez Gemini (Styl/Jakość)."
        }

    # -------------------------------------------------------
    # 2. PRZELICZENIE SEO "NA BRUDNO" (Przed zapisem)
    # -------------------------------------------------------
    data = doc.to_dict()
    # Robimy głęboką kopię stanu, żeby policzyć symulację
    import copy
    keywords_state = copy.deepcopy(data.get("keywords_state", {}))

    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]

    # Symulacja liczenia dla tego batcha
    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
             doc_tmp = nlp(original_keyword)
             target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    # Sprawdzamy kryteria SEO
    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # -------------------------------------------------------
    # 3. HARD SEO VETO (Nowość)
    # -------------------------------------------------------
    # Jeśli batch powoduje krytyczne przeoptymalizowanie, ODRZUCAMY go.
    # Kryterium: LOCKED >= 4 (Emergency) LUB nagły wzrost OVER (>10)
    
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {
                "pass": False,
                "quality_score": gemini_verdict.get("quality_score"),
                "feedback_for_writer": f"SEO CRITICAL: Tekst przeoptymalizowany! LOCKED={locked}, OVER={over}. Zredukuj użycie słów kluczowych."
            },
            "quality_alert": True,
            "info": "Tekst odrzucony przez Hard SEO Veto (za dużo fraz)."
        }

    # -------------------------------------------------------
    # 4. ZAPIS (Jeśli przeszedł i Gemini, i SEO Veto)
    # -------------------------------------------------------
    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok}
    }

    if "batches" not in data: data["batches"] = []
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    # Nadpisujemy stan licznika (bo jest zaakceptowany)
    data["keywords_state"] = keywords_state 

    doc_ref.set(data)

    meta_prompt_summary = f"UNDER={under}, OVER={over}, LOCKED={locked} | Quality={gemini_verdict.get('quality_score')}%"

    return {
        "status": "BATCH_ACCEPTED",
        "counting_mode": "uuid_hybrid",
        "gemini_feedback": gemini_verdict,
        "quality_alert": False,
        "keywords_report": [
            {
                "keyword": meta.get("keyword", "Unknown"),
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": ("INCREASE" if meta["status"] == "UNDER" else "DECREASE" if meta["status"] == "OVER" else "IGNORE")
            }
            for row_id, meta in sorted(keywords_state.items(), key=lambda item: item[1].get("keyword", ""))
        ],
        "meta_prompt_summary": meta_prompt_summary
    }

# Endpointy GET
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return jsonify(doc.to_dict() if doc.exists else {"error": "Not found"}), 200 if doc.exists else 404

@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return jsonify(doc.to_dict().get("keywords_state", {}) if doc.exists else {"error": "Not found"}), 200 if doc.exists else 404
