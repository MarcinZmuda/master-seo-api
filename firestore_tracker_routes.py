import os
import json
from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz  # ⬅️ NOWE: fuzzy-matching

tracker_routes = Blueprint("tracker_routes", __name__)

# Ładowanie spaCy
nlp = spacy.load("pl_core_news_sm")

# Parametry fuzzy-matchingu dla trackera
FUZZY_SIMILARITY_THRESHOLD = 90  # próg podobieństwa 0–100
MAX_FUZZY_WINDOW_EXPANSION = 2   # ile dodatkowych lematów dopuszczamy w środku frazy

# Konfiguracja Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ===========================================================
# ⚖️ GEMINI JUDGE (Model Stable: gemini-pro)
# ===========================================================
def evaluate_with_gemini(text, meta_trace):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "Brak klucza Gemini - skip check"}

    try:
        # Używamy gemini-pro (najbardziej stabilny w API)
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
        # Fail Open: W razie błędu API Google przepuszczamy tekst (żeby nie blokować pracy)
        return {"pass": True, "quality_score": 80, "feedback_for_writer": f"Gemini API Error: {str(e)}"}

# ===========================================================
# 🔍 Fuzzy na lematyzowanym tekście (dla trackera)
# ===========================================================
def _count_fuzzy_on_lemmas(target_tokens, text_lemma_list, exact_spans):
    """
    target_tokens: lista lematów frazy, np. ["adwokat", "rozwodowy", "warszawa"]
    text_lemma_list: lista lematów całego batcha
    exact_spans: lista (start, end) dla exact lemma matchy (żeby ich nie dublować)
    """
    if not target_tokens or not text_lemma_list:
        return 0

    text_len = len(text_lemma_list)
    target_len = len(target_tokens)
    if target_len == 0 or text_len < target_len:
        return 0

    kw_str = " ".join(target_tokens)

    # Pozycje zajęte przez exact lemma match
    used_positions = set()
    for s, e in exact_spans:
        used_positions.update(range(s, e))

    fuzzy_hits = 0

    for start in range(text_len):
        for extra in range(MAX_FUZZY_WINDOW_EXPANSION + 1):
            end = start + target_len + extra
            if end > text_len:
                break

            window_positions = range(start, end)
            if any(pos in used_positions for pos in window_positions):
                continue

            window_str = " ".join(text_lemma_list[start:end])
            score = fuzz.token_set_ratio(kw_str, window_str)

            if score >= FUZZY_SIMILARITY_THRESHOLD:
                fuzzy_hits += 1
                used_positions.update(window_positions)
                # nie szukamy dłuższego okna od tego samego startu
                break

    return fuzzy_hits

# ===========================================================
# 🧠 HYBRID COUNTING
# ===========================================================
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    """
    Hybrydowe zliczanie:
    - Exact na surowym tekście (case-insensitive)
    - Exact na lematyzowanym tekście
    - Fuzzy na lematyzowanym tekście (rapidfuzz), toleruje wtrącenia typu "w", "na"
    Zwraca max(exact_hits, lemma_hits_total).
    """
    text_lower = text_raw.lower()
    target_exact_lower = target_exact.lower()
    
    # 1. Exact na surowym tekście
    exact_hits = 0
    if target_exact_lower.strip():
        exact_hits = text_lower.count(target_exact_lower)
    
    # 2. Lemma (exact + fuzzy)
    lemma_hits = 0
    exact_spans = []
    target_tokens = target_lemma.split()

    if target_tokens:
        target_len = len(target_tokens)
        text_len = len(text_lemma_list)

        # 2a. Exact lemma match
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i + target_len] == target_tokens:
                lemma_hits += 1
                exact_spans.append((i, i + target_len))

        # 2b. Fuzzy lemma match (tylko poza exact_spans)
        fuzzy_hits = _count_fuzzy_on_lemmas(
            target_tokens=target_tokens,
            text_lemma_list=text_lemma_list,
            exact_spans=exact_spans,
        )
        lemma_hits += fuzzy_hits
                
    return max(exact_hits, lemma_hits)

def compute_status(actual, target_min, target_max):
    if actual < target_min: return "UNDER"
    # Bufor tolerancji: Pozwalamy na lekkie przekroczenie (np. o 20% lub +2)
    tolerance = max(2, int(target_max * 0.2))
    if actual > (target_max + tolerance): return "OVER"
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
    # 2. PRZELICZENIE SEO "NA BRUDNO" (Symulacja przed zapisem)
    # -------------------------------------------------------
    data = doc.to_dict()
    # Robimy głęboką kopię stanu, żeby policzyć symulację bez psucia bazy
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

    # Sprawdzamy kryteria SEO po dodaniu batcha
    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # -------------------------------------------------------
    # 3. HARD SEO VETO (Ochrona przed spamem)
    # -------------------------------------------------------
    # Jeśli ten batch spowodowałby krytyczne przeoptymalizowanie -> ODRZUCAMY
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {
                "pass": False,
                "quality_score": gemini_verdict.get("quality_score", 0),
                "feedback_for_writer": f"SEO CRITICAL: Tekst przeoptymalizowany! LOCKED={locked}, OVER={over}. Zredukuj użycie słów kluczowych."
            },
            "quality_alert": True,
            "info": "Tekst odrzucony przez Hard SEO Veto."
        }

    # -------------------------------------------------------
    # 4. ZAPIS (Skoro przeszedł Gemini i SEO Veto)
    # -------------------------------------------------------
    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok}
    }

    if "batches" not in data: data["batches"] = []
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    # Nadpisujemy stan licznika w bazie (zatwierdzamy symulację)
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
            # Sortujemy wynik alfabetycznie dla czytelności
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
