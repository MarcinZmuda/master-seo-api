import os
import json
from flask import Blueprint, jsonify
from firebase_admin import firestore
import spacy
import google.generativeai as genai

tracker_routes = Blueprint("tracker_routes", __name__)

# Ładowanie spaCy (raz przy starcie aplikacji)
nlp = spacy.load("pl_core_news_sm")

# Konfiguracja Gemini (Pobierana ze zmiennych środowiskowych Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ===========================================================
# ⚖️ GEMINI JUDGE (Sędzia z Twoją listą Banned + Myślnik)
# ===========================================================
def evaluate_with_gemini(text, meta_trace):
    """
    Wysyła tekst do Gemini w celu oceny jakości (Glass Box).
    Zwraca werdykt JSON: pass/fail, score, feedback.
    """
    if not GEMINI_API_KEY:
        # Fallback: Jeśli brak klucza, przepuszczamy tekst (Fail Open)
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "Brak klucza Gemini - skip check"}

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Pobieramy metadane od GPT (intencja i rytm)
    intent = meta_trace.get("execution_intent", "Brak")
    rhythm = meta_trace.get("rhythm_pattern_used", "Brak")
    
    # 🔥 ZAKTUALIZOWANA LISTA DLA GEMINI JUDGE
    banned_phrases_list = """
    1. WYPEŁNIACZE STARTOWE: "W dzisiejszych czasach", "W dobie...", "Od zarania dziejów", "W niniejszym artykule", "Coraz więcej osób".
    2. LENIWE ŁĄCZNIKI: "Warto zauważyć", "Należy wspomnieć", "Warto dodać", "Co więcej", "Ponadto", "Kolejnym aspektem".
    3. ZAKOŃCZENIA: "Podsumowując", "Reasumując", "W ostatecznym rozrachunku", "Biorąc wszystko pod uwagę".
    4. IDIOMY AI: "Gra warta świeczki", "Strzał w dziesiątkę", "Szyte na miarę", "Klucz do sukcesu".
    5. ASEKURANCTWO: "Wszystko zależy od indywidualnych preferencji", "Każde rozwiązanie ma wady i zalety".
    6. WZMOCNIENIA: "Nie ma wątpliwości", "Bez wątpienia", "Z całą pewnością", "Niezwykle ważne".
    7. ZNAKI: "—" (Długi myślnik/Pauza - AI nadużywa go do wtrąceń).
    """

    prompt = f"""
    Jesteś bezwzględnym Sędzią Jakości SEO (Quality Gatekeeper).
    Oceniasz fragment tekstu pod kątem naturalności, stylu HEAR i braku "AI-izmów".
    
    PARAMETRY AUTORA:
    - Intencja: {intent}
    - Deklarowany Rytm: {rhythm}
    
    LISTA ZAKAZANYCH FRAZ:
    {banned_phrases_list}
    
    KRYTERIA PUNKTACJI (0-100):
    - < 50 (ODRZUT): Występują zakazane frazy (w tym długie myślniki "—"), styl bota, listy punktowane.
    - 50-69 (SŁABY): Nudny, powtarzalny schemat zdań.
    - 70-89 (DOBRY): Naturalny język, brak zakazanych fraz, dobra asymetria.
    - 90+ (WYBITNY): Ludzki styl, flow, nieszablonowe słownictwo.

    Zwróć JSON:
    {{
        "pass": true/false, (false jeśli są zakazane słowa LUB score < 70)
        "quality_score": (0-100),
        "feedback_for_writer": "Instrukcja co poprawić (np. 'Usuń długie myślniki', 'Zmień rytm')"
    }}
    
    TEKST DO OCENY:
    "{text}"
    """
    try:
        response = model.generate_content(prompt)
        # Czyszczenie odpowiedzi z markdowna (```json ... ```)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"pass": True, "quality_score": 50, "feedback_for_writer": "Gemini Error"}


# ===========================================================
# 🧠 HYBRID COUNTING ENGINE (To jest ta nowość!)
# ===========================================================
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    """
    Liczy wystąpienia w trybie HYBRYDOWYM.
    Wystąpienie zaliczamy, jeśli:
    A) Znaleziono dokładny string (exact match case-insensitive)
    B) LUB znaleziono sekwencję lematów (lemma match)
    """
    text_lower = text_raw.lower()
    target_exact_lower = target_exact.lower()
    
    # 1. Exact Match (proste szukanie stringa)
    exact_hits = text_lower.count(target_exact_lower)
    
    # 2. Lemma Match (Sliding Window)
    lemma_hits = 0
    target_tokens = target_lemma.split()
    if target_tokens:
        target_len = len(target_tokens)
        text_len = len(text_lemma_list)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i + target_len] == target_tokens:
                lemma_hits += 1
    
    # Wynik to MAX z obu metod (żeby nie liczyć podwójnie tego samego słowa)
    # Dzięki temu "auta" zaliczy Exact(0) i Lemma(1) -> Wynik 1
    # A "auto" zaliczy Exact(1) i Lemma(1) -> Wynik 1
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
# 🧠 GŁÓWNA FUNKCJA (QUALITY GATE + HYBRID COUNTING)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return {"error": "Project not found", "status": 404}

    # -------------------------------------------------------
    # 1. QUALITY GATE (GEMINI)
    # -------------------------------------------------------
    gemini_verdict = {"pass": True, "quality_score": 100}
    
    if meta_trace:
        gemini_verdict = evaluate_with_gemini(batch_text, meta_trace)
    
    # 🔥 PRÓG JAKOŚCI = 70 punktów
    QUALITY_THRESHOLD = 70

    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < QUALITY_THRESHOLD:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "quality_alert": True,
            "info": f"Odrzucono: Wynik {gemini_verdict.get('quality_score')} < {QUALITY_THRESHOLD} lub wykryto zakazane frazy."
        }

    # -------------------------------------------------------
    # 2. ZLICZANIE HYBRYDOWE (Tylko jeśli Quality Gate = Pass)
    # -------------------------------------------------------
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})

    # Przygotowanie danych do liczenia
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha] # Dla lemma match
    # batch_text jest używany dla exact match

    for original_keyword, meta in keywords_state.items():
        # Pobieramy parametry szukania
        target_exact = meta.get("search_term_exact", original_keyword)
        target_lemma = meta.get("search_lemma", "")
        
        # Fallback dla starych projektów
        if not target_lemma:
             doc_tmp = nlp(original_keyword)
             target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        # 🔥 HYBRID COUNTING
        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        
        # Aktualizacja stanu
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    # Statystyki globalne
    under, over, locked, ok = global_keyword_stats(keywords_state)
    forced_regen = over >= 10
    emergency_exit = locked >= 1

    # -------------------------------------------------------
    # 3. ZAPIS DO BAZY
    # -------------------------------------------------------
    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok}
    }

    if "batches" not in data:
        data["batches"] = []
    
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state

    doc_ref.set(data)

    # -------------------------------------------------------
    # 4. ODPOWIEDŹ DLA GPT
    # -------------------------------------------------------
    meta_prompt_summary = f"UNDER={under}, OVER={over}, LOCKED={locked} | Quality={gemini_verdict.get('quality_score')}%"

    return {
        "status": "BATCH_ACCEPTED",
        "counting_mode": "hybrid_row",
        "gemini_feedback": gemini_verdict,
        "quality_alert": False,
        "regeneration_triggered": forced_regen,
        "emergency_exit_triggered": emergency_exit,
        "keywords_report": [
            {
                "keyword": kw,
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": ("INCREASE" if meta["status"] == "UNDER" else "DECREASE" if meta["status"] == "OVER" else "IGNORE")
            }
            for kw, meta in keywords_state.items()
        ],
        "meta_prompt_summary": meta_prompt_summary
    }

# ===========================================================
# 📌 ENDPOINTY GET
# ===========================================================
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return jsonify(doc.to_dict() if doc.exists else {"error": "Project not found"}), 200 if doc.exists else 404

@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return jsonify(doc.to_dict().get("keywords_state", {}) if doc.exists else {"error": "Project not found"}), 200 if doc.exists else 404
