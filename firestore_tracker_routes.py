import os
import json
import math
import re
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz
import language_tool_python
import textstat
import textdistance
import pysbd

tracker_routes = Blueprint("tracker_routes", __name__)

# -----------------------------------------------------------
# 🧠 spaCy – ładowane raz
# -----------------------------------------------------------
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

# -----------------------------------------------------------
# 🎯 Parametry fuzzy-matchingu
# -----------------------------------------------------------
FUZZY_SIMILARITY_THRESHOLD = 90
MAX_FUZZY_WINDOW_EXPANSION = 2
JACCARD_SIMILARITY_THRESHOLD = 0.8

# -----------------------------------------------------------
# 🧪 LanguageTool
# -----------------------------------------------------------
try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception as e:
    print(f"[LanguageTool] Init error: {e}")
    LT_TOOL_PL = None

# -----------------------------------------------------------
# ✂️ Pysbd
# -----------------------------------------------------------
SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

# -----------------------------------------------------------
# 🌐 Konfiguracja Gemini
# -----------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 📏 Stylometria: Burstiness + Fluff + Repetitions
# ===========================================================
def calculate_burstiness(text: str) -> float:
    if not text or not text.strip(): return 0.0
    try:
        raw_sentences = SENTENCE_SEGMENTER.segment(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences: return 0.0
    lengths = [len(s.split()) for s in sentences]
    n = len(lengths)
    if n == 0: return 0.0
    mean_len = sum(lengths) / n
    variance = sum((l - mean_len) ** 2 for l in lengths) / n
    return math.sqrt(variance)


def calculate_fluff_ratio(text: str) -> float:
    if not text or not text.strip(): return 0.0
    doc = nlp(text)
    adv_adj_count = sum(1 for token in doc if token.pos_ in ("ADJ", "ADV"))
    total_alpha_words = sum(1 for token in doc if token.is_alpha)
    if total_alpha_words == 0: return 0.0
    return float(adv_adj_count / total_alpha_words)


def detect_repeated_sentence_starts(text: str, prefix_words: int = 3) -> list:
    """
    Wykrywa powtarzające się początki zdań (anty-AI pattern).
    Np. 3 zdania zaczynające się od "Warto zauważyć, że".
    """
    if not text or not text.strip(): return []
    try:
        raw_sentences = SENTENCE_SEGMENTER.segment(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    prefix_counts = {}
    for s in sentences:
        words = s.split()
        if len(words) < prefix_words: continue
        # Bierzemy pierwsze N słów, lowercase
        prefix = " ".join(words[:prefix_words]).lower()
        # Ignorujemy bardzo krótkie prefiksy (np. "a potem")
        if len(prefix) < 5: continue 
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    # Zwracamy tylko te, które wystąpiły >= 2 razy
    repeated = [
        {"prefix": p, "count": c} 
        for p, c in prefix_counts.items() if c >= 2
    ]
    return repeated


# ===========================================================
# ⚖️ GEMINI JUDGE
# ===========================================================
def evaluate_with_gemini(text, meta_trace, burstiness=0.0, fluff=0.0, repeated_starts=None):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "No API Key"}

    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": "Model Init Error"}

    meta = meta_trace or {}
    intent = meta.get("execution_intent", "Brak")
    rhythm = meta.get("rhythm_pattern_used", "Brak")
    repeated_info = f"Wykryto powtórzenia początków zdań: {repeated_starts}" if repeated_starts else "Brak powtórzeń początków."

    banned_phrases_list = """
    1. WYPEŁNIACZE: "W dzisiejszych czasach", "W dobie...", "Od zarania dziejów".
    2. ŁĄCZNIKI: "Warto zauważyć", "Należy wspomnieć", "Warto dodać", "Co więcej".
    3. ZAKOŃCZENIA: "Podsumowując", "Reasumując", "W ostatecznym rozrachunku".
    4. IDIOMY: "Gra warta świeczki", "Strzał w dziesiątkę", "Klucz do sukcesu".
    5. ASEKURANCTWO: "Wszystko zależy od...", "Nie ma jednoznacznej odpowiedzi".
    6. WZMOCNIENIA: "Nie ma wątpliwości", "Bez wątpienia", "Niezwykle".
    7. ZNAKI: "—" (Długi myślnik/Pauza).
    """
    
    metrics_context = f"""
    DANE STYLOMETRYCZNE (FACTUAL DATA):
    - Burstiness (Zmienność): {burstiness:.2f} (Cel > 6.0).
    - Fluff Ratio (Wata): {fluff:.3f} (Cel < 0.15).
    - {repeated_info}
    """

    prompt = f"""
    Jesteś Surowym Sędzią Jakości SEO (HEAR 2.1).
    Oceniasz tekst pod kątem naturalności i braku "AI-izmów".
    
    INTENCJA: {intent} | RYTM: {rhythm}
    {metrics_context}
    BANNED LIST: {banned_phrases_list}

    ZASADY DECYZJI:
    1. Burstiness < 5.0 -> FAIL. Komentarz: "Rytm robotyczny. Zastosuj asymetrię zdań."
    2. Fluff Ratio > 0.18 -> FAIL. Komentarz: "Za dużo przymiotników. Usuń ozdobniki."
    3. Powtórzenia początków zdań -> FAIL. Komentarz: "Zmień strukturę zdań, nie zaczynaj tak samo."
    
    Zwróć JSON: {{ "pass": true/false, "quality_score": (0-100), "feedback_for_writer": "..." }}
    TEKST: "{text}"
    """

    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"pass": True, "quality_score": 80, "feedback_for_writer": f"API Error: {e}"}


# ===========================================================
# 🔎 Language Audit
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    result = {
        "lt_enabled": LT_TOOL_PL is not None,
        "lt_issues_count": 0,
        "lt_issues_sample": [],
        "readability": {},
        "burstiness": 0.0,
        "fluff_ratio": 0.0,
        "repeated_starts": [],
        "error": None,
    }
    if not text or not text.strip(): return result

    # LanguageTool
    if LT_TOOL_PL:
        try:
            matches = LT_TOOL_PL.check(text)
            result["lt_issues_count"] = len(matches)
            result["lt_issues_sample"] = [{"msg": m.message, "rule": m.ruleId} for m in matches[:5]]
        except Exception as e:
            result["error"] = str(e)

    # Stylometria
    try:
        burstiness_value = calculate_burstiness(text)
        fluff_value = calculate_fluff_ratio(text)
        repeated = detect_repeated_sentence_starts(text, prefix_words=3)
        
        # Statystyki podstawowe do summary
        raw_sentences = SENTENCE_SEGMENTER.segment(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        result["burstiness"] = burstiness_value
        result["fluff_ratio"] = fluff_value
        result["repeated_starts"] = repeated
        result["readability"] = {
            "sentence_count": len(sentences),
            "avg_sentence_length": (sum(len(s.split()) for s in sentences)/len(sentences)) if sentences else 0
        }
    except Exception as e:
        print(f"Stats Error: {e}")

    return result


# ===========================================================
# 🛠 Auto-fix & Fuzzy Helpers
# ===========================================================
def apply_languagetool_fixes(text: str) -> str:
    if LT_TOOL_PL is None or not text: return text
    try:
        matches = LT_TOOL_PL.check(text)
        text_list = list(text)
        for m in sorted(matches, key=lambda x: x.offset, reverse=True):
            if m.replacements:
                text_list[m.offset : m.offset+m.errorLength] = list(m.replacements[0])
        return "".join(text_list)
    except: return text

def _count_fuzzy_on_lemmas(target_tokens, text_lemma_list, exact_spans):
    if not target_tokens or not text_lemma_list: return 0
    text_len, target_len = len(text_lemma_list), len(target_tokens)
    if text_len < target_len: return 0
    
    kw_str = " ".join(target_tokens)
    kw_tokens_set = set(target_tokens)
    used_positions = set()
    for s, e in exact_spans: used_positions.update(range(s, e))

    fuzzy_hits = 0
    for start in range(text_len):
        for extra in range(MAX_FUZZY_WINDOW_EXPANSION + 1):
            end = start + target_len + extra
            if end > text_len: break
            window_positions = range(start, end)
            if any(pos in used_positions for pos in window_positions): continue
            
            window_tokens = text_lemma_list[start:end]
            if not window_tokens: continue
            
            rf_score = fuzz.token_set_ratio(kw_str, " ".join(window_tokens))
            jaccard_score = textdistance.jaccard(kw_tokens_set, set(window_tokens))

            if rf_score >= FUZZY_SIMILARITY_THRESHOLD or jaccard_score >= JACCARD_SIMILARITY_THRESHOLD:
                fuzzy_hits += 1
                used_positions.update(window_positions)
                break
    return fuzzy_hits

def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    target_exact_lower = target_exact.lower()
    exact_hits = text_lower.count(target_exact_lower) if target_exact_lower.strip() else 0

    lemma_hits = 0
    exact_spans = []
    target_tokens = target_lemma.split()
    if target_tokens:
        text_len = len(text_lemma_list)
        target_len = len(target_tokens)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i + target_len] == target_tokens:
                lemma_hits += 1
                exact_spans.append((i, i + target_len))
        lemma_hits += _count_fuzzy_on_lemmas(target_tokens, text_lemma_list, exact_spans)
    
    return max(exact_hits, lemma_hits)

# --- ZAOSTRZONY STATUS (Bez tolerancji 20%) ---
def compute_status(actual, target_min, target_max):
    if actual < target_min: return "UNDER"
    # Tolerancja minimalna (+1) tylko dla bardzo małych zakresów (np. 1-2)
    limit = target_max + 1 if (target_max - target_min <= 1) else target_max
    if actual > limit: return "OVER"
    return "OK"

def global_keyword_stats(keywords_state):
    under = sum(1 for v in keywords_state.values() if v["status"] == "UNDER")
    over = sum(1 for v in keywords_state.values() if v["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for v in keywords_state.values() if v["status"] == "OK")
    return under, over, locked, ok


# ===========================================================
# 🆕 ENDPOINT /api/language_refine
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    auto_fixed = apply_languagetool_fixes(text)
    audit = analyze_language_quality(auto_fixed)
    return jsonify({"original_text":text, "auto_fixed_text":auto_fixed, "language_audit":audit})


# ===========================================================
# 🧠 MAIN PROCESS (LOCAL LIMIT + STYLE GATE + SEO)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}

    # 1. AUDYT STYLOMETRYCZNY (Lokalny)
    language_audit = analyze_language_quality(batch_text)
    burstiness = language_audit.get("burstiness", 0.0)
    fluff = language_audit.get("fluff_ratio", 0.0)
    repeated = language_audit.get("repeated_starts", [])

    # 2. LOCAL STYLE GATE (Szybka weryfikacja przed lub razem z Gemini)
    # Jeśli parametry są tragiczne, odrzucamy od razu jako REJECTED_QUALITY
    if burstiness < 4.0 or fluff > 0.20 or len(repeated) > 2:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Style Gate Failed (Local Metrics)",
            "gemini_feedback": {
                "pass": False,
                "quality_score": 50,
                "feedback_for_writer": (
                    f"CRITICAL STYLE FAIL: Burstiness={burstiness:.1f} (wymagane >5), "
                    f"Fluff={fluff:.2f} (wymagane <0.15), "
                    f"Repeats={len(repeated)}. Popraw strukturę zdań!"
                )
            },
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony przez lokalną bramkę stylu (Burst/Fluff/Repeat)."
        }

    # 3. GEMINI JUDGE (jeśli przeszło wstępną selekcję)
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burstiness, fluff, repeated)
    
    QUALITY_THRESHOLD = 70
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < QUALITY_THRESHOLD:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony przez Gemini (Styl/Jakość)."
        }

    # 4. PRZETWARZANIE SEO + LOCAL BATCH LIMIT
    data = doc.to_dict()
    import copy
    keywords_state = copy.deepcopy(data.get("keywords_state", {}))
    
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]

    batch_local_over = [] # Lista fraz, które przesadziły w TYM batchu

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        
        # --- LOCAL BATCH LIMIT CHECK ---
        # Limit np. 4 użycia na batch dla jednej frazy
        if occurrences > 4:
            batch_local_over.append(f"{original_keyword} ({occurrences}x)")

        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    # Sprawdzenie Local Batch Limit
    if batch_local_over:
        return {
            "status": "REJECTED_SEO",
            "error": "LOCAL_BATCH_LIMIT_EXCEEDED",
            "gemini_feedback": {
                "pass": False,
                "quality_score": gemini_verdict.get("quality_score", 0),
                "feedback_for_writer": (
                    f"LOCAL SEO FAIL: W tym batchu użyłeś fraz zbyt wiele razy: {', '.join(batch_local_over)}. "
                    "Limit to max 4 użycia na batch. Rozbij to na kolejne sekcje."
                )
            },
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony – lokalne przeoptymalizowanie batcha."
        }

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # Global Limits Check
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {
                "pass": False,
                "quality_score": gemini_verdict.get("quality_score", 0),
                "feedback_for_writer": f"SEO CRITICAL: Tekst globalnie przeoptymalizowany! LOCKED={locked}, OVER={over}."
            },
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony przez Hard SEO Veto."
        }

    # Success Save
    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "language_audit": language_audit,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok},
    }
    if "batches" not in data: data["batches"] = []
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state
    doc_ref.set(data)

    readability = language_audit.get("readability", {})
    meta_prompt_summary = (
        f"UNDER={under}, OVER={over}, LOCKED={locked} | "
        f"Quality={gemini_verdict.get('quality_score')}% | "
        f"Sentences={readability.get('sentence_count', 0)}, "
        f"Burst={burstiness:.1f}, Fluff={fluff:.2f}"
    )

    return {
        "status": "BATCH_ACCEPTED",
        "counting_mode": "uuid_hybrid",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "keywords_report": [
            {
                "keyword": meta.get("keyword", "Unknown"),
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": ("INCREASE" if meta["status"] == "UNDER" else "DECREASE" if meta["status"] == "OVER" else "IGNORE"),
            }
            for row_id, meta in sorted(keywords_state.items(), key=lambda item: item[1].get("keyword", ""))
        ],
        "meta_prompt_summary": meta_prompt_summary,
    }

# Endpointy GET bez zmian
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return (jsonify(doc.to_dict()), 200) if doc.exists else (jsonify({"error": "Not found"}), 404)

@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    return (jsonify(doc.to_dict().get("keywords_state", {})), 200) if doc.exists else (jsonify({"error": "Not found"}), 404)
