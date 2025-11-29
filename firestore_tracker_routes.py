import os
import json
import math
import re
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz           # fuzzy-matching (stringowy)
import language_tool_python         # QA językowe
import textstat                     # czytelność
import textdistance                 # drugi fuzzy layer (tokenowy)
import pysbd                        # segmentacja zdań

tracker_routes = Blueprint("tracker_routes", __name__)

# -----------------------------------------------------------
# 🧠 spaCy – ładowane raz (Fallback safe)
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
# 📏 STYLOMETRIA & NLP HELPERS
# ===========================================================

def calculate_burstiness(text: str) -> float:
    """
    Odchylenie standardowe długości zdań.
    Cel: > 6.0 (Ludzie piszą zmiennym rytmem).
    """
    if not text.strip(): return 0.0
    try:
        raw_sentences = SENTENCE_SEGMENTER.segment(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if not sentences: return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    n = len(lengths)
    if n == 0: return 0.0
    
    mean_len = sum(lengths) / n
    variance = sum((l - mean_len) ** 2 for l in lengths) / n
    return math.sqrt(variance)

def calculate_fluff_ratio(text: str) -> float:
    """
    Stosunek przymiotników/przysłówków do wszystkich słów.
    Cel: < 0.15.
    """
    if not text.strip(): return 0.0
    doc = nlp(text)
    adv_adj_count = sum(1 for token in doc if token.pos_ in ("ADJ", "ADV"))
    total_alpha_words = sum(1 for token in doc if token.is_alpha)
    return float(adv_adj_count / total_alpha_words) if total_alpha_words > 0 else 0.0

def calculate_passive_ratio(text: str) -> float:
    """
    Wykrywanie strony biernej (Passive Voice).
    Heurystyka dla PL: Lemma 'zostać' + Imiesłów (ppas / VerbForm=Part).
    Cel: < 0.15.
    """
    if not text.strip(): return 0.0
    doc = nlp(text)
    passive_count = 0
    total_sents = 0
    
    for sent in doc.sents:
        total_sents += 1
        # Szukamy konstrukcji "zostać" ... "zrobiony/napisany"
        has_zostac = any(t.lemma_ == "zostać" for t in sent)
        # W modelu SM tagi bywają różne, szukamy śladów imiesłowu biernego
        has_imieslow = any(
            (t.tag_ and "ppas" in t.tag_) or 
            ("VerbForm=Part" in str(t.morph) and "Voice=Pass" in str(t.morph))
            for t in sent
        )
        if has_zostac and has_imieslow:
            passive_count += 1
            
    return passive_count / total_sents if total_sents > 0 else 0.0

def detect_repeated_sentence_starts(text: str, prefix_words: int = 3) -> list:
    """
    Wykrywa powtórzenia początków zdań (Zero Repetition Policy).
    """
    if not text.strip(): return []
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
    prefix_counts = {}
    for s in sentences:
        words = s.split()
        if len(words) < prefix_words: continue
        prefix = " ".join(words[:prefix_words]).lower()
        if len(prefix) < 5: continue # ignoruj krótkie spójniki
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
    return [{"prefix": p, "count": c} for p, c in prefix_counts.items() if c >= 2]


# ===========================================================
# ⚖️ GEMINI JUDGE (Context & Style Aware)
# ===========================================================
def evaluate_with_gemini(text, meta_trace, burstiness, fluff, passive, repeated, topic=""):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "No API Key"}

    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": "Model Init Error"}

    meta = meta_trace or {}
    intent = meta.get("execution_intent", "Brak")
    rhythm = meta.get("rhythm_pattern_used", "Brak")
    
    repeated_info = f"Wykryto powtórzenia początków: {repeated}" if repeated else "Brak powtórzeń."

    metrics_context = f"""
    DANE STYLOMETRYCZNE:
    - Burstiness: {burstiness:.2f} (Norma > 6.0).
    - Fluff Ratio: {fluff:.3f} (Norma < 0.15).
    - Passive Voice: {passive:.2f} (Norma < 0.15).
    - {repeated_info}
    """

    banned_phrases = "W dzisiejszych czasach, W dobie, Warto zauważyć, Należy wspomnieć, Podsumowując, Reasumując, Gra warta świeczki, Strzał w dziesiątkę, Co więcej."

    prompt = f"""
    Jesteś Surowym Sędzią Jakości SEO (HEAR 2.1).
    Temat artykułu: "{topic}".
    
    INTENCJA: {intent} | RYTM: {rhythm}
    {metrics_context}
    BANNED: {banned_phrases}

    ZASADY DECYZJI:
    1. Merytoryka: Jeśli treść odbiega od tematu "{topic}" -> FAIL.
    2. Styl: Jeśli Burstiness < 4.5 -> OSTRZEŻENIE (Rytm robotyczny).
    3. Styl: Jeśli Fluff > 0.18 -> OSTRZEŻENIE (Za dużo przymiotników).
    4. Styl: Jeśli Passive Voice > 0.20 -> FAIL (Zmień na stronę czynną).
    5. Powtórzenia początków -> FAIL.
    
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
        "lt_issues_count": 0, 
        "burstiness": 0.0, 
        "fluff_ratio": 0.0, 
        "passive_ratio": 0.0,
        "repeated_starts": [],
        "error": None
    }
    if not text.strip(): return result

    if LT_TOOL_PL:
        try:
            matches = LT_TOOL_PL.check(text)
            result["lt_issues_count"] = len(matches)
            result["lt_issues_sample"] = [{"msg": m.message} for m in matches[:5]]
        except Exception as e: result["error"] = str(e)

    try:
        burst = calculate_burstiness(text)
        fluff = calculate_fluff_ratio(text)
        passive = calculate_passive_ratio(text)
        rep = detect_repeated_sentence_starts(text)
        
        raw_s = SENTENCE_SEGMENTER.segment(text)
        sents = [s for s in raw_s if s.strip()]
        
        result.update({
            "burstiness": burst, 
            "fluff_ratio": fluff, 
            "passive_ratio": passive, 
            "repeated_starts": rep,
            "readability": {
                "sentence_count": len(sents),
                "avg_sentence_length": (sum(len(s.split()) for s in sents)/len(sents)) if sents else 0
            }
        })
    except Exception as e: print(f"Audit Error: {e}")

    return result


# ===========================================================
# 🛠 Helpers (Fix & Count)
# ===========================================================
def apply_languagetool_fixes(text: str) -> str:
    if not LT_TOOL_PL or not text: return text
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
    fuzzy = 0
    for start in range(text_len):
        for extra in range(MAX_FUZZY_WINDOW_EXPANSION + 1):
            end = start + target_len + extra
            if end > text_len: break
            if any(pos in used_positions for pos in range(start, end)): continue
            
            win_tok = text_lemma_list[start:end]
            if not win_tok: continue
            
            rf = fuzz.token_set_ratio(kw_str, " ".join(win_tok))
            jac = textdistance.jaccard(kw_tokens_set, set(win_tok))
            
            if rf >= FUZZY_SIMILARITY_THRESHOLD or jac >= JACCARD_SIMILARITY_THRESHOLD:
                fuzzy += 1
                used_positions.update(range(start, end))
                break
    return fuzzy

def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    exact = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
    lemma_hits = 0
    exact_spans = []
    target_tok = target_lemma.split()
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i+target_len] == target_tok:
                lemma_hits += 1
                exact_spans.append((i, i+target_len))
        lemma_hits += _count_fuzzy_on_lemmas(target_tok, text_lemma_list, exact_spans)
    return max(exact, lemma_hits)

def compute_status(actual, target_min, target_max):
    if actual < target_min: return "UNDER"
    if actual > target_max: return "OVER" # Strict Limit (No tolerance)
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
# 🧠 MAIN PROCESS (FULL V8.0 LOGIC)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}
    
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 1. AUDYT STYLOMETRYCZNY (Soft Gate)
    language_audit = analyze_language_quality(batch_text)
    burst = language_audit.get("burstiness", 0.0)
    fluff = language_audit.get("fluff_ratio", 0.0)
    passive = language_audit.get("passive_ratio", 0.0)
    rep = language_audit.get("repeated_starts", [])

    # Awaryjne odrzucenie tylko przy tragicznym stylu
    if burst < 3.5 and fluff > 0.22:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Style Gate Failed (Hard)",
            "gemini_feedback": {"pass": False, "quality_score": 40, "feedback_for_writer": "Tragiczny styl: monotonia i wata słowna."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 2. GEMINI JUDGE (Context Aware)
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burst, fluff, passive, rep, topic=topic)
    
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < 70:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Odrzucone przez Sędziego (Styl/Merytoryka)."
        }

    # 3. SEO TRACKING (Dynamic Limits)
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    batch_local_over = []
    used_h2 = (meta_trace or {}).get("used_h2", [])

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC") 

        # Dynamiczne limity per batch
        if kw_type == "BASIC": local_limit = 4
        elif kw_type == "EXTENDED": local_limit = 6
        else: local_limit = 999 

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, 
                                               meta.get("search_term_exact", ""), 
                                               meta.get("search_lemma", ""))
        
        if occurrences > local_limit:
            batch_local_over.append(f"{original_keyword} ({occurrences}x, limit={local_limit})")

        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    if batch_local_over:
        return {
            "status": "REJECTED_SEO",
            "error": "LOCAL_BATCH_LIMIT_EXCEEDED",
            "gemini_feedback": {
                "pass": False, 
                "quality_score": gemini_verdict.get("quality_score", 0), 
                "feedback_for_writer": f"Przekroczono lokalny limit użycia fraz w tym batchu: {', '.join(batch_local_over)}. Rozbij treść."
            },
            "language_audit": language_audit,
            "quality_alert": True
        }

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {"pass": False, "feedback_for_writer": f"Globalne przeoptymalizowanie! LOCKED={locked}, OVER={over}."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 4. SUKCES & SAVE (TOP_UNDER calculation)
    # Wybieramy top 5 fraz UNDER, które są najdalej od celu
    top_under_list = [
        meta.get("keyword") for _, meta in sorted(
            keywords_state.items(),
            key=lambda item: item[1].get("target_min", 0) - item[1].get("actual_uses", 0),
            reverse=True
        ) if meta["status"] == "UNDER"
    ][:5]

    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "language_audit": language_audit,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok},
        "used_h2": used_h2
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)

    readability = language_audit.get("readability", {})
    meta_prompt_summary = (
        f"UNDER={under}, OVER={over}, LOCKED={locked} | "
        f"Quality={gemini_verdict.get('quality_score')}% | "
        f"Sentences={readability.get('sentence_count', 0)}, "
        f"Burst={burst:.1f}, Fluff={fluff:.2f}, Passive={passive:.2f} | "
        f"TOP_UNDER={', '.join(top_under_list) if top_under_list else 'NONE'}"
    )

    return {
        "status": "BATCH_ACCEPTED",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "meta_prompt_summary": meta_prompt_summary,
        "keywords_report": [] 
    }


# ===========================================================
# 🔍 Endpointy GET
# ===========================================================
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
