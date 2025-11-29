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

# --- INICJALIZACJA (bez zmian) ---
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

FUZZY_SIMILARITY_THRESHOLD = 90
MAX_FUZZY_WINDOW_EXPANSION = 2
JACCARD_SIMILARITY_THRESHOLD = 0.8

try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception:
    LT_TOOL_PL = None

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 📏 Stylometria
# ===========================================================
def calculate_burstiness(text: str) -> float:
    if not text or not text.strip(): return 0.0
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
    if not text.strip(): return 0.0
    doc = nlp(text)
    adv_adj = sum(1 for t in doc if t.pos_ in ("ADJ", "ADV"))
    total = sum(1 for t in doc if t.is_alpha)
    return float(adv_adj / total) if total > 0 else 0.0

def detect_repeated_sentence_starts(text: str, prefix_words: int = 3) -> list:
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
        if len(prefix) < 5: continue
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return [{"prefix": p, "count": c} for p, c in prefix_counts.items() if c >= 2]


# ===========================================================
# ⚖️ GEMINI JUDGE (Soft Gate)
# ===========================================================
def evaluate_with_gemini(text, meta_trace, burstiness, fluff, repeated):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "No API Key"}
    
    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": "Model Init Error"}

    meta = meta_trace or {}
    intent = meta.get("execution_intent", "Brak")
    rhythm = meta.get("rhythm_pattern_used", "Brak")
    repeated_info = f"Wykryto powtórzenia początków zdań: {repeated}" if repeated else "Brak powtórzeń."
    
    metrics_context = f"""
    DANE STYLOMETRYCZNE (FACTUAL DATA):
    - Burstiness: {burstiness:.2f} (Cel > 6.0).
    - Fluff Ratio: {fluff:.3f} (Cel < 0.15).
    - {repeated_info}
    """
    
    # Lista zakazanych fraz (skrócona dla czytelności kodu, w produkcji użyj pełnej)
    banned = "W dzisiejszych czasach, W dobie, Warto zauważyć, Należy wspomnieć, Podsumowując, Reasumując, Gra warta świeczki, Strzał w dziesiątkę."

    prompt = f"""
    Jesteś Surowym Sędzią Jakości SEO (HEAR 2.1).
    INTENCJA: {intent} | RYTM: {rhythm}
    {metrics_context}
    BANNED: {banned}

    ZASADY:
    1. Burstiness < 4.5 -> OSTRZEŻENIE (Rytm robotyczny).
    2. Fluff > 0.18 -> OSTRZEŻENIE (Za dużo przymiotników).
    3. Powtórzenia początków -> FAIL (pass: false).
    4. Banned Phrases -> FAIL.
    
    Zwróć JSON: {{ "pass": true/false, "quality_score": (0-100), "feedback_for_writer": "..." }}
    TEKST: "{text}"
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": f"API Error: {e}"}

# ===========================================================
# 🔍 Helpers (LanguageTool, Fuzzy, Hybrid) - bez zmian
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    result = {"lt_issues_count": 0, "burstiness": 0.0, "fluff_ratio": 0.0, "repeated_starts": []}
    if not text.strip(): return result
    
    if LT_TOOL_PL:
        try:
            matches = LT_TOOL_PL.check(text)
            result["lt_issues_count"] = len(matches)
        except: pass

    try:
        burst = calculate_burstiness(text)
        fluff = calculate_fluff_ratio(text)
        rep = detect_repeated_sentence_starts(text)
        result.update({"burstiness": burst, "fluff_ratio": fluff, "repeated_starts": rep})
        
        raw_s = SENTENCE_SEGMENTER.segment(text)
        sents = [s for s in raw_s if s.strip()]
        result["readability"] = {
            "sentence_count": len(sents),
            "avg_sentence_length": (sum(len(s.split()) for s in sents)/len(sents)) if sents else 0
        }
    except: pass
    return result

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
    # (Kod identyczny jak wcześniej - skrócony tutaj dla czytelności)
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
            win_tokens = text_lemma_list[start:end]
            if not win_tokens: continue
            rf = fuzz.token_set_ratio(kw_str, " ".join(win_tokens))
            jac = textdistance.jaccard(kw_tokens_set, set(win_tokens))
            if rf >= FUZZY_SIMILARITY_THRESHOLD or jac >= JACCARD_SIMILARITY_THRESHOLD:
                fuzzy += 1
                used_positions.update(range(start, end))
                break
    return fuzzy

def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    exact_hits = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
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

# --- STRICT STATUS (Bez tolerancji) ---
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
# 🆕 ENDPOINT
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    auto_fixed = apply_languagetool_fixes(text)
    audit = analyze_language_quality(auto_fixed)
    return jsonify({"original_text":text, "auto_fixed_text":auto_fixed, "language_audit":audit})

# ===========================================================
# 🧠 MAIN PROCESS
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}

    # 1. SOFT STYLE GATE
    language_audit = analyze_language_quality(batch_text)
    burst = language_audit.get("burstiness", 0.0)
    fluff = language_audit.get("fluff_ratio", 0.0)
    rep = language_audit.get("repeated_starts", [])

    # "Hard Fail" tylko przy tragicznym combo
    if burst < 3.5 and fluff > 0.22:
        return {
            "status": "REJECTED_QUALITY", 
            "error": "Style Gate Failed (Hard)",
            "gemini_feedback": {"pass": False, "quality_score": 40, "feedback_for_writer": "Tragiczny styl: monotonia + wata."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 2. GEMINI JUDGE
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burst, fluff, rep)
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < 70:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 3. SEO + DYNAMIC LIMITS
    data = doc.to_dict()
    import copy
    keywords_state = copy.deepcopy(data.get("keywords_state", {}))
    
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    batch_local_over = []
    
    # Obsługa H2-aware
    used_h2 = (meta_trace or {}).get("used_h2", [])

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC") # Domyślnie BASIC

        # Dynamiczny limit
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
                "feedback_for_writer": f"Lokalny limit przekroczony dla: {', '.join(batch_local_over)}. Rozbij treść."
            },
            "language_audit": language_audit,
            "quality_alert": True
        }

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {"pass": False, "feedback_for_writer": f"Globalny limit SEO! LOCKED={locked}."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 4. SAVE & SUMMARY (z TOP_UNDER)
    # Wybieramy top 5 najbardziej "niedoładowanych" fraz
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
        f"Burst={burst:.1f}, Fluff={fluff:.2f} | "
        f"TOP_UNDER={', '.join(top_under_list) if top_under_list else 'NONE'}"
    )

    return {
        "status": "BATCH_ACCEPTED",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "meta_prompt_summary": meta_prompt_summary,
        "keywords_report": [] # Opcjonalnie, żeby nie zapychać
    }

# Endpointy GET bez zmian (omijam dla skrótu)
