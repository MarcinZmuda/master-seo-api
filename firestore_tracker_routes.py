import os
import json
import math
import re
import numpy as np
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz  # Fuzzy matching
import language_tool_python
import textstat
import textdistance
import pysbd

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INICJALIZACJA ---
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

FUZZY_SIMILARITY_THRESHOLD = 90      
MAX_FUZZY_WINDOW_EXPANSION = 2       
JACCARD_SIMILARITY_THRESHOLD = 0.8   

# Konfiguracja progu dla wykrywania zakazanych fraz (85 łapie "warto jednak zauważyć")
BANNED_FUZZY_THRESHOLD = 85 

try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception:
    LT_TOOL_PL = None

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- BAZA ZAKAZANYCH FRAZ (HEAR 2.1) ---
BANNED_PHRASES_DB = [
    "w dzisiejszych czasach", "w dobie", "od zarania dziejów",
    "warto zauważyć", "należy wspomnieć", "warto dodać", "co więcej",
    "podsumowując", "reasumując", "w ostatecznym rozrachunku", "konkludując",
    "gra warta świeczki", "strzał w dziesiątkę", "klucz do sukcesu", "wisienka na torcie",
    "wszystko zależy od", "nie ma jednoznacznej odpowiedzi",
    "nie ma wątpliwości", "bez wątpienia", "niezwykle ważny"
]


# ===========================================================
# 🕵️‍♀️ DETEKCJA ZAKAZANYCH FRAZ (FUZZY + METAPHOR)
# ===========================================================
def detect_banned_fuzzy(text: str) -> list:
    """
    Sprawdza występowanie zakazanych fraz w trybie Fuzzy Token Set.
    Wykrywa:
    - "Warto zauważyć" w "Warto jednak zauważyć" (wtrącenia).
    - "Gra warta świeczki" w "Gra nie była warta świeczki" (zmiany).
    """
    if not text.strip(): return []
    
    found_issues = []
    # Analizujemy zdanie po zdaniu, żeby nie robić false positives na całym tekście
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in sentences:
        sent_lower = sent.lower()
        for banned in BANNED_PHRASES_DB:
            # Używamy partial_token_set_ratio:
            # - Ignoruje kolejność słów
            # - Ignoruje "szum" (dodatkowe słowa wtrącone)
            score = fuzz.partial_token_set_ratio(banned, sent_lower)
            
            if score >= BANNED_FUZZY_THRESHOLD:
                # Sprawdzamy czy to nie false positive (np. bardzo krótkie słowa)
                if len(banned) < 5 and score < 100: continue
                
                found_issues.append(f"'{banned}' (match: {score}%)")

    # Usuwamy duplikaty
    return list(set(found_issues))


# ===========================================================
# 🧠 SEMANTYKA (Embeddings)
# ===========================================================
def get_embedding(text):
    if not text or not text.strip(): return None
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def calculate_semantic_score(batch_text, main_topic):
    if not batch_text or not main_topic: return 1.0
    vec_text = get_embedding(batch_text)
    vec_topic = get_embedding(main_topic)
    if not vec_text or not vec_topic: return 1.0
    
    dot = np.dot(vec_text, vec_topic)
    norm_a = np.linalg.norm(vec_text)
    norm_b = np.linalg.norm(vec_topic)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(dot / (norm_a * norm_b))


# ===========================================================
# 📏 STYLOMETRIA
# ===========================================================
def calculate_burstiness(text: str) -> float:
    if not text.strip(): return 0.0
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences: return 0.0
    lengths = [len(s.split()) for s in sentences]
    n = len(lengths)
    if n == 0: return 0.0
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / n
    return math.sqrt(variance)

def calculate_fluff_ratio(text: str) -> float:
    if not text.strip(): return 0.0
    doc = nlp(text)
    adv_adj = sum(1 for t in doc if t.pos_ in ("ADJ", "ADV"))
    total = sum(1 for t in doc if t.is_alpha)
    return float(adv_adj / total) if total > 0 else 0.0

def calculate_passive_ratio(text: str) -> float:
    if not text.strip(): return 0.0
    doc = nlp(text)
    passive = 0
    total = 0
    for sent in doc.sents:
        total += 1
        has_zostac = any(t.lemma_ == "zostać" for t in sent)
        has_imieslow = any(("VerbForm=Part" in str(t.morph) and "Voice=Pass" in str(t.morph)) for t in sent)
        if has_zostac and has_imieslow: passive += 1
    return passive / total if total > 0 else 0.0

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

def filter_repeated_starts_against_keywords(repeated_starts, keywords_state):
    if not repeated_starts or not keywords_state: return repeated_starts
    keyword_strings = [meta.get("keyword", "").lower() for meta in keywords_state.values() if meta.get("keyword")]
    filtered = []
    for item in repeated_starts:
        if any(kw in item["prefix"] for kw in keyword_strings): continue
        filtered.append(item)
    return filtered


# ===========================================================
# ⚖️ GEMINI JUDGE (Z Obsługą Fuzzy Banned)
# ===========================================================
def evaluate_with_gemini(text, meta_trace, burst, fluff, passive, repeated, banned_detected, semantic_score, topic=""):
    if not GEMINI_API_KEY:
        return {"pass": True, "quality_score": 100, "feedback_for_writer": "No API Key"}

    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except:
        return {"pass": True, "quality_score": 80, "feedback_for_writer": "Model Init Error"}

    meta = meta_trace or {}
    intent = meta.get("execution_intent", "Brak")
    rhythm = meta.get("rhythm_pattern_used", "Brak")
    
    repeated_info = f"Powtórzenia początków: {repeated}" if repeated else "Brak powtórzeń."
    banned_info = f"⚠️ WYKRYTO ZAKAZANE FRAZY (Fuzzy Match): {banned_detected}" if banned_detected else "Brak zakazanych fraz (Exact/Fuzzy)."

    metrics_context = f"""
    DANE TECHICZNE (FACTUAL DATA):
    - Burstiness: {burst:.2f} (Norma > 6.0).
    - Fluff Ratio: {fluff:.3f} (Norma < 0.15).
    - Passive Voice: {passive:.2f} (Norma < 0.15).
    - Semantic Score: {semantic_score:.2f} (Norma > 0.75).
    - {repeated_info}
    - {banned_info}
    """

    prompt = f"""
    Jesteś Surowym Sędzią Jakości SEO (HEAR 2.1).
    Temat: "{topic}".
    
    INTENCJA: {intent}
    {metrics_context}

    ZASADY DECYZJI:
    1. BANNED PHRASES: Jeśli 'banned_detected' nie jest puste -> FAIL. Komentarz: "Usuń zakazane frazy/metafory."
    2. METAFORY: Jeśli widzisz metafory typu "strzał w 10", "złoty środek" (nawet jeśli skrypt ich nie wyłapał) -> FAIL.
    3. STYL: Burstiness < 4.5 lub Fluff > 0.18 -> OSTRZEŻENIE.
    4. SEMANTYKA: Off-topic -> FAIL.
    
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
# 🔎 Language Audit
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    result = {
        "lt_issues_count": 0, "burstiness": 0.0, "fluff_ratio": 0.0, 
        "passive_ratio": 0.0, "repeated_starts": [], "banned_detected": []
    }
    if not text.strip(): return result

    if LT_TOOL_PL:
        try:
            matches = LT_TOOL_PL.check(text)
            result["lt_issues_count"] = len(matches)
            result["lt_issues_sample"] = [{"msg": m.message} for m in matches[:5]]
        except: pass

    try:
        burst = calculate_burstiness(text)
        fluff = calculate_fluff_ratio(text)
        passive = calculate_passive_ratio(text)
        rep = detect_repeated_sentence_starts(text)
        # NOWOŚĆ: Wykrywanie rozmytych zakazanych fraz
        banned = detect_banned_fuzzy(text)
        
        raw_s = SENTENCE_SEGMENTER.segment(text)
        sents = [s for s in raw_s if s.strip()]
        
        result.update({
            "burstiness": burst, 
            "fluff_ratio": fluff, 
            "passive_ratio": passive, 
            "repeated_starts": rep,
            "banned_detected": banned, # Dodajemy do raportu
            "readability": {
                "sentence_count": len(sents),
                "avg_sentence_length": (sum(len(s.split()) for s in sents)/len(sents)) if sents else 0
            }
        })
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

# ... (Funkcje fuzzy liczenia słów: _count_fuzzy_on_lemmas - bez zmian) ...
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
            if any(pos in used_positions for pos in range(start, end)): continue
            win_tok = text_lemma_list[start:end]
            if not win_tok: continue
            rf = fuzz.token_set_ratio(kw_str, " ".join(win_tok))
            jac = textdistance.jaccard(kw_tokens_set, set(win_tok))
            if rf >= FUZZY_SIMILARITY_THRESHOLD or jac >= JACCARD_SIMILARITY_THRESHOLD:
                fuzzy_hits += 1
                used_positions.update(range(start, end))
                break
    return fuzzy_hits

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
    
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 1. AUDYT JĘZYKOWY + BANNED FUZZY
    language_audit = analyze_language_quality(batch_text)
    burst = language_audit.get("burstiness", 0.0)
    fluff = language_audit.get("fluff_ratio", 0.0)
    passive = language_audit.get("passive_ratio", 0.0)
    rep = language_audit.get("repeated_starts", [])
    rep = filter_repeated_starts_against_keywords(rep, project_data.get("keywords_state", {}))
    language_audit["repeated_starts"] = rep
    banned = language_audit.get("banned_detected", [])

    semantic_score = calculate_semantic_score(batch_text, topic)
    language_audit["semantic_score"] = semantic_score

    # Hard Fail jeśli znaleziono zakazane frazy
    if banned:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Banned Phrases Detected",
            "gemini_feedback": {
                "pass": False, 
                "quality_score": 0, 
                "feedback_for_writer": f"Wykryto zakazane frazy (lub ich wariacje): {', '.join(banned)}. Usuń je lub zamień na konkrety."
            },
            "language_audit": language_audit,
            "quality_alert": True
        }

    # Hard Fail przy tragicznym stylu
    if burst < 3.5 and fluff > 0.22:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Style Gate Failed (Hard)",
            "gemini_feedback": {"pass": False, "quality_score": 40, "feedback_for_writer": "Tragiczny styl: monotonia i wata słowna."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 2. GEMINI JUDGE (przekazujemy listę banned)
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burst, fluff, passive, rep, banned, semantic_score, topic=topic)
    
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < 70:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 3. SEO TRACKING (Bez zmian)
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    batch_local_over = []
    used_h2 = (meta_trace or {}).get("used_h2", [])

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC") 
        local_limit = 4 if kw_type == "BASIC" else 6 if kw_type == "EXTENDED" else 999

        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        if occurrences > local_limit:
            batch_local_over.append(f"{original_keyword} ({occurrences}x)")
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    if batch_local_over:
        return {
            "status": "REJECTED_SEO",
            "error": "LOCAL_BATCH_LIMIT_EXCEEDED",
            "gemini_feedback": {"pass": False, "feedback_for_writer": f"Limit lokalny: {', '.join(batch_local_over)}."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    under, over, locked, ok = global_keyword_stats(keywords_state)
    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {"pass": False, "feedback_for_writer": f"Globalny limit: LOCKED={locked}."},
            "language_audit": language_audit,
            "quality_alert": True
        }

    # 4. SAVE
    top_under_list = [m.get("keyword") for _, m in sorted(keywords_state.items(), key=lambda i: i[1].get("target_min", 0)-i[1].get("actual_uses", 0), reverse=True) if m["status"]=="UNDER"][:5]

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
        f"UNDER={under}, OVER={over} | "
        f"Quality={gemini_verdict.get('quality_score')}% | "
        f"Burst={burst:.1f}, Fluff={fluff:.2f}, Passive={passive:.2f} | "
        f"BANNED={len(banned)} | "
        f"TOP_UNDER={', '.join(top_under_list) if top_under_list else 'NONE'}"
    )
    
    light_report = [{"keyword": m.get("keyword"), "status": m["status"]} for _, m in keywords_state.items() if m["status"] in ("UNDER", "OVER")]

    return {
        "status": "BATCH_ACCEPTED",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "meta_prompt_summary": meta_prompt_summary,
        "keywords_report": light_report 
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
