import os
import json
import math
import re
import numpy as np
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

try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception:
    LT_TOOL_PL = None

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 👮‍♂️ HARD GUARDRAILS (Szeryf w Pythonie) - ZACHOWANE 1:1
# ===========================================================
def validate_hard_rules(text: str) -> dict:
    errors = []
    lines = text.split('\n')
    bullet_count = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[\-\*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            bullet_count += 1
    
    if bullet_count > 0:
        errors.append(f"WYKRYTO LISTĘ ({bullet_count} pkt). Zakaz punktorów/wyliczeń. Użyj ciągłej narracji.")

    if re.search(r'(^|[\.\?!]\s+)Warto\s', text):
        errors.append("WYKRYTO 'WARTO...' na początku zdania. Zakazane otwarcie.")

    sections = re.split(r'##\s+', text)
    sections = [s for s in sections if len(s.strip()) > 50]
    
    if len(sections) >= 2:
        paragraph_counts = []
        for sec in sections:
            paras = [p for p in sec.split('\n\n') if len(p.strip()) > 20]
            paragraph_counts.append(len(paras))
        
        if len(set(paragraph_counts)) == 1 and len(paragraph_counts) > 1:
            errors.append(f"MONOTONNA STRUKTURA: Wszystkie sekcje mają po {paragraph_counts[0]} akapity. Wymagana asymetria (np. 1 vs 3).")

    if errors:
        return {"valid": False, "msg": " | ".join(errors)}
    
    return {"valid": True, "msg": "OK"}


# ===========================================================
# 🧹 SANITIZER
# ===========================================================
def sanitize_typography(text: str) -> str:
    if not text: return ""
    text = text.replace("—", " – ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===========================================================
# 📏 STYLOMETRIA & NLP HELPERS (ZACHOWANE)
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
    mean_len = sum(lengths) / n
    variance = sum((l - mean_len) ** 2 for l in lengths) / n
    return math.sqrt(variance)

def calculate_fluff_ratio(text: str) -> float:
    if not text.strip(): return 0.0
    doc = nlp(text)
    adv_adj_count = sum(1 for token in doc if token.pos_ in ("ADJ", "ADV"))
    total_alpha_words = sum(1 for token in doc if token.is_alpha)
    return float(adv_adj_count / total_alpha_words) if total_alpha_words > 0 else 0.0

def calculate_passive_ratio(text: str) -> float:
    if not text.strip(): return 0.0
    doc = nlp(text)
    passive_count = 0
    total_sents = 0
    for sent in doc.sents:
        total_sents += 1
        has_zostac = any(t.lemma_ == "zostać" for t in sent)
        has_imieslow = any(
            (t.tag_ and "ppas" in t.tag_) or 
            ("VerbForm=Part" in str(t.morph) and "Voice=Pass" in str(t.morph))
            for t in sent
        )
        if has_zostac and has_imieslow:
            passive_count += 1
    return passive_count / total_sents if total_sents > 0 else 0.0

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

# ... (Semantic Score & Embedding - ZACHOWANE) ...
def get_embedding(text):
    if not text or not text.strip(): return None
    try:
        result = genai.embed_content(
            model="models/text-embedding-004", content=text, task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e: return None

def calculate_semantic_score(batch_text, main_topic):
    if not batch_text or not main_topic: return 1.0
    vec_text = get_embedding(batch_text)
    vec_topic = get_embedding(main_topic)
    if not vec_text or not vec_topic: return 1.0
    dot_product = np.dot(vec_text, vec_topic)
    norm_a = np.linalg.norm(vec_text)
    norm_b = np.linalg.norm(vec_topic)
    return float(dot_product / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

# ... (Gemini Judge - ZACHOWANE) ...
def evaluate_with_gemini(text, meta_trace, burst, fluff, passive, repeated, banned_detected, semantic_score, topic=""):
    if not GEMINI_API_KEY: return {"pass": True, "quality_score": 100}
    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except: return {"pass": True, "quality_score": 80, "feedback": "Init Error"}

    metrics_context = f"Burst: {burst:.2f}, Fluff: {fluff:.3f}, Semantic: {semantic_score:.2f}"
    prompt = f"""
    Sędzia SEO. Temat: "{topic}".
    {metrics_context}
    Sprawdź: Bezpieczeństwo (YMYL), Logikę, Styl.
    Zwróć JSON: {{ "pass": true/false, "quality_score": 0-100, "feedback_for_writer": "..." }}
    TEKST: "{text}"
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception: return {"pass": True, "quality_score": 80}

# ... (Language Audit - ZACHOWANE) ...
def analyze_language_quality(text: str) -> dict:
    result = {"lt_issues_count": 0, "burstiness": 0.0, "fluff_ratio": 0.0, "passive_ratio": 0.0, "repeated_starts": [], "banned_detected": []}
    if not text.strip(): return result
    try:
        burst = calculate_burstiness(text)
        fluff = calculate_fluff_ratio(text)
        passive = calculate_passive_ratio(text)
        rep = detect_repeated_sentence_starts(text)
        from rapidfuzz import fuzz
        banned_phrases = ["warto zauważyć", "w dzisiejszych czasach", "podsumowując", "reasumując"]
        banned_found = []
        for s in re.split(r'(?<=[.!?])\s+', text):
            for b in banned_phrases:
                if fuzz.partial_token_set_ratio(b, s.lower()) > 85: banned_found.append(b)
            if "—" in s: banned_found.append("EM-DASH (—)")
        result.update({"burstiness": burst, "fluff_ratio": fluff, "passive_ratio": passive, "repeated_starts": rep, "banned_detected": list(set(banned_found))})
    except: pass
    return result

# ... (Fuzzy helpers - ZACHOWANE) ...
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    exact = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
    lemma_hits = 0
    target_tok = target_lemma.split()
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i+target_len] == target_tok: lemma_hits += 1
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
# 🆕 ENDPOINT: REFINE + AUTO-FIX (Ulepszony!)
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    
    clean_text = sanitize_typography(text)
    
    # 1. Hard Check (Czy w ogóle warto poprawiać?)
    hard_check = validate_hard_rules(clean_text)
    
    # 2. Audyt
    audit = analyze_language_quality(clean_text)
    if not hard_check["valid"]:
        audit["HARD_RULE_FAIL"] = hard_check["msg"]
    
    # 3. AUTO-FIX (Nowość V12)
    # Jeśli Burstiness jest dramatycznie niskie (< 4.5) i mamy klucz Gemini,
    # próbujemy to naprawić automatycznie.
    fixed_text = clean_text
    if GEMINI_API_KEY and audit["burstiness"] > 0 and audit["burstiness"] < 4.5:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash") # Szybki model
            prompt = f"""
            Przeredaguj ten tekst, aby zwiększyć różnorodność długości zdań (Burstiness).
            Mieszaj zdania bardzo krótkie (3-5 słów) z bardzo długimi.
            Zachowaj sens i słowa kluczowe.
            Tekst: {clean_text}
            """
            resp = model.generate_content(prompt)
            if resp.text:
                fixed_text = resp.text.strip()
                # Aktualizujemy audyt dla nowego tekstu
                audit = analyze_language_quality(fixed_text)
        except Exception:
            pass # Jeśli fail, zwracamy oryginał

    return jsonify({
        "original_text": text,
        "auto_fixed_text": fixed_text,
        "language_audit": audit
    })


# ===========================================================
# 🧠 MAIN PROCESS (V12.0 - Z obsługą Next Action)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}
    
    batch_text = sanitize_typography(batch_text)
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 0. HARD GUARDRAILS
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "HARD RULE VIOLATION",
            "gemini_feedback": {
                "pass": False,
                "quality_score": 0,
                "feedback_for_writer": f"CRITICAL STRUCTURE FAIL: {hard_check['msg']}. POPRAW TO."
            },
            "language_audit": {},
            "quality_alert": True,
            "next_action": "REWRITE" # V12
        }

    # 1. AUDYT
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

    if banned:
        return {"status": "REJECTED_QUALITY", "error": "Banned Phrases", "gemini_feedback": {"pass": False, "feedback_for_writer": f"Zakazane: {banned}"}, "language_audit": language_audit, "quality_alert": True, "next_action": "REWRITE"}

    # 2. GEMINI JUDGE
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burst, fluff, passive, rep, banned, semantic_score, topic=topic)
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < 70:
        return {"status": "REJECTED_QUALITY", "error": "Gemini Rejection", "gemini_feedback": gemini_verdict, "language_audit": language_audit, "quality_alert": True, "next_action": "REWRITE"}

    # 3. SEO TRACKING
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    batch_local_over = []
    global_ceiling_hit = []
    batch_basic_hits = 0
    batch_extended_hits = 0

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC") 
        local_limit = 4 if kw_type == "BASIC" else 6 
        current_global_usage = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 5)
        
        # Hybrydowe liczenie
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        
        if occurrences > 0:
            if kw_type == "BASIC": batch_basic_hits += occurrences
            elif kw_type == "EXTENDED": batch_extended_hits += occurrences
            if current_global_usage >= target_max:
                 global_ceiling_hit.append(f"{original_keyword} (MAX: {target_max})")

        if occurrences > local_limit:
            batch_local_over.append(f"{original_keyword} ({occurrences}x)")

        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    if global_ceiling_hit:
        return {"status": "REJECTED_SEO", "error": "GLOBAL_CEILING", "gemini_feedback": {"pass": False, "feedback_for_writer": f"STOP! Frazy już zrealizowane: {', '.join(global_ceiling_hit[:3])}. Usuń je."}, "language_audit": language_audit, "quality_alert": True, "next_action": "REWRITE"}

    if batch_local_over:
        return {"status": "REJECTED_SEO", "error": "LOCAL_LIMIT", "gemini_feedback": {"pass": False, "feedback_for_writer": f"Limit lokalny: {batch_local_over}"}, "language_audit": language_audit, "quality_alert": True, "next_action": "REWRITE"}

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # 4. SAVE
    top_under_list = [m.get("keyword") for _, m in sorted(keywords_state.items(), key=lambda i: i[1].get("target_min", 0)-i[1].get("actual_uses", 0), reverse=True) if m["status"]=="UNDER"][:5]

    batch_entry = {
        "text": batch_text, "gemini_audit": gemini_verdict, "language_audit": language_audit,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok},
        "used_h2": (meta_trace or {}).get("used_h2", [])
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)

    meta_prompt_summary = (
        f"UNDER={under}, OVER={over} | "
        f"Quality={gemini_verdict.get('quality_score')}% | "
        f"Burst={burst:.1f}, Fluff={fluff:.2f} | "
        f"TYPES: BASIC={batch_basic_hits}, EXT={batch_extended_hits} | "
        f"TOP_UNDER={', '.join(top_under_list) if top_under_list else 'NONE'}"
    )
    
    light_report = [{"keyword": m.get("keyword"), "status": m["status"]} for _, m in keywords_state.items() if m["status"] in ("UNDER", "OVER")]

    # 5. DECYZJA O NASTĘPNYM KROKU (V12 Logic)
    next_act = "GENERATE_NEXT"
    # Jeśli mamy dużo batchy lub wszystkie słowa są OK/OVER -> Sugeruj koniec
    if under == 0 and len(project_data["batches"]) >= 3:
        next_act = "EXPORT"
    
    return {
        "status": "BATCH_ACCEPTED",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "meta_prompt_summary": meta_prompt_summary,
        "keywords_report": light_report,
        "next_action": next_act # V12
    }
