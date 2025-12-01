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
# 👮‍♂️ HARD GUARDRAILS (Szeryf w Pythonie) - NOWOŚĆ
# ===========================================================
def validate_hard_rules(text: str) -> dict:
    """
    Bezwzględne reguły strukturalne. Jeśli tekst je łamie,
    zostaje odrzucony ZANIM trafi do Gemini czy Trackera.
    """
    errors = []

    # 1. ZAKAZ LIST WYPUNKTOWANYCH
    # Szukamy linii zaczynających się od "- ", "* ", "1. "
    # (Ignorujemy nagłówki markdown ##)
    lines = text.split('\n')
    bullet_count = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[\-\*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            bullet_count += 1
    
    if bullet_count > 0:
        errors.append(f"WYKRYTO LISTĘ ({bullet_count} pkt). Zakaz punktorów/wyliczeń. Użyj ciągłej narracji.")

    # 2. ZAKAZ 'WARTO' NA POCZĄTKU ZDANIA
    # Szukamy "Warto..." po kropce lub na początku tekstu
    if re.search(r'(^|[\.\?!]\s+)Warto\s', text):
        errors.append("WYKRYTO 'WARTO...' na początku zdania. Zakazane otwarcie.")

    # 3. ASYMETRIA AKAPITÓW (Dla batchy z wieloma H2)
    # Dzielimy tekst na sekcje H2 (##)
    sections = re.split(r'##\s+', text)
    sections = [s for s in sections if len(s.strip()) > 50] # Ignorujemy wstęp/puste
    
    if len(sections) >= 2:
        paragraph_counts = []
        for sec in sections:
            # Liczymy akapity (bloki tekstu oddzielone pustą linią)
            paras = [p for p in sec.split('\n\n') if len(p.strip()) > 20]
            paragraph_counts.append(len(paras))
        
        # Jeśli wszystkie sekcje mają tyle samo akapitów (i więcej niż 1 sekcja) -> BŁĄD
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
# 📏 STYLOMETRIA & NLP HELPERS
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


# ===========================================================
# 🧠 SEMANTYKA
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
    dot_product = np.dot(vec_text, vec_topic)
    norm_a = np.linalg.norm(vec_text)
    norm_b = np.linalg.norm(vec_topic)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(dot_product / (norm_a * norm_b))


# ===========================================================
# ⚖️ GEMINI JUDGE
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
    
    metrics_context = f"""
    DANE TECHNICZNE:
    - Burstiness: {burst:.2f} (Norma > 6.0).
    - Fluff Ratio: {fluff:.3f} (Norma < 0.15).
    - Passive Voice: {passive:.2f} (Norma < 0.15).
    - Semantic: {semantic_score:.2f} (Norma > 0.75).
    """

    prompt = f"""
    Jesteś Głównym Redaktorem Medyczno-Technicznym oraz Sędzią SEO.
    Temat: "{topic}".
    
    Twoim zadaniem jest ocena fragmentu pod kątem:
    1. BEZPIECZEŃSTWA (Fact-Checking dla tematów YMYL - zdrowie/finanse).
    2. LOGIKI (Brak powtórzeń informacji).

    {metrics_context}

    ZASADY OCENY:
    A. MERYTORYKA (YMYL): Jeśli tekst zawiera szkodliwe porady medyczne/finansowe -> FAIL (Score 0).
    B. LOGIKA: Jeśli tekst mieli w kółko to samo -> FAIL.
    C. STYL: Jeśli parametry techniczne są tragiczne -> OSTRZEŻENIE.

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
    result = {"lt_issues_count": 0, "burstiness": 0.0, "fluff_ratio": 0.0, "passive_ratio": 0.0, "repeated_starts": [], "banned_detected": []}
    if not text.strip(): return result

    try:
        burst = calculate_burstiness(text)
        fluff = calculate_fluff_ratio(text)
        passive = calculate_passive_ratio(text)
        rep = detect_repeated_sentence_starts(text)
        
        from rapidfuzz import fuzz
        banned_phrases = ["warto zauważyć", "w dzisiejszych czasach", "podsumowując", "reasumując", "gra warta świeczki"]
        banned_found = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            for b in banned_phrases:
                if fuzz.partial_token_set_ratio(b, s.lower()) > 85:
                    banned_found.append(b)
            if "—" in s:
                banned_found.append("EM-DASH (—)")

        banned_found = list(set(banned_found))
        result.update({"burstiness": burst, "fluff_ratio": fluff, "passive_ratio": passive, "repeated_starts": rep, "banned_detected": banned_found})
    except: pass
    return result

def apply_languagetool_fixes(text: str) -> str:
    # Uproszczone w tej wersji, żeby skupić się na hard rules
    return text


# ... (Fuzzy helpers - bez zmian) ...
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
    target_tok = target_lemma.split()
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i+target_len] == target_tok:
                lemma_hits += 1
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
# 🆕 ENDPOINT (Language Refine + HARD RULES)
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    
    # 1. Najpierw Sanityzacja (myślniki)
    clean_text = sanitize_typography(text)
    
    # 2. HARD GUARDRAILS CHECK (Przed audytem)
    # Jeśli tekst ma listy lub "Warto...", odrzucamy natychmiast.
    hard_check = validate_hard_rules(clean_text)
    
    # Audyt dla informacji (nawet jak fail, chcemy widzieć liczby)
    audit = analyze_language_quality(clean_text)
    
    # Jeśli hard rules złamane -> dodajemy flagę błędu do audytu
    if not hard_check["valid"]:
        audit["HARD_RULE_FAIL"] = hard_check["msg"]

    return jsonify({"original_text":text, "auto_fixed_text":clean_text, "language_audit":audit})


# ===========================================================
# 🧠 MAIN PROCESS (V11.7 with Hard Guardrails)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}
    
    batch_text = sanitize_typography(batch_text)
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 0. HARD GUARDRAILS (Ostatnia linia obrony)
    # Jeśli model zignorował ostrzeżenie z language_refine i próbuje zapisać syf -> BLOKUJEMY.
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
            "quality_alert": True
        }

    # 1. AUDYT (Standardowy)
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
        return {"status": "REJECTED_QUALITY", "error": "Banned Phrases", "gemini_feedback": {"pass": False, "feedback_for_writer": f"Zakazane: {banned}"}, "language_audit": language_audit, "quality_alert": True}

    # 2. GEMINI (Sędzia Merytoryczny)
    gemini_verdict = evaluate_with_gemini(batch_text, meta_trace, burst, fluff, passive, rep, banned, semantic_score, topic=topic)
    if not gemini_verdict.get("pass", True) or gemini_verdict.get("quality_score", 100) < 70:
        return {"status": "REJECTED_QUALITY", "error": "Gemini Rejection", "gemini_feedback": gemini_verdict, "language_audit": language_audit, "quality_alert": True}

    # 3. SEO TRACKING & GLOBAL CEILING (v11.6 Anti-Stuffing)
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

        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        
        if occurrences > 0:
            if kw_type == "BASIC": batch_basic_hits += occurrences
            elif kw_type == "EXTENDED": batch_extended_hits += occurrences
            
            # GLOBAL CEILING
            if current_global_usage >= target_max:
                 global_ceiling_hit.append(f"{original_keyword} (MAX: {target_max})")

        if occurrences > local_limit:
            batch_local_over.append(f"{original_keyword} ({occurrences}x)")

        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    if global_ceiling_hit:
        return {"status": "REJECTED_SEO", "error": "GLOBAL_CEILING", "gemini_feedback": {"pass": False, "feedback_for_writer": f"STOP! Frazy już zrealizowane: {', '.join(global_ceiling_hit[:3])}. Usuń je."}, "language_audit": language_audit, "quality_alert": True}

    if batch_local_over:
        return {"status": "REJECTED_SEO", "error": "LOCAL_LIMIT", "gemini_feedback": {"pass": False, "feedback_for_writer": f"Limit lokalny: {batch_local_over}"}, "language_audit": language_audit, "quality_alert": True}

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

    return {
        "status": "BATCH_ACCEPTED",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "meta_prompt_summary": meta_prompt_summary,
        "keywords_report": light_report 
    }

# GET endpoints bez zmian (omijam dla skrótu, są w starym kodzie)
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
