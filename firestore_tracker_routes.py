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
import datetime # ADDED THIS IMPORT

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INICJALIZACJA ---
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

# Parametry silnika (zgodne z dok. v7.6.0)
FUZZY_SIMILARITY_THRESHOLD = 90      
MAX_FUZZY_WINDOW_EXPANSION = 2       
JACCARD_SIMILARITY_THRESHOLD = 0.8   

# Inicjalizacja LanguageTool
try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception as e:
    print(f"⚠️ LanguageTool Init Error: {e}")
    LT_TOOL_PL = None

# Konfiguracja TextStat dla PL
textstat.set_lang("pl")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 👮‍♂️ HARD GUARDRAILS (Tylko struktura)
# ===========================================================
def validate_hard_rules(text: str) -> dict:
    errors = []
    # Zakaz list punktowanych w narracji
    if re.search(r'^[\-\*]\s+', text, re.MULTILINE) or re.search(r'^\d+\.\s+', text, re.MULTILINE):
        matches = len(re.findall(r'^[\-\*]\s+', text, re.MULTILINE))
        if matches > 1:
            errors.append(f"WYKRYTO LISTĘ ({matches} pkt). Zakaz punktorów w narracji.")

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
# 📏 AUDYT JĘZYKOWY (Styl + Readability + QA)
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    result = {
        "burstiness": 0.0, 
        "fluff_ratio": 0.0, 
        "passive_ratio": 0.0, 
        "readability_score": 0.0, # Flesch
        "smog_index": 0.0,        # SMOG
        "sentence_count": 0,      # Potrzebne do Pacingu
        "lt_errors": [],          
        "repeated_starts": [], 
        "banned_detected": []
    }
    
    if not text.strip(): return result
    
    try:
        # 1. Segmentacja i Burstiness
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
        if not sentences: sentences = re.split(r'(?<=[.!?])\s+', text)
        result["sentence_count"] = len(sentences)
        
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            result["burstiness"] = math.sqrt(variance)

        # 2. NLP (Fluff + Passive)
        doc = nlp(text)
        adv_adj = sum(1 for t in doc if t.pos_ in ("ADJ", "ADV"))
        total_words = sum(1 for t in doc if t.is_alpha)
        result["fluff_ratio"] = (adv_adj / total_words) if total_words > 0 else 0.0
        
        passive_cnt = 0
        for sent in doc.sents:
            if any(t.lemma_ == "zostać" for t in sent) and any("ppas" in (t.tag_ or "") for t in sent):
                passive_cnt += 1
        result["passive_ratio"] = passive_cnt / len(list(doc.sents)) if list(doc.sents) else 0.0

        # 3. TEXTSTAT (Czytelność: Flesch + SMOG)
        try:
            result["readability_score"] = textstat.flesch_reading_ease(text)
            result["smog_index"] = textstat.smog_index(text)
        except: pass

        # 4. LANGUAGE TOOL (Gramatyka)
        if LT_TOOL_PL:
            matches = LT_TOOL_PL.check(text)
            serious_errors = [m.message for m in matches if m.ruleId not in ("WHITESPACE_RULE", "UPPERCASE_SENTENCE_START")]
            result["lt_errors"] = serious_errors[:3]

        # 5. REPEATED STARTS
        prefix_counts = {}
        for s in sentences:
            words = s.split()
            if len(words) > 2:
                p = " ".join(words[:2]).lower()
                prefix_counts[p] = prefix_counts.get(p, 0) + 1
        result["repeated_starts"] = [p for p, c in prefix_counts.items() if c >= 2]

        # 6. BANNED PHRASES (Fuzzy check)
        banned_phrases = ["warto zauważyć", "w dzisiejszych czasach", "podsumowując", "reasumując", "warto dodać", "nie da się ukryć"]
        found_banned = []
        text_lower = text.lower()
        for b in banned_phrases:
            if b in text_lower:
                found_banned.append(b)
                continue
            if fuzz.partial_ratio(b, text_lower) > 92:
                found_banned.append(b)
        result["banned_detected"] = list(set(found_banned))

    except Exception as e:
        print(f"Audit Error: {e}")
    
    return result


# ===========================================================
# 🚀 HYBRID LEMMA-FUZZY COUNTER
# ===========================================================
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    text_lower = text_raw.lower()
    
    # 1. Exact
    exact_hits = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
    
    # 2. Lemma (Exact + Fuzzy)
    lemma_hits = 0
    target_tok = target_lemma.split()
    
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        used_indices = set()
        
        # A. Exact Lemma
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i+target_len] == target_tok:
                lemma_hits += 1
                for k in range(i, i+target_len): used_indices.add(k)

        # B. Fuzzy Lemma
        min_win = max(1, target_len - MAX_FUZZY_WINDOW_EXPANSION)
        max_win = target_len + MAX_FUZZY_WINDOW_EXPANSION
        target_str = " ".join(target_tok)

        for w_len in range(min_win, max_win + 1):
            if w_len > text_len: continue
            for i in range(text_len - w_len + 1):
                if any(k in used_indices for k in range(i, i+w_len)): continue
                
                window_tok = text_lemma_list[i : i+w_len]
                window_str = " ".join(window_tok)
                
                score_fuzz = fuzz.token_set_ratio(target_str, window_str)
                score_jaccard = textdistance.jaccard.normalized_similarity(target_tok, window_tok)
                
                if score_fuzz >= FUZZY_SIMILARITY_THRESHOLD or score_jaccard >= JACCARD_SIMILARITY_THRESHOLD:
                    lemma_hits += 1
                    for k in range(i, i+w_len): used_indices.add(k)

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


# ===========================================================
# 🧠 GEMINI JUDGE
# ===========================================================
def evaluate_with_gemini(text, meta_trace, burst, fluff, passive, repeated, banned_detected, semantic_score, topic="", previous_context=""):
    if not GEMINI_API_KEY: return {"pass": True, "quality_score": 100}
    try:
       model = genai.GenerativeModel("gemini-1.5-pro")
    except: return {"pass": True, "quality_score": 80, "feedback": "Init Error"}

    context_instruction = ""
    if previous_context:
        context_instruction = f"KONTEKST POPRZEDNI: '{previous_context[:300]}...'. Sprawdź spójność."

    metrics_context = (
        f"METRYKI: Burstiness={burst:.2f}, Fluff={fluff:.2f}, Semantic={semantic_score:.2f}. "
        f"Banned: {banned_detected}."
    )

    prompt = f"""
    Sędzia SEO. Temat: "{topic}".
    {context_instruction}
    
    Twoim zadaniem jest ocena bieżącego fragmentu (Batcha).
    {metrics_context}

    KRYTERIA (HEAR):
    1. Harmony: Czy tekst jest spójny?
    2. Authenticity: Czy unika AI-izmów?
    3. Rhythm: Czy zdania mają różną długość?

    Zwróć TYLKO JSON: 
    {{ "pass": true/false, "quality_score": 0-100, "feedback_for_writer": "Krótka instrukcja." }}

    TEKST: "{text}"
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception: 
        return {"pass": True, "quality_score": 80, "feedback": "Gemini Error"}


# ===========================================================
# 🆕 ENDPOINT: REFINE
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    clean_text = sanitize_typography(text)
    
    # Pełny audyt (w tym SMOG i LT)
    audit = analyze_language_quality(clean_text)
    
    return jsonify({
        "original_text": text,
        "auto_fixed_text": clean_text,
        "language_audit": audit
    })


# ===========================================================
# 🧠 MAIN PROCESS
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}
    
    batch_text = sanitize_typography(batch_text)
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 1. HARD GUARDRAILS
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "HARD RULE VIOLATION",
            "gemini_feedback": {"pass": False, "feedback_for_writer": hard_check['msg']},
            "next_action": "REWRITE"
        }

    # 2. AUDYT
    audit = analyze_language_quality(batch_text)
    warnings = []
    
    if audit.get("banned_detected"):
        warnings.append(f"⛔ Banned: {', '.join(audit['banned_detected'])}")
    if audit.get("lt_errors"):
        warnings.append(f"✍️ Gramatyka: {', '.join(audit['lt_errors'])}")
    if audit.get("readability_score", 100) < 30:
        warnings.append("📖 Tekst trudny (Flesch < 30).")
    
    # 3. SEO TRACKING & PACING CHECK
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    over_limit_hits = []
    total_batch_hits = 0 

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        target_max = meta.get("target_max", 5)
        
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")
        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join([t.lemma_.lower() for t in doc_tmp if t.is_alpha])

        occurrences = count_hybrid_occurrences(batch_text, text_lemma_list, target_exact, target_lemma)
        total_batch_hits += occurrences
        
        if occurrences > 0:
            if (meta.get("actual_uses", 0) + occurrences) > target_max:
                over_limit_hits.append(original_keyword)

        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(meta["actual_uses"], meta["target_min"], meta["target_max"])

    if over_limit_hits:
        warnings.append(f"📈 Limit SEO: {', '.join(over_limit_hits[:3])}")

    # PACING CHECK
    sentence_count = audit.get("sentence_count", 1)
    if sentence_count > 0:
        density = total_batch_hits / sentence_count
        if density > 0.4:
            warnings.append(f"🚨 Keyword Stuffing? Gęstość: {density:.2f} fraz/zdanie.")

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # 4. GEMINI JUDGE
    previous_context = ""
    existing_batches = project_data.get("batches", [])
    if existing_batches:
        last_batch = existing_batches[-1]
        previous_context = last_batch.get("text", "")[-500:]

    gemini_verdict = evaluate_with_gemini(
        text=batch_text, 
        meta_trace=meta_trace, 
        burst=audit["burstiness"], 
        fluff=audit["fluff_ratio"], 
        passive=audit["passive_ratio"], 
        repeated=audit["repeated_starts"], 
        banned_detected=audit["banned_detected"], 
        semantic_score=calculate_semantic_score(batch_text, topic), 
        topic=topic,
        previous_context=previous_context
    )

    # 5. ZAPIS (Z poprawionym timestampem!)
    batch_entry = {
        "text": batch_text, 
        "gemini_audit": gemini_verdict, 
        "language_audit": audit,
        "warnings": warnings,
        "meta_trace": meta_trace,
        "summary": {"under": under, "over": over, "ok": ok},
        "used_h2": (meta_trace or {}).get("used_h2", []),
        "timestamp": datetime.datetime.now(datetime.timezone.utc) # FIX: Native Python datetime
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)

    # 6. RESPONSE
    status = "BATCH_ACCEPTED"
    feedback_msg = "Tekst zapisany."
    if warnings:
        status = "BATCH_WARNING"
        feedback_msg = "Zapisano z UWAGAMI: " + " | ".join(warnings)

    meta_prompt_summary = f"UNDER={under} | Burst={audit['burstiness']:.1f} | Flesch={audit.get('readability_score', 0):.0f}"
    
    top_under = [m.get("keyword") for _, m in sorted(keywords_state.items(), key=lambda i: i[1].get("target_min", 0)-i[1].get("actual_uses", 0), reverse=True) if m["status"]=="UNDER"][:5]
    meta_prompt_summary += f" | BRAKI: {', '.join(top_under)}"

    next_act = "GENERATE_NEXT"
    if under == 0 and len(project_data["batches"]) >= 3:
        next_act = "EXPORT"

    return {
        "status": status,
        "gemini_feedback": {"feedback_for_writer": feedback_msg},
        "language_audit": audit,
        "meta_prompt_summary": meta_prompt_summary,
        "next_action": next_act
    }
