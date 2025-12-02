import os
import json
import math
import re
import numpy as np
import datetime 
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz           
import language_tool_python         
import textstat                     
import textdistance                 
import pysbd

# ⭐ NEW: Import SEO Optimizer functions (v12.25.1)
from seo_optimizer import (
    build_rolling_context,
    calculate_semantic_drift,
    analyze_transition_quality,
    calculate_keyword_position_score,
    optimize_for_featured_snippet
)

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

textstat.set_lang("pl")
SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 💮‍♂️ HARD GUARDRAILS
# ===========================================================
def validate_hard_rules(text: str) -> dict:
    errors = []
    if re.search(r'^[\-\*]\s+', text, re.MULTILINE) or re.search(r'^\d+\.\s+', text, re.MULTILINE):
        matches = len(re.findall(r'^[\-\*]\s+', text, re.MULTILINE))
        if matches > 1:
            errors.append(f"WYKRYTO LISTĘ ({matches} pkt). Zakaz punktów.")
    if errors:
        return {"valid": False, "msg": " | ".join(errors)}
    return {"valid": True, "msg": "OK"}

def sanitize_typography(text: str) -> str:
    if not text: return ""
    text = text.replace("—", " — ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===========================================================
# 🔍 AUDYT
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    result = {
        "burstiness": 0.0, "fluff_ratio": 0.0, "passive_ratio": 0.0, 
        "readability_score": 0.0, "smog_index": 0.0, "sentence_count": 0,
        "lt_errors": [], "repeated_starts": [], "banned_detected": []
    }
    if not text.strip(): return result
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
        if not sentences: sentences = re.split(r'(?<=[.!?])\s+', text)
        result["sentence_count"] = len(sentences)
        
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            mean = sum(lengths) / len(lengths)
            var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            result["burstiness"] = math.sqrt(var)

        doc = nlp(text)
        adv_adj = sum(1 for t in doc if t.pos_ in ("ADJ", "ADV"))
        total = sum(1 for t in doc if t.is_alpha)
        result["fluff_ratio"] = (adv_adj / total) if total > 0 else 0.0
        
        passive = 0
        for sent in doc.sents:
            if any(t.lemma_ == "zostać" for t in sent) and any("ppas" in (t.tag_ or "") for t in sent):
                passive += 1
        result["passive_ratio"] = passive / len(list(doc.sents)) if list(doc.sents) else 0.0

        try:
            result["readability_score"] = textstat.flesch_reading_ease(text)
            result["smog_index"] = textstat.smog_index(text)
        except: pass

        if LT_TOOL_PL:
            matches = LT_TOOL_PL.check(text)
            errs = [m.message for m in matches if m.ruleId not in ("WHITESPACE_RULE", "UPPERCASE_SENTENCE_START")]
            result["lt_errors"] = errs[:3]

        prefix_counts = {}
        for s in sentences:
            words = s.split()
            if len(words) > 2:
                p = " ".join(words[:2]).lower()
                prefix_counts[p] = prefix_counts.get(p, 0) + 1
        result["repeated_starts"] = [p for p, c in prefix_counts.items() if c >= 2]

        banned_phrases = ["warto zauważyć", "w dzisiejszych czasach", "podsumowując", "reasumując", "warto dodać", "nie da się ukryć"]
        found = []
        text_l = text.lower()
        for b in banned_phrases:
            if b in text_l or fuzz.partial_ratio(b, text_l) > 92: found.append(b)
        result["banned_detected"] = list(set(found))
    except Exception as e: print(f"Audit Error: {e}")
    return result


# ===========================================================
# 🚀 HYBRID COUNTER - FIXED v12.25.2
# ===========================================================

def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma, debug=False):
    """
    ✅ FIXED v12.25.2: Eliminuje loop-bug i duplikaty form fleksyjnych.
    
    ZMIANY:
    1. Tracking używanych pozycji (used_positions set)
    2. Lemma exact match najpierw zajmuje pozycje
    3. Fuzzy match TYLKO dla niezajętych pozycji
    4. Break po znalezieniu fuzzy match w danym oknie
    5. Return MAX(exact, lemma+fuzzy) zamiast sumowania
    
    Strategia:
    - Exact match (literal string)
    - Lemma match (morfologia) - tracked positions
    - Fuzzy match (tylko dla nowych pozycji) - tracked positions
    
    Zwraca: MAX z (exact, lemma+fuzzy_deduplicated)
    """
    text_lower = text_raw.lower()
    
    # === 1. EXACT MATCH (Literal) ===
    exact_hits = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
    
    # === 2. LEMMA + FUZZY MATCH (Deduplicated) ===
    lemma_fuzzy_hits = 0
    target_tok = target_lemma.split()
    
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        
        # ⭐ CRITICAL FIX: Track używane pozycje (unikamy duplikatów)
        used_positions = set()
        
        # --- 2A. EXACT LEMMA MATCH ---
        for i in range(text_len - target_len + 1):
            # Sprawdź czy to dokładnie ta sekwencja lematów
            if text_lemma_list[i : i+target_len] == target_tok:
                # Sprawdź czy pozycja nie jest zajęta
                position_range = range(i, i+target_len)
                if not any(pos in used_positions for pos in position_range):
                    lemma_fuzzy_hits += 1
                    # Zaznacz pozycje jako użyte
                    for pos in position_range:
                        used_positions.add(pos)
        
        # --- 2B. FUZZY MATCH (tylko dla NOWYCH pozycji) ---
        # TYLKO jeśli fuzzy threshold jest wysoki (90+)
        if FUZZY_SIMILARITY_THRESHOLD >= 90:
            min_win = max(1, target_len - MAX_FUZZY_WINDOW_EXPANSION)
            max_win = target_len + MAX_FUZZY_WINDOW_EXPANSION
            target_str = " ".join(target_tok)
            
            for w_len in range(min_win, max_win + 1):
                if w_len > text_len: 
                    continue
                    
                for i in range(text_len - w_len + 1):
                    position_range = range(i, i+w_len)
                    
                    # ⭐ FIX: SKIP jeśli pozycja zajęta (to już policzone!)
                    if any(pos in used_positions for pos in position_range):
                        continue
                    
                    window_tok = text_lemma_list[i : i+w_len]
                    window_str = " ".join(window_tok)
                    
                    # Fuzzy check
                    fuzzy_score = fuzz.token_set_ratio(target_str, window_str)
                    jaccard_score = textdistance.jaccard.normalized_similarity(target_tok, window_tok)
                    
                    if fuzzy_score >= FUZZY_SIMILARITY_THRESHOLD or jaccard_score >= JACCARD_SIMILARITY_THRESHOLD:
                        lemma_fuzzy_hits += 1
                        # Zaznacz pozycje jako użyte
                        for pos in position_range:
                            used_positions.add(pos)
                        # ⭐ FIX: BREAK z tego okna (nie szukaj więcej w tym miejscu)
                        break
    
    # === 3. RETURN MAX (exact lub lemma+fuzzy) ===
    # ⭐ FIX: MAX zamiast sumowania (exact i lemma mogą się pokrywać)
    final_count = max(exact_hits, lemma_fuzzy_hits)
    
    # ⭐ DEBUG MODE (do usunięcia po weryfikacji)
    if debug and (exact_hits > 5 or lemma_fuzzy_hits > 5):
        print(f"\n🔍 DEBUG COUNT for '{target_exact}':")
        print(f"   Exact hits: {exact_hits}")
        print(f"   Lemma+Fuzzy hits: {lemma_fuzzy_hits}")
        print(f"   Used positions: {len(used_positions)}")
        print(f"   Final count: {final_count}")
    
    return final_count


def validate_keyword_count(keyword, found_count, target_max, batch_text):
    """
    ✅ NEW v12.25.2: Smart validation - sprawdza czy overuse jest rzeczywisty czy artefakt.
    
    Jeśli hybrid count > 2x target_max, porównujemy z prostym countem.
    Jeśli simple count OK, a hybrid dużo wyższy → false positive.
    
    Returns:
        tuple: (validated_count, is_false_positive, warning_message)
    """
    # 1. Jeśli count > 2x target_max → prawdopodobnie bug
    if found_count > (target_max * 2):
        # Sprawdź metodą "naiwną" (prosty count w tekście)
        text_lower = batch_text.lower()
        keyword_lower = keyword.lower()
        
        # Simple count (bez fuzzy/lemma)
        simple_count = text_lower.count(keyword_lower)
        
        # Jeśli simple_count << found_count → false positive
        if simple_count <= target_max and found_count > (target_max + 3):
            warning = f"⚠️ Auto-corrected '{keyword}': Hybrid={found_count} → Simple={simple_count} (false positive detected)"
            print(warning)
            return simple_count, True, warning
    
    return found_count, False, None


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
        return genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")['embedding']
    except: return None

def calculate_semantic_score(batch_text, main_topic):
    if not batch_text or not main_topic: return 1.0
    vec_text = get_embedding(batch_text)
    vec_topic = get_embedding(main_topic)
    if not vec_text or not vec_topic: return 1.0
    return float(np.dot(vec_text, vec_topic) / (np.linalg.norm(vec_text) * np.linalg.norm(vec_topic)))

def evaluate_with_gemini(text, meta_trace, burst, fluff, passive, repeated, banned_detected, semantic_score, topic="", project_data=None):
    """
    ⭐ UPDATED v12.25.1: Używa build_rolling_context zamiast ostatnich 500 znaków
    """
    if not GEMINI_API_KEY: return {"pass": True, "quality_score": 100}
    try: model = genai.GenerativeModel("gemini-1.5-pro")
    except: return {"pass": True, "quality_score": 80}
    
    # ⭐ FIX #1: Rolling Context Window (zamiast prev_context[-500:])
    if project_data:
        ctx = build_rolling_context(project_data, window_size=3)
    else:
        ctx = ""
    
    prompt = f"""
    Sędzia SEO. Temat: "{topic}". 
    
    {ctx}
    
    METRYKI BIEŻĄCEGO BATCHA: 
    - Burstiness={burst:.1f} (cel: >6.0)
    - Fluff={fluff:.2f} (cel: <0.15)
    - Banned phrases={banned_detected}
    
    Oceń: Harmonia, Empatia, Autentyczność, Rytm (HEAR Framework).
    
    Zwróć JSON: {{ "pass": bool, "quality_score": 0-100, "feedback_for_writer": "string" }}
    
    TEKST DO OCENY: "{text[:4000]}"
    """
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        return result
    except Exception as e:
        print(f"Gemini eval error: {e}")
        return {"pass": True, "quality_score": 80, "feedback_for_writer": f"Błąd oceny: {e}"}

@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    clean_text = sanitize_typography(text)
    audit = analyze_language_quality(clean_text)
    return jsonify({"original_text": text, "auto_fixed_text": clean_text, "language_audit": audit})

# ===========================================================
# 🧠 MAIN PROCESS (Logic V12.25.2: Fixed Counting + Validation)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    """
    ⭐ UPDATED v12.25.2: 
    - Fixed count_hybrid_occurrences (eliminuje duplikaty)
    - Dodana validate_keyword_count (false positive detection)
    - Debug mode dla pierwszych 3 batchy
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status": 404}
    
    batch_text = sanitize_typography(batch_text)
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 1. HARD RULE (Struktura)
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
        return {"status": "REJECTED_QUALITY", "error": "HARD RULE", "gemini_feedback": {"feedback_for_writer": hard_check['msg']}, "next_action": "REWRITE"}

    # 2. AUDYT JĘZYKOWY (Burstiness, Fluff, etc.)
    audit = analyze_language_quality(batch_text)
    warnings = []
    if audit.get("banned_detected"): warnings.append(f"⛔ Banned: {', '.join(audit['banned_detected'])}")
    if audit.get("readability_score", 100) < 30: warnings.append("📖 Tekst trudny.")

    # ⭐ FIX #2: SEMANTIC DRIFT CHECK
    previous_batches = project_data.get("batches", [])
    drift_check = calculate_semantic_drift(batch_text, previous_batches, threshold=0.65)
    audit["semantic_drift"] = drift_check
    
    if drift_check["status"] == "DRIFT_WARNING":
        warnings.append(f"🌀 {drift_check['message']}")

    # ⭐ FIX #3: TRANSITION QUALITY ANALYSIS
    transition_check = analyze_transition_quality(
        batch_text,
        previous_batches[-1] if previous_batches else None
    )
    audit["transition_quality"] = transition_check
    
    if transition_check["status"] == "CHOPPY":
        warnings.append(f"🔗 {transition_check['message']}")

    # 3. SEO TRACKING (Critical Overuse Logic) - FIXED v12.25.2
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    over_limit_hits = []
    critical_reject = []
    
    # ⭐ DEBUG MODE: Enable dla pierwszych 3 batchy
    is_debug_mode = len(previous_batches) < 3

    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        target_max = meta.get("target_max", 5)
        
        # Logika liczenia (Hybrid Counter - FIXED)
        t_exact = meta.get("search_term_exact", kw.lower())
        t_lemma = meta.get("search_lemma", "")
        if not t_lemma: t_lemma = " ".join([t.lemma_.lower() for t in nlp(kw) if t.is_alpha])
        
        # ⭐ FIXED: count_hybrid_occurrences z deduplication
        found = count_hybrid_occurrences(batch_text, text_lemma_list, t_exact, t_lemma, debug=is_debug_mode)
        
        if found > 0:
            current_total = meta.get("actual_uses", 0)
            new_total = current_total + found
            
            # ⭐ NEW: VALIDATION LAYER (false positive detection)
            validated_total, is_false_positive, validation_warning = validate_keyword_count(
                kw, new_total, target_max, batch_text
            )
            
            if is_false_positive:
                warnings.append(validation_warning)
                new_total = validated_total  # Use corrected count
            
            # Hard Ceiling (+3 over limit)
            if new_total > target_max:
                if new_total >= (target_max + 3):
                    critical_reject.append(f"{kw} (Użyto {new_total}/{target_max})")
                else:
                    over_limit_hits.append(kw)

            meta["actual_uses"] = new_total
            meta["status"] = compute_status(new_total, meta["target_min"], target_max)
            
            # ⭐ FIX #4: POSITION-WEIGHTED SCORING
            position_score = calculate_keyword_position_score(batch_text, kw)
            meta["position_score"] = position_score["score"]
            meta["position_quality"] = position_score["quality"]
            meta["early_count"] = position_score["early_count"]
            
            # Warning jeśli słaba pozycja
            if found > 0 and position_score["quality"] in ["WEAK", "NONE"]:
                warnings.append(f"📍 '{kw}': słaba pozycja (score: {position_score['score']:.1f})")

    # JEŚLI MAMY KRYTYCZNE PRZEKROCZENIE -> REJECT!
    if critical_reject:
        return {
            "status": "REJECTED_SEO",
            "error": "CRITICAL OVERUSE",
            "gemini_feedback": {
                "pass": False, 
                "feedback_for_writer": f"⛔ DRASTYCZNE PRZEOPT. FRAZ: {', '.join(critical_reject)}. Usuń je z tego batcha natychmiast!"
            },
            "language_audit": audit,
            "next_action": "REWRITE"
        }

    if over_limit_hits:
        warnings.append(f"📈 Limit SEO (+1/2): {', '.join(over_limit_hits[:3])}")

    # Pacing check
    if audit["sentence_count"] > 0 and (sum(1 for _ in keywords_state) / audit["sentence_count"]) > 0.5:
        warnings.append("🚨 Keyword Stuffing (zbyt gęsto).")

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # 4. GEMINI JUDGE (z nowym rolling context)
    gemini_verdict = evaluate_with_gemini(
        batch_text, meta_trace, 
        audit["burstiness"], audit["fluff_ratio"], 
        0, [], audit["banned_detected"], 0, 
        topic=topic, 
        project_data=project_data
    )

    # 5. SAVE
    batch_entry = {
        "text": batch_text, 
        "gemini_audit": gemini_verdict, 
        "language_audit": audit,
        "warnings": warnings, 
        "meta_trace": meta_trace,
        "summary": {"under": under, "over": over, "ok": ok},
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    project_data["version"] = "v12.25.2"  # ⭐ Updated version
    doc_ref.set(project_data)

    status = "BATCH_WARNING" if warnings else "BATCH_ACCEPTED"
    fb_msg = ("Zapisano z UWAGAMI: " + " | ".join(warnings)) if warnings else "Zapisano."
    
    top_under = [m.get("keyword") for _, m in sorted(keywords_state.items(), key=lambda i: i[1].get("target_min", 0)-i[1].get("actual_uses", 0), reverse=True) if m["status"]=="UNDER"][:5]
    meta_summary = f"UNDER={under} | TOP_UNDER={', '.join(top_under)} | {fb_msg}"

    next_act = "EXPORT" if under == 0 and len(project_data["batches"]) >= 3 else "GENERATE_NEXT"

    return {
        "status": status,
        "gemini_feedback": {"feedback_for_writer": fb_msg},
        "language_audit": audit,
        "meta_prompt_summary": meta_summary,
        "next_action": next_act
    }
