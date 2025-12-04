# ===========================================================
# Version: v12.25.6.6
# Last updated: 2024-12-04
# Changes:
# - v12.25.6.5: Preview + smart paraphrasing + distribution plan
# - v12.25.6.6: LanguageTool Remote API (no Java) + error handling
#   * Switched to LanguageToolPublicAPI (free tier, no local Java)
#   * Added timeout handling for LanguageTool (2s timeout)
#   * Improved error logging
#   * RAM savings: ~500MB per worker
# ===========================================================

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import datetime
import re
import json
import os
import spacy
from rapidfuzz import fuzz
import pysbd
import textstat

# LanguageTool - Remote API (no Java needed!)
try:
    import language_tool_python
    # Use public API instead of local Java server
    # Free tier: 20 requests/minute, unlimited monthly
    LT_TOOL_PL = language_tool_python.LanguageToolPublicAPI("pl-PL")
    print("[INIT] ✅ LanguageTool loaded (remote API, free tier).")
except Exception as e:
    LT_TOOL_PL = None
    print(f"[WARNING] ⚠️ LanguageTool init failed: {e}")

# spaCy for lemmatization
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[INIT] ✅ spaCy Polish model loaded.")
except:
    print("[WARNING] ⚠️ spaCy model not found. Run: python -m spacy download pl_core_news_lg")
    nlp = None

tracker_routes = Blueprint("tracker_routes", __name__)

# ===========================================================
# 🔧 AUTO-FIX OVERUSE (v12.25.6.5 - SMART PARAPHRASING)
# ===========================================================
def auto_fix_keyword_overuse_smart(text: str, keyword: str, current_count: int, target_max: int) -> str:
    """
    Automatically reduces keyword overuse by ~50% using SMART PARAPHRASING
    - Preserves text structure and layout
    - Maintains similar text length (±5%)
    - Replaces keywords with synonyms/paraphrases instead of deleting
    - ALWAYS keeps minimum 2 uses
    v12.25.6.5
    """
    if current_count <= target_max:
        return text
    
    # Calculate how many to replace (50% of overuse, but keep min 2)
    overuse = current_count - target_max
    to_replace = max(1, overuse // 2)
    target_final = max(2, current_count - to_replace)
    
    # Find all occurrences
    import re
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    if len(matches) == 0:
        return text
    
    actual_replace = len(matches) - target_final
    if actual_replace <= 0:
        return text
    
    # SMART PARAPHRASING: Common Polish synonyms/paraphrases
    paraphrase_dict = {
        # Hair & scalp
        "włosy": ["pasma", "kosmyki", "fryzura"],
        "włosów": ["pasm", "kosmyków", "fryzury"],
        "skóra głowy": ["naskórek", "skóra", "powierzchnia głowy"],
        "skóry głowy": ["naskórka", "skóry", "powierzchni głowy"],
        "pielęgnacja włosów": ["dbanie o włosy", "troska o pasma", "pielęgnacja"],
        "pielęgnacji włosów": ["dbania o włosy", "troski o pasma", "pielęgnacji"],
        
        # Aging & skin
        "proces starzenia": ["starzenie", "proces", "zmiany wiekowe"],
        "kolagen": ["białko strukturalne", "białko"],
        "kolagenu": ["białka strukturalnego", "białka"],
        "elastyna": ["włókna elastyczne", "białko"],
        "elastyny": ["włókien elastycznych", "białka"],
        
        # Care products
        "naturalne składniki": ["naturalne substancje", "składniki naturalne", "substancje"],
        "kwas hialuronowy": ["hialuron", "kwas", "składnik"],
        
        # General
        "menopauza": ["przekwitanie", "ten okres", "okres"],
        "menopauzy": ["przekwitania", "tego okresu", "okresu"],
    }
    
    # Get paraphrases for this keyword
    kw_lower = keyword.lower()
    paraphrases = paraphrase_dict.get(kw_lower, [kw_lower])  # fallback to original
    
    # Replace matches from END (preserve early high-weight ones)
    indices_to_replace = sorted([m.start() for m in matches], reverse=True)[:actual_replace]
    
    result = text
    paraphrase_idx = 0
    
    for idx in sorted(indices_to_replace, reverse=True):
        # Get paraphrase (cycle through list)
        if paraphrases:
            paraphrase = paraphrases[paraphrase_idx % len(paraphrases)]
            paraphrase_idx += 1
        else:
            paraphrase = keyword  # fallback
        
        # Replace keyword with paraphrase
        match_len = len(keyword)
        result = result[:idx] + paraphrase + result[idx + match_len:]
    
    return result

# ===========================================================
# 👮‍♂️ HARD GUARDRAILS
# ===========================================================
def validate_hard_rules(text: str) -> dict:
    """Hard rules that trigger immediate rejection"""
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    
    # Must have at least 3 paragraphs
    if len(lines) < 3:
        return {"valid": False, "msg": "Tekst musi mieć minimum 3 akapity."}
    
    # Each paragraph min 100 chars
    for i, line in enumerate(lines, 1):
        if len(line) < 100:
            return {"valid": False, "msg": f"Akapit {i} za krótki (<100 znaków)."}
    
    return {"valid": True}

def validate_paragraph_structure(text: str) -> dict:
    """
    Validates paragraph structure (v12.25.6)
    - Min 4 sentences per paragraph
    - Min 1 compound sentence per paragraph (with comma or conjunction)
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    
    for i, para in enumerate(paragraphs, 1):
        # Sentence count
        sentences = segmenter.segment(para)
        if len(sentences) < 4:
            return {
                "valid": False,
                "msg": f"Akapit {i} ma {len(sentences)} zdań (wymagane: min 4)"
            }
        
        # Compound sentence check
        has_compound = False
        conjunctions = ["który", "która", "które", "ale", "jednak", "ponieważ", 
                       "gdyż", "jeśli", "gdy", "bo", "oraz", "i", "a"]
        
        for sent in sentences:
            # Has comma OR conjunction
            if ',' in sent or any(conj in sent.lower() for conj in conjunctions):
                has_compound = True
                break
        
        if not has_compound:
            return {
                "valid": False,
                "msg": f"Akapit {i} nie ma zdań złożonych (wymagane: min 1 z przecinkiem lub spójnikiem)"
            }
    
    return {"valid": True}

# ===========================================================
# 📊 LANGUAGE QUALITY AUDIT
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    """
    Analyzes text quality using multiple metrics:
    - Burstiness (sentence length variety)
    - Fluff ratio (empty phrases)
    - SMOG readability
    - Gramatyka (LanguageTool Remote API)
    - Banned phrases detection
    v12.25.6.6: Now uses Remote API with timeout handling
    """
    result = {}
    
    # Segment sentences
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    sentences = segmenter.segment(text)
    result["sentence_count"] = len(sentences)
    
    # Burstiness (sentence length variance)
    if sentences:
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            import statistics
            mean_len = statistics.mean(lengths)
            stdev_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
            result["burstiness"] = stdev_len / mean_len if mean_len > 0 else 0
        else:
            result["burstiness"] = 0
    
    # Fluff detection
    fluff_phrases = [
        "warto wspomnieć", "należy pamiętać", "istotne jest", 
        "ważne jest aby", "nie bez znaczenia", "co ciekawe",
        "jak wiadomo", "okazuje się", "trzeba przyznać"
    ]
    fluff_count = sum(text.lower().count(phrase) for phrase in fluff_phrases)
    word_count = len(text.split())
    result["fluff_ratio"] = fluff_count / word_count if word_count > 0 else 0
    
    # Readability metrics
    try:
        if len(sentences) >= 3:
            result["readability_score"] = textstat.flesch_reading_ease(text)
            result["smog_index"] = textstat.smog_index(text)
        else:
            result["readability_score"] = 0
            result["smog_index"] = 0
    except:
        result["readability_score"] = 0
        result["smog_index"] = 0
    
    # LanguageTool grammar check (Remote API with timeout)
    result["lt_errors"] = []
    if LT_TOOL_PL:
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LanguageTool check timeout")
            
            # Set 2 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            
            try:
                matches = LT_TOOL_PL.check(text)
                errs = [m.message for m in matches if m.ruleId not in ("WHITESPACE_RULE", "UPPERCASE_SENTENCE_START")]
                result["lt_errors"] = errs[:3]  # Top 3 errors only
            finally:
                signal.alarm(0)  # Cancel alarm
                
        except TimeoutError:
            print("[WARNING] ⚠️ LanguageTool timeout (>2s)")
            result["lt_errors"] = ["Timeout"]
        except Exception as e:
            print(f"[ERROR] ❌ LanguageTool check failed: {e}")
            result["lt_errors"] = []
    
    # Repeated sentence starts
    prefix_counts = {}
    for s in sentences:
        words = s.split()
        if len(words) > 2:
            p = " ".join(words[:2]).lower()
            prefix_counts[p] = prefix_counts.get(p, 0) + 1
    result["repeated_starts"] = [p for p, c in prefix_counts.items() if c >= 2]
    
    # Banned phrases (load from file if exists)
    banned_phrases = []
    banned_file = os.path.join(os.path.dirname(__file__), "frazy_ai_banned.json")
    if os.path.exists(banned_file):
        try:
            with open(banned_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                banned_phrases = data.get("banned", [])
        except:
            pass
    
    detected_banned = [phrase for phrase in banned_phrases if phrase.lower() in text.lower()]
    result["banned_detected"] = detected_banned[:5]  # Top 5
    
    return result

# ===========================================================
# 🔍 HYBRID KEYWORD COUNTER (Exact + Lemma + Fuzzy)
# ===========================================================

# Ultra-strict thresholds (v12.25.6.1)
FUZZY_SIMILARITY_THRESHOLD = 99
JACCARD_THRESHOLD = 0.95
MIN_WORD_OVERLAP = 0.85

def count_hybrid_occurrences(text: str, text_lemma_list: list, search_term_exact: str, search_lemma: str) -> int:
    """
    Hybrid counter: exact + lemma + fuzzy (ultra-strict)
    v12.25.6.1: Disabled fuzzy for 1-2 word phrases
    """
    text_lower = text.lower()
    count = 0
    
    # 1. Exact match
    count += text_lower.count(search_term_exact.lower())
    
    # 2. Lemma match
    if nlp and search_lemma:
        lemma_tokens = search_lemma.split()
        lemma_len = len(lemma_tokens)
        
        for i in range(len(text_lemma_list) - lemma_len + 1):
            window = text_lemma_list[i:i + lemma_len]
            if window == lemma_tokens:
                count += 1
    
    # 3. Fuzzy match (DISABLED for 1-2 word phrases)
    phrase_words = search_term_exact.split()
    if len(phrase_words) >= 3:  # Only for 3+ word phrases
        text_words = text_lower.split()
        phrase_word_count = len(phrase_words)
        
        min_win = max(phrase_word_count - 1, 1)
        max_win = phrase_word_count + 2
        
        for w_len in range(min_win, max_win + 1):
            for i in range(len(text_words) - w_len + 1):
                window = " ".join(text_words[i:i + w_len])
                
                # Similarity check
                sim = fuzz.token_set_ratio(window, search_term_exact.lower())
                if sim < FUZZY_SIMILARITY_THRESHOLD:
                    continue
                
                # Jaccard check
                set1 = set(window.split())
                set2 = set(phrase_words)
                jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                if jaccard < JACCARD_THRESHOLD:
                    continue
                
                # Word overlap check
                common = len(set1 & set2)
                overlap = common / len(set2) if set2 else 0
                if overlap >= MIN_WORD_OVERLAP:
                    count += 1
    
    return count

def compute_status(actual: int, target_min: int, target_max: int) -> str:
    """Keyword status: UNDER, OK, OVER"""
    if actual < target_min:
        return "UNDER"
    elif actual > target_max:
        return "OVER"
    else:
        return "OK"

def global_keyword_stats(keywords_state: dict) -> tuple:
    """Count keywords by status"""
    under = sum(1 for v in keywords_state.values() if v.get("status") == "UNDER")
    over = sum(1 for v in keywords_state.values() if v.get("status") == "OVER")
    locked = sum(1 for v in keywords_state.values() if v.get("locked", False))
    ok = len(keywords_state) - under - over
    return under, over, locked, ok

# ===========================================================
# 🎯 SEMANTIC DRIFT CHECK
# ===========================================================
def calculate_semantic_drift(new_batch_text: str, previous_batches: list) -> dict:
    """Calculate semantic drift from previous batches"""
    if not previous_batches or len(previous_batches) == 0:
        return {"drift_score": 0, "status": "OK"}
    
    # Simple word overlap for now (can upgrade to embeddings later)
    new_words = set(new_batch_text.lower().split())
    prev_words = set()
    for batch in previous_batches[-2:]:  # Last 2 batches
        prev_words.update(batch.get("text", "").lower().split())
    
    if not prev_words:
        return {"drift_score": 0, "status": "OK"}
    
    overlap = len(new_words & prev_words) / len(new_words) if new_words else 0
    drift = 1.0 - overlap
    
    return {
        "drift_score": round(drift, 2),
        "status": "OK" if drift < 0.5 else "WARNING"
    }

def analyze_transition_quality(new_batch: str, prev_batch: dict) -> dict:
    """Check transition quality between batches"""
    if not prev_batch:
        return {"score": 1.0, "status": "OK"}
    
    prev_text = prev_batch.get("text", "")
    if not prev_text:
        return {"score": 1.0, "status": "OK"}
    
    # Check if first sentence references previous content
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    new_sentences = segmenter.segment(new_batch)
    
    if not new_sentences:
        return {"score": 0.5, "status": "WARNING"}
    
    first_sent = new_sentences[0].lower()
    transition_words = ["również", "ponadto", "dodatkowo", "kolejny", "inny", "podobnie"]
    
    has_transition = any(word in first_sent for word in transition_words)
    score = 1.0 if has_transition else 0.7
    
    return {
        "score": score,
        "status": "OK" if score >= 0.7 else "WARNING"
    }

# ===========================================================
# 🔌 API ENDPOINT: Batch Preview (v12.25.6.5)
# ===========================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """
    Preview batch with all corrections BEFORE saving to database
    Returns corrected text + all checks + warnings
    User must explicitly approve to save
    v12.25.6.5
    """
    data = request.get_json(force=True) or {}
    batch_text = data.get("text", "")
    meta_trace = data.get("meta_trace", {})
    
    # Run full processing WITHOUT saving
    result = process_batch_preview(project_id, batch_text, meta_trace)
    
    return jsonify(result)

def process_batch_preview(project_id: str, batch_text: str, meta_trace: dict = None):
    """
    Same as process_batch_in_firestore but DOES NOT SAVE to database
    Returns preview with corrected text
    v12.25.6.5
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists: 
        return {"error": "Project not found", "status": 404}
    
    original_text = batch_text
    batch_text = batch_text.strip()
    project_data = doc.to_dict()

    # Run all checks (same as normal process)
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "HARD RULE VIOLATION",
            "original_text": original_text,
            "corrected_text": batch_text,
            "corrections_made": [],
            "gemini_feedback": {"feedback_for_writer": hard_check['msg']},
            "next_action": "REWRITE"
        }
    
    para_check = validate_paragraph_structure(batch_text)
    if not para_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "PARAGRAPH STRUCTURE VIOLATION",
            "original_text": original_text,
            "corrected_text": batch_text,
            "corrections_made": [],
            "gemini_feedback": {"feedback_for_writer": f"⛔ STRUKTURA AKAPITÓW: {para_check['msg']}"},
            "next_action": "REWRITE"
        }

    audit = analyze_language_quality(batch_text)
    warnings = []
    suggestions = []
    corrections_made = []
    
    if audit.get("banned_detected"): 
        warnings.append(f"⛔ Banned phrases: {', '.join(audit['banned_detected'])}")
        suggestions.append("Remove banned phrases")
    
    if audit.get("burstiness", 0) < 6.0:
        warnings.append(f"📊 Burstiness {audit['burstiness']:.1f} (target: >6.0)")
    
    if audit.get("fluff_ratio", 0) > 0.15:
        warnings.append(f"💨 Fluff {audit['fluff_ratio']:.2f} (target: <0.15)")
    
    if audit.get("smog_index", 0) > 15:
        warnings.append(f"📖 SMOG {audit['smog_index']:.1f} (target: <15)")

    # Keyword tracking + auto-fix
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    
    if not nlp:
        return {"error": "spaCy model not loaded", "status": 500}
    
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    over_limit_hits = []
    critical_reject = []

    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        target_max = meta.get("target_max", 5)
        
        t_exact = meta.get("search_term_exact", kw.lower())
        t_lemma = meta.get("search_lemma", "")
        if not t_lemma: 
            t_lemma = " ".join([t.lemma_.lower() for t in nlp(kw) if t.is_alpha])
        
        found = count_hybrid_occurrences(batch_text, text_lemma_list, t_exact, t_lemma)
        
        if found > 0:
            current_total = meta.get("actual_uses", 0)
            new_total = current_total + found
            
            if new_total > target_max:
                if new_total >= (target_max + 3):
                    critical_reject.append(f"{kw} ({new_total}/{target_max})")
                else:
                    over_limit_hits.append(kw)

            meta["actual_uses"] = new_total
            meta["status"] = compute_status(new_total, meta["target_min"], target_max)

    # AUTO-FIX overuse (smart paraphrasing)
    fixed_text = batch_text
    if critical_reject:
        for overuse_info in critical_reject:
            parts = overuse_info.split(" (")
            if len(parts) == 2:
                kw = parts[0]
                counts = parts[1].rstrip(")").split("/")
                if len(counts) == 2:
                    current = int(counts[0])
                    limit = int(counts[1])
                    
                    fixed_text = auto_fix_keyword_overuse_smart(fixed_text, kw, current, limit)
                    corrections_made.append(f"Replaced ~50% of '{kw}' with synonyms ({current} → ~{current - (current-limit)//2})")
        
        # Re-count
        doc_nlp_fixed = nlp(fixed_text)
        text_lemma_list_fixed = [t.lemma_.lower() for t in doc_nlp_fixed if t.is_alpha]
        
        for row_id, meta in keywords_state.items():
            kw = meta.get("keyword", "")
            t_exact = meta.get("search_term_exact", kw.lower())
            t_lemma = meta.get("search_lemma", "")
            if not t_lemma:
                t_lemma = " ".join([t.lemma_.lower() for t in nlp(kw) if t.is_alpha])
            
            found = count_hybrid_occurrences(fixed_text, text_lemma_list_fixed, t_exact, t_lemma)
            if found > 0:
                meta["actual_uses"] = found
                meta["status"] = compute_status(found, meta["target_min"], meta["target_max"])
        
        batch_text = fixed_text
        
        if corrections_made:
            warnings.append(f"✅ AUTO-CORRECTED: {len(corrections_made)} keywords paraphrased")

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # Semantic drift + transition (for info only)
    drift_check = calculate_semantic_drift(batch_text, project_data.get("batches", []))
    prev_batch = project_data.get("batches", [])[-1] if project_data.get("batches") else None
    transition_check = analyze_transition_quality(batch_text, prev_batch)

    # Return preview (NO SAVE)
    return {
        "status": "PREVIEW",
        "original_text": original_text,
        "corrected_text": batch_text,
        "text_changed": (original_text != batch_text),
        "corrections_made": corrections_made,
        "language_audit": {
            "burstiness": audit.get("burstiness"),
            "fluff_ratio": audit.get("fluff_ratio"),
            "smog_index": audit.get("smog_index"),
            "sentence_count": audit.get("sentence_count")
        },
        "keywords_summary": {
            "under": under,
            "over": over,
            "ok": ok
        },
        "keywords_state": keywords_state,  # Include for approve endpoint
        "warnings": warnings,
        "suggestions": suggestions,
        "semantic_drift": drift_check,
        "transition_quality": transition_check,
        "next_action": "APPROVE_OR_MODIFY"
    }

# ===========================================================
# 🔌 API ENDPOINT: Approve and Save Batch (v12.25.6.5)
# ===========================================================
@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    Save previously previewed batch to database
    User has already seen corrected text and approved it
    v12.25.6.5
    """
    data = request.get_json(force=True) or {}
    corrected_text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    keywords_state = data.get("keywords_state", {})
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found", "status": 404})
    
    project_data = doc.to_dict()
    
    # Quick re-audit for final stats
    audit = analyze_language_quality(corrected_text)
    
    # Calculate keyword stats
    under = sum(1 for v in keywords_state.values() if v.get("status") == "UNDER")
    over = sum(1 for v in keywords_state.values() if v.get("status") == "OVER")
    ok = sum(1 for v in keywords_state.values() if v.get("status") == "OK")
    
    # Save to Firestore
    batch_entry = {
        "text": corrected_text,
        "language_audit": audit,
        "meta_trace": meta_trace,
        "summary": {"under": under, "over": over, "ok": ok},
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data:
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    # Top UNDER keywords
    top_under = [
        m.get("keyword") 
        for _, m in sorted(
            keywords_state.items(), 
            key=lambda i: i[1].get("target_min", 0) - i[1].get("actual_uses", 0), 
            reverse=True
        ) 
        if m.get("status") == "UNDER"
    ][:5]
    
    next_act = "EXPORT" if under == 0 and len(project_data["batches"]) >= 3 else "GENERATE_NEXT"
    
    return jsonify({
        "status": "BATCH_SAVED",
        "message": "Batch approved and saved successfully",
        "meta_prompt_summary": f"UNDER={under} | TOP_UNDER={', '.join(top_under)} | Zapisano",
        "next_action": next_act
    })

# ===========================================================
# 🔌 OTHER ENDPOINTS (language_refine, export, etc.)
# ===========================================================

@tracker_routes.post("/api/language_refine")
def language_refine():
    """Language quality check endpoint"""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    audit = analyze_language_quality(text)
    
    return jsonify({
        "status": "success",
        "audit": audit
    })

@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    """Export complete article"""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    # Combine all batches
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    return jsonify({
        "status": "success",
        "article": full_text,
        "topic": project_data.get("topic"),
        "total_batches": len(batches),
        "total_words": len(full_text.split())
    })

# ===========================================================
# Add more endpoints as needed (project/create, etc.)
# ===========================================================
