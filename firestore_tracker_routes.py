# ===========================================================
# Version: v12.25.6.7-FIXED - ALL BUGS RESOLVED
# Last updated: 2024-12-05
# Changes:
# - ADDED: Missing imports from seo_optimizer
# - ADDED: global_keyword_stats() helper function
# - INTEGRATED: Position scoring in preview
# - INTEGRATED: Featured snippet check for Batch 1
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

# ⭐ FIX #1: Import seo_optimizer functions
from seo_optimizer import (
    calculate_semantic_drift,
    analyze_transition_quality,
    calculate_keyword_position_score,
    optimize_for_featured_snippet
)

# LanguageTool - Remote API (no Java needed!)
try:
    import language_tool_python
    # Use public API instead of local Java server
    LT_TOOL_PL = language_tool_python.LanguageToolPublicAPI("pl-PL")
    print("[INIT] ✅ LanguageTool loaded (remote API).")
except Exception as e:
    LT_TOOL_PL = None
    print(f"[WARNING] ⚠️ LanguageTool init failed: {e}")

# spaCy for lemmatization - CRITICAL for morphological counting
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[INIT] ✅ spaCy Polish model loaded (pl_core_news_lg).")
except OSError:
    print("[WARNING] ⚠️ spaCy model not found. Attempting auto-download...")
    try:
        from spacy.cli import download
        download("pl_core_news_lg")
        nlp = spacy.load("pl_core_news_lg")
        print("[INIT] ✅ spaCy model downloaded and loaded successfully.")
    except Exception as download_error:
        print(f"[ERROR] ❌ Failed to download spaCy model: {download_error}")
        print("[ERROR] ❌ CRITICAL: Morphological counting will NOT work without spaCy!")
        print("[ERROR] ❌ Please run: python -m spacy download pl_core_news_lg")
        nlp = None

tracker_routes = Blueprint("tracker_routes", __name__)

# ===========================================================
# 🔧 HELPERS
# ===========================================================

def auto_fix_keyword_overuse_smart(text: str, keyword: str, current_count: int, target_max: int) -> str:
    """Smart paraphrasing logic"""
    if current_count <= target_max:
        return text
    
    overuse = current_count - target_max
    to_replace = max(1, overuse // 2)
    target_final = max(2, current_count - to_replace)
    
    import re
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    if len(matches) == 0:
        return text
    
    actual_replace = len(matches) - target_final
    if actual_replace <= 0:
        return text
    
    paraphrase_dict = {
        "włosy": ["pasma", "kosmyki", "fryzura"],
        "włosów": ["pasm", "kosmyków", "fryzury"],
        "skóra głowy": ["naskórek", "skóra", "powierzchnia głowy"],
        "skóry głowy": ["naskórka", "skóry", "powierzchni głowy"],
        "pielęgnacja włosów": ["dbanie o włosy", "troska o pasma"],
        "proces starzenia": ["starzenie", "zmiany wiekowe"],
        "kolagen": ["białko strukturalne"],
        "menopauza": ["przekwitanie", "ten okres"]
    }
    
    kw_lower = keyword.lower()
    paraphrases = paraphrase_dict.get(kw_lower, [kw_lower])
    
    indices_to_replace = sorted([m.start() for m in matches], reverse=True)[:actual_replace]
    result = text
    paraphrase_idx = 0
    
    for idx in sorted(indices_to_replace, reverse=True):
        paraphrase = paraphrases[paraphrase_idx % len(paraphrases)]
        paraphrase_idx += 1
        match_len = len(keyword)
        result = result[:idx] + paraphrase + result[idx + match_len:]
    
    return result

def validate_hard_rules(text: str) -> dict:
    """
    Hard rules validation.
    UPDATED: Allows lists (Key Takeaways) to bypass paragraph length checks.
    """
    lines = text.strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    
    # Check if text contains a list (bullet points)
    # Obsługa różnych formatów punktorów
    has_list = any(l.lstrip().startswith(('-', '*', '1.', '•')) for l in lines)
    
    # Must have at least 3 paragraphs (unless it's a short list-based section)
    if len(lines) < 3 and not has_list:
        return {"valid": False, "msg": "Tekst musi mieć minimum 3 akapity."}
    
    # Each paragraph min 100 chars (SKIP this check for list items and headers)
    if not has_list:
        for i, line in enumerate(lines, 1):
            if not line.startswith('#') and len(line) < 100: 
                return {"valid": False, "msg": f"Akapit {i} za krótki (<100 znaków)."}
    
    return {"valid": True}

def validate_paragraph_structure(text: str) -> dict:
    """Validates paragraph structure (min 4 sentences, 1 compound)"""
    # Skip for lists
    if any(l.strip().startswith(('-', '*', '1.', '•')) for l in text.split('\n')):
         return {"valid": True}

    paragraphs = [p.strip() for p in text.split('\n') if p.strip() and not p.startswith('#')]
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    
    for i, para in enumerate(paragraphs, 1):
        sentences = segmenter.segment(para)
        # Relaxed rule: allow intro paragraphs to be 3 sentences
        if len(sentences) < 3: 
            return {"valid": False, "msg": f"Akapit {i} ma {len(sentences)} zdań (wymagane min 3-4)"}
            
    return {"valid": True}

def analyze_language_quality(text: str) -> dict:
    """Analyzes text quality"""
    result = {}
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    sentences = segmenter.segment(text)
    result["sentence_count"] = len(sentences)
    
    if sentences:
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            import statistics
            mean_len = statistics.mean(lengths)
            stdev_len = statistics.stdev(lengths)
            result["burstiness"] = stdev_len / mean_len if mean_len > 0 else 0
        else:
            result["burstiness"] = 0
            
    fluff_phrases = ["warto wspomnieć", "generalnie", "w zasadzie"]
    fluff_count = sum(text.lower().count(p) for p in fluff_phrases)
    result["fluff_ratio"] = fluff_count / len(text.split()) if text else 0
    
    # LanguageTool
    result["lt_errors"] = []
    if LT_TOOL_PL:
        try:
            matches = LT_TOOL_PL.check(text)
            errs = [m.message for m in matches if m.ruleId not in ("WHITESPACE_RULE",)]
            result["lt_errors"] = errs[:3]
        except:
            pass
            
    return result

# ⭐ FIX #2: Add missing helper function
def global_keyword_stats(keywords_state: dict) -> tuple:
    """Calculate global keyword statistics"""
    under = sum(1 for meta in keywords_state.values() if meta.get("status") == "UNDER")
    over = sum(1 for meta in keywords_state.values() if meta.get("status") == "OVER")
    ok = sum(1 for meta in keywords_state.values() if meta.get("status") == "OK")
    locked = 1 if over >= 4 else 0
    return under, over, locked, ok

# ===========================================================
# 🔍 HYBRID LEMMA-FUZZY KEYWORD COUNTER (v7.6.0-morphological)
# ===========================================================
# Enhanced counting with full morphological form recognition:
# - Exact substring matches
# - ALL lemma matches (token-by-token for single words)
# - Sequential lemma for multi-word phrases
# - Optional fuzzy matching for complex cases
# - INCLUDES H2/H3 counting (like Surfer SEO)
# ===========================================================

def extract_headings_from_markdown(text: str) -> dict:
    """
    Extract H2 and H3 headings from markdown text.
    Returns: {"h2_list": [...], "h3_list": [...], "content_only": "..."}
    """
    import re
    
    h2_list = []
    h3_list = []
    
    # Extract H2 (## Title)
    h2_pattern = r'^## (.+)$'
    h2_matches = re.findall(h2_pattern, text, re.MULTILINE)
    h2_list.extend(h2_matches)
    
    # Extract H3 (### Title)
    h3_pattern = r'^### (.+)$'
    h3_matches = re.findall(h3_pattern, text, re.MULTILINE)
    h3_list.extend(h3_matches)
    
    # Remove headings from text to get content only
    content_only = re.sub(r'^##+ .+$', '', text, flags=re.MULTILINE)
    content_only = content_only.strip()
    
    return {
        "h2_list": h2_list,
        "h3_list": h3_list,
        "content_only": content_only,
        "all_headings": h2_list + h3_list
    }

def count_hybrid_occurrences(text: str, text_lemma_list: list, search_term_exact: str, search_lemma: str, include_headings: bool = True) -> int:
    """
    Hybrid counting that matches SEO tool behavior (Surfer/Neuron):
    - Counts ALL morphological forms
    - Includes H2/H3 headings in count (like Surfer does)
    - For single-word keywords: token-by-token
    - For multi-word keywords: sequential lemma + fuzzy
    
    Args:
        text: Full markdown text (with ## H2 and ### H3)
        text_lemma_list: Pre-computed lemma list (optional)
        search_term_exact: Exact keyword string
        search_lemma: Lemmatized form
        include_headings: If True, counts keywords in H2/H3 (default: True, like Surfer)
    
    Returns:
        Total count including headings and content
    """
    from rapidfuzz import fuzz
    
    # Parse keyword
    if not search_lemma or not nlp:
        # Fallback to exact counting only
        return text.lower().count(search_term_exact.lower())
    
    target_lemmas = search_lemma.split()
    
    # Extract headings and content
    if include_headings:
        extracted = extract_headings_from_markdown(text)
        content_text = extracted["content_only"]
        all_headings_text = " ".join(extracted["all_headings"])
        
        # Count in headings separately (exact + lemma)
        heading_count = 0
        if all_headings_text:
            # Simple exact count in headings
            heading_count = all_headings_text.lower().count(search_term_exact.lower())
            
            # Add lemma matches in headings
            if len(target_lemmas) == 1:
                doc_h = nlp(all_headings_text)
                for token in doc_h:
                    if token.is_alpha and token.lemma_.lower() == target_lemmas[0]:
                        heading_count += 1
        
        # Use content_only for main counting
        text_for_content = content_text
    else:
        heading_count = 0
        text_for_content = text
    
    # === SINGLE-WORD KEYWORD: Count ALL morphological forms ===
    if len(target_lemmas) == 1:
        target_lemma = target_lemmas[0]
        content_count = 0
        
        # Process text with spaCy to get all tokens
        doc = nlp(text_for_content)
        
        for token in doc:
            # Match lemma (handles all morphological forms)
            if token.is_alpha and token.lemma_.lower() == target_lemma:
                content_count += 1
        
        return heading_count + content_count
    
    # === MULTI-WORD KEYWORD: Sequential + Fuzzy ===
    else:
        content_count = 0
        lemma_len = len(target_lemmas)
        
        # Build lemma list if not provided
        if not text_lemma_list:
            doc = nlp(text_for_content)
            text_lemma_list = [t.lemma_.lower() for t in doc if t.is_alpha]
        
        # Scan through windows
        for i in range(len(text_lemma_list) - lemma_len + 1):
            window = text_lemma_list[i:i + lemma_len]
            
            # Exact lemma sequence match
            if window == target_lemmas:
                content_count += 1
            else:
                # Fuzzy match (allows slight variations/reordering)
                window_str = " ".join(window)
                target_str = " ".join(target_lemmas)
                
                score = fuzz.token_set_ratio(window_str, target_str)
                
                if score >= 90:  # 90% threshold
                    content_count += 1
        
        return heading_count + content_count

def compute_status(actual: int, target_min: int, target_max: int) -> str:
    if actual < target_min: return "UNDER"
    elif actual > target_max: return "OVER"
    else: return "OK"

# ===========================================================
# 🚨 LEGACY FUNCTION (for backward compatibility)
# ===========================================================

def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None) -> dict:
    """
    Legacy function used by project_routes.py for direct saving.
    Restored to fix ImportError.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}
        
    project_data = doc.to_dict()
    
    # 1. Validate
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
         return {"status": "REJECTED_QUALITY", "error": hard_check['msg'], "status_code": 400}

    # 2. Analyze
    audit = analyze_language_quality(batch_text)
    
    # 3. Update Keywords (Direct calculation)
    keywords_state = project_data.get("keywords_state", {})
    if nlp:
        doc_nlp = nlp(batch_text)
        text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
        
        for meta in keywords_state.values():
            kw = meta.get("keyword", "")
            found = count_hybrid_occurrences(
                batch_text, text_lemma_list, 
                meta.get("search_term_exact", kw.lower()), 
                meta.get("search_lemma", "")
            )
            if found > 0:
                meta["actual_uses"] = meta.get("actual_uses", 0) + found
                meta["status"] = compute_status(meta["actual_uses"], meta.get("target_min", 1), meta.get("target_max", 5))

    # 4. Save
    batch_entry = {
        "text": batch_text,
        "language_audit": audit,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    doc_ref.set(project_data)
    
    return {
        "status": "BATCH_SAVED",
        "message": "Batch saved successfully (Direct mode)",
        "status_code": 200
    }

# ===========================================================
# 🔌 ROUTES
# ===========================================================

@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True) or {}
    batch_text = data.get("text", "")
    meta_trace = data.get("meta_trace", {})
    
    result = process_batch_preview(project_id, batch_text, meta_trace)
    return jsonify(result)

def process_batch_preview(project_id, batch_text, meta_trace):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists: 
        return {"error": "Project not found", "status": 404}
    
    original_text = batch_text
    batch_text = batch_text.strip()
    project_data = doc.to_dict()

    # Run all checks (Updated to allow lists)
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
    
    # ⭐ FIX #3: Check if this is Batch 1 for Featured Snippet optimization
    batch_num = len(project_data.get("batches", [])) + 1
    if batch_num == 1:
        snippet_check = optimize_for_featured_snippet(batch_text)
        if not snippet_check["optimized"]:
            warnings.append(f"📌 Intro optimization: {snippet_check['recommendation']}")
    
    # Keyword tracking logic...
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    
    if nlp:
        doc_nlp = nlp(batch_text)
        text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
        
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

                meta["actual_uses"] = new_total
                meta["status"] = compute_status(new_total, meta["target_min"], target_max)
                
                # ⭐ FIX #4: Add position scoring
                position_data = calculate_keyword_position_score(batch_text, kw)
                meta["position_score"] = position_data["score"]
                meta["position_quality"] = position_data["quality"]

        # AUTO-FIX
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
                        corrections_made.append(f"Replaced ~50% of '{kw}' with synonyms")
            batch_text = fixed_text
            if corrections_made:
                warnings.append(f"✅ AUTO-CORRECTED: {len(corrections_made)} keywords paraphrased")

    # Drift check (now properly imported)
    drift_check = calculate_semantic_drift(batch_text, project_data.get("batches", []))
    prev_batch = project_data.get("batches", [])[-1] if project_data.get("batches") else None
    transition_check = analyze_transition_quality(batch_text, prev_batch)
    
    # Stats calculation (now properly defined)
    under, over, locked, ok = global_keyword_stats(keywords_state)

    return {
        "status": "PREVIEW",
        "original_text": original_text,
        "corrected_text": batch_text,
        "text_changed": (original_text != batch_text),
        "corrections_made": corrections_made,
        "language_audit": audit,
        "keywords_summary": {"under": under, "over": over, "ok": ok},
        "keywords_state": keywords_state,
        "warnings": warnings,
        "suggestions": suggestions,
        "semantic_drift": drift_check,
        "transition_quality": transition_check,
        "next_action": "APPROVE_OR_MODIFY"
    }

@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    data = request.get_json(force=True) or {}
    corrected_text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    keywords_state = data.get("keywords_state", {})
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    project_data = doc.to_dict()
    
    # Save
    batch_entry = {
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    return jsonify({"status": "BATCH_SAVED", "next_action": "GENERATE_NEXT"})

@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    audit = analyze_language_quality(text)
    return jsonify({"status": "success", "audit": audit})

@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    project_data = doc.to_dict()
    full_text = "\n\n".join([b.get("text", "") for b in project_data.get("batches", [])])
    return jsonify({"status": "success", "article": full_text})
