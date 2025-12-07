# ===========================================================
# Version: v12.25.6.15.1 - KEYWORD ENFORCEMENT BLOCKING + TOTAL DENSITY
# Last updated: 2024-12-06
# Changes from v15:
# - ADDED: BLOCKING validation in approve_batch (not just warning)
# - ADDED: Total keyword density check (all BASIC keywords combined)
# - ADDED: LSI progress monitoring per batch
# - CHANGED: Unused BASIC keywords now BLOCK approval (400 error)
# - CHANGED: EXTENDED keywords still warn but don't block
# - All v15 features retained
# ===========================================================

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import datetime
import re
import json
import os
import copy
import spacy
import statistics
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
# 🆕 v12.25.6.15 - NEW VALIDATION FUNCTIONS
# ===========================================================

def validate_html_structure(text: str, expected_h2_count: int) -> dict:
    """
    Validate presence of proper HTML h2/h3 tags (not Markdown).
    Rejects if Markdown format detected.
    """
    # Check for HTML tags
    h2_found = len(re.findall(r'<h2[^>]*>.*?</h2>', text, re.DOTALL | re.IGNORECASE))
    h3_found = len(re.findall(r'<h3[^>]*>.*?</h3>', text, re.DOTALL | re.IGNORECASE))
    
    # Check for forbidden Markdown (## or ###)
    markdown_h2 = len(re.findall(r'^##\s+', text, re.MULTILINE))
    markdown_h3 = len(re.findall(r'^###\s+', text, re.MULTILINE))
    
    if markdown_h2 > 0 or markdown_h3 > 0:
        return {
            "valid": False,
            "error": f"❌ Markdown format detected ({markdown_h2} ##, {markdown_h3} ###). Use HTML: <h2></h2> and <h3></h3>",
            "markdown_found": {"h2": markdown_h2, "h3": markdown_h3},
            "suggestion": "Replace ## with <h2>Title</h2> and ### with <h3>Subtitle</h3>"
        }
    
    if h2_found == 0:
        return {
            "valid": False,
            "error": "❌ No <h2> tags found. Use proper HTML structure: <h2>Section Title</h2>",
            "expected": expected_h2_count,
            "found": 0
        }
    
    if h2_found != expected_h2_count:
        return {
            "valid": False,
            "warning": f"⚠️ Expected {expected_h2_count} <h2> tags, found {h2_found}",
            "expected": expected_h2_count,
            "found": h2_found
        }
    
    return {
        "valid": True,
        "h2_count": h2_found,
        "h3_count": h3_found,
        "status": "✅ HTML structure OK"
    }

def calculate_keyword_density(text: str, keyword: str) -> dict:
    """
    Calculate keyword density as % of total words.
    Google limit: 2.5% (warning), 3.0% (critical).
    """
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {"keyword": keyword, "density": 0, "status": "OK"}
    
    # Count using hybrid method (morphological)
    keyword_count = count_hybrid_occurrences(text, keyword, include_headings=True)
    
    density = (keyword_count / total_words) * 100
    
    # Status thresholds
    if density > 3.0:
        status = "CRITICAL"
    elif density > 2.5:
        status = "WARNING"
    else:
        status = "OK"
    
    return {
        "keyword": keyword,
        "count": keyword_count,
        "total_words": total_words,
        "density_percent": round(density, 2),
        "status": status,
        "recommended_max": int(total_words * 0.025)  # 2.5% of total
    }

def validate_all_densities(text: str, keywords_list: list) -> dict:
    """
    Check all BASIC keywords for overstuffing.
    Returns validation result with critical/warning lists.
    """
    critical = []
    warnings = []
    
    for kw_id, kw_data in keywords_list.items():
        if kw_data.get("type") == "BASIC":
            kw = kw_data["keyword"]
            result = calculate_keyword_density(text, kw)
            
            if result["status"] == "CRITICAL":
                critical.append({
                    "keyword": kw,
                    "density": result["density_percent"],
                    "count": result["count"],
                    "max_allowed": result["recommended_max"],
                    "suggestion": f"Reduce to {result['recommended_max']} uses (currently {result['count']})"
                })
            elif result["status"] == "WARNING":
                warnings.append({
                    "keyword": kw,
                    "density": result["density_percent"],
                    "count": result["count"]
                })
    
    if critical:
        return {
            "valid": False,
            "critical": critical,
            "error": f"❌ Keyword stuffing detected: {len(critical)} keywords over 3.0% density",
            "fix_suggestion": "Use synonyms: 'dokument', 'uprawnienia', 'zatrzymanie' instead of repeating main keyword"
        }
    
    return {
        "valid": True,
        "warnings": warnings,
        "status": "OK" if not warnings else "ACCEPTABLE"
    }

def validate_article_length(article_text: str, avg_competitor_length: int, h2_count: int) -> dict:
    """
    Validate article meets competitive length based on S1 competitor data.
    Minimum = 70% of competitor average (flexible).
    Optimal = 90-110% of competitor average.
    """
    current_words = len(article_text.split())
    
    # Calculate thresholds
    minimum_words = int(avg_competitor_length * 0.7)
    recommended_min = int(avg_competitor_length * 0.9)
    recommended_max = int(avg_competitor_length * 1.1)
    
    # Per-H2 average
    words_per_h2 = current_words / h2_count if h2_count > 0 else 0
    
    if current_words < minimum_words:
        gap = minimum_words - current_words
        return {
            "valid": False,
            "current_words": current_words,
            "minimum_words": minimum_words,
            "competitor_avg": avg_competitor_length,
            "gap": gap,
            "words_per_h2": int(words_per_h2),
            "error": f"❌ Article too short: {current_words}w (need {minimum_words}w minimum = 70% of top competitors)",
            "suggestion": f"Add {gap} more words (~{int(gap/h2_count)} words per section)"
        }
    
    if current_words < recommended_min:
        gap = recommended_min - current_words
        return {
            "valid": True,
            "current_words": current_words,
            "recommended_min": recommended_min,
            "competitor_avg": avg_competitor_length,
            "gap": gap,
            "warning": f"⚠️ Below optimal length: {current_words}w (competitors avg {avg_competitor_length}w)",
            "suggestion": f"Consider adding {gap} more words for better competitiveness"
        }
    
    return {
        "valid": True,
        "current_words": current_words,
        "competitor_avg": avg_competitor_length,
        "status": "✅ Competitive length",
        "percentage": round((current_words / avg_competitor_length) * 100, 1)
    }

def validate_lsi_coverage(full_article: str, keywords_state: dict) -> dict:
    """
    Check what % of LSI keywords are used.
    Target: 60%+ coverage (SEMI_MANDATORY).
    BLOCK if <40%, WARN if <60%, OK if ≥60%.
    """
    lsi_keywords = {k: v for k, v in keywords_state.items() if v.get("type") == "LSI_AUTO"}
    
    if not lsi_keywords:
        return {
            "valid": True,
            "coverage": 100,
            "message": "No LSI keywords in project"
        }
    
    used_count = 0
    unused = []
    used = []
    
    for kw_id, kw_data in lsi_keywords.items():
        kw = kw_data["keyword"]
        count = count_hybrid_occurrences(full_article, kw, include_headings=True)
        
        if count >= 1:
            used_count += 1
            used.append({"keyword": kw, "count": count})
        else:
            unused.append(kw)
    
    total_lsi = len(lsi_keywords)
    coverage = (used_count / total_lsi) * 100
    
    # BLOCK if <40%
    if coverage < 40:
        return {
            "valid": False,
            "coverage": round(coverage, 1),
            "used": used_count,
            "total": total_lsi,
            "unused": unused,
            "error": f"❌ LSI coverage too low: {coverage:.1f}% (need 60%+ minimum, currently only {used_count}/{total_lsi} used)",
            "suggestion": f"Add {len(unused)} more LSI keywords from list"
        }
    
    # WARN if <60%
    if coverage < 60:
        return {
            "valid": True,
            "coverage": round(coverage, 1),
            "used": used_count,
            "total": total_lsi,
            "unused": unused,
            "warning": f"⚠️ LSI coverage below optimal: {coverage:.1f}% (recommended 60%+, currently {used_count}/{total_lsi})"
        }
    
    return {
        "valid": True,
        "coverage": round(coverage, 1),
        "used": used_count,
        "total": total_lsi,
        "status": f"✅ Good LSI coverage: {coverage:.1f}% ({used_count}/{total_lsi} keywords used)"
    }

def suggest_burstiness_improvements(text: str, current_burstiness: float) -> list:
    """
    Analyze text and suggest specific improvements for burstiness.
    Returns actionable suggestions list.
    """
    segmenter = pysbd.Segmenter(language="pl", clean=False)
    sentences = segmenter.segment(text)
    
    if len(sentences) < 3:
        return ["Text too short to analyze (need 3+ sentences)"]
    
    lengths = [len(s.split()) for s in sentences]
    
    avg_length = statistics.mean(lengths)
    stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0
    
    suggestions = []
    
    # Check variance
    if stdev < 4:
        suggestions.append(f"❌ Very low sentence variety - almost all sentences same length (std dev: {stdev:.1f}, need 5+)")
        suggestions.append("→ Mix short (5-8w), medium (11-18w), and long (20-30w) sentences")
    
    # Check long sentences
    long_count = sum(1 for l in lengths if l >= 20)
    long_percent = (long_count / len(sentences)) * 100
    
    if long_percent < 15:
        needed = max(1, int(len(sentences) * 0.2) - long_count)
        suggestions.append(f"❌ Need {needed} more long sentences (20-30 words)")
        suggestions.append("→ Combine 2-3 short sentences into one complex sentence with clauses")
    
    # Check short sentences
    short_count = sum(1 for l in lengths if l <= 8)
    short_percent = (short_count / len(sentences)) * 100
    
    if short_percent > 60:
        suggestions.append(f"⚠️ Too many short sentences: {short_percent:.0f}% (should be max 50%)")
        suggestions.append("→ Expand 2-3 sentences with additional details or examples")
    
    # Specific recommendation based on score
    if current_burstiness < 5.0:
        suggestions.append("🔴 CRITICAL: Add 3-4 complex sentences immediately")
    elif current_burstiness < 6.0:
        suggestions.append("🟡 Add 1-2 longer sentences to reach minimum threshold (6.0)")
    
    return suggestions if suggestions else ["Text burstiness is acceptable"]

def calculate_total_keyword_density(text: str, keywords_state: dict) -> dict:
    """
    Calculate TOTAL density of all BASIC keywords combined.
    Prevents overall keyword stuffing even if individual keywords are within limits.
    
    Example: 
    - 5 BASIC keywords × 2% each = 10% total (too much!)
    - Limit: 8% total = avg ~1.5% per keyword = natural
    """
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {"valid": True, "total_density": 0}
    
    total_keyword_words = 0
    keyword_breakdown = []
    
    for kw_id, meta in keywords_state.items():
        if meta.get("type") == "BASIC":
            kw = meta["keyword"]
            count = count_hybrid_occurrences(text, kw, include_headings=True)
            kw_words = len(kw.split())
            kw_total_words = count * kw_words
            total_keyword_words += kw_total_words
            
            keyword_breakdown.append({
                "keyword": kw,
                "count": count,
                "words_contributed": kw_total_words
            })
    
    total_density = (total_keyword_words / total_words) * 100
    
    # Critical threshold: 8% total (allows avg 1.5-2% per keyword)
    if total_density > 8.0:
        return {
            "valid": False,
            "total_density": round(total_density, 2),
            "total_keyword_words": total_keyword_words,
            "total_words": total_words,
            "breakdown": keyword_breakdown,
            "error": f"❌ Total keyword density: {total_density:.1f}% (max 8.0%)",
            "suggestion": "Reduce keyword usage across all BASIC keywords or use more synonyms"
        }
    
    # Warning threshold: 6%
    if total_density > 6.0:
        return {
            "valid": True,
            "total_density": round(total_density, 2),
            "warning": f"⚠️ Total keyword density: {total_density:.1f}% (approaching limit of 8.0%)"
        }
    
    return {
        "valid": True,
        "total_density": round(total_density, 2),
        "status": f"✅ Total keyword density: {total_density:.1f}% (optimal)"
    }

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
                batch_text, 
                text_lemma_list, 
                meta.get("search_term_exact", kw.lower()), 
                meta.get("search_lemma", ""),
                include_headings=True
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
    
    # v12.25.6.15: Burstiness is now BLOCKING (not just warning)
    burstiness = audit.get("burstiness", 0)
    if burstiness < 6.0:
        burst_suggestions = suggest_burstiness_improvements(batch_text, burstiness)
        return {
            "status": "REJECTED_QUALITY",
            "error": "BURSTINESS TOO LOW",
            "burstiness_score": round(burstiness, 1),
            "minimum_required": 6.0,
            "suggestions": burst_suggestions,
            "gemini_feedback": {
                "feedback_for_writer": f"❌ Burstiness: {burstiness:.1f}/10 (need ≥6.0). Text is too monotonous. {' '.join(burst_suggestions[:2])}"
            },
            "next_action": "REWRITE"
        }
    
    if audit.get("fluff_ratio", 0) > 0.15:
        warnings.append(f"💨 Fluff {audit['fluff_ratio']:.2f} (target: <0.15)")
    
    # ⭐ FIX #3: Check if this is Batch 1 for Featured Snippet optimization
    batch_num = len(project_data.get("batches", [])) + 1
    if batch_num == 1:
        snippet_check = optimize_for_featured_snippet(batch_text)
        if not snippet_check["optimized"]:
            warnings.append(f"📌 Intro optimization: {snippet_check['recommendation']}")
    
    # Keyword tracking logic...
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
            
            found = count_hybrid_occurrences(
                batch_text, 
                text_lemma_list, 
                t_exact, 
                t_lemma,
                include_headings=True
            )
            
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

@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    """
    Validates that ALL BASIC and EXTENDED keywords are used at least once
    in the complete article (all batches combined).
    
    Returns:
        - valid: True if all required keywords used
        - missing_keywords: List of unused keywords
        - article_ready: True if article can be finalized
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    batches = project_data.get("batches", [])
    
    # Combine all batch texts
    full_article = "\n\n".join([b.get("text", "") for b in batches])
    
    # Check only BASIC and EXTENDED keywords
    missing_keywords = []
    unused_basic = []
    unused_extended = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        
        # Only validate BASIC and EXTENDED (skip LSI)
        if kw_type not in ["BASIC", "EXTENDED"]:
            continue
        
        keyword = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        
        # Check if keyword is used at least once
        if actual_uses == 0:
            missing_keywords.append({
                "keyword": keyword,
                "type": kw_type,
                "target_min": meta.get("target_min", 1),
                "target_max": meta.get("target_max", 5)
            })
            
            if kw_type == "BASIC":
                unused_basic.append(keyword)
            elif kw_type == "EXTENDED":
                unused_extended.append(keyword)
    
    is_valid = len(missing_keywords) == 0
    
    return jsonify({
        "valid": is_valid,
        "missing_keywords": missing_keywords,
        "unused_basic": unused_basic,
        "unused_extended": unused_extended,
        "article_ready": is_valid,
        "total_batches": len(batches),
        "article_length": len(full_article),
        "message": "All keywords used ✅" if is_valid else f"⚠️ {len(missing_keywords)} keywords not used yet"
    })

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
    
    # ===== NEW VALIDATIONS (v12.25.6.15) =====
    
    # 1. HTML Structure validation
    current_batch_num = len(project_data.get("batches", [])) + 1
    batch_h2_count = meta_trace.get("h2_count", 2)  # Expected H2 in this batch
    
    html_check = validate_html_structure(corrected_text, batch_h2_count)
    if not html_check["valid"]:
        return jsonify({
            "error": html_check["error"],
            "suggestion": html_check.get("suggestion", "Use HTML tags: <h2>Title</h2>"),
            "details": html_check
        }), 400
    
    # 2. Keyword Density validation (BASIC keywords only)
    density_check = validate_all_densities(corrected_text, keywords_state)
    if not density_check["valid"]:
        return jsonify({
            "error": "❌ Keyword stuffing detected",
            "overstuffed_keywords": density_check["critical"],
            "message": "Reduce keyword usage to <2.5% density",
            "suggestion": density_check.get("fix_suggestion", "Use synonyms")
        }), 400
    
    # ===== END NEW VALIDATIONS =====
    
    # 3. Total Keyword Density validation (all BASIC keywords combined)
    total_density_check = calculate_total_keyword_density(corrected_text, keywords_state)
    if not total_density_check["valid"]:
        return jsonify({
            "error": total_density_check["error"],
            "total_density": total_density_check["total_density"],
            "breakdown": total_density_check.get("breakdown", []),
            "suggestion": total_density_check.get("suggestion", "Reduce overall keyword usage")
        }), 400
    
    # 4. BLOCKING validation for unused BASIC/EXTENDED keywords
    unused_basic = []
    unused_extended = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        
        if kw_type == "BASIC" and actual_uses < target_min:
            unused_basic.append({
                "keyword": kw,
                "type": kw_type,
                "required": target_min,
                "used": actual_uses
            })
        elif kw_type == "EXTENDED" and actual_uses < target_min:
            unused_extended.append({
                "keyword": kw,
                "type": kw_type,
                "required": target_min,
                "used": actual_uses
            })
    
    # BLOCK if ANY BASIC keywords not meeting minimum
    if unused_basic:
        return jsonify({
            "error": "❌ Required BASIC keywords not used in this batch",
            "unused_keywords": unused_basic,
            "message": f"{len(unused_basic)} BASIC keywords below minimum usage",
            "suggestion": "Review batch text and integrate these keywords naturally before approval",
            "note": "Backend BLOCKS approval until all BASIC keywords meet minimum requirements"
        }), 400
    
    # 5. LSI progress monitoring (warning only, not blocking per batch)
    lsi_keywords = {k: v for k, v in keywords_state.items() if v.get("type") == "LSI_AUTO"}
    lsi_warning = None
    if lsi_keywords:
        lsi_used = sum(1 for v in lsi_keywords.values() if v.get("actual_uses", 0) >= 1)
        lsi_coverage = (lsi_used / len(lsi_keywords)) * 100
        
        if lsi_coverage < 30:
            lsi_warning = f"⚠️ LSI coverage: {lsi_coverage:.0f}% (target: 60%+ by export)"
    
    # Save batch (only if all validations passed)
    batch_entry = {
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    # Return with comprehensive status
    response = {
        "status": "BATCH_SAVED",
        "next_action": "GENERATE_NEXT",
        "validations": {
            "html_structure": "✅ Valid",
            "keyword_density": "✅ Within limits",
            "total_density": f"✅ {total_density_check['total_density']}%",
            "basic_keywords": "✅ All used",
            "extended_keywords": f"⚠️ {len(unused_extended)} still unused" if unused_extended else "✅ Complete"
        }
    }
    
    # Add warnings if applicable
    if unused_extended:
        response["warning"] = f"⚠️ {len(unused_extended)} EXTENDED keywords not used yet (not blocking)"
        response["unused_extended"] = unused_extended
        response["article_complete"] = False
    elif lsi_warning:
        response["warning"] = lsi_warning
        response["article_complete"] = False
    else:
        response["article_complete"] = True
        response["message"] = "✅ All required keywords used"
    
    if total_density_check.get("warning"):
        response["density_warning"] = total_density_check["warning"]
    
    return jsonify(response)

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
    batches = project_data.get("batches", [])
    keywords_state = project_data.get("keywords_state", {})
    
    # Combine all batches
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    # Check for unused BASIC/EXTENDED keywords
    unused_required = []
    all_keywords_stats = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        keyword = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        all_keywords_stats.append({
            "keyword": keyword,
            "type": kw_type,
            "uses": actual_uses,
            "target": f"{target_min}-{target_max}",
            "status": meta.get("status", "UNDER")
        })
        
        if kw_type in ["BASIC", "EXTENDED"] and actual_uses == 0:
            unused_required.append({
                "keyword": keyword,
                "type": kw_type
            })
    
    article_complete = len(unused_required) == 0
    
    # ===== NEW VALIDATIONS (v12.25.6.15) =====
    
    # 1. Article Length validation (from S1 competitor data)
    avg_competitor_length = project_data.get("avg_competitor_length", 2000)
    h2_structure = project_data.get("h2_structure", [])
    h2_count = len(h2_structure) if h2_structure else 7  # Default to 7 if not stored
    
    length_check = validate_article_length(full_text, avg_competitor_length, h2_count)
    if not length_check["valid"]:
        return jsonify({
            "error": length_check["error"],
            "current_words": length_check["current_words"],
            "minimum_words": length_check["minimum_words"],
            "competitor_avg": length_check["competitor_avg"],
            "gap": length_check["gap"],
            "suggestion": length_check["suggestion"],
            "article_complete": False
        }), 400
    
    # 2. LSI Coverage validation (60%+ required)
    lsi_check = validate_lsi_coverage(full_text, keywords_state)
    if not lsi_check["valid"]:
        return jsonify({
            "error": lsi_check["error"],
            "lsi_coverage": lsi_check["coverage"],
            "unused_lsi": lsi_check["unused"],
            "suggestion": lsi_check.get("suggestion", "Add more LSI keywords"),
            "article_complete": False
        }), 400
    
    # ===== END NEW VALIDATIONS =====
    
    # Add warnings to response if present
    export_warnings = []
    if "warning" in length_check:
        export_warnings.append(length_check["warning"])
    if "warning" in lsi_check:
        export_warnings.append(lsi_check["warning"])
    
    return jsonify({
        "status": "success",
        "article": full_text,
        "article_complete": article_complete,
        "total_batches": len(batches),
        "unused_required_keywords": unused_required,
        "all_keywords": all_keywords_stats,
        "length_info": {
            "current_words": length_check["current_words"],
            "competitor_avg": length_check["competitor_avg"],
            "percentage": length_check.get("percentage", 100),
            "status": length_check.get("status", "OK")
        },
        "lsi_coverage": {
            "coverage_percent": lsi_check["coverage"],
            "used": lsi_check["used"],
            "total": lsi_check["total"],
            "status": lsi_check.get("status", "OK")
        },
        "warnings": export_warnings if export_warnings else None,
        "warning": None if article_complete else f"⚠️ Article incomplete: {len(unused_required)} required keywords not used"
    })
