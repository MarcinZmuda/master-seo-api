# ================================================================
# seo_optimizer.py - Advanced SEO & Content Quality Optimizations
# v15.6 - FIX: Unified SpaCy Model (LG) & Full Functionality
# ================================================================

import re
import json
import math
import numpy as np
import spacy
import google.generativeai as genai
from collections import Counter
import pysbd

# Import from main tracker
# ⭐ FIX: Używamy pl_core_news_lg (zgodnie z Dockerfile)
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[SEO_OPT] ✅ Załadowano model pl_core_news_lg")
except OSError:
    print("[SEO_OPT] ⚠️ Model lg nieznaleziony, próba pobierania...")
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

# ================================================================
# 1. ROLLING CONTEXT WINDOW
# ================================================================

def build_rolling_context(project_data, window_size=3):
    batches = project_data.get("batches", [])
    if len(batches) == 0:
        return ""
    
    recent_batches = batches[-window_size:] if len(batches) >= window_size else batches
    context_parts = []
    
    for i, batch in enumerate(recent_batches):
        batch_text = batch.get("text", "")
        # Fallback regex for headers if HTML parsing fails or mixed content
        if "<h2>" in batch_text:
             h2_pattern = r'<h2[^>]*>(.+?)</h2>'
        else:
             h2_pattern = r'##\s+(.+?)(?:\n|$)'

        headings = re.findall(h2_pattern, batch_text, re.IGNORECASE)
        snippet = batch_text[:300].replace("\n", " ").strip()
        meta_trace = batch.get("meta_trace", {})
        keywords_used = [k for k, count in meta_trace.items() if count > 0][:5]
        
        context_parts.append({
            "batch_num": len(batches) - len(recent_batches) + i + 1,
            "headings": headings,
            "snippet": snippet + "...",
            "keywords": keywords_used
        })
    
    if not context_parts: return ""
    
    context_lines = ["KONTEKST ARTYKUŁU (ostatnie batche):"]
    for c in context_parts:
        h2_text = ", ".join(c["headings"]) if c["headings"] else "brak H2"
        context_lines.append(f"\nBATCH {c['batch_num']}: {h2_text}\n  Treść: {c['snippet']}")
    
    return "\n".join(context_lines)

# ================================================================
# 2. SEMANTIC COHERENCE & TRANSITIONS
# ================================================================

def get_embedding_safe(text):
    if not text or not text.strip(): return None
    try:
        # Adjust model name if needed based on your Gemini access
        result = genai.embed_content(model="models/text-embedding-004", content=text[:8000], task_type="retrieval_document")
        return result.get('embedding')
    except Exception: return None

def calculate_semantic_drift(new_batch_text, previous_batches, threshold=0.65):
    if not previous_batches: return {"drift_score": 1.0, "status": "OK"}
    
    new_vec = get_embedding_safe(new_batch_text)
    # Context is last 3 batches
    recent_text = " ".join([b.get("text", "") for b in previous_batches[-3:]])
    prev_vec = get_embedding_safe(recent_text)
    
    if not new_vec or not prev_vec: return {"status": "SKIP", "drift_score": 0}
    
    try:
        similarity = float(np.dot(new_vec, prev_vec) / (np.linalg.norm(new_vec) * np.linalg.norm(prev_vec)))
        status = "OK" if similarity >= threshold else "DRIFT_WARNING"
        return {"drift_score": round(similarity, 3), "status": status}
    except: return {"status": "ERROR"}

def analyze_transition_quality(current_batch, previous_batch):
    if not previous_batch: return {"score": 1.0, "status": "FIRST_BATCH"}
    
    # Simple transition word check to avoid heavy NLP here
    transition_words = ["jednak", "ponadto", "dlatego", "zatem", "w związku z tym", "kolejnym", "następnie"]
    has_transition = any(tw in current_batch[:200].lower() for tw in transition_words)
    
    return {"score": 1.0 if has_transition else 0.5, "has_transition": has_transition}

# ================================================================
# 3. FEATURED SNIPPET & POSITIONS
# ================================================================

def optimize_for_featured_snippet(batch_text, target_question=None):
    if len(batch_text) < 50: return {"optimized": False, "reason": "TOO_SHORT"}
    
    # Basic sentence split
    sentences = [s.strip() for s in batch_text.split('.') if s.strip()]
    if len(sentences) < 2: return {"optimized": False}
    
    intro = " ".join(sentences[:2])
    word_count = len(intro.split())
    
    is_opt = 40 <= word_count <= 60
    return {"optimized": is_opt, "word_count": word_count, "recommendation": "Intro 40-60 słów" if not is_opt else "OK"}

def calculate_keyword_position_score(batch_text, keyword):
    # Simplified for stability - checks presence
    if not keyword or keyword.lower() not in batch_text.lower():
        return {"score": 0, "quality": "NONE"}
    return {"score": 1, "quality": "PRESENT"}

__all__ = ['build_rolling_context', 'calculate_semantic_drift', 'analyze_transition_quality', 'calculate_keyword_position_score', 'optimize_for_featured_snippet']
