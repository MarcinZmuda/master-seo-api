# ================================================================
# seo_optimizer.py - Advanced SEO & Content Quality Optimizations
# v15.9 - RESTORED FULL LOGIC + FIXED SPACY MODEL
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
# 1. ROLLING CONTEXT WINDOW (Pełna wersja)
# ================================================================

def build_rolling_context(project_data, window_size=3):
    """
    Buduje kontekst strukturalny z ostatnich N batchy.
    """
    batches = project_data.get("batches", [])
    if len(batches) == 0:
        return ""
    
    recent_batches = batches[-window_size:] if len(batches) >= window_size else batches
    
    context_parts = []
    for i, batch in enumerate(recent_batches):
        batch_text = batch.get("text", "")
        
        # Fallback regex for headers
        if "<h2>" in batch_text:
             h2_pattern = r'<h2[^>]*>(.+?)</h2>'
        else:
             h2_pattern = r'##\s+(.+?)(?:\n|$)'

        headings = re.findall(h2_pattern, batch_text, re.IGNORECASE)
        snippet = batch_text[:300].replace("\n", " ").strip()
        
        meta_trace = batch.get("meta_trace", {})
        keywords_used = [k for k, count in meta_trace.items() if count > 0][:5]
        
        batch_num = len(batches) - len(recent_batches) + i + 1
        
        context_parts.append({
            "batch_num": batch_num,
            "headings": headings,
            "snippet": snippet + "...",
            "keywords": keywords_used
        })
    
    if not context_parts:
        return ""
    
    context_lines = ["KONTEKST ARTYKUŁU (ostatnie batche):"]
    for c in context_parts:
        h2_text = ", ".join(c["headings"]) if c["headings"] else "brak H2"
        kw_text = ", ".join(c["keywords"][:3]) if c["keywords"] else "brak"
        
        context_lines.append(
            f"\nBATCH {c['batch_num']}: {h2_text}\n"
            f"  Tematy: {c['snippet']}\n"
            f"  Użyte frazy: {kw_text}"
        )
    
    return "\n".join(context_lines)


# ================================================================
# 2. SEMANTIC COHERENCE CHECK (Pełna wersja)
# ================================================================

def get_embedding_safe(text):
    """Helper: Bezpieczne pobieranie embeddingu"""
    if not text or not text.strip():
        return None
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text[:8000],
            task_type="retrieval_document"
        )
        return result.get('embedding')
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def calculate_semantic_drift(new_batch_text, previous_batches, threshold=0.65):
    """
    Sprawdza, czy nowy batch nie odbiega tematycznie od reszty artykułu.
    """
    if not previous_batches or len(previous_batches) == 0:
        return {
            "drift_score": 1.0,
            "status": "OK",
            "message": "Pierwszy batch - brak porównania"
        }
    
    new_vec = get_embedding_safe(new_batch_text)
    if not new_vec:
        return {"drift_score": 0, "status": "SKIP", "message": "Brak embeddingu"}
    
    recent_batches = previous_batches[-3:] if len(previous_batches) >= 3 else previous_batches
    recent_text = " ".join([b.get("text", "") for b in recent_batches])
    
    prev_vec = get_embedding_safe(recent_text)
    if not prev_vec:
        return {"drift_score": 0, "status": "SKIP", "message": "Brak embeddingu kontekstu"}
    
    try:
        similarity = float(
            np.dot(new_vec, prev_vec) / 
            (np.linalg.norm(new_vec) * np.linalg.norm(prev_vec))
        )
    except Exception as e:
        return {"drift_score": 0, "status": "ERROR", "message": str(e)}
    
    if similarity >= threshold:
        status = "OK"
        message = f"Spójność tematyczna: {similarity*100:.1f}%"
    else:
        status = "DRIFT_WARNING"
        message = f"⚠️ Odchylenie tematyczne: {similarity*100:.1f}% (cel: >{threshold*100:.0f}%)"
    
    return {
        "drift_score": round(similarity, 3),
        "status": status,
        "message": message
    }


# ================================================================
# 3. TRANSITION QUALITY ANALYZER (Pełna wersja z NER)
# ================================================================

def analyze_transition_quality(current_batch, previous_batch):
    """
    Ocenia jakość przejścia między batches (Lexical + Transition Words + Entity Continuity).
    """
    if not previous_batch:
        return {"score": 1.0, "status": "FIRST_BATCH", "message": "Pierwszy batch"}
    
    prev_text = previous_batch.get("text", "")
    if not prev_text or not current_batch:
        return {"score": 0.5, "status": "NO_TEXT", "message": "Brak tekstu"}
    
    try:
        prev_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(prev_text) if s.strip()]
        curr_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(current_batch) if s.strip()]
    except:
        prev_sentences = [s.strip() for s in prev_text.split('.') if s.strip()]
        curr_sentences = [s.strip() for s in current_batch.split('.') if s.strip()]
    
    if not prev_sentences or not curr_sentences:
        return {"score": 0.5, "status": "NO_SENTENCES", "message": "Brak zdań"}
    
    last_prev = prev_sentences[-1].lower()
    first_curr = curr_sentences[0].lower()
    
    # 1. LEXICAL OVERLAP
    prev_words = set(re.findall(r'\w+', last_prev))
    curr_words = set(re.findall(r'\w+', first_curr))
    common_words = prev_words & curr_words
    
    lexical_overlap = len(common_words) / max(len(prev_words), len(curr_words)) if len(prev_words) > 0 else 0.0
    
    # 2. TRANSITION WORDS
    transition_words = [
        "jednak", "jednakże", "niemniej", "mimo to", "ponadto", "dodatkowo", 
        "również", "oprócz tego", "poza tym", "z kolei", "kolejnym", "następnie", 
        "dalej", "warto", "istotne", "ważne", "w związku z tym", "dlatego", "zatem"
    ]
    has_transition = any(tw in first_curr for tw in transition_words)
    
    # 3. ENTITY CONTINUITY (NER - restored logic)
    try:
        prev_doc = nlp(prev_text[-500:])
        curr_doc = nlp(current_batch[:500])
        
        prev_entities = {ent.text.lower() for ent in prev_doc.ents}
        curr_entities = {ent.text.lower() for ent in curr_doc.ents}
        
        if len(prev_entities) > 0:
            entity_overlap = len(prev_entities & curr_entities) / len(prev_entities)
        else:
            entity_overlap = 0.0
    except:
        entity_overlap = 0.0
    
    # COMBINED SCORE
    score = (lexical_overlap * 0.3 + (0.3 if has_transition else 0.0) + entity_overlap * 0.4)
    status = "SMOOTH" if score >= 0.5 else "CHOPPY"
    
    return {
        "score": round(score, 2),
        "status": status,
        "has_transition_word": has_transition,
        "entity_continuity": round(entity_overlap, 2),
        "message": f"Przejście: {status} ({score*100:.0f}%)"
    }


# ================================================================
# 4. TF-IDF POSITION-WEIGHTED SCORING (Pełna wersja)
# ================================================================

def calculate_keyword_position_score(batch_text, keyword):
    """
    Keywords w pierwszych 100 słowach = 3x ważniejsze.
    Keywords w H2/H3 = 2x ważniejsze.
    """
    if not batch_text or not keyword:
        return {"score": 0, "quality": "NONE"}
    
    doc = nlp(batch_text.lower())
    lemmas = [t.lemma_ for t in doc if t.is_alpha]
    
    keyword_doc = nlp(keyword.lower())
    keyword_lemmas = [t.lemma_ for t in keyword_doc if t.is_alpha]
    kw_len = len(keyword_lemmas)
    
    if kw_len == 0: return {"score": 0,
