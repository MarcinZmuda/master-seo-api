# ================================================================
# seo_optimizer.py - Advanced SEO & Content Quality Optimizations
# v12.25.1 - FIX #1-6, #8
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
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)


# ================================================================
# FIX #1: ROLLING CONTEXT WINDOW
# ================================================================

def build_rolling_context(project_data, window_size=3):
    """
    Zamiast 500 znaków z ostatniego batcha, buduje kontekst strukturalny
    z ostatnich N batchy.
    """
    batches = project_data.get("batches", [])
    if len(batches) == 0:
        return ""
    
    recent_batches = batches[-window_size:] if len(batches) >= window_size else batches
    
    context_parts = []
    for i, batch in enumerate(recent_batches):
        batch_text = batch.get("text", "")
        
        h2_pattern = r'##\s+(.+?)(?:\n|$)'
        headings = re.findall(h2_pattern, batch_text)
        
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
# FIX #2: SEMANTIC COHERENCE CHECK
# ================================================================

def get_embedding_safe(text):
    """Helper: Bezpieczne pobieranie embeddingu (z error handling)"""
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
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": "Nie udało się utworzyć embeddingu"
        }
    
    recent_batches = previous_batches[-3:] if len(previous_batches) >= 3 else previous_batches
    recent_text = " ".join([b.get("text", "") for b in recent_batches])
    
    prev_vec = get_embedding_safe(recent_text)
    if not prev_vec:
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": "Nie udało się utworzyć embeddingu kontekstu"
        }
    
    try:
        similarity = float(
            np.dot(new_vec, prev_vec) / 
            (np.linalg.norm(new_vec) * np.linalg.norm(prev_vec))
        )
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": str(e)
        }
    
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
# FIX #3: TRANSITION QUALITY ANALYZER
# ================================================================

def analyze_transition_quality(current_batch, previous_batch):
    """
    Ocenia jakość przejścia między batches.
    """
    if not previous_batch:
        return {
            "score": 1.0,
            "status": "FIRST_BATCH",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Pierwszy batch"
        }
    
    prev_text = previous_batch.get("text", "")
    if not prev_text or not current_batch:
        return {
            "score": 0.5,
            "status": "NO_TEXT",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Brak tekstu do analizy"
        }
    
    try:
        prev_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(prev_text) if s.strip()]
        curr_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(current_batch) if s.strip()]
    except:
        prev_sentences = [s.strip() for s in prev_text.split('.') if s.strip()]
        curr_sentences = [s.strip() for s in current_batch.split('.') if s.strip()]
    
    if not prev_sentences or not curr_sentences:
        return {
            "score": 0.5,
            "status": "NO_SENTENCES",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Brak zdań"
        }
    
    last_prev = prev_sentences[-1].lower()
    first_curr = curr_sentences[0].lower()
    
    # 1. LEXICAL OVERLAP
    prev_words = set(re.findall(r'\w+', last_prev))
    curr_words = set(re.findall(r'\w+', first_curr))
    common_words = prev_words & curr_words
    
    if len(prev_words) > 0 and len(curr_words) > 0:
        lexical_overlap = len(common_words) / max(len(prev_words), len(curr_words))
    else:
        lexical_overlap = 0.0
    
    # 2. TRANSITION WORDS DETECTION
    transition_words = [
        "jednak", "jednakże", "niemniej", "mimo to",
        "ponadto", "dodatkowo", "również", "oprócz tego", "poza tym",
        "z kolei", "kolejnym", "następnie", "dalej",
        "warto", "istotne", "ważne",
        "w związku z tym", "dlatego", "zatem"
    ]
    
    has_transition = any(tw in first_curr for tw in transition_words)
    
    # 3. ENTITY CONTINUITY (NER)
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
    score = (
        lexical_overlap * 0.3 +
        (0.3 if has_transition else 0.0) +
        entity_overlap * 0.4
    )
    
    status = "SMOOTH" if score >= 0.5 else "CHOPPY"
    
    message = f"Przejście: {'płynne' if status == 'SMOOTH' else 'urywane'} ({score*100:.0f}%)"
    if not has_transition and status == "CHOPPY":
        message += " - rozważ dodanie słowa przejściowego"
    
    return {
        "score": round(score, 2),
        "status": status,
        "has_transition_word": has_transition,
        "entity_continuity": round(entity_overlap, 2),
        "lexical_overlap": round(lexical_overlap, 2),
        "message": message
    }


# ================================================================
# FIX #4: TF-IDF POSITION-WEIGHTED SCORING
# ================================================================

def calculate_keyword_position_score(batch_text, keyword):
    """
    Keywords w pierwszych 100 słowach = 3x ważniejsze (First Paragraph Boost).
    Keywords w H2/H3 = 2x ważniejsze.
    """
    if not batch_text or not keyword:
        return {
            "score": 0,
            "positions": [],
            "avg_position": 0,
            "quality": "NONE",
            "early_count": 0
        }
    
    doc = nlp(batch_text.lower())
    tokens = [t.text for t in doc if t.is_alpha]
    lemmas = [t.lemma_ for t in doc if t.is_alpha]
    
    keyword_doc = nlp(keyword.lower())
    keyword_lemmas = [t.lemma_ for t in keyword_doc if t.is_alpha]
    kw_len = len(keyword_lemmas)
    
    if kw_len == 0:
        return {
            "score": 0,
            "positions": [],
            "avg_position": 0,
            "quality": "NONE",
            "early_count": 0
        }
    
    # Find all occurrences (lemma-based)
    positions = []
    for i in range(len(lemmas) - kw_len + 1):
        if lemmas[i:i+kw_len] == keyword_lemmas:
            positions.append(i)
    
    if not positions:
        return {
            "score": 0,
            "positions": [],
            "avg_position": 0,
            "quality": "NONE",
            "early_count": 0
        }
    
    # Calculate weighted scores
    scores = []
    early_count = 0
    
    for pos in positions:
        # Waga pozycyjna
        if pos < 100:
            weight = 3.0
            early_count += 1
        elif pos < 300:
            weight = 2.0
        else:
            weight = 1.0
        
        # Check if near heading (H2/H3)
        char_pos_estimate = pos * 5
        text_before = batch_text[max(0, char_pos_estimate-100):char_pos_estimate]
        
        if "##" in text_before:
            weight *= 1.5
        
        scores.append(weight)
    
    # Aggregate
    total_score = sum(scores)
    avg_position = int(sum(positions) / len(positions))
    
    # Quality rating
    if total_score >= 5:
        quality = "EXCELLENT"
    elif total_score >= 2:
        quality = "GOOD"
    elif total_score >= 1:
        quality = "WEAK"
    else:
        quality = "NONE"
    
    return {
        "score": round(total_score, 2),
        "positions": positions,
        "avg_position": avg_position,
        "quality": quality,
        "early_count": early_count,
        "message": f"Position quality: {quality} (score: {total_score:.1f}, early: {early_count})"
    }


# ================================================================
# FIX #8: FEATURED SNIPPET OPTIMIZER
# ================================================================

def optimize_for_featured_snippet(batch_text, target_question=None):
    """
    Sprawdza, czy tekst jest zoptymalizowany pod Featured Snippet.
    """
    if not batch_text or len(batch_text.strip()) < 50:
        return {
            "optimized": False,
            "reason": "TOO_SHORT",
            "recommendation": "Tekst za krótki (min 50 znaków)"
        }
    
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(batch_text) if s.strip()]
    except:
        sentences = [s.strip() for s in batch_text.split('.') if s.strip()]
    
    if len(sentences) < 2:
        return {
            "optimized": False,
            "reason": "NOT_ENOUGH_SENTENCES",
            "recommendation": "Brak minimum 2 zdań"
        }
    
    intro = " ".join(sentences[:2])
    word_count = len(intro.split())
    
    # Keyword overlap (jeśli podano pytanie)
    keyword_overlap = 0.0
    if target_question:
        q_doc = nlp(target_question.lower())
        q_keywords = {t.lemma_ for t in q_doc if t.is_alpha and not t.is_stop}
        
        intro_doc = nlp(intro.lower())
        intro_keywords = {t.lemma_ for t in intro_doc if t.is_alpha and not t.is_stop}
        
        if len(q_keywords) > 0:
            keyword_overlap = len(q_keywords & intro_keywords) / len(q_keywords)
    
    # Criteria check
    criteria = {
        "word_count_ok": 40 <= word_count <= 60,
        "keyword_overlap_ok": keyword_overlap >= 0.6 if target_question else True,
        "has_direct_answer": True
    }
    
    optimized = all(criteria.values())
    
    # Recommendations
    recommendations = []
    if not criteria["word_count_ok"]:
        if word_count < 40:
            recommendations.append(f"Zwiększ intro do 40-60 słów (obecnie: {word_count})")
        else:
            recommendations.append(f"Skróć intro do 40-60 słów (obecnie: {word_count})")
    
    if target_question and not criteria["keyword_overlap_ok"]:
        recommendations.append(f"Zwiększ overlap z pytaniem ({keyword_overlap*100:.0f}% → cel: 60%+)")
    
    recommendation = " | ".join(recommendations) if recommendations else "Gotowe pod Featured Snippet"
    
    return {
        "optimized": optimized,
        "word_count": word_count,
        "keyword_overlap": round(keyword_overlap, 2) if target_question else None,
        "criteria": criteria,
        "recommendation": recommendation,
        "message": f"Snippet ready: {'✓' if optimized else '✗'} ({recommendation})"
    }


# ================================================================
# EXPORT ALL FUNCTIONS
# ================================================================

__all__ = [
    'build_rolling_context',
    'calculate_semantic_drift',
    'analyze_transition_quality',
    'calculate_keyword_position_score',
    'optimize_for_featured_snippet'
]
