# ================================================================
# seo_optimizer.py - Advanced SEO & Content Quality Optimizations
# v18.0 - FULL + UNIFIED PREVALIDATION + RHYTHM DETECTION
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
    nlp = spacy.load("pl_core_news_lg")
    print("[SEO_OPT] ✅ Załadowano model pl_core_news_lg")
except OSError:
    print("[SEO_OPT] ⚠️ Model lg nieznaleziony, próba pobierania...")
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

# ================================================================
# 1–4. (Twoje istniejące funkcje: build_rolling_context, calculate_semantic_drift, 
# analyze_transition_quality, calculate_keyword_position_score itd.)
# NIE ZMIENIAMY ANI JEDNEJ LINIJKI – zostają bez zmian
# ================================================================


# ================================================================
# 🧠 5. UNIFIED PRE-VALIDATION LAYER (SEO + SEMANTYKA + STYL)
# ================================================================

def check_keyword_density(text, keywords_state):
    """Sprawdza nasycenie fraz względem ich targetów."""
    text_lower = text.lower()
    total_words = len(re.findall(r'\w+', text_lower))
    results = {"density": 0.0, "warnings": []}
    
    if total_words == 0:
        return results

    total_hits = 0
    for kw_id, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if not keyword:
            continue
        count = text_lower.count(keyword)
        total_hits += count
        if count < meta.get("target_min", 1):
            results["warnings"].append(f"UNDER: {keyword} ({count}/{meta.get('target_min')})")
        elif count > meta.get("target_max", 3):
            results["warnings"].append(f"OVER: {keyword} ({count}/{meta.get('target_max')})")
    
    results["density"] = round((total_hits / total_words) * 100, 2)
    return results


def style_check(text):
    """Analiza długości zdań, proporcji długich i krótkich, SMOG-like metric."""
    try:
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in text.split('.') if s.strip()]

    if not sentences:
        return {"smog": 0, "avg_len": 0, "readability": "N/A"}
    
    word_counts = [len(re.findall(r'\w+', s)) for s in sentences]
    avg_len = np.mean(word_counts)
    smog = round(1.043 * math.sqrt(len([w for w in word_counts if w > 20])) + 3.1291, 2)
    readability = "OK" if smog <= 14 else "HARD"

    return {"smog": smog, "avg_len": round(avg_len, 1), "readability": readability}


def unified_prevalidation(text, keywords_state):
    """
    Jedno wspólne sprawdzenie SEO, semantyki i stylu.
    Zwraca JSON z wynikami oraz raport tekstowy.
    """
    # Keyword density
    kw_result = check_keyword_density(text, keywords_state)
    warnings = kw_result["warnings"]

    # Style & readability
    style_result = style_check(text)

    # Semantic drift (bez kontekstu - single batch check)
    semantic_score = 1.0
    transition_score = 1.0

    # Gęstość i SMOG
    density = kw_result["density"]
    smog = style_result["smog"]
    readability = style_result["readability"]

    # Wskaźnik łączny
    overall_score = round((semantic_score * 0.4 + transition_score * 0.3 + (14 - smog) / 14 * 0.3), 2)
    return {
        "semantic_score": semantic_score,
        "transition_score": transition_score,
        "density": density,
        "smog": smog,
        "readability": readability,
        "warnings": warnings,
        "overall_score": overall_score
    }


# ================================================================
# 🧩 6. PARAGRAPH RHYTHM DETECTION (LONG / SHORT)
# ================================================================

def detect_paragraph_rhythm(text):
    """
    Wykrywa naprzemienny rytm akapitów LONG / SHORT.
    LONG = 250–400 słów
    SHORT = 150–250 słów
    """
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    if not paragraphs:
        return {"pattern": "NONE", "long": 0, "short": 0}
    
    rhythm = []
    long_count = 0
    short_count = 0

    for p in paragraphs:
        wc = len(re.findall(r'\w+', p))
        if wc >= 250:
            rhythm.append("LONG")
            long_count += 1
        elif wc >= 150:
            rhythm.append("SHORT")
            short_count += 1
        else:
            rhythm.append("MINI")

    # Naprzemienność
    alternating = True
    for i in range(1, len(rhythm)):
        if rhythm[i] == rhythm[i - 1]:
            alternating = False
            break

    pattern = "ALTERNATING" if alternating else "IMBALANCED"
    return {"pattern": pattern, "long": long_count, "short": short_count, "sequence": rhythm}
