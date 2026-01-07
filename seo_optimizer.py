"""
SEO OPTIMIZER - v25.0 BRAJEN SEO Engine

KLUCZOWA ZMIANA v25.0:
- Density obliczana z EXCLUSIVE counts (bez podwójnego liczenia zagnieżdżonych fraz)
- actual_uses nadal używa OVERLAPPING (każda fraza liczona osobno)

DLACZEGO:
- Google NIE liczy zagnieżdżonych fraz podwójnie dla density
- Ale dla trackowania każdej frazy chcemy wiedzieć ile razy występuje
"""

import os
import re
import json
import spacy
import textstat
from collections import Counter
from typing import List, Dict
import google.generativeai as genai
from rich import print

# v24.2: Unified keyword counting
try:
    from keyword_counter import count_single_keyword, count_multiple_keywords, get_keyword_density, count_keywords
    UNIFIED_COUNTER = True
    print("[SEO_OPT] ✅ Unified keyword counter loaded")
except ImportError:
    UNIFIED_COUNTER = False
    print("[SEO_OPT] ⚠️ keyword_counter not available, using legacy regex")

# ================================================================
# ⚙️ Konfiguracja środowiska
# ================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[SEO_OPT] ✅ Gemini API Key configured")
else:
    print("[SEO_OPT] ⚠️ GEMINI_API_KEY not set – generative features disabled")

# ================================================================
# 🧠 SEMANTIC EMBEDDINGS - Dodatkowa analiza
# ================================================================
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("[SEO_OPT] ✅ Sentence Transformers loaded (Semantic Analysis)")
    SEMANTIC_ENABLED = True
except ImportError:
    print("[SEO_OPT] ⚠️ Sentence Transformers not installed - semantic analysis disabled")
    SEMANTIC_ENABLED = False

# ================================================================
# 🧠 Ładowanie modelu spaCy (Polish)
# ================================================================
try:
    nlp = spacy.load("pl_core_news_md")
    print("[SEO_OPT] ✅ Załadowano model pl_core_news_md (Light Edition)")
except OSError:
    from spacy.cli import download
    print("[SEO_OPT] ⚠️ Model pl_core_news_md nieznaleziony – próba pobrania...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")
    print("[SEO_OPT] ✅ Model pobrany i załadowany")

# ================================================================
# 🛡️ HELPER: Safe Gemini Call (Anti-Crash)
# ================================================================
def safe_generate_content(model, prompt: str, max_retries=1):
    """Bezpieczne wywołanie Gemini z obsługą błędów długości."""
    try:
        return model.generate_content(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        if "too large" in error_msg or "exhausted" in error_msg or "400" in error_msg:
            print(f"[SEO_OPT] ⚠️ Gemini Payload too large! Truncating input... Error: {e}")
            if max_retries > 0:
                safe_prompt = prompt[:15000] + "\n\n[TRUNCATED FOR SAFETY]"
                return safe_generate_content(model, safe_prompt, max_retries - 1)
        print(f"[SEO_OPT] ❌ Gemini Critical Error: {e}")
        raise e

# ================================================================
# 🧩 Funkcja: ekstrakcja słów kluczowych z tekstu
# ================================================================
def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """Ekstrahuje najczęściej występujące rzeczowniki i frazy."""
    if not text.strip():
        return []
    doc = nlp(text.lower())
    words = [t.lemma_ for t in doc if t.pos_ in {"NOUN", "PROPN"} and len(t.text) > 2]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

# ================================================================
# 🧠 Funkcja: ocena czytelności
# ================================================================
def assess_readability(text: str) -> Dict[str, float]:
    """Zwraca ocenę trudności czytania tekstu."""
    try:
        score = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        smog = textstat.smog_index(text)
        return {"readability_score": score, "grade_level": grade, "smog": smog}
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Readability error: {e}")
        return {"readability_score": 0, "grade_level": 0, "smog": 0}

# ================================================================
# 🧩 Funkcja: optymalizacja semantyczna przez Gemini
# ================================================================
def generate_semantic_outline(topic: str, keywords: List[str]) -> str:
    """Tworzy szkic SEO na podstawie tematu i słów kluczowych."""
    if not GEMINI_API_KEY:
        return "Brak API KEY – tryb offline."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Przygotuj logiczny szkic nagłówków H2/H3 dla artykułu SEO o temacie:
        "{topic}".
        Wykorzystaj możliwie dużo z tych fraz kluczowych:
        {', '.join(keywords)}
        Format:
        - H2: ...
        - H3: ...
        """
        response = safe_generate_content(model, prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[SEO_OPT] ❌ Gemini Outline Error: {e}")
        return "Błąd podczas generowania outline (API Error)."

# ================================================================
# 🧩 Funkcja: prewalidacja tekstu SEO
# ================================================================
def validate_batch_keywords(text: str, required_keywords: List[str]) -> Dict[str, int]:
    """Sprawdza, ile słów kluczowych z listy występuje w tekście."""
    if UNIFIED_COUNTER:
        return count_multiple_keywords(text, required_keywords)
    else:
        text_lower = text.lower()
        results = {}
        for kw in required_keywords:
            results[kw] = len(re.findall(rf"\b{re.escape(kw.lower())}\b", text_lower))
        return results

# ================================================================
# 🧠 Funkcja: optymalizacja tekstu
# ================================================================
def optimize_text(text: str) -> Dict[str, any]:
    """Wykonuje kompleksową optymalizację SEO tekstu."""
    if not text.strip():
        return {"optimized_text": "", "readability_score": 0, "keywords_found": []}
    keywords = extract_keywords(text)
    readability = assess_readability(text)
    optimized_text = text
    try:
        optimized_text = re.sub(r"\s+", " ", optimized_text).strip()
        optimized_text = optimized_text[0].upper() + optimized_text[1:]
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Text cleanup failed: {e}")
    return {
        "optimized_text": optimized_text,
        "keywords_found": keywords,
        "readability_score": readability.get("readability_score", 0),
        "smog": readability.get("smog", 0),
    }

# ================================================================
# 🧩 Funkcja: walidacja SEO przez AI (opcjonalnie)
# ================================================================
def ai_validate_text(text: str, topic: str = "") -> Dict[str, any]:
    """Używa Gemini do walidacji SEO tekstu pod kątem kompletności."""
    if not GEMINI_API_KEY:
        return {"status": "skipped", "reason": "Brak klucza Gemini"}
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Oceń, czy poniższy tekst dobrze pokrywa temat "{topic}".
        Zwróć ocenę od 0 do 100 i listę brakujących elementów.
        Tekst:
        {text[:25000]} 
        """
        response = safe_generate_content(model, prompt)
        return {"status": "ok", "validation_result": response.text}
    except Exception as e:
        print(f"[SEO_OPT] ❌ AI validation failed: {e}")
        return {"status": "error", "error": str(e)}

# ================================================================
# 🧩 Pomocnicza funkcja: scalanie danych do Firestore
# ================================================================
def enrich_with_semantics(project_data: dict, text: str) -> dict:
    """Dodaje metadane semantyczne do projektu SEO."""
    try:
        keywords = extract_keywords(text)
        outline = generate_semantic_outline(project_data.get("topic", ""), keywords)
        return {
            **project_data,
            "semantic_data": {
                "extracted_keywords": keywords,
                "suggested_outline": outline,
            },
        }
    except Exception as e:
        print(f"[SEO_OPT] ❌ enrich_with_semantics error: {e}")
        return project_data

# ================================================================
# 🆕 Funkcja: Analiza rytmu akapitów
# ================================================================
def detect_paragraph_rhythm(text: str) -> str:
    """Analizuje strukturę akapitów w tekście."""
    if not text:
        return "Brak tekstu"
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return "Brak akapitów"
    lengths = [len(p.split()) for p in paragraphs]
    if not lengths:
        return "Pusty tekst"
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    if max_len > 300:
        return "🚨 Zbyt długie bloki tekstu (SEO Warning)"
    if avg_len < 20:
        return "Dynamiczny (krótkie akapity)"
    if avg_len > 80:
        return "Ciężki / Akademicki"
    variance = max(lengths) - min(lengths)
    if variance < 10 and len(paragraphs) > 3:
        return "Monotonny (powtarzalna długość)"
    return "Zbalansowany"

# ================================================================
# v25.0: NAPRAWIONE obliczanie gęstości słów kluczowych
# ================================================================
def calculate_keyword_density(text: str, keywords_state: dict) -> float:
    """
    v27.3: Wrapper dla kompatybilności - zwraca density frazy głównej.
    Pełne dane w calculate_keyword_density_detailed().
    """
    result = calculate_keyword_density_detailed(text, keywords_state)
    return result.get("main_keyword_density", 0.0)


def calculate_keyword_density_detailed(text: str, keywords_state: dict) -> dict:
    """
    v27.3: Density per keyword (jak NeuronWriter).
    
    Każda fraza ma SWOJE density:
    - "spadek po rodzicach" = 2.4%
    - "spadek" = 1.5%
    - "zachowek" = 0.2%
    
    Returns:
        {
            "main_keyword_density": 2.4,
            "max_density": 3.1,
            "avg_density": 0.8,
            "status": "OK" | "WARNING" | "STUFFING",
            "warnings": ["fraza: X% > limit"],
            "per_keyword": {
                "fraza1": {"count": 8, "density": 2.4, "status": "OK"},
                ...
            }
        }
    """
    if not text or not keywords_state:
        return {
            "main_keyword_density": 0.0,
            "max_density": 0.0,
            "avg_density": 0.0,
            "status": "OK",
            "warnings": [],
            "per_keyword": {}
        }
    
    total_words = len(text.split())
    if total_words == 0:
        return {
            "main_keyword_density": 0.0,
            "max_density": 0.0,
            "avg_density": 0.0,
            "status": "OK",
            "warnings": [],
            "per_keyword": {}
        }
    
    # Znajdź frazę główną
    main_keyword = None
    for meta in keywords_state.values():
        if meta.get("is_main_keyword") or meta.get("type", "").upper() == "MAIN":
            main_keyword = meta.get("keyword", "")
            break
    
    # Zbierz wszystkie frazy
    keywords = []
    kw_meta = {}
    for rid, meta in keywords_state.items():
        kw = (meta.get("keyword") or "").strip()
        if kw:
            keywords.append(kw)
            kw_meta[kw] = {
                "type": meta.get("type", "BASIC"),
                "is_main": meta.get("is_main_keyword", False) or kw == main_keyword
            }
    
    if not keywords:
        return {
            "main_keyword_density": 0.0,
            "max_density": 0.0,
            "avg_density": 0.0,
            "status": "OK",
            "warnings": [],
            "per_keyword": {}
        }
    
    # Policz wszystkie frazy naraz (wydajne!)
    if UNIFIED_COUNTER:
        result = count_keywords(text, keywords, return_per_segment=False, return_paragraph_stuffing=False)
        counts = result.get("exclusive", {})  # EXCLUSIVE dla density
    else:
        # Fallback
        counts = {}
        text_lower = text.lower()
        for kw in keywords:
            kw_lower = kw.lower()
            counts[kw] = len(re.findall(rf"\b{re.escape(kw_lower)}\b", text_lower))
    
    # Oblicz density dla każdej frazy
    per_keyword = {}
    densities = []
    warnings = []
    main_keyword_density = 0.0
    
    # Limity density per fraza
    DENSITY_OK = 2.0
    DENSITY_WARNING = 3.0
    DENSITY_STUFFING = 4.0
    
    for kw in keywords:
        count = counts.get(kw, 0)
        kw_words = len(kw.split())
        kw_tokens = count * kw_words
        kw_density = round((kw_tokens / total_words) * 100, 2)
        
        # Status dla tej frazy
        if kw_density > DENSITY_STUFFING:
            status = "STUFFING"
            warnings.append(f"🔴 {kw}: {kw_density}% (STUFFING!)")
        elif kw_density > DENSITY_WARNING:
            status = "WARNING"
            warnings.append(f"🟠 {kw}: {kw_density}% (za wysoko)")
        elif kw_density > DENSITY_OK:
            status = "HIGH"
        else:
            status = "OK"
        
        per_keyword[kw] = {
            "count": count,
            "density": kw_density,
            "status": status
        }
        
        if count > 0:
            densities.append(kw_density)
        
        # Zapisz density frazy głównej
        if kw_meta.get(kw, {}).get("is_main"):
            main_keyword_density = kw_density
    
    # Oblicz max i średnią
    max_density = max(densities) if densities else 0.0
    avg_density = round(sum(densities) / len(densities), 2) if densities else 0.0
    
    # Status ogólny
    if max_density > DENSITY_STUFFING:
        overall_status = "STUFFING"
    elif max_density > DENSITY_WARNING:
        overall_status = "WARNING"
    elif max_density > DENSITY_OK:
        overall_status = "HIGH"
    else:
        overall_status = "OK"
    
    return {
        "main_keyword_density": main_keyword_density,
        "max_density": max_density,
        "avg_density": avg_density,
        "status": overall_status,
        "warnings": warnings[:5],  # Max 5 warnings
        "per_keyword": per_keyword,
        "total_words": total_words
    }


# ================================================================
# 🧩 Funkcja: Semantic Keyword Coverage
# ================================================================
def semantic_keyword_coverage(text: str, keywords_state: dict) -> dict:
    """Analizuje pokrycie słów kluczowych semantycznie."""
    if not SEMANTIC_ENABLED or not keywords_state:
        return {"semantic_enabled": False, "coverage": {}}
    
    try:
        text_embedding = semantic_model.encode(text)
        coverage = {}
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            keyword_embedding = semantic_model.encode(keyword)
            similarity = cosine_similarity([text_embedding], [keyword_embedding])[0][0]
            coverage[keyword] = {
                "semantic_similarity": round(float(similarity), 3),
                "status": "COVERED" if similarity > 0.50 else "WEAK",
                "actual_uses": meta.get("actual_uses", 0),
                "type": meta.get("type", "BASIC")
            }
        return {
            "semantic_enabled": True,
            "coverage": coverage,
            "avg_similarity": round(
                sum(c["semantic_similarity"] for c in coverage.values()) / len(coverage), 3
            ) if coverage else 0.0
        }
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Semantic coverage error: {e}")
        return {"semantic_enabled": False, "error": str(e), "coverage": {}}


# ================================================================
# 🧩 unified_prevalidation() - główna funkcja walidacji
# ================================================================
def unified_prevalidation(text: str, keywords_state: dict = None) -> dict:
    """
    v25.0: Wykonuje wstępną walidację batcha SEO.
    
    WAŻNE: Density obliczana z EXCLUSIVE (bez duplikatów zagnieżdżonych fraz).
    """
    try:
        result = optimize_text(text)
        rhythm = detect_paragraph_rhythm(text)
        readability = assess_readability(text)
        
        # v27.3: Density per keyword (jak NeuronWriter)
        density = 0.0
        density_details = {}
        if keywords_state:
            density_details = calculate_keyword_density_detailed(text, keywords_state)
            density = density_details.get("main_keyword_density", 0.0)
        
        # Semantic coverage
        semantic_coverage = {}
        if keywords_state and SEMANTIC_ENABLED:
            semantic_coverage = semantic_keyword_coverage(text, keywords_state)
        
        # Warnings
        warnings = []
        if "Warning" in rhythm or "🚨" in rhythm:
            warnings.append(rhythm)
        
        # v27.3: Warnings z density_details (per keyword)
        if density_details.get("warnings"):
            warnings.extend(density_details["warnings"])
        
        # Ogólny status density (bazowany na max)
        max_density = density_details.get("max_density", 0.0)
        if max_density > 4.0:
            warnings.append(f"🔴 MAX DENSITY: {max_density:.1f}% (STUFFING!)")
        elif max_density > 3.0:
            warnings.append(f"🟠 MAX DENSITY: {max_density:.1f}% (za wysoko)")
        
        semantic_score = 0.85
        transition_score = 0.80
        
        return {
            "status": "success",
            "semantic_score": semantic_score,
            "transition_score": transition_score,
            "density": density,  # Main keyword density (kompatybilność)
            "density_details": density_details,  # v27.3: Pełne dane
            "smog": readability.get("smog", 0),
            "readability": readability.get("readability_score", 0),
            "optimized_text": result.get("optimized_text", text),
            "warnings": warnings,
            "semantic_coverage": semantic_coverage,
            "meta": {
                "readability_score": readability.get("readability_score"),
                "grade_level": readability.get("grade_level"),
                "keywords_found": result.get("keywords_found", []),
                "paragraph_rhythm": rhythm,
            },
        }
    except Exception as e:
        print(f"[SEO_OPT] ❌ unified_prevalidation error: {e}")
        return {
            "status": "error",
            "semantic_score": 0,
            "transition_score": 0,
            "density": 0,
            "density_details": {},
            "smog": 0,
            "readability": 0,
            "error": str(e),
            "warnings": [str(e)],
            "optimized_text": text,
            "semantic_coverage": {"semantic_enabled": False}
        }
