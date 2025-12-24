import os
import re
import json
import spacy
import textstat
from collections import Counter
from typing import List, Dict
import google.generativeai as genai
from rich import print

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
    """
    Bezpieczne wywołanie Gemini z obsługą błędów długości.
    Jeśli prompt jest za długi, przycina go i próbuje ponownie.
    """
    try:
        return model.generate_content(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        if "too large" in error_msg or "exhausted" in error_msg or "400" in error_msg:
            print(f"[SEO_OPT] ⚠️ Gemini Payload too large! Truncating input... Error: {e}")
            if max_retries > 0:
                # Drastyczne cięcie - bierzemy ostatnie 15k znaków lub pierwsze 15k
                safe_prompt = prompt[:15000] + "\n\n[TRUNCATED FOR SAFETY]"
                return safe_generate_content(model, safe_prompt, max_retries - 1)
        
        # Jeśli to inny błąd lub retries się skończyły
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
        return {
            "readability_score": score, 
            "grade_level": grade,
            "smog": smog
        }
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
        # Dodaj przecinki, popraw kapitalizację (prosta heurystyka)
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
        # Używamy safe_generate_content zamiast bezpośredniego wywołania
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
            "semantic_enrichment": {
                "keywords": keywords,
                "outline": outline,
            },
        }
    except Exception as e:
        print(f"[SEO_OPT] ❌ enrich_with_semantics error: {e}")
        return project_data

# ================================================================
# 🆕 Funkcja: Analiza rytmu akapitów (Essential for S1/S2)
# ================================================================
def detect_paragraph_rhythm(text: str) -> str:
    """
    Analizuje strukturę akapitów w tekście.
    Zwraca prosty opis rytmu (np. 'Dynamiczny', 'Monotonny', 'Zbyt długie bloki').
    """
    if not text:
        return "Brak tekstu"

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return "Brak akapitów"

    # Liczba słów w każdym akapicie
    lengths = [len(p.split()) for p in paragraphs]
    
    if not lengths:
        return "Pusty tekst"

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    # Prosta logika oceny rytmu SEO
    if max_len > 300:
        return "🚨 Zbyt długie bloki tekstu (SEO Warning)"
    
    if avg_len < 20:
        return "Dynamiczny (krótkie akapity)"
    
    if avg_len > 80:
        return "Ciężki / Akademicki"
    
    # Sprawdzenie wariancji (czy akapity są różnej długości)
    variance = max(lengths) - min(lengths)
    if variance < 10 and len(paragraphs) > 3:
        return "Monotonny (powtarzalna długość)"

    return "Zbalansowany"

# ================================================================
# 🧩 Funkcja: analiza gęstości słów kluczowych
# ================================================================
def calculate_keyword_density(text: str, keywords_state: dict) -> float:
    """
    Oblicza gęstość słów kluczowych w tekście.
    Zwraca procent (0-100).
    """
    if not text or not keywords_state:
        return 0.0
    
    text_lower = text.lower()
    total_words = len(text.split())
    
    if total_words == 0:
        return 0.0
    
    keyword_count = 0
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if keyword:
            keyword_count += len(re.findall(rf"\b{re.escape(keyword)}\b", text_lower))
    
    density = (keyword_count / total_words) * 100
    return round(density, 2)

# ================================================================
# 🧩 Funkcja: Semantic Keyword Coverage (obok n-gramów)
# ================================================================
def semantic_keyword_coverage(text: str, keywords_state: dict) -> dict:
    """
    Analizuje pokrycie słów kluczowych semantycznie (obok count_robust).
    Zwraca dict z semantic similarity scores dla każdego keyword.
    """
    if not SEMANTIC_ENABLED or not keywords_state:
        return {"semantic_enabled": False, "coverage": {}}
    
    try:
        # Embedding całego tekstu
        text_embedding = semantic_model.encode(text)
        
        coverage = {}
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            if not keyword:
                continue
            
            # Embedding słowa kluczowego
            keyword_embedding = semantic_model.encode(keyword)
            
            # Cosine similarity
            similarity = cosine_similarity(
                [text_embedding],
                [keyword_embedding]
            )[0][0]
            
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
                sum(c["semantic_similarity"] for c in coverage.values()) / len(coverage),
                3
            ) if coverage else 0.0
        }
        
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Semantic coverage error: {e}")
        return {"semantic_enabled": False, "error": str(e), "coverage": {}}

# ================================================================
# 🧩 Backward Compatibility Layer – unified_prevalidation()
# ================================================================
def unified_prevalidation(text: str, keywords_state: dict = None) -> dict:
    """
    POPRAWIONA implementacja unified_prevalidation – zgodna z v19.x API.
    + NOWE: Semantic keyword coverage analysis
    
    Wykonuje wstępną walidację i optymalizację batcha SEO przed analizą w Firestore.
    
    Args:
        text: Tekst do walidacji
        keywords_state: Słownik ze słowami kluczowymi (opcjonalny dla backward compatibility)
    
    Returns:
        Dict z wynikami walidacji
    """
    try:
        # Podstawowa optymalizacja tekstu
        result = optimize_text(text)
        
        # Wywołujemy funkcję rytmu
        rhythm = detect_paragraph_rhythm(text)
        
        # Ocena czytelności
        readability = assess_readability(text)
        
        # Obliczenie gęstości słów kluczowych (jeśli podano)
        density = 0.0
        if keywords_state:
            density = calculate_keyword_density(text, keywords_state)
        
        # ⭐ NOWE: Semantic coverage analysis
        semantic_coverage = {}
        if keywords_state and SEMANTIC_ENABLED:
            semantic_coverage = semantic_keyword_coverage(text, keywords_state)
        
        # Sprawdzenie ostrzeżeń
        warnings = []
        if "Warning" in rhythm or "🚨" in rhythm:
            warnings.append(rhythm)
        
        # Ostrzeżenie o zbyt wysokiej gęstości
        if density > 5.0:
            warnings.append(f"⚠️ Zbyt wysoka gęstość słów kluczowych: {density}%")
        
        # Mock semantic scores (dla backward compatibility)
        semantic_score = 0.85
        transition_score = 0.80
        
        return {
            "status": "success",
            "semantic_score": semantic_score,
            "transition_score": transition_score,
            "density": density,
            "smog": readability.get("smog", 0),
            "readability": readability.get("readability_score", 0),
            "optimized_text": result.get("optimized_text", text),
            "warnings": warnings,
            "semantic_coverage": semantic_coverage,  # ⭐ NOWE
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
            "smog": 0,
            "readability": 0,
            "error": str(e),
            "warnings": [str(e)],
            "optimized_text": text,
            "semantic_coverage": {"semantic_enabled": False}  # ⭐ NOWE
        }
