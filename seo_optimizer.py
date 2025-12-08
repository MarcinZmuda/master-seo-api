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
    print("[SEO_OPT] ⚠️ GEMINI_API_KEY not set — generative features disabled")

# ================================================================
# 🧠 Ładowanie modelu spaCy (Polish)
# ================================================================
try:
    nlp = spacy.load("pl_core_news_md")
    print("[SEO_OPT] ✅ Załadowano model pl_core_news_md (Light Edition)")
except OSError:
    from spacy.cli import download
    print("[SEO_OPT] ⚠️ Model pl_core_news_md nieznaleziony — próba pobrania...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")
    print("[SEO_OPT] ✅ Model pobrany i załadowany")

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
        return {"readability_score": score, "grade_level": grade}
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Readability error: {e}")
        return {"readability_score": 0, "grade_level": 0}

# ================================================================
# 🧩 Funkcja: optymalizacja semantyczna przez Gemini
# ================================================================
def generate_semantic_outline(topic: str, keywords: List[str]) -> str:
    """Tworzy szkic SEO na podstawie tematu i słów kluczowych."""
    if not GEMINI_API_KEY:
        return "Brak API KEY — tryb offline."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Przygotuj logiczny szkic nagłówków H2/H3 dla artykułu SEO o temacie:
        "{topic}".
        Wykorzystaj możliwie dużo z tych fraz kluczowych:
        {', '.join(keywords)}

        Format:
        - H2: ...
        - H3: ...
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[SEO_OPT] ❌ Gemini Outline Error: {e}")
        return "Błąd podczas generowania outline."

# ================================================================
# 🧩 Funkcja: prewalidacja tekstu SEO
# ================================================================
def validate_batch_keywords(text: str, required_keywords: List[str]) -> Dict[str, int]:
    """Sprawdza, ile słów kluczowych z listy występuje w tekście."""
    text_lower = text.lower()
    results = {}
    for kw in required_keywords:
        results[kw] = len(re.findall(rf"\\b{re.escape(kw.lower())}\\b", text_lower))
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
        optimized_text = re.sub(r"\\s+", " ", optimized_text).strip()
        optimized_text = optimized_text[0].upper() + optimized_text[1:]
    except Exception as e:
        print(f"[SEO_OPT] ⚠️ Text cleanup failed: {e}")

    return {
        "optimized_text": optimized_text,
        "keywords_found": keywords,
        "readability_score": readability.get("readability_score", 0),
    }

# ================================================================
# 🧩 Funkcja: walidacja SEO przez AI (opcjonalnie)
# ================================================================
def ai_validate_text(text: str, topic: str = "") -> Dict[str, any]:
    """Używa Gemini do walidacji SEO tekstu pod kątem kompletności."""
    if not GEMINI_API_KEY:
        return {"status": "skipped", "reason": "Brak klucza Gemini"}

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Oceń, czy poniższy tekst dobrze pokrywa temat "{topic}".
        Zwróć ocenę od 0 do 100 i listę brakujących elementów.

        Tekst:
        {text[:8000]}
        """
        response = model.generate_content(prompt)
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
# 🧩 Backward Compatibility Layer — unified_prevalidation()
# ================================================================
def unified_prevalidation(text: str, project_id: str = None) -> dict:
    """
    Zastępcza implementacja unified_prevalidation — zgodna z v18.x API.
    Wykonuje wstępną walidację i optymalizację batcha SEO przed analizą w Firestore.
    """
    try:
        result = optimize_text(text)
        return {
            "status": "success",
            "optimized_text": result.get("optimized_text", text),
            "meta": {
                "readability_score": result.get("readability_score"),
                "keywords_found": result.get("keywords_found", []),
                "project_id": project_id,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "optimized_text": text,
        }
