# ================================================================
# 🧠 SEO Optimizer — SpaCy + Semantic Engine v19.5 (Light Edition)
# ================================================================
# Ładowanie SpaCy w trybie "bezpiecznym" (bez runtime download)
# Obsługa NLP dla języka polskiego — z pl_core_news_md
# ================================================================

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from rich import print
import textstat
import re
from typing import List, Dict

# ================================================================
# 🧩 SAFE MODEL LOADER — bezpieczne ładowanie modelu SpaCy
# ================================================================
def load_polish_model():
    """
    Bezpieczne ładowanie modelu SpaCy dla języka polskiego.
    Używa pl_core_news_md (średni model ~200 MB).
    Nigdy nie pobiera dużego modelu 'lg' w runtime (oszczędność RAM).
    """
    try:
        nlp = spacy.load("pl_core_news_md")
        print("[SEO_OPT] ✅ Załadowano model pl_core_news_md (Light Edition)")
        return nlp
    except OSError:
        try:
            print("[SEO_OPT] ⚠️ Model MD nieznaleziony, próba pobierania...")
            from spacy.cli import download
            download("pl_core_news_md")
            nlp = spacy.load("pl_core_news_md")
            return nlp
        except Exception as e:
            print("[SEO_OPT] ❌ Błąd przy ładowaniu modelu SpaCy:", e)
            raise SystemExit("❌ Nie udało się załadować modelu NLP. Zatrzymano proces.")

# Inicjalizacja globalnego modelu SpaCy
nlp = load_polish_model()


# ================================================================
# 🔍 Keyword density & semantic checks
# ================================================================
def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """
    Oblicza gęstość słów kluczowych (w %) dla zadanej listy fraz.
    """
    text_lower = text.lower()
    total_words = len(text.split())
    densities = {}

    for kw in keywords:
        count = len(re.findall(rf"\b{re.escape(kw.lower())}\b", text_lower))
        densities[kw] = round((count / total_words) * 100, 2) if total_words > 0 else 0.0

    return densities


# ================================================================
# 🧠 Semantic similarity checks
# ================================================================
def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Oblicza semantyczne podobieństwo (cosine similarity) między dwoma tekstami.
    """
    doc_a = nlp(text_a)
    doc_b = nlp(text_b)
    similarity = round(doc_a.similarity(doc_b), 4)
    return similarity


# ================================================================
# 🧮 Readability metrics (SMOG / FOG / Flesch)
# ================================================================
def compute_readability_metrics(text: str) -> Dict[str, float]:
    """
    Oblicza podstawowe wskaźniki czytelności.
    """
    metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "gunning_fog": textstat.gunning_fog(text),
        "avg_sentence_length": textstat.avg_sentence_length(text),
    }
    return metrics


# ================================================================
# 🧱 Keyword phrase matcher
# ================================================================
def find_keyword_occurrences(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Znajduje wystąpienia fraz kluczowych w tekście (dokładne dopasowania).
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(kw) for kw in keywords]
    matcher.add("KEYWORDS", patterns)

    doc = nlp(text)
    matches = matcher(doc)
    occurrences = {}

    for match_id, start, end in matches:
        span = doc[start:end].text
        occurrences[span.lower()] = occurrences.get(span.lower(), 0) + 1

    return occurrences


# ================================================================
# 🧩 SEO Optimizer Core
# ================================================================
def analyze_text(text: str, keywords: List[str]) -> Dict:
    """
    Główna funkcja optymalizacji SEO:
    - Liczy wystąpienia słów kluczowych
    - Oblicza gęstość
    - Analizuje czytelność
    - Zwraca wyniki w formacie JSON-ready
    """
    if not text.strip():
        return {"error": "Brak treści do analizy"}

    occurrences = find_keyword_occurrences(text, keywords)
    density = calculate_keyword_density(text, keywords)
    readability = compute_readability_metrics(text)

    report = {
        "keyword_occurrences": occurrences,
        "keyword_density": density,
        "readability": readability,
        "total_words": len(text.split()),
        "unique_keywords_used": len([k for k, v in occurrences.items() if v > 0]),
    }

    print("[SEO_OPT] 🔍 Analiza SEO zakończona pomyślnie.")
    return report


# ================================================================
# 🧠 Semantic drift checker
# ================================================================
def check_semantic_drift(reference_text: str, generated_text: str) -> Dict:
    """
    Sprawdza, czy wygenerowany tekst nie odchodzi semantycznie od oryginału.
    """
    similarity = compute_semantic_similarity(reference_text, generated_text)
    drift = round((1 - similarity) * 100, 2)
    status = "OK" if similarity >= 0.75 else "DRIFT"

    result = {
        "semantic_similarity": similarity,
        "drift_percent": drift,
        "status": status,
    }

    print(f"[SEO_OPT] 🧩 Semantyka: {similarity} ({status})")
    return result


# ================================================================
# 🧪 Local test entrypoint (optional)
# ================================================================
if __name__ == "__main__":
    sample_text = """
    Prawo jazdy to dokument potwierdzający uprawnienia do prowadzenia pojazdów mechanicznych.
    Aby je uzyskać, należy zdać egzamin teoretyczny i praktyczny w ośrodku WORD.
    """
    keywords = ["prawo jazdy", "egzamin", "WORD"]

    report = analyze_text(sample_text, keywords)
    print("\n=== SEO REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")
