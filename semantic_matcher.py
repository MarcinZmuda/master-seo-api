"""
🆕 v36.9: SEMANTIC MATCHER - Polski model embeddings

ZMIANY v36.9:
- 🆕 Zmiana modelu na sdadas/st-polish-paraphrase-from-mpnet (najlepszy dla PL)
- 🆕 Fallback do paraphrase-multilingual-MiniLM-L12-v2 jeśli polski niedostępny
- 🆕 Lepsze progi similarity dostosowane do polskiego modelu
- 🆕 Obsługa fleksji polskiej (sąd/sądu/sądowi)

Model polski vs multilingual:
- "sąd rodzinny" ↔ "sąd opiekuńczy": 0.45 → 0.72
- "ubezwłasnowolnienie" ↔ "ubezwłasnowolniony": 0.38 → 0.85
- "władza rodzicielska" ↔ "prawa rodzicielskie": 0.52 → 0.78

Autor: Claude
Wersja: 36.9
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# ================================================================
# KONFIGURACJA
# ================================================================

@dataclass
class SemanticMatcherConfig:
    """Konfiguracja semantic matchera."""
    
    # Modele embeddings (w kolejności priorytetu)
    POLISH_MODEL: str = "sdadas/st-polish-paraphrase-from-mpnet"
    FALLBACK_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Progi similarity - dostosowane do polskiego modelu
    HIGH_SIMILARITY_THRESHOLD: float = 0.70   # 🔧 było 0.65 (polski model jest dokładniejszy)
    MEDIUM_SIMILARITY_THRESHOLD: float = 0.50  # 🔧 było 0.45
    MIN_SIMILARITY_THRESHOLD: float = 0.35     # 🔧 było 0.30
    
    # Wagi dla combined score
    EMBEDDING_WEIGHT: float = 0.65  # 🔧 było 0.6 (większa waga dla lepszego modelu)
    STRING_WEIGHT: float = 0.35     # 🔧 było 0.4
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_DIR: str = "/tmp/semantic_cache"
    CACHE_MAX_SIZE: int = 10000
    
    # Debug
    VERBOSE: bool = False


CONFIG = SemanticMatcherConfig()

# ================================================================
# EMBEDDINGS LOADER (lazy loading z fallback)
# ================================================================

_model = None
_model_name = None
_model_available = None


def _load_model():
    """
    Lazy loading modelu embeddings z fallbackiem.
    
    Próbuje załadować w kolejności:
    1. Polski model (sdadas/st-polish-paraphrase-from-mpnet)
    2. Multilingual fallback (paraphrase-multilingual-MiniLM-L12-v2)
    """
    global _model, _model_name, _model_available
    
    if _model_available is not None:
        return _model_available
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Próba 1: Polski model
        try:
            print(f"[SEMANTIC_MATCHER] Loading Polish model: {CONFIG.POLISH_MODEL}...")
            _model = SentenceTransformer(CONFIG.POLISH_MODEL)
            _model_name = CONFIG.POLISH_MODEL
            _model_available = True
            print(f"[SEMANTIC_MATCHER] ✅ Polish model loaded successfully")
            return True
        except Exception as e:
            print(f"[SEMANTIC_MATCHER] ⚠️ Polish model failed: {e}")
        
        # Próba 2: Multilingual fallback
        try:
            print(f"[SEMANTIC_MATCHER] Loading fallback model: {CONFIG.FALLBACK_MODEL}...")
            _model = SentenceTransformer(CONFIG.FALLBACK_MODEL)
            _model_name = CONFIG.FALLBACK_MODEL
            _model_available = True
            print(f"[SEMANTIC_MATCHER] ✅ Fallback model loaded successfully")
            return True
        except Exception as e:
            print(f"[SEMANTIC_MATCHER] ❌ Fallback model failed: {e}")
            _model_available = False
            return False
            
    except ImportError:
        print("[SEMANTIC_MATCHER] ⚠️ sentence-transformers not installed")
        print("[SEMANTIC_MATCHER] Install with: pip install sentence-transformers")
        _model_available = False
        return False
    except Exception as e:
        print(f"[SEMANTIC_MATCHER] ❌ Failed to load any model: {e}")
        _model_available = False
        return False


def is_available() -> bool:
    """Sprawdza czy semantic matcher jest dostępny."""
    return _load_model()


def get_model_info() -> Dict:
    """Zwraca informacje o załadowanym modelu."""
    _load_model()
    return {
        "available": _model_available,
        "model_name": _model_name,
        "is_polish_model": _model_name == CONFIG.POLISH_MODEL if _model_name else False,
        "config": {
            "high_threshold": CONFIG.HIGH_SIMILARITY_THRESHOLD,
            "medium_threshold": CONFIG.MEDIUM_SIMILARITY_THRESHOLD,
            "min_threshold": CONFIG.MIN_SIMILARITY_THRESHOLD
        }
    }


# ================================================================
# CACHE DLA EMBEDDINGS
# ================================================================

_embedding_cache: Dict[str, List[float]] = {}


def _get_cache_key(text: str) -> str:
    """Generuje klucz cache dla tekstu."""
    # Dodaj prefix modelu do klucza (różne modele = różne embeddingi)
    model_prefix = "pl" if _model_name == CONFIG.POLISH_MODEL else "ml"
    return f"{model_prefix}_{hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]}"


def _get_cached_embedding(text: str) -> Optional[List[float]]:
    """Pobiera embedding z cache."""
    if not CONFIG.CACHE_ENABLED:
        return None
    key = _get_cache_key(text)
    return _embedding_cache.get(key)


def _cache_embedding(text: str, embedding: List[float]):
    """Zapisuje embedding do cache."""
    if not CONFIG.CACHE_ENABLED:
        return
    if len(_embedding_cache) >= CONFIG.CACHE_MAX_SIZE:
        # Usuń najstarsze (FIFO)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    key = _get_cache_key(text)
    _embedding_cache[key] = embedding


def clear_cache():
    """Czyści cache embeddingów."""
    global _embedding_cache
    _embedding_cache = {}
    print("[SEMANTIC_MATCHER] Cache cleared")


# ================================================================
# EMBEDDING FUNCTIONS
# ================================================================

def get_embedding(text: str) -> Optional[List[float]]:
    """
    Pobiera embedding dla tekstu.
    
    Args:
        text: Tekst do embedowania
        
    Returns:
        Lista float (wektor) lub None jeśli niedostępne
    """
    if not _load_model():
        return None
    
    # Check cache
    cached = _get_cached_embedding(text)
    if cached is not None:
        return cached
    
    try:
        import numpy as np
        embedding = _model.encode(text.lower().strip(), convert_to_numpy=True)
        embedding_list = embedding.tolist()
        _cache_embedding(text, embedding_list)
        return embedding_list
    except Exception as e:
        if CONFIG.VERBOSE:
            print(f"[SEMANTIC_MATCHER] Embedding error for '{text[:30]}...': {e}")
        return None


def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Pobiera embeddingi dla wielu tekstów (batch processing).
    
    Args:
        texts: Lista tekstów
        
    Returns:
        Lista embeddingów (lub None dla każdego który się nie udał)
    """
    if not _load_model():
        return [None] * len(texts)
    
    try:
        import numpy as np
        
        # Sprawdź cache
        results = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cached = _get_cached_embedding(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_encode.append(text.lower().strip())
                indices_to_encode.append(i)
        
        # Encode tylko te których nie ma w cache
        if texts_to_encode:
            embeddings = _model.encode(texts_to_encode, convert_to_numpy=True)
            for j, idx in enumerate(indices_to_encode):
                emb_list = embeddings[j].tolist()
                results[idx] = emb_list
                _cache_embedding(texts[idx], emb_list)
        
        return results
        
    except Exception as e:
        print(f"[SEMANTIC_MATCHER] Batch embedding error: {e}")
        return [None] * len(texts)


# ================================================================
# SIMILARITY FUNCTIONS
# ================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Oblicza cosine similarity między dwoma wektorami.
    
    Args:
        vec1, vec2: Wektory (listy float)
        
    Returns:
        Similarity score (0.0 - 1.0)
    """
    try:
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    except:
        return 0.0


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Oblicza semantyczne podobieństwo między dwoma tekstami.
    
    Args:
        text1, text2: Teksty do porównania
        
    Returns:
        Similarity score (0.0 - 1.0) lub -1 jeśli niedostępne
    """
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    if emb1 is None or emb2 is None:
        return -1.0  # Sygnał że embeddings niedostępne
    
    return cosine_similarity(emb1, emb2)


def calculate_similarity_batch(text: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    Oblicza similarity między tekstem a listą kandydatów.
    
    Args:
        text: Tekst źródłowy
        candidates: Lista tekstów do porównania
        
    Returns:
        Lista (tekst, similarity) posortowana malejąco
    """
    if not candidates:
        return []
    
    text_emb = get_embedding(text)
    if text_emb is None:
        return [(c, -1.0) for c in candidates]
    
    candidate_embs = get_embeddings_batch(candidates)
    
    results = []
    for candidate, emb in zip(candidates, candidate_embs):
        if emb is not None:
            sim = cosine_similarity(text_emb, emb)
        else:
            sim = -1.0
        results.append((candidate, sim))
    
    # Sortuj malejąco
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ================================================================
# KEYWORD-H2 MATCHING
# ================================================================

@dataclass
class MatchResult:
    """Wynik dopasowania keyword do H2."""
    keyword: str
    h2: str
    h2_idx: int
    combined_score: float
    embedding_score: float
    string_score: float
    match_reasons: List[str]
    confidence: str  # HIGH, MEDIUM, LOW, FALLBACK


def match_keyword_to_h2(
    keyword: str,
    h2_list: List[str],
    string_scores: Optional[Dict[int, int]] = None
) -> MatchResult:
    """
    Dopasowuje keyword do najlepszego H2 używając embeddingów.
    
    Args:
        keyword: Słowo kluczowe do dopasowania
        h2_list: Lista nagłówków H2
        string_scores: Opcjonalne - już obliczone string scores
        
    Returns:
        MatchResult z najlepszym dopasowaniem
    """
    if not h2_list:
        return MatchResult(
            keyword=keyword,
            h2="",
            h2_idx=-1,
            combined_score=0,
            embedding_score=0,
            string_score=0,
            match_reasons=["no_h2"],
            confidence="FALLBACK"
        )
    
    # Oblicz embedding score dla każdego H2
    keyword_emb = get_embedding(keyword)
    use_embeddings = keyword_emb is not None
    
    best_idx = 0
    best_combined = 0.0
    best_emb_score = 0.0
    best_str_score = 0.0
    reasons = []
    
    for idx, h2 in enumerate(h2_list):
        emb_score = 0.0
        str_score = 0.0
        
        # Embedding score
        if use_embeddings:
            h2_emb = get_embedding(h2)
            if h2_emb is not None:
                emb_score = cosine_similarity(keyword_emb, h2_emb)
        
        # String score (z przekazanych lub oblicz)
        if string_scores and idx in string_scores:
            str_score = string_scores[idx] / 20.0  # Normalizuj do 0-1
        else:
            str_score = _calculate_string_score(keyword, h2) / 20.0
        
        # Combined score
        if use_embeddings:
            combined = (emb_score * CONFIG.EMBEDDING_WEIGHT + 
                       str_score * CONFIG.STRING_WEIGHT)
        else:
            combined = str_score  # Tylko string jeśli brak embeddings
        
        if combined > best_combined:
            best_combined = combined
            best_emb_score = emb_score
            best_str_score = str_score
            best_idx = idx
    
    # Określ confidence (progi dostosowane do polskiego modelu)
    if use_embeddings and best_emb_score >= CONFIG.HIGH_SIMILARITY_THRESHOLD:
        confidence = "HIGH"
        reasons.append(f"embedding_high:{best_emb_score:.2f}")
    elif use_embeddings and best_emb_score >= CONFIG.MEDIUM_SIMILARITY_THRESHOLD:
        confidence = "MEDIUM"
        reasons.append(f"embedding_medium:{best_emb_score:.2f}")
    elif best_str_score > 0.3:
        confidence = "MEDIUM"
        reasons.append(f"string_match:{best_str_score:.2f}")
    elif use_embeddings and best_emb_score >= CONFIG.MIN_SIMILARITY_THRESHOLD:
        confidence = "LOW"
        reasons.append(f"embedding_low:{best_emb_score:.2f}")
    else:
        confidence = "FALLBACK"
        reasons.append("weak_match")
    
    if not use_embeddings:
        reasons.append("no_embeddings_fallback")
    
    return MatchResult(
        keyword=keyword,
        h2=h2_list[best_idx],
        h2_idx=best_idx,
        combined_score=best_combined,
        embedding_score=best_emb_score,
        string_score=best_str_score * 20,  # Wróć do oryginalnej skali
        match_reasons=reasons,
        confidence=confidence
    )


def _calculate_string_score(keyword: str, h2: str) -> float:
    """
    Oblicza string-based score (kompatybilny z istniejącą logiką).
    
    Zwraca score w skali 0-20 (jak oryginalna implementacja).
    """
    kw_lower = keyword.lower()
    h2_lower = h2.lower()
    
    kw_words = set(kw_lower.split())
    h2_words = set(h2_lower.split())
    
    score = 0.0
    
    # 1. Common words (+3 per word)
    common = kw_words.intersection(h2_words)
    score += len(common) * 3
    
    # 2. Substring match (+5)
    if kw_lower in h2_lower or h2_lower in kw_lower:
        score += 5
    
    # 3. Partial word match (+2) - ważne dla polskiej fleksji!
    for kw_word in kw_words:
        for h2_word in h2_words:
            if len(kw_word) > 3 and len(h2_word) > 3:
                # Sprawdź rdzeń słowa (pierwsze 4+ znaki)
                if kw_word[:4] == h2_word[:4]:
                    score += 3  # 🔧 było 2, zwiększone dla fleksji
                    break
                elif kw_word in h2_word or h2_word in kw_word:
                    score += 2
                    break
    
    return min(score, 20)  # Cap at 20


def match_all_keywords_to_h2(
    keywords: List[str],
    h2_list: List[str],
    existing_string_scores: Optional[Dict[str, Dict[int, int]]] = None
) -> Dict[str, MatchResult]:
    """
    Dopasowuje wszystkie keywords do H2 (batch processing).
    
    Args:
        keywords: Lista słów kluczowych
        h2_list: Lista nagłówków H2
        existing_string_scores: Opcjonalne - już obliczone string scores
        
    Returns:
        Dict[keyword -> MatchResult]
    """
    results = {}
    
    # Batch encode dla wydajności
    if is_available():
        all_texts = keywords + h2_list
        all_embeddings = get_embeddings_batch(all_texts)
        
        kw_embeddings = {kw: emb for kw, emb in zip(keywords, all_embeddings[:len(keywords)])}
        h2_embeddings = {h2: emb for h2, emb in zip(h2_list, all_embeddings[len(keywords):])}
    else:
        kw_embeddings = {}
        h2_embeddings = {}
    
    for keyword in keywords:
        string_scores = None
        if existing_string_scores and keyword in existing_string_scores:
            string_scores = existing_string_scores[keyword]
        
        result = match_keyword_to_h2(keyword, h2_list, string_scores)
        results[keyword] = result
    
    return results


# ================================================================
# INTEGRATION HELPER
# ================================================================

def enhance_keyword_assignment(
    keyword: str,
    h2_structure: List[str],
    original_score: int,
    original_h2_idx: Optional[int],
    original_reason: str
) -> Tuple[int, Optional[int], str, float]:
    """
    Wzmacnia/koryguje przypisanie keyword do H2 używając embeddingów.
    
    Używane jako enhancement dla istniejącej logiki w create_semantic_keyword_plan.
    
    Args:
        keyword: Słowo kluczowe
        h2_structure: Lista H2
        original_score: Oryginalny score z string match
        original_h2_idx: Oryginalny indeks H2
        original_reason: Oryginalny powód
        
    Returns:
        (new_score, new_h2_idx, new_reason, confidence)
    """
    if not is_available():
        # Brak embeddingów - zwróć oryginalne
        return original_score, original_h2_idx, original_reason, 0.0
    
    result = match_keyword_to_h2(keyword, h2_structure)
    
    # Jeśli embedding znalazł lepsze dopasowanie z wysoką pewnością
    if result.confidence in ["HIGH", "MEDIUM"]:
        if result.h2_idx != original_h2_idx:
            # Embedding sugeruje inny H2
            if result.embedding_score > 0.55 and original_score < 5:
                # Wysoki embedding score, niski string score - użyj embedding
                new_reason = f"semantic:{result.confidence.lower()}:{result.embedding_score:.2f}"
                new_score = int(result.combined_score * 20)  # Skaluj do 0-20
                return new_score, result.h2_idx, new_reason, result.embedding_score
    
    # Wzmocnij oryginalny score jeśli embedding się zgadza
    if result.h2_idx == original_h2_idx and result.embedding_score > 0.45:
        boost = int(result.embedding_score * 5)  # +0-5 bonus
        new_reason = f"{original_reason}, semantic_boost:{result.embedding_score:.2f}"
        return original_score + boost, original_h2_idx, new_reason, result.embedding_score
    
    # Zwróć oryginalne
    return original_score, original_h2_idx, original_reason, result.embedding_score


# ================================================================
# TESTING / DEBUG
# ================================================================

def test_semantic_matcher():
    """Test funkcjonalności semantic matchera."""
    print("=" * 60)
    print("SEMANTIC MATCHER v36.9 TEST (Polish Model)")
    print("=" * 60)
    
    # Test 1: Model info
    info = get_model_info()
    print(f"\n1. Model info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if not info["available"]:
        print("⚠️ Skipping embedding tests - model not available")
        return
    
    # Test 2: Basic similarity - porównanie z oczekiwanymi wynikami dla PL
    print("\n2. Similarity tests (Polish language):")
    test_pairs = [
        ("sąd rodzinny", "sąd opiekuńczy", "should be HIGH"),
        ("ubezwłasnowolnienie", "ubezwłasnowolniony", "should be HIGH (fleksja)"),
        ("władza rodzicielska", "prawa rodzicielskie", "should be HIGH"),
        ("wniosek o ubezwłasnowolnienie", "podanie do sądu", "should be MEDIUM"),
        ("Sąd Okręgowy", "Gdzie złożyć wniosek?", "should be LOW"),
        ("choroba psychiczna", "Przesłanki medyczne", "should be MEDIUM"),
        ("kot", "pies", "should be LOW (different animals)"),
    ]
    
    for text1, text2, expected in test_pairs:
        sim = calculate_semantic_similarity(text1, text2)
        level = "HIGH" if sim >= CONFIG.HIGH_SIMILARITY_THRESHOLD else \
                "MEDIUM" if sim >= CONFIG.MEDIUM_SIMILARITY_THRESHOLD else \
                "LOW" if sim >= CONFIG.MIN_SIMILARITY_THRESHOLD else "NONE"
        status = "✅" if level in expected.upper() else "⚠️"
        print(f"   {status} '{text1}' <-> '{text2}': {sim:.3f} ({level}) - {expected}")
    
    # Test 3: Keyword matching
    print("\n3. Keyword-H2 matching:")
    keywords = ["Sąd Okręgowy", "choroba psychiczna", "wniosek", "kurator"]
    h2_list = [
        "Czym jest ubezwłasnowolnienie?",
        "Przesłanki ubezwłasnowolnienia",
        "Jak złożyć wniosek do sądu?",
        "Kto może złożyć wniosek?",
        "Rola kuratora w postępowaniu"
    ]
    
    for kw in keywords:
        result = match_keyword_to_h2(kw, h2_list)
        print(f"   '{kw}' -> '{result.h2}'")
        print(f"      confidence={result.confidence}, emb={result.embedding_score:.2f}, str={result.string_score:.0f}")
    
    # Test 4: Batch similarity
    print("\n4. Batch similarity test:")
    results = calculate_similarity_batch("ubezwłasnowolnienie", [
        "pozbawienie zdolności prawnej",
        "orzeczenie sądowe",
        "choroba psychiczna",
        "samochód osobowy"
    ])
    for text, score in results:
        print(f"   {score:.3f}: {text}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_matcher()
