# keyword_counter.py
# UNIFIED KEYWORD COUNTING - v24.3 (NeuronWriter compatible)
# 
# Jedna funkcja do liczenia fraz w CAŁYM systemie.
# 
# v24.3: ZMIANA NA OVERLAPPING (zgodność z NeuronWriter)
# - overlapping: jak NeuronWriter/Google widzi ("renta rodzinna" liczy się też jako "renta")
# - exclusive: tylko samodzielne wystąpienia (longest-match-first) - OPCJONALNE
# - inherited: ile pochodzi z dłuższych fraz
#
# DOMYŚLNIE: overlapping (use_exclusive_for_nested=False)
# - Zgadza się z NeuronWriter
# - "ubezwłasnowolnienie osoby" liczy jako: "ubezwłasnowolnienie" +1 ORAZ "osoby" +1

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Import polish_lemmatizer (Morfeusz2/spaCy)
try:
    from polish_lemmatizer import (
        init_backend, 
        lemmatize_text, 
        get_phrase_lemmas,
        get_backend_info,
        count_phrase_occurrences
    )
    _LEMMATIZER_OK = True
    init_backend()
    print(f"[KEYWORD_COUNTER] Lemmatizer loaded: {get_backend_info().get('backend', 'unknown')}")
except ImportError as e:
    _LEMMATIZER_OK = False
    print(f"[KEYWORD_COUNTER] Lemmatizer not available, using fallback: {e}")

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Wagi pozycji - konfigurowalne
DEFAULT_WEIGHTS = {
    "intro_body": 1.5,
    "h2_title": 2.0,
    "h3_title": 1.7,
    "body": 1.0,
}


@dataclass(frozen=True)
class Segment:
    """Segment tekstu z typem i tytułem."""
    kind: str
    title: str
    text: str


def _strip_html(text: str) -> str:
    """Usuwa tagi HTML i normalizuje whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_segments(raw_text: str) -> List[Segment]:
    """Dzieli tekst na segmenty według struktury H2/H3."""
    text = raw_text or ""
    lines = text.splitlines()

    segments: List[Segment] = []
    buffer: List[str] = []
    current_kind = "intro_body"
    current_title = ""

    def flush():
        nonlocal buffer, current_kind, current_title
        body = "\n".join(buffer).strip()
        if body:
            segments.append(Segment(kind=current_kind, title=current_title, text=body))
        buffer = []

    h2_pattern = re.compile(r"^\s*h2:\s*(.+)\s*$", re.IGNORECASE)
    h3_pattern = re.compile(r"^\s*h3:\s*(.+)\s*$", re.IGNORECASE)
    h2_html = re.compile(r"<h2[^>]*>([^<]+)</h2>", re.IGNORECASE)
    h3_html = re.compile(r"<h3[^>]*>([^<]+)</h3>", re.IGNORECASE)

    for ln in lines:
        m2 = h2_pattern.match(ln)
        if m2:
            flush()
            title = m2.group(1).strip()
            segments.append(Segment(kind="h2_title", title=title, text=title))
            current_kind = "body"
            current_title = title
            continue

        m3 = h3_pattern.match(ln)
        if m3:
            flush()
            title = m3.group(1).strip()
            segments.append(Segment(kind="h3_title", title=title, text=title))
            current_kind = "body"
            current_title = title
            continue

        m2_html = h2_html.search(ln)
        if m2_html:
            flush()
            title = m2_html.group(1).strip()
            segments.append(Segment(kind="h2_title", title=title, text=title))
            current_kind = "body"
            current_title = title
            continue

        m3_html = h3_html.search(ln)
        if m3_html:
            flush()
            title = m3_html.group(1).strip()
            segments.append(Segment(kind="h3_title", title=title, text=title))
            current_kind = "body"
            current_title = title
            continue

        buffer.append(ln)

    flush()
    return segments


def _count_phrase_simple(text: str, phrase: str) -> int:
    """
    Proste liczenie frazy (overlapping) - z lemmatyzacją jeśli dostępna.
    """
    if not text or not phrase:
        return 0
    
    phrase = phrase.strip()
    if not phrase:
        return 0
    
    if _LEMMATIZER_OK:
        result = count_phrase_occurrences(text, phrase)
        return result.get("count", 0)
    else:
        # Fallback: word boundaries dla bezpieczeństwa
        # Dla fraz wielowyrazowych: dopasowanie z separatorami
        words = phrase.lower().split()
        if len(words) == 1:
            # Pojedyncze słowo - word boundary
            pattern = r'\b' + re.escape(words[0]) + r'\b'
            return len(re.findall(pattern, text.lower()))
        else:
            # Wielowyrazowe - słowa oddzielone whitespace
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
            return len(re.findall(pattern, text.lower()))


def _lemmas_for_text(text: str) -> List[str]:
    """Zwraca listę lemów dla tekstu."""
    t = _strip_html(text).lower()
    if not t:
        return []
    if _LEMMATIZER_OK:
        return lemmatize_text(t)
    return [w.lower() for w in _WORD_RE.findall(t)]


def _lemmas_for_phrase(phrase: str) -> List[str]:
    """Zwraca listę lemów dla frazy kluczowej."""
    p = (phrase or "").strip().lower()
    if not p:
        return []
    if _LEMMATIZER_OK:
        return get_phrase_lemmas(p)
    return [w.lower() for w in _WORD_RE.findall(p)]


def _count_exclusive(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Liczenie EXCLUSIVE (longest-match-first).
    Konsumuje tokeny najdłuższą pasującą frazą.
    """
    # Przygotuj struktury
    kw_lemmas: Dict[str, List[str]] = {}
    kw_by_first: Dict[str, List[str]] = {}
    
    for kw in keywords:
        kw_clean = (kw or "").strip()
        if not kw_clean:
            continue
        lem = _lemmas_for_phrase(kw_clean)
        if not lem:
            continue
        kw_lemmas[kw_clean] = lem
        kw_by_first.setdefault(lem[0], []).append(kw_clean)

    # Sortuj longest-first
    for first, klist in kw_by_first.items():
        klist.sort(key=lambda k: len(kw_lemmas[k]), reverse=True)

    counts = {kw: 0 for kw in kw_lemmas.keys()}
    lemmas = _lemmas_for_text(text)
    
    i = 0
    n = len(lemmas)
    while i < n:
        first_lemma = lemmas[i]
        candidates = kw_by_first.get(first_lemma)
        
        if not candidates:
            i += 1
            continue

        matched_kw = None
        for kw in candidates:
            kw_len = len(kw_lemmas[kw])
            if i + kw_len <= n and lemmas[i:i+kw_len] == kw_lemmas[kw]:
                matched_kw = kw
                break

        if matched_kw:
            counts[matched_kw] += 1
            i += len(kw_lemmas[matched_kw])
        else:
            i += 1
    
    return counts


def count_keywords(
    text: str,
    keywords: List[str],
    *,
    weights: Optional[Dict[str, float]] = None,
    return_per_segment: bool = True,
    return_paragraph_stuffing: bool = True,
    stuffing_threshold: int = 3,
) -> Dict:
    """
    GŁÓWNA FUNKCJA - hybrydowe liczenie fraz.
    
    Zwraca:
    - overlapping: jak Google widzi (każda fraza osobno)
    - exclusive: tylko samodzielne (longest-match-first)
    - inherited: ile pochodzi z dłuższych fraz
    
    Args:
        text: Tekst do analizy
        keywords: Lista fraz kluczowych
        weights: Wagi pozycji
        return_per_segment: Czy zwrócić rozkład per sekcja
        return_paragraph_stuffing: Czy wykrywać stuffing
        stuffing_threshold: Próg stuffingu (domyślnie 3)
    
    Returns:
        {
            "overlapping": {keyword: int},      # Jak Google widzi
            "exclusive": {keyword: int},        # Tylko samodzielne
            "inherited": {keyword: int},        # Z dłuższych fraz
            "weighted": {keyword: float},       # Z wagami pozycji
            "in_headers": {keyword: int},
            "in_intro": {keyword: int},
            "stuffing_max": {keyword: int},
            "stuffing_warnings": [str],
            "per_segment": [...]
        }
    """
    weights = weights or DEFAULT_WEIGHTS
    
    # Czyść keywordy
    keywords_clean = [(kw or "").strip() for kw in keywords]
    keywords_clean = [kw for kw in keywords_clean if kw]
    
    if not keywords_clean:
        return {
            "overlapping": {},
            "exclusive": {},
            "inherited": {},
            "weighted": {},
            "in_headers": {},
            "in_intro": {},
            "stuffing_max": {},
            "stuffing_warnings": [],
            "per_segment": [],
            "lemmatizer_enabled": _LEMMATIZER_OK
        }
    
    # 1. OVERLAPPING - każda fraza osobno
    overlapping = {}
    for kw in keywords_clean:
        overlapping[kw] = _count_phrase_simple(text, kw)
    
    # 2. EXCLUSIVE - longest-match-first
    exclusive = _count_exclusive(text, keywords_clean)
    
    # 3. INHERITED = overlapping - exclusive
    inherited = {}
    for kw in keywords_clean:
        inherited[kw] = max(0, overlapping.get(kw, 0) - exclusive.get(kw, 0))
    
    # 4. Per-segment analysis (używa overlapping)
    segments = split_into_segments(text)
    in_headers = {kw: 0 for kw in keywords_clean}
    in_intro = {kw: 0 for kw in keywords_clean}
    weighted = {kw: 0.0 for kw in keywords_clean}
    per_segment = []
    
    for seg in segments:
        seg_counts = {}
        for kw in keywords_clean:
            count = _count_phrase_simple(seg.text, kw)
            if count > 0:
                seg_counts[kw] = count
        
        w = float(weights.get(seg.kind, 1.0))
        
        for kw, c in seg_counts.items():
            weighted[kw] = weighted.get(kw, 0) + c * w
            if seg.kind in ("h2_title", "h3_title"):
                in_headers[kw] = in_headers.get(kw, 0) + c
            elif seg.kind == "intro_body":
                in_intro[kw] = in_intro.get(kw, 0) + c
        
        if return_per_segment:
            per_segment.append({
                "kind": seg.kind,
                "title": seg.title,
                "counts": seg_counts,
                "weight": w,
            })
    
    # 5. Stuffing detection (używa overlapping per paragraf)
    stuffing_max = {kw: 0 for kw in keywords_clean}
    stuffing_warnings = []
    
    if return_paragraph_stuffing:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for para in paragraphs:
            for kw in keywords_clean:
                count = _count_phrase_simple(para, kw)
                if count > stuffing_max[kw]:
                    stuffing_max[kw] = count
        
        for kw, max_count in stuffing_max.items():
            if max_count > stuffing_threshold:
                stuffing_warnings.append(
                    f"⚠️ '{kw}' występuje {max_count}x w jednym akapicie - rozłóż równomiernie"
                )
    
    return {
        "overlapping": overlapping,
        "exclusive": exclusive,
        "inherited": inherited,
        "weighted": {k: round(v, 2) for k, v in weighted.items()},
        "in_headers": in_headers,
        "in_intro": in_intro,
        "stuffing_max": stuffing_max,
        "stuffing_warnings": stuffing_warnings,
        "per_segment": per_segment if return_per_segment else [],
        "lemmatizer_enabled": _LEMMATIZER_OK
    }


# ===== FUNKCJE POMOCNICZE DLA INTEGRACJI =====

def count_keywords_for_state(
    text: str, 
    keywords_state: Dict[str, dict],
    use_exclusive_for_nested: bool = False
) -> Dict[str, int]:
    """
    Wersja kompatybilna z keywords_state z Firestore.
    
    Logika:
    - Frazy MAIN: używa overlapping (pełne pokrycie)
    - Frazy BASIC które są częścią MAIN: używa exclusive (tylko samodzielne)
    - Frazy BASIC niezależne: używa overlapping
    
    Args:
        text: Tekst do analizy
        keywords_state: Dict {rid: {keyword, type, ...}}
        use_exclusive_for_nested: Czy używać exclusive dla zagnieżdżonych
    
    Returns:
        Dict {rid: count}
    """
    # Zbierz keywordy
    rid_to_keyword = {}
    keywords = []
    main_keywords = []  # Lista MAIN keywordów
    main_lemmas = {}    # {main_kw: [lemma1, lemma2, ...]}
    
    for rid, meta in keywords_state.items():
        kw = (meta.get("keyword") or "").strip()
        if kw:
            rid_to_keyword[rid] = kw
            keywords.append(kw)
            if meta.get("type", "").upper() == "MAIN" or meta.get("is_main_keyword"):
                main_keywords.append(kw)
                main_lemmas[kw] = _lemmas_for_phrase(kw)
    
    if not keywords:
        return {rid: 0 for rid in keywords_state.keys()}
    
    # Policz
    result = count_keywords(text, keywords, return_per_segment=False, return_paragraph_stuffing=False)
    overlapping = result["overlapping"]
    exclusive = result["exclusive"]
    
    def is_nested_in_main(kw: str) -> bool:
        """
        Sprawdza czy fraza jest zagnieżdżona w MAIN.
        Używa lematowych sekwencji, nie substring matching!
        
        "renta" jest nested w "renta rodzinna" bo:
          lemmas("renta") = ["renta"]
          lemmas("renta rodzinna") = ["renta", "rodzinny"]
          ["renta"] jest podciągiem ["renta", "rodzinny"]
        
        "rent" NIE jest nested w "renta rodzinna" bo:
          lemmas("rent") = ["rent"]  
          "rent" != "renta"
        """
        kw_lemmas = _lemmas_for_phrase(kw)
        if not kw_lemmas:
            return False
        
        for main_kw, m_lemmas in main_lemmas.items():
            if kw == main_kw:
                continue  # Nie porównuj z samym sobą
            if not m_lemmas:
                continue
            
            # Sprawdź czy kw_lemmas jest ciągłym podciągiem m_lemmas
            kw_len = len(kw_lemmas)
            main_len = len(m_lemmas)
            
            if kw_len >= main_len:
                continue  # Nie może być nested jeśli dłuższa lub równa
            
            # Szukaj ciągłego podciągu
            for i in range(main_len - kw_len + 1):
                if m_lemmas[i:i+kw_len] == kw_lemmas:
                    return True
        
        return False
    
    # Mapuj na rid
    # v27.2: NeuronWriter-style counting
    # WSZYSTKIE frazy używają EXCLUSIVE (longest-match-first)
    # "spadek po rodzicach" konsumuje tokeny, więc "spadek" nie dostaje +1
    batch_counts = {}
    for rid, kw in rid_to_keyword.items():
        meta = keywords_state.get(rid, {})
        kw_type = meta.get("type", "BASIC").upper()
        
        if use_exclusive_for_nested:
            # v27.2: EXCLUSIVE dla wszystkich (jak NeuronWriter)
            # Każda fraza liczona tylko gdy występuje samodzielnie
            # Nie liczona gdy jest częścią dłuższej frazy
            batch_counts[rid] = exclusive.get(kw, 0)
        else:
            # Legacy: OVERLAPPING (każda fraza osobno, nawet zagnieżdżone)
            batch_counts[rid] = overlapping.get(kw, 0)
    
    # Dla keywordów bez tekstu
    for rid in keywords_state.keys():
        if rid not in batch_counts:
            batch_counts[rid] = 0
    
    return batch_counts


def get_stuffing_warnings(text: str, keywords_state: Dict[str, dict]) -> List[str]:
    """Zwraca warningi o stuffingu."""
    keywords = [
        (meta.get("keyword") or "").strip() 
        for meta in keywords_state.values() 
        if meta.get("keyword")
    ]
    
    result = count_keywords(
        text, 
        keywords, 
        return_per_segment=False, 
        return_paragraph_stuffing=True
    )
    
    return result.get("stuffing_warnings", [])


def get_keyword_details(text: str, keywords_state: Dict[str, dict]) -> Dict:
    """
    Zwraca pełne szczegóły liczenia dla każdego keywordu.
    Do użycia w diagnostyce i UI.
    """
    keywords = []
    rid_to_kw = {}
    
    for rid, meta in keywords_state.items():
        kw = (meta.get("keyword") or "").strip()
        if kw:
            keywords.append(kw)
            rid_to_kw[rid] = kw
    
    result = count_keywords(text, keywords)
    
    details = {}
    for rid, kw in rid_to_kw.items():
        details[rid] = {
            "keyword": kw,
            "overlapping": result["overlapping"].get(kw, 0),
            "exclusive": result["exclusive"].get(kw, 0),
            "inherited": result["inherited"].get(kw, 0),
            "in_headers": result["in_headers"].get(kw, 0),
            "in_intro": result["in_intro"].get(kw, 0),
            "weighted": result["weighted"].get(kw, 0),
            "stuffing_max": result["stuffing_max"].get(kw, 0),
        }
    
    return {
        "keywords": details,
        "stuffing_warnings": result["stuffing_warnings"],
        "lemmatizer_enabled": result["lemmatizer_enabled"]
    }


# ===== PROSTE FUNKCJE DO INTEGRACJI =====
# Do użycia w final_review_routes.py i seo_optimizer.py

def count_single_keyword(text: str, keyword: str) -> int:
    """
    Liczy pojedynczy keyword w tekście.
    Używa lemmatyzacji jeśli dostępna.
    
    Do zastąpienia:
      len(re.findall(rf"\\b{re.escape(keyword)}\\b", text.lower()))
    Na:
      count_single_keyword(text, keyword)
    """
    return _count_phrase_simple(text, keyword)


def count_multiple_keywords(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Liczy wiele keywordów - zwraca overlapping counts.
    
    Do zastąpienia pętli z re.findall.
    """
    result = count_keywords(text, keywords, return_per_segment=False, return_paragraph_stuffing=False)
    return result["overlapping"]


def get_keyword_density(text: str, keywords: List[str]) -> float:
    """
    Oblicza density jako % słów kluczowych w tekście.
    Używa overlapping (jak Google widzi).
    """
    if not text:
        return 0.0
    
    # Policz słowa
    words = _WORD_RE.findall(text)
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    # Policz keyword tokens (nie frazy, tokeny!)
    keyword_tokens = 0
    for kw in keywords:
        count = _count_phrase_simple(text, kw)
        kw_words = len(kw.split())
        keyword_tokens += count * kw_words
    
    return (keyword_tokens / total_words) * 100


# ===== DIAGNOSTYKA =====

def diagnose_counting(text: str, keywords: List[str]) -> None:
    """Funkcja diagnostyczna - wyświetla szczegóły liczenia."""
    result = count_keywords(text, keywords)
    
    print("=" * 70)
    print("KEYWORD COUNTING DIAGNOSTICS (HYBRID)")
    print("=" * 70)
    print(f"Lemmatizer: {'ENABLED' if result['lemmatizer_enabled'] else 'FALLBACK'}")
    print()
    
    print("COUNTS:")
    print(f"{'Keyword':<25} {'Overlap':>8} {'Exclus':>8} {'Inherit':>8} {'Headers':>8} {'Intro':>8}")
    print("-" * 70)
    for kw in keywords:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
        o = result["overlapping"].get(kw_clean, 0)
        e = result["exclusive"].get(kw_clean, 0)
        i = result["inherited"].get(kw_clean, 0)
        h = result["in_headers"].get(kw_clean, 0)
        intro = result["in_intro"].get(kw_clean, 0)
        print(f"{kw_clean:<25} {o:>8} {e:>8} {i:>8} {h:>8} {intro:>8}")
    
    print()
    if result.get("stuffing_warnings"):
        print("STUFFING WARNINGS:")
        for w in result["stuffing_warnings"]:
            print(f"  {w}")
    
    print()
    print("LEGENDA:")
    print("  Overlap  = Jak Google widzi (każde wystąpienie frazy)")
    print("  Exclus   = Tylko samodzielne (nie w dłuższej frazie)")
    print("  Inherit  = Odziedziczone z dłuższych fraz")
    print("=" * 70)


if __name__ == "__main__":
    test_text = """
Renta rodzinna to świadczenie z ZUS dla osób, które straciły bliskiego.

h2: Czym jest renta rodzinna?

Renta rodzinna przysługuje członkom rodziny zmarłego. O rentę rodzinną 
można wnioskować w ZUS. Renta to ważne wsparcie finansowe.

h2: Kto może otrzymać rentę?

Rentę rodzinną mogą otrzymać dzieci, małżonkowie i rodzice zmarłego.
Renta renta renta - to jest stuffing w jednym akapicie.
"""
    
    keywords = ["renta rodzinna", "renta", "ZUS", "świadczenie", "rodzinna"]
    diagnose_counting(test_text, keywords)
