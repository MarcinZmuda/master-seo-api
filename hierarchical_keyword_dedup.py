"""
===============================================================================
🔢 HIERARCHICAL KEYWORD DEDUPLICATION v23.9.2
===============================================================================
Rozwiązuje problem podwójnego liczenia fraz zagnieżdżonych.

Problem:
  "renta rodzinna" zawiera "renta"
  Tekst: "renta rodzinna jest świadczeniem" 
  → Stary system: renta=1, renta rodzinna=1 (podwójne liczenie!)
  → Nowy system: renta=0, renta rodzinna=1 (deduplikacja)

Algorytm:
  1. Posortuj frazy od najdłuższych do najkrótszych
  2. Dla każdej frazy krótkiej, odejmij wystąpienia w dłuższych frazach
  3. Zwróć skorygowane liczniki

Przykład:
  Frazy: ["renta", "renta rodzinna", "renta wdowia"]
  Tekst zawiera: "renta rodzinna" 3x, "renta wdowia" 2x, "renta" samodzielnie 5x
  
  Surowe liczniki: renta=10, renta rodzinna=3, renta wdowia=2
  Po deduplikacji: renta=5, renta rodzinna=3, renta wdowia=2
  
  Bo: 10 - 3 - 2 = 5 (samodzielnych wystąpień "renta")
===============================================================================
"""

import re
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Normalizuje tekst do porównań."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower().strip()
    return text


def count_phrase_raw(text: str, phrase: str) -> int:
    """Liczy surowe wystąpienia frazy (bez deduplikacji)."""
    text_lower = text.lower()
    phrase_lower = phrase.lower().strip()
    
    if not phrase_lower:
        return 0
    
    # Dopasowanie z tolerancją na odmianę (prefix matching)
    words = phrase_lower.split()
    if len(words) == 1:
        # Pojedyncze słowo - prefix match
        pattern = r'\b' + re.escape(words[0][:4]) + r'\w*\b'
    else:
        # Wielowyrazowa fraza - każde słowo prefix match, max 2 słowa między
        stems = [re.escape(w[:4]) + r'\w*' for w in words]
        pattern = r'\b' + r'\s+(?:\w+\s+){0,2}'.join(stems) + r'\b'
    
    return len(re.findall(pattern, text_lower, re.IGNORECASE))


def is_subphrase(short: str, long: str) -> bool:
    """Sprawdza czy krótsza fraza jest częścią dłuższej."""
    short_words = set(short.lower().split())
    long_words = set(long.lower().split())
    
    # Krótsza musi mieć mniej słów i wszystkie jej słowa muszą być w dłuższej
    if len(short_words) >= len(long_words):
        return False
    
    # Sprawdź czy stemmy krótszej są w dłuższej
    short_stems = {w[:4] for w in short_words if len(w) >= 4}
    long_stems = {w[:4] for w in long_words if len(w) >= 4}
    
    return short_stems.issubset(long_stems)


def deduplicate_keyword_counts(
    text: str, 
    keywords: Dict[str, dict],
    raw_counts: Dict[str, int] = None
) -> Dict[str, int]:
    """
    Deduplikuje liczniki fraz - odejmuje wystąpienia w dłuższych frazach.
    
    Args:
        text: Tekst do analizy
        keywords: Słownik {rid: {"keyword": "fraza", "type": "BASIC|EXTENDED", ...}}
        raw_counts: Opcjonalne surowe liczniki (jeśli już policzone)
    
    Returns:
        Dict {rid: deduplicated_count}
    """
    text_normalized = normalize_text(text)
    
    # Zbierz wszystkie frazy z ich rid
    phrases: List[Tuple[str, str, str]] = []  # (rid, keyword, type)
    for rid, meta in keywords.items():
        keyword = meta.get("keyword", "").strip()
        kw_type = meta.get("type", "BASIC").upper()
        if keyword:
            phrases.append((rid, keyword, kw_type))
    
    # Posortuj od najdłuższych (wg liczby słów)
    phrases.sort(key=lambda x: len(x[1].split()), reverse=True)
    
    # Policz surowe wystąpienia
    if raw_counts:
        counts = dict(raw_counts)
    else:
        counts = {}
        for rid, keyword, _ in phrases:
            counts[rid] = count_phrase_raw(text_normalized, keyword)
    
    # Deduplikacja - dla każdej frazy odejmij wystąpienia w dłuższych
    deduplicated = {}
    
    for i, (rid, keyword, _) in enumerate(phrases):
        raw_count = counts.get(rid, 0)
        
        # Znajdź wszystkie dłuższe frazy, które zawierają tę frazę
        overlap_count = 0
        for j in range(i):  # Tylko wcześniejsze (dłuższe) frazy
            longer_rid, longer_keyword, _ = phrases[j]
            if is_subphrase(keyword, longer_keyword):
                # Odejmij wystąpienia dłuższej frazy
                overlap_count += deduplicated.get(longer_rid, counts.get(longer_rid, 0))
        
        # Skorygowany licznik = surowy - overlap (min 0)
        deduplicated[rid] = max(0, raw_count - overlap_count)
    
    return deduplicated


def deduplicate_batch_counts(
    text: str,
    keywords_state: Dict[str, dict]
) -> Dict[str, int]:
    """
    Wrapper dla process_batch_in_firestore.
    
    Args:
        text: Tekst batcha
        keywords_state: Stan keywords z Firestore
    
    Returns:
        Dict {rid: deduplicated_count} gotowy do użycia w batch_counts
    """
    return deduplicate_keyword_counts(text, keywords_state)


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================
if __name__ == "__main__":
    # Test
    test_text = """
    Renta rodzinna przysługuje członkom rodziny zmarłego. 
    Renta wdowia to szczególny rodzaj renty rodzinnej.
    Sama renta może być przyznana w różnych okolicznościach.
    Prawo do renty mają osoby niezdolne do pracy.
    Renta z tytułu niezdolności do pracy wymaga orzeczenia.
    """
    
    test_keywords = {
        "kw1": {"keyword": "renta", "type": "MAIN"},
        "kw2": {"keyword": "renta rodzinna", "type": "BASIC"},
        "kw3": {"keyword": "renta wdowia", "type": "BASIC"},
        "kw4": {"keyword": "prawo do renty", "type": "EXTENDED"},
    }
    
    print("=== TEST DEDUPLIKACJI ===")
    print(f"Tekst: {test_text[:100]}...")
    print()
    
    # Surowe liczniki
    for rid, meta in test_keywords.items():
        raw = count_phrase_raw(test_text, meta["keyword"])
        print(f"  {meta['keyword']}: {raw} (surowe)")
    
    print()
    
    # Po deduplikacji
    deduped = deduplicate_keyword_counts(test_text, test_keywords)
    for rid, meta in test_keywords.items():
        print(f"  {meta['keyword']}: {deduped[rid]} (po deduplikacji)")
