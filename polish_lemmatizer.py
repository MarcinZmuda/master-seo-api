"""
===============================================================================
🇵🇱 POLISH LEMMATIZER v26.1 - Używa współdzielonego spaCy
===============================================================================
Wykorzystuje shared_nlp.py do lemmatyzacji polskich słów.
spaCy pl_core_news_md obsługuje:
- Lemmatyzację (rozwodu → rozwód)
- Rozpoznawanie form (sądem, sądzie → sąd)
- POS tagging (rzeczownik, czasownik, przymiotnik)

v26.1: Używa shared_nlp.py + rozszerzone wzorce polskich form
===============================================================================
"""

import re
from typing import Dict, List, Set

# Import współdzielonego spaCy
try:
    from shared_nlp import get_nlp
    _SPACY_OK = True
    print("[LEMMATIZER] ✅ Using shared spaCy from shared_nlp.py")
except ImportError:
    _SPACY_OK = False
    print("[LEMMATIZER] ⚠️ shared_nlp not available, using fallback")

# Cache dla wydajności
_lemma_cache = {}
_forms_cache = {}

BACKEND = "SPACY" if _SPACY_OK else "FALLBACK"


def init_backend():
    """Inicjalizuje backend (spaCy przez shared_nlp)."""
    global BACKEND
    if _SPACY_OK:
        try:
            nlp = get_nlp()
            BACKEND = "SPACY"
            print(f"[LEMMATIZER] ✅ Backend: SPACY ({nlp.meta.get('name', 'unknown')})")
            return True
        except Exception as e:
            print(f"[LEMMATIZER] ⚠️ spaCy error: {e}")
            BACKEND = "FALLBACK"
    return False


def get_backend_info() -> Dict:
    """Zwraca info o backendzie."""
    return {
        "backend": BACKEND,
        "spacy_available": _SPACY_OK
    }


def get_lemma(word: str) -> str:
    """Zwraca lemat słowa używając spaCy."""
    word_lower = word.lower().strip()
    
    if word_lower in _lemma_cache:
        return _lemma_cache[word_lower]
    
    lemma = word_lower
    
    if _SPACY_OK:
        try:
            nlp = get_nlp()
            doc = nlp(word_lower)
            if doc and len(doc) > 0:
                lemma = doc[0].lemma_.lower()
        except:
            pass
    
    _lemma_cache[word_lower] = lemma
    return lemma


def get_all_forms(word: str) -> Set[str]:
    """Zwraca wszystkie rozpoznawane formy słowa."""
    word_lower = word.lower().strip()
    
    if word_lower in _forms_cache:
        return _forms_cache[word_lower]
    
    forms = {word_lower}
    
    # Dodaj lemat
    lemma = get_lemma(word_lower)
    forms.add(lemma)
    
    # Generuj typowe polskie formy
    forms.update(_generate_forms_from_lemma(lemma))
    if lemma != word_lower:
        forms.update(_generate_forms_from_lemma(word_lower))
    
    _forms_cache[word_lower] = forms
    return forms


def _generate_forms_from_lemma(word: str) -> Set[str]:
    """Generuje typowe polskie formy słowa."""
    forms = {word}
    
    if not word or len(word) < 2:
        return forms
    
    # -enie, -anie (uzależnienie)
    if word.endswith('enie') or word.endswith('anie'):
        base = word[:-1]
        forms.update([word, base + 'a', base + 'u', base + 'em', base + 'ami', base + 'ach', base + 'om'])
        return forms
    
    # -ość (wolność)
    if word.endswith('ość'):
        base = word[:-1]
        forms.update([word, base + 'i', base + 'ią', base + 'iom', base + 'iami', base + 'iach'])
        return forms
    
    # -acja (sytuacja)
    if word.endswith('acja'):
        base = word[:-1]
        forms.update([word, base + 'i', base + 'ę', base + 'ą', base + 'e', base + 'om', base + 'ami', base + 'ach'])
        return forms
    
    # -ąd (sąd)
    if word.endswith('ąd'):
        base = word[:-2]
        forms.update([word, base + 'ądu', base + 'ądowi', base + 'ądem', base + 'ądzie',
                     base + 'ądy', base + 'ądów', base + 'ądom', base + 'ądami', base + 'ądach'])
        return forms
    
    # -ód (rozwód) - alternacja ó/o
    if word.endswith('ód'):
        base = word[:-2]
        forms.update([word, base + 'odu', base + 'odowi', base + 'odem', base + 'odzie',
                     base + 'ody', base + 'odów', base + 'odom', base + 'odami', base + 'odach'])
        return forms
    
    # -óg (nałóg) - alternacja ó/o
    if word.endswith('óg'):
        base = word[:-2]
        forms.update([word, base + 'ogu', base + 'ogowi', base + 'ogiem',
                     base + 'ogi', base + 'ogów', base + 'ogom', base + 'ogami', base + 'ogach'])
        return forms
    
    # -yk (narkotyk)
    if word.endswith('yk'):
        base = word[:-2]
        forms.update([word, base + 'yku', base + 'ykowi', base + 'ykiem',
                     base + 'yki', base + 'yków', base + 'ykom', base + 'ykami', base + 'ykach'])
        return forms
    
    # -nik (prawnik)
    if word.endswith('nik'):
        base = word[:-3]
        forms.update([word, base + 'nika', base + 'nikowi', base + 'nikiem', base + 'niku',
                     base + 'nicy', base + 'ników', base + 'nikom', base + 'nikami', base + 'nikach'])
        return forms
    
    # -ek (małżonek)
    if word.endswith('ek') and len(word) > 3:
        base = word[:-2]
        forms.update([word, base + 'ka', base + 'kowi', base + 'kiem', base + 'ku',
                     base + 'kowie', base + 'ków', base + 'kom', base + 'kami', base + 'kach'])
        return forms
    
    # -ca (radca)
    if word.endswith('ca'):
        base = word[:-2]
        forms.update([word, base + 'cy', base + 'cę', base + 'cą',
                     base + 'ców', base + 'com', base + 'cami', base + 'cach'])
        return forms
    
    # -at (adwokat)
    if word.endswith('at') and len(word) > 3:
        base = word[:-2]
        forms.update([word, base + 'ata', base + 'atowi', base + 'atem', base + 'acie',
                     base + 'aci', base + 'atów', base + 'atom', base + 'atami', base + 'atach'])
        return forms
    
    # === PRZYMIOTNIKI ===
    
    # -ny (prawny)
    if word.endswith('ny'):
        base = word[:-2]
        forms.update([word, base + 'na', base + 'ne', base + 'nego', base + 'nej',
                     base + 'nemu', base + 'nym', base + 'ni', base + 'nych', base + 'nymi'])
        return forms
    
    # -ski, -cki (małżeński)
    if word.endswith('ski') or word.endswith('cki'):
        base = word[:-2]
        forms.update([word, base + 'ka', base + 'kie', base + 'kiego', base + 'kiej',
                     base + 'kiemu', base + 'kim', base + 'cy', base + 'kich', base + 'kimi'])
        return forms
    
    # -owy (rozwodowy)
    if word.endswith('owy'):
        base = word[:-3]
        forms.update([word, base + 'owa', base + 'owe', base + 'owego', base + 'owej',
                     base + 'owemu', base + 'owym', base + 'owi', base + 'owych', base + 'owymi'])
        return forms
    
    # === DOMYŚLNE ===
    if len(word) >= 3:
        forms.update([word, word + 'a', word + 'u', word + 'owi', word + 'em', word + 'ie',
                     word + 'y', word + 'ów', word + 'om', word + 'ami', word + 'ach'])
    
    return forms


def lemmatize_text(text: str) -> List[str]:
    """Zwraca listę lematów z tekstu."""
    if not text:
        return []
    
    if _SPACY_OK:
        try:
            nlp = get_nlp()
            doc = nlp(text.lower())
            return [token.lemma_.lower() for token in doc if token.is_alpha]
        except:
            pass
    
    words = re.findall(r'\b\w+\b', text.lower())
    return [get_lemma(w) for w in words]


def get_phrase_lemmas(phrase: str) -> List[str]:
    """Zwraca lematy dla frazy."""
    words = phrase.lower().split()
    return [get_lemma(w) for w in words]


def count_phrase_occurrences(text: str, phrase: str) -> Dict:
    """Liczy wystąpienia frazy w tekście z uwzględnieniem form."""
    text_lower = text.lower()
    phrase_lower = phrase.lower().strip()
    
    if not text_lower or not phrase_lower:
        return {"count": 0, "method": "empty", "matches": []}
    
    words = phrase_lower.split()
    
    if len(words) == 1:
        return _count_single_word(text_lower, phrase_lower)
    else:
        return _count_multi_word(text_lower, words)


def _count_single_word(text: str, word: str) -> Dict:
    """Liczy pojedyncze słowo z formami."""
    forms = get_all_forms(word)
    
    count = 0
    matches = []
    
    for form in forms:
        pattern = r'\b' + re.escape(form) + r'\b'
        found = re.findall(pattern, text, re.IGNORECASE)
        count += len(found)
        if found:
            matches.extend(found)
    
    return {
        "count": count,
        "method": BACKEND,
        "forms_checked": list(forms)[:15],
        "matches": matches[:10]
    }


def _count_multi_word(text: str, words: List[str]) -> Dict:
    """Liczy frazę wielowyrazową z formami."""
    forms_per_word = [get_all_forms(w) for w in words]
    
    text_words = re.findall(r'\b\w+\b', text.lower())
    
    count = 0
    matches = []
    
    for i in range(len(text_words) - len(words) + 1):
        match = True
        matched_phrase = []
        
        for j, word_forms in enumerate(forms_per_word):
            if text_words[i + j] not in word_forms:
                match = False
                break
            matched_phrase.append(text_words[i + j])
        
        if match:
            count += 1
            matches.append(" ".join(matched_phrase))
    
    return {
        "count": count,
        "method": BACKEND,
        "phrase_words": words,
        "matches": matches[:10]
    }


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    init_backend()
    print(f"\nBackend: {BACKEND}")
    
    print("\n=== TEST FORM ===")
    for word in ['sąd', 'rozwód', 'uzależnienie', 'prawny', 'małżeński', 'narkotyk']:
        forms = get_all_forms(word)
        print(f"  '{word}' → {len(forms)} form: {sorted(list(forms))[:8]}...")
    
    print("\n=== TEST LICZENIA ===")
    text = "Sąd orzekł rozwód. W sądzie odbyła się rozprawa rozwodowa. Sądy często orzekają."
    for phrase in ['sąd', 'rozwód', 'rozwodowy']:
        result = count_phrase_occurrences(text, phrase)
        print(f"  '{phrase}' w tekście: {result['count']}x")
