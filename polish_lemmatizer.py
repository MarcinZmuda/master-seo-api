"""
===============================================================================
🇵🇱 POLISH LEMMATIZER v29.1 - Używa współdzielonego spaCy
===============================================================================
v29.1: 
- Normalizacja myślników i symboli w frazach
- Bidirectional matching (terapii ↔ terapia)
- Obsługa fraz typu "integracja sensoryczna – pomoce"

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


# ============================================================================
# v29.1: NORMALIZACJA FRAZ Z MYŚLNIKAMI I SYMBOLAMI
# ============================================================================
def normalize_phrase(phrase: str) -> str:
    """
    v29.1: Normalizuje frazę do porównania.
    
    - Zamienia wszystkie typy myślników na spację
    - Usuwa wielokrotne spacje
    - Zamienia em dash (–), en dash (–), hyphen (-) na spację
    
    "integracja sensoryczna – pomoce" → "integracja sensoryczna pomoce"
    """
    if not phrase:
        return ""
    
    # Zamień różne typy myślników na spację
    normalized = phrase
    normalized = normalized.replace('–', ' ')  # em dash
    normalized = normalized.replace('—', ' ')  # em dash (longer)
    normalized = normalized.replace('-', ' ')   # hyphen
    normalized = normalized.replace('−', ' ')  # minus sign
    
    # Usuń wielokrotne spacje
    normalized = ' '.join(normalized.split())
    
    return normalized.lower().strip()


def normalize_text_for_matching(text: str) -> str:
    """
    v29.1: Normalizuje tekst do wyszukiwania fraz.
    """
    if not text:
        return ""
    
    normalized = text.lower()
    # Zamień myślniki na spacje
    normalized = normalized.replace('–', ' ')
    normalized = normalized.replace('—', ' ')
    normalized = normalized.replace('-', ' ')
    normalized = normalized.replace('−', ' ')
    # Usuń wielokrotne spacje
    normalized = ' '.join(normalized.split())
    
    return normalized


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
    
    # === RZECZOWNIKI ŻEŃSKIE ===
    
    # -ia (terapia, integracja) - WAŻNE!
    if word.endswith('ia') and len(word) > 3:
        base = word[:-2]
        forms.update([word, base + 'ii', base + 'ię', base + 'ią', base + 'io',
                     base + 'ie', base + 'ij', base + 'iom', base + 'iami', base + 'iach'])
        return forms
    
    # -ka (ścieżka, podróżka) - BARDZO WAŻNE!
    if word.endswith('ka') and len(word) > 3:
        base = word[:-2]
        forms.update([word, base + 'ki', base + 'kę', base + 'ką', base + 'ce',
                     base + 'ek', base + 'kom', base + 'kami', base + 'kach'])
        return forms
    
    # -a (droga, mama) - rzeczowniki żeńskie
    if word.endswith('a') and len(word) > 2 and not word.endswith('ca'):
        base = word[:-1]
        forms.update([word, base + 'y', base + 'ę', base + 'ą', base + 'ie', base + 'o',
                     base, base + 'om', base + 'ami', base + 'ach'])
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
    
    # -yczny (sensoryczny, techniczny) - BARDZO WAŻNE!
    if word.endswith('yczny'):
        base = word[:-5]
        forms.update([
            word,  # sensoryczny
            base + 'yczna', base + 'yczne',  # sensoryczna, sensoryczne
            base + 'ycznego', base + 'ycznej',  # sensorycznego, sensorycznej
            base + 'ycznemu',  # sensorycznemu
            base + 'ycznym', base + 'yczną',  # sensorycznym, sensoryczną
            base + 'yczni', base + 'ycznych', base + 'ycznymi'  # sensoryczni, sensorycznych
        ])
        return forms
    
    # -owy (rozwodowy, kolorowy)
    if word.endswith('owy'):
        base = word[:-3]
        forms.update([word, base + 'owa', base + 'owe', base + 'owego', base + 'owej',
                     base + 'owemu', base + 'owym', base + 'ową', base + 'owi', base + 'owych', base + 'owymi'])
        return forms
    
    # -ny (prawny, ciemny)
    if word.endswith('ny'):
        base = word[:-2]
        forms.update([word, base + 'na', base + 'ne', base + 'nego', base + 'nej',
                     base + 'nemu', base + 'nym', base + 'ną', base + 'ni', base + 'nych', base + 'nymi'])
        return forms
    
    # -ski, -cki (małżeński, miejski)
    if word.endswith('ski') or word.endswith('cki'):
        base = word[:-2]
        forms.update([word, base + 'ka', base + 'kie', base + 'kiego', base + 'kiej',
                     base + 'kiemu', base + 'kim', base + 'ką', base + 'cy', base + 'kich', base + 'kimi'])
        return forms
    
    # -ły (mały, biały)
    if word.endswith('ły'):
        base = word[:-2]
        forms.update([word, base + 'ła', base + 'łe', base + 'łego', base + 'łej',
                     base + 'łemu', base + 'łym', base + 'łą', base + 'li', base + 'łych', base + 'łymi'])
        return forms
    
    # === DOMYŚLNE ===
    if len(word) >= 3:
        forms.update([word, word + 'a', word + 'u', word + 'owi', word + 'em', word + 'ie',
                     word + 'y', word + 'ów', word + 'om', word + 'ami', word + 'ach',
                     word + 'ą', word + 'ę'])  # dodane formy żeńskie
    
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
    """
    v29.1: PRAWIDŁOWA LEMMATYZACJA + NORMALIZACJA MYŚLNIKÓW
    
    NAJPIERW normalizuje frazę i tekst (usuwa myślniki),
    POTEM liczy z lemmatyzacją.
    
    "integracja sensoryczna – pomoce" → szuka "integracja sensoryczna pomoce"
    """
    if not text or not phrase:
        return {"count": 0, "method": "empty", "matches": []}
    
    # v29.1: NORMALIZACJA - zamień myślniki na spacje
    phrase_normalized = normalize_phrase(phrase)
    text_normalized = normalize_text_for_matching(text)
    
    if not phrase_normalized:
        return {"count": 0, "method": "empty_after_normalize", "matches": []}
    
    # Sprawdź czy spaCy działa (czy lematy są różne od oryginału)
    test_word = "sensoryczną"
    test_lemma = get_lemma(test_word)
    spacy_works = (test_lemma != test_word)  # Jeśli zlemmatyzował, to działa
    
    if spacy_works:
        # METODA 1: Porównanie lematów (spaCy działa)
        phrase_lemmas = get_phrase_lemmas(phrase_normalized)
        text_lemmas = lemmatize_text(text_normalized)
        
        if not phrase_lemmas or not text_lemmas:
            return {"count": 0, "method": "spacy_empty", "matches": []}
        
        count = 0
        matches = []
        phrase_len = len(phrase_lemmas)
        
        for i in range(len(text_lemmas) - phrase_len + 1):
            if text_lemmas[i:i + phrase_len] == phrase_lemmas:
                count += 1
                matches.append(f"pos:{i}")
        
        return {
            "count": count,
            "method": "SPACY_LEMMA",
            "phrase_normalized": phrase_normalized,
            "phrase_lemmas": phrase_lemmas,
            "matches": matches[:10]
        }
    else:
        # METODA 2: Generowanie form (fallback)
        return _count_multi_word_with_forms(text_normalized, phrase_normalized)


def _count_multi_word_with_forms(text: str, phrase: str) -> Dict:
    """
    Fallback: dla każdego słowa frazy i tekstu generuj formy,
    sprawdź czy się przecinają (match w dowolną stronę).
    
    "terapii" ma formy: [terapii, terapia, terapią...]
    "terapia" ma formy: [terapia, terapii, terapią...]
    → Przecięcie niepuste = MATCH!
    """
    words = phrase.split()
    text_words = re.findall(r'\b\w+\b', text.lower())
    
    # Dla każdego słowa frazy, pobierz WSZYSTKIE możliwe formy (włącznie z formą bazową)
    forms_per_phrase_word = []
    for w in words:
        # Pobierz formy od tego słowa
        forms_from_word = get_all_forms(w)
        # Znajdź też formę bazową (lemat) i pobierz jej formy
        # Heurystyka: jeśli słowo kończy się na -ii, -ą, -ę, spróbuj znaleźć bazę
        base = _guess_lemma(w)
        if base != w:
            forms_from_word.update(get_all_forms(base))
        forms_per_phrase_word.append(forms_from_word)
    
    count = 0
    matches = []
    
    for i in range(len(text_words) - len(words) + 1):
        match = True
        matched_phrase = []
        
        for j, phrase_forms in enumerate(forms_per_phrase_word):
            text_word = text_words[i + j]
            # Sprawdź czy słowo tekstu jest w formach słowa frazy
            # LUB czy formy słowa tekstu przecinają się z formami słowa frazy
            text_word_forms = get_all_forms(text_word)
            
            if text_word not in phrase_forms and not phrase_forms.intersection(text_word_forms):
                match = False
                break
            matched_phrase.append(text_word)
        
        if match:
            count += 1
            matches.append(" ".join(matched_phrase))
    
    return {
        "count": count,
        "method": "FORMS_BIDIRECTIONAL",
        "phrase_words": words,
        "matches": matches[:10]
    }


def _guess_lemma(word: str) -> str:
    """
    Heurystyka: zgadnij formę podstawową (lemat) bez spaCy.
    Używane gdy spaCy niedostępny.
    """
    word = word.lower()
    
    # Rzeczowniki żeńskie w dopełniaczu/celowniku
    if word.endswith('ii'):
        return word[:-1] + 'a'  # terapii → terapia
    if word.endswith('ji'):
        return word[:-2] + 'ja'  # integracji → integracja
    if word.endswith('cji'):
        return word[:-1] + 'a'  # integracji → integracja
    
    # Przymiotniki w różnych przypadkach
    if word.endswith('ycznej'):
        return word[:-2] + 'y'  # sensorycznej → sensoryczny
    if word.endswith('yczną'):
        return word[:-1] + 'y'  # sensoryczną → sensoryczny  
    if word.endswith('ycznego'):
        return word[:-3] + 'y'  # sensorycznego → sensoryczny
    if word.endswith('ycznym'):
        return word[:-2] + 'y'  # sensorycznym → sensoryczny
    
    # Rzeczowniki żeńskie
    if word.endswith('ką'):
        return word[:-1] + 'a'  # ścieżką → ścieżka
    if word.endswith('kę'):
        return word[:-1] + 'a'  # ścieżkę → ścieżka
    if word.endswith('ce') and len(word) > 3:
        return word[:-1] + 'a'  # ścieżce → ścieżka (przybliżenie)
    if word.endswith('ki') and len(word) > 3:
        return word[:-1] + 'a'  # ścieżki → ścieżka
    
    # Rzeczowniki męskie w dopełniaczu
    if word.endswith('ów'):
        return word[:-2]  # terapeutów → terapeut
    if word.endswith('ach'):
        return word[:-3]  # przedszkolach → przedszkol? (nie idealne)
    
    return word


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
