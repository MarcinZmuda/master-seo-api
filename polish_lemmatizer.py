import re
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache

_morfeusz = None
_spacy_nlp = None
_lemma_cache = {}
_forms_cache = {}

BACKEND = None


def _init_morfeusz():
    global _morfeusz, BACKEND
    if _morfeusz is not None:
        return True
    try:
        import morfeusz2
        _morfeusz = morfeusz2.Morfeusz()
        BACKEND = "MORFEUSZ2"
        print("[LEMMATIZER] Morfeusz2 loaded")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[LEMMATIZER] Morfeusz2 error: {e}")
        return False


def _init_spacy():
    global _spacy_nlp, BACKEND
    if _spacy_nlp is not None:
        return True
    try:
        import spacy
        _spacy_nlp = spacy.load("pl_core_news_md")
        if BACKEND is None:
            BACKEND = "SPACY"
        print("[LEMMATIZER] spaCy pl_core_news_md loaded")
        return True
    except:
        try:
            import spacy
            from spacy.cli import download
            download("pl_core_news_md")
            _spacy_nlp = spacy.load("pl_core_news_md")
            if BACKEND is None:
                BACKEND = "SPACY"
            return True
        except Exception as e:
            print(f"[LEMMATIZER] spaCy error: {e}")
            return False


def init_backend():
    if _init_morfeusz():
        _init_spacy()
        return "MORFEUSZ2"
    elif _init_spacy():
        return "SPACY"
    else:
        return "PREFIX"


def get_lemma(word: str) -> str:
    word_lower = word.lower().strip()
    
    if word_lower in _lemma_cache:
        return _lemma_cache[word_lower]
    
    lemma = word_lower
    
    if _morfeusz is not None:
        try:
            analysis = _morfeusz.analyse(word_lower)
            if analysis:
                lemma = analysis[0][2][1].lower()
        except:
            pass
    elif _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(word_lower)
            if doc:
                lemma = doc[0].lemma_.lower()
        except:
            pass
    
    _lemma_cache[word_lower] = lemma
    return lemma


def get_all_forms(word: str) -> Set[str]:
    word_lower = word.lower().strip()
    
    if word_lower in _forms_cache:
        return _forms_cache[word_lower]
    
    forms = {word_lower}
    
    if _morfeusz is not None:
        try:
            lemma = get_lemma(word_lower)
            forms.add(lemma)
            
            analysis = _morfeusz.analyse(lemma)
            if analysis:
                base_lemma = analysis[0][2][1].lower()
                try:
                    generated = _morfeusz.generate(base_lemma)
                    for item in generated:
                        form = item[0].lower()
                        forms.add(form)
                except:
                    pass
        except:
            pass
    
    if len(forms) <= 2:
        forms.update(_generate_polish_forms(word_lower))
    
    _forms_cache[word_lower] = forms
    return forms


def _generate_polish_forms(word: str) -> Set[str]:
    forms = {word}
    
    # Wykryj końcówki i generuj odmiany
    for ending in ['acja', 'ość', 'enie', 'anie', 'cie']:
        if word.endswith(ending):
            base = word[:-len(ending)]
            if ending == 'acja':
                forms.update([base + 'acja', base + 'acji', base + 'ację', base + 'acją', 
                             base + 'acje', base + 'acjom', base + 'acjami', base + 'acjach',
                             base + 'acyjny', base + 'acyjna', base + 'acyjne', base + 'acyjnego'])
            elif ending == 'ość':
                forms.update([base + 'ość', base + 'ości', base + 'ością'])
            elif ending in ['enie', 'anie']:
                forms.update([base + ending, base + ending[:-1] + 'a', base + ending[:-1] + 'u',
                             base + ending[:-1] + 'em', base + ending[:-1] + 'ami'])
            return forms
    
    # Deklinacja rzeczowników
    if word.endswith('a') and not word.endswith('acja'):
        base = word[:-1]
        forms.update([base + 'a', base + 'y', base + 'ie', base + 'ę', base + 'o', 
                     base + 'ą', base + 'om', base + 'ami', base + 'ach', base + 'i'])
    elif word.endswith('o'):
        base = word[:-1]
        forms.update([base + 'o', base + 'a', base + 'u', base + 'em', base + 'ie',
                     base + 'om', base + 'ami', base + 'ach'])
    elif word.endswith('ek'):
        base = word[:-2]
        forms.update([base + 'ek', base + 'ka', base + 'ku', base + 'kiem', base + 'ków',
                     base + 'ki', base + 'kom', base + 'kami', base + 'kach'])
    elif word.endswith('ód'):
        base = word[:-2]
        forms.update([base + 'ód', base + 'odu', base + 'odowi', base + 'odem', base + 'odzie',
                     base + 'ody', base + 'odów', base + 'odom', base + 'odami', base + 'odach'])
    else:
        # Domyślna deklinacja męska
        forms.update([word + 'a', word + 'u', word + 'owi', word + 'em', word + 'ie',
                     word + 'y', word + 'ów', word + 'om', word + 'ami', word + 'ach'])
        # Jeśli kończy się spółgłoską, dodaj też -i
        if word[-1] not in 'aeiouyąęó':
            forms.update([word + 'i'])
    
    return forms


def get_phrase_lemmas(phrase: str) -> List[str]:
    words = phrase.lower().split()
    return [get_lemma(w) for w in words]


def get_phrase_all_forms(phrase: str) -> List[Set[str]]:
    words = phrase.lower().split()
    return [get_all_forms(w) for w in words]


def count_phrase_occurrences(text: str, phrase: str) -> Dict:
    text_lower = text.lower()
    phrase_lower = phrase.lower().strip()
    
    if not text_lower or not phrase_lower:
        return {"count": 0, "method": "empty", "matches": []}
    
    init_backend()
    
    words = phrase_lower.split()
    
    if len(words) == 1:
        return _count_single_word(text_lower, phrase_lower)
    else:
        return _count_multi_word(text_lower, words)


def _count_single_word(text: str, word: str) -> Dict:
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
        "method": BACKEND or "PREFIX",
        "forms_checked": list(forms)[:10],
        "matches": matches[:10]
    }


def _count_multi_word(text: str, words: List[str]) -> Dict:
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
        "method": BACKEND or "PREFIX",
        "phrase_words": words,
        "matches": matches[:10]
    }


def count_all_keywords(text: str, keywords: List[str]) -> Dict[str, Dict]:
    results = {}
    for kw in keywords:
        if kw and kw.strip():
            results[kw] = count_phrase_occurrences(text, kw)
    return results


def lemmatize_text(text: str) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    return [get_lemma(w) for w in words]


def get_backend_info() -> Dict:
    init_backend()
    return {
        "backend": BACKEND or "PREFIX",
        "morfeusz_available": _morfeusz is not None,
        "spacy_available": _spacy_nlp is not None,
        "cache_size": {
            "lemmas": len(_lemma_cache),
            "forms": len(_forms_cache)
        }
    }


def clear_cache():
    global _lemma_cache, _forms_cache
    _lemma_cache = {}
    _forms_cache = {}


if __name__ == "__main__":
    print("Testing polish_lemmatizer...")
    
    backend = init_backend()
    print(f"Backend: {backend}")
    
    test_words = ["zmywarka", "jurysdykcja", "rozwód", "prawnik"]
    for word in test_words:
        lemma = get_lemma(word)
        forms = get_all_forms(word)
        print(f"\n{word}:")
        print(f"  Lemma: {lemma}")
        print(f"  Forms: {sorted(forms)[:8]}...")
    
    test_text = """
    Zmywarka to urządzenie do mycia naczyń. Zmywarki są popularne w polskich domach.
    Wybór zmywarki zależy od wielu czynników. Nowoczesne zmywarkom oszczędzają wodę.
    """
    
    print("\n--- Count test ---")
    result = count_phrase_occurrences(test_text, "zmywarka")
    print(f"'zmywarka' in text: {result}")
