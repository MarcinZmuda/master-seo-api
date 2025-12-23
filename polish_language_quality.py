"""
===============================================================================
🇵🇱 POLISH LANGUAGE QUALITY v23.0 - Kontrola Jakości Języka Polskiego
===============================================================================
Moduł sprawdzający:
1. Kolokacje polskie (naturalne połączenia wyrazów)
2. Powtórzenia leksykalne
3. Spójność rejestru stylistycznego
4. Szyk zdania (monotonność)
5. Typowe błędy AI w polskim

Autorzy: Opracowano na podstawie:
- Wielki Słownik Języka Polskiego
- Nowy Słownik Poprawnej Polszczyzny PWN
- Praktyczny Słownik Współczesnej Polszczyzny
===============================================================================
"""

import re
from typing import Dict, List, Any, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import spacy

# ================================================================
# 🧠 Współdzielony model spaCy
# ================================================================
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
except ImportError:
    # Fallback - ładuj lokalnie
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")


# ================================================================
# 📚 ROZSZERZONA LISTA TRANSITION WORDS (z kategoryzacją funkcjonalną)
# ================================================================
TRANSITION_WORDS_CATEGORIZED = {
    "dodawanie": [
        "również", "także", "ponadto", "dodatkowo", "co więcej",
        "oprócz tego", "poza tym", "a także", "jak również", "przy czym",
        "jednocześnie", "zarazem", "w dodatku", "na dodatek", "i"
    ],
    "kontrast": [
        "jednak", "jednakże", "natomiast", "ale", "lecz", "aczkolwiek",
        "z drugiej strony", "mimo to", "niemniej", "tymczasem",
        "przeciwnie", "w przeciwieństwie do", "choć", "chociaż", "wprawdzie"
    ],
    "przyczyna": [
        "ponieważ", "bowiem", "albowiem", "gdyż", "jako że",
        "z tego powodu", "z uwagi na", "ze względu na", "dlatego że",
        "skoro", "w związku z"
    ],
    "skutek": [
        "dlatego", "zatem", "więc", "toteż", "stąd", "wobec tego",
        "w efekcie", "w rezultacie", "w konsekwencji", "skutkiem tego",
        "tym samym", "przeto", "w związku z tym"
    ],
    "czas_sekwencja": [
        "najpierw", "następnie", "potem", "później", "wcześniej",
        "uprzednio", "wówczas", "dotychczas", "tymczasem", "na początku",
        "na koniec", "w końcu", "po pierwsze", "po drugie", "po trzecie",
        "finalnie", "ostatecznie", "wreszcie"
    ],
    "przyklady": [
        "na przykład", "przykładowo", "między innymi", "m.in.", "np.",
        "chociażby", "choćby", "jak choćby", "dla przykładu",
        "weźmy pod uwagę", "rozważmy", "wyobraźmy sobie"
    ],
    "podsumowanie": [
        "podsumowując", "reasumując", "w skrócie", "krótko mówiąc",
        "ogólnie rzecz biorąc", "jednym słowem", "w konkluzji",
        "konkludując", "zatem", "tak więc", "słowem"
    ],
    "emfaza": [
        "przede wszystkim", "szczególnie", "zwłaszcza", "w szczególności",
        "głównie", "nade wszystko", "co najważniejsze", "kluczowe jest",
        "istotne jest", "warto podkreślić", "należy zauważyć"
    ],
    "warunek": [
        "jeśli", "jeżeli", "o ile", "pod warunkiem że", "w przypadku gdy",
        "gdyby", "w razie", "chyba że", "byleby", "byle"
    ],
    "porownanie": [
        "podobnie", "analogicznie", "tak samo", "w podobny sposób",
        "na podobnej zasadzie", "porównywalnie", "identycznie",
        "w przeciwieństwie", "inaczej niż", "odmiennie"
    ]
}

# Płaska lista dla kompatybilności wstecznej
ALL_TRANSITION_WORDS = []
for category, words in TRANSITION_WORDS_CATEGORIZED.items():
    ALL_TRANSITION_WORDS.extend(words)
ALL_TRANSITION_WORDS = list(set(ALL_TRANSITION_WORDS))


# ================================================================
# 🚫 ROZSZERZONA LISTA BANNED PHRASES
# ================================================================
BANNED_PHRASES_EXTENDED = {
    "puste_intensyfikatory": [
        "niezwykle istotny", "niezmiernie ważny", "absolutnie kluczowy",
        "fundamentalnie istotny", "szczególnie znaczący", "wyjątkowo ważny",
        "nadzwyczaj istotny", "szalenie ważny"
    ],
    "pseudo_empatia_ai": [
        "doskonale rozumiemy", "zdajemy sobie sprawę",
        "mamy świadomość", "jesteśmy przekonani",
        "rozumiemy twoje obawy", "wiemy jak się czujesz",
        "doceniamy twoje zainteresowanie"
    ],
    "nadmierna_formalnosc": [
        "niniejszy artykuł", "przedmiotowe zagadnienie",
        "powyższe rozważania", "poniższe informacje",
        "niniejszym informujemy", "uprzejmie informujemy",
        "mając na uwadze powyższe"
    ],
    "sztuczne_przejscia": [
        "przechodząc do kolejnego aspektu", "warto w tym miejscu zauważyć",
        "nie sposób nie wspomnieć", "godnym uwagi jest fakt",
        "w tym kontekście warto", "analizując dalej"
    ],
    "redundancja": [
        "bardzo ważne i istotne", "nowe i nowatorskie",
        "różne i rozmaite", "pełny i kompletny",
        "szybki i sprawny", "jasny i czytelny"
    ],
    "pleonazmy": [
        "cofnąć się wstecz", "kontynuować dalej",
        "powrócić z powrotem", "wzajemna współpraca",
        "spadek w dół", "wzrost w górę",
        "przyszła przyszłość", "wspólnie razem"
    ],
    "typowe_ai_openers": [
        "w dzisiejszych czasach", "w obecnych czasach",
        "w dobie", "w erze", "żyjemy w czasach",
        "warto wiedzieć", "warto pamiętać",
        "jak wiadomo", "powszechnie wiadomo",
        "każdy z nas", "wszyscy wiemy",
        "nie ulega wątpliwości", "nie da się ukryć",
        "coraz więcej osób", "coraz częściej",
        "z całą pewnością", "bez wątpienia"
    ],
    "section_openers": [
        "dlatego", "ponadto", "dodatkowo", "tym samym",
        "warto", "należy", "trzeba", "wystarczy"
    ]
}


# ================================================================
# 🔗 KOLOKACJE POLSKIE (najczęstsze błędne połączenia)
# ================================================================
INCORRECT_COLLOCATIONS = {
    # "błędna fraza": "poprawna fraza"
    "robić decyzję": "podejmować decyzję",
    "dawać uwagę": "zwracać uwagę",
    "brać pod rozważenie": "brać pod uwagę",
    "mieć opinię": "wyrażać opinię",
    "grać rolę": "odgrywać rolę",
    "silne przekonanie": "głębokie przekonanie",
    "wysoki stopień": "wysoki poziom",
    "robić wpływ": "wywierać wpływ",
    "dawać przykład": "stanowić przykład",
    "mieć miejsce": "odbywać się",  # kontekstowe
    "robić błąd": "popełniać błąd",
    "stawiać pytanie": "zadawać pytanie",
    "dawać odpowiedź": "udzielać odpowiedzi",
    "robić postęp": "czynić postępy",
    "brać odpowiedzialność": "ponosić odpowiedzialność",
    "silny argument": "mocny argument",
    "duży sukces": "wielki sukces",
    "robić wysiłek": "podejmować wysiłek",
    "wielka ilość": "duża ilość",
    "mała ilość": "niewielka ilość",
    "robić wrażenie": "sprawiać wrażenie",
}

# Poprawne kolokacje (do promocji)
PREFERRED_COLLOCATIONS = [
    "podejmować decyzję", "zwracać uwagę", "brać pod uwagę",
    "odgrywać rolę", "wywierać wpływ", "popełniać błąd",
    "zadawać pytanie", "udzielać odpowiedzi", "czynić postępy",
    "ponosić odpowiedzialność", "sprawiać wrażenie"
]


# ================================================================
# 📊 STRUKTURY DANYCH
# ================================================================
@dataclass
class LanguageQualityResult:
    """Wynik analizy jakości językowej."""
    score: float  # 0-100
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "issues_count": len(self.issues),
            "issues": self.issues[:10],  # Limit
            "warnings": self.warnings[:5],
            "recommendations": self.recommendations[:5],
            "metrics": self.metrics,
            "status": "GOOD" if self.score >= 70 else ("FAIR" if self.score >= 50 else "POOR")
        }


# ================================================================
# 🔍 FUNKCJE ANALIZY
# ================================================================

def check_collocations(text: str) -> Tuple[List[Dict], float]:
    """
    Sprawdza błędne kolokacje w tekście.
    
    Returns:
        Tuple[lista błędów, score 0-1]
    """
    text_lower = text.lower()
    errors = []
    
    for incorrect, correct in INCORRECT_COLLOCATIONS.items():
        if incorrect in text_lower:
            # Znajdź kontekst
            idx = text_lower.find(incorrect)
            context = text[max(0, idx-20):min(len(text), idx+len(incorrect)+20)]
            
            errors.append({
                "type": "COLLOCATION_ERROR",
                "found": incorrect,
                "suggested": correct,
                "context": f"...{context}..."
            })
    
    # Score: 1.0 jeśli brak błędów, maleje z każdym błędem
    score = max(0, 1 - len(errors) * 0.15)
    
    return errors, score


def check_lexical_repetitions(text: str, window_size: int = 3) -> Tuple[List[Dict], float]:
    """
    Sprawdza nadmierne powtórzenia leksykalne.
    
    Args:
        text: Tekst do analizy
        window_size: Ile zdań wstecz sprawdzać
    
    Returns:
        Tuple[lista powtórzeń, score 0-1]
    """
    doc = nlp(text[:20000])
    
    # Zbierz rzeczowniki i czasowniki (content words)
    sentences = list(doc.sents)
    repetitions = []
    
    for i, sent in enumerate(sentences):
        if i < window_size:
            continue
        
        # Słowa w bieżącym zdaniu
        current_words = set(
            token.lemma_.lower() for token in sent 
            if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 3
        )
        
        # Słowa w poprzednich zdaniach
        prev_words = Counter()
        for j in range(max(0, i - window_size), i):
            for token in sentences[j]:
                if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 3:
                    prev_words[token.lemma_.lower()] += 1
        
        # Znajdź powtórzenia
        for word in current_words:
            if prev_words[word] >= 2:  # Słowo było 2+ razy w ostatnich zdaniach
                repetitions.append({
                    "type": "LEXICAL_REPETITION",
                    "word": word,
                    "count_in_window": prev_words[word] + 1,
                    "sentence_index": i
                })
    
    # Score
    score = max(0, 1 - len(repetitions) * 0.1)
    
    return repetitions, score


def check_register_consistency(text: str) -> Tuple[List[Dict], float, str]:
    """
    Sprawdza spójność rejestru stylistycznego.
    
    Returns:
        Tuple[lista problemów, score 0-1, wykryty rejestr]
    """
    # Markery rejestrów
    FORMAL_MARKERS = [
        "niniejszy", "przedmiotowy", "powyższy", "uprzejmie",
        "w związku z powyższym", "mając na uwadze"
    ]
    
    COLLOQUIAL_MARKERS = [
        "fajny", "super", "mega", "w sumie", "ogólnie",
        "no i", "tak naprawdę", "jakby", "w ogóle"
    ]
    
    SCIENTIFIC_MARKERS = [
        "hipoteza", "metodologia", "empiryczny", "teoretyczny",
        "analiza wskazuje", "badania dowodzą", "korelacja"
    ]
    
    text_lower = text.lower()
    
    formal_count = sum(1 for m in FORMAL_MARKERS if m in text_lower)
    colloquial_count = sum(1 for m in COLLOQUIAL_MARKERS if m in text_lower)
    scientific_count = sum(1 for m in SCIENTIFIC_MARKERS if m in text_lower)
    
    # Określ dominujący rejestr
    counts = {
        "formalny": formal_count,
        "potoczny": colloquial_count,
        "naukowy": scientific_count
    }
    dominant = max(counts, key=counts.get) if max(counts.values()) > 0 else "neutralny"
    
    # Wykryj mieszanie rejestrów
    issues = []
    
    if formal_count > 0 and colloquial_count > 0:
        issues.append({
            "type": "REGISTER_MIXING",
            "message": "Mieszanie rejestru formalnego z potocznym",
            "formal_markers": formal_count,
            "colloquial_markers": colloquial_count
        })
    
    if scientific_count > 0 and colloquial_count > 0:
        issues.append({
            "type": "REGISTER_MIXING",
            "message": "Mieszanie rejestru naukowego z potocznym",
            "scientific_markers": scientific_count,
            "colloquial_markers": colloquial_count
        })
    
    # Score
    if len(issues) == 0:
        score = 1.0
    elif len(issues) == 1:
        score = 0.7
    else:
        score = 0.4
    
    return issues, score, dominant


def check_sentence_variety(text: str) -> Tuple[List[Dict], float]:
    """
    Sprawdza różnorodność struktur składniowych.
    
    Returns:
        Tuple[lista problemów, score 0-1]
    """
    doc = nlp(text[:15000])
    sentences = list(doc.sents)
    
    if len(sentences) < 5:
        return [], 1.0
    
    issues = []
    
    # 1. Sprawdź monotonię początków zdań
    starters = [sent[0].text.lower() if len(sent) > 0 else "" for sent in sentences]
    starter_counts = Counter(starters)
    
    for starter, count in starter_counts.items():
        ratio = count / len(sentences)
        if ratio > 0.2 and count > 2:  # >20% zdań zaczyna się tak samo
            issues.append({
                "type": "MONOTONOUS_STARTERS",
                "starter": starter,
                "count": count,
                "percentage": round(ratio * 100, 1)
            })
    
    # 2. Sprawdź monotonię szyku (SVO)
    svo_count = 0
    for sent in sentences:
        tokens = list(sent)
        if len(tokens) >= 3:
            # Uproszczona heurystyka: NOUN + VERB + NOUN
            pos_pattern = [t.pos_ for t in tokens[:5]]
            if "NOUN" in pos_pattern[:2] and "VERB" in pos_pattern[1:4]:
                svo_count += 1
    
    svo_ratio = svo_count / len(sentences)
    if svo_ratio > 0.7:  # >70% zdań to SVO
        issues.append({
            "type": "MONOTONOUS_WORD_ORDER",
            "svo_percentage": round(svo_ratio * 100, 1),
            "recommendation": "Urozmaić szyk zdania - użyj inwersji, konstrukcji z emfazą"
        })
    
    # 3. Sprawdź różnorodność długości zdań
    lengths = [len(list(sent)) for sent in sentences]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        
        if variance < 10:  # Zbyt podobne długości
            issues.append({
                "type": "MONOTONOUS_SENTENCE_LENGTH",
                "variance": round(variance, 2),
                "recommendation": "Mieszaj zdania krótkie (5-10 słów) z dłuższymi (15-25 słów)"
            })
    
    # Score
    score = max(0, 1 - len(issues) * 0.2)
    
    return issues, score


def check_banned_phrases(text: str) -> Tuple[List[Dict], float]:
    """
    Sprawdza obecność zakazanych fraz AI.
    
    Returns:
        Tuple[lista znalezionych fraz, score 0-1]
    """
    text_lower = text.lower()
    found = []
    
    for category, phrases in BANNED_PHRASES_EXTENDED.items():
        for phrase in phrases:
            if phrase in text_lower:
                idx = text_lower.find(phrase)
                context = text[max(0, idx-10):min(len(text), idx+len(phrase)+10)]
                
                found.append({
                    "type": "BANNED_PHRASE",
                    "category": category,
                    "phrase": phrase,
                    "context": f"...{context}..."
                })
    
    # Score
    score = max(0, 1 - len(found) * 0.1)
    
    return found, score


def check_transition_words_usage(text: str) -> Tuple[Dict, float]:
    """
    Analizuje użycie słów łączących z podziałem na kategorie.
    
    Returns:
        Tuple[analiza, score 0-1]
    """
    text_lower = text.lower()
    
    usage = {}
    total_found = 0
    
    for category, words in TRANSITION_WORDS_CATEGORIZED.items():
        count = sum(1 for word in words if word in text_lower)
        usage[category] = count
        total_found += count
    
    # Sprawdź balans kategorii
    issues = []
    
    # Za dużo "dodawanie" w stosunku do innych
    if usage.get("dodawanie", 0) > total_found * 0.4 and total_found > 5:
        issues.append("Zbyt wiele słów łączących typu 'dodawanie' (również, także, ponadto)")
    
    # Brak kontrastu
    if usage.get("kontrast", 0) == 0 and total_found > 5:
        issues.append("Brak słów kontrastujących (jednak, natomiast, ale) - tekst może być monotonny")
    
    # Za dużo skutek/przyczyna razem
    if usage.get("skutek", 0) > total_found * 0.3:
        issues.append("Nadmierne użycie słów wyrażających skutek (dlatego, zatem, więc)")
    
    analysis = {
        "by_category": usage,
        "total": total_found,
        "issues": issues,
        "balance": "OK" if len(issues) == 0 else "UNBALANCED"
    }
    
    # Score
    score = max(0.5, 1 - len(issues) * 0.15)
    
    return analysis, score


# ================================================================
# 🎯 GŁÓWNA FUNKCJA ANALIZY
# ================================================================
def analyze_polish_quality(text: str) -> LanguageQualityResult:
    """
    Główna funkcja - kompleksowa analiza jakości języka polskiego.
    
    Args:
        text: Tekst do analizy
    
    Returns:
        LanguageQualityResult z pełną analizą
    """
    if not text or len(text.strip()) < 100:
        return LanguageQualityResult(
            score=0,
            issues=[{"type": "TEXT_TOO_SHORT", "message": "Tekst zbyt krótki do analizy"}],
            warnings=["Tekst musi mieć minimum 100 znaków"],
            recommendations=[],
            metrics={}
        )
    
    all_issues = []
    all_warnings = []
    all_recommendations = []
    scores = []
    
    # 1. Kolokacje
    collocation_issues, collocation_score = check_collocations(text)
    all_issues.extend(collocation_issues)
    scores.append(collocation_score * 0.25)  # Waga 25%
    
    if collocation_issues:
        all_recommendations.append(
            f"Popraw kolokacje: {collocation_issues[0]['found']} → {collocation_issues[0]['suggested']}"
        )
    
    # 2. Powtórzenia leksykalne
    repetition_issues, repetition_score = check_lexical_repetitions(text)
    all_issues.extend(repetition_issues)
    scores.append(repetition_score * 0.20)  # Waga 20%
    
    if len(repetition_issues) > 3:
        all_warnings.append("Nadmierne powtórzenia leksykalne - użyj synonimów lub zaimków")
    
    # 3. Spójność rejestru
    register_issues, register_score, dominant_register = check_register_consistency(text)
    all_issues.extend(register_issues)
    scores.append(register_score * 0.15)  # Waga 15%
    
    if register_issues:
        all_recommendations.append(
            f"Ujednolić rejestr stylistyczny - wykryto mieszanie rejestrów"
        )
    
    # 4. Różnorodność składniowa
    variety_issues, variety_score = check_sentence_variety(text)
    all_issues.extend(variety_issues)
    scores.append(variety_score * 0.15)  # Waga 15%
    
    if variety_issues:
        for issue in variety_issues:
            if issue["type"] == "MONOTONOUS_STARTERS":
                all_recommendations.append(
                    f"Urozmaić początki zdań - {issue['percentage']}% zaczyna się od '{issue['starter']}'"
                )
    
    # 5. Banned phrases
    banned_issues, banned_score = check_banned_phrases(text)
    all_issues.extend(banned_issues)
    scores.append(banned_score * 0.15)  # Waga 15%
    
    if banned_issues:
        categories = set(i["category"] for i in banned_issues)
        all_warnings.append(f"Znaleziono typowe frazy AI: {', '.join(categories)}")
    
    # 6. Transition words
    transition_analysis, transition_score = check_transition_words_usage(text)
    scores.append(transition_score * 0.10)  # Waga 10%
    
    if transition_analysis["issues"]:
        all_recommendations.extend(transition_analysis["issues"])
    
    # Oblicz końcowy score
    final_score = sum(scores) * 100
    
    # Metryki
    metrics = {
        "collocation_score": round(collocation_score, 2),
        "repetition_score": round(repetition_score, 2),
        "register_score": round(register_score, 2),
        "variety_score": round(variety_score, 2),
        "banned_phrases_score": round(banned_score, 2),
        "transition_score": round(transition_score, 2),
        "dominant_register": dominant_register,
        "transition_analysis": transition_analysis
    }
    
    return LanguageQualityResult(
        score=final_score,
        issues=all_issues,
        warnings=all_warnings,
        recommendations=all_recommendations,
        metrics=metrics
    )


# ================================================================
# 🔧 HELPER: Szybka walidacja
# ================================================================
def quick_polish_check(text: str) -> Dict[str, Any]:
    """
    Szybka walidacja - tylko najważniejsze elementy.
    """
    result = {
        "status": "OK",
        "issues_count": 0,
        "critical": []
    }
    
    # Tylko banned phrases i kolokacje
    banned, _ = check_banned_phrases(text)
    collocations, _ = check_collocations(text)
    
    result["issues_count"] = len(banned) + len(collocations)
    
    if banned:
        result["status"] = "WARN"
        result["critical"].append(f"Frazy AI: {banned[0]['phrase']}")
    
    if collocations:
        result["status"] = "WARN"
        result["critical"].append(f"Błędna kolokacja: {collocations[0]['found']}")
    
    return result


# ================================================================
# 🔧 HELPER: Sugestie poprawy
# ================================================================
def generate_improvement_suggestions(issues: List[Dict]) -> List[str]:
    """
    Generuje konkretne sugestie poprawy na podstawie wykrytych problemów.
    """
    suggestions = []
    
    for issue in issues[:5]:
        issue_type = issue.get("type", "")
        
        if issue_type == "COLLOCATION_ERROR":
            suggestions.append(
                f"Zamień '{issue['found']}' na '{issue['suggested']}' - poprawna kolokacja polska"
            )
        
        elif issue_type == "LEXICAL_REPETITION":
            word = issue.get("word", "")
            suggestions.append(
                f"Słowo '{word}' powtarza się zbyt często - użyj synonimu lub zaimka"
            )
        
        elif issue_type == "BANNED_PHRASE":
            phrase = issue.get("phrase", "")
            category = issue.get("category", "")
            
            if category == "typowe_ai_openers":
                suggestions.append(
                    f"Usuń '{phrase}' - to typowy marker tekstu AI. Zacznij od konkretnej informacji."
                )
            elif category == "pleonazmy":
                suggestions.append(
                    f"Usuń pleonazm '{phrase}' - wyrażenie redundantne"
                )
            else:
                suggestions.append(
                    f"Rozważ usunięcie '{phrase}' - może brzmieć sztucznie"
                )
        
        elif issue_type == "MONOTONOUS_STARTERS":
            suggestions.append(
                f"Urozmaić początki zdań - {issue.get('percentage', 0)}% zaczyna się od '{issue.get('starter', '')}'"
            )
        
        elif issue_type == "REGISTER_MIXING":
            suggestions.append(
                "Ujednolicić rejestr stylistyczny - nie mieszaj języka formalnego z potocznym"
            )
    
    return suggestions
