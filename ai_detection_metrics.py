"""
===============================================================================
🤖 AI DETECTION METRICS v33.0
===============================================================================
Moduł do wykrywania tekstu wygenerowanego przez AI.

Metryki:
- Burstiness (zmienność długości zdań)
- Vocabulary Richness (TTR - Type-Token Ratio)
- Lexical Sophistication (średnia częstość słów - Zipf)
- Starter Entropy (różnorodność początków zdań)
- Word Repetition (powtórzenia słów)

Humanness Score = średnia ważona wszystkich metryk (0-100)

CRITICAL validations:
- Forbidden phrases check (BLOKUJE batch!)
- Burstiness < 1.5 (BLOKUJE batch!)
- JITTER validation
- Triplets validation
- Word repetition > 8× (BLOKUJE batch!)

v33.0 CHANGES:
- Rozszerzono SHORT_INSERTS_LIBRARY (29 wtrąceń)
- Rozszerzono SYNONYM_MAP (27 słów) + dynamiczne z synonym_service
- Nowe progi: burstiness < 1.5 = CRITICAL, < 2.0 = WARNING
- Dodano fix_instructions z konkretnymi przykładami
- Dodano analyze_sentence_distribution, generate_burstiness_fix
- Integracja z synonym_service.py
===============================================================================
"""

import re
import math
import statistics
from collections import Counter
from typing import Dict, List, Any, Tuple
from enum import Enum

# ================================================================
# 📦 Opcjonalny import wordfreq
# ================================================================
try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
    print("[AI_DETECTION] ✅ wordfreq available")
except ImportError:
    WORDFREQ_AVAILABLE = False
    print("[AI_DETECTION] ⚠️ wordfreq not available - lexical sophistication disabled")

# ================================================================
# 📦 v33.3: Opcjonalny import spacy dla POS diversity
# ================================================================
try:
    import spacy
    try:
        _nlp_pos = spacy.load("pl_core_news_sm")
        SPACY_POS_AVAILABLE = True
        print("[AI_DETECTION] ✅ spacy pl_core_news_sm loaded for POS analysis")
    except OSError:
        SPACY_POS_AVAILABLE = False
        print("[AI_DETECTION] ⚠️ spacy pl_core_news_sm not found - POS diversity disabled")
except ImportError:
    SPACY_POS_AVAILABLE = False
    print("[AI_DETECTION] ⚠️ spacy not available - POS diversity disabled")


# ================================================================
# 📊 KONFIGURACJA
# ================================================================
class AIDetectionConfig:
    """Progi dla metryk AI detection."""
    
    # Burstiness (zmienność długości zdań)
    BURSTINESS_CRITICAL_LOW = 2.0
    BURSTINESS_WARNING_LOW = 2.8
    BURSTINESS_OK_MIN = 2.8
    BURSTINESS_OK_MAX = 4.2
    BURSTINESS_WARNING_HIGH = 4.8
    BURSTINESS_CRITICAL_HIGH = 4.8
    
    # Vocabulary Richness (TTR)
    TTR_CRITICAL = 0.40
    TTR_WARNING = 0.48
    TTR_OK = 0.55
    
    # Lexical Sophistication (Zipf)
    ZIPF_CRITICAL = 5.5
    ZIPF_WARNING = 5.0
    ZIPF_OK = 4.5
    
    # Starter Entropy
    ENTROPY_CRITICAL = 0.50
    ENTROPY_WARNING = 0.65
    ENTROPY_OK = 0.75
    
    # Word Repetition
    REPETITION_OK = 5
    REPETITION_WARNING = 7
    REPETITION_CRITICAL = 8
    
    # Humanness Score
    HUMANNESS_CRITICAL = 50
    HUMANNESS_WARNING = 70
    
    # Wagi - 🔧 FIX v34.3: Zsynchronizowane z calculate_humanness_score
    WEIGHTS = {
        "burstiness": 0.25,
        "vocabulary": 0.15,      # było 0.20
        "sophistication": 0.10,  # było 0.15
        "entropy": 0.20,
        "repetition": 0.20,
        "pos_diversity": 0.10    # 🆕 v33.3
    }


class Severity(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ================================================================
# 🇵🇱 POLSKIE STOP WORDS
# ================================================================
POLISH_STOP_WORDS = {
    "i", "w", "na", "z", "do", "że", "się", "nie", "to", "o", "jak", 
    "ale", "co", "jest", "za", "po", "tak", "czy", "już", "od", "przez",
    "dla", "by", "być", "a", "więc", "też", "tylko", "lub", "oraz",
    "jego", "jej", "ich", "tym", "tego", "tej", "te", "ta", "ten",
    "który", "która", "które", "których", "którzy", "której",
    "może", "bardzo", "kiedy", "gdy", "tu", "tam", "teraz", "wtedy",
    "mnie", "mi", "ci", "cię", "go", "mu", "ją", "je", "nas", "was", "im",
    "jednak", "jeszcze", "będzie", "były", "był", "była", "było",
    "są", "będą", "mają", "ma", "można", "trzeba", "należy"
}


# ================================================================
# 🔧 FUNKCJE POMOCNICZE
# ================================================================
def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return sentences


def tokenize(text: str) -> List[str]:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    words = re.findall(r'\b[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+\b', text)
    return words


def tokenize_no_stopwords(text: str) -> List[str]:
    words = tokenize(text)
    return [w for w in words if w not in POLISH_STOP_WORDS]


# ================================================================
# 📊 METRYKI
# ================================================================
def calculate_burstiness(text: str) -> Dict[str, Any]:
    sentences = split_into_sentences(text)
    
    if len(sentences) < 5:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Za mało zdań do analizy (min 5)",
            "sentence_count": len(sentences)
        }
    
    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
    
    burstiness = (std_len / mean_len * 5) if mean_len > 0 else 0
    burstiness = round(burstiness, 2)
    
    config = AIDetectionConfig()
    if burstiness < config.BURSTINESS_CRITICAL_LOW:
        status = Severity.CRITICAL
        message = f"Tekst monotonny (burstiness {burstiness} < {config.BURSTINESS_CRITICAL_LOW}). Dodaj krótkie zdania 5-8 słów."
    elif burstiness < config.BURSTINESS_WARNING_LOW:
        status = Severity.WARNING
        message = f"Niska zmienność zdań. Dodaj więcej krótkich zdań."
    elif burstiness > config.BURSTINESS_CRITICAL_HIGH:
        status = Severity.CRITICAL
        message = f"Tekst chaotyczny (burstiness {burstiness} > {config.BURSTINESS_CRITICAL_HIGH}). Wyrównaj rytm."
    elif burstiness > config.BURSTINESS_OK_MAX:
        status = Severity.WARNING
        message = f"Za duża zmienność. Wyrównaj długości zdań."
    else:
        status = Severity.OK
        message = "Burstiness w normie"
    
    return {
        "value": burstiness,
        "status": status.value,
        "message": message,
        "sentence_count": len(sentences),
        "mean_length": round(mean_len, 1),
        "std_length": round(std_len, 1),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }


# ================================================================
# 🆕 v33.3: POS DIVERSITY (różnorodność części mowy)
# ================================================================
def calculate_pos_diversity(text: str) -> Dict[str, Any]:
    """
    v33.3: Mierzy zróżnicowanie części mowy na początku zdań.
    
    AI często zaczyna zdania od tych samych konstrukcji gramatycznych
    (np. "Warto..." - VERB, "Ważne jest..." - ADJ).
    
    Wysoka entropia POS = bardziej ludzki tekst.
    
    Returns:
        Dict z: value (0-1), status, first_pos_distribution
    """
    if not SPACY_POS_AVAILABLE:
        return {
            "value": 0.5,  # Neutral default
            "status": "DISABLED",
            "message": "spacy niedostępny - POS analysis wyłączona",
            "enabled": False
        }
    
    try:
        doc = _nlp_pos(text)
        first_pos = []
        
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct and not t.is_space]
            if tokens:
                first_pos.append(tokens[0].pos_)
        
        if len(first_pos) < 5:
            return {
                "value": 0.5,
                "status": "WARNING",
                "message": "Za mało zdań do analizy POS (min 5)",
                "enabled": True,
                "sentence_count": len(first_pos)
            }
        
        # Oblicz entropię Shannon'a dla POS
        counter = Counter(first_pos)
        total = len(first_pos)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalizuj do 0-1
        unique_pos = len(counter)
        max_entropy = math.log2(unique_pos) if unique_pos > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        normalized = round(normalized, 2)
        
        # Status
        if normalized < 0.4:
            status = "CRITICAL"
            message = f"Monotonne konstrukcje gramatyczne (POS entropy {normalized})"
        elif normalized < 0.6:
            status = "WARNING"
            message = f"Niska różnorodność gramatyczna"
        else:
            status = "OK"
            message = f"Dobra różnorodność POS"
        
        # Top 3 najczęstsze POS
        top_pos = counter.most_common(3)
        
        return {
            "value": normalized,
            "status": status,
            "message": message,
            "enabled": True,
            "unique_pos_count": unique_pos,
            "total_sentences": total,
            "top_pos": [{"pos": p, "count": c, "percent": round(c/total*100)} for p, c in top_pos],
            "pos_distribution": dict(counter)
        }
        
    except Exception as e:
        print(f"[AI_DETECTION] POS analysis error: {e}")
        return {
            "value": 0.5,
            "status": "ERROR",
            "message": f"Błąd analizy POS: {e}",
            "enabled": True
        }


# ================================================================
# 🆕 v33.0: SHORT INSERTS LIBRARY (dla fix_instructions)
# ================================================================
SHORT_INSERTS_LIBRARY = [
    # Potwierdzenia (2-4 słowa)
    "To działa.",
    "Efekt? Natychmiastowy.",
    "Proste rozwiązanie.",
    "I to nie wszystko.",
    "Ale jest więcej.",
    "Sprawdzone.",
    "Nic trudnego.",
    "Różnica jest widoczna.",
    "Brzmi skomplikowanie? Nie jest.",
    "Warto spróbować.",
    "Klucz do sukcesu.",
    "To podstawa.",
    "Efekt? Szybki.",
    "Proste, prawda?",
    "A co dalej?",
    "Działa od razu.",
    "Bez niespodzianek.",
    "Czas na konkrety.",
    "I tu zaczyna się magia.",
    "Rezultat mówi sam za siebie.",
    # Pytania retoryczne
    "Ale czy to wystarczy?",
    "Co dalej?",
    "Dlaczego to ważne?",
    "Jak to osiągnąć?",
    "A może inaczej?",
    # Akcenty dramatyczne
    "Efekt.",
    "Rezultat?",
    "Prosto.",
    "Skutecznie.",
]


# ================================================================
# 🆕 v33.0: ANALYZE SENTENCE DISTRIBUTION
# ================================================================
def analyze_sentence_distribution(text: str) -> Dict[str, Any]:
    """
    Analizuje rozkład długości zdań dla burstiness fix.
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) < 3:
        return {
            "short_count": 0, "medium_count": 0, "long_count": 0,
            "total": len(sentences), "distribution": [0, 0, 0],
            "issues": ["Za mało zdań do analizy"]
        }
    
    lengths = [len(s.split()) for s in sentences]
    short = sum(1 for l in lengths if 5 <= l <= 8)
    medium = sum(1 for l in lengths if 12 <= l <= 18)
    long = sum(1 for l in lengths if l >= 20)
    
    total = len(lengths)
    distribution = [
        round(short / total * 100, 1),
        round(medium / total * 100, 1),
        round(long / total * 100, 1)
    ]
    
    issues = []
    if distribution[0] < 10:
        issues.append(f"Za mało krótkich zdań: {distribution[0]}% vs cel 20%")
    if distribution[2] < 15:
        issues.append(f"Za mało długich zdań: {distribution[2]}% vs cel 30%")
    
    return {
        "short_count": short,
        "medium_count": medium,
        "long_count": long,
        "total": total,
        "distribution": distribution,
        "distribution_label": f"[{distribution[0]}% krótkich, {distribution[1]}% średnich, {distribution[2]}% długich]",
        "issues": issues
    }


# ================================================================
# 🆕 v33.0: GENERATE BURSTINESS FIX INSTRUCTION
# ================================================================
def generate_burstiness_fix(burstiness: float, sentence_distribution: Dict) -> Dict[str, Any]:
    """
    Generuje konkretne instrukcje naprawy burstiness.
    """
    import random
    
    if burstiness >= 2.0:
        return {"needed": False, "message": "Burstiness OK"}
    
    inserts = random.sample(SHORT_INSERTS_LIBRARY, min(3, len(SHORT_INSERTS_LIBRARY)))
    
    rewrite_examples = [
        {
            "before": "Witamina C wspomaga syntezę kolagenu, co poprawia elastyczność skóry.",
            "after": "Witamina C? Klucz do kolagenu. Wspomaga jego syntezę i poprawia elastyczność skóry – efekt widać już po kilku tygodniach."
        },
        {
            "before": "Suplementy diety zawierają wiele cennych składników odżywczych.",
            "after": "Suplementy działają. Zawierają składniki, które wspierają skórę od wewnątrz – witaminy, minerały, antyoksydanty. Proste i skuteczne."
        }
    ]
    
    # Buduj fix_instruction bez backslashy w f-string
    quoted_inserts = ['"' + s + '"' for s in inserts]
    fix_instruction = "Dodaj krótkie zdania: " + ", ".join(quoted_inserts)
    
    return {
        "needed": True,
        "burstiness": burstiness,
        "target": "≥ 2.0",
        "fix_instruction": fix_instruction,
        "insert_suggestions": inserts,
        "rewrite_example": random.choice(rewrite_examples),
        "distribution": sentence_distribution.get("distribution_label", ""),
        "tip": "Wzór: KRÓTKIE (5-8 słów) → DŁUGIE (20-30 słów) → ŚREDNIE (10-15 słów)"
    }


# ================================================================
# 🆕 v33.0: EXTENDED SYNONYM MAP
# ================================================================
SYNONYM_MAP = {
    # Skóra / uroda
    "skóra": ["cera", "naskórek", "powierzchnia skóry", "tkanka", "powłoka"],
    "witamina": ["mikroskładnik", "substancja odżywcza", "składnik", "nutrient"],
    "suplement": ["preparat", "produkt", "środek", "wsparcie"],
    "kolagen": ["białko strukturalne", "włókna kolagenowe", "substancja budulcowa"],
    "nawilżenie": ["hydratacja", "uwodnienie", "poziom wilgoci"],
    # Przymiotniki
    "ważny": ["istotny", "znaczący", "zasadniczy", "niezbędny", "doniosły"],
    "dobry": ["skuteczny", "wartościowy", "korzystny", "efektywny", "pomocny"],
    "zdrowy": ["prawidłowy", "właściwy", "optymalny"],
    "duży": ["znaczny", "spory", "pokaźny", "niemały"],
    "mały": ["niewielki", "drobny", "ograniczony"],
    "nowy": ["nowoczesny", "świeży", "najnowszy", "aktualny"],
    # Czasowniki
    "poprawia": ["wspiera", "wzmacnia", "podnosi", "ulepsza"],
    "pomaga": ["wspiera", "ułatwia", "wspomaga", "przyczynia się"],
    "zawiera": ["posiada", "obejmuje", "ma w składzie"],
    "powoduje": ["wywołuje", "skutkuje", "prowadzi do"],
    "działa": ["funkcjonuje", "pracuje", "oddziałuje", "wpływa"],
    "chroni": ["zabezpiecza", "ochrania", "osłania"],
    # Usługi / biznes
    "firma": ["przedsiębiorstwo", "spółka", "wykonawca", "usługodawca"],
    "usługa": ["świadczenie", "realizacja", "obsługa", "serwis"],
    "klient": ["zleceniodawca", "usługobiorca", "zamawiający"],
    "cena": ["koszt", "stawka", "wycena", "taryfa"],
    "profesjonalny": ["doświadczony", "wykwalifikowany", "fachowy"],
}


# ================================================================
# 🆕 v33.0: CHECK WORD REPETITION DETAILED (z dynamicznymi synonimami)
# ================================================================
def check_word_repetition_detailed(text: str, max_per_500: int = 5) -> Dict[str, Any]:
    """
    Sprawdza powtórzenia słów z dynamicznymi sugestiami synonimów.
    """
    # Próba dynamicznego importu synonym_service
    try:
        from synonym_service import get_synonyms
        use_dynamic = True
    except ImportError:
        use_dynamic = False
    
    words = re.findall(r'\b[a-ząćęłńóśźż]{4,}\b', text.lower())
    word_count = len(words)
    word_freq = Counter(words)
    
    stop_words = {'jest', 'oraz', 'jako', 'przez', 'które', 'która', 'który', 
                  'może', 'będzie', 'było', 'były', 'tego', 'tej', 'tych',
                  'bardzo', 'także', 'również', 'jednak', 'więc', 'czyli'}
    
    scale = max(1, word_count / 500)
    limit = int(max_per_500 * scale)
    
    violations = []
    warnings = []
    
    def _get_synonyms(word: str) -> List[str]:
        if use_dynamic:
            result = get_synonyms(word)
            return result.get("synonyms", [])
        return SYNONYM_MAP.get(word, [])
    
    for word, count in word_freq.most_common(20):
        if word in stop_words:
            continue
        
        if count > limit * 1.6:  # > 8× = CRITICAL
            synonyms = _get_synonyms(word)
            violations.append({
                "word": word, "count": count, "limit": limit,
                "synonyms": synonyms,
                "suggestion": f"Użyj: {', '.join(synonyms[:3])}" if synonyms else "Znajdź synonimy"
            })
        elif count > limit:  # > 5× = WARNING
            synonyms = _get_synonyms(word)
            warnings.append({
                "word": word, "count": count, "limit": limit, "synonyms": synonyms
            })
    
    if violations:
        status = Severity.CRITICAL
        viol_str = ', '.join([f'{v["word"]}({v["count"]}×)' for v in violations[:3]])
        message = f"🔴 POWTÓRZENIA: {viol_str}"
        should_block = True
    elif warnings:
        status = Severity.WARNING
        warn_str = ', '.join([f'{w["word"]}({w["count"]}×)' for w in warnings[:3]])
        message = f"⚠️ Powtórzenia: {warn_str}"
        should_block = False
    else:
        status = Severity.OK
        message = "Powtórzenia OK ✓"
        should_block = False
    
    top_words = [(w, c) for w, c in word_freq.most_common(10) if w not in stop_words][:5]
    
    return {
        "status": status.value,
        "violations": violations,
        "warnings": warnings,
        "message": message,
        "top_words": top_words,
        "should_block": should_block
    }


def calculate_vocabulary_richness(text: str) -> Dict[str, Any]:
    words = tokenize_no_stopwords(text)
    
    if len(words) < 50:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Za mało słów do analizy (min 50)",
            "word_count": len(words)
        }
    
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    ttr = round(ttr, 3)
    
    config = AIDetectionConfig()
    if ttr < config.TTR_CRITICAL:
        status = Severity.CRITICAL
        message = f"Bardzo ubogi zasób słów (TTR {ttr} < {config.TTR_CRITICAL})"
    elif ttr < config.TTR_WARNING:
        status = Severity.WARNING
        message = f"Mało urozmaicone słownictwo. Użyj synonimów."
    elif ttr >= config.TTR_OK:
        status = Severity.OK
        message = "Bogate słownictwo"
    else:
        status = Severity.WARNING
        message = "Słownictwo poniżej optimum"
    
    return {
        "value": ttr,
        "status": status.value,
        "message": message,
        "unique_words": len(unique_words),
        "total_words": len(words)
    }


def calculate_lexical_sophistication(text: str) -> Dict[str, Any]:
    if not WORDFREQ_AVAILABLE:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "wordfreq niedostępny",
            "available": False
        }
    
    words = tokenize_no_stopwords(text)
    
    if len(words) < 50:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Za mało słów do analizy",
            "available": True
        }
    
    zipf_scores = []
    for word in words:
        freq = zipf_frequency(word, 'pl')
        if freq > 0:
            zipf_scores.append(freq)
    
    if not zipf_scores:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Nie udało się obliczyć częstości słów",
            "available": True
        }
    
    avg_zipf = statistics.mean(zipf_scores)
    avg_zipf = round(avg_zipf, 2)
    
    config = AIDetectionConfig()
    if avg_zipf > config.ZIPF_CRITICAL:
        status = Severity.CRITICAL
        message = f"Zbyt proste słownictwo (avg Zipf {avg_zipf} > {config.ZIPF_CRITICAL})"
    elif avg_zipf > config.ZIPF_WARNING:
        status = Severity.WARNING
        message = f"Słownictwo dość podstawowe"
    elif avg_zipf <= config.ZIPF_OK:
        status = Severity.OK
        message = "Dobry mix słownictwa"
    else:
        status = Severity.WARNING
        message = "Słownictwo w normie"
    
    return {
        "value": avg_zipf,
        "status": status.value,
        "message": message,
        "words_analyzed": len(zipf_scores),
        "available": True
    }


def calculate_starter_entropy(text: str) -> Dict[str, Any]:
    sentences = split_into_sentences(text)
    
    if len(sentences) < 5:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Za mało zdań do analizy",
            "sentence_count": len(sentences)
        }
    
    starters = []
    for s in sentences:
        words = s.split()[:3]
        if words:
            starter = ' '.join(words).lower()
            starters.append(starter)
    
    if not starters:
        return {
            "value": 0,
            "status": Severity.WARNING.value,
            "message": "Nie znaleziono starterów"
        }
    
    counter = Counter(starters)
    total = len(starters)
    
    entropy = 0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    max_entropy = math.log2(total) if total > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    normalized_entropy = round(normalized_entropy, 2)
    
    repetitive = {k: v for k, v in counter.items() if v >= 2}
    
    config = AIDetectionConfig()
    if normalized_entropy < config.ENTROPY_CRITICAL:
        status = Severity.CRITICAL
        message = f"Bardzo powtarzalne początki zdań (entropy {normalized_entropy})"
    elif normalized_entropy < config.ENTROPY_WARNING:
        status = Severity.WARNING
        message = f"Mało różnorodne początki zdań"
    elif normalized_entropy >= config.ENTROPY_OK:
        status = Severity.OK
        message = "Dobra różnorodność początków zdań"
    else:
        status = Severity.WARNING
        message = "Różnorodność starterów poniżej optimum"
    
    suggestions = []
    for starter, count in sorted(repetitive.items(), key=lambda x: -x[1])[:3]:
        suggestions.append(f"Zmień starter '{starter}' (użyty {count}×)")
    
    return {
        "value": normalized_entropy,
        "status": status.value,
        "message": message,
        "unique_starters": len(counter),
        "total_sentences": len(sentences),
        "repetitive_starters": repetitive,
        "suggestions": suggestions
    }


def calculate_word_repetition(text: str) -> Dict[str, Any]:
    words = tokenize_no_stopwords(text)
    
    if len(words) < 50:
        return {
            "value": 1.0,
            "status": Severity.WARNING.value,
            "message": "Za mało słów do analizy",
            "word_count": len(words)
        }
    
    counter = Counter(words)
    config = AIDetectionConfig()
    
    overused = {}
    warnings_list = []
    critical_list = []
    
    for word, count in counter.most_common(20):
        if count > config.REPETITION_CRITICAL:
            critical_list.append({"word": word, "count": count})
            overused[word] = count
        elif count > config.REPETITION_WARNING:
            warnings_list.append({"word": word, "count": count})
            overused[word] = count
        elif count > config.REPETITION_OK:
            warnings_list.append({"word": word, "count": count})
    
    overused_count = sum(overused.values())
    score = 1 - (overused_count / len(words)) if words else 1
    score = round(max(0, score), 2)
    
    if critical_list:
        status = Severity.CRITICAL
        message = f"Słowa powtórzone > {config.REPETITION_CRITICAL}×: {', '.join([c['word'] for c in critical_list[:3]])}"
    elif warnings_list:
        status = Severity.WARNING
        message = f"Słowa powtórzone > {config.REPETITION_OK}×. Użyj synonimów."
    else:
        status = Severity.OK
        message = "Brak nadmiernych powtórzeń"
    
    # 🔧 FIX v34.3: Usunięto lokalną SYNONYM_MAP - używamy globalnej (27 słów)
    suggestions = []
    for word in overused:
        if word in SYNONYM_MAP:  # Używa globalnej SYNONYM_MAP z linii 431
            suggestions.append(f"'{word}' → {', '.join(SYNONYM_MAP[word][:3])}")
    
    return {
        "value": score,
        "status": status.value,
        "message": message,
        "overused_words": overused,
        "critical_words": critical_list,
        "warning_words": warnings_list,
        "suggestions": suggestions[:5]
    }


# ================================================================
# 🎯 GŁÓWNA FUNKCJA - HUMANNESS SCORE
# ================================================================
def calculate_humanness_score(text: str) -> Dict[str, Any]:
    config = AIDetectionConfig()
    
    burstiness = calculate_burstiness(text)
    vocabulary = calculate_vocabulary_richness(text)
    sophistication = calculate_lexical_sophistication(text)
    entropy = calculate_starter_entropy(text)
    repetition = calculate_word_repetition(text)
    
    # v33.3: POS diversity
    pos_diversity = calculate_pos_diversity(text)
    
    def normalize_burstiness(val):
        if val < config.BURSTINESS_CRITICAL_LOW:
            return 0.0
        elif val < config.BURSTINESS_OK_MIN:
            return (val - config.BURSTINESS_CRITICAL_LOW) / (config.BURSTINESS_OK_MIN - config.BURSTINESS_CRITICAL_LOW) * 0.5
        elif val <= config.BURSTINESS_OK_MAX:
            return 1.0
        elif val < config.BURSTINESS_CRITICAL_HIGH:
            return 1.0 - (val - config.BURSTINESS_OK_MAX) / (config.BURSTINESS_CRITICAL_HIGH - config.BURSTINESS_OK_MAX) * 0.5
        else:
            return 0.0
    
    def normalize_ttr(val):
        if val >= config.TTR_OK:
            return 1.0
        elif val >= config.TTR_WARNING:
            return 0.5 + (val - config.TTR_WARNING) / (config.TTR_OK - config.TTR_WARNING) * 0.5
        elif val >= config.TTR_CRITICAL:
            return (val - config.TTR_CRITICAL) / (config.TTR_WARNING - config.TTR_CRITICAL) * 0.5
        else:
            return 0.0
    
    def normalize_zipf(val):
        if not WORDFREQ_AVAILABLE or val == 0:
            return 0.5
        if val <= config.ZIPF_OK:
            return 1.0
        elif val <= config.ZIPF_WARNING:
            return 0.5 + (config.ZIPF_WARNING - val) / (config.ZIPF_WARNING - config.ZIPF_OK) * 0.5
        elif val <= config.ZIPF_CRITICAL:
            return (config.ZIPF_CRITICAL - val) / (config.ZIPF_CRITICAL - config.ZIPF_WARNING) * 0.5
        else:
            return 0.0
    
    def normalize_entropy(val):
        if val >= config.ENTROPY_OK:
            return 1.0
        elif val >= config.ENTROPY_WARNING:
            return 0.5 + (val - config.ENTROPY_WARNING) / (config.ENTROPY_OK - config.ENTROPY_WARNING) * 0.5
        elif val >= config.ENTROPY_CRITICAL:
            return (val - config.ENTROPY_CRITICAL) / (config.ENTROPY_WARNING - config.ENTROPY_CRITICAL) * 0.5
        else:
            return 0.0
    
    # v33.3: Normalize POS diversity (same scale as entropy)
    def normalize_pos(val):
        if not SPACY_POS_AVAILABLE or val == 0:
            return 0.5  # Neutral if disabled
        if val >= 0.6:
            return 1.0
        elif val >= 0.4:
            return 0.5 + (val - 0.4) / 0.2 * 0.5
        else:
            return val / 0.4 * 0.5
    
    scores = {
        "burstiness": normalize_burstiness(burstiness.get("value", 0)),
        "vocabulary": normalize_ttr(vocabulary.get("value", 0)),
        "sophistication": normalize_zipf(sophistication.get("value", 0)),
        "entropy": normalize_entropy(entropy.get("value", 0)),
        "repetition": repetition.get("value", 1.0),
        "pos_diversity": normalize_pos(pos_diversity.get("value", 0.5))  # v33.3
    }
    
    # 🔧 FIX v34.3: Używamy wag z konfiguracji (jedno źródło prawdy)
    weights = config.WEIGHTS
    
    humanness = sum(scores[k] * weights.get(k, 0) for k in scores)
    humanness_score = round(humanness * 100, 0)
    
    if humanness_score < config.HUMANNESS_CRITICAL:
        status = Severity.CRITICAL
        overall_message = f"CRITICAL: Tekst wygląda na AI (score {humanness_score}). Przepisz!"
    elif humanness_score < config.HUMANNESS_WARNING:
        status = Severity.WARNING
        overall_message = f"WARNING: Tekst wymaga poprawy (score {humanness_score})"
    else:
        status = Severity.OK
        overall_message = f"OK: Tekst wygląda naturalnie (score {humanness_score})"
    
    all_warnings = []
    if burstiness.get("status") != "OK":
        all_warnings.append(burstiness.get("message"))
    if vocabulary.get("status") != "OK":
        all_warnings.append(vocabulary.get("message"))
    if sophistication.get("status") not in ["OK", "WARNING"] or sophistication.get("value", 0) > config.ZIPF_WARNING:
        all_warnings.append(sophistication.get("message"))
    if entropy.get("status") != "OK":
        all_warnings.append(entropy.get("message"))
    if repetition.get("status") != "OK":
        all_warnings.append(repetition.get("message"))
    # v33.3: POS diversity warnings
    if pos_diversity.get("status") not in ["OK", "DISABLED"] and pos_diversity.get("enabled", True):
        all_warnings.append(pos_diversity.get("message"))
    
    all_suggestions = []
    all_suggestions.extend(entropy.get("suggestions", []))
    all_suggestions.extend(repetition.get("suggestions", []))
    
    return {
        "humanness_score": int(humanness_score),
        "status": status.value,
        "message": overall_message,
        "components": {
            "burstiness": burstiness,
            "vocabulary_richness": vocabulary,
            "lexical_sophistication": sophistication,
            "starter_entropy": entropy,
            "word_repetition": repetition,
            "pos_diversity": pos_diversity  # v33.3
        },
        "normalized_scores": scores,
        "warnings": all_warnings[:5],
        "suggestions": all_suggestions[:5]
    }


# ================================================================
# 🔍 QUICK CHECK
# ================================================================
def quick_ai_check(text: str) -> Dict[str, Any]:
    burstiness = calculate_burstiness(text)
    humanness = calculate_humanness_score(text)
    
    return {
        "humanness_score": humanness["humanness_score"],
        "status": humanness["status"],
        "burstiness": burstiness["value"],
        "top_warning": humanness["warnings"][0] if humanness["warnings"] else None
    }


# ================================================================
# 🆕 v33.0: CRITICAL: FORBIDDEN PHRASES CHECK (rozszerzono!)
# ================================================================
FORBIDDEN_PATTERNS = [
    # Frazy typowe dla AI
    (r'\bwarto wiedzieć\b', "warto wiedzieć"),
    (r'\bnależy pamiętać\b', "należy pamiętać"),
    (r'\bnależy podkreślić\b', "należy podkreślić"),
    (r'\bkluczowy aspekt\b', "kluczowy aspekt"),
    (r'\bkompleksowe rozwiązanie\b', "kompleksowe rozwiązanie"),
    (r'\bholistyczne podejście\b', "holistyczne podejście"),
    (r'\bw dzisiejszych czasach\b', "w dzisiejszych czasach"),
    (r'\bnie ulega wątpliwości\b', "nie ulega wątpliwości"),
    (r'\bcoraz więcej osób\b', "coraz więcej osób"),
    (r'\bw tym artykule\b', "w tym artykule"),
    (r'\bpodsumowując\b', "podsumowując"),
    (r'\bjak już wspomniano\b', "jak już wspomniano"),
    (r'\bkażdy z nas\b', "każdy z nas"),
    (r'\bnie jest tajemnicą\b', "nie jest tajemnicą"),
    (r'\bpowszechnie wiadomo\b', "powszechnie wiadomo"),
    (r'\btrudno przecenić\b', "trudno przecenić"),
    (r'\bw erze\s+\w+\b', "w erze..."),
    (r'\bw dobie\s+\w+\b', "w dobie..."),
    (r'\bw obliczu\b', "w obliczu"),
    (r'\bna przestrzeni lat\b', "na przestrzeni lat"),
]

# 🆕 v33.0: Słowa zakazane (pojedyncze)
FORBIDDEN_WORDS = [
    "kluczowy", "kompleksowy", "innowacyjny", "holistyczny", 
    "transformacyjny", "fundamentalny", "niewątpliwie", "wieloaspektowy",
    "przełomowy", "bezsprzecznie", "rewolucyjny", "optymalizować"
]

# 🆕 v33.0: Replacements dla zakazanych fraz
FORBIDDEN_REPLACEMENTS = {
    "coraz więcej osób": "wiele osób",
    "w dzisiejszych czasach": "[USUŃ]",
    "warto wiedzieć": "[USUŃ]",
    "należy podkreślić": "[USUŃ]",
    "podsumowując": "[zamień na konkretne zakończenie]",
    "w tym artykule": "[NIGDY nie używaj]",
    "kluczowy": "istotny/ważny",
    "kompleksowy": "pełny/całościowy",
    "innowacyjny": "nowoczesny/nowatorski",
    "holistyczny": "całościowy",
}

def check_forbidden_phrases(text: str) -> Dict[str, Any]:
    """
    🆕 v33.0: Sprawdza zakazane frazy i słowa.
    Zwraca should_block=True jeśli znaleziono ≥1 frazę!
    """
    text_lower = text.lower()
    found_phrases = []
    found_words = []
    replacements = []
    
    # Sprawdź frazy
    for pattern, name in FORBIDDEN_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_phrases.append(name)
            if name in FORBIDDEN_REPLACEMENTS:
                replacements.append(f"'{name}' → {FORBIDDEN_REPLACEMENTS[name]}")
    
    # Sprawdź pojedyncze słowa
    for word in FORBIDDEN_WORDS:
        if re.search(rf'\b{word}\b', text_lower, re.IGNORECASE):
            found_words.append(word)
            if word in FORBIDDEN_REPLACEMENTS:
                replacements.append(f"'{word}' → {FORBIDDEN_REPLACEMENTS[word]}")
    
    all_found = found_phrases + found_words
    
    if all_found:
        # 🔴 v33.0: BLOKUJ jeśli znaleziono zakazane frazy!
        status = Severity.CRITICAL
        message = f"🚫 ZAKAZANE FRAZY ({len(all_found)}×): {', '.join(all_found[:5])}"
        should_block = True
    else:
        status = Severity.OK
        message = "Brak zakazanych fraz ✓"
        should_block = False
    
    return {
        "status": status.value,
        "forbidden_found": all_found,
        "phrases": found_phrases,
        "words": found_words,
        "count": len(all_found),
        "message": message,
        "replacements": replacements,
        "should_block": should_block
    }


# ================================================================
# 🆕 CRITICAL: JITTER VALIDATION
# ================================================================
def validate_jitter(current_paragraphs: int, previous_paragraphs: int = None) -> Dict[str, Any]:
    if previous_paragraphs is None:
        return {
            "status": Severity.OK.value,
            "message": "Pierwszy batch - JITTER OK",
            "current": current_paragraphs,
            "previous": None
        }
    
    if current_paragraphs == previous_paragraphs:
        return {
            "status": Severity.WARNING.value,
            "message": f"JITTER fail: {current_paragraphs}ak = poprzedni ({previous_paragraphs}ak). Zmień liczbę akapitów!",
            "current": current_paragraphs,
            "previous": previous_paragraphs
        }
    
    return {
        "status": Severity.OK.value,
        "message": f"JITTER OK: {current_paragraphs}ak ≠ {previous_paragraphs}ak",
        "current": current_paragraphs,
        "previous": previous_paragraphs
    }


# ================================================================
# 🆕 CRITICAL: TRIPLETS VALIDATION
# ================================================================
def validate_triplets(text: str, s1_relationships: List[Dict]) -> Dict[str, Any]:
    if not s1_relationships:
        return {
            "status": Severity.OK.value,
            "message": "Brak tripletów z S1 do sprawdzenia",
            "found": 0,
            "expected": 0
        }
    
    text_lower = text.lower()
    found = []
    
    for rel in s1_relationships:
        subject = rel.get("subject", "").lower()
        predicate = rel.get("predicate", "").lower()
        obj = rel.get("object", "").lower()
        
        if subject and predicate and obj:
            if subject in text_lower and predicate in text_lower and obj in text_lower:
                found.append(rel)
    
    expected = min(3, len(s1_relationships))
    
    if len(found) >= 2:
        status = Severity.OK
        message = f"Znaleziono {len(found)} tripletów (min 2)"
    elif len(found) == 1:
        status = Severity.WARNING
        message = f"Tylko 1 triplet znaleziony (min 2)"
    else:
        status = Severity.WARNING
        message = f"Brak tripletów z S1 (min 2)"
    
    return {
        "status": status.value,
        "message": message,
        "found": len(found),
        "expected": expected,
        "triplets_found": found[:5]
    }


# ================================================================
# 🎯 FULL AI DETECTION (z CRITICAL validations)
# ================================================================
def full_ai_detection(
    text: str, 
    previous_paragraphs: int = None,
    s1_relationships: List[Dict] = None
) -> Dict[str, Any]:
    """
    Pełna analiza AI detection + walidacje CRITICAL.
    """
    humanness = calculate_humanness_score(text)
    forbidden = check_forbidden_phrases(text)
    
    current_paragraphs = len(re.split(r'\n\s*\n', text.strip()))
    jitter = validate_jitter(current_paragraphs, previous_paragraphs)
    
    triplets = validate_triplets(text, s1_relationships or [])
    
    statuses = [
        humanness["status"],
        forbidden["status"],
        jitter["status"],
        triplets["status"]
    ]
    
    if "CRITICAL" in statuses:
        overall_status = Severity.CRITICAL.value
    elif "WARNING" in statuses:
        overall_status = Severity.WARNING.value
    else:
        overall_status = Severity.OK.value
    
    all_warnings = humanness.get("warnings", [])
    if forbidden["status"] != "OK":
        all_warnings.append(forbidden["message"])
    if jitter["status"] != "OK":
        all_warnings.append(jitter["message"])
    if triplets["status"] != "OK":
        all_warnings.append(triplets["message"])
    
    return {
        "humanness_score": humanness["humanness_score"],
        "status": overall_status,
        "components": humanness["components"],
        "validations": {
            "forbidden_phrases": forbidden,
            "jitter": jitter,
            "triplets": triplets
        },
        "warnings": all_warnings[:7],
        "suggestions": humanness.get("suggestions", [])[:5]
    }


# ================================================================
# 🆕 FAZA 2: ENTITY SPLIT 60/40
# ================================================================
def calculate_entity_split(text: str, s1_entities: List[Dict]) -> Dict[str, Any]:
    """
    Oblicza proporcję Core vs Supporting entities.
    
    Cel: 60% Core, 40% Supporting
    """
    if not s1_entities:
        return {
            "status": "NO_DATA",
            "message": "Brak danych o encjach z S1",
            "core_ratio": 0,
            "supporting_ratio": 0
        }
    
    text_lower = text.lower()
    
    # Rozdziel encje na Core i Supporting
    core_entities = []
    supporting_entities = []
    
    for e in s1_entities:
        category = e.get("category", "").upper()
        importance = e.get("importance", 0.5)
        
        # Jeśli brak category, użyj importance do klasyfikacji
        if category == "CORE" or (not category and importance >= 0.6):
            core_entities.append(e)
        elif category == "SUPPORTING" or (not category and importance < 0.6):
            supporting_entities.append(e)
        else:
            # Domyślnie jako supporting
            supporting_entities.append(e)
    
    # Zlicz znalezione
    core_found = 0
    supporting_found = 0
    core_used = []
    supporting_used = []
    
    for e in core_entities:
        name = e.get("name", e.get("text", "")).lower()
        if name and name in text_lower:
            core_found += 1
            core_used.append(name)
    
    for e in supporting_entities:
        name = e.get("name", e.get("text", "")).lower()
        if name and name in text_lower:
            supporting_found += 1
            supporting_used.append(name)
    
    total_found = core_found + supporting_found
    
    if total_found == 0:
        return {
            "status": "WARNING",
            "message": "Nie znaleziono żadnych encji w tekście",
            "core_ratio": 0,
            "supporting_ratio": 0,
            "core_found": 0,
            "supporting_found": 0
        }
    
    core_ratio = core_found / total_found
    supporting_ratio = supporting_found / total_found
    
    # Status: OK jeśli core_ratio między 0.55 a 0.65
    if 0.55 <= core_ratio <= 0.65:
        status = "OK"
        message = f"Entity split OK: {core_ratio:.0%} core / {supporting_ratio:.0%} supporting"
    elif core_ratio > 0.65:
        status = "WARNING"
        message = f"Za dużo Core entities ({core_ratio:.0%}). Dodaj więcej Supporting (ubezpieczenie, certyfikaty, normy)"
    else:
        status = "WARNING"
        message = f"Za mało Core entities ({core_ratio:.0%}). Dodaj więcej Core (główne tematy)"
    
    return {
        "status": status,
        "message": message,
        "core_ratio": round(core_ratio, 2),
        "supporting_ratio": round(supporting_ratio, 2),
        "core_found": core_found,
        "supporting_found": supporting_found,
        "core_total": len(core_entities),
        "supporting_total": len(supporting_entities),
        "core_used": core_used[:10],
        "supporting_used": supporting_used[:10]
    }


# ================================================================
# 🆕 FAZA 2: TOPIC COMPLETENESS
# ================================================================
def calculate_topic_completeness(text: str, s1_topics: List[Dict]) -> Dict[str, Any]:
    """
    Oblicza pokrycie tematów z S1.
    """
    if not s1_topics:
        return {
            "status": "NO_DATA",
            "score": 0,
            "message": "Brak danych o tematach z S1"
        }
    
    text_lower = text.lower()
    
    # Rozdziel tematy według priorytetu
    must_topics = []
    high_topics = []
    medium_topics = []
    
    for t in s1_topics:
        priority = t.get("priority", "MEDIUM").upper()
        if priority == "MUST":
            must_topics.append(t)
        elif priority == "HIGH":
            high_topics.append(t)
        else:
            medium_topics.append(t)
    
    # Sprawdź pokrycie
    def check_topic_covered(topic):
        name = topic.get("name", topic.get("subtopic", "")).lower()
        keywords = topic.get("keywords", [])
        
        # Sprawdź nazwę
        if name and name in text_lower:
            return True
        
        # Sprawdź słowa kluczowe
        for kw in keywords:
            if kw.lower() in text_lower:
                return True
        
        # Sprawdź sample_h2 jeśli istnieje
        sample_h2 = topic.get("sample_h2", "").lower()
        if sample_h2:
            words = sample_h2.split()
            matches = sum(1 for w in words if len(w) > 3 and w in text_lower)
            if matches >= len(words) * 0.5:
                return True
        
        return False
    
    # Zlicz pokryte
    must_covered = [t for t in must_topics if check_topic_covered(t)]
    high_covered = [t for t in high_topics if check_topic_covered(t)]
    medium_covered = [t for t in medium_topics if check_topic_covered(t)]
    
    # Oblicz score (MUST ma najwyższą wagę)
    total_weight = len(must_topics) * 3 + len(high_topics) * 2 + len(medium_topics) * 1
    covered_weight = len(must_covered) * 3 + len(high_covered) * 2 + len(medium_covered) * 1
    
    score = covered_weight / total_weight if total_weight > 0 else 0
    score = round(score, 2)
    
    # Znajdź brakujące MUST i HIGH
    must_missing = [t.get("name", t.get("subtopic", "unknown")) for t in must_topics if t not in must_covered]
    high_missing = [t.get("name", t.get("subtopic", "unknown")) for t in high_topics if t not in high_covered]
    
    # Status
    if score >= 0.8:
        status = "OK"
        message = f"Dobre pokrycie tematów ({score:.0%})"
    elif score >= 0.6:
        status = "WARNING"
        message = f"Pokrycie tematów {score:.0%} - dodaj brakujące"
    else:
        status = "WARNING"
        message = f"Niskie pokrycie tematów ({score:.0%}) - pilnie uzupełnij!"
    
    return {
        "status": status,
        "score": score,
        "score_percent": round(score * 100, 1),
        "message": message,
        "must_covered": len(must_covered),
        "must_total": len(must_topics),
        "high_covered": len(high_covered),
        "high_total": len(high_topics),
        "must_missing": must_missing[:5],
        "high_missing": high_missing[:5]
    }


# ================================================================
# 🆕 FAZA 2: BATCH HISTORY TRACKING
# ================================================================
def analyze_batch_trend(batch_history: List[Dict]) -> Dict[str, Any]:
    """
    Analizuje trend metryk między batchami.
    """
    if not batch_history or len(batch_history) < 2:
        return {
            "trend": "insufficient_data",
            "message": "Za mało danych do analizy trendu"
        }
    
    # Pobierz ostatnie 3 batche (lub mniej jeśli brak)
    recent = batch_history[-3:]
    
    # Analizuj humanness score
    humanness_scores = [b.get("humanness_score", 0) for b in recent]
    
    # Oblicz trend
    if len(humanness_scores) >= 2:
        first_half = sum(humanness_scores[:len(humanness_scores)//2 + 1]) / (len(humanness_scores)//2 + 1)
        second_half = sum(humanness_scores[len(humanness_scores)//2:]) / (len(humanness_scores) - len(humanness_scores)//2)
        
        diff = second_half - first_half
        
        if diff > 5:
            trend = "improving"
            trend_message = f"📈 Trend rosnący (+{diff:.1f} punktów)"
        elif diff < -5:
            trend = "declining"
            trend_message = f"📉 Trend spadkowy ({diff:.1f} punktów)"
        else:
            trend = "stable"
            trend_message = "➡️ Trend stabilny"
    else:
        trend = "stable"
        trend_message = "➡️ Trend stabilny"
    
    # Średnie metryki
    avg_humanness = sum(humanness_scores) / len(humanness_scores) if humanness_scores else 0
    
    burstiness_scores = [b.get("burstiness", 0) for b in recent if b.get("burstiness")]
    avg_burstiness = sum(burstiness_scores) / len(burstiness_scores) if burstiness_scores else 0
    
    return {
        "trend": trend,
        "message": trend_message,
        "avg_humanness": round(avg_humanness, 1),
        "avg_burstiness": round(avg_burstiness, 2),
        "batches_analyzed": len(recent),
        "last_scores": humanness_scores
    }


def create_batch_record(
    batch_number: int,
    humanness_score: int,
    burstiness: float,
    paragraphs: int,
    entity_density: float = 0,
    topic_completeness: float = 0
) -> Dict[str, Any]:
    """
    Tworzy rekord batcha do historii.
    """
    return {
        "batch": batch_number,
        "humanness_score": humanness_score,
        "burstiness": round(burstiness, 2),
        "paragraphs": paragraphs,
        "entity_density": round(entity_density, 2),
        "topic_completeness": round(topic_completeness, 2)
    }


# ================================================================
# 🆕 FAZA 3: PER-SENTENCE SCORING
# ================================================================
AI_PATTERN_FLAGS = [
    (r'\bwarto\b', "warto"),
    (r'\bnależy\b', "należy"),
    (r'\bkluczowy\b', "kluczowy"),
    (r'\bkompleksowy\b', "kompleksowy"),
    (r'\binnowacyjny\b', "innowacyjny"),
    (r'\bprofesjonalny\b', "profesjonalny"),
    (r'\bwysokiej jakości\b', "wysokiej jakości"),
    (r'\bszeroki zakres\b', "szeroki zakres"),
    (r'\bw pełni\b', "w pełni"),
    (r'\bw szczególności\b', "w szczególności"),
]

GENERIC_STARTERS = [
    "firma oferuje",
    "firma zapewnia",
    "firma gwarantuje",
    "usługi obejmują",
    "klienci otrzymują",
    "warto wiedzieć",
    "należy pamiętać",
    "ważne jest",
]


def score_single_sentence(sentence: str) -> Dict[str, Any]:
    """
    Ocenia pojedyncze zdanie pod kątem AI-like patterns.
    """
    sentence_lower = sentence.lower().strip()
    words = sentence.split()
    word_count = len(words)
    
    # Flagi AI
    ai_flags = []
    for pattern, name in AI_PATTERN_FLAGS:
        if re.search(pattern, sentence_lower):
            ai_flags.append(name)
    
    # Sprawdź starter
    starter = ' '.join(words[:3]).lower() if len(words) >= 3 else sentence_lower
    generic_starter = any(gs in starter for gs in GENERIC_STARTERS)
    
    # Oblicz score zdania (0-100)
    score = 100
    
    # Kary
    if ai_flags:
        score -= len(ai_flags) * 15  # -15 za każdą flagę AI
    
    if generic_starter:
        score -= 20  # -20 za generyczny starter
    
    # Kara za zbyt równą długość (typowe dla AI: 12-18 słów)
    if 12 <= word_count <= 18:
        score -= 5  # lekka kara za "średnią" długość
    
    # Bonus za krótkie (<8) lub długie (>25) zdania
    if word_count < 8 or word_count > 25:
        score += 10
    
    score = max(0, min(100, score))
    
    # Status
    if score >= 70:
        status = "OK"
    elif score >= 50:
        status = "WARNING"
    else:
        status = "AI_LIKE"
    
    return {
        "text": sentence[:80] + ("..." if len(sentence) > 80 else ""),
        "word_count": word_count,
        "starter": starter,
        "score": score,
        "status": status,
        "ai_flags": ai_flags,
        "generic_starter": generic_starter
    }


def score_sentences(text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Ocenia wszystkie zdania w tekście.
    Zwraca posortowane od najgorszych.
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return {
            "status": "NO_DATA",
            "message": "Brak zdań do analizy",
            "sentences": []
        }
    
    scored = []
    for s in sentences:
        result = score_single_sentence(s)
        scored.append(result)
    
    # Sortuj od najgorszych (najniższy score)
    scored.sort(key=lambda x: x["score"])
    
    # Statystyki
    scores = [s["score"] for s in scored]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    ai_like_count = sum(1 for s in scored if s["status"] == "AI_LIKE")
    warning_count = sum(1 for s in scored if s["status"] == "WARNING")
    ok_count = sum(1 for s in scored if s["status"] == "OK")
    
    # Status ogólny
    if ai_like_count >= 3:
        overall_status = "CRITICAL"
        message = f"Znaleziono {ai_like_count} zdań wyglądających na AI. Przepisz je!"
    elif ai_like_count >= 1 or warning_count >= 5:
        overall_status = "WARNING"
        message = f"Znaleziono {ai_like_count} AI-like i {warning_count} warning zdań"
    else:
        overall_status = "OK"
        message = "Zdania wyglądają naturalnie"
    
    # Sugestie poprawy dla najgorszych zdań
    suggestions = []
    for s in scored[:5]:  # Top 5 najgorszych
        if s["status"] in ["AI_LIKE", "WARNING"]:
            if s["ai_flags"]:
                suggestions.append(f"Zdanie '{s['text'][:40]}...' - usuń: {', '.join(s['ai_flags'][:2])}")
            elif s["generic_starter"]:
                suggestions.append(f"Zdanie '{s['text'][:40]}...' - zmień starter")
    
    return {
        "status": overall_status,
        "message": message,
        "total_sentences": len(sentences),
        "avg_score": round(avg_score, 1),
        "ai_like_count": ai_like_count,
        "warning_count": warning_count,
        "ok_count": ok_count,
        "worst_sentences": scored[:limit],
        "suggestions": suggestions[:5]
    }


# ================================================================
# 🆕 FAZA 3: N-GRAM NATURALNESS CHECK (z wordfreq)
# ================================================================

# Znane nienaturalne frazy AI (blacklist)
AI_BLACKLIST_NGRAMS = [
    "kluczowy aspekt",
    "holistyczne podejście", 
    "innowacyjne rozwiązanie",
    "strategiczne znaczenie",
    "fundamentalne znaczenie",
    "nie ulega wątpliwości",
    "warto zauważyć że",
    "należy podkreślić że",
    "kompleksowe rozwiązanie",
    "szeroki zakres usług",
    "indywidualne podejście",
    "wysoki standard",
    "pełen profesjonalizm",
    "bogaty doświadczenie",
    "dynamicznie rozwijający",
]

# Nadużywane frazy SEO (nie błąd, ale za często = AI)
OVERUSED_SEO_PHRASES = [
    "firma oferuje",
    "profesjonalne usługi", 
    "wysoka jakość",
    "kompleksowa obsługa",
    "konkurencyjne ceny",
    "doświadczony zespół",
    "wieloletnie doświadczenie",
    "szeroka oferta",
    "najwyższa jakość",
]


def get_word_frequency(word: str) -> float:
    """
    Zwraca częstość słowa (skala Zipf 0-7).
    Jeśli wordfreq niedostępny, zwraca domyślną wartość.
    """
    if not WORDFREQ_AVAILABLE:
        return 4.0  # średnia domyślna
    
    try:
        freq = zipf_frequency(word, 'pl')
        return freq if freq > 0 else 1.0  # nieznane słowa = rzadkie
    except:
        return 4.0


def calculate_ngram_frequency(ngram: str) -> Dict[str, Any]:
    """
    Oblicza średnią częstość n-gramu na podstawie częstości słów.
    """
    words = ngram.lower().split()
    if not words:
        return {"ngram": ngram, "avg_freq": 0, "min_freq": 0}
    
    freqs = [get_word_frequency(w) for w in words if w not in POLISH_STOP_WORDS]
    
    if not freqs:
        # Wszystkie słowa to stop words - wysoka częstość
        return {"ngram": ngram, "avg_freq": 6.0, "min_freq": 6.0}
    
    return {
        "ngram": ngram,
        "avg_freq": round(sum(freqs) / len(freqs), 2),
        "min_freq": round(min(freqs), 2),
        "word_count": len(words)
    }


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Wyciąga n-gramy z tekstu.
    """
    # Usuń HTML i normalizuj
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def check_ngram_naturalness(text: str) -> Dict[str, Any]:
    """
    Sprawdza naturalność fraz w tekście używając wordfreq.
    
    Metoda:
    1. Wyciąga bigramy i trigramy
    2. Oblicza częstość każdego n-gramu (średnia Zipf słów)
    3. Identyfikuje rzadkie/nienaturalne frazy
    4. Sprawdza blacklistę AI
    5. Sprawdza nadużywane frazy SEO
    """
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) < 50:
        return {
            "status": "NO_DATA",
            "message": "Za mało tekstu do analizy n-gramów",
            "wordfreq_available": WORDFREQ_AVAILABLE
        }
    
    # 1. Sprawdź blacklistę AI
    ai_phrases_found = []
    for phrase in AI_BLACKLIST_NGRAMS:
        count = text_lower.count(phrase)
        if count > 0:
            ai_phrases_found.append({"phrase": phrase, "count": count})
    
    # 2. Sprawdź nadużywane frazy SEO
    overused_found = []
    for phrase in OVERUSED_SEO_PHRASES:
        count = text_lower.count(phrase)
        if count >= 2:  # 2+ = nadużywane
            overused_found.append({"phrase": phrase, "count": count})
    
    # 3. Wyciągnij i przeanalizuj bigramy (jeśli wordfreq dostępny)
    unusual_ngrams = []
    low_freq_ngrams = []
    
    if WORDFREQ_AVAILABLE:
        bigrams = extract_ngrams(text, n=2)
        
        # Zlicz bigramy
        bigram_counts = Counter(bigrams)
        
        # Analizuj najczęstsze bigramy (potencjalnie nadużywane)
        for bigram, count in bigram_counts.most_common(30):
            if count >= 3:  # Powtórzone 3+ razy
                freq_data = calculate_ngram_frequency(bigram)
                
                # Jeśli niska średnia częstość = dziwna fraza
                if freq_data["avg_freq"] < 3.5:
                    unusual_ngrams.append({
                        "ngram": bigram,
                        "count": count,
                        "avg_freq": freq_data["avg_freq"],
                        "reason": "low_frequency"
                    })
                # Jeśli wysoka częstość ale dużo powtórzeń = nadużywane
                elif count >= 5:
                    unusual_ngrams.append({
                        "ngram": bigram,
                        "count": count,
                        "avg_freq": freq_data["avg_freq"],
                        "reason": "overused"
                    })
        
        # Znajdź ogólnie rzadkie bigramy (min_freq < 2.5)
        unique_bigrams = list(set(bigrams))[:100]  # Sprawdź max 100
        for bigram in unique_bigrams:
            freq_data = calculate_ngram_frequency(bigram)
            if freq_data["min_freq"] < 2.0 and freq_data["min_freq"] > 0:
                low_freq_ngrams.append({
                    "ngram": bigram,
                    "min_freq": freq_data["min_freq"],
                    "avg_freq": freq_data["avg_freq"]
                })
        
        # Sortuj po częstości (najrzadsze najpierw)
        low_freq_ngrams.sort(key=lambda x: x["min_freq"])
        low_freq_ngrams = low_freq_ngrams[:10]
    
    # 4. Oblicz naturalness score
    penalty = 0
    
    # Kary za AI phrases (największa kara)
    penalty += len(ai_phrases_found) * 0.15
    
    # Kary za nadużywane SEO
    penalty += len(overused_found) * 0.08
    
    # Kary za unusual ngrams
    penalty += len(unusual_ngrams) * 0.05
    
    # Kary za low freq ngrams
    penalty += min(len(low_freq_ngrams) * 0.03, 0.15)
    
    naturalness_score = max(0, 1.0 - penalty)
    naturalness_score = round(naturalness_score, 2)
    
    # 5. Status
    if naturalness_score >= 0.75:
        status = "OK"
        message = f"Frazy brzmią naturalnie (score {naturalness_score})"
    elif naturalness_score >= 0.5:
        status = "WARNING"
        message = f"Niektóre frazy wymagają poprawy (score {naturalness_score})"
    else:
        status = "CRITICAL"
        message = f"Wiele fraz brzmi nienaturalnie/AI (score {naturalness_score})"
    
    # 6. Sugestie
    suggestions = []
    
    # Sugestie dla AI phrases (priorytet)
    for item in ai_phrases_found[:3]:
        suggestions.append(f"❌ Usuń AI-frazę: '{item['phrase']}'")
    
    # Sugestie dla nadużywanych
    for item in overused_found[:2]:
        suggestions.append(f"⚠️ Ogranicz '{item['phrase']}' (użyte {item['count']}×)")
    
    # Sugestie dla unusual
    for item in unusual_ngrams[:2]:
        if item["reason"] == "overused":
            suggestions.append(f"📝 Zmniejsz powtórzenia: '{item['ngram']}' ({item['count']}×)")
        else:
            suggestions.append(f"📝 Sprawdź frazę: '{item['ngram']}' (rzadka)")
    
    return {
        "status": status,
        "message": message,
        "naturalness_score": naturalness_score,
        "wordfreq_available": WORDFREQ_AVAILABLE,
        
        # Szczegóły
        "ai_phrases_found": ai_phrases_found[:5],
        "overused_seo_phrases": overused_found[:5],
        "unusual_ngrams": unusual_ngrams[:5],
        "low_frequency_ngrams": low_freq_ngrams[:5],
        
        # Statystyki
        "stats": {
            "ai_phrases_count": len(ai_phrases_found),
            "overused_count": len(overused_found),
            "unusual_count": len(unusual_ngrams)
        },
        
        "suggestions": suggestions[:7]
    }


# ================================================================
# 🎯 FAZA 3: FULL ADVANCED ANALYSIS
# ================================================================
def full_advanced_analysis(
    text: str,
    previous_paragraphs: int = None,
    s1_relationships: List[Dict] = None,
    s1_entities: List[Dict] = None,
    s1_topics: List[Dict] = None
) -> Dict[str, Any]:
    """
    Pełna zaawansowana analiza tekstu - wszystkie metryki.
    """
    # Podstawowa analiza AI
    humanness = calculate_humanness_score(text)
    forbidden = check_forbidden_phrases(text)
    
    # Walidacje
    current_paragraphs = len(re.split(r'\n\s*\n', text.strip()))
    jitter = validate_jitter(current_paragraphs, previous_paragraphs)
    triplets = validate_triplets(text, s1_relationships or [])
    
    # Faza 2
    entity_split = calculate_entity_split(text, s1_entities or [])
    topic_completeness = calculate_topic_completeness(text, s1_topics or [])
    
    # Faza 3
    sentence_analysis = score_sentences(text, limit=10)
    ngram_analysis = check_ngram_naturalness(text)
    
    # Łączny status
    statuses = [
        humanness["status"],
        forbidden["status"],
        sentence_analysis["status"],
        ngram_analysis["status"]
    ]
    
    if "CRITICAL" in statuses:
        overall_status = "CRITICAL"
    elif statuses.count("WARNING") >= 2:
        overall_status = "WARNING"
    elif "WARNING" in statuses:
        overall_status = "OK"  # pojedynczy warning OK
    else:
        overall_status = "OK"
    
    # Zbierz wszystkie sugestie
    all_suggestions = []
    all_suggestions.extend(humanness.get("suggestions", []))
    all_suggestions.extend(sentence_analysis.get("suggestions", []))
    all_suggestions.extend(ngram_analysis.get("suggestions", []))
    
    # Zbierz wszystkie warnings
    all_warnings = humanness.get("warnings", [])
    if forbidden["status"] != "OK":
        all_warnings.append(forbidden["message"])
    if sentence_analysis["status"] != "OK":
        all_warnings.append(sentence_analysis["message"])
    if ngram_analysis["status"] != "OK":
        all_warnings.append(ngram_analysis["message"])
    
    return {
        "overall_status": overall_status,
        "humanness_score": humanness["humanness_score"],
        
        # Podstawowe metryki
        "components": humanness["components"],
        
        # Walidacje (Faza 1)
        "validations": {
            "forbidden_phrases": forbidden,
            "jitter": jitter,
            "triplets": triplets
        },
        
        # Faza 2
        "entity_split": entity_split,
        "topic_completeness": topic_completeness,
        
        # Faza 3
        "sentence_analysis": {
            "status": sentence_analysis["status"],
            "avg_score": sentence_analysis["avg_score"],
            "ai_like_count": sentence_analysis["ai_like_count"],
            "worst_sentences": sentence_analysis["worst_sentences"][:5]
        },
        "ngram_analysis": ngram_analysis,
        
        # Podsumowanie
        "warnings": all_warnings[:10],
        "suggestions": all_suggestions[:10]
    }
