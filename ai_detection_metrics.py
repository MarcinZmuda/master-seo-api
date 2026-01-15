"""
===============================================================================
🔍 AI DETECTION METRICS v1.0 for BRAJEN
===============================================================================
Metryki proxy dla wykrywalności tekstu AI.

Wymaga: pip install wordfreq

Metryki:
- Burstiness (variance długości zdań) - już w BRAJEN, tu ulepszony
- Vocabulary Richness (TTR)
- Lexical Sophistication (via wordfreq)
- Sentence Starter Entropy
- Word Repetition Detection

Progi dwupoziomowe:
- CRITICAL: tekst prawie na pewno AI
- WARNING: tekst podejrzany, wymaga poprawek
===============================================================================
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# WORDFREQ INTEGRATION
# ============================================================================
try:
    from wordfreq import word_frequency, zipf_frequency
    WORDFREQ_AVAILABLE = True
    print("[AI_DETECTION] ✅ wordfreq loaded - using real frequency data")
except ImportError:
    WORDFREQ_AVAILABLE = False
    print("[AI_DETECTION] ⚠️ wordfreq not installed - using fallback method")
    print("[AI_DETECTION] 💡 Install with: pip install wordfreq")


# ============================================================================
# KONFIGURACJA - PROGI DWUPOZIOMOWE
# ============================================================================
@dataclass
class AIDetectionConfig:
    """Progi dla wykrywania AI - dwupoziomowe (CRITICAL/WARNING)"""
    
    # BURSTINESS (variance długości zdań)
    # AI pisze monotonnie, ludzie zmieniają rytm
    BURSTINESS_CRITICAL_LOW: float = 2.0    # Poniżej = CRITICAL
    BURSTINESS_WARNING_LOW: float = 2.8     # Poniżej = WARNING  
    BURSTINESS_OPTIMAL_MIN: float = 3.2     # Optimum start
    BURSTINESS_OPTIMAL_MAX: float = 3.8     # Optimum end
    BURSTINESS_WARNING_HIGH: float = 4.2    # Powyżej = WARNING
    BURSTINESS_CRITICAL_HIGH: float = 4.8   # Powyżej = CRITICAL
    
    # VOCABULARY RICHNESS (Type-Token Ratio)
    # AI używa mniejszego zasobu słów
    TTR_CRITICAL_LOW: float = 0.40          # Poniżej = CRITICAL
    TTR_WARNING_LOW: float = 0.48           # Poniżej = WARNING
    TTR_OPTIMAL: float = 0.55               # Cel
    
    # LEXICAL SOPHISTICATION (wordfreq)
    # AI preferuje częste, "bezpieczne" słowa
    # Mierzymy jako średni Zipf score - im niższy, tym rzadsze słowa
    ZIPF_CRITICAL_HIGH: float = 5.5         # Powyżej = CRITICAL (za proste słowa)
    ZIPF_WARNING_HIGH: float = 5.0          # Powyżej = WARNING
    ZIPF_OPTIMAL: float = 4.2               # Cel (mix częstych i rzadkich)
    
    # SENTENCE STARTER ENTROPY
    # AI zaczyna zdania podobnie (podmiot + czasownik)
    ENTROPY_CRITICAL_LOW: float = 0.50      # Poniżej = CRITICAL
    ENTROPY_WARNING_LOW: float = 0.65       # Poniżej = WARNING
    ENTROPY_OPTIMAL: float = 0.80           # Cel
    
    # WORD REPETITION
    # Ile razy to samo słowo (poza stop words) może się powtórzyć
    REPETITION_WARNING: int = 5             # Powyżej = WARNING
    REPETITION_CRITICAL: int = 8            # Powyżej = CRITICAL


CONFIG = AIDetectionConfig()


# ============================================================================
# POLSKIE STOP WORDS (nie liczą się do repetition/sophistication)
# ============================================================================
POLISH_STOP_WORDS = {
    # Spójniki
    "i", "a", "oraz", "ale", "lecz", "jednak", "jednakże", "natomiast",
    "lub", "albo", "czy", "ani", "więc", "zatem", "dlatego", "bo",
    "ponieważ", "gdyż", "że", "żeby", "aby", "jeśli", "jeżeli", "gdy",
    "kiedy", "jak", "choć", "chociaż", "mimo", "czyli", "to",
    
    # Przyimki
    "w", "we", "z", "ze", "do", "od", "na", "po", "za", "przed", "pod",
    "nad", "między", "przez", "dla", "bez", "u", "o", "przy", "ku",
    
    # Zaimki
    "ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one",
    "ten", "ta", "to", "ci", "te", "który", "która", "które",
    "co", "kto", "jaki", "jaka", "jakie", "każdy", "wszystko",
    "się", "siebie", "sobie", "mnie", "mi", "cię", "go", "jej", "ich",
    
    # Partykuły i przysłówki
    "nie", "tak", "już", "jeszcze", "też", "także", "również", "nawet",
    "tylko", "bardzo", "bardziej", "najbardziej", "więcej", "mniej",
    "teraz", "potem", "tam", "tu", "tutaj", "gdzie", "zawsze", "nigdy",
    
    # Czasowniki posiłkowe
    "być", "jest", "są", "był", "była", "było", "będzie",
    "mieć", "ma", "mają", "miał", "można", "trzeba",
}


# ============================================================================
# METRYKI
# ============================================================================

def calculate_burstiness_v2(text: str) -> Dict:
    """
    Ulepszona wersja burstiness z dwupoziomowymi progami.
    
    Zwraca:
    - value: surowa wartość burstiness
    - status: OK / WARNING / CRITICAL
    - details: szczegóły (min/max/avg długość zdania)
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
    
    if len(sentences) < 5:
        return {
            "value": 3.5,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało zdań do analizy (min 5)",
            "details": {"sentence_count": len(sentences)}
        }
    
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    
    # Coefficient of variation * 5 (skalowanie do ~1-5)
    raw_burstiness = (std_dev / mean) * 5 if mean > 0 else 0
    burstiness = round(raw_burstiness, 2)
    
    # Dwupoziomowa ocena
    if burstiness < CONFIG.BURSTINESS_CRITICAL_LOW:
        status = "CRITICAL"
        message = f"Zdania zbyt monotonne ({burstiness:.2f}) - silny sygnał AI"
    elif burstiness < CONFIG.BURSTINESS_WARNING_LOW:
        status = "WARNING"
        message = f"Zdania mało zróżnicowane ({burstiness:.2f}) - dodaj krótkie zdania"
    elif burstiness > CONFIG.BURSTINESS_CRITICAL_HIGH:
        status = "CRITICAL"
        message = f"Zdania zbyt chaotyczne ({burstiness:.2f}) - tekst nieczytelny"
    elif burstiness > CONFIG.BURSTINESS_WARNING_HIGH:
        status = "WARNING"
        message = f"Zdania zbyt zróżnicowane ({burstiness:.2f}) - wyrównaj rytm"
    else:
        status = "OK"
        message = f"Burstiness w normie ({burstiness:.2f})"
    
    return {
        "value": burstiness,
        "status": status,
        "message": message,
        "details": {
            "sentence_count": len(sentences),
            "avg_length": round(mean, 1),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_dev": round(std_dev, 2),
            "lengths_sample": lengths[:10]
        }
    }


def calculate_vocabulary_richness(text: str) -> Dict:
    """
    Type-Token Ratio (TTR) - stosunek unikalnych słów do wszystkich.
    
    AI: < 0.45 (powtarza te same słowa)
    Human: > 0.55 (bogaty zasób słownictwa)
    """
    words = re.findall(r'\b[a-ząćęłńóśźż]+\b', text.lower())
    
    if len(words) < 100:
        return {
            "value": 0.50,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało słów do analizy TTR (min 100)",
            "details": {"word_count": len(words)}
        }
    
    unique = len(set(words))
    ttr = unique / len(words)
    ttr = round(ttr, 3)
    
    if ttr < CONFIG.TTR_CRITICAL_LOW:
        status = "CRITICAL"
        message = f"Bardzo ubogi zasób słów ({ttr:.3f}) - silny sygnał AI"
    elif ttr < CONFIG.TTR_WARNING_LOW:
        status = "WARNING"
        message = f"Mało zróżnicowane słownictwo ({ttr:.3f}) - użyj synonimów"
    else:
        status = "OK"
        message = f"Vocabulary richness OK ({ttr:.3f})"
    
    return {
        "value": ttr,
        "status": status,
        "message": message,
        "details": {
            "total_words": len(words),
            "unique_words": unique
        }
    }


def calculate_lexical_sophistication(text: str) -> Dict:
    """
    Mierzy czy tekst używa "prostych" czy "zaawansowanych" słów.
    Używa wordfreq do sprawdzenia częstości każdego słowa.
    
    Zipf scale: 1-7 gdzie 7 = najczęstsze (np. "i", "to")
    AI preferuje słowa o wysokim Zipf (częste, "bezpieczne")
    Ludzie mieszają częste z rzadkimi
    """
    if not WORDFREQ_AVAILABLE:
        return {
            "value": 4.5,
            "status": "UNAVAILABLE",
            "message": "wordfreq nie zainstalowany - zainstaluj: pip install wordfreq",
            "details": {}
        }
    
    words = re.findall(r'\b[a-ząćęłńóśźż]{4,}\b', text.lower())
    # Filtruj stop words
    content_words = [w for w in words if w not in POLISH_STOP_WORDS]
    
    if len(content_words) < 50:
        return {
            "value": 4.5,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało słów do analizy (min 50 content words)",
            "details": {"content_word_count": len(content_words)}
        }
    
    # Oblicz średni Zipf frequency
    zipf_scores = []
    rare_words = []
    common_words = []
    
    for word in content_words:
        zipf = zipf_frequency(word, 'pl')
        zipf_scores.append(zipf)
        
        if zipf < 3.0:  # Rzadkie słowo
            rare_words.append((word, zipf))
        elif zipf > 5.5:  # Bardzo częste słowo
            common_words.append((word, zipf))
    
    avg_zipf = sum(zipf_scores) / len(zipf_scores)
    avg_zipf = round(avg_zipf, 2)
    
    # Procent rzadkich słów (Zipf < 3.0)
    rare_ratio = len(rare_words) / len(content_words)
    
    if avg_zipf > CONFIG.ZIPF_CRITICAL_HIGH:
        status = "CRITICAL"
        message = f"Zbyt proste słownictwo (avg Zipf {avg_zipf:.2f}) - silny sygnał AI"
    elif avg_zipf > CONFIG.ZIPF_WARNING_HIGH:
        status = "WARNING"
        message = f"Słownictwo zbyt podstawowe (avg Zipf {avg_zipf:.2f}) - dodaj specjalistyczne terminy"
    else:
        status = "OK"
        message = f"Lexical sophistication OK (avg Zipf {avg_zipf:.2f})"
    
    return {
        "value": avg_zipf,
        "status": status,
        "message": message,
        "details": {
            "content_words": len(content_words),
            "rare_words_count": len(rare_words),
            "rare_ratio": round(rare_ratio, 3),
            "sample_rare": sorted(set([w for w, z in rare_words]))[:10],
            "sample_overused_common": sorted(set([w for w, z in common_words]))[:10]
        }
    }


def calculate_starter_entropy(text: str) -> Dict:
    """
    Mierzy różnorodność początków zdań.
    
    AI często zaczyna zdania podobnie (podmiot + czasownik).
    Ludzie zaczynają od okoliczników, pytań, spójników.
    
    Entropia Shannona znormalizowana do 0-1.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
    
    if len(sentences) < 8:
        return {
            "value": 0.70,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało zdań do analizy (min 8)",
            "details": {"sentence_count": len(sentences)}
        }
    
    # Pobierz pierwsze słowo każdego zdania
    starters = []
    for s in sentences:
        words = s.split()
        if words:
            # Normalizuj (lowercase, usuń interpunkcję)
            starter = re.sub(r'[^\w]', '', words[0].lower())
            if starter:
                starters.append(starter)
    
    if len(starters) < 5:
        return {
            "value": 0.70,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało starterów do analizy",
            "details": {}
        }
    
    # Entropia Shannona
    counts = Counter(starters)
    total = len(starters)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    
    # Normalizacja do 0-1 (max entropia gdy wszystkie różne)
    max_entropy = math.log2(total) if total > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    normalized_entropy = round(normalized_entropy, 3)
    
    # Znajdź najczęstsze startery
    most_common = counts.most_common(5)
    overused = [(word, count, round(count/total*100, 1)) 
                for word, count in most_common if count >= 3]
    
    if normalized_entropy < CONFIG.ENTROPY_CRITICAL_LOW:
        status = "CRITICAL"
        message = f"Zdania zaczynają się bardzo podobnie ({normalized_entropy:.2f}) - silny sygnał AI"
    elif normalized_entropy < CONFIG.ENTROPY_WARNING_LOW:
        status = "WARNING"
        message = f"Mało zróżnicowane początki zdań ({normalized_entropy:.2f})"
    else:
        status = "OK"
        message = f"Starter entropy OK ({normalized_entropy:.2f})"
    
    return {
        "value": normalized_entropy,
        "status": status,
        "message": message,
        "details": {
            "sentence_count": len(sentences),
            "unique_starters": len(counts),
            "overused_starters": overused,
            "raw_entropy": round(entropy, 3)
        }
    }


def calculate_word_repetition(text: str) -> Dict:
    """
    Wykrywa nadmierne powtarzanie tych samych słów.
    
    Nie liczy stop words ani krótkich słów (< 4 litery).
    Zwraca listę słów użytych > 5 razy.
    """
    words = re.findall(r'\b[a-ząćęłńóśźż]{4,}\b', text.lower())
    content_words = [w for w in words if w not in POLISH_STOP_WORDS]
    
    if len(content_words) < 50:
        return {
            "value": 100,
            "status": "INSUFFICIENT_DATA",
            "message": "Za mało słów do analizy",
            "details": {}
        }
    
    counts = Counter(content_words)
    
    # Znajdź nadużywane słowa
    critical_overused = [(w, c) for w, c in counts.most_common(20) 
                         if c >= CONFIG.REPETITION_CRITICAL]
    warning_overused = [(w, c) for w, c in counts.most_common(20) 
                        if CONFIG.REPETITION_WARNING <= c < CONFIG.REPETITION_CRITICAL]
    
    # Score: 100 - (penalty za każde nadużyte słowo)
    score = 100 - (len(critical_overused) * 15) - (len(warning_overused) * 5)
    score = max(0, score)
    
    if critical_overused:
        status = "CRITICAL"
        words_list = ", ".join([f"'{w}' ({c}×)" for w, c in critical_overused[:3]])
        message = f"Silne powtórzenia: {words_list}"
    elif warning_overused:
        status = "WARNING"
        words_list = ", ".join([f"'{w}' ({c}×)" for w, c in warning_overused[:3]])
        message = f"Powtórzenia: {words_list} - rozważ synonimy"
    else:
        status = "OK"
        message = "Brak nadmiernych powtórzeń"
    
    return {
        "value": score,
        "status": status,
        "message": message,
        "details": {
            "critical_overused": critical_overused,
            "warning_overused": warning_overused,
            "total_content_words": len(content_words),
            "unique_content_words": len(counts)
        }
    }


# ============================================================================
# GŁÓWNA FUNKCJA WALIDACJI
# ============================================================================

def validate_ai_detection(text: str) -> Dict:
    """
    Główna funkcja - kompletna analiza wykrywalności AI.
    
    Zwraca:
    - humanness_score: 0-100 (im wyższy, tym mniej wykrywalny jako AI)
    - status: OK / WARNING / CRITICAL
    - components: szczegóły każdej metryki
    - warnings: lista problemów do poprawienia
    - suggestions: konkretne sugestie
    """
    
    # Oblicz wszystkie metryki
    burstiness = calculate_burstiness_v2(text)
    vocabulary = calculate_vocabulary_richness(text)
    sophistication = calculate_lexical_sophistication(text)
    entropy = calculate_starter_entropy(text)
    repetition = calculate_word_repetition(text)
    
    # Normalizuj do 0-100
    def status_to_score(result: Dict, optimal_value: float = None) -> float:
        """Konwertuje status na score 0-100"""
        status = result.get("status", "OK")
        if status == "CRITICAL":
            return 20
        elif status == "WARNING":
            return 50
        elif status == "INSUFFICIENT_DATA":
            return 60
        elif status == "UNAVAILABLE":
            return 70  # Nie penalizuj za brak wordfreq
        else:
            return 85
    
    scores = {
        "burstiness": status_to_score(burstiness),
        "vocabulary": status_to_score(vocabulary),
        "sophistication": status_to_score(sophistication),
        "entropy": status_to_score(entropy),
        "repetition": repetition["value"]  # już jest 0-100
    }
    
    # Ważona suma
    weights = {
        "burstiness": 0.25,
        "vocabulary": 0.20,
        "sophistication": 0.15,
        "entropy": 0.20,
        "repetition": 0.20
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    final_score = round(final_score)
    
    # Zbierz ostrzeżenia
    warnings = []
    for name, result in [("Burstiness", burstiness), ("Vocabulary", vocabulary),
                         ("Sophistication", sophistication), ("Entropy", entropy),
                         ("Repetition", repetition)]:
        if result["status"] in ["CRITICAL", "WARNING"]:
            warnings.append(result["message"])
    
    # Status końcowy
    critical_count = sum(1 for r in [burstiness, vocabulary, sophistication, entropy, repetition]
                        if r["status"] == "CRITICAL")
    warning_count = sum(1 for r in [burstiness, vocabulary, sophistication, entropy, repetition]
                       if r["status"] == "WARNING")
    
    if critical_count >= 2 or final_score < 40:
        overall_status = "CRITICAL"
    elif critical_count >= 1 or warning_count >= 2 or final_score < 60:
        overall_status = "WARNING"
    else:
        overall_status = "OK"
    
    # Sugestie
    suggestions = generate_improvement_suggestions(burstiness, vocabulary, sophistication, entropy, repetition)
    
    return {
        "humanness_score": final_score,
        "status": overall_status,
        "components": {
            "burstiness": burstiness,
            "vocabulary_richness": vocabulary,
            "lexical_sophistication": sophistication,
            "starter_entropy": entropy,
            "word_repetition": repetition
        },
        "scores": scores,
        "warnings": warnings,
        "suggestions": suggestions,
        "summary": f"Humanness: {final_score}/100 | Status: {overall_status} | Issues: {len(warnings)}"
    }


def generate_improvement_suggestions(burstiness, vocabulary, sophistication, entropy, repetition) -> List[str]:
    """Generuje konkretne sugestie poprawy."""
    suggestions = []
    
    if burstiness["status"] in ["CRITICAL", "WARNING"]:
        if burstiness["value"] < CONFIG.BURSTINESS_WARNING_LOW:
            suggestions.append("🔹 Dodaj krótkie zdania (5-8 słów) po długich zdaniach")
            suggestions.append("🔹 Użyj pytań retorycznych dla urozmaicenia rytmu")
        else:
            suggestions.append("🔹 Wyrównaj długości zdań - niektóre są zbyt krótkie")
    
    if entropy["status"] in ["CRITICAL", "WARNING"]:
        suggestions.append("🔹 Zacznij niektóre zdania od okolicznika czasu (Wczoraj..., W 2024 roku...)")
        suggestions.append("🔹 Zacznij zdanie od okolicznika miejsca (W Warszawie..., Na rynku...)")
        suggestions.append("🔹 Użyj pytania na początku akapitu")
    
    if repetition["status"] in ["CRITICAL", "WARNING"]:
        overused = repetition["details"].get("critical_overused", []) + repetition["details"].get("warning_overused", [])
        for word, count in overused[:3]:
            suggestions.append(f"🔹 Znajdź synonim dla '{word}' (użyte {count}×)")
    
    if sophistication["status"] in ["CRITICAL", "WARNING"] and WORDFREQ_AVAILABLE:
        suggestions.append("🔹 Dodaj specjalistyczne/branżowe terminy zamiast ogólnych słów")
        common = sophistication["details"].get("sample_overused_common", [])
        if common:
            suggestions.append(f"🔹 Ogranicz bardzo częste słowa: {', '.join(common[:5])}")
    
    return suggestions


# ============================================================================
# ENDPOINT HELPER - do użycia w Flask
# ============================================================================

def create_ai_detection_response(text: str) -> Tuple[Dict, int]:
    """
    Helper do użycia jako endpoint Flask.
    
    Użycie w master_api.py:
    
    @app.post("/api/ai_detection")
    def ai_detection_endpoint():
        data = request.get_json()
        text = data.get("text", "")
        return create_ai_detection_response(text)
    """
    if not text or len(text) < 200:
        return {
            "status": "ERROR",
            "message": "Tekst za krótki (min 200 znaków)"
        }, 400
    
    result = validate_ai_detection(text)
    return result, 200


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    # Test z przykładowym tekstem
    test_text = """
    Firma przeprowadzkowa oferuje kompleksowe usługi transportowe. Profesjonalny zespół 
    zajmuje się pakowaniem i przewozem mebli. Każde zlecenie jest realizowane terminowo.
    
    Przeprowadzki mieszkań wymagają odpowiedniego przygotowania. Klienci często pytają 
    o cennik usług. Warto wcześniej zamówić wycenę. Ekipa przyjeżdża punktualnie.
    
    Transport mebli to nasza specjalność. Oferujemy również usługi magazynowania.
    Współpracujemy z klientami indywidualnymi i firmami. Gwarantujemy bezpieczeństwo 
    przewożonych rzeczy. Posiadamy odpowiednie ubezpieczenie OC.
    """
    
    result = validate_ai_detection(test_text)
    
    print("\n" + "="*60)
    print("🔍 AI DETECTION RESULTS")
    print("="*60)
    print(f"\n📊 HUMANNESS SCORE: {result['humanness_score']}/100")
    print(f"📌 STATUS: {result['status']}")
    print(f"\n⚠️ WARNINGS ({len(result['warnings'])}):")
    for w in result['warnings']:
        print(f"   • {w}")
    print(f"\n💡 SUGGESTIONS:")
    for s in result['suggestions']:
        print(f"   {s}")
    print("\n" + "="*60)
