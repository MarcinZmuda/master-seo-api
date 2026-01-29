"""
===============================================================================
📊 BATCH COMPLEXITY CALCULATOR v28.1
===============================================================================
Hybrydowy system obliczania długości i liczby akapitów dla batchy.

Czynniki uwzględniane:
1. Typ H2 (pytające vs informacyjne) - bazowy profil
2. N-gramy powiązane z H2 - modyfikator złożoności
3. Encje do zdefiniowania - każda wymaga miejsca
4. Keywords do wplecenia - minimum dla zachowania density
5. PAA match - bonus za snippet potential

Score 0-100 → mapowany na profil: short/medium/long/extended
===============================================================================
"""

import re
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict


# ================================================================
# 📋 PROFILE DŁUGOŚCI
# ================================================================

@dataclass
class LengthProfile:
    """Profil długości batcha."""
    name: str
    paragraphs_min: int
    paragraphs_max: int
    words_min: int
    words_max: int
    snippet_required: bool
    description: str


PROFILES = {
    "short": LengthProfile(
        name="short",
        paragraphs_min=2,
        paragraphs_max=3,
        words_min=300,     # 🆕 v41.2: +100 (było 200)
        words_max=450,     # 🆕 v41.2: +100 (było 350)
        snippet_required=False,
        description="Zwięzła sekcja - proste pytanie lub krótka informacja"
    ),
    "medium": LengthProfile(
        name="medium",
        paragraphs_min=2,
        paragraphs_max=4,
        words_min=400,     # 🆕 v41.2: +100 (było 300)
        words_max=600,     # 🆕 v41.2: +100 (było 500)
        snippet_required=True,
        description="Standardowa sekcja - wymaga rozwinięcia"
    ),
    "long": LengthProfile(
        name="long",
        paragraphs_min=3,
        paragraphs_max=4,
        words_min=500,     # 🆕 v41.2: +100 (było 400)
        words_max=700,     # 🆕 v41.2: +100 (było 600)
        snippet_required=True,
        description="Rozbudowana sekcja - wiele aspektów do omówienia"
    ),
    "extended": LengthProfile(
        name="extended",
        paragraphs_min=3,
        paragraphs_max=4,  # 🆕 v41.2: max 4 (było 7)
        words_min=600,     # 🆕 v41.2: +100 (było 500)
        words_max=850,     # 🆕 v41.2: +100 (było 750)
        snippet_required=True,
        description="Bardzo rozbudowana sekcja - kompleksowe omówienie"
    ),
    "intro": LengthProfile(
        name="intro",
        paragraphs_min=2,
        paragraphs_max=3,
        words_min=250,     # 🆕 v41.2: +100 (było 150)
        words_max=350,     # 🆕 v41.2: +100 (było 250)
        snippet_required=True,
        description="Intro - hook + direct answer"
    )
}


# ================================================================
# 🏷️ KLASYFIKACJA TYPU H2
# ================================================================

H2_TYPE_CONFIG = {
    # Pytania liczbowe - zazwyczaj krótkie, konkretne
    "ile_kosztuje": {
        "patterns": ["ile kosztuje", "cena", "koszt", "wydatek", "budżet", "cennik"],
        "base_score": 25,
        "snippet_boost": True,
        "typical_profile": "short"
    },
    "ile_trwa": {
        "patterns": ["ile trwa", "jak długo", "czas ", "okres", "termin"],
        "base_score": 20,
        "snippet_boost": True,
        "typical_profile": "short"
    },
    
    # Pytania procesowe - wymagają kroków, dłuższe
    "jak_zrobic": {
        "patterns": ["jak zrobić", "jak wykonać", "jak przygotować", "krok po kroku", 
                     "instrukcja", "poradnik", "sposób na"],
        "base_score": 55,
        "snippet_boost": True,
        "typical_profile": "long"
    },
    "jak_wybrac": {
        "patterns": ["jak wybrać", "jak dobrać", "na co zwrócić", "kryteria", 
                     "co wziąć pod uwagę"],
        "base_score": 45,
        "snippet_boost": True,
        "typical_profile": "medium"
    },
    
    # Pytania binarne - krótkie odpowiedzi
    "czy_mozna": {
        "patterns": ["czy można", "czy da się", "czy warto", "czy opłaca", 
                     "czy trzeba", "czy należy", "czy powinno"],
        "base_score": 25,
        "snippet_boost": True,
        "typical_profile": "short"
    },
    
    # Definicje i wyjaśnienia
    "co_to_jest": {
        "patterns": ["co to jest", "co to znaczy", "czym jest", "definicja", 
                     "co oznacza"],
        "base_score": 35,
        "snippet_boost": True,
        "typical_profile": "medium"
    },
    
    # Listy i porównania - elastyczne, zależą od ilości elementów
    "rodzaje": {
        "patterns": ["rodzaje", "typy", "odmiany", "warianty", "kategorie", "klasyfikacja"],
        "base_score": 45,
        "snippet_boost": False,
        "typical_profile": "long"
    },
    "porownanie": {
        "patterns": ["porównanie", "różnice", "vs", "versus", "a może", 
                     "co lepsze", "który lepszy"],
        "base_score": 50,
        "snippet_boost": False,
        "typical_profile": "long"
    },
    "top_lista": {
        "patterns": ["top ", "najlepsze", "ranking", "zestawienie", "polecane", 
                     "lista ", "przegląd"],
        "base_score": 55,
        "snippet_boost": False,
        "typical_profile": "extended"
    },
    
    # Przyczynowo-skutkowe
    "przyczyny": {
        "patterns": ["przyczyny", "powody", "dlaczego", "z jakiego powodu", 
                     "skąd się bierze"],
        "base_score": 40,
        "snippet_boost": True,
        "typical_profile": "medium"
    },
    "skutki": {
        "patterns": ["skutki", "konsekwencje", "efekty", "co grozi", "następstwa"],
        "base_score": 40,
        "snippet_boost": True,
        "typical_profile": "medium"
    },
    
    # Problemy i rozwiązania
    "problemy": {
        "patterns": ["problemy", "trudności", "wady", "błędy", "pułapki", 
                     "czego unikać", "na co uważać"],
        "base_score": 45,
        "snippet_boost": False,
        "typical_profile": "medium"
    },
    "rozwiazania": {
        "patterns": ["rozwiązania", "jak naprawić", "jak rozwiązać", "co zrobić gdy", 
                     "jak poradzić"],
        "base_score": 50,
        "snippet_boost": True,
        "typical_profile": "long"
    },
    
    # Informacyjne ogólne
    "informacyjne": {
        "patterns": [],  # default
        "base_score": 35,
        "snippet_boost": False,
        "typical_profile": "medium"
    }
}


def classify_h2_type(h2_title: str) -> Tuple[str, Dict]:
    """
    Klasyfikuje H2 według typu semantycznego.
    
    Returns:
        (typ_h2, config_dict)
    """
    h2_lower = h2_title.lower().strip()
    
    # Sprawdź każdy typ po kolei
    for h2_type, config in H2_TYPE_CONFIG.items():
        if h2_type == "informacyjne":
            continue  # default na końcu
        
        for pattern in config["patterns"]:
            if pattern in h2_lower:
                return h2_type, config
    
    # Default: informacyjne
    return "informacyjne", H2_TYPE_CONFIG["informacyjne"]


# ================================================================
# 🔗 SEMANTIC MATCHING - N-gramy i Encje
# ================================================================

def normalize_text(text: str) -> str:
    """Normalizuje tekst do porównań."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_word_overlap(text1: str, text2: str) -> float:
    """Oblicza overlap słów między dwoma tekstami (0-1)."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def find_related_ngrams(h2_title: str, ngrams: List[str], threshold: float = 0.15) -> List[str]:
    """
    Znajduje n-gramy semantycznie powiązane z H2.
    """
    related = []
    h2_normalized = normalize_text(h2_title)
    h2_words = set(h2_normalized.split())
    
    for ngram in ngrams:
        ngram_normalized = normalize_text(ngram)
        ngram_words = set(ngram_normalized.split())
        
        # Metoda 1: Word overlap
        overlap = calculate_word_overlap(h2_title, ngram)
        
        # Metoda 2: Czy n-gram zawiera słowa z H2?
        common_words = h2_words & ngram_words
        
        # Metoda 3: Czy H2 zawiera cały n-gram?
        contains_ngram = ngram_normalized in h2_normalized
        
        if overlap >= threshold or len(common_words) >= 2 or contains_ngram:
            related.append(ngram)
    
    return related


def find_related_entities(h2_title: str, entities: List, threshold: float = 0.2) -> List:
    """
    Znajduje encje powiązane z H2.
    """
    related = []
    h2_normalized = normalize_text(h2_title)
    
    for entity in entities:
        entity_name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
        entity_normalized = normalize_text(entity_name)
        
        # Sprawdź czy nazwa encji jest w H2
        if entity_normalized in h2_normalized:
            related.append(entity)
            continue
        
        # Sprawdź overlap słów
        overlap = calculate_word_overlap(h2_title, entity_name)
        if overlap >= threshold:
            related.append(entity)
    
    return related


def find_matching_paa(h2_title: str, paa_questions: List[str], threshold: float = 0.25) -> List[str]:
    """
    Znajduje pytania PAA powiązane z H2.
    """
    matching = []
    
    for paa in paa_questions:
        overlap = calculate_word_overlap(h2_title, paa)
        if overlap >= threshold:
            matching.append(paa)
    
    return matching


# ================================================================
# 📊 GŁÓWNA FUNKCJA - COMPLEXITY SCORE
# ================================================================

@dataclass
class ComplexityResult:
    """Wynik analizy złożoności batcha."""
    score: int  # 0-100
    profile: LengthProfile
    factors: Dict[str, Any]
    reasoning: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "profile_name": self.profile.name,
            "paragraphs_min": self.profile.paragraphs_min,
            "paragraphs_max": self.profile.paragraphs_max,
            "words_min": self.profile.words_min,
            "words_max": self.profile.words_max,
            "snippet_required": self.profile.snippet_required,
            "factors": self.factors,
            "reasoning": self.reasoning
        }


def calculate_batch_complexity(
    h2_title: str,
    ngrams: List[str] = None,
    entities: List = None,
    keywords_for_batch: List[Dict] = None,
    paa_questions: List[str] = None,
    is_intro: bool = False,
    is_final: bool = False
) -> ComplexityResult:
    """
    🎯 Oblicza złożoność batcha na podstawie wielu czynników.
    
    Args:
        h2_title: Tytuł H2 (lub "INTRO" dla intro)
        ngrams: N-gramy z S1
        entities: Encje z S1
        keywords_for_batch: Keywords przypisane do tego batcha
        paa_questions: Pytania PAA z S1
        is_intro: Czy to batch intro
        is_final: Czy to ostatni batch
    
    Returns:
        ComplexityResult ze score i profilem
    """
    ngrams = ngrams or []
    entities = entities or []
    keywords_for_batch = keywords_for_batch or []
    paa_questions = paa_questions or []
    
    reasoning = []
    factors = {}
    
    # ========================================
    # INTRO - specjalny przypadek
    # ========================================
    if is_intro:
        return ComplexityResult(
            score=30,
            profile=PROFILES["intro"],
            factors={"type": "intro"},
            reasoning=["INTRO: Stały profil - hook + direct answer (40-60 słów)"]
        )
    
    # ========================================
    # 1. TYP H2 - bazowy score (0-55 pkt)
    # ========================================
    h2_type, h2_config = classify_h2_type(h2_title)
    base_score = h2_config["base_score"]
    
    factors["h2_type"] = h2_type
    factors["h2_base_score"] = base_score
    reasoning.append(f"Typ H2: '{h2_type}' → bazowy score {base_score}")
    
    score = base_score
    
    # ========================================
    # 2. N-GRAMY powiązane (0-20 pkt)
    # ========================================
    related_ngrams = find_related_ngrams(h2_title, ngrams)
    ngram_score = min(20, len(related_ngrams) * 3)
    
    factors["related_ngrams_count"] = len(related_ngrams)
    factors["related_ngrams"] = related_ngrams[:5]  # max 5 do response
    factors["ngram_score"] = ngram_score
    
    if related_ngrams:
        reasoning.append(f"N-gramy: {len(related_ngrams)} powiązanych → +{ngram_score} pkt")
        score += ngram_score
    
    # ========================================
    # 3. ENCJE do zdefiniowania (0-20 pkt)
    # ========================================
    related_entities = find_related_entities(h2_title, entities)
    entity_score = min(20, len(related_entities) * 5)
    
    factors["related_entities_count"] = len(related_entities)
    factors["entity_score"] = entity_score
    
    if related_entities:
        entity_names = [e.get("name", str(e)) if isinstance(e, dict) else str(e) 
                       for e in related_entities[:3]]
        factors["related_entities"] = entity_names
        reasoning.append(f"Encje: {len(related_entities)} ({', '.join(entity_names)}) → +{entity_score} pkt")
        score += entity_score
    
    # ========================================
    # 4. KEYWORDS - minimum dla density (0-15 pkt)
    # ========================================
    keywords_count = len(keywords_for_batch)
    total_keyword_uses = sum(k.get("uses_this_batch", 1) for k in keywords_for_batch)
    
    # Więcej keywords = potrzeba więcej miejsca
    keyword_score = min(15, total_keyword_uses * 2)
    
    factors["keywords_count"] = keywords_count
    factors["total_keyword_uses"] = total_keyword_uses
    factors["keyword_score"] = keyword_score
    
    if keywords_count > 0:
        reasoning.append(f"Keywords: {keywords_count} fraz ({total_keyword_uses} użyć) → +{keyword_score} pkt")
        score += keyword_score
    
    # ========================================
    # 5. PAA match - snippet potential (0-15 pkt)
    # ========================================
    matching_paa = find_matching_paa(h2_title, paa_questions)
    paa_score = min(15, len(matching_paa) * 5)
    
    factors["matching_paa_count"] = len(matching_paa)
    factors["matching_paa"] = matching_paa[:3]  # max 3 do response
    factors["paa_score"] = paa_score
    
    if matching_paa:
        reasoning.append(f"PAA match: {len(matching_paa)} pytań → +{paa_score} pkt (snippet potential!)")
        score += paa_score
    
    # ========================================
    # 6. FINAL BATCH - bonus (0-10 pkt)
    # ========================================
    if is_final:
        final_bonus = 10
        score += final_bonus
        reasoning.append(f"Ostatni batch: +{final_bonus} pkt (upewnij się że wszystko jest)")
        factors["final_bonus"] = final_bonus
    
    # ========================================
    # MAPOWANIE SCORE → PROFIL
    # ========================================
    score = min(100, max(0, score))  # clamp 0-100
    
    # 🆕 v35.8: VARIABILITY JITTER - żeby batche były zróżnicowane
    # Losowe odchylenie ±12 punktów
    jitter = random.randint(-12, 12)
    score_with_jitter = min(100, max(0, score + jitter))
    
    # Zapamiętaj oryginalny score ale użyj score_with_jitter do profilu
    factors["original_score"] = score
    factors["jitter"] = jitter
    factors["score_with_jitter"] = score_with_jitter
    
    reasoning.append(f"VARIABILITY: base {score} + jitter {jitter:+d} = {score_with_jitter}")
    
    if score_with_jitter >= 65:
        profile = PROFILES["extended"]
    elif score_with_jitter >= 45:
        profile = PROFILES["long"]
    elif score_with_jitter >= 28:
        profile = PROFILES["medium"]
    else:
        profile = PROFILES["short"]
    
    # Użyj score_with_jitter jako final score
    score = score_with_jitter
    
    factors["final_score"] = score
    
    # Snippet info z H2 config
    if h2_config.get("snippet_boost"):
        factors["snippet_recommended"] = True
        reasoning.append(f"⚠️ Typ '{h2_type}' wymaga snippet block!")
    
    reasoning.append(f"→ Końcowy score: {score}/100 → profil: {profile.name.upper()}")
    reasoning.append(f"→ {profile.paragraphs_min}-{profile.paragraphs_max} akapitów, {profile.words_min}-{profile.words_max} słów")
    
    return ComplexityResult(
        score=score,
        profile=profile,
        factors=factors,
        reasoning=reasoning
    )


# ================================================================
# 🔧 HELPER: Batch planner integration
# ================================================================

def calculate_complexity_for_batch_plan(
    batch_number: int,
    total_batches: int,
    h2_sections: List[str],
    all_ngrams: List[str] = None,
    all_entities: List = None,
    keywords_for_batch: List[Dict] = None,
    paa_questions: List[str] = None
) -> Dict:
    """
    Wrapper do użycia w batch_planner.py
    
    Returns:
        Dict gotowy do włączenia w BatchPlan
    """
    is_intro = (batch_number == 1)
    is_final = (batch_number == total_batches)
    
    # Złącz wszystkie H2 dla tego batcha
    if is_intro:
        h2_title = "INTRO"
    elif h2_sections:
        h2_title = " | ".join(h2_sections)
    else:
        h2_title = f"Batch {batch_number}"
    
    result = calculate_batch_complexity(
        h2_title=h2_title,
        ngrams=all_ngrams,
        entities=all_entities,
        keywords_for_batch=keywords_for_batch,
        paa_questions=paa_questions,
        is_intro=is_intro,
        is_final=is_final
    )
    
    # 🆕 v41.2: SKALOWANIE Z LICZBĄ H2!
    # Każda sekcja H2 wymaga minimum 250 słów (było 150)
    MIN_WORDS_PER_H2 = 250
    h2_count = len(h2_sections) if h2_sections else 1
    
    words_min = result.profile.words_min
    words_max = result.profile.words_max
    para_min = result.profile.paragraphs_min
    para_max = result.profile.paragraphs_max
    
    h2_multiplier = 1.0
    if h2_count > 1:
        # Skaluj słowa: każda dodatkowa H2 dodaje ~80% bazowej długości
        h2_multiplier = 1 + (h2_count - 1) * 0.8
        words_min = int(words_min * h2_multiplier)
        words_max = int(words_max * h2_multiplier)
        # 🆕 v41.2: Max 4 akapity per H2
        para_min = h2_count * 2  # Min 2 akapity per H2
        para_max = h2_count * 4  # Max 4 akapity per H2
    
    # Enforce absolute minimum: 250 słów per H2 (było 150)
    words_min = max(words_min, h2_count * MIN_WORDS_PER_H2)
    
    return {
        "complexity_score": result.score,
        "profile_name": result.profile.name,
        "target_words_min": words_min,
        "target_words_max": words_max,
        "target_paragraphs_min": para_min,
        "target_paragraphs_max": para_max,
        "snippet_required": result.profile.snippet_required,
        "factors": {**result.factors, "h2_count": h2_count, "h2_multiplier": h2_multiplier},
        "reasoning": result.reasoning + [f"🆕 H2 scaling: {h2_count} sekcji × {MIN_WORDS_PER_H2} min = {words_min}-{words_max} słów, max 4 akapity/H2"]
    }


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    # Przykładowe dane
    test_h2s = [
        "Ile kosztuje konserwacja noża?",
        "Jak prawidłowo naostrzyć nóż kuchenny krok po kroku?",
        "Rodzaje stali używanych w nożach",
        "Czy można myć noże w zmywarce?",
        "Co to jest twardość stali HRC?",
        "Najlepsze noże kuchenne - ranking 2024"
    ]
    
    test_ngrams = [
        "konserwacja noża",
        "ostrzyć nóż",
        "stal nierdzewna",
        "twardość hrc",
        "noże kuchenne",
        "zmywarka",
        "kąt ostrzenia",
        "ranking noży"
    ]
    
    test_entities = [
        {"name": "HRC", "type": "term"},
        {"name": "stal damasceńska", "type": "material"},
        {"name": "nóż szefa kuchni", "type": "product"},
        {"name": "ostrzałka", "type": "tool"}
    ]
    
    test_paa = [
        "Jak często ostrzyć noże?",
        "Czy noże można myć w zmywarce?",
        "Jaka stal jest najlepsza na noże?",
        "Ile kosztuje dobry nóż kuchenny?"
    ]
    
    print("=" * 70)
    print("  BATCH COMPLEXITY CALCULATOR - TEST")
    print("=" * 70)
    
    for h2 in test_h2s:
        result = calculate_batch_complexity(
            h2_title=h2,
            ngrams=test_ngrams,
            entities=test_entities,
            paa_questions=test_paa
        )
        
        print(f"\n📌 H2: \"{h2}\"")
        print(f"   Score: {result.score}/100 → {result.profile.name.upper()}")
        print(f"   Akapity: {result.profile.paragraphs_min}-{result.profile.paragraphs_max}")
        print(f"   Słowa: {result.profile.words_min}-{result.profile.words_max}")
        for r in result.reasoning:
            print(f"   • {r}")
