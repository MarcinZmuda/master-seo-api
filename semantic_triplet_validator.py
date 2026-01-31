"""
===============================================================================
🔍 SEMANTIC TRIPLET VALIDATOR v2.0
===============================================================================
Walidacja semantyczna tripletów z pełną integracją stacka synonimów.

ZMIANY v2.0:
- 🆕 Integracja z contextual_synonyms_v41.py (100+ słów)
- 🆕 Integracja z synonym_service.py (plWordNet + cache + LLM fallback)
- 🆕 Dynamiczne pobieranie synonimów zamiast hardcoded listy
- 🆕 Rozszerzone formy bierne (automatyczne generowanie)
- 🆕 Lepsze matchowanie z użyciem embeddingów (opcjonalne)

PROBLEM: "Sąd rodzinny ustala miejsce pobytu" x3 brzmi jak robot

ROZWIĄZANIE: Akceptuj warianty semantyczne:
- "Miejsce pobytu jest ustalane przez sąd" (passive)
- "Sąd rodzinny decyduje o miejscu pobytu" (synonym)
- "Sąd rodzinny wyznacza miejsce pobytu" (synonym)

===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# IMPORT SYNONYM STACK
# ============================================================================

# 1. Kontekstowe synonimy (100+ słów, kategoryzowane)
try:
    from contextual_synonyms_v41 import (
        CONTEXTUAL_SYNONYMS_V41,
        get_synonyms_v41
    )
    CONTEXTUAL_SYNONYMS_AVAILABLE = True
    print("[TRIPLET_VALIDATOR] ✅ contextual_synonyms_v41 loaded")
except ImportError:
    CONTEXTUAL_SYNONYMS_V41 = {}
    CONTEXTUAL_SYNONYMS_AVAILABLE = False
    print("[TRIPLET_VALIDATOR] ⚠️ contextual_synonyms_v41 not available")

# 2. Synonym service (plWordNet + cache + LLM)
try:
    from synonym_service import get_synonyms as get_synonyms_service
    SYNONYM_SERVICE_AVAILABLE = True
    print("[TRIPLET_VALIDATOR] ✅ synonym_service loaded")
except ImportError:
    SYNONYM_SERVICE_AVAILABLE = False
    print("[TRIPLET_VALIDATOR] ⚠️ synonym_service not available")

# 3. Semantic matcher (embeddings) - opcjonalne
try:
    from semantic_matcher import calculate_semantic_similarity, is_available as embeddings_available
    EMBEDDINGS_AVAILABLE = embeddings_available()
    print(f"[TRIPLET_VALIDATOR] {'✅' if EMBEDDINGS_AVAILABLE else '⚠️'} embeddings {'loaded' if EMBEDDINGS_AVAILABLE else 'not available'}")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("[TRIPLET_VALIDATOR] ⚠️ semantic_matcher not available")


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class TripletValidatorConfig:
    """Konfiguracja walidatora tripletów."""
    # Progi similarity
    EXACT_MATCH_THRESHOLD: float = 1.0
    SEMANTIC_MATCH_THRESHOLD: float = 0.55
    PARTIAL_MATCH_THRESHOLD: float = 0.35
    
    # Wagi komponentów tripletu
    SUBJECT_WEIGHT: float = 0.35
    VERB_WEIGHT: float = 0.30
    OBJECT_WEIGHT: float = 0.35
    
    # Embedding boost (gdy dostępne)
    USE_EMBEDDINGS: bool = True
    EMBEDDING_BOOST: float = 0.15  # Bonus za wysokie embedding similarity


CONFIG = TripletValidatorConfig()


# ============================================================================
# STRUKTURY DANYCH
# ============================================================================

@dataclass
class TripletMatch:
    """Wynik dopasowania tripletu do zdania."""
    triplet: Dict
    matched_sentence: str
    similarity_score: float
    match_type: str  # "exact", "semantic", "partial", "none"
    match_details: Dict = None
    
    def __post_init__(self):
        if self.match_details is None:
            self.match_details = {}


# ============================================================================
# FALLBACK: PODSTAWOWE SYNONIMY CZASOWNIKÓW (gdy brak zewnętrznych źródeł)
# ============================================================================

VERB_SYNONYMS_FALLBACK = {
    # Ustalanie/decydowanie
    "ustala": ["decyduje o", "określa", "wyznacza", "rozstrzyga", "stanowi", "przesądza"],
    "decyduje": ["ustala", "rozstrzyga", "przesądza", "postanawia", "orzeka"],
    "określa": ["ustala", "wyznacza", "definiuje", "precyzuje", "wskazuje"],
    "wyznacza": ["ustala", "określa", "mianuje", "wskazuje"],
    
    # Regulowanie
    "reguluje": ["normuje", "określa", "stanowi o", "porządkuje", "systematyzuje"],
    "normuje": ["reguluje", "porządkuje", "standaryzuje"],
    
    # Naruszanie
    "narusza": ["łamie", "przekracza", "nie respektuje", "pogwałca", "ignoruje"],
    "łamie": ["narusza", "przekracza", "gwałci"],
    
    # Wymaganie
    "wymaga": ["zobowiązuje do", "nakazuje", "potrzebuje", "zakłada", "obliguje"],
    "nakazuje": ["wymaga", "zobowiązuje", "poleca", "zarządza"],
    
    # Rozpatrywanie
    "rozpatruje": ["bada", "analizuje", "zajmuje się", "rozważa", "ocenia"],
    "bada": ["rozpatruje", "analizuje", "sprawdza", "weryfikuje"],
    
    # Orzekanie
    "orzeka": ["decyduje", "postanawia", "rozstrzyga", "wydaje wyrok", "stwierdza"],
    "rozstrzyga": ["orzeka", "decyduje", "postanawia", "przesądza"],
    
    # Reprezentowanie
    "reprezentuje": ["działa w imieniu", "występuje za", "zastępuje", "broni interesów"],
    "zastępuje": ["reprezentuje", "działa za", "występuje w miejsce"],
    
    # Ochrona
    "chroni": ["zabezpiecza", "ochrania", "strzeże", "broni", "osłania"],
    "zabezpiecza": ["chroni", "gwarantuje", "zapewnia ochronę"],
    
    # Skutkowanie
    "skutkuje": ["powoduje", "prowadzi do", "wywołuje", "pociąga za sobą"],
    "powoduje": ["skutkuje", "wywołuje", "sprawia", "doprowadza do"],
    
    # Przyznawanie
    "przyznaje": ["nadaje", "udziela", "daje", "przekazuje"],
    "nadaje": ["przyznaje", "udziela", "daje"],
    
    # Ograniczanie
    "ogranicza": ["limituje", "zawęża", "redukuje", "zmniejsza"],
    "pozbawia": ["odbiera", "zabiera", "usuwa"],
}


# ============================================================================
# FORMY BIERNE - AUTOMATYCZNE GENEROWANIE
# ============================================================================

PASSIVE_PATTERNS = {
    # Wzorzec: czasownik -> forma bierna
    "ustala": "jest ustalane przez",
    "decyduje": "jest decydowane przez",
    "określa": "jest określane przez",
    "wyznacza": "jest wyznaczane przez",
    "reguluje": "jest regulowane przez",
    "rozpatruje": "jest rozpatrywane przez",
    "orzeka": "jest orzekane przez",
    "chroni": "jest chronione przez",
    "reprezentuje": "jest reprezentowane przez",
    "wymaga": "jest wymagane przez",
    "przyznaje": "jest przyznawane przez",
    "nadaje": "jest nadawane przez",
    "ogranicza": "jest ograniczane przez",
    "pozbawia": "jest pozbawiane przez",
    "narusza": "jest naruszane przez",
}


def generate_passive_form(verb: str) -> Optional[str]:
    """
    Generuje formę bierną dla czasownika.
    
    Jeśli nie ma w mapie, próbuje wygenerować automatycznie.
    """
    verb_lower = verb.lower().strip()
    
    # 1. Sprawdź mapę
    if verb_lower in PASSIVE_PATTERNS:
        return PASSIVE_PATTERNS[verb_lower]
    
    # 2. Automatyczne generowanie dla czasowników na -uje, -a
    if verb_lower.endswith("uje"):
        # reguluje -> jest regulowane przez
        stem = verb_lower[:-3]
        return f"jest {stem}owane przez"
    
    if verb_lower.endswith("a") and len(verb_lower) > 3:
        # ustala -> jest ustalane przez
        stem = verb_lower[:-1]
        return f"jest {stem}ane przez"
    
    return None


# ============================================================================
# GŁÓWNA FUNKCJA: POBIERANIE SYNONIMÓW
# ============================================================================

def get_verb_synonyms(verb: str) -> List[str]:
    """
    Pobiera synonimy czasownika z pełnego stacka.
    
    Kolejność źródeł:
    1. contextual_synonyms_v41 (100+ słów, zoptymalizowane dla SEO)
    2. synonym_service (plWordNet + cache + LLM fallback)
    3. VERB_SYNONYMS_FALLBACK (hardcoded backup)
    
    Returns:
        Lista synonimów (bez duplikatów)
    """
    verb_lower = verb.lower().strip()
    synonyms = set()
    
    # 1. Kontekstowe synonimy (najlepsze dla SEO)
    if CONTEXTUAL_SYNONYMS_AVAILABLE and verb_lower in CONTEXTUAL_SYNONYMS_V41:
        synonyms.update(CONTEXTUAL_SYNONYMS_V41[verb_lower])
    
    # 2. Synonym service (plWordNet + cache + LLM)
    if SYNONYM_SERVICE_AVAILABLE:
        try:
            result = get_synonyms_service(verb_lower)
            if result and result.get("synonyms"):
                synonyms.update(result["synonyms"])
        except Exception as e:
            print(f"[TRIPLET_VALIDATOR] synonym_service error: {e}")
    
    # 3. Fallback do lokalnej mapy
    if verb_lower in VERB_SYNONYMS_FALLBACK:
        synonyms.update(VERB_SYNONYMS_FALLBACK[verb_lower])
    
    # Usuń oryginalne słowo jeśli przypadkiem jest w synonimach
    synonyms.discard(verb_lower)
    
    return list(synonyms)


def get_component_synonyms(component: str) -> List[str]:
    """
    Pobiera synonimy dla dowolnego komponentu (subject/object).
    
    Używa tego samego stacka co get_verb_synonyms.
    """
    component_lower = component.lower().strip()
    synonyms = set()
    
    # 1. Kontekstowe synonimy
    if CONTEXTUAL_SYNONYMS_AVAILABLE:
        # Sprawdź całą frazę
        if component_lower in CONTEXTUAL_SYNONYMS_V41:
            synonyms.update(CONTEXTUAL_SYNONYMS_V41[component_lower])
        
        # Sprawdź poszczególne słowa
        for word in component_lower.split():
            if word in CONTEXTUAL_SYNONYMS_V41:
                synonyms.update(CONTEXTUAL_SYNONYMS_V41[word])
    
    # 2. Synonym service
    if SYNONYM_SERVICE_AVAILABLE:
        try:
            result = get_synonyms_service(component_lower)
            if result and result.get("synonyms"):
                synonyms.update(result["synonyms"])
        except Exception:
            pass
    
    return list(synonyms)


# ============================================================================
# FUNKCJE MATCHOWANIA
# ============================================================================

def normalize(text: str) -> str:
    """Normalizuje tekst do porównania."""
    return re.sub(r'[^\w\s]', ' ', text.lower()).strip()


def match_component(target: str, sentence: str, use_synonyms: bool = True) -> Tuple[float, str]:
    """
    Sprawdza czy komponent (subject/object) jest w zdaniu.
    
    Args:
        target: Szukany komponent
        sentence: Zdanie do przeszukania
        use_synonyms: Czy używać synonimów
        
    Returns:
        (score, match_type)
    """
    target_norm = normalize(target)
    sentence_norm = normalize(sentence)
    
    # 1. Exact match
    if target_norm in sentence_norm:
        return 1.0, "exact"
    
    # 2. Synonimy
    if use_synonyms:
        synonyms = get_component_synonyms(target)
        for syn in synonyms:
            if normalize(syn) in sentence_norm:
                return 0.9, "synonym"
    
    # 3. Word overlap
    target_words = set(target_norm.split())
    sentence_words = set(sentence_norm.split())
    overlap = len(target_words & sentence_words) / len(target_words) if target_words else 0
    
    if overlap >= 0.6:
        return overlap, "partial"
    
    # 4. Main word match (słowa > 4 znaki)
    main_words = [w for w in target_words if len(w) > 4]
    if main_words:
        for main_word in main_words:
            if main_word in sentence_norm:
                return 0.5, "main_word"
    
    return 0.0, "none"


def match_verb(verb: str, sentence: str) -> Tuple[float, str]:
    """
    Sprawdza czasownik z synonimami i formą bierną.
    
    Args:
        verb: Czasownik do znalezienia
        sentence: Zdanie do przeszukania
        
    Returns:
        (score, match_type)
    """
    verb_norm = normalize(verb)
    sentence_norm = normalize(sentence)
    
    # 1. Exact match
    if verb_norm in sentence_norm:
        return 1.0, "exact"
    
    # 2. Synonimy (z pełnego stacka)
    synonyms = get_verb_synonyms(verb)
    for syn in synonyms:
        syn_norm = normalize(syn)
        if syn_norm in sentence_norm:
            return 0.9, "synonym"
    
    # 3. Forma bierna
    passive = generate_passive_form(verb)
    if passive and normalize(passive) in sentence_norm:
        return 0.85, "passive"
    
    # 4. Częściowe dopasowanie (rdzeń czasownika)
    if len(verb_norm) > 4:
        stem = verb_norm[:-2]  # Usuń końcówkę
        if stem in sentence_norm:
            return 0.6, "stem"
    
    return 0.0, "none"


def calculate_embedding_similarity(triplet: Dict, sentence: str) -> float:
    """
    Oblicza similarity między tripletem a zdaniem używając embeddingów.
    
    Returns:
        Score 0.0-1.0 lub -1 jeśli embeddingi niedostępne
    """
    if not EMBEDDINGS_AVAILABLE or not CONFIG.USE_EMBEDDINGS:
        return -1.0
    
    try:
        # Zamień triplet na zdanie wzorcowe
        template = f"{triplet.get('subject', '')} {triplet.get('verb', '')} {triplet.get('object', '')}"
        similarity = calculate_semantic_similarity(template, sentence)
        return similarity
    except Exception as e:
        print(f"[TRIPLET_VALIDATOR] Embedding error: {e}")
        return -1.0


# ============================================================================
# GŁÓWNA WALIDACJA
# ============================================================================

def validate_triplet_in_sentence(triplet: Dict, sentence: str) -> TripletMatch:
    """
    Sprawdza czy triplet jest semantycznie wyrażony w zdaniu.
    
    Args:
        triplet: {"subject": "...", "verb": "...", "object": "..."}
        sentence: Zdanie do sprawdzenia
        
    Returns:
        TripletMatch z wynikiem
    """
    subject = triplet.get("subject", "")
    verb = triplet.get("verb", "")
    obj = triplet.get("object", "")
    
    # Dopasuj komponenty
    subject_score, subject_type = match_component(subject, sentence)
    verb_score, verb_type = match_verb(verb, sentence)
    object_score, object_type = match_component(obj, sentence)
    
    # Oblicz ważony score
    weighted_score = (
        subject_score * CONFIG.SUBJECT_WEIGHT +
        verb_score * CONFIG.VERB_WEIGHT +
        object_score * CONFIG.OBJECT_WEIGHT
    )
    
    # Embedding boost (opcjonalnie)
    embedding_score = calculate_embedding_similarity(triplet, sentence)
    if embedding_score > 0.6:
        weighted_score = min(1.0, weighted_score + CONFIG.EMBEDDING_BOOST)
    
    # Określ typ dopasowania
    if weighted_score >= CONFIG.EXACT_MATCH_THRESHOLD * 0.95:
        match_type = "exact"
    elif weighted_score >= CONFIG.SEMANTIC_MATCH_THRESHOLD:
        match_type = "semantic"
    elif weighted_score >= CONFIG.PARTIAL_MATCH_THRESHOLD:
        match_type = "partial"
    else:
        match_type = "none"
    
    return TripletMatch(
        triplet=triplet,
        matched_sentence=sentence,
        similarity_score=weighted_score,
        match_type=match_type,
        match_details={
            "subject": {"score": subject_score, "type": subject_type},
            "verb": {"score": verb_score, "type": verb_type},
            "object": {"score": object_score, "type": object_type},
            "embedding_score": embedding_score if embedding_score >= 0 else None
        }
    )


def validate_triplets_in_text(text: str, triplets: List[Dict]) -> Dict:
    """
    Waliduje wszystkie triplety w tekście.
    
    Args:
        text: Pełny tekst do przeszukania
        triplets: Lista tripletów do zwalidowania
        
    Returns:
        {
            "passed": bool,
            "matched": int,
            "total": int,
            "missing": List[Dict],
            "results": List[TripletMatch],
            "score": float
        }
    """
    if not triplets:
        return {
            "passed": True,
            "matched": 0,
            "total": 0,
            "missing": [],
            "results": [],
            "score": 1.0
        }
    
    # Podziel tekst na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    results = []
    matched_triplets = []
    missing_triplets = []
    
    for triplet in triplets:
        best_match = None
        best_score = 0
        
        # Znajdź najlepsze dopasowanie w zdaniach
        for sentence in sentences:
            match = validate_triplet_in_sentence(triplet, sentence)
            if match.similarity_score > best_score:
                best_score = match.similarity_score
                best_match = match
        
        if best_match and best_match.match_type in ["semantic", "exact"]:
            results.append(best_match)
            matched_triplets.append(triplet)
        else:
            missing_triplets.append(triplet)
            # Dodaj też najlepsze partial match do results (dla debugowania)
            if best_match:
                results.append(best_match)
    
    score = len(matched_triplets) / len(triplets) if triplets else 1.0
    
    return {
        "passed": len(missing_triplets) == 0,
        "matched": len(matched_triplets),
        "total": len(triplets),
        "missing": missing_triplets,
        "results": results,
        "score": score
    }


# ============================================================================
# GENEROWANIE INSTRUKCJI DLA AGENTA
# ============================================================================

def generate_semantic_instruction(triplet: Dict) -> str:
    """
    Generuje instrukcję dla agenta jak wyrazić triplet.
    
    Pokazuje różne akceptowalne formy, zachęcając do różnorodności.
    """
    subject = triplet.get("subject", "")
    verb = triplet.get("verb", "")
    obj = triplet.get("object", "")
    
    # Pobierz synonimy
    verb_synonyms = get_verb_synonyms(verb)[:3]
    passive = generate_passive_form(verb)
    
    instruction = f"""
🔗 RELACJA: {subject} → {verb} → {obj}

✅ AKCEPTOWANE FORMY (wybierz JEDNĄ, nie powtarzaj!):
   • "{subject.capitalize()} {verb} {obj}." (podstawowa)"""
    
    if passive:
        instruction += f"""
   • "{obj.capitalize()} {passive} {subject}." (bierna)"""
    
    if verb_synonyms:
        for syn in verb_synonyms[:2]:
            instruction += f"""
   • "{subject.capitalize()} {syn} {obj}." (synonim)"""
    
    instruction += """

❌ UNIKAJ wielokrotnego użycia tej samej formy!
"""
    return instruction


def generate_all_instructions(triplets: List[Dict]) -> str:
    """Generuje instrukcje dla wszystkich tripletów."""
    if not triplets:
        return ""
    
    lines = ["\n" + "=" * 60]
    lines.append("🔗 RELACJE DO WYRAŻENIA (semantycznie)")
    lines.append("=" * 60)
    
    for i, triplet in enumerate(triplets, 1):
        lines.append(f"\n{i}. {generate_semantic_instruction(triplet)}")
    
    return "\n".join(lines)


# ============================================================================
# DIAGNOSTYKA
# ============================================================================

def get_validator_status() -> Dict:
    """Zwraca status walidatora i dostępnych źródeł."""
    return {
        "version": "2.0",
        "contextual_synonyms_available": CONTEXTUAL_SYNONYMS_AVAILABLE,
        "synonym_service_available": SYNONYM_SERVICE_AVAILABLE,
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "fallback_verbs_count": len(VERB_SYNONYMS_FALLBACK),
        "passive_patterns_count": len(PASSIVE_PATTERNS),
        "config": {
            "semantic_threshold": CONFIG.SEMANTIC_MATCH_THRESHOLD,
            "use_embeddings": CONFIG.USE_EMBEDDINGS
        }
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 SEMANTIC TRIPLET VALIDATOR v2.0 - TEST")
    print("=" * 60)
    
    # Status
    status = get_validator_status()
    print(f"\n📊 STATUS:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test synonimów
    print(f"\n📚 TEST SYNONIMÓW:")
    test_verbs = ["ustala", "orzeka", "chroni", "wymaga"]
    for verb in test_verbs:
        synonyms = get_verb_synonyms(verb)
        print(f"   {verb} → {synonyms[:5]}")
    
    # Test tripletów
    print(f"\n🔗 TEST TRIPLETÓW:")
    triplet = {
        "subject": "sąd rodzinny",
        "verb": "ustala",
        "object": "miejsce pobytu dziecka"
    }
    
    test_sentences = [
        "Sąd rodzinny ustala miejsce pobytu dziecka.",  # exact
        "Miejsce pobytu dziecka jest ustalane przez sąd rodzinny.",  # passive
        "Sąd rodzinny decyduje o miejscu pobytu dziecka.",  # synonym
        "Sąd rodzinny określa gdzie dziecko będzie mieszkać.",  # semantic
        "Rodzice ustalają wspólnie miejsce pobytu.",  # wrong subject
        "Sąd wydał orzeczenie w sprawie alimentów.",  # unrelated
    ]
    
    print(f"\n   Triplet: {triplet['subject']} → {triplet['verb']} → {triplet['object']}\n")
    
    for sentence in test_sentences:
        match = validate_triplet_in_sentence(triplet, sentence)
        status_icon = "✅" if match.match_type in ["semantic", "exact"] else "❌"
        print(f"   {status_icon} {match.similarity_score:.2f} | {match.match_type:8} | {sentence[:50]}...")
        if match.match_details:
            details = match.match_details
            print(f"      └─ S:{details['subject']['type']} V:{details['verb']['type']} O:{details['object']['type']}")
    
    # Test pełnej walidacji
    print(f"\n📄 TEST PEŁNEJ WALIDACJI:")
    text = """
    Sąd rodzinny decyduje o miejscu pobytu dziecka w przypadku konfliktu między rodzicami.
    Kurator reprezentuje interesy małoletniego w postępowaniu sądowym.
    Orzeczenie może być zaskarżone w terminie 14 dni.
    """
    
    triplets = [
        {"subject": "sąd rodzinny", "verb": "ustala", "object": "miejsce pobytu"},
        {"subject": "kurator", "verb": "reprezentuje", "object": "małoletniego"},
        {"subject": "sąd", "verb": "wydaje", "object": "orzeczenie"},  # missing
    ]
    
    result = validate_triplets_in_text(text, triplets)
    print(f"   Passed: {result['passed']}")
    print(f"   Matched: {result['matched']}/{result['total']}")
    print(f"   Score: {result['score']:.2f}")
    if result['missing']:
        print(f"   Missing: {result['missing']}")
    
    # Instrukcja
    print(f"\n📝 PRZYKŁADOWA INSTRUKCJA:")
    print(generate_semantic_instruction(triplet))
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
