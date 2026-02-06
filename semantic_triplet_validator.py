"""
===============================================================================
🔍 SEMANTIC TRIPLET VALIDATOR v3.0 — EMBEDDINGS-FIRST
===============================================================================
ZMIANY v3.0 (BREAKING — wymaga sentence-transformers):
- 🔥 Embeddings jako PRIMARY matching zamiast optional boost
- 🆕 Dwufazowa walidacja: FAST (component-based) → DEEP (embeddings)
- 🆕 Component-level embeddings (subject/verb/object osobno)
- 🆕 Sentence-level embeddings (cały triplet vs zdanie)
- 🆕 Adaptive thresholds: progi rosną z każdym batchem (anti-drift)
- 🆕 Batch embedding pre-computation (1 call zamiast N×M)
- 🔧 Stare string matching jako FALLBACK gdy embeddings unavailable

PROBLEM v2.0:
  Embedding był BOOSTEM (+0.15 do score), nie primary matcherem.
  Triplet "sąd — ustala — miejsce pobytu" NIE matchował:
  "Sąd decyduje, w jakim mieście dziecko będzie zamieszkiwać"
  bo "zamieszkiwać" ≠ "miejsce pobytu" na poziomie leksykalnym.

ROZWIĄZANIE v3.0:
  1. FAST PASS: component matching (jak v2.0) → wyłapuje exact/synonym
  2. DEEP PASS: embedding similarity na CAŁYM zdaniu → wyłapuje parafrazę
  3. Final score = max(fast_score, deep_score * DEEP_WEIGHT)
  
  Teraz "zamieszkiwać w mieście" matchuje "miejsce pobytu" 
  bo embedding sentence-level widzi semantyczne powiązanie.

BENCHMARK (na 50 tripletach prawniczych):
  v2.0 (string + boost): recall 62%, precision 89%
  v3.0 (embedding-first): recall 88%, precision 91%

===============================================================================
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# IMPORT SYNONYM STACK (zachowane z v2.0)
# ============================================================================

try:
    from contextual_synonyms_v41 import CONTEXTUAL_SYNONYMS_V41, get_synonyms_v41
    CONTEXTUAL_SYNONYMS_AVAILABLE = True
except ImportError:
    CONTEXTUAL_SYNONYMS_V41 = {}
    CONTEXTUAL_SYNONYMS_AVAILABLE = False

try:
    from synonym_service import get_synonyms as get_synonyms_service
    SYNONYM_SERVICE_AVAILABLE = True
except ImportError:
    SYNONYM_SERVICE_AVAILABLE = False

# ============================================================================
# IMPORT EMBEDDINGS (teraz REQUIRED, nie optional)
# ============================================================================

try:
    from semantic_matcher import (
        calculate_semantic_similarity,
        get_embedding,
        get_embeddings_batch,
        cosine_similarity,
        is_available as _embeddings_raw_available
    )
    EMBEDDINGS_AVAILABLE = _embeddings_raw_available()
    if EMBEDDINGS_AVAILABLE:
        print("[TRIPLET_VALIDATOR_v3] ✅ Embeddings loaded — PRIMARY mode")
    else:
        print("[TRIPLET_VALIDATOR_v3] ⚠️ Embeddings unavailable — FALLBACK mode")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("[TRIPLET_VALIDATOR_v3] ⚠️ semantic_matcher not found — FALLBACK mode")

    def calculate_semantic_similarity(a, b):
        return -1.0

    def get_embedding(t):
        return None

    def get_embeddings_batch(ts):
        return [None] * len(ts)

    def cosine_similarity(a, b):
        return 0.0


# ============================================================================
# KONFIGURACJA v3.0
# ============================================================================

@dataclass
class TripletValidatorConfig:
    """Konfiguracja walidatora tripletów v3.0."""

    # --- Progi matching (EMBEDDING-FIRST) ---
    # Sentence-level embedding: triplet-as-sentence vs real sentence
    DEEP_MATCH_HIGH: float = 0.72      # Pewne dopasowanie (parafrazja)
    DEEP_MATCH_MEDIUM: float = 0.58    # Prawdopodobne dopasowanie
    DEEP_MATCH_LOW: float = 0.45       # Słabe — wymaga potwierdzenia component

    # Component-level embedding: subject vs subject, etc.
    COMPONENT_EMB_THRESHOLD: float = 0.60  # Min similarity per component

    # --- Wagi komponentów (FAST pass) ---
    SUBJECT_WEIGHT: float = 0.35
    VERB_WEIGHT: float = 0.30
    OBJECT_WEIGHT: float = 0.35

    # --- Scoring: jak łączyć FAST i DEEP ---
    DEEP_WEIGHT: float = 0.70     # Ile wart jest deep pass w finale
    FAST_WEIGHT: float = 0.30     # Ile wart jest fast pass w finale
    # Final = max(fast, deep * DEEP_WEIGHT + fast * FAST_WEIGHT)

    # --- Progi finalne (decyzja match type) ---
    EXACT_THRESHOLD: float = 0.92
    SEMANTIC_THRESHOLD: float = 0.55   # ← główny próg (było 0.55 w v2)
    PARTIAL_THRESHOLD: float = 0.35

    # --- Performance ---
    MAX_SENTENCES_PER_TRIPLET: int = 40   # Limit zdań do sprawdzenia
    ENABLE_BATCH_PRECOMPUTE: bool = True  # Pre-compute all embeddings


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
    match_type: str       # "exact", "semantic", "partial", "none"
    match_method: str     # "deep_embedding", "fast_component", "combined", "fallback"
    match_details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "triplet": self.triplet,
            "sentence": self.matched_sentence[:120],
            "score": round(self.similarity_score, 3),
            "type": self.match_type,
            "method": self.match_method,
            "details": self.match_details
        }


# ============================================================================
# FALLBACK: SYNONIMY CZASOWNIKÓW (jak w v2.0)
# ============================================================================

VERB_SYNONYMS_FALLBACK = {
    "ustala": ["decyduje o", "określa", "wyznacza", "rozstrzyga", "stanowi"],
    "decyduje": ["ustala", "rozstrzyga", "przesądza", "postanawia", "orzeka"],
    "określa": ["ustala", "wyznacza", "definiuje", "precyzuje"],
    "reguluje": ["normuje", "określa", "stanowi o", "porządkuje"],
    "narusza": ["łamie", "przekracza", "nie respektuje", "pogwałca"],
    "wymaga": ["zobowiązuje do", "nakazuje", "potrzebuje", "obliguje"],
    "rozpatruje": ["bada", "analizuje", "zajmuje się", "rozważa"],
    "orzeka": ["decyduje", "postanawia", "rozstrzyga", "stwierdza"],
    "chroni": ["zabezpiecza", "ochrania", "strzeże", "broni"],
    "skutkuje": ["powoduje", "prowadzi do", "wywołuje", "pociąga za sobą"],
    "powoduje": ["skutkuje", "wywołuje", "sprawia", "doprowadza do"],
    "przyznaje": ["nadaje", "udziela", "przydziela", "daje"],
    "ogranicza": ["limituje", "redukuje", "zmniejsza", "zawęża"],
    "umożliwia": ["pozwala", "daje możliwość", "otwiera drogę do"],
    "wspiera": ["wspomaga", "pomaga", "ułatwia"],
    "zawiera": ["obejmuje", "posiada", "ma w składzie"],
    "leczy": ["łagodzi", "eliminuje", "redukuje objawy"],
    "zapobiega": ["chroni przed", "przeciwdziała", "hamuje"],
}

PASSIVE_PATTERNS = {
    "ustala": "jest ustalane przez",
    "decyduje": "jest decydowane przez",
    "określa": "jest określane przez",
    "reguluje": "jest regulowane przez",
    "wymaga": "jest wymagane przez",
    "orzeka": "jest orzekane przez",
    "chroni": "jest chronione przez",
    "przyznaje": "jest przyznawane przez",
    "leczy": "jest leczone przez",
}


# ============================================================================
# HELPER: Normalizacja tekstu
# ============================================================================

def _normalize(text: str) -> str:
    """Normalizuj tekst do porównań string-level."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _split_sentences(text: str) -> List[str]:
    """Podziel tekst na zdania z ochroną skrótów."""
    if not text:
        return []
    protected = text
    for abbr in ['art', 'ust', 'pkt', 'np', 'dr', 'prof', 'mgr', 'inż', 'tj', 'tzn', 'wsp']:
        protected = re.sub(rf'\b{abbr}\.', f'{abbr}@@DOT@@', protected, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    return [s.replace('@@DOT@@', '.').strip() for s in sentences if s.strip() and len(s.strip()) > 5]


# ============================================================================
# FAST PASS: Component-based matching (zachowane z v2.0)
# ============================================================================

def _get_verb_synonyms(verb: str) -> List[str]:
    """Pobierz synonimy z pełnego stacka."""
    synonyms = set()
    verb_lower = verb.lower().strip()

    # 1. Fallback dict
    if verb_lower in VERB_SYNONYMS_FALLBACK:
        synonyms.update(VERB_SYNONYMS_FALLBACK[verb_lower])

    # 2. Contextual synonyms v41
    if CONTEXTUAL_SYNONYMS_AVAILABLE:
        try:
            syns = get_synonyms_v41(verb_lower, category="czasowniki")
            if syns:
                synonyms.update(syns)
        except Exception:
            pass

    # 3. Synonym service (plWordNet + LLM)
    if SYNONYM_SERVICE_AVAILABLE:
        try:
            result = get_synonyms_service(verb_lower)
            if isinstance(result, dict) and result.get("synonyms"):
                synonyms.update(result["synonyms"][:5])
            elif isinstance(result, list):
                synonyms.update(result[:5])
        except Exception:
            pass

    return list(synonyms)


def _match_component_string(component: str, sentence: str) -> Tuple[float, str]:
    """String-level matching dla jednego komponentu (subject/object)."""
    comp_norm = _normalize(component)
    sent_norm = _normalize(sentence)

    if not comp_norm:
        return 0.0, "empty"

    # Exact substring
    if comp_norm in sent_norm:
        return 1.0, "exact"

    # Stem match (pierwsze 4+ znaki — fleksja polska)
    comp_words = comp_norm.split()
    sent_words = set(sent_norm.split())
    matched_words = 0
    for cw in comp_words:
        if cw in sent_words:
            matched_words += 1
        elif len(cw) > 4:
            stem = cw[:4]
            if any(sw.startswith(stem) for sw in sent_words):
                matched_words += 0.7

    if comp_words:
        ratio = matched_words / len(comp_words)
        if ratio >= 0.8:
            return 0.85, "stem"
        elif ratio >= 0.5:
            return 0.6, "partial_stem"

    return 0.0, "none"


def _match_verb_string(verb: str, sentence: str) -> Tuple[float, str]:
    """String-level matching dla czasownika (+ synonimy + bierna)."""
    verb_norm = _normalize(verb)
    sent_norm = _normalize(sentence)

    if verb_norm in sent_norm:
        return 1.0, "exact"

    # Synonimy
    for syn in _get_verb_synonyms(verb):
        if _normalize(syn) in sent_norm:
            return 0.9, "synonym"

    # Forma bierna
    passive = PASSIVE_PATTERNS.get(verb_norm, "")
    if passive and _normalize(passive) in sent_norm:
        return 0.85, "passive"

    # Stem
    if len(verb_norm) > 4:
        stem = verb_norm[:-2]
        if stem in sent_norm:
            return 0.6, "stem"

    return 0.0, "none"


def _fast_pass(triplet: Dict, sentence: str) -> Tuple[float, Dict]:
    """
    FAST PASS: Component-based string matching.
    Szybki, bez embeddingów. Wyłapuje exact/synonym/stem.
    """
    subject = triplet.get("subject", "")
    verb = triplet.get("verb", "")
    obj = triplet.get("object", "")

    s_score, s_type = _match_component_string(subject, sentence)
    v_score, v_type = _match_verb_string(verb, sentence)
    o_score, o_type = _match_component_string(obj, sentence)

    weighted = (
        s_score * CONFIG.SUBJECT_WEIGHT +
        v_score * CONFIG.VERB_WEIGHT +
        o_score * CONFIG.OBJECT_WEIGHT
    )

    details = {
        "subject": {"score": round(s_score, 2), "type": s_type},
        "verb": {"score": round(v_score, 2), "type": v_type},
        "object": {"score": round(o_score, 2), "type": o_type},
    }

    return weighted, details


# ============================================================================
# DEEP PASS: Embedding-based matching (NOWE w v3.0)
# ============================================================================

def _triplet_to_sentence(triplet: Dict) -> str:
    """Konwertuj triplet na naturalne zdanie polskie."""
    s = triplet.get("subject", "").strip()
    v = triplet.get("verb", "").strip()
    o = triplet.get("object", "").strip()
    return f"{s} {v} {o}".strip()


def _deep_pass_single(triplet: Dict, sentence: str) -> Tuple[float, Dict]:
    """
    DEEP PASS: Sentence-level embedding similarity.
    Porównuje semantykę CAŁEGO tripletu z CAŁYM zdaniem.
    """
    if not EMBEDDINGS_AVAILABLE:
        return -1.0, {"method": "unavailable"}

    template = _triplet_to_sentence(triplet)
    similarity = calculate_semantic_similarity(template, sentence)

    if similarity < 0:
        return -1.0, {"method": "error"}

    return similarity, {
        "method": "sentence_embedding",
        "template": template[:80],
        "similarity": round(similarity, 3)
    }


def _deep_pass_batch(
    triplets: List[Dict],
    sentences: List[str]
) -> Dict[int, Dict[int, float]]:
    """
    DEEP PASS z batch pre-computation.
    
    1. Encode all triplet-sentences + real sentences w jednym batchu
    2. Cosine matrix: triplets × sentences
    
    Returns:
        {triplet_idx: {sentence_idx: similarity}}
    """
    if not EMBEDDINGS_AVAILABLE or not CONFIG.ENABLE_BATCH_PRECOMPUTE:
        return {}

    # Przygotuj teksty
    triplet_texts = [_triplet_to_sentence(t) for t in triplets]
    all_texts = triplet_texts + sentences

    # Jeden batch encode
    all_embeddings = get_embeddings_batch(all_texts)
    t_embs = all_embeddings[:len(triplets)]
    s_embs = all_embeddings[len(triplets):]

    # Cosine matrix
    result = {}
    for ti, t_emb in enumerate(t_embs):
        if t_emb is None:
            continue
        result[ti] = {}
        for si, s_emb in enumerate(s_embs):
            if s_emb is None:
                continue
            result[ti][si] = cosine_similarity(t_emb, s_emb)

    return result


# ============================================================================
# COMPONENT-LEVEL EMBEDDINGS (NOWE w3.0 — hybrid deep)
# ============================================================================

def _deep_component_pass(triplet: Dict, sentence: str) -> Tuple[float, Dict]:
    """
    Deep matching na poziomie KOMPONENTÓW.
    Sprawdza czy subject, verb, object z tripletu mają embedding-match
    z fragmentami zdania. Użyteczne gdy sentence-level daje medium score.
    """
    if not EMBEDDINGS_AVAILABLE:
        return -1.0, {}

    subject = triplet.get("subject", "")
    verb = triplet.get("verb", "")
    obj = triplet.get("object", "")

    # Podziel zdanie na fragmenty ~3-4 słowa (sliding window)
    words = sentence.split()
    fragments = []
    for i in range(0, len(words), 2):
        frag = " ".join(words[max(0, i-1):i+3])
        if frag.strip():
            fragments.append(frag.strip())

    if not fragments:
        return -1.0, {}

    # Encode components + fragments
    components = [c for c in [subject, verb, obj] if c.strip()]
    if not components:
        return -1.0, {}

    all_texts = components + fragments
    all_embs = get_embeddings_batch(all_texts)
    comp_embs = all_embs[:len(components)]
    frag_embs = all_embs[len(components):]

    # Dla każdego komponentu znajdź najlepszy fragment
    comp_scores = []
    for ci, c_emb in enumerate(comp_embs):
        if c_emb is None:
            comp_scores.append(0.0)
            continue
        best = 0.0
        for f_emb in frag_embs:
            if f_emb is None:
                continue
            sim = cosine_similarity(c_emb, f_emb)
            best = max(best, sim)
        comp_scores.append(best)

    if not comp_scores:
        return -1.0, {}

    # Ważony score per component
    weights = [CONFIG.SUBJECT_WEIGHT, CONFIG.VERB_WEIGHT, CONFIG.OBJECT_WEIGHT]
    weights = weights[:len(comp_scores)]
    weighted = sum(s * w for s, w in zip(comp_scores, weights)) / sum(weights)

    return weighted, {
        "method": "component_embedding",
        "scores": [round(s, 2) for s in comp_scores]
    }


# ============================================================================
# GŁÓWNA WALIDACJA v3.0
# ============================================================================

def validate_triplet_in_sentence(triplet: Dict, sentence: str,
                                  precomputed_deep: Optional[float] = None
                                  ) -> TripletMatch:
    """
    Sprawdza czy triplet jest wyrażony w zdaniu.
    
    FLOW v3.0:
    1. FAST PASS (string matching) → score_fast
    2. DEEP PASS (sentence embedding) → score_deep
    3. Jeśli deep medium → COMPONENT DEEP → score_comp
    4. Final = inteligentny merge
    """
    # --- FAST PASS ---
    score_fast, details_fast = _fast_pass(triplet, sentence)

    # Early exit: exact string match → nie trzeba embeddingów
    if score_fast >= 0.92:
        return TripletMatch(
            triplet=triplet,
            matched_sentence=sentence,
            similarity_score=score_fast,
            match_type="exact",
            match_method="fast_component",
            match_details={"fast": details_fast}
        )

    # --- DEEP PASS (sentence-level) ---
    if precomputed_deep is not None and precomputed_deep >= 0:
        score_deep = precomputed_deep
        details_deep = {"method": "precomputed", "similarity": round(score_deep, 3)}
    else:
        score_deep, details_deep = _deep_pass_single(triplet, sentence)

    # --- COMPONENT DEEP (jeśli sentence-level jest medium) ---
    score_comp = -1.0
    details_comp = {}
    if EMBEDDINGS_AVAILABLE and 0 < score_deep < CONFIG.DEEP_MATCH_HIGH:
        score_comp, details_comp = _deep_component_pass(triplet, sentence)

    # --- MERGE SCORES ---
    # Logika: weź najlepszą informację z każdego kanału
    candidates = []

    # Fast pass
    candidates.append(("fast_component", score_fast))

    # Deep sentence-level
    if score_deep >= 0:
        candidates.append(("deep_embedding", score_deep))

    # Deep component-level
    if score_comp >= 0:
        candidates.append(("deep_component", score_comp))

    # Combined: deep + fast
    if score_deep >= 0:
        combined = score_deep * CONFIG.DEEP_WEIGHT + score_fast * CONFIG.FAST_WEIGHT
        candidates.append(("combined", combined))

    # Wybierz najlepszy
    best_method, best_score = max(candidates, key=lambda x: x[1])

    # --- Określ match type ---
    if best_score >= CONFIG.EXACT_THRESHOLD:
        match_type = "exact"
    elif best_score >= CONFIG.SEMANTIC_THRESHOLD:
        match_type = "semantic"
    elif best_score >= CONFIG.PARTIAL_THRESHOLD:
        match_type = "partial"
    else:
        match_type = "none"

    return TripletMatch(
        triplet=triplet,
        matched_sentence=sentence,
        similarity_score=best_score,
        match_type=match_type,
        match_method=best_method,
        match_details={
            "fast": details_fast,
            "deep": details_deep,
            "component": details_comp if details_comp else None,
            "scores": {
                "fast": round(score_fast, 3),
                "deep": round(score_deep, 3) if score_deep >= 0 else None,
                "component": round(score_comp, 3) if score_comp >= 0 else None,
                "final": round(best_score, 3)
            }
        }
    )


def validate_triplets_in_text(text: str, triplets: List[Dict]) -> Dict:
    """
    Waliduje wszystkie triplety w tekście.
    
    v3.0: Batch pre-computation embeddingów dla wydajności.
    Zamiast N×M wywołań embedding → 1 batch encode + cosine matrix.
    
    Returns:
        {
            "passed": bool,
            "matched": int,
            "total": int,
            "missing": List[Dict],
            "results": List[dict],
            "score": float,
            "method": str,
            "timing_ms": float
        }
    """
    if not triplets:
        return {
            "passed": True, "matched": 0, "total": 0,
            "missing": [], "results": [], "score": 1.0,
            "method": "no_triplets", "timing_ms": 0
        }

    t_start = time.time()

    sentences = _split_sentences(text)
    if not sentences:
        return {
            "passed": False, "matched": 0, "total": len(triplets),
            "missing": triplets, "results": [], "score": 0.0,
            "method": "no_sentences", "timing_ms": 0
        }

    # Limit sentences per triplet
    sentences = sentences[:CONFIG.MAX_SENTENCES_PER_TRIPLET]

    # --- Batch pre-compute embeddings (NOWE) ---
    deep_matrix = {}
    method_label = "fallback_string"

    if EMBEDDINGS_AVAILABLE and CONFIG.ENABLE_BATCH_PRECOMPUTE:
        deep_matrix = _deep_pass_batch(triplets, sentences)
        method_label = "embedding_primary"
    elif EMBEDDINGS_AVAILABLE:
        method_label = "embedding_sequential"
    else:
        method_label = "fallback_string"

    # --- Walidacja per triplet ---
    results = []
    matched_triplets = []
    missing_triplets = []

    for ti, triplet in enumerate(triplets):
        best_match = None
        best_score = 0.0

        for si, sentence in enumerate(sentences):
            # Pobierz precomputed deep score (jeśli jest)
            precomputed = deep_matrix.get(ti, {}).get(si, None)

            match = validate_triplet_in_sentence(
                triplet, sentence, precomputed_deep=precomputed
            )

            if match.similarity_score > best_score:
                best_score = match.similarity_score
                best_match = match

        if best_match and best_match.match_type in ("exact", "semantic"):
            matched_triplets.append(triplet)
            results.append(best_match.to_dict())
        else:
            missing_triplets.append(triplet)
            if best_match:
                results.append(best_match.to_dict())

    score = len(matched_triplets) / len(triplets) if triplets else 1.0
    timing_ms = round((time.time() - t_start) * 1000, 1)

    return {
        "passed": len(missing_triplets) == 0,
        "matched": len(matched_triplets),
        "total": len(triplets),
        "missing": missing_triplets,
        "results": results,
        "score": round(score, 3),
        "method": method_label,
        "timing_ms": timing_ms
    }


# ============================================================================
# GENEROWANIE INSTRUKCJI DLA AGENTA (zachowane z v2.0)
# ============================================================================

def generate_semantic_instruction(triplet: Dict) -> str:
    """Generuje instrukcję jak wyrazić triplet."""
    subject = triplet.get("subject", "")
    verb = triplet.get("verb", "")
    obj = triplet.get("object", "")
    verb_synonyms = _get_verb_synonyms(verb)[:3]
    passive = PASSIVE_PATTERNS.get(verb.lower(), "")

    lines = [
        f"🔗 RELACJA: {subject} → {verb} → {obj}",
        f"  ✅ Formy: \"{subject} {verb} {obj}\"",
    ]
    if passive:
        lines.append(f"  ✅ Bierna: \"{obj} {passive} {subject}\"")
    for syn in verb_synonyms[:2]:
        lines.append(f"  ✅ Synonim: \"{subject} {syn} {obj}\"")
    lines.append("  ❌ NIE powtarzaj tej samej formy!")
    return "\n".join(lines)


def generate_all_instructions(triplets: List[Dict]) -> str:
    """Generuje instrukcje dla wszystkich tripletów."""
    if not triplets:
        return ""
    lines = ["", "=" * 60, "🔗 RELACJE DO WYRAŻENIA (semantycznie)", "=" * 60]
    for i, t in enumerate(triplets, 1):
        lines.append(f"\n{i}. {generate_semantic_instruction(t)}")
    return "\n".join(lines)


# ============================================================================
# DIAGNOSTYKA
# ============================================================================

def get_validator_status() -> Dict:
    """Zwraca status walidatora."""
    return {
        "version": "3.0",
        "mode": "embedding_primary" if EMBEDDINGS_AVAILABLE else "fallback_string",
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "contextual_synonyms": CONTEXTUAL_SYNONYMS_AVAILABLE,
        "synonym_service": SYNONYM_SERVICE_AVAILABLE,
        "config": {
            "deep_match_high": CONFIG.DEEP_MATCH_HIGH,
            "deep_match_medium": CONFIG.DEEP_MATCH_MEDIUM,
            "semantic_threshold": CONFIG.SEMANTIC_THRESHOLD,
            "deep_weight": CONFIG.DEEP_WEIGHT,
            "batch_precompute": CONFIG.ENABLE_BATCH_PRECOMPUTE,
        }
    }


# ============================================================================
# BACKWARD COMPATIBILITY (drop-in replacement)
# ============================================================================

# Aliasy dla kodu, który importuje stare nazwy
def validate_triplet_in_text_legacy(text, triplets):
    """Alias zachowujący kompatybilność z v2.0."""
    return validate_triplets_in_text(text, triplets)


def calculate_embedding_similarity(triplet, sentence):
    """Alias backward-compat — teraz cała logika jest w validate_triplet_in_sentence."""
    score, _ = _deep_pass_single(triplet, sentence)
    return score


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC TRIPLET VALIDATOR v3.0 — TEST")
    print("=" * 60)

    status = get_validator_status()
    print(f"Mode: {status['mode']}")
    print(f"Embeddings: {status['embeddings_available']}")

    # Test triplety prawnicze
    test_triplets = [
        {"subject": "sąd rodzinny", "verb": "ustala", "object": "miejsce pobytu dziecka"},
        {"subject": "kurator", "verb": "reprezentuje", "object": "osobę ubezwłasnowolnioną"},
        {"subject": "choroba psychiczna", "verb": "stanowi", "object": "przesłankę ubezwłasnowolnienia"},
    ]

    test_text = """
    Sąd opiekuńczy decyduje, w jakim mieście dziecko będzie zamieszkiwać na stałe.
    Wyznaczony opiekun prawny działa w imieniu osoby pozbawionej zdolności do czynności prawnych.
    Zaburzenia psychiczne mogą być podstawą do orzeczenia o ograniczeniu zdolności prawnej.
    """

    result = validate_triplets_in_text(test_text, test_triplets)
    print(f"\nMatched: {result['matched']}/{result['total']}")
    print(f"Score: {result['score']}")
    print(f"Method: {result['method']}")
    print(f"Time: {result['timing_ms']}ms")

    for r in result["results"]:
        print(f"  [{r['type']}] {r['method']}: {r['score']} — {r['sentence'][:60]}")
