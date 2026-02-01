"""
===============================================================================
🔄 KEYWORD SPACING VALIDATOR v1.0
===============================================================================
Rozwiązuje problem keyword stuffingu przez:

1. SPACING CHECK - minimalna odległość między powtórzeniami
2. SYNONYM SUGGESTIONS - gdy za blisko, sugeruj alternatywy  
3. DISTRIBUTION SCORE - czy frazy są równomiernie rozłożone
4. PRE-BATCH CONTEXT - info gdzie była ostatnio użyta fraza

INTEGRACJA:
- pre_batch_info: dodaj last_usage_info dla każdej frazy
- batch_simple: waliduj spacing przed zatwierdzeniem
- instrukcje agenta: "NIE UŻYWAJ X, użyj synonimu Y"

===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math


# ============================================================================
# KONFIGURACJA
# ============================================================================

# Minimalna odległość (w słowach) między powtórzeniami tej samej frazy
MINIMUM_SPACING = {
    "MAIN": 60,       # Fraza główna - co ~60 słów OK
    "BASIC": 80,      # BASIC - co ~80 słów  
    "EXTENDED": 120,  # EXTENDED - co ~120 słów
    "H2": 150,        # H2 header terms - co ~150 słów
}

# Maksymalny % exact match (reszta musi być odmianami/synonimami)
MAX_EXACT_MATCH_RATIO = 0.50  # Max 50% może być identyczne

# Próg stuffingu w akapicie
PARAGRAPH_STUFFING_THRESHOLD = 2  # Max 2x ta sama fraza w akapicie

# Próg stuffingu w zdaniu
SENTENCE_STUFFING_THRESHOLD = 1  # Max 1x ta sama fraza w zdaniu

# Ilość słów na końcu poprzedniego batcha do sprawdzenia
PREVIOUS_BATCH_CONTEXT_WORDS = 80


# ============================================================================
# SYNONIMY DLA FRAZ SEO (rozszerzenie contextual_synonyms)
# ============================================================================

# Synonimy dla całych fraz (nie pojedynczych słów)
PHRASE_SYNONYMS = {
    # Medyczne
    "zespół turnera": [
        "ta aberracja chromosomalna", 
        "to schorzenie genetyczne",
        "omawiany zespół",
        "ta jednostka kliniczna",
        "turner syndrome"  # dla kontekstu międzynarodowego
    ],
    "choroba genetyczna": [
        "schorzenie genetyczne",
        "zaburzenie genetyczne", 
        "wada wrodzona",
        "ta choroba"
    ],
    "aberracja chromosomalna": [
        "zaburzenie chromosomowe",
        "anomalia genetyczna",
        "ta aberracja"
    ],
    
    # Prawne
    "sąd rodzinny": [
        "sąd opiekuńczy",
        "ten sąd",
        "właściwy sąd",
        "organ orzekający"
    ],
    "władza rodzicielska": [
        "prawa rodzicielskie",
        "opieka rodzicielska",
        "ta władza"
    ],
    "miejsce pobytu dziecka": [
        "miejsce zamieszkania dziecka",
        "adres dziecka",
        "to miejsce"
    ],
    "porwanie rodzicielskie": [
        "uprowadzenie przez rodzica",
        "samowolne zabranie dziecka",
        "to porwanie"
    ],
    
    # Ogólne
    "niski wzrost": [
        "niskorosłość",
        "niedobór wzrostu",
        "mniejszy wzrost"
    ],
}

# Zaimki/określenia zastępcze uniwersalne
UNIVERSAL_SUBSTITUTES = {
    "MEDICAL": [
        "ta choroba", "to schorzenie", "ta jednostka", 
        "omawiany zespół", "opisywane zaburzenie"
    ],
    "LEGAL": [
        "ten sąd", "ta instytucja", "właściwy organ",
        "omawiana sprawa", "przedmiotowa kwestia"
    ],
    "GENERIC": [
        "ta kwestia", "omawiany temat", "przedmiotowe zagadnienie"
    ]
}


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class PhrasePosition:
    """Pozycja frazy w tekście."""
    phrase: str
    word_position: int  # Pozycja w słowach od początku
    char_position: int  # Pozycja w znakach
    context: str  # Kilka słów przed i po


@dataclass  
class SpacingViolation:
    """Naruszenie minimalnej odległości."""
    phrase: str
    phrase_type: str
    position1: int
    position2: int
    actual_distance: int
    min_required: int
    severity: str  # CRITICAL, WARNING
    suggestion: str


@dataclass
class LastUsageInfo:
    """Info o ostatnim użyciu frazy - dla pre_batch_info."""
    phrase: str
    words_ago: int  # Ile słów temu (od końca poprzedniego batcha)
    can_use_now: bool  # Czy można użyć na początku nowego batcha
    suggested_wait: int  # Ile słów poczekać
    alternatives: List[str]  # Synonimy do użycia zamiast


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def tokenize_to_words(text: str) -> List[str]:
    """Tokenizuje tekst na słowa."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


def find_phrase_positions(
    text: str, 
    phrase: str,
    use_lemmatization: bool = False
) -> List[PhrasePosition]:
    """
    Znajduje wszystkie pozycje frazy w tekście.
    
    Args:
        text: Tekst do przeszukania
        phrase: Fraza do znalezienia
        use_lemmatization: Czy używać lemmatyzacji (wymaga polish_lemmatizer)
    
    Returns:
        Lista pozycji frazy
    """
    if not text or not phrase:
        return []
    
    positions = []
    text_lower = text.lower()
    phrase_lower = phrase.lower().strip()
    words = tokenize_to_words(text)
    phrase_words = phrase_lower.split()
    phrase_len = len(phrase_words)
    
    if phrase_len == 0:
        return []
    
    # Szukaj frazy w tekście
    for i in range(len(words) - phrase_len + 1):
        window = words[i:i + phrase_len]
        
        # Sprawdź dopasowanie (z tolerancją na końcówki fleksyjne)
        match = True
        for pw, tw in zip(phrase_words, window):
            # Heurystyka: pierwsze 4 litery muszą się zgadzać (dla fleksji)
            min_len = min(len(pw), len(tw), 4)
            if len(pw) >= 4 and len(tw) >= 4:
                if pw[:min_len] != tw[:min_len]:
                    match = False
                    break
            elif pw != tw:
                match = False
                break
        
        if match:
            # Znajdź pozycję znakową
            char_pos = 0
            word_count = 0
            for m in re.finditer(r'\b\w+\b', text_lower):
                if word_count == i:
                    char_pos = m.start()
                    break
                word_count += 1
            
            # Kontekst (3 słowa przed i po)
            context_start = max(0, i - 3)
            context_end = min(len(words), i + phrase_len + 3)
            context = " ".join(words[context_start:context_end])
            
            positions.append(PhrasePosition(
                phrase=phrase,
                word_position=i,
                char_position=char_pos,
                context=context
            ))
    
    return positions


def check_spacing_violations(
    text: str,
    phrase: str,
    phrase_type: str = "BASIC"
) -> List[SpacingViolation]:
    """
    Sprawdza czy fraza nie występuje zbyt blisko poprzedniego użycia.
    
    Returns:
        Lista naruszeń (pusta = wszystko OK)
    """
    min_spacing = MINIMUM_SPACING.get(phrase_type.upper(), 80)
    positions = find_phrase_positions(text, phrase)
    
    if len(positions) < 2:
        return []
    
    violations = []
    
    for i in range(1, len(positions)):
        distance = positions[i].word_position - positions[i-1].word_position
        
        if distance < min_spacing:
            severity = "CRITICAL" if distance < min_spacing // 2 else "WARNING"
            
            # Pobierz synonimy
            alternatives = get_phrase_alternatives(phrase, phrase_type)
            alt_str = ", ".join(alternatives[:3]) if alternatives else "użyj zaimka"
            
            violations.append(SpacingViolation(
                phrase=phrase,
                phrase_type=phrase_type,
                position1=positions[i-1].word_position,
                position2=positions[i].word_position,
                actual_distance=distance,
                min_required=min_spacing,
                severity=severity,
                suggestion=f"Zamień jedno użycie na: {alt_str}"
            ))
    
    return violations


def get_phrase_alternatives(phrase: str, phrase_type: str = "BASIC") -> List[str]:
    """
    Zwraca alternatywy dla frazy (synonimy + zaimki).
    
    Args:
        phrase: Fraza oryginalna
        phrase_type: Typ frazy (MAIN, BASIC, EXTENDED)
    
    Returns:
        Lista alternatyw
    """
    phrase_lower = phrase.lower().strip()
    alternatives = []
    
    # 1. Sprawdź dokładne synonimy frazy
    if phrase_lower in PHRASE_SYNONYMS:
        alternatives.extend(PHRASE_SYNONYMS[phrase_lower])
    
    # 2. Dodaj uniwersalne zamienniki
    # Wykryj domenę na podstawie frazy
    domain = detect_domain(phrase)
    if domain in UNIVERSAL_SUBSTITUTES:
        alternatives.extend(UNIVERSAL_SUBSTITUTES[domain])
    
    # 3. Dodaj formy fleksyjne jako "alternatywy"
    # (w sensie: "użyj dopełniacza zamiast mianownika")
    if " " in phrase:
        # Wielowyrazowa - sugeruj odmianę
        alternatives.append(f"odmiana: '{phrase}' w innym przypadku")
    
    # Usuń duplikaty, zachowaj kolejność
    seen = set()
    unique = []
    for alt in alternatives:
        if alt.lower() not in seen:
            seen.add(alt.lower())
            unique.append(alt)
    
    return unique[:6]  # Max 6 alternatyw


def detect_domain(phrase: str) -> str:
    """Wykrywa domenę frazy (MEDICAL, LEGAL, GENERIC)."""
    phrase_lower = phrase.lower()
    
    medical_markers = ["zespół", "choroba", "schorzenie", "objaw", "leczenie", 
                       "pacjent", "diagnoza", "genetycz", "chromosom"]
    legal_markers = ["sąd", "prawo", "ustawa", "wyrok", "rodzic", "dziecko",
                     "porwanie", "władza", "opiek"]
    
    for marker in medical_markers:
        if marker in phrase_lower:
            return "MEDICAL"
    
    for marker in legal_markers:
        if marker in phrase_lower:
            return "LEGAL"
    
    return "GENERIC"


# ============================================================================
# PRE-BATCH CONTEXT - info dla agenta
# ============================================================================

def get_last_usage_info(
    previous_batch_text: str,
    phrase: str,
    phrase_type: str = "BASIC"
) -> LastUsageInfo:
    """
    Zwraca info o ostatnim użyciu frazy w poprzednim batchu.
    Do użycia w pre_batch_info.
    
    Args:
        previous_batch_text: Tekst poprzedniego batcha
        phrase: Fraza do sprawdzenia
        phrase_type: Typ frazy
    
    Returns:
        LastUsageInfo z info czy można użyć i alternatywami
    """
    min_spacing = MINIMUM_SPACING.get(phrase_type.upper(), 80)
    
    if not previous_batch_text:
        return LastUsageInfo(
            phrase=phrase,
            words_ago=999,
            can_use_now=True,
            suggested_wait=0,
            alternatives=[]
        )
    
    # Weź ostatnie N słów
    words = tokenize_to_words(previous_batch_text)
    last_words = words[-PREVIOUS_BATCH_CONTEXT_WORDS:] if len(words) > PREVIOUS_BATCH_CONTEXT_WORDS else words
    last_text = " ".join(last_words)
    
    # Znajdź pozycje frazy
    positions = find_phrase_positions(last_text, phrase)
    
    if not positions:
        return LastUsageInfo(
            phrase=phrase,
            words_ago=999,
            can_use_now=True,
            suggested_wait=0,
            alternatives=[]
        )
    
    # Ostatnia pozycja
    last_pos = positions[-1].word_position
    words_ago = len(last_words) - last_pos
    
    # Czy można użyć na początku nowego batcha?
    can_use = words_ago >= min_spacing
    suggested_wait = max(0, min_spacing - words_ago) if not can_use else 0
    
    # Alternatywy jeśli nie można
    alternatives = get_phrase_alternatives(phrase, phrase_type) if not can_use else []
    
    return LastUsageInfo(
        phrase=phrase,
        words_ago=words_ago,
        can_use_now=can_use,
        suggested_wait=suggested_wait,
        alternatives=alternatives
    )


def generate_spacing_instructions(
    keywords_state: Dict[str, dict],
    previous_batch_text: str = ""
) -> Dict:
    """
    Generuje instrukcje spacing dla agenta.
    Dodaj wynik do pre_batch_info.
    
    Returns:
        {
            "spacing_rules": [...],
            "avoid_at_start": [...],
            "can_use_freely": [...],
            "fleksja_reminder": str
        }
    """
    result = {
        "spacing_rules": [],
        "avoid_at_start": [],
        "can_use_freely": [],
        "fleksja_reminder": "🔄 FORMY FLEKSYJNE: 'zespołu turnera' = 'zespół turnera' = 'zespołem turnera'"
    }
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            continue
        
        min_spacing = MINIMUM_SPACING.get(phrase_type, 80)
        
        # Dodaj regułę spacing
        result["spacing_rules"].append({
            "phrase": phrase,
            "type": phrase_type,
            "min_spacing": min_spacing,
            "rule": f"'{phrase}' → min {min_spacing} słów między użyciami"
        })
        
        # Sprawdź poprzedni batch
        if previous_batch_text:
            last_usage = get_last_usage_info(previous_batch_text, phrase, phrase_type)
            
            if not last_usage.can_use_now:
                result["avoid_at_start"].append({
                    "phrase": phrase,
                    "words_ago": last_usage.words_ago,
                    "wait": last_usage.suggested_wait,
                    "alternatives": last_usage.alternatives,
                    "instruction": f"⚠️ '{phrase}' była {last_usage.words_ago} słów temu - poczekaj ~{last_usage.suggested_wait} słów lub użyj: {', '.join(last_usage.alternatives[:2])}"
                })
            else:
                result["can_use_freely"].append(phrase)
    
    return result


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_batch_spacing(
    batch_text: str,
    keywords_state: Dict[str, dict],
    previous_batch_text: str = ""
) -> Dict:
    """
    Waliduje spacing w batchu.
    Używaj w batch_simple przed zatwierdzeniem.
    
    Returns:
        {
            "is_valid": bool,
            "score": float (0-100),
            "violations": [...],
            "paragraph_stuffing": [...],
            "sentence_stuffing": [...],
            "suggestions": [...]
        }
    """
    result = {
        "is_valid": True,
        "score": 100.0,
        "violations": [],
        "paragraph_stuffing": [],
        "sentence_stuffing": [],
        "suggestions": []
    }
    
    # Połącz z poprzednim batchem dla sprawdzenia ciągłości
    full_text = (previous_batch_text + "\n\n" + batch_text) if previous_batch_text else batch_text
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            continue
        
        # 1. Spacing violations (w połączonym tekście)
        violations = check_spacing_violations(full_text, phrase, phrase_type)
        for v in violations:
            result["violations"].append({
                "phrase": v.phrase,
                "type": v.phrase_type,
                "distance": v.actual_distance,
                "min_required": v.min_required,
                "severity": v.severity,
                "suggestion": v.suggestion
            })
            
            if v.severity == "CRITICAL":
                result["score"] -= 15
            else:
                result["score"] -= 8
        
        # 2. Paragraph stuffing (tylko w nowym batchu)
        para_stuff = check_paragraph_stuffing(batch_text, phrase)
        result["paragraph_stuffing"].extend(para_stuff)
        result["score"] -= len(para_stuff) * 5
        
        # 3. Sentence stuffing (tylko w nowym batchu)
        sent_stuff = check_sentence_stuffing(batch_text, phrase)
        result["sentence_stuffing"].extend(sent_stuff)
        result["score"] -= len(sent_stuff) * 20  # Bardzo poważne
    
    # Clamp score
    result["score"] = max(0, min(100, result["score"]))
    
    # Determine validity
    result["is_valid"] = (
        result["score"] >= 60 and
        len(result["sentence_stuffing"]) == 0
    )
    
    # Generate suggestions
    if result["violations"]:
        result["suggestions"].append(
            "Rozłóż frazy bardziej równomiernie - użyj synonimów lub form fleksyjnych"
        )
    
    if result["paragraph_stuffing"]:
        result["suggestions"].append(
            "Niektóre akapity mają za dużo tej samej frazy - rozdziel na więcej akapitów"
        )
    
    if result["sentence_stuffing"]:
        result["suggestions"].append(
            "❌ KRYTYCZNE: Ta sama fraza 2x w jednym zdaniu - przepisz!"
        )
    
    return result


def check_paragraph_stuffing(text: str, phrase: str) -> List[str]:
    """Sprawdza czy fraza nie jest za często w jednym akapicie."""
    warnings = []
    paragraphs = re.split(r'\n\s*\n|\n', text)
    
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        
        positions = find_phrase_positions(para, phrase)
        if len(positions) > PARAGRAPH_STUFFING_THRESHOLD:
            warnings.append(
                f"Akapit {i+1}: '{phrase}' występuje {len(positions)}x (max {PARAGRAPH_STUFFING_THRESHOLD})"
            )
    
    return warnings


def check_sentence_stuffing(text: str, phrase: str) -> List[str]:
    """Sprawdza czy fraza nie jest 2x w tym samym zdaniu."""
    warnings = []
    sentences = re.split(r'[.!?]+', text)
    
    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
        
        positions = find_phrase_positions(sent, phrase)
        if len(positions) > SENTENCE_STUFFING_THRESHOLD:
            warnings.append(
                f"Zdanie {i+1}: '{phrase}' występuje {len(positions)}x w jednym zdaniu!"
            )
    
    return warnings


# ============================================================================
# DISTRIBUTION SCORE
# ============================================================================

def calculate_distribution_score(text: str, phrase: str) -> Dict:
    """
    Oblicza jak równomiernie rozłożona jest fraza w tekście.
    
    Returns:
        {
            "score": float (0-100),
            "is_even": bool,
            "gaps": [...],  # Lista odstępów między użyciami
            "cv": float,    # Coefficient of variation
            "suggestion": str
        }
    """
    positions = find_phrase_positions(text, phrase)
    
    if len(positions) < 2:
        return {
            "score": 100.0,
            "is_even": True,
            "gaps": [],
            "cv": 0.0,
            "suggestion": ""
        }
    
    # Oblicz odstępy
    gaps = []
    for i in range(1, len(positions)):
        gap = positions[i].word_position - positions[i-1].word_position
        gaps.append(gap)
    
    # Oblicz CV (coefficient of variation)
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_gap if mean_gap > 0 else 0
    
    # Score: niższe CV = lepszy rozkład
    # CV < 0.3 = świetny, CV > 0.7 = słaby
    if cv < 0.3:
        score = 100.0
    elif cv < 0.5:
        score = 80.0
    elif cv < 0.7:
        score = 60.0
    else:
        score = 40.0
    
    is_even = cv < 0.5
    
    suggestion = ""
    if not is_even:
        min_gap = min(gaps)
        max_gap = max(gaps)
        suggestion = f"Fraza jest nierównomiernie rozłożona (odstępy: {min_gap}-{max_gap} słów). Wyrównaj rozkład."
    
    return {
        "score": score,
        "is_even": is_even,
        "gaps": gaps,
        "cv": round(cv, 2),
        "suggestion": suggestion
    }


# ============================================================================
# FORMAT FOR PROMPT
# ============================================================================

def format_spacing_instructions_for_prompt(instructions: Dict) -> str:
    """Formatuje instrukcje spacing do promptu agenta."""
    lines = []
    
    lines.append("\n" + "=" * 60)
    lines.append("📏 SPACING RULES - Odstępy między frazami")
    lines.append("=" * 60)
    
    # Fleksja reminder
    lines.append(f"\n{instructions['fleksja_reminder']}")
    
    # Avoid at start (najważniejsze!)
    if instructions["avoid_at_start"]:
        lines.append("\n⚠️ NA POCZĄTKU BATCHA - UNIKAJ:")
        for item in instructions["avoid_at_start"]:
            lines.append(f"   • {item['instruction']}")
    
    # Spacing rules
    if instructions["spacing_rules"]:
        lines.append("\n📐 MINIMALNE ODSTĘPY:")
        for rule in instructions["spacing_rules"][:5]:
            lines.append(f"   • {rule['rule']}")
    
    # Can use freely
    if instructions["can_use_freely"]:
        lines.append(f"\n✅ MOŻNA UŻYĆ OD RAZU: {', '.join(instructions['can_use_freely'][:5])}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN - TEST
# ============================================================================

if __name__ == "__main__":
    # Test na tekście z keyword stuffingiem
    test_text = """
Zespół Turnera jest jedną z rzadkich jednostek klinicznych. Zespół Turnera 
to choroba genetyczna, jednak jednocześnie podkreśla się, że zespół Turnera 
nie jest chorobą w rozumieniu stanu całkowicie wykluczającego funkcjonowanie.

W praktyce klinicznej zespół Turnera jest chorobą o bardzo zróżnicowanym przebiegu.
Przypadki zespołu Turnera różnią się nasileniem objawów.

Do najczęstszych objawów zespołu Turnera należy niski wzrost. Niski wzrost 
występuje u większości pacjentek z tym zespołem.
"""
    
    previous_batch = """
Ostatni akapit poprzedniego batcha wspomina o zespole Turnera i jego objawach.
Zespół Turnera jest często diagnozowany w dzieciństwie.
"""
    
    keywords_state = {
        "k1": {"keyword": "zespół turnera", "type": "MAIN"},
        "k2": {"keyword": "choroba genetyczna", "type": "BASIC"},
        "k3": {"keyword": "niski wzrost", "type": "BASIC"},
    }
    
    print("=" * 70)
    print("TEST KEYWORD SPACING VALIDATOR")
    print("=" * 70)
    
    # 1. Test spacing instructions
    print("\n📋 SPACING INSTRUCTIONS:")
    instructions = generate_spacing_instructions(keywords_state, previous_batch)
    print(format_spacing_instructions_for_prompt(instructions))
    
    # 2. Test batch validation
    print("\n" + "=" * 70)
    print("📊 BATCH VALIDATION:")
    print("=" * 70)
    
    validation = validate_batch_spacing(test_text, keywords_state, previous_batch)
    
    print(f"\nValid: {'✅ TAK' if validation['is_valid'] else '❌ NIE'}")
    print(f"Score: {validation['score']}/100")
    
    if validation["violations"]:
        print("\n⚠️ SPACING VIOLATIONS:")
        for v in validation["violations"]:
            print(f"  - '{v['phrase']}': {v['distance']} słów (min {v['min_required']}) [{v['severity']}]")
    
    if validation["paragraph_stuffing"]:
        print("\n⚠️ PARAGRAPH STUFFING:")
        for p in validation["paragraph_stuffing"]:
            print(f"  - {p}")
    
    if validation["sentence_stuffing"]:
        print("\n❌ SENTENCE STUFFING:")
        for s in validation["sentence_stuffing"]:
            print(f"  - {s}")
    
    if validation["suggestions"]:
        print("\n💡 SUGGESTIONS:")
        for s in validation["suggestions"]:
            print(f"  - {s}")
    
    # 3. Test distribution
    print("\n" + "=" * 70)
    print("📈 DISTRIBUTION ANALYSIS:")
    print("=" * 70)
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "")
        dist = calculate_distribution_score(test_text, phrase)
        print(f"\n'{phrase}':")
        print(f"  Score: {dist['score']}/100 | CV: {dist['cv']} | Even: {dist['is_even']}")
        if dist["gaps"]:
            print(f"  Gaps: {dist['gaps']}")
        if dist["suggestion"]:
            print(f"  → {dist['suggestion']}")
