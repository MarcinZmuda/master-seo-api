"""
===============================================================================
🇵🇱 NATURAL POLISH INSTRUCTIONS v1.0
===============================================================================
Rozwiązuje problem keyword stuffingu przez:

1. INFORMACJĘ O FLEKSJI - agent wie że formy odmiany są liczone
2. MINIMUM SPACING - fraza nie może być zbyt blisko poprzedniego użycia
3. SYNONYM ROTATION - wymusza różnorodność form
4. REPETITION DETECTOR - wykrywa nienaturalne powtórzenia

INTEGRACJA:
- Dodaj do smart_batch_instructions.py
- Dodaj do pre_batch_info response
===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# KONFIGURACJA
# ============================================================================

# Minimalna odległość (w słowach) między powtórzeniami tej samej frazy
MINIMUM_SPACING = {
    "MAIN": 60,      # Fraza główna - co ~60 słów OK
    "BASIC": 80,     # BASIC - co ~80 słów
    "EXTENDED": 120  # EXTENDED - co ~120 słów
}

# Maksymalny % exact match (reszta musi być odmianami/synonimami)
MAX_EXACT_MATCH_RATIO = 0.50  # Max 50% może być identyczne

# Próg wykrywania stuffingu (frazy w jednym akapicie)
STUFFING_THRESHOLD = 2  # Max 2x ta sama fraza w jednym akapicie


# ============================================================================
# FLEKSJA INFO - dodaj do instrukcji dla agenta
# ============================================================================

FLEKSJA_INSTRUCTION_PL = """
🔄 FORMY FLEKSYJNE LICZĄ SIĘ AUTOMATYCZNIE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System automatycznie rozpoznaje WSZYSTKIE odmiany jako to samo słowo:

✅ "zespół turnera" = "zespołu turnera" = "zespołem turnera" = "zespole turnera"
✅ "sąd rodzinny" = "sądu rodzinnego" = "sądem rodzinnym" = "sądzie rodzinnym"

⚡ CO TO ZNACZY DLA CIEBIE:
• Pisz NATURALNIE po polsku
• Używaj różnych przypadków gramatycznych
• NIE MUSISZ powtarzać frazy w formie podstawowej
• System zaliczy "zespołu turnera" jako użycie frazy "zespół turnera"

❌ ŹLE (keyword stuffing):
"Zespół Turnera jest chorobą. Zespół Turnera dotyka kobiet. Zespół Turnera wymaga..."

✅ DOBRZE (naturalny polski):
"Zespół Turnera jest chorobą genetyczną. Osoby dotknięte tym zespołem wymagają 
specjalistycznej opieki. W przypadku zespołu Turnera kluczowa jest wczesna diagnoza."
"""

FLEKSJA_INSTRUCTION_SHORT = """
🔄 FLEKSJA: Odmiany frazy liczą się jako jedno użycie!
   "zespół turnera" = "zespołu turnera" = "zespołem turnera"
   Pisz naturalnie, używaj różnych przypadków.
"""


# ============================================================================
# SPACING VALIDATOR - wykrywa zbyt bliskie powtórzenia
# ============================================================================

@dataclass
class SpacingViolation:
    """Naruszenie minimalnego odstępu między frazami."""
    phrase: str
    position1: int
    position2: int
    distance: int
    min_required: int
    suggestion: str


def find_phrase_positions(text: str, phrase: str) -> List[int]:
    """
    Znajduje pozycje (w słowach) gdzie występuje fraza.
    Uwzględnia formy fleksyjne poprzez proste dopasowanie.
    """
    if not text or not phrase:
        return []
    
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    
    # Tokenizuj tekst na słowa z pozycjami
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Szukaj frazy (może być wielowyrazowa)
    phrase_words = phrase_lower.split()
    phrase_len = len(phrase_words)
    
    positions = []
    
    # Proste dopasowanie (dla pełnej wersji użyj lemmatyzacji)
    for i in range(len(words) - phrase_len + 1):
        window = words[i:i + phrase_len]
        
        # Sprawdź czy pasuje (z tolerancją na końcówki fleksyjne)
        match = True
        for pw, tw in zip(phrase_words, window):
            # Prosta heurystyka: 4 pierwsze litery muszą się zgadzać
            if len(pw) >= 4 and len(tw) >= 4:
                if pw[:4] != tw[:4]:
                    match = False
                    break
            elif pw != tw:
                match = False
                break
        
        if match:
            positions.append(i)
    
    return positions


def check_phrase_spacing(
    text: str, 
    phrase: str, 
    phrase_type: str = "BASIC"
) -> Tuple[bool, Optional[SpacingViolation]]:
    """
    Sprawdza czy fraza nie występuje zbyt blisko poprzedniego użycia.
    
    Returns:
        (is_ok, violation_or_none)
    """
    min_spacing = MINIMUM_SPACING.get(phrase_type.upper(), 80)
    
    positions = find_phrase_positions(text, phrase)
    
    if len(positions) < 2:
        return True, None
    
    # Sprawdź odległości między kolejnymi wystąpieniami
    for i in range(1, len(positions)):
        distance = positions[i] - positions[i-1]
        
        if distance < min_spacing:
            return False, SpacingViolation(
                phrase=phrase,
                position1=positions[i-1],
                position2=positions[i],
                distance=distance,
                min_required=min_spacing,
                suggestion=f"Użyj synonimu lub odmiany zamiast powtarzać '{phrase}' - odstęp {distance} słów jest za mały (min {min_spacing})"
            )
    
    return True, None


def validate_all_spacing(
    text: str, 
    keywords_state: Dict[str, dict]
) -> List[SpacingViolation]:
    """
    Sprawdza spacing dla wszystkich fraz w tekście.
    
    Returns:
        Lista naruszeń (pusta = wszystko OK)
    """
    violations = []
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            continue
        
        is_ok, violation = check_phrase_spacing(text, phrase, phrase_type)
        if not is_ok and violation:
            violations.append(violation)
    
    return violations


# ============================================================================
# REPETITION DETECTOR - wykrywa nienaturalne powtórzenia
# ============================================================================

def detect_paragraph_stuffing(text: str, phrase: str, threshold: int = 2) -> List[str]:
    """
    Wykrywa stuffing w pojedynczych akapitach.
    
    Returns:
        Lista ostrzeżeń (pusta = OK)
    """
    warnings = []
    
    # Podziel na akapity
    paragraphs = re.split(r'\n\s*\n|\n', text)
    
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        
        positions = find_phrase_positions(para, phrase)
        
        if len(positions) > threshold:
            warnings.append(
                f"Akapit {i+1}: Fraza '{phrase}' występuje {len(positions)}x "
                f"(max {threshold}x na akapit) - rozłóż na więcej akapitów"
            )
    
    return warnings


def detect_sentence_repetition(text: str, phrase: str) -> List[str]:
    """
    Wykrywa powtórzenia w tym samym zdaniu.
    
    Returns:
        Lista ostrzeżeń
    """
    warnings = []
    
    # Podziel na zdania
    sentences = re.split(r'[.!?]+', text)
    
    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
        
        positions = find_phrase_positions(sent, phrase)
        
        if len(positions) > 1:
            warnings.append(
                f"Zdanie {i+1}: Fraza '{phrase}' występuje {len(positions)}x "
                f"w jednym zdaniu - to brzmi nienaturalnie"
            )
    
    return warnings


# ============================================================================
# INSTRUKCJE DLA AGENTA - format dla pre_batch_info
# ============================================================================

def generate_natural_writing_instructions(
    keywords_state: Dict[str, dict],
    previous_batch_text: str = ""
) -> Dict:
    """
    Generuje instrukcje naturalnego pisania dla agenta.
    
    Dodaj wynik do pre_batch_info jako "natural_writing_instructions".
    """
    instructions = {
        "fleksja_info": FLEKSJA_INSTRUCTION_SHORT,
        "spacing_rules": [],
        "avoid_repetition": [],
        "general_tips": []
    }
    
    # Spacing rules dla każdej frazy
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            continue
        
        min_spacing = MINIMUM_SPACING.get(phrase_type, 80)
        
        instructions["spacing_rules"].append({
            "phrase": phrase,
            "type": phrase_type,
            "min_spacing": min_spacing,
            "rule": f"'{phrase}' - min {min_spacing} słów między użyciami"
        })
    
    # Jeśli mamy poprzedni batch, sprawdź końcówkę
    if previous_batch_text:
        # Sprawdź ostatnie 50 słów poprzedniego batcha
        last_words = previous_batch_text.split()[-50:]
        last_text = " ".join(last_words)
        
        for rid, meta in keywords_state.items():
            phrase = meta.get("keyword", "").strip()
            if not phrase:
                continue
            
            positions = find_phrase_positions(last_text, phrase)
            if positions:
                last_pos = positions[-1]
                words_ago = 50 - last_pos
                
                if words_ago < 30:
                    instructions["avoid_repetition"].append({
                        "phrase": phrase,
                        "warning": f"'{phrase}' była użyta {words_ago} słów temu (na końcu poprzedniego batcha)",
                        "suggestion": f"Zacznij ten batch BEZ '{phrase}' - użyj synonimu lub poczekaj ~{MINIMUM_SPACING.get(meta.get('type', 'BASIC').upper(), 80) - words_ago} słów"
                    })
    
    # General tips
    instructions["general_tips"] = [
        "Używaj RÓŻNYCH przypadków gramatycznych (mianownik, dopełniacz, biernik...)",
        "Synonim lub opis zamiast powtórzenia: 'ta choroba', 'omawiany zespół', 'to schorzenie'",
        "Rozkładaj frazy równomiernie w tekście, nie grupuj na początku/końcu",
        "Jeden akapit = max 2 użycia tej samej frazy"
    ]
    
    return instructions


def format_natural_instructions_for_prompt(instructions: Dict) -> str:
    """
    Formatuje instrukcje do dodania do promptu dla agenta.
    """
    lines = []
    
    lines.append("\n" + "=" * 60)
    lines.append("🇵🇱 NATURALNY POLSKI - JAK PISAĆ")
    lines.append("=" * 60)
    
    # Fleksja info
    lines.append(instructions["fleksja_info"])
    
    # Spacing rules (podsumowanie)
    if instructions["spacing_rules"]:
        lines.append("\n📏 ODSTĘPY MIĘDZY POWTÓRZENIAMI:")
        for rule in instructions["spacing_rules"][:5]:  # Max 5
            lines.append(f"   • {rule['rule']}")
    
    # Avoid repetition (ważne!)
    if instructions["avoid_repetition"]:
        lines.append("\n⚠️ UWAGA - UNIKAJ NA POCZĄTKU TEGO BATCHA:")
        for item in instructions["avoid_repetition"]:
            lines.append(f"   • {item['warning']}")
            lines.append(f"     → {item['suggestion']}")
    
    # General tips
    lines.append("\n💡 WSKAZÓWKI:")
    for tip in instructions["general_tips"]:
        lines.append(f"   • {tip}")
    
    return "\n".join(lines)


# ============================================================================
# WALIDACJA POST-BATCH - sprawdź przed zatwierdzeniem
# ============================================================================

def validate_natural_writing(
    text: str,
    keywords_state: Dict[str, dict],
    previous_batch_text: str = ""
) -> Dict:
    """
    Waliduje czy tekst jest napisany naturalnie.
    
    Returns:
        {
            "is_natural": bool,
            "score": float (0-100),
            "spacing_violations": [...],
            "stuffing_warnings": [...],
            "sentence_repetitions": [...],
            "suggestions": [...]
        }
    """
    result = {
        "is_natural": True,
        "score": 100.0,
        "spacing_violations": [],
        "stuffing_warnings": [],
        "sentence_repetitions": [],
        "suggestions": []
    }
    
    full_text = (previous_batch_text + "\n\n" + text) if previous_batch_text else text
    
    for rid, meta in keywords_state.items():
        phrase = meta.get("keyword", "").strip()
        phrase_type = meta.get("type", "BASIC").upper()
        
        if not phrase:
            continue
        
        # 1. Spacing check
        is_ok, violation = check_phrase_spacing(full_text, phrase, phrase_type)
        if not is_ok and violation:
            result["spacing_violations"].append({
                "phrase": violation.phrase,
                "distance": violation.distance,
                "min_required": violation.min_required,
                "suggestion": violation.suggestion
            })
            result["score"] -= 10
        
        # 2. Paragraph stuffing (tylko w nowym batchu)
        stuffing = detect_paragraph_stuffing(text, phrase, STUFFING_THRESHOLD)
        result["stuffing_warnings"].extend(stuffing)
        result["score"] -= len(stuffing) * 5
        
        # 3. Sentence repetition (tylko w nowym batchu)
        sent_rep = detect_sentence_repetition(text, phrase)
        result["sentence_repetitions"].extend(sent_rep)
        result["score"] -= len(sent_rep) * 15
    
    # Clamp score
    result["score"] = max(0, min(100, result["score"]))
    
    # Determine if natural
    result["is_natural"] = (
        result["score"] >= 70 and
        len(result["sentence_repetitions"]) == 0
    )
    
    # Generate suggestions
    if result["spacing_violations"]:
        result["suggestions"].append(
            "Rozłóż frazy bardziej równomiernie - niektóre są zbyt blisko siebie"
        )
    
    if result["stuffing_warnings"]:
        result["suggestions"].append(
            "Niektóre akapity mają za dużo powtórzeń tej samej frazy - rozdziel na więcej akapitów"
        )
    
    if result["sentence_repetitions"]:
        result["suggestions"].append(
            "KRYTYCZNE: Powtórzenia w tym samym zdaniu brzmią bardzo nienaturalnie - przepisz te zdania"
        )
    
    return result


# ============================================================================
# MAIN - test
# ============================================================================

if __name__ == "__main__":
    # Test
    test_text = """
Zespół Turnera jest jedną z rzadkich jednostek klinicznych. Zespół Turnera 
to choroba genetyczna, jednak jednocześnie podkreśla się, że zespół Turnera 
nie jest chorobą w rozumieniu stanu całkowicie wykluczającego samodzielne funkcjonowanie.

W praktyce klinicznej zespół Turnera jest chorobą o bardzo zróżnicowanym przebiegu.
Przypadki zespołu Turnera różnią się nasileniem objawów.

Częstość zespołu Turnera szacuje się na około 1:2500.
"""
    
    keywords_state = {
        "k1": {"keyword": "zespół turnera", "type": "MAIN"},
        "k2": {"keyword": "choroba genetyczna", "type": "BASIC"},
    }
    
    print("=" * 70)
    print("TEST NATURAL POLISH INSTRUCTIONS")
    print("=" * 70)
    
    # Generate instructions
    instructions = generate_natural_writing_instructions(keywords_state)
    print(format_natural_instructions_for_prompt(instructions))
    
    print("\n" + "=" * 70)
    print("WALIDACJA TEKSTU")
    print("=" * 70)
    
    # Validate
    result = validate_natural_writing(test_text, keywords_state)
    
    print(f"\nNaturalność: {'✅ TAK' if result['is_natural'] else '❌ NIE'}")
    print(f"Score: {result['score']}/100")
    
    if result["spacing_violations"]:
        print("\n⚠️ SPACING VIOLATIONS:")
        for v in result["spacing_violations"]:
            print(f"  - {v['phrase']}: {v['distance']} słów (min {v['min_required']})")
    
    if result["stuffing_warnings"]:
        print("\n⚠️ STUFFING:")
        for w in result["stuffing_warnings"]:
            print(f"  - {w}")
    
    if result["sentence_repetitions"]:
        print("\n❌ SENTENCE REPETITIONS:")
        for r in result["sentence_repetitions"]:
            print(f"  - {r}")
    
    if result["suggestions"]:
        print("\n💡 SUGGESTIONS:")
        for s in result["suggestions"]:
            print(f"  - {s}")
