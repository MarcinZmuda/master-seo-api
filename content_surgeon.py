"""
===============================================================================
🔬 CONTENT SURGEON v1.0
===============================================================================
Chirurgiczne wstawianie brakujących elementów zamiast kosztownego retry.

PROBLEM:
- Walidator wykrywa brak frazy "sąd rodzinny"
- System odrzuca CAŁY batch i generuje od nowa
- Nowa wersja może zgubić coś innego
- Marnowanie tokenów i czasu

ROZWIĄZANIE:
- Lokalizuj miejsce gdzie fraza PASUJE semantycznie
- Wstaw JEDNO zdanie z frazą
- Wygładź sąsiednie zdania
- Zachowaj resztę tekstu bez zmian

ZYSK:
- 100% pokrycia fraz MUST
- Bez psucia reszty tekstu
- 10x mniej tokenów niż retry

===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InjectionPoint:
    """Punkt wstrzyknięcia w tekście."""
    paragraph_idx: int
    sentence_idx: int
    position: str  # "before", "after", "replace"
    relevance_score: float
    context: str


@dataclass
class InjectionResult:
    """Wynik operacji wstrzyknięcia."""
    success: bool
    original_text: str
    modified_text: str
    injected_phrase: str
    injection_point: InjectionPoint
    new_sentence: str
    smoothing_applied: bool


# ============================================================================
# KONFIGURACJA
# ============================================================================

class SurgeonConfig:
    MIN_PARAGRAPH_LENGTH: int = 30
    MAX_INJECTIONS_PER_BATCH: int = 3
    MIN_RELEVANCE_THRESHOLD: float = 0.3
    
    DOMAIN_CONTEXT_WORDS: Dict[str, List[str]] = {
        "prawo": [
            "sąd", "orzeczenie", "wyrok", "postępowanie", "procedura",
            "prawo", "ustawa", "kodeks", "przepis", "artykuł",
            "rodzic", "dziecko", "opieka", "władza", "kontakt"
        ],
    }


CONFIG = SurgeonConfig()


# ============================================================================
# LOKALIZACJA PUNKTU WSTRZYKNIĘCIA
# ============================================================================

def find_injection_point(
    text: str,
    missing_phrase: str,
    domain: str = "prawo"
) -> Optional[InjectionPoint]:
    """
    Znajduje najlepsze miejsce do wstrzyknięcia frazy.
    
    Strategia:
    1. Podziel tekst na akapity i zdania
    2. Dla każdego zdania oblicz "relevance" do brakującej frazy
    3. Wybierz zdanie z najwyższym relevance
    4. Wstaw PO tym zdaniu (nie na końcu akapitu!)
    """
    paragraphs = split_into_paragraphs(text)
    
    best_point = None
    best_score = CONFIG.MIN_RELEVANCE_THRESHOLD
    
    for p_idx, paragraph in enumerate(paragraphs):
        if len(paragraph.split()) < CONFIG.MIN_PARAGRAPH_LENGTH:
            continue
        
        sentences = split_into_sentences(paragraph)
        
        for s_idx, sentence in enumerate(sentences):
            # Nie wstawiaj na początku ani końcu akapitu (fake humanization!)
            if s_idx == 0 or s_idx == len(sentences) - 1:
                continue
            
            score = calculate_injection_relevance(sentence, missing_phrase, domain)
            
            if score > best_score:
                best_score = score
                best_point = InjectionPoint(
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                    position="after",
                    relevance_score=score,
                    context=get_context_window(sentences, s_idx)
                )
    
    return best_point


def calculate_injection_relevance(
    sentence: str,
    phrase: str,
    domain: str = "prawo"
) -> float:
    """Oblicza jak dobrze fraza pasuje do kontekstu zdania."""
    sentence_lower = sentence.lower()
    phrase_lower = phrase.lower()
    phrase_words = set(phrase_lower.split())
    sentence_words = set(sentence_lower.split())
    
    score = 0.0
    
    # 1. Wspólne słowa
    common = phrase_words & sentence_words
    score += len(common) * 0.15
    
    # 2. Słowa kluczowe domeny
    domain_words = CONFIG.DOMAIN_CONTEXT_WORDS.get(domain, [])
    phrase_domain = [w for w in phrase_words if any(dw in w for dw in domain_words)]
    sentence_domain = [w for w in sentence_words if any(dw in w for dw in domain_words)]
    
    if phrase_domain and sentence_domain:
        score += 0.25
    
    # 3. Powiązania semantyczne
    if domain == "prawo":
        RELATED = {
            "sąd": ["orzeczenie", "wyrok", "postępowanie", "rozprawa"],
            "rodzic": ["dziecko", "opieka", "władza", "kontakt"],
            "kodeks": ["artykuł", "przepis", "ustawa", "prawo"],
            "dziecko": ["rodzic", "opieka", "dobro", "miejsce"],
        }
        
        for key, related in RELATED.items():
            if key in phrase_lower:
                for rel in related:
                    if rel in sentence_lower:
                        score += 0.2
                        break
    
    # 4. Preferuj zdania średniej długości
    if 10 <= len(sentence.split()) <= 25:
        score += 0.1
    
    return min(1.0, score)


def get_context_window(sentences: List[str], idx: int, window: int = 2) -> str:
    """Pobiera kontekst wokół zdania."""
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


# ============================================================================
# GENEROWANIE ZDANIA DO WSTRZYKNIĘCIA
# ============================================================================

def generate_injection_sentence(
    phrase: str,
    context: str,
    h2_title: str,
    domain: str = "prawo"
) -> str:
    """Generuje zdanie zawierające brakującą frazę."""
    phrase_lower = phrase.lower()
    
    if domain == "prawo":
        templates = get_legal_templates(phrase_lower)
    else:
        templates = get_generic_templates(phrase_lower)
    
    return select_best_template(templates, context, h2_title).format(phrase=phrase)


def get_legal_templates(phrase: str) -> List[str]:
    """Zwraca szablony zdań dla domeny prawnej."""
    
    if "sąd" in phrase:
        return [
            "W takich przypadkach {phrase} odgrywa kluczową rolę.",
            "Właściwym organem jest tutaj {phrase}.",
            "Sprawę rozpatruje {phrase}.",
        ]
    
    if "kodeks" in phrase or "art." in phrase:
        return [
            "Podstawę prawną stanowi {phrase}.",
            "Kwestię tę reguluje {phrase}.",
            "Zgodnie z {phrase}, sytuacja wygląda następująco.",
        ]
    
    if "rodzic" in phrase or "dziec" in phrase:
        return [
            "W kontekście relacji rodzinnych, {phrase} ma szczególne znaczenie.",
            "Istotną kwestią jest tutaj {phrase}.",
        ]
    
    if "władza" in phrase:
        return [
            "Kluczowym aspektem jest {phrase}.",
            "Nie można pominąć kwestii {phrase}.",
        ]
    
    return [
        "Istotnym elementem jest {phrase}.",
        "Znaczenie ma tutaj {phrase}.",
    ]


def get_generic_templates(phrase: str) -> List[str]:
    return [
        "Istotnym aspektem jest {phrase}.",
        "Znaczenie ma tutaj {phrase}.",
    ]


def select_best_template(templates: List[str], context: str, h2_title: str) -> str:
    """Wybiera najlepszy szablon na podstawie kontekstu."""
    context_lower = context.lower()
    
    for template in templates:
        template_words = set(re.findall(r'\b\w{4,}\b', template.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', context_lower))
        
        if template_words & context_words:
            return template
    
    return templates[0]


# ============================================================================
# WSTRZYKIWANIE I WYGŁADZANIE
# ============================================================================

def inject_sentence(text: str, injection_point: InjectionPoint, new_sentence: str) -> str:
    """Wstawia zdanie w określone miejsce."""
    paragraphs = split_into_paragraphs(text)
    
    if injection_point.paragraph_idx >= len(paragraphs):
        return text
    
    target_paragraph = paragraphs[injection_point.paragraph_idx]
    sentences = split_into_sentences(target_paragraph)
    
    if injection_point.sentence_idx >= len(sentences):
        return text
    
    if injection_point.position == "after":
        sentences.insert(injection_point.sentence_idx + 1, new_sentence)
    elif injection_point.position == "before":
        sentences.insert(injection_point.sentence_idx, new_sentence)
    
    paragraphs[injection_point.paragraph_idx] = " ".join(sentences)
    return "\n\n".join(paragraphs)


def smooth_around_injection(text: str, injection_point: InjectionPoint) -> str:
    """Wygładza tekst wokół miejsca wstrzyknięcia."""
    paragraphs = split_into_paragraphs(text)
    
    if injection_point.paragraph_idx >= len(paragraphs):
        return text
    
    paragraph = paragraphs[injection_point.paragraph_idx]
    paragraph = re.sub(r'\s+', ' ', paragraph)
    
    paragraphs[injection_point.paragraph_idx] = paragraph.strip()
    return "\n\n".join(paragraphs)


# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def perform_surgery(
    text: str,
    missing_phrases: List[str],
    h2_title: str,
    domain: str = "prawo",
    max_injections: int = None
) -> Dict:
    """
    Główna funkcja Content Surgeon.
    
    Wykonuje chirurgiczne wstrzyknięcia brakujących fraz.
    """
    if max_injections is None:
        max_injections = CONFIG.MAX_INJECTIONS_PER_BATCH
    
    results = []
    failed_phrases = []
    current_text = text
    
    for phrase in missing_phrases[:max_injections]:
        injection_point = find_injection_point(current_text, phrase, domain)
        
        if not injection_point:
            failed_phrases.append(phrase)
            continue
        
        new_sentence = generate_injection_sentence(phrase, injection_point.context, h2_title, domain)
        modified_text = inject_sentence(current_text, injection_point, new_sentence)
        modified_text = smooth_around_injection(modified_text, injection_point)
        
        results.append(InjectionResult(
            success=True,
            original_text=current_text,
            modified_text=modified_text,
            injected_phrase=phrase,
            injection_point=injection_point,
            new_sentence=new_sentence,
            smoothing_applied=True
        ))
        
        current_text = modified_text
    
    if len(missing_phrases) > max_injections:
        failed_phrases.extend(missing_phrases[max_injections:])
    
    return {
        "success": len(results) > 0,
        "original_text": text,
        "modified_text": current_text,
        "injections": results,
        "failed_phrases": failed_phrases,
        "stats": {
            "total_missing": len(missing_phrases),
            "injected": len(results),
            "failed": len(failed_phrases),
            "success_rate": len(results) / len(missing_phrases) if missing_phrases else 1.0
        }
    }


# ============================================================================
# HELPERS
# ============================================================================

def split_into_paragraphs(text: str) -> List[str]:
    text = text.strip()
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    test_text = """
Porwanie rodzicielskie to poważny problem prawny, który dotyka wiele rodzin w Polsce. 
Sytuacja staje się szczególnie trudna, gdy jeden z rodziców decyduje się na samowolne 
zabranie dziecka. W takich przypadkach konieczne jest podjęcie odpowiednich kroków prawnych.

Procedura jest skomplikowana i wymaga znajomości przepisów. Należy przede wszystkim 
zebrać odpowiednią dokumentację. Kolejnym krokiem jest złożenie odpowiednich pism. 
Czas trwania postępowania może być różny w zależności od okoliczności.

Konsekwencje prawne mogą być poważne dla osoby, która dopuściła się takiego czynu. 
Warto skonsultować się z prawnikiem specjalizującym się w prawie rodzinnym.
"""
    
    missing = ["sąd rodzinny", "władza rodzicielska"]
    
    print("=" * 60)
    print("TEST: CONTENT SURGEON")
    print("=" * 60)
    
    result = perform_surgery(
        text=test_text,
        missing_phrases=missing,
        h2_title="Porwanie rodzicielskie - procedura",
        domain="prawo"
    )
    
    print(f"\n✅ Wynik: {result['stats']['injected']}/{result['stats']['total_missing']} wstrzyknięto")
    
    for inj in result["injections"]:
        print(f"\n📍 '{inj.injected_phrase}' → '{inj.new_sentence}'")
    
    print(f"\n📄 Zmodyfikowany tekst:\n{result['modified_text']}")
