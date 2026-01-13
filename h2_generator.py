# ================================================================
# 🏗️ H2 VALIDATOR v29.2 - Semantic HTML + Content Relevancy
# ================================================================
# CLAUDE TWORZY H2 - API TYLKO WALIDUJE!
#
# Ten moduł:
# - Waliduje plan H2 stworzony przez Claude
# - Sprawdza coverage fraz użytkownika
# - Sprawdza Semantic HTML (hierarchia)
# - Sprawdza Content Relevancy (H2 odpowiada H1)
# - Może SUGEROWAĆ poprawki, ale NIE GENERUJE H2
#
# Funkcje główne:
# - validate_h2_plan() - walidacja planu
# - check_phrase_coverage() - czy frazy pokryte
# - calculate_relevancy() - relevancy score
# ================================================================

import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

# ================================================================
# INTENT TEMPLATES - struktura artykułu na podstawie intencji
# ================================================================

INTENT_TEMPLATES = {
    "informational": {
        "description": "Artykuł informacyjny/edukacyjny",
        "structure": [
            {"type": "definition", "pattern": "Czym jest/są {keyword}?", "required": True},
            {"type": "types", "pattern": "Rodzaje {keyword}", "required": False},
            {"type": "benefits", "pattern": "Korzyści z {keyword}", "required": False},
            {"type": "how_it_works", "pattern": "Jak działa {keyword}?", "required": False},
            {"type": "application", "pattern": "{keyword} - zastosowanie", "required": False},
            {"type": "faq", "pattern": "Najczęściej zadawane pytania", "required": True}
        ]
    },
    "how_to": {
        "description": "Poradnik krok po kroku",
        "structure": [
            {"type": "definition", "pattern": "Co to jest {keyword}?", "required": True},
            {"type": "prerequisites", "pattern": "Co potrzebujesz do {keyword}?", "required": False},
            {"type": "steps", "pattern": "Jak {keyword} krok po kroku", "required": True},
            {"type": "tips", "pattern": "Wskazówki i porady", "required": False},
            {"type": "mistakes", "pattern": "Najczęstsze błędy", "required": False},
            {"type": "faq", "pattern": "FAQ", "required": True}
        ]
    },
    "commercial": {
        "description": "Artykuł porównawczy/zakupowy",
        "structure": [
            {"type": "definition", "pattern": "Czym jest {keyword}?", "required": True},
            {"type": "criteria", "pattern": "Jak wybrać {keyword}?", "required": True},
            {"type": "types", "pattern": "Rodzaje {keyword}", "required": False},
            {"type": "comparison", "pattern": "Porównanie {keyword}", "required": False},
            {"type": "price", "pattern": "Cena {keyword} - ile kosztuje?", "required": False},
            {"type": "where_to_buy", "pattern": "Gdzie kupić {keyword}?", "required": False},
            {"type": "faq", "pattern": "FAQ", "required": True}
        ]
    },
    "listicle": {
        "description": "Artykuł listowy/ranking",
        "structure": [
            {"type": "intro", "pattern": "Najlepsze {keyword} - przegląd", "required": True},
            {"type": "criteria", "pattern": "Kryteria wyboru", "required": False},
            {"type": "list_item", "pattern": "{keyword} - opcja {n}", "required": True, "repeat": 5},
            {"type": "comparison", "pattern": "Porównanie opcji", "required": False},
            {"type": "recommendation", "pattern": "Którą opcję wybrać?", "required": True}
        ]
    }
}

# ================================================================
# H2 TYPE MAPPINGS - mapowanie fraz na typy H2
# ================================================================

PHRASE_TYPE_HINTS = {
    # Słowa kluczowe sugerujące typ H2
    "definition": ["czym jest", "co to", "definicja", "znaczenie"],
    "types": ["rodzaje", "typy", "odmiany", "warianty", "kategorie"],
    "benefits": ["korzyści", "zalety", "plusy", "dlaczego warto"],
    "how_it_works": ["jak działa", "mechanizm", "zasada działania"],
    "application": ["zastosowanie", "wykorzystanie", "gdzie stosować"],
    "how_to": ["jak zrobić", "jak stworzyć", "jak wybrać", "instrukcja"],
    "price": ["cena", "koszt", "ile kosztuje", "cennik"],
    "comparison": ["porównanie", "vs", "różnice", "co lepsze"],
    "mistakes": ["błędy", "problemy", "czego unikać"],
    "tips": ["wskazówki", "porady", "triki", "sekrety"],
    "faq": ["pytania", "faq", "q&a"]
}

# ================================================================
# NATURAL H2 PATTERNS - naturalne wzorce nagłówków
# ================================================================

H2_PATTERNS = {
    "definition": [
        "Czym jest {phrase}?",
        "Co to jest {phrase}?",
        "{phrase} - definicja i znaczenie",
        "{phrase} - co warto wiedzieć?"
    ],
    "types": [
        "Rodzaje {phrase}",
        "{phrase} - typy i odmiany",
        "Jakie są rodzaje {phrase}?",
        "Podział {phrase}"
    ],
    "benefits": [
        "Korzyści z {phrase}",
        "Zalety {phrase}",
        "Dlaczego warto stosować {phrase}?",
        "{phrase} - najważniejsze korzyści"
    ],
    "how_it_works": [
        "Jak działa {phrase}?",
        "Zasada działania {phrase}",
        "{phrase} - jak to funkcjonuje?"
    ],
    "application": [
        "{phrase} - zastosowanie",
        "Gdzie stosować {phrase}?",
        "Praktyczne wykorzystanie {phrase}",
        "{phrase} w praktyce"
    ],
    "how_to": [
        "Jak stworzyć {phrase}?",
        "Jak wybrać {phrase}?",
        "{phrase} krok po kroku",
        "Jak zacząć z {phrase}?"
    ],
    "price": [
        "Cena {phrase} - ile kosztuje?",
        "Ile kosztuje {phrase}?",
        "{phrase} - cennik i koszty",
        "Koszt {phrase}"
    ],
    "comparison": [
        "Porównanie {phrase}",
        "{phrase} - co wybrać?",
        "Najlepsze {phrase} - ranking"
    ],
    "context": [
        "{phrase} - dlaczego jest ważne?",
        "Znaczenie {phrase}",
        "Rola {phrase}"
    ],
    "additional": [
        "{phrase} - wszystko co musisz wiedzieć",
        "{phrase} - kompletny przewodnik",
        "Najważniejsze informacje o {phrase}"
    ],
    "faq": [
        "Najczęściej zadawane pytania o {phrase}",
        "FAQ - {phrase}",
        "{phrase} - pytania i odpowiedzi"
    ]
}


# ================================================================
# MAIN FUNCTIONS
# ================================================================

def generate_h2_plan(
    main_keyword: str,
    h2_phrases: List[str],
    search_intent: str = "informational",
    entities: List[Dict] = None,
    paa_questions: List[str] = None,
    competitor_h2: List[str] = None,
    article_h2_count: int = None
) -> Dict:
    """
    Generuje optymalny plan H2.
    
    Args:
        main_keyword: Główna fraza (H1)
        h2_phrases: Frazy które MUSZĄ być w H2
        search_intent: Intencja wyszukiwania z S1
        entities: Encje z S1
        paa_questions: Pytania PAA z S1
        competitor_h2: H2 konkurencji z S1
        article_h2_count: Ile H2 (jeśli None, oblicza automatycznie)
    
    Returns:
        Dict z h2_plan, h3_suggestions, coverage
    """
    
    entities = entities or []
    paa_questions = paa_questions or []
    competitor_h2 = competitor_h2 or []
    
    # 1. Określ template na podstawie intent
    template = get_intent_template(search_intent)
    
    # 2. Określ ilość H2
    if article_h2_count is None:
        article_h2_count = calculate_h2_count(len(h2_phrases), len(template["structure"]))
    
    # 3. Generuj bazowy plan H2
    h2_plan = []
    used_phrases = []
    
    # 4. Pierwszy H2 - ZAWSZE definicja z główną frazą
    first_h2 = generate_natural_h2(main_keyword, "definition")
    h2_plan.append({
        "position": 1,
        "h2": first_h2,
        "phrase_used": main_keyword,
        "type": "definition",
        "relevancy_score": 100,
        "source": "main_keyword"
    })
    
    # 5. Wpleć frazy użytkownika
    position = 2
    for phrase in h2_phrases:
        if phrase.lower() == main_keyword.lower():
            continue  # Główna fraza już użyta
            
        # Znajdź najlepszy typ H2 dla tej frazy
        h2_type = detect_phrase_type(phrase, template["structure"], position)
        
        # Wygeneruj naturalny nagłówek
        h2_text = generate_natural_h2(phrase, h2_type)
        
        # Oblicz relevancy do H1
        relevancy = calculate_relevancy(h2_text, main_keyword)
        
        h2_plan.append({
            "position": position,
            "h2": h2_text,
            "phrase_used": phrase,
            "type": h2_type,
            "relevancy_score": relevancy,
            "source": "user_phrase"
        })
        used_phrases.append(phrase)
        position += 1
    
    # 6. Uzupełnij strukturę z template (jeśli potrzeba więcej H2)
    remaining_slots = article_h2_count - len(h2_plan)
    if remaining_slots > 0:
        additional_h2s = fill_from_template(
            template, 
            main_keyword, 
            h2_plan, 
            entities, 
            paa_questions,
            remaining_slots
        )
        for h2 in additional_h2s:
            h2["position"] = position
            h2_plan.append(h2)
            position += 1
    
    # 7. Ostatni H2 - FAQ (jeśli nie ma)
    if not any(h["type"] == "faq" for h in h2_plan):
        faq_h2 = generate_natural_h2(main_keyword, "faq")
        h2_plan.append({
            "position": position,
            "h2": faq_h2,
            "phrase_used": None,
            "type": "faq",
            "relevancy_score": 80,
            "source": "template"
        })
    
    # 8. Sortuj po position
    h2_plan = sorted(h2_plan, key=lambda x: x["position"])
    
    # 9. Renumeruj pozycje
    for i, h2 in enumerate(h2_plan, 1):
        h2["position"] = i
    
    # 10. Generuj sugestie H3
    h3_suggestions = generate_h3_suggestions(h2_plan, entities, paa_questions)
    
    # 11. Raport pokrycia
    coverage = generate_coverage_report(h2_plan, h2_phrases, main_keyword)
    
    return {
        "h2_plan": h2_plan,
        "h3_suggestions": h3_suggestions,
        "coverage": coverage,
        "meta": {
            "intent": search_intent,
            "template_used": template["description"],
            "total_h2": len(h2_plan)
        }
    }


def get_intent_template(intent: str) -> Dict:
    """Zwraca template dla danej intencji."""
    # Mapowanie wariantów
    intent_map = {
        "informational": "informational",
        "informacyjny": "informational",
        "how_to": "how_to",
        "poradnik": "how_to",
        "how-to": "how_to",
        "commercial": "commercial",
        "commercial investigation": "commercial",
        "komercyjny": "commercial",
        "transactional": "commercial",
        "listicle": "listicle",
        "lista": "listicle",
        "ranking": "listicle"
    }
    
    normalized = intent_map.get(intent.lower(), "informational")
    return INTENT_TEMPLATES.get(normalized, INTENT_TEMPLATES["informational"])


def calculate_h2_count(phrase_count: int, template_size: int) -> int:
    """Oblicza optymalną ilość H2."""
    # Minimum: frazy + 2 (intro + faq)
    # Maximum: 8
    min_h2 = max(phrase_count + 2, 5)
    max_h2 = 8
    
    return min(max(min_h2, template_size), max_h2)


def detect_phrase_type(phrase: str, structure: List[Dict], position: int) -> str:
    """Wykrywa najlepszy typ H2 dla frazy."""
    phrase_lower = phrase.lower()
    
    # Sprawdź czy fraza zawiera słowa kluczowe dla typu
    for h2_type, keywords in PHRASE_TYPE_HINTS.items():
        for kw in keywords:
            if kw in phrase_lower:
                return h2_type
    
    # Jeśli nie wykryto, dobierz na podstawie pozycji
    position_types = {
        2: "context",      # Druga sekcja - kontekst
        3: "types",        # Trzecia - rodzaje
        4: "benefits",     # Czwarta - korzyści
        5: "application",  # Piąta - zastosowanie
        6: "how_to",       # Szósta - jak to zrobić
        7: "tips"          # Siódma - wskazówki
    }
    
    return position_types.get(position, "additional")


def generate_natural_h2(phrase: str, h2_type: str) -> str:
    """Generuje naturalnie brzmiący nagłówek H2."""
    patterns = H2_PATTERNS.get(h2_type, H2_PATTERNS["additional"])
    
    # Wybierz pierwszy pattern (najprostszy)
    pattern = patterns[0]
    
    # Wstaw frazę
    h2 = pattern.format(phrase=phrase)
    
    # Kapitalizacja pierwszej litery
    h2 = h2[0].upper() + h2[1:] if h2 else h2
    
    # Sprawdź długość (max 60 znaków)
    if len(h2) > 60:
        # Skróć do prostszej formy
        h2 = f"{phrase.capitalize()} - {h2_type_to_polish(h2_type)}"
    
    return h2


def h2_type_to_polish(h2_type: str) -> str:
    """Tłumaczy typ H2 na polski."""
    translations = {
        "definition": "definicja",
        "types": "rodzaje",
        "benefits": "korzyści",
        "how_it_works": "jak działa",
        "application": "zastosowanie",
        "how_to": "poradnik",
        "price": "cena",
        "comparison": "porównanie",
        "context": "znaczenie",
        "additional": "informacje",
        "faq": "FAQ",
        "tips": "wskazówki",
        "mistakes": "błędy"
    }
    return translations.get(h2_type, "informacje")


def calculate_relevancy(h2_text: str, main_keyword: str) -> int:
    """
    Oblicza relevancy H2 do H1 (głównej frazy).
    
    Zasady:
    - 100: H2 zawiera główną frazę
    - 80-99: H2 zawiera część głównej frazy
    - 60-79: H2 semantycznie powiązany
    - <60: Słabe powiązanie
    """
    h2_lower = h2_text.lower()
    kw_lower = main_keyword.lower()
    
    # Pełne dopasowanie
    if kw_lower in h2_lower:
        return 100
    
    # Częściowe dopasowanie (słowa z frazy)
    kw_words = set(kw_lower.split())
    h2_words = set(h2_lower.split())
    common = kw_words.intersection(h2_words)
    
    if common:
        ratio = len(common) / len(kw_words)
        return int(70 + (ratio * 30))
    
    # Podobieństwo tekstu
    similarity = SequenceMatcher(None, h2_lower, kw_lower).ratio()
    return int(50 + (similarity * 30))


def fill_from_template(
    template: Dict,
    main_keyword: str,
    existing_h2: List[Dict],
    entities: List[Dict],
    paa_questions: List[str],
    slots: int
) -> List[Dict]:
    """Uzupełnia plan H2 z template i danych S1."""
    additional = []
    existing_types = {h["type"] for h in existing_h2}
    
    # Dodaj brakujące typy z template
    for item in template["structure"]:
        if len(additional) >= slots:
            break
            
        if item["type"] not in existing_types and item.get("required", False):
            h2_text = generate_natural_h2(main_keyword, item["type"])
            additional.append({
                "h2": h2_text,
                "phrase_used": None,
                "type": item["type"],
                "relevancy_score": 75,
                "source": "template"
            })
            existing_types.add(item["type"])
    
    # Dodaj z PAA jeśli jeszcze są sloty
    for paa in paa_questions[:slots - len(additional)]:
        if len(additional) >= slots:
            break
            
        # PAA jako H2 (pytanie)
        additional.append({
            "h2": paa if paa.endswith("?") else f"{paa}?",
            "phrase_used": None,
            "type": "paa",
            "relevancy_score": 70,
            "source": "paa"
        })
    
    return additional


def generate_h3_suggestions(
    h2_plan: List[Dict],
    entities: List[Dict],
    paa_questions: List[str]
) -> Dict[str, List[str]]:
    """Generuje sugestie H3 dla każdego H2."""
    suggestions = {}
    
    for h2 in h2_plan:
        pos = str(h2.get("position", "0"))
        h2_type = h2.get("type", "")
        
        # H3 sugestie na podstawie typu H2
        if h2_type == "types":
            suggestions[pos] = ["Typ pierwszy", "Typ drugi", "Typ trzeci"]
        elif h2_type == "benefits":
            suggestions[pos] = ["Korzyść 1", "Korzyść 2", "Korzyść 3"]
        elif h2_type == "how_to":
            suggestions[pos] = ["Krok 1", "Krok 2", "Krok 3"]
        elif h2_type == "faq":
            # Użyj PAA jako H3
            suggestions[pos] = paa_questions[:5] if paa_questions else []
        else:
            # Puste - nie zawsze potrzeba H3
            suggestions[pos] = []
    
    return suggestions


def generate_coverage_report(
    h2_plan: List[Dict],
    h2_phrases: List[str],
    main_keyword: str
) -> Dict:
    """Generuje raport pokrycia fraz."""
    used_phrases = [h.get("phrase_used", "") for h in h2_plan if h.get("phrase_used")]
    
    # Sprawdź które frazy użytkownika są pokryte
    phrases_covered = []
    phrases_missing = []
    
    for phrase in h2_phrases:
        phrase_lower = phrase.lower()
        found = False
        
        for used in used_phrases:
            if used and phrase_lower in used.lower():
                found = True
                break
        
        # Sprawdź też w treści H2
        if not found:
            for h2 in h2_plan:
                if phrase_lower in h2.get("h2", "").lower():
                    found = True
                    break
        
        if found:
            phrases_covered.append(phrase)
        else:
            phrases_missing.append(phrase)
    
    return {
        "main_keyword_in_h2": any(main_keyword.lower() in h["h2"].lower() for h in h2_plan),
        "phrases_covered": phrases_covered,
        "phrases_missing": phrases_missing,
        "coverage_percent": int((len(phrases_covered) / len(h2_phrases) * 100)) if h2_phrases else 100,
        "all_phrases_covered": len(phrases_missing) == 0
    }


# ================================================================
# VALIDATION
# ================================================================

def validate_h2_plan(h2_plan: List[Dict], main_keyword: str) -> Dict:
    """Waliduje plan H2 pod kątem Semantic HTML i Content Relevancy."""
    issues = []
    warnings = []
    
    # 1. Sprawdź czy pierwszy H2 zawiera główną frazę
    if h2_plan:
        first_h2_text = h2_plan[0].get("h2", "").lower()
        if main_keyword.lower() not in first_h2_text:
            issues.append("Pierwszy H2 nie zawiera głównej frazy")
    
    # 2. Sprawdź relevancy każdego H2
    for h2 in h2_plan:
        relevancy = h2.get("relevancy_score", 100)  # domyślnie 100 jeśli brak
        if relevancy < 60:
            warnings.append(f"H2 #{h2.get('position', '?')} ma niską relevancy ({relevancy})")
    
    # 3. Sprawdź długość H2
    for h2 in h2_plan:
        h2_text = h2.get("h2", "")
        if len(h2_text) > 60:
            warnings.append(f"H2 #{h2.get('position', '?')} jest za długi ({len(h2_text)} znaków)")
    
    # 4. Sprawdź duplikaty
    h2_texts = [h.get("h2", "").lower() for h in h2_plan]
    if len(h2_texts) != len(set(h2_texts)):
        issues.append("Plan zawiera zduplikowane H2")
    
    # 5. Sprawdź czy jest FAQ na końcu
    if h2_plan and h2_plan[-1].get("type", "") != "faq":
        warnings.append("Ostatni H2 nie jest FAQ")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    # Test
    result = generate_h2_plan(
        main_keyword="pomoce sensoryczne w przedszkolu",
        h2_phrases=["integracja sensoryczna", "ścieżka sensoryczna", "zabawki montessori"],
        search_intent="informational",
        entities=[{"name": "integracja sensoryczna"}, {"name": "przedszkole"}],
        paa_questions=["Ile kosztuje ścieżka sensoryczna?", "Czy integracja jest refundowana?"]
    )
    
    print("\n=== H2 PLAN ===")
    for h2 in result["h2_plan"]:
        print(f"{h2['position']}. {h2['h2']}")
        print(f"   Phrase: {h2.get('phrase_used', 'N/A')} | Type: {h2.get('type', 'N/A')} | Relevancy: {h2.get('relevancy_score', 'N/A')}")
    
    print("\n=== COVERAGE ===")
    print(f"Covered: {result['coverage']['phrases_covered']}")
    print(f"Missing: {result['coverage']['phrases_missing']}")
    print(f"All covered: {result['coverage']['all_phrases_covered']}")
