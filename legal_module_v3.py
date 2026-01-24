"""
LEGAL MODULE v3.0 - BRAJEN SEO Engine
Moduł wykrywania i obsługi treści prawnych (YMYL).

Funkcje:
- detect_category: Wykrywa kategorię treści (prawo, finanse, zdrowie, etc.)
- is_legal_topic: Sprawdza czy temat jest prawny
- get_legal_sources: Pobiera źródła prawne (SAOS, ustawy)
- enhance_with_legal_context: Wzbogaca projekt o kontekst prawny

Autor: BRAJEN SEO Engine v36.3
"""

import re
import os
from typing import Dict, List, Optional, Tuple

# ================================================================
# CATEGORY DETECTION
# ================================================================

# Słowa kluczowe dla kategorii PRAWO
LEGAL_KEYWORDS = [
    # Instytucje prawne
    "ubezwłasnowolnienie", "ubezwłasnowolnić", "ubezwłasnowolnienia",
    "testament", "spadek", "dziedziczenie", "zachowek",
    "rozwód", "alimenty", "separacja", "władza rodzicielska",
    "umowa", "kontrakt", "zobowiązanie", "wierzytelność",
    "odszkodowanie", "zadośćuczynienie", "szkoda",
    "kredyt", "hipoteka", "zastaw", "poręczenie",
    "spółka", "firma", "działalność gospodarcza",
    "wykroczenie", "przestępstwo", "kara", "wyrok",
    
    # Procedury
    "postępowanie sądowe", "pozew", "apelacja", "kasacja",
    "wniosek", "skarga", "odwołanie", "zażalenie",
    "mediacja", "arbitraż", "ugoda",
    
    # Podmioty prawne
    "sąd", "prokurator", "adwokat", "radca prawny",
    "notariusz", "komornik", "kurator", "biegły",
    "opiekun prawny", "przedstawiciel ustawowy",
    
    # Akty prawne
    "kodeks cywilny", "kodeks karny", "kodeks pracy",
    "ustawa", "rozporządzenie", "przepis", "artykuł",
    "prawo", "regulacja", "norma prawna",
    
    # Pojęcia prawne
    "zdolność prawna", "zdolność do czynności prawnych",
    "osoba fizyczna", "osoba prawna",
    "pełnomocnictwo", "reprezentacja",
    "przedawnienie", "termin", "zasiedzenie",
    "własność", "posiadanie", "użytkowanie",
    "służebność", "hipoteka", "zastaw"
]

# Słowa kluczowe dla kategorii FINANSE
FINANCE_KEYWORDS = [
    "inwestycja", "inwestować", "portfel",
    "akcje", "obligacje", "fundusze",
    "giełda", "rynek finansowy", "trading",
    "emerytura", "ike", "ikze", "ppk",
    "podatek", "pit", "vat", "cit",
    "księgowość", "bilans", "rachunek",
    "bank", "bankowość", "konto",
    "ubezpieczenie", "polisa", "składka"
]

# Słowa kluczowe dla kategorii ZDROWIE
HEALTH_KEYWORDS = [
    "choroba", "leczenie", "terapia",
    "lekarz", "szpital", "przychodnia",
    "lek", "dawkowanie", "recepta",
    "diagnoza", "objawy", "symptomy",
    "operacja", "zabieg", "rehabilitacja",
    "dieta", "odżywianie", "suplementy",
    "psycholog", "psychiatra", "psychoterapia",
    "depresja", "lęk", "zaburzenia"
]


def detect_category(text: str, topic: str = "") -> Dict:
    """
    Wykryj kategorię treści na podstawie tekstu i tematu.
    
    Args:
        text: Tekst do analizy (może być pusty)
        topic: Temat/fraza główna
        
    Returns:
        Dict z kategorią i pewnością:
        {
            "category": "prawo" | "finanse" | "zdrowie" | "general",
            "confidence": float (0.0-1.0),
            "is_ymyl": bool,
            "detected_keywords": List[str],
            "recommendations": List[str]
        }
    """
    combined_text = f"{topic} {text}".lower()
    
    # Zlicz dopasowania dla każdej kategorii
    legal_matches = []
    finance_matches = []
    health_matches = []
    
    for kw in LEGAL_KEYWORDS:
        if kw.lower() in combined_text:
            legal_matches.append(kw)
    
    for kw in FINANCE_KEYWORDS:
        if kw.lower() in combined_text:
            finance_matches.append(kw)
    
    for kw in HEALTH_KEYWORDS:
        if kw.lower() in combined_text:
            health_matches.append(kw)
    
    # Określ dominującą kategorię
    scores = {
        "prawo": len(legal_matches),
        "finanse": len(finance_matches),
        "zdrowie": len(health_matches)
    }
    
    max_category = max(scores, key=scores.get)
    max_score = scores[max_category]
    
    # Oblicz pewność
    total_keywords = sum(scores.values())
    if total_keywords == 0:
        confidence = 0.0
        category = "general"
        detected = []
    else:
        confidence = min(1.0, max_score / 5)  # 5+ dopasowań = 100% pewności
        category = max_category if max_score >= 2 else "general"
        detected = {
            "prawo": legal_matches,
            "finanse": finance_matches,
            "zdrowie": health_matches
        }.get(category, [])
    
    # YMYL check
    is_ymyl = category in ["prawo", "finanse", "zdrowie"] and confidence >= 0.4
    
    # Rekomendacje
    recommendations = []
    if category == "prawo":
        recommendations = [
            "Używaj precyzyjnej terminologii prawnej",
            "Odwołuj się do konkretnych przepisów (np. art. X KC)",
            "Zachowaj formalny, profesjonalny ton",
            "Unikaj porad prawnych - opisuj stan prawny",
            "Rozważ dodanie cytatów z orzeczeń sądowych"
        ]
    elif category == "finanse":
        recommendations = [
            "Podawaj aktualne dane i statystyki",
            "Ostrzegaj przed ryzykiem inwestycyjnym",
            "Nie dawaj konkretnych porad inwestycyjnych",
            "Zachowaj obiektywizm"
        ]
    elif category == "zdrowie":
        recommendations = [
            "Odwołuj się do źródeł medycznych",
            "Zachęcaj do konsultacji z lekarzem",
            "Unikaj diagnozowania",
            "Opisuj fakty, nie dawaj porad medycznych"
        ]
    
    return {
        "category": category,
        "confidence": round(confidence, 2),
        "is_ymyl": is_ymyl,
        "detected_keywords": detected[:10],  # Max 10
        "recommendations": recommendations,
        "scores": scores
    }


def is_legal_topic(topic: str, text: str = "") -> bool:
    """
    Sprawdź czy temat jest prawny.
    
    Args:
        topic: Temat/fraza główna
        text: Dodatkowy tekst do analizy
        
    Returns:
        True jeśli temat jest prawny
    """
    result = detect_category(text, topic)
    return result["category"] == "prawo" and result["confidence"] >= 0.4


def get_legal_context_for_prompt(topic: str) -> str:
    """
    Generuj kontekst prawny do wstrzyknięcia w prompt GPT.
    
    Args:
        topic: Temat artykułu
        
    Returns:
        Sekcja promptu z kontekstem prawnym
    """
    result = detect_category("", topic)
    
    if result["category"] != "prawo":
        return ""
    
    lines = [
        "",
        "=" * 60,
        "⚖️ KONTEKST PRAWNY (YMYL)",
        "=" * 60,
        "",
        "Ten artykuł dotyczy tematyki PRAWNEJ. Przestrzegaj zasad:",
        "",
        "1. TERMINOLOGIA:",
        "   - Używaj precyzyjnych terminów prawnych",
        "   - Definiuj pojęcia przy pierwszym użyciu",
        "   - Odwołuj się do aktów prawnych (KC, KPC, KK)",
        "",
        "2. ŹRÓDŁA:",
        "   - Cytuj przepisy (np. 'zgodnie z art. 13 KC')",
        "   - Możesz odwołać się do orzeczeń sądowych",
        "   - Unikaj podawania konkretnych porad prawnych",
        "",
        "3. TON:",
        "   - Formalny, profesjonalny",
        "   - Obiektywny (opisuj stan prawny, nie oceniaj)",
        "   - Bezosobowy (unikaj 'ty', 'Państwo')",
        "",
        "4. ZASTRZEŻENIA:",
        "   - NIE pisz 'skonsultuj się z prawnikiem' w każdym akapicie",
        "   - Jedno zastrzeżenie w intro lub outro wystarczy",
        "",
    ]
    
    if result["detected_keywords"]:
        lines.append(f"Wykryte pojęcia prawne: {', '.join(result['detected_keywords'][:5])}")
        lines.append("")
    
    return "\n".join(lines)


# ================================================================
# SAOS INTEGRATION (placeholder)
# ================================================================

def search_saos_judgments(query: str, max_results: int = 3) -> List[Dict]:
    """
    Wyszukaj orzeczenia w bazie SAOS (Sąd Najwyższy, NSA).
    
    TODO: Zaimplementować faktyczne połączenie z API SAOS
    
    Args:
        query: Zapytanie do wyszukania
        max_results: Maksymalna liczba wyników
        
    Returns:
        Lista orzeczeń
    """
    # Placeholder - w przyszłości połączyć z SAOS API
    return []


def get_relevant_law_articles(topic: str) -> List[Dict]:
    """
    Pobierz relevantne artykuły ustaw dla tematu.
    
    TODO: Zaimplementować bazę przepisów
    
    Args:
        topic: Temat do wyszukania
        
    Returns:
        Lista artykułów
    """
    # Placeholder
    return []


# ================================================================
# PROJECT ENHANCEMENT
# ================================================================

def enhance_project_with_legal(project_data: Dict, main_keyword: str, h2_list: List[str]) -> Dict:
    """
    Wzbogać dane projektu o kontekst prawny.
    
    Args:
        project_data: Dane projektu
        main_keyword: Fraza główna
        h2_list: Lista nagłówków H2
        
    Returns:
        Rozszerzone project_data
    """
    # Wykryj kategorię
    h2_text = " ".join(h2_list) if h2_list else ""
    category_info = detect_category(h2_text, main_keyword)
    
    project_data["content_category"] = category_info
    
    if category_info["category"] == "prawo":
        project_data["legal_context"] = {
            "is_legal": True,
            "confidence": category_info["confidence"],
            "detected_terms": category_info["detected_keywords"],
            "tone": "formal",
            "style": "bezosobowy",
            "citations_recommended": True,
            "disclaimer_required": True
        }
        
        # Dodaj instrukcję prawną
        project_data["legal_instruction"] = get_legal_context_for_prompt(main_keyword)
    
    return project_data


# ================================================================
# EXPORT
# ================================================================

# Funkcje eksportowane dla innych modułów
LEGAL_MODULE_ENABLED = True

__all__ = [
    "detect_category",
    "is_legal_topic",
    "get_legal_context_for_prompt",
    "enhance_project_with_legal",
    "search_saos_judgments",
    "get_relevant_law_articles",
    "LEGAL_MODULE_ENABLED"
]


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("=== LEGAL MODULE v3.0 TEST ===\n")
    
    # Test detekcji kategorii
    test_topics = [
        "Ubezwłasnowolnienie osoby chorej psychicznie",
        "Jak napisać testament",
        "Najlepsze fundusze inwestycyjne 2024",
        "Objawy depresji u nastolatków",
        "Przepis na ciasto czekoladowe"
    ]
    
    for topic in test_topics:
        result = detect_category("", topic)
        print(f"'{topic}'")
        print(f"  → Kategoria: {result['category']} ({result['confidence']*100:.0f}%)")
        print(f"  → YMYL: {result['is_ymyl']}")
        print(f"  → Słowa kluczowe: {result['detected_keywords'][:3]}")
        print()
    
    # Test kontekstu prawnego
    print("=" * 50)
    print("KONTEKST PRAWNY DLA PROMPTU:")
    print("=" * 50)
    context = get_legal_context_for_prompt("Ubezwłasnowolnienie osoby chorej psychicznie")
    print(context)
