"""
===============================================================================
📚 CONTEXTUAL SYNONYMS v41.0 - Rozszerzona mapa synonimów
===============================================================================

Rozszerzenie istniejącej mapy CONTEXTUAL_SYNONYMS z dynamic_humanization.py.

ZASADY:
1. Tylko PEWNE synonimy - sprawdzone w słownikach/plWordNet
2. Synonimy muszą pasować do kontekstu SEO/content writing
3. Brak kalki z angielskiego (chyba że są naturalne w polskim)
4. Grupowanie według kategorii dla łatwiejszego zarządzania

ŹRÓDŁA:
- plWordNet (195k+ lemmatów)
- Słownik Języka Polskiego PWN
- Praktyka SEO copywritingu

===============================================================================
"""

from typing import Dict, List


# ============================================================================
# ROZSZERZONA MAPA CONTEXTUAL_SYNONYMS v41
# ============================================================================

CONTEXTUAL_SYNONYMS_V41: Dict[str, List[str]] = {
    
    # ========================================================================
    # CZASOWNIKI - najczęściej powtarzane
    # ========================================================================
    
    # Możliwość/konieczność
    "można": ["da się", "istnieje możliwość", "jest opcja", "jest możliwe"],
    "należy": ["trzeba", "wymaga się", "konieczne jest", "powinno się"],
    "trzeba": ["należy", "konieczne jest", "wymaga się", "niezbędne jest"],
    "musi": ["powinien", "jest zobowiązany", "ma obowiązek"],
    "powinien": ["ma obowiązek", "zaleca się aby", "wskazane jest aby"],
    
    # Wymaganie/potrzeba
    "wymaga": ["potrzebuje", "niezbędne jest", "konieczne jest", "zakłada"],
    "potrzebuje": ["wymaga", "niezbędne mu jest", "konieczne jest"],
    
    # Umożliwianie
    "pozwala": ["umożliwia", "daje możliwość", "otwiera drogę do", "sprawia że można"],
    "umożliwia": ["pozwala", "daje szansę na", "stwarza warunki do"],
    "zapewnia": ["gwarantuje", "daje", "oferuje", "dostarcza"],
    
    # Odnoszenie się
    "dotyczy": ["odnosi się do", "obejmuje", "tyczy się", "wiąże się z"],
    "obejmuje": ["zawiera", "uwzględnia", "składa się z", "dotyczy"],
    
    # Bycie/stanowienie
    "stanowi": ["jest", "reprezentuje", "tworzy", "pełni funkcję"],
    "jest": ["stanowi", "bywa", "okazuje się"],
    
    # Oferowanie
    "oferuje": ["proponuje", "udostępnia", "daje", "zapewnia"],
    "proponuje": ["oferuje", "sugeruje", "przedstawia"],
    
    # Działanie/wykonywanie
    "wykonuje": ["realizuje", "przeprowadza", "robi", "dokonuje"],
    "przeprowadza": ["wykonuje", "realizuje", "prowadzi"],
    "prowadzi": ["realizuje", "wykonuje", "kieruje"],
    
    # Otrzymywanie
    "otrzymuje": ["dostaje", "uzyskuje", "nabywa"],
    "uzyskuje": ["otrzymuje", "zdobywa", "osiąga"],
    
    # Wpływanie
    "wpływa": ["oddziałuje", "ma wpływ", "determinuje", "kształtuje"],
    "powoduje": ["wywołuje", "sprawia", "skutkuje"],
    "skutkuje": ["powoduje", "prowadzi do", "wywołuje"],
    
    # ========================================================================
    # PRZYMIOTNIKI - łatwe do nadużycia
    # ========================================================================
    
    # Ważność
    "ważny": ["istotny", "znaczący", "kluczowy", "zasadniczy"],
    "istotny": ["ważny", "znaczący", "kluczowy", "doniosły"],
    "znaczący": ["istotny", "ważny", "doniosły", "niemały"],
    
    # Jakość
    "dobry": ["skuteczny", "wartościowy", "odpowiedni", "właściwy", "trafny"],
    "skuteczny": ["efektywny", "działający", "sprawdzony", "wydajny"],
    "właściwy": ["odpowiedni", "stosowny", "adekwatny", "prawidłowy"],
    "odpowiedni": ["właściwy", "stosowny", "adekwatny", "pasujący"],
    
    # Rozmiar/skala
    "duży": ["znaczny", "pokaźny", "spory", "niemały", "wysoki"],
    "mały": ["niewielki", "drobny", "skromny", "ograniczony", "niski"],
    "wysoki": ["znaczny", "duży", "pokaźny", "ponadprzeciętny"],
    "niski": ["niewielki", "mały", "ograniczony", "minimalny"],
    
    # Szybkość
    "szybki": ["sprawny", "błyskawiczny", "niezwłoczny", "prędki", "ekspresowy"],
    "wolny": ["powolny", "stopniowy", "niespiesznie"],
    
    # Cena
    "tani": ["ekonomiczny", "przystępny cenowo", "budżetowy", "niedrogi"],
    "drogi": ["kosztowny", "wysoki cenowo", "premium", "cenny"],
    
    # Nowość
    "nowy": ["świeży", "najnowszy", "aktualny", "niedawny", "współczesny"],
    "stary": ["wcześniejszy", "poprzedni", "dawny", "dotychczasowy"],
    
    # Trudność
    "trudny": ["wymagający", "skomplikowany", "złożony", "niełatwy"],
    "łatwy": ["prosty", "nieskomplikowany", "przystępny", "bezproblemowy"],
    "prosty": ["łatwy", "nieskomplikowany", "klarowny", "zrozumiały"],
    
    # Profesjonalizm
    "profesjonalny": ["fachowy", "wykwalifikowany", "doświadczony", "kompetentny"],
    "doświadczony": ["wprawiony", "praktykowany", "biegły", "wytrawny"],
    
    # ========================================================================
    # RZECZOWNIKI - kontekstowe
    # ========================================================================
    
    # Osoby
    "osoba": ["człowiek", "jednostka", "ktoś", "zainteresowany"],
    "człowiek": ["osoba", "jednostka", "istota"],
    "klient": ["odbiorca", "zamawiający", "kupujący", "kontrahent"],
    "specjalista": ["ekspert", "fachowiec", "znawca", "profesjonalista"],
    
    # Sprawy/kwestie
    "sprawa": ["kwestia", "zagadnienie", "przypadek", "temat"],
    "kwestia": ["sprawa", "zagadnienie", "problem", "temat"],
    "problem": ["trudność", "kłopot", "wyzwanie", "kwestia"],
    
    # Metody/sposoby
    "sposób": ["metoda", "forma", "droga", "technika"],
    "metoda": ["sposób", "technika", "procedura", "podejście"],
    "rozwiązanie": ["sposób", "metoda", "remedium", "odpowiedź"],
    
    # Procesy
    "proces": ["procedura", "przebieg", "tok", "postępowanie"],
    "procedura": ["proces", "tryb", "postępowanie", "kolejność"],
    "etap": ["faza", "stadium", "krok", "okres"],
    
    # Warunki
    "warunek": ["wymóg", "kryterium", "przesłanka", "okoliczność"],
    "wymóg": ["warunek", "kryterium", "wymaganie", "nakaz"],
    
    # Korzyści
    "korzyść": ["zaleta", "atut", "plus", "wartość dodana", "pożytek"],
    "zaleta": ["korzyść", "atut", "plus", "mocna strona"],
    "wada": ["minus", "słaba strona", "niedostatek", "usterka"],
    
    # Rezultaty
    "wynik": ["rezultat", "efekt", "skutek", "następstwo"],
    "efekt": ["wynik", "rezultat", "skutek", "konsekwencja"],
    "skutek": ["efekt", "wynik", "następstwo", "konsekwencja"],
    
    # Cel
    "cel": ["zamiar", "intencja", "dążenie", "plan"],
    
    # Informacje
    "informacja": ["wiadomość", "dane", "wskazówka", "komunikat"],
    "dane": ["informacje", "szczegóły", "fakty"],
    
    # ========================================================================
    # FRAZY DO ZAMIANY (prepozycjonalne)
    # ========================================================================
    
    "w przypadku": ["gdy", "jeśli", "kiedy", "w razie"],
    "w celu": ["aby", "żeby", "dla"],
    "ze względu na": ["z powodu", "przez", "wskutek", "z racji"],
    "w kontekście": ["przy", "podczas", "w ramach", "odnośnie"],
    "pod względem": ["jeśli chodzi o", "w kwestii", "w aspekcie"],
    "na rzecz": ["dla", "w interesie", "na korzyść"],
    
    # ========================================================================
    # PRAWNICZE (rozszerzenie dla YMYL)
    # ========================================================================
    
    "sąd": ["organ sądowy", "instancja", "trybunał"],
    "wyrok": ["orzeczenie", "rozstrzygnięcie", "decyzja", "werdykt"],
    "pozew": ["wniosek", "pismo procesowe", "powództwo"],
    "strona": ["uczestnik", "podmiot", "interesant"],
    "prawo": ["przepisy", "regulacje", "normy prawne", "ustawodawstwo"],
    "ustawa": ["akt prawny", "regulacja", "przepisy"],
    "przepis": ["regulacja", "norma", "zasada prawna"],
    "kara": ["sankcja", "grzywna", "konsekwencja prawna"],
    "obowiązek": ["powinność", "zobowiązanie", "nakaz"],
    "uprawnienie": ["prawo", "możliwość", "prerogatywa"],
    
    # ========================================================================
    # MEDYCZNE (rozszerzenie dla YMYL)
    # ========================================================================
    
    "choroba": ["schorzenie", "dolegliwość", "przypadłość", "jednostka chorobowa"],
    "leczenie": ["terapia", "kuracja", "postępowanie lecznicze"],
    "pacjent": ["chory", "osoba leczona", "podopieczny"],
    "lekarz": ["specjalista", "medyk", "klinicysta"],
    "badanie": ["diagnostyka", "testy", "analiza"],
    "objaw": ["symptom", "oznaka", "manifestacja"],
    "lek": ["preparat", "medykament", "środek farmaceutyczny"],
    
    # ========================================================================
    # FINANSOWE (rozszerzenie dla YMYL)
    # ========================================================================
    
    "koszt": ["wydatek", "nakład", "cena", "opłata"],
    "opłata": ["należność", "koszt", "taksa", "prowizja"],
    "cena": ["koszt", "wartość", "kwota"],
    "pieniądze": ["środki", "finanse", "kapitał", "fundusze"],
    "kredyt": ["pożyczka", "finansowanie", "zobowiązanie"],
    "rata": ["spłata", "płatność", "należność"],
    "zysk": ["dochód", "zarobek", "przychód", "korzyść finansowa"],
    "strata": ["uszczerbek", "szkoda finansowa", "deficyt"],
}


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def get_synonyms_v41(word: str, max_count: int = 5) -> List[str]:
    """
    Zwraca synonimy dla słowa z rozszerzonej mapy.
    
    Args:
        word: Słowo do znalezienia synonimów
        max_count: Maksymalna liczba synonimów do zwrócenia
        
    Returns:
        Lista synonimów (może być pusta)
    """
    word_lower = word.lower().strip()
    synonyms = CONTEXTUAL_SYNONYMS_V41.get(word_lower, [])
    return synonyms[:max_count]


def get_synonyms_batch_v41(words: List[str], max_per_word: int = 3) -> Dict[str, List[str]]:
    """
    Zwraca synonimy dla wielu słów naraz.
    
    Args:
        words: Lista słów
        max_per_word: Max synonimów na słowo
        
    Returns:
        Dict {słowo: [synonimy]}
    """
    result = {}
    for word in words:
        synonyms = get_synonyms_v41(word, max_per_word)
        if synonyms:
            result[word] = synonyms
    return result


def suggest_replacement_v41(word: str, context: str = "") -> Dict:
    """
    Sugeruje zamiennik dla często powtarzanego słowa.
    
    Kompatybilne z istniejącym API dynamic_humanization.py
    
    Args:
        word: Słowo do zamiany
        context: Opcjonalny kontekst (nieużywany w tej wersji)
        
    Returns:
        Dict z sugestiami
    """
    synonyms = get_synonyms_v41(word, max_count=3)
    
    return {
        "word": word,
        "suggestions": synonyms,
        "count": len(synonyms),
        "source": "contextual_synonyms_v41"
    }


def get_stats_v41() -> Dict:
    """Zwraca statystyki mapy synonimów."""
    total_synonyms = sum(len(v) for v in CONTEXTUAL_SYNONYMS_V41.values())
    
    return {
        "words_count": len(CONTEXTUAL_SYNONYMS_V41),
        "total_synonyms": total_synonyms,
        "avg_synonyms_per_word": round(total_synonyms / len(CONTEXTUAL_SYNONYMS_V41), 1),
        "version": "41.0"
    }


# ============================================================================
# KATEGORIE (dla raportowania)
# ============================================================================

SYNONYM_CATEGORIES = {
    "czasowniki": [
        "można", "należy", "trzeba", "musi", "powinien",
        "wymaga", "potrzebuje", "pozwala", "umożliwia", "zapewnia",
        "dotyczy", "obejmuje", "stanowi", "jest", "oferuje", "proponuje",
        "wykonuje", "przeprowadza", "prowadzi", "otrzymuje", "uzyskuje",
        "wpływa", "powoduje", "skutkuje"
    ],
    "przymiotniki": [
        "ważny", "istotny", "znaczący", "dobry", "skuteczny",
        "właściwy", "odpowiedni", "duży", "mały", "wysoki", "niski",
        "szybki", "wolny", "tani", "drogi", "nowy", "stary",
        "trudny", "łatwy", "prosty", "profesjonalny", "doświadczony"
    ],
    "rzeczowniki": [
        "osoba", "człowiek", "klient", "specjalista",
        "sprawa", "kwestia", "problem", "sposób", "metoda", "rozwiązanie",
        "proces", "procedura", "etap", "warunek", "wymóg",
        "korzyść", "zaleta", "wada", "wynik", "efekt", "skutek", "cel",
        "informacja", "dane"
    ],
    "frazy": [
        "w przypadku", "w celu", "ze względu na", "w kontekście",
        "pod względem", "na rzecz"
    ],
    "prawnicze": [
        "sąd", "wyrok", "pozew", "strona", "prawo", "ustawa",
        "przepis", "kara", "obowiązek", "uprawnienie"
    ],
    "medyczne": [
        "choroba", "leczenie", "pacjent", "lekarz", "badanie",
        "objaw", "lek"
    ],
    "finansowe": [
        "koszt", "opłata", "cena", "pieniądze", "kredyt", "rata",
        "zysk", "strata"
    ]
}


def get_category_stats() -> Dict[str, int]:
    """Zwraca liczbę słów w każdej kategorii."""
    return {cat: len(words) for cat, words in SYNONYM_CATEGORIES.items()}


# ============================================================================
# INTEGRACJA Z DYNAMIC_HUMANIZATION.PY
# ============================================================================

"""
INTEGRACJA:

1. W dynamic_humanization.py, zamień CONTEXTUAL_SYNONYMS na import:

   from contextual_synonyms_v41 import (
       CONTEXTUAL_SYNONYMS_V41 as CONTEXTUAL_SYNONYMS,
       get_synonyms_v41,
       get_synonyms_batch_v41
   )

2. Zamień get_synonyms_for_word() na:

   def get_synonyms_for_word(word: str, context: str = "") -> List[str]:
       # Najpierw lokalna mapa v41
       synonyms = get_synonyms_v41(word, max_count=5)
       if synonyms:
           return synonyms
       
       # Fallback do synonym_service (plWordNet)
       if SYNONYM_SERVICE_AVAILABLE:
           result = _get_synonyms_external(word, context=context)
           return result.get("synonyms", [])[:5]
       
       return []

3. Funkcja get_synonyms_batch() pozostaje bez zmian - używa tych samych źródeł.
"""


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    stats = get_stats_v41()
    print(f"📚 CONTEXTUAL SYNONYMS v41 Statistics:")
    print(f"   Words: {stats['words_count']}")
    print(f"   Total synonyms: {stats['total_synonyms']}")
    print(f"   Avg per word: {stats['avg_synonyms_per_word']}")
    
    print(f"\n📊 Categories:")
    for cat, count in get_category_stats().items():
        print(f"   {cat}: {count} słów")
    
    print(f"\n🧪 Test examples:")
    test_words = ["ważny", "można", "sąd", "koszt", "w przypadku"]
    for word in test_words:
        syns = get_synonyms_v41(word)
        print(f"   {word} → {syns}")
