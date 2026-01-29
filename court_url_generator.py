# court_url_generator.py
# 🆕 v41.4: Generator linków do oficjalnych portali orzeczeń
"""
===============================================================================
🏛️ COURT URL GENERATOR v1.0
===============================================================================

Generuje linki do OFICJALNYCH portali orzeczeń sądowych zamiast SAOS.

Portale:
- orzeczenia.ms.gov.pl - Portal Ministerstwa Sprawiedliwości (główny)
- orzeczenia.[miasto].so.gov.pl - Sądy Okręgowe
- orzeczenia.nsa.gov.pl - NSA
- sn.pl/orzecznictwo - Sąd Najwyższy

===============================================================================
"""

import re
from typing import Dict, Optional, Tuple

# ============================================================================
# MAPOWANIE SĄDÓW → PORTALE
# ============================================================================

# Sądy Okręgowe → subdomeny
COURT_TO_SUBDOMAIN = {
    # Sądy Okręgowe
    "łódź": "lodz",
    "łodzi": "lodz",
    "warszawa": "warszawa",
    "warszawie": "warszawa",
    "warszawa-praga": "warszawa-praga",
    "kraków": "krakow",
    "krakowie": "krakow",
    "poznań": "poznan",
    "poznaniu": "poznan",
    "wrocław": "wroclaw",
    "wrocławiu": "wroclaw",
    "gdańsk": "gdansk",
    "gdańsku": "gdansk",
    "katowice": "katowice",
    "katowicach": "katowice",
    "lublin": "lublin",
    "lublinie": "lublin",
    "szczecin": "szczecin",
    "szczecinie": "szczecin",
    "bydgoszcz": "bydgoszcz",
    "bydgoszczy": "bydgoszcz",
    "białystok": "bialystok",
    "białymstoku": "bialystok",
    "rzeszów": "rzeszow",
    "rzeszowie": "rzeszow",
    "olsztyn": "olsztyn",
    "olsztynie": "olsztyn",
    "opole": "opole",
    "opolu": "opole",
    "kielce": "kielce",
    "kielcach": "kielce",
    "gliwice": "gliwice",
    "gliwicach": "gliwice",
    "częstochowa": "czestochowa",
    "częstochowie": "czestochowa",
    "toruń": "torun",
    "toruniu": "torun",
    "elbląg": "elblag",
    "elblągu": "elblag",
    "legnica": "legnica",
    "legnicy": "legnica",
    "nowy sącz": "nowy-sacz",
    "nowym sączu": "nowy-sacz",
    "płock": "plock",
    "płocku": "plock",
    "radom": "radom",
    "radomiu": "radom",
    "siedlce": "siedlce",
    "siedlcach": "siedlce",
    "sieradz": "sieradz",
    "sieradzu": "sieradz",
    "słupsk": "slupsk",
    "słupsku": "slupsk",
    "suwałki": "suwalki",
    "suwałkach": "suwalki",
    "tarnobrzeg": "tarnobrzeg",
    "tarnobrzegu": "tarnobrzeg",
    "tarnów": "tarnow",
    "tarnowie": "tarnow",
    "zamość": "zamosc",
    "zamościu": "zamosc",
    "zielona góra": "zielona-gora",
    "zielonej górze": "zielona-gora",
}

# Typ sądu → portal
COURT_TYPE_PORTALS = {
    "SUPREME": "http://www.sn.pl/orzecznictwo",
    "ADMINISTRATIVE": "https://orzeczenia.nsa.gov.pl",
    "CONSTITUTIONAL": "https://trybunal.gov.pl/orzecznictwo",
    "COMMON": None,  # Zależy od miasta
}


def extract_city_from_court_name(court_name: str) -> Optional[str]:
    """Wyciąga miasto z nazwy sądu."""
    if not court_name:
        return None
    
    court_lower = court_name.lower()
    
    # Szukaj "w [miasto]" lub "we [miasto]"
    match = re.search(r'\b(?:w|we)\s+(\w+(?:\s+\w+)?)', court_lower)
    if match:
        city = match.group(1).strip()
        return city
    
    return None


def get_court_portal_url(court_name: str, court_type: str = "COMMON") -> Tuple[str, str]:
    """
    Zwraca URL portalu orzeczeń dla danego sądu.
    
    Returns:
        (base_url, portal_type)
        - base_url: URL portalu (bez konkretnego orzeczenia)
        - portal_type: "so" | "sn" | "nsa" | "ms" | "unknown"
    """
    
    # 1. Sprawdź typ sądu
    if court_type in COURT_TYPE_PORTALS and COURT_TYPE_PORTALS[court_type]:
        return COURT_TYPE_PORTALS[court_type], court_type.lower()
    
    # 2. Dla sądów powszechnych - szukaj miasta
    city = extract_city_from_court_name(court_name)
    
    if city:
        # Normalizuj miasto
        city_normalized = city.lower().strip()
        
        # Sprawdź mapowanie
        if city_normalized in COURT_TO_SUBDOMAIN:
            subdomain = COURT_TO_SUBDOMAIN[city_normalized]
            
            # Określ typ sądu (SO, SR, SA)
            court_lower = court_name.lower()
            if "okręgowy" in court_lower or "okręgowego" in court_lower:
                return f"https://orzeczenia.{subdomain}.so.gov.pl", "so"
            elif "apelacyjny" in court_lower or "apelacyjnego" in court_lower:
                return f"https://orzeczenia.{subdomain}.sa.gov.pl", "sa"
            elif "rejonowy" in court_lower or "rejonowego" in court_lower:
                # Sądy rejonowe często na portalu SO
                return f"https://orzeczenia.{subdomain}.so.gov.pl", "sr"
    
    # 3. Fallback - portal MS
    return "https://orzeczenia.ms.gov.pl", "ms"


def generate_search_url(court_name: str, signature: str, court_type: str = "COMMON") -> Dict[str, str]:
    """
    Generuje URL do wyszukania orzeczenia na oficjalnym portalu.
    
    Ponieważ bezpośredni link wymaga znajomości wewnętrznego ID,
    generujemy link do WYSZUKIWARKI z wypełnioną sygnaturą.
    
    Returns:
        {
            "portal_url": "https://orzeczenia.lodz.so.gov.pl",
            "search_url": "https://orzeczenia.lodz.so.gov.pl/search?signature=II+C+895/18",
            "portal_type": "so",
            "instruction": "Wyszukaj sygnaturę: II C 895/18"
        }
    """
    
    portal_url, portal_type = get_court_portal_url(court_name, court_type)
    
    # Przygotuj sygnaturę do URL
    signature_encoded = signature.replace(" ", "+").replace("/", "%2F")
    
    # Generuj URL wyszukiwania (format zależy od portalu)
    if portal_type == "so" or portal_type == "sa" or portal_type == "sr":
        # Portal sądu - wyszukiwarka
        search_url = f"{portal_url}/search?caseNumber={signature_encoded}"
    elif portal_type == "sn":
        # Sąd Najwyższy
        search_url = f"{portal_url}/Sites/orzecznictwo/Orzeczenia3/Szukaj.aspx"
    elif portal_type == "nsa":
        # NSA
        search_url = f"{portal_url}/cbo/query"
    else:
        # Portal MS
        search_url = f"{portal_url}/search?signature={signature_encoded}"
    
    return {
        "portal_url": portal_url,
        "search_url": search_url,
        "portal_type": portal_type,
        "signature": signature,
        "instruction": f"Wyszukaj sygnaturę: {signature}",
        "note": "Link do wyszukiwarki portalu orzeczeń (bezpośredni link wymaga ID orzeczenia)"
    }


def format_judgment_source(
    court_name: str,
    signature: str,
    date: str,
    saos_url: str = None,
    court_type: str = "COMMON"
) -> Dict[str, str]:
    """
    Formatuje pełne źródło orzeczenia w standardzie prawniczym.
    
    🆕 v41.4: Profesjonalne cytowanie - portal + sygnatura, bez pełnego URL.
    Standard prawniczy: czytelnik sam wyszukuje po sygnaturze.
    
    Returns:
        {
            "citation": "wyrok Sądu Okręgowego w Łodzi z dnia 2 października 2019 r., sygn. II C 895/18",
            "official_portal": "orzeczenia.lodz.so.gov.pl",
            "source_note": "(dostępny na: orzeczenia.lodz.so.gov.pl)",
            "full_citation": "wyrok Sądu Okręgowego w Łodzi z dnia 2 października 2019 r., sygn. II C 895/18 (dostępny na: orzeczenia.lodz.so.gov.pl)"
        }
    """
    
    portal_url, portal_type = get_court_portal_url(court_name, court_type)
    
    # Wyciągnij samą domenę (bez https://)
    portal_domain = portal_url.replace("https://", "").replace("http://", "").rstrip("/")
    
    # Formatuj cytowanie w standardzie prawniczym
    # Określ typ orzeczenia i popraw gramatykę (dopełniacz!)
    court_lower = court_name.lower()
    
    # Konwersja na dopełniacz
    court_genitive = court_name
    if "sąd okręgowy" in court_lower:
        court_genitive = court_name.replace("Sąd Okręgowy", "Sądu Okręgowego").replace("sąd okręgowy", "Sądu Okręgowego")
    elif "sąd rejonowy" in court_lower:
        court_genitive = court_name.replace("Sąd Rejonowy", "Sądu Rejonowego").replace("sąd rejonowy", "Sądu Rejonowego")
    elif "sąd apelacyjny" in court_lower:
        court_genitive = court_name.replace("Sąd Apelacyjny", "Sądu Apelacyjnego").replace("sąd apelacyjny", "Sądu Apelacyjnego")
    elif "sąd najwyższy" in court_lower:
        court_genitive = "Sądu Najwyższego"
    elif "naczelny sąd administracyjny" in court_lower or "nsa" in court_lower:
        court_genitive = "Naczelnego Sądu Administracyjnego"
    elif "wojewódzki sąd administracyjny" in court_lower or "wsa" in court_lower:
        court_genitive = court_name.replace("Wojewódzki Sąd Administracyjny", "Wojewódzkiego Sądu Administracyjnego")
    
    if "najwyższy" in court_lower:
        judgment_type = f"wyrok {court_genitive}"
    elif "apelacyjny" in court_lower:
        judgment_type = f"wyrok {court_genitive}"
    elif "okręgowy" in court_lower:
        judgment_type = f"wyrok {court_genitive}"
    elif "rejonowy" in court_lower:
        judgment_type = f"wyrok {court_genitive}"
    elif "administracyjny" in court_lower or "nsa" in court_lower.replace(" ", ""):
        judgment_type = f"wyrok {court_genitive}"
    else:
        judgment_type = f"orzeczenie {court_genitive}"
    
    citation = f"{judgment_type} z dnia {date}, sygn. {signature}"
    source_note = f"(dostępny na: {portal_domain})"
    full_citation = f"{citation} {source_note}"
    
    return {
        "citation": citation,
        "official_portal": portal_domain,  # Tylko domena, nie pełny URL
        "portal_url": portal_url,  # Pełny URL do portalu (nie do orzeczenia)
        "source_note": source_note,
        "full_citation": full_citation,
        "signature": signature,
        "portal_type": portal_type,
        # Instrukcja dla GPT
        "citation_instruction": f"Cytuj jako: \"{citation}\" i dodaj {source_note}"
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test
    result = format_judgment_source(
        court_name="Sąd Okręgowy w Łodzi",
        signature="II C 895/18",
        date="2 października 2019 r.",
        saos_url="https://www.saos.org.pl/judgments/123456"
    )
    
    print("=" * 60)
    print("TEST: Sąd Okręgowy w Łodzi, II C 895/18")
    print("=" * 60)
    for k, v in result.items():
        print(f"{k}: {v}")
    
    print("")
    print("=" * 60)
    print("PRZYKŁAD CYTOWANIA W ARTYKULE:")
    print("=" * 60)
    print(result.get("full_citation", ""))
    print("")
    print("LUB:")
    print(f"\"...jak wskazał {result.get('citation', '')}...\"")
    print(f"\"{result.get('source_note', '')}\"")
