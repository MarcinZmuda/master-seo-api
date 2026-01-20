# saos_client.py
# BRAJEN Legal Module - Klient SAOS API v3.1
# Z pełną treścią do scoringu + FILTROWANIE SYGNATUR

"""
===============================================================================
🏛️ SAOS CLIENT v3.1
===============================================================================

Klient do System Analizy Orzeczeń Sądowych (SAOS).
https://www.saos.org.pl/api

Zmiany w v3.1:
- 🆕 Filtrowanie po SYGNATURZE (wydział C/K/U)
- 🆕 Wykrywanie przedmiotu sprawy vs kontekst uboczny

Zmiany w v3:
- Zwraca full_text do scoringu
- Lepsze wyciąganie fragmentów

===============================================================================
"""

import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class SAOSConfig:
    BASE_URL: str = "https://www.saos.org.pl/api"
    SEARCH_ENDPOINT: str = "/search/judgments"
    JUDGMENT_ENDPOINT: str = "/judgments"
    DEFAULT_PAGE_SIZE: int = 15
    MAX_PAGE_SIZE: int = 25
    DEFAULT_MIN_YEAR: int = 2020
    TIMEOUT: int = 15
    
    COURT_TYPES = {
        "COMMON": "Sądy Powszechne",
        "SUPREME": "Sąd Najwyższy",
        "ADMINISTRATIVE": "Sądy Administracyjne",
        "CONSTITUTIONAL": "Trybunał Konstytucyjny",
        "NATIONAL_APPEAL_CHAMBER": "Krajowa Izba Odwoławcza"
    }
    
    # 🆕 v3.1: MAPOWANIE WYDZIAŁÓW - które sygnatury dla jakich tematów
    # Sygnatura zawiera literę wydziału: C=cywilny, K=karny, U=ubezpieczenia, P=pracy
    DIVISION_CODES: Dict[str, List[str]] = field(default_factory=lambda: {
        "cywilne": ["C", "Ca", "ACa", "Cz", "ACz", "CZP", "CSK", "CNP"],  # Cywilne
        "rodzinne": ["C", "Ca", "ACa", "RC", "RCa", "CZP"],  # Rodzinne = też cywilne
        "karne": ["K", "Ka", "AKa", "Kz", "AKz", "KZP", "KK"],  # Karne
        "pracy": ["P", "Pa", "APa", "Pz", "APz", "PZP"],  # Prawo pracy
        "ubezpieczenia": ["U", "Ua", "AUa", "Uz", "AUz", "UZP"],  # Ubezpieczenia społeczne
        "administracyjne": ["SA", "OSA", "GSK", "NSA", "OSK"]  # Administracyjne
    })
    
    # 🆕 v3.1: MAPOWANIE TEMAT → DOZWOLONE WYDZIAŁY
    TOPIC_TO_DIVISIONS: Dict[str, List[str]] = field(default_factory=lambda: {
        # Prawo rodzinne → TYLKO cywilne/rodzinne
        "alimenty": ["cywilne", "rodzinne"],
        "rozwód": ["cywilne", "rodzinne"],
        "separacja": ["cywilne", "rodzinne"],
        "opieka nad dzieckiem": ["cywilne", "rodzinne"],
        "władza rodzicielska": ["cywilne", "rodzinne"],
        "ubezwłasnowolnienie": ["cywilne", "rodzinne"],  # 🆕 KLUCZOWE!
        "kuratela": ["cywilne", "rodzinne"],
        "przysposobienie": ["cywilne", "rodzinne"],
        "adopcja": ["cywilne", "rodzinne"],
        
        # Prawo spadkowe → TYLKO cywilne
        "spadek": ["cywilne"],
        "testament": ["cywilne"],
        "dziedziczenie": ["cywilne"],
        "zachowek": ["cywilne"],
        
        # Prawo cywilne → TYLKO cywilne
        "umowa": ["cywilne"],
        "odszkodowanie": ["cywilne"],
        "zadośćuczynienie": ["cywilne"],
        "nieruchomość": ["cywilne"],
        "służebność": ["cywilne"],
        "hipoteka": ["cywilne"],
        
        # Prawo pracy → TYLKO pracy/cywilne
        "wypowiedzenie": ["pracy", "cywilne"],
        "mobbing": ["pracy", "cywilne"],
        "wynagrodzenie": ["pracy"],
        "zwolnienie": ["pracy"],
        
        # Prawo karne → TYLKO karne
        "przestępstwo": ["karne"],
        "kara": ["karne"],
        "oskarżenie": ["karne"],
        "jazda po alkoholu": ["karne"],
        "kradzież": ["karne"],
        "niealimentacja": ["karne"],  # Art. 209 KK
        
        # Ubezpieczenia społeczne → TYLKO ubezpieczenia
        "emerytura": ["ubezpieczenia"],
        "renta": ["ubezpieczenia"],
        "zasiłek": ["ubezpieczenia"],
        "niezdolność do pracy": ["ubezpieczenia"],
    })


CONFIG = SAOSConfig()


# ============================================================================
# KLIENT SAOS
# ============================================================================

class SAOSClient:
    """Klient do komunikacji z SAOS API."""
    
    def __init__(self):
        self.base_url = CONFIG.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BRAJEN-Legal-Module/3.0"
        })
    
    def search_judgments(
        self,
        keyword: str,
        court_type: Optional[str] = None,
        page_size: int = CONFIG.DEFAULT_PAGE_SIZE,
        min_year: Optional[int] = CONFIG.DEFAULT_MIN_YEAR,
        sorting_field: str = "JUDGMENT_DATE",
        sorting_direction: str = "DESC"
    ) -> Dict[str, Any]:
        """Wyszukuje orzeczenia w SAOS."""
        
        url = f"{self.base_url}{CONFIG.SEARCH_ENDPOINT}"
        
        params = {
            "all": keyword,
            "pageSize": min(page_size, CONFIG.MAX_PAGE_SIZE),
            "pageNumber": 0,
            "sortingField": sorting_field,
            "sortingDirection": sorting_direction
        }
        
        if court_type and court_type in CONFIG.COURT_TYPES:
            params["courtType"] = court_type
        
        if min_year:
            params["judgmentDateFrom"] = f"{min_year}-01-01"
        
        try:
            response = self.session.get(url, params=params, timeout=CONFIG.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {
                "error": str(e),
                "items": [],
                "info": {"totalResults": 0}
            }
    
    def search_and_format(
        self,
        keyword: str,
        court_type: Optional[str] = None,
        max_results: int = 15,
        min_year: Optional[int] = CONFIG.DEFAULT_MIN_YEAR,
        filter_by_topic: bool = True  # 🆕 v3.1
    ) -> Dict[str, Any]:
        """
        Wyszukuje orzeczenia i formatuje do scoringu.
        Zwraca PEŁNĄ TREŚĆ (full_text) dla każdego orzeczenia.
        
        🆕 v3.1: Filtruje orzeczenia po wydziale (sygnatura) zgodnie z tematem!
        """
        
        # Pobierz więcej niż max_results bo część odfiltrujemy
        fetch_count = max_results * 2 if filter_by_topic else max_results
        
        raw_results = self.search_judgments(
            keyword=keyword,
            court_type=court_type,
            page_size=fetch_count,
            min_year=min_year
        )
        
        if "error" in raw_results:
            return {
                "status": "ERROR",
                "error": raw_results["error"],
                "judgments": []
            }
        
        judgments = []
        filtered_out = []
        
        # 🆕 v3.1: Pobierz dozwolone wydziały dla tematu
        allowed_divisions = self._get_allowed_divisions(keyword)
        allowed_codes = self._get_division_codes(allowed_divisions)
        
        for item in raw_results.get("items", []):
            judgment = self._format_judgment(item, keyword)
            if judgment:
                # 🆕 v3.1: Filtruj po sygnaturze
                if filter_by_topic and allowed_codes:
                    signature = judgment.get("signature", "")
                    division_code = self._extract_division_code(signature)
                    
                    if division_code and division_code not in allowed_codes:
                        filtered_out.append({
                            "signature": signature,
                            "division": division_code,
                            "reason": f"Wydział {division_code} nie pasuje do tematu '{keyword}' (dozwolone: {allowed_codes})"
                        })
                        continue
                
                judgments.append(judgment)
        
        return {
            "status": "OK",
            "keyword": keyword,
            "total_found": raw_results.get("info", {}).get("totalResults", 0),
            "returned": len(judgments),
            "filtered_out": len(filtered_out),  # 🆕 v3.1
            "filtered_details": filtered_out[:5] if filtered_out else [],  # 🆕 v3.1
            "allowed_divisions": allowed_divisions,  # 🆕 v3.1
            "judgments": judgments[:max_results]  # Ogranicz do max_results
        }
    
    # ================================================================
    # 🆕 v3.1: METODY FILTROWANIA SYGNATUR
    # ================================================================
    
    def _get_allowed_divisions(self, keyword: str) -> List[str]:
        """Zwraca dozwolone kategorie wydziałów dla tematu."""
        keyword_lower = keyword.lower()
        
        # Szukaj dopasowania w mapowaniu
        for topic, divisions in CONFIG.TOPIC_TO_DIVISIONS.items():
            if topic in keyword_lower or keyword_lower in topic:
                return divisions
        
        # Fallback: wszystkie cywilne jeśli nie znaleziono
        return ["cywilne", "rodzinne"]
    
    def _get_division_codes(self, divisions: List[str]) -> List[str]:
        """Zwraca kody wydziałów (z sygnatur) dla kategorii."""
        codes = []
        for div in divisions:
            codes.extend(CONFIG.DIVISION_CODES.get(div, []))
        return codes
    
    def _extract_division_code(self, signature: str) -> Optional[str]:
        """
        Wyciąga kod wydziału z sygnatury.
        
        Przykłady:
        - "XII K 103/24" → "K" (karny)
        - "III Ca 456/23" → "Ca" (cywilny apelacyjny)
        - "IX U 324/24" → "U" (ubezpieczenia społeczne)
        - "I ACa 190/18" → "ACa" (apelacyjny cywilny)
        - "III CZP 12/23" → "CZP" (Izba Cywilna SN)
        """
        if not signature:
            return None
        
        # Wzorce dla różnych formatów sygnatur
        patterns = [
            r'\b([IVX]+)\s+([A-Z]{1,4})\s+\d+',  # "III Ca 456/23"
            r'\b([IVX]+)\s+([A-Za-z]{1,4})\s+\d+',  # "I ACa 190/18"
            r'\b([A-Z]{1,4})\s+\d+/\d+',  # "CZP 12/23"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, signature, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Ostatnia grupa to zwykle kod wydziału
                code = groups[-1] if len(groups) > 1 else groups[0]
                return code.upper()
        
        return None
    
    def _format_judgment(self, item: Dict, keyword: str) -> Optional[Dict]:
        """Formatuje pojedyncze orzeczenie."""
        
        try:
            judgment_id = item.get("id")
            judgment_date = item.get("judgmentDate", "")
            
            court_cases = item.get("courtCases", [])
            signature = court_cases[0].get("caseNumber", "") if court_cases else ""
            
            division = item.get("division", {})
            court = division.get("court", {})
            court_name = court.get("name", "")
            court_type = item.get("courtType", "")
            
            # PEŁNA TREŚĆ do scoringu
            full_text = item.get("textContent", "")
            
            # Krótki excerpt (do wyświetlenia)
            excerpt = self._extract_excerpt(full_text, keyword, 250)
            
            url = f"https://www.saos.org.pl/judgments/{judgment_id}"
            formatted_date = self._format_date(judgment_date)
            
            citation = self._generate_citation(
                court_name=court_name,
                judgment_date=formatted_date,
                signature=signature
            )
            
            return {
                "id": judgment_id,
                "signature": signature,
                "date": judgment_date,
                "formatted_date": formatted_date,
                "court": court_name,
                "court_type": court_type,
                "full_text": full_text,     # PEŁNA TREŚĆ do scoringu
                "excerpt": excerpt,          # Krótki fragment
                "url": url,
                "citation": citation
            }
        except Exception as e:
            print(f"[SAOS] Error formatting judgment: {e}")
            return None
    
    def _extract_excerpt(self, text: str, keyword: str, max_length: int) -> str:
        """Wyciąga fragment tekstu wokół słowa kluczowego."""
        
        if not text:
            return ""
        
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        pos = text_lower.find(keyword_lower)
        if pos == -1:
            return text[:max_length].strip() + "..."
        
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + max_length // 2)
        
        excerpt = text[start:end].strip()
        
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        
        return excerpt
    
    def _format_date(self, date_str: str) -> str:
        """Formatuje datę do polskiego formatu."""
        
        if not date_str:
            return ""
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            months = [
                "", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
                "lipca", "sierpnia", "września", "października", "listopada", "grudnia"
            ]
            return f"{date_obj.day} {months[date_obj.month]} {date_obj.year} r."
        except ValueError:
            return date_str
    
    def _generate_citation(
        self,
        court_name: str,
        judgment_date: str,
        signature: str
    ) -> str:
        """Generuje gotową cytację."""
        
        if not all([court_name, judgment_date, signature]):
            return ""
        
        short_court = court_name
        if "Sąd Najwyższy" in court_name:
            short_court = "Sąd Najwyższy"
        elif "Sąd Apelacyjny" in court_name:
            short_court = court_name.replace("Sąd Apelacyjny", "SA")
        elif "Sąd Okręgowy" in court_name:
            short_court = court_name.replace("Sąd Okręgowy", "SO")
        elif "Sąd Rejonowy" in court_name:
            short_court = court_name.replace("Sąd Rejonowy", "SR")
        
        return f"wyrok {short_court} z dnia {judgment_date} (sygn. {signature})"


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_client = None

def get_saos_client() -> SAOSClient:
    """Zwraca singleton klienta SAOS."""
    global _client
    if _client is None:
        _client = SAOSClient()
    return _client


def search_judgments(keyword: str, **kwargs) -> Dict[str, Any]:
    """Skrót do wyszukiwania orzeczeń."""
    return get_saos_client().search_and_format(keyword, **kwargs)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🏛️ SAOS Client v3.0 Test\n")
    
    results = search_judgments("alimenty", max_results=3)
    
    print(f"Status: {results['status']}")
    print(f"Znaleziono: {results.get('total_found', 0)}")
    
    for j in results.get("judgments", []):
        print(f"\n📄 {j['signature']} ({j['formatted_date']})")
        print(f"   Sąd: {j['court']}")
        print(f"   Full text length: {len(j.get('full_text', ''))} znaków")
        print(f"   Excerpt: {j['excerpt'][:100]}...")
