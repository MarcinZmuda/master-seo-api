# saos_client.py
# BRAJEN Legal Module - Klient SAOS API v3.0
# Z pełną treścią do scoringu

"""
===============================================================================
🏛️ SAOS CLIENT v3.0
===============================================================================

Klient do System Analizy Orzeczeń Sądowych (SAOS).
https://www.saos.org.pl/api

Zmiany w v3:
- Zwraca full_text do scoringu
- Lepsze wyciąganie fragmentów

===============================================================================
"""

import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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
    DEFAULT_PAGE_SIZE: int = 10
    MAX_PAGE_SIZE: int = 20
    DEFAULT_MIN_YEAR: int = 2020
    TIMEOUT: int = 15
    
    COURT_TYPES = {
        "COMMON": "Sądy Powszechne",
        "SUPREME": "Sąd Najwyższy",
        "ADMINISTRATIVE": "Sądy Administracyjne",
        "CONSTITUTIONAL": "Trybunał Konstytucyjny",
        "NATIONAL_APPEAL_CHAMBER": "Krajowa Izba Odwoławcza"
    }


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
        max_results: int = 10,
        min_year: Optional[int] = CONFIG.DEFAULT_MIN_YEAR
    ) -> Dict[str, Any]:
        """
        Wyszukuje orzeczenia i formatuje do scoringu.
        Zwraca PEŁNĄ TREŚĆ (full_text) dla każdego orzeczenia.
        """
        
        raw_results = self.search_judgments(
            keyword=keyword,
            court_type=court_type,
            page_size=max_results,
            min_year=min_year
        )
        
        if "error" in raw_results:
            return {
                "status": "ERROR",
                "error": raw_results["error"],
                "judgments": []
            }
        
        judgments = []
        
        for item in raw_results.get("items", []):
            judgment = self._format_judgment(item, keyword)
            if judgment:
                judgments.append(judgment)
        
        return {
            "status": "OK",
            "keyword": keyword,
            "total_found": raw_results.get("info", {}).get("totalResults", 0),
            "returned": len(judgments),
            "judgments": judgments
        }
    
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
