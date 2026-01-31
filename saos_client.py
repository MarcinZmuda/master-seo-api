# saos_client.py
# BRAJEN Legal Module - Klient SAOS API v3.2
# Z pełną treścią do scoringu + FILTROWANIE SYGNATUR + PROFESJONALNE CYTOWANIA

"""
===============================================================================
🏛️ SAOS CLIENT v3.2
===============================================================================

Klient do System Analizy Orzeczeń Sądowych (SAOS).
https://www.saos.org.pl/api

🆕 Zmiany w v3.2:
- Integracja z court_url_generator dla profesjonalnych cytowań
- official_portal w każdym wyniku (domena portalu orzeczeń)
- full_citation w standardzie prawniczym
- portal_type dla identyfikacji typu sądu

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
# 🆕 v3.2: PROFESJONALNE CYTOWANIA
# ============================================================================
try:
    from court_url_generator import (
        format_judgment_source,
        get_court_portal_url,
        generate_search_url,
        COURT_TO_SUBDOMAIN
    )
    COURT_URL_GENERATOR_AVAILABLE = True
    print("[SAOS_CLIENT] ✅ Court URL Generator loaded - professional citations enabled")
except ImportError:
    COURT_URL_GENERATOR_AVAILABLE = False
    print("[SAOS_CLIENT] ⚠️ Court URL Generator not available - using basic citations")
    
    def format_judgment_source(court_name, signature, date, saos_url=None, court_type="COMMON"):
        """Fallback: podstawowe cytowanie."""
        return {
            "citation": f"wyrok {court_name} z dnia {date}, sygn. {signature}",
            "official_portal": "orzeczenia.ms.gov.pl",
            "portal_url": "https://orzeczenia.ms.gov.pl",
            "source_note": "(orzeczenia.ms.gov.pl)",
            "full_citation": f"wyrok {court_name} z dnia {date}, sygn. {signature} (orzeczenia.ms.gov.pl)",
            "signature": signature,
            "portal_type": "ms"
        }


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class SAOSConfig:
    BASE_URL: str = "https://www.saos.org.pl/api"
    SEARCH_ENDPOINT: str = "/search/judgments"
    JUDGMENT_ENDPOINT: str = "/judgments"
    DEFAULT_PAGE_SIZE: int = 15
    MAX_PAGE_SIZE: int = 50
    DEFAULT_MIN_YEAR: int = 2015
    TIMEOUT: int = 15
    
    COURT_TYPES = {
        "COMMON": "Sądy Powszechne",
        "SUPREME": "Sąd Najwyższy",
        "ADMINISTRATIVE": "Sądy Administracyjne",
        "CONSTITUTIONAL": "Trybunał Konstytucyjny",
        "NATIONAL_APPEAL_CHAMBER": "Krajowa Izba Odwoławcza"
    }
    
    # Mapowanie wydziałów - które sygnatury dla jakich tematów
    DIVISION_CODES: Dict[str, List[str]] = field(default_factory=lambda: {
        "cywilne": ["C", "Ca", "ACa", "Cz", "ACz", "CZP", "CSK", "CNP"],
        "rodzinne": ["C", "Ca", "ACa", "RC", "RCa", "CZP"],
        "karne": ["K", "Ka", "AKa", "Kz", "AKz", "KZP", "KK"],
        "pracy": ["P", "Pa", "APa", "Pz", "APz", "PZP"],
        "ubezpieczenia": ["U", "Ua", "AUa", "Uz", "AUz", "UZP"],
        "administracyjne": ["SA", "OSA", "GSK", "NSA", "OSK"]
    })
    
    TOPIC_TO_DIVISIONS: Dict[str, List[str]] = field(default_factory=lambda: {
        # Prawo rodzinne
        "alimenty": ["cywilne", "rodzinne"],
        "rozwód": ["cywilne", "rodzinne"],
        "separacja": ["cywilne", "rodzinne"],
        "opieka nad dzieckiem": ["cywilne", "rodzinne"],
        "władza rodzicielska": ["cywilne", "rodzinne"],
        "ubezwłasnowolnienie": ["cywilne", "rodzinne"],
        "kuratela": ["cywilne", "rodzinne"],
        "przysposobienie": ["cywilne", "rodzinne"],
        "adopcja": ["cywilne", "rodzinne"],
        
        # Prawo spadkowe
        "spadek": ["cywilne"],
        "testament": ["cywilne"],
        "dziedziczenie": ["cywilne"],
        "zachowek": ["cywilne"],
        
        # Prawo cywilne
        "umowa": ["cywilne"],
        "odszkodowanie": ["cywilne"],
        "zadośćuczynienie": ["cywilne"],
        "nieruchomość": ["cywilne"],
        "służebność": ["cywilne"],
        "hipoteka": ["cywilne"],
        
        # Prawo pracy
        "wypowiedzenie": ["pracy", "cywilne"],
        "mobbing": ["pracy", "cywilne"],
        "wynagrodzenie": ["pracy"],
        "zwolnienie": ["pracy"],
        
        # Prawo karne
        "przestępstwo": ["karne"],
        "kara": ["karne"],
        "oskarżenie": ["karne"],
    })


CONFIG = SAOSConfig()


# ============================================================================
# KLIENT SAOS
# ============================================================================

class SAOSClient:
    """Klient do SAOS API z filtrowaniem sygnatur i profesjonalnymi cytowaniami."""
    
    def __init__(self, config: SAOSConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BRAJEN-SEO-Engine/3.2"
        })
    
    def search_and_format(
        self,
        query: str,
        court_type: str = None,
        min_date: str = None,
        max_results: int = 10,
        theme_phrase: str = None
    ) -> Dict[str, Any]:
        """
        Wyszukuje i formatuje orzeczenia z SAOS.
        
        Args:
            query: Fraza do wyszukania
            court_type: Typ sądu (COMMON, SUPREME, etc.)
            min_date: Minimalna data (YYYY-MM-DD)
            max_results: Maksymalna liczba wyników
            theme_phrase: Główna fraza tematu (do filtrowania sygnatur)
            
        Returns:
            Dict z judgments, status, total_found
        """
        
        min_date = min_date or f"{self.config.DEFAULT_MIN_YEAR}-01-01"
        
        params = {
            "all": query,
            "judgmentDateFrom": min_date,
            "pageSize": min(max_results * 3, self.config.MAX_PAGE_SIZE),
            "sortingField": "JUDGMENT_DATE",
            "sortingDirection": "DESC"
        }
        
        if court_type:
            params["courtType"] = court_type
        
        url = f"{self.config.BASE_URL}{self.config.SEARCH_ENDPOINT}"
        
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.config.TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
        except requests.RequestException as e:
            print(f"[SAOS] Request error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "judgments": [],
                "total_found": 0
            }
        
        items = data.get("items", [])
        
        if not items:
            return {
                "status": "no_results",
                "judgments": [],
                "total_found": 0,
                "query": query
            }
        
        # Określ dozwolone wydziały dla tematu
        main_topic = theme_phrase or query
        allowed_divisions = self._get_allowed_divisions(main_topic)
        
        judgments = []
        for item in items:
            # Pobierz pełne dane orzeczenia
            judgment_id = item.get("id")
            if not judgment_id:
                continue
            
            full_item = self._fetch_full_judgment(judgment_id)
            if not full_item:
                full_item = item
            
            # Filtruj po sygnaturze
            signature = self._extract_signature(full_item)
            if allowed_divisions and not self._is_signature_allowed(signature, allowed_divisions):
                print(f"[SAOS] ⏭️ Pominięto {signature} - nieodpowiedni wydział dla '{main_topic}'")
                continue
            
            # Potwierdź że temat jest PRZEDMIOTEM sprawy
            is_subject, reason = self._confirm_subject_matter(full_item, main_topic)
            if not is_subject:
                print(f"[SAOS] ⏭️ Pominięto {signature} - {reason}")
                continue
            
            judgment = self._format_judgment(full_item, query)
            if judgment:
                judgments.append(judgment)
            
            if len(judgments) >= max_results:
                break
        
        return {
            "status": "success" if judgments else "no_results",
            "judgments": judgments,
            "total_found": len(judgments),
            "query": query,
            "filtered_by_division": bool(allowed_divisions),
            "professional_citations": COURT_URL_GENERATOR_AVAILABLE
        }
    
    def _fetch_full_judgment(self, judgment_id: int) -> Optional[Dict]:
        """Pobiera pełne dane orzeczenia (z treścią)."""
        url = f"{self.config.BASE_URL}{self.config.JUDGMENT_ENDPOINT}/{judgment_id}"
        
        try:
            response = self.session.get(url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[SAOS] Error fetching judgment {judgment_id}: {e}")
            return None
    
    def _extract_signature(self, item: Dict) -> str:
        """Wyciąga sygnaturę z orzeczenia."""
        court_cases = item.get("courtCases", [])
        if court_cases:
            return court_cases[0].get("caseNumber", "")
        return ""
    
    def _get_allowed_divisions(self, topic: str) -> List[str]:
        """Zwraca listę dozwolonych kodów wydziałów dla tematu."""
        topic_lower = topic.lower()
        
        # Znajdź temat w mapowaniu
        for key, divisions in self.config.TOPIC_TO_DIVISIONS.items():
            if key in topic_lower:
                # Rozwiń kategorie do kodów wydziałów
                all_codes = []
                for div in divisions:
                    codes = self.config.DIVISION_CODES.get(div, [])
                    all_codes.extend(codes)
                return list(set(all_codes))
        
        return []  # Brak ograniczeń
    
    def _is_signature_allowed(self, signature: str, allowed_codes: List[str]) -> bool:
        """Sprawdza czy sygnatura jest z dozwolonego wydziału."""
        if not signature or not allowed_codes:
            return True
        
        # Wyciągnij kod wydziału z sygnatury (np. "II C 123/20" → "C")
        match = re.search(r'\b([IVXLC]+)\s*([A-Za-z]{1,4})\s*\d', signature)
        if match:
            division_code = match.group(2).upper()
            return division_code in [c.upper() for c in allowed_codes]
        
        return True
    
    def _confirm_subject_matter(self, item: Dict, main_topic: str) -> tuple:
        """
        Potwierdza że temat jest PRZEDMIOTEM sprawy, nie tylko kontekstem.
        """
        full_text = item.get("textContent", "")
        if not full_text:
            return True, "Brak treści do sprawdzenia"
        
        text_to_check = full_text[:2000].lower()
        keyword_lower = main_topic.lower()
        
        # Frazy potwierdzające przedmiot sprawy
        confirmation_phrases = [
            f"w sprawie o {keyword_lower}",
            f"o {keyword_lower}",
            f"w sprawie z powództwa o {keyword_lower}",
            f"w przedmiocie {keyword_lower}",
            f"dotyczącą {keyword_lower}",
            f"dotyczącej {keyword_lower}",
            f"o zasądzenie {keyword_lower}",
            f"o ustanowienie {keyword_lower}",
            f"o orzeczenie {keyword_lower}",
            f"wniosek o {keyword_lower}",
        ]
        
        for phrase in confirmation_phrases:
            if phrase.lower() in text_to_check:
                return True, f"Znaleziono: '{phrase}'"
        
        first_500 = text_to_check[:500]
        if keyword_lower in first_500:
            context_indicators = [
                "ubezwłasnowolniony powód",
                "ubezwłasnowolniona pozwana",
                "będąc ubezwłasnowolnion",
                "jako ubezwłasnowolnion",
                "osoby ubezwłasnowolnionej",
                "przedstawiciel ubezwłasnowolnionego",
            ]
            for indicator in context_indicators:
                if indicator in text_to_check:
                    return False, f"Temat to tylko kontekst: '{indicator}'"
        
        return False, f"Brak fraz potwierdzających przedmiot sprawy dla '{main_topic}'"
    
    def _format_judgment(self, item: Dict, keyword: str) -> Optional[Dict]:
        """
        Formatuje pojedyncze orzeczenie.
        
        🆕 v3.2: Dodaje profesjonalne cytowanie z court_url_generator
        """
        
        try:
            judgment_id = item.get("id")
            judgment_date = item.get("judgmentDate", "")
            
            court_cases = item.get("courtCases", [])
            signature = court_cases[0].get("caseNumber", "") if court_cases else ""
            
            division = item.get("division", {})
            court = division.get("court", {})
            court_name = court.get("name", "")
            court_type = item.get("courtType", "COMMON")
            
            full_text = item.get("textContent", "")
            excerpt = self._extract_excerpt(full_text, keyword, 250)
            
            saos_url = f"https://www.saos.org.pl/judgments/{judgment_id}"
            formatted_date = self._format_date(judgment_date)
            
            # 🆕 v3.2: PROFESJONALNE CYTOWANIE
            citation_data = format_judgment_source(
                court_name=court_name,
                signature=signature,
                date=formatted_date,
                saos_url=saos_url,
                court_type=court_type
            )
            
            return {
                "id": judgment_id,
                "signature": signature,
                "date": judgment_date,
                "formatted_date": formatted_date,
                "court": court_name,
                "court_type": court_type,
                "full_text": full_text,
                "excerpt": excerpt,
                "url": saos_url,
                # 🆕 v3.2: Profesjonalne cytowania
                "citation": citation_data.get("citation", ""),
                "full_citation": citation_data.get("full_citation", ""),
                "official_portal": citation_data.get("official_portal", ""),
                "portal_url": citation_data.get("portal_url", ""),
                "portal_type": citation_data.get("portal_type", ""),
                "source_note": citation_data.get("source_note", ""),
                "citation_instruction": citation_data.get("citation_instruction", "")
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
    print("🏛️ SAOS Client v3.2 Test\n")
    print(f"Professional citations: {COURT_URL_GENERATOR_AVAILABLE}\n")
    
    results = search_judgments("alimenty", max_results=3)
    
    print(f"Status: {results['status']}")
    print(f"Znaleziono: {results.get('total_found', 0)}")
    
    for j in results.get("judgments", []):
        print(f"\n📄 {j['signature']} ({j['formatted_date']})")
        print(f"   Sąd: {j['court']}")
        print(f"   Full text length: {len(j.get('full_text', ''))} znaków")
        print(f"   🆕 Citation: {j.get('citation', 'N/A')}")
        print(f"   🆕 Portal: {j.get('official_portal', 'N/A')}")
        print(f"   🆕 Full citation: {j.get('full_citation', 'N/A')[:100]}...")
