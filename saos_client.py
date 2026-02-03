# saos_client.py
# BRAJEN Legal Module - SAOS Client v4.0
# Zoptymalizowany na podstawie dokumentacji SAOS API

"""
===============================================================================
🏛️ SAOS CLIENT v4.0 - Zoptymalizowana wersja
===============================================================================

Zmiany względem v3.x:
1. MULTI-QUERY: Szukamy po frazie + przepisie + kombinacji
2. FILTROWANIE PO REPERTORIUM: Ns, C, RC, RNs (zamiast błędnego regex)
3. PEŁNA TREŚĆ: Pobieramy /api/judgments/{ID} dla tezy
4. LUCENE SYNTAX: Używamy AND, OR, cudzysłowów
5. RATE LIMITING: 0.5s między requestami

Kody Dziennika Ustaw:
- Kodeks Cywilny: 1964/16
- Kodeks Rodzinny i Opiekuńczy: 1964/9
- Kodeks Postępowania Cywilnego: 1964/43

===============================================================================
"""

import requests
import re
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta


# ============================================================================
# PROFESJONALNE CYTOWANIA (opcjonalne)
# ============================================================================
try:
    from court_url_generator import format_judgment_source
    COURT_URL_GENERATOR_AVAILABLE = True
    print("[SAOS_CLIENT] ✅ Court URL Generator loaded")
except ImportError:
    COURT_URL_GENERATOR_AVAILABLE = False
    print("[SAOS_CLIENT] ⚠️ Court URL Generator not available - using basic citations")
    
    def format_judgment_source(court_name, signature, date_str, saos_url=None, court_type="COMMON"):
        return {
            "citation": f"postanowienie {court_name} z dnia {date_str}, sygn. {signature}",
            "full_citation": f"postanowienie {court_name} z dnia {date_str}, sygn. {signature}",
            "official_portal": "orzeczenia.ms.gov.pl",
            "portal_url": "https://orzeczenia.ms.gov.pl",
        }


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class SAOSConfig:
    """Konfiguracja klienta SAOS API."""
    
    BASE_URL: str = "https://www.saos.org.pl/api"
    SEARCH_ENDPOINT: str = "/search/judgments"
    JUDGMENT_ENDPOINT: str = "/judgments"
    
    # Limity
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 100
    REQUEST_DELAY: float = 0.5
    TIMEOUT: int = 15
    
    # Domyślne filtry
    DEFAULT_MIN_YEAR: int = 2018
    DEFAULT_COURT_TYPE: str = "COMMON"
    
    # Kody Dziennika Ustaw
    LAW_JOURNAL_CODES: Dict[str, str] = field(default_factory=lambda: {
        "kc": "1964/16",
        "kro": "1964/9",
        "kpc": "1964/43",
    })
    
    # Repertoria dla spraw cywilnych/rodzinnych
    CIVIL_FAMILY_REPERTORIA: Set[str] = field(default_factory=lambda: {
        "C", "Ca", "ACa", "Cz", "ACz",
        "Ns", "ANs",
        "Co", "ACo",
        "Nc",
        "GC", "GCo",
        "RC", "RCa",
        "RNs",
        "Nsm",
        "CZP", "CSK", "CNP",
    })
    
    # Mapowanie tematów na przepisy i query
    TOPIC_CONFIG: Dict[str, Dict] = field(default_factory=lambda: {
        "ubezwłasnowolnienie": {
            "law_code": "1964/16",
            "articles": ["art. 13", "art. 16"],
            "queries": [
                '"ubezwłasnowolnienie całkowite"',
                '"ubezwłasnowolnienie częściowe"',
                'ubezwłasnowolnienie AND "choroba psychiczna"',
                'ubezwłasnowolnienie AND demencja',
                'ubezwłasnowolnienie AND otępienie',
                'ubezwłasnowolnienie AND "choroba Alzheimera"',
            ],
            "repertoria": {"Ns", "ANs"},
            "court_level": "REGIONAL",
        },
        "alimenty": {
            "law_code": "1964/9",
            "articles": ["art. 133", "art. 135"],
            "queries": [
                'alimenty AND "usprawiedliwione potrzeby"',
                '"podwyższenie alimentów"',
                '"obniżenie alimentów"',
            ],
            "repertoria": {"RNs", "RC", "Ns", "C"},
            "court_level": None,
        },
        "rozwód": {
            "law_code": "1964/9",
            "articles": ["art. 56", "art. 57"],
            "queries": [
                'rozwód AND "trwały rozkład pożycia"',
                'rozwód AND "wina małżonka"',
            ],
            "repertoria": {"RC", "C", "Ca", "ACa"},
            "court_level": "REGIONAL",
        },
        "spadek": {
            "law_code": "1964/16",
            "articles": ["art. 922", "art. 931"],
            "queries": [
                '"stwierdzenie nabycia spadku"',
                'spadek AND dziedziczenie',
            ],
            "repertoria": {"Ns", "C", "Ca"},
            "court_level": None,
        },
        "zachowek": {
            "law_code": "1964/16",
            "articles": ["art. 991"],
            "queries": [
                'zachowek AND "uprawniony do zachowku"',
            ],
            "repertoria": {"C", "Ca", "ACa"},
            "court_level": None,
        },
        "odszkodowanie": {
            "law_code": "1964/16",
            "articles": ["art. 415", "art. 445"],
            "queries": [
                'odszkodowanie AND szkoda',
                'zadośćuczynienie',
            ],
            "repertoria": {"C", "Ca", "ACa"},
            "court_level": None,
        },
        "władza rodzicielska": {
            "law_code": "1964/9",
            "articles": ["art. 92", "art. 107", "art. 111"],
            "queries": [
                '"władza rodzicielska" AND ograniczenie',
                '"władza rodzicielska" AND pozbawienie',
            ],
            "repertoria": {"RNs", "Nsm", "RC"},
            "court_level": None,
        },
    })


CONFIG = SAOSConfig()


# ============================================================================
# GŁÓWNA KLASA KLIENTA
# ============================================================================

class SAOSClient:
    """Zoptymalizowany klient SAOS API v4.0"""
    
    def __init__(self, config: SAOSConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json; charset=UTF-8",
            "User-Agent": "BRAJEN-SEO-Legal/4.0"
        })
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.REQUEST_DELAY:
            time.sleep(self.config.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _search_request(self, params: Dict) -> Dict[str, Any]:
        """Wykonuje request do SAOS search API."""
        self._rate_limit()
        url = f"{self.config.BASE_URL}{self.config.SEARCH_ENDPOINT}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[SAOS] ❌ Request error: {e}")
            return {"items": [], "info": {"totalResults": 0}}
    
    def _fetch_full_judgment(self, judgment_id: int) -> Optional[Dict]:
        """Pobiera pełne dane orzeczenia."""
        self._rate_limit()
        url = f"{self.config.BASE_URL}{self.config.JUDGMENT_ENDPOINT}/{judgment_id}"
        
        try:
            response = self.session.get(url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data)
        except requests.RequestException as e:
            print(f"[SAOS] ⚠️ Cannot fetch judgment {judgment_id}: {e}")
            return None
    
    # ========================================================================
    # PARSOWANIE SYGNATURY
    # ========================================================================
    
    def _parse_case_number(self, case_number: str) -> Optional[Dict]:
        """Parsuje sygnaturę i wyciąga repertorium."""
        patterns = [
            r'^([IVXLC]+)\s+(\d*)\s*([A-Za-z]+)\s+(\d+)/(\d+)$',
            r'^([IVXLC]+)\s*([A-Za-z]+)\s+(\d+)/(\d+)$',
        ]
        
        case_number = case_number.strip()
        
        for pattern in patterns:
            match = re.match(pattern, case_number)
            if match:
                groups = match.groups()
                if len(groups) == 5:
                    return {"repertorium": groups[2]}
                elif len(groups) == 4:
                    return {"repertorium": groups[1]}
        return None
    
    def _matches_repertoria(self, case_number: str, allowed_repertoria: Set[str]) -> bool:
        """Sprawdza czy sygnatura pasuje do dozwolonych repertoriów."""
        parsed = self._parse_case_number(case_number)
        if parsed:
            return parsed["repertorium"] in allowed_repertoria
        return True
    
    # ========================================================================
    # EKSTRAKCJA TEZY
    # ========================================================================
    
    def _extract_excerpt(self, text_content: str, keyword: str, max_length: int = 300) -> str:
        """Wyciąga fragment z treści orzeczenia."""
        if not text_content:
            return ""
        
        text = re.sub(r'\s+', ' ', text_content).strip()
        
        # Szukaj fragmentu z <em> tagami
        em_pattern = r'.{0,100}<em>[^<]+</em>.{0,200}'
        em_matches = re.findall(em_pattern, text, re.IGNORECASE)
        if em_matches:
            excerpt = re.sub(r'<[^>]+>', '', em_matches[0]).strip()
            if len(excerpt) > max_length:
                excerpt = excerpt[:max_length] + "..."
            return excerpt
        
        # Szukaj frazy kluczowej
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        pos = text_lower.find(keyword_lower)
        
        if pos != -1:
            start = max(0, pos - 100)
            end = min(len(text), pos + max_length)
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
            return excerpt
        
        # Fallback: początek
        for prefix in ["POSTANOWIENIE", "WYROK", "UZASADNIENIE", "W IMIENIU"]:
            if text.upper().startswith(prefix):
                text = text[len(prefix):].strip()
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    # ========================================================================
    # FORMATOWANIE WYNIKU
    # ========================================================================
    
    def _format_judgment(self, item: Dict, keyword: str, full_data: Dict = None) -> Dict:
        """Formatuje orzeczenie."""
        
        judgment_id = item.get("id")
        judgment_date = item.get("judgmentDate", "")
        court_type = item.get("courtType", "COMMON")
        
        court_cases = item.get("courtCases", [])
        case_number = court_cases[0].get("caseNumber", "") if court_cases else ""
        
        division = item.get("division", {})
        court = division.get("court", {})
        court_name = court.get("name", "")
        
        if full_data:
            text_content = full_data.get("textContent", item.get("textContent", ""))
        else:
            text_content = item.get("textContent", "")
        
        excerpt = self._extract_excerpt(text_content, keyword)
        
        saos_url = f"https://www.saos.org.pl/judgments/{judgment_id}"
        formatted_date = self._format_date(judgment_date)
        
        parsed = self._parse_case_number(case_number)
        repertorium = parsed["repertorium"] if parsed else ""
        
        # Cytowanie
        citation_data = format_judgment_source(
            court_name=court_name,
            signature=case_number,
            date=formatted_date,
            saos_url=saos_url,
            court_type=court_type
        )
        
        return {
            "id": judgment_id,
            "signature": case_number,
            "repertorium": repertorium,
            "date": judgment_date,
            "formatted_date": formatted_date,
            "court": court_name,
            "court_type": court_type,
            "excerpt": excerpt,
            "full_text": text_content,
            "url": saos_url,
            "citation": citation_data.get("citation", ""),
            "full_citation": citation_data.get("full_citation", ""),
            "official_portal": citation_data.get("official_portal", ""),
            "portal_url": citation_data.get("portal_url", ""),
        }
    
    def _format_date(self, date_str: str) -> str:
        """Formatuje datę."""
        if not date_str:
            return ""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            months = ["", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
                     "lipca", "sierpnia", "września", "października", "listopada", "grudnia"]
            return f"{dt.day} {months[dt.month]} {dt.year} r."
        except:
            return date_str
    
    # ========================================================================
    # GŁÓWNA FUNKCJA WYSZUKIWANIA
    # ========================================================================
    
    def search_for_topic(
        self,
        topic: str,
        max_results: int = 10,
        min_year: int = None,
        fetch_full_text: bool = True
    ) -> Dict[str, Any]:
        """
        Wyszukuje orzeczenia dla tematu prawnego.
        
        Używa MULTI-QUERY: frazy + przepisy + filtrowanie po repertorium
        """
        
        min_year = min_year or self.config.DEFAULT_MIN_YEAR
        topic_lower = topic.lower()
        
        # Znajdź konfigurację
        topic_config = None
        for key, cfg in self.config.TOPIC_CONFIG.items():
            if key in topic_lower:
                topic_config = cfg
                break
        
        print(f"[SAOS] 🔍 Szukam orzeczeń dla: '{topic}'")
        
        if not topic_config:
            print(f"[SAOS] ⚠️ Brak konfiguracji dla '{topic}'")
            topic_config = {
                "queries": [f'"{topic}"'],
                "repertoria": self.config.CIVIL_FAMILY_REPERTORIA,
            }
        
        all_judgments = {}
        stats = {"total_api": 0, "filtered": 0, "duplicates": 0}
        
        # STRATEGIA 1: Query frazowe
        for query in topic_config.get("queries", []):
            params = {
                "all": query,
                "courtType": self.config.DEFAULT_COURT_TYPE,
                "judgmentDateFrom": f"{min_year}-01-01",
                "pageSize": self.config.DEFAULT_PAGE_SIZE,
                "sortingField": "JUDGMENT_DATE",
                "sortingDirection": "DESC",
            }
            
            if topic_config.get("court_level"):
                params["ccCourtType"] = topic_config["court_level"]
            
            result = self._search_request(params)
            items = result.get("items", [])
            stats["total_api"] += len(items)
            
            print(f"[SAOS] 📝 Query '{query[:40]}...' → {len(items)} wyników")
            
            allowed_repertoria = topic_config.get("repertoria", self.config.CIVIL_FAMILY_REPERTORIA)
            
            for item in items:
                judgment_id = item.get("id")
                if judgment_id in all_judgments:
                    stats["duplicates"] += 1
                    continue
                
                court_cases = item.get("courtCases", [])
                if court_cases:
                    case_number = court_cases[0].get("caseNumber", "")
                    if not self._matches_repertoria(case_number, allowed_repertoria):
                        stats["filtered"] += 1
                        print(f"[SAOS] ⏭️ {case_number} - złe repertorium")
                        continue
                
                full_data = None
                if fetch_full_text:
                    full_data = self._fetch_full_judgment(judgment_id)
                
                formatted = self._format_judgment(item, topic, full_data)
                all_judgments[judgment_id] = formatted
                print(f"[SAOS] ✅ {formatted['signature']}")
                
                if len(all_judgments) >= max_results * 2:
                    break
            
            if len(all_judgments) >= max_results:
                break
        
        # STRATEGIA 2: Query po przepisie
        if len(all_judgments) < max_results and topic_config.get("law_code"):
            law_code = topic_config["law_code"]
            
            for article in topic_config.get("articles", [])[:2]:
                params = {
                    "lawJournalEntryCode": law_code,
                    "all": f'"{article}"',
                    "courtType": self.config.DEFAULT_COURT_TYPE,
                    "judgmentDateFrom": f"{min_year}-01-01",
                    "pageSize": 30,
                    "sortingField": "JUDGMENT_DATE",
                    "sortingDirection": "DESC",
                }
                
                result = self._search_request(params)
                items = result.get("items", [])
                
                print(f"[SAOS] 📜 Przepis '{article}' → {len(items)} wyników")
                
                for item in items:
                    judgment_id = item.get("id")
                    if judgment_id in all_judgments:
                        continue
                    
                    full_data = None
                    if fetch_full_text:
                        full_data = self._fetch_full_judgment(judgment_id)
                    
                    formatted = self._format_judgment(item, topic, full_data)
                    all_judgments[judgment_id] = formatted
                    
                    if len(all_judgments) >= max_results * 2:
                        break
        
        # SCORING
        judgments_list = list(all_judgments.values())
        for j in judgments_list:
            score = 0
            try:
                year = int(j.get("date", "2000")[:4])
                score += (year - 2015) * 2
            except:
                pass
            if j.get("excerpt") and len(j["excerpt"]) > 50:
                score += 20
            if j.get("repertorium") in topic_config.get("repertoria", set()):
                score += 30
            j["relevance_score"] = score
        
        judgments_list.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        final_judgments = judgments_list[:max_results]
        
        print(f"[SAOS] ✅ Znaleziono {len(final_judgments)} orzeczeń")
        print(f"[SAOS] 📊 Stats: API={stats['total_api']}, filtered={stats['filtered']}, dupes={stats['duplicates']}")
        
        return {
            "status": "success" if final_judgments else "no_results",
            "topic": topic,
            "judgments": final_judgments,
            "total_found": len(final_judgments),
            "stats": stats,
        }
    
    # ========================================================================
    # LEGACY API (kompatybilność wsteczna)
    # ========================================================================
    
    def search_and_format(self, query: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Legacy: podstawowe wyszukiwanie."""
        return self.search_for_topic(query, max_results, **kwargs)
    
    def search_judgments(self, keyword: str, **kwargs) -> Dict[str, Any]:
        """Legacy alias."""
        return self.search_for_topic(keyword, **kwargs)


# ============================================================================
# SINGLETON I EKSPORTY
# ============================================================================

_client: Optional[SAOSClient] = None

def get_saos_client() -> SAOSClient:
    """Zwraca singleton klienta SAOS."""
    global _client
    if _client is None:
        _client = SAOSClient()
    return _client


def search_judgments(topic: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
    """Główna funkcja do wyszukiwania orzeczeń."""
    return get_saos_client().search_for_topic(topic, max_results, **kwargs)


SAOS_AVAILABLE = True

__all__ = [
    "SAOSClient",
    "SAOSConfig",
    "CONFIG",
    "get_saos_client",
    "search_judgments",
    "SAOS_AVAILABLE",
    "COURT_URL_GENERATOR_AVAILABLE",
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏛️ SAOS CLIENT v4.0 TEST")
    print("=" * 60)
    
    result = search_judgments("ubezwłasnowolnienie", max_results=3)
    
    print(f"\nStatus: {result['status']}")
    print(f"Znaleziono: {result['total_found']}")
    
    for j in result.get("judgments", []):
        print(f"\n📄 {j['signature']} ({j['formatted_date']})")
        print(f"   Sąd: {j['court']}")
        print(f"   Repertorium: {j['repertorium']}")
        print(f"   Excerpt: {j['excerpt'][:100]}..." if j.get('excerpt') else "   -")
