# local_court_scraper.py
# BRAJEN Legal Module - Scraper lokalnych portali orzeczeń v3.5
# Uzupełnia SAOS o orzeczenia z portali indywidualnych sądów

"""
===============================================================================
🏛️ LOCAL COURT SCRAPER v3.5
===============================================================================

Scraper dla lokalnych portali orzeczeń sądów powszechnych.
Portale te NIE są zindeksowane w SAOS!

Przykład: https://orzeczenia.warszawa.so.gov.pl

v3.5 ZMIANY:
- Prawidłowy format URL wyszukiwania: /search/advanced/{keyword}/$N/{court_code}/...
- Prawidłowy format URL treści: /content/{keyword}/{id}
- Kodowanie polskich znaków: ó → $00f3, ł → $0142

Struktura URL:
- Wyszukiwanie: /search/advanced/{keyword}/$N/{court_code}/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1
- Szczegóły: /details/{keyword}/{ID}
- Treść: /content/{keyword}/{ID}

===============================================================================
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
import urllib.parse


# ============================================================================
# KONFIGURACJA LOKALNYCH PORTALI
# ============================================================================

@dataclass
class LocalCourtConfig:
    """Konfiguracja scrapera lokalnych portali."""
    
    TIMEOUT: int = 15
    MAX_RESULTS: int = 20
    
    # Główne portale sądów okręgowych
    COURT_PORTALS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "warszawa.so": {
            "name": "Sąd Okręgowy w Warszawie",
            "code": "15450500",
            "base_url": "https://orzeczenia.warszawa.so.gov.pl"
        },
        "warszawapraga.so": {
            "name": "Sąd Okręgowy Warszawa-Praga",
            "code": "15451000",
            "base_url": "https://orzeczenia.warszawapraga.so.gov.pl"
        },
        "krakow.so": {
            "name": "Sąd Okręgowy w Krakowie",
            "code": "15201000",
            "base_url": "https://orzeczenia.krakow.so.gov.pl"
        },
        "gdansk.so": {
            "name": "Sąd Okręgowy w Gdańsku",
            "code": "15101500",
            "base_url": "https://orzeczenia.gdansk.so.gov.pl"
        },
        "wroclaw.so": {
            "name": "Sąd Okręgowy we Wrocławiu",
            "code": "15502500",
            "base_url": "https://orzeczenia.wroclaw.so.gov.pl"
        },
        "poznan.so": {
            "name": "Sąd Okręgowy w Poznaniu",
            "code": "15351000",
            "base_url": "https://orzeczenia.poznan.so.gov.pl"
        },
        "lodz.so": {
            "name": "Sąd Okręgowy w Łodzi",
            "code": "15251000",
            "base_url": "https://orzeczenia.lodz.so.gov.pl"
        },
        "katowice.so": {
            "name": "Sąd Okręgowy w Katowicach",
            "code": "15152000",
            "base_url": "https://orzeczenia.katowice.so.gov.pl"
        },
        "lublin.so": {
            "name": "Sąd Okręgowy w Lublinie",
            "code": "15300500",
            "base_url": "https://orzeczenia.lublin.so.gov.pl"
        },
        "szczecin.so": {
            "name": "Sąd Okręgowy w Szczecinie",
            "code": "15551500",
            "base_url": "https://orzeczenia.szczecin.so.gov.pl"
        },
    })
    
    # v3.5: Mapa polskich znaków → kody URL
    POLISH_CHARS: Dict[str, str] = field(default_factory=lambda: {
        'ą': '$0105', 'ć': '$0107', 'ę': '$0119', 'ł': '$0142',
        'ń': '$0144', 'ó': '$00f3', 'ś': '$015b', 'ź': '$017a', 'ż': '$017c',
        'Ą': '$0104', 'Ć': '$0106', 'Ę': '$0118', 'Ł': '$0141',
        'Ń': '$0143', 'Ó': '$00d3', 'Ś': '$015a', 'Ź': '$0179', 'Ż': '$017b'
    })


CONFIG = LocalCourtConfig()


# ============================================================================
# SCRAPER LOKALNYCH PORTALI
# ============================================================================

class LocalCourtScraper:
    """Scraper dla lokalnych portali orzeczeń v3.5."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
        })
    
    def _encode_keyword_for_url(self, keyword: str) -> str:
        """
        v3.5: Koduje polskie znaki do formatu URL portali.
        
        Przykłady:
        - "rozwód" → "rozw$00f3d"
        - "ubezwłasnowolnienie" → "ubezw$0142asnowolnienie"
        """
        result = keyword
        for char, code in CONFIG.POLISH_CHARS.items():
            result = result.replace(char, code)
        result = result.replace(' ', '+')
        return result
    
    def search_all_portals(
        self,
        keyword: str,
        max_results: int = CONFIG.MAX_RESULTS,
        theme_phrase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Przeszukuje wszystkie lokalne portale."""
        all_results = []
        errors = []
        
        results_per_portal = max(3, max_results // len(CONFIG.COURT_PORTALS))
        
        for portal_key, portal_info in CONFIG.COURT_PORTALS.items():
            try:
                portal_results = self.search_portal(
                    base_url=portal_info["base_url"],
                    keyword=keyword,
                    max_results=results_per_portal,
                    court_code=portal_info.get("code")
                )
                
                for result in portal_results.get("judgments", []):
                    result["source_portal"] = portal_info["name"]
                    result["source_url"] = portal_info["base_url"]
                
                all_results.extend(portal_results.get("judgments", []))
                
            except Exception as e:
                errors.append({"portal": portal_info["name"], "error": str(e)})
        
        all_results.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return {
            "status": "success" if all_results else "no_results",
            "keyword": keyword,
            "total_found": len(all_results),
            "judgments": all_results[:max_results],
            "errors": errors if errors else None
        }
    
    def search_portal(
        self,
        base_url: str,
        keyword: str,
        max_results: int = 10,
        theme_phrase: Optional[str] = None,
        court_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Przeszukuje pojedynczy portal.
        
        Przykład URL:
            https://orzeczenia.warszawa.so.gov.pl/search/advanced/alimenty/$N/15450500/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1
        """
        encoded_keyword = self._encode_keyword_for_url(keyword)
        court_id = court_code or "$N"
        
        search_url = f"{base_url}/search/advanced/{encoded_keyword}/$N/{court_id}/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1"
        
        print(f"[LOCAL_SCRAPER] 🔍 URL: {search_url}")
        
        try:
            response = self.session.get(search_url, timeout=CONFIG.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            judgments = self._parse_search_results(soup, base_url, keyword)
            
            return {
                "status": "success",
                "search_url": search_url,
                "judgments": judgments[:max_results]
            }
            
        except requests.RequestException as e:
            print(f"[LOCAL_SCRAPER] ❌ Error: {e}")
            return {"status": "error", "error": str(e), "judgments": []}
    
    def _parse_search_results(self, soup: BeautifulSoup, base_url: str, keyword: str) -> List[Dict]:
        """Parsuje wyniki wyszukiwania z HTML."""
        results = []
        encoded_keyword = self._encode_keyword_for_url(keyword)
        
        result_links = (
            soup.select("a[href*='/details/']") or 
            soup.select("a[href*='/content/']") or
            soup.select("table.searchResults a")
        )
        
        seen_ids = set()
        
        for link in result_links[:30]:
            try:
                href = link.get("href", "")
                if not href:
                    continue
                
                judgment_id = self._extract_judgment_id(href)
                if not judgment_id or judgment_id in seen_ids:
                    continue
                
                seen_ids.add(judgment_id)
                text = link.get_text(strip=True)
                signature = self._extract_signature(text)
                
                # v3.5: Prawidłowy format URL
                content_url = f"{base_url}/content/{encoded_keyword}/{judgment_id}"
                details_url = f"{base_url}/details/{encoded_keyword}/{judgment_id}"
                
                results.append({
                    "id": judgment_id,
                    "signature": signature or text[:50],
                    "date": self._extract_date_from_text(text),
                    "formatted_date": "",
                    "court": "",
                    "excerpt": "",
                    "url": content_url,
                    "details_url": details_url,
                })
                
            except Exception as e:
                print(f"[LOCAL_SCRAPER] Parse error: {e}")
                continue
        
        # Pobierz szczegóły
        for result in results[:10]:
            try:
                details = self._fetch_judgment_details(base_url, result["id"], keyword)
                result.update(details)
            except Exception as e:
                print(f"[LOCAL_SCRAPER] Details error: {e}")
        
        return results
    
    def _fetch_judgment_details(self, base_url: str, judgment_id: str, keyword: str) -> Dict:
        """Pobiera szczegóły orzeczenia."""
        encoded_keyword = self._encode_keyword_for_url(keyword)
        details_url = f"{base_url}/details/{encoded_keyword}/{judgment_id}"
        
        try:
            response = self.session.get(details_url, timeout=CONFIG.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            date = ""
            court = ""
            signature = ""
            
            meta_table = soup.select_one("table.metaTable") or soup.select_one(".details") or soup.select_one("table")
            if meta_table:
                for row in meta_table.select("tr"):
                    cells = row.select("td, th")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        if "data" in label and ("orzeczenia" in label or "wydania" in label):
                            date = value
                        elif "sąd" in label:
                            court = value
                        elif "sygnatura" in label:
                            signature = value
            
            content_url = f"{base_url}/content/{encoded_keyword}/{judgment_id}"
            excerpt = self._fetch_excerpt(content_url, keyword)
            
            return {
                "date": self._normalize_date(date),
                "formatted_date": date,
                "court": court,
                "signature": signature,
                "excerpt": excerpt,
            }
            
        except Exception as e:
            return {}
    
    def _fetch_excerpt(self, content_url: str, keyword: str, max_len: int = 300) -> str:
        """Pobiera fragment treści orzeczenia."""
        try:
            response = self.session.get(content_url, timeout=CONFIG.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            keyword_lower = keyword.lower()
            text_lower = text.lower()
            
            pos = text_lower.find(keyword_lower)
            if pos == -1:
                return text[:max_len] + "..." if len(text) > max_len else text
            
            start = max(0, pos - max_len // 2)
            end = min(len(text), pos + max_len // 2)
            
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
            
            return excerpt
            
        except Exception:
            return ""
    
    def _extract_judgment_id(self, href: str) -> Optional[str]:
        """Wyciąga ID orzeczenia z URL."""
        patterns = [
            r'/(?:details|content)/[^/]+/([^\s/]+)$',
            r'/(?:details|content)/\$N/([^\s/]+)',
            r'/([0-9]{15}_[^/\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, href)
            if match:
                return match.group(1)
        return None
    
    def _extract_signature(self, text: str) -> Optional[str]:
        """Wyciąga sygnaturę z tekstu."""
        patterns = [
            r'\b([IVX]+\s+[A-Za-z]{1,4}\s+\d+/\d{2,4})\b',
            r'(?:sygn\.?\s*(?:akt\s*)?:?\s*)([IVX]+\s+[A-Za-z]+\s+\d+/\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_date_from_text(self, text: str) -> str:
        """Wyciąga datę z tekstu."""
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 1:
                    return match.group(1)
                elif len(match.groups()) == 3:
                    g = match.groups()
                    if g[0].isdigit() and len(g[0]) == 4:
                        return f"{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}"
                    else:
                        return f"{g[2]}-{g[1].zfill(2)}-{g[0].zfill(2)}"
        return ""
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalizuje datę do formatu YYYY-MM-DD."""
        if not date_str:
            return ""
        
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
        
        months = {
            'stycznia': '01', 'lutego': '02', 'marca': '03', 'kwietnia': '04',
            'maja': '05', 'czerwca': '06', 'lipca': '07', 'sierpnia': '08',
            'września': '09', 'października': '10', 'listopada': '11', 'grudnia': '12'
        }
        
        pattern = r'(\d{1,2})\s+(\w+)\s+(\d{4})'
        match = re.search(pattern, date_str)
        if match:
            day = match.group(1).zfill(2)
            month = months.get(match.group(2).lower(), '01')
            year = match.group(3)
            return f"{year}-{month}-{day}"
        
        return date_str


# ============================================================================
# INTEGRACJA Z SAOS
# ============================================================================

def search_judgments_combined(
    keyword: str,
    include_saos: bool = True,
    include_local: bool = True,
    max_results: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """Przeszukuje zarówno SAOS jak i lokalne portale."""
    all_judgments = []
    sources = []
    
    if include_saos:
        try:
            from saos_client import search_judgments as saos_search
            saos_results = saos_search(keyword, max_results=max_results // 2, **kwargs)
            
            if saos_results.get("status") == "success":
                for j in saos_results.get("judgments", []):
                    j["source"] = "SAOS"
                all_judgments.extend(saos_results.get("judgments", []))
                sources.append("SAOS")
        except ImportError:
            print("[COMBINED] SAOS client not available")
        except Exception as e:
            print(f"[COMBINED] SAOS error: {e}")
    
    if include_local:
        try:
            scraper = LocalCourtScraper()
            local_results = scraper.search_all_portals(keyword, max_results=max_results // 2)
            
            if local_results.get("judgments"):
                for j in local_results.get("judgments", []):
                    j["source"] = f"Portal: {j.get('source_portal', 'lokalny')}"
                all_judgments.extend(local_results.get("judgments", []))
                sources.append("Lokalne portale")
        except Exception as e:
            print(f"[COMBINED] Local scraper error: {e}")
    
    # Deduplikacja
    seen_signatures = set()
    unique_judgments = []
    for j in all_judgments:
        sig = j.get("signature", "")
        if sig and sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_judgments.append(j)
        elif not sig:
            unique_judgments.append(j)
    
    unique_judgments.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    return {
        "status": "success" if unique_judgments else "no_results",
        "keyword": keyword,
        "total_found": len(unique_judgments),
        "judgments": unique_judgments[:max_results],
        "sources": sources
    }


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_scraper = None

def get_local_scraper() -> LocalCourtScraper:
    """Zwraca singleton scrapera."""
    global _scraper
    if _scraper is None:
        _scraper = LocalCourtScraper()
    return _scraper


def search_local_courts(keyword: str, **kwargs) -> Dict[str, Any]:
    """Skrót do wyszukiwania w lokalnych portalach."""
    return get_local_scraper().search_all_portals(keyword, **kwargs)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🏛️ Local Court Scraper v3.5 Test\n")
    
    scraper = LocalCourtScraper()
    
    print("Test kodowania polskich znaków:")
    print(f"  'rozwód' → '{scraper._encode_keyword_for_url('rozwód')}'")
    print(f"  'ubezwłasnowolnienie' → '{scraper._encode_keyword_for_url('ubezwłasnowolnienie')}'")
    print()
    
    print("Przykładowy URL wyszukiwania:")
    base = "https://orzeczenia.warszawa.so.gov.pl"
    keyword = "alimenty"
    encoded = scraper._encode_keyword_for_url(keyword)
    url = f"{base}/search/advanced/{encoded}/$N/15450500/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1"
    print(f"  {url}")
