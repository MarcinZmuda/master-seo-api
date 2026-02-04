"""
===============================================================================
🇵🇱 POLISH HEALTH SCRAPER v1.0
===============================================================================
Scraper polskich instytucji zdrowotnych (YMYL Authority Layer).

Źródła:
- NIZP-PZH (Narodowy Instytut Zdrowia Publicznego) - TOP authority
- AOTMiT (Agencja Oceny Technologii Medycznych) - rekomendacje
- Ministerstwo Zdrowia - wytyczne oficjalne
- NFZ - informacje dla pacjentów

Te źródła NIE mają publicznych API, więc używamy:
1. Scraping stron wynikowych
2. RSS feeds (gdzie dostępne)
3. Bezpośrednie linki do PDF/dokumentów
===============================================================================
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import re
from urllib.parse import urljoin, quote_plus
import time


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class PolishHealthConfig:
    """Konfiguracja scrapera."""
    
    TIMEOUT: int = 15
    REQUEST_DELAY: float = 1.0  # Szanujemy serwery
    
    # Źródła z metadanymi
    SOURCES: Dict[str, Dict] = field(default_factory=lambda: {
        "pzh": {
            "name": "Narodowy Instytut Zdrowia Publicznego PZH",
            "short_name": "NIZP-PZH",
            "base_url": "https://www.pzh.gov.pl",
            "search_pattern": "/szukaj/?s={query}",
            "authority": "TOP",
            "type": "research_institute",
            "description": "Główna instytucja badawcza zdrowia publicznego w Polsce"
        },
        "aotmit": {
            "name": "Agencja Oceny Technologii Medycznych i Taryfikacji",
            "short_name": "AOTMiT",
            "base_url": "https://www.aotm.gov.pl",
            "search_pattern": None,  # Brak wyszukiwarki, tylko przeglądanie
            "rekomendacje_url": "/rekomendacje-i-opinie/rekomendacje-prezesa/",
            "authority": "TOP",
            "type": "health_technology_assessment",
            "description": "Oficjalne rekomendacje dot. refundacji i stosowania technologii medycznych"
        },
        "mz": {
            "name": "Ministerstwo Zdrowia",
            "short_name": "MZ",
            "base_url": "https://www.gov.pl/web/zdrowie",
            "search_pattern": None,  # GOV.PL ma własny system
            "authority": "HIGH",
            "type": "government",
            "description": "Oficjalne komunikaty i wytyczne rządowe"
        },
        "nfz": {
            "name": "Narodowy Fundusz Zdrowia",
            "short_name": "NFZ",
            "base_url": "https://www.nfz.gov.pl",
            "search_pattern": "/szukaj?query={query}",
            "authority": "HIGH",
            "type": "government",
            "description": "Informacje o świadczeniach i programach zdrowotnych"
        },
        "mp": {
            "name": "Medycyna Praktyczna",
            "short_name": "MP",
            "base_url": "https://www.mp.pl",
            "search_pattern": "/szukaj?q={query}",
            "authority": "HIGH",
            "type": "medical_portal",
            "description": "Wiodący polski portal medyczny (dla lekarzy)"
        }
    })
    
    # Słowa kluczowe wskazujące na wysoką jakość dokumentu
    QUALITY_INDICATORS: List[str] = field(default_factory=lambda: [
        "wytyczne", "zalecenia", "rekomendacje", "standard",
        "raport", "opracowanie", "ekspertyza", "konsensus",
        "protokół", "schemat postępowania", "algorytm"
    ])
    
    # Słowa wykluczające (niższa jakość)
    EXCLUDE_INDICATORS: List[str] = field(default_factory=lambda: [
        "reklama", "sponsorowane", "promocja", "konkurs"
    ])


CONFIG = PolishHealthConfig()


# ============================================================================
# SCRAPER
# ============================================================================

class PolishHealthScraper:
    """Scraper polskich instytucji zdrowotnych."""
    
    def __init__(self, config: PolishHealthConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8"
        })
        self._last_request_time = 0
        
        print("[POLISH_HEALTH] ✅ Scraper initialized (5 sources)")
    
    def _rate_limit(self):
        """Rate limiting - szanujemy polskie serwery."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.REQUEST_DELAY:
            time.sleep(self.config.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _make_absolute_url(self, base_url: str, url: str) -> str:
        """Konwertuje relatywny URL na absolutny."""
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return urljoin(base_url, url)
    
    # ========================================================================
    # GŁÓWNA METODA
    # ========================================================================
    
    def search_all(
        self,
        query: str,
        max_results_per_source: int = 5,
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """
        Przeszukuje wszystkie polskie źródła.
        
        Args:
            query: Zapytanie (po polsku!)
            max_results_per_source: Max wyników z każdego źródła
            sources: Lista źródeł do przeszukania (None = wszystkie)
        
        Returns:
            {
                "status": "OK",
                "query": "...",
                "total_found": 15,
                "results": [...],
                "sources_checked": ["pzh", "aotmit", ...]
            }
        """
        all_results = []
        errors = []
        sources_checked = []
        
        sources_to_check = sources or list(self.config.SOURCES.keys())
        
        for source_key in sources_to_check:
            if source_key not in self.config.SOURCES:
                continue
                
            source_info = self.config.SOURCES[source_key]
            sources_checked.append(source_key)
            
            try:
                print(f"[POLISH_HEALTH] 🔍 Searching {source_info['short_name']}...")
                
                results = self._search_source(
                    source_key, source_info, query, max_results_per_source
                )
                
                # Dodaj metadane źródła
                for r in results:
                    r["source_key"] = source_key
                    r["source_name"] = source_info["name"]
                    r["source_short"] = source_info["short_name"]
                    r["authority"] = source_info["authority"]
                    r["source_type"] = source_info["type"]
                
                all_results.extend(results)
                print(f"[POLISH_HEALTH] ✅ {source_info['short_name']}: {len(results)} results")
                
            except Exception as e:
                error_msg = f"{source_info['short_name']}: {str(e)}"
                errors.append(error_msg)
                print(f"[POLISH_HEALTH] ⚠️ {error_msg}")
        
        # Sortuj po autorytecie
        authority_order = {"TOP": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_results.sort(key=lambda x: authority_order.get(x.get("authority", "LOW"), 3))
        
        # Dodaj scoring jakości
        for r in all_results:
            r["quality_score"] = self._calculate_quality_score(r)
        
        # Sortuj po quality_score w ramach tego samego authority
        all_results.sort(key=lambda x: (-authority_order.get(x.get("authority", "LOW"), 3), -x.get("quality_score", 0)))
        
        return {
            "status": "OK" if all_results else "NO_RESULTS",
            "query": query,
            "total_found": len(all_results),
            "results": all_results,
            "sources_checked": sources_checked,
            "errors": errors if errors else None
        }
    
    def _calculate_quality_score(self, result: Dict) -> int:
        """Oblicza score jakości dokumentu (0-100)."""
        score = 50  # Bazowy
        
        text = (result.get("title", "") + " " + result.get("excerpt", "")).lower()
        
        # Bonus za słowa jakościowe
        for indicator in self.config.QUALITY_INDICATORS:
            if indicator in text:
                score += 10
        
        # Malus za słowa wykluczające
        for indicator in self.config.EXCLUDE_INDICATORS:
            if indicator in text:
                score -= 20
        
        # Bonus za PDF (często bardziej oficjalne)
        if result.get("url", "").endswith(".pdf"):
            score += 15
        
        # Bonus za TOP authority
        if result.get("authority") == "TOP":
            score += 20
        
        return max(0, min(100, score))
    
    # ========================================================================
    # SCRAPERY DLA POSZCZEGÓLNYCH ŹRÓDEŁ
    # ========================================================================
    
    def _search_source(
        self,
        source_key: str,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Dispatcher do odpowiedniego scrapera."""
        
        method_map = {
            "pzh": self._search_pzh,
            "aotmit": self._search_aotmit,
            "mz": self._search_mz,
            "nfz": self._search_nfz,
            "mp": self._search_mp
        }
        
        method = method_map.get(source_key, self._search_generic)
        return method(source_info, query, max_results)
    
    def _search_pzh(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Wyszukiwanie w NIZP-PZH."""
        self._rate_limit()
        
        results = []
        base_url = source_info["base_url"]
        search_url = f"{base_url}/szukaj/?s={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # PZH używa różnych struktur - szukamy artykułów
            articles = soup.find_all("article")[:max_results]
            
            if not articles:
                # Fallback - szukaj div z wynikami
                articles = soup.find_all("div", class_=re.compile(r"(post|entry|result|item)"))[:max_results]
            
            for article in articles:
                title_elem = article.find(["h2", "h3", "h4", "a"])
                link_elem = article.find("a", href=True)
                excerpt_elem = article.find("p")
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = self._make_absolute_url(base_url, link_elem["href"]) if link_elem else ""
                    excerpt = excerpt_elem.get_text(strip=True)[:300] if excerpt_elem else ""
                    
                    if title and len(title) > 10:  # Filtruj śmieci
                        results.append({
                            "title": title,
                            "url": url,
                            "excerpt": excerpt,
                            "document_type": "article"
                        })
                        
        except Exception as e:
            print(f"[POLISH_HEALTH] ⚠️ PZH error: {e}")
        
        return results
    
    def _search_aotmit(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """
        Wyszukiwanie rekomendacji AOTMiT.
        AOTMiT nie ma wyszukiwarki - pobieramy listę rekomendacji.
        """
        self._rate_limit()
        
        results = []
        base_url = source_info["base_url"]
        rekom_url = base_url + source_info.get("rekomendacje_url", "/rekomendacje/")
        
        try:
            response = self.session.get(rekom_url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Szukaj linków do rekomendacji
            links = soup.find_all("a", href=True)
            query_lower = query.lower()
            
            for link in links:
                text = link.get_text(strip=True)
                href = link["href"]
                
                # Filtruj po query
                if query_lower in text.lower() or any(word in text.lower() for word in query_lower.split()):
                    if len(text) > 20 and ("rekomendacja" in text.lower() or "opinia" in text.lower() or ".pdf" in href):
                        results.append({
                            "title": text[:200],
                            "url": self._make_absolute_url(base_url, href),
                            "excerpt": "Oficjalna rekomendacja AOTMiT",
                            "document_type": "recommendation"
                        })
                        
                        if len(results) >= max_results:
                            break
                            
        except Exception as e:
            print(f"[POLISH_HEALTH] ⚠️ AOTMiT error: {e}")
        
        return results
    
    def _search_mz(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """
        Wyszukiwanie w serwisie gov.pl/zdrowie.
        GOV.PL ma specyficzną strukturę.
        """
        self._rate_limit()
        
        results = []
        # GOV.PL używa własnego API wyszukiwania
        search_url = f"https://www.gov.pl/web/zdrowie/szukaj?query={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Szukaj wyników
            items = soup.find_all("div", class_=re.compile(r"(search-result|article|news)"))[:max_results]
            
            for item in items:
                title_elem = item.find(["h2", "h3", "a"])
                link_elem = item.find("a", href=True)
                excerpt_elem = item.find("p")
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = link_elem["href"] if link_elem else ""
                    
                    if not url.startswith("http"):
                        url = "https://www.gov.pl" + url
                    
                    excerpt = excerpt_elem.get_text(strip=True)[:300] if excerpt_elem else ""
                    
                    if title and len(title) > 10:
                        results.append({
                            "title": title,
                            "url": url,
                            "excerpt": excerpt,
                            "document_type": "government"
                        })
                        
        except Exception as e:
            print(f"[POLISH_HEALTH] ⚠️ MZ error: {e}")
        
        return results
    
    def _search_nfz(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Wyszukiwanie w NFZ."""
        self._rate_limit()
        
        results = []
        base_url = source_info["base_url"]
        search_url = f"{base_url}/szukaj?query={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # NFZ struktura
            items = soup.find_all(["article", "div"], class_=re.compile(r"(result|item|post)"))[:max_results]
            
            for item in items:
                title_elem = item.find(["h2", "h3", "a"])
                link_elem = item.find("a", href=True)
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    url = self._make_absolute_url(base_url, link_elem["href"])
                    
                    if title and len(title) > 10:
                        results.append({
                            "title": title,
                            "url": url,
                            "excerpt": "",
                            "document_type": "nfz_info"
                        })
                        
        except Exception as e:
            print(f"[POLISH_HEALTH] ⚠️ NFZ error: {e}")
        
        return results
    
    def _search_mp(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Wyszukiwanie w Medycyna Praktyczna."""
        self._rate_limit()
        
        results = []
        base_url = source_info["base_url"]
        search_url = f"{base_url}/szukaj?q={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # MP struktura
            items = soup.find_all("div", class_=re.compile(r"(search|result|article)"))[:max_results]
            
            for item in items:
                title_elem = item.find(["h2", "h3", "a"])
                link_elem = item.find("a", href=True)
                excerpt_elem = item.find("p")
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    url = self._make_absolute_url(base_url, link_elem["href"])
                    excerpt = excerpt_elem.get_text(strip=True)[:300] if excerpt_elem else ""
                    
                    if title and len(title) > 10:
                        results.append({
                            "title": title,
                            "url": url,
                            "excerpt": excerpt,
                            "document_type": "medical_article"
                        })
                        
        except Exception as e:
            print(f"[POLISH_HEALTH] ⚠️ MP error: {e}")
        
        return results
    
    def _search_generic(
        self,
        source_info: Dict,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Generyczny scraper (fallback)."""
        return []
    
    # ========================================================================
    # METODY POMOCNICZE
    # ========================================================================
    
    def get_source_info(self, source_key: str) -> Optional[Dict]:
        """Zwraca informacje o źródle."""
        return self.config.SOURCES.get(source_key)
    
    def list_sources(self) -> List[Dict]:
        """Zwraca listę wszystkich źródeł."""
        return [
            {
                "key": k,
                "name": v["name"],
                "short_name": v["short_name"],
                "authority": v["authority"],
                "type": v["type"]
            }
            for k, v in self.config.SOURCES.items()
        ]


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_scraper: Optional[PolishHealthScraper] = None


def get_polish_health_scraper() -> PolishHealthScraper:
    """Zwraca singleton scrapera."""
    global _scraper
    if _scraper is None:
        _scraper = PolishHealthScraper()
    return _scraper


def search_polish_health(
    query: str,
    max_results_per_source: int = 5,
    sources: List[str] = None
) -> Dict[str, Any]:
    """
    Główna funkcja do wyszukiwania w polskich źródłach.
    
    Args:
        query: Zapytanie po polsku (np. "cukrzyca leczenie")
        max_results_per_source: Max wyników z każdego źródła
        sources: Lista źródeł ["pzh", "aotmit", "mz", "nfz", "mp"] lub None (wszystkie)
    
    Example:
        >>> result = search_polish_health("cukrzyca typu 2 leczenie")
        >>> for r in result["results"]:
        ...     print(f"[{r['source_short']}] {r['title']}")
    """
    return get_polish_health_scraper().search_all(
        query=query,
        max_results_per_source=max_results_per_source,
        sources=sources
    )


def search_pzh(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Wyszukuje tylko w NIZP-PZH (najwyższy autorytet)."""
    return search_polish_health(query, max_results, sources=["pzh"])


def search_aotmit(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Wyszukuje tylko rekomendacje AOTMiT."""
    return search_polish_health(query, max_results, sources=["aotmit"])


# ============================================================================
# EXPORT
# ============================================================================

POLISH_HEALTH_AVAILABLE = True

__all__ = [
    "PolishHealthScraper",
    "PolishHealthConfig",
    "CONFIG",
    "get_polish_health_scraper",
    "search_polish_health",
    "search_pzh",
    "search_aotmit",
    "POLISH_HEALTH_AVAILABLE"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🇵🇱 POLISH HEALTH SCRAPER v1.0 TEST")
    print("=" * 60)
    
    scraper = get_polish_health_scraper()
    
    # Pokaż dostępne źródła
    print("\n📚 Dostępne źródła:")
    for source in scraper.list_sources():
        print(f"   [{source['authority']}] {source['short_name']}: {source['name']}")
    
    # Test wyszukiwania
    print("\n🔍 Test: Wyszukiwanie 'cukrzyca typu 2'...")
    result = scraper.search_all(
        query="cukrzyca typu 2",
        max_results_per_source=2,
        sources=["pzh", "aotmit"]  # Tylko TOP authority
    )
    
    print(f"Status: {result['status']}")
    print(f"Total found: {result['total_found']}")
    print(f"Sources checked: {result['sources_checked']}")
    
    for r in result.get("results", [])[:5]:
        print(f"\n📄 [{r['source_short']}] (Authority: {r['authority']})")
        print(f"   Title: {r['title'][:70]}...")
        print(f"   URL: {r['url'][:60]}...")
        print(f"   Quality: {r.get('quality_score', '?')}/100")
