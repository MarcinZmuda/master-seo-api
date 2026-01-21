# google_judgment_fallback.py
# BRAJEN Legal Module - Google Fallback Search v1.0
# Fallback gdy SAOS i lokalne portale nie zwrócą wyników

"""
===============================================================================
🔍 GOOGLE JUDGMENT FALLBACK v1.0
===============================================================================

Wyszukuje orzeczenia przez Google gdy inne źródła zawiodą.

Zapytanie: site:(orzeczenia.*.gov.pl OR saos.org.pl OR sn.pl) "art. X k.c."

===============================================================================
"""

import requests
import re
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class GoogleFallbackConfig:
    """Konfiguracja wyszukiwania Google."""
    
    # User agents do rotacji (unikamy blokady)
    USER_AGENTS: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ])
    
    # Domeny z orzeczeniami
    LEGAL_DOMAINS: List[str] = field(default_factory=lambda: [
        "orzeczenia.ms.gov.pl",      # Portal Orzeczeń Sądów Powszechnych
        "saos.org.pl",                # SAOS
        "sn.pl",                      # Sąd Najwyższy
        "nsa.gov.pl",                 # Naczelny Sąd Administracyjny
        "trybunal.gov.pl",            # Trybunał Konstytucyjny
    ])
    
    # Portale lokalne (orzeczenia.*.so.gov.pl)
    LOCAL_PORTAL_PATTERN: str = "orzeczenia.*.gov.pl"
    
    TIMEOUT: int = 10
    MAX_RESULTS: int = 5
    
    # Delay między requestami (sekundy) - unikamy rate limit
    MIN_DELAY: float = 1.0
    MAX_DELAY: float = 3.0
    
    # Wzorce do ekstrakcji danych z wyników
    SIGNATURE_PATTERNS: List[str] = field(default_factory=lambda: [
        r'([IVX]+)\s+([A-Z]{1,4})\s+(\d+)/(\d{2,4})',  # III Ca 456/23
        r'([IVX]+)\s+([A-Za-z]{1,4})\s+(\d+)/(\d{2,4})',  # I ACa 190/18
        r'(sygn\.?\s*(?:akt\s*)?:?\s*)([IVX]+\s+[A-Za-z]+\s+\d+/\d+)',  # sygn. akt: ...
    ])


CONFIG = GoogleFallbackConfig()


# ============================================================================
# BUDOWANIE ZAPYTANIA
# ============================================================================

def build_google_query(
    articles: List[str],
    keyword: Optional[str] = None,
    include_local_portals: bool = True
) -> str:
    """
    Buduje optymalne zapytanie Google dla orzeczeń.
    
    Args:
        articles: Lista przepisów (np. ["art. 13 k.c.", "art. 544 k.p.c."])
        keyword: Dodatkowe słowo kluczowe (np. "ubezwłasnowolnienie")
        include_local_portals: Czy szukać też na lokalnych portalach
    
    Returns:
        Zapytanie Google
    
    Examples:
        >>> build_google_query(["art. 13 k.c."], "ubezwłasnowolnienie")
        'site:(orzeczenia.ms.gov.pl OR saos.org.pl OR sn.pl OR nsa.gov.pl OR orzeczenia.*.gov.pl) "art. 13 k.c." ubezwłasnowolnienie'
    """
    
    # 1. Buduj część site:
    domains = CONFIG.LEGAL_DOMAINS.copy()
    if include_local_portals:
        domains.append(CONFIG.LOCAL_PORTAL_PATTERN)
    
    site_part = "site:(" + " OR ".join(domains) + ")"
    
    # 2. Buduj część z przepisami (w cudzysłowach dla exact match)
    articles_part = " ".join([f'"{art}"' for art in articles[:3]])  # Max 3 przepisy
    
    # 3. Dodaj keyword jeśli jest
    keyword_part = ""
    if keyword:
        # Wyczyść keyword - usuń przepisy jeśli są w nim
        clean_keyword = keyword
        for art in articles:
            clean_keyword = clean_keyword.replace(art, "").strip()
        if clean_keyword:
            keyword_part = f' {clean_keyword}'
    
    # 4. Złóż zapytanie
    query = f'{site_part} {articles_part}{keyword_part}'
    
    return query.strip()


def normalize_article_for_search(article: str) -> List[str]:
    """
    Normalizuje przepis do różnych wariantów wyszukiwania.
    
    Args:
        article: Przepis (np. "art. 13 k.c.")
    
    Returns:
        Lista wariantów do wyszukania
    
    Examples:
        >>> normalize_article_for_search("art. 13 k.c.")
        ["art. 13 k.c.", "art 13 kc", "artykuł 13 kodeksu cywilnego"]
    """
    
    variants = [article]
    
    # Wariant bez kropek
    no_dots = article.replace(".", "").replace("  ", " ")
    if no_dots != article:
        variants.append(no_dots)
    
    # Rozwinięcia skrótów
    expansions = {
        "k.c.": "kodeksu cywilnego",
        "k.p.c.": "kodeksu postępowania cywilnego", 
        "k.r.o.": "kodeksu rodzinnego i opiekuńczego",
        "k.k.": "kodeksu karnego",
        "k.p.k.": "kodeksu postępowania karnego",
        "k.p.": "kodeksu pracy",
        "k.p.a.": "kodeksu postępowania administracyjnego",
    }
    
    for abbrev, full in expansions.items():
        if abbrev in article.lower():
            expanded = article.lower().replace(abbrev, full).replace("art.", "artykuł")
            variants.append(expanded)
            break
    
    return variants


# ============================================================================
# WYSZUKIWANIE GOOGLE
# ============================================================================

class GoogleJudgmentSearcher:
    """Wyszukiwarka orzeczeń przez Google."""
    
    def __init__(self):
        self.session = requests.Session()
        self._last_request_time = 0
    
    def _get_headers(self) -> Dict[str, str]:
        """Zwraca headers z losowym User-Agent."""
        return {
            "User-Agent": random.choice(CONFIG.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    def _wait_for_rate_limit(self):
        """Czeka odpowiednią ilość czasu między requestami."""
        elapsed = time.time() - self._last_request_time
        min_wait = CONFIG.MIN_DELAY
        
        if elapsed < min_wait:
            delay = random.uniform(CONFIG.MIN_DELAY, CONFIG.MAX_DELAY)
            time.sleep(delay)
        
        self._last_request_time = time.time()
    
    def search(
        self,
        articles: List[str],
        keyword: Optional[str] = None,
        max_results: int = CONFIG.MAX_RESULTS
    ) -> Dict[str, Any]:
        """
        Wyszukuje orzeczenia przez Google.
        
        Args:
            articles: Lista przepisów do wyszukania
            keyword: Dodatkowe słowo kluczowe
            max_results: Maksymalna liczba wyników
        
        Returns:
            Dict z wynikami
        """
        
        if not articles:
            return {
                "status": "error",
                "error": "No articles provided",
                "judgments": []
            }
        
        # Buduj zapytanie
        query = build_google_query(articles, keyword)
        print(f"[GOOGLE_FALLBACK] 🔍 Query: {query}")
        
        # Wykonaj wyszukiwanie
        self._wait_for_rate_limit()
        
        try:
            # URL wyszukiwania Google
            search_url = "https://www.google.com/search"
            params = {
                "q": query,
                "num": min(max_results + 5, 20),  # Pobierz więcej, przefiltrujemy
                "hl": "pl",
                "gl": "pl",
            }
            
            response = self.session.get(
                search_url,
                params=params,
                headers=self._get_headers(),
                timeout=CONFIG.TIMEOUT
            )
            
            # Sprawdź czy nie ma CAPTCHA
            if "captcha" in response.text.lower() or response.status_code == 429:
                print("[GOOGLE_FALLBACK] ⚠️ Rate limited or CAPTCHA")
                return {
                    "status": "rate_limited",
                    "error": "Google rate limit or CAPTCHA",
                    "judgments": []
                }
            
            response.raise_for_status()
            
            # Parsuj wyniki
            judgments = self._parse_search_results(response.text, articles, keyword)
            
            return {
                "status": "success" if judgments else "no_results",
                "query": query,
                "source": "google_fallback",
                "total_found": len(judgments),
                "judgments": judgments[:max_results]
            }
            
        except requests.RequestException as e:
            print(f"[GOOGLE_FALLBACK] ❌ Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "judgments": []
            }
    
    def _parse_search_results(
        self,
        html: str,
        articles: List[str],
        keyword: Optional[str]
    ) -> List[Dict]:
        """Parsuje wyniki wyszukiwania Google."""
        
        soup = BeautifulSoup(html, "html.parser")
        results = []
        
        # Znajdź wyniki organiczne
        # Google używa różnych struktur, próbujemy kilka selektorów
        result_divs = (
            soup.select("div.g") or 
            soup.select("div[data-hveid]") or
            soup.select("div.tF2Cxc")
        )
        
        for div in result_divs:
            try:
                # Znajdź link
                link = div.select_one("a[href^='http']") or div.select_one("a")
                if not link:
                    continue
                
                url = link.get("href", "")
                
                # Filtruj tylko domeny z orzeczeniami
                if not self._is_legal_domain(url):
                    continue
                
                # Znajdź tytuł
                title_elem = div.select_one("h3") or link
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Znajdź snippet/opis
                snippet_elem = (
                    div.select_one("div.VwiC3b") or 
                    div.select_one("span.aCOpRe") or
                    div.select_one("div[data-sncf]")
                )
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                # Wyciągnij sygnaturę z tytułu lub snippetu
                signature = self._extract_signature(title + " " + snippet)
                
                # Wyciągnij nazwę sądu
                court = self._extract_court(url, title, snippet)
                
                # Wyciągnij datę
                date = self._extract_date(title + " " + snippet)
                
                # Sprawdź czy wynik jest trafny (zawiera szukane przepisy)
                relevance = self._check_relevance(title + " " + snippet, articles, keyword)
                
                if relevance > 0:
                    results.append({
                        "signature": signature or self._extract_signature_from_url(url),
                        "court": court,
                        "date": date,
                        "title": title[:200],
                        "excerpt": snippet[:300],
                        "url": url,
                        "source": "google_fallback",
                        "relevance_score": relevance,
                        "matched_articles": [a for a in articles if a.lower() in (title + snippet).lower()]
                    })
                    
            except Exception as e:
                print(f"[GOOGLE_FALLBACK] Parse error: {e}")
                continue
        
        # Sortuj po relevance
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return results
    
    def _is_legal_domain(self, url: str) -> bool:
        """Sprawdza czy URL jest z domeny z orzeczeniami."""
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Sprawdź główne domeny
            for legal_domain in CONFIG.LEGAL_DOMAINS:
                if legal_domain in domain:
                    return True
            
            # Sprawdź lokalne portale (orzeczenia.*.gov.pl)
            if re.match(r'orzeczenia\.[a-z]+\.(?:so\.)?gov\.pl', domain):
                return True
            
            # Wzorzec dla orzeczenia.MIASTO.so.gov.pl
            if "orzeczenia" in domain and "gov.pl" in domain:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _extract_signature(self, text: str) -> Optional[str]:
        """Wyciąga sygnaturę orzeczenia z tekstu."""
        
        for pattern in CONFIG.SIGNATURE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Zwróć pełne dopasowanie lub złóż z grup
                if len(match.groups()) >= 4:
                    return f"{match.group(1)} {match.group(2)} {match.group(3)}/{match.group(4)}"
                elif len(match.groups()) >= 2:
                    return match.group(2) if match.group(2) else match.group(0)
                return match.group(0)
        
        return None
    
    def _extract_signature_from_url(self, url: str) -> str:
        """Próbuje wyciągnąć sygnaturę z URL."""
        
        # Wzorzec dla ID orzeczeń w URL
        # np. 154505000001903_VI_Cz_000266_2015_Uz_2015-06-26_001
        match = re.search(r'(\d+)_([IVX]+)_([A-Za-z]+)_(\d+)_(\d{4})', url)
        if match:
            return f"{match.group(2)} {match.group(3)} {match.group(4)}/{match.group(5)[-2:]}"
        
        # Dla SAOS: /judgments/123456
        match = re.search(r'/judgments/(\d+)', url)
        if match:
            return f"SAOS #{match.group(1)}"
        
        return ""
    
    def _extract_court(self, url: str, title: str, snippet: str) -> str:
        """Wyciąga nazwę sądu."""
        
        text = (title + " " + snippet).lower()
        
        # Mapowanie domen na sądy
        if "sn.pl" in url:
            return "Sąd Najwyższy"
        if "nsa.gov.pl" in url:
            return "Naczelny Sąd Administracyjny"
        if "trybunal.gov.pl" in url:
            return "Trybunał Konstytucyjny"
        
        # Szukaj w tekście
        court_patterns = [
            (r'sąd(?:u)?\s+najwyższ', "Sąd Najwyższy"),
            (r'sąd(?:u)?\s+apelacyjn\w+\s+w\s+(\w+)', "Sąd Apelacyjny w {0}"),
            (r'sąd(?:u)?\s+okręgow\w+\s+w\s+(\w+)', "Sąd Okręgowy w {0}"),
            (r'sąd(?:u)?\s+rejonow\w+\s+w\s+(\w+)', "Sąd Rejonowy w {0}"),
            (r's[oar]\s+w\s+(\w+)', "Sąd w {0}"),
        ]
        
        for pattern, template in court_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "{0}" in template and match.groups():
                    return template.format(match.group(1).capitalize())
                return template
        
        # Z URL portalu lokalnego
        match = re.search(r'orzeczenia\.(\w+)\.(?:so\.)?gov\.pl', url)
        if match:
            city = match.group(1).capitalize()
            return f"Sąd Okręgowy w {city}"
        
        return ""
    
    def _extract_date(self, text: str) -> str:
        """Wyciąga datę z tekstu."""
        
        # Wzorce dat
        patterns = [
            r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})',  # 15.03.2023
            r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',  # 2023-03-15
            r'(\d{1,2})\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+(\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # Konwertuj do YYYY-MM-DD
                    if groups[0].isdigit() and len(groups[0]) == 4:
                        return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
                    elif groups[2].isdigit() and len(groups[2]) == 4:
                        if groups[1].isdigit():
                            return f"{groups[2]}-{groups[1].zfill(2)}-{groups[0].zfill(2)}"
                        else:
                            # Miesiąc słownie
                            months = {
                                'stycznia': '01', 'lutego': '02', 'marca': '03',
                                'kwietnia': '04', 'maja': '05', 'czerwca': '06',
                                'lipca': '07', 'sierpnia': '08', 'września': '09',
                                'października': '10', 'listopada': '11', 'grudnia': '12'
                            }
                            month = months.get(groups[1].lower(), '01')
                            return f"{groups[2]}-{month}-{groups[0].zfill(2)}"
        
        return ""
    
    def _check_relevance(
        self,
        text: str,
        articles: List[str],
        keyword: Optional[str]
    ) -> int:
        """
        Sprawdza trafność wyniku.
        
        Returns:
            Score 0-100
        """
        
        score = 0
        text_lower = text.lower()
        
        # Punkty za przepisy
        for article in articles:
            if article.lower() in text_lower:
                score += 30
            # Sprawdź warianty
            for variant in normalize_article_for_search(article):
                if variant.lower() in text_lower:
                    score += 10
                    break
        
        # Punkty za keyword
        if keyword and keyword.lower() in text_lower:
            score += 20
        
        # Punkty za słowa kluczowe orzecznictwa
        legal_terms = ["orzeczenie", "wyrok", "postanowienie", "uzasadnienie", "sygn", "sąd"]
        for term in legal_terms:
            if term in text_lower:
                score += 5
        
        return min(score, 100)


# ============================================================================
# GŁÓWNA FUNKCJA - SINGLETON
# ============================================================================

_searcher = None

def get_google_searcher() -> GoogleJudgmentSearcher:
    """Zwraca singleton searchera."""
    global _searcher
    if _searcher is None:
        _searcher = GoogleJudgmentSearcher()
    return _searcher


def search_google_fallback(
    articles: List[str],
    keyword: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Główna funkcja - wyszukuje orzeczenia przez Google.
    
    Args:
        articles: Lista przepisów (np. ["art. 13 k.c."])
        keyword: Dodatkowe słowo kluczowe
        max_results: Max wyników (default 5)
    
    Returns:
        Dict z wynikami
    
    Example:
        >>> result = search_google_fallback(["art. 13 k.c."], "ubezwłasnowolnienie")
        >>> print(result["judgments"][0]["url"])
        "https://orzeczenia.warszawa.so.gov.pl/content/..."
    """
    
    print(f"[GOOGLE_FALLBACK] 🔍 Searching: articles={articles}, keyword={keyword}")
    
    return get_google_searcher().search(
        articles=articles,
        keyword=keyword,
        max_results=max_results
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🔍 Google Judgment Fallback v1.0 Test\n")
    
    # Test budowania zapytania
    query = build_google_query(["art. 13 k.c.", "art. 544 k.p.c."], "ubezwłasnowolnienie")
    print(f"Query: {query}\n")
    
    # Test normalizacji
    variants = normalize_article_for_search("art. 13 k.c.")
    print(f"Variants for 'art. 13 k.c.': {variants}\n")
    
    # Test wyszukiwania (ostrożnie - może być rate limit!)
    print("Testing search (may be rate limited)...")
    result = search_google_fallback(["art. 13 k.c."], "ubezwłasnowolnienie", max_results=3)
    
    print(f"Status: {result['status']}")
    print(f"Found: {result.get('total_found', 0)}")
    
    for j in result.get("judgments", []):
        print(f"\n📄 {j.get('signature', 'N/A')}")
        print(f"   Court: {j.get('court', 'N/A')}")
        print(f"   URL: {j.get('url', 'N/A')[:80]}...")
        print(f"   Relevance: {j.get('relevance_score', 0)}")
