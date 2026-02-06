"""
===============================================================================
🔬 PUBMED CLIENT v1.0 - NCBI E-utilities
===============================================================================
Klient do wyszukiwania publikacji medycznych w PubMed.

Używa NCBI E-utilities API:
- ESearch: wyszukiwanie → lista PMID
- EFetch: pobieranie szczegółów (abstrakt, autorzy, journal)

Rate limits:
- Bez API key: 3 req/sek
- Z API key: 10 req/sek

Dokumentacja: https://www.ncbi.nlm.nih.gov/books/NBK25497/
===============================================================================
"""

import os
import requests
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from xml.etree import ElementTree
import re


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class PubMedConfig:
    """Konfiguracja klienta PubMed."""
    
    BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Credentials - z ENV lub domyślne
    API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("NCBI_API_KEY"))
    EMAIL: str = field(default_factory=lambda: os.getenv("NCBI_EMAIL", ""))
    TOOL_NAME: str = "BRAJEN-SEO-Medical"
    
    # Rate limiting
    REQUEST_DELAY_WITH_KEY: float = 0.11  # ~10 req/sek
    REQUEST_DELAY_NO_KEY: float = 0.35    # ~3 req/sek
    TIMEOUT: int = 20
    
    # Defaults
    DEFAULT_RETMAX: int = 20
    DEFAULT_SORT: str = "relevance"  # lub "pub_date"
    
    # Filtry jakości
    MIN_YEAR: int = 2015
    
    # Preferowane typy publikacji (hierarchia dowodów)
    PREFERRED_TYPES: List[str] = field(default_factory=lambda: [
        "Meta-Analysis",
        "Systematic Review",
        "Randomized Controlled Trial",
        "Review",
        "Clinical Trial",
        "Guideline",
        "Practice Guideline"
    ])
    
    @property
    def request_delay(self) -> float:
        """Zwraca opóźnienie między requestami."""
        return self.REQUEST_DELAY_WITH_KEY if self.API_KEY else self.REQUEST_DELAY_NO_KEY


CONFIG = PubMedConfig()


# ============================================================================
# KLIENT PUBMED
# ============================================================================

class PubMedClient:
    """Klient NCBI E-utilities dla PubMed."""
    
    def __init__(self, config: PubMedConfig = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"{self.config.TOOL_NAME}/1.0"
        })
        self._last_request_time = 0
        
        # Log status
        if self.config.API_KEY:
            print(f"[PUBMED] ✅ API Key configured (10 req/sek)")
        else:
            print(f"[PUBMED] ⚠️ No API Key - limited to 3 req/sek")
    
    def _rate_limit(self):
        """Rate limiting zgodny z NCBI policy."""
        elapsed = time.time() - self._last_request_time
        delay = self.config.request_delay
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()
    
    def _build_base_params(self) -> Dict[str, str]:
        """Buduje podstawowe parametry dla każdego requestu."""
        params = {
            "tool": self.config.TOOL_NAME,
        }
        if self.config.EMAIL:
            params["email"] = self.config.EMAIL
        if self.config.API_KEY:
            params["api_key"] = self.config.API_KEY
        return params
    
    # ========================================================================
    # WYSZUKIWANIE
    # ========================================================================
    
    def search(
        self,
        query: str,
        max_results: int = 20,
        min_year: int = None,
        article_types: List[str] = None,
        sort: str = None
    ) -> Dict[str, Any]:
        """
        Wyszukuje publikacje w PubMed.
        
        Args:
            query: Zapytanie (może zawierać MeSH terms, operatory AND/OR)
            max_results: Maksymalna liczba wyników
            min_year: Minimalny rok publikacji
            article_types: Filtry typu (Review, RCT, Meta-Analysis, etc.)
            sort: Sortowanie ("relevance" lub "pub_date")
        
        Returns:
            {
                "status": "OK",
                "pmids": ["12345", "67890", ...],
                "count": 150,
                "query_translation": "diabetes[MeSH] AND treatment[Title]"
            }
        
        Example:
            >>> client.search("diabetes type 2 treatment", max_results=10)
            >>> client.search('"metformin"[MeSH] AND "type 2 diabetes"[MeSH]')
        """
        self._rate_limit()
        
        # Buduj query z filtrami
        full_query = self._build_query(query, min_year, article_types)
        
        params = self._build_base_params()
        params.update({
            "db": "pubmed",
            "term": full_query,
            "retmax": min(max_results, 10000),  # NCBI limit
            "sort": sort or self.config.DEFAULT_SORT,
            "retmode": "json",
            "usehistory": "y"
        })
        
        try:
            url = f"{self.config.BASE_URL}/esearch.fcgi"
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            result = data.get("esearchresult", {})
            
            # Sprawdź błędy
            if "ERROR" in result:
                return {
                    "status": "ERROR",
                    "error": result.get("ERROR", "Unknown error"),
                    "pmids": [],
                    "count": 0
                }
            
            return {
                "status": "OK",
                "pmids": result.get("idlist", []),
                "count": int(result.get("count", 0)),
                "query_translation": result.get("querytranslation", ""),
                "webenv": result.get("webenv"),
                "query_key": result.get("querykey")
            }
            
        except requests.exceptions.Timeout:
            return {"status": "ERROR", "error": "Request timeout", "pmids": [], "count": 0}
        except requests.exceptions.RequestException as e:
            return {"status": "ERROR", "error": str(e), "pmids": [], "count": 0}
        except Exception as e:
            print(f"[PUBMED] ❌ Search error: {e}")
            return {"status": "ERROR", "error": str(e), "pmids": [], "count": 0}
    
    def _build_query(
        self,
        base_query: str,
        min_year: int = None,
        article_types: List[str] = None
    ) -> str:
        """
        Buduje pełne zapytanie PubMed z filtrami.
        
        Dodaje:
        - Filtr roku publikacji
        - Filtr typu artykułu
        - Filtr języka (EN/PL)
        """
        parts = [f"({base_query})"]
        
        # Filtr roku
        year = min_year or self.config.MIN_YEAR
        parts.append(f"({year}:3000[pdat])")
        
        # Filtr typu artykułu (opcjonalny)
        if article_types:
            type_filters = [f'"{t}"[pt]' for t in article_types]
            parts.append(f"({' OR '.join(type_filters)})")
        
        # Filtr języka - angielski lub polski
        parts.append("(english[la] OR polish[la])")
        
        return " AND ".join(parts)
    
    # ========================================================================
    # POBIERANIE SZCZEGÓŁÓW
    # ========================================================================
    
    def fetch_details(
        self,
        pmids: List[str],
        include_abstract: bool = True
    ) -> List[Dict]:
        """
        Pobiera szczegóły publikacji po PMID.
        
        Args:
            pmids: Lista identyfikatorów PubMed
            include_abstract: Czy pobierać abstrakt
        
        Returns:
            Lista słowników z danymi publikacji:
            {
                "pmid": "12345678",
                "title": "Article title",
                "authors": ["Smith J", "Doe A"],
                "authors_short": "Smith et al.",
                "journal": "Nature Medicine",
                "journal_abbrev": "Nat Med",
                "year": "2023",
                "abstract": "Background: ...",
                "doi": "10.1038/...",
                "publication_types": ["Randomized Controlled Trial"],
                "mesh_terms": ["Diabetes Mellitus", "Metformin"],
                "source": "PubMed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
            }
        """
        if not pmids:
            return []
        
        self._rate_limit()
        
        # NCBI pozwala max 200 ID per request
        pmids_batch = pmids[:200]
        
        params = self._build_base_params()
        params.update({
            "db": "pubmed",
            "id": ",".join(pmids_batch),
            "retmode": "xml",
            "rettype": "abstract" if include_abstract else "docsum"
        })
        
        try:
            url = f"{self.config.BASE_URL}/efetch.fcgi"
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            
            return self._parse_pubmed_xml(response.text)
            
        except Exception as e:
            print(f"[PUBMED] ❌ Fetch error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parsuje XML z EFetch do listy publikacji."""
        publications = []
        
        try:
            root = ElementTree.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                pub = self._extract_article_data(article)
                if pub:
                    publications.append(pub)
                    
        except ElementTree.ParseError as e:
            print(f"[PUBMED] ❌ XML parse error: {e}")
        except Exception as e:
            print(f"[PUBMED] ❌ Parse error: {e}")
        
        return publications
    
    def _extract_article_data(self, article_elem) -> Optional[Dict]:
        """Wyciąga dane z pojedynczego elementu PubmedArticle."""
        try:
            medline = article_elem.find(".//MedlineCitation")
            if medline is None:
                return None
            
            # PMID
            pmid = medline.findtext(".//PMID", "")
            
            article = medline.find(".//Article")
            if article is None:
                return None
            
            # Tytuł
            title = article.findtext(".//ArticleTitle", "")
            # Usuń tagi HTML z tytułu
            title = re.sub(r'<[^>]+>', '', title)
            
            # Autorzy
            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                
                if last:
                    if fore:
                        authors.append(f"{last} {fore}")
                    elif initials:
                        authors.append(f"{last} {initials}")
                    else:
                        authors.append(last)
            
            # Journal
            journal_elem = article.find(".//Journal")
            journal = ""
            journal_abbrev = ""
            if journal_elem is not None:
                journal = journal_elem.findtext("Title", "")
                journal_abbrev = journal_elem.findtext("ISOAbbreviation", "")
            
            # Rok publikacji
            year = ""
            pub_date = article.find(".//Journal/JournalIssue/PubDate")
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                if not year:
                    medline_date = pub_date.findtext("MedlineDate", "")
                    if medline_date:
                        # Format: "2023 Jan-Feb" lub "2023"
                        match = re.search(r'(\d{4})', medline_date)
                        if match:
                            year = match.group(1)
            
            # Abstract
            abstract_parts = []
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                label = abstract_text.get("Label", "")
                content = "".join(abstract_text.itertext()) or ""
                
                if label and content:
                    abstract_parts.append(f"{label}: {content}")
                elif content:
                    abstract_parts.append(content)
            
            abstract = " ".join(abstract_parts)
            
            # DOI i inne identyfikatory
            doi = ""
            pmc = ""
            article_ids = article_elem.find(".//PubmedData/ArticleIdList")
            if article_ids is not None:
                for id_elem in article_ids.findall("ArticleId"):
                    id_type = id_elem.get("IdType", "")
                    if id_type == "doi":
                        doi = id_elem.text or ""
                    elif id_type == "pmc":
                        pmc = id_elem.text or ""
            
            # Typ publikacji
            pub_types = []
            for pt in article.findall(".//PublicationTypeList/PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)
            
            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            # Keywords
            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text)
            
            return {
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "authors_short": self._format_authors_short(authors),
                "journal": journal,
                "journal_abbrev": journal_abbrev or journal,
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "pmc": pmc,
                "publication_types": pub_types,
                "mesh_terms": mesh_terms[:10],  # Limit
                "keywords": keywords[:10],
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            print(f"[PUBMED] ⚠️ Extract error: {e}")
            return None
    
    def _format_authors_short(self, authors: List[str], max_authors: int = 3) -> str:
        """
        Formatuje autorów do krótkiej formy.
        
        Examples:
            ["Smith J"] → "Smith"
            ["Smith J", "Doe A"] → "Smith i Doe"
            ["Smith J", "Doe A", "Brown B", "White C"] → "Smith i wsp."
        """
        if not authors:
            return ""
        
        # Wyciągnij samo nazwisko
        def get_surname(author: str) -> str:
            parts = author.split()
            return parts[0] if parts else author
        
        if len(authors) == 1:
            return get_surname(authors[0])
        elif len(authors) == 2:
            return f"{get_surname(authors[0])} i {get_surname(authors[1])}"
        else:
            return f"{get_surname(authors[0])} i wsp."
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def search_and_fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wyszukuje i od razu pobiera szczegóły (główna metoda).
        🆕 v44.6: Z cache'em (TTL 24h).
        """
        # 🆕 Cache check
        try:
            from ymyl_cache import ymyl_cache
            cache_key = f"{query}|{max_results}|{kwargs.get('min_year', '')}"
            cached = ymyl_cache.get("pubmed", cache_key)
            if cached:
                print(f"[PUBMED] 📦 Cache HIT dla '{query[:40]}'")
                return cached
        except ImportError:
            ymyl_cache = None
            cache_key = None

        # Wyszukaj
        search_result = self.search(query, max_results, **kwargs)
        
        if search_result["status"] != "OK":
            return search_result
        
        if not search_result["pmids"]:
            return {
                "status": "NO_RESULTS",
                "query": query,
                "total_found": 0,
                "publications": [],
                "query_translation": search_result.get("query_translation", "")
            }
        
        # Pobierz szczegóły
        publications = self.fetch_details(search_result["pmids"])
        
        result = {
            "status": "OK",
            "query": query,
            "total_found": search_result["count"],
            "returned": len(publications),
            "publications": publications,
            "query_translation": search_result.get("query_translation", "")
        }
        
        # 🆕 Cache SET
        try:
            if ymyl_cache and cache_key:
                ymyl_cache.set("pubmed", cache_key, result)
        except Exception:
            pass
        
        return result
    
    def search_mesh(
        self,
        mesh_term: str,
        subheadings: List[str] = None,
        max_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wyszukuje po terminie MeSH (Medical Subject Headings).
        
        Args:
            mesh_term: Termin MeSH (np. "Diabetes Mellitus, Type 2")
            subheadings: Podkategorie MeSH (np. ["therapy", "drug therapy"])
            max_results: Maksymalna liczba wyników
        
        Example:
            >>> client.search_mesh("Diabetes Mellitus, Type 2", ["therapy"])
        """
        # Buduj zapytanie MeSH
        if subheadings:
            subh_str = ",".join(subheadings)
            query = f'"{mesh_term}/{subh_str}"[MeSH]'
        else:
            query = f'"{mesh_term}"[MeSH]'
        
        return self.search_and_fetch(query, max_results, **kwargs)
    
    def search_clinical_query(
        self,
        query: str,
        category: str = "therapy",
        max_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wyszukuje z użyciem Clinical Queries filters.
        
        Args:
            query: Zapytanie
            category: Kategoria ("therapy", "diagnosis", "prognosis", "etiology")
            max_results: Maksymalna liczba wyników
        
        Clinical Queries to specjalne filtry PubMed optymalizujące
        wyszukiwanie dla konkretnych pytań klinicznych.
        """
        # Clinical Query filters (PubMed)
        filters = {
            "therapy": "therapy[sb]",
            "diagnosis": "diagnosis[sb]",
            "prognosis": "prognosis[sb]",
            "etiology": "etiology[sb]",
            "clinical_prediction": "clinical prediction guides[sb]"
        }
        
        clinical_filter = filters.get(category, "therapy[sb]")
        full_query = f"({query}) AND {clinical_filter}"
        
        return self.search_and_fetch(full_query, max_results, **kwargs)


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_client: Optional[PubMedClient] = None


def get_pubmed_client() -> PubMedClient:
    """Zwraca singleton klienta PubMed."""
    global _client
    if _client is None:
        _client = PubMedClient()
    return _client


def search_pubmed(
    query: str,
    max_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Główna funkcja do wyszukiwania w PubMed.
    
    Args:
        query: Zapytanie (tekst lub MeSH terms)
        max_results: Max wyników (default 10)
        **kwargs: min_year, article_types, sort
    
    Returns:
        Dict z publikacjami
    
    Example:
        >>> result = search_pubmed("diabetes type 2 metformin", max_results=5)
        >>> for pub in result["publications"]:
        ...     print(f"{pub['authors_short']} ({pub['year']}): {pub['title']}")
    """
    return get_pubmed_client().search_and_fetch(query, max_results, **kwargs)


def search_pubmed_mesh(
    mesh_term: str,
    subheadings: List[str] = None,
    max_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """Wyszukiwanie po terminie MeSH."""
    return get_pubmed_client().search_mesh(mesh_term, subheadings, max_results, **kwargs)


# ============================================================================
# EXPORT
# ============================================================================

PUBMED_AVAILABLE = True

__all__ = [
    "PubMedClient",
    "PubMedConfig",
    "CONFIG",
    "get_pubmed_client",
    "search_pubmed",
    "search_pubmed_mesh",
    "PUBMED_AVAILABLE"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 PUBMED CLIENT v1.0 TEST")
    print("=" * 60)
    
    client = get_pubmed_client()
    
    # Test wyszukiwania
    print("\n📚 Test: Wyszukiwanie 'diabetes type 2 treatment'...")
    result = client.search_and_fetch(
        "diabetes type 2 treatment",
        max_results=3,
        article_types=["Review", "Meta-Analysis"]
    )
    
    print(f"Status: {result['status']}")
    print(f"Total found: {result.get('total_found', 0)}")
    print(f"Returned: {result.get('returned', 0)}")
    
    for pub in result.get("publications", [])[:3]:
        print(f"\n📄 PMID: {pub['pmid']}")
        print(f"   Title: {pub['title'][:80]}...")
        print(f"   Authors: {pub['authors_short']}")
        print(f"   Journal: {pub['journal_abbrev']} ({pub['year']})")
        print(f"   Types: {', '.join(pub['publication_types'][:3])}")
        print(f"   DOI: {pub['doi']}")
