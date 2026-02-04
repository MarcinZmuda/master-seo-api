"""
===============================================================================
📚 MEDICAL CITATION GENERATOR v1.0
===============================================================================
Generator cytowań dla publikacji medycznych.

Obsługiwane style:
- NLM (Vancouver) - standard medyczny, używany przez PubMed
- APA 7th - standard psychologii/nauk społecznych
- ICMJE - International Committee of Medical Journal Editors

NLM (Vancouver) Example:
  Smith J, Doe A, Brown B. Article title. J Name. 2023;12(3):45-50. doi:10.1234/xxx

APA 7th Example:
  Smith, J., Doe, A., & Brown, B. (2023). Article title. Journal Name, 12(3), 45-50. https://doi.org/10.1234/xxx
===============================================================================
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# STYLE CYTOWAŃ
# ============================================================================

class CitationStyle(str, Enum):
    """Dostępne style cytowań."""
    NLM = "nlm"           # Vancouver - standard medyczny
    APA = "apa"           # APA 7th edition
    ICMJE = "icmje"       # ICMJE format
    POLISH = "polish"     # Format polski


# ============================================================================
# GENERATOR CYTOWAŃ
# ============================================================================

class MedicalCitationGenerator:
    """Generator cytowań medycznych."""
    
    def __init__(self, default_style: CitationStyle = CitationStyle.NLM):
        self.default_style = default_style
    
    def format_citation(
        self,
        publication: Dict,
        style: CitationStyle = None
    ) -> Dict[str, str]:
        """
        Formatuje cytowanie publikacji.
        
        Args:
            publication: Słownik z danymi publikacji
            style: Styl cytowania (default: NLM)
        
        Returns:
            {
                "inline": "Smith i wsp., 2023",
                "full": "Smith J, Doe A. Title. Journal. 2023;...",
                "doi_link": "https://doi.org/...",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/...",
                "style": "NLM"
            }
        """
        style = style or self.default_style
        
        formatters = {
            CitationStyle.NLM: self._format_nlm,
            CitationStyle.APA: self._format_apa,
            CitationStyle.ICMJE: self._format_icmje,
            CitationStyle.POLISH: self._format_polish
        }
        
        formatter = formatters.get(style, self._format_nlm)
        return formatter(publication)
    
    # ========================================================================
    # NLM (VANCOUVER) FORMAT
    # ========================================================================
    
    def _format_nlm(self, pub: Dict) -> Dict[str, str]:
        """
        Format NLM (Vancouver) - standard medyczny.
        
        Format:
        Authors. Title. Journal Abbrev. Year;Volume(Issue):Pages. doi:XXX
        
        Example:
        Smith J, Doe A, Brown B, et al. Metformin in diabetes. Lancet. 2023;401(5):123-130. doi:10.1016/xxx
        """
        # Autorzy
        authors = pub.get("authors", [])
        authors_str = self._format_authors_nlm(authors)
        
        # Tytuł (bez kropki na końcu jeśli jest)
        title = pub.get("title", "").rstrip(".")
        
        # Journal
        journal = pub.get("journal_abbrev") or pub.get("journal", "")
        
        # Rok i pozostałe dane
        year = pub.get("year", "")
        volume = pub.get("volume", "")
        issue = pub.get("issue", "")
        pages = pub.get("pages", "")
        doi = pub.get("doi", "")
        pmid = pub.get("pmid", "")
        
        # Buduj pełne cytowanie
        parts = [f"{authors_str}.", f"{title}.", f"{journal}."]
        
        # Rok i volume/issue/pages
        if year:
            year_part = year
            if volume:
                year_part += f";{volume}"
                if issue:
                    year_part += f"({issue})"
            if pages:
                year_part += f":{pages}"
            parts.append(year_part + ".")
        
        # DOI lub PMID
        if doi:
            parts.append(f"doi:{doi}")
        elif pmid:
            parts.append(f"PMID: {pmid}")
        
        full_citation = " ".join(parts)
        
        # Inline citation
        inline = self._format_inline_polish(authors, year)
        
        return {
            "inline": inline,
            "full": full_citation,
            "doi_link": f"https://doi.org/{doi}" if doi else "",
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            "style": "NLM (Vancouver)"
        }
    
    def _format_authors_nlm(self, authors: List[str], max_authors: int = 6) -> str:
        """
        Formatuje autorów w stylu NLM.
        
        Format: LastName Initials (np. Smith JA)
        >6 autorów: first 6, et al.
        """
        if not authors:
            return "[No authors listed]"
        
        formatted = []
        for author in authors[:max_authors]:
            formatted.append(self._author_to_nlm(author))
        
        if len(authors) > max_authors:
            return ", ".join(formatted) + ", et al"
        
        return ", ".join(formatted)
    
    def _author_to_nlm(self, author: str) -> str:
        """Konwertuje autora do formatu NLM (LastName Initials)."""
        parts = author.split()
        if len(parts) >= 2:
            last_name = parts[0]
            initials = "".join([p[0].upper() for p in parts[1:] if p])
            return f"{last_name} {initials}"
        return author
    
    # ========================================================================
    # APA 7TH FORMAT
    # ========================================================================
    
    def _format_apa(self, pub: Dict) -> Dict[str, str]:
        """
        Format APA 7th edition.
        
        Format:
        Authors (Year). Title. Journal Name, Volume(Issue), Pages. https://doi.org/XXX
        
        Example:
        Smith, J., Doe, A., & Brown, B. (2023). Metformin in diabetes. The Lancet, 401(5), 123-130. https://doi.org/10.1016/xxx
        """
        # Autorzy
        authors = pub.get("authors", [])
        authors_str = self._format_authors_apa(authors)
        
        # Tytuł
        title = pub.get("title", "")
        
        # Journal (pełna nazwa, kursywa w rzeczywistym użyciu)
        journal = pub.get("journal", "")
        
        # Metadane
        year = pub.get("year", "n.d.")
        volume = pub.get("volume", "")
        issue = pub.get("issue", "")
        pages = pub.get("pages", "")
        doi = pub.get("doi", "")
        
        # Buduj cytowanie
        parts = [f"{authors_str} ({year}).", f"{title}.", f"{journal}"]
        
        if volume:
            vol_part = volume
            if issue:
                vol_part += f"({issue})"
            parts[-1] += f", {vol_part}"
        
        if pages:
            parts[-1] += f", {pages}"
        
        parts[-1] += "."
        
        if doi:
            parts.append(f"https://doi.org/{doi}")
        
        full_citation = " ".join(parts)
        
        # Inline citation (APA style)
        inline = self._format_inline_apa(authors, year)
        
        return {
            "inline": inline,
            "full": full_citation,
            "doi_link": f"https://doi.org/{doi}" if doi else "",
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{pub.get('pmid', '')}/" if pub.get('pmid') else "",
            "style": "APA 7th"
        }
    
    def _format_authors_apa(self, authors: List[str], max_authors: int = 20) -> str:
        """
        Formatuje autorów w stylu APA.
        
        Format: LastName, I. (np. Smith, J. A.)
        2 autorów: Smith, J. A., & Doe, B. C.
        3-20: Smith, J. A., Doe, B. C., & Brown, D. E.
        21+: First 19..., Last
        """
        if not authors:
            return "[No authors listed]"
        
        formatted = [self._author_to_apa(a) for a in authors]
        
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]}, & {formatted[1]}"
        elif len(formatted) <= max_authors:
            return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
        else:
            # 21+ autorów
            first_19 = formatted[:19]
            return ", ".join(first_19) + ", ... " + formatted[-1]
    
    def _author_to_apa(self, author: str) -> str:
        """Konwertuje autora do formatu APA (LastName, I.)."""
        parts = author.split()
        if len(parts) >= 2:
            last_name = parts[0]
            initials = ". ".join([p[0].upper() for p in parts[1:] if p]) + "."
            return f"{last_name}, {initials}"
        return author
    
    def _format_inline_apa(self, authors: List[str], year: str) -> str:
        """Format inline dla APA."""
        if not authors:
            return f"([No authors], {year})"
        
        surnames = [a.split()[0] for a in authors if a]
        
        if len(surnames) == 1:
            return f"({surnames[0]}, {year})"
        elif len(surnames) == 2:
            return f"({surnames[0]} & {surnames[1]}, {year})"
        else:
            return f"({surnames[0]} et al., {year})"
    
    # ========================================================================
    # ICMJE FORMAT
    # ========================================================================
    
    def _format_icmje(self, pub: Dict) -> Dict[str, str]:
        """
        Format ICMJE (International Committee of Medical Journal Editors).
        Bardzo podobny do NLM, ale z drobnymi różnicami.
        """
        # Używamy NLM jako bazy
        return self._format_nlm(pub)
    
    # ========================================================================
    # FORMAT POLSKI
    # ========================================================================
    
    def _format_polish(self, pub: Dict) -> Dict[str, str]:
        """
        Format polski - zoptymalizowany dla polskich artykułów.
        
        Format:
        Autorzy: Tytuł. Czasopismo Rok; Tom(Numer): Strony.
        """
        authors = pub.get("authors", [])
        authors_str = self._format_authors_nlm(authors)
        
        title = pub.get("title", "").rstrip(".")
        journal = pub.get("journal", "")
        year = pub.get("year", "")
        volume = pub.get("volume", "")
        issue = pub.get("issue", "")
        pages = pub.get("pages", "")
        doi = pub.get("doi", "")
        
        # Buduj
        parts = [f"{authors_str}:", f"{title}.", f"{journal}"]
        
        if year:
            parts[-1] += f" {year}"
        if volume:
            parts[-1] += f"; {volume}"
            if issue:
                parts[-1] += f"({issue})"
        if pages:
            parts[-1] += f": {pages}"
        parts[-1] += "."
        
        if doi:
            parts.append(f"DOI: {doi}")
        
        full_citation = " ".join(parts)
        
        # Inline
        inline = self._format_inline_polish(authors, year)
        
        return {
            "inline": inline,
            "full": full_citation,
            "doi_link": f"https://doi.org/{doi}" if doi else "",
            "pubmed_link": f"https://pubmed.ncbi.nlm.nih.gov/{pub.get('pmid', '')}/" if pub.get('pmid') else "",
            "style": "Polski"
        }
    
    def _format_inline_polish(self, authors: List[str], year: str) -> str:
        """
        Format inline dla polskiego stylu.
        
        1 autor: "Kowalski (2023)"
        2 autorzy: "Kowalski i Nowak (2023)"
        3+ autorów: "Kowalski i wsp. (2023)"
        """
        if not authors:
            return f"(Brak autorów, {year})"
        
        # Wyciągnij nazwiska
        surnames = []
        for author in authors:
            parts = author.split()
            if parts:
                surnames.append(parts[0])
        
        if len(surnames) == 1:
            return f"{surnames[0]} ({year})"
        elif len(surnames) == 2:
            return f"{surnames[0]} i {surnames[1]} ({year})"
        else:
            return f"{surnames[0]} i wsp. ({year})"
    
    # ========================================================================
    # NOWY FORMAT: ŹRÓDŁO Z LINKIEM (v1.1)
    # ========================================================================
    
    def format_source_link(
        self,
        source_type: str,
        url: str,
        source_name: str = None
    ) -> str:
        """
        Formatuje cytowanie jako link do źródła.
        
        NOWY FORMAT v1.1:
        - Każde źródło cytowane TYLKO RAZ w artykule
        - Format: (źródło: [Nazwa](URL))
        
        Args:
            source_type: Typ źródła (pubmed, clinicaltrials, pzh, aotmit, etc.)
            url: URL do źródła
            source_name: Opcjonalna nazwa (jeśli inna niż domyślna)
        
        Returns:
            str: "(źródło: [PubMed](https://pubmed.ncbi.nlm.nih.gov/12345/))"
        
        Examples:
            >>> format_source_link("pubmed", "https://pubmed.ncbi.nlm.nih.gov/35842190/")
            "(źródło: [PubMed](https://pubmed.ncbi.nlm.nih.gov/35842190/))"
            
            >>> format_source_link("pzh", "https://www.pzh.gov.pl/artykul/")
            "(źródło: [NIZP-PZH](https://www.pzh.gov.pl/artykul/))"
        """
        # Mapowanie typów źródeł na nazwy wyświetlane
        SOURCE_NAMES = {
            "pubmed": "PubMed",
            "clinicaltrials": "ClinicalTrials.gov",
            "pzh": "NIZP-PZH",
            "aotmit": "AOTMiT",
            "mz": "Ministerstwo Zdrowia",
            "nfz": "NFZ",
            "mp": "Medycyna Praktyczna",
            "who": "WHO",
            "cdc": "CDC",
            "ema": "EMA"
        }
        
        # Użyj podanej nazwy lub domyślnej
        display_name = source_name or SOURCE_NAMES.get(source_type.lower(), source_type)
        
        return f"(źródło: [{display_name}]({url}))"
    
    def format_pubmed_source_link(self, pmid: str) -> str:
        """Formatuje link do PubMed na podstawie PMID."""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return self.format_source_link("pubmed", url)
    
    def format_clinicaltrials_source_link(self, nct_id: str) -> str:
        """Formatuje link do ClinicalTrials.gov na podstawie NCT ID."""
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        return self.format_source_link("clinicaltrials", url)
    
    def format_polish_source_link(self, source_type: str, url: str) -> str:
        """Formatuje link do polskiego źródła zdrowotnego."""
        return self.format_source_link(source_type, url)
    
    # ========================================================================
    # CYTOWANIE BADANIA KLINICZNEGO
    # ========================================================================
    
    def format_clinical_trial_citation(
        self,
        study: Dict,
        style: CitationStyle = None
    ) -> Dict[str, str]:
        """
        Formatuje cytowanie badania klinicznego z ClinicalTrials.gov.
        
        Args:
            study: Słownik z danymi badania (z clinicaltrials_client)
            style: Styl cytowania
        
        Returns:
            Cytowanie w wybranym stylu
        """
        nct_id = study.get("nct_id", "")
        title = study.get("brief_title") or study.get("title", "")
        sponsor = study.get("lead_sponsor", "")
        first_posted = study.get("first_posted", "")
        url = study.get("url", f"https://clinicaltrials.gov/study/{nct_id}")
        
        # Wyciągnij rok
        year = ""
        if first_posted:
            year = first_posted[:4]
        
        # Format NLM dla badań klinicznych
        if style == CitationStyle.APA or (style is None and self.default_style == CitationStyle.APA):
            full = f"{sponsor}. ({year}). {title}. ClinicalTrials.gov Identifier: {nct_id}. {url}"
            inline = f"({sponsor}, {year})"
        else:
            # NLM
            full = f"{title}. ClinicalTrials.gov. {year}. Identifier: {nct_id}. Available at: {url}"
            inline = f"ClinicalTrials.gov ({nct_id})"
        
        return {
            "inline": inline,
            "full": full,
            "nct_id": nct_id,
            "url": url,
            "style": str(style or self.default_style).upper()
        }


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_generator = None


def get_citation_generator(style: CitationStyle = CitationStyle.NLM) -> MedicalCitationGenerator:
    """Zwraca singleton generatora."""
    global _generator
    if _generator is None:
        _generator = MedicalCitationGenerator(default_style=style)
    return _generator


def format_citation(
    publication: Dict,
    style: CitationStyle = CitationStyle.NLM
) -> Dict[str, str]:
    """
    Główna funkcja do formatowania cytowania.
    
    Example:
        >>> citation = format_citation(publication, CitationStyle.NLM)
        >>> print(citation["full"])
        "Smith J, Doe A. Title. Journal. 2023;12:45-50. doi:10.1234/xxx"
    """
    return get_citation_generator().format_citation(publication, style)


def format_inline(publication: Dict) -> str:
    """Zwraca tylko inline citation."""
    citation = format_citation(publication)
    return citation["inline"]


def format_full(publication: Dict, style: CitationStyle = CitationStyle.NLM) -> str:
    """Zwraca tylko pełne cytowanie."""
    citation = format_citation(publication, style)
    return citation["full"]


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "MedicalCitationGenerator",
    "CitationStyle",
    "get_citation_generator",
    "format_citation",
    "format_inline",
    "format_full"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📚 MEDICAL CITATION GENERATOR v1.0 TEST")
    print("=" * 60)
    
    # Test publication
    test_pub = {
        "pmid": "12345678",
        "title": "Metformin for type 2 diabetes: A systematic review and meta-analysis",
        "authors": ["Smith John", "Doe Anna", "Brown Michael", "White Sarah", "Green Thomas", "Black Lisa", "Blue Mark"],
        "journal": "The Lancet Diabetes & Endocrinology",
        "journal_abbrev": "Lancet Diabetes Endocrinol",
        "year": "2023",
        "volume": "11",
        "issue": "5",
        "pages": "345-358",
        "doi": "10.1016/S2213-8587(23)00123-4"
    }
    
    generator = MedicalCitationGenerator()
    
    # Test wszystkich stylów
    for style in CitationStyle:
        print(f"\n{'='*60}")
        print(f"📝 Style: {style.value.upper()}")
        print("="*60)
        
        citation = generator.format_citation(test_pub, style)
        
        print(f"\nInline: {citation['inline']}")
        print(f"\nFull:\n{citation['full']}")
        
        if citation['doi_link']:
            print(f"\nDOI Link: {citation['doi_link']}")
    
    # Test clinical trial
    print(f"\n{'='*60}")
    print("🧪 Clinical Trial Citation")
    print("="*60)
    
    test_trial = {
        "nct_id": "NCT04267848",
        "brief_title": "Study of Metformin in Type 2 Diabetes",
        "lead_sponsor": "National Institute of Diabetes",
        "first_posted": "2023-05-15",
        "url": "https://clinicaltrials.gov/study/NCT04267848"
    }
    
    trial_citation = generator.format_clinical_trial_citation(test_trial)
    print(f"\nInline: {trial_citation['inline']}")
    print(f"\nFull:\n{trial_citation['full']}")
