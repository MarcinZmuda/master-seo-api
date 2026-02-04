"""
===============================================================================
🏥 BRAJEN MEDICAL MODULE v1.0
===============================================================================
Moduł do obsługi treści medycznych (YMYL Health) dla BRAJEN SEO Engine.

Źródła danych:
- PubMed (NCBI E-utilities) - publikacje naukowe
- ClinicalTrials.gov - badania kliniczne
- Polskie instytucje (PZH, AOTMiT, MZ, NFZ) - lokalne authority
- Claude AI - weryfikacja i scoring publikacji

Instalacja:
    pip install -r requirements.txt
    cp .env.example .env  # Uzupełnij klucze API

Użycie:
    from .medical_module import (
        detect_category,
        get_medical_context_for_article,
        validate_medical_article,
        MEDICAL_DISCLAIMER
    )
    
    # Sprawdź czy temat jest medyczny
    result = detect_category("leczenie cukrzycy typu 2")
    
    # Pobierz kontekst dla artykułu
    context = get_medical_context_for_article("leczenie cukrzycy typu 2")
    
    # Waliduj gotowy artykuł
    validation = validate_medical_article(article_text)

Flask Integration:
    from .medical_routes import medical_routes
    app.register_blueprint(medical_routes)

Autor: BRAJEN SEO Engine
Wersja: 1.0
===============================================================================
"""

__version__ = "1.0.0"
__author__ = "BRAJEN SEO Engine"

# ============================================================================
# GŁÓWNE EKSPORTY
# ============================================================================

# Główny moduł
from .medical_module import (
    detect_category,
    get_medical_context_for_article,
    validate_medical_article,
    MEDICAL_DISCLAIMER,
    MEDICAL_DISCLAIMER_SHORT,
    CONFIG,
    PUBMED_AVAILABLE,
    CLINICALTRIALS_AVAILABLE,
    POLISH_HEALTH_AVAILABLE,
    CLAUDE_VERIFIER_AVAILABLE
)

# Flask routes
from .medical_routes import (
    medical_routes,
    enhance_project_with_medical,
    check_medical_on_export
)

# Klienty (dla bezpośredniego dostępu)
from .pubmed_client import (
    PubMedClient,
    search_pubmed,
    search_pubmed_mesh,
    get_pubmed_client
)

from .clinicaltrials_client import (
    ClinicalTrialsClient,
    search_clinical_trials,
    search_completed_trials,
    get_clinicaltrials_client
)

from .polish_health_scraper import (
    PolishHealthScraper,
    search_polish_health,
    search_pzh,
    search_aotmit,
    get_polish_health_scraper
)

# Detektor terminów
from .medical_term_detector import (
    MedicalTermDetector,
    detect_medical_topic,
    build_pubmed_query,
    get_search_strategy,
    get_medical_term_detector
)

# Claude verifier
from .claude_medical_verifier import (
    verify_publications_with_claude,
    get_evidence_level,
    get_evidence_label
)

# Cytowania
from .medical_citation_generator import (
    MedicalCitationGenerator,
    CitationStyle,
    format_citation,
    format_inline,
    format_full,
    get_citation_generator
)

# ============================================================================
# NOWY FORMAT CYTOWAŃ v1.1
# ============================================================================

def format_source_link(source_type: str, url: str, source_name: str = None) -> str:
    """
    Formatuje cytowanie jako link do źródła.
    
    NOWY FORMAT v1.1:
    - Każde źródło cytowane TYLKO RAZ w artykule
    - Format: (źródło: [Nazwa](URL))
    
    Args:
        source_type: Typ źródła (pubmed, clinicaltrials, pzh, aotmit, etc.)
        url: URL do źródła
        source_name: Opcjonalna nazwa (jeśli inna niż domyślna)
    
    Examples:
        >>> format_source_link("pubmed", "https://pubmed.ncbi.nlm.nih.gov/35842190/")
        "(źródło: [PubMed](https://pubmed.ncbi.nlm.nih.gov/35842190/))"
    """
    gen = get_citation_generator()
    return gen.format_source_link(source_type, url, source_name)

# ============================================================================
# ALL EXPORTS
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Main functions
    "detect_category",
    "get_medical_context_for_article",
    "validate_medical_article",
    "format_source_link",  # NEW v1.1
    
    # Constants
    "MEDICAL_DISCLAIMER",
    "MEDICAL_DISCLAIMER_SHORT",
    "CONFIG",
    
    # Availability flags
    "PUBMED_AVAILABLE",
    "CLINICALTRIALS_AVAILABLE",
    "POLISH_HEALTH_AVAILABLE",
    "CLAUDE_VERIFIER_AVAILABLE",
    
    # Flask
    "medical_routes",
    "enhance_project_with_medical",
    "check_medical_on_export",
    
    # PubMed
    "PubMedClient",
    "search_pubmed",
    "search_pubmed_mesh",
    "get_pubmed_client",
    
    # ClinicalTrials
    "ClinicalTrialsClient",
    "search_clinical_trials",
    "search_completed_trials",
    "get_clinicaltrials_client",
    
    # Polish sources
    "PolishHealthScraper",
    "search_polish_health",
    "search_pzh",
    "search_aotmit",
    "get_polish_health_scraper",
    
    # Term detector
    "MedicalTermDetector",
    "detect_medical_topic",
    "build_pubmed_query",
    "get_search_strategy",
    "get_medical_term_detector",
    
    # Claude verifier
    "verify_publications_with_claude",
    "get_evidence_level",
    "get_evidence_label",
    
    # Citations
    "MedicalCitationGenerator",
    "CitationStyle",
    "format_citation",
    "format_inline",
    "format_full",
    "get_citation_generator"
]
