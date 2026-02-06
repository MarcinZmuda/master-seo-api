"""
===============================================================================
🏥 MEDICAL ROUTES v1.0 - Flask Endpoints
===============================================================================
Endpointy API dla modułu medycznego BRAJEN.

Endpoints:
- GET  /api/medical/status          → Status modułu
- POST /api/medical/detect          → Wykrywa kategorię
- POST /api/medical/get_context     → Główny endpoint - pobiera źródła
- POST /api/medical/search/pubmed   → Bezpośrednie wyszukiwanie PubMed
- POST /api/medical/search/trials   → Bezpośrednie wyszukiwanie ClinicalTrials
- POST /api/medical/search/polish   → Bezpośrednie wyszukiwanie PL
- POST /api/medical/validate        → Walidacja artykułu
- GET  /api/medical/disclaimer      → Zwraca disclaimer

Integracja z BRAJEN:
Dodaj do master_api.py:
    from medical_routes import medical_routes
    app.register_blueprint(medical_routes)

===============================================================================
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

# ============================================================================
# IMPORT MODUŁU
# ============================================================================

try:
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
        CLAUDE_VERIFIER_AVAILABLE,
        TERM_DETECTOR_AVAILABLE,
        CITATION_GENERATOR_AVAILABLE
    )
    MEDICAL_MODULE_ENABLED = True
    print("[MEDICAL_ROUTES] ✅ Medical Module loaded")
except ImportError as e:
    MEDICAL_MODULE_ENABLED = False
    PUBMED_AVAILABLE = False
    CLINICALTRIALS_AVAILABLE = False
    POLISH_HEALTH_AVAILABLE = False
    CLAUDE_VERIFIER_AVAILABLE = False
    MEDICAL_DISCLAIMER = ""
    MEDICAL_DISCLAIMER_SHORT = ""
    print(f"[MEDICAL_ROUTES] ⚠️ Medical Module not available: {e}")

# Import poszczególnych klientów (dla bezpośredniego dostępu)
try:
    from .pubmed_client import search_pubmed
except ImportError:
    search_pubmed = None

try:
    from .clinicaltrials_client import search_clinical_trials, search_completed_trials
except ImportError:
    search_clinical_trials = None
    search_completed_trials = None

try:
    from .polish_health_scraper import search_polish_health
except ImportError:
    search_polish_health = None


# ============================================================================
# BLUEPRINT
# ============================================================================

medical_routes = Blueprint("medical_routes", __name__)


# ============================================================================
# ENDPOINTS
# ============================================================================

@medical_routes.route("/api/medical/status", methods=["GET"])
def medical_status():
    """
    Status modułu medycznego.
    
    Response:
    {
        "medical_module_enabled": true,
        "version": "1.0",
        "sources": {...},
        "config": {...}
    }
    """
    return jsonify({
        "medical_module_enabled": MEDICAL_MODULE_ENABLED,
        "version": "1.0",
        "sources": {
            "pubmed": PUBMED_AVAILABLE,
            "clinicaltrials": CLINICALTRIALS_AVAILABLE,
            "polish_health": POLISH_HEALTH_AVAILABLE,
            "claude_verifier": CLAUDE_VERIFIER_AVAILABLE,
            "term_detector": TERM_DETECTOR_AVAILABLE if MEDICAL_MODULE_ENABLED else False,
            "citation_generator": CITATION_GENERATOR_AVAILABLE if MEDICAL_MODULE_ENABLED else False
        },
        "config": {
            "max_citations_per_article": CONFIG.MAX_CITATIONS_PER_ARTICLE if MEDICAL_MODULE_ENABLED else 3,
            "min_year": CONFIG.MIN_YEAR if MEDICAL_MODULE_ENABLED else 2015,
            "preferred_article_types": CONFIG.PREFERRED_ARTICLE_TYPES if MEDICAL_MODULE_ENABLED else []
        },
        "endpoints": [
            "/api/medical/status",
            "/api/medical/detect",
            "/api/medical/get_context",
            "/api/medical/search/pubmed",
            "/api/medical/search/trials",
            "/api/medical/search/polish",
            "/api/medical/validate",
            "/api/medical/disclaimer"
        ]
    })


@medical_routes.route("/api/medical/detect", methods=["POST"])
def detect_medical_category():
    """
    Wykrywa czy temat jest medyczny.
    
    Request:
    {
        "main_keyword": "leczenie cukrzycy typu 2",
        "additional_keywords": ["metformina", "dieta"]  // opcjonalne
    }
    
    Response:
    {
        "category": "medycyna",
        "is_ymyl": true,
        "confidence": 0.95,
        "specialization": "endokrynologia",
        ...
    }
    """
    if not MEDICAL_MODULE_ENABLED:
        return jsonify({"error": "Medical module not available"}), 503
    
    data = request.get_json() or {}
    main_keyword = data.get("main_keyword", "")
    
    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400
    
    result = detect_category(
        main_keyword=main_keyword,
        additional_keywords=data.get("additional_keywords", [])
    )
    
    return jsonify(result)


@medical_routes.route("/api/medical/get_context", methods=["POST"])
def get_medical_context():
    """
    Główny endpoint - pobiera kontekst medyczny dla artykułu.
    
    Request:
    {
        "main_keyword": "leczenie cukrzycy typu 2",
        "additional_keywords": [],
        "max_results": 3,
        "include_clinical_trials": true,
        "include_polish_sources": true,
        "force_enable": false,
        "compact": true  // ← NOWE: dla GPT Actions (mniejsza odpowiedź)
    }
    
    Response (compact=true):
    {
        "status": "OK",
        "is_medical": true,
        "sources": [
            {"type": "pubmed", "title": "...", "url": "...", "cite_as": "..."},
            ...
        ],
        "instruction_short": "...",
        "disclaimer": "..."
    }
    """
    if not MEDICAL_MODULE_ENABLED:
        return jsonify({"error": "Medical module not available"}), 503
    
    data = request.get_json() or {}
    main_keyword = data.get("main_keyword", "")
    compact = data.get("compact", True)  # Domyślnie compact dla GPT
    
    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400
    
    result = get_medical_context_for_article(
        main_keyword=main_keyword,
        additional_keywords=data.get("additional_keywords", []),
        max_results=data.get("max_results", 3),  # Domyślnie 3
        include_clinical_trials=data.get("include_clinical_trials", True),
        include_polish_sources=data.get("include_polish_sources", True),
        force_enable=data.get("force_enable", False)
    )
    
    # COMPACT MODE dla GPT Actions (mniejsza odpowiedź)
    if compact:
        return jsonify(_make_compact_response(result, main_keyword))
    
    return jsonify(result)


def _make_compact_response(result: Dict[str, Any], main_keyword: str) -> Dict[str, Any]:
    """
    Tworzy kompaktową odpowiedź dla GPT Actions (<50KB).
    """
    sources = []
    
    # Publikacje PubMed (tylko essentials)
    for pub in result.get("publications", [])[:3]:
        pmid = pub.get('pmid', '')
        url = pub.get('url', f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        sources.append({
            "type": "pubmed",
            "title": pub.get('title', '')[:100],
            "authors": pub.get('authors_short', ''),
            "year": pub.get('year', ''),
            "url": url,
            "cite_as": f"(źródło: [PubMed]({url}))"
        })
    
    # Badania kliniczne
    for study in result.get("clinical_trials", [])[:2]:
        nct_id = study.get('nct_id', '')
        url = study.get('url', f"https://clinicaltrials.gov/study/{nct_id}")
        sources.append({
            "type": "clinicaltrials",
            "title": study.get('brief_title', '')[:100],
            "nct_id": nct_id,
            "status": study.get('status_pl', study.get('status', '')),
            "url": url,
            "cite_as": f"(źródło: [ClinicalTrials.gov]({url}))"
        })
    
    # Polskie źródła
    SOURCE_NAMES = {"PZH": "NIZP-PZH", "NIZP-PZH": "NIZP-PZH", "AOTMiT": "AOTMiT", "MZ": "Ministerstwo Zdrowia"}
    for src in result.get("polish_sources", [])[:2]:
        source_short = src.get('source_short', 'PL')
        source_name = SOURCE_NAMES.get(source_short, source_short)
        url = src.get('url', '')
        sources.append({
            "type": "polish",
            "source": source_name,
            "title": src.get('title', '')[:100],
            "url": url,
            "cite_as": f"(źródło: [{source_name}]({url}))"
        })
    
    # Krótka instrukcja
    instruction_short = f"""
ARTYKUŁ MEDYCZNY: {main_keyword}

ZASADY CYTOWANIA:
• Format: (źródło: [Nazwa](URL))
• Każde źródło cytuj TYLKO RAZ w artykule
• Dodaj disclaimer na końcu

ŹRÓDŁA DO UŻYCIA:
""".strip()
    
    for i, src in enumerate(sources, 1):
        instruction_short += f"\n{i}. {src['cite_as']} - {src['title'][:50]}..."
    
    instruction_short += f"""

DISCLAIMER (dodaj na końcu):
Ten artykuł ma charakter informacyjny i nie zastępuje porady lekarskiej.
"""
    
    category = result.get("category", {})
    
    return {
        "status": result.get("status", "OK"),
        "is_medical": category.get("is_ymyl", False),
        "confidence": category.get("confidence", 0),
        "specialization": category.get("specialization"),
        "total_sources": len(sources),
        "sources": sources,
        "instruction": instruction_short,
        "disclaimer": "Ten artykuł ma charakter informacyjny i nie zastępuje porady lekarskiej. W przypadku problemów zdrowotnych skonsultuj się z lekarzem.",
        "citation_format": "(źródło: [Nazwa](URL))",
        "medical_module_version": "1.1"
    }


# ============================================================================
# BEZPOŚREDNIE WYSZUKIWANIE
# ============================================================================

@medical_routes.route("/api/medical/search/pubmed", methods=["POST"])
def search_pubmed_direct():
    """
    Bezpośrednie wyszukiwanie w PubMed.
    
    Request:
    {
        "query": "diabetes type 2 metformin",
        "max_results": 10,
        "min_year": 2018,
        "article_types": ["Systematic Review", "Meta-Analysis"]
    }
    
    Response:
    {
        "status": "OK",
        "query": "...",
        "total_found": 1500,
        "publications": [...]
    }
    """
    if not PUBMED_AVAILABLE or search_pubmed is None:
        return jsonify({"error": "PubMed client not available"}), 503
    
    data = request.get_json() or {}
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "query is required"}), 400
    
    result = search_pubmed(
        query=query,
        max_results=data.get("max_results", 10),
        min_year=data.get("min_year"),
        article_types=data.get("article_types")
    )
    
    return jsonify(result)


@medical_routes.route("/api/medical/search/trials", methods=["POST"])
def search_trials_direct():
    """
    Bezpośrednie wyszukiwanie w ClinicalTrials.gov.
    
    Request:
    {
        "condition": "diabetes",
        "intervention": "metformin",
        "max_results": 10,
        "status": ["COMPLETED"],
        "phase": ["PHASE3", "PHASE4"],
        "completed_only": true
    }
    
    Response:
    {
        "status": "OK",
        "total_count": 500,
        "studies": [...]
    }
    """
    if not CLINICALTRIALS_AVAILABLE or search_clinical_trials is None:
        return jsonify({"error": "ClinicalTrials client not available"}), 503
    
    data = request.get_json() or {}
    condition = data.get("condition", "")
    
    if not condition:
        return jsonify({"error": "condition is required"}), 400
    
    # Użyj search_completed_trials jeśli completed_only=true
    if data.get("completed_only", False):
        result = search_completed_trials(
            condition=condition,
            intervention=data.get("intervention"),
            max_results=data.get("max_results", 10)
        )
    else:
        result = search_clinical_trials(
            condition=condition,
            intervention=data.get("intervention"),
            max_results=data.get("max_results", 10),
            status=data.get("status"),
            phase=data.get("phase")
        )
    
    return jsonify(result)


@medical_routes.route("/api/medical/search/polish", methods=["POST"])
def search_polish_direct():
    """
    Bezpośrednie wyszukiwanie w polskich źródłach.
    
    Request:
    {
        "query": "cukrzyca typu 2 leczenie",
        "max_results_per_source": 5,
        "sources": ["pzh", "aotmit"]
    }
    
    Response:
    {
        "status": "OK",
        "total_found": 10,
        "results": [...]
    }
    """
    if not POLISH_HEALTH_AVAILABLE or search_polish_health is None:
        return jsonify({"error": "Polish Health scraper not available"}), 503
    
    data = request.get_json() or {}
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "query is required"}), 400
    
    result = search_polish_health(
        query=query,
        max_results_per_source=data.get("max_results_per_source", 5),
        sources=data.get("sources")
    )
    
    return jsonify(result)


# ============================================================================
# WALIDACJA
# ============================================================================

@medical_routes.route("/api/medical/validate", methods=["POST"])
def validate_article():
    """
    Waliduje artykuł medyczny.
    
    Sprawdza:
    - Obecność cytowań naukowych
    - Obecność disclaimera
    - Liczbę cytowań
    
    Request:
    {
        "full_text": "Cały tekst artykułu..."
    }
    
    Response:
    {
        "valid": true,
        "citations_found": 3,
        "has_disclaimer": true,
        "warnings": [],
        "suggestions": []
    }
    """
    if not MEDICAL_MODULE_ENABLED:
        return jsonify({"error": "Medical module not available"}), 503
    
    data = request.get_json() or {}
    full_text = data.get("full_text", "")
    
    if not full_text:
        return jsonify({"error": "full_text is required"}), 400
    
    result = validate_medical_article(
        full_text,
        provided_publications=data.get("provided_publications")
    )
    
    return jsonify(result)


@medical_routes.route("/api/medical/disclaimer", methods=["GET"])
def get_disclaimer():
    """
    Zwraca tekst disclaimera medycznego.
    
    Query params:
    - short=true → zwraca krótką wersję
    
    Response:
    {
        "disclaimer": "...",
        "short": "..."
    }
    """
    return jsonify({
        "disclaimer": MEDICAL_DISCLAIMER if MEDICAL_MODULE_ENABLED else "",
        "short": MEDICAL_DISCLAIMER_SHORT if MEDICAL_MODULE_ENABLED else ""
    })


# ============================================================================
# HELPER: Integracja z project_routes
# ============================================================================

def enhance_project_with_medical(
    project_data: Dict,
    main_keyword: str,
    h2_list: list = None
) -> Dict:
    """
    Wzbogaca dane projektu o kontekst medyczny.
    
    Używane przez project_routes.py przy tworzeniu projektu.
    
    Args:
        project_data: Słownik z danymi projektu
        main_keyword: Główne słowo kluczowe
        h2_list: Lista H2 (opcjonalne)
    
    Returns:
        project_data wzbogacone o medical_context
    """
    if not MEDICAL_MODULE_ENABLED:
        project_data["medical_context"] = {
            "medical_module_active": False,
            "reason": "Medical module not available"
        }
        return project_data
    
    h2_list = h2_list or []
    
    # Pobierz kontekst
    medical_context = get_medical_context_for_article(
        main_keyword=main_keyword,
        additional_keywords=h2_list,
        include_clinical_trials=True,
        include_polish_sources=True
    )
    
    if medical_context.get("status") == "NOT_MEDICAL":
        project_data["detected_category"] = "general"
        project_data["medical_context"] = {
            "medical_module_active": False,
            "reason": "Temat nie jest medyczny"
        }
    else:
        project_data["detected_category"] = "medycyna"
        project_data["medical_context"] = medical_context
        project_data["medical_instruction"] = medical_context.get("instruction", "")
        project_data["medical_publications"] = medical_context.get("publications", [])
        project_data["medical_disclaimer"] = medical_context.get("disclaimer", "")
        
        # Stats
        project_data["medical_stats"] = {
            "publications_found": len(medical_context.get("publications", [])),
            "clinical_trials_found": len(medical_context.get("clinical_trials", [])),
            "polish_sources_found": len(medical_context.get("polish_sources", [])),
            "sources_used": medical_context.get("sources_used", [])
        }
    
    return project_data


def check_medical_on_export(full_text: str, category: str, provided_publications: list = None) -> Dict[str, Any]:
    """
    Sprawdza wymagania medyczne przed eksportem.
    🆕 v44.6: Anti-hallucination — przekazuje publikacje do walidacji.
    """
    if category != "medycyna" or not MEDICAL_MODULE_ENABLED:
        return {"medical_check": "SKIPPED", "category": category}
    
    validation = validate_medical_article(full_text, provided_publications=provided_publications)
    
    return {
        "medical_check": "PASSED" if validation["valid"] else "WARNING",
        "citations_found": validation["citations_found"],
        "verified_urls": validation.get("verified_urls", []),
        "hallucinated_urls": validation.get("hallucinated_urls", []),
        "has_disclaimer": validation["has_disclaimer"],
        "warnings": validation["warnings"],
        "suggestions": validation["suggestions"]
    }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "medical_routes",
    "enhance_project_with_medical",
    "check_medical_on_export"
]
