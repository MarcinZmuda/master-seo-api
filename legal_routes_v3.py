# legal_routes_v3.py
# BRAJEN Legal Module v3.5 - Z detektorem artykułów
# Claude wykrywa przepisy, szukamy orzeczeń po artykułach

"""
===============================================================================
🏛️ LEGAL ROUTES v3.5
===============================================================================

Endpointy:
- GET  /api/legal/status
- POST /api/legal/detect           ← wykrywa kategorię
- POST /api/legal/detect_articles  ← 🆕 wykrywa przepisy (Claude)
- POST /api/legal/get_judgments    ← 🆕 przepisy + orzeczenia
- POST /api/legal/get_context      ← główny endpoint (legacy)
- POST /api/legal/validate
- POST /api/legal/test_scoring

===============================================================================
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List

# Import modułu v3.0 (legacy)
try:
    from legal_module_v3 import (
        detect_category,
        get_legal_context_for_article,
        validate_article_citations,
        score_judgment,
        SAOS_AVAILABLE,
        LEGAL_DISCLAIMER,
        CONFIG
    )
    LEGAL_MODULE_ENABLED = True
    print("[LEGAL_ROUTES] ✅ Legal Module v3.0 loaded (with scoring)")
except ImportError as e:
    LEGAL_MODULE_ENABLED = False
    SAOS_AVAILABLE = False
    LEGAL_DISCLAIMER = ""
    print(f"[LEGAL_ROUTES] ⚠️ Legal Module not available: {e}")

# 🆕 v3.5: Import detektora artykułów (Claude)
try:
    from legal_article_detector import (
        detect_legal_articles,
        get_judgments_for_topic
    )
    ARTICLE_DETECTOR_AVAILABLE = True
    print("[LEGAL_ROUTES] ✅ Article Detector v3.5 loaded (Claude)")
except ImportError as e:
    ARTICLE_DETECTOR_AVAILABLE = False
    print(f"[LEGAL_ROUTES] ⚠️ Article Detector not available: {e}")


# ============================================================================
# BLUEPRINT
# ============================================================================

legal_routes = Blueprint("legal_routes", __name__)


# ============================================================================
# ENDPOINTY
# ============================================================================

@legal_routes.route("/api/legal/status", methods=["GET"])
def legal_status():
    """Status modułu prawnego."""
    return jsonify({
        "legal_module_enabled": LEGAL_MODULE_ENABLED,
        "article_detector_enabled": ARTICLE_DETECTOR_AVAILABLE,
        "saos_available": SAOS_AVAILABLE,
        "version": "3.5",
        "features": [
            "auto_detection",
            "judgment_scoring", 
            "quality_filtering",
            "article_detection_claude"  # 🆕
        ],
        "max_citations_per_article": 2,
        "min_score_to_use": 40
    })


@legal_routes.route("/api/legal/detect", methods=["POST"])
def detect_category_endpoint():
    """
    Wykrywa czy artykuł dotyczy prawa.
    
    Request:
    {
        "main_keyword": "alimenty na dziecko",
        "additional_keywords": ["jak obliczyć"]
    }
    """
    if not LEGAL_MODULE_ENABLED:
        return jsonify({"error": "Legal module not available"}), 503
    
    data = request.get_json() or {}
    main_keyword = data.get("main_keyword", "")
    
    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400
    
    result = detect_category(
        main_keyword=main_keyword,
        additional_keywords=data.get("additional_keywords", [])
    )
    
    return jsonify(result)


# ============================================================================
# 🆕 v3.5: NOWE ENDPOINTY - DETEKCJA ARTYKUŁÓW
# ============================================================================

@legal_routes.route("/api/legal/detect_articles", methods=["POST"])
def detect_articles_endpoint():
    """
    🆕 v3.5: Wykrywa przepisy prawne dla tematu używając Claude.
    
    Request:
    {
        "topic": "ubezwłasnowolnienie całkowite"
    }
    
    Response:
    {
        "status": "OK",
        "articles": ["art. 13 k.c.", "art. 544 k.p.c."],
        "main_act": "Kodeks cywilny",
        "method": "claude",
        "search_queries": ["art. 13 k.c.", "art 13 kc", ...]
    }
    """
    if not ARTICLE_DETECTOR_AVAILABLE:
        return jsonify({
            "error": "Article detector not available",
            "hint": "Check if legal_article_detector.py is in project and ANTHROPIC_API_KEY is set"
        }), 503
    
    data = request.get_json() or {}
    topic = data.get("topic", "")
    
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    
    result = detect_legal_articles(topic)
    
    return jsonify(result)


@legal_routes.route("/api/legal/get_judgments", methods=["POST"])
def get_judgments_endpoint():
    """
    🆕 v3.5: Pełny flow - wykrywa przepisy i zwraca orzeczenia.
    
    Request:
    {
        "topic": "ubezwłasnowolnienie całkowite",
        "max_results": 5
    }
    
    Response:
    {
        "status": "OK",
        "topic": "ubezwłasnowolnienie całkowite",
        "detected_articles": ["art. 13 k.c.", "art. 544 k.p.c."],
        "main_act": "Kodeks cywilny",
        "total_found": 5,
        "judgments": [
            {
                "signature": "I Ns 36/23",
                "date": "2024-06-20",
                "court": "Sąd Okręgowy w Warszawie",
                "matched_article": "art. 13 k.c.",
                "excerpt": "...przesłanki z art. 13 k.c. wymagają...",
                "url": "https://..."
            }
        ],
        "instruction": "UŻYJ MAKSYMALNIE 2 ORZECZEŃ..."
    }
    """
    if not ARTICLE_DETECTOR_AVAILABLE:
        return jsonify({
            "error": "Article detector not available",
            "hint": "Check if legal_article_detector.py is in project and ANTHROPIC_API_KEY is set"
        }), 503
    
    data = request.get_json() or {}
    topic = data.get("topic", "")
    max_results = data.get("max_results", 5)
    
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    
    result = get_judgments_for_topic(topic, max_results=max_results)
    
    return jsonify(result)


# ============================================================================
# LEGACY ENDPOINTY (v3.0)
# ============================================================================

@legal_routes.route("/api/legal/get_context", methods=["POST"])
def get_context_endpoint():
    """
    Pobiera kontekst prawny - integracja z SAOS API (saos.org.pl/api/search/judgments).

    Standaryzacja pól w top_judgments[]:
    ZAWSZE: signature, court, date, summary, type
    + aliasy: caseNumber, courtName, judgmentDate, excerpt, judgmentType
    ZAWSZE: legal_acts (nie acts)

    Request:
    {
        "main_keyword": "alimenty na dziecko",
        "additional_keywords": [],
        "force_enable": false,
        "article_hints": [],
        "search_queries": []
    }
    """
    if not LEGAL_MODULE_ENABLED:
        return jsonify({"error": "Legal module not available"}), 503

    data = request.get_json() or {}
    main_keyword = data.get("main_keyword", "")

    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400

    result = get_legal_context_for_article(
        main_keyword=main_keyword,
        additional_keywords=data.get("additional_keywords", []),
        force_enable=data.get("force_enable", False),
        article_hints=data.get("article_hints", []),
        search_queries=data.get("search_queries", []),
    )

    # Standardize top_judgments[] fields
    standardized_judgments = []
    for j in result.get("judgments", []):
        std_j = {
            # Primary standardized field names
            "signature": j.get("signature", j.get("caseNumber", "")),
            "court": j.get("court", j.get("courtName", "")),
            "date": j.get("date", j.get("judgmentDate", "")),
            "summary": j.get("summary", j.get("excerpt", j.get("text", ""))),
            "type": j.get("type", j.get("judgmentType", "")),
            # Aliases for backwards compatibility
            "caseNumber": j.get("signature", j.get("caseNumber", "")),
            "courtName": j.get("court", j.get("courtName", "")),
            "judgmentDate": j.get("date", j.get("judgmentDate", "")),
            "excerpt": j.get("summary", j.get("excerpt", j.get("text", ""))),
            "judgmentType": j.get("type", j.get("judgmentType", "")),
        }
        # Preserve extra fields
        for key in ("relevance_score", "source", "matched_article", "url",
                     "official_portal", "portal_url", "citation", "full_citation"):
            if key in j:
                std_j[key] = j[key]
        standardized_judgments.append(std_j)

    result["top_judgments"] = standardized_judgments
    # Keep "judgments" for backwards compatibility
    result["judgments"] = standardized_judgments

    # Standardize legal_acts (not acts)
    if "acts" in result and "legal_acts" not in result:
        result["legal_acts"] = result.pop("acts")

    return jsonify(result)


@legal_routes.route("/api/legal/validate", methods=["POST"])
def validate_endpoint():
    """
    Waliduje artykuł - liczba sygnatur + disclaimer.
    
    Request:
    {
        "full_text": "Cały tekst artykułu..."
    }
    """
    if not LEGAL_MODULE_ENABLED:
        return jsonify({"error": "Legal module not available"}), 503
    
    data = request.get_json() or {}
    full_text = data.get("full_text", "")
    
    if not full_text:
        return jsonify({"error": "full_text is required"}), 400
    
    result = validate_article_citations(
        full_text,
        provided_judgments=data.get("provided_judgments")
    )
    
    return jsonify(result)


@legal_routes.route("/api/legal/test_scoring", methods=["POST"])
def test_scoring_endpoint():
    """
    Testuje scoring na podanym tekście (do debugowania).
    
    Request:
    {
        "text": "Treść orzeczenia...",
        "keyword": "alimenty"
    }
    """
    if not LEGAL_MODULE_ENABLED:
        return jsonify({"error": "Legal module not available"}), 503
    
    data = request.get_json() or {}
    text = data.get("text", "")
    keyword = data.get("keyword", "")
    
    if not text or not keyword:
        return jsonify({"error": "text and keyword are required"}), 400
    
    result = score_judgment(text, keyword)
    
    return jsonify(result)


@legal_routes.route("/api/legal/disclaimer", methods=["GET"])
def get_disclaimer():
    """Zwraca tekst disclaimera."""
    return jsonify({
        "disclaimer": LEGAL_DISCLAIMER if LEGAL_MODULE_ENABLED else ""
    })


# ============================================================================
# HELPER: Integracja z project/create
# ============================================================================

def enhance_project_with_legal(
    project_data: Dict,
    main_keyword: str,
    h2_list: List[str]
) -> Dict:
    """
    🆕 v3.5: Wzbogaca dane projektu o kontekst prawny.
    
    Używa nowego detektora artykułów (Claude określa przepisy).
    Fallback na v3.0 jeśli detektor niedostępny.
    """
    
    # ================================================================
    # 1. NAJPIERW SPRÓBUJ NOWY DETEKTOR (v3.5)
    # ================================================================
    if ARTICLE_DETECTOR_AVAILABLE:
        try:
            print(f"[LEGAL] v3.5: Wykrywam przepisy dla '{main_keyword}'")
            
            legal_data = get_judgments_for_topic(main_keyword, max_results=5)
            
            if legal_data.get("status") == "OK":
                project_data["legal_context"] = {
                    "legal_module_active": True,
                    "version": "3.5",
                    "method": "article_detector"
                }
                project_data["detected_category"] = "prawo"
                project_data["detected_articles"] = legal_data.get("detected_articles", [])
                project_data["legal_instruction"] = legal_data.get("instruction", "")
                project_data["legal_judgments"] = legal_data.get("judgments", [])
                project_data["legal_stats"] = {
                    "total_found": legal_data.get("total_found", 0),
                    "articles_searched": legal_data.get("detected_articles", []),
                    "main_act": legal_data.get("main_act", "")
                }
                
                judgments_count = len(legal_data.get("judgments", []))
                articles = legal_data.get("detected_articles", [])
                
                if judgments_count > 0:
                    print(f"[LEGAL] ✅ v3.5: {judgments_count} orzeczeń dla przepisów {articles}")
                else:
                    print(f"[LEGAL] ⚠️ v3.5: Brak orzeczeń dla przepisów {articles}")
                
                return project_data
            
            elif legal_data.get("status") == "NOT_LEGAL":
                # Temat nie jest prawny
                project_data["detected_category"] = "inne"
                project_data["legal_context"] = {
                    "legal_module_active": False,
                    "reason": legal_data.get("reason", "Temat nie wymaga orzeczeń")
                }
                print(f"[LEGAL] ℹ️ Temat '{main_keyword}' nie jest prawny")
                return project_data
            
            else:
                # Status ERROR lub inny - spróbuj fallback
                print(f"[LEGAL] ⚠️ v3.5 zwrócił status: {legal_data.get('status')}, fallback na v3.0")
                
        except Exception as e:
            print(f"[LEGAL] ⚠️ Article detector error: {e}, fallback na v3.0")
    
    # ================================================================
    # 2. FALLBACK NA STARĄ WERSJĘ (v3.0)
    # ================================================================
    if LEGAL_MODULE_ENABLED:
        print(f"[LEGAL] v3.0 fallback: '{main_keyword}'")
        
        legal_context = get_legal_context_for_article(
            main_keyword=main_keyword,
            additional_keywords=h2_list
        )
        
        project_data["legal_context"] = legal_context
        project_data["legal_context"]["version"] = "3.0"
        project_data["detected_category"] = legal_context.get("category", {}).get("detected_category", "inne")
        
        if legal_context.get("legal_module_active"):
            project_data["legal_instruction"] = legal_context.get("instruction", "")
            project_data["legal_judgments"] = legal_context.get("judgments", [])
            project_data["legal_stats"] = legal_context.get("stats", {})
            
            judgments_count = len(project_data.get("legal_judgments", []))
            print(f"[LEGAL] ✅ v3.0: {judgments_count} orzeczeń")
    else:
        # Żaden moduł niedostępny
        project_data["legal_context"] = {
            "legal_module_active": False,
            "reason": "Legal module not available"
        }
        project_data["detected_category"] = "inne"
    
    return project_data


def check_legal_on_export(full_text: str, category: str, provided_judgments: list = None) -> Dict[str, Any]:
    """
    Sprawdza wymagania prawne przed eksportem.
    🆕 v44.6: Anti-hallucination — przekazuje dostarczone orzeczenia do walidacji.
    """
    if category != "prawo" or not LEGAL_MODULE_ENABLED:
        return {"legal_check": "SKIPPED"}
    
    validation = validate_article_citations(full_text, provided_judgments=provided_judgments)
    
    warnings = []
    
    if validation["signatures_count"] == 0:
        warnings.append("Rozważ dodanie 1-2 orzeczeń sądowych")
    elif validation["signatures_count"] > CONFIG.MAX_CITATIONS_PER_ARTICLE:
        warnings.append(f"Za dużo sygnatur ({validation['signatures_count']})")
    
    if not validation["has_disclaimer"]:
        warnings.append("Dodaj disclaimer na końcu artykułu")
    
    # 🆕 v44.6: Hallucination warnings
    if validation.get("signatures_hallucinated"):
        warnings.append(
            f"⚠️ {len(validation['signatures_hallucinated'])} wymyślonych sygnatur: "
            f"{', '.join(validation['signatures_hallucinated'][:2])}"
        )
    
    return {
        "legal_check": "PASSED" if validation["valid"] else "WARNING",
        "signatures_count": validation["signatures_count"],
        "signatures_verified": validation.get("signatures_verified", []),
        "signatures_hallucinated": validation.get("signatures_hallucinated", []),
        "has_disclaimer": validation["has_disclaimer"],
        "warnings": warnings
    }
