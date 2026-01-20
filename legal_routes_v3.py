# legal_routes_v3.py
# BRAJEN Legal Module v3.0 - Endpointy ze scoringiem

"""
===============================================================================
🏛️ LEGAL ROUTES v3.0
===============================================================================

Endpointy:
- GET  /api/legal/status
- POST /api/legal/detect
- POST /api/legal/get_context   ← główny endpoint
- POST /api/legal/validate
- POST /api/legal/test_scoring  ← do debugowania

===============================================================================
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List

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
    print(f"[LEGAL_ROUTES] ⚠️ Legal Module not available: {e}")


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
        "saos_available": SAOS_AVAILABLE,
        "version": "3.0",
        "features": ["auto_detection", "judgment_scoring", "quality_filtering"],
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


@legal_routes.route("/api/legal/get_context", methods=["POST"])
def get_context_endpoint():
    """
    Pobiera kontekst prawny - 2 najlepsze orzeczenia po scoringu.
    
    Request:
    {
        "main_keyword": "alimenty na dziecko",
        "additional_keywords": [],
        "force_enable": false
    }
    
    Response:
    {
        "legal_module_active": true,
        "stats": {
            "total_found": 12543,
            "analyzed": 15,
            "passed_scoring": 8
        },
        "judgments": [
            {
                "citation": "wyrok SN z dnia 15.03.2023 (III CZP 12/23)",
                "score": 90,
                "score_details": ["✓ Zawiera przepisy", "✓ Wyrok merytoryczny", ...],
                "articles_in_text": ["art. 133 KRO", "art. 135 KRO"],
                "excerpt": "..."
            }
        ],
        "instruction": "⚖️ MODUŁ PRAWNY..."
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
        force_enable=data.get("force_enable", False)
    )
    
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
    
    result = validate_article_citations(full_text)
    
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
        "disclaimer": LEGAL_DISCLAIMER
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
    Wzbogaca dane projektu o kontekst prawny.
    Wywoływane w /api/project/create.
    """
    if not LEGAL_MODULE_ENABLED:
        return project_data
    
    legal_context = get_legal_context_for_article(
        main_keyword=main_keyword,
        additional_keywords=h2_list
    )
    
    project_data["legal_context"] = legal_context
    project_data["detected_category"] = legal_context.get("category", {}).get("detected_category", "inne")
    
    if legal_context.get("legal_module_active"):
        project_data["legal_instruction"] = legal_context.get("instruction", "")
        project_data["legal_judgments"] = legal_context.get("judgments", [])
        project_data["legal_stats"] = legal_context.get("stats", {})
    
    return project_data


def check_legal_on_export(full_text: str, category: str) -> Dict[str, Any]:
    """
    Sprawdza wymagania prawne przed eksportem.
    """
    if category != "prawo" or not LEGAL_MODULE_ENABLED:
        return {"legal_check": "SKIPPED"}
    
    validation = validate_article_citations(full_text)
    
    warnings = []
    
    if validation["citations_found"] == 0:
        warnings.append("Rozważ dodanie 1-2 orzeczeń sądowych")
    elif validation["citations_found"] > validation["citations_limit"]:
        warnings.append(f"Za dużo sygnatur ({validation['citations_found']})")
    
    if not validation["has_disclaimer"]:
        warnings.append("Dodaj disclaimer na końcu artykułu")
    
    return {
        "legal_check": validation["status"],
        "citations_found": validation["citations_found"],
        "has_disclaimer": validation["has_disclaimer"],
        "warnings": warnings
    }
