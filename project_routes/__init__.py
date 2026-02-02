"""
PROJECT ROUTES PACKAGE v44.1
============================
Modularny projekt_routes - wydzielony z monolitycznego pliku 5,800+ linii.

Struktura:
- helpers.py - funkcje pomocnicze (entity, density, coverage, suggested)
- h2_routes.py - endpointy H2 (Blueprint)
- semantic_planning.py - semantyczny planner fraz
- (TODO) create_routes.py - tworzenie projektu
- (TODO) batch_routes.py - operacje batch
- (TODO) phrase_routes.py - analiza fraz

Użycie w main app:
    from project_routes import h2_routes
    app.register_blueprint(h2_routes)

Lub bezpośredni import:
    from project_routes.helpers import calculate_suggested_v25
    from project_routes.semantic_planning import create_semantic_keyword_plan
"""

# Import sub-modules
from . import helpers
from . import semantic_planning

# Import Blueprint (lazy - may fail if dependencies missing)
try:
    from .h2_routes import h2_routes
    H2_ROUTES_AVAILABLE = True
except ImportError as e:
    h2_routes = None
    H2_ROUTES_AVAILABLE = False
    print(f"[PROJECT_ROUTES] ⚠️ h2_routes not available: {e}")

# Re-export all helpers
from .helpers import (
    # Constants
    DENSITY_OPTIMAL_MIN,
    DENSITY_OPTIMAL_MAX,
    DENSITY_ACCEPTABLE_MAX,
    DENSITY_WARNING_MAX,
    DENSITY_MAX,
    SOFT_CAP_THRESHOLD,
    SHORT_KEYWORD_MAX_WORDS,
    SHORT_KEYWORD_MAX_REDUCTION,
    SHORT_KEYWORD_ABSOLUTE_MAX,
    GEMINI_MODEL,
    
    # Entity helpers
    get_entities_to_introduce,
    get_already_defined_entities,
    get_overused_phrases,
    get_synonyms_for_overused,
    
    # Keyword distribution
    distribute_extended_keywords,
    get_section_length_guidance,
    
    # Density & soft cap
    get_adjusted_target_max,
    check_soft_cap,
    get_density_status,
    
    # Coverage
    validate_coverage,
    
    # Synonyms
    detect_main_keyword_synonyms,
    
    # Suggested calculation
    calculate_suggested_v25,
)

# Re-export semantic planning
from .semantic_planning import (
    THEMATIC_RULES,
    create_semantic_keyword_plan,
)

__version__ = "44.1"
__all__ = (
    helpers.__all__ + 
    semantic_planning.__all__ + 
    ["h2_routes", "H2_ROUTES_AVAILABLE"]
)
