"""
PROJECT ROUTES - v30.1 BRAJEN SEO Engine - OPTIMIZED

ZMIANY v30.1 OPTIMIZED:
- 🆕 Best-of-N domyślnie WŁĄCZONE (use_best_of_n=True)
- 🆕 Auto-approve po 2 próbach (było 3)
- 🆕 Funkcja distribute_extended_keywords dla lepszego rozłożenia fraz
- 🆕 Timeout 45s (było 60s)

ZMIANY v30.0:
- 🆕 LEGAL MODULE: Auto-detekcja kategorii "prawo"
- 🆕 SAOS Integration: Pobieranie orzeczeń sądowych
- 🆕 Judgment Scoring: Wybór najlepszych orzeczeń (40+ pkt)
- 🆕 Max 2 citations per article
- 🆕 legal_instruction w response z project/create

ZMIANY v29.2:
- NOWY ENDPOINT: generateH2Plan - generuje H2 na podstawie Semantic HTML + Content Relevancy
- H2 Generator: Intent matching, Related subtopics, PAA integration
- Frazy użytkownika MUSZĄ być w H2 (ale naturalnie!)

ZMIANY v29.1:
- NOWE PRIORYTETY: Jakość tekstu > Encje > SEO
- Elastyczne podejście do fraz (min 1×, nie blokuje za "za mało")
- Lemmatyzacja fraz (ścieżka sensoryczna = ścieżką sensoryczną)
- Wykrywanie tautologii i pleonazmów
- Auto-approve po 3 próbach

ZMIANY v26.1:
- Best-of-N batch selection (generuje 3 wersje, wybiera najlepszą)
- Intro excluded from density calculation
- Polish quality validation integrated
- EXCLUSIVE counting dla actual_uses (nie overlapping)
- Soft cap + short keyword protection
- Synonimy przy przekroczeniu fraz

LOGIKA FRAZ v29.2:
- BASIC/MAIN: min 1× (MUSI być), zalecane ilości to CEL nie wymóg
- EXTENDED: min 1× (MUSI być), potem OK
- Stuffing (>MAX): JEDYNY BLOKER!
- Brak (0×): WARNING, Claude uzupełni
- Underused (<target ale >0): OK
"""

import uuid
import re
import os
import json
import math
import spacy
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# v26.1: Keyword synonyms for exceeded keywords
try:
    from keyword_synonyms import (
        generate_exceeded_warning, 
        generate_softcap_warning,
        generate_synonyms_prompt_section,
        get_synonyms
    )
    SYNONYMS_ENABLED = True
    print("[PROJECT] Keyword synonyms module loaded")
except ImportError as e:
    SYNONYMS_ENABLED = False
    print(f"[PROJECT] Keyword synonyms not available: {e}")

# v26.1: Best-of-N batch selection
try:
    from batch_best_of_n import select_best_batch, BestOfNConfig
    BEST_OF_N_ENABLED = True
    print("[PROJECT] Best-of-N module loaded")
except ImportError as e:
    BEST_OF_N_ENABLED = False
    print(f"[PROJECT] Best-of-N not available: {e}")

# v24.0: Batch planner integration
try:
    from batch_planner import create_article_plan
    BATCH_PLANNER_ENABLED = True
    print("[PROJECT] Batch Planner loaded")
except ImportError as e:
    BATCH_PLANNER_ENABLED = False
    print(f"[PROJECT] Batch Planner not available: {e}")

# 🆕 v30.0: Legal Module integration
try:
    from legal_routes_v3 import enhance_project_with_legal, LEGAL_MODULE_ENABLED
    print("[PROJECT] ✅ Legal Module v3.0 loaded")
except ImportError as e:
    LEGAL_MODULE_ENABLED = False
    def enhance_project_with_legal(project_data, main_keyword, h2_list):
        return project_data
    print(f"[PROJECT] ⚠️ Legal Module not available: {e}")

# v27.4: Polish language quality check
try:
    from polish_language_quality import (
        quick_polish_check,
        check_collocations,
        check_banned_phrases,
        INCORRECT_COLLOCATIONS
    )
    POLISH_QUALITY_ENABLED = True
    print("[PROJECT] ✅ Polish Language Quality module loaded")
except ImportError as e:
    POLISH_QUALITY_ENABLED = False
    print(f"[PROJECT] ⚠️ Polish Quality not available: {e}")
    BATCH_PLANNER_ENABLED = False
    print(f"[PROJECT] Batch Planner not available: {e}")

# v29.2: H2 Generator - Semantic HTML + Content Relevancy
try:
    from h2_generator import generate_h2_plan, validate_h2_plan
    H2_GENERATOR_ENABLED = True
    print("[PROJECT] ✅ H2 Generator module loaded")
except ImportError as e:
    H2_GENERATOR_ENABLED = False
    print(f"[PROJECT] ⚠️ H2 Generator not available: {e}")

# ================================================================
# v29.3: Entity & N-gram Guidance Helpers
# ================================================================

def get_entities_to_introduce(top_entities: list, batch_num: int, total_batches: int, previous_texts: list) -> list:
    """
    Zwraca encje do wprowadzenia w tym batchu.
    Rozdziela encje równomiernie między batche.
    """
    if not top_entities:
        return []
    
    # Połącz poprzednie teksty
    previous_text = " ".join(previous_texts).lower()
    
    # Encje jeszcze nie użyte
    unused_entities = []
    for entity in top_entities:
        entity_name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
        if entity_name.lower() not in previous_text:
            unused_entities.append(entity)
    
    if not unused_entities:
        return []
    
    # Rozdziel między pozostałe batche
    remaining_batches = max(total_batches - batch_num + 1, 1)
    entities_per_batch = max(1, len(unused_entities) // remaining_batches)
    
    # Weź encje dla tego batcha
    start_idx = 0
    end_idx = min(entities_per_batch + 1, len(unused_entities))
    
    result = []
    for entity in unused_entities[start_idx:end_idx]:
        if isinstance(entity, dict):
            result.append({
                "name": entity.get("name", ""),
                "type": entity.get("type", "CONCEPT"),
                "definition_hint": entity.get("definition_hint", "")
            })
        else:
            result.append({
                "name": str(entity),
                "type": "CONCEPT",
                "definition_hint": ""
            })
    
    return result[:3]  # Max 3 na batch


def get_already_defined_entities(previous_texts: list) -> list:
    """
    Zwraca encje już zdefiniowane w poprzednich batchach.
    """
    if not previous_texts:
        return []
    
    previous_text = " ".join(previous_texts).lower()
    
    # Wzorce definicji
    definition_patterns = [
        r'(\w+[\w\s]*)\s+to\s+(?:proces|metoda|technika|sposób|narzędzie)',
        r'(\w+[\w\s]*),\s+czyli\s+',
        r'(\w+[\w\s]*)\s+opracował[a]?\s+',
        r'(\w+[\w\s]*)\s+stworzy[łl][a]?\s+',
        r'dr\.?\s+(\w+\s+\w+)',
        r'(\w+\s+\w+),\s+(?:amerykańsk|polsk|włosk)',
    ]
    
    defined = set()
    for pattern in definition_patterns:
        matches = re.findall(pattern, previous_text)
        for match in matches:
            if len(match) > 3:  # Min 4 znaki
                defined.add(match.strip())
    
    return list(defined)[:10]


def get_overused_phrases(previous_texts: list, main_keyword: str) -> list:
    """
    Znajduje frazy użyte zbyt często (>5x).
    """
    if not previous_texts:
        return []
    
    previous_text = " ".join(previous_texts).lower()
    
    # Zlicz wystąpienia głównej frazy i jej wariantów
    overused = []
    
    # Główna fraza
    main_count = previous_text.count(main_keyword.lower())
    if main_count > 5:
        overused.append({
            "phrase": main_keyword,
            "count": main_count,
            "warning": f"Użyto {main_count}x - rozważ synonimy"
        })
    
    # Sprawdź popularne frazy
    common_phrases = [
        "integracja sensoryczna",
        "pomoce sensoryczne", 
        "terapia si",
        "rozwój dziecka",
        "ścieżka sensoryczna"
    ]
    
    for phrase in common_phrases:
        if phrase != main_keyword.lower():
            count = previous_text.count(phrase)
            if count > 4:
                overused.append({
                    "phrase": phrase,
                    "count": count,
                    "warning": f"Użyto {count}x - rozważ synonimy"
                })
    
    return overused


def get_synonyms_for_overused(previous_texts: list, main_keyword: str) -> dict:
    """
    Zwraca synonimy dla nadużywanych fraz.
    """
    # Import słownika synonimów
    SYNONYM_MAP = {
        "pomoce sensoryczne": [
            "narzędzia terapeutyczne",
            "sprzęt SI",
            "akcesoria sensoryczne",
            "materiały do stymulacji"
        ],
        "integracja sensoryczna": [
            "SI",
            "terapia integracji sensorycznej",
            "przetwarzanie sensoryczne"
        ],
        "dziecko": [
            "maluch",
            "przedszkolak",
            "najmłodsi",
            "pociecha"
        ],
        "rozwój": [
            "postęp",
            "doskonalenie",
            "kształtowanie"
        ],
        "ścieżka sensoryczna": [
            "tor sensoryczny",
            "ścieżka dotykowa",
            "mata sensoryczna"
        ],
        "terapia": [
            "zajęcia terapeutyczne",
            "sesja",
            "ćwiczenia"
        ]
    }
    
    result = {}
    
    # Dla głównej frazy
    main_lower = main_keyword.lower()
    for key, synonyms in SYNONYM_MAP.items():
        if key in main_lower or main_lower in key:
            result[main_keyword] = synonyms
            break
    
    # Dla innych popularnych
    if not previous_texts:
        return result
    
    previous_text = " ".join(previous_texts).lower()
    
    for phrase, synonyms in SYNONYM_MAP.items():
        if phrase in previous_text and phrase not in result:
            count = previous_text.count(phrase)
            if count > 3:
                result[phrase] = synonyms
    
    return result


# ================================================================
# 🆕 v30.1: DISTRIBUTE EXTENDED KEYWORDS
# ================================================================
def distribute_extended_keywords(extended_keywords: List[Dict], total_batches: int) -> Dict[int, List[Dict]]:
    """
    🆕 v30.1: Rozdziela EXTENDED frazy równomiernie między batche.
    
    Zamiast wymagać wszystkich 25 EXTENDED w każdym batchu,
    rozdziela je tak, żeby każdy batch miał 3-5 unikalnych.
    
    Args:
        extended_keywords: Lista fraz EXTENDED
        total_batches: Liczba batchów w artykule
    
    Returns:
        Dict {batch_num: [keywords_for_this_batch]}
    """
    if not extended_keywords or total_batches < 1:
        return {}
    
    distribution = {i: [] for i in range(1, total_batches + 1)}
    
    # Rozdziel równomiernie
    keywords_per_batch = max(3, len(extended_keywords) // total_batches)
    
    for i, kw in enumerate(extended_keywords):
        batch_num = (i // keywords_per_batch) + 1
        if batch_num > total_batches:
            batch_num = total_batches
        distribution[batch_num].append(kw)
    
    # Upewnij się, że każdy batch ma min 2 i max 6
    for batch_num in distribution:
        if len(distribution[batch_num]) > 6:
            # Przenieś nadmiar do następnych batchów
            excess = distribution[batch_num][6:]
            distribution[batch_num] = distribution[batch_num][:6]
            
            for j, kw in enumerate(excess):
                next_batch = ((batch_num + j) % total_batches) + 1
                if len(distribution[next_batch]) < 6:
                    distribution[next_batch].append(kw)
    
    return distribution


def get_section_length_guidance(batch_num: int, total_batches: int, batch_type: str) -> dict:
    """
    Zwraca guidance o różnej długości sekcji.
    Każdy batch dostaje INNĄ zalecaną długość żeby uniknąć monotonii.
    """
    # Wzorce długości dla różnych batchów
    LENGTH_PATTERNS = {
        1: {"profile": "SHORT", "range": "180-220", "reason": "Intro - zwięzłe wprowadzenie"},
        2: {"profile": "LONG", "range": "350-400", "reason": "Główny temat - rozbudowana treść"},
        3: {"profile": "MEDIUM", "range": "250-300", "reason": "Rozwinięcie tematu"},
        4: {"profile": "LONG", "range": "320-380", "reason": "Praktyczne porady - więcej szczegółów"},
        5: {"profile": "MEDIUM", "range": "240-280", "reason": "Uzupełnienie tematu"},
        6: {"profile": "SHORT", "range": "200-250", "reason": "Sekcja przed FAQ - krótsza"},
    }
    
    # Pobierz pattern dla tego batcha
    pattern = LENGTH_PATTERNS.get(batch_num, {"profile": "MEDIUM", "range": "250-300", "reason": "Standardowa sekcja"})
    
    # Specjalne przypadki
    if batch_type == "INTRO":
        pattern = {"profile": "SHORT", "range": "150-200", "reason": "Intro musi być zwięzłe"}
    elif batch_type == "FAQ":
        pattern = {"profile": "VARIABLE", "range": "40-60 per answer", "reason": "FAQ - różne długości odpowiedzi"}
    
    return {
        "batch_number": batch_num,
        "recommended_profile": pattern["profile"],
        "recommended_range": pattern["range"],
        "reason": pattern["reason"],
        "variety_reminder": "⚠️ Sekcje MUSZĄ mieć RÓŻNE długości! NIE pisz wszystkich po ~250 słów!",
        "distribution_hint": {
            "short_sections": "1-2 sekcje: 180-220 słów",
            "medium_sections": "2-3 sekcje: 250-300 słów",
            "long_sections": "1-2 sekcje: 350-400 słów"
        }
    }


# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING]  GEMINI_API_KEY not set - LSI enrichment fallback mode")

# spaCy model
try:
    nlp = spacy.load("pl_core_news_md")
    print("[INIT]  spaCy pl_core_news_md loaded")
except OSError:
    from spacy.cli import download
    print(" Downloading pl_core_news_md fallback...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)

#  GEMINI MODEL - centralnie zdefiniowany
GEMINI_MODEL = "gemini-2.5-flash"

# ============================================================================
# v25.0: DENSITY CONFIGURATION
# ============================================================================
DENSITY_OPTIMAL_MIN = 0.5
DENSITY_OPTIMAL_MAX = 1.5
DENSITY_ACCEPTABLE_MAX = 2.0
DENSITY_WARNING_MAX = 2.5
DENSITY_MAX = 3.0

# ============================================================================
# v26.1: SOFT CAP & SHORT KEYWORD CONFIGURATION
# ============================================================================
# Soft cap - ostrzegaj PRZED osiągnięciem max (np. 75% = ostrzeżenie przy 75% max)
SOFT_CAP_THRESHOLD = 0.75  # 75% max = WARNING "zbliżasz się do limitu"

# Krótkie frazy (1-2 słowa) mają automatycznie niższy max
# Bo są częściej używane naturalnie i łatwo je przeoptymalizować
SHORT_KEYWORD_MAX_WORDS = 2  # Frazy <= 2 słów = "krótkie"
SHORT_KEYWORD_MAX_REDUCTION = 0.6  # Krótkie frazy mają 60% normalnego max
SHORT_KEYWORD_ABSOLUTE_MAX = 8  # Absolutny max dla krótkich fraz

def get_adjusted_target_max(keyword: str, original_max: int, word_count: int = None) -> int:
    """
    v26.1: Zwraca skorygowany target_max dla frazy.
    Krótkie frazy (1-2 słowa) mają niższy max żeby uniknąć przeoptymalizowania.
    """
    if word_count is None:
        word_count = len(keyword.split())
    
    if word_count <= SHORT_KEYWORD_MAX_WORDS:
        # Krótka fraza - zmniejsz max
        reduced_max = int(original_max * SHORT_KEYWORD_MAX_REDUCTION)
        return min(reduced_max, SHORT_KEYWORD_ABSOLUTE_MAX)
    
    return original_max

def check_soft_cap(actual: int, target_max: int, keyword: str) -> dict:
    """
    v26.1: Sprawdza czy fraza zbliża się do limitu (soft cap).
    Zwraca warning jeśli actual >= 75% target_max.
    """
    if target_max <= 0:
        return None
    
    usage_ratio = actual / target_max
    
    if usage_ratio >= 1.0:
        return {
            "type": "EXCEEDED",
            "keyword": keyword,
            "actual": actual,
            "max": target_max,
            "percent": round(usage_ratio * 100),
            "message": f"❌ PRZEKROCZONO! '{keyword}' użyta {actual}x (max: {target_max})"
        }
    elif usage_ratio >= SOFT_CAP_THRESHOLD:
        remaining = target_max - actual
        return {
            "type": "SOFT_CAP_WARNING",
            "keyword": keyword,
            "actual": actual,
            "max": target_max,
            "remaining": remaining,
            "percent": round(usage_ratio * 100),
            "message": f"⚠️ UWAGA: '{keyword}' zbliża się do limitu ({actual}/{target_max} = {round(usage_ratio*100)}%). Zostało: {remaining}x"
        }
    
    return None

def get_density_status(density: float) -> tuple:
    """v25.0: Zwraca status density z kolorowym oznaczeniem."""
    if density < DENSITY_OPTIMAL_MIN:
        return "LOW", f"⚪ Za nisko ({density:.1f}%) - dodaj więcej fraz"
    elif density <= DENSITY_OPTIMAL_MAX:
        return "OPTIMAL", f"✅ Optymalne ({density:.1f}%)"
    elif density <= DENSITY_ACCEPTABLE_MAX:
        return "ACCEPTABLE", f"🟢 OK ({density:.1f}%)"
    elif density <= DENSITY_WARNING_MAX:
        return "WARNING", f"🟡 Wysoko ({density:.1f}%) - uważaj"
    elif density <= DENSITY_MAX:
        return "HIGH", f"🟠 Za wysoko ({density:.1f}%) - ogranicz"
    else:
        return "STUFFING", f"🔴 KEYWORD STUFFING ({density:.1f}%) - przepisz!"


# ================================================================
# v29.1: COVERAGE VALIDATION
# ================================================================
def validate_coverage(keywords_state: dict) -> dict:
    """
    v29.1: Sprawdza coverage dla BASIC i EXTENDED keywords.
    
    NOWA LOGIKA:
    - BASIC: min 1× (hard requirement), target to CEL nie wymóg
    - EXTENDED: min 1× (hard requirement)
    - Underused (>0 ale <target): OK, tylko warning
    - Missing (0×): CRITICAL
    """
    basic_total = 0
    basic_covered = 0
    basic_missing = []
    basic_target_met = 0
    
    extended_total = 0
    extended_covered = 0
    extended_missing = []
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        keyword = meta.get("keyword", "")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        
        if kw_type in ["BASIC", "MAIN"]:
            basic_total += 1
            if actual >= 1:
                basic_covered += 1
            else:
                basic_missing.append(keyword)
            
            if actual >= target_min:
                basic_target_met += 1
                
        elif kw_type == "EXTENDED":
            extended_total += 1
            if actual >= 1:
                extended_covered += 1
            else:
                extended_missing.append(keyword)
    
    basic_coverage = (basic_covered / basic_total * 100) if basic_total > 0 else 100
    extended_coverage = (extended_covered / extended_total * 100) if extended_total > 0 else 100
    
    return {
        "basic": {
            "total": basic_total,
            "covered": basic_covered,
            "coverage_percent": round(basic_coverage, 1),
            "target_met": basic_target_met,
            "missing": basic_missing[:5],
            "status": "OK" if basic_coverage == 100 else "INCOMPLETE"
        },
        "extended": {
            "total": extended_total,
            "covered": extended_covered,
            "coverage_percent": round(extended_coverage, 1),
            "missing": extended_missing[:5],
            "status": "OK" if extended_coverage == 100 else "INCOMPLETE"
        },
        "overall_coverage": round((basic_coverage + extended_coverage) / 2, 1) if extended_total > 0 else basic_coverage
    }


# ================================================================
#  v22.4: SYNONYM DETECTION dla frazy głównej
# ================================================================
def detect_main_keyword_synonyms(main_keyword: str) -> list:
    """Używa Gemini do znalezienia synonimów frazy głównej."""
    if not GEMINI_API_KEY:
        return []
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
Podaj 2-4 SYNONIMY lub WARIANTY dla frazy: "{main_keyword}"

ZASADY:
- Tylko frazy które znaczą TO SAMO
- Mogą być używane zamiennie w tekście SEO
- Format: jeden synonim na linię, bez numeracji

Odpowiedź (tylko synonimy):
"""
        response = model.generate_content(prompt)
        synonyms = [s.strip() for s in response.text.strip().split('\n') if s.strip() and len(s.strip()) > 2]
        return synonyms[:4]
    except Exception as e:
        print(f"[SYNONYM]  Error: {e}")
        return []


# ================================================================
# v29.1: CALCULATE SUGGESTED - nowa logika coverage-first
# ================================================================
def calculate_suggested_v25(
    keyword: str,
    kw_type: str,
    actual: int,
    target_min: int,
    target_max: int,
    remaining_batches: int,
    total_batches: int,
    current_batch: int,
    is_main: bool = False
) -> dict:
    """
    v29.1: Logika suggested z elastycznym podejściem.
    
    NOWE ZASADY:
    - BASIC: min 1× (hard), target to CEL (zalecane, nie wymóg)
    - EXTENDED: min 1× (hard), potem OK
    - Krótkie frazy: automatycznie niższy max (ochrona przed stuffingiem)
    - Underused: sugestia ale NIE blokuje
    """
    
    # v26.1: Skoryguj max dla krótkich fraz (nie dla EXTENDED)
    word_count = len(keyword.split())
    if kw_type != "EXTENDED" and word_count <= SHORT_KEYWORD_MAX_WORDS:
        adjusted_max = get_adjusted_target_max(keyword, target_max, word_count)
        if adjusted_max < target_max:
            target_max = adjusted_max
    
    remaining_to_max = max(0, target_max - actual)
    remaining_to_min = max(0, target_min - actual)
    
    # v26.1: Sprawdź soft cap
    soft_cap_info = check_soft_cap(actual, target_max, keyword)
    soft_cap_warning = soft_cap_info.get("message") if soft_cap_info else None
    
    # === EXTENDED: min 1x, może być więcej ===
    if kw_type == "EXTENDED":
        if actual == 0:
            # v27.2: KAŻDY batch powinien użyć proporcjonalną liczbę EXTENDED
            # Nie używamy hash - rozdzielamy równomiernie
            # W ostatnich batchach wszystkie nieużyte EXTENDED muszą być użyte
            
            if remaining_batches <= 2:
                # Ostatnie 2 batchy - KRYTYCZNE, użyj wszystkie nieużyte
                return {
                    "suggested": 1,
                    "priority": "CRITICAL",
                    "instruction": f"🔴 KRYTYCZNE - MUSISZ użyć min 1x (zostały {remaining_batches} batchy!)",
                    "hard_max_this_batch": 2,
                    "flexibility": "NONE",
                    "adjusted_max": target_max
                }
            elif remaining_batches <= 3:
                # Przedostatnie batchy - HIGH priority
                return {
                    "suggested": 1,
                    "priority": "HIGH",
                    "instruction": f"📌 WPLEĆ min 1x (extended - zostały {remaining_batches} batchy)",
                    "hard_max_this_batch": 2,
                    "flexibility": "LOW",
                    "adjusted_max": target_max
                }
            else:
                # Wczesne batchy - ale i tak zachęcaj do użycia
                should_use = (hash(keyword) % total_batches) == (current_batch - 1)
                if should_use:
                    return {
                        "suggested": 1,
                        "priority": "HIGH",
                        "instruction": f"📌 WPLEĆ min 1x w tym batchu (extended)",
                        "hard_max_this_batch": 2,
                        "flexibility": "LOW",
                        "adjusted_max": target_max
                    }
                else:
                    return {
                        "suggested": 0,
                        "priority": "SCHEDULED",
                    "instruction": f"⏳ Zaplanowana na późniejszy batch",
                    "hard_max_this_batch": 2,
                    "flexibility": "MEDIUM",
                    "adjusted_max": target_max
                }
        else:
            # v27.2: EXTENDED już użyte min 1x - OK, może być więcej
            remaining_to_max = max(0, target_max - actual)
            if remaining_to_max == 0:
                return {
                    "suggested": 0,
                    "priority": "LOCKED",
                    "instruction": f"🔒 LOCKED - limit osiągnięty ({actual}/{target_max})",
                    "hard_max_this_batch": 0,
                    "flexibility": "NONE",
                    "adjusted_max": target_max
                }
            else:
                return {
                    "suggested": 0,
                    "priority": "OK",
                    "instruction": f"✅ OK ({actual}x) - możesz użyć więcej (max {target_max})",
                    "hard_max_this_batch": min(2, remaining_to_max),
                    "flexibility": "HIGH",
                    "adjusted_max": target_max
                }
    
    # === BASIC / MAIN ===
    
    # v26.1: EXCEEDED
    if actual > target_max:
        return {
            "suggested": 0,
            "priority": "EXCEEDED",
            "instruction": f"❌ EXCEEDED ({actual}/{target_max}) - NIE UŻYWAJ!",
            "hard_max_this_batch": 0,
            "flexibility": "NONE",
            "adjusted_max": target_max,
            "short_keyword": word_count <= SHORT_KEYWORD_MAX_WORDS
        }
    
    # v26.1: LOCKED (osiągnięto max)
    if remaining_to_max == 0:
        return {
            "suggested": 0,
            "priority": "LOCKED",
            "instruction": f"🔒 LOCKED - limit osiągnięty ({target_max}x)",
            "hard_max_this_batch": 0,
            "flexibility": "NONE",
            "adjusted_max": target_max
        }
    
    # v26.1: SOFT CAP WARNING (zbliża się do limitu)
    if soft_cap_info and soft_cap_info["type"] == "SOFT_CAP_WARNING":
        return {
            "suggested": 0,
            "priority": "SOFT_CAP",
            "instruction": soft_cap_warning,
            "hard_max_this_batch": remaining_to_max,
            "flexibility": "LOW",
            "adjusted_max": target_max,
            "remaining": remaining_to_max
        }
    
    if remaining_batches > 0:
        needed_for_target = math.ceil(remaining_to_min / remaining_batches) if remaining_to_min > 0 else 0
        allowed_per_batch = math.ceil(remaining_to_max / remaining_batches)
        suggested = min(needed_for_target, allowed_per_batch) if needed_for_target > 0 else 0
    else:
        suggested = remaining_to_min if remaining_to_min > 0 else 0
    
    # === MAIN KEYWORD ===
    if is_main:
        min_per_batch = max(1, target_min // total_batches)
        suggested = max(suggested, min_per_batch)
        
        if remaining_to_min > 0:
            return {
                "suggested": suggested,
                "priority": "CRITICAL",
                "instruction": f"🔴 FRAZA GŁÓWNA - użyj {suggested}-{suggested+1}x (brakuje {remaining_to_min} do target)",
                "hard_max_this_batch": suggested + 2,
                "flexibility": "LOW",
                "adjusted_max": target_max
            }
        else:
            return {
                "suggested": max(1, suggested),
                "priority": "HIGH",
                "instruction": f"🔴 FRAZA GŁÓWNA - użyj {max(1, suggested)}x (target OK, używaj częściej niż synonimy!)",
                "hard_max_this_batch": suggested + 2,
                "flexibility": "MEDIUM",
                "adjusted_max": target_max
            }
    
    # === BASIC - COVERAGE CHECK ===
    if actual == 0:
        if remaining_batches <= 2:
            return {
                "suggested": max(1, suggested),
                "priority": "CRITICAL",
                "instruction": f"🔴 BRAK COVERAGE! Użyj min 1x (cel: {target_min}-{target_max})",
                "hard_max_this_batch": max(2, suggested + 1),
                "flexibility": "LOW",
                "adjusted_max": target_max
            }
        else:
            return {
                "suggested": max(1, suggested),
                "priority": "HIGH",
                "instruction": f"🟠 Użyj min 1x (cel: {target_min}-{target_max})",
                "hard_max_this_batch": max(2, suggested + 1),
                "flexibility": "MEDIUM",
                "adjusted_max": target_max
            }
    
    if actual < target_min:
        if remaining_batches <= 1:
            return {
                "suggested": remaining_to_min,
                "priority": "CRITICAL",
                "instruction": f"🔴 OSTATNI BATCH! Potrzeba jeszcze {remaining_to_min}x (actual: {actual}/{target_min})",
                "hard_max_this_batch": remaining_to_max,
                "flexibility": "LOW",
                "adjusted_max": target_max
            }
        else:
            return {
                "suggested": suggested,
                "priority": "HIGH",
                "instruction": f"🟠 Dąż do target: użyj ~{suggested}x (actual: {actual}, cel: {target_min}-{target_max})",
                "hard_max_this_batch": suggested + 2,
                "flexibility": "MEDIUM",
                "adjusted_max": target_max
            }
    
    return {
        "suggested": 0,
        "priority": "NORMAL",
        "instruction": f"🟢 OK ({actual}x, cel: {target_min}-{target_max}) - opcjonalnie więcej",
        "hard_max_this_batch": min(2, remaining_to_max),
        "flexibility": "HIGH",
        "adjusted_max": target_max
    }


# ================================================================
# 🧠 H2 SUGGESTIONS (Claude primary, Gemini fallback) v27.0
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    """Generuje sugestie H2 używając Claude (primary) lub Gemini (fallback)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "")
    if not topic:
        return jsonify({"error": "Required: topic or main_keyword"}), 400
    
    serp_h2_patterns = data.get("serp_h2_patterns", [])
    target_keywords = data.get("target_keywords", [])
    target_count = min(data.get("target_count", 6), 6)
    
    # Build prompt (shared between Claude and Gemini)
    competitor_context = ""
    if serp_h2_patterns:
        competitor_context = f"""
WZORCE H2 Z KONKURENCJI (TOP 10 SERP):
{chr(10).join(f"- {h2}" for h2 in serp_h2_patterns[:20])}
"""
    
    keywords_context = ""
    if target_keywords:
        keywords_context = f"""
FRAZY KLUCZOWE DO WPLECENIA W H2:
{', '.join(target_keywords[:10])}
"""
    
    prompt = f"""Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu SEO o temacie: "{topic}"

{competitor_context}
{keywords_context}

KRYTYCZNE ZASADY:
1. MAX 1 H2 z frazą główną "{topic}"! Reszta: synonimy lub naturalne tytuły
2. NIE UŻYWAJ ogólników: "dokument", "wniosek", "sprawa", "proces"
3. Każdy H2 powinien mieć 5-8 słów (max 70 znaków)
4. Minimum 30% H2 w formie pytania (Jak...?, Ile...?, Gdzie...?)
5. NIE używaj: "Wstęp", "Podsumowanie", "Zakończenie", "FAQ"

FORMAT: Zwróć TYLKO listę {target_count} H2, każdy w nowej linii, bez numeracji."""
    
    suggestions = []
    model_used = "fallback"
    
    # === TRY CLAUDE FIRST ===
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            print(f"[H2_SUGGESTIONS] Trying Claude for: {topic}")
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = response.content[0].text.strip()
            raw_suggestions = raw_text.split('\n')
            suggestions = [
                h2.strip().lstrip('•-–—0123456789.). ')
                for h2 in raw_suggestions 
                if h2.strip() and len(h2.strip()) > 5
            ][:target_count]
            
            model_used = "claude-sonnet-4-20250514"
            print(f"[H2_SUGGESTIONS] ✅ Claude generated {len(suggestions)} H2")
            
        except Exception as e:
            print(f"[H2_SUGGESTIONS] ⚠️ Claude failed: {e}, trying Gemini...")
            suggestions = []
    
    # === FALLBACK TO GEMINI ===
    if not suggestions and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            print(f"[H2_SUGGESTIONS] Trying Gemini for: {topic}")
            response = model.generate_content(prompt)
            
            raw_suggestions = response.text.strip().split('\n')
            suggestions = [
                h2.strip().lstrip('•-–—0123456789.). ')
                for h2 in raw_suggestions 
                if h2.strip() and len(h2.strip()) > 5
            ][:target_count]
            
            model_used = GEMINI_MODEL
            print(f"[H2_SUGGESTIONS] ✅ Gemini generated {len(suggestions)} H2")
            
        except Exception as e:
            print(f"[H2_SUGGESTIONS] ⚠️ Gemini failed: {e}")
            suggestions = []
    
    # === STATIC FALLBACK ===
    if not suggestions:
        suggestions = [
            f"Czym jest {topic}?",
            f"Jak działa {topic}?",
            f"Korzyści z {topic}",
            f"Kiedy warto skorzystać z {topic}?",
            f"Ile kosztuje {topic}?",
            f"Najczęstsze pytania o {topic}"
        ][:target_count]
        model_used = "static_fallback"
        print(f"[H2_SUGGESTIONS] ⚠️ Using static fallback")
    
    # Analyze main keyword coverage
    topic_lower = topic.lower()
    h2_with_main = sum(1 for h2 in suggestions if topic_lower in h2.lower())
    
    if h2_with_main > 1:
        print(f"[H2_SUGGESTIONS] ⚠️ Za dużo H2 z frazą główną ({h2_with_main}). Zalecane: max 1")
    
    return jsonify({
        "status": "OK" if model_used != "static_fallback" else "FALLBACK",
        "suggestions": suggestions,
        "topic": topic,
        "model": model_used,
        "count": len(suggestions),
        "main_keyword_in_h2": {
            "count": h2_with_main,
            "max_recommended": 1,
            "overoptimized": h2_with_main > 1,
            "note": "Max 1 H2 z frazą główną. Reszta: synonimy lub naturalne tytuły."
        },
        "action_required": "USER_H2_INPUT_NEEDED"
    }), 200

# ================================================================
# FINALIZE H2
# ================================================================
@project_routes.post("/api/project/finalize_h2")
def finalize_h2():
    """Łączy sugestie H2 z frazami usera."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    suggested_h2 = data.get("suggested_h2", [])
    user_h2_phrases = data.get("user_h2_phrases", [])
    topic = data.get("topic", "")
    
    if not suggested_h2:
        return jsonify({"error": "Required: suggested_h2"}), 400
    
    if not GEMINI_API_KEY or not user_h2_phrases:
        return jsonify({
            "status": "OK",
            "final_h2": suggested_h2,
            "message": "No user phrases or Gemini unavailable"
        }), 200
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
Masz sugestie H2 dla artykulu o "{topic}":
{chr(10).join(f"- {h2}" for h2 in suggested_h2)}

User chce zeby w H2 byly frazy:
{chr(10).join(f"- {phrase}" for phrase in user_h2_phrases)}

Zmodyfikuj H2 zeby KAZDA fraza usera pojawila sie w przynajmniej jednym H2.
Zachowaj naturalnosc, 6-15 slow kazdy H2, min 30% w formie pytania.

Zwroc TYLKO liste H2, kazdy w nowej linii.
"""
        
        response = model.generate_content(prompt)
        
        final_h2 = [
            h2.strip().lstrip('-0123456789.). ')
            for h2 in response.text.strip().split('\n')
            if h2.strip() and len(h2.strip()) > 5
        ]
        
        covered = []
        uncovered = []
        for phrase in user_h2_phrases:
            if any(phrase.lower() in h2.lower() for h2 in final_h2):
                covered.append(phrase)
            else:
                uncovered.append(phrase)
        
        return jsonify({
            "status": "OK",
            "final_h2": final_h2,
            "coverage": {
                "covered_phrases": covered,
                "uncovered_phrases": uncovered,
                "coverage_percent": round(len(covered) / len(user_h2_phrases) * 100, 1) if user_h2_phrases else 100
            }
        }), 200
        
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e), "final_h2": suggested_h2}), 500


# ================================================================
# 🏗️ VALIDATE H2 PLAN v29.2 - Claude tworzy, API waliduje
# ================================================================
@project_routes.post("/api/project/<project_id>/validate_h2_plan")
def validate_h2_plan_endpoint(project_id):
    """
    Waliduje plan H2 stworzony przez Claude.
    
    CLAUDE TWORZY H2 według zasad → API WALIDUJE
    
    INPUT:
    {
        "main_keyword": "pomoce sensoryczne w przedszkolu",
        "h2_phrases": ["integracja sensoryczna", "ścieżka sensoryczna"],
        "h2_plan": [
            {"h2": "Czym są pomoce sensoryczne?", "phrase_used": "pomoce sensoryczne"},
            {"h2": "Integracja sensoryczna - dlaczego?", "phrase_used": "integracja sensoryczna"},
            ...
        ]
    }
    
    OUTPUT:
    {
        "valid": true/false,
        "coverage": {"all_phrases_covered": true, "missing": []},
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
    """
    if not H2_GENERATOR_ENABLED:
        return jsonify({
            "error": "H2 Validator module not available",
            "fallback": True
        }), 500
    
    data = request.get_json() or {}
    
    main_keyword = data.get("main_keyword", "")
    h2_phrases = data.get("h2_phrases", [])
    h2_plan = data.get("h2_plan", [])
    
    if not main_keyword:
        return jsonify({"error": "main_keyword is required"}), 400
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required (list of H2 from Claude)"}), 400
    
    try:
        # Walidacja planu
        validation = validate_h2_plan(h2_plan, main_keyword)
        
        # Sprawdź coverage fraz
        coverage = check_phrase_coverage(h2_plan, h2_phrases, main_keyword)
        
        # Ogólna ocena
        is_valid = validation["valid"] and coverage["all_phrases_covered"]
        
        # Zapisz do projektu jeśli valid
        if is_valid:
            db = firestore.client()
            project_ref = db.collection("projects").document(project_id)
            if project_ref.get().exists:
                # Normalizuj h2_plan do listy dict
                normalized_plan = []
                for i, h2 in enumerate(h2_plan, 1):
                    if isinstance(h2, str):
                        normalized_plan.append({
                            "position": i,
                            "h2": h2,
                            "phrase_used": None
                        })
                    else:
                        h2["position"] = i
                        normalized_plan.append(h2)
                
                project_ref.update({
                    "h2_plan": normalized_plan,
                    "h2_coverage": coverage,
                    "h2_validated_at": firestore.SERVER_TIMESTAMP
                })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "valid": is_valid,
            "validation": validation,
            "coverage": coverage,
            "message": "Plan H2 zaakceptowany!" if is_valid else "Plan H2 wymaga poprawek"
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def check_phrase_coverage(h2_plan: list, h2_phrases: list, main_keyword: str) -> dict:
    """Sprawdza czy wszystkie frazy użytkownika są pokryte w H2."""
    
    # Zbierz wszystkie H2 jako tekst
    h2_texts = []
    for h2 in h2_plan:
        if isinstance(h2, str):
            h2_texts.append(h2.lower())
        elif isinstance(h2, dict):
            h2_texts.append(h2.get("h2", "").lower())
    
    all_h2_text = " ".join(h2_texts)
    
    # Sprawdź główną frazę
    main_keyword_covered = main_keyword.lower() in all_h2_text
    
    # Sprawdź frazy użytkownika
    covered = []
    missing = []
    
    for phrase in h2_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in all_h2_text:
            covered.append(phrase)
        else:
            # Sprawdź też odmiany (częściowe dopasowanie)
            phrase_words = set(phrase_lower.split())
            found_partial = False
            for h2_text in h2_texts:
                h2_words = set(h2_text.split())
                if phrase_words.issubset(h2_words) or len(phrase_words.intersection(h2_words)) >= len(phrase_words) * 0.7:
                    found_partial = True
                    break
            
            if found_partial:
                covered.append(phrase)
            else:
                missing.append(phrase)
    
    coverage_percent = (len(covered) / len(h2_phrases) * 100) if h2_phrases else 100
    
    return {
        "main_keyword_covered": main_keyword_covered,
        "phrases_covered": covered,
        "phrases_missing": missing,
        "coverage_percent": round(coverage_percent, 1),
        "all_phrases_covered": len(missing) == 0 and main_keyword_covered
    }


@project_routes.post("/api/project/<project_id>/save_h2_plan")
def save_h2_plan_endpoint(project_id):
    """
    Zapisuje plan H2 do projektu (bez walidacji - zaufaj Claude).
    
    INPUT:
    {
        "h2_plan": [
            "Czym są pomoce sensoryczne?",
            "Integracja sensoryczna - dlaczego?",
            ...
        ]
    }
    """
    data = request.get_json() or {}
    h2_plan = data.get("h2_plan", [])
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required"}), 400
    
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        
        if not project_ref.get().exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        # Normalizuj do listy dict
        normalized_plan = []
        for i, h2 in enumerate(h2_plan, 1):
            if isinstance(h2, str):
                normalized_plan.append({
                    "position": i,
                    "h2": h2
                })
            else:
                h2["position"] = i
                normalized_plan.append(h2)
        
        project_ref.update({
            "h2_plan": normalized_plan,
            "h2_saved_at": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": normalized_plan,
            "message": "Plan H2 zapisany!"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@project_routes.get("/api/project/<project_id>/h2_plan")
def get_h2_plan(project_id):
    """
    Pobiera zapisany plan H2 dla projektu.
    """
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        project = project_doc.to_dict()
        
        h2_plan = project.get("h2_plan", [])
        
        if not h2_plan:
            return jsonify({
                "status": "NOT_GENERATED",
                "message": "H2 plan not generated yet. Call POST /generate_h2_plan first."
            }), 200
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": h2_plan,
            "h3_suggestions": project.get("h2_h3_suggestions", {}),
            "coverage": project.get("h2_coverage", {}),
            "meta": project.get("h2_plan_meta", {})
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@project_routes.post("/api/project/<project_id>/update_h2_plan")
def update_h2_plan(project_id):
    """
    Aktualizuje plan H2 (po modyfikacjach użytkownika).
    
    INPUT:
    {
        "h2_plan": [
            {"position": 1, "h2": "...", ...},
            ...
        ]
    }
    """
    data = request.get_json() or {}
    h2_plan = data.get("h2_plan", [])
    
    if not h2_plan:
        return jsonify({"error": "h2_plan is required"}), 400
    
    try:
        db = firestore.client()
        project_ref = db.collection("projects").document(project_id)
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            return jsonify({"error": f"Project {project_id} not found"}), 404
        
        project = project_doc.to_dict()
        main_keyword = project.get("main_keyword", "")
        
        # Walidacja nowego planu
        if H2_GENERATOR_ENABLED:
            validation = validate_h2_plan(h2_plan, main_keyword)
        else:
            validation = {"valid": True, "issues": [], "warnings": []}
        
        # Zapisz zaktualizowany plan
        project_ref.update({
            "h2_plan": h2_plan,
            "h2_plan_updated_at": firestore.SERVER_TIMESTAMP,
            "h2_plan_validation": validation
        })
        
        return jsonify({
            "status": "OK",
            "project_id": project_id,
            "h2_plan": h2_plan,
            "validation": validation
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
#  PROJECT CREATE - v25.0
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    """Tworzy nowy projekt SEO w Firestore."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "").strip()
    if not topic:
        return jsonify({"error": "Required field: topic or main_keyword"}), 400
    
    h2_structure = data.get("h2_structure", [])
    raw_keywords = data.get("keywords_list") or data.get("keywords", [])
    target_length = data.get("target_length", 3000)
    source = data.get("source", "unknown")
    
    total_planned_batches = data.get("total_planned_batches")
    if not total_planned_batches:
        total_planned_batches = max(2, min(6, math.ceil(len(h2_structure) / 2))) if h2_structure else 4

    main_keyword_synonyms = detect_main_keyword_synonyms(topic)
    print(f"[PROJECT]  Main keyword synonyms for '{topic}': {main_keyword_synonyms}")

    firestore_keywords = {}
    main_keyword_found = False
    
    for item in raw_keywords:
        term = item.get("term") or item.get("keyword", "")
        term = term.strip() if term else ""
        
        if not term:
            continue
        
        doc = nlp(term)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        min_val = item.get("min") or item.get("target_min", 1)
        max_val = item.get("max") or item.get("target_max", 5)
        
        row_id = item.get("id") or str(uuid.uuid4())
        
        is_main = term.lower() == topic.lower()
        if is_main:
            main_keyword_found = True
            min_val = max(min_val, max(6, target_length // 350))
            max_val = max(max_val, target_length // 150)
        
        is_synonym_of_main = term.lower() in [s.lower() for s in main_keyword_synonyms]
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": min_val,
            "target_max": max_val,
            "display_limit": min_val + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN" if is_main else item.get("type", "BASIC").upper(),
            "is_main_keyword": is_main,
            "is_synonym_of_main": is_synonym_of_main,
            "remaining_max": max_val,
            "optimal_target": max_val
        }
    
    if not main_keyword_found:
        main_min = max(6, target_length // 350)
        main_max = target_length // 150
        
        doc = nlp(topic)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        firestore_keywords["main_keyword_auto"] = {
            "keyword": topic,
            "search_term_exact": topic.lower(),
            "search_lemma": search_lemma,
            "target_min": main_min,
            "target_max": main_max,
            "display_limit": main_min + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN",
            "is_main_keyword": True,
            "is_synonym_of_main": False,
            "remaining_max": main_max,
            "optimal_target": main_max
        }
        print(f"[PROJECT]  Auto-added main keyword '{topic}' with min={main_min}, max={main_max}")

    # ================================================================
    # v26.1: AUTO-REDUKCJA target_max dla NESTED KEYWORDS (INCLUSIVE)
    # W stylu NeuronWriter: "radca prawny" liczy się jako:
    #   - "radca prawny" → 1
    #   - "radca" → 1 (bo słowo "radca" jest w środku)
    #   - "prawny" → 1 (bo słowo "prawny" jest w środku)
    # 
    # Musimy obniżyć target_max krótszej frazy proporcjonalnie do tego
    # ile razy będzie "dziedziczona" z dłuższych fraz.
    # ================================================================
    all_keywords = [(rid, meta.get("keyword", "").lower(), meta.get("keyword", "").lower().split()) 
                    for rid, meta in firestore_keywords.items()]
    
    for rid, meta in firestore_keywords.items():
        keyword_lower = meta.get("keyword", "").lower()
        keyword_words = set(keyword_lower.split())  # słowa z tej frazy
        original_max = meta.get("target_max", 5)
        
        # Znajdź dłuższe frazy które zawierają WSZYSTKIE słowa z tej frazy
        # (lub tę frazę jako substring)
        containing_keywords = []
        for other_rid, other_kw, other_words in all_keywords:
            if other_rid == rid:
                continue
            if len(other_words) <= len(keyword_words):
                continue  # Dłuższa fraza musi mieć więcej słów
            
            # Sprawdź czy wszystkie słowa z krótkiej frazy są w dłuższej
            # LUB czy krótka fraza jest substringiem dłuższej
            words_match = keyword_words.issubset(set(other_words))
            substring_match = keyword_lower in other_kw
            
            if words_match or substring_match:
                other_meta = firestore_keywords[other_rid]
                containing_keywords.append({
                    "keyword": other_kw,
                    "max": other_meta.get("target_max", 1),
                    "match_type": "words" if words_match else "substring"
                })
        
        if containing_keywords:
            # Oblicz ile razy ta fraza będzie liczona przez dłuższe frazy
            inherited_count = sum(kw["max"] for kw in containing_keywords)
            
            # Obniż target_max o inherited_count (ale min 2)
            adjusted_max = max(2, original_max - inherited_count)
            
            if adjusted_max < original_max:
                firestore_keywords[rid]["target_max"] = adjusted_max
                firestore_keywords[rid]["remaining_max"] = adjusted_max
                firestore_keywords[rid]["original_max"] = original_max
                firestore_keywords[rid]["nested_in"] = [kw["keyword"] for kw in containing_keywords]
                firestore_keywords[rid]["inherited_reduction"] = inherited_count
                
                print(f"[PROJECT] ⚠️ NESTED: '{meta.get('keyword')}' max {original_max}→{adjusted_max} "
                      f"(zawarta w: {[kw['keyword'] for kw in containing_keywords]})")

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    
    s1_data = data.get("s1_data", {})
    
    project_data = {
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "batches_plan": [],
        "total_batches": 0,
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        "source": source,
        "version": "v25.0",
        "manual_mode": False if source == "n8n-brajen-workflow" else True,
        "output_format": "clean_text_with_headers",
        "s1_data": s1_data
    }
    
    batch_plan_dict = None
    if BATCH_PLANNER_ENABLED and h2_structure:
        try:
            # v28.1: Zbierz dane dla batch_complexity
            ngrams = [n.get("ngram", "") for n in s1_data.get("ngrams", []) if n.get("weight", 0) > 0.3]
            
            # v28.1: Encje z S1
            entities = []
            for e in s1_data.get("entities", []):
                if isinstance(e, dict):
                    entities.append(e.get("name", str(e)))
                else:
                    entities.append(str(e))
            
            # v28.1: PAA z S1
            paa_questions = [p.get("question", "") for p in s1_data.get("paa", [])]
            
            article_plan = create_article_plan(
                h2_structure=h2_structure,
                keywords_state=firestore_keywords,
                main_keyword=topic,
                target_length=target_length,
                ngrams=ngrams[:20],
                entities=entities[:15],  # v28.1
                paa_questions=paa_questions[:10],  # v28.1
                max_batches=6
            )
            batch_plan_dict = article_plan.to_dict()
            project_data["batch_plan"] = batch_plan_dict
            project_data["total_planned_batches"] = article_plan.total_batches
            total_planned_batches = article_plan.total_batches
            print(f"[PROJECT] Generated batch_plan: {article_plan.total_batches} batches, ~{article_plan.total_target_words} words")
        except Exception as e:
            print(f"[PROJECT] batch_plan failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ================================================================
    # 🆕 v30.0: Legal Module - auto-detekcja i pobieranie orzeczeń
    # ================================================================
    if LEGAL_MODULE_ENABLED:
        try:
            project_data = enhance_project_with_legal(
                project_data=project_data,
                main_keyword=topic,
                h2_list=h2_structure
            )
            if project_data.get("detected_category") == "prawo":
                judgments_count = len(project_data.get("legal_judgments", []))
                print(f"[PROJECT] ⚖️ Legal module active: category=prawo, {judgments_count} judgments loaded")
        except Exception as e:
            print(f"[PROJECT] ⚠️ Legal module error: {e}")
    
    doc_ref.set(project_data)
    
    # v27.2: Policz ile BASIC vs EXTENDED
    basic_count = sum(1 for k in firestore_keywords.values() if k.get("type", "BASIC").upper() in ["BASIC", "MAIN"])
    extended_count = sum(1 for k in firestore_keywords.values() if k.get("type", "").upper() == "EXTENDED")
    
    print(f"[PROJECT] Created project {doc_ref.id}: {topic} ({len(firestore_keywords)} keywords: {basic_count} BASIC, {extended_count} EXTENDED, {total_planned_batches} planned batches)")
    
    # v27.2: WARNING jeśli brak EXTENDED
    warning = None
    if extended_count == 0 and len(firestore_keywords) > 5:
        warning = "⚠️ BRAK FRAZ EXTENDED! Upewnij się że wysyłasz 'type': 'EXTENDED' w keywords_list"

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "keywords_count": len(firestore_keywords),
        "keywords_breakdown": {
            "basic": basic_count,
            "extended": extended_count,
            "warning": warning
        },
        "h2_sections": len(h2_structure),
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        "source": source,
        "batch_plan": batch_plan_dict,
        "has_featured_snippet": bool(s1_data.get("featured_snippet")),
        # 🆕 v30.0: Legal Module fields
        "detected_category": project_data.get("detected_category", "inne"),
        "legal_module_active": project_data.get("legal_context", {}).get("legal_module_active", False),
        "legal_instruction": project_data.get("legal_instruction"),
        "legal_judgments": project_data.get("legal_judgments", []),
        "version": "v30.0"
    }), 201


# ================================================================
#  CONVERT KEYWORDS TO EXTENDED (v27.3)
# ================================================================
@project_routes.post("/api/project/<project_id>/convert_to_extended")
def convert_to_extended(project_id):
    """
    v27.3: Konwertuje wybrane frazy BASIC na EXTENDED.
    
    Body:
    {
        "keywords": ["fraza1", "fraza2", ...]  // lista fraz do konwersji
    }
    
    lub
    
    {
        "all_with_target_1": true  // konwertuj wszystkie z target_max=1
    }
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    
    body = request.get_json() or {}
    keywords_to_convert = body.get("keywords", [])
    all_with_target_1 = body.get("all_with_target_1", False)
    
    converted = []
    skipped = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        
        # Pomiń już EXTENDED i MAIN
        if kw_type == "EXTENDED" or kw_type == "MAIN" or is_main:
            continue
        
        should_convert = False
        
        # Konwertuj jeśli na liście
        if keyword.lower() in [k.lower() for k in keywords_to_convert]:
            should_convert = True
        
        # Konwertuj jeśli target_max=1 i flaga ustawiona
        if all_with_target_1 and target_max == 1:
            should_convert = True
        
        if should_convert:
            keywords_state[rid]["type"] = "EXTENDED"
            keywords_state[rid]["target_min"] = 1
            keywords_state[rid]["target_max"] = max(1, target_max)
            converted.append(keyword)
        else:
            if keyword in keywords_to_convert:
                skipped.append({"keyword": keyword, "reason": "not found or already EXTENDED/MAIN"})
    
    # Zapisz do Firestore
    if converted:
        doc_ref.update({"keywords_state": keywords_state})
    
    # Policz nowe statystyki
    basic_count = sum(1 for k in keywords_state.values() if k.get("type", "BASIC").upper() in ["BASIC"])
    extended_count = sum(1 for k in keywords_state.values() if k.get("type", "").upper() == "EXTENDED")
    main_count = sum(1 for k in keywords_state.values() if k.get("type", "").upper() == "MAIN" or k.get("is_main_keyword"))
    
    return jsonify({
        "status": "OK",
        "converted": converted,
        "converted_count": len(converted),
        "skipped": skipped,
        "keywords_breakdown": {
            "main": main_count,
            "basic": basic_count,
            "extended": extended_count,
            "total": len(keywords_state)
        }
    }), 200


# ================================================================
#  GET PROJECT STATUS
# ================================================================
@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """Zwraca aktualny status projektu z coverage info."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    
    # v25.0: Coverage
    coverage = validate_coverage(keywords_state)
    
    keyword_summary = []
    locked_keywords = []
    near_limit_keywords = []
    
    main_keyword_uses = 0
    synonym_uses = 0
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        
        kw_info = {
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "status": meta.get("status"),
            "remaining_max": remaining,
            "is_main_keyword": meta.get("is_main_keyword", False),
            "is_synonym_of_main": meta.get("is_synonym_of_main", False)
        }
        keyword_summary.append(kw_info)
        
        if meta.get("is_main_keyword"):
            main_keyword_uses = actual
        elif meta.get("is_synonym_of_main"):
            synonym_uses += actual
        
        if remaining == 0:
            locked_keywords.append({
                "keyword": meta.get("keyword"),
                "message": f" LOCKED: '{meta.get('keyword')}' osiągnęło limit {target_max}x"
            })
        elif remaining <= 3:
            near_limit_keywords.append({
                "keyword": meta.get("keyword"),
                "remaining": remaining
            })
    
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 1.0
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "main_keyword": main_keyword,
        "batch_count": len(batches),
        "total_planned_batches": data.get("total_planned_batches", 4),
        "keywords_summary": keyword_summary,
        "locked_keywords": locked_keywords,
        "near_limit_keywords": near_limit_keywords,
        "coverage": coverage,
        "main_vs_synonyms": {
            "main_uses": main_keyword_uses,
            "synonym_uses": synonym_uses,
            "main_ratio": round(main_ratio, 2),
            "valid": main_ratio >= 0.3
        },
        "version": "v25.0"
    }), 200


# ================================================================
#  PRE-BATCH INFO - v25.0
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """v28.1: Używa batch_plan dla zróżnicowanych długości batchy."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    h2_structure = data.get("h2_structure", [])
    total_planned_batches = data.get("total_planned_batches", 4)
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    main_keyword_synonyms = data.get("main_keyword_synonyms", [])
    s1_data = data.get("s1_data", {})
    
    # v28.1: Pobierz batch_plan
    batch_plan = data.get("batch_plan", {})
    
    current_batch_num = len(batches) + 1
    remaining_batches = max(1, total_planned_batches - len(batches))
    
    # Batch type
    if current_batch_num == 1:
        batch_type = "INTRO"
    elif current_batch_num >= total_planned_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # Intro guidance
    intro_guidance = None
    if batch_type == "INTRO":
        featured_snippet = s1_data.get("featured_snippet", {})
        ai_overview = s1_data.get("ai_overview", {})  # v27.1: Google SGE
        serp_analysis = s1_data.get("serp_analysis", {})
        
        # Fallback - ai_overview może być w serp_analysis
        if not ai_overview:
            ai_overview = serp_analysis.get("ai_overview", {})
        
        intro_guidance = {
            "direct_answer_required": True,
            "direct_answer_length": "40-60 slow",
            "first_sentence_must_contain": main_keyword,
            "featured_snippet": None,
            "ai_overview": None  # v27.1
        }
        
        # Featured Snippet
        if featured_snippet and featured_snippet.get("answer"):
            intro_guidance["featured_snippet"] = {
                "google_answer": featured_snippet.get("answer", "")[:500],
                "source_type": featured_snippet.get("type", "unknown"),
                "hint": "Napisz LEPSZA, pelniejsza wersje tej odpowiedzi. NIE kopiuj."
            }
        
        # v27.1: AI Overview (Google SGE)
        if ai_overview and ai_overview.get("text"):
            intro_guidance["ai_overview"] = {
                "google_sge_answer": ai_overview.get("text", "")[:800],
                "sources_count": len(ai_overview.get("sources", [])),
                "hint": "Google SGE pokazuje te informacje. Twój wstep powinien byc LEPSZY i bardziej szczegolowy."
            }
    
    # Coverage
    coverage = validate_coverage(keywords_state)
    
    # Density - v27.3: per keyword
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    current_density = 0
    density_details = {}
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        current_density = prevalidation.get("density", 0)
        density_details = prevalidation.get("density_details", {})
    
    density_status, density_msg = get_density_status(current_density)
    
    # Main vs synonyms
    main_keyword_uses = 0
    synonym_uses = 0
    main_keyword_meta = None
    
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            main_keyword_uses = meta.get("actual_uses", 0)
            main_keyword_meta = meta
        elif meta.get("is_synonym_of_main"):
            synonym_uses += meta.get("actual_uses", 0)
    
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 1.0
    
    ratio_warning = None
    if current_batch_num > 1 and main_ratio < 0.30:
        ratio_warning = f"⚠️ Main keyword ratio {main_ratio:.0%} < 30%. Użyj więcej '{main_keyword}'!"
    
    # v33.3: Wcześniejsze obliczenie remaining_h2 dla dopasowania n-gramów
    h2_structure = data.get("h2_structure", [])
    used_h2_early = []
    for batch in batches:
        batch_text = batch.get("text", "")
        h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
        used_h2_early.extend([(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]])
    remaining_h2_early = [h2 for h2 in h2_structure if h2 not in used_h2_early]
    
    # v33.3: N-gramy dopasowane do H2 (zamiast sekwencyjnych)
    ngrams = s1_data.get("ngrams", [])
    top_ngrams_objs = [n for n in ngrams if n.get("weight", 0) > 0.4][:15]
    top_ngrams = [n.get("ngram", "") for n in top_ngrams_objs]
    
    # Pobierz użyte n-gramy z poprzednich batchów
    batches_so_far = data.get("batches", [])
    used_ngrams = get_used_ngrams_from_batches(batches_so_far, top_ngrams)
    
    # Pobierz H2 dla tego batcha
    current_h2 = remaining_h2_early[0] if remaining_h2_early else main_keyword
    
    # v33.3: Dopasuj n-gramy do H2 zamiast sekwencyjnego przydzielania
    if current_batch_num == 1:
        # Batch 1 (intro) - użyj n-gramów związanych z main keyword
        batch_ngrams = get_ngrams_for_h2(main_keyword, top_ngrams_objs, used_ngrams, max_ngrams=4)
    else:
        # Pozostałe batche - dopasuj do H2
        batch_ngrams = get_ngrams_for_h2(current_h2, top_ngrams_objs, used_ngrams, max_ngrams=4)
    
    # Fallback na sekwencyjne jeśli brak dopasowań
    if not batch_ngrams and top_ngrams:
        ngrams_per_batch = max(3, len(top_ngrams) // total_planned_batches)
        start_idx = (current_batch_num - 1) * ngrams_per_batch
        end_idx = min(start_idx + ngrams_per_batch + 2, len(top_ngrams))
        batch_ngrams = top_ngrams[start_idx:end_idx]
    
    # v28.0: Entity SEO - wyciągnij encje z s1_data
    entity_seo = s1_data.get("entity_seo", {})
    entities = entity_seo.get("entities", [])
    entity_relationships = entity_seo.get("entity_relationships", [])
    topical_coverage = entity_seo.get("topical_coverage", [])
    
    # Top encje do wspomnienia (max 8)
    top_entities = [e for e in entities if e.get("importance", 0) > 0.5][:8]
    # Top relacje (max 5)
    top_relationships = entity_relationships[:5]
    # MUST topics
    must_topics = [t for t in topical_coverage if t.get("priority") == "MUST"][:5]
    
    # v28.0: Dodatkowe dane z S1
    serp_analysis = s1_data.get("serp_analysis", {})
    
    # PAA - pytania użytkowników
    paa_questions = serp_analysis.get("paa_questions", [])
    paa_for_batch = []
    if paa_questions and current_batch_num <= 3:  # PAA tylko w pierwszych 3 batchach
        paa_per_batch = max(1, len(paa_questions) // 3)
        start_paa = (current_batch_num - 1) * paa_per_batch
        paa_for_batch = paa_questions[start_paa:start_paa + paa_per_batch][:2]
    
    # Related searches - powiązane tematy
    related_searches = serp_analysis.get("related_searches", [])[:6]
    
    # Semantic keyphrases (LSI) - jeśli dostępne
    semantic_keyphrases = s1_data.get("semantic_keyphrases", [])
    lsi_keywords = [kp.get("phrase", "") for kp in semantic_keyphrases if kp.get("score", 0) > 0.7][:6]
    
    # Keyword categorization
    basic_must_use = []
    basic_target = []
    basic_done = []
    extended_this_batch = []
    extended_done = []
    extended_scheduled = []
    locked_exceeded = []
    
    main_keyword_info = None
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        is_synonym = meta.get("is_synonym_of_main", False)
        
        suggested_info = calculate_suggested_v25(
            keyword=keyword,
            kw_type=kw_type,
            actual=actual,
            target_min=target_min,
            target_max=target_max,
            remaining_batches=remaining_batches,
            total_batches=total_planned_batches,
            current_batch=current_batch_num,
            is_main=is_main
        )
        
        # v27.2: Jasne instrukcje ile użyć W TYM BATCHU
        suggested_use = suggested_info["suggested"]
        hard_max = suggested_info["hard_max_this_batch"]
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_total": f"{target_min}-{target_max}",  # cel na CAŁY artykuł
            "use_this_batch": f"{suggested_use}-{hard_max}" if suggested_use > 0 else "0",  # użyj W TYM BATCHU
            "suggested": suggested_use,
            "priority": suggested_info["priority"],
            "instruction": suggested_info["instruction"],
            "hard_max_this_batch": hard_max,
            "flexibility": suggested_info["flexibility"],
            "is_main": is_main,
            "is_synonym": is_synonym
        }
        
        if is_main:
            main_keyword_info = kw_info
        elif suggested_info["priority"] in ["EXCEEDED", "LOCKED"]:
            locked_exceeded.append(kw_info)
        elif kw_type == "EXTENDED":
            if suggested_info["priority"] == "DONE":
                extended_done.append(keyword)
            elif suggested_info["priority"] == "SCHEDULED":
                extended_scheduled.append(keyword)
            else:
                extended_this_batch.append(kw_info)
        else:
            if actual == 0:
                basic_must_use.append(kw_info)
            elif actual < target_min:
                basic_target.append(kw_info)
            else:
                basic_done.append(kw_info)
    
    # v27.2: Wymuszenie proporcjonalnego użycia EXTENDED
    # Jeśli jest dużo nieużytych EXTENDED, przenieś część ze SCHEDULED do this_batch
    total_unused_extended = len(extended_this_batch) + len(extended_scheduled)
    if total_unused_extended > 0 and remaining_batches > 0:
        # Ile EXTENDED powinno być użyte w tym batchu?
        extended_per_batch = math.ceil(total_unused_extended / remaining_batches)
        
        # Jeśli mamy za mało w this_batch, przenieś ze SCHEDULED
        while len(extended_this_batch) < extended_per_batch and extended_scheduled:
            kw_to_move = extended_scheduled.pop(0)
            # Znajdź pełne info o tej frazie
            for rid, meta in keywords_state.items():
                if meta.get("keyword") == kw_to_move:
                    extended_this_batch.append({
                        "keyword": kw_to_move,
                        "type": "EXTENDED",
                        "actual": 0,
                        "target": "1-1",
                        "suggested": 1,
                        "priority": "HIGH",
                        "instruction": f"📌 WPLEĆ 1x (przesuniete z kolejnych batchy)",
                        "hard_max_this_batch": 1,
                        "flexibility": "LOW"
                    })
                    break
        
        # Dodaj info o wymaganej liczbie EXTENDED
        if extended_per_batch > 0:
            print(f"[PRE_BATCH] Batch {current_batch_num}: wymaga {extended_per_batch} EXTENDED, ma {len(extended_this_batch)}")
    
    # Used H2
    used_h2 = []
    for batch in batches:
        batch_text = batch.get("text", "")
        h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
        used_h2.extend([(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]])
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    # Last sentences
    last_sentences = ""
    if batches:
        last_batch_text = batches[-1].get("text", "")
        clean_last = re.sub(r'<[^>]+>', '', last_batch_text)
        clean_last = re.sub(r'^h[23]:\s*.+$', '', clean_last, flags=re.MULTILINE)
        sentences = re.split(r'[.!?]+', clean_last)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        if len(sentences) >= 2:
            last_sentences = ". ".join(sentences[-2:]) + "."
        elif sentences:
            last_sentences = sentences[-1] + "."
    
    # GPT Prompt - v27.4: WYMUSZAJĄCY użycie fraz
    prompt_sections = []
    prompt_sections.append("="*60)
    prompt_sections.append("⚠️ KRYTYCZNE INSTRUKCJE - PRZECZYTAJ UWAŻNIE!")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append(f"📝 BATCH #{current_batch_num} z {total_planned_batches} ({batch_type})")
    prompt_sections.append("")
    
    basic_cov = coverage.get("basic", {}).get("coverage_percent", 100)
    ext_cov = coverage.get("extended", {}).get("coverage_percent", 100)
    prompt_sections.append(f"📊 COVERAGE: BASIC {basic_cov:.0f}% | EXTENDED {ext_cov:.0f}%")
    
    # v27.3: Density per keyword
    max_density = density_details.get("max_density", 0)
    prompt_sections.append(f"📈 DENSITY: main={current_density:.1f}% | max={max_density:.1f}% ({density_status})")
    
    # Pokaż warnings jeśli są
    density_warnings = density_details.get("warnings", [])
    if density_warnings:
        for warn in density_warnings[:3]:
            prompt_sections.append(f"   {warn}")
    prompt_sections.append("")
    
    if ratio_warning:
        prompt_sections.append(f"⚠️ {ratio_warning}")
        prompt_sections.append("")
    
    # v27.4: Oblicz ile fraz MUSI być użytych w tym batchu
    total_unused_basic = len(basic_must_use)
    total_unused_extended = len(extended_this_batch) + len(extended_scheduled)
    total_unused = total_unused_basic + total_unused_extended
    
    # Ile fraz na ten batch (proporcjonalnie)
    if remaining_batches > 0:
        basic_this_batch_count = max(3, math.ceil(total_unused_basic / remaining_batches))
        extended_this_batch_count = max(2, math.ceil(total_unused_extended / remaining_batches))
    else:
        basic_this_batch_count = total_unused_basic
        extended_this_batch_count = total_unused_extended
    
    # Wybierz konkretne frazy do tego batcha
    basic_for_this_batch = basic_must_use[:basic_this_batch_count]
    extended_for_this_batch = extended_this_batch[:extended_this_batch_count]
    
    prompt_sections.append("="*60)
    prompt_sections.append("🔴🔴🔴 OBOWIĄZKOWE FRAZY DO UŻYCIA W TYM BATCHU 🔴🔴🔴")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append("❗ KAŻDA fraza z poniższej listy MUSI pojawić się w tekście!")
    prompt_sections.append("❗ Nie możesz pominąć ŻADNEJ frazy - to warunek konieczny!")
    prompt_sections.append("❗ Wpleć frazy naturalnie w zdania, nie zmieniaj ich formy!")
    prompt_sections.append("")
    
    if main_keyword_info:
        prompt_sections.append(f"🎯 FRAZA GŁÓWNA: \"{main_keyword}\"")
        prompt_sections.append(f"   → Użyj DOKŁADNIE {main_keyword_info['use_this_batch']}x w tym batchu")
        prompt_sections.append("")
    
    if basic_for_this_batch:
        prompt_sections.append(f"📋 BASIC - MUSISZ UŻYĆ WSZYSTKIE ({len(basic_for_this_batch)} fraz):")
        for i, kw in enumerate(basic_for_this_batch, 1):
            prompt_sections.append(f"   {i}. \"{kw['keyword']}\" ← OBOWIĄZKOWO 1x")
        prompt_sections.append("")
    
    if extended_for_this_batch:
        prompt_sections.append(f"📋 EXTENDED - MUSISZ UŻYĆ WSZYSTKIE ({len(extended_for_this_batch)} fraz):")
        for i, kw in enumerate(extended_for_this_batch, 1):
            prompt_sections.append(f"   {i}. \"{kw['keyword']}\" ← OBOWIĄZKOWO 1x")
        prompt_sections.append("")
    
    # Pokaż pozostałe nieużyte (info)
    basic_remaining = basic_must_use[basic_this_batch_count:]
    extended_remaining = extended_this_batch[extended_this_batch_count:] + extended_scheduled
    
    if basic_remaining or extended_remaining:
        prompt_sections.append(f"📌 POZOSTAŁE NIEUŻYTE (do kolejnych batchy: {len(basic_remaining)} BASIC + {len(extended_remaining)} EXTENDED)")
        prompt_sections.append("")
    
    prompt_sections.append("="*60)
    prompt_sections.append("✅ CHECKLIST PRZED WYSŁANIEM:")
    prompt_sections.append(f"   [ ] Fraza główna użyta {main_keyword_info['use_this_batch'] if main_keyword_info else 1}x")
    prompt_sections.append(f"   [ ] Wszystkie {len(basic_for_this_batch)} fraz BASIC użyte")
    prompt_sections.append(f"   [ ] Wszystkie {len(extended_for_this_batch)} fraz EXTENDED użyte")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    
    if basic_target:
        prompt_sections.append("🟠 OPCJONALNE - DĄŻ DO TARGET (jeśli zmieścisz):")
        for kw in basic_target[:3]:
            prompt_sections.append(f"   • \"{kw['keyword']}\" → {kw['use_this_batch']}x")
        prompt_sections.append("")
    
    if batch_ngrams:
        prompt_sections.append("💡 N-GRAMY (wpleć naturalnie):")
        for ngram in batch_ngrams[:4]:
            prompt_sections.append(f"   • \"{ngram}\"")
        prompt_sections.append("")
    
    # v28.0: Entity SEO - encje do wspomnienia
    if top_entities:
        prompt_sections.append("🏢 ENCJE DO WSPOMNIENIA (Entity SEO):")
        prompt_sections.append("   Wspomnij te nazwy własne naturalnie w tekście:")
        for ent in top_entities[:5]:
            ent_type = ent.get("type", "")
            type_label = {"ORGANIZATION": "firma/inst.", "PERSON": "osoba", "LOCATION": "miejsce"}.get(ent_type, "")
            if type_label:
                prompt_sections.append(f"   • {ent.get('text', '')} ({type_label})")
            else:
                prompt_sections.append(f"   • {ent.get('text', '')}")
        prompt_sections.append("")
    
    if top_relationships:
        prompt_sections.append("🔗 RELACJE DO OPISANIA:")
        for rel in top_relationships[:3]:
            prompt_sections.append(f"   • {rel.get('subject', '')} → {rel.get('verb', '')} → {rel.get('object', '')}")
        prompt_sections.append("")
    
    # v28.0: PAA - pytania użytkowników (odpowiedz na nie w tekście)
    if paa_for_batch:
        prompt_sections.append("❓ PYTANIA UŻYTKOWNIKÓW (odpowiedz w tekście):")
        for paa in paa_for_batch:
            q = paa.get("question", "")
            if q:
                prompt_sections.append(f"   • {q}")
        prompt_sections.append("")
    
    # v28.0: LSI keywords (semantic keyphrases)
    if lsi_keywords:
        prompt_sections.append("🔤 LSI KEYWORDS (wpleć naturalnie):")
        prompt_sections.append(f"   {', '.join(lsi_keywords)}")
        prompt_sections.append("")
    
    # v28.0: Related searches (powiązane tematy - inspiracja)
    if related_searches and current_batch_num <= 2:  # tylko w pierwszych batchach
        prompt_sections.append("🔍 POWIĄZANE TEMATY (opcjonalnie nawiąż):")
        prompt_sections.append(f"   {', '.join(related_searches[:4])}")
        prompt_sections.append("")
    
    # v27.2: Sekcja ZABRONIONYCH fraz - rozdzielona na typy
    # Zbierz wszystkie zabronione: locked + exceeded + extended_done
    forbidden_basic = []
    forbidden_extended = []
    
    for kw in locked_exceeded:
        if kw.get('type', 'BASIC').upper() == 'EXTENDED':
            forbidden_extended.append(kw['keyword'])
        else:
            forbidden_basic.append(kw['keyword'])
    
    # EXTENDED DONE też są zabronione (już użyte 1x)
    forbidden_extended.extend(extended_done)
    
    # Wyświetl zabronione
    if forbidden_basic or forbidden_extended:
        prompt_sections.append("=" * 50)
        prompt_sections.append("🚫 ZABRONIONE FRAZY (NIE UŻYWAJ!):")
        
        if forbidden_basic:
            prompt_sections.append(f"   BASIC (limit osiągnięty): {', '.join(forbidden_basic[:10])}")
            if len(forbidden_basic) > 10:
                prompt_sections.append(f"   ... i {len(forbidden_basic) - 10} więcej BASIC")
        
        if forbidden_extended:
            prompt_sections.append(f"   EXTENDED (już użyte 1x): {', '.join(forbidden_extended[:10])}")
            if len(forbidden_extended) > 10:
                prompt_sections.append(f"   ... i {len(forbidden_extended) - 10} więcej EXTENDED")
        
        prompt_sections.append("=" * 50)
        prompt_sections.append("")
    
    if remaining_h2:
        prompt_sections.append("📋 H2 DO NAPISANIA:")
        for h2 in remaining_h2[:3]:
            prompt_sections.append(f"   • {h2}")
        prompt_sections.append("")
    
    if last_sentences:
        prompt_sections.append(f"🔗 KONTYNUUJ OD: \"{last_sentences[:80]}...\"")
        prompt_sections.append("")
    
    # ================================================================
    # v27.2: DYNAMIC BATCH LENGTH - oblicz minimalną długość na podstawie fraz
    # ================================================================
    # Formuła: 
    # 1. Policz WSZYSTKIE pozostałe użycia fraz (do końca artykułu)
    # 2. Podziel przez remaining_batches = ile użyć na TEN batch
    # 3. min_words = (uses_this_batch * avg_phrase_length) / target_density
    
    # Policz WSZYSTKIE pozostałe użycia (nie tylko ten batch)
    total_remaining_basic = 0
    total_remaining_extended = 0
    avg_phrase_words = 0
    phrase_count = 0
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        keyword = meta.get("keyword", "")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        if not keyword:
            continue
        
        phrase_count += 1
        avg_phrase_words += len(keyword.split())
        
        if kw_type == "EXTENDED":
            # EXTENDED: potrzebuje min 1x
            if actual < 1:
                total_remaining_extended += 1
        else:
            # BASIC: potrzebuje min target_min
            remaining = max(0, target_min - actual)
            total_remaining_basic += remaining
    
    # Średnia długość frazy
    if phrase_count > 0:
        avg_phrase_words = avg_phrase_words / phrase_count
    else:
        avg_phrase_words = 2.0  # domyślnie 2 słowa
    
    # PODZIEL przez remaining_batches = ile na TEN batch
    total_remaining_all = total_remaining_basic + total_remaining_extended
    
    if remaining_batches > 0:
        uses_this_batch = math.ceil(total_remaining_all / remaining_batches)
    else:
        uses_this_batch = total_remaining_all
    
    # v28.1: UŻYJ BATCH_PLAN jeśli dostępny
    suggested_min_words = None
    suggested_max_words = None
    suggested_paragraphs_min = 3  # v28.1: default
    suggested_paragraphs_max = 4  # v28.1: default
    length_profile = "medium"  # v28.1: default
    complexity_score = 50  # v28.1: default
    complexity_reasoning = []  # v28.1: dlaczego taka długość
    snippet_required = True  # v28.1
    batch_plan_used = False
    
    if batch_plan and "batches" in batch_plan:
        batch_plans_list = batch_plan.get("batches", [])
        # Znajdź plan dla current_batch_num
        for bp in batch_plans_list:
            if bp.get("batch_number") == current_batch_num:
                suggested_min_words = bp.get("target_words_min")
                suggested_max_words = bp.get("target_words_max")
                # v28.1: Pobierz wszystkie nowe pola
                suggested_paragraphs_min = bp.get("target_paragraphs_min", 3)
                suggested_paragraphs_max = bp.get("target_paragraphs_max", 4)
                length_profile = bp.get("length_profile", "medium")
                complexity_score = bp.get("complexity_score", 50)
                complexity_reasoning = bp.get("complexity_reasoning", [])
                snippet_required = bp.get("snippet_required", True)
                batch_plan_used = True
                print(f"[PRE_BATCH] batch_plan: batch {current_batch_num}, score={complexity_score}, profile={length_profile}, {suggested_min_words}-{suggested_max_words} words")
                break
    
    # FALLBACK: jeśli brak batch_plan, oblicz dynamicznie
    if not suggested_min_words:
        TARGET_DENSITY_FOR_CALC = 1.5
        
        if uses_this_batch > 0:
            min_words_for_density = int((uses_this_batch * avg_phrase_words) / (TARGET_DENSITY_FOR_CALC / 100))
        else:
            min_words_for_density = 200
        
        # Podstawowa długość zależy od typu batcha - ZMNIEJSZONE WARTOŚCI!
        if batch_type == "INTRO":
            base_min_words = 120
            base_max_words = 180
        elif batch_type == "FINAL":
            base_min_words = 250
            base_max_words = 400
        else:
            base_min_words = 280
            base_max_words = 450
        
        # Weź większą z: bazowej i obliczonej dla density
        suggested_min_words = max(base_min_words, min_words_for_density)
        suggested_max_words = max(base_max_words, suggested_min_words + 100)
        
        # Limit maksymalny
        suggested_min_words = min(suggested_min_words, 600)
        suggested_max_words = min(suggested_max_words, 800)
    
    batch_length_info = {
        "suggested_min": suggested_min_words,
        "suggested_max": suggested_max_words,
        "paragraphs_min": suggested_paragraphs_min,
        "paragraphs_max": suggested_paragraphs_max,
        "length_profile": length_profile,
        "complexity_score": complexity_score,  # v28.1
        "complexity_reasoning": complexity_reasoning,  # v28.1: DLACZEGO taka długość
        "snippet_required": snippet_required,  # v28.1
        "total_remaining": total_remaining_all,
        "uses_this_batch": uses_this_batch,
        "remaining_batches": remaining_batches,
        "from_batch_plan": batch_plan_used,
        "reason": f"Pozostało {total_remaining_all} użyć fraz / {remaining_batches} batchy = ~{uses_this_batch} na ten batch",
        "density_note": f"Przy {suggested_min_words} słowach utrzymasz density w normie"
    }
    
    prompt_sections.append("="*50)
    plan_note = " (z batch_plan)" if batch_plan_used else ""
    prompt_sections.append(f"📏 DŁUGOŚĆ BATCHA{plan_note}: {suggested_min_words}-{suggested_max_words} słów")
    prompt_sections.append(f"📄 AKAPITY: {suggested_paragraphs_min}-{suggested_paragraphs_max}")
    prompt_sections.append(f"🎯 SCORE ZŁOŻONOŚCI: {complexity_score}/100 → profil: {length_profile.upper()}")
    
    # v28.1: Pokaż DLACZEGO taka długość (max 2 powody)
    if complexity_reasoning:
        prompt_sections.append(f"💡 DLACZEGO TAKA DŁUGOŚĆ:")
        for reason in complexity_reasoning[:2]:
            prompt_sections.append(f"   • {reason}")
    
    if snippet_required:
        prompt_sections.append(f"⚡ SNIPPET WYMAGANY: Pierwszych 40-60 słów = bezpośrednia odpowiedź!")
    
    prompt_sections.append(f"   Pozostało {total_remaining_all} użyć fraz / {remaining_batches} batchy = ~{uses_this_batch} na ten batch")
    if uses_this_batch > 15:
        prompt_sections.append(f"   ⚠️ DUŻO FRAZ! Pisz dłuższe sekcje żeby zmieścić wszystkie.")
    prompt_sections.append("")
    
    # v27.4: FINALNE PODSUMOWANIE z konkretną listą
    prompt_sections.append("="*60)
    prompt_sections.append("🎯 FINALNE PODSUMOWANIE - CO MUSISZ ZROBIĆ:")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    prompt_sections.append(f"W tym batchu MUSISZ użyć DOKŁADNIE tych fraz:")
    prompt_sections.append("")
    
    all_required = []
    if main_keyword_info:
        all_required.append(f"• \"{main_keyword}\" × {main_keyword_info['use_this_batch']}")
    for kw in basic_for_this_batch:
        all_required.append(f"• \"{kw['keyword']}\" × 1")
    for kw in extended_for_this_batch:
        all_required.append(f"• \"{kw['keyword']}\" × 1")
    
    for req in all_required:
        prompt_sections.append(f"   {req}")
    
    prompt_sections.append("")
    prompt_sections.append(f"RAZEM: {len(all_required)} fraz do wplecenia")
    prompt_sections.append("")
    prompt_sections.append("❌ Jeśli pominiesz KTÓRĄKOLWIEK frazę - batch będzie ODRZUCONY!")
    prompt_sections.append("="*60)
    prompt_sections.append("")
    
    prompt_sections.append("="*50)
    prompt_sections.append("✍️ STYL:")
    prompt_sections.append(f"   • Sekcje H2: różna długość (min {suggested_min_words // 2} słów na sekcję)")
    prompt_sections.append("   • Akapity: 40-150 słów")
    prompt_sections.append("   • H3: max 2-3 na artykuł")
    prompt_sections.append("   • Listy wypunktowane: dozwolone w miarę potrzeb")
    prompt_sections.append("   • Format: h2: / h3:")
    prompt_sections.append("="*50)
    
    gpt_prompt = "\n".join(prompt_sections)
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "batch_number": current_batch_num,
        "batch_type": batch_type,
        "intro_guidance": intro_guidance,
        "total_planned_batches": total_planned_batches,
        "remaining_batches": remaining_batches,
        
        # v27.2: Dynamic batch length
        "batch_length": batch_length_info,
        
        "coverage": {
            "basic": coverage.get("basic", {}),
            "extended": coverage.get("extended", {}),
            "overall": coverage.get("overall_coverage", 100)
        },
        
        "density": {
            "current": current_density,
            "max_density": density_details.get("max_density", 0),
            "avg_density": density_details.get("avg_density", 0),
            "status": density_status,
            "message": density_msg,
            "optimal_range": f"{DENSITY_OPTIMAL_MIN}-{DENSITY_OPTIMAL_MAX}%",
            "warnings": density_details.get("warnings", []),
            "per_keyword_top5": dict(list(density_details.get("per_keyword", {}).items())[:5])
        },
        
        "main_keyword": {
            "keyword": main_keyword,
            "info": main_keyword_info,
            "ratio": round(main_ratio, 2),
            "ratio_warning": ratio_warning
        },
        
        "keywords": {
            "basic_must_use": basic_must_use,
            "basic_target": basic_target,
            "basic_done": [kw["keyword"] for kw in basic_done],
            "extended_this_batch": extended_this_batch,
            "extended_done": extended_done,
            "extended_scheduled": extended_scheduled,
            "locked_exceeded": locked_exceeded
        },
        
        "ngrams_for_batch": batch_ngrams,
        
        # v28.0: Entity SEO
        "entity_seo": {
            "top_entities": top_entities,
            "relationships": top_relationships,
            "must_topics": must_topics,
            "total_entities": len(entities),
            "enabled": bool(entity_seo)
        },
        
        # v29.3: Entity guidance for batch
        "entities_for_batch": {
            "to_introduce": get_entities_to_introduce(
                top_entities, 
                current_batch_num, 
                total_planned_batches,
                [b.get("text", "") for b in data.get("batches", [])]
            ),
            "already_defined": get_already_defined_entities(
                [b.get("text", "") for b in data.get("batches", [])]
            ),
            "suggested_relationships": top_relationships[:2] if current_batch_num > 1 else []
        },
        
        # v29.3: N-gram diversity guidance
        "ngram_guidance": {
            "overused_phrases": get_overused_phrases(
                [b.get("text", "") for b in data.get("batches", [])],
                main_keyword
            ),
            "suggested_synonyms": get_synonyms_for_overused(
                [b.get("text", "") for b in data.get("batches", [])],
                main_keyword
            ),
            "lsi_to_include": lsi_keywords[:3] if lsi_keywords else batch_ngrams[:3]
        },
        
        # v29.3: Section length variety guidance
        "section_length_guidance": get_section_length_guidance(
            current_batch_num,
            total_planned_batches,
            batch_type
        ),
        
        # v28.0: Dodatkowe dane SERP
        "serp_enrichment": {
            "paa_for_batch": paa_for_batch,
            "lsi_keywords": lsi_keywords,
            "related_searches": related_searches
        },
        
        "h2_remaining": remaining_h2,
        "h2_used": used_h2,
        
        # v29.2: H2 Plan z generatora
        "h2_plan": data.get("h2_plan", []),
        "h2_plan_meta": data.get("h2_plan_meta", {}),
        
        "gpt_prompt": gpt_prompt,
        
        "version": "v29.2"
    }), 200


# ================================================================
#  ADD BATCH
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    return jsonify(result), 200


# ================================================================
#  APPROVE BATCH - v28.2 z CLAUDE REVIEWER
# ================================================================
@project_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch_with_review(project_id):
    """
    v28.3: Approve batch z automatycznym review przez Claude.
    
    NOWOŚĆ v28.3:
    - Auto-approve po 2 próbach (attempt >= 2)
    - Lemmatyzacja fraz (ścieżka sensoryczna = ścieżką sensoryczną)
    - Wykrywanie tautologii przez Claude
    
    Flow:
    1. Quick checks (Python) - frazy, długość
    2. Claude review - pełna analiza semantyczna
    3. Jeśli CORRECTED → zwróć poprawiony tekst
    4. Jeśli APPROVED → zapisz do Firestore
    5. Jeśli REJECTED → zwróć do przepisania
    
    Request:
    {
        "text": "h2: Tytuł...",
        "skip_review": false,  // opcjonalne - pomiń Claude
        "force_save": false,   // opcjonalne - zapisz mimo warnings
        "attempt": 1           // NOWE! numer próby (1, 2, 3...)
    }
    
    Po attempt >= 2: automatyczne force_save=True (auto-approve)
    """
    from dataclasses import asdict
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    
    batch_text = data.get("corrected_text") or data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "No text provided"}), 400
    
    skip_review = data.get("skip_review", False)
    force_save = data.get("force_save", False)
    attempt = data.get("attempt", 1)  # v28.3: numer próby
    
    # v30.1 OPTIMIZED: AUTO-APPROVE po 2 próbach (było 3)
    if attempt >= 2 and not force_save:
        print(f"[APPROVE_BATCH] ⚡ Auto-approve: attempt={attempt} >= 3, force_save=True")
        force_save = True
    
    db = firestore.client()
    project_ref = db.collection("seo_projects").document(project_id)
    project_doc = project_ref.get()
    
    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = project_doc.to_dict()
    
    # ============================================
    # ZBUDUJ CONTEXT DLA REVIEWERA
    # ============================================
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    batch_plan = project_data.get("batch_plan", {})
    current_batch = project_data.get("current_batch_num", 1)
    
    # Znajdź wymagane frazy (nieużyte)
    keywords_required = []
    keywords_forbidden = []
    
    # Main keyword
    main_kw_count = 2
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            actual = meta.get("actual_uses", 0)
            target = meta.get("target_max", 10)
            remaining = max(0, target - actual)
            main_kw_count = min(3, max(1, remaining // max(1, project_data.get("total_planned_batches", 4) - current_batch + 1)))
            break
    
    keywords_required.append({"keyword": main_keyword, "count": main_kw_count})
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword or keyword.lower() == main_keyword.lower():
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 5)
        
        if kw_type == "EXTENDED":
            if actual >= 1:
                keywords_forbidden.append(keyword)
            else:
                keywords_required.append({"keyword": keyword, "count": 1})
        else:
            if actual >= target_max:
                keywords_forbidden.append(keyword)
            elif actual < target_min:
                keywords_required.append({"keyword": keyword, "count": 1})
    
    # Limit fraz na batch
    keywords_required = keywords_required[:12]
    
    # Długość z batch_plan
    target_words_min = 200
    target_words_max = 500
    target_para_min = 2
    target_para_max = 5
    snippet_required = True
    complexity_score = 50
    
    if batch_plan and "batches" in batch_plan:
        for bp in batch_plan.get("batches", []):
            if bp.get("batch_number") == current_batch:
                target_words_min = bp.get("target_words_min", 200)
                target_words_max = bp.get("target_words_max", 500)
                target_para_min = bp.get("target_paragraphs_min", 2)
                target_para_max = bp.get("target_paragraphs_max", 5)
                snippet_required = bp.get("snippet_required", True)
                complexity_score = bp.get("complexity_score", 50)
                break
    
    # Last sentences
    article_content = project_data.get("article_content", "")
    last_sentences = article_content[-200:] if len(article_content) > 200 else article_content
    
    review_context = {
        "topic": project_data.get("topic", ""),
        "h2_current": project_data.get("h2_remaining", [])[:2],
        "keywords_required": keywords_required,
        "keywords_forbidden": keywords_forbidden,
        "last_sentences": last_sentences,
        "target_words_min": target_words_min,
        "target_words_max": target_words_max,
        "target_paragraphs_min": target_para_min,
        "target_paragraphs_max": target_para_max,
        "main_keyword": main_keyword,
        "main_keyword_count": main_kw_count,
        "batch_number": current_batch,
        "snippet_required": snippet_required,
        "complexity_score": complexity_score
    }
    
    # ============================================
    # CLAUDE REVIEW
    # ============================================
    try:
        from claude_reviewer import review_batch, ReviewResult
        
        result = review_batch(batch_text, review_context, skip_claude=skip_review)
        
        # QUICK_CHECK_FAILED - zwróć do poprawy
        if result.status == "QUICK_CHECK_FAILED":
            issues_list = [asdict(i) for i in result.issues]
            return jsonify({
                "status": "QUICK_CHECK_FAILED",
                "needs_correction": True,
                "issues": issues_list,
                "correction_prompt": build_correction_prompt(issues_list, batch_text),
                "message": result.summary,
                "word_count": result.word_count,
                "attempt": attempt,
                "next_attempt": attempt + 1,  # v28.3: GPT powinien przekazać to w następnym requeście
                "auto_approve_at": 2  # v30.1: info że po 2 próbie będzie auto-approve
            }), 200
        
        # REJECTED - wymaga przepisania
        if result.status == "REJECTED":
            issues_list = [asdict(i) for i in result.issues]
            return jsonify({
                "status": "REJECTED",
                "needs_correction": True,
                "issues": issues_list,
                "correction_prompt": f"Tekst wymaga przepisania. {result.summary}",
                "message": result.summary,
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "auto_approve_at": 2
            }), 200
        
        # CORRECTED - Claude poprawił
        if result.status == "CORRECTED" and result.corrected_text:
            # Użyj poprawionego tekstu
            batch_text = result.corrected_text
            issues_list = [asdict(i) for i in result.issues]
            
            if not force_save:
                # Zwróć do akceptacji przez GPT
                return jsonify({
                    "status": "CORRECTED",
                    "needs_correction": False,
                    "corrected_text": batch_text,
                    "original_text": result.original_text,
                    "issues": issues_list,
                    "message": f"Claude poprawił tekst: {result.summary}",
                    "word_count": result.word_count,
                    "instruction": "Użyj corrected_text i wyślij ponownie z force_save=true",
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "auto_approve_at": 2
                }), 200
        
        # APPROVED lub force_save - zapisz
        print(f"[APPROVE_BATCH] ✅ Review passed: {result.status}, saving batch")
        
    except ImportError:
        print(f"[APPROVE_BATCH] ⚠️ claude_reviewer not available, saving without review")
    except Exception as e:
        print(f"[APPROVE_BATCH] ⚠️ Review error: {e}, saving anyway")
    
    # ============================================
    # ZAPISZ DO FIRESTORE
    # ============================================
    meta_trace = data.get("meta_trace", {})
    save_result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    return jsonify({
        "status": "APPROVED",
        "needs_correction": False,
        "saved": True,
        "batch_number": save_result.get("batch_number"),
        "word_count": len(batch_text.split()),
        "message": "Batch zatwierdzony i zapisany",
        **save_result
    }), 200


def build_correction_prompt(issues: list, original_text: str) -> str:
    """Buduje prompt do poprawienia tekstu."""
    lines = ["Popraw poniższy tekst:"]
    
    for issue in issues:
        severity = issue.get("severity", "warning")
        desc = issue.get("description", "")
        if severity == "critical":
            lines.append(f"❌ KRYTYCZNE: {desc}")
        else:
            lines.append(f"⚠️ {desc}")
    
    lines.append("\n--- TEKST DO POPRAWY ---")
    lines.append(original_text[:1000])
    if len(original_text) > 1000:
        lines.append("...")
    lines.append("\n--- WYŚLIJ POPRAWIONY TEKST ---")
    
    return "\n".join(lines)


# ================================================================
#  PREVIEW BATCH
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """Preview batch z walidacją."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    s1_data = project_data.get("s1_data", {})

    report = unified_prevalidation(batch_text, keywords_state)
    
    warnings = report.get("warnings", [])
    errors = []
    
    # v25.0: Density check
    density = report.get("density", 0)
    density_status, density_msg = get_density_status(density)
    if density_status in ["HIGH", "STUFFING"]:
        warnings.append({
            "type": "DENSITY_HIGH",
            "density": density,
            "status": density_status,
            "message": density_msg
        })
    
    # v28.1: Usunięto limit 1 listy - teraz max 3 (informacyjnie)
    list_count = count_bullet_lists(batch_text)
    # Tylko informacja, nie blokuje
    # if list_count > 3:
    #     warnings.append({
    #         "type": "TOO_MANY_LISTS",
    #         "count": list_count,
    #         "max": 3,
    #         "message": f"Dużo list ({list_count}). Rozważ ograniczenie."
    #     })
    
    h3_validation = validate_h3_length(batch_text, min_words=80)
    if h3_validation["issues"]:
        for issue in h3_validation["issues"]:
            warnings.append({
                "type": "H3_TOO_SHORT",
                "h3": issue["h3"],
                "word_count": issue["word_count"],
                "min": 80,
                "message": f"H3 '{issue['h3']}' za krótkie ({issue['word_count']} słów, min 80)"
            })
    
    main_synonym_check = check_main_vs_synonyms_in_text(batch_text, main_keyword, keywords_state)
    if not main_synonym_check["valid"]:
        warnings.append({
            "type": "SYNONYM_OVERUSE",
            "main_count": main_synonym_check["main_count"],
            "synonym_total": main_synonym_check["synonym_total"],
            "ratio": main_synonym_check["main_ratio"],
            "message": main_synonym_check["warning"]
        })
    
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.5][:10]
    ngram_check = check_ngram_coverage_in_text(batch_text, top_ngrams)
    if ngram_check["coverage"] < 0.5:
        warnings.append({
            "type": "LOW_NGRAM_COVERAGE",
            "coverage": ngram_check["coverage"],
            "missing": ngram_check["missing"][:3],
            "message": f"Niskie pokrycie n-gramów ({ngram_check['coverage']:.0%})"
        })
    
    # v27.4: Polish Language Quality Check
    polish_quality = {"status": "DISABLED", "issues": []}
    if POLISH_QUALITY_ENABLED:
        try:
            polish_quality = quick_polish_check(batch_text)
            
            # Dodaj szczegóły kolokacji i banned phrases
            collocations, _ = check_collocations(batch_text)
            banned, _ = check_banned_phrases(batch_text)
            
            polish_quality["collocations"] = collocations[:5]  # Max 5
            polish_quality["banned_phrases"] = banned[:5]  # Max 5
            
            # Dodaj warnings jeśli są problemy
            for coll in collocations[:3]:
                warnings.append({
                    "type": "COLLOCATION_ERROR",
                    "found": coll.get("found", ""),
                    "suggested": coll.get("suggested", ""),
                    "message": f"Błędna kolokacja: '{coll.get('found')}' → '{coll.get('suggested')}'"
                })
            
            for bp in banned[:3]:
                warnings.append({
                    "type": "BANNED_PHRASE",
                    "phrase": bp.get("phrase", ""),
                    "category": bp.get("category", ""),
                    "message": f"Fraza AI: '{bp.get('phrase')}' - usuń lub przeformułuj"
                })
                
        except Exception as e:
            print(f"[PREVIEW_BATCH] ⚠️ Polish quality check error: {e}")
            polish_quality = {"status": "ERROR", "error": str(e)}
    
    status = "OK"
    if errors:
        status = "ERROR"
    elif len(warnings) > 2:
        status = "WARN"
    
    # v28.1: GRAMMAR VALIDATION - sprawdź przed zapisem!
    grammar_validation = {"is_valid": True, "error_count": 0, "correction_needed": False}
    try:
        from grammar_middleware import validate_batch_full
        grammar_validation = validate_batch_full(batch_text)
        
        if not grammar_validation["is_valid"]:
            status = "NEEDS_CORRECTION"
            print(f"[PREVIEW_BATCH] ⚠️ Grammar issues: {grammar_validation['grammar']['error_count']} errors, banned: {grammar_validation['banned_phrases']['found']}")
            
            # Dodaj do warnings
            if grammar_validation["grammar"]["error_count"] > 0:
                warnings.append({
                    "type": "GRAMMAR_ERRORS",
                    "count": grammar_validation["grammar"]["error_count"],
                    "message": f"Wykryto {grammar_validation['grammar']['error_count']} błędów gramatycznych - popraw przed zapisem!"
                })
            
            for phrase in grammar_validation["banned_phrases"]["found"]:
                warnings.append({
                    "type": "BANNED_PHRASE_DETECTED",
                    "phrase": phrase,
                    "message": f"Usuń frazę: '{phrase}'"
                })
    except ImportError:
        print(f"[PREVIEW_BATCH] ⚠️ grammar_middleware not available")
    except Exception as e:
        print(f"[PREVIEW_BATCH] ⚠️ Grammar check error: {e}")
    
    # v27.0: Zapisz tekst do last_preview (fallback dla approve_batch)
    try:
        db.collection("seo_projects").document(project_id).update({
            "last_preview": {
                "text": batch_text,
                "status": status,
                "grammar_valid": grammar_validation.get("is_valid", True),
                "timestamp": firestore.SERVER_TIMESTAMP
            }
        })
        print(f"[PREVIEW_BATCH] ✅ Zapisano last_preview ({len(batch_text)} znaków)")
    except Exception as e:
        print(f"[PREVIEW_BATCH] ⚠️ Nie udało się zapisać last_preview: {e}")
    
    # v28.1: Jeśli błędy gramatyczne - zwróć prompt do poprawy
    response_data = {
        "status": status,
        "semantic_score": report.get("semantic_score", 0),
        "density": density,
        "density_status": density_status,
        "warnings": warnings,
        "errors": errors,
        "validations": {
            "lists": {"count": list_count, "valid": True},  # v28.1: brak limitu list
            "h3_length": h3_validation,
            "main_vs_synonyms": main_synonym_check,
            "ngram_coverage": ngram_check,
            "density": {"value": density, "status": density_status, "message": density_msg},
            "polish_quality": polish_quality,
            "grammar": grammar_validation  # v28.1
        },
        "last_preview_saved": True,
        "version": "v28.1"
    }
    
    # v28.1: Jeśli wymaga korekty - dodaj prompt
    if grammar_validation.get("correction_needed"):
        response_data["needs_correction"] = True
        response_data["correction_prompt"] = grammar_validation.get("correction_prompt", "")
        response_data["instruction"] = "POPRAW błędy i wyślij ponownie do preview_batch"
    
    return jsonify(response_data), 200


# ================================================================
# v26.1: BEST-OF-N BATCH GENERATION
# ================================================================
@project_routes.post("/api/project/<project_id>/generate_best_batch")
def generate_best_batch(project_id):
    """
    v26.1: Generuje N wersji batcha i zwraca najlepszą.
    
    Request body:
    {
        "prompt": "Treść promptu do generowania batcha",
        "n_candidates": 3,  // opcjonalne, default 3
        "min_score": 60     // opcjonalne, minimalny akceptowalny score
    }
    
    Response:
    {
        "status": "OK" | "WARN",
        "selected_content": "...",
        "selected_score": 85.2,
        "selected_variant": 2,
        "all_candidates": [...],
        "meets_minimum": true,
        "selection_reason": "..."
    }
    """
    if not BEST_OF_N_ENABLED:
        return jsonify({
            "error": "Best-of-N module not available",
            "fallback": "Use standard preview_batch endpoint"
        }), 501
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Field 'prompt' required"}), 400
    
    n_candidates = data.get("n_candidates", 3)
    min_score = data.get("min_score", 60)
    
    # Pobierz dane projektu
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    
    # Wywołaj Best-of-N selection
    try:
        result = select_best_batch(
            base_prompt=prompt,
            keywords_state=keywords_state,
            main_keyword=main_keyword,
            n_candidates=n_candidates
        )
    except Exception as e:
        return jsonify({
            "error": f"Generation failed: {str(e)}",
            "status": "ERROR"
        }), 500
    
    if result.get("error"):
        return jsonify({
            "error": result.get("error"),
            "status": "ERROR"
        }), 500
    
    # Określ status
    meets_minimum = result.get("meets_minimum", False)
    status = "OK" if meets_minimum else "WARN"
    
    return jsonify({
        "status": status,
        "selected_content": result.get("selected_content"),
        "selected_score": result.get("selected_score"),
        "selected_variant": result.get("selected_variant"),
        "all_candidates": result.get("all_candidates", []),
        "meets_minimum": meets_minimum,
        "selection_reason": result.get("selection_reason"),
        "component_scores": result.get("component_scores", {}),
        "issues": result.get("issues", []),
        "warnings": result.get("warnings", []),
        "version": "v26.1"
    }), 200


@project_routes.post("/api/project/<project_id>/preview_batch_v2")
def preview_batch_v2(project_id):
    """
    v26.1: Preview batch z opcjonalnym Best-of-N.
    
    Jeśli use_best_of_n=true i podano prompt, generuje 3 wersje.
    Jeśli podano text, waliduje jak dotychczas.
    
    Request body:
    {
        "text": "...",           // opcjonalne - do walidacji istniejącego tekstu
        "prompt": "...",         // opcjonalne - do generowania Best-of-N
        "use_best_of_n": true,   // opcjonalne, default false
        "n_candidates": 3        // opcjonalne
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    use_best_of_n = data.get("use_best_of_n", True)  # 🔧 v30.1: Domyślnie WŁĄCZONE
    prompt = data.get("prompt")
    batch_text = data.get("text") or data.get("batch_text")
    
    # Jeśli Best-of-N i mamy prompt
    if use_best_of_n and prompt and BEST_OF_N_ENABLED:
        # Przekieruj do generate_best_batch
        db = firestore.client()
        doc = db.collection("seo_projects").document(project_id).get()
        if not doc.exists:
            return jsonify({"error": "Project not found"}), 404
        
        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
        
        n_candidates = data.get("n_candidates", 3)
        
        try:
            result = select_best_batch(
                base_prompt=prompt,
                keywords_state=keywords_state,
                main_keyword=main_keyword,
                n_candidates=n_candidates
            )
            
            # Zwróć w formacie kompatybilnym z preview_batch
            return jsonify({
                "status": "OK" if result.get("meets_minimum") else "WARN",
                "method": "best_of_n",
                "selected_content": result.get("selected_content"),
                "selected_score": result.get("selected_score"),
                "all_candidates": result.get("all_candidates", []),
                "selection_reason": result.get("selection_reason"),
                "warnings": [{"type": "INFO", "message": result.get("selection_reason")}],
                "errors": [],
                "version": "v26.1"
            }), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Best-of-N failed: {str(e)}",
                "fallback": "Provide 'text' for standard validation"
            }), 500
    
    # Standardowa walidacja (jak w preview_batch)
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'prompt' with use_best_of_n required"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    s1_data = project_data.get("s1_data", {})

    report = unified_prevalidation(batch_text, keywords_state)
    
    warnings = report.get("warnings", [])
    errors = []
    
    density = report.get("density", 0)
    density_status, density_msg = get_density_status(density)
    if density_status in ["HIGH", "STUFFING"]:
        warnings.append({
            "type": "DENSITY_HIGH",
            "density": density,
            "status": density_status,
            "message": density_msg
        })
    
    # v28.1: Usunięto limit list
    list_count = count_bullet_lists(batch_text)
    
    status = "OK"
    if errors:
        status = "ERROR"
    elif len(warnings) > 2:
        status = "WARN"
    
    return jsonify({
        "status": status,
        "method": "standard",
        "density": density,
        "density_status": density_status,
        "warnings": warnings,
        "errors": errors,
        "version": "v26.1"
    }), 200


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def count_bullet_lists(text: str) -> int:
    """Liczy bloki list wypunktowanych."""
    lines = text.split('\n')
    list_blocks = 0
    in_list = False
    
    for line in lines:
        is_bullet = bool(re.match(r'^\s*[-•*]\s+|^\s*\d+\.\s+', line.strip()))
        
        if is_bullet and not in_list:
            list_blocks += 1
            in_list = True
        elif not is_bullet and line.strip():
            in_list = False
    
    html_lists = len(re.findall(r'<ul>|<ol>', text, re.IGNORECASE))
    
    return list_blocks + html_lists


def validate_h3_length(text: str, min_words: int = 80) -> dict:
    """Sprawdza czy sekcje H3 mają minimalną długość."""
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    issues = []
    sections = []
    
    for i, match in enumerate(h3_matches):
        h3_title = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = len(text)
        
        next_header = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
        if next_header:
            end = start + next_header.start()
        
        section_text = text[start:end].strip()
        section_text = re.sub(r'<[^>]+>', '', section_text)
        word_count = len(section_text.split())
        
        sections.append({"h3": h3_title, "word_count": word_count})
        
        if word_count < min_words:
            issues.append({
                "h3": h3_title,
                "word_count": word_count,
                "min_required": min_words,
                "deficit": min_words - word_count
            })
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "sections": sections,
        "total_h3": len(h3_matches)
    }


def check_main_vs_synonyms_in_text(text: str, main_keyword: str, keywords_state: dict) -> dict:
    """Sprawdza proporcję frazy głównej vs synonimy w tekście."""
    text_lower = text.lower()
    
    main_count = len(re.findall(rf"\b{re.escape(main_keyword.lower())}\b", text_lower))
    
    synonym_counts = {}
    synonym_total = 0
    
    for rid, meta in keywords_state.items():
        if meta.get("is_synonym_of_main"):
            keyword = meta.get("keyword", "").lower()
            count = len(re.findall(rf"\b{re.escape(keyword)}\b", text_lower))
            if count > 0:
                synonym_counts[meta.get("keyword")] = count
                synonym_total += count
    
    total = main_count + synonym_total
    main_ratio = main_count / total if total > 0 else 1.0
    
    return {
        "main_keyword": main_keyword,
        "main_count": main_count,
        "synonyms": synonym_counts,
        "synonym_total": synonym_total,
        "total": total,
        "main_ratio": round(main_ratio, 2),
        "valid": main_ratio >= 0.3,
        "warning": f"Za dużo synonimów! '{main_keyword}' ma tylko {main_ratio:.0%}. Zamień synonimy." if main_ratio < 0.3 else None
    }


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    v33.3: Oblicza podobieństwo między dwoma tekstami (0-1).
    Używa Jaccard similarity na słowach.
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Usuń stop words polskie
    stop_words = {'i', 'w', 'na', 'do', 'z', 'się', 'nie', 'to', 'że', 'o', 'jak', 'ale', 'po', 'co', 'tak', 'za', 'od', 'czy', 'tylko', 'są', 'jest', 'dla', 'oraz', 'przez', 'przy', 'już', 'być', 'ma', 'te', 'ten', 'ta', 'tym'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def get_ngrams_for_h2(
    h2_title: str,
    all_ngrams: List[dict],
    used_ngrams: List[str],
    max_ngrams: int = 4
) -> List[str]:
    """
    v33.3: Dopasowuje n-gramy do tematu H2 używając similarity.
    
    Zamiast przydzielać n-gramy sekwencyjnie, wybiera te które 
    są najbardziej semantycznie związane z nagłówkiem H2.
    
    Args:
        h2_title: Tytuł nagłówka H2 dla tego batcha
        all_ngrams: Lista wszystkich n-gramów z S1 (z weight)
        used_ngrams: Lista już użytych n-gramów w poprzednich batchach
        max_ngrams: Max liczba n-gramów do zwrócenia
    
    Returns:
        Lista n-gramów dopasowanych do H2
    """
    if not h2_title or not all_ngrams:
        return []
    
    # Filtruj już użyte
    available = [n for n in all_ngrams if n.get("ngram", "") not in used_ngrams]
    
    if not available:
        return []
    
    # Oblicz score dla każdego n-grama: similarity + weight
    scored = []
    for ngram_obj in available:
        ngram = ngram_obj.get("ngram", "")
        weight = ngram_obj.get("weight", 0.5)
        
        # Similarity do H2
        similarity = calculate_text_similarity(h2_title, ngram)
        
        # Bonus jeśli n-gram zawiera słowo z H2
        h2_words = set(h2_title.lower().split())
        ngram_words = set(ngram.lower().split())
        word_overlap_bonus = 0.3 if h2_words & ngram_words else 0
        
        # Final score
        score = similarity * 0.4 + weight * 0.4 + word_overlap_bonus * 0.2
        scored.append((ngram, score))
    
    # Sortuj po score malejąco
    scored.sort(key=lambda x: -x[1])
    
    # Zwróć top n-gramy
    return [s[0] for s in scored[:max_ngrams]]


def get_used_ngrams_from_batches(batches: List[dict], all_ngrams: List[str]) -> List[str]:
    """
    v33.3: Zbiera n-gramy które już zostały użyte w poprzednich batchach.
    """
    used = []
    all_text = " ".join([b.get("text", "") for b in batches]).lower()
    
    for ngram in all_ngrams:
        if ngram.lower() in all_text:
            used.append(ngram)
    
    return used


def check_ngram_coverage_in_text(text: str, required_ngrams: list) -> dict:
    """Sprawdza pokrycie n-gramów w tekście."""
    text_lower = text.lower()
    used = []
    missing = []
    
    for ngram in required_ngrams:
        if ngram and ngram.lower() in text_lower:
            used.append(ngram)
        elif ngram:
            missing.append(ngram)
    
    coverage = len(used) / len(required_ngrams) if required_ngrams else 1.0
    
    return {
        "coverage": round(coverage, 2),
        "used": used,
        "missing": missing,
        "valid": coverage >= 0.6
    }


# ================================================================
#  AUTO-CORRECT ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """Automatyczna korekta batcha."""
    data = request.get_json() or {}
    batch_text = data.get("text") or data.get("batch_text")
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    
    if not batch_text:
        batches = project_data.get("batches", [])
        if batches:
            for batch in reversed(batches):
                if batch.get("text"):
                    batch_text = batch.get("text")
                    break
    
    if not batch_text:
        batches = project_data.get("batches", [])
        all_texts = [b.get("text", "") for b in batches if b.get("text")]
        if all_texts:
            batch_text = "\n\n".join(all_texts)
    
    if not batch_text:
        return jsonify({
            "error": "No text provided",
            "hint": "Brak zapisanych batchy w projekcie lub wszystkie są puste",
            "batches_count": len(project_data.get("batches", []))
        }), 400
    
    keywords_state = project_data.get("keywords_state", {})
    
    under_keywords = []
    over_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        min_target = meta.get("target_min", 0)
        max_target = meta.get("target_max", 999)
        keyword = meta.get("keyword", "")
        
        if actual < min_target:
            under_keywords.append({
                "keyword": keyword,
                "missing": min_target - actual,
                "current": actual,
                "target_min": min_target
            })
        elif actual > max_target:
            over_keywords.append({
                "keyword": keyword,
                "excess": actual - max_target,
                "current": actual,
                "target_max": max_target
            })
    
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "corrected_text": batch_text
        }), 200
    
    if not GEMINI_API_KEY:
        return jsonify({"status": "ERROR", "error": "Gemini API not configured"}), 500
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        correction_instructions = []
        if under_keywords:
            under_list = "\n".join([f"  - '{kw['keyword']}': Dodaj {kw['missing']}x" for kw in under_keywords[:10]])
            correction_instructions.append(f"DODAJ te frazy:\n{under_list}")
        
        if over_keywords:
            over_list = "\n".join([f"  - '{kw['keyword']}': Usuń {kw['excess']}x" for kw in over_keywords[:5]])
            correction_instructions.append(f"USUŃ nadmiar:\n{over_list}")
        
        correction_prompt = f"""
Popraw tekst SEO:

{chr(10).join(correction_instructions)}

ZASADY:
1. Zachowaj h2: i h3:
2. Dodawaj frazy naturalnie
3. Zachowaj styl

TEKST:
{batch_text[:12000]}

Zwróć TYLKO poprawiony tekst.
"""
        
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        corrected_text = re.sub(r'^```(?:html)?\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        batches = project_data.get("batches", [])
        auto_saved = False
        new_metrics = {}
        
        if batches:
            batches[-1]["text"] = corrected_text
            batches[-1]["auto_corrected"] = True
            new_metrics = unified_prevalidation(corrected_text, keywords_state)
            batches[-1]["burstiness"] = new_metrics.get("burstiness", 0)
            batches[-1]["density"] = new_metrics.get("density", 0)
            doc_ref.update({"batches": batches})
            auto_saved = True
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "auto_saved": auto_saved,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords]
        }), 200
        
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ================================================================
# 📄 GET FULL ARTICLE (przed eksportem)
# ================================================================
@project_routes.get("/api/project/<project_id>/full_article")
def get_full_article(project_id):
    """
    v26.1: Zwraca pełną treść artykułu przed eksportem.
    GPT używa tego do przeglądu całości.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    keywords_state = data.get("keywords_state", {})
    
    # Złóż pełny tekst
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    # Walidacja końcowa
    coverage = validate_coverage(keywords_state)
    
    # Density
    density = 0
    density_status = "UNKNOWN"
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        density = prevalidation.get("density", 0)
        density_status, _ = get_density_status(density)
    
    # Statystyki
    word_count = len(full_text.split())
    h2_count = full_text.lower().count("h2:")
    h3_count = full_text.lower().count("h3:")
    
    return jsonify({
        "status": "OK",
        "full_article": full_text,
        "stats": {
            "word_count": word_count,
            "batch_count": len(batches),
            "h2_count": h2_count,
            "h3_count": h3_count
        },
        "coverage": coverage,
        "density": {
            "value": round(density, 2),
            "status": density_status
        },
        "topic": data.get("topic"),
        "main_keyword": data.get("main_keyword"),
        "version": "v26.1"
    }), 200


# ================================================================
# 🤖 GEMINI REVIEW (S5)
# ================================================================
@project_routes.post("/api/project/<project_id>/gemini_review")
def gemini_review(project_id):
    """
    v26.1: Wysyła artykuł do Gemini do analizy jakości.
    
    Request body (opcjonalne):
    {
        "focus": ["readability", "seo", "polish_quality"]  // na czym się skupić
    }
    
    Response:
    {
        "status": "APPROVED" | "NEEDS_REVISION",
        "score": 85,
        "recommendations": [...],
        "analysis": {...}
    }
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    topic = data.get("topic", "")
    main_keyword = data.get("main_keyword", topic)
    
    if not full_text or len(full_text) < 500:
        return jsonify({
            "error": "Article too short for review",
            "min_length": 500,
            "current_length": len(full_text)
        }), 400
    
    # Request body
    request_data = request.get_json() or {}
    focus_areas = request_data.get("focus", ["readability", "seo", "polish_quality"])
    
    # Prompt do Gemini
    review_prompt = f"""Przeanalizuj poniższy artykuł SEO i oceń jego jakość.

TEMAT: {topic}
GŁÓWNA FRAZA: {main_keyword}

ARTYKUŁ:
{full_text[:8000]}

OCEŃ (skala 1-100) i podaj rekomendacje dla:
1. CZYTELNOŚĆ - czy tekst jest płynny, zrozumiały, dobrze sformatowany?
2. SEO - czy struktura H2/H3 jest logiczna, czy frazy są naturalnie wplecione?
3. JAKOŚĆ JĘZYKA - czy nie ma błędów, sztucznych fraz AI, powtórzeń?
4. WARTOŚĆ MERYTORYCZNA - czy artykuł odpowiada na pytania użytkownika?

Odpowiedz w formacie JSON:
{{
    "overall_score": <1-100>,
    "scores": {{
        "readability": <1-100>,
        "seo": <1-100>,
        "polish_quality": <1-100>,
        "content_value": <1-100>
    }},
    "status": "APPROVED" lub "NEEDS_REVISION",
    "recommendations": [
        {{"area": "...", "issue": "...", "suggestion": "..."}}
    ],
    "strengths": ["...", "..."],
    "critical_issues": ["..."] 
}}
"""
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            review_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048
            )
        )
        
        if not response or not response.text:
            return jsonify({
                "error": "Empty response from Gemini",
                "status": "ERROR"
            }), 500
        
        # Parsuj JSON z odpowiedzi
        response_text = response.text.strip()
        
        # Usuń markdown code blocks jeśli są
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback - zwróć surową odpowiedź
            analysis = {
                "overall_score": 70,
                "status": "NEEDS_REVISION",
                "raw_response": response.text[:1000],
                "parse_error": True
            }
        
        # Zapisz wynik review do projektu
        doc.reference.update({
            "gemini_review": {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "analysis": analysis,
                "status": analysis.get("status", "UNKNOWN")
            }
        })
        
        return jsonify({
            "status": analysis.get("status", "UNKNOWN"),
            "overall_score": analysis.get("overall_score", 0),
            "scores": analysis.get("scores", {}),
            "recommendations": analysis.get("recommendations", []),
            "strengths": analysis.get("strengths", []),
            "critical_issues": analysis.get("critical_issues", []),
            "version": "v26.1"
        }), 200
        
    except Exception as e:
        print(f"[GEMINI_REVIEW] Error: {e}")
        return jsonify({
            "error": f"Gemini review failed: {str(e)}",
            "status": "ERROR"
        }), 500


# ================================================================
# 💾 SAVE FINAL ARTICLE (przed eksportem)
# ================================================================
@project_routes.post("/api/project/<project_id>/save_final")
def save_final_article(project_id):
    """
    v26.1: Zapisuje finalną wersję artykułu do bazy.
    Wywoływane po przejściu wszystkich review.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    batches = data.get("batches", [])
    keywords_state = data.get("keywords_state", {})
    
    # Złóż pełny tekst
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    # Konwertuj na HTML
    def convert_markers_to_html(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                title = stripped[3:].strip()
                result.append(f'<h2>{title}</h2>')
            elif stripped.lower().startswith('h3:'):
                title = stripped[3:].strip()
                result.append(f'<h3>{title}</h3>')
            elif stripped.startswith('- ') or stripped.startswith('• '):
                result.append(f'<li>{stripped[2:]}</li>')
            elif stripped:
                result.append(f'<p>{stripped}</p>')
        return '\n'.join(result)
    
    article_html = convert_markers_to_html(full_text)
    
    # Walidacja końcowa
    coverage = validate_coverage(keywords_state)
    
    # Density
    density = 0
    if full_text:
        prevalidation = unified_prevalidation(full_text, keywords_state)
        density = prevalidation.get("density", 0)
    
    # Zapisz do bazy
    final_data = {
        "final_article": {
            "text": full_text,
            "html": article_html,
            "word_count": len(full_text.split()),
            "saved_at": firestore.SERVER_TIMESTAMP
        },
        "final_stats": {
            "coverage": coverage,
            "density": round(density, 2),
            "batch_count": len(batches)
        },
        "status": "FINAL_SAVED"
    }
    
    doc.reference.update(final_data)
    
    return jsonify({
        "status": "SAVED",
        "message": "Final article saved to database",
        "word_count": len(full_text.split()),
        "coverage": coverage,
        "density": round(density, 2),
        "ready_for_export": True,
        "version": "v26.1"
    }), 200


# ================================================================
# 📦 EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    def convert_markers_to_html(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                title = stripped[3:].strip()
                result.append(f'<h2>{title}</h2>')
            elif stripped.lower().startswith('h3:'):
                title = stripped[3:].strip()
                result.append(f'<h3>{title}</h3>')
            else:
                result.append(line)
        return '\n'.join(result)
    
    article_html = convert_markers_to_html(full_text)
    
    # v25.0: Coverage info
    keywords_state = data.get("keywords_state", {})
    coverage = validate_coverage(keywords_state)

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_text": full_text,
        "article_html": article_html,
        "batch_count": len(batches),
        "coverage": coverage,
        "version": "v25.0"
    }), 200


# ================================================================
# 🔄 ALIASES
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    return auto_correct_batch(project_id)


@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    return preview_batch(project_id)


# ================================================================
# v27.2: PHRASE ANALYSIS - dokładne sprawdzenie fraz w tekście
# ================================================================
@project_routes.post("/api/project/<project_id>/analyze_phrases")
def analyze_phrases(project_id):
    """
    Analizuje DOKŁADNE wystąpienia fraz BASIC i EXTENDED w tekście.
    Pokazuje:
    - Gdzie dokładnie fraza występuje (indeks znaków)
    - W jakiej formie (oryginalna vs zlemmatyzowana)
    - Porównanie: regex vs lemmatizer vs firestore
    
    Użycie: przed FAQ żeby sprawdzić które frazy trzeba jeszcze użyć.
    """
    import re
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Field 'text' required"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Import keyword_counter
    try:
        from keyword_counter import count_keywords, count_keywords_for_state
        COUNTER_AVAILABLE = True
    except ImportError:
        COUNTER_AVAILABLE = False
    
    # Zbierz frazy
    all_phrases = []
    for rid, meta in keywords_state.items():
        kw = meta.get("keyword", "").strip()
        kw_type = meta.get("type", "BASIC").upper()
        if not kw:
            continue
        
        all_phrases.append({
            "rid": rid,
            "keyword": kw,
            "type": kw_type,
            "target_min": meta.get("target_min", 1),
            "target_max": meta.get("target_max", 5),
            "actual_in_firestore": meta.get("actual_uses", 0)
        })
    
    # Funkcja szukania DOKŁADNYCH wystąpień (regex, bez lemmatyzacji)
    def find_exact_regex(text_to_search: str, phrase: str) -> list:
        """Znajduje wszystkie dokładne wystąpienia frazy (case-insensitive, regex)."""
        matches = []
        text_lower = text_to_search.lower()
        phrase_lower = phrase.lower()
        
        # Szukaj z word boundaries
        pattern = r'\b' + re.escape(phrase_lower) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start = match.start()
            end = match.end()
            original_form = text_to_search[start:end]
            
            ctx_start = max(0, start - 25)
            ctx_end = min(len(text_to_search), end + 25)
            context = text_to_search[ctx_start:ctx_end]
            
            matches.append({
                "pos": f"{start}-{end}",
                "found": original_form,
                "ctx": f"...{context}..."
            })
        
        return matches
    
    # Policz każdą frazę na 3 sposoby
    analysis = []
    
    # 1. Unified counter (jeśli dostępny)
    if COUNTER_AVAILABLE:
        keywords_list = [p["keyword"] for p in all_phrases]
        unified_result = count_keywords(text, keywords_list, return_per_segment=False, return_paragraph_stuffing=False)
        overlapping = unified_result.get("overlapping", {})
        exclusive = unified_result.get("exclusive", {})
    else:
        overlapping = {}
        exclusive = {}
    
    for phrase_info in all_phrases:
        kw = phrase_info["keyword"]
        
        # Regex count (dokładne dopasowanie, bez lemmatyzacji)
        regex_matches = find_exact_regex(text, kw)
        regex_count = len(regex_matches)
        
        # Unified counter counts
        overlap_count = overlapping.get(kw, 0)
        excl_count = exclusive.get(kw, 0)
        
        # Firestore value
        firestore_count = phrase_info["actual_in_firestore"]
        
        # Status
        target_min = phrase_info["target_min"] if phrase_info["type"] != "EXTENDED" else 1
        
        # Wykryj rozbieżności
        discrepancy = None
        if regex_count != overlap_count:
            discrepancy = f"REGEX({regex_count}) != LEMMA({overlap_count})"
        
        analysis.append({
            "keyword": kw,
            "type": phrase_info["type"],
            "rid": phrase_info["rid"],
            
            # 3 metody liczenia
            "count_regex": regex_count,           # Dokładne dopasowanie (bez odmian)
            "count_overlapping": overlap_count,   # Z lemmatyzacją (Google-style)
            "count_exclusive": excl_count,        # Bez zagnieżdżonych
            "count_firestore": firestore_count,   # Zapisane w Firestore
            
            # Targety
            "target_min": target_min,
            "target_max": phrase_info["target_max"],
            
            # Status
            "status": "✅" if overlap_count >= target_min else "❌",
            "discrepancy": discrepancy,
            
            # Przykłady (max 5)
            "examples": regex_matches[:5]
        })
    
    # Podsumowanie
    basic_analysis = [a for a in analysis if a["type"] == "BASIC"]
    extended_analysis = [a for a in analysis if a["type"] == "EXTENDED"]
    
    basic_missing = [a["keyword"] for a in basic_analysis if a["count_overlapping"] < a["target_min"]]
    extended_missing = [a["keyword"] for a in extended_analysis if a["count_overlapping"] < 1]
    
    # Wykryj problemy
    problems = []
    for a in analysis:
        if a["discrepancy"]:
            problems.append(f"{a['keyword']}: {a['discrepancy']}")
        if a["count_firestore"] != a["count_overlapping"]:
            problems.append(f"{a['keyword']}: Firestore({a['count_firestore']}) != Text({a['count_overlapping']})")
    
    return jsonify({
        "project_id": project_id,
        "text_length": len(text),
        "word_count": len(text.split()),
        "counter_type": "unified_lemmatizer" if COUNTER_AVAILABLE else "regex_only",
        
        "summary": {
            "basic_total": len(basic_analysis),
            "basic_covered": len(basic_analysis) - len(basic_missing),
            "basic_missing": len(basic_missing),
            "extended_total": len(extended_analysis),
            "extended_covered": len(extended_analysis) - len(extended_missing),
            "extended_missing": len(extended_missing)
        },
        
        "missing_basic": basic_missing,
        "missing_extended": extended_missing,
        
        "problems_detected": problems[:10],
        
        "analysis": analysis,
        
        "legend": {
            "count_regex": "Dokładne dopasowanie (bez odmian)",
            "count_overlapping": "Z lemmatyzacją + zagnieżdżone (Google-style)",
            "count_exclusive": "Z lemmatyzacją, BEZ zagnieżdżonych",
            "count_firestore": "Wartość zapisana w Firestore (może być nieaktualna)"
        }
    })
