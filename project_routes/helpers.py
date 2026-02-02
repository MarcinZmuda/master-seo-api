"""
PROJECT ROUTES - HELPERS v44.1
==============================
Funkcje pomocnicze wydzielone z project_routes.py

Zawiera:
- Entity & N-gram helpers
- Keyword distribution
- Density configuration
- Coverage validation
- calculate_suggested_v25

Autor: BRAJEN SEO Engine
"""

import re
import math
import os
from typing import List, Dict, Any, Optional

# ================================================================
# GEMINI CONFIG
# ================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# DENSITY CONFIGURATION (v25.0)
# ================================================================
DENSITY_OPTIMAL_MIN = 0.5
DENSITY_OPTIMAL_MAX = 1.5
DENSITY_ACCEPTABLE_MAX = 2.0
DENSITY_WARNING_MAX = 2.5
DENSITY_MAX = 3.0


# ================================================================
# SOFT CAP & SHORT KEYWORD CONFIGURATION (v26.1)
# ================================================================
SOFT_CAP_THRESHOLD = 0.75
SHORT_KEYWORD_MAX_WORDS = 2
SHORT_KEYWORD_MAX_REDUCTION = 0.6
SHORT_KEYWORD_ABSOLUTE_MAX = 8


# ================================================================
# Entity & N-gram Guidance Helpers (v29.3)
# ================================================================

def get_entities_to_introduce(top_entities: list, batch_num: int, total_batches: int, previous_texts: list) -> list:
    """
    Zwraca encje do wprowadzenia w tym batchu.
    Rozdziela encje równomiernie między batche.
    """
    if not top_entities:
        return []
    
    previous_text = " ".join(previous_texts).lower()
    
    unused_entities = []
    for entity in top_entities:
        entity_name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
        if entity_name.lower() not in previous_text:
            unused_entities.append(entity)
    
    if not unused_entities:
        return []
    
    remaining_batches = max(total_batches - batch_num + 1, 1)
    entities_per_batch = max(1, len(unused_entities) // remaining_batches)
    
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
    
    return result[:3]


def get_already_defined_entities(previous_texts: list) -> list:
    """Zwraca encje już zdefiniowane w poprzednich batchach."""
    if not previous_texts:
        return []
    
    previous_text = " ".join(previous_texts).lower()
    
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
            if len(match) > 3:
                defined.add(match.strip())
    
    return list(defined)[:10]


def get_overused_phrases(previous_texts: list, main_keyword: str) -> list:
    """Znajduje frazy użyte zbyt często (>5x)."""
    if not previous_texts:
        return []
    
    try:
        if isinstance(previous_texts, str):
            previous_texts = [previous_texts]
        elif not isinstance(previous_texts, list):
            previous_texts = list(previous_texts) if hasattr(previous_texts, '__iter__') else []
        previous_texts = [str(t) for t in previous_texts if t is not None]
    except Exception as e:
        print(f"[WARNING] get_overused_phrases input error: {e}")
        return []
    
    if not previous_texts:
        return []
    
    previous_text = " ".join(previous_texts).lower()
    overused = []
    
    main_count = previous_text.count(main_keyword.lower())
    if main_count > 5:
        overused.append({
            "phrase": main_keyword,
            "count": main_count,
            "warning": f"Użyto {main_count}x - rozważ synonimy"
        })
    
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
    """Zwraca synonimy dla nadużywanych fraz."""
    try:
        if previous_texts is None:
            previous_texts = []
        elif isinstance(previous_texts, str):
            previous_texts = [previous_texts]
        elif not isinstance(previous_texts, list):
            previous_texts = list(previous_texts) if hasattr(previous_texts, '__iter__') else []
        previous_texts = [str(t) for t in previous_texts if t is not None]
    except Exception as e:
        print(f"[WARNING] get_synonyms_for_overused input error: {e}")
        previous_texts = []
    
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
    
    main_lower = main_keyword.lower()
    for key, synonyms in SYNONYM_MAP.items():
        if key in main_lower or main_lower in key:
            result[main_keyword] = synonyms
            break
    
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
# DISTRIBUTE EXTENDED KEYWORDS (v30.1)
# ================================================================

def distribute_extended_keywords(extended_keywords: List[Dict], total_batches: int) -> Dict[int, List[Dict]]:
    """
    v30.1: Rozdziela EXTENDED frazy równomiernie między batche.
    """
    if not extended_keywords or total_batches < 1:
        return {}
    
    distribution = {i: [] for i in range(1, total_batches + 1)}
    
    keywords_per_batch = max(3, len(extended_keywords) // total_batches)
    
    for i, kw in enumerate(extended_keywords):
        batch_num = (i // keywords_per_batch) + 1
        if batch_num > total_batches:
            batch_num = total_batches
        distribution[batch_num].append(kw)
    
    for batch_num in distribution:
        if len(distribution[batch_num]) > 6:
            excess = distribution[batch_num][6:]
            distribution[batch_num] = distribution[batch_num][:6]
            
            for j, kw in enumerate(excess):
                next_batch = ((batch_num + j) % total_batches) + 1
                if len(distribution[next_batch]) < 6:
                    distribution[next_batch].append(kw)
    
    return distribution


def get_section_length_guidance(batch_num: int, total_batches: int, batch_type: str) -> dict:
    """Zwraca guidance o różnej długości sekcji."""
    LENGTH_PATTERNS = {
        1: {"profile": "SHORT", "range": "180-220", "reason": "Intro - zwięzłe wprowadzenie"},
        2: {"profile": "LONG", "range": "350-400", "reason": "Główny temat - rozbudowana treść"},
        3: {"profile": "MEDIUM", "range": "250-300", "reason": "Rozwinięcie tematu"},
        4: {"profile": "LONG", "range": "320-380", "reason": "Praktyczne porady - więcej szczegółów"},
        5: {"profile": "MEDIUM", "range": "240-280", "reason": "Uzupełnienie tematu"},
        6: {"profile": "SHORT", "range": "200-250", "reason": "Sekcja przed FAQ - krótsza"},
    }
    
    pattern = LENGTH_PATTERNS.get(batch_num, {"profile": "MEDIUM", "range": "250-300", "reason": "Standardowa sekcja"})
    
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


# ================================================================
# DENSITY & SOFT CAP FUNCTIONS (v26.1)
# ================================================================

def get_adjusted_target_max(keyword: str, original_max: int, word_count: int = None) -> int:
    """v26.1: Zwraca skorygowany target_max dla frazy."""
    if word_count is None:
        word_count = len(keyword.split())
    
    if word_count <= SHORT_KEYWORD_MAX_WORDS:
        reduced_max = int(original_max * SHORT_KEYWORD_MAX_REDUCTION)
        return min(reduced_max, SHORT_KEYWORD_ABSOLUTE_MAX)
    
    return original_max


def check_soft_cap(actual: int, target_max: int, keyword: str) -> Optional[dict]:
    """v26.1: Sprawdza czy fraza zbliża się do limitu (soft cap)."""
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
# COVERAGE VALIDATION (v29.1)
# ================================================================

def validate_coverage(keywords_state: dict) -> dict:
    """v29.1: Sprawdza coverage dla BASIC i EXTENDED keywords."""
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
# SYNONYM DETECTION (v22.4)
# ================================================================

def detect_main_keyword_synonyms(main_keyword: str) -> list:
    """Używa Gemini do znalezienia synonimów frazy głównej."""
    if not GENAI_AVAILABLE or not GEMINI_API_KEY:
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
        print(f"[SYNONYM] Error: {e}")
        return []


# ================================================================
# CALCULATE SUGGESTED v25 (v29.1)
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
    """
    
    word_count = len(keyword.split())
    if kw_type != "EXTENDED" and word_count <= SHORT_KEYWORD_MAX_WORDS:
        adjusted_max = get_adjusted_target_max(keyword, target_max, word_count)
        if adjusted_max < target_max:
            target_max = adjusted_max
    
    remaining_to_max = max(0, target_max - actual)
    remaining_to_min = max(0, target_min - actual)
    
    soft_cap_info = check_soft_cap(actual, target_max, keyword)
    soft_cap_warning = soft_cap_info.get("message") if soft_cap_info else None
    
    # === EXTENDED ===
    if kw_type == "EXTENDED":
        if actual == 0:
            if remaining_batches <= 2:
                return {
                    "suggested": 1,
                    "priority": "CRITICAL",
                    "instruction": f"🔴 KRYTYCZNE - MUSISZ użyć min 1x (zostały {remaining_batches} batchy!)",
                    "hard_max_this_batch": 2,
                    "flexibility": "NONE",
                    "adjusted_max": target_max
                }
            elif remaining_batches <= 3:
                return {
                    "suggested": 1,
                    "priority": "HIGH",
                    "instruction": f"📌 WPLEĆ min 1x (extended - zostały {remaining_batches} batchy)",
                    "hard_max_this_batch": 2,
                    "flexibility": "LOW",
                    "adjusted_max": target_max
                }
            else:
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
    
    if remaining_to_max == 0:
        return {
            "suggested": 0,
            "priority": "LOCKED",
            "instruction": f"🔒 LOCKED - limit osiągnięty ({target_max}x)",
            "hard_max_this_batch": 0,
            "flexibility": "NONE",
            "adjusted_max": target_max
        }
    
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
# EXPORTS
# ================================================================

__all__ = [
    # Constants
    "DENSITY_OPTIMAL_MIN",
    "DENSITY_OPTIMAL_MAX", 
    "DENSITY_ACCEPTABLE_MAX",
    "DENSITY_WARNING_MAX",
    "DENSITY_MAX",
    "SOFT_CAP_THRESHOLD",
    "SHORT_KEYWORD_MAX_WORDS",
    "SHORT_KEYWORD_MAX_REDUCTION",
    "SHORT_KEYWORD_ABSOLUTE_MAX",
    "GEMINI_MODEL",
    
    # Entity helpers
    "get_entities_to_introduce",
    "get_already_defined_entities",
    "get_overused_phrases",
    "get_synonyms_for_overused",
    
    # Keyword distribution
    "distribute_extended_keywords",
    "get_section_length_guidance",
    
    # Density & soft cap
    "get_adjusted_target_max",
    "check_soft_cap",
    "get_density_status",
    
    # Coverage
    "validate_coverage",
    
    # Synonyms
    "detect_main_keyword_synonyms",
    
    # Suggested calculation
    "calculate_suggested_v25",
]
