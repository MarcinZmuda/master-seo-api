"""
===============================================================================
🔍 UNIFIED VALIDATOR v35.1 (BRAJEN) - OPTIMIZED
===============================================================================
v35.1 ZMIANY OPTYMALIZACYJNE:
- 🆕 JITTER_ENABLED = False (domyślnie wyłączone - zbędne warningi)
- 🆕 SUBOPTIMAL_BURSTINESS zmienione z WARNING na INFO (nie blokuje)
- 🆕 Elastyczne progi w zależności od attempt number
- 🆕 Zmniejszona penalizacja punktowa za warningi (5 → 3)
- 🆕 AUTO_APPROVE_THRESHOLD = 2 (zamiast 3)

EFEKT: -40% iteracji, -70% czasu na batch
===============================================================================
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# v40.2: Import from core_metrics (Single Source of Truth)
try:
    from core_metrics import (
        calculate_burstiness_simple as _calculate_burstiness_core,
        split_into_sentences as _split_sentences_core,
        TRANSITION_WORDS_PL as TRANSITION_WORDS_CORE
    )
    CORE_METRICS_AVAILABLE = True
except ImportError:
    CORE_METRICS_AVAILABLE = False
    TRANSITION_WORDS_CORE = None
    print("[VALIDATOR] ⚠️ core_metrics not available, using local functions")

# ================================================================
# 🧠 Współdzielony model spaCy
# ================================================================
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
    print("[VALIDATOR] ✅ Używam współdzielonego modelu spaCy")
except ImportError:
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
        print("[VALIDATOR] ⚠️ Załadowano lokalny model pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")


# ================================================================
# 📊 STAŁE KONFIGURACYJNE - v35.1 OPTIMIZED
# ================================================================
class ValidationConfig:
    """
    Centralna konfiguracja wszystkich progów walidacji.
    
    🆕 v35.1 OPTIMIZED:
    - JITTER wyłączone domyślnie
    - Poluzowane progi burstiness
    - Mniejsze penalizacje
    """
    
    # ================================================================
    # 🆕 v35.1: FLAGI OPTYMALIZACYJNE
    # ================================================================
    JITTER_ENABLED = False           # 🆕 Wyłącz JITTER (powodował zbędne warningi)
    AUTO_APPROVE_THRESHOLD = 2       # 🆕 Auto-approve po 2 próbach (było 3)
    RELAXED_MODE_AFTER_ATTEMPT = 1   # 🆕 Poluzuj progi od próby 2
    
    # ================================================================
    # BURSTINESS - POLUZOWANE PROGI v35.1
    # ================================================================
    BURSTINESS_CRITICAL_LOW = 1.3    # 🔧 było 1.5 (CV 0.26)
    BURSTINESS_WARNING_LOW = 1.8     # 🔧 było 2.0 (CV 0.36)
    BURSTINESS_OPTIMAL_MIN = 2.2     # 🔧 było 2.5 (CV 0.44) - KLUCZOWA ZMIANA
    BURSTINESS_OPTIMAL_MAX = 4.2     # 🔧 było 4.0 (CV 0.84)
    BURSTINESS_WARNING_HIGH = 4.8    # 🔧 było 4.5 (CV 0.96)
    BURSTINESS_CRITICAL_HIGH = 5.5   # 🔧 było 5.0 (CV 1.1)
    
    # Stare aliasy dla kompatybilności
    BURSTINESS_MIN = BURSTINESS_WARNING_LOW
    BURSTINESS_MAX = BURSTINESS_WARNING_HIGH
    BURSTINESS_OPTIMAL = 3.2
    
    # ================================================================
    # INNE PROGI - POLUZOWANE
    # ================================================================
    TRANSITION_RATIO_MIN = 0.20      # 🔧 było 0.25
    TRANSITION_RATIO_MAX = 0.55      # 🔧 było 0.50
    DENSITY_MAX = 3.5                # 🔧 było 3.0
    DENSITY_WARNING = 3.0            # 🔧 było 2.5
    H3_MIN_WORDS = 60                # 🔧 było 80
    LIST_MIN = 1
    LIST_MAX = 3                     # 🔧 było 2
    INTRO_MIN_WORDS = 35             # 🔧 było 40
    INTRO_MAX_WORDS = 70             # 🔧 było 60
    MAIN_KEYWORD_RATIO_MIN = 0.25    # 🔧 było 0.30
    H2_MAIN_KEYWORD_MAX = 2          # 🔧 było 1
    NGRAM_COVERAGE_MIN = 0.50        # 🔧 było 0.60
    
    # ================================================================
    # 🆕 v35.1: PENALIZACJE PUNKTOWE - ZMNIEJSZONE
    # ================================================================
    PENALTY_CRITICAL = 20            # 🔧 było 25
    PENALTY_ERROR = 12               # 🔧 było 15
    PENALTY_WARNING = 3              # 🔧 było 5
    PENALTY_INFO = 0                 # 🔧 było 1
    
    # 🆕 v44.1: TRANSITION_WORDS z core_metrics (Single Source of Truth)
    TRANSITION_WORDS = list(TRANSITION_WORDS_CORE) if TRANSITION_WORDS_CORE else [
        "również", "także", "ponadto", "jednak", "natomiast", "dlatego", "ponieważ"
    ]
    
    BANNED_SECTION_OPENERS = [
        "dlatego", "ponadto", "dodatkowo", "w związku z tym", 
        "tym samym", "warto", "należy"
    ]
    
    BANNED_INTRO_OPENERS = [
        "w dzisiejszych czasach", "warto wiedzieć", "jak wiadomo",
        "każdy z nas", "coraz więcej osób", "nie ulega wątpliwości",
        "nie da się ukryć"
    ]
    
    @classmethod
    def get_relaxed_config(cls, attempt: int = 1):
        """
        🆕 v35.1: Zwraca poluzowane progi dla kolejnych prób.
        Im więcej prób, tym bardziej elastyczne progi.
        """
        if attempt <= cls.RELAXED_MODE_AFTER_ATTEMPT:
            return cls  # Standardowe progi
        
        # Dynamicznie poluzuj progi
        relaxation = 1.0 + (attempt - 1) * 0.15  # +15% na próbę
        
        class RelaxedConfig(cls):
            BURSTINESS_OPTIMAL_MIN = cls.BURSTINESS_OPTIMAL_MIN / relaxation
            DENSITY_MAX = cls.DENSITY_MAX * relaxation
            NGRAM_COVERAGE_MIN = cls.NGRAM_COVERAGE_MIN / relaxation
            H3_MIN_WORDS = int(cls.H3_MIN_WORDS / relaxation)
        
        return RelaxedConfig


# ================================================================
# 📊 SEMANTIC CONFIG (v31.0) - bez zmian
# ================================================================
@dataclass
class SemanticConfig:
    """Progi walidacji semantycznej."""
    ENTITY_DENSITY_MIN: float = 2.5
    ENTITY_DENSITY_MAX: float = 7.0
    HARD_ENTITY_RATIO_MIN: float = 0.15
    TOPIC_COMPLETENESS_MIN: float = 0.60
    ENTITY_GAP_MIN: float = 0.40
    SOURCE_EFFORT_MIN: float = 0.35


ENTITY_TYPE_WEIGHTS = {
    "PERSON": 1.5, "PER": 1.5,
    "ORGANIZATION": 1.3, "ORG": 1.3,
    "LEGAL_ACT": 1.4,
    "PUBLICATION": 1.2,
    "STANDARD": 1.1,
    "PRODUCT": 1.0,
    "LOCATION": 0.8, "LOC": 0.8, "GPE": 0.8,
    "DATE": 0.6,
}

HARD_ENTITY_TYPES = {"PERSON", "PER", "ORGANIZATION", "ORG", "LEGAL_ACT", "PUBLICATION", "STANDARD"}


# ================================================================
# 📋 STRUKTURY DANYCH
# ================================================================
class Severity(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Severity
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details
        }


@dataclass
class ValidationResult:
    is_valid: bool
    score: int
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    keywords_analysis: Dict[str, Any] = field(default_factory=dict)
    structure_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "keywords_analysis": self.keywords_analysis,
            "structure_analysis": self.structure_analysis,
            "summary": {
                "critical": len([i for i in self.issues if i.severity == Severity.CRITICAL]),
                "errors": len([i for i in self.issues if i.severity == Severity.ERROR]),
                "warnings": len([i for i in self.issues if i.severity == Severity.WARNING]),
                "infos": len([i for i in self.issues if i.severity == Severity.INFO])
            }
        }
    
    def get_critical(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]
    
    def get_errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


# ================================================================
# 🔧 FUNKCJE POMOCNICZE
# ================================================================
def count_words(text: str) -> int:
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return len(clean.split()) if clean else 0


def extract_intro(text: str) -> str:
    h2_match = re.search(r'(^h2:|<h2)', text, re.MULTILINE | re.IGNORECASE)
    if h2_match:
        return text[:h2_match.start()].strip()
    return text[:500]


def extract_h2_titles(text: str) -> List[str]:
    titles = []
    titles.extend(re.findall(r'^h2:\s*(.+)$', text, re.MULTILINE | re.IGNORECASE))
    titles.extend(re.findall(r'<h2[^>]*>([^<]+)</h2>', text, re.IGNORECASE))
    return [t.strip() for t in titles if t.strip()]


def extract_h3_sections(text: str) -> List[Dict[str, Any]]:
    sections = []
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    for i, match in enumerate(h3_matches):
        title = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = len(text)
        next_header = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
        if next_header:
            end = start + next_header.start()
        section_text = text[start:end].strip()
        section_text = re.sub(r'<[^>]+>', '', section_text)
        word_count = len(section_text.split())
        sections.append({"title": title, "word_count": word_count, "position": i})
    return sections


def count_lists(text: str) -> int:
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


def lemmatize_text(text: str) -> List[str]:
    clean = re.sub(r'<[^>]+>', ' ', text.lower())
    clean = re.sub(r'\s+', ' ', clean)
    doc = nlp(clean)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


def count_keyword_occurrences(text_lemmas: List[str], keyword: str) -> int:
    kw_doc = nlp(keyword.lower())
    kw_lemmas = [t.lemma_.lower() for t in kw_doc if t.is_alpha]
    if not kw_lemmas:
        return 0
    kw_len = len(kw_lemmas)
    count = 0
    for i in range(len(text_lemmas) - kw_len + 1):
        if text_lemmas[i:i + kw_len] == kw_lemmas:
            count += 1
    return count


# ================================================================
# 📊 WALIDATORY METRYK
# ================================================================

# v40.2: Use core_metrics if available
if CORE_METRICS_AVAILABLE:
    def calculate_burstiness(text: str) -> float:
        """Deleguje do core_metrics.calculate_burstiness_simple"""
        return _calculate_burstiness_core(text)
else:
    def calculate_burstiness(text: str) -> float:
        """FALLBACK - use core_metrics instead"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 3:
            return 3.5
        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if not mean:
            return 3.5
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        raw_score = math.sqrt(variance) / mean
        return round(raw_score * 5, 2)


def validate_burstiness(burstiness: float, issues: List[ValidationIssue], attempt: int = 1) -> Dict[str, Any]:
    """
    🆕 v35.1 OPTIMIZED: Walidacja burstiness z elastycznymi progami.
    
    ZMIANY:
    - SUBOPTIMAL_BURSTINESS → INFO zamiast WARNING (nie blokuje!)
    - Poluzowane progi dla kolejnych prób
    """
    config = ValidationConfig.get_relaxed_config(attempt)
    cv_value = burstiness / 5
    details = {
        "value": burstiness, 
        "cv": round(cv_value, 2),
        "status": "OK", 
        "level": None,
        "attempt": attempt  # 🆕 Info o próbie
    }
    
    if burstiness < config.BURSTINESS_CRITICAL_LOW:
        details["status"] = "CRITICAL"
        details["level"] = "critical_low"
        issues.append(ValidationIssue(
            "CRITICAL_LOW_BURSTINESS", 
            f"Burstiness KRYTYCZNIE niski: {burstiness:.2f} (CV {cv_value:.2f} < 0.26) - silny sygnał AI!", 
            Severity.CRITICAL,
            {"value": burstiness, "cv": cv_value, "threshold": config.BURSTINESS_CRITICAL_LOW}
        ))
    elif burstiness < config.BURSTINESS_WARNING_LOW:
        details["status"] = "WARNING"
        details["level"] = "warning_low"
        issues.append(ValidationIssue(
            "LOW_BURSTINESS", 
            f"Burstiness za niski: {burstiness:.2f} (CV {cv_value:.2f} < 0.36)", 
            Severity.WARNING, 
            {"value": burstiness, "cv": cv_value, "threshold": config.BURSTINESS_WARNING_LOW}
        ))
    elif burstiness < config.BURSTINESS_OPTIMAL_MIN:
        # 🆕 v35.1: Zmienione z WARNING na INFO - nie blokuje!
        details["status"] = "INFO"
        details["level"] = "below_optimal"
        issues.append(ValidationIssue(
            "SUBOPTIMAL_BURSTINESS", 
            f"Burstiness poniżej optymalnego: {burstiness:.2f} (CV {cv_value:.2f}) - OK, nie blokuje", 
            Severity.INFO,  # 🔧 było WARNING
            {"value": burstiness, "cv": cv_value, "threshold": config.BURSTINESS_OPTIMAL_MIN}
        ))
    elif burstiness > config.BURSTINESS_CRITICAL_HIGH:
        details["status"] = "CRITICAL"
        details["level"] = "critical_high"
        issues.append(ValidationIssue(
            "CRITICAL_HIGH_BURSTINESS", 
            f"Burstiness KRYTYCZNIE wysoki: {burstiness:.2f} (CV {cv_value:.2f} > 1.1)", 
            Severity.CRITICAL, 
            {"value": burstiness, "cv": cv_value, "threshold": config.BURSTINESS_CRITICAL_HIGH}
        ))
    elif burstiness > config.BURSTINESS_WARNING_HIGH:
        details["status"] = "WARNING"
        details["level"] = "warning_high"
        issues.append(ValidationIssue(
            "HIGH_BURSTINESS", 
            f"Burstiness za wysoki: {burstiness:.2f} (CV {cv_value:.2f} > 0.96)", 
            Severity.WARNING, 
            {"value": burstiness, "cv": cv_value, "threshold": config.BURSTINESS_WARNING_HIGH}
        ))
    else:
        if config.BURSTINESS_OPTIMAL_MIN <= burstiness <= config.BURSTINESS_OPTIMAL_MAX:
            details["level"] = "optimal"
        else:
            details["level"] = "acceptable"
    
    return details


def calculate_transition_ratio(text: str) -> Dict[str, Any]:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences)}
    transition_count = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()[:100]
        if any(tw in sentence_lower for tw in ValidationConfig.TRANSITION_WORDS):
            transition_count += 1
    ratio = transition_count / len(sentences)
    return {"ratio": round(ratio, 3), "count": transition_count, "total": len(sentences)}


def calculate_density(text: str, keywords_state: Dict) -> float:
    if not text or not keywords_state:
        return 0.0
    text_lower = text.lower()
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    keyword_count = 0
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if keyword:
            keyword_count += len(re.findall(rf"\b{re.escape(keyword)}\b", text_lower))
    return round((keyword_count / total_words) * 100, 2)


# ================================================================
# 🆕 v36.5: TEMPLATE PATTERN DETECTION
# ================================================================

TEMPLATE_PATTERNS = {
    "conditional": {
        "patterns": [
            r"wtedy,?\s+gdy",
            r"w przypadku,?\s+gdy", 
            r"w sytuacji,?\s+gdy",
            r"jeśli.*to\s+",
            r"kiedy.*wówczas"
        ],
        "max_allowed": 2,
        "severity": "WARNING"
    },
    "emphasis": {
        "patterns": [
            r"warto\s+(?:zauważyć|podkreślić|wspomnieć|wiedzieć)",
            r"należy\s+(?:zauważyć|podkreślić|wspomnieć|pamiętać)",
            r"trzeba\s+(?:zauważyć|podkreślić|wspomnieć|pamiętać)",
            r"istotne\s+(?:jest|będzie)",
            r"kluczowe\s+(?:jest|znaczenie)"
        ],
        "max_allowed": 2,
        "severity": "WARNING"
    },
    "negation_universal": {
        "patterns": [
            r"nie\s+każd[yaei]",
            r"nie\s+zawsze",
            r"nie\s+oznacza\s+(?:to\s+)?(?:że|automatycznie)",
            r"nie\s+(?:jest|będzie)\s+(?:to\s+)?(?:jedyn|konieczn)"
        ],
        "max_allowed": 2,
        "severity": "WARNING"
    },
    "structure_markers": {
        "patterns": [
            r"ma\s+to\s+miejsce",
            r"dotyczy\s+to\s+(?:przede\s+wszystkim|głównie|szczególnie)",
            r"odnosi\s+się\s+(?:to\s+)?do",
            r"chodzi\s+(?:tutaj\s+)?o"
        ],
        "max_allowed": 2,
        "severity": "INFO"
    },
    "ai_openers": {
        "patterns": [
            r"^w\s+dzisiejszych\s+czasach",
            r"^w\s+obecnych\s+czasach", 
            r"^współcześnie",
            r"^w\s+niniejszym\s+artykule",
            r"^artykuł\s+(?:ten\s+)?(?:omawia|przedstawia|opisuje)"
        ],
        "max_allowed": 0,
        "severity": "ERROR"
    }
}


def check_template_patterns(text: str) -> List[ValidationIssue]:
    """
    🆕 v36.5: Wykrywa nadużywane wzorce szablonowe charakterystyczne dla AI.
    
    Sprawdza:
    - Powtarzające się struktury warunkowe ("wtedy, gdy")
    - Schematyczne wprowadzenia ("warto zauważyć")
    - Uniwersalne negacje ("nie każdy", "nie zawsze")
    - Znaczniki strukturalne ("ma to miejsce")
    - AI openers ("w dzisiejszych czasach")
    
    Returns:
        Lista ValidationIssue z wykrytymi problemami
    """
    issues = []
    text_lower = text.lower()
    
    template_stats = {}
    
    for category, config in TEMPLATE_PATTERNS.items():
        category_count = 0
        matched_patterns = []
        
        for pattern in config["patterns"]:
            try:
                matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                category_count += len(matches)
                if matches:
                    matched_patterns.extend(matches[:3])
            except re.error:
                continue
        
        template_stats[category] = {
            "count": category_count,
            "max_allowed": config["max_allowed"],
            "examples": matched_patterns[:3]
        }
        
        if category_count > config["max_allowed"]:
            severity = Severity.ERROR if config["severity"] == "ERROR" else (
                Severity.WARNING if config["severity"] == "WARNING" else Severity.INFO
            )
            
            examples_str = ", ".join(f'"{m}"' for m in matched_patterns[:2]) if matched_patterns else ""
            
            issues.append(ValidationIssue(
                f"TEMPLATE_OVERUSE_{category.upper()}",
                f"Wzorzec '{category}' użyty {category_count}× (max {config['max_allowed']}). {examples_str}",
                severity,
                {
                    "category": category,
                    "count": category_count,
                    "max_allowed": config["max_allowed"],
                    "examples": matched_patterns[:3]
                }
            ))
    
    # Sprawdź ogólną templateowość
    total_templates = sum(s["count"] for s in template_stats.values())
    if total_templates > 8:
        issues.append(ValidationIssue(
            "HIGH_TEMPLATE_DENSITY",
            f"Wysoka templateowość: {total_templates} wzorców - tekst brzmi sztucznie",
            Severity.WARNING,
            {"total_templates": total_templates, "breakdown": template_stats}
        ))
    
    return issues, template_stats


# ================================================================
# 🎯 GŁÓWNA FUNKCJA WALIDACJI - v35.1 OPTIMIZED
# ================================================================
def validate_content(
    text: str,
    keywords_state: Dict = None,
    main_keyword: str = None,
    required_ngrams: List[str] = None,
    is_intro_batch: bool = False,
    existing_lists_count: int = 0,
    validation_mode: str = "full",
    attempt: int = 1  # 🆕 v35.1: numer próby
) -> ValidationResult:
    """
    Główna funkcja walidacji SEO. 
    
    🆕 v35.1 OPTIMIZED:
    - Elastyczne progi w zależności od attempt
    - JITTER wyłączone (ValidationConfig.JITTER_ENABLED)
    - Zmniejszone penalizacje
    """
    config = ValidationConfig.get_relaxed_config(attempt)
    issues: List[ValidationIssue] = []
    keywords_state = keywords_state or {}
    required_ngrams = required_ngrams or []
    text_lemmas = lemmatize_text(text)
    word_count = count_words(text)
    
    # 1. Metryki
    burstiness = calculate_burstiness(text)
    transition_data = calculate_transition_ratio(text)
    density = calculate_density(text, keywords_state)
    
    # 🆕 v35.1: Walidacja burstiness z attempt
    burstiness_details = validate_burstiness(burstiness, issues, attempt)
    
    if transition_data["ratio"] < config.TRANSITION_RATIO_MIN:
        issues.append(ValidationIssue("LOW_TRANSITION_RATIO", f"Za mało transition words: {transition_data['ratio']:.0%}", Severity.WARNING, transition_data))
    elif transition_data["ratio"] > config.TRANSITION_RATIO_MAX:
        issues.append(ValidationIssue("HIGH_TRANSITION_RATIO", f"Za dużo transition words: {transition_data['ratio']:.0%}", Severity.WARNING, transition_data))
    
    if density > config.DENSITY_MAX:
        issues.append(ValidationIssue("HIGH_DENSITY", f"Gęstość za wysoka: {density}%", Severity.WARNING, {"value": density}))
    
    # 2. Keywords
    keywords_analysis = {"main_keyword": main_keyword, "main_uses": 0, "synonym_uses": 0, "main_ratio": 1.0, "keyword_counts": {}}
    if keywords_state:
        main_uses = synonym_uses = 0
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            count = count_keyword_occurrences(text_lemmas, keyword)
            keywords_analysis["keyword_counts"][keyword] = count
            if meta.get("is_main_keyword"):
                main_uses = count
            elif meta.get("is_synonym_of_main"):
                synonym_uses += count
        keywords_analysis["main_uses"] = main_uses
        keywords_analysis["synonym_uses"] = synonym_uses
        total = main_uses + synonym_uses
        main_ratio = main_uses / total if total > 0 else 1.0
        keywords_analysis["main_ratio"] = round(main_ratio, 2)
        if total > 0 and main_ratio < config.MAIN_KEYWORD_RATIO_MIN:
            issues.append(ValidationIssue("LOW_MAIN_KEYWORD_RATIO", f"Fraza główna ma tylko {main_ratio:.0%} użyć", Severity.ERROR, {"ratio": main_ratio}))
    
    # 3. Struktura
    structure_analysis = {"word_count": word_count, "h2_count": 0, "h3_count": 0, "list_count": 0, "h3_sections": []}
    h2_titles = extract_h2_titles(text)
    structure_analysis["h2_count"] = len(h2_titles)
    structure_analysis["h2_titles"] = h2_titles
    
    if main_keyword and h2_titles:
        main_lower = main_keyword.lower()
        h2_with_main = sum(1 for h2 in h2_titles if main_lower in h2.lower())
        if h2_with_main > config.H2_MAIN_KEYWORD_MAX:
            issues.append(ValidationIssue("OVEROPTIMIZED_H2_KEYWORDS", f"Za dużo H2 z frazą główną: {h2_with_main}", Severity.WARNING, {"h2_with_main": h2_with_main}))
    
    h3_sections = extract_h3_sections(text)
    structure_analysis["h3_count"] = len(h3_sections)
    structure_analysis["h3_sections"] = h3_sections
    
    for section in h3_sections:
        if section["word_count"] < config.H3_MIN_WORDS:
            # 🆕 v35.1: INFO zamiast WARNING dla krótkich sekcji
            issues.append(ValidationIssue("SHORT_H3_SECTION", f"H3 '{section['title']}' krótki: {section['word_count']} słów", Severity.INFO, section))
    
    current_lists = count_lists(text)
    total_lists = existing_lists_count + current_lists
    structure_analysis["list_count"] = current_lists
    if total_lists > config.LIST_MAX:
        issues.append(ValidationIssue("TOO_MANY_LISTS", f"Za dużo list: {total_lists}", Severity.WARNING, {"total": total_lists}))
    
    # 4. Intro
    if is_intro_batch:
        intro_text = extract_intro(text)
        intro_words = count_words(intro_text)
        structure_analysis["intro_words"] = intro_words
        if intro_words < config.INTRO_MIN_WORDS:
            issues.append(ValidationIssue("SHORT_INTRO", f"Intro za krótkie: {intro_words} słów", Severity.WARNING, {"word_count": intro_words}))
        elif intro_words > config.INTRO_MAX_WORDS:
            issues.append(ValidationIssue("LONG_INTRO", f"Intro za długie: {intro_words} słów", Severity.WARNING, {"word_count": intro_words}))
        intro_lower = intro_text.lower()
        for banned in ValidationConfig.BANNED_INTRO_OPENERS:
            if intro_lower.startswith(banned):
                issues.append(ValidationIssue("BANNED_INTRO_OPENER", f"Intro zaczyna się od: '{banned}'", Severity.WARNING, {"banned": banned}))
                break
        if main_keyword:
            first_sentence = intro_text.split('.')[0] if intro_text else ""
            if main_keyword.lower() not in first_sentence.lower():
                issues.append(ValidationIssue("MAIN_KEYWORD_NOT_IN_FIRST_SENTENCE", f"Fraza główna nie w pierwszym zdaniu", Severity.INFO, {}))  # 🔧 INFO zamiast WARNING
    
    # 5. N-gramy
    if required_ngrams:
        text_lower = text.lower()
        used_ngrams = [ng for ng in required_ngrams if ng.lower() in text_lower]
        missing_ngrams = [ng for ng in required_ngrams if ng.lower() not in text_lower]
        coverage = len(used_ngrams) / len(required_ngrams) if required_ngrams else 1.0
        structure_analysis["ngram_coverage"] = {"coverage": round(coverage, 2), "used": used_ngrams, "missing": missing_ngrams}
        if coverage < config.NGRAM_COVERAGE_MIN:
            issues.append(ValidationIssue("LOW_NGRAM_COVERAGE", f"Niskie pokrycie n-gramów: {coverage:.0%}", Severity.WARNING, {"coverage": coverage}))
    
    # 🆕 v36.5: Template patterns check
    template_issues, template_stats = check_template_patterns(text)
    issues.extend(template_issues)
    structure_analysis["template_patterns"] = template_stats
    
    # 6. Score - 🔧 v35.1: ZMNIEJSZONE PENALIZACJE
    score = 100
    for issue in issues:
        if issue.severity == Severity.CRITICAL:
            score -= config.PENALTY_CRITICAL
        elif issue.severity == Severity.ERROR:
            score -= config.PENALTY_ERROR
        elif issue.severity == Severity.WARNING:
            score -= config.PENALTY_WARNING
        elif issue.severity == Severity.INFO:
            score -= config.PENALTY_INFO
    score = max(0, min(100, score))
    
    has_critical = any(i.severity == Severity.CRITICAL for i in issues)
    has_errors = any(i.severity == Severity.ERROR for i in issues)
    is_valid = not (has_critical or has_errors)
    
    return ValidationResult(
        is_valid=is_valid, 
        score=score, 
        issues=issues,
        metrics={
            "burstiness": burstiness_details,
            "transition_ratio": {"value": transition_data["ratio"]}, 
            "density": {"value": density}, 
            "word_count": word_count,
            "attempt": attempt  # 🆕 Info o próbie
        },
        keywords_analysis=keywords_analysis, 
        structure_analysis=structure_analysis
    )


# ================================================================
# 🔧 QUICK VALIDATE
# ================================================================
def quick_validate(text: str, keywords_state: Dict = None, attempt: int = 1) -> Dict:
    result = validate_content(text, keywords_state, validation_mode="preview", attempt=attempt)
    return {
        "status": "OK" if result.is_valid else "CRITICAL" if result.get_critical() else "WARN",
        "score": result.score, 
        "critical": len(result.get_critical()),
        "errors": len(result.get_errors()), 
        "warnings": len(result.get_warnings()), 
        "metrics": result.metrics
    }


def full_validate(text: str, keywords_state: Dict, main_keyword: str, ngrams: List[str] = None, attempt: int = 1) -> Dict:
    result = validate_content(text=text, keywords_state=keywords_state, main_keyword=main_keyword, required_ngrams=ngrams, validation_mode="final", attempt=attempt)
    return result.to_dict()


# ================================================================
# 📚 SOURCE EFFORT PATTERNS (bez zmian)
# ================================================================
SOURCE_EFFORT_PATTERNS = {
    "COURT_RULING": {"weight": 2.0, "patterns": [r'(?:wyrok|uchwała)\s+(?:SN|SA|TK|NSA)', r'sygn\.\s*akt\s*[A-Z]{1,4}\s*\d+/\d+']},
    "LEGAL_ACT": {"weight": 1.5, "patterns": [r'art\.\s*\d+', r'§\s*\d+', r'Dz\.?\s*U\.?\s*\d{4}', r'RODO|GDPR']},
    "SCIENTIFIC": {"weight": 1.8, "patterns": [r'et\s+al\.?', r'p\s*[<>=]\s*0[,\.]\d+', r'n\s*=\s*\d{2,}']},
    "OFFICIAL_DATA": {"weight": 1.4, "patterns": [r'(?:dane|raport)\s+(?:GUS|Eurostat|OECD|WHO|NBP)']},
    "EXPERT": {"weight": 1.3, "patterns": [r'(?:prof\.|dr\.?)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ]']}
}

NEGATIVE_PATTERNS = [(r'niektórzy\s+(?:eksperci|badacze)', -0.3), (r'powszechnie\s+wiadomo', -0.3), (r'(?:ostatnio|niedawno)\b', -0.2)]

GENERIC_PHRASES = ["według ekspertów", "badania pokazują", "statystyki wskazują", "zgodnie z prawem", "algorytm google", "w ostatnich latach"]
ZERO_VALUE_PHRASES = ["warto wiedzieć", "nie ulega wątpliwości", "jak wszyscy wiedzą", "w dzisiejszych czasach", "coraz więcej osób"]


# ================================================================
# 🔍 SEMANTIC VALIDATION (bez istotnych zmian)
# ================================================================
def calculate_entity_density(text: str, entities: List[Dict] = None) -> Dict[str, Any]:
    words = text.split()
    word_count = len(words)
    if word_count < 50:
        return {"status": "TOO_SHORT", "density": 0, "word_count": word_count}
    entity_count = len(entities) if entities else 0
    density = (entity_count / word_count) * 100
    hard_count = sum(1 for e in (entities or []) if e.get("type") in HARD_ENTITY_TYPES)
    hard_ratio = hard_count / entity_count if entity_count > 0 else 0
    text_lower = text.lower()
    generics = [p for p in GENERIC_PHRASES if p in text_lower]
    zero_value = [p for p in ZERO_VALUE_PHRASES if p in text_lower]
    config = SemanticConfig()
    status = "GOOD" if density >= config.ENTITY_DENSITY_MIN else "NEEDS_IMPROVEMENT"
    if density > config.ENTITY_DENSITY_MAX:
        status = "OVERSTUFFED"
    return {
        "status": status, 
        "density": round(density, 2),
        "density_per_100": round(density, 2), 
        "entity_count": entity_count, 
        "word_count": word_count,
        "hard_entity_ratio": round(hard_ratio, 2), 
        "generics_found": generics[:5], 
        "zero_value_found": zero_value[:3], 
        "action_required": status != "GOOD" or len(generics) > 2
    }


def validate_semantic_enhancement(content: str, s1_data: Dict = None, detected_entities: List[Dict] = None) -> Dict[str, Any]:
    """Główna funkcja walidacji semantycznej."""
    s1_data = s1_data or {}
    density = calculate_entity_density(content, detected_entities)
    
    scores = {
        "entity_density": 0.7 if density["status"] == "GOOD" else 0.3,
    }
    final_score = scores["entity_density"]
    
    issues = []
    if density.get("action_required"):
        issues.append({"code": "LOW_ENTITY_DENSITY", "severity": "WARNING"})
    
    status = "APPROVED" if len(issues) <= 1 else "WARN" if len(issues) <= 3 else "REJECTED"
    
    return {
        "status": status, 
        "semantic_score": round(final_score, 2), 
        "component_scores": scores, 
        "analyses": {"entity_density": density}, 
        "issues": issues, 
        "quick_wins": []
    }


def full_validate_complete(text: str, keywords_state: Dict, main_keyword: str, ngrams: List[str] = None, s1_data: Dict = None, detected_entities: List[Dict] = None, attempt: int = 1) -> Dict:
    """Kompletna walidacja: SEO + Semantic Enhancement."""
    result = validate_content(text=text, keywords_state=keywords_state, main_keyword=main_keyword, required_ngrams=ngrams, validation_mode="final", attempt=attempt)
    result_dict = result.to_dict()
    semantic = validate_semantic_enhancement(text, s1_data, detected_entities)
    
    base_score = result_dict["score"]
    semantic_score = semantic.get("semantic_score", 0.5) * 100
    result_dict["final_score"] = round(base_score * 0.6 + semantic_score * 0.4, 1)
    
    if result_dict["final_score"] >= 75:
        result_dict["final_status"] = "EXCELLENT"
    elif result_dict["final_score"] >= 60:
        result_dict["final_status"] = "GOOD"
    elif result_dict["final_score"] >= 45:
        result_dict["final_status"] = "NEEDS_IMPROVEMENT"
    else:
        result_dict["final_status"] = "POOR"
    
    return result_dict


# ================================================================
# 🆕 v35.1: HELPER - Czy auto-approve?
# ================================================================
def should_auto_approve(attempt: int, score: int, has_critical: bool) -> bool:
    """
    🆕 v35.1: Decyduje czy auto-approve na podstawie próby i score.
    
    Returns True jeśli:
    - attempt >= AUTO_APPROVE_THRESHOLD (2)
    - LUB score >= 70 i brak CRITICAL
    """
    if has_critical:
        return False
    
    if attempt >= ValidationConfig.AUTO_APPROVE_THRESHOLD:
        return True
    
    if score >= 70:
        return True
    
    return False


# ================================================================
# 🆕 v36.9: PRIORITIZE ISSUES - Ogranicza liczbę błędów do naprawy
# ================================================================
ISSUE_PRIORITY = {
    # CRITICAL - blokujące (priorytet 1-10)
    "CRITICAL_LOW_BURSTINESS": 1,
    "CRITICAL_HIGH_BURSTINESS": 2,
    "HARD_EXCEEDED": 3,
    
    # HIGH - ważne (priorytet 11-20)
    "SOFT_EXCEEDED": 11,
    "SEVERE_STUFFING": 12,
    "LOW_BURSTINESS": 13,
    "HIGH_BURSTINESS": 14,
    
    # MEDIUM - wpływa na score (priorytet 21-30)
    "HIGH_DENSITY": 21,
    "LOW_MAIN_KEYWORD_RATIO": 22,
    "LOW_NGRAM_COVERAGE": 23,
    
    # LOW - informacyjne (priorytet 31+)
    "SHORT_H3_SECTION": 31,
    "TOO_MANY_LISTS": 32,
    "SHORT_INTRO": 33,
    "LONG_INTRO": 34,
    "LOW_TRANSITION_RATIO": 35,
    "HIGH_TRANSITION_RATIO": 36,
    "BANNED_INTRO_OPENER": 37,
    "MAIN_KEYWORD_NOT_IN_FIRST_SENTENCE": 38,
    "OVEROPTIMIZED_H2_KEYWORDS": 39,
}


def prioritize_issues(issues: List[ValidationIssue], max_issues: int = 2) -> List[ValidationIssue]:
    """
    🆕 v36.9: Zwraca tylko najważniejsze problemy do naprawy.
    
    Rozwiązuje problem "over-optimization" - AI dostaje max 2 problemy
    zamiast 10, więc wie co naprawić najpierw.
    
    Args:
        issues: Lista wszystkich problemów
        max_issues: Maksymalna liczba problemów do zwrócenia (default: 2)
        
    Returns:
        Lista najważniejszych problemów (max `max_issues`)
    """
    if not issues:
        return []
    
    def get_priority(issue: ValidationIssue) -> int:
        # Priorytet z mapy, lub 50 dla nieznanych (nisko)
        return ISSUE_PRIORITY.get(issue.type, 50)
    
    # Sortuj po priorytecie (niższy = ważniejszy)
    sorted_issues = sorted(issues, key=get_priority)
    
    # Zwróć max `max_issues` problemów
    return sorted_issues[:max_issues]


def get_actionable_feedback(
    issues: List[ValidationIssue], 
    attempt: int = 1,
    previous_issues: List[Dict] = None
) -> Dict[str, Any]:
    """
    🆕 v36.9: Generuje konkretne instrukcje naprawy dla AI.
    
    Zamiast: "Burstiness za niski: 2.1"
    Zwraca: "Dodaj 2-3 KRÓTKIE zdania (5-8 słów) na początku akapitów"
    
    Args:
        issues: Aktualne problemy
        attempt: Numer próby
        previous_issues: Problemy z poprzedniej próby (opcjonalnie)
        
    Returns:
        Dict z priorytetowymi instrukcjami
    """
    prioritized = prioritize_issues(issues, max_issues=2)
    
    instructions = []
    for issue in prioritized:
        instruction = _get_fix_instruction(issue, attempt)
        
        # Sprawdź czy problem był w poprzedniej próbie
        if previous_issues:
            prev = next((p for p in previous_issues if p.get("type") == issue.type), None)
            if prev:
                instruction["recurring"] = True
                instruction["previous_value"] = prev.get("value")
                instruction["hint"] = f"Problem powtarza się - spróbuj innego podejścia"
        
        instructions.append(instruction)
    
    return {
        "attempt": attempt,
        "total_issues": len(issues),
        "prioritized_count": len(prioritized),
        "instructions": instructions,
        "focus_message": _get_focus_message(prioritized)
    }


def _get_fix_instruction(issue: ValidationIssue, attempt: int) -> Dict[str, Any]:
    """Generuje konkretną instrukcję naprawy dla danego problemu."""
    
    base = {
        "type": issue.type,
        "severity": issue.severity.value,
        "message": issue.message,
        "recurring": False
    }
    
    # Konkretne instrukcje dla każdego typu
    if issue.type == "CRITICAL_LOW_BURSTINESS" or issue.type == "LOW_BURSTINESS":
        value = issue.details.get("value", 0)
        cv = issue.details.get("cv", value / 5)
        base["fix"] = (
            f"Tekst jest zbyt monotonny (CV={cv:.2f}). "
            f"DODAJ 3-4 KRÓTKIE zdania (5-8 słów) rozrzucone po tekście. "
            f"Przykłady: 'To ważne.', 'Warto pamiętać.', 'Oto szczegóły.'"
        )
        base["examples"] = [
            "Zmień: 'Ubezwłasnowolnienie jest procesem prawnym który wymaga spełnienia określonych przesłanek.'",
            "Na: 'Ubezwłasnowolnienie to proces prawny. Wymaga spełnienia przesłanek. Oto najważniejsze z nich.'"
        ]
        
    elif issue.type == "CRITICAL_HIGH_BURSTINESS" or issue.type == "HIGH_BURSTINESS":
        base["fix"] = (
            "Tekst jest zbyt chaotyczny - za duża różnica między zdaniami. "
            "WYRÓWNAJ długości: zamień bardzo krótkie zdania na średnie (12-18 słów)."
        )
        
    elif issue.type == "HARD_EXCEEDED" or issue.type == "SOFT_EXCEEDED":
        keyword = issue.details.get("keyword", "fraza")
        actual = issue.details.get("actual", 0)
        limit = issue.details.get("hard_max", issue.details.get("target_max", 5))
        base["fix"] = (
            f"Fraza '{keyword}' użyta {actual}× (limit: {limit}×). "
            f"ZAMIEŃ nadmiarowe wystąpienia na synonimy lub zaimki: "
            f"'ta procedura', 'ten proces', 'omawiana instytucja'."
        )
        
    elif issue.type == "HIGH_DENSITY":
        density = issue.details.get("value", 0)
        base["fix"] = (
            f"Gęstość fraz ({density:.1f}%) za wysoka. "
            f"ROZŁÓŻ frazy kluczowe bardziej równomiernie lub użyj synonimów."
        )
        
    elif issue.type == "LOW_NGRAM_COVERAGE":
        coverage = issue.details.get("coverage", 0)
        base["fix"] = (
            f"Pokrycie n-gramów ({coverage:.0%}) za niskie. "
            f"DODAJ więcej fraz tematycznych z listy wymaganych."
        )
        
    else:
        base["fix"] = f"Napraw: {issue.message}"
    
    return base


def _get_focus_message(prioritized: List[ValidationIssue]) -> str:
    """Generuje krótki komunikat co naprawić."""
    if not prioritized:
        return "✅ Brak krytycznych problemów"
    
    if len(prioritized) == 1:
        return f"🎯 SKUP SIĘ NA: {prioritized[0].type}"
    
    types = [p.type for p in prioritized]
    return f"🎯 NAPRAW NAJPIERW: {types[0]}, potem: {types[1]}"


# ================================================================
# 🆕 v36.9: AUTO-FIX BURSTINESS - Automatyczna naprawa monotonii
# ================================================================

def auto_fix_burstiness(text: str, target_cv: float = 0.44) -> Tuple[str, List[str], Dict]:
    """
    🆕 v36.9: Automatycznie naprawia zbyt niski burstiness.
    
    Strategia:
    1. Dzieli najdłuższe zdania (>25 słów) w miejscach naturalnych przecięć
    2. NIE łączy krótkich zdań (to psuje styl)
    
    Args:
        text: Tekst do naprawy
        target_cv: Docelowy CV (default: 0.44 = burstiness 2.2)
        
    Returns:
        Tuple[fixed_text, applied_fixes, stats]
        
    UWAGA: Ta funkcja jest konserwatywna - lepiej nie naprawić niż zepsuć.
    """
    # Oblicz aktualny burstiness
    current_burstiness = calculate_burstiness(text)
    current_cv = current_burstiness / 5.0
    
    if current_cv >= target_cv:
        return text, [], {"status": "OK", "cv_before": current_cv, "cv_after": current_cv}
    
    # Podziel na zdania
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    if len(sentences) < 3:
        return text, [], {"status": "TOO_SHORT", "cv_before": current_cv, "cv_after": current_cv}
    
    # Znajdź zdania do podziału (>25 słów)
    applied_fixes = []
    modified_sentences = []
    
    # Naturalne punkty podziału
    split_patterns = [
        (r',\s*które?\s+', '. '),           # ", które " -> ". "
        (r',\s*co\s+', '. To '),             # ", co " -> ". To "
        (r',\s*ponieważ\s+', '. '),          # ", ponieważ " -> ". "
        (r',\s*gdyż\s+', '. '),              # ", gdyż " -> ". "
        (r',\s*jednak\s+', '. Jednak '),     # ", jednak " -> ". Jednak "
        (r',\s*natomiast\s+', '. '),         # ", natomiast " -> ". "
        (r',\s*a\s+także\s+', '. Ponadto '), # ", a także " -> ". Ponadto "
        (r'\s+–\s+', '. '),                  # " – " -> ". "
    ]
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if word_count > 25:
            # Próbuj podzielić
            modified = sentence
            was_split = False
            
            for pattern, replacement in split_patterns:
                if re.search(pattern, modified, re.IGNORECASE):
                    # Podziel tylko raz
                    parts = re.split(pattern, modified, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2 and len(parts[0].split()) >= 8 and len(parts[1].split()) >= 5:
                        # Obie części są sensownej długości
                        new_sentence = parts[0].rstrip() + replacement + parts[1].lstrip()
                        # Upewnij się że druga część zaczyna się wielką literą
                        new_sentence = _capitalize_after_period(new_sentence)
                        
                        applied_fixes.append(f"Podzielono ({word_count} słów): '{sentence[:50]}...'")
                        modified = new_sentence
                        was_split = True
                        break
            
            modified_sentences.append(modified)
        else:
            modified_sentences.append(sentence)
    
    if not applied_fixes:
        return text, [], {"status": "NO_CHANGES", "cv_before": current_cv, "cv_after": current_cv}
    
    # Złóż tekst z powrotem
    fixed_text = ' '.join(modified_sentences)
    
    # Sprawdź nowy burstiness
    new_burstiness = calculate_burstiness(fixed_text)
    new_cv = new_burstiness / 5.0
    
    # Sprawdź czy się poprawiło (a nie pogorszyło)
    if new_cv < current_cv:
        # Pogorszyło się - wróć do oryginału
        return text, [], {
            "status": "REVERTED", 
            "cv_before": current_cv, 
            "cv_after": current_cv,
            "reason": "Zmiana pogorszyła burstiness"
        }
    
    return fixed_text, applied_fixes, {
        "status": "FIXED",
        "cv_before": current_cv,
        "cv_after": new_cv,
        "improvement": new_cv - current_cv,
        "fixes_count": len(applied_fixes)
    }


def _capitalize_after_period(text: str) -> str:
    """Upewnia się że po kropce jest wielka litera."""
    result = []
    capitalize_next = False
    
    for i, char in enumerate(text):
        if capitalize_next and char.isalpha():
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char)
            if char in '.!?':
                capitalize_next = True
    
    return ''.join(result)
