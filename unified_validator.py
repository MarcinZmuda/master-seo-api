"""
===============================================================================
🔍 UNIFIED VALIDATOR v35.0 (BRAJEN)
===============================================================================
v35.0 ZMIANY (zgodnie z dokumentem f.pdf - badania NKJP):
- Progi burstiness zgodne z CV: CRITICAL < 1.5 (CV 0.3), OK >= 2.5 (CV 0.5)
- Dodano wartość CV w komunikatach diagnostycznych
- Nowy poziom SUBOPTIMAL dla burstiness w strefie 2.0-2.5

v32.0 ZMIANY:
- Dwupoziomowe progi burstiness (CRITICAL/WARNING)
- Severity.CRITICAL dodane
===============================================================================
"""

import re
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

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
# 📊 STAŁE KONFIGURACYJNE - v35.0
# 🔧 Progi zgodne z dokumentem f.pdf (badania NKJP)
# ================================================================
class ValidationConfig:
    """
    Centralna konfiguracja wszystkich progów walidacji.
    
    🆕 v35.0 - PROGI ZGODNE Z BADANIAMI NKJP (f.pdf):
    - Burstiness: CV > 0.5 = ludzki, CV < 0.3 = AI
    - Formuła: burstiness = CV * 5
    """
    
    # ================================================================
    # BURSTINESS - zgodnie z dokumentem f.pdf
    # CV (współczynnik zmienności) = σ / μ
    # System używa: burstiness = CV * 5
    # ================================================================
    BURSTINESS_CRITICAL_LOW = 1.5   # CV 0.3 - próg AI (było 2.0)
    BURSTINESS_WARNING_LOW = 2.0    # CV 0.4 - strefa neutralna (było 2.8)
    BURSTINESS_OPTIMAL_MIN = 2.5    # CV 0.5 - próg ludzkiego (było 2.8/3.2)
    BURSTINESS_OPTIMAL_MAX = 4.0    # CV 0.8 - górna granica OK (było 4.2/3.8)
    BURSTINESS_WARNING_HIGH = 4.5   # CV 0.9 (było 4.2)
    BURSTINESS_CRITICAL_HIGH = 5.0  # CV 1.0 - tekst chaotyczny (było 4.8)
    
    # Stare aliasy dla kompatybilności
    BURSTINESS_MIN = BURSTINESS_WARNING_LOW
    BURSTINESS_MAX = BURSTINESS_WARNING_HIGH
    BURSTINESS_OPTIMAL = 3.25       # środek przedziału 2.5-4.0
    
    TRANSITION_RATIO_MIN = 0.25
    TRANSITION_RATIO_MAX = 0.50
    DENSITY_MAX = 3.0
    DENSITY_WARNING = 2.5
    H3_MIN_WORDS = 80
    LIST_MIN = 1
    LIST_MAX = 2
    INTRO_MIN_WORDS = 40
    INTRO_MAX_WORDS = 60
    MAIN_KEYWORD_RATIO_MIN = 0.30
    H2_MAIN_KEYWORD_MAX = 1
    NGRAM_COVERAGE_MIN = 0.60
    
    TRANSITION_WORDS = [
        "również", "także", "ponadto", "dodatkowo", "co więcej",
        "oprócz tego", "poza tym", "jednak", "jednakże", "natomiast",
        "ale", "z drugiej strony", "mimo to", "niemniej", "dlatego",
        "w związku z tym", "w rezultacie", "ponieważ", "zatem", "więc",
        "na przykład", "przykładowo", "między innymi", "m.in.", "np.",
        "po pierwsze", "po drugie", "następnie", "potem", "na koniec"
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


# ================================================================
# 📊 SEMANTIC CONFIG (v31.0)
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
# 📋 STRUKTURY DANYCH - v32.0 z CRITICAL
# ================================================================
class Severity(Enum):
    CRITICAL = "CRITICAL"  # 🆕 v32.0
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
def calculate_burstiness(text: str) -> float:
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


def validate_burstiness(burstiness: float, issues: List[ValidationIssue]) -> Dict[str, Any]:
    """
    🆕 v35.0: Walidacja burstiness zgodna z dokumentem f.pdf (NKJP).
    
    Progi CV (współczynnik zmienności = burstiness / 5):
    - CV < 0.3 = CRITICAL (silny sygnał AI)
    - CV 0.3-0.4 = WARNING (strefa neutralna)
    - CV 0.5-0.8 = OK (ludzki tekst)
    - CV > 0.9 = WARNING (tekst chaotyczny)
    - CV > 1.0 = CRITICAL
    """
    cv_value = burstiness / 5  # Konwersja na współczynnik zmienności
    details = {
        "value": burstiness, 
        "cv": round(cv_value, 2),
        "status": "OK", 
        "level": None
    }
    
    if burstiness < ValidationConfig.BURSTINESS_CRITICAL_LOW:
        details["status"] = "CRITICAL"
        details["level"] = "critical_low"
        issues.append(ValidationIssue(
            "CRITICAL_LOW_BURSTINESS", 
            f"Burstiness KRYTYCZNIE niski: {burstiness:.2f} (CV {cv_value:.2f} < 0.3) - silny sygnał AI!", 
            Severity.CRITICAL,
            {"value": burstiness, "cv": cv_value, "threshold": ValidationConfig.BURSTINESS_CRITICAL_LOW}
        ))
    elif burstiness < ValidationConfig.BURSTINESS_WARNING_LOW:
        details["status"] = "WARNING"
        details["level"] = "warning_low"
        issues.append(ValidationIssue(
            "LOW_BURSTINESS", 
            f"Burstiness za niski: {burstiness:.2f} (CV {cv_value:.2f} < 0.4) - strefa neutralna/podejrzana", 
            Severity.WARNING, 
            {"value": burstiness, "cv": cv_value, "threshold": ValidationConfig.BURSTINESS_WARNING_LOW}
        ))
    elif burstiness < ValidationConfig.BURSTINESS_OPTIMAL_MIN:
        details["status"] = "INFO"
        details["level"] = "below_optimal"
        issues.append(ValidationIssue(
            "SUBOPTIMAL_BURSTINESS", 
            f"Burstiness poniżej optymalnego: {burstiness:.2f} (CV {cv_value:.2f} < 0.5) - blisko progu", 
            Severity.WARNING, 
            {"value": burstiness, "cv": cv_value, "threshold": ValidationConfig.BURSTINESS_OPTIMAL_MIN}
        ))
    elif burstiness > ValidationConfig.BURSTINESS_CRITICAL_HIGH:
        details["status"] = "CRITICAL"
        details["level"] = "critical_high"
        issues.append(ValidationIssue(
            "CRITICAL_HIGH_BURSTINESS", 
            f"Burstiness KRYTYCZNIE wysoki: {burstiness:.2f} (CV {cv_value:.2f} > 1.0) - tekst chaotyczny!", 
            Severity.CRITICAL, 
            {"value": burstiness, "cv": cv_value, "threshold": ValidationConfig.BURSTINESS_CRITICAL_HIGH}
        ))
    elif burstiness > ValidationConfig.BURSTINESS_WARNING_HIGH:
        details["status"] = "WARNING"
        details["level"] = "warning_high"
        issues.append(ValidationIssue(
            "HIGH_BURSTINESS", 
            f"Burstiness za wysoki: {burstiness:.2f} (CV {cv_value:.2f} > 0.9) - zdania zbyt zróżnicowane", 
            Severity.WARNING, 
            {"value": burstiness, "cv": cv_value, "threshold": ValidationConfig.BURSTINESS_WARNING_HIGH}
        ))
    else:
        if ValidationConfig.BURSTINESS_OPTIMAL_MIN <= burstiness <= ValidationConfig.BURSTINESS_OPTIMAL_MAX:
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
# 🎯 GŁÓWNA FUNKCJA WALIDACJI - v32.0
# ================================================================
def validate_content(
    text: str,
    keywords_state: Dict = None,
    main_keyword: str = None,
    required_ngrams: List[str] = None,
    is_intro_batch: bool = False,
    existing_lists_count: int = 0,
    validation_mode: str = "full"
) -> ValidationResult:
    """Główna funkcja walidacji SEO. v32.0: Dwupoziomowe progi burstiness."""
    issues: List[ValidationIssue] = []
    keywords_state = keywords_state or {}
    required_ngrams = required_ngrams or []
    text_lemmas = lemmatize_text(text)
    word_count = count_words(text)
    
    # 1. Metryki
    burstiness = calculate_burstiness(text)
    transition_data = calculate_transition_ratio(text)
    density = calculate_density(text, keywords_state)
    
    # 🆕 v32.0: Dwupoziomowa walidacja burstiness
    burstiness_details = validate_burstiness(burstiness, issues)
    
    if transition_data["ratio"] < ValidationConfig.TRANSITION_RATIO_MIN:
        issues.append(ValidationIssue("LOW_TRANSITION_RATIO", f"Za mało transition words: {transition_data['ratio']:.0%}", Severity.WARNING, transition_data))
    elif transition_data["ratio"] > ValidationConfig.TRANSITION_RATIO_MAX:
        issues.append(ValidationIssue("HIGH_TRANSITION_RATIO", f"Za dużo transition words: {transition_data['ratio']:.0%}", Severity.WARNING, transition_data))
    
    if density > ValidationConfig.DENSITY_MAX:
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
        if total > 0 and main_ratio < ValidationConfig.MAIN_KEYWORD_RATIO_MIN:
            issues.append(ValidationIssue("LOW_MAIN_KEYWORD_RATIO", f"Fraza główna ma tylko {main_ratio:.0%} użyć", Severity.ERROR, {"ratio": main_ratio}))
    
    # 3. Struktura
    structure_analysis = {"word_count": word_count, "h2_count": 0, "h3_count": 0, "list_count": 0, "h3_sections": []}
    h2_titles = extract_h2_titles(text)
    structure_analysis["h2_count"] = len(h2_titles)
    structure_analysis["h2_titles"] = h2_titles
    
    if main_keyword and h2_titles:
        main_lower = main_keyword.lower()
        h2_with_main = sum(1 for h2 in h2_titles if main_lower in h2.lower())
        if h2_with_main > ValidationConfig.H2_MAIN_KEYWORD_MAX:
            issues.append(ValidationIssue("OVEROPTIMIZED_H2_KEYWORDS", f"Za dużo H2 z frazą główną: {h2_with_main}", Severity.WARNING, {"h2_with_main": h2_with_main}))
    
    h3_sections = extract_h3_sections(text)
    structure_analysis["h3_count"] = len(h3_sections)
    structure_analysis["h3_sections"] = h3_sections
    
    for section in h3_sections:
        if section["word_count"] < ValidationConfig.H3_MIN_WORDS:
            issues.append(ValidationIssue("SHORT_H3_SECTION", f"H3 '{section['title']}' za krótki: {section['word_count']} słów", Severity.WARNING, section))
    
    current_lists = count_lists(text)
    total_lists = existing_lists_count + current_lists
    structure_analysis["list_count"] = current_lists
    if total_lists > ValidationConfig.LIST_MAX:
        issues.append(ValidationIssue("TOO_MANY_LISTS", f"Za dużo list: {total_lists}", Severity.WARNING, {"total": total_lists}))
    
    # 4. Intro
    if is_intro_batch:
        intro_text = extract_intro(text)
        intro_words = count_words(intro_text)
        structure_analysis["intro_words"] = intro_words
        if intro_words < ValidationConfig.INTRO_MIN_WORDS:
            issues.append(ValidationIssue("SHORT_INTRO", f"Intro za krótkie: {intro_words} słów", Severity.WARNING, {"word_count": intro_words}))
        elif intro_words > ValidationConfig.INTRO_MAX_WORDS:
            issues.append(ValidationIssue("LONG_INTRO", f"Intro za długie: {intro_words} słów", Severity.WARNING, {"word_count": intro_words}))
        intro_lower = intro_text.lower()
        for banned in ValidationConfig.BANNED_INTRO_OPENERS:
            if intro_lower.startswith(banned):
                issues.append(ValidationIssue("BANNED_INTRO_OPENER", f"Intro zaczyna się od: '{banned}'", Severity.WARNING, {"banned": banned}))
                break
        if main_keyword:
            first_sentence = intro_text.split('.')[0] if intro_text else ""
            if main_keyword.lower() not in first_sentence.lower():
                issues.append(ValidationIssue("MAIN_KEYWORD_NOT_IN_FIRST_SENTENCE", f"Fraza główna nie w pierwszym zdaniu", Severity.WARNING, {}))
    
    # 5. N-gramy
    if required_ngrams:
        text_lower = text.lower()
        used_ngrams = [ng for ng in required_ngrams if ng.lower() in text_lower]
        missing_ngrams = [ng for ng in required_ngrams if ng.lower() not in text_lower]
        coverage = len(used_ngrams) / len(required_ngrams) if required_ngrams else 1.0
        structure_analysis["ngram_coverage"] = {"coverage": round(coverage, 2), "used": used_ngrams, "missing": missing_ngrams}
        if coverage < ValidationConfig.NGRAM_COVERAGE_MIN:
            issues.append(ValidationIssue("LOW_NGRAM_COVERAGE", f"Niskie pokrycie n-gramów: {coverage:.0%}", Severity.WARNING, {"coverage": coverage}))
    
    # 6. Score - v32.0: CRITICAL ma większy wpływ
    score = 100
    for issue in issues:
        if issue.severity == Severity.CRITICAL:
            score -= 25
        elif issue.severity == Severity.ERROR:
            score -= 15
        elif issue.severity == Severity.WARNING:
            score -= 5
        elif issue.severity == Severity.INFO:
            score -= 1
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
            "word_count": word_count
        },
        keywords_analysis=keywords_analysis, 
        structure_analysis=structure_analysis
    )


# ================================================================
# 🔧 QUICK VALIDATE
# ================================================================
def quick_validate(text: str, keywords_state: Dict = None) -> Dict:
    result = validate_content(text, keywords_state, validation_mode="preview")
    return {
        "status": "OK" if result.is_valid else "CRITICAL" if result.get_critical() else "WARN",
        "score": result.score, 
        "critical": len(result.get_critical()),
        "errors": len(result.get_errors()), 
        "warnings": len(result.get_warnings()), 
        "metrics": result.metrics
    }


def full_validate(text: str, keywords_state: Dict, main_keyword: str, ngrams: List[str] = None) -> Dict:
    result = validate_content(text=text, keywords_state=keywords_state, main_keyword=main_keyword, required_ngrams=ngrams, validation_mode="final")
    return result.to_dict()


# ================================================================
# 📚 SOURCE EFFORT PATTERNS (v31.0)
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
# 🔍 SEMANTIC VALIDATION (v31.0)
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


def calculate_topic_completeness(content: str, s1_topics: List[Dict]) -> Dict[str, Any]:
    if not s1_topics:
        return {"status": "NO_DATA", "score": 0.5}
    content_lower = content.lower()
    must_high = [t for t in s1_topics if t.get("priority") in ["MUST", "HIGH"]]
    covered, missing = [], []
    for topic in must_high:
        subtopic = topic.get("subtopic", "").lower()
        words = subtopic.split()
        matches = sum(1 for w in words if w in content_lower)
        if matches / len(words) >= 0.5 if words else False:
            covered.append(topic)
        else:
            missing.append({"topic": subtopic, "priority": topic.get("priority"), "sample_h2": topic.get("sample_h2", "")})
    score = len(covered) / len(must_high) if must_high else 1.0
    config = SemanticConfig()
    return {"status": "GOOD" if score >= config.TOPIC_COMPLETENESS_MIN else "NEEDS_IMPROVEMENT", "score": round(score, 2), "covered_count": len(covered), "missing_count": len(missing), "missing_topics": missing[:5], "action_required": score < config.TOPIC_COMPLETENESS_MIN}


def calculate_entity_gap(content: str, s1_entities: List[Dict]) -> Dict[str, Any]:
    if not s1_entities:
        return {"status": "NO_DATA", "score": 0.5}
    content_lower = content.lower()
    important = [e for e in s1_entities if e.get("importance", 0) >= 0.3]
    found, missing = [], []
    for entity in important:
        text = entity.get("text", "").lower()
        importance = entity.get("importance", 0.5)
        sources = entity.get("sources_count", 1)
        if text in content_lower:
            found.append(entity)
        else:
            priority = "CRITICAL" if importance >= 0.7 and sources >= 4 else "HIGH" if importance >= 0.5 and sources >= 2 else "MEDIUM"
            missing.append({"entity": entity.get("text"), "type": entity.get("type"), "priority": priority, "importance": importance})
    missing.sort(key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["priority"], 3))
    score = len(found) / len(important) if important else 1.0
    config = SemanticConfig()
    return {"status": "GOOD" if score >= config.ENTITY_GAP_MIN else "WEAK", "score": round(score, 2), "found_count": len(found), "missing_count": len(missing), "critical_missing": [m for m in missing if m["priority"] == "CRITICAL"][:5], "high_missing": [m for m in missing if m["priority"] == "HIGH"][:5], "action_required": score < config.ENTITY_GAP_MIN}


def calculate_source_effort(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count < 100:
        return {"status": "TOO_SHORT", "score": 0}
    signals_found = defaultdict(list)
    total_weight = 0
    for category, data in SOURCE_EFFORT_PATTERNS.items():
        for pattern in data["patterns"]:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                signals_found[category].extend(matches[:3])
                total_weight += data["weight"] * len(matches)
    penalty = 0
    negatives = []
    for pattern, pen in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            penalty += pen
            negatives.append(pattern)
    normalized = (total_weight / (word_count / 500)) if word_count > 0 else 0
    score = max(0, min(1.0, (normalized / 10) + penalty + min(0.15, len(signals_found) * 0.03)))
    config = SemanticConfig()
    return {"status": "GOOD" if score >= config.SOURCE_EFFORT_MIN else "NEEDS_IMPROVEMENT", "score": round(score, 2), "signals_found": {k: len(v) for k, v in signals_found.items()}, "negative_patterns": negatives[:3], "action_required": score < config.SOURCE_EFFORT_MIN}


def validate_semantic_enhancement(content: str, s1_data: Dict = None, detected_entities: List[Dict] = None) -> Dict[str, Any]:
    """Główna funkcja walidacji semantycznej (v31.0)."""
    s1_data = s1_data or {}
    density = calculate_entity_density(content, detected_entities)
    topics = s1_data.get("topical_coverage", [])
    completeness = calculate_topic_completeness(content, topics)
    entities = s1_data.get("entities", s1_data.get("entity_seo", {}).get("entities", []))
    gap = calculate_entity_gap(content, entities)
    effort = calculate_source_effort(content)
    
    scores = {
        "entity_density": 0.7 if density["status"] == "GOOD" else 0.3,
        "topic_completeness": completeness.get("score", 0.5),
        "entity_gap": gap.get("score", 0.5),
        "source_effort": effort.get("score", 0.5)
    }
    final_score = sum(scores.values()) / 4
    
    issues = []
    if density.get("action_required"):
        issues.append({"code": "LOW_ENTITY_DENSITY", "severity": "WARNING"})
    if completeness.get("action_required"):
        issues.append({"code": "LOW_TOPIC_COMPLETENESS", "severity": "WARNING"})
    if gap.get("action_required"):
        issues.append({"code": "HIGH_ENTITY_GAP", "severity": "WARNING"})
    if effort.get("action_required"):
        issues.append({"code": "LOW_SOURCE_EFFORT", "severity": "WARNING"})
    
    status = "APPROVED" if len(issues) <= 1 else "WARN" if len(issues) <= 3 else "REJECTED"
    
    quick_wins = []
    if density.get("generics_found"):
        quick_wins.append(f"Zamień ogólniki: {', '.join(density['generics_found'][:2])}")
    if completeness.get("missing_topics"):
        t = completeness["missing_topics"][0]
        quick_wins.append(f"Dodaj H2 o: {t.get('sample_h2', t.get('topic'))}")
    if gap.get("critical_missing"):
        e = gap["critical_missing"][0]
        quick_wins.append(f"Dodaj encję: {e['entity']} ({e['type']})")
    if effort.get("score", 1) < 0.4:
        quick_wins.append("Dodaj źródła: orzecznictwo, akty prawne lub dane")
    
    return {"status": status, "semantic_score": round(final_score, 2), "component_scores": scores, "analyses": {"entity_density": density, "topic_completeness": completeness, "entity_gap": gap, "source_effort": effort}, "issues": issues, "quick_wins": quick_wins[:5]}


def extend_validation_result(base_result: Dict, semantic_result: Dict) -> Dict:
    """Rozszerza wynik walidacji o dane semantyczne."""
    base_result["semantic_enhancement"] = {"score": semantic_result.get("semantic_score"), "status": semantic_result.get("status"), "component_scores": semantic_result.get("component_scores", {})}
    for issue in semantic_result.get("issues", []):
        base_result.setdefault("issues", []).append(issue)
    base_result["semantic_quick_wins"] = semantic_result.get("quick_wins", [])
    return base_result


# ================================================================
# 🎯 FULL VALIDATE COMPLETE (SEO + Semantic)
# ================================================================
def full_validate_complete(text: str, keywords_state: Dict, main_keyword: str, ngrams: List[str] = None, s1_data: Dict = None, detected_entities: List[Dict] = None) -> Dict:
    """Kompletna walidacja: SEO + Semantic Enhancement."""
    result = validate_content(text=text, keywords_state=keywords_state, main_keyword=main_keyword, required_ngrams=ngrams, validation_mode="final")
    result_dict = result.to_dict()
    semantic = validate_semantic_enhancement(text, s1_data, detected_entities)
    result_dict = extend_validation_result(result_dict, semantic)
    
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
