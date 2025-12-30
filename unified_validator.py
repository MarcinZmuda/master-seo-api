"""
===============================================================================
🔍 UNIFIED VALIDATOR v23.0 - Jeden moduł walidacyjny dla całego systemu
===============================================================================
Rozwiązuje PROBLEM 2: Redundantne Walidacje

Zamiast walidować w:
- preview_batch
- approve_batch  
- final_review
- pre_batch_info

Teraz JEDNA funkcja validate_content() używana wszędzie.

===============================================================================
"""

import re
import math
import spacy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# ================================================================
# 🧠 Współdzielony model spaCy
# ================================================================
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
    print("[VALIDATOR] ✅ Używam współdzielonego modelu spaCy")
except ImportError:
    # Fallback - ładuj lokalnie
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
        print("[VALIDATOR] ⚠️ Załadowano lokalny model pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")


# ================================================================
# 📊 STAŁE KONFIGURACYJNE (łatwe do zmiany w jednym miejscu)
# ================================================================
class ValidationConfig:
    """Centralna konfiguracja wszystkich progów walidacji."""
    
    # Metryki jakości tekstu
    # v25.2: Rozszerzone zakresy burstiness (bardziej elastyczne dla blogów)
    BURSTINESS_MIN = 2.5
    BURSTINESS_MAX = 4.5
    BURSTINESS_OPTIMAL = 3.5
    
    TRANSITION_RATIO_MIN = 0.25
    TRANSITION_RATIO_MAX = 0.50
    
    DENSITY_MAX = 3.0
    DENSITY_WARNING = 2.5
    
    # Struktura
    H3_MIN_WORDS = 80
    LIST_MIN = 1
    LIST_MAX = 2
    
    # Intro
    INTRO_MIN_WORDS = 40
    INTRO_MAX_WORDS = 60
    
    # Fraza główna
    MAIN_KEYWORD_RATIO_MIN = 0.30
    H2_MAIN_KEYWORD_MAX = 1  # v26.1: Max 1 H2 z frazą główną (unikamy przeoptymalizowania)
    
    # N-gramy
    NGRAM_COVERAGE_MIN = 0.60
    
    # Transition words (Polish)
    TRANSITION_WORDS = [
        "również", "także", "ponadto", "dodatkowo", "co więcej",
        "oprócz tego", "poza tym", "jednak", "jednakże", "natomiast",
        "ale", "z drugiej strony", "mimo to", "niemniej", "dlatego",
        "w związku z tym", "w rezultacie", "ponieważ", "zatem", "więc",
        "na przykład", "przykładowo", "między innymi", "m.in.", "np.",
        "po pierwsze", "po drugie", "następnie", "potem", "na koniec"
    ]
    
    # Banned openers
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
# 📋 STRUKTURY DANYCH
# ================================================================
class Severity(Enum):
    """Poziomy ważności problemów."""
    ERROR = "ERROR"      # Blokuje zatwierdzenie
    WARNING = "WARNING"  # Ostrzeżenie, można kontynuować
    INFO = "INFO"        # Informacja


@dataclass
class ValidationIssue:
    """Pojedynczy problem wykryty podczas walidacji."""
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
    """Kompletny wynik walidacji."""
    is_valid: bool
    score: int  # 0-100
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
                "errors": len([i for i in self.issues if i.severity == Severity.ERROR]),
                "warnings": len([i for i in self.issues if i.severity == Severity.WARNING]),
                "infos": len([i for i in self.issues if i.severity == Severity.INFO])
            }
        }
    
    def get_errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


# ================================================================
# 🔧 FUNKCJE POMOCNICZE
# ================================================================
def count_words(text: str) -> int:
    """Liczy słowa w tekście (bez tagów HTML)."""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return len(clean.split()) if clean else 0


def extract_intro(text: str) -> str:
    """Wyciąga intro (tekst przed pierwszym h2)."""
    # Szukaj pierwszego h2 (format marker lub HTML)
    h2_match = re.search(r'(^h2:|<h2)', text, re.MULTILINE | re.IGNORECASE)
    if h2_match:
        return text[:h2_match.start()].strip()
    return text[:500]  # Fallback - pierwsze 500 znaków


def extract_h2_titles(text: str) -> List[str]:
    """Wyciąga tytuły H2 z tekstu."""
    titles = []
    # Format marker: h2: Tytuł
    titles.extend(re.findall(r'^h2:\s*(.+)$', text, re.MULTILINE | re.IGNORECASE))
    # Format HTML: <h2>Tytuł</h2>
    titles.extend(re.findall(r'<h2[^>]*>([^<]+)</h2>', text, re.IGNORECASE))
    return [t.strip() for t in titles if t.strip()]


def extract_h3_sections(text: str) -> List[Dict[str, Any]]:
    """Wyciąga sekcje H3 z ich treścią."""
    sections = []
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    for i, match in enumerate(h3_matches):
        title = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        
        # Znajdź koniec sekcji (następny nagłówek lub koniec tekstu)
        end = len(text)
        next_header = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
        if next_header:
            end = start + next_header.start()
        
        section_text = text[start:end].strip()
        section_text = re.sub(r'<[^>]+>', '', section_text)
        word_count = len(section_text.split())
        
        sections.append({
            "title": title,
            "word_count": word_count,
            "position": i
        })
    
    return sections


def count_lists(text: str) -> int:
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
    
    # HTML lists
    html_lists = len(re.findall(r'<ul>|<ol>', text, re.IGNORECASE))
    
    return list_blocks + html_lists


def lemmatize_text(text: str) -> List[str]:
    """Zwraca listę lematów z tekstu."""
    clean = re.sub(r'<[^>]+>', ' ', text.lower())
    clean = re.sub(r'\s+', ' ', clean)
    doc = nlp(clean)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


def count_keyword_occurrences(text_lemmas: List[str], keyword: str) -> int:
    """Liczy wystąpienia frazy kluczowej (exact lemma match)."""
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
    """
    Oblicza burstiness - zróżnicowanie długości zdań.
    Target: 3.2-3.8
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    
    if not mean:
        return 0.0
    
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    raw_score = math.sqrt(variance) / mean
    normalized = raw_score * 5
    
    return round(normalized, 2)


def calculate_transition_ratio(text: str) -> Dict[str, Any]:
    """
    Oblicza stosunek zdań z transition words.
    Target: 25-50%
    """
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return {"ratio": 1.0, "count": 0, "total": len(sentences)}
    
    transition_count = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()[:100]
        has_transition = any(tw in sentence_lower for tw in ValidationConfig.TRANSITION_WORDS)
        if has_transition:
            transition_count += 1
    
    ratio = transition_count / len(sentences)
    
    return {
        "ratio": round(ratio, 3),
        "count": transition_count,
        "total": len(sentences)
    }


def calculate_density(text: str, keywords_state: Dict) -> float:
    """
    Oblicza gęstość słów kluczowych.
    """
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
    
    density = (keyword_count / total_words) * 100
    return round(density, 2)


# ================================================================
# 🎯 GŁÓWNA FUNKCJA WALIDACJI
# ================================================================
def validate_content(
    text: str,
    keywords_state: Dict = None,
    main_keyword: str = None,
    required_ngrams: List[str] = None,
    is_intro_batch: bool = False,
    existing_lists_count: int = 0,
    validation_mode: str = "full"  # "full", "preview", "final"
) -> ValidationResult:
    """
    🎯 GŁÓWNA FUNKCJA WALIDACJI - używana wszędzie w systemie.
    
    Args:
        text: Tekst do walidacji
        keywords_state: Stan słów kluczowych z projektu
        main_keyword: Fraza główna
        required_ngrams: N-gramy do sprawdzenia (z S1)
        is_intro_batch: Czy to pierwszy batch (z intro)
        existing_lists_count: Ile list już jest w artykule
        validation_mode: Tryb walidacji
    
    Returns:
        ValidationResult z kompletną analizą
    """
    issues: List[ValidationIssue] = []
    keywords_state = keywords_state or {}
    required_ngrams = required_ngrams or []
    
    # Przygotuj lematy tekstu (raz dla wszystkich sprawdzeń)
    text_lemmas = lemmatize_text(text)
    word_count = count_words(text)
    
    # ================================================================
    # 1. METRYKI JAKOŚCI TEKSTU
    # ================================================================
    burstiness = calculate_burstiness(text)
    transition_data = calculate_transition_ratio(text)
    density = calculate_density(text, keywords_state)
    
    # Walidacja burstiness
    if burstiness < ValidationConfig.BURSTINESS_MIN:
        issues.append(ValidationIssue(
            code="LOW_BURSTINESS",
            message=f"Burstiness za niski: {burstiness} (min {ValidationConfig.BURSTINESS_MIN})",
            severity=Severity.WARNING,
            details={"value": burstiness, "min": ValidationConfig.BURSTINESS_MIN}
        ))
    elif burstiness > ValidationConfig.BURSTINESS_MAX:
        issues.append(ValidationIssue(
            code="HIGH_BURSTINESS",
            message=f"Burstiness za wysoki: {burstiness} (max {ValidationConfig.BURSTINESS_MAX})",
            severity=Severity.WARNING,
            details={"value": burstiness, "max": ValidationConfig.BURSTINESS_MAX}
        ))
    
    # Walidacja transition ratio
    if transition_data["ratio"] < ValidationConfig.TRANSITION_RATIO_MIN:
        issues.append(ValidationIssue(
            code="LOW_TRANSITION_RATIO",
            message=f"Za mało transition words: {transition_data['ratio']:.0%} (min {ValidationConfig.TRANSITION_RATIO_MIN:.0%})",
            severity=Severity.WARNING,
            details=transition_data
        ))
    elif transition_data["ratio"] > ValidationConfig.TRANSITION_RATIO_MAX:
        issues.append(ValidationIssue(
            code="HIGH_TRANSITION_RATIO",
            message=f"Za dużo transition words: {transition_data['ratio']:.0%} (max {ValidationConfig.TRANSITION_RATIO_MAX:.0%})",
            severity=Severity.WARNING,
            details=transition_data
        ))
    
    # Walidacja density
    if density > ValidationConfig.DENSITY_MAX:
        issues.append(ValidationIssue(
            code="HIGH_DENSITY",
            message=f"Gęstość słów kluczowych za wysoka: {density}% (max {ValidationConfig.DENSITY_MAX}%)",
            severity=Severity.WARNING,
            details={"value": density, "max": ValidationConfig.DENSITY_MAX}
        ))
    
    # ================================================================
    # 2. ANALIZA SŁÓW KLUCZOWYCH
    # ================================================================
    keywords_analysis = {
        "main_keyword": main_keyword,
        "main_uses": 0,
        "synonym_uses": 0,
        "main_ratio": 1.0,
        "keyword_counts": {}
    }
    
    if keywords_state:
        main_uses = 0
        synonym_uses = 0
        
        for rid, meta in keywords_state.items():
            keyword = meta.get("keyword", "")
            is_main = meta.get("is_main_keyword", False)
            is_synonym = meta.get("is_synonym_of_main", False)
            
            count = count_keyword_occurrences(text_lemmas, keyword)
            keywords_analysis["keyword_counts"][keyword] = count
            
            if is_main:
                main_uses = count
            elif is_synonym:
                synonym_uses += count
        
        keywords_analysis["main_uses"] = main_uses
        keywords_analysis["synonym_uses"] = synonym_uses
        
        total = main_uses + synonym_uses
        main_ratio = main_uses / total if total > 0 else 1.0
        keywords_analysis["main_ratio"] = round(main_ratio, 2)
        
        # Walidacja proporcji main vs synonyms
        if total > 0 and main_ratio < ValidationConfig.MAIN_KEYWORD_RATIO_MIN:
            issues.append(ValidationIssue(
                code="LOW_MAIN_KEYWORD_RATIO",
                message=f"Fraza główna ma tylko {main_ratio:.0%} użyć (min {ValidationConfig.MAIN_KEYWORD_RATIO_MIN:.0%}). Zamień synonimy na '{main_keyword}'.",
                severity=Severity.ERROR,
                details={
                    "main_keyword": main_keyword,
                    "main_uses": main_uses,
                    "synonym_uses": synonym_uses,
                    "ratio": main_ratio
                }
            ))
    
    # ================================================================
    # 3. ANALIZA STRUKTURY
    # ================================================================
    structure_analysis = {
        "word_count": word_count,
        "h2_count": 0,
        "h3_count": 0,
        "list_count": 0,
        "h3_sections": []
    }
    
    # H2 analysis
    h2_titles = extract_h2_titles(text)
    structure_analysis["h2_count"] = len(h2_titles)
    structure_analysis["h2_titles"] = h2_titles
    
    # Sprawdź coverage H2 z main keyword - v26.1: max 1 H2 z keyword
    if main_keyword and h2_titles:
        main_lower = main_keyword.lower()
        h2_with_main = sum(1 for h2 in h2_titles if main_lower in h2.lower())
        structure_analysis["h2_with_main_keyword"] = h2_with_main
        structure_analysis["h2_main_keyword_max"] = ValidationConfig.H2_MAIN_KEYWORD_MAX
        
        # v26.1: Ostrzegaj gdy ZA DUŻO H2 z keyword (przeoptymalizowanie)
        if h2_with_main > ValidationConfig.H2_MAIN_KEYWORD_MAX:
            issues.append(ValidationIssue(
                code="OVEROPTIMIZED_H2_KEYWORDS",
                message=f"Za dużo H2 z frazą główną: {h2_with_main} (max {ValidationConfig.H2_MAIN_KEYWORD_MAX}). Użyj synonimów w pozostałych.",
                severity=Severity.WARNING,
                details={
                    "h2_with_main": h2_with_main,
                    "max_recommended": ValidationConfig.H2_MAIN_KEYWORD_MAX,
                    "total_h2": len(h2_titles),
                    "suggestion": "Max 1 H2 z frazą główną. Reszta: synonimy lub naturalne tytuły."
                }
            ))
    
    # H3 analysis
    h3_sections = extract_h3_sections(text)
    structure_analysis["h3_count"] = len(h3_sections)
    structure_analysis["h3_sections"] = h3_sections
    
    for section in h3_sections:
        if section["word_count"] < ValidationConfig.H3_MIN_WORDS:
            issues.append(ValidationIssue(
                code="SHORT_H3_SECTION",
                message=f"H3 '{section['title']}' ma tylko {section['word_count']} słów (min {ValidationConfig.H3_MIN_WORDS})",
                severity=Severity.WARNING,
                details=section
            ))
    
    # List count
    current_lists = count_lists(text)
    total_lists = existing_lists_count + current_lists
    structure_analysis["list_count"] = current_lists
    structure_analysis["total_lists_in_article"] = total_lists
    
    if total_lists > ValidationConfig.LIST_MAX:
        issues.append(ValidationIssue(
            code="TOO_MANY_LISTS",
            message=f"Za dużo list w artykule: {total_lists} (max {ValidationConfig.LIST_MAX})",
            severity=Severity.WARNING,
            details={"current": current_lists, "total": total_lists, "max": ValidationConfig.LIST_MAX}
        ))
    
    # ================================================================
    # 4. WALIDACJA INTRO (tylko dla pierwszego batcha)
    # ================================================================
    if is_intro_batch:
        intro_text = extract_intro(text)
        intro_words = count_words(intro_text)
        structure_analysis["intro_words"] = intro_words
        
        # Sprawdź długość intro
        if intro_words < ValidationConfig.INTRO_MIN_WORDS:
            issues.append(ValidationIssue(
                code="SHORT_INTRO",
                message=f"Intro za krótkie: {intro_words} słów (min {ValidationConfig.INTRO_MIN_WORDS})",
                severity=Severity.WARNING,
                details={"word_count": intro_words, "min": ValidationConfig.INTRO_MIN_WORDS}
            ))
        elif intro_words > ValidationConfig.INTRO_MAX_WORDS:
            issues.append(ValidationIssue(
                code="LONG_INTRO",
                message=f"Intro za długie: {intro_words} słów (max {ValidationConfig.INTRO_MAX_WORDS})",
                severity=Severity.WARNING,
                details={"word_count": intro_words, "max": ValidationConfig.INTRO_MAX_WORDS}
            ))
        
        # Sprawdź banned openers
        intro_lower = intro_text.lower()
        for banned in ValidationConfig.BANNED_INTRO_OPENERS:
            if intro_lower.startswith(banned):
                issues.append(ValidationIssue(
                    code="BANNED_INTRO_OPENER",
                    message=f"Intro zaczyna się od zakazanej frazy: '{banned}'",
                    severity=Severity.WARNING,
                    details={"banned_phrase": banned}
                ))
                break
        
        # Sprawdź czy main keyword jest w pierwszym zdaniu
        if main_keyword:
            first_sentence = intro_text.split('.')[0] if intro_text else ""
            if main_keyword.lower() not in first_sentence.lower():
                issues.append(ValidationIssue(
                    code="MAIN_KEYWORD_NOT_IN_FIRST_SENTENCE",
                    message=f"Fraza główna '{main_keyword}' powinna być w pierwszym zdaniu intro",
                    severity=Severity.WARNING,
                    details={"main_keyword": main_keyword}
                ))
    
    # ================================================================
    # 5. POKRYCIE N-GRAMÓW
    # ================================================================
    if required_ngrams:
        text_lower = text.lower()
        used_ngrams = [ng for ng in required_ngrams if ng.lower() in text_lower]
        missing_ngrams = [ng for ng in required_ngrams if ng.lower() not in text_lower]
        coverage = len(used_ngrams) / len(required_ngrams) if required_ngrams else 1.0
        
        structure_analysis["ngram_coverage"] = {
            "coverage": round(coverage, 2),
            "used": used_ngrams,
            "missing": missing_ngrams
        }
        
        if coverage < ValidationConfig.NGRAM_COVERAGE_MIN:
            issues.append(ValidationIssue(
                code="LOW_NGRAM_COVERAGE",
                message=f"Niskie pokrycie n-gramów: {coverage:.0%} (min {ValidationConfig.NGRAM_COVERAGE_MIN:.0%})",
                severity=Severity.WARNING,
                details={
                    "coverage": coverage,
                    "missing": missing_ngrams[:5]
                }
            ))
    
    # ================================================================
    # 6. SPRAWDŹ BANNED SECTION OPENERS
    # ================================================================
    h2_pattern = re.compile(r'(?:^h2:|</h2>)\s*(?:<p>)?([^.!?\n]+)', re.MULTILINE | re.IGNORECASE)
    for match in h2_pattern.findall(text):
        first_word = match.strip().split()[0].lower() if match.strip() else ""
        for banned in ValidationConfig.BANNED_SECTION_OPENERS:
            if first_word == banned or match.strip().lower().startswith(banned):
                issues.append(ValidationIssue(
                    code="BANNED_SECTION_OPENER",
                    message=f"Sekcja zaczyna się od '{banned}' - przenieś dalej w akapicie",
                    severity=Severity.INFO,
                    details={"banned_word": banned, "context": match[:50]}
                ))
                break
    
    # ================================================================
    # 7. OBLICZ SCORE
    # ================================================================
    score = 100
    for issue in issues:
        if issue.severity == Severity.ERROR:
            score -= 15
        elif issue.severity == Severity.WARNING:
            score -= 5
        elif issue.severity == Severity.INFO:
            score -= 1
    
    score = max(0, min(100, score))
    
    # is_valid = brak ERROR
    is_valid = not any(i.severity == Severity.ERROR for i in issues)
    
    # ================================================================
    # ZWRÓĆ WYNIK
    # ================================================================
    return ValidationResult(
        is_valid=is_valid,
        score=score,
        issues=issues,
        metrics={
            "burstiness": {
                "value": burstiness,
                "target": f"{ValidationConfig.BURSTINESS_MIN}-{ValidationConfig.BURSTINESS_MAX}",
                "status": "OK" if ValidationConfig.BURSTINESS_MIN <= burstiness <= ValidationConfig.BURSTINESS_MAX else "WARN"
            },
            "transition_ratio": {
                "value": transition_data["ratio"],
                "target": f"{ValidationConfig.TRANSITION_RATIO_MIN}-{ValidationConfig.TRANSITION_RATIO_MAX}",
                "status": "OK" if ValidationConfig.TRANSITION_RATIO_MIN <= transition_data["ratio"] <= ValidationConfig.TRANSITION_RATIO_MAX else "WARN"
            },
            "density": {
                "value": density,
                "target": f"<{ValidationConfig.DENSITY_MAX}%",
                "status": "OK" if density <= ValidationConfig.DENSITY_MAX else "WARN"
            },
            "word_count": word_count
        },
        keywords_analysis=keywords_analysis,
        structure_analysis=structure_analysis
    )


# ================================================================
# 🔧 HELPER: Szybka walidacja (dla preview)
# ================================================================
def quick_validate(text: str, keywords_state: Dict = None) -> Dict:
    """
    Szybka walidacja - zwraca uproszczony wynik.
    Używane w preview_batch.
    """
    result = validate_content(text, keywords_state, validation_mode="preview")
    return {
        "status": "OK" if result.is_valid else "WARN",
        "score": result.score,
        "errors": len(result.get_errors()),
        "warnings": len(result.get_warnings()),
        "metrics": result.metrics
    }


# ================================================================
# 🔧 HELPER: Pełna walidacja (dla final_review)
# ================================================================
def full_validate(
    text: str,
    keywords_state: Dict,
    main_keyword: str,
    ngrams: List[str] = None
) -> Dict:
    """
    Pełna walidacja - dla final_review.
    Zwraca kompletny raport.
    """
    result = validate_content(
        text=text,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        required_ngrams=ngrams,
        validation_mode="final"
    )
    return result.to_dict()


# ================================================================
# 🏆 E-E-A-T VALIDATION (Google 2024+)
# ================================================================
def validate_eeat(text: str) -> Dict[str, Any]:
    """
    Waliduje sygnały E-E-A-T w tekście.
    
    Returns:
        Dict z analizą E-E-A-T i rekomendacjami
    """
    try:
        from entity_ngram_analyzer import analyze_eeat, extract_entities
        
        entities = extract_entities(text)
        eeat_result = analyze_eeat(text, entities)
        return eeat_result.to_dict()
        
    except ImportError:
        # Fallback jeśli entity_ngram_analyzer niedostępny
        return _basic_eeat_check(text)


def _basic_eeat_check(text: str) -> Dict[str, Any]:
    """
    Podstawowa walidacja E-E-A-T bez pełnego entity_ngram_analyzer.
    """
    text_lower = text.lower()
    
    # Expertise signals
    expertise_signals = [
        "ekspert", "specjalista", "doświadczenie", "praktyka",
        "profesjonalny", "certyfikowany"
    ]
    expertise_found = sum(1 for s in expertise_signals if s in text_lower)
    expertise_score = min(1.0, expertise_found / 3)
    
    # Authority signals
    authority_signals = [
        "według", "zgodnie z", "badania", "dane", "raport",
        "ministerstwo", "ustawa", "art."
    ]
    authority_found = sum(1 for s in authority_signals if s in text_lower)
    authority_score = min(1.0, authority_found / 4)
    
    # Trust signals
    trust_signals = [
        "źródło", "stan na", "2024", "2025", "oficjalny",
        "potwierdzone", "dz.u."
    ]
    trust_found = sum(1 for s in trust_signals if s in text_lower)
    trust_score = min(1.0, trust_found / 3)
    
    overall = (expertise_score * 0.35 + authority_score * 0.35 + trust_score * 0.30)
    
    recommendations = []
    if expertise_score < 0.4:
        recommendations.append("Używaj terminologii branżowej i specjalistycznej")
    if authority_score < 0.4:
        recommendations.append("Dodaj źródła prawne (Dz.U., art., rozporządzenia UE)")
    if trust_score < 0.4:
        recommendations.append("Zwiększ wiarygodność (aktualne daty, konkretne liczby, paragrafy)")
    
    return {
        "scores": {
            "expertise": round(expertise_score, 2),
            "authority": round(authority_score, 2),
            "trust": round(trust_score, 2),
            "overall": round(overall, 2)
        },
        "signals_found": {
            "expertise": expertise_found,
            "authority": authority_found,
            "trust": trust_found
        },
        "recommendations": recommendations if recommendations else ["E-E-A-T OK"],
        "status": "GOOD" if overall >= 0.5 else "NEEDS_IMPROVEMENT"
    }


# ================================================================
# 🔧 HELPER: Full validation with E-E-A-T
# ================================================================
# ================================================================
# 🛡️ HELPFUL CONTENT CHECK (Google HCU)
# ================================================================
def check_helpful_content(text: str, keywords_state: Dict) -> Dict[str, Any]:
    """
    Sprawdza sygnały Helpful Content zgodnie z Google HCU.
    Dodane na podstawie rekomendacji Google.
    """
    warnings = []
    total_words = len(text.split())
    
    if total_words < 100:
        return {"is_helpful": True, "warnings": [], "metrics": {}}
    
    text_lower = text.lower()
    
    # 1. Keyword stuffing detection
    total_keyword_count = 0
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if keyword:
            total_keyword_count += text_lower.count(keyword)
    
    keyword_density = (total_keyword_count / total_words) * 100
    
    if keyword_density > 3.0:
        warnings.append({
            "type": "KEYWORD_STUFFING",
            "message": f"Gęstość słów kluczowych {keyword_density:.1f}% przekracza bezpieczny próg 3%",
            "severity": "HIGH"
        })
    elif keyword_density > 2.5:
        warnings.append({
            "type": "KEYWORD_DENSITY_WARNING",
            "message": f"Gęstość słów kluczowych {keyword_density:.1f}% zbliża się do progu",
            "severity": "MEDIUM"
        })
    
    # 2. Uniqueness ratio (vocabulary diversity)
    words = text_lower.split()
    unique_words = set(words)
    uniqueness_ratio = len(unique_words) / len(words)
    
    if uniqueness_ratio < 0.35:
        warnings.append({
            "type": "THIN_CONTENT",
            "message": f"Niska różnorodność słownictwa ({uniqueness_ratio:.1%}) - treść może być powtarzalna",
            "severity": "HIGH"
        })
    elif uniqueness_ratio < 0.45:
        warnings.append({
            "type": "LOW_VOCABULARY_DIVERSITY",
            "message": f"Umiarkowana różnorodność słownictwa ({uniqueness_ratio:.1%})",
            "severity": "MEDIUM"
        })
    
    # 3. AI-generated content patterns
    ai_patterns = [
        (r"podsumowując,?\s+(można|warto|należy)", "podsumowujące przejście"),
        (r"w dzisiejszych czasach.{0,30}(warto|należy)", "typowy opener AI"),
        (r"(niezwykle|niezmiernie|absolutnie)\s+(ważn|istotn)", "pusty intensyfikator"),
        (r"nie ulega wątpliwości,?\s+że", "fałszywa pewność"),
        (r"każdy z nas\s+(wie|zdaje|rozumie)", "fałszywa uniwersalność"),
    ]
    
    ai_matches = []
    for pattern, desc in ai_patterns:
        if re.search(pattern, text_lower):
            ai_matches.append(desc)
    
    if len(ai_matches) >= 2:
        warnings.append({
            "type": "AI_CONTENT_SIGNALS",
            "message": f"Wykryto {len(ai_matches)} wzorce typowe dla AI: {', '.join(ai_matches[:3])}",
            "severity": "HIGH"
        })
    elif len(ai_matches) == 1:
        warnings.append({
            "type": "AI_PATTERN_DETECTED",
            "message": f"Wykryto wzorzec AI: {ai_matches[0]}",
            "severity": "LOW"
        })
    
    # 4. Value check - czy treść daje wartość?
    # Heurystyka: obecność konkretów (liczby, daty, nazwy własne)
    concrete_patterns = [
        r'\d{4}\s*r',  # Daty
        r'\d+\s*%',     # Procenty
        r'\d+\s*(zł|PLN|euro|EUR)', # Kwoty
        r'art\.\s*\d+', # Artykuły prawne
    ]
    
    concrete_count = sum(
        len(re.findall(p, text)) 
        for p in concrete_patterns
    )
    
    concrete_ratio = concrete_count / (total_words / 100)  # per 100 słów
    
    if concrete_ratio < 0.5:
        warnings.append({
            "type": "LOW_CONCRETE_VALUE",
            "message": "Mało konkretnych danych (liczby, daty, fakty) - treść może być zbyt ogólna",
            "severity": "MEDIUM"
        })
    
    # Determine if helpful
    high_severity_count = len([w for w in warnings if w["severity"] == "HIGH"])
    
    return {
        "is_helpful": high_severity_count == 0,
        "warnings": warnings,
        "metrics": {
            "keyword_density": round(keyword_density, 2),
            "uniqueness_ratio": round(uniqueness_ratio, 3),
            "ai_patterns_found": len(ai_matches),
            "concrete_value_ratio": round(concrete_ratio, 2)
        },
        "score": max(0, 100 - high_severity_count * 25 - (len(warnings) - high_severity_count) * 10)
    }


# ================================================================
# 🔮 BIGRAM PREDICTABILITY CHECK (AI Detection)
# ================================================================
def check_bigram_predictability(text: str) -> Dict[str, Any]:
    """
    Sprawdza przewidywalność bigramów - typowe dla AI.
    Dodane na podstawie rekomendacji OpenAI.
    """
    # Common AI bigrams (zbyt gładkie, przewidywalne przejścia)
    PREDICTABLE_BIGRAMS = [
        # Przejścia
        ("warto", "zauważyć"), ("należy", "pamiętać"), ("warto", "podkreślić"),
        ("można", "stwierdzić"), ("trzeba", "zaznaczyć"), ("warto", "wspomnieć"),
        # Intensyfikatory
        ("niezwykle", "ważne"), ("bardzo", "istotne"), ("szczególnie", "ważne"),
        ("absolutnie", "kluczowe"), ("niezmiernie", "istotne"),
        # Pseudo-pewność
        ("nie", "ulega"), ("bez", "wątpienia"), ("z", "pewnością"),
        # AI connectors
        ("w", "związku"), ("tym", "samym"), ("co", "więcej"),
        ("ponadto", "warto"), ("dodatkowo", "należy"),
        # Filler phrases
        ("jak", "wiadomo"), ("jak", "wiemy"), ("każdy", "wie"),
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) < 50:
        return {"predictability_score": 0, "predictable_bigrams": [], "verdict": "TOO_SHORT"}
    
    # Extract bigrams from text
    text_bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    # Count predictable bigrams
    found_predictable = []
    for bigram in PREDICTABLE_BIGRAMS:
        count = text_bigrams.count(bigram)
        if count > 0:
            found_predictable.append({
                "bigram": f"{bigram[0]} {bigram[1]}",
                "count": count
            })
    
    total_bigrams = len(text_bigrams)
    predictable_count = sum(b["count"] for b in found_predictable)
    predictability_ratio = predictable_count / total_bigrams if total_bigrams > 0 else 0
    
    # Score: 0 = human-like, 100 = very AI-like
    predictability_score = min(100, predictability_ratio * 1000)  # Scale up
    
    if predictability_score > 15:
        verdict = "LIKELY_AI"
    elif predictability_score > 8:
        verdict = "MIXED"
    else:
        verdict = "LIKELY_HUMAN"
    
    return {
        "predictability_score": round(predictability_score, 1),
        "predictable_bigrams": found_predictable[:5],
        "predictable_count": predictable_count,
        "total_bigrams": total_bigrams,
        "verdict": verdict,
        "recommendation": "Przepisz używając mniej przewidywalnych przejść" if verdict == "LIKELY_AI" else None
    }


def full_validate_with_eeat(
    text: str,
    keywords_state: Dict,
    main_keyword: str,
    ngrams: List[str] = None
) -> Dict:
    """
    Pełna walidacja z E-E-A-T analysis.
    """
    # Standardowa walidacja
    result = validate_content(
        text=text,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        required_ngrams=ngrams,
        validation_mode="final"
    )
    
    result_dict = result.to_dict()
    
    # Dodaj E-E-A-T
    result_dict["eeat_analysis"] = validate_eeat(text)
    
    # Dostosuj score na podstawie E-E-A-T
    eeat_score = result_dict["eeat_analysis"]["scores"]["overall"]
    if eeat_score < 0.4:
        result_dict["score"] = max(0, result_dict["score"] - 10)
        result_dict["issues"].append({
            "code": "LOW_EEAT",
            "message": "Niski poziom sygnałów E-E-A-T - dodaj źródła prawne (Dz.U., art., rozporządzenia)",
            "severity": "WARNING",
            "details": result_dict["eeat_analysis"]
        })
    
    return result_dict


# ================================================================
# 🇵🇱 POLISH LANGUAGE QUALITY CHECK
# ================================================================
def validate_polish_quality(text: str) -> Dict[str, Any]:
    """
    Waliduje jakość języka polskiego.
    
    Returns:
        Dict z analizą jakości językowej
    """
    try:
        from polish_language_quality import analyze_polish_quality
        
        result = analyze_polish_quality(text)
        return result.to_dict()
        
    except ImportError:
        # Fallback jeśli moduł niedostępny
        return _basic_polish_check(text)


def _basic_polish_check(text: str) -> Dict[str, Any]:
    """
    Podstawowa walidacja polskiego bez pełnego modułu.
    """
    text_lower = text.lower()
    issues = []
    
    # Podstawowe banned phrases
    basic_banned = [
        "w dzisiejszych czasach", "warto wiedzieć", "nie ulega wątpliwości",
        "jak wiadomo", "każdy z nas", "coraz więcej osób"
    ]
    
    for phrase in basic_banned:
        if phrase in text_lower:
            issues.append({"type": "BANNED_PHRASE", "phrase": phrase})
    
    # Podstawowe błędne kolokacje
    basic_collocations = {
        "robić decyzję": "podejmować decyzję",
        "dawać uwagę": "zwracać uwagę",
        "grać rolę": "odgrywać rolę"
    }
    
    for wrong, correct in basic_collocations.items():
        if wrong in text_lower:
            issues.append({
                "type": "COLLOCATION_ERROR",
                "found": wrong,
                "suggested": correct
            })
    
    score = max(0, 100 - len(issues) * 15)
    
    return {
        "score": score,
        "issues_count": len(issues),
        "issues": issues,
        "status": "GOOD" if score >= 70 else "NEEDS_IMPROVEMENT",
        "recommendations": [
            f"Popraw: {i['found']} → {i['suggested']}" 
            for i in issues if i["type"] == "COLLOCATION_ERROR"
        ][:3]
    }


# ================================================================
# 🎯 FULL VALIDATION WITH ALL CHECKS
# ================================================================
def full_validate_complete(
    text: str,
    keywords_state: Dict,
    main_keyword: str,
    ngrams: List[str] = None
) -> Dict:
    """
    Kompletna walidacja: SEO + E-E-A-T + Jakość Językowa + HCU + AI Detection.
    """
    # Podstawowa walidacja SEO
    result = validate_content(
        text=text,
        keywords_state=keywords_state,
        main_keyword=main_keyword,
        required_ngrams=ngrams,
        validation_mode="final"
    )
    
    result_dict = result.to_dict()
    
    # Dodaj E-E-A-T
    result_dict["eeat_analysis"] = validate_eeat(text)
    
    # Dodaj analizę jakości języka polskiego
    result_dict["polish_quality"] = validate_polish_quality(text)
    
    # 🆕 Helpful Content Check (Google HCU)
    result_dict["helpful_content"] = check_helpful_content(text, keywords_state)
    
    # 🆕 AI Detection (Bigram Predictability)
    result_dict["ai_detection"] = check_bigram_predictability(text)
    
    # Oblicz końcowy score (średnia ważona)
    base_score = result_dict["score"]
    eeat_score = result_dict["eeat_analysis"]["scores"]["overall"] * 100
    polish_score = result_dict["polish_quality"]["score"]
    hcu_score = result_dict["helpful_content"]["score"]
    
    # AI penalty
    ai_penalty = 0
    if result_dict["ai_detection"]["verdict"] == "LIKELY_AI":
        ai_penalty = 15
    elif result_dict["ai_detection"]["verdict"] == "MIXED":
        ai_penalty = 5
    
    # Wagi: SEO 40%, E-E-A-T 20%, Polish 15%, HCU 25%
    final_score = (
        base_score * 0.40 + 
        eeat_score * 0.20 + 
        polish_score * 0.15 + 
        hcu_score * 0.25
    ) - ai_penalty
    
    result_dict["final_score"] = round(max(0, min(100, final_score)), 1)
    
    # Zbierz wszystkie rekomendacje
    all_recommendations = []
    
    # Z SEO
    for issue in result_dict.get("issues", []):
        if issue.get("severity") in ["ERROR", "WARNING"]:
            all_recommendations.append(issue.get("message", ""))
    
    # Z E-E-A-T
    eeat_recs = result_dict["eeat_analysis"].get("recommendations", [])
    all_recommendations.extend([r for r in eeat_recs if r != "E-E-A-T OK"][:2])
    
    # Z Polish Quality
    polish_recs = result_dict["polish_quality"].get("recommendations", [])
    all_recommendations.extend(polish_recs[:2])
    
    # 🆕 Z Helpful Content
    for warning in result_dict["helpful_content"].get("warnings", []):
        if warning.get("severity") == "HIGH":
            all_recommendations.append(f"HCU: {warning.get('message', '')}")
    
    # 🆕 Z AI Detection
    if result_dict["ai_detection"]["verdict"] == "LIKELY_AI":
        all_recommendations.append(
            f"AI DETECTION: Wykryto przewidywalne wzorce ({result_dict['ai_detection']['predictability_score']}%). "
            f"Przepisz: {', '.join(b['bigram'] for b in result_dict['ai_detection']['predictable_bigrams'][:3])}"
        )
    
    result_dict["all_recommendations"] = all_recommendations[:10]
    
    # Status końcowy
    if result_dict["final_score"] >= 75:
        result_dict["final_status"] = "EXCELLENT"
    elif result_dict["final_score"] >= 60:
        result_dict["final_status"] = "GOOD"
    elif result_dict["final_score"] >= 45:
        result_dict["final_status"] = "NEEDS_IMPROVEMENT"
    else:
        result_dict["final_status"] = "POOR"
    
    return result_dict
