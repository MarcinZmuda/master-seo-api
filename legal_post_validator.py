"""
🆕 v36.8: LEGAL POST-VALIDATOR - Walidacja poprawności prawnej po generacji

Sprawdza:
- Poprawność przypisania sądów (Rejonowy vs Okręgowy)
- Poprawność artykułów kodeksów
- Spójność terminologii prawnej
- YMYL safety checks

Autor: Claude
Wersja: 36.8
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

# ================================================================
# KONFIGURACJA
# ================================================================

@dataclass
class LegalValidatorConfig:
    """Konfiguracja walidatora prawnego."""
    
    # Progi severity
    HIGH_SEVERITY_THRESHOLD: int = 3   # Błędy krytyczne
    MEDIUM_SEVERITY_THRESHOLD: int = 5 # Błędy średnie
    
    # Flagi
    VALIDATE_COURT_TYPES: bool = True
    VALIDATE_ARTICLE_REFERENCES: bool = True
    VALIDATE_LEGAL_TERMS: bool = True
    REQUIRE_DISCLAIMER: bool = True

CONFIG = LegalValidatorConfig()

# ================================================================
# LEGAL KNOWLEDGE BASE
# ================================================================

# Właściwość rzeczowa sądów w Polsce
COURT_JURISDICTION = {
    "sąd okręgowy": {
        "matters": [
            "ubezwłasnowolnienie",
            "sprawy o prawa majątkowe powyżej 75000",
            "sprawy o ochronę dóbr osobistych",
            "sprawy z zakresu prawa prasowego",
            "sprawy o rozwód",
            "sprawy o separację",
            "sprawy o unieważnienie małżeństwa",
            "sprawy o ustalenie istnienia małżeństwa",
            "sprawy o prawa niemajątkowe",
            "sprawy z zakresu prawa autorskiego"
        ],
        "wrong_for": [
            "alimenty",
            "drobne sprawy cywilne",
            "sprawy o zapłatę do 75000",
            "wykroczenia"
        ]
    },
    "sąd rejonowy": {
        "matters": [
            "alimenty",
            "sprawy o zapłatę do 75000 zł",
            "sprawy spadkowe",
            "postępowanie upominawcze",
            "sprawy lokatorskie",
            "sprawy pracownicze",
            "wykroczenia"
        ],
        "wrong_for": [
            "ubezwłasnowolnienie",
            "rozwód",
            "separacja"
        ]
    }
}

# Mapowanie artykułów do kodeksów
ARTICLE_SOURCES = {
    # Kodeks cywilny - ubezwłasnowolnienie
    "art. 13": {"source": "k.c.", "topic": "ubezwłasnowolnienie całkowite"},
    "art. 14": {"source": "k.c.", "topic": "skutki ubezwłasnowolnienia całkowitego"},
    "art. 16": {"source": "k.c.", "topic": "ubezwłasnowolnienie częściowe"},
    "art. 17": {"source": "k.c.", "topic": "skutki ubezwłasnowolnienia częściowego"},
    
    # Kodeks cywilny - inne
    "art. 23": {"source": "k.c.", "topic": "dobra osobiste"},
    "art. 24": {"source": "k.c.", "topic": "ochrona dóbr osobistych"},
    "art. 415": {"source": "k.c.", "topic": "odpowiedzialność deliktowa"},
    "art. 445": {"source": "k.c.", "topic": "zadośćuczynienie"},
    
    # Kodeks postępowania cywilnego
    "art. 544": {"source": "k.p.c.", "topic": "postępowanie o ubezwłasnowolnienie"},
    "art. 545": {"source": "k.p.c.", "topic": "legitymacja w sprawach o ubezwłasnowolnienie"},
    "art. 547": {"source": "k.p.c.", "topic": "wysłuchanie osoby"},
    
    # Kodeks rodzinny i opiekuńczy
    "art. 175": {"source": "k.r.o.", "topic": "kuratela"},
    "art. 178": {"source": "k.r.o.", "topic": "kurator osoby częściowo ubezwłasnowolnionej"},
}

# Terminy prawne i ich poprawne użycie
LEGAL_TERMS_VALIDATION = {
    "ubezwłasnowolnienie całkowite": {
        "requires_context": ["choroba psychiczna", "niedorozwój umysłowy", "zaburzenia psychiczne"],
        "correct_court": "sąd okręgowy",
        "effects": ["brak zdolności do czynności prawnych", "ustanowienie opiekuna"]
    },
    "ubezwłasnowolnienie częściowe": {
        "requires_context": ["choroba psychiczna", "niedorozwój umysłowy", "pijaństwo", "narkomania"],
        "correct_court": "sąd okręgowy",
        "effects": ["ograniczona zdolność do czynności prawnych", "ustanowienie kuratora"]
    },
    "zdolność do czynności prawnych": {
        "types": ["pełna", "ograniczona", "brak"],
        "related": ["osoba pełnoletnia", "ubezwłasnowolnienie"]
    }
}

# ================================================================
# VALIDATION FUNCTIONS
# ================================================================

@dataclass
class ValidationIssue:
    """Pojedynczy problem walidacji."""
    type: str
    severity: str  # HIGH, MEDIUM, LOW
    message: str
    location: str  # Fragment tekstu
    suggestion: str
    legal_basis: Optional[str] = None

def validate_court_references(text: str, topic: str) -> List[ValidationIssue]:
    """
    Waliduje czy sądy są poprawnie przypisane do spraw.
    
    Args:
        text: Tekst do walidacji
        topic: Temat artykułu (np. "ubezwłasnowolnienie")
        
    Returns:
        Lista problemów walidacji
    """
    issues = []
    text_lower = text.lower()
    
    # Sprawdź dla każdego typu sądu
    for court_type, info in COURT_JURISDICTION.items():
        if court_type in text_lower:
            # Sprawdź czy temat jest odpowiedni dla tego sądu
            topic_lower = topic.lower()
            
            # Czy temat jest w "wrong_for"?
            for wrong_matter in info.get("wrong_for", []):
                if wrong_matter in topic_lower:
                    # Znajdź fragment z sądem
                    pattern = rf'.{{0,50}}{re.escape(court_type)}.{{0,50}}'
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    location = matches[0] if matches else court_type
                    
                    # Znajdź właściwy sąd
                    correct_court = None
                    for other_court, other_info in COURT_JURISDICTION.items():
                        if wrong_matter in [m.lower() for m in other_info.get("matters", [])]:
                            correct_court = other_court
                            break
                    
                    issues.append(ValidationIssue(
                        type="WRONG_COURT",
                        severity="HIGH",
                        message=f"Błędna właściwość sądu: '{court_type}' nie rozpatruje spraw o {wrong_matter}",
                        location=location[:100],
                        suggestion=f"Zmień na '{correct_court}'" if correct_court else "Sprawdź właściwość sądu",
                        legal_basis="Art. 17 i nast. k.p.c. - właściwość rzeczowa sądów"
                    ))
    
    return issues

def validate_article_references(text: str) -> List[ValidationIssue]:
    """
    Waliduje czy artykuły są poprawnie cytowane z właściwego kodeksu.
    
    Args:
        text: Tekst do walidacji
        
    Returns:
        Lista problemów walidacji
    """
    issues = []
    
    # Znajdź wszystkie wzmianki o artykułach
    # Pattern: art. 13, art.13, artykuł 13, itp.
    article_pattern = r'art(?:ykuł)?\.?\s*(\d+)(?:\s*§\s*\d+)?'
    matches = re.finditer(article_pattern, text.lower())
    
    for match in matches:
        article_num = match.group(1)
        article_key = f"art. {article_num}"
        
        # Kontekst wokół artykułu
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].lower()
        
        if article_key in ARTICLE_SOURCES:
            expected = ARTICLE_SOURCES[article_key]
            expected_source = expected["source"]
            
            # Sprawdź czy źródło jest podane
            source_patterns = ["k.c.", "k.p.c.", "k.r.o.", "k.k.", "kodeks cywilny", 
                             "kodeks postępowania cywilnego", "kodeks rodzinny"]
            
            source_found = any(src in context for src in source_patterns)
            
            if not source_found:
                issues.append(ValidationIssue(
                    type="MISSING_ARTICLE_SOURCE",
                    severity="MEDIUM",
                    message=f"Brak źródła dla {article_key}",
                    location=context[:100],
                    suggestion=f"Dodaj źródło: '{article_key} {expected_source}'",
                    legal_basis=f"{article_key} {expected_source} - {expected['topic']}"
                ))
            elif expected_source not in context:
                # Sprawdź czy podane jest INNE źródło (błąd)
                for src in source_patterns:
                    if src in context and src != expected_source:
                        issues.append(ValidationIssue(
                            type="WRONG_ARTICLE_SOURCE",
                            severity="HIGH",
                            message=f"Błędne źródło dla {article_key}: znaleziono '{src}', powinno być '{expected_source}'",
                            location=context[:100],
                            suggestion=f"Popraw na: '{article_key} {expected_source}'",
                            legal_basis=f"{article_key} {expected_source} - {expected['topic']}"
                        ))
                        break
    
    return issues

def validate_legal_terminology(text: str, topic: str) -> List[ValidationIssue]:
    """
    Waliduje spójność terminologii prawnej.
    
    Args:
        text: Tekst do walidacji
        topic: Temat artykułu
        
    Returns:
        Lista problemów walidacji
    """
    issues = []
    text_lower = text.lower()
    
    for term, info in LEGAL_TERMS_VALIDATION.items():
        if term in text_lower:
            # Sprawdź kontekst
            required_context = info.get("requires_context", [])
            has_context = any(ctx in text_lower for ctx in required_context)
            
            if required_context and not has_context:
                issues.append(ValidationIssue(
                    type="MISSING_LEGAL_CONTEXT",
                    severity="MEDIUM",
                    message=f"Termin '{term}' użyty bez wymaganego kontekstu",
                    location=term,
                    suggestion=f"Dodaj kontekst: {', '.join(required_context[:3])}",
                    legal_basis=None
                ))
            
            # Sprawdź właściwy sąd
            correct_court = info.get("correct_court")
            if correct_court:
                wrong_courts = [c for c in COURT_JURISDICTION.keys() if c != correct_court]
                for wrong_court in wrong_courts:
                    # Sprawdź czy wrong_court jest w tym samym akapicie co term
                    pattern = rf'{re.escape(term)}.{{0,200}}{re.escape(wrong_court)}|{re.escape(wrong_court)}.{{0,200}}{re.escape(term)}'
                    if re.search(pattern, text_lower):
                        issues.append(ValidationIssue(
                            type="WRONG_COURT_FOR_TERM",
                            severity="HIGH",
                            message=f"'{term}' niepoprawnie powiązane z '{wrong_court}'",
                            location=f"{term} ... {wrong_court}",
                            suggestion=f"Sprawy o {term} rozpatruje {correct_court}",
                            legal_basis="Właściwość rzeczowa sądów"
                        ))
    
    return issues

def check_legal_disclaimer(text: str) -> List[ValidationIssue]:
    """
    Sprawdza czy artykuł zawiera wymagany disclaimer prawny.
    
    Args:
        text: Tekst do walidacji
        
    Returns:
        Lista problemów (pusta jeśli disclaimer jest)
    """
    issues = []
    text_lower = text.lower()
    
    disclaimer_indicators = [
        "nie stanowi porady prawnej",
        "charakter informacyjny",
        "konsultacja z prawnikiem",
        "zastrzeżenie prawne",
        "porada prawna",
        "nie jest poradą prawną"
    ]
    
    has_disclaimer = any(ind in text_lower for ind in disclaimer_indicators)
    
    if not has_disclaimer and CONFIG.REQUIRE_DISCLAIMER:
        issues.append(ValidationIssue(
            type="MISSING_DISCLAIMER",
            severity="MEDIUM",
            message="Brak zastrzeżenia prawnego (disclaimer)",
            location="Koniec artykułu",
            suggestion="Dodaj: 'Niniejszy artykuł ma charakter informacyjny i nie stanowi porady prawnej.'",
            legal_basis="Wymóg YMYL - artykuły prawne powinny zawierać disclaimer"
        ))
    
    return issues

# ================================================================
# MAIN VALIDATION FUNCTION
# ================================================================

def validate_legal_content(
    text: str,
    topic: str,
    detected_category: str = "prawo",
    validate_all: bool = True
) -> Dict[str, Any]:
    """
    Główna funkcja walidacji treści prawnej.
    
    Args:
        text: Tekst do walidacji
        topic: Temat artykułu
        detected_category: Wykryta kategoria
        validate_all: Czy wykonać wszystkie walidacje
        
    Returns:
        Wyniki walidacji
    """
    if detected_category != "prawo":
        return {
            "validated": False,
            "reason": "Not a legal article",
            "issues": [],
            "passed": True
        }
    
    all_issues: List[ValidationIssue] = []
    
    # 1. Walidacja sądów
    if CONFIG.VALIDATE_COURT_TYPES:
        all_issues.extend(validate_court_references(text, topic))
    
    # 2. Walidacja artykułów
    if CONFIG.VALIDATE_ARTICLE_REFERENCES:
        all_issues.extend(validate_article_references(text))
    
    # 3. Walidacja terminologii
    if CONFIG.VALIDATE_LEGAL_TERMS:
        all_issues.extend(validate_legal_terminology(text, topic))
    
    # 4. Sprawdzenie disclaimera
    if CONFIG.REQUIRE_DISCLAIMER:
        all_issues.extend(check_legal_disclaimer(text))
    
    # Podsumowanie
    high_severity = [i for i in all_issues if i.severity == "HIGH"]
    medium_severity = [i for i in all_issues if i.severity == "MEDIUM"]
    low_severity = [i for i in all_issues if i.severity == "LOW"]
    
    passed = len(high_severity) == 0
    
    return {
        "validated": True,
        "passed": passed,
        "total_issues": len(all_issues),
        "high_severity_count": len(high_severity),
        "medium_severity_count": len(medium_severity),
        "low_severity_count": len(low_severity),
        "issues": [
            {
                "type": i.type,
                "severity": i.severity,
                "message": i.message,
                "location": i.location[:100] if i.location else "",
                "suggestion": i.suggestion,
                "legal_basis": i.legal_basis
            }
            for i in all_issues
        ],
        "summary": {
            "court_issues": len([i for i in all_issues if "COURT" in i.type]),
            "article_issues": len([i for i in all_issues if "ARTICLE" in i.type]),
            "terminology_issues": len([i for i in all_issues if "TERM" in i.type or "CONTEXT" in i.type]),
            "disclaimer_missing": any(i.type == "MISSING_DISCLAIMER" for i in all_issues)
        }
    }

# ================================================================
# INTEGRATION WITH APPROVE_BATCH
# ================================================================

def validate_batch_legal_content(
    batch_text: str,
    project_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Waliduje batch pod kątem poprawności prawnej.
    Używane w approve_batch.
    
    Args:
        batch_text: Tekst batcha
        project_data: Dane projektu
        
    Returns:
        Wynik walidacji z ewentualnymi ostrzeżeniami
    """
    topic = project_data.get("topic", project_data.get("main_keyword", ""))
    detected_category = project_data.get("detected_category", "general")
    
    result = validate_legal_content(batch_text, topic, detected_category)
    
    # Konwertuj na format warnings
    warnings = []
    
    if result.get("validated") and not result.get("passed"):
        for issue in result.get("issues", []):
            if issue["severity"] == "HIGH":
                warnings.append({
                    "type": "LEGAL_VALIDATION_ERROR",
                    "message": issue["message"],
                    "suggestion": issue["suggestion"],
                    "severity": "HIGH"
                })
            elif issue["severity"] == "MEDIUM":
                warnings.append({
                    "type": "LEGAL_VALIDATION_WARNING",
                    "message": issue["message"],
                    "suggestion": issue["suggestion"],
                    "severity": "MEDIUM"
                })
    
    return {
        "valid": result.get("passed", True),
        "warnings": warnings,
        "details": result
    }

# ================================================================
# TESTING
# ================================================================

def test_legal_post_validator():
    """Test legal post validator."""
    print("="*60)
    print("LEGAL POST-VALIDATOR TEST")
    print("="*60)
    
    test_text = """
    Ubezwłasnowolnienie całkowite to instytucja prawa cywilnego.
    Zgodnie z art. 13 osobę można ubezwłasnowolnić całkowicie,
    jeżeli wskutek choroby psychicznej nie jest w stanie kierować swoim postępowaniem.
    
    Wniosek o ubezwłasnowolnienie składa się do sądu rejonowego.
    Sąd powołuje biegłego psychiatrę do zbadania osoby.
    
    Artykuł 544 k.p.c. reguluje postępowanie w sprawach o ubezwłasnowolnienie.
    """
    
    print(f"\nTekst testowy: {len(test_text.split())} słów")
    
    result = validate_legal_content(test_text, "ubezwłasnowolnienie", "prawo")
    
    print(f"\n1. Wynik walidacji:")
    print(f"   Passed: {result['passed']}")
    print(f"   Total issues: {result['total_issues']}")
    print(f"   High severity: {result['high_severity_count']}")
    print(f"   Medium severity: {result['medium_severity_count']}")
    
    print(f"\n2. Znalezione problemy:")
    for issue in result["issues"]:
        print(f"   [{issue['severity']}] {issue['type']}")
        print(f"      {issue['message']}")
        print(f"      Sugestia: {issue['suggestion']}")
        print()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_legal_post_validator()
