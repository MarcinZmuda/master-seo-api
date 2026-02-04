"""
===============================================================================
🏥 MEDICAL MODULE v1.0 - BRAJEN SEO Engine
===============================================================================
Główny moduł do obsługi treści medycznych (YMYL Health).

Pipeline:
1. Detekcja → czy temat jest medyczny
2. Mapowanie → polskie terminy → MeSH/angielskie
3. Wyszukiwanie → PubMed + ClinicalTrials + PL sources
4. Weryfikacja → Claude wybiera najlepsze źródła
5. Cytowania → format NLM/APA
6. Walidacja → sprawdzenie gotowego artykułu

Eksportowane funkcje:
- detect_category: Wykrywa czy temat medyczny
- get_medical_context_for_article: Główna funkcja - pobiera źródła
- validate_medical_article: Waliduje cytaty
- MEDICAL_DISCLAIMER: Tekst disclaimera

Autor: BRAJEN SEO Engine v44.2
===============================================================================
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# ============================================================================
# IMPORT KOMPONENTÓW
# ============================================================================

# Źródło 1: PubMed
PUBMED_AVAILABLE = False
try:
    from .pubmed_client import (
        search_pubmed,
        search_pubmed_mesh,
        get_pubmed_client,
        PUBMED_AVAILABLE as _PUBMED
    )
    PUBMED_AVAILABLE = _PUBMED
    print("[MEDICAL_MODULE] ✅ PubMed Client loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ PubMed Client not available: {e}")

# Źródło 2: ClinicalTrials.gov
CLINICALTRIALS_AVAILABLE = False
try:
    from .clinicaltrials_client import (
        search_clinical_trials,
        search_completed_trials,
        get_clinicaltrials_client,
        CLINICALTRIALS_AVAILABLE as _CT
    )
    CLINICALTRIALS_AVAILABLE = _CT
    print("[MEDICAL_MODULE] ✅ ClinicalTrials Client loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ ClinicalTrials Client not available: {e}")

# Źródło 3: Polskie instytucje
POLISH_HEALTH_AVAILABLE = False
try:
    from .polish_health_scraper import (
        search_polish_health,
        search_pzh,
        search_aotmit,
        POLISH_HEALTH_AVAILABLE as _PL
    )
    POLISH_HEALTH_AVAILABLE = _PL
    print("[MEDICAL_MODULE] ✅ Polish Health Scraper loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ Polish Health Scraper not available: {e}")

# Detektor terminów
TERM_DETECTOR_AVAILABLE = False
try:
    from .medical_term_detector import (
        detect_medical_topic,
        get_search_strategy,
        build_pubmed_query
    )
    TERM_DETECTOR_AVAILABLE = True
    print("[MEDICAL_MODULE] ✅ Term Detector loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ Term Detector not available: {e}")

# Claude Verifier
CLAUDE_VERIFIER_AVAILABLE = False
try:
    from .claude_medical_verifier import (
        verify_publications_with_claude,
        get_evidence_level,
        get_evidence_label,
        CLAUDE_MEDICAL_VERIFIER_AVAILABLE as _CV
    )
    CLAUDE_VERIFIER_AVAILABLE = _CV
    print("[MEDICAL_MODULE] ✅ Claude Verifier loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ Claude Verifier not available: {e}")

# Generator cytowań
CITATION_GENERATOR_AVAILABLE = False
try:
    from .medical_citation_generator import (
        format_citation,
        format_inline,
        CitationStyle
    )
    CITATION_GENERATOR_AVAILABLE = True
    print("[MEDICAL_MODULE] ✅ Citation Generator loaded")
except ImportError as e:
    print(f"[MEDICAL_MODULE] ⚠️ Citation Generator not available: {e}")


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class MedicalConfig:
    """Konfiguracja modułu medycznego."""
    
    # Limity
    MAX_CITATIONS_PER_ARTICLE: int = 3
    MAX_PUBMED_RESULTS: int = 10
    MAX_CLINICAL_TRIALS: int = 5
    MAX_POLISH_RESULTS: int = 5
    
    # Scoring
    MIN_RELEVANCE_SCORE: int = 40
    
    # Filtry jakości
    MIN_YEAR: int = 2015
    PREFERRED_ARTICLE_TYPES: List[str] = field(default_factory=lambda: [
        "Systematic Review",
        "Meta-Analysis",
        "Randomized Controlled Trial",
        "Clinical Trial",
        "Guideline",
        "Practice Guideline"
    ])
    
    # Cache
    CACHE_TTL_HOURS: int = 24


CONFIG = MedicalConfig()


# ============================================================================
# DISCLAIMER
# ============================================================================

MEDICAL_DISCLAIMER = """
ZASTRZEŻENIE: Niniejszy artykuł ma charakter wyłącznie informacyjny i edukacyjny. 
Nie stanowi porady medycznej ani nie zastępuje konsultacji z lekarzem lub innym 
wykwalifikowanym pracownikiem służby zdrowia. W przypadku problemów zdrowotnych 
należy skonsultować się z lekarzem. Autor nie ponosi odpowiedzialności za 
ewentualne skutki zastosowania informacji zawartych w artykule.
""".strip()

MEDICAL_DISCLAIMER_SHORT = """
Ten artykuł ma charakter informacyjny i nie zastępuje porady lekarskiej. 
W przypadku problemów zdrowotnych skonsultuj się z lekarzem.
""".strip()


# ============================================================================
# DETEKCJA KATEGORII
# ============================================================================

def detect_category(
    main_keyword: str,
    additional_keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Wykrywa czy temat jest medyczny (YMYL Health).
    
    Args:
        main_keyword: Główne słowo kluczowe (np. "leczenie cukrzycy")
        additional_keywords: Dodatkowe słowa kluczowe
    
    Returns:
        {
            "category": "medycyna" | "general",
            "is_ymyl": True/False,
            "confidence": 0.0-1.0,
            "specialization": "endokrynologia" | None,
            "detected_keywords": {...},
            "medical_module_enabled": True/False,
            "sources_available": {...}
        }
    """
    additional_keywords = additional_keywords or []
    
    # Użyj detektora terminów
    if TERM_DETECTOR_AVAILABLE:
        detection = detect_medical_topic(main_keyword, additional_keywords)
        
        return {
            "category": "medycyna" if detection["is_medical"] else "general",
            "is_ymyl": detection["is_ymyl"],
            "confidence": detection["confidence"],
            "specialization": detection.get("specialization"),
            "detected_keywords": detection.get("detected_keywords", {}),
            "english_query": detection.get("english_query", ""),
            "mesh_suggestions": detection.get("mesh_suggestions", []),
            "medical_module_enabled": True,
            "sources_available": {
                "pubmed": PUBMED_AVAILABLE,
                "clinicaltrials": CLINICALTRIALS_AVAILABLE,
                "polish_health": POLISH_HEALTH_AVAILABLE,
                "claude_verifier": CLAUDE_VERIFIER_AVAILABLE
            }
        }
    
    # Fallback - prosta detekcja
    medical_keywords = [
        "choroba", "leczenie", "lek", "terapia", "objaw", "diagnoza",
        "badanie", "zdrowie", "lekarz", "szpital", "cukrzyca", "rak",
        "serce", "depresja", "szczepionka", "antybiotyk"
    ]
    
    text = " ".join([main_keyword] + additional_keywords).lower()
    matches = [kw for kw in medical_keywords if kw in text]
    
    is_medical = len(matches) >= 1
    confidence = min(1.0, len(matches) / 3)
    
    return {
        "category": "medycyna" if is_medical else "general",
        "is_ymyl": is_medical,
        "confidence": round(confidence, 2),
        "specialization": None,
        "detected_keywords": {"matches": matches},
        "medical_module_enabled": is_medical,
        "sources_available": {
            "pubmed": PUBMED_AVAILABLE,
            "clinicaltrials": CLINICALTRIALS_AVAILABLE,
            "polish_health": POLISH_HEALTH_AVAILABLE,
            "claude_verifier": CLAUDE_VERIFIER_AVAILABLE
        }
    }


# ============================================================================
# GŁÓWNA FUNKCJA - POBIERANIE KONTEKSTU
# ============================================================================

def get_medical_context_for_article(
    main_keyword: str,
    additional_keywords: List[str] = None,
    max_results: int = None,
    include_clinical_trials: bool = True,
    include_polish_sources: bool = True,
    force_enable: bool = False
) -> Dict[str, Any]:
    """
    Główna funkcja - pobiera kontekst medyczny dla artykułu.
    
    Pipeline:
    1. Wykryj czy temat medyczny
    2. Wygeneruj strategię wyszukiwania
    3. Szukaj w PubMed
    4. Szukaj w ClinicalTrials.gov (opcjonalnie)
    5. Szukaj w polskich źródłach (opcjonalnie)
    6. Weryfikuj przez Claude
    7. Generuj instrukcję dla GPT
    
    Args:
        main_keyword: Główne słowo kluczowe (po polsku)
        additional_keywords: Dodatkowe słowa kluczowe
        max_results: Max wyników (default: CONFIG.MAX_CITATIONS_PER_ARTICLE)
        include_clinical_trials: Czy szukać badań klinicznych
        include_polish_sources: Czy szukać w polskich źródłach
        force_enable: Wymuś działanie nawet dla nie-medycznych tematów
    
    Returns:
        {
            "status": "OK" | "NOT_MEDICAL" | "NO_RESULTS" | "ERROR",
            "category": {...},
            "total_found": int,
            "publications": [...],
            "clinical_trials": [...],
            "polish_sources": [...],
            "instruction": "...",
            "disclaimer": "..."
        }
    """
    additional_keywords = additional_keywords or []
    max_results = max_results or CONFIG.MAX_CITATIONS_PER_ARTICLE
    
    # 1. DETEKCJA KATEGORII
    category = detect_category(main_keyword, additional_keywords)
    
    if not category["is_ymyl"] and not force_enable:
        return {
            "status": "NOT_MEDICAL",
            "category": category,
            "reason": "Temat nie został sklasyfikowany jako medyczny",
            "total_found": 0,
            "publications": [],
            "clinical_trials": [],
            "polish_sources": [],
            "instruction": "",
            "medical_module_active": False
        }
    
    # 2. STRATEGIA WYSZUKIWANIA
    search_strategy = None
    if TERM_DETECTOR_AVAILABLE:
        search_strategy = get_search_strategy(main_keyword)
    
    # Zmienne do zbierania wyników
    all_publications = []
    clinical_trials = []
    polish_sources = []
    sources_used = []
    
    # 3. SZUKAJ W PUBMED
    if PUBMED_AVAILABLE:
        try:
            pubmed_query = search_strategy["pubmed_query"] if search_strategy else main_keyword
            
            print(f"[MEDICAL_MODULE] 🔬 PubMed search: {pubmed_query}")
            
            pubmed_result = search_pubmed(
                query=pubmed_query,
                max_results=CONFIG.MAX_PUBMED_RESULTS,
                min_year=CONFIG.MIN_YEAR,
                article_types=CONFIG.PREFERRED_ARTICLE_TYPES
            )
            
            if pubmed_result.get("status") == "OK":
                pubs = pubmed_result.get("publications", [])
                for pub in pubs:
                    pub["_source"] = "pubmed"
                all_publications.extend(pubs)
                sources_used.append("PubMed")
                print(f"[MEDICAL_MODULE] ✅ PubMed: {len(pubs)} results")
                
        except Exception as e:
            print(f"[MEDICAL_MODULE] ⚠️ PubMed error: {e}")
    
    # 4. SZUKAJ W CLINICALTRIALS.GOV
    if include_clinical_trials and CLINICALTRIALS_AVAILABLE:
        try:
            ct_condition = search_strategy.get("clinicaltrials_condition") if search_strategy else None
            ct_intervention = search_strategy.get("clinicaltrials_intervention") if search_strategy else None
            
            if ct_condition:
                print(f"[MEDICAL_MODULE] 🧪 ClinicalTrials search: {ct_condition}")
                
                ct_result = search_completed_trials(
                    condition=ct_condition,
                    intervention=ct_intervention,
                    max_results=CONFIG.MAX_CLINICAL_TRIALS
                )
                
                if ct_result.get("status") == "OK":
                    studies = ct_result.get("studies", [])
                    clinical_trials.extend(studies)
                    sources_used.append("ClinicalTrials.gov")
                    print(f"[MEDICAL_MODULE] ✅ ClinicalTrials: {len(studies)} results")
                    
        except Exception as e:
            print(f"[MEDICAL_MODULE] ⚠️ ClinicalTrials error: {e}")
    
    # 5. SZUKAJ W POLSKICH ŹRÓDŁACH
    if include_polish_sources and POLISH_HEALTH_AVAILABLE:
        try:
            polish_query = search_strategy.get("polish_query") if search_strategy else main_keyword
            
            print(f"[MEDICAL_MODULE] 🇵🇱 Polish sources search: {polish_query}")
            
            pl_result = search_polish_health(
                query=polish_query,
                max_results_per_source=2,  # 2 z każdego źródła
                sources=["pzh", "aotmit"]  # Tylko TOP authority
            )
            
            if pl_result.get("status") == "OK":
                pl_items = pl_result.get("results", [])
                polish_sources.extend(pl_items)
                sources_used.append("Polish Health (PZH, AOTMiT)")
                print(f"[MEDICAL_MODULE] ✅ Polish: {len(pl_items)} results")
                
        except Exception as e:
            print(f"[MEDICAL_MODULE] ⚠️ Polish sources error: {e}")
    
    # 6. WERYFIKACJA PRZEZ CLAUDE
    verified_publications = all_publications
    
    if CLAUDE_VERIFIER_AVAILABLE and all_publications:
        try:
            print(f"[MEDICAL_MODULE] 🤖 Claude verification...")
            
            verification = verify_publications_with_claude(
                article_topic=main_keyword,
                publications=all_publications,
                max_to_select=max_results
            )
            
            if verification.get("status") == "OK":
                verified_publications = verification.get("selected", [])
                print(f"[MEDICAL_MODULE] ✅ Claude selected: {len(verified_publications)}")
            
        except Exception as e:
            print(f"[MEDICAL_MODULE] ⚠️ Claude verification error: {e}")
    else:
        # Bez Claude - weź pierwsze N
        verified_publications = all_publications[:max_results]
    
    # 7. GENERUJ INSTRUKCJĘ
    instruction = _build_instruction(
        main_keyword=main_keyword,
        publications=verified_publications,
        clinical_trials=clinical_trials[:2],
        polish_sources=polish_sources[:2],
        max_citations=max_results
    )
    
    # WYNIK
    total_found = len(verified_publications) + len(clinical_trials) + len(polish_sources)
    
    return {
        "status": "OK" if total_found > 0 else "NO_RESULTS",
        "category": category,
        "search_strategy": search_strategy,
        "total_found": total_found,
        "publications": verified_publications,
        "clinical_trials": clinical_trials[:CONFIG.MAX_CLINICAL_TRIALS],
        "polish_sources": polish_sources[:CONFIG.MAX_POLISH_RESULTS],
        "sources_used": sources_used,
        "instruction": instruction,
        "disclaimer": MEDICAL_DISCLAIMER,
        "disclaimer_short": MEDICAL_DISCLAIMER_SHORT,
        "medical_module_active": True,
        "config": {
            "max_citations": CONFIG.MAX_CITATIONS_PER_ARTICLE,
            "min_year": CONFIG.MIN_YEAR
        }
    }


def _build_instruction(
    main_keyword: str,
    publications: List[Dict],
    clinical_trials: List[Dict],
    polish_sources: List[Dict],
    max_citations: int
) -> str:
    """Buduje instrukcję dla GPT."""
    
    lines = [
        "",
        "=" * 60,
        "🏥 KONTEKST MEDYCZNY (YMYL Health)",
        "=" * 60,
        "",
        f"Temat artykułu: {main_keyword}",
        "",
        "Ten artykuł dotyczy tematyki MEDYCZNEJ. Przestrzegaj zasad:",
        "",
        "1. ŹRÓDŁA NAUKOWE:",
        f"   • Cytuj MAX {max_citations} publikacje z poniższej listy",
        "   • Używaj formatu: (Autor i wsp., Rok)",
        "   • NIE wymyślaj badań/autorów!",
        "",
        "2. JĘZYK:",
        "   • Używaj precyzyjnej terminologii medycznej",
        "   • Wyjaśniaj trudne terminy dla laików",
        "",
        "3. DISCLAIMER:",
        "   • OBOWIĄZKOWO dodaj zastrzeżenie na końcu artykułu",
        "",
    ]
    
    # Publikacje PubMed
    if publications:
        lines.append("🔬 PUBLIKACJE NAUKOWE (PubMed):")
        lines.append("")
        
        for i, pub in enumerate(publications[:max_citations], 1):
            # Generuj cytowanie
            if CITATION_GENERATOR_AVAILABLE:
                citation = format_citation(pub)
                inline = citation["inline"]
                full = citation["full"]
            else:
                inline = pub.get("authors_short", "Unknown")
                full = f"{pub.get('title', 'N/A')}"
            
            evidence = pub.get("evidence_level", "?")
            evidence_label = get_evidence_label(evidence) if CLAUDE_VERIFIER_AVAILABLE else ""
            
            lines.append(f"═══ PUBLIKACJA #{i} ═══")
            lines.append(f"   📌 Cytuj jako: {inline}")
            lines.append(f"   📄 Tytuł: {pub.get('title', 'N/A')[:80]}...")
            lines.append(f"   👥 Autorzy: {pub.get('authors_short', 'N/A')}")
            lines.append(f"   📰 Źródło: {pub.get('journal_abbrev', pub.get('journal', 'N/A'))} ({pub.get('year', 'N/A')})")
            lines.append(f"   ⭐ Poziom dowodów: {evidence} - {evidence_label}")
            lines.append(f"   🔗 URL: {pub.get('url', 'N/A')}")
            
            if pub.get("doi"):
                lines.append(f"   📎 DOI: {pub['doi']}")
            
            lines.append("")
    
    # Badania kliniczne
    if clinical_trials:
        lines.append("🧪 BADANIA KLINICZNE (ClinicalTrials.gov):")
        lines.append("")
        
        for study in clinical_trials[:2]:
            lines.append(f"   📋 {study.get('nct_id', 'N/A')}: {study.get('brief_title', 'N/A')[:60]}...")
            lines.append(f"      Status: {study.get('status_pl', study.get('status', 'N/A'))}")
            lines.append(f"      Faza: {', '.join(study.get('phases_pl', study.get('phases', [])))}")
            lines.append(f"      URL: {study.get('url', 'N/A')}")
            lines.append("")
    
    # Polskie źródła
    if polish_sources:
        lines.append("🇵🇱 POLSKIE ŹRÓDŁA (dla Trust signals):")
        lines.append("")
        
        for source in polish_sources[:2]:
            lines.append(f"   📄 [{source.get('source_short', 'PL')}] {source.get('title', 'N/A')[:60]}...")
            lines.append(f"      URL: {source.get('url', 'N/A')[:60]}...")
            lines.append("")
    
    lines.append("=" * 60)
    lines.append("⚠️ OBOWIĄZKOWY DISCLAIMER (dodaj na końcu artykułu):")
    lines.append("=" * 60)
    lines.append("")
    lines.append(MEDICAL_DISCLAIMER_SHORT)
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# WALIDACJA ARTYKUŁU
# ============================================================================

def validate_medical_article(full_text: str) -> Dict[str, Any]:
    """
    Waliduje artykuł medyczny.
    
    Sprawdza:
    - Obecność cytowań
    - Obecność disclaimera
    - Liczbę cytowań
    
    Returns:
        {
            "valid": True/False,
            "citations_found": int,
            "has_disclaimer": True/False,
            "warnings": [...],
            "suggestions": [...]
        }
    """
    warnings = []
    suggestions = []
    
    text_lower = full_text.lower()
    
    # Sprawdź cytowania (format: Autor i wsp., 2023 lub Autor et al., 2023)
    import re
    citation_patterns = [
        r'\([A-Z][a-ząćęłńóśźż]+ i wsp\.,? \d{4}\)',  # Polski
        r'\([A-Z][a-z]+ et al\.,? \d{4}\)',            # Angielski
        r'\([A-Z][a-z]+,? \d{4}\)',                    # Prosty
    ]
    
    citations_found = 0
    for pattern in citation_patterns:
        citations_found += len(re.findall(pattern, full_text, re.IGNORECASE))
    
    # Sprawdź disclaimer
    disclaimer_keywords = [
        "zastrzeżenie",
        "nie stanowi porady",
        "konsultacja z lekarzem",
        "skonsultuj się z lekarzem",
        "charakter informacyjny"
    ]
    has_disclaimer = any(kw in text_lower for kw in disclaimer_keywords)
    
    # Walidacja
    if citations_found == 0:
        warnings.append("Brak cytowań naukowych")
        suggestions.append("Dodaj cytowania w formacie: (Autor i wsp., 2023)")
    
    if citations_found > CONFIG.MAX_CITATIONS_PER_ARTICLE * 2:
        warnings.append(f"Za dużo cytowań ({citations_found})")
        suggestions.append(f"Ogranicz do {CONFIG.MAX_CITATIONS_PER_ARTICLE} najważniejszych")
    
    if not has_disclaimer:
        warnings.append("Brak disclaimera medycznego")
        suggestions.append("Dodaj zastrzeżenie na końcu artykułu")
    
    return {
        "valid": len(warnings) == 0,
        "citations_found": citations_found,
        "has_disclaimer": has_disclaimer,
        "warnings": warnings,
        "suggestions": suggestions,
        "disclaimer_template": MEDICAL_DISCLAIMER_SHORT
    }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # Główne funkcje
    "detect_category",
    "get_medical_context_for_article",
    "validate_medical_article",
    
    # Stałe
    "MEDICAL_DISCLAIMER",
    "MEDICAL_DISCLAIMER_SHORT",
    "CONFIG",
    
    # Dostępność źródeł
    "PUBMED_AVAILABLE",
    "CLINICALTRIALS_AVAILABLE",
    "POLISH_HEALTH_AVAILABLE",
    "CLAUDE_VERIFIER_AVAILABLE",
    "TERM_DETECTOR_AVAILABLE",
    "CITATION_GENERATOR_AVAILABLE"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 MEDICAL MODULE v1.0 TEST")
    print("=" * 60)
    
    print(f"\n📦 Dostępne źródła:")
    print(f"   PubMed:         {'✅' if PUBMED_AVAILABLE else '❌'}")
    print(f"   ClinicalTrials: {'✅' if CLINICALTRIALS_AVAILABLE else '❌'}")
    print(f"   Polish Health:  {'✅' if POLISH_HEALTH_AVAILABLE else '❌'}")
    print(f"   Claude:         {'✅' if CLAUDE_VERIFIER_AVAILABLE else '❌'}")
    print(f"   Term Detector:  {'✅' if TERM_DETECTOR_AVAILABLE else '❌'}")
    print(f"   Citations:      {'✅' if CITATION_GENERATOR_AVAILABLE else '❌'}")
    
    # Test detekcji
    print(f"\n{'='*60}")
    print("🔍 Test detekcji kategorii:")
    print("="*60)
    
    test_topics = [
        "leczenie cukrzycy typu 2",
        "przepis na ciasto czekoladowe",
        "objawy zawału serca"
    ]
    
    for topic in test_topics:
        result = detect_category(topic)
        status = "✅ MEDYCZNY" if result["is_ymyl"] else "❌ NIE-MEDYCZNY"
        print(f"\n'{topic}'")
        print(f"   → {status} (confidence: {result['confidence']})")
        print(f"   → Specjalizacja: {result.get('specialization', 'N/A')}")
    
    # Test pobierania kontekstu
    print(f"\n{'='*60}")
    print("📚 Test pobierania kontekstu medycznego:")
    print("="*60)
    
    result = get_medical_context_for_article(
        main_keyword="leczenie cukrzycy typu 2 metforminą",
        max_results=2,
        include_clinical_trials=True,
        include_polish_sources=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Total found: {result['total_found']}")
    print(f"Sources used: {result.get('sources_used', [])}")
    print(f"Publications: {len(result.get('publications', []))}")
    print(f"Clinical trials: {len(result.get('clinical_trials', []))}")
    print(f"Polish sources: {len(result.get('polish_sources', []))}")
    
    # Pokaż fragment instrukcji
    instruction = result.get("instruction", "")
    if instruction:
        print(f"\n📝 Instrukcja (fragment):")
        print(instruction[:500] + "...")
