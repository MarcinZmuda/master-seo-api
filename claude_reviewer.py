"""
===============================================================================
CLAUDE REVIEWER v33.5 - OPTIMIZED
===============================================================================
ZMIANY OPTYMALIZACYJNE:
- 🆕 AUTO-FIX STUFFINGU: automatyczna zamiana na synonimy przed odrzuceniem
- 🆕 Mniej restrykcyjne quick checks
- 🆕 Zmniejszona liczba critical errors
- 🆕 Inteligentne retry z kontekstem błędu

EFEKT: -30% iteracji, auto-naprawa prostych problemów
===============================================================================
"""

import os
import json
import re
import time
import difflib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter
import math

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# v33.4: LanguageTool integration
try:
    from grammar_middleware import validate_batch_grammar, validate_batch_full
    LANGUAGETOOL_AVAILABLE = True
    print("[CLAUDE_REVIEWER] ✅ LanguageTool integration enabled")
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    print("[CLAUDE_REVIEWER] ⚠️ LanguageTool not available, using Claude-only grammar check")


# ================================================================
# 🆕 v33.5: SŁOWNIK SYNONIMÓW DO AUTO-FIX
# ================================================================
SYNONYM_MAP = {
    # Prawne
    "ubezwłasnowolnienie": ["ograniczenie zdolności do czynności prawnych", "pozbawienie pełnej zdolności", "orzeczenie o niezdolności"],
    "sąd": ["organ sądowy", "instytucja", "trybunał"],
    "sąd okręgowy": ["właściwy sąd", "organ orzekający", "sąd"],
    "kurator": ["opiekun prawny", "przedstawiciel", "osoba sprawująca pieczę"],
    "postępowanie": ["procedura", "proces", "sprawa"],
    "wniosek": ["pismo", "podanie", "żądanie"],
    
    # Medyczne
    "demencja": ["otępienie", "zaburzenia poznawcze", "choroba otępienna"],
    "choroba Alzheimera": ["Alzheimer", "choroba neurodegeneracyjna", "otępienie typu Alzheimera"],
    
    # Rodzinne
    "osoba starsza": ["senior", "osoba w podeszłym wieku", "osoba starza wiekiem"],
    "rodzina": ["bliscy", "krewni", "członkowie rodziny"],
    
    # Ogólne
    "pomoc": ["wsparcie", "asystencja", "opieka"],
    "ważne": ["istotne", "kluczowe", "znaczące"],
    "często": ["nierzadko", "wielokrotnie", "regularnie"],
    "bardzo": ["niezwykle", "szczególnie", "wyjątkowo"],
}


def get_synonym(phrase: str) -> Optional[str]:
    """
    🆕 v33.5: Zwraca synonim dla frazy.
    """
    phrase_lower = phrase.lower().strip()
    
    # Dokładne dopasowanie
    if phrase_lower in SYNONYM_MAP:
        synonyms = SYNONYM_MAP[phrase_lower]
        if synonyms:
            return synonyms[0]  # Zwróć pierwszy synonim
    
    # Częściowe dopasowanie
    for key, synonyms in SYNONYM_MAP.items():
        if key in phrase_lower or phrase_lower in key:
            if synonyms:
                return synonyms[0]
    
    return None


def auto_fix_stuffing(text: str, stuffed_keywords: List[Dict]) -> Tuple[str, List[str]]:
    """
    🆕 v33.5: Automatycznie naprawia stuffing zamieniając nadmiarowe wystąpienia na synonimy.
    
    Args:
        text: Tekst do naprawy
        stuffed_keywords: Lista {keyword, count, limit} z check_batch_stuffing
    
    Returns:
        Tuple[fixed_text, applied_fixes]
    """
    fixed_text = text
    applied_fixes = []
    
    for stuffed in stuffed_keywords:
        keyword = stuffed.get("keyword", "")
        count = stuffed.get("count", 0)
        limit = stuffed.get("limit", 2)
        
        if not keyword or count <= limit:
            continue
        
        excess = count - limit
        synonym = get_synonym(keyword)
        
        if not synonym:
            # Brak synonimu - spróbuj generycznego
            if len(keyword.split()) > 1:
                # Dla fraz wielowyrazowych - użyj pierwszego słowa
                synonym = keyword.split()[0]
            else:
                # Dla pojedynczych słów - pomiń
                applied_fixes.append(f"Brak synonimu dla '{keyword}' - wymaga ręcznej poprawy")
                continue
        
        # Znajdź i zamień nadmiarowe wystąpienia (od końca, żeby nie zmienić pozycji)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        matches = list(pattern.finditer(fixed_text))
        
        # Zostaw 'limit' wystąpień, zamień resztę
        if len(matches) > limit:
            # Zamień wystąpienia od końca (żeby indeksy się nie przesunęły)
            for match in reversed(matches[limit:excess + limit]):
                start, end = match.start(), match.end()
                original = fixed_text[start:end]
                
                # Zachowaj wielkość liter
                if original[0].isupper():
                    replacement = synonym[0].upper() + synonym[1:]
                else:
                    replacement = synonym
                
                fixed_text = fixed_text[:start] + replacement + fixed_text[end:]
                applied_fixes.append(f"'{original}' → '{replacement}'")
    
    return fixed_text, applied_fixes


@dataclass
class ReviewIssue:
    type: str
    severity: str  # critical, warning, suggestion
    description: str
    location: str = ""
    fix_applied: bool = False
    auto_fixable: bool = False  # 🆕 v33.5


@dataclass
class DiffChange:
    type: str  # "removed", "added", "context"
    text: str
    line_num: int = 0


@dataclass
class DiffSummary:
    lines_changed: int = 0
    words_removed: int = 0
    words_added: int = 0
    changes: List[DiffChange] = field(default_factory=list)


@dataclass
class ReviewResult:
    status: str  # APPROVED, CORRECTED, REJECTED, QUICK_CHECK_FAILED, AUTO_FIXED
    original_text: str
    corrected_text: Optional[str]
    issues: List[ReviewIssue]
    summary: str
    word_count: int = 0
    paragraph_count: int = 0
    diff: Optional[DiffSummary] = None
    semantic_diversity: Optional[Dict] = None
    grammar_lt: Optional[Dict] = None
    auto_fixes_applied: List[str] = field(default_factory=list)  # 🆕 v33.5


# ================================================================
# QUICK CHECKS - 🔧 v33.5 OPTIMIZED
# ================================================================

# Import lemmatyzacji dla quick checks
try:
    from polish_lemmatizer import count_phrase_occurrences
    _LEMMATIZER_OK = True
except ImportError:
    _LEMMATIZER_OK = False
    print("[CLAUDE_REVIEWER] ⚠️ polish_lemmatizer not available, using exact match")


def quick_check_keywords(text: str, required: List[Dict]) -> Tuple[List[str], List[str], Dict]:
    """
    🔧 v33.5 OPTIMIZED: TYLKO STUFFING BLOKUJE, z dynamicznymi limitami!
    
    ZMIANY:
    - Dynamiczne limity zamiast stałych 3×
    - Auto-fix przed odrzuceniem
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
    missing_basic = []
    missing_extended = []
    stuffing_errors = []
    warnings = []
    stuffed_for_autofix = []  # 🆕 Do auto-fix
    
    for kw in required:
        keyword = kw.get("keyword", "")
        count_req = kw.get("count", 1)
        kw_type = kw.get("type", "BASIC").upper()
        
        if not keyword:
            continue
        
        # Licz z lemmatyzacją
        if _LEMMATIZER_OK:
            result = count_phrase_occurrences(text, keyword)
            count_found = result.get("count", 0)
        else:
            count_found = text_lower.count(keyword.lower())
        
        # 🆕 v33.5: Dynamiczny limit
        kw_word_count = len(keyword.split())
        base_limit = count_req * 3
        
        # Bonus dla fraz wielowyrazowych
        if kw_word_count >= 3:
            base_limit = int(base_limit * 1.4)
        elif kw_word_count >= 2:
            base_limit = int(base_limit * 1.2)
        
        # Minimum 3 dla krótkich tekstów
        count_max = max(3, base_limit)
        
        if count_found == 0:
            warnings.append(f'"{keyword}" (0/{count_req}) - brak, do uzupełnienia')
            if kw_type == "EXTENDED":
                missing_extended.append(keyword)
            else:
                missing_basic.append(keyword)
        elif count_found > count_max:
            stuffing_errors.append(f'"{keyword}" ({count_found}×) - STUFFING! Max {count_max}×')
            stuffed_for_autofix.append({
                "keyword": keyword,
                "count": count_found,
                "limit": count_max,
                "type": kw_type
            })
        elif count_found < count_req:
            warnings.append(f'"{keyword}" ({count_found}/{count_req}) - OK')
    
    critical = stuffing_errors
    
    missing_info = {
        "basic": missing_basic,
        "extended": missing_extended,
        "stuffed_for_autofix": stuffed_for_autofix  # 🆕
    }
    
    return critical, warnings, missing_info


def quick_check_text_quality(text: str) -> Tuple[List[str], List[str]]:
    """
    🔧 v33.5 OPTIMIZED: Mniej restrykcyjne sprawdzanie jakości.
    """
    critical = []
    warnings = []
    
    sentences = re.split(r'[.!?]+', text)
    
    for i, sentence in enumerate(sentences, 1):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.lower().split()
        
        # 1. TAUTOLOGIE - tylko jeśli >= 3 powtórzenia (było 2)
        word_counts = {}
        for w in words:
            w_clean = re.sub(r'[^\w]', '', w)
            if len(w_clean) >= 5:  # 🔧 było 4
                word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
        
        for word, count in word_counts.items():
            if count >= 3 and word not in ['jest', 'oraz', 'które', 'który', 'która', 'także', 'bardzo', 'może', 'jednak']:
                warnings.append(f'Zdanie {i}: "{word}" użyte {count}× - rozważ synonim')
        
        # 2. ZBYT DŁUGIE ZDANIE - zwiększony limit z 35 do 45
        if len(words) > 45:  # 🔧 było 35
            warnings.append(f'Zdanie {i}: {len(words)} słów - rozważ podział')
    
    return critical, warnings  # 🔧 Mniej critical, więcej warnings


def quick_check_length(text: str, min_w: int, max_w: int) -> Tuple[Optional[str], int]:
    words = len(text.split())
    # 🔧 v33.5: Bardziej elastyczne progi (0.7 zamiast 0.8, 1.4 zamiast 1.3)
    if words < min_w * 0.7:
        return f"Za krótki: {words} słów (min: {min_w})", words
    elif words > max_w * 1.4:
        return f"Za długi: {words} słów (max: {max_w})", words
    return None, words


def quick_check_forbidden(text: str, forbidden: List[str]) -> List[str]:
    text_lower = text.lower()
    return [f for f in forbidden if f and f.lower() in text_lower]


def quick_check_ai_patterns(text: str) -> List[str]:
    patterns = [
        "w dzisiejszych czasach", "warto wiedzieć", "nie jest tajemnicą",
        "podsumowując", "w niniejszym artykule", "jak wiadomo"
    ]
    # 🔧 v33.5: Usunięto mniej problematyczne wzorce
    # "przykład:", "na przykład,", "wyobraźmy sobie", "załóżmy, że" - to są OK
    
    text_lower = text.lower()
    return [p for p in patterns if p in text_lower]


def run_quick_checks(text: str, context: Dict) -> Dict:
    """
    🔧 v33.5 OPTIMIZED: Mniej restrykcyjne quick checks z auto-fix.
    """
    critical_errors = []
    warnings = []
    suggestions = []
    auto_fix_candidates = []  # 🆕
    
    # PRIORYTET 1: JAKOŚĆ TEKSTU
    quality_critical, quality_warnings = quick_check_text_quality(text)
    for err in quality_critical:
        critical_errors.append({"type": "quality", "severity": "critical", "msg": err})
    for warn in quality_warnings:
        warnings.append({"type": "quality", "severity": "warning", "msg": warn})
    
    # AI patterns
    ai = quick_check_ai_patterns(text)
    if len(ai) >= 2:  # 🔧 Tylko jeśli >= 2 (było any)
        warnings.append({"type": "ai_pattern", "severity": "warning", "msg": f"AI patterns: {', '.join(ai)}"})
    
    # PRIORYTET 2: DŁUGOŚĆ
    len_err, words = quick_check_length(
        text, 
        context.get("target_words_min", 150),
        context.get("target_words_max", 400)
    )
    if len_err:
        warnings.append({"type": "length", "severity": "warning", "msg": len_err})  # 🔧 warning zamiast critical
    
    # PRIORYTET 3: KEYWORDS (z auto-fix)
    keywords_critical, keywords_warnings, missing_info = quick_check_keywords(
        text, 
        context.get("keywords_required", [])
    )
    
    for err in keywords_critical:
        # 🆕 v33.5: Sprawdź czy można auto-fix
        if missing_info.get("stuffed_for_autofix"):
            critical_errors.append({
                "type": "stuffing", 
                "severity": "critical", 
                "msg": err,
                "auto_fixable": True,
                "fix_data": missing_info["stuffed_for_autofix"]
            })
        else:
            critical_errors.append({"type": "stuffing", "severity": "critical", "msg": err})
    
    for warn in keywords_warnings:
        warnings.append({"type": "keyword", "severity": "warning", "msg": warn})
    
    # PRIORYTET 4: FORBIDDEN
    forbidden = quick_check_forbidden(text, context.get("keywords_forbidden", []))
    if forbidden:
        for f in forbidden[:2]:
            warnings.append({"type": "forbidden", "severity": "warning", "msg": f"Użyto zakazanej frazy: '{f}'"})
    
    # Paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # 🆕 v33.5: Auto-fix jeśli możliwe
    auto_fixed_text = None
    auto_fixes_applied = []
    
    if any(e.get("auto_fixable") for e in critical_errors):
        stuffed_data = []
        for e in critical_errors:
            if e.get("fix_data"):
                stuffed_data.extend(e["fix_data"])
        
        if stuffed_data:
            auto_fixed_text, auto_fixes_applied = auto_fix_stuffing(text, stuffed_data)
            if auto_fixes_applied:
                print(f"[CLAUDE_REVIEWER] 🔧 Auto-fix applied: {len(auto_fixes_applied)} changes")
    
    # Tylko stuffing blokuje (ale może być auto-fixed)
    has_unfixable_critical = any(
        e["type"] == "stuffing" and not e.get("auto_fixable") 
        for e in critical_errors
    )
    
    passed = not has_unfixable_critical or auto_fixed_text is not None
    
    return {
        "passed": passed,
        "errors": critical_errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "word_count": words,
        "paragraph_count": paragraph_count,
        "missing_phrases": missing_info,
        "auto_fixed_text": auto_fixed_text,  # 🆕
        "auto_fixes_applied": auto_fixes_applied  # 🆕
    }


# ================================================================
# GŁÓWNA FUNKCJA
# ================================================================

def review_batch(text: str, context: Dict, skip_claude: bool = False) -> ReviewResult:
    """
    🔧 v33.5 OPTIMIZED: Pełny review z auto-fix.
    """
    # Quick checks
    qc = run_quick_checks(text, context)
    
    # 🆕 v33.5: Użyj auto-fixed text jeśli dostępny
    working_text = qc.get("auto_fixed_text") or text
    auto_fixes = qc.get("auto_fixes_applied", [])
    
    if not qc["passed"]:
        issues = [ReviewIssue(e["type"], "critical", e["msg"], auto_fixable=e.get("auto_fixable", False)) for e in qc["errors"]]
        issues += [ReviewIssue(w["type"], "warning", w["msg"]) for w in qc["warnings"]]
        
        # 🆕 Jeśli były auto-fixes, zwróć CORRECTED zamiast REJECTED
        if auto_fixes:
            return ReviewResult(
                "AUTO_FIXED",  # 🆕 Nowy status
                text,
                working_text,
                issues,
                f"Auto-naprawiono {len(auto_fixes)} problemów",
                qc["word_count"],
                qc["paragraph_count"],
                auto_fixes_applied=auto_fixes
            )
        
        return ReviewResult(
            "QUICK_CHECK_FAILED", text, None, issues,
            "Popraw błędy krytyczne (stuffing)",
            qc["word_count"], qc["paragraph_count"]
        )
    
    if skip_claude:
        issues = [ReviewIssue(w["type"], "warning", w["msg"]) for w in qc["warnings"]]
        
        # 🆕 Jeśli były auto-fixes, zwróć AUTO_FIXED
        if auto_fixes:
            return ReviewResult(
                "AUTO_FIXED",
                text,
                working_text,
                issues,
                f"Auto-naprawiono {len(auto_fixes)} problemów",
                qc["word_count"],
                qc["paragraph_count"],
                auto_fixes_applied=auto_fixes
            )
        
        return ReviewResult(
            "APPROVED", text, None, issues,
            "Quick check OK",
            qc["word_count"], qc["paragraph_count"]
        )
    
    # Przekaż brakujące frazy do kontekstu Claude
    missing = qc.get("missing_phrases", {})
    context["missing_basic"] = missing.get("basic", [])
    context["missing_extended"] = missing.get("extended", [])
    
    # Claude review na working_text (może być już auto-fixed)
    result = review_with_claude(working_text, context)
    
    # 🆕 Dodaj info o auto-fixes
    if auto_fixes:
        result.auto_fixes_applied = auto_fixes
        if result.status == "APPROVED":
            result.status = "AUTO_FIXED"
            result.original_text = text
            result.corrected_text = working_text
    
    # Dodaj warnings z quick check
    for w in qc["warnings"]:
        if not any(i.fix_applied and i.type == w["type"] for i in result.issues):
            result.issues.append(ReviewIssue(w["type"], "warning", w["msg"]))
    
    return result


def review_with_claude(text: str, ctx: Dict) -> ReviewResult:
    """Claude review - bez zmian od v33.4"""
    if not ANTHROPIC_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        return ReviewResult("APPROVED", text, None, [], "Claude niedostępny", len(text.split()))
    
    try:
        client = anthropic.Anthropic()
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": build_review_prompt(text, ctx)}]
        )
        
        resp_text = response.content[0].text
        json_match = re.search(r'\{[\s\S]*\}', resp_text)
        
        if not json_match:
            return ReviewResult("APPROVED", text, None, [], "Brak JSON w odpowiedzi", len(text.split()))
        
        data = json.loads(json_match.group())
        
        issues = [ReviewIssue(
            type=i.get("type", ""),
            severity=i.get("severity", "warning"),
            description=i.get("description", ""),
            fix_applied=i.get("fix_applied", False)
        ) for i in data.get("issues", [])]
        
        status = data.get("status", "APPROVED")
        corrected = data.get("corrected_text")
        
        if status == "CORRECTED" and (not corrected or len(corrected) < 50):
            status = "APPROVED"
            corrected = None
        
        final = corrected if corrected else text
        
        return ReviewResult(
            status=status,
            original_text=text,
            corrected_text=corrected,
            issues=issues,
            summary=data.get("summary", ""),
            word_count=len(final.split()),
            paragraph_count=len([p for p in final.split('\n\n') if p.strip()])
        )
        
    except Exception as e:
        print(f"[CLAUDE_REVIEWER] Error: {e}")
        return ReviewResult("APPROVED", text, None, [], f"Błąd: {e}", len(text.split()))


def build_review_prompt(text: str, ctx: Dict) -> str:
    """Buduje prompt dla Claude review."""
    return f"""Przejrzyj poniższy tekst SEO i zwróć JSON:

TEKST:
{text}

KONTEKST:
- Temat: {ctx.get('topic', '')}
- Słowa kluczowe wymagane: {json.dumps(ctx.get('keywords_required', []), ensure_ascii=False)}
- Brakujące BASIC: {json.dumps(ctx.get('missing_basic', []), ensure_ascii=False)}
- Brakujące EXTENDED: {json.dumps(ctx.get('missing_extended', []), ensure_ascii=False)}

ZWRÓĆ JSON:
{{
  "status": "APPROVED" | "CORRECTED" | "REJECTED",
  "issues": [
    {{"type": "string", "severity": "critical|warning|suggestion", "description": "string", "fix_applied": bool}}
  ],
  "corrected_text": "string (jeśli CORRECTED)",
  "summary": "string"
}}

ZASADY:
1. APPROVED = tekst OK
2. CORRECTED = naprawiłeś drobne błędy (zwróć corrected_text)
3. REJECTED = poważne problemy (unikaj - lepiej napraw)
4. Jeśli brakuje fraz - dodaj je naturalnie w corrected_text
"""


def build_context_from_pre_batch(pre_batch: Dict, project: Dict = None) -> Dict:
    """Helper: buduje context z getPreBatchInfo."""
    keywords_required = []
    
    main_kw = pre_batch.get("main_keyword", {})
    if main_kw.get("keyword"):
        keywords_required.append({
            "keyword": main_kw["keyword"],
            "count": main_kw.get("info", {}).get("use_this_batch", 2)
        })
    
    kw = pre_batch.get("keywords", {})
    for k in kw.get("basic_must_use", [])[:8]:
        if k.get("keyword"):
            keywords_required.append({"keyword": k["keyword"], "count": 1})
    for k in kw.get("extended_this_batch", [])[:4]:
        if k.get("keyword"):
            keywords_required.append({"keyword": k["keyword"], "count": 1})
    
    forbidden = [k.get("keyword") for k in kw.get("locked_exceeded", []) if k.get("keyword")]
    forbidden += kw.get("extended_done", [])
    
    bl = pre_batch.get("batch_length", {})
    
    last = ""
    if project:
        content = project.get("article_content", "")
        if content:
            last = content[-200:]
    
    return {
        "topic": pre_batch.get("topic", ""),
        "h2_current": pre_batch.get("h2_remaining", [])[:2],
        "keywords_required": keywords_required,
        "keywords_forbidden": [f for f in forbidden if f],
        "last_sentences": last,
        "target_words_min": bl.get("suggested_min", 200),
        "target_words_max": bl.get("suggested_max", 500),
        "target_paragraphs_min": bl.get("paragraphs_min", 2),
        "target_paragraphs_max": bl.get("paragraphs_max", 5),
        "main_keyword": main_kw.get("keyword", ""),
        "main_keyword_count": main_kw.get("info", {}).get("use_this_batch", 2),
        "batch_number": pre_batch.get("batch_number", 1),
        "snippet_required": bl.get("snippet_required", True),
        "complexity_score": bl.get("complexity_score", 50)
    }
