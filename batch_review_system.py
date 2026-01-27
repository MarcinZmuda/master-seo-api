"""
===============================================================================
🔍 BATCH REVIEW SYSTEM v37.5
===============================================================================
Kompleksowy system review i auto-poprawek po każdym batchu.

ZMIANY v37.5:
- 🔧 FIX: exceeded_by - zamienia DOKŁADNIE tyle ile przekroczono
- 🔧 FIX: max_replacements dynamiczne (nie hardcoded 1-2)
- 🆕 Pole exceeded_by w ReviewIssue
- 🆕 Lepsza informacja w fix_suggestion ("Zamień 3× na...")

FLOW:
1. Batch przychodzi → walidacja
2. Wykrycie problemów → kategoryzacja
3. AUTO-FIX dla prostych problemów
4. CLAUDE-FIX dla złożonych problemów
5. Zwrot poprawionego batcha LUB instrukcji dla GPT

CO MOŻNA AUTO-NAPRAWIĆ (bez Claude):
├─ Stuffing → zamiana na synonimy
├─ Burstiness za niski → podział długich zdań
├─ Exceeded keywords → zamiana na synonimy
├─ Powtórzenia słów → zamiana na synonimy
├─ Brakujące przecinki → dodanie (LanguageTool)
└─ Literówki → korekta (LanguageTool)

CO WYMAGA CLAUDE:
├─ Nienaturalne konstrukcje → przepisanie
├─ Brak spójności z poprzednim batchem → przepisanie
├─ Zbyt szablonowy styl AI → humanizacja
├─ Brakujące frazy (UNDER) → dodanie w kontekście
└─ Złe rozmieszczenie fraz → reorganizacja
===============================================================================
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# ================================================================
# IMPORTS
# ================================================================
try:
    from keyword_synonyms import get_synonyms, LEGAL_SYNONYMS
    SYNONYMS_AVAILABLE = True
except ImportError:
    SYNONYMS_AVAILABLE = False
    def get_synonyms(kw, max_synonyms=4): return []
    LEGAL_SYNONYMS = {}

try:
    from ai_detection_metrics import (
        calculate_burstiness,
        calculate_humanness_score,
        analyze_sentence_distribution
    )
    AI_METRICS_AVAILABLE = True
except ImportError:
    AI_METRICS_AVAILABLE = False

try:
    from grammar_middleware import validate_batch_grammar
    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False

try:
    from unified_validator import auto_fix_burstiness
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False

try:
    from claude_reviewer import auto_fix_stuffing, review_with_claude
    CLAUDE_REVIEWER_AVAILABLE = True
except ImportError:
    CLAUDE_REVIEWER_AVAILABLE = False


# ================================================================
# ENUMS & DATACLASSES
# ================================================================
class IssueSeverity(Enum):
    INFO = "INFO"           # Informacja, nie wymaga akcji
    WARNING = "WARNING"     # Ostrzeżenie, można zignorować
    ERROR = "ERROR"         # Błąd, wymaga poprawy
    CRITICAL = "CRITICAL"   # Krytyczny, blokuje batch


class IssueType(Enum):
    # Auto-fixable
    STUFFING = "STUFFING"
    BURSTINESS_LOW = "BURSTINESS_LOW"
    EXCEEDED_KEYWORD = "EXCEEDED_KEYWORD"
    WORD_REPETITION = "WORD_REPETITION"
    GRAMMAR = "GRAMMAR"
    SPELLING = "SPELLING"
    
    # Claude-fixable
    UNNATURAL_STYLE = "UNNATURAL_STYLE"
    COHERENCE = "COHERENCE"
    UNDER_KEYWORD = "UNDER_KEYWORD"
    AI_PATTERN = "AI_PATTERN"
    TEMPLATE_DETECTED = "TEMPLATE_DETECTED"
    
    # Manual only
    FACTUAL_ERROR = "FACTUAL_ERROR"
    MISSING_ENTITY = "MISSING_ENTITY"


@dataclass
class ReviewIssue:
    """Pojedynczy problem znaleziony w batchu."""
    type: IssueType
    severity: IssueSeverity
    message: str
    location: Optional[str] = None  # Fragment tekstu
    auto_fixable: bool = False
    claude_fixable: bool = False
    fix_suggestion: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    exceeded_by: int = 0  # 🆕 v37.5: Ile trzeba zamienić na synonimy


@dataclass
class ReviewResult:
    """Wynik review batcha."""
    status: str  # "APPROVED" | "AUTO_FIXED" | "NEEDS_CLAUDE" | "NEEDS_REWRITE" | "REJECTED"
    original_text: str
    fixed_text: Optional[str]
    issues: List[ReviewIssue]
    auto_fixes_applied: List[str]
    claude_fixes_needed: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "has_fixes": self.fixed_text is not None and self.fixed_text != self.original_text,
            "issues_count": len(self.issues),
            "auto_fixes_applied": self.auto_fixes_applied,
            "claude_fixes_needed": self.claude_fixes_needed,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "issues": [
                {
                    "type": i.type.value,
                    "severity": i.severity.value,
                    "message": i.message,
                    "auto_fixable": i.auto_fixable,
                    "claude_fixable": i.claude_fixable,
                    "fix_suggestion": i.fix_suggestion,
                    "synonyms": i.synonyms,
                    "exceeded_by": i.exceeded_by  # 🆕 v37.5
                }
                for i in self.issues
            ]
        }


# ================================================================
# GŁÓWNA FUNKCJA REVIEW
# ================================================================
def review_batch_comprehensive(
    batch_text: str,
    keywords_state: Dict[str, Dict],
    batch_counts: Dict[str, int],
    previous_batch_text: Optional[str] = None,
    auto_fix: bool = True,
    use_claude_for_complex: bool = False
) -> ReviewResult:
    """
    🔍 Kompleksowy review batcha z auto-poprawkami.
    
    Args:
        batch_text: Tekst batcha do sprawdzenia
        keywords_state: Stan wszystkich keywords z target_min/max
        batch_counts: Ile razy każda fraza występuje w tym batchu
        previous_batch_text: Tekst poprzedniego batcha (dla spójności)
        auto_fix: Czy automatycznie naprawiać proste problemy
        use_claude_for_complex: Czy użyć Claude dla złożonych problemów
        
    Returns:
        ReviewResult z wynikiem review i opcjonalnie naprawionym tekstem
    """
    issues: List[ReviewIssue] = []
    auto_fixes_applied: List[str] = []
    claude_fixes_needed: List[str] = []
    
    current_text = batch_text
    
    # ================================================================
    # 1. METRYKI PRZED
    # ================================================================
    metrics_before = _calculate_metrics(current_text)
    
    # ================================================================
    # 2. SPRAWDZENIE EXCEEDED KEYWORDS
    # ================================================================
    exceeded_issues = _check_exceeded_keywords(keywords_state, batch_counts)
    issues.extend(exceeded_issues)
    
    # Auto-fix exceeded przez zamianę na synonimy
    # 🆕 v37.5: Zamienia DOKŁADNIE tyle ile przekroczono (nie hardcoded 1-2)
    if auto_fix:
        for issue in exceeded_issues:
            if issue.auto_fixable and issue.synonyms and issue.exceeded_by > 0:
                # Zamień dokładnie tyle wystąpień ile trzeba żeby wrócić do limitu
                replacements_needed = issue.exceeded_by
                current_text, fix_applied = _replace_with_synonym(
                    current_text, 
                    issue.location,  # keyword
                    issue.synonyms,
                    max_replacements=replacements_needed  # 🆕 v37.5: Dynamiczne!
                )
                if fix_applied:
                    auto_fixes_applied.append(fix_applied)
    
    # ================================================================
    # 3. SPRAWDZENIE STUFFINGU (za dużo w jednym akapicie)
    # ================================================================
    stuffing_issues = _check_stuffing(current_text, keywords_state)
    issues.extend(stuffing_issues)
    
    if auto_fix and CLAUDE_REVIEWER_AVAILABLE:
        stuffed_keywords = [
            {"keyword": i.location, "count": 5, "limit": 3}
            for i in stuffing_issues if i.auto_fixable
        ]
        if stuffed_keywords:
            current_text, stuffing_fixes = auto_fix_stuffing(current_text, stuffed_keywords)
            auto_fixes_applied.extend(stuffing_fixes)
    
    # ================================================================
    # 4. SPRAWDZENIE BURSTINESS
    # ================================================================
    burstiness_issues = _check_burstiness(current_text)
    issues.extend(burstiness_issues)
    
    if auto_fix and VALIDATOR_AVAILABLE:
        for issue in burstiness_issues:
            if issue.auto_fixable and issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]:
                current_text, burstiness_fixes, stats = auto_fix_burstiness(current_text)
                auto_fixes_applied.extend(burstiness_fixes)
                break
    
    # ================================================================
    # 5. SPRAWDZENIE GRAMATYKI (LanguageTool)
    # ================================================================
    if GRAMMAR_AVAILABLE:
        grammar_issues = _check_grammar(current_text)
        issues.extend(grammar_issues)
        
        # Auto-fix prostych błędów gramatycznych
        if auto_fix:
            for issue in grammar_issues:
                if issue.auto_fixable and issue.fix_suggestion:
                    current_text = current_text.replace(issue.location, issue.fix_suggestion)
                    auto_fixes_applied.append(f"Gramatyka: '{issue.location}' → '{issue.fix_suggestion}'")
    
    # ================================================================
    # 6. SPRAWDZENIE WZORCÓW AI
    # ================================================================
    ai_pattern_issues = _check_ai_patterns(current_text)
    issues.extend(ai_pattern_issues)
    
    for issue in ai_pattern_issues:
        if issue.claude_fixable:
            claude_fixes_needed.append(issue.message)
    
    # ================================================================
    # 7. SPRAWDZENIE SPÓJNOŚCI Z POPRZEDNIM BATCHEM
    # ================================================================
    if previous_batch_text:
        coherence_issues = _check_coherence(current_text, previous_batch_text)
        issues.extend(coherence_issues)
        
        for issue in coherence_issues:
            if issue.claude_fixable:
                claude_fixes_needed.append(issue.message)
    
    # ================================================================
    # 8. SPRAWDZENIE UNDER KEYWORDS (za mało użyć)
    # ================================================================
    under_issues = _check_under_keywords(keywords_state, batch_counts)
    issues.extend(under_issues)
    
    for issue in under_issues:
        if issue.severity == IssueSeverity.WARNING:
            claude_fixes_needed.append(issue.message)
    
    # ================================================================
    # 9. METRYKI PO
    # ================================================================
    metrics_after = _calculate_metrics(current_text)
    
    # ================================================================
    # 10. OKREŚL STATUS
    # ================================================================
    has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
    has_errors = any(i.severity == IssueSeverity.ERROR for i in issues)
    has_unfixed_errors = any(
        i.severity == IssueSeverity.ERROR and not i.auto_fixable 
        for i in issues
    )
    
    if has_critical:
        status = "REJECTED"
    elif claude_fixes_needed and use_claude_for_complex:
        status = "NEEDS_CLAUDE"
    elif has_unfixed_errors:
        status = "NEEDS_REWRITE"
    elif auto_fixes_applied:
        status = "AUTO_FIXED"
    else:
        status = "APPROVED"
    
    return ReviewResult(
        status=status,
        original_text=batch_text,
        fixed_text=current_text if current_text != batch_text else None,
        issues=issues,
        auto_fixes_applied=auto_fixes_applied,
        claude_fixes_needed=claude_fixes_needed,
        metrics_before=metrics_before,
        metrics_after=metrics_after
    )


# ================================================================
# FUNKCJE SPRAWDZAJĄCE
# ================================================================
def _check_exceeded_keywords(
    keywords_state: Dict[str, Dict],
    batch_counts: Dict[str, int]
) -> List[ReviewIssue]:
    """Sprawdza czy frazy BASIC przekroczyły limity."""
    issues = []
    
    for rid, count in batch_counts.items():
        if count == 0:
            continue
            
        meta = keywords_state.get(rid, {})
        kw_type = meta.get("type", "BASIC").upper()
        
        # Tylko BASIC i MAIN
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        keyword = meta.get("keyword", "")
        current = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        new_total = current + count
        
        if new_total > target_max:
            exceeded_by = new_total - target_max
            exceed_pct = (exceeded_by / target_max * 100) if target_max > 0 else 100
            
            synonyms = get_synonyms(keyword) if SYNONYMS_AVAILABLE else []
            
            if exceed_pct >= 50:
                severity = IssueSeverity.CRITICAL
                auto_fixable = bool(synonyms)  # Można naprawić jeśli mamy synonimy
            else:
                severity = IssueSeverity.WARNING
                auto_fixable = bool(synonyms)
            
            issues.append(ReviewIssue(
                type=IssueType.EXCEEDED_KEYWORD,
                severity=severity,
                message=f"'{keyword}' exceeded: {new_total}/{target_max} (+{exceed_pct:.0f}%)",
                location=keyword,
                auto_fixable=auto_fixable,
                fix_suggestion=f"Zamień {exceeded_by}× na: {', '.join(synonyms[:2])}" if synonyms else None,
                synonyms=synonyms[:4],
                exceeded_by=exceeded_by  # 🆕 v37.5: Ile dokładnie trzeba zamienić
            ))
    
    return issues


def _check_stuffing(
    text: str,
    keywords_state: Dict[str, Dict]
) -> List[ReviewIssue]:
    """Sprawdza stuffing (za dużo fraz w jednym akapicie)."""
    issues = []
    paragraphs = text.split('\n\n')
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        if not keyword or len(keyword) < 3:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        for para in paragraphs:
            count = para.lower().count(keyword)
            if count > 3:
                synonyms = get_synonyms(keyword) if SYNONYMS_AVAILABLE else []
                
                issues.append(ReviewIssue(
                    type=IssueType.STUFFING,
                    severity=IssueSeverity.ERROR,
                    message=f"Stuffing: '{meta.get('keyword')}' występuje {count}× w jednym akapicie",
                    location=meta.get("keyword"),
                    auto_fixable=bool(synonyms),
                    fix_suggestion=f"Zamień część na: {', '.join(synonyms[:2])}" if synonyms else None,
                    synonyms=synonyms[:4]
                ))
                break
    
    return issues


def _check_burstiness(text: str) -> List[ReviewIssue]:
    """Sprawdza burstiness (zmienność zdań)."""
    issues = []
    
    if not AI_METRICS_AVAILABLE:
        return issues
    
    result = calculate_burstiness(text)
    cv = result.get("cv", 0)
    burstiness = result.get("value", 0)
    
    if cv < 0.26:
        issues.append(ReviewIssue(
            type=IssueType.BURSTINESS_LOW,
            severity=IssueSeverity.CRITICAL,
            message=f"Burstiness CRITICAL: CV={cv:.2f} (<0.26 = sygnał AI). Dodaj krótkie zdania 2-10 słów.",
            auto_fixable=VALIDATOR_AVAILABLE,
            fix_suggestion="Podziel długie zdania lub dodaj krótkie wtrącenia"
        ))
    elif cv < 0.36:
        issues.append(ReviewIssue(
            type=IssueType.BURSTINESS_LOW,
            severity=IssueSeverity.WARNING,
            message=f"Burstiness WARNING: CV={cv:.2f} (<0.36). Rozważ więcej krótkich zdań.",
            auto_fixable=VALIDATOR_AVAILABLE,
            fix_suggestion="Dodaj dynamiczne, krótkie zdania"
        ))
    
    return issues


def _check_grammar(text: str) -> List[ReviewIssue]:
    """Sprawdza gramatykę (LanguageTool)."""
    issues = []
    
    if not GRAMMAR_AVAILABLE:
        return issues
    
    try:
        result = validate_batch_grammar(text)
        for error in result.get("errors", [])[:5]:  # Max 5 błędów
            issues.append(ReviewIssue(
                type=IssueType.GRAMMAR,
                severity=IssueSeverity.WARNING,
                message=error.get("message", "Błąd gramatyczny"),
                location=error.get("context", ""),
                auto_fixable=bool(error.get("replacements")),
                fix_suggestion=error.get("replacements", [None])[0]
            ))
    except:
        pass
    
    return issues


def _check_ai_patterns(text: str) -> List[ReviewIssue]:
    """Sprawdza wzorce typowe dla AI."""
    issues = []
    
    # Wzorce AI do wykrycia
    AI_PATTERNS = [
        (r'\bWarto\s+(zauważyć|podkreślić|wspomnieć)\b', "Szablonowe 'Warto zauważyć'"),
        (r'\bNależy\s+pamiętać\b', "Szablonowe 'Należy pamiętać'"),
        (r'\bIstotne\s+jest\b', "Szablonowe 'Istotne jest'"),
        (r'\bW\s+kontekście\b', "Nadużyte 'W kontekście'"),
        (r'\bNa\s+przestrzeni\s+lat\b', "Nadużyte 'Na przestrzeni lat'"),
        (r'\bW\s+dzisiejszych\s+czasach\b', "Banał 'W dzisiejszych czasach'"),
        (r'\bPodsumowując\s*,', "Szablonowe 'Podsumowując'"),
        (r'\bBiorąc\s+pod\s+uwagę\b', "Nadużyte 'Biorąc pod uwagę'"),
    ]
    
    for pattern, description in AI_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) >= 2:  # Wykryj jeśli 2+ wystąpienia
            issues.append(ReviewIssue(
                type=IssueType.AI_PATTERN,
                severity=IssueSeverity.WARNING,
                message=f"Wzorzec AI: {description} ({len(matches)}×)",
                location=matches[0] if matches else "",
                claude_fixable=True,
                fix_suggestion="Przepisz na bardziej naturalny styl"
            ))
    
    # Sprawdź monotonię zdań (koncentracja w przedziale 15-22 słów)
    if AI_METRICS_AVAILABLE:
        try:
            dist = analyze_sentence_distribution(text)
            ai_concentration = dist.get("ai_concentration", 0)
            if ai_concentration > 60:
                issues.append(ReviewIssue(
                    type=IssueType.AI_PATTERN,
                    severity=IssueSeverity.WARNING,
                    message=f"Monotonia AI: {ai_concentration:.0f}% zdań w przedziale 15-22 słów",
                    claude_fixable=True,
                    fix_suggestion="Zróżnicuj długość zdań (krótkie, średnie, długie)"
                ))
        except:
            pass
    
    return issues


def _check_coherence(
    current_text: str,
    previous_text: str
) -> List[ReviewIssue]:
    """Sprawdza spójność z poprzednim batchem."""
    issues = []
    
    # Pobierz ostatnie zdanie poprzedniego batcha
    prev_sentences = previous_text.split('.')
    last_prev = prev_sentences[-2] if len(prev_sentences) > 1 else prev_sentences[-1]
    
    # Pobierz pierwsze zdanie obecnego batcha
    curr_sentences = current_text.split('.')
    first_curr = curr_sentences[0] if curr_sentences else ""
    
    # Sprawdź czy jest jakiekolwiek połączenie
    transition_words = [
        "jednak", "natomiast", "ponadto", "dodatkowo", "co więcej",
        "w związku z", "dlatego", "zatem", "tym samym", "jednocześnie"
    ]
    
    has_transition = any(tw in first_curr.lower() for tw in transition_words)
    
    # Jeśli poprzedni batch kończy się pytaniem, sprawdź czy odpowiadamy
    if last_prev.strip().endswith('?'):
        if not any(word in first_curr.lower() for word in ["odpowiedź", "tak", "nie", "to zależy", "przede wszystkim"]):
            issues.append(ReviewIssue(
                type=IssueType.COHERENCE,
                severity=IssueSeverity.WARNING,
                message="Poprzedni batch kończy się pytaniem - rozważ bezpośrednią odpowiedź",
                claude_fixable=True,
                fix_suggestion="Zacznij od odpowiedzi na pytanie"
            ))
    
    return issues


def _check_under_keywords(
    keywords_state: Dict[str, Dict],
    batch_counts: Dict[str, int]
) -> List[ReviewIssue]:
    """Sprawdza frazy które są UNDER (za mało użyć)."""
    issues = []
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        keyword = meta.get("keyword", "")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        status = meta.get("status", "UNDER")
        
        # Jeśli status to UNDER i batch nie dodał żadnego użycia
        if status == "UNDER" and batch_counts.get(rid, 0) == 0:
            remaining = target_min - actual
            if remaining > 0:
                issues.append(ReviewIssue(
                    type=IssueType.UNDER_KEYWORD,
                    severity=IssueSeverity.INFO,
                    message=f"'{keyword}' UNDER: {actual}/{target_min} - rozważ dodanie {remaining}× więcej",
                    location=keyword,
                    claude_fixable=True,
                    fix_suggestion=f"Dodaj frazę '{keyword}' w naturalnym kontekście"
                ))
    
    return issues


# ================================================================
# FUNKCJE POMOCNICZE
# ================================================================
def _calculate_metrics(text: str) -> Dict[str, Any]:
    """Oblicza metryki dla tekstu."""
    metrics = {
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[.!?]+', text))
    }
    
    if AI_METRICS_AVAILABLE:
        try:
            burstiness = calculate_burstiness(text)
            metrics["burstiness"] = burstiness.get("value", 0)
            metrics["cv"] = burstiness.get("cv", 0)
            
            humanness = calculate_humanness_score(text)
            metrics["humanness_score"] = humanness.get("humanness_score", 0)
        except:
            pass
    
    return metrics


def _replace_with_synonym(
    text: str,
    keyword: str,
    synonyms: List[str],
    max_replacements: int = 1
) -> Tuple[str, Optional[str]]:
    """Zamienia keyword na synonim w tekście."""
    if not synonyms or not keyword:
        return text, None
    
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    if not matches:
        return text, None
    
    # Zamień od końca (żeby indeksy się nie przesunęły)
    replacements_made = 0
    fixed_text = text
    
    for match in reversed(matches):
        if replacements_made >= max_replacements:
            break
        
        start, end = match.start(), match.end()
        original = fixed_text[start:end]
        synonym = synonyms[replacements_made % len(synonyms)]
        
        # Zachowaj wielkość liter
        if original[0].isupper():
            replacement = synonym[0].upper() + synonym[1:]
        else:
            replacement = synonym
        
        fixed_text = fixed_text[:start] + replacement + fixed_text[end:]
        replacements_made += 1
    
    if replacements_made > 0:
        return fixed_text, f"Zamieniono '{keyword}' → '{synonyms[0]}' ({replacements_made}×)"
    
    return text, None


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================
def get_review_summary(result: ReviewResult) -> str:
    """Generuje czytelne podsumowanie review."""
    lines = []
    lines.append(f"📋 REVIEW STATUS: {result.status}")
    lines.append(f"   Issues: {len(result.issues)}")
    
    if result.auto_fixes_applied:
        lines.append(f"\n✅ AUTO-FIXED ({len(result.auto_fixes_applied)}):")
        for fix in result.auto_fixes_applied[:5]:
            lines.append(f"   • {fix}")
    
    if result.claude_fixes_needed:
        lines.append(f"\n🤖 NEEDS CLAUDE ({len(result.claude_fixes_needed)}):")
        for fix in result.claude_fixes_needed[:5]:
            lines.append(f"   • {fix}")
    
    critical = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL]
    if critical:
        lines.append(f"\n❌ CRITICAL ({len(critical)}):")
        for issue in critical:
            lines.append(f"   • {issue.message}")
    
    if result.metrics_before and result.metrics_after:
        lines.append(f"\n📊 METRICS:")
        for key in ["burstiness", "cv", "humanness_score"]:
            before = result.metrics_before.get(key, "?")
            after = result.metrics_after.get(key, "?")
            if before != after:
                lines.append(f"   {key}: {before} → {after}")
    
    return "\n".join(lines)


def generate_claude_fix_prompt(result: ReviewResult) -> str:
    """Generuje prompt dla Claude do naprawy batcha."""
    if not result.claude_fixes_needed:
        return ""
    
    prompt_lines = [
        "# ZADANIE: Napraw poniższe problemy w tekście",
        "",
        "## PROBLEMY DO NAPRAWY:",
    ]
    
    for i, fix in enumerate(result.claude_fixes_needed, 1):
        prompt_lines.append(f"{i}. {fix}")
    
    prompt_lines.extend([
        "",
        "## ZASADY:",
        "- Zachowaj sens i strukturę tekstu",
        "- Popraw tylko wskazane problemy",
        "- Użyj naturalnego, ludzkiego stylu",
        "- Zróżnicuj długość zdań (2-10, 12-18, 20-30 słów)",
        "",
        "## TEKST DO POPRAWY:",
        "```",
        result.original_text if not result.fixed_text else result.fixed_text,
        "```",
        "",
        "Zwróć TYLKO poprawiony tekst, bez komentarzy."
    ])
    
    return "\n".join(prompt_lines)


# ================================================================
# 🆕 v37.1: CLAUDE SMART-FIX z PRE_BATCH_INFO
# ================================================================
@dataclass
class SmartFixResult:
    """Wynik Claude Smart-Fix."""
    success: bool
    original_text: str
    fixed_text: Optional[str]
    changes_made: List[str]
    keywords_added: List[str]
    keywords_replaced: List[Dict[str, str]]  # {original: replacement}
    prompt_used: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "has_changes": self.fixed_text is not None and self.fixed_text != self.original_text,
            "changes_made": self.changes_made,
            "keywords_added": self.keywords_added,
            "keywords_replaced": self.keywords_replaced,
            "error": self.error
        }


def build_smart_fix_prompt(
    batch_text: str,
    pre_batch_info: Dict,
    review_result: ReviewResult,
    keywords_state: Dict[str, Dict]
) -> str:
    """
    🆕 v37.1: Buduje inteligentny prompt dla Claude z pełnym kontekstem.
    
    Claude dostaje:
    - Kontekst sekcji (H2, poprzedni batch)
    - Które frazy DODAĆ (UNDER)
    - Które frazy ZAMIENIĆ (EXCEEDED) + synonimy
    - Których fraz NIE UŻYWAĆ (RESERVED)
    - Konkretne problemy do naprawy
    """
    sections = []
    
    # ================================================================
    # NAGŁÓWEK
    # ================================================================
    sections.append("=" * 70)
    sections.append("🔧 CLAUDE SMART-FIX - INTELIGENTNA POPRAWA BATCHA")
    sections.append("=" * 70)
    sections.append("")
    sections.append("ZADANIE: Popraw tekst zachowując ~95% treści. Wprowadź MINIMALNE zmiany.")
    sections.append("")
    
    # ================================================================
    # KONTEKST SEKCJI
    # ================================================================
    sections.append("-" * 70)
    sections.append("📌 KONTEKST SEKCJI")
    sections.append("-" * 70)
    
    # H2
    h2_section = pre_batch_info.get("h2_section") or pre_batch_info.get("current_h2", "")
    if h2_section:
        sections.append(f"Sekcja H2: \"{h2_section}\"")
    
    # Batch number
    batch_num = pre_batch_info.get("batch_number", pre_batch_info.get("current_batch_num", "?"))
    total_batches = pre_batch_info.get("total_batches", pre_batch_info.get("total_planned_batches", "?"))
    sections.append(f"Batch: {batch_num} z {total_batches}")
    
    # Poprzedni batch
    last_sentences = pre_batch_info.get("last_sentences", "")
    if last_sentences:
        sections.append(f"\nPoprzedni batch kończył się:")
        sections.append(f"  \"{last_sentences[:200]}{'...' if len(last_sentences) > 200 else ''}\"")
        sections.append("  → Zadbaj o płynne przejście!")
    
    sections.append("")
    
    # ================================================================
    # PROBLEMY DO NAPRAWY
    # ================================================================
    if review_result.issues or review_result.claude_fixes_needed:
        sections.append("-" * 70)
        sections.append("❌ PROBLEMY DO NAPRAWY")
        sections.append("-" * 70)
        
        # Z review_result
        for i, issue in enumerate(review_result.issues[:10], 1):
            if issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]:
                sections.append(f"{i}. [{issue.type.value}] {issue.message}")
                if issue.fix_suggestion:
                    sections.append(f"   → {issue.fix_suggestion}")
        
        # Dodatkowe z claude_fixes_needed
        for fix in review_result.claude_fixes_needed[:5]:
            if fix not in [i.message for i in review_result.issues]:
                sections.append(f"• {fix}")
        
        sections.append("")
    
    # ================================================================
    # FRAZY DO DODANIA (UNDER)
    # ================================================================
    under_keywords = _get_under_keywords(keywords_state, pre_batch_info)
    
    if under_keywords:
        sections.append("-" * 70)
        sections.append("➕ FRAZY DO DODANIA (UNDER - brakuje w tekście)")
        sections.append("-" * 70)
        sections.append("Dodaj te frazy w NATURALNYCH miejscach (1× każdą):")
        sections.append("")
        
        for kw_info in under_keywords[:8]:
            keyword = kw_info["keyword"]
            actual = kw_info["actual"]
            target_min = kw_info["target_min"]
            context_hint = kw_info.get("context_hint", "")
            
            sections.append(f"  ➕ \"{keyword}\" (obecnie: {actual}×, min: {target_min}×)")
            if context_hint:
                sections.append(f"      Pasuje do: {context_hint}")
        
        sections.append("")
    
    # ================================================================
    # FRAZY DO ZAMIANY (EXCEEDED)
    # ================================================================
    exceeded_keywords = _get_exceeded_keywords_with_synonyms(keywords_state, pre_batch_info)
    
    if exceeded_keywords:
        sections.append("-" * 70)
        sections.append("🔄 FRAZY DO ZAMIANY (EXCEEDED - za dużo)")
        sections.append("-" * 70)
        sections.append("Zamień CZĘŚĆ wystąpień na podane synonimy:")
        sections.append("")
        
        for kw_info in exceeded_keywords[:6]:
            keyword = kw_info["keyword"]
            actual = kw_info["actual"]
            target_max = kw_info["target_max"]
            to_replace = kw_info["to_replace"]
            synonyms = kw_info["synonyms"]
            
            sections.append(f"  🔄 \"{keyword}\" ({actual}×, max: {target_max}×) → zamień {to_replace}×")
            if synonyms:
                sections.append(f"      Użyj: {', '.join(synonyms[:3])}")
        
        sections.append("")
    
    # ================================================================
    # FRAZY ZAREZERWOWANE (NIE UŻYWAĆ!)
    # ================================================================
    reserved = pre_batch_info.get("reserved_keywords", [])
    if not reserved:
        # Spróbuj z semantic_batch_plan
        semantic_plan = pre_batch_info.get("semantic_batch_plan", {})
        reserved = semantic_plan.get("reserved_keywords", [])
    
    if reserved:
        sections.append("-" * 70)
        sections.append("🚫 FRAZY ZAREZERWOWANE (NIE DODAWAJ!)")
        sections.append("-" * 70)
        sections.append("Te frazy są przeznaczone na INNE batche:")
        sections.append("")
        
        for rk in reserved[:10]:
            if isinstance(rk, dict):
                kw = rk.get("keyword", "")
                reserved_for = rk.get("reserved_for_h2", rk.get("reserved_for_batch", ""))
                sections.append(f"  🚫 \"{kw}\" → zarezerwowane dla: {reserved_for}")
            else:
                sections.append(f"  🚫 \"{rk}\"")
        
        sections.append("")
    
    # ================================================================
    # LIMITY FRAZ
    # ================================================================
    keyword_limits = pre_batch_info.get("keyword_limits", [])
    if keyword_limits:
        # Pokaż tylko STOP i OSTROŻNIE
        stop_keywords = [k for k in keyword_limits if k.get("status") == "🛑 STOP"]
        caution_keywords = [k for k in keyword_limits if k.get("status") == "⚠️ OSTROŻNIE"]
        
        if stop_keywords or caution_keywords:
            sections.append("-" * 70)
            sections.append("📊 LIMITY FRAZ")
            sections.append("-" * 70)
            
            if stop_keywords:
                sections.append("🛑 STOP (limit wyczerpany - NIE UŻYWAJ):")
                for kw in stop_keywords[:8]:
                    sections.append(f"  ❌ \"{kw['keyword']}\" = {kw['actual']}/{kw['target_max']}")
            
            if caution_keywords:
                sections.append("⚠️ OSTROŻNIE (bliskie limitu):")
                for kw in caution_keywords[:8]:
                    sections.append(f"  ⚠️ \"{kw['keyword']}\" = {kw['actual']}/{kw['target_max']} → max {kw.get('max_this_batch', 1)}× więcej")
            
            sections.append("")
    
    # ================================================================
    # ZASADY
    # ================================================================
    sections.append("-" * 70)
    sections.append("📋 ZASADY POPRAWY")
    sections.append("-" * 70)
    sections.append("""
1. ZACHOWAJ ~95% oryginalnego tekstu - minimalne zmiany!
2. DODAJ brakujące frazy (UNDER) w naturalnych miejscach
3. ZAMIEŃ exceeded frazy na podane synonimy
4. USUŃ szablonowe wyrażenia AI ("Warto zauważyć", "Istotne jest")
5. ZRÓŻNICUJ długość zdań (krótkie 2-10, średnie 12-18, długie 20-30 słów)
6. ZACHOWAJ spójność z poprzednim batchem
7. NIE DODAWAJ fraz zarezerwowanych na inne batche
""")
    
    # ================================================================
    # TEKST DO POPRAWY
    # ================================================================
    sections.append("-" * 70)
    sections.append("📝 TEKST DO POPRAWY")
    sections.append("-" * 70)
    sections.append("```")
    sections.append(batch_text)
    sections.append("```")
    sections.append("")
    sections.append("-" * 70)
    sections.append("Zwróć TYLKO poprawiony tekst, bez komentarzy ani wyjaśnień.")
    sections.append("-" * 70)
    
    return "\n".join(sections)


def _get_under_keywords(
    keywords_state: Dict[str, Dict],
    pre_batch_info: Dict
) -> List[Dict]:
    """Pobiera frazy UNDER (niedostatecznie użyte) przypisane do tego batcha."""
    under = []
    
    # Frazy przypisane do tego batcha z semantic_plan
    assigned = []
    semantic_plan = pre_batch_info.get("semantic_batch_plan", {})
    if semantic_plan:
        assigned = semantic_plan.get("assigned_keywords", [])
    
    # Frazy MUST USE z pre_batch_info
    basic_must_use = pre_batch_info.get("basic_must_use", [])
    for item in basic_must_use:
        if isinstance(item, dict):
            assigned.append(item.get("keyword", ""))
        else:
            assigned.append(str(item))
    
    h2_section = pre_batch_info.get("h2_section", "")
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        status = meta.get("status", "UNDER")
        
        # Czy ta fraza jest UNDER i przypisana do tego batcha?
        is_assigned = keyword.lower() in [a.lower() for a in assigned]
        is_under = actual < target_min or status == "UNDER"
        
        if is_under and (is_assigned or actual == 0):
            context_hint = ""
            # Sprawdź czy pasuje do H2
            if h2_section:
                kw_words = set(keyword.lower().split())
                h2_words = set(h2_section.lower().split())
                if kw_words & h2_words:
                    context_hint = f"powiązane z H2 '{h2_section}'"
            
            under.append({
                "keyword": keyword,
                "actual": actual,
                "target_min": target_min,
                "target_max": meta.get("target_max", 10),
                "context_hint": context_hint,
                "is_assigned": is_assigned
            })
    
    # Sortuj: przypisane najpierw, potem te z actual=0
    under.sort(key=lambda x: (not x["is_assigned"], x["actual"]))
    
    return under[:10]


def _get_exceeded_keywords_with_synonyms(
    keywords_state: Dict[str, Dict],
    pre_batch_info: Dict
) -> List[Dict]:
    """Pobiera frazy EXCEEDED z synonimami."""
    exceeded = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type not in ["BASIC", "MAIN"]:
            continue
        
        actual = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        
        if actual > target_max:
            exceeded_by = actual - target_max
            
            # Pobierz synonimy
            synonyms = get_synonyms(keyword) if SYNONYMS_AVAILABLE else []
            
            exceeded.append({
                "keyword": keyword,
                "actual": actual,
                "target_max": target_max,
                "exceeded_by": exceeded_by,
                "to_replace": exceeded_by,  # 🆕 v37.5: Bez sztucznego limitu
                "synonyms": synonyms[:4]
            })
    
    # Sortuj po exceeded_by (najwięcej przekroczone najpierw)
    exceeded.sort(key=lambda x: -x["exceeded_by"])
    
    return exceeded[:8]


def claude_smart_fix(
    batch_text: str,
    pre_batch_info: Dict,
    review_result: ReviewResult,
    keywords_state: Dict[str, Dict],
    api_key: Optional[str] = None
) -> SmartFixResult:
    """
    🆕 v37.1: Claude inteligentnie poprawia batch z pełnym kontekstem.
    
    Args:
        batch_text: Tekst do poprawy
        pre_batch_info: Pełne info o batchu (H2, limity, poprzedni batch)
        review_result: Wynik review z wykrytymi problemami
        keywords_state: Stan wszystkich fraz
        api_key: Opcjonalny API key (domyślnie z env)
        
    Returns:
        SmartFixResult z poprawionym tekstem
    """
    import os
    
    # Zbuduj prompt
    prompt = build_smart_fix_prompt(
        batch_text=batch_text,
        pre_batch_info=pre_batch_info,
        review_result=review_result,
        keywords_state=keywords_state
    )
    
    # Sprawdź czy mamy Anthropic
    try:
        import anthropic
    except ImportError:
        return SmartFixResult(
            success=False,
            original_text=batch_text,
            fixed_text=None,
            changes_made=[],
            keywords_added=[],
            keywords_replaced=[],
            prompt_used=prompt,
            error="anthropic library not installed"
        )
    
    # API key
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return SmartFixResult(
            success=False,
            original_text=batch_text,
            fixed_text=None,
            changes_made=[],
            keywords_added=[],
            keywords_replaced=[],
            prompt_used=prompt,
            error="ANTHROPIC_API_KEY not set"
        )
    
    try:
        client = anthropic.Anthropic(api_key=key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        fixed_text = message.content[0].text.strip()
        
        # Wyczyść z markdown jeśli Claude dodał
        if fixed_text.startswith("```"):
            lines = fixed_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            fixed_text = "\n".join(lines)
        
        # Analizuj co się zmieniło
        changes_made = []
        keywords_added = []
        keywords_replaced = []
        
        # Sprawdź dodane frazy
        under_keywords = _get_under_keywords(keywords_state, pre_batch_info)
        for kw_info in under_keywords:
            keyword = kw_info["keyword"]
            if keyword.lower() in fixed_text.lower() and keyword.lower() not in batch_text.lower():
                keywords_added.append(keyword)
                changes_made.append(f"Dodano frazę: '{keyword}'")
        
        # Sprawdź zamienione frazy
        exceeded_keywords = _get_exceeded_keywords_with_synonyms(keywords_state, pre_batch_info)
        for kw_info in exceeded_keywords:
            keyword = kw_info["keyword"]
            original_count = batch_text.lower().count(keyword.lower())
            new_count = fixed_text.lower().count(keyword.lower())
            
            if new_count < original_count:
                replaced = original_count - new_count
                # Sprawdź które synonimy użyto
                for syn in kw_info.get("synonyms", []):
                    if syn.lower() in fixed_text.lower():
                        keywords_replaced.append({
                            "original": keyword,
                            "replacement": syn,
                            "count": replaced
                        })
                        changes_made.append(f"Zamieniono '{keyword}' → '{syn}' ({replaced}×)")
                        break
        
        return SmartFixResult(
            success=True,
            original_text=batch_text,
            fixed_text=fixed_text,
            changes_made=changes_made,
            keywords_added=keywords_added,
            keywords_replaced=keywords_replaced,
            prompt_used=prompt
        )
        
    except Exception as e:
        return SmartFixResult(
            success=False,
            original_text=batch_text,
            fixed_text=None,
            changes_made=[],
            keywords_added=[],
            keywords_replaced=[],
            prompt_used=prompt,
            error=str(e)
        )


def should_use_claude_smart_fix(review_result: ReviewResult) -> bool:
    """Decyduje czy warto użyć Claude Smart-Fix."""
    # Użyj Claude jeśli:
    # 1. Są problemy wymagające Claude
    if review_result.claude_fixes_needed:
        return True
    
    # 2. Są CRITICAL/ERROR issues które nie zostały auto-naprawione
    unfixed_serious = [
        i for i in review_result.issues 
        if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]
        and not i.auto_fixable
    ]
    if unfixed_serious:
        return True
    
    # 3. Są problemy z wzorcami AI
    ai_patterns = [i for i in review_result.issues if i.type == IssueType.AI_PATTERN]
    if len(ai_patterns) >= 2:
        return True
    
    return False


def get_pre_batch_info_for_claude(
    project_data: Dict,
    batch_number: int
) -> Dict:
    """
    🆕 v37.1: Przygotowuje pre_batch_info dla Claude Smart-Fix.
    
    Wyciąga z project_data wszystkie informacje potrzebne Claude:
    - Kontekst H2
    - Poprzedni batch (last_sentences)
    - Limity fraz
    - Frazy assigned/reserved
    """
    keywords_state = project_data.get("keywords_state", {})
    batches = project_data.get("batches", [])
    h2_structure = project_data.get("h2_structure", [])
    total_planned_batches = project_data.get("total_planned_batches", 4)
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    semantic_plan = project_data.get("semantic_keyword_plan", {})
    
    # ================================================================
    # 1. KONTEKST BATCHA
    # ================================================================
    remaining_batches = max(1, total_planned_batches - batch_number + 1)
    
    # Batch type
    if batch_number == 1:
        batch_type = "INTRO"
    elif batch_number >= total_planned_batches:
        batch_type = "FINAL"
    else:
        batch_type = "CONTENT"
    
    # ================================================================
    # 2. H2 SECTION
    # ================================================================
    # Znajdź które H2 już użyte
    used_h2 = []
    for batch in batches[:batch_number-1]:
        batch_text = batch.get("text", "")
        import re
        h2_in_batch = re.findall(r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)', batch_text, re.MULTILINE | re.IGNORECASE)
        used_h2.extend([(m[0] or m[1]).strip() for m in h2_in_batch if m[0] or m[1]])
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    current_h2 = remaining_h2[0] if remaining_h2 else main_keyword
    
    # ================================================================
    # 3. LAST SENTENCES (poprzedni batch)
    # ================================================================
    last_sentences = ""
    if batch_number > 1 and len(batches) >= batch_number - 1:
        prev_batch_idx = batch_number - 2  # -2 bo indeksowane od 0 i chcemy poprzedni
        if prev_batch_idx >= 0 and prev_batch_idx < len(batches):
            last_batch_text = batches[prev_batch_idx].get("text", "")
            import re
            clean_last = re.sub(r'<[^>]+>', '', last_batch_text)
            clean_last = re.sub(r'^h[23]:\s*.+$', '', clean_last, flags=re.MULTILINE)
            sentences = re.split(r'[.!?]+', clean_last)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
            if len(sentences) >= 2:
                last_sentences = ". ".join(sentences[-2:]) + "."
            elif sentences:
                last_sentences = sentences[-1] + "."
    
    # ================================================================
    # 4. SEMANTIC BATCH PLAN
    # ================================================================
    semantic_batch_plan = {}
    if semantic_plan and "batch_plans" in semantic_plan:
        for bp in semantic_plan.get("batch_plans", []):
            if bp.get("batch_number") == batch_number:
                semantic_batch_plan = bp
                break
    
    # ================================================================
    # 5. KEYWORD LIMITS
    # ================================================================
    keyword_limits = []
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        remaining_to_max = max(0, target_max - actual)
        
        # Oblicz max dla tego batcha
        if remaining_batches > 0 and remaining_to_max > 0:
            import math
            max_this_batch = math.ceil(remaining_to_max / remaining_batches)
        else:
            max_this_batch = remaining_to_max
        
        # Status
        if actual >= target_max:
            status = "🛑 STOP"
            max_this_batch = 0
        elif remaining_to_max <= 2 and remaining_to_max > 0:
            status = "⚠️ OSTROŻNIE"
        elif actual >= target_min:
            status = "✅ OK"
        else:
            status = "📌 UŻYJ"
        
        keyword_limits.append({
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "remaining": remaining_to_max,
            "max_this_batch": max_this_batch,
            "status": status
        })
    
    # Sortuj
    priority_order = {"🛑 STOP": 0, "⚠️ OSTROŻNIE": 1, "📌 UŻYJ": 2, "✅ OK": 3}
    keyword_limits.sort(key=lambda x: (priority_order.get(x["status"], 99), -x.get("actual", 0)))
    
    # ================================================================
    # 6. BASIC MUST USE (frazy które MUSZĄ być użyte w tym batchu)
    # ================================================================
    basic_must_use = []
    for kl in keyword_limits:
        if kl["type"] in ["BASIC", "MAIN"] and kl["status"] == "📌 UŻYJ":
            basic_must_use.append({
                "keyword": kl["keyword"],
                "actual": kl["actual"],
                "target_min": kl["target_min"],
                "target_max": kl["target_max"]
            })
    
    # ================================================================
    # 7. RESERVED KEYWORDS
    # ================================================================
    reserved_keywords = []
    if semantic_batch_plan:
        reserved_keywords = semantic_batch_plan.get("reserved_keywords", [])
    
    return {
        # Kontekst
        "batch_number": batch_number,
        "total_batches": total_planned_batches,
        "total_planned_batches": total_planned_batches,
        "batch_type": batch_type,
        "remaining_batches": remaining_batches,
        
        # H2
        "h2_section": current_h2,
        "current_h2": current_h2,
        "remaining_h2": remaining_h2,
        "h2_structure": h2_structure,
        
        # Poprzedni batch
        "last_sentences": last_sentences,
        
        # Semantic plan
        "semantic_batch_plan": semantic_batch_plan,
        "assigned_keywords": semantic_batch_plan.get("assigned_keywords", []),
        "universal_keywords": semantic_batch_plan.get("universal_keywords", []),
        "reserved_keywords": reserved_keywords,
        
        # Limity
        "keyword_limits": keyword_limits,
        "basic_must_use": basic_must_use,
        
        # Main keyword
        "main_keyword": main_keyword
    }


# ================================================================
