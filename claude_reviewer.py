# claude_reviewer.py
# v28.2 - Claude jako Reviewer/Editor batchy
#
# System sprawdzania i poprawiania batchy przez Claude API.
# Sprawdza: SEO, długość, powtórzenia, gramatykę, AI patterns, halucynacje

import os
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ReviewIssue:
    type: str  # seo, length, repetition, grammar, ai_pattern, hallucination, coherence
    severity: str  # critical, warning, suggestion
    description: str
    location: str = ""
    fix_applied: bool = False


@dataclass
class ReviewResult:
    status: str  # APPROVED, CORRECTED, REJECTED, QUICK_CHECK_FAILED
    original_text: str
    corrected_text: Optional[str]
    issues: List[ReviewIssue]
    summary: str
    word_count: int = 0
    paragraph_count: int = 0


# ================================================================
# QUICK CHECKS (Python, bez API)
# ================================================================
# v29.0: NOWE PRIORYTETY
# 1. JAKOŚĆ TEKSTU (tautologie, gramatyka) → CRITICAL
# 2. ENCJE + N-GRAMY w odpowiednich miejscach → WARNING  
# 3. SŁOWA KLUCZOWE: min 1×, NIE stuffing → tylko stuffing blokuje
# ================================================================

# Import lemmatyzacji dla quick checks
try:
    from polish_lemmatizer import count_phrase_occurrences
    _LEMMATIZER_OK = True
except ImportError:
    _LEMMATIZER_OK = False
    print("[CLAUDE_REVIEWER] ⚠️ polish_lemmatizer not available, using exact match")


def quick_check_keywords(text: str, required: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    v29.0: NOWE PODEJŚCIE DO FRAZ
    
    - Fraza występuje 0× → CRITICAL (musi być chociaż raz)
    - Fraza występuje 1-N× gdzie N < max → OK (nie blokuj!)
    - Fraza występuje > max×2.5 → CRITICAL (stuffing)
    
    NIE BLOKUJEMY za "za mało użyć" jeśli fraza jest w tekście!
    """
    text_lower = text.lower()
    missing_completely = []  # 0 wystąpień - CRITICAL
    stuffing_warnings = []   # za dużo - CRITICAL
    suggestions = []         # mogłoby być więcej - tylko info
    
    for kw in required:
        keyword = kw.get("keyword", "")
        count_req = kw.get("count", 1)
        count_max = kw.get("max", count_req * 3)  # max = 3× wymagane
        if not keyword:
            continue
        
        # Licz z lemmatyzacją
        if _LEMMATIZER_OK:
            result = count_phrase_occurrences(text, keyword)
            count_found = result.get("count", 0)
        else:
            count_found = text_lower.count(keyword.lower())
        
        # LOGIKA v29.0:
        if count_found == 0:
            # CRITICAL: fraza w ogóle nie występuje
            missing_completely.append(f'"{keyword}" (0/{count_req}) - BRAK W TEKŚCIE')
        elif count_found > count_max:
            # CRITICAL: stuffing
            stuffing_warnings.append(f'"{keyword}" ({count_found}×) - STUFFING! Max {count_max}×')
        elif count_found < count_req:
            # OK ale mogłoby być więcej - NIE BLOKUJ, tylko info
            suggestions.append(f'"{keyword}" ({count_found}/{count_req}) - OK, ale mogłoby być więcej')
    
    # CRITICAL = missing_completely + stuffing
    # WARNINGS = suggestions (nie blokują!)
    critical = missing_completely + stuffing_warnings
    
    return critical, suggestions


def quick_check_text_quality(text: str) -> Tuple[List[str], List[str]]:
    """
    v29.0: NOWY CHECK - Jakość tekstu (PRIORYTET 1!)
    
    Sprawdza:
    - Tautologie (słowo powtórzone w jednym zdaniu)
    - Pleonazmy ("przedszkole...w przedszkolu")
    - Zbyt długie zdania (>35 słów)
    - Strona bierna nadużywana
    """
    import re
    
    critical = []
    warnings = []
    
    # Podziel na zdania
    sentences = re.split(r'[.!?]+', text)
    
    for i, sentence in enumerate(sentences, 1):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.lower().split()
        
        # 1. TAUTOLOGIE - to samo słowo 2+ razy w zdaniu (min 4 litery)
        word_counts = {}
        for w in words:
            w_clean = re.sub(r'[^\w]', '', w)
            if len(w_clean) >= 4:
                word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
        
        for word, count in word_counts.items():
            if count >= 2 and word not in ['jest', 'oraz', 'które', 'który', 'która', 'także', 'bardzo']:
                # Sprawdź czy to nie odmiana
                if word in ['przedszkole', 'przedszkolu', 'przedszkolnym', 'przedszkolnych']:
                    critical.append(f'Zdanie {i}: tautologia "przedszkol*" powtórzone {count}× - POPRAW!')
                elif word in ['sensoryczny', 'sensoryczna', 'sensoryczne', 'sensorycznych', 'sensorycznym']:
                    if count >= 3:
                        warnings.append(f'Zdanie {i}: "sensoryczn*" użyte {count}× - rozważ synonim')
        
        # 2. ZBYT DŁUGIE ZDANIE
        if len(words) > 35:
            warnings.append(f'Zdanie {i}: {len(words)} słów - rozważ podział')
    
    # 3. PLEONAZMY GLOBALNE
    text_lower = text.lower()
    
    # "przedszkole...w przedszkolu" w tym samym akapicie
    paragraphs = text.split('\n\n')
    for p_idx, para in enumerate(paragraphs, 1):
        para_lower = para.lower()
        if 'przedszkole' in para_lower and 'w przedszkolu' in para_lower:
            # Sprawdź czy to nie jest "pomoce sensoryczne w przedszkolu" (fraza kluczowa)
            if 'pomoce sensoryczne w przedszkolu' not in para_lower:
                critical.append(f'Akapit {p_idx}: pleonazm "przedszkole...w przedszkolu" - zamień jedno na "placówka/obiekt"')
    
    return critical, warnings


def quick_check_length(text: str, min_w: int, max_w: int) -> Tuple[Optional[str], int]:
    words = len(text.split())
    if words < min_w * 0.8:
        return f"Za krótki: {words} słów (min: {min_w})", words
    elif words > max_w * 1.3:
        return f"Za długi: {words} słów (max: {max_w})", words
    return None, words


def quick_check_forbidden(text: str, forbidden: List[str]) -> List[str]:
    text_lower = text.lower()
    return [f for f in forbidden if f and f.lower() in text_lower]


def quick_check_ai_patterns(text: str) -> List[str]:
    patterns = [
        "w dzisiejszych czasach", "warto wiedzieć", "nie jest tajemnicą",
        "podsumowując", "w niniejszym artykule", "jak wiadomo",
        "przykład:", "na przykład,", "wyobraźmy sobie", "załóżmy, że"
    ]
    text_lower = text.lower()
    return [p for p in patterns if p in text_lower]


def run_quick_checks(text: str, context: Dict) -> Dict:
    """
    v29.0: NOWE PRIORYTETY
    
    PRIORYTET 1: Jakość tekstu (tautologie, pleonazmy) → CRITICAL
    PRIORYTET 2: Encje/n-gramy w odpowiednich miejscach → WARNING
    PRIORYTET 3: Słowa kluczowe (min 1×, nie stuffing) → tylko stuffing/brak blokuje
    """
    critical_errors = []  # Blokują zapis
    warnings = []         # Tylko info, nie blokują
    suggestions = []      # Sugestie optymalizacji
    
    # ============================================
    # PRIORYTET 1: JAKOŚĆ TEKSTU (CRITICAL!)
    # ============================================
    quality_critical, quality_warnings = quick_check_text_quality(text)
    for err in quality_critical:
        critical_errors.append({"type": "quality", "severity": "critical", "msg": err})
    for warn in quality_warnings:
        warnings.append({"type": "quality", "severity": "warning", "msg": warn})
    
    # AI patterns - też jakość
    ai = quick_check_ai_patterns(text)
    if ai:
        warnings.append({"type": "ai_pattern", "severity": "warning", "msg": f"AI patterns: {', '.join(ai)}"})
    
    # ============================================
    # PRIORYTET 2: DŁUGOŚĆ (ale elastyczna)
    # ============================================
    len_err, words = quick_check_length(
        text, 
        context.get("target_words_min", 150),
        context.get("target_words_max", 500)
    )
    if len_err:
        # Za krótki = critical, za długi = warning (można skrócić)
        if "za krótki" in len_err.lower():
            critical_errors.append({"type": "length", "severity": "critical", "msg": len_err})
        else:
            warnings.append({"type": "length", "severity": "warning", "msg": len_err})
    
    # ============================================
    # PRIORYTET 3: KEYWORDS (nowa logika!)
    # ============================================
    # critical = brak frazy LUB stuffing
    # suggestions = mogłoby być więcej (NIE BLOKUJE!)
    kw_critical, kw_suggestions = quick_check_keywords(text, context.get("keywords_required", []))
    
    for err in kw_critical:
        critical_errors.append({"type": "seo", "severity": "critical", "msg": err})
    for sug in kw_suggestions:
        suggestions.append({"type": "seo", "severity": "info", "msg": sug})
    
    # Forbidden keywords - zawsze critical
    forbidden = quick_check_forbidden(text, context.get("keywords_forbidden", []))
    if forbidden:
        critical_errors.append({"type": "seo", "severity": "critical", "msg": f"Zabronione frazy: {', '.join(forbidden)}"})
    
    # ============================================
    # STATS
    # ============================================
    paras = len([p for p in text.split('\n\n') if p.strip() and len(p) > 30])
    
    return {
        "passed": len(critical_errors) == 0,  # Tylko CRITICAL blokuje!
        "errors": critical_errors,
        "warnings": warnings,
        "suggestions": suggestions,  # Nowe - nie blokują
        "word_count": words,
        "paragraph_count": paras,
        "priority_summary": {
            "quality_issues": len([e for e in critical_errors if e["type"] == "quality"]),
            "seo_issues": len([e for e in critical_errors if e["type"] == "seo"]),
            "length_issues": len([e for e in critical_errors if e["type"] == "length"])
        }
    }


# ================================================================
# CLAUDE REVIEW
# ================================================================

def build_review_prompt(text: str, ctx: Dict) -> str:
    """v29.1: Prompt z nowymi priorytetami + przywrócone ważne elementy"""
    
    required = "\n".join([f'  • "{k["keyword"]}" (min 1×, zalecane {k.get("count",1)}×)' 
                          for k in ctx.get("keywords_required", []) if k.get("keyword")])
    forbidden = ", ".join(ctx.get("keywords_forbidden", [])) or "brak"
    h2_list = "\n".join([f"  • {h}" for h in ctx.get("h2_current", [])]) or "  (brak)"
    
    # Główna fraza
    main_kw = ctx.get("main_keyword", "")
    main_kw_count = ctx.get("main_keyword_count", 2)
    
    # Snippet info
    snippet_info = "TAK (40-60 słów na początku)" if ctx.get("snippet_required") else "NIE"
    
    return f"""Jesteś redaktorem i stylistą języka polskiego. Sprawdź i POPRAW tekst.

## PRIORYTETY (w tej kolejności!):

### 🔴 PRIORYTET 1: JAKOŚĆ TEKSTU (NAJWAŻNIEJSZE!)
Tekst musi być poprawny, naturalny i przyjemny w czytaniu.

SPRAWDŹ I POPRAW:
- **TAUTOLOGIE**: "przedszkole... w przedszkolu" → zamień jedno na "placówka/obiekt/sala"
- **PLEONAZMY**: "nowoczesne przedszkole wyposażone jest w pomoce w przedszkolu" → DRAMAT!
- **POWTÓRZENIA**: To samo słowo 2× w zdaniu (poza spójnikami) → użyj synonimu
- **STRONA BIERNA**: "jest wyposażone w" → "posiada", "oferuje", "zawiera"
- **DŁUGIE ZDANIA**: >30 słów → podziel na 2
- **GRAMATYKA**: Błędy, kolokacje, naturalność języka polskiego
- **AI PATTERNS**: "W dzisiejszych czasach", "Warto wiedzieć", "Nie jest tajemnicą" → USUŃ
- **HALUCYNACJE**: Wymyślone statystyki, badania, fakty bez źródła → USUŃ

### 🟡 PRIORYTET 2: ENCJE I N-GRAMY
Upewnij się, że kluczowe pojęcia są zdefiniowane/wyjaśnione przy pierwszym użyciu.

### 🟢 PRIORYTET 3: SŁOWA KLUCZOWE (elastyczne!)
Frazy powinny występować NATURALNIE. Lepiej 1× naturalnie niż 3× sztucznie!

GŁÓWNA FRAZA: "{main_kw}" (min {main_kw_count}×)

POZOSTAŁE FRAZY (min 1×, zalecane ilości to cel, nie wymóg):
{required}

❌ NIE RÓB: wstawiania fraz "na siłę" które psują naturalność
✅ TAK RÓB: wplataj frazy gdzie pasują do kontekstu

ZABRONIONE: {forbidden}

---

## KONTEKST
- Temat: {ctx.get("topic", "")}
- Batch: #{ctx.get("batch_number", 1)}
- H2: {h2_list}
- Słowa: {ctx.get("target_words_min", 200)}-{ctx.get("target_words_max", 500)}
- Akapity: {ctx.get("target_paragraphs_min", 2)}-{ctx.get("target_paragraphs_max", 5)}
- Snippet: {snippet_info}

## TEKST DO SPRAWDZENIA:
{text}

---

## ODPOWIEDŹ (tylko JSON):
```json
{{
  "status": "APPROVED | CORRECTED | REJECTED",
  "quality_score": 1-10,
  "issues": [
    {{"priority": 1, "type": "tautologia|pleonazm|gramatyka|halucynacja|ai_pattern", "description": "...", "fix_applied": true}},
    {{"priority": 2, "type": "encja_brak", "description": "...", "fix_applied": false}},
    {{"priority": 3, "type": "fraza_brak", "description": "...", "fix_applied": true}}
  ],
  "corrected_text": "pełny poprawiony tekst (tylko jeśli CORRECTED)",
  "summary": "co poprawiono"
}}
```

ZASADY:
- APPROVED = jakość OK, frazy OK (nawet jeśli nie idealnie po ilości)
- CORRECTED = poprawiłeś błędy jakościowe, zwróć pełny tekst
- REJECTED = tekst nie do uratowania (za krótki, same błędy, halucynacje)
- Zachowaj format h2: / h3:
- NIE dopisuj tekstu jeśli za krótki → zwróć REJECTED
- PRIORYTET 1 (jakość) ważniejszy niż dokładne ilości fraz!"""


def review_with_claude(text: str, ctx: Dict) -> ReviewResult:
    if not ANTHROPIC_AVAILABLE or not os.environ.get("ANTHROPIC_API_KEY"):
        return ReviewResult("APPROVED", text, None, [], "Claude niedostępny", len(text.split()))
    
    try:
        client = anthropic.Anthropic()
        start = time.time()
        
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


# ================================================================
# GŁÓWNA FUNKCJA
# ================================================================

def review_batch(text: str, context: Dict, skip_claude: bool = False) -> ReviewResult:
    """
    Pełny review: Quick Checks + Claude.
    """
    # Quick checks
    qc = run_quick_checks(text, context)
    
    if not qc["passed"]:
        issues = [ReviewIssue(e["type"], "critical", e["msg"]) for e in qc["errors"]]
        issues += [ReviewIssue(w["type"], "warning", w["msg"]) for w in qc["warnings"]]
        return ReviewResult(
            "QUICK_CHECK_FAILED", text, None, issues,
            "Popraw błędy krytyczne",
            qc["word_count"], qc["paragraph_count"]
        )
    
    if skip_claude:
        issues = [ReviewIssue(w["type"], "warning", w["msg"]) for w in qc["warnings"]]
        return ReviewResult(
            "APPROVED", text, None, issues,
            "Quick check OK",
            qc["word_count"], qc["paragraph_count"]
        )
    
    # Claude review
    result = review_with_claude(text, context)
    
    # Dodaj warnings z quick check
    for w in qc["warnings"]:
        if not any(i.fix_applied and i.type == w["type"] for i in result.issues):
            result.issues.append(ReviewIssue(w["type"], "warning", w["msg"]))
    
    return result


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
