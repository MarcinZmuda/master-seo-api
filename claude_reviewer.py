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

def quick_check_keywords(text: str, required: List[Dict]) -> Tuple[List[str], List[str]]:
    text_lower = text.lower()
    missing, warnings = [], []
    
    for kw in required:
        keyword = kw.get("keyword", "")
        count_req = kw.get("count", 1)
        if not keyword:
            continue
        count_found = text_lower.count(keyword.lower())
        if count_found < count_req:
            missing.append(f'"{keyword}" ({count_found}/{count_req})')
        elif count_found > count_req * 2.5:
            warnings.append(f'"{keyword}" użyte {count_found}x - keyword stuffing?')
    return missing, warnings


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
    errors, warnings = [], []
    
    # Keywords
    missing, kw_warn = quick_check_keywords(text, context.get("keywords_required", []))
    if missing:
        errors.append({"type": "seo", "msg": f"Brakujące frazy: {', '.join(missing)}"})
    warnings.extend([{"type": "seo", "msg": w} for w in kw_warn])
    
    # Length
    len_err, words = quick_check_length(
        text, 
        context.get("target_words_min", 150),
        context.get("target_words_max", 500)
    )
    if len_err:
        errors.append({"type": "length", "msg": len_err})
    
    # Forbidden
    forbidden = quick_check_forbidden(text, context.get("keywords_forbidden", []))
    if forbidden:
        errors.append({"type": "seo", "msg": f"Zabronione frazy: {', '.join(forbidden)}"})
    
    # AI patterns
    ai = quick_check_ai_patterns(text)
    if ai:
        warnings.append({"type": "ai_pattern", "msg": f"AI patterns: {', '.join(ai)}"})
    
    paras = len([p for p in text.split('\n\n') if p.strip() and len(p) > 30])
    
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "word_count": words,
        "paragraph_count": paras
    }


# ================================================================
# CLAUDE REVIEW
# ================================================================

def build_review_prompt(text: str, ctx: Dict) -> str:
    required = "\n".join([f'  • "{k["keyword"]}" × {k.get("count",1)}' 
                          for k in ctx.get("keywords_required", []) if k.get("keyword")])
    forbidden = ", ".join(ctx.get("keywords_forbidden", [])) or "brak"
    h2_list = "\n".join([f"  • {h}" for h in ctx.get("h2_current", [])]) or "  (brak)"
    
    return f"""Jesteś redaktorem SEO. Sprawdź i POPRAW tekst.

## KONTEKST
- Temat: {ctx.get("topic", "")}
- Batch: #{ctx.get("batch_number", 1)}
- Główna fraza: "{ctx.get("main_keyword", "")}" × {ctx.get("main_keyword_count", 2)}

## H2: {h2_list}

## FRAZY WYMAGANE (dokładna forma!):
{required}

## ZABRONIONE: {forbidden}

## PARAMETRY
- Słowa: {ctx.get("target_words_min", 200)}-{ctx.get("target_words_max", 500)}
- Akapity: {ctx.get("target_paragraphs_min", 2)}-{ctx.get("target_paragraphs_max", 5)}
- Snippet: {"TAK" if ctx.get("snippet_required") else "NIE"}

## TEKST:
{text}

---

## SPRAWDŹ:
1. SEO: Czy wszystkie frazy są? Wpleć brakujące naturalnie.
2. DŁUGOŚĆ: Sekcje zbalansowane? Akapity 40-150 słów?
3. POWTÓRZENIA: Ten sam temat 2x? Powtórzone zdania?
4. GRAMATYKA: Błędy, kolokacje, naturalność?
5. AI PATTERNS: "W dzisiejszych czasach", "Warto wiedzieć" → USUŃ
6. HALUCYNACJE: Wymyślone statystyki/fakty → USUŃ
7. SPÓJNOŚĆ: Płynne przejścia między akapitami?

## ODPOWIEDŹ (tylko JSON):
```json
{{
  "status": "APPROVED | CORRECTED | REJECTED",
  "issues": [{{"type": "...", "severity": "critical|warning", "description": "...", "fix_applied": true}}],
  "corrected_text": "pełny poprawiony tekst (tylko jeśli CORRECTED)",
  "summary": "co poprawiono"
}}
```

ZASADY:
- APPROVED = wszystko OK
- CORRECTED = poprawiłeś błędy, zwróć pełny tekst w corrected_text
- REJECTED = wymaga przepisania (za krótki, same halucynacje)
- Zachowaj format h2: / h3:
- NIE dopisuj jeśli za krótki → REJECTED"""


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
