"""
===============================================================================
🔬 PERPLEXITY AI DETECTOR v1.1 — HuggingFace Inference API (Option C)
===============================================================================
Wykrywa tekst AI mierząc przewidywalność tokenów.

ZERO LOKALNYCH ZALEŻNOŚCI — nie potrzebuje torch ani transformers.
Jedyne zależności: requests (jest w requirements.txt).

JAK DZIAŁA:
  1. Tokenizuje tekst (proste whitespace + regex — nie potrzebuje HF tokenizer)
  2. Losowo sampeluje 25-40 słów z tekstu (pomija stop-words)
  3. Dla każdego słowa: maskuje je → wysyła do HF fill-mask API
  4. Sprawdza: na którym miejscu w top-K jest oryginalne słowo?
  5. Oblicza metryki z rozkładu ranków

METRYKI:
  • mean_rank     — średni ranking oryginalnego tokena (niższy = bardziej AI)
  • rank_variance — wariancja ranków (niższa = bardziej AI)  
  • top1_ratio    — % tokenów zgadniętych na 1. miejscu (wyższy = AI)
  • miss_ratio    — % tokenów spoza top-K (wyższy = ludzki tekst)
  • rank_cv       — CV rozkładu ranków (niższy = AI)

KONFIGURACJA:
  1. Ustaw zmienną env HF_TOKEN (darmowy token z huggingface.co/settings/tokens)
  2. Opcjonalnie: zmień HF_PERPLEXITY_URL na self-hosted Space
  
  HF_TOKEN=hf_xxxxxxxxxxxxx  (free tier: ~30k req/month)

SELF-HOSTED (nieograniczone zapytania):
  Wgraj plik `hf_perplexity_space/app.py` jako HuggingFace Space.
  Ustaw HF_PERPLEXITY_URL=https://your-name-perplexity.hf.space/analyze
  → zero rate limitów, ~200ms per tekst

INTEGRACJA:
  from perplexity_ai_detector import (
      analyze_perplexity,         # Pełna analiza
      get_perplexity_score,       # Quick score 0-100
      get_perplexity_for_moe,     # Format MoE
      is_available,               # Czy HF token jest ustawiony
  )

===============================================================================
"""

import os
import re
import json
import time
import random
import hashlib
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[PERPLEXITY] ❌ requests not installed")


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class PerplexityConfig:
    """Konfiguracja detektora perplexity (HF API)."""

    # --- HuggingFace API ---
    HF_MODEL: str = "sdadas/polish-roberta-base"
    HF_API_URL: str = "https://api-inference.huggingface.co/models/{model}"

    # Self-hosted Space URL (jeśli ustawiony — użyj zamiast HF API)
    SELF_HOSTED_URL: str = ""

    # --- Sampling ---
    SAMPLE_SIZE: int = 35
    MIN_WORD_LENGTH: int = 3
    TOP_K: int = 15
    MAX_CONCURRENT: int = 8

    # --- Progi ---
    MEAN_RANK_AI: float = 2.5
    MEAN_RANK_HUMAN: float = 6.0

    TOP1_RATIO_AI: float = 0.50
    TOP1_RATIO_HUMAN: float = 0.25

    MISS_RATIO_AI: float = 0.10
    MISS_RATIO_HUMAN: float = 0.35

    RANK_CV_AI: float = 0.60
    RANK_CV_HUMAN: float = 1.20

    # --- Wagi ---
    WEIGHT_MEAN_RANK: float = 0.25
    WEIGHT_TOP1: float = 0.20
    WEIGHT_MISS: float = 0.25
    WEIGHT_RANK_CV: float = 0.30

    # --- Severity ---
    BLOCKING_ENABLED: bool = False
    CRITICAL_SCORE: float = 25.0
    WARNING_SCORE: float = 50.0

    # --- Cache ---
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 200

    # --- Timeout ---
    REQUEST_TIMEOUT: int = 10


CONFIG = PerplexityConfig()


# ============================================================================
# HF TOKEN
# ============================================================================

HF_TOKEN = os.getenv("HF_TOKEN", "")
SELF_HOSTED_URL = os.getenv("HF_PERPLEXITY_URL", CONFIG.SELF_HOSTED_URL)

if HF_TOKEN:
    print("[PERPLEXITY] ✅ HF_TOKEN configured (Option C: API mode)")
elif SELF_HOSTED_URL:
    print(f"[PERPLEXITY] ✅ Self-hosted: {SELF_HOSTED_URL}")
else:
    print("[PERPLEXITY] ⚠️ No HF_TOKEN or HF_PERPLEXITY_URL — disabled")
    print("[PERPLEXITY] Set HF_TOKEN=hf_xxx (free: huggingface.co/settings/tokens)")


def is_available() -> bool:
    return REQUESTS_AVAILABLE and bool(HF_TOKEN or SELF_HOSTED_URL)


# ============================================================================
# STOP-WORDS
# ============================================================================

POLISH_STOPWORDS = {
    "i", "w", "na", "z", "do", "nie", "się", "jest", "to", "że", "o", "jak",
    "ale", "za", "co", "od", "po", "tak", "czy", "przez", "ich", "jego",
    "jej", "tego", "tym", "tej", "ten", "ta", "te", "lub", "oraz", "a",
    "też", "już", "tylko", "może", "był", "była", "było", "były",
    "być", "są", "będzie", "jako", "dla", "ze", "pod", "nad", "przed",
    "między", "przy", "bez", "ku", "aby", "by", "więc", "bo", "gdy",
    "gdyż", "lecz", "ani", "sobie", "tutaj", "tam", "tu", "mi", "go",
    "mu", "nas", "was", "je", "ją", "nim", "niej", "nich",
    "bardzo", "mnie", "jakie", "które", "który", "która",
    "których", "którym", "jeszcze", "kiedy", "gdzie", "swoje",
}


# ============================================================================
# CACHE
# ============================================================================

_cache: Dict[str, Dict] = {}


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _get_cached(text: str) -> Optional[Dict]:
    if not CONFIG.CACHE_ENABLED:
        return None
    return _cache.get(_cache_key(text))


def _set_cache(text: str, result: Dict):
    if not CONFIG.CACHE_ENABLED:
        return
    if len(_cache) >= CONFIG.CACHE_MAX_SIZE:
        del _cache[next(iter(_cache))]
    _cache[_cache_key(text)] = result


# ============================================================================
# TOKENIZACJA I SAMPLING
# ============================================================================

def _tokenize_simple(text: str) -> List[Tuple[str, int]]:
    """Prosta tokenizacja — słowa + pozycja."""
    tokens = []
    for m in re.finditer(r'[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ]+', text):
        word = m.group()
        if len(word) >= CONFIG.MIN_WORD_LENGTH:
            tokens.append((word, m.start()))
    return tokens


def _sample_tokens(tokens: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    """Stratified sampling z pominięciem stop-words."""
    content_tokens = [
        (w, p) for w, p in tokens
        if w.lower() not in POLISH_STOPWORDS
    ]

    if len(content_tokens) <= n:
        return content_tokens

    segment_size = len(content_tokens) // 5
    sampled = []
    per_segment = n // 5 + 1

    for i in range(5):
        start = i * segment_size
        end = start + segment_size if i < 4 else len(content_tokens)
        segment = content_tokens[start:end]
        if segment:
            sampled.extend(random.sample(segment, min(per_segment, len(segment))))

    if len(sampled) > n:
        sampled = random.sample(sampled, n)

    return sampled


# ============================================================================
# HF API: FILL-MASK
# ============================================================================

def _build_masked_text(text: str, word: str, position: int) -> str:
    """Podmienia słowo na <mask> w kontekście ±200 znaków."""
    ctx_start = max(0, position - 200)
    ctx_end = min(len(text), position + len(word) + 200)
    context = text[ctx_start:ctx_end]
    word_offset = position - ctx_start
    return (context[:word_offset] + "<mask>" + context[word_offset + len(word):]).strip()


def _call_hf_fill_mask(masked_text: str) -> Optional[List[Dict]]:
    """Wywołuje HF fill-mask API."""
    url = CONFIG.HF_API_URL.format(model=CONFIG.HF_MODEL)
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.post(
            url, headers=headers,
            json={"inputs": masked_text, "parameters": {"top_k": CONFIG.TOP_K}},
            timeout=CONFIG.REQUEST_TIMEOUT
        )

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            # Model loading — retry
            time.sleep(3)
            resp2 = requests.post(
                url, headers=headers,
                json={"inputs": masked_text, "parameters": {"top_k": CONFIG.TOP_K}},
                timeout=CONFIG.REQUEST_TIMEOUT + 20
            )
            return resp2.json() if resp2.status_code == 200 else None
        elif resp.status_code == 429:
            print("[PERPLEXITY] ⚠️ Rate limited")
            return None
        else:
            return None

    except Exception:
        return None


def _call_self_hosted(text: str) -> Optional[Dict]:
    """
    Wywołuje self-hosted HF Space.
    
    Obsługuje dwa formaty:
    1. Gradio API: POST /api/analyze → {"data": ["text"]} → {"data": [{...}]}
    2. Custom API: POST /analyze → {"text": "..."} → {...}
    """
    url = SELF_HOSTED_URL.rstrip("/")
    
    try:
        # Próba 1: Gradio API format
        resp = requests.post(
            url,
            json={"data": [text]},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            # Gradio wraps in {"data": [result]}
            if isinstance(data, dict) and "data" in data:
                result = data["data"][0] if data["data"] else None
                if isinstance(result, str):
                    import json
                    result = json.loads(result)
                return result
            # Direct format (custom endpoint)
            return data
        return None
    except Exception as e:
        print(f"[PERPLEXITY] ⚠️ Self-hosted error: {e}")
        return None


# ============================================================================
# ANALIZA JEDNEGO TOKENA
# ============================================================================

def _analyze_token(text: str, word: str, position: int) -> Optional[Dict]:
    """Maskuje słowo, sprawdza ranking w predykcjach."""
    masked = _build_masked_text(text, word, position)
    predictions = _call_hf_fill_mask(masked)

    if not predictions or not isinstance(predictions, list):
        return None

    word_lower = word.lower().strip()

    for rank_idx, pred in enumerate(predictions):
        pred_token = pred.get("token_str", "").strip().lower()
        # Exact match
        if pred_token == word_lower:
            return {"word": word, "rank": rank_idx + 1,
                    "score": pred.get("score", 0), "in_top_k": True}
        # Stem match (fleksja polska)
        if len(word_lower) > 4 and len(pred_token) > 4:
            if pred_token[:4] == word_lower[:4]:
                return {"word": word, "rank": rank_idx + 1,
                        "score": pred.get("score", 0), "in_top_k": True}

    return {"word": word, "rank": CONFIG.TOP_K + 1,
            "score": 0.0, "in_top_k": False}


# ============================================================================
# METRYKI
# ============================================================================

def _compute_metrics(token_results: List[Dict]) -> Dict[str, float]:
    if not token_results or len(token_results) < 5:
        return {}

    ranks = [t["rank"] for t in token_results]
    scores = [t["score"] for t in token_results]
    n = len(ranks)

    mean_rank = statistics.mean(ranks)
    std_rank = statistics.stdev(ranks) if n > 1 else 0
    cv_rank = std_rank / mean_rank if mean_rank > 0 else 0

    return {
        "mean_rank": round(mean_rank, 2),
        "median_rank": round(statistics.median(ranks), 2),
        "std_rank": round(std_rank, 2),
        "rank_cv": round(cv_rank, 3),
        "top1_ratio": round(sum(1 for r in ranks if r == 1) / n, 3),
        "top3_ratio": round(sum(1 for r in ranks if r <= 3) / n, 3),
        "miss_ratio": round(sum(1 for t in token_results if not t["in_top_k"]) / n, 3),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0,
        "tokens_analyzed": n,
    }


def _score_metric(value: float, ai_thr: float, human_thr: float,
                  higher_is_human: bool = True) -> float:
    if higher_is_human:
        if value >= human_thr: return 100.0
        if value <= ai_thr: return 0.0
        return ((value - ai_thr) / (human_thr - ai_thr)) * 100.0
    else:
        if value <= human_thr: return 100.0
        if value >= ai_thr: return 0.0
        return ((ai_thr - value) / (ai_thr - human_thr)) * 100.0


# ============================================================================
# GŁÓWNA ANALIZA
# ============================================================================

def analyze_perplexity(text: str) -> Dict[str, Any]:
    """
    Pełna analiza przewidywalności tokenów.
    HF API mode: ~35 równoległych requestów, ~3-8s.
    Self-hosted mode: 1 request, ~200ms.
    """
    t_start = time.time()

    if not is_available():
        return _err("unavailable", t_start)

    if len(text.split()) < 30:
        return _err("text_too_short", t_start)

    cached = _get_cached(text)
    if cached:
        cached["from_cache"] = True
        return cached

    # Self-hosted (fast path)
    if SELF_HOSTED_URL:
        result = _call_self_hosted(text)
        if result:
            timing = round((time.time() - t_start) * 1000, 1)
            output = {
                "available": True,
                "score": result.get("score", -1),
                "verdict": result.get("verdict", "unknown"),
                "metrics": result.get("metrics", {}),
                "component_scores": result.get("component_scores", {}),
                "timing_ms": timing,
                "tokens_analyzed": result.get("tokens_analyzed", 0),
                "api_calls": 1,
                "mode": "self_hosted",
            }
            _set_cache(text, output)
            return output
        return _err("self_hosted_error", t_start)

    # HF API (standard path)
    tokens = _tokenize_simple(text)
    sampled = _sample_tokens(tokens, CONFIG.SAMPLE_SIZE)

    if len(sampled) < 10:
        return _err("too_few_content_words", t_start)

    # Równoległe zapytania
    token_results = []
    api_calls = 0

    with ThreadPoolExecutor(max_workers=CONFIG.MAX_CONCURRENT) as executor:
        futures = {
            executor.submit(_analyze_token, text, word, pos): word
            for word, pos in sampled
        }
        for future in as_completed(futures):
            api_calls += 1
            try:
                r = future.result()
                if r:
                    token_results.append(r)
            except Exception:
                pass

    if len(token_results) < 8:
        return _err("insufficient_results", t_start)

    metrics = _compute_metrics(token_results)

    # Component scores
    cs = {
        "mean_rank": round(_score_metric(metrics["mean_rank"],
                                          CONFIG.MEAN_RANK_AI, CONFIG.MEAN_RANK_HUMAN, True), 1),
        "top1_ratio": round(_score_metric(metrics["top1_ratio"],
                                           CONFIG.TOP1_RATIO_AI, CONFIG.TOP1_RATIO_HUMAN, False), 1),
        "miss_ratio": round(_score_metric(metrics["miss_ratio"],
                                           CONFIG.MISS_RATIO_AI, CONFIG.MISS_RATIO_HUMAN, True), 1),
        "rank_cv": round(_score_metric(metrics["rank_cv"],
                                        CONFIG.RANK_CV_AI, CONFIG.RANK_CV_HUMAN, True), 1),
    }

    final = max(0, min(100,
        cs["mean_rank"] * CONFIG.WEIGHT_MEAN_RANK +
        cs["top1_ratio"] * CONFIG.WEIGHT_TOP1 +
        cs["miss_ratio"] * CONFIG.WEIGHT_MISS +
        cs["rank_cv"] * CONFIG.WEIGHT_RANK_CV
    ))

    verdict = ("likely_ai" if final < CONFIG.CRITICAL_SCORE else
               "uncertain" if final < CONFIG.WARNING_SCORE else
               "likely_human")

    output = {
        "available": True,
        "score": round(final, 1),
        "verdict": verdict,
        "metrics": metrics,
        "component_scores": cs,
        "timing_ms": round((time.time() - t_start) * 1000, 1),
        "tokens_analyzed": len(token_results),
        "api_calls": api_calls,
        "mode": "hf_api",
        "model": CONFIG.HF_MODEL,
    }

    _set_cache(text, output)
    return output


def _err(reason: str, t_start: float) -> Dict:
    return {
        "available": reason != "unavailable",
        "score": -1,
        "verdict": reason,
        "metrics": {},
        "component_scores": {},
        "timing_ms": round((time.time() - t_start) * 1000, 1),
        "tokens_analyzed": 0, "api_calls": 0,
        "mode": "hf_api" if HF_TOKEN else "none",
    }


# ============================================================================
# FORMAT MoE
# ============================================================================

def get_perplexity_for_moe(batch_text: str) -> Dict:
    """Format kompatybilny z MoE batch validator."""
    result = analyze_perplexity(batch_text)

    if not result["available"] or result["score"] < 0:
        return {
            "expert": "PERPLEXITY_EXPERT", "version": "1.1",
            "severity": "info", "score": -1,
            "message": f"Perplexity unavailable: {result['verdict']}",
            "issues": [], "action": "CONTINUE"
        }

    score = result["score"]
    metrics = result["metrics"]
    cs = result["component_scores"]

    if score < CONFIG.CRITICAL_SCORE and CONFIG.BLOCKING_ENABLED:
        severity, action = "critical", "FIX_AND_RETRY"
    elif score < CONFIG.WARNING_SCORE:
        severity, action = "warning", "CONTINUE"
    else:
        severity, action = "info", "CONTINUE"

    issues = []
    if cs.get("rank_cv", 100) < 30:
        issues.append({"metric": "rank_cv",
            "message": f"Niska wariancja przewidywalności (CV={metrics.get('rank_cv',0):.2f})",
            "fix_hint": "Dodaj nietypowe słownictwo, nazwy własne, kolokwializmy"})
    if cs.get("top1_ratio", 100) < 30:
        issues.append({"metric": "top1_ratio",
            "message": f"{metrics.get('top1_ratio',0)*100:.0f}% słów idealnie przewidywalnych",
            "fix_hint": "Zamień oczywiste sformułowania na mniej typowe synonimy"})
    if cs.get("miss_ratio", 100) < 30:
        issues.append({"metric": "miss_ratio",
            "message": f"Tylko {metrics.get('miss_ratio',0)*100:.0f}% słów zaskakujących",
            "fix_hint": "Wpleć terminy specjalistyczne, cytaty, liczby"})

    return {
        "expert": "PERPLEXITY_EXPERT", "version": "1.1",
        "severity": severity, "score": round(score, 1),
        "verdict": result["verdict"],
        "message": (f"PPL: {score:.0f}/100 ({result['verdict']}) | "
                    f"Rank: {metrics.get('mean_rank',0):.1f}, "
                    f"Top1: {metrics.get('top1_ratio',0)*100:.0f}%, "
                    f"Miss: {metrics.get('miss_ratio',0)*100:.0f}%, "
                    f"CV: {metrics.get('rank_cv',0):.2f} | "
                    f"{result.get('api_calls',0)} calls"),
        "issues": issues, "action": action,
        "timing_ms": result["timing_ms"],
    }


def get_perplexity_score(text: str) -> float:
    """Quick score: 0-100. -1 jeśli niedostępne."""
    return analyze_perplexity(text)["score"]


def get_detector_status() -> Dict:
    return {
        "version": "1.1",
        "mode": "self_hosted" if SELF_HOSTED_URL else ("hf_api" if HF_TOKEN else "disabled"),
        "available": is_available(),
        "model": CONFIG.HF_MODEL,
        "self_hosted_url": SELF_HOSTED_URL or None,
        "hf_token_set": bool(HF_TOKEN),
        "blocking_enabled": CONFIG.BLOCKING_ENABLED,
        "sample_size": CONFIG.SAMPLE_SIZE,
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PERPLEXITY AI DETECTOR v1.1 — Option C")
    print("=" * 60)

    status = get_detector_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    if not is_available():
        print("\n⚠️ Aby włączyć:")
        print("  export HF_TOKEN=hf_xxxxxxxxxxxxx")
        print("  (free token: huggingface.co/settings/tokens)")
