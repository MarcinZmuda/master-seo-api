"""
HuggingFace Space: Polish Perplexity Scorer
============================================
Self-hosted backend for BRAJEN perplexity_ai_detector.py

Deploy:
  1. Create new Space on huggingface.co/new-space
  2. Select: Gradio, CPU Basic (Free), Python 3.10
  3. Upload: app.py + requirements.txt
  4. Set env in BRAJEN: HF_PERPLEXITY_URL=https://YOUR-NAME-polish-perplexity.hf.space/api/analyze

Endpoint: POST /api/analyze  {"text": "..."}
Returns:  {"score": 72.3, "verdict": "likely_human", "metrics": {...}, ...}
"""

import re
import math
import time
import random
import statistics
from typing import Dict, List, Tuple, Optional
import gradio as gr
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ============================================================================
# MODEL (loaded once at startup)
# ============================================================================

MODEL_NAME = "sdadas/polish-roberta-base"
print(f"[SPACE] Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()
for p in model.parameters():
    p.requires_grad = False

print(f"[SPACE] ✅ Model loaded")

MASK_ID = tokenizer.mask_token_id
SPECIAL_IDS = {tokenizer.cls_token_id, tokenizer.sep_token_id,
               tokenizer.pad_token_id, MASK_ID}
SPECIAL_IDS.discard(None)

# ============================================================================
# CONFIG
# ============================================================================

SAMPLE_SIZE = 40
MIN_WORD_LEN = 3
STRIDE = 5              # Batch masking stride
MAX_TOKENS = 512

# Progi (kalibracja na polskim tekście)
MEAN_PPL_AI = 35.0
MEAN_PPL_HUMAN = 65.0
PPL_CV_AI = 0.50
PPL_CV_HUMAN = 0.85
LOW_PPL_RATIO_AI = 0.55
LOW_PPL_RATIO_HUMAN = 0.35
SPIKE_RATIO_AI = 0.03
SPIKE_RATIO_HUMAN = 0.10

W_MEAN = 0.20
W_CV = 0.35
W_LOW = 0.25
W_SPIKE = 0.20

POLISH_STOPWORDS = {
    "i", "w", "na", "z", "do", "nie", "się", "jest", "to", "że", "o", "jak",
    "ale", "za", "co", "od", "po", "tak", "czy", "przez", "ich", "jego",
    "jej", "tego", "tym", "tej", "ten", "ta", "te", "lub", "oraz", "a",
    "też", "już", "tylko", "może", "był", "była", "było", "były",
    "być", "są", "będzie", "jako", "dla", "ze", "pod", "nad", "przed",
    "między", "przy", "bez", "ku", "aby", "by", "więc", "bo", "gdy",
    "gdyż", "lecz", "ani", "sobie", "tam", "tu", "mi", "go",
    "mu", "nas", "was", "je", "ją", "nim", "niej", "nich",
    "bardzo", "mnie", "które", "który", "która", "których", "którym",
    "jeszcze", "kiedy", "gdzie", "swoje",
}

# ============================================================================
# PER-TOKEN PERPLEXITY (batch masked — fast)
# ============================================================================

def compute_token_perplexities(text: str) -> List[float]:
    """
    Pseudo-log-likelihood perplexity per token.
    Batch masking: co STRIDE-ty token jednocześnie → ~5x szybsze.
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_TOKENS, padding=False
    )
    input_ids = inputs["input_ids"][0]
    n = len(input_ids)
    if n < 5:
        return []

    valid_idx = [i for i in range(n) if input_ids[i].item() not in SPECIAL_IDS]
    if not valid_idx:
        return []

    perplexities = [0.0] * len(valid_idx)
    idx_map = {vi: pos for pos, vi in enumerate(valid_idx)}

    with torch.no_grad():
        for offset in range(STRIDE):
            batch_idx = [i for i in valid_idx if i % STRIDE == offset]
            if not batch_idx:
                continue

            masked = input_ids.clone().unsqueeze(0)
            originals = {}
            for i in batch_idx:
                originals[i] = masked[0, i].item()
                masked[0, i] = MASK_ID

            logits = model(masked).logits[0]

            for i in batch_idx:
                log_probs = torch.log_softmax(logits[i], dim=-1)
                lp = log_probs[originals[i]].item()
                perplexities[idx_map[i]] = math.exp(-lp)

    return perplexities


# ============================================================================
# SCORING
# ============================================================================

def score_metric(val, ai_thr, human_thr, higher_human=True):
    if higher_human:
        if val >= human_thr: return 100.0
        if val <= ai_thr: return 0.0
        return (val - ai_thr) / (human_thr - ai_thr) * 100
    else:
        if val <= human_thr: return 100.0
        if val >= ai_thr: return 0.0
        return (ai_thr - val) / (ai_thr - human_thr) * 100


def analyze(text: str) -> Dict:
    """Full perplexity analysis."""
    t0 = time.time()

    words = text.split()
    if len(words) < 30:
        return {"score": -1, "verdict": "text_too_short",
                "metrics": {}, "component_scores": {},
                "tokens_analyzed": 0, "timing_ms": 0}

    ppls = compute_token_perplexities(text)
    if len(ppls) < 10:
        return {"score": -1, "verdict": "calc_failed",
                "metrics": {}, "component_scores": {},
                "tokens_analyzed": 0,
                "timing_ms": round((time.time()-t0)*1000, 1)}

    # Metrics
    mean_ppl = statistics.mean(ppls)
    std_ppl = statistics.stdev(ppls)
    cv_ppl = std_ppl / mean_ppl if mean_ppl > 0 else 0
    low_ratio = sum(1 for p in ppls if p < 10) / len(ppls)
    spike_ratio = sum(1 for p in ppls if p > 200) / len(ppls)

    metrics = {
        "mean_ppl": round(mean_ppl, 2),
        "median_ppl": round(statistics.median(ppls), 2),
        "std_ppl": round(std_ppl, 2),
        "cv_ppl": round(cv_ppl, 3),
        "low_ppl_ratio": round(low_ratio, 3),
        "spike_ratio": round(spike_ratio, 3),
        "tokens_analyzed": len(ppls),
    }

    # Component scores
    s_mean = score_metric(mean_ppl, MEAN_PPL_AI, MEAN_PPL_HUMAN, True)
    s_cv = score_metric(cv_ppl, PPL_CV_AI, PPL_CV_HUMAN, True)
    s_low = score_metric(low_ratio, LOW_PPL_RATIO_AI, LOW_PPL_RATIO_HUMAN, False)
    s_spike = score_metric(spike_ratio, SPIKE_RATIO_AI, SPIKE_RATIO_HUMAN, True)

    cs = {
        "mean_ppl": round(s_mean, 1),
        "ppl_burstiness": round(s_cv, 1),
        "low_ppl_ratio": round(s_low, 1),
        "spike_ratio": round(s_spike, 1),
    }

    final = max(0, min(100,
        s_mean * W_MEAN + s_cv * W_CV + s_low * W_LOW + s_spike * W_SPIKE))

    verdict = ("likely_ai" if final < 25 else
               "uncertain" if final < 50 else "likely_human")

    return {
        "score": round(final, 1),
        "verdict": verdict,
        "metrics": metrics,
        "component_scores": cs,
        "tokens_analyzed": len(ppls),
        "timing_ms": round((time.time()-t0)*1000, 1),
        "model": MODEL_NAME,
    }


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def gradio_analyze(text: str) -> str:
    """Gradio wrapper — returns formatted string for UI."""
    if not text or len(text.split()) < 30:
        return "⚠️ Tekst za krótki (min. 30 słów)"

    r = analyze(text)
    if r["score"] < 0:
        return f"❌ Błąd: {r['verdict']}"

    emoji = "🤖" if r["verdict"] == "likely_ai" else "🤔" if r["verdict"] == "uncertain" else "👤"
    m = r["metrics"]

    return (
        f"{emoji} Score: {r['score']}/100 — {r['verdict']}\n\n"
        f"Mean PPL: {m['mean_ppl']}\n"
        f"PPL CV (burstiness): {m['cv_ppl']}\n"
        f"Low-PPL ratio: {m['low_ppl_ratio']*100:.1f}%\n"
        f"Spike ratio: {m['spike_ratio']*100:.1f}%\n"
        f"Tokens analyzed: {m['tokens_analyzed']}\n"
        f"Time: {r['timing_ms']}ms"
    )


# API endpoint (POST /api/analyze)
def api_analyze(text: str) -> Dict:
    """API endpoint for perplexity_ai_detector.py integration."""
    return analyze(text)


with gr.Blocks(title="Polish Perplexity Scorer") as demo:
    gr.Markdown("# 🔬 Polish Perplexity AI Detector")
    gr.Markdown("Wklej tekst poniżej. Score 0-100 (0 = prawdopodobnie AI, 100 = prawdopodobnie ludzki).")

    with gr.Row():
        inp = gr.Textbox(label="Tekst do analizy", lines=10,
                         placeholder="Wklej tekst po polsku (min. 30 słów)...")
    btn = gr.Button("Analizuj", variant="primary")
    out = gr.Textbox(label="Wynik", lines=8)

    btn.click(fn=gradio_analyze, inputs=inp, outputs=out)

    # API endpoint
    api_fn = gr.Interface(
        fn=api_analyze,
        inputs=gr.Textbox(),
        outputs=gr.JSON(),
        api_name="analyze"
    )

demo.launch()
