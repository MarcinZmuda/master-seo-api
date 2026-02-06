---
title: Polish Perplexity Scorer
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Polish Perplexity AI Detector

Backend for BRAJEN SEO API perplexity analysis.

**API endpoint:** `POST /api/analyze`

```json
{"text": "Tekst do analizy..."}
```

**Response:**
```json
{
  "score": 72.3,
  "verdict": "likely_human",
  "metrics": {
    "mean_ppl": 58.4,
    "cv_ppl": 0.92,
    "low_ppl_ratio": 0.31,
    "spike_ratio": 0.08
  },
  "tokens_analyzed": 187,
  "timing_ms": 230
}
```
