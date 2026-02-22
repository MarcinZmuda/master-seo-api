"""
===============================================================================
spaCy SALIENCE ENDPOINT v1.0
===============================================================================
Endpoint NER + salience analysis oparty o spaCy (pl_core_news_md).

Analizuje tekst pod kątem:
1. Named Entity Recognition (NER) — wykrywanie encji (PER, ORG, LOC, MISC)
2. Entity Salience — prominencja każdej encji:
   - pozycja pierwszego wystąpienia (im wcześniej, tym ważniejsze)
   - częstotliwość (ile razy encja występuje)
   - rozproszenie (spread) — czy encja jest przez cały tekst
   - obecność w intro (pierwsze 200 znaków)
3. Lemmatyzacja i POS tagging dla kontekstu

ENDPOINT:
    POST /api/spacy/salience

REQUEST:
    {
        "text": "Artykuł do analizy...",
        "main_keyword": "opcjonalne główne słowo kluczowe",
        "top_n": 20  // opcjonalnie, ile encji zwrócić (default: 20)
    }

RESPONSE:
    {
        "entities": [...],
        "main_keyword_salience": {...},
        "stats": {...},
        "model": "pl_core_news_md"
    }

===============================================================================
"""

import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional


# ════════════════════════════════════════════════════════════════════
# NLP MODEL LOADING
# ════════════════════════════════════════════════════════════════════

_nlp = None
_nlp_model_name = None


def _get_nlp():
    """Lazy-load spaCy model via nlp_config."""
    global _nlp, _nlp_model_name
    if _nlp is not None:
        return _nlp

    try:
        from nlp_config import get_nlp, get_nlp_info
        _nlp = get_nlp()
        info = get_nlp_info()
        _nlp_model_name = info.get("loaded_model", "unknown")
        print(f"[SPACY_SALIENCE] ✅ Model loaded: {_nlp_model_name}")
        return _nlp
    except Exception as e:
        print(f"[SPACY_SALIENCE] ❌ Failed to load NLP model: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ════════════════════════════════════════════════════════════════════

def _compute_entity_salience(
    text: str,
    entity_text: str,
    text_len: int,
) -> Dict[str, Any]:
    """
    Oblicza salience (prominencję) pojedynczej encji w tekście.

    Scoring:
    - Pozycja pierwszego wystąpienia: 0-40 pkt
    - Częstotliwość: 0-30 pkt
    - Rozproszenie (spread): 0-30 pkt
    """
    text_lower = text.lower()
    entity_lower = entity_text.lower()
    pattern = rf'\b{re.escape(entity_lower)}\b'
    matches = list(re.finditer(pattern, text_lower))

    if not matches:
        return {
            "score": 0,
            "frequency": 0,
            "first_position_pct": 100.0,
            "is_in_intro": False,
            "spread": 0.0,
            "status": "MISSING",
        }

    first_pos = matches[0].start()
    first_pos_pct = (first_pos / text_len * 100) if text_len > 0 else 100.0
    is_in_intro = first_pos < 200

    frequency = len(matches)
    words_count = len(text_lower.split())
    freq_per_100 = (frequency / words_count * 100) if words_count > 0 else 0

    # Spread — how distributed the entity is across the text
    positions = [m.start() / text_len for m in matches] if text_len > 0 else []
    spread = (max(positions) - min(positions)) if len(positions) > 1 else 0.0

    # --- Score ---
    score = 0

    # Position score (0-40)
    if first_pos_pct < 2:
        score += 40
    elif first_pos_pct < 5:
        score += 30
    elif first_pos_pct < 10:
        score += 20
    elif first_pos_pct < 25:
        score += 10

    # Frequency score (0-30)
    if 0.5 <= freq_per_100 <= 2.0:
        score += 30
    elif 0.3 <= freq_per_100 <= 3.0:
        score += 20
    elif freq_per_100 > 0:
        score += 10

    # Spread score (0-30)
    if spread > 0.7:
        score += 30
    elif spread > 0.4:
        score += 20
    elif spread > 0.1:
        score += 10

    score = min(100, score)

    if score >= 70:
        status = "HIGH"
    elif score >= 40:
        status = "MEDIUM"
    else:
        status = "LOW"

    return {
        "score": score,
        "frequency": frequency,
        "frequency_per_100": round(freq_per_100, 2),
        "first_position_pct": round(first_pos_pct, 1),
        "is_in_intro": is_in_intro,
        "spread": round(spread, 3),
        "status": status,
    }


def analyze_salience(
    text: str,
    main_keyword: Optional[str] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Full spaCy NER + salience analysis.

    Args:
        text: Text to analyze
        main_keyword: Optional primary keyword to track separately
        top_n: Max entities to return (sorted by salience desc)

    Returns:
        Dict with entities, main_keyword_salience, stats, model
    """
    nlp = _get_nlp()

    # Limit text length for performance
    max_chars = 100_000
    truncated = len(text) > max_chars
    analysis_text = text[:max_chars]
    text_len = len(analysis_text)

    doc = nlp(analysis_text)

    # --- NER extraction ---
    entity_counts: Counter = Counter()
    entity_labels: Dict[str, str] = {}
    entity_first_pos: Dict[str, int] = {}

    for ent in doc.ents:
        key = ent.text.strip()
        if len(key) < 2:
            continue
        entity_counts[key] += 1
        if key not in entity_labels:
            entity_labels[key] = ent.label_
            entity_first_pos[key] = ent.start_char

    # --- Compute salience for each entity ---
    entities_with_salience = []
    for entity_text, count in entity_counts.most_common():
        sal = _compute_entity_salience(analysis_text, entity_text, text_len)
        entities_with_salience.append({
            "text": entity_text,
            "label": entity_labels.get(entity_text, "MISC"),
            "ner_count": count,
            "salience": sal,
        })

    # Sort by salience score desc, then frequency desc
    entities_with_salience.sort(
        key=lambda e: (e["salience"]["score"], e["ner_count"]),
        reverse=True,
    )

    top_entities = entities_with_salience[:top_n]

    # --- Main keyword salience ---
    main_kw_salience = None
    if main_keyword:
        main_kw_salience = _compute_entity_salience(
            analysis_text, main_keyword, text_len
        )

    # --- Stats ---
    label_distribution = defaultdict(int)
    for e in entities_with_salience:
        label_distribution[e["label"]] += 1

    stats = {
        "total_entities_found": len(entities_with_salience),
        "unique_entity_count": len(entity_counts),
        "text_length": text_len,
        "truncated": truncated,
        "label_distribution": dict(label_distribution),
        "avg_salience": round(
            sum(e["salience"]["score"] for e in entities_with_salience)
            / max(len(entities_with_salience), 1),
            1,
        ),
    }

    return {
        "entities": top_entities,
        "main_keyword_salience": main_kw_salience,
        "stats": stats,
        "model": _nlp_model_name or "unknown",
    }


# ════════════════════════════════════════════════════════════════════
# FLASK ROUTE (register in master_api.py)
# ════════════════════════════════════════════════════════════════════

def register_routes(app):
    """Register /api/spacy/salience endpoint."""
    from flask import request, jsonify

    @app.route("/api/spacy/salience", methods=["POST"])
    def spacy_salience():
        """
        spaCy NER + entity salience analysis.

        Request:
            {"text": "...", "main_keyword": "...", "top_n": 20}
        Response:
            {"entities": [...], "main_keyword_salience": {...}, "stats": {...}, "model": "..."}
        """
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Invalid or missing JSON body"}), 400

            text = data.get("text", "")
            if not text:
                return jsonify({"error": "text is required"}), 400

            main_keyword = data.get("main_keyword", None)
            top_n = data.get("top_n", 20)

            if not isinstance(top_n, int) or top_n < 1:
                top_n = 20

            result = analyze_salience(
                text=text,
                main_keyword=main_keyword,
                top_n=top_n,
            )

            return jsonify(result)

        except RuntimeError as e:
            return jsonify({
                "error": "spaCy model not available",
                "detail": str(e),
            }), 503
        except Exception as e:
            print(f"[SPACY_SALIENCE] ❌ Error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/spacy/status", methods=["GET"])
    def spacy_status():
        """Health-check for spaCy NLP module."""
        try:
            nlp = _get_nlp()
            from nlp_config import get_nlp_info
            info = get_nlp_info()
            return jsonify({
                "status": "OK",
                "model": info.get("loaded_model", "unknown"),
                "mode": info.get("mode", "unknown"),
                "has_morfeusz": info.get("has_morfeusz", False),
                "available_models": info.get("available_models", []),
                "pipeline": list(nlp.pipe_names),
            })
        except Exception as e:
            return jsonify({
                "status": "ERROR",
                "error": str(e),
            }), 503
