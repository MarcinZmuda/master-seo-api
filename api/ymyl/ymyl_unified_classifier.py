"""
═══════════════════════════════════════════════════════════════════════
🛡️ UNIFIED YMYL CLASSIFIER v1.0 — Detect + Enrich in one Claude call
═══════════════════════════════════════════════════════════════════════

Replaces separate keyword-based legal/medical detection with
a single Claude Sonnet call that:

1. CLASSIFIES the topic (prawo / zdrowie / finanse / general)
2. ENRICHES with source hints:
   - Legal: specific law articles (art. 178a k.k., art. 87 k.w.)
   - Medical: conditions, ICD-10, drugs, MeSH terms, specialization
   - Finance: regulations, institutions
3. Returns search queries for downstream source fetching

Cost: ~$0.01 per call Sonnet — negligible vs Opus content generation.

Usage in app.py:
    result = detect_and_enrich(main_keyword)
    if result["is_legal"]:
        # Use result["legal"]["articles"] for SAOS search
        # Use result["legal"]["search_queries"] for judgment lookup
    if result["is_medical"]:
        # Use result["medical"]["mesh_terms"] for PubMed
        # Use result["medical"]["search_queries"] for source lookup

═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import re
from typing import Dict, Any, Optional, List

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLASSIFIER_MODEL = os.getenv("YMYL_CLASSIFIER_MODEL", "claude-sonnet-4-5-20250514")

_client = None

try:
    import anthropic
    if ANTHROPIC_API_KEY:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print(f"[YMYL_UNIFIED] ✅ Claude classifier ready ({CLASSIFIER_MODEL})")
    else:
        print("[YMYL_UNIFIED] ⚠️ ANTHROPIC_API_KEY not set")
except ImportError:
    print("[YMYL_UNIFIED] ⚠️ anthropic not installed")


# ════════════════════════════════════════════════════════════════════
# PROMPT — single call: classify + enrich
# ════════════════════════════════════════════════════════════════════

UNIFIED_PROMPT = """Jesteś ekspertem od klasyfikacji treści YMYL (Your Money or Your Life) dla polskiego SEO.

TEMAT ARTYKUŁU: "{topic}"

ZADANIE: Sklasyfikuj temat i — jeśli jest YMYL — podaj KONKRETNE źródła do zacytowania.

═══ KATEGORIE ═══
• "prawo" — Każdy temat dotyczący przepisów, kar, postępowań, praw, obowiązków, umów.
  Przykłady: jazda po alkoholu, alimenty, rozwód, mandat, eksmisja, spadek, umowa o pracę,
  zakup mieszkania (aspekt prawny), konfiskata, wykroczenie, przestępstwo, odszkodowanie.
  
• "zdrowie" — Każdy temat dotyczący chorób, leków, terapii, objawów, badań, diety zdrowotnej.
  Przykłady: cukrzyca, nadciśnienie, depresja, lek na X, objawy Y, dieta przy Z,
  szczepionki, rehabilitacja, ciąża, zespół Turnera, ADHD, migrena.

• "finanse" — Każdy temat dotyczący pieniędzy, inwestycji, podatków, kredytów, ubezpieczeń.
  Przykłady: PIT-37, kredyt hipoteczny, OFE, kryptowaluty, emerytura, PPK, lokata.

• "general" — Tematy nie-YMYL (hobby, kulinaria, ogrodnictwo, rozrywka, turystyka).

═══ DLA PRAWA — podaj konkretne przepisy ═══
Zidentyfikuj 2-5 artykułów ustaw, które są PODSTAWĄ PRAWNĄ tematu.
Używaj skrótów: k.c., k.r.o., k.p.c., k.k., k.p., k.w., k.s.h.
Pełne nazwy ustaw szczególnych (np. "Ustawa prawo o ruchu drogowym z dnia 20.06.1997").

═══ DLA ZDROWIA — podaj kontekst medyczny ═══
Zidentyfikuj:
- Warunek/choroba medyczna (polską i łacińską nazwę)
- Specjalizacja lekarska
- 2-3 terminy MeSH (angielskie, do wyszukiwania PubMed)
- Kluczowe leki/substancje (jeśli dotyczy)
- Kod ICD-10 (jeśli znasz)

═══ DLA FINANSÓW — podaj kontekst regulacyjny ═══
- Akty prawne (ustawy, rozporządzenia)
- Instytucje (KNF, NBP, ZUS, US)
- Formularze/deklaracje (jeśli dotyczy)

═══ FORMAT ODPOWIEDZI — TYLKO JSON ═══

Jeśli PRAWO:
{{
  "category": "prawo",
  "confidence": 0.95,
  "reasoning": "Krótkie uzasadnienie",
  "legal": {{
    "articles": ["art. 178a § 1 k.k.", "art. 87 § 1 k.w."],
    "acts": ["Kodeks karny", "Kodeks wykroczeń", "Ustawa prawo o ruchu drogowym"],
    "search_queries": ["art 178a kk", "jazda pod wpływem alkoholu orzeczenia"],
    "key_concepts": ["stan nietrzeźwości", "stan po użyciu alkoholu", "zakaz prowadzenia pojazdów"]
  }}
}}

Jeśli ZDROWIE:
{{
  "category": "zdrowie",
  "confidence": 0.90,
  "reasoning": "Krótkie uzasadnienie",
  "medical": {{
    "condition": "Cukrzyca typu 2",
    "condition_latin": "Diabetes mellitus type 2",
    "icd10": "E11",
    "specialization": "endokrynologia",
    "mesh_terms": ["Diabetes Mellitus, Type 2", "Metformin", "Glycemic Control"],
    "key_drugs": ["metformina", "glimepiryd", "insulina"],
    "search_queries": ["diabetes type 2 treatment guidelines", "cukrzyca typu 2 leczenie"],
    "evidence_note": "Powołuj się na wytyczne PTD 2024, ADA Standards of Care"
  }}
}}

Jeśli FINANSE:
{{
  "category": "finanse",
  "confidence": 0.85,
  "reasoning": "Krótkie uzasadnienie",
  "finance": {{
    "regulations": ["Ustawa o podatku dochodowym od osób fizycznych"],
    "institutions": ["US", "KIS"],
    "forms": ["PIT-37", "PIT-36"],
    "search_queries": ["rozliczenie PIT 2024", "podatek dochodowy osoby fizyczne"]
  }}
}}

Jeśli GENERAL:
{{
  "category": "general",
  "confidence": 0.80,
  "reasoning": "Krótkie uzasadnienie"
}}"""


# ════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════

def detect_and_enrich(
    main_keyword: str,
    additional_keywords: List[str] = None,
) -> Dict[str, Any]:
    """
    Unified YMYL detection + enrichment.
    
    Returns:
    {
        "category": "prawo"|"zdrowie"|"finanse"|"general",
        "is_ymyl": True/False,
        "is_legal": True/False,
        "is_medical": True/False,
        "is_finance": True/False,
        "confidence": 0.0-1.0,
        "reasoning": "...",
        "detection_method": "claude_sonnet_unified",
        
        # Only if legal:
        "legal": {
            "articles": ["art. 178a § 1 k.k."],
            "acts": ["Kodeks karny"],
            "search_queries": ["art 178a kk"],
            "key_concepts": ["stan nietrzeźwości"]
        },
        
        # Only if medical:
        "medical": {
            "condition": "...",
            "mesh_terms": ["..."],
            "search_queries": ["..."],
            ...
        },
        
        # Only if finance:
        "finance": { ... }
    }
    """
    if not _client:
        print("[YMYL_UNIFIED] ⚠️ Claude not available, falling back to keyword detection")
        return _keyword_fallback(main_keyword, additional_keywords)
    
    additional = additional_keywords or []
    topic = main_keyword
    if additional:
        topic += f" (kontekst: {', '.join(additional[:5])})"
    
    try:
        response = _client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=500,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": UNIFIED_PROMPT.format(topic=topic)
            }]
        )
        
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            print(f"[YMYL_UNIFIED] ⚠️ No JSON in response: {raw[:100]}")
            return _keyword_fallback(main_keyword, additional_keywords)
        
        result = json.loads(json_match.group())
        
        # Validate category
        category = result.get("category", "general")
        if category not in ("prawo", "zdrowie", "finanse", "general"):
            category = "general"
        
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        is_ymyl = category != "general" and confidence >= 0.5
        
        output = {
            "category": category,
            "is_ymyl": is_ymyl,
            "is_legal": category == "prawo" and is_ymyl,
            "is_medical": category == "zdrowie" and is_ymyl,
            "is_finance": category == "finanse" and is_ymyl,
            "confidence": round(confidence, 2),
            "reasoning": result.get("reasoning", ""),
            "detection_method": "claude_sonnet_unified",
        }
        
        # Attach enrichment data
        if category == "prawo" and "legal" in result:
            output["legal"] = result["legal"]
        elif category == "zdrowie" and "medical" in result:
            output["medical"] = result["medical"]
        elif category == "finanse" and "finance" in result:
            output["finance"] = result["finance"]
        
        print(f"[YMYL_UNIFIED] ✅ '{main_keyword}' → {category} ({confidence}) "
              f"| {result.get('reasoning', '')[:60]}")
        
        if category == "prawo" and "legal" in result:
            arts = result["legal"].get("articles", [])
            print(f"[YMYL_UNIFIED]    📜 Przepisy: {', '.join(arts[:5])}")
        elif category == "zdrowie" and "medical" in result:
            mesh = result["medical"].get("mesh_terms", [])
            print(f"[YMYL_UNIFIED]    🏥 MeSH: {', '.join(mesh[:3])}")
        
        return output
        
    except json.JSONDecodeError as e:
        print(f"[YMYL_UNIFIED] ⚠️ JSON parse error: {e}")
        return _keyword_fallback(main_keyword, additional_keywords)
    except Exception as e:
        print(f"[YMYL_UNIFIED] ⚠️ Claude error: {e}")
        return _keyword_fallback(main_keyword, additional_keywords)


# ════════════════════════════════════════════════════════════════════
# KEYWORD FALLBACK — when Claude unavailable
# ════════════════════════════════════════════════════════════════════

# Minimal keyword sets — just for fallback, not primary detection
_LEGAL_SIGNALS = {
    "kodeks", "ustawa", "sąd", "wyrok", "kara", "grzywna", "mandat",
    "przestępstwo", "wykroczenie", "prawo", "przepis", "artykuł",
    "pozew", "apelacja", "alimenty", "rozwód", "spadek", "testament",
    "odszkodowanie", "umowa", "notariusz", "adwokat", "komornik",
    "pozbawienie wolności", "zakaz prowadzenia", "promil", "alkohol",
    "narkotyki", "konfiskata", "recydywa", "prokuratura",
}

_MEDICAL_SIGNALS = {
    "chorob", "leczeni", "terapi", "lekarz", "szpital", "objawy",
    "lek ", "dawkowani", "diagno", "operacj", "rehabilitacj",
    "szczepion", "cukrzyc", "nadciśnieni", "nowotwór", "depresj",
    "antybiotyk", "insulin", "astm", "zawał", "migren", "alergi",
    "zespół", "syndrom", "zapaleni", "infekcj",
}

_FINANCE_SIGNALS = {
    "podatek", "pit", "vat", "cit", "kredyt", "pożyczk", "hipote",
    "inwesty", "akcj", "obligacj", "giełd", "emerytur", "zus",
    "ubezpiecz", "polis", "oszczędnoś", "lokata", "krypto",
}


def _keyword_fallback(main_keyword: str, additional_keywords: List[str] = None) -> Dict[str, Any]:
    """Simple keyword fallback when Claude is unavailable."""
    all_text = main_keyword.lower()
    if additional_keywords:
        all_text += " " + " ".join(k.lower() for k in additional_keywords)
    
    legal_hits = sum(1 for s in _LEGAL_SIGNALS if s in all_text)
    medical_hits = sum(1 for s in _MEDICAL_SIGNALS if s in all_text)
    finance_hits = sum(1 for s in _FINANCE_SIGNALS if s in all_text)
    
    scores = {"prawo": legal_hits, "zdrowie": medical_hits, "finanse": finance_hits}
    best = max(scores, key=scores.get)
    best_score = scores[best]
    
    if best_score == 0:
        return {
            "category": "general",
            "is_ymyl": False,
            "is_legal": False,
            "is_medical": False,
            "is_finance": False,
            "confidence": 0.0,
            "reasoning": "Brak sygnałów YMYL (keyword fallback)",
            "detection_method": "keyword_fallback",
        }
    
    confidence = min(1.0, best_score / 3)
    is_ymyl = confidence >= 0.3
    
    return {
        "category": best,
        "is_ymyl": is_ymyl,
        "is_legal": best == "prawo" and is_ymyl,
        "is_medical": best == "zdrowie" and is_ymyl,
        "is_finance": best == "finanse" and is_ymyl,
        "confidence": round(confidence, 2),
        "reasoning": f"Keyword fallback: {best_score} sygnałów {best}",
        "detection_method": "keyword_fallback",
    }


# ════════════════════════════════════════════════════════════════════
# FLASK ROUTE (register in master_api.py)
# ════════════════════════════════════════════════════════════════════

def register_routes(app):
    """Register /api/ymyl/detect_and_enrich endpoint."""
    from flask import request, jsonify
    
    @app.route("/api/ymyl/detect_and_enrich", methods=["POST"])
    def ymyl_detect_and_enrich():
        """
        Unified YMYL detection + enrichment.
        
        Request: {"main_keyword": "jazda po alkoholu", "additional_keywords": []}
        Response: {category, is_ymyl, is_legal, is_medical, confidence, legal/medical/finance enrichment}
        """
        data = request.get_json() or {}
        main_keyword = data.get("main_keyword", "")
        
        if not main_keyword:
            return jsonify({"error": "main_keyword is required"}), 400
        
        result = detect_and_enrich(
            main_keyword=main_keyword,
            additional_keywords=data.get("additional_keywords", [])
        )
        
        return jsonify(result)
