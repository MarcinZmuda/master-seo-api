import os
import json
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz          # fuzzy-matching (stringowy)
import language_tool_python         # QA językowe
import textstat                     # czytelność
import textdistance                 # drugi fuzzy layer (tokenowy)
import pysbd                        # segmentacja zdań

tracker_routes = Blueprint("tracker_routes", __name__)

# -----------------------------------------------------------
# 🧠 spaCy – ładowane raz
# -----------------------------------------------------------
nlp = spacy.load("pl_core_news_sm")

# -----------------------------------------------------------
# 🎯 Parametry fuzzy-matchingu dla trackera
# -----------------------------------------------------------
FUZZY_SIMILARITY_THRESHOLD = 90      # próg podobieństwa 0–100 (rapidfuzz)
MAX_FUZZY_WINDOW_EXPANSION = 2       # ile dodatkowych lematów dopuszczamy w środku frazy
JACCARD_SIMILARITY_THRESHOLD = 0.8   # próg podobieństwa Jaccarda (0–1)

# -----------------------------------------------------------
# 🧪 Inicjalizacja LanguageTool dla polskiego
# -----------------------------------------------------------
try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception as e:
    print(f"[LanguageTool] Init error: {e}")
    LT_TOOL_PL = None

# -----------------------------------------------------------
# ✂️ Pysbd – segmentacja zdań (PL)
# -----------------------------------------------------------
SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

# -----------------------------------------------------------
# 🌐 Konfiguracja Gemini
# -----------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# ⚖️ GEMINI JUDGE (Model Stable: gemini-pro)
# ===========================================================
def evaluate_with_gemini(text, meta_trace):
    if not GEMINI_API_KEY:
        return {
            "pass": True,
            "quality_score": 100,
            "feedback_for_writer": "Brak klucza Gemini - skip check"
        }

    try:
        model = genai.GenerativeModel("gemini-pro")
    except Exception:
        return {
            "pass": True,
            "quality_score": 80,
            "feedback_for_writer": "Model Init Error"
        }

    meta = meta_trace or {}
    intent = meta.get("execution_intent", "Brak")
    rhythm = meta.get("rhythm_pattern_used", "Brak")

    banned_phrases_list = """
    1. WYPEŁNIACZE: "W dzisiejszych czasach", "W dobie...", "Od zarania dziejów".
    2. ŁĄCZNIKI: "Warto zauważyć", "Należy wspomnieć", "Warto dodać".
    3. ZAKOŃCZENIA: "Podsumowując", "Reasumując", "W ostatecznym rozrachunku".
    4. IDIOMY: "Gra warta świeczki", "Strzał w dziesiątkę".
    5. ASEKURANCTWO: "Wszystko zależy od...", "Nie ma jednoznacznej odpowiedzi".
    6. WZMOCNIENIA: "Nie ma wątpliwości", "Bez wątpienia".
    7. ZNAKI: "—" (Długi myślnik/Pauza).
    """

    prompt = f"""
    Jesteś Sędzią Jakości SEO.
    Oceniasz fragment tekstu pod kątem naturalności i braku "AI-izmów".
    
    INTENCJA: {intent} | RYTM: {rhythm}
    BANNED LIST: {banned_phrases_list}
    
    Zwróć JSON:
    {{
        "pass": true/false,
        "quality_score": (0-100),
        "feedback_for_writer": "Instrukcja co poprawić"
    }}
    TEKST: "{text}"
    """

    try:
        response = model.generate_content(prompt)
        clean = (
            response.text.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        return json.loads(clean)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {
            "pass": True,
            "quality_score": 80,
            "feedback_for_writer": f"Gemini API Error: {str(e)}"
        }


# ===========================================================
# 🔎 LanguageTool + pysbd + textstat – Polish QA Gate (raport)
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    """
    Analiza jakości językowej batcha:
    - błędy / sugestie z LanguageTool (polski),
    - metryki czytelności z textstat,
    - liczba zdań i średnia długość zdań z pysbd.

    Nie blokuje tekstu – służy jako raport dla UI / meta_prompt.
    """
    result = {
        "lt_enabled": LT_TOOL_PL is not None,
        "lt_issues_count": 0,
        "lt_issues_sample": [],
        "readability": {},
        "error": None,
    }

    if not text or not text.strip():
        return result

    # ---------- LanguageTool – błędy / sugestie ----------
    if LT_TOOL_PL is not None:
        try:
            matches = LT_TOOL_PL.check(text)
            result["lt_issues_count"] = len(matches)
            sample = []
            for m in matches[:10]:
                sample.append(
                    {
                        "offset": m.offset,
                        "length": m.errorLength,
                        "rule_id": m.ruleId,
                        "message": m.message,
                        "replacements": m.replacements[:5],
                    }
                )
            result["lt_issues_sample"] = sample
        except Exception as e:
            print(f"[LanguageTool] Check error: {e}")
            result["error"] = f"LanguageTool error: {str(e)}"

    # ---------- pysbd + textstat – czytelność ----------
    try:
        raw_sentences = SENTENCE_SEGMENTER.segment(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        sentence_count = len(sentences)
        word_count = sum(len(s.split()) for s in sentences) if sentences else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        smog_index = textstat.smog_index(text)

        readability = {
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "smog_index": smog_index,
            "avg_sentence_length": avg_sentence_length,
            "sentence_count": sentence_count,
            "word_count": word_count,
        }
        result["readability"] = readability
    except Exception as e:
        print(f"[pysbd/textstat] Error: {e}")
        if result["error"] is None:
            result["error"] = f"readability error: {str(e)}"

    return result


# ===========================================================
# 🛠 LanguageTool – AUTO-FIX (mechaniczne poprawki)
# ===========================================================
def apply_languagetool_fixes(text: str) -> str:
    """
    Automatyczne poprawianie tekstu na bazie LanguageTool:
    - bierze pierwszą proponowaną poprawkę dla każdego błędu,
    - stosuje ją od końca tekstu, żeby nie popsuć offsetów.
    """
    if LT_TOOL_PL is None or not text or not text.strip():
        return text

    try:
        matches = LT_TOOL_PL.check(text)
    except Exception as e:
        print(f"[LanguageTool] Auto-fix error: {e}")
        return text

    text_list = list(text)
    for m in sorted(matches, key=lambda x: x.offset, reverse=True):
        if not m.replacements:
            continue
        replacement = m.replacements[0]
        start = m.offset
        end = m.offset + m.errorLength
        text_list[start:end] = list(replacement)

    return "".join(text_list)


# ===========================================================
# 🔍 Fuzzy na lematyzowanym tekście (dla trackera)
# ===========================================================
def _count_fuzzy_on_lemmas(target_tokens, text_lemma_list, exact_spans):
    """
    target_tokens: lista lematów frazy, np. ["adwokat", "rozwodowy", "warszawa"]
    text_lemma_list: lista lematów całego batcha
    exact_spans: lista (start, end) dla exact lemma matchy (żeby ich nie dublować)

    Używamy:
    - rapidfuzz.token_set_ratio (stringowo),
    - textdistance.jaccard (tokenowo).
    """
    if not target_tokens or not text_lemma_list:
        return 0

    text_len = len(text_lemma_list)
    target_len = len(target_tokens)
    if target_len == 0 or text_len < target_len:
        return 0

    kw_str = " ".join(target_tokens)
    kw_tokens_set = set(target_tokens)

    used_positions = set()
    for s, e in exact_spans:
        used_positions.update(range(s, e))

    fuzzy_hits = 0

    for start in range(text_len):
        for extra in range(MAX_FUZZY_WINDOW_EXPANSION + 1):
            end = start + target_len + extra
            if end > text_len:
                break

            window_positions = range(start, end)
            if any(pos in used_positions for pos in window_positions):
                continue

            window_tokens = text_lemma_list[start:end]
            if not window_tokens:
                continue

            window_str = " ".join(window_tokens)

            rf_score = fuzz.token_set_ratio(kw_str, window_str)
            jaccard_score = textdistance.jaccard(
                kw_tokens_set,
                set(window_tokens)
            )

            if (
                rf_score >= FUZZY_SIMILARITY_THRESHOLD
                or jaccard_score >= JACCARD_SIMILARITY_THRESHOLD
            ):
                fuzzy_hits += 1
                used_positions.update(window_positions)
                break

    return fuzzy_hits


# ===========================================================
# 🧠 HYBRID COUNTING
# ===========================================================
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma):
    """
    Hybrydowe zliczanie:
    - Exact na surowym tekście (case-insensitive)
    - Exact na lematyzowanym tekście
    - Fuzzy na lematyzowanym tekście (rapidfuzz + textdistance)
    """
    text_lower = text_raw.lower()
    target_exact_lower = target_exact.lower()

    exact_hits = 0
    if target_exact_lower.strip():
        exact_hits = text_lower.count(target_exact_lower)

    lemma_hits = 0
    exact_spans = []
    target_tokens = target_lemma.split()

    if target_tokens:
        target_len = len(target_tokens)
        text_len = len(text_lemma_list)

        for i in range(text_len - target_len + 1):
            if text_lemma_list[i: i + target_len] == target_tokens:
                lemma_hits += 1
                exact_spans.append((i, i + target_len))

        fuzzy_hits = _count_fuzzy_on_lemmas(
            target_tokens=target_tokens,
            text_lemma_list=text_lemma_list,
            exact_spans=exact_spans,
        )
        lemma_hits += fuzzy_hits

    return max(exact_hits, lemma_hits)


def compute_status(actual, target_min, target_max):
    if actual < target_min:
        return "UNDER"
    tolerance = max(2, int(target_max * 0.2))
    if actual > (target_max + tolerance):
        return "OVER"
    return "OK"


def global_keyword_stats(keywords_state):
    under = sum(1 for v in keywords_state.values() if v["status"] == "UNDER")
    over = sum(1 for v in keywords_state.values() if v["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for v in keywords_state.values() if v["status"] == "OK")
    return under, over, locked, ok


# ===========================================================
# 🆕 ENDPOINT: /api/language_refine – auto-fix + audyt
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    data = request.get_json(force=True) or {}
    text = data.get("text", "") or ""

    auto_fixed = apply_languagetool_fixes(text)
    audit = analyze_language_quality(auto_fixed)

    return jsonify(
        {
            "original_text": text,
            "auto_fixed_text": auto_fixed,
            "language_audit": audit,
        }
    )


# ===========================================================
# 🧠 MAIN PROCESS (HARD SEO VETO)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()

    if not doc.exists:
        return {"error": "Project not found", "status": 404}

    language_audit = analyze_language_quality(batch_text)
    gemini_verdict = {"pass": True, "quality_score": 100}
    if meta_trace is not None:
        gemini_verdict = evaluate_with_gemini(batch_text, meta_trace)

    QUALITY_THRESHOLD = 70
    if not gemini_verdict.get("pass", True) or gemini_verdict.get(
        "quality_score", 100
    ) < QUALITY_THRESHOLD:
        return {
            "status": "REJECTED_QUALITY",
            "error": "Quality Gate Failed",
            "gemini_feedback": gemini_verdict,
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony przez Gemini (Styl/Jakość).",
        }

    data = doc.to_dict()
    import copy

    keywords_state = copy.deepcopy(data.get("keywords_state", {}))

    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]

    for row_id, meta in keywords_state.items():
        original_keyword = meta.get("keyword", "")
        target_exact = meta.get("search_term_exact", original_keyword.lower())
        target_lemma = meta.get("search_lemma", "")

        if not target_lemma:
            doc_tmp = nlp(original_keyword)
            target_lemma = " ".join(
                [t.lemma_.lower() for t in doc_tmp if t.is_alpha]
            )

        occurrences = count_hybrid_occurrences(
            batch_text,
            text_lemma_list,
            target_exact,
            target_lemma,
        )
        meta["actual_uses"] += occurrences
        meta["status"] = compute_status(
            meta["actual_uses"], meta["target_min"], meta["target_max"]
        )

    under, over, locked, ok = global_keyword_stats(keywords_state)

    if locked >= 4 or over >= 15:
        return {
            "status": "REJECTED_SEO",
            "error": "SEO Limits Exceeded",
            "gemini_feedback": {
                "pass": False,
                "quality_score": gemini_verdict.get("quality_score", 0),
                "feedback_for_writer": (
                    f"SEO CRITICAL: Tekst przeoptymalizowany! "
                    f"LOCKED={locked}, OVER={over}. "
                    f"Zredukuj użycie słów kluczowych."
                ),
            },
            "language_audit": language_audit,
            "quality_alert": True,
            "info": "Tekst odrzucony przez Hard SEO Veto.",
        }

    batch_entry = {
        "text": batch_text,
        "gemini_audit": gemini_verdict,
        "language_audit": language_audit,
        "summary": {"under": under, "over": over, "locked": locked, "ok": ok},
    }

    if "batches" not in data:
        data["batches"] = []
    data["batches"].append(batch_entry)
    data["total_batches"] = len(data["batches"])
    data["keywords_state"] = keywords_state

    doc_ref.set(data)

    readability = language_audit.get("readability", {})
    sentence_count = readability.get("sentence_count", 0)
    avg_sentence_length = readability.get("avg_sentence_length", 0.0)

    meta_prompt_summary = (
        f"UNDER={under}, OVER={over}, LOCKED={locked} | "
        f"Quality={gemini_verdict.get('quality_score')}% | "
        f"LT_issues={language_audit.get('lt_issues_count', 0)} | "
        f"Sentences={sentence_count}, Avg_len={avg_sentence_length:.1f}"
    )

    return {
        "status": "BATCH_ACCEPTED",
        "counting_mode": "uuid_hybrid",
        "gemini_feedback": gemini_verdict,
        "language_audit": language_audit,
        "quality_alert": False,
        "keywords_report": [
            {
                "keyword": meta.get("keyword", "Unknown"),
                "actual_uses": meta["actual_uses"],
                "target_range": f"{meta['target_min']}–{meta['target_max']}",
                "status": meta["status"],
                "priority_instruction": (
                    "INCREASE"
                    if meta["status"] == "UNDER"
                    else "DECREASE"
                    if meta["status"] == "OVER"
                    else "IGNORE"
                ),
            }
            for row_id, meta in sorted(
                keywords_state.items(),
                key=lambda item: item[1].get("keyword", ""),
            )
        ],
        "meta_prompt_summary": meta_prompt_summary,
    }


# ===========================================================
# 🔍 Endpointy GET – podgląd projektu i liczników
# ===========================================================
@tracker_routes.get("/api/project/<project_id>")
def get_project(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    return jsonify(doc.to_dict()), 200


@tracker_routes.get("/api/project/<project_id>/keywords")
def get_keywords_state(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    return jsonify(doc.to_dict().get("keywords_state", {})), 200
