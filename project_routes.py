# ================================================================
# project_routes.py — Warstwa Project Management (v7.0.2 - Stable + Meta Prompt Summary)
# ================================================================

import json
import re
from flask import Blueprint, request, jsonify
from datetime import datetime
import requests
from firebase_admin import firestore
import spacy
from statistics import mean

# ---------------------------------------------------------------
# 🔐 Inicjalizacja Firebase i spaCy
# ---------------------------------------------------------------
db = None
project_bp = Blueprint("project_routes", __name__)

try:
    NLP = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie.")
except OSError:
    NLP = None
    print("❌ BŁĄD: Nie można załadować modelu spaCy 'pl_core_news_sm'.")


# ---------------------------------------------------------------
# 🧠 Funkcje językowe
# ---------------------------------------------------------------
def lemmatize_text(text):
    """Zwraca listę lematów z tekstu (bez interpunkcji)."""
    if not NLP:
        return re.findall(r'\b\w+\b', text.lower())
    doc = NLP(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]


def get_root_prefix(word):
    """Wyznacza dynamiczny rdzeń semantyczny słowa."""
    if len(word) <= 6:
        return word
    vowels = 'aeiouyąęó'
    root_len = 6
    for i, ch in enumerate(word):
        if ch in vowels:
            root_len = i + 3
            break
    return word[:max(6, root_len)]


def extract_context_matches(text, root_prefix, related_terms=None):
    """Wykrywa kontekstowe wystąpienia rdzenia w otoczeniu powiązanych terminów."""
    if not NLP:
        return 0
    doc = NLP(text.lower())
    related_terms = related_terms or []
    contextual_matches = 0

    for token in doc:
        if token.lemma_.startswith(root_prefix):
            context = " ".join([t.text for t in token.subtree])
            if any(term in context for term in related_terms):
                contextual_matches += 1
    return contextual_matches


# ---------------------------------------------------------------
# 🔧 Pomocnicze funkcje Firestore
# ---------------------------------------------------------------
def call_s1_analysis(topic):
    """Bezpieczne wywołanie analizy S1 z timeoutem."""
    try:
        r = requests.post("http://localhost:10000/api/s1_analysis", json={"topic": topic}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Błąd S1 Analysis: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------
# 🧩 Parser briefu z obsługą BASIC / EXTENDED
# ---------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """Parsuje tekst briefu i wyciąga słowa kluczowe (BASIC / EXTENDED) + nagłówki H2."""
    keywords_dict = {}
    headers_list = []

    cleaned_text = "\n".join([s.strip() for s in brief_text.splitlines() if s.strip()])
    section_regex = r"((?:BASIC|EXTENDED|H2)\s+(?:TEXT|HEADERS)\s+TERMS)\s*:\s*=*\s*([\s\S]*?)(?=\n[A-Z\s]+(?:TEXT|HEADERS)\s+TERMS|$)"
    keyword_regex = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$", re.UNICODE)
    keyword_regex_single = re.compile(r"^\s*(.*?)\s*:\s*(\d+)x\s*$", re.UNICODE)

    for match in re.finditer(section_regex, cleaned_text, re.IGNORECASE):
        section_name_raw = match.group(1).upper()
        section_content = match.group(2)

        section_type = "BASIC"
        if "EXTENDED" in section_name_raw:
            section_type = "EXTENDED"

        if "H2" in section_name_raw:
            for line in section_content.splitlines():
                if line.strip():
                    headers_list.append(line.strip())
            continue

        for line in section_content.splitlines():
            line = line.strip()
            if not line:
                continue

            kw_match = keyword_regex.match(line)
            if kw_match:
                keyword = kw_match.group(1).strip()
                min_val = int(kw_match.group(2))
                max_val = int(kw_match.group(3))
            else:
                kw_match_single = keyword_regex_single.match(line)
                if kw_match_single:
                    keyword = kw_match_single.group(1).strip()
                    min_val = max_val = int(kw_match_single.group(2))
                else:
                    continue

            keyword_lemmas = lemmatize_text(keyword)
            root_prefix = get_root_prefix(keyword_lemmas[0]) if keyword_lemmas else get_root_prefix(keyword)

            if section_type == "EXTENDED":
                min_val = max(1, round(min_val * 0.5))
                max_val = max(1, round(max_val * 0.5))

            keywords_dict[keyword] = {
                "type": section_type,
                "target_min": min_val,
                "target_max": max_val,
                "actual": 0,
                "contextual_hits": 0,
                "semantic_coverage": 0.0,
                "locked": False,
                "lemmas": keyword_lemmas,
                "lemma_len": len(keyword_lemmas),
                "root_prefix": root_prefix
            }

    return keywords_dict, headers_list


# ---------------------------------------------------------------
# ✅ /api/project/create
# ---------------------------------------------------------------
@project_bp.route("/api/project/create", methods=["POST"])
def create_project():
    try:
        global db
        if not db:
            return jsonify({"error": "Firestore nie jest połączony"}), 503
        if not NLP:
            return jsonify({"error": "Model spaCy nie jest załadowany"}), 500

        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "").strip()
        brief_text = data.get("brief_text", "")

        if not topic:
            return jsonify({"error": "Brak 'topic'"}), 400

        print(f"[DEBUG] Tworzenie projektu Firestore: {topic}")
        keywords_state, headers_list = parse_brief_to_keywords(brief_text)

        s1_data = call_s1_analysis(topic)
        if "error" in s1_data:
            print(f"[WARN] S1 Analysis nieudana: {s1_data['error']}")

        doc_ref = db.collection("seo_projects").document()
        doc_ref.set({
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "brief_text": brief_text[:8000],
            "keywords_state": keywords_state,
            "headers_suggestions": headers_list,
            "s1_data": s1_data,
            "batches": [],
            "status": "created"
        })

        print(f"[INFO] ✅ Projekt {doc_ref.id} utworzony ({len(keywords_state)} fraz).")
        return jsonify({
            "status": "✅ Projekt utworzony",
            "project_id": doc_ref.id,
            "topic": topic,
            "keywords": len(keywords_state)
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({
            "status": "ERROR",
            "message": "Nie udało się utworzyć projektu (timeout lub błąd Firestore).",
            "error": str(e)
        }), 500


# ---------------------------------------------------------------
# 🧮 /api/project/<id>/add_batch
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    try:
        global db
        if not db:
            return jsonify({"error": "Brak połączenia z Firestore"}), 503

        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()
        keywords_state = project_data.get("keywords_state", {})
        text_input = (request.get_json(silent=True) or {}).get("text", "")

        if not text_input.strip():
            return jsonify({"error": "Brak treści"}), 400

        text_lower = text_input.lower()
        text_lemmas = lemmatize_text(text_input)
        counts_in_batch = {}

        for kw, meta in keywords_state.items():
            lemmas = meta.get("lemmas", [])
            root_prefix = meta.get("root_prefix", "")
            kw_len = meta.get("lemma_len", 0)
            lemma_count = sum(
                1 for i in range(len(text_lemmas) - kw_len + 1)
                if text_lemmas[i:i + kw_len] == lemmas
            )
            root_count = len(re.findall(rf"\b{re.escape(root_prefix)}\w*\b", text_lower))
            contextual_hits = extract_context_matches(text_input, root_prefix, ["sąd", "wniosek", "osoba", "postępowanie"])
            meta["actual"] += max(lemma_count, root_count)
            meta["contextual_hits"] += contextual_hits
            meta["semantic_coverage"] = round(root_count / (lemma_count + 1), 2)
            counts_in_batch[kw] = max(lemma_count, root_count)

            if meta["actual"] > meta["target_max"] + 3:
                meta["locked"], meta["status"] = True, "LOCKED"
            elif meta["actual"] > meta["target_max"]:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"

        keywords_report = [
            {
                "keyword": k,
                "type": v.get("type", "BASIC"),
                "actual_uses": v.get("actual", 0),
                "contextual_hits": v.get("contextual_hits", 0),
                "semantic_coverage": v.get("semantic_coverage", 0.0),
                "target_range": f"{v.get('target_min', 0)}-{v.get('target_max', 0)}",
                "status": v.get("status", "OK"),
                "priority_instruction": (
                    "DO_NOT_USE" if v.get("status") == "LOCKED"
                    else "PRIORITY_USE" if v.get("status") == "UNDER"
                    else "USE_AS_NEEDED"
                )
            }
            for k, v in keywords_state.items()
        ]

        extended_count = sum(1 for k in keywords_state.values() if k.get("type") == "EXTENDED")
        basic_count = sum(1 for k in keywords_state.values() if k.get("type") == "BASIC")
        print(f"[BATCH {project_id}] ✅ BASIC: {basic_count}, EXTENDED: {extended_count}")

        doc_ref.update({
            "keywords_state": keywords_state,
            "batches": firestore.ArrayUnion([{
                "created_at": datetime.utcnow().isoformat(),
                "text": text_input[:5000],
                "counts": counts_in_batch
            }]),
            "updated_at": datetime.utcnow().isoformat()
        })

        batch_number = len(project_data.get("batches", [])) + 1
        report = generate_semantic_report(batch_number, keywords_report)
        meta_prompt_summary = build_meta_prompt_summary(batch_number, keywords_report)

        print("\n" + "=" * 70)
        print(report)
        print("- META PROMPT SUMMARY -------------------------------")
        print(meta_prompt_summary)
        print("=" * 70 + "\n")

        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "counts_in_batch": counts_in_batch,
            "keywords_report": keywords_report,
            "batch_text_preview": text_input[:5000],
            "meta_prompt_summary": meta_prompt_summary
        }), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/add_batch: {e}")
        return jsonify({"error": f"Błąd /api/project/add_batch: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧹 DELETE /api/project/<id>
# ---------------------------------------------------------------
@project_bp.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project_final(project_id):
    try:
        global db
        if not db:
            return jsonify({"error": "Brak połączenia z Firestore"}), 503

        doc_ref = db.collection("seo_projects").document(project_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje"}), 404

        project_data = doc.to_dict()
        summary = {
            "topic": project_data.get("topic", "nieznany temat"),
            "total_batches": len(project_data.get("batches", [])),
            "total_length": sum(b.get("length", 0) for b in project_data.get("batches", [])),
            "locked_terms": [k for k, v in project_data.get("keywords_state", {}).items() if v.get("locked")],
            "timestamp": datetime.utcnow().isoformat()
        }

        db.collection("seo_projects_archive").document(project_id).set(summary)
        doc_ref.delete()

        return jsonify({"status": f"✅ Projekt {project_id} został usunięty.", "summary": summary}), 200

    except Exception as e:
        print(f"❌ Błąd DELETE: {e}")
        return jsonify({"error": f"Błąd /api/project DELETE: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧠 Meta-prompt summary dla GPT
# ---------------------------------------------------------------
def build_meta_prompt_summary(batch_number, keywords_report):
    """Buduje zwięzły meta-prompt na podstawie raportu słów kluczowych."""
    try:
        under_terms = [k["keyword"] for k in keywords_report if k.get("status") in ["UNDER", "NOT_USED"]]
        over_terms = [k["keyword"] for k in keywords_report if k.get("status") == "OVER"]
        locked_terms = [k["keyword"] for k in keywords_report if k.get("status") == "LOCKED"]
        extended_under = [
            k["keyword"]
            for k in keywords_report
            if k.get("type") == "EXTENDED" and k.get("status") in ["UNDER", "NOT_USED"]
        ]

        summary = (
            f"BATCH {batch_number} – podsumowanie semantyczne.\n"
            f"UNDER (do wzmocnienia): {', '.join(under_terms) or 'brak'}.\n"
            f"OVER (ogranicz / nie wzmacniaj): {', '.join(over_terms) or 'brak'}.\n"
            f"LOCKED (zakaz użycia): {', '.join(locked_terms) or 'brak'}.\n"
            f"EXTENDED do uzupełnienia: {', '.join(extended_under) or 'brak'}.\n"
            "DZIAŁANIE: w kolejnym batchu wzmacniaj UNDER, "
            "nie używaj LOCKED, redukuj OVER do minimum, a EXTENDED dawkuj naturalnie."
        )
        return summary
    except Exception as e:
        return f"[BŁĄD META-PROMPT SUMMARY] {e}"


# ---------------------------------------------------------------
# 🧩 Lokalny raport semantyczny
# ---------------------------------------------------------------
def generate_semantic_report(batch_number, keywords_report):
    try:
        over_terms = [k["keyword"] for k in keywords_report if k["status"] in ["OVER", "LOCKED"]]
        under_terms = [k["keyword"] for k in keywords_report if k["status"] in ["UNDER", "NOT_USED"]]
        extended_under = [k["keyword"] for k in keywords_report if k["type"] == "EXTENDED" and k["status"] in ["UNDER", "NOT_USED"]]
        avg_sem = round(mean([k["semantic_coverage"] for k in keywords_report]), 2)
        avg_ctx = round(mean([k["contextual_hits"] for k in keywords_report]), 2)

        report = (
            f"─────────────── SEMANTIC REPORT BATCH {batch_number} ───────────────\n"
            f"Frazy zredukowane: {', '.join(over_terms) or 'brak'}\n"
            f"Frazy priorytetowe: {', '.join(under_terms) or 'brak'}\n"
            f"EXTENDED do uzupełnienia: {', '.join(extended_under) or 'brak'}\n"
            f"→ semantic_coverage: {avg_sem} | contextual_hits: {avg_ctx}\n"
            "──────────────────────────────────────────────────────────────\n"
        )
        return report
    except Exception as e:
        return f"[BŁĄD RAPORTU] {e}"


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    global db
    db = _db
    app.register_blueprint(project_bp)
    print("✅ [INIT] project_routes zarejestrowany.")
