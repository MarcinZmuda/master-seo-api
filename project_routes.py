# ================================================================
# project_routes.py — Warstwa Project Management (v6.9.1 - Semantic Root + Context Matching + Local Report)
# ================================================================

import json
import base64
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
    try:
        r = requests.post("http://localhost:10000/api/s1_analysis", json={"topic": topic}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Błąd wywołania S1 Analysis: {str(e)}"}


def parse_brief_to_keywords(brief_text):
    """Parsuje tekst briefu i wyciąga słowa kluczowe + nagłówki H2."""
    keywords_dict = {}
    headers_list = []

    cleaned_text = "\n".join([s.strip() for s in brief_text.splitlines() if s.strip()])
    section_regex = r"((?:BASIC|EXTENDED|H2)\s+(?:TEXT|HEADERS)\s+TERMS)\s*:\s*=*\s*([\s\S]*?)(?=\n[A-Z\s]+(?:TEXT|HEADERS)\s+TERMS|$)"
    keyword_regex = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*-\s*(\d+)x\s*$", re.UNICODE)
    keyword_regex_single = re.compile(r"^\s*(.*?)\s*:\s*(\d+)x\s*$", re.UNICODE)

    for match in re.finditer(section_regex, cleaned_text, re.IGNORECASE):
        section_name_raw = match.group(1).upper()
        section_content = match.group(2)

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

            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "actual": 0,
                "actual_tokens": 0,
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

        keywords_state, headers_list = parse_brief_to_keywords(brief_text)
        s1_data = call_s1_analysis(topic)

        doc_ref = db.collection("seo_projects").document()
        doc_ref.set({
            "topic": topic,
            "created_at": datetime.utcnow().isoformat(),
            "brief_text": brief_text[:5000],
            "keywords_state": keywords_state,
            "headers_suggestions": headers_list,
            "s1_data": s1_data,
            "batches": [],
            "status": "created"
        })

        return jsonify({
            "status": "✅ Projekt utworzony",
            "project_id": doc_ref.id,
            "topic": topic,
            "keywords": len(keywords_state)
        }), 201

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project/create: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧮 /api/project/<id>/add_batch — dodaje batch treści
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
        batches = project_data.get("batches", [])

        text_input = (request.get_json(silent=True) or {}).get("text", "")
        if not text_input.strip():
            return jsonify({"error": "Brak treści"}), 400

        text_lower = text_input.lower()
        text_lemmas = lemmatize_text(text_input)
        text_lemmas_len = len(text_lemmas)

        counts_in_batch = {}

        # 🔁 Główna pętla liczenia
        for kw, meta in keywords_state.items():
            keyword_lemmas = meta.get("lemmas", [])
            root_prefix = meta.get("root_prefix", "")
            kw_len = meta.get("lemma_len", 0)

            lemma_count = 0
            for i in range(text_lemmas_len - kw_len + 1):
                if text_lemmas[i:i + kw_len] == keyword_lemmas:
                    lemma_count += 1

            regex_pattern = rf"\b{re.escape(root_prefix)}\w*\b"
            root_matches = re.findall(regex_pattern, text_lower)
            root_count = len(root_matches)

            contextual_hits = extract_context_matches(text_input, root_prefix, ["sąd", "wniosek", "osoba", "postępowanie"])
            semantic_coverage = round(root_count / (lemma_count + 1), 2)

            meta["actual"] = meta.get("actual", 0) + max(lemma_count, root_count)
            meta["actual_tokens"] = meta.get("actual_tokens", 0) + root_count
            meta["contextual_hits"] = meta.get("contextual_hits", 0) + contextual_hits
            meta["semantic_coverage"] = semantic_coverage
            counts_in_batch[kw] = max(lemma_count, root_count)

            if meta["actual"] > meta["target_max"] + 3:
                meta["locked"] = True
                meta["status"] = "LOCKED"
            elif meta["actual"] > meta["target_max"]:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"

        batch_entry = {
            "created_at": datetime.utcnow().isoformat(),
            "length": len(text_input),
            "counts": counts_in_batch,
            "text": text_input[:5000]
        }
        batches.append(batch_entry)

        doc_ref.update({
            "batches": firestore.ArrayUnion([batch_entry]),
            "keywords_state": keywords_state,
            "updated_at": datetime.utcnow().isoformat()
        })

        locked_terms, keywords_report = [], []
        for kw, meta in keywords_state.items():
            status = meta.get("status", "OK")
            priority_instruction = "USE_AS_NEEDED"
            if status == "LOCKED":
                priority_instruction = "DO_NOT_USE"
                locked_terms.append(kw)
            elif status == "UNDER":
                if meta.get("actual", 0) == 0:
                    status = "NOT_USED"
                    priority_instruction = "CRITICAL_PRIORITY_USE"
                else:
                    priority_instruction = "PRIORITY_USE"

            keywords_report.append({
                "keyword": kw,
                "actual_uses": meta.get("actual", 0),
                "actual_tokens": meta.get("actual_tokens", 0),
                "contextual_hits": meta.get("contextual_hits", 0),
                "semantic_coverage": meta.get("semantic_coverage", 0.0),
                "target_range": f"{meta.get('target_min', 0)}-{meta.get('target_max', 0)}",
                "status": status,
                "priority_instruction": priority_instruction
            })

        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "counts_in_batch": counts_in_batch,
            "keywords_report": keywords_report,
            "locked_terms": locked_terms,
            "updated_keywords": len(keywords_state)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project/add_batch: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧹 /api/project/<id> — usuwa projekt
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
        keywords_state = project_data.get("keywords_state", {})
        batches = project_data.get("batches", [])

        summary = {
            "topic": project_data.get("topic", "nieznany temat"),
            "total_batches": len(batches),
            "total_length": sum(b.get("length", 0) for b in batches),
            "locked_terms": [k for k, v in keywords_state.items() if v.get("locked")],
            "timestamp": datetime.utcnow().isoformat(),
        }

        db.collection("seo_projects_archive").document(project_id).set(summary)
        doc_ref.delete()

        return jsonify({"status": f"✅ Projekt {project_id} został usunięty.", "summary": summary}), 200

    except Exception as e:
        return jsonify({"error": f"Błąd /api/project DELETE: {str(e)}"}), 500


# ---------------------------------------------------------------
# 🧩 Lokalny raport semantyczny (bez zapisu do Firestore)
# ---------------------------------------------------------------
def generate_semantic_report(batch_number: int, keywords_report: list, headers: list = None, structure_info: dict = None) -> str:
    """Generuje czytelny raport semantyczny i meta-prompt po zakończeniu batcha."""
    try:
        over_terms = [k["keyword"] for k in keywords_report if k["status"] in ["OVER", "LOCKED"]]
        under_terms = [k["keyword"] for k in keywords_report if k["status"] in ["UNDER", "NOT_USED"]]
        semantic_vals = [k.get("semantic_coverage", 0) for k in keywords_report if isinstance(k.get("semantic_coverage", 0), (int, float))]
        context_vals = [k.get("contextual_hits", 0) for k in keywords_report if isinstance(k.get("contextual_hits", 0), (int, float))]

        avg_sem = round(mean(semantic_vals), 2) if semantic_vals else 0
        avg_ctx = round(mean(context_vals), 2) if context_vals else 0

        section_descriptions = ""
        if headers:
            for h in headers:
                section_descriptions += f"H2: „{h}”\nOpis: sekcja tematyczna dotycząca zagadnienia {h.lower()}.\n"

        structure_text = ""
        if structure_info:
            structure_text = f"Struktura: {structure_info.get('akapitów', '—')} akapitów, " \
                             f"średnio {structure_info.get('średnio_zdań', '—')} zdań w akapicie (>5)\n"

        over_block = (
            "Frazy zredukowane do niezbędnego minimum\n"
            "(utrzymane wyłącznie w kontekstach merytorycznych, gdy bez nich zdanie traci sens):\n"
        )
        over_block += "→ " + ", ".join(over_terms) + "\n" if over_terms else "→ Brak fraz przekroczonych\n"

        under_block = "Frazy priorytetowe (niedostateczne użycie lub brak kontekstu):\n"
        under_block += "→ " + ", ".join(under_terms) + "\n" if under_terms else "→ Wszystkie frazy w normie\n"

        if avg_sem > 1.5:
            quality_note = "Uwaga: możliwe przeoptymalizowanie fraz — w kolejnym batchu redukuj nadmiar odmian."
        elif avg_sem < 0.8:
            quality_note = "Zwiększ różnorodność językową — dodaj warianty fleksyjne i równoważniki znaczeniowe."
        else:
            quality_note = "Balans językowy i kontekstowy utrzymany."

        report = (
            "──────────────────────────────\n"
            f"Batch nr {batch_number}\n\n"
            f"{section_descriptions}"
            f"{structure_text}\n"
            f"{over_block}\n"
            f"{under_block}\n"
            "──────────────────────────────\n"
            f"Wskaźniki jakości semantycznej:\n"
            f"→ Średni semantic_coverage: {avg_sem}\n"
            f"→ Średni contextual_hits: {avg_ctx}%\n"
            f"→ Ocena: {quality_note}\n"
            "──────────────────────────────\n\n"
            "**Meta-prompt dla kolejnego batcha:**\n"
            "Na podstawie powyższego raportu wygeneruj kolejny batch zgodnie z wytycznymi:\n"
            " - zachowaj styl i rytm narracyjny,\n"
            " - ogranicz użycie fraz z listy 'zredukowane' do kontekstów merytorycznych,\n"
            " - wpleć naturalnie frazy z listy 'priorytetowe',\n"
            " - popraw contextual_hits i semantic_coverage,\n"
            " - utrzymaj długość akapitów i ton narracyjny z poprzednich sekcji.\n"
            "──────────────────────────────"
        )

        return report

    except Exception as e:
        return f"[BŁĄD RAPORTU] Nie udało się wygenerować raportu semantycznego: {e}"


# ---------------------------------------------------------------
# 🔧 Rejestracja blueprinta
# ---------------------------------------------------------------
def register_project_routes(app, _db=None):
    global db
    db = _db
    app.register_blueprint(project_bp)
    if db:
        print("✅ [DEBUG] project_routes: Firestore aktywne.")
    else:
        print("⚠️ [DEBUG] project_routes: instancja db == None.")
