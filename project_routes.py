# ================================================================
# project_routes.py — Warstwa Project Management (v7.2.1 - Final Logic)
# Zintegrowane: Hierarchiczne Liczenie + Zwrot Danych Krytycznych
# ================================================================

import json
import re
from flask import Blueprint, request, jsonify
from datetime import datetime
import requests
from firebase_admin import firestore
import spacy
from statistics import mean
from collections import Counter # Nowy import

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
# 🔗 Hierarchiczne Liczenie (Zaczerpnięte z hierarchical_keyword_counter.py)
# ---------------------------------------------------------------
def apply_hierarchical_counting(raw_counts):
    """
    Oblicza hierarchiczne zliczanie fraz kluczowych.
    Jeśli krótsze słowo występuje w dłuższym, licznik krótszej frazy jest zwiększany.
    """
    if not isinstance(raw_counts, dict):
        return raw_counts

    keywords = sorted(raw_counts.keys(), key=len, reverse=True)
    hierarchical_counts = raw_counts.copy()

    for i, long_kw in enumerate(keywords):
        for short_kw in keywords[i + 1:]:
            # Sprawdzenie, czy krótsza fraza jest pełnym słowem w dłuższej
            if short_kw in long_kw and re.search(r'\b' + re.escape(short_kw) + r'\b', long_kw):
                hierarchical_counts[short_kw] += raw_counts.get(long_kw, 0)
    
    return hierarchical_counts


# ---------------------------------------------------------------
# 🔧 Pomocnicze funkcje Firestore
# ---------------------------------------------------------------
def call_s1_analysis(topic):
    """Bezpieczne wywołanie analizy S1 z timeoutem."""
    try:
        # W środowisku Render/Gunicorn to powinno działać
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
# 🧮 /api/project/<id>/add_batch (ZMIENIONO)
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
        
        # 1. Obliczanie surowych wystąpień (RAW COUNTS)
        raw_counts_in_batch = {}
        
        for kw, meta in keywords_state.items():
            lemmas = meta.get("lemmas", [])
            root_prefix = meta.get("root_prefix", "")
            kw_len = meta.get("lemma_len", 0)
            
            # Liczenie dokładnych lematów
            lemma_count = sum(
                1 for i in range(len(text_lemmas) - kw_len + 1)
                if text_lemmas[i:i + kw_len] == lemmas
            )
            # Liczenie rdzenia semantycznego
            root_count = len(re.findall(rf"\b{re.escape(root_prefix)}\w*\b", text_lower))
            
            # Zliczanie semantyczne (max z obu)
            raw_counts_in_batch[kw] = max(lemma_count, root_count)

        # 2. Zastosowanie liczenia hierarchicznego (HIERARCHICAL COUNTS)
        hierarchical_counts = apply_hierarchical_counting(raw_counts_in_batch)
        
        locked_count_global = 0
        critical_over_diff = 0
        
        # 3. Aktualizacja globalnego stanu
        for kw, meta in keywords_state.items():
            count_in_batch = hierarchical_counts.get(kw, 0)
            
            # Kontekstualne hity (bez zmian)
            root_prefix = meta.get("root_prefix", "")
            contextual_hits = extract_context_matches(text_input, root_prefix, ["sąd", "wniosek", "osoba", "postępowanie"])
            
            # Aktualizacja globalnego stanu
            meta["actual"] += count_in_batch
            meta["contextual_hits"] += contextual_hits
            
            # Weryfikacja statusu (względem target_max i krytycznych polityk)
            over_diff = meta["actual"] - meta["target_max"]
            
            if over_diff > 3: # Target_max + 3 (krytyczna blokada)
                meta["locked"], meta["status"] = True, "LOCKED"
            elif over_diff > 0:
                meta["status"] = "OVER"
            elif meta["actual"] < meta["target_min"]:
                meta["status"] = "UNDER"
            else:
                meta["status"] = "OK"

            # Zliczanie globalnego LOCKED i CRITICAL OVER
            if meta.get("locked"):
                locked_count_global += 1
            if over_diff >= 10:
                critical_over_diff = max(critical_over_diff, over_diff)

        # 4. Budowanie raportu
        keywords_report = [
            {
                "keyword": k,
                "type": v.get("type", "BASIC"),
                "actual_uses": v.get("actual", 0),
                "contextual_hits": v.get("contextual_hits", 0),
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

        batch_number = len(project_data.get("batches", [])) + 1
        meta_prompt_summary = build_meta_prompt_summary(batch_number, keywords_report)
        
        # 5. Aktualizacja Firestore
        doc_ref.update({
            "keywords_state": keywords_state,
            "batches": firestore.ArrayUnion([{
                "created_at": datetime.utcnow().isoformat(),
                "text": text_input[:5000],
                "counts": hierarchical_counts # Zapisujemy hierarchiczne zliczenie
            }]),
            "updated_at": datetime.utcnow().isoformat()
        })

        print(f"[BATCH {project_id}] ✅ Zapisano hierarchiczne liczenie.")
        
        # 6. Zwrot Danych Krytycznych dla GPT
        return jsonify({
            "status": "OK",
            "batch_length": len(text_input),
            "locked_count_global": locked_count_global,
            "critical_over_difference": critical_over_diff,
            "keywords_report": keywords_report,
            "meta_prompt_summary": meta_prompt_summary
        }), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/add_batch: {e}")
        return jsonify({"error": f"Błąd /api/project/add_batch: {str(e)}"}), 500

# ... (Pozostałe funkcje / DELETE / Meta Prompt Summary bez zmian)
# ... (Rejestracja blueprinta na końcu)
