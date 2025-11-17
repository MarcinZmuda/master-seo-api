# ================================================================
# firestore_tracker_routes.py — Warstwa Batch + Keyword Tracker
# v7.3.0-firestore-continuous-lemma (stabilna wersja produkcyjna)
# ================================================================

from flask import Blueprint, request, jsonify
from datetime import datetime
import spacy

# ---------------------------------------------------------------
# 🔧 Inicjalizacja blueprinta i Firestore
# ---------------------------------------------------------------
tracker_bp = Blueprint("firestore_tracker_routes", __name__)
db = None  # ustawiane w register_tracker_routes

# Ładujemy model spaCy tylko raz (lemmaMode=ON)
try:
    nlp = spacy.load("pl_core_news_sm")
    print("✅ Model spaCy (pl_core_news_sm) załadowany poprawnie (lemmaMode=ON).")
except Exception as e:
    print(f"❌ Błąd ładowania modelu spaCy: {e}")
    nlp = None


# ---------------------------------------------------------------
# 🧠 Pomocnicze: lematyzacja tekstu do listy tokenów
# ---------------------------------------------------------------
def lemmatize_text(text: str) -> list[str]:
    """
    Zwraca listę zlematyzowanych, znormalizowanych tokenów.
    - tylko litery/cyfry,
    - lower-case,
    - pomija spacje, znaki interpunkcyjne itd.
    """
    if not nlp:
        # awaryjnie: fallback bez spaCy (czysty split po whitespace)
        return [t.lower() for t in text.split() if t.strip()]

    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower()
        if not lemma:
            continue
        lemmas.append(lemma)
    return lemmas


def lemmatize_phrase(phrase: str) -> list[str]:
    """
    Lematylizuje frazę (np. 'adwokat rozwodowy w warszawie')
    do listy lematów ['adwokat', 'rozwodowy', 'w', 'warszawa'].
    """
    return lemmatize_text(phrase)


# ---------------------------------------------------------------
# 🧮 Liczenie lematów w batchu
# ---------------------------------------------------------------
def count_phrase_lemmas_in_text(batch_lemmas: list[str], phrase_lemmas: list[str]) -> int:
    """
    Liczy ile razy N-gram (fraza w lematycznej postaci) pojawia się
    w zlematyzowanym tekście batcha.

    Przykład:
    batch_lemmas = ['adwokat', 'rozwodowy', 'w', 'warszawa', 'adwokat', 'rozwód']
    phrase_lemmas = ['adwokat', 'rozwodowy']
    -> wynik = 1
    """
    if not batch_lemmas or not phrase_lemmas:
        return 0

    n = len(batch_lemmas)
    m = len(phrase_lemmas)

    if m > n:
        return 0

    count = 0
    for i in range(n - m + 1):
        if batch_lemmas[i : i + m] == phrase_lemmas:
            count += 1
    return count


# ---------------------------------------------------------------
# 🧩 Aktualizacja keywords_state na podstawie nowego batcha
# ---------------------------------------------------------------
def update_keywords_state_with_batch(project_doc, batch_text: str) -> dict:
    """
    Aktualizuje keywords_state projektu na podstawie nowego batcha.

    Zakładany format w Firestore:
    seo_projects/{project_id}:
    {
        "topic": "...",
        "keywords_state": [
            {"term": "adwokat", "min": 12, "max": 18, "actual": 5},
            {"term": "warszawa", "min": 5, "max": 16, "actual": 3},
            ...
        ],
        "counting_mode": "lemma",
        "continuous_counting": True,
        "prefer_local_tracker": False,
        "total_batches": 1,
        ...
    }
    """
    data = project_doc.to_dict() or {}
    keywords_state = data.get("keywords_state", [])
    counting_mode = data.get("counting_mode", "lemma")
    continuous_counting = bool(data.get("continuous_counting", True))
    prefer_local_tracker = bool(data.get("prefer_local_tracker", False))

    batch_lemmas = lemmatize_text(batch_text)

    under_terms = []
    over_terms = []
    locked_terms = []
    ok_terms = []

    # aktualizujemy po kolei wszystkie słowa kluczowe
    for kw in keywords_state:
        term = kw.get("term", "")
        target_min = int(kw.get("min", 0))
        target_max = int(kw.get("max", 0))
        actual = int(kw.get("actual", 0))

        phrase_lemmas = lemmatize_phrase(term)
        delta = count_phrase_lemmas_in_text(batch_lemmas, phrase_lemmas)

        # continuous lemma counting → dodajemy delta do istniejącego stanu
        new_actual = actual + delta
        kw["actual"] = new_actual

        # ustalenie statusu frazy zgodnie z Twoją logiką:
        # UNDER: new_actual < min
        # OK:    min <= new_actual <= max
        # OVER:  new_actual > max
        # LOCKED: new_actual > max + 10 (twardy stuffing)
        status = "UNDER"
        if new_actual < target_min:
            status = "UNDER"
        elif target_min <= new_actual <= target_max:
            status = "OK"
        elif target_max < new_actual <= (target_max + 10):
            status = "OVER"
        else:
            # mocno przeoptymalizowane
            status = "LOCKED"

        kw["status"] = status

        if status == "UNDER":
            under_terms.append(kw)
        elif status == "OK":
            ok_terms.append(kw)
        elif status == "OVER":
            over_terms.append(kw)
        elif status == "LOCKED":
            locked_terms.append(kw)

    over_terms_count = len(over_terms)
    locked_terms_count = len(locked_terms)
    under_terms_count = len(under_terms)
    ok_terms_count = len(ok_terms)

    # progi zgodne z dokumentacją:
    # Forced Regeneration → OVER ≥ 10
    # Emergency Exit → LOCKED ≥ 4
    regeneration_triggered = over_terms_count >= 10
    emergency_exit_triggered = locked_terms_count >= 4

    meta_prompt_summary = (
        f"UNDER={under_terms_count}, OVER={over_terms_count}, "
        f"LOCKED={locked_terms_count}, OK={ok_terms_count} – global lemma"
    )

    # aktualizacja projektu w Firestore
    project_ref = project_doc.reference
    total_batches = int(data.get("total_batches", 0)) + 1

    project_ref.update(
        {
            "keywords_state": keywords_state,
            "total_batches": total_batches,
            "last_batch_at": datetime.utcnow().isoformat(),
            "last_meta_prompt_summary": meta_prompt_summary,
            "last_under_terms_count": under_terms_count,
            "last_over_terms_count": over_terms_count,
            "last_locked_terms_count": locked_terms_count,
            "last_ok_terms_count": ok_terms_count,
        }
    )

    # budujemy raport w formacie zgodnym z OpenAPI
    keywords_report = []
    for kw in keywords_state:
        keywords_report.append(
            {
                "keyword": kw.get("term", ""),
                "actual_uses": int(kw.get("actual", 0)),
                "target_range": f"{int(kw.get('min', 0))}–{int(kw.get('max', 0))}",
                "status": kw.get("status", "UNDER"),
                "priority_instruction": _priority_instruction_from_status(
                    kw.get("status", "UNDER")
                ),
            }
        )

    return {
        "status": "OK",
        "counting_mode": counting_mode,
        "continuous_counting": continuous_counting,
        "prefer_local_tracker": prefer_local_tracker,
        "regeneration_triggered": regeneration_triggered,
        "emergency_exit_triggered": emergency_exit_triggered,
        "keywords_report": keywords_report,
        "meta_prompt_summary": meta_prompt_summary,
    }


def _priority_instruction_from_status(status: str) -> str:
    """
    Daje krótką instrukcję dla GPT na podstawie statusu frazy.
    """
    status = (status or "").upper()
    if status == "UNDER":
        return "Wzmocnij tę frazę w kolejnych batchach (priorytet: wysoki)."
    if status == "OVER":
        return "Ogranicz użycie tej frazy w kolejnych batchach (priorytet: wysoki)."
    if status == "LOCKED":
        return "Nie używaj tej frazy więcej w tym projekcie (LOCKED)."
    return "Fraza w normie; możesz używać naturalnie."


# ================================================================
# ✍ /api/project/<project_id>/add_batch — analiza batcha + counting
# ================================================================
@tracker_bp.route("/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    """
    Endpoint zgodny z OpenAPI:

    POST /api/project/{project_id}/add_batch
    {
        "text": "Treść batcha do analizy SEO.",
        "counting_mode": "lemma",           # opcjonalnie
        "prefer_local_tracker": false       # ignorowane, zawsze False
    }

    Zwraca:
    {
        "status": "OK",
        "counting_mode": "lemma",
        "continuous_counting": true,
        "prefer_local_tracker": false,
        "regeneration_triggered": false,
        "emergency_exit_triggered": false,
        "keywords_report": [...],
        "meta_prompt_summary": "UNDER=..., OVER=..., LOCKED=..., OK=..."
    }
    """
    global db
    if db is None:
        return (
            jsonify(
                {
                    "status": "ERROR",
                    "message": "Brak połączenia z Firestore (db is None).",
                }
            ),
            500,
        )

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return (
            jsonify(
                {
                    "status": "ERROR",
                    "message": "Pole 'text' (treść batcha) jest wymagane i musi być niepuste.",
                }
            ),
            400,
        )

    # pobieramy projekt
    project_ref = db.collection("seo_projects").document(project_id)
    project_doc = project_ref.get()

    if not project_doc.exists:
        return (
            jsonify(
                {
                    "status": "ERROR",
                    "message": f"Projekt {project_id} nie istnieje w kolekcji 'seo_projects'.",
                }
            ),
            404,
        )

    try:
        result = update_keywords_state_with_batch(project_doc, text)
        print(
            f"[INFO] 🔢 Batch zapisany i przetworzony dla projektu {project_id}. "
            f"{result['meta_prompt_summary']}"
        )
        return jsonify(result), 200
    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania batcha dla projektu {project_id}: {e}")
        return (
            jsonify(
                {
                    "status": "ERROR",
                    "message": "Błąd podczas przetwarzania batcha.",
                    "details": str(e),
                }
            ),
            500,
        )


# ===============================================================
# 🔧 Rejestracja blueprinta
# ===============================================================
def register_tracker_routes(app, _db=None):
    """
    Rejestruje blueprint firestore_tracker_routes pod prefixem /api
    i ustawia globalne db.
    """
    global db
    db = _db
    app.register_blueprint(tracker_bp, url_prefix="/api")
    print(
        "✅ [INIT] firestore_tracker_routes zarejestrowany pod prefixem /api "
        "(v7.3.0-firestore-continuous-lemma)."
    )
