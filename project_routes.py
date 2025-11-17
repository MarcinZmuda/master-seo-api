import logging
import re
from typing import Dict, Any, List, Tuple

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

project_bp = Blueprint("project_bp", __name__)


# ==========================
# 🔧 Pomocnicze funkcje
# ==========================


def parse_keyword_line(line: str) -> Tuple[str, int, int]:
    """
    Parsuje pojedynczą linię w formacie:
    - "fraza: 3-5x"
    - "fraza: 8–12x" (z dywizem / półpauzą)

    Zwraca:
    (fraza, min_count, max_count)

    Podnosi ValueError, jeśli format jest nieprawidłowy.
    """
    original_line = line
    line = line.strip()
    if not line:
        raise ValueError("Pusta linia")

    # Rozdzielenie na "fraza" i "zakres"
    if ":" not in line:
        raise ValueError(f"Brak dwukropka w linii: {original_line!r}")

    phrase_part, range_part = line.split(":", 1)
    phrase = phrase_part.strip()

    # Usuwamy końcówkę "x" oraz spacje
    range_part = range_part.strip().lower().replace("x", "").strip()

    # Obsługa zakresów "3-5" oraz "3–5" (dywiz/półpauza)
    range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", range_part)
    if not range_match:
        raise ValueError(f"Nieprawidłowy zakres w linii: {original_line!r}")

    min_count = int(range_match.group(1))
    max_count = int(range_match.group(2))

    if min_count > max_count:
        raise ValueError(
            f"Nieprawidłowy zakres (min > max) w linii: {original_line!r}"
        )

    return phrase, min_count, max_count


def parse_brief_to_keywords(brief: str) -> List[Dict[str, Any]]:
    """
    Parsuje cały brief tekstowy (BASIC TEXT TERMS + EXTENDED TEXT TERMS)
    i zwraca listę obiektów keywordów.

    Zakładamy strukturę:

    BASIC TEXT TERMS:
    fraza: 3-5x
    fraza2: 8–12x

    EXTENDED TEXT TERMS:
    fraza3: 4-10x
    ...

    EXTENDED mają domyślnie obniżone widełki o 50% (zaokrąglenie w górę).
    """
    lines = brief.splitlines()

    keywords: List[Dict[str, Any]] = []
    current_section = None  # "BASIC" albo "EXTENDED"

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        # Wykrywanie sekcji
        if line.upper().startswith("BASIC TEXT TERMS"):
            current_section = "BASIC"
            continue
        if line.upper().startswith("EXTENDED TEXT TERMS"):
            current_section = "EXTENDED"
            continue

        # Pomijamy linie nagłówkowe lub śmieci
        if current_section is None:
            continue

        # Próba sparsowania frazy
        try:
            phrase, min_count, max_count = parse_keyword_line(line)
        except ValueError as e:
            logger.warning(f"⏭ Pominięto linię w briefie: {line!r} ({e})")
            continue

        # EXTENDED => obniżamy widełki o 50%
        if current_section == "EXTENDED":
            # Zaokrąglamy w górę (co najmniej 1)
            min_count = max(1, (min_count + 1) // 2)
            max_count = max(1, (max_count + 1) // 2)

        keywords.append(
            {
                "phrase": phrase,
                "section": current_section,
                "target_min": min_count,
                "target_max": max_count,
                # Liczniki startowe (Firestore i tak będzie nadpisywał)
                "actual_count": 0,
                "status": "UNDER",  # domyślnie
            }
        )

    logger.info(f"🧠 parse_brief_to_keywords → {len(keywords)} fraz sparsowanych.")
    return keywords


# ==========================
# 🌐 Endpointy projektowe
# ==========================


def register_project_routes(app, db):
    """
    Rejestruje endpointy projektowe na aplikacji Flask.

    Uwaga: tutaj obsługujemy tylko:
    - tworzenie projektu (`/api/project/create`)
    - pobieranie / usuwanie projektu (`/api/project/<id>`)
    - prosty endpoint diagnostyczny

    Endpoint `/api/project/<id>/add_batch` jest obsługiwany wyłącznie przez:
    - `firestore_tracker_routes.py` (tracker_bp)

    Dzięki temu:
    - nie ma pętli HTTP do samego siebie,
    - cała logika zliczania batchy (lemma) jest skupiona w jednym miejscu.
    """

    @project_bp.route("/project/create", methods=["POST"])
    def create_project():
        """
        Tworzy nowy projekt SEO w Firestore na podstawie briefu tekstowego.

        Oczekiwany JSON:
        {
          "topic": "adwokat rozwód Warszawa",
          "brief_text": "BASIC TEXT TERMS:\\nfraza: 3-5x\\n...",
          "meta": {...}  // dowolne dane pomocnicze
        }
        """
        data = request.get_json(force=True, silent=True) or {}
        topic = data.get("topic") or data.get("main_keyword") or "unknown_topic"
        brief_text = data.get("brief_text") or data.get("brief") or ""
        meta = data.get("meta") or {}

        if not brief_text.strip():
            return (
                jsonify(
                    {
                        "error": "Brak brief_text — nie mogę utworzyć projektu bez listy fraz.",
                        "ok": False,
                    }
                ),
                400,
            )

        logger.info(f"[DEBUG] Tworzenie projektu Firestore: {topic}")

        # Parsowanie briefu
        keywords_list = parse_brief_to_keywords(brief_text)

        # Dokument w Firestore
        project_doc = {
            "topic": topic,
            "meta": meta,
            "keywords_state": keywords_list,
            "counting_mode": "lemma",
            "continuous_counting": True,
            "created_at": db.SERVER_TIMESTAMP if hasattr(db, "SERVER_TIMESTAMP") else None,
            "updated_at": db.SERVER_TIMESTAMP if hasattr(db, "SERVER_TIMESTAMP") else None,
            "status": "ACTIVE",
        }

        # Zapis do Firestore
        projects_collection = db.collection("seo_projects")
        project_ref = projects_collection.document()
        project_ref.set(project_doc)

        project_id = project_ref.id
        logger.info(f"✅ Projekt {project_id} utworzony ({len(keywords_list)} fraz).")

        return jsonify({"ok": True, "project_id": project_id, "keywords": keywords_list})

    @project_bp.route("/project/<project_id>", methods=["GET"])
    def get_project(project_id):
        """
        Zwraca pełny dokument projektu z Firestore.
        """
        project_ref = db.collection("seo_projects").document(project_id)
        doc = project_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje.", "ok": False}), 404

        data = doc.to_dict()
        data["id"] = project_id
        return jsonify({"ok": True, "project": data})

    @project_bp.route("/project/<project_id>", methods=["DELETE"])
    def delete_project(project_id):
        """
        Usuwa projekt z Firestore (lub oznacza jako zamknięty, jeśli chcesz).
        """
        project_ref = db.collection("seo_projects").document(project_id)
        doc = project_ref.get()
        if not doc.exists:
            return jsonify({"error": "Projekt nie istnieje.", "ok": False}), 404

        project_ref.delete()
        logger.info(f"🧹 Projekt {project_id} usunięty z Firestore.")

        return jsonify({"ok": True, "deleted_project_id": project_id})

    @project_bp.route("/project/debug/routes", methods=["GET"])
    def debug_routes():
        """
        Prosty endpoint diagnostyczny — pokazuje, że project_routes działają.
        """
        return jsonify(
            {
                "ok": True,
                "message": "project_routes online",
                "endpoints": [
                    "POST /api/project/create",
                    "GET /api/project/<project_id>",
                    "DELETE /api/project/<project_id>",
                    #   /api/project/<id>/add_batch  →  OBSŁUGUJE firestore_tracker_routes
                ],
            }
        )

    # Rejestrujemy blueprint na /api
    app.register_blueprint(project_bp, url_prefix="/api")
    logger.info("✅ [INIT] project_routes zarejestrowany pod prefixem /api")
