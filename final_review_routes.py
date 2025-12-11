# ================================================================
# 🧠 final_review_routes.py — Expert Review & Interactive Correction (v19.6)
# ================================================================
"""
Tryb interaktywny:
1️⃣ Po zakończeniu artykułu system wysyła tekst do Gemini i tworzy raport.
2️⃣ Wynik raportu zwracany jest użytkownikowi (bez korekty).
3️⃣ Backend pyta: „Czy chcesz wprowadzić poprawki?"
4️⃣ Jeśli użytkownik potwierdzi — drugi endpoint generuje poprawioną wersję.

Ustalenia (surgical patch, bez refaktorów pobocznych):
- Nie dublujemy generatorów final review: jeżeli final_review już istnieje w Firestore,
  endpoint /final_review zwraca istniejący raport (chyba że wymusisz regenerację).
- Model do review i korekt jest sterowany env: FINAL_REVIEW_MODEL (fallback: gemini-2.0-flash-exp).
"""

import os
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import google.generativeai as genai

# ------------------------------------------------------------
# 🔧 Konfiguracja Gemini
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[REVIEW] ✅ Gemini API aktywny (Final Review Mode)")
else:
    print("[REVIEW] ⚠️ Brak GEMINI_API_KEY — Final Review nieaktywny")

FINAL_REVIEW_MODEL = os.getenv("FINAL_REVIEW_MODEL", "gemini-2.0-flash-exp")

# ------------------------------------------------------------
# 🔧 Inicjalizacja Blueprint
# ------------------------------------------------------------
final_review_routes = Blueprint("final_review_routes", __name__)

# ------------------------------------------------------------
# 🧩 Utils
# ------------------------------------------------------------
def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "tak", "t")

def _join_full_article(batches: list) -> str:
    return "\n\n".join([b.get("text", "") for b in (batches or [])]).strip()

# ------------------------------------------------------------
# 🧠 1) Końcowy raport (bez korekty)
# ------------------------------------------------------------
@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    """Tworzy końcowy raport Gemini i pyta, czy zastosować poprawki."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key not configured"}), 500

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict() or {}
    batches = data.get("batches", [])
    if not batches:
        return jsonify({"error": "No article content found"}), 400

    full_article = _join_full_article(batches)
    if not full_article:
        return jsonify({"error": "Empty article text"}), 400

    # ✅ Guard przed dublowaniem: jeżeli raport już istnieje, zwróć go,
    # chyba że wymuszono regenerację.
    force_regenerate = _truthy(request.args.get("force"))
    body = request.get_json(silent=True) or {}
    if isinstance(body, dict) and body.get("force") is True:
        force_regenerate = True

    existing = data.get("final_review")
    if existing and not force_regenerate:
        if isinstance(existing, dict):
            return jsonify({
                "status": existing.get("status", "REVIEW_READY"),
                "project_id": project_id,
                "review": existing.get("review_text"),
                "model": existing.get("model"),
                "article_length": existing.get("article_length"),
                "note": "Zwrócono istniejący final_review z Firestore. Aby przeliczyć, użyj ?force=true lub {force:true}."
            }), 200
        return jsonify({
            "status": "REVIEW_READY",
            "project_id": project_id,
            "review": existing,
            "note": "Zwrócono istniejący final_review z Firestore. Aby przeliczyć, użyj ?force=true lub {force:true}."
        }), 200

    try:
        print(f"[REVIEW] 🔍 Analiza CAŁEGO artykułu projektu {project_id} ({len(full_article)} znaków)...")
        model = genai.GenerativeModel(FINAL_REVIEW_MODEL)

        review_prompt = (
            "Podaj w punktach szczegółową ocenę przesłanego artykułu pod kątem:\n"
            "1. merytorycznym (zgodność faktów, aktualność, błędy logiczne),\n"
            "2. redakcyjnym (struktura, powtórzenia, styl),\n"
            "3. językowym (poprawność gramatyczna, płynność),\n"
            "a także zaproponuj konkretne poprawki dla każdego problemu.\n\n"
            f"---\n{full_article}"
        )

        review_response = model.generate_content(review_prompt)
        review_text = (review_response.text or "").strip()
        if not review_text:
            return jsonify({"error": "Empty review from Gemini"}), 502

    except Exception as e:
        print(f"[REVIEW] ❌ Błąd podczas generowania raportu: {e}")
        return jsonify({"error": str(e)}), 500

    # 🔹 Zapisz sam raport (bez korekty)
    try:
        doc_ref.update({
            "final_review": {
                "review_text": review_text,
                "created_at": firestore.SERVER_TIMESTAMP,
                "model": FINAL_REVIEW_MODEL,
                "status": "REVIEW_READY",
                "article_length": len(full_article)
            }
        })
        print(f"[REVIEW] ✅ Raport zapisany w Firestore (bez korekty) → {project_id}")
    except Exception as e:
        print(f"[REVIEW] ⚠️ Błąd zapisu raportu: {e}")

    return jsonify({
        "status": "REVIEW_READY",
        "project_id": project_id,
        "review": review_text,
        "model": FINAL_REVIEW_MODEL,
        "article_length": len(full_article),
        "next_action": f"Czy chcesz wprowadzić poprawki automatycznie? (POST /api/project/{project_id}/apply_final_corrections)"
    }), 200

# ------------------------------------------------------------
# ✏️ 2) Zastosuj poprawki (po akceptacji użytkownika)
# ------------------------------------------------------------
@final_review_routes.post("/api/project/<project_id>/apply_final_corrections")
def apply_final_corrections(project_id):
    """Tworzy poprawioną wersję artykułu na podstawie wcześniejszego raportu."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key not configured"}), 500

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict() or {}
    final_review = data.get("final_review", {})
    review_text = final_review.get("review_text") if isinstance(final_review, dict) else None
    batches = data.get("batches", [])
    full_article = _join_full_article(batches)

    if not review_text:
        return jsonify({"error": "No final review found. Generate review first."}), 400
    if not full_article:
        return jsonify({"error": "Empty article text"}), 400

    try:
        print(f"[REVIEW] ✏️ Generowanie poprawionej wersji artykułu ({project_id})...")
        model = genai.GenerativeModel(FINAL_REVIEW_MODEL)

        correction_prompt = (
            "Na podstawie poniższego raportu popraw artykuł.\n"
            "Wprowadź poprawki merytoryczne, redakcyjne i językowe.\n"
            "Zachowaj strukturę oraz sens, ale usuń błędy i popraw płynność.\n\n"
            "RAPORT:\n---\n" + review_text + "\n\n"
            "ARTYKUŁ:\n---\n" + full_article
        )

        correction_response = model.generate_content(correction_prompt)
        corrected_text = (correction_response.text or "").strip()
        if not corrected_text:
            return jsonify({"error": "Empty correction from Gemini"}), 502

    except Exception as e:
        print(f"[REVIEW] ❌ Błąd generowania korekty: {e}")
        return jsonify({"error": str(e)}), 500

    # 🔹 Zapisz poprawiony tekst
    try:
        doc_ref.update({
            "final_review.corrected_text": corrected_text,
            "final_review.status": "CORRECTED",
            "final_review.updated_at": firestore.SERVER_TIMESTAMP,
            "final_review.model": FINAL_REVIEW_MODEL
        })
        print(f"[REVIEW] ✅ Poprawiona wersja zapisana w Firestore → {project_id}")
    except Exception as e:
        print(f"[REVIEW] ⚠️ Błąd zapisu korekty: {e}")

    return jsonify({
        "status": "CORRECTION_APPLIED",
        "project_id": project_id,
        "corrected_text": corrected_text
    }), 200
