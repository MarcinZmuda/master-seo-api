# ================================================================
# 🧠 final_review_routes.py — Expert Review & Interactive Correction (v19.5)
# ================================================================
"""
Tryb interaktywny:
1️⃣ Po zakończeniu artykułu system wysyła tekst do Gemini i tworzy raport.
2️⃣ Wynik raportu zwracany jest użytkownikowi (bez korekty).
3️⃣ Backend pyta: „Czy chcesz wprowadzić poprawki?"
4️⃣ Jeśli użytkownik potwierdzi — drugi endpoint generuje poprawioną wersję.
"""

import os
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import google.generativeai as genai

# ------------------------------------------------------------
# 🔧 Konfiguracja Gemini
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("[REVIEW] ⚠️ Brak klucza GEMINI_API_KEY — moduł nieaktywny")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[REVIEW] ✅ Gemini 1.5 Pro aktywny dla final review")

# ------------------------------------------------------------
# 🔧 Inicjalizacja Blueprint
# ------------------------------------------------------------
final_review_routes = Blueprint("final_review_routes", __name__)

# ------------------------------------------------------------
# 🧩 Główna funkcja: analiza końcowa (bez korekty)
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

    data = doc.to_dict()
    batches = data.get("batches", [])
    if not batches:
        return jsonify({"error": "No article content found"}), 400

    # 🔹 Złącz artykuł
    full_article = "\n\n".join([b.get("text", "") for b in batches]).strip()
    if not full_article:
        return jsonify({"error": "Empty article text"}), 400

    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # ✅ POPRAWKA: Usunięto [:15000] - teraz Gemini analizuje CAŁY artykuł
    review_prompt = (
        "Podaj w punktach szczegółową ocenę przesłanego artykułu pod kątem:\n"
        "1. merytorycznym (zgodność faktów, aktualność, błędy logiczne),\n"
        "2. redakcyjnym (struktura, powtórzenia, styl),\n"
        "3. językowym (poprawność gramatyczna, płynność),\n"
        "a także zaproponuj konkretne poprawki dla każdego problemu.\n\n"
        "Artykuł:\n---\n" + full_article  # ⭐ BEZ LIMITU!
    )

    try:
        print(f"[REVIEW] 🔍 Analiza CAŁEGO artykułu projektu {project_id} ({len(full_article)} znaków)...")
        review_response = model.generate_content(review_prompt)
        review_text = review_response.text.strip()
    except Exception as e:
        print(f"[REVIEW] ❌ Błąd podczas generowania raportu: {e}")
        return jsonify({"error": str(e)}), 500

    # 🔹 Zapisz sam raport (bez korekty)
    try:
        doc_ref.update({
            "final_review": {
                "review_text": review_text,
                "corrected_text": None,
                "created_at": firestore.SERVER_TIMESTAMP,
                "model": "gemini-1.5-pro",
                "status": "REVIEW_READY",
                "article_length": len(full_article)  # ⭐ DODANO tracking długości
            }
        })
        print(f"[REVIEW] ✅ Raport zapisany w Firestore (bez korekty) → {project_id}")
    except Exception as e:
        print(f"[REVIEW] ⚠️ Błąd zapisu raportu: {e}")

    return jsonify({
        "status": "REVIEW_READY",
        "project_id": project_id,
        "review": review_text,
        "article_length": len(full_article),  # ⭐ DODANO info o długości
        "next_action": "Czy chcesz wprowadzić poprawki automatycznie? (POST /api/project/<id>/apply_final_corrections)"
    }), 200


# ------------------------------------------------------------
# 🧩 Drugi etap: zastosowanie poprawek po potwierdzeniu
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

    data = doc.to_dict()
    final_review = data.get("final_review", {})
    batches = data.get("batches", [])
    if not final_review or not batches:
        return jsonify({"error": "Missing review or article text"}), 400

    review_text = final_review.get("review_text", "")
    full_article = "\n\n".join([b.get("text", "") for b in batches]).strip()
    if not review_text or not full_article:
        return jsonify({"error": "Invalid review or article"}), 400

    model = genai.GenerativeModel("gemini-1.5-pro")
    correction_prompt = (
        "Na podstawie poniższego raportu wprowadź poprawki do artykułu, "
        "zachowując sens, styl i strukturę (H2/H3).\n\n"
        f"RAPORT:\n{review_text}\n\n---\n\nARTYKUŁ DO POPRAWY:\n{full_article}"
    )

    try:
        print(f"[REVIEW] ✏️ Generowanie poprawionej wersji artykułu ({project_id})...")
        correction_response = model.generate_content(correction_prompt)
        corrected_text = correction_response.text.strip()
    except Exception as e:
        print(f"[REVIEW] ❌ Błąd podczas korekty: {e}")
        return jsonify({"error": str(e)}), 500

    try:
        doc_ref.update({
            "final_review.corrected_text": corrected_text,
            "final_review.status": "CORRECTED",
            "final_review.updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"[REVIEW] ✅ Poprawiona wersja zapisana w Firestore → {project_id}")
    except Exception as e:
        print(f"[REVIEW] ⚠️ Błąd zapisu korekty: {e}")

    return jsonify({
        "status": "CORRECTION_APPLIED",
        "project_id": project_id,
        "corrected_text": corrected_text
    }), 200
