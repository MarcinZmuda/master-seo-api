# ================================================================
# 🧠 final_review_routes.py — Expert Review & Auto Correction (v18.5)
# ================================================================
"""
Moduł końcowego audytu artykułu po zakończeniu wszystkich batchy.
1. Łączy tekst z Firestore.
2. Wysyła zapytanie do Gemini (merytoryka, redakcja, język).
3. Zapisuje raport oraz poprawioną wersję w Firestore.
4. Można wywołać ręcznie lub automatycznie po eksporcie.
"""

import os
from flask import Blueprint, jsonify
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
# 🧩 Główna funkcja: analiza i korekta
# ------------------------------------------------------------
@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    """
    Analizuje gotowy artykuł i zwraca raport + opcjonalnie poprawioną wersję.
    """
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

    # 🔹 Złącz pełny tekst artykułu
    full_article = "\n\n".join([b.get("text", "") for b in batches]).strip()
    if not full_article:
        return jsonify({"error": "Article text empty"}), 400

    model = genai.GenerativeModel("gemini-1.5-pro")

    # --------------------------------------------------------
    # 1️⃣ AUDYT EKSPERCKI
    # --------------------------------------------------------
    review_prompt = (
        "Podaj w punktach szczegółową ocenę przesłanego artykułu pod kątem:\n"
        "1. merytorycznym (zgodność faktów, aktualność, błędy logiczne),\n"
        "2. redakcyjnym (struktura, powtórzenia, styl),\n"
        "3. językowym (poprawność gramatyczna, płynność),\n"
        "a także zaproponuj konkretne poprawki dla każdego problemu.\n\n"
        "Artykuł:\n---\n" + full_article[:15000]  # limit zabezpieczający
    )

    try:
        print(f"[REVIEW] 🔍 Analiza artykułu projektu {project_id}...")
        review_response = model.generate_content(review_prompt)
        review_text = review_response.text.strip()
    except Exception as e:
        print(f"[REVIEW] ❌ Błąd podczas generowania oceny: {e}")
        return jsonify({"error": str(e)}), 500

    # --------------------------------------------------------
    # 2️⃣ AUTOMATYCZNA KOREKTA (jeśli aktywna)
    # --------------------------------------------------------
    corrected_text = None
    if os.getenv("AUTO_CORRECT_AFTER_REVIEW", "true").lower() == "true":
        try:
            correction_prompt = (
                "Popraw poniższy artykuł zgodnie z sugestiami z raportu, "
                "zachowując sens, ton i oryginalną strukturę H2/H3.\n"
                "RAPORT:\n" + review_text + "\n\n"
                "---\n\nARTYKUŁ DO POPRAWY:\n" + full_article
            )
            print("[REVIEW] ✏️ Generowanie poprawionej wersji artykułu...")
            correction_response = model.generate_content(correction_prompt)
            corrected_text = correction_response.text.strip()
        except Exception as e:
            print(f"[REVIEW] ⚠️ Błąd korekty: {e}")
            corrected_text = None

    # --------------------------------------------------------
    # 3️⃣ Zapis wyników w Firestore
    # --------------------------------------------------------
    try:
        doc_ref.update({
            "final_review": {
                "review_text": review_text,
                "corrected_text": corrected_text,
                "created_at": firestore.SERVER_TIMESTAMP,
                "model": "gemini-1.5-pro",
                "auto_correct_applied": bool(corrected_text)
            }
        })
        print(f"[REVIEW] ✅ Raport końcowy zapisany w Firestore → {project_id}")
    except Exception as e:
        print(f"[REVIEW] ⚠️ Błąd zapisu do Firestore: {e}")

    # --------------------------------------------------------
    # 4️⃣ Zwrócenie wyników do frontu / API
    # --------------------------------------------------------
    return jsonify({
        "status": "REVIEW_COMPLETE",
        "project_id": project_id,
        "review": review_text,
        "corrected_text": corrected_text,
        "auto_correct": bool(corrected_text)
    }), 200
