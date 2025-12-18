import os
import json
import re
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import google.generativeai as genai

# ------------------------------------------------------------
# Konfiguracja
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
FINAL_REVIEW_MODEL = os.getenv("FINAL_REVIEW_MODEL", "gemini-2.5-flash")

final_review_routes = Blueprint("final_review_routes", __name__)

# ------------------------------------------------------------
# Domyślne banned phrases
# ------------------------------------------------------------
DEFAULT_BANNED = {
    "phrases": [
        "warto zauważyć", "warto podkreślić", "warto wspomnieć", "warto dodać",
        "w dzisiejszych czasach", "w obecnych czasach",
        "podsumowując", "reasumując",
        "nie da się ukryć", "nie ulega wątpliwości",
        "należy pamiętać", "należy zauważyć",
        "kluczowe znaczenie", "odgrywa kluczową rolę",
        "jak wiadomo", "powszechnie wiadomo",
        "oczywiste jest", "jasne jest"
    ],
    "openers": ["Dlatego", "Ponadto", "Dodatkowo", "Warto", "Należy"],
    "typography": ["—", "–", "…"]
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_banned(db):
    try:
        doc = db.collection("seo_config").document("banned_phrases").get()
        if doc.exists:
            return doc.to_dict()
    except:
        pass
    return DEFAULT_BANNED

def _join_article(batches):
    return "\n\n".join([b.get("text", "") for b in (batches or [])]).strip()

# ------------------------------------------------------------
# FINAL REVIEW - Audyt redakcyjny
# ------------------------------------------------------------
@final_review_routes.post("/api/project/<project_id>/final_review")
def perform_final_review(project_id):
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API not configured"}), 500

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict() or {}
    batches = data.get("batches", [])
    full_article = _join_article(batches)
    
    if not full_article:
        return jsonify({"error": "Empty article"}), 400

    # Sprawdź czy już istnieje
    force = request.args.get("force", "").lower() in ("1", "true")
    body = request.get_json(silent=True) or {}
    if body.get("force"):
        force = True
        
    existing = data.get("final_review")
    if existing and not force:
        return jsonify({
            "status": "REVIEW_READY",
            "review": existing.get("review_data") if isinstance(existing, dict) else existing,
            "note": "Użyj ?force=true aby ponowić"
        }), 200

    # Pobierz banned phrases
    banned = _get_banned(db)
    banned_phrases = banned.get("phrases", DEFAULT_BANNED["phrases"])
    banned_typography = banned.get("typography", DEFAULT_BANNED["typography"])
    
    # Pre-scan
    article_lower = full_article.lower()
    found_phrases = [p for p in banned_phrases if p.lower() in article_lower]
    found_typo = [c for c in banned_typography if c in full_article]
    
    word_count = len(full_article.split())

    # ============================================
    # PROMPT - PROSTY I JASNY
    # ============================================
    prompt = f"""Jesteś korektorem i redaktorem. Sprawdź artykuł przed publikacją.

## SPRAWDŹ:

### A. BŁĘDY JĘZYKOWE
1. Ortografia i literówki
2. Interpunkcja (przecinki, kropki)
3. Spójność czasów (czas teraźniejszy vs przeszły)
4. Zbyt długie zdania (>30 słów)

### B. POWTÓRZENIA
5. Powtórzenia słów w sąsiednich zdaniach
6. **Te same informacje w różnych sekcjach** - czy coś nie zostało powiedziane 2x?
7. **Podobne zdania/akapity** - czy autor nie napisał tego samego innymi słowami?
8. **Nakładające się tematy w H2** - czy sekcje się nie dublują?

### C. SPÓJNOŚĆ ARTYKUŁU
9. Czy artykuł płynie logicznie od sekcji do sekcji?
10. Czy nie ma skoków tematycznych?
11. Czy wnioski/podsumowania nie powtarzają wstępu?

### D. FACT-CHECK (WAŻNE!)
12. **Błędne fakty** - czy są twierdzenia niezgodne z prawdą?
13. **Nieaktualne informacje** - czy dane/statystyki mogą być przestarzałe?
14. **Niespójności logiczne** - czy autor nie zaprzecza sam sobie?
15. **Wątpliwe twierdzenia** - czy coś brzmi nieprawdopodobnie i wymaga weryfikacji?
16. **Brakujące zastrzeżenia** - czy przy radach medycznych/prawnych/finansowych są odpowiednie disclaimery?

## USUŃ TE FRAZY (brzmią sztucznie):
{', '.join(banned_phrases[:15])}

## USUŃ TĘ TYPOGRAFIĘ:
— → przecinek lub kropka
– → myślnik zwykły lub "do"  
… → kropka

## PRE-SCAN (już wykryte):
- Zakazane frazy: {', '.join(found_phrases) if found_phrases else 'brak'}
- Zła typografia: {', '.join(found_typo) if found_typo else 'brak'}

## ARTYKUŁ ({word_count} słów):
{full_article}

## ODPOWIEDZ W JSON:
{{
  "status": "OK|WYMAGA_POPRAWEK|WYMAGA_WERYFIKACJI",
  "podsumowanie": "1-2 zdania ogólnej oceny",
  
  "bledy_jezykowe": [
    {{"typ": "ortografia|interpunkcja|czas|zdanie_za_dlugie", "lokalizacja": "cytat 3-5 słów", "poprawka": "jak poprawić"}}
  ],
  
  "powtorzenia": [
    {{"typ": "slowo|informacja|sekcja", "gdzie": "H2: X i H2: Y", "co_sie_powtarza": "opis", "sugestia": "usuń z sekcji X lub połącz"}}
  ],
  
  "spojnosc": {{
    "ocena": "dobra|srednia|slaba",
    "problemy": ["opis problemu jeśli są"]
  }},
  
  "fact_check": {{
    "status": "OK|WYMAGA_WERYFIKACJI|BLEDY",
    "problemy": [
      {{"twierdzenie": "cytat z artykułu", "problem": "dlaczego wątpliwe/błędne", "waznosc": "minor|major|krytyczny", "sugestia": "jak poprawić lub zweryfikować"}}
    ]
  }},
  
  "zakazane_frazy": {json.dumps(found_phrases)},
  "zla_typografia": {json.dumps(found_typo)},
  
  "statystyki": {{
    "bledy_jezykowe": 0,
    "powtorzenia": 0,
    "fact_check_problemy": 0,
    "zakazane_frazy": {len(found_phrases)},
    "typografia": {len(found_typo)}
  }}
}}"""

    try:
        model = genai.GenerativeModel(FINAL_REVIEW_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 2048}
        )
        
        text = (response.text or "").strip()
        text = re.sub(r'^```json?\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        
        try:
            review_data = json.loads(text)
        except:
            match = re.search(r'\{[\s\S]*\}', text)
            review_data = json.loads(match.group()) if match else {"raw": text[:500]}
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Zapisz
    doc_ref.update({
        "final_review": {
            "review_data": review_data,
            "created_at": firestore.SERVER_TIMESTAMP,
            "status": "REVIEW_READY"
        }
    })

    return jsonify({
        "status": "REVIEW_READY",
        "review": review_data,
        "article_length": word_count
    }), 200


# ------------------------------------------------------------
# APPLY CORRECTIONS - Korekta
# ------------------------------------------------------------
@final_review_routes.post("/api/project/<project_id>/apply_final_corrections")
def apply_final_corrections(project_id):
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API not configured"}), 500

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict() or {}
    review = data.get("final_review", {})
    review_data = review.get("review_data", {}) if isinstance(review, dict) else {}
    
    batches = data.get("batches", [])
    full_article = _join_article(batches)
    
    if not full_article:
        return jsonify({"error": "Empty article"}), 400

    errors = review_data.get("bledy_jezykowe", [])
    repetitions = review_data.get("powtorzenia", [])
    fact_issues = review_data.get("fact_check", {}).get("problemy", [])
    banned_found = review_data.get("zakazane_frazy", [])
    typo_found = review_data.get("zla_typografia", [])

    # ============================================
    # PROMPT KOREKTY
    # ============================================
    prompt = f"""Popraw artykuł. Wprowadź TYLKO te zmiany:

## BŁĘDY JĘZYKOWE DO POPRAWIENIA:
{json.dumps(errors, ensure_ascii=False, indent=2) if errors else 'Brak'}

## POWTÓRZENIA DO USUNIĘCIA/POŁĄCZENIA:
{json.dumps(repetitions, ensure_ascii=False, indent=2) if repetitions else 'Brak'}
(usuń zduplikowane informacje - zostaw tylko w jednej sekcji)

## PROBLEMY MERYTORYCZNE DO POPRAWIENIA:
{json.dumps(fact_issues, ensure_ascii=False, indent=2) if fact_issues else 'Brak'}
(popraw błędne fakty, usuń wątpliwe twierdzenia lub dodaj zastrzeżenia)

## FRAZY DO USUNIĘCIA:
{', '.join(banned_found) if banned_found else 'Brak'}
(zamień na naturalne odpowiedniki lub usuń)

## TYPOGRAFIA DO ZAMIANY:
— → przecinek lub kropka
– → zwykły myślnik
… → kropka

## ZASADY:
- NIE zmieniaj struktury
- NIE usuwaj treści
- NIE dodawaj nowych zdań
- Zachowaj nagłówki (h2:, h3:, <h2>, <h3>)

## ARTYKUŁ:
{full_article}

Zwróć TYLKO poprawiony artykuł:"""

    try:
        model = genai.GenerativeModel(FINAL_REVIEW_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 8192}
        )
        
        corrected = (response.text or "").strip()
        corrected = re.sub(r'^```(?:markdown|text|html)?\n?', '', corrected)
        corrected = re.sub(r'\n?```$', '', corrected)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Zapisz
    doc_ref.update({
        "final_review.corrected_text": corrected,
        "final_review.status": "CORRECTED",
        "final_review.updated_at": firestore.SERVER_TIMESTAMP
    })

    return jsonify({
        "status": "CORRECTED",
        "corrected_text": corrected,
        "changes": {
            "errors_fixed": len(errors),
            "repetitions_fixed": len(repetitions),
            "fact_issues_fixed": len(fact_issues),
            "phrases_removed": len(banned_found),
            "typography_fixed": len(typo_found)
        }
    }), 200


# ------------------------------------------------------------
# Config endpoints
# ------------------------------------------------------------
@final_review_routes.get("/api/config/banned_phrases")
def get_banned():
    db = firestore.client()
    return jsonify(_get_banned(db)), 200

@final_review_routes.post("/api/config/banned_phrases")
def set_banned():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    db = firestore.client()
    db.collection("seo_config").document("banned_phrases").set(data, merge=True)
    return jsonify({"status": "OK"}), 200
