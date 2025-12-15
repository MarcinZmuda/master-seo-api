import uuid
import re
import os
import json
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING] ⚠️ GEMINI_API_KEY not set - LSI enrichment fallback mode")

# spaCy model
try:
    nlp = spacy.load("pl_core_news_md")
    print("[INIT] ✅ spaCy pl_core_news_md loaded")
except OSError:
    from spacy.cli import download
    print("⚠️ Downloading pl_core_news_md fallback...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)

# ⭐ GEMINI MODEL - centralnie zdefiniowany
GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# 🧠 H2 SUGGESTIONS (Gemini-powered)
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    """
    Generuje sugestie H2 używając Gemini na podstawie:
    - topic/main_keyword
    - wzorców H2 z konkurencji (serp_h2_patterns)
    - target keywords
    
    Zwraca listę maksymalnie 6 H2 (hard limit zgodny z seo_rules.json).
    Wstęp (intro) NIE jest H2 - to osobny element bez nagłówka.
    
    ⚠️ WAŻNE: To są tylko PROPOZYCJE. User musi podać SWOJE H2,
    które zostaną połączone z propozycjami w finalną strukturę.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "")
    if not topic:
        return jsonify({"error": "Required: topic or main_keyword"}), 400
    
    serp_h2_patterns = data.get("serp_h2_patterns", [])
    target_keywords = data.get("target_keywords", [])
    target_count = min(data.get("target_count", 6), 6)
    
    # Jeśli brak Gemini API - zwróć podstawowe sugestie
    if not GEMINI_API_KEY:
        fallback_suggestions = [
            f"Czym jest {topic}?",
            f"Jak działa {topic}?",
            f"Korzyści z {topic}",
            f"Kiedy warto skorzystać z {topic}?",
            f"Ile kosztuje {topic}?",
            f"Najczęstsze pytania o {topic}"
        ]
        return jsonify({
            "status": "FALLBACK",
            "suggestions": fallback_suggestions[:target_count],
            "message": "Gemini unavailable - basic suggestions generated",
            "model": "fallback",
            "action_required": "USER_H2_INPUT_NEEDED"
        }), 200
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        competitor_context = ""
        if serp_h2_patterns:
            competitor_context = f"""
WZORCE H2 Z KONKURENCJI (TOP 10 SERP):
{chr(10).join(f"- {h2}" for h2 in serp_h2_patterns[:20])}
"""
        
        keywords_context = ""
        if target_keywords:
            keywords_context = f"""
FRAZY KLUCZOWE DO WPLECENIA W H2:
{', '.join(target_keywords[:10])}
"""
        
        prompt = f"""
Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu SEO o temacie: "{topic}"

WAŻNE: Artykuł będzie miał WSTĘP (bez nagłówka H2) + {target_count} sekcji H2.

{competitor_context}
{keywords_context}

ZASADY:
1. Wygeneruj DOKŁADNIE {target_count} H2 (nie więcej, nie mniej)
2. Każdy H2 powinien mieć 6-12 słów
3. Minimum 50% H2 powinno być w formie pytania (Jak...?, Dlaczego...?, Kiedy...?, Co...?)
4. Maksimum 30% H2 może zawierać główne słowo kluczowe
5. NIE używaj ogólnikowych tytułów jak: "Wstęp", "Podsumowanie", "Zakończenie", "FAQ"
6. H2 powinny tworzyć logiczną strukturę artykułu
7. Uwzględnij intencję wyszukiwania (informacyjna/transakcyjna)
8. Wpleć naturalnie frazy kluczowe gdzie to możliwe

FORMAT ODPOWIEDZI:
Zwróć TYLKO listę {target_count} H2, każdy w nowej linii, bez numeracji ani punktorów.
"""
        
        print(f"[H2_SUGGESTIONS] Generating {target_count} H2 for: {topic}")
        response = model.generate_content(prompt)
        
        raw_suggestions = response.text.strip().split('\n')
        suggestions = [
            h2.strip().lstrip('•-–—0123456789.). ')
            for h2 in raw_suggestions 
            if h2.strip() and len(h2.strip()) > 5
        ][:target_count]
        
        print(f"[H2_SUGGESTIONS] ✅ Generated {len(suggestions)} H2 suggestions")
        
        return jsonify({
            "status": "OK",
            "suggestions": suggestions,
            "topic": topic,
            "model": GEMINI_MODEL,
            "count": len(suggestions),
            "action_required": "USER_H2_INPUT_NEEDED",
            "message": "To są PROPOZYCJE. Teraz podaj SWOJE H2, które chcesz wpleść, a system połączy je w finalną strukturę."
        }), 200
        
    except Exception as e:
        print(f"[H2_SUGGESTIONS] ❌ Error: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "suggestions": []
        }), 500


# ================================================================
# 🧱 PROJECT CREATE
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    """
    Tworzy nowy projekt SEO w Firestore.
    
    ⭐ NOWA LOGIKA LIMITÓW:
    - GPT widzi: target_min + 1 (ostrzegawczy limit)
    - Backend liczy do: target_max (rzeczywisty limit)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "").strip()
    if not topic:
        return jsonify({"error": "Required field: topic or main_keyword"}), 400
    
    h2_structure = data.get("h2_structure", [])
    raw_keywords = data.get("keywords_list") or data.get("keywords", [])
    target_length = data.get("target_length", 3000)
    source = data.get("source", "unknown")

    firestore_keywords = {}
    for item in raw_keywords:
        term = item.get("term") or item.get("keyword", "")
        term = term.strip() if term else ""
        
        if not term:
            continue
        
        doc = nlp(term)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        min_val = item.get("min") or item.get("target_min", 1)
        max_val = item.get("max") or item.get("target_max", 5)
        
        row_id = item.get("id") or str(uuid.uuid4())
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": min_val,
            "target_max": max_val,
            # ⭐ NOWE: display_limit to co widzi GPT (min+1)
            "display_limit": min_val + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": item.get("type", "BASIC").upper(),
            "remaining_max": max_val,
            "optimal_target": max_val
        }

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_data = {
        "topic": topic,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "batches_plan": [],
        "total_batches": 0,
        "target_length": target_length,
        "source": source,
        "version": "v22.1",
        "manual_mode": False if source == "n8n-brajen-workflow" else True,
        # ⭐ NOWE: format output
        "output_format": "clean_text_with_headers"
    }
    doc_ref.set(project_data)
    
    print(f"[PROJECT] ✅ Created project {doc_ref.id}: {topic} ({len(firestore_keywords)} keywords)")

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "keywords_count": len(firestore_keywords),
        "keywords": len(firestore_keywords),
        "h2_sections": len(h2_structure),
        "target_length": target_length,
        "source": source
    }), 201


# ================================================================
# 📊 GET PROJECT STATUS - z info o LOCKED frazach
# ================================================================
@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """
    Zwraca aktualny status projektu z informacją o LOCKED frazach.
    
    ⭐ NOWE:
    - locked_keywords: lista fraz które osiągnęły limit
    - display_limits: limity które widzi GPT (min+1)
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    
    keyword_summary = []
    locked_keywords = []
    near_limit_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        display_limit = target_min + 1  # Co widzi GPT
        
        kw_info = {
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": actual,
            "display_limit": display_limit,  # ⭐ GPT widzi to
            "target_max": target_max,  # Backend limit
            "status": meta.get("status"),
            "remaining_max": remaining
        }
        keyword_summary.append(kw_info)
        
        # ⭐ Zbierz LOCKED frazy
        if remaining == 0:
            locked_keywords.append({
                "keyword": meta.get("keyword"),
                "message": f"🔒 LOCKED: '{meta.get('keyword')}' osiągnęło limit {target_max}x. Użyj SYNONIMÓW!"
            })
        elif remaining <= 3:
            near_limit_keywords.append({
                "keyword": meta.get("keyword"),
                "remaining": remaining,
                "message": f"⚠️ NEAR LIMIT: '{meta.get('keyword')}' - zostało tylko {remaining}x"
            })
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": len(batches),
        "keywords_count": len(keywords_state),
        "keywords": keyword_summary,
        # ⭐ NOWE - wyraźne info o blokadach
        "locked_keywords": locked_keywords,
        "near_limit_keywords": near_limit_keywords,
        "warnings_before_batch": locked_keywords + near_limit_keywords,
        "source": data.get("source", "unknown"),
        "has_final_review": "final_review" in data
    }), 200


# ================================================================
# 📋 PRE-BATCH INFO - info przed pisaniem batcha
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """
    Zwraca informacje potrzebne PRZED napisaniem batcha:
    - Które frazy są LOCKED (użyj synonimów)
    - Które frazy są NEAR_LIMIT
    - Ile zostało do napisania
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    
    locked = []
    near_limit = []
    safe = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword")
        kw_type = meta.get("type", "BASIC")
        actual = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        display_limit = meta.get("target_min", 0) + 1
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "display_limit": display_limit,
            "remaining_max": remaining
        }
        
        if remaining == 0:
            kw_info["status"] = "LOCKED"
            kw_info["action"] = f"🔒 NIE UŻYWAJ '{keyword}' - użyj synonimów!"
            locked.append(kw_info)
        elif remaining <= 3:
            kw_info["status"] = "NEAR_LIMIT"
            kw_info["action"] = f"⚠️ Ostrożnie z '{keyword}' - zostało {remaining}x"
            near_limit.append(kw_info)
        else:
            kw_info["status"] = "SAFE"
            safe.append(kw_info)
    
    return jsonify({
        "project_id": project_id,
        "locked_keywords": locked,
        "near_limit_keywords": near_limit,
        "safe_keywords": safe,
        "summary": {
            "locked_count": len(locked),
            "near_limit_count": len(near_limit),
            "safe_count": len(safe)
        },
        "instructions": {
            "locked": "Dla LOCKED fraz użyj SYNONIMÓW - NIE używaj dokładnej frazy!",
            "near_limit": "Dla NEAR_LIMIT fraz - użyj max 1x w tym batchu",
            "format": "Pisz czystym tekstem. Tylko <h2> i <h3> jako tagi, reszta bez HTML."
        }
    }), 200


# ================================================================
# ✏️ ADD BATCH
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    rhythm = result.get("meta", {}).get("paragraph_rhythm", "Unknown")
    result["batch_text_snippet"] = batch_text[:50] + "..."
    result["paragraph_rhythm"] = rhythm

    return jsonify(result), 200


# ================================================================
# 🔍 MANUAL CORRECTION ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/manual_correct")
def manual_correct_batch(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    corrected_text = data.get("text") or data.get("batch_text") or data.get("corrected_text")
    if not corrected_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    precheck = unified_prevalidation(corrected_text, keywords_state)

    summary = (
        f"Semantic drift: {precheck['semantic_score']:.2f}, "
        f"Transition: {precheck['transition_score']:.2f}, "
        f"Density: {precheck['density']:.2f}, "
        f"Warnings: {len(precheck['warnings'])}"
    )

    if forced:
        print("[FORCED APPROVAL] Saving corrected batch despite warnings.")

    batch_data = {
        "id": str(uuid.uuid4()),
        "text": corrected_text,
        "meta_trace": meta_trace,
        "status": "FORCED" if forced else "APPROVED",
        "language_audit": {
            "semantic_score": precheck["semantic_score"],
            "transition_score": precheck["transition_score"],
            "density": precheck["density"]
        },
        "warnings": precheck["warnings"],
        "corrected": True
    }

    doc_ref = db.collection("seo_projects").document(project_id)
    doc_ref.update({
        "batches": firestore.ArrayUnion([batch_data]),
        "total_batches": firestore.Increment(1)
    })

    return jsonify({
        "status": "CORRECTED_SAVED",
        "project_id": project_id,
        "summary": summary,
        "forced": forced
    }), 200


# ================================================================
# 🆕 AUTO-CORRECT ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """
    Automatyczna korekta batcha używając Gemini.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text") or data.get("corrected_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    under_keywords = []
    over_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        min_target = meta.get("target_min", 0)
        max_target = meta.get("target_max", 999)
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC")
        
        if actual < min_target:
            under_keywords.append({
                "keyword": keyword,
                "missing": min_target - actual,
                "type": kw_type,
                "current": actual,
                "target_min": min_target
            })
        elif actual > max_target:
            over_keywords.append({
                "keyword": keyword,
                "excess": actual - max_target,
                "type": kw_type,
                "current": actual,
                "target_max": max_target
            })
    
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "message": "All keywords within target ranges",
            "corrected_text": batch_text,
            "keyword_report": {"under": [], "over": []}
        }), 200
    
    if not GEMINI_API_KEY:
        return jsonify({
            "status": "ERROR",
            "error": "Gemini API key not configured",
            "keyword_report": {"under": under_keywords, "over": over_keywords}
        }), 500
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        correction_instructions = []
        
        if under_keywords:
            under_list = "\n".join([
                f"  - '{kw['keyword']}': Dodaj {kw['missing']}× (obecnie {kw['current']}/{kw['target_min']})"
                for kw in under_keywords
            ])
            correction_instructions.append(f"DODAJ te frazy naturalnie:\n{under_list}")
        
        if over_keywords:
            over_list = "\n".join([
                f"  - '{kw['keyword']}': Usuń {kw['excess']}× (obecnie {kw['current']}, max {kw['target_max']})"
                for kw in over_keywords
            ])
            correction_instructions.append(f"USUŃ nadmiar tych fraz:\n{over_list}")
        
        correction_prompt = f"""
Popraw poniższy tekst SEO według instrukcji:

{chr(10).join(correction_instructions)}

ZASADY:
1. Zachowaj nagłówki <h2> i <h3>
2. Reszta tekstu ma być CZYSTYM TEKSTEM (bez <p>, bez <strong>, bez list)
3. Dodawaj frazy naturalnie w kontekście
4. Usuwaj frazy poprzez parafrazy lub synonimy
5. Zachowaj profesjonalny, formalny styl
6. Ta sama fraza BASIC nie może występować częściej niż 1x na 3 zdania

TEKST DO POPRAWY:
---
{batch_text[:10000]}
---

Zwróć TYLKO poprawiony tekst, bez żadnych komentarzy.
"""
        
        print(f"[AUTO_CORRECT] Wysyłam do Gemini: {len(under_keywords)} UNDER, {len(over_keywords)} OVER")
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        
        # Usuń ewentualne markdown/html wrappery
        corrected_text = re.sub(r'^```(?:html)?\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        print(f"[AUTO_CORRECT] ✅ Gemini zwrócił poprawiony tekst ({len(corrected_text)} znaków)")
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords],
            "keyword_report": {"under": under_keywords, "over": over_keywords},
            "correction_summary": f"Dodano {len(under_keywords)} fraz, usunięto nadmiar {len(over_keywords)} fraz"
        }), 200
        
    except Exception as e:
        print(f"[AUTO_CORRECT] ❌ Błąd Gemini: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "keyword_report": {"under": under_keywords, "over": over_keywords}
        }), 500


# ================================================================
# 🧠 UNIFIED PRE-VALIDATION
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text = data.get("text") or data.get("batch_text")
    if not text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})

    report = unified_prevalidation(text, keywords_state)

    return jsonify({
        "status": "CHECKED",
        "semantic_score": report["semantic_score"],
        "transition_score": report["transition_score"],
        "density": report["density"],
        "warnings": report["warnings"],
        "summary": f"Semantic: {report['semantic_score']:.2f}, "
                   f"Transition: {report['transition_score']:.2f}, "
                   f"Density: {report['density']:.2f}, "
                   f"Warnings: {len(report['warnings'])}"
    }), 200


# ================================================================
# 🆕 FORCE APPROVE
# ================================================================
@project_routes.post("/api/project/<project_id>/force_approve_batch")
def force_approve_batch(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    print("[FORCE APPROVE] User requested forced save.")
    return manual_correct_batch(project_id)


# ================================================================
# 📦 EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_text": full_text,  # ⭐ Czysty tekst, nie HTML
        "batch_count": len(batches),
        "version": "v22.1"
    }), 200


# ================================================================
# 🔄 ALIAS: auto_correct_keywords
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    """Alias dla auto_correct - kompatybilność z OpenAPI schema."""
    return auto_correct_batch(project_id)
