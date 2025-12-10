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

# spaCy model - POPRAWKA: Używamy wersji MD (Medium) zgodnej z Dockerfile
try:
    nlp = spacy.load("pl_core_news_md")
    print("[INIT] ✅ spaCy pl_core_news_md loaded")
except OSError:
    from spacy.cli import download
    print("⚠️ Downloading pl_core_news_md fallback...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)

# ================================================================
# 🧱 H2 CUSTOM LIST + BASIC/EXTENDED + PROJECT INIT
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Required field: topic"}), 400

    topic = data["topic"].strip()
    h2_structure = data.get("h2_structure", [])
    raw_keywords = data.get("keywords_list", [])

    firestore_keywords = {}
    for item in raw_keywords:
        term = item.get("term", "").strip()
        if not term:
            continue
        doc = nlp(term)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        row_id = str(uuid.uuid4())
        target_max = item.get("max", 5)
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": item.get("min", 1),
            "target_max": target_max,
            "optimal_target": target_max,  # ⭐ TARGET = MAX!
            "remaining_max": target_max,   # ⭐ Initially = max (nothing used yet)
            "actual_uses": 0,
            "status": "UNDER",
            "type": item.get("type", "BASIC").upper()
        }

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_data = {
        "topic": topic,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "total_batches": 0,
        "version": "v18.0",
        "manual_mode": True
    }
    doc_ref.set(project_data)

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "keywords": len(firestore_keywords),
        "h2_sections": len(h2_structure)
    }), 201

# ================================================================
# ✏️ ADD BATCH + MANUAL CORRECTION MODE
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})

    # 🔍 Wstępna analiza używa process_batch_in_firestore, które wewnętrznie wywołuje unified_prevalidation
    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    # Wyciągamy rytm jeśli istnieje w wyniku
    rhythm = result.get("meta", {}).get("paragraph_rhythm", "Unknown")
    print(f"[DEBUG] Paragraph rhythm: {rhythm}")
    
    result["batch_text_snippet"] = batch_text[:50] + "..."
    result["paragraph_rhythm"] = rhythm

    return jsonify(result), 200

# ================================================================
# 🔍 MANUAL CORRECTION ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/manual_correct")
def manual_correct_batch(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    corrected_text = data["text"]
    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

    # Wykonaj prewalidację wszystkiego
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

    # Zapis do Firestore
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
# 🆕 AUTO-CORRECT ENDPOINT (S6)
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """
    Automatyczna korekta batcha używając Gemini 1.5 Flash:
    - Analizuje które frazy są UNDER lub OVER
    - Generuje poprawioną wersję tekstu
    - Zachowuje strukturę HTML i ton
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    batch_text = data["text"]
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # 🔍 Sprawdź które frazy są UNDER lub OVER
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
    
    # Jeśli nie ma czego korygować
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "message": "All keywords within target ranges",
            "corrected_text": batch_text,
            "keyword_report": {
                "under": [],
                "over": []
            }
        }), 200
    
    # 🤖 Użyj Gemini do inteligentnej korekty
    if not GEMINI_API_KEY:
        return jsonify({
            "status": "ERROR",
            "error": "Gemini API key not configured - cannot perform auto-correction",
            "keyword_report": {
                "under": under_keywords,
                "over": over_keywords
            }
        }), 500
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Przygotuj instrukcje korekty
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
1. Zachowaj WSZYSTKIE tagi HTML (<h2>, <h3>, <p>)
2. NIE zmieniaj struktury ani tonu tekstu
3. Dodawaj frazy naturalnie w kontekście
4. Usuwaj frazy poprzez parafrazy lub synonimy
5. Zachowaj profesjonalny, formalny styl

TEKST DO POPRAWY:
---
{batch_text[:10000]}
---

Zwróć TYLKO poprawiony tekst HTML, bez żadnych komentarzy.
"""
        
        print(f"[AUTO_CORRECT] Wysyłam do Gemini: {len(under_keywords)} UNDER, {len(over_keywords)} OVER")
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        
        # Usuń ewentualne markdown wrapper
        corrected_text = re.sub(r'^```html\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        print(f"[AUTO_CORRECT] ✅ Gemini zwrócił poprawiony tekst ({len(corrected_text)} znaków)")
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords],
            "keyword_report": {
                "under": under_keywords,
                "over": over_keywords
            },
            "correction_summary": f"Dodano {len(under_keywords)} fraz, usunięto nadmiar {len(over_keywords)} fraz"
        }), 200
        
    except Exception as e:
        print(f"[AUTO_CORRECT] ❌ Błąd Gemini: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "keyword_report": {
                "under": under_keywords,
                "over": over_keywords
            }
        }), 500

# ================================================================
# 🧠 UNIFIED PRE-VALIDATION (SEO + SEMANTICS + STYLE)
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})

    report = unified_prevalidation(data["text"], keywords_state)

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
# 🆕 FORCE APPROVE (ZAPIS MIMO BŁĘDÓW)
# ================================================================
@project_routes.post("/api/project/<project_id>/force_approve_batch")
def force_approve_batch(project_id):
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' is required"}), 400

    batch_text = data["text"]
    meta_trace = data.get("meta_trace", {})

    print("[FORCE APPROVE] User requested forced save.")
    return manual_correct_batch(project_id)

# ================================================================
# 📦 EXPORT (Late HTML Injection)
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

    html_ready = f"<article><p>{full_text.replace(chr(10)*2, '</p><p>')}</p></article>"

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_html": html_ready,
        "batch_count": len(batches),
        "version": "v18.0"
    }), 200
