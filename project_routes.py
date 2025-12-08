import uuid
import re
import os
import json
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation, detect_paragraph_rhythm

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
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": item.get("min", 1),
            "target_max": item.get("max", 5),
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

    # 🔍 Wstępna analiza rytmu akapitów
    rhythm = detect_paragraph_rhythm(batch_text)
    print(f"[DEBUG] Paragraph rhythm: {rhythm}")

    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    result["batch_text_snippet"] = batch_text[:50] + "..."
    result["paragraph_rhythm"] = rhythm

    return jsonify(result), 200

# ================================================================
# 🔁 MANUAL CORRECTION ENDPOINT
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
