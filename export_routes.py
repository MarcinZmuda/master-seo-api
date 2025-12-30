"""
Export Routes - v26.1 
Eksport artykułów do DOCX/HTML/TXT + Editorial Review (Gemini)

ZMIANY v26.1:
- Naprawiony editorial_review - lepsze składanie treści z batchów
- Debug info gdy batche są puste
- Obsługa różnych formatów batch.text
"""

import os
import re
import io
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

# Firebase Firestore only
from firebase_admin import firestore

# Document generation
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Gemini for Editorial Review
import google.generativeai as genai

export_routes = Blueprint("export_routes", __name__)

# Gemini config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def html_to_text(html: str) -> str:
    """Konwertuje HTML na czysty tekst."""
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n\n## \1\n\n', html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n\n### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_article_structure(text: str) -> list:
    """Parsuje artykuł na strukturę nagłówków i paragrafów."""
    elements = []
    
    # Normalizuj format
    text = re.sub(r'^h2:\s*(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^h3:\s*(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    parts = re.split(r'(<h[23][^>]*>.*?</h[23]>)', text, flags=re.IGNORECASE | re.DOTALL)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        h2_match = re.match(r'<h2[^>]*>(.*?)</h2>', part, re.IGNORECASE | re.DOTALL)
        h3_match = re.match(r'<h3[^>]*>(.*?)</h3>', part, re.IGNORECASE | re.DOTALL)
        
        if h2_match:
            elements.append({"type": "h2", "content": h2_match.group(1).strip()})
        elif h3_match:
            elements.append({"type": "h3", "content": h3_match.group(1).strip()})
        else:
            clean_text = html_to_text(part)
            if clean_text:
                elements.append({"type": "paragraph", "content": clean_text})
    
    return elements


# ================================================================
# v26.1: HELPER - Składanie treści z batchów
# ================================================================
def extract_text_from_batches(batches: list) -> tuple:
    """
    v26.1: Bezpieczne składanie treści z batchów.
    Obsługuje różne formaty: batch.text, batch.content, batch.html
    
    Returns:
        (full_text, debug_info)
    """
    texts = []
    debug_info = {
        "total_batches": len(batches),
        "batches_with_text": 0,
        "batches_empty": 0,
        "fields_found": set()
    }
    
    for i, batch in enumerate(batches):
        # Próbuj różne pola
        text = None
        
        # Kolejność priorytetów
        for field in ["text", "content", "html", "batch_text", "raw_text"]:
            if field in batch and batch[field]:
                text = str(batch[field]).strip()
                debug_info["fields_found"].add(field)
                break
        
        if text:
            texts.append(text)
            debug_info["batches_with_text"] += 1
        else:
            debug_info["batches_empty"] += 1
            print(f"[EXPORT] ⚠️ Batch {i} jest pusty. Klucze: {list(batch.keys())}")
    
    debug_info["fields_found"] = list(debug_info["fields_found"])
    full_text = "\n\n".join(texts)
    
    return full_text, debug_info


# ================================================================
# EDITORIAL REVIEW - v26.1 (naprawiony)
# ================================================================
@export_routes.post("/api/project/<project_id>/editorial_review")
def editorial_review(project_id):
    """
    v26.1: Weryfikacja przez Gemini jako redaktora naczelnego.
    NAPRAWIONE: Lepsze składanie treści z batchów + debug info.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    if not batches:
        return jsonify({
            "error": "No batches found",
            "hint": "Artykuł nie ma zapisanych batchów. Użyj addBatch aby zapisać treść.",
            "project_keys": list(project_data.keys())
        }), 400
    
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API not configured"}), 500
    
    # v26.1: Bezpieczne składanie treści
    full_text, debug_info = extract_text_from_batches(batches)
    
    # Sprawdź czy mamy treść
    word_count = len(full_text.split()) if full_text else 0
    
    if word_count < 50:
        return jsonify({
            "error": f"Article too short for review ({word_count} words)",
            "debug": debug_info,
            "hint": "Batche istnieją ale nie zawierają tekstu w polu 'text'. Sprawdź strukturę batchów.",
            "sample_batch_keys": list(batches[0].keys()) if batches else []
        }), 400
    
    # Pobierz kategorię z requestu lub z projektu
    data = request.get_json(force=True) if request.is_json else {}
    category = data.get("category") or project_data.get("category") or "prawo"
    topic = project_data.get("topic") or project_data.get("main_keyword", "")
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Jesteś redaktorem naczelnym portalu internetowego oraz specjalistą od tematyki: {category}.

Oceń poniższy artykuł pt. "{topic}" pod kątem:

1. **MERYTORYKA** (0-10): 
   - Czy informacje są poprawne i aktualne?
   - Czy są błędy rzeczowe?
   - Czy brakuje ważnych informacji?

2. **STRUKTURA** (0-10):
   - Czy artykuł jest logicznie zorganizowany?
   - Czy nagłówki dobrze odzwierciedlają treść?
   - Czy jest odpowiednia długość akapitów?

3. **STYL I JĘZYK** (0-10):
   - Czy tekst jest czytelny i płynny?
   - Czy nie ma powtórzeń?
   - Czy język jest odpowiedni dla grupy docelowej?

4. **SEO** (0-10):
   - Czy fraza główna jest odpowiednio użyta?
   - Czy nagłówki są przyjazne dla SEO?
   - Czy są linki wewnętrzne/zewnętrzne (lub ich brak)?

5. **KONKRETNE POPRAWKI**:
   Wymień 3-5 konkretnych miejsc do poprawy z cytatem i sugestią.

---
ARTYKUŁ ({word_count} słów):

{full_text[:15000]}

---
Odpowiedz w formacie JSON:
{{
  "overall_score": <średnia 0-10>,
  "scores": {{
    "merytoryka": <0-10>,
    "struktura": <0-10>,
    "styl": <0-10>,
    "seo": <0-10>
  }},
  "summary": "<2-3 zdania podsumowania>",
  "corrections": [
    {{"issue": "<problem>", "quote": "<cytat z artykułu>", "suggestion": "<jak poprawić>"}},
    ...
  ],
  "recommendation": "APPROVE" | "NEEDS_REVISION" | "MAJOR_REWRITE"
}}
"""
        
        response = model.generate_content(prompt)
        review_text = response.text.strip()
        
        # Próbuj sparsować JSON
        try:
            # Wyciągnij JSON z odpowiedzi (może być otoczony markdown)
            json_match = re.search(r'\{[\s\S]*\}', review_text)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                review_data = {"raw_response": review_text}
        except:
            review_data = {"raw_response": review_text}
        
        # Zapisz w Firestore
        db.collection("seo_projects").document(project_id).update({
            "editorial_review": {
                "review": review_data,
                "category": category,
                "word_count": word_count,
                "created_at": firestore.SERVER_TIMESTAMP
            }
        })
        
        return jsonify({
            "status": review_data.get("recommendation", "UNKNOWN"),
            "overall_score": review_data.get("overall_score"),
            "scores": review_data.get("scores", {}),
            "summary": review_data.get("summary", ""),
            "corrections": review_data.get("corrections", []),
            "word_count": word_count,
            "debug": debug_info,
            "version": "v26.1"
        }), 200
        
    except Exception as e:
        print(f"[EDITORIAL_REVIEW] Error: {e}")
        return jsonify({
            "error": f"Review failed: {str(e)}",
            "word_count": word_count,
            "debug": debug_info
        }), 500


# ================================================================
# EXPORT STATUS
# ================================================================
@export_routes.get("/api/project/<project_id>/export_status")
def export_status(project_id):
    """Sprawdza status projektu przed eksportem."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    # v26.1: Użyj nowej funkcji
    full_text, debug_info = extract_text_from_batches(batches)
    word_count = len(full_text.split()) if full_text else 0
    
    editorial = project_data.get("editorial_review", {})
    
    return jsonify({
        "project_id": project_id,
        "topic": project_data.get("topic"),
        "status": project_data.get("status", "UNKNOWN"),
        "batches_count": len(batches),
        "word_count": word_count,
        "debug": debug_info,
        "editorial_review": {
            "done": bool(editorial),
            "score": editorial.get("review", {}).get("overall_score"),
            "recommendation": editorial.get("review", {}).get("recommendation")
        },
        "ready_for_export": word_count >= 500 and debug_info["batches_with_text"] > 0,
        "actions": {
            "editorial_review": f"POST /api/project/{project_id}/editorial_review",
            "export_docx": f"GET /api/project/{project_id}/export/docx",
            "export_html": f"GET /api/project/{project_id}/export/html",
            "export_txt": f"GET /api/project/{project_id}/export/txt"
        },
        "version": "v26.1"
    }), 200


# ================================================================
# EXPORT DOCX
# ================================================================
@export_routes.get("/api/project/<project_id>/export/docx")
def export_docx(project_id):
    """Eksportuje artykuł do formatu DOCX."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    topic = project_data.get("topic", "Artykuł")
    elements = parse_article_structure(full_text)
    
    # Twórz dokument
    document = Document()
    
    # Tytuł
    title = document.add_heading(topic, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Treść
    for el in elements:
        if el["type"] == "h2":
            document.add_heading(el["content"], level=2)
        elif el["type"] == "h3":
            document.add_heading(el["content"], level=3)
        elif el["type"] == "paragraph":
            p = document.add_paragraph(el["content"])
            p.style.font.size = Pt(11)
    
    # Zapisz do bufora
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".docx"
    
    return Response(
        buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ================================================================
# EXPORT HTML
# ================================================================
@export_routes.get("/api/project/<project_id>/export/html")
def export_html(project_id):
    """Eksportuje artykuł do formatu HTML."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    topic = project_data.get("topic", "Artykuł")
    
    # Konwertuj markery na HTML
    html = full_text
    html = re.sub(r'^h2:\s*(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^h3:\s*(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Paragrafy
    lines = html.split('\n')
    processed = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('<h'):
            if line.startswith('- ') or line.startswith('• '):
                processed.append(f'<li>{line[2:]}</li>')
            else:
                processed.append(f'<p>{line}</p>')
        else:
            processed.append(line)
    
    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>{topic}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; margin-top: 30px; }}
        h3 {{ color: #555; }}
        p {{ margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>{topic}</h1>
    {''.join(processed)}
</body>
</html>"""
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".html"
    
    return Response(
        html_content,
        mimetype="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ================================================================
# EXPORT TXT
# ================================================================
@export_routes.get("/api/project/<project_id>/export/txt")
def export_txt(project_id):
    """Eksportuje artykuł do formatu TXT."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    topic = project_data.get("topic", "Artykuł")
    
    # Konwertuj do czytelnego tekstu
    txt = f"{topic}\n{'='*len(topic)}\n\n"
    txt += re.sub(r'^h2:\s*(.+)$', r'\n## \1\n', full_text, flags=re.MULTILINE)
    txt = re.sub(r'^h3:\s*(.+)$', r'\n### \1\n', txt, flags=re.MULTILINE)
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".txt"
    
    return Response(
        txt,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ================================================================
# ALIAS dla kompatybilności
# ================================================================
@export_routes.post("/api/project/<project_id>/gemini_review")
def gemini_review_alias(project_id):
    """Alias do editorial_review dla kompatybilności z v26.1"""
    return editorial_review(project_id)
