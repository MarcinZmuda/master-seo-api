"""
Export Routes - v23.8 (Simple)
Eksport artykułów do DOCX/HTML/TXT - bezpośredni download bez Storage
"""

import os
import re
import io
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

# Firebase Firestore only
from firebase_admin import firestore

# Document generation
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

export_routes = Blueprint("export_routes", __name__)


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
    
    pattern = r'(<h2[^>]*>.*?</h2>|<h3[^>]*>.*?</h3>|[^<]+)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        match = match.strip()
        if not match:
            continue
            
        if re.match(r'<h2[^>]*>', match, re.IGNORECASE):
            title = re.sub(r'</?h2[^>]*>', '', match, flags=re.IGNORECASE).strip()
            if title:
                elements.append({"type": "h2", "content": title})
        elif re.match(r'<h3[^>]*>', match, re.IGNORECASE):
            title = re.sub(r'</?h3[^>]*>', '', match, flags=re.IGNORECASE).strip()
            if title:
                elements.append({"type": "h3", "content": title})
        else:
            clean = re.sub(r'<[^>]+>', '', match).strip()
            if clean and len(clean) > 10:
                elements.append({"type": "p", "content": clean})
    
    return elements


def generate_docx(project_data: dict) -> io.BytesIO:
    """Generuje plik DOCX."""
    doc = Document()
    
    topic = project_data.get("topic") or project_data.get("main_keyword", "Artykuł")
    
    # Tytuł
    title = doc.add_heading(topic, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f"Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("─" * 50)
    
    # Treść
    batches = project_data.get("batches", [])
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    elements = parse_article_structure(full_text)
    
    for elem in elements:
        if elem["type"] == "h2":
            doc.add_heading(elem["content"], level=1)
        elif elem["type"] == "h3":
            doc.add_heading(elem["content"], level=2)
        elif elem["type"] == "p":
            p = doc.add_paragraph(elem["content"])
            p.paragraph_format.space_after = Pt(12)
    
    # FAQ
    paa_data = project_data.get("paa_section", {})
    questions = paa_data.get("questions", [])
    if questions:
        doc.add_page_break()
        doc.add_heading("FAQ - Najczęściej zadawane pytania", level=1)
        for q in questions:
            doc.add_heading(q.get("question", ""), level=2)
            doc.add_paragraph(q.get("answer", ""))
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_html(project_data: dict) -> str:
    """Generuje HTML."""
    topic = project_data.get("topic") or project_data.get("main_keyword", "Artykuł")
    
    batches = project_data.get("batches", [])
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    elements = parse_article_structure(full_text)
    
    html = f'''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>{topic}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.6; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #4a4a8a; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
h3 {{ color: #0f3460; }}
p {{ text-align: justify; }}
.meta {{ color: #666; font-size: 12px; }}
.faq {{ background: #f5f5f5; padding: 20px; margin-top: 40px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>{topic}</h1>
<p class="meta">Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
'''
    
    for elem in elements:
        if elem["type"] == "h2":
            html += f'<h2>{elem["content"]}</h2>\n'
        elif elem["type"] == "h3":
            html += f'<h3>{elem["content"]}</h3>\n'
        elif elem["type"] == "p":
            html += f'<p>{elem["content"]}</p>\n'
    
    # FAQ
    paa_data = project_data.get("paa_section", {})
    questions = paa_data.get("questions", [])
    if questions:
        html += '<div class="faq"><h2>FAQ</h2>'
        for q in questions:
            html += f'<h3>{q.get("question", "")}</h3>'
            html += f'<p>{q.get("answer", "")}</p>'
        html += '</div>'
    
    html += '</body></html>'
    return html


# ============================================================================
# ROUTES
# ============================================================================

@export_routes.get("/api/project/<project_id>/export/docx")
def export_docx(project_id):
    """Eksport do DOCX - bezpośredni download."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    if not project_data.get("batches"):
        return jsonify({"error": "No content to export"}), 400
    
    try:
        docx_buffer = generate_docx(project_data)
        
        topic = project_data.get("topic") or project_data.get("main_keyword", "article")
        safe_topic = re.sub(r'[^\w\s-]', '', topic)[:30].strip().replace(' ', '_')
        filename = f"{safe_topic}_{project_id[:8]}.docx"
        
        return Response(
            docx_buffer.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"[EXPORT] ❌ DOCX error: {e}")
        return jsonify({"error": str(e)}), 500


@export_routes.get("/api/project/<project_id>/export/html")
def export_html(project_id):
    """Eksport do HTML - bezpośredni download."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    if not project_data.get("batches"):
        return jsonify({"error": "No content to export"}), 400
    
    try:
        html_content = generate_html(project_data)
        
        topic = project_data.get("topic") or project_data.get("main_keyword", "article")
        safe_topic = re.sub(r'[^\w\s-]', '', topic)[:30].strip().replace(' ', '_')
        filename = f"{safe_topic}_{project_id[:8]}.html"
        
        return Response(
            html_content,
            mimetype="text/html; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"[EXPORT] ❌ HTML error: {e}")
        return jsonify({"error": str(e)}), 500


@export_routes.get("/api/project/<project_id>/export/txt")
def export_txt(project_id):
    """Eksport do TXT."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    if not project_data.get("batches"):
        return jsonify({"error": "No content to export"}), 400
    
    batches = project_data.get("batches", [])
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    clean_text = html_to_text(full_text)
    
    topic = project_data.get("topic") or project_data.get("main_keyword", "Artykuł")
    
    output = f"""{'='*60}
{topic.upper()}
{'='*60}
Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

{clean_text}
"""
    
    # FAQ
    paa_data = project_data.get("paa_section", {})
    questions = paa_data.get("questions", [])
    if questions:
        output += f"\n\n{'='*60}\nFAQ\n{'='*60}\n\n"
        for q in questions:
            output += f"P: {q.get('question', '')}\nO: {q.get('answer', '')}\n\n"
    
    topic = project_data.get("topic") or "article"
    safe_topic = re.sub(r'[^\w\s-]', '', topic)[:30].strip().replace(' ', '_')
    filename = f"{safe_topic}_{project_id[:8]}.txt"
    
    return Response(
        output,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@export_routes.get("/api/project/<project_id>/export")
def export_info(project_id):
    """Lista dostępnych formatów eksportu."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    return jsonify({
        "project_id": project_id,
        "topic": project_data.get("topic") or project_data.get("main_keyword"),
        "batches_count": len(batches),
        "has_faq": bool(project_data.get("paa_section", {}).get("questions")),
        "export_formats": {
            "docx": f"/api/project/{project_id}/export/docx",
            "html": f"/api/project/{project_id}/export/html",
            "txt": f"/api/project/{project_id}/export/txt"
        }
    }), 200
