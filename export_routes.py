"""
Export Routes - v27.0
Eksport artykułów do DOCX/HTML/TXT + Editorial Review (Claude API)

ZMIANY v27.0:
- Zmiana z Gemini na Claude API (dokładniejsza analiza)
- Uniwersalny prompt (bez specyficznych przykładów branżowych)
- Zwraca analizę + poprawiony tekst
- Wplata nieużyte frazy BASIC/EXTENDED
- Wymusza minimalną długość tekstu (nie skraca)
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

# Claude API for Editorial Review
import anthropic

export_routes = Blueprint("export_routes", __name__)

# Claude config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
claude_client = None
if ANTHROPIC_API_KEY:
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("[EXPORT] ✅ Claude API configured")
else:
    print("[EXPORT] ⚠️ ANTHROPIC_API_KEY not set")

# Fallback: Gemini (jeśli Claude niedostępny)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai = None
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    if not ANTHROPIC_API_KEY:
        print("[EXPORT] ℹ️ Using Gemini as fallback")


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
# v27.0: HELPER - Składanie treści z batchów
# ================================================================
def extract_text_from_batches(batches: list) -> tuple:
    """
    Bezpieczne składanie treści z batchów.
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
        text = None
        
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
# EDITORIAL REVIEW - v27.0 (Claude API + poprawiony tekst)
# ================================================================
@export_routes.post("/api/project/<project_id>/editorial_review")
def editorial_review(project_id):
    """
    v27.0: Weryfikacja przez Claude jako redaktora naczelnego.
    Zwraca: analiza + poprawiony tekst + wplecione nieużyte frazy.
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
    
    # Sprawdź czy mamy API
    if not claude_client and not GEMINI_API_KEY:
        return jsonify({"error": "No AI API configured (need ANTHROPIC_API_KEY or GEMINI_API_KEY)"}), 500
    
    # Składanie treści z batchów
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
    
    # Pobierz temat z projektu
    topic = project_data.get("topic") or project_data.get("main_keyword", "artykuł")
    
    # v27.0: Pobierz nieużyte frazy BASIC i EXTENDED
    keywords_state = project_data.get("keywords_state", {})
    unused_basic = []
    unused_extended = []
    
    for rid, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        
        if actual == 0 and kw:
            if kw_type == "BASIC":
                unused_basic.append(kw)
            elif kw_type == "EXTENDED":
                unused_extended.append(kw)
    
    # Sekcja nieużytych fraz dla promptu
    unused_keywords_section = ""
    if unused_basic or unused_extended:
        unused_keywords_section = "\n=== ⚠️ NIEWYKORZYSTANE FRAZY SEO - MUSISZ JE WPLEŚĆ! ===\n"
        if unused_basic:
            unused_keywords_section += f"\n**BASIC (KRYTYCZNE - muszą być w tekście min. 1x):**\n"
            unused_keywords_section += "\n".join(f"• {kw}" for kw in unused_basic)
            unused_keywords_section += "\n"
        if unused_extended:
            unused_keywords_section += f"\n**EXTENDED (dodaj naturalnie 1x każdą):**\n"
            unused_keywords_section += "\n".join(f"• {kw}" for kw in unused_extended)
            unused_keywords_section += "\n"
        unused_keywords_section += "\nINSTRUKCJA: Wpleć te frazy NATURALNIE w poprawiony tekst. Rozbuduj istniejące akapity lub dodaj nowe zdania.\n"
    
    try:
        # ============================================================
        # UNIWERSALNY PROMPT DO CLAUDE
        # ============================================================
        prompt = f"""Jesteś REDAKTOREM NACZELNYM i EKSPERTEM MERYTORYCZNYM w tematyce artykułu.

Otrzymujesz artykuł pt. "{topic}" do weryfikacji i korekty.
{unused_keywords_section}
=== TWOJE ZADANIA ===

**ZADANIE 1: ZNAJDŹ I NAPRAW BŁĘDY KRYTYCZNE**

1. **HALUCYNACJE DANYCH** - Zmyślone statystyki, daty, nazwy raportów → USUŃ lub zamień na ogólne stwierdzenie
2. **BŁĘDY TERMINOLOGICZNE** - Mylenie pojęć fachowych specyficznych dla tematu artykułu → POPRAW
3. **ZABURZONA CHRONOLOGIA** - Nielogiczna kolejność kroków/instrukcji → PRZESTAW prawidłowo
4. **NADMIERNE UPROSZCZENIA** - "zawsze/każdy/nigdy/wszyscy" → "zazwyczaj/często/w większości przypadków"
5. **NIEAKTUALNE INFORMACJE** - Przestarzałe dane, przepisy, procedury → USUŃ lub zaznacz że wymaga weryfikacji

**ZADANIE 2: POPRAW JAKOŚĆ TEKSTU**

- Usuń typowe frazy AI: "w dzisiejszych czasach", "warto wiedzieć", "nie ulega wątpliwości", "należy pamiętać"
- Popraw błędne kolokacje
- Usuń powtórzenia i rozwlekłe fragmenty

**ZADANIE 3: WYMAGANIA KRYTYCZNE**

⚠️ DŁUGOŚĆ: Tekst MUSI mieć MINIMUM {word_count} słów! Nie skracaj - możesz tylko wydłużyć!
⚠️ FORMAT: Zachowaj strukturę HTML/Markdown (H2, H3, paragrafy)
⚠️ LINKI: NIE dodawaj żadnych linków

=== ODPOWIEDZ W FORMACIE JSON ===

{{
  "analysis": {{
    "overall_score": <0-10>,
    "scores": {{"merytoryka": <0-10>, "struktura": <0-10>, "styl": <0-10>, "seo": <0-10>}},
    "critical_errors_found": [
      {{"type": "<TYP>", "original": "<cytat>", "fixed": "<poprawka>"}}
    ],
    "keywords_added": ["<wplecione frazy>"],
    "minor_fixes": ["<drobne poprawki>"],
    "summary": "<2-3 zdania>"
  }},
  "corrected_article": "<CAŁY POPRAWIONY ARTYKUŁ - MINIMUM {word_count} SŁÓW!>"
}}

=== ARTYKUŁ DO WERYFIKACJI ({word_count} słów) ===

{full_text}
"""
        
        # v27.0: Użyj Claude API (preferowany) lub Gemini (fallback)
        if claude_client:
            print(f"[EDITORIAL_REVIEW] Using Claude API for project {project_id} ({word_count} words)")
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            review_text = response.content[0].text.strip()
            ai_model = "claude"
        elif genai:
            print(f"[EDITORIAL_REVIEW] Using Gemini (fallback) for project {project_id}")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            review_text = response.text.strip()
            ai_model = "gemini"
        else:
            return jsonify({
                "error": "No AI API configured",
                "hint": "Set ANTHROPIC_API_KEY or GEMINI_API_KEY"
            }), 500
        
        # Parsuj JSON z odpowiedzi
        try:
            json_match = re.search(r'\{[\s\S]*\}', review_text)
            if json_match:
                review_data = json.loads(json_match.group())
            else:
                review_data = {"raw_response": review_text}
        except:
            review_data = {"raw_response": review_text}
        
        # Wyciągnij analysis i corrected_article
        analysis = review_data.get("analysis", review_data)
        corrected_article = review_data.get("corrected_article", "")
        corrected_word_count = len(corrected_article.split()) if corrected_article else 0
        
        # Zapisz w Firestore
        db.collection("seo_projects").document(project_id).update({
            "editorial_review": {
                "analysis": analysis,
                "corrected_article": corrected_article,
                "original_word_count": word_count,
                "corrected_word_count": corrected_word_count,
                "unused_keywords": {
                    "basic": unused_basic,
                    "extended": unused_extended
                },
                "ai_model": ai_model,
                "created_at": firestore.SERVER_TIMESTAMP
            }
        })
        
        return jsonify({
            "status": analysis.get("recommendation", "REVIEWED"),
            "overall_score": analysis.get("overall_score"),
            "scores": analysis.get("scores", {}),
            "critical_errors": analysis.get("critical_errors_found", []),
            "keywords_added": analysis.get("keywords_added", []),
            "minor_fixes": analysis.get("minor_fixes", []),
            "summary": analysis.get("summary", ""),
            "corrected_article": corrected_article,
            "word_count": {
                "original": word_count,
                "corrected": corrected_word_count
            },
            "unused_keywords_input": {
                "basic": unused_basic,
                "extended": unused_extended
            },
            "ai_model": ai_model,
            "version": "v27.0"
        }), 200
        
    except Exception as e:
        print(f"[EDITORIAL_REVIEW] Error: {e}")
        import traceback
        traceback.print_exc()
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
            "score": editorial.get("analysis", {}).get("overall_score"),
            "corrected_word_count": editorial.get("corrected_word_count"),
            "ai_model": editorial.get("ai_model")
        },
        "ready_for_export": word_count >= 500 and debug_info["batches_with_text"] > 0,
        "actions": {
            "editorial_review": f"POST /api/project/{project_id}/editorial_review",
            "export_docx": f"GET /api/project/{project_id}/export/docx",
            "export_html": f"GET /api/project/{project_id}/export/html",
            "export_txt": f"GET /api/project/{project_id}/export/txt"
        },
        "version": "v27.0"
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
        print(f"[EXPORT_DOCX] Using corrected article ({len(corrected.split())} words)")
    else:
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
    else:
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
    else:
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
# ALIASY dla kompatybilności
# ================================================================
@export_routes.post("/api/project/<project_id>/gemini_review")
def gemini_review_alias(project_id):
    """Alias do editorial_review dla kompatybilności"""
    return editorial_review(project_id)


@export_routes.post("/api/project/<project_id>/claude_review")
def claude_review_alias(project_id):
    """Alias do editorial_review"""
    return editorial_review(project_id)
