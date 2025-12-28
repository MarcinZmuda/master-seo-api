import os
import json
import re
from flask import Blueprint, jsonify, request
from firebase_admin import firestore

# ------------------------------------------------------------
# Blueprint
# ------------------------------------------------------------
paa_routes = Blueprint("paa_routes", __name__)

# ------------------------------------------------------------
# 1. ANALYZE - Przygotuj dane do PAA (dla Custom GPT)
# ------------------------------------------------------------
@paa_routes.get("/api/project/<project_id>/paa/analyze")
def analyze_for_paa(project_id):
    """
    v23.8: Analizuje projekt i zwraca:
    - Semantic gaps (tematy niepokryte w artykule!)
    - Niewykorzystane frazy EXTENDED
    - Niewykorzystane frazy BASIC
    - Niewykorzystane H2
    - PAA z SERP
    
    Custom GPT używa tych danych do napisania sekcji PAA.
    FAQ powinno wypełniać LUKI SEMANTYCZNE!
    """
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict() or {}
    
    # Dane projektu
    main_keyword = data.get("topic") or data.get("main_keyword", "")
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    original_h2_list = data.get("h2_list", [])
    
    # PAA z SERP
    serp_paa = data.get("serp_data", {}).get("paa_questions", [])
    if not serp_paa:
        serp_paa = data.get("paa_questions", [])
    
    # --------------------------------------------
    # v23.8: SEMANTIC GAPS ANALYSIS
    # --------------------------------------------
    semantic_gaps = []
    full_article = "\n\n".join([b.get("text", "") for b in batches])
    
    try:
        from text_analyzer import analyze_semantic_coverage
        if full_article and keywords_state:
            keywords_list = [meta.get("keyword", "") for meta in keywords_state.values() if meta.get("keyword")]
            sem_result = analyze_semantic_coverage(full_article, keywords_list)
            semantic_gaps = sem_result.get("gaps", [])
    except Exception as e:
        print(f"[PAA] Semantic analysis skipped: {e}")
    
    # --------------------------------------------
    # Znajdź NIEWYKORZYSTANE frazy
    # --------------------------------------------
    unused_extended = []
    unused_basic = []
    underused = []
    
    for kw_id, kw_data in keywords_state.items():
        current = kw_data.get("current_count", 0)
        target_min = kw_data.get("target_min", 1)
        keyword = kw_data.get("keyword", "")
        kw_type = kw_data.get("type", "BASIC")
        
        if current == 0:
            if kw_type == "EXTENDED":
                unused_extended.append(keyword)
            else:
                unused_basic.append(keyword)
        elif current < target_min:
            underused.append({
                "keyword": keyword,
                "type": kw_type,
                "used": current,
                "target": target_min,
                "missing": target_min - current
            })
    
    # --------------------------------------------
    # Znajdź UŻYTE H2 (tematy już omówione - do unikania w FAQ!)
    # --------------------------------------------
    used_h2 = []
    
    # Format h2:
    for match in re.findall(r'^h2:\s*(.+)$', full_article, re.MULTILINE | re.IGNORECASE):
        used_h2.append(match.strip())
    # Format <h2>
    for match in re.findall(r'<h2[^>]*>([^<]+)</h2>', full_article, re.IGNORECASE):
        used_h2.append(match.strip())
    
    # Tematy do UNIKANIA w FAQ (już omówione w artykule)
    topics_to_avoid = used_h2.copy()
    
    # Znajdź niewykorzystane H2 z oryginalnej listy
    used_h2_lower = set([h.lower() for h in used_h2])
    unused_h2 = [h2 for h2 in original_h2_list if h2.lower().strip() not in used_h2_lower]
    
    # --------------------------------------------
    # Znajdź niewykorzystane PAA z SERP
    # --------------------------------------------
    unused_paa = []
    article_lower = full_article.lower()
    
    for paa_q in serp_paa:
        paa_clean = paa_q.lower().replace("?", "").strip()
        paa_words = [w for w in paa_clean.split() if len(w) > 3]
        
        if paa_words:
            words_in_article = sum(1 for w in paa_words if w in article_lower)
            if words_in_article / len(paa_words) < 0.5:
                unused_paa.append(paa_q)
    
    # --------------------------------------------
    # Response dla Custom GPT - v23.9 WSZYSTKIE nieużyte frazy
    # --------------------------------------------
    return jsonify({
        "status": "READY_FOR_PAA",
        "project_id": project_id,
        "main_keyword": main_keyword,
        
        # v23.8: SEMANTIC GAPS - najważniejsze!
        "semantic_gaps": {
            "keywords": semantic_gaps,  # WSZYSTKIE gaps
            "reason": "Te tematy NIE są pokryte w artykule - FAQ powinno je wypełnić!"
        },
        
        # v23.9: WSZYSTKIE nieużyte frazy (nie tylko 10)
        "unused_keywords": {
            "basic": unused_basic,      # WSZYSTKIE nieużyte BASIC
            "extended": unused_extended, # WSZYSTKIE nieużyte EXTENDED
            "total": len(unused_basic) + len(unused_extended),
            "instruction": "Wpleć te frazy w odpowiedzi FAQ!"
        },
        
        "underused_keywords": underused,  # Frazy poniżej minimum
        
        "unused_h2": unused_h2,  # Tematy z planu które nie zostały użyte
        
        "serp_paa": unused_paa,  # Pytania z Google PAA
        
        "avoid_in_faq": {
            "topics": topics_to_avoid,
            "reason": "Te tematy są już omówione w artykule - NIE powtarzaj!"
        },
        
        "summary": {
            "semantic_gaps": len(semantic_gaps),
            "basic_unused": len(unused_basic),
            "extended_unused": len(unused_extended),
            "underused": len(underused),
            "h2_unused": len(unused_h2),
            "serp_paa": len(unused_paa)
        },
        
        "instructions": {
            "goal": "Napisz sekcję FAQ z 3-5 pytaniami",
            "priority_1": "Wypełnij SEMANTIC GAPS - tematy niepokryte!",
            "priority_2": "Użyj WSZYSTKICH nieużytych fraz (basic + extended)",
            "priority_3": "Użyj pytań z serp_paa (prawdziwe z Google)",
            "critical": "FAQ NIE MOŻE powtarzać tematów z artykułu!",
            "format": {
                "question": "5-10 słów, zacznij od Jak/Czy/Co/Dlaczego",
                "answer": "80-120 słów, pierwsze zdanie = odpowiedź"
            }
        }
    }), 200


# ------------------------------------------------------------
# 2. SAVE - Zapisz wygenerowaną sekcję PAA
# ------------------------------------------------------------
@paa_routes.post("/api/project/<project_id>/paa/save")
def save_paa_section(project_id):
    """
    Zapisuje sekcję PAA wygenerowaną przez Custom GPT.
    
    Expected body:
    {
        "questions": [
            {
                "question": "Pytanie?",
                "answer": "Odpowiedź...",
                "keywords_used": ["fraza1", "fraza2"]
            }
        ]
    }
    """
    
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON data"}), 400
    
    questions = body.get("questions", [])
    if not questions:
        return jsonify({"error": "No questions provided"}), 400
    
    if len(questions) < 1 or len(questions) > 5:
        return jsonify({"error": "Expected 1-5 questions"}), 400
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    # Walidacja struktury
    validated_questions = []
    all_keywords = []
    
    for i, q in enumerate(questions):
        if not q.get("question") or not q.get("answer"):
            return jsonify({"error": f"Question {i+1} missing question or answer"}), 400
        
        validated_questions.append({
            "question": q["question"].strip(),
            "answer": q["answer"].strip(),
            "keywords_used": q.get("keywords_used", []),
            "word_count": len(q["answer"].split())
        })
        all_keywords.extend(q.get("keywords_used", []))
    
    # Generuj HTML z Schema.org
    html_output = _generate_faq_schema_html(validated_questions)
    
    # Generuj tekst z markerami (format artykułu)
    marker_output = _generate_marker_format(validated_questions)
    
    # Zapisz do Firestore
    paa_data = {
        "questions": validated_questions,
        "html_schema": html_output,
        "marker_format": marker_output,
        "keywords_used": list(set(all_keywords)),
        "question_count": len(validated_questions),
        "created_at": firestore.SERVER_TIMESTAMP
    }
    
    try:
        doc_ref.update({"paa_section": paa_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "status": "PAA_SAVED",
        "project_id": project_id,
        "questions_saved": len(validated_questions),
        "keywords_used": list(set(all_keywords)),
        "html_schema": html_output,
        "marker_format": marker_output
    }), 200


# ------------------------------------------------------------
# 3. GET - Pobierz zapisaną sekcję PAA
# ------------------------------------------------------------
@paa_routes.get("/api/project/<project_id>/paa")
def get_paa_section(project_id):
    """Pobiera zapisaną sekcję PAA."""
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict() or {}
    paa_section = data.get("paa_section")
    
    if not paa_section:
        return jsonify({
            "status": "NOT_GENERATED",
            "message": "PAA not generated yet",
            "next_step": f"GET /api/project/{project_id}/paa/analyze"
        }), 200
    
    return jsonify({
        "status": "OK",
        "paa_section": paa_section
    }), 200


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def _generate_faq_schema_html(questions: list) -> str:
    """Generuje HTML z Schema.org FAQPage markup."""
    
    items_html = []
    for q in questions:
        item = f'''    <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
      <h3 itemprop="name">{q["question"]}</h3>
      <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
        <p itemprop="text">{q["answer"]}</p>
      </div>
    </div>'''
        items_html.append(item)
    
    html = f'''<section itemscope itemtype="https://schema.org/FAQPage">
  <h2>Najczęściej zadawane pytania</h2>
{chr(10).join(items_html)}
</section>'''
    
    return html


def _generate_marker_format(questions: list) -> str:
    """Generuje tekst w formacie markerów (h2:/h3:)."""
    
    lines = ["h2: Najczęściej zadawane pytania", ""]
    
    for q in questions:
        lines.append(f"h3: {q['question']}")
        lines.append("")
        lines.append(q["answer"])
        lines.append("")
    
    return "\n".join(lines)
