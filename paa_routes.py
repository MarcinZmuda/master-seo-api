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
    
    # PAA z SERP — check multiple storage paths
    serp_paa = data.get("serp_data", {}).get("paa_questions", [])
    if not serp_paa:
        serp_paa = data.get("paa_questions", [])
    if not serp_paa:
        # v55.1: PAA saved by Brajn2026 in s1_data.paa (list of dicts with "question" key)
        s1_paa_raw = data.get("s1_data", {}).get("paa", [])
        serp_paa = [
            p.get("question", "") if isinstance(p, dict) else str(p)
            for p in s1_paa_raw if p
        ]
        serp_paa = [q for q in serp_paa if q]  # remove empty strings
    
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
    # v27.2: Używamy actual_uses (nie current_count!)
    # --------------------------------------------
    unused_extended = []
    unused_basic = []
    underused = []
    keyword_status = []  # v27.2: Pełny status fraz dla GPT
    
    for kw_id, kw_data in keywords_state.items():
        # v27.2: actual_uses to właściwe pole!
        current = kw_data.get("actual_uses", kw_data.get("current_count", 0))
        target_min = kw_data.get("target_min", 1)
        target_max = kw_data.get("target_max", 999)
        keyword = kw_data.get("keyword", "")
        kw_type = kw_data.get("type", "BASIC").upper()
        
        # Status dla GPT
        keyword_status.append({
            "keyword": keyword,
            "type": kw_type,
            "actual": current,
            "target_min": target_min,
            "target_max": target_max,
            "remaining": max(0, target_max - current)
        })
        
        if current == 0:
            if kw_type == "EXTENDED":
                unused_extended.append(keyword)
            elif kw_type in ["BASIC", "MAIN", "ENTITY"]:
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
    # Response dla Custom GPT - v27.2 z keyword_status
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
            "instruction": "⚠️ WPLEĆ WSZYSTKIE EXTENDED W FAQ! Każda MUSI być użyta 1x."
        },
        
        # v27.2: Pełny status fraz - żeby GPT wiedział ile może użyć
        "keyword_status": {
            "all_keywords": keyword_status,
            "warning": "⚠️ Nie przekraczaj target_max! Sprawdź 'remaining' przed użyciem."
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
            "goal": "Napisz sekcję FAQ z 3 pytaniami",
            "priority_1": "⚠️ UŻYJ WSZYSTKICH unused_keywords.extended! Każda fraza 1x w odpowiedzi.",
            "priority_2": "Użyj nieużytych BASIC jeśli są",
            "priority_3": "Użyj pytań z serp_paa (prawdziwe z Google)",
            "critical": "FAQ NIE MOŻE powtarzać tematów z artykułu!",
            "warning": "⚠️ Sprawdź keyword_status.remaining zanim użyjesz frazy - nie przekraczaj limitów!",
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
    
    # v27.2: Przelicz frazy w FAQ i zaktualizuj keywords_state
    faq_text = "\n".join([f"{q['question']} {q['answer']}" for q in validated_questions])
    
    rescan_result = {"rescanned": False}
    try:
        from keyword_counter import count_keywords_for_state
        
        project_data = doc.to_dict() or {}
        keywords_state = project_data.get("keywords_state", {})
        
        if keywords_state and faq_text:
            # Policz frazy w FAQ
            faq_counts = count_keywords_for_state(faq_text, keywords_state, use_exclusive_for_nested=False)
            
            # Dodaj do actual_uses
            changes = []
            for rid, faq_count in faq_counts.items():
                if faq_count > 0:
                    meta = keywords_state.get(rid, {})
                    old_actual = meta.get("actual_uses", 0)
                    new_actual = old_actual + faq_count
                    meta["actual_uses"] = new_actual
                    
                    # Przelicz status
                    min_t = meta.get("target_min", 0)
                    max_t = meta.get("target_max", 999)
                    if new_actual < min_t:
                        meta["status"] = "UNDER"
                    elif new_actual >= max_t:
                        meta["status"] = "OVER" if new_actual > max_t else "OPTIMAL"
                    else:
                        meta["status"] = "OK"
                    
                    keywords_state[rid] = meta
                    changes.append({
                        "keyword": meta.get("keyword"),
                        "added_in_faq": faq_count,
                        "new_total": new_actual
                    })
            
            if changes:
                # Zapisz zaktualizowany keywords_state
                doc_ref.update({"keywords_state": keywords_state})
                print(f"[PAA_SAVE] ✅ Rescan: {len(changes)} keywords updated from FAQ")
                rescan_result = {"rescanned": True, "changes": changes}
    except Exception as e:
        print(f"[PAA_SAVE] ⚠️ Rescan failed: {e}")
    
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
        "rescan_result": rescan_result,  # v27.2: Info o przeliczonych frazach
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
