# ================================================================
# 🔧 POPRAWKI DO paa_routes.py
# ================================================================
# Zamień TYLKO funkcję analyze_for_paa na poniższą wersję.
# Reszta pliku zostaje bez zmian.
# ================================================================

# ZMIEŃ LINIĘ ~39 (odczyt keywords):
# BYŁO:
#   current = kw_data.get("current_count", 0)
# ZMIEŃ NA:
#   current = kw_data.get("actual_uses", kw_data.get("current_count", 0))

# ZMIEŃ LINIĘ ~27 (odczyt H2):
# BYŁO:
#   original_h2_list = data.get("h2_list", [])
# ZMIEŃ NA:
#   original_h2_list = data.get("h2_structure", data.get("h2_list", []))

# ZMIEŃ LINIĘ ~29-31 (odczyt PAA z SERP):
# BYŁO:
#   serp_paa = data.get("serp_data", {}).get("paa_questions", [])
# ZMIEŃ NA:
#   s1_data = data.get("s1_data", data.get("serp_data", {}))
#   serp_analysis = s1_data.get("serp_analysis", s1_data)
#   serp_paa = serp_analysis.get("paa_questions", [])

# ================================================================
# PEŁNA POPRAWIONA FUNKCJA (skopiuj całą):
# ================================================================

@paa_routes.get("/api/project/<project_id>/paa/analyze")
def analyze_for_paa(project_id):
    """
    Analizuje projekt i zwraca dane do PAA.
    v23.1 - kompatybilny z nową strukturą danych.
    """
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict() or {}
    
    # Dane projektu
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    
    # ============================================================
    # v23 FIX: Czytaj h2_structure (v23) z fallback na h2_list (v22)
    # ============================================================
    original_h2_list = data.get("h2_structure", data.get("h2_list", []))
    
    # ============================================================
    # v23 FIX: PAA z s1_data.serp_analysis (v23) z fallback na serp_data (v22)
    # ============================================================
    s1_data = data.get("s1_data", data.get("serp_data", {}))
    serp_analysis = s1_data.get("serp_analysis", s1_data)
    serp_paa = serp_analysis.get("paa_questions", [])
    if not serp_paa:
        serp_paa = data.get("paa_questions", [])
    
    # --------------------------------------------
    # Znajdź NIEWYKORZYSTANE frazy
    # ============================================================
    # v23 FIX: Czytaj actual_uses (v23) z fallback na current_count (v22)
    # ============================================================
    # --------------------------------------------
    unused_extended = []
    unused_basic = []
    underused = []
    
    for kw_id, kw_data in keywords_state.items():
        # v23 FIX: actual_uses zamiast current_count
        current = kw_data.get("actual_uses", kw_data.get("current_count", 0))
        target_min = kw_data.get("target_min", 1)
        keyword = kw_data.get("keyword", "")
        kw_type = kw_data.get("type", "BASIC")
        
        if current == 0:
            if kw_type == "EXTENDED":
                unused_extended.append(keyword)
            elif not kw_data.get("is_main_keyword"):  # v23: nie dodawaj main keyword
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
    full_article = "\n\n".join([b.get("text", "") for b in batches])
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
    # Response dla Custom GPT
    # --------------------------------------------
    return jsonify({
        "status": "READY_FOR_PAA",
        "project_id": project_id,
        "main_keyword": main_keyword,
        
        "unused_data": {
            "extended_keywords": unused_extended[:10],
            "basic_keywords": unused_basic[:10],
            "underused_keywords": underused[:5],
            "unused_h2": unused_h2[:5],
            "serp_paa": unused_paa[:5]
        },
        
        "avoid_in_faq": {
            "topics_already_covered": topics_to_avoid,
            "reason": "Te tematy są już omówione w artykule - FAQ musi odpowiadać na INNE pytania!"
        },
        
        "original_serp_paa": serp_paa[:10],
        
        "summary": {
            "extended_unused": len(unused_extended),
            "basic_unused": len(unused_basic),
            "h2_unused": len(unused_h2),
            "serp_paa_available": len(unused_paa),
            "topics_to_avoid": len(topics_to_avoid)
        },
        
        "instructions": {
            "goal": "Napisz sekcję PAA z 3 pytaniami",
            "critical_rule": "FAQ NIE MOŻE powtarzać tematów z artykułu! Sprawdź 'avoid_in_faq'.",
            "priority": "1. serp_paa (prawdziwe pytania z Google!), 2. extended_keywords, 3. unused_h2",
            "question_format": "Zacznij od: Jak/Czy/Co/Dlaczego/Kiedy/Ile/Czego (5-10 słów)",
            "answer_format": "80-120 słów, pierwsze zdanie = bezpośrednia odpowiedź",
            "save_endpoint": f"POST /api/project/{project_id}/paa/save"
        }
    }), 200
