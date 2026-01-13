"""
🆕 v31.4: Internal Linking Suggestions Endpoint
Dodaj do master_api.py

Analizuje artykuł i entity_relationships z S1 żeby zasugerować 
3 miejsca na linki wewnętrzne.
"""

import re
from flask import jsonify, request

# ================================================================
# 🆕 v31.4: INTERNAL LINKING SUGGESTIONS
# ================================================================
@app.post("/api/project/<project_id>/internal_linking_suggestions")
def get_internal_linking_suggestions(project_id):
    """
    🆕 v31.4: Generuje sugestie linków wewnętrznych.
    
    Analizuje:
    1. article_content - pełny tekst artykułu
    2. entity_relationships z S1 - relacje między encjami
    3. entities z S1 - encje z importance scores
    
    Zwraca max 3 sugestie linków wewnętrznych.
    """
    data = request.get_json(force=True)
    
    article_content = data.get("article_content", "")
    entity_relationships = data.get("entity_relationships", [])
    entities = data.get("entities", [])
    site_structure = data.get("site_structure", [])
    
    if not article_content or len(article_content) < 500:
        return jsonify({
            "error": "article_content too short (min 500 chars)",
            "suggestions": []
        }), 400
    
    # ============================================
    # 1. Znajdź potencjalne miejsca na linki
    # ============================================
    link_opportunities = []
    article_lower = article_content.lower()
    
    # Podziel artykuł na sekcje (po H2)
    sections = re.split(r'##\s+', article_content)
    
    # a) Szukaj fraz z entity_relationships
    for rel in entity_relationships:
        subject = rel.get("subject", "")
        obj = rel.get("object", "")
        rel_type = rel.get("type", "unknown")
        
        for phrase in [subject, obj]:
            if not phrase or len(phrase) < 3:
                continue
            
            phrase_lower = phrase.lower()
            
            # Znajdź gdzie występuje w artykule
            if phrase_lower in article_lower:
                # Znajdź kontekst
                idx = article_lower.find(phrase_lower)
                context_start = max(0, idx - 50)
                context_end = min(len(article_content), idx + len(phrase) + 50)
                context = article_content[context_start:context_end]
                
                # Znajdź w której sekcji
                section_name = "INTRO"
                char_count = 0
                for i, section in enumerate(sections):
                    char_count += len(section)
                    if idx < char_count:
                        # Wyciągnij nazwę sekcji (pierwsza linia)
                        section_lines = section.strip().split('\n')
                        if section_lines:
                            section_name = section_lines[0][:50]
                        break
                
                # Znajdź importance z entities
                importance = 0.5
                for ent in entities:
                    if ent.get("text", "").lower() == phrase_lower:
                        importance = ent.get("importance", 0.5)
                        break
                
                # Generuj sugerowany URL
                slug = phrase_lower.replace(" ", "-").replace("ą", "a").replace("ę", "e")
                slug = slug.replace("ó", "o").replace("ś", "s").replace("ł", "l")
                slug = slug.replace("ż", "z").replace("ź", "z").replace("ć", "c").replace("ń", "n")
                suggested_url = f"/{slug}/"
                
                link_opportunities.append({
                    "phrase": phrase,
                    "location": f"sekcja '{section_name}'",
                    "context": f"...{context}...",
                    "suggested_anchor": phrase,
                    "suggested_url": suggested_url,
                    "entity_importance": importance,
                    "relationship_type": rel_type,
                    "score": importance + (0.2 if rel_type in ["offers", "requires"] else 0.1)
                })
    
    # b) Szukaj encji z wysokim importance
    for ent in entities:
        ent_text = ent.get("text", "")
        importance = ent.get("importance", 0)
        ent_type = ent.get("type", "")
        
        if importance < 0.6 or not ent_text:
            continue
        
        ent_lower = ent_text.lower()
        
        # Sprawdź czy już nie mamy tej frazy
        existing_phrases = [op["phrase"].lower() for op in link_opportunities]
        if ent_lower in existing_phrases:
            continue
        
        if ent_lower in article_lower:
            idx = article_lower.find(ent_lower)
            context_start = max(0, idx - 50)
            context_end = min(len(article_content), idx + len(ent_text) + 50)
            context = article_content[context_start:context_end]
            
            # Generuj URL
            slug = ent_lower.replace(" ", "-")
            for pl, en in [("ą","a"),("ę","e"),("ó","o"),("ś","s"),("ł","l"),("ż","z"),("ź","z"),("ć","c"),("ń","n")]:
                slug = slug.replace(pl, en)
            
            link_opportunities.append({
                "phrase": ent_text,
                "location": "artykuł",
                "context": f"...{context}...",
                "suggested_anchor": ent_text,
                "suggested_url": f"/{slug}/",
                "entity_importance": importance,
                "relationship_type": f"entity_{ent_type}",
                "score": importance
            })
    
    # ============================================
    # 2. Sortuj i wybierz top 3
    # ============================================
    link_opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Usuń duplikaty (podobne frazy)
    seen_phrases = set()
    unique_opportunities = []
    for op in link_opportunities:
        phrase_key = op["phrase"].lower()[:20]  # Pierwsze 20 znaków
        if phrase_key not in seen_phrases:
            seen_phrases.add(phrase_key)
            unique_opportunities.append(op)
    
    # Weź top 3
    top_suggestions = unique_opportunities[:3]
    
    # Usuń score z outputu
    for sug in top_suggestions:
        sug.pop("score", None)
    
    # ============================================
    # 3. Jeśli podano site_structure, dopasuj URLs
    # ============================================
    if site_structure:
        for sug in top_suggestions:
            phrase_lower = sug["phrase"].lower()
            best_match = None
            best_score = 0
            
            for page in site_structure:
                page_keywords = page.get("keywords", [])
                page_title = page.get("title", "").lower()
                
                # Sprawdź dopasowanie
                score = 0
                if phrase_lower in page_title:
                    score += 0.5
                for kw in page_keywords:
                    if phrase_lower in kw.lower() or kw.lower() in phrase_lower:
                        score += 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = page.get("url", "")
            
            if best_match and best_score > 0.3:
                sug["suggested_url"] = best_match
                sug["url_matched"] = True
    
    # ============================================
    # 4. Zwróć wynik
    # ============================================
    return jsonify({
        "suggestions": top_suggestions,
        "total_link_opportunities": len(unique_opportunities),
        "rules_applied": [
            "max_3_links_per_article",
            "priority_by_entity_importance",
            "natural_anchor_text",
            "entity_relationships_first"
        ]
    }), 200


# ================================================================
# Przykład użycia w Claude Computer Use:
# ================================================================
"""
# Po saveFullArticle, wywołaj:

curl -X POST https://master-seo-api.onrender.com/api/project/PROJECT_ID/internal_linking_suggestions \\
  -H "Content-Type: application/json" \\
  -d '{
    "article_content": "PEŁNY TEKST ARTYKUŁU",
    "entity_relationships": [
      {"subject": "firma przeprowadzkowa", "verb": "oferuje", "object": "transport mebli", "type": "offers"},
      {"subject": "przeprowadzka", "verb": "wymaga", "object": "pakowanie rzeczy", "type": "requires"}
    ],
    "entities": [
      {"text": "Warszawa", "type": "LOCATION", "importance": 0.92},
      {"text": "transport mebli", "type": "CONCEPT", "importance": 0.71}
    ]
  }'

# Response:
{
  "suggestions": [
    {
      "phrase": "transport mebli",
      "location": "sekcja 'Cennik przeprowadzek'",
      "context": "...profesjonalny transport mebli w Warszawie...",
      "suggested_anchor": "transport mebli",
      "suggested_url": "/transport-mebli/",
      "entity_importance": 0.71,
      "relationship_type": "offers"
    },
    ...
  ],
  "total_link_opportunities": 8,
  "rules_applied": ["max_3_links_per_article", ...]
}
"""
