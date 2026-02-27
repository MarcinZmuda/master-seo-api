"""
PROJECT ROUTES - SEMANTIC PLANNING v44.1
========================================
Semantyczny planner rozmieszczenia fraz w artykule.

Zawiera:
- create_semantic_keyword_plan - plan rozmieszczenia fraz

Autor: BRAJEN SEO Engine v44.1
"""

from typing import List, Dict


# ================================================================
# THEMATIC RULES
# ================================================================
THEMATIC_RULES = {
    "definition": {
        "h2_patterns": ["czym jest", "co to", "definicja", "pojęcie"],
        "keyword_patterns": ["oznacza", "definicja", "pojęcie", "istota", "jest instytucją"]
    },
    "conditions": {
        "h2_patterns": ["kiedy", "przesłanki", "warunki", "w jakich"],
        "keyword_patterns": ["przesłanka", "warunek", "choroba psychiczna", "niedorozwój", "zaburzenia"]
    },
    "procedure": {
        "h2_patterns": ["jak", "procedura", "postępowanie", "wniosek", "krok po kroku"],
        "keyword_patterns": ["wniosek", "złożyć", "sąd okręgowy", "dokumenty", "postępowanie"]
    },
    "effects": {
        "h2_patterns": ["skutki", "konsekwencje", "prawa", "co oznacza"],
        "keyword_patterns": ["opiekun", "przedstawiciel", "utrata", "ograniczenie", "zdolność"]
    },
    "types": {
        "h2_patterns": ["rodzaje", "typy", "całkowite", "częściowe", "różnice"],
        "keyword_patterns": ["całkowite", "częściowe", "różnica", "rodzaj"]
    }
}


# ================================================================
# SEMANTIC KEYWORD PLAN
# ================================================================

def create_semantic_keyword_plan(
    h2_structure: List[str],
    keywords_state: Dict,
    s1_data: Dict,
    main_keyword: str,
    total_batches: int
) -> Dict:
    """
    v36.0: Tworzy semantyczny plan rozmieszczenia fraz w artykule.
    
    Zamiast dawać GPT wszystkie frazy w każdym batchu,
    przypisujemy frazy do KONKRETNYCH sekcji H2 na podstawie:
    - Dopasowania słów (keyword matching)
    - N-gramów (co-occurrence z S1)
    - Encji (semantic context)
    - Heurystyk tematycznych
    
    Returns:
        {
            "batch_plans": [...],
            "keyword_assignments": {...}
        }
    """
    
    # Pobierz dane z S1
    ngrams = s1_data.get("ngrams", [])
    entities = s1_data.get("entity_seo", {}).get("entities", []) or s1_data.get("entities", [])
    
    # Normalizuj encje
    entity_names = []
    for e in entities:
        if isinstance(e, dict):
            entity_names.append(e.get("name", "").lower())
        else:
            entity_names.append(str(e).lower())
    
    # Normalizuj n-gramy
    ngram_list = []
    for n in ngrams:
        if isinstance(n, dict):
            ngram_list.append(n.get("ngram", "").lower())
        else:
            ngram_list.append(str(n).lower())
    
    # Zbierz wszystkie frazy
    all_keywords = {}
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if keyword:
            all_keywords[keyword.lower()] = {
                "rid": rid,
                "keyword": keyword,
                "type": meta.get("type", "BASIC").upper(),
                "target_min": meta.get("target_min", 1),
                "target_max": meta.get("target_max", 5),
                "is_main": meta.get("is_main_keyword", False)
            }
    
    # ================================================================
    # KROK 1: Dopasuj H2 do kategorii tematycznych
    # ================================================================
    h2_categories = {}
    for i, h2 in enumerate(h2_structure):
        h2_lower = h2.lower()
        h2_words = set(h2_lower.split())
        
        best_category = None
        best_score = 0
        
        for category, rules in THEMATIC_RULES.items():
            score = 0
            for pattern in rules["h2_patterns"]:
                if pattern in h2_lower:
                    score += 2
                for word in pattern.split():
                    if word in h2_words:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_category = category
        
        h2_categories[i] = {
            "h2": h2,
            "category": best_category,
            "score": best_score
        }
    
    # ================================================================
    # KROK 2: Przypisz frazy do H2
    # ================================================================
    keyword_assignments = {}
    # v67 FIX: Use max(total_batches, h2_count) for dict keys
    # h2_structure can differ from total_batches after dynamic scaling
    _max_batch = max(total_batches, len(h2_structure) + 1)
    batch_keywords = {i: [] for i in range(_max_batch + 1)}
    universal_keywords = []
    
    for kw_lower, kw_data in all_keywords.items():
        keyword = kw_data["keyword"]
        kw_type = kw_data["type"]
        is_main = kw_data["is_main"]
        target_max = kw_data["target_max"]
        
        # Fraza główna → uniwersalna
        if is_main or keyword.lower() == main_keyword.lower():
            universal_keywords.append(keyword)
            keyword_assignments[keyword] = {"batches": "all", "reason": "main_keyword"}
            continue
        
        # Frazy z wysokim limitem → uniwersalne
        if target_max >= total_batches:
            universal_keywords.append(keyword)
            keyword_assignments[keyword] = {"batches": "all", "reason": f"high_limit ({target_max})"}
            continue
        
        # Dopasuj do H2
        best_h2_idx = None
        best_match_score = 0
        best_reason = ""
        
        kw_words = set(kw_lower.split())
        
        for h2_idx, h2_info in h2_categories.items():
            h2 = h2_info["h2"]
            h2_lower = h2.lower()
            h2_words = set(h2_lower.split())
            category = h2_info["category"]
            
            score = 0
            reason_parts = []
            
            # 1. Bezpośrednie dopasowanie słów
            common_words = kw_words.intersection(h2_words)
            if common_words:
                score += len(common_words) * 3
                reason_parts.append(f"common_words:{list(common_words)}")
            
            # 2. Substring match
            if kw_lower in h2_lower or h2_lower in kw_lower:
                score += 5
                reason_parts.append("substring_match")
            
            # 3. Heurystyki tematyczne
            if category:
                rules = THEMATIC_RULES.get(category, {})
                for pattern in rules.get("keyword_patterns", []):
                    if pattern in kw_lower:
                        score += 4
                        reason_parts.append(f"thematic:{category}")
                        break
            
            # 4. N-gram co-occurrence
            for ngram in ngram_list[:30]:
                if kw_lower in ngram and any(w in ngram for w in h2_words):
                    score += 2
                    reason_parts.append("ngram_cooccur")
                    break
            
            if score > best_match_score:
                best_match_score = score
                best_h2_idx = h2_idx
                best_reason = ", ".join(reason_parts) if reason_parts else "default"
        
        # Przypisz do najlepszego H2
        if best_match_score > 0 and best_h2_idx is not None:
            batch_num = best_h2_idx + 2
            if batch_num > total_batches:
                batch_num = total_batches
            
            if batch_num not in batch_keywords:
                batch_keywords[batch_num] = []
            batch_keywords[batch_num].append(keyword)
            keyword_assignments[keyword] = {
                "batch": batch_num,
                "h2": h2_structure[best_h2_idx] if best_h2_idx < len(h2_structure) else None,
                "score": best_match_score,
                "reason": best_reason
            }
        else:
            # Hash distribution
            keyword_hash = hash(keyword) % total_batches
            batch_num = keyword_hash + 1
            
            if batch_num not in batch_keywords:
                batch_keywords[batch_num] = []
            batch_keywords[batch_num].append(keyword)
            keyword_assignments[keyword] = {
                "batch": batch_num,
                "h2": None,
                "score": 0,
                "reason": "hash_distribution"
            }
    
    # ================================================================
    # KROK 3: Przypisz encje do H2
    # ================================================================
    entity_assignments = {}
    batch_entities = {i: [] for i in range(total_batches + 1)}
    
    for entity in entity_names[:20]:
        entity_words = set(entity.split())
        best_h2_idx = None
        best_score = 0
        
        for h2_idx, h2_info in h2_categories.items():
            h2_words = set(h2_info["h2"].lower().split())
            common = entity_words.intersection(h2_words)
            score = len(common) * 2
            
            if score > best_score:
                best_score = score
                best_h2_idx = h2_idx
        
        if best_score > 0 and best_h2_idx is not None:
            batch_num = min(best_h2_idx + 2, total_batches)
            batch_entities[batch_num].append(entity)
            entity_assignments[entity] = {"batch": batch_num, "score": best_score}
        else:
            batch_num = (hash(entity) % total_batches) + 1
            batch_entities[batch_num].append(entity)
            entity_assignments[entity] = {"batch": batch_num, "score": 0}
    
    # ================================================================
    # KROK 4: Przypisz n-gramy do H2
    # ================================================================
    ngram_assignments = {}
    batch_ngrams = {i: [] for i in range(total_batches + 1)}
    
    for ngram in ngram_list[:30]:
        ngram_words = set(ngram.split())
        best_h2_idx = None
        best_score = 0
        
        for h2_idx, h2_info in h2_categories.items():
            h2_words = set(h2_info["h2"].lower().split())
            common = ngram_words.intersection(h2_words)
            score = len(common)
            
            if score > best_score:
                best_score = score
                best_h2_idx = h2_idx
        
        if best_score > 0 and best_h2_idx is not None:
            batch_num = min(best_h2_idx + 2, total_batches)
            batch_ngrams[batch_num].append(ngram)
            ngram_assignments[ngram] = {"batch": batch_num}
        else:
            batch_num = (hash(ngram) % total_batches) + 1
            batch_ngrams[batch_num].append(ngram)
            ngram_assignments[ngram] = {"batch": batch_num}
    
    # ================================================================
    # KROK 5: Zbuduj finalne batch_plans
    # ================================================================
    batch_plans = []
    
    for batch_num in range(1, total_batches + 1):
        h2_idx = batch_num - 2
        h2_for_batch = h2_structure[h2_idx] if 0 <= h2_idx < len(h2_structure) else None
        
        assigned_kws = batch_keywords.get(batch_num, [])
        
        reserved_kws = []
        for other_batch, kws in batch_keywords.items():
            if other_batch != batch_num and other_batch > 0:
                for kw in kws:
                    if kw not in universal_keywords:
                        reserved_kws.append({
                            "keyword": kw,
                            "reserved_for_batch": other_batch,
                            "reserved_for_h2": h2_structure[other_batch - 2] if 0 <= other_batch - 2 < len(h2_structure) else None
                        })
        
        batch_plan = {
            "batch_number": batch_num,
            "batch_type": "intro" if batch_num == 1 else ("final" if batch_num == total_batches else "content"),
            "h2": h2_for_batch,
            "h2_category": h2_categories.get(h2_idx, {}).get("category") if h2_idx >= 0 else None,
            "assigned_keywords": assigned_kws,
            "universal_keywords": universal_keywords,
            "reserved_keywords": reserved_kws[:20],
            "assigned_entities": batch_entities.get(batch_num, []),
            "assigned_ngrams": batch_ngrams.get(batch_num, [])
        }
        
        batch_plans.append(batch_plan)
    
    return {
        "batch_plans": batch_plans,
        "keyword_assignments": keyword_assignments,
        "entity_assignments": entity_assignments,
        "ngram_assignments": ngram_assignments,
        "universal_keywords": universal_keywords,
        "h2_categories": h2_categories,
        "stats": {
            "total_keywords": len(all_keywords),
            "universal_count": len(universal_keywords),
            "assigned_count": len(keyword_assignments) - len(universal_keywords),
            "total_entities": len(entity_names),
            "total_ngrams": len(ngram_list)
        }
    }


# ================================================================
# EXPORTS
# ================================================================
__all__ = [
    "THEMATIC_RULES",
    "create_semantic_keyword_plan",
]
