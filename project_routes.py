import uuid
import re
import os
import json
import math
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# v23.9: Współdzielony model spaCy (oszczędność RAM)
from shared_nlp import get_nlp
nlp = get_nlp()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING] ⚠️ GEMINI_API_KEY not set - LSI enrichment fallback mode")
project_routes = Blueprint("project_routes", __name__)

# ⭐ GEMINI MODEL - centralnie zdefiniowany
GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# ⭐ v22.4: SYNONYM DETECTION dla frazy głównej
# ================================================================
def detect_main_keyword_synonyms(main_keyword: str) -> list:
    """
    Używa Gemini do znalezienia synonimów frazy głównej.
    """
    if not GEMINI_API_KEY:
        return []
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
Podaj 2-4 SYNONIMY lub WARIANTY dla frazy: "{main_keyword}"

ZASADY:
- Tylko frazy które znaczą TO SAMO
- Mogą być używane zamiennie w tekście SEO
- Format: jeden synonim na linię, bez numeracji

Odpowiedź (tylko synonimy):
"""
        response = model.generate_content(prompt)
        synonyms = [s.strip() for s in response.text.strip().split('\n') if s.strip() and len(s.strip()) > 2]
        return synonyms[:4]
    except Exception as e:
        print(f"[SYNONYM] ❌ Error: {e}")
        return []


# ================================================================
# 🧠 H2 SUGGESTIONS (Gemini-powered) - v22.4: z wymuszeniem fraz
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    """
    Generuje sugestie H2 używając Gemini.
    ⭐ v22.4: H2 MUSZĄ zawierać frazę główną w min. 20%
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "")
    if not topic:
        return jsonify({"error": "Required: topic or main_keyword"}), 400
    
    serp_h2_patterns = data.get("serp_h2_patterns", [])
    target_keywords = data.get("target_keywords", [])
    target_count = min(data.get("target_count", 6), 6)
    
    # Jeśli brak Gemini API - zwróć podstawowe sugestie
    if not GEMINI_API_KEY:
        fallback_suggestions = [
            f"Czym jest {topic}?",
            f"Jak działa {topic}?",
            f"Korzyści z {topic}",
            f"Kiedy warto skorzystać z {topic}?",
            f"Ile kosztuje {topic}?",
            f"Najczęstsze pytania o {topic}"
        ]
        return jsonify({
            "status": "FALLBACK",
            "suggestions": fallback_suggestions[:target_count],
            "message": "Gemini unavailable - basic suggestions generated",
            "model": "fallback"
        }), 200
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        competitor_context = ""
        if serp_h2_patterns:
            competitor_context = f"""
WZORCE H2 Z KONKURENCJI (TOP 10 SERP):
{chr(10).join(f"- {h2}" for h2 in serp_h2_patterns[:20])}
"""
        
        keywords_context = ""
        if target_keywords:
            keywords_context = f"""
FRAZY KLUCZOWE DO WPLECENIA W H2:
{', '.join(target_keywords[:10])}
"""
        
        # ⭐ v22.4: NOWY PROMPT z wymuszeniem frazy głównej w H2
        prompt = f"""
Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu SEO o temacie: "{topic}"

{competitor_context}
{keywords_context}

⭐ KRYTYCZNE ZASADY:
1. Minimum {max(2, target_count // 2)} z {target_count} H2 MUSI zawierać frazę "{topic}" lub jej wariant!
2. NIE UŻYWAJ ogólników: "dokument", "wniosek", "sprawa", "proces"
3. UŻYWAJ konkretnej frazy głównej: "{topic}"
4. Każdy H2 powinien mieć 6-12 słów
5. Minimum 30% H2 w formie pytania (Jak...?, Ile...?, Gdzie...?)
6. NIE używaj: "Wstęp", "Podsumowanie", "Zakończenie", "FAQ"

PRZYKŁAD DLA "pozew o rozwód":
✅ "Jak napisać pozew o rozwód krok po kroku?"
✅ "Jakie dokumenty dołączyć do pozwu o rozwód?"
✅ "Ile kosztuje pozew o rozwód w 2025 roku?"
❌ "Jak przygotować dokumenty?" (brak frazy głównej!)
❌ "Najważniejsze elementy wniosku" (ogólnik!)

FORMAT: Zwróć TYLKO listę {target_count} H2, każdy w nowej linii.
"""
        
        print(f"[H2_SUGGESTIONS] Generating {target_count} H2 for: {topic}")
        response = model.generate_content(prompt)
        
        raw_suggestions = response.text.strip().split('\n')
        suggestions = [
            h2.strip().lstrip('•-–—0123456789.). ')
            for h2 in raw_suggestions 
            if h2.strip() and len(h2.strip()) > 5
        ][:target_count]
        
        # ⭐ v22.4: Walidacja - ile H2 zawiera frazę główną
        topic_lower = topic.lower()
        h2_with_main = sum(1 for h2 in suggestions if topic_lower in h2.lower())
        coverage = h2_with_main / len(suggestions) if suggestions else 0
        
        print(f"[H2_SUGGESTIONS] ✅ Generated {len(suggestions)} H2, {h2_with_main} contain main keyword ({coverage:.0%})")
        
        return jsonify({
            "status": "OK",
            "suggestions": suggestions,
            "topic": topic,
            "model": GEMINI_MODEL,
            "count": len(suggestions),
            "main_keyword_coverage": {
                "h2_with_main_keyword": h2_with_main,
                "total_h2": len(suggestions),
                "coverage_percent": round(coverage * 100, 1),
                "valid": coverage >= 0.2
            },
            "action_required": "USER_H2_INPUT_NEEDED"
        }), 200
        
    except Exception as e:
        print(f"[H2_SUGGESTIONS] ❌ Error: {e}")
        return jsonify({"status": "ERROR", "error": str(e), "suggestions": []}), 500


# ================================================================
# 🧱 PROJECT CREATE - v22.4: z MAIN_KEYWORD i synonimami
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    """
    Tworzy nowy projekt SEO w Firestore.
    
    ⭐ v22.4 NOWE:
    - is_main_keyword flag
    - main_keyword_synonyms 
    - Wyższe minimum dla frazy głównej
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    topic = data.get("topic") or data.get("main_keyword", "").strip()
    if not topic:
        return jsonify({"error": "Required field: topic or main_keyword"}), 400
    
    h2_structure = data.get("h2_structure", [])
    raw_keywords = data.get("keywords_list") or data.get("keywords", [])
    target_length = data.get("target_length", 3000)
    source = data.get("source", "unknown")
    
    total_planned_batches = data.get("total_planned_batches")
    if not total_planned_batches:
        total_planned_batches = max(2, min(6, math.ceil(len(h2_structure) / 2))) if h2_structure else 4

    # ⭐ v22.4: Wykryj synonimy frazy głównej
    main_keyword_synonyms = detect_main_keyword_synonyms(topic)
    print(f"[PROJECT] 🔍 Main keyword synonyms for '{topic}': {main_keyword_synonyms}")

    firestore_keywords = {}
    main_keyword_found = False
    
    for item in raw_keywords:
        term = item.get("term") or item.get("keyword", "")
        term = term.strip() if term else ""
        
        if not term:
            continue
        
        doc = nlp(term)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        min_val = item.get("min") or item.get("target_min", 1)
        max_val = item.get("max") or item.get("target_max", 5)
        
        row_id = item.get("id") or str(uuid.uuid4())
        
        # ⭐ v22.4: Sprawdź czy to fraza główna
        is_main = term.lower() == topic.lower()
        if is_main:
            main_keyword_found = True
            # Wyższe minimum dla frazy głównej: ~1x na 350 słów
            min_val = max(min_val, max(6, target_length // 350))
            max_val = max(max_val, target_length // 150)
        
        # ⭐ v22.4: Sprawdź czy to synonim frazy głównej
        is_synonym_of_main = term.lower() in [s.lower() for s in main_keyword_synonyms]
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": min_val,
            "target_max": max_val,
            "display_limit": min_val + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN" if is_main else item.get("type", "BASIC").upper(),
            "is_main_keyword": is_main,
            "is_synonym_of_main": is_synonym_of_main,
            "remaining_max": max_val,
            "optimal_target": max_val
        }
    
    # ⭐ v22.4: Jeśli fraza główna nie była w liście - dodaj ją
    if not main_keyword_found:
        main_min = max(6, target_length // 350)
        main_max = target_length // 150
        
        doc = nlp(topic)
        search_lemma = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
        
        firestore_keywords["main_keyword_auto"] = {
            "keyword": topic,
            "search_term_exact": topic.lower(),
            "search_lemma": search_lemma,
            "target_min": main_min,
            "target_max": main_max,
            "display_limit": main_min + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": "MAIN",
            "is_main_keyword": True,
            "is_synonym_of_main": False,
            "remaining_max": main_max,
            "optimal_target": main_max
        }
        print(f"[PROJECT] ⭐ Auto-added main keyword '{topic}' with min={main_min}, max={main_max}")

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_data = {
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "batches_plan": [],
        "total_batches": 0,
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        "source": source,
        "version": "v22.4",
        "manual_mode": False if source == "n8n-brajen-workflow" else True,
        "output_format": "clean_text_with_headers"
    }
    doc_ref.set(project_data)
    
    print(f"[PROJECT] ✅ Created project {doc_ref.id}: {topic} ({len(firestore_keywords)} keywords, {total_planned_batches} planned batches)")

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "main_keyword": topic,
        "main_keyword_synonyms": main_keyword_synonyms,
        "keywords_count": len(firestore_keywords),
        "h2_sections": len(h2_structure),
        "total_planned_batches": total_planned_batches,
        "target_length": target_length,
        "source": source
    }), 201


# ================================================================
# 📊 GET PROJECT STATUS - v22.4: z info o MAIN vs SYNONYMS
# ================================================================
@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """
    Zwraca aktualny status projektu.
    ⭐ v22.4: Dodaje proporcje main keyword vs synonyms
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    
    keyword_summary = []
    locked_keywords = []
    near_limit_keywords = []
    
    # ⭐ v22.4: Track main vs synonyms
    main_keyword_uses = 0
    synonym_uses = 0
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        
        kw_info = {
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "status": meta.get("status"),
            "remaining_max": remaining,
            "is_main_keyword": meta.get("is_main_keyword", False),
            "is_synonym_of_main": meta.get("is_synonym_of_main", False)
        }
        keyword_summary.append(kw_info)
        
        # ⭐ v22.4: Sumuj użycia main vs synonyms
        if meta.get("is_main_keyword"):
            main_keyword_uses = actual
        elif meta.get("is_synonym_of_main"):
            synonym_uses += actual
        
        if remaining == 0:
            locked_keywords.append({
                "keyword": meta.get("keyword"),
                "message": f"🔒 LOCKED: '{meta.get('keyword')}' osiągnęło limit {target_max}x"
            })
        elif remaining <= 3:
            near_limit_keywords.append({
                "keyword": meta.get("keyword"),
                "remaining": remaining
            })
    
    # ⭐ v22.4: Oblicz proporcję
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 0
    
    main_vs_synonym_status = {
        "main_keyword": main_keyword,
        "main_uses": main_keyword_uses,
        "synonym_uses": synonym_uses,
        "total": total_main_and_synonyms,
        "main_ratio": round(main_ratio, 2),
        "valid": main_ratio >= 0.3,  # Main powinno być >= 30%
        "warning": None if main_ratio >= 0.3 else f"⚠️ Fraza główna ma tylko {main_ratio:.0%} użyć. Zamień synonimy na '{main_keyword}'!"
    }
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": len(batches),
        "keywords_count": len(keywords_state),
        "keywords": keyword_summary,
        "locked_keywords": locked_keywords,
        "near_limit_keywords": near_limit_keywords,
        "main_vs_synonyms": main_vs_synonym_status,
        "source": data.get("source", "unknown"),
        "has_final_review": "final_review" in data
    }), 200


# ================================================================
# 📋 PRE-BATCH INFO - v22.4: z n-gramami i proporcjami
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """
    v23.8: Dodaje:
    - N-gramy do wplecenia w batch
    - Proporcje main vs synonyms
    - Ostrzeżenie o nadużywaniu synonimów
    - Semantic gaps analysis
    - Running main_keyword ratio check
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    h2_structure = data.get("h2_structure", [])
    total_planned_batches = data.get("total_planned_batches", 4)
    main_keyword = data.get("main_keyword", data.get("topic", ""))
    main_keyword_synonyms = data.get("main_keyword_synonyms", [])
    s1_data = data.get("s1_data", {})
    
    current_batch_num = len(batches) + 1
    remaining_batches = max(1, total_planned_batches - len(batches))
    
    # ================================================================
    # v23.8: SEMANTIC GAPS ANALYSIS
    # ================================================================
    semantic_gaps = []
    full_text = ""
    try:
        from text_analyzer import analyze_semantic_coverage
        full_text = "\n\n".join([b.get("text", "") for b in batches])
        if full_text and keywords_state:
            keywords_list = [meta.get("keyword", "") for meta in keywords_state.values() if meta.get("keyword")]
            sem_result = analyze_semantic_coverage(full_text, keywords_list)
            semantic_gaps = sem_result.get("gaps", [])
    except Exception as e:
        print(f"[PRE_BATCH] Semantic analysis skipped: {e}")
    
    # ================================================================
    # ANALIZA MAIN vs SYNONYMS
    # ================================================================
    main_keyword_uses = 0
    synonym_uses = 0
    main_keyword_meta = None
    
    for rid, meta in keywords_state.items():
        if meta.get("is_main_keyword"):
            main_keyword_uses = meta.get("actual_uses", 0)
            main_keyword_meta = meta
        elif meta.get("is_synonym_of_main"):
            synonym_uses += meta.get("actual_uses", 0)
    
    total_main_and_synonyms = main_keyword_uses + synonym_uses
    main_ratio = main_keyword_uses / total_main_and_synonyms if total_main_and_synonyms > 0 else 1.0
    
    # v23.8: Early warning dla main_ratio
    ratio_warning = None
    if current_batch_num > 1 and main_ratio < 0.30:
        ratio_warning = f"Main keyword ratio {main_ratio:.0%} < 30%. Użyj więcej '{main_keyword}'!"
    
    # ================================================================
    # N-GRAMY DO WPLECIENIA W TYM BATCHU
    # ================================================================
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.4][:15]
    
    ngrams_per_batch = max(3, len(top_ngrams) // total_planned_batches)
    start_idx = (current_batch_num - 1) * ngrams_per_batch
    end_idx = min(start_idx + ngrams_per_batch + 2, len(top_ngrams))
    batch_ngrams = top_ngrams[start_idx:end_idx]
    
    # ================================================================
    # 📊 ANALIZA FRAZ Z PEŁNYM PLANEM
    # ================================================================
    keyword_plan = []
    critical_keywords = []
    high_priority = []
    normal_keywords = []
    low_priority = []
    locked_keywords = []
    exceeded_keywords = []
    extended_unused = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword")
        kw_type = meta.get("type", "BASIC")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        is_synonym = meta.get("is_synonym_of_main", False)
        
        remaining_to_max = max(0, target_max - actual)
        remaining_to_min = max(0, target_min - actual)
        
        # ================================================================
        # ⭐ v22.5: NOWA KONSERWATYWNA LOGIKA SUGGESTED
        # ================================================================
        # Bazuje na target_max - im niższy max, tym mniej forsujemy
        
        if target_max <= 2:
            # NISKOPRIORYTETOWE (max 1-2x w całym artykule)
            # Użyj raz i zapomnij - nie forsuj w każdym batchu
            max_per_batch = 1
            if actual >= target_min:
                suggested = 0  # Już mamy minimum, nie sugeruj więcej
            else:
                suggested = 1  # Potrzebujemy jeszcze 1
        elif target_max <= 6:
            # ŚREDNIE (max 3-6x w całym artykule)
            # Max 1-2 na batch, nie więcej
            max_per_batch = 2
            if remaining_to_max > 0 and remaining_batches > 0:
                suggested = min(2, max(1, remaining_to_min)) if remaining_to_min > 0 else 1
            else:
                suggested = 0
        elif target_max <= 15:
            # WYŻSZE (max 7-15x w całym artykule)
            # Max 2-3 na batch
            max_per_batch = 3
            if remaining_to_max > 0 and remaining_batches > 0:
                suggested = min(3, math.ceil(remaining_to_max / remaining_batches))
            else:
                suggested = 0
        else:
            # WYSOKIE (max 16+, np. "włos" 45x)
            # Tu możemy więcej - max 5-8 na batch
            max_per_batch = min(8, math.ceil(target_max / total_planned_batches))
            if remaining_to_max > 0 and remaining_batches > 0:
                suggested = min(max_per_batch, math.ceil(remaining_to_max / remaining_batches))
            else:
                suggested = 0
        
        # Jeśli nie osiągnęliśmy minimum - podnieś suggested
        if remaining_to_min > 0 and remaining_batches > 0:
            min_needed = math.ceil(remaining_to_min / remaining_batches)
            suggested = max(suggested, min(min_needed, max_per_batch))
        
        # ================================================================
        # PRIORYTET I REASON
        # ================================================================
        
        # ⭐ MAIN KEYWORD - najwyższy priorytet
        if is_main:
            if remaining_to_min > 0:
                priority = "CRITICAL"
                reason = f"🔴 FRAZA GŁÓWNA! Potrzeba {remaining_to_min}x do min"
                suggested = max(suggested, math.ceil(remaining_to_min / remaining_batches))
            else:
                priority = "HIGH"
                reason = f"🟠 FRAZA GŁÓWNA - używaj częściej niż synonimów!"
        # ⭐ Synonim nadużywany
        elif is_synonym and main_ratio < 0.3:
            priority = "LOW"
            reason = f"⚠️ SYNONIM - za dużo! Używaj '{main_keyword}'"
            suggested = 0
        # EXCEEDED
        elif actual > target_max:
            priority = "EXCEEDED"
            reason = f"❌ Już {actual}x (max {target_max}x) - NIE UŻYWAJ!"
            suggested = 0
        # LOCKED
        elif remaining_to_max == 0:
            priority = "LOCKED"
            reason = f"🔒 Max osiągnięty ({target_max}x)"
            suggested = 0
        # CRITICAL - ostatni batch, brakuje do min
        elif remaining_to_min > 0 and remaining_batches == 1:
            priority = "CRITICAL"
            reason = f"🔴 OSTATNI BATCH! Potrzeba {remaining_to_min}x"
            suggested = min(remaining_to_min, max_per_batch + 2)  # Pozwól na więcej w ostatnim
        # HIGH - UNDER (brakuje do minimum)
        elif remaining_to_min > 0:
            priority = "HIGH"
            reason = f"🟠 UNDER - brakuje {remaining_to_min}x"
        # EXTENDED nieużyte
        elif kw_type == "EXTENDED" and actual == 0:
            priority = "HIGH"
            reason = f"🟠 EXTENDED - wpleć naturalnie"
            suggested = 1
            extended_unused.append(keyword)
        # LOW - frazy z max 1-2, już użyte
        elif target_max <= 2 and actual >= target_min:
            priority = "LOW"
            reason = f"⚪ Użyte ({actual}x) - opcjonalne"
            suggested = 0
        # NORMAL - OK, w zakresie
        elif actual >= target_min and remaining_to_max > 0:
            priority = "NORMAL"
            reason = f"🟢 OK ({actual}/{target_min}-{target_max})"
            # Dla NORMAL z niskim max - nie sugeruj
            if target_max <= 6:
                suggested = 0
        # LOW - pozostałe
        else:
            priority = "LOW"
            reason = f"⚪ Opcjonalne ({actual}x)"
            suggested = 0
        
        kw_info = {
            "keyword": keyword,
            "type": kw_type,
            "priority": priority,
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "remaining_to_min": remaining_to_min,
            "remaining_to_max": remaining_to_max,
            "max_per_batch": max_per_batch,
            "suggested": suggested,
            "reason": reason,
            "is_main_keyword": is_main,
            "is_synonym_of_main": is_synonym
        }
        
        keyword_plan.append(kw_info)
        
        if priority == "EXCEEDED":
            exceeded_keywords.append(kw_info)
        elif priority == "LOCKED":
            locked_keywords.append(kw_info)
        elif priority == "CRITICAL":
            critical_keywords.append(kw_info)
        elif priority == "HIGH":
            high_priority.append(kw_info)
        elif priority == "NORMAL":
            normal_keywords.append(kw_info)
        else:
            low_priority.append(kw_info)
    
    priority_order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3, "LOCKED": 4, "EXCEEDED": 5}
    keyword_plan.sort(key=lambda x: priority_order.get(x["priority"], 99))
    
    # ================================================================
    # 📝 ANALIZA POPRZEDNICH BATCHÓW
    # ================================================================
    used_h2 = []
    used_h3 = []
    all_topics_covered = []
    last_sentences = ""
    
    for batch in batches:
        batch_text = batch.get("text", "")
        h2_in_batch = re.findall(r'<h2[^>]*>(.*?)</h2>', batch_text, re.IGNORECASE | re.DOTALL)
        h2_in_batch += re.findall(r'^h2:\s*(.+)$', batch_text, re.MULTILINE | re.IGNORECASE)
        h3_in_batch = re.findall(r'<h3[^>]*>(.*?)</h3>', batch_text, re.IGNORECASE | re.DOTALL)
        h3_in_batch += re.findall(r'^h3:\s*(.+)$', batch_text, re.MULTILINE | re.IGNORECASE)
        
        used_h2.extend([h.strip() for h in h2_in_batch])
        used_h3.extend([h.strip() for h in h3_in_batch])
        all_topics_covered.extend(h2_in_batch + h3_in_batch)
    
    if batches:
        last_batch_text = batches[-1].get("text", "")
        clean_last = re.sub(r'<[^>]+>', '', last_batch_text)
        clean_last = re.sub(r'^h[23]:\s*.+$', '', clean_last, flags=re.MULTILINE)
        sentences = re.split(r'[.!?]+', clean_last)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        if len(sentences) >= 2:
            last_sentences = ". ".join(sentences[-2:]) + "."
        elif sentences:
            last_sentences = sentences[-1] + "."
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    # ================================================================
    # 📝 GENERUJ PROMPT DLA GPT - v23.8 z SEMANTIC GAPS
    # ================================================================
    prompt_sections = []
    prompt_sections.append(f"📋 BATCH #{current_batch_num} z {total_planned_batches}")
    prompt_sections.append("")
    
    # v23.8: WARNING o ratio
    if ratio_warning:
        prompt_sections.append(f"⚠️ {ratio_warning}")
        prompt_sections.append("")
    
    # SEMANTIC GAPS - v23.8
    if semantic_gaps:
        prompt_sections.append("🔍 LUKI TEMATYCZNE (wypełnij!):")
        for gap in semantic_gaps[:4]:
            prompt_sections.append(f"  • {gap}")
        prompt_sections.append("")
    
    # FRAZA GŁÓWNA
    if main_keyword_meta:
        main_suggested = max(2, math.ceil(main_keyword_meta.get("target_min", 6) / total_planned_batches))
        prompt_sections.append("="*50)
        prompt_sections.append(f"🔴 FRAZA GŁÓWNA: \"{main_keyword}\"")
        prompt_sections.append(f"  → użyj {main_suggested}-{main_suggested+1}x w tym batchu")
        if main_ratio < 0.3:
            prompt_sections.append(f"  ⚠️ Za dużo synonimów! ({main_ratio:.0%})")
        prompt_sections.append("="*50)
        prompt_sections.append("")
    
    # CRITICAL (MUSISZ użyć) - tylko te z suggested > 0
    critical_to_show = [k for k in critical_keywords if k.get("suggested", 0) > 0 and not k.get("is_main_keyword")]
    if critical_to_show:
        prompt_sections.append("🔴 MUSISZ UŻYĆ:")
        for kw in critical_to_show[:5]:
            prompt_sections.append(f"  • {kw['keyword']}: {kw['suggested']}x")
        prompt_sections.append("")
    
    # HIGH PRIORITY - tylko te z suggested > 0
    high_to_show = [k for k in high_priority if k.get("suggested", 0) > 0 and not k.get("is_main_keyword")]
    if high_to_show:
        prompt_sections.append("🟠 WPLEĆ (priorytet):")
        for kw in high_to_show[:6]:
            prompt_sections.append(f"  • {kw['keyword']}: {kw['suggested']}x")
        prompt_sections.append("")
    
    # N-GRAMY - max 3
    if batch_ngrams:
        prompt_sections.append("📝 N-GRAMY (wpleć naturalnie):")
        for ngram in batch_ngrams[:3]:
            prompt_sections.append(f"  • \"{ngram}\"")
        prompt_sections.append("")
    
    # EXCEEDED + LOCKED - tylko ostrzeżenie
    blocked = exceeded_keywords + locked_keywords
    if blocked:
        blocked_names = [k['keyword'] for k in blocked[:5]]
        prompt_sections.append(f"❌ NIE UŻYWAJ: {', '.join(blocked_names)}")
        prompt_sections.append("")
    
    # H2 do napisania - BEZ sztywnych długości
    if remaining_h2:
        prompt_sections.append("✏️ H2 DO NAPISANIA:")
        for h2 in remaining_h2[:3]:
            prompt_sections.append(f"  • {h2}")
        prompt_sections.append("")
    
    # Poprzednie tematy - skrócone
    if all_topics_covered:
        prompt_sections.append(f"📖 NIE POWIELAJ: {', '.join(all_topics_covered[:4])}")
        prompt_sections.append("")
    
    # Kontynuacja
    if last_sentences:
        prompt_sections.append(f"🔗 KONTYNUUJ OD: \"{last_sentences[:80]}...\"")
        prompt_sections.append("")
    
    # ZASADY - v22.5: różnorodność struktury
    prompt_sections.append("="*50)
    prompt_sections.append("📝 STYL NATURALNY:")
    prompt_sections.append("  • Sekcje H2: różna długość (200-600 słów)")
    prompt_sections.append("  • Akapity: różna długość (40-150 słów)")
    prompt_sections.append("  • H3: tylko gdy NAPRAWDĘ potrzebne (max 2-3 na artykuł)")
    prompt_sections.append("  • Max 1 lista wypunktowana")
    prompt_sections.append("  • Format: h2: / h3:")
    prompt_sections.append("="*50)
    
    gpt_prompt = "\n".join(prompt_sections)
    
    # ================================================================
    # 📊 RESPONSE - v23.9 OPTYMALIZACJA
    # ================================================================
    
    # === BASIC keywords - PEŁNE DANE ===
    basic_keywords = []
    for kw in keyword_plan:
        if kw.get("type") == "BASIC":
            basic_keywords.append({
                "keyword": kw.get("keyword"),
                "actual": kw.get("actual", 0),
                "target": f"{kw.get('target_min', 0)}-{kw.get('target_max', 999)}",
                "remaining": kw.get("remaining_to_max", 0),
                "priority": kw.get("priority"),
                "suggested": kw.get("suggested", 0),
                "reason": kw.get("reason", "")
            })
    
    # === EXTENDED keywords - TYLKO NAZWY ===
    extended_keywords = [kw.get("keyword") for kw in keyword_plan if kw.get("type") == "EXTENDED"]
    
    # === BLOCKED - tylko nazwy ===
    blocked_names = [kw.get("keyword") for kw in locked_keywords + exceeded_keywords]
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "batch_number": current_batch_num,
        "total_planned_batches": total_planned_batches,
        "remaining_batches": remaining_batches,
        
        # Semantic gaps
        "semantic_gaps": semantic_gaps[:5],
        
        # Ratio warning
        "ratio_warning": ratio_warning,
        
        # Main keyword status
        "main_keyword_status": {
            "main_keyword": main_keyword,
            "main_uses": main_keyword_uses,
            "synonym_uses": synonym_uses,
            "main_ratio": round(main_ratio, 2),
            "warning": None if main_ratio >= 0.3 else f"Ratio {main_ratio:.0%} < 30%! Użyj więcej \'{main_keyword}\'"
        },
        
        # N-gramy
        "batch_ngrams": batch_ngrams,
        
        # v23.9: BASIC pełne, EXTENDED tylko nazwy
        "basic_keywords": basic_keywords,
        "extended_keywords": extended_keywords,
        "blocked_keywords": blocked_names,
        
        # H2
        "h2_remaining": remaining_h2,
        "h2_already_written": used_h2,
        
        # Context
        "last_sentences": last_sentences,
        
        # Summary
        "summary": {
            "basic_count": len(basic_keywords),
            "extended_count": len(extended_keywords),
            "blocked_count": len(blocked_names),
            "h2_remaining": len(remaining_h2),
            "semantic_gaps_count": len(semantic_gaps)
        },
        
        # Instructions
        "instructions": {
            "basic": "Użyj fraz BASIC wg priorytetów i suggested",
            "extended": "Wpleć naturalnie frazy EXTENDED gdy pasują",
            "blocked": "NIE używaj fraz z blocked_keywords",
            "main_keyword": f"\'{main_keyword}\' ≥30% użyć"
        }
    }), 200


# ================================================================
# ✏️ ADD BATCH - bez zmian
# ================================================================
@project_routes.post("/api/project/<project_id>/add_batch")
def add_batch_to_project(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    result = process_batch_in_firestore(project_id, batch_text, meta_trace)
    
    return jsonify(result), 200


# ================================================================
# 🔍 PREVIEW BATCH - v22.4: z walidacją list i H3
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """
    ⭐ v22.4: Dodaje walidację:
    - Liczba list wypunktowanych
    - Długość sekcji H3
    - Proporcja main vs synonyms
    - Pokrycie n-gramów
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    main_keyword = project_data.get("main_keyword", project_data.get("topic", ""))
    s1_data = project_data.get("s1_data", {})

    # Podstawowa prewalidacja
    report = unified_prevalidation(batch_text, keywords_state)
    
    warnings = report.get("warnings", [])
    errors = []
    
    # ⭐ v22.4: WALIDACJA LIST
    list_count = count_bullet_lists(batch_text)
    if list_count > 1:
        warnings.append({
            "type": "TOO_MANY_LISTS",
            "count": list_count,
            "max": 1,
            "message": f"Za dużo list ({list_count}). Max 1 na artykuł!"
        })
    
    # ⭐ v22.4: WALIDACJA H3
    h3_validation = validate_h3_length(batch_text, min_words=80)
    if h3_validation["issues"]:
        for issue in h3_validation["issues"]:
            warnings.append({
                "type": "H3_TOO_SHORT",
                "h3": issue["h3"],
                "word_count": issue["word_count"],
                "min": 80,
                "message": f"H3 '{issue['h3']}' za krótkie ({issue['word_count']} słów, min 80)"
            })
    
    # ⭐ v22.4: WALIDACJA MAIN vs SYNONYMS
    main_synonym_check = check_main_vs_synonyms_in_text(batch_text, main_keyword, keywords_state)
    if not main_synonym_check["valid"]:
        warnings.append({
            "type": "SYNONYM_OVERUSE",
            "main_count": main_synonym_check["main_count"],
            "synonym_total": main_synonym_check["synonym_total"],
            "ratio": main_synonym_check["main_ratio"],
            "message": main_synonym_check["warning"]
        })
    
    # ⭐ v22.4: WALIDACJA N-GRAMÓW
    ngrams = s1_data.get("ngrams", [])
    top_ngrams = [n.get("ngram", "") for n in ngrams if n.get("weight", 0) > 0.5][:10]
    ngram_check = check_ngram_coverage_in_text(batch_text, top_ngrams)
    if ngram_check["coverage"] < 0.5:
        warnings.append({
            "type": "LOW_NGRAM_COVERAGE",
            "coverage": ngram_check["coverage"],
            "missing": ngram_check["missing"][:3],
            "message": f"Niskie pokrycie n-gramów ({ngram_check['coverage']:.0%})"
        })
    
    # Określ status
    status = "OK"
    if errors:
        status = "ERROR"
    elif len(warnings) > 2:
        status = "WARN"
    
    return jsonify({
        "status": status,
        "semantic_score": report.get("semantic_score", 0),
        "density": report.get("density", 0),
        "warnings": warnings,
        "errors": errors,
        "validations": {
            "lists": {"count": list_count, "valid": list_count <= 1},
            "h3_length": h3_validation,
            "main_vs_synonyms": main_synonym_check,
            "ngram_coverage": ngram_check
        }
    }), 200


# ================================================================
# 🔧 HELPER FUNCTIONS - v22.4
# ================================================================
def count_bullet_lists(text: str) -> int:
    """Liczy bloki list wypunktowanych."""
    lines = text.split('\n')
    list_blocks = 0
    in_list = False
    
    for line in lines:
        is_bullet = bool(re.match(r'^\s*[-•*]\s+|^\s*\d+\.\s+', line.strip()))
        
        if is_bullet and not in_list:
            list_blocks += 1
            in_list = True
        elif not is_bullet and line.strip():
            in_list = False
    
    # HTML lists
    html_lists = len(re.findall(r'<ul>|<ol>', text, re.IGNORECASE))
    
    return list_blocks + html_lists


def validate_h3_length(text: str, min_words: int = 80) -> dict:
    """Sprawdza czy sekcje H3 mają minimalną długość."""
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    h3_matches = list(re.finditer(h3_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    issues = []
    sections = []
    
    for i, match in enumerate(h3_matches):
        h3_title = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = len(text)
        
        next_header = re.search(r'^h[23]:|<h[23]', text[start:], re.MULTILINE | re.IGNORECASE)
        if next_header:
            end = start + next_header.start()
        
        section_text = text[start:end].strip()
        section_text = re.sub(r'<[^>]+>', '', section_text)
        word_count = len(section_text.split())
        
        sections.append({"h3": h3_title, "word_count": word_count})
        
        if word_count < min_words:
            issues.append({
                "h3": h3_title,
                "word_count": word_count,
                "min_required": min_words,
                "deficit": min_words - word_count
            })
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "sections": sections,
        "total_h3": len(h3_matches)
    }


def check_main_vs_synonyms_in_text(text: str, main_keyword: str, keywords_state: dict) -> dict:
    """Sprawdza proporcję frazy głównej vs synonimy w tekście."""
    text_lower = text.lower()
    
    main_count = len(re.findall(rf"\b{re.escape(main_keyword.lower())}\b", text_lower))
    
    synonym_counts = {}
    synonym_total = 0
    
    for rid, meta in keywords_state.items():
        if meta.get("is_synonym_of_main"):
            keyword = meta.get("keyword", "").lower()
            count = len(re.findall(rf"\b{re.escape(keyword)}\b", text_lower))
            if count > 0:
                synonym_counts[meta.get("keyword")] = count
                synonym_total += count
    
    total = main_count + synonym_total
    main_ratio = main_count / total if total > 0 else 1.0
    
    return {
        "main_keyword": main_keyword,
        "main_count": main_count,
        "synonyms": synonym_counts,
        "synonym_total": synonym_total,
        "total": total,
        "main_ratio": round(main_ratio, 2),
        "valid": main_ratio >= 0.3,
        "warning": f"Za dużo synonimów! '{main_keyword}' ma tylko {main_ratio:.0%}. Zamień synonimy." if main_ratio < 0.3 else None
    }


def check_ngram_coverage_in_text(text: str, required_ngrams: list) -> dict:
    """Sprawdza pokrycie n-gramów w tekście."""
    text_lower = text.lower()
    used = []
    missing = []
    
    for ngram in required_ngrams:
        if ngram and ngram.lower() in text_lower:
            used.append(ngram)
        elif ngram:
            missing.append(ngram)
    
    coverage = len(used) / len(required_ngrams) if required_ngrams else 1.0
    
    return {
        "coverage": round(coverage, 2),
        "used": used,
        "missing": missing,
        "valid": coverage >= 0.6
    }


# ================================================================
# 🆕 AUTO-CORRECT ENDPOINT - v22.4
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """
    Automatyczna korekta batcha.
    ⭐ v22.4: Auto-save do Firestore
    """
    data = request.get_json() or {}
    batch_text = data.get("text") or data.get("batch_text")
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    
    # ⭐ v22.5: Lepsze pobieranie tekstu
    if not batch_text:
        batches = project_data.get("batches", [])
        if batches:
            # Szukaj ostatniego batcha z tekstem
            for batch in reversed(batches):
                if batch.get("text"):
                    batch_text = batch.get("text")
                    break
    
    # Jeśli nadal brak - scal wszystkie batche
    if not batch_text:
        batches = project_data.get("batches", [])
        all_texts = [b.get("text", "") for b in batches if b.get("text")]
        if all_texts:
            batch_text = "\n\n".join(all_texts)
    
    if not batch_text:
        return jsonify({
            "error": "No text provided",
            "hint": "Brak zapisanych batchy w projekcie lub wszystkie są puste",
            "batches_count": len(project_data.get("batches", []))
        }), 400
    
    keywords_state = project_data.get("keywords_state", {})
    
    under_keywords = []
    over_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        min_target = meta.get("target_min", 0)
        max_target = meta.get("target_max", 999)
        keyword = meta.get("keyword", "")
        
        if actual < min_target:
            under_keywords.append({
                "keyword": keyword,
                "missing": min_target - actual,
                "current": actual,
                "target_min": min_target
            })
        elif actual > max_target:
            over_keywords.append({
                "keyword": keyword,
                "excess": actual - max_target,
                "current": actual,
                "target_max": max_target
            })
    
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "corrected_text": batch_text
        }), 200
    
    if not GEMINI_API_KEY:
        return jsonify({"status": "ERROR", "error": "Gemini API not configured"}), 500
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        correction_instructions = []
        if under_keywords:
            under_list = "\n".join([f"  - '{kw['keyword']}': Dodaj {kw['missing']}x" for kw in under_keywords[:10]])
            correction_instructions.append(f"DODAJ te frazy:\n{under_list}")
        
        if over_keywords:
            over_list = "\n".join([f"  - '{kw['keyword']}': Usuń {kw['excess']}x" for kw in over_keywords[:5]])
            correction_instructions.append(f"USUŃ nadmiar:\n{over_list}")
        
        correction_prompt = f"""
Popraw tekst SEO:

{chr(10).join(correction_instructions)}

ZASADY:
1. Zachowaj h2: i h3:
2. Dodawaj frazy naturalnie
3. Zachowaj styl

TEKST:
{batch_text[:12000]}

Zwróć TYLKO poprawiony tekst.
"""
        
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        corrected_text = re.sub(r'^```(?:html)?\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        # ⭐ v22.4: Auto-save
        batches = project_data.get("batches", [])
        auto_saved = False
        new_metrics = {}
        
        if batches:
            batches[-1]["text"] = corrected_text
            batches[-1]["auto_corrected"] = True
            new_metrics = unified_prevalidation(corrected_text, keywords_state)
            batches[-1]["burstiness"] = new_metrics.get("burstiness", 0)
            batches[-1]["density"] = new_metrics.get("density", 0)
            doc_ref.update({"batches": batches})
            auto_saved = True
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "auto_saved": auto_saved,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords]
        }), 200
        
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ================================================================
# 📦 EXPORT
# ================================================================
@project_routes.get("/api/project/<project_id>/export")
def export_project_data(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404

    data = doc.to_dict()
    batches = data.get("batches", [])
    full_text = "\n\n".join(b.get("text", "") for b in batches)
    
    def convert_markers_to_html(text):
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith('h2:'):
                title = stripped[3:].strip()
                result.append(f'<h2>{title}</h2>')
            elif stripped.lower().startswith('h3:'):
                title = stripped[3:].strip()
                result.append(f'<h3>{title}</h3>')
            else:
                result.append(line)
        return '\n'.join(result)
    
    article_html = convert_markers_to_html(full_text)

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_text": full_text,
        "article_html": article_html,
        "batch_count": len(batches),
        "version": "v22.4"
    }), 200


# ================================================================
# 🔄 ALIASES
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    return auto_correct_batch(project_id)


@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    return preview_batch(project_id)
