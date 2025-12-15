import uuid
import re
import os
import json
import math
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING] ⚠️ GEMINI_API_KEY not set - LSI enrichment fallback mode")

# spaCy model
try:
    nlp = spacy.load("pl_core_news_md")
    print("[INIT] ✅ spaCy pl_core_news_md loaded")
except OSError:
    from spacy.cli import download
    print("⚠️ Downloading pl_core_news_md fallback...")
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)

# ⭐ GEMINI MODEL - centralnie zdefiniowany
GEMINI_MODEL = "gemini-2.5-flash"


# ================================================================
# 🧠 H2 SUGGESTIONS (Gemini-powered)
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    """
    Generuje sugestie H2 używając Gemini na podstawie:
    - topic/main_keyword
    - wzorców H2 z konkurencji (serp_h2_patterns)
    - target keywords
    
    Zwraca listę maksymalnie 6 H2 (hard limit zgodny z seo_rules.json).
    Wstęp (intro) NIE jest H2 - to osobny element bez nagłówka.
    
    ⚠️ WAŻNE: To są tylko PROPOZYCJE. User musi podać SWOJE H2,
    które zostaną połączone z propozycjami w finalną strukturę.
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
            "model": "fallback",
            "action_required": "USER_H2_INPUT_NEEDED"
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
        
        prompt = f"""
Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu SEO o temacie: "{topic}"

WAŻNE: Artykuł będzie miał WSTĘP (bez nagłówka H2) + {target_count} sekcji H2.

{competitor_context}
{keywords_context}

ZASADY:
1. Wygeneruj DOKŁADNIE {target_count} H2 (nie więcej, nie mniej)
2. Każdy H2 powinien mieć 6-12 słów
3. Minimum 50% H2 powinno być w formie pytania (Jak...?, Dlaczego...?, Kiedy...?, Co...?)
4. Maksimum 30% H2 może zawierać główne słowo kluczowe
5. NIE używaj ogólnikowych tytułów jak: "Wstęp", "Podsumowanie", "Zakończenie", "FAQ"
6. H2 powinny tworzyć logiczną strukturę artykułu
7. Uwzględnij intencję wyszukiwania (informacyjna/transakcyjna)
8. Wpleć naturalnie frazy kluczowe gdzie to możliwe

FORMAT ODPOWIEDZI:
Zwróć TYLKO listę {target_count} H2, każdy w nowej linii, bez numeracji ani punktorów.
"""
        
        print(f"[H2_SUGGESTIONS] Generating {target_count} H2 for: {topic}")
        response = model.generate_content(prompt)
        
        raw_suggestions = response.text.strip().split('\n')
        suggestions = [
            h2.strip().lstrip('•-–—0123456789.). ')
            for h2 in raw_suggestions 
            if h2.strip() and len(h2.strip()) > 5
        ][:target_count]
        
        print(f"[H2_SUGGESTIONS] ✅ Generated {len(suggestions)} H2 suggestions")
        
        return jsonify({
            "status": "OK",
            "suggestions": suggestions,
            "topic": topic,
            "model": GEMINI_MODEL,
            "count": len(suggestions),
            "action_required": "USER_H2_INPUT_NEEDED",
            "message": "To są PROPOZYCJE. Teraz podaj SWOJE H2, które chcesz wpleść, a system połączy je w finalną strukturę."
        }), 200
        
    except Exception as e:
        print(f"[H2_SUGGESTIONS] ❌ Error: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "suggestions": []
        }), 500


# ================================================================
# 🧱 PROJECT CREATE
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    """
    Tworzy nowy projekt SEO w Firestore.
    
    ⭐ NOWA LOGIKA LIMITÓW:
    - GPT widzi: target_min + 1 (ostrzegawczy limit)
    - Backend liczy do: target_max (rzeczywisty limit)
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
    
    # ⭐ NOWE: Liczba planowanych batchów (domyślnie = liczba H2 / 2, min 2, max 6)
    total_planned_batches = data.get("total_planned_batches")
    if not total_planned_batches:
        total_planned_batches = max(2, min(6, math.ceil(len(h2_structure) / 2))) if h2_structure else 4

    firestore_keywords = {}
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
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "search_term_exact": term.lower(),
            "search_lemma": search_lemma,
            "target_min": min_val,
            "target_max": max_val,
            # ⭐ NOWE: display_limit to co widzi GPT (min+1)
            "display_limit": min_val + 1,
            "actual_uses": 0,
            "status": "UNDER",
            "type": item.get("type", "BASIC").upper(),
            "remaining_max": max_val,
            "optimal_target": max_val
        }

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    project_data = {
        "topic": topic,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "batches_plan": [],
        "total_batches": 0,
        "total_planned_batches": total_planned_batches,  # ⭐ NOWE
        "target_length": target_length,
        "source": source,
        "version": "v22.1",
        "manual_mode": False if source == "n8n-brajen-workflow" else True,
        # ⭐ NOWE: format output
        "output_format": "clean_text_with_headers"
    }
    doc_ref.set(project_data)
    
    print(f"[PROJECT] ✅ Created project {doc_ref.id}: {topic} ({len(firestore_keywords)} keywords, {total_planned_batches} planned batches)")

    return jsonify({
        "status": "CREATED",
        "project_id": doc_ref.id,
        "topic": topic,
        "keywords_count": len(firestore_keywords),
        "keywords": len(firestore_keywords),
        "h2_sections": len(h2_structure),
        "total_planned_batches": total_planned_batches,  # ⭐ NOWE
        "target_length": target_length,
        "source": source
    }), 201


# ================================================================
# 📊 GET PROJECT STATUS - z info o LOCKED frazach
# ================================================================
@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    """
    Zwraca aktualny status projektu z informacją o LOCKED frazach.
    
    ⭐ NOWE:
    - locked_keywords: lista fraz które osiągnęły limit
    - display_limits: limity które widzi GPT (min+1)
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    
    keyword_summary = []
    locked_keywords = []
    near_limit_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        remaining = max(0, target_max - actual)
        display_limit = target_min + 1  # Co widzi GPT
        
        kw_info = {
            "keyword": meta.get("keyword"),
            "type": meta.get("type", "BASIC"),
            "actual": actual,
            "display_limit": display_limit,  # ⭐ GPT widzi to
            "target_max": target_max,  # Backend limit
            "status": meta.get("status"),
            "remaining_max": remaining
        }
        keyword_summary.append(kw_info)
        
        # ⭐ Zbierz LOCKED frazy
        if remaining == 0:
            locked_keywords.append({
                "keyword": meta.get("keyword"),
                "message": f"🔒 LOCKED: '{meta.get('keyword')}' osiągnęło limit {target_max}x. Użyj SYNONIMÓW!"
            })
        elif remaining <= 3:
            near_limit_keywords.append({
                "keyword": meta.get("keyword"),
                "remaining": remaining,
                "message": f"⚠️ NEAR LIMIT: '{meta.get('keyword')}' - zostało tylko {remaining}x"
            })
    
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": len(batches),
        "keywords_count": len(keywords_state),
        "keywords": keyword_summary,
        # ⭐ NOWE - wyraźne info o blokadach
        "locked_keywords": locked_keywords,
        "near_limit_keywords": near_limit_keywords,
        "warnings_before_batch": locked_keywords + near_limit_keywords,
        "source": data.get("source", "unknown"),
        "has_final_review": "final_review" in data
    }), 200


# ================================================================
# 📋 PRE-BATCH INFO - PEŁNA OPTYMALIZACJA TARGET=MAX
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    """
    Zwraca PEŁNY PLAN przed napisaniem batcha:
    - Rozkład fraz w batchach (A)
    - Priorytetyzacja UNDER (B)
    - Automatyczne balansowanie (C)
    - Smart display_limit (D)
    - Max per batch (E) - zapobiega przeoptymalizacji
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords_state = data.get("keywords_state", {})
    batches = data.get("batches", [])
    h2_structure = data.get("h2_structure", [])
    total_planned_batches = data.get("total_planned_batches", 4)  # Domyślnie 4
    
    current_batch_num = len(batches) + 1
    remaining_batches = max(1, total_planned_batches - len(batches))
    
    # ================================================================
    # 📊 ANALIZA FRAZ Z PEŁNYM PLANEM
    # ================================================================
    keyword_plan = []
    critical_keywords = []  # UNDER + ostatni batch
    high_priority = []      # UNDER
    normal_keywords = []    # OK, można użyć
    low_priority = []       # Już >= min
    locked_keywords = []    # = max, nie używaj
    exceeded_keywords = []  # > max
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword")
        kw_type = meta.get("type", "BASIC")
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 0)
        target_max = meta.get("target_max", 999)
        
        remaining_to_max = max(0, target_max - actual)
        remaining_to_min = max(0, target_min - actual)
        
        # ⭐ SMART CALCULATIONS
        # Max per batch = równomierny rozkład (zapobiega stuffing)
        max_per_batch = max(1, math.ceil(target_max / total_planned_batches))
        
        # Suggested = ile użyć w tym batchu dla równomiernego rozkładu
        if remaining_to_max > 0 and remaining_batches > 0:
            suggested = min(
                math.ceil(remaining_to_max / remaining_batches),
                max_per_batch
            )
        else:
            suggested = 0
        
        # Jeśli UNDER - zwiększ suggested żeby nadrobić
        if remaining_to_min > 0:
            min_needed_per_batch = math.ceil(remaining_to_min / remaining_batches)
            suggested = max(suggested, min_needed_per_batch)
            # Ale nie więcej niż max_per_batch
            suggested = min(suggested, max_per_batch + 1)  # +1 dla UNDER
        
        # ⭐ PRIORYTET
        if actual > target_max:
            priority = "EXCEEDED"
            reason = f"❌ Już {actual}x (max {target_max}x) - NIE UŻYWAJ!"
            suggested = 0
        elif remaining_to_max == 0:
            priority = "LOCKED"
            reason = f"🔒 Osiągnięto max {target_max}x - użyj SYNONIMÓW"
            suggested = 0
        elif remaining_to_min > 0 and remaining_batches == 1:
            priority = "CRITICAL"
            reason = f"🔴 OSTATNI BATCH! Potrzeba {remaining_to_min}x do min!"
            suggested = remaining_to_min
        elif remaining_to_min > 0:
            priority = "HIGH"
            reason = f"🟠 UNDER - brakuje {remaining_to_min}x (cel: {target_min}x)"
        elif actual >= target_min and remaining_to_max > 0:
            priority = "NORMAL"
            reason = f"🟢 OK ({actual}/{target_min}-{target_max}) - sugerowane {suggested}x"
        else:
            priority = "LOW"
            reason = f"⚪ Wystarczy ({actual}x >= min {target_min}x)"
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
            "reason": reason
        }
        
        keyword_plan.append(kw_info)
        
        # Kategoryzuj
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
    
    # Sortuj keyword_plan po priorytecie
    priority_order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3, "LOCKED": 4, "EXCEEDED": 5}
    keyword_plan.sort(key=lambda x: priority_order.get(x["priority"], 99))
    
    # ================================================================
    # 📝 ANALIZA POPRZEDNICH BATCHÓW
    # ================================================================
    used_h2 = []
    used_h3 = []
    all_topics_covered = []
    last_sentences = ""
    
    for i, batch in enumerate(batches):
        batch_text = batch.get("text", "")
        
        h2_in_batch = re.findall(r'<h2[^>]*>(.*?)</h2>', batch_text, re.IGNORECASE | re.DOTALL)
        h3_in_batch = re.findall(r'<h3[^>]*>(.*?)</h3>', batch_text, re.IGNORECASE | re.DOTALL)
        
        used_h2.extend([h.strip() for h in h2_in_batch])
        used_h3.extend([h.strip() for h in h3_in_batch])
        all_topics_covered.extend(h2_in_batch + h3_in_batch)
    
    if batches:
        last_batch_text = batches[-1].get("text", "")
        clean_last = re.sub(r'<[^>]+>', '', last_batch_text)
        sentences = re.split(r'[.!?]+', clean_last)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            last_sentences = ". ".join(sentences[-2:]) + "."
        elif sentences:
            last_sentences = sentences[-1] + "."
    
    remaining_h2 = [h2 for h2 in h2_structure if h2 not in used_h2]
    
    # ================================================================
    # 📝 GENERUJ PROMPT DLA GPT
    # ================================================================
    prompt_sections = []
    prompt_sections.append(f"📋 BATCH #{current_batch_num} z {total_planned_batches} (zostało: {remaining_batches})")
    prompt_sections.append("")
    
    # CRITICAL
    if critical_keywords:
        prompt_sections.append("🔴 CRITICAL (MUSISZ użyć - ostatni batch!):")
        for kw in critical_keywords:
            prompt_sections.append(f"  • {kw['keyword']}: UŻYJ {kw['suggested']}x!")
    
    # EXCEEDED
    if exceeded_keywords:
        prompt_sections.append("\n❌ EXCEEDED (NIE UŻYWAJ!):")
        for kw in exceeded_keywords:
            prompt_sections.append(f"  • {kw['keyword']}")
    
    # LOCKED
    if locked_keywords:
        prompt_sections.append("\n🔒 LOCKED (użyj SYNONIMÓW):")
        for kw in locked_keywords:
            prompt_sections.append(f"  • {kw['keyword']}")
    
    # HIGH (UNDER)
    if high_priority:
        prompt_sections.append("\n🟠 PRIORYTET (UNDER - wpleć!):")
        for kw in high_priority:
            prompt_sections.append(f"  • {kw['keyword']}: użyj {kw['suggested']}x (brakuje {kw['remaining_to_min']}x)")
    
    # NORMAL
    if normal_keywords:
        prompt_sections.append("\n🟢 NORMALNE (sugerowane użycie):")
        for kw in normal_keywords[:5]:  # Max 5
            prompt_sections.append(f"  • {kw['keyword']}: max {kw['max_per_batch']}x (sugerowane: {kw['suggested']}x)")
    
    # LOW
    if low_priority:
        prompt_sections.append("\n⚪ OPCJONALNE (już OK):")
        for kw in low_priority[:3]:  # Max 3
            prompt_sections.append(f"  • {kw['keyword']}")
    
    # Poprzednie tematy
    if all_topics_covered:
        prompt_sections.append("\n\n📖 NIE POWIELAJ:")
        for topic in all_topics_covered[:8]:
            prompt_sections.append(f"  • {topic}")
    
    # Ostatnie zdania
    if last_sentences:
        prompt_sections.append(f"\n\n🔗 KONTYNUUJ OD:")
        prompt_sections.append(f"  \"{last_sentences[:150]}...\"" if len(last_sentences) > 150 else f"  \"{last_sentences}\"")
    
    # H2 do napisania
    if remaining_h2:
        prompt_sections.append(f"\n\n✏️ H2 DO NAPISANIA:")
        for h2 in remaining_h2[:3]:
            prompt_sections.append(f"  • {h2}")
    
    gpt_prompt = "\n".join(prompt_sections)
    
    # ================================================================
    # 📊 PODSUMOWANIE
    # ================================================================
    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "batch_number": current_batch_num,
        "total_planned_batches": total_planned_batches,
        "remaining_batches": remaining_batches,
        
        # ⭐ PEŁNY PLAN FRAZ (posortowany po priorytecie)
        "keyword_plan": keyword_plan,
        
        # Kategoryzowane
        "critical_keywords": critical_keywords,
        "high_priority_keywords": high_priority,
        "normal_keywords": normal_keywords,
        "low_priority_keywords": low_priority,
        "locked_keywords": locked_keywords,
        "exceeded_keywords": exceeded_keywords,
        
        # Struktura
        "h2_structure": h2_structure,
        "h2_already_written": used_h2,
        "h2_remaining": remaining_h2,
        "topics_already_covered": all_topics_covered,
        "last_sentences": last_sentences,
        
        # Podsumowanie
        "summary": {
            "critical_count": len(critical_keywords),
            "high_priority_count": len(high_priority),
            "normal_count": len(normal_keywords),
            "low_count": len(low_priority),
            "locked_count": len(locked_keywords),
            "exceeded_count": len(exceeded_keywords),
            "h2_written": len(used_h2),
            "h2_remaining": len(remaining_h2)
        },
        
        # ⭐ GOTOWY PROMPT
        "gpt_prompt": gpt_prompt,
        
        "instructions": {
            "critical": "MUSISZ użyć tych fraz - ostatnia szansa!",
            "high": "Priorytet - wpleć w batch",
            "normal": "Użyj sugerowaną ilość (max_per_batch)",
            "low": "Opcjonalne - już wystarczy",
            "locked": "NIE UŻYWAJ - użyj synonimów",
            "exceeded": "BŁĄD - za dużo użyć!",
            "max_per_batch": "Nie przekraczaj max_per_batch żeby uniknąć stuffingu"
        }
    }), 200


# ================================================================
# ✏️ ADD BATCH
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
    
    rhythm = result.get("meta", {}).get("paragraph_rhythm", "Unknown")
    result["batch_text_snippet"] = batch_text[:50] + "..."
    result["paragraph_rhythm"] = rhythm

    return jsonify(result), 200


# ================================================================
# 🔍 MANUAL CORRECTION ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/manual_correct")
def manual_correct_batch(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    corrected_text = data.get("text") or data.get("batch_text") or data.get("corrected_text")
    if not corrected_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})
    forced = data.get("forced", False)

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
# 🆕 AUTO-CORRECT ENDPOINT
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    """
    Automatyczna korekta batcha używając Gemini.
    Jeśli nie podano tekstu, pobiera ostatni batch z projektu.
    """
    data = request.get_json() or {}

    batch_text = data.get("text") or data.get("batch_text") or data.get("corrected_text")
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    
    # ⭐ NOWE: Jeśli brak tekstu, pobierz ostatni batch
    if not batch_text:
        batches = project_data.get("batches", [])
        if batches:
            batch_text = batches[-1].get("text", "")
            print(f"[AUTO_CORRECT] 📥 Pobrano ostatni batch z Firestore ({len(batch_text)} znaków)")
        
    if not batch_text:
        return jsonify({"error": "No text provided and no batches in project"}), 400
    
    keywords_state = project_data.get("keywords_state", {})
    
    under_keywords = []
    over_keywords = []
    
    for rid, meta in keywords_state.items():
        actual = meta.get("actual_uses", 0)
        min_target = meta.get("target_min", 0)
        max_target = meta.get("target_max", 999)
        keyword = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC")
        
        if actual < min_target:
            under_keywords.append({
                "keyword": keyword,
                "missing": min_target - actual,
                "type": kw_type,
                "current": actual,
                "target_min": min_target
            })
        elif actual > max_target:
            over_keywords.append({
                "keyword": keyword,
                "excess": actual - max_target,
                "type": kw_type,
                "current": actual,
                "target_max": max_target
            })
    
    if not under_keywords and not over_keywords:
        return jsonify({
            "status": "NO_CORRECTIONS_NEEDED",
            "message": "All keywords within target ranges",
            "corrected_text": batch_text,
            "keyword_report": {"under": [], "over": []}
        }), 200
    
    if not GEMINI_API_KEY:
        return jsonify({
            "status": "ERROR",
            "error": "Gemini API key not configured",
            "keyword_report": {"under": under_keywords, "over": over_keywords}
        }), 500
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        correction_instructions = []
        
        if under_keywords:
            under_list = "\n".join([
                f"  - '{kw['keyword']}': Dodaj {kw['missing']}× (obecnie {kw['current']}/{kw['target_min']})"
                for kw in under_keywords
            ])
            correction_instructions.append(f"DODAJ te frazy naturalnie:\n{under_list}")
        
        if over_keywords:
            over_list = "\n".join([
                f"  - '{kw['keyword']}': Usuń {kw['excess']}× (obecnie {kw['current']}, max {kw['target_max']})"
                for kw in over_keywords
            ])
            correction_instructions.append(f"USUŃ nadmiar tych fraz:\n{over_list}")
        
        correction_prompt = f"""
Popraw poniższy tekst SEO według instrukcji:

{chr(10).join(correction_instructions)}

ZASADY:
1. Zachowaj nagłówki <h2> i <h3>
2. Reszta tekstu ma być CZYSTYM TEKSTEM (bez <p>, bez <strong>, bez list)
3. Dodawaj frazy naturalnie w kontekście
4. Usuwaj frazy poprzez parafrazy lub synonimy
5. Zachowaj profesjonalny, formalny styl
6. Ta sama fraza BASIC nie może występować częściej niż 1x na 3 zdania

TEKST DO POPRAWY:
---
{batch_text[:10000]}
---

Zwróć TYLKO poprawiony tekst, bez żadnych komentarzy.
"""
        
        print(f"[AUTO_CORRECT] Wysyłam do Gemini: {len(under_keywords)} UNDER, {len(over_keywords)} OVER")
        response = model.generate_content(correction_prompt)
        corrected_text = response.text.strip()
        
        # Usuń ewentualne markdown/html wrappery
        corrected_text = re.sub(r'^```(?:html)?\n?', '', corrected_text)
        corrected_text = re.sub(r'\n?```$', '', corrected_text)
        
        print(f"[AUTO_CORRECT] ✅ Gemini zwrócił poprawiony tekst ({len(corrected_text)} znaków)")
        
        return jsonify({
            "status": "AUTO_CORRECTED",
            "corrected_text": corrected_text,
            "added_keywords": [kw["keyword"] for kw in under_keywords],
            "removed_keywords": [kw["keyword"] for kw in over_keywords],
            "keyword_report": {"under": under_keywords, "over": over_keywords},
            "correction_summary": f"Dodano {len(under_keywords)} fraz, usunięto nadmiar {len(over_keywords)} fraz"
        }), 200
        
    except Exception as e:
        print(f"[AUTO_CORRECT] ❌ Błąd Gemini: {e}")
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "keyword_report": {"under": under_keywords, "over": over_keywords}
        }), 500


# ================================================================
# 🧠 UNIFIED PRE-VALIDATION
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_all_checks")
def preview_all_checks(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text = data.get("text") or data.get("batch_text")
    if not text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})

    report = unified_prevalidation(text, keywords_state)

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
# 🆕 FORCE APPROVE
# ================================================================
@project_routes.post("/api/project/<project_id>/force_approve_batch")
def force_approve_batch(project_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    batch_text = data.get("text") or data.get("batch_text")
    if not batch_text:
        return jsonify({"error": "Field 'text' or 'batch_text' is required"}), 400

    meta_trace = data.get("meta_trace", {})

    print("[FORCE APPROVE] User requested forced save.")
    return manual_correct_batch(project_id)


# ================================================================
# 📦 EXPORT
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

    return jsonify({
        "status": "EXPORT_READY",
        "topic": data.get("topic"),
        "article_text": full_text,  # ⭐ Czysty tekst, nie HTML
        "batch_count": len(batches),
        "version": "v22.1"
    }), 200


# ================================================================
# 🔄 ALIAS: auto_correct_keywords
# ================================================================
@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    """Alias dla auto_correct - kompatybilność z OpenAPI schema."""
    return auto_correct_batch(project_id)
