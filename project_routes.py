import uuid
import re
import os
import math
import random
import spacy
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from firestore_tracker_routes import process_batch_in_firestore
import google.generativeai as genai
from seo_optimizer import unified_prevalidation

# Konfiguracja API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Domyślny model (najnowszy flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Ładowanie SpaCy (fallback)
try:
    nlp = spacy.load("pl_core_news_md")
except OSError:
    from spacy.cli import download
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

project_routes = Blueprint("project_routes", __name__)

# ================================================================
# 🧠 H2 SUGGESTIONS (Generowanie struktury)
# ================================================================
@project_routes.post("/api/project/s1_h2_suggestions")
def generate_h2_suggestions():
    data = request.get_json() or {}
    topic = data.get("topic") or data.get("main_keyword", "")
    target_count = min(data.get("target_count", 6), 8)
    
    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"suggestions": [f"Co to jest {topic}?", f"Zalety {topic}", "Podsumowanie"]}), 200

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
        Jesteś ekspertem SEO. Wygeneruj DOKŁADNIE {target_count} nagłówków H2 dla artykułu: "{topic}".
        
        ZASADY:
        1. Min 50% to pytania.
        2. Max 30% zawiera frazę kluczową wprost.
        3. Brak "Wstęp", "Zakończenie".
        4. Intencja: Informacyjno-Ekspercka.
        
        Zwróć TYLKO listę H2, po jednym w linii.
        """
        response = model.generate_content(prompt)
        suggestions = [line.strip().lstrip("-•1234567890. ") for line in response.text.splitlines() if line.strip()]
        
        return jsonify({
            "status": "OK",
            "suggestions": suggestions[:target_count],
            "model": GEMINI_MODEL
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# 🧱 PROJECT CREATE
# ================================================================
@project_routes.post("/api/project/create")
def create_project():
    data = request.get_json() or {}
    topic = data.get("topic", "").strip()
    h2_structure = data.get("h2_structure", [])
    raw_keywords = data.get("keywords_list") or []
    
    # Automatyczne planowanie liczby batchy
    total_batches = data.get("total_planned_batches")
    if not total_batches:
        total_batches = max(2, min(6, math.ceil(len(h2_structure) / 2))) if h2_structure else 4

    firestore_keywords = {}
    for item in raw_keywords:
        term = item.get("term") or item.get("keyword")
        if not term: continue
        
        row_id = str(uuid.uuid4())
        min_val = item.get("min", 1)
        max_val = item.get("max", 5)
        
        firestore_keywords[row_id] = {
            "keyword": term,
            "target_min": min_val,
            "target_max": max_val,
            "actual_uses": 0,
            "status": "UNDER",
            "type": item.get("type", "BASIC").upper()
        }

    db = firestore.client()
    doc_ref = db.collection("seo_projects").document()
    doc_ref.set({
        "topic": topic,
        "h2_structure": h2_structure,
        "keywords_state": firestore_keywords,
        "created_at": firestore.SERVER_TIMESTAMP,
        "batches": [],
        "total_planned_batches": total_batches,
        "version": "v22.1"
    })
    
    return jsonify({"project_id": doc_ref.id, "status": "CREATED"}), 201

# ================================================================
# 📋 PRE-BATCH INFO (Smart Balancing + Anti-Stuffing Fix)
# ================================================================
@project_routes.get("/api/project/<project_id>/pre_batch_info")
def get_pre_batch_info(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    h2_structure = data.get("h2_structure", [])
    total_planned = data.get("total_planned_batches", 4)
    
    current_batch_num = len(batches) + 1
    remaining_batches = max(1, total_planned - len(batches))
    
    # --- 1. KEYWORD ANALYSIS & ROTATION ---
    keyword_plan = []
    critical_kws, high_kws, locked_kws, normal_kws = [], [], [], []
    
    for _, meta in keywords.items():
        kw = meta["keyword"]
        actual = meta.get("actual_uses", 0)
        t_min = meta.get("target_min", 1)
        t_max = meta.get("target_max", 5)
        
        rem_max = max(0, t_max - actual)
        rem_min = max(0, t_min - actual)
        
        # Max per batch: restrykcyjny (max 1 jeśli blisko limitu)
        max_per_batch = max(1, math.ceil(t_max / total_planned))
        if rem_max <= 2:
            max_per_batch = 1
        
        # Suggested logic
        suggested = 0
        if rem_min > 0:
            suggested = math.ceil(rem_min / remaining_batches)
        elif rem_max > 0 and remaining_batches > 0:
            suggested = 0 if actual >= t_min else 1
        
        # PRIORITY ASSIGNMENT
        priority = "LOW"
        if actual > t_max: priority = "EXCEEDED"
        elif rem_max == 0: priority = "LOCKED"
        elif rem_min > 0 and remaining_batches == 1: priority = "CRITICAL"
        elif rem_min > 0: priority = "HIGH"
        elif actual >= t_min: priority = "NORMAL"
        
        plan_entry = {
            "keyword": kw, "priority": priority, "actual": actual,
            "target_min": t_min, "target_max": t_max,
            "suggested": suggested, "max_per_batch": max_per_batch
        }
        keyword_plan.append(plan_entry)
        
        if priority == "CRITICAL": critical_kws.append(plan_entry)
        elif priority == "HIGH": high_kws.append(plan_entry)
        elif priority == "LOCKED": locked_kws.append(plan_entry)
        elif priority == "NORMAL": normal_kws.append(plan_entry)

    # --- 2. ANTI-STUFFING FILTER (TO JEST NOWOŚĆ) ---
    # Sortujemy NORMAL po tym ile brakuje do MAX (im więcej brakuje, tym wyższy priorytet)
    normal_kws.sort(key=lambda x: (x['target_max'] - x['actual']), reverse=True)
    
    # Wybieramy max 3 frazy NORMAL do tego batcha. Resztę ukrywamy przed GPT.
    # To zapobiega sytuacji, gdzie model widzi 15 fraz "OK" i próbuje użyć wszystkich naraz.
    selected_normal_kws = normal_kws[:3]

    # --- 3. CONTEXT ---
    used_h2 = []
    last_sentences = ""
    if batches:
        for b in batches:
            used_h2.extend(re.findall(r'<h2[^>]*>(.*?)</h2>', b.get("text", ""), re.IGNORECASE))
        clean_last = re.sub(r'<[^>]+>', '', batches[-1].get("text", ""))
        sents = [s.strip() for s in clean_last.split('.') if s.strip()]
        last_sentences = ". ".join(sents[-2:]) + "." if len(sents) >= 2 else ""

    h2_remaining = [h for h in h2_structure if h not in used_h2]

    # --- 4. PROMPT CONSTRUCTION ---
    prompt_lines = [
        f"📝 PISZESZ BATCH {current_batch_num} z {total_planned}. Pozostało batchy: {remaining_batches}.",
        f"KONTYNUACJA TEKSTU: '...{last_sentences}'" if last_sentences else "",
        "\n🛑 ZADANIE: Napisz kolejne sekcje H2/H3 z listy:",
        *[f"- {h}" for h in h2_remaining[:2]], 
        "\n⚡ WYTYCZNE SŁÓW KLUCZOWYCH (BEZWZGLĘDNE):"
    ]
    
    if critical_kws:
        prompt_lines.append("🔴 CRITICAL (MUSISZ UŻYĆ - to ostatnia szansa!):")
        for k in critical_kws: 
            prompt_lines.append(f"  - {k['keyword']} (użyj naturalnie min. {k['suggested']} raz)")
        
    if high_kws:
        prompt_lines.append("🟠 PRIORYTET (Brakuje do minimum):")
        for k in high_kws: 
            prompt_lines.append(f"  - {k['keyword']} (spróbuj wpleść 1 raz)")
            
    if selected_normal_kws:
        prompt_lines.append("🟢 DODATKOWE (Tylko jeśli pasują do kontekstu):")
        for k in selected_normal_kws: 
            prompt_lines.append(f"  - {k['keyword']} (max {k['max_per_batch']} raz, nie na siłę)")

    if locked_kws:
        prompt_lines.append("❌ ZABRONIONE (Limit osiągnięty - użyj synonimów):")
        blocked_list = [k['keyword'] for k in locked_kws[:5]]
        prompt_lines.append(f"  - {', '.join(blocked_list)}")

    prompt_lines.append("\n⚠️ UWAGA: Nie wymieniaj słów po przecinku. Nie upychaj ich (keyword stuffing). Tekst musi brzmieć naturalnie dla człowieka.")

    return jsonify({
        "batch_number": current_batch_num,
        "gpt_prompt": "\n".join(prompt_lines),
        "keyword_plan": keyword_plan,
        "h2_remaining": h2_remaining,
        "critical_keywords": critical_kws,
        "high_priority_keywords": high_kws
    }), 200

# ================================================================
# 🔄 ACTIONS: PREVIEW, APPROVE, EXPORT
# ================================================================
@project_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch_route(project_id):
    data = request.get_json(force=True)
    return jsonify(process_batch_in_firestore(project_id, data.get("batch_text"), forced=False, preview_only=True))

@project_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch_route(project_id):
    data = request.get_json(force=True)
    return jsonify(process_batch_in_firestore(project_id, data.get("batch_text"), forced=data.get("forced", False)))

@project_routes.get("/api/project/<project_id>/export")
def export_project(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    
    batches = doc.to_dict().get("batches", [])
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    return jsonify({"article_text": full_text, "status": "EXPORTED"}), 200

@project_routes.get("/api/project/<project_id>/status")
def get_project_status(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Project not found"}), 404
    
    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    
    keyword_summary = []
    for _, meta in keywords.items():
        keyword_summary.append({
            "keyword": meta.get("keyword"),
            "actual": meta.get("actual_uses", 0),
            "target_min": meta.get("target_min"),
            "target_max": meta.get("target_max"),
            "status": meta.get("status")
        })
    
    return jsonify({
        "project_id": project_id,
        "total_batches": len(batches),
        "total_planned_batches": data.get("total_planned_batches"),
        "keywords": keyword_summary
    }), 200

# Aliasy dla kompatybilności
@project_routes.post("/api/project/<project_id>/auto_correct")
def auto_correct_batch(project_id):
    # Prosta implementacja placeholder lub przekierowanie
    # W pełnej wersji tutaj byłaby logika korekty, ale dla uproszczenia
    # skupiamy się na głównym flow.
    return jsonify({"status": "NOT_IMPLEMENTED_IN_LIGHT_VERSION"}), 501

@project_routes.post("/api/project/<project_id>/auto_correct_keywords")
def auto_correct_keywords_alias(project_id):
    return auto_correct_batch(project_id)
