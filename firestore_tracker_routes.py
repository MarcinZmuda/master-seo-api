"""
SEO Content Tracker Routes - v15.8
Zmiany:
- Burstiness: Wyłączono blokowanie (Warning only / Info)
- Keyword Max: Włączono TWARDĄ blokadę (Error 400 przy przekroczeniu target_max)
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
import spacy

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INIT SPACY (LG for Polish) ---
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[TRACKER] ✅ Załadowano model pl_core_news_lg")
except OSError:
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

# ============================================================================
# 1. ROBUST COUNTING (EXACT + LEMMA)
# ============================================================================
def count_robust(doc, keyword_meta):
    """
    Liczy max(exact, lemma).
    """
    if not doc: return 0
    kw_exact = keyword_meta.get("keyword", "").lower().strip()
    
    # Lemma lookup
    kw_lemma = keyword_meta.get("search_lemma", "")
    if not kw_lemma:
        kw_doc = nlp(kw_exact)
        kw_lemma = " ".join([t.lemma_.lower() for t in kw_doc if t.is_alpha])
    
    # 1. Exact count
    count_exact = doc.text.lower().count(kw_exact)
    
    # 2. Lemma count reconstruction
    doc_lemma_str = " ".join([t.lemma_.lower() for t in doc if t.is_alpha])
    count_lemma = doc_lemma_str.count(kw_lemma)
    
    return max(count_exact, count_lemma)

# ============================================================================
# 2. VALIDATIONS
# ============================================================================
def validate_structure(text, expected_h2_count=2):
    # Markdown Check
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Użyto Markdown (##). Wymagany czysty HTML (<h2>)."}
    
    # H2 Count
    h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL)
    if len(h2_matches) < expected_h2_count:
        return {"valid": False, "error": f"❌ Za mało H2. Jest {len(h2_matches)}, wymagane {expected_h2_count}."}
    
    # Banned Headers Check
    banned = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie", "wnioski", "konkluzja", "introduction"]
    for h2 in h2_matches:
        clean = h2.lower().strip()
        for ban in banned:
            if ban == clean or clean.startswith(f"{ban} ") or clean.endswith(f" {ban}"):
                return {"valid": False, "error": f"❌ ZAKAZANY NAGŁÓWEK: '{h2}'. H2 musi być opisowy i zawierać słowa kluczowe SEO!"}
    
    return {"valid": True}

def calculate_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3: return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    if mean == 0: return 0.0
    return round((math.sqrt(variance) / mean) * 10, 1)

def validate_quality(text, keywords_state, current_doc):
    # 1. Burstiness (INFO ONLY - NO BLOCK)
    burstiness = calculate_burstiness(text)
    # Ignorujemy wynik burstiness, nie zwracamy błędu.
    
    # 2. Density Check (Stuffing Guard - Local)
    word_count = len(text.split())
    if word_count > 0:
        for row_id, meta in keywords_state.items():
            count = count_robust(current_doc, meta)
            density = (count / word_count) * 100
            if density > 4.0: # Podniesiony limit lokalny, globalny jest ważniejszy
                return {"valid": False, "error": f"❌ Keyword Stuffing: '{meta['keyword']}' ma {density:.1f}% w tym fragmencie (max 4.0%)."}
    
    return {"valid": True}

# ============================================================================
# 3. DYNAMIC INSTRUCTION GENERATOR
# ============================================================================
def generate_gpt_instruction(current_batch, total_batches, keywords_state, batch_len_target):
    remaining_batches = max(1, total_batches - current_batch)
    
    instruction = f"""✅ BATCH {current_batch} ZATWIERDZONY.
⏩ INSTRUKCJA NA BATCH {current_batch + 1}:

1. CEL: Napisz ok. {batch_len_target} słów.
2. STRUKTURA: Naprzemiennie LONG (250w+) i SHORT (150w).
3. OBOWIĄZKOWE SŁOWA KLUCZOWE (Użyj w tym fragmencie):
"""
    targets = []
    for meta in keywords_state.values():
        if meta.get("type") == "BASIC":
            total_needed = meta.get("target_max", 0)
            current = meta.get("actual_uses", 0)
            remaining_uses = max(0, total_needed - current)
            
            if remaining_uses > 0:
                per_batch = math.ceil(remaining_uses / remaining_batches)
                target = per_batch
                if remaining_batches == 1:
                    min_needed = max(0, meta.get("target_min", 1) - current)
                    target = max(target, min_needed)

                if target > 0:
                    targets.append(f"   - '{meta['keyword']}': użyj ok. {target}x")
    
    instruction += "\n".join(targets[:6])
    
    unused_ext = [m['keyword'] for m in keywords_state.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) == 0]
    if unused_ext:
        instruction += "\n\n4. FRAZY WSPIERAJĄCE (Wpleć 2-3 z listy):\n" + "\n".join([f"   - {k}" for k in unused_ext[:5]])
        
    return instruction

# ============================================================================
# 4. LEGACY SUPPORT
# ============================================================================
def process_batch_in_firestore(project_id, batch_text, meta_trace=None):
    # Wrapper dla kompatybilności z project_routes
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found", "status_code": 404}
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    doc_nlp = nlp(batch_text)
    
    for row_id, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        keywords_state[row_id] = meta
    
    batch_entry = {
        "text": batch_text, "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": calculate_burstiness(batch_text)
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    return {"status": "BATCH_SAVED", "message": "Legacy OK", "status_code": 200}

# ============================================================================
# ROUTES
# ============================================================================
@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True) or {}
    text = data.get("batch_text", "")
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    doc_nlp = nlp(text)
    
    updated = {}
    for rid, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        current_total = meta.get("actual_uses", 0)
        updated[rid] = {
            **meta, 
            "actual_uses": current_total + count, 
            "used_in_current_batch": count
        }
        
    return jsonify({"status": "PREVIEW", "corrected_text": text, "keywords_state": updated}), 200

@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    data = request.get_json(force=True) or {}
    text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    current_batch = len(project_data.get("batches", [])) + 1
    plan = project_data.get("batches_plan", []) or [{"word_count": 500}] * 5
    
    # 1. Update Counts
    doc_nlp = nlp(text)
    keywords_state = project_data.get("keywords_state", {})
    
    for rid, meta in keywords_state.items():
        count = count_robust(doc_nlp, meta)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count
        keywords_state[rid] = meta

    # 2. VALIDATIONS
    
    # A. Structure
    struct_check = validate_structure(text, meta_trace.get("h2_count", 2))
    if not struct_check["valid"]: return jsonify(struct_check), 400
    
    # B. Quality (Burstiness removed as blocker)
    qual_check = validate_quality(text, keywords_state, doc_nlp)
    if not qual_check["valid"]: return jsonify(qual_check), 400

    # C. ⭐ HARD MAX LIMIT CHECK (NEW)
    # Sprawdzamy czy którykolwiek keyword nie przekroczył TOTAL MAX
    for rid, meta in keywords_state.items():
        if meta.get("type") == "BASIC":
            actual = meta.get("actual_uses", 0)
            maximum = meta.get("target_max", 999)
            if actual > maximum:
                return jsonify({
                    "error": f"❌ PRZEKROCZONO LIMIT: '{meta['keyword']}' użyto łącznie {actual}x. Maksymalny limit to {maximum}x. Usuń {actual - maximum} wystąpień z tego batcha."
                }), 400

    # 3. SAVE
    batch_entry = {
        "batch_number": current_batch,
        "text": text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "validations": "OK"
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    # 4. NEXT INSTRUCTION
    is_complete = current_batch >= len(plan)
    next_len = plan[current_batch]["word_count"] if not is_complete else 0
    instruction = generate_gpt_instruction(current_batch, len(plan), keywords_state, next_len) if not is_complete else "ARTYKUŁ UKOŃCZONY."

    return jsonify({
        "status": "BATCH_SAVED",
        "batch_number": current_batch,
        "article_complete": is_complete,
        "gpt_instruction": instruction
    }), 200

# (Walidacja i eksport - bez zmian)
@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    db = firestore.client()
    kw = db.collection("seo_projects").document(project_id).get().to_dict().get("keywords_state", {})
    ext_missing = [m['keyword'] for m in kw.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) == 0]
    
    if ext_missing:
        return jsonify({"error": f"❌ Brakuje fraz EXTENDED ({len(ext_missing)}).", "missing": ext_missing}), 400
        
    return jsonify({"status": "ARTICLE_READY"}), 200

@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    data = firestore.client().collection("seo_projects").document(project_id).get().to_dict()
    full_text = "\n\n".join([b.get("text", "") for b in data.get("batches", [])])
    return jsonify({"status": "EXPORTED", "full_article": full_text}), 200
