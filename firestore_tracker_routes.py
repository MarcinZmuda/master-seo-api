"""
SEO Content Tracker Routes - v15.5 FIXED
DYNAMIC TARGET SYSTEM INTEGRATED WITH SIMPLIFIED GPT INSTRUCTIONS
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime

tracker_routes = Blueprint("tracker_routes", __name__)

# ============================================================================
# HELPER FUNCTIONS - Keyword Counting
# ============================================================================

def count_hybrid_occurrences(text, keyword, include_headings=True):
    """
    Count keyword occurrences (exact + lemma matching).
    """
    if not text or not keyword:
        return 0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Simple exact count
    count = text_lower.count(keyword_lower)
    
    return count


# ============================================================================
# VALIDATION FUNCTIONS (HARDENED)
# ============================================================================

def validate_html_and_headers(text, expected_h2_count=2):
    """
    Validate HTML structure AND check for banned headers.
    """
    # 1. Check for Markdown
    if "##" in text or "###" in text:
        return {
            "valid": False,
            "error": "❌ Markdown detected (## or ###). Use HTML tags: <h2>, <h3>",
            "suggestion": "Replace ## with <h2> and ### with <h3>"
        }
    
    # 2. Extract H2 content
    h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL)
    
    # 3. Check H2 count
    if len(h2_matches) < expected_h2_count:
        return {
            "valid": False,
            "error": f"❌ Expected {expected_h2_count} H2 sections, found {len(h2_matches)}"
        }
    
    # 4. ⭐ CHECK BANNED HEADERS (Hard Validation)
    banned_headers = [
        "wstęp", "podsumowanie", "wprowadzenie", "zakończenie", 
        "konkluzja", "introduction", "summary", "wnioski"
    ]
    
    for h2_text in h2_matches:
        clean_h2 = h2_text.lower().strip()
        # Check exact match or "Wstęp do..."
        for ban in banned_headers:
            if clean_h2 == ban or clean_h2.startswith(f"{ban} ") or clean_h2.endswith(f" {ban}"):
                return {
                    "valid": False,
                    "error": f"❌ ZAKAZANY NAGŁÓWEK: '{h2_text}'. Nie używaj ogólników typu '{ban}'. H2 musi być opisowy i zawierać słowa kluczowe."
                }
    
    return {"valid": True}


def validate_all_densities(text, keywords_state):
    """Check per-keyword density."""
    word_count = len(text.split())
    if word_count == 0:
        return {"valid": False, "error": "Empty text"}
    
    critical = []
    
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        
        density = (actual_uses / word_count) * 100
        
        if density > 3.0:
            critical.append({
                "keyword": kw,
                "density": round(density, 2),
                "actual_uses": actual_uses,
                "limit": 3.0
            })
    
    if critical:
        return {
            "valid": False,
            "critical": critical,
            "fix_suggestion": "Reduce keyword usage or use synonyms",
            "error": "❌ Keyword stuffing detected (>3.0%)"
        }
    
    return {"valid": True}


def calculate_total_keyword_density(text, keywords_state):
    """Check total keyword density."""
    word_count = len(text.split())
    if word_count == 0:
        return {"valid": False, "error": "Empty text"}
    
    total_uses = sum(meta.get("actual_uses", 0) for meta in keywords_state.values())
    total_density = (total_uses / word_count) * 100
    
    if total_density > 8.0:
        return {
            "valid": False,
            "error": f"❌ Total keyword density: {total_density:.1f}% (max 8.0%)",
            "total_density": round(total_density, 2)
        }
    
    return {"valid": True, "total_density": round(total_density, 2)}


def calculate_burstiness(text):
    """Calculate sentence length variance."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    
    mean_length = sum(lengths) / len(lengths)
    variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    
    if mean_length == 0:
        return 0.0
    
    burstiness = (std_dev / mean_length) * 10
    return round(burstiness, 1)


def validate_section_structure(text, batch_num):
    """
    Validate alternating section structure:
    Section 1 and 2 must alternate LONG/SHORT.
    """
    sections = re.split(r'<h2[^>]*>.*?</h2>', text, flags=re.IGNORECASE | re.DOTALL)
    sections = [s.strip() for s in sections if s.strip()]
    
    if len(sections) < 2:
        return {"valid": True}  # Not enough sections to validate
    
    section_lengths = [len(s.split()) for s in sections[:2]]  # Only check first 2
    
    def categorize(word_count):
        if word_count >= 250:
            return "LONG"
        elif word_count >= 150:
            return "SHORT"
        else:
            return "TOO_SHORT"
    
    sec1_type = categorize(section_lengths[0])
    sec2_type = categorize(section_lengths[1])
    
    # Check if any too short
    if "TOO_SHORT" in [sec1_type, sec2_type]:
        return {
            "valid": False,
            "error": f"❌ Section too short (<150 words)",
            "section_lengths": section_lengths
        }
    
    # Check if both are same type (not alternating)
    if sec1_type == sec2_type:
        return {
            "valid": False,
            "error": f"❌ Sections not alternating: both {sec1_type}",
            "section_lengths": section_lengths,
            "suggestion": "Section 1 LONG → Section 2 SHORT, or vice versa"
        }
    
    return {"valid": True, "pattern": f"{sec1_type} → {sec2_type}"}


# ============================================================================
# v15.4 DYNAMIC TARGET CALCULATION
# ============================================================================

def calculate_dynamic_target(
    keyword_meta,
    batch_number,
    batch_length,
    total_batches,
    remaining_batches_info
):
    """
    Calculate dynamic target for keyword in current batch.
    Used internally to generate GPT instructions.
    """
    
    # 1. Calculate remaining average
    total_target = keyword_meta.get("target", 0)
    if total_target == 0:
        min_val = keyword_meta.get("target_min", 1)
        max_val = keyword_meta.get("target_max", 999)
        total_target = round((min_val + max_val) / 2)
    
    already_used = keyword_meta.get("actual_uses", 0)
    remaining_target = max(0, total_target - already_used)
    
    remaining_batches = max(1, total_batches - batch_number + 1)
    
    avg_per_batch = remaining_target / remaining_batches
    
    # 2. Batch bonus (decreasing)
    batch_bonus = max(0, 4 - batch_number)
    
    # 3. Length factor
    if remaining_batches_info and len(remaining_batches_info) > 0:
        remaining_lengths = [batch_length] + [b.get("word_count", 300) for b in remaining_batches_info]
        avg_remaining_length = sum(remaining_lengths) / len(remaining_lengths)
    else:
        avg_remaining_length = batch_length or 300
    
    length_factor = batch_length / avg_remaining_length if avg_remaining_length > 0 else 1.0
    
    # 4. Calculate target
    raw_target = (avg_per_batch + batch_bonus) * length_factor
    target = max(1, round(raw_target))
    
    # 5. Tolerance
    if remaining_batches == 1:
        tolerance = 1
    else:
        tolerance = 2
    
    min_acceptable = max(0, target - tolerance) # Can be 0 if target is low
    if remaining_target > 0 and min_acceptable == 0 and remaining_batches == 1:
        min_acceptable = 1 # Force use in last batch if remaining
        
    max_acceptable = target + tolerance
    
    return {
        "target": target,
        "min": min_acceptable,
        "max": max_acceptable,
        "tolerance": tolerance,
        "explanation": "Calculated by backend"
    }


def validate_against_dynamic_target(actual_uses, target_info, keyword):
    min_val = target_info["min"]
    max_val = target_info["max"]
    target = target_info["target"]
    
    if actual_uses < min_val:
        return {
            "valid": False,
            "status": "BELOW_TARGET",
            "message": f"❌ {keyword}: {actual_uses}× too low (need {min_val}-{max_val}×)",
            "diff": min_val - actual_uses
        }
    
    if actual_uses > max_val:
        return {
            "valid": False,
            "status": "ABOVE_TARGET",
            "message": f"❌ {keyword}: {actual_uses}× too high (limit {max_val}×)",
            "diff": actual_uses - max_val
        }
    
    return {
        "valid": True,
        "status": "OK",
        "message": f"✅ {keyword}: OK"
    }


# ============================================================================
# LEGACY FUNCTION (for project_routes.py compatibility)
# ============================================================================

def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None) -> dict:
    """Legacy function wrapper."""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        count_in_batch = count_hybrid_occurrences(batch_text, kw, include_headings=True)
        meta["actual_uses"] = meta.get("actual_uses", 0) + count_in_batch
    
    burstiness_score = calculate_burstiness(batch_text)
    
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness_score
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    doc_ref.set(project_data)
    
    return {"status": "BATCH_SAVED", "message": "Batch saved (legacy)", "status_code": 200}


# ============================================================================
# ROUTES
# ============================================================================

@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    """
    Preview batch with keyword counting (no save).
    """
    data = request.get_json(force=True) or {}
    batch_text = data.get("batch_text", "")
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    updated_state = {}
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        count_in_batch = count_hybrid_occurrences(batch_text, kw, include_headings=True)
        previous_uses = meta.get("actual_uses", 0)
        
        updated_state[row_id] = {
            **meta,
            "actual_uses": previous_uses + count_in_batch,
            "used_in_current_batch": count_in_batch
        }
    
    return jsonify({
        "status": "PREVIEW",
        "corrected_text": batch_text,
        "keywords_state": updated_state
    }), 200


@tracker_routes.post("/api/project/<project_id>/approve_batch")
def approve_batch(project_id):
    """
    Approve batch with VALIDATIONS and generate HUMAN-READABLE instructions for GPT.
    """
    data = request.get_json(force=True) or {}
    corrected_text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    keywords_state = data.get("keywords_state", {})
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    current_batch_num = len(project_data.get("batches", [])) + 1
    batches_plan = project_data.get("batches_plan", []) or [{"word_count": 500}] * 5
    
    # ===== 1. HARD VALIDATIONS =====
    
    # HTML & Headers
    h2_count_expected = meta_trace.get("h2_count", 2)
    html_check = validate_html_and_headers(corrected_text, h2_count_expected)
    if not html_check["valid"]:
        return jsonify(html_check), 400
    
    # Densities
    density_check = validate_all_densities(corrected_text, keywords_state)
    if not density_check["valid"]:
        return jsonify({"error": density_check["error"], "details": density_check.get("critical")}), 400
    
    total_den_check = calculate_total_keyword_density(corrected_text, keywords_state)
    if not total_den_check["valid"]:
        return jsonify(total_den_check), 400
    
    # Burstiness
    burstiness_score = calculate_burstiness(corrected_text)
    if burstiness_score < 6.0:
        return jsonify({
            "error": f"❌ Zbyt mała różnorodność zdań (Burstiness: {burstiness_score}). Minimum 6.0.",
            "suggestion": "Przeplataj bardzo krótkie zdania (3-5 słów) z długimi (25+ słów)."
        }), 400
    
    # Section Structure
    structure_check = validate_section_structure(corrected_text, current_batch_num)
    if not structure_check["valid"]:
        return jsonify(structure_check), 400
    
    # Dynamic Targets (Audit only - block if extreme deviation)
    batch_length = len(corrected_text.split())
    total_batches = len(batches_plan)
    remaining_info_current = batches_plan[current_batch_num:] if current_batch_num < len(batches_plan) else []
    
    dynamic_validations = []
    
    for row_id, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() != "BASIC": continue
        
        target_info = calculate_dynamic_target(meta, current_batch_num, batch_length, total_batches, remaining_info_current)
        validation = validate_against_dynamic_target(meta.get("used_in_current_batch", 0), target_info, meta["keyword"])
        dynamic_validations.append(validation)
        
        # Optional: You can uncomment this to BLOCK on dynamic targets
        # if not validation["valid"]: return jsonify({"error": validation["message"]}), 400

    # ===== 2. SAVE BATCH =====
    
    batch_entry = {
        "batch_number": current_batch_num,
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "validations": {
            "burstiness": burstiness_score,
            "structure": structure_check.get("pattern", "OK"),
            "html": "OK"
        },
        "dynamic_validations": dynamic_validations
    }
    
    if "batches" not in project_data:
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    doc_ref.set(project_data)
    
    # ===== 3. GENERATE INSTRUCTIONS FOR NEXT BATCH =====
    
    next_batch_num = current_batch_num + 1
    article_complete = current_batch_num >= total_batches
    
    gpt_instruction = ""
    next_targets_list = []
    
    if not article_complete:
        next_info = batches_plan[next_batch_num - 1] if next_batch_num - 1 < len(batches_plan) else {"word_count": 400}
        target_len = next_info.get("word_count", 400)
        
        gpt_instruction = f"""✅ BATCH {current_batch_num} ZATWIERDZONY.
        
⏩ ZADANIE NA BATCH {next_batch_num}:
1. DŁUGOŚĆ: Celujemy w ok. {target_len} słów.
2. STRUKTURA: Pamiętaj o naprzemienności (LONG <-> SHORT).
3. SŁOWA KLUCZOWE (Obowiązkowe użycie w tym fragmencie):
"""
        next_remaining_info = batches_plan[next_batch_num:] if next_batch_num < len(batches_plan) else []
        
        for row_id, meta in keywords_state.items():
            if meta.get("type") == "BASIC":
                next_calc = calculate_dynamic_target(meta, next_batch_num, target_len, total_batches, next_remaining_info)
                t_min, t_max = next_calc["min"], next_calc["max"]
                
                if t_max > 0:
                    gpt_instruction += f"   - '{meta['keyword']}': użyj {t_min}-{t_max} razy\n"
                    next_targets_list.append({"keyword": meta['keyword'], "range": f"{t_min}-{t_max}"})

        # Add Extended keywords reminder
        unused_extended = [m['keyword'] for m in keywords_state.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) == 0]
        if unused_extended:
            gpt_instruction += "\n4. FRAZY WSPIERAJĄCE (Użyj przynajmniej 2-3 z listy):\n"
            for kw in unused_extended[:5]:
                gpt_instruction += f"   - {kw}\n"
    
    else:
        gpt_instruction = "ARTYKUŁ UKOŃCZONY. Przejdź do walidacji."

    # ===== 4. RESPONSE =====
    return jsonify({
        "status": "BATCH_SAVED",
        "batch_number": current_batch_num,
        "article_complete": article_complete,
        "validations": batch_entry["validations"],
        "gpt_instruction": gpt_instruction,  # ⭐ GPT reads this text
        "next_batch_targets": next_targets_list, # For debug/frontend display
        "progress": f"Batch {current_batch_num}/{total_batches} complete"
    }), 200


@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    """
    Final validation before export.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    incomplete_basic = []
    for row_id, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() != "BASIC": continue
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        if actual < target_min:
            incomplete_basic.append({"keyword": meta["keyword"], "actual": actual, "min": target_min})
            
    # Check EXTENDED (100% required)
    total_extended = 0
    used_extended = 0
    incomplete_extended = []
    
    for row_id, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() == "EXTENDED":
            total_extended += 1
            if meta.get("actual_uses", 0) >= 1:
                used_extended += 1
            else:
                incomplete_extended.append({"keyword": meta["keyword"]})
                
    extended_coverage = (used_extended / total_extended * 100) if total_extended > 0 else 100
    
    if extended_coverage < 100:
        return jsonify({
            "error": f"❌ EXTENDED coverage: {extended_coverage:.0f}% (need 100%)",
            "incomplete_extended": incomplete_extended
        }), 400
        
    if incomplete_basic:
        return jsonify({"error": "❌ BASIC keywords below minimum", "incomplete": incomplete_basic}), 400
        
    return jsonify({"status": "ARTICLE_READY", "message": "✅ All validations passed"}), 200


@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    """
    Export complete article.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    return jsonify({
        "status": "EXPORTED",
        "full_article": full_text,
        "batch_count": len(batches),
        "word_count": len(full_text.split())
    }), 200
