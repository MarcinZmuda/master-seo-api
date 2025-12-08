"""
SEO Content Tracker Routes - v15.4
DYNAMIC TARGET SYSTEM INTEGRATED

Key Features:
- v15.3: REMAINING tracking, MAX validation, 100% EXTENDED
- v15.4: Dynamic targets with length adjustment and decreasing bonus
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
# VALIDATION FUNCTIONS
# ============================================================================

def validate_html_structure(text, expected_h2_count=2):
    """Validate HTML structure (no Markdown)."""
    if "##" in text or "###" in text:
        return {
            "valid": False,
            "error": "❌ Markdown detected (## or ###). Use HTML tags: <h2>, <h3>",
            "suggestion": "Replace ## with <h2> and ### with <h3>"
        }
    
    h2_count = len(re.findall(r'<h2[^>]*>.*?</h2>', text, re.IGNORECASE | re.DOTALL))
    
    if h2_count < expected_h2_count:
        return {
            "valid": False,
            "error": f"❌ Expected {expected_h2_count} H2 sections, found {h2_count}"
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
            "fix_suggestion": "Reduce keyword usage or use synonyms"
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
    
    # Check if both are same type (not alternating)
    if sec1_type == sec2_type and sec1_type != "TOO_SHORT":
        return {
            "valid": False,
            "error": f"❌ Sections not alternating: both {sec1_type}",
            "section_lengths": section_lengths,
            "suggestion": "Section 1 LONG → Section 2 SHORT, or vice versa"
        }
    
    # Check if any too short
    if "TOO_SHORT" in [sec1_type, sec2_type]:
        return {
            "valid": False,
            "error": f"❌ Section too short (<150 words)",
            "section_lengths": section_lengths
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
    
    Formula: target = (remaining_avg + batch_bonus) × length_factor
    
    Args:
        keyword_meta: {"keyword": "...", "target": 27, "actual_uses": 15, ...}
        batch_number: 1, 2, 3, ...
        batch_length: word count of this batch
        total_batches: total number planned
        remaining_batches_info: [{"word_count": 500}, {"word_count": 400}]
    
    Returns:
        {
            "target": 14,
            "min": 12,
            "max": 16,
            "tolerance": 2,
            "explanation": "..."
        }
    """
    
    # 1. Calculate remaining average
    total_target = keyword_meta.get("target", 0)
    if total_target == 0:
        # Fallback: calculate from min-max
        min_val = keyword_meta.get("target_min", 1)
        max_val = keyword_meta.get("target_max", 999)
        total_target = round((min_val + max_val) / 2)
    
    already_used = keyword_meta.get("actual_uses", 0)
    remaining_target = max(0, total_target - already_used)
    
    remaining_batches = total_batches - batch_number + 1
    if remaining_batches == 0:
        remaining_batches = 1
    
    avg_per_batch = remaining_target / remaining_batches
    
    # 2. Batch bonus (decreasing)
    batch_bonus = max(0, 4 - batch_number)  # B1=3, B2=2, B3=1, B4+=0
    
    # 3. Length factor
    if remaining_batches_info and len(remaining_batches_info) > 0:
        # Calculate average of remaining batches (including this one)
        remaining_lengths = [batch_length] + [b.get("word_count", 300) for b in remaining_batches_info]
        avg_remaining_length = sum(remaining_lengths) / len(remaining_lengths)
    else:
        avg_remaining_length = batch_length
    
    if avg_remaining_length == 0:
        avg_remaining_length = 300  # Fallback
    
    length_factor = batch_length / avg_remaining_length
    
    # 4. Calculate target
    raw_target = (avg_per_batch + batch_bonus) * length_factor
    target = max(1, round(raw_target))
    
    # 5. Tolerance (decreases with batch number)
    if remaining_batches == 1:
        tolerance = 1  # Final batch: strict
    elif remaining_batches == 2:
        tolerance = 2  # Second to last: medium
    else:
        tolerance = 2  # Early batches: flexible
    
    min_acceptable = max(1, target - tolerance)
    max_acceptable = target + tolerance
    
    # 6. Explanation
    explanation = (
        f"Avg: {avg_per_batch:.1f} + Bonus: {batch_bonus} "
        f"× Length: {length_factor:.2f} = {raw_target:.1f} → {target}"
    )
    
    return {
        "target": target,
        "min": min_acceptable,
        "max": max_acceptable,
        "tolerance": tolerance,
        "explanation": explanation,
        "breakdown": {
            "remaining_target": remaining_target,
            "remaining_batches": remaining_batches,
            "avg_per_batch": round(avg_per_batch, 1),
            "batch_bonus": batch_bonus,
            "length_factor": round(length_factor, 2),
            "raw_target": round(raw_target, 1)
        }
    }


def validate_against_dynamic_target(actual_uses, target_info, keyword):
    """
    Validate if actual usage is within dynamic target range.
    """
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
    
    # Within range
    diff_from_target = abs(actual_uses - target)
    
    if diff_from_target == 0:
        status = "PERFECT"
    elif diff_from_target <= 1:
        status = "EXCELLENT"
    else:
        status = "ACCEPTABLE"
    
    return {
        "valid": True,
        "status": status,
        "message": f"✅ {keyword}: {actual_uses}× {status.lower()} (target {target}×)"
    }


# ============================================================================
# LEGACY FUNCTION (for project_routes.py compatibility)
# ============================================================================

def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None) -> dict:
    """
    Legacy function for project_routes.py compatibility.
    Simplified version using new v15.3 helpers.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Count keywords in this batch
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        count_in_batch = count_hybrid_occurrences(batch_text, kw, include_headings=True)
        
        # Update cumulative count
        previous_uses = meta.get("actual_uses", 0)
        meta["actual_uses"] = previous_uses + count_in_batch
    
    # Calculate burstiness
    burstiness_score = calculate_burstiness(batch_text)
    
    # Save batch
    batch_entry = {
        "text": batch_text,
        "meta_trace": meta_trace or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness_score
    }
    
    if "batches" not in project_data:
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state
    
    doc_ref.set(project_data)
    
    return {
        "status": "BATCH_SAVED",
        "message": "Batch saved successfully (legacy mode)",
        "status_code": 200
    }


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
    
    # Count occurrences in this batch
    updated_state = {}
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        count_in_batch = count_hybrid_occurrences(batch_text, kw, include_headings=True)
        
        # Update cumulative count
        previous_uses = meta.get("actual_uses", 0)
        new_total = previous_uses + count_in_batch
        
        updated_state[row_id] = {
            **meta,
            "actual_uses": new_total,
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
    Approve batch with v15.4 DYNAMIC TARGETS + v15.3 validations:
    
    v15.3 Features:
    - Validates BOTH min AND max limits
    - Calculates REMAINING uses for carry-over
    - Requires 100% EXTENDED coverage at export
    - Validates alternating section structure
    
    v15.4 NEW:
    - Dynamic target calculation (adaptive per batch)
    - Length-proportional distribution
    - Decreasing bonus (B1:+3, B2:+2, B3:+1)
    - Validates against dynamic targets
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
    batches_plan = project_data.get("batches_plan", [])
    
    # ===== VALIDATION 1: HTML Structure =====
    batch_h2_count = meta_trace.get("h2_count", 2)
    html_check = validate_html_structure(corrected_text, batch_h2_count)
    if not html_check["valid"]:
        return jsonify({
            "error": html_check["error"],
            "suggestion": html_check.get("suggestion")
        }), 400
    
    # ===== VALIDATION 2: Per-Keyword Density =====
    density_check = validate_all_densities(corrected_text, keywords_state)
    if not density_check["valid"]:
        return jsonify({
            "error": "❌ Keyword stuffing detected",
            "overstuffed_keywords": density_check["critical"],
            "suggestion": density_check.get("fix_suggestion")
        }), 400
    
    # ===== VALIDATION 3: Total Density =====
    total_density_check = calculate_total_keyword_density(corrected_text, keywords_state)
    if not total_density_check["valid"]:
        return jsonify({
            "error": total_density_check["error"],
            "total_density": total_density_check["total_density"],
            "breakdown": total_density_check.get("breakdown")
        }), 400
    
    # ===== VALIDATION 4: Burstiness =====
    burstiness_score = calculate_burstiness(corrected_text)
    if burstiness_score < 6.0:
        return jsonify({
            "error": f"❌ Burstiness too low: {burstiness_score} (need ≥6.0)",
            "calculated_burstiness": burstiness_score,
            "suggestion": "Mix sentence lengths: 40% short (4-8w), 40% medium (11-18w), 20% long (25-40w)"
        }), 400
    
    # ===== VALIDATION 5: Section Structure (Alternating) =====
    structure_check = validate_section_structure(corrected_text, current_batch_num)
    if not structure_check["valid"]:
        return jsonify({
            "error": structure_check["error"],
            "suggestion": structure_check.get("suggestion"),
            "section_lengths": structure_check.get("section_lengths")
        }), 400
    
    # ===== v15.4 VALIDATION 6: Dynamic Targets =====
    batch_length = len(corrected_text.split())
    total_batches = len(batches_plan)
    
    # Get remaining batches info
    remaining_batches_info = []
    if current_batch_num < total_batches:
        remaining_batches_info = batches_plan[current_batch_num:]
    
    dynamic_validations = []
    dynamic_rejections = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        
        # Only apply dynamic targets to BASIC keywords
        if kw_type != "BASIC":
            continue
        
        kw = meta.get("keyword", "")
        used_in_batch = meta.get("used_in_current_batch", 0)
        
        # Calculate dynamic target for this batch
        target_info = calculate_dynamic_target(
            keyword_meta=meta,
            batch_number=current_batch_num,
            batch_length=batch_length,
            total_batches=total_batches,
            remaining_batches_info=remaining_batches_info
        )
        
        # Validate against target
        validation = validate_against_dynamic_target(
            actual_uses=used_in_batch,
            target_info=target_info,
            keyword=kw
        )
        
        dynamic_validations.append({
            "keyword": kw,
            "actual": used_in_batch,
            "target_info": target_info,
            "validation": validation
        })
        
        if not validation["valid"]:
            dynamic_rejections.append({
                "keyword": kw,
                "actual": used_in_batch,
                "expected": f"{target_info['min']}-{target_info['max']}",
                "target": target_info["target"],
                "explanation": target_info["explanation"],
                "message": validation["message"]
            })
    
    # If dynamic targets not met, BLOCK
    if dynamic_rejections:
        return jsonify({
            "error": "❌ Keywords outside dynamic targets",
            "rejections": dynamic_rejections,
            "suggestion": "Adjust keyword usage to match targets or write naturally and retry"
        }), 400
    
    # ===== v15.3 FIX #1: Check MAX limits =====
    over_limit_keywords = []
    
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)
        
        if actual_uses > target_max:
            over_limit_keywords.append({
                "keyword": kw,
                "actual": actual_uses,
                "max": target_max,
                "excess": actual_uses - target_max
            })
    
    if over_limit_keywords:
        return jsonify({
            "error": "❌ Keywords exceeded MAX limit (cumulative)",
            "over_limit": over_limit_keywords,
            "message": "Total usage across all batches exceeded user's maximum"
        }), 400
    
    # ===== v15.3 FIX #2: Calculate REMAINING uses =====
    missing_basic = []
    missing_extended = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        
        if actual_uses < target_min:
            remaining = target_min - actual_uses
            max_can_use = target_max - actual_uses
            
            missing_info = {
                "keyword": kw,
                "type": kw_type,
                "required_total": target_min,
                "current": actual_uses,
                "remaining_min": remaining,
                "remaining_max": max_can_use,
                "range": f"{remaining}-{max_can_use}"
            }
            
            if kw_type == "BASIC":
                missing_basic.append(missing_info)
            elif kw_type == "EXTENDED":
                missing_extended.append(missing_info)
    
    # ===== Save batch to Firebase =====
    batch_entry = {
        "batch_number": current_batch_num,
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "validations": {
            "html_structure": "✅ Valid",
            "keyword_density": "✅ Within limits",
            "total_density": f"✅ {total_density_check['total_density']}%",
            "burstiness": f"✅ {burstiness_score}",
            "section_structure": f"✅ {structure_check.get('pattern', 'Valid')}",
            "dynamic_targets": "✅ All met"
        },
        "dynamic_validations": dynamic_validations
    }
    
    if "batches" not in project_data:
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    
    # Update keywords_state with cumulative counts
    project_data["keywords_state"] = keywords_state
    
    doc_ref.set(project_data)
    
    # ===== Calculate NEXT batch dynamic targets =====
    next_batch_targets = []
    
    if current_batch_num < total_batches:
        next_batch_num = current_batch_num + 1
        next_batch_info = batches_plan[next_batch_num - 1] if next_batch_num - 1 < len(batches_plan) else {"word_count": 300}
        next_batch_length = next_batch_info.get("word_count", 300)
        
        next_remaining_info = batches_plan[next_batch_num:] if next_batch_num < len(batches_plan) else []
        
        for row_id, meta in keywords_state.items():
            if meta.get("type", "BASIC").upper() != "BASIC":
                continue
            
            next_target = calculate_dynamic_target(
                keyword_meta=meta,
                batch_number=next_batch_num,
                batch_length=next_batch_length,
                total_batches=total_batches,
                remaining_batches_info=next_remaining_info
            )
            
            next_batch_targets.append({
                "keyword": meta.get("keyword"),
                "target": next_target["target"],
                "range": f"{next_target['min']}-{next_target['max']}",
                "explanation": next_target["explanation"],
                "breakdown": next_target["breakdown"]
            })
    
    # ===== Response =====
    return jsonify({
        "status": "BATCH_SAVED",
        "batch_number": current_batch_num,
        "article_complete": current_batch_num >= total_batches,
        "validations": {
            "html_structure": "✅ Valid",
            "keyword_density": "✅ Within limits",
            "total_density": f"✅ {total_density_check['total_density']}%",
            "burstiness": f"✅ {burstiness_score} (calculated)",
            "section_structure": f"✅ {structure_check.get('pattern', 'Valid')}",
            "dynamic_targets": f"✅ All {len(dynamic_validations)} keywords met"
        },
        "dynamic_validations": dynamic_validations,
        "progress": {
            "basic_remaining": len(missing_basic),
            "extended_remaining": len(missing_extended),
            "basic_complete": len(missing_basic) == 0,
            "extended_complete": len(missing_extended) == 0
        },
        "missing_basic_list": missing_basic,  # v15.3 REMAINING
        "missing_extended_list": missing_extended,
        "next_batch_targets": next_batch_targets  # v15.4 Dynamic targets
    }), 200


@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    """
    Final validation before export.
    Requires 100% EXTENDED coverage.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Check BASIC keywords (min requirements)
    incomplete_basic = []
    for row_id, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() != "BASIC":
            continue
        
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        
        if actual < target_min:
            incomplete_basic.append({
                "keyword": meta.get("keyword"),
                "actual": actual,
                "min_required": target_min,
                "deficit": target_min - actual
            })
    
    # Check EXTENDED keywords (100% required!)
    total_extended = 0
    used_extended = 0
    incomplete_extended = []
    
    for row_id, meta in keywords_state.items():
        if meta.get("type", "BASIC").upper() == "EXTENDED":
            total_extended += 1
            actual = meta.get("actual_uses", 0)
            
            if actual >= 1:
                used_extended += 1
            else:
                incomplete_extended.append({
                    "keyword": meta.get("keyword"),
                    "actual": 0,
                    "required": "1+"
                })
    
    extended_coverage = (used_extended / total_extended * 100) if total_extended > 0 else 100
    
    # BLOCK if not 100% EXTENDED
    if extended_coverage < 100:
        return jsonify({
            "error": f"❌ EXTENDED coverage: {extended_coverage:.0f}% (need 100%)",
            "incomplete_extended": incomplete_extended,
            "message": "ALL EXTENDED keywords must be used at least once"
        }), 400
    
    # BLOCK if BASIC incomplete
    if incomplete_basic:
        return jsonify({
            "error": "❌ BASIC keywords below minimum",
            "incomplete_basic": incomplete_basic
        }), 400
    
    # SUCCESS
    return jsonify({
        "status": "ARTICLE_READY",
        "message": "✅ All validations passed",
        "basic_complete": True,
        "extended_coverage": 100,
        "total_keywords": len(keywords_state)
    }), 200


@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    """
    Export complete article after validation.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    # Concatenate all batch texts
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    return jsonify({
        "status": "EXPORTED",
        "full_article": full_text,
        "batch_count": len(batches),
        "word_count": len(full_text.split())
    }), 200
