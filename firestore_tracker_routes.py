# firestore_tracker_routes_v15.3_FINAL.py
# CRITICAL FIXES:
# 1. Tracks REMAINING uses per keyword (max - current)
# 2. Validates BOTH min AND max limits
# 3. EXTENDED must reach 100% (not 70%)
# 4. Enforces alternating section lengths per batch
# 5. Sends complete keyword list to Firebase

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import datetime
import re
import math
from collections import Counter

tracker_routes = Blueprint('tracker_routes', __name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_hybrid_occurrences(text, keyword, include_headings=True):
    """Count keyword occurrences in text (case-insensitive)"""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Simple word boundary matching
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    matches = re.findall(pattern, text_lower)
    
    return len(matches)

def calculate_keyword_density(text, keyword, count):
    """Calculate keyword density percentage"""
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    keyword_words = len(keyword.split())
    keyword_total_words = count * keyword_words
    
    density = (keyword_total_words / total_words) * 100
    return round(density, 2)

def validate_html_structure(text, expected_h2_count):
    """Validate HTML structure"""
    h2_count = len(re.findall(r'<h2[^>]*>', text, re.IGNORECASE))
    
    # Check for Markdown
    if re.search(r'^##\s', text, re.MULTILINE):
        return {
            "valid": False,
            "error": "❌ Markdown detected (##). Use HTML: <h2>Title</h2>",
            "suggestion": "Convert all ## to <h2> tags"
        }
    
    # Check H2 count
    if h2_count < expected_h2_count:
        return {
            "valid": False,
            "error": f"❌ Expected {expected_h2_count} H2 tags, found {h2_count}",
            "suggestion": f"Add {expected_h2_count - h2_count} more H2 sections"
        }
    
    return {"valid": True, "h2_count": h2_count}

def validate_all_densities(text, keywords_state):
    """Validate per-keyword density (<3% for BASIC)"""
    critical = []
    warnings = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        if kw_type != "BASIC":
            continue
            
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        
        density = calculate_keyword_density(text, kw, actual_uses)
        
        if density > 3.0:
            critical.append({
                "keyword": kw,
                "density": density,
                "limit": 3.0
            })
        elif density > 2.5:
            warnings.append({
                "keyword": kw,
                "density": density
            })
    
    if critical:
        return {
            "valid": False,
            "critical": critical,
            "fix_suggestion": "Reduce usage or replace with synonyms"
        }
    
    return {"valid": True, "warnings": warnings}

def calculate_total_keyword_density(text, keywords_state):
    """Calculate TOTAL density of all BASIC keywords combined"""
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {"valid": True, "total_density": 0}
    
    total_keyword_words = 0
    breakdown = []
    
    for row_id, meta in keywords_state.items():
        if meta.get("type") == "BASIC":
            kw = meta["keyword"]
            count = meta.get("actual_uses", 0)
            kw_words = len(kw.split())
            kw_total_words = count * kw_words
            total_keyword_words += kw_total_words
            
            breakdown.append({
                "keyword": kw,
                "uses": count,
                "words": kw_total_words
            })
    
    total_density = (total_keyword_words / total_words) * 100
    
    if total_density > 8.0:
        return {
            "valid": False,
            "error": f"❌ Total BASIC density: {total_density:.1f}% (max 8.0%)",
            "total_density": round(total_density, 2),
            "breakdown": breakdown,
            "suggestion": "Reduce overall keyword usage across all BASIC keywords"
        }
    
    return {
        "valid": True,
        "total_density": round(total_density, 2),
        "breakdown": breakdown
    }

def calculate_burstiness(text):
    """Calculate sentence length variance (burstiness score)"""
    # Extract sentences (simple split)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    if len(sentences) < 3:
        return 0.0
    
    # Count words per sentence
    lengths = [len(s.split()) for s in sentences]
    
    # Calculate std dev and mean
    mean_length = sum(lengths) / len(lengths)
    variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    
    if mean_length == 0:
        return 0.0
    
    # Burstiness score (normalized to 0-10 scale)
    burstiness = (std_dev / mean_length) * 10
    
    return round(burstiness, 1)

def validate_section_structure(batch_text, batch_num):
    """
    Validate alternating section lengths:
    - If section 1 LONG → section 2 must be SHORT
    - If section 1 SHORT → section 2 must be LONG
    - Section 3 can be any length
    
    LONG: 250-400 words
    SHORT: 150-250 words
    """
    # Extract H2 sections
    sections = re.split(r'<h2[^>]*>.*?</h2>', batch_text, flags=re.IGNORECASE | re.DOTALL)
    sections = [s.strip() for s in sections if len(s.strip()) > 50]  # Remove intro/empty
    
    if len(sections) < 2:
        return {"valid": True}  # Not enough sections to validate
    
    # Count words per section
    section_lengths = []
    for section in sections[:3]:  # Check first 3 sections
        words = section.split()
        section_lengths.append(len(words))
    
    # Categorize sections
    def categorize(word_count):
        if word_count >= 250:
            return "LONG"
        elif word_count <= 250:
            return "SHORT"
        else:
            return "MEDIUM"
    
    if len(section_lengths) >= 2:
        sec1_type = categorize(section_lengths[0])
        sec2_type = categorize(section_lengths[1])
        
        # Validate alternation
        if sec1_type == "LONG" and sec2_type != "SHORT":
            return {
                "valid": False,
                "error": f"❌ Section 1 is LONG ({section_lengths[0]}w) but Section 2 is not SHORT ({section_lengths[1]}w)",
                "suggestion": "Shorten Section 2 to 150-250 words",
                "section_lengths": section_lengths
            }
        elif sec1_type == "SHORT" and sec2_type != "LONG":
            return {
                "valid": False,
                "error": f"❌ Section 1 is SHORT ({section_lengths[0]}w) but Section 2 is not LONG ({section_lengths[1]}w)",
                "suggestion": "Expand Section 2 to 250-400 words",
                "section_lengths": section_lengths
            }
    
    return {
        "valid": True,
        "section_lengths": section_lengths,
        "pattern": f"{categorize(section_lengths[0])} → {categorize(section_lengths[1]) if len(section_lengths) > 1 else 'N/A'}"
    }

# ============================================================================
# LEGACY FUNCTION (for project_routes.py compatibility)
# ============================================================================

def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None) -> dict:
    """
    Legacy function used by project_routes.py for direct saving.
    Kept for backward compatibility with existing imports.
    
    NOTE: New code should use approve_batch endpoint instead.
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"error": "Project not found", "status_code": 404}
        
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Count keyword occurrences in this batch
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
    """Preview batch with keyword counting"""
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
    Approve batch with v15.3 CRITICAL FIXES:
    1. Validates BOTH min AND max limits
    2. Calculates REMAINING uses for next batch
    3. Requires 100% EXTENDED coverage at export
    4. Validates alternating section structure
    5. Saves complete keyword state to Firebase
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
    
    # ===== FIX #1: Check MAX limits (not just min) =====
    over_limit_keywords = []
    
    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_max = meta.get("target_max", 999)  # User-provided max
        
        if actual_uses > target_max:
            over_limit_keywords.append({
                "keyword": kw,
                "actual": actual_uses,
                "max": target_max,
                "excess": actual_uses - target_max
            })
    
    if over_limit_keywords:
        return jsonify({
            "error": "❌ Keywords exceeded MAX limit",
            "over_limit": over_limit_keywords,
            "message": "Reduce usage in this batch or use synonyms"
        }), 400
    
    # ===== FIX #2: Calculate REMAINING uses for carry-over =====
    missing_basic = []
    missing_extended = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        
        # Calculate REMAINING uses needed
        if actual_uses < target_min:
            remaining = target_min - actual_uses
            max_can_use = target_max - actual_uses
            
            missing_info = {
                "keyword": kw,
                "type": kw_type,
                "required_total": target_min,
                "current": actual_uses,
                "remaining_min": remaining,  # Must use at least this many
                "remaining_max": max_can_use,  # Can use up to this many
                "range": f"{remaining}-{max_can_use}"  # NEW: Show remaining range
            }
            
            if kw_type == "BASIC":
                missing_basic.append(missing_info)
            elif kw_type == "EXTENDED":
                missing_extended.append(missing_info)
    
    # Save batch to Firebase
    batch_entry = {
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "burstiness": burstiness_score,
        "section_structure": structure_check.get("pattern", "N/A")
    }
    
    if "batches" not in project_data:
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    
    # ===== FIX #3: Save COMPLETE keyword state to Firebase =====
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)
    
    # Prepare response
    article_complete = (len(missing_basic) == 0 and len(missing_extended) == 0)
    
    response = {
        "status": "BATCH_SAVED",
        "batch_number": current_batch_num,
        "next_action": "GENERATE_NEXT" if not article_complete else "READY_FOR_EXPORT",
        "article_complete": article_complete,
        "validations": {
            "html_structure": "✅ Valid",
            "keyword_density": "✅ Within limits",
            "total_density": f"✅ {total_density_check['total_density']}%",
            "burstiness": f"✅ {burstiness_score} (calculated)",
            "section_structure": f"✅ {structure_check.get('pattern', 'N/A')}"
        },
        "progress": {
            "basic_remaining": len(missing_basic),
            "extended_remaining": len(missing_extended),
            "basic_complete": len(missing_basic) == 0,
            "extended_complete": len(missing_extended) == 0
        }
    }
    
    # ===== FIX #4: CARRY-OVER lists with REMAINING ranges =====
    if missing_basic:
        response["warning_basic"] = f"⚠️ {len(missing_basic)} BASIC keywords need more uses"
        response["missing_basic_list"] = missing_basic
        
    if missing_extended:
        response["warning_extended"] = f"⚠️ {len(missing_extended)} EXTENDED keywords carry over"
        response["missing_extended_list"] = missing_extended
    
    return jsonify(response), 200

@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    """
    Final validation before export
    v15.3 FIX: EXTENDED must reach 100% (not 70%)
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # Check completeness
    incomplete_basic = []
    incomplete_extended = []
    
    for row_id, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        kw = meta.get("keyword", "")
        actual_uses = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        
        if actual_uses < target_min:
            if kw_type == "BASIC":
                incomplete_basic.append({
                    "keyword": kw,
                    "required": target_min,
                    "current": actual_uses,
                    "missing": target_min - actual_uses
                })
            elif kw_type == "EXTENDED":
                incomplete_extended.append({
                    "keyword": kw,
                    "required": target_min,
                    "current": actual_uses
                })
    
    # ===== FIX #5: EXTENDED must be 100% (not 70%) =====
    total_extended = sum(1 for v in keywords_state.values() if v.get("type") == "EXTENDED")
    used_extended = sum(1 for v in keywords_state.values() 
                       if v.get("type") == "EXTENDED" and v.get("actual_uses", 0) >= 1)
    
    if total_extended > 0:
        extended_coverage = (used_extended / total_extended) * 100
    else:
        extended_coverage = 100
    
    # BLOCK if <100% EXTENDED
    if extended_coverage < 100:
        return jsonify({
            "valid": False,
            "article_ready": False,
            "error": f"❌ EXTENDED coverage: {extended_coverage:.0f}% (need 100%)",
            "missing_extended": incomplete_extended,
            "total_extended": total_extended,
            "used_extended": used_extended,
            "message": "ALL EXTENDED keywords must be used at least once"
        }), 400
    
    if incomplete_basic:
        return jsonify({
            "valid": False,
            "article_ready": False,
            "missing_basic": incomplete_basic,
            "message": "Some BASIC keywords below minimum"
        }), 400
    
    return jsonify({
        "valid": True,
        "article_ready": True,
        "extended_coverage": extended_coverage,
        "message": "✅ Article complete and ready for export"
    }), 200

@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    """Export complete article"""
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    # Combine all batch texts
    full_text = "\n\n".join([b.get("text", "") for b in batches])
    
    return jsonify({
        "status": "EXPORTED",
        "article_html": full_text,
        "total_batches": len(batches),
        "total_words": len(full_text.split())
    }), 200
