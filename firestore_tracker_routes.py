"""
SEO Content Tracker Routes - v15.6 FIXED NLP COUNTING
Naprawiono problem zliczania (actual_uses: 0) poprzez dodanie obsługi lematyzacji SpaCy.
"""

from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import re
import math
import datetime
import spacy

tracker_routes = Blueprint("tracker_routes", __name__)

# ============================================================================
# INIT SPACY (CRITICAL FOR POLISH MORPHOLOGY)
# ============================================================================
try:
    nlp = spacy.load("pl_core_news_lg")
    print("[TRACKER] ✅ Załadowano model pl_core_news_lg do zliczania fraz")
except OSError:
    print("[TRACKER] ⚠️ Model lg nieznaleziony, próba pobierania...")
    from spacy.cli import download
    download("pl_core_news_lg")
    nlp = spacy.load("pl_core_news_lg")

# ============================================================================
# HELPER FUNCTIONS - Keyword Counting (ROBUST)
# ============================================================================

def count_robust(doc, keyword_meta):
    """
    Liczy wystąpienia biorąc pod uwagę odmianę słów (Lematyzacja).
    Args:
        doc: Obiekt spaCy (przetworzony tekst batcha)
        keyword_meta: Metadane słowa z Firestore (zawierają 'search_lemma')
    """
    if not doc:
        return 0
        
    # 1. Przygotuj formy słowa kluczowego
    kw_exact = keyword_meta.get("keyword", "").lower().strip()
    
    # Pobierz lemat z metadanych lub wygeneruj w locie (fallback)
    kw_lemma = keyword_meta.get("search_lemma", "")
    if not kw_lemma:
        # Fallback jeśli brak w bazie
        kw_doc = nlp(kw_exact)
        kw_lemma = " ".join([t.lemma_.lower() for t in kw_doc if t.is_alpha])
    else:
        kw_lemma = kw_lemma.lower().strip()

    # 2. Zliczanie Exact (proste)
    text_lower = doc.text.lower()
    count_exact = text_lower.count(kw_exact)
    
    # 3. Zliczanie Lemma (zaawansowane - ignoruje odmianę)
    # Rekonstrukcja tekstu jako ciąg lematów
    text_lemma_str = " ".join([t.lemma_.lower() for t in doc if t.is_alpha])
    count_lemma = text_lemma_str.count(kw_lemma)
    
    # Zwracamy większą wartość (zazwyczaj lemma wyłapie więcej)
    return max(count_exact, count_lemma)


# ============================================================================
# VALIDATION FUNCTIONS (HARDENED)
# ============================================================================

def validate_html_and_headers(text, expected_h2_count=2):
    """Walidacja HTML i zakazanych nagłówków."""
    if "##" in text or "###" in text:
        return {"valid": False, "error": "❌ Wykryto Markdown (##/###). Użyj czystego HTML: <h2>, <h3>."}
    
    h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', text, re.IGNORECASE | re.DOTALL)
    
    if len(h2_matches) < expected_h2_count:
        return {"valid": False, "error": f"❌ Za mało sekcji H2. Oczekiwano: {expected_h2_count}, znaleziono: {len(h2_matches)}"}
    
    banned_headers = ["wstęp", "podsumowanie", "wprowadzenie", "zakończenie", "konkluzja", "introduction", "summary", "wnioski"]
    for h2_text in h2_matches:
        clean_h2 = h2_text.lower().strip()
        for ban in banned_headers:
            if clean_h2 == ban or clean_h2.startswith(f"{ban} ") or clean_h2.endswith(f" {ban}"):
                return {"valid": False, "error": f"❌ ZAKAZANY NAGŁÓWEK: '{h2_text}'. H2 musi być opisowy i zawierać słowa kluczowe."}
    
    return {"valid": True}

def validate_all_densities(text, keywords_state, current_doc=None):
    """Walidacja density na podstawie zliczeń NLP."""
    word_count = len(text.split())
    if word_count == 0: return {"valid": False, "error": "Empty text"}
    
    # Jeśli nie mamy doc, tworzymy go
    if not current_doc:
        current_doc = nlp(text)

    critical = []
    for row_id, meta in keywords_state.items():
        # Używamy robust counting dla dokładności
        actual_in_batch = count_robust(current_doc, meta)
        # UWAGA: Tu sprawdzamy tylko ten batch dla ostrzeżenia o density lokalnym
        # Ale density zazwyczaj liczy się globalnie lub per batch. 
        # Tutaj liczymy lokalne nasycenie w batchu.
        
        density = (actual_in_batch / word_count) * 100
        if density > 3.0:
            critical.append({"keyword": meta["keyword"], "density": round(density, 2), "limit": 3.0})
            
    if critical:
        return {"valid": False, "critical": critical, "error": "❌ Keyword stuffing detected (>3.0%)"}
    return {"valid": True}

def calculate_total_keyword_density(text, keywords_state):
    word_count = len(text.split())
    if word_count == 0: return {"valid": False}
    total_uses = sum(meta.get("actual_uses", 0) for meta in keywords_state.values())
    total_density = (total_uses / word_count) * 100
    if total_density > 8.0:
        return {"valid": False, "error": f"❌ Total density {total_density:.1f}% > 8.0%", "total_density": total_density}
    return {"valid": True, "total_density": round(total_density, 2)}

def calculate_burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3: return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    if mean == 0: return 0.0
    return round((math.sqrt(variance) / mean) * 10, 1)

def validate_section_structure(text, batch_num):
    sections = re.split(r'<h2[^>]*>.*?</h2>', text, flags=re.IGNORECASE | re.DOTALL)
    sections = [s.strip() for s in sections if s.strip()]
    if len(sections) < 2: return {"valid": True}
    
    lengths = [len(s.split()) for s in sections[:2]]
    types = ["LONG" if l >= 250 else "SHORT" if l >= 150 else "TOO_SHORT" for l in lengths]
    
    if "TOO_SHORT" in types:
        return {"valid": False, "error": "❌ Jedna z sekcji jest za krótka (<150 słów)."}
    if types[0] == types[1]:
        return {"valid": False, "error": f"❌ Brak naprzemienności: Obie sekcje są {types[0]}. Wymagane: LONG -> SHORT."}
        
    return {"valid": True, "pattern": f"{types[0]} -> {types[1]}"}

# ============================================================================
# DYNAMIC TARGET CALCULATION
# ============================================================================

def calculate_dynamic_target(keyword_meta, batch_number, batch_length, total_batches, remaining_batches_info):
    total_target = keyword_meta.get("target", 0) or round((keyword_meta.get("target_min", 1) + keyword_meta.get("target_max", 999)) / 2)
    already_used = keyword_meta.get("actual_uses", 0)
    remaining_target = max(0, total_target - already_used)
    
    remaining_batches = max(1, total_batches - batch_number + 1)
    avg_per_batch = remaining_target / remaining_batches
    batch_bonus = max(0, 4 - batch_number)
    
    remaining_lens = [batch_length] + [b.get("word_count", 300) for b in remaining_batches_info]
    avg_rem_len = sum(remaining_lens) / len(remaining_lens) if remaining_lens else 300
    length_factor = batch_length / avg_rem_len if avg_rem_len > 0 else 1.0
    
    target = max(1, round((avg_per_batch + batch_bonus) * length_factor))
    tolerance = 1 if remaining_batches == 1 else 2
    
    return {
        "target": target,
        "min": max(0, target - tolerance),
        "max": target + tolerance
    }

def validate_against_dynamic_target(actual_in_batch, target_info, keyword):
    min_val = target_info["min"]
    max_val = target_info["max"]
    
    if actual_in_batch < min_val:
        return {"valid": False, "status": "BELOW", "message": f"❌ {keyword}: {actual_in_batch}× (cel: {min_val}-{max_val})"}
    if actual_in_batch > max_val:
        return {"valid": False, "status": "ABOVE", "message": f"❌ {keyword}: {actual_in_batch}× (limit: {max_val})"}
    return {"valid": True, "status": "OK", "message": "OK"}

# ============================================================================
# ROUTES
# ============================================================================

@tracker_routes.post("/api/project/<project_id>/preview_batch")
def preview_batch(project_id):
    data = request.get_json(force=True) or {}
    batch_text = data.get("batch_text", "")
    
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    # ⭐ OPTIMIZATION: Process text ONCE
    doc_nlp = nlp(batch_text)
    
    updated_state = {}
    for row_id, meta in keywords_state.items():
        # Używamy count_robust zamiast prostego count
        count_in_batch = count_robust(doc_nlp, meta)
        
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
    data = request.get_json(force=True) or {}
    corrected_text = data.get("corrected_text", "")
    meta_trace = data.get("meta_trace", {})
    keywords_state = data.get("keywords_state", {}) # State from Preview (already counted?)
    
    # UWAGA: Front-end może przysłać keywords_state, ale bezpieczniej przeliczyć na Backendzie
    # żeby mieć pewność NLP.
    
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    
    project_data = doc.to_dict()
    current_batch_num = len(project_data.get("batches", [])) + 1
    batches_plan = project_data.get("batches_plan", []) or [{"word_count": 500}] * 5
    
    # ⭐ RE-COUNT with NLP (Safety Check)
    doc_nlp = nlp(corrected_text)
    
    # Aktualizujemy stan keywords na podstawie tego batcha (incremental update)
    # Pobieramy stan z bazy (najnowszy zatwierdzony)
    db_keywords_state = project_data.get("keywords_state", {})
    
    # Słownik do walidacji dynamicznej
    current_batch_counts = {}

    for row_id, meta in db_keywords_state.items():
        count_in_batch = count_robust(doc_nlp, meta)
        
        # Aktualizacja globalna
        meta["actual_uses"] = meta.get("actual_uses", 0) + count_in_batch
        current_batch_counts[meta["keyword"]] = count_in_batch
        
        # Update keyword state in memory for saving
        keywords_state[row_id] = meta

    # 1. HARD VALIDATIONS
    h2_expected = meta_trace.get("h2_count", 2)
    html_check = validate_html_and_headers(corrected_text, h2_expected)
    if not html_check["valid"]: return jsonify(html_check), 400
    
    density_check = validate_all_densities(corrected_text, keywords_state, current_doc=doc_nlp)
    if not density_check["valid"]: return jsonify({"error": density_check["error"], "details": density_check.get("critical")}), 400
    
    total_den_check = calculate_total_keyword_density(corrected_text, keywords_state)
    if not total_den_check["valid"]: return jsonify(total_den_check), 400
    
    burst_score = calculate_burstiness(corrected_text)
    if burst_score < 6.0: return jsonify({"error": f"❌ Burstiness {burst_score} < 6.0", "suggestion": "Zróżnicuj zdania."}), 400

    struct_check = validate_section_structure(corrected_text, current_batch_num)
    if not struct_check["valid"]: return jsonify(struct_check), 400

    # 2. SAVE
    batch_entry = {
        "batch_number": current_batch_num,
        "text": corrected_text,
        "meta_trace": meta_trace,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "validations": {"burstiness": burst_score, "html": "OK"}
    }
    
    if "batches" not in project_data: project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["keywords_state"] = keywords_state # Zapisujemy zaktualizowany stan
    
    doc_ref.set(project_data)
    
    # 3. GENERATE INSTRUCTIONS
    next_batch_num = current_batch_num + 1
    article_complete = current_batch_num >= len(batches_plan)
    gpt_instruction = ""
    next_targets_list = []
    
    if not article_complete:
        next_info = batches_plan[next_batch_num - 1] if next_batch_num - 1 < len(batches_plan) else {"word_count": 400}
        target_len = next_info.get("word_count", 400)
        
        gpt_instruction = f"""✅ BATCH {current_batch_num} ZATWIERDZONY.
⏩ ZADANIE NA BATCH {next_batch_num} (Długość: ~{target_len} słów):
1. STRUKTURA: Pamiętaj o naprzemienności (LONG <-> SHORT).
2. SŁOWA KLUCZOWE (Obowiązkowe w tym fragmencie):
"""
        next_rem_info = batches_plan[next_batch_num:]
        for row_id, meta in keywords_state.items():
            if meta.get("type") == "BASIC":
                calc = calculate_dynamic_target(meta, next_batch_num, target_len, len(batches_plan), next_rem_info)
                if calc["max"] > 0:
                    gpt_instruction += f"   - '{meta['keyword']}': {calc['min']}-{calc['max']}x\n"
                    next_targets_list.append({"keyword": meta['keyword'], "range": f"{calc['min']}-{calc['max']}"})
        
        unused_ext = [m['keyword'] for m in keywords_state.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) == 0]
        if unused_ext:
            gpt_instruction += "\n3. FRAZY WSPIERAJĄCE (Użyj 2-3):\n" + "\n".join([f"   - {k}" for k in unused_ext[:5]])

    else:
        gpt_instruction = "ARTYKUŁ UKOŃCZONY. Walidacja."

    return jsonify({
        "status": "BATCH_SAVED",
        "batch_number": current_batch_num,
        "article_complete": article_complete,
        "gpt_instruction": gpt_instruction,
        "next_batch_targets": next_targets_list,
        "progress": f"{current_batch_num}/{len(batches_plan)}"
    }), 200

# (Walidacja końcowa i eksport - bez zmian, zakładam że masz je z poprzednich plików)
@tracker_routes.post("/api/project/<project_id>/validate_article")
def validate_article(project_id):
    # Skrócona wersja dla spójności
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    data = doc.to_dict()
    kw_state = data.get("keywords_state", {})
    
    # Check Extended
    ext_total = sum(1 for m in kw_state.values() if m.get("type") == "EXTENDED")
    ext_used = sum(1 for m in kw_state.values() if m.get("type") == "EXTENDED" and m.get("actual_uses", 0) > 0)
    
    if ext_total > 0 and ext_used < ext_total:
        return jsonify({"error": f"❌ EXTENDED coverage: {int(ext_used/ext_total*100)}% (need 100%)"}), 400
        
    return jsonify({"status": "ARTICLE_READY"}), 200

@tracker_routes.get("/api/project/<project_id>/export")
def export_article(project_id):
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists: return jsonify({"error": "Not found"}), 404
    data = doc.to_dict()
    full_text = "\n\n".join([b.get("text", "") for b in data.get("batches", [])])
    return jsonify({"status": "EXPORTED", "full_article": full_text}), 200
