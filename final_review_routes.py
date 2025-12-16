import datetime
import re
import spacy
import os
from firebase_admin import firestore
from seo_optimizer import unified_prevalidation

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    nlp = spacy.load("pl_core_news_md")
except OSError:
    from spacy.cli import download
    download("pl_core_news_md")
    nlp = spacy.load("pl_core_news_md")

def process_batch_in_firestore(project_id, batch_text, meta_trace=None, forced=False, preview_only=False):
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Project not found"}

    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    
    # 1. Analiza tekstu (Lematyzacja)
    clean_text = re.sub(r"<[^>]+>", " ", batch_text or "").lower()
    doc_nlp = nlp(clean_text)
    text_lemmas = [t.lemma_ for t in doc_nlp if t.is_alpha]
    
    batch_counts = {}
    exceeded = []
    
    # 2. Zliczanie fraz (Strict Sequence Matching)
    for rid, meta in keywords.items():
        kw_lemma_tokens = [t.lemma_.lower() for t in nlp(meta["keyword"]) if t.is_alpha]
        count = 0
        if len(kw_lemma_tokens) <= len(text_lemmas):
            for i in range(len(text_lemmas) - len(kw_lemma_tokens) + 1):
                if text_lemmas[i:i+len(kw_lemma_tokens)] == kw_lemma_tokens:
                    count += 1
        batch_counts[rid] = count
        
        current = meta.get("actual_uses", 0)
        max_limit = meta.get("target_max", 5)
        
        if not preview_only:
            meta["actual_uses"] = current + count
            # Aktualizacja statusu
            if meta["actual_uses"] > max_limit: meta["status"] = "EXCEEDED"
            elif meta["actual_uses"] >= meta.get("target_min", 1): meta["status"] = "OK"
        
        # Check Limits
        total_predicted = current + count
        if total_predicted > max_limit:
            exceeded.append({
                "keyword": meta["keyword"],
                "limit": max_limit,
                "current": total_predicted
            })

    # 3. Metryki
    precheck = unified_prevalidation(batch_text, keywords)
    warnings = precheck.get("warnings", [])
    
    if exceeded:
        warnings.append(f"⚠️ EXCEEDED KEYWORDS: {', '.join([k['keyword'] for k in exceeded])}")

    status = "APPROVED"
    if warnings: status = "WARN"
    if forced: status = "FORCED"

    # 4. Zapis do bazy
    if not preview_only:
        batch_entry = {
            "text": batch_text,
            "timestamp": datetime.datetime.utcnow(),
            "status": status,
            "warnings": warnings,
            "metrics": precheck.get("metrics")
        }
        data.setdefault("batches", []).append(batch_entry)
        data["keywords_state"] = keywords
        doc_ref.set(data)
        
        # 🟢 TRIGGER FINAL REVIEW (Jeśli to ostatni batch)
        total_planned = data.get("total_planned_batches", 0)
        total_current = len(data.get("batches", []))
        
        if GEMINI_API_KEY and total_planned > 0 and total_current >= total_planned:
            try:
                # Dynamic import to avoid circular dependency
                from final_review_routes import perform_final_review_logic
                print(f"[TRACKER] 🏁 Final batch ({total_current}/{total_planned}) -> Triggering Gemini Review")
                review_res = perform_final_review_logic(project_id, data, data.get("batches"))
                return {
                    "status": status, 
                    "warnings": warnings, 
                    "final_review": review_res.get("review_text"),
                    "final_review_status": "GENERATED"
                }
            except Exception as e:
                print(f"[TRACKER] Final review trigger failed: {e}")

    return {
        "status": status,
        "warnings": warnings,
        "exceeded_keywords": exceeded,
        "metrics": precheck.get("metrics")
    }
