import os
import json
import math
import re
import numpy as np
import datetime 
from flask import Blueprint, jsonify, request
from firebase_admin import firestore
import spacy
import google.generativeai as genai
from rapidfuzz import fuzz           
import language_tool_python         
import textstat                     
import textdistance                 
import pysbd                        

# ===========================================================
# Version: v12.25.6.2
# Last updated: 2024-12-03
# Changes: 
# - v12.25.4: Disabled Gemini auto-fix (quality issues), warnings only
# - v12.25.5: Fixed false positive keyword counting (stricter fuzzy thresholds)
#   * FUZZY_SIMILARITY_THRESHOLD: 90 → 95
#   * MAX_FUZZY_WINDOW_EXPANSION: 2 → 1
#   * JACCARD_THRESHOLD: 0.8 → 0.85
#   * Added 60% word overlap requirement for fuzzy matches
# - v12.25.6: Added paragraph structure validation
#   * Min 4 sentences per paragraph (was 3)
#   * Min 1 compound sentence per paragraph (with comma or conjunction)
#   * Backend REJECTS if paragraph rules violated
# - v12.25.6.1: ULTRA-STRICT fuzzy matching (eliminate all false positives)
#   * FUZZY_SIMILARITY_THRESHOLD: 95 → 98 → 99
#   * JACCARD_THRESHOLD: 0.85 → 0.90 → 0.95
#   * Word overlap: 60% → 75% → 85%
#   * Fuzzy DISABLED for 1-2 word phrases (exact + lemma only)
# - v12.25.6.2: Changed overuse policy (WARNING instead of REJECT)
#   * Keyword overuse +3 now gives WARNING instead of REJECTED_SEO
#   * Batch saves normally with warning
#   * Allows user to manually fix if needed
#   * Protects minimum 2 uses per keyword in fix suggestions
# ===========================================================

tracker_routes = Blueprint("tracker_routes", __name__)

# --- INICJALIZACJA ---
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

FUZZY_SIMILARITY_THRESHOLD = 98  # v12.25.6.1: Increased from 95 to 98 (eliminate false positives)    
MAX_FUZZY_WINDOW_EXPANSION = 1   # v12.25.5: Reduced from 2 to 1 (stricter matching)    
JACCARD_SIMILARITY_THRESHOLD = 0.90  # v12.25.6.1: Increased from 0.85 to 0.90   

try:
    LT_TOOL_PL = language_tool_python.LanguageTool("pl-PL")
except Exception:
    LT_TOOL_PL = None

textstat.set_lang("pl")
SENTENCE_SEGMENTER = pysbd.Segmenter(language="pl", clean=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ===========================================================
# 👮‍♂️ HARD GUARDRAILS
# ===========================================================
def validate_paragraph_structure(text: str) -> dict:
    """
    Waliduje strukturę akapitów:
    - Min 4 zdania na akapit
    - Min 1 zdanie złożone (z przecinkiem lub spójnikiem) na akapit
    v12.25.6
    """
    errors = []
    
    # Rozdziel na akapity (podwójny newline lub ## heading)
    paragraphs = re.split(r'\n\n+|^##\s+.+$', text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and not p.startswith('#')]
    
    for i, para in enumerate(paragraphs, 1):
        # Segmentacja zdań
        try:
            sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(para) if s.strip()]
        except:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
        
        if len(sentences) < 4:
            errors.append(f"Akapit {i}: {len(sentences)} zdań (min: 4)")
        
        # Sprawdź zdanie złożone (z przecinkiem lub spójnikiem)
        has_compound = False
        for sent in sentences:
            # Zdanie złożone: ma przecinek LUB spójnik (który, ale, jednak, ponieważ, gdyż, jeśli)
            if ',' in sent or any(conj in sent.lower() for conj in [' który', ' która', ' które', ' ale ', ' jednak ', ' ponieważ ', ' gdyż ', ' jeśli ', ' gdy ', ' bo ']):
                has_compound = True
                break
        
        if not has_compound and len(sentences) >= 4:
            errors.append(f"Akapit {i}: brak zdania złożonego (min: 1)")
    
    if errors:
        return {"valid": False, "msg": " | ".join(errors[:3])}  # Max 3 errors
    return {"valid": True, "msg": "OK"}

def validate_hard_rules(text: str) -> dict:
    """Sprawdza twarde reguły (listy punktowane)"""
    errors = []
    if re.search(r'^[\-\*]\s+', text, re.MULTILINE) or re.search(r'^\d+\.\s+', text, re.MULTILINE):
        matches = len(re.findall(r'^[\-\*]\s+', text, re.MULTILINE))
        if matches > 1:
            errors.append(f"WYKRYTO LISTĘ ({matches} pkt). Zakaz punktorów.")
    if errors:
        return {"valid": False, "msg": " | ".join(errors)}
    return {"valid": True, "msg": "OK"}

def sanitize_typography(text: str) -> str:
    """Podstawowa korekta typografii"""
    if not text: return ""
    text = text.replace("—", " – ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===========================================================
# 📏 AUDYT JĘZYKOWY
# ===========================================================
def analyze_language_quality(text: str) -> dict:
    """
    Analizuje tekst pod kątem:
    - Burstiness (różnorodność długości zdań)
    - Fluff ratio (przymiotniki/przysłówki)
    - Passive voice
    - Readability (SMOG)
    - Gramatyka (LanguageTool)
    - Powtarzające się początki zdań
    - Banned phrases
    """
    result = {
        "burstiness": 0.0, "fluff_ratio": 0.0, "passive_ratio": 0.0, 
        "readability_score": 0.0, "smog_index": 0.0, "sentence_count": 0,
        "lt_errors": [], "repeated_starts": [], "banned_detected": []
    }
    if not text.strip(): return result
    
    try:
        # Segmentacja zdań
        sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(text) if s.strip()]
        if not sentences: sentences = re.split(r'(?<=[.!?])\s+', text)
        result["sentence_count"] = len(sentences)
        
        # Burstiness (wariancja długości zdań)
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            mean = sum(lengths) / len(lengths)
            var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            result["burstiness"] = math.sqrt(var)

        # spaCy analysis
        doc = nlp(text)
        
        # Fluff ratio (przymiotniki + przysłówki)
        adv_adj = sum(1 for t in doc if t.pos_ in ("ADJ", "ADV"))
        total = sum(1 for t in doc if t.is_alpha)
        result["fluff_ratio"] = (adv_adj / total) if total > 0 else 0.0
        
        # Passive voice (strona bierna)
        passive = 0
        for sent in doc.sents:
            if any(t.lemma_ == "zostać" for t in sent) and any("ppas" in (t.tag_ or "") for t in sent):
                passive += 1
        result["passive_ratio"] = passive / len(list(doc.sents)) if list(doc.sents) else 0.0

        # Readability
        try:
            result["readability_score"] = textstat.flesch_reading_ease(text)
            result["smog_index"] = textstat.smog_index(text)
        except: pass

        # LanguageTool grammar check
        if LT_TOOL_PL:
            matches = LT_TOOL_PL.check(text)
            errs = [m.message for m in matches if m.ruleId not in ("WHITESPACE_RULE", "UPPERCASE_SENTENCE_START")]
            result["lt_errors"] = errs[:3]

        # Repeated sentence starts
        prefix_counts = {}
        for s in sentences:
            words = s.split()
            if len(words) > 2:
                p = " ".join(words[:2]).lower()
                prefix_counts[p] = prefix_counts.get(p, 0) + 1
        result["repeated_starts"] = [p for p, c in prefix_counts.items() if c >= 2]

        # Banned phrases
        banned_phrases = [
            "warto zauważyć", "w dzisiejszych czasach", "podsumowując", 
            "reasumując", "warto dodać", "nie da się ukryć"
        ]
        found = []
        text_l = text.lower()
        for b in banned_phrases:
            if b in text_l or fuzz.partial_ratio(b, text_l) > 92: 
                found.append(b)
        result["banned_detected"] = list(set(found))
        
    except Exception as e: 
        print(f"Audit Error: {e}")
    
    return result

# ===========================================================
# 🚀 HYBRID KEYWORD COUNTER (v12.25.2 - Fixed Loop Bug)
# ===========================================================
def count_hybrid_occurrences(text_raw, text_lemma_list, target_exact, target_lemma, debug=False):
    """
    Hybrid counter: exact + lemma + fuzzy
    v12.25.2: Fixed false positive loop bug with position tracking
    v12.25.5: Stricter thresholds + validation to reduce false positives
    """
    text_lower = text_raw.lower()
    exact_hits = text_lower.count(target_exact.lower()) if target_exact.strip() else 0
    
    lemma_hits = 0
    target_tok = target_lemma.split()
    fuzzy_matches = []  # v12.25.5: Track what fuzzy catches (for debugging)
    
    if target_tok:
        text_len = len(text_lemma_list)
        target_len = len(target_tok)
        used_indices = set()
        
        # Exact Lemma Match
        for i in range(text_len - target_len + 1):
            if text_lemma_list[i : i+target_len] == target_tok:
                lemma_hits += 1
                for k in range(i, i+target_len): 
                    used_indices.add(k)
        
        # Fuzzy Lemma Match
        min_win = max(1, target_len - MAX_FUZZY_WINDOW_EXPANSION)
        max_win = target_len + MAX_FUZZY_WINDOW_EXPANSION
        target_str = " ".join(target_tok)
        
        for w_len in range(min_win, max_win + 1):
            if w_len > text_len: continue
            for i in range(text_len - w_len + 1):
                # Skip if already used
                if any(k in used_indices for k in range(i, i+w_len)): 
                    continue
                
                window_tok = text_lemma_list[i : i+w_len]
                window_str = " ".join(window_tok)
                
                # v12.25.6.1: ULTRA-STRICT VALIDATION - prevent false positives
                # 1. Skip fuzzy for short phrases (1-2 words) - exact + lemma only
                if len(target_tok) <= 2:
                    continue  # Fuzzy disabled for short phrases
                
                # 2. Require at least 75% word overlap (increased from 60%)
                common_words = set(target_tok) & set(window_tok)
                word_overlap = len(common_words) / len(target_tok) if target_tok else 0
                
                if word_overlap < 0.75:
                    continue  # Skip if less than 75% words match
                
                # 3. Fuzzy match (now with ultra-high thresholds: 98/0.90)
                token_set = fuzz.token_set_ratio(target_str, window_str)
                jaccard = textdistance.jaccard.normalized_similarity(target_tok, window_tok)
                
                if (token_set >= FUZZY_SIMILARITY_THRESHOLD or jaccard >= JACCARD_SIMILARITY_THRESHOLD):
                    lemma_hits += 1
                    
                    # v12.25.5: Log what fuzzy caught (for debugging)
                    if debug:
                        fuzzy_matches.append({
                            "matched": window_str,
                            "token_set_score": token_set,
                            "jaccard_score": jaccard,
                            "word_overlap": word_overlap
                        })
                    
                    for k in range(i, i+w_len): 
                        used_indices.add(k)
    
    # v12.25.5: Debug logging
    if debug and fuzzy_matches:
        print(f"\n🔍 FUZZY DEBUG for '{target_exact}':")
        print(f"  Exact hits: {exact_hits}")
        print(f"  Lemma hits: {lemma_hits}")
        print(f"  Fuzzy matches found: {len(fuzzy_matches)}")
        for m in fuzzy_matches[:5]:  # Show first 5
            print(f"    - '{m['matched']}' (token_set: {m['token_set_score']}, overlap: {m['word_overlap']:.1%})")
    
    # Return MAX (not sum) to avoid double-counting
    return max(exact_hits, lemma_hits)

def compute_status(actual, target_min, target_max):
    """Oblicza status keyword: UNDER/OK/OVER"""
    if actual < target_min: return "UNDER"
    if actual > target_max: return "OVER"
    return "OK"

def global_keyword_stats(keywords_state):
    """Globalne statystyki keywords"""
    under = sum(1 for v in keywords_state.values() if v["status"] == "UNDER")
    over = sum(1 for v in keywords_state.values() if v["status"] == "OVER")
    locked = 1 if over >= 4 else 0
    ok = sum(1 for v in keywords_state.values() if v["status"] == "OK")
    return under, over, locked, ok

# ===========================================================
# 🎯 SEMANTIC & TRANSITION ANALYSIS (v12.25.1)
# ===========================================================
def get_embedding(text):
    """Get Gemini embedding for semantic analysis"""
    if not text or not text.strip(): return None
    try:
        return genai.embed_content(
            model="models/text-embedding-004", 
            content=text, 
            task_type="retrieval_document"
        )['embedding']
    except: 
        return None

def calculate_semantic_drift(batch_text, previous_batches, threshold=0.65):
    """
    Sprawdza semantic drift między nowym batchem a poprzednimi
    Returns: {drift_score, status, message}
    """
    if not previous_batches or len(previous_batches) == 0:
        return {
            "drift_score": 1.0,
            "status": "OK",
            "message": "Pierwszy batch - brak porównania"
        }
    
    # Embedding nowego batcha
    new_vec = get_embedding(batch_text)
    if not new_vec:
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": "Nie udało się utworzyć embeddingu"
        }
    
    # Embedding ostatnich 3 batchy (kontekst)
    recent_batches = previous_batches[-3:] if len(previous_batches) >= 3 else previous_batches
    recent_text = " ".join([b.get("text", "") for b in recent_batches])
    
    prev_vec = get_embedding(recent_text)
    if not prev_vec:
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": "Nie udało się utworzyć embeddingu kontekstu"
        }
    
    # Cosine similarity
    try:
        similarity = float(
            np.dot(new_vec, prev_vec) / 
            (np.linalg.norm(new_vec) * np.linalg.norm(prev_vec))
        )
    except Exception as e:
        return {
            "drift_score": 0,
            "status": "EMBEDDING_FAILED",
            "message": str(e)
        }
    
    # Ocena
    if similarity >= threshold:
        status = "OK"
        message = f"Spójność tematyczna: {similarity*100:.1f}%"
    else:
        status = "DRIFT_WARNING"
        message = f"⚠️ Odchylenie tematyczne: {similarity*100:.1f}% (cel: >{threshold*100:.0f}%)"
    
    return {
        "drift_score": round(similarity, 3),
        "status": status,
        "message": message
    }

def analyze_transition_quality(current_batch, previous_batch):
    """
    Ocenia płynność przejścia między batches
    Returns: {score, status, has_transition_word, entity_continuity, message}
    """
    if not previous_batch:
        return {
            "score": 1.0,
            "status": "FIRST_BATCH",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Pierwszy batch"
        }
    
    prev_text = previous_batch.get("text", "")
    if not prev_text or not current_batch:
        return {
            "score": 0.5,
            "status": "NO_TEXT",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Brak tekstu do analizy"
        }
    
    # Segmentacja zdań
    try:
        prev_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(prev_text) if s.strip()]
        curr_sentences = [s.strip() for s in SENTENCE_SEGMENTER.segment(current_batch) if s.strip()]
    except:
        prev_sentences = [s.strip() for s in prev_text.split('.') if s.strip()]
        curr_sentences = [s.strip() for s in current_batch.split('.') if s.strip()]
    
    if not prev_sentences or not curr_sentences:
        return {
            "score": 0.5,
            "status": "NO_SENTENCES",
            "has_transition_word": False,
            "entity_continuity": 0.0,
            "message": "Brak zdań"
        }
    
    last_prev = prev_sentences[-1].lower()
    first_curr = curr_sentences[0].lower()
    
    # 1. LEXICAL OVERLAP
    prev_words = set(re.findall(r'\w+', last_prev))
    curr_words = set(re.findall(r'\w+', first_curr))
    common_words = prev_words & curr_words
    
    if len(prev_words) > 0 and len(curr_words) > 0:
        lexical_overlap = len(common_words) / max(len(prev_words), len(curr_words))
    else:
        lexical_overlap = 0.0
    
    # 2. TRANSITION WORDS
    transition_words = [
        "jednak", "jednakże", "niemniej", "mimo to",
        "ponadto", "dodatkowo", "również", "oprócz tego", "poza tym",
        "z kolei", "kolejnym", "następnie", "dalej",
        "warto", "istotne", "ważne",
        "w związku z tym", "dlatego", "zatem"
    ]
    
    has_transition = any(tw in first_curr for tw in transition_words)
    
    # 3. ENTITY CONTINUITY (NER)
    try:
        prev_doc = nlp(prev_text[-500:])
        curr_doc = nlp(current_batch[:500])
        
        prev_entities = {ent.text.lower() for ent in prev_doc.ents}
        curr_entities = {ent.text.lower() for ent in curr_doc.ents}
        
        if len(prev_entities) > 0:
            entity_overlap = len(prev_entities & curr_entities) / len(prev_entities)
        else:
            entity_overlap = 0.0
    except:
        entity_overlap = 0.0
    
    # COMBINED SCORE
    score = (
        lexical_overlap * 0.3 +
        (0.3 if has_transition else 0.0) +
        entity_overlap * 0.4
    )
    
    status = "SMOOTH" if score >= 0.5 else "CHOPPY"
    
    message = f"Przejście: {'płynne' if status == 'SMOOTH' else 'urywane'} ({score*100:.0f}%)"
    if not has_transition and status == "CHOPPY":
        message += " - rozważ dodanie słowa przejściowego"
    
    return {
        "score": round(score, 2),
        "status": status,
        "has_transition_word": has_transition,
        "entity_continuity": round(entity_overlap, 2),
        "lexical_overlap": round(lexical_overlap, 2),
        "message": message
    }

# ===========================================================
# 🤖 GEMINI JUDGE (v12.25.4 - NO AUTO-FIX)
# ===========================================================
def evaluate_with_gemini(batch_text, project_data, audit):
    """
    Gemini ocenia batch pod kątem HEAR Framework
    v12.25.4: TYLKO WARNINGS - NO AUTO-FIX
    """
    if not GEMINI_API_KEY: 
        return {
            "pass": True, 
            "quality_score": 80,
            "warnings": [],
            "suggestions": []
        }
    
    try: 
        model = genai.GenerativeModel("gemini-1.5-flash")
    except: 
        return {
            "pass": True, 
            "quality_score": 80,
            "warnings": [],
            "suggestions": []
        }
    
    # Rolling context (last 3 batches)
    context_parts = []
    batches = project_data.get("batches", [])
    if batches:
        recent_batches = batches[-3:] if len(batches) >= 3 else batches
        for i, batch in enumerate(recent_batches):
            batch_text_prev = batch.get("text", "")
            h2_pattern = r'##\s+(.+?)(?:\n|$)'
            headings = re.findall(h2_pattern, batch_text_prev)
            context_parts.append({
                "batch_num": len(batches) - len(recent_batches) + i + 1,
                "headings": headings,
                "snippet": batch_text_prev[:200] + "..."
            })
    
    context_summary = ""
    if context_parts:
        context_lines = ["KONTEKST ARTYKUŁU (ostatnie batche):"]
        for c in context_parts:
            h2_text = ", ".join(c["headings"]) if c["headings"] else "brak H2"
            context_lines.append(f"\nBATCH {c['batch_num']}: {h2_text}\n  {c['snippet']}")
        context_summary = "\n".join(context_lines)
    
    topic = project_data.get("topic", "Nieznany")
    
    prompt = f"""
Evaluate this Polish text batch for HEAR Framework compliance.

TOPIC: "{topic}"

{context_summary}

NEW BATCH:
{batch_text}

METRICS (already calculated):
- Burstiness: {audit.get('burstiness', 0):.1f} (target: >6.0)
- Fluff ratio: {audit.get('fluff_ratio', 0):.2f} (target: <0.15)
- SMOG index: {audit.get('smog_index', 0):.1f} (target: <15)
- Passive voice: {audit.get('passive_ratio', 0)*100:.0f}% (target: <30%)
- Banned phrases: {audit.get('banned_detected', [])}

Analyze:
1. Overall quality (HEAR: Harmonia, Empatia, Autentyczność, Rytm)
2. Identify specific issues
3. Suggest improvements FOR USER TO APPLY

Return ONLY JSON:
{{
  "pass": true/false,
  "quality_score": 0-100,
  "warnings": ["list of specific issues"],
  "suggestions": ["list of how to fix - for USER, not auto-fix"],
  "feedback_for_writer": "brief overall assessment"
}}

CRITICAL: NO auto-fix text generation. Warnings and suggestions only.
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        # Ensure required fields
        if "warnings" not in result:
            result["warnings"] = []
        if "suggestions" not in result:
            result["suggestions"] = []
        if "pass" not in result:
            result["pass"] = True
        if "quality_score" not in result:
            result["quality_score"] = 80
            
        return result
        
    except Exception as e:
        print(f"Gemini evaluation error: {e}")
        return {
            "pass": True, 
            "quality_score": 80,
            "warnings": [],
            "suggestions": [],
            "feedback_for_writer": "Gemini niedostępny - zaakceptowano batch"
        }

# ===========================================================
# 🔌 API ENDPOINT: Language Refine
# ===========================================================
@tracker_routes.post("/api/language_refine")
def language_refine():
    """
    Audyt językowy batcha (bez zapisu do Firestore)
    Returns: metryki + podstawowa korekta typografii
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    clean_text = sanitize_typography(text)
    audit = analyze_language_quality(clean_text)
    
    return jsonify({
        "original_text": text, 
        "auto_fixed_text": clean_text, 
        "language_audit": audit
    })

# ===========================================================
# 🧠 MAIN PROCESS (v12.25.4 - NO AUTO-FIX)
# ===========================================================
def process_batch_in_firestore(project_id: str, batch_text: str, meta_trace: dict = None):
    """
    Główna logika zapisu batcha:
    1. Hard rules check (listy)
    2. Language audit
    3. Keyword tracking (hybrid counter)
    4. Semantic drift check
    5. Transition quality check
    6. Gemini judge (warnings only)
    7. Save to Firestore
    
    v12.25.4: NO AUTO-FIX - user's original text saved
    """
    db = firestore.client()
    doc_ref = db.collection("seo_projects").document(project_id)
    doc = doc_ref.get()
    
    if not doc.exists: 
        return {"error": "Project not found", "status": 404}
    
    batch_text = sanitize_typography(batch_text)
    project_data = doc.to_dict()
    topic = project_data.get("topic", "Nieznany")

    # 1. HARD RULE CHECK (struktura)
    hard_check = validate_hard_rules(batch_text)
    if not hard_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "HARD RULE VIOLATION",
            "gemini_feedback": {
                "feedback_for_writer": hard_check['msg']
            },
            "next_action": "REWRITE"
        }
    
    # 1.5 PARAGRAPH STRUCTURE CHECK (v12.25.6: min 4 sentences, 1 compound)
    para_check = validate_paragraph_structure(batch_text)
    if not para_check["valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "error": "PARAGRAPH STRUCTURE VIOLATION",
            "gemini_feedback": {
                "feedback_for_writer": f"⛔ STRUKTURA AKAPITÓW: {para_check['msg']}"
            },
            "next_action": "REWRITE"
        }

    # 2. LANGUAGE AUDIT
    audit = analyze_language_quality(batch_text)
    warnings = []
    suggestions = []
    
    if audit.get("banned_detected"): 
        warnings.append(f"⛔ Banned phrases: {', '.join(audit['banned_detected'])}")
        suggestions.append("Remove banned phrases")
    
    if audit.get("burstiness", 0) < 6.0:
        warnings.append(f"📊 Burstiness {audit['burstiness']:.1f} (target: >6.0)")
        suggestions.append("Vary sentence length (mix short 5-8 words with long 20-25 words)")
    
    if audit.get("fluff_ratio", 0) > 0.15:
        warnings.append(f"💨 Fluff {audit['fluff_ratio']:.2f} (target: <0.15)")
        suggestions.append("Reduce adjectives/adverbs")
    
    if audit.get("smog_index", 0) > 15:
        warnings.append(f"📖 SMOG {audit['smog_index']:.1f} (target: <15)")
        suggestions.append("Use simpler words and shorter sentences")
    
    if audit.get("readability_score", 100) < 30: 
        warnings.append("📖 Tekst trudny do czytania")
        suggestions.append("Simplify language")

    # 3. SEO TRACKING (Critical Overuse Logic)
    import copy
    keywords_state = copy.deepcopy(project_data.get("keywords_state", {}))
    doc_nlp = nlp(batch_text)
    text_lemma_list = [t.lemma_.lower() for t in doc_nlp if t.is_alpha]
    
    over_limit_hits = []
    critical_reject = []

    for row_id, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        target_max = meta.get("target_max", 5)
        
        # Hybrid counting
        t_exact = meta.get("search_term_exact", kw.lower())
        t_lemma = meta.get("search_lemma", "")
        if not t_lemma: 
            t_lemma = " ".join([t.lemma_.lower() for t in nlp(kw) if t.is_alpha])
        
        found = count_hybrid_occurrences(batch_text, text_lemma_list, t_exact, t_lemma)
        
        if found > 0:
            current_total = meta.get("actual_uses", 0)
            new_total = current_total + found
            
            # HARD CEILING LOGIC (+3 over limit = REJECT)
            if new_total > target_max:
                if new_total >= (target_max + 3):
                    critical_reject.append(f"{kw} ({new_total}/{target_max})")
                else:
                    over_limit_hits.append(kw)

            meta["actual_uses"] = new_total
            meta["status"] = compute_status(new_total, meta["target_min"], target_max)

    # CRITICAL OVERUSE -> WARNING (not REJECT)
    # v12.25.6.2: Changed from REJECT to WARNING - allow batch save with warning
    if critical_reject:
        warnings.append(f"🚨 CRITICAL OVERUSE: {', '.join(critical_reject)}")
        suggestions.append(f"⚠️ Drastycznie przekroczone limity: {', '.join(critical_reject)}. Rozważ usunięcie nadmiarowych wystąpień (ale zostaw min 2x każdego).")

    if over_limit_hits:
        warnings.append(f"📈 Keyword limit exceeded (+1-2): {', '.join(over_limit_hits[:3])}")
        suggestions.append(f"Consider removing 1-2 instances of: {', '.join(over_limit_hits[:3])}")

    # Keyword pacing check
    if audit["sentence_count"] > 0 and (sum(1 for _ in keywords_state) / audit["sentence_count"]) > 0.5:
        warnings.append("🚨 Keyword stuffing detected (too dense)")
        suggestions.append("Spread keywords more naturally throughout text")

    under, over, locked, ok = global_keyword_stats(keywords_state)
    
    # 4. SEMANTIC DRIFT CHECK (v12.25.1)
    drift_check = calculate_semantic_drift(batch_text, project_data.get("batches", []))
    if drift_check["status"] == "DRIFT_WARNING":
        warnings.append(drift_check["message"])
        suggestions.append("Review topic coherence - consider rewriting to stay on-topic")
    
    # 5. TRANSITION QUALITY CHECK (v12.25.1)
    prev_batch = project_data.get("batches", [])[-1] if project_data.get("batches") else None
    transition_check = analyze_transition_quality(batch_text, prev_batch)
    if transition_check["status"] == "CHOPPY":
        warnings.append(transition_check["message"])
        suggestions.append("Add transition word (jednak, ponadto, z kolei) at beginning of batch")
    
    # 6. GEMINI JUDGE (v12.25.4 - NO AUTO-FIX)
    gemini_verdict = evaluate_with_gemini(batch_text, project_data, audit)
    
    # Merge Gemini warnings with our warnings
    if gemini_verdict.get("warnings"):
        warnings.extend(gemini_verdict["warnings"])
    if gemini_verdict.get("suggestions"):
        suggestions.extend(gemini_verdict["suggestions"])

    # 7. SAVE TO FIRESTORE
    # v12.25.4: ALWAYS save original text (no auto-fix applied)
    batch_entry = {
        "text": batch_text,  # ORIGINAL text from user
        "gemini_audit": gemini_verdict,
        "language_audit": audit,
        "semantic_drift": drift_check,
        "transition_quality": transition_check,
        "warnings": warnings,
        "suggestions": suggestions,
        "meta_trace": meta_trace,
        "summary": {"under": under, "over": over, "ok": ok},
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    }
    
    if "batches" not in project_data: 
        project_data["batches"] = []
    project_data["batches"].append(batch_entry)
    project_data["total_batches"] = len(project_data["batches"])
    project_data["keywords_state"] = keywords_state
    doc_ref.set(project_data)

    # STATUS DETERMINATION
    status = "BATCH_WARNING" if warnings else "BATCH_ACCEPTED"
    fb_msg = ("Zapisano z UWAGAMI" if warnings else "Zapisano")
    
    # Top UNDER keywords
    top_under = [
        m.get("keyword") 
        for _, m in sorted(
            keywords_state.items(), 
            key=lambda i: i[1].get("target_min", 0) - i[1].get("actual_uses", 0), 
            reverse=True
        ) 
        if m["status"] == "UNDER"
    ][:5]
    
    meta_summary = f"UNDER={under} | TOP_UNDER={', '.join(top_under)} | {fb_msg}"

    # NEXT ACTION
    next_act = "EXPORT" if under == 0 and len(project_data["batches"]) >= 3 else "GENERATE_NEXT"

    return {
        "status": status,
        "gemini_feedback": {
            "pass": gemini_verdict.get("pass", True),
            "quality_score": gemini_verdict.get("quality_score", 80),
            "feedback_for_writer": gemini_verdict.get("feedback_for_writer", fb_msg)
        },
        "language_audit": {
            "burstiness": audit.get("burstiness"),
            "fluff_ratio": audit.get("fluff_ratio"),
            "smog_index": audit.get("smog_index"),
            "passive_ratio": audit.get("passive_ratio"),
            "sentence_count": audit.get("sentence_count"),
            "lt_errors": audit.get("lt_errors"),
            "banned_detected": audit.get("banned_detected")
        },
        "semantic_drift": drift_check,
        "transition_quality": transition_check,
        "warnings": warnings,
        "suggestions": suggestions,
        "meta_prompt_summary": meta_summary,
        "next_action": next_act
    }
