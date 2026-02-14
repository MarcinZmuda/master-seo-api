import os
import json
import re
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# ================================================================
# 🔥 Firestore Initialization – kompatybilne z Render
# ================================================================
FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
if not FIREBASE_CREDS_JSON:
    raise RuntimeError(
        "❌ Brak zmiennej środowiskowej FIREBASE_CREDS_JSON – "
        "wgraj JSON z Service Account jako string do ENV."
    )

try:
    creds_dict = json.loads(FIREBASE_CREDS_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Niepoprawny JSON w FIREBASE_CREDS_JSON: {e}")

cred = credentials.Certificate(creds_dict)
firebase_app = initialize_app(cred)
db = firestore.client()

# ================================================================
# ⚙️ Flask App Initialization
# ================================================================
app = Flask(__name__)

# 🔧 FIX: Zwiększenie limitu payloadu do 32MB (dla dużych analiz SERP/S1)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

CORS(app)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERSION = "v45.3.1"  # 🆕 AI Middleware, forced mode fix, Anti-Frankenstein enabled

# ================================================================
# 🆕 v32.4: Firestore persistence for projects
# ================================================================
PROJECTS = {}  # Cache w pamięci
PROJECTS_COLLECTION = "seo_projects"  # Ta sama kolekcja co export_routes!


def save_project_to_firestore(project_id: str, data: dict):
    """Zapisuje projekt do Firestore."""
    try:
        doc_ref = db.collection(PROJECTS_COLLECTION).document(project_id)
        doc_ref.set(data, merge=True)
        print(f"[FIRESTORE] ✅ Saved project: {project_id}")
    except Exception as e:
        print(f"[FIRESTORE] ❌ Error saving: {e}")


def load_project_from_firestore(project_id: str) -> dict:
    """Ładuje projekt z Firestore."""
    try:
        doc_ref = db.collection(PROJECTS_COLLECTION).document(project_id)
        doc = doc_ref.get()
        if doc.exists:
            print(f"[FIRESTORE] ✅ Loaded project: {project_id}")
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"[FIRESTORE] ❌ Error loading: {e}")
        return None


def get_project(project_id: str) -> dict:
    """Pobiera projekt z cache lub Firestore."""
    if project_id in PROJECTS:
        return PROJECTS[project_id]
    
    data = load_project_from_firestore(project_id)
    if data:
        PROJECTS[project_id] = data
        return data
    return None


def update_project(project_id: str, data: dict):
    """Aktualizuje projekt w cache i Firestore."""
    if project_id not in PROJECTS:
        PROJECTS[project_id] = {}
    PROJECTS[project_id].update(data)
    save_project_to_firestore(project_id, PROJECTS[project_id])

# ================================================================
# 🧠 Check if semantic analysis is available
# ================================================================
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_ENABLED = True
    print("[MASTER] ✅ Semantic analysis available")
except ImportError:
    SEMANTIC_ENABLED = False
    print("[MASTER] ⚠️ Semantic analysis NOT available (sentence-transformers not installed)")

# ================================================================
# 🆕 v31.0: Semantic Enhancement Integration
# ================================================================
try:
    from unified_validator import (
        validate_content,
        validate_semantic_enhancement,
        full_validate_complete,
        quick_validate,
        calculate_entity_density  # 🆕 v31.3
    )
    SEMANTIC_ENHANCEMENT_ENABLED = True
    print("[MASTER] ✅ Semantic Enhancement v31.0 loaded")
except ImportError as e:
    SEMANTIC_ENHANCEMENT_ENABLED = False
    print(f"[MASTER] ⚠️ Semantic Enhancement not available: {e}")

# ================================================================
# 🆕 v35.5: KEYWORD COUNTER (OVERLAPPING - NeuronWriter compatible)
# ================================================================
try:
    from keyword_counter import count_keywords_for_state, count_keywords
    KEYWORD_COUNTER_ENABLED = True
    print("[MASTER] ✅ Keyword Counter v24.3 loaded (OVERLAPPING mode)")
except ImportError as e:
    KEYWORD_COUNTER_ENABLED = False
    print(f"[MASTER] ⚠️ Keyword Counter not available: {e}")

# ================================================================
# 🆕 v33.0: AI Detection Integration + Humanization
# ================================================================
try:
    from ai_detection_metrics import (
        calculate_humanness_score,
        quick_ai_check,
        check_forbidden_phrases,
        validate_jitter,
        validate_triplets,
        full_ai_detection,
        # Faza 2
        calculate_entity_split,
        calculate_topic_completeness,
        analyze_batch_trend,
        create_batch_record,
        # Faza 3
        score_sentences,
        check_ngram_naturalness,
        full_advanced_analysis,
        # 🆕 v33.0: Humanization helpers
        analyze_sentence_distribution,
        generate_burstiness_fix,
        check_word_repetition_detailed,
        SYNONYM_MAP
    )
    AI_DETECTION_ENABLED = True
    print("[MASTER] ✅ AI Detection v33.0 + Humanization loaded")
except ImportError as e:
    AI_DETECTION_ENABLED = False
    print(f"[MASTER] ⚠️ AI Detection not available: {e}")

# 🆕 v41.0: Sentence Splitter (dzielenie długich zdań)
SENTENCE_SPLITTER_AVAILABLE = False
try:
    from dynamic_humanization import suggest_sentence_splits, split_long_sentences
    SENTENCE_SPLITTER_AVAILABLE = True
    print("[MASTER] ✅ Sentence Splitter v41.0 loaded")
except ImportError as e:
    print(f"[MASTER] ⚠️ Sentence Splitter not available: {e}")
    def suggest_sentence_splits(text, **kwargs):
        return []
    def split_long_sentences(text, **kwargs):
        return {"modified_text": text, "splits": [], "split_count": 0, "stats": {}, "before_after": []}

# ================================================================
# 🆕 v34.0: Legal Module (SAOS Integration)
# ================================================================
try:
    from legal_routes_v3 import (
        legal_routes,
        enhance_project_with_legal,
        check_legal_on_export,
        LEGAL_MODULE_ENABLED as LEGAL_ROUTES_ENABLED
    )
    LEGAL_MODULE_ENABLED = LEGAL_ROUTES_ENABLED
    print("[MASTER] ✅ Legal Module v3.0 loaded (SAOS integration)")
except ImportError as e:
    LEGAL_MODULE_ENABLED = False
    legal_routes = None
    print(f"[MASTER] ⚠️ Legal Module not available: {e}")

# ================================================================
# 🆕 v37.0: Medical Module (PubMed, ClinicalTrials, Polish Health)
# ================================================================
try:
    from medical_module.medical_routes import (
        medical_routes,
        enhance_project_with_medical,
        check_medical_on_export
    )
    from medical_module.medical_module import (
        PUBMED_AVAILABLE,
        CLINICALTRIALS_AVAILABLE,
        POLISH_HEALTH_AVAILABLE,
        CLAUDE_VERIFIER_AVAILABLE
    )
    MEDICAL_MODULE_ENABLED = True
    print("[MASTER] ✅ Medical Module v1.1 loaded")
    print(f"[MASTER]    ├─ PubMed: {'✅' if PUBMED_AVAILABLE else '❌'}")
    print(f"[MASTER]    ├─ ClinicalTrials: {'✅' if CLINICALTRIALS_AVAILABLE else '❌'}")
    print(f"[MASTER]    ├─ Polish Health: {'✅' if POLISH_HEALTH_AVAILABLE else '❌'}")
    print(f"[MASTER]    └─ Claude Verifier: {'✅' if CLAUDE_VERIFIER_AVAILABLE else '❌'}")
except Exception as e:
    # Łap WSZYSTKIE błędy, nie tylko ImportError
    import traceback
    MEDICAL_MODULE_ENABLED = False
    PUBMED_AVAILABLE = False
    CLINICALTRIALS_AVAILABLE = False
    POLISH_HEALTH_AVAILABLE = False
    CLAUDE_VERIFIER_AVAILABLE = False
    medical_routes = None
    print(f"[MASTER] ❌ Medical Module FAILED to load!")
    print(f"[MASTER]    Error type: {type(e).__name__}")
    print(f"[MASTER]    Error message: {e}")
    print(f"[MASTER]    Full traceback:")
    traceback.print_exc()

# ================================================================
# 🔗 N-gram API Configuration (for S1 proxy)
# ================================================================
NGRAM_API_URL = os.getenv("NGRAM_API_URL", "https://gpt-ngram-api.onrender.com")

# Sprawdź czy URL już zawiera endpoint
if "/api/ngram_entity_analysis" in NGRAM_API_URL:
    NGRAM_BASE_URL = NGRAM_API_URL.replace("/api/ngram_entity_analysis", "")
    NGRAM_ANALYSIS_ENDPOINT = NGRAM_API_URL
    print(f"[MASTER] 🔗 N-gram API URL (full endpoint detected): {NGRAM_ANALYSIS_ENDPOINT}")
else:
    NGRAM_BASE_URL = NGRAM_API_URL
    NGRAM_ANALYSIS_ENDPOINT = f"{NGRAM_API_URL}/api/ngram_entity_analysis"
    print(f"[MASTER] 🔗 N-gram API URL (base URL): {NGRAM_BASE_URL}")

print(f"[MASTER] 🎯 S1 Analysis endpoint: {NGRAM_ANALYSIS_ENDPOINT}")

# ================================================================
# 📦 Import blueprintów (po inicjalizacji Firestore)
# ================================================================
from project_routes import project_routes
from firestore_tracker_routes import tracker_routes
from seo_optimizer import unified_prevalidation
from final_review_routes import final_review_routes
from paa_routes import paa_routes
from export_routes import export_routes  # v23.9: Eksport DOCX/HTML/TXT + Editorial Review

# 🆕 v44.2: H2 routes (wydzielone z project_routes)
try:
    from project_helpers import h2_routes, H2_ROUTES_AVAILABLE
    print("[MASTER_API] ✅ H2 routes loaded from project_helpers")
except ImportError as e:
    H2_ROUTES_AVAILABLE = False
    h2_routes = None
    print(f"[MASTER_API] ⚠️ H2 routes not available: {e}")

# v29.3: Entity SEO routes
try:
    from entity_routes import entity_routes
    ENTITY_ROUTES_ENABLED = True
    print("[MASTER_API] ✅ Entity routes loaded")
except ImportError as e:
    ENTITY_ROUTES_ENABLED = False
    entity_routes = None
    print(f"[MASTER_API] ⚠️ Entity routes not available: {e}")

# ================================================================
# 🔗 Rejestracja blueprintów
# ================================================================
app.register_blueprint(project_routes)
app.register_blueprint(tracker_routes)
app.register_blueprint(final_review_routes)
app.register_blueprint(paa_routes)
app.register_blueprint(export_routes)  # v23.9: Eksport + Editorial Review

# 🆕 v44.2: H2 routes (wydzielone z project_routes)
if H2_ROUTES_AVAILABLE and h2_routes:
    app.register_blueprint(h2_routes)
    print("[MASTER_API] ✅ H2 routes registered")

# v29.3: Entity routes (jeśli dostępne)
if ENTITY_ROUTES_ENABLED and entity_routes:
    app.register_blueprint(entity_routes)
    print("[MASTER_API] ✅ Entity routes registered")

# 🆕 v34.0: Legal Module routes (jeśli dostępne)
if LEGAL_MODULE_ENABLED and legal_routes:
    app.register_blueprint(legal_routes)
    print("[MASTER_API] ✅ Legal routes registered (SAOS integration)")

# 🆕 v37.0: Medical Module routes (jeśli dostępne)
if MEDICAL_MODULE_ENABLED and medical_routes:
    app.register_blueprint(medical_routes)
    print("[MASTER_API] ✅ Medical routes registered (PubMed, ClinicalTrials, Polish Health)")

# 🆕 v47.2: Unified YMYL Classifier (Claude-based, replaces keyword detection)
try:
    from ymyl.ymyl_unified_classifier import register_routes as register_ymyl_routes
    register_ymyl_routes(app)
    print("[MASTER_API] ✅ Unified YMYL classifier registered (/api/ymyl/detect_and_enrich)")
except ImportError as e:
    print(f"[MASTER_API] ⚠️ Unified YMYL classifier not available: {e}")

# ================================================================
# 🔗 S1 PROXY ENDPOINTS (przekierowanie do N-gram API)
# ================================================================
@app.post("/api/s1_analysis")
def s1_analysis_proxy():
    """
    Proxy endpoint dla S1 analysis.
    v31.0: + semantic_enhancement data
    """
    data = request.get_json(force=True)
    
    # v27.0: Normalizuj nazwę parametru - N-gram API oczekuje "main_keyword"
    if "keyword" in data and "main_keyword" not in data:
        data["main_keyword"] = data["keyword"]
    if "main_keyword" not in data:
        return jsonify({"error": "Required: keyword or main_keyword"}), 400
    
    # v23.9: Domyślnie 6 stron zamiast 30
    if "max_urls" not in data:
        data["max_urls"] = 6
    if "top_results" not in data:
        data["top_results"] = 6
    
    keyword = data.get("main_keyword", "")
    print(f"[S1_PROXY] 📡 Forwarding S1 analysis for '{keyword}' to {NGRAM_ANALYSIS_ENDPOINT}")
    print(f"[S1_PROXY] 📦 Request body: main_keyword='{keyword}', keys={list(data.keys())}")
    
    try:
        # v27.0: Explicit UTF-8 encoding
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json=data,
            timeout=90,
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            }
        )
        
        print(f"[S1_PROXY] 📬 Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "No error message"
            print(f"[S1_PROXY] ❌ N-gram API error {response.status_code}: {error_text}")
            return jsonify({
                "error": "N-gram API error",
                "status_code": response.status_code,
                "details": error_text,
                "sent_keyword": keyword,
                "sent_keys": list(data.keys())
            }), response.status_code
        
        result = response.json()
        print(f"[S1_PROXY] ✅ S1 analysis completed successfully")
        
        # =============================================================
        # v27.0: AUTOMATYCZNE OBLICZANIE recommended_length
        # =============================================================
        word_counts = []
        
        # Szukaj word_count w danych konkurencji
        serp_data = result.get("serp_analysis", {}) or {}
        competitors = serp_data.get("competitors", []) or result.get("competitors", []) or []
        
        for comp in competitors:
            if isinstance(comp, dict):
                wc = comp.get("word_count") or comp.get("wordCount") or comp.get("content_length", 0)
                if wc and wc > 100:
                    word_counts.append(wc)
        
        # Heurystyka jeśli brak danych word_count
        if not word_counts:
            ngrams_count = len(result.get("ngrams", []) or result.get("hybrid_ngrams", []) or [])
            h2_count = len(serp_data.get("competitor_h2_patterns", []) or [])
            
            if ngrams_count > 50 or h2_count > 15:
                estimated = 4000
            elif ngrams_count > 30 or h2_count > 10:
                estimated = 3000
            elif ngrams_count > 15 or h2_count > 5:
                estimated = 2000
            else:
                estimated = 1500
            
            word_counts = [estimated]
            print(f"[S1_PROXY] ℹ️ No word_count data, estimated: {estimated} (ngrams={ngrams_count}, h2={h2_count})")
        
        # Oblicz statystyki
        word_counts.sort()
        n = len(word_counts)
        
        if n > 0:
            median = word_counts[n // 2] if n % 2 == 1 else (word_counts[n // 2 - 1] + word_counts[n // 2]) // 2
            avg = sum(word_counts) // n
            recommended = int(median * 1.1)
            recommended = round(recommended / 100) * 100
            recommended = max(1000, min(6000, recommended))
        else:
            median = 3000
            avg = 3000
            recommended = 3000
        
        result["recommended_length"] = recommended
        result["length_analysis"] = {
            "word_counts": word_counts,
            "median": median,
            "average": avg,
            "recommended": recommended,
            "analyzed_urls": len(word_counts),
            "note": "Rekomendacja = mediana + 10%, zaokrąglone do 100"
        }
        
        print(f"[S1_PROXY] 📏 Length analysis: median={median}, recommended={recommended} words")
        
        # Semantic analysis
        if SEMANTIC_ENABLED:
            try:
                from seo_optimizer import semantic_keyword_coverage
                
                if "keywords" in result:
                    sample_text = ""
                    ft = result.get("full_text_sample") or result.get("full_text_content") or ""
                    if isinstance(ft, str) and ft.strip():
                        sample_text = ft
                    else:
                        parts = []
                        fs = serp_data.get("featured_snippet")
                        if isinstance(fs, dict):
                            for k in ("snippet", "text", "answer"):
                                v = fs.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                        elif isinstance(fs, str) and fs.strip():
                            parts.append(fs.strip())
                        
                        paa = serp_data.get("paa_questions", [])
                        if isinstance(paa, list):
                            for item in paa:
                                if isinstance(item, dict):
                                    q = item.get("question") or item.get("q")
                                    a = item.get("answer") or item.get("snippet") or item.get("a")
                                    if isinstance(q, str) and q.strip():
                                        parts.append(q.strip())
                                    if isinstance(a, str) and a.strip():
                                        parts.append(a.strip())
                                elif isinstance(item, str) and item.strip():
                                    parts.append(item.strip())
                        
                        sample_text = "\n".join(parts)
                    
                    sample_text = (sample_text or "")[:5000]
                    
                    if sample_text.strip():
                        dummy_kw_state = {
                            str(i): {"keyword": kw, "actual_uses": 0}
                            for i, kw in enumerate(result.get("keywords", []))
                        }
                        semantic_cov = semantic_keyword_coverage(sample_text, dummy_kw_state)
                        result["semantic_analysis"] = semantic_cov
                        print(f"[S1_PROXY] ✅ Added semantic analysis to S1 result")
            except Exception as e:
                print(f"[S1_PROXY] ⚠️ Semantic analysis failed: {e}")
        
        # v29.3: Enhanced entity_seo response
        try:
            entity_seo = result.get("entity_seo", {})
            entities = entity_seo.get("entities", [])
            
            # Dodaj must_mention_entities (encje obecne u 80%+ konkurencji)
            if entities and "must_mention_entities" not in entity_seo:
                # Top 5 encji z najwyższą importance
                top_entities = sorted(
                    entities, 
                    key=lambda x: x.get("importance", 0) if isinstance(x, dict) else 0, 
                    reverse=True
                )[:5]
                must_mention = [
                    e.get("name", str(e)) if isinstance(e, dict) else str(e)
                    for e in top_entities
                ]
                entity_seo["must_mention_entities"] = must_mention
            
            # Dodaj synonyms do ngrams
            ngrams = result.get("ngrams", [])
            if ngrams and "synonyms" not in result:
                # Podstawowe synonimy
                BASIC_SYNONYMS = {
                    "pomoce sensoryczne": ["narzędzia terapeutyczne", "sprzęt SI", "akcesoria sensoryczne"],
                    "integracja sensoryczna": ["SI", "terapia SI", "przetwarzanie sensoryczne"],
                    "dziecko": ["maluch", "przedszkolak", "najmłodsi"],
                    "rozwój": ["postęp", "kształtowanie", "doskonalenie"]
                }
                
                result["synonyms"] = {}
                main_kw_lower = keyword.lower()
                for key, syns in BASIC_SYNONYMS.items():
                    if key in main_kw_lower or main_kw_lower in key:
                        result["synonyms"][keyword] = syns
                        break
            
            result["entity_seo"] = entity_seo
            
            # 🔧 v32.5: Propaguj entity_relationships na główny poziom (dla kompatybilności z GPT)
            relationships = entity_seo.get("entity_relationships", [])
            if relationships:
                result["entity_relationships"] = relationships
                print(f"[S1_PROXY] ✅ Propagated {len(relationships)} entity_relationships to top level")
            
            # 🆕 v32.5: Propaguj entities i topical_coverage na główny poziom
            entities = entity_seo.get("entities", [])
            if entities:
                result["entities"] = entities
                print(f"[S1_PROXY] ✅ Propagated {len(entities)} entities to top level")
            
            topical = entity_seo.get("topical_coverage", [])
            if topical:
                result["topics"] = topical  # alias dla GPT
                result["topical_coverage"] = topical
                print(f"[S1_PROXY] ✅ Propagated {len(topical)} topics to top level")
            
            # 🆕 v46.0: Propaguj concept_entities i topical_summary
            concept_entities = entity_seo.get("concept_entities", [])
            if concept_entities:
                result["concept_entities"] = concept_entities
                print(f"[S1_PROXY] ✅ Propagated {len(concept_entities)} concept entities to top level")
            
            topical_summary = entity_seo.get("topical_summary", {})
            if topical_summary and topical_summary.get("status") == "OK":
                result["topical_summary"] = topical_summary
                print(f"[S1_PROXY] ✅ Propagated topical summary "
                      f"(must_cover: {topical_summary.get('must_cover_count', 0)}, "
                      f"should_cover: {topical_summary.get('should_cover_count', 0)})")
            
            # 🆕 v47.0: Propaguj salience, co-occurrence i placement instructions
            entity_salience = entity_seo.get("entity_salience", [])
            if entity_salience:
                result["entity_salience"] = entity_salience
                print(f"[S1_PROXY] ✅ Propagated {len(entity_salience)} entity salience scores")
            
            entity_cooccurrence = entity_seo.get("entity_cooccurrence", [])
            if entity_cooccurrence:
                result["entity_cooccurrence"] = entity_cooccurrence
                print(f"[S1_PROXY] ✅ Propagated {len(entity_cooccurrence)} co-occurrence pairs")
            
            entity_placement = entity_seo.get("entity_placement", {})
            if entity_placement and entity_placement.get("status") == "OK":
                result["entity_placement"] = entity_placement
                print(f"[S1_PROXY] ✅ Propagated entity placement instructions")
            
            print(f"[S1_PROXY] ✅ Enhanced entity_seo with must_mention_entities")
            
        except Exception as e:
            print(f"[S1_PROXY] ⚠️ Entity enhancement failed: {e}")
        
        # =============================================================
        # 🆕 v31.0: Add semantic_enhancement hints to S1 response
        # =============================================================
        if SEMANTIC_ENHANCEMENT_ENABLED:
            try:
                # Przygotuj dane dla GPT - co ma sprawdzać w każdym batchu
                entity_seo = result.get("entity_seo", {})
                entities = entity_seo.get("entities", [])
                topical_coverage = result.get("topical_coverage", [])
                
                # Critical entities (importance >= 0.7, sources >= 4)
                critical_entities = [
                    {"text": e.get("text") or e.get("name"), "type": e.get("type"), "importance": e.get("importance")}
                    for e in entities
                    if isinstance(e, dict) and e.get("importance", 0) >= 0.7 and e.get("sources_count", 0) >= 4
                ][:5]
                
                # High priority entities (importance >= 0.5, sources >= 2)
                high_entities = [
                    {"text": e.get("text") or e.get("name"), "type": e.get("type"), "importance": e.get("importance")}
                    for e in entities
                    if isinstance(e, dict) and e.get("importance", 0) >= 0.5 and e.get("sources_count", 0) >= 2
                    and e not in critical_entities
                ][:5]
                
                # Must topics
                must_topics = [
                    {"topic": t.get("subtopic"), "sample_h2": t.get("sample_h2")}
                    for t in topical_coverage
                    if isinstance(t, dict) and t.get("priority") == "MUST"
                ][:5]
                
                result["semantic_enhancement_hints"] = {
                    "critical_entities": critical_entities,
                    "high_entities": high_entities,
                    "must_topics": must_topics,
                    "must_cover_concepts": (
                        result.get("topical_summary", {}).get("must_cover", [])[:10]
                    ),
                    "concept_instruction": (
                        result.get("topical_summary", {}).get("agent_instruction", "")
                    ),
                    "placement_instruction": (
                        result.get("entity_placement", {}).get("placement_instruction", "")
                    ),
                    "primary_entity": (
                        result.get("entity_placement", {}).get("primary_entity", {})
                    ),
                    "cooccurrence_pairs": (
                        result.get("entity_placement", {}).get("cooccurrence_pairs", [])[:5]
                    ),
                    "first_paragraph_entities": (
                        result.get("entity_placement", {}).get("first_paragraph_entities", [])
                    ),
                    "h2_entities": (
                        result.get("entity_placement", {}).get("h2_entities", [])
                    ),
                    "checkpoints": {
                        "batch_1": "H1 contains primary entity, first paragraph has primary + 2 secondary entities",
                        "batch_3": "entity_density >= 2.5, min 50% critical entities, min 30% must_cover_concepts, co-occurring pairs in same paragraphs",
                        "batch_5": "topic_completeness >= 50%, source_effort signals, concept coverage >= 50%, all E-A-V triples described",
                        "pre_faq": "all critical entities, all MUST topics, all must_cover_concepts, H2s contain secondary entities"
                    },
                    "version": "v47.0"
                }
                
                print(f"[S1_PROXY] ✅ Added semantic_enhancement_hints (critical={len(critical_entities)}, high={len(high_entities)}, must_topics={len(must_topics)})")
                
            except Exception as e:
                print(f"[S1_PROXY] ⚠️ Semantic enhancement hints failed: {e}")
        
        return jsonify(result), 200
        
    except requests.exceptions.Timeout:
        print(f"[S1_PROXY] ⏱️ Timeout after 90s")
        return jsonify({
            "error": "N-gram API timeout",
            "message": "SERP analysis took too long (>90s). Try with fewer sources."
        }), 504
        
    except requests.exceptions.ConnectionError:
        print(f"[S1_PROXY] ❌ Connection error to {NGRAM_ANALYSIS_ENDPOINT}")
        return jsonify({
            "error": "Cannot connect to N-gram API",
            "ngram_api_url": NGRAM_ANALYSIS_ENDPOINT,
            "message": "Check if N-gram API service is running"
        }), 503
        
    except Exception as e:
        print(f"[S1_PROXY] ❌ Unexpected error: {e}")
        return jsonify({
            "error": "S1 proxy error",
            "message": str(e)
        }), 500


# ================================================================
# 🆕 v31.0: SEMANTIC VALIDATION ENDPOINT
# ================================================================
@app.post("/api/semantic_validate")
def semantic_validate():
    """
    🆕 v31.0: Walidacja semantyczna treści.
    
    Sprawdza:
    - Entity Density (gęstość encji)
    - Topic Completeness (kompletność tematyczna vs S1)
    - Entity Gap (brakujące encje vs konkurencja)
    - Source Effort (sygnały wysiłku badawczego)
    
    Request body:
    {
        "text": "treść do walidacji",
        "s1_data": {...},  // opcjonalne - dane z S1
        "entities": [...],  // opcjonalne - wykryte encje
        "keywords_state": {...},  // opcjonalne - dla full validation
        "main_keyword": "...",  // opcjonalne
        "mode": "semantic" | "full"  // domyślnie "semantic"
    }
    """
    if not SEMANTIC_ENHANCEMENT_ENABLED:
        return jsonify({
            "error": "Semantic Enhancement not available",
            "message": "unified_validator module not loaded"
        }), 503
    
    data = request.get_json(force=True)
    
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"]
    s1_data = data.get("s1_data", {})
    entities = data.get("entities", [])
    mode = data.get("mode", "semantic")
    
    try:
        if mode == "full":
            # Pełna walidacja SEO + Semantic
            keywords_state = data.get("keywords_state", {})
            main_keyword = data.get("main_keyword", "")
            ngrams = data.get("ngrams", [])
            
            result = full_validate_complete(
                text=text,
                keywords_state=keywords_state,
                main_keyword=main_keyword,
                ngrams=ngrams,
                s1_data=s1_data,
                detected_entities=entities
            )
        else:
            # Tylko semantic validation
            result = validate_semantic_enhancement(
                content=text,
                s1_data=s1_data,
                detected_entities=entities
            )
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[SEMANTIC_VALIDATE] ❌ Error: {e}")
        return jsonify({
            "error": "Validation failed",
            "message": str(e)
        }), 500


# ================================================================
# 🆕 v31.0: QUICK SEMANTIC CHECK (for GPT pre-batch)
# ================================================================
@app.post("/api/quick_semantic_check")
def quick_semantic_check():
    """
    🆕 v31.0: Szybka walidacja semantyczna dla GPT.
    
    Zwraca tylko najważniejsze metryki:
    - entity_density_ok (bool)
    - generics_found (list)
    - source_effort_score (float)
    - quick_wins (list)
    """
    if not SEMANTIC_ENHANCEMENT_ENABLED:
        return jsonify({"status": "unavailable"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if not text or len(text) < 100:
        return jsonify({"status": "text_too_short"}), 400
    
    try:
        result = validate_semantic_enhancement(content=text)
        
        # Zwróć uproszczony wynik
        density = result.get("analyses", {}).get("entity_density", {})
        effort = result.get("analyses", {}).get("source_effort", {})
        
        return jsonify({
            "status": result.get("status", "UNKNOWN"),
            "semantic_score": result.get("semantic_score", 0),
            "entity_density_ok": density.get("status") == "GOOD",
            "generics_found": density.get("generics_found", [])[:3],
            "source_effort_score": effort.get("score", 0),
            "quick_wins": result.get("quick_wins", [])[:3]
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/synthesize_topics")
def synthesize_topics_proxy():
    """Proxy dla synthesize_topics."""
    data = request.get_json(force=True)

    if isinstance(data, dict):
        ngrams = data.get("ngrams")
        if isinstance(ngrams, list) and ngrams and isinstance(ngrams[0], dict):
            data["ngrams"] = [x.get("ngram", "") for x in ngrams if isinstance(x, dict) and x.get("ngram")]
    
    try:
        response = requests.post(
            f"{NGRAM_BASE_URL}/api/synthesize_topics",
            json=data,
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# 🆕 v23.9: ANALYZE SERP LENGTH - mediana długości z konkurencji
# ============================================================================
@app.post("/api/analyze_serp_length")
def analyze_serp_length():
    """
    Analizuje długość artykułów konkurencji i zwraca rekomendowaną długość.
    """
    data = request.get_json(force=True)
    keyword = data.get("keyword") or data.get("main_keyword", "")
    
    if not keyword:
        return jsonify({"error": "Missing keyword"}), 400
    
    try:
        response = requests.post(
            NGRAM_ANALYSIS_ENDPOINT,
            json={"keyword": keyword, "max_urls": 6, "top_results": 6},
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            return jsonify({
                "keyword": keyword,
                "recommended_length": 3000,
                "source": "default",
                "message": "Could not analyze SERP, using default 3000 words"
            }), 200
        
        result = response.json()
        word_counts = []
        
        serp_data = result.get("serp_analysis", {}) or {}
        competitors = serp_data.get("competitors", []) or result.get("competitors", []) or []
        
        for comp in competitors:
            if isinstance(comp, dict):
                wc = comp.get("word_count") or comp.get("wordCount") or comp.get("content_length", 0)
                if wc and wc > 100:
                    word_counts.append(wc)
        
        if not word_counts:
            ngrams_count = len(result.get("ngrams", []) or result.get("hybrid_ngrams", []) or [])
            estimated = max(2000, min(5000, 2000 + ngrams_count * 20))
            word_counts = [estimated]
        
        word_counts.sort()
        n = len(word_counts)
        
        if n == 0:
            median = 3000
        elif n % 2 == 0:
            median = (word_counts[n//2 - 1] + word_counts[n//2]) // 2
        else:
            median = word_counts[n//2]
        
        avg = sum(word_counts) // n if n > 0 else 3000
        min_wc = min(word_counts) if word_counts else 2000
        max_wc = max(word_counts) if word_counts else 4000
        
        recommended = int(median * 1.1)
        recommended = max(1500, min(6000, recommended))
        
        return jsonify({
            "keyword": keyword,
            "analyzed_competitors": n,
            "word_counts": word_counts,
            "statistics": {
                "median": median,
                "average": avg,
                "min": min_wc,
                "max": max_wc
            },
            "recommended_length": recommended,
            "source": "serp_analysis",
            "note": f"Rekomendowana długość {recommended} słów (mediana konkurencji + 10%)"
        }), 200
        
    except Exception as e:
        print(f"[SERP_LENGTH] ❌ Error: {e}")
        return jsonify({
            "keyword": keyword,
            "recommended_length": 3000,
            "source": "fallback",
            "error": str(e)
        }), 200


@app.post("/api/generate_compliance_report")
def compliance_report_proxy():
    """Proxy dla generate_compliance_report."""
    data = request.get_json(force=True)
    
    try:
        response = requests.post(
            f"{NGRAM_BASE_URL}/api/generate_compliance_report",
            json=data,
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/s1_health")
def s1_health_check():
    """Sprawdza czy N-gram API service jest dostępny."""
    try:
        response = requests.get(f"{NGRAM_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            ngram_status = response.json()
            return jsonify({
                "status": "ok",
                "ngram_api_status": ngram_status,
                "ngram_base_url": NGRAM_BASE_URL,
                "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
                "proxy_enabled": True,
                "semantic_enabled": SEMANTIC_ENABLED,
                "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED
            }), 200
        else:
            return jsonify({
                "status": "degraded",
                "ngram_api_status": "error",
                "ngram_base_url": NGRAM_BASE_URL,
                "proxy_enabled": True,
                "semantic_enabled": SEMANTIC_ENABLED,
                "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED
            }), 200
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e),
            "ngram_base_url": NGRAM_BASE_URL,
            "proxy_enabled": True,
            "semantic_enabled": SEMANTIC_ENABLED,
            "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED
        }), 503


# ================================================================
# 🧠 MASTER DEBUG ROUTES (diagnostyka)
# ================================================================
@app.get("/api/master_debug/<project_id>")
def master_debug(project_id):
    """Pełna diagnostyka projektu."""
    doc = db.collection("seo_projects").document(project_id).get()
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404

    data = doc.to_dict()
    keywords = data.get("keywords_state", {})
    batches = data.get("batches", [])
    total_batches = len(batches)

    all_warnings = []
    for b in batches:
        warns = b.get("warnings", [])
        if warns:
            all_warnings.extend(warns)

    semantic_scores = [
        b.get("language_audit", {}).get("semantic_score")
        for b in batches if b.get("language_audit")
    ]
    avg_semantic = (
        round(sum([s for s in semantic_scores if s]) / len(semantic_scores), 3)
        if semantic_scores else 0
    )

    return jsonify({
        "project_id": project_id,
        "topic": data.get("topic"),
        "total_batches": total_batches,
        "keywords_count": len(keywords),
        "warnings_total": len(all_warnings),
        "avg_semantic_score": avg_semantic,
        "avg_density": round(
            sum([b.get("language_audit", {}).get("density", 0) for b in batches]) / max(1, total_batches), 2
        ),
        "burstiness_avg": round(
            sum([b.get("burstiness", 0) for b in batches]) / max(1, total_batches), 2
        ),
        "last_update": batches[-1]["timestamp"].isoformat() if batches else None,
        "lsi_keywords": data.get("lsi_enrichment", {}).get("count", 0),
        "has_final_review": "final_review" in data,
        "semantic_enabled": SEMANTIC_ENABLED,
        "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED
    }), 200


# ================================================================
# 🚨 ERROR HANDLERS (Globalne)
# ================================================================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request Entity Too Large", "message": "Payload przekracza 32MB"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500


# ================================================================
# 🆕 v35.5: VERIFY KEYWORDS ENDPOINT (dla GPT do sprawdzenia w locie)
# ================================================================
@app.post("/api/verify_keywords")
def verify_keywords():
    """
    🆕 v35.5: Endpoint do weryfikacji użycia fraz w tekście.
    
    GPT może wywołać to w trakcie pisania aby sprawdzić ile razy użył każdej frazy.
    
    Request body:
    {
        "text": "tekst do sprawdzenia",
        "keywords": {
            "sąd": {"target_min": 5, "target_max": 9},
            "demencja": {"target_min": 1, "target_max": 3},
            ...
        }
    }
    
    Returns:
    {
        "counts": {"sąd": 7, "demencja": 2, ...},
        "status": {"sąd": "OK", "demencja": "OK", ...},
        "warnings": [...],
        "blockers": [...]
    }
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    keywords = data.get("keywords", {})
    
    if not text:
        return jsonify({"error": "Missing text"}), 400
    if not keywords:
        return jsonify({"error": "Missing keywords"}), 400
    
    # Przelicz
    counts = count_keywords_from_text(text, keywords)
    
    # Oceń status
    status = {}
    warnings = []
    blockers = []
    
    for phrase, meta in keywords.items():
        count = counts.get(phrase, 0)
        if isinstance(meta, dict):
            target_min = meta.get("target_min", 1)
            target_max = meta.get("target_max", 999)
        else:
            target_min = 1
            target_max = 999
        
        if count > target_max:
            status[phrase] = "OVER"
            blockers.append({
                "phrase": phrase,
                "count": count,
                "max": target_max,
                "message": f"🔴 '{phrase}' = {count}× (max {target_max})"
            })
        elif count == 0:
            status[phrase] = "MISSING"
            warnings.append({
                "phrase": phrase,
                "count": 0,
                "min": target_min,
                "message": f"⚠️ '{phrase}' = 0× (min {target_min})"
            })
        elif count < target_min:
            status[phrase] = "UNDER"
            warnings.append({
                "phrase": phrase,
                "count": count,
                "min": target_min,
                "message": f"⚠️ '{phrase}' = {count}× (min {target_min})"
            })
        elif count >= target_max:
            status[phrase] = "LOCKED"
        else:
            status[phrase] = "OK"
    
    return jsonify({
        "counts": counts,
        "status": status,
        "warnings": warnings,
        "blockers": blockers,
        "has_blockers": len(blockers) > 0,
        "counting_method": "OVERLAPPING (NeuronWriter compatible)",
        "word_count": len(text.split())
    }), 200


# ================================================================
# 🔧 MEDICAL MODULE DEBUG ENDPOINT
# ================================================================
@app.get("/api/debug/medical")
def debug_medical_module():
    """
    Endpoint diagnostyczny dla modułu medycznego.
    Sprawdza szczegóły dlaczego moduł może nie działać.
    """
    import os
    import sys
    
    debug_info = {
        "medical_module_enabled": MEDICAL_MODULE_ENABLED,
        "checks": {}
    }
    
    # 1. Sprawdź czy folder istnieje
    module_path = os.path.join(os.path.dirname(__file__), 'medical_module')
    debug_info["checks"]["folder_exists"] = os.path.exists(module_path)
    debug_info["checks"]["folder_path"] = module_path
    
    # 2. Sprawdź pliki w folderze
    if os.path.exists(module_path):
        files = os.listdir(module_path)
        debug_info["checks"]["files_in_folder"] = files
        debug_info["checks"]["__init__.py_exists"] = "__init__.py" in files
        debug_info["checks"]["medical_routes.py_exists"] = "medical_routes.py" in files
        debug_info["checks"]["medical_module.py_exists"] = "medical_module.py" in files
    else:
        debug_info["checks"]["files_in_folder"] = []
        debug_info["checks"]["error"] = "Folder medical_module/ nie istnieje!"
    
    # 3. Sprawdź czy moduł jest w sys.modules
    debug_info["checks"]["in_sys_modules"] = "medical_module" in sys.modules
    
    # 4. Spróbuj zaimportować i złap błąd
    if not MEDICAL_MODULE_ENABLED:
        try:
            from medical_module.medical_routes import medical_routes as test_routes
            debug_info["checks"]["import_test"] = "SUCCESS"
        except Exception as e:
            import traceback
            debug_info["checks"]["import_test"] = "FAILED"
            debug_info["checks"]["import_error_type"] = type(e).__name__
            debug_info["checks"]["import_error_message"] = str(e)
            debug_info["checks"]["import_traceback"] = traceback.format_exc()
    else:
        debug_info["checks"]["import_test"] = "ALREADY_LOADED"
    
    # 5. Zmienne środowiskowe
    debug_info["env"] = {
        "NCBI_API_KEY": "SET" if os.getenv("NCBI_API_KEY") else "NOT_SET",
        "NCBI_EMAIL": "SET" if os.getenv("NCBI_EMAIL") else "NOT_SET",
        "ANTHROPIC_API_KEY": "SET" if os.getenv("ANTHROPIC_API_KEY") else "NOT_SET"
    }
    
    # 6. Python path
    debug_info["python_path"] = sys.path[:5]  # Pierwsze 5 ścieżek
    
    return jsonify(debug_info), 200


# ================================================================
# 🏥 HEALTHCHECK
# ================================================================
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Master SEO API działa",
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "modules": [
            "project_routes",
            "firestore_tracker_routes",
            "final_review_routes",
            "paa_routes",
            "export_routes",
            "seo_optimizer",
            "s1_proxy (to N-gram API)",
            "semantic_enhancement v31.0" if SEMANTIC_ENHANCEMENT_ENABLED else "semantic_enhancement (disabled)",
            "legal_module v3.0" if LEGAL_MODULE_ENABLED else "legal_module (disabled)",
            "medical_module v1.0" if MEDICAL_MODULE_ENABLED else "medical_module (disabled)"
        ],
        "debug_mode": DEBUG_MODE,
        "firebase_connected": True,
        "ngram_base_url": NGRAM_BASE_URL,
        "ngram_analysis_endpoint": NGRAM_ANALYSIS_ENDPOINT,
        "s1_proxy_enabled": True,
        "semantic_enabled": SEMANTIC_ENABLED,
        "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED,
        "legal_module_enabled": LEGAL_MODULE_ENABLED,
        "medical_module_enabled": MEDICAL_MODULE_ENABLED,
        "medical_sources": {
            "pubmed": PUBMED_AVAILABLE,
            "clinicaltrials": CLINICALTRIALS_AVAILABLE,
            "polish_health": POLISH_HEALTH_AVAILABLE,
            "claude_verifier": CLAUDE_VERIFIER_AVAILABLE
        } if MEDICAL_MODULE_ENABLED else None,
        "features_v37_0": [
            "Medical Module v1.0",
            "PubMed NCBI E-utilities integration",
            "ClinicalTrials.gov API v2",
            "Polish Health Sources (PZH, AOTMiT, MZ, NFZ)",
            "Claude AI medical evidence verification",
            "NLM/APA citation formatting",
            "/api/medical/* endpoints"
        ]
    }), 200


# ================================================================
# 🔎 VERSION CHECK
# ================================================================
@app.get("/api/version")
def version_info():
    return jsonify({
        "engine": "BRAJEN SEO Engine",
        "api_version": VERSION,
        "components": {
            "project_routes": "v23.9-optimized",
            "firestore_tracker_routes": "v23.9-minimal-response",
            "paa_routes": "v23.9-all-unused",
            "export_routes": "v23.9-editorial-review",
            "seo_optimizer": "v23.9",
            "final_review_routes": "v23.9",
            "s1_proxy": "v31.0 (semantic hints)",
            "unified_validator": "v31.0 (semantic enhancement)",
            "legal_module": "v3.0 (SAOS integration)" if LEGAL_MODULE_ENABLED else "disabled",
            "medical_module": "v1.0 (PubMed, ClinicalTrials, Polish Health)" if MEDICAL_MODULE_ENABLED else "disabled"
        },
        "optimizations": {
            "approve_batch_response": "~500B (was 220KB)",
            "pre_batch_info_response": "~5-10KB (was 30-60KB)",
            "s1_default_urls": 6
        },
        "semantic_enhancement": {
            "enabled": SEMANTIC_ENHANCEMENT_ENABLED,
            "features": [
                "entity_density",
                "topic_completeness",
                "entity_gap",
                "source_effort"
            ]
        },
        "medical_module": {
            "enabled": MEDICAL_MODULE_ENABLED,
            "sources": {
                "pubmed": PUBMED_AVAILABLE,
                "clinicaltrials": CLINICALTRIALS_AVAILABLE,
                "polish_health": POLISH_HEALTH_AVAILABLE,
                "claude_verifier": CLAUDE_VERIFIER_AVAILABLE
            } if MEDICAL_MODULE_ENABLED else {},
            "endpoints": [
                "/api/medical/status",
                "/api/medical/detect",
                "/api/medical/get_context",
                "/api/medical/search/pubmed",
                "/api/medical/search/trials",
                "/api/medical/search/polish",
                "/api/medical/validate",
                "/api/medical/disclaimer"
            ] if MEDICAL_MODULE_ENABLED else []
        },
        "environment": {
            "debug_mode": DEBUG_MODE,
            "firebase_connected": True,
            "ngram_base_url": NGRAM_BASE_URL,
            "semantic_enabled": SEMANTIC_ENABLED,
            "semantic_enhancement_enabled": SEMANTIC_ENHANCEMENT_ENABLED,
            "legal_module_enabled": LEGAL_MODULE_ENABLED,
            "medical_module_enabled": MEDICAL_MODULE_ENABLED
        }
    }), 200


# ================================================================
# 🧩 MANUAL CHECK ENDPOINT (test unified_prevalidation)
# ================================================================
@app.post("/api/manual_check")
def manual_check():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400

    dummy_keywords = data.get("keywords_state", {})
    result = unified_prevalidation(data["text"], dummy_keywords)

    response = {
        "status": "CHECK_OK",
        "semantic_score": result["semantic_score"],
        "density": result["density"],
        "smog": result["smog"],
        "readability": result["readability"],
        "warnings": result["warnings"],
        "semantic_coverage": result.get("semantic_coverage", {})
    }
    
    # 🆕 v31.0: Dodaj semantic enhancement check jeśli dostępne
    if SEMANTIC_ENHANCEMENT_ENABLED:
        try:
            sem_result = validate_semantic_enhancement(content=data["text"])
            response["semantic_enhancement"] = {
                "status": sem_result.get("status"),
                "score": sem_result.get("semantic_score"),
                "quick_wins": sem_result.get("quick_wins", [])[:3]
            }
        except Exception as e:
            response["semantic_enhancement"] = {"error": str(e)}

    return jsonify(response), 200


# ================================================================
# 🧩 AUTO FINAL REVIEW TRIGGER (po eksporcie)
# ================================================================
@app.post("/api/auto_final_review/<project_id>")
def auto_final_review(project_id):
    from final_review_routes import perform_final_review
    try:
        response = perform_final_review(project_id)
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# 🆕 v35.5: HELPER - Przelicz frazy z tekstu (OVERLAPPING)
# ================================================================
def count_keywords_from_text(text: str, keywords_state: dict) -> dict:
    """
    🆕 v35.5: Przelicza RZECZYWISTE użycia fraz z tekstu.
    
    Używa OVERLAPPING counting (zgodne z NeuronWriter):
    - "ubezwłasnowolnienie osoby" liczy jako +1 "ubezwłasnowolnienie" + +1 "osoby"
    
    Args:
        text: Pełny tekst do analizy
        keywords_state: Dict z frazami {phrase: {target_min, target_max, ...}}
    
    Returns:
        Dict {phrase: actual_count}
    """
    if not text or not keywords_state:
        return {}
    
    if KEYWORD_COUNTER_ENABLED:
        # Użyj keyword_counter z lemmatyzacją (OVERLAPPING mode)
        # Konwertuj format na oczekiwany przez count_keywords_for_state
        state_for_counter = {}
        for phrase, data in keywords_state.items():
            if isinstance(data, dict):
                state_for_counter[phrase] = {
                    "keyword": phrase,
                    "type": data.get("type", "BASIC"),
                    "target_min": data.get("target_min", 1),
                    "target_max": data.get("target_max", 999)
                }
            else:
                state_for_counter[phrase] = {
                    "keyword": phrase,
                    "type": "BASIC",
                    "target_min": 1,
                    "target_max": 999
                }
        
        # use_exclusive_for_nested=False = OVERLAPPING (jak NeuronWriter)
        counts = count_keywords_for_state(text, state_for_counter, use_exclusive_for_nested=False)
        
        # Mapuj z powrotem na phrase -> count
        result = {}
        for phrase in keywords_state.keys():
            result[phrase] = counts.get(phrase, 0)
        return result
    else:
        # Fallback: proste liczenie regex (case-insensitive)
        clean_text = re.sub(r'<[^>]+>', ' ', text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        result = {}
        for phrase in keywords_state.keys():
            phrase_lower = phrase.lower()
            # Użyj word boundaries dla bezpieczeństwa
            pattern = r'\b' + re.escape(phrase_lower).replace(r'\ ', r'\s+') + r'\b'
            matches = re.findall(pattern, clean_text, re.IGNORECASE)
            result[phrase] = len(matches)
        return result


# ================================================================
# 🆕 v31.3: APPROVE BATCH (z metrykami semantycznymi na bieżąco)
# ================================================================
@app.post("/api/approveBatch")
def approve_batch():
    """
    🆕 v32.5: Zatwierdzenie batcha z walidacją fraz + metrykami semantycznymi + AI Detection.
    
    Łączy walidację keyword (stuffing/missing) z bieżącym śledzeniem:
    - Entity density (dla dotychczasowej treści)
    - Entities used vs missing
    - Topic completeness estimate
    - Checkpoint alerts
    - AI Detection (humanness_score, burstiness, JITTER)
    
    Request body:
    {
        "project_id": "abc123",
        "batch_number": 3,
        "batch_content": "treść tego batcha",
        "accumulated_content": "cała treść do tej pory (batche 1-3)",
        "previous_paragraphs": 2,  // 🆕 v32.5: ile akapitów miał poprzedni batch (dla JITTER)
        "keywords_state": {
            "basic": {"przeprowadzki": {"target": 21, "current": 8}, ...},
            "extended": {"tanie przeprowadzki warszawa": {"target": 1, "current": 1}, ...}
        },
        "s1_data": {
            "entities": [...],
            "topics": [...],
            "entity_relationships": [...]  // triplety
        },
        "total_batches": 7
    }
    """
    data = request.get_json(force=True)
    
    # Required fields
    project_id = data.get("project_id", "unknown")
    batch_number = data.get("batch_number", 1)
    batch_content = data.get("batch_content", "")
    accumulated_content = data.get("accumulated_content", batch_content)
    keywords_state = data.get("keywords_state", {})
    s1_data = data.get("s1_data", {})
    total_batches = data.get("total_batches", 7)
    
    # 🔧 v32.5 FIX: Pobierz previous_paragraphs z requestu (dla JITTER validation)
    request_previous_paragraphs = data.get("previous_paragraphs")
    
    # ============================================
    # 🆕 v35.6: POBIERZ FRAZY Z FIRESTORE (nie ufaj GPT!)
    # ============================================
    # GPT może nie wysłać keywords_state - pobierz z projektu!
    if not keywords_state or (not keywords_state.get("basic") and not keywords_state.get("extended")):
        print(f"[APPROVE_BATCH] ⚠️ keywords_state pusty/brak - pobieram z Firestore")
        project_data = get_project(project_id)
        if project_data:
            # Pobierz frazy zapisane przy tworzeniu projektu
            # Format Firestore: {row_id: {keyword, type, target_min, target_max, ...}}
            stored_keywords = project_data.get("keywords_state", {})
            if stored_keywords:
                # Konwertuj z formatu Firestore na format {basic: {phrase: {...}}, extended: {...}}
                basic_kw_tmp = {}
                extended_kw_tmp = {}
                
                for row_id, meta in stored_keywords.items():
                    phrase = meta.get("keyword", "")
                    if not phrase:
                        continue
                    
                    kw_type = meta.get("type", "BASIC").upper()
                    target_min = meta.get("target_min", meta.get("min", 1))
                    target_max = meta.get("target_max", meta.get("max", 10))
                    
                    if kw_type == "EXTENDED":
                        extended_kw_tmp[phrase] = {
                            "target_min": 1,
                            "target_max": 1,
                            "type": "EXTENDED"
                        }
                    else:  # BASIC lub MAIN
                        basic_kw_tmp[phrase] = {
                            "target_min": target_min,
                            "target_max": target_max,
                            "type": kw_type
                        }
                
                keywords_state = {"basic": basic_kw_tmp, "extended": extended_kw_tmp}
                print(f"[APPROVE_BATCH] ✅ Załadowano z Firestore: {len(basic_kw_tmp)} BASIC + {len(extended_kw_tmp)} EXTENDED")
    
    basic_kw = keywords_state.get("basic", {})
    extended_kw = keywords_state.get("extended", {})
    
    # Zbierz wszystkie frazy do przeliczenia
    all_keywords = {}
    for phrase, state in basic_kw.items():
        all_keywords[phrase] = {
            "type": "BASIC",
            "target_min": state.get("target_min", state.get("target", 1)),
            "target_max": state.get("target_max", state.get("target_min", 1) * 4),
            "gpt_reported": state.get("current", 0)  # Co GPT myśli
        }
    for phrase, state in extended_kw.items():
        all_keywords[phrase] = {
            "type": "EXTENDED",
            "target_min": 1,
            "target_max": 1,
            "gpt_reported": state.get("current", 0)
        }
    
    # 🔥 PRZELICZ RZECZYWISTE UŻYCIA Z TEKSTU
    real_counts = count_keywords_from_text(accumulated_content, all_keywords)
    
    print(f"[APPROVE_BATCH] 📊 Real keyword counting for {len(all_keywords)} phrases")
    
    # ============================================
    # 1. WALIDACJA FRAZ (z RZECZYWISTYMI wartościami!)
    # ============================================
    keyword_warnings = []
    keyword_blockers = []
    keyword_details = []  # 🆕 Szczegóły dla GPT
    
    # Sprawdź BASIC
    for phrase, meta in all_keywords.items():
        if meta["type"] != "BASIC":
            continue
            
        real_count = real_counts.get(phrase, 0)
        gpt_count = meta["gpt_reported"]
        target_min = meta["target_min"]
        target_max = meta["target_max"]
        
        # 🆕 Loguj różnice między GPT a rzeczywistością
        if real_count != gpt_count:
            print(f"[APPROVE_BATCH] ⚠️ '{phrase}': GPT={gpt_count}, REAL={real_count}")
        
        detail = {
            "phrase": phrase,
            "type": "BASIC",
            "real_count": real_count,
            "gpt_reported": gpt_count,
            "target_min": target_min,
            "target_max": target_max,
            "remaining": max(0, target_max - real_count)
        }
        
        if real_count > target_max:
            detail["status"] = "OVER"
            keyword_blockers.append({
                "type": "STUFFING",
                "phrase": phrase,
                "current": real_count,
                "max": target_max,
                "message": f"🔴 STUFFING: '{phrase}' = {real_count}× (max {target_max}) - PRZEKROCZONO!"
            })
        elif real_count == 0:
            detail["status"] = "MISSING"
            keyword_warnings.append({
                "type": "MISSING",
                "phrase": phrase,
                "current": 0,
                "target": target_min,
                "message": f"⚠️ BRAK: '{phrase}' nie użyte (cel: min {target_min}×)"
            })
        elif real_count < target_min:
            detail["status"] = "UNDER"
            keyword_warnings.append({
                "type": "UNDER_TARGET",
                "phrase": phrase,
                "current": real_count,
                "target": target_min,
                "message": f"⚠️ '{phrase}' = {real_count}× (cel: min {target_min}×)"
            })
        elif real_count >= target_max:
            detail["status"] = "LOCKED"
        else:
            detail["status"] = "OK"
        
        keyword_details.append(detail)
    
    # Sprawdź EXTENDED
    extended_used = 0
    extended_total = 0
    extended_missing = []
    
    for phrase, meta in all_keywords.items():
        if meta["type"] != "EXTENDED":
            continue
        
        extended_total += 1
        real_count = real_counts.get(phrase, 0)
        gpt_count = meta["gpt_reported"]
        
        if real_count != gpt_count:
            print(f"[APPROVE_BATCH] ⚠️ EXTENDED '{phrase}': GPT={gpt_count}, REAL={real_count}")
        
        detail = {
            "phrase": phrase,
            "type": "EXTENDED",
            "real_count": real_count,
            "gpt_reported": gpt_count,
            "target_min": 1,
            "target_max": 1
        }
        
        if real_count >= 1:
            extended_used += 1
            detail["status"] = "DONE"
            
            if real_count > 1:
                detail["status"] = "OVER"
                keyword_warnings.append({
                    "type": "EXTENDED_OVERFLOW",
                    "phrase": phrase,
                    "current": real_count,
                    "message": f"⚠️ EXTENDED '{phrase}' = {real_count}× (powinno być dokładnie 1×)"
                })
        else:
            extended_missing.append(phrase)
            detail["status"] = "MISSING"
        
        keyword_details.append(detail)
    
    # ============================================
    # 2. METRYKI SEMANTYCZNE (na bieżąco)
    # ============================================
    semantic_progress = {
        "entity_density_current": 0,
        "word_count_total": 0,
        "entities_used": [],
        "entities_missing_critical": [],
        "topic_completeness_estimate": 0,
        "extended_progress": f"{extended_used}/{extended_total}"
    }
    
    if SEMANTIC_ENHANCEMENT_ENABLED and accumulated_content:
        try:
            # Entity density
            density_result = calculate_entity_density(accumulated_content)
            semantic_progress["entity_density_current"] = density_result.get("density", 0)
            semantic_progress["word_count_total"] = density_result.get("word_count", 0)
            
            # 🔧 v32.5 FIX: Entities są w entity_seo, nie na głównym poziomie
            entity_seo = s1_data.get("entity_seo", {})
            s1_entities = entity_seo.get("entities", [])
            # Fallback na główny poziom dla kompatybilności
            if not s1_entities:
                s1_entities = s1_data.get("entities", [])
            
            content_lower = accumulated_content.lower()
            
            entities_used = []
            entities_missing = []
            
            for ent in s1_entities:
                ent_name = ent.get("name", "")
                importance = ent.get("importance", 0)
                sources = ent.get("sources", 0)
                
                if ent_name.lower() in content_lower:
                    entities_used.append(ent_name)
                elif importance >= 0.7 or sources >= 4:
                    entities_missing.append({
                        "name": ent_name,
                        "importance": importance,
                        "priority": "CRITICAL" if (importance >= 0.7 and sources >= 4) else "HIGH"
                    })
            
            semantic_progress["entities_used"] = entities_used[:10]
            semantic_progress["entities_missing_critical"] = [e["name"] for e in entities_missing if e["priority"] == "CRITICAL"][:5]
            
            # Topic completeness estimate
            s1_topics = s1_data.get("topics", [])
            if s1_topics:
                topics_covered = 0
                must_topics = [t for t in s1_topics if t.get("priority") in ["MUST", "HIGH"]]
                
                for topic in must_topics:
                    topic_name = topic.get("name", "").lower()
                    if topic_name in content_lower:
                        topics_covered += 1
                
                if must_topics:
                    semantic_progress["topic_completeness_estimate"] = round(
                        (topics_covered / len(must_topics)) * 100, 1
                    )
                    
        except Exception as e:
            print(f"[APPROVE_BATCH] Semantic calc error: {e}")
    
    # ============================================
    # 3. CHECKPOINT ALERTS
    # ============================================
    checkpoint_alerts = []
    
    # Checkpoint: Batch 3 - min 2 encje PERSON/ORG
    if batch_number >= 3:
        if len(semantic_progress["entities_used"]) < 2:
            checkpoint_alerts.append({
                "checkpoint": "BATCH_3",
                "status": "WARNING",
                "message": f"Batch 3+: użyto tylko {len(semantic_progress['entities_used'])} encji (min: 2)"
            })
    
    # Checkpoint: Batch 5 - topic_completeness >= 50%
    if batch_number >= 5:
        if semantic_progress["topic_completeness_estimate"] < 50:
            checkpoint_alerts.append({
                "checkpoint": "BATCH_5",
                "status": "WARNING",
                "message": f"Batch 5+: topic_completeness {semantic_progress['topic_completeness_estimate']}% (min: 50%)"
            })
    
    # Checkpoint: Pre-FAQ (batch N-1) - entity_density >= 2.5
    if batch_number >= total_batches - 1:
        if semantic_progress["entity_density_current"] < 2.5:
            checkpoint_alerts.append({
                "checkpoint": "PRE_FAQ",
                "status": "WARNING",
                "message": f"Pre-FAQ: entity_density {semantic_progress['entity_density_current']:.1f} (min: 2.5)"
            })
        
        # Sprawdź czy są nieużyte CRITICAL encje
        if semantic_progress["entities_missing_critical"]:
            checkpoint_alerts.append({
                "checkpoint": "PRE_FAQ",
                "status": "ALERT",
                "message": f"Brakujące CRITICAL encje: {', '.join(semantic_progress['entities_missing_critical'][:3])}"
            })
    
    # ============================================
    # 4. 🆕 v32.0: AI DETECTION
    # ============================================
    ai_detection_result = {
        "humanness_score": 0,
        "status": "DISABLED",
        "burstiness": 0,
        "warnings": []
    }
    
    if AI_DETECTION_ENABLED and batch_content:
        try:
            # 🔧 v32.5 FIX: previous_paragraphs - priorytet z requestu, fallback z Firestore
            previous_paragraphs = request_previous_paragraphs  # z requestu
            if previous_paragraphs is None:
                # Fallback: pobierz z Firestore jeśli nie było w request
                project_data = get_project(project_id)
                if project_data:
                    previous_paragraphs = project_data.get("last_batch_paragraphs")
            
            print(f"[APPROVE_BATCH] 📊 previous_paragraphs={previous_paragraphs} (from {'request' if request_previous_paragraphs is not None else 'firestore'})")
            
            # 🔧 v32.5 FIX: Pobierz entity_relationships z entity_seo (nie z głównego s1_data!)
            entity_seo = s1_data.get("entity_seo", {})
            s1_relationships = entity_seo.get("entity_relationships", [])
            
            # Fallback: jeśli brak, spróbuj z głównego poziomu (kompatybilność wsteczna)
            if not s1_relationships:
                s1_relationships = s1_data.get("entity_relationships", [])
            
            print(f"[APPROVE_BATCH] 📊 Found {len(s1_relationships)} entity_relationships")
            
            # Pełna analiza AI detection
            ai_result = full_ai_detection(
                text=batch_content,
                previous_paragraphs=previous_paragraphs,
                s1_relationships=s1_relationships
            )
            
            ai_detection_result = {
                "humanness_score": ai_result.get("humanness_score", 0),
                "status": ai_result.get("status", "OK"),
                "burstiness": ai_result.get("components", {}).get("burstiness", {}).get("value", 0),
                "warnings": ai_result.get("warnings", [])[:5],
                "validations": ai_result.get("validations", {})
            }
            
            # 🆕 v32.4: Zapisz current paragraphs do Firestore
            import re
            current_paragraphs = len(re.split(r'\n\s*\n', batch_content.strip()))
            update_project(project_id, {"last_batch_paragraphs": current_paragraphs})
            
            # Dodaj checkpoint alert jeśli AI score < 50 (CRITICAL)
            if ai_result.get("humanness_score", 100) < 50:
                checkpoint_alerts.append({
                    "checkpoint": f"BATCH_{batch_number}",
                    "status": "CRITICAL",
                    "message": f"AI Detection CRITICAL: humanness_score {ai_result['humanness_score']} < 50. Przepisz batch!"
                })
            elif ai_result.get("humanness_score", 100) < 70:
                checkpoint_alerts.append({
                    "checkpoint": f"BATCH_{batch_number}",
                    "status": "WARNING",
                    "message": f"AI Detection WARNING: humanness_score {ai_result['humanness_score']} < 70"
                })
            
            # Sprawdź forbidden phrases
            forbidden = ai_result.get("validations", {}).get("forbidden_phrases", {})
            if forbidden.get("status") == "CRITICAL":
                checkpoint_alerts.append({
                    "checkpoint": f"BATCH_{batch_number}",
                    "status": "CRITICAL",
                    "message": f"Forbidden phrases: {', '.join(forbidden.get('forbidden_found', [])[:3])}"
                })
            
        except Exception as e:
            print(f"[APPROVE_BATCH] AI Detection error: {e}")
            ai_detection_result["error"] = str(e)
    
    # ============================================
    # 5. 🆕 FAZA 2: ENTITY SPLIT + TOPIC COMPLETENESS
    # ============================================
    entity_split_result = {"status": "DISABLED"}
    topic_completeness_result = {"status": "DISABLED"}
    
    if AI_DETECTION_ENABLED and accumulated_content:
        try:
            # 🔧 v32.5 FIX: Entity Split 60/40 - entities są w entity_seo
            entity_seo = s1_data.get("entity_seo", {})
            s1_entities = entity_seo.get("entities", [])
            if not s1_entities:
                s1_entities = s1_data.get("entities", [])
            entity_split_result = calculate_entity_split(accumulated_content, s1_entities)
            
            # 🔧 v32.5 FIX: Topic Completeness - topics mogą być w różnych miejscach
            s1_topics = (
                entity_seo.get("topical_coverage", []) or  # preferowane miejsce
                s1_data.get("topics", []) or
                s1_data.get("topical_coverage", [])
            )
            topic_completeness_result = calculate_topic_completeness(accumulated_content, s1_topics)
            
            # Dodaj do semantic_progress
            semantic_progress["entity_split"] = entity_split_result
            semantic_progress["topic_completeness"] = topic_completeness_result.get("score_percent", 0)
            semantic_progress["topics_missing"] = topic_completeness_result.get("must_missing", []) + topic_completeness_result.get("high_missing", [])
            
        except Exception as e:
            print(f"[APPROVE_BATCH] Entity Split/Topic error: {e}")
    
    # ============================================
    # 🆕 v36.0: SEMANTIC KEYWORD PLAN VALIDATION
    # Sprawdź czy assigned_keywords dla tego batcha zostały użyte
    # ============================================
    semantic_plan_warnings = []
    assigned_keywords_used = []
    assigned_keywords_missing = []
    
    try:
        project_data = get_project(project_id) or {}
        semantic_plan = project_data.get("semantic_keyword_plan", {})
        
        if semantic_plan:
            batch_plans = semantic_plan.get("batch_plans", [])
            
            # Znajdź plan dla bieżącego batcha
            current_batch_plan = None
            for bp in batch_plans:
                if bp.get("batch_number") == batch_number:
                    current_batch_plan = bp
                    break
            
            if current_batch_plan:
                assigned_kws = current_batch_plan.get("assigned_keywords", [])
                content_lower = accumulated_content.lower()
                
                for kw in assigned_kws:
                    if kw.lower() in content_lower:
                        assigned_keywords_used.append(kw)
                    else:
                        assigned_keywords_missing.append(kw)
                
                # Warning dla nieużytych assigned_keywords
                if assigned_keywords_missing and batch_number < total_batches:
                    semantic_plan_warnings.append({
                        "type": "SEMANTIC_ASSIGNED_MISSING",
                        "message": f"⚠️ Przypisane frazy dla tego batcha nieużyte: {', '.join(assigned_keywords_missing[:5])}",
                        "missing": assigned_keywords_missing,
                        "h2": current_batch_plan.get("h2"),
                        "batch": batch_number
                    })
                    print(f"[APPROVE_BATCH] 🎯 Semantic: {len(assigned_keywords_used)} assigned used, {len(assigned_keywords_missing)} missing")
    except Exception as e:
        print(f"[APPROVE_BATCH] Semantic plan validation error: {e}")
    
    # ============================================
    # 6. 🆕 FAZA 2: BATCH HISTORY TRACKING
    # ============================================
    batch_history = []
    batch_trend = {"trend": "insufficient_data"}
    
    if AI_DETECTION_ENABLED:
        try:
            # 🆕 v32.4: Pobierz z Firestore
            project_data = get_project(project_id) or {}
            
            # Pobierz historię
            batch_history = project_data.get("batch_history", [])
            
            # Stwórz rekord dla tego batcha
            current_record = create_batch_record(
                batch_number=batch_number,
                humanness_score=ai_detection_result.get("humanness_score", 0),
                burstiness=ai_detection_result.get("burstiness", 0),
                paragraphs=project_data.get("last_batch_paragraphs", 0),
                entity_density=semantic_progress.get("entity_density_current", 0),
                topic_completeness=topic_completeness_result.get("score", 0)
            )
            
            # Dodaj do historii i zapisz do Firestore
            batch_history.append(current_record)
            update_project(project_id, {"batch_history": batch_history})
            
            # Analizuj trend
            batch_trend = analyze_batch_trend(batch_history)
            
        except Exception as e:
            print(f"[APPROVE_BATCH] Batch history error: {e}")
    
    # ============================================
    # 🆕 v33.0: HUMANIZATION CHECKS + FIX INSTRUCTIONS
    # ============================================
    fix_instructions = []
    should_block_humanization = False
    
    if AI_DETECTION_ENABLED and batch_content:
        try:
            # 1. BURSTINESS CHECK
            # 🔧 v35.7: Próg 2.0 (CV < 0.4) - blokuje monotonne teksty
            burstiness_val = ai_detection_result.get("burstiness", 0)
            BURSTINESS_BLOCK_THRESHOLD = 2.0   # 🔧 Zmienione z 1.2
            BURSTINESS_WARNING_THRESHOLD = 2.5 # 🔧 Zmienione z 1.6
            
            if burstiness_val < BURSTINESS_BLOCK_THRESHOLD:
                should_block_humanization = True
                sent_dist = analyze_sentence_distribution(batch_content)
                burst_fix = generate_burstiness_fix(burstiness_val, sent_dist)
                
                # 🆕 v41.0: Sugestie podziału długich zdań zamiast sztucznych wstawek
                split_suggestions = []
                if SENTENCE_SPLITTER_AVAILABLE:
                    split_suggestions = suggest_sentence_splits(batch_content, max_suggestions=3)
                
                fix_instructions.append({
                    "type": "BURSTINESS_CRITICAL",
                    "message": f"🔴 Burstiness {burstiness_val} < {BURSTINESS_BLOCK_THRESHOLD} - PRZEPISZ batch!",
                    "fix": burst_fix.get("fix_instruction", ""),
                    "inserts": burst_fix.get("insert_suggestions", []),
                    "sentence_splits": split_suggestions,  # 🆕 v41.0
                    "example": burst_fix.get("rewrite_example", {}),
                    "distribution": sent_dist.get("distribution_label", "")
                })
            elif burstiness_val < BURSTINESS_WARNING_THRESHOLD:
                sent_dist = analyze_sentence_distribution(batch_content)
                burst_fix = generate_burstiness_fix(burstiness_val, sent_dist)
                
                # 🆕 v41.0: Sugestie podziału długich zdań
                split_suggestions = []
                if SENTENCE_SPLITTER_AVAILABLE:
                    split_suggestions = suggest_sentence_splits(batch_content, max_suggestions=2)
                
                fix_instructions.append({
                    "type": "BURSTINESS_WARNING",
                    "message": f"⚠️ Burstiness {burstiness_val} < {BURSTINESS_WARNING_THRESHOLD} - popraw w tym batchu",
                    "fix": burst_fix.get("fix_instruction", ""),
                    "inserts": burst_fix.get("insert_suggestions", []),
                    "sentence_splits": split_suggestions,  # 🆕 v41.0
                })
            
            # 2. FORBIDDEN PHRASES CHECK
            # 🔧 v35.7: Forbidden phrases to teraz WARNING, nie BLOCKED
            # Powód: zbyt często blokuje w tekstach prawnych/medycznych
            forbidden = ai_detection_result.get("validations", {}).get("forbidden_phrases", {})
            if forbidden.get("should_block", False):
                # should_block_humanization = True  # 🔧 WYŁĄCZONE!
                fix_instructions.append({
                    "type": "FORBIDDEN_PHRASES_WARNING",  # Zmienione z FORBIDDEN_PHRASES
                    "message": f"⚠️ Zakazane frazy (zamień jeśli możliwe): {', '.join(forbidden.get('forbidden_found', [])[:5])}",
                    "replacements": forbidden.get("replacements", []),
                    "action": "ZAMIEŃ wskazane frazy (nie blokuje batcha)"
                })
            
            # 3. WORD REPETITION CHECK
            # 🔧 v35.7: Word repetition to teraz WARNING, nie BLOCKED
            # Powód: w tematycznych artykułach (np. prawnych) główne słowo MUSI się powtarzać
            word_rep = check_word_repetition_detailed(batch_content)
            if word_rep.get("should_block", False):
                # should_block_humanization = True  # 🔧 WYŁĄCZONE!
                fix_instructions.append({
                    "type": "WORD_REPETITION_WARNING",  # Zmienione z CRITICAL
                    "message": word_rep.get("message", "").replace("🔴", "⚠️"),
                    "violations": word_rep.get("violations", [])[:3],
                    "action": "Rozważ synonimy (nie blokuje batcha)"
                })
            elif word_rep.get("status") == "WARNING":
                fix_instructions.append({
                    "type": "WORD_REPETITION_WARNING",
                    "message": word_rep.get("message", ""),
                    "warnings": word_rep.get("warnings", [])[:3]
                })
            
            # Dodaj word_repetition do ai_detection_result
            ai_detection_result["word_repetition"] = {
                "top_words": word_rep.get("top_words", []),
                "status": word_rep.get("status", "OK")
            }
            
        except Exception as e:
            print(f"[APPROVE_BATCH] Humanization check error: {e}")
    
    # ============================================
    # 7. WYNIK - 🔧 v35.7: ZŁAGODZONE BLOKOWANIE
    # ============================================
    has_blockers = len(keyword_blockers) > 0
    
    # 🔧 v35.7: BASIC Stuffing blokuje TYLKO przy >50% przekroczeniu (1.5x limit)
    # Przykład: max=10, użyto 16× = blokada. Użyto 14× = tylko warning.
    severe_stuffing = False
    for blocker in keyword_blockers:
        if blocker.get("type") == "STUFFING":
            actual = blocker.get("current", 0)
            max_allowed = blocker.get("max", 999)
            # Blokuj tylko jeśli przekroczenie > 50% (1.5x limit)
            if actual > max_allowed * 1.5:
                severe_stuffing = True
                print(f"[APPROVE_BATCH] 🔴 SEVERE STUFFING: {blocker.get('phrase')} = {actual}× (max {max_allowed}, próg {max_allowed * 1.5:.0f})")
                break
            else:
                print(f"[APPROVE_BATCH] ⚠️ Minor stuffing (not blocking): {blocker.get('phrase')} = {actual}× (max {max_allowed})")
    
    # 🔧 v35.7: AI CRITICAL już NIE blokuje - tylko warning
    # Powód: Claude i tak generuje dobry tekst, a zbyt restrykcyjne AI detection
    # powoduje nieskończone pętle poprawek
    ai_critical = ai_detection_result.get("status") == "CRITICAL"
    # ai_critical_blocks = False  # Wyłączone!
    
    # 🔧 v35.7: Zmieniona logika - łagodniejsza
    # BLOCKED tylko gdy:
    # 1. Poważne stuffing (>50% przekroczenia) LUB
    # 2. Burstiness < 1.2 (ekstremalnie monotonne)
    final_should_block = severe_stuffing or should_block_humanization
    
    # 🆕 v36.0: Dodaj semantic_plan_warnings do keyword_warnings
    keyword_warnings.extend(semantic_plan_warnings)
    
    # 🔧 v35.7: Jeśli zwykłe stuffing (bez severe), zmień na WARNING
    if has_blockers and not severe_stuffing:
        for blocker in keyword_blockers:
            blocker["severity"] = "WARNING"
            blocker["message"] = blocker.get("message", "").replace("🔴", "⚠️")
    
    # 🔧 v35.7: Zaktualizowane action_required
    if severe_stuffing:
        action_required = "FIX_STUFFING_SEVERE"
    elif should_block_humanization:
        action_required = "FIX_HUMANIZATION"
    elif has_blockers:  # Mniejsze stuffing - tylko warning
        action_required = "CHECK_STUFFING_WARNING"
    elif ai_critical:
        action_required = "CHECK_AI_WARNING"  # Nie blokuje, tylko warning
    elif keyword_warnings or fix_instructions:
        action_required = "CHECK_WARNINGS"
    else:
        action_required = "CONTINUE"
    
    result = {
        "status": "BLOCKED" if final_should_block else "SAVED",
        "project_id": project_id,
        "batch_number": batch_number,
        "batch_total": total_batches,
        
        # 🆕 v35.5: RZECZYWISTE użycia fraz (nie GPT!)
        "keyword_validation": {
            "blockers": keyword_blockers,
            "warnings": keyword_warnings,
            "extended_progress": {
                "used": extended_used,
                "total": extended_total,
                "missing": extended_missing[:10]
            },
            # 🆕 SZCZEGÓŁY DLA GPT - pokazuje RZECZYWISTE wartości
            "keyword_details": keyword_details,
            "counting_method": "OVERLAPPING (NeuronWriter compatible)"
        },
        
        # Metryki semantyczne na bieżąco
        "semantic_progress": semantic_progress,
        
        # 🆕 v32.0: AI Detection
        "ai_detection": ai_detection_result,
        
        # 🆕 v33.0: Fix Instructions (WAŻNE!)
        "fix_instructions": fix_instructions,
        
        # 🆕 Faza 2: Entity Split
        "entity_split": entity_split_result,
        
        # 🆕 Faza 2: Topic Completeness (szczegóły)
        "topic_completeness": topic_completeness_result,
        
        # 🆕 v36.0: Semantic Keyword Plan validation
        "semantic_plan_validation": {
            "assigned_used": assigned_keywords_used,
            "assigned_missing": assigned_keywords_missing,
            "warnings": semantic_plan_warnings
        },
        
        # 🆕 Faza 2: Batch History & Trend
        "batch_trend": batch_trend,
        
        # Alerty checkpointów
        "checkpoint_alerts": checkpoint_alerts,
        
        # Podsumowanie dla GPT
        "summary": {
            "can_continue": not final_should_block,
            "action_required": action_required,
            "fix_count": len(fix_instructions),
            "next_step": f"Batch {batch_number + 1}/{total_batches}" if not final_should_block and batch_number < total_batches else ("APPLY_FIXES" if final_should_block else "FINALIZE")
        }
    }
    
    return jsonify(result), 200


# ================================================================
# 🆕 v32.0: AI DETECTION ENDPOINTS
# ================================================================
@app.post("/api/ai_detection")
def ai_detection_endpoint():
    """
    🆕 v32.0: Pełna analiza wykrywalności AI.
    
    Request body:
    {
        "text": "treść do analizy (min 200 znaków)",
        "previous_paragraphs": 3,  // opcjonalne - dla JITTER check
        "s1_relationships": [...]  // opcjonalne - dla triplets check
    }
    
    Returns:
    {
        "humanness_score": 72,
        "status": "OK" | "WARNING" | "CRITICAL",
        "components": {...},
        "validations": {
            "forbidden_phrases": {...},
            "jitter": {...},
            "triplets": {...}
        },
        "warnings": [...],
        "suggestions": [...]
    }
    """
    if not AI_DETECTION_ENABLED:
        return jsonify({
            "error": "AI Detection not available",
            "message": "Install wordfreq: pip install wordfreq"
        }), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 200:
        return jsonify({
            "error": "Text too short",
            "message": "Minimum 200 characters required",
            "length": len(text)
        }), 400
    
    previous_paragraphs = data.get("previous_paragraphs")
    s1_relationships = data.get("s1_relationships", [])
    
    try:
        result = full_ai_detection(
            text=text,
            previous_paragraphs=previous_paragraphs,
            s1_relationships=s1_relationships
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "AI Detection failed",
            "message": str(e)
        }), 500


@app.post("/api/quick_ai_check")
def quick_ai_check_endpoint():
    """
    🆕 v32.0: Szybki check AI - tylko podstawowe metryki.
    
    Request body:
    {
        "text": "treść do analizy"
    }
    
    Returns:
    {
        "humanness_score": 72,
        "status": "OK" | "WARNING" | "CRITICAL",
        "burstiness": 3.2,
        "top_warning": "..."
    }
    """
    if not AI_DETECTION_ENABLED:
        return jsonify({
            "error": "AI Detection not available"
        }), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 100:
        return jsonify({
            "error": "Text too short",
            "message": "Minimum 100 characters required"
        }), 400
    
    try:
        result = quick_ai_check(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "Quick AI check failed",
            "message": str(e)
        }), 500


# ================================================================
# 🆕 v32.2: FAZA 3 ENDPOINTS
# ================================================================
@app.post("/api/score_sentences")
def score_sentences_endpoint():
    """
    🆕 v32.2: Analiza poszczególnych zdań.
    
    Request body:
    {
        "text": "treść do analizy",
        "limit": 10  // opcjonalne - ile najgorszych zdań zwrócić
    }
    
    Returns:
    {
        "status": "OK" | "WARNING" | "CRITICAL",
        "total_sentences": 25,
        "avg_score": 72.5,
        "ai_like_count": 2,
        "worst_sentences": [...],
        "suggestions": [...]
    }
    """
    if not AI_DETECTION_ENABLED:
        return jsonify({"error": "AI Detection not available"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    limit = data.get("limit", 10)
    
    if len(text) < 200:
        return jsonify({
            "error": "Text too short",
            "message": "Minimum 200 characters required"
        }), 400
    
    try:
        result = score_sentences(text, limit=limit)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "Sentence scoring failed",
            "message": str(e)
        }), 500


@app.post("/api/ngram_naturalness")
def ngram_naturalness_endpoint():
    """
    🆕 v32.2: Sprawdza naturalność fraz (n-gramów).
    
    Request body:
    {
        "text": "treść do analizy"
    }
    
    Returns:
    {
        "status": "OK" | "WARNING" | "CRITICAL",
        "naturalness_score": 0.75,
        "unnatural_list": ["kluczowy aspekt", ...],
        "suggestions": [...]
    }
    """
    if not AI_DETECTION_ENABLED:
        return jsonify({"error": "AI Detection not available"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 200:
        return jsonify({
            "error": "Text too short",
            "message": "Minimum 200 characters required"
        }), 400
    
    try:
        result = check_ngram_naturalness(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "N-gram check failed",
            "message": str(e)
        }), 500


@app.post("/api/full_analysis")
def full_analysis_endpoint():
    """
    🆕 v32.2: Pełna zaawansowana analiza (wszystkie metryki).
    
    Request body:
    {
        "text": "treść do analizy",
        "previous_paragraphs": 3,
        "s1_relationships": [...],
        "s1_entities": [...],
        "s1_topics": [...]
    }
    
    Returns kompletną analizę ze wszystkich faz.
    """
    if not AI_DETECTION_ENABLED:
        return jsonify({"error": "AI Detection not available"}), 503
    
    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if len(text) < 200:
        return jsonify({
            "error": "Text too short",
            "message": "Minimum 200 characters required"
        }), 400
    
    try:
        result = full_advanced_analysis(
            text=text,
            previous_paragraphs=data.get("previous_paragraphs"),
            s1_relationships=data.get("s1_relationships", []),
            s1_entities=data.get("s1_entities", []),
            s1_topics=data.get("s1_topics", [])
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "Full analysis failed",
            "message": str(e)
        }), 500


# ================================================================
# 🆕 v31.3: SAVE FULL ARTICLE (scalanie batchy)
# ================================================================
@app.post("/api/project/<project_id>/save_full_article")
def save_full_article(project_id):
    """
    🆕 v31.3: Zapisuje pełną treść artykułu do projektu.
    
    Wywołaj PO wszystkich batchach, PRZED getFinalReview!
    
    Request body:
    {
        "full_content": "cały tekst artykułu (wszystkie batche scalone)",
        "word_count": 1100,
        "h2_count": 7
    }
    """
    data = request.get_json(force=True)
    
    full_content = data.get("full_content", "")
    word_count = data.get("word_count", len(full_content.split()))
    h2_count = data.get("h2_count", 0)
    
    if not full_content or len(full_content) < 500:
        return jsonify({
            "status": "ERROR",
            "message": "full_content too short (min 500 chars)"
        }), 400
    
    # 🆕 v32.4: Zapisz do Firestore (persystentnie)
    update_project(project_id, {
        "full_article": {
            "content": full_content,
            "word_count": word_count,
            "h2_count": h2_count,
            "saved_at": datetime.utcnow().isoformat()
        }
    })
    
    return jsonify({
        "status": "SAVED",
        "project_id": project_id,
        "word_count": word_count,
        "h2_count": h2_count,
        "message": "Full article saved to Firestore. Ready for export."
    }), 200


# ================================================================
# 🆕 v31.3: GET FULL ARTICLE
# ================================================================
@app.get("/api/project/<project_id>/full_article")
def get_full_article(project_id):
    """Pobiera zapisany pełny artykuł."""
    # 🆕 v32.4: Pobierz z Firestore
    project_data = get_project(project_id)
    if not project_data:
        return jsonify({"status": "ERROR", "message": "Project not found"}), 404
    
    full_article = project_data.get("full_article")
    if not full_article:
        return jsonify({"status": "ERROR", "message": "No full article saved. Call saveFullArticle first."}), 404
    
    return jsonify({
        "status": "OK",
        "project_id": project_id,
        "content": full_article.get("content", ""),
        "word_count": full_article.get("word_count", 0),
        "h2_count": full_article.get("h2_count", 0),
        "saved_at": full_article.get("saved_at")
    }), 200


# ================================================================
# 🆕 v33.0: SYNONYM SERVICE ENDPOINT
# ================================================================
@app.post("/api/synonyms")
def get_synonyms_endpoint():
    """
    🆕 v33.0: Pobiera synonimy dla słowa lub listy słów.
    
    Request body:
    {
        "word": "skóra",  // lub
        "words": ["skóra", "witamina", "dobry"],
        "context": "artykuł o suplementach"  // opcjonalne
    }
    """
    try:
        from synonym_service import get_synonyms
        
        data = request.get_json() or {}
        word = data.get("word")
        words = data.get("words", [])
        context = data.get("context", "")
        
        if word:
            result = get_synonyms(word, context)
            return jsonify({"status": "OK", "results": {word: result}}), 200
        
        elif words:
            results = {}
            for w in words[:20]:  # max 20 słów
                results[w] = get_synonyms(w, context)
            return jsonify({"status": "OK", "results": results}), 200
        
        else:
            return jsonify({"status": "ERROR", "message": "Provide 'word' or 'words'"}), 400
            
    except ImportError:
        # Fallback do SYNONYM_MAP z ai_detection_metrics
        data = request.get_json() or {}
        word = data.get("word", "")
        synonyms = SYNONYM_MAP.get(word.lower(), [])
        return jsonify({
            "status": "OK",
            "results": {word: {"word": word.lower(), "synonyms": synonyms, "source": "static", "count": len(synonyms)}}
        }), 200
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@app.get("/api/synonyms/<word>")
def get_synonym_simple(word: str):
    """
    🆕 v33.0: Prosty GET dla pojedynczego słowa.
    GET /api/synonyms/skóra
    """
    try:
        from synonym_service import get_synonyms
        result = get_synonyms(word)
        return jsonify(result), 200
    except ImportError:
        synonyms = SYNONYM_MAP.get(word.lower(), [])
        return jsonify({
            "word": word.lower(),
            "synonyms": synonyms,
            "source": "static_fallback",
            "count": len(synonyms)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# 🏃 Local Run
# ================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Master SEO API {VERSION} on port {port}")
    print(f"🔧 Debug mode: {DEBUG_MODE}")
    print(f"🔗 S1 Proxy enabled → {NGRAM_ANALYSIS_ENDPOINT}")
    print(f"📊 Keyword Counter: {'ENABLED ✅ (OVERLAPPING)' if KEYWORD_COUNTER_ENABLED else 'DISABLED ⚠️'}")
    print(f"🧠 Semantic analysis: {'ENABLED ✅' if SEMANTIC_ENABLED else 'DISABLED ⚠️'}")
    print(f"🆕 Semantic Enhancement v31.0: {'ENABLED ✅' if SEMANTIC_ENHANCEMENT_ENABLED else 'DISABLED ⚠️'}")
    print(f"🤖 AI Detection v33.0: {'ENABLED ✅' if AI_DETECTION_ENABLED else 'DISABLED ⚠️'}")
    print(f"⚖️ Legal Module v3.0: {'ENABLED ✅' if LEGAL_MODULE_ENABLED else 'DISABLED ⚠️'}")
    print(f"🏥 Medical Module v1.0: {'ENABLED ✅' if MEDICAL_MODULE_ENABLED else 'DISABLED ⚠️'}")
    if MEDICAL_MODULE_ENABLED:
        print(f"   ├─ PubMed: {'✅' if PUBMED_AVAILABLE else '❌'}")
        print(f"   ├─ ClinicalTrials: {'✅' if CLINICALTRIALS_AVAILABLE else '❌'}")
        print(f"   ├─ Polish Health: {'✅' if POLISH_HEALTH_AVAILABLE else '❌'}")
        print(f"   └─ Claude Verifier: {'✅' if CLAUDE_VERIFIER_AVAILABLE else '❌'}")
    print(f"📦 Features v37.0:")
    print(f"   ─── 🏥 MEDICAL MODULE (v37.0) ───")
    print(f"   ✅ PubMed NCBI E-utilities")
    print(f"   ✅ ClinicalTrials.gov API v2")
    print(f"   ✅ Polish Health (PZH, AOTMiT, MZ, NFZ)")
    print(f"   ✅ Claude AI evidence verification")
    print(f"   ✅ NLM/APA citation formatting")
    print(f"   ─── 🆕 KEYWORD FIX (v35.5) ───")
    print(f"   ✅ Real-time counting in approveBatch")
    print(f"   ✅ OVERLAPPING mode (NeuronWriter)")
    print(f"   ✅ /api/verify_keywords endpoint")
    print(f"   ─── Faza 1 (CRITICAL) ───")
    print(f"   ✅ AI Detection (humanness_score)")
    print(f"   ✅ Forbidden phrases check")
    print(f"   ✅ JITTER validation")
    print(f"   ✅ Triplets validation")
    print(f"   ─── Faza 2 (HIGH) ───")
    print(f"   ✅ Entity Split 60/40 tracking")
    print(f"   ✅ Topic Completeness w approveBatch")
    print(f"   ✅ Batch History & Trend tracking")
    print(f"   ─── Faza 3 (ADVANCED) ───")
    print(f"   ✅ Per-sentence scoring")
    print(f"   ✅ N-gram naturalness check")
    print(f"   ✅ Full advanced analysis")
    print(f"   ─── Legal Module ───")
    print(f"   ✅ Auto-detection (prawo category)")
    print(f"   ✅ SAOS integration (judgments)")
    print(f"   ─── Medical Module ───")
    print(f"   ✅ Auto-detection (medycyna category)")
    print(f"   ✅ Evidence-based sources")
    print(f"   ─── Endpoints ───")
    print(f"   /api/approveBatch (🆕 real counting)")
    print(f"   /api/verify_keywords")
    print(f"   /api/ai_detection")
    print(f"   /api/legal/status")
    print(f"   /api/medical/status (🆕)\n")
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
