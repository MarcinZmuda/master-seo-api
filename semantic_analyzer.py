import re
from typing import List, Dict

_semantic_model = None
_cosine_similarity = None
SEMANTIC_ENABLED = False

def _load_semantic_model():
    global _semantic_model, _cosine_similarity, SEMANTIC_ENABLED
    
    if _semantic_model is not None:
        return SEMANTIC_ENABLED
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        _cosine_similarity = cosine_similarity
        SEMANTIC_ENABLED = True
        return True
    except ImportError:
        SEMANTIC_ENABLED = False
        return False


def split_into_sentences(text: str) -> List[str]:
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def sentence_keyword_similarity(sentence: str, keyword: str) -> float:
    if not _load_semantic_model():
        return 0.0
    
    try:
        sent_emb = _semantic_model.encode(sentence)
        kw_emb = _semantic_model.encode(keyword)
        similarity = _cosine_similarity([sent_emb], [kw_emb])[0][0]
        return float(similarity)
    except:
        return 0.0


def analyze_semantic_coverage(
    text: str, 
    keywords: List[str],
    high_threshold: float = 0.65,
    medium_threshold: float = 0.50,
    h1_threshold: float = 0.80
) -> Dict:
    if not _load_semantic_model():
        return {"semantic_enabled": False, "error": "Model not available"}
    
    sentences = split_into_sentences(text)
    
    if not sentences:
        return {
            "semantic_enabled": True,
            "sentence_count": 0,
            "keywords": {},
            "gaps": keywords,
            "overall_coverage": 0.0
        }
    
    try:
        sentence_embeddings = _semantic_model.encode(sentences)
    except:
        return {"semantic_enabled": False, "error": "Encoding failed"}
    
    results = {}
    gaps = []
    
    for keyword in keywords:
        if not keyword or not keyword.strip():
            continue
            
        try:
            kw_embedding = _semantic_model.encode(keyword)
            similarities = _cosine_similarity([kw_embedding], sentence_embeddings)[0]
            
            high_matches = []
            medium_matches = []
            
            for i, sim in enumerate(similarities):
                if sim >= high_threshold:
                    high_matches.append({
                        "sentence": sentences[i][:100] + "..." if len(sentences[i]) > 100 else sentences[i],
                        "similarity": round(float(sim), 3)
                    })
                elif sim >= medium_threshold:
                    medium_matches.append({
                        "sentence": sentences[i][:100] + "..." if len(sentences[i]) > 100 else sentences[i],
                        "similarity": round(float(sim), 3)
                    })
            
            high_matches.sort(key=lambda x: x["similarity"], reverse=True)
            medium_matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            best_sim = float(max(similarities))
            best_idx = int(similarities.argmax())
            
            if len(high_matches) >= 2:
                status = "STRONG"
            elif len(high_matches) >= 1 or len(medium_matches) >= 2:
                status = "MODERATE"
            elif len(medium_matches) >= 1:
                status = "WEAK"
            else:
                status = "GAP"
                gaps.append(keyword)
            
            results[keyword] = {
                "status": status,
                "high_matches": len(high_matches),
                "medium_matches": len(medium_matches),
                "best_similarity": round(best_sim, 3),
                "best_sentence": sentences[best_idx][:150] + "..." if len(sentences[best_idx]) > 150 else sentences[best_idx],
                "top_matches": high_matches[:3]
            }
            
        except:
            results[keyword] = {"status": "ERROR"}
    
    statuses = [r["status"] for r in results.values() if "status" in r]
    strong_count = statuses.count("STRONG")
    moderate_count = statuses.count("MODERATE")
    weak_count = statuses.count("WEAK")
    gap_count = statuses.count("GAP")
    total = len(statuses) if statuses else 1
    
    overall = (strong_count * 1.0 + moderate_count * 0.7 + weak_count * 0.3) / total
    
    return {
        "semantic_enabled": True,
        "sentence_count": len(sentences),
        "keyword_count": len(keywords),
        "keywords": results,
        "gaps": gaps,
        "summary": {
            "strong": strong_count,
            "moderate": moderate_count,
            "weak": weak_count,
            "gaps": gap_count
        },
        "overall_coverage": round(overall, 2)
    }


def find_semantic_gaps(text: str, keywords: List[str], threshold: float = 0.50) -> List[Dict]:
    result = analyze_semantic_coverage(text, keywords, medium_threshold=threshold)
    
    if not result.get("semantic_enabled"):
        return []
    
    gaps = []
    for keyword, data in result.get("keywords", {}).items():
        if data.get("status") in ["GAP", "WEAK"]:
            gaps.append({
                "keyword": keyword,
                "status": data.get("status"),
                "best_similarity": data.get("best_similarity", 0)
            })
    
    return sorted(gaps, key=lambda x: x.get("best_similarity", 0))


def count_semantic_occurrences(text: str, keyword: str, threshold: float = 0.60) -> int:
    if not _load_semantic_model():
        return 0
    
    sentences = split_into_sentences(text)
    if not sentences:
        return 0
    
    try:
        sentence_embeddings = _semantic_model.encode(sentences)
        kw_embedding = _semantic_model.encode(keyword)
        similarities = _cosine_similarity([kw_embedding], sentence_embeddings)[0]
        return sum(1 for sim in similarities if sim >= threshold)
    except:
        return 0


def analyze_section_coherence(
    text: str,
    drift_threshold: float = 0.6,
    global_threshold: float = 0.5
) -> Dict:
    """
    Fix #61: Coherence / Topic drift detection.
    Splits text by H2 headers, computes cosine similarity between consecutive
    sections, flags drops below drift_threshold as topic drift.

    Returns:
        {
            "coherence_enabled": True,
            "section_count": N,
            "pairwise_scores": [{"from": "H2a", "to": "H2b", "similarity": 0.72}, ...],
            "avg_coherence": 0.71,
            "min_coherence": 0.55,
            "drift_alerts": [{"from": "H2a", "to": "H2b", "similarity": 0.42}],
            "global_coherence": 0.68,
            "score": 75  # 0-100
        }
    """
    if not _load_semantic_model():
        return {"coherence_enabled": False, "error": "Model not available"}

    # Split by H2 headers (HTML or markdown)
    # Matches <h2>...</h2> or ## ...
    h2_pattern = re.compile(r'<h2[^>]*>(.*?)</h2>|^##\s+(.+)', re.IGNORECASE | re.MULTILINE)

    sections = []
    matches = list(h2_pattern.finditer(text))

    if len(matches) < 2:
        return {
            "coherence_enabled": True,
            "section_count": len(matches),
            "pairwise_scores": [],
            "avg_coherence": 1.0,
            "min_coherence": 1.0,
            "drift_alerts": [],
            "global_coherence": 1.0,
            "score": 100,
            "note": "Za mało sekcji H2 do analizy coherence"
        }

    for i, match in enumerate(matches):
        title = (match.group(1) or match.group(2) or "").strip()
        # Clean HTML tags from title
        title = re.sub(r'<[^>]+>', '', title).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        # Clean HTML for embedding
        clean_text = re.sub(r'<[^>]+>', ' ', section_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text.split()) >= 10:  # min 10 words
            sections.append({"title": title, "text": clean_text})

    if len(sections) < 2:
        return {
            "coherence_enabled": True,
            "section_count": len(sections),
            "pairwise_scores": [],
            "avg_coherence": 1.0,
            "min_coherence": 1.0,
            "drift_alerts": [],
            "global_coherence": 1.0,
            "score": 100,
            "note": "Za mało sekcji z treścią do analizy"
        }

    try:
        # Encode all sections
        embeddings = _semantic_model.encode([s["text"] for s in sections])

        # Pairwise consecutive similarity
        pairwise = []
        for i in range(len(sections) - 1):
            sim = float(_cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            pairwise.append({
                "from": sections[i]["title"],
                "to": sections[i + 1]["title"],
                "similarity": round(sim, 3)
            })

        # Drift alerts (below threshold)
        drift_alerts = [p for p in pairwise if p["similarity"] < drift_threshold]

        scores = [p["similarity"] for p in pairwise]
        avg_coherence = sum(scores) / len(scores) if scores else 1.0
        min_coherence = min(scores) if scores else 1.0

        # Global coherence: avg similarity of all pairs (not just consecutive)
        all_pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(_cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
                all_pairs.append(sim)
        global_coherence = sum(all_pairs) / len(all_pairs) if all_pairs else 1.0

        # Score 0-100
        # avg_coherence 0.8+ = 100, 0.6 = 70, 0.4 = 40, 0.2 = 10
        score = max(0, min(100, int(avg_coherence * 125 - 25)))
        # Penalty for drift alerts
        score = max(0, score - len(drift_alerts) * 8)

        return {
            "coherence_enabled": True,
            "section_count": len(sections),
            "pairwise_scores": pairwise,
            "avg_coherence": round(avg_coherence, 3),
            "min_coherence": round(min_coherence, 3),
            "drift_alerts": drift_alerts,
            "drift_count": len(drift_alerts),
            "global_coherence": round(global_coherence, 3),
            "score": score,
            "sections": [s["title"] for s in sections],
        }
    except Exception as e:
        return {"coherence_enabled": False, "error": str(e)}


def semantic_validation(text: str, keywords_state: Dict, min_coverage: float = 0.4) -> Dict:
    keywords = [
        meta.get("keyword", "") 
        for meta in keywords_state.values() 
        if meta.get("keyword")
    ]
    
    if not keywords:
        return {"valid": True, "semantic_enabled": False}
    
    result = analyze_semantic_coverage(text, keywords)
    
    if not result.get("semantic_enabled"):
        return {"valid": True, "semantic_enabled": False}
    
    overall = result.get("overall_coverage", 0)
    gaps = result.get("gaps", [])
    
    return {
        "valid": overall >= min_coverage,
        "semantic_enabled": True,
        "overall_coverage": overall,
        "min_required": min_coverage,
        "gaps": gaps,
        "gap_count": len(gaps),
        "summary": result.get("summary", {}),
        "warning": f"Pokrycie {overall:.0%} < {min_coverage:.0%}" if overall < min_coverage else None
    }
