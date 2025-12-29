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
