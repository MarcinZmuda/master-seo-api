import re
import math
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

# v40.2: Import from core_metrics (Single Source of Truth)
try:
    from core_metrics import (
        calculate_burstiness_simple as _calculate_burstiness_core,
        split_into_sentences as _split_sentences_core
    )
    CORE_METRICS_AVAILABLE = True
except ImportError:
    CORE_METRICS_AVAILABLE = False

_semantic_model = None
_cosine_similarity = None
_sentence_cache = {}
_embedding_cache = {}

POLISH_STEMS = {
    'a': '', 'ą': '', 'e': '', 'ę': '', 'i': '', 'o': '', 'ó': '', 'u': '', 'y': '',
    'ow': '', 'ów': '', 'om': '', 'ami': '', 'ach': '', 'ie': '', 'ię': '',
    'em': '', 'iem': '', 'ą': '', 'ę': '', 'mi': '', 'owi': '', 'owie': '',
    'ość': '', 'ości': '', 'ością': '', 'ego': '', 'emu': '', 'ym': '', 'im': '',
    'ej': '', 'ą': '', 'ich': '', 'ych': '', 'imi': '', 'ymi': '',
    'cie': '', 'cji': '', 'cja': '', 'cję': '', 'cją': '',
    'nia': '', 'nie': '', 'niu': '', 'niem': '', 'ń': ''
}


def _load_semantic_model():
    global _semantic_model, _cosine_similarity
    if _semantic_model is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        _cosine_similarity = cosine_similarity
        return True
    except:
        return False


def get_stem(word: str, min_len: int = 4) -> str:
    word = word.lower().strip()
    if len(word) <= min_len:
        return word
    for suffix in sorted(POLISH_STEMS.keys(), key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            return word[:-len(suffix)]
    return word[:min_len] if len(word) > min_len else word


def split_sentences(text: str) -> List[str]:
    cache_key = hash(text[:500])
    if cache_key in _sentence_cache:
        return _sentence_cache[cache_key]
    
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    result = [s.strip() for s in sentences if len(s.split()) >= 5]
    
    _sentence_cache[cache_key] = result
    return result


def count_forms(text: str, keyword: str) -> int:
    if not text or not keyword:
        return 0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    if ' ' in keyword_lower:
        words = keyword_lower.split()
        stems = [get_stem(w) for w in words]
        
        count = 0
        text_words = text_lower.split()
        for i in range(len(text_words) - len(stems) + 1):
            match = True
            for j, stem in enumerate(stems):
                if not text_words[i + j].startswith(stem):
                    match = False
                    break
            if match:
                count += 1
        return count
    else:
        stem = get_stem(keyword_lower)
        return len(re.findall(r'\b' + re.escape(stem) + r'\w*', text_lower))


def calculate_density(text: str, keyword: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    count = count_forms(text, keyword)
    return round(count / len(words), 4)


# v40.2: Use core_metrics if available
if CORE_METRICS_AVAILABLE:
    def calculate_burstiness(text: str) -> float:
        """Deleguje do core_metrics.calculate_burstiness_simple"""
        return _calculate_burstiness_core(text)
else:
    def calculate_burstiness(text: str) -> float:
        """FALLBACK - use core_metrics instead"""
        sentences = split_sentences(text)
        if len(sentences) < 3:
            return 3.5
        
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        
        if mean_len == 0:
            return 3.5
        
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        burstiness = std_dev / mean_len
        return round(min(max(burstiness * 5, 1.0), 5.0), 2)


def calculate_transition_ratio(text: str) -> Dict:
    TRANSITION_WORDS = {
        'jednakże', 'jednak', 'natomiast', 'niemniej', 'aczkolwiek',
        'ponadto', 'dodatkowo', 'ponadto', 'również', 'także', 'też',
        'dlatego', 'zatem', 'więc', 'wobec tego', 'w związku z tym',
        'przede wszystkim', 'po pierwsze', 'po drugie', 'wreszcie', 'następnie',
        'przykładowo', 'na przykład', 'między innymi', 'w szczególności',
        'innymi słowy', 'to znaczy', 'mianowicie', 'czyli',
        'podsumowując', 'reasumując', 'ostatecznie', 'w rezultacie',
        'w przeciwieństwie', 'z drugiej strony', 'mimo to', 'pomimo',
        'co więcej', 'co ważne', 'warto zauważyć', 'należy podkreślić'
    }
    
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    
    if word_count == 0:
        return {"ratio": 0, "count": 0, "word_count": 0}
    
    transition_count = sum(1 for tw in TRANSITION_WORDS if tw in text_lower)
    ratio = transition_count / (word_count / 100)
    
    return {
        "ratio": round(ratio / 100, 2),
        "count": transition_count,
        "word_count": word_count
    }


def get_embeddings(texts: List[str], use_cache: bool = True) -> Optional[List]:
    if not _load_semantic_model():
        return None
    
    if use_cache:
        uncached = []
        uncached_idx = []
        for i, t in enumerate(texts):
            key = hash(t[:200])
            if key not in _embedding_cache:
                uncached.append(t)
                uncached_idx.append(i)
        
        if uncached:
            new_embeddings = _semantic_model.encode(uncached)
            for i, idx in enumerate(uncached_idx):
                key = hash(texts[idx][:200])
                _embedding_cache[key] = new_embeddings[i]
        
        return [_embedding_cache[hash(t[:200])] for t in texts]
    else:
        return _semantic_model.encode(texts)


def semantic_similarity(text1: str, text2: str) -> float:
    if not _load_semantic_model():
        return 0.0
    
    embeddings = get_embeddings([text1, text2])
    if embeddings is None:
        return 0.0
    
    sim = _cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim)


def analyze_semantic_coverage(
    text: str,
    keywords: List[str],
    h1_threshold: float = 0.80,
    h2_threshold: float = 0.50
) -> Dict:
    if not _load_semantic_model():
        return {"enabled": False}
    
    sentences = split_sentences(text)
    if not sentences:
        return {"enabled": True, "coverage": 0.0, "gaps": keywords}
    
    sent_embeddings = get_embeddings(sentences)
    if sent_embeddings is None:
        return {"enabled": False}
    
    results = {}
    gaps = []
    
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        
        kw_emb = get_embeddings([kw])
        if kw_emb is None:
            continue
        
        similarities = _cosine_similarity([kw_emb[0]], sent_embeddings)[0]
        best_sim = float(max(similarities))
        
        high_count = sum(1 for s in similarities if s >= 0.65)
        med_count = sum(1 for s in similarities if 0.50 <= s < 0.65)
        
        if high_count >= 2:
            status = "STRONG"
        elif high_count >= 1 or med_count >= 2:
            status = "MODERATE"
        elif med_count >= 1:
            status = "WEAK"
        else:
            status = "GAP"
            gaps.append(kw)
        
        results[kw] = {
            "status": status,
            "best_similarity": round(best_sim, 3),
            "high_matches": high_count,
            "medium_matches": med_count
        }
    
    statuses = [r["status"] for r in results.values()]
    strong = statuses.count("STRONG")
    moderate = statuses.count("MODERATE")
    weak = statuses.count("WEAK")
    total = len(statuses) if statuses else 1
    
    coverage = (strong * 1.0 + moderate * 0.7 + weak * 0.3) / total
    
    return {
        "enabled": True,
        "coverage": round(coverage, 2),
        "gaps": gaps,
        "keywords": results,
        "summary": {"strong": strong, "moderate": moderate, "weak": weak, "gaps": len(gaps)}
    }


def extract_headers(text: str) -> Dict[str, List[str]]:
    h1_pattern = r'(?:^h1:\s*(.+)$|<h1[^>]*>([^<]+)</h1>)'
    h2_pattern = r'(?:^h2:\s*(.+)$|<h2[^>]*>([^<]+)</h2>)'
    h3_pattern = r'(?:^h3:\s*(.+)$|<h3[^>]*>([^<]+)</h3>)'
    
    h1 = [(m[0] or m[1]).strip() for m in re.findall(h1_pattern, text, re.MULTILINE | re.IGNORECASE) if m[0] or m[1]]
    h2 = [(m[0] or m[1]).strip() for m in re.findall(h2_pattern, text, re.MULTILINE | re.IGNORECASE) if m[0] or m[1]]
    h3 = [(m[0] or m[1]).strip() for m in re.findall(h3_pattern, text, re.MULTILINE | re.IGNORECASE) if m[0] or m[1]]
    
    return {"h1": h1, "h2": h2, "h3": h3}


def full_text_analysis(
    text: str,
    main_keyword: str,
    keywords: List[str],
    title: str = "",
    h1: str = ""
) -> Dict:
    sentences = split_sentences(text)
    word_count = len(text.split())
    
    main_count = count_forms(text, main_keyword)
    main_density = calculate_density(text, main_keyword)
    
    keyword_counts = {kw: count_forms(text, kw) for kw in keywords if kw}
    
    burstiness = calculate_burstiness(text)
    transition = calculate_transition_ratio(text)
    
    semantic = analyze_semantic_coverage(text, keywords)
    
    headers = extract_headers(text)
    
    title_h1_match = 1.0
    if title and h1:
        title_h1_match = semantic_similarity(title, h1)
    
    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "main_keyword": {
            "keyword": main_keyword,
            "count": main_count,
            "density": main_density,
            "density_ok": main_density <= 0.015
        },
        "keywords": keyword_counts,
        "metrics": {
            "burstiness": burstiness,
            "burstiness_ok": 3.2 <= burstiness <= 3.8,
            "transition_ratio": transition["ratio"],
            "transition_ok": 0.20 <= transition["ratio"] <= 0.50
        },
        "semantic": semantic,
        "structure": {
            "headers": headers,
            "title_h1_similarity": round(title_h1_match, 2),
            "title_h1_ok": title_h1_match >= 0.60
        }
    }


def generate_batch_suggestions(
    project_data: Dict,
    current_keywords_state: Dict,
    completed_h2: List[str]
) -> Dict:
    h2_list = project_data.get("h2_list", [])
    pending_h2 = [h for h in h2_list if h not in completed_h2]
    
    if not pending_h2:
        return {"complete": True}
    
    next_h2 = pending_h2[0]
    
    under_keywords = []
    avoid_keywords = []
    
    for kw_id, meta in current_keywords_state.items():
        current = meta.get("current", 0)
        min_target = meta.get("min", 0)
        max_target = meta.get("max", 999)
        keyword = meta.get("keyword", "")
        
        if current < min_target:
            under_keywords.append({
                "keyword": keyword,
                "current": current,
                "needed": min_target - current
            })
        elif current >= max_target:
            avoid_keywords.append(keyword)
    
    under_keywords.sort(key=lambda x: x["needed"], reverse=True)
    
    full_text = project_data.get("full_text", "")
    keywords = [meta.get("keyword", "") for meta in current_keywords_state.values() if meta.get("keyword")]
    
    semantic_gaps = []
    if full_text and keywords:
        result = analyze_semantic_coverage(full_text, keywords)
        semantic_gaps = result.get("gaps", [])
    
    remaining_h2 = len(pending_h2)
    base_words = 300
    if remaining_h2 <= 2:
        suggested_words = 400
    elif remaining_h2 <= 4:
        suggested_words = 320
    else:
        suggested_words = base_words
    
    return {
        "complete": False,
        "next_h2": next_h2,
        "remaining_h2_count": remaining_h2,
        "must_use_keywords": [k["keyword"] for k in under_keywords[:5]],
        "keyword_priorities": under_keywords[:5],
        "avoid_keywords": avoid_keywords,
        "semantic_gaps": semantic_gaps[:5],
        "suggested_word_count": suggested_words,
        "guidance": _generate_guidance(next_h2, under_keywords, semantic_gaps)
    }


def _generate_guidance(h2: str, under_keywords: List[Dict], gaps: List[str]) -> str:
    parts = [f"Pisząc sekcję '{h2}':"]
    
    if under_keywords:
        kws = ", ".join([k["keyword"] for k in under_keywords[:3]])
        parts.append(f"- Użyj koniecznie: {kws}")
    
    if gaps:
        parts.append(f"- Wypełnij luki tematyczne: {', '.join(gaps[:3])}")
    
    return " ".join(parts)


def calculate_running_ratio(
    main_keyword: str,
    synonyms: List[str],
    current_keywords_state: Dict
) -> Dict:
    main_count = 0
    synonym_count = 0
    
    main_lower = main_keyword.lower()
    synonyms_lower = [s.lower() for s in synonyms]
    
    for kw_id, meta in current_keywords_state.items():
        keyword = meta.get("keyword", "").lower()
        current = meta.get("current", 0)
        
        if keyword == main_lower or get_stem(keyword) == get_stem(main_lower):
            main_count += current
        elif keyword in synonyms_lower:
            synonym_count += current
    
    total = main_count + synonym_count
    ratio = main_count / total if total > 0 else 1.0
    
    return {
        "main_count": main_count,
        "synonym_count": synonym_count,
        "total": total,
        "ratio": round(ratio, 2),
        "ratio_ok": ratio >= 0.30,
        "warning": f"Ratio {ratio:.0%} < 30%. Użyj więcej '{main_keyword}'!" if ratio < 0.30 else None
    }


def clear_caches():
    global _sentence_cache, _embedding_cache
    _sentence_cache = {}
    _embedding_cache = {}
