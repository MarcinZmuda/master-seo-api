from flask import Blueprint, request, jsonify
import concurrent.futures
import time
import logging
import os
import requests
from bs4 import BeautifulSoup
import re

s1_routes = Blueprint("s1_routes", __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERP_API_KEY = os.getenv("SERP_API_KEY")

def call_serp_api(topic: str, target_keywords: list) -> dict:
    if not SERP_API_KEY:
        logger.error("❌ SERP_API_KEY not set!")
        raise ValueError("SERP_API_KEY environment variable not configured")
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": topic,
            "location": "Poland",
            "hl": "pl",
            "gl": "pl",
            "api_key": SERP_API_KEY,
            "num": 10
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        logger.error(f"❌ SerpAPI error: {e}")
        raise

def extract_top_urls(serp_results: dict, limit: int = 5) -> list:
    urls = []
    organic_results = serp_results.get("organic_results", [])
    
    for result in organic_results[:limit]:
        url = result.get("link")
        if url:
            urls.append(url)
    
    logger.info(f"📊 Extracted {len(urls)} URLs from SERP")
    return urls

def extract_h2_data(html_content: str) -> dict:
    """
    Extract both H2 count AND titles from HTML.
    Returns: {"count": int, "titles": list}
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        h2_tags = soup.find_all('h2')
        
        h2_titles = []
        for h2 in h2_tags:
            text = h2.get_text().strip()
            # Clean up - remove extra whitespace
            text = ' '.join(text.split())
            
            # Filter out very short/empty H2
            if text and len(text) > 3:
                h2_titles.append(text)
        
        return {
            "count": len(h2_titles),
            "titles": h2_titles
        }
    except Exception as e:
        logger.warning(f"Error extracting H2: {e}")
        return {"count": 0, "titles": []}

def extract_text_content(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except:
        return ""

def process_single_url(url: str, timeout: int = 10) -> dict:
    try:
        start_time = time.time()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
        
        h2_data = extract_h2_data(response.text)
        text_content = extract_text_content(response.text)[:5000]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Fetched {url} in {elapsed:.2f}s - H2: {h2_data['count']}")
        
        return {
            "url": url,
            "h2_count": h2_data["count"],
            "h2_titles": h2_data["titles"],
            "content": text_content,
            "status": "success",
            "fetch_time": elapsed
        }
        
    except requests.Timeout:
        logger.warning(f"⏱️ Timeout fetching {url}")
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching {url}: {e}")
        return None

def fetch_competitor_data_parallel(urls: list, max_workers: int = 5, timeout: int = 10) -> list:
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(process_single_url, url, timeout): url 
            for url in urls
        }
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                    logger.info(f"✅ Processed {url}")
            except Exception as e:
                logger.error(f"❌ Exception processing {url}: {e}")
    
    return results

def extract_common_topics(competitor_data: list) -> list:
    all_text = " ".join([c.get("content", "") for c in competitor_data])
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = {}
    
    for word in words:
        if len(word) > 4:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    common_topics = [
        {"topic": word, "frequency": freq}
        for word, freq in sorted_words[:10]
    ]
    
    return common_topics

def extract_top_ngrams(competitor_data: list, n: int = 4, top_k: int = 20) -> list:
    """
    Extract real n-grams (default 4-grams) from competitor text.
    Returns top_k most frequent n-grams.
    """
    from collections import Counter
    
    all_text = " ".join([c.get("content", "") for c in competitor_data])
    
    # Tokenize (Polish-friendly - includes ąćęłńóśźż)
    words = re.findall(r'\b[a-ząćęłńóśźż]+\b', all_text.lower())
    
    # Create n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        
        # Basic filtering - skip if contains too many common stopwords
        stopword_count = sum(1 for stop in ["to", "jest", "się", "być", "oraz", "które"] if stop in ngram)
        if stopword_count < 2:  # Allow max 1 stopword per 4-gram
            ngrams.append(ngram)
    
    # Count frequency
    ngram_freq = Counter(ngrams)
    
    # Return top k
    return [
        {"ngram": ng, "frequency": freq}
        for ng, freq in ngram_freq.most_common(top_k)
    ]

def analyze_h2_topics(competitor_data: list) -> list:
    """
    Analyze H2 titles from competitors to find common topics.
    Returns topics sorted by frequency across competitors.
    """
    from collections import Counter
    
    all_h2_titles = []
    h2_by_competitor = []
    
    # Collect all H2 titles
    for competitor in competitor_data:
        h2_titles = competitor.get("h2_titles", [])
        all_h2_titles.extend(h2_titles)
        h2_by_competitor.append({
            "url": competitor.get("url", ""),
            "h2_titles": h2_titles
        })
    
    if not all_h2_titles:
        return []
    
    # Extract 2-3 word phrases from H2 titles (main topics)
    topic_freq = Counter()
    topic_examples = {}
    
    for h2 in all_h2_titles:
        h2_lower = h2.lower()
        words = re.findall(r'\b[a-ząćęłńóśźż]+\b', h2_lower)
        
        # Extract 2-word and 3-word phrases
        for length in [3, 2]:  # Try 3-word first, then 2-word
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                
                # Filter meaningful phrases (min 10 chars, not all stopwords)
                if len(phrase) >= 10:
                    stopwords = ["to", "jest", "się", "być", "oraz", "które", "przez", "dla"]
                    if not all(word in stopwords for word in words[i:i+length]):
                        topic_freq[phrase] += 1
                        
                        # Store example H2 titles for this topic
                        if phrase not in topic_examples:
                            topic_examples[phrase] = []
                        if h2 not in topic_examples[phrase]:
                            topic_examples[phrase].append(h2)
    
    # Calculate coverage (how many competitors have this topic)
    total_competitors = len(competitor_data)
    
    # Return top 15 topics with examples
    return [
        {
            "topic": topic,
            "frequency": freq,
            "coverage": f"{min(freq, total_competitors)}/{total_competitors}",
            "example_titles": topic_examples.get(topic, [])[:3]  # Max 3 examples
        }
        for topic, freq in topic_freq.most_common(15)
    ]

@s1_routes.post("/api/s1_analysis")
def s1_analysis():
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "")
    target_keywords = data.get("target_keywords", [])
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400
    
    logger.info(f"🔍 S1 Analysis started: {topic}")
    start_time = time.time()
    
    try:
        serp_results = call_serp_api(topic, target_keywords)
        top_urls = extract_top_urls(serp_results, limit=5)
        
        if not top_urls:
            return jsonify({"error": "No URLs found in SERP results"}), 500
        
        logger.info(f"📊 Found {len(top_urls)} competitor URLs")
        
    except Exception as e:
        logger.error(f"❌ SerpAPI failed: {e}")
        return jsonify({"error": f"Failed to fetch SERP results: {str(e)}"}), 500
    
    try:
        competitor_data = fetch_competitor_data_parallel(
            urls=top_urls,
            max_workers=5,
            timeout=10
        )
        
        logger.info(f"✅ Fetched {len(competitor_data)}/{len(top_urls)} competitors")
        
        if len(competitor_data) == 0:
            return jsonify({"error": "Failed to fetch any competitor data"}), 500
        
    except Exception as e:
        logger.error(f"❌ Parallel fetch failed: {e}")
        return jsonify({"error": "Failed to process competitors"}), 500
    
    try:
        h2_counts = [c.get("h2_count", 0) for c in competitor_data]
        avg_h2 = sum(h2_counts) / len(h2_counts) if h2_counts else 7
        avg_h2 = round(avg_h2)
        
        logger.info(f"📊 H2 counts: {h2_counts} → avg: {avg_h2}")
        
        # NEW: Calculate average article length from competitors
        competitor_lengths = []
        for comp in competitor_data:
            word_count = len(comp.get("full_text", "").split())
            if word_count > 0:
                competitor_lengths.append(word_count)
        
        if competitor_lengths:
            avg_competitor_length = int(sum(competitor_lengths) / len(competitor_lengths))
            min_length = min(competitor_lengths)
            max_length = max(competitor_lengths)
        else:
            avg_competitor_length = 2000  # Default if can't calculate
            min_length = 1500
            max_length = 2500
        
        logger.info(f"📊 Article lengths: avg={avg_competitor_length}w, range={min_length}-{max_length}w")
        
        # NEW: Analyze H2 topics from competitors
        h2_topics = analyze_h2_topics(competitor_data)
        logger.info(f"📊 Found {len(h2_topics)} unique H2 topics")
        
        # Extract common words (legacy - kept for backward compatibility)
        common_topics = extract_common_topics(competitor_data)
        
        # Extract real n-grams (not hardcoded)
        top_ngrams = extract_top_ngrams(competitor_data, n=4, top_k=20)
        logger.info(f"📊 Extracted {len(top_ngrams)} 4-grams")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ S1 Analysis completed in {elapsed:.2f}s")
        
        return jsonify({
            "status": "success",
            "analysis": {
                "avg_h2_count": avg_h2,
                "h2_counts": h2_counts,
                "avg_article_length": avg_competitor_length,  # NEW
                "article_length_range": {  # NEW
                    "min": min_length,
                    "max": max_length,
                    "avg": avg_competitor_length
                },
                "h2_topics": h2_topics,  # NEW: Actual H2 topics with examples
                "common_topics": common_topics,  # Legacy: single words
                "top_ngrams": top_ngrams,  # Real n-grams (not hardcoded)
                "competitors_analyzed": len(competitor_data),
                "competitor_urls": [c["url"] for c in competitor_data],
                "processing_time": round(elapsed, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return jsonify({"error": f"Failed to analyze competitors: {str(e)}"}), 500

@s1_routes.get("/api/s1_health")
def s1_health():
    serp_configured = bool(SERP_API_KEY)
    
    return jsonify({
        "status": "healthy",
        "service": "s1_analysis",
        "version": "12.25.6.15",
        "serp_api_configured": serp_configured,
        "features": [
            "real_serp_api",
            "parallel_url_fetching",
            "h2_counting",
            "topic_extraction"
        ]
    })
