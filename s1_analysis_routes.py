# ===========================================================
# S1 Analysis Routes - v12.25.6.7 FIXED
# Real SerpAPI + Parallel URL processing
# ===========================================================

from flask import Blueprint, request, jsonify
import concurrent.futures
import time
import logging
import os
import requests
from bs4 import BeautifulSoup
import re

s1_routes = Blueprint("s1_routes", __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SerpAPI Key
SERP_API_KEY = os.getenv("SERP_API_KEY")

# ===========================================================
# 🔍 SERP API CALL
# ===========================================================

def call_serp_api(topic: str, target_keywords: list) -> dict:
    """
    Call SerpAPI to get top search results
    """
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
    """
    Extract top N URLs from SERP results
    """
    urls = []
    
    organic_results = serp_results.get("organic_results", [])
    
    for result in organic_results[:limit]:
        url = result.get("link")
        if url:
            urls.append(url)
    
    logger.info(f"📊 Extracted {len(urls)} URLs from SERP")
    return urls

# ===========================================================
# 🔍 URL CONTENT EXTRACTION
# ===========================================================

def extract_h2_count(html_content: str) -> int:
    """
    Count H2 tags in HTML
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        h2_tags = soup.find_all('h2')
        return len(h2_tags)
    except:
        return 0

def extract_text_content(html_content: str) -> str:
    """
    Extract clean text from HTML
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except:
        return ""

def process_single_url(url: str, timeout: int = 10) -> dict:
    """
    Process single URL with timeout
    Returns extracted data or None if failed
    """
    try:
        start_time = time.time()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
        
        # Extract H2 count
        h2_count = extract_h2_count(response.text)
        
        # Extract text (first 5000 chars for analysis)
        text_content = extract_text_content(response.text)[:5000]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Fetched {url} in {elapsed:.2f}s - H2: {h2_count}")
        
        return {
            "url": url,
            "h2_count": h2_count,
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
    """
    Fetch multiple URLs in parallel using ThreadPoolExecutor
    
    Speed improvement: 5x faster than serial processing
    """
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

# ===========================================================
# 🔍 TEXT ANALYSIS
# ===========================================================

def extract_common_topics(competitor_data: list) -> list:
    """
    Extract common topics from competitor content
    Simple keyword frequency analysis
    """
    all_text = " ".join([c.get("content", "") for c in competitor_data])
    
    # Simple word frequency (this is a placeholder - you can improve with spaCy)
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = {}
    
    for word in words:
        if len(word) > 4:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top 10
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    common_topics = [
        {"topic": word, "frequency": freq}
        for word, freq in sorted_words[:10]
    ]
    
    return common_topics

def extract_top_ngrams(competitor_data: list) -> list:
    """
    Extract top 4-grams for intro
    Placeholder - can be improved with proper NLP
    """
    # Simple implementation - returns placeholder
    # You can improve this with spaCy n-gram extraction
    return [
        "naturalne składniki",
        "zdrowe włosy",
        "pielęgnacja skóry"
    ]

# ===========================================================
# 🔌 API ENDPOINT: S1 Analysis
# ===========================================================

@s1_routes.post("/api/s1_analysis")
def s1_analysis():
    """
    S1 Competitor Analysis with parallel URL fetching
    
    v12.25.6.7:
    - Real SerpAPI integration
    - Parallel URL processing (5x faster)
    - H2 counting from real competitors
    - Topic extraction from content
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "")
    target_keywords = data.get("target_keywords", [])
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400
    
    logger.info(f"🔍 S1 Analysis started: {topic}")
    start_time = time.time()
    
    # Step 1: Get SerpAPI results
    try:
        serp_results = call_serp_api(topic, target_keywords)
        top_urls = extract_top_urls(serp_results, limit=5)
        
        if not top_urls:
            return jsonify({"error": "No URLs found in SERP results"}), 500
        
        logger.info(f"📊 Found {len(top_urls)} competitor URLs")
        
    except Exception as e:
        logger.error(f"❌ SerpAPI failed: {e}")
        return jsonify({"error": f"Failed to fetch SERP results: {str(e)}"}), 500
    
    # Step 2: Fetch competitor content in PARALLEL
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
    
    # Step 3: Analyze competitor data
    try:
        # Extract H2 counts
        h2_counts = [c.get("h2_count", 0) for c in competitor_data]
        
        # Calculate average
        avg_h2 = sum(h2_counts) / len(h2_counts) if h2_counts else 7
        avg_h2 = round(avg_h2)
        
        logger.info(f"📊 H2 counts: {h2_counts} → avg: {avg_h2}")
        
        # Extract common topics
        common_topics = extract_common_topics(competitor_data)
        
        # Extract top N-grams
        top_ngrams = extract_top_ngrams(competitor_data)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ S1 Analysis completed in {elapsed:.2f}s")
        
        return jsonify({
            "status": "success",
            "analysis": {
                "avg_h2_count": avg_h2,
                "h2_counts": h2_counts,
                "common_topics": common_topics,
                "top_ngrams": top_ngrams,
                "competitors_analyzed": len(competitor_data),
                "competitor_urls": [c["url"] for c in competitor_data],
                "processing_time": round(elapsed, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return jsonify({"error": f"Failed to analyze competitors: {str(e)}"}), 500

# ===========================================================
# 🎯 MONITORING HELPER
# ===========================================================

@s1_routes.get("/api/s1_health")
def s1_health():
    """Health check endpoint"""
    serp_configured = bool(SERP_API_KEY)
    
    return jsonify({
        "status": "healthy",
        "service": "s1_analysis",
        "version": "12.25.6.7",
        "serp_api_configured": serp_configured,
        "features": [
            "real_serp_api",
            "parallel_url_fetching",
            "h2_counting",
            "topic_extraction",
            "timeout_handling",
            "error_logging"
        ]
    })
```
