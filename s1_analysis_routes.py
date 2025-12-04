# ===========================================================
# S1 Analysis Routes - v12.25.6.6
# Competitor analysis with parallel URL processing
# ===========================================================

from flask import Blueprint, request, jsonify
import concurrent.futures
import time
import logging

s1_routes = Blueprint("s1_routes", __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# 🔍 PARALLEL URL PROCESSING
# ===========================================================

def process_single_url(url: str, timeout: int = 10) -> dict:
    """
    Process single URL with timeout
    Returns extracted data or None if failed
    """
    try:
        start_time = time.time()
        
        # Your LangExtract call here
        # content = call_langextract(url, timeout=timeout)
        # ... processing logic ...
        
        # Placeholder - replace with actual logic
        import requests
        response = requests.get(url, timeout=timeout)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Fetched {url} in {elapsed:.2f}s")
        
        return {
            "url": url,
            "content": response.text[:5000],  # First 5000 chars
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
    
    Args:
        urls: List of URLs to fetch
        max_workers: Number of parallel threads (default: 5)
        timeout: Timeout per URL in seconds (default: 10)
    
    Returns:
        List of successfully fetched data dicts
    
    Speed improvement: 5x faster than serial processing
    Example: 5 URLs @ 10s each = 50s serial → 10s parallel
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all URL fetch tasks
        future_to_url = {
            executor.submit(process_single_url, url, timeout): url 
            for url in urls
        }
        
        # Collect results as they complete
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
# 🔌 API ENDPOINT: S1 Analysis
# ===========================================================

@s1_routes.post("/api/s1_analysis")
def s1_analysis():
    """
    S1 Competitor Analysis with parallel URL fetching
    
    Changes in v12.25.6.6:
    - Parallel URL processing (5x faster)
    - Timeout handling per URL
    - Better error logging
    - Graceful degradation (continues if some URLs fail)
    """
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "")
    target_keywords = data.get("target_keywords", [])
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400
    
    logger.info(f"🔍 S1 Analysis started: {topic}")
    start_time = time.time()
    
    # Step 1: Get SerpAPI results (still serial, but fast)
    try:
        # serp_results = call_serp_api(topic, target_keywords)
        # top_urls = extract_top_urls(serp_results, limit=5)
        
        # Placeholder
        top_urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
            "https://example.com/article4",
            "https://example.com/article5"
        ]
        
        logger.info(f"📊 Found {len(top_urls)} competitor URLs")
        
    except Exception as e:
        logger.error(f"❌ SerpAPI failed: {e}")
        return jsonify({"error": "Failed to fetch SERP results"}), 500
    
    # Step 2: Fetch competitor content in PARALLEL (5x faster!)
    try:
        competitor_data = fetch_competitor_data_parallel(
            urls=top_urls,
            max_workers=5,  # Process 5 URLs simultaneously
            timeout=10      # 10s timeout per URL
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
        h2_counts = []
        for comp in competitor_data:
            # count_h2_in_html(comp['content'])
            h2_count = 7  # Placeholder
            h2_counts.append(h2_count)
        
        # Calculate average
        avg_h2 = sum(h2_counts) / len(h2_counts) if h2_counts else 8
        avg_h2 = round(avg_h2)
        
        logger.info(f"📊 H2 counts: {h2_counts} → avg: {avg_h2}")
        
        # Extract common topics (placeholder)
        common_topics = [
            {"topic": "Causes and symptoms", "frequency": 5},
            {"topic": "Prevention methods", "frequency": 4},
            {"topic": "Natural solutions", "frequency": 4},
            {"topic": "Professional treatments", "frequency": 3},
        ]
        
        # Extract top N-grams (placeholder)
        top_ngrams = [
            "zdrowe włosy",
            "pielęgnacja skóry",
            "naturalne składniki"
        ]
        
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
                "processing_time": round(elapsed, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return jsonify({"error": "Failed to analyze competitors"}), 500

# ===========================================================
# 🎯 MONITORING HELPER
# ===========================================================

@s1_routes.get("/api/s1_health")
def s1_health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "s1_analysis",
        "version": "12.25.6.6",
        "features": [
            "parallel_url_fetching",
            "timeout_handling",
            "error_logging"
        ]
    })
