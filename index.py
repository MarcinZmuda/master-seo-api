import os
import json
import re
import requests
from collections import Counter, defaultdict
from flask import Flask, request, jsonify
import spacy
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# 🆕 v28.0: trafilatura for clean content extraction (eliminates CSS garbage)
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
    print("[S1] ✅ trafilatura loaded — clean content extraction")
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("[S1] ⚠️ trafilatura not installed — using regex fallback (may include CSS garbage)")

# ======================================================
# ⭐ v22.3 LIMITS - zapobieganie OOM
# ======================================================
MAX_CONTENT_SIZE = 30000      # Max 30KB per page (było unlimited → 175KB crash)
MAX_TOTAL_CONTENT = 200000    # Max 200KB total content
SCRAPE_TIMEOUT = 8            # 8 sekund timeout per page (było 10)
SKIP_DOMAINS = ['bip.', '.pdf', 'gov.pl/dana/', '/uploads/files/']  # Skip duże dokumenty

# ======================================================
# 🔑 SerpAPI Configuration
# ======================================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
if SERPAPI_KEY:
    print("[S1] ✅ SerpAPI key configured")
else:
    print("[S1] ⚠️ SERPAPI_KEY not set — auto-fetch disabled")

# ======================================================
# 🔥 Firebase Initialization (Safe for Render & Local)
# ======================================================
if not firebase_admin._apps:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print(f"[S1] ✅ Firebase initialized from credentials file: {cred_path}")
        else:
            firebase_admin.initialize_app()
            print("[S1] ✅ Firebase initialized with default credentials")
    except Exception as e:
        print(f"[S1] ⚠️ Firebase init skipped: {e}")

# ======================================================
# ⚙️ Gemini (Google Generative AI) Configuration
# ======================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[S1] ✅ Gemini API configured")
else:
    print("[S1] ⚠️ GEMINI_API_KEY not set — semantic extraction fallback active")

# ======================================================
# 🧠 Import local modules (compatible with both local and Render)
# ======================================================
try:
    from .synthesize_topics import synthesize_topics
    from .generate_compliance_report import generate_compliance_report
    from .entity_extractor import perform_entity_seo_analysis
except ImportError:
    from synthesize_topics import synthesize_topics
    from generate_compliance_report import generate_compliance_report
    from entity_extractor import perform_entity_seo_analysis

# Flag do włączania/wyłączania Entity SEO
ENTITY_SEO_ENABLED = os.getenv("ENTITY_SEO_ENABLED", "true").lower() == "true"
print(f"[S1] {'✅' if ENTITY_SEO_ENABLED else '⚠️'} Entity SEO: {'ENABLED' if ENTITY_SEO_ENABLED else 'DISABLED'}")

# 🆕 v45.0: Causal Triplet Extractor
CAUSAL_EXTRACTOR_ENABLED = False
try:
    try:
        from .causal_extractor import extract_causal_triplets, format_causal_for_agent
    except ImportError:
        from causal_extractor import extract_causal_triplets, format_causal_for_agent
    CAUSAL_EXTRACTOR_ENABLED = True
    print("[S1] ✅ Causal Triplet Extractor v1.0 enabled")
except ImportError:
    print("[S1] ℹ️ Causal Triplet Extractor not available")

# 🆕 v45.0: Gap Analyzer
GAP_ANALYZER_ENABLED = False
try:
    try:
        from .gap_analyzer import analyze_content_gaps
    except ImportError:
        from gap_analyzer import analyze_content_gaps
    GAP_ANALYZER_ENABLED = True
    print("[S1] ✅ Gap Analyzer v1.0 enabled")
except ImportError:
    print("[S1] ℹ️ Gap Analyzer not available")


def _generate_paa_claude_fallback(keyword: str, serp_data: dict) -> list:
    """
    Generate PAA questions using Claude when SerpAPI returns no related_questions.
    Uses top SERP snippets + AI Overview as context.
    """
    try:
        import anthropic, os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[S1] ⚠️ PAA fallback: brak ANTHROPIC_API_KEY")
            return []

        # Build context from snippets
        snippets = []
        for r in serp_data.get("organic_results", [])[:6]:
            s = r.get("snippet", "")
            if s:
                snippets.append(s)
        
        ai_overview_text = ""
        aio = serp_data.get("ai_overview", {})
        if isinstance(aio, dict):
            ai_overview_text = aio.get("text", "") or aio.get("snippet", "")
        elif isinstance(aio, str):
            ai_overview_text = aio

        context_parts = []
        if snippets:
            context_parts.append("Fragmenty z SERP:\n" + "\n".join(f"- {s}" for s in snippets))
        if ai_overview_text:
            context_parts.append(f"Google AI Overview:\n{ai_overview_text[:400]}")

        context = "\n\n".join(context_parts) if context_parts else f"Temat: {keyword}"

        prompt = f"""Dla zapytania "{keyword}" wygeneruj 6 pytań z sekcji Google "Ludzie pytają też" (PAA).
        
Kontekst z SERP:
{context}

Zwróć TYLKO JSON array (bez markdown):
[
  {{"question": "Pytanie 1?", "answer": "Krótka odpowiedź 1-2 zdania"}},
  {{"question": "Pytanie 2?", "answer": "Krótka odpowiedź 1-2 zdania"}}
]

6 pytań. Pytania muszą być naturalne, jak rzeczywiście zadają je użytkownicy Google."""

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        
        # Parse JSON
        first = raw.find("[")
        last = raw.rfind("]")
        if first == -1 or last == -1:
            return []
        
        import json
        items = json.loads(raw[first:last+1])
        result = []
        for item in items[:6]:
            if isinstance(item, dict) and item.get("question"):
                result.append({
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "source": "claude_fallback"
                })
        
        print(f"[S1] ✅ Claude PAA fallback: {len(result)} questions generated")
        return result
        
    except Exception as e:
        print(f"[S1] ⚠️ PAA fallback error: {e}")
        return []


app = Flask(__name__)

# ======================================================
# 🧩 Load spaCy model (preinstalled lightweight version)
# ======================================================
try:
    nlp = spacy.load("pl_core_news_sm")
    print("[S1] ✅ spaCy pl_core_news_sm loaded")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")
    print("[S1] ✅ spaCy model downloaded and loaded")

# ======================================================
# ⭐ v22.3 Helper: Check if URL should be skipped
# ======================================================
def should_skip_url(url):
    """Sprawdza czy URL powinien być pominięty (duże dokumenty, PDF, BIP)."""
    url_lower = url.lower()
    for skip_pattern in SKIP_DOMAINS:
        if skip_pattern in url_lower:
            return True
    # Skip jeśli URL kończy się na rozszerzenie pliku
    if any(url_lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
        return True
    return False

# ======================================================
# 🧠 Helper: Semantic extraction using Gemini Flash
# ======================================================
def extract_semantic_tags_gemini(text, top_n=10):
    """Używa Google Gemini Flash do wyciągnięcia fraz semantycznych."""
    if not GEMINI_API_KEY or not (text or "").strip():
        return []

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Jesteś ekspertem SEO. Przeanalizuj poniższy tekst i wypisz {top_n} najważniejszych fraz kluczowych (semantic keywords), które najlepiej oddają jego sens.
        Zwróć TYLKO listę po przecinku, bez numerowania.

        TEKST: {text[:8000]}...
        """
        response = model.generate_content(prompt)
        keywords = [k.strip() for k in (response.text or "").split(",") if k.strip()]
        return [{"phrase": kw, "score": 0.95 - (i * 0.02)} for i, kw in enumerate(keywords[:top_n])]
    except Exception as e:
        print(f"[S1] ❌ Gemini Semantic Error: {e}")
        return []

# ======================================================
# 💡 Helper: Generate Content Hints (inspiracje dla GPT)
# ======================================================
def generate_content_hints(serp_analysis, main_keyword):
    """
    Przekształca surowe dane SERP w subtelne wskazówki dla GPT.
    To są INSPIRACJE, nie twarde reguły - GPT ma je traktować jako tło.
    """
    hints = {}

    # 1️⃣ INTRO INSPIRATION - z Featured Snippet / AI Overview
    featured = serp_analysis.get("featured_snippet")
    if featured and isinstance(featured, dict) and featured.get("answer"):
        hints["intro_inspiration"] = {
            "google_promotes": featured.get("answer", "")[:500],
            "source_type": featured.get("type", "unknown"),
            "hint": "Google wyróżnia tę odpowiedź w wynikach. Rozważ napisanie lepszego/pełniejszego wstępu który naturalnie odpowiada na to samo pytanie. NIE kopiuj - napisz wartościowszą wersję."
        }

    # 2️⃣ QUESTIONS USERS ASK - z PAA
    paa = serp_analysis.get("paa_questions", [])
    if paa:
        questions = [q.get("question", "") for q in paa if isinstance(q, dict) and q.get("question")][:6]
        hints["questions_users_ask"] = {
            "questions": questions,
            "hint": "Użytkownicy często pytają o te rzeczy. Jeśli pasują do tematu, rozważ naturalne poruszenie w treści. Nie musisz odpowiadać na wszystkie - wybierz relevantne."
        }

        # Bonus: krótkie odpowiedzi jako kontekst
        qa_context = []
        for q in paa[:3]:
            if isinstance(q, dict) and q.get("question") and q.get("answer"):
                qa_context.append({
                    "q": q.get("question"),
                    "current_answer": (q.get("answer", "") or "")[:200]
                })
        if qa_context:
            hints["questions_users_ask"]["current_answers_preview"] = qa_context

    # 3️⃣ RELATED TOPICS - z Related Searches
    related = serp_analysis.get("related_searches", [])
    if related:
        hints["related_topics"] = {
            "topics": related[:8],
            "hint": "Powiązane frazy wyszukiwane przez użytkowników. Mogą naturalnie pojawić się w tekście jeśli są relevantne. Nie upychaj na siłę."
        }

    # 4️⃣ COMPETITOR INSIGHTS - z tytułów i snippetów
    titles = serp_analysis.get("competitor_titles", [])
    snippets = serp_analysis.get("competitor_snippets", [])
    if titles or snippets:
        hints["competitor_insights"] = {
            "hint": "Tak konkurencja prezentuje temat w SERP. Tylko dla orientacji - Twoje podejście może być inne i lepsze."
        }
        if titles:
            hints["competitor_insights"]["title_patterns"] = titles[:5]
        if snippets:
            hints["competitor_insights"]["description_samples"] = snippets[:3]

    # 5️⃣ STRUCTURE INSPIRATION - z H2 konkurencji
    h2_patterns = serp_analysis.get("competitor_h2_patterns", [])
    if h2_patterns:
        unique_h2 = list(dict.fromkeys(h2_patterns))[:10]
        hints["structure_inspiration"] = {
            "competitor_sections": unique_h2,
            "hint": "Przykładowe sekcje używane przez konkurencję. Twoja struktura może być inna - to tylko kontekst co inni poruszają."
        }

    # 6️⃣ META HINT - ogólna wskazówka
    hints["_meta"] = {
        "interpretation": "Te wskazówki to TŁO i INSPIRACJA, nie checklist. Artykuł ma być naturalny, wartościowy i unikalny. Używaj tych danych żeby lepiej zrozumieć intencję użytkownika, nie żeby mechanicznie odpowiadać na każdy punkt.",
        "priority": "Jakość treści > dopasowanie do SERP"
    }

    return hints

# ======================================================
# 🔍 Helper: Fetch sources from SerpAPI (FULL SERP DATA)
# ======================================================
def fetch_serp_sources(keyword, num_results=10):
    """
    Pobiera PEŁNE dane z Google przez SerpAPI:
    - Organic results (top 10 stron) + scrapuje ich pełną treść
    - PAA (People Also Ask)
    - Featured Snippet
    - Related Searches
    - Tytuły i snippety z SERP
    
    ⭐ v22.3: Dodano limity rozmiaru i skip dla dużych dokumentów
    """
    empty_result = {
        "sources": [],
        "paa": [],
        "featured_snippet": None,
        "ai_overview": None,  # v27.0
        "related_searches": [],
        "serp_titles": [],
        "serp_snippets": []
    }

    if not SERPAPI_KEY:
        print("[S1] ⚠️ SerpAPI key not configured - cannot fetch sources")
        return empty_result

    try:
        print(f"[S1] 🔍 Fetching FULL SERP data for: {keyword}")
        serp_response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": keyword,
                "api_key": SERPAPI_KEY,
                "num": num_results,
                "hl": "pl",
                "gl": "pl"
            },
            timeout=30
        )

        if serp_response.status_code != 200:
            print(f"[S1] ❌ SerpAPI error: {serp_response.status_code}")
            return empty_result

        serp_data = serp_response.json()

        # ⭐ v27.0: Wyciągnij AI Overview (Google SGE)
        ai_overview = None
        ai_overview_data = serp_data.get("ai_overview", {})
        if ai_overview_data:
            ai_overview = {
                "text": ai_overview_data.get("text", "") or ai_overview_data.get("snippet", ""),
                "sources": [
                    {
                        "title": src.get("title", ""),
                        "link": src.get("link", ""),
                        "snippet": src.get("snippet", "")
                    }
                    for src in ai_overview_data.get("sources", [])[:5]
                ],
                "text_blocks": ai_overview_data.get("text_blocks", [])
            }
            print(f"[S1] ✅ Found AI Overview ({len(ai_overview.get('text', ''))} chars)")

        # ⭐ 2. Wyciągnij PAA (People Also Ask)
        paa_questions = []
        related_questions = serp_data.get("related_questions", [])
        for q in related_questions:
            paa_questions.append({
                "question": q.get("question", ""),
                "answer": q.get("snippet", ""),
                "source": q.get("link", ""),
                "title": q.get("title", "")
            })
        if paa_questions:
            print(f"[S1] ✅ Found {len(paa_questions)} PAA questions")
        else:
            print(f"[S1] ⚠️ No PAA from SerpAPI — generating with Claude fallback...")
            paa_questions = _generate_paa_claude_fallback(main_keyword, serp_data)

        # ⭐ 3. Wyciągnij Featured Snippet (Answer Box)
        featured_snippet = None
        answer_box = serp_data.get("answer_box", {})
        if answer_box:
            featured_snippet = {
                "type": answer_box.get("type", "unknown"),
                "title": answer_box.get("title", ""),
                "answer": answer_box.get("answer", "") or answer_box.get("snippet", ""),
                "source": answer_box.get("link", ""),
                "displayed_link": answer_box.get("displayed_link", "")
            }
            print(f"[S1] ✅ Found Featured Snippet: {featured_snippet.get('type')}")

        # ⭐ 4. Wyciągnij Related Searches
        related_searches = []
        for rs in serp_data.get("related_searches", []):
            query = rs.get("query", "")
            if query:
                related_searches.append(query)
        if related_searches:
            print(f"[S1] ✅ Found {len(related_searches)} related searches")

        # ⭐ 5. Wyciągnij tytuły i snippety z organic results
        organic_results = serp_data.get("organic_results", [])
        serp_titles = []
        serp_snippets = []

        for result in organic_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            if title:
                serp_titles.append(title)
            if snippet:
                serp_snippets.append(snippet)

        if not organic_results:
            print("[S1] ⚠️ No organic results from SerpAPI")
            return {
                "sources": [],
                "paa": paa_questions,
                "featured_snippet": featured_snippet,
                "ai_overview": ai_overview,  # v27.0
                "related_searches": related_searches,
                "serp_titles": serp_titles,
                "serp_snippets": serp_snippets
            }

        print(f"[S1] ✅ Found {len(organic_results)} SERP results")

        # ⭐ 6. Scrapuj PEŁNĄ treść każdej strony + strukturę H2
        # ⭐ v47.1: Parallel scraping with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        def _scrape_one(item):
            """Scrape a single URL — runs in thread pool."""
            url = item.get("link", "")
            title = item.get("title", "")
            if not url:
                return None
            if should_skip_url(url):
                print(f"[S1] ⏭️ Skipping large doc pattern: {url[:50]}...")
                return None

            t0 = _time.time()
            try:
                page_response = requests.get(
                    url,
                    timeout=SCRAPE_TIMEOUT,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                )

                if page_response.status_code != 200:
                    print(f"[S1] ⚠️ HTTP {page_response.status_code} from {url[:40]}")
                    return None

                # v52.4: Smart encoding — requests domyślnie używa ISO-8859-1 dla text/html
                # bez deklaracji charset w nagłówkach, co powoduje pojÄciem zamiast pojęciem.
                content_type = page_response.headers.get('Content-Type', '')
                if 'charset=' in content_type.lower():
                    raw_html = page_response.text  # Zaufaj zadeklarowanemu charset
                else:
                    try:
                        raw_html = page_response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            raw_html = page_response.content.decode('windows-1250')
                        except UnicodeDecodeError:
                            raw_html = page_response.content.decode('utf-8', errors='replace')

                # Limit content size PRZED przetwarzaniem
                if len(raw_html) > MAX_CONTENT_SIZE * 2:
                    print(f"[S1] ⚠️ Content too large ({len(raw_html)} chars), truncating: {url[:40]}")
                    raw_html = raw_html[:MAX_CONTENT_SIZE * 2]

                # Wyciągnij H2 PRZED usunięciem tagów
                h2_tags = re.findall(r'<h2[^>]*>(.*?)</h2>', raw_html, re.IGNORECASE | re.DOTALL)
                h2_clean = [re.sub(r'<[^>]+>', '', h).strip() for h in h2_tags]
                h2_clean = [h for h in h2_clean if h and len(h) < 200 and not re.search(r'[{};]|webkit|moz-|flex-|align-items', h, re.IGNORECASE)]

                # Ekstrakcja treści — trafilatura lub regex fallback
                content = None
                if TRAFILATURA_AVAILABLE:
                    try:
                        content = trafilatura.extract(
                            raw_html,
                            include_comments=False,
                            include_tables=True,
                            no_fallback=False,
                            favor_precision=True
                        )
                    except Exception as e:
                        print(f"[S1] ⚠️ trafilatura failed for {url[:40]}: {e}")
                        content = None

                # Fallback: regex
                if not content:
                    content = raw_html
                    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = re.sub(r'\s+', ' ', content).strip()

                content = content[:MAX_CONTENT_SIZE]
                elapsed = _time.time() - t0

                if len(content) > 500:
                    word_count = len(content.split())
                    print(f"[S1] ✅ Scraped {len(content)} chars ({word_count} words), {len(h2_clean)} H2 from {url[:40]} [{elapsed:.1f}s]")
                    return {
                        "url": url,
                        "title": title,
                        "content": content,
                        "h2_structure": h2_clean[:15],
                        "word_count": word_count
                    }
                else:
                    print(f"[S1] ⚠️ Too short content from {url[:40]}")
                    return None

            except requests.exceptions.Timeout:
                print(f"[S1] ⏱️ Timeout for {url[:40]} (>{SCRAPE_TIMEOUT}s)")
                return None
            except Exception as e:
                print(f"[S1] ⚠️ Scrape error for {url[:40]}: {e}")
                return None

        # Launch all scrapes in parallel (max 6 threads)
        scrape_targets = [r for r in organic_results[:num_results] if r.get("link")]
        t_start = _time.time()
        print(f"[S1] 🚀 Parallel scraping {len(scrape_targets)} pages...")

        sources = []
        total_content_size = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_scrape_one, item): item for item in scrape_targets}
            for future in as_completed(futures):
                result = future.result()
                if result and total_content_size < MAX_TOTAL_CONTENT:
                    sources.append(result)
                    total_content_size += len(result["content"])

        t_elapsed = _time.time() - t_start
        print(f"[S1] ✅ Parallel scrape done: {len(sources)} sources ({total_content_size} chars) in {t_elapsed:.1f}s")


        return {
            "sources": sources,
            "paa": paa_questions,
            "featured_snippet": featured_snippet,
            "ai_overview": ai_overview,  # v27.0
            "related_searches": related_searches,
            "serp_titles": serp_titles,
            "serp_snippets": serp_snippets
        }

    except Exception as e:
        print(f"[S1] ❌ SerpAPI fetch error: {e}")
        return empty_result

# ======================================================
# 🔍 Endpoint: N-gram + Semantic + SERP Analysis + Firestore Save
# ======================================================
@app.route("/api/ngram_entity_analysis", methods=["POST"])
def perform_ngram_analysis():
    data = request.get_json(force=True)
    
    # v27.0: Akceptuj zarówno "keyword" jak i "main_keyword"
    main_keyword = data.get("main_keyword") or data.get("keyword", "")
    
    sources = data.get("sources", [])
    top_n = int(data.get("top_n", 30))
    project_id = data.get("project_id")

    # ⭐ Zmienne na dodatkowe dane SERP
    paa_questions = []
    featured_snippet = None
    ai_overview = None  # v27.0: Google SGE
    related_searches = []
    serp_titles = []
    serp_snippets = []
    h2_patterns = []

    # ⭐ AUTO-FETCH: Jeśli brak sources, pobierz PEŁNE dane z SerpAPI
    if not sources:
        if not main_keyword:
            return jsonify({"error": "Brak main_keyword do analizy"}), 400

        print(f"[S1] 🔄 No sources provided - auto-fetching FULL SERP data...")
        serp_result = fetch_serp_sources(main_keyword, num_results=8)  # ⭐ v22.3: Reduced from 10 to 8

        # Wyciągnij wszystkie dane z rezultatu
        sources = serp_result.get("sources", [])
        paa_questions = serp_result.get("paa", [])
        featured_snippet = serp_result.get("featured_snippet")
        ai_overview = serp_result.get("ai_overview")  # v27.0
        related_searches = serp_result.get("related_searches", [])
        serp_titles = serp_result.get("serp_titles", [])
        serp_snippets = serp_result.get("serp_snippets", [])

        if not sources:
            return jsonify({
                "error": "Nie udało się pobrać źródeł z SerpAPI",
                "hint": "Sprawdź czy SERPAPI_KEY jest ustawiony i ważny",
                "main_keyword": main_keyword,
                "paa": paa_questions,
                "related_searches": related_searches
            }), 400

    print(f"[S1] 🔍 Analiza n-gramów dla: {main_keyword}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣ NLP Statystyczne (N-gramy)
    # v52.0: LEMMA-BASED N-GRAMS + HIGH-SIGNAL SOURCES
    #
    # Problem który rozwiązujemy:
    # A) FLEKSJA: "wpływem alkoholu", "wpływu alkoholu", "wpływ alkoholu" to ta
    #    sama fraza - Surfer liczy je razem, Brajn liczył jako 3 osobne n-gramy.
    #    FIX: indeksujemy po LEMATACH (canonical form), zachowujemy najczęstszą
    #    formę powierzchniową do wyświetlania.
    #
    # B) BRAKUJĄCE FRAZY: "warunkowe umorzenie" pojawia się w related_searches /
    #    PAA / snippetach ale rzadko w treści stron (bo to krótkie strony).
    #    FIX: PAA + related_searches + snippety = "high-signal source" - niższy
    #    próg freq dla tych fraz (wystarczy 1x, nie 2x).
    # ═══════════════════════════════════════════════════════════════════════════
    ngram_presence = defaultdict(set)
    ngram_freqs = Counter()
    ngram_per_source = defaultdict(lambda: Counter())
    lemma_surface_freq = defaultdict(Counter)  # lemma_key → {surface_form: count}
    all_text_content = []

    def _lemmatize_tokens(text_content, limit=50000):
        """Zwraca dwie listy: tokeny raw i tokeny-lematy (wyrównane, tylko alfa)."""
        doc = nlp(text_content[:limit])
        raw_toks, lem_toks = [], []
        for t in doc:
            if t.is_alpha:
                raw_toks.append(t.text.lower())
                lem_toks.append(t.lemma_.lower())
        return raw_toks, lem_toks

    def _build_ngrams_for_source(raw_toks, lem_toks, src_label, src_idx):
        """Buduje n-gramy używając LEMATÓW jako klucza, surface form do wyświetlania."""
        for n in range(2, 5):
            for i in range(len(lem_toks) - n + 1):
                lemma_key = " ".join(lem_toks[i:i + n])
                surface_form = " ".join(raw_toks[i:i + n])
                ngram_freqs[lemma_key] += 1
                ngram_presence[lemma_key].add(src_label)
                ngram_per_source[lemma_key][src_idx] += 1
                lemma_surface_freq[lemma_key][surface_form] += 1

    # ── Główne źródła: scraped pages ──────────────────────────────────────────
    for src_idx, src in enumerate(sources):
        content = (src.get("content", "") or "").lower()
        if not content.strip():
            continue
        all_text_content.append(src.get("content", ""))
        src_h2 = src.get("h2_structure", [])
        if src_h2:
            h2_patterns.extend(src_h2)
        raw_toks, lem_toks = _lemmatize_tokens(content)
        _build_ngrams_for_source(raw_toks, lem_toks, src.get("url", f"src_{src_idx}"), src_idx)

    # ── v52.0: High-signal sources: PAA + related searches + SERP snippets ────
    # Google sam selekcjonuje te frazy - zawierają ważne słowa kluczowe których
    # brak w krótkich stronach SERP (np. "warunkowe umorzenie", "dożywotni zakaz").
    HIGH_SIGNAL_SRC_IDX = len(sources)
    HIGH_SIGNAL_LABEL = "__google_signals__"
    high_signal_texts = []

    for paa_item in paa_questions:
        q = paa_item.get("question", "") if isinstance(paa_item, dict) else str(paa_item)
        if q:
            high_signal_texts.append(q)
    for rs in related_searches:
        q = rs if isinstance(rs, str) else (rs.get("query", "") or rs.get("text", ""))
        if q:
            high_signal_texts.append(q)
    for title in serp_titles:
        if title:
            high_signal_texts.append(title)
    for snippet in serp_snippets:
        if snippet:
            high_signal_texts.append(snippet)

    if high_signal_texts:
        combined_signal = " . ".join(high_signal_texts)
        raw_hs, lem_hs = _lemmatize_tokens(combined_signal, limit=20000)
        _build_ngrams_for_source(raw_hs, lem_hs, HIGH_SIGNAL_LABEL, HIGH_SIGNAL_SRC_IDX)
        print(f"[S1] 🎯 High-signal: {len(high_signal_texts)} tekstów (PAA+related+snippets) → dodane do n-gramów")

    # ── Resolve best surface form per lemma-key ────────────────────────────────
    lemma_to_surface = {}
    for lemma_key, surface_counts in lemma_surface_freq.items():
        lemma_to_surface[lemma_key] = surface_counts.most_common(1)[0][0]

    max_freq = max(ngram_freqs.values()) if ngram_freqs else 1
    num_sources = len(sources)
    results = []

    for ngram, freq in ngram_freqs.items():
        # v52.0: Oddzielny próg dla high-signal vs stron
        page_presence = {s for s in ngram_presence[ngram] if s != HIGH_SIGNAL_LABEL}
        page_freq = sum(
            cnt for idx, cnt in ngram_per_source[ngram].items()
            if idx != HIGH_SIGNAL_SRC_IDX
        )
        is_high_signal_only = (HIGH_SIGNAL_LABEL in ngram_presence[ngram]
                               and not page_presence)
        # Stary filtr: min 2x w stronach; nowy: high-signal przechodzi przy freq>=1
        if page_freq < 2 and not is_high_signal_only:
            continue
        # v52.0: Wyświetlamy najczęstszą formę powierzchniową, nie lemat
        display_ngram = lemma_to_surface.get(ngram, ngram)

        page_presence_set = {s for s in ngram_presence[ngram] if s != HIGH_SIGNAL_LABEL}
        freq_norm = page_freq / max_freq if max_freq else 0
        site_score = len(page_presence_set) / num_sources if num_sources else 0
        weight = round(freq_norm * 0.5 + site_score * 0.5, 4)

        # Boost: fraza zawiera główne słowo kluczowe
        if main_keyword and main_keyword.lower() in display_ngram:
            weight += 0.1
        # Boost: fraza pochodzi z high-signal source (PAA/related/snippet)
        if HIGH_SIGNAL_LABEL in ngram_presence[ngram]:
            weight += 0.08

        # v51/v52: Per-source frequency stats (Surfer-style ranges) — tylko prawdziwe strony
        per_src = ngram_per_source.get(ngram, {})
        all_counts = [per_src.get(i, 0) for i in range(num_sources)]
        non_zero = sorted([c for c in all_counts if c > 0])

        if non_zero:
            freq_min = non_zero[0]
            freq_max = non_zero[-1]
            mid = len(non_zero) // 2
            freq_median = non_zero[mid] if len(non_zero) % 2 == 1 else (non_zero[mid-1] + non_zero[mid]) // 2
        else:
            freq_min = freq_median = freq_max = 0

        results.append({
            "ngram": display_ngram,          # najczęstsza forma powierzchniowa
            "ngram_lemma": ngram,            # lemat (do dedup w keyword_counter)
            "freq": page_freq,               # tylko z prawdziwych stron
            "freq_total": freq,              # łącznie z high-signal
            "is_high_signal": is_high_signal_only,
            "weight": min(1.0, weight),
            "site_distribution": f"{len(page_presence_set)}/{num_sources}",
            "freq_per_source": all_counts,
            "freq_min": freq_min,
            "freq_median": freq_median,
            "freq_max": freq_max
        })

    results = sorted(results, key=lambda x: x["weight"], reverse=True)[:top_n]

    # 2️⃣ Semantyka (Gemini Flash)
    full_text_sample = " ".join(all_text_content)[:15000]
    semantic_keyphrases = extract_semantic_tags_gemini(full_text_sample)

    # ⭐ Unikalne H2 z konkurencji (bez duplikatów)
    unique_h2_patterns = list(dict.fromkeys(h2_patterns))[:30]

    # ⭐ Przygotuj serp_analysis
    serp_analysis_data = {
        "paa_questions": paa_questions,
        "featured_snippet": featured_snippet,
        "ai_overview": ai_overview,  # v27.0: Google SGE
        "related_searches": related_searches,
        "competitor_titles": serp_titles[:10],
        "competitor_snippets": serp_snippets[:10],
        "competitor_h2_patterns": unique_h2_patterns,
        # v27.0: Dodaj competitors z word_count dla recommended_length
        "competitors": [
            {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "word_count": src.get("word_count", 0),
                "h2_count": len(src.get("h2_structure", []))
            }
            for src in sources
        ]
    }

    # 3️⃣ Content Hints - WYŁĄCZONE v28.0 (duplikuje dane z serp_analysis)
    # content_hints = generate_content_hints(serp_analysis_data, main_keyword)

    # 4️⃣ 🆕 Entity SEO Analysis (v28.0)
    entity_seo_data = None
    if ENTITY_SEO_ENABLED and sources:
        try:
            print(f"[S1] 🧠 Running Entity SEO analysis...")
            entity_seo_data = perform_entity_seo_analysis(
                nlp=nlp,
                sources=sources,
                main_keyword=main_keyword,
                h2_patterns=unique_h2_patterns
            )
            print(f"[S1] ✅ Entity SEO: {entity_seo_data.get('entity_seo_summary', {}).get('total_entities', 0)} entities found")
        except Exception as e:
            print(f"[S1] ⚠️ Entity SEO error (non-critical): {e}")
            entity_seo_data = {"error": str(e), "status": "FAILED"}

    # 5️⃣ 🆕 Causal Triplet Extraction (v45.0)
    causal_data = None
    if CAUSAL_EXTRACTOR_ENABLED and sources:
        try:
            print(f"[S1] 🔗 Running Causal Triplet Extraction...")
            causal_triplets = extract_causal_triplets(
                texts=[s.get("content", "") for s in sources],
                main_keyword=main_keyword
            )
            causal_data = {
                "count": len(causal_triplets),
                "chains": [t.to_dict() for t in causal_triplets if t.is_chain],
                "singles": [t.to_dict() for t in causal_triplets if not t.is_chain],
                "agent_instruction": format_causal_for_agent(causal_triplets, main_keyword)
            }
            print(f"[S1] ✅ Causal Triplets: {len(causal_triplets)} found "
                  f"({sum(1 for t in causal_triplets if t.is_chain)} chains)")
        except Exception as e:
            print(f"[S1] ⚠️ Causal extraction error (non-critical): {e}")
            causal_data = {"error": str(e), "status": "FAILED"}

    # 6️⃣ 🆕 Content Gap Analysis (v45.0)
    content_gaps_data = None
    if GAP_ANALYZER_ENABLED and sources:
        try:
            print(f"[S1] 📊 Running Gap Analysis...")
            content_gaps_data = analyze_content_gaps(
                competitor_texts=[s.get("content", "") for s in sources],
                competitor_h2s=unique_h2_patterns,
                paa_questions=paa_questions,
                related_searches=related_searches,
                main_keyword=main_keyword
            )
            print(f"[S1] ✅ Content Gaps: {content_gaps_data.get('total_gaps', 0)} gaps found")
        except Exception as e:
            print(f"[S1] ⚠️ Gap Analysis error (non-critical): {e}")
            content_gaps_data = {"error": str(e), "status": "FAILED"}

    # ⭐ PEŁNA ODPOWIEDŹ z wszystkimi danymi SERP
    response_payload = {
        "main_keyword": main_keyword,
        "ngrams": results,
        "semantic_keyphrases": semantic_keyphrases,

        # ✅ NOWE (MINIMALNA ZMIANA): zwracamy próbkę pełnych treści konkurencji,
        # aby Master API mogło liczyć semantic coverage na realnym korpusie.
        # Zachowujemy kompatybilność wsteczną przez alias "serp_content".
        "full_text_sample": full_text_sample,
        "serp_content": full_text_sample,

        # ⭐ Pełna analiza SERP (surowe dane)
        "serp_analysis": serp_analysis_data,

        # ⭐ Content Hints - WYŁĄCZONE v28.0 (BRAJEN używa serp_analysis bezpośrednio)
        # "content_hints": content_hints,

        # 🆕 Entity SEO (v28.0)
        "entity_seo": entity_seo_data,

        # 🆕 Causal Triplets (v45.0)
        "causal_triplets": causal_data,

        # 🆕 Content Gaps (v45.0)
        "content_gaps": content_gaps_data,

        "summary": {
            "total_sources": len(sources),
            "sources_auto_fetched": not bool(data.get("sources", [])),
            "paa_count": len(paa_questions),
            "has_featured_snippet": featured_snippet is not None,
            "has_ai_overview": ai_overview is not None,
            "related_searches_count": len(related_searches),
            "h2_patterns_found": len(unique_h2_patterns),
            "entity_seo_enabled": ENTITY_SEO_ENABLED,
            "entities_found": entity_seo_data.get("entity_seo_summary", {}).get("total_entities", 0) if entity_seo_data else 0,
            "causal_triplets_found": causal_data.get("count", 0) if causal_data else 0,
            "content_gaps_found": content_gaps_data.get("total_gaps", 0) if content_gaps_data else 0,
            "engine": "v28.0",
            "lsi_candidates": len(semantic_keyphrases),
        }
    }

    # 3️⃣ Firestore Save (optional)
    if project_id:
        try:
            db = firestore.client()
            doc_ref = db.collection("seo_projects").document(project_id)
            if doc_ref.get().exists:
                avg_len = (
                    sum(len(t.split()) for t in all_text_content) // len(all_text_content)
                    if all_text_content else 0
                )
                doc_ref.update({
                    "s1_data": response_payload,
                    "lsi_enrichment": {"enabled": True, "count": len(semantic_keyphrases)},
                    "avg_competitor_length": avg_len,
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
                response_payload["saved_to_firestore"] = True
                print(f"[S1] ✅ Wyniki n-gramów zapisane do Firestore → {project_id}")
            else:
                response_payload["saved_to_firestore"] = False
                print(f"[S1] ⚠️ Nie znaleziono projektu {project_id}")
        except Exception as e:
            print(f"[S1] ❌ Firestore error: {e}")
            response_payload["firestore_error"] = str(e)

    return jsonify(response_payload)

# ======================================================
# 🧩 Pozostałe Endpointy (Proxy)
# ======================================================
@app.route("/api/synthesize_topics", methods=["POST"])
def perform_synthesize_topics():
    data = request.get_json(force=True)
    ngrams = data.get("ngrams", [])

    # ✅ NOWE (MINIMALNA ZMIANA): obsługa listy dictów {ngram: "..."} dla kompatybilności.
    if isinstance(ngrams, list) and ngrams and isinstance(ngrams[0], dict):
        ngrams = [x.get("ngram", "") for x in ngrams if isinstance(x, dict) and x.get("ngram")]

    return jsonify(synthesize_topics(ngrams, data.get("headings", [])))

@app.route("/api/generate_compliance_report", methods=["POST"])
def perform_generate_compliance_report():
    data = request.get_json(force=True)
    return jsonify(generate_compliance_report(data.get("text", ""), data.get("keyword_state", {})))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "v28.0",
        "limits": {
            "max_content_per_page": MAX_CONTENT_SIZE,
            "max_total_content": MAX_TOTAL_CONTENT,
            "scrape_timeout": SCRAPE_TIMEOUT,
            "skip_domains": SKIP_DOMAINS
        },
        "features": {
            "gemini_enabled": bool(GEMINI_API_KEY),
            "serpapi_enabled": bool(SERPAPI_KEY),
            "paa_extraction": True,
            "featured_snippet_extraction": True,
            "ai_overview_extraction": True,
            "related_searches_extraction": True,
            "competitor_h2_analysis": True,
            "competitor_word_count": True,
            "full_content_scraping": True,
            "oom_protection": True,
            "keyword_alias_support": True,
            # v28.0: Entity SEO
            "entity_seo_enabled": ENTITY_SEO_ENABLED,
            "entity_extraction": ENTITY_SEO_ENABLED,
            "topical_coverage": ENTITY_SEO_ENABLED,
            "entity_relationships": ENTITY_SEO_ENABLED,
            # v28.0: content_hints WYŁĄCZONE (BRAJEN używa serp_analysis)
            "content_hints_generation": False,
            # v45.0: Causal Triplets + Gap Analysis
            "causal_triplets_enabled": CAUSAL_EXTRACTOR_ENABLED,
            "gap_analysis_enabled": GAP_ANALYZER_ENABLED,
        }
    })

# ======================================================
# 🧩 Uruchomienie lokalne
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
