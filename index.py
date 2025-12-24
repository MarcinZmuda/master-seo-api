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
except ImportError:
    from synthesize_topics import synthesize_topics
    from generate_compliance_report import generate_compliance_report

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
                "related_searches": related_searches,
                "serp_titles": serp_titles,
                "serp_snippets": serp_snippets
            }

        print(f"[S1] ✅ Found {len(organic_results)} SERP results")

        # ⭐ 6. Scrapuj PEŁNĄ treść każdej strony + strukturę H2
        sources = []
        total_content_size = 0  # ⭐ v22.3: Track total size
        
        for result in organic_results[:num_results]:
            url = result.get("link", "")
            title = result.get("title", "")
            if not url:
                continue
            
            # ⭐ v22.3: Skip duże dokumenty (BIP, PDF, etc.)
            if should_skip_url(url):
                print(f"[S1] ⏭️ Skipping large doc pattern: {url[:50]}...")
                continue
            
            # ⭐ v22.3: Stop jeśli przekroczono total limit
            if total_content_size >= MAX_TOTAL_CONTENT:
                print(f"[S1] ⚠️ Total content limit reached ({MAX_TOTAL_CONTENT} chars), stopping scrape")
                break

            try:
                print(f"[S1] 📄 Scraping: {url[:60]}...")
                page_response = requests.get(
                    url,
                    timeout=SCRAPE_TIMEOUT,  # ⭐ v22.3: Reduced timeout
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                )

                if page_response.status_code == 200:
                    content = page_response.text
                    
                    # ⭐ v22.3: Limit content size PRZED przetwarzaniem
                    if len(content) > MAX_CONTENT_SIZE * 2:  # Raw HTML jest ~2x większy
                        print(f"[S1] ⚠️ Content too large ({len(content)} chars), truncating: {url[:40]}")
                        content = content[:MAX_CONTENT_SIZE * 2]

                    # ⭐ Wyciągnij H2 przed usunięciem tagów
                    h2_tags = re.findall(r'<h2[^>]*>(.*?)</h2>', content, re.IGNORECASE | re.DOTALL)
                    h2_clean = [re.sub(r'<[^>]+>', '', h).strip() for h in h2_tags]
                    h2_clean = [h for h in h2_clean if h and len(h) < 200]  # ⭐ v22.3: Skip too long H2

                    # Usuń script, style, nav, footer, header
                    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    # Usuń wszystkie tagi HTML
                    content = re.sub(r'<[^>]+>', ' ', content)
                    # Usuń wielokrotne spacje
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    # ⭐ v22.3: Final content limit
                    content = content[:MAX_CONTENT_SIZE]

                    if len(content) > 500:
                        sources.append({
                            "url": url,
                            "title": title,
                            "content": content,
                            "h2_structure": h2_clean[:15]
                        })
                        total_content_size += len(content)  # ⭐ v22.3: Track size
                        print(f"[S1] ✅ Scraped {len(content)} chars, {len(h2_clean)} H2 from {url[:40]}")
                    else:
                        print(f"[S1] ⚠️ Too short content from {url[:40]}")

            except requests.exceptions.Timeout:
                print(f"[S1] ⏱️ Timeout for {url[:40]} (>{SCRAPE_TIMEOUT}s)")
                continue
            except Exception as e:
                print(f"[S1] ⚠️ Scrape error for {url[:40]}: {e}")
                continue

        print(f"[S1] ✅ Successfully scraped {len(sources)} sources ({total_content_size} total chars)")

        return {
            "sources": sources,
            "paa": paa_questions,
            "featured_snippet": featured_snippet,
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
    main_keyword = data.get("main_keyword", "")
    sources = data.get("sources", [])
    top_n = int(data.get("top_n", 30))
    project_id = data.get("project_id")

    # ⭐ Zmienne na dodatkowe dane SERP
    paa_questions = []
    featured_snippet = None
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

    # 1️⃣ NLP Statystyczne (N-gramy)
    ngram_presence = defaultdict(set)
    ngram_freqs = Counter()
    all_text_content = []

    for src in sources:
        content = (src.get("content", "") or "").lower()
        if not content.strip():
            continue

        all_text_content.append(src.get("content", ""))

        # ⭐ Zbierz struktury H2 z konkurencji
        src_h2 = src.get("h2_structure", [])
        if src_h2:
            h2_patterns.extend(src_h2)

        # ⭐ v22.3: Limit content for NLP processing
        doc = nlp(content[:50000])  # Reduced from 100000
        tokens = [t.text.lower() for t in doc if t.is_alpha]

        for n in range(2, 5):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                ngram_freqs[ngram] += 1
                ngram_presence[ngram].add(src.get("url", "unknown"))

    max_freq = max(ngram_freqs.values()) if ngram_freqs else 1
    results = []

    for ngram, freq in ngram_freqs.items():
        if freq < 2:
            continue
        freq_norm = freq / max_freq
        site_score = len(ngram_presence[ngram]) / len(sources) if sources else 0
        weight = round(freq_norm * 0.5 + site_score * 0.5, 4)
        if main_keyword and main_keyword.lower() in ngram:
            weight += 0.1
        results.append({
            "ngram": ngram,
            "freq": freq,
            "weight": min(1.0, weight),
            "site_distribution": f"{len(ngram_presence[ngram])}/{len(sources)}"
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
        "related_searches": related_searches,
        "competitor_titles": serp_titles[:10],
        "competitor_snippets": serp_snippets[:10],
        "competitor_h2_patterns": unique_h2_patterns,
    }

    # 3️⃣ Content Hints - subtelne wskazówki dla GPT
    content_hints = generate_content_hints(serp_analysis_data, main_keyword)

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

        # ⭐ Content Hints - inspiracje dla GPT
        "content_hints": content_hints,

        "summary": {
            "total_sources": len(sources),
            "sources_auto_fetched": not bool(data.get("sources", [])),
            "paa_count": len(paa_questions),
            "has_featured_snippet": featured_snippet is not None,
            "related_searches_count": len(related_searches),
            "h2_patterns_found": len(unique_h2_patterns),
            "content_hints_generated": bool(content_hints),
            "engine": "v22.3-oom-fix",  # ⭐ v22.3
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
        "engine": "v22.3-oom-fix",  # ⭐ v22.3
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
            "related_searches_extraction": True,
            "competitor_h2_analysis": True,
            "full_content_scraping": True,
            "content_hints_generation": True,
            "oom_protection": True  # ⭐ v22.3
        }
    })

# ======================================================
# 🧩 Uruchomienie lokalne
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
