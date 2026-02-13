"""
===============================================================================
🧹 CONTENT EXTRACTOR v1.0 — Czysta ekstrakcja treści ze stron konkurencji
===============================================================================
Zamiennik regex-based scrapera w index.py.
Używa trafilatura + BeautifulSoup zamiast regex do:
1. Ekstrakcji czystego tekstu artykułu (bez JS, CSS, nav, footer)
2. Ekstrakcji struktury nagłówków H1-H4
3. Filtrowania URL-i non-article (YouTube, social media, etc.)
4. Walidacji jakości wyekstrahowanego tekstu

Drop-in replacement: podmień scraping loop w fetch_serp_sources()

Autor: BRAJEN Team
Data: 2025
===============================================================================
"""

import re
import requests
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup, Comment

# ================================================================
# 📦 TRAFILATURA — import z graceful fallback
# ================================================================
try:
    import trafilatura
    from trafilatura.settings import use_config
    
    # Konfiguracja trafilatura dla lepszych wyników
    _TRAF_CONFIG = use_config()
    _TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "200")
    _TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
    
    TRAFILATURA_AVAILABLE = True
    print("[EXTRACTOR] ✅ trafilatura loaded — clean content extraction enabled")
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("[EXTRACTOR] ⚠️ trafilatura not installed — using BeautifulSoup fallback")


# ================================================================
# 🚫 URL FILTERING — skip non-article URLs
# ================================================================

# Domeny/patterny które nigdy nie zawierają artykułów
_SKIP_URL_PATTERNS = [
    # Video platforms
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
    "tiktok.com", "twitch.tv",
    # Social media
    "facebook.com", "twitter.com", "x.com/", "instagram.com",
    "linkedin.com/posts", "reddit.com",
    # Documents & files
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Government/BIP large docs
    "bip.", "gov.pl/dana/", "/uploads/files/",
    # Forums & Q&A (often low quality scrapes)
    "forum.", "quora.com",
    # E-commerce (product pages, not articles)
    "allegro.pl/oferta/", "olx.pl/",
    # Image hosting
    "imgur.com", "flickr.com",
    # Maps
    "maps.google", "google.com/maps",
]

# File extensions to skip
_SKIP_EXTENSIONS = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", 
                    ".zip", ".rar", ".gz", ".mp4", ".mp3", ".wav", ".avi"]


def should_skip_url(url: str) -> bool:
    """
    Sprawdza czy URL powinien być pominięty.
    Rozszerzona wersja — łapie YouTube, social media, pliki binarne.
    """
    url_lower = url.lower()
    
    # Check skip patterns
    for pattern in _SKIP_URL_PATTERNS:
        if pattern in url_lower:
            return True
    
    # Check file extensions
    # Wyciągnij ścieżkę bez query string
    path = url_lower.split("?")[0].split("#")[0]
    for ext in _SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    
    return False


# ================================================================
# 🧹 GARBAGE DETECTION — czy tekst to śmieci CSS/JS?
# ================================================================

# Wzorce wskazujące na CSS/JS garbage w tekście
_GARBAGE_PATTERNS = [
    r'\.ast[-_]',           # Astra WordPress theme CSS
    r'kevlar_\w+',          # YouTube JS flags
    r'ytplayer|ytcfg',      # YouTube player
    r'webpack|__webpack',   # Webpack bundle
    r'var\s*\(\s*--',       # CSS variables
    r'\{[^}]*:\s*\w+;',    # CSS rules
    r'enable_\w+.*:true',   # JSON config flags
    r'@media\s*\(',         # CSS media queries
    r'font-family:',        # CSS font declarations
    r'\.wp-block-',         # WordPress block CSS
    r'border-radius:',      # CSS property
    r'padding:|margin:',    # CSS properties
    r'display:\s*flex',     # CSS flexbox
    r'background-color:',   # CSS property
]

_GARBAGE_RE = re.compile('|'.join(_GARBAGE_PATTERNS), re.IGNORECASE)


def _calculate_garbage_ratio(text: str) -> float:
    """
    Oblicza jaki procent tekstu to śmieci CSS/JS.
    Returns: 0.0 (czysty) — 1.0 (totalny garbage)
    """
    if not text or len(text) < 100:
        return 1.0
    
    # Podziel na fragmenty po ~200 znaków i sprawdź każdy
    chunk_size = 200
    chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 5000), chunk_size)]
    
    if not chunks:
        return 1.0
    
    garbage_chunks = 0
    for chunk in chunks:
        # Garbage jeśli: dużo specjalnych znaków LUB matchuje garbage patterns
        special_ratio = sum(1 for c in chunk if c in '{}();:[]<>=@#._-') / max(len(chunk), 1)
        has_garbage_pattern = bool(_GARBAGE_RE.search(chunk))
        
        if special_ratio > 0.15 or has_garbage_pattern:
            garbage_chunks += 1
    
    return garbage_chunks / len(chunks)


def _is_content_clean(text: str, min_words: int = 50) -> bool:
    """
    Sprawdza czy wyekstrahowany tekst to prawdziwa treść, nie śmieci.
    """
    if not text:
        return False
    
    word_count = len(text.split())
    if word_count < min_words:
        return False
    
    garbage_ratio = _calculate_garbage_ratio(text)
    if garbage_ratio > 0.3:  # >30% garbage = odrzuć
        return False
    
    return True


# ================================================================
# 📄 MAIN EXTRACTION — trafilatura + BeautifulSoup
# ================================================================

def extract_content(html: str, url: str = "") -> Optional[str]:
    """
    Wyciąga czysty tekst artykułu z raw HTML.
    
    Pipeline:
    1. trafilatura (najlepsze wyniki) 
    2. BeautifulSoup fallback (jeśli trafilatura zawiedzie)
    3. Walidacja jakości
    
    Returns: czysty tekst lub None jeśli ekstrakcja nie powiodła się
    """
    if not html or len(html) < 200:
        return None
    
    content = None
    
    # ---- METODA 1: trafilatura (najlepsza) ----
    if TRAFILATURA_AVAILABLE:
        try:
            content = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True,
                config=_TRAF_CONFIG,
                url=url or None,
            )
            if content and _is_content_clean(content):
                print(f"[EXTRACTOR] 🧹 trafilatura: {len(content)} chars → clean")
                return content
            else:
                print(f"[EXTRACTOR] ⚠️ trafilatura output rejected (garbage or too short)")
                content = None
        except Exception as e:
            print(f"[EXTRACTOR] ⚠️ trafilatura error: {e}")
            content = None
    
    # ---- METODA 2: BeautifulSoup (fallback) ----
    try:
        content = _extract_with_beautifulsoup(html)
        if content and _is_content_clean(content):
            print(f"[EXTRACTOR] 🍜 BeautifulSoup: {len(content)} chars → clean")
            return content
        else:
            print(f"[EXTRACTOR] ⚠️ BeautifulSoup output rejected")
            content = None
    except Exception as e:
        print(f"[EXTRACTOR] ⚠️ BeautifulSoup error: {e}")
    
    return content


def _extract_with_beautifulsoup(html: str) -> Optional[str]:
    """
    Ekstrakcja treści za pomocą BeautifulSoup.
    Próbuje znaleźć główny kontener artykułu.
    """
    soup = BeautifulSoup(html, "lxml")
    
    # 1. Usuń śmieciowe elementy
    for tag_name in ["script", "style", "noscript", "svg", "iframe",
                     "nav", "footer", "header", "aside", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    
    # Usuń komentarze HTML
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    
    # Usuń elementy z typowymi klasami śmieciowymi
    _garbage_classes = [
        "cookie", "sidebar", "widget", "advertisement", "ad-", "ads-",
        "social-share", "share-buttons", "related-posts", "breadcrumb",
        "menu", "navigation", "comment", "popup", "modal", "newsletter"
    ]
    for cls in _garbage_classes:
        for tag in soup.find_all(class_=re.compile(cls, re.IGNORECASE)):
            tag.decompose()
    
    for id_name in ["cookie", "sidebar", "comments", "footer", "menu", "popup"]:
        for tag in soup.find_all(id=re.compile(id_name, re.IGNORECASE)):
            tag.decompose()
    
    # 2. Szukaj głównego kontenera artykułu (od najbardziej specyficznego)
    article_selectors = [
        "article",
        "[role='main']",
        ".entry-content",       # WordPress
        ".post-content",        # Common blog
        ".article-content",     # News sites
        ".article-body",
        ".content-area",
        "#content",
        "main",
        ".post",
    ]
    
    main_content = None
    for selector in article_selectors:
        found = soup.select_one(selector)
        if found:
            text = found.get_text(separator="\n", strip=True)
            if len(text) > 300:  # Minimum sensownej treści
                main_content = text
                break
    
    # 3. Fallback: cały body
    if not main_content:
        body = soup.find("body")
        if body:
            main_content = body.get_text(separator="\n", strip=True)
    
    if not main_content:
        return None
    
    # 4. Cleanup tekstu
    # Usuń wielokrotne puste linie
    main_content = re.sub(r'\n{3,}', '\n\n', main_content)
    # Usuń linie z samymi spacjami
    main_content = re.sub(r'\n\s+\n', '\n\n', main_content)
    # Usuń potencjalne resztki CSS/JS (krótkie linijki z dużo specjalnymi znakami)
    lines = main_content.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            clean_lines.append('')
            continue
        # Skip linii które wyglądają jak CSS/JS
        special_count = sum(1 for c in line if c in '{}();:[]<>=@#')
        if len(line) > 0 and special_count / len(line) > 0.2:
            continue
        # Skip bardzo krótkich linii z dużo kropkami/myślnikami (menu items)
        if len(line) < 30 and line.count('.') + line.count('|') + line.count('›') > 2:
            continue
        clean_lines.append(line)
    
    main_content = '\n'.join(clean_lines)
    # Normalizuj spacje w obrębie linii
    main_content = re.sub(r'[ \t]+', ' ', main_content)
    # Finalne trimowanie
    main_content = re.sub(r'\n{3,}', '\n\n', main_content).strip()
    
    return main_content


# ================================================================
# 📋 HEADING EXTRACTION — H1-H4 structure
# ================================================================

def extract_headings(html: str) -> Dict[str, List[str]]:
    """
    Wyciąga nagłówki H1-H4 z HTML za pomocą BeautifulSoup.
    Znacznie lepsze niż regex — radzi sobie z:
    - nested tags wewnątrz nagłówków
    - atrybutami HTML
    - encoded entities
    
    Returns: {"h1": [...], "h2": [...], "h3": [...], "h4": [...]}
    """
    result = {"h1": [], "h2": [], "h3": [], "h4": []}
    
    if not html:
        return result
    
    try:
        soup = BeautifulSoup(html, "lxml")
        
        for level in ["h1", "h2", "h3", "h4"]:
            for tag in soup.find_all(level):
                text = tag.get_text(strip=True)
                # Filtruj śmieci
                if not text:
                    continue
                if len(text) > 200:  # Za długi — prawdopodobnie garbage
                    continue
                if len(text) < 2:   # Za krótki
                    continue
                # Skip jeśli wygląda jak CSS/JS
                if re.search(r'[{};]|webkit|moz-|flex-|align-items|\.ast-|\.wp-', text, re.IGNORECASE):
                    continue
                # Skip jeśli to głównie cyfry/specjalne znaki
                alpha_count = sum(1 for c in text if c.isalpha())
                if len(text) > 0 and alpha_count / len(text) < 0.4:
                    continue
                
                result[level].append(text)
    
    except Exception as e:
        print(f"[EXTRACTOR] ⚠️ Heading extraction error: {e}")
    
    return result


# ================================================================
# 🔗 FULL PIPELINE — scrape + extract for one URL
# ================================================================

# Konfiguracja
DEFAULT_TIMEOUT = 8
DEFAULT_MAX_HTML_SIZE = 80000   # 80KB raw HTML (więcej niż 30K, bo trafilatura wytnie)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def scrape_and_extract(
    url: str,
    title: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    max_html_size: int = DEFAULT_MAX_HTML_SIZE,
    max_content_size: int = 30000,
) -> Optional[Dict]:
    """
    Pobiera URL i ekstrakcjonuje czystą treść + nagłówki.
    
    Drop-in replacement for the scraping loop in fetch_serp_sources().
    
    Returns: {
        "url": str,
        "title": str, 
        "content": str,         # Czysty tekst artykułu
        "h2_structure": [str],  # Lista H2
        "h1": [str],            # Lista H1 (opcjonalne)
        "h3": [str],            # Lista H3 (opcjonalne) 
        "word_count": int,      # Liczba słów
    } or None jeśli ekstrakcja się nie powiodła
    """
    # 1. Skip non-article URLs
    if should_skip_url(url):
        print(f"[EXTRACTOR] ⏭️ Skipping non-article URL: {url[:60]}")
        return None
    
    # 2. Fetch HTML
    try:
        print(f"[EXTRACTOR] 📄 Fetching: {url[:60]}...")
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            allow_redirects=True,
        )
        
        if response.status_code != 200:
            print(f"[EXTRACTOR] ❌ HTTP {response.status_code} for {url[:40]}")
            return None
        
        raw_html = response.text
        
    except requests.exceptions.Timeout:
        print(f"[EXTRACTOR] ⏱️ Timeout for {url[:40]} (>{timeout}s)")
        return None
    except Exception as e:
        print(f"[EXTRACTOR] ❌ Fetch error for {url[:40]}: {e}")
        return None
    
    # 3. Limit raw HTML size
    if len(raw_html) > max_html_size:
        print(f"[EXTRACTOR] ⚠️ HTML too large ({len(raw_html)} chars), truncating: {url[:40]}")
        raw_html = raw_html[:max_html_size]
    
    # 4. Extract headings (z pełnego HTML, przed jakimkolwiek cleanup)
    headings = extract_headings(raw_html)
    h2_clean = headings["h2"][:15]
    
    # 5. Extract clean content
    content = extract_content(raw_html, url=url)
    
    if not content:
        print(f"[EXTRACTOR] ❌ No clean content from {url[:40]}")
        return None
    
    # 6. Final content limit
    content = content[:max_content_size]
    word_count = len(content.split())
    
    if word_count < 50:
        print(f"[EXTRACTOR] ⚠️ Too short ({word_count} words) from {url[:40]}")
        return None
    
    print(f"[EXTRACTOR] ✅ {len(content)} chars ({word_count} words), "
          f"{len(h2_clean)} H2 from {url[:40]}")
    
    return {
        "url": url,
        "title": title,
        "content": content,
        "h2_structure": h2_clean,
        "h1": headings["h1"][:3],
        "h3": headings["h3"][:20],
        "word_count": word_count,
    }


# ================================================================
# 📋 BATCH EXTRACTION — for fetch_serp_sources() integration
# ================================================================

def extract_serp_sources(
    organic_results: List[Dict],
    num_results: int = 10,
    max_total_content: int = 200000,
    max_content_per_page: int = 30000,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict]:
    """
    Przetwarza listę organic results z SerpAPI i zwraca czyste źródła.
    
    Drop-in replacement for the scraping loop in fetch_serp_sources().
    
    Args:
        organic_results: Lista wyników z SerpAPI (.get("organic_results"))
        num_results: Max stron do scrapowania
        max_total_content: Max łączny rozmiar treści
        max_content_per_page: Max rozmiar treści per strona
        timeout: Timeout per request
    
    Returns: Lista źródeł [{url, title, content, h2_structure, word_count}, ...]
    """
    sources = []
    total_content_size = 0
    
    for result in organic_results[:num_results]:
        url = result.get("link", "")
        title = result.get("title", "")
        
        if not url:
            continue
        
        # Stop jeśli przekroczono total limit
        if total_content_size >= max_total_content:
            print(f"[EXTRACTOR] ⚠️ Total content limit reached "
                  f"({max_total_content} chars), stopping")
            break
        
        # Scrape + extract
        source = scrape_and_extract(
            url=url,
            title=title,
            timeout=timeout,
            max_content_size=max_content_per_page,
        )
        
        if source:
            sources.append(source)
            total_content_size += len(source["content"])
    
    print(f"[EXTRACTOR] ✅ Extracted {len(sources)} clean sources "
          f"({total_content_size} total chars)")
    
    return sources
