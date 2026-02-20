"""
BRAJEN SEO Web App v45.2.2
==========================
Standalone web app that orchestrates BRAJEN SEO API + Anthropic Claude for text generation.
Replaces unreliable GPT Custom Actions with deterministic code-driven workflow.

Deploy: Render (render.yaml included)
Auth: Simple login/password via environment variable
"""

import os
import json
import time
import uuid
import hashlib
import logging
import secrets
import threading
import queue
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, Response,
    session, redirect, url_for, stream_with_context, send_file
)
import requests as http_requests
import anthropic
from prompt_builder import (
    build_system_prompt, build_user_prompt,
    build_faq_system_prompt, build_faq_user_prompt,
    build_h2_plan_system_prompt, build_h2_plan_user_prompt
)

# Optional: OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import re as _re

# AI Middleware: inteligentne czyszczenie danych i smart retry
from ai_middleware import (
    process_s1_for_pipeline,
    smart_retry_batch,
    should_use_smart_retry,
    synthesize_article_memory,
    ai_synthesize_memory,
    check_sentence_length,
    sentence_length_retry,
    validate_batch_domain,
    fix_batch_domain_errors,
)
from keyword_dedup import deduplicate_keywords
from entity_salience import (
    check_entity_salience,
    generate_article_schema,
    schema_to_html,
    generate_topical_map,
    build_entity_salience_instructions,
    is_salience_available,
    analyze_entities_google_nlp,
    analyze_subject_position,
    analyze_style_consistency,
    analyze_ymyl_references,
)

# v50.7: Polish NLP validator (NKJP corpus norms)
try:
    from polish_nlp_validator import validate_polish_text, get_polish_nlp_summary
    POLISH_NLP_AVAILABLE = True
except ImportError:
    POLISH_NLP_AVAILABLE = False

# v50.7: LanguageTool integration (corpus-based grammar/collocation checker)
try:
    from languagetool_checker import check_text as lt_check_text, get_summary as lt_get_summary
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False

# ================================================================
# 🗑️ CSS/JS GARBAGE FILTER: czyści śmieci z S1 danych
# ================================================================
_CSS_GARBAGE_PATTERNS = _re.compile(
    r'(?:'
    # CSS properties & values
    r'webkit|moz-|ms-flex|align-items|display\s*:|flex-pack|'
    r'font-family|background-color|border-bottom|text-shadow|'
    r'position\s*:|padding\s*:|margin\s*:|transform\s*:|'
    r'transition|scrollbar|\.uk-|\.et_pb_|\.rp-|'
    r'min-width|max-width|overflow|z-index|opacity|'
    r'hover\{|active\{|:after|:before|calc\(|'
    r'woocommerce|gutters|inline-flex|box-pack|'
    r'data-[a-z]|aria-|role=|tabindex|'
    # BEM class notation (block__element, block--modifier)
    r'\w+__\w+|'
    r'\w+--\w+|'
    # CSS selectors & combinators
    r'focus-visible|#close|,#|\.css|'
    # WordPress / CMS artifacts
    r'\bvar\s+wp\b|wp-block|wp-embed|'
    r'block\s*embed|content\s*block|text\s*block|'
    r'input\s*type|'
    # HTML/UI element names
    r'^(header|footer|sidebar|nav|mega)\s*-?\s*menu$|'
    r'^sub-?\s*menu$|'
    r'^mega\s+menu$|'
    # Generic CSS patterns
    r'^\w+\.\w+$|'
    r'[{};]'
    r')',
    _re.IGNORECASE
)

_CSS_NGRAM_EXACT = {
    "min width", "width min", "ms flex", "align items", "flex pack",
    "box pack", "table table", "decoration decoration", "inline flex",
    "webkit box", "webkit text", "moz box", "moz flex",
    "box align", "flex align", "flex direction", "flex wrap",
    "justify content", "text decoration", "font weight", "font size",
    "line height", "border radius", "box shadow", "text align",
    "text transform", "letter spacing", "word spacing", "white space",
    "min height", "max height", "list style", "vertical align",
    "before before", "data widgets", "widgets footer", "footer widget",
    "focus focus", "root root", "not not",
    # WordPress / CMS
    "var wp", "block embed", "content block", "text block", "input type",
    "wp block", "wp embed", "post type", "nav menu", "menu item",
    "header menu", "sub menu", "mega menu", "footer menu",
    "widget area", "sidebar widget", "page template",
    # v45.4.1: Extended: catches observed CSS garbage from dashboard
    "list list", "heading heading", "container expand", "expand container",
    "container item", "container container", "table responsive",
    "heading heading heading", "list list list", "list list list list",
    "container expand container", "form form", "button button",
    "image utf", "image image", "form input", "input input",
    "expand expand", "item item", "block block", "section section",
    "row row", "column column", "grid grid", "card card",
    "wrapper wrapper", "inner inner", "outer outer",
    "responsive table", "responsive responsive",
    # v49: CSS variable patterns from SERP scraping
    "ast global", "global color", "ast global color", "var ast",
    "var ast global", "var ast global color", "var global",
    "global ast", "color inherit", "inherit color",
    # v50.4: WordPress social sharing widgets / footer artifacts
    "block social", "social link", "social block", "link block",
    "style logos", "logos only", "only social", "social link block",
    "link block social", "logos only social", "style logos only",
    "only social link", "logos only social link",
    "style logos only social", "social link block social",
    "only social link block", "wp preset", "preset gradient",
    # v50.7 FIX 39: CSS @font-face declaration fragments
    "font family", "face font", "font style", "font weight",
    "weight font", "display swap", "swap src", "src url",
    "url blog", "blog wp", "content fonts", "unicode range",
    "face font family", "weight font display", "font display swap",
    "display swap src", "swap src url", "src url blog",
    "font face", "woff2 format", "woff format", "ttf format",
    "font awesome", "awesome regular", "awesome solid", "awesome brands",
    # v50.7: WordPress content/blog patterns
    "wp content", "content uploads", "content themes", "content plugins",
    "wp includes", "wp json", "wp admin",
}

_CSS_ENTITY_WORDS = {
    "inline", "button", "active", "hover", "flex", "grid", "block",
    "none", "inherit", "auto", "hidden", "visible", "relative",
    "absolute", "fixed", "static", "center", "wrap", "nowrap",
    "bold", "normal", "italic", "transparent", "solid", "dotted",
    "pointer", "default", "disabled", "checked", "focus",
    "where", "not", "root", "before", "after",
    # HTML/UI elements
    "menu", "submenu", "sidebar", "footer", "header", "widget",
    "navbar", "dropdown", "modal", "tooltip", "carousel",
    "accordion", "breadcrumb", "pagination", "thumbnail",
    # v49: CSS variable tokens & font names
    "ast", "var", "global", "color", "sich", "un", "uw",
    "xl", "ac", "arrow", "dim",
    "menlo", "monaco", "consolas", "courier", "arial", "helvetica",
    "verdana", "georgia", "roboto", "poppins", "raleway",
    # v50.4: Scraper artifacts: English words spaCy misclassifies as entities
    # These are CSS class names, color names, or HTML content words that
    # appear in competitor pages and get extracted as Polish entities.
    "vivid", "bluish", "muted", "faded", "bright", "subtle", "crisp",
    "reviews", "review", "rating", "ratings", "share", "shares",
    "click", "submit", "cancel", "close", "open", "toggle", "expand",
    "czyste", "clean", "dark", "light", "primary", "secondary",
    "success", "warning", "danger", "info", "muted",
    # v50.4: Social media / platform names (scraper picks up footer links)
    "facebook", "twitter", "instagram", "linkedin", "youtube",
    "pinterest", "tiktok", "snapchat", "whatsapp", "telegram",
    "bandcamp", "bluesky", "deviantart", "fivehundredpx", "mastodon",
    "reddit", "tumblr", "flickr", "vimeo", "soundcloud", "spotify",
    # v50.4: WordPress/CMS artifact words
    "preset", "logos", "embed", "widget", "template", "shortcode",
    "plugin", "theme", "customizer", "gutenberg", "elementor",
    # v50.5 FIX 23: Wikipedia sidebar language names
    # Scraper extracts language links from Wikipedia interlanguage sidebar.
    # spaCy misclassifies these as PERSON/LOC entities with high salience.
    "asturianu", "azərbaycanca", "afrikaans", "aragonés", "bân",
    "català", "čeština", "cymraeg", "dansk", "eesti", "esperanto",
    "euskara", "galego", "hrvatski", "ido", "interlingua",
    "íslenska", "italiano", "kurdî", "latina", "latviešu",
    "lietuvių", "magyar", "македонски", "bahasa", "melayu",
    "nordfriisk", "nynorsk", "occitan", "oʻzbekcha", "piemontèis",
    "português", "română", "shqip", "sicilianu", "slovenčina",
    "slovenščina", "srpskohrvatski", "suomi", "svenska", "tagalog",
    "türkçe", "українська", "tiếng", "việt", "volapük",
    "walon", "winaray", "ייִדיש",
    "башҡортса", "беларуская", "български", "қазақша", "кыргызча",
    "монгол", "русский", "српски", "татарча", "тоҷикӣ", "ўзбекча",
    "العربية", "فارسی", "עברית", "हिन्दी", "বাংলা",
    "ગુજરાતી", "ಕನ್ನಡ", "தமிழ்", "తెలుగు",
    "中文", "日本語", "한국어", "粵語",
    # Common multi-word Wikipedia language labels
    "fiji hindi", "basa jawa", "basa sunda", "kreyòl ayisyen",
    # v50.5 FIX 24: Wikipedia/website navigation artifacts
    # Buttons, links, and navigation elements scraped as entities
    "przejdź", "sprawdź", "edytuj", "historia", "dyskusja",
    "zaloguj", "utwórz", "szukaj", "wyszukaj", "pokaż",
    "ukryj", "rozwiń", "zwiń", "zamknij", "otwórz",
    "czytaj", "wyświetl", "pobierz", "udostępnij",
    "read", "view", "edit", "search", "login", "signup",
    "subscribe", "download", "upload", "skip", "next", "previous",
    "more", "less", "show", "hide", "back", "forward",
}

def _is_css_garbage(text):
    if not text or not isinstance(text, str):
        return True
    text = text.strip()
    if len(text) < 2:
        return True
    special = sum(1 for c in text if c in '{}:;()[]<>=#.@')
    if len(text) > 0 and special / len(text) > 0.15:
        return True
    if text.lower() in _CSS_NGRAM_EXACT:
        return True
    if text.lower() in _CSS_ENTITY_WORDS:
        return True
    # v47.2: CSS compound tokens: inherit;color, section{display, serif;font
    t_lower = text.lower()
    if _re.match(r'^[\w-]+[;{}\[\]:]+[\w-]+$', t_lower):
        _CSS_TOKENS = {
            "inherit", "color", "display", "flex", "block", "inline", "grid",
            "none", "auto", "hidden", "visible", "solid", "dotted", "dashed",
            "bold", "normal", "italic", "pointer", "cursor", "border",
            "margin", "padding", "font", "section", "strong", "help",
            "center", "wrap", "cover", "contain", "serif", "sans",
            "position", "relative", "absolute", "fixed", "opacity",
            "background", "transform", "overflow", "scroll", "width",
            "height", "text", "decoration", "underline", "uppercase",
            "hover", "focus", "active", "image", "repeat", "content",
            "table", "row", "column", "collapse", "weight", "size", "style",
        }
        parts = _re.split(r'[;{}\[\]:]+', t_lower)
        parts = [p.strip('-') for p in parts if p]
        if parts and any(p in _CSS_TOKENS for p in parts):
            return True
    # v47.2: Font names
    _FONT_NAMES = {
        "menlo", "monaco", "consolas", "courier", "arial", "helvetica",
        "verdana", "georgia", "tahoma", "trebuchet", "lucida", "roboto",
        "poppins", "raleway", "montserrat", "lato", "inter",
    }
    if t_lower in _FONT_NAMES:
        return True
    # v45.4.1: Detect repeated-word patterns ("list list list", "heading heading")
    words = text.lower().split()
    if len(words) >= 2 and len(set(words)) == 1:
        return True  # All words identical ("list list", "heading heading heading")
    if len(words) >= 3 and len(set(words)) <= 2:
        return True  # 3+ words but only 1-2 unique ("container expand container")
    # v45.4.1: Detect CSS class-like multi-word tokens
    # Only flag if ALL words are short ASCII-only AND match common CSS vocabulary
    _CSS_VOCAB = {
        'list', 'heading', 'container', 'expand', 'item', 'image', 'form',
        'table', 'responsive', 'button', 'section', 'row', 'column', 'grid',
        'card', 'wrapper', 'inner', 'outer', 'block', 'embed', 'content',
        'input', 'label', 'icon', 'link', 'nav', 'tab', 'panel', 'modal',
        'badge', 'alert', 'toast', 'spinner', 'loader', 'overlay', 'toggle',
        'dropdown', 'collapse', 'accordion', 'breadcrumb', 'pagination',
        'thumbnail', 'carousel', 'slider', 'progress', 'tooltip', 'popover',
        'utf', 'meta', 'viewport', 'charset', 'script', 'noscript',
        'dim', 'cover', 'inherit', 'font', 'serif', 'sans', 'display',
        'border', 'margin', 'padding', 'strong', 'color',
        # v49: CSS variable tokens
        'ast', 'var', 'global', 'min', 'max', 'wp',
        # v50.7 FIX 39: CSS font-face declaration fragments from @font-face rules
        'family', 'face', 'style', 'weight', 'swap', 'src', 'url',
        'unicode', 'range', 'fonts', 'woff', 'woff2', 'ttf', 'eot', 'svg',
        'format', 'local', 'fallback', 'optional', 'preload',
        # v50.7: Font Awesome / icon fonts scraped as entities
        'awesome', 'regular', 'solid', 'brands', 'duotone', 'sharp',
        'fa', 'fab', 'fas', 'far', 'fal', 'fad',
        # v50.7: Blog/CMS URL fragments
        'blog', 'post', 'page', 'category', 'tag', 'author', 'archive',
        'sidebar', 'footer', 'header', 'nav', 'menu',
    }
    if len(words) >= 2 and all(w in _CSS_VOCAB for w in words):
        return True
    # v50.4: Sentence fragments: real entities are max 5-6 words,
    # scraper sometimes extracts entire sentence fragments as "entities"
    if len(words) > 6:
        return True
    # v50.4: Pure ASCII single words that aren't Polish proper nouns
    # These are typically CSS class names, HTML element names, or English words
    # that spaCy misclassifies as entities in Polish competitor pages.
    if len(words) == 1 and text.isascii() and text[0].islower():
        return True  # Lowercase single ASCII word = never a Polish entity
    # v50.5 FIX 23: Multi-word Wikipedia sidebar artifacts
    # When scraper concatenates adjacent language links: "Asturianu Azərbaycanca"
    # Check if ALL words in the text are known Wikipedia language names
    if len(words) >= 2:
        _all_wiki_lang = all(w.lower() in _CSS_ENTITY_WORDS for w in words)
        if _all_wiki_lang:
            return True  # All words are blocked terms → garbage
    # v50.5 FIX 23: Detect non-Polish/non-English single capitalized words
    # Wikipedia sidebar contains language names in native script (Türkçe, Čeština...)
    # Polish proper nouns contain Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż)
    # but NOT characters like ə, ö, ü, ç, ş, ð, þ, ñ etc.
    if len(words) == 1 and len(text) >= 3 and text[0].isupper():
        _NON_POLISH_CHARS = set("əöüçşðþñãâêîôûàèìòùäëïü")
        if any(c.lower() in _NON_POLISH_CHARS for c in text):
            return True  # Contains non-Polish diacritics → likely Wikipedia language name
    # v50.7 FIX 39: Hex color codes (A7FF, FEFC, FF00, 3B82F6 etc.)
    # Scraper extracts CSS hex colors as "entities"
    if len(words) == 1 and _re.match(r'^[0-9A-Fa-f]{3,8}$', text):
        return True  # Pure hex string → CSS color code
    # v50.7 FIX 39: Font Awesome / icon font declarations
    if 'font awesome' in t_lower or 'fontawesome' in t_lower:
        return True
    # v50.7 FIX 39: CSS strings with quotes ('"Font Awesome 6 Regular";')
    if '"' in text or "'" in text:
        # Entities shouldn't contain quotes, these are CSS font-family values
        stripped = text.replace('"', '').replace("'", '').replace(';', '').strip()
        if stripped.lower() in {'font awesome', 'font awesome 6', 'font awesome 6 regular',
                                'font awesome 6 free', 'font awesome 6 brands',
                                'font awesome 5', 'font awesome 5 free'}:
            return True
        # Any string with semicolons + quotes = CSS
        if ';' in text:
            return True
    # v50.7 FIX 39: Detect CSS @font-face artifacts in multi-word strings
    _FONT_FACE_WORDS = {'font', 'family', 'face', 'style', 'weight', 'swap',
                        'src', 'url', 'unicode', 'range', 'format', 'woff',
                        'woff2', 'ttf', 'eot', 'local', 'awesome', 'regular'}
    if len(words) >= 2 and all(w in _FONT_FACE_WORDS for w in words):
        return True
    return bool(_CSS_GARBAGE_PATTERNS.search(text))

def _extract_text(item):
    """Extract text value from entity dict or string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return (item.get("entity") or item.get("text") or item.get("name")
                or item.get("ngram") or item.get("phrase") or "")
    return str(item)

def _filter_entities(entities):
    if not entities:
        return []
    clean = []
    brand_count = 0
    for ent in entities:
        if isinstance(ent, dict):
            text = ent.get("text", "") or ent.get("entity", "") or ent.get("name", "")
            if _is_css_garbage(text):
                continue
            # v50: Brand entity cap: max 2 brand entities per article
            if _is_brand_entity(text):
                brand_count += 1
                if brand_count > 2:
                    continue  # Skip excess brands
            clean.append(ent)
        elif isinstance(ent, str):
            if _is_css_garbage(ent):
                continue
            if _is_brand_entity(ent):
                brand_count += 1
                if brand_count > 2:
                    continue
            clean.append(ent)
    return clean


# v50: Brand entity detection patterns
_BRAND_PATTERNS = {
    # Energy companies (common in "prąd" articles)
    "tauron", "pge", "enea", "energa", "innogy", "rwe", "e.on", "edf",
    # Telecom
    "orange", "play", "t-mobile", "plus", "polkomtel", "vectra",
    # Banks
    "pko", "mbank", "ing", "santander", "pekao", "bnp paribas", "millennium",
    # Insurance
    "pzu", "warta", "ergo hestia", "allianz", "generali", "axa",
    # Tech / general
    "allegro", "amazon", "google", "microsoft", "apple", "samsung",
    # Legal entity suffixes
    "s.a.", "sp. z o.o.", "sp.j.", "s.c.",
}

def _is_brand_entity(text: str) -> bool:
    """Check if entity text looks like a brand/company name."""
    if not text:
        return False
    t = text.lower().strip()
    # Direct match
    if t in _BRAND_PATTERNS:
        return True
    # Partial match (e.g. "TAURON Dystrybucja S.A.")
    for pattern in _BRAND_PATTERNS:
        if pattern in t:
            return True
    # Heuristic: ends with legal entity suffix
    if any(t.endswith(suf) for suf in (" s.a.", " sp. z o.o.", " sp.j.", " s.c.", " sa", " sp z oo")):
        return True
    return False

def _filter_ngrams(ngrams):
    if not ngrams:
        return []
    clean = []
    for ng in ngrams:
        text = ng.get("ngram", ng) if isinstance(ng, dict) else str(ng)
        if not _is_css_garbage(text):
            clean.append(ng)
    return clean


# v50.7 FIX 40: AI cleanup for n-grams and causal triplets
# Uses Claude Haiku (cheap, ~$0.005/call) to filter scraper garbage
# that regex-based _is_css_garbage() misses.
_AI_CLEANUP_MODEL = "claude-haiku-4-5-20251001"

def _ai_cleanup_all_s1_data(main_keyword: str, ngrams: list, causal_chains: list, 
                            causal_singles: list, placement_instruction: str,
                            entity_salience: list, entities: list) -> dict:
    """v50.7 FIX 45: One AI call to clean ALL scraper data at once.
    
    Replaces regex whack-a-mole with AI that understands context.
    Cost: ~$0.005-0.01 per call (Claude Haiku), ~2-3s.
    
    Returns dict with cleaned versions of all inputs.
    """
    # Build concise input for AI
    ng_texts = []
    for ng in ngrams[:40]:
        text = ng.get("ngram", ng) if isinstance(ng, dict) else str(ng)
        ng_texts.append(text)
    
    causal_texts = []
    for c in (causal_chains + causal_singles)[:10]:
        cause = c.get("cause", c.get("from", ""))
        effect = c.get("effect", c.get("to", ""))
        causal_texts.append(f"{cause} → {effect}")
    
    sal_texts = []
    for s in entity_salience[:25]:
        ent = s.get("entity", s.get("text", "")) if isinstance(s, dict) else str(s)
        sal = s.get("salience", 0) if isinstance(s, dict) else 0
        typ = s.get("type", "") if isinstance(s, dict) else ""
        sal_texts.append(f"{ent} ({typ}, {sal:.2f})")
    
    ent_texts = []
    for e in entities[:25]:
        text = e.get("text", e.get("entity", "")) if isinstance(e, dict) else str(e)
        ent_texts.append(text)

    prompt = f"""Temat artykułu: "{main_keyword}"

Dane poniżej pochodzą ze scrapera stron konkurencji w SERP.
DUŻO z nich to ŚMIECI: fragmenty CSS (@font-face, font-family, display:block),
kody kolorów (hex: A7FF, FF00), nazwy fontów (Menlo, Monaco, Font Awesome),
nawigacja (menu, sidebar), klasy CSS (relative;display), nazwy języków z Wikipedii,
fragmenty URL (wp-content, blog/wp), urwane zdania (zaczynające się od małej litery
lub od przyrostka słowa), elementy UI.

ZADANIE: Z każdej sekcji zwróć TYLKO elementy MERYTORYCZNIE związane z "{main_keyword}".
Odrzuć wszelkie śmieci techniczne, CSS, HTML, nawigacyjne.

=== N-GRAMY ===
{chr(10).join(ng_texts) if ng_texts else "(brak)"}

=== RELACJE KAUZALNE ===
{chr(10).join(causal_texts) if causal_texts else "(brak)"}

=== PLACEMENT INSTRUCTION (tekst) ===
{placement_instruction[:800] if placement_instruction else "(brak)"}

=== ENTITY SALIENCE ===
{chr(10).join(sal_texts) if sal_texts else "(brak)"}

=== NAMED ENTITIES ===
{chr(10).join(ent_texts) if ent_texts else "(brak)"}

Odpowiedz TYLKO w JSON (bez markdown, bez ```):
{{
  "ngrams": ["ngram1", "ngram2"],
  "causal": ["cause → effect", ...],
  "placement": "oczyszczony tekst placement instruction (bez linii z CSS/śmieciami, zachowaj strukturę emoji 🎯📌📋📎🔺)",
  "salience": ["entity1", "entity2", ...],
  "entities": ["entity1", "entity2", ...]
}}

Jeśli sekcja ma SAME śmieci, zwróć pustą listę/string."""

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0)
        response = client.messages.create(
            model=_AI_CLEANUP_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        
        # --- N-grams: keep only AI-approved ---
        clean_ng_set = set(n.lower() for n in result.get("ngrams", []))
        filtered_ngrams = []
        for ng in ngrams:
            ng_text = (ng.get("ngram", ng) if isinstance(ng, dict) else str(ng)).lower()
            if ng_text in clean_ng_set:
                filtered_ngrams.append(ng)
        
        # --- Causal: keep by cause match ---
        approved_causes = set()
        for ct in result.get("causal", []):
            parts = ct.split("→")
            if parts:
                approved_causes.add(parts[0].strip().lower())
        filtered_chains = [c for c in causal_chains 
                          if c.get("cause", c.get("from", "")).lower().strip() in approved_causes
                          or any(ac in c.get("cause", c.get("from", "")).lower() for ac in approved_causes)]
        filtered_singles = [c for c in causal_singles
                           if c.get("cause", c.get("from", "")).lower().strip() in approved_causes
                           or any(ac in c.get("cause", c.get("from", "")).lower() for ac in approved_causes)]
        
        # --- Placement: use AI-cleaned version ---
        clean_placement = result.get("placement", placement_instruction) or placement_instruction
        
        # --- Salience: keep only AI-approved entities ---
        clean_sal_set = set(s.lower() for s in result.get("salience", []))
        filtered_salience = []
        for s in entity_salience:
            ent = (s.get("entity", s.get("text", "")) if isinstance(s, dict) else str(s)).lower()
            if ent in clean_sal_set or any(cs in ent for cs in clean_sal_set):
                filtered_salience.append(s)
        
        # --- Entities: keep only AI-approved ---
        clean_ent_set = set(e.lower() for e in result.get("entities", []))
        filtered_entities = []
        for e in entities:
            text = (e.get("text", e.get("entity", "")) if isinstance(e, dict) else str(e)).lower()
            if text in clean_ent_set or any(ce in text for ce in clean_ent_set):
                filtered_entities.append(e)
        
        logger.info(f"[AI_CLEANUP] ngrams:{len(ngrams)}→{len(filtered_ngrams)} | "
                    f"causal:{len(causal_chains)+len(causal_singles)}→{len(filtered_chains)+len(filtered_singles)} | "
                    f"salience:{len(entity_salience)}→{len(filtered_salience)} | "
                    f"entities:{len(entities)}→{len(filtered_entities)} | "
                    f"placement:{'cleaned' if clean_placement != placement_instruction else 'unchanged'}")
        
        return {
            "ngrams": filtered_ngrams,
            "causal_chains": filtered_chains,
            "causal_singles": filtered_singles,
            "placement_instruction": clean_placement,
            "entity_salience": filtered_salience,
            "entities": filtered_entities,
        }
    except Exception as e:
        logger.warning(f"[AI_CLEANUP] Failed: {e}, falling back to unfiltered data")
        return {
            "ngrams": ngrams,
            "causal_chains": causal_chains,
            "causal_singles": causal_singles,
            "placement_instruction": placement_instruction,
            "entity_salience": entity_salience,
            "entities": entities,
        }
# ============================================================
# FIX #21: YMYL Cache helpers
# ============================================================
def _get_cached_ymyl(project_id, db):
    """
    Check Firestore cache for YMYL data.
    Returns cached dict or None if not found or expired.
    """
    if not db or not project_id:
        return None
    try:
        doc = db.collection("ymyl_cache").document(project_id).get()
        if doc.exists:
            data = doc.to_dict()
            # Check expiration (24 hours)
            import time
            timestamp = data.get("_cached_at", 0)
            if time.time() - timestamp < 86400:  # 24 hours
                return data.get("ymyl_data")
    except Exception as e:
        logger.debug(f"[YMYL_CACHE] Get failed: {e}")
    return None


def _cache_ymyl(project_id, ymyl_data, db):
    """
    Save YMYL data to Firestore cache.
    Returns True if successful, False otherwise.
    """
    if not db or not project_id:
        return False
    try:
        import time
        db.collection("ymyl_cache").document(project_id).set({
            "ymyl_data": ymyl_data,
            "_cached_at": time.time(),
        })
        return True
    except Exception as e:
        logger.debug(f"[YMYL_CACHE] Set failed: {e}")
        return False


# ============================================================
# v50.7 FIX 46: LOCAL YMYL DETECTION (replaces master-seo-api call)
# Single Claude Haiku call → classifies + enriches
# Eliminates 404 error from broken /api/ymyl/detect_and_enrich
# ============================================================
_YMYL_PROMPT = """Klasyfikuj temat: "{topic}"

Określ kategorię YMYL (Your Money Your Life):
- "prawo": jeśli temat dotyczy prawa, kar, przepisów, wyroków, umów, rozwodów, przestępstw
- "zdrowie": jeśli dotyczy zdrowia, chorób, leków, terapii, objawów, diagnoz
- "finanse": jeśli dotyczy inwestycji, kredytów, podatków, ubezpieczeń, oszczędności
- "general": wszystko inne

Odpowiedz TYLKO w JSON (bez markdown):
{{
  "category": "prawo"|"zdrowie"|"finanse"|"general",
  "confidence": 0.0-1.0,
  "reasoning": "krótkie uzasadnienie po polsku",
  "ymyl_intensity": "full"|"light"|"none",
  "legal": {{"articles": ["art. X k.k."], "acts": ["Kodeks karny"], "key_concepts": ["..."], "search_queries": ["..."]}},
  "medical": {{"condition": "...", "mesh_terms": [], "search_queries": []}},
  "finance": {{"regulations": [], "search_queries": []}}
}}

Wypełnij TYLKO sekcję odpowiadającą kategorii. Resztę zostaw pustą."""


def _detect_ymyl_local(main_keyword: str) -> dict:
    """Local YMYL detection using Claude Haiku. ~$0.003, ~1s. v50.7 FIX 48: Auto-retry."""
    try:
        def _call():
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0)
            return client.messages.create(
                model=_AI_CLEANUP_MODEL,  # Haiku (cheap + fast)
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": _YMYL_PROMPT.format(topic=main_keyword)}]
            )
        response = _llm_call_with_retry(_call)
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        
        category = result.get("category", "general")
        result["is_legal"] = category == "prawo"
        result["is_medical"] = category in ("zdrowie", "medycyna")
        result["is_finance"] = category in ("finanse", "finance")
        result["is_ymyl"] = category != "general"
        result["detection_method"] = "local_haiku"
        
        # Ensure all sections exist
        result.setdefault("legal", {})
        result.setdefault("medical", {})
        result.setdefault("finance", {})
        result.setdefault("ymyl_intensity", "full" if result["is_ymyl"] else "none")
        result.setdefault("confidence", 0.8)
        result.setdefault("reasoning", "")
        
        logger.info(f"[YMYL_LOCAL] {main_keyword} → {category} ({result.get('confidence', 0):.1f}) {result.get('reasoning', '')[:60]}")
        return result
    except Exception as e:
        logger.warning(f"[YMYL_LOCAL] Failed: {e}")
        return {
            "category": "general", "is_ymyl": False, "is_legal": False,
            "is_medical": False, "is_finance": False, "confidence": 0,
            "reasoning": f"Detection failed: {e}", "detection_method": "fallback",
            "ymyl_intensity": "none", "legal": {}, "medical": {}, "finance": {},
        }


def _detect_ymyl(main_keyword: str) -> dict:
    """
    YMYL detection with master-seo-api enrichment.

    Flow:
    1. Call _detect_ymyl_local as pre-filter
    2. If not YMYL: add detected_category, return
    3. If YMYL: call master-seo-api /api/ymyl/detect_and_enrich for enrichment
    4. Return enriched data with normalized detected_category
    """
    try:
        # Step 1: Pre-filter with local detection
        local_result = _detect_ymyl_local(main_keyword)
        detected_category = local_result.get("category", "general")
        is_ymyl = local_result.get("is_ymyl", False)

        # Step 2: If not YMYL, add category and return early
        if not is_ymyl:
            local_result["detected_category"] = detected_category
            local_result["enrichment_method"] = "local_only"
            return local_result

        # Step 3: If YMYL, try to enrich via master-seo-api
        try:
            master_api_url = os.environ.get("MASTER_SEO_API_URL", "http://localhost:5001")
            api_key = os.environ.get("MASTER_SEO_API_KEY", "")

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            enrich_response = http_requests.post(
                f"{master_api_url}/api/ymyl/detect_and_enrich",
                json={"keyword": main_keyword, "local_detection": local_result},
                headers=headers,
                timeout=5
            )

            if enrich_response.status_code == 200:
                # Fix #55: Guard against empty/non-JSON responses
                _resp_text = enrich_response.text.strip()
                if not _resp_text or not _resp_text.startswith('{'):
                    logger.warning(f"[YMYL_ENRICH] Master API returned empty/non-JSON: {_resp_text[:100]}, using local result")
                else:
                    enriched = enrich_response.json()
                    # Normalize detected_category from response
                    enriched["detected_category"] = enriched.get("detected_category", detected_category)
                    enriched["enrichment_method"] = "master_api_enriched"
                    logger.info(f"[YMYL_ENRICH] {main_keyword} enriched via master-seo-api")
                    return enriched
            else:
                logger.warning(f"[YMYL_ENRICH] Master API returned {enrich_response.status_code}, using local result")
        except Exception as e:
            logger.warning(f"[YMYL_ENRICH] Master API call failed: {e}, using local result")

        # Step 4: Fallback to local result with enrichment_method set
        local_result["detected_category"] = detected_category
        local_result["enrichment_method"] = "local_fallback"
        return local_result

    except Exception as e:
        logger.error(f"[YMYL_ENRICH] Error in _detect_ymyl: {e}")
        return {
            "category": "general", "is_ymyl": False, "is_legal": False,
            "is_medical": False, "is_finance": False, "confidence": 0,
            "reasoning": f"YMYL detection error: {e}", "detection_method": "error",
            "detected_category": "general", "enrichment_method": "error",
            "ymyl_intensity": "none", "legal": {}, "medical": {}, "finance": {},
        }


# When N-gram API fails to provide ai_topical_entities (common),
# generate proper topical entities using a fast LLM call.
# This replaces CSS/HTML garbage with real topic-based entities.
#
# Based on:
# - Patent US10235423B2: entity relatedness & notability
# - Patent US9009192B1: identifying central entities
# - Dunietz & Gillick (2014): entity salience
# - Document "Topical entities w SEO": topical entities = concepts
#   that define and contextualize a topic in Knowledge Graph
# ============================================================

_TOPICAL_ENTITY_PROMPT = """Jesteś ekspertem semantic SEO. Dla podanego tematu wygeneruj topical entities oraz N-gramy frazowe — koncepty, osoby, jednostki, prawa, urządzenia i pojęcia, które definiują ten temat w Knowledge Graph Google, PLUS frazy kluczowe które realnie pojawią się w tekście u konkurencji.

ZASADY:
1. Encje MUSZĄ być tematyczne, bezpośrednio powiązane z tematem, nie z komercyjnymi stronami w SERP
2. Encja główna = sam temat (lub jego najbardziej precyzyjna forma)
3. Encje wtórne = 16-20 kluczowych konceptów powiązanych (podtypy, pojęcia prawne/medyczne/techniczne, procesy, konsekwencje, wyjątki, edge cases)
4. Dla każdej encji: 1 trójka E-A-V (Encja → Atrybut → Wartość)
5. 5-8 par co-occurrence (encje które powinny występować blisko siebie w tekście)
6. 10-15 semantic_ngrams — 2-4 wyrazowe frazy które MUSZĄ się pojawić w dobrym artykule o tym temacie (nie encje, ale konkretne wyrażenia jak „warunkowe umorzenie postępowania", „kara pozbawienia wolności", „stan po użyciu alkoholu")
7. NIE dodawaj firm komercyjnych, dat, cen, taryf
8. Odpowiedź TYLKO w JSON, bez markdown, bez komentarzy

FORMAT JSON:
{
  "primary_entity": {"text": "...", "type": "CONCEPT"},
  "secondary_entities": [
    {"text": "...", "type": "PERSON|CONCEPT|UNIT|LAW|DEVICE|EVENT|PROCESS", "eav": "encja → atrybut → wartość"}
  ],
  "semantic_ngrams": [
    {"phrase": "...", "importance": "HIGH|MEDIUM", "reason": "dlaczego ważne"}
  ],
  "svo_triples": [
    {"subject": "encja", "verb": "czasownik/relacja", "object": "wartość/encja", "context": "opcjonalny kontekst"}
  ],
  "cooccurrence_pairs": [
    {"entity1": "...", "entity2": "...", "reason": "dlaczego blisko"}
  ],
  "placement_instruction": "Krótka instrukcja rozmieszczenia encji w tekście (2-3 zdania)"
}

Dla svo_triples: wygeneruj 10-15 trójek Subject→Verb→Object które MODEL MUSI wyrazić w tekście.
Przykłady dla "jazda po alkoholu":
  {"subject": "jazda po alkoholu", "verb": "skutkuje", "object": "zakazem prowadzenia pojazdów 3-15 lat"}
  {"subject": "sąd", "verb": "orzeka obligatoryjnie", "object": "zakaz prowadzenia przy art. 178a §1"}
  {"subject": "stężenie alkoholu", "verb": "decyduje o kwalifikacji", "object": "przestępstwo vs wykroczenie (próg 0,5 promila)"}
  {"subject": "blokada alkoholowa", "verb": "umożliwia skrócenie", "object": "zakazu prowadzenia pojazdów"}
To są FAKTY MERYTORYCZNE które MUSZĄ znaleźć się w artykule — nie styl, nie encje ogólne."""


def _generate_topical_entities(main_keyword: str, h2_plan: list = None) -> dict:
    """Generate topical entities for keyword using fast LLM call.
    
    Returns dict with: primary_entity, secondary_entities, cooccurrence_pairs,
    placement_instruction. Returns empty dict on failure.
    
    Uses gpt-4.1-mini for speed (~1-2s) and cost efficiency.
    """
    if not OPENAI_API_KEY:
        logger.warning("[TOPICAL_ENTITIES] No OpenAI API key, skipping")
        return {}
    
    try:
        import openai as _openai
        client = _openai.OpenAI(api_key=OPENAI_API_KEY)
        
        h2_context = ""
        if h2_plan:
            h2_context = f"\nPlan H2 artykułu: {' | '.join(h2_plan[:8])}"
        
        user_msg = f"Temat: \"{main_keyword}\"{h2_context}\n\nWygeneruj topical entities dla tego tematu."
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _TOPICAL_ENTITY_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=1200,
            timeout=15
        )
        
        raw = response.choices[0].message.content.strip()
        # Clean potential markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        
        if not isinstance(result, dict) or "primary_entity" not in result:
            logger.warning(f"[TOPICAL_ENTITIES] Invalid response structure")
            return {}
        
        logger.info(f"[TOPICAL_ENTITIES] ✅ Generated {len(result.get('secondary_entities', []))} topical entities for '{main_keyword}'")
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"[TOPICAL_ENTITIES] JSON parse error: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[TOPICAL_ENTITIES] Error: {e}")
        return {}


def _topical_to_entity_list(topical_result: dict) -> list:
    """Convert topical entity result to standard entity list format.
    
    Returns list of dicts compatible with clean_entities / ai_topical format:
    [{"text": "...", "type": "...", "eav": "...", "source": "topical_generator"}]
    """
    if not topical_result:
        return []
    
    entities = []
    
    # Primary entity first
    primary = topical_result.get("primary_entity", {})
    if primary and primary.get("text"):
        entities.append({
            "text": primary["text"],
            "entity": primary["text"],
            "type": primary.get("type", "CONCEPT"),
            "eav": primary.get("eav", ""),
            "source": "topical_generator",
            "is_primary": True
        })
    
    # Secondary entities — expanded to 20
    for ent in topical_result.get("secondary_entities", [])[:20]:
        if ent and ent.get("text"):
            entities.append({
                "text": ent["text"],
                "entity": ent["text"],
                "type": ent.get("type", "CONCEPT"),
                "eav": ent.get("eav", ""),
                "source": "topical_generator",
                "is_primary": False
            })
    
    return entities


def _topical_to_ngrams(topical_result: dict) -> list:
    """Extract semantic_ngrams from topical entity result.
    
    Returns list of dicts in clean_ngrams format:
    [{"ngram": "...", "freq_median": 1, "freq_max": 3, "site_distribution": "1/5", "source": "topical_generator"}]
    """
    if not topical_result:
        return []
    
    ngrams = []
    for ng in topical_result.get("semantic_ngrams", [])[:15]:
        if not ng:
            continue
        phrase = ng.get("phrase", "") if isinstance(ng, dict) else str(ng)
        if not phrase or len(phrase) < 4:
            continue
        importance = (ng.get("importance", "MEDIUM") if isinstance(ng, dict) else "MEDIUM").upper()
        # Map importance to frequency targets
        freq_median = 3 if importance == "HIGH" else 1
        freq_max = 6 if importance == "HIGH" else 3
        ngrams.append({
            "ngram": phrase,
            "freq_median": freq_median,
            "freq_max": freq_max,
            "site_distribution": "2/5",  # treat as present in 2 competitors
            "source": "topical_generator",
            "importance": importance,
        })
    return ngrams



def _topical_to_eav(topical_result: dict) -> list:
    """Extract EAV triples from topical entity result for batch prompt injection.
    
    Returns list of dicts:
    [{"entity": "kodeks karny", "attribute": "penalizuje", "value": "jazdę po alkoholu art. 178a", "type": "CONCEPT"}]
    """
    if not topical_result:
        return []
    
    eav_list = []
    
    # Primary entity EAV
    primary = topical_result.get("primary_entity", {})
    if primary and primary.get("eav"):
        eav_raw = primary["eav"]
        parts = [p.strip() for p in eav_raw.split("→")]
        if len(parts) >= 3:
            eav_list.append({
                "entity": primary.get("text", parts[0]),
                "attribute": parts[1],
                "value": parts[2],
                "type": primary.get("type", "CONCEPT"),
                "is_primary": True,
            })
    
    # Secondary entities EAV
    for ent in topical_result.get("secondary_entities", [])[:18]:
        if not ent or not ent.get("eav"):
            continue
        eav_raw = ent["eav"]
        parts = [p.strip() for p in eav_raw.split("→")]
        if len(parts) >= 3:
            eav_list.append({
                "entity": ent.get("text", parts[0]),
                "attribute": parts[1],
                "value": parts[2],
                "type": ent.get("type", "CONCEPT"),
                "is_primary": False,
            })
    
    return eav_list


def _topical_to_svo(topical_result: dict) -> list:
    """Extract SVO triples from topical entity result.
    
    Returns list of dicts:
    [{"subject": "sąd", "verb": "orzeka obligatoryjnie", "object": "zakaz prowadzenia", "context": "przy art. 178a §1"}]
    """
    if not topical_result:
        return []
    
    svo_list = []
    for triple in topical_result.get("svo_triples", [])[:15]:
        if not isinstance(triple, dict):
            continue
        subj = triple.get("subject", "")
        verb = triple.get("verb", "")
        obj = triple.get("object", "")
        if subj and verb and obj:
            svo_list.append({
                "subject": subj,
                "verb": verb,
                "object": obj,
                "context": triple.get("context", ""),
            })
    return svo_list


def _topical_to_placement_instruction(topical_result: dict, main_keyword: str) -> str:
    """Build placement instruction from topical entities.
    
    Generates structured placement rules following entity salience research:
    - Primary entity → H1 + first sentence
    - Secondary entities → H2 + first paragraphs
    - E-A-V triples → explicit description in text
    - Co-occurrence pairs → same paragraph
    """
    if not topical_result:
        return ""
    
    lines = []
    primary = topical_result.get("primary_entity", {})
    secondary = topical_result.get("secondary_entities", [])[:8]
    cooc = topical_result.get("cooccurrence_pairs", [])[:5]
    
    # Primary entity
    if primary and primary.get("text"):
        p_text = primary["text"]
        lines.append(f'🎯 ENCJA GŁÓWNA: "{p_text}"')
        lines.append(f'   → W tytule H1 i w pierwszym zdaniu artykułu')
        lines.append(f'   → Jako PODMIOT zdań (nie dopełnienie)')
        if primary.get("eav"):
            lines.append(f'   → Opisz wprost: {primary["eav"]}')
    
    # First paragraph entities
    fp_ents = [e["text"] for e in secondary[:3] if e.get("text")]
    if fp_ents:
        lines.append(f'')
        lines.append(f'📌 PIERWSZY AKAPIT (100 słów): Wprowadź razem z encją główną:')
        lines.append(f'   {", ".join(fp_ents)}')
    
    # H2 entities
    h2_ents = [e for e in secondary if e.get("text")]
    if h2_ents:
        lines.append(f'')
        lines.append(f'📋 ENCJE TEMATYCZNE (do rozmieszczenia w tekście):')
        for e in h2_ents:
            eav = f': {e["eav"]}' if e.get("eav") else ""
            lines.append(f'   • "{e["text"]}" ({e.get("type", "CONCEPT")}){eav}')
    
    # Co-occurrence pairs
    if cooc:
        lines.append(f'')
        lines.append(f'🔗 CO-OCCURRENCE (umieść w TYM SAMYM akapicie):')
        for pair in cooc:
            e1 = pair.get("entity1", "")
            e2 = pair.get("entity2", "")
            reason = pair.get("reason", "")
            if e1 and e2:
                lines.append(f'   • "{e1}" + "{e2}"{" (" + reason + ")" if reason else ""}')
    
    return "\n".join(lines)


def _topical_to_cooccurrence(topical_result: dict) -> list:
    """Extract co-occurrence pairs in standard format."""
    if not topical_result:
        return []
    pairs = []
    for pair in topical_result.get("cooccurrence_pairs", [])[:8]:
        if pair.get("entity1") and pair.get("entity2"):
            pairs.append({
                "entity1": pair["entity1"],
                "entity2": pair["entity2"],
                "source": "topical_generator"
            })
    return pairs

def _filter_h2_patterns(patterns):
    """Filter H2 patterns: remove CSS garbage AND navigation elements."""
    # v49: Navigation terms that appear as H2 on scraped pages
    _NAV_H2_TERMS = {
        "wyszukiwarka", "nawigacja", "moje strony", "mapa serwisu", "mapa strony",
        "biuletyn informacji publicznej", "redakcja serwisu", "dostępność",
        "nota prawna", "polityka prywatności", "regulamin", "deklaracja dostępności",
        "newsletter", "social media", "archiwum", "logowanie", "rejestracja",
        "komenda miejska", "komenda powiatowa", "inne wersje portalu",
        "kontakt", "o nas", "strona główna", "menu główne", "szukaj",
        "przydatne linki", "informacje", "stopka", "cookie",
    }
    if not patterns:
        return []
    clean = []
    for p in patterns:
        text = p if isinstance(p, str) else (p.get("pattern", "") if isinstance(p, dict) else str(p))
        if not text or len(text) <= 3:
            continue
        t_lower = text.strip().lower()
        # Skip CSS garbage
        if _is_css_garbage(text):
            continue
        # v49: Skip navigation H2s
        if t_lower in _NAV_H2_TERMS:
            continue
        # Skip if contains nav term (partial match for longer phrases)
        if any(nav in t_lower for nav in _NAV_H2_TERMS if len(nav) >= 8):
            continue
        # Skip very short generic H2s
        if len(text.strip()) < 5:
            continue
        clean.append(p)
    return clean


def _filter_cooccurrence(pairs):
    """Remove co-occurrence pairs where either entity is CSS/nav garbage."""
    if not pairs:
        return []
    clean = []
    for pair in pairs:
        if isinstance(pair, dict):
            e1 = pair.get("entity_1", pair.get("entity1", ""))
            e2 = pair.get("entity_2", pair.get("entity2", ""))
            if isinstance(e1, list) and len(e1) >= 2:
                e1, e2 = str(e1[0]), str(e1[1])
            if not _is_css_garbage(str(e1)) and not _is_css_garbage(str(e2)):
                clean.append(pair)
        elif isinstance(pair, str):
            if not _is_css_garbage(pair):
                clean.append(pair)
    return clean


def _sanitize_placement_instruction(text):
    """Remove lines from placement instruction that reference garbage entities."""
    if not text or not isinstance(text, str):
        return ""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        quoted = _re.findall(r'"([^"]+)"', line)
        has_garbage = any(_is_css_garbage(q) for q in quoted)
        if has_garbage:
            continue
        # v50.7 FIX 39: Also check unquoted entity-like words on the line
        # Placement lines like "📎 ENCJE: A7FF, bluish, vivid" have no quotes
        line_lower = line.lower()
        # Check for hex color codes in the line
        hex_matches = _re.findall(r'\b[A-Fa-f0-9]{4,8}\b', line)
        if hex_matches and not any(c.isalpha() and c.lower() not in 'abcdef' for m in hex_matches for c in m):
            # Line contains hex color codes with no other alpha → suspicious
            pure_hex = [m for m in hex_matches if _re.match(r'^[0-9A-Fa-f]{4,8}$', m)]
            if len(pure_hex) >= 2:
                continue  # Multiple hex codes → CSS color line
        # v50.7: Check for Font Awesome references
        if 'font awesome' in line_lower or 'fontawesome' in line_lower:
            continue
        # v50.7: Check for CSS property patterns anywhere in the line
        if any(css_pat in line_lower for css_pat in [
            'placeholder{', 'relative;', 'serif;', '{color', 'display:', 
            'font-family', '@font-face', 'woff2', '.woff', '.ttf',
        ]):
            continue
        # v50.4: Filter lines where a PERSON entity appears alongside a brand
        if quoted and any(_is_brand_entity(q) for q in quoted):
            non_brand_quoted = [q for q in quoted if not _is_brand_entity(q)]
            line_lower = line.lower()
            if "person" in line_lower and non_brand_quoted:
                continue  # Skip: this is a brand contact person
        # v50.4: Filter relation lines that are scraped sentence fragments
        if "→" in line:
            parts = line.split("→")
            if len(parts) >= 3:
                relation_value = parts[-1].strip()
                if len(relation_value.split()) > 8:
                    continue
        clean_lines.append(line)
    result = "\n".join(clean_lines).strip()
    # v50.4: If >60% of instruction was garbage, data is too contaminated
    if len(result) < len(text) * 0.4:
        return ""
    return result


# ============================================================
# CONFIG
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "brajen-seo-secret-" + str(uuid.uuid4()))

BRAJEN_API = os.environ.get("BRAJEN_API_URL", "https://master-seo-api.onrender.com")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")

REQUEST_TIMEOUT = 120
HEAVY_REQUEST_TIMEOUT = 360  # For editorial_review, final_review, full_article (6 min)
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]

# v50.7 FIX 48: Auto-retry for transient LLM API errors (429 quota, 529 overloaded, 503 unavailable)
LLM_RETRY_MAX = 3
LLM_RETRY_DELAYS = [10, 30, 60]  # seconds between retries (429/503)
LLM_RETRY_DELAYS_529 = [5, 15]   # krótkie dla 529 — fail fast, przejdź do fallback modelu
LLM_RETRYABLE_CODES = {429, 503, 529}
LLM_529_MAX_RETRIES = 2  # max 2 retry dla 529 zamiast 3 — szybsze przełączenie na Haiku

def _llm_call_with_retry(fn, *args, **kwargs):
    """Wrap LLM API call with retry on transient errors.
    
    Retries on: 429 (rate limit/quota), 503 (unavailable), 529 (overloaded).
    Does NOT retry on: 400 (bad request), 401 (auth), 404, etc.
    """
    last_error = None
    for attempt in range(LLM_RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            # Extract HTTP status code from various API client exceptions
            status = getattr(e, 'status_code', None) or getattr(e, 'status', None)
            if status is None:
                # anthropic.APIStatusError / openai.APIStatusError store it in .status_code
                err_str = str(e)
                for code in LLM_RETRYABLE_CODES:
                    if str(code) in err_str:
                        status = code
                        break
            
            if status in LLM_RETRYABLE_CODES:
                # v52.4: 529 = serwer przeciążony — fail fast i przejdź do fallback modelu
                is_529 = (status == 529)
                max_r = LLM_529_MAX_RETRIES if is_529 else LLM_RETRY_MAX
                delays = LLM_RETRY_DELAYS_529 if is_529 else LLM_RETRY_DELAYS
                if attempt < max_r:
                    delay = delays[min(attempt, len(delays) - 1)]
                    logger.warning(f"[LLM_RETRY] {status} attempt {attempt+1}/{max_r+1}, retry in {delay}s: {str(e)[:120]}")
                    time.sleep(delay)
                    continue
            raise  # Non-retryable or max retries exceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auth: require env vars, no hardcoded fallbacks
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
APP_USERNAME = os.environ.get("APP_USERNAME", "")
if not APP_PASSWORD or not APP_USERNAME:
    logger.critical("⚠️ APP_PASSWORD and APP_USERNAME must be set as environment variables!")

# Store active jobs in memory (for SSE) with TTL cleanup
active_jobs = {}
_JOBS_TTL_HOURS = 6


def _cleanup_old_jobs():
    """Remove jobs older than TTL to prevent memory leaks."""
    cutoff = datetime.utcnow() - timedelta(hours=_JOBS_TTL_HOURS)
    stale = [jid for jid, job in active_jobs.items()
             if job.get("created_at", datetime.utcnow()) < cutoff]
    for jid in stale:
        del active_jobs[jid]
    if stale:
        logger.info(f"[CLEANUP] Removed {len(stale)} stale jobs")


# ============================================================
# AUTH
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        # Timing-safe comparison to prevent timing attacks
        user_ok = secrets.compare_digest(username.encode(), APP_USERNAME.encode()) if APP_USERNAME else False
        pass_ok = secrets.compare_digest(password.encode(), APP_PASSWORD.encode()) if APP_PASSWORD else False
        if user_ok and pass_ok:
            session["logged_in"] = True
            session["user"] = username
            return redirect(url_for("index"))
        error = "Nieprawidłowy login lub hasło"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ============================================================

# ─────────────────────────────────────────────────────────
# Wikipedia fetch for YMYL legal enrichment
# ─────────────────────────────────────────────────────────
import urllib.request as _urllib_req, urllib.parse as _urllib_parse, json as _json_mod

def _fetch_wikipedia_legal_article(article_ref):
    """Fetch Wikipedia summary for a legal article ref like 'art. 178a k.k.'"""
    import re as _re
    q = article_ref.strip()
    q = _re.sub(r'art\.\s*', 'Art. ', q, flags=_re.IGNORECASE)
    q = q.replace('k.k.', 'Kodeks karny').replace('k.w.', 'Kodeks wykroczen')
    q = q.replace('k.c.', 'Kodeks cywilny').replace('k.p.', 'Kodeks postepowania cywilnego')
    try:
        search_url = "https://pl.wikipedia.org/w/api.php?" + _urllib_parse.urlencode({
            "action": "query", "list": "search", "srsearch": q,
            "format": "json", "srlimit": 3, "srprop": "snippet"
        })
        req = _urllib_req.Request(search_url, headers={"User-Agent": "Brajn2026/1.0"})
        with _urllib_req.urlopen(req, timeout=8) as r:
            data = _json_mod.loads(r.read())
        results = data.get("query", {}).get("search", [])
        if not results:
            return {"found": False, "article_ref": article_ref}
        page_title = results[0]["pageid"]
        extract_url = "https://pl.wikipedia.org/w/api.php?" + _urllib_parse.urlencode({
            "action": "query", "pageids": page_title, "prop": "extracts|info",
            "exintro": True, "explaintext": True, "inprop": "url", "format": "json"
        })
        req2 = _urllib_req.Request(extract_url, headers={"User-Agent": "Brajn2026/1.0"})
        with _urllib_req.urlopen(req2, timeout=8) as r2:
            data2 = _json_mod.loads(r2.read())
        pages = data2.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        extract = (page.get("extract") or "").strip()[:600]
        url = page.get("fullurl", "https://pl.wikipedia.org")
        title = page.get("title", results[0].get("title", ""))
        if extract:
            return {"found": True, "article_ref": article_ref, "title": title, "url": url, "extract": extract, "source": "Wikipedia (pl)"}
        return {"found": False, "article_ref": article_ref}
    except Exception as e:
        return {"found": False, "article_ref": article_ref, "error": str(e)[:60]}


def _enrich_legal_with_wikipedia(articles):
    """Fetch Wikipedia for up to 4 legal article references."""
    results = []
    seen = set()
    for art in articles[:6]:
        r = _fetch_wikipedia_legal_article(art)
        if r.get("found") and r.get("title") not in seen:
            seen.add(r["title"])
            results.append(r)
        if len(results) >= 4:
            break
    return results



# BRAJEN API CLIENT
# ============================================================
def brajen_call(method, endpoint, json_data=None, timeout=None):
    """Call BRAJEN API with retry logic for cold starts."""
    url = f"{BRAJEN_API}{endpoint}"
    req_timeout = timeout or REQUEST_TIMEOUT

    for attempt in range(MAX_RETRIES):
        try:
            if method == "get":
                resp = http_requests.get(url, timeout=req_timeout)
            else:
                resp = http_requests.post(url, json=json_data, timeout=req_timeout)

            if resp.status_code in (200, 201):
                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type:
                    return {"ok": True, "data": resp.json()}
                else:
                    return {"ok": True, "binary": True, "content": resp.content,
                            "headers": dict(resp.headers)}

            logger.warning(f"BRAJEN {method.upper()} {endpoint} → {resp.status_code}")
            if resp.status_code >= 500 and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue

            return {"ok": False, "status": resp.status_code,
                    "error": resp.text[:500]}

        except http_requests.exceptions.Timeout:
            logger.warning(f"BRAJEN timeout: {endpoint} (attempt {attempt+1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return {"ok": False, "status": 0, "error": "Timeout (Render cold start?)"}

        except http_requests.exceptions.ConnectionError as e:
            logger.warning(f"BRAJEN connection error: {endpoint}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return {"ok": False, "status": 0, "error": str(e)[:200]}

    return {"ok": False, "status": 0, "error": "All retries exhausted"}


# ============================================================
# TEXT POST-PROCESSING: strip duplicate headers, clean artifacts
# ============================================================
def _clean_batch_text(text):
    """Remove duplicate ## headers when h2: format exists, strip markdown artifacts."""
    if not text:
        return text
    lines = text.split("\n")
    has_h2_prefix = any(l.strip().startswith("h2:") for l in lines)
    has_h3_prefix = any(l.strip().startswith("h3:") for l in lines)
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove ## headers if h2: format is already used
        if has_h2_prefix and stripped.startswith("## ") and not stripped.startswith("## h2:"):
            continue
        if has_h3_prefix and stripped.startswith("### ") and not stripped.startswith("### h3:"):
            continue
        # Remove stray markdown bold headers
        if has_h2_prefix and _re.match(r'^\*\*[^*]+\*\*$', stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ============================================================
# H2 PLAN GENERATOR (from S1 + user phrases)
# ============================================================
def generate_h2_plan(main_keyword, mode, s1_data, basic_terms, extended_terms, user_h2_hints=None):
    """
    Generate optimal H2 structure from S1 analysis data.
    v45.3: Uses prompt_builder for readable prompts instead of json.dumps().
    v50.7 FIX 48: Auto-retry on 429/529.
    """
    # Extract S1 insights for fallback
    suggested_h2s = (s1_data.get("content_gaps") or {}).get("suggested_new_h2s", [])
    
    # Parse user phrases (strip ranges), for topic context only
    all_user_phrases = []
    for term_str in (basic_terms + extended_terms):
        kw = term_str.strip().split(":")[0].strip()
        if kw:
            all_user_phrases.append(kw)
    
    # Build prompts via prompt_builder
    system_prompt = build_h2_plan_system_prompt()
    user_prompt = build_h2_plan_user_prompt(
        main_keyword, mode, s1_data, all_user_phrases, user_h2_hints
    )

    # v52.4: _generate_claude ma pełny fallback chain (Sonnet→Haiku na 529)
    # _llm_call_with_retry bez fallback powoduje crash zamiast przełączenia modelu
    try:
        response_text = _generate_claude(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
        )
    except Exception as e:
        raise RuntimeError(f"H2 plan generation failed: {e}") from e
    
    # Parse JSON response
    h2_list = None
    try:
        clean = response_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list) and len(parsed) >= 2:
            h2_list = parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    if not h2_list:
        # Fallback: extract lines that look like H2s
        h2_lines = [l.strip().strip('"').strip("'").strip(",").strip('"') 
                 for l in response_text.split("\n") if l.strip() and not l.strip().startswith("[") and not l.strip().startswith("]")]
        if h2_lines:
            h2_list = h2_lines
    
    if not h2_list:
        # Ultimate fallback
        h2_list = suggested_h2s[:7] + ["Najczęściej zadawane pytania"] if suggested_h2s else [main_keyword, "Najczęściej zadawane pytania"]
    
    # ═══ v50.8 FIX 50: Enforce H2 count limits based on mode ═══
    MAX_H2 = {"fast": 4, "standard": 14}  # fast=3+FAQ, standard=up to 13+FAQ
    max_allowed = MAX_H2.get(mode, 10)
    
    if len(h2_list) > max_allowed:
        logger.info(f"[H2_PLAN] ✂️ Trimming {len(h2_list)} H2s to {max_allowed} (mode={mode})")
        # Keep FAQ at the end
        has_faq = any("pytani" in h.lower() for h in h2_list[-2:])
        if has_faq:
            faq = [h for h in h2_list if "pytani" in h.lower()][-1]
            content_h2s = [h for h in h2_list if "pytani" not in h.lower()]
            h2_list = content_h2s[:max_allowed - 1] + [faq]
        else:
            h2_list = h2_list[:max_allowed]
    
    return h2_list



# ============================================================
# TEXT GENERATION (Claude + OpenAI)
# ============================================================
def generate_batch_text(pre_batch, h2, batch_type, article_memory=None, engine="claude", openai_model=None, temperature=None):
    """Generate batch text using optimized prompts built from pre_batch data.
    
    v45.3: Replaces raw json.dumps() with structured natural language prompts
    that Claude can follow effectively. Uses prompt_builder module.
    v50.8 FIX 49: Adaptive thinking (effort) + web search for YMYL.
    """
    system_prompt = build_system_prompt(pre_batch, batch_type)
    user_prompt = build_user_prompt(pre_batch, h2, batch_type, article_memory)

    if engine == "openai" and OPENAI_API_KEY:
        return _generate_openai(system_prompt, user_prompt, model=openai_model, temperature=temperature)
    else:
        # v50.9 FIX 53: Thinking only for YMYL. Regular content uses user temperature.
        is_ymyl = pre_batch.get("_is_ymyl", False)
        ymyl_intensity = pre_batch.get("_ymyl_intensity", "none")
        
        # Thinking (effort) only for YMYL where accuracy matters
        if ymyl_intensity == "full":
            effort = "high"
        elif ymyl_intensity == "light":
            effort = "medium"
        else:
            effort = None  # No thinking, user temperature controls output
        
        # Web search: only for YMYL content (legal/medical/finance)
        use_web_search = is_ymyl and ymyl_intensity == "full"
        
        return _generate_claude(system_prompt, user_prompt,
                                effort=effort, web_search=use_web_search,
                                temperature=temperature)


def _generate_claude(system_prompt, user_prompt, effort=None, web_search=False, temperature=None):
    """Generate text using Anthropic Claude.
    
    v50.7 FIX 48: Auto-retry on 429/529.
    v50.8 FIX 49: Adaptive thinking (effort) + web search for YMYL.
    v50.9 FIX 52: User-configurable temperature.
    
    Args:
        effort: "high" | "medium" | "low" | None (None = no effort param, uses temperature)
        web_search: True = enable web_search tool (for YMYL fact verification)
        temperature: 0.0-1.0, user-configured. When thinking is enabled, forced to 1.
    """
    # v50.9: User temperature (default 0.7 if not set)
    user_temp = temperature if temperature is not None else 0.7
    
    def _call():
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=0)
        
        kwargs = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        
        # v50.8 FIX 49: Adaptive thinking: budget_tokens scales by task difficulty.
        # max_tokens must exceed budget_tokens (thinking counts against it).
        # YMYL (legal/medical) → more reasoning → better accuracy
        # Regular content → less reasoning → faster, cheaper
        if effort:
            kwargs["temperature"] = 1  # Required: temperature must be 1 with thinking
            budget = {"high": 3000, "medium": 1500, "low": 500}.get(effort, 1500)
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget
            }
            # max_tokens must be > budget_tokens; ensure 4000 for output
            kwargs["max_tokens"] = budget + 4000
        else:
            kwargs["temperature"] = user_temp  # v50.9: user-configurable temperature
        
        # v50.8 FIX 49: Web search tool for YMYL content
        # Claude searches the web to verify legal/medical facts during generation
        if web_search:
            kwargs["tools"] = [
                {"type": "web_search_20250305", "name": "web_search"}
            ]
        
        response = client.messages.create(**kwargs)
        
        # v50.8: Parse response: may contain thinking blocks + text + web search results
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        
        return "\n".join(text_parts).strip()
    
    try:
        return _llm_call_with_retry(_call)
    except Exception as e:
        # v50.8: Graceful fallback: if thinking/web_search not supported, retry without
        err_str = str(e).lower()
        if effort and ("thinking" in err_str or "budget_tokens" in err_str or "temperature" in err_str):
            logger.warning(f"[FIX49] Thinking not supported, falling back: {str(e)[:100]}")
            effort = None  # Disable for retry
            return _llm_call_with_retry(_call)
        if web_search and ("web_search" in err_str or "tool" in err_str):
            logger.warning(f"[FIX49] Web search not supported, falling back: {str(e)[:100]}")
            web_search = False  # Disable for retry
            return _llm_call_with_retry(_call)
        raise


def _generate_openai(system_prompt, user_prompt, model=None, temperature=None):
    """Generate text using OpenAI GPT. v50.7 FIX 48: Auto-retry on 429/529."""
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI not installed, falling back to Claude")
        return _generate_claude(system_prompt, user_prompt, temperature=temperature)
    
    effective_model = model or OPENAI_MODEL
    user_temp = temperature if temperature is not None else 0.7
    
    # v50.7 FIX 43: GPT-5.x and o-series use max_completion_tokens, not max_tokens
    use_new_param = any(effective_model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
    token_param = {"max_completion_tokens": 4000} if use_new_param else {"max_tokens": 4000}
    
    def _call():
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=user_temp,
            **token_param
        )
        return response.choices[0].message.content.strip()
    return _llm_call_with_retry(_call)


def generate_faq_text(paa_data, pre_batch=None, engine="claude", openai_model=None, temperature=None):
    """Generate FAQ section using optimized prompts.
    
    v45.3: Uses prompt_builder for structured instructions instead of json.dumps().
    v45.3.4: Handles paa_data as list or dict.
    """
    # Normalize: if paa_data is a list, wrap it as dict
    if isinstance(paa_data, list):
        paa_data = {"serp_paa": paa_data}
    elif not isinstance(paa_data, dict):
        paa_data = {}

    system_prompt = build_faq_system_prompt(pre_batch)
    user_prompt = build_faq_user_prompt(paa_data, pre_batch)

    if engine == "openai" and OPENAI_API_KEY:
        return _generate_openai(system_prompt, user_prompt, model=openai_model, temperature=temperature)
    else:
        # FAQ = simple Q&A, no thinking needed
        return _generate_claude(system_prompt, user_prompt, effort=None, temperature=temperature)


# ============================================================
# WORKFLOW ORCHESTRATOR (SSE)
# ============================================================
def run_workflow_sse(job_id, main_keyword, mode, h2_structure, basic_terms, extended_terms, engine="claude", openai_model=None, temperature=None):
    """
    Full BRAJEN workflow as a generator yielding SSE events.
    Follows PROMPT_v45_2.md EXACTLY:
    KROK 1: S1 → 2: YMYL → 3: (H2 already provided) → 4: Create → 5: Hierarchy →
    6: Batch Loop → 7: PAA → 8: Final Review → 9: Editorial → 10: Export
    """
    # Per-session model override for OpenAI
    effective_openai_model = openai_model or OPENAI_MODEL
    def emit(event_type, data):
        """Yield SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    job = active_jobs.get(job_id, {})
    step_times = {}  # {step_num: {"start": time, "end": time}}
    workflow_start = time.time()
    
    engine_label = "OpenAI " + effective_openai_model if engine == "openai" else "Claude " + ANTHROPIC_MODEL
    temp_label = f" [temp={temperature}]" if temperature is not None else ""
    yield emit("log", {"msg": f"🚀 Workflow: {main_keyword} [{mode}] [🤖 {engine_label}]{temp_label}"})
    
    if engine == "openai" and not OPENAI_API_KEY:
        yield emit("log", {"msg": "⚠️ OPENAI_API_KEY nie ustawiony, fallback na Claude"})
        engine = "claude"

    def step_start(num):
        step_times[num] = {"start": time.time()}

    def step_done(num):
        if num in step_times:
            step_times[num]["end"] = time.time()
            elapsed = step_times[num]["end"] - step_times[num]["start"]
            step_times[num]["elapsed"] = round(elapsed, 1)
            return round(elapsed, 1)
        return 0

    try:
        # ─── KROK 1: S1 Analysis ───
        step_start(1)
        yield emit("step", {"step": 1, "name": "S1 Analysis", "status": "running"})
        yield emit("log", {"msg": f"POST /api/s1_analysis → {main_keyword}"})

        s1_result = brajen_call("post", "/api/s1_analysis", {"main_keyword": main_keyword})
        if not s1_result["ok"]:
            yield emit("workflow_error", {"step": 1, "msg": f"S1 Analysis failed: {s1_result.get('error', 'unknown')}"})
            return

        s1_raw = s1_result["data"]
        
        # Debug: log S1 response structure for diagnostics
        la = s1_raw.get("length_analysis", {})
        sa = s1_raw.get("serp_analysis", {})
        logger.info(f"[S1_DEBUG] top keys: {sorted(s1_raw.keys())}")
        logger.info(f"[S1_DEBUG] length_analysis: rec={la.get('recommended')}, med={la.get('median')}, avg={la.get('average')}, urls={la.get('analyzed_urls')}")
        logger.info(f"[S1_DEBUG] serp_analysis keys: {sorted(sa.keys()) if sa else 'EMPTY'}")
        logger.info(f"[S1_DEBUG] recommended_length(top): {s1_raw.get('recommended_length')}, median_length(top): {s1_raw.get('median_length')}")
        # PAA diagnostic — shows in workflow logs
        _paa_raw = s1_raw.get("paa") or s1_raw.get("paa_questions") or sa.get("paa_questions") or []
        yield emit("log", {"msg": f"🔍 S1 PAA debug: s1_raw.paa={len(s1_raw.get('paa') or [])}, s1_raw.paa_questions={len(s1_raw.get('paa_questions') or [])}, serp_analysis.paa_questions={len(sa.get('paa_questions') or [])}, s1_raw top keys={list(s1_raw.keys())[:8]}"})
        
        # ═══ AI MIDDLEWARE: Clean S1 data ═══
        s1 = process_s1_for_pipeline(s1_raw, main_keyword)
        cleanup_stats = s1.get("_cleanup_stats", {})
        cleanup_method = cleanup_stats.get("method", "unknown")
        items_removed = cleanup_stats.get("items_removed", 0)
        ai_entity_panel = s1.get("_ai_entity_panel") or {}
        garbage_summary = ai_entity_panel.get("garbage_summary", "")
        if garbage_summary:
            yield emit("log", {"msg": f"🧹 S1 cleanup ({cleanup_method}): {garbage_summary}"})
        
        h2_patterns = len((s1.get("competitor_h2_patterns") or (s1.get("serp_analysis") or {}).get("competitor_h2_patterns") or []))
        causal_count = (s1.get("causal_triplets") or {}).get("count", 0)
        gaps_count = ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("total_gaps", 0)
        suggested_h2s = ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("suggested_new_h2s", [])

        step_done(1)
        yield emit("step", {"step": 1, "name": "S1 Analysis", "status": "done",
                            "detail": f"{h2_patterns} H2 patterns | {causal_count} causal triplets | {gaps_count} content gaps"})
        
        # S1 data for UI, already cleaned by Claude Sonnet middleware
        entity_seo = s1.get("entity_seo") or {}
        raw_entities = entity_seo.get("top_entities", entity_seo.get("entities", []))[:35]
        raw_must_mention = entity_seo.get("must_mention_entities", [])[:15]
        raw_ngrams = (s1.get("ngrams") or s1.get("hybrid_ngrams") or [])[:60]
        serp_analysis = s1.get("serp_analysis") or {}
        raw_h2_patterns = (s1.get("competitor_h2_patterns") or serp_analysis.get("competitor_h2_patterns") or [])[:30]

        # v48.0: Claude already cleaned, lightweight safety net only
        clean_entities = _filter_entities(raw_entities)[:18]
        clean_must_mention = _filter_entities(raw_must_mention)[:8]
        clean_ngrams = _filter_ngrams(raw_ngrams)[:30]
        clean_h2_patterns = _filter_h2_patterns(raw_h2_patterns)[:20]

        # v48.0: Read Claude's topical/named entity split
        ai_topical = entity_seo.get("ai_topical_entities", [])
        ai_named = entity_seo.get("ai_named_entities", [])
        ai_entity_panel = s1.get("_ai_entity_panel") or {}

        # ═══ v50.4 FIX 20: TOPICAL ENTITY GENERATOR ═══
        # If N-gram API didn't produce ai_topical_entities (common failure),
        # generate proper topical entities using a fast LLM call.
        # This prevents CSS artifacts (vivid, bluish, reviews) from becoming
        # the "primary entity" in the article.
        topical_gen_result = {}
        topical_gen_entities = []
        topical_gen_placement = ""
        topical_gen_cooc = []
        topical_gen_eav = []   # EAV triples: entity → attribute → value
        topical_gen_svo = []   # SVO triples: subject → verb → object

        if not ai_topical:
            # Check if scraper entities are mostly garbage
            _clean_count = len(clean_entities)
            _raw_count = len(raw_entities)
            _garbage_ratio = 1.0 - (_clean_count / max(_raw_count, 1))
            
            # Generate topical entities if:
            # - No AI topical entities from N-gram API, AND
            # - Either high garbage ratio (>40% filtered) or very few clean entities
            if _garbage_ratio > 0.4 or _clean_count < 4:
                yield emit("log", {"msg": f"🧬 Encje ze scrapera niskiej jakości ({_clean_count}/{_raw_count} przefiltrowanych), generuję topical entities..."})
                topical_gen_result = _generate_topical_entities(main_keyword)
                
                if topical_gen_result:
                    topical_gen_entities = _topical_to_entity_list(topical_gen_result)
                    topical_gen_placement = _topical_to_placement_instruction(topical_gen_result, main_keyword)
                    topical_gen_cooc = _topical_to_cooccurrence(topical_gen_result)
                    topical_gen_ngrams = _topical_to_ngrams(topical_gen_result)
                    topical_gen_eav = _topical_to_eav(topical_gen_result)
                    topical_gen_svo = _topical_to_svo(topical_gen_result)

                    # Override: use topical entities as primary
                    ai_topical = topical_gen_entities
                    clean_entities = topical_gen_entities[:18]

                    # Merge semantic ngrams from topical generator into clean_ngrams
                    existing_ng_texts = set(
                        (ng.get("ngram", "") if isinstance(ng, dict) else str(ng)).lower()
                        for ng in clean_ngrams
                    )
                    new_tg_ngrams = [
                        ng for ng in topical_gen_ngrams
                        if ng.get("ngram", "").lower() not in existing_ng_texts
                    ]
                    clean_ngrams = clean_ngrams + new_tg_ngrams
                    if new_tg_ngrams:
                        yield emit("log", {"msg": f"📐 +{len(new_tg_ngrams)} semantic n-gramów z topical generatora → łącznie {len(clean_ngrams)}"})
                    if topical_gen_eav:
                        yield emit("log", {"msg": f"🔺 EAV trójki: {len(topical_gen_eav)} (encja→atrybut→wartość)"})
                    if topical_gen_svo:
                        yield emit("log", {"msg": f"🔗 SVO trójki: {len(topical_gen_svo)} (podmiot→relacja→obiekt)"})

                    _ent_names = [_extract_text(e) for e in topical_gen_entities[:5]]
                    yield emit("log", {"msg": f"🧬 Topical entities ({len(topical_gen_entities)}): {', '.join(_ent_names)}"})
                else:
                    yield emit("log", {"msg": "⚠️ Topical entity generation failed, używam przefiltrowanych encji ze scrapera"})
            else:
                # Scraper entities OK — still run topical gen for semantic ngrams
                yield emit("log", {"msg": f"✅ Encje ze scrapera OK ({_clean_count} clean) — generuję semantic n-gramy..."})
                topical_gen_result = _generate_topical_entities(main_keyword)
                if topical_gen_result:
                    topical_gen_ngrams = _topical_to_ngrams(topical_gen_result)
                    topical_gen_entities = _topical_to_entity_list(topical_gen_result)
                    topical_gen_placement = _topical_to_placement_instruction(topical_gen_result, main_keyword)
                    topical_gen_cooc = _topical_to_cooccurrence(topical_gen_result)
                    topical_gen_eav = _topical_to_eav(topical_gen_result)
                    topical_gen_svo = _topical_to_svo(topical_gen_result)
                    existing_ng_texts = set(
                        (ng.get("ngram", "") if isinstance(ng, dict) else str(ng)).lower()
                        for ng in clean_ngrams
                    )
                    new_tg_ngrams = [ng for ng in topical_gen_ngrams if ng.get("ngram", "").lower() not in existing_ng_texts]
                    clean_ngrams = clean_ngrams + new_tg_ngrams
                    yield emit("log", {"msg": f"📐 +{len(new_tg_ngrams)} semantic n-gramów → łącznie {len(clean_ngrams)}"})
                    if topical_gen_eav:
                        yield emit("log", {"msg": f"🔺 EAV: {len(topical_gen_eav)} | SVO: {len(topical_gen_svo)} trójek semantycznych"})

        # If Claude/N-gram API produced topical entities, use them as primary
        if ai_topical:
            clean_entities = ai_topical[:18]
            yield emit("log", {"msg": f"🧠 Topical entities: {', '.join(_extract_text(e) for e in ai_topical[:6])}"})
        if ai_named:
            yield emit("log", {"msg": f"🏷️ Named entities (AI, filtered): {', '.join(_extract_text(e) for e in ai_named[:5])}"})

        # Legacy: AI-extracted entities fallback
        ai_entities = entity_seo.get("ai_extracted_entities", [])
        if ai_entities and not ai_topical and len(clean_entities) < 5:
            yield emit("log", {"msg": f"🤖 Uzupełniam encje z AI: {', '.join(str(e) for e in ai_entities[:5])}"})

        # v48.0: concept_entities = topical from generator (or Claude, or backend)
        concept_entities = ai_topical if ai_topical else _filter_entities(
            entity_seo.get("concept_entities", []) or s1.get("concept_entities", [])
        )[:15]
        topical_summary_raw = entity_seo.get("topical_summary", {}) or s1.get("topical_summary", {})
        if isinstance(topical_summary_raw, str):
            topical_summary = {"agent_instruction": topical_summary_raw} if topical_summary_raw else {}
        else:
            topical_summary = topical_summary_raw

        # v48.0: Emit AI entity panel for dashboard
        cleanup_stats = s1.get("_cleanup_stats") or {}
        if ai_entity_panel:
            yield emit("ai_entity_panel", ai_entity_panel)
            gs = ai_entity_panel.get("garbage_summary", "")
            if gs:
                yield emit("log", {"msg": f"🧹 S1 cleanup ({cleanup_stats.get('method', '?')}): {gs[:100]}"})

        # v47.0: Read entity_salience, co-occurrence, placement from backend
        entity_seo_raw = s1.get("entity_seo") or {}
        backend_entity_salience = _filter_entities(entity_seo_raw.get("entity_salience", []) or s1.get("entity_salience", []))
        backend_entity_cooccurrence = _filter_cooccurrence(entity_seo_raw.get("entity_cooccurrence", []) or s1.get("entity_cooccurrence", []))
        backend_entity_placement = (
            s1.get("entity_placement") or
            entity_seo_raw.get("entity_placement", {})
        )
        backend_placement_instruction = _sanitize_placement_instruction(
            (s1.get("semantic_enhancement_hints") or {}).get("placement_instruction", "") or
            (backend_entity_placement.get("placement_instruction", "") if isinstance(backend_entity_placement, dict) else "")
        )
        # v47.0: Read enhanced semantic hints
        sem_hints = s1.get("semantic_enhancement_hints") or s1.get("semantic_hints") or {}
        backend_first_para_entities = _filter_entities(
            sem_hints.get("first_paragraph_entities", []) or
            (backend_entity_placement.get("first_paragraph_entities", []) if isinstance(backend_entity_placement, dict) else [])
        )
        backend_h2_entities = _filter_entities(
            sem_hints.get("h2_entities", []) or
            (backend_entity_placement.get("h2_entities", []) if isinstance(backend_entity_placement, dict) else [])
        )
        backend_cooccurrence_pairs = _filter_cooccurrence(
            sem_hints.get("cooccurrence_pairs", []) or
            (backend_entity_placement.get("cooccurrence_pairs", []) if isinstance(backend_entity_placement, dict) else [])
        )[:5]
        # v47.0: must_cover_concepts & concept_instruction from semantic_enhancement_hints
        must_cover_concepts = _filter_entities(sem_hints.get("must_cover_concepts", []) or (topical_summary.get("must_cover", []) if isinstance(topical_summary, dict) else []))
        concept_instruction = _sanitize_placement_instruction(sem_hints.get("concept_instruction", "") or (topical_summary.get("agent_instruction", "") if isinstance(topical_summary, dict) else ""))

        # ═══ v50.4 FIX 20: Override backend placement with topical-generated data ═══
        # When topical entity generator was used, its output is BETTER than
        # the scraper-sourced placement (which may contain CSS artifacts,
        # brand contacts, and sentence fragments from competitor pages).
        if topical_gen_placement:
            backend_placement_instruction = topical_gen_placement
            yield emit("log", {"msg": "🧬 Placement instruction: z topical entity generator (zamiast scrapera)"})
        elif ai_topical and not topical_gen_entities:
            # v50.7 FIX 44: N-gram API gave entities but no placement, build from entities
            # v50.7 FIX 47: Use _extract_text(): handles str+dict
            _ai_names = [_extract_text(e) for e in ai_topical[:8] if _extract_text(e)]
            if _ai_names:
                _lines = [
                    f'🎯 ENCJA GŁÓWNA: "{_ai_names[0]}"',
                    f'   → W tytule H1 i w pierwszym zdaniu artykułu',
                ]
                if len(_ai_names) > 1:
                    _lines.append(f'📌 PIERWSZY AKAPIT: Wprowadź razem: {", ".join(_ai_names[:3])}')
                if len(_ai_names) > 3:
                    _lines.append(f'📋 ENCJE TEMATYCZNE:')
                    for _n in _ai_names[1:]:
                        _lines.append(f'   • "{_n}" (CONCEPT)')
                backend_placement_instruction = "\n".join(_lines)
                yield emit("log", {"msg": f"🧬 Placement instruction: wygenerowane z ai_topical entities ({len(_ai_names)} encji)"})
        if topical_gen_cooc:
            backend_cooccurrence_pairs = topical_gen_cooc + backend_cooccurrence_pairs[:2]
            yield emit("log", {"msg": f"🧬 Co-occurrence: {len(topical_gen_cooc)} par z topical generator"})
        if (topical_gen_entities or ai_topical) and not must_cover_concepts:
            # Use clean topical entities as must_cover_concepts
            must_cover_concepts = (topical_gen_entities or ai_topical)[:14]

        # ═══ v50.4 FIX 21 + v50.7 FIX 44: Override ALL contamination paths ═══
        # Override with clean topical entities regardless of source:
        # - topical_gen_entities: from topical generator (when scraper data was garbage)
        # - ai_topical: from N-gram API concept extraction (when API provided entities)
        # Without this, sem_hints/placement/salience keep raw S1 CSS garbage.
        _override_entities = topical_gen_entities or ai_topical or []
        if _override_entities:
            # Override first paragraph entities with topical primary + top 2 secondary
            backend_first_para_entities = _override_entities[:3]
            # Override H2 entities with remaining topical entities
            backend_h2_entities = _override_entities[3:8]
            # Override entity salience with topical-generated entities
            # (prevents "Asturianu Azərbaycanca" as primary in dashboard)
            backend_entity_salience = []
            for i, ent in enumerate(_override_entities[:12]):
                _sal = round(0.85 - (i * 0.06), 2)  # Primary=0.85, decreasing
                backend_entity_salience.append({
                    # v50.7 FIX 47: Use _extract_text(): entities can be str OR dict
                    "entity": _extract_text(ent),
                    "salience": max(0.05, _sal),
                    "type": ent.get("type", "CONCEPT") if isinstance(ent, dict) else "CONCEPT",
                    "source": "topical_override"
                })
            yield emit("log", {"msg": f"🧬 Entity salience + first_para + H2: nadpisane ({len(backend_entity_salience)} encji, src={'topical_gen' if topical_gen_entities else 'ai_topical'})"})

            # v50.5 FIX 35: Also override backend_entity_placement for dashboard display
            # v50.7 FIX 47: Use _extract_text(): handles str+dict
            _fp_names = [_extract_text(e) for e in backend_first_para_entities]
            _h2_names = [_extract_text(e) for e in backend_h2_entities]
            backend_entity_placement = {
                "first_paragraph_entities": _fp_names,
                "h2_entities": _h2_names,
                "placement_instruction": backend_placement_instruction,
                "source": "topical_override"
            }

            # v50.7 FIX 34: Override sem_hints with clean topical data
            # v50.7 FIX 47: Use _extract_text(): handles str+dict
            _primary_name = _extract_text(_override_entities[0]) if _override_entities else main_keyword
            _secondary_names = [_extract_text(e) for e in _override_entities[1:4]]
            sem_hints = {
                # v50.7 FIX 44: Include BOTH "text" and "entity" keys
                # Dashboard reads .entity, backend reads .text
                "primary_entity": {"text": _primary_name, "entity": _primary_name, "type": "CONCEPT", "salience": 0.85, "source": "topical_override"},
                "secondary_entities": [{"text": n, "entity": n, "type": "CONCEPT"} for n in _secondary_names],
                "must_cover_concepts": [_extract_text(e) for e in (must_cover_concepts or _override_entities[:8])],
                "placement_instruction": backend_placement_instruction,
                "first_paragraph_entities": _fp_names,
                "h2_entities": _h2_names,
                "cooccurrence_pairs": backend_cooccurrence_pairs[:5] if backend_cooccurrence_pairs else [],
                "concept_instruction": concept_instruction,
                "checkpoints": {
                    "batch_1": f"H1 contains '{_primary_name}', first paragraph mentions {', '.join(_secondary_names[:2])}",
                    "batch_3": "entity_density >= 2.5, min 50% critical entities, min 30% must_cover_concepts",
                    "batch_5": "topic_completeness >= 50%, concept coverage >= 50%",
                    "pre_faq": "all critical entities present, all MUST topics covered",
                },
                "source": "topical_override"
            }
            yield emit("log", {"msg": f"🧬 sem_hints: nadpisane (primary: {_primary_name}, src={'topical_gen' if topical_gen_entities else 'ai_topical'})"})

        if backend_entity_salience:
            yield emit("log", {"msg": f"🔬 Entity Salience: {len(backend_entity_salience)} encji z analizy konkurencji"})
        if backend_entity_cooccurrence:
            yield emit("log", {"msg": f"🔗 Co-occurrence: {len(backend_entity_cooccurrence)} par encji"})
        if backend_placement_instruction:
            yield emit("log", {"msg": "📐 Placement instructions: wygenerowane z analizy konkurencji"})
        if must_cover_concepts:
            yield emit("log", {"msg": f"💡 Must-cover concepts: {len(must_cover_concepts)} pojęć tematycznych"})

        # v45.4.1: Filter semantic_keyphrases (Gemini may return YouTube/JS garbage)
        raw_semantic_kp = s1.get("semantic_keyphrases") or []
        clean_semantic_kp = [kp for kp in raw_semantic_kp if not _is_css_garbage(
            kp.get("phrase", kp) if isinstance(kp, dict) else str(kp)
        )]

        # v45.4.1: Filter causal triplets: remove CSS-contaminated extractions
        def _filter_causal(triplets):
            """Remove causal triplets where cause/effect looks like CSS or truncated."""
            if not triplets:
                return []
            clean = []
            for t in triplets:
                cause = t.get("cause", t.get("from", ""))
                effect = t.get("effect", t.get("to", ""))
                # Skip if cause or effect is too short, too long, or CSS garbage
                if len(cause) < 5 or len(effect) < 5:
                    continue
                if len(cause) > 120 or len(effect) > 120:
                    continue  # Truncated sentence fragments
                if _is_css_garbage(cause) or _is_css_garbage(effect):
                    continue
                # v50.7 FIX 39: Detect truncated sentence fragments
                # "unkiem, że opiera się..." starts mid-word → garbage
                cause_stripped = cause.strip()
                effect_stripped = effect.strip()
                if cause_stripped and cause_stripped[0].islower() and not cause_stripped.startswith(("np.", "tj.", "m.in.")):
                    # Starts mid-word/mid-sentence, likely truncated scrape
                    # Check if first word looks like a Polish suffix (ends with -iem, -iem, -ych, -ów)
                    first_word = cause_stripped.split()[0].rstrip(",.:;")
                    if len(first_word) < 4 or first_word.endswith(("iem", "iem", "ych", "ów", "ami", "ach", "owi")):
                        continue
                if effect_stripped and effect_stripped[0].islower() and not effect_stripped.startswith(("np.", "tj.", "m.in.")):
                    first_word = effect_stripped.split()[0].rstrip(",.:;")
                    if len(first_word) < 4 or first_word.endswith(("iem", "iem", "ych", "ów", "ami", "ach", "owi")):
                        continue
                clean.append(t)
            return clean

        raw_causal_chains = (s1.get("causal_triplets") or {}).get("chains", [])[:10]
        raw_causal_singles = (s1.get("causal_triplets") or {}).get("singles", [])[:10]
        clean_causal_chains = _filter_causal(raw_causal_chains)
        clean_causal_singles = _filter_causal(raw_causal_singles)

        # v50.7 FIX 45: Comprehensive AI cleanup: one call cleans EVERYTHING
        # Replaces regex whack-a-mole with AI that understands context (~$0.008, ~2s)
        try:
            ai_cleanup = _ai_cleanup_all_s1_data(
                main_keyword=main_keyword,
                ngrams=clean_ngrams,
                causal_chains=clean_causal_chains,
                causal_singles=clean_causal_singles,
                placement_instruction=backend_placement_instruction,
                entity_salience=backend_entity_salience,
                entities=clean_entities,
            )
            _pre = {"ng": len(clean_ngrams), "cc": len(clean_causal_chains)+len(clean_causal_singles),
                    "sal": len(backend_entity_salience), "ent": len(clean_entities)}
            
            clean_ngrams = ai_cleanup["ngrams"]
            clean_causal_chains = ai_cleanup["causal_chains"]
            clean_causal_singles = ai_cleanup["causal_singles"]
            total_causal = len(clean_causal_chains) + len(clean_causal_singles)
            if not total_causal:
                yield emit("log", {"msg": f"⚠️ Causal triplets: brak danych (raw chains={len(raw_causal_chains)}, singles={len(raw_causal_singles)})"})
            backend_placement_instruction = ai_cleanup["placement_instruction"]
            backend_entity_salience = ai_cleanup["entity_salience"]
            clean_entities = ai_cleanup["entities"]
            
            _post = {"ng": len(clean_ngrams), "cc": len(clean_causal_chains)+len(clean_causal_singles),
                     "sal": len(backend_entity_salience), "ent": len(clean_entities)}
            
            changes = []
            if _pre["ng"] != _post["ng"]: changes.append(f"n-gramy {_pre['ng']}→{_post['ng']}")
            if _pre["cc"] != _post["cc"]: changes.append(f"kauzalne {_pre['cc']}→{_post['cc']}")
            if _pre["sal"] != _post["sal"]: changes.append(f"salience {_pre['sal']}→{_post['sal']}")
            if _pre["ent"] != _post["ent"]: changes.append(f"encje {_pre['ent']}→{_post['ent']}")
            
            if changes:
                yield emit("log", {"msg": f"🧹 AI cleanup: {' | '.join(changes)}"})
            else:
                yield emit("log", {"msg": "🧹 AI cleanup: dane czyste, bez zmian"})
        except Exception as ai_err:
            logger.warning(f"[AI_CLEANUP] Error in workflow: {ai_err}")

        if concept_entities:
            yield emit("log", {"msg": f"🧠 Concept entities: {len(concept_entities)} (z topical_entity_extractor)"})
        if len(clean_ngrams) < len(raw_ngrams) * 0.5:
            yield emit("log", {"msg": f"⚠️ N-gramy: {len(raw_ngrams) - len(clean_ngrams)}/{len(raw_ngrams)} odfiltrowane jako CSS garbage"})
        # PAA diagnostics + Fix #40: Claude fallback for PAA
        paa_debug = s1.get("paa") or s1.get("paa_questions") or serp_analysis.get("paa_questions") or []
        if not paa_debug:
            yield emit("log", {"msg": f"⚠️ PAA: brak pytań w s1.paa={len(s1.get('paa') or [])}, s1.paa_questions={len(s1.get('paa_questions') or [])}, serp.paa_questions={len(serp_analysis.get('paa_questions') or [])}"})
            # Fix #40: Generate PAA with Claude fallback when API returns empty
            try:
                import anthropic as _ant_paa
                _paa_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
                if _paa_key:
                    yield emit("log", {"msg": "🔄 PAA fallback: generuję pytania z Claude..."})
                    _paa_client = _ant_paa.Anthropic(api_key=_paa_key)
                    _serp_ctx = []
                    for _cs in serp_analysis.get("competitor_snippets", [])[:6]:
                        if isinstance(_cs, str) and _cs.strip():
                            _serp_ctx.append(_cs[:200])
                    _related = serp_analysis.get("related_searches", [])
                    _paa_prompt = (
                        f'Dla frazy "{main_keyword}" wygeneruj 6 pytań które użytkownicy zadają w sekcji "Ludzie pytają też" (PAA) w Google.\n'
                        f'Kontekst z SERP:\n{chr(10).join(_serp_ctx[:4])}\n'
                        f'Related searches: {", ".join(_related[:5])}\n\n'
                        'Zwróć TYLKO JSON array: [{"question": "...", "answer": "krótka odpowiedź 1-2 zdania"}]\n'
                        'Pytania po polsku, konkretne, rzeczowe.'
                    )
                    _paa_resp = _paa_client.messages.create(
                        model="claude-haiku-4-5-20251001", max_tokens=800, temperature=0,
                        messages=[{"role": "user", "content": _paa_prompt}]
                    )
                    _paa_text = _paa_resp.content[0].text.strip()
                    import re as _re_paa
                    _paa_m = _re_paa.search(r"\[[\s\S]*\]", _paa_text)
                    if _paa_m:
                        _paa_gen = json.loads(_paa_m.group())
                        paa_debug = [{"question": q.get("question",""), "answer": q.get("answer",""), "source": "claude_fallback"} for q in _paa_gen if q.get("question")]
                        yield emit("log", {"msg": f"✅ PAA fallback: {len(paa_debug)} pytań wygenerowanych z Claude"})
                    else:
                        yield emit("log", {"msg": "⚠️ PAA fallback: nie udało się sparsować JSON"})
            except Exception as _paa_err:
                yield emit("log", {"msg": f"⚠️ PAA fallback error: {str(_paa_err)[:100]}"})
        else:
            yield emit("log", {"msg": f"✅ PAA: {len(paa_debug)} pytań z SERP"})
        yield emit("s1_data", {
            # Stats for top bar, backend nests these in length_analysis{}
            "recommended_length": s1.get("recommended_length") or (s1.get("length_analysis") or {}).get("recommended"),
            "median_length": s1.get("median_length") or (s1.get("length_analysis") or {}).get("median", 0),
            "average_length": (s1.get("length_analysis") or {}).get("average") or s1.get("average_length") or s1.get("avg_length"),
            "analyzed_urls": (s1.get("length_analysis") or {}).get("analyzed_urls") or s1.get("analyzed_urls") or s1.get("urls_analyzed") or s1.get("competitor_count"),
            "word_counts": (s1.get("length_analysis") or {}).get("word_counts") or s1.get("word_counts") or [],
            "length_analysis": s1.get("length_analysis") or {},
            # SERP competitor data
            "serp_competitors": (s1.get("serp_analysis") or {}).get("competitors", s1.get("competitors", []))[:10],
            "competitor_titles": serp_analysis.get("competitor_titles", [])[:10],
            "competitor_snippets": serp_analysis.get("competitor_snippets", [])[:10],
            # Competitor structure
            "h2_patterns_count": len(clean_h2_patterns),
            "competitor_h2_patterns": clean_h2_patterns,
            "search_intent": s1.get("search_intent") or serp_analysis.get("search_intent", ""),
            "serp_sources": s1.get("serp_sources") or serp_analysis.get("competitor_urls") or s1.get("competitor_urls") or [],
            "featured_snippet": s1.get("featured_snippet") or serp_analysis.get("featured_snippet"),
            "ai_overview": s1.get("ai_overview") or serp_analysis.get("ai_overview"),
            "related_searches": s1.get("related_searches") or serp_analysis.get("related_searches") or [],
            # PAA: check multiple locations + Fix #40 fallback
            "paa_questions": (paa_debug if paa_debug else (s1.get("paa") or s1.get("paa_questions") or serp_analysis.get("paa_questions") or (s1_raw.get("serp_analysis") or {}).get("paa_questions") or s1_raw.get("paa") or []))[:10],
            # Causal triplets
            "causal_triplets_count": len(clean_causal_chains) + len(clean_causal_singles),
            "causal_count_chains": len(clean_causal_chains),
            "causal_count_singles": len(clean_causal_singles),
            "causal_chains": clean_causal_chains,
            "causal_singles": clean_causal_singles,
            "causal_instruction": (s1.get("causal_triplets") or {}).get("agent_instruction", ""),
            # Gap analysis
            "content_gaps_count": gaps_count,
            "content_gaps": (s1.get("content_gaps") or {}),
            "suggested_h2s": suggested_h2s,
            "paa_unanswered": ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("paa_unanswered", []),
            "subtopic_missing": ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("subtopic_missing", []),
            "depth_missing": ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("depth_missing", []),
            "gaps_instruction": ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("instruction", ""),
            # Entity SEO: v48.0: topical entities primary
            "entity_seo": {
                "top_entities": clean_entities,
                "must_mention": clean_must_mention,
                "ai_extracted": ai_entities[:5] if ai_entities else [],
                "entity_count": (s1.get("entity_seo") or {}).get("entity_count", len(clean_entities)),
                "relations": (s1.get("entity_seo") or {}).get("relations", [])[:10],
                "topical_coverage": (s1.get("entity_seo") or {}).get("topical_coverage", [])[:10],
                # v48.0: Topical (primary) vs Named (secondary) from Claude
                "topical_entities": ai_topical[:18] if ai_topical else concept_entities[:18],
                "named_entities": ai_named[:8] if ai_named else [],
                "concept_entities": concept_entities,
                "topical_summary": topical_summary,
                # v47.0: Salience, co-occurrence, placement from backend
                "entity_salience": backend_entity_salience[:25],
                "entity_cooccurrence": backend_entity_cooccurrence[:10],
                "entity_placement": backend_entity_placement if isinstance(backend_entity_placement, dict) else {},
                # v48.0: Cleanup info
                "cleanup_method": cleanup_stats.get("method", "unknown"),
            },
            # v47.0: Placement instruction (top-level for easy access)
            "placement_instruction": backend_placement_instruction,
            # v47.0: Concept coverage fields
            "must_cover_concepts": must_cover_concepts[:14],
            "concept_instruction": concept_instruction,
            # N-grams
            "ngrams": clean_ngrams,
            "semantic_keyphrases": clean_semantic_kp,
            # Phrase hierarchy
            "phrase_hierarchy_preview": s1.get("phrase_hierarchy_preview") or {},
            # Depth signals
            "depth_signals": s1.get("depth_signals") or {},
            "depth_missing_items": s1.get("depth_missing_items") or ({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("depth_missing", []),
            # YMYL hints
            "ymyl_hints": s1.get("ymyl_hints") or s1.get("ymyl_signals") or {},
            # PAA (already included above with serp_analysis fallback)
            "paa_unanswered_count": len(({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("paa_unanswered", [])),
            # Agent instructions
            "agent_instructions": s1.get("agent_instructions") or {},
            "semantic_hints": sem_hints,
            # Meta
            "competitive_summary": s1.get("_competitive_summary", "")
        })

        # ═══ ENTITY SALIENCE: Build instructions from topical entities (primary) ═══
        # v48.0: Topical entities first, then NER, then fallback
        s1_must_mention = []
        if ai_topical:
            s1_must_mention = ai_topical[:5]
        elif clean_must_mention:
            s1_must_mention = clean_must_mention
        if ai_entities and len(s1_must_mention) < 5:
            s1_must_mention += ai_entities[:3]
        entity_salience_instructions = build_entity_salience_instructions(
            main_keyword=main_keyword,
            entities_from_s1=s1_must_mention
        )
        if is_salience_available():
            yield emit("log", {"msg": "🔬 Entity Salience: Google NLP API aktywne, walidacja po zakończeniu artykułu"})
        else:
            yield emit("log", {"msg": "ℹ️ Entity Salience: instrukcje pozycjonowania encji aktywne (brak API key dla walidacji)"})

        # ─── KROK 2: YMYL Detection (Unified Claude Classifier) ───
        step_start(2)
        yield emit("step", {"step": 2, "name": "YMYL Detection", "status": "running"})

        # v47.2: ONE Claude Sonnet call → classifies + returns search hints
        # v50.7 FIX 46: Run LOCALLY (Haiku) instead of broken brajen_call to master-seo-api
        # 🆕 Fix #13 v4.2: Use _detect_ymyl (pre-filter + master enrichment) instead of _detect_ymyl_local
        ymyl_data = _detect_ymyl(main_keyword)
        is_legal = ymyl_data.get("is_legal", False)
        is_medical = ymyl_data.get("is_medical", False)
        is_finance = ymyl_data.get("is_finance", False)
        ymyl_confidence = ymyl_data.get("confidence", 0)
        ymyl_reasoning = ymyl_data.get("reasoning", "")
        # v50: YMYL intensity: full/light/none
        ymyl_intensity = ymyl_data.get("ymyl_intensity", "none")
        light_ymyl_note = ymyl_data.get("light_ymyl_note", "")
        
        if ymyl_reasoning:
            intensity_emoji = {"full": "🔴", "light": "🟡", "none": "⚪"}.get(ymyl_intensity, "⚪")
            yield emit("log", {"msg": f"🧠 YMYL klasyfikacja: {ymyl_data.get('category', '?')} ({ymyl_confidence}) intensity={ymyl_intensity} {intensity_emoji} | {ymyl_reasoning[:80]}"})

        legal_context = None
        medical_context = None
        ymyl_enrichment = {}  # Claude's hints for downstream

        if is_legal:
            legal_hints = ymyl_data.get("legal", {})
            articles = legal_hints.get("articles", [])
            arts_str = ", ".join(articles[:4]) if articles else "brak"
            yield emit("log", {"msg": f"⚖️ Temat prawny YMYL, przepisy: {arts_str}. Pobieram orzeczenia..."})
            
            # v47.2: Pass Claude's article hints to SAOS search
            lc = brajen_call("post", "/api/legal/get_context", {
                "main_keyword": main_keyword,
                "force_enable": True,  # Claude already classified, skip keyword gate
                "article_hints": articles,  # art. 178a k.k. etc.
                "search_queries": legal_hints.get("search_queries", []),
            })
            if lc["ok"]:
                legal_context = lc["data"]
            
            # Wikipedia enrichment for legal articles
            _wiki_articles = []
            if articles:
                yield emit("log", {"msg": f"📖 Wikipedia: szukam {len(articles[:4])} przepisów..."})
                _wiki_articles = _enrich_legal_with_wikipedia(articles)
                if _wiki_articles:
                    yield emit("log", {"msg": f"✅ Wikipedia: {len(_wiki_articles)} artykułów ({', '.join(w['title'][:25] for w in _wiki_articles)})"})
                    yield emit("legal_wiki_sources", {"articles": _wiki_articles})
                else:
                    yield emit("log", {"msg": "⚠️ Wikipedia: brak wyników"})
            
            ymyl_enrichment["legal"] = legal_hints
            ymyl_enrichment["_wiki_articles"] = _wiki_articles

        if is_medical:
            medical_hints = ymyl_data.get("medical", {})
            mesh = medical_hints.get("mesh_terms", [])
            spec = medical_hints.get("specialization", "")
            yield emit("log", {"msg": f"🏥 Temat medyczny YMYL: {spec} | MeSH: {', '.join(mesh[:3])}. Pobieram źródła..."})
            
            # v47.2: Pass Claude's MeSH hints to PubMed search
            mc = brajen_call("post", "/api/medical/get_context", {
                "main_keyword": main_keyword,
                "force_enable": True,  # Claude already classified
                "mesh_hints": mesh,  # MeSH terms for PubMed
                "condition_en": medical_hints.get("condition_latin", ""),
                "specialization": spec,
                "key_drugs": medical_hints.get("key_drugs", []),
                "evidence_note": medical_hints.get("evidence_note", ""),
            })
            if mc["ok"]:
                medical_context = mc["data"]
            else:
                # Fix #58: Fallback medical context when master API fails
                logger.warning("[YMYL] Medical context API failed, using local fallback")
                medical_context = {
                    "active": True,
                    "specialization": spec or "medycyna ogólna",
                    "condition": medical_hints.get("condition", main_keyword),
                    "condition_latin": medical_hints.get("condition_latin", ""),
                    "icd10": medical_hints.get("icd10", ""),
                    "key_drugs": medical_hints.get("key_drugs", []),
                    "evidence_note": medical_hints.get("evidence_note", "Użyj aktualnych wytycznych klinicznych."),
                    "medical_instruction": (
                        "OBOWIĄZKOWE DLA ARTYKUŁU MEDYCZNEGO:\n"
                        "1. Dodaj disclaimer: 'Artykuł ma charakter informacyjny i nie zastępuje konsultacji lekarskiej.'\n"
                        "2. Powołaj się na min. 2 instytucje (np. WHO, PTOiAu, NFZ, MZ).\n"
                        "3. Użyj sformułowań opartych na dowodach: 'badania wskazują', 'według wytycznych'.\n"
                        "4. W ostatnim akapicie każdej sekcji: zachęta do wizyty u specjalisty."
                    ),
                    "top_publications": [],
                    "mesh_terms": mesh,
                }
                yield emit("log", {"msg": "⚠️ Medical context: użyto lokalnego fallbacku (brak odpowiedzi z master API)"})

            ymyl_enrichment["medical"] = medical_hints
        
        if is_finance:
            ymyl_enrichment["finance"] = ymyl_data.get("finance", {})
            yield emit("log", {"msg": f"💰 Temat finansowy YMYL: {ymyl_reasoning[:60]}"})

        ymyl_detail = f"Legal: {'TAK' if is_legal else 'NIE'} | Medical: {'TAK' if is_medical else 'NIE'} | Finance: {'TAK' if is_finance else 'NIE'}"

        # ═══ EMIT YMYL CONTEXT FOR DASHBOARD ═══
        ymyl_panel_data = {
            "is_legal": is_legal,
            "is_medical": is_medical,
            "is_finance": is_finance,
            "ymyl_intensity": ymyl_intensity,
            "classification": {
                "category": ymyl_data.get("category", "general"),
                "confidence": ymyl_confidence,
                "reasoning": ymyl_reasoning,
                "method": ymyl_data.get("detection_method", "unknown"),
                "ymyl_intensity": ymyl_intensity,
            },
            "enrichment": ymyl_enrichment,  # v47.2: Claude's articles/MeSH/etc.
            "legal": {},
            "medical": {},
        }
        if legal_context:
            judgments_raw = legal_context.get("top_judgments") or []
            judgments_clean = []
            _legal_enrich_hints = ymyl_enrichment.get("legal", {})
            _articles_hints = _legal_enrich_hints.get("articles", [])
            _arts_str = " ".join(_articles_hints).lower()
            _is_criminal = any(x in _arts_str for x in ["k.k.", "kk", "k.w.", "kw", "kodeks karny"])
            _is_civil = any(x in _arts_str for x in ["k.c.", "kc", "k.r.o.", "kodeks cywilny"])
            _CRIM_SIG = ("ii k", "iii k", "iv k", "aka", "ako", "akz", "ii ka", "iii ka", "iv ka")
            _CIV_SIG = (" i c ", " ii c ", " iii c ", " aca ", " aco ")
            _skipped_sigs = []
            for j in judgments_raw[:10]:
                if not isinstance(j, dict): continue
                sig = (j.get("signature", j.get("caseNumber", "")) or "").lower()
                sig_p = " " + sig + " "
                if _is_criminal and not _is_civil and any(p in sig_p for p in _CIV_SIG):
                    _skipped_sigs.append(sig); continue
                if _is_civil and not _is_criminal and any(p in sig_p for p in _CRIM_SIG):
                    _skipped_sigs.append(sig); continue
                judgments_clean.append({
                    "signature": j.get("signature", j.get("caseNumber", "")),
                    "court": j.get("court", j.get("courtName", "")),
                    "date": j.get("date", j.get("judgmentDate", "")),
                    "summary": (j.get("summary", j.get("excerpt", "")))[:150],
                    "type": j.get("type", j.get("judgmentType", "")),
                    "matched_article": j.get("matched_article", ""),
                })
            if _skipped_sigs:
                yield emit("log", {"msg": f"⚠️ Pominięto {len(_skipped_sigs)} orzeczeń (błędna gałąź): {', '.join(_skipped_sigs[:3])}"})
            # v47.2: Use Claude's article hints as primary source for legal acts
            legal_enrich = ymyl_enrichment.get("legal", {})
            legal_acts = legal_enrich.get("acts", [])
            if legal_acts and isinstance(legal_acts, list):
                legal_acts = [{"name": a} if isinstance(a, str) else a for a in legal_acts[:8]]
            else:
                # Fallback: extract from context
                legal_acts = legal_context.get("legal_acts") or legal_context.get("acts") or []
                if not legal_acts and legal_context.get("legal_instruction"):
                    import re as _re
                    act_patterns = _re.findall(
                        r'(?:ustaw[aąy]?\s+(?:z\s+dnia\s+)?\d{1,2}\s+\w+\s+\d{4}[^.]*|'
                        r'[Kk]odeks\s+\w+[^.]*|'
                        r'[Rr]ozporządzeni[eua][^.]*\d{4}[^.]*|'
                        r'[Dd]yrektyw[aąy][^.]*\d{4}[^.]*)',
                        legal_context.get("legal_instruction", "")
                    )
                    legal_acts = [{"name": a.strip()[:120]} for a in act_patterns[:8]]
            
            # v47.2: Add Claude's specific articles to panel
            legal_articles = legal_enrich.get("articles", [])

            ymyl_panel_data["legal"] = {
                "instruction_preview": (legal_context.get("legal_instruction", ""))[:300],
                "judgments": judgments_clean,
                "judgments_count": len(judgments_raw),
                "judgments_skipped": _skipped_sigs[:5],
                "legal_acts": legal_acts[:8] if isinstance(legal_acts, list) else [],
                "legal_articles": legal_articles[:6],
                "citation_hint": legal_context.get("citation_hint", ""),
                "wiki_articles": ymyl_enrichment.get("_wiki_articles", []),
            }

        if medical_context:
            pubs_raw = medical_context.get("top_publications") or []
            pubs_clean = []
            for p in pubs_raw[:10]:
                if isinstance(p, dict):
                    pubs_clean.append({
                        "title": (p.get("title", ""))[:120],
                        "authors": (p.get("authors", ""))[:80],
                        "year": p.get("year", ""),
                        "pmid": p.get("pmid", ""),
                        "journal": (p.get("journal", ""))[:60],
                        "evidence_level": p.get("evidence_level", p.get("level", "")),
                        "study_type": p.get("study_type", p.get("type", "")),
                    })
            # Evidence level breakdown
            evidence_levels = {}
            for p in pubs_clean:
                lvl = p.get("evidence_level") or p.get("study_type") or "unknown"
                evidence_levels[lvl] = evidence_levels.get(lvl, 0) + 1

            ymyl_panel_data["medical"] = {
                "instruction_preview": (medical_context.get("medical_instruction", ""))[:300],
                "publications": pubs_clean,
                "publications_count": len(pubs_raw),
                "evidence_levels": evidence_levels,
                "guidelines": medical_context.get("guidelines") or [],
            }

        yield emit("ymyl_context", ymyl_panel_data)

        step_done(2)
        yield emit("step", {"step": 2, "name": "YMYL Detection", "status": "done", "detail": ymyl_detail})

        # ─── v51: Auto-generate BASIC phrases from S1 entity + ngram frequency ───
        if not basic_terms:
            auto_basic = []
            seen_texts = set()
            
            # === 1. ENTITIES (primary, Surfer-style) ===
            # Topical entities have per-source frequency from competition
            all_entity_sources = []
            _first_topical_seen = False  # v57.1: track if first topical entity was processed
            if ai_topical:
                all_entity_sources.extend(ai_topical)
            if clean_entities:
                all_entity_sources.extend(clean_entities)
            
            for ent in all_entity_sources:
                if not isinstance(ent, dict):
                    continue
                text = (ent.get("text") or ent.get("entity") or ent.get("display_text") or "").strip()
                if not text or text.lower() in seen_texts:
                    continue
                
                freq_min = ent.get("freq_min", 0)
                freq_max = ent.get("freq_max", 0)
                freq_median = ent.get("freq_median", 0)
                sources_count = ent.get("sources_count", 0)
                is_topical = (
                    ent.get("source") in ("topical_generator", "ai_cleanup", "concept_entities")
                    or ent.get("type", "").upper() == "TOPICAL"
                    or ent.get("eav") or ent.get("is_primary")
                )
                
                if is_topical:
                    # v57.1 FIX: Topical entities bypass freq filter — use salience-based targets
                    # First topical entity = primary (highest salience), rest = secondary
                    is_primary = ent.get("is_primary") or not _first_topical_seen
                    _first_topical_seen = True
                    if is_primary:
                        target_min, target_max = 3, 8
                    else:
                        target_min, target_max = 2, 5
                elif sources_count >= 2 and freq_median >= 1:
                    # Original: entity with competition frequency data
                    target_min = max(1, freq_median)
                    target_max = max(target_min + 1, (freq_median + freq_max) // 2)
                    target_min = min(target_min, 25)
                    target_max = min(target_max, 30)
                else:
                    continue
                
                auto_basic.append(f"{text}: {target_min}-{target_max}x")
                seen_texts.add(text.lower())
            
            entity_count = len(auto_basic)
            
            # === 2. N-GRAMS (supplementary) ===
            for ng in (clean_ngrams or []):
                if not isinstance(ng, dict):
                    continue
                text = ng.get("ngram", "")
                if not text or text.lower() in seen_texts:
                    continue
                
                freq_median = ng.get("freq_median", 0)
                freq_max = ng.get("freq_max", 0)
                sites = ng.get("site_distribution", "0/0")
                
                try:
                    site_count = int(sites.split("/")[0])
                    site_total = int(sites.split("/")[1]) if "/" in sites else 1
                except (ValueError, IndexError):
                    site_count = 0
                    site_total = 1
                
                # v58: Relaxed filter — include rare but topically valid n-grams
                # OLD: site_count < 2 or freq_median < 2 (missed ~70% of Surfer-comparable phrases)
                # NEW: include if present in ≥1 competitor OR if freq_median ≥ 1
                if site_count < 1 and freq_median < 1:
                    continue
                
                # Rare phrases (in 1 competitor or low freq) → suggest low target
                if site_count <= 1 or freq_median <= 1:
                    target_min = 1
                    target_max = max(2, freq_max or 2)
                else:
                    target_min = max(1, freq_median)
                    target_max = max(target_min + 1, (freq_median + freq_max) // 2)
                target_min = min(target_min, 25)
                target_max = min(target_max, 30)
                
                auto_basic.append(f"{text}: {target_min}-{target_max}x")
                seen_texts.add(text.lower())
            
            if auto_basic:
                basic_terms = auto_basic[:40]
                ngram_count = len(auto_basic) - entity_count
                yield emit("log", {"msg": f"📊 Auto-BASIC z S1: {len(basic_terms)} fraz ({entity_count} encji + {ngram_count} n-gramów)"})
                yield emit("auto_basic_terms", {"terms": basic_terms})
                for term in basic_terms[:5]:
                    yield emit("log", {"msg": f"  • {term}"})
                if len(basic_terms) > 5:
                    yield emit("log", {"msg": f"  ... i {len(basic_terms) - 5} więcej"})

        # ─── KROK 3: H2 Planning (auto from S1 + phrase optimization) ───
        step_start(3)
        yield emit("step", {"step": 3, "name": "H2 Planning", "status": "running"})

        if not h2_structure or len(h2_structure) == 0:
            # Fully automatic: generate H2 from S1
            yield emit("log", {"msg": "Generuję strukturę H2 z analizy S1 (liczba H2 = tyle ile wymaga temat)..."})
            h2_structure = generate_h2_plan(
                main_keyword=main_keyword,
                mode=mode,
                s1_data=s1,
                basic_terms=basic_terms,
                extended_terms=extended_terms
            )
        elif len(h2_structure) > 0:
            # User provided hints, use them as hints, optimize with S1
            user_hints = list(h2_structure)  # save original
            yield emit("log", {"msg": f"Optymalizuję {len(user_hints)} wskazówek H2 na podstawie S1..."})
            h2_structure = generate_h2_plan(
                main_keyword=main_keyword,
                mode=mode,
                s1_data=s1,
                basic_terms=basic_terms,
                extended_terms=extended_terms,
                user_h2_hints=user_hints
            )

        # Emit the final H2 plan for the UI
        yield emit("h2_plan", {"h2_list": h2_structure, "count": len(h2_structure)})
        yield emit("log", {"msg": f"Plan H2 ({len(h2_structure)} sekcji): {' | '.join(h2_structure)}"})
        step_done(3)
        yield emit("step", {"step": 3, "name": "H2 Planning", "status": "done",
                            "detail": f"{len(h2_structure)} nagłówków H2"})

        # ─── KROK 4: Create Project ───
        step_start(4)
        yield emit("step", {"step": 4, "name": "Create Project", "status": "running"})

        # Build keywords array
        keywords = [{"keyword": main_keyword, "type": "MAIN", "target_min": 8, "target_max": 25}]
        for term_str in basic_terms:
            parts = term_str.strip().split(":")
            kw = parts[0].strip()
            if not kw or kw == main_keyword:
                continue
            tmin, tmax = 1, 5
            if len(parts) > 1:
                range_str = parts[1].strip()
                if "-" in range_str:
                    try:
                        range_parts = range_str.replace("x", "").split("-")
                        tmin = int(range_parts[0].strip())
                        tmax = int(range_parts[1].strip())
                    except (ValueError, IndexError):
                        pass
            keywords.append({"keyword": kw, "type": "BASIC", "target_min": tmin, "target_max": tmax})

        for term_str in extended_terms:
            parts = term_str.strip().split(":")
            kw = parts[0].strip()
            if not kw or kw == main_keyword:
                continue
            tmin, tmax = 1, 2
            if len(parts) > 1:
                range_str = parts[1].strip()
                if "-" in range_str:
                    try:
                        range_parts = range_str.replace("x", "").split("-")
                        tmin = int(range_parts[0].strip())
                        tmax = int(range_parts[1].strip())
                    except (ValueError, IndexError):
                        pass
            keywords.append({"keyword": kw, "type": "EXTENDED", "target_min": tmin, "target_max": tmax})

        # ═══ v57 FIX: Add concept entities as type="ENTITY" for separate tracking ═══
        # Concept entities from S1/topical generator get tracked like keywords
        # but with type="ENTITY" so panel shows them separately.
        _existing_kw_lower = {k["keyword"].lower() for k in keywords}
        _entity_sources = must_cover_concepts or concept_entities or []
        entity_kw_count = 0
        for ent in _entity_sources[:12]:
            ent_text = (_extract_text(ent) if isinstance(ent, dict) else str(ent)).strip()
            if not ent_text or ent_text.lower() in _existing_kw_lower or ent_text.lower() == main_keyword.lower():
                continue
            is_primary = ent.get("is_primary", False) if isinstance(ent, dict) else False
            tmin = 3 if is_primary else 2
            tmax = 8 if is_primary else 5
            keywords.append({"keyword": ent_text, "type": "ENTITY", "target_min": tmin, "target_max": tmax})
            _existing_kw_lower.add(ent_text.lower())
            entity_kw_count += 1
        if entity_kw_count:
            yield emit("log", {"msg": f"🧬 Entity keywords: {entity_kw_count} encji dodanych jako type=ENTITY"})

        # ═══ Keyword deduplication (word-boundary safe) ═══
        pre_dedup_count = len(keywords)
        keywords = deduplicate_keywords(keywords, main_keyword)
        if len(keywords) < pre_dedup_count:
            yield emit("log", {"msg": f"🧹 Dedup: {pre_dedup_count} → {len(keywords)} keywords (usunięto {pre_dedup_count - len(keywords)} duplikatów)"})

        yield emit("log", {"msg": f"Keywords: {len(keywords)} ({sum(1 for k in keywords if k['type']=='BASIC')} BASIC, {sum(1 for k in keywords if k['type']=='EXTENDED')} EXTENDED, {sum(1 for k in keywords if k['type']=='ENTITY')} ENTITY)"})

        # Filter entity_seo before sending to project (remove CSS garbage)
        filtered_entity_seo = (s1.get("entity_seo") or {}).copy()
        if "top_entities" in filtered_entity_seo:
            filtered_entity_seo["top_entities"] = _filter_entities(filtered_entity_seo["top_entities"])
        if "entities" in filtered_entity_seo:
            filtered_entity_seo["entities"] = _filter_entities(filtered_entity_seo["entities"])
        if "must_mention_entities" in filtered_entity_seo:
            filtered_entity_seo["must_mention_entities"] = _filter_entities(filtered_entity_seo["must_mention_entities"])

        project_payload = {
            "main_keyword": main_keyword,
            "mode": mode,
            "h2_structure": h2_structure,
            "keywords": keywords,
            "s1_data": {
                "causal_triplets": (s1.get("causal_triplets") or {}),
                "content_gaps": (s1.get("content_gaps") or {}),
                "entity_seo": filtered_entity_seo,
                "paa": (s1.get("paa") or []),
                "ngrams": _filter_ngrams((s1.get("ngrams") or [])[:30]),
                "competitor_h2_patterns": _filter_h2_patterns((s1.get("competitor_h2_patterns") or [])[:30])
            },
            "target_length": 3500 if mode == "standard" else 2000,
            "is_legal": is_legal,
            "is_medical": is_medical,
            "is_finance": is_finance,
            "is_ymyl": is_legal or is_medical or is_finance,
            # v50: YMYL intensity for conditional pipeline behavior
            "ymyl_intensity": ymyl_intensity,
            "light_ymyl_note": light_ymyl_note,
            "legal_context": legal_context,
            "legal_wiki_articles": ymyl_enrichment.get("_wiki_articles", []),
            "medical_context": medical_context,
            # v47.2: Claude's YMYL enrichment (articles, MeSH, evidence notes)
            "ymyl_enrichment": ymyl_enrichment,
        }

        create_result = brajen_call("post", "/api/project/create", project_payload)
        if not create_result["ok"]:
            yield emit("workflow_error", {"step": 4, "msg": f"Create Project failed: {create_result.get('error', 'unknown')}"})
            return

        project = create_result["data"]
        project_id = project.get("project_id")
        total_batches = project.get("total_planned_batches", len(h2_structure))

        step_done(4)
        yield emit("step", {"step": 4, "name": "Create Project", "status": "done",
                            "detail": f"ID: {project_id} | Mode: {mode} | Batche: {total_batches}"})
        yield emit("project", {"project_id": project_id, "total_batches": total_batches})

        # Store project_id in job
        job["project_id"] = project_id

        # ─── KROK 5: Phrase Hierarchy ───
        step_start(5)
        yield emit("step", {"step": 5, "name": "Phrase Hierarchy", "status": "running"})
        hier_result = brajen_call("get", f"/api/project/{project_id}/phrase_hierarchy")
        phrase_hierarchy_data = {}
        if hier_result["ok"]:
            hier = hier_result["data"]
            phrase_hierarchy_data = hier  # Store for injection into pre_batch
            strategy = (hier.get("strategies") or {})
            # Emit phrase hierarchy preview to frontend
            hier_preview = hier.get("strategies") or hier.get("phrase_hierarchy") or hier
            if isinstance(hier_preview, dict):
                yield emit("log", {"msg": f"🔤 Phrase Hierarchy: {len(hier_preview)} strategii ({', '.join(list(hier_preview.keys())[:3])})"})
                yield emit("phrase_hierarchy", {"phrase_hierarchy_preview": hier_preview})
            step_done(5)
            yield emit("step", {"step": 5, "name": "Phrase Hierarchy", "status": "done",
                                "detail": json.dumps(strategy, ensure_ascii=False)[:200]})
        else:
            yield emit("step", {"step": 5, "name": "Phrase Hierarchy", "status": "warning",
                                "detail": "Nie udało się pobrać, kontynuuję"})

        # ─── KROK 6: Batch Loop ───
        step_start(6)
        yield emit("step", {"step": 6, "name": "Batch Loop", "status": "running",
                            "detail": f"0/{total_batches} batchy"})

        # ═══ AI MIDDLEWARE: Track accepted batches for memory ═══
        accepted_batches_log = []
        # Fix #56: Global cross-batch keyword counter
        _global_main_kw_count = 0
        _GLOBAL_KW_MAX = 6  # max occurrences of main keyword in ENTIRE article

        # Fix #60: Featured Snippet tracking — lists and tables across article
        _global_list_count = 0  # count of <ul>/<ol> across all batches
        _global_table_count = 0  # count of <table> across all batches

        # Fix #59: Calculate per-batch word target from S1 recommended_length
        _s1_rec_length = (s1.get("recommended_length") if s1 else 0) or 0
        if _s1_rec_length and total_batches > 0:
            # FAQ batch is shorter (~20% of budget), rest split evenly
            _faq_budget = int(_s1_rec_length * 0.15)
            _content_budget = _s1_rec_length - _faq_budget
            _words_per_batch = max(120, int(_content_budget / max(1, total_batches - 1)))
            _batch_word_override = {
                "min_words": max(100, _words_per_batch - 30),
                "max_words": _words_per_batch + 40,
                "suggested_min": _words_per_batch - 20,
                "suggested_max": _words_per_batch + 30,
                "target": _words_per_batch,
            }
            yield emit("log", {"msg": f"📏 S1 rec: {_s1_rec_length} słów → per batch: ~{_words_per_batch} słów ({total_batches} batchy)"})
        else:
            _batch_word_override = None

        # ═══ ENTITY CONTENT PLAN — assign lead entity per batch/H2 ═══
        # Each H2 section gets ONE lead entity as "paragraph subject opener".
        # Prevents every akapit from starting with the same main keyword.
        #
        # Algorithm:
        # - Batch 1 (INTRO): always main keyword
        # - Batch N (H2): pick secondary entity whose text overlaps most with H2 title
        # - Fallback: cycle through secondary entities in order (1, 2, 3…)
        def _build_entity_content_plan(h2_list, main_kw, secondary_entities):
            """Returns list[str]: lead entity name per batch (index 0 = batch 1)."""
            plan = []
            used_indices = set()

            def _best_entity_for_h2(h2_title, exclude=None):
                """Find secondary entity most relevant to given H2 title."""
                h2_words = set(h2_title.lower().split())
                best_idx, best_score = None, 0
                for i, ent in enumerate(secondary_entities):
                    if exclude and i in exclude:
                        continue
                    name = (_extract_text(ent) if isinstance(ent, dict) else str(ent)).lower()
                    ent_words = set(name.split())
                    overlap = len(h2_words & ent_words)
                    # Partial match — check if any ent word is substring of h2
                    partial = sum(1 for w in ent_words if any(w in hw or hw in w for hw in h2_words))
                    score = overlap * 2 + partial
                    if score > best_score and i not in used_indices:
                        best_idx, best_score = i, score
                return best_idx

            for i, h2 in enumerate(h2_list):
                batch_num_local = i + 1
                if batch_num_local == 1:
                    # INTRO always uses main keyword
                    plan.append(main_kw)
                else:
                    idx = _best_entity_for_h2(h2)
                    if idx is not None:
                        ent = secondary_entities[idx]
                        plan.append(_extract_text(ent) if isinstance(ent, dict) else str(ent))
                        used_indices.add(idx)
                    else:
                        # Fallback: cycle unused secondary entities
                        cycle_idx = (batch_num_local - 2) % max(1, len(secondary_entities))
                        ent = secondary_entities[cycle_idx]
                        plan.append(_extract_text(ent) if isinstance(ent, dict) else str(ent))

            return plan

        _secondary_for_plan = [e for e in (must_cover_concepts or []) if
                               (_extract_text(e) if isinstance(e, dict) else str(e)) != main_keyword]
        _entity_content_plan = _build_entity_content_plan(
            h2_structure, main_keyword, _secondary_for_plan
        )
        if _entity_content_plan:
            plan_preview = " | ".join(
                f"B{i+1}:{n}" for i, n in enumerate(_entity_content_plan)
            )
            yield emit("log", {"msg": f"🗂️ Entity content plan: {plan_preview}"})

        for batch_num in range(1, total_batches + 1):
            yield emit("batch_start", {"batch": batch_num, "total": total_batches})
            yield emit("log", {"msg": f"── BATCH {batch_num}/{total_batches} ──"})

            # 6a: Get pre_batch_info
            yield emit("log", {"msg": f"GET /pre_batch_info"})
            pre_result = brajen_call("get", f"/api/project/{project_id}/pre_batch_info")
            if not pre_result["ok"]:
                yield emit("log", {"msg": f"⚠️ pre_batch_info error: {pre_result.get('error', '')[:100]}"})
                continue

            pre_batch = pre_result["data"]
            batch_type = pre_batch.get("batch_type", "CONTENT")
            
            # ═══ BATCH 1 = INTRO: First batch must always be introduction ═══
            if batch_num == 1 and batch_type not in ("INTRO", "intro"):
                batch_type = "INTRO"
                yield emit("log", {"msg": "📝 Batch 1 → wymuszony typ INTRO (wstęp artykułu)"})

            # ═══ Inject phrase hierarchy data for prompt_builder ═══
            if phrase_hierarchy_data:
                pre_batch["_phrase_hierarchy"] = phrase_hierarchy_data

            # ═══ Inject entity salience instructions for prompt_builder ═══
            if entity_salience_instructions:
                pre_batch["_entity_salience_instructions"] = entity_salience_instructions

            # ═══ Inject YMYL flags for depth signals ═══
            pre_batch["_is_ymyl"] = is_legal or is_medical or is_finance
            # v50: Pass intensity to prompt_builder for conditional legal/medical injection
            pre_batch["_ymyl_intensity"] = ymyl_intensity
            if light_ymyl_note:
                pre_batch["_light_ymyl_note"] = light_ymyl_note
            
            # ═══ v47.2: Inject YMYL enrichment for prompt builder ═══
            if ymyl_enrichment:
                pre_batch["_ymyl_enrichment"] = ymyl_enrichment
                # v50: Removed redundant aliases (_ymyl_key_concepts, _ymyl_evidence_note,
                # _ymyl_specialization) , data consumed through _ymyl_enrichment parent dict
                # in _fmt_legal_medical() as ymyl_enrich.get("legal"/"medical").

            # ═══ Inject last depth score for adaptive depth signals ═══
            if accepted_batches_log:
                last_accepted = accepted_batches_log[-1]
                last_depth = last_accepted.get("depth_score")
                if last_depth is not None:
                    pre_batch["_last_depth_score"] = last_depth

            # ═══ v47.0: Inject backend placement instructions for prompt_builder ═══
            if backend_placement_instruction:
                pre_batch["_backend_placement_instruction"] = backend_placement_instruction
            if backend_cooccurrence_pairs:
                pre_batch["_cooccurrence_pairs"] = backend_cooccurrence_pairs
            if backend_first_para_entities:
                pre_batch["_first_paragraph_entities"] = backend_first_para_entities
            if backend_h2_entities:
                pre_batch["_h2_entities"] = backend_h2_entities
            # v47.0: Concept coverage for prompt
            if concept_instruction:
                pre_batch["_concept_instruction"] = concept_instruction
            if must_cover_concepts:
                pre_batch["_must_cover_concepts"] = must_cover_concepts

            # Fix #59: Override batch_length with S1-correlated target
            if _batch_word_override:
                pre_batch["batch_length"] = _batch_word_override

            # ═══ Inject EAV + SVO semantic triples ═══
            if topical_gen_eav:
                pre_batch["_eav_triples"] = topical_gen_eav
            if topical_gen_svo:
                pre_batch["_svo_triples"] = topical_gen_svo
            # Fix #57: Inject semantic keyphrases for natural phrase usage
            if clean_semantic_kp:
                pre_batch["_semantic_keyphrases"] = clean_semantic_kp

            # ═══ ENTITY CONTENT PLAN — inject lead entity for this batch/H2 ═══
            if _entity_content_plan and batch_num <= len(_entity_content_plan):
                pre_batch["_section_lead_entity"] = _entity_content_plan[batch_num - 1]
            elif main_keyword:
                pre_batch["_section_lead_entity"] = main_keyword

            # Get current H2 from API (most reliable) or fallback to our plan
            h2_remaining = (pre_batch.get("h2_remaining") or [])
            semantic_plan = pre_batch.get("semantic_batch_plan") or {}
            if h2_remaining:
                current_h2 = h2_remaining[0]
            elif semantic_plan.get("h2"):
                current_h2 = semantic_plan["h2"]
            else:
                current_h2 = h2_structure[min(batch_num-1, len(h2_structure)-1)]

            must_kw = (pre_batch.get("keywords") or {}).get("basic_must_use", [])
            ext_kw = (pre_batch.get("keywords") or {}).get("extended_this_batch", [])
            stop_kw = (pre_batch.get("keyword_limits") or {}).get("stop_keywords", [])

            yield emit("log", {"msg": f"Typ: {batch_type} | H2: {current_h2}"})
            yield emit("log", {"msg": f"MUST: {len(must_kw)} | EXTENDED: {len(ext_kw)} | STOP: {len(stop_kw)}"})

            # Emit batch instructions for UI display
            caution_kw = (pre_batch.get("keyword_limits") or {}).get("caution_keywords", [])
            batch_length_info = pre_batch.get("batch_length") or {}
            enhanced_data = pre_batch.get("enhanced") or {}
            
            yield emit("batch_instructions", {
                "batch": batch_num,
                "total": total_batches,
                "batch_type": batch_type,
                "h2": current_h2,
                "h2_remaining": h2_remaining[:5],
                "target_words": batch_length_info.get("suggested_min", batch_length_info.get("target", "?")),
                "word_range": f"{batch_length_info.get('suggested_min', '?')}-{batch_length_info.get('suggested_max', '?')}",
                "must_keywords": [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in must_kw],
                "extended_keywords": [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in ext_kw],
                "stop_keywords": [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in stop_kw][:10],
                "caution_keywords": [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in caution_kw][:10],
                "coverage": pre_batch.get("coverage") or {},
                "density": pre_batch.get("density") or {},
                "has_gpt_instructions": bool(pre_batch.get("gpt_instructions_v39")),
                "has_gpt_prompt": bool(pre_batch.get("gpt_prompt")),
                "has_article_memory": bool(pre_batch.get("article_memory")),
                "has_enhanced": bool(enhanced_data),
                "has_style": bool(pre_batch.get("style_instructions")),
                "has_legal": bool((pre_batch.get("legal_context") or {}).get("active")),
                "has_medical": bool((pre_batch.get("medical_context") or {}).get("active")),
                "semantic_plan": {
                    "h2": (pre_batch.get("semantic_batch_plan") or {}).get("h2"),
                    "profile": (pre_batch.get("semantic_batch_plan") or {}).get("profile"),
                    "score": (pre_batch.get("semantic_batch_plan") or {}).get("score")
                },
                "entities_to_define": (enhanced_data.get("entities_to_define") or [])[:5],
                "experience_markers": bool(enhanced_data.get("experience_markers")),
                "continuation_context": bool(enhanced_data.get("continuation_context")),
                "paa_from_serp": (enhanced_data.get("paa_from_serp") or [])[:3],
                "main_keyword_ratio": (pre_batch.get("main_keyword") or {}).get("ratio"),
                "intro_guidance": pre_batch.get("intro_guidance", "") if batch_type == "INTRO" else "",
                # v45 flags
                "has_causal_context": bool(enhanced_data.get("causal_context")),
                "has_information_gain": bool(enhanced_data.get("information_gain")),
                "has_smart_instructions": bool(enhanced_data.get("smart_instructions")),
                "has_phrase_hierarchy": bool(enhanced_data.get("phrase_hierarchy")),
                "has_entity_salience": bool(entity_salience_instructions),
                "has_continuation_v39": bool(pre_batch.get("continuation_v39")),
                # v47.0 flags
                "has_backend_placement": bool(backend_placement_instruction),
                "has_cooccurrence": bool(backend_cooccurrence_pairs),
                "has_concepts": bool(must_cover_concepts or concept_instruction),
            })

            # 6c: Generate text
            has_instructions = bool(pre_batch.get("gpt_instructions_v39"))
            has_enhanced = bool(pre_batch.get("enhanced"))
            has_memory = bool(pre_batch.get("article_memory"))
            has_causal = bool(enhanced_data.get("causal_context"))
            has_smart = bool(enhanced_data.get("smart_instructions"))
            # v50.8 FIX 49: Determine effort/web_search for logging
            _is_ymyl = pre_batch.get("_is_ymyl", False)
            _ymyl_int = pre_batch.get("_ymyl_intensity", "none")
            _effort = "high" if _ymyl_int == "full" else ("medium" if _ymyl_int == "light" else None)
            _effort_label = _effort or f"temp={temperature or 0.7}"
            _web = _is_ymyl and _ymyl_int == "full"
            yield emit("log", {"msg": f"Generuję tekst przez {'🟢 ' + effective_openai_model if engine == 'openai' else '🟣 ' + ANTHROPIC_MODEL}... [effort={_effort_label} web={'✅' if _web else '—'} instr={'✅' if has_instructions else '❌'} enhanced={'✅' if has_enhanced else '❌'} memory={'✅' if has_memory else '❌'}]"})

            if batch_type == "FAQ":
                # FAQ batch: first analyze PAA
                paa_result = brajen_call("get", f"/api/project/{project_id}/paa/analyze")
                paa_data = paa_result["data"] if paa_result["ok"] else {}
                text = generate_faq_text(paa_data, pre_batch, engine=engine, openai_model=effective_openai_model, temperature=temperature)
            else:
                # ═══ AI MIDDLEWARE: Article memory fallback ═══
                article_memory = pre_batch.get("article_memory")
                if not article_memory and accepted_batches_log:
                    # Backend didn't provide memory, synthesize locally
                    if len(accepted_batches_log) >= 3:
                        article_memory = ai_synthesize_memory(accepted_batches_log, main_keyword)
                        yield emit("log", {"msg": f"🧠 AI Middleware: synteza pamięci artykułu ({len(accepted_batches_log)} batchy)"})
                    else:
                        article_memory = synthesize_article_memory(accepted_batches_log)
                        if article_memory.get("topics_covered"):
                            yield emit("log", {"msg": f"🧠 Lokalna pamięć: {len(article_memory.get('topics_covered', []))} tematów"})
                
                text = generate_batch_text(
                    pre_batch, current_h2, batch_type,
                    article_memory, engine=engine, openai_model=effective_openai_model,
                    temperature=temperature
                )

            word_count = len(text.split())
            yield emit("log", {"msg": f"Wygenerowano {word_count} słów"})

            # Post-process: strip duplicate ## headers (Claude sometimes outputs both h2: and ##)
            text = _clean_batch_text(text)

            # 6d-6g: Submit with retry logic
            # Max 4 attempts: original + 2 AI smart retries + 1 forced
            max_attempts = 4
            batch_accepted = False

            for attempt in range(max_attempts):
                forced = (attempt == max_attempts - 1)  # Last attempt is always forced
                submit_data = {"text": text}
                if forced:
                    submit_data["forced"] = True
                    yield emit("log", {"msg": "⚡ Forced mode ON, wymuszam zapis"})

                yield emit("log", {"msg": f"POST /batch_simple (próba {attempt + 1}/{max_attempts})"})
                submit_result = brajen_call("post", f"/api/project/{project_id}/batch_simple", submit_data)

                if not submit_result["ok"]:
                    yield emit("log", {"msg": f"❌ Submit error: {submit_result.get('error', '')[:100]}"})
                    break

                result = submit_result["data"]
                accepted = result.get("accepted", False)
                action = result.get("action", "CONTINUE")
                quality = (result.get("quality") or {})
                depth = result.get("depth_score")
                exceeded = (result.get("exceeded_keywords") or [])

                yield emit("batch_result", {
                    "batch": batch_num,
                    "accepted": accepted,
                    "action": action,
                    "quality_score": quality.get("score"),
                    "quality_grade": quality.get("grade"),
                    "depth_score": depth,
                    "exceeded": [e.get("keyword", "") for e in exceeded] if exceeded else [],
                    "word_count": len(text.split()) if text else 0,
                    "text_preview": text if accepted else ""
                })

                if accepted:
                    batch_accepted = True
                    yield emit("log", {"msg": f"✅ Batch {batch_num} accepted! Score: {quality.get('score')}/100"})
                    # Content integrity check
                    if text:
                        _ci = []; tl = text.lower()
                        if "mg/100 ml" in tl or "mg/100ml" in tl:
                            _ci.append("❌ JEDNOSTKI: 'mg/100 ml' → promile lub mg/dm³")
                        if "odpowiednich przepisów" in tl or "właściwych przepisów" in tl:
                            _ci.append("❌ PLACEHOLDER: wstaw konkretny artykuł (art. X k.k.)")
                        if ("do 2 lat" in tl or "2 lata więzienia" in tl) and "alkohol" in tl:
                            _ci.append("❌ KARA: 'do 2 lat' → art. 178a §1 = do 3 lat (2023)")
                        if "ciągu 2 lat" in tl and "recydyw" in tl:
                            _ci.append("❌ RECYDYWA: brak limitu czasowego w art. 178a §4")
                        for w in _ci:
                            yield emit("log", {"msg": w})
                    # ── Sentence length post-check ──
                    sl = check_sentence_length(text)
                    if sl["needs_retry"] and attempt < max_attempts - 1:
                        yield emit("log", {"msg": f"✂️ Zdania za długie (śr. {sl['avg_len']} słów, {sl['long_count']} ponad limit) — skracam..."})
                        text_shortened = sentence_length_retry(text, h2=current_h2, avg_len=sl["avg_len"], long_count=sl["long_count"])
                        sl_after = check_sentence_length(text_shortened)
                        if sl_after["avg_len"] < sl["avg_len"]:
                            text = text_shortened
                            yield emit("log", {"msg": f"✅ Po skróceniu: śr. {sl_after['avg_len']} słów/zdanie"})
                        else:
                            yield emit("log", {"msg": f"⚠️ Skracanie nie poprawiło wyniku, zostawiam oryginał"})
                    # ═══ DOMAIN VALIDATOR (Warstwa 2) ═══
                    _dv_category = "prawo" if is_legal else ("medycyna" if is_medical else ("finanse" if is_finance else ""))
                    if _dv_category and text:
                        _dv = validate_batch_domain(text, _dv_category, batch_num)
                        if not _dv.get("skipped"):
                            if not _dv.get("clean"):
                                _dv_errors = _dv.get("errors", [])
                                _dv_quick = _dv.get("quick_hits", [])
                                _dv_log = [e.get("found", e.get("type", "?")) for e in _dv_errors[:3]]
                                if _dv_quick:
                                    _dv_log = _dv_quick[:3]
                                yield emit("log", {"msg": f"🔴 DOMAIN VALIDATOR: {len(_dv_errors or _dv_quick)} błędów terminologicznych — naprawiam... ({', '.join(_dv_log)})"})
                                if attempt < max_attempts - 1:
                                    text = fix_batch_domain_errors(text, _dv, _dv_category, h2=current_h2)
                                    yield emit("log", {"msg": f"✅ Domain fix: tekst poprawiony ({len(text.split())} słów)"})
                                else:
                                    yield emit("log", {"msg": "⚠️ Domain errors — forced mode, pomijam auto-fix"})
                            else:
                                yield emit("log", {"msg": f"✅ Domain validator: czysto [{_dv_category}]"})

                    # Fix #44 + Fix #54: Keyword anti-stuffing with context-aware replacements
                    import re as _re_stuff
                    _main_kw_lower = main_keyword.lower()
                    _kw_count = len(_re_stuff.findall(_re_stuff.escape(_main_kw_lower), text.lower()))
                    _text_words = len(text.split())
                    if _kw_count > 3 and _text_words > 0:
                        _kw_density = _kw_count / _text_words * 100
                        if _kw_density > 2.5 or _kw_count > 4:
                            yield emit("log", {"msg": f"⚠️ Keyword stuffing: '{main_keyword}' x{_kw_count} ({_kw_density:.1f}%) — za dużo powtórzeń"})
                            if attempt < max_attempts - 1:
                                # Fix #54: Context-aware replacements instead of hardcoded legal terms
                                _destuff_replacements = [
                                    "to zagadnienie", "ten temat", "ta kwestia",
                                    "omawiany problem", "ten aspekt", "wspomniane zjawisko",
                                ]
                                _seen = 0
                                def _destuff(m):
                                    nonlocal _seen
                                    _seen += 1
                                    if _seen <= 2:
                                        return m.group(0)
                                    return _destuff_replacements[(_seen - 3) % len(_destuff_replacements)]
                                text = _re_stuff.sub(_re_stuff.escape(main_keyword), _destuff, text, flags=_re_stuff.IGNORECASE)
                                yield emit("log", {"msg": f"✅ Destuffed: zredukowano '{main_keyword}' z {_kw_count} do max 2 wystąpień"})

                    # Fix #56: Global cross-batch keyword anti-stuffing
                    _batch_kw_count = len(_re_stuff.findall(_re_stuff.escape(_main_kw_lower), text.lower()))
                    _remaining_budget = max(0, _GLOBAL_KW_MAX - _global_main_kw_count)
                    if _batch_kw_count > _remaining_budget:
                        # Reduce to fit within global budget
                        _g_seen = 0
                        def _global_destuff(m):
                            nonlocal _g_seen
                            _g_seen += 1
                            if _g_seen <= _remaining_budget:
                                return m.group(0)
                            _reps = ["to zagadnienie", "ten temat", "ta kwestia",
                                     "omawiany problem", "ten aspekt", "wspomniane zjawisko"]
                            return _reps[(_g_seen - _remaining_budget - 1) % len(_reps)]
                        text = _re_stuff.sub(_re_stuff.escape(main_keyword), _global_destuff, text, flags=_re_stuff.IGNORECASE)
                        _final_count = len(_re_stuff.findall(_re_stuff.escape(_main_kw_lower), text.lower()))
                        yield emit("log", {"msg": f"🌐 Global destuff: '{main_keyword}' budżet {_remaining_budget}, batch miał {_batch_kw_count} → {_final_count}"})
                        _global_main_kw_count += _final_count
                    else:
                        _global_main_kw_count += _batch_kw_count

                    # Fix #60: Track snippet elements (lists, tables, answer-first)
                    import re as _re_snippet
                    _batch_lists = len(_re_snippet.findall(r'<(?:ul|ol)\b', text, _re_snippet.IGNORECASE))
                    _batch_tables = len(_re_snippet.findall(r'<table\b', text, _re_snippet.IGNORECASE))
                    _global_list_count += _batch_lists
                    _global_table_count += _batch_tables
                    if _batch_lists or _batch_tables:
                        yield emit("log", {"msg": f"📋 Snippet elements: {_batch_lists} list(y), {_batch_tables} tabel(a) [global: {_global_list_count} list, {_global_table_count} table]"})

                    # Check answer-first: first paragraph under H2 should be 40-58 words
                    _first_p_match = _re_snippet.search(r'<p[^>]*>(.*?)</p>', text, _re_snippet.DOTALL)
                    if _first_p_match:
                        _first_p_text = _re_snippet.sub(r'<[^>]+>', '', _first_p_match.group(1)).strip()
                        _first_p_words = len(_first_p_text.split())
                        if _first_p_words < 35:
                            yield emit("log", {"msg": f"⚠️ Snippet: pierwszy akapit za krótki ({_first_p_words} słów, cel: 40-58)"})
                        elif _first_p_words > 65:
                            yield emit("log", {"msg": f"⚠️ Snippet: pierwszy akapit za długi ({_first_p_words} słów, cel: 40-58)"})

                    # Track for memory
                    accepted_batches_log.append({
                        "text": text, "h2": current_h2, "batch_num": batch_num,
                        "depth_score": depth
                    })
                    break

                # Not accepted, decide retry strategy
                if forced:
                    yield emit("log", {"msg": f"⚠️ Batch {batch_num} w forced mode, kontynuuję"})
                    accepted_batches_log.append({
                        "text": text, "h2": current_h2, "batch_num": batch_num,
                        "depth_score": depth
                    })
                    break

                # ═══ AI MIDDLEWARE: Smart retry ═══
                if exceeded and should_use_smart_retry(result, attempt + 1):
                    yield emit("log", {"msg": f"🤖 AI Smart Retry: Sonnet przepisuje tekst (zamiana {len(exceeded)} fraz)..."})
                    text = smart_retry_batch(
                        original_text=text,
                        exceeded_keywords=exceeded,
                        pre_batch=pre_batch,
                        h2=current_h2,
                        batch_type=batch_type,
                        attempt_num=attempt + 1
                    )
                    new_word_count = len(text.split())
                    yield emit("log", {"msg": f"🔄 Smart retry: {new_word_count} słów, próba {attempt + 2}/{max_attempts}"})
                    text = _clean_batch_text(text)
                else:
                    # Fallback: mechanical fix for non-exceeded issues
                    fixes_applied = 0
                    if exceeded:
                        for exc in exceeded:
                            kw = exc.get("keyword", "")
                            synonyms = (exc.get("use_instead") or exc.get("synonyms") or [])
                            if synonyms and kw and kw in text:
                                syn = synonyms[0] if isinstance(synonyms[0], str) else str(synonyms[0])
                                text = text.replace(kw, syn, 1)
                                fixes_applied += 1
                                yield emit("log", {"msg": f"🔧 Zamiana: '{kw}' → '{syn}'"})
                    yield emit("log", {"msg": f"🔄 Retry: naprawiono {fixes_applied} fraz, próba {attempt + 2}/{max_attempts}"})

            # Save FAQ if applicable
            if batch_type == "FAQ" and batch_accepted:
                yield emit("log", {"msg": "Zapisuję FAQ/PAA (Schema.org)..."})
                questions = []
                lines = text.split("\n")
                current_q, current_a = None, []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("h3:") or stripped.startswith("### "):
                        if current_q and current_a:
                            questions.append({"question": current_q, "answer": " ".join(current_a)})
                        current_q = stripped.replace("h3:", "").replace("###", "").strip()
                        current_a = []
                    elif current_q and stripped:
                        current_a.append(stripped)
                if current_q and current_a:
                    questions.append({"question": current_q, "answer": " ".join(current_a)})
                if questions:
                    brajen_call("post", f"/api/project/{project_id}/paa/save", {"questions": questions})

            yield emit("step", {"step": 6, "name": "Batch Loop", "status": "running",
                                "detail": f"{batch_num}/{total_batches} batchy"})

        step_done(6)
        yield emit("step", {"step": 6, "name": "Batch Loop", "status": "done",
                            "detail": f"{total_batches}/{total_batches} batchy"})

        # Emit article memory state for dashboard
        if article_memory:
            mem = article_memory if isinstance(article_memory, dict) else {}
            yield emit("article_memory", {
                "topics_covered": mem.get("topics_covered", [])[:20],
                "open_threads": mem.get("open_threads", [])[:10],
                "entities_introduced": mem.get("entities_introduced", [])[:15],
                "defined_terms": mem.get("defined_terms", [])[:15],
                "thesis": mem.get("thesis", ""),
                "tone": mem.get("tone", ""),
                "batch_count": len(accepted_batches_log),
            })

        # Fix #60: Post-loop snippet element check
        _snippet_total = _global_list_count + _global_table_count
        if _snippet_total < 2:
            yield emit("log", {"msg": f"⚠️ SNIPPET WARNING: artykuł ma tylko {_global_list_count} list i {_global_table_count} tabel (wymóg: min 2 elementy łącznie). Dodaj listy <ul>/<ol> w kolejnych sekcjach."})
        else:
            yield emit("log", {"msg": f"✅ Snippet elements OK: {_global_list_count} list, {_global_table_count} tabel"})

        # ─── KROK 7: PAA Check ───
        step_start(7)
        yield emit("step", {"step": 7, "name": "PAA Analyze & Save", "status": "running"})
        try:
            paa_check = brajen_call("get", f"/api/project/{project_id}/paa")
            paa_data_check = paa_check.get("data") if paa_check.get("ok") else None
            paa_has_section = isinstance(paa_data_check, dict) and paa_data_check.get("paa_section")
            if not paa_has_section:
                yield emit("log", {"msg": "Brak FAQ, analizuję PAA i generuję..."})
                paa_analyze = brajen_call("get", f"/api/project/{project_id}/paa/analyze")
                if paa_analyze["ok"] and paa_analyze.get("data"):
                    # Fetch pre_batch for FAQ context (stop keywords, style, memory)
                    faq_pre = brajen_call("get", f"/api/project/{project_id}/pre_batch_info", timeout=HEAVY_REQUEST_TIMEOUT)
                    faq_pre_batch = faq_pre["data"] if faq_pre.get("ok") and isinstance(faq_pre.get("data"), dict) else None
                    paa_data_for_faq = paa_analyze["data"] if isinstance(paa_analyze.get("data"), dict) else {}
                    # v50.7 FIX 37: Ensure faq_pre_batch is a dict (some endpoints return str)
                    if faq_pre_batch and not isinstance(faq_pre_batch, dict):
                        logger.warning(f"[FAQ] pre_batch is {type(faq_pre_batch).__name__}, forcing None")
                        faq_pre_batch = None
                    try:
                        faq_text = generate_faq_text(paa_data_for_faq, faq_pre_batch, engine=engine, openai_model=effective_openai_model, temperature=temperature)
                    except AttributeError as ae:
                        logger.warning(f"[FAQ] generate_faq_text AttributeError: {ae}, retrying without pre_batch")
                        faq_text = generate_faq_text(paa_data_for_faq, None, engine=engine, openai_model=effective_openai_model, temperature=temperature)
                    if faq_text and faq_text.strip():
                        brajen_call("post", f"/api/project/{project_id}/batch_simple", {"text": faq_text})
                        # Extract and save
                        questions = []
                        lines = faq_text.split("\n")
                        cq, ca = None, []
                        for line in lines:
                            s = line.strip()
                            if s.startswith("h3:") or s.startswith("### "):
                                if cq and ca:
                                    questions.append({"question": cq, "answer": " ".join(ca)})
                                cq = s.replace("h3:", "").replace("###", "").strip()
                                ca = []
                            elif cq and s:
                                ca.append(s)
                        if cq and ca:
                            questions.append({"question": cq, "answer": " ".join(ca)})
                        if questions:
                            brajen_call("post", f"/api/project/{project_id}/paa/save", {"questions": questions})

                        # Emit PAA data for dashboard
                        paa_from_serp = (s1.get("paa") or s1.get("paa_questions") or [])
                        yield emit("paa_data", {
                            "questions_generated": len(questions) if questions else 0,
                            "faq_text_length": len(faq_text) if faq_text else 0,
                            "paa_questions_from_serp": len(paa_from_serp),
                            "paa_unanswered": len(({} if not isinstance(s1.get("content_gaps"), dict) else s1.get("content_gaps")).get("paa_unanswered", [])),
                            "status": "generated",
                        })
                    else:
                        yield emit("log", {"msg": "⚠️ Brak danych PAA, pomijam FAQ"})
                else:
                    yield emit("log", {"msg": "⚠️ PAA analyze pusty, pomijam FAQ"})
                step_done(7)
                yield emit("step", {"step": 7, "name": "PAA Analyze & Save", "status": "done"})
            else:
                yield emit("step", {"step": 7, "name": "PAA Analyze & Save", "status": "done",
                                    "detail": "FAQ już zapisane"})
        except Exception as faq_err:
            logger.warning(f"FAQ generation error (non-fatal): {faq_err}")
            yield emit("log", {"msg": f"⚠️ FAQ error: {str(faq_err)[:80]}, pomijam, kontynuuję"})
            yield emit("step", {"step": 7, "name": "PAA Analyze & Save", "status": "warning",
                                "detail": "Błąd FAQ, pominięto"})

        # ─── KROK 8: Content Editorial (merytoryczny) ───
        step_start(8)
        yield emit("step", {"step": 8, "name": "Content Editorial", "status": "running"})
        yield emit("log", {"msg": "POST /content_editorial..."})
        content_editorial_result = brajen_call("post", f"/api/project/{project_id}/content_editorial", timeout=HEAVY_REQUEST_TIMEOUT)
        if content_editorial_result["ok"]:
            ced = content_editorial_result["data"]
            ced_status = ced.get("status", "OK")
            ced_score = ced.get("score", 100)
            ced_critical = ced.get("critical_count", 0)
            ced_warnings = ced.get("warning_count", 0)
            detail = f"Status: {ced_status} | Score: {ced_score}/100 | Krytyczne: {ced_critical} | Ostrzeżenia: {ced_warnings}"
            yield emit("content_editorial", {
                "status": ced_status,
                "score": ced_score,
                "critical_count": ced_critical,
                "warning_count": ced_warnings,
                "issues": ced.get("issues", [])[:5],
                "summary": ced.get("summary", ""),
                "blocked": ced.get("blocked", False),
            })
            if ced.get("blocked"):
                yield emit("log", {"msg": f"⚠️ Content Editorial: BLOCKED — {ced.get('blocked_reason', '')}. Artykuł wymaga poprawy merytorycznej."})
                yield emit("step", {"step": 8, "name": "Content Editorial", "status": "warning", "detail": f"BLOCKED: {ced.get('blocked_reason', '')[:80]}"})
            else:
                step_done(8)
                yield emit("step", {"step": 8, "name": "Content Editorial", "status": "done", "detail": detail})
        else:
            yield emit("step", {"step": 8, "name": "Content Editorial", "status": "warning", "detail": "Nie udało się, kontynuuję"})

        # ─── KROK 9: Final Review ───
        step_start(9)
        yield emit("step", {"step": 9, "name": "Final Review", "status": "running"})
        yield emit("log", {"msg": "GET /final_review..."})
        final_result = brajen_call("get", f"/api/project/{project_id}/final_review", timeout=HEAVY_REQUEST_TIMEOUT)
        if final_result["ok"]:
            final = final_result["data"]
            # Unwrap cached response format (GET returns {"status":"EXISTS","final_review":{...}})
            if final.get("status") == "EXISTS" and "final_review" in final:
                final = final["final_review"]
            final_score = final.get("quality_score", final.get("score", "?"))
            final_status = final.get("status", "?")
            
            # v51 FIX: Read structured data from correct paths
            validations = final.get("validations") or {}
            kw_validation = validations.get("missing_keywords") or {}
            
            # Build proper missing/overuse lists from structured data
            actual_missing = []
            for kw in (kw_validation.get("priority_to_add", {}).get("to_add_by_claude", []) or [])[:5]:
                actual_missing.append(f"Wpleć '{kw.get('keyword', '')}' min. {kw.get('target_min', 1)}x")
            
            # Overuse warnings (separate from missing)
            overuse_warnings = []
            for kw in (kw_validation.get("within_tolerance", []) or [])[:3]:
                excess = kw.get("actual", 0) - kw.get("target_max", 0)
                overuse_warnings.append(f"🟡 Rozważ usunięcie {excess}x '{kw.get('keyword', '')}' ({kw.get('actual', 0)}/{kw.get('target_max', 0)})")
            for kw in (kw_validation.get("stuffing", []) or [])[:3]:
                excess = kw.get("actual", 0) - kw.get("target_max", 0)
                overuse_warnings.append(f"🔴 USUŃ {excess}x '{kw.get('keyword', '')}' ({kw.get('actual', 0)}/{kw.get('target_max', 0)})")
            
            # H3 length issues
            h3_issues = []
            for issue in (validations.get("h3_length", {}).get("issues", []) or [])[:3]:
                h3_issues.append(f"Rozbuduj H3 '{issue.get('h3', '')}' o {issue.get('deficit', 0)} słów")
            
            # Combined recommendations from API (fallback)
            all_recommendations = final.get("recommendations") or []
            
            # What we show in "Brakujące" = only actual missing keywords
            missing_kw = actual_missing
            # Issues = overuse + H3 + other issues from API
            issues = (final.get("issues") or final.get("all_issues") or [])

            yield emit("final_review", {
                "score": final_score,
                "status": final_status,
                # v51: Separate missing vs overuse vs H3
                "missing_keywords_count": len(missing_kw),
                "missing_keywords": missing_kw[:10],
                "overuse_warnings": overuse_warnings[:5],
                "h3_issues": h3_issues[:5],
                "issues_count": len(issues) if isinstance(issues, list) else 0,
                "issues": issues[:5] if isinstance(issues, list) else [],
                # v51: Full recommendations from API
                "recommendations": all_recommendations[:10],
                "recommendations_count": len(all_recommendations),
                # v50.7: Add issues_summary for dashboard
                "issues_summary": final.get("issues_summary") or {},
                # v50.7: Stuffing info
                "stuffing": (final.get("validations") or {}).get("missing_keywords", {}).get("stuffing", [])[:5],
                "priority_to_add": (final.get("validations") or {}).get("missing_keywords", {}).get("priority_to_add", {}).get("to_add_by_claude", [])[:5],
                # P5: Quality breakdown
                "quality_breakdown": {
                    "keywords": final.get("keyword_score", final.get("keywords_score")),
                    "humanness": final.get("humanness_score", final.get("ai_score")),
                    "grammar": final.get("grammar_score"),
                    "structure": final.get("structure_score"),
                    "semantic": final.get("semantic_score"),
                    "depth": final.get("depth_score"),
                    "coherence": final.get("coherence_score"),
                },
                "density": final.get("density") or final.get("keyword_density"),
                "word_count": final.get("word_count") or final.get("total_words"),
                "basic_coverage": final.get("basic_coverage"),
                "extended_coverage": final.get("extended_coverage"),
                # v50.7: Entity scoring
                "entity_scoring": final.get("entity_scoring") or {},
            })

            step_done(8)
            yield emit("step", {"step": 8, "name": "Final Review", "status": "done",
                                "detail": f"Score: {final_score}/100 | Status: {final_status}"})

            # YMYL validation
            ymyl_validation = {"legal": None, "medical": None}
            if is_legal:
                yield emit("log", {"msg": "Walidacja prawna..."})
                full_art = brajen_call("get", f"/api/project/{project_id}/full_article")
                if full_art["ok"] and full_art["data"].get("full_article"):
                    legal_val = brajen_call("post", "/api/legal/validate",
                               {"full_text": full_art["data"]["full_article"]})
                    if legal_val["ok"]:
                        ymyl_validation["legal"] = legal_val.get("data") or {}
                        yield emit("log", {"msg": f"⚖️ Legal validation: {(legal_val.get('data') or {}).get('status', 'done')}"})
            if is_medical:
                yield emit("log", {"msg": "Walidacja medyczna..."})
                full_art = brajen_call("get", f"/api/project/{project_id}/full_article")
                if full_art["ok"] and full_art["data"].get("full_article"):
                    med_val = brajen_call("post", "/api/medical/validate",
                               {"full_text": full_art["data"]["full_article"]})
                    if med_val["ok"]:
                        ymyl_validation["medical"] = med_val.get("data") or {}
                        yield emit("log", {"msg": f"🏥 Medical validation: {(med_val.get('data') or {}).get('status', 'done')}"})
            if ymyl_validation["legal"] or ymyl_validation["medical"]:
                yield emit("ymyl_validation", ymyl_validation)
        else:
            yield emit("step", {"step": 8, "name": "Final Review", "status": "warning",
                                "detail": "Nie udało się, kontynuuję"})

        # ─── CITATION PASS (YMYL only) ───
        # Fix #38: Citation pass — TYLKO orzeczenia i ISAP, BEZ Wikipedii
        # Wikipedia służy do opisu encji, nie jako źródło cytowań
        if is_legal and judgments_clean:
            yield emit("log", {"msg": "📎 Citation pass — dopasowuję orzeczenia do tekstu..."})
            try:
                _cit_art = brajen_call("get", f"/api/project/{project_id}/full_article")
                if _cit_art["ok"] and _cit_art["data"].get("full_article"):
                    _art_text = _cit_art["data"]["full_article"]
                    _cit_sources = []
                    for j in judgments_clean[:5]:
                        sig = j.get("signature", "")
                        if not sig: continue
                        _cit_sources.append(
                            "ORZECZENIE [" + (j.get("matched_article") or "prawo karne") + "]: "
                            + sig + ", " + j.get("court","") + " (" + j.get("date","") + ")"
                            + (" — " + j.get("summary","")[:100] if j.get("summary") else "")
                        )
                    # Fix #38: NIE dodajemy Wikipedii jako źródła cytatów
                    # Wikipedia jest używana tylko do wzbogacania opisu encji w prompt_builder
                    if _cit_sources:
                        _cit_sys = (
                            "Jesteś redaktorem prawnym. Wstaw cytaty do artykułu TYLKO tam gdzie "
                            "akapit merytorycznie pokrywa się z danym orzeczeniem lub przepisem.\n\n"
                            "ZASADY:\n"
                            "1. Cytuj orzeczenie TYLKO gdy akapit dotyczy dokładnie tego zagadnienia\n"
                            "2. NIE cytuj Wikipedii — nigdy nie pisz 'zob. Wikipedia', 'Wikipedia podaje' itp.\n"
                            "3. NIE zmieniaj treści — tylko dopisz cytat w nawiasie na końcu zdania\n"
                            "4. Jeśli akapit nie pasuje — zostaw bez zmian\n"
                            "5. Zwróć TYLKO artykuł, bez komentarzy\n"
                            "6. Orzeczenia karne (II K, AKa) → tylko akapity o sankcjach karnych\n"
                            "7. Dopuszczalne źródła cytatów: orzeczenia sądowe (sygn. akt), ISAP, Dz.U."
                        )
                        _sep = chr(10)
                        _cit_usr = ("ARTYKUL:" + _sep + _art_text + _sep + _sep + "---" + _sep + "DOSTEPNE CYTATY:" + _sep + _sep.join(_cit_sources) + _sep + _sep + "Zwroc artykul z wstawionymi cytatami.")
                        _cit_res = _generate_claude(_cit_sys, _cit_usr, effort="low", web_search=False, temperature=0.1)
                        if _cit_res and len(_cit_res) > len(_art_text) * 0.8:
                            yield emit("article_citation_pass", {"text": _cit_res, "sources_count": len(_cit_sources)})
                            yield emit("log", {"msg": f"✅ Citation pass: {len(_cit_sources)} źródeł wstawionych"})
                        else:
                            yield emit("log", {"msg": "⚠️ Citation pass: wynik zbyt krótki, pomijam"})
            except Exception as _ce:
                yield emit("log", {"msg": f"⚠️ Citation pass błąd: {str(_ce)[:80]}"})

        # ─── KROK 10: Editorial Review ───
        step_start(10)
        yield emit("step", {"step": 10, "name": "Editorial Review", "status": "running"})
        yield emit("log", {"msg": "POST /editorial_review, to może chwilę potrwać..."})

        editorial_result = {"ok": False}  # v50.7: safety init for FIX 41
        editorial_result = brajen_call("post", f"/api/project/{project_id}/editorial_review", timeout=HEAVY_REQUEST_TIMEOUT)
        if editorial_result["ok"]:
            ed = editorial_result["data"]
            score = ed.get("overall_score", "?")
            diff = (ed.get("diff_result") or {})
            rollback = (ed.get("rollback") or {})
            word_guard = (ed.get("word_count_guard") or {})

            detail = f"Ocena: {score}/10 | Zmiany: {diff.get('applied', 0)}/{diff.get('total_changes_parsed', 0)}"
            if word_guard:
                detail += f" | Słowa: {word_guard.get('original', '?')}→{word_guard.get('corrected', '?')}"

            # Fix #37: Pełne dane editorial dla UI
            _ef = ed.get("editorial_feedback") or {}
            _summary = (
                _ef.get("summary")
                or _ef.get("recenzja_ogolna")
                or ed.get("summary")
                or ""
            )
            # Zbierz WSZYSTKIE styl_i_jezyk + merytoryka + struktura do errors_found
            _errors_found = []
            for err in (_ef.get("errors_to_fix") or ed.get("errors_to_fix") or [])[:10]:
                _errors_found.append(err)
            for item in (_ef.get("styl_i_jezyk") or [])[:5]:
                _errors_found.append({"type": item.get("problem", "STYL"), "description": item.get("sugestia", ""), "original": item.get("cytat", "")})
            for item in (_ef.get("merytoryka") or [])[:5]:
                _errors_found.append({"type": "MERYTORYKA", "description": item.get("uwaga", ""), "original": item.get("cytat", ""), "sekcja": item.get("sekcja", "")})
            for item in (_ef.get("halucynacje") or [])[:3]:
                _errors_found.append({"type": "HALUCYNACJA", "description": item.get("dlaczego_falsz", ""), "original": item.get("cytat", "")})

            yield emit("editorial", {
                "score": score,
                "changes_applied": diff.get("applied", 0),
                "changes_failed": diff.get("failed", 0),
                "word_count_before": word_guard.get("original"),
                "word_count_after": word_guard.get("corrected"),
                "rollback": rollback.get("triggered", False),
                "rollback_reason": rollback.get("reason", ""),
                "feedback": _ef,
                # v50.7 FIX 41: Add change details for expanded panel
                "applied_changes": (diff.get("applied_changes") or ed.get("applied_changes") or [])[:15],
                "failed_changes": (diff.get("failed_changes") or ed.get("failed_changes") or [])[:10],
                "summary": _summary,
                "errors_found": _errors_found[:15],
                "grammar_fixes": (ed.get("grammar_correction") or {}).get("fixes", 0),
                "grammar_removed": (ed.get("grammar_correction") or {}).get("removed", [])[:5],
                # Fix #37: Dodatkowe dane recenzji
                "scores": ed.get("scores") or {},
                "luki_tresciowe": (_ef.get("luki_tresciowe") or [])[:5],
                "brakujace_encje": (_ef.get("brakujace_encje") or [])[:8],
                "struktura_i_narracja": (_ef.get("struktura_i_narracja") or [])[:5],
            })

            if rollback.get("triggered"):
                yield emit("log", {"msg": f"⚠️ ROLLBACK: {rollback.get('reason', 'unknown')}"})

            # v50.7 FIX 41: Re-emit corrected article to update preview
            if not rollback.get("triggered"):
                corrected_text = ed.get("corrected_article", "")
                if corrected_text and len(corrected_text.strip()) > 50:
                    corrected_wc = len(corrected_text.split())
                    yield emit("article", {
                        "text": corrected_text,
                        "word_count": corrected_wc,
                        "source": "editorial_review",
                    })
                    yield emit("log", {"msg": f"📝 Podgląd zaktualizowany po editorial ({corrected_wc} słów)"})
            step_done(10)  # Fix #35: było step_done(9) — literówka
            yield emit("step", {"step": 10, "name": "Editorial Review", "status": "done", "detail": detail})
        else:
            ed_error = editorial_result.get("error", "unknown")
            ed_status = editorial_result.get("status", "?")
            yield emit("log", {"msg": f"⚠️ Editorial Review → {ed_status}: {ed_error[:200]}"})
            # Fix #35: Emit editorial event even on failure so UI shows the card with error info
            yield emit("editorial", {
                "score": 0,
                "changes_applied": 0,
                "changes_failed": 0,
                "word_count_before": None,
                "word_count_after": None,
                "rollback": False,
                "rollback_reason": "",
                "feedback": {},
                "applied_changes": [],
                "failed_changes": [],
                "summary": f"Editorial Review nie powiódł się: {ed_error[:150]}",
                "errors_found": [],
                "grammar_fixes": 0,
                "grammar_removed": [],
                "_failed": True,
                "_error": ed_error[:200],
                "_status_code": ed_status,
            })
            yield emit("step", {"step": 10, "name": "Editorial Review", "status": "warning",
                                "detail": f"Nie udało się ({ed_status}), artykuł bez recenzji"})

        # ─── KROK 11: Export ───
        step_start(11)
        yield emit("step", {"step": 11, "name": "Export", "status": "running"})

        # Get full article
        full_result = brajen_call("get", f"/api/project/{project_id}/full_article", timeout=HEAVY_REQUEST_TIMEOUT)
        if full_result["ok"]:
            full = full_result["data"]
            stats = (full.get("stats") or {})
            coverage = (full.get("coverage") or {})

            # v50.7 FIX 41: Use editorial corrected article if available
            article_text = full.get("full_article", "")
            if editorial_result and editorial_result.get("ok"):
                ed_corrected = (editorial_result.get("data") or {}).get("corrected_article", "")
                if ed_corrected and len(ed_corrected.strip()) > 50:
                    ed_rollback = ((editorial_result.get("data") or {}).get("rollback") or {}).get("triggered", False)
                    if not ed_rollback:
                        article_text = ed_corrected
                        yield emit("log", {"msg": f"📝 Export: użyto tekst po editorial review ({len(ed_corrected.split())} słów)"})

            yield emit("article", {
                "text": article_text,
                "word_count": len(article_text.split()) if article_text else 0,
                "h2_count": stats.get("h2_count", 0),
                "h3_count": stats.get("h3_count", 0),
                "coverage": coverage,
                "density": (full.get("density") or {})
            })

            # ═══ ENTITY SALIENCE: Google NLP API validation ═══
            full_text = article_text
            salience_result = {}
            nlp_entities = []
            subject_pos = {}
            
            # Subject position analysis: always runs (free, no API)
            if full_text:
                try:
                    subject_pos = analyze_subject_position(full_text, main_keyword)
                    sp_score = subject_pos.get("score", 0)
                    sr = subject_pos.get("subject_ratio", 0)
                    yield emit("log", {"msg": (
                        f"📐 Subject Position: score {sp_score}/100 | "
                        f"podmiot: {subject_pos.get('subject_position', 0)}/{subject_pos.get('sentences_with_entity', 0)} zdań ({sr:.0%}) | "
                        f"H2: {subject_pos.get('h2_entity_count', 0)} | "
                        f"1. zdanie: {'✅' if subject_pos.get('first_sentence_has_entity') else '❌'}"
                    )})
                except Exception as sp_err:
                    logger.warning(f"Subject position analysis failed: {sp_err}")

            # ═══ ANTI-FRANKENSTEIN: Style consistency analysis (free, always runs) ═══
            style_metrics = {}
            if full_text:
                try:
                    style_metrics = analyze_style_consistency(full_text)
                    st_score = style_metrics.get("score", 0)
                    yield emit("log", {"msg": (
                        f"🎭 Anti-Frankenstein: score {st_score}/100 | "
                        f"CV zdań: {style_metrics.get('cv_sentences', 0):.2f} | "
                        f"passive: {style_metrics.get('passive_ratio', 0):.0%} | "
                        f"śr. zdanie: {style_metrics.get('avg_sentence_length', 0):.0f} słów"
                    )})
                    yield emit("style_analysis", {
                        "score": st_score,
                        "sentence_count": style_metrics.get("sentence_count", 0),
                        "paragraph_count": style_metrics.get("paragraph_count", 0),
                        "avg_sentence_length": style_metrics.get("avg_sentence_length", 0),
                        "cv_sentences": style_metrics.get("cv_sentences", 0),
                        "avg_paragraph_length": style_metrics.get("avg_paragraph_length", 0),
                        "cv_paragraphs": style_metrics.get("cv_paragraphs", 0),
                        "passive_ratio": style_metrics.get("passive_ratio", 0),
                        "transition_ratio": style_metrics.get("transition_ratio", 0),
                        "repetition_ratio": style_metrics.get("repetition_ratio", 0),
                        "issues": style_metrics.get("issues", []),
                    })
                except Exception as style_err:
                    logger.warning(f"Style analysis failed: {style_err}")

            # ═══ FIX #61: COHERENCE / TOPIC DRIFT DETECTION ═══
            coherence_result = {}
            if full_text:
                try:
                    _master_url = os.environ.get("MASTER_SEO_API_URL", "http://localhost:5001")
                    _master_key = os.environ.get("MASTER_SEO_API_KEY", "")
                    _coh_headers = {}
                    if _master_key:
                        _coh_headers["Authorization"] = f"Bearer {_master_key}"

                    yield emit("log", {"msg": "🔗 Coherence: analiza spójności sekcji H2..."})
                    _coh_resp = http_requests.post(
                        f"{_master_url}/api/coherence",
                        json={"text": full_text, "drift_threshold": 0.6},
                        headers=_coh_headers,
                        timeout=30
                    )
                    if _coh_resp.status_code == 200:
                        coherence_result = _coh_resp.json()
                        coh_score = coherence_result.get("score", 0)
                        coh_avg = coherence_result.get("avg_coherence", 0)
                        coh_min = coherence_result.get("min_coherence", 0)
                        drift_count = coherence_result.get("drift_count", 0)
                        sections = coherence_result.get("section_count", 0)

                        yield emit("log", {"msg": (
                            f"🔗 Coherence: score {coh_score}/100 | "
                            f"avg: {coh_avg:.3f} | min: {coh_min:.3f} | "
                            f"sekcje: {sections} | drifts: {drift_count}"
                        )})

                        # Log drift alerts
                        for alert in coherence_result.get("drift_alerts", [])[:3]:
                            yield emit("log", {"msg": (
                                f"   ⚠️ Topic drift: \"{alert['from']}\" → \"{alert['to']}\" "
                                f"(similarity: {alert['similarity']:.3f} < 0.6)"
                            )})

                        yield emit("coherence", {
                            "enabled": True,
                            "score": coh_score,
                            "avg_coherence": coh_avg,
                            "min_coherence": coh_min,
                            "section_count": sections,
                            "drift_count": drift_count,
                            "drift_alerts": coherence_result.get("drift_alerts", []),
                            "pairwise_scores": coherence_result.get("pairwise_scores", []),
                            "global_coherence": coherence_result.get("global_coherence", 0),
                            "sections": coherence_result.get("sections", []),
                        })
                    else:
                        yield emit("log", {"msg": f"⚠️ Coherence: master API returned {_coh_resp.status_code}"})
                except Exception as coh_err:
                    logger.warning(f"Coherence analysis failed: {coh_err}")
                    yield emit("log", {"msg": f"⚠️ Coherence error: {str(coh_err)[:80]}"})

            # ═══ POLISH NLP VALIDATOR: NKJP corpus norms check (free, always runs) ═══
            polish_nlp = {}
            if full_text and POLISH_NLP_AVAILABLE:
                try:
                    polish_nlp = validate_polish_text(full_text)
                    pn_score = polish_nlp.get("score", 0)
                    m = polish_nlp.get("metrics", {})
                    yield emit("log", {"msg": (
                        f"🇵🇱 Polish NLP: score {pn_score}/100 | "
                        f"śr. wyraz: {m.get('avg_word_length', 0):.1f} zn | "
                        f"śr. zdanie: {m.get('avg_sentence_length', 0):.0f} słów | "
                        f"diakrytyki: {m.get('diacritics_pct', 0):.1f}% | "
                        f"FOG-PL: {m.get('fog_pl', 0):.0f} | "
                        f"przecinki: {m.get('comma_conjunction_ratio', 0):.0%}"
                    )})
                    # Log issues
                    issues = polish_nlp.get("issues", [])
                    if issues:
                        yield emit("log", {"msg": f"   ⚠️ Issues: {' | '.join(issues[:3])}"})
                    # Log collocation errors
                    coll_issues = polish_nlp.get("collocation_issues", [])
                    if coll_issues:
                        for ci in coll_issues[:3]:
                            yield emit("log", {"msg": f"   📝 Kolokacja: \"{ci['wrong']}\" → \"{ci['correct']}\" ({ci['count']}×)"})
                    # Emit to dashboard
                    yield emit("polish_nlp", {
                        "score": pn_score,
                        "avg_word_length": m.get("avg_word_length", 0),
                        "avg_sentence_length": m.get("avg_sentence_length", 0),
                        "diacritics_pct": m.get("diacritics_pct", 0),
                        "vowel_pct": m.get("vowel_pct", 0),
                        "fog_pl": m.get("fog_pl", 0),
                        "comma_conjunction_ratio": m.get("comma_conjunction_ratio", 0),
                        "sentence_cv": m.get("sentence_length_cv", 0),
                        "collocation_errors": m.get("collocation_errors", 0),
                        "hapax_ratio": m.get("hapax_ratio", 0),
                        "type_token_ratio": m.get("type_token_ratio", 0),
                        "issues": issues,
                        "recommendations": polish_nlp.get("recommendations", []),
                    })
                except Exception as pnlp_err:
                    logger.warning(f"Polish NLP validation failed: {pnlp_err}")

            # ═══ LANGUAGETOOL: Corpus-based grammar/collocation/punctuation check ═══
            lt_result = {}
            if full_text and LANGUAGETOOL_AVAILABLE:
                try:
                    lt_result = lt_check_text(full_text)
                    lt_score = lt_result.get("score", 0)
                    cats = lt_result.get("categories", {})
                    available = lt_result.get("api_available", False)
                    if available:
                        yield emit("log", {"msg": (
                            f"🔍 LanguageTool: score {lt_score}/100 | "
                            f"gramatyka: {cats.get('GRAMMAR', 0)} | "
                            f"kolokacje: {cats.get('COLLOCATIONS', 0)} | "
                            f"interpunkcja: {cats.get('PUNCTUATION', 0)} | "
                            f"styl: {cats.get('STYLE', 0) + cats.get('REDUNDANCY', 0)} | "
                            f"literówki: {cats.get('TYPOS', 0)}"
                        )})
                        # Log top issues
                        for issue in lt_result.get("issues", [])[:5]:
                            yield emit("log", {"msg": (
                                f"   📝 [{issue['category_name']}] {issue['message'][:80]}"
                                + (f" → {', '.join(issue['replacements'][:2])}" if issue.get('replacements') else "")
                            )})
                        # Emit to dashboard
                        yield emit("languagetool", {
                            "score": lt_score,
                            "total_issues": lt_result.get("total_issues", 0),
                            "categories": cats,
                            "collocation_issues": lt_result.get("collocation_issues", []),
                            "grammar_issues": lt_result.get("grammar_issues", []),
                            "punctuation_issues": lt_result.get("punctuation_issues", []),
                            "style_issues": lt_result.get("style_issues", []),
                        })
                    else:
                        yield emit("log", {"msg": "⚠️ LanguageTool API niedostępne, pominięto sprawdzanie"})
                except Exception as lt_err:
                    logger.warning(f"LanguageTool check failed: {lt_err}")

            # ═══ YMYL INTELLIGENCE: Analyze legal/medical references in text ═══
            if full_text and (is_legal or is_medical):
                try:
                    ymyl_refs = analyze_ymyl_references(full_text, legal_context, medical_context)
                    
                    if is_legal:
                        lr = ymyl_refs.get("legal", {})
                        yield emit("log", {"msg": (
                            f"⚖️ YMYL Legal: score {lr.get('score', 0)}/100 | "
                            f"akty: {len(lr.get('acts_found', []))} | "
                            f"orzeczenia: {len(lr.get('judgments_found', []))} | "
                            f"art.: {len(lr.get('articles_cited', []))} | "
                            f"disclaimer: {'✅' if lr.get('disclaimer_present') else '❌'}"
                        )})
                    
                    if is_medical:
                        mr = ymyl_refs.get("medical", {})
                        yield emit("log", {"msg": (
                            f"🏥 YMYL Medical: score {mr.get('score', 0)}/100 | "
                            f"PMID: {len(mr.get('pmids_found', []))} | "
                            f"badania: {len(mr.get('studies_referenced', []))} | "
                            f"instytucje: {len(mr.get('institutions_found', []))} | "
                            f"disclaimer: {'✅' if mr.get('disclaimer_present') else '❌'}"
                        )})
                    
                    yield emit("ymyl_analysis", ymyl_refs)
                except Exception as ymyl_err:
                    logger.warning(f"YMYL analysis failed: {ymyl_err}")
            
            if full_text and is_salience_available():
                yield emit("log", {"msg": "🔬 Entity Salience: analiza artykułu przez Google NLP API..."})
                try:
                    salience_result = check_entity_salience(full_text, main_keyword)
                    nlp_entities = salience_result.get("entities", [])
                    
                    main_sal = salience_result.get("main_salience", 0)
                    is_dominant = salience_result.get("is_main_dominant", False)
                    sal_score = salience_result.get("score", 0)
                    top_ent = salience_result.get("top_entity") or {}
                    
                    top_name = top_ent.get("name", "?")
                    top_sal = top_ent.get("salience", 0)
                    dom_str = "DOMINUJE" if is_dominant else f"Dominuje: {top_name} ({top_sal:.2f})"
                    yield emit("log", {"msg": f"Salience: {main_keyword} = {main_sal:.2f} | {dom_str} | Score: {sal_score}/100"})
                    
                    yield emit("entity_salience", {
                        "enabled": True,
                        "score": sal_score,
                        "main_keyword": main_keyword,
                        "main_salience": round(main_sal, 4),
                        "is_dominant": is_dominant,
                        "top_entity": {
                            "name": top_ent.get("name", ""),
                            "salience": round(top_ent.get("salience", 0), 4),
                            "type": top_ent.get("type", ""),
                        } if top_ent else None,
                        "entities": [
                            {"name": e["name"], "salience": round(e["salience"], 4), 
                             "type": e["type"], "has_wikipedia": bool(e.get("wikipedia_url")),
                             "has_kg": bool(e.get("mid"))}
                            for e in nlp_entities[:12]
                        ],
                        "issues": salience_result.get("issues", []),
                        "recommendations": salience_result.get("recommendations", []),
                        "subject_position": subject_pos,
                    })
                except Exception as sal_err:
                    logger.warning(f"Entity salience check failed: {sal_err}")
                    yield emit("log", {"msg": f"⚠️ Salience check error: {str(sal_err)[:80]}"})
            elif full_text:
                yield emit("entity_salience", {
                    "enabled": False,
                    "score": None,
                    "message": "Ustaw GOOGLE_NLP_API_KEY aby włączyć walidację salience",
                    "subject_position": subject_pos,
                })

            # ═══ SCHEMA.ORG JSON-LD: Generate from real NLP entities ═══
            try:
                article_schema = generate_article_schema(
                    main_keyword=main_keyword,
                    entities=nlp_entities,
                    date_published=datetime.now().strftime("%Y-%m-%d"),
                    date_modified=datetime.now().strftime("%Y-%m-%d"),
                    h2_list=h2_structure,
                )
                schema_html = schema_to_html(article_schema)
                
                yield emit("schema_org", {
                    "json_ld": article_schema,
                    "html": schema_html,
                    "entity_count": len(nlp_entities),
                    "has_main_entity": bool(article_schema.get("@graph", [{}])[0].get("about")),
                    "mentions_count": len(article_schema.get("@graph", [{}])[0].get("mentions", [])),
                })
                yield emit("log", {"msg": f"📋 Schema.org: Article + {len(article_schema.get('@graph', [{}])[0].get('mentions', []))} mentions generated"})
            except Exception as schema_err:
                logger.warning(f"Schema generation error: {schema_err}")

            # ═══ TOPICAL MAP: Entity-based content architecture ═══
            try:
                topical_map = generate_topical_map(
                    main_keyword=main_keyword,
                    s1_data=s1,
                    nlp_entities=nlp_entities,
                )
                clusters = topical_map.get("clusters", [])
                if clusters:
                    yield emit("topical_map", {
                        "pillar": topical_map["pillar"],
                        "clusters": clusters[:12],
                        "internal_links": topical_map.get("internal_links", [])[:20],
                        "total_clusters": len(clusters),
                    })
                    yield emit("log", {"msg": f"🗺️ Topical Map: {len(clusters)} klastrów treści wokół \"{main_keyword}\""})
            except Exception as tm_err:
                logger.warning(f"Topical map error: {tm_err}")

        # Export HTML
        export_result = brajen_call("get", f"/api/project/{project_id}/export/html")
        if export_result["ok"]:
            if export_result.get("binary"):
                # Save binary export
                export_path = f"/tmp/brajen_export_{project_id}.html"
                with open(export_path, "wb") as f:
                    f.write(export_result["content"])
                job["export_html"] = export_path
            else:
                content = export_result["data"] if isinstance(export_result["data"], str) else json.dumps(export_result["data"])
                export_path = f"/tmp/brajen_export_{project_id}.html"
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(content)
                job["export_html"] = export_path

        # Export DOCX
        export_docx = brajen_call("get", f"/api/project/{project_id}/export/docx")
        if export_docx["ok"] and export_docx.get("binary"):
            export_path = f"/tmp/brajen_export_{project_id}.docx"
            with open(export_path, "wb") as f:
                f.write(export_docx["content"])
            job["export_docx"] = export_path

        step_done(10)
        yield emit("step", {"step": 11, "name": "Export", "status": "done",
                            "detail": "HTML + DOCX gotowe"})

        # ─── DONE ───
        total_elapsed = round(time.time() - workflow_start, 1)
        yield emit("log", {"msg": f"⏱️ Workflow zakończony w {total_elapsed}s"})
        yield emit("done", {
            "project_id": project_id,
            "word_count": stats.get("word_count", 0) if full_result["ok"] else 0,
            "exports": {
                "html": bool(job.get("export_html")),
                "docx": bool(job.get("export_docx"))
            },
            "timing": {
                "total_seconds": total_elapsed,
                "steps": {str(k): v.get("elapsed", 0) for k, v in step_times.items()},
            }
        })

    except Exception as e:
        logger.exception(f"Workflow error: {e}")
        yield emit("workflow_error", {"step": 0, "msg": f"Unexpected error: {str(e)}"})


# ============================================================
# ARTICLE EDITOR: Chat + Inline editing with Claude
# ============================================================
@app.route("/api/edit", methods=["POST"])
@login_required
def edit_article():
    """Edit article via Claude based on user instruction."""
    data = request.json
    instruction = (data.get("instruction") or "").strip()
    article_text = (data.get("article_text") or "").strip()
    selected_text = (data.get("selected_text") or "").strip()
    job_id = data.get("job_id", "")

    if not instruction or not article_text:
        return jsonify({"error": "Brak instrukcji lub tekstu artykułu"}), 400

    if selected_text:
        system_prompt = (
            "Jesteś redaktorem artykułu SEO. Użytkownik zaznaczył fragment tekstu i chce go zmienić. "
            "Zwróć TYLKO poprawiony fragment, nie cały artykuł. "
            "Zachowaj formatowanie (h2:, h3: itd). Nie dodawaj komentarzy."
        )
        user_prompt = (
            f"CAŁY ARTYKUŁ (kontekst):\n{article_text[:6000]}\n\n"
            f"═══ ZAZNACZONY FRAGMENT ═══\n{selected_text}\n\n"
            f"═══ INSTRUKCJA ═══\n{instruction}\n\n"
            f"Zwróć TYLKO poprawiony fragment (zamiennik za zaznaczony tekst):"
        )
    else:
        system_prompt = (
            "Jesteś redaktorem artykułu SEO. Użytkownik prosi o zmianę w artykule. "
            "Zwróć CAŁY poprawiony artykuł z naniesionymi zmianami. "
            "Zachowaj formatowanie (h2:, h3: itd). Nie dodawaj komentarzy ani wyjaśnień. TYLKO tekst artykułu."
        )
        user_prompt = (
            f"ARTYKUŁ:\n{article_text}\n\n"
            f"═══ INSTRUKCJA ═══\n{instruction}\n\n"
            f"Zwróć poprawiony artykuł:"
        )

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        
        # v50.7 FIX 48: Auto-retry on 429/529
        def _call():
            return client.messages.create(
                model=model, max_tokens=8000,
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt
            )
        response = _llm_call_with_retry(_call)
        result_text = response.content[0].text.strip()
        return jsonify({
            "ok": True, "edited_text": result_text,
            "edit_type": "inline" if selected_text else "full",
            "model": model,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        })
    except Exception as e:
        logger.exception(f"Edit error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate", methods=["POST"])
@login_required
def validate_article():
    """Validate edited article via backend API."""
    data = request.json
    article_text = (data.get("article_text") or "").strip()
    job_id = data.get("job_id", "")
    if not article_text:
        return jsonify({"error": "Brak tekstu artykułu"}), 400
    job = active_jobs.get(job_id, {})
    project_id = job.get("project_id")
    if not project_id:
        return jsonify({"error": "Brak project_id, uruchom najpierw workflow"}), 400
    try:
        result = brajen_call("post", f"/api/project/{project_id}/validate_full_article",
                             {"full_text": article_text})
        if result["ok"]:
            return jsonify({"ok": True, "validation": result["data"]})
        fr = brajen_call("get", f"/api/project/{project_id}/final_review")
        if fr["ok"]:
            return jsonify({"ok": True, "validation": fr["data"]})
        return jsonify({"error": "Walidacja niedostępna"}), 500
    except Exception as e:
        logger.exception(f"Validate error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session.get("user", ""))


@app.route("/api/engines")
@login_required
def get_engines():
    """Return available AI engines and their models."""
    return jsonify({
        "engines": {
            "claude": {
                "available": bool(ANTHROPIC_API_KEY),
                "model": ANTHROPIC_MODEL,
            },
            "openai": {
                "available": bool(OPENAI_API_KEY) and OPENAI_AVAILABLE,
                "model": OPENAI_MODEL,
            },
        },
        "default": "claude",
    })


@app.route("/api/start", methods=["POST"])
@login_required
def start_workflow():
    """Start workflow and return job_id."""
    data = request.json

    main_keyword = data.get("main_keyword", "").strip()
    if not main_keyword:
        return jsonify({"error": "Brak hasła głównego"}), 400

    mode = data.get("mode", "standard")
    h2_list = [h.strip() for h in (data.get("h2_structure") or []) if h.strip()]
    basic_terms = [t.strip() for t in (data.get("basic_terms") or []) if t.strip()]
    extended_terms = [t.strip() for t in (data.get("extended_terms") or []) if t.strip()]
    custom_instructions = (data.get("custom_instructions") or "").strip()
    engine = data.get("engine", "claude")  # "claude" or "openai"
    openai_model_override = data.get("openai_model")  # per-session model override
    user_temperature = data.get("temperature")  # 0.0-1.0 or None
    if user_temperature is not None:
        user_temperature = max(0.0, min(1.0, float(user_temperature)))

    # H2 is now OPTIONAL : if empty, will be auto-generated from S1

    job_id = str(uuid.uuid4())[:8]
    
    # Cleanup old jobs to prevent memory leaks
    _cleanup_old_jobs()
    
    active_jobs[job_id] = {
        "main_keyword": main_keyword,
        "mode": mode,
        "engine": engine,
        "openai_model": openai_model_override,
        "temperature": user_temperature,
        "h2_structure": h2_list,
        "basic_terms": basic_terms,
        "extended_terms": extended_terms,
        "custom_instructions": custom_instructions,
        "status": "running",
        "created": datetime.now().isoformat(),
        "created_at": datetime.utcnow()
    }

    return jsonify({"job_id": job_id})


def stream_with_keepalive(generator_fn, keepalive_interval=15):
    """Run SSE generator in background thread, inject keepalive pings to prevent proxy timeouts."""
    q = queue.Queue()

    def run():
        try:
            for item in generator_fn():
                q.put(item)
        except Exception as e:
            q.put(f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n")
        finally:
            q.put(None)  # sentinel = stream finished

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    while True:
        try:
            item = q.get(timeout=keepalive_interval)
            if item is None:
                break
            yield item
        except queue.Empty:
            # No event for {keepalive_interval}s : send SSE comment to keep connection alive
            yield ": keepalive\n\n"


@app.route("/api/stream/<job_id>")
@login_required
def stream_workflow(job_id):
    """SSE endpoint for workflow progress with keepalive."""
    job = active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    data = job

    # Pass basic/extended through query params from frontend
    basic_terms = request.args.get("basic_terms", "")
    extended_terms = request.args.get("extended_terms", "")

    def generate_with_terms():
        bt = json.loads(basic_terms) if basic_terms else (data.get("basic_terms") or [])
        et = json.loads(extended_terms) if extended_terms else (data.get("extended_terms") or [])
        yield from run_workflow_sse(
            job_id=job_id,
            main_keyword=data["main_keyword"],
            mode=data["mode"],
            h2_structure=data["h2_structure"],
            basic_terms=bt,
            extended_terms=et,
            engine=data.get("engine", "claude"),
            openai_model=data.get("openai_model"),
            temperature=data.get("temperature")
        )

    return Response(
        stream_with_context(stream_with_keepalive(generate_with_terms, keepalive_interval=15)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/api/export/<job_id>/<fmt>")
@login_required
def download_export(job_id, fmt):
    """Download exported file."""
    job = active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    key = f"export_{fmt}"
    path = job.get(key)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"Export {fmt} not available"}), 404

    mime = {
        "html": "text/html",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain"
    }.get(fmt, "application/octet-stream")

    return send_file(path, mimetype=mime, as_attachment=True,
                     download_name=f"article_{job_id}.{fmt}")


@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({"status": "ok", "version": "45.3.2"})


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "false").lower() == "true")
