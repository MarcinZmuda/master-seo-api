# legal_article_detector.py
# BRAJEN Legal Module v3.5 - Wykrywanie przepisów przez AI
# Claude/Gemini określa kluczowe artykuły na podstawie tematu

"""
===============================================================================
🏛️ LEGAL ARTICLE DETECTOR v1.0
===============================================================================

Zamiast hardkodowanego mapowania TEMAT → PRZEPISY,
Claude/Gemini dynamicznie określa kluczowe artykuły.

Flow:
1. Input: "ubezwłasnowolnienie całkowite"
2. AI: ["art. 13 k.c.", "art. 544 k.p.c.", "art. 545 k.p.c."]
3. Szukamy orzeczeń po tych przepisach

===============================================================================
"""

import os
import json
import re
from typing import Dict, List, Any, Optional

# Konfiguracja AI
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model Claude do wykrywania przepisów (szybki i tani)
CLAUDE_MODEL = "claude-3-haiku-20240307"

# Inicjalizacja
_anthropic_client = None

try:
    import anthropic
    if ANTHROPIC_API_KEY:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print(f"[ARTICLE_DETECTOR] ✅ Using Claude ({CLAUDE_MODEL})")
    else:
        print("[ARTICLE_DETECTOR] ⚠️ ANTHROPIC_API_KEY not set, using regex fallback")
except ImportError as e:
    print(f"[ARTICLE_DETECTOR] ⚠️ anthropic not installed: {e}")


# ============================================================================
# PROMPT DO WYKRYWANIA PRZEPISÓW
# ============================================================================

DETECT_ARTICLES_PROMPT = """Jesteś ekspertem prawa polskiego. Na podstawie tematu artykułu określ KLUCZOWE przepisy prawne.

TEMAT ARTYKUŁU: {topic}

ZADANIE:
Podaj 2-4 najważniejsze przepisy które SĄ PODSTAWĄ PRAWNĄ tego tematu.
Format: "art. X § Y ustawy" lub "art. X k.c./k.r.o./k.p.c./k.k."

SKRÓTY USTAW:
- k.c. = Kodeks cywilny
- k.r.o. = Kodeks rodzinny i opiekuńczy  
- k.p.c. = Kodeks postępowania cywilnego
- k.k. = Kodeks karny
- k.p. = Kodeks pracy

PRZYKŁADY:

Temat: "alimenty na dziecko"
Przepisy: ["art. 133 k.r.o.", "art. 135 k.r.o."]

Temat: "ubezwłasnowolnienie całkowite"
Przepisy: ["art. 13 k.c.", "art. 544 k.p.c."]

Temat: "zachowek po rodzicach"
Przepisy: ["art. 991 k.c.", "art. 994 k.c."]

Temat: "rozwód z orzeczeniem o winie"
Przepisy: ["art. 56 k.r.o.", "art. 57 k.r.o."]

Temat: "odszkodowanie za wypadek"
Przepisy: ["art. 415 k.c.", "art. 445 k.c."]

ODPOWIEDZ TYLKO W FORMACIE JSON:
{{"articles": ["art. X k.c.", "art. Y k.p.c."], "main_act": "nazwa ustawy"}}

Jeśli temat NIE jest prawny, odpowiedz:
{{"articles": [], "main_act": null, "reason": "Temat nie wymaga podstawy prawnej"}}
"""


# ============================================================================
# FUNKCJE WYKRYWANIA
# ============================================================================

def detect_legal_articles(topic: str) -> Dict[str, Any]:
    """
    Wykrywa kluczowe przepisy prawne dla danego tematu.
    
    Args:
        topic: Temat artykułu (np. "ubezwłasnowolnienie całkowite")
    
    Returns:
        {
            "status": "OK" | "NOT_LEGAL" | "ERROR",
            "articles": ["art. 13 k.c.", ...],
            "main_act": "Kodeks cywilny",
            "search_queries": ["art. 13 k.c.", "art. 544 k.p.c."]
        }
    """
    
    if not topic:
        return {
            "status": "ERROR",
            "error": "Brak tematu",
            "articles": [],
            "search_queries": []
        }
    
    # Użyj Claude do wykrycia przepisów
    if _anthropic_client:
        result = _detect_with_claude(topic)
    else:
        # Fallback: proste wykrywanie regex
        result = _detect_with_regex(topic)
    
    # Dodaj search_queries (format do wyszukiwania)
    if result.get("articles"):
        result["search_queries"] = _format_search_queries(result["articles"])
    else:
        result["search_queries"] = []
    
    return result


def _detect_with_claude(topic: str) -> Dict[str, Any]:
    """Wykrywanie przepisów przez Claude."""
    
    if not _anthropic_client:
        return _detect_with_regex(topic)
    
    try:
        prompt = DETECT_ARTICLES_PROMPT.format(topic=topic)
        
        response = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        
        # Parsuj JSON
        json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            if data.get("articles"):
                return {
                    "status": "OK",
                    "articles": data["articles"],
                    "main_act": data.get("main_act", ""),
                    "method": "claude"
                }
            else:
                return {
                    "status": "NOT_LEGAL",
                    "reason": data.get("reason", "Temat nie jest prawny"),
                    "articles": [],
                    "method": "claude"
                }
        
        return {
            "status": "ERROR",
            "error": "Parse error",
            "articles": [],
            "method": "claude"
        }
        
    except Exception as e:
        print(f"[ARTICLE_DETECTOR] Claude error: {e}")
        return _detect_with_regex(topic)


def _detect_with_regex(topic: str) -> Dict[str, Any]:
    """
    Fallback: proste wykrywanie na podstawie słów kluczowych.
    Używane gdy AI niedostępne.
    """
    
    topic_lower = topic.lower()
    
    # Podstawowe mapowanie (minimalny fallback)
    BASIC_MAP = {
        "ubezwłasnowolnienie": {
            "articles": ["art. 13 k.c.", "art. 16 k.c."],
            "main_act": "Kodeks cywilny"
        },
        "alimenty": {
            "articles": ["art. 133 k.r.o.", "art. 135 k.r.o."],
            "main_act": "Kodeks rodzinny i opiekuńczy"
        },
        "rozwód": {
            "articles": ["art. 56 k.r.o.", "art. 57 k.r.o."],
            "main_act": "Kodeks rodzinny i opiekuńczy"
        },
        "zachowek": {
            "articles": ["art. 991 k.c.", "art. 994 k.c."],
            "main_act": "Kodeks cywilny"
        },
        "spadek": {
            "articles": ["art. 922 k.c.", "art. 931 k.c."],
            "main_act": "Kodeks cywilny"
        },
        "odszkodowanie": {
            "articles": ["art. 415 k.c.", "art. 471 k.c."],
            "main_act": "Kodeks cywilny"
        },
        "zadośćuczynienie": {
            "articles": ["art. 445 k.c.", "art. 448 k.c."],
            "main_act": "Kodeks cywilny"
        },
    }
    
    for keyword, data in BASIC_MAP.items():
        if keyword in topic_lower:
            return {
                "status": "OK",
                "articles": data["articles"],
                "main_act": data["main_act"],
                "method": "regex_fallback"
            }
    
    return {
        "status": "NOT_LEGAL",
        "reason": "Nie rozpoznano tematu prawnego",
        "articles": [],
        "method": "regex_fallback"
    }


def _format_search_queries(articles: List[str]) -> List[str]:
    """
    Formatuje artykuły do zapytań wyszukiwania.
    
    Różne portale mogą mieć różne formaty:
    - SAOS: "art. 13 k.c."
    - Lokalne: "art. 13 § 1" lub "art. 13"
    """
    
    queries = []
    
    for art in articles:
        # Oryginał
        queries.append(art)
        
        # Bez kropek
        no_dots = art.replace(".", "")
        if no_dots != art:
            queries.append(no_dots)
        
        # Sam numer artykułu (np. "art. 13")
        num_match = re.search(r'art\.?\s*(\d+)', art, re.IGNORECASE)
        if num_match:
            queries.append(f"art. {num_match.group(1)}")
    
    # Usuń duplikaty zachowując kolejność
    seen = set()
    unique = []
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(q)
    
    return unique


# ============================================================================
# INTEGRACJA Z SAOS
# ============================================================================

def search_by_articles(
    articles: List[str],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Szuka orzeczeń po wykrytych artykułach.
    
    Używa SAOS API parametru referencedRegulation.
    """
    
    try:
        from saos_client import get_saos_client
        
        client = get_saos_client()
        all_judgments = []
        
        for article in articles[:3]:  # Max 3 artykuły
            results = client.search_judgments(
                keyword=article,
                page_size=max_results // len(articles) + 2
            )
            
            if results.get("items"):
                for item in results["items"]:
                    # Sprawdź czy rzeczywiście powołuje ten artykuł
                    text = item.get("textContent", "")
                    if article.lower() in text.lower():
                        formatted = client._format_judgment(item, article)
                        if formatted:
                            formatted["matched_article"] = article
                            all_judgments.append(formatted)
        
        # Deduplikacja
        seen = set()
        unique = []
        for j in all_judgments:
            sig = j.get("signature", "")
            if sig and sig not in seen:
                seen.add(sig)
                unique.append(j)
        
        return {
            "status": "success",
            "articles_searched": articles,
            "total_found": len(unique),
            "judgments": unique[:max_results]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "judgments": []
        }


# ============================================================================
# GŁÓWNA FUNKCJA - PEŁNY FLOW
# ============================================================================

def get_judgments_for_topic(
    topic: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Pełny flow: temat → przepisy → orzeczenia.
    
    Args:
        topic: Temat artykułu
        max_results: Max orzeczeń do zwrócenia
    
    Returns:
        {
            "status": "OK",
            "topic": "ubezwłasnowolnienie całkowite",
            "detected_articles": ["art. 13 k.c.", ...],
            "judgments": [...],
            "instruction": "Użyj tych orzeczeń..."
        }
    """
    
    print(f"[ARTICLE_DETECTOR] 🔍 Temat: '{topic}'")
    
    # 1. Wykryj przepisy
    detection = detect_legal_articles(topic)
    
    if detection["status"] == "NOT_LEGAL":
        return {
            "status": "NOT_LEGAL",
            "topic": topic,
            "reason": detection.get("reason", "Temat nie jest prawny"),
            "detected_articles": [],
            "judgments": [],
            "instruction": ""
        }
    
    if detection["status"] == "ERROR" or not detection.get("articles"):
        return {
            "status": "NO_ARTICLES",
            "topic": topic,
            "error": detection.get("error", "Nie wykryto przepisów"),
            "detected_articles": [],
            "judgments": [],
            "instruction": ""
        }
    
    articles = detection["articles"]
    print(f"[ARTICLE_DETECTOR] 📚 Wykryto przepisy: {articles}")
    
    # 2. Szukaj orzeczeń - wielopoziomowy fallback
    search_result = search_by_articles(articles, max_results=max_results * 2)
    
    if not search_result.get("judgments"):
        # Fallback 1: szukaj po temacie w SAOS
        print(f"[ARTICLE_DETECTOR] ⚠️ Brak orzeczeń po artykułach, fallback na SAOS temat")
        try:
            from saos_client import search_judgments
            fallback = search_judgments(topic, max_results=max_results)
            search_result = {
                "judgments": fallback.get("judgments", []),
                "fallback": "saos_topic"
            }
        except:
            pass
    
    if not search_result.get("judgments"):
        # Fallback 2: szukaj przez Google
        print(f"[ARTICLE_DETECTOR] ⚠️ Brak wyników z SAOS, fallback na Google")
        try:
            from google_judgment_fallback import search_google_fallback
            google_result = search_google_fallback(
                articles=articles,
                keyword=topic,
                max_results=max_results
            )
            if google_result.get("judgments"):
                search_result = {
                    "judgments": google_result.get("judgments", []),
                    "fallback": "google"
                }
                print(f"[ARTICLE_DETECTOR] ✅ Google fallback: {len(search_result['judgments'])} wyników")
        except Exception as e:
            print(f"[ARTICLE_DETECTOR] ⚠️ Google fallback error: {e}")
    
    judgments = search_result.get("judgments", [])[:max_results]
    
    # 3. Buduj instrukcję
    instruction = _build_instruction(topic, articles, judgments)
    
    return {
        "status": "OK",
        "topic": topic,
        "detected_articles": articles,
        "main_act": detection.get("main_act", ""),
        "detection_method": detection.get("method", ""),
        "total_found": len(judgments),
        "judgments": judgments,
        "instruction": instruction
    }


def _build_instruction(
    topic: str,
    articles: List[str],
    judgments: List[Dict]
) -> str:
    """Buduje instrukcję dla GPT jak użyć orzeczeń."""
    
    if not judgments:
        return f"Nie znaleziono orzeczeń dla '{topic}'. Artykuł może być bez cytowań."
    
    # Formatuj orzeczenia
    citations = []
    for j in judgments[:2]:
        sig = j.get("signature", "")
        date = j.get("formatted_date", j.get("date", ""))
        court = j.get("court", "")
        url = j.get("url", "")
        
        if sig:
            citations.append(f"- {court}, {sig} z {date}\n  Link: {url}")
    
    instruction = f"""ORZECZENIA DLA TEMATU: {topic}

PODSTAWA PRAWNA: {', '.join(articles)}

UŻYJ MAKSYMALNIE 2 ORZECZEŃ:
{chr(10).join(citations)}

JAK UŻYĆ:
1. Wpleć naturalnie w tekst (nie na siłę)
2. Powołaj się na przepis + orzeczenie
3. Użyj sygnatury i daty DOKŁADNIE jak podano
4. Link do SAOS w formacie: [wyrok SO ... (sygnatura)](url)

PRZYKŁAD:
"Zgodnie z art. 13 k.c., osoba może być ubezwłasnowolniona całkowicie, 
jeżeli nie jest w stanie kierować swoim postępowaniem. Jak wskazał 
Sąd Okręgowy w Warszawie w postanowieniu z dnia 20 czerwca 2024 r. 
(sygn. I Ns 36/23), sam wiek nie stanowi przesłanki ubezwłasnowolnienia."
"""
    
    return instruction


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏛️ LEGAL ARTICLE DETECTOR v1.0 TEST")
    print("=" * 60)
    
    test_topics = [
        "ubezwłasnowolnienie całkowite",
        "alimenty na dziecko",
        "rozwód z orzeczeniem o winie",
        "zachowek po rodzicach",
        "najlepsze restauracje w Warszawie",  # nie-prawny
    ]
    
    for topic in test_topics:
        print(f"\n{'─' * 40}")
        print(f"TEMAT: {topic}")
        print(f"{'─' * 40}")
        
        result = detect_legal_articles(topic)
        
        print(f"Status: {result['status']}")
        print(f"Metoda: {result.get('method', 'N/A')}")
        
        if result.get("articles"):
            print(f"Przepisy: {result['articles']}")
            print(f"Ustawa: {result.get('main_act', 'N/A')}")
            print(f"Queries: {result.get('search_queries', [])}")
        else:
            print(f"Powód: {result.get('reason', result.get('error', 'N/A'))}")
