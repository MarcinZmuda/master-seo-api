"""
===============================================================================
🔤 SYNONYM SERVICE v33.0
===============================================================================
Serwis synonimów dla języka polskiego z cache w Firestore.

Strategie:
1. CACHE: Sprawdź Firestore (collection: synonyms_cache)
2. PLWORDNET_API: Zapytaj plWordNet API (jeśli dostępne)
3. LLM_FALLBACK: Użyj Claude/GPT do wygenerowania synonimów
4. STATIC_MAP: Ostateczny fallback do statycznej mapy

Cache format w Firestore:
{
    "word": "skóra",
    "synonyms": ["cera", "naskórek", "powłoka"],
    "source": "plwordnet|llm|static",
    "created_at": "2024-01-16T..."
}
===============================================================================
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime

# ================================================================
# 📚 STATYCZNA MAPA (FALLBACK)
# ================================================================
STATIC_SYNONYM_MAP = {
    # Skóra / uroda
    "skóra": ["cera", "naskórek", "powierzchnia skóry", "tkanka", "powłoka"],
    "witamina": ["mikroskładnik", "substancja odżywcza", "składnik", "nutrient"],
    "suplement": ["preparat", "produkt", "środek", "wsparcie"],
    "kolagen": ["białko strukturalne", "włókna kolagenowe", "substancja budulcowa"],
    "nawilżenie": ["hydratacja", "uwodnienie", "poziom wilgoci"],
    
    # Przymiotniki
    "ważny": ["istotny", "znaczący", "zasadniczy", "niezbędny", "doniosły"],
    "dobry": ["skuteczny", "wartościowy", "korzystny", "efektywny", "pomocny"],
    "zdrowy": ["prawidłowy", "właściwy", "optymalny"],
    "duży": ["znaczny", "spory", "pokaźny", "niemały"],
    "mały": ["niewielki", "drobny", "ograniczony"],
    "nowy": ["nowoczesny", "świeży", "najnowszy", "aktualny"],
    
    # Czasowniki
    "poprawia": ["wspiera", "wzmacnia", "podnosi", "ulepsza"],
    "pomaga": ["wspiera", "ułatwia", "wspomaga", "przyczynia się"],
    "zawiera": ["posiada", "obejmuje", "ma w składzie"],
    "powoduje": ["wywołuje", "skutkuje", "prowadzi do"],
    "działa": ["funkcjonuje", "pracuje", "oddziałuje", "wpływa"],
    "chroni": ["zabezpiecza", "ochrania", "osłania"],
    
    # Usługi / biznes
    "firma": ["przedsiębiorstwo", "spółka", "wykonawca", "usługodawca"],
    "usługa": ["świadczenie", "realizacja", "obsługa", "serwis"],
    "klient": ["zleceniodawca", "usługobiorca", "zamawiający"],
    "cena": ["koszt", "stawka", "wycena", "taryfa"],
    "profesjonalny": ["doświadczony", "wykwalifikowany", "fachowy"],
}

# ================================================================
# 🔧 KONFIGURACJA
# ================================================================
PLWORDNET_API_URL = "http://slowosiec.ws.clarin-pl.eu/plwordnet-api/senses/search"
PLWORDNET_TIMEOUT = 2  # sekundy

# Anthropic API (dla LLM fallback)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


# ================================================================
# 📦 FIRESTORE CACHE
# ================================================================
_db = None

def _get_firestore():
    """Lazy init Firestore."""
    global _db
    if _db is None:
        try:
            from firebase_admin import firestore
            _db = firestore.client()
        except:
            pass
    return _db


def get_cached_synonyms(word: str) -> Optional[List[str]]:
    """Pobiera synonimy z cache Firestore."""
    db = _get_firestore()
    if not db:
        return None
    
    try:
        doc = db.collection("synonyms_cache").document(word.lower()).get()
        if doc.exists:
            data = doc.to_dict()
            return data.get("synonyms", [])
    except Exception as e:
        print(f"[SYNONYM_CACHE] Error reading: {e}")
    
    return None


def save_to_cache(word: str, synonyms: List[str], source: str):
    """Zapisuje synonimy do cache Firestore."""
    db = _get_firestore()
    if not db or not synonyms:
        return
    
    try:
        db.collection("synonyms_cache").document(word.lower()).set({
            "word": word.lower(),
            "synonyms": synonyms[:10],  # max 10 synonimów
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        })
        print(f"[SYNONYM_CACHE] Saved: {word} -> {synonyms[:3]}...")
    except Exception as e:
        print(f"[SYNONYM_CACHE] Error saving: {e}")


# ================================================================
# 🌐 PLWORDNET API
# ================================================================
def get_synonyms_plwordnet(word: str) -> Optional[List[str]]:
    """
    Pobiera synonimy z plWordNet API.
    Zwraca None jeśli API niedostępne.
    """
    try:
        response = requests.get(
            PLWORDNET_API_URL,
            params={"lemma": word},
            timeout=PLWORDNET_TIMEOUT
        )
        
        if response.ok:
            data = response.json()
            
            # Wyciągnij synonimy z synsetów
            synonyms = set()
            for sense in data.get("senses", []):
                synset = sense.get("synset", {})
                for unit in synset.get("lexical_units", []):
                    lemma = unit.get("lemma", "")
                    if lemma and lemma.lower() != word.lower():
                        synonyms.add(lemma)
            
            if synonyms:
                return list(synonyms)[:10]
    
    except requests.Timeout:
        print(f"[PLWORDNET] Timeout for: {word}")
    except Exception as e:
        print(f"[PLWORDNET] Error: {e}")
    
    return None


# ================================================================
# 🤖 LLM FALLBACK (Claude)
# ================================================================
def get_synonyms_llm(word: str, context: str = "") -> Optional[List[str]]:
    """
    Generuje synonimy używając Claude API.
    """
    if not ANTHROPIC_API_KEY:
        print("[SYNONYM_LLM] No ANTHROPIC_API_KEY")
        return None
    
    try:
        prompt = f"""Podaj 5-8 synonimów dla polskiego słowa "{word}".
{f'Kontekst użycia: {context}' if context else ''}

Odpowiedz TYLKO listą słów oddzielonych przecinkami, bez numeracji i wyjaśnień.
Przykład: cera, naskórek, powłoka, tkanka"""

        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",  # najtańszy model
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=5
        )
        
        if response.ok:
            data = response.json()
            content = data.get("content", [{}])[0].get("text", "")
            
            # Parsuj odpowiedź
            synonyms = [s.strip() for s in content.split(",")]
            synonyms = [s for s in synonyms if s and s.lower() != word.lower()]
            
            if synonyms:
                return synonyms[:8]
    
    except Exception as e:
        print(f"[SYNONYM_LLM] Error: {e}")
    
    return None


# ================================================================
# 🎯 GŁÓWNA FUNKCJA
# ================================================================
def get_synonyms(word: str, context: str = "", use_cache: bool = True) -> Dict:
    """
    Pobiera synonimy dla słowa z różnych źródeł.
    
    Args:
        word: Słowo do znalezienia synonimów
        context: Opcjonalny kontekst (np. "artykuł o witaminach")
        use_cache: Czy używać cache Firestore
    
    Returns:
        {
            "word": "skóra",
            "synonyms": ["cera", "naskórek", ...],
            "source": "cache|plwordnet|llm|static",
            "count": 5
        }
    """
    word_lower = word.lower().strip()
    
    # 1. CACHE
    if use_cache:
        cached = get_cached_synonyms(word_lower)
        if cached:
            return {
                "word": word_lower,
                "synonyms": cached,
                "source": "cache",
                "count": len(cached)
            }
    
    # 2. PLWORDNET API
    plwordnet_result = get_synonyms_plwordnet(word_lower)
    if plwordnet_result:
        save_to_cache(word_lower, plwordnet_result, "plwordnet")
        return {
            "word": word_lower,
            "synonyms": plwordnet_result,
            "source": "plwordnet",
            "count": len(plwordnet_result)
        }
    
    # 3. LLM FALLBACK
    llm_result = get_synonyms_llm(word_lower, context)
    if llm_result:
        save_to_cache(word_lower, llm_result, "llm")
        return {
            "word": word_lower,
            "synonyms": llm_result,
            "source": "llm",
            "count": len(llm_result)
        }
    
    # 4. STATIC MAP
    static_result = STATIC_SYNONYM_MAP.get(word_lower, [])
    if static_result:
        return {
            "word": word_lower,
            "synonyms": static_result,
            "source": "static",
            "count": len(static_result)
        }
    
    # Brak synonimów
    return {
        "word": word_lower,
        "synonyms": [],
        "source": "none",
        "count": 0
    }


def get_synonyms_batch(words: List[str], context: str = "") -> Dict[str, List[str]]:
    """
    Pobiera synonimy dla wielu słów naraz.
    """
    result = {}
    for word in words:
        data = get_synonyms(word, context)
        result[word] = data["synonyms"]
    return result


# ================================================================
# 🔧 INTEGRACJA Z AI_DETECTION_METRICS
# ================================================================
def suggest_synonym_for_repetition(word: str, count: int, context: str = "") -> Dict:
    """
    Sugeruje synonim dla nadmiernie powtórzonego słowa.
    
    Args:
        word: Powtórzone słowo
        count: Ile razy wystąpiło
        context: Kontekst artykułu
    
    Returns:
        {
            "word": "skóra",
            "count": 7,
            "suggestion": "Zamień na: cera, naskórek, powłoka",
            "synonyms": ["cera", "naskórek", "powłoka"]
        }
    """
    data = get_synonyms(word, context)
    
    if data["synonyms"]:
        top_synonyms = data["synonyms"][:3]
        suggestion = f"Zamień '{word}' ({count}×) na: {', '.join(top_synonyms)}"
    else:
        suggestion = f"Słowo '{word}' powtórzone {count}× - znajdź synonimy ręcznie"
    
    return {
        "word": word,
        "count": count,
        "suggestion": suggestion,
        "synonyms": data["synonyms"]
    }


# ================================================================
# TEST
# ================================================================
if __name__ == "__main__":
    print("=== TEST SYNONYM SERVICE ===")
    
    # Test statyczny
    result = get_synonyms("skóra", use_cache=False)
    print(f"skóra: {result}")
    
    # Test dla słowa spoza mapy
    result = get_synonyms("samochód", use_cache=False)
    print(f"samochód: {result}")
    
    # Test batch
    words = ["witamina", "dobry", "firma"]
    batch_result = get_synonyms_batch(words)
    print(f"Batch: {batch_result}")
    
    # Test suggestion
    suggestion = suggest_synonym_for_repetition("skóra", 7)
    print(f"Suggestion: {suggestion}")
