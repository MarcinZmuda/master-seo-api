"""
===============================================================================
🧠 ENTITY ROUTES v29.3 - Entity Validation & Synonym API
===============================================================================
Endpointy dla Entity SEO:
- POST /api/project/{id}/validate_entities - walidacja użycia encji
- GET /api/synonyms/{phrase} - synonimy dla fraz (LSI)
===============================================================================
"""

from flask import Blueprint, request, jsonify
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import json
import os

entity_routes = Blueprint('entity_routes', __name__)

# ================================================================
# 📚 SŁOWNIK SYNONIMÓW (LSI)
# ================================================================

SYNONYM_DATABASE = {
    # Pomoce sensoryczne
    "pomoce sensoryczne": [
        "narzędzia terapeutyczne",
        "sprzęt SI",
        "akcesoria sensoryczne",
        "materiały do stymulacji zmysłów",
        "pomoce do terapii",
        "zestawy sensoryczne",
        "sprzęt terapeutyczny"
    ],
    "integracja sensoryczna": [
        "SI",
        "terapia integracji sensorycznej",
        "przetwarzanie sensoryczne",
        "stymulacja zmysłów",
        "terapia SI",
        "integracja zmysłowa"
    ],
    "ścieżka sensoryczna": [
        "tor sensoryczny",
        "ścieżka dotykowa",
        "ścieżka zmysłowa",
        "tor dotykowy",
        "mata sensoryczna"
    ],
    "dziecko": [
        "maluch",
        "przedszkolak",
        "najmłodsi",
        "pociecha",
        "wychowanek",
        "dziecko przedszkolne",
        "brzdąc"
    ],
    "rozwój": [
        "postęp",
        "doskonalenie",
        "kształtowanie",
        "wzrost",
        "dojrzewanie",
        "progres"
    ],
    "przedszkole": [
        "placówka przedszkolna",
        "ośrodek przedszkolny",
        "grupa przedszkolna",
        "zerówka",
        "placówka edukacyjna"
    ],
    "terapia": [
        "zajęcia terapeutyczne",
        "sesja terapeutyczna",
        "leczenie",
        "rehabilitacja",
        "wsparcie terapeutyczne"
    ],
    "zmysły": [
        "percepcja",
        "odczuwanie",
        "postrzeganie",
        "receptory",
        "układ sensoryczny"
    ],
    # Ogólne SEO
    "ważny": [
        "istotny",
        "znaczący",
        "kluczowy",
        "fundamentalny",
        "decydujący"
    ],
    "skuteczny": [
        "efektywny",
        "działający",
        "sprawdzony",
        "wydajny",
        "rezultatywny"
    ],
    "metoda": [
        "sposób",
        "technika",
        "podejście",
        "strategia",
        "system"
    ],
    "problem": [
        "trudność",
        "wyzwanie",
        "kwestia",
        "zagadnienie",
        "kłopot"
    ]
}

# LSI Keywords dla popularnych tematów
LSI_DATABASE = {
    "integracja sensoryczna": [
        "przetwarzanie bodźców",
        "stymulacja zmysłów",
        "układ nerwowy",
        "koordynacja ruchowa",
        "motoryka mała",
        "motoryka duża",
        "równowaga",
        "percepcja",
        "zmysł dotyku",
        "zmysł równowagi",
        "propriocepcja",
        "układ przedsionkowy"
    ],
    "pomoce sensoryczne": [
        "ścieżka sensoryczna",
        "piłki sensoryczne",
        "maty dotykowe",
        "panele sensoryczne",
        "huśtawki terapeutyczne",
        "dyski balansowe",
        "faktury",
        "tekstury"
    ],
    "przedszkole": [
        "edukacja przedszkolna",
        "nauczyciel przedszkolny",
        "grupa wiekowa",
        "zajęcia przedszkolne",
        "adaptacja",
        "socjalizacja"
    ],
    "montessori": [
        "samodzielność dziecka",
        "przygotowane otoczenie",
        "nauka przez działanie",
        "materiały montessori",
        "pedagogika montessori"
    ]
}

# ================================================================
# 🔍 KNOWN ENTITIES DATABASE
# ================================================================

KNOWN_ENTITIES = {
    # Osoby
    "jean ayres": {
        "canonical": "Anna Jean Ayres",
        "type": "PERSON",
        "aliases": ["jean ayres", "a. jean ayres", "dr ayres", "ayres"],
        "definition_hint": "amerykańska terapeutka zajęciowa i psycholog, twórczyni teorii integracji sensorycznej"
    },
    "maria montessori": {
        "canonical": "Maria Montessori",
        "type": "PERSON",
        "aliases": ["montessori", "maria montessori"],
        "definition_hint": "włoska lekarka i pedagog, twórczyni metody Montessori"
    },
    # Koncepcje
    "integracja sensoryczna": {
        "canonical": "integracja sensoryczna",
        "type": "CONCEPT",
        "aliases": ["si", "integracja zmysłowa", "terapia si"],
        "definition_hint": "proces neurologiczny organizowania bodźców zmysłowych"
    },
    "propriocepcja": {
        "canonical": "propriocepcja",
        "type": "CONCEPT",
        "aliases": ["zmysł proprioceptywny", "czucie głębokie"],
        "definition_hint": "zmysł informujący mózg o położeniu ciała w przestrzeni"
    },
    "układ przedsionkowy": {
        "canonical": "układ przedsionkowy",
        "type": "CONCEPT",
        "aliases": ["zmysł równowagi", "układ błędnikowy"],
        "definition_hint": "część ucha wewnętrznego odpowiedzialna za równowagę"
    }
}


# ================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================

def normalize_text(text: str) -> str:
    """Normalizuje tekst do porównań."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_entity_in_text(text: str, entity_name: str, aliases: List[str] = None) -> Tuple[int, bool]:
    """
    Znajduje encję w tekście.
    Zwraca: (count, is_defined)
    """
    text_normalized = normalize_text(text)
    text_lower = text.lower()
    
    names_to_check = [entity_name.lower()]
    if aliases:
        names_to_check.extend([a.lower() for a in aliases])
    
    total_count = 0
    for name in names_to_check:
        # Szukaj jako całe słowa
        pattern = r'\b' + re.escape(normalize_text(name)) + r'\b'
        matches = re.findall(pattern, text_normalized)
        total_count += len(matches)
    
    # Sprawdź czy jest definicja (heurystyka)
    is_defined = False
    definition_patterns = [
        r'to\s+\w+',  # "X to ..."
        r'czyli\s+',  # "X, czyli ..."
        r'jest\s+to',  # "X jest to ..."
        r'polega\s+na',  # "X polega na ..."
        r'opracował',  # "X opracował/a"
        r'stworzy[łl]',  # "X stworzył/a"
        r'[\(\)]',  # "X (skrót od Y)"
    ]
    
    for name in names_to_check:
        for pattern in definition_patterns:
            full_pattern = re.escape(name) + r'[,\s]+' + pattern
            if re.search(full_pattern, text_lower):
                is_defined = True
                break
    
    return total_count, is_defined


def extract_relationships(text: str) -> List[str]:
    """Wyciąga relacje między encjami z tekstu."""
    relationships = []
    
    relationship_patterns = [
        (r'(\w+)\s+opracował[a]?\s+(\w+)', "{0} → opracował → {1}"),
        (r'(\w+)\s+stworzy[łl][a]?\s+(\w+)', "{0} → stworzył → {1}"),
        (r'(\w+)\s+wpływa\s+na\s+(\w+)', "{0} → wpływa na → {1}"),
        (r'(\w+)\s+prowadzi\s+do\s+(\w+)', "{0} → prowadzi do → {1}"),
        (r'(\w+)\s+wspiera\s+(\w+)', "{0} → wspiera → {1}"),
        (r'(\w+)\s+jest\s+częścią\s+(\w+)', "{0} → jest częścią → {1}"),
        (r'(\w+)\s+obejmuje\s+(\w+)', "{0} → obejmuje → {1}"),
    ]
    
    text_lower = text.lower()
    
    for pattern, template in relationship_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) >= 2:
                rel = template.format(match[0], match[1])
                relationships.append(rel)
    
    return relationships[:10]  # Max 10


def calculate_entity_score(
    entities_used: List[Dict],
    must_have: List[str],
    total_words: int
) -> int:
    """Oblicza score encji (0-100)."""
    score = 0
    
    # 1. Coverage of must_have (40 points)
    if must_have:
        covered = sum(1 for e in entities_used if e["name"].lower() in [m.lower() for m in must_have])
        coverage_pct = covered / len(must_have)
        score += int(coverage_pct * 40)
    else:
        score += 40  # Brak must_have = pełne punkty
    
    # 2. Entity density (30 points)
    unique_entities = len(entities_used)
    optimal_density = total_words / 100 * 3  # ~3 encje na 100 słów
    density_ratio = min(unique_entities / max(optimal_density, 1), 1.5)
    score += int(min(density_ratio, 1) * 30)
    
    # 3. Definitions (30 points)
    defined_count = sum(1 for e in entities_used if e.get("defined", False))
    if entities_used:
        definition_ratio = defined_count / len(entities_used)
        score += int(definition_ratio * 30)
    
    return min(score, 100)


# ================================================================
# 📡 ENDPOINTS
# ================================================================

@entity_routes.post("/api/project/<project_id>/validate_entities")
def validate_entities(project_id):
    """
    Waliduje użycie encji w artykule.
    
    Request:
        {
            "text": "Treść artykułu..."
        }
    
    Response:
        {
            "entity_score": 75,
            "entities_used": [...],
            "entities_missing": [...],
            "entity_density": {...},
            "relationships_found": [...],
            "suggestions": [...]
        }
    """
    from master_api import get_project
    
    try:
        project = get_project(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        data = request.get_json(force=True)
        text = data.get("text", "")
        
        if not text:
            # Spróbuj wziąć z projektu
            batches = project.get("batches", [])
            text = "\n\n".join([b.get("text", "") for b in batches])
        
        if not text:
            return jsonify({"error": "No text to validate"}), 400
        
        # Pobierz must_have entities z S1
        s1_data = project.get("s1_data", {})
        entity_seo = s1_data.get("entity_seo", {})
        must_mention = entity_seo.get("must_mention_entities", [])
        top_entities_s1 = entity_seo.get("top_entities", [])
        
        # Jeśli nie ma must_mention, weź top entities
        if not must_mention and top_entities_s1:
            must_mention = [e.get("name", "") for e in top_entities_s1[:5]]
        
        # Dodaj known entities dla tematu
        main_keyword = project.get("main_keyword", "").lower()
        if "sensorycz" in main_keyword or "integracja" in main_keyword:
            must_mention.extend(["jean ayres", "propriocepcja", "układ przedsionkowy"])
        
        must_mention = list(set([m for m in must_mention if m]))
        
        # Analizuj tekst
        text_normalized = normalize_text(text)
        total_words = len(text.split())
        
        entities_used = []
        entities_missing = []
        
        # Sprawdź known entities
        for entity_key, entity_data in KNOWN_ENTITIES.items():
            count, is_defined = find_entity_in_text(
                text, 
                entity_data["canonical"],
                entity_data.get("aliases", [])
            )
            
            if count > 0:
                entities_used.append({
                    "name": entity_data["canonical"],
                    "type": entity_data["type"],
                    "count": count,
                    "defined": is_defined
                })
        
        # Sprawdź must_mention
        for entity_name in must_mention:
            found = False
            for used in entities_used:
                if entity_name.lower() in used["name"].lower():
                    found = True
                    break
            
            if not found:
                # Sprawdź bezpośrednio w tekście
                count, is_defined = find_entity_in_text(text, entity_name)
                if count > 0:
                    entities_used.append({
                        "name": entity_name,
                        "type": "UNKNOWN",
                        "count": count,
                        "defined": is_defined
                    })
                else:
                    entities_missing.append(entity_name)
        
        # Znajdź relacje
        relationships = extract_relationships(text)
        
        # Oblicz score
        entity_score = calculate_entity_score(entities_used, must_mention, total_words)
        
        # Oblicz density
        unique_entities = len(entities_used)
        total_mentions = sum(e["count"] for e in entities_used)
        
        entity_density = {
            "total_entities": unique_entities,
            "total_mentions": total_mentions,
            "density_per_100_words": round(total_mentions / max(total_words, 1) * 100, 2),
            "unique_per_100_words": round(unique_entities / max(total_words, 1) * 100, 2)
        }
        
        # Generuj sugestie
        suggestions = []
        
        if entities_missing:
            suggestions.append(f"Dodaj brakujące encje: {', '.join(entities_missing[:3])}")
        
        undefined = [e["name"] for e in entities_used if not e.get("defined", False)]
        if undefined:
            suggestions.append(f"Zdefiniuj encje przy pierwszym użyciu: {', '.join(undefined[:3])}")
        
        if entity_density["density_per_100_words"] < 2:
            suggestions.append("Gęstość encji za niska - dodaj więcej odniesień do kluczowych pojęć")
        
        if not relationships:
            suggestions.append("Brak wykrytych relacji między encjami - połącz je przyczynowo (X wpływa na Y)")
        
        return jsonify({
            "project_id": project_id,
            "entity_score": entity_score,
            "entities_used": sorted(entities_used, key=lambda x: x["count"], reverse=True),
            "entities_missing": entities_missing,
            "entity_density": entity_density,
            "relationships_found": relationships,
            "suggestions": suggestions,
            "must_mention_entities": must_mention,
            "total_words": total_words
        }), 200
        
    except Exception as e:
        print(f"[ENTITY_VALIDATE] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@entity_routes.get("/api/synonyms/<path:phrase>")
def get_synonyms(phrase):
    """
    Zwraca synonimy dla frazy (LSI).
    
    Response:
        {
            "phrase": "pomoce sensoryczne",
            "synonyms": [...],
            "variants": [...],
            "lsi_related": [...]
        }
    """
    try:
        phrase_normalized = normalize_text(phrase)
        phrase_lower = phrase.lower().strip()
        
        # Szukaj dokładnego dopasowania
        synonyms = SYNONYM_DATABASE.get(phrase_lower, [])
        
        # Jeśli nie znaleziono, szukaj częściowego dopasowania
        if not synonyms:
            for key, syns in SYNONYM_DATABASE.items():
                if phrase_lower in key or key in phrase_lower:
                    synonyms = syns
                    break
        
        # Warianty (odmiany)
        variants = []
        if phrase_lower.endswith("y"):
            variants.append(phrase_lower[:-1] + "a")
            variants.append(phrase_lower[:-1] + "ami")
        elif phrase_lower.endswith("a"):
            variants.append(phrase_lower[:-1] + "y")
            variants.append(phrase_lower[:-1] + "ą")
        
        # LSI related
        lsi_related = []
        for key, lsi_list in LSI_DATABASE.items():
            if phrase_lower in key or key in phrase_lower:
                lsi_related = lsi_list
                break
        
        # Jeśli nie znaleziono nic, zwróć puste listy
        return jsonify({
            "phrase": phrase,
            "synonyms": synonyms[:10],
            "variants": variants[:5],
            "lsi_related": lsi_related[:10],
            "found": bool(synonyms or lsi_related)
        }), 200
        
    except Exception as e:
        print(f"[SYNONYMS] ❌ Error: {e}")
        return jsonify({
            "phrase": phrase,
            "synonyms": [],
            "variants": [],
            "lsi_related": [],
            "error": str(e)
        }), 200


@entity_routes.get("/api/entities/known")
def get_known_entities():
    """Zwraca listę znanych encji."""
    return jsonify({
        "entities": [
            {
                "name": data["canonical"],
                "type": data["type"],
                "aliases": data.get("aliases", [])
            }
            for key, data in KNOWN_ENTITIES.items()
        ],
        "total": len(KNOWN_ENTITIES)
    }), 200


@entity_routes.get("/api/synonyms/database")
def get_synonym_database():
    """Zwraca całą bazę synonimów."""
    return jsonify({
        "synonyms": SYNONYM_DATABASE,
        "lsi": LSI_DATABASE,
        "total_phrases": len(SYNONYM_DATABASE),
        "total_lsi_topics": len(LSI_DATABASE)
    }), 200


# ================================================================
# 🧪 TEST
# ================================================================

if __name__ == "__main__":
    # Test synonyms
    test_phrase = "pomoce sensoryczne"
    print(f"\n🔍 Synonimy dla: {test_phrase}")
    syns = SYNONYM_DATABASE.get(test_phrase, [])
    print(f"   Synonimy: {syns}")
    
    # Test entity detection
    test_text = """
    Integrację sensoryczną opracowała dr Jean Ayres, amerykańska terapeutka zajęciowa.
    Metoda Montessori, stworzona przez Marię Montessori, wspiera samodzielność dziecka.
    Propriocepcja wpływa na równowagę i koordynację ruchową.
    """
    
    print(f"\n🔍 Test entity detection:")
    for entity_key, entity_data in KNOWN_ENTITIES.items():
        count, defined = find_entity_in_text(test_text, entity_data["canonical"], entity_data.get("aliases"))
        if count > 0:
            print(f"   ✅ {entity_data['canonical']}: {count}x, defined={defined}")
    
    print(f"\n🔍 Test relationships:")
    rels = extract_relationships(test_text)
    for rel in rels:
        print(f"   → {rel}")
