# concept_map_extractor.py
# BRAJEN v34.0 - Semantic Entity SEO
# Ekstrakcja Mapy Pojęć (Concept Map) z Gemini AI

import json
import re
import os
from typing import Dict, List, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[CONCEPT_MAP] ⚠️ google-generativeai not installed")

# Konfiguracja Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[CONCEPT_MAP] ✅ Gemini configured")


# ============================================================================
# PROMPT DO EKSTRAKCJI CONCEPT MAP
# ============================================================================

CONCEPT_MAP_PROMPT = """Analizujesz temat: "{main_keyword}"

Na podstawie tekstów konkurencji, stwórz MAPĘ POJĘĆ (Concept Map) dla tego tematu.

TEKSTY KONKURENCJI:
{competitor_texts}

ZADANIE:
Zidentyfikuj strukturę semantyczną tematu:

1. ENCJA GŁÓWNA (Main Entity)
   - Co jest PRZEDMIOTEM usługi/artykułu? (nie słowo kluczowe, ale KONCEPT)
   - Typ encji wg schema.org (Service, Product, Place, Organization, Person, Event, etc.)

2. ENCJE WSPIERAJĄCE (Supporting Entities)
   - Pojęcia które MUSZĄ wystąpić żeby tekst był ekspercki
   - Podziel na: tools (narzędzia), processes (procesy), attributes (cechy), 
     locations (miejsca), certifications (certyfikaty), related_services

3. RELACJE SEMANTYCZNE
   - Związki między encją główną a wspierającymi
   - Format: subject -> predicate -> object

4. TRÓJKA KLASYFIKACYJNA (Classification Triplet)
   - 3 słowa które MUSZĄ być w pierwszych 100 słowach artykułu
   - [Typ usługi/produktu] + [Kontekst/Lokalizacja] + [Główny atrybut]

5. PROXIMITY CLUSTERS (Grupy Bliskości)
   - Które słowa MUSZĄ występować blisko siebie?
   - Jeśli piszesz o "fortepianie" - blisko muszą być "pasy", "wnoszenie", "ciężar"

6. SEMANTIC CONFIDENCE TERMS
   - Frazy które potwierdzają ekspertyzę (np. "ubezpieczenie OCP do 100 000 zł")

Zwróć TYLKO JSON:
{{
  "main_entity": {{
    "name": "nazwa encji (np. 'Usługa transportowa' nie 'przeprowadzki')",
    "type": "typ schema.org",
    "definition": "krótka definicja w kontekście artykułu"
  }},
  "supporting_entities": {{
    "tools": ["narzędzie1", "narzędzie2"],
    "processes": ["proces1", "proces2"],
    "attributes": ["atrybut1", "atrybut2"],
    "locations": ["lokalizacja1"],
    "certifications": ["certyfikat1"],
    "related_services": ["usługa1"]
  }},
  "relationships": [
    {{"subject": "encja1", "predicate": "wymaga", "object": "encja2"}},
    {{"subject": "encja1", "predicate": "oferuje", "object": "encja3"}}
  ],
  "classification_triplet": {{
    "service_type": "słowo opisujące typ",
    "context": "lokalizacja lub kontekst",
    "main_attribute": "główna cecha"
  }},
  "proximity_clusters": [
    {{
      "anchor": "słowo_kluczowe",
      "must_have_nearby": ["kontekst1", "kontekst2", "kontekst3"],
      "max_distance": 25
    }}
  ],
  "semantic_confidence_terms": [
    "fraza ekspercka 1",
    "fraza ekspercka 2"
  ]
}}

PRZYKŁAD dla "przeprowadzki warszawa":
- main_entity: "Usługa transportowa" (nie "przeprowadzki")
- supporting_entities.tools: ["winda meblowa", "pasy transportowe", "folia bąbelkowa"]
- proximity_cluster: {{"anchor": "fortepian", "must_have_nearby": ["pasy", "wnoszenie", "ciężar"]}}
- classification_triplet: {{"service_type": "przeprowadzki", "context": "Warszawa", "main_attribute": "profesjonalne"}}
"""


# ============================================================================
# FUNKCJE EKSTRAKCJI
# ============================================================================

def extract_concept_map(
    main_keyword: str,
    competitor_texts: List[str],
    gemini_model: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Ekstrahuje mapę pojęć (Concept Map) dla tematu.
    
    Args:
        main_keyword: Główna fraza kluczowa
        competitor_texts: Lista tekstów konkurencji
        gemini_model: Model Gemini do użycia
        
    Returns:
        Dict z kluczami: status, concept_map
    """
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        print("[CONCEPT_MAP] ⚠️ Gemini not available, using fallback")
        return {
            "status": "FALLBACK",
            "concept_map": get_fallback_concept_map(main_keyword)
        }
    
    # Połącz teksty konkurencji (max 15k znaków)
    combined_texts = "\n\n---ARTYKUŁ---\n\n".join([t[:3000] for t in competitor_texts[:5]])
    
    prompt = CONCEPT_MAP_PROMPT.format(
        main_keyword=main_keyword,
        competitor_texts=combined_texts[:15000]
    )
    
    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        
        # Wyciągnij JSON z odpowiedzi
        text = response.text.strip()
        
        # Usuń markdown code blocks jeśli są
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        concept_map = json.loads(text)
        
        # Walidacja i uzupełnienie brakujących kluczy
        concept_map = validate_concept_map(concept_map, main_keyword)
        
        print(f"[CONCEPT_MAP] ✅ Extracted for '{main_keyword}': "
              f"{len(concept_map.get('relationships', []))} relationships, "
              f"{len(concept_map.get('proximity_clusters', []))} clusters")
        
        return {
            "status": "OK",
            "concept_map": concept_map
        }
        
    except json.JSONDecodeError as e:
        print(f"[CONCEPT_MAP] ⚠️ JSON parse error: {e}")
        return {
            "status": "JSON_ERROR",
            "error": str(e),
            "concept_map": get_fallback_concept_map(main_keyword)
        }
    except Exception as e:
        print(f"[CONCEPT_MAP] ⚠️ Error: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "concept_map": get_fallback_concept_map(main_keyword)
        }


def validate_concept_map(concept_map: Dict, main_keyword: str) -> Dict:
    """Waliduje i uzupełnia brakujące pola w concept_map."""
    
    # Wymagane klucze z domyślnymi wartościami
    defaults = {
        "main_entity": {
            "name": main_keyword,
            "type": "Thing",
            "definition": f"Artykuł o temacie: {main_keyword}"
        },
        "supporting_entities": {
            "tools": [],
            "processes": [],
            "attributes": [],
            "locations": [],
            "certifications": [],
            "related_services": []
        },
        "relationships": [],
        "classification_triplet": {
            "service_type": main_keyword.split()[0] if main_keyword else "",
            "context": main_keyword.split()[-1] if len(main_keyword.split()) > 1 else "",
            "main_attribute": "profesjonalny"
        },
        "proximity_clusters": [],
        "semantic_confidence_terms": []
    }
    
    # Uzupełnij brakujące klucze
    for key, default_value in defaults.items():
        if key not in concept_map:
            concept_map[key] = default_value
        elif isinstance(default_value, dict):
            for sub_key, sub_default in default_value.items():
                if sub_key not in concept_map[key]:
                    concept_map[key][sub_key] = sub_default
    
    return concept_map


def get_fallback_concept_map(main_keyword: str) -> Dict:
    """Fallback gdy Gemini zawiedzie - generuje podstawową mapę."""
    
    words = main_keyword.lower().split()
    
    return {
        "main_entity": {
            "name": main_keyword,
            "type": "Thing",
            "definition": f"Artykuł o temacie: {main_keyword}"
        },
        "supporting_entities": {
            "tools": [],
            "processes": [],
            "attributes": ["profesjonalny", "doświadczony"],
            "locations": [w for w in words if len(w) > 4],
            "certifications": [],
            "related_services": []
        },
        "relationships": [],
        "classification_triplet": {
            "service_type": words[0] if words else "",
            "context": words[-1] if len(words) > 1 else "",
            "main_attribute": "profesjonalny"
        },
        "proximity_clusters": [],
        "semantic_confidence_terms": []
    }


def flatten_supporting_entities(supporting_entities: Dict) -> List[str]:
    """Spłaszcza słownik encji wspierających do listy."""
    all_entities = []
    for category, entities in supporting_entities.items():
        if isinstance(entities, list):
            all_entities.extend(entities)
    return list(set(all_entities))  # usuń duplikaty


# ============================================================================
# DODATKOWE FUNKCJE POMOCNICZE
# ============================================================================

def get_proximity_instructions(proximity_clusters: List[Dict]) -> List[str]:
    """Generuje instrukcje tekstowe dla GPT na podstawie proximity_clusters."""
    instructions = []
    
    for cluster in proximity_clusters:
        anchor = cluster.get("anchor", "")
        nearby = cluster.get("must_have_nearby", [])
        max_dist = cluster.get("max_distance", 25)
        
        if anchor and nearby:
            nearby_str = ", ".join(nearby[:5])
            instructions.append(
                f"Gdy użyjesz '{anchor}', w promieniu {max_dist} słów "
                f"MUSZĄ pojawić się min. 2 z: [{nearby_str}]"
            )
    
    return instructions


def format_concept_map_for_gpt(concept_map: Dict, batch_number: int = 1) -> str:
    """Formatuje concept_map jako instrukcje tekstowe dla GPT."""
    
    lines = []
    
    # Main entity
    main_entity = concept_map.get("main_entity", {})
    if main_entity.get("name"):
        lines.append(f"📌 ENCJA GŁÓWNA: {main_entity['name']} ({main_entity.get('type', 'Thing')})")
    
    # Supporting entities
    supporting = concept_map.get("supporting_entities", {})
    flat_entities = flatten_supporting_entities(supporting)
    if flat_entities:
        lines.append(f"\n📚 ENCJE WSPIERAJĄCE (użyj min. 3-4 w batchu):")
        lines.append(", ".join(flat_entities[:12]))
    
    # Proximity rules
    proximity = concept_map.get("proximity_clusters", [])
    if proximity:
        lines.append(f"\n🔗 ZASADY BLISKOŚCI SEMANTYCZNEJ:")
        for instr in get_proximity_instructions(proximity):
            lines.append(f"  • {instr}")
    
    # Lead paragraph (tylko dla batch 1)
    if batch_number == 1:
        triplet = concept_map.get("classification_triplet", {})
        if any(triplet.values()):
            lines.append(f"\n🎯 ZŁOTY AKAPIT (pierwsze 100 słów MUSI zawierać):")
            lines.append(f"  [{triplet.get('service_type', '?')}] + "
                        f"[{triplet.get('context', '?')}] + "
                        f"[{triplet.get('main_attribute', '?')}]")
    
    # Semantic confidence terms
    confidence = concept_map.get("semantic_confidence_terms", [])
    if confidence:
        lines.append(f"\n💎 FRAZY EKSPERCKIE (użyj 1-2 w batchu):")
        lines.append(", ".join(confidence[:5]))
    
    return "\n".join(lines)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test z przykładowymi danymi
    test_keyword = "przeprowadzki warszawa"
    test_texts = [
        "Przeprowadzki w Warszawie to usługa wymagająca doświadczenia. "
        "Profesjonalna firma przeprowadzkowa oferuje pakowanie, transport mebli "
        "i rozładunek. Ubezpieczenie OCP chroni przed szkodami.",
        
        "Transport mebli w Warszawie wymaga windy meblowej do ciężkich przedmiotów. "
        "Fortepian wymaga specjalnych pasów transportowych i doświadczonej ekipy."
    ]
    
    result = extract_concept_map(test_keyword, test_texts)
    print("\n" + "="*60)
    print("CONCEPT MAP RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("GPT INSTRUCTIONS:")
    print("="*60)
    print(format_concept_map_for_gpt(result["concept_map"], batch_number=1))
