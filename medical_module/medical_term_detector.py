"""
===============================================================================
🔬 MEDICAL TERM DETECTOR v1.0
===============================================================================
Wykrywanie kategorii medycznych i mapowanie na terminy MeSH.

Funkcje:
1. Wykrywanie czy temat jest medyczny (YMYL Health)
2. Mapowanie polskich terminów na angielskie MeSH
3. Wykrywanie specjalizacji medycznej
4. Generowanie optymalnych zapytań PubMed

MeSH (Medical Subject Headings) - kontrolowany słownik NCBI
używany do indeksowania artykułów w PubMed.
===============================================================================
"""

import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class MedicalTermConfig:
    """Konfiguracja detektora."""
    
    # Minimalna pewność dla kategoryzacji jako medyczny
    MIN_CONFIDENCE: float = 0.4
    
    # Słowa kluczowe kategorii medycznych
    MEDICAL_KEYWORDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "choroby": [
            "choroba", "schorzenie", "zespół", "syndrom", "zaburzenie",
            "infekcja", "zakażenie", "nowotwór", "rak", "guz",
            "niewydolność", "zapalenie", "marskość", "cukrzyca",
            "nadciśnienie", "miażdżyca", "astma", "depresja"
        ],
        "leczenie": [
            "leczenie", "terapia", "kuracja", "farmakoterapia",
            "chemioterapia", "radioterapia", "operacja", "zabieg",
            "rehabilitacja", "profilaktyka", "szczepienie", "dawkowanie"
        ],
        "leki": [
            "lek", "tabletka", "kapsułka", "syrop", "maść", "zastrzyk",
            "antybiotyk", "szczepionka", "insulina", "steryd",
            "przeciwbólowy", "przeciwzapalny", "metformina", "aspiryna"
        ],
        "objawy": [
            "objaw", "symptom", "ból", "gorączka", "kaszel",
            "duszność", "nudności", "wymioty", "biegunka", "zawroty",
            "osłabienie", "zmęczenie", "bezsenność", "świąd"
        ],
        "diagnostyka": [
            "diagnoza", "badanie", "analiza", "test", "screening",
            "usg", "tomografia", "rezonans", "rtg", "biopsja",
            "morfologia", "glukoza", "cholesterol", "ciśnienie"
        ],
        "anatomia": [
            "serce", "płuca", "wątroba", "nerka", "mózg", "żołądek",
            "jelito", "kość", "mięsień", "skóra", "oko", "ucho",
            "trzustka", "tarczyca", "prostata", "macica"
        ],
        "specjalizacje": [
            "kardiologia", "neurologia", "onkologia", "pediatria",
            "ginekologia", "urologia", "dermatologia", "psychiatria",
            "ortopedia", "okulistyka", "laryngologia", "endokrynologia"
        ],
        "procedury": [
            "operacja", "zabieg", "transplantacja", "przeszczep",
            "endoskopia", "kolonoskopia", "gastroskopia", "angioplastyka",
            "bypass", "hemodializa", "chemioterapia"
        ]
    })
    
    # Mapowanie PL → EN (popularne terminy)
    TERM_TRANSLATIONS: Dict[str, str] = field(default_factory=lambda: {
        # Choroby
        "cukrzyca": "diabetes mellitus",
        "cukrzyca typu 2": "type 2 diabetes mellitus",
        "cukrzyca typu 1": "type 1 diabetes mellitus",
        "nadciśnienie": "hypertension",
        "nadciśnienie tętnicze": "arterial hypertension",
        "miażdżyca": "atherosclerosis",
        "zawał serca": "myocardial infarction",
        "udar mózgu": "stroke",
        "astma": "asthma",
        "astma oskrzelowa": "bronchial asthma",
        "depresja": "depression",
        "zaburzenia lękowe": "anxiety disorders",
        "schizofrenia": "schizophrenia",
        "choroba alzheimera": "alzheimer disease",
        "choroba parkinsona": "parkinson disease",
        "stwardnienie rozsiane": "multiple sclerosis",
        "reumatoidalne zapalenie stawów": "rheumatoid arthritis",
        "łuszczyca": "psoriasis",
        "rak piersi": "breast neoplasms",
        "rak płuca": "lung neoplasms",
        "rak jelita grubego": "colorectal neoplasms",
        "białaczka": "leukemia",
        "chłoniak": "lymphoma",
        "otyłość": "obesity",
        "niedokrwistość": "anemia",
        "osteoporoza": "osteoporosis",
        "niewydolność serca": "heart failure",
        "niewydolność nerek": "renal insufficiency",
        "marskość wątroby": "liver cirrhosis",
        "zapalenie płuc": "pneumonia",
        "grypa": "influenza",
        "covid": "covid-19",
        "covid-19": "covid-19",
        
        # Leki
        "metformina": "metformin",
        "insulina": "insulin",
        "aspiryna": "aspirin",
        "ibuprofen": "ibuprofen",
        "paracetamol": "acetaminophen",
        "antybiotyk": "anti-bacterial agents",
        "statyny": "hydroxymethylglutaryl-coa reductase inhibitors",
        "beta-blokery": "adrenergic beta-antagonists",
        "inhibitory ace": "angiotensin-converting enzyme inhibitors",
        
        # Terapie
        "chemioterapia": "drug therapy",
        "radioterapia": "radiotherapy",
        "immunoterapia": "immunotherapy",
        "fizjoterapia": "physical therapy modalities",
        "psychoterapia": "psychotherapy",
        
        # Badania
        "tomografia komputerowa": "tomography, x-ray computed",
        "rezonans magnetyczny": "magnetic resonance imaging",
        "usg": "ultrasonography",
        "ekg": "electrocardiography",
        "morfologia krwi": "blood cell count",
        
        # Ogólne
        "leczenie": "treatment",
        "terapia": "therapy",
        "profilaktyka": "prevention",
        "diagnostyka": "diagnosis",
        "objawy": "symptoms",
        "skutki uboczne": "adverse effects",
        "dawkowanie": "drug dosage"
    })
    
    # Mapowanie na specjalizacje
    SPECIALIZATION_KEYWORDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "kardiologia": ["serce", "zawał", "nadciśnienie", "arytmia", "niewydolność serca"],
        "neurologia": ["mózg", "udar", "padaczka", "migrena", "alzheimer", "parkinson"],
        "onkologia": ["rak", "nowotwór", "guz", "chemioterapia", "białaczka", "chłoniak"],
        "endokrynologia": ["cukrzyca", "tarczyca", "hormony", "insulina", "nadczynność"],
        "gastroenterologia": ["żołądek", "jelito", "wątroba", "trzustka", "refluks"],
        "pulmonologia": ["płuca", "astma", "pochp", "zapalenie płuc", "duszność"],
        "reumatologia": ["stawy", "reumatyzm", "artretyzm", "łuszczyca", "toczeń"],
        "psychiatria": ["depresja", "lęk", "schizofrenia", "bezsenność", "uzależnienie"],
        "dermatologia": ["skóra", "egzema", "trądzik", "łuszczyca", "melanoma"],
        "pediatria": ["dziecko", "niemowlę", "szczepienia", "rozwój"],
        "ginekologia": ["ciąża", "macica", "jajniki", "miesiączka", "menopauza"]
    })


CONFIG = MedicalTermConfig()


# ============================================================================
# DETEKTOR
# ============================================================================

class MedicalTermDetector:
    """Detektor i mapper terminów medycznych."""
    
    def __init__(self, config: MedicalTermConfig = None):
        self.config = config or CONFIG
        print("[MEDICAL_TERM] ✅ Detector initialized")
    
    def detect_medical_topic(
        self,
        topic: str,
        additional_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        Wykrywa czy temat jest medyczny i zwraca metadane.
        
        Args:
            topic: Główny temat (np. "leczenie cukrzycy typu 2")
            additional_keywords: Dodatkowe słowa kluczowe
        
        Returns:
            {
                "is_medical": True/False,
                "confidence": 0.0-1.0,
                "category": "leczenie" | "choroby" | ...,
                "specialization": "endokrynologia" | ...,
                "detected_keywords": [...],
                "mesh_suggestions": [...],
                "english_query": "..."
            }
        """
        additional_keywords = additional_keywords or []
        all_text = " ".join([topic] + additional_keywords).lower()
        
        # Wykryj słowa kluczowe
        detected = {}
        for category, keywords in self.config.MEDICAL_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in all_text]
            if matches:
                detected[category] = matches
        
        # Oblicz confidence
        total_matches = sum(len(v) for v in detected.values())
        confidence = min(1.0, total_matches / 3)  # 3+ matches = 100%
        
        # Określ główną kategorię
        main_category = None
        if detected:
            main_category = max(detected.keys(), key=lambda k: len(detected[k]))
        
        # Wykryj specjalizację
        specialization = self._detect_specialization(all_text)
        
        # Wygeneruj sugestie MeSH
        mesh_suggestions = self._get_mesh_suggestions(topic)
        
        # Przetłumacz na angielski
        english_query = self._translate_to_english(topic)
        
        is_medical = confidence >= self.config.MIN_CONFIDENCE
        
        return {
            "is_medical": is_medical,
            "is_ymyl": is_medical,  # YMYL Health
            "confidence": round(confidence, 2),
            "category": main_category,
            "specialization": specialization,
            "detected_keywords": detected,
            "mesh_suggestions": mesh_suggestions,
            "english_query": english_query,
            "original_topic": topic
        }
    
    def _detect_specialization(self, text: str) -> Optional[str]:
        """Wykrywa specjalizację medyczną."""
        text_lower = text.lower()
        
        best_spec = None
        best_score = 0
        
        for spec, keywords in self.config.SPECIALIZATION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_spec = spec
        
        return best_spec if best_score >= 1 else None
    
    def _get_mesh_suggestions(self, topic: str) -> List[str]:
        """Generuje sugestie terminów MeSH."""
        topic_lower = topic.lower()
        suggestions = []
        
        # Znajdź pasujące tłumaczenia
        for pl_term, en_term in self.config.TERM_TRANSLATIONS.items():
            if pl_term in topic_lower:
                suggestions.append(en_term)
        
        return suggestions[:5]  # Max 5
    
    def _translate_to_english(self, topic: str) -> str:
        """
        Tłumaczy polski temat na angielskie zapytanie PubMed.
        
        Używa słownika tłumaczeń + zachowuje nieznane słowa.
        """
        topic_lower = topic.lower()
        result = topic_lower
        
        # Sortuj po długości (dłuższe frazy najpierw)
        sorted_terms = sorted(
            self.config.TERM_TRANSLATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for pl_term, en_term in sorted_terms:
            if pl_term in result:
                result = result.replace(pl_term, en_term)
        
        return result.strip()
    
    def build_pubmed_query(
        self,
        topic: str,
        include_mesh: bool = True,
        focus: str = None
    ) -> str:
        """
        Buduje zoptymalizowane zapytanie PubMed.
        
        Args:
            topic: Temat po polsku
            include_mesh: Czy dołączyć terminy MeSH
            focus: "therapy" | "diagnosis" | "etiology" | None
        
        Returns:
            Zapytanie PubMed (po angielsku)
        """
        detection = self.detect_medical_topic(topic)
        
        query_parts = []
        
        # Użyj tłumaczenia
        if detection["english_query"]:
            query_parts.append(detection["english_query"])
        
        # Dodaj MeSH terms
        if include_mesh and detection["mesh_suggestions"]:
            mesh_terms = [f'"{term}"[MeSH]' for term in detection["mesh_suggestions"][:2]]
            if mesh_terms:
                query_parts.append(f"({' OR '.join(mesh_terms)})")
        
        # Dodaj focus
        if focus:
            focus_map = {
                "therapy": "therapy[sh]",
                "treatment": "therapy[sh]",
                "diagnosis": "diagnosis[sh]",
                "etiology": "etiology[sh]",
                "prevention": "prevention[sh]"
            }
            if focus.lower() in focus_map:
                query_parts.append(focus_map[focus.lower()])
        
        return " AND ".join(query_parts) if query_parts else detection["english_query"]
    
    def get_search_strategy(self, topic: str) -> Dict[str, Any]:
        """
        Generuje pełną strategię wyszukiwania.
        
        Returns:
            {
                "pubmed_query": "...",
                "clinicaltrials_condition": "...",
                "clinicaltrials_intervention": "...",
                "polish_query": "...",
                "recommended_filters": {...}
            }
        """
        detection = self.detect_medical_topic(topic)
        
        # PubMed query
        pubmed_query = self.build_pubmed_query(topic)
        
        # ClinicalTrials.gov
        ct_condition = None
        ct_intervention = None
        
        if detection["mesh_suggestions"]:
            # Pierwszy MeSH term jako condition
            ct_condition = detection["mesh_suggestions"][0]
            
            # Jeśli jest lek/terapia - jako intervention
            for mesh in detection["mesh_suggestions"][1:]:
                if any(kw in mesh.lower() for kw in ["therapy", "treatment", "drug"]):
                    ct_intervention = mesh
                    break
        
        # Polski query (dla PZH, MZ)
        polish_query = topic
        
        # Rekomendowane filtry
        filters = {
            "article_types": ["Systematic Review", "Meta-Analysis", "Randomized Controlled Trial"],
            "min_year": 2018
        }
        
        # Dostosuj filtry do kategorii
        if detection["category"] == "leczenie":
            filters["article_types"].append("Clinical Trial")
            filters["article_types"].append("Guideline")
        elif detection["category"] == "diagnostyka":
            filters["article_types"].append("Diagnostic Study")
        
        return {
            "detection": detection,
            "pubmed_query": pubmed_query,
            "clinicaltrials_condition": ct_condition,
            "clinicaltrials_intervention": ct_intervention,
            "polish_query": polish_query,
            "recommended_filters": filters
        }


# ============================================================================
# CLAUDE ENHANCEMENT (opcjonalne)
# ============================================================================

def enhance_with_claude(
    topic: str,
    detector: MedicalTermDetector = None
) -> Dict[str, Any]:
    """
    Używa Claude do lepszego wykrycia terminów i tłumaczenia.
    
    Fallback na lokalny detektor jeśli Claude niedostępny.
    """
    detector = detector or MedicalTermDetector()
    
    # Podstawowa detekcja
    result = detector.detect_medical_topic(topic)
    
    # Sprawdź czy Claude dostępny
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return result
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        prompt = f"""Jesteś ekspertem medycznym. Przeanalizuj temat artykułu i zwróć:
1. Czy to temat medyczny (YMYL Health)
2. Odpowiednie terminy MeSH (po angielsku)
3. Optymalne zapytanie PubMed

TEMAT: "{topic}"

Odpowiedz JSON:
{{
    "is_medical": true/false,
    "mesh_terms": ["term1", "term2"],
    "pubmed_query": "optimized query",
    "specialization": "cardiology/neurology/etc or null"
}}"""
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        import json
        text = response.content[0].text
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        data = json.loads(text.strip())
        
        # Merge with local detection
        result["mesh_suggestions"] = data.get("mesh_terms", result["mesh_suggestions"])
        result["english_query"] = data.get("pubmed_query", result["english_query"])
        if data.get("specialization"):
            result["specialization"] = data["specialization"]
        result["enhanced_by_claude"] = True
        
    except Exception as e:
        print(f"[MEDICAL_TERM] ⚠️ Claude enhancement failed: {e}")
        result["enhanced_by_claude"] = False
    
    return result


# ============================================================================
# SINGLETON & HELPERS
# ============================================================================

_detector = None


def get_medical_term_detector() -> MedicalTermDetector:
    """Zwraca singleton detektora."""
    global _detector
    if _detector is None:
        _detector = MedicalTermDetector()
    return _detector


def detect_medical_topic(topic: str, **kwargs) -> Dict[str, Any]:
    """Główna funkcja do wykrywania tematu medycznego."""
    return get_medical_term_detector().detect_medical_topic(topic, **kwargs)


def build_pubmed_query(topic: str, **kwargs) -> str:
    """Buduje zapytanie PubMed."""
    return get_medical_term_detector().build_pubmed_query(topic, **kwargs)


def get_search_strategy(topic: str) -> Dict[str, Any]:
    """Generuje pełną strategię wyszukiwania."""
    return get_medical_term_detector().get_search_strategy(topic)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "MedicalTermDetector",
    "MedicalTermConfig",
    "get_medical_term_detector",
    "detect_medical_topic",
    "build_pubmed_query",
    "get_search_strategy",
    "enhance_with_claude"
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 MEDICAL TERM DETECTOR v1.0 TEST")
    print("=" * 60)
    
    detector = MedicalTermDetector()
    
    test_topics = [
        "leczenie cukrzycy typu 2 metforminą",
        "objawy zawału serca",
        "szczepionka na covid skutki uboczne",
        "przepis na ciasto",  # Nie medyczny
        "dieta ketogeniczna a cukrzyca"
    ]
    
    for topic in test_topics:
        print(f"\n{'='*60}")
        print(f"📋 Topic: {topic}")
        print("="*60)
        
        result = detector.detect_medical_topic(topic)
        
        print(f"Is Medical: {'✅' if result['is_medical'] else '❌'}")
        print(f"Confidence: {result['confidence']}")
        print(f"Category: {result['category']}")
        print(f"Specialization: {result['specialization']}")
        print(f"English Query: {result['english_query']}")
        print(f"MeSH Suggestions: {result['mesh_suggestions']}")
        
        # Strategia wyszukiwania
        strategy = detector.get_search_strategy(topic)
        print(f"\nPubMed Query: {strategy['pubmed_query']}")
        print(f"CT Condition: {strategy['clinicaltrials_condition']}")
