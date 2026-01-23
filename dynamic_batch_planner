"""
🎯 DYNAMIC BATCH PLANNER v1.0
Token Budgeting Algorithm - inteligentne pakowanie H2 do batchy

Rozwiązuje problem "Frankensteina" przez:
1. Grupowanie krótkich sekcji (definicje, wstępy) → płynniejszy tekst
2. Izolowanie gęstych sekcji (procedury, tabele) → precyzja
3. Dynamiczne dostosowanie do złożoności treści

Autor: SEO Master API v36.2
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class SectionComplexity(Enum):
    """Poziom złożoności sekcji - wpływa na decyzję o łączeniu"""
    LIGHT = "light"       # Definicje, wstępy, krótkie info → można łączyć
    MEDIUM = "medium"     # Standardowe treści → zależy od długości
    HEAVY = "heavy"       # Procedury, tabele, gęste instrukcje → izolować


@dataclass
class H2Section:
    """Reprezentacja pojedynczej sekcji H2"""
    h2: str
    category: str = ""                    # np. "definition", "procedure", "effects"
    target_words: int = 400               # docelowa liczba słów
    assigned_keywords: List[str] = field(default_factory=list)
    assigned_entities: List[str] = field(default_factory=list)
    assigned_ngrams: List[str] = field(default_factory=list)
    complexity: SectionComplexity = SectionComplexity.MEDIUM
    guidance: str = ""                    # wskazówki dla GPT
    keyword_density_required: float = 0.0 # wymagana gęstość fraz
    
    def estimate_complexity(self) -> SectionComplexity:
        """Automatycznie oszacuj złożoność na podstawie cech sekcji"""
        h2_lower = self.h2.lower()
        
        # HEAVY indicators
        heavy_patterns = [
            "procedura", "krok po kroku", "instrukcja", "jak złożyć",
            "dokumenty", "wymagania", "formularz", "wniosek",
            "tabela", "zestawienie", "porównanie", "lista",
            "szczegółow", "kompletny przewodnik"
        ]
        
        # LIGHT indicators
        light_patterns = [
            "czym jest", "co to", "definicja", "pojęcie",
            "wprowadzenie", "wstęp", "historia", "geneza",
            "podsumowanie", "zakończenie", "faq", "pytania"
        ]
        
        for pattern in heavy_patterns:
            if pattern in h2_lower:
                return SectionComplexity.HEAVY
        
        for pattern in light_patterns:
            if pattern in h2_lower:
                return SectionComplexity.LIGHT
        
        # Dodatkowe heurystyki
        if len(self.assigned_keywords) > 8:
            return SectionComplexity.HEAVY
        if self.target_words > 600:
            return SectionComplexity.HEAVY
        if self.target_words < 250:
            return SectionComplexity.LIGHT
            
        return SectionComplexity.MEDIUM
    
    def to_dict(self) -> dict:
        return {
            "h2": self.h2,
            "category": self.category,
            "target_words": self.target_words,
            "assigned_keywords": self.assigned_keywords,
            "assigned_entities": self.assigned_entities,
            "assigned_ngrams": self.assigned_ngrams,
            "complexity": self.complexity.value,
            "guidance": self.guidance,
            "keyword_density_required": self.keyword_density_required
        }


@dataclass
class DynamicBatch:
    """Batch zawierający 1 lub więcej sekcji H2"""
    batch_number: int
    sections: List[H2Section] = field(default_factory=list)
    batch_type: str = "MULTI_SECTION"     # "INTRO", "SINGLE_SECTION", "MULTI_SECTION", "FINAL"
    total_target_words: Tuple[int, int] = (400, 600)  # (min, max)
    
    def add_section(self, section: H2Section):
        self.sections.append(section)
    
    def get_total_words(self) -> int:
        return sum(s.target_words for s in self.sections)
    
    def get_all_keywords(self) -> List[str]:
        keywords = []
        for s in self.sections:
            keywords.extend(s.assigned_keywords)
        return keywords
    
    def get_all_entities(self) -> List[str]:
        entities = []
        for s in self.sections:
            entities.extend(s.assigned_entities)
        return entities
    
    def to_dict(self) -> dict:
        return {
            "batch_number": self.batch_number,
            "batch_type": self.batch_type,
            "sections": [s.to_dict() for s in self.sections],
            "total_target_words": f"{self.total_target_words[0]}-{self.total_target_words[1]}",
            "section_count": len(self.sections),
            "h2_list": [s.h2 for s in self.sections]
        }


class DynamicBatchPlanner:
    """
    Token Budgeting Algorithm
    
    Pakuje sekcje H2 do batchy optymalizując:
    1. Spójność tematyczną (grupuje powiązane sekcje)
    2. Obciążenie tokenowe (max ~1000 słów na batch)
    3. Złożoność (izoluje gęste sekcje)
    """
    
    # Limity
    SOFT_WORD_LIMIT = 800    # preferowany max słów na batch
    HARD_WORD_LIMIT = 1200   # absolutny max (jakość spada powyżej)
    MIN_WORDS_FOR_BATCH = 200  # min słów żeby nie robić micro-batchy
    
    # Kategorie które można łączyć
    COMBINABLE_CATEGORIES = {
        ("definition", "types"),
        ("definition", "history"),
        ("introduction", "definition"),
        ("effects", "summary"),
        ("procedure", "documents"),  # ale tylko jeśli krótkie
        ("faq", "summary"),
    }
    
    def __init__(self, 
                 h2_structure: List[str],
                 semantic_plan: Optional[dict] = None,
                 s1_data: Optional[dict] = None,
                 target_article_length: int = 3500):
        """
        Args:
            h2_structure: Lista nagłówków H2
            semantic_plan: Plan semantyczny z przypisaniem fraz do H2
            s1_data: Dane z S1 (encje, n-gramy)
            target_article_length: Docelowa długość artykułu w słowach
        """
        self.h2_structure = h2_structure
        self.semantic_plan = semantic_plan or {}
        self.s1_data = s1_data or {}
        self.target_article_length = target_article_length
        
        # Zbuduj sekcje
        self.sections = self._build_sections()
        
    def _build_sections(self) -> List[H2Section]:
        """Konwertuj H2 structure na obiekty H2Section z metadanymi"""
        sections = []
        batch_plans = self.semantic_plan.get("batch_plans", [])
        
        # Domyślna długość per sekcja
        avg_words_per_h2 = self.target_article_length // max(len(self.h2_structure), 1)
        
        for i, h2 in enumerate(self.h2_structure):
            # Znajdź plan semantyczny dla tego H2
            h2_plan = None
            for bp in batch_plans:
                if bp.get("h2") == h2:
                    h2_plan = bp
                    break
            
            section = H2Section(
                h2=h2,
                category=self._detect_category(h2),
                target_words=h2_plan.get("target_words", avg_words_per_h2) if h2_plan else avg_words_per_h2,
                assigned_keywords=h2_plan.get("assigned_keywords", []) if h2_plan else [],
                assigned_entities=h2_plan.get("assigned_entities", []) if h2_plan else [],
                assigned_ngrams=h2_plan.get("assigned_ngrams", []) if h2_plan else [],
                guidance=h2_plan.get("guidance", "") if h2_plan else ""
            )
            
            # Auto-detect complexity
            section.complexity = section.estimate_complexity()
            
            # Adjust target words based on keyword count
            if len(section.assigned_keywords) > 6:
                section.target_words = max(section.target_words, 500)
            
            sections.append(section)
        
        return sections
    
    def _detect_category(self, h2: str) -> str:
        """Wykryj kategorię H2 na podstawie tekstu"""
        h2_lower = h2.lower()
        
        category_patterns = {
            "definition": ["czym jest", "co to", "definicja", "pojęcie", "znaczenie"],
            "types": ["rodzaje", "typy", "podział", "klasyfikacja", "formy"],
            "history": ["historia", "geneza", "rozwój", "ewolucja", "początki"],
            "procedure": ["procedura", "jak", "krok", "proces", "etapy", "instrukcja"],
            "documents": ["dokumenty", "wymagane", "wniosek", "formularz", "załączniki"],
            "conditions": ["przesłanki", "warunki", "wymogi", "kryteria", "kiedy"],
            "effects": ["skutki", "konsekwencje", "efekty", "następstwa", "rezultaty"],
            "costs": ["koszt", "cena", "opłaty", "ile kosztuje", "wydatki"],
            "faq": ["faq", "pytania", "najczęściej", "q&a"],
            "summary": ["podsumowanie", "zakończenie", "wnioski", "konkluzja"]
        }
        
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in h2_lower:
                    return category
        
        return "content"  # domyślna kategoria
    
    def _can_combine(self, section1: H2Section, section2: H2Section) -> bool:
        """Sprawdź czy dwie sekcje można połączyć w jeden batch"""
        # HEAVY sekcje nigdy nie łączymy
        if section1.complexity == SectionComplexity.HEAVY:
            return False
        if section2.complexity == SectionComplexity.HEAVY:
            return False
        
        # Sprawdź czy kategorie są kompatybilne
        cat_pair = (section1.category, section2.category)
        cat_pair_rev = (section2.category, section1.category)
        
        if cat_pair in self.COMBINABLE_CATEGORIES or cat_pair_rev in self.COMBINABLE_CATEGORIES:
            return True
        
        # Dwie LIGHT sekcje zawsze można łączyć
        if section1.complexity == SectionComplexity.LIGHT and section2.complexity == SectionComplexity.LIGHT:
            return True
        
        # MEDIUM + LIGHT można łączyć jeśli suma < SOFT_LIMIT
        if section1.target_words + section2.target_words <= self.SOFT_WORD_LIMIT:
            return True
        
        return False
    
    def plan_batches(self) -> List[DynamicBatch]:
        """
        Główny algorytm Token Budgeting
        
        Returns:
            Lista DynamicBatch z optymalnym rozkładem sekcji
        """
        if not self.sections:
            return []
        
        batches = []
        current_batch = DynamicBatch(batch_number=1)
        
        # Pierwszy batch to zawsze INTRO (bez H2 lub z pierwszym H2)
        # Traktujemy intro jako specjalny przypadek
        intro_batch = DynamicBatch(
            batch_number=1,
            batch_type="INTRO",
            total_target_words=(300, 500)
        )
        
        # Jeśli pierwszy H2 to definicja, włącz go do intro
        first_section = self.sections[0]
        start_idx = 0
        
        if first_section.complexity == SectionComplexity.LIGHT and first_section.target_words <= 300:
            intro_batch.add_section(first_section)
            intro_batch.total_target_words = (400, 600)
            start_idx = 1
        
        batches.append(intro_batch)
        
        # Przetwarzaj pozostałe sekcje
        current_batch = DynamicBatch(batch_number=2)
        
        for i, section in enumerate(self.sections[start_idx:], start=start_idx):
            # Czy możemy dodać tę sekcję do obecnego batcha?
            potential_words = current_batch.get_total_words() + section.target_words
            
            can_add = False
            if not current_batch.sections:
                # Pusty batch - zawsze dodaj
                can_add = True
            elif potential_words <= self.HARD_WORD_LIMIT:
                # Sprawdź czy możemy połączyć
                last_section = current_batch.sections[-1]
                if self._can_combine(last_section, section) and potential_words <= self.SOFT_WORD_LIMIT:
                    can_add = True
            
            if can_add:
                current_batch.add_section(section)
            else:
                # Zamknij obecny batch i zacznij nowy
                if current_batch.sections:
                    self._finalize_batch(current_batch)
                    batches.append(current_batch)
                
                current_batch = DynamicBatch(batch_number=len(batches) + 1)
                current_batch.add_section(section)
        
        # Dodaj ostatni batch
        if current_batch.sections:
            # Sprawdź czy to FINAL
            last_section = current_batch.sections[-1]
            if last_section.category in ["summary", "faq"] or "podsumowanie" in last_section.h2.lower():
                current_batch.batch_type = "FINAL"
            
            self._finalize_batch(current_batch)
            batches.append(current_batch)
        
        return batches
    
    def _finalize_batch(self, batch: DynamicBatch):
        """Ustaw finalne parametry batcha"""
        total_words = batch.get_total_words()
        
        # Ustaw typ batcha
        if len(batch.sections) == 1:
            batch.batch_type = "SINGLE_SECTION"
        else:
            batch.batch_type = "MULTI_SECTION"
        
        # Ustaw target words z marginesem
        margin = int(total_words * 0.15)
        batch.total_target_words = (
            max(200, total_words - margin),
            total_words + margin
        )
    
    def get_batch_plan_dict(self) -> dict:
        """Zwróć plan batchy jako dict (do zapisu w Firestore)"""
        batches = self.plan_batches()
        
        return {
            "algorithm": "token_budgeting_v1",
            "total_batches": len(batches),
            "total_h2_sections": len(self.sections),
            "target_article_length": self.target_article_length,
            "batches": [b.to_dict() for b in batches],
            "section_distribution": {
                "light": sum(1 for s in self.sections if s.complexity == SectionComplexity.LIGHT),
                "medium": sum(1 for s in self.sections if s.complexity == SectionComplexity.MEDIUM),
                "heavy": sum(1 for s in self.sections if s.complexity == SectionComplexity.HEAVY)
            }
        }


def create_dynamic_batch_plan(
    h2_structure: List[str],
    semantic_plan: Optional[dict] = None,
    s1_data: Optional[dict] = None,
    target_length: int = 3500
) -> dict:
    """
    Główna funkcja do tworzenia dynamicznego planu batchy.
    
    Używana w create_project() zamiast sztywnego podziału 1 H2 = 1 batch.
    
    Args:
        h2_structure: Lista nagłówków H2
        semantic_plan: Plan semantyczny (opcjonalny)
        s1_data: Dane z S1 (opcjonalny)
        target_length: Docelowa długość artykułu
        
    Returns:
        Dict z planem batchy do zapisu w Firestore
    """
    planner = DynamicBatchPlanner(
        h2_structure=h2_structure,
        semantic_plan=semantic_plan,
        s1_data=s1_data,
        target_article_length=target_length
    )
    
    return planner.get_batch_plan_dict()


def get_batch_sections_for_pre_batch(
    dynamic_batch_plan: dict,
    current_batch_num: int
) -> dict:
    """
    Pobierz sekcje dla konkretnego batcha do użycia w pre_batch_info.
    
    Returns:
        Dict w formacie dla response API:
        {
            "batch_type": "MULTI_SECTION",
            "sections": [
                {"h2": "...", "guidance": "...", "target_length": "200-300"},
                ...
            ],
            "total_target_words": "400-600"
        }
    """
    batches = dynamic_batch_plan.get("batches", [])
    
    for batch in batches:
        if batch.get("batch_number") == current_batch_num:
            sections_formatted = []
            
            for section in batch.get("sections", []):
                sections_formatted.append({
                    "h2": section.get("h2", ""),
                    "category": section.get("category", ""),
                    "guidance": section.get("guidance", ""),
                    "target_length": f"{int(section.get('target_words', 400) * 0.85)}-{int(section.get('target_words', 400) * 1.15)}",
                    "assigned_keywords": section.get("assigned_keywords", []),
                    "assigned_entities": section.get("assigned_entities", []),
                    "complexity": section.get("complexity", "medium")
                })
            
            return {
                "batch_type": batch.get("batch_type", "CONTENT"),
                "sections": sections_formatted,
                "total_target_words": batch.get("total_target_words", "400-600"),
                "section_count": len(sections_formatted),
                "h2_list": batch.get("h2_list", [])
            }
    
    # Fallback jeśli nie znaleziono batcha
    return {
        "batch_type": "CONTENT",
        "sections": [],
        "total_target_words": "400-600",
        "section_count": 0,
        "h2_list": []
    }


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    # Przykładowa struktura H2
    h2_structure = [
        "Czym jest ubezwłasnowolnienie",
        "Rodzaje ubezwłasnowolnienia",
        "Przesłanki ubezwłasnowolnienia",
        "Procedura sądowa krok po kroku",
        "Wymagane dokumenty",
        "Skutki prawne ubezwłasnowolnienia",
        "Najczęściej zadawane pytania (FAQ)"
    ]
    
    # Utwórz plan
    plan = create_dynamic_batch_plan(
        h2_structure=h2_structure,
        target_length=4000
    )
    
    print("=== DYNAMIC BATCH PLAN ===")
    print(f"Total batches: {plan['total_batches']}")
    print(f"Total H2 sections: {plan['total_h2_sections']}")
    print(f"Section distribution: {plan['section_distribution']}")
    print()
    
    for batch in plan["batches"]:
        print(f"BATCH {batch['batch_number']} ({batch['batch_type']}):")
        print(f"  Target words: {batch['total_target_words']}")
        print(f"  Sections: {batch['h2_list']}")
        print()
