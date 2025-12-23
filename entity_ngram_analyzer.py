"""
===============================================================================
🧠 ENTITY N-GRAM ANALYZER v23.0 - Hybrid Semantic Analysis
===============================================================================
Łączy dwa podejścia:
1. N-gramy (statystyczne) - nadal ważne dla Polish SEO
2. Entity Extraction (semantyczne) - Knowledge Graph alignment

+ E-E-A-T Enhancement dla Google 2024+
===============================================================================
"""

import re
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import spacy

# ================================================================
# 🧠 Współdzielony model spaCy z NER
# ================================================================
try:
    from shared_nlp import get_nlp
    nlp = get_nlp()
    print("[ENTITY_ANALYZER] ✅ Używam współdzielonego modelu spaCy")
except ImportError:
    # Fallback - ładuj lokalnie
    import spacy
    try:
        nlp = spacy.load("pl_core_news_md")
        print("[ENTITY_ANALYZER] ⚠️ Załadowano lokalny model pl_core_news_md")
    except OSError:
        from spacy.cli import download
        download("pl_core_news_md")
        nlp = spacy.load("pl_core_news_md")


# ================================================================
# 📊 KONFIGURACJA
# ================================================================
@dataclass
class EntityConfig:
    """Konfiguracja analizy encji."""
    
    # Wagi dla hybrid score
    NGRAM_WEIGHT = 0.5       # N-gramy nadal ważne
    ENTITY_WEIGHT = 0.3      # Encje dla Knowledge Graph
    CONTEXT_WEIGHT = 0.2     # Kontekst użycia
    
    # Entity types ważne dla SEO
    PRIORITY_ENTITY_TYPES = [
        "persName",    # Osoby (eksperci, autorzy)
        "orgName",     # Organizacje (firmy, instytucje)
        "placeName",   # Miejsca (lokalizacje)
        "date",        # Daty (aktualność)
        "geogName",    # Geografia
    ]
    
    # Polish NER labels mapping (spaCy pl_core_news)
    SPACY_LABEL_MAP = {
        "persName": ["PER", "PERSON"],
        "orgName": ["ORG", "ORGANIZATION"], 
        "placeName": ["LOC", "GPE", "LOCATION"],
        "date": ["DATE", "TIME"],
        "geogName": ["LOC", "GPE"],
        "money": ["MONEY"],
        "percent": ["PERCENT"],
    }
    
    # E-E-A-T keywords po polsku
    EXPERTISE_SIGNALS = [
        "ekspert", "specjalista", "doktor", "profesor", "mgr", "inż.",
        "certyfikowany", "licencjonowany", "doświadczenie", "praktyka",
        "wieloletni", "autoryzowany", "akredytowany"
    ]
    
    AUTHORITY_SIGNALS = [
        "według", "zgodnie z", "na podstawie", "badania pokazują",
        "statystyki wskazują", "dane z", "raport", "analiza",
        "ministerstwo", "urząd", "instytut", "uniwersytet"
    ]
    
    TRUST_SIGNALS = [
        "źródło:", "dane:", "stan na", "aktualizacja",
        "dz.u.", "art.", "§", "ustawa", "rozporządzenie",
        "potwierdzone", "zweryfikowane", "oficjalny"
    ]


# ================================================================
# 📦 STRUKTURY DANYCH
# ================================================================
@dataclass
class ExtractedEntity:
    """Wyekstrahowana encja z tekstu."""
    text: str
    label: str
    normalized_label: str
    start_char: int
    end_char: int
    context: str  # Tekst wokół encji
    frequency: int = 1
    importance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label,
            "normalized_label": self.normalized_label,
            "context": self.context,
            "frequency": self.frequency,
            "importance_score": round(self.importance_score, 3)
        }


@dataclass
class HybridNgram:
    """N-gram wzbogacony o informacje o encjach."""
    ngram: str
    frequency: int
    site_distribution: str
    ngram_weight: float
    contains_entity: bool = False
    entity_type: Optional[str] = None
    entity_boost: float = 0.0
    hybrid_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ngram": self.ngram,
            "frequency": self.frequency,
            "site_distribution": self.site_distribution,
            "weight": round(self.hybrid_score, 4),
            "contains_entity": self.contains_entity,
            "entity_type": self.entity_type,
            "components": {
                "ngram_weight": round(self.ngram_weight, 4),
                "entity_boost": round(self.entity_boost, 4)
            }
        }


@dataclass
class EEATAnalysis:
    """Analiza sygnałów E-E-A-T w tekście."""
    expertise_score: float = 0.0
    authority_score: float = 0.0
    trust_score: float = 0.0
    overall_score: float = 0.0
    signals_found: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "scores": {
                "expertise": round(self.expertise_score, 2),
                "authority": round(self.authority_score, 2),
                "trust": round(self.trust_score, 2),
                "overall": round(self.overall_score, 2)
            },
            "signals_found": self.signals_found,
            "recommendations": self.recommendations,
            "status": "GOOD" if self.overall_score >= 0.6 else "NEEDS_IMPROVEMENT"
        }


# ================================================================
# 🔧 ENTITY EXTRACTION
# ================================================================
def extract_entities(text: str, context_window: int = 50) -> List[ExtractedEntity]:
    """
    Wyciąga Named Entities z tekstu używając spaCy NER.
    
    Args:
        text: Tekst do analizy
        context_window: Ile znaków kontekstu wokół encji
    
    Returns:
        Lista ExtractedEntity
    """
    if not text or len(text.strip()) < 10:
        return []
    
    doc = nlp(text[:50000])  # Limit dla wydajności
    entities = []
    entity_counts = Counter()
    
    for ent in doc.ents:
        # Pobierz kontekst
        start = max(0, ent.start_char - context_window)
        end = min(len(text), ent.end_char + context_window)
        context = text[start:end].replace('\n', ' ').strip()
        
        # Normalizuj label
        normalized_label = normalize_entity_label(ent.label_)
        
        entity_counts[ent.text.lower()] += 1
        
        entities.append(ExtractedEntity(
            text=ent.text,
            label=ent.label_,
            normalized_label=normalized_label,
            start_char=ent.start_char,
            end_char=ent.end_char,
            context=context,
            frequency=1
        ))
    
    # Zaktualizuj frequency
    for entity in entities:
        entity.frequency = entity_counts[entity.text.lower()]
        entity.importance_score = calculate_entity_importance(entity)
    
    return entities


def normalize_entity_label(spacy_label: str) -> str:
    """Normalizuje label spaCy do standardowego formatu."""
    config = EntityConfig()
    
    for normalized, spacy_labels in config.SPACY_LABEL_MAP.items():
        if spacy_label in spacy_labels:
            return normalized
    
    return spacy_label.lower()


def calculate_entity_importance(entity: ExtractedEntity) -> float:
    """
    Oblicza ważność encji dla SEO.
    
    Czynniki:
    - Typ encji (osoby/organizacje ważniejsze)
    - Częstość występowania
    - Długość nazwy (dłuższe = bardziej specyficzne)
    """
    config = EntityConfig()
    score = 0.0
    
    # Typ encji
    if entity.normalized_label in config.PRIORITY_ENTITY_TYPES:
        score += 0.3
    
    # Częstość (log scale)
    import math
    score += min(0.3, math.log(entity.frequency + 1) * 0.1)
    
    # Długość (2-4 słowa = optymalne)
    word_count = len(entity.text.split())
    if 2 <= word_count <= 4:
        score += 0.2
    elif word_count == 1:
        score += 0.1
    
    # Czy zawiera wielką literę (proper noun)
    if entity.text[0].isupper():
        score += 0.1
    
    return min(1.0, score)


# ================================================================
# 🔧 HYBRID N-GRAM ANALYSIS
# ================================================================
def analyze_hybrid_ngrams(
    text: str,
    entities: List[ExtractedEntity],
    existing_ngrams: List[Dict] = None
) -> List[HybridNgram]:
    """
    Łączy analizę n-gramów z informacjami o encjach.
    
    N-gramy zawierające encje dostają boost.
    """
    config = EntityConfig()
    
    # Jeśli mamy już n-gramy z S1, wzbogać je
    if existing_ngrams:
        return enrich_existing_ngrams(existing_ngrams, entities)
    
    # W przeciwnym razie wygeneruj nowe
    return generate_hybrid_ngrams(text, entities)


def enrich_existing_ngrams(
    ngrams: List[Dict],
    entities: List[ExtractedEntity]
) -> List[HybridNgram]:
    """
    Wzbogaca istniejące n-gramy o informacje o encjach.
    """
    config = EntityConfig()
    entity_texts = {e.text.lower() for e in entities}
    entity_map = {e.text.lower(): e for e in entities}
    
    hybrid_ngrams = []
    
    for ng_data in ngrams:
        ngram_text = ng_data.get("ngram", "").lower()
        ngram_weight = ng_data.get("weight", 0.5)
        
        # Sprawdź czy n-gram zawiera encję
        contains_entity = False
        entity_type = None
        entity_boost = 0.0
        
        for entity_text in entity_texts:
            if entity_text in ngram_text or ngram_text in entity_text:
                contains_entity = True
                entity = entity_map.get(entity_text)
                if entity:
                    entity_type = entity.normalized_label
                    # Boost zależny od typu encji
                    if entity.normalized_label in config.PRIORITY_ENTITY_TYPES:
                        entity_boost = 0.15
                    else:
                        entity_boost = 0.05
                break
        
        # Oblicz hybrid score
        hybrid_score = (
            ngram_weight * config.NGRAM_WEIGHT +
            entity_boost * config.ENTITY_WEIGHT +
            (0.1 if contains_entity else 0) * config.CONTEXT_WEIGHT
        )
        
        # Normalizuj do 0-1
        hybrid_score = min(1.0, hybrid_score / (config.NGRAM_WEIGHT + config.ENTITY_WEIGHT + config.CONTEXT_WEIGHT))
        
        hybrid_ngrams.append(HybridNgram(
            ngram=ng_data.get("ngram", ""),
            frequency=ng_data.get("freq", 1),
            site_distribution=ng_data.get("site_distribution", "0/0"),
            ngram_weight=ngram_weight,
            contains_entity=contains_entity,
            entity_type=entity_type,
            entity_boost=entity_boost,
            hybrid_score=hybrid_score
        ))
    
    # Sortuj po hybrid_score
    hybrid_ngrams.sort(key=lambda x: x.hybrid_score, reverse=True)
    
    return hybrid_ngrams


def generate_hybrid_ngrams(
    text: str,
    entities: List[ExtractedEntity]
) -> List[HybridNgram]:
    """
    Generuje n-gramy z uwzględnieniem encji.
    """
    config = EntityConfig()
    
    # Standardowa ekstrakcja n-gramów
    doc = nlp(text.lower()[:30000])
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    
    ngram_freqs = Counter()
    for n in range(2, 5):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngram_freqs[ngram] += 1
    
    # Filtruj i wzbogać
    entity_texts = {e.text.lower() for e in entities}
    entity_map = {e.text.lower(): e for e in entities}
    
    max_freq = max(ngram_freqs.values()) if ngram_freqs else 1
    hybrid_ngrams = []
    
    for ngram, freq in ngram_freqs.items():
        if freq < 2:
            continue
        
        ngram_weight = freq / max_freq
        
        # Entity check
        contains_entity = False
        entity_type = None
        entity_boost = 0.0
        
        for entity_text in entity_texts:
            if entity_text in ngram or ngram in entity_text:
                contains_entity = True
                entity = entity_map.get(entity_text)
                if entity:
                    entity_type = entity.normalized_label
                    entity_boost = 0.15 if entity.normalized_label in config.PRIORITY_ENTITY_TYPES else 0.05
                break
        
        hybrid_score = ngram_weight * 0.7 + entity_boost * 0.3
        
        hybrid_ngrams.append(HybridNgram(
            ngram=ngram,
            frequency=freq,
            site_distribution="local",
            ngram_weight=ngram_weight,
            contains_entity=contains_entity,
            entity_type=entity_type,
            entity_boost=entity_boost,
            hybrid_score=min(1.0, hybrid_score)
        ))
    
    hybrid_ngrams.sort(key=lambda x: x.hybrid_score, reverse=True)
    return hybrid_ngrams[:50]


# ================================================================
# 🏆 E-E-A-T ANALYSIS
# ================================================================
def analyze_eeat(text: str, entities: List[ExtractedEntity] = None) -> EEATAnalysis:
    """
    Analizuje sygnały E-E-A-T w tekście.
    
    E-E-A-T (Google 2023+):
    - Experience (pomijamy - trudne do automatycznej oceny)
    - Expertise - sygnały ekspertyzy
    - Authoritativeness - sygnały autorytetu
    - Trustworthiness - sygnały wiarygodności
    """
    config = EntityConfig()
    text_lower = text.lower()
    
    signals_found = {
        "expertise": [],
        "authority": [],
        "trust": []
    }
    
    # ================================================================
    # EXPERTISE - szukaj sygnałów ekspertyzy
    # ================================================================
    expertise_count = 0
    for signal in config.EXPERTISE_SIGNALS:
        if signal in text_lower:
            expertise_count += 1
            # Znajdź kontekst
            idx = text_lower.find(signal)
            context = text[max(0, idx-30):min(len(text), idx+len(signal)+30)]
            signals_found["expertise"].append(f"{signal}: ...{context}...")
    
    # Bonus za encje typu PERSON (eksperci)
    if entities:
        person_entities = [e for e in entities if e.normalized_label == "persName"]
        expertise_count += len(person_entities) * 0.5
        for pe in person_entities[:3]:
            signals_found["expertise"].append(f"Osoba: {pe.text}")
    
    expertise_score = min(1.0, expertise_count / 5)  # Normalizuj
    
    # ================================================================
    # AUTHORITY - szukaj sygnałów autorytetu
    # ================================================================
    authority_count = 0
    for signal in config.AUTHORITY_SIGNALS:
        if signal in text_lower:
            authority_count += 1
            idx = text_lower.find(signal)
            context = text[max(0, idx-20):min(len(text), idx+len(signal)+50)]
            signals_found["authority"].append(f"{signal}: ...{context}...")
    
    # Bonus za encje typu ORG (instytucje)
    if entities:
        org_entities = [e for e in entities if e.normalized_label == "orgName"]
        authority_count += len(org_entities) * 0.3
        for oe in org_entities[:3]:
            signals_found["authority"].append(f"Organizacja: {oe.text}")
    
    authority_score = min(1.0, authority_count / 6)
    
    # ================================================================
    # TRUST - szukaj sygnałów wiarygodności
    # ================================================================
    trust_count = 0
    for signal in config.TRUST_SIGNALS:
        if signal in text_lower:
            trust_count += 1
            idx = text_lower.find(signal)
            context = text[max(0, idx-20):min(len(text), idx+len(signal)+50)]
            signals_found["trust"].append(f"{signal}: ...{context}...")
    
    # Bonus za daty (aktualność)
    if entities:
        date_entities = [e for e in entities if e.normalized_label == "date"]
        # Sprawdź czy są aktualne daty (2023, 2024, 2025)
        recent_dates = [d for d in date_entities if any(y in d.text for y in ["2023", "2024", "2025"])]
        trust_count += len(recent_dates) * 0.5
        for rd in recent_dates[:2]:
            signals_found["trust"].append(f"Aktualna data: {rd.text}")
    
    trust_score = min(1.0, trust_count / 5)
    
    # ================================================================
    # OVERALL SCORE
    # ================================================================
    overall_score = (expertise_score * 0.35 + authority_score * 0.35 + trust_score * 0.30)
    
    # ================================================================
    # RECOMMENDATIONS
    # ================================================================
    recommendations = []
    
    if expertise_score < 0.4:
        recommendations.append(
            "EXPERTISE: Dodaj sygnały ekspertyzy - wspomnij o doświadczeniu, "
            "kwalifikacjach, użyj terminologii branżowej. Np. 'Jako praktykujący prawnik...' "
            "lub 'Z wieloletnim doświadczeniem w...'"
        )
    
    if authority_score < 0.4:
        recommendations.append(
            "AUTHORITY: Dodaj cytaty ze źródeł - 'Według danych GUS...', "
            "'Badania przeprowadzone przez...', 'Zgodnie z art. X ustawy...'"
        )
    
    if trust_score < 0.4:
        recommendations.append(
            "TRUST: Zwiększ wiarygodność - dodaj aktualne daty, odwołania do przepisów, "
            "konkretne statystyki ze źródłem. Np. 'Stan prawny na 2024 rok' lub "
            "'Dane Ministerstwa Sprawiedliwości (2024)'"
        )
    
    if not recommendations:
        recommendations.append("E-E-A-T OK - treść zawiera wystarczające sygnały ekspertyzy i wiarygodności.")
    
    return EEATAnalysis(
        expertise_score=expertise_score,
        authority_score=authority_score,
        trust_score=trust_score,
        overall_score=overall_score,
        signals_found=signals_found,
        recommendations=recommendations
    )


# ================================================================
# 🎯 GŁÓWNA FUNKCJA ANALIZY
# ================================================================
def analyze_content_semantics(
    text: str,
    s1_ngrams: List[Dict] = None,
    main_keyword: str = None
) -> Dict[str, Any]:
    """
    Główna funkcja - wykonuje pełną analizę semantyczną.
    
    Args:
        text: Tekst do analizy
        s1_ngrams: N-gramy z analizy S1 (opcjonalne)
        main_keyword: Fraza główna (opcjonalne)
    
    Returns:
        Dict z pełną analizą
    """
    result = {
        "entities": [],
        "hybrid_ngrams": [],
        "eeat_analysis": {},
        "entity_summary": {},
        "recommendations": []
    }
    
    if not text or len(text.strip()) < 50:
        result["error"] = "Text too short for analysis"
        return result
    
    # 1. Extract entities
    entities = extract_entities(text)
    result["entities"] = [e.to_dict() for e in entities[:30]]
    
    # Entity summary
    entity_types = Counter(e.normalized_label for e in entities)
    result["entity_summary"] = {
        "total_entities": len(entities),
        "unique_entities": len(set(e.text.lower() for e in entities)),
        "by_type": dict(entity_types),
        "top_entities": [
            {"text": e.text, "type": e.normalized_label, "score": round(e.importance_score, 2)}
            for e in sorted(entities, key=lambda x: x.importance_score, reverse=True)[:10]
        ]
    }
    
    # 2. Hybrid n-gram analysis
    hybrid_ngrams = analyze_hybrid_ngrams(text, entities, s1_ngrams)
    result["hybrid_ngrams"] = [ng.to_dict() for ng in hybrid_ngrams[:30]]
    
    # N-gram stats
    entity_ngrams = [ng for ng in hybrid_ngrams if ng.contains_entity]
    result["ngram_stats"] = {
        "total_ngrams": len(hybrid_ngrams),
        "entity_enriched": len(entity_ngrams),
        "entity_enriched_percent": round(len(entity_ngrams) / max(1, len(hybrid_ngrams)) * 100, 1),
        "avg_hybrid_score": round(
            sum(ng.hybrid_score for ng in hybrid_ngrams) / max(1, len(hybrid_ngrams)), 3
        )
    }
    
    # 3. E-E-A-T analysis
    eeat = analyze_eeat(text, entities)
    result["eeat_analysis"] = eeat.to_dict()
    
    # 4. Combined recommendations
    recommendations = []
    
    # Entity recommendations
    if result["entity_summary"]["total_entities"] < 5:
        recommendations.append(
            "ENTITIES: Dodaj więcej konkretnych nazw - osób, organizacji, miejsc, dat. "
            "To pomaga Google zrozumieć kontekst i powiązać z Knowledge Graph."
        )
    
    # N-gram recommendations
    if result["ngram_stats"]["entity_enriched_percent"] < 20:
        recommendations.append(
            "N-GRAMY: Wiele n-gramów nie zawiera encji. Rozważ naturalne wplecenie "
            "konkretnych nazw (instytucji, ekspertów) w kluczowe frazy."
        )
    
    # E-E-A-T recommendations
    recommendations.extend(eeat.recommendations)
    
    result["recommendations"] = recommendations
    
    return result


# ================================================================
# 🔧 HELPER: Generate E-E-A-T Enhanced Prompt
# ================================================================
def generate_eeat_prompt_enhancement(topic: str, current_eeat: EEATAnalysis = None) -> str:
    """
    Generuje dodatek do prompta GPT wzmacniający E-E-A-T.
    """
    prompt_parts = []
    
    prompt_parts.append("""
⭐ E-E-A-T ENHANCEMENT (Google 2024+):
Wzmocnij sygnały ekspertyzy, autorytetu i wiarygodności:
""")
    
    if current_eeat and current_eeat.expertise_score < 0.5:
        prompt_parts.append("""
📚 EXPERTISE - Dodaj sygnały ekspertyzy:
   • Użyj precyzyjnej terminologii branżowej
   • Wspomnij o praktyce/doświadczeniu (np. "W praktyce prawniczej...")
   • Użyj konstrukcji pokazujących wiedzę (np. "Kluczowym aspektem jest...")
""")
    
    if current_eeat and current_eeat.authority_score < 0.5:
        prompt_parts.append("""
🏛️ AUTHORITY - Dodaj cytaty i źródła:
   • "Zgodnie z art. X ustawy..."
   • "Według danych [instytucja]..."
   • "Badania przeprowadzone przez [organizacja] pokazują..."
   • Wspominaj uznane instytucje (ministerstwa, uniwersytety, instytuty)
""")
    
    if current_eeat and current_eeat.trust_score < 0.5:
        prompt_parts.append("""
✅ TRUST - Zwiększ wiarygodność:
   • Dodaj aktualne daty (stan na 2024)
   • Podawaj konkretne liczby ze źródłem
   • Odwołuj się do przepisów prawnych
   • Używaj precyzyjnych danych zamiast ogólników
""")
    
    prompt_parts.append("""
💡 PRZYKŁAD DOBREGO TEKSTU z E-E-A-T:
"Pozew o rozwód to pismo procesowe inicjujące postępowanie rozwodowe przed sądem okręgowym 
(art. 56 Kodeksu rodzinnego i opiekuńczego). Według danych Ministerstwa Sprawiedliwości (2024), 
rocznie do polskich sądów wpływa około 65 000 pozwów rozwodowych. W mojej wieloletniej praktyce 
jako adwokat specjalizujący się w prawie rodzinnym obserwuję, że prawidłowo przygotowany pozew 
znacząco przyspiesza postępowanie."

❌ PRZYKŁAD ZŁEGO TEKSTU (brak E-E-A-T):
"Rozwód to trudna sprawa. Wiele osób się rozwodzi. Warto wiedzieć jak to zrobić dobrze."
""")
    
    return "\n".join(prompt_parts)


# ================================================================
# 🔧 HELPER: Check Entity Coverage
# ================================================================
def check_entity_coverage(
    generated_text: str,
    competitor_entities: List[Dict]
) -> Dict[str, Any]:
    """
    Sprawdza czy wygenerowany tekst pokrywa kluczowe encje z konkurencji.
    """
    our_entities = extract_entities(generated_text)
    our_entity_texts = {e.text.lower() for e in our_entities}
    
    # Competitor entities (już wyekstrahowane)
    comp_entity_texts = {e.get("text", "").lower() for e in competitor_entities}
    
    # Coverage
    matched = our_entity_texts & comp_entity_texts
    missing = comp_entity_texts - our_entity_texts
    unique_ours = our_entity_texts - comp_entity_texts
    
    coverage = len(matched) / max(1, len(comp_entity_texts))
    
    return {
        "coverage": round(coverage, 2),
        "matched_entities": list(matched)[:10],
        "missing_entities": list(missing)[:10],
        "our_unique_entities": list(unique_ours)[:10],
        "recommendation": (
            "OK - good entity coverage" if coverage >= 0.5 
            else f"Consider adding: {', '.join(list(missing)[:5])}"
        )
    }
