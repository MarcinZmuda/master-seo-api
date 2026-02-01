"""
===============================================================================
🔬 PHRASE HIERARCHY MODULE v1.0 - BRAJEN SEO Integration
===============================================================================
Globalny moduł analizy hierarchii fraz kluczowych.

FUNKCJE:
1. analyze_phrase_hierarchy() - analiza po S1/create project
2. get_effective_targets() - efektywne targety z uwzględnieniem rozszerzeń
3. get_batch_hierarchy_context() - kontekst hierarchii dla pre_batch_info
4. format_hierarchy_for_agent() - instrukcje dla agenta

INTEGRACJA:
- Wywołanie po /api/project/create → zapisuje hierarchię w projekcie
- Rozszerzenie /api/project/{id}/pre_batch_info → dodaje kontekst hierarchii
- Aktualizacja promptu agenta → świadomość hierarchii

===============================================================================
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class HierarchyConfig:
    """Konfiguracja modułu hierarchii."""
    
    # Minimalna liczba rozszerzeń żeby fraza była rdzeniem
    MIN_EXTENSIONS_FOR_ROOT: int = 2
    
    # Maksymalna długość rdzenia (w słowach)
    MAX_ROOT_WORDS: int = 4
    
    # Próg dla "extensions_sufficient" - ile % targetu pokrywają rozszerzenia
    EXTENSIONS_SUFFICIENT_THRESHOLD: float = 0.8
    
    # Czy uwzględniać formy fleksyjne
    USE_INFLECTION: bool = True


CONFIG = HierarchyConfig()


# ============================================================================
# FLEKSJA POLSKA
# ============================================================================

def get_phrase_variants(phrase: str) -> Set[str]:
    """
    Generuje warianty fleksyjne frazy.
    
    Obsługuje typowe polskie przypadki:
    - zespół X → zespołu X, zespołem X, zespole X
    - leczenie → leczenia, leczeniu, leczeniem
    """
    phrase_lower = phrase.lower().strip()
    variants = {phrase_lower}
    
    # Specjalne przypadki dla "zespół X"
    if phrase_lower.startswith("zespół "):
        suffix = phrase_lower[7:]
        variants.update([
            f"zespół {suffix}",
            f"zespołu {suffix}",
            f"zespołowi {suffix}",
            f"zespołem {suffix}",
            f"zespole {suffix}",
        ])
    
    # Specjalne dla "X turnera" / "turnera"
    if "turnera" in phrase_lower:
        # Już w dopełniaczu, dodaj inne przypadki głównego słowa
        pass
    
    # Formy z "-enie" (leczenie, rozpoznanie)
    if phrase_lower.endswith("enie"):
        base = phrase_lower[:-4]
        variants.update([
            f"{base}enie",
            f"{base}enia",
            f"{base}eniu",
            f"{base}eniem",
        ])
    
    # Formy z "-owanie" (diagnozowanie)
    if phrase_lower.endswith("owanie"):
        base = phrase_lower[:-6]
        variants.update([
            f"{base}owanie",
            f"{base}owania",
            f"{base}owaniu",
            f"{base}owaniem",
        ])
    
    # Formy z "-yka/-ika" (diagnostyka, genetyka)
    if phrase_lower.endswith("yka") or phrase_lower.endswith("ika"):
        base = phrase_lower[:-1]
        variants.update([
            phrase_lower,
            f"{base}i",
            f"{base}ę",
            f"{base}ą",
        ])
    
    # Normalizacja myślników
    for v in list(variants):
        variants.add(v.replace(' – ', ' '))
        variants.add(v.replace(' - ', ' '))
        variants.add(v.replace('–', ' '))
        variants.add(v.replace('-', ' '))
    
    return variants


def normalize_phrase(phrase: str) -> str:
    """Normalizuje frazę do porównań."""
    p = phrase.lower().strip()
    p = re.sub(r'[–—-]', ' ', p)
    p = re.sub(r'\s+', ' ', p)
    return p


def phrase_contains_root(phrase: str, root: str, use_inflection: bool = True) -> bool:
    """
    Sprawdza czy fraza zawiera rdzeń (w dowolnej formie fleksyjnej).
    
    Args:
        phrase: Dłuższa fraza do sprawdzenia
        root: Rdzeń do znalezienia
        use_inflection: Czy używać form fleksyjnych
        
    Returns:
        True jeśli fraza zawiera rdzeń
    """
    phrase_norm = normalize_phrase(phrase)
    root_norm = normalize_phrase(root)
    
    # Prosta zawartość
    if root_norm in phrase_norm:
        return True
    
    # Sprawdzenie z fleksją
    if use_inflection:
        root_variants = get_phrase_variants(root)
        for variant in root_variants:
            if variant in phrase_norm:
                return True
    
    # Sprawdzenie word-by-word dla fraz wielowyrazowych
    root_words = root_norm.split()
    if len(root_words) >= 2:
        # Sprawdź czy główne słowo (ostatnie) występuje
        last_word = root_words[-1]
        if last_word in phrase_norm:
            # Sprawdź czy pierwsze słowo (lub jego odmiana) też jest
            first_word = root_words[0]
            if use_inflection:
                first_variants = get_phrase_variants(first_word)
                for fv in first_variants:
                    if fv in phrase_norm:
                        return True
            elif first_word in phrase_norm:
                return True
    
    return False


# ============================================================================
# STRUKTURY DANYCH
# ============================================================================

@dataclass
class PhraseHierarchy:
    """Wynik analizy hierarchii fraz."""
    
    # Wszystkie frazy z metadanymi
    all_phrases: List[Dict] = field(default_factory=list)
    
    # Rdzenie i ich rozszerzenia
    roots: Dict[str, Dict] = field(default_factory=dict)
    # Format: {root: {"type": str, "target_min": int, "target_max": int, 
    #                 "extensions": [...], "extensions_by_type": {...}}}
    
    # Mapowania
    phrase_to_root: Dict[str, str] = field(default_factory=dict)
    phrase_to_entities: Dict[str, List[Dict]] = field(default_factory=dict)
    phrase_to_triplets: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # Efektywne targety
    effective_targets: Dict[str, Dict] = field(default_factory=dict)
    
    # Encje bez pasujących fraz (trzeba wpleść osobno)
    unmatched_entities: List[Dict] = field(default_factory=list)
    
    # Statystyki
    stats: Dict = field(default_factory=dict)


@dataclass
class EffectiveTarget:
    """Efektywny target dla rdzenia."""
    root: str
    original_min: int
    original_max: int
    extensions_count: int
    effective_min: int
    effective_max: int
    strategy: str  # "extensions_sufficient", "mixed", "need_standalone"
    
    def to_dict(self) -> Dict:
        return {
            "root": self.root,
            "original_min": self.original_min,
            "original_max": self.original_max,
            "extensions_count": self.extensions_count,
            "effective_min": self.effective_min,
            "effective_max": self.effective_max,
            "strategy": self.strategy
        }


# ============================================================================
# GŁÓWNA FUNKCJA ANALIZY
# ============================================================================

def analyze_phrase_hierarchy(
    keywords: List[Dict],
    entities: List[Dict] = None,
    triplets: List[Dict] = None,
    h2_terms: List[str] = None,
    config: HierarchyConfig = None
) -> PhraseHierarchy:
    """
    GŁÓWNA FUNKCJA - analizuje hierarchię fraz po S1/create project.
    
    Args:
        keywords: Lista fraz z targetami
            [{"keyword": str, "type": str, "target_min": int, "target_max": int}]
        entities: Encje z S1
            [{"name": str, "priority": str, "importance": float}]
        triplets: Triplety semantyczne
            [{"subject": str, "verb": str, "object": str}]
        h2_terms: Terminy z nagłówków H2
        config: Konfiguracja (opcjonalna)
        
    Returns:
        PhraseHierarchy z pełną mapą relacji
    """
    if config is None:
        config = CONFIG
    
    result = PhraseHierarchy()
    entities = entities or []
    triplets = triplets or []
    h2_terms = h2_terms or []
    
    # 1. Zbierz wszystkie frazy
    all_phrases = []
    phrase_meta = {}
    
    for kw in keywords:
        phrase = kw.get("keyword", kw.get("phrase", "")).strip()
        if not phrase:
            continue
        
        ptype = kw.get("type", "BASIC").upper()
        tmin = kw.get("target_min", 1)
        tmax = kw.get("target_max", 10)
        
        # Parsuj "1-3x" format jeśli potrzeba
        if isinstance(tmin, str):
            if "-" in tmin:
                parts = tmin.replace("x", "").split("-")
                tmin = int(parts[0])
                tmax = int(parts[1]) if len(parts) > 1 else tmin
            else:
                tmin = int(tmin.replace("x", ""))
                tmax = tmin
        
        all_phrases.append(phrase)
        phrase_meta[phrase] = {
            "type": ptype,
            "target_min": tmin,
            "target_max": tmax
        }
    
    # Dodaj H2 terms jeśli nie ma ich w keywords
    for term in h2_terms:
        if term not in phrase_meta:
            all_phrases.append(term)
            phrase_meta[term] = {
                "type": "H2_TERM",
                "target_min": 1,
                "target_max": 3
            }
    
    result.all_phrases = [
        {"phrase": p, **phrase_meta[p]} 
        for p in all_phrases
    ]
    
    # 2. Identyfikuj rdzenie (frazy zawarte w wielu innych)
    potential_roots = []
    
    for phrase in all_phrases:
        phrase_norm = normalize_phrase(phrase)
        word_count = len(phrase_norm.split())
        
        # Znajdź frazy które zawierają tę frazę
        contained_in = []
        for other in all_phrases:
            if other != phrase:
                if phrase_contains_root(other, phrase, config.USE_INFLECTION):
                    contained_in.append(other)
        
        # Jest rdzeniem jeśli krótka i ma rozszerzenia
        if (len(contained_in) >= config.MIN_EXTENSIONS_FOR_ROOT and 
            word_count <= config.MAX_ROOT_WORDS):
            potential_roots.append({
                "phrase": phrase,
                "extensions": contained_in,
                "count": len(contained_in),
                "word_count": word_count
            })
    
    # Sortuj: najpierw najczęstsze, potem najkrótsze
    potential_roots.sort(key=lambda x: (-x["count"], x["word_count"]))
    
    # 3. Zbuduj hierarchię rdzeni (unikaj duplikatów rozszerzeń)
    used_as_extension = set()
    
    for root_info in potential_roots:
        root = root_info["phrase"]
        all_extensions = root_info["extensions"]
        
        # Filtruj rozszerzenia już przypisane do innego rdzenia
        available_extensions = [
            e for e in all_extensions 
            if e not in used_as_extension
        ]
        
        if not available_extensions:
            continue
        
        meta = phrase_meta.get(root, {})
        
        # Grupuj rozszerzenia według typu
        by_type = defaultdict(list)
        for ext in available_extensions:
            ext_meta = phrase_meta.get(ext, {})
            ext_type = ext_meta.get("type", "UNKNOWN")
            by_type[ext_type].append(ext)
        
        result.roots[root] = {
            "type": meta.get("type", "BASIC"),
            "target_min": meta.get("target_min", 1),
            "target_max": meta.get("target_max", 10),
            "extensions": available_extensions,
            "extensions_count": len(available_extensions),
            "extensions_by_type": dict(by_type)
        }
        
        # Oznacz rozszerzenia i mapuj do rdzenia
        for ext in available_extensions:
            result.phrase_to_root[ext] = root
            used_as_extension.add(ext)
    
    # 4. Oblicz efektywne targety
    for root, info in result.roots.items():
        ext_count = info["extensions_count"]
        original_min = info["target_min"]
        original_max = info["target_max"]
        
        # Zakładamy że każde rozszerzenie będzie użyte min 1x
        effective_min = max(0, original_min - ext_count)
        effective_max = max(0, original_max - ext_count)
        
        # Określ strategię
        if ext_count >= original_max * config.EXTENSIONS_SUFFICIENT_THRESHOLD:
            strategy = "extensions_sufficient"
        elif ext_count >= original_min:
            strategy = "mixed"
        else:
            strategy = "need_standalone"
        
        result.effective_targets[root] = EffectiveTarget(
            root=root,
            original_min=original_min,
            original_max=original_max,
            extensions_count=ext_count,
            effective_min=effective_min,
            effective_max=effective_max,
            strategy=strategy
        ).to_dict()
    
    # 5. Mapuj encje do fraz
    for entity in entities:
        entity_name = entity.get("name", entity.get("entity", ""))
        if not entity_name:
            continue
        
        entity_norm = normalize_phrase(entity_name)
        matched = False
        
        for phrase in all_phrases:
            phrase_norm = normalize_phrase(phrase)
            
            # Dopasowanie: dokładne lub częściowe
            if (entity_norm == phrase_norm or 
                entity_norm in phrase_norm or 
                phrase_norm in entity_norm or
                phrase_contains_root(phrase, entity_name)):
                
                if phrase not in result.phrase_to_entities:
                    result.phrase_to_entities[phrase] = []
                
                result.phrase_to_entities[phrase].append({
                    "entity": entity_name,
                    "priority": entity.get("priority", "SHOULD"),
                    "importance": entity.get("importance", 0.5)
                })
                matched = True
        
        if not matched:
            result.unmatched_entities.append({
                "entity": entity_name,
                "priority": entity.get("priority", "SHOULD")
            })
    
    # 6. Mapuj triplety do fraz
    for i, triplet in enumerate(triplets):
        subject = triplet.get("subject", "")
        obj = triplet.get("object", "")
        verb = triplet.get("verb", "")
        
        for phrase in all_phrases:
            phrase_norm = normalize_phrase(phrase)
            
            # Sprawdź czy fraza zawiera subject i/lub object
            has_subject = phrase_contains_root(phrase, subject) if subject else False
            has_object = phrase_contains_root(phrase, obj) if obj else False
            
            if has_subject or has_object:
                if phrase not in result.phrase_to_triplets:
                    result.phrase_to_triplets[phrase] = []
                
                result.phrase_to_triplets[phrase].append({
                    "index": i,
                    "triplet": triplet,
                    "role": "FULL" if (has_subject and has_object) else 
                           ("SUBJECT" if has_subject else "OBJECT")
                })
    
    # 7. Statystyki
    result.stats = {
        "total_phrases": len(all_phrases),
        "roots_count": len(result.roots),
        "main_count": sum(1 for p in result.all_phrases if p["type"] == "MAIN"),
        "basic_count": sum(1 for p in result.all_phrases if p["type"] == "BASIC"),
        "extended_count": sum(1 for p in result.all_phrases if p["type"] == "EXTENDED"),
        "h2_terms_count": sum(1 for p in result.all_phrases if p["type"] == "H2_TERM"),
        "entity_phrases": len(result.phrase_to_entities),
        "triplet_phrases": len(result.phrase_to_triplets),
        "unmatched_entities": len(result.unmatched_entities),
        "extensions_sufficient_roots": sum(
            1 for t in result.effective_targets.values() 
            if t["strategy"] == "extensions_sufficient"
        )
    }
    
    return result


# ============================================================================
# KONTEKST DLA PRE_BATCH_INFO
# ============================================================================

def get_batch_hierarchy_context(
    hierarchy: PhraseHierarchy,
    batch_phrases: List[str],
    current_counts: Dict[str, int] = None
) -> Dict[str, Any]:
    """
    Generuje kontekst hierarchii dla konkretnego batcha.
    Do użycia w enhanced_pre_batch.py
    
    Args:
        hierarchy: Wynik analyze_phrase_hierarchy()
        batch_phrases: Frazy przypisane do tego batcha
        current_counts: Aktualne zliczenia (opcjonalne)
        
    Returns:
        Kontekst hierarchii dla batcha
    """
    current_counts = current_counts or {}
    
    context = {
        "roots_covered_by_batch": [],
        "extensions_to_use": [],
        "entity_phrases": [],
        "triplet_phrases": [],
        "effective_targets_for_batch": {},
        "writing_tips": []
    }
    
    # 1. Jakie rdzenie będą pokryte przez frazy w batchu?
    roots_covered = set()
    
    for phrase in batch_phrases:
        # Czy ta fraza jest rozszerzeniem jakiegoś rdzenia?
        root = hierarchy.phrase_to_root.get(phrase)
        if root:
            roots_covered.add(root)
            context["extensions_to_use"].append({
                "phrase": phrase,
                "root": root
            })
        
        # Czy ta fraza jest powiązana z encjami?
        if phrase in hierarchy.phrase_to_entities:
            context["entity_phrases"].append({
                "phrase": phrase,
                "entities": hierarchy.phrase_to_entities[phrase]
            })
        
        # Czy ta fraza jest powiązana z tripletami?
        if phrase in hierarchy.phrase_to_triplets:
            context["triplet_phrases"].append({
                "phrase": phrase,
                "triplets": hierarchy.phrase_to_triplets[phrase]
            })
    
    context["roots_covered_by_batch"] = list(roots_covered)
    
    # 2. Efektywne targety dla rdzeni w batchu
    for root in roots_covered:
        if root in hierarchy.effective_targets:
            context["effective_targets_for_batch"][root] = hierarchy.effective_targets[root]
    
    # 3. Wskazówki dla agenta
    if roots_covered:
        context["writing_tips"].append(
            f"✅ Użycie fraz EXTENDED automatycznie zaspokaja rdzenie: {', '.join(list(roots_covered)[:3])}"
        )
    
    if context["entity_phrases"]:
        entity_names = [e["entities"][0]["entity"] for e in context["entity_phrases"][:3]]
        context["writing_tips"].append(
            f"🔵 Priorytetowe encje do wplecenia: {', '.join(entity_names)}"
        )
    
    if context["triplet_phrases"]:
        context["writing_tips"].append(
            f"🔺 {len(context['triplet_phrases'])} fraz realizuje triplety semantyczne"
        )
    
    # Sprawdź czy są rdzenie z strategią "extensions_sufficient"
    sufficient_roots = [
        root for root in roots_covered
        if hierarchy.effective_targets.get(root, {}).get("strategy") == "extensions_sufficient"
    ]
    if sufficient_roots:
        context["writing_tips"].append(
            f"💡 Dla rdzeni [{', '.join(sufficient_roots[:2])}] rozszerzenia WYSTARCZĄ - nie musisz ich powtarzać osobno!"
        )
    
    return context


# ============================================================================
# FORMATOWANIE DLA AGENTA
# ============================================================================

def format_hierarchy_for_agent(
    hierarchy: PhraseHierarchy,
    include_full_list: bool = False
) -> str:
    """
    Formatuje hierarchię jako tekst dla agenta/promptu.
    
    Args:
        hierarchy: Wynik analizy
        include_full_list: Czy pokazać pełną listę rozszerzeń
        
    Returns:
        Sformatowany tekst
    """
    lines = []
    lines.append("=" * 60)
    lines.append("📊 HIERARCHIA FRAZ - GLOBALNA MAPA")
    lines.append("=" * 60)
    
    # Statystyki
    s = hierarchy.stats
    lines.append(f"\n📈 PODSUMOWANIE:")
    lines.append(f"   Wszystkich fraz: {s['total_phrases']}")
    lines.append(f"   Rdzeni (frazy zawarte w innych): {s['roots_count']}")
    lines.append(f"   Fraz = encjom z S1: {s['entity_phrases']}")
    lines.append(f"   Fraz realizujących triplety: {s['triplet_phrases']}")
    
    if s['extensions_sufficient_roots'] > 0:
        lines.append(f"   ⚡ Rdzenie gdzie rozszerzenia wystarczą: {s['extensions_sufficient_roots']}")
    
    # Top rdzenie
    lines.append(f"\n🌳 RDZENIE I ICH ROZSZERZENIA:")
    lines.append("-" * 50)
    
    sorted_roots = sorted(
        hierarchy.roots.items(),
        key=lambda x: -x[1]["extensions_count"]
    )[:7]  # Top 7 rdzeni
    
    for root, info in sorted_roots:
        ext_count = info["extensions_count"]
        eff_target = hierarchy.effective_targets.get(root, {})
        strategy = eff_target.get("strategy", "unknown")
        
        strategy_icon = {
            "extensions_sufficient": "✅",
            "mixed": "⚖️",
            "need_standalone": "⚠️"
        }.get(strategy, "❓")
        
        lines.append(f"\n🔷 \"{root}\" ({info['type']})")
        lines.append(f"   Target: {info['target_min']}-{info['target_max']}x")
        lines.append(f"   Rozszerzeń: {ext_count}")
        lines.append(f"   Efektywny samodzielny: {eff_target.get('effective_min', '?')}-{eff_target.get('effective_max', '?')}x")
        lines.append(f"   Strategia: {strategy_icon} {strategy}")
        
        if include_full_list:
            by_type = info.get("extensions_by_type", {})
            for typ, phrases in by_type.items():
                lines.append(f"      {typ}: {', '.join(phrases[:3])}{'...' if len(phrases) > 3 else ''}")
    
    # Encje bez dopasowania
    if hierarchy.unmatched_entities:
        lines.append(f"\n⚠️ ENCJE BEZ PASUJĄCYCH FRAZ (wpleć osobno!):")
        for ent in hierarchy.unmatched_entities[:5]:
            lines.append(f"   • {ent['entity']} ({ent['priority']})")
    
    # Wskazówki
    lines.append(f"\n💡 KLUCZOWE ZASADY:")
    lines.append("   1. Użycie ROZSZERZENIA automatycznie zaspokaja RDZEŃ")
    lines.append("   2. Frazy = encjom mają PRIORYTET (budują topical authority)")
    lines.append("   3. NIE powtarzaj rdzenia 40x jeśli masz 30 rozszerzeń!")
    lines.append("   4. Używaj RÓŻNYCH FORM FLEKSYJNYCH (zespołu, zespołem, zespole)")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def format_hierarchy_summary_short(hierarchy: PhraseHierarchy) -> str:
    """
    Krótkie podsumowanie hierarchii (do pre_batch_info).
    """
    s = hierarchy.stats
    
    # Top 3 rdzenie z najliczniejszymi rozszerzeniami
    top_roots = sorted(
        hierarchy.roots.items(),
        key=lambda x: -x[1]["extensions_count"]
    )[:3]
    
    root_info = []
    for root, info in top_roots:
        eff = hierarchy.effective_targets.get(root, {})
        strategy = eff.get("strategy", "?")
        root_info.append(f'"{root}" ({info["extensions_count"]} ext, {strategy})')
    
    return f"""📊 HIERARCHIA: {s['roots_count']} rdzeni, {s['entity_phrases']} fraz=encji
🌳 Top rdzenie: {' | '.join(root_info)}
💡 Użycie EXTENDED automatycznie zaspokaja rdzenie!"""


# ============================================================================
# SERIALIZACJA (do zapisu w Firestore)
# ============================================================================

def hierarchy_to_dict(hierarchy: PhraseHierarchy) -> Dict:
    """Konwertuje PhraseHierarchy do słownika (do zapisu w Firestore)."""
    return {
        "roots": hierarchy.roots,
        "phrase_to_root": hierarchy.phrase_to_root,
        "phrase_to_entities": hierarchy.phrase_to_entities,
        "phrase_to_triplets": hierarchy.phrase_to_triplets,
        "effective_targets": hierarchy.effective_targets,
        "unmatched_entities": hierarchy.unmatched_entities,
        "stats": hierarchy.stats
    }


def dict_to_hierarchy(data: Dict) -> PhraseHierarchy:
    """Odtwarza PhraseHierarchy ze słownika (z Firestore)."""
    h = PhraseHierarchy()
    h.roots = data.get("roots", {})
    h.phrase_to_root = data.get("phrase_to_root", {})
    h.phrase_to_entities = data.get("phrase_to_entities", {})
    h.phrase_to_triplets = data.get("phrase_to_triplets", {})
    h.effective_targets = data.get("effective_targets", {})
    h.unmatched_entities = data.get("unmatched_entities", [])
    h.stats = data.get("stats", {})
    return h


# ============================================================================
# EKSPORTY
# ============================================================================

__version__ = "1.0"
__all__ = [
    # Główne funkcje
    "analyze_phrase_hierarchy",
    "get_batch_hierarchy_context",
    "format_hierarchy_for_agent",
    "format_hierarchy_summary_short",
    
    # Serializacja
    "hierarchy_to_dict",
    "dict_to_hierarchy",
    
    # Typy
    "PhraseHierarchy",
    "EffectiveTarget",
    "HierarchyConfig",
    "CONFIG"
]
