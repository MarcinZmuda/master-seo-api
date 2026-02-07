"""
===============================================================================
SMART BATCH INSTRUCTIONS v41.1
===============================================================================
Generuje KONKRETNE instrukcje dla każdego batcha:

PROBLEMY KTÓRE ROZWIĄZUJE:
1. Agent dostaje 40 fraz i ignoruje większość
2. Triplety są "opisowe" nie "relacyjne"
3. Brak przykładowych zdań

ROZWIĄZANIE:
1. Max 5 fraz MUST_USE per batch z przykładami
2. Max 3 triplety z DOSŁOWNYMI zdaniami do wstawienia
3. Kontekst: jak powiązać z aktualnym H2

v41.1: Nowy moduł
===============================================================================
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PhraseInstruction:
    """Instrukcja użycia konkretnej frazy."""
    phrase: str
    type: str  # BASIC, EXTENDED, MAIN
    priority: str  # MUST, SHOULD, NICE
    current_uses: int
    target_min: int
    target_max: int
    suggested_this_batch: int
    example_sentence: str
    context_hint: str
    integration_tip: str


@dataclass
class TripletInstruction:
    """Instrukcja użycia tripletu S-V-O."""
    subject: str
    verb: str
    object: str
    priority: str  # MUST, SHOULD, NICE
    literal_sentence: str  # Dokładne zdanie do wstawienia
    alternative_sentences: List[str]  # Alternatywy
    where_to_use: str  # "intro", "first_paragraph", "any"


@dataclass
class BatchInstructions:
    """Kompletne instrukcje dla batcha."""
    batch_number: int
    batch_type: str
    current_h2: List[str]
    
    # Frazy
    must_use_phrases: List[PhraseInstruction]
    should_use_phrases: List[PhraseInstruction]
    avoid_phrases: List[str]
    
    # Triplety
    must_write_triplets: List[TripletInstruction]
    should_write_triplets: List[TripletInstruction]
    
    # Humanizacja
    sentence_variety_tips: List[str]
    short_sentences_to_include: List[str]
    
    # Summary for agent
    executive_summary: str
    checklist: List[str]


# ============================================================================
# PHRASE EXAMPLE GENERATOR
# ============================================================================

def generate_phrase_example(
    phrase: str,
    phrase_type: str,
    h2_context: str,
    domain: str = "prawo"
) -> Tuple[str, str, str]:
    """
    Generuje przykładowe zdanie, kontekst i tip dla frazy.
    
    Returns:
        (example_sentence, context_hint, integration_tip)
    """
    phrase_lower = phrase.lower()
    
    # === PRAWO ===
    if domain == "prawo":
        # Sądy i procedury
        if "sąd" in phrase_lower:
            if "okręgowy" in phrase_lower:
                return (
                    f"Wniosek należy złożyć do sądu okręgowego właściwego dla miejsca zamieszkania.",
                    "Użyj w kontekście właściwości miejscowej",
                    "Wpleć w zdanie o procedurze lub właściwości sądu"
                )
            elif "rodzinny" in phrase_lower:
                return (
                    f"Sąd rodzinny rozstrzyga spory dotyczące władzy rodzicielskiej i kontaktów z dzieckiem.",
                    "Użyj w kontekście kompetencji sądu",
                    "Połącz z informacją o zakresie spraw"
                )
            else:
                return (
                    f"Sąd wydaje orzeczenie po rozpatrzeniu wszystkich okoliczności sprawy.",
                    "Użyj w kontekście procesu sądowego",
                    "Naturalnie wpleć w opis procedury"
                )
        
        # Władza rodzicielska
        if "władz" in phrase_lower and "rodzic" in phrase_lower:
            return (
                f"Ograniczenie władzy rodzicielskiej następuje w drodze postanowienia sądu.",
                "Użyj w kontekście ograniczenia/pozbawienia",
                "Połącz z konsekwencjami dla rodzica"
            )
        
        # Miejsce pobytu
        if "miejsc" in phrase_lower and "pobyt" in phrase_lower:
            if "ustal" in phrase_lower:
                return (
                    f"Ustalenie miejsca pobytu dziecka wymaga złożenia wniosku do sądu rodzinnego.",
                    "Użyj w kontekście procedury sądowej",
                    "Zawsze połącz z dobrem dziecka"
                )
            return (
                f"Sąd określa miejsce pobytu dziecka kierując się jego dobrem.",
                "Użyj w kontekście decyzji sądowej",
                "Zawsze połącz z dobrem dziecka"
            )
        
        # Uprowadzenie/porwanie
        if "uprowadz" in phrase_lower:
            return (
                f"Uprowadzenie dziecka jest przestępstwem ściganym na podstawie art. 211 k.k.",
                "Użyj w kontekście odpowiedzialności karnej",
                "Wyjaśnij różnicę z porwaniem rodzicielskim"
            )
        if "porwan" in phrase_lower:
            return (
                f"Porwanie rodzicielskie polega na samowolnym zabraniu dziecka przez jednego z rodziców.",
                "Użyj w kontekście definicji",
                "Odróżnij od uprowadzenia w sensie karnym"
            )
        
        # Artykuły kodeksu
        if "art" in phrase_lower or "kodeks" in phrase_lower:
            return (
                f"Zgodnie z {phrase}, odpowiedzialność karna powstaje gdy...",
                "Użyj jako podstawę prawną",
                "Podaj konkretne przesłanki"
            )
        
        # Kontakty z dzieckiem
        if "kontakt" in phrase_lower:
            return (
                f"Rodzic ma prawo do {phrase} niezależnie od miejsca zamieszkania.",
                "Użyj w kontekście praw rodzica",
                "Oddziel od miejsca zamieszkania"
            )
        
        # Wniosek
        if "wnios" in phrase_lower:
            return (
                f"{phrase.capitalize()} składa się do sądu opiekuńczego.",
                "Użyj w kontekście wszczęcia procedury",
                "Wskaż kto może złożyć i gdzie"
            )
    
    # === DOMYŚLNY ===
    return (
        f"W kontekście {h2_context}, {phrase} ma istotne znaczenie.",
        f"Użyj naturalnie w sekcji '{h2_context}'",
        "Wpleć w główny tok narracji, nie na siłę"
    )


# ============================================================================
# TRIPLET SENTENCE GENERATOR
# ============================================================================

def generate_triplet_sentences(
    subject: str,
    verb: str,
    obj: str,
    domain: str = "prawo"
) -> Tuple[str, List[str]]:
    """
    Generuje dosłowne zdanie i alternatywy dla tripletu S-V-O.
    
    Returns:
        (literal_sentence, alternative_sentences)
    """
    # Normalizacja
    s = subject.strip()
    v = verb.strip()
    o = obj.strip()
    
    # Główne zdanie (proste, relacyjne)
    literal = f"{s.capitalize()} {v} {o}."
    
    # Alternatywy (różne konstrukcje)
    alternatives = []
    
    # Konstrukcja bierna
    if v in ["ustala", "wydaje", "orzeka", "rozstrzyga"]:
        alternatives.append(f"{o.capitalize()} jest {v.replace('a', 'any').replace('e', 'any')} przez {s}.")
    
    # Konstrukcja z "to"
    alternatives.append(f"To {s} {v} {o}.")
    
    # Konstrukcja pytająca (retoryczne)
    alternatives.append(f"Kto {v} {o}? {s.capitalize()}.")
    
    # Konstrukcja z kontekstem
    if domain == "prawo":
        alternatives.append(f"W polskim systemie prawnym {s} {v} {o}.")
        alternatives.append(f"Zgodnie z przepisami, {s} {v} {o}.")
    
    return literal, alternatives[:3]


# ============================================================================
# SMART KEYWORD SELECTOR
# ============================================================================

def select_must_use_phrases(
    keywords_state: Dict,
    current_batch_num: int,
    total_batches: int,
    current_h2: List[str],
    already_well_covered: List[str] = None
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Wybiera MAX 5 fraz MUST_USE i MAX 5 SHOULD_USE dla tego batcha.
    
    Strategia:
    1. Priorytety: BASIC nieużyte > EXTENDED nieużyte > BASIC below target
    2. Powiązanie z H2 (jeśli możliwe)
    3. Rozłożenie równomierne przez batche
    
    Returns:
        (must_use, should_use, avoid)
    """
    if already_well_covered is None:
        already_well_covered = []
    
    remaining_batches = max(1, total_batches - current_batch_num + 1)
    h2_text = " ".join(current_h2).lower()
    
    # Kategoryzuj frazy
    unused_basic = []
    unused_extended = []
    below_target = []
    near_limit = []
    ok_phrases = []
    
    for rid, meta in keywords_state.items():
        keyword = meta.get("keyword", "")
        if not keyword:
            continue
        
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        target_min = meta.get("target_min", 1)
        target_max = meta.get("target_max", 999)
        is_main = meta.get("is_main_keyword", False)
        
        # Skip main keyword (handled separately)
        if is_main:
            continue
        
        # Skip already well covered
        if keyword.lower() in [k.lower() for k in already_well_covered]:
            continue
        
        # Calculate relevance to H2
        h2_relevance = 0.5
        for word in keyword.lower().split():
            if len(word) > 3 and word in h2_text:
                h2_relevance = 1.0
                break
        
        info = {
            "keyword": keyword,
            "type": kw_type,
            "actual": actual,
            "target_min": target_min,
            "target_max": target_max,
            "remaining_needed": max(0, target_min - actual),
            "remaining_allowed": max(0, target_max - actual),
            "h2_relevance": h2_relevance
        }
        
        # Categorize
        if actual == 0:
            if kw_type == "BASIC":
                unused_basic.append(info)
            else:
                unused_extended.append(info)
        elif actual < target_min:
            below_target.append(info)
        elif target_max - actual <= 2:
            near_limit.append(info)
        else:
            ok_phrases.append(info)
    
    # Sort by H2 relevance
    unused_basic.sort(key=lambda x: -x["h2_relevance"])
    unused_extended.sort(key=lambda x: -x["h2_relevance"])
    below_target.sort(key=lambda x: (-x["h2_relevance"], -x["remaining_needed"]))
    
    # === SELECT MUST_USE (max 5) ===
    must_use = []
    
    # 1. Nieużyte BASIC (priorytet!)
    for info in unused_basic[:3]:
        info["priority"] = "MUST"
        info["reason"] = "BASIC nieużyta - MUSI być w artykule"
        must_use.append(info)
    
    # 2. Nieużyte EXTENDED (jeśli zostało miejsce)
    if len(must_use) < 5:
        for info in unused_extended[:2]:
            info["priority"] = "MUST"
            info["reason"] = "EXTENDED nieużyta"
            must_use.append(info)
            if len(must_use) >= 5:
                break
    
    # 3. Below target
    if len(must_use) < 5:
        for info in below_target[:2]:
            info["priority"] = "MUST"
            info["reason"] = f"Poniżej minimum ({info['actual']}/{info['target_min']})"
            must_use.append(info)
            if len(must_use) >= 5:
                break
    
    # === SELECT SHOULD_USE (max 5) ===
    should_use = []
    
    # Pozostałe nieużyte
    remaining_unused = [x for x in unused_basic + unused_extended if x not in must_use]
    for info in remaining_unused[:3]:
        info["priority"] = "SHOULD"
        info["reason"] = "Warto użyć - nieużyta jeszcze"
        should_use.append(info)
    
    # OK phrases relevant to H2
    h2_relevant = [x for x in ok_phrases if x["h2_relevance"] == 1.0]
    for info in h2_relevant[:2]:
        info["priority"] = "SHOULD"
        info["reason"] = "Pasuje do aktualnego H2"
        should_use.append(info)
        if len(should_use) >= 5:
            break
    
    # === AVOID ===
    avoid = [x["keyword"] for x in near_limit]
    
    return must_use, should_use, avoid


# ============================================================================
# SMART TRIPLET SELECTOR
# ============================================================================

def select_triplets_for_batch(
    s1_data: Dict,
    current_batch_num: int,
    total_batches: int,
    current_h2: List[str],
    already_used_triplets: List[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Wybiera MAX 3 triplety MUST i MAX 2 SHOULD dla tego batcha.
    
    Returns:
        (must_triplets, should_triplets)
    """
    if already_used_triplets is None:
        already_used_triplets = []
    
    entity_seo = s1_data.get("entity_seo", {})
    relationships = entity_seo.get("entity_relationships", [])
    entities = entity_seo.get("entities", [])
    
    # Build entity importance map
    entity_importance = {}
    for e in entities:
        name = e.get("name", "").lower()
        entity_importance[name] = {
            "importance": e.get("importance", 0.5),
            "sources": e.get("sources_count", 1)
        }
    
    h2_text = " ".join(current_h2).lower()
    
    # Score and categorize triplets
    scored_triplets = []
    for rel in relationships:
        subject = rel.get("subject", "")
        verb = rel.get("verb", "")
        obj = rel.get("object", "")
        
        if not subject or not verb or not obj:
            continue
        
        triplet_key = f"{subject}-{verb}-{obj}".lower()
        if triplet_key in [t.lower() for t in already_used_triplets]:
            continue
        
        # Calculate score
        score = 0
        
        # Subject importance
        subj_info = entity_importance.get(subject.lower(), {"importance": 0.5, "sources": 1})
        score += subj_info["importance"] * 30
        score += min(subj_info["sources"], 5) * 5
        
        # Object importance
        obj_info = entity_importance.get(obj.lower(), {"importance": 0.5, "sources": 1})
        score += obj_info["importance"] * 20
        
        # H2 relevance
        for word in subject.lower().split() + obj.lower().split():
            if len(word) > 3 and word in h2_text:
                score += 15
        
        # Generate sentences
        literal, alternatives = generate_triplet_sentences(subject, verb, obj, "prawo")
        
        scored_triplets.append({
            "subject": subject,
            "verb": verb,
            "object": obj,
            "score": score,
            "literal_sentence": literal,
            "alternative_sentences": alternatives,
            "importance": subj_info["importance"],
            "sources": subj_info["sources"]
        })
    
    # Sort by score
    scored_triplets.sort(key=lambda x: -x["score"])
    
    # === SELECT MUST (max 3) ===
    must_triplets = []
    for t in scored_triplets:
        if t["importance"] >= 0.7 or t["sources"] >= 4:
            t["priority"] = "MUST"
            must_triplets.append(t)
            if len(must_triplets) >= 3:
                break
    
    # === SELECT SHOULD (max 2) ===
    should_triplets = []
    for t in scored_triplets:
        if t not in must_triplets and (t["importance"] >= 0.5 or t["sources"] >= 3):
            t["priority"] = "SHOULD"
            should_triplets.append(t)
            if len(should_triplets) >= 2:
                break
    
    return must_triplets, should_triplets


# ============================================================================
# HUMANIZATION TIPS
# ============================================================================

def generate_humanization_tips(
    batch_type: str,
    current_h2: List[str],
    domain: str = "prawo"
) -> Tuple[List[str], List[str]]:
    """
    Generuje konkretne tipy humanizacyjne.
    
    ⚠️ v45.0: Usunięto statyczną bibliotekę krótkich zdań.
    GPT dostawał gotowe "Sąd orzeka." | "Ale uwaga." i kopiował je
    verbatim w setkach artykułów — tworząc nowy marker AI.
    
    Krótkie zdania teraz generowane wyłącznie z kontekstu akapitu
    (instrukcja z dynamic_humanization.py).
    
    Returns:
        (variety_tips, short_sentence_rules)
    """
    # Tips
    tips = [
        "Przeplataj długości: 5 słów → 22 słowa → 8 słów → 28 słów → 6 słów",
        "Krótkie zdania W ŚRODKU akapitu, nie tylko na końcu",
        "Nie twórz krótkich zdań oderwanych od treści akapitu"
    ]
    
    # ⚠️ v45.0: Zamiast gotowych zdań — REGUŁY tworzenia
    short_sentence_rules = [
        "Weź kluczowy fakt z poprzedniego zdania i skondensuj do 3-8 słów",
        "Krótkie zdanie MUSI zawierać termin/nazwę/liczbę z tego akapitu",
        "TEST: czy to zdanie pasowałoby do INNEGO artykułu? Jeśli tak → przepisz"
    ]
    
    # Add batch-specific tips
    if batch_type == "INTRO":
        tips.append("INTRO: Zacznij od krótkiego zdania (5-8 słów) jako hook")
        tips.append("Direct answer w pierwszym akapicie")
    elif batch_type == "FINAL":
        tips.append("FINAL: Podsumuj kluczowe punkty, ale NIE powtarzaj definicji")
    
    return tips, short_sentence_rules


# ============================================================================
# MAIN: GENERATE SMART INSTRUCTIONS
# ============================================================================

def generate_smart_batch_instructions(
    keywords_state: Dict,
    s1_data: Dict,
    current_batch_num: int,
    total_batches: int,
    current_h2: List[str],
    batch_type: str,
    already_well_covered: List[str] = None,
    already_used_triplets: List[str] = None,
    domain: str = "prawo"
) -> Dict[str, Any]:
    """
    Generuje KONKRETNE instrukcje dla batcha.
    
    Returns:
        Dict with smart instructions ready for agent.
    """
    # Select phrases
    must_phrases, should_phrases, avoid_phrases = select_must_use_phrases(
        keywords_state=keywords_state,
        current_batch_num=current_batch_num,
        total_batches=total_batches,
        current_h2=current_h2,
        already_well_covered=already_well_covered
    )
    
    # Select triplets
    must_triplets, should_triplets = select_triplets_for_batch(
        s1_data=s1_data,
        current_batch_num=current_batch_num,
        total_batches=total_batches,
        current_h2=current_h2,
        already_used_triplets=already_used_triplets
    )
    
    # Humanization tips
    variety_tips, short_sentences = generate_humanization_tips(batch_type, current_h2, domain)
    
    # Generate phrase instructions with examples
    must_phrase_instructions = []
    for p in must_phrases:
        example, context, tip = generate_phrase_example(
            p["keyword"], p["type"], current_h2[0] if current_h2 else "", domain
        )
        must_phrase_instructions.append({
            "phrase": p["keyword"],
            "type": p["type"],
            "priority": "MUST",
            "current_uses": p["actual"],
            "target": f"{p['target_min']}-{p['target_max']}",
            "reason": p["reason"],
            "example_sentence": example,
            "context_hint": context,
            "integration_tip": tip
        })
    
    should_phrase_instructions = []
    for p in should_phrases:
        example, context, tip = generate_phrase_example(
            p["keyword"], p["type"], current_h2[0] if current_h2 else "", domain
        )
        should_phrase_instructions.append({
            "phrase": p["keyword"],
            "type": p["type"],
            "priority": "SHOULD",
            "current_uses": p["actual"],
            "target": f"{p['target_min']}-{p['target_max']}",
            "reason": p["reason"],
            "example_sentence": example,
            "context_hint": context,
            "integration_tip": tip
        })
    
    # Generate triplet instructions
    must_triplet_instructions = []
    for t in must_triplets:
        must_triplet_instructions.append({
            "subject": t["subject"],
            "verb": t["verb"],
            "object": t["object"],
            "priority": "MUST",
            "literal_sentence": t["literal_sentence"],
            "alternative_sentences": t["alternative_sentences"],
            "instruction": f"Napisz DOSŁOWNIE lub użyj alternatywy: '{t['literal_sentence']}'"
        })
    
    should_triplet_instructions = []
    for t in should_triplets:
        should_triplet_instructions.append({
            "subject": t["subject"],
            "verb": t["verb"],
            "object": t["object"],
            "priority": "SHOULD",
            "literal_sentence": t["literal_sentence"],
            "alternative_sentences": t["alternative_sentences"],
            "instruction": f"Jeśli pasuje do kontekstu: '{t['literal_sentence']}'"
        })
    
    # Build checklist
    checklist = []
    
    # Phrases checklist
    for p in must_phrase_instructions:
        checklist.append(f"☐ Użyj frazy: \"{p['phrase']}\" (MUST)")
    for p in should_phrase_instructions[:2]:
        checklist.append(f"☐ Rozważ frazę: \"{p['phrase']}\" (SHOULD)")
    
    # Triplets checklist
    for t in must_triplet_instructions:
        checklist.append(f"☐ Napisz: \"{t['literal_sentence']}\" (MUST)")
    
    # Structure checklist
    checklist.append("☐ Min 1 krótkie zdanie (3-8 słów) W ŚRODKU akapitu")
    checklist.append("☐ Zróżnicuj długości zdań (nie wszystkie 15-20 słów)")
    
    # Executive summary
    summary_parts = []
    if must_phrase_instructions:
        phrases_list = ", ".join([f"\"{p['phrase']}\"" for p in must_phrase_instructions[:3]])
        summary_parts.append(f"UŻYJ FRAZ: {phrases_list}")
    if must_triplet_instructions:
        triplet_list = "; ".join([f"\"{t['literal_sentence']}\"" for t in must_triplet_instructions[:2]])
        summary_parts.append(f"NAPISZ ZDANIA: {triplet_list}")
    if avoid_phrases:
        avoid_list = ", ".join(avoid_phrases[:3])
        summary_parts.append(f"UNIKAJ (blisko limitu): {avoid_list}")
    
    executive_summary = " | ".join(summary_parts) if summary_parts else "Napisz naturalnie, użyj fraz z listy SHOULD"
    
    return {
        "batch_number": current_batch_num,
        "batch_type": batch_type,
        "current_h2": current_h2,
        
        # Phrases
        "must_use_phrases": must_phrase_instructions,
        "should_use_phrases": should_phrase_instructions,
        "avoid_phrases": avoid_phrases,
        
        # Triplets
        "must_write_triplets": must_triplet_instructions,
        "should_write_triplets": should_triplet_instructions,
        
        # Humanization
        "sentence_variety_tips": variety_tips,
        "short_sentences_library": short_sentences,
        
        # Summary
        "executive_summary": executive_summary,
        "checklist": checklist,
        
        # Stats
        "stats": {
            "must_phrases_count": len(must_phrase_instructions),
            "should_phrases_count": len(should_phrase_instructions),
            "must_triplets_count": len(must_triplet_instructions),
            "avoid_count": len(avoid_phrases)
        }
    }


# ============================================================================
# FORMAT FOR GPT PROMPT
# ============================================================================

def format_instructions_for_gpt(instructions: Dict) -> str:
    """
    Formatuje instrukcje do czytelnego promptu dla GPT.
    """
    lines = []
    
    # Header
    lines.append(f"📋 INSTRUKCJE DLA BATCH {instructions['batch_number']} ({instructions['batch_type']})")
    lines.append("=" * 60)
    
    # Executive summary
    lines.append(f"\n🎯 PODSUMOWANIE: {instructions['executive_summary']}")
    
    # H2
    if instructions["current_h2"]:
        lines.append(f"\n📌 H2 w tym batchu: {', '.join(instructions['current_h2'])}")
    
    # Must use phrases
    if instructions["must_use_phrases"]:
        lines.append("\n" + "━" * 40)
        lines.append("✅ FRAZY MUST (MUSZĄ być użyte):")
        for p in instructions["must_use_phrases"]:
            lines.append(f"\n  📍 \"{p['phrase']}\" ({p['type']})")
            lines.append(f"     Stan: {p['current_uses']}/{p['target']}")
            lines.append(f"     Przykład: {p['example_sentence']}")
            lines.append(f"     Tip: {p['integration_tip']}")
    
    # Should use phrases
    if instructions["should_use_phrases"]:
        lines.append("\n" + "━" * 40)
        lines.append("🔶 FRAZY SHOULD (warto użyć):")
        for p in instructions["should_use_phrases"][:3]:
            lines.append(f"  • \"{p['phrase']}\" - {p['reason']}")
    
    # Avoid phrases
    if instructions["avoid_phrases"]:
        lines.append("\n" + "━" * 40)
        lines.append("⛔ UNIKAJ (blisko limitu):")
        lines.append(f"  {', '.join(instructions['avoid_phrases'][:5])}")
    
    # Must write triplets
    if instructions["must_write_triplets"]:
        lines.append("\n" + "━" * 40)
        lines.append("🔗 TRIPLETY MUST (napisz DOSŁOWNIE):")
        for t in instructions["must_write_triplets"]:
            lines.append(f"\n  📍 {t['subject']} → {t['verb']} → {t['object']}")
            lines.append(f"     ✏️ \"{t['literal_sentence']}\"")
            if t["alternative_sentences"]:
                lines.append(f"     Alternatywy: {'; '.join(t['alternative_sentences'][:2])}")
    
    # Should write triplets
    if instructions["should_write_triplets"]:
        lines.append("\n" + "━" * 40)
        lines.append("🔶 TRIPLETY SHOULD (jeśli pasuje):")
        for t in instructions["should_write_triplets"]:
            lines.append(f"  • \"{t['literal_sentence']}\"")
    
    # Humanization
    lines.append("\n" + "━" * 40)
    lines.append("✨ HUMANIZACJA:")
    for tip in instructions["sentence_variety_tips"][:3]:
        lines.append(f"  • {tip}")
    
    # v45.0: Reguły zamiast gotowych zdań
    lines.append("\n  ✂️ Krótkie zdania (3-8 słów) — REGUŁY:")
    for rule in instructions["short_sentences_library"][:3]:
        lines.append(f"  • {rule}")
    
    # Checklist
    lines.append("\n" + "━" * 40)
    lines.append("☑️ CHECKLIST przed wysłaniem:")
    for item in instructions["checklist"][:6]:
        lines.append(f"  {item}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    keywords_state = {
        "k1": {"keyword": "porwanie rodzicielskie", "type": "MAIN", "actual_uses": 5, "target_min": 10, "target_max": 25, "is_main_keyword": True},
        "k2": {"keyword": "sąd rodzinny", "type": "BASIC", "actual_uses": 0, "target_min": 3, "target_max": 10},
        "k3": {"keyword": "ustalenie miejsca pobytu dziecka", "type": "BASIC", "actual_uses": 0, "target_min": 2, "target_max": 8},
        "k4": {"keyword": "władza rodzicielska", "type": "BASIC", "actual_uses": 2, "target_min": 3, "target_max": 12},
        "k5": {"keyword": "uprowadzenie dziecka", "type": "EXTENDED", "actual_uses": 0, "target_min": 1, "target_max": 5},
        "k6": {"keyword": "art. 211 kodeksu karnego", "type": "EXTENDED", "actual_uses": 1, "target_min": 1, "target_max": 3},
    }
    
    s1_data = {
        "entity_seo": {
            "entities": [
                {"name": "sąd rodzinny", "importance": 0.85, "sources_count": 6},
                {"name": "władza rodzicielska", "importance": 0.80, "sources_count": 5},
                {"name": "miejsce pobytu dziecka", "importance": 0.75, "sources_count": 4},
            ],
            "entity_relationships": [
                {"subject": "sąd rodzinny", "verb": "ustala", "object": "miejsce pobytu dziecka"},
                {"subject": "rodzic", "verb": "narusza", "object": "prawa drugiego rodzica"},
                {"subject": "Konwencja haska", "verb": "reguluje", "object": "uprowadzenie za granicę"},
            ]
        }
    }
    
    result = generate_smart_batch_instructions(
        keywords_state=keywords_state,
        s1_data=s1_data,
        current_batch_num=2,
        total_batches=5,
        current_h2=["Procedura sądowa w sprawach o miejsce pobytu"],
        batch_type="CONTENT"
    )
    
    print(format_instructions_for_gpt(result))
