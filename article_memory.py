"""
🧠 ARTICLE MEMORY v1.0
Running Summary + Thesis Tracking - kontekst globalny dla AI

Rozwiązuje problem "Ślepego Pisarza":
- AI widzi nie tylko ostatnie zdania, ale całą strukturę argumentacji
- Może odwoływać się do wcześniejszych tez ("Jak wspomniano...")
- Zachowuje spójność narracyjną przez cały artykuł

Autor: SEO Master API v36.2
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class KeyClaim:
    """Pojedyncze twierdzenie/claim z artykułu"""
    batch_number: int
    section_h2: str
    claim_text: str
    entities_mentioned: List[str] = field(default_factory=list)
    can_reference: bool = True  # czy można się do tego odwoływać
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OpenThread:
    """Wątek otwarty (zapowiedziany, ale nie rozwinięty)"""
    introduced_in_batch: int
    thread_description: str
    expected_in_section: str = ""  # w której sekcji powinien być rozwinięty
    resolved: bool = False
    resolved_in_batch: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArticleMemory:
    """
    Pamięć artykułu - przechowuje kontekst globalny
    
    Aktualizowana po każdym batchu, używana w pre_batch_info
    """
    project_id: str
    
    # Główna teza artykułu (generowana z intro)
    thesis: str = ""
    thesis_keywords: List[str] = field(default_factory=list)
    
    # Kluczowe twierdzenia z każdego batcha
    key_claims: List[KeyClaim] = field(default_factory=list)
    
    # Wprowadzone encje (z batch number)
    introduced_entities: Dict[str, int] = field(default_factory=dict)  # entity -> batch_num
    
    # Zdefiniowane pojęcia (można używać bez wyjaśniania)
    defined_terms: Dict[str, str] = field(default_factory=dict)  # term -> krótka definicja
    
    # Otwarte wątki (zapowiedziane tematy)
    open_threads: List[OpenThread] = field(default_factory=list)
    
    # Dynamiczne streszczenie (max 200 słów)
    running_summary: str = ""
    
    # Ton i styl (aktualizowane przez style_analyzer)
    tone_summary: str = ""
    
    # Meta
    last_updated_batch: int = 0
    total_words_written: int = 0
    
    def add_claim(self, batch_num: int, h2: str, claim: str, entities: List[str] = None):
        """Dodaj kluczowe twierdzenie z batcha"""
        self.key_claims.append(KeyClaim(
            batch_number=batch_num,
            section_h2=h2,
            claim_text=claim,
            entities_mentioned=entities or []
        ))
    
    def introduce_entity(self, entity: str, batch_num: int):
        """Zaznacz że encja została wprowadzona"""
        if entity not in self.introduced_entities:
            self.introduced_entities[entity] = batch_num
    
    def define_term(self, term: str, definition: str):
        """Dodaj zdefiniowane pojęcie"""
        self.defined_terms[term] = definition
    
    def add_open_thread(self, batch_num: int, description: str, expected_section: str = ""):
        """Dodaj otwarty wątek (zapowiedź tematu)"""
        self.open_threads.append(OpenThread(
            introduced_in_batch=batch_num,
            thread_description=description,
            expected_in_section=expected_section
        ))
    
    def resolve_thread(self, thread_description: str, batch_num: int):
        """Oznacz wątek jako rozwiązany"""
        for thread in self.open_threads:
            if thread.thread_description == thread_description and not thread.resolved:
                thread.resolved = True
                thread.resolved_in_batch = batch_num
                break
    
    def get_unresolved_threads(self) -> List[OpenThread]:
        """Pobierz nierozwiązane wątki"""
        return [t for t in self.open_threads if not t.resolved]
    
    def get_claims_for_section(self, h2: str) -> List[KeyClaim]:
        """Pobierz twierdzenia z danej sekcji"""
        return [c for c in self.key_claims if c.section_h2 == h2]
    
    def get_recent_claims(self, last_n: int = 3) -> List[KeyClaim]:
        """Pobierz ostatnie N twierdzeń"""
        return self.key_claims[-last_n:] if self.key_claims else []
    
    def to_dict(self) -> dict:
        """Serializacja do dict (dla Firestore)"""
        return {
            "project_id": self.project_id,
            "thesis": self.thesis,
            "thesis_keywords": self.thesis_keywords,
            "key_claims": [c.to_dict() for c in self.key_claims],
            "introduced_entities": self.introduced_entities,
            "defined_terms": self.defined_terms,
            "open_threads": [t.to_dict() for t in self.open_threads],
            "running_summary": self.running_summary,
            "tone_summary": self.tone_summary,
            "last_updated_batch": self.last_updated_batch,
            "total_words_written": self.total_words_written
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ArticleMemory":
        """Deserializacja z dict"""
        memory = cls(project_id=data.get("project_id", ""))
        memory.thesis = data.get("thesis", "")
        memory.thesis_keywords = data.get("thesis_keywords", [])
        memory.key_claims = [
            KeyClaim(**c) for c in data.get("key_claims", [])
        ]
        memory.introduced_entities = data.get("introduced_entities", {})
        memory.defined_terms = data.get("defined_terms", {})
        memory.open_threads = [
            OpenThread(**t) for t in data.get("open_threads", [])
        ]
        memory.running_summary = data.get("running_summary", "")
        memory.tone_summary = data.get("tone_summary", "")
        memory.last_updated_batch = data.get("last_updated_batch", 0)
        memory.total_words_written = data.get("total_words_written", 0)
        return memory
    
    def generate_context_for_gpt(self, current_batch_num: int, current_h2: str = "") -> str:
        """
        Generuj kontekst dla GPT do wstrzyknięcia w prompt.
        
        Returns:
            String z kontekstem artykułu do dodania do gpt_prompt
        """
        lines = []
        lines.append("=" * 60)
        lines.append("🧠 PAMIĘĆ ARTYKUŁU - KONTEKST GLOBALNY")
        lines.append("=" * 60)
        lines.append("")
        
        # Teza główna
        if self.thesis:
            lines.append(f"📌 GŁÓWNA TEZA ARTYKUŁU:")
            lines.append(f"   \"{self.thesis}\"")
            lines.append("")
        
        # Running summary
        if self.running_summary:
            lines.append(f"📝 CO JUŻ NAPISANO (streszczenie):")
            lines.append(f"   {self.running_summary}")
            lines.append("")
        
        # Kluczowe twierdzenia z poprzednich batchy
        if self.key_claims:
            lines.append(f"📋 KLUCZOWE TWIERDZENIA Z POPRZEDNICH SEKCJI:")
            for claim in self.key_claims[-5:]:  # max 5 ostatnich
                lines.append(f"   • [Batch {claim.batch_number}] {claim.claim_text}")
            lines.append("")
            lines.append("   💡 Możesz się odwoływać: \"Jak wspomniano wcześniej...\"")
            lines.append("")
        
        # Wprowadzone encje
        if self.introduced_entities:
            defined = list(self.introduced_entities.keys())[:10]
            lines.append(f"✅ POJĘCIA JUŻ ZDEFINIOWANE (nie tłumacz ponownie):")
            lines.append(f"   {', '.join(defined)}")
            lines.append("")
        
        # Otwarte wątki
        unresolved = self.get_unresolved_threads()
        if unresolved:
            lines.append(f"⏳ OTWARTE WĄTKI (zapowiedziane tematy do rozwinięcia):")
            for thread in unresolved[:3]:
                expected = f" → rozwiń w: {thread.expected_in_section}" if thread.expected_in_section else ""
                lines.append(f"   • {thread.thread_description}{expected}")
            lines.append("")
        
        # Ton
        if self.tone_summary:
            lines.append(f"🎨 TON ARTYKUŁU:")
            lines.append(f"   {self.tone_summary}")
            lines.append("")
        
        return "\n".join(lines)


def extract_thesis_from_intro(intro_text: str, main_keyword: str) -> Tuple[str, List[str]]:
    """
    Wyekstrahuj główną tezę z batcha intro.
    
    Args:
        intro_text: Tekst pierwszego batcha
        main_keyword: Główna fraza kluczowa
        
    Returns:
        Tuple (thesis_text, thesis_keywords)
    """
    # Usuń tagi HTML
    clean_text = re.sub(r'<[^>]+>', ' ', intro_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Podziel na zdania
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return "", []
    
    # Pierwszych 2-3 zdań to zazwyczaj teza
    thesis_sentences = sentences[:3]
    thesis = ". ".join(thesis_sentences) + "."
    
    # Ogranicz długość
    if len(thesis) > 300:
        thesis = thesis[:300].rsplit(' ', 1)[0] + "..."
    
    # Wyekstrahuj kluczowe słowa z tezy
    thesis_keywords = [main_keyword]
    
    # Dodaj rzeczowniki z tezy (prosty regex)
    words = re.findall(r'\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,}\b', thesis)
    thesis_keywords.extend(words[:5])
    
    return thesis, list(set(thesis_keywords))


def extract_claims_from_batch(
    batch_text: str,
    batch_number: int,
    h2_sections: List[str]
) -> List[KeyClaim]:
    """
    Wyekstrahuj kluczowe twierdzenia z batcha.
    
    Heurystyka: Szukamy zdań z silnymi czasownikami/sformułowaniami.
    """
    claims = []
    
    # Usuń tagi HTML
    clean_text = re.sub(r'<[^>]+>', ' ', batch_text)
    clean_text = re.sub(r'^h[23]:\s*.+$', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Podziel na zdania
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]
    
    # Wskaźniki kluczowych twierdzeń
    claim_indicators = [
        r'\bpolega na\b',
        r'\boznacza\b',
        r'\bjest to\b',
        r'\bstanowi\b',
        r'\bwymaga\b',
        r'\bprowadzi do\b',
        r'\bskutkuje\b',
        r'\bpozwala na\b',
        r'\bkonieczne jest\b',
        r'\bnależy\b',
        r'\bwarto\b',
        r'\bkluczowe\b',
        r'\bistotne\b',
        r'\bnajważniejsz\b',
    ]
    
    h2 = h2_sections[0] if h2_sections else "Sekcja"
    
    for sentence in sentences:
        for indicator in claim_indicators:
            if re.search(indicator, sentence.lower()):
                # To potencjalne twierdzenie
                claim_text = sentence[:200] if len(sentence) > 200 else sentence
                
                # Wyekstrahuj encje (proste: słowa zaczynające się wielką literą)
                entities = re.findall(r'\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b', claim_text)
                entities = [e for e in entities if len(e) > 3][:3]
                
                claims.append(KeyClaim(
                    batch_number=batch_number,
                    section_h2=h2,
                    claim_text=claim_text,
                    entities_mentioned=entities
                ))
                break  # Jeden claim per zdanie
        
        if len(claims) >= 3:  # Max 3 claims per batch
            break
    
    return claims


def detect_open_threads(batch_text: str, batch_number: int) -> List[OpenThread]:
    """
    Wykryj otwarte wątki (zapowiedzi tematów) w tekście.
    
    Szuka fraz typu "w dalszej części", "poniżej omówimy", itp.
    """
    threads = []
    
    # Wzorce zapowiedzi
    forward_patterns = [
        (r'w dalszej części\s+(?:artykułu\s+)?(?:omówimy|przedstawimy|opiszemy)\s+(.{10,60})', "later"),
        (r'poniżej\s+(?:znajdziesz|przedstawimy|omówimy)\s+(.{10,60})', "below"),
        (r'w kolejnej sekcji\s+(.{10,60})', "next_section"),
        (r'wrócimy do\s+(.{10,50})\s+(?:później|w dalszej części)', "return"),
        (r'szczegóły\s+(?:dotyczące\s+)?(.{10,50})\s+(?:znajdziesz|przedstawimy)\s+(?:poniżej|dalej)', "details"),
    ]
    
    clean_text = re.sub(r'<[^>]+>', ' ', batch_text)
    
    for pattern, thread_type in forward_patterns:
        matches = re.findall(pattern, clean_text.lower())
        for match in matches:
            threads.append(OpenThread(
                introduced_in_batch=batch_number,
                thread_description=match.strip()[:100]
            ))
    
    return threads[:2]  # Max 2 wątki per batch


def update_article_memory(
    memory: ArticleMemory,
    batch_text: str,
    batch_number: int,
    h2_sections: List[str],
    entities_used: List[str] = None
) -> ArticleMemory:
    """
    Aktualizuj pamięć artykułu po zatwierdzeniu batcha.
    
    Args:
        memory: Istniejąca pamięć artykułu
        batch_text: Tekst zatwierdzonego batcha
        batch_number: Numer batcha
        h2_sections: Nagłówki H2 w tym batchu
        entities_used: Lista użytych encji (opcjonalnie)
        
    Returns:
        Zaktualizowana ArticleMemory
    """
    # Wyekstrahuj tezę z intro
    if batch_number == 1 and not memory.thesis:
        # Potrzebujemy main_keyword - spróbuj wyekstrahować
        # W prawdziwym użyciu to będzie przekazane z projektu
        main_keyword = ""
        thesis, keywords = extract_thesis_from_intro(batch_text, main_keyword)
        memory.thesis = thesis
        memory.thesis_keywords = keywords
    
    # Wyekstrahuj claims
    new_claims = extract_claims_from_batch(batch_text, batch_number, h2_sections)
    memory.key_claims.extend(new_claims)
    
    # Ogranicz liczbę claims (keep recent)
    if len(memory.key_claims) > 15:
        memory.key_claims = memory.key_claims[-15:]
    
    # Dodaj encje
    if entities_used:
        for entity in entities_used:
            memory.introduce_entity(entity, batch_number)
    
    # Wykryj otwarte wątki
    new_threads = detect_open_threads(batch_text, batch_number)
    memory.open_threads.extend(new_threads)
    
    # Aktualizuj running_summary
    memory.running_summary = _generate_running_summary(memory, batch_text, batch_number)
    
    # Policz słowa
    word_count = len(re.findall(r'\b\w+\b', batch_text))
    memory.total_words_written += word_count
    memory.last_updated_batch = batch_number
    
    return memory


def _generate_running_summary(memory: ArticleMemory, new_batch_text: str, batch_num: int) -> str:
    """Generuj dynamiczne streszczenie artykułu (max 150 słów)"""
    
    # Zbierz kluczowe elementy
    parts = []
    
    if memory.thesis:
        parts.append(f"Artykuł omawia: {memory.thesis[:100]}")
    
    if memory.key_claims:
        # Pogrupuj claims per batch
        claims_summary = []
        for claim in memory.key_claims[-5:]:
            claims_summary.append(f"Batch {claim.batch_number}: {claim.claim_text[:50]}...")
        
        if claims_summary:
            parts.append("Omówiono: " + "; ".join(claims_summary[-3:]))
    
    if memory.introduced_entities:
        entities = list(memory.introduced_entities.keys())[:5]
        parts.append(f"Zdefiniowano pojęcia: {', '.join(entities)}")
    
    summary = " ".join(parts)
    
    # Ogranicz długość
    if len(summary) > 500:
        summary = summary[:500].rsplit(' ', 1)[0] + "..."
    
    return summary


def create_article_memory(project_id: str, main_keyword: str = "") -> ArticleMemory:
    """Utwórz nową pamięć artykułu"""
    memory = ArticleMemory(project_id=project_id)
    if main_keyword:
        memory.thesis_keywords = [main_keyword]
    return memory


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    # Utwórz pamięć
    memory = create_article_memory("test_project", "ubezwłasnowolnienie")
    
    # Symuluj batch 1 (intro)
    intro_text = """
    Ubezwłasnowolnienie to instytucja prawna mająca na celu ochronę osób, 
    które z powodu choroby psychicznej lub innych zaburzeń nie są w stanie 
    samodzielnie kierować swoim postępowaniem. W dalszej części artykułu 
    omówimy procedurę sądową oraz skutki prawne tego rozwiązania.
    """
    
    memory = update_article_memory(
        memory=memory,
        batch_text=intro_text,
        batch_number=1,
        h2_sections=["Wprowadzenie"],
        entities_used=["Kodeks Cywilny"]
    )
    
    print("=== ARTICLE MEMORY PO BATCH 1 ===")
    print(f"Thesis: {memory.thesis}")
    print(f"Claims: {len(memory.key_claims)}")
    print(f"Open threads: {[t.thread_description for t in memory.open_threads]}")
    print()
    
    # Generuj kontekst dla GPT
    context = memory.generate_context_for_gpt(current_batch_num=2)
    print("=== KONTEKST DLA GPT (BATCH 2) ===")
    print(context)
