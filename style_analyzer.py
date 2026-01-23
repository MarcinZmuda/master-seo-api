"""
🎨 STYLE ANALYZER v1.0
Persona Fingerprint - analiza i utrzymanie spójności stylu

Rozwiązuje problem niespójnego tonu między batchami:
- Analizuje formalność, długość zdań, użycie strony biernej
- Generuje "fingerprint" stylu do wstrzyknięcia w kolejne batche
- Wykrywa odchylenia od ustalnego tonu

Autor: SEO Master API v36.2
"""

import re
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class FormalityLevel(Enum):
    """Poziom formalności tekstu"""
    CASUAL = "casual"           # Ty, proste słowa, potoczny
    SEMI_FORMAL = "semi_formal" # Mieszany, ale profesjonalny
    FORMAL = "formal"           # Państwo, Pan/Pani, terminologia
    ACADEMIC = "academic"       # Bardzo formalny, naukowy


class PersonalPronouns(Enum):
    """Sposób zwracania się do czytelnika"""
    TY = "ty"               # "Możesz złożyć wniosek..."
    WY = "wy"               # "Możecie złożyć..."  
    PANSTWO = "panstwo"     # "Państwo mogą złożyć..."
    BEZOSOBOWO = "bezosobowo"  # "Wniosek można złożyć..."


@dataclass
class StyleFingerprint:
    """
    Fingerprint stylistyczny tekstu.
    
    Używany do utrzymania spójności między batchami.
    """
    # Formalność (0.0 = casual, 1.0 = academic)
    formality_score: float = 0.5
    formality_level: FormalityLevel = FormalityLevel.SEMI_FORMAL
    
    # Struktura zdań
    sentence_length_avg: float = 18.0      # średnia długość zdania (słowa)
    sentence_length_std: float = 5.0       # odchylenie standardowe
    sentence_variety: float = 0.3          # współczynnik zmienności (CV)
    
    # Głos
    passive_voice_ratio: float = 0.15      # % zdań w stronie biernej
    personal_pronouns: PersonalPronouns = PersonalPronouns.BEZOSOBOWO
    
    # Styl
    transition_words_ratio: float = 0.25   # % zdań ze słowami przejściowymi
    question_ratio: float = 0.05           # % pytań retorycznych
    example_ratio: float = 0.1             # % zdań z przykładami
    
    # Słownictwo
    avg_word_length: float = 6.5           # średnia długość słowa
    complex_words_ratio: float = 0.15      # % słów > 3 sylaby
    
    # Wzorcowe elementy (do naśladowania)
    example_sentences: List[str] = field(default_factory=list)  # 2-3 wzorcowe zdania
    preferred_transitions: List[str] = field(default_factory=list)  # preferowane słowa łączące
    forbidden_patterns: List[str] = field(default_factory=list)  # czego unikać
    
    # Meta
    analyzed_batches: int = 0
    total_sentences_analyzed: int = 0
    
    def to_dict(self) -> dict:
        return {
            "formality_score": self.formality_score,
            "formality_level": self.formality_level.value,
            "sentence_length_avg": self.sentence_length_avg,
            "sentence_length_std": self.sentence_length_std,
            "sentence_variety": self.sentence_variety,
            "passive_voice_ratio": self.passive_voice_ratio,
            "personal_pronouns": self.personal_pronouns.value,
            "transition_words_ratio": self.transition_words_ratio,
            "question_ratio": self.question_ratio,
            "example_ratio": self.example_ratio,
            "avg_word_length": self.avg_word_length,
            "complex_words_ratio": self.complex_words_ratio,
            "example_sentences": self.example_sentences,
            "preferred_transitions": self.preferred_transitions,
            "forbidden_patterns": self.forbidden_patterns,
            "analyzed_batches": self.analyzed_batches,
            "total_sentences_analyzed": self.total_sentences_analyzed
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StyleFingerprint":
        fp = cls()
        fp.formality_score = data.get("formality_score", 0.5)
        fp.formality_level = FormalityLevel(data.get("formality_level", "semi_formal"))
        fp.sentence_length_avg = data.get("sentence_length_avg", 18.0)
        fp.sentence_length_std = data.get("sentence_length_std", 5.0)
        fp.sentence_variety = data.get("sentence_variety", 0.3)
        fp.passive_voice_ratio = data.get("passive_voice_ratio", 0.15)
        fp.personal_pronouns = PersonalPronouns(data.get("personal_pronouns", "bezosobowo"))
        fp.transition_words_ratio = data.get("transition_words_ratio", 0.25)
        fp.question_ratio = data.get("question_ratio", 0.05)
        fp.example_ratio = data.get("example_ratio", 0.1)
        fp.avg_word_length = data.get("avg_word_length", 6.5)
        fp.complex_words_ratio = data.get("complex_words_ratio", 0.15)
        fp.example_sentences = data.get("example_sentences", [])
        fp.preferred_transitions = data.get("preferred_transitions", [])
        fp.forbidden_patterns = data.get("forbidden_patterns", [])
        fp.analyzed_batches = data.get("analyzed_batches", 0)
        fp.total_sentences_analyzed = data.get("total_sentences_analyzed", 0)
        return fp
    
    def generate_style_instructions(self) -> str:
        """Generuj instrukcje stylistyczne dla GPT"""
        lines = []
        lines.append("=" * 60)
        lines.append("🎨 STYL PISANIA - ZACHOWAJ SPÓJNOŚĆ!")
        lines.append("=" * 60)
        lines.append("")
        
        # Formalność
        formality_desc = {
            FormalityLevel.CASUAL: "Nieformalny, przyjazny, prosty język",
            FormalityLevel.SEMI_FORMAL: "Profesjonalny ale przystępny",
            FormalityLevel.FORMAL: "Formalny, używaj 'Państwo', 'Pan/Pani'",
            FormalityLevel.ACADEMIC: "Bardzo formalny, terminologia naukowa"
        }
        lines.append(f"📊 FORMALNOŚĆ: {formality_desc[self.formality_level]}")
        lines.append("")
        
        # Zwracanie się
        pronouns_desc = {
            PersonalPronouns.TY: "Zwracaj się per 'Ty' (możesz, powinieneś)",
            PersonalPronouns.WY: "Zwracaj się per 'Wy' (możecie, powinniście)",
            PersonalPronouns.PANSTWO: "Używaj 'Państwo' (mogą Państwo, Państwa sprawa)",
            PersonalPronouns.BEZOSOBOWO: "Pisz bezosobowo (można, należy, warto)"
        }
        lines.append(f"👤 FORMA ZWRACANIA: {pronouns_desc[self.personal_pronouns]}")
        lines.append("")
        
        # Zdania
        lines.append(f"📏 DŁUGOŚĆ ZDAŃ:")
        lines.append(f"   • Średnio: {self.sentence_length_avg:.0f} słów (zakres: {self.sentence_length_avg-5:.0f}-{self.sentence_length_avg+5:.0f})")
        lines.append(f"   • Zmienność: {'Wysoka - mieszaj krótkie i długie' if self.sentence_variety > 0.35 else 'Umiarkowana - zachowaj równomierne'}")
        lines.append("")
        
        # Głos
        if self.passive_voice_ratio > 0.25:
            lines.append(f"🔊 GŁOS: Częściej strona bierna (jest wykonywane, zostaje złożony)")
        elif self.passive_voice_ratio < 0.1:
            lines.append(f"🔊 GŁOS: Preferuj stronę czynną (wykonuje się, składa się)")
        else:
            lines.append(f"🔊 GŁOS: Mieszaj stronę czynną i bierną naturalnie")
        lines.append("")
        
        # Przykładowe zdania
        if self.example_sentences:
            lines.append(f"✨ WZORCOWE ZDANIA Z POPRZEDNICH BATCHY:")
            for ex in self.example_sentences[:2]:
                lines.append(f"   \"{ex[:100]}...\"" if len(ex) > 100 else f"   \"{ex}\"")
            lines.append("")
        
        # Preferowane przejścia
        if self.preferred_transitions:
            lines.append(f"🔗 PREFEROWANE SŁOWA ŁĄCZĄCE:")
            lines.append(f"   {', '.join(self.preferred_transitions[:6])}")
            lines.append("")
        
        # Zakazane wzorce
        if self.forbidden_patterns:
            lines.append(f"⛔ UNIKAJ TYCH SFORMUŁOWAŃ:")
            for pattern in self.forbidden_patterns[:4]:
                lines.append(f"   • {pattern}")
            lines.append("")
        
        return "\n".join(lines)


class StyleAnalyzer:
    """
    Analizator stylu tekstu.
    
    Używany po każdym batchu do aktualizacji fingerprinta.
    """
    
    # Słowa formalne
    FORMAL_WORDS = {
        "należy", "powinno", "wymaga", "stanowi", "zgodnie",
        "państwo", "pani", "pana", "przedmiotowy", "niniejszy",
        "powyższy", "stosownie", "właściwy", "odpowiedni"
    }
    
    # Słowa nieformalne
    INFORMAL_WORDS = {
        "fajnie", "super", "mega", "bardzo", "naprawdę",
        "normalnie", "po prostu", "w sumie", "generalnie",
        "szczerze", "właściwie", "chyba"
    }
    
    # Słowa przejściowe
    TRANSITION_WORDS = [
        "jednak", "natomiast", "ponadto", "dodatkowo", "również",
        "w związku z tym", "dlatego", "zatem", "tym samym",
        "przede wszystkim", "po pierwsze", "po drugie",
        "z kolei", "następnie", "wreszcie", "podsumowując",
        "innymi słowy", "to znaczy", "mianowicie"
    ]
    
    # Wzorce strony biernej (polskiej)
    PASSIVE_PATTERNS = [
        r'\bjest\s+\w+[aoy]n[aey]?\b',  # jest wykonany/a/e
        r'\bzostał[aoy]?\s+\w+[aoy]n[aey]?\b',  # został złożony
        r'\bzostaje\s+\w+[aoy]n[aey]?\b',  # zostaje wykonany
        r'\bbyło\s+\w+[aoy]n[aey]?\b',  # było zrobione
    ]
    
    # Wzorce przykładów
    EXAMPLE_PATTERNS = [
        r'\bna przykład\b',
        r'\bnp\.\s',
        r'\bprzykładowo\b',
        r'\bwyobraźmy sobie\b',
        r'\bzałóżmy,? że\b',
        r'\bw praktyce\b'
    ]
    
    def __init__(self, existing_fingerprint: Optional[StyleFingerprint] = None):
        self.fingerprint = existing_fingerprint or StyleFingerprint()
    
    def analyze_batch(self, batch_text: str) -> StyleFingerprint:
        """
        Analizuj batch i zaktualizuj fingerprint.
        
        Args:
            batch_text: Tekst batcha do analizy
            
        Returns:
            Zaktualizowany StyleFingerprint
        """
        # Wyczyść tekst
        clean_text = self._clean_text(batch_text)
        
        # Podziel na zdania
        sentences = self._split_sentences(clean_text)
        
        if len(sentences) < 3:
            return self.fingerprint
        
        # Analizuj metryki
        new_metrics = self._compute_metrics(clean_text, sentences)
        
        # Połącz z istniejącym fingerprintem (weighted average)
        self._merge_metrics(new_metrics)
        
        # Znajdź przykładowe zdania
        self._find_example_sentences(sentences)
        
        # Znajdź preferowane przejścia
        self._find_preferred_transitions(clean_text)
        
        # Aktualizuj meta
        self.fingerprint.analyzed_batches += 1
        self.fingerprint.total_sentences_analyzed += len(sentences)
        
        return self.fingerprint
    
    def _clean_text(self, text: str) -> str:
        """Usuń tagi HTML i nagłówki"""
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'^h[23]:\s*.+$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def _split_sentences(self, text: str) -> List[str]:
        """Podziel tekst na zdania"""
        # Uwzględnij skróty
        text = re.sub(r'\b(np|m\.in|tj|tzw|itd|itp|ok|ul|art|ust|pkt)\.\s', r'\1<DOT> ', text)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def _compute_metrics(self, text: str, sentences: List[str]) -> dict:
        """Oblicz metryki stylistyczne"""
        metrics = {}
        
        # Długość zdań
        sentence_lengths = [len(s.split()) for s in sentences]
        metrics["sentence_length_avg"] = statistics.mean(sentence_lengths)
        metrics["sentence_length_std"] = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 5.0
        metrics["sentence_variety"] = metrics["sentence_length_std"] / metrics["sentence_length_avg"] if metrics["sentence_length_avg"] > 0 else 0.3
        
        # Formalność
        text_lower = text.lower()
        formal_count = sum(1 for w in self.FORMAL_WORDS if w in text_lower)
        informal_count = sum(1 for w in self.INFORMAL_WORDS if w in text_lower)
        
        total_words = len(text.split())
        formality_raw = (formal_count - informal_count) / max(total_words / 100, 1)
        metrics["formality_score"] = max(0, min(1, 0.5 + formality_raw * 0.1))
        
        # Strona bierna
        passive_count = sum(1 for pattern in self.PASSIVE_PATTERNS 
                          for _ in re.findall(pattern, text_lower))
        metrics["passive_voice_ratio"] = passive_count / max(len(sentences), 1)
        
        # Słowa przejściowe
        transition_count = sum(1 for t in self.TRANSITION_WORDS if t in text_lower)
        metrics["transition_words_ratio"] = transition_count / max(len(sentences), 1)
        
        # Pytania
        question_count = text.count('?')
        metrics["question_ratio"] = question_count / max(len(sentences), 1)
        
        # Przykłady
        example_count = sum(1 for pattern in self.EXAMPLE_PATTERNS 
                          for _ in re.findall(pattern, text_lower))
        metrics["example_ratio"] = example_count / max(len(sentences), 1)
        
        # Długość słów
        words = re.findall(r'\b\w+\b', text)
        if words:
            metrics["avg_word_length"] = statistics.mean(len(w) for w in words)
            # Słowa > 3 sylaby (przybliżenie: > 8 liter)
            complex_count = sum(1 for w in words if len(w) > 8)
            metrics["complex_words_ratio"] = complex_count / len(words)
        else:
            metrics["avg_word_length"] = 6.5
            metrics["complex_words_ratio"] = 0.15
        
        # Zaimki osobowe
        metrics["personal_pronouns"] = self._detect_pronouns(text_lower)
        
        # Poziom formalności
        if metrics["formality_score"] > 0.7:
            metrics["formality_level"] = FormalityLevel.FORMAL
        elif metrics["formality_score"] > 0.55:
            metrics["formality_level"] = FormalityLevel.SEMI_FORMAL
        elif metrics["formality_score"] < 0.35:
            metrics["formality_level"] = FormalityLevel.CASUAL
        else:
            metrics["formality_level"] = FormalityLevel.SEMI_FORMAL
        
        return metrics
    
    def _detect_pronouns(self, text: str) -> PersonalPronouns:
        """Wykryj sposób zwracania się"""
        ty_count = len(re.findall(r'\b(możesz|musisz|powinieneś|twój|twoja|twoje|ciebie|ci)\b', text))
        wy_count = len(re.findall(r'\b(możecie|musicie|powinniście|wasz|wasza|wasze|was|wam)\b', text))
        panstwo_count = len(re.findall(r'\b(państwo|państwa|państwu|pana|pani|pańsk)\b', text))
        bezos_count = len(re.findall(r'\b(można|należy|warto|trzeba|powinno się)\b', text))
        
        counts = {
            PersonalPronouns.TY: ty_count,
            PersonalPronouns.WY: wy_count,
            PersonalPronouns.PANSTWO: panstwo_count,
            PersonalPronouns.BEZOSOBOWO: bezos_count
        }
        
        return max(counts, key=counts.get)
    
    def _merge_metrics(self, new_metrics: dict):
        """Połącz nowe metryki z istniejącym fingerprintem"""
        # Waga dla nowych danych (im więcej batchy, tym mniejsza waga nowych)
        weight = 1 / (self.fingerprint.analyzed_batches + 1)
        old_weight = 1 - weight
        
        # Metryki liczbowe - weighted average
        numeric_fields = [
            "formality_score", "sentence_length_avg", "sentence_length_std",
            "sentence_variety", "passive_voice_ratio", "transition_words_ratio",
            "question_ratio", "example_ratio", "avg_word_length", "complex_words_ratio"
        ]
        
        for field in numeric_fields:
            old_val = getattr(self.fingerprint, field)
            new_val = new_metrics.get(field, old_val)
            setattr(self.fingerprint, field, old_weight * old_val + weight * new_val)
        
        # Enum fields - użyj nowych jeśli to pierwszy batch, inaczej zachowaj
        if self.fingerprint.analyzed_batches == 0:
            self.fingerprint.formality_level = new_metrics.get("formality_level", FormalityLevel.SEMI_FORMAL)
            self.fingerprint.personal_pronouns = new_metrics.get("personal_pronouns", PersonalPronouns.BEZOSOBOWO)
    
    def _find_example_sentences(self, sentences: List[str]):
        """Znajdź wzorcowe zdania (dobrze napisane)"""
        good_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            # Kryteria dobrego zdania:
            # - 12-25 słów
            # - Zawiera słowo przejściowe lub przykład
            # - Nie zaczyna się od "I" (lista)
            
            if 12 <= word_count <= 25:
                has_transition = any(t in sentence.lower() for t in self.TRANSITION_WORDS[:10])
                has_example = any(re.search(p, sentence.lower()) for p in self.EXAMPLE_PATTERNS)
                
                if has_transition or has_example:
                    good_sentences.append(sentence)
        
        # Zachowaj max 3 przykładowe zdania
        if good_sentences:
            self.fingerprint.example_sentences = good_sentences[:3]
    
    def _find_preferred_transitions(self, text: str):
        """Znajdź preferowane słowa przejściowe"""
        text_lower = text.lower()
        
        transition_counts = {}
        for t in self.TRANSITION_WORDS:
            count = text_lower.count(t)
            if count > 0:
                transition_counts[t] = count
        
        # Top 6 najczęściej używanych
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        self.fingerprint.preferred_transitions = [t[0] for t in sorted_transitions[:6]]
    
    def check_style_deviation(self, new_batch_text: str) -> Dict:
        """
        Sprawdź czy nowy batch odbiega od ustalonego stylu.
        
        Returns:
            Dict z informacjami o odchyleniach
        """
        clean_text = self._clean_text(new_batch_text)
        sentences = self._split_sentences(clean_text)
        
        if len(sentences) < 3:
            return {"deviations": [], "severity": "NONE"}
        
        new_metrics = self._compute_metrics(clean_text, sentences)
        
        deviations = []
        
        # Sprawdź odchylenie długości zdań
        len_diff = abs(new_metrics["sentence_length_avg"] - self.fingerprint.sentence_length_avg)
        if len_diff > 5:
            deviations.append({
                "type": "sentence_length",
                "expected": f"{self.fingerprint.sentence_length_avg:.0f} słów",
                "actual": f"{new_metrics['sentence_length_avg']:.0f} słów",
                "suggestion": f"Dostosuj długość zdań (obecnie: {new_metrics['sentence_length_avg']:.0f}, cel: {self.fingerprint.sentence_length_avg:.0f})"
            })
        
        # Sprawdź zaimki
        if new_metrics["personal_pronouns"] != self.fingerprint.personal_pronouns:
            deviations.append({
                "type": "pronouns",
                "expected": self.fingerprint.personal_pronouns.value,
                "actual": new_metrics["personal_pronouns"].value,
                "suggestion": f"Używaj formy '{self.fingerprint.personal_pronouns.value}' zamiast '{new_metrics['personal_pronouns'].value}'"
            })
        
        # Sprawdź formalność
        form_diff = abs(new_metrics["formality_score"] - self.fingerprint.formality_score)
        if form_diff > 0.2:
            deviations.append({
                "type": "formality",
                "expected": f"{self.fingerprint.formality_score:.2f}",
                "actual": f"{new_metrics['formality_score']:.2f}",
                "suggestion": "Dostosuj poziom formalności do poprzednich batchy"
            })
        
        # Określ severity
        if len(deviations) >= 3:
            severity = "HIGH"
        elif len(deviations) >= 1:
            severity = "MEDIUM"
        else:
            severity = "NONE"
        
        return {
            "deviations": deviations,
            "severity": severity,
            "recommendation": "Popraw tekst zgodnie z sugestiami" if deviations else "Styl zgodny"
        }


def analyze_style(text: str, existing_fingerprint: Optional[dict] = None) -> dict:
    """
    Główna funkcja do analizy stylu.
    
    Args:
        text: Tekst do analizy
        existing_fingerprint: Istniejący fingerprint (dict) do aktualizacji
        
    Returns:
        Dict z zaktualizowanym fingerprintem
    """
    fp = StyleFingerprint.from_dict(existing_fingerprint) if existing_fingerprint else StyleFingerprint()
    analyzer = StyleAnalyzer(fp)
    updated_fp = analyzer.analyze_batch(text)
    return updated_fp.to_dict()


def check_style_consistency(new_text: str, fingerprint_dict: dict) -> dict:
    """
    Sprawdź spójność stylu nowego tekstu z fingerprintem.
    
    Args:
        new_text: Nowy tekst do sprawdzenia
        fingerprint_dict: Fingerprint z poprzednich batchy
        
    Returns:
        Dict z informacjami o odchyleniach
    """
    fp = StyleFingerprint.from_dict(fingerprint_dict)
    analyzer = StyleAnalyzer(fp)
    return analyzer.check_style_deviation(new_text)


def generate_style_prompt(fingerprint_dict: dict) -> str:
    """
    Generuj instrukcje stylistyczne do wstrzyknięcia w prompt GPT.
    
    Args:
        fingerprint_dict: Fingerprint z poprzednich batchy
        
    Returns:
        String z instrukcjami stylistycznymi
    """
    fp = StyleFingerprint.from_dict(fingerprint_dict)
    return fp.generate_style_instructions()


# ============================================
# PRZYKŁAD UŻYCIA
# ============================================
if __name__ == "__main__":
    # Przykładowy tekst
    sample_text = """
    Ubezwłasnowolnienie to instytucja prawna, która ma na celu ochronę osób 
    niezdolnych do samodzielnego kierowania swoim postępowaniem. Należy pamiętać, 
    że procedura ta wymaga spełnienia określonych przesłanek. 
    
    Po pierwsze, osoba musi cierpieć na chorobę psychiczną lub inną dysfunkcję. 
    Po drugie, stan ten musi uniemożliwiać jej samodzielne funkcjonowanie. 
    
    W praktyce oznacza to, że można złożyć wniosek do sądu okręgowego. 
    Wniosek powinien zawierać dokumentację medyczną oraz uzasadnienie.
    """
    
    # Analizuj
    fingerprint = analyze_style(sample_text)
    
    print("=== STYLE FINGERPRINT ===")
    print(f"Formality: {fingerprint['formality_score']:.2f} ({fingerprint['formality_level']})")
    print(f"Sentence length: {fingerprint['sentence_length_avg']:.1f} ± {fingerprint['sentence_length_std']:.1f}")
    print(f"Passive voice: {fingerprint['passive_voice_ratio']:.1%}")
    print(f"Personal pronouns: {fingerprint['personal_pronouns']}")
    print(f"Transitions: {fingerprint['preferred_transitions']}")
    print()
    
    # Generuj instrukcje
    instructions = generate_style_prompt(fingerprint)
    print("=== STYLE INSTRUCTIONS FOR GPT ===")
    print(instructions)
