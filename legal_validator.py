"""
===============================================================================
⚖️ LEGAL HARD-LOCK VALIDATOR v38.0
===============================================================================
Wymusza użycie TYLKO zatwierdzonych przepisów prawnych w treściach YMYL.

FUNKCJE:
- Whitelist zamiast blacklist - tylko zatwierdzone przepisy dozwolone
- Wykrywa nielegalne artykuły (spoza detected_articles)
- Wykrywa nielegalne orzeczenia (spoza legal_judgments)
- Auto-remove lub flag dla review

INTEGRACJA:
- Wywoływany w batch_simple dla projektów is_legal=True
- Może automatycznie usuwać nielegalne przepisy
- Dodaje warnings do response
===============================================================================
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


class ViolationType(Enum):
    """Typ naruszenia."""
    ILLEGAL_ARTICLE = "ILLEGAL_ARTICLE"
    ILLEGAL_JUDGMENT = "ILLEGAL_JUDGMENT"
    SUSPICIOUS_CITATION = "SUSPICIOUS_CITATION"


class ViolationSeverity(Enum):
    """Powaga naruszenia."""
    CRITICAL = "CRITICAL"  # Pewna halucynacja
    WARNING = "WARNING"    # Możliwa halucynacja
    INFO = "INFO"          # Do sprawdzenia


@dataclass
class LegalViolation:
    """Pojedyncze naruszenie legal hard-lock."""
    type: ViolationType
    severity: ViolationSeverity
    found_text: str
    normalized: str
    position: int
    context: str  # Fragment tekstu wokół naruszenia
    suggestion: str
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "found_text": self.found_text,
            "normalized": self.normalized,
            "position": self.position,
            "context": self.context,
            "suggestion": self.suggestion
        }


@dataclass
class LegalRemoval:
    """Informacja o usunięciu."""
    original: str
    replacement: str
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason
        }


@dataclass
class LegalValidationResult:
    """Wynik walidacji legal hard-lock."""
    is_valid: bool
    violations: List[LegalViolation]
    removals: List[LegalRemoval]
    processed_text: str
    stats: Dict[str, int]
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "violations_count": len(self.violations),
            "removals_count": len(self.removals),
            "violations": [v.to_dict() for v in self.violations],
            "removals": [r.to_dict() for r in self.removals],
            "stats": self.stats
        }


class LegalHardLockValidator:
    """
    Walidator wymuszający użycie TYLKO zatwierdzonych przepisów prawnych.
    
    ZASADA: Whitelist, nie blacklist.
    Każdy przepis/orzeczenie MUSI być na liście dozwolonych.
    """
    
    # ================================================================
    # WZORCE WYKRYWAJĄCE PRZEPISY
    # ================================================================
    ARTICLE_PATTERNS = [
        # art. 13 § 1 k.c.
        (r"art\.?\s*(\d+[a-z]?)\s*(§\s*\d+(?:\s*(?:pkt|ust\.?)\s*\d+)?)?\s*(k\.?[cp]\.?[cp]?\.?|k\.?r\.?o\.?|k\.?k\.?|k\.?s\.?h\.?|k\.?p\.?|u\.?s\.?p\.?)",
         "standard"),
        
        # artykuł 13 kodeksu cywilnego
        (r"artykuł(?:u|em|owi)?\s+(\d+[a-z]?)\s*(§\s*\d+)?\s*(kodeksu\s+\w+(?:\s+\w+)?)",
         "verbose"),
        
        # przepis art. 13
        (r"przepis(?:u|em|y)?\s+art\.?\s*(\d+[a-z]?)\s*(§\s*\d+)?",
         "reference"),
        
        # § 13 rozporządzenia
        (r"§\s*(\d+)\s*(ust\.?\s*\d+)?\s*(rozporządzeni[aeu]\s+\w+)?",
         "paragraph"),
        
        # ustawa z dnia... art. 5
        (r"ustaw[ayę]\s+(?:z\s+dnia\s+)?[\d\s\w]+\s+art\.?\s*(\d+)",
         "statute"),
    ]
    
    # ================================================================
    # WZORCE WYKRYWAJĄCE SYGNATURY ORZECZEŃ
    # ================================================================
    JUDGMENT_PATTERNS = [
        # III CZP 15/20
        (r"([IVX]+)\s+([A-Z]{2,5})\s+(\d+/\d+)",
         "signature"),
        
        # sygn. akt III CZP 15/20
        (r"sygn\.?\s*(?:akt\.?)?\s*:?\s*([IVX]+)\s*([A-Z]{2,5})\s*(\d+/\d+)",
         "with_prefix"),
        
        # wyrok SN z dnia... III CZP 15/20
        (r"(?:wyrok|postanowienie|uchwała)\s+(?:SN|SA|SO|SR|NSA|WSA|TK)\s+(?:z\s+dnia\s+)?[\d\.\s\w]+,?\s*(?:sygn\.?\s*)?([IVX]+\s*[A-Z]+\s*\d+/\d+)",
         "full_citation"),
        
        # orzeczenie z dnia... (I ACa 123/19)
        (r"(?:orzeczenie|wyrok)\s+z\s+dnia\s+[\d\.\s\w]+\s*\(([IVX]+\s*[A-Z]+\s*\d+/\d+)\)",
         "in_parentheses"),
    ]
    
    # ================================================================
    # NORMALIZACJA KODEKSÓW
    # ================================================================
    CODE_NORMALIZATION = {
        # Kodeks cywilny
        "kc": "k.c.",
        "k.c": "k.c.",
        "k.c.": "k.c.",
        "kodeksu cywilnego": "k.c.",
        "kodeks cywilny": "k.c.",
        
        # Kodeks postępowania cywilnego
        "kpc": "k.p.c.",
        "k.p.c": "k.p.c.",
        "k.p.c.": "k.p.c.",
        "kodeksu postępowania cywilnego": "k.p.c.",
        
        # Kodeks rodzinny i opiekuńczy
        "kro": "k.r.o.",
        "k.r.o": "k.r.o.",
        "k.r.o.": "k.r.o.",
        "kodeksu rodzinnego": "k.r.o.",
        
        # Kodeks karny
        "kk": "k.k.",
        "k.k": "k.k.",
        "k.k.": "k.k.",
        "kodeksu karnego": "k.k.",
        
        # Kodeks postępowania karnego
        "kpk": "k.p.k.",
        "k.p.k": "k.p.k.",
        "k.p.k.": "k.p.k.",
        
        # Kodeks spółek handlowych
        "ksh": "k.s.h.",
        "k.s.h": "k.s.h.",
        "k.s.h.": "k.s.h.",
        
        # Kodeks pracy
        "kp": "k.p.",
        "k.p": "k.p.",
        "k.p.": "k.p.",
        "kodeksu pracy": "k.p.",
    }
    
    # ================================================================
    # BEZPIECZNE ZAMIENNIKI
    # ================================================================
    SAFE_REPLACEMENTS = {
        "article": "odpowiednich przepisów prawa",
        "judgment": "orzecznictwa sądowego",
        "statute": "właściwych regulacji prawnych",
    }
    
    def __init__(self):
        self.stats = {
            "articles_checked": 0,
            "judgments_checked": 0,
            "violations_found": 0,
            "auto_removed": 0
        }
    
    def validate_batch(
        self,
        batch_text: str,
        legal_whitelist: dict,
        auto_remove: bool = True,
        strict_mode: bool = True
    ) -> LegalValidationResult:
        """
        Waliduje batch pod kątem nielegalnych przepisów.
        
        Args:
            batch_text: Tekst do walidacji
            legal_whitelist: Dozwolone przepisy i orzeczenia
            auto_remove: Czy automatycznie usuwać nielegalne
            strict_mode: Czy traktować każde nieznane jako violation
            
        Returns:
            LegalValidationResult z wynikiem walidacji
        """
        self.stats = {
            "articles_checked": 0,
            "judgments_checked": 0,
            "violations_found": 0,
            "auto_removed": 0
        }
        
        violations = []
        removals = []
        processed_text = batch_text
        
        # Buduj sety dozwolonych
        allowed_articles = self._build_allowed_articles(legal_whitelist)
        allowed_judgments = self._build_allowed_judgments(legal_whitelist)
        
        # 1. Sprawdź artykuły
        article_violations, article_removals, processed_text = self._check_articles(
            text=processed_text,
            allowed=allowed_articles,
            auto_remove=auto_remove,
            strict_mode=strict_mode
        )
        violations.extend(article_violations)
        removals.extend(article_removals)
        
        # 2. Sprawdź orzeczenia
        judgment_violations, judgment_removals, processed_text = self._check_judgments(
            text=processed_text,
            allowed=allowed_judgments,
            auto_remove=auto_remove,
            strict_mode=strict_mode
        )
        violations.extend(judgment_violations)
        removals.extend(judgment_removals)
        
        # Aktualizuj statystyki
        self.stats["violations_found"] = len(violations)
        self.stats["auto_removed"] = len(removals)
        
        is_valid = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]) == 0
        
        return LegalValidationResult(
            is_valid=is_valid,
            violations=violations,
            removals=removals,
            processed_text=processed_text,
            stats=self.stats.copy()
        )
    
    def _build_allowed_articles(self, whitelist: dict) -> Set[str]:
        """Buduje set dozwolonych artykułów (znormalizowanych)."""
        allowed = set()
        
        for article in whitelist.get("articles", []):
            # Dodaj pełną formę
            full = article.get("full", "")
            if full:
                normalized = self._normalize_article(full)
                allowed.add(normalized)
            
            # Dodaj warianty
            code = article.get("code", "")
            art_num = article.get("article", "")
            para = article.get("paragraph", "")
            
            if code and art_num:
                # art. 13 k.c.
                base = f"art. {art_num} {code}"
                allowed.add(self._normalize_article(base))
                
                if para:
                    # art. 13 § 1 k.c.
                    with_para = f"art. {art_num} § {para} {code}"
                    allowed.add(self._normalize_article(with_para))
        
        return allowed
    
    def _build_allowed_judgments(self, whitelist: dict) -> Set[str]:
        """Buduje set dozwolonych sygnatur (znormalizowanych)."""
        allowed = set()
        
        for judgment in whitelist.get("judgments", []):
            signature = judgment.get("signature", "")
            if signature:
                normalized = self._normalize_signature(signature)
                allowed.add(normalized)
        
        return allowed
    
    def _check_articles(
        self,
        text: str,
        allowed: Set[str],
        auto_remove: bool,
        strict_mode: bool
    ) -> Tuple[List[LegalViolation], List[LegalRemoval], str]:
        """Sprawdza artykuły w tekście."""
        violations = []
        removals = []
        processed_text = text
        
        for pattern, pattern_type in self.ARTICLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.stats["articles_checked"] += 1
                
                found_text = match.group(0).strip()
                normalized = self._normalize_article(found_text)
                
                # Sprawdź czy dozwolony
                if self._is_article_allowed(normalized, allowed):
                    continue
                
                # Pobierz kontekst
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Określ severity
                if strict_mode:
                    severity = ViolationSeverity.CRITICAL
                else:
                    # Jeśli wygląda jak prawdziwy przepis, może być halucynacją
                    severity = ViolationSeverity.WARNING
                
                violation = LegalViolation(
                    type=ViolationType.ILLEGAL_ARTICLE,
                    severity=severity,
                    found_text=found_text,
                    normalized=normalized,
                    position=match.start(),
                    context=context,
                    suggestion=f"Usuń lub zamień na: '{self.SAFE_REPLACEMENTS['article']}'"
                )
                violations.append(violation)
                
                if auto_remove:
                    replacement = self.SAFE_REPLACEMENTS['article']
                    processed_text = processed_text.replace(found_text, replacement, 1)
                    
                    removals.append(LegalRemoval(
                        original=found_text,
                        replacement=replacement,
                        reason="Przepis spoza whitelist"
                    ))
        
        return violations, removals, processed_text
    
    def _check_judgments(
        self,
        text: str,
        allowed: Set[str],
        auto_remove: bool,
        strict_mode: bool
    ) -> Tuple[List[LegalViolation], List[LegalRemoval], str]:
        """Sprawdza sygnatury orzeczeń w tekście."""
        violations = []
        removals = []
        processed_text = text
        
        for pattern, pattern_type in self.JUDGMENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.stats["judgments_checked"] += 1
                
                found_text = match.group(0).strip()
                signature = self._extract_signature(match, pattern_type)
                normalized = self._normalize_signature(signature)
                
                # Sprawdź czy dozwolony
                if normalized in allowed:
                    continue
                
                # Pobierz kontekst
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Orzeczenia są bardziej ryzykowne - zawsze CRITICAL
                severity = ViolationSeverity.CRITICAL
                
                violation = LegalViolation(
                    type=ViolationType.ILLEGAL_JUDGMENT,
                    severity=severity,
                    found_text=found_text,
                    normalized=normalized,
                    position=match.start(),
                    context=context,
                    suggestion="Usuń - orzeczenie spoza zweryfikowanej listy"
                )
                violations.append(violation)
                
                if auto_remove:
                    # Dla orzeczeń - usuń całe zdanie
                    processed_text = self._remove_sentence_containing(
                        processed_text, 
                        found_text
                    )
                    
                    removals.append(LegalRemoval(
                        original=found_text,
                        replacement="[usunięto nieweryfikowalne orzeczenie]",
                        reason="Orzeczenie spoza whitelist - potencjalna halucynacja"
                    ))
        
        return violations, removals, processed_text
    
    def _normalize_article(self, article: str) -> str:
        """Normalizuje zapis artykułu do porównywalnej formy."""
        article = article.lower().strip()
        
        # Usuń wielokrotne spacje
        article = re.sub(r"\s+", " ", article)
        
        # Normalizuj "artykuł" -> "art."
        article = re.sub(r"artykuł(?:u|em|owi)?", "art.", article)
        
        # Normalizuj kodeksy
        for variant, normalized in self.CODE_NORMALIZATION.items():
            article = article.replace(variant, normalized)
        
        # Normalizuj paragraf
        article = re.sub(r"§\s*", "§ ", article)
        article = re.sub(r"ust\.\s*", "ust. ", article)
        article = re.sub(r"pkt\s*", "pkt ", article)
        
        return article.strip()
    
    def _normalize_signature(self, signature: str) -> str:
        """Normalizuje sygnaturę orzeczenia."""
        signature = signature.upper().strip()
        signature = re.sub(r"\s+", " ", signature)
        return signature
    
    def _is_article_allowed(self, normalized: str, allowed: Set[str]) -> bool:
        """Sprawdza czy artykuł jest na whitelist."""
        # Dokładne dopasowanie
        if normalized in allowed:
            return True
        
        # Sprawdź bez paragrafu (art. 13 k.c. dopasowuje art. 13 § 1 k.c.)
        for allowed_art in allowed:
            # Jeśli sprawdzany jest bardziej ogólny (bez §), a dozwolony ma §
            if normalized in allowed_art:
                return True
            # Jeśli sprawdzany ma §, a dozwolony jest bardziej ogólny
            if allowed_art in normalized:
                return True
        
        return False
    
    def _extract_signature(self, match: re.Match, pattern_type: str) -> str:
        """Wyciąga sygnaturę z match."""
        groups = match.groups()
        
        if pattern_type in ["signature", "with_prefix"]:
            if len(groups) >= 3:
                return f"{groups[0]} {groups[1]} {groups[2]}"
        elif pattern_type in ["full_citation", "in_parentheses"]:
            if groups:
                return groups[0] if groups[0] else match.group(0)
        
        return match.group(0)
    
    def _remove_sentence_containing(self, text: str, fragment: str) -> str:
        """Usuwa całe zdanie zawierające fragment."""
        # Podziel na zdania
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtruj zdania bez fragmentu
        filtered = []
        for sentence in sentences:
            if fragment not in sentence:
                filtered.append(sentence)
            else:
                # Dodaj placeholder
                filtered.append("[...]")
        
        # Scal z powrotem, usuwając wielokrotne [...]
        result = " ".join(filtered)
        result = re.sub(r'\[\.\.\.\]\s*\[\.\.\.\]', '[...]', result)
        
        return result


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def create_legal_whitelist(detected_articles: List[str], legal_judgments: List[dict]) -> dict:
    """
    Tworzy whitelist na podstawie wykrytych przepisów i orzeczeń.
    
    Args:
        detected_articles: Lista wykrytych artykułów, np. ["art. 13 k.c.", "art. 16 k.c."]
        legal_judgments: Lista orzeczeń z SAOS/Google
        
    Returns:
        Whitelist do użycia w LegalHardLockValidator
    """
    whitelist = {
        "articles": [],
        "judgments": [],
        "source": "detected_articles + legal_judgments"
    }
    
    # Parsuj artykuły
    for art_str in detected_articles:
        parsed = _parse_article_string(art_str)
        if parsed:
            whitelist["articles"].append(parsed)
    
    # Parsuj orzeczenia
    for judgment in legal_judgments:
        if isinstance(judgment, dict):
            whitelist["judgments"].append({
                "court": judgment.get("court", ""),
                "signature": judgment.get("signature", ""),
                "url": judgment.get("url", "")
            })
        elif isinstance(judgment, str):
            whitelist["judgments"].append({
                "signature": judgment,
                "court": "",
                "url": ""
            })
    
    return whitelist


def _parse_article_string(art_str: str) -> Optional[dict]:
    """Parsuje string artykułu do struktury."""
    
    # art. 13 § 1 k.c.
    match = re.match(
        r"art\.?\s*(\d+[a-z]?)\s*(§\s*(\d+))?\s*(k\.?[a-z\.]+)",
        art_str.lower()
    )
    
    if match:
        return {
            "article": match.group(1),
            "paragraph": match.group(3) if match.group(3) else "",
            "code": match.group(4).replace(".", "").replace(" ", "") + ".",
            "full": art_str
        }
    
    # Fallback - zapisz jako jest
    return {
        "full": art_str,
        "article": "",
        "paragraph": "",
        "code": ""
    }


def add_common_legal_articles(whitelist: dict, legal_category: str) -> dict:
    """
    Dodaje często używane artykuły dla danej kategorii prawnej.
    
    Rozszerza whitelist o artykuły, które są powszechnie cytowane
    i prawdopodobnie poprawne.
    """
    
    COMMON_ARTICLES = {
        "prawo cywilne": [
            {"article": "23", "code": "k.c.", "full": "art. 23 k.c."},  # Dobra osobiste
            {"article": "24", "code": "k.c.", "full": "art. 24 k.c."},  # Ochrona dóbr osobistych
            {"article": "415", "code": "k.c.", "full": "art. 415 k.c."},  # Odpowiedzialność deliktowa
            {"article": "471", "code": "k.c.", "full": "art. 471 k.c."},  # Odpowiedzialność kontraktowa
        ],
        "prawo rodzinne": [
            {"article": "23", "code": "k.r.o.", "full": "art. 23 k.r.o."},  # Równe prawa małżonków
            {"article": "56", "code": "k.r.o.", "full": "art. 56 k.r.o."},  # Rozwód
            {"article": "87", "code": "k.r.o.", "full": "art. 87 k.r.o."},  # Władza rodzicielska
        ],
        "ubezwłasnowolnienie": [
            {"article": "13", "paragraph": "1", "code": "k.c.", "full": "art. 13 § 1 k.c."},
            {"article": "13", "paragraph": "2", "code": "k.c.", "full": "art. 13 § 2 k.c."},
            {"article": "16", "paragraph": "1", "code": "k.c.", "full": "art. 16 § 1 k.c."},
            {"article": "16", "paragraph": "2", "code": "k.c.", "full": "art. 16 § 2 k.c."},
            {"article": "544", "code": "k.p.c.", "full": "art. 544 k.p.c."},
            {"article": "545", "code": "k.p.c.", "full": "art. 545 k.p.c."},
            {"article": "546", "code": "k.p.c.", "full": "art. 546 k.p.c."},
            {"article": "547", "code": "k.p.c.", "full": "art. 547 k.p.c."},
            {"article": "548", "code": "k.p.c.", "full": "art. 548 k.p.c."},
            {"article": "549", "code": "k.p.c.", "full": "art. 549 k.p.c."},
            {"article": "550", "code": "k.p.c.", "full": "art. 550 k.p.c."},
            {"article": "551", "code": "k.p.c.", "full": "art. 551 k.p.c."},
            {"article": "552", "code": "k.p.c.", "full": "art. 552 k.p.c."},
            {"article": "553", "code": "k.p.c.", "full": "art. 553 k.p.c."},
            {"article": "554", "code": "k.p.c.", "full": "art. 554 k.p.c."},
            {"article": "555", "code": "k.p.c.", "full": "art. 555 k.p.c."},
            {"article": "556", "code": "k.p.c.", "full": "art. 556 k.p.c."},
            {"article": "557", "code": "k.p.c.", "full": "art. 557 k.p.c."},
            {"article": "558", "code": "k.p.c.", "full": "art. 558 k.p.c."},
            {"article": "559", "code": "k.p.c.", "full": "art. 559 k.p.c."},
            {"article": "560", "code": "k.p.c.", "full": "art. 560 k.p.c."},
        ],
    }
    
    # Dodaj artykuły dla kategorii
    common = COMMON_ARTICLES.get(legal_category, [])
    
    existing_fulls = {a.get("full", "").lower() for a in whitelist.get("articles", [])}
    
    for article in common:
        if article.get("full", "").lower() not in existing_fulls:
            whitelist["articles"].append(article)
    
    return whitelist
