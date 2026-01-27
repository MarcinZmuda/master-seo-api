"""
===============================================================================
🛡️ KEYWORD CONFLICT VALIDATOR v38.3
===============================================================================
Zapobiega tworzeniu projektów z konfliktami fraz.

PROBLEM KTÓRY ROZWIĄZUJE:
- Fraza BASIC (np. "ubezwłasnowolnienie") jest limitowana (6-24x)
- Ta sama fraza jest w H2 (nagłówek) lub MAIN keyword
- H2 są WYMAGANE strukturalnie → każdy H2 = +1 użycie
- MAIN jest WYMAGANE → kolejne użycie
- → NIESKOŃCZONA PĘTLA REWRITE (niemożliwe do spełnienia warunki)

ROZWIĄZANIE:
- Walidacja PRZED createProject
- Blokada jeśli BASIC ⊂ MAIN lub BASIC ∈ H2
- Auto-degradacja do EXTENDED lub usunięcie z BASIC

INTEGRACJA:
- Wywołaj validate_keywords_before_create() PRZED API call
- Jeśli zwróci błędy → NIE twórz projektu
===============================================================================
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KeywordConflict:
    """Reprezentuje konflikt fraz."""
    keyword: str
    conflict_type: str  # "MAIN_OVERLAP", "H2_OVERLAP", "H2_CONTAINS"
    conflicting_with: str  # np. "Ubezwłasnowolnienie osoby chorej psychicznie"
    severity: str  # "CRITICAL", "WARNING"
    recommendation: str


class KeywordConflictValidator:
    """
    Waliduje czy frazy BASIC nie kolidują z MAIN/H2.
    
    ZASADA ZŁOTA:
    Fraza strukturalna (MAIN/H2) ≠ fraza limitowana (BASIC)
    """
    
    def __init__(self):
        self.conflicts: List[KeywordConflict] = []
    
    def validate(
        self,
        main_keyword: str,
        h2_structure: List[str],
        keywords_list: List[Dict],
        auto_fix: bool = False
    ) -> Tuple[bool, List[KeywordConflict], List[Dict]]:
        """
        Waliduje frazy przed utworzeniem projektu.
        
        Args:
            main_keyword: Główna fraza (MAIN)
            h2_structure: Lista nagłówków H2
            keywords_list: Lista fraz z type: BASIC/EXTENDED
            auto_fix: Czy automatycznie naprawić konflikty
            
        Returns:
            (is_valid, conflicts, fixed_keywords_list)
        """
        self.conflicts = []
        main_lower = main_keyword.lower().strip()
        h2_lower = [h.lower().strip() for h in h2_structure]
        
        # Wyciągnij tokeny z MAIN i H2
        main_tokens = set(self._tokenize(main_lower))
        h2_tokens = set()
        for h2 in h2_lower:
            h2_tokens.update(self._tokenize(h2))
        
        fixed_keywords = []
        
        for kw in keywords_list:
            term = kw.get("term", "").lower().strip()
            kw_type = kw.get("type", "BASIC")
            
            if kw_type != "BASIC":
                fixed_keywords.append(kw)
                continue
            
            conflict = self._check_conflict(term, main_lower, h2_lower, main_tokens, h2_tokens)
            
            if conflict:
                self.conflicts.append(conflict)
                
                if auto_fix:
                    # Degraduj do EXTENDED
                    fixed_kw = kw.copy()
                    fixed_kw["type"] = "EXTENDED"
                    fixed_kw["_auto_degraded"] = True
                    fixed_kw["_conflict_reason"] = conflict.conflict_type
                    fixed_keywords.append(fixed_kw)
                    print(f"[CONFLICT VALIDATOR] ⚠️ Auto-degraded '{term}' BASIC → EXTENDED ({conflict.conflict_type})")
                else:
                    fixed_keywords.append(kw)
            else:
                fixed_keywords.append(kw)
        
        is_valid = len([c for c in self.conflicts if c.severity == "CRITICAL"]) == 0
        
        return is_valid, self.conflicts, fixed_keywords
    
    def _tokenize(self, text: str) -> List[str]:
        """Wyciąga tokeny (słowa) z tekstu."""
        # Usuń interpunkcję i podziel na słowa
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w.strip() for w in text.split() if len(w.strip()) > 2]
    
    def _check_conflict(
        self,
        basic_term: str,
        main_lower: str,
        h2_lower: List[str],
        main_tokens: set,
        h2_tokens: set
    ) -> Optional[KeywordConflict]:
        """Sprawdza czy fraza BASIC koliduje z MAIN/H2."""
        
        basic_tokens = set(self._tokenize(basic_term))
        
        # 1. CRITICAL: BASIC == MAIN (identyczne)
        if basic_term == main_lower:
            return KeywordConflict(
                keyword=basic_term,
                conflict_type="MAIN_IDENTICAL",
                conflicting_with=main_lower,
                severity="CRITICAL",
                recommendation=f"Usuń '{basic_term}' z BASIC - jest identyczna z MAIN keyword"
            )
        
        # 2. CRITICAL: BASIC ⊂ MAIN (BASIC jest częścią MAIN)
        if basic_term in main_lower:
            return KeywordConflict(
                keyword=basic_term,
                conflict_type="MAIN_CONTAINS",
                conflicting_with=main_lower,
                severity="CRITICAL",
                recommendation=f"Usuń '{basic_term}' z BASIC - jest częścią MAIN keyword '{main_lower}'"
            )
        
        # 3. CRITICAL: BASIC == H2 (identyczne z nagłówkiem)
        for h2 in h2_lower:
            if basic_term == h2:
                return KeywordConflict(
                    keyword=basic_term,
                    conflict_type="H2_IDENTICAL",
                    conflicting_with=h2,
                    severity="CRITICAL",
                    recommendation=f"Usuń '{basic_term}' z BASIC - jest identyczna z H2 '{h2}'"
                )
        
        # 4. CRITICAL: BASIC ⊂ H2 (BASIC jest częścią nagłówka)
        for h2 in h2_lower:
            if basic_term in h2:
                return KeywordConflict(
                    keyword=basic_term,
                    conflict_type="H2_CONTAINS",
                    conflicting_with=h2,
                    severity="CRITICAL",
                    recommendation=f"Usuń '{basic_term}' z BASIC - jest częścią H2 '{h2}'"
                )
        
        # 5. WARNING: Pojedynczy token BASIC jest w MAIN/H2
        if len(basic_tokens) == 1:
            single_token = list(basic_tokens)[0]
            
            if single_token in main_tokens:
                return KeywordConflict(
                    keyword=basic_term,
                    conflict_type="SINGLE_TOKEN_IN_MAIN",
                    conflicting_with=main_lower,
                    severity="WARNING",
                    recommendation=f"Rozważ usunięcie '{basic_term}' z BASIC - token występuje w MAIN"
                )
            
            if single_token in h2_tokens:
                return KeywordConflict(
                    keyword=basic_term,
                    conflict_type="SINGLE_TOKEN_IN_H2",
                    conflicting_with="[multiple H2]",
                    severity="WARNING",
                    recommendation=f"Rozważ usunięcie '{basic_term}' z BASIC - token występuje w H2"
                )
        
        # 6. WARNING: Główny token BASIC (pierwszy/najdłuższy) w strukturze
        if basic_tokens:
            main_token = max(basic_tokens, key=len)  # Najdłuższy token
            
            if main_token in main_tokens and len(main_token) > 4:
                return KeywordConflict(
                    keyword=basic_term,
                    conflict_type="MAIN_TOKEN_OVERLAP",
                    conflicting_with=main_lower,
                    severity="WARNING",
                    recommendation=f"Rozważ degradację '{basic_term}' do EXTENDED - główny token '{main_token}' w MAIN"
                )
        
        return None


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================

_validator = KeywordConflictValidator()


def validate_keywords_before_create(
    main_keyword: str,
    h2_structure: List[str],
    keywords_list: List[Dict],
    auto_fix: bool = True
) -> Dict:
    """
    Waliduje frazy przed utworzeniem projektu.
    
    WYWOŁAJ TO PRZED /api/project/create!
    
    Args:
        main_keyword: Główna fraza
        h2_structure: Lista H2
        keywords_list: Lista fraz [{"term": "x", "min": 1, "max": 5, "type": "BASIC"}, ...]
        auto_fix: Czy auto-naprawić konflikty (domyślnie True)
        
    Returns:
        {
            "is_valid": bool,
            "can_create": bool,
            "conflicts": [...],
            "critical_count": int,
            "warning_count": int,
            "fixed_keywords": [...],  # Poprawiona lista (jeśli auto_fix)
            "message": str
        }
    """
    is_valid, conflicts, fixed_keywords = _validator.validate(
        main_keyword=main_keyword,
        h2_structure=h2_structure,
        keywords_list=keywords_list,
        auto_fix=auto_fix
    )
    
    critical_count = len([c for c in conflicts if c.severity == "CRITICAL"])
    warning_count = len([c for c in conflicts if c.severity == "WARNING"])
    
    # Możemy tworzyć projekt jeśli:
    # - Brak CRITICAL (lub auto_fix je naprawił)
    can_create = critical_count == 0 or auto_fix
    
    if critical_count > 0 and not auto_fix:
        message = f"❌ BLOKADA: {critical_count} konfliktów CRITICAL. Nie można utworzyć projektu."
    elif critical_count > 0 and auto_fix:
        message = f"⚠️ Naprawiono {critical_count} konfliktów CRITICAL (auto-degradacja do EXTENDED)"
    elif warning_count > 0:
        message = f"⚠️ {warning_count} potencjalnych konfliktów (WARNING) - projekt można utworzyć"
    else:
        message = "✅ Brak konfliktów fraz - projekt można utworzyć"
    
    return {
        "is_valid": is_valid,
        "can_create": can_create,
        "conflicts": [
            {
                "keyword": c.keyword,
                "type": c.conflict_type,
                "conflicting_with": c.conflicting_with,
                "severity": c.severity,
                "recommendation": c.recommendation
            }
            for c in conflicts
        ],
        "critical_count": critical_count,
        "warning_count": warning_count,
        "fixed_keywords": fixed_keywords if auto_fix else keywords_list,
        "message": message
    }


def get_conflict_report(
    main_keyword: str,
    h2_structure: List[str],
    keywords_list: List[Dict]
) -> str:
    """
    Generuje raport konfliktu (dla GPT do wyświetlenia).
    """
    result = validate_keywords_before_create(
        main_keyword=main_keyword,
        h2_structure=h2_structure,
        keywords_list=keywords_list,
        auto_fix=False
    )
    
    lines = [
        "## 🛡️ WALIDACJA KONFLIKTÓW FRAZ",
        "",
        f"**Status:** {result['message']}",
        "",
    ]
    
    if result["conflicts"]:
        lines.append("### Wykryte konflikty:")
        lines.append("")
        
        for c in result["conflicts"]:
            icon = "🔴" if c["severity"] == "CRITICAL" else "🟡"
            lines.append(f"{icon} **{c['keyword']}** ({c['type']})")
            lines.append(f"   Koliduje z: `{c['conflicting_with']}`")
            lines.append(f"   → {c['recommendation']}")
            lines.append("")
    
    if not result["can_create"]:
        lines.append("### ❌ AKCJA WYMAGANA")
        lines.append("")
        lines.append("Przed utworzeniem projektu musisz:")
        lines.append("1. Usunąć frazy CRITICAL z listy BASIC")
        lines.append("2. LUB przenieść je do EXTENDED")
        lines.append("3. LUB użyć auto_fix=True")
    
    return "\n".join(lines)


# ================================================================
# PRZYKŁAD UŻYCIA
# ================================================================

if __name__ == "__main__":
    # Test z przypadkiem który powodował nieskończoną pętlę
    main = "Ubezwłasnowolnienie osoby chorej psychicznie"
    h2 = [
        "Ubezwłasnowolnienie",
        "Czym jest ubezwłasnowolnienie",
        "Osoba chora i osoba chora psychicznie"
    ]
    keywords = [
        {"term": "ubezwłasnowolnienie", "min": 6, "max": 24, "type": "BASIC"},  # CONFLICT!
        {"term": "osoba chora psychicznie", "min": 1, "max": 2, "type": "BASIC"},  # CONFLICT!
        {"term": "sąd", "min": 5, "max": 12, "type": "BASIC"},  # OK
        {"term": "wniosek o ubezwłasnowolnienie", "min": 1, "max": 4, "type": "BASIC"},  # OK
    ]
    
    print("=" * 60)
    print("TEST: Walidacja przed createProject")
    print("=" * 60)
    print()
    
    # Bez auto-fix
    result = validate_keywords_before_create(main, h2, keywords, auto_fix=False)
    print("BEZ AUTO-FIX:")
    print(f"  can_create: {result['can_create']}")
    print(f"  critical: {result['critical_count']}, warning: {result['warning_count']}")
    print(f"  message: {result['message']}")
    print()
    
    # Z auto-fix
    result = validate_keywords_before_create(main, h2, keywords, auto_fix=True)
    print("Z AUTO-FIX:")
    print(f"  can_create: {result['can_create']}")
    print(f"  critical: {result['critical_count']}, warning: {result['warning_count']}")
    print(f"  message: {result['message']}")
    print()
    
    print("NAPRAWIONE FRAZY:")
    for kw in result["fixed_keywords"]:
        degraded = " [AUTO-DEGRADED]" if kw.get("_auto_degraded") else ""
        print(f"  {kw['term']}: {kw['type']}{degraded}")
    print()
    
    print("=" * 60)
    print("RAPORT DLA GPT:")
    print("=" * 60)
    print(get_conflict_report(main, h2, keywords))
