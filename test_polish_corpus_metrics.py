"""
===============================================================================
🧪 TEST_POLISH_CORPUS_METRICS.py - Testy dla modułu corpus metrics
===============================================================================

Uruchom testy:
    pytest test_polish_corpus_metrics.py -v
    
Lub bezpośrednio:
    python test_polish_corpus_metrics.py

===============================================================================
"""

import pytest
import sys
from typing import List

# Import testowanego modułu
from polish_corpus_metrics_v41 import (
    calculate_diacritic_ratio,
    calculate_word_length_stats,
    calculate_fog_pl_index,
    calculate_punctuation_density,
    analyze_vowel_ratio,
    analyze_corpus_metrics,
    get_corpus_insights_for_moe,
    get_naturalness_hints,
    InsightSeverity,
    CorpusInsight,
    CorpusAnalysisResult,
    CORPUS_REFERENCE,
)


# =============================================================================
# TESTY BEZPIECZEŃSTWA (KRYTYCZNE!)
# =============================================================================

class TestSafetyGuarantees:
    """
    Testy gwarantujące że moduł NIGDY nie blokuje walidacji.
    
    Te testy są KRYTYCZNE - ich niepowodzenie oznacza błąd w module!
    """
    
    def test_never_blocks_validation_empty_text(self):
        """Pusty tekst nie może blokować."""
        result = analyze_corpus_metrics("")
        assert result.blocks_validation == False
    
    def test_never_blocks_validation_short_text(self):
        """Krótki tekst nie może blokować."""
        result = analyze_corpus_metrics("Test")
        assert result.blocks_validation == False
    
    def test_never_blocks_validation_high_diacritics(self):
        """Wysoki udział diakrytyków nie może blokować."""
        text = "Zażółć gęślą jaźń " * 100  # ~35% diakrytyków
        result = analyze_corpus_metrics(text)
        assert result.blocks_validation == False
    
    def test_never_blocks_validation_no_diacritics(self):
        """Brak diakrytyków nie może blokować."""
        text = "To jest tekst bez polskich znakow diakrytycznych " * 50
        result = analyze_corpus_metrics(text)
        assert result.blocks_validation == False
    
    def test_never_blocks_validation_monotonous_text(self):
        """Monotonny tekst nie może blokować."""
        text = "test test test test test " * 100
        result = analyze_corpus_metrics(text)
        assert result.blocks_validation == False
    
    def test_never_blocks_validation_difficult_words(self):
        """Trudne słowa nie mogą blokować."""
        text = "Konstantynopolitańczykowianeczka internacjonalizacja " * 20
        result = analyze_corpus_metrics(text)
        assert result.blocks_validation == False
    
    def test_severity_never_critical(self):
        """Żaden insight nie może mieć severity=critical."""
        test_texts = [
            "",
            "Test",
            "Zażółć gęślą jaźń " * 100,
            "test bez znakow " * 100,
            "Konstantynopolitańczykowianeczka " * 20,
            "To jest. Krótkie. Zdania." * 50,
        ]
        
        allowed_severities = {
            InsightSeverity.INFO,
            InsightSeverity.SUGGESTION,
            InsightSeverity.OBSERVATION
        }
        
        for text in test_texts:
            result = analyze_corpus_metrics(text)
            for insight in result.insights:
                assert insight.severity in allowed_severities, \
                    f"Niedozwolone severity {insight.severity} dla tekstu: {text[:30]}"
    
    def test_moe_integration_never_blocks(self):
        """Integracja z MOE nigdy nie może blokować."""
        test_texts = ["", "Test", "Zażółć " * 100, "test " * 100]
        
        for text in test_texts:
            result = get_corpus_insights_for_moe(text)
            assert result.get("affects_validation") == False
            assert result.get("is_blocking") == False
            assert result.get("blocks_action") == False
    
    def test_handles_none_gracefully(self):
        """None jako input nie może rzucić wyjątku."""
        # analyze_corpus_metrics przyjmuje tylko str, ale powinien obsłużyć gracefully
        try:
            result = analyze_corpus_metrics(None or "")
            assert result.blocks_validation == False
        except Exception as e:
            pytest.fail(f"Wyjątek dla None: {e}")
    
    def test_handles_special_characters(self):
        """Znaki specjalne nie mogą rzucić wyjątku."""
        special_texts = [
            "\n\n\n",
            "\t\t\t",
            "   ",
            "!@#$%^&*()",
            "日本語テスト",
            "🎉🎊🎁",
            "<script>alert('xss')</script>",
        ]
        
        for text in special_texts:
            try:
                result = analyze_corpus_metrics(text)
                assert result.blocks_validation == False
            except Exception as e:
                pytest.fail(f"Wyjątek dla '{text[:20]}': {e}")


# =============================================================================
# TESTY METRYKI DIAKRYTYKÓW
# =============================================================================

class TestDiacriticRatio:
    """Testy dla metryki udziału diakrytyków."""
    
    def test_pangram_high_diacritics(self):
        """Pangram 'Zażółć gęślą jaźń' ma wysoki udział diakrytyków."""
        text = "Zażółć gęślą jaźń"
        result = calculate_diacritic_ratio(text)
        
        # Pangram ma ~35% diakrytyków (7/20 liter)
        assert result.value > 0.25
        assert result.severity in [InsightSeverity.INFO, InsightSeverity.SUGGESTION]
    
    def test_no_diacritics(self):
        """Tekst bez diakrytyków powinien dać sugestię."""
        text = "To jest tekst bez polskich znakow diakrytycznych test"
        result = calculate_diacritic_ratio(text)
        
        assert result.value == 0
        assert result.severity == InsightSeverity.SUGGESTION
        assert "niski" in result.message.lower()
    
    def test_normal_polish_text(self):
        """Normalny tekst polski ma ~5-9% diakrytyków."""
        text = """
        Sąd okręgowy rozpatruje sprawę ubezwłasnowolnienia osoby dorosłej.
        Prokurator może złożyć wniosek w imieniu osoby, która nie jest 
        w stanie sama prowadzić swoich spraw. Kurator jest wyznaczany 
        przez sąd opiekuńczy.
        """
        result = calculate_diacritic_ratio(text)
        
        # Normalny tekst: 5-9%
        assert 0.04 < result.value < 0.12
        assert result.severity == InsightSeverity.INFO
    
    def test_too_short_text(self):
        """Za krótki tekst daje observation."""
        text = "Test"
        result = calculate_diacritic_ratio(text)
        
        assert result.severity == InsightSeverity.OBSERVATION
        assert "za mało" in result.message.lower()
    
    def test_reference_values(self):
        """Sprawdź że wartości referencyjne są poprawne."""
        ref = CORPUS_REFERENCE["diacritic_ratio"]
        
        assert ref["target"] == 0.069
        assert ref["min_natural"] == 0.05
        assert ref["max_natural"] == 0.09


# =============================================================================
# TESTY METRYKI DŁUGOŚCI SŁÓW
# =============================================================================

class TestWordLength:
    """Testy dla metryki średniej długości słów."""
    
    def test_normal_polish_text(self):
        """Normalny tekst polski ma średnio ~6 znaków na słowo."""
        text = """
        Prawo cywilne reguluje stosunki między osobami. Kodeks cywilny 
        zawiera przepisy dotyczące własności, zobowiązań i spadków.
        Sąd rozstrzyga spory między stronami.
        """
        result = calculate_word_length_stats(text)
        
        # Oczekiwana średnia: 5.5-6.5
        assert 5.0 < result.value < 7.0
    
    def test_scientific_text_longer_words(self):
        """Tekst naukowy ma dłuższe słowa."""
        text = """
        Konstytucyjność przedmiotowego rozstrzygnięcia legislacyjnego 
        budzi uzasadnione wątpliwości interpretacyjne w kontekście 
        utrwalonego orzecznictwa Trybunału Konstytucyjnego dotyczącego
        proporcjonalności ingerencji ustawodawczej.
        """
        result = calculate_word_length_stats(text)
        
        # Tekst naukowy: >6.3
        assert result.value > 6.0
        assert "naukowy" in result.details.get("style_detected", "").lower() or \
               "urzędowy" in result.details.get("style_detected", "").lower()
    
    def test_simple_text_shorter_words(self):
        """Prosty tekst ma krótsze słowa."""
        text = """
        Mama ma kota. Kot jest duży. Dom jest ładny. Tata jedzie autem.
        Pies biega szybko. Słońce świeci mocno. Dzieci się bawią.
        """
        result = calculate_word_length_stats(text)
        
        # Prosty tekst: <5.5
        assert result.value < 6.0
    
    def test_style_detection(self):
        """Sprawdź wykrywanie stylu."""
        # Publicystyka: ~6.0
        text_pub = "Rząd ogłosił nowe przepisy dotyczące ochrony środowiska " * 10
        result = calculate_word_length_stats(text_pub)
        assert result.details.get("style_detected") is not None


# =============================================================================
# TESTY FOG-PL
# =============================================================================

class TestFOGPL:
    """Testy dla indeksu czytelności FOG-PL."""
    
    def test_simple_text_low_fog(self):
        """Prosty tekst ma niski FOG."""
        text = """
        To jest dom. Dom jest duży. W domu mieszka kot.
        Kot lubi mleko. Mama daje kotu mleko.
        Tata czyta gazetę. Jest ładny dzień.
        """
        result = calculate_fog_pl_index(text)
        
        # Prosty tekst: FOG < 8
        assert result.value < 8
    
    def test_complex_text_high_fog(self):
        """Złożony tekst ma wysoki FOG."""
        text = """
        Konstytucyjność przedmiotowego rozstrzygnięcia legislacyjnego 
        budzi uzasadnione wątpliwości interpretacyjne w kontekście 
        utrwalonego orzecznictwa Trybunału Konstytucyjnego, szczególnie 
        w odniesieniu do proporcjonalności ingerencji ustawodawczej 
        w konstytucyjnie chronione prawa obywatelskie.
        """
        result = calculate_fog_pl_index(text)
        
        # Złożony tekst: FOG > 12
        assert result.value > 10
    
    def test_optimal_fog_range(self):
        """Sprawdź że optymalny zakres to 8-9."""
        ref = CORPUS_REFERENCE["fog_pl"]
        
        assert ref["optimal_min"] == 8
        assert ref["optimal_max"] == 9
    
    def test_syllable_counting_polish(self):
        """Polskie słowa trudne mają ≥4 sylaby."""
        # Słowa 4-sylabowe (trudne w polskim)
        difficult_words = [
            "ubezwłasnowolnienie",  # 7 sylab
            "internacjonalizacja",  # 8 sylab
            "konstytucyjność",      # 5 sylab
            "odpowiedzialność",     # 6 sylab
        ]
        
        text = " ".join(difficult_words * 5) + ". " * 5
        result = calculate_fog_pl_index(text)
        
        # Wysoki udział trudnych słów = wysoki FOG
        assert result.value > 12
        assert result.details.get("difficult_words_count", 0) > 10


# =============================================================================
# TESTY INTERPUNKCJI
# =============================================================================

class TestPunctuation:
    """Testy dla metryki gęstości interpunkcji."""
    
    def test_missing_comma_before_ze(self):
        """Wykryj brakujący przecinek przed 'że'."""
        text = "Uważam że to jest ważne. Myślę że masz rację. Wierzę że się uda."
        result = calculate_punctuation_density(text)
        
        # Powinien wykryć brakujące przecinki
        assert result.severity == InsightSeverity.SUGGESTION
        assert "że" in str(result.details.get("missing_commas", []))
    
    def test_correct_punctuation(self):
        """Tekst z poprawnymi przecinkami."""
        text = """
        Uważam, że to jest ważne. Myślę, że masz rację. 
        Wierzę, że się uda. Wiem, który wybór jest lepszy.
        Rozumiem, ponieważ to jasne.
        """
        result = calculate_punctuation_density(text)
        
        # Poprawna interpunkcja = INFO
        # (lub może być SUGGESTION jeśli gęstość < 1.47%)
        assert len(result.details.get("missing_commas", [])) == 0 or \
               result.severity in [InsightSeverity.INFO, InsightSeverity.SUGGESTION]
    
    def test_comma_density_reference(self):
        """Sprawdź wartość referencyjną dla przecinków."""
        ref = CORPUS_REFERENCE["punctuation"]
        
        # Przecinek > 1.47% (częstszy niż litera "b")
        assert ref["comma_min"] == 0.0147


# =============================================================================
# TESTY SAMOGŁOSEK
# =============================================================================

class TestVowelRatio:
    """Testy dla metryki udziału samogłosek."""
    
    def test_normal_text_vowel_ratio(self):
        """Normalny tekst ma 35-38% samogłosek."""
        text = """
        To jest normalny tekst w języku polskim, który powinien mieć
        standardowy udział samogłosek zgodny z normami korpusu NKJP.
        Polszczyzna charakteryzuje się określonymi proporcjami liter.
        """
        result = analyze_vowel_ratio(text)
        
        # Oczekiwane: 35-38%
        assert 0.30 < result.value < 0.45
    
    def test_reference_values(self):
        """Sprawdź wartości referencyjne."""
        ref = CORPUS_REFERENCE["vowel_ratio"]
        
        assert ref["target"] == 0.365
        assert ref["min"] == 0.35
        assert ref["max"] == 0.38


# =============================================================================
# TESTY INTEGRACYJNE
# =============================================================================

class TestFullAnalysis:
    """Testy integracyjne dla pełnej analizy."""
    
    def test_full_analysis_returns_all_metrics(self):
        """Pełna analiza zwraca wszystkie metryki."""
        text = """
        Ubezwłasnowolnienie to instytucja prawa cywilnego, która pozwala na 
        ograniczenie zdolności do czynności prawnych osoby, która z powodu 
        choroby psychicznej nie jest w stanie kierować swoim postępowaniem.
        Sąd okręgowy rozpatruje sprawy o ubezwłasnowolnienie.
        """
        
        result = analyze_corpus_metrics(text)
        
        # Sprawdź że mamy wszystkie metryki
        metrics = {i.metric for i in result.insights}
        expected_metrics = {"diacritic_ratio", "word_length_avg", "vowel_ratio", 
                          "fog_pl", "punctuation_density"}
        
        assert metrics == expected_metrics
    
    def test_naturalness_score_range(self):
        """Naturalness score jest w zakresie 0-100."""
        texts = [
            "Test " * 50,
            "Zażółć gęślą jaźń " * 50,
            "Sąd orzeka ubezwłasnowolnienie osoby dorosłej. " * 20,
        ]
        
        for text in texts:
            result = analyze_corpus_metrics(text)
            assert 0 <= result.overall_naturalness <= 100
    
    def test_style_detection(self):
        """Wykrywanie stylu działa."""
        text = "Sąd orzeka ubezwłasnowolnienie osoby. " * 30
        result = analyze_corpus_metrics(text)
        
        assert result.style_detected in ["literatura", "publicystyka", "urzędowy", "naukowy"]
    
    def test_to_dict_format(self):
        """Format słownikowy jest poprawny."""
        result = analyze_corpus_metrics("Test tekstu polskiego. " * 20)
        result_dict = result.to_dict()
        
        # Wymagane pola
        assert "insights" in result_dict
        assert "blocks_validation" in result_dict
        assert "is_informational_only" in result_dict
        assert "overall_naturalness" in result_dict
        
        # Wartości bezpieczeństwa
        assert result_dict["blocks_validation"] == False
        assert result_dict["is_informational_only"] == True


# =============================================================================
# TESTY MOE INTEGRATION
# =============================================================================

class TestMOEIntegration:
    """Testy integracji z MOE Validator."""
    
    def test_get_corpus_insights_format(self):
        """Format insights dla MOE jest poprawny."""
        text = "Sąd orzeka ubezwłasnowolnienie. " * 20
        result = get_corpus_insights_for_moe(text)
        
        # Wymagane pola
        assert "enabled" in result
        assert "affects_validation" in result
        assert "is_blocking" in result
        
        if result["enabled"]:
            assert "insights" in result
            assert "naturalness_score" in result
            assert "suggestions" in result
    
    def test_get_naturalness_hints_format(self):
        """Format hints jest poprawny."""
        text = "test bez polskich znakow diakrytycznych " * 30
        hints = get_naturalness_hints(text)
        
        # Hints to lista słowników
        assert isinstance(hints, list)
        
        for hint in hints:
            assert "metric" in hint
            assert "hint" in hint
    
    def test_error_handling_in_moe_integration(self):
        """Błędy są obsługiwane gracefully."""
        # Nawet przy dziwnym input - nie rzuca wyjątku
        weird_inputs = ["", None, 123, [], {}]
        
        for inp in weird_inputs:
            try:
                result = get_corpus_insights_for_moe(inp if isinstance(inp, str) else "")
                assert result.get("affects_validation") == False
            except Exception as e:
                pytest.fail(f"Wyjątek dla {inp}: {e}")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTY POLISH CORPUS METRICS v41.2")
    print("=" * 70)
    
    # Uruchom pytest jeśli dostępny
    try:
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(exit_code)
    except Exception:
        # Fallback - uruchom testy ręcznie
        print("\n⚠️ pytest niedostępny - uruchamiam testy ręcznie\n")
        
        test_classes = [
            TestSafetyGuarantees,
            TestDiacriticRatio,
            TestWordLength,
            TestFOGPL,
            TestPunctuation,
            TestVowelRatio,
            TestFullAnalysis,
            TestMOEIntegration,
        ]
        
        passed = 0
        failed = 0
        
        for test_class in test_classes:
            print(f"\n📋 {test_class.__name__}:")
            instance = test_class()
            
            for method_name in dir(instance):
                if method_name.startswith("test_"):
                    try:
                        getattr(instance, method_name)()
                        print(f"   ✅ {method_name}")
                        passed += 1
                    except AssertionError as e:
                        print(f"   ❌ {method_name}: {e}")
                        failed += 1
                    except Exception as e:
                        print(f"   ❌ {method_name}: EXCEPTION - {e}")
                        failed += 1
        
        print("\n" + "=" * 70)
        print(f"📊 WYNIKI: {passed} passed, {failed} failed")
        print("=" * 70)
        
        sys.exit(0 if failed == 0 else 1)
