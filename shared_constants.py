"""
Shared sentence length constants for Brajn2026.
Fix #9 v4.2: Ujednolicenie targetow dlugosci zdan.
Fix #53 v4.3: Nowe limity — naturalna polszczyzna publicystyczna.
Fix #45.3 v4.4: Ujednolicenie z prompt_builder.py
v5.0: Prompt v2 — rozluznione limity. Prompt NIE wymusza dlugosci zdan.
      Validator uzywa szerszych tolerancji (NKJP: naturalna wariancja 3-30+).

Uzywany przez: prompt_builder.py, ai_middleware.py
"""

# Srednia dlugosc zdania (target)
SENTENCE_AVG_TARGET = 13       # NKJP publicystyczny (bylo: 15)
SENTENCE_AVG_TARGET_MIN = 8    # dolna granica (bylo: 12)
SENTENCE_AVG_TARGET_MAX = 20   # gorna granica (bylo: 18)

# Maksymalna dlugosc pojedynczego zdania
SENTENCE_SOFT_MAX = 30         # warning (bylo: 22)
SENTENCE_HARD_MAX = 40         # odrzucenie (bylo: 25)

# Progi dla walidatora
SENTENCE_AVG_MAX_ALLOWED = 22  # max srednia zanim retry (bylo: 20)
SENTENCE_RETRY_THRESHOLD = 30  # hard retry (bylo: 25)

# Max przecinkow w jednym zdaniu
SENTENCE_MAX_COMMAS = 4        # NKJP: przecinek czestszy niz litera "b" (bylo: 2)

# Fix #44: Keyword anti-stuffing — bez zmian
KEYWORD_MAIN_MAX_PER_BATCH = 2
KEYWORD_MIN_SPACING_WORDS = 80
