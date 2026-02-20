"""
Shared sentence length constants for Brajn2026.
Fix #9 v4.2: Ujednolicenie targetow dlugosci zdan.
Fix #34: Zaostrzenie — krotsze, prostsze zdania (max 1 przecinek).
Fix #44: Dalsze zaostrzenie — HARD_MAX 25→22, retry threshold 20→16.
Fix #53: Rebalans — avg 8 slow za krotko, poluzowanie targetow.

Uzywany przez: prompt_builder.py, ai_middleware.py
"""

# Srednia dlugosc zdania (target)
SENTENCE_AVG_TARGET = 14       # czytelne, ale nie urwane zdania (bylo 12)
SENTENCE_AVG_TARGET_MIN = 10   # dolna granica (bylo 8 — za krotko!)
SENTENCE_AVG_TARGET_MAX = 18   # gorna granica sredniej (bylo 14)

# Maksymalna dlugosc pojedynczego zdania
SENTENCE_SOFT_MAX = 22         # warning jesli przekroczone (bylo 18)
SENTENCE_HARD_MAX = 28         # odrzucenie/retry jesli przekroczone (bylo 22)

# Progi dla walidatora
SENTENCE_AVG_MAX_ALLOWED = 19  # max srednia zanim retry (bylo 15)
SENTENCE_RETRY_THRESHOLD = 21  # hard retry jesli srednia > 21 (bylo 16)

# Struktura zdania
SENTENCE_MAX_COMMAS = 2        # max 2 przecinki w zdaniu (bylo 1 — zbyt restrykcyjne)

# Fix #44: Keyword anti-stuffing
KEYWORD_MAIN_MAX_PER_BATCH = 2   # max uzyc glownej frazy w jednym batchu
KEYWORD_MIN_SPACING_WORDS = 80   # min odleglosc miedzy powtorzeniami
