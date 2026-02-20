"""
═══════════════════════════════════════════════════════════
BRAJEN PROMPT BUILDER v1.1
═══════════════════════════════════════════════════════════
Converts raw pre_batch data into optimized, readable prompts.

v1.1 changes:
  - _fmt_keywords(): calculates remaining from actual + target_total
    (backend sends these but NOT remaining directly)
  - Shows hard_max_this_batch so Claude knows per-batch limits
  - Clearer MUST/EXTENDED/STOP formatting

Architecture:
  SYSTEM PROMPT = Expert persona + Writing techniques
  USER PROMPT   = Structured instructions from data
═══════════════════════════════════════════════════════════
"""

import json
import logging

# Fix #9 v4.2 + Fix #34: import shared sentence-length constants (zaostrzenie)
try:
    from shared_constants import (
        SENTENCE_AVG_TARGET, SENTENCE_AVG_TARGET_MIN, SENTENCE_AVG_TARGET_MAX,
        SENTENCE_SOFT_MAX, SENTENCE_HARD_MAX, SENTENCE_AVG_MAX_ALLOWED,
        SENTENCE_MAX_COMMAS
    )
except ImportError:
    # Fallback defaults — Fix #34: zaostrzenie
    SENTENCE_AVG_TARGET = 12
    SENTENCE_AVG_TARGET_MIN = 8
    SENTENCE_AVG_TARGET_MAX = 15
    SENTENCE_SOFT_MAX = 20
    SENTENCE_HARD_MAX = 25
    SENTENCE_AVG_MAX_ALLOWED = 16
    SENTENCE_MAX_COMMAS = 1

_pb_logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ════════════════════════════════════════════════════════════

def _word_trim(text, max_chars):
    """Ucina tekst do max_chars na granicy slowa. Dodaje '...' jesli ucial."""
    if not text or len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    nl = chr(10)
    last_break = max(trimmed.rfind(" "), trimmed.rfind(nl), trimmed.rfind(". "))
    if last_break > max_chars // 2:
        trimmed = trimmed[:last_break]
    return trimmed.rstrip(" ,;:") + "..."


def build_system_prompt(pre_batch, batch_type):
    """
    Build system prompt = rola + cel + zasady + przykłady.
    v52.5: Nowa architektura — ROLA/CEL/ODBIORCA/TON + ZASADY + FEW-SHOT.
    gpt_instructions_v39 i gpt_prompt przeniesione do user promptu.
    """
    pre_batch = pre_batch or {}

    parts = []

    # ════════════════════════════════════════════════════════════
    # ROLA
    # ════════════════════════════════════════════════════════════
    parts.append("""<role>
Jesteś redaktorem naczelnym specjalistycznych serwisów branżowych
z 20-letnim doświadczeniem redakcyjnym i merytorycznym.
Publikujesz teksty eksperckie dla wymagającego czytelnika.

Nie jesteś copywriterem sprzedażowym.
Nie jesteś blogerem.
Nie jesteś chatbotem.

Twoim standardem jest jakość redakcyjna właściwa dla mediów specjalistycznych.
</role>""")

    # ════════════════════════════════════════════════════════════
    # CEL NADRZĘDNY
    # ════════════════════════════════════════════════════════════
    parts.append("""<goal>
Twoim celem jest wyczerpanie Search Intent użytkownika,
a nie "napisanie tekstu SEO".

Tekst ma:
  • rozwiązać problem,
  • odpowiedzieć na wszystkie logiczne pytania wynikające z tematu,
  • uporządkować wiedzę,
  • budować pełny kontekst przyczynowo-skutkowy,
  • tworzyć klaster tematyczny wokół zagadnienia.

SEO jest efektem ubocznym kompletności i precyzji.
</goal>""")

    # ════════════════════════════════════════════════════════════
    # ODBIORCA
    # ════════════════════════════════════════════════════════════
    parts.append("""<audience>
Domyślnie: czytelnik zaawansowany.
  • Używaj terminologii branżowej naturalnie.
  • Nie definiuj oczywistości dla zaawansowanych.
  • Jeśli artykuł kierowany jest do laika — zdefiniuj termin
    przy pierwszym użyciu krótko i rzeczowo.

Nigdy nie upraszczaj nadmiernie, jeśli kontekst tego nie wymaga.
</audience>""")

    # ════════════════════════════════════════════════════════════
    # TON I STYL
    # ════════════════════════════════════════════════════════════
    parts.append("""<tone>
Tematy prawne / medyczne / finansowe (YMYL):
  • ton formalny,
  • język precyzyjny,
  • brak potoczności,
  • brak metafor i kolokwializmów.

Tematy praktyczne / lifestylowe:
  • przystępny, ale nadal rzeczowy,
  • bez frywolności.
</tone>""")

    # ════════════════════════════════════════════════════════════
    # EPISTEMOLOGIA — ZASADA ŹRÓDEŁ
    # ════════════════════════════════════════════════════════════
    parts.append("""<epistemology>
SKĄD BIERZESZ WIEDZĘ — ZASADA BEZWZGLĘDNA:

Twoja wiedza pochodzi WYŁĄCZNIE z:
  1. Stron konkurencji z SERP (podane w danych) — czytasz fakty, NIE kopiujesz zdań
  2. Przepisów prawnych i orzeczeń sądowych (podane wprost w kontekście)
  3. Artykułów Wikipedia (podane wprost) — możesz cytować jako źródło uzupełniające
  4. Danych liczbowych z podanych źródeł — tylko gdy potwierdzone min. na 2 stronach SERP

❌ ZAKAZ BEZWZGLĘDNY — halucynacji faktograficznych:
  • Nie wymyślaj liczb, dat, statystyk, wyroków, sygnatur, instytucji
  • Nie wymyślaj nazw badań, raportów, publikacji naukowych
  • Nie podawaj wartości, kwot, terminów, artykułów ustaw których nie masz w danych
  • Nie "uzupełniaj luk" własnymi domysłami — lepiej pomiń niż zmyśl

JEŚLI NIE WIESZ → OPUŚĆ zdanie:
  • Brakuje sygnatury? → nie cytuj wyroku wcale
  • Nie znasz artykułu ustawy? → usuń zdanie z odwołaniem do prawa
  • Masz sprzeczne dane? → podaj zakres lub pomiń
</epistemology>""")

    # ════════════════════════════════════════════════════════════
    # TERMINOLOGIA I ENCJE
    # ════════════════════════════════════════════════════════════
    parts.append("""<entities>
Buduj klastry semantyczne, nie luźne słowa kluczowe.

  "rozwód" → pozew, władza rodzicielska, alimenty,
              orzeczenie o winie, podział majątku
  "kredyt hipoteczny" → zdolność kredytowa, wkład własny,
                        RRSO, marża banku
  "jazda po alkoholu" → art. 178a KK, stan nietrzeźwości,
                        zakaz prowadzenia, świadczenie pieniężne

Encje: powiązane logicznie, osadzone w kontekście
przyczynowo-skutkowym, naturalne w strukturze tekstu.
Nie stosuj przypadkowych wypełniaczy encyjnych.
</entities>""")

    # ════════════════════════════════════════════════════════════
    # ZASADY PISANIA
    # ════════════════════════════════════════════════════════════
    parts.append("""<rules>

FEATURED SNIPPET OPTIMIZATION (KRYTYCZNE dla pozycji 0)

ANSWER-FIRST: Pod każdym H2 MUSISZ zacząć od bezpośredniej odpowiedzi 40-58 słów.
Te 40-58 słów to "snippet-ready passage" — Google może je wyciąć jako Featured Snippet.
Odpowiedź musi być SAMODZIELNA (bez "jak wspomniano", "dlatego właśnie").
Po snippet-ready passage rozwijasz temat w kolejnych akapitach.

LISTY HTML: W CAŁYM artykule MUSISZ użyć DOKŁADNIE 2 wypunktowań:
  • Użyj <ul> dla kolekcji (objawy, cechy, typy) lub <ol> dla kroków/procesu
  • Każda lista: 5-8 elementów, każdy <li> to 1 konkretne zdanie (nie samo słowo)
  • Lista MUSI być poprzedzona zdaniem wprowadzającym kończącym się dwukropkiem
  • Rozmieść listy w RÓŻNYCH sekcjach H2 (nie obie w jednej)

TABELA HTML (opcjonalnie, max 1 na artykuł):
  • Użyj <table> (NIE CSS grid) do porównań, danych liczbowych, typów
  • Max 4-5 kolumn, 3-6 wierszy + nagłówek <thead>
  • Komórki krótkie (≤25 znaków)
  • Tabela ZAMIAST jednego z wypunktowań (czyli: 2 listy LUB 1 lista + 1 tabela)

PASSAGE-FIRST + RÓŻNORODNOŚĆ OTWARĆ
Każda sekcja H2 MUSI zaczynać się INNYM wzorcem składniowym.
ZAKAZ: dwie sąsiednie sekcje o identycznej strukturze pierwszego zdania.

Dostępne wzorce otwarcia sekcji — rotuj między nimi:

  A) LICZBA / FAKT (zaczyna od konkretu):
     „Mandaty za jazdę po alkoholu wahają się od 2500 do 30 000 zł..."
     „Trzy lata pozbawienia wolności — tyle grozi za pierwsze wykroczenie..."

  B) WARUNEK / PRÓG (zaczyna od „jeśli/gdy/przy"):
     „Gdy stężenie alkoholu przekracza 0,5 promila, czyn staje się przestępstwem..."
     „Przy pozytywnym wyniku testu policja zatrzymuje prawo jazdy na miejscu..."

  C) SKUTEK WPROST (zaczyna od konsekwencji):
     „Konfiskata pojazdu grozi każdemu, kto zostanie skazany po raz drugi..."
     „Zakaz prowadzenia trwa od 3 do 15 lat — sąd nie może go skrócić..."

  D) KONTRAST / ROZRÓŻNIENIE (zaczyna od różnicy):
     „Wykroczenie i przestępstwo — granica przebiega dokładnie przy 0,2 promila..."
     „Recydywista i osoba karana po raz pierwszy odpowiadają inaczej..."

  E) PODMIOT + ORZECZENIE (klasyczne, ale nie zawsze pierwsze):
     „Stan po użyciu alkoholu to poziom 0,2–0,5 promila we krwi..."
     „Przepadek pojazdu obowiązuje automatycznie od nowelizacji z 2023 roku..."

  F) PYTANIE + NATYCHMIASTOWA ODPOWIEDŹ (pytanie retoryczne tylko jako opener):
     „Czy można ubiegać się o warunkowe umorzenie? Tak — ale tylko przy pierwszym wykroczeniu..."

REGUŁA: batch 1=wzorzec A lub B, batch 2=inny, batch 3=inny itd.
W obrębie jednego batcha każda sekcja H3 też musi startować innym wzorcem.

SEARCH INTENT COVERAGE
Pokryj: pytania jawne, pytania domyślne, konsekwencje praktyczne,
ryzyka, alternatywy, wyjątki.

KAUZALNOŚĆ
Buduj ciągi: przyczyna → mechanizm → skutek → konsekwencja praktyczna.
Wzorce: powoduje, skutkuje, prowadzi do, zapobiega, w wyniku, ponieważ
✅ "Wzrost temperatury powyżej 100°C powoduje wrzenie, co prowadzi do parowania."
❌ "Temperatura wynosi X°C." (suche stwierdzenie bez funkcji)

BURSTINESS — rytm zdań (cel: CV zdań 0.30–0.45, śr. 14–18 słów)

Rozkład długości zdań w każdym akapicie:
  • 20% krótkich (do 10 słów) — fakty, definicje, konkrety
  • 55% średnich (11–20 słów) — rdzeń tekstu, naturalny styl
  • 25% dłuższych (21–26 słów) — złożone wyjaśnienia, MAX 2 przecinki

TWARDE LIMITY:
  • ŻADNE zdanie nie może przekroczyć 28 słów — jeśli tak jest, ROZBIJ je.
  • Średnia w całym batchu: cel 14–18 słów/zdanie (max dopuszczalna: 19).
  • MAX 2 PRZECINKI na zdanie. Zdanie z 3+ przecinkami = ZA ZŁOŻONE → rozbij.
  • NIE ZACZYNAJ wielu zdań od tej samej frazy — to spam, nie treść ekspercka.
  • WAŻNE: Unikaj URWANYCH zdań (3-6 słów bez treści). Każde zdanie musi nieść informację.

Reguła przecinków:
  ✅ „Zakaz prowadzenia pojazdów trwa od 3 do 15 lat i nie podlega zawieszeniu."
  ✅ „Mandat wynosi od 2500 zł, a w przypadku recydywy górna granica to 30 000 zł."
  ❌ „Kierowca może otrzymać mandat w wysokości od 2500 do 30 000 zł, a sąd dodatkowo cofa prawo jazdy, co oznacza zakaz prowadzenia, który trwa minimum 3 lata." (4 przecinki = za złożone)

Technika rozbijania:
  ✅ Jedno zdanie = jedna główna myśl. Dopuszczalne jedno rozwinięcie po przecinku.
  ✅ Długa wyliczanka → zdanie wprowadzające + lista HTML (ul/li)
  ✅ Zamiast łańcucha „bo… ponieważ… gdyż…" → nowe zdanie.

Sygnały Frankenstein (równa długość wszystkich zdań): monotonne. UNIKAJ.
  ✅ Krótkie zdanie niesie konkret: "Zakaz trwa od 3 do 15 lat."
  ❌ ZAKAZ zdań-dramatyzatorów (krótkie zdanie jako "myśl" lub "pointa"):
    "Granice są sztywne." / "Sąd patrzy. I słucha." / "I protokół."
    "To nie jest sprawa na skróty." / "Liczy się uzasadnienie."
    "W tle zostaje pytanie." — tania publicystyka, nie tekst ekspercki.

SUBJECT POSITION — (reguła rotacji encji wstrzykiwana dynamicznie per batch poniżej)

SENTENCE LENGTH — długość zdań (KRYTYCZNE dla czytelności)
  Maksimum bezwzględne: 28 słów (HARD_MAX). Rozbij zdania >28 słów.
  Cel średniej: 14–18 słów na zdanie (target: 14, max dopuszczalna: 19).
  MAX 2 przecinki na zdanie. Unikaj URWANYCH mini-zdań (3-6 słów).
  ✅ „Zakaz trwa od 3 do 15 lat. Sąd nie może od niego odstąpić."
  ❌ „Zakaz prowadzenia pojazdów mechanicznych, który sąd obligatoryjnie orzeka na mocy art. 178a Kodeksu karnego, obowiązuje przez okres od 3 do nawet 15 lat i nie podlega warunkowemu zawieszeniu."

SPACING — ANTYSPAM
Minimalna odległość między powtórzeniami frazy:
  MAIN: ~80 słów | BASIC: ~100 słów | EXTENDED: ~120 słów
  Nie klasteruj kilku fraz w jednym zdaniu.
  ABSOLUTNY ZAKAZ: nie powtarzaj głównej frazy w każdym akapicie.
  ABSOLUTNY ZAKAZ: nie zaczynaj 2+ zdań w jednym batchu od tej samej frazy kluczowej.
  Używaj synonimów, zaimków, omówień. Powtórzenie = spam.
  ❌ "Jazda po alkoholu... Jazda po alkoholu... Jazda po alkoholu..."
  ✅ "Prowadzenie pod wpływem... To zachowanie... Taki czyn..."

FLEKSJA
Odmiana frazy = jedno użycie.
  "zakaz prowadzenia" = "zakazu prowadzenia" = "zakazem prowadzenia"
  Pisz naturalnie, używaj różnych przypadków gramatycznych.

ANTY-AI — zakaz fraz-klisz (BEZWZGLĘDNY ZAKAZ — wszystkie tematy, zawsze)

KATEGORIA 1 — Zapowiadacze wagi (zamiast nich: podaj fakt wprost)
  „warto zauważyć / podkreślić / pamiętać / wiedzieć / mieć na uwadze"
  „należy podkreślić / zaznaczyć / mieć świadomość / wspomnieć"
  „co istotne / co ważne / co kluczowe / co warte uwagi"
  „kluczowe jest / kluczowym aspektem / kluczową kwestią"
  „nie ulega wątpliwości / nie można zapomnieć / nie można pominąć"
  „istotnym elementem jest / ważnym elementem jest / istotną kwestią"
  ✅ Zamiast: „Warto zauważyć, że zakaz trwa 3 lata." → „Zakaz trwa 3 lata."

KATEGORIA 2 — Puste przejścia i zapowiedzi
  „w tym kontekście / w kontekście powyższego / w tym miejscu"
  „przejdźmy teraz do / przyjrzyjmy się / skupmy się na"
  „kolejnym ważnym aspektem jest / następnym krokiem jest"
  „w dalszej części artykułu / jak wspomniano wcześniej (bez ref.)"
  „to prowadzi do kolejnego aspektu / to rodzi pytanie"
  ✅ Zamiast: „Przyjrzyjmy się karom." → H2: „Kary" + pierwsze zdanie z danymi.

KATEGORIA 3 — Fałszywe podsumowania i wnioski
  „podsumowując / podsumowując powyższe / reasumując"
  „w świetle powyższego / w związku z powyższym / jak widać"
  „można zatem stwierdzić / należy zatem podkreślić"
  „z powyższego wynika / wniosek jest następujący"
  „to kluczowa różnica / to najważniejsza kwestia"
  ✅ Zamiast: „Podsumowując, sankcje są surowe." → Zakończ sekcję konkretnym faktem.

KATEGORIA 4 — Nadmierny formalizm AI
  „każdorazowo należy / każdorazowo warto / każdorazowo wymaga"
  „rekomendowana jest konsultacja / zalecana jest konsultacja"
  „ze względu na złożoność / ze względu na specyfikę tematu"
  „ze względu na powyższe okoliczności / mając na uwadze powyższe"
  „w praktyce oznacza to / w praktyce wygląda to następująco"
  „należy zwrócić szczególną uwagę / wymaga szczególnej uwagi"
  ✅ Zamiast: „Ze względu na złożoność zagadnienia..." → Podaj konkret.

KATEGORIA 5 — Dramatyzatory i teatr
  „Granice są sztywne." / „Sąd patrzy. I słucha." / „I protokół."
  „To nie jest sprawa na skróty." / „Liczy się uzasadnienie."
  „W tle zostaje pytanie." / „Prawo nie wybacza."
  Krótkie zdanie jako dramatyczna pointa — ZAKAZ.
  ✅ Krótkie zdanie = TYLKO twarda liczba lub definicja.

KATEGORIA 6 — Placeholder-zdania (wtrącenia bez treści)
  „Istotnym elementem jest [powtórzenie frazy MUST bez treści]."
  „[Encja] jest ważnym pojęciem w tym kontekście."
  „Temat ten zasługuje na szczególną uwagę."
  Każde zdanie MUSI dodawać nową informację — nie zapowiadać jej.

KATEGORIA 7 — Phantom-placeholder prawny (BEZWZGLĘDNY ZAKAZ)
  ❌ „odpowiednich przepisów prawa" — ZAWSZE podaj konkretny artykuł: „art. 178a § 1 k.k."
  ❌ „właściwych przepisów" / „stosownych regulacji" / „obowiązujących przepisów" bez numeru — ZAKAZ
  ❌ „zgodnie z przepisami" bez podania jakich — ZAKAZ
  ❌ „do 2 lat więzienia" dla art. 178a § 1 k.k. — BŁĄD: nowelizacja 2023 = do 3 lat
  ❌ „recydywa w ciągu 2 lat" — BŁĄD: prawo karne nie definiuje recydywy terminem
  ❌ Sygnatura „I C" lub „II C" w kontekście konfiskaty pojazdu — BŁĄD: to sprawa cywilna
  ❌ „mg/100 ml" jako jednostka alkoholu — BŁĄD: używaj promili (‰) lub mg/dm³
  Reguła: jeśli nie znasz konkretnego artykułu → usuń zdanie, NIE zastępuj ogólnikiem.

KATEGORIA 8 — Halucynacje terminologiczne w prawie o alkoholu (BEZWZGLĘDNY ZAKAZ)
  ❌ „alkohol z natury" / „alkohol z urodzenia" — NONSENS, nie istnieje takie pojęcie
  ❌ „stężenie alkoholu z natury" / „promile z natury" / „promile z urodzenia" — NONSENS
  ❌ „opilstwo" — archaizm, nie używany w aktualnym prawie karnym
  ❌ „pijaństwo" w kontekście prawnym — używaj: „stan nietrzeźwości"
  ❌ „obsługiwał pojazd" / „zakaz obsługi pojazdu" — BŁĄD: używaj „prowadził pojazd" / „zakaz prowadzenia pojazdu"
  ✅ Poprawna terminologia: „stan po użyciu alkoholu" (0,2–0,5‰) | „stan nietrzeźwości" (powyżej 0,5‰)
  ✅ Jednostki: promile (‰) | mg/dm³ w wydychanym powietrzu (NIE: mg/100ml)

ANTY-POWTÓRZENIA
Zdefiniowałeś pojęcie raz — nie definiuj ponownie.
Odwołuj się: "wspomniany wcześniej X".
Brak powtórzeń leksykalnych w sąsiednich akapitach.
Brak powielania tej samej konstrukcji składniowej.

ANTY-MYŚLNIKI
Myślniki (—) stosuj MAX 1 na 3 akapity.
✅ Używaj przecinków, dwukropków, nawiasów, średników.
❌ "Wyrok — choć kontrowersyjny — został utrzymany." (co zdanie)
Nadmiar myślników = sygnał tekstu AI.

ANTY-PYTANIA-RETORYCZNE
MAX 1 pytanie retoryczne na sekcję H2.
❌ "Jak to wygląda w praktyce?", "Co to oznacza?", "Czy zawsze?"
✅ Przejdź bezpośrednio do informacji.

ANTY-FILLER
Każde zdanie MUSI dodawać nową informację.
❌ Truizmy: "Przewodnik elektryczny przewodzi prąd."
❌ Puste przejścia: "To prowadzi do kolejnego aspektu."
❌ Zapowiedzi: "Kolejna część artykułu wyjaśnia..."
❌ Puste podsumowania: "To kluczowa różnica technologiczna."
✅ "Miedź przewodzi prąd 6× lepiej niż żelazo, dlatego stanowi
   60% okablowania domowego."

ANTY-BRAND-STUFFING
Nazwy firm/marek: MAX 2× w całym artykule.

CYTOWANIE ŹRÓDEŁ (YMYL)
✅ Ustawy, artykuły KK/KC/KW, badania, instytucje oficjalne.
❌ Encje jako źródła: "Wikipedia podaje...", "Według [encji]..."
Podawaj fakty bezpośrednio. Źródło z nazwy — MAX 1 raz na artykuł.

ANTY-HALUCYNACJA
Jeśli brak pewnych danych — pomiń lub opisz zasadę ogólnie.
❌ Wymyślone statystyki, rozporządzenia, daty, ceny.
✅ Zasada ogólna bez numerów ustaw gdy nie masz pewności.

POLSZCZYZNA (NKJP, 1,8 mld segmentów)
→ PRZECINKI: obowiązkowe przed: że, który/a/e, ponieważ, gdyż,
  aby, żeby, jednak, lecz, ale.
  Brak przecinka przed "że" = natychmiastowy sygnał AI.
→ KOLOKACJE — używaj poprawnych połączeń:
  podjąć decyzję (NIE: zrobić), odnieść sukces (NIE: mieć),
  popełnić błąd (NIE: zrobić), ponieść konsekwencje (NIE: mieć),
  wysoki poziom (NIE: duży), wysokie ryzyko (NIE: duże),
  odgrywać rolę (NIE: pełnić), silny ból (NIE: duży),
  rzęsisty deszcz (NIE: duży), wysunąć propozycję (NIE: dać).
→ DŁUGOŚĆ ZDAŃ: średnio 10–15 słów (styl publicystyczny).
  NIE pisz wszystkich zdań jednej długości — to sygnał AI.
→ ŚREDNIA DŁUGOŚĆ WYRAZU: 6 znaków (±0,5).
  Nie nadużywaj nominalizacji.
→ DIAKRYTYKI: naturalny tekst ma ~7% ą,ę,ć,ł,ń,ó,ś,ź,ż.
→ Unikaj pleonazmów: "wzajemna współpraca",
  "aktualna sytuacja na dziś", "krótkie streszczenie".
→ Mieszaj przypadki gramatyczne — nie powtarzaj frazy w mianowniku.

FORMAT
h2:/h3: dla nagłówków. Zero markdown, HTML, gwiazdek.

</rules>""")

    # ════════════════════════════════════════════════════════════
    # DYNAMIC: SUBJECT POSITION — per-batch entity rotation
    # Injected HERE (not in static <rules>) so encja rotates per H2
    # ════════════════════════════════════════════════════════════
    section_lead = pre_batch.get("_section_lead_entity", "")
    main_kw = (pre_batch.get("main_keyword") or {}).get("keyword", "") if isinstance(pre_batch.get("main_keyword"), dict) else str(pre_batch.get("main_keyword") or "")
    if not section_lead:
        section_lead = main_kw

    if section_lead:
        # Build rotation list: lead entity first, then other MUST entities from pre_batch
        must_ents_raw = pre_batch.get("_must_cover_concepts") or pre_batch.get("enhanced", {}).get("must_cover_entities") or []
        must_names = []
        for e in must_ents_raw:
            name = (e.get("text", e.get("entity", "")) if isinstance(e, dict) else str(e)).strip()
            if name and name != section_lead and name not in must_names:
                must_names.append(name)

        # Build rotation instruction
        rotation_entities = [section_lead] + must_names[:3]
        if len(rotation_entities) == 1:
            rotation_str = '"' + section_lead + '"'
        else:
            rotation_str = " | ".join(
                f"akapit {i+1}: \"{e}\"" for i, e in enumerate(rotation_entities)
            )

        fallback_ent = must_names[0] if must_names else "Sad/Sprawca"
        sp_note = "" if section_lead == main_kw else (
            f"\n  (Encja glowna \"{main_kw}\" moze sie pojawiac, ale nie jest podmiotem tej sekcji.)"
        )
        rule_body = (
            "<subject_position_rule>\n"
            f"TEMAT TEJ SEKCJI: \"{section_lead}\"\n"
            f"W tej sekcji H2 kazdy akapit musi miec INNA encje jako podmiot otwierajacy.{sp_note}\n"
            "\n"
            f"ROTACJA PODMIOTOW - kolejnosc akapitow:\n"
            f"  {rotation_str}\n"
            "\n"
            "ZASADA: kazdy kolejny akapit otwiera INNA encja z powyzszej listy jako podmiot gramatyczny.\n"
            "Jesli sekcja ma 4 akapity -> 4 rozne encje jako podmiot pierwszego zdania.\n"
            "\n"
            "Przyklad rotacji (3 akapity):\n"
            f"  Akapit 1: \"{section_lead} [orzeczenie]...\"\n"
            f"  Akapit 2: \"{fallback_ent} [orzeczenie]...\"\n"
            "  Akapit 3: Liczba/fakt na poczatku lub kolejna encja MUST\n"
            "\n"
            "ZAKAZ: dwa akapity z rzedu otwarte ta sama encja.\n"
            "ZAKAZ: 'Istotnym aspektem jest [encja]...' - to orzecznik, nie podmiot.\n"
            "ZAKAZ: 'Zgodnie z przepisami o [encja]...' - to dopelnienie, nie podmiot.\n"
            "\n"
            "Google salience: podmiot x pozycja = 3-6x wyzszy wynik niz encja w dopelnieniu.\n"
            "Rotacja podmiotow = naturalne pokrycie wszystkich kluczowych encji tematu.\n"
            "</subject_position_rule>"
        )
        parts.append(rule_body)

    # ════════════════════════════════════════════════════════════
    # Fix #57: SEMANTIC KEYPHRASES — natural compound phrases
    # ════════════════════════════════════════════════════════════
    sem_kp = pre_batch.get("_semantic_keyphrases") or []
    if sem_kp:
        kp_lines = []
        for kp in sem_kp[:8]:
            phrase = kp.get("phrase", kp) if isinstance(kp, dict) else str(kp)
            if phrase:
                kp_lines.append(f"  • {phrase}")
        if kp_lines:
            parts.append(
                "<semantic_keyphrases>\n"
                "FRAZY SEMANTYCZNE — użyj minimum 3 z poniższych jako KOMPLETNE FRAZY (nie rozbijaj na osobne słowa):\n"
                + "\n".join(kp_lines) + "\n"
                "Każda fraza powinna pojawić się jako spójny ciąg słów w jednym zdaniu.\n"
                "Przykład: zamiast 'diagnostyka słuchu. Dziecka dotyczy...' → 'diagnostyka słuchu dziecka obejmuje...'\n"
                "</semantic_keyphrases>"
            )

    # ════════════════════════════════════════════════════════════
    # FEW-SHOT EXAMPLES
    # (Anthropic/OpenAI: przykłady skuteczniejsze niż instrukcje)
    # ════════════════════════════════════════════════════════════
    parts.append("""<examples>

PRZYKŁAD ZŁY — czego NIE pisać:
<example_bad>
Jazda po alkoholu to poważne przestępstwo w Polsce. Sąd patrzy. I słucha.
Granice są sztywne. Kancelaria posiada duże doświadczenie w sprawach
karnych ruchu drogowego. Kancelaria posiada duże doświadczenie w sprawach
karnych ruchu drogowego. Ta instytucja daje sądowi możliwość odstąpienia
od wymierzenia środka, co warto zauważyć i należy podkreślić.
</example_bad>
Błędy: dramatyzatory ("Sąd patrzy. I słucha."), powtórzenie zdania 2×,
frazy AI ("warto zauważyć"), brak liczb, puste stwierdzenia.

PRZYKŁAD DOBRY — tak pisz:
<example_good>
Skazanie z art. 178a § 1 KK grozi pozbawieniem wolności do 3 lat
oraz obligatoryjnym zakazem prowadzenia pojazdów od 3 do 15 lat.
Sąd nie ma tu uznaniowości — zakaz jest obowiązkowy przy każdym
wyroku skazującym, niezależnie od okoliczności łagodzących.
Jedyną zmienną pozostaje jego wymiar, który sąd ustala biorąc pod
uwagę stopień zawinienia i dotychczasową karalność sprawcy.
</example_good>
Zalety: konkretny artykuł KK, konkretne liczby (3 lata, 3–15 lat),
kauzalność (obligatoryjny → brak uznaniowości → jedyna zmienna),
zero fraz AI, zero powtórzeń.

</examples>""")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# Schema guard — field validation
# ═══════════════════════════════════════════════════════════

_CRITICAL_FIELDS = [
    "keywords",             # keyword list: without this, article has no SEO
    "main_keyword",         # primary keyword
    "batch_number",         # batch sequencing
]
_IMPORTANT_FIELDS = [
    "gpt_instructions_v39", # backend writing instructions
    "enhanced",             # enhanced_pre_batch AI data
    "h2_remaining",         # H2 structure
    "article_memory",       # context from previous batches
    "keyword_limits",       # STOP/EXCEEDED rules
    "coverage",             # keyword coverage state
]

def _schema_guard(pre_batch):
    """Validate pre_batch has critical fields. Log warnings for missing."""
    missing_critical = [f for f in _CRITICAL_FIELDS if f not in pre_batch or pre_batch[f] is None]
    missing_important = [f for f in _IMPORTANT_FIELDS if f not in pre_batch or pre_batch[f] is None]

    if missing_critical:
        _pb_logger.warning(
            f"⚠️ SCHEMA GUARD: Missing CRITICAL fields: {missing_critical}. "
            f"Backend may have changed API. Article quality will be degraded."
        )
    if missing_important:
        _pb_logger.info(
            f"ℹ️ Schema guard: Missing optional fields: {missing_important} "
            f"(batch {pre_batch.get('batch_number', '?')})"
        )

    # Validate enhanced sub-fields if enhanced exists
    enhanced = pre_batch.get("enhanced") or {}
    if enhanced:
        expected_enhanced = [
            "smart_instructions_formatted", "causal_context",
            "information_gain", "relations_to_establish"
        ]
        missing_enh = [f for f in expected_enhanced if not enhanced.get(f)]
        if missing_enh:
            _pb_logger.info(f"ℹ️ Enhanced missing: {missing_enh}")


def build_user_prompt(pre_batch, h2, batch_type, article_memory=None):
    """
    Main user prompt builder.
    Converts ALL pre_batch fields into readable, actionable instructions.
    Each section is wrapped in try/except so one bad field won't crash generation.
    """
    pre_batch = pre_batch or {}
    sections = []

    # ── RE-ANCHOR: krótkie przypomnienie roli (dokumentacja Anthropic: re-anchor w user prompcie) ──
    # Dla YMYL dodaje ostrzeżenie o weryfikacji wyroków
    detected_category = pre_batch.get("detected_category", "")
    if detected_category == "prawo":
        sections.append(
            "Piszesz jako redaktor naczelny — ton formalny, zero frywolności. "
            "Wyroki cytuj TYLKO jeśli sygnatura pasuje do gałęzi prawa artykułu "
            "(II K/AKa = karne, I C/ACa = cywilne). Szczegółowe zasady w system prompcie."
        )
    elif detected_category in ("medycyna", "finanse"):
        sections.append(
            "Piszesz jako redaktor naczelny — ton formalny, precyzyjny. "
            "Cytuj TYLKO pewne dane. Szczegółowe zasady w system prompcie."
        )
    else:
        sections.append(
            "Piszesz jako redaktor naczelny — rzeczowo, bez frywolności. "
            "Szczegółowe zasady w system prompcie."
        )

    # ── OPENING PATTERN — per-batch rotation (zapobiega identycznym otwarciom sekcji) ──
    _OPENING_PATTERNS = [
        ("A", "LICZBA/FAKT",
         "Zacznij sekcje od konkretnej liczby, daty lub wartosci. Np: '3 lata - tyle wynosi...', 'Od 2500 do 30 000 zl...'"),
        ("B", "WARUNEK",
         "Zacznij sekcje od warunku lub progu. Np: 'Gdy stezenie przekracza...', 'Jesli kierowca...', 'Przy kazdym kolejnym...'"),
        ("C", "SKUTEK WPROST",
         "Zacznij sekcje od konsekwencji. Np: 'Konfiskata grozi kazdemu...', 'Zakaz trwa od 3 do 15 lat - sad nie moze...'"),
        ("D", "KONTRAST",
         "Zacznij sekcje od rozroznienia. Np: 'Wykroczenie i przestepstwo - granica przebiega...', 'Recydywista odpowiada inaczej...'"),
        ("E", "PODMIOT+ORZECZENIE",
         "Zacznij sekcje klasycznie: podmiot + orzeczenie z konkretem. Np: 'Stan po uzyciu alkoholu to poziom 0,2-0,5 promila...'"),
        ("F", "PYTANIE+ODPOWIEDZ",
         "Zacznij sekcje pytaniem z natychmiastowa odpowiedzia. Np: 'Czy mozna unikac zakazu? Tak, ale tylko gdy...'"),
    ]
    batch_num = pre_batch.get("batch_number", 1) or 1
    if batch_type in ("INTRO", "intro"):
        pattern_idx = 0  # INTRO: zawsze liczba/fakt dla silnego otwarcia
    else:
        pattern_idx = (batch_num - 1) % len(_OPENING_PATTERNS)
    p_letter, p_name, p_desc = _OPENING_PATTERNS[pattern_idx]
    sections.append(
        f"OTWARCIE TEJ SEKCJI — wzorzec {p_letter} ({p_name}):\n"
        f"{p_desc}\n"
        f"ZAKAZ: nie zaczynaj od encji jako podmiotu w stylu '[X] jest/to/oznacza' — to wzorzec już użyty w poprzednich sekcjach."
    )

    # ── SCHEMA GUARD: validate critical fields from backend ──
    _schema_guard(pre_batch)

    formatters = [
        # ── TIER 1: NON-NEGOTIABLE (backend hard rules) ──
        lambda: _fmt_batch_header(pre_batch, h2, batch_type),
        lambda: _fmt_keywords(pre_batch),           # MUST/STOP/EXCEEDED: hardest constraints
        lambda: _fmt_smart_instructions(pre_batch),  # enhanced_pre_batch AI instructions
        lambda: _fmt_legal_medical(pre_batch),        # YMYL: legal compliance, non-negotiable

        # ── TIER 2: BACKEND WRITE INSTRUCTIONS (gpt_instructions_v39 etc.) ──
        lambda: _fmt_semantic_plan(pre_batch, h2),
        lambda: _fmt_coverage_density(pre_batch),
        lambda: _fmt_phrase_hierarchy(pre_batch),
        lambda: _fmt_continuation(pre_batch),
        lambda: _fmt_article_memory(article_memory),
        lambda: _fmt_h2_remaining(pre_batch),

        # ── TIER 3: CONTENT CONTEXT (enrichment data) ──
        lambda: _fmt_entity_salience(pre_batch),     # entity positioning rules (salience only)
        # _fmt_entities REMOVED v45.4.1: gpt_instructions_v39 already contains
        # curated "🧠 ENCJE:" section (max 3/batch, importance≥0.7, with HOW hints).
        # Our version duplicated it with dirtier, unfiltered data from S1.
        # _fmt_ngrams REMOVED v45.4.1: raw statistical n-grams from competitor
        # pages often contain CSS/JS artifacts ("button button", "block embed").
        # Custom GPT never sees these and produces better text without them.
        lambda: _fmt_serp_enrichment(pre_batch),
        lambda: _fmt_causal_context(pre_batch),
        lambda: _fmt_depth_signals(pre_batch),       # depth signals when previous batch scored low
        lambda: _fmt_experience_markers(pre_batch),
        lambda: _fmt_natural_polish(pre_batch),      # v50: fleksja, spacing, anti-stuffing

        # ── TIER 4: SOFT GUIDELINES (format, style, intro) ──
        lambda: _fmt_intro_guidance(pre_batch, batch_type),
        lambda: _fmt_style(pre_batch),
        lambda: _fmt_output_format(h2, batch_type),
    ]

    for fmt in formatters:
        try:
            result = fmt()
            if result:
                sections.append(result)
        except Exception:
            pass

    return "\n\n".join(sections)


# ════════════════════════════════════════════════════════════
# SECTION FORMATTERS
# ════════════════════════════════════════════════════════════

def _fmt_batch_header(pre_batch, h2, batch_type):
    batch_number = pre_batch.get("batch_number", 1)
    total_batches = pre_batch.get("total_planned_batches", 1)
    batch_length = pre_batch.get("batch_length") or {}

    min_w = batch_length.get("min_words", 350)
    max_w = batch_length.get("max_words", 500)

    section_length = pre_batch.get("section_length_guidance") or {}
    length_hint = ""
    if section_length:
        suggested = section_length.get("suggested_words") or section_length.get("target_words")
        if suggested:
            length_hint = f"\nSugerowana długość tej sekcji: ~{suggested} słów."

    h2_instruction = ""
    if batch_type not in ("INTRO", "intro"):
        h2_instruction = f"\nZaczynaj DOKŁADNIE od: h2: {h2}"

    return f"""═══ BATCH {batch_number}/{total_batches}: {batch_type} ═══
Sekcja H2: "{h2}"
Długość: {min_w}-{max_w} słów{length_hint}{h2_instruction}"""


def _fmt_intro_guidance(pre_batch, batch_type):
    if batch_type not in ("INTRO", "intro"):
        return ""
    guidance = pre_batch.get("intro_guidance", "")

    main_kw = pre_batch.get("main_keyword") or {}
    kw_name = main_kw.get("keyword", "") if isinstance(main_kw, dict) else str(main_kw)

    parts = ["═══ WPROWADZENIE (WSTĘP ARTYKUŁU) ═══",
             "To jest PIERWSZY batch, piszesz WSTĘP artykułu.",
             "MUSISZ:",
             f'  1. Wpleć frazę główną ("{kw_name}") w PIERWSZE zdanie' if kw_name else "  1. Frazę główną umieść w pierwszym zdaniu",
             "  2. Zacząć od angażującego haka (hook): pytanie, statystyka, scenariusz",
             "  3. Przedstawić GŁÓWNĄ TEZĘ artykułu w 1-2 zdaniach",
             "  4. Zapowiedzieć co czytelnik znajdzie dalej (bez listy H2!)",
             "  5. NIE zaczynać od definicji ani od 'W dzisiejszych czasach...'",
             "  6. NIE dodawać nagłówka h2: (wstęp nie ma nagłówka",
             "  7. Utrzymać zwięzłość; wstęp to 80-150 słów"]

    if guidance:
        if isinstance(guidance, dict):
            hook = guidance.get("hook", "")
            angle = guidance.get("angle", "")
            if hook:
                parts.append(f"\nHak otwierający: {hook}")
            if angle:
                parts.append(f"Kąt artykułu: {angle}")
        else:
            parts.append(f"\n{guidance}")

    # AI Overview — tylko we wstępie, żeby intro odpowiadało na to co Google już pokazuje
    serp = pre_batch.get("serp_enrichment") or {}
    ai_ov = serp.get("ai_overview") or {}
    if isinstance(ai_ov, dict):
        ai_ov_text = ai_ov.get("text", "") or ""
    elif isinstance(ai_ov, str):
        ai_ov_text = ai_ov
    else:
        ai_ov_text = ""
    if ai_ov_text and len(ai_ov_text) > 50:
        parts.append("\n═══ GOOGLE AI OVERVIEW ═══")
        parts.append("Google wyświetla użytkownikom ten tekst ZANIM klikną w artykuł.")
        parts.append("Wstęp MUSI nawiązywać do tego kontekstu i obiecywać głębszą odpowiedź:")
        parts.append(f"  {ai_ov_text[:500]}")

    return "\n".join(parts)


def _fmt_smart_instructions(pre_batch):
    """Smart instructions from enhanced_pre_batch : THE most valuable field."""
    enhanced = pre_batch.get("enhanced") or {}
    smart = enhanced.get("smart_instructions_formatted", "")
    if smart:
        return f"═══ INSTRUKCJE DLA TEGO BATCHA ═══\n{smart[:1000]}"
    return ""


def _parse_target_max(target_total_str):
    """
    Parse target_max from backend's target_total field.
    Backend sends target_total as "min-max" string (e.g., "2-6").
    Returns max value as int, or 0 if unparseable.
    """
    if not target_total_str:
        return 0
    if isinstance(target_total_str, (int, float)):
        return int(target_total_str)
    try:
        parts = str(target_total_str).replace("x", "").split("-")
        if len(parts) >= 2:
            return int(parts[-1].strip())
        return int(parts[0].strip())
    except (ValueError, IndexError):
        return 0


def _fmt_keywords(pre_batch):
    """
    Format keywords section with CALCULATED remaining_max.
    
    v1.1: Backend sends actual (current uses) and target_total ("min-max")
    but NOT remaining. We calculate: remaining = target_max - actual.
    Also shows hard_max_this_batch so Claude knows per-batch limits.
    """
    keywords_info = pre_batch.get("keywords") or {}
    keyword_limits = pre_batch.get("keyword_limits") or {}
    soft_caps = pre_batch.get("soft_cap_recommendations") or {}

    # ── MUST USE (with calculated remaining) ──
    must_raw = keywords_info.get("basic_must_use", [])
    must_lines = []
    for kw in must_raw:
        if isinstance(kw, dict):
            name = kw.get("keyword", "")
            
            # Calculate remaining from actual + target_total
            actual = kw.get("actual", kw.get("actual_uses", kw.get("current_count", 0)))
            target_total = kw.get("target_total", "")
            target_max = _parse_target_max(target_total) or kw.get("target_max", 0)
            hard_max = kw.get("hard_max_this_batch", "")
            use_range = kw.get("use_this_batch", "")
            
            # Explicit remaining from backend (if sent), otherwise calculate
            remaining = kw.get("remaining", kw.get("remaining_max", ""))
            if not remaining and target_max and isinstance(actual, (int, float)):
                remaining = max(0, target_max - int(actual))
            
            # Build descriptive line
            parts_line = [f'"{name}"']
            if remaining:
                parts_line.append(f"zostało {remaining}× ogółem")
            if hard_max:
                parts_line.append(f"max {hard_max}× w tym batchu")
            elif use_range:
                parts_line.append(f"cel: {use_range}× w tym batchu")
            
            must_lines.append(f'  • {", ".join(parts_line)}')
        else:
            must_lines.append(f'  • "{kw}"')

    # ── EXTENDED (with remaining) ──
    ext_raw = keywords_info.get("extended_this_batch", [])
    ext_lines = []
    for kw in ext_raw:
        if isinstance(kw, dict):
            name = kw.get("keyword", "")
            actual = kw.get("actual", kw.get("actual_uses", 0))
            target_total = kw.get("target_total", "")
            target_max = _parse_target_max(target_total) or kw.get("target_max", 0)
            remaining = kw.get("remaining", kw.get("remaining_max", ""))
            if not remaining and target_max and isinstance(actual, (int, float)):
                remaining = max(0, target_max - int(actual))
            
            line = f'  • "{name}"'
            if remaining:
                line += f" , zostało {remaining}×"
            ext_lines.append(line)
        else:
            ext_lines.append(f'  • "{kw}"')

    # ── STOP ──
    stop_raw = keyword_limits.get("stop_keywords") or []
    stop_lines = []
    for s in stop_raw:
        if isinstance(s, dict):
            name = s.get("keyword", "")
            current = s.get("current_count", s.get("current", s.get("actual", "?")))
            max_c = s.get("max_count", s.get("max", s.get("target_max", "?")))
            stop_lines.append(f'  • "{name}" (już {current}×, limit {max_c}) , STOP!')
        else:
            stop_lines.append(f'  • "{s}"')

    # ── CAUTION ──
    caution_raw = keyword_limits.get("caution_keywords") or []
    caution_lines = []
    for c in caution_raw:
        if isinstance(c, dict):
            name = c.get("keyword", "")
            current = c.get("current_count", c.get("current", c.get("actual", "")))
            max_c = c.get("max_count", c.get("max", c.get("target_max", "")))
            line = f'  • "{name}"'
            if current and max_c:
                line += f" ({current}/{max_c})"
            line += " , max 1× w tym batchu"
            caution_lines.append(line)
        else:
            caution_lines.append(f'  • "{c}" , max 1×')

    # ── SOFT CAPS ──
    soft_notes = []
    if soft_caps:
        for kw_name, info in soft_caps.items():
            if isinstance(info, dict):
                action = info.get("action", "")
                if action and action != "OK":
                    soft_notes.append(f'  ℹ️ "{kw_name}": {action}')

    # ── Build section ──
    parts = ["═══ FRAZY KLUCZOWE ═══"]

    if must_lines:
        parts.append("🔴 OBOWIĄZKOWE (wpleć naturalnie w tekst):")
        parts.extend(must_lines)

    if ext_lines:
        parts.append("\n🟡 ROZSZERZONE (użyj jeśli pasują do kontekstu):")
        parts.extend(ext_lines)

    if stop_lines:
        parts.append("\n🛑 STOP, NIE UŻYWAJ (przekroczone limity!):")
        parts.extend(stop_lines)

    if caution_lines:
        parts.append("\n⚠️ OSTROŻNIE, użyj max 1× lub pomiń:")
        parts.extend(caution_lines)

    if soft_notes:
        parts.append("")
        parts.extend(soft_notes)

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_semantic_plan(pre_batch, h2):
    plan = pre_batch.get("semantic_batch_plan") or {}
    if not plan:
        return ""

    parts = ["═══ CO PISAĆ W TEJ SEKCJI ═══"]

    h2_coverage = plan.get("h2_coverage") or {}
    for h2_name, info in h2_coverage.items():
        if isinstance(info, dict):
            angle = info.get("semantic_angle", "")
            must = info.get("must_phrases", [])
            if angle:
                parts.append(f'Kąt semantyczny: {angle}')
            if must:
                phrases = ", ".join(f'"{p}"' for p in must[:5])
                parts.append(f'Obowiązkowe frazy w tej sekcji: {phrases}')

    density_targets = plan.get("density_targets") or {}
    overall = density_targets.get("overall")
    if overall:
        parts.append(f'Docelowa gęstość fraz: {overall}%')

    direction = plan.get("content_direction") or plan.get("writing_direction", "")
    if direction:
        parts.append(f'Kierunek treści: {direction}')

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_entity_salience(pre_batch):
    """Entity salience instructions : grammatical positioning, hierarchy.
    
    Based on:
    - Patent US10235423B2 (entity metrics)
    - Patent US9251473B2 (salient items in documents)
    - Dunietz & Gillick (2014) entity salience research
    - Google Cloud NLP API salience scoring
    
    v47.0: Also includes backend placement instructions from competitor analysis
    (entity_salience.py in gpt-ngram-api: salience scoring, co-occurrence, placement)
    
    Data sources:
    - pre_batch["_entity_salience_instructions"] : local positioning rules (from entity_salience.py frontend)
    - pre_batch["_backend_placement_instruction"] : backend placement from competitor analysis
    - pre_batch["_concept_instruction"] : topical concepts agent instruction
    - pre_batch["_must_cover_concepts"] : concept entities that must be covered
    """
    parts = []
    
    # 1. Local salience positioning rules
    local_instructions = pre_batch.get("_entity_salience_instructions", "")
    if local_instructions:
        parts.append(local_instructions)
    
    # 2. v47.0: Backend placement instructions (from gpt-ngram-api competitor analysis)
    backend_placement = pre_batch.get("_backend_placement_instruction", "")
    if backend_placement:
        parts.append("═══ ROZMIESZCZENIE ENCJI (z analizy konkurencji) ═══")
        parts.append(
            "⚠️ TO SĄ WSKAZÓWKI TECHNICZNE — NIE kopiuj ich dosłownie do tekstu!\n"
            "Użyj jako inspirację/tło. Pisz własne zdania. Nie przepisuj fragmentów poniżej."
        )
        parts.append(backend_placement)
    
    # 3. v47.0: Concept instruction + must-cover concepts
    # v52.1: Dodano instrukcję fleksji — encje podawane są w mianowniku, Claude musi je odmieniać
    FLEXION_NOTE = (
        "\n⚠️ FLEKSJA: Pojęcia są w mianowniku — odmieniaj je przez przypadki zależnie od kontekstu. "
        'Np. "gałka meblowa" → "gałki meblowej" (dop.), "gałkę meblową" (bier.). '
        "Gramatyczna poprawność > dosłowne powtórzenie formy bazowej."
    )
    concept_instr = pre_batch.get("_concept_instruction", "")
    must_concepts = pre_batch.get("_must_cover_concepts", [])
    if concept_instr:
        parts.append(concept_instr + FLEXION_NOTE)
    elif must_concepts:
        # Build instruction from concept list if no agent instruction provided
        concept_names = [c.get("text", c) if isinstance(c, dict) else str(c) for c in must_concepts[:10]]
        parts.append(
            "═══ POJĘCIA TEMATYCZNE (z analizy konkurencji) ═══\n"
            f"Następujące pojęcia pojawiają się u konkurencji, wpleć naturalnie w tekst:\n"
            f"{', '.join(concept_names)}"
            + FLEXION_NOTE
        )
    
    # 4. v50: Co-occurrence pairs: encje które MUSZĄ być blisko siebie
    cooc_pairs = pre_batch.get("_cooccurrence_pairs") or []
    if cooc_pairs:
        cooc_lines = []
        for pair in cooc_pairs[:8]:
            if isinstance(pair, dict):
                e1 = pair.get("entity1", pair.get("source", ""))
                e2 = pair.get("entity2", pair.get("target", ""))
                if e1 and e2:
                    cooc_lines.append(f'  • "{e1}" + "{e2}"  (w tym samym akapicie)')
            elif isinstance(pair, str) and "+" in pair:
                cooc_lines.append(f"  • {pair}  (w tym samym akapicie)")
        if cooc_lines:
            parts.append(
                "═══ WSPÓŁWYSTĘPOWANIE ENCJI (co-occurrence) ═══\n"
                "Następujące pary encji często pojawiają się RAZEM u konkurencji.\n"
                "Umieść je W TYM SAMYM AKAPICIE , bliskość buduje kontekst semantyczny:\n"
                + "\n".join(cooc_lines)
            )
    
    # 5. v50: First paragraph entities: encje z pierwszego akapitu top10
    first_para_ents = pre_batch.get("_first_paragraph_entities") or []
    if first_para_ents:
        fp_names = []
        for ent in first_para_ents[:6]:
            name = ent.get("entity", ent.get("text", ent)) if isinstance(ent, dict) else str(ent)
            if name:
                fp_names.append(f'"{name}"')
        if fp_names:
            parts.append(
                "PIERWSZY AKAPIT, encje tematyczne:\n"
                f"Wprowadź w pierwszym akapicie: {', '.join(fp_names)}.\n"
                "⚠️ To POJĘCIA do opisania, NIE źródła do cytowania. Nie pisz '[encja] podaje/potwierdza...'."
            )
    
    # 6. v50: H2 entities: encje tematyczne do rozmieszczenia w H2
    h2_ents = pre_batch.get("_h2_entities") or []
    if h2_ents:
        h2_names = []
        for ent in h2_ents[:8]:
            name = ent.get("entity", ent.get("text", ent)) if isinstance(ent, dict) else str(ent)
            if name:
                h2_names.append(f'"{name}"')
        if h2_names:
            parts.append(
                "ENCJE TEMATYCZNE W H2:\n"
                f"Rozłóż w tekście: {', '.join(h2_names)}.\n"
                "⚠️ To POJĘCIA do opisania, NIE źródła. Nie pisz '[encja] podaje...'."
            )

    # 7. EAV triples: encja → atrybut → wartość
    # Mówią modelowi CO NAPISAĆ o każdej encji — konkretny fakt, nie tylko nazwa
    eav_triples = pre_batch.get("_eav_triples") or []
    if eav_triples:
        eav_lines = ["═══ CECHY ENCJI — Entity Attribute Value (NAPISZ TE FAKTY) ═══",
                     "Dla każdej poniższej encji MUSISZ wyrazić podany fakt w tekście.",
                     "Nie kopiuj dosłownie — zbuduj naturalne zdanie zawierające tę relację.",
                     ""]
        primary_eav = [e for e in eav_triples if e.get("is_primary")]
        secondary_eav = [e for e in eav_triples if not e.get("is_primary")]
        if primary_eav:
            e = primary_eav[0]
            eav_lines.append(f'🎯 GŁÓWNA: "{e["entity"]}" → {e["attribute"]} → {e["value"]}')
        for e in secondary_eav[:10]:
            eav_lines.append(f'   • "{e["entity"]}" ({e.get("type","")}) → {e["attribute"]} → {e["value"]}')
        eav_lines.append("")
        eav_lines.append("✅ Przykład zamiany EAV na zdanie:")
        eav_lines.append('   EAV: "kodeks karny → penalizuje → jazdę po alkoholu art. 178a"')
        eav_lines.append('   ZDANIE: "Art. 178a Kodeksu karnego penalizuje prowadzenie pojazdu w stanie"')
        eav_lines.append('          "nietrzeźwości — przewiduje karę do 3 lat pozbawienia wolności."')
        parts.append("\n".join(eav_lines))

    # 8. SVO triples: podmiot → relacja → obiekt  
    # Gotowe fakty do wbudowania w tekst — rdzeń knowledge graph artykułu
    svo_triples = pre_batch.get("_svo_triples") or []
    if svo_triples:
        svo_lines = ["═══ TRÓJKI SEMANTYCZNE SVO — fakty OBOWIĄZKOWE w artykule ═══",
                     "Każda trójka to fakt który MUSI znaleźć się gdzieś w artykule.",
                     "Możesz rozłożyć je na różne sekcje — ważne żeby były obecne.",
                     ""]
        for i, t in enumerate(svo_triples[:12], 1):
            ctx = f' [{t["context"]}]' if t.get("context") else ""
            svo_lines.append(f'  {i}. {t["subject"]} → {t["verb"]} → {t["object"]}{ctx}')
        svo_lines.append("")
        svo_lines.append("Google Knowledge Graph indeksuje te relacje. Im więcej z nich pojawi")
        svo_lines.append("się jako wyraźne zdania (nie wtrącenia), tym wyższy topic authority.")
        parts.append("\n".join(svo_lines))

    return "\n\n".join(parts) if parts else ""


# _fmt_entities REMOVED v45.4.1 → v50 cleanup: function deleted.
# gpt_instructions_v39 already contains curated "🧠 ENCJE:" section
# (max 3/batch, importance≥0.7, with HOW hints). Our version duplicated it
# with dirtier, unfiltered data from S1.

# _fmt_ngrams REMOVED v45.4.1 → v50 cleanup: function deleted.
# Raw statistical n-grams from competitor pages often contain CSS/JS artifacts
# ("button button", "block embed"). Custom GPT produces better text without them.


def _fmt_serp_enrichment(pre_batch):
    serp = pre_batch.get("serp_enrichment") or {}
    enhanced = pre_batch.get("enhanced") or {}

    paa = (serp.get("paa_for_batch") or enhanced.get("paa_from_serp") or [])
    lsi = (serp.get("lsi_keywords") or [])

    if not paa and not lsi:
        return ""

    parts = ["═══ WZBOGACENIE Z SERP ═══"]

    if paa:
        parts.append("Pytania które ludzie zadają w Google (PAA), odpowiedz na 1-2 w tekście:")
        for q in paa[:5]:
            q_text = q.get("question", q) if isinstance(q, dict) else q
            if q_text:
                parts.append(f'  ❓ {q_text}')

    if lsi:
        lsi_names = [l.get("keyword", l) if isinstance(l, dict) else l for l in lsi[:8]]
        parts.append(f'\nFrazy LSI (bliskoznaczne, wpleć naturalnie): {", ".join(lsi_names)}')

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_continuation(pre_batch):
    continuation = pre_batch.get("continuation_v39") or {}
    enhanced = pre_batch.get("enhanced") or {}
    cont_ctx = enhanced.get("continuation_context") or {}

    last_h2 = cont_ctx.get("last_h2") or continuation.get("last_h2", "")
    last_ending = cont_ctx.get("last_paragraph_ending") or continuation.get("last_paragraph_ending", "")
    last_topic = cont_ctx.get("last_topic") or continuation.get("last_topic", "")
    transition_hint = continuation.get("transition_hint", "")

    if not last_h2 and not last_ending:
        return ""

    parts = ["═══ KONTYNUACJA ═══",
             "Poprzedni batch zakończył się na:"]

    if last_h2:
        parts.append(f'  Ostatni H2: "{last_h2}"')
    if last_ending:
        ending_preview = last_ending[:150] + ("..." if len(last_ending) > 150 else "")
        parts.append(f'  Ostatnie zdanie: "{ending_preview}"')
    if last_topic:
        parts.append(f'  Temat: {last_topic}')

    parts.append("\nZacznij PŁYNNIE: nawiąż do poprzedniego wątku, ale nie powtarzaj zakończenia.")
    if transition_hint:
        parts.append(f'Sugerowane przejście: {transition_hint}')

    return "\n".join(parts)


def _fmt_article_memory(article_memory):
    if not article_memory:
        return ""

    parts = ["═══ PAMIĘĆ ARTYKUŁU (KRYTYCZNE, nie powtarzaj!) ═══"]

    if isinstance(article_memory, dict):
        topics = article_memory.get("topics_covered") or article_memory.get("covered_topics") or []
        if topics:
            parts.append("Sekcje już napisane:")
            for t in topics[:10]:
                if isinstance(t, str):
                    parts.append(f'  ✓ {t}')
                elif isinstance(t, dict):
                    parts.append(f'  ✓ {t.get("topic", t.get("h2", ""))}')

        facts = article_memory.get("key_facts_used") or article_memory.get("facts", [])
        # v50.5 FIX 30: Also extract key_points and avoid_repetition from AI memory
        key_points = article_memory.get("key_points") or []
        avoid_rep = article_memory.get("avoid_repetition") or []
        
        all_facts = list(facts) + list(key_points)
        if all_facts:
            parts.append("\nFakty/definicje już podane (NIE POWTARZAJ, odwołuj się: 'wspomniany wcześniej'):")
            for f in all_facts[:12]:
                parts.append(f'  • {f}' if isinstance(f, str) else f'  • {json.dumps(f, ensure_ascii=False)[:100]}')

        if avoid_rep:
            parts.append("\n⛔ TE ZDANIA I FRAZY BYŁY JUŻ UŻYTE — NIE POWTARZAJ ICH DOSŁOWNIE:")
            parts.append("   (możesz użyć tego samego SENSU, ale innymi słowami)")
            for r in avoid_rep[:8]:
                parts.append(f'  ❌ ZAKAZ: "{r}"')

        phrases_used = article_memory.get("phrases_used") or {}
        if phrases_used:
            high_use = [(k, v) for k, v in phrases_used.items()
                        if isinstance(v, (int, float)) and v >= 3]
            if high_use:
                parts.append("\nFrazy już często użyte (ogranicz):")
                for name, count in high_use[:8]:
                    parts.append(f'  • "{name}" (już {count}×)')
        
        # v50.5 FIX 30: Add strong anti-repetition instruction
        if topics and len(topics) >= 2:
            parts.append(
                "\n⚠️ ZASADA ANTY-POWTÓRZEŃ: Jeśli pojęcie (np. prawo Ohma, definicja ampera) "
                "zostało ZDEFINIOWANE w poprzedniej sekcji, NIE definiuj go ponownie. "
                "Zamiast tego: użyj go w nowym kontekście lub odnieś się krótko: "
                "'zgodnie z omówionym wcześniej prawem Ohma'. "
                "Powtórzenie definicji = utrata punktów jakości."
            )
    elif isinstance(article_memory, str):
        parts.append(_word_trim(article_memory, 1500))

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_coverage_density(pre_batch):
    coverage = pre_batch.get("coverage") or {}
    density = pre_batch.get("density") or {}
    main_kw = pre_batch.get("main_keyword") or {}
    keyword_tracking = pre_batch.get("keyword_tracking") or {}

    if not coverage and not density and not main_kw:
        return ""

    parts = ["═══ STATUS POKRYCIA FRAZ ═══"]

    if main_kw:
        kw_name = main_kw.get("keyword", "") if isinstance(main_kw, dict) else str(main_kw)
        synonyms = main_kw.get("synonyms", []) if isinstance(main_kw, dict) else []
        if kw_name:
            parts.append(f'Hasło główne: "{kw_name}"')
        if synonyms:
            parts.append(f'Synonimy (używaj zamiennie): {", ".join(synonyms[:5])}')

    current_cov = coverage.get("current", coverage.get("current_coverage"))
    target_cov = coverage.get("target", coverage.get("target_coverage"))
    if current_cov is not None and target_cov is not None:
        parts.append(f'\nPokrycie fraz: {current_cov}% z docelowych {target_cov}%')

    missing = coverage.get("missing_phrases") or coverage.get("uncovered") or []
    if missing:
        parts.append("⚠️ BRAKUJĄCE FRAZY, wpleć w tym batchu:")
        for m in missing[:8]:
            name = m.get("keyword", m) if isinstance(m, dict) else m
            parts.append(f'  → "{name}"')

    if density:
        current_d = density.get("current")
        target_range = density.get("target_range") or []
        if current_d is not None:
            range_str = f'{target_range[0]}-{target_range[1]}%' if len(target_range) >= 2 else "1.5-2.5%"
            status = "✅ w normie" if target_range and len(target_range) >= 2 and target_range[0] <= current_d <= target_range[1] else "⚠️ do korekty"
            parts.append(f'\nGęstość fraz: {current_d}% (cel: {range_str}) {status}')

        overused_d = density.get("overused") or []
        if overused_d:
            over_names = ", ".join(f'"{o}"' if isinstance(o, str) else f'"{o.get("keyword", "")}"' for o in overused_d[:5])
            parts.append(f'Nadużywane: {over_names}, użyj synonimów')

    if keyword_tracking:
        total_kw = keyword_tracking.get("total_keywords", 0)
        covered_kw = keyword_tracking.get("covered", 0)
        if total_kw and covered_kw:
            parts.append(f'\nTracking: {covered_kw}/{total_kw} fraz pokrytych')

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_style(pre_batch):
    style = pre_batch.get("style_instructions") or pre_batch.get("style_instructions_v39") or {}

    if not style:
        return ""

    parts = ["═══ STYL ═══"]

    if isinstance(style, dict):
        tone = style.get("tone", "")
        if tone:
            parts.append(f'Ton: {tone}')

        para_len = style.get("paragraph_length", "")
        if para_len:
            parts.append(f'Długość akapitów: {para_len} słów')

        forbidden = style.get("forbidden_phrases") or style.get("avoid_phrases") or []
        if forbidden:
            parts.append(f'ZAKAZANE zwroty: {", ".join(f"{f}" for f in forbidden[:8])}')

        preferred = style.get("preferred_phrases") or style.get("use_phrases") or []
        if preferred:
            parts.append(f'Preferowane zwroty: {", ".join(preferred[:5])}')

        persona = style.get("persona", "")
        if persona:
            parts.append(f'Perspektywa: {persona}')
    elif isinstance(style, str):
        parts.append(_word_trim(style, 500))

    return "\n".join(parts) if len(parts) > 1 else ""


def _fmt_legal_medical(pre_batch):
    legal_ctx = pre_batch.get("legal_context") or {}
    medical_ctx = pre_batch.get("medical_context") or {}
    ymyl_enrich = pre_batch.get("_ymyl_enrichment") or {}
    ymyl_intensity = pre_batch.get("_ymyl_intensity", "full")

    parts = []

    # v50: For "light" YMYL: DON'T inject full legal/medical framework
    if ymyl_intensity == "light":
        light_note = pre_batch.get("_light_ymyl_note", "")
        if light_note:
            parts.append("═══ ASPEKT REGULACYJNY (peryferyjny, NIE główny temat!) ═══")
            parts.append(f"  {light_note}")
            parts.append("  ⚠️ OGRANICZENIE: Wspomnij o regulacjach MAX 1-2 razy w CAŁYM artykule.")
            parts.append("  NIE cytuj artykułów ustaw, NIE dodawaj sygnatur orzeczeń,")
            parts.append("  NIE dodawaj disclaimera o konsultacji z prawnikiem/lekarzem.")
            parts.append("  Artykuł jest EDUKACYJNY/TECHNICZNY, nie prawniczy/medyczny.")
        return "\n".join(parts) if parts else ""

    if legal_ctx and legal_ctx.get("active"):
        parts.append("═══ KONTEKST PRAWNY (YMYL) ═══")
        parts.append("Ten artykuł dotyczy tematyki prawnej. MUSISZ:")
        parts.append("  1. Cytować realne przepisy i orzeczenia — ALE TYLKO te pasujące do gałęzi prawa artykułu")
        parts.append("  2. Dodać disclaimer o konsultacji z prawnikiem")
        parts.append("  3. NIE wymyślać sygnatur ani dat orzeczeń")
        parts.append("")
        parts.append("🚫 BŁĘDY KRYTYCZNE — BEZWZGLĘDNY ZAKAZ:")
        parts.append("  • JEDNOSTKI: mg/100 ml → BŁĄD. Używaj: promile (‰) lub mg/dm³")
        parts.append("  • KARA 178a §1: do 2 lat → BŁĄD. Prawidłowo: do 3 lat (nowelizacja 2023)")
        parts.append("  • RECYDYWA: nie definiuj terminem '2 lat' — brak takiego wymogu")
        parts.append("  • SYGNATURA I C / II C w kontekście konfiskaty → BŁĄD: to sprawa cywilna")
        parts.append("  • PLACEHOLDER 'odpowiednich przepisów' → zawsze podaj konkretny art.")
        
        # Inject Wikipedia articles if available
        wiki_arts = pre_batch.get("legal_wiki_articles") or []
        if wiki_arts:
            parts.append("")
            parts.append("WIKIPEDIA — TREŚĆ PRZEPISÓW (możesz cytować jako źródło uzupełniające):")
            for w in wiki_arts[:4]:
                if w.get("found"):
                    parts.append(f"  [{w['article_ref']}] {w['title']}:")
                    parts.append(f"  {w['extract'][:300]}")
                    parts.append(f"  Źródło: {w['url']}")
                    parts.append("")
        parts.append("")
        parts.append("⚠️ WERYFIKACJA ORZECZEŃ — OBOWIĄZKOWA:")
        parts.append("  Sygnatura zdradza typ sprawy:")
        parts.append("  • II K, III K, AKa, AKo, AKz = KARNA — pasuje do art. KK, KW")
        parts.append("  • I C, II C, ACa, ACo = CYWILNA — pasuje do art. KC, KRO")
        parts.append("  • I P, II P, Pa = PRACY — pasuje do KP")
        parts.append("  ❌ NIE cytuj wyroku cywilnego (I C, II C) w artykule o prawie KARNYM")
        parts.append("  ❌ NIE cytuj wyroku karnego (II K) w artykule o prawie CYWILNYM")
        parts.append("  Jeśli żaden z podanych wyroków nie pasuje do gałęzi prawa — pomiń cytowania,")
        parts.append("  napisz artykuł bez sygnatur. Lepiej brak cytatu niż błędny.")
        
        # v47.2: Claude's enrichment: specific articles and concepts
        legal_enrich = ymyl_enrich.get("legal", {})
        if legal_enrich.get("articles"):
            parts.append("")
            parts.append("PODSTAWA PRAWNA (kluczowe przepisy):")
            for art in legal_enrich["articles"][:5]:
                parts.append(f"  • {art}")
        if legal_enrich.get("acts"):
            parts.append(f"  Ustawy: {', '.join(legal_enrich['acts'][:4])}")
        if legal_enrich.get("key_concepts"):
            parts.append(f"  Kluczowe pojęcia: {', '.join(legal_enrich['key_concepts'][:6])}")
        
        parts.append("")
        parts.append("FORMATY CYTOWAŃ PRAWNYCH:")
        parts.append('  • Przepisy: "art. 13 § 1 k.c.", "art. 58 § 2 k.r.o."')
        parts.append('  • Wyroki: "wyrok SN z 12.03.2021, III CZP 45/19"')
        parts.append('  • Dziennik Ustaw: "Dz.U. 2023 poz. 1234"')
        parts.append('  Causal legal: "niedopełnienie obowiązku skutkuje...", "brak zgłoszenia prowadzi do..."')

        instruction = legal_ctx.get("legal_instruction", "")
        if instruction:
            parts.append(f'\n{instruction[:600]}')

        judgments = legal_ctx.get("top_judgments") or []
        if judgments:
            parts.append("\nOrzeczenia do zacytowania:")
            for j in judgments[:3]:
                if isinstance(j, dict):
                    sig = j.get("signature", j.get("caseNumber", ""))
                    court = j.get("court", j.get("courtName", ""))
                    date = j.get("date", j.get("judgmentDate", ""))
                    matched = j.get("matched_article", "")
                    line = f'  • {sig}, {court} ({date})'
                    if matched:
                        line += f' [dot. {matched}]'
                    parts.append(line)

        citation_hint = legal_ctx.get("citation_hint", "")
        if citation_hint:
            parts.append(f'\n{citation_hint}')

    if medical_ctx and medical_ctx.get("active"):
        if parts:
            parts.append("")
        parts.append("═══ KONTEKST MEDYCZNY (YMYL) ═══")
        parts.append("Ten artykuł dotyczy tematyki zdrowotnej. MUSISZ:")
        parts.append("  1. Cytować źródła naukowe (podane niżej lub ogólne: 'badania wskazują', 'według wytycznych')")
        parts.append("  2. NIE wymyślać statystyk ani nazw badań")
        parts.append("  3. W OSTATNIM batchu: dodać disclaimer 'Artykuł ma charakter informacyjny i nie zastępuje konsultacji lekarskiej.'")
        parts.append("  4. Powołać się na min. 1 instytucję (np. WHO, NFZ, PTOiAu, MZ, Cochrane) per batch")
        parts.append("  5. Użyć min. 1 sformułowania opartego na dowodach per batch: 'badania wskazują...', 'według meta-analizy...'")
        parts.append("  WAŻNE: Artykuł bez źródeł medycznych = YMYL score 0/100 = odrzucenie.")
        
        # v47.2: Claude's enrichment: specialization, evidence guidelines
        med_enrich = ymyl_enrich.get("medical", {})
        if med_enrich.get("specialization"):
            parts.append(f"\n  Specjalizacja: {med_enrich['specialization']}")
        if med_enrich.get("condition"):
            cond = med_enrich["condition"]
            latin = med_enrich.get("condition_latin", "")
            icd = med_enrich.get("icd10", "")
            parts.append(f"  Choroba/stan: {cond}" + (f" ({latin})" if latin else "") + (f" [ICD-10: {icd}]" if icd else ""))
        if med_enrich.get("key_drugs"):
            parts.append(f"  Kluczowe leki: {', '.join(med_enrich['key_drugs'][:5])}")
        if med_enrich.get("evidence_note"):
            parts.append(f"\n  ⚠️ WYTYCZNE: {med_enrich['evidence_note']}")
        
        parts.append("")
        parts.append("FORMATY CYTOWAŃ MEDYCZNYCH:")
        parts.append('  • "Smith i wsp. (2023)", "Kowalski et al. (2024)"')
        parts.append('  • "PMID:12345678", "DOI:10.1000/xyz"')
        parts.append("")
        parts.append("HIERARCHIA DOWODÓW (cytuj najwyższy dostępny):")
        parts.append("  1. Meta-analiza / Przegląd systematyczny (najsilniejszy)")
        parts.append("  2. RCT (badanie randomizowane)")
        parts.append("  3. Badanie kohortowe")
        parts.append("  4. Opis przypadku")
        parts.append("  5. Opinia eksperta (najsłabszy)")
        parts.append('  Causal medical: "nieleczone prowadzi do...", "brak terapii skutkuje..."')

        instruction = medical_ctx.get("medical_instruction", "")
        if instruction:
            parts.append(f'\n{instruction[:600]}')

        publications = medical_ctx.get("top_publications") or []
        if publications:
            parts.append("\nPublikacje do zacytowania:")
            for p in publications[:5]:
                if isinstance(p, dict):
                    title = p.get("title", "")[:80]
                    authors = p.get("authors", "")[:40]
                    year = p.get("year", "")
                    pmid = p.get("pmid", "")
                    parts.append(f'  • {authors} ({year}): "{title}" PMID:{pmid}')

    return "\n".join(parts) if parts else ""


def _fmt_experience_markers(pre_batch):
    enhanced = pre_batch.get("enhanced") or {}
    markers = enhanced.get("experience_markers") or []

    if not markers:
        return ""

    parts = ["═══ SYGNAŁY DOŚWIADCZENIA (E-E-A-T) ═══",
             "Wpleć min 1 sygnał, że autor MA doświadczenie z tematem:"]

    for m in markers[:5]:
        if isinstance(m, str):
            parts.append(f'  • {m}')
        elif isinstance(m, dict):
            parts.append(f'  • {m.get("marker", m.get("text", ""))}')

    return "\n".join(parts)


def _fmt_causal_context(pre_batch):
    enhanced = pre_batch.get("enhanced") or {}
    causal = enhanced.get("causal_context", "")
    info_gain = enhanced.get("information_gain", "")

    parts = []

    if causal:
        parts.append("═══ KONTEKST PRZYCZYNOWO-SKUTKOWY ═══")
        parts.append(_word_trim(causal, 500))

    if info_gain:
        if parts:
            parts.append("")
        parts.append("═══ INFORMATION GAIN (przewaga nad konkurencją) ═══")
        parts.append(_word_trim(info_gain, 500))

    return "\n".join(parts) if parts else ""


def _fmt_depth_signals(pre_batch):
    """Depth signals: inject when previous batch scored low on depth
    or always for FULL YMYL content.
    
    v50: Only force for full YMYL intensity, not light.
    Based on 10 depth signals from GPT prompt with weights.
    """
    last_depth = pre_batch.get("_last_depth_score")
    is_ymyl = pre_batch.get("_is_ymyl", False)
    ymyl_intensity = pre_batch.get("_ymyl_intensity", "none")
    is_full_ymyl = is_ymyl and ymyl_intensity == "full"
    
    # Only force depth for FULL YMYL, not light
    threshold = 40 if is_full_ymyl else 30
    if last_depth is not None and last_depth >= threshold and not is_full_ymyl:
        return ""
    
    # If no depth data at all and not full YMYL, skip
    if last_depth is None and not is_full_ymyl:
        return ""
    
    parts = ["═══ SYGNAŁY GŁĘBOKOŚCI (dodaj od najwyższej wagi) ═══"]
    
    if last_depth is not None:
        parts.append(f"⚠️ Ostatni batch: depth {last_depth}/100 (próg: {threshold}). Dodaj więcej konkretów!")
    
    parts.append("")
    # v50: Legal references only for FULL YMYL
    if is_full_ymyl:
        parts.append("WAGA 2.5: referencje prawne (art. k.c., wyroki SN, Dz.U.) + naukowe (PMID, DOI, badania)")
    parts.append('WAGA 2.0: konkretne liczby (kwoty PLN, %, okresy, NIE "około")')
    parts.append('WAGA 1.8: nazwane instytucje (konkretny sąd/urząd, NIE "właściwy sąd") + praktyczne porady (w praktyce, częsty błąd)')
    parts.append("WAGA 1.5: wyjaśnienia przyczynowe (ponieważ, w wyniku) + wyjątki (z wyjątkiem, chyba że) + konkretne daty")
    parts.append("WAGA 1.2: porównania (w odróżnieniu od) | WAGA 1.0: kroki procedur (najpierw/następnie)")
    
    return "\n".join(parts)


def _fmt_natural_polish(pre_batch):
    """v50: Natural Polish writing instructions: fleksja, spacing, anti-stuffing.

    Based on natural_polish_instructions.py (master-seo-api-main).
    Inlined here because prompt_builder runs in Brajn, not master.
    
    Prevents keyword stuffing by teaching Claude that:
    1. Polish inflected forms count as the same keyword
    2. Minimum spacing between repetitions is required
    3. Max 2 uses of same phrase per paragraph
    """
    # Get keywords from pre_batch
    keywords_info = pre_batch.get("keywords") or {}
    must_kw = keywords_info.get("basic_must_use") or []
    ext_kw = keywords_info.get("extended_this_batch") or []

    all_kw = []
    for kw in must_kw + ext_kw:
        if isinstance(kw, dict):
            name = kw.get("keyword", "")
            kw_type = kw.get("type", "BASIC").upper()
        elif isinstance(kw, str):
            name = kw
            kw_type = "BASIC"
        else:
            continue
        if name:
            all_kw.append((name, kw_type))

    if not all_kw:
        return ""

    # Spacing rules
    SPACING = {"MAIN": 60, "BASIC": 80, "EXTENDED": 120}

    parts = ["═══ NATURALNY POLSKI, ANTY-STUFFING ═══"]

    parts.append(
        "🔄 FLEKSJA: Odmiany frazy liczą się jako jedno użycie!\n"
        '   "zespół turnera" = "zespołu turnera" = "zespołem turnera"\n'
        "   Pisz naturalnie, używaj różnych przypadków gramatycznych.\n"
        "   NIE MUSISZ powtarzać frazy w mianowniku. System zaliczy każdą odmianę."
    )

    spacing_lines = []
    for name, kw_type in all_kw[:8]:
        spacing = SPACING.get(kw_type, 80)
        spacing_lines.append(f'  • "{name}" ({kw_type}): min {spacing} słów między powtórzeniami')
    if spacing_lines:
        parts.append("📏 ODSTĘPY MIĘDZY POWTÓRZENIAMI:\n" + "\n".join(spacing_lines))

    parts.append(
        "⚠️ ZASADY:\n"
        "  • Max 2× ta sama fraza w jednym akapicie\n"
        "  • Rozkładaj frazy RÓWNOMIERNIE w tekście (nie grupuj na początku/końcu)\n"
        "  • Zamiast powtórzenia użyj: synonimu, zaimka, opisu ('ta choroba', 'omawiany zespół')\n"
        "  • Podmiot → dopełnienie → synonim → kolejny akapit → ponownie fraza"
    )

    return "\n".join(parts)


def _fmt_phrase_hierarchy(pre_batch):
    """Format phrase hierarchy: roots, extensions, strategy.
    
    Data sources (checked in order):
    1. pre_batch["enhanced"]["phrase_hierarchy"]: from enhanced_pre_batch.py
    2. pre_batch["_phrase_hierarchy"]: injected by app.py from /phrase_hierarchy endpoint
    """
    hier = (pre_batch.get("enhanced") or {}).get("phrase_hierarchy") or pre_batch.get("_phrase_hierarchy") or {}
    if not hier:
        return ""

    parts = ["═══ HIERARCHIA FRAZ ═══"]

    strategies = hier.get("strategies") or {}

    # 1. Extensions sufficient: don't repeat root standalone
    ext_suff = strategies.get("extensions_sufficient") or {}
    ext_roots = ext_suff.get("roots") or []
    if ext_roots:
        parts.append("RDZENIE POKRYTE ROZSZERZENIAMI (NIE powtarzaj samodzielnie!):")
        for root_info in ext_roots[:8]:
            if isinstance(root_info, dict):
                root = root_info.get("root", root_info.get("keyword", ""))
                extensions = root_info.get("extensions", [])
                ext_list = ", ".join(f'"{e}"' if isinstance(e, str) else f'"{e.get("keyword", "")}"' for e in extensions[:5])
                parts.append(f'  • "{root}" → używaj rozszerzeń: {ext_list}')
            elif isinstance(root_info, str):
                parts.append(f'  • "{root_info}" → używaj rozszerzeń zamiast rdzenia')

    # 2. Mixed: some standalone + extensions
    mixed = strategies.get("mixed") or {}
    mixed_roots = mixed.get("roots") or []
    if mixed_roots:
        parts.append("RDZENIE MIESZANE (kilka samodzielnych użyć + rozszerzenia):")
        for root_info in mixed_roots[:8]:
            if isinstance(root_info, dict):
                root = root_info.get("root", root_info.get("keyword", ""))
                standalone = root_info.get("standalone_uses", "1-2")
                extensions = root_info.get("extensions", [])
                ext_list = ", ".join(f'"{e}"' if isinstance(e, str) else f'"{e.get("keyword", "")}"' for e in extensions[:5])
                parts.append(f'  • "{root}" → {standalone}× samodzielnie + rozszerzenia: {ext_list}')
            elif isinstance(root_info, str):
                parts.append(f'  • "{root_info}" → kilka samodzielnie + rozszerzenia')

    # 3. Need standalone: extensions insufficient
    standalone = strategies.get("need_standalone") or {}
    standalone_roots = standalone.get("roots") or []
    if standalone_roots:
        parts.append("RDZENIE WYMAGAJĄCE SAMODZIELNYCH UŻYĆ:")
        for root_info in standalone_roots[:8]:
            if isinstance(root_info, dict):
                root = root_info.get("root", root_info.get("keyword", ""))
                target = root_info.get("remaining", root_info.get("target", "?"))
                parts.append(f'  • "{root}" → użyj samodzielnie jeszcze ~{target}×')
            elif isinstance(root_info, str):
                parts.append(f'  • "{root_info}" → użyj samodzielnie')

    # 4. Entity phrases (if available)
    entity_phrases = hier.get("entity_phrases") or []
    if entity_phrases:
        ep_list = ", ".join(f'"{e}"' if isinstance(e, str) else f'"{e.get("keyword", "")}"' for e in entity_phrases[:6])
        parts.append(f"FRAZY ENCYJNE (wpleć naturalnie): {ep_list}")

    # 5. Triplet phrases (if available)
    triplet_phrases = hier.get("triplet_phrases") or []
    if triplet_phrases:
        tp_list = ", ".join(f'"{t}"' if isinstance(t, str) else f'"{t.get("keyword", "")}"' for t in triplet_phrases[:6])
        parts.append(f"FRAZY TRIPLETOWE (relacje do wplecenia): {tp_list}")

    if len(parts) <= 1:
        return ""

    return "\n".join(parts)


def _fmt_h2_remaining(pre_batch):
    h2_remaining = pre_batch.get("h2_remaining") or []
    if not h2_remaining:
        return ""

    h2_list = ", ".join(f'"{h}"' for h in h2_remaining[:6])
    return f"═══ PLAN ═══\nPozostałe sekcje H2 w artykule: {h2_list}\nNie zachodź na ich tematy. Zostaną pokryte później."


def _fmt_output_format(h2, batch_type):
    if batch_type in ("INTRO", "intro"):
        return f"""═══ FORMAT ODPOWIEDZI ═══
Pisz TYLKO treść wstępu. NIE zaczynaj od "h2:". Wstęp nie ma nagłówka.
80-150 słów. Frazę główną wpleć w PIERWSZE zdanie.
NIE dodawaj komentarzy, wyjaśnień. TYLKO treść wstępu."""
    
    return f"""═══ FORMAT ODPOWIEDZI ═══
Pisz TYLKO treść tego batcha. Zaczynaj dokładnie od:

h2: {h2}

Potem: akapity tekstu (40-150 słów każdy), opcjonalnie h3: [podsekcja].
NIE dodawaj komentarzy, wyjaśnień, podsumowań. TYLKO treść artykułu."""


# ════════════════════════════════════════════════════════════
# FAQ PROMPT BUILDER
# ════════════════════════════════════════════════════════════

def build_faq_system_prompt(pre_batch=None):
    """System prompt for FAQ generation."""
    base = (
        "Jesteś doświadczonym polskim copywriterem SEO. "
        "Piszesz sekcję FAQ: zwięzłe, konkretne odpowiedzi na pytania użytkowników. "
        "Każda odpowiedź ma szansę trafić do Google Featured Snippet. Pisz bezpośrednio i merytorycznie."
    )

    gpt_instructions = ""
    if pre_batch:
        gpt_instructions = pre_batch.get("gpt_instructions_v39", "")

    if gpt_instructions:
        return base + "\n\n" + gpt_instructions
    return base


def build_faq_user_prompt(paa_data, pre_batch=None):
    """User prompt for FAQ generation."""
    # Normalize: if paa_data is a list (raw PAA questions), wrap it
    if isinstance(paa_data, list):
        paa_data = {"serp_paa": paa_data}
    elif not isinstance(paa_data, dict):
        paa_data = {}
    paa_questions = paa_data.get("serp_paa") or []
    unused = paa_data.get("unused_keywords") or {}
    avoid = paa_data.get("avoid_in_faq") or []
    if isinstance(avoid, dict):
        avoid = avoid.get("topics") or []
    elif not isinstance(avoid, list):
        avoid = []
    instructions_raw = paa_data.get("instructions", "")
    if isinstance(instructions_raw, dict):
        parts = []
        for k, v in instructions_raw.items():
            if isinstance(v, str):
                parts.append(f"• {v}")
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, str):
                        parts.append(f"• {sk}: {sv}")
        instructions = "\n".join(parts)
    elif isinstance(instructions_raw, str):
        instructions = instructions_raw
    else:
        instructions = ""

    enhanced_paa = []
    if pre_batch:
        enhanced = pre_batch.get("enhanced") or {}
        if not isinstance(enhanced, dict):
            enhanced = {}
        enhanced_paa = enhanced.get("paa_from_serp") or []
        if not isinstance(enhanced_paa, list):
            enhanced_paa = []

    keyword_limits = {}
    if pre_batch:
        keyword_limits = pre_batch.get("keyword_limits") or {}
        if not isinstance(keyword_limits, dict):
            keyword_limits = {}
    stop_raw = keyword_limits.get("stop_keywords") or []
    stop_names = [s.get("keyword", s) if isinstance(s, dict) else s for s in stop_raw]

    style = {}
    if pre_batch:
        style = pre_batch.get("style_instructions") or {}

    sections = []

    sections.append("""═══ SEKCJA FAQ ═══
Napisz sekcję FAQ. Zaczynaj DOKŁADNIE od:
h2: Najczęściej zadawane pytania""")

    all_paa = list(dict.fromkeys(paa_questions + enhanced_paa))
    if all_paa:
        sections.append("Pytania z Google (People Also Ask), to NAPRAWDĘ pytają użytkownicy:")
        for i, q in enumerate(all_paa[:8], 1):
            q_text = q.get("question", q) if isinstance(q, dict) else q
            if q_text and q_text.strip():
                sections.append(f'  {i}. {q_text}')
        sections.append("Wybierz 4-6 najlepszych. Możesz przeformułować, ale zachowaj sens.")

    if unused:
        if isinstance(unused, dict):
            unused_list = []
            for cat, items in unused.items():
                if isinstance(items, list):
                    unused_list.extend(items[:5])
                elif isinstance(items, str):
                    unused_list.append(items)
            if unused_list:
                names = ", ".join(f'"{u}"' if isinstance(u, str) else f'"{u.get("keyword", "")}"' for u in unused_list[:8])
                sections.append(f'\nFrazy jeszcze nieużyte, wpleć w odpowiedzi: {names}')
        elif isinstance(unused, list):
            names = ", ".join(f'"{u}"' for u in unused[:8])
            sections.append(f'\nFrazy jeszcze nieużyte, wpleć w odpowiedzi: {names}')

    if avoid:
        topics = ", ".join(f'"{a}"' if isinstance(a, str) else f'"{a.get("topic", "")}"' for a in avoid[:8])
        sections.append(f'\nNIE powtarzaj tematów już pokrytych w artykule: {topics}')

    if stop_names:
        sections.append(f'\n🛑 STOP, NIE UŻYWAJ: {", ".join(f"{s}" for s in stop_names[:5])}')

    if style:
        forbidden = style.get("forbidden_phrases") or []
        if forbidden:
            sections.append(f'ZAKAZANE zwroty: {", ".join(forbidden[:5])}')

    if pre_batch and pre_batch.get("article_memory"):
        mem = pre_batch["article_memory"]
        if isinstance(mem, dict):
            topics = mem.get("topics_covered") or []
            if topics:
                topic_names = [t if isinstance(t, str) else t.get("topic", "") for t in topics[:6]]
                sections.append(f'\nTematy z artykułu (nie powtarzaj): {", ".join(topic_names)}')

    if instructions:
        sections.append(f'\n{instructions}')

    sections.append("""
═══ FORMAT ═══
h2: Najczęściej zadawane pytania

h3: [Pytanie, 5-10 słów, zaczynaj od Jak/Czy/Co/Dlaczego/Ile]
[Odpowiedź 60-120 słów]
→ Zdanie 1: BEZPOŚREDNIA odpowiedź
→ Zdanie 2-3: rozwinięcie z konkretem
→ Zdanie 4: praktyczna wskazówka lub wyjątek

Napisz 4-6 pytań. Pisz TYLKO treść, bez komentarzy.""")

    return "\n\n".join(sections)


# ════════════════════════════════════════════════════════════
# H2 PLAN PROMPT BUILDER
# ════════════════════════════════════════════════════════════

def build_h2_plan_system_prompt():
    """System prompt for H2 plan generation."""
    return (
        "Jesteś ekspertem SEO z 10-letnim doświadczeniem w planowaniu architektury treści. "
        "Tworzysz logiczne, wyczerpujące struktury nagłówków H2, które pokrywają temat kompleksowo "
        "i dają przewagę nad konkurencją dzięki pokryciu luk treściowych."
    )


def build_h2_plan_user_prompt(main_keyword, mode, s1_data, all_user_phrases, user_h2_hints=None):
    """Build readable H2 plan prompt from S1 analysis data."""
    s1_data = s1_data or {}
    competitor_h2 = s1_data.get("competitor_h2_patterns") or []
    suggested_h2s = (s1_data.get("content_gaps") or {}).get("suggested_new_h2s", [])
    content_gaps = s1_data.get("content_gaps") or {}
    causal_triplets = s1_data.get("causal_triplets") or {}
    paa = s1_data.get("paa") or s1_data.get("paa_questions") or []
    # v52.0: Related searches - Google sugeruje te pytania/frazy użytkownikom
    serp_analysis = s1_data.get("serp_analysis") or {}
    related_searches = (s1_data.get("related_searches")
                        or serp_analysis.get("related_searches") or [])

    sections = []

    mode_desc = "standard = pełny artykuł" if mode == "standard" else "fast = krótki artykuł, max 3 sekcje"
    sections.append(f"""HASŁO GŁÓWNE: {main_keyword}
TRYB: {mode} ({mode_desc})""")

    if competitor_h2:
        # Sort by count descending if available
        def _h2_count(h):
            if isinstance(h, dict):
                return h.get("count", h.get("sources", 0))
            return 0
        sorted_h2 = sorted(competitor_h2[:30], key=_h2_count, reverse=True)
        total_sources = max((_h2_count(sorted_h2[0]) for _ in [1]), default=1) or 1

        lines = ["═══ WZORCE H2 KONKURENCJI — posortowane po popularności ═══",
                 "Liczba przy H2 = ilu konkurentów używa tego tematu.",
                 "H2 z wysoką liczbą = MUST HAVE w Twoim artykule (użytkownicy tego szukają)."]
        for i, h in enumerate(sorted_h2[:20], 1):
            if isinstance(h, dict):
                pattern = h.get("text", h.get("pattern", h.get("h2", str(h))))
                count = _h2_count(h)
                bar = "█" * min(count, 8)
                lines.append(f"  {i:2}. [{bar:<8}] {count}× — {pattern}")
            elif isinstance(h, str):
                lines.append(f"  {i:2}. {h}")
        sections.append("\n".join(lines))

    if suggested_h2s:
        lines = ["═══ SUGEROWANE NOWE H2 (luki, tego NIKT z konkurencji nie pokrywa) ═══"]
        for h in suggested_h2s[:10]:
            h_text = h if isinstance(h, str) else h.get("h2", h.get("title", str(h)))
            lines.append(f"  • {h_text}")
        sections.append("\n".join(lines))

    # Content gaps: ordered by priority (GPT prompt: PAA_UNANSWERED > DEPTH_MISSING > SUBTOPIC_MISSING)
    gap_priority_map = {
        "paa_unanswered": ("🔴 HIGH", "PAA bez odpowiedzi"),
        "depth_missing": ("🟡 MED-HIGH", "Brak głębi"),
        "subtopic_missing": ("🟢 MED", "Brakujący podtemat"),
        "gaps": ("", "Luka"),
    }
    all_gaps = []
    for key in ("paa_unanswered", "depth_missing", "subtopic_missing", "gaps"):
        priority, label = gap_priority_map.get(key, ("", ""))
        items = content_gaps.get(key) or []
        for item in items[:5]:
            gap_text = item if isinstance(item, str) else item.get("gap", item.get("topic", str(item)))
            if gap_text and gap_text not in [g[0] for g in all_gaps]:
                all_gaps.append((gap_text, priority, label))
    if all_gaps:
        lines = ["═══ LUKI TREŚCIOWE (tematy do pokrycia, priorytet od najwyższego) ═══"]
        for gap_text, priority, label in all_gaps[:10]:
            prefix = f"[{priority}] " if priority else ""
            lines.append(f"  • {prefix}{gap_text}")
        sections.append("\n".join(lines))

    if paa:
        lines = ["═══ PYTANIA PAA (People Also Ask z Google) ═══"]
        for q in paa[:8]:
            q_text = q.get("question", q) if isinstance(q, dict) else q
            if q_text:
                lines.append(f"  ❓ {q_text}")
        sections.append("\n".join(lines))

    # v52.0: Related searches - Google podpowiada te frazy po wpisaniu main_keyword.
    # Zawierają intencje których często BRAK w H2 konkurencji (np. "warunkowe umorzenie",
    # "dożywotni zakaz", "organizmie wynosi") - ważny signal dla tematycznego pokrycia H2.
    if related_searches:
        rs_texts = []
        for rs in related_searches[:12]:
            rs_t = rs if isinstance(rs, str) else (rs.get("query", "") or rs.get("text", ""))
            if rs_t:
                rs_texts.append(rs_t)
        if rs_texts:
            lines = ["═══ RELATED SEARCHES (Google podpowiada po main_keyword) ═══",
                     "Użyj tych fraz jako wskazówek tematycznych przy tworzeniu H2.",
                     "Wiele z nich to podtematy których BRAK u konkurencji — Twoja szansa:"]
            for rs_t in rs_texts:
                lines.append(f"  🔍 {rs_t}")
            sections.append("\n".join(lines))

    triplet_list = (causal_triplets.get("chains") or causal_triplets.get("singles")
                    or causal_triplets.get("triplets") or [])[:8]
    if triplet_list:
        lines = ["═══ PRZYCZYNOWE ZALEŻNOŚCI (cause→effect z konkurencji) ═══",
                 "Confidence: 🔴 ≥0.9 UŻYJ | 🟡 ≥0.6 gdy pasuje | 🟢 <0.6 opcjonalnie",
                 "is_chain=True (A→B→C) = najcenniejsze. Buduj logiczny przepływ"]
        for t in triplet_list:
            if isinstance(t, dict):
                cause = t.get("cause", t.get("subject", ""))
                effect = t.get("effect", t.get("object", ""))
                conf = t.get("confidence", 0)
                is_chain = t.get("is_chain", False)
                
                # Priority indicator
                if conf >= 0.9:
                    ind = "🔴"
                elif conf >= 0.6:
                    ind = "🟡"
                else:
                    ind = "🟢"
                chain_tag = " [CHAIN]" if is_chain else ""
                conf_str = f" ({conf:.1f})" if conf else ""
                lines.append(f"  {ind} {cause} → {effect}{conf_str}{chain_tag}")
            elif isinstance(t, str):
                lines.append(f"  • {t}")
        sections.append("\n".join(lines))

    # Fix #48: Entity-driven H2 generation — top entities should influence H2 names
    entity_seo = s1_data.get("entity_seo") or {}
    concept_ents = entity_seo.get("concept_entities") or entity_seo.get("topical_entities") or []
    must_mention = entity_seo.get("must_mention") or []
    top_named = entity_seo.get("top_entities") or []
    entity_salience = entity_seo.get("entity_salience") or []

    all_ents = []
    seen_ent = set()
    for src in [concept_ents, must_mention, top_named]:
        for e in src[:15]:
            name = e if isinstance(e, str) else (e.get("text") or e.get("entity") or e.get("display_text") or "")
            name_low = name.lower().strip()
            if name_low and name_low not in seen_ent and name_low != main_keyword.lower():
                seen_ent.add(name_low)
                sal = 0
                for se in entity_salience:
                    if isinstance(se, dict) and (se.get("entity", "")).lower() == name_low:
                        sal = se.get("salience", 0)
                        break
                all_ents.append((name, sal))

    if all_ents:
        # Sort by salience descending
        all_ents.sort(key=lambda x: x[1], reverse=True)
        lines = ["═══ TOP ENCJE Z KONKURENCJI — UŻYJ W NAZEWNICTWIE H2 ═══",
                 "Poniższe encje pojawiają się najczęściej u konkurencji.",
                 "ZASADA: Każde H2 powinno zawierać 1-2 encje z tej listy.",
                 "To daje H2 efekt typu Surfer/NeuronWriter — H2 bogate w encje.",
                 "NIE kopiuj dosłownie, ale wplataj naturalnie w nazwy H2.",
                 "Przykład: zamiast 'Konsekwencje' → 'Konsekwencje prawne i utrata prawa jazdy'",
                 ""]
        for i, (name, sal) in enumerate(all_ents[:14], 1):
            sal_str = f" (salience: {sal:.2f})" if sal > 0 else ""
            priority = "🔴 MUST" if i <= 5 else ("🟡 HIGH" if i <= 10 else "🟢 OPT")
            lines.append(f"  {i:2}. [{priority}] {name}{sal_str}")
        sections.append("\n".join(lines))

    if user_h2_hints:
        h2_hints_list = "\n".join(f'  • "{h}"' for h in user_h2_hints[:10])
        sections.append(f"""═══ FRAZY H2 UŻYTKOWNIKA ═══

Użytkownik podał te frazy z myślą o nagłówkach H2.
Wykorzystaj je w nagłówkach tam, gdzie brzmią naturalnie po polsku.
Nie musisz użyć każdej, ale nie ignoruj ich. Dopasuj z wyczuciem.

Jeśli fraza brzmi sztucznie jako nagłówek, przeformułuj lub pomiń (trafi do treści).

FRAZY H2:
{h2_hints_list}""")

    if all_user_phrases:
        phrases_text = ", ".join(f'"{p}"' for p in all_user_phrases[:15])
        sections.append(f"""═══ KONTEKST TEMATYCZNY (frazy BASIC/EXTENDED) ═══

Poniższe frazy będą użyte W TREŚCI artykułu (nie w nagłówkach).
Podaję je żebyś wiedział jaki zakres tematyczny artykuł musi pokryć
i zaplanował H2 tak, by każda fraza miała naturalną sekcję:

{phrases_text}""")

    fast_note = "Tryb fast: DOKŁADNIE 3 sekcje + FAQ (4 H2 łącznie)." if mode == "fast" else ""
    
    # v50.8 FIX 50: H2 scaling: minimum 5-6 sekcji nawet dla krótkich artykułów.
    # Więcej sekcji = lepsza struktura, lepsze SEO, łatwiejsze skanowanie.
    length_analysis = s1_data.get("length_analysis") or {}
    rec_length = length_analysis.get("recommended") or s1_data.get("recommended_length") or 0
    median_length = length_analysis.get("median") or s1_data.get("median_length") or 0
    
    if mode != "fast":
        target = rec_length or (median_length * 2) or 1500
        if target <= 1000:
            h2_range = "5-6"
            h2_min, h2_max = 5, 6
        elif target <= 2000:
            h2_range = "6-8"
            h2_min, h2_max = 6, 8
        elif target <= 3500:
            h2_range = "7-9"
            h2_min, h2_max = 7, 9
        else:
            h2_range = "8-12"
            h2_min, h2_max = 8, 12
        
        fast_note = (
            f"Tryb standard: {h2_range} sekcji + FAQ ({h2_min+1}-{h2_max+1} H2 łącznie).\n"
            f"   UWAGA: Rekomendowana długość artykułu: ~{target} słów (mediana konkurencji: {median_length}).\n"
            f"   Każda sekcja H2 = ~{target // (h2_max + 1)}-{target // h2_min} słów.\n"
            f"   NIE GENERUJ więcej niż {h2_max + 1} H2 (wliczając FAQ)!"
        )
    
    h2_hint_rule = ("Uwzględnij frazy H2 użytkownika w nagłówkach, o ile brzmią naturalnie."
                    if user_h2_hints else "Dobierz nagłówki na podstawie S1 i luk treściowych.")

    sections.append(f"""═══ ZASADY ═══

1. LICZBA H2: {fast_note}
2. OSTATNI H2 MUSI być: "Najczęściej zadawane pytania"
3. Pokryj najważniejsze wzorce z konkurencji + luki treściowe (przewaga nad konkurencją)
4. {h2_hint_rule}
5. Logiczna narracja: od ogółu do szczegółu, chronologicznie, lub problemowo
6. NIE powtarzaj hasła głównego dosłownie w każdym H2
7. H2 muszą brzmieć naturalnie po polsku, żadnego keyword stuffingu
8. ENCJE W H2: Każde H2 powinno zawierać 1-2 encje z listy TOP ENCJI powyżej.
   To poprawia topical authority i pokrycie tematyczne (jak w Surfer/NeuronWriter).
   Nie upychaj na siłę, ale naturalnie wplataj encje w nazwy H2.
9. Preferuj H2 konkretne i informacyjne (z liczbami, encjami, terminami) nad ogólnikowe.
   ❌ "Kary" → ✅ "Kary za jazdę po alkoholu — grzywna, zakaz i więzienie"
   ❌ "Procedura" → ✅ "Badanie alkomatem i procedura kontroli drogowej"

═══ FORMAT ODPOWIEDZI ═══

Odpowiedz TYLKO JSON array, bez markdown, bez komentarzy:
["H2 pierwszy", "H2 drugi", ..., "Najczęściej zadawane pytania"]""")

    return "\n\n".join(sections)
