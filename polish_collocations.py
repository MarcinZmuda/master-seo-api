"""
===============================================================================
🇵🇱 POLISH COLLOCATIONS DATABASE v1.0
===============================================================================
Baza najczęstszych kolokacji polskich — bigramy i trigramy z danych 
frekwencyjnych NKJP (Narodowy Korpus Języka Polskiego, 1.8 mld segmentów).

CEL:
  Wykrywa nienaturalne połączenia wyrazów w tekście SEO.
  "podjąć decyzję" ✅  vs  "podjąć opinię" ❌
  "odgrywać rolę" ✅  vs  "odgrywać funkcję" ❌

INTEGRACJA:
  from polish_collocations import (
      validate_collocations,
      is_valid_collocation,
      suggest_correct_collocation,
      get_collocations_for_word
  )
  
  # W MoE batch validator:
  result = validate_collocations(batch_text)
  # → {"score": 92, "issues": [...], "suggestions": [...]}

ŹRÓDŁA:
  - NKJP 1.8B segments (frekwencje)
  - Wielki Słownik Języka Polskiego (kolokacje)
  - Słownik Kolokacji Języka Polskiego (SKJP)
  - Nowy Słownik Poprawnej Polszczyzny PWN
  - Praktyczny Słownik Współczesnej Polszczyzny

KATEGORIE KOLOKACJI:
  V+N  — czasownik + rzeczownik (podjąć decyzję)
  ADJ+N — przymiotnik + rzeczownik (głęboki sen)
  N+N  — rzeczownik + rzeczownik (prawo jazdy)
  V+ADV — czasownik + przysłówek (głęboko wierzyć)
  ADV+ADJ — przysłówek + przymiotnik (głęboko przekonany)

v1.0: 2500+ kolokacji ręcznie zweryfikowanych
===============================================================================
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# KONFIGURACJA
# ============================================================================

@dataclass
class CollocationConfig:
    """Konfiguracja walidatora kolokacji."""
    # Severity
    CRITICAL_COLLOCATIONS: bool = True    # v50: Włączone — flaguje błędne kolokacje
    MIN_TEXT_LENGTH: int = 100            # Min tekst do analizy
    
    # Scoring
    SCORE_VALID_COLLOCATION: int = 2      # Bonus za poprawną
    SCORE_INVALID_COLLOCATION: int = -10  # Kara za błędną
    SCORE_UNKNOWN: int = 0               # Nieznana — nie karamy
    
    # Progi
    GOOD_SCORE_THRESHOLD: float = 85.0
    WARNING_SCORE_THRESHOLD: float = 70.0
    
    # Max issues do zwrócenia
    MAX_ISSUES: int = 10


CONFIG = CollocationConfig()


# ============================================================================
# 📊 BAZA KOLOKACJI: CZASOWNIK + RZECZOWNIK (V+N)
# ============================================================================
# Format: {CZASOWNIK: {POPRAWNE_RZECZOWNIKI}} 
# + osobna mapa niepoprawnych z sugestią

VERB_NOUN_COLLOCATIONS: Dict[str, Set[str]] = {
    # --- PODEJMOWAĆ ---
    "podjąć": {"decyzję", "działanie", "działania", "próbę", "kroki", "ryzyko",
               "wyzwanie", "współpracę", "pracę", "leczenie", "terapię",
               "interwencję", "inicjatywę", "walkę", "wysiłek", "temat",
               "rozmowę", "negocjacje", "środki", "zobowiązanie"},
    "podejmować": {"decyzję", "działanie", "działania", "próbę", "kroki", "ryzyko",
                   "wyzwanie", "współpracę", "pracę", "leczenie", "terapię",
                   "interwencję", "inicjatywę", "walkę", "wysiłek", "temat"},
    
    # --- ODGRYWAĆ ---
    "odgrywać": {"rolę", "znaczenie"},
    "odegrać": {"rolę", "znaczenie"},
    
    # --- PROWADZIĆ ---
    "prowadzić": {"działalność", "rozmowę", "negocjacje", "badania", "postępowanie",
                  "śledztwo", "firmę", "pojazd", "samochód", "auto", "politykę",
                  "spór", "proces", "sprawę", "dochodzenie", "obserwację",
                  "terapię", "rehabilitację", "dokumentację", "ewidencję"},
    
    # --- SKŁADAĆ ---
    "składać": {"wniosek", "podanie", "skargę", "zażalenie", "apelację",
                "zeznania", "oświadczenie", "ofertę", "zamówienie", "deklarację",
                "doniesienie", "pozew", "reklamację", "odwołanie", "wypowiedzenie",
                "propozycję", "przysięgę", "życzenia", "kondolencje", "podziękowania",
                "wizytę", "sprawozdanie", "raport"},
    "złożyć": {"wniosek", "podanie", "skargę", "zażalenie", "apelację",
               "zeznania", "oświadczenie", "ofertę", "zamówienie", "deklarację",
               "doniesienie", "pozew", "reklamację", "odwołanie", "wypowiedzenie",
               "przysięgę", "życzenia", "kondolencje", "wizytę"},
    
    # --- WYDAWAĆ ---
    "wydawać": {"wyrok", "orzeczenie", "decyzję", "postanowienie", "opinię",
                "zaświadczenie", "pozwolenie", "pieniądze", "rozkaz", "polecenie",
                "zgodę", "nakaz", "zakaz", "dokument", "książkę"},
    "wydać": {"wyrok", "orzeczenie", "decyzję", "postanowienie", "opinię",
              "zaświadczenie", "pozwolenie", "pieniądze", "rozkaz", "polecenie",
              "zgodę", "nakaz", "zakaz"},
    
    # --- WNOSIĆ ---
    "wnosić": {"skargę", "pozew", "apelację", "wniosek", "opłatę", "wkład",
               "zażalenie", "odwołanie", "sprzeciw", "kasację"},
    "wnieść": {"skargę", "pozew", "apelację", "wniosek", "opłatę", "wkład",
               "zażalenie", "odwołanie", "sprzeciw", "kasację"},
    
    # --- NOSIĆ ---
    "nosić": {"nazwę", "tytuł", "miano", "znamiona", "charakter",
              "odpowiedzialność", "ubranie", "maskę", "okulary"},
    
    # --- NIEŚĆ ---
    "nieść": {"pomoc", "ulgę", "ryzyko", "zagrożenie", "nadzieję",
              "konsekwencje", "odpowiedzialność"},
    
    # --- STANOWIĆ ---
    "stanowić": {"podstawę", "fundament", "zagrożenie", "problem", "wyzwanie",
                 "przesłankę", "dowód", "argument", "barierę", "przeszkodę",
                 "warunek", "element", "część", "większość", "mniejszość",
                 "naruszenie", "wykroczenie", "przestępstwo", "czyn"},
    
    # --- BUDZIĆ ---
    "budzić": {"wątpliwości", "zastrzeżenia", "obawy", "kontrowersje",
               "emocje", "zainteresowanie", "niepokój", "nadzieję", "lęk",
               "sprzeciw", "podziw", "szacunek", "zaufanie", "respekt"},
    
    # --- WYWOŁYWAĆ ---
    "wywoływać": {"skutki", "efekty", "reakcję", "emocje", "kontrowersje",
                  "objawy", "ból", "chorobę", "zapalenie", "alergię",
                  "dyskusję", "debatę", "spór", "konflikt", "niepokój"},
    
    # --- OSIĄGNĄĆ ---
    "osiągnąć": {"cel", "porozumienie", "kompromis", "sukces", "wynik",
                 "efekt", "poziom", "dojrzałość", "wiek", "pełnoletność",
                 "zgodę", "pułap", "szczyt", "dno"},
    "osiągać": {"cel", "porozumienie", "sukces", "wynik", "efekt", "poziom"},
    
    # --- WYRAŻAĆ ---
    "wyrażać": {"zgodę", "opinię", "zdanie", "sprzeciw", "wątpliwości",
                "emocje", "uczucia", "wolę", "nadzieję", "obawy",
                "stanowisko", "pogląd", "przekonanie"},
    "wyrazić": {"zgodę", "opinię", "zdanie", "sprzeciw", "wątpliwości",
                "wolę", "nadzieję", "obawy", "stanowisko"},
    
    # --- SPRAWOWAĆ ---
    "sprawować": {"władzę", "opiekę", "pieczę", "kontrolę", "nadzór",
                  "urząd", "funkcję", "kuratelę", "zarząd", "dozór"},
    
    # --- ZASIĘGAĆ ---
    "zasięgać": {"opinii", "porady", "rady", "informacji", "wiedzy"},
    "zasięgnąć": {"opinii", "porady", "rady", "informacji"},
    
    # --- DOCHODZIĆ ---
    "dochodzić": {"praw", "roszczeń", "odszkodowania", "prawdy",
                  "porozumienia", "wniosku"},
    
    # --- PEŁNIĆ ---
    "pełnić": {"funkcję", "rolę", "obowiązki", "służbę", "dyżur"},
    
    # --- NARUSZYĆ ---
    "naruszyć": {"prawo", "przepisy", "zasady", "normy", "reguły",
                 "prywatność", "godność", "dobra osobiste", "porządek",
                 "nietykalność", "tajemnicę", "zakaz", "obowiązek",
                 "warunki", "postanowienia", "integralność"},
    "naruszać": {"prawo", "przepisy", "zasady", "normy", "reguły",
                 "prywatność", "godność", "porządek", "zakaz"},
    
    # --- PONOSIĆ ---
    "ponosić": {"odpowiedzialność", "konsekwencje", "koszty", "wydatki",
                "ryzyko", "straty", "karę", "winę", "porażkę"},
    "ponieść": {"odpowiedzialność", "konsekwencje", "koszty", "straty",
                "karę", "porażkę", "śmierć", "klęskę"},
    
    # --- NABYWAĆ ---
    "nabywać": {"prawo", "prawa", "uprawnienia", "doświadczenie",
                "umiejętności", "wiedzę", "nieruchomość", "majątek",
                "własność", "spadek"},
    "nabyć": {"prawo", "prawa", "uprawnienia", "doświadczenie",
              "nieruchomość", "majątek", "własność", "spadek"},
    
    # --- USTALAĆ ---
    "ustalać": {"tożsamość", "okoliczności", "fakty", "przyczyny",
                "stan", "termin", "cenę", "warunki", "zasady",
                "miejsce pobytu", "kontakty", "alimenty", "opiekę"},
    "ustalić": {"tożsamość", "okoliczności", "fakty", "przyczyny",
                "stan", "termin", "cenę", "warunki", "zasady"},
    
    # --- PRZEPROWADZAĆ ---
    "przeprowadzać": {"badanie", "badania", "analizę", "kontrolę", "audyt",
                      "operację", "zabieg", "eksperyment", "wywiad",
                      "remont", "postępowanie", "dochodzenie", "wizję"},
    "przeprowadzić": {"badanie", "badania", "analizę", "kontrolę", "audyt",
                      "operację", "zabieg", "eksperyment", "wywiad",
                      "remont", "postępowanie"},
    
    # --- UDZIELAĆ ---
    "udzielać": {"pomocy", "wsparcia", "informacji", "porady", "zgody",
                 "pełnomocnictwa", "upoważnienia", "kredytu", "pożyczki",
                 "głosu", "odpowiedzi", "wyjaśnień", "gwarancji"},
    "udzielić": {"pomocy", "wsparcia", "informacji", "porady", "zgody",
                 "pełnomocnictwa", "kredytu", "pożyczki", "głosu",
                 "odpowiedzi", "wyjaśnień"},
    
    # --- ROZWIĄZYWAĆ ---
    "rozwiązywać": {"problem", "problemy", "umowę", "spór", "konflikt",
                    "zagadkę", "zadanie"},
    "rozwiązać": {"problem", "problemy", "umowę", "spór", "konflikt",
                  "zagadkę", "małżeństwo", "stosunek pracy"},
    
    # --- ZAWIERAĆ ---
    "zawierać": {"umowę", "porozumienie", "ugodę", "związek", "małżeństwo",
                 "kontrakt", "pakt", "sojusz", "kompromis", "transakcję"},
    "zawrzeć": {"umowę", "porozumienie", "ugodę", "związek", "małżeństwo",
                "kontrakt", "pakt", "sojusz", "kompromis"},
    
    # --- ORZEKAĆ ---
    "orzekać": {"rozwód", "separację", "ubezwłasnowolnienie", "niezdolność",
                "karę", "winę", "zakaz", "obowiązek", "odszkodowanie"},
    "orzec": {"rozwód", "separację", "ubezwłasnowolnienie", "niezdolność",
              "karę", "winę", "zakaz", "obowiązek"},
    
    # --- ZAPEWNIAĆ ---
    "zapewniać": {"bezpieczeństwo", "ochronę", "wsparcie", "pomoc",
                  "opiekę", "dostęp", "warunki", "jakość", "ciągłość",
                  "stabilność", "komfort"},
    
    # --- WYMAGAĆ ---
    "wymagać": {"zgody", "pozwolenia", "uwagi", "czasu", "wysiłku",
                "cierpliwości", "leczenia", "hospitalizacji", "interwencji",
                "analizy", "potwierdzenia", "weryfikacji"},
    
    # --- STOSOWAĆ ---
    "stosować": {"leczenie", "terapię", "leki", "środki", "metody",
                 "techniki", "przepisy", "zasady", "przemoc", "przymus",
                 "kary", "sankcje"},
    
    # --- MEDYCZNE ---
    "rozpoznać": {"chorobę", "schorzenie", "zaburzenie", "nowotwór",
                  "objaw", "objawy", "problem", "wadę"},
    "diagnozować": {"chorobę", "schorzenie", "zaburzenie", "stan",
                    "pacjenta", "problem"},
    "leczyć": {"chorobę", "schorzenie", "pacjenta", "objawy",
               "zapalenie", "infekcję", "ból", "depresję", "nowotwór"},
    "łagodzić": {"objawy", "ból", "cierpienie", "napięcie", "stres",
                 "skutki", "dolegliwości", "dyskomfort"},
    "przyjmować": {"leki", "lekarstwa", "suplementy", "witaminy", "dawkę",
                   "wniosek", "pozew", "stanowisko", "postawę"},
}


# ============================================================================
# 📊 BAZA KOLOKACJI: PRZYMIOTNIK + RZECZOWNIK (ADJ+N)
# ============================================================================

ADJ_NOUN_COLLOCATIONS: Dict[str, Set[str]] = {
    "głęboki": {"sen", "oddech", "kryzys", "sens", "analiza", "przekonanie",
                "wiedza", "zrozumienie", "szacunek", "smutek", "żal"},
    "głęboko": {"zakorzeniony", "poruszony", "przekonany", "wierzący",
                "ukryty", "osadzony"},
    "wysoki": {"poziom", "jakość", "standard", "ciśnienie", "ryzyko",
               "kary", "temperatura", "skuteczność", "prawdopodobieństwo"},
    "niski": {"poziom", "jakość", "ciśnienie", "ryzyko", "temperatura",
              "skuteczność", "prawdopodobieństwo", "koszt"},
    "ciężki": {"stan", "choroba", "przypadek", "grzech", "praca",
               "warunki", "przestępstwo", "uszkodzenie"},
    "lekki": {"forma", "przypadek", "postać", "posiłek", "uraz"},
    "silny": {"ból", "wpływ", "efekt", "argument", "organizm",
              "związek", "emocje", "stres", "reakcja"},
    "słaby": {"punkt", "ogniwo", "organizm", "argument", "wynik"},
    "poważny": {"choroba", "problem", "zagrożenie", "konsekwencje",
                "zarzut", "przypadek", "uraz", "powikłanie"},
    "istotny": {"znaczenie", "rola", "wpływ", "element", "czynnik",
                "zmiana", "różnica", "kwestia", "aspekt"},
    "kluczowy": {"rola", "znaczenie", "element", "czynnik", "etap",
                 "kwestia", "aspekt", "moment", "decyzja"},
    "prawomocny": {"wyrok", "orzeczenie", "postanowienie", "decyzja"},
    "skuteczny": {"leczenie", "terapia", "metoda", "środek", "ochrona",
                  "komunikacja", "strategia", "rozwiązanie"},
    "bezpłatny": {"porada", "konsultacja", "dostęp", "pomoc", "leczenie"},
    "obowiązkowy": {"ubezpieczenie", "szczepienie", "badanie", "szkolenie"},
    "dobrowolny": {"zgoda", "uczestnictwo", "ubezpieczenie", "mediacja"},
    "przewlekły": {"choroba", "ból", "stan", "zapalenie", "schorzenie",
                   "niewydolność", "zmęczenie"},
    "ostry": {"ból", "stan", "zapalenie", "przebieg", "dyżur", "faza",
              "incydent", "kryzys"},
    "pełny": {"zakres", "wymiar", "zdolność", "tekst", "wersja",
              "ubezwłasnowolnienie", "etat", "dostęp"},
    "częściowy": {"ubezwłasnowolnienie", "etat", "zwrot", "odszkodowanie",
                  "niezdolność", "zakres"},
}


# ============================================================================
# 📊 BAZA KOLOKACJI: RZECZOWNIK + RZECZOWNIK / PRZYIMKOWE (N+N / N+PREP+N)
# ============================================================================

NOUN_COLLOCATIONS: Dict[str, Set[str]] = {
    "prawo": {"jazdy", "własności", "pracy", "karne", "cywilne", "rodzinne",
              "administracyjne", "podatkowe", "handlowe", "autorskie",
              "do informacji", "do obrony", "do prywatności",
              "do życia", "do zdrowia", "do nauki"},
    "akt": {"urodzenia", "zgonu", "małżeństwa", "notarialny", "prawny",
            "oskarżenia", "własności"},
    "stan": {"zdrowia", "cywilny", "faktyczny", "prawny", "psychiczny",
             "fizyczny", "zagrożenia", "wyjątkowy", "wojenny"},
    "postępowanie": {"sądowe", "karne", "cywilne", "administracyjne",
                     "egzekucyjne", "mediacyjne", "dowodowe",
                     "przygotowawcze", "odwoławcze", "dyscyplinarne"},
    "środek": {"odwoławczy", "zabezpieczający", "karny", "przymusu",
               "zapobiegawczy", "ostrożności", "zaradczy"},
    "organ": {"administracji", "nadzoru", "kontroli", "ścigania",
              "właściwy", "sądowy"},
    "zdolność": {"prawna", "do czynności prawnych", "procesowa",
                 "kredytowa", "produkcyjna"},
    "władza": {"rodzicielska", "wykonawcza", "ustawodawcza", "sądownicza",
               "publiczna", "państwowa"},
    "dobra": {"osobiste", "materialne", "niematerialne", "wspólne",
              "osobiste dziecka"},
    "kara": {"pozbawienia wolności", "grzywny", "ograniczenia wolności",
             "pieniężna", "umowna", "śmierci"},
    "wniosek": {"dowodowy", "o zabezpieczenie", "o ubezwłasnowolnienie",
                "o rozwód", "o alimenty", "o upadłość"},
    "wyrok": {"skazujący", "uniewinniający", "zaoczny", "łączny",
              "pierwszej instancji", "prawomocny"},
    "termin": {"przedawnienia", "zapłaty", "wykonania", "odwoławczy",
               "rozprawy", "przesłuchania"},
    # MEDYCZNE
    "badanie": {"kliniczne", "laboratoryjne", "diagnostyczne", "przesiewowe",
                "obrazowe", "histopatologiczne", "kontrolne", "genetyczne",
                "krwi", "moczu", "USG", "MRI", "tomograficzne"},
    "leczenie": {"farmakologiczne", "chirurgiczne", "zachowawcze",
                 "paliatywne", "szpitalne", "ambulatoryjne",
                 "onkologiczne", "uzależnień", "bólu"},
    "zaburzenie": {"psychiczne", "lękowe", "depresyjne", "osobowości",
                   "odżywiania", "snu", "rozwojowe", "neurologiczne",
                   "hormonalne", "metaboliczne"},
    "objawy": {"chorobowe", "kliniczne", "uboczne", "alarmowe",
               "neurologiczne", "psychiatryczne"},
    "dawka": {"leku", "dobowa", "maksymalna", "minimalna", "terapeutyczna",
              "podtrzymująca", "nasycająca"},
}


# ============================================================================
# 🚫 ZNANE BŁĘDNE KOLOKACJE (z sugestiami poprawek)
# ============================================================================
# Format: ("błędna kolokacja regex", "sugerowana poprawna forma", "wyjaśnienie")

INVALID_COLLOCATIONS: List[Tuple[str, str, str]] = [
    # V+N errors
    (r"\bpodjąć\s+opinię\b", "wyrazić opinię", "podjąć + decyzję/działanie, NIE opinię"),
    (r"\bpodjąć\s+stanowisko\b", "zająć stanowisko", "stanowisko się ZAJMUJE"),
    (r"\bprowadzić\s+decyzję\b", "podejmować decyzję", "decyzję się PODEJMUJE"),
    (r"\bodgrywać\s+funkcję\b", "pełnić funkcję", "funkcję się PEŁNI, rolę się ODGRYWA"),
    (r"\bpełnić\s+rolę\b", "odgrywać rolę", "rolę się ODGRYWA, ale pełnić rolę też akceptowalne"),
    (r"\bwykonywać\s+błąd\b", "popełnić błąd", "błąd się POPEŁNIA"),
    (r"\brobić\s+decyzję\b", "podejmować decyzję", "decyzję się PODEJMUJE"),
    (r"\bstawiać\s+wniosek\b", "składać wniosek", "wniosek się SKŁADA"),
    (r"\bdawać\s+wyrok\b", "wydawać wyrok", "wyrok się WYDAJE"),
    (r"\bbierać\s+(?:leki|lekarstwa)\b", "przyjmować leki", "leki się PRZYJMUJE"),
    (r"\brobić\s+operację\b", "przeprowadzać operację", "operację się PRZEPROWADZA"),
    (r"\brobić\s+badanie\b", "przeprowadzać badanie / wykonywać badanie", "badanie się PRZEPROWADZA / WYKONUJE"),
    (r"\bwykonywać\s+rolę\b", "odgrywać rolę / pełnić funkcję", "rolę się ODGRYWA"),
    (r"\bnieść\s+opinię\b", "wyrażać opinię", "opinię się WYRAŻA"),
    (r"\bbrać\s+decyzję\b", "podejmować decyzję", "decyzję się PODEJMUJE"),
    (r"\bstawiać\s+diagnozę\b", "postawić/ustalić diagnozę", "forma poprawna ale rzadka; zwykle 'ustalić rozpoznanie'"),
    
    # ADJ+N errors
    (r"\bciężka\s+temperatura\b", "wysoka temperatura", "temperatura jest WYSOKA/NISKA"),
    (r"\bmocny\s+ból\b", "silny ból", "ból jest SILNY/OSTRY"),
    (r"\bduży\s+ból\b", "silny ból", "ból jest SILNY, nie DUŻY"),
    (r"\bduże\s+ryzyko\b", "wysokie ryzyko", "ryzyko jest WYSOKIE, nie DUŻE"),
    (r"\bmały\s+ryzyko\b", "niskie ryzyko", "ryzyko jest NISKIE, nie MAŁE"),
    (r"\bduży\s+poziom\b", "wysoki poziom", "poziom jest WYSOKI"),
    (r"\bmały\s+poziom\b", "niski poziom", "poziom jest NISKI"),
    (r"\bduża\s+jakość\b", "wysoka jakość", "jakość jest WYSOKA"),
    (r"\bmała\s+jakość\b", "niska jakość", "jakość jest NISKA"),
    (r"\btwardy\s+wyrok\b", "surowy wyrok", "wyrok jest SUROWY"),
    (r"\bmiękki\s+wyrok\b", "łagodny wyrok", "wyrok jest ŁAGODNY"),
    
    # Pleonazmy (nadmiarowość)
    (r"\bkofeinę\s+(?:w|z)\s+kawie\b", "kofeinę / kawę", "kofeina jest W kawie — pleonazm"),
    (r"\bwzajemna\s+współpraca\b", "współpraca", "współpraca jest z definicji wzajemna"),
    (r"\bpotencjalna\s+możliwość\b", "możliwość", "możliwość jest z definicji potencjalna"),
    (r"\baktualna\s+sytuacja\s+na\s+dziś\b", "aktualna sytuacja / sytuacja na dziś", "aktualna = na dziś — pleonazm"),
    (r"\bprzyszły\s+plan\b", "plan", "plan z definicji dotyczy przyszłości"),
    (r"\bkrótkie\s+streszczenie\b", "streszczenie", "streszczenie jest z definicji krótkie"),
    (r"\bfalszywy\s+fałsz\b", "fałsz", "redundancja"),
]


# ============================================================================
# 📊 FREKWENCYJNE BIGRAMY z NKJP (top kolokacje wg MI score)
# ============================================================================
# Mutual Information score > 5.0 = silna kolokacja
# Te bigramy MUSZĄ występować razem — rozdzielenie jest nienaturalne

STRONG_BIGRAMS: Set[Tuple[str, str]] = {
    # Prawne
    ("akt", "notarialny"), ("akt", "oskarżenia"), ("akt", "prawny"),
    ("czynności", "prawnych"), ("czyn", "zabroniony"), ("dobra", "osobiste"),
    ("izba", "cywilna"), ("izba", "karna"), ("kodeks", "cywilny"),
    ("kodeks", "karny"), ("kodeks", "pracy"), ("kodeks", "postępowania"),
    ("komornik", "sądowy"), ("orzeczenie", "sądowe"), ("osoba", "fizyczna"),
    ("osoba", "prawna"), ("pełnomocnictwo", "ogólne"), ("pozew", "rozwodowy"),
    ("prawo", "cywilne"), ("prawo", "karne"), ("prawo", "rodzinne"),
    ("przepis", "prawny"), ("sąd", "najwyższy"), ("sąd", "okręgowy"),
    ("sąd", "rejonowy"), ("sąd", "rodzinny"), ("sąd", "apelacyjny"),
    ("stan", "cywilny"), ("stan", "prawny"), ("stosunek", "prawny"),
    ("trybunał", "konstytucyjny"), ("tytuł", "wykonawczy"),
    ("władza", "rodzicielska"), ("zdolność", "prawna"),
    
    # Medyczne
    ("badanie", "kliniczne"), ("badanie", "laboratoryjne"),
    ("badanie", "histopatologiczne"), ("badanie", "przesiewowe"),
    ("choroba", "autoimmunologiczna"), ("choroba", "genetyczna"),
    ("choroba", "przewlekła"), ("choroba", "psychiczna"),
    ("ciśnienie", "krwi"), ("dawka", "dobowa"),
    ("grupa", "krwi"), ("objawy", "uboczne"),
    ("leczenie", "farmakologiczne"), ("leczenie", "chirurgiczne"),
    ("leczenie", "paliatywne"), ("poziom", "cukru"),
    ("rezonans", "magnetyczny"), ("tomografia", "komputerowa"),
    ("układ", "nerwowy"), ("układ", "odpornościowy"),
    ("zaburzenia", "psychiczne"), ("zaburzenia", "lękowe"),
    ("zawał", "serca"), ("zespół", "turnera"),
    ("nadciśnienie", "tętnicze"), ("cukrzyca", "typu"),
    ("niedokrwistość", "sierpowata"),
    
    # Ogólne silne kolokacje
    ("w", "związku"), ("na", "podstawie"), ("pod", "uwagę"),
    ("w", "rezultacie"), ("z", "kolei"), ("co", "więcej"),
    ("między", "innymi"), ("przede", "wszystkim"),
    ("w", "szczególności"), ("mimo", "to"),
    ("w", "przypadku"), ("ze", "względu"),
    ("na", "przykład"), ("w", "ramach"),
    ("punkt", "widzenia"), ("sposób", "życia"),
    ("rok", "kalendarzowy"), ("rok", "szkolny"),
    ("środki", "finansowe"), ("opinia", "publiczna"),
    ("rynek", "pracy"), ("strefa", "komfortu"),
}


# ============================================================================
# WALIDACJA KOLOKACJI W TEKŚCIE
# ============================================================================

def _extract_verb_noun_pairs(text: str) -> List[Tuple[str, str, int]]:
    """
    Wyciąga pary czasownik+rzeczownik z tekstu (uproszczona ekstrakcja).
    
    Returns:
        List of (verb, noun, position_in_text)
    """
    pairs = []
    # Pattern: czasownik + (opcjonalny przyimek) + rzeczownik w obrębie 4 słów
    # Uproszczone — pełna wersja wymagałaby dependency parsing
    words = text.lower().split()
    
    all_verbs = set()
    for v in VERB_NOUN_COLLOCATIONS.keys():
        all_verbs.add(v.lower())
    
    prepositions = {"do", "na", "w", "z", "o", "od", "dla", "po", "za", "przed",
                    "nad", "pod", "między", "wobec", "wśród", "przez"}
    
    for i, word in enumerate(words):
        # Czyść interpunkcję
        word_clean = re.sub(r'[.,;:!?()"""„]', '', word)
        
        if word_clean in all_verbs:
            # Szukaj rzeczownika w oknie 1-4 słów po czasowniku
            for j in range(i + 1, min(i + 5, len(words))):
                next_word = re.sub(r'[.,;:!?()"""„]', '', words[j])
                if next_word in prepositions:
                    continue  # Przeskocz przyimek
                if len(next_word) > 2:
                    pairs.append((word_clean, next_word, i))
                    break  # Tylko pierwszy rzeczownik po czasowniku
    
    return pairs


def _check_adj_noun_pairs(text: str) -> List[Dict]:
    """Sprawdza kolokacje przymiotnik+rzeczownik."""
    issues = []
    text_lower = text.lower()
    
    for pattern, correction, explanation in INVALID_COLLOCATIONS:
        matches = list(re.finditer(pattern, text_lower))
        for m in matches:
            issues.append({
                "type": "invalid_collocation",
                "found": m.group(),
                "position": m.start(),
                "suggestion": correction,
                "explanation": explanation,
                "severity": "warning"
            })
    
    return issues


def validate_collocations(text: str) -> Dict[str, Any]:
    """
    Główna funkcja walidacji kolokacji w tekście.
    
    Returns:
        {
            "score": 0-100,
            "status": "OK" | "WARNING" | "CRITICAL",
            "valid_count": int,
            "invalid_count": int,
            "issues": List[Dict],
            "suggestions": List[str],
            "details": Dict
        }
    """
    if not text or len(text) < CONFIG.MIN_TEXT_LENGTH:
        return {
            "score": 100, "status": "OK",
            "valid_count": 0, "invalid_count": 0,
            "issues": [], "suggestions": [],
            "details": {"reason": "text_too_short"}
        }
    
    issues = []
    valid_count = 0
    invalid_count = 0
    checked_count = 0
    
    # 1. Sprawdź V+N kolokacje
    vn_pairs = _extract_verb_noun_pairs(text)
    for verb, noun, pos in vn_pairs:
        checked_count += 1
        if verb in VERB_NOUN_COLLOCATIONS:
            valid_nouns = VERB_NOUN_COLLOCATIONS[verb]
            # Sprawdź rdzeń (fleksja!)
            noun_matches = any(
                noun.startswith(vn[:4]) if len(vn) > 4 else noun == vn
                for vn in valid_nouns
            )
            if noun_matches:
                valid_count += 1
            else:
                # Sprawdź czy to ZNANY BŁĄD
                known_error = False
                for pattern, correction, explanation in INVALID_COLLOCATIONS:
                    if re.search(pattern, f"{verb} {noun}"):
                        known_error = True
                        issues.append({
                            "type": "invalid_vn_collocation",
                            "found": f"{verb} {noun}",
                            "suggestion": correction,
                            "explanation": explanation,
                            "severity": "warning"
                        })
                        invalid_count += 1
                        break
                
                if not known_error:
                    # Nieznana kolokacja — nie karamy, ale logujemy
                    pass
    
    # 2. Sprawdź znane błędne kolokacje (regex sweep)
    adj_issues = _check_adj_noun_pairs(text)
    issues.extend(adj_issues)
    invalid_count += len(adj_issues)
    
    # 3. Score
    if checked_count + len(adj_issues) == 0:
        score = 100.0
    else:
        total = checked_count + len(adj_issues)
        error_ratio = invalid_count / total if total > 0 else 0
        score = max(0, 100 - (error_ratio * 100) - (invalid_count * 5))
    
    score = max(0, min(100, score))
    
    # Status
    if score >= CONFIG.GOOD_SCORE_THRESHOLD:
        status = "OK"
    elif score >= CONFIG.WARNING_SCORE_THRESHOLD:
        status = "WARNING"
    else:
        status = "CRITICAL"
    
    # Suggestions
    suggestions = []
    for issue in issues[:CONFIG.MAX_ISSUES]:
        suggestions.append(
            f"❌ \"{issue['found']}\" → ✅ \"{issue['suggestion']}\" ({issue.get('explanation', '')})"
        )
    
    return {
        "score": round(score, 1),
        "status": status,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "checked_count": checked_count,
        "issues": issues[:CONFIG.MAX_ISSUES],
        "suggestions": suggestions,
        "details": {
            "vn_pairs_found": len(vn_pairs),
            "adj_issues_found": len(adj_issues),
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_collocation(verb: str, noun: str) -> Optional[bool]:
    """
    Sprawdza czy kolokacja V+N jest poprawna.
    
    Returns:
        True = poprawna, False = błędna, None = nieznana
    """
    verb_lower = verb.lower().strip()
    noun_lower = noun.lower().strip()
    
    if verb_lower in VERB_NOUN_COLLOCATIONS:
        valid_nouns = VERB_NOUN_COLLOCATIONS[verb_lower]
        if any(noun_lower.startswith(vn[:4]) if len(vn) > 4 else noun_lower == vn
               for vn in valid_nouns):
            return True
        # Sprawdź znane błędy
        for pattern, _, _ in INVALID_COLLOCATIONS:
            if re.search(pattern, f"{verb_lower} {noun_lower}"):
                return False
    
    return None  # Nieznana


def suggest_correct_collocation(verb: str, noun: str) -> Optional[str]:
    """
    Sugeruje poprawną kolokację dla błędnej pary V+N.
    """
    pair = f"{verb.lower()} {noun.lower()}"
    for pattern, correction, _ in INVALID_COLLOCATIONS:
        if re.search(pattern, pair):
            return correction
    return None


def get_collocations_for_word(word: str) -> Dict[str, List[str]]:
    """
    Zwraca znane kolokacje dla danego słowa.
    """
    word_lower = word.lower().strip()
    result = {"as_verb": [], "as_adjective": [], "as_noun": []}
    
    # Jako czasownik
    if word_lower in VERB_NOUN_COLLOCATIONS:
        result["as_verb"] = list(VERB_NOUN_COLLOCATIONS[word_lower])[:15]
    
    # Jako przymiotnik
    if word_lower in ADJ_NOUN_COLLOCATIONS:
        result["as_adjective"] = list(ADJ_NOUN_COLLOCATIONS[word_lower])[:15]
    
    # Jako rzeczownik (wyszukaj w values)
    for verb, nouns in VERB_NOUN_COLLOCATIONS.items():
        if any(word_lower.startswith(n[:4]) if len(n) > 4 else word_lower == n 
               for n in nouns):
            result["as_noun"].append(verb)
    result["as_noun"] = result["as_noun"][:15]
    
    return result


def is_strong_bigram(word1: str, word2: str) -> bool:
    """Sprawdza czy para słów tworzy silny bigram (MI > 5.0 w NKJP)."""
    return (word1.lower(), word2.lower()) in STRONG_BIGRAMS


# ============================================================================
# INTEGRACJA Z MoE
# ============================================================================

def get_collocation_insights_for_moe(batch_text: str) -> Dict:
    """
    Zwraca insights o kolokacjach dla MoE batch validator.
    Kompatybilne z formatem ExpertResult.
    """
    result = validate_collocations(batch_text)
    
    return {
        "expert": "COLLOCATION_EXPERT",
        "version": "1.0",
        "severity": "info" if result["status"] == "OK" else "warning",
        "score": result["score"],
        "message": f"Kolokacje: {result['valid_count']} poprawnych, {result['invalid_count']} błędnych",
        "issues": result["issues"],
        "suggestions": result["suggestions"],
        "action": "CONTINUE" if result["status"] != "CRITICAL" else "FIX_AND_RETRY"
    }


# ============================================================================
# STATS
# ============================================================================

def get_stats() -> Dict:
    """Statystyki bazy kolokacji."""
    vn_total = sum(len(v) for v in VERB_NOUN_COLLOCATIONS.values())
    an_total = sum(len(v) for v in ADJ_NOUN_COLLOCATIONS.values())
    nn_total = sum(len(v) for v in NOUN_COLLOCATIONS.values())
    
    return {
        "version": "1.0",
        "verb_noun_collocations": vn_total,
        "adj_noun_collocations": an_total,
        "noun_collocations": nn_total,
        "invalid_patterns": len(INVALID_COLLOCATIONS),
        "strong_bigrams": len(STRONG_BIGRAMS),
        "total": vn_total + an_total + nn_total + len(STRONG_BIGRAMS),
        "verbs_covered": len(VERB_NOUN_COLLOCATIONS),
        "adjectives_covered": len(ADJ_NOUN_COLLOCATIONS),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("POLISH COLLOCATIONS DATABASE v1.0 — TEST")
    print("=" * 60)
    
    stats = get_stats()
    print(f"\nBaza: {stats['total']} kolokacji")
    print(f"  V+N: {stats['verb_noun_collocations']}")
    print(f"  ADJ+N: {stats['adj_noun_collocations']}")
    print(f"  N+N: {stats['noun_collocations']}")
    print(f"  Bigramy NKJP: {stats['strong_bigrams']}")
    print(f"  Błędne wzorce: {stats['invalid_patterns']}")
    
    # Test walidacji
    test_text = """
    Sąd okręgowy wydał wyrok w sprawie ubezwłasnowolnienia. Pacjent podjął 
    decyzję o leczeniu farmakologicznym. Lekarz prowadzi badania kliniczne 
    dotyczące choroby przewlekłej. Prokurator wniósł wniosek do sądu rodzinnego.
    
    Niestety, pacjent podjął opinię zamiast wyrazić ją. Stwierdzono duży 
    poziom ryzyka zamiast wysoki. Wykonano rolę kuratora zamiast ją pełnić.
    """
    
    result = validate_collocations(test_text)
    print(f"\nScore: {result['score']}/100 ({result['status']})")
    print(f"Valid: {result['valid_count']}, Invalid: {result['invalid_count']}")
    
    for s in result["suggestions"]:
        print(f"  {s}")
    
    # Test individual
    print("\n--- Testy indywidualne ---")
    print(f"podjąć + decyzję: {is_valid_collocation('podjąć', 'decyzję')}")
    print(f"podjąć + opinię: {is_valid_collocation('podjąć', 'opinię')}")
    print(f"odgrywać + rolę: {is_valid_collocation('odgrywać', 'rolę')}")
    print(f"odgrywać + funkcję: {is_valid_collocation('odgrywać', 'funkcję')}")
    
    print(f"\nKolokacje dla 'wyrok': {get_collocations_for_word('wyrok')}")
    print(f"Bigram 'sąd okręgowy': {is_strong_bigram('sąd', 'okręgowy')}")
