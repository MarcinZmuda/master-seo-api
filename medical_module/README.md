# 🏥 BRAJEN Medical Module v1.0

Moduł do obsługi treści medycznych (YMYL Health) dla **BRAJEN SEO Engine v44.2**.

Wzorowany na architekturze `legal_module_v3` - ta sama filozofia multi-source z graceful degradation.

---

## 📋 Spis treści

- [Funkcjonalności](#-funkcjonalności)
- [Architektura](#-architektura)
- [Instalacja](#-instalacja)
- [Konfiguracja](#-konfiguracja)
- [Użycie](#-użycie)
- [API Endpoints](#-api-endpoints)
- [Integracja z BRAJEN](#-integracja-z-brajen)
- [Cytowania](#-cytowania)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Funkcjonalności

| Funkcja | Opis |
|---------|------|
| **Detekcja YMYL** | Automatyczne wykrywanie tematów medycznych |
| **PubMed Search** | Wyszukiwanie publikacji naukowych (NCBI E-utilities) |
| **ClinicalTrials** | Badania kliniczne z ClinicalTrials.gov API v2 |
| **Polskie źródła** | Scraping PZH, AOTMiT, MZ, NFZ |
| **Claude Verifier** | AI scoring publikacji (hierarchia dowodów EBM) |
| **Cytowania** | Automatyczne formatowanie NLM/APA |
| **Walidacja** | Sprawdzanie artykułu przed publikacją |

---

## 🏗 Architektura

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MEDICAL MODULE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ 1. DETECT   │───►│ 2. SEARCH   │───►│ 3. VERIFY (Claude)      │  │
│  │ Czy YMYL?   │    │ 4 źródła    │    │ Hierarchia dowodów      │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│                            │                       │                │
│  ┌─────────────────────────┴───────────────────────┘                │
│  │                                                                  │
│  │  ŹRÓDŁA DANYCH:                                                  │
│  │  ├─ 🔬 PubMed (NCBI E-utilities) - publikacje naukowe           │
│  │  ├─ 🧪 ClinicalTrials.gov API v2 - badania kliniczne            │
│  │  ├─ 🇵🇱 Polish Health (PZH, AOTMiT, MZ, NFZ) - lokalne authority │
│  │  └─ 🤖 Claude AI - weryfikacja i scoring                        │
│  │                                                                  │
│  └──────────────────────────────────────────────────────────────────┘
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ 4. CITE     │───►│ 5. VALIDATE │───►│ 6. INSTRUCTION          │  │
│  │ NLM/APA     │    │ Disclaimer? │    │ dla GPT                 │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Instalacja

### 1. Skopiuj moduł do BRAJEN

```bash
# Skopiuj cały katalog medical_module do projektu BRAJEN
cp -r medical_module/ /path/to/brajen/

# Lub jako submoduł
cd /path/to/brajen
git submodule add <repo_url> medical_module
```

### 2. Zainstaluj zależności

```bash
cd medical_module
pip install -r requirements.txt
```

### 3. Skonfiguruj klucze API

```bash
# Skopiuj przykładowy .env
cp .env.example .env

# Lub dodaj do istniejącego .env w głównym katalogu BRAJEN
cat .env >> /path/to/brajen/.env
```

---

## ⚙️ Konfiguracja

### Zmienne środowiskowe

| Zmienna | Wymagana | Opis |
|---------|----------|------|
| `NCBI_API_KEY` | ❌ Zalecana | Zwiększa limit PubMed z 3 do 10 req/sek |
| `NCBI_EMAIL` | ❌ Zalecana | Wymagany przez NCBI policy |
| `ANTHROPIC_API_KEY` | ❌ Opcjonalna | Dla Claude verifier (używa istniejący z BRAJEN) |

### Plik `.env`

```env
# NCBI PubMed
NCBI_API_KEY=your_ncbi_api_key_here
NCBI_EMAIL=your_email@example.com

# Anthropic (użyj swojego klucza z BRAJEN)
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Konfiguracja modułu

Edytuj `medical_module.py` → `MedicalConfig`:

```python
@dataclass
class MedicalConfig:
    MAX_CITATIONS_PER_ARTICLE: int = 3    # Max cytowań
    MAX_PUBMED_RESULTS: int = 10          # Max wyników z PubMed
    MIN_YEAR: int = 2015                  # Filtr roku
    PREFERRED_ARTICLE_TYPES: List[str]    # Preferowane typy
```

---

## 🚀 Użycie

### Podstawowe użycie

```python
from medical_module import (
    detect_category,
    get_medical_context_for_article,
    validate_medical_article,
    MEDICAL_DISCLAIMER
)

# 1. Sprawdź czy temat jest medyczny
result = detect_category("leczenie cukrzycy typu 2")
print(f"Is YMYL: {result['is_ymyl']}")
print(f"Confidence: {result['confidence']}")
print(f"Specialization: {result['specialization']}")

# 2. Pobierz kontekst dla artykułu (główna funkcja)
context = get_medical_context_for_article(
    main_keyword="leczenie cukrzycy typu 2",
    additional_keywords=["metformina", "dieta"],
    max_results=3,
    include_clinical_trials=True,
    include_polish_sources=True
)

print(f"Status: {context['status']}")
print(f"Publications: {len(context['publications'])}")
print(f"Clinical trials: {len(context['clinical_trials'])}")

# 3. Użyj instrukcji w GPT
instruction = context['instruction']
# → Przekaż do GPT jako kontekst

# 4. Waliduj gotowy artykuł
validation = validate_medical_article(article_text)
if not validation['valid']:
    print(f"Warnings: {validation['warnings']}")
    print(f"Suggestions: {validation['suggestions']}")
```

### Bezpośredni dostęp do źródeł

```python
# PubMed
from medical_module import search_pubmed

result = search_pubmed(
    query="diabetes type 2 metformin",
    max_results=10,
    min_year=2020,
    article_types=["Systematic Review", "Meta-Analysis"]
)

for pub in result['publications']:
    print(f"{pub['authors_short']} ({pub['year']}): {pub['title']}")

# ClinicalTrials.gov
from medical_module import search_completed_trials

result = search_completed_trials(
    condition="type 2 diabetes",
    intervention="metformin",
    max_results=5
)

for study in result['studies']:
    print(f"{study['nct_id']}: {study['brief_title']}")

# Polskie źródła
from medical_module import search_polish_health

result = search_polish_health(
    query="cukrzyca typu 2 leczenie",
    sources=["pzh", "aotmit"]
)

for item in result['results']:
    print(f"[{item['source_short']}] {item['title']}")
```

---

## 🔌 API Endpoints

### Flask Integration

```python
# W master_api.py lub app.py
from medical_module import medical_routes

app.register_blueprint(medical_routes)
```

### Dostępne endpointy

| Endpoint | Method | Opis |
|----------|--------|------|
| `/api/medical/status` | GET | Status modułu i źródeł |
| `/api/medical/detect` | POST | Wykrywanie kategorii YMYL |
| `/api/medical/get_context` | POST | **Główny** - pobiera źródła |
| `/api/medical/search/pubmed` | POST | Bezpośrednie wyszukiwanie PubMed |
| `/api/medical/search/trials` | POST | Bezpośrednie wyszukiwanie ClinicalTrials |
| `/api/medical/search/polish` | POST | Bezpośrednie wyszukiwanie PL |
| `/api/medical/validate` | POST | Walidacja artykułu |
| `/api/medical/disclaimer` | GET | Tekst disclaimera |

### Przykłady requestów

```bash
# Status
curl http://localhost:5000/api/medical/status

# Detekcja
curl -X POST http://localhost:5000/api/medical/detect \
  -H "Content-Type: application/json" \
  -d '{"main_keyword": "leczenie cukrzycy typu 2"}'

# Pobierz kontekst
curl -X POST http://localhost:5000/api/medical/get_context \
  -H "Content-Type: application/json" \
  -d '{
    "main_keyword": "leczenie cukrzycy typu 2",
    "additional_keywords": ["metformina"],
    "max_results": 3
  }'
```

---

## 🔗 Integracja z BRAJEN

### W `project_routes.py`

```python
from medical_module import enhance_project_with_medical

@app.route('/api/project/create', methods=['POST'])
def create_project():
    # ... istniejący kod ...
    
    # Dodaj kontekst medyczny
    project_data = enhance_project_with_medical(
        project_data=project_data,
        main_keyword=main_keyword,
        h2_list=h2_list
    )
    
    return jsonify(project_data)
```

### W `gpt_instruction_builder.py`

```python
def build_instruction(project_data):
    instruction = ""
    
    # ... istniejący kod ...
    
    # Dodaj kontekst medyczny jeśli dostępny
    if project_data.get('medical_context', {}).get('medical_module_active'):
        instruction += project_data.get('medical_instruction', '')
    
    return instruction
```

### W eksporcie artykułu

```python
from medical_module import check_medical_on_export

def export_article(article_text, category):
    # Sprawdź wymagania medyczne
    check = check_medical_on_export(article_text, category)
    
    if check['medical_check'] == 'WARNING':
        print(f"⚠️ Warnings: {check['warnings']}")
        print(f"💡 Suggestions: {check['suggestions']}")
    
    # ... kontynuuj eksport ...
```

---

## 📚 Cytowania

### Style

| Styl | Format | Użycie |
|------|--------|--------|
| **NLM** | `Smith J, Doe A. Title. J Name. 2023;12:45-50.` | Medycyna (default) |
| **APA** | `Smith, J., & Doe, A. (2023). Title. Journal.` | Psychologia |

### Przykład

```python
from medical_module import format_citation, CitationStyle

citation = format_citation(publication, CitationStyle.NLM)

print(citation['inline'])  # "Smith i wsp. (2023)"
print(citation['full'])    # "Smith J, Doe A, et al. Title. J. 2023;..."
print(citation['doi_link']) # "https://doi.org/10.1234/..."
```

### Hierarchia dowodów (EBM)

| Level | Typ | Wiarygodność |
|-------|-----|--------------|
| 1 ⭐⭐⭐⭐⭐ | Meta-analizy, Systematic Reviews, Guidelines | Najwyższa |
| 2 ⭐⭐⭐⭐ | RCT (Randomized Controlled Trials) | Wysoka |
| 3 ⭐⭐⭐ | Cohort studies, Reviews | Średnia |
| 4 ⭐⭐ | Case series | Niska |
| 5 ⭐ | Case reports, Expert opinion | Bardzo niska |

---

## 🔧 Troubleshooting

### "PubMed Client not available"

```bash
# Sprawdź czy requests jest zainstalowany
pip install requests

# Sprawdź .env
echo $NCBI_API_KEY
```

### "Rate limit exceeded" (PubMed)

```bash
# Dodaj API key do .env
NCBI_API_KEY=your_key_here

# Lub zwiększ delay w pubmed_client.py
REQUEST_DELAY_NO_KEY: float = 0.5  # 2 req/sek
```

### "Claude verification error"

```bash
# Sprawdź ANTHROPIC_API_KEY
# Moduł będzie działać bez Claude (fallback selection)
```

### "Polish sources timeout"

Polskie strony mogą być wolne. Zwiększ timeout:

```python
# W polish_health_scraper.py
TIMEOUT: int = 30  # zamiast 15
```

---

## 📝 Disclaimer

```
ZASTRZEŻENIE: Niniejszy artykuł ma charakter wyłącznie informacyjny 
i edukacyjny. Nie stanowi porady medycznej ani nie zastępuje konsultacji 
z lekarzem lub innym wykwalifikowanym pracownikiem służby zdrowia. 
W przypadku problemów zdrowotnych należy skonsultować się z lekarzem.
```

---

## 📁 Struktura plików

```
medical_module/
├── __init__.py                 # Eksporty
├── medical_module.py           # 🏥 Główny orchestrator
├── pubmed_client.py            # 🔬 NCBI E-utilities
├── clinicaltrials_client.py    # 🧪 ClinicalTrials.gov API
├── polish_health_scraper.py    # 🇵🇱 PZH, AOTMiT, MZ, NFZ
├── medical_term_detector.py    # 🔍 Detekcja + MeSH mapping
├── claude_medical_verifier.py  # 🤖 AI scoring
├── medical_citation_generator.py # 📚 Cytowania NLM/APA
├── medical_routes.py           # 🌐 Flask endpoints
├── requirements.txt            # 📦 Zależności
├── .env                        # 🔑 Klucze API
├── .env.example                # 🔑 Przykład .env
└── README.md                   # 📖 Dokumentacja
```

---

## 📊 Porównanie z Legal Module

| Aspekt | Legal Module | Medical Module |
|--------|--------------|----------------|
| Główne źródło | SAOS API | PubMed E-utilities |
| Drugie źródło | 10 portali SO | ClinicalTrials.gov |
| Polskie źródła | - | MZ, PZH, NFZ, AOTMiT |
| Claude scoring | Weryfikacja przepisów | Hierarchia dowodów EBM |
| Max cytaty | 2 sygnatury | 3 publikacje |
| Format cytowań | Prawniczy | NLM/APA |

---

## 🆘 Support

W razie problemów:
1. Sprawdź logi: `[MEDICAL_MODULE]`, `[PUBMED]`, `[CLINICALTRIALS]`
2. Testuj komponenty osobno (każdy plik ma `if __name__ == "__main__"`)
3. Sprawdź dostępność API: `curl https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi`

---

**Autor:** BRAJEN SEO Engine  
**Wersja:** 1.0.0  
**Licencja:** Proprietary
