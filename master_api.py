import os
import re
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# -------------------------------------------------------------------
# ✅ KROK 1: Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
# WAŻNE: W Render.com musisz utworzyć zmienną środowiskową o nazwie
# "FIREBASE_CREDS_JSON" i wkleić do niej CAŁĄ ZAWARTOŚĆ
# pliku serviceAccountKey.json, który pobierzesz z Firebase.
# 
# UWAGA: Musisz włączyć "Firestore Database" w swoim projekcie Firebase.
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
    if not FIREBASE_CREDS_JSON:
        print("❌ KRYTYCZNY BŁĄD: Brak zmiennej środowiskowej FIREBASE_CREDS_JSON.")
        # W trybie lokalnym, spróbuj załadować plik
        if os.path.exists('serviceAccountKey.json'):
            print("🔧 Znaleziono lokalny plik 'serviceAccountKey.json'. Używam go...")
            cred = credentials.Certificate('serviceAccountKey.json')
        else:
            raise ValueError("Brak FIREBASE_CREDS_JSON i serviceAccountKey.json")
    else:
        # Parsowanie JSON-a ze zmiennej środowiskowej
        creds_dict = json.loads(FIREBASE_CREDS_JSON)
        cred = credentials.Certificate(creds_dict)

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Pomyślnie połączono z Firestore.")
except Exception as e:
    print(f"❌ KRYTYCZNY BŁĄD: Nie można zainicjować Firebase: {e}")
    db = None

# --- Konfiguracja SerpAPI (dla S1, jeśli nadal potrzebne) ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
# NGRAM_API_URL już nie jest potrzebne, sami liczymy!


# -------------------------------------------------------------------
# ✅ KROK 2: Logika Parsowania Briefu
# -------------------------------------------------------------------
def parse_brief_to_keywords(brief_text):
    """
    Paruje brief (BASIC i EXTENDED) do struktury bazy danych.
    """
    keywords_dict = {}
    # Regex do znalezienia sekcji BASIC lub EXTENDED
    section_regex = r'(BASIC TEXT TERMS|EXTENDED TEXT TERMS):\s*={10,}\s*([\s\S]*?)(?=\n[A-Z\s]+ TERMS:|$)'
    # Regex do znalezienia linii ze słowem kluczowym
    keyword_regex = re.compile(r'^\s*(.*?):\s*(\d+)-(\d+)x\s*$', re.UNICODE)
    keyword_regex_single = re.compile(r'^\s*(.*?):\s*(\d+)x\s*$', re.UNICODE)

    for match in re.finditer(section_regex, brief_text, re.IGNORECASE):
        section_content = match.group(2)
        for line in section_content.splitlines():
            line = line.strip()
            if not line:
                continue

            kw_match = keyword_regex.match(line)
            if kw_match:
                keyword = kw_match.group(1).strip()
                min_val = int(kw_match.group(2))
                max_val = int(kw_match.group(3))
            else:
                kw_match_single = keyword_regex_single.match(line)
                if kw_match_single:
                    keyword = kw_match_single.group(1).strip()
                    min_val = int(kw_match_single.group(2))
                    max_val = int(kw_match_single.group(2)) # min i max są takie same
                else:
                    continue # Linia bez zakresu, ignorujemy (np. H2 HEADERS)

            # Zapisujemy stan początkowy i docelowy
            keywords_dict[keyword] = {
                "target_min": min_val,
                "target_max": max_val,
                "remaining_min": min_val,
                "remaining_max": max_val,
                "actual": 0,
                "locked": False # Do Twojej reguły max + 3
            }
            
    return keywords_dict

# -------------------------------------------------------------------
# ✅ KROK 3: Logika Hierarchicznego Liczenia (Kluczowy element)
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
    """
    Liczy słowa kluczowe hierarchicznie (od najdłuższego do najkrótszego).
    """
    text_lower = full_text.lower()
    
    # Sortujemy słowa kluczowe od najdłuższego do najkrótszego
    # To jest klucz do hierarchicznego liczenia
    sorted_keywords = sorted(keywords_dict.keys(), key=len, reverse=True)
    
    counts = {k: 0 for k in keywords_dict}
    
    # Tworzymy tekst-maskę, w którym będziemy "wycinać" znalezione frazy
    masked_text = text_lower
    
    for kw in sorted_keywords:
        kw_lower = kw.lower()
        
        # Używamy \b (word boundary) aby liczyć tylko całe słowa/frazy
        try:
            matches = re.findall(r'\b' + re.escape(kw_lower) + r'\b', masked_text)
            count = len(matches)
            counts[kw] = count
            
            # "Wycinamy" znalezione frazy z maski, aby nie policzyć ich podwójnie
            # (np. "prawnik" wewnątrz "prawnik rozwodowy")
            if count > 0:
                masked_text = re.sub(r'\b' + re.escape(kw_lower) + r'\b', "X" * len(kw), masked_text, count=count)
        except re.error as e:
            print(f"Błąd regex dla frazy '{kw}': {e}")
            continue

    return counts

# -------------------------------------------------------------------
# ✅ KROK 4: Nowe Endpointy (Architektura v5)
# -------------------------------------------------------------------

@app.route("/api/project/create", methods=["POST"])
def create_project():
    """
    Tworzy nowy projekt na podstawie briefu.
    Oczekuje briefu jako surowy tekst (plain text) w body.
    """
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        brief_text = request.data.decode('utf-8')
        if not brief_text:
            return jsonify({"error": "Brak briefu w body żądania."}), 400
            
        keywords_state = parse_brief_to_keywords(brief_text)
        
        if not keywords_state:
            return jsonify({"error": "Nie udało się sparsować słów kluczowych z briefu. Sprawdź format."}), 400
            
        # Tworzy nowy projekt w kolekcji 'seo_projects'
        doc_ref = db.collection('seo_projects').document()
        
        project_data = {
            "keywords_state": keywords_state,
            "full_text": "",
            "batches": []
        }
        doc_ref.set(project_data)
        
        return jsonify({
            "status": "Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state)
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500


@app.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    """
    Dodaje nowy batch tekstu do projektu, przelicza całość i zwraca raport.
    Oczekuje tekstu batcha jako surowy tekst (plain text) w body.
    """
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Projekt o podanym ID nie istnieje."}), 404
            
        project_data = doc.to_dict()
        current_keywords_state = project_data.get('keywords_state', {})
        current_full_text = project_data.get('full_text', "")
        
        batch_text = request.data.decode('utf-8')
        if not batch_text:
            return jsonify({"error": "Brak tekstu w body żądania."}), 400
            
        # Dodajemy nowy tekst do całości
        new_full_text = current_full_text + "\n\n" + batch_text
        
        # Przeliczamy USAGE na podstawie CAŁEGO tekstu
        new_counts = calculate_hierarchical_counts(new_full_text, current_keywords_state)
        
        report_for_gpt = []
        
        # Aktualizujemy stan i generujemy raport
        for keyword, state in current_keywords_state.items():
            
            # Sprawdzamy, czy fraza nie jest zablokowana
            if state.get('locked', False):
                report_for_gpt.append(f"{keyword}: LOCKED (Użyto max + 3)")
                continue

            state['actual'] = new_counts.get(keyword, 0)
            
            # Aktualizujemy pozostałe min/max
            state['remaining_min'] = max(0, state['target_min'] - state['actual'])
            state['remaining_max'] = max(0, state['target_max'] - state['actual'])
            
            status = "OK"
            
            # TWOJA REGUŁA: max + 3
            if state['actual'] >= state['target_max'] + 3:
                state['locked'] = True
                status = f"LOCKED (Użyto {state['actual']} / Cel: {state['target_max']}. Przekroczono o 3+)"
            elif state['actual'] > state['target_max']:
                status = f"OVER (Użyto {state['actual']} / Cel: {state['target_max']})"
            elif state['actual'] < state['target_min']:
                status = f"UNDER (Użyto {state['actual']} / Cel: {state['target_min']})"

            report_for_gpt.append(f"{keyword}: {state['actual']} użyto / Cel: {state['target_min']}-{state['target_max']} / Pozostało: {state['remaining_min']}-{state['remaining_max']} / Status: {status}")

        # Zapisujemy zaktualizowany stan w bazie
        doc_ref.update({
            "keywords_state": current_keywords_state,
            "full_text": new_full_text,
            "batches": firestore.ArrayUnion([batch_text])
        })
        
        # Zwracamy raport tekstowy dla GPT
        return jsonify(report_for_gpt), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id}/add_batch: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v5.0-firestore-stateful",
        "message": "Master SEO API (Firestore Edition) działa poprawnie."
    }), 200

# -------------------------------------------------------------------
# Stary endpoint S1 (opcjonalny, jeśli nadal go chcesz)
# Można go zintegrować, aby działał niezależnie
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    # ... (cały kod S1 można wkleić tutaj bez zmian, jeśli nadal go potrzebujesz) ...
    # ... (na razie pomijam dla czytelności, bo nie jest częścią logiki S3) ...
    return jsonify({"message": "S1 Analysis (do implementacji, jeśli potrzebne)"})
# -------------------------------------------------------------------

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

