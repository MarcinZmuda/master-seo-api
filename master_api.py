import os
import re
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter # ✅ DODANO IMPORT

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
# ✅ DODANO NGRAM API POTRZEBNE DLA S1
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"


# -------------------------------------------------------------------
# ✅ DODANO FUNKCJE POMOCNICZE DLA S1
# -------------------------------------------------------------------
def call_api_with_json(url, payload, name):
    """Uniwersalna funkcja POST JSON z obsługą błędów."""
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd API {name}: {e}")
        return {"error": f"Błąd połączenia z {name}", "details": str(e)}

def call_serpapi(topic):
    """Pobiera wyniki z SerpAPI."""
    params = {"api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"}
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Błąd SerpAPI: {e}")
        return None

def call_langextract(url):
    """Pobiera tekst z URL przy użyciu LangExtract."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract")

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


# -------------------------------------------------------------------
# ✅ NOWY ENDPOINT: Czyszczenie bazy danych po zakończeniu pracy
# -------------------------------------------------------------------
@app.route("/api/project/<project_id>", methods=["DELETE"])
def delete_project(project_id):
    """
    Usuwa projekt (dokument) z bazy danych po zakończeniu pracy.
    """
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    try:
        doc_ref = db.collection('seo_projects').document(project_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "Projekt o podanym ID nie istnieje."}), 404
        
        # Usuń dokument
        doc_ref.delete()
        
        return jsonify({"status": f"Projekt {project_id} został pomyślnie usunięty."}), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id} [DELETE]: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500
# -------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v5.1-firestore-cleanup", # Zaktualizowano wersję
        "message": "Master SEO API (Firestore Edition) działa poprawnie."
    }), 200

# -------------------------------------------------------------------
# ✅ PRZYWRÓCONO PEŁNY KOD ENDPOINTU S1
# -------------------------------------------------------------------
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    """
    Wykonuje pełną analizę konkurencji (S1), niezależną od logiki S3.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Brak danych JSON"}), 400
        topic = data.get("topic")
        if not topic:
            return jsonify({"error": "Brak 'topic'"}), 400
    except Exception:
         return jsonify({"error": "Nieprawidłowy format JSON"}), 400

    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Błąd pobierania danych z SerpApi"}), 502

    top_urls = [res.get("link") for res in serp_data.get("organic_results", [])[:5]]

    source_log, h2_counts, all_headings, combined_text = [], [], [], ""
    successful_sources = 0

    for url in top_urls:
        if successful_sources >= 3:
            break
        content = call_langextract(url)
        if content and content.get("content"):
            successful_sources += 1
            h2s = content.get("h2", [])
            all_headings.extend([h.strip().lower() for h in h2s])
            h2_counts.append(len(h2s))
            combined_text += content.get("content", "") + "\n\n"
            source_log.append({"url": url, "status": "Success", "h2_count": len(h2s)})
        else:
            source_log.append({"url": url, "status": "Failure", "error": content.get("error")})

    if not combined_text:
        # Zabezpieczenie, gdy żaden URL nie zwrócił treści
        print("❌ Błąd S1: Nie udało się pobrać treści z żadnego URL-a.")
        return jsonify({
            "identified_urls": top_urls,
            "processing_report": source_log,
            "error": "Nie udało się pobrać treści z żadnego z top 5 URL-i.",
            "serp_features": {
                "ai_overview": serp_data.get("ai_overview"),
                "people_also_ask": serp_data.get("related_questions"),
                "featured_snippets": serp_data.get("answer_box")
            }
        }), 502

    if all_headings:
        heading_counts = Counter(all_headings)
        top_10_headings = [heading for heading, count in heading_counts.most_common(10)]
    else:
        top_10_headings = []

    # Wywołanie API Ngram do wzbogacenia S1
    ngram_data = call_api_with_json(
        NGRAM_API_URL,
        {"text": combined_text, "main_keyword": topic},
        "Ngram API (S1)"
    )

    return jsonify({
        "identified_urls": top_urls,
        "processing_report": source_log,
        "competitive_metrics": {
            "avg_h2_per_article": round(sum(h2_counts) / len(h2_counts), 1) if h2_counts else 0,
            "min_h2": min(h2_counts) if h2_counts else 0,
            "max_h2": max(h2_counts) if h2_counts else 0,
        },
        "top_competitor_headings": top_10_headings,
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box")
        },
        "s1_enrichment": ngram_data
    })
# -------------------------------------------------------------------

# --- Uruchomienie ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

