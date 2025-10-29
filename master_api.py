import os
import re
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# -------------------------------------------------------------------
# KROK 1: Konfiguracja Firebase (Firestore)
# -------------------------------------------------------------------
try:
    FIREBASE_CREDS_JSON = os.getenv("FIREBASE_CREDS_JSON")
    if not FIREBASE_CREDS_JSON:
        print("❌ KRYTYCZNY BŁĄD: Brak zmiennej środowiskowej FIREBASE_CREDS_JSON.")
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

# -------------------------------------------------------------------
# Konfiguracja API i funkcje pomocnicze (dla S1)
# -------------------------------------------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"

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
# KROK 2: Parser JSON (Teraz jedyna metoda)
# -------------------------------------------------------------------
def parse_brief_from_json(data_json):
    """
    Paruje brief z gotowego formatu JSON.
    Oczekuje: {"basic_terms": {"fraza": "1-5x"}, "extended_terms": ...}
    """
    keywords_state = {}
    
    # Źródła terminów (np. basic_terms, extended_terms)
    terms_sources = [
        data_json.get('basic_terms', {}),
        data_json.get('extended_terms', {})
    ]

    for terms_dict in terms_sources:
        if not isinstance(terms_dict, dict):
            # Jeśli "basic_terms" nie jest słownikiem (np. jest listą), pomiń
            continue
            
        for keyword, limits in terms_dict.items():
            try:
                # Czyszczenie formatu limitu (usuwa spacje, 'x', zamienia na małe litery)
                limits_clean = str(limits).strip().lower().replace(' ', '')
                
                if '-' in limits_clean:
                    min_val_str, max_val_str = limits_clean.replace('x', '').split('-')
                    min_val = int(min_val_str)
                    max_val = int(max_val_str)
                else:
                    min_val = max_val = int(limits_clean.replace('x', ''))
                
                keywords_state[keyword.strip()] = {
                    "target_min": min_val,
                    "target_max": max_val,
                    "remaining_min": min_val,
                    "remaining_max": max_val,
                    "actual": 0,
                    "locked": False
                }
            except Exception as e:
                # Jeśli format limitu jest zły (np. "jeden do pięciu"), pomijamy
                print(f"Błąd parsowania limitu JSON '{limits}' dla frazy '{keyword}': {e}")
                continue
                
    return keywords_state

# -------------------------------------------------------------------
# KROK 3: Logika Hierarchicznego Liczenia
# -------------------------------------------------------------------
def calculate_hierarchical_counts(full_text, keywords_dict):
    """
    Liczy słowa kluczowe hierarchicznie (od najdłuższego do najkrótszego).
    """
    text_lower = full_text.lower()
    
    # Sortujemy słowa kluczowe od najdłuższego do najkrótszego
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
            if count > 0:
                masked_text = re.sub(r'\b' + re.escape(kw_lower) + r'\b', "X" * len(kw), masked_text, count=count)
        except re.error as e:
            print(f"Błąd regex dla frazy '{kw}': {e}")
            continue

    return counts

# -------------------------------------------------------------------
# ✅ KROK 4: Endpoint `/api/project/create` (Tylko JSON)
# -------------------------------------------------------------------
@app.route("/api/project/create", methods=["POST"])
def create_project_json_only():
    """
    Tworzy nowy projekt. Akceptuje WYŁĄCZNIE application/json.
    """
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    # Wymuś sprawdzenie nagłówka Content-Type
    if not request.is_json:
        return jsonify({"error": "Błąd: Oczekiwano nagłówka 'Content-Type: application/json'."}), 415 # Unsupported Media Type

    data_json = request.get_json()
    if not data_json:
        return jsonify({"error": "Brak danych JSON w body."}), 400

    # Przetwarzamy dane JSON na nasz format bazy danych
    keywords_state = parse_brief_from_json(data_json)

    if not keywords_state:
        return jsonify({
            "error": "Nie udało się sparsować słów kluczowych. Sprawdź format JSON: 'basic_terms' i 'extended_terms' muszą być słownikami (obiektami), np. {'fraza': '1-5x'}."
        }), 400

    # Zapisujemy projekt w Firestore
    try:
        doc_ref = db.collection('seo_projects').document()
        project_data = {
            "keywords_state": keywords_state,
            "full_text": "",
            "batches": [],
            # Zapiszmy też H2, jeśli przyszły w JSON-ie
            "h2_terms": data_json.get('h2_terms', [])
        }
        doc_ref.set(project_data)
        
        return jsonify({
            "status": "Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state)
        }), 201

    except Exception as e:
        print(f"❌ Błąd /api/project/create (zapis Firestore): {e}")
        return jsonify({"error": f"Wystąpił błąd serwera przy zapisie: {e}"}), 500

# -------------------------------------------------------------------
# KROK 5: Pozostałe Endpointy
# -------------------------------------------------------------------

@app.route("/api/project/<project_id>/add_batch", methods=["POST"])
def add_batch_to_project(project_id):
    """
    Dodaje nowy batch tekstu (plain text) do projektu, przelicza całość.
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
            
        new_full_text = current_full_text + "\n\n" + batch_text
        
        # Przeliczamy USAGE na podstawie CAŁEGO tekstu
        new_counts = calculate_hierarchical_counts(new_full_text, current_keywords_state)
        
        report_for_gpt = []
        
        # Aktualizujemy stan i generujemy raport
        for keyword, state in current_keywords_state.items():
            
            if state.get('locked', False):
                report_for_gpt.append(f"{keyword}: LOCKED (Użyto max + 3)")
                continue

            state['actual'] = new_counts.get(keyword, 0)
            
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
        
        return jsonify(report_for_gpt), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id}/add_batch: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500


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
        
        doc_ref.delete()
        
        return jsonify({"status": f"Projekt {project_id} został pomyślnie usunięty."}), 200

    except Exception as e:
        print(f"❌ Błąd /api/project/{project_id} [DELETE]: {e}")
        return jsonify({"error": f"Wystąpił błąd serwera: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v6.0-json-only", # Zaktualizowano wersję
        "message": "Master SEO API (Firestore Edition) działa poprawnie."
    }), 200


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
