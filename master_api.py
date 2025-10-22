import os
import re
import requests
import random
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- Inicjalizacja ---
load_dotenv()
app = Flask(__name__)

# --- Konfiguracja ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"
NGRAM_API_URL = "https://gpt-ngram-api.onrender.com/api/ngram_entity_analysis"
SYNTHESIZE_API_URL = "https://gpt-ngram-api.onrender.com/api/synthesize_topics"
COMPLIANCE_API_URL = "https://gpt-ngram-api.onrender.com/api/generate_compliance_report"

# --- 1. Globalne zasady i persona dla LLM ---
GLOBAL_PROMPT_RULES = """
<PERSONA>
Jesteś Lead SEO Content Architect: precyzyjny w strategii, ludzki w narracji.
Znasz SEO techniczne, semantyczne i językowe. Piszesz jak człowiek, nie jak AI.
</PERSONA>

<CORE_RULES>
- Używaj tonu analityczno-narracyjnego. Artykuł ma być wciągający, ale precyzyjny.
- Każda sekcja H2 musi mieć inną liczbę akapitów niż poprzednia (1-3 akapity), ale nadrzędne jest pokrycie tematu.
- Minimalna długość akapitu: co najmniej 6 pełnych, rozbudowanych zdań.
- Tekst musi zachować narracyjną spójność i rytm oraz być poprawnie napisany w języku polskim.
- Unikaj klisz AI i powtórzeń fraz.
</CORE_RULES>
"""

# --- Helpery API ---
def call_api_with_json(url, payload, name):
    """Pomocnik do wywoływania innych API z obsługą błędów."""
    try:
        r = requests.post(url, json=payload, timeout=40)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return {"error": f"Nie udało się połączyć z {name}", "details": str(e)}

def call_llm(prompt):
    """Wywołuje API OpenAI do generowania tekstu."""
    if not OPENAI_API_KEY:
        return {"error": "Brak klucza OPENAI_API_KEY"}
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Błąd OpenAI API: {e}")
        return f"Błąd podczas generowania tekstu: {e}"

def call_serpapi(topic):
    params = {"api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"}
    try:
        r = requests.get(SERPAPI_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ Błąd SerpAPI:", e)
        return None

def call_langextract(url):
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

# --- Endpoint: S1 ANALYSIS ("Czarna Skrzynka") ---
@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Brak parametru 'topic'"}), 400
    
    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Nie udało się pobrać danych z SerpApi"}), 502

    organic_results = serp_data.get("organic_results", [])
    top_5_urls = [res.get("link") for res in organic_results[:5]]

    successful_sources, source_processing_log, h2_count_list = [], [], []
    combined_text_content = ""

    for url in top_5_urls:
        if len(successful_sources) >= 3:
            break
        content = call_langextract(url)
        if content and not content.get("error") and content.get("content"):
            current_text = content.get("content", "")
            combined_text_content += current_text + "\n\n"
            h2s = content.get("h2", [])
            h2_count_list.append(len(h2s))
            successful_sources.append(url)
            source_processing_log.append({"url": url, "status": "Success", "h2_count": len(h2s), "length": len(current_text)})
        else:
            source_processing_log.append({"url": url, "status": "Failure", "reason": content.get("error", "Brak treści")})

    avg_h2 = sum(h2_count_list) / len(h2_count_list) if h2_count_list else 0
    
    ngram_payload = {"text": combined_text_content, "main_keyword": topic}
    ngram_data = call_api_with_json(NGRAM_API_URL, ngram_payload, "Ngram API")
    
    return jsonify({
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "competitive_metrics": {
            "avg_h2_per_article": round(avg_h2, 1),
            "min_h2": min(h2_count_list) if h2_count_list else 0,
            "max_h2": max(h2_count_list) if h2_count_list else 0
        },
        "serp_features": {
            "people_also_ask": serp_data.get("related_questions")
        },
        "s1_enrichment": {
            "entities": ngram_data.get("entities"),
            "ngrams": ngram_data.get("ngrams"),
            "error": ngram_data.get("error") 
        }
    })

# --- NOWY ENDPOINT: S2-S4 GENEROWANIE ARTYKUŁU ---
@app.route("/api/generate_article", methods=["POST"])
def generate_article():
    input_data = request.get_json()

    # --- S2: Planowanie ---
    # --- 2. Parsowanie briefu H2 z wytycznymi ---
    h2_sections = []
    for line in input_data.get("brief_h2", "").split('\n'):
        if line.strip():
            parts = line.split(' - ', 1)
            title = parts[0].strip()
            guidelines = parts[1].strip() if len(parts) > 1 else "Napisz merytoryczną treść dla tej sekcji."
            h2_sections.append({"title": title, "guidelines": guidelines})

    master_keyword_string = input_data.get("brief_basic", "")
    keywords_to_track = parse_keyword_string(master_keyword_string)
    for kw in keywords_to_track:
        keywords_to_track[kw]['used'] = 0

    ngrams_data = input_data.get("s1_enrichment", {}).get("ngrams", {})
    top_ngrams = []
    if ngrams_data:
        for key in ["2gram", "3gram", "4gram"]:
            if key in ngrams_data:
                top_ngrams.extend([item["ngram"] for item in ngrams_data[key][:5]])
    
    full_article_text = ""
    
    # --- S3: Iteracyjne Pisanie ---
    print("✍️ Generowanie wstępu...")
    intro_prompt = f"""
        {GLOBAL_PROMPT_RULES}
        <TASK>
        Napisz angażujący wstęp do artykułu na temat: "{input_data.get('topic', '')}".
        Wpleć naturalnie 2-3 z poniższych n-gramów.
        </TASK>
        <NGRAMS_PL>
        {chr(10).join(top_ngrams[:5])}
        </NGRAMS_PL>
        <STRUCTURE_RULES>
        - Dokładnie 1 akapit.
        - Minimum 6 pełnych zdań.
        </STRUCTURE_RULES>
    """
    introduction = call_llm(intro_prompt)
    full_article_text += introduction + "\n\n"

    last_paragraph_count = 1
    for section in h2_sections:
        title = section["title"]
        guidelines = section["guidelines"]
        print(f"✍️ Generowanie sekcji: {title}...")
        
        keywords_block_list = []
        exhausted_block_list = []
        for kw, data in keywords_to_track.items():
            remaining = data['max_allowed'] - data['used']
            if remaining > 0:
                keywords_block_list.append(f"{kw} (pozostało: {remaining})")
            else:
                exhausted_block_list.append(kw)

        keywords_block = "<KEYWORDS_PL>\n" + "\n".join(keywords_block_list) + "\n</KEYWORDS_PL>" if keywords_block_list else "<KEYWORDS_PL>\n(brak)\n</KEYWORDS_PL>"
        exhausted_note = "<NOTE_PL>\nNie możesz już używać tych fraz:\n" + "\n".join(exhausted_block_list) + "\n</NOTE_PL>" if exhausted_block_list else ""
        
        possible_counts = [c for c in [1, 2, 3] if c != last_paragraph_count]
        paragraph_count = random.choice(possible_counts) if possible_counts else 2
        last_paragraph_count = paragraph_count

        section_prompt = f"""
            {GLOBAL_PROMPT_RULES}
            <CHAPTER_TITLE_PL>
            {title}
            </CHAPTER_TITLE_PL>
            <WRITING_GUIDELINES_PL>
            {guidelines}
            </WRITING_GUIDELINES_PL>
            {keywords_block}
            <NGRAMS_PL>
            {chr(10).join(top_ngrams)}
            </NGRAMS_PL>
            {exhausted_note}
            <STRUCTURE_RULES>
            - Liczba akapitów: dokładnie {paragraph_count}.
            </STRUCTURE_RULES>
        """
        
        section_text = call_llm(section_prompt)
        full_article_text += f"## {title}\n{section_text}\n\n"

        verification_report = verify_s3_keywords_internal(full_article_text, master_keyword_string)
        for kw, report_data in verification_report.get("keyword_report", {}).items():
            if kw in keywords_to_track:
                keywords_to_track[kw]['used'] = report_data['used']

    # --- S4: Humanizing ---
    print("✍️ Redakcja humanizująca (S4)...")
    humanize_prompt = f"""
        {GLOBAL_PROMPT_RULES}
        <TASK>
        Przeprowadź redakcję humanizującą (framework HEAR) na poniższym tekście. Popraw rytm, spójność i ton, aby brzmiał w pełni naturalnie. Nie zmieniaj merytoryki. Zwróć tylko poprawiony tekst, bez dodatkowych komentarzy.
        </TASK>
        <TEXT_TO_EDIT>
        {full_article_text}
        </TEXT_TO_EDIT>
    """
    final_article = call_llm(humanize_prompt)

    return jsonify({"final_article": final_article})

# Funkcja wewnętrzna do weryfikacji
def verify_s3_keywords_internal(text, keywords_with_ranges):
    keywords = parse_keyword_string(keywords_with_ranges)
    text_lower = text.lower()
    report = {}
    for phrase, rng in keywords.items():
        count = text_lower.count(phrase)
        status = "OK"
        if count < rng["min_allowed"]: status = "UNDER"
        elif count > rng["max_allowed"]: status = "OVER"
        report[phrase] = { "used": count, "status": status }
    return {"keyword_report": report}

# --- Pozostałe Endpointy (Bez zmian) ---
@app.route("/api/h2_distribution", methods=["POST"])
def h2_distribution():
    # ... (bez zmian)
    pass

def parse_keyword_string(keyword_data):
    if isinstance(keyword_data, dict):
        return keyword_data
    result = {}
    pattern = re.compile(r"^\s*(.+?)\s*(?:\((\d+)\s*-\s*(\d+)\))?\s*$")
    for line in keyword_data.splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if match:
            phrase, min_val, max_val = match.groups()
            phrase = phrase.strip().lower()
            min_allowed = int(min_val) if min_val else 1
            max_allowed = int(max_val) if max_val else 5
            result[phrase] = {
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "allowed_range": f"{min_allowed}-{max_allowed}"
            }
    return result

@app.route("/api/s3_verify_keywords", methods=["POST"])
def verify_s3_keywords():
    data = request.get_json()
    text = data.get("text")
    keywords_with_ranges = data.get("keywords_with_ranges")
    if not isinstance(text, str) or not keywords_with_ranges:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges'"}), 400
    
    report = verify_s3_keywords_internal(text, keywords_with_ranges)
    keywords = parse_keyword_string(keywords_with_ranges)
    for phrase, data in report.get("keyword_report", {}).items():
        if phrase in keywords:
            data.update(keywords[phrase])
            
    return jsonify(report)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "✅ OK", "version": "3.6-full-automation", "message": "master_api działa poprawnie"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 3000), debug=True)

