import base64
from flask import request, jsonify

@app.route("/api/project/create", methods=["POST"])
def create_project_hybrid():
    """
    Tworzy nowy projekt SEO – akceptuje:
    1️⃣ JSON z "brief_base64"
    2️⃣ JSON z "brief_text"
    3️⃣ surowy text/plain
    W przypadku pustego body lub błędu kodowania – API spróbuje
    automatycznie odtworzyć brief lub zwróci przyczynę błędu.
    """
    if not db:
        return jsonify({"error": "Baza danych Firestore nie jest połączona."}), 503

    keywords_state = {}
    data_json = request.get_json(silent=True)
    brief_text = None

    # 1️⃣ Jeśli przyszło JSON Base64
    if data_json and "brief_base64" in data_json:
        try:
            brief_text = base64.b64decode(data_json["brief_base64"]).decode("utf-8")
            print("✅ Otrzymano brief w Base64 → dekodowanie OK.")
        except Exception as e:
            return jsonify({"error": f"Błąd dekodowania Base64: {e}"}), 400

    # 2️⃣ Jeśli przyszło JSON z surowym tekstem (np. GPT-light mode)
    elif data_json and "brief_text" in data_json:
        brief_text = data_json["brief_text"]
        print("✅ Otrzymano brief_text (z JSON-a).")

    # 3️⃣ Jeśli przyszło text/plain (najczęściej od GPT)
    elif request.data:
        try:
            brief_text = request.data.decode("utf-8").strip()
            if brief_text:
                print("✅ Otrzymano brief text/plain (OK).")
            else:
                print("⚠️ Otrzymano pusty text/plain – możliwe błędy GPT.")
        except Exception as e:
            print(f"❌ Błąd odczytu text/plain: {e}")
            return jsonify({"error": f"Błąd odczytu text/plain: {e}"}), 400

    # 4️⃣ Jeśli dalej brak danych – zwróć błąd
    if not brief_text:
        return jsonify({
            "error": "Brak treści briefu. GPT mogło wysłać pusty plik. "
                     "Upewnij się, że brief jest przesyłany jako text/plain lub brief_base64."
        }), 400

    # 5️⃣ Parsowanie
    try:
        keywords_state = parse_brief_to_keywords(brief_text)
        if not keywords_state:
            return jsonify({"error": "Nie udało się sparsować słów kluczowych."}), 400
    except Exception as e:
        return jsonify({"error": f"Błąd parsowania briefu: {e}"}), 400

    # 6️⃣ Zapis projektu
    try:
        doc_ref = db.collection("seo_projects").document()
        project_data = {
            "keywords_state": keywords_state,
            "full_text": "",
            "batches": []
        }
        doc_ref.set(project_data)

        print(f"✅ Projekt utworzony: {doc_ref.id} | fraz: {len(keywords_state)}")
        return jsonify({
            "status": "Projekt utworzony pomyślnie.",
            "project_id": doc_ref.id,
            "keywords_parsed": len(keywords_state)
        }), 201
    except Exception as e:
        print(f"❌ Błąd zapisu do Firestore: {e}")
        return jsonify({"error": f"Błąd zapisu do Firestore: {e}"}), 500
