"""
Export Routes - v29.0
Eksport artykułów do DOCX/HTML/TXT + Editorial Review (Claude API)

ZMIANY v29.0:
- DIFF-BASED: Claude zwraca tylko zmiany, nie przepisuje całości
- BURSTINESS VALIDATION: Sprawdza CV przed/po
- AUTO-ROLLBACK: Automatycznie przywraca oryginał jeśli CV spadło
- DIFF OUTPUT: Szczegółowa lista zmian applied/failed
- MANUAL ROLLBACK: Nowy endpoint /editorial_review/rollback

ZMIANY v27.1:
- rescan_keywords_after_editorial() - przelicza frazy po Claude review

ZMIANY v27.0:
- Zmiana z Gemini na Claude API
- Wplata nieużyte frazy BASIC/EXTENDED
"""

import os
import re
import io
import json
import statistics
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

# Firebase Firestore only
from firebase_admin import firestore

# Document generation
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Claude API for Editorial Review
import anthropic

# v27.1: Import keyword counter do rescan
try:
    from keyword_counter import count_keywords, count_keywords_for_state
    KEYWORD_COUNTER_OK = True
    print("[EXPORT] ✅ keyword_counter imported for rescan")
except ImportError:
    KEYWORD_COUNTER_OK = False
    print("[EXPORT] ⚠️ keyword_counter not available - rescan will use fallback")

# v29.0: Burstiness validation for editorial review
try:
    from ai_detection_metrics import calculate_burstiness
    BURSTINESS_CHECK_OK = True
    print("[EXPORT] ✅ Burstiness validation enabled")
except ImportError:
    BURSTINESS_CHECK_OK = False
    print("[EXPORT] ⚠️ Burstiness validation not available")

# v27.1: Ładowanie promptu z pliku (opcjonalne)
EDITORIAL_PROMPT_TEMPLATE = None
try:
    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "editorial_prompt.json")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
            EDITORIAL_PROMPT_TEMPLATE = prompt_data.get("full_prompt_assembled")
            print(f"[EXPORT] ✅ Loaded editorial prompt from file (v{prompt_data.get('version', '?')})")
except Exception as e:
    print(f"[EXPORT] ℹ️ Using built-in prompt (file load failed: {e})")

export_routes = Blueprint("export_routes", __name__)

# Claude config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
claude_client = None
if ANTHROPIC_API_KEY:
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("[EXPORT] ✅ Claude API configured")
else:
    print("[EXPORT] ⚠️ ANTHROPIC_API_KEY not set")

# Fallback: Gemini (jeśli Claude niedostępny)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        if not ANTHROPIC_API_KEY:
            print("[EXPORT] ℹ️ Using Gemini as fallback")
    except ImportError:
        genai = None


def html_to_text(html: str) -> str:
    """Konwertuje HTML na czysty tekst."""
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n\n## \1\n\n', html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n\n### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_article_structure(text: str) -> list:
    """Parsuje artykuł na strukturę nagłówków i paragrafów."""
    elements = []
    
    # Normalizuj format
    text = re.sub(r'^h2:\s*(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^h3:\s*(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    parts = re.split(r'(<h[23][^>]*>.*?</h[23]>)', text, flags=re.IGNORECASE | re.DOTALL)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        h2_match = re.match(r'<h2[^>]*>(.*?)</h2>', part, re.IGNORECASE | re.DOTALL)
        h3_match = re.match(r'<h3[^>]*>(.*?)</h3>', part, re.IGNORECASE | re.DOTALL)
        
        if h2_match:
            elements.append({"type": "h2", "content": h2_match.group(1).strip()})
        elif h3_match:
            elements.append({"type": "h3", "content": h3_match.group(1).strip()})
        else:
            clean_text = html_to_text(part)
            if clean_text:
                elements.append({"type": "paragraph", "content": clean_text})
    
    return elements


# ================================================================
# v27.0: HELPER - Składanie treści z batchów
# ================================================================
def extract_text_from_batches(batches: list) -> tuple:
    """
    Bezpieczne składanie treści z batchów.
    Obsługuje różne formaty: batch.text, batch.content, batch.html
    
    Returns:
        (full_text, debug_info)
    """
    texts = []
    debug_info = {
        "total_batches": len(batches),
        "batches_with_text": 0,
        "batches_empty": 0,
        "fields_found": set()
    }
    
    for i, batch in enumerate(batches):
        text = None
        
        for field in ["text", "content", "html", "batch_text", "raw_text"]:
            if field in batch and batch[field]:
                text = str(batch[field]).strip()
                debug_info["fields_found"].add(field)
                break
        
        if text:
            texts.append(text)
            debug_info["batches_with_text"] += 1
        else:
            debug_info["batches_empty"] += 1
            print(f"[EXPORT] ⚠️ Batch {i} jest pusty. Klucze: {list(batch.keys())}")
    
    debug_info["fields_found"] = list(debug_info["fields_found"])
    full_text = "\n\n".join(texts)
    
    return full_text, debug_info


# ================================================================
# v27.1: RESCAN KEYWORDS AFTER EDITORIAL
# ================================================================
def rescan_keywords_after_editorial(project_id: str, corrected_article: str) -> dict:
    """
    v27.1: Przelicza frazy od ZERA po edycji Claude.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return {"error": "Project not found", "rescanned": False}
    
    project_data = doc.to_dict()
    keywords_state = project_data.get("keywords_state", {})
    
    if not keywords_state:
        return {"error": "No keywords_state", "rescanned": False}
    
    if not corrected_article or len(corrected_article.strip()) < 50:
        return {"error": "No corrected_article to scan", "rescanned": False}
    
    # Zbierz wszystkie keywordy
    keywords = []
    rid_to_keyword = {}
    for rid, meta in keywords_state.items():
        kw = (meta.get("keyword") or "").strip()
        if kw:
            keywords.append(kw)
            rid_to_keyword[rid] = kw
    
    if not keywords:
        return {"error": "No keywords to scan", "rescanned": False}
    
    # Przelicz frazy
    changes = []
    new_counts = {}
    
    if KEYWORD_COUNTER_OK:
        new_counts = count_keywords_for_state(corrected_article, keywords_state, use_exclusive_for_nested=False)
        print(f"[RESCAN] Using keyword_counter for {len(keywords)} keywords")
    else:
        # Fallback - proste liczenie
        clean_text = re.sub(r'<[^>]+>', ' ', corrected_article.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        for rid, kw in rid_to_keyword.items():
            kw_lower = kw.lower()
            count = clean_text.count(kw_lower)
            new_counts[rid] = count
        print(f"[RESCAN] Using fallback counter for {len(keywords)} keywords")
    
    # Porównaj i zaktualizuj
    for rid, meta in keywords_state.items():
        old_actual = meta.get("actual_uses", 0)
        new_actual = new_counts.get(rid, 0)
        
        if old_actual != new_actual:
            kw = meta.get("keyword", "?")
            kw_type = meta.get("type", "BASIC")
            changes.append({
                "keyword": kw,
                "type": kw_type,
                "old": old_actual,
                "new": new_actual,
                "diff": new_actual - old_actual
            })
            print(f"[RESCAN] '{kw}' ({kw_type}): {old_actual} → {new_actual}")
        
        # NADPISZ actual_uses
        meta["actual_uses"] = new_actual
        
        # Przelicz status
        min_t = meta.get("target_min", 1 if meta.get("type", "").upper() == "BASIC" else 1)
        max_t = meta.get("target_max", 999)
        
        if new_actual < min_t:
            meta["status"] = "UNDER"
        elif new_actual == max_t:
            meta["status"] = "OPTIMAL"
        elif min_t <= new_actual < max_t:
            meta["status"] = "OK"
        else:
            meta["status"] = "OVER"
        
        meta["remaining_max"] = max(0, max_t - new_actual)
        keywords_state[rid] = meta
    
    # Zapisz do Firebase
    try:
        db.collection("seo_projects").document(project_id).update({
            "keywords_state": keywords_state,
            "last_rescan": {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "changes_count": len(changes),
                "trigger": "editorial_review"
            }
        })
        print(f"[RESCAN] ✅ Saved {len(changes)} changes to Firebase")
    except Exception as e:
        print(f"[RESCAN] ❌ Firebase save error: {e}")
        return {"error": str(e), "rescanned": False, "changes": changes}
    
    # Policz nowe coverage
    basic_total = 0
    basic_covered = 0
    extended_total = 0
    extended_covered = 0
    
    for rid, meta in keywords_state.items():
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        
        if kw_type == "BASIC" or kw_type == "MAIN":
            basic_total += 1
            if actual >= 1:
                basic_covered += 1
        elif kw_type == "EXTENDED":
            extended_total += 1
            if actual >= 1:
                extended_covered += 1
    
    basic_pct = round(100 * basic_covered / basic_total, 1) if basic_total > 0 else 100
    extended_pct = round(100 * extended_covered / extended_total, 1) if extended_total > 0 else 100
    
    return {
        "rescanned": True,
        "changes_count": len(changes),
        "changes": changes,
        "coverage_after_rescan": {
            "basic": f"{basic_covered}/{basic_total} ({basic_pct}%)",
            "extended": f"{extended_covered}/{extended_total} ({extended_pct}%)"
        },
        "missing_after_rescan": {
            "basic": [c["keyword"] for c in changes if c["type"].upper() in ["BASIC", "MAIN"] and c["new"] == 0],
            "extended": [c["keyword"] for c in changes if c["type"].upper() == "EXTENDED" and c["new"] == 0]
        }
    }


# ================================================================
# v29.0: DIFF PARSING HELPERS
# ================================================================
def parse_diff_response(response_text: str) -> list:
    """
    Parsuje odpowiedź Claude w formacie DIFF.
    
    Format wejściowy:
    [ZMIANA 1]
    ZNAJDŹ: "dokładny cytat"
    ZAMIEŃ: "poprawiona wersja"
    POWÓD: wyjaśnienie
    
    Zwraca: [{"find": "...", "replace": "...", "reason": "..."}]
    """
    changes = []
    
    # Wzorzec dla pojedynczej zmiany
    pattern = r'\[ZMIANA \d+\]\s*\n\s*ZNAJDŹ:\s*["\'](.+?)["\']\s*\n\s*ZAMIEŃ:\s*["\'](.+?)["\']\s*\n\s*POWÓD:\s*(.+?)(?=\n\s*\[ZMIANA|\Z)'
    
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        find_text = match[0].strip()
        replace_text = match[1].strip()
        reason = match[2].strip()
        
        if find_text and len(find_text) > 5:
            changes.append({
                "find": find_text,
                "replace": replace_text,
                "reason": reason,
                "applied": False
            })
    
    # Fallback: prostszy wzorzec
    if not changes:
        simple_pattern = r'ZNAJDŹ:\s*["\']?(.+?)["\']?\s*\n\s*ZAMIEŃ:\s*["\']?(.+?)["\']?\s*(?:\n|$)'
        simple_matches = re.findall(simple_pattern, response_text, re.DOTALL)
        
        for match in simple_matches:
            find_text = match[0].strip().strip('"\'')
            replace_text = match[1].strip().strip('"\'')
            
            if find_text and len(find_text) > 5:
                changes.append({
                    "find": find_text,
                    "replace": replace_text,
                    "reason": "auto-parsed",
                    "applied": False
                })
    
    return changes


def apply_diffs(original_text: str, changes: list) -> tuple:
    """
    Aplikuje zmiany diff do tekstu.
    
    Zwraca: (modified_text, applied_changes, failed_changes)
    """
    modified = original_text
    applied = []
    failed = []
    
    for change in changes:
        find_text = change["find"]
        replace_text = change["replace"]
        
        # Próba 1: Dokładne dopasowanie
        if find_text in modified:
            modified = modified.replace(find_text, replace_text, 1)
            change["applied"] = True
            applied.append(change)
            continue
        
        # Próba 2: Dopasowanie ignorujące whitespace
        find_normalized = ' '.join(find_text.split())
        text_lines = modified.split('\n')
        found = False
        
        for i, line in enumerate(text_lines):
            line_normalized = ' '.join(line.split())
            if find_normalized in line_normalized:
                # Zamień w tej linii
                text_lines[i] = line.replace(find_text.strip(), replace_text.strip())
                modified = '\n'.join(text_lines)
                change["applied"] = True
                change["match_type"] = "normalized"
                applied.append(change)
                found = True
                break
        
        if found:
            continue
        
        # Nie udało się
        change["applied"] = False
        change["error"] = "Fragment nie znaleziony w tekście"
        failed.append(change)
    
    return modified, applied, failed


def calculate_diff_stats(original: str, modified: str) -> dict:
    """Oblicza statystyki różnic między tekstami."""
    original_words = original.split()
    modified_words = modified.split()
    
    original_sentences = [s for s in re.split(r'[.!?]+', original) if s.strip()]
    modified_sentences = [s for s in re.split(r'[.!?]+', modified) if s.strip()]
    
    words_added = len(modified_words) - len(original_words)
    sentences_diff = len(modified_sentences) - len(original_sentences)
    
    # Procent zmian
    if len(original) > 0:
        changes = sum(1 for a, b in zip(original, modified) if a != b)
        changes += abs(len(original) - len(modified))
        change_percent = round((changes / len(original)) * 100, 1)
    else:
        change_percent = 100
    
    return {
        "words_original": len(original_words),
        "words_modified": len(modified_words),
        "words_diff": words_added,
        "sentences_original": len(original_sentences),
        "sentences_modified": len(modified_sentences),
        "sentences_diff": sentences_diff,
        "change_percent": min(change_percent, 100)
    }


# ================================================================
# EDITORIAL REVIEW - v29.0 (DIFF-BASED + BURSTINESS VALIDATION)
# ================================================================
@export_routes.post("/api/project/<project_id>/editorial_review")
def editorial_review(project_id):
    """
    v29.0: DIFF-BASED Editorial Review z walidacją burstiness.
    
    NOWOŚCI:
    - DIFF-BASED: Claude zwraca tylko zmiany, nie przepisuje całości
    - BURSTINESS VALIDATION: Sprawdza CV przed/po
    - AUTO-ROLLBACK: Automatycznie przywraca oryginał jeśli CV spadło
    - DIFF OUTPUT: Szczegółowa lista zmian
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    # Pobierz treść
    full_text = None
    debug_info = {}
    
    if project_data.get("full_article", {}).get("content"):
        full_text = project_data["full_article"]["content"]
        debug_info = {"source": "full_article"}
    else:
        batches = project_data.get("batches", [])
        if batches:
            full_text, debug_info = extract_text_from_batches(batches)
            debug_info["source"] = "batches"
    
    if not full_text:
        return jsonify({
            "error": "No content found",
            "hint": "Użyj save_full_article lub addBatch aby zapisać treść.",
            "project_keys": list(project_data.keys())
        }), 400
    
    # Sprawdź API
    if not claude_client and not genai:
        return jsonify({"error": "No AI API configured (need ANTHROPIC_API_KEY or GEMINI_API_KEY)"}), 500
    
    word_count = len(full_text.split()) if full_text else 0
    
    if word_count < 50:
        return jsonify({
            "error": f"Article too short for review ({word_count} words)",
            "debug": debug_info,
            "hint": "Użyj save_full_article lub addBatch aby zapisać treść."
        }), 400
    
    topic = project_data.get("topic") or project_data.get("main_keyword", "artykuł")
    
    # ================================================================
    # v29.0: BURSTINESS CHECK - PRZED
    # ================================================================
    burstiness_before = None
    if BURSTINESS_CHECK_OK:
        try:
            burstiness_before = calculate_burstiness(full_text)
            print(f"[EDITORIAL_REVIEW] 📊 Burstiness PRZED: CV={burstiness_before.get('cv', 0)}")
        except Exception as e:
            print(f"[EDITORIAL_REVIEW] ⚠️ Burstiness check failed: {e}")
    
    # v27.0: Pobierz nieużyte frazy BASIC i EXTENDED
    keywords_state = project_data.get("keywords_state", {})
    unused_basic = []
    unused_extended = []
    
    for rid, meta in keywords_state.items():
        kw = meta.get("keyword", "")
        kw_type = meta.get("type", "BASIC").upper()
        actual = meta.get("actual_uses", 0)
        
        if actual == 0 and kw:
            if kw_type == "BASIC":
                unused_basic.append(kw)
            elif kw_type == "EXTENDED":
                unused_extended.append(kw)
    
    # Sekcja nieużytych fraz dla promptu
    unused_keywords_section = ""
    if unused_basic or unused_extended:
        unused_keywords_section = "\n=== ⚠️ NIEWYKORZYSTANE FRAZY SEO ===\n"
        if unused_basic:
            unused_keywords_section += f"BASIC: {', '.join(unused_basic[:15])}\n"
        if unused_extended:
            unused_keywords_section += f"EXTENDED: {', '.join(unused_extended[:15])}\n"
    
    # 🆕 v44.5: YMYL context for editorial review
    detected_category = project_data.get("detected_category", "inne")
    ymyl_section = ""
    if detected_category == "prawo":
        ymyl_section = """
=== ⚖️ ARTYKUŁ PRAWNY (YMYL) ===
Ten artykuł dotyczy tematyki prawnej. Sprawdź SZCZEGÓLNIE:
1. Czy artykuł zawiera cytowania przepisów (art. X § Y ustawy Z)?
2. Czy są odwołania do orzeczeń sądowych (sygn. akt)?
3. Czy NIE MA wymyślonych sygnatur, dat ani przepisów?
4. Czy na końcu jest zastrzeżenie prawne (disclaimer)?
5. Jeśli brakuje cytowań — DODAJ je w sekcji "luki_tresciowe"!
"""
    elif detected_category == "medycyna":
        ymyl_section = """
=== 🏥 ARTYKUŁ MEDYCZNY (YMYL) ===
Ten artykuł dotyczy tematyki medycznej/zdrowotnej. Sprawdź SZCZEGÓLNIE:
1. Czy artykuł zawiera odwołania do publikacji naukowych?
2. Czy cytowane źródła istnieją (PMID, autorzy, rok)?
3. Czy NIE MA wymyślonych statystyk, badań ani leków?
4. Czy na końcu jest zastrzeżenie medyczne (disclaimer)?
5. Czy tekst nie zawiera niebezpiecznych rad zdrowotnych?
6. Jeśli brakuje cytowań — DODAJ je w sekcji "luki_tresciowe"!
"""

    try:
        analysis = None
        corrected_article = ""
        ai_model = "unknown"
        applied_changes = []
        failed_changes = []
        
        if claude_client:
            # ============================================================
            # WYWOŁANIE 1: RECENZJA REDAKTORSKA (v41.0)
            # ============================================================
            analysis_prompt = f"""Jesteś REDAKTOREM NACZELNYM polskiego wydawnictwa specjalizującego się w treściach eksperckich.

Dostajesz artykuł pt. "{topic}" ({word_count} słów). Napisz PEŁNĄ RECENZJĘ REDAKTORSKĄ.
{unused_keywords_section}
{ymyl_section}

=== NADRZĘDNA ZASADA ===
🔴 ROZBUDOWUJ treść, NIE USUWAJ. Każda Twoja sugestia powinna prowadzić do ROZWINIĘCIA tekstu, nie skrócenia go.

=== STRUKTURA RECENZJI (odpowiedz TYLKO JSON) ===

{{
  "overall_score": <0-10>,
  "scores": {{
    "merytoryka": <0-10>,
    "struktura": <0-10>,
    "styl": <0-10>,
    "seo": <0-10>
  }},

  "editorial_feedback": {{
    
    "recenzja_ogolna": "<3-5 zdań: ogólna ocena artykułu. Co jest mocne? Co wymaga pracy? Jaki ton ma tekst i czy jest odpowiedni? Czy artykuł odpowiada na pytanie czytelnika?>",
    
    "merytoryka": [
      {{
        "sekcja": "<H2/H3 którego dotyczy>",
        "uwaga": "<co jest nie tak merytorycznie>",
        "cytat": "<fragment tekstu — min 10 słów>",
        "sugestia": "<jak ROZBUDOWAĆ ten fragment — dodaj co konkretnie dopisać>"
      }}
    ],
    
    "styl_i_jezyk": [
      {{
        "problem": "powtórzenie|niezręczność|fraza_AI|kolokacja|strona_bierna|monotonia",
        "cytat": "<fragment z tekstu>",
        "sugestia": "<jak poprawić — rozwiń, nie skracaj>"
      }}
    ],
    
    "struktura_i_narracja": [
      {{
        "uwaga": "<problem ze strukturą: brak przejścia, nierówne proporcje sekcji, brak podsumowania, luka logiczna>",
        "gdzie": "<między którymi sekcjami / w której sekcji>",
        "sugestia": "<co DOPISAĆ żeby naprawić>"
      }}
    ],
    
    "luki_tresciowe": [
      {{
        "brakujacy_temat": "<czego czytelnik mógłby szukać, a tekst tego nie pokrywa>",
        "gdzie_dodac": "<w której sekcji najlepiej rozbudować>",
        "sugestia": "<2-3 zdania co konkretnie dopisać>"
      }}
    ],

    "halucynacje": [
      {{
        "cytat": "<fragment z wymyśloną statystyką/datą/źródłem>",
        "dlaczego_falsz": "<krótkie wyjaśnienie>"
      }}
    ]
  }},

  "errors_to_fix": [
    {{
      "type": "HALUCYNACJA|TERMINOLOGIA|FRAZA_AI|KOLOKACJA|STYL|ROZBUDUJ",
      "priority": <1-3 gdzie 1=krytyczne>,
      "original": "<cytat z tekstu — min 10 słów>",
      "replacement": "<poprawka — DŁUŻSZA lub RÓWNA oryginałowi>",
      "action": "ROZBUDUJ|POPRAW|USUŃ_HALUCYNACJĘ"
    }}
  ],

  "keywords_to_add": {json.dumps(unused_basic + unused_extended, ensure_ascii=False)},
  
  "summary": "<2-3 zdania: najważniejsze co trzeba zrobić z tym artykułem>"
}}

=== WSKAZÓWKI ===
- W "merytoryka" szukaj: brak źródeł, nieprecyzyjne twierdzenia, nadmierne uproszczenia, luki w argumentacji
- W "styl_i_jezyk" szukaj: powtórzenia słów w sąsiednich zdaniach, frazy AI ("warto zauważyć", "kluczowym elementem"), monotonny rytm zdań
- W "struktura_i_narracja" szukaj: sekcje za krótkie (<80 słów), brak płynnego przejścia, skok tematyczny
- W "luki_tresciowe" szukaj: pytania które czytelnik mógłby mieć, a tekst na nie nie odpowiada
- W "halucynacje" szukaj: konkretne liczby, daty, nazwy badań — czy brzmią wiarygodnie?
- Każda sugestia powinna mówić CO DOPISAĆ, nie co usunąć

=== ARTYKUŁ ({word_count} słów) ===

{full_text}"""

            print(f"[EDITORIAL_REVIEW] ========== CALL 1: RECENZJA REDAKTORSKA ==========")
            
            response1 = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6000,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            analysis_text = response1.content[0].text.strip()
            ai_model = "claude"
            
            # Parsuj JSON analizy
            try:
                clean = analysis_text
                if "```json" in clean:
                    clean = re.sub(r'```json\s*', '', clean)
                    clean = re.sub(r'```\s*$', '', clean)
                elif "```" in clean:
                    clean = re.sub(r'```\s*', '', clean)
                
                first_brace = clean.find('{')
                last_brace = clean.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    analysis = json.loads(clean[first_brace:last_brace + 1])
                    print(f"[EDITORIAL_REVIEW] ✅ Analysis parsed: score={analysis.get('overall_score')}")
            except Exception as e:
                print(f"[EDITORIAL_REVIEW] ⚠️ Analysis parse error: {e}")
                analysis = {"overall_score": 5, "summary": "Analiza nie sparsowana", "raw": analysis_text[:500]}
            
            # ============================================================
            # WYWOŁANIE 2: DIFF-BASED CORRECTION (v29.0)
            # ============================================================
            errors_list = analysis.get("errors_to_fix", []) if analysis else []
            keywords_to_add = analysis.get("keywords_to_add", []) if analysis else []
            
            diff_prompt = f"""Przeanalizuj artykuł pt. "{topic}" i zwróć TYLKO ZMIANY w formacie diff.

🔴 NADRZĘDNA ZASADA: ROZBUDOWUJ, NIE USUWAJ!
Artykuł ma {word_count} słów. Po Twoich zmianach musi mieć CO NAJMNIEJ tyle samo.
Każda zmiana powinna ROZSZERZAĆ treść — dodawać szczegóły, kontekst, przykłady.

⛔ KRYTYCZNE ZASADY:
- Zwróć MAX 15 zmian (tylko najważniejsze!)
- NIE przepisuj całego artykułu
- Krótkie zdania (2-5 słów) są CELOWE — NIE łącz ich!
- Zachowaj styl i rytm tekstu
- Cytat w ZNAJDŹ musi być DOKŁADNY (min 10 słów dla kontekstu)
- ZAMIEŃ musi być DŁUŻSZY lub RÓWNY co ZNAJDŹ (nigdy krótszy!)
- Jedyny wyjątek od zakazu usuwania: halucynacje (zmyślone dane/statystyki)

=== JAK ROZBUDOWYWAĆ ===
- Dodaj drugą część zdania po przecinku/myślniku z dodatkowym kontekstem
- Rozwiń ogólnik o konkretny przykład lub dane
- Wpleć brakującą frazę SEO jako naturalne dopowiedzenie
- Zamień pustą frazę AI na treść merytoryczną (nie usuwaj — zamień na coś wartościowego)

=== BŁĘDY DO POPRAWY ===
{json.dumps(errors_list[:10], ensure_ascii=False, indent=2) if errors_list else "Brak krytycznych błędów — rozbuduj słabsze fragmenty i wpleć brakujące frazy."}

=== FRAZY DO WPLECENIA (rozbudowując istniejące zdania!) ===
{', '.join(keywords_to_add[:10]) if keywords_to_add else "Wszystkie frazy są w tekście."}

=== FORMAT ODPOWIEDZI ===

[ZMIANA 1]
ZNAJDŹ: "dokładny cytat z artykułu (min 10 słów dla kontekstu)"
ZAMIEŃ: "rozbudowana wersja z zachowaniem stylu — DŁUŻSZA niż oryginał"
POWÓD: krótkie wyjaśnienie (max 10 słów)

[ZMIANA 2]
ZNAJDŹ: "..."
ZAMIEŃ: "..."
POWÓD: ...

(kontynuuj do max 15 zmian)

=== ARTYKUŁ DO ANALIZY ({word_count} słów — nie zmniejszaj!) ===

{full_text}"""

            print(f"[EDITORIAL_REVIEW] ========== CALL 2: DIFF-BASED CORRECTION ==========")
            
            response2 = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": diff_prompt}]
            )
            diff_response = response2.content[0].text.strip()
            
            # Parsuj diffy
            changes = parse_diff_response(diff_response)
            print(f"[EDITORIAL_REVIEW] 📝 Parsed {len(changes)} changes from Claude response")
            
            # Aplikuj diffy
            if changes:
                corrected_article, applied_changes, failed_changes = apply_diffs(full_text, changes)
                print(f"[EDITORIAL_REVIEW] ✅ Applied: {len(applied_changes)}, ❌ Failed: {len(failed_changes)}")
            else:
                corrected_article = full_text
                print(f"[EDITORIAL_REVIEW] ℹ️ No changes parsed, using original text")
            
        elif genai:
            # Gemini fallback (stary sposób - przepisuje całość)
            print(f"[EDITORIAL_REVIEW] Using Gemini (fallback) for project {project_id}")
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            old_prompt = f"""Jesteś REDAKTOREM. Popraw artykuł i zwróć JSON:
{{"analysis": {{"overall_score": <0-10>, "summary": "<podsumowanie>"}}, "corrected_article": "<cały poprawiony tekst>"}}

Artykuł ({word_count} słów):
{full_text[:10000]}"""
            
            response = model.generate_content(old_prompt)
            review_text = response.text.strip()
            ai_model = "gemini"
            
            try:
                clean = review_text
                if "```json" in clean:
                    clean = re.sub(r'```json\s*', '', clean)
                    clean = re.sub(r'```\s*$', '', clean)
                first_brace = clean.find('{')
                last_brace = clean.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    data = json.loads(clean[first_brace:last_brace + 1])
                    analysis = data.get("analysis", {})
                    corrected_article = data.get("corrected_article", "")
            except:
                analysis = {"overall_score": 5, "summary": "Gemini parse error"}
                corrected_article = full_text
        else:
            return jsonify({
                "error": "No AI API configured",
                "hint": "Set ANTHROPIC_API_KEY or GEMINI_API_KEY"
            }), 500
        
        # Fallback jeśli brak corrected_article
        if not corrected_article or len(corrected_article) < 100:
            print(f"[EDITORIAL_REVIEW] ⚠️ Using original text as fallback")
            corrected_article = full_text
        
        # ================================================================
        # v41.0: WORD COUNT GUARD — artykuł nie powinien się skrócić
        # ================================================================
        corrected_wc = len(corrected_article.split())
        word_count_shrinkage = False
        shrinkage_pct = 0
        
        if corrected_wc < word_count and corrected_article != full_text:
            shrinkage_pct = round((1 - corrected_wc / word_count) * 100, 1)
            print(f"[EDITORIAL_REVIEW] ⚠️ WORD COUNT SHRINKAGE: {word_count} → {corrected_wc} (-{shrinkage_pct}%)")
            
            if shrinkage_pct > 5:
                # Skrócenie o >5% = rollback do oryginału
                word_count_shrinkage = True
                print(f"[EDITORIAL_REVIEW] 🔴 AUTO-ROLLBACK: artykuł skrócił się o {shrinkage_pct}% — przywracam oryginał")
                corrected_article = full_text
        
        # ================================================================
        # v29.0: BURSTINESS CHECK - PO + AUTO-ROLLBACK
        # ================================================================
        burstiness_after = None
        rollback_triggered = False
        rollback_reason = None
        
        if BURSTINESS_CHECK_OK and burstiness_before and corrected_article != full_text:
            try:
                burstiness_after = calculate_burstiness(corrected_article)
                cv_before = burstiness_before.get("cv", 0)
                cv_after = burstiness_after.get("cv", 0)
                
                print(f"[EDITORIAL_REVIEW] 📊 Burstiness PO: CV={cv_after}")
                
                # ROLLBACK jeśli CV spadło o więcej niż 0.1 LUB spadło poniżej 0.3
                cv_drop = cv_before - cv_after
                
                if cv_drop > 0.1:
                    rollback_triggered = True
                    rollback_reason = f"CV spadło o {cv_drop:.2f} (z {cv_before:.2f} do {cv_after:.2f}) — tekst stał się zbyt monotonny"
                    print(f"[EDITORIAL_REVIEW] ⚠️ AUTO-ROLLBACK: {rollback_reason}")
                    corrected_article = full_text
                    
                elif cv_after < 0.3 and cv_before >= 0.3:
                    rollback_triggered = True
                    rollback_reason = f"CV spadło poniżej progu AI ({cv_after:.2f} < 0.3) — tekst wygląda na wygenerowany przez AI"
                    print(f"[EDITORIAL_REVIEW] ⚠️ AUTO-ROLLBACK: {rollback_reason}")
                    corrected_article = full_text
                    
            except Exception as e:
                print(f"[EDITORIAL_REVIEW] ⚠️ Burstiness after check failed: {e}")
        
        # v41.0: Rollback z powodu skrócenia artykułu
        if word_count_shrinkage and not rollback_triggered:
            rollback_triggered = True
            rollback_reason = f"Artykuł skrócił się o {shrinkage_pct}% ({word_count} → {corrected_wc} słów) — editorial review powinien ROZBUDOWYWAĆ, nie skracać"
        
        # Statystyki diff
        diff_stats = calculate_diff_stats(full_text, corrected_article)
        
        # v28.1: GRAMMAR CORRECTION (jeśli nie było rollbacku)
        grammar_stats = {"fixes": 0, "removed": []}
        if not rollback_triggered:
            try:
                from grammar_checker import full_correction
                grammar_result = full_correction(corrected_article)
                corrected_article = grammar_result["corrected"]
                grammar_stats = {
                    "fixes": grammar_result["grammar_fixes"],
                    "removed": grammar_result["phrases_removed"],
                    "backend": grammar_result.get("backend", "unknown")
                }
                if grammar_stats["fixes"] > 0 or grammar_stats["removed"]:
                    print(f"[EDITORIAL_REVIEW] ✅ Grammar: {grammar_stats['fixes']} fixes, {len(grammar_stats['removed'])} phrases removed")
            except ImportError:
                print(f"[EDITORIAL_REVIEW] ⚠️ grammar_checker not available, skipping grammar correction")
            except Exception as e:
                print(f"[EDITORIAL_REVIEW] ⚠️ Grammar correction error: {e}")
        
        corrected_word_count = len(corrected_article.split())
        
        # Zapisz w Firestore
        db.collection("seo_projects").document(project_id).update({
            "editorial_review": {
                "analysis": analysis,
                "corrected_article": corrected_article,
                "original_article": full_text,  # v29.0: Zawsze zachowuj oryginał!
                "original_word_count": word_count,
                "corrected_word_count": corrected_word_count,
                "unused_keywords": {
                    "basic": unused_basic,
                    "extended": unused_extended
                },
                # v29.0: Nowe pola
                "applied_changes": applied_changes[:20],
                "failed_changes": failed_changes[:10],
                "burstiness_before": burstiness_before,
                "burstiness_after": burstiness_after,
                "rollback_triggered": rollback_triggered,
                "rollback_reason": rollback_reason,
                "diff_stats": diff_stats,
                "grammar_correction": grammar_stats,
                "ai_model": ai_model,
                "version": "v41.0-expand",
                "created_at": firestore.SERVER_TIMESTAMP
            }
        })
        
        # v27.1: RESCAN KEYWORDS
        rescan_result = {"rescanned": False, "reason": "no_corrected_article"}
        if not rollback_triggered and corrected_article and len(corrected_article.strip()) > 50:
            print(f"[EDITORIAL_REVIEW] Running rescan_keywords for project {project_id}...")
            rescan_result = rescan_keywords_after_editorial(project_id, corrected_article)
            print(f"[EDITORIAL_REVIEW] Rescan result: {rescan_result.get('changes_count', 0)} changes")
        
        return jsonify({
            "status": "REVIEWED",
            "version": "v41.0-DIFF-EXPAND",
            "overall_score": analysis.get("overall_score") if analysis else None,
            "scores": analysis.get("scores", {}) if analysis else {},
            "summary": analysis.get("summary", "") if analysis else "",
            
            # 🆕 v41.0: PEŁNA RECENZJA REDAKTORSKA
            "editorial_feedback": analysis.get("editorial_feedback", {}) if analysis else {},
            
            # v29.0: DIFF OUTPUT
            "diff_result": {
                "total_changes_parsed": len(applied_changes) + len(failed_changes),
                "applied": len(applied_changes),
                "failed": len(failed_changes),
                "applied_changes": applied_changes[:10],
                "failed_changes": failed_changes[:5]
            },
            "diff_stats": diff_stats,
            
            # v29.0: BURSTINESS VALIDATION
            "burstiness_check": {
                "enabled": BURSTINESS_CHECK_OK,
                "before": {
                    "cv": burstiness_before.get("cv") if burstiness_before else None,
                    "value": burstiness_before.get("value") if burstiness_before else None,
                    "status": burstiness_before.get("status") if burstiness_before else None
                },
                "after": {
                    "cv": burstiness_after.get("cv") if burstiness_after else None,
                    "value": burstiness_after.get("value") if burstiness_after else None,
                    "status": burstiness_after.get("status") if burstiness_after else None
                },
                "cv_change": round((burstiness_after.get("cv", 0) - burstiness_before.get("cv", 0)), 3) if (burstiness_before and burstiness_after) else None
            },
            
            # v41.0: WORD COUNT GUARD
            "word_count_guard": {
                "original": word_count,
                "corrected": corrected_word_count,
                "shrinkage_pct": shrinkage_pct,
                "rollback_triggered": word_count_shrinkage,
                "policy": "Editorial review powinien ROZBUDOWYWAĆ treść, nie skracać"
            },
            
            # v29.0: ROLLBACK INFO
            "rollback": {
                "triggered": rollback_triggered,
                "reason": rollback_reason,
                "original_preserved": True
            },
            
            "corrected_article": corrected_article,
            "word_count": {
                "original": word_count,
                "corrected": corrected_word_count
            },
            "unused_keywords_input": {
                "basic": unused_basic,
                "extended": unused_extended
            },
            "ai_model": ai_model,
            "grammar_correction": grammar_stats,
            "rescan_result": rescan_result,
            
            # 🆕 v44.5: YMYL validation
            "ymyl_check": {
                "detected_category": detected_category,
                "is_ymyl": detected_category in ("prawo", "medycyna"),
                "ymyl_reviewed": detected_category in ("prawo", "medycyna")
            } if detected_category in ("prawo", "medycyna") else None
        }), 200
        
    except Exception as e:
        print(f"[EDITORIAL_REVIEW] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Review failed: {str(e)}",
            "word_count": word_count,
            "debug": debug_info
        }), 500


# ================================================================
# v29.0: MANUAL ROLLBACK ENDPOINT
# ================================================================
@export_routes.post("/api/project/<project_id>/editorial_review/rollback")
def editorial_rollback(project_id):
    """
    v29.0: Ręczny rollback do oryginalnego tekstu.
    Użyj gdy automatyczny rollback nie zadziałał lub chcesz przywrócić oryginał.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    editorial = project_data.get("editorial_review", {})
    
    original = editorial.get("original_article")
    if not original:
        return jsonify({
            "error": "No original article saved",
            "hint": "Original article is only available after editorial_review v29.0+"
        }), 400
    
    current = editorial.get("corrected_article", "")
    
    if original == current:
        return jsonify({
            "status": "NO_CHANGE",
            "message": "Tekst już jest oryginalny (poprzedni rollback lub brak zmian)"
        }), 200
    
    # Przywróć oryginał
    db.collection("seo_projects").document(project_id).update({
        "editorial_review.corrected_article": original,
        "editorial_review.manual_rollback": True,
        "editorial_review.rollback_at": firestore.SERVER_TIMESTAMP,
        "editorial_review.pre_rollback_article": current
    })
    
    return jsonify({
        "status": "ROLLED_BACK",
        "message": "Przywrócono oryginalny tekst",
        "word_count": {
            "original": len(original.split()),
            "was_corrected": len(current.split())
        }
    }), 200


# ================================================================
# EXPORT STATUS
# ================================================================
@export_routes.get("/api/project/<project_id>/export_status")
def export_status(project_id):
    """Sprawdza status projektu przed eksportem."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    full_text = None
    debug_info = {"source": "none"}
    batches_count = 0
    
    if project_data.get("full_article", {}).get("content"):
        full_text = project_data["full_article"]["content"]
        debug_info = {"source": "full_article"}
    else:
        batches = project_data.get("batches", [])
        batches_count = len(batches)
        if batches:
            full_text, debug_info = extract_text_from_batches(batches)
            debug_info["source"] = "batches"
    
    word_count = len(full_text.split()) if full_text else 0
    
    editorial = project_data.get("editorial_review", {})
    
    return jsonify({
        "project_id": project_id,
        "topic": project_data.get("topic", ""),
        "has_content": bool(full_text),
        "word_count": word_count,
        "batches_count": batches_count,
        "content_source": debug_info.get("source", "none"),
        "editorial_review": {
            "done": bool(editorial),
            "score": editorial.get("analysis", {}).get("overall_score") if editorial else None,
            "version": editorial.get("version", "unknown"),
            "rollback_triggered": editorial.get("rollback_triggered", False),
            "burstiness_cv_before": editorial.get("burstiness_before", {}).get("cv") if editorial.get("burstiness_before") else None,
            "burstiness_cv_after": editorial.get("burstiness_after", {}).get("cv") if editorial.get("burstiness_after") else None
        },
        "available_exports": {
            "docx": f"/api/project/{project_id}/export/docx",
            "html": f"/api/project/{project_id}/export/html",
            "txt": f"/api/project/{project_id}/export/txt"
        },
        "next_steps": {
            "editorial_review": f"POST /api/project/{project_id}/editorial_review",
            "export_docx": f"GET /api/project/{project_id}/export/docx",
            "export_html": f"GET /api/project/{project_id}/export/html"
        }
    }), 200


# ================================================================
# EXPORT DOCX
# ================================================================
@export_routes.get("/api/project/<project_id>/export/docx")
def export_docx(project_id):
    """Eksportuje artykuł do formatu DOCX."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    # v32.4: Priorytet źródeł treści
    # 1. editorial_review.corrected_article (po Claude review)
    # 2. full_article.content (zapisany przez save_full_article)
    # 3. batches (surowe batche)
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
        print(f"[EXPORT_DOCX] Using corrected article ({len(corrected.split())} words)")
    elif project_data.get("full_article", {}).get("content"):
        full_text = project_data["full_article"]["content"]
        print(f"[EXPORT_DOCX] Using full_article ({len(full_text.split())} words)")
    else:
        batches = project_data.get("batches", [])
        full_text, _ = extract_text_from_batches(batches)
        print(f"[EXPORT_DOCX] Using batches ({len(full_text.split()) if full_text else 0} words)")
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    # 🆕 v44.5: YMYL disclaimer guard
    detected_category = project_data.get("detected_category", "inne")
    if detected_category == "prawo" and "zastrzeżenie" not in full_text.lower():
        full_text += (
            "\n\n---\n\n"
            "**Zastrzeżenie prawne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i nie stanowi porady prawnej. W indywidualnych sprawach zalecamy konsultację "
            "z wykwalifikowanym prawnikiem."
        )
        print(f"[EXPORT_DOCX] ⚖️ Auto-dodano brakujący disclaimer prawny")
    elif detected_category == "medycyna" and "zastrzeżenie" not in full_text.lower():
        full_text += (
            "\n\n---\n\n"
            "**Zastrzeżenie medyczne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i edukacyjny. Nie stanowi porady medycznej ani nie zastępuje konsultacji "
            "z lekarzem lub innym wykwalifikowanym specjalistą."
        )
        print(f"[EXPORT_DOCX] 🏥 Auto-dodano brakujący disclaimer medyczny")
    
    topic = project_data.get("topic", "Artykuł")
    elements = parse_article_structure(full_text)
    
    # Twórz dokument
    document = Document()
    
    # Tytuł
    title = document.add_heading(topic, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Treść
    for el in elements:
        if el["type"] == "h2":
            document.add_heading(el["content"], level=2)
        elif el["type"] == "h3":
            document.add_heading(el["content"], level=3)
        elif el["type"] == "paragraph":
            p = document.add_paragraph(el["content"])
            p.style.font.size = Pt(11)
    
    # Zapisz do bufora
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    
    # Bezpieczna nazwa pliku
    polish_map = str.maketrans({
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    })
    safe_topic = topic.translate(polish_map)
    filename = re.sub(r'[^\w\s-]', '', safe_topic)[:50] + ".docx"
    
    from urllib.parse import quote
    filename_utf8 = quote(re.sub(r'[^\w\s-]', '', topic)[:50] + ".docx")
    
    return Response(
        buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename={filename}; filename*=UTF-8''{filename_utf8}"
        }
    )


# ================================================================
# EXPORT HTML
# ================================================================
@export_routes.get("/api/project/<project_id>/export/html")
def export_html(project_id):
    """Eksportuje artykuł do formatu HTML."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
    elif project_data.get("full_article", {}).get("content"):
        full_text = project_data["full_article"]["content"]
    else:
        batches = project_data.get("batches", [])
        full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    # 🆕 v44.5: YMYL disclaimer guard (HTML export)
    detected_category = project_data.get("detected_category", "inne")
    if detected_category == "prawo" and "zastrzeżenie" not in full_text.lower():
        full_text += (
            "\n\n---\n\n"
            "**Zastrzeżenie prawne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i nie stanowi porady prawnej. W indywidualnych sprawach zalecamy konsultację "
            "z wykwalifikowanym prawnikiem."
        )
    elif detected_category == "medycyna" and "zastrzeżenie" not in full_text.lower():
        full_text += (
            "\n\n---\n\n"
            "**Zastrzeżenie medyczne:** Niniejszy artykuł ma charakter wyłącznie informacyjny "
            "i edukacyjny. Nie stanowi porady medycznej ani nie zastępuje konsultacji "
            "z lekarzem lub innym wykwalifikowanym specjalistą."
        )
    
    topic = project_data.get("topic", "Artykuł")
    
    # Konwertuj markery na HTML
    html = full_text
    html = re.sub(r'^h2:\s*(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^h3:\s*(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Paragrafy
    lines = html.split('\n')
    processed = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('<h'):
            if line.startswith('- ') or line.startswith('• '):
                processed.append(f'<li>{line[2:]}</li>')
            else:
                processed.append(f'<p>{line}</p>')
        else:
            processed.append(line)
    
    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>{topic}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; margin-top: 30px; }}
        h3 {{ color: #555; }}
        p {{ margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>{topic}</h1>
    {''.join(processed)}
</body>
</html>"""
    
    polish_map = str.maketrans({
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    })
    safe_topic = topic.translate(polish_map)
    filename = re.sub(r'[^\w\s-]', '', safe_topic)[:50] + ".html"
    
    from urllib.parse import quote
    filename_utf8 = quote(re.sub(r'[^\w\s-]', '', topic)[:50] + ".html")
    
    return Response(
        html_content,
        mimetype="text/html",
        headers={
            "Content-Disposition": f"attachment; filename={filename}; filename*=UTF-8''{filename_utf8}"
        }
    )


# ================================================================
# EXPORT TXT
# ================================================================
@export_routes.get("/api/project/<project_id>/export/txt")
def export_txt(project_id):
    """Eksportuje artykuł do formatu TXT."""
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
    elif project_data.get("full_article", {}).get("content"):
        full_text = project_data["full_article"]["content"]
    else:
        batches = project_data.get("batches", [])
        full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
    topic = project_data.get("topic", "Artykuł")
    
    # Konwertuj do czytelnego tekstu
    txt = f"{topic}\n{'='*len(topic)}\n\n"
    txt += re.sub(r'^h2:\s*(.+)$', r'\n## \1\n', full_text, flags=re.MULTILINE)
    txt = re.sub(r'^h3:\s*(.+)$', r'\n### \1\n', txt, flags=re.MULTILINE)
    
    polish_map = str.maketrans({
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    })
    safe_topic = topic.translate(polish_map)
    filename = re.sub(r'[^\w\s-]', '', safe_topic)[:50] + ".txt"
    
    from urllib.parse import quote
    filename_utf8 = quote(re.sub(r'[^\w\s-]', '', topic)[:50] + ".txt")
    
    return Response(
        txt,
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}; filename*=UTF-8''{filename_utf8}"
        }
    )


# ================================================================
# ALIASY dla kompatybilności
# ================================================================
@export_routes.post("/api/project/<project_id>/gemini_review")
def gemini_review_alias(project_id):
    """Alias do editorial_review dla kompatybilności"""
    return editorial_review(project_id)


@export_routes.post("/api/project/<project_id>/claude_review")
def claude_review_alias(project_id):
    """Alias do editorial_review"""
    return editorial_review(project_id)
