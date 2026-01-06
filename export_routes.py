"""
Export Routes - v27.1
Eksport artykułów do DOCX/HTML/TXT + Editorial Review (Claude API)

ZMIANY v27.1:
- NOWE: rescan_keywords_after_editorial() - przelicza frazy po Claude review
- Naprawia bug gdzie keywords_state nie był aktualizowany po edycji Claude
- Teraz raport pokazuje RZECZYWISTE pokrycie fraz, nie stare dane

ZMIANY v27.0:
- Zmiana z Gemini na Claude API (dokładniejsza analiza)
- Uniwersalny prompt (bez specyficznych przykładów branżowych)
- Zwraca analizę + poprawiony tekst
- Wplata nieużyte frazy BASIC/EXTENDED
- Wymusza minimalną długość tekstu (nie skraca)
"""

import os
import re
import io
import json
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
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    if not ANTHROPIC_API_KEY:
        print("[EXPORT] ℹ️ Using Gemini as fallback")


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
    
    PROBLEM który to rozwiązuje:
    - S3 (approve_batch) zlicza frazy → actual_uses = X
    - S5 (Claude) może USUNĄĆ frazy z tekstu
    - ALE keywords_state nie był aktualizowany!
    - Raport pokazywał "100% coverage" mimo że frazy zniknęły
    
    ROZWIĄZANIE:
    - Po Claude review przeliczamy WSZYSTKIE frazy od zera
    - Nadpisujemy actual_uses RZECZYWISTYMI wartościami
    - Teraz raport i NeuronWriter się zgadzają!
    
    Returns:
        dict z wynikami rescanu i listą zmian
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
        # Użyj keyword_counter (z lemmatyzacją)
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
# EDITORIAL REVIEW - v27.0 (Claude API + poprawiony tekst)
# ================================================================
@export_routes.post("/api/project/<project_id>/editorial_review")
def editorial_review(project_id):
    """
    v27.0: Weryfikacja przez Claude jako redaktora naczelnego.
    Zwraca: analiza + poprawiony tekst + wplecione nieużyte frazy.
    """
    db = firestore.client()
    doc = db.collection("seo_projects").document(project_id).get()
    
    if not doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    project_data = doc.to_dict()
    batches = project_data.get("batches", [])
    
    if not batches:
        return jsonify({
            "error": "No batches found",
            "hint": "Artykuł nie ma zapisanych batchów. Użyj addBatch aby zapisać treść.",
            "project_keys": list(project_data.keys())
        }), 400
    
    # Sprawdź czy mamy API
    if not claude_client and not GEMINI_API_KEY:
        return jsonify({"error": "No AI API configured (need ANTHROPIC_API_KEY or GEMINI_API_KEY)"}), 500
    
    # Składanie treści z batchów
    full_text, debug_info = extract_text_from_batches(batches)
    
    # Sprawdź czy mamy treść
    word_count = len(full_text.split()) if full_text else 0
    
    if word_count < 50:
        return jsonify({
            "error": f"Article too short for review ({word_count} words)",
            "debug": debug_info,
            "hint": "Batche istnieją ale nie zawierają tekstu w polu 'text'. Sprawdź strukturę batchów.",
            "sample_batch_keys": list(batches[0].keys()) if batches else []
        }), 400
    
    # Pobierz temat z projektu
    topic = project_data.get("topic") or project_data.get("main_keyword", "artykuł")
    
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
        unused_keywords_section = "\n=== ⚠️ NIEWYKORZYSTANE FRAZY SEO - MUSISZ JE WPLEŚĆ! ===\n"
        if unused_basic:
            unused_keywords_section += f"\n**BASIC (KRYTYCZNE - muszą być w tekście min. 1x):**\n"
            unused_keywords_section += "\n".join(f"• {kw}" for kw in unused_basic)
            unused_keywords_section += "\n"
        if unused_extended:
            unused_keywords_section += f"\n**EXTENDED (dodaj naturalnie 1x każdą):**\n"
            unused_keywords_section += "\n".join(f"• {kw}" for kw in unused_extended)
            unused_keywords_section += "\n"
        unused_keywords_section += "\nINSTRUKCJA: Wpleć te frazy NATURALNIE w poprawiony tekst. Rozbuduj istniejące akapity lub dodaj nowe zdania.\n"
    
    try:
        # ============================================================
        # v27.1: PROMPT Z PLIKU LUB WBUDOWANY
        # ============================================================
        if EDITORIAL_PROMPT_TEMPLATE:
            # Użyj promptu z pliku editorial_prompt.json
            prompt = EDITORIAL_PROMPT_TEMPLATE.format(
                topic=topic,
                word_count=word_count,
                full_text=full_text,
                unused_keywords_section=unused_keywords_section
            )
            print(f"[EDITORIAL_REVIEW] Using prompt from file")
        else:
            # Fallback - wbudowany prompt
            prompt = f"""Jesteś REDAKTOREM NACZELNYM i EKSPERTEM MERYTORYCZNYM w tematyce artykułu.

Otrzymujesz artykuł pt. "{topic}" do weryfikacji i korekty.
{unused_keywords_section}
=== TWOJE ZADANIA ===

**ZADANIE 1: ZNAJDŹ I NAPRAW BŁĘDY KRYTYCZNE**

1. **HALUCYNACJE DANYCH** - Zmyślone statystyki, daty, nazwy raportów → USUŃ lub zamień na ogólne stwierdzenie
2. **BŁĘDY TERMINOLOGICZNE** - Mylenie pojęć fachowych specyficznych dla tematu artykułu → POPRAW
3. **ZABURZONA CHRONOLOGIA** - Nielogiczna kolejność kroków/instrukcji → PRZESTAW prawidłowo
4. **NADMIERNE UPROSZCZENIA** - "zawsze/każdy/nigdy/wszyscy" → "zazwyczaj/często/w większości przypadków"
5. **NIEAKTUALNE INFORMACJE** - Przestarzałe dane, przepisy, procedury → USUŃ lub zaznacz że wymaga weryfikacji

**ZADANIE 2: POPRAW JAKOŚĆ TEKSTU**

- Usuń typowe frazy AI: "w dzisiejszych czasach", "warto wiedzieć", "nie ulega wątpliwości", "należy pamiętać"
- Popraw błędne kolokacje
- Usuń powtórzenia i rozwlekłe fragmenty

**ZADANIE 3: WYMAGANIA KRYTYCZNE**

⚠️ DŁUGOŚĆ: Tekst MUSI mieć MINIMUM {word_count} słów! Nie skracaj - możesz tylko wydłużyć!
⚠️ FORMAT: Zachowaj strukturę HTML/Markdown (H2, H3, paragrafy)
⚠️ LINKI: NIE dodawaj żadnych linków

=== ODPOWIEDZ W FORMACIE JSON ===

{{
  "analysis": {{
    "overall_score": <0-10>,
    "scores": {{"merytoryka": <0-10>, "struktura": <0-10>, "styl": <0-10>, "seo": <0-10>}},
    "critical_errors_found": [
      {{"type": "<TYP>", "original": "<cytat>", "fixed": "<poprawka>"}}
    ],
    "keywords_added": ["<wplecione frazy>"],
    "minor_fixes": ["<drobne poprawki>"],
    "summary": "<2-3 zdania>"
  }},
  "corrected_article": "<CAŁY POPRAWIONY ARTYKUŁ - MINIMUM {word_count} SŁÓW!>"
}}

=== ARTYKUŁ DO WERYFIKACJI ({word_count} słów) ===

{full_text}
"""
            print(f"[EDITORIAL_REVIEW] Using built-in prompt")
        
        # v27.1: DEBUG - loguj prompt i odpowiedź
        print(f"[EDITORIAL_REVIEW] ========== PROMPT START ==========")
        print(f"[EDITORIAL_REVIEW] Prompt length: {len(prompt)} chars")
        print(f"[EDITORIAL_REVIEW] First 500 chars: {prompt[:500]}")
        print(f"[EDITORIAL_REVIEW] ========== PROMPT END ==========")
        
        # v27.0: Użyj Claude API (preferowany) lub Gemini (fallback)
        if claude_client:
            print(f"[EDITORIAL_REVIEW] Using Claude API for project {project_id} ({word_count} words)")
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            review_text = response.content[0].text.strip()
            ai_model = "claude"
            
            # v27.1: DEBUG - loguj odpowiedź
            print(f"[EDITORIAL_REVIEW] ========== RESPONSE START ==========")
            print(f"[EDITORIAL_REVIEW] Response length: {len(review_text)} chars")
            print(f"[EDITORIAL_REVIEW] First 1000 chars: {review_text[:1000]}")
            print(f"[EDITORIAL_REVIEW] ========== RESPONSE END ==========")
            
        elif genai:
            print(f"[EDITORIAL_REVIEW] Using Gemini (fallback) for project {project_id}")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            review_text = response.text.strip()
            ai_model = "gemini"
        else:
            return jsonify({
                "error": "No AI API configured",
                "hint": "Set ANTHROPIC_API_KEY or GEMINI_API_KEY"
            }), 500
        
        # v27.1: Ulepszone parsowanie JSON z odpowiedzi
        review_data = None
        parse_error = None
        
        try:
            # 1. Usuń markdown code blocks jeśli są
            clean_text = review_text
            if "```json" in clean_text:
                clean_text = re.sub(r'```json\s*', '', clean_text)
                clean_text = re.sub(r'```\s*$', '', clean_text)
            elif "```" in clean_text:
                clean_text = re.sub(r'```\s*', '', clean_text)
            
            # 2. Znajdź JSON - szukaj od pierwszego { do ostatniego }
            first_brace = clean_text.find('{')
            last_brace = clean_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = clean_text[first_brace:last_brace + 1]
                review_data = json.loads(json_str)
                print(f"[EDITORIAL_REVIEW] ✅ JSON parsed successfully")
            else:
                parse_error = "No valid JSON braces found"
                print(f"[EDITORIAL_REVIEW] ⚠️ {parse_error}")
                
        except json.JSONDecodeError as e:
            parse_error = f"JSON decode error: {str(e)}"
            print(f"[EDITORIAL_REVIEW] ⚠️ {parse_error}")
            # Próba naprawy - czasem Claude dodaje przecinek na końcu
            try:
                # Usuń trailing comma przed }
                fixed_json = re.sub(r',\s*}', '}', json_str)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                review_data = json.loads(fixed_json)
                print(f"[EDITORIAL_REVIEW] ✅ JSON parsed after fix")
                parse_error = None
            except:
                pass
        except Exception as e:
            parse_error = f"Parse error: {str(e)}"
            print(f"[EDITORIAL_REVIEW] ⚠️ {parse_error}")
        
        # Fallback jeśli parsowanie nie powiodło się
        if review_data is None:
            review_data = {
                "raw_response": review_text[:2000],
                "parse_error": parse_error
            }
            print(f"[EDITORIAL_REVIEW] Using raw response fallback")
        
        # Wyciągnij analysis i corrected_article
        analysis = review_data.get("analysis", {})
        if not analysis and "raw_response" in review_data:
            analysis = {
                "overall_score": 0,
                "summary": "Parsowanie odpowiedzi nie powiodło się",
                "parse_error": review_data.get("parse_error")
            }
        
        corrected_article = review_data.get("corrected_article", "")
        
        # v27.1: Jeśli nie ma corrected_article, spróbuj wyciągnąć z raw_response
        if not corrected_article and "raw_response" in review_data:
            raw = review_data.get("raw_response", "")
            if "corrected_article" in raw:
                match = re.search(r'"corrected_article"\s*:\s*"([\s\S]*?)"(?=\s*})', raw)
                if match:
                    corrected_article = match.group(1)
                    print(f"[EDITORIAL_REVIEW] Extracted corrected_article from raw ({len(corrected_article)} chars)")
        
        corrected_word_count = len(corrected_article.split()) if corrected_article else 0
        
        # Zapisz w Firestore
        db.collection("seo_projects").document(project_id).update({
            "editorial_review": {
                "analysis": analysis,
                "corrected_article": corrected_article,
                "original_word_count": word_count,
                "corrected_word_count": corrected_word_count,
                "unused_keywords": {
                    "basic": unused_basic,
                    "extended": unused_extended
                },
                "ai_model": ai_model,
                "created_at": firestore.SERVER_TIMESTAMP
            }
        })
        
        # v27.1: RESCAN KEYWORDS - przelicz frazy od zera po edycji Claude!
        rescan_result = {"rescanned": False, "reason": "no_corrected_article"}
        if corrected_article and len(corrected_article.strip()) > 50:
            print(f"[EDITORIAL_REVIEW] Running rescan_keywords for project {project_id}...")
            rescan_result = rescan_keywords_after_editorial(project_id, corrected_article)
            print(f"[EDITORIAL_REVIEW] Rescan result: {rescan_result.get('changes_count', 0)} changes")
        
        return jsonify({
            "status": analysis.get("recommendation", "REVIEWED"),
            "overall_score": analysis.get("overall_score"),
            "scores": analysis.get("scores", {}),
            "critical_errors": analysis.get("critical_errors_found", []),
            "keywords_added": analysis.get("keywords_added", []),
            "minor_fixes": analysis.get("minor_fixes", []),
            "summary": analysis.get("summary", ""),
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
            "version": "v27.1",
            "rescan_result": rescan_result  # v27.1: wynik przeliczenia fraz
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
    batches = project_data.get("batches", [])
    
    full_text, debug_info = extract_text_from_batches(batches)
    word_count = len(full_text.split()) if full_text else 0
    
    editorial = project_data.get("editorial_review", {})
    
    return jsonify({
        "project_id": project_id,
        "topic": project_data.get("topic"),
        "status": project_data.get("status", "UNKNOWN"),
        "batches_count": len(batches),
        "word_count": word_count,
        "debug": debug_info,
        "editorial_review": {
            "done": bool(editorial),
            "score": editorial.get("analysis", {}).get("overall_score"),
            "corrected_word_count": editorial.get("corrected_word_count"),
            "ai_model": editorial.get("ai_model")
        },
        "ready_for_export": word_count >= 500 and debug_info["batches_with_text"] > 0,
        "actions": {
            "editorial_review": f"POST /api/project/{project_id}/editorial_review",
            "export_docx": f"GET /api/project/{project_id}/export/docx",
            "export_html": f"GET /api/project/{project_id}/export/html",
            "export_txt": f"GET /api/project/{project_id}/export/txt"
        },
        "version": "v27.0"
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
        print(f"[EXPORT_DOCX] Using corrected article ({len(corrected.split())} words)")
    else:
        batches = project_data.get("batches", [])
        full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
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
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".docx"
    
    return Response(
        buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
    else:
        batches = project_data.get("batches", [])
        full_text, _ = extract_text_from_batches(batches)
    
    if not full_text:
        return jsonify({"error": "No content to export"}), 400
    
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
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".html"
    
    return Response(
        html_content,
        mimetype="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
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
    
    # v27.0: Użyj poprawionego tekstu jeśli istnieje
    editorial = project_data.get("editorial_review", {})
    corrected = editorial.get("corrected_article", "")
    
    if corrected:
        full_text = corrected
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
    
    filename = re.sub(r'[^\w\s-]', '', topic)[:50] + ".txt"
    
    return Response(
        txt,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
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
