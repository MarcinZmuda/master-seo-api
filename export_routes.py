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
    
    # v32.4: Priorytet źródeł treści
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
    
    # Sprawdź czy mamy API
    if not claude_client and not GEMINI_API_KEY:
        return jsonify({"error": "No AI API configured (need ANTHROPIC_API_KEY or GEMINI_API_KEY)"}), 500
    
    # v32.4: full_text już mamy z wcześniejszego kodu
    word_count = len(full_text.split()) if full_text else 0
    
    if word_count < 50:
        return jsonify({
            "error": f"Article too short for review ({word_count} words)",
            "debug": debug_info,
            "hint": "Użyj save_full_article lub addBatch aby zapisać treść."
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
        # v27.2: DWA WYWOŁANIA CLAUDE - osobno analiza, osobno tekst
        # ============================================================
        
        # WYWOŁANIE 1: ANALIZA
        analysis_prompt = f"""Jesteś REDAKTOREM NACZELNYM i EKSPERTEM MERYTORYCZNYM.

Przeanalizuj artykuł pt. "{topic}" i znajdź błędy do poprawy.
{unused_keywords_section}

=== ZADANIA ANALIZY ===

1. **HALUCYNACJE DANYCH** - Zmyślone statystyki, daty, nazwy raportów
2. **BŁĘDY TERMINOLOGICZNE** - Mylenie pojęć fachowych
3. **ZABURZONA CHRONOLOGIA** - Nielogiczna kolejność kroków
4. **NADMIERNE UPROSZCZENIA** - "zawsze/każdy/nigdy" zamiast "zazwyczaj/często"
5. **FRAZY AI** - "w dzisiejszych czasach", "warto wiedzieć", "nie ulega wątpliwości"
6. **BŁĘDNE KOLOKACJE** - "robić decyzję" zamiast "podejmować decyzję"

=== ODPOWIEDZ TYLKO JSON ===

{{
  "overall_score": <0-10>,
  "scores": {{"merytoryka": <0-10>, "struktura": <0-10>, "styl": <0-10>, "seo": <0-10>}},
  "errors_to_fix": [
    {{"type": "<TYP>", "original": "<cytat z tekstu>", "replacement": "<poprawka>"}}
  ],
  "keywords_to_add": {unused_basic + unused_extended if (unused_basic or unused_extended) else []},
  "summary": "<2-3 zdania podsumowania>"
}}

=== ARTYKUŁ ({word_count} słów) ===

{full_text}"""

        print(f"[EDITORIAL_REVIEW] ========== CALL 1: ANALYSIS ==========")
        print(f"[EDITORIAL_REVIEW] Analysis prompt length: {len(analysis_prompt)} chars")
        
        analysis = None
        corrected_article = ""
        ai_model = "unknown"
        
        if claude_client:
            # WYWOŁANIE 1: Analiza
            print(f"[EDITORIAL_REVIEW] Calling Claude for ANALYSIS...")
            response1 = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            analysis_text = response1.content[0].text.strip()
            ai_model = "claude"
            
            print(f"[EDITORIAL_REVIEW] Analysis response: {len(analysis_text)} chars")
            print(f"[EDITORIAL_REVIEW] Analysis preview: {analysis_text[:500]}")
            
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
            
            # WYWOŁANIE 2: Poprawiony tekst
            errors_list = analysis.get("errors_to_fix", []) if analysis else []
            keywords_to_add = analysis.get("keywords_to_add", []) if analysis else []
            
            correction_prompt = f"""Popraw poniższy artykuł pt. "{topic}".

=== PRIORYTETY (w tej kolejności!) ===

🔴 PRIORYTET 1: JAKOŚĆ TEKSTU
- Tautologie ("przedszkole...w przedszkolu") → zamień na synonim
- Pleonazmy i powtórzenia → usuń lub zamień
- Strona bierna → zamień na czynną gdzie możliwe
- AI patterns ("W dzisiejszych czasach") → USUŃ
- Halucynacje (wymyślone fakty) → USUŃ

🟡 PRIORYTET 2: ENCJE I N-GRAMY
- Kluczowe pojęcia zdefiniowane przy pierwszym użyciu

🟢 PRIORYTET 3: FRAZY SEO (elastycznie!)
- Wpleć naturalnie, NIE "na siłę"
- Lepiej 1× naturalnie niż 3× sztucznie

=== POPRAWKI DO WPROWADZENIA ===
{json.dumps(errors_list, ensure_ascii=False, indent=2) if errors_list else "Brak krytycznych błędów."}

=== FRAZY DO WPLECENIA (jeśli brakuje, wpleć naturalnie) ===
{', '.join(keywords_to_add) if keywords_to_add else "Wszystkie frazy są w tekście."}

=== WYMAGANIA TECHNICZNE ===
- Zachowaj strukturę HTML/Markdown (H2, H3)
- MINIMUM {word_count} słów (nie skracaj!)
- NIE dodawaj linków
- Zachowaj naturalny, płynny styl

=== ZWRÓĆ TYLKO POPRAWIONY TEKST (bez komentarzy, bez JSON) ===

=== ORYGINALNY ARTYKUŁ ===

{full_text}"""

            print(f"[EDITORIAL_REVIEW] ========== CALL 2: CORRECTION ==========")
            print(f"[EDITORIAL_REVIEW] Correction prompt length: {len(correction_prompt)} chars")
            
            response2 = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                messages=[{"role": "user", "content": correction_prompt}]
            )
            corrected_article = response2.content[0].text.strip()
            
            print(f"[EDITORIAL_REVIEW] ✅ Corrected article: {len(corrected_article)} chars, {len(corrected_article.split())} words")
            
        elif genai:
            print(f"[EDITORIAL_REVIEW] Using Gemini (fallback) for project {project_id}")
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Gemini: jeden prompt (stary sposób)
            old_prompt = f"""Jesteś REDAKTOREM. Popraw artykuł i zwróć JSON:
{{"analysis": {{"overall_score": <0-10>, "summary": "<podsumowanie>"}}, "corrected_article": "<cały poprawiony tekst>"}}

Artykuł ({word_count} słów):
{full_text}"""
            
            response = model.generate_content(old_prompt)
            review_text = response.text.strip()
            ai_model = "gemini"
            
            # Parsuj Gemini response
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
        
        # v28.1: GRAMMAR CORRECTION - popraw błędy gramatyczne + usuń banned phrases
        grammar_stats = {"fixes": 0, "removed": []}
        try:
            from grammar_checker import full_correction
            grammar_result = full_correction(corrected_article)
            corrected_article = grammar_result["corrected"]
            grammar_stats = {
                "fixes": grammar_result["grammar_fixes"],
                "removed": grammar_result["phrases_removed"],
                "backend": grammar_result["backend"]
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
                "original_word_count": word_count,
                "corrected_word_count": corrected_word_count,
                "unused_keywords": {
                    "basic": unused_basic,
                    "extended": unused_extended
                },
                "grammar_correction": grammar_stats,  # v28.1
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
            "version": "v28.1",
            "grammar_correction": grammar_stats,  # v28.1: statystyki korekty gramatycznej
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
    
    # v32.4: Priorytet źródeł treści
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
        "topic": project_data.get("topic"),
        "status": project_data.get("status", "UNKNOWN"),
        "batches_count": batches_count,
        "word_count": word_count,
        "debug": debug_info,
        "editorial_review": {
            "done": bool(editorial),
            "score": editorial.get("analysis", {}).get("overall_score"),
            "corrected_word_count": editorial.get("corrected_word_count"),
            "ai_model": editorial.get("ai_model")
        },
        "ready_for_export": word_count >= 500,
        "actions": {
            "editorial_review": f"POST /api/project/{project_id}/editorial_review",
            "export_docx": f"GET /api/project/{project_id}/export/docx",
            "export_html": f"GET /api/project/{project_id}/export/html",
            "export_txt": f"GET /api/project/{project_id}/export/txt"
        },
        "version": "v32.4"
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
    # 3. batches[] (stary sposób)
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
    
    # 🔧 FIX: Bezpieczna nazwa pliku (bez polskich znaków w HTTP header)
    # Transliteracja polskich znaków
    polish_map = str.maketrans({
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    })
    safe_topic = topic.translate(polish_map)
    filename = re.sub(r'[^\w\s-]', '', safe_topic)[:50] + ".docx"
    
    # RFC 5987: filename* dla UTF-8 w nagłówku
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
    
    # v32.4: Priorytet źródeł treści
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
    
    # 🔧 FIX: Bezpieczna nazwa pliku
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
    
    # v32.4: Priorytet źródeł treści
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
    
    # 🔧 FIX: Bezpieczna nazwa pliku
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
