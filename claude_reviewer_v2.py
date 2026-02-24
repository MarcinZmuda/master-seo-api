"""
Claude Reviewer v2.1 — 4 checks zamiast 15.
Import: from claude_reviewer_v2 import build_review_prompt_v2
"""


def build_review_prompt_v2(text, ctx):
    topic = ctx.get('topic', '')
    keywords_required = ctx.get('keywords_required', [])
    missing_basic = ctx.get('missing_basic', [])
    missing_extended = ctx.get('missing_extended', [])
    is_ymyl = ctx.get('is_ymyl', False)
    batch_number = ctx.get('batch_number', 1)
    total_batches = ctx.get('total_batches', 8)
    entities_must = ctx.get('entities_must', [])
    triplets = ctx.get('triplets', [])

    kw_section = ""
    if keywords_required:
        kw_items = []
        for kw in keywords_required[:8]:
            if isinstance(kw, dict):
                kw_items.append(f'"{kw.get("keyword", "")}"')
            else:
                kw_items.append(f'"{kw}"')
        kw_section = f"\nFRAZY WYMAGANE: {', '.join(kw_items)}"

    missing_section = ""
    if missing_basic:
        missing_section += f"\n🔴 BRAKUJĄCE: {', '.join(missing_basic[:4])} — WPLEĆ NATURALNIE!"
    if missing_extended:
        missing_section += f"\n🟡 OPCJONALNE: {', '.join(missing_extended[:3])}"

    entity_section = ""
    if entities_must:
        ent_names = [e.get('entity', e) if isinstance(e, dict) else str(e) for e in entities_must[:5]]
        entity_section = f"\nENCJE: {', '.join(ent_names)}"

    triplet_section = ""
    if triplets:
        t_strs = []
        for t in triplets[:3]:
            if isinstance(t, dict):
                s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
                if s and v and o:
                    t_strs.append(f"{s} → {v} → {o}")
        if t_strs:
            triplet_section = f"\nRELACJE: {'; '.join(t_strs)}"

    ymyl_note = ""
    if is_ymyl:
        ymyl_note = "\n⚖️ YMYL: sprawdź poprawność sygnatur, artykułów ustaw, jednostek."

    return f"""Przejrzyj batch artykułu "{topic}" (batch {batch_number}/{total_batches}) i zwróć JSON.

TEKST:
{text}
{kw_section}{missing_section}{entity_section}{triplet_section}{ymyl_note}

SPRAWDŹ (TYLKO te 4 rzeczy):

1. BRAKUJĄCE FRAZY
   Frazy z listy powyżej nie występują w tekście → wpleć je NATURALNIE
   w istniejące zdania. Nie dodawaj nowych akapitów.
   Odmiana fleksyjna liczy się jako użycie ("zakazu prowadzenia" = "zakaz prowadzenia").

2. STUFFING
   Fraza powtórzona >2× w jednym akapicie → zamień jedno wystąpienie
   na synonim lub zaimek ("ta kwestia", "omawiany aspekt").

3. DŁUGIE ZDANIA
   Zdanie >35 słów → rozbij na 2 krótsze zdania.

4. HALUCYNACJA
   Zmyślona data, liczba, sygnatura wyroku lub nazwa badania
   bez pokrycia w danych kontekstowych → USUŃ to zdanie.
   Nie zastępuj ogólnikiem — po prostu usuń.

NIE POPRAWIAJ: stylu, tonu, składni, długości akapitów, doboru słów.
To robi editorial pipeline.

ODPOWIEDŹ (TYLKO JSON, bez markdown):
{{
  "status": "APPROVED|CORRECTED",
  "issues": [
    {{"type": "missing_phrase|stuffing|long_sentence|hallucination",
      "severity": "critical|warning",
      "description": "...",
      "fix_applied": true|false}}
  ],
  "corrected_text": "PEŁNY poprawiony tekst jeśli CORRECTED, pusty string jeśli APPROVED",
  "summary": "1 zdanie podsumowujące"
}}

ZASADY:
• APPROVED = tekst OK (max 1-2 drobne warnings)
• CORRECTED = naprawiłeś 1+ problemów, zwróć PEŁNY corrected_text
• Nie używaj REJECTED — zawsze napraw jeśli możesz
• Jeśli brakuje frazy — WPLEĆ ją (zmień JEDNO istniejące zdanie)"""
