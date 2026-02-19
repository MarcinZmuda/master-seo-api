"""
prompt_logger.py — Centralny logger promptów dla master-seo-api.
Zapisuje każdy prompt LLM do /tmp/master_prompts_log.txt z kontekstem:
  - etap workflow (batch_review, editorial, final_review, h2_plan, ymyl, itp.)
  - numer batcha, keyword, engine
  - pełny system + user prompt

Endpoint podglądu: GET /api/debug/master_prompts (tylko zalogowani)
Endpoint czyszczenia: GET /api/debug/master_prompts?clear=1
"""

import os
import datetime
import logging

logger = logging.getLogger(__name__)

LOG_PATH = "/tmp/master_prompts_log.txt"
MAX_SIZE = 3 * 1024 * 1024   # 3 MB
TAIL_ON_TRIM = 150 * 1024    # zostaw ostatnie 150 KB po przycięciu


def log_prompt(
    stage: str,
    system_prompt: str = None,
    user_prompt: str = None,
    *,
    keyword: str = "",
    batch: int = None,
    engine: str = "",
    extra: dict = None,
):
    """
    Zapisuje prompt do logu.

    Parametry:
        stage       — nazwa etapu, np. "batch_review", "editorial", "final_review",
                      "h2_plan", "ymyl_classify", "faq", "s1_cleanup", "smart_retry"
        system_prompt — treść system promptu (opcjonalnie)
        user_prompt   — treść user promptu
        keyword       — główne słowo kluczowe
        batch         — numer batcha (None jeśli nie dotyczy)
        engine        — "claude" / "openai" / ""
        extra         — dodatkowe dane do logowania (dict)
    """
    try:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        sep = "=" * 70
        batch_str = f" | batch={batch}" if batch is not None else ""
        engine_str = f" | engine={engine}" if engine else ""
        extra_str = ""
        if extra:
            extra_str = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())

        lines = [
            f"\n{sep}",
            f"[{ts}] STAGE={stage}{batch_str}{engine_str} | keyword={keyword}{extra_str}",
            sep,
        ]

        if system_prompt:
            lines.append("--- SYSTEM PROMPT ---")
            lines.append(system_prompt.strip())
            lines.append("")

        if user_prompt:
            lines.append("--- USER PROMPT ---")
            lines.append(user_prompt.strip())

        lines.append("")
        entry = "\n".join(lines)

        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)

        # Przytnij jeśli za duży
        size = os.path.getsize(LOG_PATH)
        if size > MAX_SIZE:
            with open(LOG_PATH, "rb") as fh:
                fh.seek(-TAIL_ON_TRIM, 2)
                tail = fh.read()
            with open(LOG_PATH, "wb") as fh:
                fh.write(b"[...log przycieto...]\n" + tail)

    except Exception as e:
        logger.debug(f"[PROMPT_LOGGER] Błąd zapisu: {e}")


def get_log_html(tail_bytes: int = 30000) -> str:
    """Zwraca log jako HTML do wyświetlenia w przeglądarce."""
    if not os.path.exists(LOG_PATH):
        content = "(brak logów — uruchom workflow)"
        size = 0
    else:
        size = os.path.getsize(LOG_PATH)
        with open(LOG_PATH, "rb") as fh:
            if size > tail_bytes:
                fh.seek(-tail_bytes, 2)
                content = "[...] " + fh.read().decode("utf-8", errors="replace")
            else:
                content = fh.read().decode("utf-8", errors="replace")

    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#0f0f0f;color:#e0e0e0;font:13px monospace;padding:20px;margin:0}"
        "pre{white-space:pre-wrap;word-break:break-all;line-height:1.5}"
        ".sep{color:#444;} .stage{color:#fbbf24;font-weight:bold}"
        ".sys{color:#86efac} .usr{color:#93c5fd} .meta{color:#a78bfa}"
        "a{color:#fb923c}"
        "</style>"
        f"<title>Master Prompts Log</title></head><body>"
        f"<p style='color:#888'>Rozmiar: {size:,} B | "
        f"<a href='?tail=100000'>Więcej (100KB)</a> | "
        f"<a href='?clear=1'>Wyczyść</a></p>"
        f"<pre>{escaped}</pre>"
        "</body></html>"
    )


def clear_log():
    """Czyści plik logu."""
    try:
        open(LOG_PATH, "w").close()
        return True
    except Exception:
        return False
