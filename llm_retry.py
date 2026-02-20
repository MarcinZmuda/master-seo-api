"""
LLM Retry wrapper for master-seo-api.
Fix #23 v4.2: Wzorowany na Brajn2026 _llm_call_with_retry().

Obsluguje retryable HTTP errors: 429 (rate limit), 503 (overloaded), 529 (overloaded).
Non-retryable errors (400, 401, 403, 404) sa natychmiast propagowane.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Retryable status codes
LLM_RETRYABLE_CODES = {429, 503, 529}

# Retry config
LLM_RETRY_MAX = 3
LLM_RETRY_DELAYS = [10, 30, 60]  # seconds between retries
LLM_529_DELAYS = [5, 15]  # shorter delays for 529 (temporary overload)


def llm_call_with_retry(fn, *args, max_retries=None, initial_backoff=None, **kwargs):
    """
    Retry wrapper for Claude/Anthropic API calls.

    Args:
        fn: callable to execute
        *args, **kwargs: passed to fn
        max_retries: override default retry count (optional)
        initial_backoff: ignored (kept for API compat with claude_reviewer fallback)

    Returns:
        Result from fn(*args, **kwargs)

    Raises:
        Last exception if all retries exhausted, or non-retryable error immediately.
    """
    retries = max_retries if max_retries is not None else LLM_RETRY_MAX
    last_error = None

    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Check if it's an Anthropic API error with status code
            status_code = getattr(e, 'status_code', None)

            if status_code is not None:
                # Known HTTP error
                if status_code not in LLM_RETRYABLE_CODES:
                    # Non-retryable (400, 401, 403, 404, etc.) — raise immediately
                    logger.error(f"[LLM_RETRY] Non-retryable error {status_code}: {e}")
                    raise

                # Retryable error
                is_529 = (status_code == 529)
                max_r = 2 if is_529 else retries
                delays = LLM_529_DELAYS if is_529 else LLM_RETRY_DELAYS

                if attempt < max_r:
                    delay = delays[min(attempt, len(delays) - 1)]
                    logger.warning(
                        f"[LLM_RETRY] {status_code} attempt {attempt + 1}/{max_r + 1}, "
                        f"waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"[LLM_RETRY] {status_code} exhausted after {attempt + 1} attempts"
                    )
                    raise
            else:
                # Unknown exception type (network error, timeout, etc.)
                if attempt < retries:
                    delay = LLM_RETRY_DELAYS[min(attempt, len(LLM_RETRY_DELAYS) - 1)]
                    logger.warning(
                        f"[LLM_RETRY] Unknown error attempt {attempt + 1}/{retries + 1}: "
                        f"{type(e).__name__}: {e}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"[LLM_RETRY] Failed after {attempt + 1} attempts: {e}")
                    raise

    raise last_error
