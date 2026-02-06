"""
===============================================================================
📦 YMYL API CACHE v1.0
===============================================================================
Prosty file-based cache dla zewnętrznych API (SAOS, PubMed, ClinicalTrials).

Eliminuje duplikowane requesty:
- 10 artykułów o "ubezwłasnowolnienie" = 1 request do SAOS zamiast 10
- Cache TTL: 24h (konfigurowalne)
- Automatyczne czyszczenie starych wpisów

Użycie:
    from ymyl_cache import ymyl_cache

    # Sprawdź cache
    cached = ymyl_cache.get("saos", "ubezwłasnowolnienie")
    if cached:
        return cached

    # Fetch i zapisz
    result = saos_search("ubezwłasnowolnienie")
    ymyl_cache.set("saos", "ubezwłasnowolnienie", result)
===============================================================================
"""

import os
import json
import time
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path


class YMYLCache:
    """File-based cache z TTL."""

    def __init__(self, cache_dir: str = None, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir = Path(cache_dir or os.getenv(
            "YMYL_CACHE_DIR",
            os.path.join(os.path.dirname(__file__), ".ymyl_cache")
        ))
        self._ensure_dir()
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
        print(f"[YMYL_CACHE] ✅ dir={self.cache_dir}, TTL={ttl_hours}h")

    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_hash(self, namespace: str, key: str) -> str:
        raw = f"{namespace}:{key.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _filepath(self, namespace: str, key: str) -> Path:
        return self.cache_dir / f"{namespace}_{self._key_hash(namespace, key)}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Pobierz z cache. None jeśli brak lub expired."""
        fp = self._filepath(namespace, key)
        if not fp.exists():
            self._stats["misses"] += 1
            return None

        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            ts = data.get("_ts", 0)

            if time.time() - ts > self.ttl_seconds:
                fp.unlink(missing_ok=True)
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return data.get("value")

        except (json.JSONDecodeError, OSError):
            fp.unlink(missing_ok=True)
            self._stats["misses"] += 1
            return None

    def set(self, namespace: str, key: str, value: Any) -> bool:
        """Zapisz do cache."""
        fp = self._filepath(namespace, key)
        try:
            payload = {
                "_ts": time.time(),
                "_ns": namespace,
                "_key": key[:100],
                "value": value
            }
            fp.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                          encoding="utf-8")
            self._stats["sets"] += 1
            return True
        except (OSError, TypeError) as e:
            print(f"[YMYL_CACHE] ⚠️ Write error: {e}")
            return False

    def invalidate(self, namespace: str, key: str):
        """Usuń konkretny wpis."""
        self._filepath(namespace, key).unlink(missing_ok=True)

    def clear_namespace(self, namespace: str):
        """Wyczyść cały namespace."""
        for fp in self.cache_dir.glob(f"{namespace}_*.json"):
            fp.unlink(missing_ok=True)

    def clear_expired(self) -> int:
        """Usuń expired wpisy. Zwraca liczbę usuniętych."""
        removed = 0
        now = time.time()
        for fp in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if now - data.get("_ts", 0) > self.ttl_seconds:
                    fp.unlink()
                    removed += 1
            except Exception:
                fp.unlink(missing_ok=True)
                removed += 1
        return removed

    def get_stats(self) -> Dict:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "hit_rate": round(self._stats["hits"] / total, 2) if total else 0,
            "cached_files": len(list(self.cache_dir.glob("*.json"))),
        }


# Singleton
ymyl_cache = YMYLCache()

__all__ = ["ymyl_cache", "YMYLCache"]
