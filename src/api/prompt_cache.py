"""Semantic prompt cache — identical prompts served from cache with zero cost.

Stores: (profile, model, tools_hash, messages_hash) → response
Cache expiry: configurable TTL
"""

import hashlib
import json
import time
from typing import Optional

from .logging_config import get_logger

logger = get_logger("lcp.cache")


class PromptCache:
    """In-memory prompt response cache with TTL."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000):
        self._cache: dict[str, tuple[float, dict]] = {}  # key → (expires_at, response)
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._hits: int = 0
        self._misses: int = 0

    def _make_key(self, profile: str, model: str, body: dict) -> str:
        """Create a cache key from request parameters."""
        messages = json.dumps(body.get("messages", []), sort_keys=True)
        tools = json.dumps(body.get("tools", []), sort_keys=True) if body.get("tools") else ""
        max_tokens = body.get("max_tokens", 0)

        raw = f"{profile}|{model}|{max_tokens}|{messages}|{tools}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, profile: str, model: str, body: dict) -> Optional[dict]:
        """Retrieve cached response. Returns None on miss."""
        key = self._make_key(profile, model, body)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            logger.debug("cache_miss", key=key[:12], entries=len(self._cache),
                         hits=self._hits, misses=self._misses)
            return None

        expires, response = entry
        if time.time() > expires:
            del self._cache[key]
            self._misses += 1
            logger.debug("cache_expired", key=key[:12], entries=len(self._cache))
            return None

        self._hits += 1
        logger.debug("cache_hit", key=key[:12], hits=self._hits, misses=self._misses)
        return response

    def set(self, profile: str, model: str, body: dict, response: dict) -> None:
        """Store a response in the cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
            logger.debug("cache_evicted", key=oldest_key[:12], entries=len(self._cache))

        key = self._make_key(profile, model, body)
        self._cache[key] = (time.time() + self._ttl, response)
        logger.debug("cache_set", key=key[:12], entries=len(self._cache),
                     max_entries=self._max_entries)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    @property
    def stats(self) -> dict:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
        }


# Global instance
_prompt_cache = PromptCache()


def get_prompt_cache() -> PromptCache:
    return _prompt_cache


# ── Component-runtime adapter (Phase C) ──────────────────────────────
# Dep-free leaf: no requires, no teardown. Registered so Phase D's uniform
# runtime boot can own it alongside the rest.
class PromptCacheComponent:
    name = "prompt_cache"
    requires = []
    provides = ["prompt_cache"]

    @property
    def service(self):
        return get_prompt_cache()

    def setup(self, rt):
        return None
