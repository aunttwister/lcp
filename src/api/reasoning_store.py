"""Reasoning-content store — remembers DeepSeek thinking-mode reasoning.

DeepSeek requires ``reasoning_content`` to be passed back on tool-calling
assistant turns (HTTP 400 otherwise). Agents / Copilot often strip this field
when rebuilding multi-turn history, so LCP captures the real reasoning content
as it flows through the gateway and re-attaches it to later requests.

Storage is keyed by ``tool_call_id`` (a stable unique identifier per assistant
tool call). When a subsequent request contains an assistant message with a
``tool_calls`` entry whose id we've seen but no ``reasoning_content``, we
inject the previously-captured content — so the provider receives the genuine
chain-of-thought rather than an empty placeholder.

In-memory with TTL + bounded size (mirrors the prompt-cache pattern). State is
lost on restart, which is acceptable: multi-turn reasoning windows are short.
"""

import time
from typing import Optional

from .logging_config import get_logger

logger = get_logger("lcp.reasoning_store")

_DEFAULT_TTL_SECONDS = 3600        # 1 hour
_DEFAULT_MAX_ENTRIES = 2048        # ~2048 tool-call turns


class ReasoningStore:
    """Stores reasoning_content keyed by tool_call_id."""

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS,
                 max_entries: int = _DEFAULT_MAX_ENTRIES):
        self._by_tool_call_id: dict[str, str] = {}
        self._stored_at: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    def capture(self, tool_call_ids, reasoning_content: str) -> None:
        """Record reasoning_content for the given tool_call_id(s).

        ``tool_call_ids`` may be a single id string or an iterable of ids.
        Best-effort: no-op on empty ids; never raises.
        """
        if not tool_call_ids:
            return
        if isinstance(tool_call_ids, str):
            tool_call_ids = [tool_call_ids]
        rc = reasoning_content or ""
        now = time.time()
        for tc_id in tool_call_ids:
            if not tc_id:
                continue
            self._by_tool_call_id[tc_id] = rc
            self._stored_at[tc_id] = now
        self._prune()
        logger.debug("reasoning_captured", tool_call_ids=list(tool_call_ids),
                     chars=len(rc))

    def get_for_tool_call_id(self, tool_call_id: str) -> Optional[str]:
        """Return stored reasoning_content for a tool_call_id, or None.

        Expired entries are treated as missing (and lazily dropped).
        """
        if not tool_call_id:
            return None
        stored_at = self._stored_at.get(tool_call_id)
        if stored_at is None:
            return None
        if time.time() - stored_at > self._ttl:
            self._forget(tool_call_id)
            return None
        return self._by_tool_call_id.get(tool_call_id)

    def rehydrate(self, messages: list[dict]) -> list[dict]:
        """Attach stored reasoning_content to assistant messages that carry a
        known tool_call_id but no reasoning_content.

        Mutates and returns ``messages``. Missing lookups are left untouched
        (the caller's empty-injection fallback still applies as a last resort).
        """
        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            if "reasoning_content" in msg:
                continue
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", tc.get("tool_call_id"))
                if not tc_id:
                    continue
                rc = self.get_for_tool_call_id(tc_id)
                if rc:
                    msg["reasoning_content"] = rc
                    logger.debug(
                        "reasoning_rehydrated",
                        tool_call_id=tc_id,
                        chars=len(rc),
                    )
                    break
        return messages

    def _forget(self, tool_call_id: str) -> None:
        self._by_tool_call_id.pop(tool_call_id, None)
        self._stored_at.pop(tool_call_id, None)

    def _prune(self) -> None:
        """Drop expired entries, then oldest beyond max_entries."""
        now = time.time()
        expired = [
            k for k, t in self._stored_at.items()
            if now - t > self._ttl
        ]
        for k in expired:
            self._forget(k)
        overflow = len(self._by_tool_call_id) - self._max_entries
        if overflow > 0:
            # Drop oldest by stored_at
            oldest = sorted(self._stored_at, key=self._stored_at.get)[:overflow]
            for k in oldest:
                self._forget(k)

    def clear(self) -> None:
        self._by_tool_call_id.clear()
        self._stored_at.clear()

    def __len__(self) -> int:
        return len(self._by_tool_call_id)


# ── Module-level singleton ────────────────────────────────────────────────

_reasoning_store: ReasoningStore | None = None


def get_reasoning_store() -> ReasoningStore:
    """Get or create the reasoning-content store singleton."""
    global _reasoning_store
    if _reasoning_store is None:
        _reasoning_store = ReasoningStore()
    return _reasoning_store


# ── Component-runtime adapter (Phase C) ──────────────────────────────
# Dep-free leaf: no requires, no teardown.
class ReasoningStoreComponent:
    name = "reasoning_store"
    requires = []
    provides = ["reasoning_store"]

    @property
    def service(self):
        return get_reasoning_store()

    def setup(self, rt):
        return None
