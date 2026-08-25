"""Memory harness — auto-recall relevant facts and inject them into requests.

The memory backend is a manual store/recall API. This harness makes memory
functional *inside* the request pipeline: when the memory module is active and
``plugins.memory.auto_recall`` is enabled, the latest user message is embedded,
the top-k most relevant memories for the profile are recalled, and they are
injected as a delimited context block into the outgoing chat messages.

It is intentionally a clean seam:
* No-op when memory isn't active or auto_recall is disabled (request flows
  unchanged).
* Config-driven (``top_k``, ``min_score``) so behaviour is tunable.
* Memory stays separate from routing — this only adds context for the model.
"""

from __future__ import annotations

from typing import Any, Optional

from ..logging_config import get_logger

logger = get_logger("lcp.memory.harness")


def _latest_user_text(messages: list[dict]) -> str:
    """Return the most recent non-empty user message text."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")]
            if parts:
                return " ".join(parts).strip()
    return ""


def _build_context_block(memories: list[dict]) -> str:
    """Render recalled memories into a delimited context block for the model."""
    lines = []
    for m in memories or []:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        meta = m.get("metadata") or {}
        prefix = ""
        if meta.get("host"):
            prefix = f"[{meta['host']}] "
        lines.append(f"- {prefix}{content}")
    if not lines:
        return ""
    return "The following facts were recalled from this profile's memory and may be relevant:\n" + "\n".join(lines)


def recall_for_request(
    messages: list[dict],
    profile: str = "default",
    top_k: int = 3,
    min_score: float = 0.0,
    tag_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Recall the top-k most relevant memories for the latest user message.

    Returns ``[{id, content, metadata, tags, score}]`` sorted best-first,
    filtered to ``score >= min_score``. Returns [] when memory is inactive or
    the query is empty.
    """
    try:
        from . import get_memory
        backend = get_memory()
    except Exception:  # noqa: BLE001
        return []
    if backend is None:
        return []
    query = _latest_user_text(messages)
    if not query:
        return []
    try:
        results = backend.recall(
            query, top_k=top_k,
            tag_filter=tag_filter,
            profile=profile,
        )
    except Exception as exc:  # noqa: BLE001 — never break the request
        logger.warning("memory_recall_failed", profile=profile, error=str(exc))
        return []
    if min_score > 0:
        results = [r for r in results if (r.get("score") or 0) >= min_score]
    return results


def inject_memory_context(
    messages: list[dict],
    profile: str = "default",
    enabled: bool = True,
    top_k: int = 3,
    min_score: float = 0.0,
    tag_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Return ``messages`` with recalled memory injected as a context block.

    When memory is inactive or ``enabled`` is False, returns ``messages``
    unchanged. Otherwise prepends a system message containing the recalled
    facts (inserted before the existing system prompt, or as a new leading
    system message).
    """
    if not enabled:
        return messages
    memories = recall_for_request(
        messages, profile=profile, top_k=top_k,
        min_score=min_score, tag_filter=tag_filter,
    )
    if not memories:
        return messages
    block = _build_context_block(memories)
    if not block:
        return messages

    out = list(messages or [])
    # If the first message is already a system prompt, merge context into it;
    # otherwise prepend a new system context message.
    if out and out[0].get("role") == "system":
        existing = out[0].get("content", "")
        if isinstance(existing, str):
            merged = block + "\n\n" + existing if existing else block
            out = [dict(out[0], content=merged)] + out[1:]
            return out
    return [{"role": "system", "content": block}] + out


def config_for(config) -> dict:
    """Return the auto-recall harness config from ``plugins.memory``."""
    try:
        plugins = (getattr(config, "plugins", None) or {}) if config is not None else {}
    except Exception:  # noqa: BLE001
        plugins = {}
    mem_cfg = (plugins.get("memory") or {}) if isinstance(plugins, dict) else {}
    return {
        "enabled": bool(mem_cfg.get("auto_recall", False)),
        "top_k": int(mem_cfg.get("top_k", 3) or 3),
        "min_score": float(mem_cfg.get("min_score", 0.0) or 0.0),
        "tag_filter": mem_cfg.get("tag_filter"),
    }
