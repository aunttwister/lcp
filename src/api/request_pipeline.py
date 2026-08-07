"""Request pipeline — chat completion lifecycle.

Handles the full request flow:
  auth → strip tools → cache check → try provider chain → calculate cost → record
"""

import json
import os
import random
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable, TypeVar

from .circuit_breaker import get_circuit_breaker
from .cost_estimator import estimate_from_request
from .cost_plugins import get_registry, init_plugins
from .exceptions import (
    AllProvidersFailedError,
    ConfigError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderInternalError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolBlockedError,
)
from .logging_config import get_logger
from .models import get_session, Request as RequestModel
from .prompt_cache import get_prompt_cache
from .token_verifier import get_token_verifier

logger = get_logger("lcp.pipeline")

# Generic return type for call_with_retry — it returns whatever request_fn returns.
T = TypeVar("T")


# ── Tool Stripping ───────────────────────────────────────────────────────────

def strip_forbidden_tools(body: dict, forbidden: list[str] | None) -> tuple[dict, list[str]]:
    """Remove forbidden tools from the request body. Returns (modified_body, blocked_tools)."""
    if forbidden is None:
        # ALL tools forbidden — strip everything
        blocked = []
        if "tools" in body and body["tools"]:
            blocked = [t.get("function", {}).get("name", "unknown") for t in body["tools"]]
            body["tools"] = []
        return body, blocked

    if not forbidden or "tools" not in body or not body["tools"]:
        return body, []

    blocked = []
    kept = []
    for tool in body["tools"]:
        name = tool.get("function", {}).get("name", "")
        if name in forbidden:
            blocked.append(name)
        else:
            kept.append(tool)

    body["tools"] = kept
    return body, blocked


# ── Prefix Cache Normalization ───────────────────────────────────────────────

def normalize_messages_for_cache(messages: list[dict]) -> list[dict]:
    """Make the cacheable prefix deterministic across requests.

    DeepSeek/OpenAI cache from token 0. Any difference in the prefix
    (trailing whitespace, different tool ordering) breaks the cache.
    This normalizes so identical logical prompts produce identical
    token sequences — maximizing cache-hit rate.

    IMPORTANT: Preserves tool_call_id on tool messages — providers require it.
    IMPORTANT: Preserves reasoning_content on assistant messages — DeepSeek
    thinking mode requires this field to be passed back in conversation history.
    """
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system" and isinstance(content, str):
            content = content.rstrip()
        if role == "tool":
            # Preserve tool_call_id — providers require it for validation
            normalized.append({
                "role": role,
                "content": content,
                "tool_call_id": msg.get("tool_call_id", ""),
            })
        elif role == "assistant" and msg.get("tool_calls"):
            # Preserve tool_calls on assistant messages so sanitize_messages()
            # can match them to tool responses. Without this, normalize strips
            # tool_calls → sanitize orphans all tool messages → conversation
            # collapses to flat user/assistant history → provider sees no tool
            # context and stops responding mid-stream.
            entry = {
                "role": role,
                "content": content,
                "tool_calls": msg["tool_calls"],
            }
            if msg.get("reasoning_content"):
                entry["reasoning_content"] = msg["reasoning_content"]
            if msg.get("name"):
                entry["name"] = msg["name"]
            normalized.append(entry)
        else:
            entry = {"role": role, "content": content}
            # DeepSeek thinking mode requires reasoning_content to be passed
            # back in subsequent requests — omit it and get HTTP 400.
            if role == "assistant" and msg.get("reasoning_content"):
                entry["reasoning_content"] = msg["reasoning_content"]
            if msg.get("name"):
                entry["name"] = msg["name"]
            normalized.append(entry)
    return normalized


def normalize_tools_for_cache(tools: list[dict]) -> list[dict]:
    """Sort tool definitions by function name for deterministic ordering."""
    if not tools:
        return tools
    return sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))


def has_image_content(messages: list[dict]) -> bool:
    """Return True if any message contains an image_url content block."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    return True
    return False


# ── Message Sanitization ─────────────────────────────────────────────────────

def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Fix malformed conversation histories before forwarding to providers.

    Two cases handled:
      1. Dangling assistant tool_calls (no matching tool response) ->
         remove the tool_calls AND the orphaned tool responses.
      2. Orphaned tool messages (tool_call_id not declared by any assistant) ->
         remove the orphaned tool messages.

    This prevents deepseek 400 errors:
      - 'missing field tool_call_id' (tool msg without id)
      - 'tool must be a response to preceding tool_calls' (tool msg with no
        assistant that declared the call)
    """
    if not messages:
        return messages

    # Build set of all tool_call_ids declared by assistant messages
    declared_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", tc.get("tool_call_id"))
                if tc_id:
                    declared_ids.add(tc_id)

    # Build set of tool_call_ids referenced by tool messages
    referenced_ids = set()
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            referenced_ids.add(msg["tool_call_id"])

    # Dangling = declared by assistant but never answered by tool msg
    dangling_declared = declared_ids - referenced_ids
    # Orphaned = referenced by tool msg but never declared by assistant
    orphaned_referenced = referenced_ids - declared_ids

    if not dangling_declared and not orphaned_referenced:
        return messages

    ids_to_remove = dangling_declared | orphaned_referenced

    sanitized = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") in ids_to_remove:
            # Convert orphaned tool response to user message (preserve context)
            sanitized.append({"role": "user", "content": msg.get("content", "")})
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            kept_calls = [
                tc for tc in msg["tool_calls"]
                if tc.get("id", tc.get("tool_call_id")) not in ids_to_remove
            ]
            if kept_calls:
                sanitized.append({**msg, "tool_calls": kept_calls})
            elif msg.get("content"):
                sanitized.append({**msg, "tool_calls": None})
            # else: drop assistant msg that only had dangling tool_calls
        else:
            sanitized.append(msg)

    logger.warning(
        "messages_sanitized",
        dangling_declared=len(dangling_declared),
        orphaned_referenced=len(orphaned_referenced),
        converted_to_user=len(orphaned_referenced),
        removed_ids=len(ids_to_remove),
        original_len=len(messages),
        sanitized_len=len(sanitized),
    )
    return sanitized


# ── Cost Calculation ─────────────────────────────────────────────────────────

def compute_cache_savings(provider_name: str, model: str, cache_hit_tokens: int,
                          config) -> float:
    """Estimate dollars saved via provider prefix caching.

    For 'cost' savings type: cache_hit_tokens × (miss_price − hit_price).
    For other types (latency/none): returns 0.0.
    """
    if cache_hit_tokens <= 0:
        return 0.0
    cc = config.get_provider_cache_config(provider_name)
    if cc.get("savings") != "cost":
        return 0.0
    try:
        pricing = config.get_pricing(provider_name, model)
        return (cache_hit_tokens / 1_000_000) * (
            pricing["cache_miss"] - pricing["cache_hit"]
        )
    except Exception:
        return 0.0


def read_cache_hit_tokens(provider_name: str, response_body: dict | None,
                          config) -> int:
    """Read cache-hit tokens from provider response, using configured field name.

    Falls back to the standard 'prompt_cache_hit_tokens' field when the config
    doesn't declare a provider-specific field (or when called from tests with
    a minimal mock config).
    """
    if response_body is None:
        return 0
    usage = response_body.get("usage", {})
    field = "prompt_cache_hit_tokens"  # default for DeepSeek/OpenAI
    if hasattr(config, "get_provider_cache_config"):
        cc = config.get_provider_cache_config(provider_name)
        if isinstance(cc, dict) and cc.get("hit_field"):
            field = cc["hit_field"]
    return usage.get(field, 0)


def calculate_cost(provider: str, model: str, body: dict, response_body: dict | None,
                   config) -> dict:
    """Calculate token usage and cost from request+response.

    Tries the plugin registry first (for plugin-provided cost tracking),
    then falls back to the generic config-based pricing.
    """
    usage = response_body.get("usage", {}) if response_body else {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_hit = read_cache_hit_tokens(provider, response_body, config)
    cache_miss = usage.get("prompt_cache_miss_tokens",
                           prompt_tokens - cache_hit if prompt_tokens > cache_hit else 0)

    usage_for_plugin = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
    }

    # Try plugin registry first
    plugin_cost = get_registry().calculate_cost(provider, model, usage_for_plugin)
    if plugin_cost is not None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss,
            "cost": round(plugin_cost, 8),
        }

    # Fall back to config-based pricing
    pricing = config.get_pricing(provider, model)

    cache_hit_cost = (cache_hit / 1_000_000) * pricing["cache_hit"]
    cache_miss_cost = (cache_miss / 1_000_000) * pricing["cache_miss"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cost": round(cache_hit_cost + cache_miss_cost + output_cost, 8),
    }


# ── Request Forwarding ───────────────────────────────────────────────────────

def forward_request(provider_cfg: dict, body: dict, config):
    """Forward a request to a provider.

    If body['stream'] is True, returns (chunk_iterable, status_code) where
    chunk_iterable yields raw bytes as they arrive from the upstream.
    Otherwise returns (response_body_dict, status_code).
    """
    api_key = os.environ.get(provider_cfg.get("api_key_env", ""))
    if not api_key:
        provider_name = provider_cfg["provider"]
        # Check for inline api_key in provider config first
        provider_data = config.providers.get(provider_name, {})
        if isinstance(provider_data, dict) and provider_data.get("api_key"):
            api_key = provider_data["api_key"]
        if not api_key:
            api_key = config.get_provider_key(provider_name)

    streaming = body.get("stream", False)

    url = f"{provider_cfg['base_url']}/chat/completions"
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LLMControlPlane/1.0",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        if streaming:
            # Return a closure that yields chunks and closes the response when done.
            # This avoids buffering the entire SSE stream in memory.
            def chunk_reader():
                try:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    resp.close()
            return chunk_reader(), resp.status
        else:
            raw = resp.read()
            resp.close()
            response_body = json.loads(raw.decode("utf-8"))
            return response_body, resp.status
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")[:500]
        status = e.code
        if status == 401 or status == 403:
            raise ProviderAuthError(f"Provider {provider_cfg['provider']} rejected auth: {status}")
        elif status == 429:
            raise ProviderRateLimitError(f"Provider {provider_cfg['provider']} rate limited")
        elif 400 <= status < 500:
            # 4xx (non-auth, non-rate-limit) — bad request, the body is the problem.
            # Do NOT fall back to another provider; the same body will fail again.
            raise ProviderBadRequestError(
                f"Provider {provider_cfg['provider']} HTTP {status}: {error_body}"
            )
        elif status >= 500:
            # 5xx — provider-side transient failure. Distinct from auth so the
            # circuit breaker and retry logic can treat it appropriately.
            raise ProviderInternalError(
                f"Provider {provider_cfg['provider']} HTTP {status}: {error_body}"
            )
        raise ProviderAuthError(f"Provider {provider_cfg['provider']} HTTP {status}: {error_body}")
    except urllib.error.URLError as e:
        raise ProviderTimeoutError(f"Provider {provider_cfg['provider']} unreachable: {e.reason}")


def call_with_retry(request_fn: Callable[[], T], retry_cfg=None) -> T:
    """Call request_fn, retrying transient provider errors (5xx / 429) with backoff.

    ``retry_cfg`` is a dict with keys: max_attempts, backoff_base,
    backoff_multiplier, max_backoff, jitter. When missing or empty, a single
    attempt is made (no retry) — preserving prior behavior for callers that
    don't opt in.

    Retryable errors: ``ProviderInternalError`` (5xx), ``ProviderRateLimitError``
    (429) — transient, worth re-attempting.

    Non-retryable errors propagate immediately: ``ProviderBadRequestError``
    (the body is wrong — retrying can't help), ``ProviderAuthError`` (the key is
    rejected — retrying won't fix it), ``ProviderTimeoutError`` (provider may be
    down — the chain fallback handles it instead).

    Returns whatever ``request_fn`` returns on success; re-raises the last
    retryable error once attempts are exhausted.
    """
    retry_cfg = retry_cfg or {}
    max_attempts = max(1, int(retry_cfg.get("max_attempts", 1)))
    backoff_base = float(retry_cfg.get("backoff_base", 0.5))
    backoff_multiplier = float(retry_cfg.get("backoff_multiplier", 2))
    max_backoff = float(retry_cfg.get("max_backoff", 10))
    jitter = bool(retry_cfg.get("jitter", True))

    for attempt in range(1, max_attempts + 1):
        try:
            return request_fn()
        except (ProviderInternalError, ProviderRateLimitError) as e:
            if attempt >= max_attempts:
                # All retryable attempts exhausted — surface the last error so
                # the chain fallback (or caller) can decide what to do.
                raise
            delay = min(backoff_base * (backoff_multiplier ** (attempt - 1)), max_backoff)
            if jitter:
                delay *= random.uniform(0.5, 1.5)
            logger.info(
                "provider_retry",
                attempt=attempt,
                max_attempts=max_attempts,
                delay_ms=int(delay * 1000),
                error=str(e),
            )
            time.sleep(delay)
    # Unreachable — the loop body always returns (success) or re-raises, because
    # max_attempts is clamped to >= 1. Kept for the type checker's benefit.
    raise RuntimeError("retry loop exited unexpectedly")  # pragma: no cover


def _has_healthy_alternative(cb, chain: list[dict], profile_name: str,
                             start_idx: int, config) -> bool:
    """Return True if any chain step after start_idx has a healthy provider.

    Used for degraded-provider gating: if a degraded provider is next in line
    but a healthy provider is coming up, skip the degraded one.
    """
    for step in chain[start_idx + 1:]:
        p = step["provider"]
        url = step.get("base_url") or config.providers.get(p, {}).get("api_base", "")
        if cb.status_of(p, url, profile_name) == "healthy":
            return True
    return False


def try_chain(profile_name: str, profile_cfg: dict, body: dict, config) -> tuple[dict, int, str, str]:
    """Try each provider in the chain. Returns (response, status, provider, model)."""
    cb = get_circuit_breaker()
    errors = []
    chain_len = len(profile_cfg["chain"])
    for i, step in enumerate(profile_cfg["chain"]):
        provider_name = step["provider"]
        base_url = step.get("base_url") or config.providers.get(provider_name, {}).get("api_base", "")
        model = step["model"]

        logger.info(
            "chain_attempt",
            profile=profile_name,
            provider=provider_name,
            model=model,
            attempt=i + 1,
            chain_len=chain_len,
        )

        # Check vision support — fail gracefully so user knows why
        if has_image_content(body.get("messages", [])):
            model_limits = config.model_limits.get(model, {})
            if not model_limits.get("supports_vision", False):
                logger.warning(
                    "vision_not_supported",
                    profile=profile_name,
                    provider=provider_name,
                    model=model,
                )
                errors.append(f"{provider_name}/{model}: model does not support vision/image input")
                continue

        # Check circuit breaker
        if not cb.is_available(provider_name, base_url, profile_name):
            logger.warning(
                "provider_skipped_circuit_breaker",
                provider=provider_name,
                profile=profile_name,
            )
            errors.append(f"{provider_name}: circuit breaker open")
            continue

        # Degraded gating — prefer healthy alternatives over a degraded provider.
        # Only route to degraded when no healthy option remains in the chain.
        if cb.status_of(provider_name, base_url, profile_name) == "degraded" \
                and _has_healthy_alternative(cb, profile_cfg["chain"], profile_name, i, config):
            logger.warning(
                "provider_degraded_skipped",
                provider=provider_name,
                profile=profile_name,
            )
            errors.append(f"{provider_name}: degraded (healthy alternative available)")
            continue

        # Set model in body
        body["model"] = model

        # Normalize for prefix caching — deterministic message/tool ordering
        # so repeated logical prompts hit the provider's KV-cache.
        if "messages" in body:
            body["messages"] = normalize_messages_for_cache(body["messages"])
        if "tools" in body and body["tools"]:
            body["tools"] = normalize_tools_for_cache(body["tools"])

        # Sanitize AFTER normalization — normalization strips tool_calls from
        # assistant messages, so any tool messages left behind become orphaned.
        # Running here ensures we see the final message shape before forwarding.
        if "messages" in body:
            body["messages"] = sanitize_messages(body["messages"])

        # Add API key env and base_url to step config
        step_with_key = {
            **step,
            "api_key_env": config.providers[provider_name]["api_key_env"],
            "base_url": base_url,
        }

        # Per-provider retry config (5xx/429 retried with backoff). Real Config
        # objects expose .retry; mocks without it fall back to a single attempt.
        retry_cfg = getattr(config, "retry", None)
        if not isinstance(retry_cfg, dict):
            retry_cfg = {}

        try:
            t0 = time.time()
            resp, status = call_with_retry(
                lambda: forward_request(step_with_key, body, config),
                retry_cfg,
            )
            hop_ms = int((time.time() - t0) * 1000)
            cb.record_success(provider_name, base_url, profile_name)
            logger.info(
                "chain_success",
                profile=profile_name,
                provider=provider_name,
                model=model,
                attempt=i + 1,
                chain_len=chain_len,
                latency_ms=hop_ms,
            )
            return resp, status, provider_name, model
        except ProviderBadRequestError as e:
            # Bad request — the problem is in the body, not the provider.
            # Falling back to another provider just wastes attempts.
            # Re-raise immediately so the client sees the real error.
            cb.record_failure(provider_name, base_url, profile_name,
                              error_type=type(e).__name__)
            logger.error(
                "chain_bad_request",
                profile=profile_name,
                provider=provider_name,
                model=model,
                attempt=i + 1,
                chain_len=chain_len,
                error=str(e),
            )
            raise AllProvidersFailedError(
                f"Provider {provider_name} rejected the request as invalid: {e}"
            ) from e
        except (ProviderTimeoutError, ProviderAuthError, ProviderRateLimitError,
                ProviderInternalError) as e:
            cb.record_failure(provider_name, base_url, profile_name,
                              error_type=type(e).__name__,
                              error_reason=str(e))
            errors.append(f"{provider_name}: {e}")
            logger.warning(
                "chain_fallback",
                profile=profile_name,
                provider=provider_name,
                model=model,
                attempt=i + 1,
                chain_len=chain_len,
                error=str(e),
                next=profile_cfg["chain"][i + 1]["provider"] if i + 1 < chain_len else "none",
            )
        except ConfigError as e:
            # Provider config problem (e.g. missing API key env var) — this
            # provider can't work, but the next one might. Fall back.
            cb.record_failure(provider_name, base_url, profile_name,
                              error_type="ConfigError",
                              error_reason=str(e))
            errors.append(f"{provider_name}: {e}")
            logger.error(
                "chain_config_error",
                profile=profile_name,
                provider=provider_name,
                model=model,
                attempt=i + 1,
                chain_len=chain_len,
                error=str(e),
            )

    raise AllProvidersFailedError(f"All providers failed for {profile_name}: {'; '.join(errors)}")


# ── Cost Recording ───────────────────────────────────────────────────────────

def record_cost(engine, profile: str, model: str, provider: str, cost_info: dict,
                success: bool, error_type: str | None, tools_blocked: list[str],
                error_detail: str | None = None) -> None:
    """Record cost data to SQLite and track against budgets."""
    cost = cost_info.get("cost", 0)

    with get_session(engine) as session:
        req = RequestModel(
            timestamp=datetime.now(timezone.utc).isoformat(),
            profile=profile,
            model=model,
            provider=provider,
            prompt_tokens=cost_info.get("prompt_tokens", 0),
            completion_tokens=cost_info.get("completion_tokens", 0),
            cache_hit_tokens=cost_info.get("cache_hit_tokens", 0),
            cache_miss_tokens=cost_info.get("cache_miss_tokens", 0),
            cost=cost,
            latency_ms=cost_info.get("latency_ms", 0),
            success=1 if success else 0,
            error_type=error_type,
            error_detail=error_detail,
            tools_blocked=",".join(tools_blocked) if tools_blocked else None,
        )
        session.add(req)
        session.commit()

    # Track spend against key (when key auth is wired in)

    # ── Plugin hooks ──────────────────────────────────────────────────────
    # If the provider has a plugin, let it record the tokens (e.g. llama.cpp
    # local tracking, or future plugins that need request-level callbacks).
    if success:
        plugin = get_registry().for_provider(provider)
        if plugin is not None and hasattr(plugin, "record_tokens"):
            try:
                plugin.record_tokens(
                    model,
                    prompt_tokens=cost_info.get("prompt_tokens", 0)
                                 + cost_info.get("cache_miss_tokens", 0),
                    completion_tokens=cost_info.get("completion_tokens", 0),
                    cache_hit_tokens=cost_info.get("cache_hit_tokens", 0),
                )
            except Exception as exc:
                logger.warning("plugin_record_tokens_failed",
                               provider=provider, error=str(exc))
