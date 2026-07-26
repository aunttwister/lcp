"""Request pipeline — chat completion lifecycle.

Handles the full request flow:
  auth → strip tools → cache check → try provider chain → calculate cost → record
"""

import json
import os
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .circuit_breaker import get_circuit_breaker
from .cost_estimator import estimate_from_request
from .exceptions import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolBlockedError,
)
from .logging_config import get_logger
from .models import get_session, Request as RequestModel
from .prompt_cache import get_prompt_cache
from .token_verifier import get_token_verifier

logger = get_logger("lcp.pipeline")

# Tool names that Hermes agents can request
KNOWN_HERMES_TOOLS = {
    "read_file", "write_file", "patch", "search_files", "terminal",
    "execute_code", "memory", "session_search", "process", "delegate_task",
    "send_message", "skill_manage", "todo", "vision_analyze", "web",
    "cronjob", "text_to_speech",
}


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
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def normalize_tools_for_cache(tools: list[dict]) -> list[dict]:
    """Sort tool definitions by function name for deterministic ordering."""
    if not tools:
        return tools
    return sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))


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
            # Drop orphaned tool response
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
    """Calculate token usage and cost from request+response."""
    pricing = config.get_pricing(provider, model)

    usage = response_body.get("usage", {}) if response_body else {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_hit = read_cache_hit_tokens(provider, response_body, config)
    cache_miss = usage.get("prompt_cache_miss_tokens", prompt_tokens - cache_hit if prompt_tokens > cache_hit else 0)

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

    If body['stream'] is True, returns (raw_sse_bytes, status_code) with raw_sse_bytes being
    the full SSE response as bytes.
    Otherwise returns (response_body_dict, status_code).
    """
    api_key = os.environ.get(provider_cfg.get("api_key_env", ""))
    if not api_key:
        provider_name = provider_cfg["provider"]
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            if streaming:
                return raw, resp.status
            response_body = json.loads(raw.decode("utf-8"))
            return response_body, resp.status
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")[:500]
        status = e.code
        if status == 401 or status == 403:
            raise ProviderAuthError(f"Provider {provider_cfg['provider']} rejected auth: {status}")
        elif status == 429:
            raise ProviderRateLimitError(f"Provider {provider_cfg['provider']} rate limited")
        raise ProviderAuthError(f"Provider {provider_cfg['provider']} HTTP {status}: {error_body}")
    except urllib.error.URLError as e:
        raise ProviderTimeoutError(f"Provider {provider_cfg['provider']} unreachable: {e.reason}")


def try_chain(profile_name: str, profile_cfg: dict, body: dict, config) -> tuple[dict, int, str, str]:
    """Try each provider in the chain. Returns (response, status, provider, model)."""
    cb = get_circuit_breaker()
    errors = []
    for step in profile_cfg["chain"]:
        provider_name = step["provider"]
        base_url = step.get("base_url") or config.providers.get(provider_name, {}).get("api_base", "")
        model = step["model"]

        # Check circuit breaker
        if not cb.is_available(provider_name, base_url, profile_name):
            logger.warning(
                "provider_skipped_circuit_breaker",
                provider=provider_name,
                profile=profile_name,
            )
            errors.append(f"{provider_name}: circuit breaker open")
            continue

        # Set model in body
        body["model"] = model

        # Normalize for prefix caching — deterministic message/tool ordering
        # so repeated logical prompts hit the provider's KV-cache.
        if "messages" in body:
            body["messages"] = normalize_messages_for_cache(body["messages"])
        if "tools" in body and body["tools"]:
            body["tools"] = normalize_tools_for_cache(body["tools"])

        # Add API key env to step config
        step_with_key = {**step, "api_key_env": config.providers[provider_name]["api_key_env"]}

        try:
            resp, status = forward_request(step_with_key, body, config)
            cb.record_success(provider_name, base_url, profile_name)
            return resp, status, provider_name, model
        except (ProviderTimeoutError, ProviderAuthError, ProviderRateLimitError) as e:
            cb.record_failure(provider_name, base_url, profile_name)
            errors.append(f"{provider_name}: {e}")
            logger.error("provider_failed", provider=provider_name, error=str(e))

    raise AllProvidersFailedError(f"All providers failed for {profile_name}: {'; '.join(errors)}")


# ── Cost Recording ───────────────────────────────────────────────────────────

def record_cost(engine, profile: str, model: str, provider: str, cost_info: dict,
                success: bool, error_type: str | None, tools_blocked: list[str]) -> None:
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
            tools_blocked=",".join(tools_blocked) if tools_blocked else None,
        )
        session.add(req)
        session.commit()

    # Track spend against key (when key auth is wired in)
