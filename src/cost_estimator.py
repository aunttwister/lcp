"""Pre-request cost estimation using tiktoken."""

from typing import Optional

import tiktoken


# Approximate token pricing per 1M tokens (fallback if config unavailable)
# Keys match the gateway.yaml pricing convention: cache_miss = input, output = output
_DEFAULT_PRICING = {
    "deepseek-v4-pro": {"cache_miss": 0.435, "output": 0.87},
    "deepseek-v4-flash": {"cache_miss": 0.14, "output": 0.28},
}

# Encodings — deepseek models use cl100k_base (same as GPT-4)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> int:
    """Estimate token count for a chat completion request.

    Uses tiktoken with cl100k_base encoding. This is an approximation —
    exact counts require provider-specific tokenizers.
    """
    token_count = 0
    for msg in messages:
        # Role
        token_count += 4  # approximate per-message overhead
        # Content
        content = msg.get("content", "")
        if isinstance(content, str):
            token_count += len(_ENCODING.encode(content))
        elif isinstance(content, list):
            # Vision content blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    token_count += len(_ENCODING.encode(block.get("text", "")))

    if tools:
        # Rough estimate: 50 tokens per tool definition
        for tool in tools:
            tool_str = str(tool)
            token_count += len(_ENCODING.encode(tool_str))

    return token_count


def estimate_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
    """Estimate token counts for messages and tools."""
    msg_tokens = count_tokens(messages, tools)
    tools_tokens = count_tokens(tools) if tools else 0
    return {
        "messages": msg_tokens,
        "tools": tools_tokens,
        "total": msg_tokens + tools_tokens,
    }


def estimate_cost(
    model: str,
    input_tokens: int,
    max_tokens: int = 1024,
    pricing: Optional[dict] = None,
) -> dict:
    """Estimate cost for a request.

    Returns:
        {"input_tokens": int, "estimated_output_tokens": int,
         "estimated_input_cost": float, "estimated_output_cost": float,
         "estimated_total_cost": float, "currency": "USD"}
    """
    if pricing is None:
        pricing = _DEFAULT_PRICING.get(model, {"cache_miss": 0.435, "output": 0.87})

    input_cost = (input_tokens / 1_000_000) * pricing.get("cache_miss", pricing.get("input", 0.435))
    output_cost = (max_tokens / 1_000_000) * pricing.get("output", 0.87)

    return {
        "input_tokens": input_tokens,
        "estimated_output_tokens": max_tokens,
        "estimated_input_cost": round(input_cost, 8),
        "estimated_output_cost": round(output_cost, 8),
        "estimated_total_cost": round(input_cost + output_cost, 8),
        "currency": "USD",
    }


def estimate_from_request(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 1024,
    pricing: Optional[dict] = None,
) -> dict:
    """Full cost estimation from request parameters."""
    input_tokens = count_tokens(messages, tools)
    return estimate_cost(model, input_tokens, max_tokens, pricing)
