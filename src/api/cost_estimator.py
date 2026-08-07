"""Pre-request cost estimation using a character-based heuristic.

Token count is approximate (~0.25 tokens per character for English text).
Real, billed costs come from the provider's ``usage`` block in the response,
not from this estimator. This module exists only for the pre-request
X-Estimated-Cost header and the dynamic flash/pro router.
"""

from typing import Optional

from .logging_config import get_logger

logger = get_logger("lcp.cost_estimator")


# Approximate token pricing per 1M tokens (fallback if config unavailable)
# Keys match the gateway.yaml pricing convention: cache_miss = input, output = output
_DEFAULT_PRICING = {
    "deepseek-v4-pro": {"cache_miss": 0.435, "output": 0.87},
    "deepseek-v4-flash": {"cache_miss": 0.14, "output": 0.28},
}

# Approximate ratio for English text: ~4 characters per token for BPE tokenizers
# (cl100k_base, the encoding used by DeepSeek/OpenAI models).
_CHARS_PER_TOKEN = 4


def count_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> int:
    """Estimate token count for a chat completion request.

    Uses ~4 chars/token heuristic. Exact counts require provider-specific
    tokenizers — but the real billed cost comes from the provider's response
    ``usage`` block, not from this estimate.
    """
    def _tokenize(text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN)

    token_count = 0
    for msg in messages:
        token_count += 4  # approximate per-message overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            token_count += _tokenize(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    token_count += _tokenize(block.get("text", ""))

    if tools:
        for tool in tools:
            token_count += _tokenize(str(tool))

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
    result = estimate_cost(model, input_tokens, max_tokens, pricing)
    logger.debug(
        "cost_estimated",
        model=model,
        input_tokens=input_tokens,
        estimated_cost=result["estimated_total_cost"],
    )
    return result
