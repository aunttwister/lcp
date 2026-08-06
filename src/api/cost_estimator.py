"""Pre-request cost estimation using tiktoken."""

from typing import Optional

import tiktoken

from .logging_config import get_logger

logger = get_logger("lcp.cost_estimator")


# Approximate token pricing per 1M tokens (fallback if config unavailable)
# Keys match the gateway.yaml pricing convention: cache_miss = input, output = output
_DEFAULT_PRICING = {
    "deepseek-v4-pro": {"cache_miss": 0.435, "output": 0.87},
    "deepseek-v4-flash": {"cache_miss": 0.14, "output": 0.28},
}

# Encodings — deepseek models use cl100k_base (same as GPT-4).
# Lazily loaded on first use: tiktoken downloads the tokenizer data from
# OpenAI's CDN, which fails in containers without internet access.
_ENCODING: Optional[object] = None
_ENCODING_FAILED: bool = False


def _get_encoding():
    """Load tiktoken encoding, caching the result. Returns None on failure."""
    global _ENCODING, _ENCODING_FAILED
    if _ENCODING is not None:
        return _ENCODING
    if _ENCODING_FAILED:
        return None
    try:
        _t0 = __import__("time").monotonic()
        _ENCODING = tiktoken.get_encoding("cl100k_base")
        logger.info("tiktoken_encoding_loaded",
                    encoding="cl100k_base",
                    load_ms=round((__import__("time").monotonic() - _t0) * 1000, 1))
        return _ENCODING
    except Exception as e:
        _ENCODING_FAILED = True
        logger.warning("tiktoken_encoding_failed", error=str(e))
        return None


def count_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> int:
    """Estimate token count for a chat completion request.

    Uses tiktoken with cl100k_base encoding. This is an approximation —
    exact counts require provider-specific tokenizers.

    Falls back to a simple character-based heuristic when tiktoken is
    unavailable (e.g. container without internet during first startup).
    """
    enc = _get_encoding()

    def _tokenize(text: str) -> int:
        """Encode text using tiktoken, or fall back to ~0.25 Tok/char."""
        if enc is not None:
            return len(enc.encode(text))
        return max(1, len(text) // 4)

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
