# Feature: DeepSeek Thinking-Mode Reasoning Recovery

**Created:** 2026-08-08
**Status:** ✅ implemented (0.5.x)

## Problem

DeepSeek's thinking mode (both `deepseek-v4-pro` and `deepseek-v4-flash`) requires
`reasoning_content` to be passed back on tool-calling assistant turns in multi-turn
conversations. Per the [official docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls):

> *"For requests carrying the `tools` parameter, the `reasoning_content` must be
> fully passed back to the API in all subsequent requests. If your code does not
> correctly pass back `reasoning_content`, the API will return a 400 error."*

Agents and clients (Hermes, VS Code Copilot, etc.) often strip `reasoning_content`
when rebuilding multi-turn conversation history before sending the next request.
The resulting HTTP 400 propagates through LCP as a 502 to the client:

```
Provider deepseek HTTP 400: {
  "error": {
    "message": "The `reasoning_content` in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "code": "invalid_request_error"
  }
}
```

## Solution: Reasoning Store

LCP captures `reasoning_content` as it flows through the gateway from the provider
and re-attaches it when the client strips it on subsequent requests.

### Capture

When the provider returns a response containing `reasoning_content` and `tool_calls`:

- **Non-streaming:** parsed from `choices[0].message.reasoning_content`
- **Streaming (SSE):** accumulated from `delta.reasoning_content` across chunks

The content is keyed by `tool_call_id` — the stable unique identifier DeepSeek emits
per tool call. Agents preserve `tool_call_id` in their conversation history (tool
responses reference them), so the match is reliable across turns.

### Re-attachment

On the next request, LCP inspects assistant messages carrying `tool_calls` but missing
`reasoning_content`. If a matching `tool_call_id` is found in the store, the **genuine**
reasoning text is injected — the provider receives the real chain-of-thought it
produced earlier.

### Fallback

If no stored content exists (first turn after restart, or the conversation started
before LCP was in the path), an empty string is injected as a presence placeholder.
DeepSeek validates **field presence** for the 400 error path, so this prevents the
error even without the real content.

### Data Flow

```
Turn 1: agent → LCP → DeepSeek
        DeepSeek returns: {"choices":[{"message":{
          "reasoning_content": "User wants a function...",
          "tool_calls": [{"id":"call_abc", ...}]
        }}]}
        LCP captures: call_abc → "User wants a function..."

Turn 2: agent → LCP (history missing reasoning_content)
        LCP rehydrates: call_abc → "User wants a function..."
        LCP forwards message with real reasoning_content → DeepSeek
        DeepSeek receives the genuine chain-of-thought — no 400
```

### Design Decisions

| Decision | Rationale |
|---|---|
| Keyed by `tool_call_id` | Only stable identifier preserved by agents across turns; tool responses validate against it |
| In-memory (not persisted) | Multi-turn reasoning windows are short; empty-string fallback covers restart gaps |
| 1-hour TTL | Most agent conversations complete within minutes; keeps memory bounded |
| 2048-entry cap | Ample for a single-user gateway with deep agent loops |
| Best-effort (never raises) | A capture/rehydration failure must never break request processing |

### Limitations

- **Restart amnesia:** the store is lost on gateway restart. Conversations spanning a
  restart revert to the empty-string fallback (still no 400, but the provider loses
  reasoning context).
- **tool_call_id rewriting:** if an agent rewrites `tool_call_id` values when
  rebuilding history, the lookup won't match. This is uncommon — tool responses
  reference tool_call_id, so rewriting breaks tool validation too.
- **Non-tool-call turns:** per the docs, assistant turns *without* tool_calls
  don't need `reasoning_content` passed back (the API ignores it). LCP only
  injects on tool-call turns.

## Configuration

Models that support thinking mode are flagged in `gateway.yaml`:

```yaml
model_limits:
  deepseek-v4-pro:
    context_window: 1000000
    max_output_tokens: 384000
    supports_vision: false
    supports_thinking: true      # enables reasoning-content injection
  deepseek-v4-flash:
    context_window: 1000000
    max_output_tokens: 384000
    supports_vision: false
    supports_thinking: true
```

The injection only activates when `supports_thinking: true` — non-thinking models
are left untouched.

## Related

- [Provider health dashboard](provider-health.md)
- Circuit breaker: `ProviderCreditsError` detection for insufficient-balance errors
  (open code workspaces returning `CreditsError`)
- [DeepSeek Thinking Mode docs](https://api-docs.deepseek.com/guides/thinking_mode)
