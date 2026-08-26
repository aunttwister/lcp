# Harness-Agnostic System-Prompt Handling

**Created:** 2026-08-26
**Status:** implemented (structural preamble detection)
**Depends on:** `features/routing-observability.md` (rationale capture + decision log)

---

## Problem

Some coding-agent harnesses (VS Code Copilot Chat, OpenCode, ...) send their
**system prompt as a `role="user"` message** because their chat API has no
`System` role:

- VS Code's `LanguageModelChatMessageRole` is **only `User = 1` / `Assistant = 2`**
  (confirmed in `vscode.d.ts` and the VS Code API docs: *"This is either the
  user or the assistant"*). A chat model provider **cannot** receive a
  `role="system"` message from VS Code.
- GitHub issue search shows **no known/bug report** for "system prompt sent as
  user role" — it is an inherent API limitation, not a VS Code bug.
- Standard APIs DO have a distinct role: OpenAI uses `developer`/`system`,
  Anthropic uses a top-level `system` field. LCP already skips `role="system"`.

Consequence: when the system prompt is the newest "user" message, the keyword
classifier sees its incidental words (`debugging`, `tests`, ...) and **misroutes**
(e.g. decision #497 → `debugging`). String-comparison fixes are futile because
the echo is not a verbatim copy — the harness interpolates the model name
("...using coder." vs "...using GitHub Copilot.").

## Approach: STRUCTURAL, harness-agnostic detection

We do NOT hardcode any harness's prompt text. A system prompt (sent as user) is
structurally distinct from a real user message:

1. **LONG** — a multi-sentence block (`_PREAMBLE_MIN_LEN = 120` chars); a real
   first user message ("debug this", "hi", "write tests") is short.
2. **GENERIC** — no CONCRETE task instruction (only incidental single-word
   keywords, which we ignore; strong multi-word task phrases disqualify it).
3. **POSITION** — it's the first user message / appears before real content.

Behavior:
- `_extract_intent_text` walks newest-first (unchanged). When the newest
  genuine user message is preamble-like, it is flagged
  `meta["preamble"]=True` (not dropped — visible in decisions).
- If the preamble **appends a real instruction** after a blank line, the tail is
  kept as intent (don't lose the request).
- `classify_task_detail`: a preamble-flagged intent is NEVER keyword-matched —
  it routes to a **neutral default** (`agentic_multi_step`) with `path="preamble"`.
  This prevents the #497 misclassification deterministically.

## Why this is the right approach

- **No harness names** — works for VS Code, OpenCode, and any future harness
  (a new harness with a differently-worded prompt is handled automatically).
- **No phrase lists** — nothing breaks when a harness rewords its prompt.
- **Deterministic** — the same input always yields the same result.
- **Uses the existing role=system skip** for well-behaved harnesses (OpenAI/
  Anthropic-style); the structural rule only covers the VS-Code-style gap.

## Implementation (src/api/router.py)

- `_PREAMBLE_MIN_LEN = 120`
- `_preamble_head(text)` — the leading block (before the first blank line), so a
  preamble + appended instruction is still recognized on its own head.
- `_is_preamble_like(text)` — long + generic (only strong multi-word task
  phrases count as "concrete") + not a continuation.
- `_preamble_tail(text)` — the real instruction appended after a blank line.
- `_extract_intent_text`: flag preamble (`meta["preamble"]=True`,
  `meta["source"]="preamble"`), keep appended tail, remember last preamble for
  the nothing-genuine-survived fallback.
- `classify_task_detail`: preamble-flagged intent → `agentic_multi_step`,
  `path="preamble"` (before keyword matching).

## Tests (tests/test_router.py)

- `test_extract_intent_text_preamble_only_neutralizes` — long generic preamble
  as the newest msg → flagged preamble (not dropped).
- `test_classify_preamble_routes_to_agentic` — preamble-only intent →
  `agentic_multi_step` + `path="preamble"` (not misrouted to debugging).
- `test_extract_intent_text_short_system_match_not_echo` — short real message
  ("debug this") is NOT treated as a preamble.
- `test_extract_intent_text_combined_preamble_keeps_tail` — preamble + real
  instruction after a blank line keeps the tail.

## Verification

1. `pytest -q` — full suite passes.
2. Manual: paste a long generic preamble as the only user msg → routes to
   agentic (path=preamble) instead of keyword-matching it.
3. Confirm a real instruction ("debug this") still routes to debugging.
4. `judge_routing.py replay` on dev: preamble-driven misroutes drop.

## Research references

- VS Code API: `LanguageModelChatMessageRole` (User=1, Assistant=2 only).
- VS Code API docs: `LanguageModelChatMessage` — "This is either the user or
  the assistant."
- GitHub issues: no reported bug for system-prompt-as-user (design limitation).
- OpenAI: `developer`/`system` role; Anthropic: top-level `system` field.
