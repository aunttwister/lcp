"""Client-wrapper / message-intent normalisation.

Pure text helpers that pull the *current user intent* out of the raw message
list a client sends. Clients (VS Code Copilot Chat, OpenCode, and others) wrap
their request in injected context -- attachment / context / env wrappers,
reply-quotes, mention tokens, assistant-generated system-prompt preambles and
role="user" tool echoes -- and the router must classify the real instruction,
not the wrapper.

HARNESS-AGNOSTIC BY DESIGN: no client tag names or prompt texts are enumerated.
Wrappers are detected STRUCTURALLY (a wrapper is DELIMITED -- it begins with an
angle bracket or a square bracket -- and a harness preamble is LONG + GENERIC),
so a client that invents a brand-new wrapper or rewords its preamble tomorrow is
still handled.

This module has no dependency on the router; router.py imports these names back
so that ``src.api.router.<name>`` keeps resolving (tests import and monkeypatch
helpers through that namespace).

Extracted verbatim from ``router.py``; ``TASK_SIGNALS`` moved with the block
because ``_is_preamble_like`` / ``_is_continuation`` consume it (router re-imports
it for ``classify_task_detail``).
"""


from __future__ import annotations

import re
from typing import Optional


# Task-classification signal table. Moved with the intent heuristics that
# consume it (`_is_preamble_like`, `_is_continuation`); router.py re-imports it.
TASK_SIGNALS: dict[str, list[str]] = {
    "agentic_multi_step": [
        "you are an ai agent", "you are a coding agent",
        "autonomous", "multi-step", "multi step",
        "tools:", "function call", "tool call",
    ],
    "unit_tests": [
        "unit test", "unit tests", "write tests", "write a test",
        "test case", "test cases", "test suite", "add tests",
        "create tests", "pytest", "unittest", "test coverage",
        "mocking", "mock object", "mock the",
    ],
    "code_generation": [
        "write a function", "implement", "create a script",
        "write code", "def ", "class ", "import ",
        "write a program", "in python", "in javascript",
        "in rust", "in go", "html", "css", "react",
    ],
    "debugging": [
        "debug", "error", "exception", "traceback",
        "stack trace", "why does this fail", "not working",
        "bug", "fix this", "what's wrong",
    ],
    "research_deep": [
        "explain", "analyze", "compare and contrast",
        "research", "literature review", "in detail",
        "comprehensive", "thorough",
    ],
    "reasoning_chain": [
        "solve", "proof", "prove", "calculate",
        "logic puzzle", "step by step", "mathematical",
        "equation", "theorem",
    ],
    "planning": [
        "design", "architecture", "architect", "system design",
        "how should i structure", "roadmap", "data model", "schema",
        "tech stack", "plan the", "make a plan", "create a plan",
    ],
}


# Task types that carry concrete user intent (vs. ``agentic_multi_step``, which
# is the generic "I am an agent" preamble most agents send). Classification
# checks these against the system prompt / user messages FIRST so the agentic
# catch-all can't mask e.g. a planning request.
_SPECIFIC_TASKS = (
    "unit_tests", "debugging", "code_generation",
    "planning", "reasoning_chain", "research_deep",
)


# ── Intent extraction — newest genuine user instruction ──────────────────────
#
# The gateway classifies the CURRENT intent of each incoming request. The
# client sends the whole conversation on every request, so "current intent" is
# the newest user message that actually carries intent. Three things must be
# filtered out before we pick it:
#
#   1. Assistant/system/tool-role messages (never user intent).
#   2. Tool results that some clients send with role="user" — a violation of
#      the OpenAI role="tool" / Anthropic tool_result schemas. Their
#      "test"/"error" echoes would hijack classification.
#   3. Continuation acknowledgements ("continue", "yes", "ok", …) — they
#      carry no new intent, so we keep walking back to the last real
#      instruction.

# Tool-result prefixes on user-role messages (observed client marker:
# "[tool result] ...", plus common shapes from other clients).
_TOOL_RESULT_PREFIXES = (
    "[tool", "[tool result", "[function result", "[file result",
    "<tool_result", "<result>", "tool result", "tool ran without output",
    "tool call:", "[tool_call", "tool output:", "tool response:",
    # VS Code Copilot Chat sends terminal output as a user-role echo
    # ("Terminal output: bash: warning: ..."). It is command output, not user
    # intent — skipping it lets the walk continue to the real instruction.
    "terminal output:",
)

# Client-injected context wrappers some agents prepend as role="user" messages
# (attachments / browser pages / environment context). They are NOT user
# instructions and must be skipped like tool results — otherwise the newest
# "genuine instruction" becomes "<attachments> <attachment id=...> No bro...".
# Client-injected wrappers arrive as role="user" messages. We deliberately do
# NOT enumerate client tag names (<attachments>, <context>, ...) — a new client
# could add a new wrapper tomorrow and a list breaks. The signal is STRUCTURAL:
# a wrapper is DELIMITED (starts with an angle bracket "<tag" or a square
# bracket "[label]"). Genuine user text starts with neither, so a client that
# adds a brand-new wrapper tag is still caught. The only named shapes kept here
# are non-delimited phrases that are unmistakably client-generated.
_CLIENT_CONTEXT_PHRASES = (
    # VS Code Copilot Chat model-feedback echo.
    "you just executed tool calls but returned an empty response",
)

# Username / participant prefix Copilot prepends to a message: "[alice] hi".
# We STRIP the prefix (not skip the message) so the real instruction survives.
_MENTION_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+", re.IGNORECASE)


# Regex shapes that mark tool/test-run output rather than an instruction.
_TOOL_RESULT_PATTERNS = (
    r"ran \d+ tests?", r"\d+/\d+ tests?",
    r"all tests? passed", r"\d+ tests? (passed|failed|skipped)",
    r"traceback \(most recent call last\)", r"exit code[:\s]\d+",
    r"command finished", r"\d+ passed", r"\d+ failed",
)

# Continuation / acknowledgement messages with no new intent. Skipping them is
# only meaningful when a real instruction exists earlier — the backward walk
# naturally continues past them to find it, and the first-user-message fallback
# covers a conversation that is continuations only.
_CONTINUATIONS = {
    "continue", "please continue", "continue please", "yes", "yeah", "yep",
    "ok", "okay", "go on", "keep going", "and then", "next", "proceed",
    "sounds good", "thanks", "thank you", "perfect", "great", "go ahead",
    "sure", "alright", "got it", "cool", "more", "ok then",
}

# Short instructions that must NOT be treated as continuations (e.g. "fix it",
# "make it work", "explain this").
_SHORT_INSTRUCTION_STARTERS = (
    "fix", "write", "make", "create", "do", "explain", "plan", "debug",
    "run", "show", "give", "add", "change", "update", "remove", "help",
    "implement", "test", "review", "refactor", "analyze", "solve", "design",
)


def _content_text(msg: dict) -> str:
    """Extract plain text from a message's content (str or content blocks)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block and isinstance(block["text"], str):
                parts.append(block["text"])
            elif block.get("type") == "tool_result":
                c = block.get("content")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            parts.append(sub["text"])
        return " ".join(parts)
    return ""


def _message_has_tool_calls(msg: dict) -> bool:
    """True when an assistant message issued tool calls (OpenAI ``tool_calls``
    or Anthropic ``tool_use`` blocks)."""
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_call")
            for b in content
        )
    return False


def _has_tool_result_blocks(msg: dict) -> bool:
    """True when a user message carries Anthropic ``tool_result`` blocks."""
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _matches_tool_result_patterns(text: str) -> bool:
    """True when *text* looks like tool/test output rather than an instruction."""
    lower = (text or "").strip().lower()
    if not lower:
        return False
    for prefix in _TOOL_RESULT_PREFIXES:
        if lower.startswith(prefix):
            return True
    for pat in _TOOL_RESULT_PATTERNS:
        if re.search(pat, lower):
            return True
    if lower.lstrip().startswith("diff --git"):
        return True
    # Bare JSON array (common tool-result shape).
    if lower.startswith("[") and lower.endswith("]") and '"' in lower:
        return True
    return False


def _is_client_context(text: str) -> bool:
    """True when *text* is a client-injected wrapper rather than a genuine user
    instruction.

    Harness-agnostic: no tag/bracket NAMES are enumerated. A wrapper is
    DELIMITED — it begins with an angle bracket (XML-style tag) or a square
    bracket (mention / reply-quote / harness notice). Genuine user text starts
    with neither, so a client that adds a brand-new wrapper tag is still caught.
    """
    lower = (text or "").strip().lower()
    if not lower:
        return False
    if lower[0] == "<":
        return True
    if lower[0] == "[":
        # Bracket-wrapped: a mention / reply-quote / harness notice. But tool
        # echoes also arrive as "[tool result] …" / "[tool] …" — those are tool
        # output, not client context, so leave them for the tool-result path.
        return not _matches_tool_result_patterns(text)
    return any(lower.startswith(p) for p in _CLIENT_CONTEXT_PHRASES)


# The client's EXPLICIT user-message container. When a harness wraps the real
# request in <userRequest>...</userRequest> (VS Code Copilot Chat does), that
# content IS the user instruction — the single most reliable signal we have.
# Allowed spellings: userRequest / user_request / user-request / user request.
_USER_REQUEST_RE = re.compile(
    r"<\s*user[\s_-]*request\b[^>]*>(.*?)<\s*/\s*user[\s_-]*request\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Innermost paired XML-style tag (ANY name) together with its content. Metadata
# wrappers (attachments, context, editorContext, ...) are stripped INCLUDING
# their content — no names are enumerated; any tag that is not the user-request
# container is metadata.
_INNER_TAG_PAIR_RE = re.compile(
    r"<\s*[a-zA-Z_][\w.-]*\b[^>]*>[^<]*<\s*/\s*[a-zA-Z_][\w.-]*\s*>",
    re.DOTALL,
)

# A self-closing tag (``<tag …/>``) — metadata with no separate close.
_SELF_CLOSING_TAG_RE = re.compile(r"<\s*[a-zA-Z_][\w.-]*\b[^>]*/\s*>")

# An opening tag (``<tag …>``), used to detect UNCLOSED wrappers: if one remains
# after stripping complete pairs, the wrapper was truncated and the text after
# it is wrapper body, not an instruction.
_OPENING_TAG_RE = re.compile(r"<\s*[a-zA-Z_][\w.-]*\b[^>]*>")


def _strip_complete_pairs(raw: str) -> str:
    """Remove complete ``<tag>…</tag>`` pairs (INCLUDING their content) and
    self-closing ``<tag …/>`` tags, innermost-first.

    Metadata wrappers (attachments, context, editorContext, …) are stripped
    regardless of name — no client tag names are enumerated. Unclosed opening
    tags and genuine free text are left in place.
    """
    prev = None
    while prev != raw:
        prev = raw
        raw = _INNER_TAG_PAIR_RE.sub("", raw)
        raw = _SELF_CLOSING_TAG_RE.sub("", raw)
    return raw


def _strip_bracket_segments(raw: str) -> str:
    """Strip LEADING ``[...]`` segments (mention / reply-quote / harness notice
    prefix). Interior brackets in real text are preserved. Returns the tail."""
    while raw.startswith("["):
        end = raw.find("]")
        if end == -1:
            return raw  # unclosed — not a segment we can strip
        raw = raw[end + 1:]
    return raw


def _context_tail(text: str) -> Optional[str]:
    """Return the genuine user instruction out of a client-injected wrapper.

    Harness-agnostic (no tag-name enumeration):
      1. ``<userRequest>…</userRequest>`` — the client's own "this is the user
         message" container wins outright (its content is the instruction).
      2. A leading ``[…]`` segment (mention, ``[Replying to: "…"]`` quote,
         harness notice) is stripped; the remainder is the instruction.
      3. Any other delimited tags are stripped as metadata (INCLUDING their
         content); the remaining free text is the instruction.

    Returns None when no genuine instruction remains (attachment-only or
    truncated wrappers), so the walk keeps going back to a real user message.
    """
    if not _is_client_context(text):
        return None
    raw = (text or "").strip()

    # 1. The client's explicit user-message container.
    m = _USER_REQUEST_RE.search(raw)
    if m:
        content = m.group(1).strip(" \n\t:-")
        content = _strip_chat_mentions(_strip_mention(content))
        return content if content and content.lower().lstrip() not in _CONTINUATIONS else None

    # 2. Bracketed prefix (mention / reply-quote / notice).
    if raw.startswith("["):
        stripped = _strip_bracket_segments(raw)
        if stripped == raw:
            return None  # unclosed bracket — truncated/unknown notice, not intent
        rest = _strip_chat_mentions(stripped).strip(" \n\t:-")
        return rest if rest and rest.lower().lstrip() not in _CONTINUATIONS else None

    # 3. XML-style wrapper without an explicit user-request container.
    #    Remove complete tag pairs (incl. content). If an UNCLOSED opening tag
    #    remains, the wrapper was truncated — the text after it is wrapper body,
    #    not an instruction — so only text BEFORE it can be intent (usually
    #    none, so the walk continues back to a real user message).
    rest = _strip_complete_pairs(raw)
    m = _OPENING_TAG_RE.search(rest)
    if m:
        rest = rest[:m.start()]
    rest = _strip_chat_mentions(rest).strip(" \n\t:-")
    return rest if rest and rest.lower().lstrip() not in _CONTINUATIONS else None


def _strip_client_context_from_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with client-injected context content removed.

    Attachments / context / env / editor / reminder blocks are not user intent,
    so they must not feed any routing heuristic (token count, casual scan). A
    real instruction trailing a wrapper is preserved as that message's content;
    wrapper-only messages are dropped.
    """
    out: list[dict] = []
    for msg in messages or []:
        text = _content_text(msg)
        if _is_client_context(text):
            tail = _context_tail(text)
            if not tail or tail.lower().lstrip() in _CONTINUATIONS:
                continue  # wrapper-only → no intent, drop
            m = dict(msg)
            m["content"] = tail
            out.append(m)
        else:
            out.append(msg)
    return out


def _strip_mention(text: str) -> str:
    """Strip a leading ``[username]`` / ``[participant]`` prefix, keeping the
    real instruction that follows (e.g. ``[aunttwister] can you set the SEO
    job...`` -> ``can you set the SEO job...``)."""
    return _MENTION_PREFIX_RE.sub("", text, count=1).strip()


# VS Code Copilot Chat inserts inline mention tokens (``@file:src/foo.py``,
# ``@selection:...``, ``@terminal:1``, ``@workspace:...``, ...) INTO the user's
# text, plus paste references (``#attachment:Pasted text #1``). These are
# client-injected references, not user intent — and their incidental tokens
# skew semantic classification (a "let's plan for @file:component-runtime.md"
# message tipped to code_generation because of the "file"/"component" tokens).
# Strip them position-independently, mirroring how we strip the XML context
# blocks. Structural: any ``@word:content`` token, plus the ``#attachment``
# paste-reference namespace. Emails (``a@b.com``, no colon after the local
# part) are left untouched.
_CHAT_MENTION_RE = re.compile(r"@[a-zA-Z][\w-]*:[^\s]+")
# VS Code paste reference: "#attachment:Pasted text #1".
_ATTACHMENT_REF_RE = re.compile(r"#attachment:[^\n]*")


def _strip_chat_mentions(text: str) -> str:
    """Remove client-injected ``@mention:…`` tokens and ``#attachment`` paste
    references from *text*."""
    if not text or ("@" not in text and "#attachment" not in text):
        return text
    return _ATTACHMENT_REF_RE.sub("", _CHAT_MENTION_RE.sub("", text)).strip()


# ── Harness-agnostic system-prompt preamble detection ────────────────────────
# A system prompt sent as role=user (VS Code-style harness) has NO harness-
# specific text we can rely on (each harness words it differently, and the
# model name may be interpolated). But it is STRUCTURALLY distinct from a real
# user message: it is LONG and GENERIC (a multi-sentence block with no concrete
# task instruction). We detect it by those properties only — position (first
# user message) + length + genericity. Deterministic, harness-agnostic.

# Minimum length (chars) before we treat a user message as a "system prompt"
# candidate — a real first user message ("debug this", "hi", "write tests") is
# short; a system prompt is a long block.
_PREAMBLE_MIN_LEN = 120


def _preamble_head(text: str) -> str:
    """Return the text BEFORE the first blank line (the preamble block), so a
    preamble with an appended instruction is still recognized as preamble-like
    on its own head."""
    raw = (text or "").strip()
    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    return parts[0].strip()


def _is_preamble_like(text: str) -> bool:
    """True when *text* (or its leading block, if it appends an instruction) is
    a system-prompt preamble (any harness).

    Structural: long + generic (no CONCRETE task keyword). A short message or a
    real instruction that merely mentions a task word is not preamble-like.
    """
    head = _preamble_head(text)
    if not head:
        return False
    norm = re.sub(r"\s+", " ", head.lower())
    if len(norm) < _PREAMBLE_MIN_LEN:
        return False
    if _is_continuation(head):
        return False
    # Generic => no concrete task signal. Single-word task keywords ("debug",
    # "error", "bug", "explain", ...) appear INCIDENTALLY in any preamble's
    # boilerplate, so only strong multi-word / specific phrases count.
    for task in _SPECIFIC_TASKS:
        for kw in TASK_SIGNALS[task]:
            if " " not in kw and len(kw) <= 6:
                continue
            if kw in norm:
                return False
    return True


def _preamble_tail(text: str) -> Optional[str]:
    """When a preamble-like message appends a REAL instruction after a blank
    line (``<preamble>\\n\\n<real request>``), return the tail; else None."""
    if not _is_preamble_like(text):
        return None
    raw = (text or "").strip()
    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) < 2:
        return None
    rest = parts[1].strip(" \n\t:-")
    norm = rest.lower().lstrip()
    if not rest or norm in _CONTINUATIONS:
        return None
    return rest


def _is_tool_result(msg: dict, msgs: list[dict], i: int) -> bool:
    """True when a user-role message is really a tool result (schema violation).

    Detection layers, most structural first:
      1. Explicit markers: ``tool_call_id`` (OpenAI) or ``tool_result`` blocks.
      2. The message immediately before is the assistant that MADE the call.
      3. Content heuristics (prefixes, test-run/traceback/exit-code shapes).
    """
    if msg.get("tool_call_id") is not None:
        return True
    if _has_tool_result_blocks(msg):
        return True
    j = i - 1
    if (j >= 0 and msgs[j].get("role") == "assistant"
            and _message_has_tool_calls(msgs[j])):
        return True
    return _matches_tool_result_patterns(_content_text(msg))


def _is_continuation(text: str) -> bool:
    """True when *text* is a continuation/acknowledgement with no new intent."""
    norm = re.sub(r"[^a-z0-9\s]", "", (text or "").strip().lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in _CONTINUATIONS:
        return True
    words = norm.split()
    if len(words) <= 3:
        # A short message with a task keyword or an instruction starter is intent.
        for task in _SPECIFIC_TASKS:
            for kw in TASK_SIGNALS[task]:
                if kw in norm:
                    return False
        if any(w in _SHORT_INSTRUCTION_STARTERS for w in words):
            return False
        return True
    return False


def _extract_intent_text(messages: list[dict]) -> tuple[str, dict]:
    """Return (intent_text, meta) — the NEWEST genuine user instruction.

    Walks the conversation backward (most recent first), skipping system /
    assistant / tool-role messages, tool results sent as role="user", client
    context wrappers (attachments / reply-quotes / model-feedback), system-
    prompt preambles sent as role="user" (VS Code-style harnesses), and
    continuation acknowledgements. The first survivor is the current intent.
    Falls back to the first user message when nothing genuine survives.

    A preamble that appends a real instruction after a blank line keeps the tail
    as the intent. When the ONLY candidate is a preamble, it is returned with
    ``meta["preamble"] = True`` so the classifier can neutralize it (route to a
    neutral default) instead of keyword-matching the boilerplate.
    """
    meta = {"source": "none", "skipped_tool": 0, "skipped_cont": 0,
            "skipped_context": 0, "skipped_preamble": 0, "preamble": False}
    msgs = messages or []
    if not msgs:
        return "", meta
    first_user_text = ""
    last_preamble_text = ""
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        role = msg.get("role")
        if role == "system":
            continue
        text = _content_text(msg)
        if role == "assistant":
            continue
        if role == "tool":
            meta["skipped_tool"] += 1
            continue
        # role == "user"
        if text.strip() and not _is_client_context(text) and not _is_preamble_like(text):
            first_user_text = text  # overwrite → ends as the earliest user msg
        if _is_tool_result(msg, msgs, i):
            meta["skipped_tool"] += 1
            continue
        if _is_client_context(text):
            tail = _context_tail(text)
            if tail and tail.lower().lstrip() not in _CONTINUATIONS:
                # A wrapper that appends a real instruction keeps the tail.
                meta["source"] = "last_instruction"
                return tail, meta
            # Attachments / browser pages / env context are not instructions —
            # keep walking back to the real user message.
            meta["skipped_context"] += 1
            continue
        if _is_preamble_like(text):
            tail = _preamble_tail(text)
            if tail and tail.lower().lstrip() not in _CONTINUATIONS:
                # A preamble that appends a real instruction keeps the tail.
                meta["source"] = "last_instruction"
                return tail, meta
            # Preamble-only user message → not an instruction. Remember it so
            # the classifier can neutralize it if nothing genuine survives.
            last_preamble_text = text
            meta["skipped_preamble"] += 1
            meta["preamble"] = True
            continue
        if _is_continuation(text):
            # The backward walk keeps looking for the last real instruction; the
            # first-user-message fallback covers an all-continuation tail.
            meta["skipped_cont"] += 1
            continue
        meta["source"] = "last_instruction"
        stripped = _strip_chat_mentions(_strip_mention(text))
        return (stripped or text).strip(), meta
    # Nothing genuine survived: if we saw a preamble, return it flagged so the
    # classifier can route to a neutral default (path=preamble).
    if last_preamble_text:
        meta["source"] = "preamble"
        return last_preamble_text.strip(), meta
    meta["source"] = "first_user_fallback"
    stripped = _strip_chat_mentions(_strip_mention(first_user_text))
    return (stripped or first_user_text).strip(), meta
