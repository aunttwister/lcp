"""Intelligent model routing — prompt classification → capability scoring → best-fit model.

Routing strategies, from simple to smart:
  1. CapabilityRouter — task classification + benchmark-derived scores (recommended)
  2. Disabled — static chain (current default)

The CapabilityRouter loads per-model scores from the model_capabilities DB table,
classifies each incoming prompt into a task type (agentic, unit tests, coding,
debugging, reasoning, planning, chat), and scores all available models to pick
the best fit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .cost_estimator import count_tokens
from .logging_config import get_logger

logger = get_logger("lcp.router")


# ── Model-ID normalization (llama.cpp gguf paths, quantization) ──────────────

# Common GGUF quantization tags, matched as the last path segment.
_QUANT_RE = __import__("re").compile(
    r"(q\d+[a-z]*(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32|i?q\d+_\d+|[a-z]+\d+(?:\.\d+)?b)"
)


def normalize_model_id(model: str) -> str:
    """Normalize a raw provider-side model ID into a clean logical name.

    - strips ``/models/`` and leading slashes (llama.cpp file paths)
    - strips the ``.gguf`` extension
    - lowercases and collapses whitespace

    e.g. ``/models/qwen3.6-27b-q4_k_m.gguf`` → ``qwen3.6-27b-q4_k_m``.
    """
    if not model:
        return model
    name = str(model).strip()
    # Strip path prefix like /models/ or leading slash
    if name.startswith("/models/"):
        name = name[len("/models/"):]
    elif name.startswith("/"):
        name = name.lstrip("/")
    # Take the last path segment
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.strip().lower()


def detect_quantization(model: str) -> Optional[str]:
    """Return a quantization tag like ``Q4_K_M``, or None.

    Looks for GGUF quantization tokens (``q4_k_m``, ``q4_0``, ``f16``, …) in
    the model ID. Returns the canonical uppercase form when found.
    """
    if not model:
        return None
    name = str(model).strip().lower()
    if name.endswith(".gguf"):
        name = name[:-5]
    # Strip any path so we look at the filename stem.
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    m = _QUANT_RE.search(name)
    if not m:
        return None
    return m.group(1).upper()


# ── Task classification (keyword/heuristic, no ML dependency) ────────────────

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

CASUAL_SIGNALS = [
    "hello", "hi ", "hey", "thanks", "thank you", "how are you",
    "what's up", "good morning", "good night",
]

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

# ── Harness-agnostic system-prompt preamble detection ────────────────────────
# Some harnesses (VS Code Copilot Chat, OpenCode, ...) send their system prompt
# as a role="user" message because their chat API has no System role (VS Code's
# LanguageModelChatMessageRole is only User/Assistant). We do NOT hardcode any
# harness's prompt text — the signal is STRUCTURAL: a preamble is long + generic
# (no concrete task instruction). This handles any harness, now or future.
# (Standard APIs — OpenAI developer/system, Anthropic system field — already
# arrive as role=system and are skipped by the walk.)

# A user message longer than this with no concrete task signal is preamble-like.
_PREAMBLE_MIN_LEN = 120

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


@dataclass
class ClassifyResult:
    """Full rationale for a routing classification decision.

    ``task`` is the final task string (the only thing the old ``classify_task``
    returned); the rest explains WHY, so a decision can be replayed and judged
    later.
    """
    task: str
    path: str                     # semantic | agentic_prompt | preamble |
                                  #   tool_count | token_count | casual | default
    keyword: Optional[str] = None  # matched signal keyword (agentic/casual) or None
    intent_text: str = ""          # the "newest genuine user instruction" classified
    intent_meta: Optional[dict] = None  # {source, skipped_tool, skipped_cont}
    semantic: Optional[list] = None     # top-N (task, score) or None
    min_score: Optional[float] = None   # semantic gate applied (or None)
    sem_available: bool = False         # embedder was up
    tool_count: int = 0
    token_count: int = 0


def classify_task_detail(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 1024,
) -> ClassifyResult:
    """Classify a request into a task type, returning the FULL rationale.

    Same decision order as ``classify_task`` (task strings are identical) but
    also records which stage won (``path``), the matched keyword, the intent
    message, and the semantic scores. See ``classify_task`` for the strategy
    details.
    """
    # Gather USER text and the FULL conversation text (user + assistant + tool)
    # for casual detection. Neither includes the system prompt. Client-injected
    # context (attachments / env / editor / reminders) is NOT user intent, so it
    # is excluded from BOTH inputs — otherwise incidental words inside an
    # attached document ("error", "pytest", ...) hijack keyword/casual routing.
    # A real instruction trailing a wrapper is preserved via _context_tail.
    user_text = ""
    combined = ""
    for msg in messages or []:
        if msg.get("role") == "system":
            continue
        text = _content_text(msg)
        if _is_client_context(text):
            tail = _context_tail(text)
            if tail and tail.lower().lstrip() not in _CONTINUATIONS:
                chunk = _strip_chat_mentions(tail).lower() + " "
                combined += chunk
                if msg.get("role") == "user":
                    user_text += chunk
            continue
        chunk = _strip_chat_mentions(text).lower() + " "
        combined += chunk
        if msg.get("role") == "user":
            user_text += chunk

    system_text = ""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if isinstance(content, str):
            system_text = content.lower()

    # 1. The CURRENT intent: the newest GENUINE user instruction. We walk the
    #    conversation backward, skipping assistant/system/tool messages, tool
    #    results that some clients send with role="user" (a schema violation —
    #    their "[tool result] Ran 12 tests" echoes hijack intent), and
    #    continuation acknowledgements ("continue", "yes", …) which carry no
    #    new intent. A mid-session "let's plan the next feature" therefore
    #    reclassifies instead of inheriting the first message's task forever.
    intent_text_raw, intent_meta = _extract_intent_text(messages or [])
    intent_lc = intent_text_raw.strip().lower() or user_text.strip()
    intent_text = intent_text_raw.strip() or user_text.strip()

    tool_count = len(tools) if tools else 0

    # 1.0 Preamble neutralization: the only candidate intent is a system-prompt
    #     preamble (VS Code-style harness sent its system prompt as role=user).
    #     A system prompt is NEVER a task — keyword-matching it would misroute
    #     (e.g. the preamble's incidental "debugging" -> debugging). Route to a
    #     neutral agentic default and record path="preamble" so it's visible.
    if intent_meta and intent_meta.get("preamble"):
        return ClassifyResult(
            task="agentic_multi_step", path="preamble",
            intent_text=intent_text, intent_meta=intent_meta,
            tool_count=tool_count,
        )

    # 1. SEMANTIC classification — the SOLE task classifier. Meaning
    #    (embedding similarity to per-task exemplar centroids) determines the
    #    task type. There is deliberately NO keyword fallback: keyword lists
    #    are brittle, hardcode vocabulary, and drift across harnesses and
    #    phrasings (a typo like "ake a plan" misses every planning keyword).
    #    When the embedder is unavailable or the top score is below the
    #    confidence gate, we fall through to the structural degraded-mode
    #    heuristics below (agentic prompt / tool count / token count / casual).
    semantic: Optional[list] = None
    min_score: Optional[float] = None
    sem_available = False
    try:
        from .task_classifier import get_semantic_classifier
        clf = get_semantic_classifier()
        if clf is not None and intent_lc.strip():
            sem_available = True
            min_score = clf.min_score
            scores = clf.top_scores(intent_lc.strip(), 5)
            if scores:
                semantic = [(t, round(s, 4)) for t, s in scores]
                if scores[0][1] >= clf.min_score:
                    return ClassifyResult(
                        task=scores[0][0], path="semantic",
                        intent_text=intent_text, intent_meta=intent_meta,
                        semantic=semantic, min_score=min_score,
                        sem_available=True, tool_count=tool_count,
                    )
    except Exception:  # noqa: BLE001 — never let classification break routing
        pass

    # 2. Agentic system prompt — the generic agent preamble.
    for kw in TASK_SIGNALS["agentic_multi_step"]:
        if kw in system_text:
            return ClassifyResult(
                task="agentic_multi_step", path="agentic_prompt", keyword=kw,
                intent_text=intent_text, intent_meta=intent_meta,
                semantic=semantic, min_score=min_score,
                sem_available=sem_available, tool_count=tool_count,
            )

    # Tool count signal — many tools = agentic
    if tool_count > 5:
        return ClassifyResult(
            task="agentic_multi_step", path="tool_count",
            intent_text=intent_text, intent_meta=intent_meta,
            semantic=semantic, min_score=min_score,
            sem_available=sem_available, tool_count=tool_count,
        )

    # Token count signal — very long prompt = research_deep. Count WITHOUT
    # client-injected context so a large attachment can't push a short request
    # over the threshold.
    token_count = count_tokens(_strip_client_context_from_messages(messages), tools)
    if token_count > 8000:
        return ClassifyResult(
            task="research_deep", path="token_count",
            intent_text=intent_text, intent_meta=intent_meta,
            semantic=semantic, min_score=min_score,
            sem_available=sem_available, tool_count=tool_count,
            token_count=token_count,
        )

    # Check casual signals over the full conversation (casual is harmless and
    # the user may have greeted before the assistant/tool context).
    for kw in CASUAL_SIGNALS:
        if kw in combined:
            return ClassifyResult(
                task="casual_chat", path="casual", keyword=kw,
                intent_text=intent_text, intent_meta=intent_meta,
                semantic=semantic, min_score=min_score,
                sem_available=sem_available, tool_count=tool_count,
            )

    # Default: the most common LCP use case is agentic coding
    return ClassifyResult(
        task="code_generation", path="default",
        intent_text=intent_text, intent_meta=intent_meta,
        semantic=semantic, min_score=min_score,
        sem_available=sem_available, tool_count=tool_count,
    )


def classify_task(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 1024,
) -> str:
    """Classify a request into a task type.

    The CURRENT intent — the newest GENUINE user instruction — is walked
    backward from the most recent message, skipping assistant/tool messages,
    tool results sent with role="user" (a schema violation), client context
    wrappers (attachments / env / editor / reminders), system-prompt
    preambles, and continuation acknowledgements.

    Priority (first match wins):
      1. Semantic classification (embedding similarity to per-task exemplar
         centroids) of the intent message — the SOLE task classifier. No
         keyword fallback: meaning drives the task, not exact strings.
      2. Agentic system prompt ("you are an AI agent", "tools:", …).
      3. Tool-count / token-count heuristics, then casual (over the full
         conversation), then the code_generation default — degraded-mode
         fallbacks when the embedder is unavailable or below its gate.

    Returns only the task string; use ``classify_task_detail`` for the full
    rationale.
    """
    return classify_task_detail(messages, tools, max_tokens).task


def _summarize_conversation(
    messages: list[dict],
    max_content: int = 200,
    max_total: int = 4000,
) -> list[dict]:
    """Return a shape-preserving, content-trimmed copy of *messages*.

    Keeps ``role``, ``tool_calls`` (id + name + trimmed args) and
    ``tool_call_id`` so the structural checks in ``_extract_intent_text``
    behave identically when the summary is replayed. Content is trimmed from
    the END so leading tool-result markers (``[tool result]``, ``<tool_result``,
    ...) survive the trim. If the serialized size still exceeds ``max_total``,
    the OLDEST messages are dropped (the intent walk is newest-first) and a
    placeholder system message records how many were dropped.
    """

    def _trim(text, limit):
        if not text:
            return ""
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + f"… [{len(text) - limit} chars omitted]"

    def _one(msg):
        out = {}
        if msg.get("role") is not None:
            out["role"] = msg["role"]
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                out["content"] = _trim(content, max_content)
        elif isinstance(content, list):
            blocks = []
            for b in content[:8]:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    blocks.append({"type": "tool_result",
                                   "content": _trim(_content_text(b), 120)})
                elif b.get("type") == "text" and b.get("text"):
                    blocks.append({"type": "text", "text": _trim(b["text"], max_content)})
                elif b.get("type") == "image_url":
                    blocks.append({"type": "image_url"})
                else:
                    blocks.append({"type": b.get("type", "unknown")})
            if blocks:
                out["content"] = blocks
        if msg.get("tool_call_id"):
            out["tool_call_id"] = msg["tool_call_id"]
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            kept = []
            for tc in tcs[:8]:
                if not isinstance(tc, dict):
                    continue
                item = {"id": tc.get("id", "")}
                if tc.get("type"):
                    item["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn:
                    item["function"] = {"name": fn.get("name", "")}
                    if fn.get("arguments"):
                        item["function"]["arguments"] = _trim(str(fn["arguments"]), max_content)
                kept.append(item)
            if kept:
                out["tool_calls"] = kept
        return out

    summarized = [_one(m) for m in (messages or [])]
    if not summarized:
        return []
    dropped = 0
    while len(summarized) > 1 and len(json.dumps(summarized, ensure_ascii=False)) > max_total:
        summarized = summarized[1:]
        dropped += 1
    if dropped:
        summarized.insert(0, {"role": "system", "content": f"[{dropped} older messages omitted]"})
    return summarized


# ── CapabilityRouter — DB-backed, task-classifying, N-model scorer ────────────

# Default scores for models not in the DB (conservative: assume pro-level)
DEFAULT_CAPABILITY: dict[str, float] = {
    "deepseek-v4-pro": 0.85,
    "deepseek-v4-flash": 0.65,
}

# Cost bias: how much to boost cheaper models (0.0 = pure capability, 0.3 = strong cost bias)
DEFAULT_COST_BIAS = 0.15

# Hysteresis: only override/reorder when the best step beats the default by this much.
_HYSTERESIS = 0.05

# Known model pricing (USD per 1M output tokens) — from gateway.yaml
_MODEL_PRICES: dict[str, float] = {
    "deepseek-v4-pro": 0.87,
    "deepseek-v4-flash": 0.27,
}

# Health bonus per circuit-breaker status (adds to a step's score).
_HEALTH_BONUS: dict[str, float] = {
    "healthy": 0.05,
    "degraded": -0.03,
    "dead": -0.25,
}
# Score penalty when a provider's cached usage suggests it is running low.
_LOW_CREDIT_PENALTY = -0.10


# ── Model registry — DB-backed, explicit, no runtime name parsing ────────────
#
# Each provider uses its own model-ID convention, and each benchmark publishes
# its own (often dated) names. Instead of guessing from string patterns, we
# keep ONE explicit registry (persisted in the ``model_registry`` table) that
# maps every provider-side model ID back to a logical name and pins that
# logical name to the benchmark snapshot it should be scored by.
#
#   logical_name:   the canonical gateway name (also the key in _MODEL_PRICES
#                   and pricing configs — used for pricing and aggregation).
#   benchmark_key:  the STABLE, release-independent model key inside the seeded
#                   capability matrix (LiveBench / Arena data).
#   provider_mappings: {provider: provider-side model ID}. The mapping VALUES
#                   are the provider-side spellings the reverse index is
#                   built from.
#
# The curated defaults live in seed_capabilities.DEFAULT_MODEL_REGISTRY and are
# seeded into the DB on first run. After seeding, the DB is the source of truth
# and is editable via the /models page — no code changes required when a
# provider rolls a new dated snapshot.


_registry_cache: Optional[dict[str, dict]] = None
_registry_db_path: Optional[str] = None


def get_model_registry(db_path: str = "data/costs.db") -> dict[str, dict]:
    """Return the model registry (cached), loading/seeding from DB as needed."""
    global _registry_cache, _registry_db_path
    if _registry_cache is not None and _registry_db_path == db_path:
        return _registry_cache
    from .seed_capabilities import load_model_registry, seed_model_registry
    registry = load_model_registry(db_path)
    if not registry:
        seed_model_registry(db_path)
        registry = load_model_registry(db_path)
    _registry_cache = registry
    _registry_db_path = db_path
    return _registry_cache


def invalidate_registry_cache() -> None:
    """Clear the cached registry so the next lookup re-reads the DB."""
    global _registry_cache, _registry_db_path
    _registry_cache = None
    _registry_db_path = None


def _alias_to_logical(registry: dict[str, dict]) -> dict[str, str]:
    """Build reverse index: provider-side model ID / benchmark key → logical name."""
    index: dict[str, str] = {}
    for logical, entry in registry.items():
        index[logical.lower()] = logical
        if entry.get("benchmark_key"):
            index.setdefault(entry["benchmark_key"].lower(), logical)
        for provider_side in (entry.get("provider_mappings") or {}).values():
            if provider_side:
                index.setdefault(provider_side.lower(), logical)
    return index


def logical_model_name(model: str, db_path: str = "data/costs.db") -> str:
    """Map any model ID to its logical gateway name via the DB registry.

    Unknown names are normalized (strips ``/models/`` prefix and ``.gguf``
    extension, lowercased) so a llama.cpp path like
    ``/models/qwen3.6-27b-q4_k_m.gguf`` resolves to ``qwen3.6-27b-q4_k_m``.
    """
    if not model:
        return model
    registry = get_model_registry(db_path)
    key = model.strip().lower()
    mapped = _alias_to_logical(registry).get(key)
    if mapped:
        return mapped
    return normalize_model_id(key)


def benchmark_model_name(logical: str, db_path: str = "data/costs.db") -> str:
    """Return the benchmark snapshot key for a logical model name."""
    registry = get_model_registry(db_path)
    entry = registry.get(logical)
    if entry:
        return entry["benchmark_key"]
    return logical


def provider_model_name(logical: str, provider: str, db_path: str = "data/costs.db") -> str:
    """Return the provider-side model ID for a logical model + provider.

    Uses the registry's explicit ``provider_mappings`` (e.g. Command Code's
    ``deepseek/deepseek-v4-pro`` vs OpenCode's bare ``deepseek-v4-pro``).
    Falls back to the logical name unchanged when the provider is unmapped.
    """
    registry = get_model_registry(db_path)
    entry = registry.get(logical)
    if entry:
        mapping = entry.get("provider_mappings") or {}
        if provider in mapping:
            return mapping[provider]
    return logical


class CapabilityRouter:
    """Routes to the best available model using capability scores + cost awareness."""

    def __init__(
        self,
        enabled: bool = False,
        db_path: str = "data/costs.db",
        cost_bias: float = DEFAULT_COST_BIAS,
    ):
        self.enabled = enabled
        self.db_path = db_path
        self.cost_bias = cost_bias
        self._matrix: Optional[dict[str, dict[str, float]]] = None
        # Bounded log of recent routing decisions, surfaced in the UI
        # (/api/routing/status, Providers → Routing tab).
        self._decisions: list[dict] = []

    # ── Matrix ────────────────────────────────────────────────────────────

    def load_matrix(self) -> dict[str, dict[str, float]]:
        """Load capability matrix from DB. Cached in memory."""
        if self._matrix is not None:
            return self._matrix
        try:
            from .seed_capabilities import load_capability_matrix
            self._matrix = load_capability_matrix(self.db_path)
            logger.info("capability_matrix_loaded", tasks=len(self._matrix))
        except Exception as e:
            logger.warning("capability_matrix_load_failed", error=str(e))
            self._matrix = {}
        return self._matrix

    def invalidate_matrix(self) -> None:
        """Drop the cached matrix so the next call re-reads the DB.

        Called when a benchmark run completes or the registry changes, so
        routing picks up fresh scores without a restart.
        """
        self._matrix = None

    def _has_profile_override(self, profile: str) -> bool:
        """True when *profile* has any per-profile routing setting stored."""
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is None:
                return False
            from .cost_cache import SettingsStore
            for key in (
                f"{SettingsStore.ROUTING_ENABLED_KEY}:{profile}",
                f"{SettingsStore.ROUTING_POLICY_KEY}:{profile}",
                f"{SettingsStore.ROUTING_MIN_SCORE_KEY}:{profile}",
                f"{SettingsStore.ROUTING_RULES_KEY}:{profile}",
            ):
                if settings.get(key, None) is not None:
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    # ── Policy + decisions ────────────────────────────────────────────────

    def _effective_policy(self, config: Optional[object] = None,
                          profile: Optional[str] = None) -> tuple[str, float]:
        """Return (policy, min_score) for a scope: runtime settings override
        config.

        A per-profile override (``routing_policy:<profile>`` /
        ``routing_min_score:<profile>``) wins when set; otherwise the global
        setting; otherwise the config value. Policy ∈ {eager, cost_first,
        explore}; min_score is a 0–1 floor below which a reorder is never
        recommended.
        """
        policy = "eager"
        min_score = 0.0
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                try:
                    policy = settings.get_routing_policy(default=policy, profile=profile)
                except TypeError:
                    policy = settings.get_routing_policy(default=policy)
                try:
                    min_score = settings.get_routing_min_score(default=min_score, profile=profile)
                except TypeError:
                    min_score = settings.get_routing_min_score(default=min_score)
        except Exception:  # noqa: BLE001
            pass
        if config is not None:
            try:
                dr = config.dynamic_routing or {}
                policy = dr.get("policy", policy)
                min_score = float(dr.get("min_score", min_score))
            except Exception:  # noqa: BLE001
                pass
        return policy, min_score

    def is_enabled(self, config: Optional[object] = None,
                   profile: Optional[str] = None) -> bool:
        """Effective enabled state for a scope: a per-profile runtime toggle
        (``routing_enabled:<profile>``) wins, then the global toggle (settings
        table, UI), then the boot-time value (seeded from config).

        ``config`` is accepted for API symmetry (callers pass it through) but
        the boot-time ``self.enabled`` already reflects config at startup.
        """
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                try:
                    override = settings.get_routing_enabled(default=None, profile=profile)
                except TypeError:
                    override = settings.get_routing_enabled(default=None)
                if override is not None:
                    return override
        except Exception:  # noqa: BLE001 — toggle must never break routing
            pass
        return self.enabled

    def _decision_base(self, detail, messages, profile, policy, note, fired_desc) -> dict:
        """Shared decision fields for every ``_record_decision`` site.

        Carries the routing rationale (path, keyword, intent text, semantic
        top-N, semantic gate) plus a shape-preserving summary of the classified
        conversation so a decision can be replayed and judged later. All six
        ``select_step`` record sites merge this base with their action-specific
        fields, keeping the decision log consistent as the routing logic evolves.
        """
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "profile": profile or "",
            "task": detail.task,
            "policy": policy,
            "rules": fired_desc,
            "note": note,
            "path": detail.path,
            "keyword": detail.keyword,
            "intent_text": (detail.intent_text or "")[:500],
            "semantic_json": json.dumps(detail.semantic) if detail.semantic else None,
            "min_score": detail.min_score,
            "sem_available": detail.sem_available,
            "conversation_json": json.dumps(_summarize_conversation(messages), ensure_ascii=False),
        }

    def _record_decision(self, decision: dict) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > 50:
            self._decisions = self._decisions[-50:]
        # Persist to the DB so decisions survive restarts/rebuilds. Best-effort:
        # a DB failure must never break routing.
        try:
            from .models import RoutingDecision, get_session
            from .models import get_engine as _get_engine
            import json as _json
            with get_session(_get_engine(self.db_path)) as session:
                session.add(RoutingDecision(
                    ts=decision.get("ts"),
                    profile=decision.get("profile") or "",
                    task=decision.get("task") or "",
                    policy=decision.get("policy") or "",
                    action=decision.get("action") or "",
                    provider=decision.get("provider"),
                    model=decision.get("model"),
                    score=decision.get("score"),
                    rules_json=_json.dumps(decision.get("rules") or []),
                    from_provider=decision.get("from_provider"),
                    from_model=decision.get("from_model"),
                    note=decision.get("note"),
                    path=decision.get("path"),
                    keyword=decision.get("keyword"),
                    intent_text=decision.get("intent_text"),
                    semantic_json=decision.get("semantic_json"),
                    min_score=decision.get("min_score"),
                    sem_available=decision.get("sem_available"),
                    conversation_json=decision.get("conversation_json"),
                ))
                session.commit()
        except Exception:  # noqa: BLE001 — persistence must never break routing
            pass

    def recent_decisions(self, limit: int = 25) -> list[dict]:
        """Return the most recent routing decisions, newest first.

        Reads from the persisted ``routing_decisions`` table first (survives
        restarts); falls back to the in-memory buffer when the DB isn't
        available.
        """
        try:
            from .models import RoutingDecision, get_session
            from .models import get_engine as _get_engine
            import json as _json
            with get_session(_get_engine(self.db_path)) as session:
                rows = (
                    session.query(RoutingDecision)
                    .order_by(RoutingDecision.id.desc())
                    .limit(limit)
                    .all()
                )
                if rows:
                    return [{
                        "ts": r.ts, "profile": r.profile, "task": r.task,
                        "policy": r.policy, "action": r.action,
                        "provider": r.provider, "model": r.model,
                        "score": r.score,
                        "rules": _json.loads(r.rules_json) if r.rules_json else [],
                        "from_provider": r.from_provider, "from_model": r.from_model,
                        "note": r.note,
                        "path": r.path, "keyword": r.keyword,
                        "intent_text": r.intent_text,
                        "semantic_json": r.semantic_json,
                        "min_score": r.min_score, "sem_available": r.sem_available,
                        "conversation_json": r.conversation_json,
                    } for r in rows]
        except Exception:  # noqa: BLE001 — fall back to in-memory
            pass
        return list(self._decisions[-limit:])

    def get_model_score(self, model: str, task: str) -> float:
        """Get capability score for a model on a task (0.0–1.0).

        Resolves the model ID through the explicit registry: alias → logical
        name → benchmark snapshot, then looks up the score. Unknown models
        fall back to a default.
        """
        matrix = self.load_matrix()
        task_scores = matrix.get(task, {})

        logical = logical_model_name(model, self.db_path)
        benchmark = benchmark_model_name(logical, self.db_path)

        if benchmark != model or logical != model:
            logger.debug(
                "capability_resolved",
                model=model,
                logical=logical,
                benchmark=benchmark,
                task=task,
            )

        return task_scores.get(benchmark, DEFAULT_CAPABILITY.get(logical, 0.5))

    # ── Provider-aware selection (Phase 2) ────────────────────────────────

    def _health_bonus(self, step: dict, profile: Optional[str] = None,
                      config: Optional[object] = None) -> float:
        """Bonus/penalty from the circuit breaker's per-provider health.

        A 96% model on a degraded provider can lose to a 94% model on a
        healthy one. Returns 0 when the breaker is unavailable (e.g. tests).
        """
        try:
            from .circuit_breaker import get_circuit_breaker
            provider = step["provider"]
            base_url = step.get("base_url") or ""
            if config is not None and not base_url:
                base_url = (config.providers or {}).get(provider, {}).get("api_base", "")
            cb = get_circuit_breaker()
            status = cb.status_of(provider, base_url, profile or "")
            return _HEALTH_BONUS.get(status, 0.0)
        except Exception:  # noqa: BLE001 — tiebreaker must never break routing
            return 0.0

    def _provider_available(self, step: dict, profile: Optional[str] = None,
                            config: Optional[object] = None) -> bool:
        """True when the step's provider is available to the circuit breaker.

        This is the breaker's provider-level GATE (dead/hard-tripped providers
        are excluded). When dynamic routing is enabled, the router decides
        model selection + ordering and the breaker only gates providers — so
        the router never proposes a provider the breaker would skip. Returns
        True when the breaker is unavailable (e.g. tests) to never break
        routing.
        """
        try:
            from .circuit_breaker import get_circuit_breaker
            provider = step["provider"]
            base_url = step.get("base_url") or ""
            if config is not None and not base_url:
                base_url = (config.providers or {}).get(provider, {}).get("api_base", "")
            cb = get_circuit_breaker()
            return cb.is_available(provider, base_url, profile or "")
        except Exception:  # noqa: BLE001 — gate must never break routing
            return True

    def _provider_serves_model(self, provider: str, logical_model: str,
                               config: Optional[object] = None) -> bool:
        """True when *provider* exposes *logical_model* (per gateway.yaml models).

        Providers with no explicit model list are treated as serving the model
        (optimistic), so a prefer never silently drops a provider just because
        its model list is unconfigured. Never raises.
        """
        try:
            pcfg = (config.providers or {}).get(provider, {}) or {}
            models = pcfg.get("models") or []
            if not models:
                return True
            for m in models:
                if logical_model_name(m, self.db_path) == logical_model:
                    return True
        except Exception:  # noqa: BLE001 — never break routing
            return True
        return False

    def _credit_bonus(self, provider: str) -> float:
        """Penalty when a provider's cached usage suggests low credits.

        Reads the DB cost cache (never scrapes): opencode monthly% >= 95,
        commandcode remaining credits <= $5, deepseek balance <= $1.
        """
        try:
            from .cost_cache import get_cost_cache
            cache = get_cost_cache()
            if cache is None:
                return 0.0
            sub = cache.get(provider, "subscription")
            if sub:
                p = sub["payload"] or {}
                if p.get("_error"):
                    return _LOW_CREDIT_PENALTY
                if provider == "opencode":
                    mpct = p.get("monthly_pct")
                    if mpct is not None and mpct >= 95:
                        return _LOW_CREDIT_PENALTY
                elif provider == "commandcode":
                    rem = p.get("monthly_credits_remaining")
                    if rem is not None and rem <= 5.0:
                        return _LOW_CREDIT_PENALTY
            bal = cache.get(provider, "balance")
            if bal:
                p = bal["payload"] or {}
                b = p.get("balance")
                if isinstance(b, dict):
                    avail = b.get("available")
                    if avail is not None and avail <= 1.0:
                        return _LOW_CREDIT_PENALTY
        except Exception:  # noqa: BLE001 — tiebreaker must never break routing
            pass
        return 0.0

    def score_step(self, step: dict, task: str, profile: Optional[str] = None,
                   config: Optional[object] = None, bias: Optional[float] = None) -> float:
        """Score a single ``{provider, model}`` chain step.

        capability + cost-bias boost + health bonus + credit penalty.
        ``bias`` overrides the default cost bias (used by ``cost_first``).
        """
        model = step["model"]
        capability = self.get_model_score(model, task)
        price = _MODEL_PRICES.get(model, 1.0)
        max_price = max(_MODEL_PRICES.values()) if _MODEL_PRICES else 1.0
        cost_factor = price / max_price if max_price > 0 else 0.5
        b = self.cost_bias if bias is None else bias
        score = capability + b * (1.0 - cost_factor)
        score += self._health_bonus(step, profile, config)
        score += self._credit_bonus(step["provider"])
        return score

    # ── Deterministic model selection (unified rule & scoring) ─────────────
    #
    # The rule (``prefer``) path and the scoring (``reorder``) path MUST agree
    # on the same (provider, model). Today they diverge: the prefer rule
    # expands a model across providers (chain order), while scoring scores
    # literal chain steps — so ``deepseek-v4-flash`` resolved to
    # commandcode/…flash under a rule but deepseek/…flash under reorder.
    #
    # The unified algorithm: choose the target LOGICAL MODEL (provider-
    # agnostic), then resolve the provider ONCE in ``_build_chain_for_model`` —
    # the single chain-builder both paths funnel through.

    def _score_model(self, logical_model: str, task: str,
                     bias: Optional[float] = None) -> float:
        """Score a LOGICAL MODEL (provider-agnostic) for a task.

        capability + cost-bias boost. Deliberately NO health/credit — those are
        provider-level and belong to chain ordering, not model selection, so the
        rule path and the scoring path agree on the same model. Delegates to
        ``score_step`` with an empty provider (no health/credit tiebreakers).
        """
        return self.score_step(
            {"provider": "", "model": logical_model},
            task, bias=bias,
        )

    def _candidate_models(self, chain: list, blocked_models: set,
                          config: Optional[object] = None) -> set:
        """All logical models choosable from the (available, unblocked) chain.

        Enumerates the models on the chain steps plus each provider's declared
        ``models`` list, minus blocked models. Provider-agnostic — the same
        candidate set regardless of how the provider is eventually resolved.
        """
        cand: set[str] = set()
        try:
            providers = (config.providers or {}) if config is not None else {}
        except Exception:  # noqa: BLE001
            providers = {}
        for step in chain:
            logical = logical_model_name(step["model"], self.db_path)
            cand.add(logical)
            if providers:
                pcfg = providers.get(step["provider"]) or {}
                for m in pcfg.get("models") or []:
                    cand.add(logical_model_name(m, self.db_path))
        return cand - blocked_models

    def _choose_target_model(self, candidates: set, task: str, policy: str,
                             min_score: float, bias: Optional[float],
                             default_logical: str) -> Optional[str]:
        """Pick the target logical model from *candidates* (provider-agnostic).

        - ``eager`` / ``cost_first``: the highest-scoring model, unless it only
          beats the chain default within hysteresis (avoid flapping).
        - ``explore``: weighted random among models within hysteresis of the
          best (spread traffic / A/B over MODELS).
        Returns None when the best model is below ``min_score`` (static chain).
        """
        scored = sorted(
            ((self._score_model(m, task, bias), m) for m in candidates),
            key=lambda x: -x[0],
        )
        if not scored:
            return None
        best_score, best_model = scored[0]
        if best_score < min_score:
            return None
        if policy == "explore" and len(scored) > 1:
            import random
            cutoff = best_score - _HYSTERESIS
            pool = [m for s, m in scored if s >= cutoff]
            if len(pool) > 1:
                weights = [max(self._score_model(m, task, bias), 0.0) + 0.05
                           for m in pool]
                return random.choices(pool, weights=weights, k=1)[0]
        # eager / cost_first: don't flap away from the default within hysteresis.
        default_score = self._score_model(default_logical, task, bias)
        if best_model != default_logical and best_score <= default_score + _HYSTERESIS:
            return default_logical
        return best_model

    def _provider_health_rank(self, step: dict, profile: Optional[str] = None,
                              config: Optional[object] = None) -> int:
        """0=healthy, 1=degraded, 2=dead (dead is normally pre-filtered)."""
        try:
            from .circuit_breaker import get_circuit_breaker
            provider = step["provider"]
            base_url = step.get("base_url") or ""
            if config is not None and not base_url:
                base_url = (config.providers or {}).get(provider, {}).get("api_base", "")
            cb = get_circuit_breaker()
            status = cb.status_of(provider, base_url, profile or "")
            return {"healthy": 0, "degraded": 1, "dead": 2}.get(status, 0)
        except Exception:  # noqa: BLE001
            return 0

    def _provider_credit_rank(self, provider: str) -> int:
        """0=has credits, 1=low/zero credits (drained account).

        Used for CHAIN ORDERING so the router prefers a funded provider over a
        drained one when both serve the target model (a drained provider 400s
        with "insufficient credits" and wastes the attempt).

        Unlike ``_credit_bonus`` (scoring penalty) this is BALANCE-FIRST: an
        explicit available-credit balance overrides the ``monthly_pct >= 95``
        heuristic. That matters for opencode, which can be at 100% monthly yet
        still hold real dollars (e.g. $7.81 available) — it must rank as funded.
        """
        try:
            from .cost_cache import get_cost_cache
            cache = get_cost_cache()
            if cache is None:
                return 0
            # Strongest signal: explicit available balance → funded.
            bal = cache.get(provider, "balance")
            if bal:
                p = bal["payload"] or {}
                avail = p.get("available_credits")
                if avail is not None and avail > 1.0:
                    return 0
                b = p.get("balance")
                if isinstance(b, dict):
                    a = b.get("available")
                    if a is not None and a > 1.0:
                        return 0
            # Fall back to subscription heuristics (mirrors _credit_bonus).
            sub = cache.get(provider, "subscription")
            if sub:
                p = sub["payload"] or {}
                if p.get("_error"):
                    return 1
                if provider == "opencode":
                    mpct = p.get("monthly_pct")
                    if mpct is not None and mpct >= 95:
                        return 1
                elif provider == "commandcode":
                    rem = p.get("monthly_credits_remaining")
                    if rem is not None and rem <= 5.0:
                        return 1
            # Balance present but drained (<= $1) → drained.
            bal = cache.get(provider, "balance")
            if bal:
                p = bal["payload"] or {}
                b = p.get("balance")
                if isinstance(b, dict):
                    a = b.get("available")
                    if a is not None and a <= 1.0:
                        return 1
                avail = p.get("available_credits")
                if avail is not None and avail <= 1.0:
                    return 1
            return 0
        except Exception:  # noqa: BLE001 — ordering must never break routing
            return 0

    def _build_chain_for_model(self, chain: list, target_model: str,
                               preferred_provider: Optional[str] = None,
                               profile: Optional[str] = None,
                               config: Optional[object] = None) -> list:
        """Build the DETERMINISTIC chain for *target_model*.

        Walks the ORIGINAL chain in order, emitting one step per provider that
        serves the target (provider-model form via the registry), then the
        original non-target steps as fallbacks (tuple-deduped). Target
        providers are ordered: ``preferred_provider`` first, then healthy
        before degraded, then funded before drained (credits) — stable by chain
        order within each band.

        This is the SINGLE resolver shared by the prefer path and the scoring
        path — so both yield the same provider for the same target model.
        """
        target_steps: list[dict] = []
        seen_providers: set[str] = set()
        for step in chain:
            p = step["provider"]
            if p in seen_providers:
                continue
            logical = logical_model_name(step["model"], self.db_path)
            if logical == target_model or self._provider_serves_model(p, target_model, config):
                seen_providers.add(p)
                s = {"provider": p,
                     "model": provider_model_name(target_model, p, self.db_path)}
                if step.get("base_url"):
                    s["base_url"] = step["base_url"]
                target_steps.append(s)
        target_tuples = {(s["provider"], s["model"]) for s in target_steps}
        fallbacks = [dict(s) for s in chain
                     if (s["provider"], s["model"]) not in target_tuples]

        def _key(s):
            return (
                0 if (preferred_provider and s["provider"] == preferred_provider) else 1,
                self._provider_health_rank(s, profile, config),
                self._provider_credit_rank(s["provider"]),
            )
        target_steps.sort(key=_key)
        return target_steps + fallbacks

    # ── Routing rules (UI-defined overrides) ─────────────────────────────

    def _rules(self, config: Optional[object] = None,
               profile: Optional[str] = None) -> list:
        """Return the effective routing-rules list for a scope.

        A per-profile rules list (``routing_rules:<profile>``) replaces the
        global list when set; otherwise the global setting wins; otherwise
        ``config`` ``dynamic_routing.rules`` seeds defaults.
        """
        try:
            from .cost_cache import get_settings
            settings = get_settings()
            if settings is not None:
                try:
                    stored = settings.get_routing_rules(profile=profile)
                except TypeError:
                    stored = settings.get_routing_rules()
                if stored:
                    return stored
        except Exception:  # noqa: BLE001
            pass
        if config is not None:
            try:
                dr = config.dynamic_routing or {}
                return list(dr.get("rules", []) or [])
            except Exception:  # noqa: BLE001
                pass
        return []

    @staticmethod
    def _rule_matches(rule: dict, task: str, profile: str) -> bool:
        """A rule matches when its profile/task equal (or '*' / contains)."""
        if not rule.get("enabled", True):
            return False
        rp = rule.get("profile", "*")
        if rp not in ("*", "") and rp != profile:
            return False
        rt = rule.get("task", "*")
        if rt in ("*", ""):
            return True
        if isinstance(rt, list):
            return task in rt
        return rt == task

    def _rule_target(self, rule: dict, step: dict) -> bool:
        """True if the step matches a prefer/block rule's provider/model.

        ``"*"`` / ``""`` act as wildcards, so a rule can target ONLY a model
        (any provider) or ONLY a provider (any model). At least one concrete
        provider/model must be present.

        Model IDs are normalized through the registry (``logical_model_name``)
        so a rule written with the logical name (e.g. ``deepseek-v4-pro``)
        matches a chain step whose model is a provider-side ID (e.g.
        commandcode's ``deepseek/deepseek-v4-pro``). Provider names are
        compared literally (they are plain config names).
        """
        provider = rule.get("provider")
        model = rule.get("model")
        if provider and provider not in ("*", "") and provider != step["provider"]:
            return False
        if model and model not in ("*", ""):
            rule_model = logical_model_name(model, self.db_path)
            step_model = logical_model_name(step["model"], self.db_path)
            if rule_model != step_model:
                return False
        has_provider = bool(provider and provider not in ("*", ""))
        has_model = bool(model and model not in ("*", ""))
        return has_provider or has_model

    def _apply_blocks(self, chain: list, task: str, profile: str,
                      config: Optional[object] = None
                      ) -> tuple[list, list, set, set]:
        """Apply ``block`` rules to a COPY of the chain.

        Returns ``(candidates, fired, blocked_providers, blocked_models)``.
        The blocked-model set feeds ``_candidate_models`` so a model-only block
        removes the model from consideration everywhere, not just the chain
        steps that already carry it.
        """
        rules = self._rules(config, profile)
        if not rules:
            return list(chain), [], set(), set()
        candidates = [dict(step) for step in chain]
        fired: list[dict] = []
        blocked_providers: set[str] = set()
        blocked_models: set[str] = set()
        for rule in rules:
            if rule.get("action") != "block" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            before = len(candidates)
            candidates = [s for s in candidates if not self._rule_target(rule, s)]
            if len(candidates) < before:
                if rule.get("provider") and not rule.get("model"):
                    blocked_providers.add(rule["provider"])
                elif rule.get("model") and not rule.get("provider"):
                    blocked_models.add(logical_model_name(rule["model"], self.db_path))
                fired.append({
                    "action": "block",
                    "provider": rule.get("provider") or "*",
                    "model": rule.get("model") or "*",
                    "profile": rule.get("profile", "*"),
                    "task": rule.get("task", "*"),
                })
        return candidates, fired, blocked_providers, blocked_models

    def _resolve_prefer(self, chain: list, task: str, profile: str,
                        config: Optional[object] = None
                        ) -> tuple[Optional[str], Optional[str], list]:
        """Resolve ``prefer`` rules to ``(target_model, preferred_provider, fired)``.

        - A model prefer (provider wildcard or specific) that passes its
          ``min_score`` gate and has >= 1 chain provider serving it returns the
          preferred logical model as ``target_model`` — the chain is then built
          for it by ``_build_chain_for_model`` (the SAME resolver the scoring
          path uses, so the provider is deterministic).
        - A provider-only prefer is a PROVIDER TIEBREAK, not a model mandate:
          the model is still chosen by scoring, and the provider goes first in
          the built chain only if it serves the chosen model.
        - A prefer whose model no provider serves falls through to scoring.
        """
        rules = self._rules(config, profile)
        fired: list[dict] = []
        for rule in rules:
            if rule.get("action") != "prefer" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            rp = rule.get("provider") or "*"
            rm = rule.get("model") or "*"
            gate = rule.get("min_score")
            pref_logical = None if rm in ("*", "") else logical_model_name(rm, self.db_path)

            if pref_logical is not None:
                if gate is not None:
                    cap = self.get_model_score(pref_logical, task)
                    if cap < float(gate):
                        fired.append({
                            "action": "prefer_skipped_low_score",
                            "provider": rp, "model": rm,
                            "score": round(cap, 3), "min_score": float(gate),
                        })
                        continue
                serving = [
                    s["provider"] for s in chain
                    if (rp in ("*", "") or s["provider"] == rp)
                    and self._provider_serves_model(s["provider"], pref_logical, config)
                ]
                if serving:
                    fired.append({
                        "action": "prefer", "provider": rp, "model": rm,
                        "steps": len(set(serving)),
                    })
                    return pref_logical, (None if rp in ("*", "") else rp), fired
                # Preferred model served by no provider → fall through to scoring.
                fired.append({"action": "prefer_unserved", "provider": rp, "model": rm})
                continue

            # Provider-only prefer → provider tiebreak (model still scored).
            if not any(self._rule_target(rule, s) for s in chain):
                continue
            fired.append({"action": "prefer", "provider": rp, "model": rm,
                          "provider_only": True})
            return None, (None if rp in ("*", "") else rp), fired
        return None, None, fired

    def _apply_rules(self, chain: list, task: str, profile: str,
                     config: Optional[object] = None) -> tuple[list, list]:
        """Apply block/prefer rules to a COPY of the chain.

        Returns (candidates, fired_rule_descriptions). ``block`` removes
        matching steps; ``prefer`` moves the first matching step to the front
        (with an optional ``min_score`` gate). The global ``min_score`` floor
        still applies afterwards.
        """
        rules = self._rules(config, profile)
        if not rules:
            return list(chain), []
        candidates = [dict(step) for step in chain]
        fired: list[dict] = []
        blocked_providers: set[str] = set()

        # Pass 1 — blocks (also collect provider-wide blocks).
        for rule in rules:
            if rule.get("action") != "block" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            before = len(candidates)
            candidates = [s for s in candidates if not self._rule_target(rule, s)]
            if len(candidates) < before:
                if rule.get("provider") and not rule.get("model"):
                    blocked_providers.add(rule["provider"])
                fired.append({
                    "action": "block",
                    "provider": rule.get("provider") or "*",
                    "model": rule.get("model") or "*",
                    "profile": rule.get("profile", "*"),
                    "task": rule.get("task", "*"),
                })

        # Pass 2 — prefers (first-match wins). A prefer rule means "this task
        # must use the preferred model on EVERY provider that serves it, in the
        # chain's provider order, before any other model". We EXPAND the rule to
        # one step per unique provider (deduped, first-seen order) that serves
        # the preferred model — so a degraded provider falls to the NEXT provider
        # of the SAME model, not to a cheaper one. Provider-only prefers (no
        # model) keep the group-to-front behavior. Later prefers are ignored.
        for rule in rules:
            if rule.get("action") != "prefer" or not self._rule_matches(rule, task, profile):
                continue
            if not rule.get("provider") and not rule.get("model"):
                continue
            rp = rule.get("provider") or "*"
            rm = rule.get("model") or "*"
            gate = rule.get("min_score")

            pref_logical = None if rm in ("*", "") else logical_model_name(rm, self.db_path)

            if pref_logical is not None:
                # min_score gate on the preferred model's capability.
                if gate is not None:
                    cap = self.get_model_score(pref_logical, task)
                    if cap < float(gate):
                        fired.append({
                            "action": "prefer_skipped_low_score",
                            "provider": rp, "model": rm,
                            "score": round(cap, 3), "min_score": float(gate),
                        })
                        continue

                # One preferred step per unique provider (chain order) that
                # serves the preferred model.
                preferred: list[dict] = []
                pref_keys: set[tuple] = set()
                seen: set[str] = set()
                for s in candidates:
                    p = s["provider"]
                    if p in seen:
                        continue
                    seen.add(p)
                    if rp not in ("*", "") and p != rp:
                        continue
                    if not self._provider_serves_model(p, pref_logical, config):
                        continue
                    model_id = provider_model_name(pref_logical, p, self.db_path)
                    step = {"provider": p, "model": model_id}
                    if s.get("base_url"):
                        step["base_url"] = s["base_url"]
                    preferred.append(step)
                    pref_keys.add((p, model_id))

                if not preferred:
                    continue
                rest = [s for s in candidates
                        if (s["provider"], s["model"]) not in pref_keys]
                candidates = preferred + rest
                fired.append({
                    "action": "prefer",
                    "provider": rp, "model": rm,
                    "profile": rule.get("profile", "*"),
                    "task": rule.get("task", "*"),
                    "steps": len(preferred),
                })
                break

            # Provider-only prefer: group the provider's steps to the front.
            matches = [i for i, s in enumerate(candidates)
                       if self._rule_target(rule, s)]
            if not matches:
                continue
            matched = [candidates[i] for i in matches]
            match_set = set(matches)
            rest = [s for i, s in enumerate(candidates) if i not in match_set]
            candidates = matched + rest
            fired.append({
                "action": "prefer",
                "provider": rp, "model": rm,
                "profile": rule.get("profile", "*"),
                "task": rule.get("task", "*"),
                "steps": len(matched),
            })
            break

        return candidates, fired

    def select_step(self, messages: list[dict], tools: Optional[list[dict]] = None,
                    max_tokens: int = 1024, chain: Optional[list[dict]] = None,
                    profile: Optional[str] = None, config: Optional[object] = None,
                    ) -> Optional[list[dict]]:
        """Provider-aware selection: reorder a COPY of the chain so the best
        (provider, model) step is tried first — DETERMINISTICALLY.

        Routing ON follows the CIRCUIT BREAKER PROVIDER CHAIN order, but the
        MODEL is chosen by classification + scoring/rules:
          1. classify -> task
          2. drop dead/tripped providers (breaker hard gate)
          3. apply ``block`` rules
          4. choose the target LOGICAL MODEL: a fired ``prefer`` rule wins;
             otherwise score candidate models (provider-agnostic) under the
             policy (``eager``/``cost_first``/``explore``), ``min_score`` floor
          5. build the chain for that model ONCE via ``_build_chain_for_model``
             (preferred provider first, healthy before degraded, original order
             otherwise; non-target steps as fallbacks)

        Because the prefer path and the scoring path share that single
        chain-builder, they ALWAYS agree on the (provider, model) — the rule
        path can no longer pick commandcode/…flash while reorder picks
        deepseek/…flash for the same target model.

        Returns None only when routing is off / the chain is empty / the best
        model is below ``min_score`` (static chain) / the built head equals the
        original head (keep_default). The caller applies the returned copy.
        """
        # Attempt the profile-scoped enabled check + policy; tolerate
        # duck-typed/monkeypatched implementations that only accept ``config``.
        try:
            enabled = self.is_enabled(config, profile)
        except TypeError:
            enabled = self.is_enabled(config)
        if not enabled or not chain:
            return None
        try:
            policy, min_score = self._effective_policy(config, profile)
        except TypeError:
            policy, min_score = self._effective_policy(config)
        detail = classify_task_detail(messages, tools, max_tokens)
        task = detail.task

        # The circuit breaker only GATES PROVIDERS; the router owns model
        # selection and ordering. Drop steps whose provider is currently
        # unavailable (dead / hard-tripped) so the ordering never proposes a
        # provider the breaker would skip. (When routing is off, try_chain uses
        # the static chain + breaker as before.)
        chain = [s for s in chain if self._provider_available(s, profile, config)]
        if not chain:
            return None

        # policy rules override the policy for this scope.
        for rule in self._rules(config, profile):
            if rule.get("action") == "policy" and self._rule_matches(rule, task, profile or ""):
                p = rule.get("policy")
                if p in ("eager", "cost_first", "explore"):
                    policy = p

        # 3. blocks.
        chain, fired_blocks, blocked_providers, blocked_models = self._apply_blocks(
            chain, task, profile or "", config)
        if not chain:
            return None

        # 4. prefer -> target model / provider tiebreak.
        target_model, preferred_provider, fired_prefer = self._resolve_prefer(
            chain, task, profile or "", config)
        fired_desc = [f["action"] for f in fired_blocks] + [f["action"] for f in fired_prefer]

        # Audit: why didn't a rule fire? Surfaces silent rule misses in the
        # decision log (e.g. "rules exist for planning but task=agentic…").
        note = None
        rules = self._rules(config, profile)
        if rules:
            scope_matched = [r for r in rules
                             if self._rule_matches(r, task, profile or "")]
            if scope_matched and not fired_desc:
                note = (
                    f"{len(scope_matched)} rule(s) matched scope for task "
                    f"{task!r} but none fired"
                )
            elif scope_matched:
                note = (
                    f"{len(scope_matched)} rule(s) matched scope; fired: "
                    f"{', '.join(fired_desc) or 'none'}"
                )

        bias = min(self.cost_bias + 0.15, 0.5) if policy == "cost_first" else None
        prefer_fired = "prefer" in fired_desc

        # 4b. No prefer mandate -> score candidate MODELS (provider-agnostic).
        if target_model is None:
            default_logical = logical_model_name(chain[0]["model"], self.db_path)
            candidates = self._candidate_models(chain, blocked_models, config)
            target_model = self._choose_target_model(
                candidates, task, policy, min_score, bias, default_logical)
            if target_model is None:
                # Best model below min_score -> keep the static chain.
                head_score = self._score_model(default_logical, task, bias)
                self._record_decision({
                    **self._decision_base(detail, messages, profile or "", policy, note, fired_desc),
                    "action": "below_min_score", "model": chain[0]["model"],
                    "provider": chain[0]["provider"],
                    "score": round(head_score, 3),
                })
                return None

        # 5. resolve the provider ONCE (rule path == scoring path).
        result = self._build_chain_for_model(
            chain, target_model, preferred_provider, profile, config)
        if not result:
            return None
        head = result[0]
        head_score = self._score_model(target_model, task, bias)
        head_unchanged = (head["provider"] == chain[0]["provider"]
                          and head["model"] == chain[0]["model"])

        if prefer_fired:
            action = "prefer"
        elif policy == "explore" and head_unchanged:
            action = "keep_default"
        elif policy == "explore":
            action = "explore"
        elif head_unchanged:
            action = "keep_default"
        else:
            action = "reorder"

        if head_unchanged and not prefer_fired:
            self._record_decision({
                **self._decision_base(detail, messages, profile or "", policy, note, fired_desc),
                "action": "keep_default", "model": head["model"],
                "provider": head["provider"], "score": round(head_score, 3),
            })
            return None

        if action == "reorder":
            logger.info(
                "router_step_override",
                task=task, profile=profile or "",
                from_provider=chain[0]["provider"], from_model=chain[0]["model"],
                to_provider=head["provider"], to_model=head["model"],
                default_score=round(self._score_model(
                    logical_model_name(chain[0]["model"], self.db_path), task, bias), 3),
                recommended_score=round(head_score, 3),
            )
        self._record_decision({
            **self._decision_base(detail, messages, profile or "", policy, note, fired_desc),
            "action": action, "model": head["model"], "provider": head["provider"],
            "score": round(head_score, 3),
            "from_model": chain[0]["model"], "from_provider": chain[0]["provider"],
        })
        return result


# ── Global instances ──────────────────────────────────────────────────────────

_dynamic_router = CapabilityRouter(enabled=False)


def get_dynamic_router() -> CapabilityRouter:
    return _dynamic_router


def init_router(db_path: str = "data/costs.db", enabled: bool = False,
                cost_bias: float = DEFAULT_COST_BIAS):
    """Initialize the global router. Call once at startup."""
    global _dynamic_router
    _dynamic_router = CapabilityRouter(enabled=enabled, db_path=db_path,
                                       cost_bias=cost_bias)
    if enabled:
        _dynamic_router.load_matrix()  # warm the cache


def sync_router_enabled_from_settings() -> bool:
    """Re-apply the persisted ``routing_enabled`` toggle to the global router.

    Boot seeds ``enabled`` from ``gateway.yaml`` (which may be the ``false``
    baseline). Once the settings store is available (after ``init_settings``),
    this re-syncs so the effective state — and the boot log — reflects the UI
    toggle. Returns the effective enabled state.
    """
    global _dynamic_router
    try:
        from .cost_cache import get_settings
        settings = get_settings()
        if settings is not None:
            override = settings.get_routing_enabled(default=None)
            if override is not None:
                _dynamic_router.enabled = override
    except Exception:  # noqa: BLE001 — never fail boot
        pass
    return _dynamic_router.enabled


def invalidate_router_matrix() -> None:
    """Drop the global router's cached capability matrix.

    Called when a benchmark run completes or the registry changes so routing
    picks up fresh scores without a restart.
    """
    _dynamic_router.invalidate_matrix()


def _status_for_profile(router, config, profile: Optional[str]) -> dict:
    """Build one profile-scoped routing-status block."""
    policy, min_score = router._effective_policy(config, profile)
    return {
        "enabled": router.is_enabled(config, profile),
        "policy": policy,
        "min_score": min_score,
        "rules": router._rules(config, profile),
        # Per-profile has no matrix of its own; decisions are global (recent
        # decisions are not filtered by profile here — UI can filter).
        "has_override": bool(profile and router._has_profile_override(profile)),
    }


def routing_status(config: Optional[object] = None) -> dict:
    """Return a snapshot for the UI (Providers → Routing tab).

    Includes the GLOBAL enabled state, effective policy/min_score, the top
    recommended model per task (from the capability matrix, restricted to
    models that are actually selected — i.e. referenced by a profile chain),
    the active rules, and recent decisions. Also includes a ``per_profile``
    map so the UI can show each profile's effective routing state.
    """
    router = get_dynamic_router()

    # Models the router can actually pick = those named in any profile chain.
    selected: set[str] = set()
    try:
        profiles = (config or {}).profiles or {}
        for pcfg in profiles.values():
            for step in (pcfg.get("chain") or []):
                m = step.get("model")
                if not m:
                    continue
                selected.add(m)
                selected.add(normalize_model_id(m))
                try:
                    selected.add(benchmark_model_name(
                        logical_model_name(m, router.db_path), router.db_path))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        selected = set()

    profile_names: list = []
    providers: list = []
    try:
        if config is not None:
            profile_names = sorted((config.profiles or {}).keys())
            providers = sorted((config.providers or {}).keys())
    except Exception:  # noqa: BLE001
        pass
    per_task: dict[str, dict] = {}
    try:
        matrix = router.load_matrix()
        for task, scores in matrix.items():
            if not scores:
                continue
            # Only recommend models that are selected; fall back to the overall
            # top when no selection info is available (e.g. tests/no config).
            candidates = {k: v for k, v in scores.items()
                          if not selected or k in selected}
            if not candidates:
                continue
            top = max(candidates.items(), key=lambda kv: kv[1])
            per_task[task] = {"model": top[0], "score": round(top[1], 3)}
    except Exception:  # noqa: BLE001
        pass

    policy, min_score = router._effective_policy(config)
    per_profile = {p: _status_for_profile(router, config, p) for p in profile_names}
    return {
        "enabled": router.is_enabled(config),
        "policy": policy,
        "min_score": min_score,
        "rules": router._rules(config),
        "per_task": per_task,
        "recent_decisions": router.recent_decisions(25),
        # For rule-dropdowns in the UI: available profiles and providers.
        "profiles": profile_names,
        "providers": providers,
        "per_profile": per_profile,
    }
