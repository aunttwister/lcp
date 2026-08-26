#!/usr/bin/env python3
"""Probe intent classification for the LCP dynamic router.

Tests how different user intents are classified — and what text is actually
used — without starting the gateway. For each case it prints:
  - the extracted intent text (the "newest genuine user instruction")
  - how it was chosen (source + how many tool/continuation messages were skipped)
  - the classified task
  - the top semantic (embedding) scores, when the embedder is available

Examples
--------
    # One-off: each argument becomes a one-turn conversation (system + user)
    .venv/bin/python scripts/probe_intent.py "can we plan the next feature"
    .venv/bin/python scripts/probe_intent.py \\
        "fix this error" "write pytest tests" "hello there"

    # A full conversation from a JSON file (multi-turn, tool echoes, etc.)
    .venv/bin/python scripts/probe_intent.py --messages /tmp/msgs.json

    # The built-in demo cases (planning, debugging, tests, tool echoes, ...)
    .venv/bin/python scripts/probe_intent.py --all

JSON messages format (list of OpenAI-style messages):
    [
      {"role": "system", "content": "You are a coding agent. tools: terminal"},
      {"role": "user", "content": "can we plan the next feature?"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
      {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed"}
    ]
"""

import json
import os
import sys

# Make `src` importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.router import classify_task, _extract_intent_text  # noqa: E402

AGENT = {"role": "system", "content": "You are a coding agent. tools: terminal, patch"}

DEMO = {
    "planning": [AGENT, {"role": "user", "content": "can we plan the next feature in features folder?"}],
    "debugging": [AGENT, {"role": "user", "content": "why does this endpoint return a 500? debug the traceback"}],
    "unit_tests": [AGENT, {"role": "user", "content": "write a pytest suite for the new module with mocking"}],
    "code_generation": [AGENT, {"role": "user", "content": "review this file and make the changes we discussed"}],
    "reasoning_chain": [AGENT, {"role": "user", "content": "calculate the time complexity of this recurrence relation"}],
    "research_deep": [AGENT, {"role": "user", "content": "research vector databases and summarize in detail"}],
    "casual": [AGENT, {"role": "user", "content": "hello, how are you?"}],
    "sticky-debugging regression (tool echo + new plan)": [
        AGENT,
        {"role": "user", "content": "fix this error in the code"},
        {"role": "assistant", "content": "Let me reproduce."},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed. test_plan.py: FAILED"},
        {"role": "assistant", "content": "I see the bug."},
        {"role": "user", "content": "can we plan the next feature in features folder?"},
    ],
    "tool-result-as-user (old unit_tests hijack)": [
        AGENT,
        {"role": "user", "content": "can we plan the next feature in features folder?"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "[tool result] Ran 12 tests, 3 failed. test_plan.py: FAILED"},
    ],
    "continuation tail keeps intent": [
        AGENT,
        {"role": "user", "content": "why does this endpoint return a 500? debug the traceback"},
        {"role": "assistant", "content": "running..."},
        {"role": "user", "content": "continue"},
    ],
}


def semantic_top(text: str, k: int = 3):
    """Return the top-k per-task cosine scores for *text*, or None if the
    embedder isn't available."""
    if not text:
        return None
    try:
        from src.api.task_classifier import _cosine, get_semantic_classifier
    except Exception:
        return None
    try:
        clf = get_semantic_classifier()
        if clf is None:
            return None
        clf._build_centroids()
        vec = clf._embed([text])[0]
        scores = sorted(
            ((t, round(_cosine(vec, c), 3)) for t, c in (clf._centroids or {}).items()),
            key=lambda x: -x[1],
        )
        return scores[:k]
    except Exception:
        return None


def probe(messages: list, label: str = "case") -> None:
    intent, meta = _extract_intent_text(messages)
    task = classify_task(messages)
    print(f"\n=== {label} ===")
    print(f"  messages: {len(messages)}")
    print(f"  intent_text : {intent[:140]!r}")
    print(f"  source      : {meta['source']}  "
          f"(skipped tool={meta['skipped_tool']}, continuation={meta['skipped_cont']})")
    print(f"  classify    : {task}")
    scores = semantic_top(intent)
    if scores:
        print("  semantic top: " + ", ".join(f"{t}={s}" for t, s in scores))
    else:
        print("  semantic    : embedder not available")


def main(argv) -> int:
    if "--all" in argv:
        for label, msgs in DEMO.items():
            probe(msgs, label)
        return 0

    if "--messages" in argv:
        idx = argv.index("--messages")
        path = argv[idx + 1]
        with open(path) as f:
            probe(json.load(f), f"from {path}")
        return 0

    texts = [a for a in argv if not a.startswith("--")]
    if texts:
        for text in texts:
            probe([AGENT, {"role": "user", "content": text}], text)
        return 0

    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
