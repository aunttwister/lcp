# Routing Decision Observability & Classifier Debugger

**Created:** 2026-08-25
**Status:** backend implemented (2026-08-26); probe API + UI pending
**Last updated:** 2026-08-26

**Implemented 2026-08-26**: `ClassifyResult` + `classify_task_detail` (with
`path`/`keyword`/`intent_text`/`semantic` rationale),
`SemanticClassifier.top_scores` + public `min_score`, rationale columns on
`routing_decisions` (migration `020`), wired through `_record_decision` /
`recent_decisions`. Still pending from this plan: `POST /api/routing/probe`,
the UI decision drill-down + probe card, and the stale semantic-classifier
cache fix. (The judgment feature `features/routing-judgment.md` builds on this
backend.)

---

## Problem

Every routing decision is a black box. `classify_task` returns a bare task
string; the persisted `routing_decisions` table records *what* was chosen but
not *why*:

- **No path** — did the keyword scanner fire, the semantic (embedding)
  classifier, the agentic system prompt, tool-count, or the default?
- **No matched keyword** — which signal actually matched (`debug`? `error`?
  `test suite`?).
- **No intent text** — what message was actually classified (the "newest
  genuine user instruction" after the `2b29505` extraction fix).
- **No semantic scores** — the per-task cosine scores and the `min_score` gate
  are computed inside `SemanticClassifier` and thrown away; only the winner (or
  `None`) comes back.
- **No intent provenance** — how many tool echoes / continuations were skipped
  to find the intent message.

Consequence: the recurring "why is this routing to unit_tests/debugging?"
questions required forensics across logs and code. The observability feature
makes every decision explainable at a glance, and adds a live probe tool so a
prompt can be tested without sending a real request.

## Current state (verified)

- `classify_task` (src/api/router.py) extracts intent via `_extract_intent_text`
  (already returns `(text, meta)` where meta = `{source, skipped_tool,
  skipped_cont}` — commit `2b29505`), then keyword → semantic → agentic →
  tool-count → token-count → casual → default.
- `SemanticClassifier.classify` (src/api/task_classifier.py) computes per-task
  cosine scores but returns only the winner; `get_semantic_classifier()` reads
  `plugins.router` (enabled / min_score / embedding.model / device).
- `select_step` records decisions via `_record_decision` →
  `routing_decisions` row (ts, profile, task, policy, action, provider, model,
  score, rules_json, from_provider, from_model, note). `recent_decisions` reads
  DB first.
- `routing_decisions` is auto-created by `Base.metadata.create_all` (no Alembic
  migration); `create_all` never alters an existing table → new columns need an
  idempotent hand-written migration (pattern: `019_add_provider_health.py`).
- Routing tab (src/ui/templates/jinja/pages/providers.html,
  `renderRoutingDecisions`) renders the decision table but drops `note` and has
  no drill-down.
- `classify_task` has 27 call sites → keep its `-> str` signature; add a detail
  variant alongside.

## Design

### 1. Classification rationale (backend)

Add a `ClassifyResult` dataclass in `src/api/router.py`:

```
task          # final task string
path          # which stage won: keyword:<task> | semantic | agentic_prompt |
              #   tool_count | token_count | casual | default
keyword       # the exact TASK_SIGNALS keyword that matched (or None)
intent_text   # the extracted "newest genuine user instruction"
intent_meta   # {source, skipped_tool, skipped_cont} from _extract_intent_text
semantic      # top-N (task, score) from the embedder, or None when unavailable
min_score     # the semantic gate that was applied (0.35 default)
sem_available # whether the embedder was up
tool_count    # len(tools)
token_count   # count_tokens(messages, tools)
```

- Refactor `classify_task` internals into
  `classify_task_detail(messages, tools, max_tokens) -> ClassifyResult`.
  `classify_task(...) -> str` becomes a thin wrapper (`detail.task`) so all 27
  call sites are unchanged.
- `task_classifier.py`: add `SemanticClassifier.top_scores(text, k) ->
  list[(task, float)]` (sorted desc) and make `classify()` delegate to it; this
  exposes the scores for both the decision rationale and the probe.
- Populate `path` strings at each branch.

### 2. Persist the rationale

Extend `RoutingDecision` (src/api/models.py) with:

| column          | type      | notes                                  |
|-----------------|-----------|----------------------------------------|
| `path`          | String    | which stage won                         |
| `keyword`       | String    | matched keyword (nullable)              |
| `intent_text`   | Text      | truncated to ~500 chars                 |
| `semantic_json` | Text      | top-5 (task, score) JSON (nullable)     |
| `min_score`     | Float     | semantic gate applied (nullable)        |
| `sem_available` | Boolean   | embedder up (nullable)                  |

- New idempotent Alembic migration `alembic/versions/020_add_routing_signal_columns.py`
  (per-column `ADD COLUMN` guarded by `sa.inspect`, mirroring `019`; safe when
  the table was already created with the new columns by `create_all` on a fresh DB).
- `_record_decision` writes the new fields; `select_step` passes the
  `ClassifyResult` into the decision dict; `recent_decisions` returns them.

### 3. Probe API

- `POST /api/routing/probe` accepting `{text}` (single prompt → one-turn
  conversation) **or** `{messages: [...]}` (full conversation — reproduces
  real multi-turn misrouting, including tool-result-as-user shapes). Returns
  the full `ClassifyResult` dict (+ `intent_meta`).
- Also fix the redundant `routing_status(self.config)` calls in the profile
  branch of `_serve_routing_status_api` (currently invoked 4× per request).

### 4. UI (Providers → Routing tab)

- **Decision table**: render the `note` inline; make rows expandable → detail
  panel showing path, matched keyword, intent text (truncated + full), semantic
  top-5 scores with mini bars, min_score, intent source/skips.
- **Classifier probe card** at the top of the tab: textarea + "Classify"
  button → `POST /api/routing/probe` → task badge + path + per-signal
  breakdown. Pre-populated from the last clicked decision for easy iteration.
- Keep `escC`/`fmtTime` self-contained in `providers.html` (gotcha `b6e1888`).

### 5. Tests & docs

- `tests/test_router.py`: `classify_task_detail` path coverage (keyword /
  semantic / agentic / tool_count / token_count / casual / default);
  `classify_task` still returns a plain string; persistence round-trip of the
  new fields (extend the existing DB round-trip test).
- `tests/test_task_classifier_semantic.py`: `top_scores` ordering; `classify`
  uses it.
- New probe-endpoint tests (text / messages-array / empty validation) in
  `tests/test_settings_endpoints.py` or a new file.
- Migration test: pre-create `routing_decisions` without the new columns →
  `020` adds them idempotently.
- `features/routing-observability.md` = this doc.

## Also in scope (small fixes surfaced by this work)

1. **Stale semantic-classifier cache**: `get_semantic_classifier()` caches
   `_classifier` once; a runtime change to `plugins.router.min_score` / model
   won't apply until restart or setup install/uninstall. Add
   `invalidate_semantic_classifier()` on config save for `plugins.router`, or
   cache-key on the config values (min_score + model name + enabled) so edits
   apply immediately. This makes the min_score knob actually live — and the
   probe/detail will surface the *effective* value.
2. Expose semantic availability + effective min_score in the probe/detail (so
   "the embedder is down / threshold is 0.35" is visible, not assumed).

## Out of scope (follow-ups)

- Editable per-task exemplars in the UI (semantic-router style tuning).
- Session stickiness (per-message classification already re-derives intent).
- Agentic sub-classification (planning vs execute vs general).
- Feedback/learning loop on decisions.

## Verification

1. `pytest -q` — full suite passes (existing 94 router tests unchanged).
2. `pytest tests/test_router.py tests/test_task_classifier_semantic.py tests/test_settings_endpoints.py`.
3. Manual: start gateway → Providers → Routing → paste a real coder prompt in
   the probe → confirm task/path/keyword/semantic scores/intent text.
4. Confirm a live request's decision row shows the rationale + note; confirm a
   decision that came from the semantic path shows the per-task scores.
5. `alembic upgrade head` against a copy of the prod DB (hermes-bridge) —
   idempotent.
6. Push to `dev`; redeploy via `git pull` + `docker compose up -d --build`.

## Decisions

- Keep `classify_task -> str` for backward compat; rationale via
  `classify_task_detail`.
- Rationale embedded in the existing `routing_decisions` rows (no new table).
- New columns via hand-written idempotent Alembic `020` (`create_all` can't
  alter existing tables).
- Probe accepts `text` OR `messages[]` for exact-request reproduction.
- Semantic scores capped at top-5 in persisted rows / UI.
