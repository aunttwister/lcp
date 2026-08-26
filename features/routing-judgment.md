# Routing Judgment — Replay & Human Review of Dynamic Routing Decisions

**Created:** 2026-08-26
**Status:** implemented (backend + script; UI follow-up pending)
**Depends on:** `features/routing-observability.md` (backend parts: `classify_task_detail` +
rationale columns + `_record_decision` wiring)

**Implemented 2026-08-26**: `_summarize_conversation` + `conversation_json`,
`_decision_base` (all six `_record_decision` sites), `routing_judgments` model,
migrations `020` + `021`, `scripts/judge_routing.py` (replay / review / pending /
report). Replay also shows recorded→replayed path drift. Tests:
`tests/test_routing_judgments.py` + router/semantic additions. Full suite
1602 passed.

---

## Problem

`routing_decisions` makes *what* the router chose auditable, but there is no way
to **judge whether it chose well**:

- The **input is not captured** — only the output (`task`, `provider`, `model`,
  `action`). To judge a decision you need to know what the classifier actually
  saw (the intent text / conversation shape), and it isn't stored.
- Decisions are **not re-checkable over time** — when router code changes (e.g.
  the upcoming deterministic-routing rewrite), there is no way to tell whether
  yesterday's decisions would be the same today (path drift).
- There is **no human judgment loop** — no way to mark a decision correct /
  wrong, so real-traffic mistakes never become a labeled dataset that could
  later seed regression tests or tuning.

Goal (per user, 2026-08-26): a **periodic, on-demand judgment workflow** on the
dev version — replay recent decisions with their rationale and reclassify them,
and review batches by hand, labeling them. The labeled set accumulates over time
into a real-traffic corpus.

## Current state (verified)

- `RoutingDecision` (src/api/models.py:350) columns: `ts, profile, task, policy,
  action, provider, model, score, rules_json, from_provider, from_model, note`.
  No input captured.
- `select_step(messages, tools, max_tokens, ...)` (router.py:1081) has `messages`
  in scope at all five `_record_decision` sites (prefer / below_min_score /
  explore / keep_default / reorder). Each site builds its dict inline.
- `_extract_intent_text(messages)` (router.py:292) returns `(text, meta)`;
  `classify_task` (router.py:333) is the entry point. `classify_task_detail`
  (planned in routing-observability.md) does not exist yet.
- DB path: `COST_DB` env var → seed `database.path` (`/app/data/costs.db`).
- `scripts/probe_intent.py` already boots `src` offline with a sys.path shim —
  the judge script reuses this pattern.
- `routing_decisions` is created by `create_all` (no migration); new columns /
  tables need an idempotent hand-written migration (pattern: `019`).

## Design

### 1. Capture the input: `conversation_json` on every decision

Add a `conversation_json` (Text, nullable) column to `RoutingDecision` holding a
**shape-preserving, content-trimmed** copy of the messages `select_step`
classified. Enough to *reproduce* the classification, without heavy payloads.

New helper `_summarize_conversation(messages, max_content=200, max_total=4000)`
in `src/api/router.py`:

- Per message keep: `role`; `content` **trimmed from the end** to
  `max_content` chars with a `… [N chars omitted]` marker — trimming from the
  end preserves leading tool-result markers (`[tool result]`, `[tool_output]`)
  that `_is_tool_result` / `_has_tool_result_blocks` rely on; keep the *start*
  of tool-result bodies (first ~120 chars) so pattern matches behave identically.
- Preserve `tool_calls` (id + name + trimmed args) and `tool_call_id` so the
  structural checks in `_extract_intent_text` behave identically.
- Cap total serialized size at `max_total`; when exceeded, drop **oldest**
  messages and prepend a `{"role": "system", "content": "[N older messages omitted]"}` 
  marker — the intent walk goes newest-first, so the tail is what matters.
  (Caveat: if the first user message is dropped and no genuine instruction is
  found, the first-message fallback can't reproduce — flagged as approximate.)

Round-trip guarantee (tested): `_extract_intent_text(desummarize(summarize(m)))
== _extract_intent_text(m)` for the known shapes (plain, tool-call, tool-result,
continuation). Semantic scores on replay are approximate if content was trimmed
heavily — surfaced in the report, not hidden.

### 2. Centralize decision recording: `_decision_base(...)`

The five `_record_decision` sites each repeat `ts/profile/task/policy/rules/
note`. To carry the rationale + conversation without five edits, add:

```python
def _decision_base(self, detail: ClassifyResult, messages, profile, policy,
                   note, fired_desc) -> dict:
    """Shared decision fields: ts, profile, task, policy, rules, note, plus the
    observability rationale (path, keyword, intent_text, semantic_json,
    min_score, sem_available) and the summarized conversation."""
    return {
        "ts": ..., "profile": profile or "", "task": detail.task,
        "policy": policy, "rules": fired_desc, "note": note,
        # rationale from routing-observability.md:
        "path": detail.path, "keyword": detail.keyword,
        "intent_text": detail.intent_text,
        "semantic_json": json.dumps(detail.semantic) if detail.semantic else None,
        "min_score": detail.min_score, "sem_available": detail.sem_available,
        # new for judgment:
        "conversation_json": json.dumps(_summarize_conversation(messages)),
    }
```

Each branch becomes `self._record_decision({**self._decision_base(...),
"action": "...", "model": ..., "provider": ..., "score": ...})`. This is stable
across the upcoming deterministic-routing rewrite — whatever logic runs, it
feeds the same base dict.

`select_step` computes `detail = classify_task_detail(messages, tools,
max_tokens)` once at the top and uses `detail.task` for routing.

### 3. Judgment table: `routing_judgments`

New model in `src/api/models.py`:

| column        | type    | notes                                  |
|---------------|---------|----------------------------------------|
| `id`          | Integer | PK autoincrement                        |
| `decision_id` | Integer | routing_decisions.id (loose ref)        |
| `profile`     | String  | denormalized for quick review queries   |
| `task`        | String  | denormalized                            |
| `path`        | String  | denormalized (which stage won)          |
| `verdict`     | String  | `correct | wrong | ambiguous`           |
| `expected_task`| String | nullable — what it *should* have been   |
| `note`        | Text    | nullable                                |
| `judged_at`   | String  | default now ISO                         |

Indexed on `decision_id` and `ts`-equivalent ordering (`id`). This is the
accumulating labeled real-traffic dataset.

### 4. Migration `alembic/versions/021_add_routing_judgment.py`

Idempotent (mirror `019`):
- `sa.inspect(conn)` guarded `ALTER TABLE routing_decisions ADD COLUMN conversation_json`.
- `sa.inspect(conn)` guarded `CREATE TABLE routing_judgments` (skipped when
  `create_all` already made it on a fresh DB).

### 5. Judge script `scripts/judge_routing.py`

Same sys.path bootstrap as `probe_intent.py`; resolves the DB with
`--db PATH` → `COST_DB` env → seed default.

- **`--replay [--limit N] [--since ISO] [--profile P]`** (default mode): pull
  recent `routing_decisions` (newest first), desummarize `conversation_json`,
  re-run `classify_task_detail` offline, print a table: `id, ts, profile,
  original task/path → current task/path, action/provider/model, intent_text
  (truncated), semantic top-N`, and a flag:
  - `SAME` — reclassification matches the recorded task.
  - `DRIFT` — the current code would classify differently (router changed since
    the decision) → investigate.
  - `NODATA` — no `conversation_json` (pre-migration rows) — show rationale-only.
  - Aggregates: task distribution, path distribution, DRIFT count, judged
    verdicts if present.
- **`--review [--limit N] [--filter profile|task|path|since]`**: interactive
  batch review of **unjudged** decisions. For each: print decision + rationale
  (task, path, keyword, intent_text, source/skips, semantic top-N, conversation
  role-shape) and prompt `[c] correct [w] wrong [a] ambiguous [s] skip [q] quit`;
  on wrong/ambiguous prompt for `expected_task` + optional note; write a
  `routing_judgments` row.
- **`--pending [--filter ...]`**: list unjudged decisions matching the filter
  (drives the review queue / queue size).
- **`--report [--out FILE.md]`**: aggregate the judged set — verdict
  distribution; per-task accuracy `correct/(correct+wrong)`; per-path accuracy;
  confusion matrix of `expected_task × actual task`; recent drift summary.
- `--json` for machine-readable output.

### 6. Optional UI (follow-up, NOT v1)

The observability plan's decision drill-down (Providers → Routing tab) is the
natural home for a **Judge** affordance (correct/wrong buttons writing to
`routing_judgments`) and an inline `--report` summary. Skipped for v1 because the
user chose an on-demand script.

## Implementation order

1. **routing-observability.md backend** (prerequisite): `ClassifyResult` +
   `classify_task_detail`, `SemanticClassifier.top_scores`, rationale columns +
   migration `020`, `_record_decision`/`recent_decisions` wiring.
2. **This feature**: `_summarize_conversation` + `conversation_json`,
   `_decision_base` refactor, `routing_judgments` + migration `021`,
   `scripts/judge_routing.py`.
3. Deterministic-routing rewrite (separate plan) — feeds the same `_decision_base`,
   no capture changes needed.

## Tests

- `tests/test_router.py`: `_summarize_conversation` (role/tool_calls/tool_call_id
  preserved; end-trimming keeps `[tool result]` markers; size cap drops oldest);
  round-trip `_extract_intent_text(summarize(m)) == _extract_intent_text(m)`;
  `_decision_base` includes rationale + conversation_json; persistence round-trip
  of `conversation_json` via `recent_decisions`.
- New `tests/test_routing_judgments.py`: `routing_judgments` create/write/read;
  migration `021` idempotency (pre-create table without new column / without the
  judgment table).
- Script smoke: `--replay` and `--report` against a tmp DB seeded with a few
  decisions (construct `RoutingDecision` rows directly, incl. `conversation_json`).

## Verification

1. `pytest -q` — full suite passes (existing 94 router tests unchanged).
2. Manual: run gateway with routing on → make a few requests → `python
   scripts/judge_routing.py --replay --limit 10` shows rationale + SAME/DRIFT →
   `--review` labels 2–3 → `--report` shows accuracy/confusion.
3. Confirm a pre-migration DB (rows without `conversation_json`) shows NODATA
   and doesn't crash.
4. `alembic upgrade head` against a copy of the prod DB — idempotent.
5. Push to `dev`; redeploy via `git pull` + `docker compose up -d --build`.

## Decisions

- Judge **offline** (script), not in the request path — zero overhead on live
  traffic; on-demand per user choice.
- Store **intent text + conversation shape** (not full messages): enough to
  reproduce classification; keeps rows small; avoids persisting tool-result
  bodies / secrets.
- Human verdicts accumulate in `routing_judgments` (new table), not columns on
  `routing_decisions` — one decision can be re-judged, and the table is a
  clean labeled dataset.
- Replay reclassification is *current-code* vs *recorded-decision* — that is
  exactly the drift signal wanted, not a bug.
- Semantic scores on replay are approximate (trimmed content) — surfaced, not
  hidden.

## Out of scope (follow-ups)

- **Golden-set auto-feed**: promote `verdict=correct` reviewed real-traffic cases
  into a labeled regression corpus (`features/routing-cases.json`) that
  `--golden` runs — closes the loop; probe_intent DEMO is the seed.
- UI judge buttons on the Routing tab.
- Scheduled cron/systemd report (user chose on-demand for now).
