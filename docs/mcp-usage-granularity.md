# MCP usage granularity — a first-class LCP feature

**Status:** design, not implemented. **Owner surface:** `src/api/conv_store.py` (ingestion),
`src/api/conv_store.py` + `src/server/handler.py` (read side), `alembic/` (backfill).

## The goal

Report per-tool usage — calls, bytes returned into the conversation, and the tokens/cost those rows
were part of — **inside LCP**, with no external dashboard, no per-report scripts, and no dependence on
any other service in the estate.

## Why it is currently invisible

LCP already captures every message, every tool row and every request. The blind spot is one field.

`_tool_name_lookup()` builds `tool_call_id → function.name` from the assistant turn and `_prepare()`
stamps that name onto the matching `role="tool"` row. That is correct for a direct tool call. But when
an agent reaches tools through a **deferred catalog**, the assistant turn's function name is the wrapper
(`tool_call`), and the real tool name — including every `mcp__*` name — is buried inside its serialised
`arguments`, under `calls[].name`. The result row is then stored as `tool_call`.

Consequence, measured on the live database: per-tool accounting works up to the switch-over date and
returns **zero** afterwards. A panel built on `tool_name LIKE 'mcp__%'` reports "no MCP usage" while
tens of thousands of MCP calls sit in the same table under a flattened name. This is the same failure
class the module docstring already records for the unnamed-row bug ("21,891 unnamed tool rows, still
arriving daily"): the name exists, it is just not where the reader looks.

## The change — at ingestion, once

Everything downstream then works with no further work: the Usage page, conversation detail, audit
views, threshold alerts, and any future panel read the same stored rows.

1. **`_tool_name_lookup()`**: when a resolved name is a known wrapper, unwrap it.
   - A wrapper is a tool whose `arguments` JSON carries a list at `calls[]`.
   - For each entry take `name`; when the wrapper passes several calls, the row belongs to the first,
     and the remainder are recorded as a comma-joined list so nothing is lost.
   - Guarded: any parse failure leaves the original name and increments a counter, never raises.
2. **`_prepare()`**: unchanged in shape — it already prefers a `tool_call_id` lookup. Only the lookup's
   output changes, which is why the fix stays small.
3. **Rollup**: `conversations.tool_call_count` already counts calls; add nothing. Per-tool counts come
   from a query over `conv_messages`, which is indexed by conversation and by `tool_name`.
4. **Backfill**: a one-off pass that rewrites `tool_name` for rows where it is a wrapper and the
   arguments parse. Idempotent, batched, and it records how many rows it could not parse — that number
   is part of the report, not a hidden loss.

## Read side — no new store, no new service

One read model, served by LCP itself:

```
GET /api/usage/tools?from=&to=&profile=&tool=
  { "tools": [
      { "tool": "mcp__<server>__<name>", "calls": N, "result_bytes": B,
        "avg_bytes": B/N, "max_bytes": M, "conversations": C }
    ],
    "unparsed_wrapper_rows": U }
```

- `calls` counts assistant turns whose `tool_calls[]` name the tool.
- `result_bytes` sums `content_len` on the matching tool rows — the bytes that actually entered the
  conversation.
- `unparsed_wrapper_rows` is reported rather than swallowed; a number that quietly shrinks is worse
  than a number that admits its gap.
- `from`/`to` use the same text-timestamp convention as the rest of the store.

The Usage page renders this as a table beside cost and tokens. No chart library is required for the
first version; the data is the feature.

## The counterfactual — the one thing LCP cannot see

LCP observes what **entered** the conversation. It cannot observe what a code-executing tool fetched
upstream and discarded, because that traffic never reaches LCP. A "how much did code mode save" figure
therefore needs one more input, and it must be **pushed to LCP**, not read from a sibling system:

```
POST /api/usage/tool-runs        (authenticated, same token model as the rest of the write API)
  { "profile": "...", "conversation_id": "...", "run_id": "...",
    "tool": "mcp__<server>__<name>", "inner_calls": N,
    "upstream_bytes": B, "returned_bytes": R, "sandbox_ms": T }
```

A server that compacts work into one call posts one row per run; LCP then reports, per conversation,
**avoided bytes (B − R) against incurred cost**, both measured where each actually happened. No
external dependency: the producer is whatever executed the code, and LCP owns the record.

## What "saved" honestly means

Token accounting must not be flattered. On the measured estate, **>95% of prompt tokens are cache hits**
and a small minority are cache misses; a tool result is re-sent on every later request in its
conversation, mostly at cache-read price. So a saving is:

1. the cache-miss on first insertion,
2. the re-sends at cache-read price,
3. context-window headroom and prefill latency.

Reporting "total prompt tokens avoided" overstates it by roughly the cache ratio and should not be a
headline number. Cache-miss tokens per conversation and requests per conversation are the honest
denominators, and both are already in `requests`.

## Deliberately out of scope

- No new service, no sidecar database, no log pipeline. This is a column, a backfill, one read endpoint
  and one page section.
- No dependency on an external dashboard, a scrape target or a remote runtime for the measurement
  feature; those can read the endpoint if they exist, but nothing here requires them.
- Code-mode execution itself is a separate concern (it needs a JavaScript isolate to run model-written
  code). If that host is ever required by LCP, it must be a **local** component that LCP reaches over
  HTTP like any other backend — never an external account or a hosted control plane.

## Test plan

- Unit: lookup unwrapping — direct call, single wrapped call, multi-call wrapper, malformed arguments,
  wrapper whose arguments are a string rather than an object (the shape that silently produced NULL
  names before).
- Unit: `_prepare` leaves a direct tool name untouched.
- Integration: ingest a batch containing a wrapped MCP call, read `/api/usage/tools`, assert one call
  and the summed bytes.
- Backfill: run twice, assert the second run changes zero rows and reports the same unparsed count.
- Regression: a row whose arguments do not parse must still be stored and counted under the wrapper
  name, never dropped.
