# Deterministic Routing — unify rule & scoring provider selection

**Created:** 2026-08-25 (plan, Pro-validated)
**Status:** implemented 2026-08-26 (commit after `845d31b`)
**Depends on:** `features/routing-observability.md` backend (rationale capture) +
`features/routing-judgment.md` (replay) — used to verify the fix on hermes-bridge.

---

## Problem (verified live on hermes-bridge replay, 2026-08-26)

For the SAME profile (`coder`), the same target model resolved to DIFFERENT
providers depending on which path decided it:

| row | task            | path     | result                        |
|-----|-----------------|----------|-------------------------------|
| 17:30 | `code_generation` | `prefer` | `commandcode/deepseek/deepseek-v4-flash` |
| 17:32 | `debugging`      | `reorder` | `deepseek/deepseek-v4-flash` |

Root cause: **two code paths resolve one decision differently**.
- `_apply_rules` model-prefer **expands the model to one step per provider that
  serves it, in chain order** → first chain provider = `commandcode` → prefer
  pins `commandcode/…flash`.
- `score_step` scores **literal chain steps** `{provider, model}` → the only
  flash step in the chain is `deepseek/deepseek-v4-flash` → reorder picks it.

User mandate: "WE CAN'T HAVE ONE PROVIDER SELECTED FOR RULE ANOTHER SELECTED
FOR REORDER. PLEASE REVISE. PLAN THIS CAREFULLY."

## Design: score the model, resolve the provider once

When routing is ON:
1. `classify` → `task` (unchanged).
2. Drop dead/tripped providers (breaker hard gate, unchanged).
3. Apply `block` rules (`_apply_blocks`).
4. Decide the **target logical model** (provider-agnostic):
   - a `prefer` rule fires (task/profile + `min_score` gate + ≥1 provider
     serves it) → target = rule's model;
   - otherwise score **candidate models**: `capability + cost_bias×(1−cost_factor)`
     (`_score_model`, NO health/credit), policy (`eager`/`cost_first`/`explore`)
     picks (`_choose_target_model`), `min_score` floor → `None` (static chain).
5. Build the chain for that model ONCE via `_build_chain_for_model`:
   walk the original chain in order, emit `{provider, provider_model_name(target,
   provider)}` per serving provider; preferred provider first, then healthy
   before degraded (chain order within each band); original non-target steps
   kept as fallbacks.

Both paths now funnel through the same chain-builder → identical (provider,
model). For the coder chain with `target = deepseek-v4-flash`:

```
[commandcode/deepseek/deepseek-v4-flash,   # link1 serves flash (registry)
 deepseek/deepseek-v4-flash,               # link2
 opencode/deepseek/deepseek-v4-flash]      # link3 serves flash
+ [commandcode/deepseek/deepseek-v4-pro,    # fallbacks (original non-target)
   opencode/deepseek-v4-pro, opencode/ox-alpha-free]
```

## Rule semantics (minimal change)

- **`block`** — unchanged (applied first; also yields `blocked_models` for
  candidate enumeration).
- **`policy`** — unchanged (overrides before model selection).
- **provider-only `prefer`** — a **provider tiebreak**, NOT a model mandate:
  the model is still chosen by scoring; the provider leads *if* it serves the
  chosen model. (Forcing a model here would re-create the divergence.)
- Model-prefer whose model **no provider serves** → falls through to scoring,
  with `prefer_unserved` in the audit trail.

## Circuit breaker interaction

- Dead/tripped → hard gate (unchanged).
- Degraded → kept, but **healthy before degraded** among target-model providers;
  all-degraded → chain order.
- `_health_bonus`/`_credit_bonus` moved OUT of model selection into chain
  ordering (`_provider_health_rank`); `score_step` retained for compatibility.

## Functions (src/api/router.py)

New: `_score_model`, `_candidate_models`, `_choose_target_model`,
`_provider_health_rank`, `_build_chain_for_model`, `_apply_blocks`,
`_resolve_prefer`.
`select_step` rewritten: signature / `classify_task` / public API unchanged.
`keep_default` stays a `return None` (head unchanged) to preserve caller
semantics (request_pipeline treats None as "keep original chain"); it is now a
LABEL too (recorded action). `_apply_rules` retained as a compatibility layer
(legacy chain transform) — `select_step` no longer calls it.

## Tests (tests/test_router.py)

- **HEADLINE** `test_flash_prefer_and_scoring_agree` — prefer path and scoring
  path both resolve `commandcode/deepseek/deepseek-v4-flash` for the same chain.
- `test_degraded_target_provider_falls_to_healthy_same_model` — healthy
  providers lead a degraded one for the same target model.
- `test_provider_only_prefer_is_tiebreak_not_mandate` — provider-only prefer
  doesn't force a model; action stays `prefer`.
- `test_prefer_model_served_by_no_provider_falls_through` — unserved prefer
  falls to scoring with `prefer_unserved` audit.
- `test_build_chain_for_model_deterministic_order` — shared builder emits
  target steps (chain order) then non-target fallbacks.
- Existing `_apply_rules`/`score_step`/selection tests updated where needed.

Full suite: **1610 passed**.

## Verification (deploy)

1. `pytest -q` — 1610 passed.
2. Push to `dev`; hermes-bridge: `git pull` + `docker compose up -d --build`.
3. `docker compose exec lcp python3 scripts/judge_routing.py replay --limit 50`:
   after the attachments fix (separate commit `845d31b`) the `intent` column
   shows real instructions and spurious DRIFT/NOINTENT collapse.
4. Confirm code_generation (prefer flash) and debugging (scoring → flash) now
   both route to `commandcode/deepseek/deepseek-v4-flash` — same provider.
5. `judge_routing.py review` + `report` to build the labeled dataset.

## Follow-ups (not in this change)

- `features/routing-observability.md`: probe endpoint + UI drill-down (pending).
- Editable per-task exemplars / feedback loop on `routing_judgments`.
