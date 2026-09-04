# Merit Order — how LCP orders the fallback chain

**Created:** 2026-09-04
**Status:** implemented (chain-as-source-of-truth + credit-status abstraction +
credits-error fallthrough)
**Depends on:** `features/deterministic-routing.md` (unified rule/scoring
provider resolution), `features/routing-observability.md` (decision log).

---

## What "merit order" means here

The **merit order** is the ordered list of `{provider, model}` steps the gateway
tries for a request — the same concept as an electricity market's dispatch
order: the cheapest/most-preferred source is tried first, and you only move down
the list when a higher-ranked source fails. In LCP the chain is:

1. **built** by the dynamic router (`select_step` → `_build_chain_for_model`),
2. **executed** by `try_chain` in `request_pipeline.py` (circuit breaker,
   degraded gating, per-step retry, fallthrough on failure).

When dynamic routing is **off**, the merit order is simply the profile's static
chain, walked top-to-bottom. When routing is **on**, the router reorders a copy
of the chain so the best (provider, model) for the classified task is tried
first — deterministically.

---

## The full decision pipeline (`select_step`)

```
classify(messages, tools) ──► task (code_generation, debugging, planning, …)
        │
        ▼
1. drop dead/tripped providers        (circuit-breaker hard gate)
        │
        ▼
2. apply block rules                  (remove blocked providers/models)
        │
        ▼
3. resolve prefer rules               (→ target_model + preferred_provider)
        │
        ▼
4. no prefer mandate? ──► score candidate MODELS (provider-agnostic)
        │                     capability + cost_bias·(1−cost_factor)
        │                     policy: eager / cost_first / explore
        │                     min_score floor → static chain
        ▼
5. build the chain for the target model ONCE  (_build_chain_for_model)
        │
        ▼
6. try_chain executes the merit order       (breaker → degraded → retry → next)
```

Steps 3, 4, and 5 all funnel through the **same** chain-builder, so the rule
path and the scoring path can never disagree on the (provider, model) — the
original divergence (prefer picked `commandcode/…flash`, reorder picked
`deepseek/…flash`) is structurally impossible.

---

## Chain-as-source-of-truth (the core contract)

**The profile's chain is the single source of truth for which provider serves
which model.** A provider is a *target* provider for a model **only when its
chain step uses that model**. The provider's global `models` list in
`gateway.yaml` is **never** consulted when building the merit order.

Concretely, in `_build_chain_for_model`:

```python
for step in chain:
    logical = logical_model_name(step["model"], self.db_path)
    if logical == target_model:          # ← chain step model, NOT provider models list
        target_steps.append({provider, provider_model_name(target_model, provider)})
```

### Why this matters (the live L2 bug)

The L2 profile chain had a commandcode step using `z-ai/glm-5.3-flash`, but
commandcode's **global** `models` list also contained `deepseek-v4-flash`. The
old code treated commandcode as a deepseek-v4-flash provider (because of the
global list), so routing to `deepseek-v4-flash` produced a
`commandcode/deepseek/deepseek-v4-flash` head — which 400'd on
"insufficient credits" and, because that error was misclassified, **aborted the
whole chain** instead of falling through.

Under chain-as-source-of-truth:

- commandcode's chain step is `glm-5.3-flash` → commandcode is **never** a
  deepseek-v4-flash target.
- The flash merit order is built only from chain steps whose model IS flash
  (here: the deepseek step).
- commandcode (glm) and llamacpp (qwen) remain **fallbacks** — tried only after
  every flash link fails.

### Where the contract is enforced

| Site | Behavior |
|---|---|
| `_build_chain_for_model` | target steps = chain steps whose model == target |
| `_resolve_prefer` | "served" = a chain step uses the preferred model |
| `_apply_rules` (prefer) | expands only to chain steps whose model == preferred |
| `_candidate_models` | candidates = chain-step models only (no provider global list) |

The old `_provider_serves_model` helper (which consulted the global `models`
list) was removed — it was the source of the leak.

---

## Ordering the target steps (the merit-order sort)

Once the target steps are collected, they are sorted by a stable key:

```python
def _key(s):
    return (
        0 if (preferred_provider and s["provider"] == preferred_provider) else 1,  # 1. prefer
        self._provider_health_rank(s, profile, config),                            # 2. health
        self._provider_credit_rank(s["provider"]),                                 # 3. credits
    )
```

1. **Preferred provider first** — a `prefer` rule that names a provider (or a
   provider-only prefer whose provider serves the chosen model) leads.
2. **Healthy before degraded** — `0=healthy, 1=degraded, 2=dead` (dead is
   normally pre-filtered). A degraded provider is only reached when no healthy
   target provider remains.
3. **Funded before drained** — `0=has credits, 1=drained`. A provider whose
   cached balance/subscription says it's out of credits is pushed to the back,
   so a request doesn't waste an attempt on a provider that will 400.

Within each band the sort is **stable**, so the profile's original chain order
is preserved — the result is deterministic for the same inputs.

### Fallbacks are ordered too

The non-target original steps are appended after the target steps **and sorted
by the same key**. So when the target head fails, the router tries the best
fallback (e.g. a funded deepseek-v4-flash) before a drained provider — it does
not blindly walk the raw chain into a provider that will 400 on
"insufficient credits".

### Context gate

With `request_tokens > 0`, fallback steps whose model's context window can't
hold the request (+ an output reserve) are **dropped**, so a dead head never
falls through to a too-small-context model that 400s on `exceed_context_size`.

---

## Credit status — provider-agnostic via plugins

The router never hardcodes per-provider "drained" heuristics. Each cost plugin
owns its own payload shape and answers a single question:

```python
# CostPlugin base
def credit_status(self, subscription=None, balance=None) -> str:
    """'funded' | 'drained' | 'unknown' (unknown is treated as funded)."""
    return "unknown"
```

| Plugin | Drained when |
|---|---|
| `OpenCodeCostPlugin` | balance-first: explicit available balance > $1 → **funded** (even at 100% monthly); no balance + `monthly_pct >= 95` → drained; balance ≤ $1 → drained |
| `CommandCodeCostPlugin` | `monthly_credits_remaining + purchased_credits <= $5` (either can be the cause of "insufficient credits") |
| `DeepSeekCostPlugin` | `balance <= $1` |

The router's `_provider_credit_rank` and `_credit_bonus` both delegate to
`_provider_credit_status(provider)`, which resolves the plugin and asks it.
This keeps provider-specific knowledge in the plugins (where it belongs) and
the router provider-agnostic.

---

## Credits errors fall through (not abort)

A provider that 400s with an out-of-credits message must **fall through to the
next link** and trip the circuit breaker — it is not a bad request, and a
drained account won't self-heal.

`_is_credits_error` in `request_pipeline.py` matches these markers in the error
body (case-insensitive), regardless of HTTP status:

```
creditserror | insufficient balance | insufficient funds | insufficient credits
| out of credits | no credits | billing | payment required
```

`"insufficient credits"` was added for Command Code's exact wording. When
matched, the error becomes `ProviderCreditsError`, which `try_chain` handles by:

1. recording a circuit-breaker failure for the provider,
2. logging a `chain_fallback` event with the next provider,
3. **continuing to the next step** — the request is not lost.

Without this, the 400 fell into the generic `4xx → ProviderBadRequestError`
branch, which re-raises `AllProvidersFailedError` and aborts the whole chain.

---

## Worked example — the L2 profile

Profile chain (5 steps, as configured in the UI):

```
1. opencode / deepseek-v4-pro
2. deepseek / deepseek-v4-flash
3. commandcode / z-ai/glm-5.3-flash
4. llamacpp / qwen3.8-flash-next
5. opencode / ox-alpha-free
```

A request classified as `code_generation` scores `deepseek-v4-flash` as the
target model. The merit order is built:

1. **Target steps** — only chain steps whose model is flash: step 2 →
   `deepseek/deepseek-v4-flash`.
2. **Fallbacks** — steps 1, 3, 4, 5, ordered by health then credits:
   - opencode (funded, healthy) → `opencode/deepseek-v4-pro`
   - llamacpp (local, healthy) → `llamacpp/qwen3.8-flash-next`
   - commandcode (drained) → `commandcode/z-ai/glm-5.3-flash`
   - opencode → `opencode/ox-alpha-free`

Resulting merit order:

```
1. deepseek / deepseek-v4-flash        ← target head
2. opencode / deepseek-v4-pro          ← funded fallback
3. llamacpp / qwen3.8-flash-next       ← healthy fallback
4. commandcode / z-ai/glm-5.3-flash    ← drained fallback (last)
5. opencode / ox-alpha-free
```

**commandcode is never tried for deepseek-v4-flash** — its chain step is glm.
If the deepseek head 400s on credits, the request falls through to opencode
(not commandcode), and commandcode is only reached if everything else fails.

---

## Reading the decision log

Each routed request writes a `routing_decisions` row (and a `chain_attempt` /
`chain_fallback` / `chain_bad_request` log line). Key fields:

| Field | Meaning |
|---|---|
| `task` | classified task |
| `action` | `prefer` / `reorder` / `keep_default` / `below_min_score` |
| `rules` | fired rule actions, e.g. `["prefer"]`, `["prefer_unserved"]` |
| `provider` / `model` | the resolved head of the merit order |
| `score` | capability score of the chosen model |
| `note` | audit note when rules matched scope but none fired |

`prefer_unserved` appears when a prefer rule names a model that **no chain step
uses** — the router falls through to scoring instead of force-picking a
provider.

---

## Configuration reference

- `dynamic_routing.enabled` (global) / `routing_enabled:<profile>` (per-profile)
  — master switch.
- `dynamic_routing.cost_bias` — scoring weight on cost.
- Per-profile `chain` — the source of truth for provider→model availability.
- `providers.<name>.models` — **only** used for the UI dropdown and provider
  discovery; **not** consulted for merit-order construction.
- Routing rules (Providers → Routing): `prefer` / `block` / `policy` with
  optional `min_score` gates.

---

## Tests

- `tests/test_router.py`:
  - `test_build_chain_for_model_chain_step_is_source_of_truth` — commandcode's
    global models include flash but its chain step is glm → not a flash target.
  - `test_build_chain_for_model_deterministic_order` — only chain steps whose
    model is the target become target heads.
  - `test_flash_prefer_and_scoring_agree` — prefer and scoring resolve the same
    provider (deepseek) for flash.
  - `test_apply_rules_prefer_expands_across_providers` — prefer expands only to
    chain steps that use the preferred model.
  - `test_provider_only_prefer_is_tiebreak_not_mandate` — provider-only prefer
    leads only when the provider's chain step uses the scored model.
  - `test_build_chain_for_model_orders_funded_provider_first` /
    `test_build_chain_for_model_sorts_fallbacks_by_credit` — credit ordering.
- `tests/test_cost_plugins_{opencode,commandcode,deepseek}.py` — `credit_status`
  per plugin (balance-first funded, monthly+purchased drained, balance drained,
  unknown).
- `tests/test_request_pipeline_helpers.py` —
  `test_commandcode_insufficient_credits_raises_credits_error` — "insufficient
  credits" → `ProviderCreditsError` (falls through, not abort).

Full suite: **1760 passed, 15 deselected**.