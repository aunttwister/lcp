# Intelligent Routing — Dynamic Router, Council, and Harness

**Created:** 2026-08-10
**Status:** draft (pre-implementation research complete)

---

## Overview

LCP currently routes requests statically: a profile has a hardcoded provider chain,
and the `DynamicRouter` (`src/api/router.py`) only does binary flash-vs-pro selection
based on raw token/tool counts. This document proposes a three-tier vision:

| Tier | What | When |
|------|------|------|
| **1. Capability-matrix router** | Prompt classification → model scoring → best-fit routing | Now |
| **2. LLM Council** | Parallel generation → debate/aggregation for higher quality | Later |
| **3. True Harness** | Self-improving agent state with trajectory-backed refinements | Far future |

---

## Tier 1: Capability-Matrix Dynamic Router

### Current State (what we have)

`src/api/router.py` — `DynamicRouter` with 3 hardcoded thresholds:
- `tokens > 2000` → pro
- `tools > 3` → pro
- `max_tokens > 2048` → pro

Hardcoded to exactly two models (`deepseek-v4-pro` / `deepseek-v4-flash`).
Disabled by default (`self.enabled = False`). Only consulted in `src/ui/dashboard.py`
— **not wired into the request pipeline**.

### What Open Source Projects Do

Our research covered 7 projects. The most relevant to LCP's needs are:

**RouteLLM (LMSYS)** — Pairwise routing via learned model embeddings:
- Each model has a 128-dim embedding vector
- Each prompt gets embedded via `text-embedding-3-small`
- Compatibility score = `classifier(model_embed ⊙ prompt_embed)`
- Only supports 2 models at a time; no explicit task classification

**IRT-Router (NotDiamond, ACL 2025)** — Multi-dimensional capability model:
- Learns latent "knowledge dimensions" via Multidimensional Item Response Theory (MIRT)
- Each LLM → capability vector; each query → requirement vector
- Dot product predicts performance — supports N models
- Requires per-model performance data for training (not available to LCP)

**OpenRouter Auto Beta** — Explicit task classification + market signals:
- Lightweight classifier assigns prompt to ~30 task types
- Ranks models by community spend-share per task (trailing 7 days)
- Cost-tier filtering + session stickiness
- **Proprietary** but the architecture is well-documented

**RoRF (NotDiamond)** — Random Forest on prompt embeddings:
- Replaces RouteLLM's MF with a RandomForestClassifier
- Embeds prompt, classifies into 4-label (model_a_wins/tie/model_b_wins)
- Works on CPU, fast to train, but still pairwise only

### Proposed Architecture

LCP will take a **hybrid approach**: explicit task classification (like OpenRouter)
combined with a configurable capability matrix (like IRT-Router, but human-curated
rather than trained).

```
Request arrives
  │
  ├─ 1. CLASSIFY: Keyword + heuristic classifier → task type
  │     (no ML dependency, zero latency, transparent rules)
  │
  ├─ 2. SCORE: For each available model, look up task→model fit score
  │     from CAPABILITY_MATRIX (config-driven, model-independent)
  │
  ├─ 3. BOOST: Add cost-awareness bonus for cheaper models
  │     (so flash wins on tasks where it's "good enough")
  │
  ├─ 4. FILTER: Remove models that can't serve (context window, modality,
  │     circuit breaker dead, not in provider chain)
  │
  └─ 5. RANK: Pick top model, with fallback chain = profile chain
```

### Task Classification (Step 1)

No ML. A layered keyword/heuristic classifier that examines:
1. **System prompt patterns** — `"You are a coding agent"` vs `"You are a triage router"`
2. **Tool counts and types** — many tools → agentic, `execute_code` → coding, `web_search` → research
3. **Message content signals** — debug keywords, math notation, natural language patterns
4. **Request metadata** — `max_tokens` expectation, streaming flag

```python
# Task types (extensible, config-driven)
TASK_TYPES = [
    "agentic_multi_step",    # Many tools, long system prompt, complex instructions
    "code_generation",       # "write a function", "implement", "create a script"
    "debugging",             # Error messages, stack traces, "why does this fail"
    "research_deep",         # "explain", "analyze", "compare", web_search tool
    "long_document",         # >8K input tokens, "summarize", document Q&A
    "reasoning_chain",       # Step-by-step logic, math, "prove", "solve"
    "casual_chat",           # Short greetings, small talk, simple Q&A
    "planning",              # "design", "architecture", "how should I structure"
]
```

### Capability Matrix (Step 2)

**Model-independent by design.** Models are discovered from config (`providers.*.models`)
and the matrix maps `task_type → {model_name: fitness_score}`.

```yaml
# config/gateway.yaml (new section)
dynamic_routing:
  enabled: true
  # Optional: override the default cost multiplier for flash-bias
  cost_bias_strength: 0.15

  capability_matrix:
    agentic_multi_step:
      deepseek-v4-pro: 1.0
      deepseek-v4-flash: 0.4
    code_generation:
      deepseek-v4-pro: 1.0
      deepseek-v4-flash: 0.7
    debugging:
      deepseek-v4-pro: 0.9
      deepseek-v4-flash: 0.8
    research_deep:
      deepseek-v4-pro: 1.0
      deepseek-v4-flash: 0.5
    long_document:
      deepseek-v4-pro: 0.9
      deepseek-v4-flash: 0.3
    reasoning_chain:
      deepseek-v4-pro: 1.0
      deepseek-v4-flash: 0.6
    casual_chat:
      deepseek-v4-pro: 0.5  # Overkill — flash is plenty
      deepseek-v4-flash: 1.0
    planning:
      deepseek-v4-pro: 1.0
      deepseek-v4-flash: 0.6

  # Cost multipliers (from pricing config, but normalized for comparison)
  # flash is ~5-10x cheaper than pro; this is already in gateway.yaml pricing
```

Scores are **0.0–1.0** where 1.0 = best-in-class fit. Unknown models default to
config-based pricing (already in `gateway.yaml`). Unknown tasks default to `pro`
(safe fallback — same as current behavior).

The scoring formula:

```python
def score_model(task: str, model: str, matrix, pricing, cost_bias: float) -> float:
    capability = matrix.get(task, {}).get(model, 0.5)  # default: unknown fit
    cost_factor = normalize_cost(pricing.get(model, 1.0))  # 0=cheapest, 1=most expensive
    return capability + cost_bias * (1.0 - cost_factor)  # boost cheaper models
```

### Integration Points

1. **`src/api/router.py`** — New `CapabilityRouter` class replacing `DynamicRouter`
2. **`src/api/config.py`** — Parse `dynamic_routing` section from `gateway.yaml`
3. **`src/api/request_pipeline.py`** — Wire router into `try_chain()`: override model
   in first chain entry when router recommends a different one
4. **`src/ui/dashboard.py`** — Show routing decisions in request log
5. **Profile-level override**: `profiles.{name}.dynamic_routing: false` disables
   for specific profiles (e.g., `cron` always flash, `coder` is already manually chosen)

### What this replaces

- `DynamicRouter` class → `CapabilityRouter` class
- Hardcoded `MODEL_MAP` → Config-driven capability matrix
- Binary flash/pro decision → N-model scoring
- `enabled=False` default → Config `dynamic_routing.enabled`

---

## Tier 2: LLM Council

### Concept

When quality matters more than latency/cost, route a single user request through
**multiple models in parallel**, then either:
- Aggregate responses into a synthesis (MoA pattern)
- Have models debate and converge (MIT MAD pattern)
- Use a judge model to pick the best answer (MAD with judge)

### Three Council Modes

| Mode | Flow | Latency | Cost | Best For |
|------|------|---------|------|----------|
| **Synthesis** | N models → aggregator → response | 1× (parallel) | N+1 | Planning, design, research |
| **Debate** | N models critique each other × R rounds | R× (sequential) | N×R | Reasoning, math, logic |
| **Judge** | N models → judge picks best | 1× (parallel) | N+1 | Factual Q&A, code review |

### LCP Implementation Approach

LCP's role is the **orchestrator**, not the debater. The gateway:

1. Receives a single `/chat/completions` request
2. Fans out to N providers in parallel (reusing existing `forward_request` infrastructure)
3. Waits for all responses (with configurable timeout)
4. Feeds them into an aggregator/judge model (or streams the debate)
5. Returns the final synthesis to the client

```yaml
# config/gateway.yaml (future)
council:
  enabled: false  # off by default — opt-in per profile
  profiles:
    planning:
      mode: synthesis
      models: [deepseek-v4-pro, deepseek-v4-flash]
      aggregator: deepseek-v4-pro
      timeout: 120s
    reasoning:
      mode: debate
      models: [deepseek-v4-pro, deepseek-v4-flash]
      rounds: 2
```

### Council vs Capability Router

These two features are **complementary, not competing**:
- **Capability Router** replaces one model with a better one (1:1 substitution, saves cost)
- **Council** uses multiple models together (N:1 fan-out, increases quality)
- The router picks the **cheapest model that can do the job**
- The council picks the **best answer from multiple perspectives**

The router is appropriate for 95% of requests. The council is for the 5% where
you're willing to spend 3-5× more for a better answer.

---

## Tier 3: True Harness

### Concept

The Continual Harness (Prime Agent's pattern) is a **self-improving agent state**
that persists across sessions. The agent learns from its own trajectory and edits
its own prompts, memories, skills, and subagent definitions.

Formally: $H = (\rho, G, K, M)$ where:
- $\rho$ = supplemental prompt notes
- $G$ = persistent subagent specs
- $K$ = skills (SKILL.md-like Python modules)
- $M$ = named memories (facts/patterns learned from trajectory)

### Why This Matters for LCP

LCP currently operates at the **request level** — each chat completion is stateless.
The harness would make LCP **session-aware**:

1. **Per-profile memory** — "Last 3 times the `coder` profile hit circuit breaker on `opencode`, the error was rate-limiting. Try `deepseek` first next time."
2. **Prompt refinement** — LCP notices that `l2` profile requests keep getting tool-blocked errors because Hermes includes `write_file` in tools. LCP auto-strips it before forwarding.
3. **Skill patterns** — LCP observes that the `career` profile always asks the same 3 questions before a job search. LCP pre-fills the system prompt.
4. **Subagent routing** — Complex `coder` tasks fan out to sub-profiles automatically.

### LCP's Unique Position

Unlike Prime Agent (which runs the harness inside the agent process), LCP sits
**between the agent and the providers**. This means LCP can:

- Inject harness state into system prompts **transparently** (no agent changes needed)
- Track **cross-agent patterns** (all your Hermes profiles, VS Code Copilot, etc.)
- Apply harness refinements **at the network level** (like a CDN for prompt engineering)

### Phased Approach

| Phase | What | How |
|-------|------|-----|
| **3a: Memory** | Simple key-value memories per profile, SQLite-backed | `/api/harness/{profile}/memories` CRUD |
| **3b: Prompt injection** | Inject relevant memories into system prompt before forwarding | Middleware in `request_pipeline.py` |
| **3c: Auto-memory** | LCP auto-creates memories from repeated error patterns | Hooks into circuit breaker + error log |
| **3d: `/refine`** | Agent or admin can trigger refinement via API | LLM call analyzes trajectory → proposes CRUD |
| **3e: Skills** | Per-profile skill packages (Python modules or prompt templates) | `/api/harness/{profile}/skills` |

---

## Research Sources

### Dynamic Routing
- **RouteLLM**: <https://github.com/lm-sys/RouteLLM> — Pairwise model routing via learned embeddings + preference data. 4 router types (MF, SW-Ranking, BERT, CausalLLM). Paper: <https://arxiv.org/abs/2406.18665>
- **RoRF**: <https://github.com/Not-Diamond/RoRF> — Random Forest on prompt embeddings. 4-class pairwise routing. CPU-only, fast training.
- **IRT-Router**: <https://github.com/Not-Diamond/IRT-Router> — MIRT-based multi-model routing with learned capability dimensions. ACL 2025. Paper: <https://arxiv.org/abs/2506.01048>
- **OpenRouter Auto Beta**: <https://openrouter.ai/docs/features/model-routing> — Explicit task classification (30 types) + market spend-share ranking. Proprietary, but well-documented.
- **Pulze KNN Router**: <https://github.com/pulzeai-oss/knn-router> — KNN in embedding space, weighted target ranking. Go.
- **Semantic Router**: <https://github.com/aurelio-labs/semantic-router> — Intent routing via cosine similarity. 7k+ stars.

### Council / Mixture-of-Agents
- **MoA (Together AI)**: <https://github.com/togethercomputer/MoA> — Layered parallel generation + aggregation. 65.1% on AlpacaEval 2.0. Paper: <https://arxiv.org/abs/2406.04692>
- **Multi-Agent Debate (MIT)**: <https://arxiv.org/abs/2305.14325> — Iterative debate improves factuality and reasoning
- **MAD (Tencent)**: <https://arxiv.org/abs/2305.19118> — Debate with judge; identified Degeneration-of-Thought problem
- **Prime Agent RLM**: <https://www.primeintellect.ai/blog/prime-agent> — Hierarchical delegation with `rlm(prompt)` subagent spawning + A2A messaging

### Harness
- **Prime Agent Continual Harness**: <https://arxiv.org/abs/2605.09998> — Self-improving agent state with CRUD surface, trajectory-backed refinements
- **Prime Agent repo**: <https://github.com/PrimeIntellect-ai/prime-agent>

### LiteLLM (for comparison)
- **LiteLLM Router**: <https://docs.litellm.ai/docs/routing> — 5 load-balancing strategies (same-model, multi-deployment). Does NOT do content-based model selection.
- **LiteLLM repo**: <https://github.com/BerriAI/litellm> — 56k stars, 100+ providers
