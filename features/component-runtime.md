# Feature: Component Runtime (Spatiotemporal Composability)

**Created:** 2026-08-16
**Status:** implemented (2026-08-28)
**Phase:** cross-cutting (post-Phase 7, architectural)

> **Implemented.** The reference documentation for the shipped runtime lives
> in [`docs/component-runtime.md`](../docs/component-runtime.md). This file
> retains the original design rationale; the code has since diverged in minor
> ways (e.g. `model_registry`/`secret_key` are not components; the router uses
> `[engine, settings]`; request-path resolution goes through
> `resolve_service(key, fallback)`).

## North Star

Replace the ~13 ad-hoc module-level singletons and the order-sensitive bootstrap in `src/main.py` with a declarative **component runtime**: one context object that owns the shared dependencies (`config`, `engine`, `data_dir`), where every module *declares* what it needs (`requires`), *declares* what it publishes (`provides`), and *returns its own cleanup* (`dispose`). The runtime wires the graph, and reloads only the component whose declared dependency changed.

The design is grounded in *"A Programming Paradigm for Spatiotemporal Composability"* (Yifan Shi, Wei Zhang — Peking University; Tianyi Cui — DeepSeek-AI; https://github.com/cordiverse/paper), applied to the pattern LCP already uses in `cost_plugins/`.

## Paper in two terms

The paper asks one question — how do you add/remove/replace pieces at runtime without restarting or breaking neighbors — and answers it with two mechanisms, each a "lift" of a compile-time PL concept to a runtime mechanism:

- **Effect** = what a component *does to* the shared environment. **Revertible effect** = every mutation carries an explicit inverse (undo), tracked by the runtime and replayed in LIFO order on teardown. → solves *temporal* composability (clean removal).
- **Coeffect** = what a component *needs from* the shared environment. **Reactive coeffect** = a component *declares* its dependencies instead of reaching for them; the runtime notifies it when a declared dependency appears, disappears, or changes provider. → solves *spatial* composability (safe inter-dependency).

Both live in one **context** object arranged in a **tree** (each component gets a child context). A component is ACTIVE only while all its declared dependencies are satisfied. This is the whole paradigm; the paper's 60-theorem metatheory is not needed to apply it.

## What LCP does today (observed)

LCP wires itself through **13 module-level singletons** using **3 different idioms**, sequenced by hand in `main.py`:

| Singleton | Idiom | Deps |
|---|---|---|
| `config` | `get_config()` / `init_config()` | — (loads YAML) |
| `engine` (DB) | `get_engine(db_path)` | — |
| `circuit_breaker` | `get_circuit_breaker(config)` then `.attach_engine(engine)` | config → engine (post-hoc) |
| `key_manager` | `get_key_manager()` / `init_key_manager(engine, data_dir)` | engine, data_dir |
| `credential_store` | `get_credential_store()` / `init_credential_store(engine, data_dir)` | engine, data_dir |
| `alert_manager` | `get_alert_manager(engine)` / `init_alert_manager(engine)` | engine |
| `cost_plugins` | `get_registry()` / `init_plugins(engine=…)` + import-time auto-register | engine |
| `prompt_cache` | `get_prompt_cache()` | — |
| `token_verifier` | `get_token_verifier()` | — |
| `reasoning_store` | `get_reasoning_store()` | engine (implicit) |
| `dynamic_router` | `get_dynamic_router()` / `init_router(db_path, enabled)` | db_path |
| `model_registry` | `get_model_registry(db_path)` + module cache | db_path |
| `secret_key` | `get_secret_key(data_dir)` | data_dir |

The bootstrap is order-sensitive — `main.py` line 46 carries the comment `# Database (must exist before plugins that query it)`, and `engine` is injected three different ways (`init_plugins(engine=…)`, `init_alert_manager(engine)`, `get_circuit_breaker().attach_engine(engine)`).

Two structural facts stand out:

1. **`config` and `engine` are the two hub dependencies** — nearly every module touches at least one.
2. **`cost_plugins/` is already a real component system** — identity (`provider_name`), lifecycle hooks (`on_startup`/`on_shutdown`), collision detection on `register()`. This plan *generalizes* it rather than inventing a new pattern.

## Target architecture

```
                        ┌───────────────────────────────────────────────┐
                        │  Runtime (the context)                        │
                        │  owns: config, engine, data_dir, components   │
                        │  resolves: requires → provides, topologically │
                        │  tracks: dispose stack (LIFO) per component   │
                        └───────────────┬───────────────────────────────┘
                                        │  wires
   ┌──────────────┬──────────────┬──────┴─────┬──────────────┬──────────────┐
   │circuit_breaker│ alert_manager│key_manager │credential_store│ cost_plugins │
   │  requires:    │ requires:    │requires:   │ requires:      │ requires:    │
   │  [config,     │  [engine]    │[engine,    │  [engine,      │  [engine]    │
   │   engine]     │              │ data_dir]  │   data_dir]    │ provides:    │
   │ provides: []  │ provides: [] │ provides: []│ provides: []   │  [pricing,*] │
   └──────────────┴──────────────┴────────────┴────────────────┴──────────────┘

   main.py reduces to:
       rt = Runtime(config, engine, data_dir)
       rt.register(CircuitBreaker(), AlertManager(), KeyManager(), ...)
       rt.start()          # topo-sort, setup each, wire providers
       rt.serve_forever()  # request path reads rt instead of get_*()
```

## Component contract

```python
# src/api/component.py

class Component(ABC):
    """A unit of composition. Declares deps (coeffects) and returns its own undo."""

    name: str                  # stable id, used as the registry key
    requires: list[str] = []   # keys read from the runtime
    provides: list[str] = []   # keys published to the runtime

    @abstractmethod
    def setup(self, rt: "Runtime") -> Callable[[], None] | None:
        """Acquire resources and bind dependencies.

        Returns an optional cleanup closure — the *inverse* of every effect
        this component performs. The runtime stacks these and replays them
        in LIFO order on teardown. This replaces best-effort on_shutdown().
        """

    def on_dependency_change(self, key: str) -> None:
        """Called when a declared dependency's provider changes (reactive coeffect).
        Default: no-op. Components that must react to a provider swap override this;
        the runtime may reload them if the new provider differs."""


# src/api/runtime.py

class Runtime:
    """The context. Owns shared deps and the component graph."""

    def __init__(self, config, engine, data_dir: str): ...

    def register(self, comp: Component) -> None:
        """Register a component. Collision on name or provides-key is an error."""

    def resolve(self, key: str) -> Any:
        """Resolve a provided key. Raises UndeclaredDependency if unsatisfied."""

    def start(self) -> None:
        """Topo-sort components by requires/provides; setup each; track disposers."""

    def reload(self, name: str) -> None:
        """Dispose + re-setup one component (and notify/reload its dependents)."""

    def shutdown(self) -> None:
        """Replay all tracked disposers in LIFO order."""
```

The `requires`/`provides` keys are the **hub deps** only — not a full DI framework. Initially three keys exist: `config`, `engine`, `data_dir`. That is the whole wiring surface; anything finer-grained is over-engineering for a single-container stdlib app.

## Migration path (incremental, additive)

No big-bang rewrite. The existing `get_*`/`init_*` singletons remain as **thin facades** over the runtime during the transition, so the request path in `handler.py` keeps working until each call site is migrated:

```python
# During transition — singleton stays, but delegates to the runtime
_runtime: Runtime | None = None

def get_circuit_breaker(config=None) -> CircuitBreaker:
    if _runtime is not None:
        return _runtime.resolve("circuit_breaker")
    # legacy path (tests, boot before runtime exists)
    ...
```

Components are migrated one at a time, each a self-contained commit. `cost_plugins/` is the pilot because it already has `provider_name` + lifecycle hooks — the closest existing thing to the contract.

## Config reconciliation (incremental hot-reload)

`config.check_reload()` today re-reads the *entire* `gateway.yaml` and swaps `self._data`. Replace with a **diff against the previous data**: for each changed `providers`/`profiles`/`pricing` entry, dispatch the least-disruptive update (rebuild that one component) instead of a full reload. This is the paper's per-entry "least-disruptive operation" (Cordis §5.2.1). Providers whose config is untouched stay live across a reload.

## Non-goals (explicitly out of scope)

- **Do not make the cost ledger revertible.** A billed token or a sent request is *outside the system boundary* (paper §6.1) — it can only be recorded or compensated, never undone. The append-only `costs.db` is already correct.
- **Do not port Cordis.** Python's `__get__` descriptors, decorators, and `importlib` provide everything the TypeScript `Proxy` + Node module registry give the paper (paper §6.4). This is a convention + a registry, not a new dependency.
- **Ignore the metatheory.** The paper's preservation/progress/confluence proofs are not needed — only the two mechanisms (declare deps + revertible effects) and the loader's reconciliation.
- **No new infra.** Single container, Python stdlib + SQLite, as today. No PostgreSQL, Redis, or a service mesh.

## Implementation Plan

### Phase A: Runtime + Component base

| Step | File | What |
|---|---|---|
| A1 | `src/api/component.py` | `Component` ABC: `name`, `requires`, `provides`, `setup() -> disposer`, `on_dependency_change()`. |
| A2 | `src/api/runtime.py` | `Runtime`: `register`, `resolve`, `start` (topo-sort), `reload`, `shutdown` (LIFO disposers). `UndeclaredDependency` exception. |

### Phase B: Pilot — migrate `cost_plugins`

| Step | File | What |
|---|---|---|
| B1 | `src/api/cost_plugins/base.py` | Subclass `Component`. `requires = ["engine"]`. `setup()` returns a disposer instead of relying on `on_shutdown`. |
| B2 | `src/api/cost_plugins/{deepseek,opencode,llamacpp,commandcode}.py` | Auto-register against the runtime instead of the module-level `get_registry()`. |
| B3 | `src/api/cost_plugins/base.py` | Delete the `hasattr(plugin, "set_engine")` injection hack (pitfall #42) — deps now declared, not probed. |

### Phase C: Migrate remaining singletons

| Step | File | What |
|---|---|---|
| C1 | `src/api/circuit_breaker.py` | `requires = ["config", "engine"]`. Kill the `.attach_engine()` post-hoc step. |
| C2 | `src/api/key_manager.py`, `credential_store.py`, `alert_manager.py` | `requires = ["engine", "data_dir"]` / `["engine"]`. Replace `init_*` with `setup()` returning disposers. |
| C3 | `src/api/prompt_cache.py`, `token_verifier.py`, `reasoning_store.py` | Register as components; declare the (few) deps they actually use. |
| C4 | `src/api/router.py` | `CapabilityRouter` + `model_registry` as a component; `requires = ["engine"]`. |

### Phase D: Rewire bootstrap + request path

| Step | File | What |
|---|---|---|
| D1 | `src/main.py` | Collapse the hand-sequenced bootstrap into `rt = Runtime(...); rt.register(...); rt.start()`. Delete the order-sensitive comments. |
| D2 | `src/server/server.py` | Replace class-attribute wiring (`ConfiguredHandler.config/.engine`) with a reference to the runtime. |
| D3 | `src/server/handler.py` | Replace `get_*()` call sites with `rt.resolve(...)` (or keep singletons as facades until the last one migrates). |

### Phase E: Config reconciliation + tests

| Step | File | What |
|---|---|---|
| E1 | `src/api/config.py` | Diff-based `check_reload()` — per-entry reconciliation instead of whole-file swap. |
| E2 | `tests/test_runtime.py` | Topo-sort order, `requires`/`provides` resolution, unsatisfied dep raises, disposer replay is LIFO, reload notifies dependents. |
| E3 | `tests/test_component.py` | ABC cannot instantiate, default `on_dependency_change` is a no-op, disposer returned from `setup` runs on `shutdown`. |
| E4 | `tests/conftest.py` | Shared `Runtime` fixture; migrate existing mock configs to register components. |

## Relevant Files (13 new, 15 modified)

**New:**
- `src/api/component.py`
- `src/api/runtime.py`
- `tests/test_runtime.py`
- `tests/test_component.py`

**Modified:**
- `src/main.py` (bootstrap collapse)
- `src/server/server.py` (runtime ref instead of class attrs)
- `src/server/handler.py` (resolve instead of get_*)
- `src/api/config.py` (diff-based reconciliation)
- `src/api/circuit_breaker.py`, `key_manager.py`, `credential_store.py`, `alert_manager.py`, `prompt_cache.py`, `token_verifier.py`, `reasoning_store.py`, `router.py` (component migration)
- `src/api/cost_plugins/base.py` + `deepseek.py`, `opencode.py`, `llamacpp.py`, `commandcode.py` (pilot migration)
- `tests/conftest.py`

## Verification

1. `python3 -m pytest tests/ -q` — all existing + new tests pass (118 baseline preserved).
2. Boot log shows a single `runtime_started` line with the topo-sorted component order, replacing the 8 individual `startup_step` lines.
3. `docker compose up -d --build` — container starts, `/health` returns `ok`, dashboard renders, no dangling-state regressions (pitfall #32 headers, #17 silent-except).
4. Edit one provider in `gateway.yaml` + `touch` → only that provider's component reloads (log line `component_reloaded`, not `config_loaded`).
5. `Runtime.shutdown()` → disposers fire in reverse registration order (verified by a test that records call order).

## Status of this proposal

Open. Pending Pavle's review before any code changes. Phase B (the `cost_plugins` pilot) is the smallest self-contained slice that proves the pattern end-to-end before the full singleton migration.
