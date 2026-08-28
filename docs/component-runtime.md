# Component Runtime

LCP's bootstrap used to be a hand-sequenced list of ~16 steps that wired ~13
module-level singletons in a specific order. It is now a **declarative
component runtime**: every subsystem declares what it needs (`requires`),
what it publishes (`provides`), and returns its own cleanup. The runtime
topologically orders the components, starts them, and tears them down in
reverse.

This document describes the contract and how the gateway is wired. The design
rationale lives in [`features/component-runtime.md`](../features/component-runtime.md).

---

## 1. The contract

Two small modules define everything.

### `src/api/component.py` — `Component`

```python
class Component(ABC):
    name: str                  # stable registry key
    requires: list[str] = []   # keys read from the runtime (coeffects)
    provides: list[str] = []   # keys published to the runtime (effects)

    def setup(self, rt) -> Disposer | None:
        """Acquire resources. Return the inverse (cleanup) closure."""

    def on_dependency_change(self, key: str) -> None:
        """React when a declared dependency's provider changes."""
```

The **disposer** returned from `setup` is the inverse of everything the
component did — it replaces ad-hoc `finally` blocks and best-effort
`on_shutdown` hooks.

### `src/api/runtime.py` — `Runtime`

| Method | What it does |
|---|---|
| `register(comp)` | Add a component; errors on name/provides collisions or reserved root keys |
| `resolve(key)` | Resolve a published key (or root key) to a value; `UndeclaredDependency` when missing/inactive |
| `start()` | Topologically sort by `requires`/`provides` and call each `setup` in dependency order |
| `reload(name)` | Dispose + re-setup one component and notify its dependents |
| `shutdown()` | Replay every disposer in **LIFO** order (exact inverse of setup) |

**Root keys** owned directly by the runtime (not by any component): `config`,
`engine`, `data_dir`.

**Boot never breaks.** A component whose dependencies can't be satisfied
(missing key, or a dependency cycle) is logged and marked INACTIVE; a `setup`
exception is logged and the component is marked inactive. The rest of the
gateway starts regardless — a failed optional module degrades instead of
taking the process down.

## 2. The component graph

All 13 components, their declared dependencies, and what they publish:

| Component (`name`) | `requires` | `provides` | Publishes (`service`) |
|---|---|---|---|
| `settings` | `engine` | `settings` | `SettingsStore` |
| `cost_cache` | `engine` | `cost_cache` | `CostPluginCache` |
| `refresher` | `cost_cache`, `settings` | `refresher` | `CacheRefresher` (disposer stops the background thread) |
| `circuit_breaker` | `config`, `engine` | `circuit_breaker` | `CircuitBreaker` (engine attached at construction) |
| `key_manager` | `engine`, `data_dir` | `key_manager` | `KeyManager` |
| `credential_store` | `engine`, `data_dir` | `credential_store` | `CredentialStore` |
| `alert_manager` | `engine` | `alert_manager` | `AlertManager` |
| `dynamic_router` | `engine`, `settings` | `dynamic_router` | `CapabilityRouter` (settings toggle folded into `setup`) |
| `memory` | `config` | `memory` | memory backend (disposer = `shutdown_memory`) |
| `cost_plugins` | `engine` | `cost_plugins`, `pricing` | `PluginRegistry` (disposer runs each plugin's `on_shutdown`) |
| `prompt_cache` | — | `prompt_cache` | `PromptCache` |
| `token_verifier` | — | `token_verifier` | `TokenVerifier` |
| `reasoning_store` | — | `reasoning_store` | `ReasoningStore` |

Notes:

- **`engine` is the hub dependency** — nearly every component reads it.
- **Constructor injection** replaced the old post-hoc mutations:
  `CircuitBreaker.attach_engine()`, `AlertManager._engine = engine` getter
  mutation, and `hasattr(plugin, "set_engine")` probing are all gone — deps
  are declared and injected up front.
- The dep-free leaves (`prompt_cache`, `token_verifier`,
  `reasoning_store`) are duck-typed components so the runtime owns them
  uniformly.

## 3. Boot

`src/main.py::build_runtime()` is the whole bootstrap:

```python
rt = Runtime(config=config, engine=engine, data_dir=data_dir)
rt.register(SettingsComponent())
rt.register(CostCacheComponent())
rt.register(RefresherComponent())
rt.register(CircuitBreakerComponent())
rt.register(KeyManagerComponent())
rt.register(CredentialStoreComponent())
rt.register(AlertManagerComponent())
rt.register(RouterComponent(db_path=..., enabled=..., cost_bias=...))
rt.register(MemoryComponent())
rt.register(CostPluginsComponent())
rt.register(PromptCacheComponent())
rt.register(TokenVerifierComponent())
rt.register(ReasoningStoreComponent())

for mod in (circuit_breaker, cost_cache, cost_plugins, key_manager,
            credential_store, alert_manager, router, memory):
    mod.bind_runtime(rt)

rt.start()   # topo-sort + setup each, in dependency order
```

`main()` then starts the refresher's background thread, recovers stale
benchmark runs, creates the HTTP server, and serves. `finally: rt.shutdown()`
replays the disposers in reverse — stopping the refresher, releasing memory,
and running each plugin's `on_shutdown` (e.g. llama.cpp's usage-cache persist).

The boot log shows a single `runtime_started` line with the topo-ordered
component list, replacing the old 16 `startup_step` lines.

## 4. Request path

One active runtime is the single source of truth. Request-time code resolves
services through the central accessor rather than reaching for a module global:

```python
from src.api.runtime import resolve_service, get_runtime

breaker = resolve_service("circuit_breaker", fallback=get_circuit_breaker)
```

- `get_runtime()` / `bind_active_runtime(rt)` — the active runtime.
- `resolve_service(key, fallback)` — returns the component's published
  `service` when a runtime is bound and that component is active; otherwise
  returns the `fallback` (a legacy module singleton, invoked lazily when
  callable). This keeps the request path working standalone and in tests.

The hot path (`server/handler.py`, `server/sse_helpers.py`,
`api/request_pipeline.py`, and all of `server/endpoints.py`) resolves through
`resolve_service`; each module's `get_*()` facade remains for the fallback
path and for tests.

## 5. Why this matters

- **Order bugs are impossible** — `requires`/`provides` replaces "make sure
  `init_plugins` runs after `get_engine`".
- **Teardown can't leak** — LIFO disposer replay guarantees the refresher
  thread, memory backend, and plugin state are cleaned up in the correct
  order.
- **Degradation is explicit** — a failed/missing dependency marks a component
  inactive and the gateway keeps serving, instead of a half-initialized
  singleton.
- **One source of truth** — a single runtime object owns the shared
  `config`/`engine`/`data_dir` and every subsystem, instead of ~13 module
  globals scattered across the codebase.

## 6. Tests

- `tests/test_component.py` — the `Component` ABC contract.
- `tests/test_runtime.py` — topo order, LIFO shutdown, unsatisfied/cycle
  handling, reload notifies dependents, setup-failure isolation, and the
  `get_runtime`/`resolve_service` accessor.
- `tests/test_runtime_*.py` — per-component registration, provides, facade
  delegation, and disposers.
- `tests/test_boot.py` — `build_runtime` wiring and a real temp-DB full boot.
