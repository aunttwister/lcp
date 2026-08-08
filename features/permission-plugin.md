# Feature: Permission Layer as Plugin (Phase 7)

**Created:** 2026-08-06
**Status:** open
**Phase:** 7 (from PLAN.md)

## North Star

Replace the flat `forbidden_tools` denylist with a permission plugin system — `PermissionPlugin` ABC + `PermissionRegistry` singleton, mirroring the existing `cost_plugins/` pattern exactly. The built-in plugin resolves capabilities from `gateway.yaml` modes + resources. Fails-closed (default deny), `deny_globally` cross-cutting layer, emits `X-LCP-Capabilities` header. Single choke point for all agents — VS Code on laptop, Hermes on VPS, cron jobs — all go through the same gateway and the same policy file.

## Why Plugin?

The permission model may grow beyond simple YAML modes — future plugins could call an OPA server, check LDAP groups, or integrate with a secrets manager. Plugins make the enforcement logic swappable without touching the pipeline. Same architecture as `cost_plugins/`: ABC base, singleton registry, auto-register on import.

## Architecture

```
                               ┌──────────────────────────────────┐
  Agent → LCP Gateway (:8734) →│ 1. Auth (who are you?)          │
                               │ 2. Permission Plugin.resolve()   │
                               │    ├─ deny_globally (first pass) │
                               │    ├─ expand mode → tool set     │
                               │    ├─ default deny (fails-closed)│
                               │    ├─ strip blocked from body    │
                               │    ├─ inject policy message      │
                               │    └─ return PermissionResult    │
                               │ 3. emit X-LCP-Capabilities       │
                               │ 4. forward to upstream LLM       │
                               └──────────────────────────────────┘
```

## Plugin Contract

```python
# src/api/permission_plugins/base.py

@dataclass
class PermissionResult:
    allowed_tools: list[str]       # tool names the LLM may see
    blocked_tools: list[str]       # stripped from the request
    mode: str | None               # resolved mode (ro/rw/dev)
    resources: dict                # granted resources {ssh: [deploy], memory: recall}
    capability_header: str         # value for X-LCP-Capabilities header
    policy_message: str | None     # optional system message to inject


class PermissionPlugin(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier, e.g. 'builtin'."""
        ...

    def resolve(self, profile: str, body: dict,
                profile_cfg: dict, config) -> PermissionResult:
        """Return permission decision for a request."""
        ...
```

## Builtin Plugin: YAML-Driven

```yaml
# gateway.yaml — new top-level section
permissions:
  deny_globally: [cronjob, skill_manage]   # cross-cutting, no override

  modes:                                    # capability bundles
    ro:  [read_file, search_files, session_search, web_search]
    rw:  [ro, write_file, patch]            # ro + writes (nested expansion)
    dev: [rw, terminal, execute_code, memory, delegate_task, clarify, todo, skill_view]

  resources:                                # named resources profiles can request
    ssh_keys:
      deploy:          { hosts: [node01, gateway] }
      readonly-backup: { hosts: ["*"] }

  profiles:
    l2:
      mode: dev
      resources: { ssh: [deploy], memory: retain }
    l1:
      mode: ro
    career:
      mode: ro
    cron:
      mode: ro
      resources: { ssh: [readonly-backup], memory: recall }
    coder:
      mode: dev
```

### Mode Expansion (Recursive)

`rw: [ro, write_file, patch]` → `ro` expands to `[read_file, search_files, session_search, web_search]` → final `rw` set includes all of them plus `write_file` and `patch`. Maintainers only list additions.

### Fails-Closed Default

Any tool in the request body that is **not** in the resolved mode's allowed set is blocked. A new tool added to Hermes passes silently under `forbidden_tools` but is blocked by default under the permission matrix until explicitly added to a mode.

### Backward Compat

If the `permissions` section is absent from `gateway.yaml`, the builtin plugin falls back to the legacy `forbidden_tools` behavior. No breaking change for existing deployments.

## Emitted Header

Every response carries `X-LCP-Capabilities` so the calling agent's execution layer can enforce:

```
X-LCP-Capabilities: mode=dev,ssh:deploy,memory:retain
```

Future Phase 8 daemon reads this header + the shared `gateway.yaml` to enforce at the OS level.

## vs Current `forbidden_tools`

| Aspect | Current (forbidden_tools) | New (permission_matrix) |
|---|---|---|
| Default | Pass-through (fails open) | Deny (fails closed) |
| New tool added to Hermes | Silently passes through | Blocked until whitelisted |
| Cross-cutting blocks | Repeat `cronjob` in every profile | `deny_globally` once |
| Profile maintenance | Maintain growing deny list | Select a mode |
| Audit surface | "What tools are blocked?" | "What can this profile do?" |
| Extensibility | Hardcoded in pipeline | Plugin — swap implementation |

---

## Implementation Plan

### Phase A: Permission Plugin Infrastructure

| Step | File | What |
|---|---|---|
| A1 | `src/api/permission_plugins/__init__.py` | Package init, exports `PermissionPlugin`, `PermissionRegistry`, `get_registry`, `init_plugins`. Import builtin plugin to trigger self-registration. |
| A2 | `src/api/permission_plugins/base.py` | `PermissionPlugin` ABC, `PermissionResult` dataclass, `PermissionRegistry` singleton (mirrors `PluginRegistry`). |
| A3 | `src/api/permission_plugins/builtin.py` | `BuiltinPermissionPlugin(provider_name="builtin")`. Reads YAML, mode expansion, deny_globally, capability header, backward compat fallback. Auto-registers on import. |

### Phase B: Config Layer

| Step | File | What |
|---|---|---|
| B1 | `config/gateway.yaml` | Remove `forbidden_tools` from all profiles. Add `permissions` top-level section with `deny_globally`, `modes`, `resources`, `profiles.{name}.mode`. |
| B2 | `src/api/config.py` | Add `permissions` property. Relax validation — accept either `permissions` or `forbidden_tools`. |

### Phase C: Pipeline + Handler Integration

| Step | File | What |
|---|---|---|
| C1 | `src/api/request_pipeline.py` | New `apply_permission_matrix(body, profile, profile_cfg, config) → (body, blocked_tools, capabilities)`. Calls plugin registry, strips tools, injects policy message. Keep old `strip_forbidden_tools` as backward-compat wrapper. |
| C2 | `src/server/handler.py` | Replace `strip_forbidden_tools(...)` call with `apply_permission_matrix(...)`. Emit `X-LCP-Capabilities` via `_pending_headers`. |

### Phase D: Dashboard + Endpoints

| Step | File | What |
|---|---|---|
| D1 | `src/ui/templates/jinja/pages/profiles.html` | Replace "Blocked Tools" column with "Mode" + "Allowed Tools". Mode selector dropdown (ro/rw/dev) in edit modal. JS sends `permissions: {mode: ...}`. |
| D2 | `src/server/endpoints.py` | Profile CRUD: accept `permissions` field, keep `forbidden` for backward compat in responses. |

### Phase E: Tests

| Step | File | What |
|---|---|---|
| E1 | `tests/test_permission_plugins_base.py` | ABC cannot instantiate, dummy plugin defaults, PermissionResult creation, registry singleton, register/duplicate/delegation. |
| E2 | `tests/test_permission_plugins_builtin.py` | Mode expansion (ro, rw nested, dev nested, unknown → empty), deny_globally cross-cutting, forbidden_tools fallback, default deny, capability header building, policy injection, integration with mock config. |
| E3 | `tests/test_main.py` | Replace `TestStripForbiddenTools` with `TestApplyPermissionMatrix`. Backward compat, mode-based resolution, capability header. |
| E4 | `tests/test_handler_inprocess.py` | Update profile_cfg fixtures: remove `forbidden_tools`, add permissions. |
| E5 | `tests/conftest.py` | Update shared profile fixtures to use permissions section. |

---

## Relevant Files (14 total — 6 new, 8 modified)

**New:**
- `src/api/permission_plugins/__init__.py`
- `src/api/permission_plugins/base.py`
- `src/api/permission_plugins/builtin.py`
- `tests/test_permission_plugins_base.py`
- `tests/test_permission_plugins_builtin.py`

**Modified:**
- `src/api/request_pipeline.py`
- `src/api/config.py`
- `src/server/handler.py`
- `src/server/endpoints.py`
- `src/ui/templates/jinja/pages/profiles.html`
- `config/gateway.yaml`
- `tests/test_main.py`
- `tests/test_handler_inprocess.py`
- `tests/conftest.py`

---

## Verification

1. `make test` or `.venv/bin/python -m pytest -q` — all existing + new tests pass
2. Manual: `curl /l2/chat/completions` with tools → only dev-allowed tools pass through
3. Manual: check `X-LCP-Capabilities` header: `mode=dev,ssh:deploy,memory:retain`
4. Manual: `curl /cron/chat/completions` with `write_file` → blocked, `X-LCP-Capabilities: mode=ro,...`
5. Dashboard: open profiles page → mode selector shows ro/rw/dev → save → config persisted
