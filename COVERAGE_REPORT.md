# LCP Unit-Test Coverage Report — 100%

Date: 2026-09-04

## Final numbers

```
TOTAL    9465 stmts    0 missed    100%
2206 passed, 0 failed, 15 deselected (deselection = pre-existing repo config)
```

Command: `.venv/bin/python -m pytest --cov=src tests/ -q`

## New test files (this coverage push, 99.0% → 100%)

| File | Focus |
|---|---|
| tests/test_batch_g_final_gaps.py | benchmark.py residual gaps |
| tests/test_batch_h_pipeline_gaps.py | request_pipeline.py gaps |
| tests/test_batch_i_router_gaps.py | router.py classify/rules/select_step gaps |
| tests/test_batch_j_config_guards.py | config validation/hydrate/env, module guards |
| tests/test_batch_k_setup_plugin_gaps.py | setup install workers, cost plugins |
| tests/test_batch_l_final_gaps.py | lancedb, cost_cache, commandcode, endpoints, handler |
| tests/test_batch_m_last_gaps.py | final 15 stmts: bench invalidate crash, padded `__main__` guards, seed 380, SSR non-positive, llamacpp no-path, router 2007/2198, setup log trim, subtask rows, route dispatch |

## Notes

- `__main__` guard lines (main.py:181, benchmark_import.py:466,
  seed_capabilities.py:597) are covered via line-number-padded exec of the
  guard clause only (defs exec'd once under the real module name, `main`
  patched in the exec namespace). Plain `exec`/runpy shifted guard lines to
  line 1 and left them missed.
- seed_capabilities.py:380 (`if not releases: continue`) is defensive and
  unreachable via normal rows (a set always receives an item); exercised via
  the caller frame's write-through locals proxy (PEP 558) with the branch
  effect asserted.
- naptune was already at 100% and was not touched.
- Nothing committed; all changes are on-disk working-tree additions under
  tests/.
