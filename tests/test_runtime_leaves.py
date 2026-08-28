"""Phase C tests: dep-free leaf components (prompt_cache, token_verifier,
reasoning_store) and the memory component."""

import pytest

from src.api.runtime import Runtime


def test_leaf_components_register_and_provide():
    from src.api.prompt_cache import PromptCacheComponent
    from src.api.token_verifier import TokenVerifierComponent
    from src.api.reasoning_store import ReasoningStoreComponent
    rt = Runtime()
    rt.register(PromptCacheComponent())
    rt.register(TokenVerifierComponent())
    rt.register(ReasoningStoreComponent())
    rt.start()
    assert rt.is_active("prompt_cache") is True
    assert rt.is_active("token_verifier") is True
    assert rt.is_active("reasoning_store") is True
    # Facades still return the module-level instances (dep-free, unchanged).
    from src.api.prompt_cache import get_prompt_cache
    from src.api.token_verifier import get_token_verifier
    from src.api.reasoning_store import get_reasoning_store
    assert get_prompt_cache() is not None
    assert get_token_verifier() is not None
    assert get_reasoning_store() is not None


def test_memory_component_setup_and_dispose(monkeypatch):
    from src.api.memory import MemoryComponent
    import src.api.memory as mem
    # Avoid building a real LanceDB backend: patch init_memory to a no-op.
    calls = []
    monkeypatch.setattr(mem, "init_memory", lambda cfg: calls.append("init"))
    rt = Runtime(config=object())
    rt.register(MemoryComponent())
    rt.start()
    assert rt.is_active("memory") is True
    assert calls == ["init"]
    # shutdown runs the disposer (shutdown_memory), which must not raise.
    rt.shutdown()
