"""Tests for the Component ABC (src/api/component.py)."""

import pytest

from src.api.component import Component
from src.api.runtime import Runtime


def test_component_is_abstract():
    """Component cannot be instantiated without implementing setup."""
    with pytest.raises(TypeError):
        Component()  # type: ignore[abstract]


def test_component_requires_name():
    class _NoName(Component):
        def setup(self, rt):
            return None

    with pytest.raises(TypeError):
        _NoName()


def test_component_setup_returns_disposer_run_on_shutdown():
    calls = []

    class _C(Component):
        name = "c"

        def setup(self, rt):
            assert rt.resolve("config") is not None
            return lambda: calls.append("disposed")

    rt = Runtime(config=object())
    rt.register(_C())
    rt.start()
    assert rt.is_active("c") is True
    rt.shutdown()
    assert calls == ["disposed"]


def test_component_default_on_dependency_change_is_noop():
    calls = []

    class _C(Component):
        name = "c"

        def setup(self, rt):
            return None

    rt = Runtime()
    c = _C()
    rt.register(c)
    rt.start()
    # Default override is a no-op — must not raise.
    c.on_dependency_change("whatever")
    assert calls == []


def test_component_provides_publishes_keys():
    class _P(Component):
        name = "provider"
        provides = ["pricing"]

        def setup(self, rt):
            return None

    rt = Runtime()
    rt.register(_P())
    rt.start()
    assert rt.resolve("pricing") is rt.resolve("provider")
