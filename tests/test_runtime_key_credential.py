"""Phase C tests: key_manager + credential_store as runtime components."""

import pytest

from src.api.runtime import Runtime


@pytest.fixture(autouse=True)
def _reset_km_cs_globals():
    import src.api.key_manager as km
    import src.api.credential_store as cs
    import src.api.runtime as runtime
    yield
    runtime._active_runtime = None
    km._key_manager = None
    cs._credential_store = None


@pytest.fixture
def rt():
    import src.api.key_manager as km
    import src.api.credential_store as cs
    from src.api.key_manager import KeyManagerComponent
    from src.api.credential_store import CredentialStoreComponent
    rt = Runtime(engine=object(), data_dir="/data")
    rt.register(KeyManagerComponent())
    rt.register(CredentialStoreComponent())
    rt.start()
    km.bind_runtime(rt)
    cs.bind_runtime(rt)
    return rt


def test_components_register_and_provide(rt):
    assert rt.is_active("key_manager") is True
    assert rt.is_active("credential_store") is True
    assert rt.resolve("key_manager").manager is not None
    assert rt.resolve("credential_store").store is not None


def test_components_inject_engine_and_data_dir(rt):
    km = rt.resolve("key_manager").manager
    cs = rt.resolve("credential_store").store
    assert km._engine is not None
    assert cs._engine is not None


def test_facades_delegate_when_bound(rt):
    from src.api.key_manager import get_key_manager
    from src.api.credential_store import get_credential_store
    assert get_key_manager() is rt.resolve("key_manager").manager
    assert get_credential_store() is rt.resolve("credential_store").store


def test_facades_fall_back_when_unbound(monkeypatch):
    import src.api.key_manager as km
    import src.api.credential_store as cs
    from src.api.key_manager import get_key_manager
    from src.api.credential_store import get_credential_store
    monkeypatch.setattr("src.api.runtime._active_runtime", None)
    km._key_manager = None
    cs._credential_store = None
    assert get_key_manager() is None       # no engine passed → None (legacy)
    assert get_credential_store() is None
    e = object()
    assert get_key_manager(engine=e) is not None
    assert get_credential_store(engine=e) is not None
