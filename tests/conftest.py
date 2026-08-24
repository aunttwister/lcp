"""Shared test fixtures and configuration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_yaml_config(temp_dir):
    """Create a temporary gateway.yaml with valid structure and return its path."""
    config = {
        "server": {"port": 8734, "default_profile": "l2"},
        "profiles": {
            "l2": {
                "forbidden_tools": ["write_file"],
                "chain": [
                    {"provider": "test_prov", "model": "test-model", "base_url": "https://test.api/v1"}
                ],
            },
            "l1": {
                "forbidden_tools": ["terminal"],
                "chain": [
                    {"provider": "test_prov", "model": "test-model-flash", "base_url": "https://test.api/v1"}
                ],
            },
        },
        "providers": {
            "test_prov": {
                "api_key_env": "TEST_API_KEY",
                "api_base": "https://test.api/v1",
                "models": ["test-model", "test-model-flash"],
            }
        },
        "pricing": [
            {"provider": "test_prov", "model": "test-model", "cache_hit": 0.01, "cache_miss": 0.5, "output": 1.0},
            {"provider": "test_prov", "model": "test-model-flash", "cache_hit": 0.005, "cache_miss": 0.1, "output": 0.2},
        ],
        "circuit_breaker": {
            "failures_degraded": 3,
            "failures_dead": 6,
            "degraded_cooldown_seconds": 30,
            "dead_cooldown_seconds": 120,
        },
        "database": {"path": str(temp_dir / "test.db"), "wal_mode": True},
    }
    path = temp_dir / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def mock_config(temp_yaml_config):
    """Return a loaded Config object from a temp YAML file."""
    from src.api.config import Config
    return Config(str(temp_yaml_config))


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary SQLite database and return its path + engine."""
    from src.api.models import get_engine, Base
    db_path = str(temp_dir / "test.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return db_path, engine


@pytest.fixture
def mock_handler():
    """Create a minimal mock HTTP handler for testing."""
    h = MagicMock()
    h.send_response = MagicMock()
    h.send_header = MagicMock()
    h.end_headers = MagicMock()
    h.wfile = MagicMock()
    h.rfile = MagicMock()
    h.headers = {}
    h.path = "/"
    return h


@pytest.fixture(autouse=True)
def _init_circuit_breaker():
    """Initialize the circuit breaker singleton before each test."""
    from unittest.mock import MagicMock
    from src.api.circuit_breaker import get_circuit_breaker
    cfg = MagicMock()
    cfg.circuit_breaker = {
        "failures_dead": 5,
        "dead_cooldown_seconds": 300,
        "failures_degraded": 3,
        "degraded_cooldown_seconds": 60,
    }
    get_circuit_breaker(cfg)


@pytest.fixture(autouse=True)
def _reset_credential_store():
    """Reset the credential store singleton before each test."""
    import src.api.credential_store as cs
    cs._credential_store = None
    yield
    cs._credential_store = None
