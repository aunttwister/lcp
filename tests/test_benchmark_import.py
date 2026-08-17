"""Tests for benchmark_import.py — the JSON → SQLite import pipeline."""
import json
import os
import tempfile

import pytest

from src.api.benchmark_import import (
    discover_files,
    import_bundled,
    import_file,
    parse_payload,
)


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from src.api.models import get_engine, Base
    engine = get_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield path
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


def _payload():
    return {
        "schema_id": "livebench",
        "release_label": "2026-06-25",
        "models": {
            "test-model": {
                "releases": {"2026-06-25": {
                    "reasoning": 90.0, "coding": 80.0, "math": 85.0,
                    "language": 70.0, "overall": 81.25,
                }},
                "subtasks": {
                    "reasoning": {"theory_of_mind": 90.0, "zebra_puzzle": 80.0},
                    "coding": {"code_generation": 75.0},
                },
            },
        },
    }


def test_parse_payload_flattens_rows():
    schema_id, release, rows = parse_payload(_payload())
    assert schema_id == "livebench"
    assert release == "2026-06-25"
    assert any(r["task"] is None and r["category"] == "reasoning" for r in rows)
    assert any(r["task"] == "theory_of_mind" for r in rows)


def test_parse_payload_derives_missing_top_level():
    """Models with subtasks but no releases get derived top-level scores."""
    payload = {
        "schema_id": "livebench",
        "release_label": "2026-06-25",
        "models": {
            "sub-only": {
                "subtasks": {
                    "reasoning": {"theory_of_mind": 100.0, "zebra_puzzle": 50.0},
                    "coding": {"code_generation": 60.0},
                },
            },
        },
    }
    _, _, rows = parse_payload(payload)
    top = {r["category"]: r["value"] for r in rows if r["task"] is None}
    assert top["reasoning"] == 75.0
    assert top["coding"] == 60.0
    assert top["overall"] == 67.5


def test_parse_payload_requires_schema():
    with pytest.raises(ValueError):
        parse_payload({"models": {}})


def test_discover_files_finds_bundled():
    files = discover_files()
    assert any(f.endswith("livebench_2026_06_25.json") for f in files)


def test_import_file_writes_metrics_and_typed_rows(db_path):
    fd, jpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(jpath, "w") as f:
        json.dump(_payload(), f)

    import_file(db_path, jpath)

    from src.api.models import (
        CapabilityMetric, ModelCapability, ModelCapabilitySubtask,
        get_engine, get_session,
    )
    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        assert session.query(CapabilityMetric).count() == 8  # 5 top-level + 3 subtasks
        caps = session.query(ModelCapability).filter_by(model="test-model").all()
        assert caps  # top-level typed rows exist
        subs = session.query(ModelCapabilitySubtask).filter_by(model="test-model").all()
        assert len(subs) == 3
    finally:
        session.close()
        engine.dispose()
        os.unlink(jpath)


def test_import_bundled_seeds_known_models(db_path):
    import_bundled(db_path)

    from src.api.models import (
        CapabilityMetric, ModelCapability, ModelCapabilitySubtask,
        get_engine, get_session,
    )
    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        assert session.query(CapabilityMetric).count() > 0
        # Subtask-only models get top-level typed rows.
        luna = session.query(ModelCapability).filter_by(model="gpt-5.6-luna").all()
        assert luna
        minimax = session.query(ModelCapability).filter_by(model="minimax-m3").all()
        assert minimax
        # Subtasks exist for deepseek-v4-pro.
        subs = session.query(ModelCapabilitySubtask).filter_by(model="deepseek-v4-pro").all()
        assert subs
    finally:
        session.close()
        engine.dispose()
