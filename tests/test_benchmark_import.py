"""Tests for benchmark_import.py — the JSON → SQLite import pipeline."""
import json
import os
import tempfile

import pytest

from src.api.benchmark_import import (
    discover_files,
    import_bundled,
    import_file,
    import_csv_file,
    normalize_csv_model_key,
    parse_livebench_csv,
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


# ── CSV parsing ─────────────────────────────────────────────────────────────

def _csv_text():
    return (
        "model,AMPS_Hard,code_generation,code_completion,theory_of_mind,zebra_puzzle,"
        "javascript,python,typescript,connections,typos,summarize,story_generation,"
        "simplify,paraphrase,consecutive_events,tablejoin,tablereformat,"
        "integrals_with_game,math_comp,olympiad,logic_with_navigation,spatial,plot_unscrambling\n"
        "gpt-5.5-xhigh,98.0,82.609,81.69,84.615,100.0,63.636,55.0,43.333,100.0,88.0,"
        "72.317,72.55,66.317,71.733,88.823,55.904,100.0,99.0,96.078,91.351,76.0,98.0,74.089\n"
        "claude-opus-5-max-effort,99.01,80.282,82.609,78.846,100.0,77.273,75.0,43.333,"
        "99.333,92.0,66.433,61.317,61.583,65.733,77.571,51.962,94.118,97.0,94.118,92.796,"
        "86.0,100.0,74.731\n"
        "unmapped-model,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,"
        "15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0\n"
    )


def test_parse_livebench_csv_normalizes_and_derives():
    schema, rel, rows = parse_livebench_csv(_csv_text())
    assert schema == "livebench"
    assert rel == "2026-06-25"
    models = {r["model"] for r in rows}
    assert "gpt-5.5-thinking" in models
    assert "claude-opus-5" in models
    assert "unmapped-model" not in models

    # Subtask rows exist for the thinking models.
    subs = {(r["model"], r["category"], r["task"]) for r in rows if r["task"] is not None}
    assert ("gpt-5.5-thinking", "reasoning", "theory_of_mind") in subs
    assert ("claude-opus-5", "reasoning", "zebra_puzzle") in subs

    # Top-level rows derived from subtask averages.
    top = {(r["model"], r["category"]) for r in rows if r["task"] is None}
    assert ("gpt-5.5-thinking", "overall") in top
    assert ("gpt-5.5-thinking", "reasoning") in top


def test_parse_livebench_csv_empty():
    schema, rel, rows = parse_livebench_csv("model,foo\n")
    assert schema == "livebench"
    assert rows == []


def test_normalize_csv_model_key():
    assert normalize_csv_model_key("gpt-5.5-xhigh") == "gpt-5.5-thinking"
    assert normalize_csv_model_key("claude-opus-5-max-effort") == "claude-opus-5"
    assert normalize_csv_model_key("unmapped-model") is None
    assert normalize_csv_model_key("") is None


def test_import_csv_file_writes_typed_rows(db_path):
    fd, cpath = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(cpath, "w") as f:
        f.write(_csv_text())

    import_csv_file(db_path, cpath)

    from src.api.models import ModelCapabilitySubtask, get_engine, get_session
    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        subs = session.query(ModelCapabilitySubtask).filter_by(model="gpt-5.5-thinking").all()
        assert subs
    finally:
        session.close()
        engine.dispose()
        os.unlink(cpath)


def test_discover_files_finds_bundled():
    files = discover_files()
    assert any(f.endswith("table_2026_06_25.csv") for f in files)


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
        # The previously-subtask-less thinking models now have subtasks.
        gpt_thinking = session.query(ModelCapabilitySubtask).filter_by(model="gpt-5.5-thinking").all()
        assert gpt_thinking
        opus = session.query(ModelCapabilitySubtask).filter_by(model="claude-opus-5").all()
        assert opus
    finally:
        session.close()
        engine.dispose()


def test_import_json_string(db_path):
    """Uploaded JSON (a parsed dict) materializes typed rows."""
    from src.api.benchmark_import import import_json_string
    from src.api.models import ModelCapability, get_engine, get_session

    import_json_string(db_path, _payload())
    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        rows = session.query(ModelCapability).filter_by(model="test-model").all()
        assert rows
    finally:
        session.close()
        engine.dispose()


def test_parse_multipart_upload():
    """The multipart parser extracts the file field + optional release field."""
    from src.server.endpoints import _parse_multipart_upload

    payload = json.dumps({"schema_id": "x", "release_label": "r", "models": {}}).encode()
    body = (
        b"--BND\r\n"
        b'Content-Disposition: form-data; name="release"\r\n\r\n'
        b"2026-06-25\r\n"
        b"--BND\r\n"
        b'Content-Disposition: form-data; name="file"; filename="d.json"\r\n'
        b"Content-Type: application/json\r\n\r\n"
        + payload +
        b"\r\n--BND--\r\n"
    )
    data, release = _parse_multipart_upload(body, "multipart/form-data; boundary=BND")
    assert release == "2026-06-25"
    assert json.loads(data)["schema_id"] == "x"


def test_parse_multipart_upload_missing_file():
    from src.server.endpoints import _parse_multipart_upload
    body = (
        b"--BND\r\n"
        b'Content-Disposition: form-data; name="release"\r\n\r\n'
        b"2026-06-25\r\n--BND--\r\n"
    )
    with pytest.raises(ValueError, match="missing 'file'"):
        _parse_multipart_upload(body, "multipart/form-data; boundary=BND")


def test_parse_multipart_upload_no_boundary():
    from src.server.endpoints import _parse_multipart_upload
    with pytest.raises(ValueError, match="boundary"):
        _parse_multipart_upload(b"data", "multipart/form-data")
