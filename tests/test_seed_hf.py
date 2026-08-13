"""Tests for LiveBench HF ingestion in seed_capabilities."""
import sys

import pytest

from src.api.seed_capabilities import aggregate_judgments, seed_livebench_hf


# ── aggregate_judgments (pure) ──────────────────────────────────────────────

def test_aggregate_basic_correctness():
    rows = [
        {"model": "m1", "question_id": "q1", "category": "coding", "score": 1, "tstamp": 10},
        {"model": "m1", "question_id": "q2", "category": "coding", "score": 0, "tstamp": 10},
        {"model": "m1", "question_id": "q3", "category": "math", "score": 1, "tstamp": 10},
    ]
    agg = aggregate_judgments(rows)
    assert agg[("m1", "coding")] == [1, 2]
    assert agg[("m1", "math")] == [1, 1]


def test_aggregate_dedupes_reruns_latest_wins():
    # Same model+question judged twice: 0 first (stale), 1 later (latest).
    rows = [
        {"model": "m1", "question_id": "q1", "category": "coding", "score": 0, "tstamp": 10},
        {"model": "m1", "question_id": "q1", "category": "coding", "score": 1, "tstamp": 20},
    ]
    agg = aggregate_judgments(rows)
    assert agg[("m1", "coding")] == [1, 1]  # latest score wins, counted once


def test_aggregate_skips_missing_fields():
    rows = [
        {"model": "", "question_id": "q1", "category": "coding", "score": 1, "tstamp": 10},
        {"model": "m1", "question_id": "", "category": "coding", "score": 1, "tstamp": 10},
        {"model": "m1", "question_id": "q1", "category": "", "score": 1, "tstamp": 10},
        {"model": "m1", "question_id": "q1", "category": "coding", "score": None, "tstamp": 10},
        {"model": "m1", "question_id": "q2", "category": "coding", "score": 1, "tstamp": 10},
    ]
    agg = aggregate_judgments(rows)
    assert agg == {("m1", "coding"): [1, 1]}


def test_aggregate_lowercases_model_and_category():
    rows = [
        {"model": "DeepSeek-V3", "question_id": "q1", "category": "Coding", "score": 1, "tstamp": 10},
    ]
    agg = aggregate_judgments(rows)
    assert agg[("deepseek-v3", "coding")] == [1, 1]


# ── seed_livebench_hf (with mocked HF dataset) ──────────────────────────────

def test_seed_livebench_hf_upserts(tmp_path, monkeypatch):
    from src.api.models import Base, get_engine, get_session, ModelCapability

    db_path = str(tmp_path / "seed.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    # Fake streaming dataset.
    class FakeDS:
        def __iter__(self):
            return iter([
                {"model": "m1", "question_id": "q1", "category": "coding", "score": 1, "tstamp": 10},
                {"model": "m1", "question_id": "q2", "category": "coding", "score": 0, "tstamp": 10},
                {"model": "m1", "question_id": "q3", "category": "math", "score": 1, "tstamp": 10},
            ])

    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *a, **k: FakeDS(),
    )

    count = seed_livebench_hf(db_path)
    assert count == 2  # 3 rows → 2 task types (coding, math)

    with get_session(engine) as session:
        rows = {r.task_type: r for r in session.query(ModelCapability).filter_by(
            model="m1", source="livebench").all()}
        assert rows["code_generation"].score == pytest.approx(0.5)   # 1/2 correct
        assert rows["code_generation"].raw_score == pytest.approx(50.0)
        assert rows["code_generation"].benchmark_category == "coding"
        assert rows["reasoning_chain"].score == pytest.approx(1.0)   # math 1/1


def test_seed_livebench_hf_overwrites_existing_livebench_row(tmp_path, monkeypatch):
    from src.api.models import Base, get_engine, get_session, ModelCapability

    db_path = str(tmp_path / "seed2.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    # Pre-seed a stale livebench row for m1/code_generation.
    with get_session(engine) as session:
        session.add(ModelCapability(
            model="m1", task_type="code_generation", score=0.99,
            source="livebench", benchmark_category="coding", raw_score=99.0,
        ))
        session.commit()

    class FakeDS:
        def __iter__(self):
            return iter([
                {"model": "m1", "question_id": "q1", "category": "coding", "score": 0, "tstamp": 10},
                {"model": "m1", "question_id": "q2", "category": "coding", "score": 0, "tstamp": 10},
            ])

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: FakeDS())

    seed_livebench_hf(db_path)

    with get_session(engine) as session:
        row = session.query(ModelCapability).filter_by(
            model="m1", task_type="code_generation", source="livebench").first()
        assert row.score == pytest.approx(0.0)  # overwritten by 0/2 correct
        assert row.raw_score == pytest.approx(0.0)


def test_seed_livebench_hf_missing_datasets_returns_zero(tmp_path, monkeypatch):
    from src.api.models import Base, get_engine

    db_path = str(tmp_path / "seed3.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    # Force the ImportError path.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("no datasets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert seed_livebench_hf(db_path) == 0
