"""Tests for seed_capabilities.py — LiveBench snapshot derivation + seeding."""
import os
import tempfile

import pytest

from src.api.seed_capabilities import (
    LIVEBENCH_DATA,
    derive_category_scores,
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


def test_derive_category_scores_matches_hand_typed():
    """Derived top-level scores must equal the hand-typed leaderboard values."""
    from src.api.livebench_tasks import LIVEBENCH_TASKS

    for model in sorted(set(LIVEBENCH_DATA) & set(LIVEBENCH_TASKS)):
        hand = LIVEBENCH_DATA[model]
        latest_release = list(hand.keys())[-1]
        hand_cats = {k: v for k, v in hand[latest_release].items() if k != "overall"}
        derived = derive_category_scores(LIVEBENCH_TASKS[model])
        derived_cats = {k: v for k, v in derived.items() if k != "overall"}
        assert derived_cats == hand_cats, f"{model}: derived {derived_cats} != hand {hand_cats}"
        assert derived["overall"] == hand[latest_release]["overall"]


def test_seed_livebench_seeds_derived_models(db_path):
    """Seeding stores top-level scores for subtask-only models (gpt-5.6-luna)."""
    from src.api.models import ModelCapability, get_session, get_engine
    from src.api.seed_capabilities import seed_livebench

    seed_livebench(db_path)

    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        rows = session.query(ModelCapability).filter_by(
            model="gpt-5.6-luna", source="livebench"
        ).all()
        assert rows, "gpt-5.6-luna should have seeded capability rows"
        by_task = {r.task_type: r.score for r in rows}
        assert by_task["casual_chat"] == pytest.approx(0.726, abs=1e-3)  # language 72.6
        assert by_task["reasoning_chain"] == pytest.approx(0.872, abs=1e-3)  # math 87.2 (math after reasoning, matching hand-typed)
    finally:
        session.close()
        engine.dispose()


def test_seed_livebench_seeds_minimax(db_path):
    """minimax-m3 (subtask-only) should also get top-level scores."""
    from src.api.models import ModelCapability, get_session, get_engine
    from src.api.seed_capabilities import seed_livebench

    seed_livebench(db_path)

    engine = get_engine(db_path)
    session = get_session(engine)
    try:
        rows = session.query(ModelCapability).filter_by(
            model="minimax-m3", source="livebench"
        ).all()
        assert rows, "minimax-m3 should have seeded capability rows"
    finally:
        session.close()
        engine.dispose()
