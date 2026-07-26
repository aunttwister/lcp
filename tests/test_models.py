"""Tests for models.py"""
import os
import tempfile
import pytest
import sys
from sqlalchemy import create_engine
from src.api.models import Base, Request, get_engine, get_session

def test_table_name():
    assert Request.__tablename__ == "requests"

def test_has_columns():
    cols = [c.name for c in Request.__table__.columns]
    for name in ["id", "timestamp", "profile", "model", "provider",
                 "prompt_tokens", "completion_tokens", "cache_hit_tokens",
                 "cache_miss_tokens", "cost", "latency_ms", "success"]:
        assert name in cols

def test_cost_column_float():
    col = Request.__table__.columns["cost"]
    assert str(col.type) == "FLOAT"

def test_prompt_tokens_default():
    col = Request.__table__.columns["prompt_tokens"]
    assert col.default.arg == 0

def test_success_default():
    col = Request.__table__.columns["success"]
    assert col.default.arg == 1

def test_sqlite_tmp_file():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        engine.dispose()
    finally:
        os.unlink(db_path)
        # Also clean up WAL/SHM if created
        for ext in ["-wal", "-shm"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass

def test_session_context():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        session = get_session(engine)
        req = Request(
            timestamp="2026-06-18T00:00:00",
            profile="l2",
            model="deepseek-v4-pro",
            provider="deepseek",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            latency_ms=500,
        )
        session.add(req)
        session.commit()
        session.close()
        engine.dispose()
    finally:
        os.unlink(db_path)
        for ext in ["-wal", "-shm"]:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass
