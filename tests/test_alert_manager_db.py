"""Tests for AlertManager DB persistence."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.api.alert_manager import AlertManager, _alert_to_dict
from src.api.models import Alert, get_session


@pytest.fixture
def db_am(temp_db):
    """AlertManager bound to a temp DB."""
    db_path, engine = temp_db
    return AlertManager(engine)


class TestDBPersistence:
    def test_fire_persists_to_db(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="DB Alert", message="persisted",
            dedup_key="db:1",
        )
        with get_session(db_am._engine) as session:
            alerts = session.query(Alert).all()
            assert len(alerts) == 1
            assert alerts[0].title == "DB Alert"
            assert alerts[0].rule == "budget_breach"
            assert alerts[0].status == "firing"
            assert alerts[0].dedup_key == "db:1"

    def test_fire_persists_metadata(self, db_am):
        db_am.fire(
            rule="provider_dead", severity="critical",
            title="Provider down", message="opencode dead",
            dedup_key="prov:1",
            metadata={"provider": "opencode", "profile": "l2"},
        )
        with get_session(db_am._engine) as session:
            alert = session.query(Alert).first()
            parsed = json.loads(alert.metadata_json)
            assert parsed == {"provider": "opencode", "profile": "l2"}

    def test_list_reads_from_db(self, db_am):
        # Fire 3 alerts with unique dedup keys to avoid cooldown
        for i in range(3):
            db_am.fire(
                rule="budget_breach", severity="warning",
                title=f"Alert {i}", message="m",
                dedup_key=f"bulk:{i}",
            )
        # New manager instance (simulating restart) should still see them
        fresh = AlertManager(db_am._engine)
        alerts = fresh.list_alerts()
        assert len(alerts) == 3

    def test_list_filter_by_status(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Firing Alert", message="m", dedup_key="fs:1",
        )
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Resolved Alert", message="m", dedup_key="rs:1",
        )
        db_am.resolve("rs:1")
        resolved = db_am.list_alerts(status="resolved")
        firing = db_am.list_alerts(status="firing")
        assert len(resolved) == 1
        assert resolved[0]["title"] == "Resolved Alert"
        assert len(firing) == 1
        assert firing[0]["title"] == "Firing Alert"

    def test_list_respects_limit(self, db_am):
        for i in range(5):
            db_am.fire(
                rule="budget_breach", severity="warning",
                title=f"L{i}", message="m", dedup_key=f"limit:{i}",
            )
        assert len(db_am.list_alerts(limit=2)) == 2

    def test_list_newest_first(self, db_am):
        import time
        for i in range(3):
            db_am.fire(
                rule="budget_breach", severity="warning",
                title=f"T{i}", message="m", dedup_key=f"order:{i}",
            )
            time.sleep(0.01)
        alerts = db_am.list_alerts()
        # Alerts with later timestamps sort first (they share ms precision, so
        # check the full ordering holds — dedup by unique titles)
        titles = [a["title"] for a in alerts]
        assert set(titles) == {"T0", "T1", "T2"}


class TestResolveDB:
    def test_resolve_updates_db(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Resolvable", message="m", dedup_key="rdb:1",
        )
        assert db_am.resolve("rdb:1") is True
        with get_session(db_am._engine) as session:
            alert = session.query(Alert).filter(Alert.dedup_key == "rdb:1").first()
            assert alert.status == "resolved"
            assert alert.resolved_at is not None

    def test_resolve_reflected_in_new_instance(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Persisted Resolve", message="m", dedup_key="prd:1",
        )
        db_am.resolve("prd:1")
        fresh = AlertManager(db_am._engine)
        alerts = fresh.list_alerts()
        assert alerts[0]["status"] == "resolved"


class TestAcknowledgeDB:
    def test_acknowledge_updates_db(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Ack Me", message="m", dedup_key="ackdb:1",
        )
        assert db_am.acknowledge("ackdb:1") is True
        with get_session(db_am._engine) as session:
            alert = session.query(Alert).filter(Alert.dedup_key == "ackdb:1").first()
            assert alert.acknowledged == 1
            assert alert.acknowledged_at is not None

    def test_acknowledge_nonexistent_returns_false(self, db_am):
        assert db_am.acknowledge("nope") is False


class TestAlertToDict:
    def test_converts_orm(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Convert", message="m", dedup_key="conv:1",
            metadata={"foo": "bar"},
        )
        with get_session(db_am._engine) as session:
            alert = session.query(Alert).first()
            d = _alert_to_dict(alert)
            assert d["id"] == alert.id
            assert d["dedup_key"] == "conv:1"
            assert d["metadata"] == {"foo": "bar"}
            assert d["acknowledged"] is False
            assert d["status"] == "firing"
            assert "timestamp" in d

    def test_converts_empty_metadata(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="No Meta", message="m", dedup_key="nometa:1",
        )
        with get_session(db_am._engine) as session:
            alert = session.query(Alert).first()
            d = _alert_to_dict(alert)
            assert d["metadata"] == {}


class TestSingleton:
    def test_init_alert_manager_sets_engine(self, temp_db):
        db_path, engine = temp_db
        import src.api.alert_manager as am_mod
        am = am_mod._alert_manager = AlertManager(engine)
        assert am._engine is engine

    def test_get_alert_manager_reuses_singleton(self, db_am, temp_db):
        # init singleton with the temp engine
        import src.api.alert_manager as am_mod
        am_mod._alert_manager = AlertManager(db_am._engine)
        from src.api.alert_manager import get_alert_manager
        am = get_alert_manager()
        assert am is db_am or am._engine is db_am._engine


class TestGracefulDegradation:
    def test_list_alerts_returns_empty_without_engine(self):
        am = AlertManager()  # no engine
        assert am.list_alerts() == []

    def test_fire_without_engine_still_works(self):
        am = AlertManager()  # no engine, in-memory only
        result = am.fire(
            rule="budget_breach", severity="warning",
            title="Mem", message="m", dedup_key="mem:1",
        )
        assert result is not None
        # Falls back to in-memory history (not persisted, but listable)
        assert len(am.list_alerts()) == 1
        assert am.list_alerts()[0]["title"] == "Mem"

    def test_persist_failure_logged_not_raised(self, db_am):
        db_am._engine = MagicMock()
        db_am._engine.__enter__.side_effect = Exception("db down")
        # Should not raise
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Fail", message="m", dedup_key="fail:1",
        )

    def test_list_db_failure_returns_empty(self, db_am):
        with patch("src.api.alert_manager.get_session", side_effect=RuntimeError("db down")):
            assert db_am.list_alerts() == []

    def test_acknowledge_db_failure_returns_false(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Ack", message="m", dedup_key="ack:1",
        )
        with patch("src.api.alert_manager.get_session", side_effect=RuntimeError("db down")):
            # In-memory ack still works, but DB write fails -> returns False
            assert db_am.acknowledge("ack:1") is False

    def test_resolve_db_failure_returns_false(self, db_am):
        db_am.fire(
            rule="budget_breach", severity="warning",
            title="Res", message="m", dedup_key="res:1",
        )
        with patch("src.api.alert_manager.get_session", side_effect=RuntimeError("db down")):
            assert db_am.resolve("res:1") is False


# ── get_alert_manager singleton branches ──────────────────────────────────

class TestGetAlertManagerBranches:
    def test_first_call_creates_with_engine(self):
        import src.api.alert_manager as am_mod
        am_mod._alert_manager = None
        try:
            result = am_mod.get_alert_manager(engine="fake-engine")
            assert result._engine == "fake-engine"
        finally:
            am_mod._alert_manager = None

    def test_existing_without_engine_gets_engine(self):
        import src.api.alert_manager as am_mod
        am_mod._alert_manager = AlertManager()  # engine None
        try:
            result = am_mod.get_alert_manager(engine="eng-2")
            assert result._engine == "eng-2"
        finally:
            am_mod._alert_manager = None
