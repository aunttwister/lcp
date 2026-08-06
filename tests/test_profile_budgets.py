"""Tests for profile budget enforcement and profile+key budget interaction."""

import pytest
from unittest.mock import MagicMock, patch
from src.api.models import Budget, ApiKey, get_session
from src.server.handler import LCPHandler


def _make_handler(temp_db):
    _, engine = temp_db
    h = LCPHandler.__new__(LCPHandler)
    h.engine = engine
    return h, engine


def _seed_pb(session, name="P", profile="l2", amount=100.0, spend=0.0,
             action="block", status="active"):
    session.add(Budget(name=name, key_id=None, profile=profile,
                amount=amount, current_spend=spend, period="monthly",
                threshold_pct="50,80", action=action, status=status))
    session.commit()


def _mk_key(session, hh="h", pf="sk-A", nm="Alice"):
    k = ApiKey(key_hash=hh, key_prefix=pf, name=nm,
               allowed_profiles="l2", spend_limit=0, total_spend=0, status="active")
    session.add(k)
    session.commit()
    return k


class TestProfileBlocking:
    def test_exceeded_blocks(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="L2 Cap", profile="l2", spend=100.0)
        assert h._check_budget_block("l2") == "L2 Cap"

    def test_not_exceeded_allows(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="L2 Cap", profile="l2", spend=50.0)
        assert h._check_budget_block("l2") is None

    def test_action_log_never_blocks(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="L2 Warn", profile="l2", spend=100.0, action="log")
        assert h._check_budget_block("l2") is None

    def test_scoped_to_correct_profile(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="L2 Cap", profile="l2", spend=100.0)
        assert h._check_budget_block("career") is None
        assert h._check_budget_block("l1") is None

    def test_matching_profiles_only(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="L2 Cap", profile="l2", spend=100.0)
            _seed_pb(s, name="C Cap", profile="career", amount=50.0, spend=50.0)
        assert h._check_budget_block("l2") == "L2 Cap"
        assert h._check_budget_block("career") == "C Cap"
        assert h._check_budget_block("l1") is None

    def test_global_blocks_all(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="GB", profile=None, amount=10.0, spend=10.0)
        assert h._check_budget_block("l2") == "GB"
        assert h._check_budget_block("career") == "GB"

    def test_exceeded_status_blocks(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, name="Dead", profile="l2", amount=10.0, spend=10.0, status="exceeded")
        assert h._check_budget_block("l2") == "Dead"

class TestProfileSpend:
    def test_increments(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=0)
        h._increment_budget_spend("l2", 12.5)
        with get_session(e) as s:
            b = s.query(Budget).filter(Budget.profile == "l2").first()
            assert b.current_spend == 12.5

    def test_wrong_profile_untouched(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=0)
        h._increment_budget_spend("career", 50.0)
        with get_session(e) as s:
            b = s.query(Budget).filter(Budget.profile == "l2").first()
            assert b.current_spend == 0.0

    def test_global_incremented_any_profile(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile=None, name="GB", spend=0)
        h._increment_budget_spend("l2", 30.0)
        with get_session(e) as s:
            b = s.query(Budget).filter(Budget.profile.is_(None)).first()
            assert b.current_spend == 30.0

    def test_both_incremented(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", name="L2", spend=0)
            _seed_pb(s, profile=None, name="GB", amount=500.0, spend=0)
        h._increment_budget_spend("l2", 25.0)
        with get_session(e) as s:
            l2 = s.query(Budget).filter(Budget.profile == "l2").first()
            gl = s.query(Budget).filter(Budget.profile.is_(None)).first()
            assert l2.current_spend == 25.0
            assert gl.current_spend == 25.0

    def test_threshold_breach(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=40.0)
        breaches = h._increment_budget_spend("l2", 15.0)
        assert len(breaches) == 1
        assert breaches[0]["threshold"] == 50
        assert breaches[0]["spend_pct"] == 55.0

    def test_marks_exceeded(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=95.0)
        h._increment_budget_spend("l2", 10.0)
        with get_session(e) as s:
            b = s.query(Budget).filter(Budget.profile == "l2").first()
            assert b.status == "exceeded"


class TestProfilePlusKey:
    def _seed_both(self, s, kid, pa=100.0, ps=0.0, ka=100.0, ks=0.0,
                   paction="block", kaction="block"):
        s.add_all([
            Budget(name="P-L2", key_id=None, profile="l2",
                   amount=pa, current_spend=ps, period="monthly",
                   threshold_pct="50,80", action=paction, status="active"),
            Budget(name="K-Bob", key_id=kid, profile=None,
                   amount=ka, current_spend=ks, period="total",
                   threshold_pct="80", action=kaction, status="active"),
        ])
        s.commit()

    def test_profile_exceeded_blocks_key_fine(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s)
            kid = k.id
            self._seed_both(s, kid, ps=100.0, ks=10.0)
        assert h._check_budget_block("l2", kid) == "P-L2"

    def test_key_exceeded_blocks_profile_fine(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h2", pf="sk-B", nm="Bob")
            kid = k.id
            self._seed_both(s, kid, ps=10.0, ks=100.0)
        assert h._check_budget_block("l2", kid) == "K-Bob"

    def test_neither_exceeded_allows(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h3", pf="sk-C", nm="Carol")
            kid = k.id
            self._seed_both(s, kid, ps=10.0, ks=10.0)
        assert h._check_budget_block("l2", kid) is None

    def test_both_increment_together(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h4", pf="sk-D", nm="Dave")
            kid = k.id
            self._seed_both(s, kid)
        h._increment_budget_spend("l2", 15.0, key_id=kid)
        with get_session(e) as s:
            pb = s.query(Budget).filter(Budget.profile == "l2", Budget.key_id.is_(None)).first()
            kb = s.query(Budget).filter(Budget.key_id == kid).first()
            assert pb.current_spend == 15.0
            assert kb.current_spend == 15.0

    def test_key_total_spend_synced(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h5", pf="sk-E", nm="Eve")
            kid = k.id
            self._seed_both(s, kid)
        h._increment_budget_spend("l2", 42.0, key_id=kid)
        with get_session(e) as s:
            kr = s.query(ApiKey).filter(ApiKey.id == kid).first()
            assert kr.total_spend == 42.0

    def test_multi_accumulate(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h6", pf="sk-F", nm="Frank")
            kid = k.id
            self._seed_both(s, kid)
        h._increment_budget_spend("l2", 10.0, key_id=kid)
        h._increment_budget_spend("l2", 20.0, key_id=kid)
        h._increment_budget_spend("l2", 30.0, key_id=kid)
        with get_session(e) as s:
            kb = s.query(Budget).filter(Budget.key_id == kid).first()
            kr = s.query(ApiKey).filter(ApiKey.id == kid).first()
            assert kb.current_spend == 60.0
            assert kr.total_spend == 60.0

    def test_no_profile_only_key_increments(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h7", pf="sk-G", nm="Grace")
            kid = k.id
            s.add(Budget(name="K-Only", key_id=kid, profile=None,
                         amount=100.0, current_spend=0, period="total",
                         threshold_pct="80", action="block", status="active"))
            s.commit()
        h._increment_budget_spend("l2", 15.0, key_id=kid)
        with get_session(e) as s:
            kb = s.query(Budget).filter(Budget.key_id == kid).first()
            pb = s.query(Budget).filter(Budget.profile == "l2", Budget.key_id.is_(None)).first()
            assert kb.current_spend == 15.0
            assert pb is None

    def test_no_key_only_profile_increments(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            k = _mk_key(s, hh="h8", pf="sk-H", nm="Heidi")
            kid = k.id
            self._seed_both(s, kid)
        h._increment_budget_spend("l2", 10.0, key_id=None)
        with get_session(e) as s:
            pb = s.query(Budget).filter(Budget.profile == "l2", Budget.key_id.is_(None)).first()
            kb = s.query(Budget).filter(Budget.key_id == kid).first()
            assert pb.current_spend == 10.0
            assert kb.current_spend == 0.0


class TestProfileTrackSpend:
    def test_alert_on_breach(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=40.0)
        with patch("src.server.handler.get_alert_manager") as m:
            m.return_value = MagicMock()
            h._track_budget_spend("l2", 15.0)
            m.return_value.fire.assert_called_once()
            c = m.return_value.fire.call_args[1]
            assert c["rule"] == "budget_breach"
            assert "55.0%" in c["title"]
            assert c["metadata"]["threshold"] == 50

    def test_critical_at_100(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=95.0)
            # Need a 100 threshold to trigger a breach at that level
            b = s.query(Budget).filter(Budget.profile == "l2").first()
            b.threshold_pct = "50,80,100"
            s.commit()
        with patch("src.server.handler.get_alert_manager") as m:
            m.return_value = MagicMock()
            h._track_budget_spend("l2", 10.0)
            assert m.return_value.fire.call_args[1]["severity"] == "critical"

    def test_no_alert_without_breach(self, temp_db):
        h, e = _make_handler(temp_db)
        with get_session(e) as s:
            _seed_pb(s, profile="l2", spend=10.0)
        with patch("src.server.handler.get_alert_manager") as m:
            m.return_value = MagicMock()
            h._track_budget_spend("l2", 5.0)
            m.return_value.fire.assert_not_called()
