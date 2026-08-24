"""Tests for key_manager.py — API key CRUD, rotation, validation, spend tracking."""

import pytest
from src.api.models import get_engine, Base, ApiKey
from src.api.key_manager import init_key_manager


@pytest.fixture
def km(temp_dir):
    """Create a KeyManager with in-memory SQLite."""
    db_path = str(temp_dir / "keys_test.db")
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return init_key_manager(engine, str(temp_dir))


class TestCreateKey:
    def test_creates_with_defaults(self, km):
        result = km.create_key(name="Test Key")
        assert result["ok"] is True
        assert result["key"].startswith("lcp_")
        assert len(result["key"]) == 52  # lcp_ + 48 hex chars
        assert result["name"] == "Test Key"
        assert result["allowed_profiles"] is None
        assert result["spend_limit"] == 0.0

    def test_creates_with_profiles(self, km):
        result = km.create_key(name="Dev Key", allowed_profiles="l2,l1")
        assert result["allowed_profiles"] == "l2,l1"

    def test_creates_with_spend_limit(self, km):
        result = km.create_key(name="Limited Key", spend_limit=50.0)
        assert result["spend_limit"] == 50.0

    def test_key_prefix_matches(self, km):
        result = km.create_key(name="Prefix Test")
        assert result["key_prefix"] == result["key"][:12]

    def test_each_key_is_unique(self, km):
        k1 = km.create_key(name="Key A")
        k2 = km.create_key(name="Key B")
        assert k1["key"] != k2["key"]
        assert k1["id"] != k2["id"]


class TestListKeys:
    def test_empty_list(self, km):
        assert km.list_keys() == []

    def test_lists_created_keys(self, km):
        km.create_key(name="K1")
        km.create_key(name="K2")
        keys = km.list_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"K1", "K2"}

    def test_does_not_expose_hash(self, km):
        km.create_key(name="Secret")
        keys = km.list_keys()
        assert "key_hash" not in keys[0]


class TestGetKey:
    def test_returns_key_by_id(self, km):
        created = km.create_key(name="Find Me")
        found = km.get_key(created["id"])
        assert found["name"] == "Find Me"

    def test_returns_none_for_missing(self, km):
        assert km.get_key(999) is None


class TestValidateKey:
    def test_validates_correct_key(self, km):
        created = km.create_key(name="Valid Key")
        result = km.validate_key(created["key"])
        assert result is not None
        assert result["name"] == "Valid Key"

    def test_rejects_wrong_key(self, km):
        km.create_key(name="Real")
        assert km.validate_key("lcp_badkey123") is None

    def test_rejects_revoked_key(self, km):
        created = km.create_key(name="To Revoke")
        km.revoke_key(created["id"])
        assert km.validate_key(created["key"]) is None

    def test_rejects_expired_key(self, km):
        from datetime import datetime, timezone, timedelta
        past = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        created = km.create_key(name="Expired", expires_at=past)
        assert km.validate_key(created["key"]) is None

    def test_updates_last_used(self, km):
        created = km.create_key(name="Used")
        km.validate_key(created["key"])
        key = km.get_key(created["id"])
        assert key["last_used_at"] is not None

    def test_returns_profile_access(self, km):
        created = km.create_key(name="Profiled", allowed_profiles="l2")
        result = km.validate_key(created["key"])
        assert result["allowed_profiles"] == "l2"
        assert result["spend_limit"] == 0.0


class TestRotateKey:
    def test_rotates_and_revokes_old(self, km):
        created = km.create_key(name="Rotate Me")
        rotated = km.rotate_key(created["id"])

        assert rotated["ok"] is True
        assert rotated["old_id"] == created["id"]
        assert rotated["id"] != created["id"]
        assert rotated["key"].startswith("lcp_")

        # Old key should be revoked
        old = km.get_key(created["id"])
        assert old["status"] == "revoked"
        assert old["revoked_at"] is not None

        # New key should be active
        new = km.get_key(rotated["id"])
        assert new["status"] == "active"

    def test_rotate_preserves_settings(self, km):
        created = km.create_key(name="Preserve", allowed_profiles="l2,l1", spend_limit=100)
        rotated = km.rotate_key(created["id"])

        new = km.get_key(rotated["id"])
        assert new["name"] == "Preserve"
        assert new["allowed_profiles"] == "l2,l1"

    def test_rotate_nonexistent(self, km):
        assert km.rotate_key(999) is None


class TestRevokeKey:
    def test_revokes_active_key(self, km):
        created = km.create_key(name="Active")
        assert km.revoke_key(created["id"]) is True
        key = km.get_key(created["id"])
        assert key["status"] == "revoked"
        assert key["revoked_at"] is not None

    def test_revoke_twice_returns_true(self, km):
        created = km.create_key(name="Double")
        km.revoke_key(created["id"])
        assert km.revoke_key(created["id"]) is True  # idempotent

    def test_revoke_nonexistent(self, km):
        assert km.revoke_key(999) is False


class TestLegacyMigration:
    def test_migrates_json_keys(self, temp_dir):
        """Verify legacy JSON keys are migrated to DB on init."""
        import json
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        legacy = {
            "keys": [{
                "id": "abc12345",
                "profile": "l2",
                "label": "Legacy Key",
                "hash": "abc123hash",
                "created": "2024-01-01T00:00:00",
                "last_used": None,
            }]
        }
        with open(data_dir / "api_keys.json", "w") as f:
            json.dump(legacy, f)

        db_path = str(temp_dir / "migrate_test.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        km2 = init_key_manager(engine, str(data_dir))

        keys = km2.list_keys()
        assert len(keys) >= 1
        names = [k["name"] for k in keys]
        assert "Legacy Key" in names

    def test_migrate_empty_keys_list(self, temp_dir):
        """Empty keys list in legacy JSON skips migration loop."""
        import json
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        with open(data_dir / "api_keys.json", "w") as f:
            json.dump({"keys": []}, f)

        db_path = str(temp_dir / "empty_migrate.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        km2 = init_key_manager(engine, str(data_dir))
        assert km2.list_keys() == []

    def test_migrate_json_decode_error(self, temp_dir):
        """Invalid JSON in legacy file triggers exception handler."""
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        with open(data_dir / "api_keys.json", "w") as f:
            f.write("not valid json")

        db_path = str(temp_dir / "bad_migrate.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)
        km2 = init_key_manager(engine, str(data_dir))
        # Falls through with no keys migrated
        assert km2.list_keys() == []


class TestValidateKeyEdgeCases:
    def test_invalid_expiry_format(self, km):
        """Invalid expires_at format is caught by ValueError handler."""
        created = km.create_key(name="Bad Expiry")
        from src.api.models import get_session
        with get_session(km._engine) as session:
            key = session.query(ApiKey).filter(ApiKey.id == created["id"]).first()
            key.expires_at = "not-a-date"
            session.commit()
        # ValueError is caught, key is treated as valid
        result = km.validate_key(created["key"])
        assert result is not None
        assert result["name"] == "Bad Expiry"


class TestKeyManagerSingleton:
    def test_get_key_manager_creates_when_none(self, temp_dir):
        """get_key_manager creates a new instance when _key_manager is None."""
        import src.api.key_manager as km_mod
        km_mod._key_manager = None  # reset global

        db_path = str(temp_dir / "get_km.db")
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

        km = km_mod.get_key_manager(engine, str(temp_dir))
        assert km is not None
        assert km.list_keys() == []
