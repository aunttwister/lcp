"""OpenCode console-API paths: model discovery + money/range helpers.

Regression context (2026-09-26): OpenCode serves **no** ``/models`` route on its
inference base, so LCP's generic discovery (``{api_base}/models``) returned a
bare HTTP 404 for this provider and "Discover Models" was unusable.  Discovery
now goes through a plugin override that reads the console config ``whitelist``.
These tests pin that behaviour, the payload parsing, and the guarantee that
discovery never touches the inference base.
"""

from __future__ import annotations

import pytest

from api.cost_plugins import opencode_api as oa
from api.cost_plugins.opencode import OpenCodeCostPlugin

CONFIG_PAYLOAD = {
    "config": {
        "enterprise": {"url": "https://opencode.ai/console"},
        "provider": {
            "opencode": {
                "name": "DefaultPavle / OpenCode",
                "api": "https://opencode.ai/inference/openai/v1",
                "whitelist": [
                    "deepseek-v4-flash",
                    "deepseek-v4-pro",
                    "glm-5.3-flash",
                    "deepseek-v4-flash",  # duplicate on purpose
                ],
            }
        },
    }
}


@pytest.fixture()
def plugin():
    return OpenCodeCostPlugin()


def _with_key(monkeypatch, instance, token="oc_sk_test"):
    """Give *instance* a console token without touching the credential store."""
    monkeypatch.setattr(instance, "_token", lambda **kw: token, raising=False)


# ── console_model_ids ──────────────────────────────────────────────────────

def test_console_model_ids_reads_nested_whitelist(monkeypatch):
    monkeypatch.setattr(oa, "fetch_console_config", lambda token: CONFIG_PAYLOAD)
    ids = oa.console_model_ids("k")
    assert ids == ["deepseek-v4-flash", "deepseek-v4-pro", "glm-5.3-flash"]


def test_console_model_ids_dedupes_and_sorts(monkeypatch):
    monkeypatch.setattr(oa, "fetch_console_config", lambda token: CONFIG_PAYLOAD)
    ids = oa.console_model_ids("k")
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_console_model_ids_tolerates_unfamiliar_shape(monkeypatch):
    for payload in (None, {}, {"config": {}}, {"config": {"provider": "x"}},
                    {"config": {"provider": {"other": {}}}},
                    {"config": {"provider": {"opencode": {"whitelist": "nope"}}}}):
        monkeypatch.setattr(oa, "fetch_console_config", lambda token, p=payload: p)
        assert oa.console_model_ids("k") == []


# ── money + range helpers ──────────────────────────────────────────────────

def test_micro_cents_to_usd_matches_vendor_units():
    # 4097780847 micro-cents == $40.97780847 (the live 7d total on 2026-09-26)
    assert oa.micro_cents_to_usd("4097780847") == 40.97780847
    assert oa.micro_cents_to_usd(445) == 0.00000445
    assert oa.micro_cents_to_usd(None) is None
    assert oa.micro_cents_to_usd("not-a-number") is None


@pytest.mark.parametrize("value", ["24h", "7d", "30d"])
def test_check_range_accepts_the_three_console_ranges(value):
    assert oa._check_range(value) == value


def test_check_range_defaults_to_7d_and_rejects_junk():
    assert oa._check_range("") == "7d"
    assert oa._check_range(None) == "7d"
    with pytest.raises(ValueError):
        oa._check_range("90d")


# ── discover_models ────────────────────────────────────────────────────────

def test_discover_models_returns_console_catalog(plugin, monkeypatch):
    _with_key(monkeypatch, plugin)
    monkeypatch.setattr(oa, "console_model_ids", lambda token: ["a", "b"])
    assert plugin.discover_models("https://opencode.ai/inference/openai/v1") == [
        {"id": "a"}, {"id": "b"},
    ]


def test_discover_models_without_key_returns_none(plugin, monkeypatch):
    _with_key(monkeypatch, plugin, token="")
    called = []
    monkeypatch.setattr(oa, "console_model_ids",
                        lambda token: called.append(token) or ["a"])
    assert plugin.discover_models("https://x/v1") is None
    assert called == [], "must not call the API without a token"


def test_discover_models_returns_none_on_api_error(plugin, monkeypatch):
    _with_key(monkeypatch, plugin)

    def boom(token):
        raise oa.ConsoleApiError(403, "https://console.opencode.ai/api/config",
                                 "Forbidden")

    monkeypatch.setattr(oa, "console_model_ids", boom)
    assert plugin.discover_models("https://x/v1") is None


def test_discover_models_returns_none_on_empty_catalog(plugin, monkeypatch):
    _with_key(monkeypatch, plugin)
    monkeypatch.setattr(oa, "console_model_ids", lambda token: [])
    assert plugin.discover_models("https://x/v1") is None


def test_discover_models_never_requests_the_inference_base(plugin, monkeypatch):
    """The 404 came from GET {api_base}/models — discovery must not do that."""
    _with_key(monkeypatch, plugin)
    monkeypatch.setattr(oa, "console_model_ids", lambda token: ["a"])

    def forbidden(*args, **kwargs):  # pragma: no cover - only fires on regression
        raise AssertionError("discovery must not perform its own HTTP request")

    monkeypatch.setattr(oa, "urlopen", forbidden)
    assert plugin.discover_models("https://opencode.ai/inference/openai/v1") == [{"id": "a"}]


def test_discover_models_survives_an_unexpected_exception(plugin, monkeypatch):
    _with_key(monkeypatch, plugin)

    def boom(token):
        raise RuntimeError("network down")

    monkeypatch.setattr(oa, "console_model_ids", boom)
    assert plugin.discover_models("https://x/v1") is None


# ── preset ─────────────────────────────────────────────────────────────────

def test_preset_points_at_the_current_inference_base(plugin):
    """Regression: the preset still advertised the retired /zen/go/v1 base."""
    assert plugin.preset["api_base"] == "https://opencode.ai/inference/openai/v1"


# ── usage fetchers build the documented routes ─────────────────────────────

@pytest.mark.parametrize("fn,expected", [
    (lambda t: oa.fetch_usage_summary(t, "24h"), "/usage/summary?range=24h"),
    (lambda t: oa.fetch_usage_cost_by_day(t, "7d"), "/usage/cost-by-day?range=7d"),
    (lambda t: oa.fetch_usage_models(t, "30d"), "/usage/models?range=30d"),
    (lambda t: oa.fetch_account_credits(t), "/billing/account"),
])
def test_route_construction(monkeypatch, fn, expected):
    seen = {}

    def fake_get(path, token, timeout=15):
        seen["path"] = path
        return {}

    monkeypatch.setattr(oa, "_console_get", fake_get)
    fn("tok")
    assert seen["path"] == expected


def test_console_get_classifies_403_as_permission_not_auth(monkeypatch):
    """A 403 must carry its own status so callers don't call it a dead key."""
    import io
    from urllib.error import HTTPError

    def fake_urlopen(req, timeout=15):
        raise HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"_tag":"Forbidden"}'))

    monkeypatch.setattr(oa, "urlopen", fake_urlopen)
    with pytest.raises(oa.ConsoleApiError) as exc:
        oa.fetch_account_credits("tok")
    assert exc.value.status == 403
    assert exc.value.tag == "Forbidden"
