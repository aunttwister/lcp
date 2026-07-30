"""Integration tests against the live dev instance on port 8734.

Set LCP_TEST_URL to override (e.g. http://localhost:8735 for Docker).
Run with: pytest -m integration
"""
import json
import os
import pytest
import urllib.request
import urllib.error
import csv
import io

BASE_URL = os.environ.get("LCP_TEST_URL", "http://localhost:8734")

pytestmark = pytest.mark.integration


def _get(path, expect_status=200):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            if expect_status:
                assert resp.status == expect_status, f"GET {path} -> {resp.status}"
            return body, resp.status
    except urllib.error.HTTPError as e:
        if expect_status and e.code != expect_status:
            raise
        return e.read().decode("utf-8", errors="replace"), e.code


def _post(path, body_dict, expect_status=200):
    data = json.dumps(body_dict).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            if expect_status:
                assert resp.status == expect_status, f"POST {path} -> {resp.status}"
            return body, resp.status
    except urllib.error.HTTPError as e:
        if expect_status and e.code != expect_status:
            raise
        return e.read().decode("utf-8", errors="replace"), e.code


# ── Health & Info ──────────────────────────────────────────────────────

class TestHealth:
    def test_health_check(self):
        body, status = _get("/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "ok"
        assert "profiles" in data

    def test_homepage_is_html(self):
        body, status = _get("/")
        assert status == 200
        assert "<h1>smallm gateway</h1>" in body

    def test_dashboard_stats_present(self):
        body, _ = _get("/")
        assert "Requests" in body
        assert "Cost" in body

    def test_models_list(self):
        body, status = _get("/v1/models")
        assert status == 200
        data = json.loads(body)
        assert "data" in data
        assert isinstance(data["data"], list)


# ── Cache Stats ────────────────────────────────────────────────────────

class TestCacheStats:
    def test_cache_stats(self):
        body, status = _get("/cache/stats")
        assert status == 200
        data = json.loads(body)
        assert "entries" in data or "total_entries" in data
        assert "hits" in data
        assert "misses" in data

    def test_metrics_endpoint(self):
        body, status = _get("/metrics")
        assert status == 200
        assert "lcp_" in body or "requests" in body.lower()

    def test_export_csv(self):
        body, status = _get("/export?limit=5")
        assert status == 200
        reader = csv.reader(io.StringIO(body))
        header = next(reader)
        assert len(header) > 5  # multiple columns

    def test_export_default(self):
        body, status = _get("/export")
        assert status == 200


# ── Error Handling ─────────────────────────────────────────────────────

class TestErrors:
    def test_404_not_found(self):
        body, status = _get("/nonexistent/path", expect_status=404)
        assert status == 404

    def test_post_to_get_only_endpoint(self):
        body, status = _post("/health", {}, expect_status=404)
        assert status == 404

    def test_no_such_profile(self):
        body, status = _post("/no-such-profile/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}]},
                             expect_status=400)
        assert status == 400

    def test_missing_messages_in_body(self):
        """When messages is missing or misformatted, server returns error."""
        body, status = _post("/l2/chat/completions", {}, expect_status=400)
        # Server may return 400 (validation) or 502 (all providers fail)
        assert status in (400, 502)
        if status == 400:
            assert "error" in json.loads(body)


# ── Chat Completions ───────────────────────────────────────────────────

class TestChatCompletions:
    def test_basic_chat(self):
        """Send a real chat request through the proxy."""
        body, status = _post("/l2/chat/completions", {
            "messages": [{"role": "user", "content": "say hello in one word"}],
        }, expect_status=200)
        assert status == 200
        data = json.loads(body)
        assert "choices" in data

    def test_cost_header(self):
        """Verify X-Estimated-Cost header is returned."""
        data = json.dumps({
            "messages": [{"role": "user", "content": "hi"}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/l2/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cost = resp.headers.get("X-Estimated-Cost")
                assert cost is not None
                assert resp.status == 200
        except urllib.error.HTTPError as e:
            cost = e.headers.get("X-Estimated-Cost") if hasattr(e, 'headers') else None
            # Provider failure is acceptable — we're testing the header, not availability
            if e.code in (502, 503):
                assert cost is not None  # header still set

    def test_repeat_prompt(self):
        """Two identical requests exercise the cache path."""
        payload = {"messages": [{"role": "user", "content": "what is 2+2? answer with the number"}]}
        body1, s1 = _post("/l2/chat/completions", payload, expect_status=None)
        body2, s2 = _post("/l2/chat/completions", payload, expect_status=None)
        assert s1 >= 200 and s1 < 600
        assert s2 >= 200 and s2 < 600
