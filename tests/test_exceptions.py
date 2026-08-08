"""Tests for exceptions.py"""
import pytest
import sys
from src.api.exceptions import (
    LCPError, ConfigError, AuthError, ForbiddenError, CreditExhaustedError,
    ProviderError, ProviderTimeoutError, ProviderRateLimitError,
    ProviderAuthError, ProviderCreditsError, ProviderBadRequestError,
    ProviderInternalError, AllProvidersFailedError, ToolBlockedError,
)

def test_base_is_lcp_error():
    e = LCPError("base")
    assert isinstance(e, Exception)

def test_config_errors_are_lcp():
    e = ConfigError("bad config")
    assert isinstance(e, LCPError)
    assert isinstance(e, ConfigError)

def test_provider_errors_are_lcp():
    e = ProviderError("provider issue")
    assert isinstance(e, LCPError)

def test_auth_error_is_provider():
    e = ProviderAuthError("unauthorized")
    assert isinstance(e, ProviderError)
    assert isinstance(e, LCPError)


def test_credits_error_is_provider():
    e = ProviderCreditsError("out of credits")
    assert isinstance(e, ProviderError)
    assert isinstance(e, LCPError)

def test_rate_limit_is_provider():
    e = ProviderRateLimitError("rate limited")
    assert isinstance(e, ProviderError)

def test_timeout_is_provider():
    e = ProviderTimeoutError("timeout")
    assert isinstance(e, ProviderError)

def test_base_message():
    e = LCPError("something broke")
    assert "something broke" in str(e)

def test_tool_blocked():
    e = ToolBlockedError("execute_bash")
    assert "execute_bash" in str(e)

def test_all_providers_failed():
    e = AllProvidersFailedError("all 3 providers down")
    assert "3" in str(e)

def test_credit_exhausted_is_lcp():
    e = CreditExhaustedError()
    assert isinstance(e, LCPError)

def test_auth_error_message():
    e = AuthError("bad credentials")
    assert "bad credentials" in str(e)

# ── New: error codes, status codes, structured payloads ────────────────────

ALL_EXCEPTIONS = [
    LCPError, ConfigError, AuthError, ForbiddenError, CreditExhaustedError,
    ProviderError, ProviderTimeoutError, ProviderRateLimitError,
    ProviderAuthError, ProviderBadRequestError, ProviderInternalError,
    AllProvidersFailedError, ToolBlockedError,
]


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_every_exception_has_code_and_status(exc_cls):
    """Every LCPError subclass exposes a stable code and HTTP status."""
    e = exc_cls("boom")
    assert isinstance(e.code, str)
    assert e.code.startswith("LCP-")
    assert e.code[4:].isdigit()
    assert isinstance(e.status_code, int)
    assert 400 <= e.status_code <= 599


@pytest.mark.parametrize("exc_cls,expected", [
    (AuthError, "LCP-1001"),
    (CreditExhaustedError, "LCP-1002"),
    (ForbiddenError, "LCP-1003"),
    (ProviderTimeoutError, "LCP-2001"),
    (ProviderRateLimitError, "LCP-2002"),
    (ProviderAuthError, "LCP-2003"),
    (ProviderCreditsError, "LCP-2006"),
    (ProviderBadRequestError, "LCP-2004"),
    (ProviderInternalError, "LCP-2005"),
    (AllProvidersFailedError, "LCP-3001"),
    (ToolBlockedError, "LCP-4002"),
])
def test_known_error_codes(exc_cls, expected):
    assert exc_cls("x").code == expected


@pytest.mark.parametrize("exc_cls,expected", [
    (AuthError, 401),
    (CreditExhaustedError, 429),
    (ForbiddenError, 403),
    (ProviderTimeoutError, 504),
    (ProviderRateLimitError, 429),
    (ProviderBadRequestError, 400),
    (ProviderCreditsError, 402),
    (ProviderInternalError, 502),
    (AllProvidersFailedError, 502),
    (ToolBlockedError, 403),
])
def test_known_status_codes(exc_cls, expected):
    assert exc_cls("x").status_code == expected


def test_provider_internal_error_inherits_provider():
    e = ProviderInternalError("503 from provider")
    assert isinstance(e, ProviderError)
    assert isinstance(e, LCPError)


def test_forbidden_error_inherits_lcp():
    e = ForbiddenError("no access")
    assert isinstance(e, LCPError)
    assert e.status_code == 403


def test_to_dict_returns_code_and_message():
    e = ProviderBadRequestError("model does not exist")
    payload = e.to_dict()
    assert payload == {"code": "LCP-2004", "message": "model does not exist"}


def test_to_dict_omits_details_when_empty():
    e = AuthError("bad key")
    assert "details" not in e.to_dict()


def test_to_dict_includes_details():
    e = AuthError("bad key", details={"key_id": 42})
    payload = e.to_dict()
    assert payload["details"] == {"key_id": 42}


def test_to_dict_uses_client_message_override():
    """AllProvidersFailedError must not leak provider internals to clients."""
    e = AllProvidersFailedError("All providers failed for l2: secretco: HTTP 500: sk-abc123...")
    payload = e.to_dict()
    assert payload["code"] == "LCP-3001"
    assert "secretco" not in payload["message"]
    assert "sk-abc123" not in payload["message"]
    assert "providers failed" in payload["message"]


def test_plain_str_still_has_full_detail():
    """The raw exception string keeps full detail for server-side logging."""
    e = AllProvidersFailedError("All providers failed for l2: secretco: HTTP 500")
    assert "secretco" in str(e)
    assert "secretco" not in e.to_dict()["message"]
