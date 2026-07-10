"""Tests for exceptions.py"""
import pytest
import sys
sys.path.insert(0, "/opt/lcp")
from src.exceptions import (
    LCPError, ConfigError, AuthError, CreditExhaustedError,
    ProviderError, ProviderTimeoutError, ProviderRateLimitError,
    ProviderAuthError, AllProvidersFailedError, ToolBlockedError,
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
