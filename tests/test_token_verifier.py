"""Tests for token_verifier.py"""
from src.api.token_verifier import TokenVerifier, get_token_verifier

def test_normal_response():
    v = TokenVerifier(threshold=0.5)
    msgs = [{"role": "user", "content": "hello world"}]
    result = v.verify(msgs, {"prompt_tokens": 10, "completion_tokens": 5})
    assert "suspicious" in result

def test_empty_messages():
    v = TokenVerifier()
    result = v.verify([], {"prompt_tokens": 10})
    assert result["estimated_prompt_tokens"] == 0

def test_no_usage():
    v = TokenVerifier()
    result = v.verify([{"role": "user", "content": "hi"}], {})
    assert result["provider_prompt_tokens"] == 0

def test_stats():
    v = TokenVerifier(threshold=0.1)
    msgs = [{"role": "user", "content": "hi"}]
    v.verify(msgs, {"prompt_tokens": 500})
    assert v.stats["checks"] == 1

def test_suspicious_detected():
    v = TokenVerifier(threshold=0.01)
    msgs = [{"role": "user", "content": "hi"}]
    result = v.verify(msgs, {"prompt_tokens": 500})
    assert result["suspicious"] is True

def test_singleton():
    assert get_token_verifier() is get_token_verifier()
