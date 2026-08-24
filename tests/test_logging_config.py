"""Tests for logging_config.py"""
from src.api.logging_config import setup_logging, get_logger

def test_setup_info():
    setup_logging("INFO")

def test_setup_debug():
    setup_logging("DEBUG")

def test_logger_bind():
    logger = get_logger("test.module")
    assert hasattr(logger, "bind")

def test_logger_info():
    logger = get_logger("test.info")
    logger.info("test message", key="value")
