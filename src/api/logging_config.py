"""Structured JSON logging via structlog. Outputs to stdout for docker logs."""

import logging
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON to stdout."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also suppress noisy library loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("http.server").setLevel(logging.WARNING)


def get_logger(name: str = "lcp"):
    """Get a bound structlog logger."""
    return structlog.get_logger(name)
