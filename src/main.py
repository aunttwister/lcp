"""LCP gateway — entry point. Loads env, boots server."""

import os
import time
from pathlib import Path

from .api.config import init_config
from .api.logging_config import setup_logging, get_logger
from .api.models import get_engine, Base
from .api.circuit_breaker import get_circuit_breaker
from .api.key_manager import get_key_manager
from .api.alert_manager import init_alert_manager
from .server import create_server

logger = get_logger("lcp.main")


def _startup_step(name: str, t0: float) -> float:
    """Log a startup step and its duration. Returns the new timestamp."""
    t1 = time.monotonic()
    logger.info("startup_step", step=name, elapsed_ms=round((t1 - t0) * 1000, 1))
    return t1


def main():
    """Entry point — start the HTTP server."""
    _boot = time.monotonic()
    t0 = _boot

    config = init_config()
    t0 = _startup_step("config_load", t0)

    cfg = config.server

    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    logger.info("startup_begin", version=__import__("src").__version__,
                port=int(os.environ.get("LISTEN_PORT", str(cfg.get("port", 8734)))),
                config=os.environ.get("LCP_CONFIG", "") or "default",
                log_level=os.environ.get("LOG_LEVEL", "INFO"))

    get_circuit_breaker(config)
    t0 = _startup_step("circuit_breaker_init", t0)
    logger.info("circuit_breaker_initialized")

    # ── Database (must exist before plugins that query it) ──────────────
    db_path = os.environ.get("COST_DB", config.database.get("path", "/app/data/costs.db"))
    assert db_path is not None
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    t0 = _startup_step("db_init", t0)
    logger.info("db_initialized", path=db_path)

    # Initialize cost tracking plugins (imports auto-register via __init__.py)
    from .api.cost_plugins import init_plugins
    init_plugins(engine=engine)
    t0 = _startup_step("cost_plugins_init", t0)
    logger.info("cost_plugins_initialized")

    # Initialize alert manager with DB engine for persistence
    init_alert_manager(engine)
    t0 = _startup_step("alert_manager_init", t0)
    logger.info("alert_manager_initialized")

    # Attach engine to circuit breaker so failover events persist to DB
    get_circuit_breaker().attach_engine(engine)
    t0 = _startup_step("circuit_breaker_engine", t0)
    logger.info("circuit_breaker_engine_attached")

    data_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "data"

    port = int(os.environ.get("LISTEN_PORT", str(cfg.get("port", 8734))))
    server = create_server(config, engine, port)
    t0 = _startup_step("server_create", t0)

    profiles = list(config.profiles.keys()) if hasattr(config, "profiles") else []
    providers = list(config.providers.keys()) if hasattr(config, "providers") else []
    logger.info("startup_complete", port=port, profiles=profiles, providers=providers,
                total_boot_ms=round((time.monotonic() - _boot) * 1000, 1))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
        server.shutdown()


if __name__ == "__main__":
    main()
