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

    # Initialize the dynamic router (benchmark-driven model selection) with
    # the config's enabled/cost_bias. Disabled when not configured.
    from .api.router import init_router
    try:
        dr = config.dynamic_routing or {}
        if not isinstance(dr, dict):
            dr = {}
    except Exception:  # noqa: BLE001 — mocked/partial configs fall back to disabled
        dr = {}
    init_router(db_path, enabled=bool(dr.get("enabled", False)),
                cost_bias=float(dr.get("cost_bias", 0.15)))
    t0 = _startup_step("router_init", t0)
    logger.info("dynamic_router_initialized", enabled=bool(dr.get("enabled", False)))

    # Recover benchmark runs left queued/running by a previous process.
    try:
        from .api.benchmark import recover_stale_runs
        recovered = recover_stale_runs(engine)
        if recovered:
            logger.info("benchmark_recovered", count=recovered)
    except Exception as exc:  # noqa: BLE001 — never block boot
        logger.warning("benchmark_recovery_failed", error=str(exc))

    # Initialize cost tracking plugins (imports auto-register via __init__.py)
    from .api.cost_plugins import init_plugins
    init_plugins(engine=engine)
    t0 = _startup_step("cost_plugins_init", t0)
    logger.info("cost_plugins_initialized")

    # Initialize the cost-plugin cache + background refresher. The refresher
    # owns ALL live scraping; the HTTP endpoints read the cache only.
    from .api.cost_cache import init_cost_cache, init_refresher, init_settings
    settings = init_settings(engine)
    cache = init_cost_cache(engine)
    refresher = init_refresher(cache, settings)
    refresher.start()
    t0 = _startup_step("cost_cache_init", t0)
    logger.info("cost_cache_initialized", ttl_minutes=settings.get_ttl_minutes())

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
    finally:
        from .api.cost_cache import stop_refresher
        stop_refresher()


if __name__ == "__main__":
    main()
