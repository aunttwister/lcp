"""LLM Control Plane — entry point. Loads env, boots server."""

import os
from pathlib import Path

from .api.config import init_config
from .api.logging_config import setup_logging
from .api.models import get_engine, Base
from .api.circuit_breaker import get_circuit_breaker
from .api.key_manager import get_key_manager
from .server import create_server

# ── Load static template assets ──────────────────────────────────────────
_DASHBOARD_CSS: str = ""
try:
    _templates_dir = __import__("pathlib").Path(__file__).parent / "ui" / "templates"
    _DASHBOARD_CSS = (_templates_dir / "dashboard.css").read_text()
except Exception:
    pass


def main():
    """Entry point — start the HTTP server."""
    config = init_config()
    cfg = config.server

    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    get_circuit_breaker(config)

    db_path = os.environ.get("COST_DB", config.database.get("path", "/app/data/costs.db"))
    assert db_path is not None
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    data_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "data"
    get_key_manager(data_dir)

    port = int(os.environ.get("LISTEN_PORT", str(cfg.get("port", 8734))))
    server = create_server(config, engine, port)

    version = __import__("src").__version__
    print(f"LLM Control Plane v{version} listening on :{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
