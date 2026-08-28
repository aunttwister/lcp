"""Server factory — creates and configures the ThreadingHTTPServer."""

from http.server import ThreadingHTTPServer

from ..api.logging_config import get_logger
from .handler import LCPHandler

logger = get_logger("lcp.server")


def create_server(config, engine, port=8734):
    """Create and configure the HTTP server.

    The key manager and credential store are owned by the component runtime
    (Phase D): ``main.py`` builds the runtime before calling this, so the
    handler's ``get_*()`` facades resolve the runtime-owned instances. No
    legacy force-init here.
    """

    class ConfiguredHandler(LCPHandler):
        pass

    ConfiguredHandler.config = config
    ConfiguredHandler.engine = engine

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
