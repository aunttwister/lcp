"""Server factory — creates and configures the ThreadingHTTPServer."""

from http.server import ThreadingHTTPServer

from ..api.logging_config import get_logger
from ..api.key_manager import init_key_manager
from ..api.credential_store import init_credential_store
from .handler import LCPHandler

logger = get_logger("lcp.server")


def create_server(config, engine, port=8734):
    """Create and configure the HTTP server."""

    class ConfiguredHandler(LCPHandler):
        pass

    ConfiguredHandler.config = config
    ConfiguredHandler.engine = engine

    # Initialize key manager with engine
    init_key_manager(engine, "data")
    # Initialize credential store (encrypted provider API keys)
    init_credential_store(engine, "data")

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
