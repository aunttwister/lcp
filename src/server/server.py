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

    # Key manager + credential store are owned by the component runtime when
    # it is active (Phase D) — the runtime's components inject the engine and
    # the resolved data_dir at setup. Direct/test callers without a bound
    # runtime fall back to the legacy force-init so the server still works
    # standalone.
    from ..api.key_manager import is_runtime_bound as km_bound
    from ..api.credential_store import is_runtime_bound as cs_bound
    if not km_bound():
        init_key_manager(engine, "data")
    if not cs_bound():
        init_credential_store(engine, "data")

    server = ThreadingHTTPServer(("0.0.0.0", port), ConfiguredHandler)
    logger.info("server_created", port=port, profiles=list(config.profiles.keys()))
    return server
