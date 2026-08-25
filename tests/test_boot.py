"""Boot-path smoke tests for the entry point (src/main.py) and the server
factory (src/server/server.py).

These were the two least-covered modules (0% and 43%): they are bootstrap
code that only runs when the process starts, so unit tests mock the heavy
dependencies and verify the wiring.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def boot_config():
    """A minimal config object that main() can consume."""
    cfg = MagicMock()
    cfg.server = {"port": 8734}
    cfg.database = {"path": "/tmp/test.db"}
    cfg.profiles = {"l2": {"chain": []}, "l1": {"chain": []}}
    cfg.providers = {}
    cfg.dynamic_routing = {"enabled": False, "cost_bias": 0.15}
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# src.main.main()
# ═══════════════════════════════════════════════════════════════════════

class TestMain:
    def test_main_starts_server(self, boot_config):
        import src.main
        server = MagicMock()
        engine = MagicMock()
        settings = MagicMock()

        with patch.object(src.main, "get_engine", return_value=engine) as mock_engine, \
             patch.object(src.main, "setup_logging") as mock_log, \
             patch.object(src.main, "init_alert_manager") as mock_alert, \
             patch.object(src.main, "create_server", return_value=server) as mock_server, \
             patch("src.api.cost_cache.init_settings", return_value=settings) as mock_settings, \
             patch("src.api.config.init_config", return_value=boot_config) as mock_init_cfg, \
             patch("src.api.cost_plugins.init_plugins") as mock_plugins:

            src.main.main()

        mock_log.assert_called_once()
        mock_engine.assert_called_once_with("/app/data/costs.db")
        mock_plugins.assert_called_once_with(engine=engine)
        mock_alert.assert_called_once_with(engine)
        mock_settings.assert_called_once_with(engine)
        mock_init_cfg.assert_called_once_with(store=settings)
        mock_server.assert_called_once()
        server.serve_forever.assert_called_once()

    def test_main_uses_env_overrides(self, boot_config):
        import src.main
        server = MagicMock()
        engine = MagicMock()

        with patch.object(src.main, "get_engine", return_value=engine) as mock_engine, \
             patch.object(src.main, "init_alert_manager"), \
             patch.object(src.main, "create_server", return_value=server), \
             patch("src.api.cost_cache.init_settings", return_value=MagicMock()), \
             patch("src.api.config.init_config", return_value=boot_config), \
             patch("src.api.cost_plugins.init_plugins"), \
             patch.dict("os.environ", {"COST_DB": "/env/costs.db", "LISTEN_PORT": "9000"}, clear=False):

            src.main.main()

        mock_engine.assert_called_once_with("/env/costs.db")
        assert server.serve_forever is not None

    def test_main_shuts_down_on_keyboard_interrupt(self, boot_config):
        import src.main
        server = MagicMock()
        server.serve_forever.side_effect = KeyboardInterrupt()
        engine = MagicMock()

        with patch.object(src.main, "get_engine", return_value=engine), \
             patch.object(src.main, "setup_logging"), \
             patch.object(src.main, "init_alert_manager"), \
             patch.object(src.main, "create_server", return_value=server), \
             patch("src.api.cost_cache.init_settings", return_value=MagicMock()), \
             patch("src.api.config.init_config", return_value=boot_config), \
             patch("src.api.cost_plugins.init_plugins"):

            src.main.main()  # should not raise

        server.shutdown.assert_called_once()

    def test_main_initializes_router_with_config(self, boot_config):
        """main() wires the dynamic router from config.dynamic_routing."""
        import src.main
        boot_config.dynamic_routing = {"enabled": True, "cost_bias": 0.3}
        engine = MagicMock()

        with patch.object(src.main, "get_engine", return_value=engine), \
             patch.object(src.main, "init_alert_manager"), \
             patch.object(src.main, "create_server", return_value=MagicMock()), \
             patch("src.api.cost_cache.init_settings", return_value=MagicMock()), \
             patch("src.api.config.init_config", return_value=boot_config), \
             patch("src.api.cost_plugins.init_plugins"), \
             patch("src.api.router.init_router") as mock_router:

            src.main.main()

        mock_router.assert_called_once()
        _, kwargs = mock_router.call_args
        assert kwargs["enabled"] is True
        assert kwargs["cost_bias"] == 0.3

    def test_main_router_defaults_disabled(self, boot_config):
        """A config with dynamic_routing disabled leaves the router disabled."""
        import src.main
        engine = MagicMock()

        with patch.object(src.main, "get_engine", return_value=engine), \
             patch.object(src.main, "init_alert_manager"), \
             patch.object(src.main, "create_server", return_value=MagicMock()), \
             patch("src.api.cost_cache.init_settings", return_value=MagicMock()), \
             patch("src.api.config.init_config", return_value=boot_config), \
             patch("src.api.cost_plugins.init_plugins"), \
             patch("src.api.router.init_router") as mock_router:

            src.main.main()

        _, kwargs = mock_router.call_args
        assert kwargs["enabled"] is False


# ═══════════════════════════════════════════════════════════════════════
# src.server.server.create_server()
# ═══════════════════════════════════════════════════════════════════════

class TestCreateServer:
    def test_creates_http_server(self, boot_config):
        from src.server.server import create_server
        engine = MagicMock()
        http_server_cls = MagicMock()

        with patch("src.server.server.ThreadingHTTPServer", http_server_cls), \
             patch("src.server.server.init_key_manager") as mock_km:

            result = create_server(boot_config, engine, 8734)

        mock_km.assert_called_once_with(engine, "data")
        # ThreadingHTTPServer constructed with address + handler class
        addr, handler_cls = http_server_cls.call_args[0]
        assert addr == ("0.0.0.0", 8734)
        # The configured handler inherits LCPHandler and carries config/engine
        assert handler_cls.config is boot_config
        assert handler_cls.engine is engine
        assert result == http_server_cls.return_value
