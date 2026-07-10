"""Dynamic model routing — short prompts → flash, long → pro."""

from typing import Optional

from .cost_estimator import count_tokens
from .logging_config import get_logger

logger = get_logger("lcp.router")


class DynamicRouter:
    """Routes requests to appropriate models based on complexity heuristics."""

    # Thresholds
    SHORT_PROMPT_THRESHOLD = 500    # tokens: use flash for prompts under this
    LONG_PROMPT_THRESHOLD = 2000    # tokens: use pro for prompts over this
    TOOL_COUNT_THRESHOLD = 3        # tools: use pro if more than this many tools

    # Model mapping per tier
    MODEL_MAP = {
        "flash": "deepseek-v4-flash",
        "pro": "deepseek-v4-pro",
    }

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def should_use_flash(self, messages: list[dict], tools: Optional[list[dict]] = None,
                         max_tokens: int = 1024) -> bool:
        """Determine if a request should use the flash (cheaper) model."""
        token_count = count_tokens(messages, tools)
        tool_count = len(tools) if tools else 0

        # Simple heuristic: combine token count and tool complexity
        if token_count > self.LONG_PROMPT_THRESHOLD:
            return False  # long prompt → pro
        if tool_count > self.TOOL_COUNT_THRESHOLD:
            return False  # many tools → pro
        if max_tokens > 2048:
            return False  # expecting long output → pro

        return True  # default to flash for simple requests

    def get_recommended_model(self, messages: list[dict], tools: Optional[list[dict]] = None,
                              max_tokens: int = 1024) -> str:
        """Get recommended model based on request complexity."""
        if self.enabled and self.should_use_flash(messages, tools, max_tokens):
            return self.MODEL_MAP["flash"]
        return self.MODEL_MAP["pro"]


# Global instance — disabled by default
_dynamic_router = DynamicRouter(enabled=False)


def get_dynamic_router() -> DynamicRouter:
    return _dynamic_router
