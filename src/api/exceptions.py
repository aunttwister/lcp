"""Custom exception hierarchy for the LCP gateway."""


class LCPError(Exception):
    """Base exception for all LCP gateway errors."""


class ConfigError(LCPError):
    """Configuration validation or loading error."""


class AuthError(LCPError):
    """Authentication or authorization failure."""


class CreditExhaustedError(LCPError):
    """User or team has exhausted their credit limit."""


class ProviderError(LCPError):
    """Upstream provider returned an error."""


class ProviderTimeoutError(ProviderError):
    """Upstream provider timed out."""


class ProviderRateLimitError(ProviderError):
    """Upstream provider rate limited us."""


class ProviderAuthError(ProviderError):
    """Upstream provider rejected our API key."""


class AllProvidersFailedError(LCPError):
    """All providers in chain are unavailable."""


class ToolBlockedError(LCPError):
    """Requested tool is blocked by profile policy."""
