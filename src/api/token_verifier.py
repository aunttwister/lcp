"""Token count verification — compare provider-reported usage against local estimate."""

import tiktoken

from .logging_config import get_logger

logger = get_logger("lcp.token_verifier")

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Discrepancy threshold beyond which we log a warning
DISCREPANCY_THRESHOLD = 0.25  # 25% difference


class TokenVerifier:
    """Verifies provider-reported token counts against local tiktoken estimates."""

    def __init__(self, threshold: float = DISCREPANCY_THRESHOLD):
        self.threshold = threshold
        self._warnings: int = 0
        self._checks: int = 0

    def verify(self, messages: list[dict], response_usage: dict) -> dict:
        """Compare provider usage against local estimate.

        Returns verification result dict.
        """
        self._checks += 1

        provider_prompt = response_usage.get("prompt_tokens", 0)
        provider_completion = response_usage.get("completion_tokens", 0)

        # Estimate prompt tokens
        estimated_prompt = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                estimated_prompt += len(_ENCODING.encode(content))
            estimated_prompt += 4  # overhead

        prompt_diff_pct = (
            abs(provider_prompt - estimated_prompt) / max(provider_prompt, 1)
            if provider_prompt > 0
            else 0
        )

        result = {
            "provider_prompt_tokens": provider_prompt,
            "estimated_prompt_tokens": estimated_prompt,
            "prompt_discrepancy_pct": round(prompt_diff_pct, 4),
            "provider_completion_tokens": provider_completion,
            "suspicious": prompt_diff_pct > self.threshold,
        }

        if result["suspicious"]:
            self._warnings += 1
            logger.warning(
                "token_discrepancy",
                provider=provider_prompt,
                estimated=estimated_prompt,
                pct=round(prompt_diff_pct, 2),
            )

        # Log periodic stats every 1000 checks
        if self._checks % 1000 == 0:
            logger.info(
                "token_verifier_stats",
                checks=self._checks,
                warnings=self._warnings,
                warning_rate=round(self._warnings / self._checks, 4),
            )

        return result

    @property
    def stats(self) -> dict:
        return {
            "checks": self._checks,
            "warnings": self._warnings,
            "warning_rate": round(self._warnings / max(self._checks, 1), 4),
        }


# Global instance
_token_verifier = TokenVerifier()


def get_token_verifier() -> TokenVerifier:
    return _token_verifier
