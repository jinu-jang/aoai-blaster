import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, RateLimitError

from azure_openai_blaster.azure_deployment import AzureDeploymentConfig


@dataclass
class AzureEndpointState:
    cfg: AzureDeploymentConfig
    """Configuration for this endpoint."""
    client: AzureOpenAI
    """HTTP client for this endpoint."""

    # Availability / cooldown
    cooldown_until: float = 0.0
    """Monotonic time until which this endpoint is considered unavailable."""
    disabled: bool = False
    """Whether this endpoint is permanently disabled."""
    disabled_reason: Optional[str] = None
    """Reason for permanent disablement, if any."""

    # Simple health / error tracking
    failure_streak: int = 0
    """Number of consecutive transient failures."""
    last_error: Optional[BaseException] = None
    """Last transient error encountered."""

    # Counters / observability
    total_requests: int = 0
    total_rate_limits: int = 0

    # error tracking
    error_counts: Dict[str, int] = field(default_factory=dict)
    """Count of errors by exception type name."""
    error_samples: List[str] = field(default_factory=list)
    """Sample error messages for debugging/reporting."""
    max_error_samples: int = 50
    """Maximum number of error messages to keep."""

    # auto-disable threshold
    auto_disable_threshold: int = 5
    """Consecutive failures required before auto-disabling this endpoint."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    def available(self, now: Optional[float] = None) -> bool:
        """Return True if the endpoint can be used right now."""
        if self.disabled:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.cooldown_until

    def _record_error(self, exc: BaseException) -> None:
        """
        Internal helper; caller must hold self.lock.
        Updates last_error, error_counts, error_samples, and rate-limit counter.
        """
        self.last_error = exc

        key = type(exc).__name__
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        # Track a bounded list of example error messages
        if len(self.error_samples) < self.max_error_samples:
            self.error_samples.append(f"{key}: {exc}")

        # Special-case rate limits for observability
        if isinstance(exc, RateLimitError):
            self.total_rate_limits += 1

    def _maybe_auto_disable(self, exc: BaseException) -> None:
        """
        Auto-disable if the failure streak has exceeded the configured threshold.
        Caller must hold self.lock.
        """
        if (
            not self.disabled
            and self.auto_disable_threshold > 0
            and self.failure_streak >= self.auto_disable_threshold
        ):
            self.disabled = True
            self.disabled_reason = (
                f"Auto-disabled after {self.failure_streak} consecutive failures; "
                f"last error: {type(exc).__name__}: {exc}"
            )
            logging.warning(
                f"Endpoint {self.cfg.name} auto-disabled due to repeated failures. "
                f"Last error: {exc}"
            )

    def set_cooldown(
        self,
        cooldown_until: float,
        exc: Optional[RateLimitError] = None,
    ) -> None:
        """
        Update this endpoint's cooldown time.

        Parameters
        ----------
        cooldown_until : float
            Absolute monotonic timestamp (time.monotonic()) until which this
            endpoint should be considered unavailable.
        exc : Optional[RateLimitError]
            Exception that triggered the cooldown (e.g., RateLimitError).
            Used only for tracking/observability; does not affect core logic.

        Notes
        -----
        - We never shorten an existing cooldown.
        - This avoids race conditions where multiple threads attempt to
          adjust cooldown from slightly different time references.
        """
        with self.lock:
            if exc is not None:
                self._record_error(exc)

                # Auto-disable if we’ve crossed the configured threshold
                self._maybe_auto_disable(exc)

            # Extend cooldown only if it increases the window.
            if cooldown_until > self.cooldown_until:
                self.cooldown_until = cooldown_until

    def note_success(self) -> None:
        """Reset transient error tracking on a successful call."""
        with self.lock:
            self.failure_streak = 0
            self.last_error = None
            self.total_requests += 1

    def note_transient_error(
        self,
        exc: BaseException,
        base_cooldown: float = 1.0,
    ) -> None:
        """
        Record a transient infra / timeout error and back off progressively.

        The cooldown is exponential in the failure streak:
        backoff = base_cooldown * (2 ** (failure_streak - 1))
        """
        now = time.monotonic()
        with self.lock:
            self.failure_streak += 1
            self._record_error(exc)

            backoff = base_cooldown * (2 ** (self.failure_streak - 1))
            new_until = now + backoff
            if new_until > self.cooldown_until:
                self.cooldown_until = new_until

            # Auto-disable if we’ve crossed the configured threshold
            self._maybe_auto_disable(exc)

    def disable(self, reason: str) -> None:
        """Permanently take this endpoint out of rotation due to config errors."""
        with self.lock:
            self.disabled = True
            self.disabled_reason = reason

    def report(self) -> Dict[str, Any]:
        """
        Snapshot of this endpoint's state and error statistics,
        suitable for logging or metrics export.
        """
        with self.lock:
            return {
                "endpoint": getattr(self.cfg, "name", repr(self.cfg)),
                "disabled": self.disabled,
                "disabled_reason": self.disabled_reason,
                "cooldown_until": self.cooldown_until,
                "failure_streak": self.failure_streak,
                "last_error": repr(self.last_error) if self.last_error else None,
                "total_requests": self.total_requests,
                "total_rate_limits": self.total_rate_limits,
                "error_counts": dict(self.error_counts),
                "error_samples": list(self.error_samples),
            }
