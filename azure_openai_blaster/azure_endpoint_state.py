import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from openai import AzureOpenAI


@dataclass
class AzureDeploymentConfig:
    """Configuration for a single Azure OpenAI deployment endpoint."""

    name: str
    """Name of deployment; for logging purposes."""
    endpoint: str
    """Azure OpenAI endpoint URL."""
    api_key: str | Literal["default", "interactive"]
    """API key for authentication.
    - Specify "default" for DefaultAzureCredential() based auth.
    - Specify "interactive" for InteractiveBrowserCredential() based auth.
    """
    model: Optional[str] = None
    """Model name."""
    weight: int = 1
    """Weight for weighted round-robin load balancing."""


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

    # (Optional) counters for observability
    total_requests: int = 0
    total_rate_limits: int = 0

    lock: threading.Lock = field(default_factory=threading.Lock)

    def available(self, now: Optional[float] = None) -> bool:
        """Return True if the endpoint can be used right now."""
        if self.disabled:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.cooldown_until

    def set_cooldown(self, cooldown_until: float) -> None:
        """
        Update this endpoint's cooldown time.

        Parameters
        ----------
        cooldown_until : float
            Absolute monotonic timestamp (time.monotonic()) until which this
            endpoint should be considered unavailable.
            The caller must compute this based on the Retry-After value
            at the moment the 429 was received.

        Notes
        -----
        - We never shorten an existing cooldown.
        - This avoids race conditions where multiple threads attempt to
        adjust cooldown from slightly different time references.
        """
        with self.lock:
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
        self, exc: BaseException, base_cooldown: float = 1.0
    ) -> None:
        """Back off progressively on repeated infra / timeout errors."""
        with self.lock:
            self.failure_streak += 1
            self.last_error = exc
            backoff = base_cooldown * (2 ** (self.failure_streak - 1))
            new_until = time.monotonic() + backoff
            if new_until > self.cooldown_until:
                self.cooldown_until = new_until

    def disable(self, reason: str) -> None:
        """Permanently take this endpoint out of rotation due to config errors."""
        with self.lock:
            self.disabled = True
            self.disabled_reason = reason
