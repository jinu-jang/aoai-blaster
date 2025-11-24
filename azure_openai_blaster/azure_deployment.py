from dataclasses import dataclass
from typing import Literal, Optional


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
    rpm_limit: Optional[int] = None
    """(Unused) Requests-per-minute limit for this endpoint."""
    tpm_limit: Optional[int] = None
    """(Unused) Tokens-per-minute limit for this endpoint."""
    weight: int = 1
    """Weight for weighted round-robin load balancing."""
