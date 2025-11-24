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
    model: str
    """Model name as deployed in Azure OpenAI."""
    api_version: str = "2025-01-01-preview"
    """API version to use. Defaults to '2025-01-01-preview'."""
    rpm_limit: Optional[int] = None
    """(Unused) Requests-per-minute limit for this endpoint."""
    tpm_limit: Optional[int] = None
    """(Unused) Tokens-per-minute limit for this endpoint."""
    weight: int = 1
    """Weight for weighted round-robin load balancing."""

    # Passed directly to inference
    temperature: Optional[float] = None
    """Temperature setting for completions."""
    max_completion_tokens: Optional[int] = None
    """Maximum number of tokens to generate in completions."""
