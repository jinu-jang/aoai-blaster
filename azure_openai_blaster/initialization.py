import logging

from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

from azure_openai_blaster.azure_deployment import AzureDeploymentConfig
from azure_openai_blaster.azure_endpoint_state import AzureEndpointState


def make_client(cfg: AzureDeploymentConfig) -> AzureOpenAI:
    if cfg.api_key.lower() not in ("default", "az", "interactive") and cfg.api_key:
        return AzureOpenAI(
            api_key=cfg.api_key,
            azure_endpoint=cfg.endpoint,
            api_version=cfg.api_version,
        )

    # Token-based auth
    if cfg.api_key.lower() == "interactive":
        cred = InteractiveBrowserCredential()
        logging.info(f"Using InteractiveBrowserCredential for deployment '{cfg.name}'")
    elif cfg.api_key.lower() == "az":
        cred = AzureCliCredential()
        logging.info(f"Using AzureCliCredential for deployment '{cfg.name}'")
    else:
        logging.info(f"Using DefaultAzureCredential for deployment '{cfg.name}'")
        cred = DefaultAzureCredential()

    token_provider = get_bearer_token_provider(
        cred, "https://cognitiveservices.azure.com/.default"
    )
    # Trigger a token fetch to ensure it works.
    # Allows us to fail before multi-threading starts.
    _ = token_provider()

    return AzureOpenAI(
        azure_endpoint=cfg.endpoint,
        api_version=cfg.api_version,
        azure_ad_token_provider=token_provider,
    )


def build_endpoint_states(config: dict) -> list[AzureEndpointState]:
    states: list[AzureEndpointState] = []
    for dep in config["deployments"]:
        cfg = AzureDeploymentConfig(**dep)
        client = make_client(cfg)
        states.append(AzureEndpointState(cfg=cfg, client=client))
    return states
