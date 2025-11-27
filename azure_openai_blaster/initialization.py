import logging

from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,
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

    token = cred.get_token("https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        api_key=token.token,
        azure_endpoint=cfg.endpoint,
        api_version=cfg.api_version,
    )


def build_endpoint_states(config: dict) -> list[AzureEndpointState]:
    states: list[AzureEndpointState] = []
    for dep in config["deployments"]:
        cfg = AzureDeploymentConfig(**dep)
        client = make_client(cfg)
        states.append(AzureEndpointState(cfg=cfg, client=client))
    return states
