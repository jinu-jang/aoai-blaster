import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from openai import APITimeoutError, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from azure_openai_blaster._error_handler import parse_retry_after_seconds
from azure_openai_blaster._oai_typeguard import is_chat_message_list
from azure_openai_blaster.azure_endpoint_state import AzureEndpointState


@dataclass
class RequestResult:
    """Unified format for LLM endpoint request results."""

    ok: bool
    """Whether the request succeeded."""
    retryable: bool
    """Whether the request can be retried."""
    response: Optional[str] = None
    """Response content, if successful."""
    error: Optional[BaseException] = None
    """Error encountered, if any."""


def invoke_endpoint(
    ep: AzureEndpointState,
    messages: list[dict] | list[ChatCompletionMessageParam],
    **kwargs,
) -> RequestResult:
    if not is_chat_message_list(messages):
        raise ValueError("messages must be a list of ChatCompletionMessageParam")
    try:
        resp = (
            _stream_completion(ep, messages)
            if kwargs.get("stream")
            else _completion(ep, messages)
        )
        ep.note_success()
        return RequestResult(ok=True, retryable=False, response=resp)

    except RateLimitError as e:
        retry_after = parse_retry_after_seconds(e)

        if not retry_after:
            logging.warning(
                f"RateLimitError from {ep.cfg.name} "
                "without Retry-After header nor parseable message. "
                "Defaulting to 15s cooldown. Error: "
                f"{e}"
            )
            retry_after = 15.0
        else:
            logging.info(
                f"RateLimitError from {ep.cfg.name}; "
                f"setting cooldown for {retry_after} seconds."
            )

        cooldown_until = time.monotonic() + retry_after
        ep.set_cooldown(cooldown_until, exc=e)
        return RequestResult(ok=False, retryable=True, error=e)

    except AuthenticationError as e:
        ep.disable("auth error")
        logging.error(
            f"AuthenticationError from {ep.cfg.name}; disabling endpoint. Error: {e}"
        )
        return RequestResult(ok=False, retryable=False, error=e)

    except APITimeoutError as e:
        ep.note_transient_error(e, base_cooldown=1.0)
        logging.warning(
            f"APITimeoutError from {ep.cfg.name}; applying transient error cooldown. Error: {e}"
        )
        return RequestResult(ok=False, retryable=True, error=e)

    except BadRequestError as e:
        logging.error(
            f"BadRequestError sent to {ep.cfg.name}; not retrying. Error: {e}"
        )

        # per-request bug, not endpoint bug
        return RequestResult(ok=False, retryable=False, error=e)


def _completion(
    ep: AzureEndpointState, messages: Iterable[ChatCompletionMessageParam]
) -> str:
    resp = ep.client.chat.completions.create(
        model=ep.cfg.model,
        messages=messages,
        temperature=ep.cfg.temperature,
        max_completion_tokens=ep.cfg.max_completion_tokens,
        stream=False,
    )
    # If token usage present, we could keep stats here; not needed for gating
    return resp.choices[0].message.content or ""


def _stream_completion(
    ep: AzureEndpointState, messages: Iterable[ChatCompletionMessageParam]
) -> str:
    chunks = ep.client.chat.completions.create(
        model=ep.cfg.model,
        messages=messages,
        temperature=ep.cfg.temperature,
        max_completion_tokens=ep.cfg.max_completion_tokens,
        stream=True,
    )

    parts = []
    # Some SDKs emit a final "usage" block; we don't depend on it here.
    for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            parts.append(delta.content)
    return "".join(parts)
