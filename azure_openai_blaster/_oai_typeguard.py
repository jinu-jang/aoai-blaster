from typing import Any, TypeGuard

from openai.types.chat import ChatCompletionMessageParam


def is_chat_message(obj: Any) -> TypeGuard[ChatCompletionMessageParam]:
    """Type guard to validate an arbitrary dict is a ChatCompletionMessageParam."""
    if not isinstance(obj, dict):
        return False

    role = obj.get("role")
    if not isinstance(role, str):
        return False

    # role-discriminated checks
    if role in ["user", "system", "developer", "function"]:
        allowed_keys = {"role", "content", "name"}
        if not set(obj.keys()).issubset(allowed_keys):
            return False
        return "content" in obj

    elif role == "assistant":
        allowed_keys = {
            "role",
            "audio",
            "content",
            "function_call",
            "name",
            "refusal",
            "tool_calls",
        }
        if not set(obj.keys()).issubset(allowed_keys):
            return False

        has_content = ("content" in obj) and (obj.get("content") is not None)
        has_tool_calls = "tool_calls" in obj
        has_function_call = "function_call" in obj

        # content is required unless tool_calls or function_call is present
        return has_content or has_tool_calls or has_function_call

    elif role == "tool":
        return "content" in obj and "tool_call_id" in obj

    return False


def is_chat_message_list(
    obj: Any,
) -> TypeGuard[list[ChatCompletionMessageParam]]:
    if not isinstance(obj, list):
        return False
    return all(is_chat_message(m) for m in obj)
