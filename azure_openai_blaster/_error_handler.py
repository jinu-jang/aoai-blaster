import logging
import re
from typing import Optional

from openai import RateLimitError

_RE_TRY_AGAIN_IN = re.compile(r"try again in (\d+)\s*seconds", re.IGNORECASE)


def parse_retry_after_seconds(exc: RateLimitError) -> Optional[float]:
    """Try to extract a Retry-After in seconds from the error.

    We prefer headers, but will fall back to parsing the message string
    `"Rate limit is exceeded. Try again in 60 seconds."`.
    """
    # 1) Try HTTP header if present
    try:
        headers = getattr(exc, "response", None)
        if headers is not None:
            retry_after = exc.response.headers.get("retry-after")
            logging.info(
                f"Parsed Retry-After header: {retry_after} -> {float(retry_after)}"
            )
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
    except Exception:
        pass

    # 2) Fallback: parse message text
    msg = str(exc)
    m = _RE_TRY_AGAIN_IN.search(msg)
    if m:
        try:
            logging.info(f"Parsed Retry-After message: {m.group(1)} seconds")
            return float(m.group(1))
        except ValueError:
            return None

    return None
