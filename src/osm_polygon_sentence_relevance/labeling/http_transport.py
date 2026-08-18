"""Shared JSON-over-HTTP request plumbing for labeling engines."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

Urlopen = Callable[..., Any]


def _post_json(
    endpoint: str,
    payload: Mapping[str, object],
    timeout: float,
    *,
    urlopen: Urlopen,
) -> object:
    """POST a JSON payload and decode the JSON response.

    Transport errors and response-shape validation remain with each engine so
    they can preserve their existing error types and messages.
    """

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)
