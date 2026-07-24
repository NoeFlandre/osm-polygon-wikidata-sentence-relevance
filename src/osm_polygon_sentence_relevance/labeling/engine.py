"""Inference-engine boundary shared by vLLM and llama.cpp servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .prompt import LABEL_RESPONSE_JSON_SCHEMA


class EngineError(RuntimeError):
    """Raised when the inference server fails its contract."""


class LabelEngine(Protocol):
    """Minimal bulk inference surface consumed by the runner."""

    def generate(self, messages: Sequence[list[dict[str, str]]]) -> list[str]:
        """Return one structured response for each prompt in input order."""


Transport = Callable[[Mapping[str, object], float], Mapping[str, object]]


def _http_transport(
    endpoint: str, payload: Mapping[str, object], timeout: float
) -> Mapping[str, object]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value: Any = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise EngineError("inference request failed") from exc
    if not isinstance(value, dict):
        raise EngineError("inference server returned an invalid response")
    return value


class OpenAICompatibleEngine:
    """Concurrent client for a local vLLM or llama.cpp chat server."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        concurrency: int = 32,
        timeout_seconds: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("concurrency must be a positive integer")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint
        self.model = model
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _one(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 384,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sentence_relevance",
                    "strict": True,
                    "schema": LABEL_RESPONSE_JSON_SCHEMA,
                },
            },
        }
        try:
            response = (
                self.transport(payload, self.timeout_seconds)
                if self.transport is not None
                else _http_transport(self.endpoint, payload, self.timeout_seconds)
            )
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError("inference request failed") from exc
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("inference server returned an invalid response") from exc
        if not isinstance(content, str):
            raise EngineError("inference server returned an invalid response")
        return content

    def generate(self, messages: Sequence[list[dict[str, str]]]) -> list[str]:
        """Generate concurrently while preserving request order."""

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(self._one, messages))


__all__ = ["EngineError", "LabelEngine", "OpenAICompatibleEngine"]
