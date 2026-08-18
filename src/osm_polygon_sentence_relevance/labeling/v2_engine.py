"""llama.cpp client for one-token yes/no log-probability classification."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, cast

from .http_transport import _post_json
from .v2_contracts import V2LogitRecord
from .worker_pool import _ReusableWorkerPool


class V2EngineError(RuntimeError):
    """Raised when the binary score contract is not available."""


Transport = Callable[[Mapping[str, object], float], Mapping[str, object]]


class V2Engine(Protocol):
    """Minimal engine contract required by the resumable V2 runner."""

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        sentence_ids: Sequence[str] | None = None,
    ) -> list[V2LogitRecord]: ...


def _http_transport(
    endpoint: str, payload: Mapping[str, object], timeout: float
) -> Mapping[str, object]:
    try:
        value = _post_json(
            endpoint,
            payload,
            timeout,
            urlopen=urllib.request.urlopen,
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise V2EngineError("binary inference request failed") from exc
    if not isinstance(value, dict):
        raise V2EngineError("binary inference response is not an object")
    return cast(Mapping[str, object], value)


def _score_map(response: Mapping[str, object]) -> dict[str, float]:
    try:
        choices = _sequence(response["choices"])
        choice = _mapping(choices[0])
        logprobs = _mapping(choice["logprobs"])
        content = _sequence(logprobs["content"])
        first = _mapping(content[0])
        top = first["top_logprobs"]
    except (KeyError, IndexError, TypeError, V2EngineError) as exc:
        raise V2EngineError("response does not contain first-token logprobs") from exc
    if not isinstance(top, list):
        raise V2EngineError("response does not contain token alternatives")
    result: dict[str, float] = {}
    for entry in top:
        if not isinstance(entry, Mapping):
            continue
        token = entry.get("token")
        value = entry.get("logprob")
        if not isinstance(token, str) or not isinstance(value, (int, float)):
            continue
        normalized = token.strip().lower()
        if normalized in {"yes", "no"} and math.isfinite(float(value)):
            result[normalized] = float(value)
    if set(result) != {"yes", "no"}:
        raise V2EngineError("first-token alternatives must contain exact yes and no")
    return result


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise V2EngineError("response contains a non-object logprob field")
    return cast(Mapping[str, object], value)


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise V2EngineError("response contains a non-list logprob field")
    return value


class V2LogitEngine:
    """Concurrent llama.cpp client that never asks the model to generate JSON."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        concurrency: int = 16,
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
        self._workers = _ReusableWorkerPool(
            max_workers=self.concurrency,
            error_type=V2EngineError,
            executor_factory=ThreadPoolExecutor,
        )

    def close(self) -> None:
        """Release the reusable request workers after a run."""

        self._workers.close()

    def _one(self, messages: list[dict[str, str]]) -> V2LogitRecord:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1,
            "seed": 0,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = (
                self.transport(payload, self.timeout_seconds)
                if self.transport is not None
                else _http_transport(self.endpoint, payload, self.timeout_seconds)
            )
        except V2EngineError:
            raise
        except Exception as exc:
            raise V2EngineError("binary inference request failed") from exc
        scores = _score_map(response)
        margin = scores["yes"] - scores["no"]
        return V2LogitRecord(
            sentence_id="pending",
            place_relevance="yes" if margin > 0 else "no",
            yes_logprob=scores["yes"],
            no_logprob=scores["no"],
        )

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        sentence_ids: Sequence[str] | None = None,
    ) -> list[V2LogitRecord]:
        """Return score records in request order, bound to the supplied IDs."""

        ids = (
            list(sentence_ids)
            if sentence_ids is not None
            else ["pending"] * len(messages)
        )
        if len(ids) != len(messages):
            raise ValueError("sentence_ids must match message count")
        executor = self._workers.get()
        records = list(executor.map(self._one, messages))
        return [
            V2LogitRecord(
                sentence_id=sentence_id,
                place_relevance=record.place_relevance,
                yes_logprob=record.yes_logprob,
                no_logprob=record.no_logprob,
            )
            for sentence_id, record in zip(ids, records, strict=True)
        ]


__all__ = ["V2Engine", "V2EngineError", "V2LogitEngine"]
