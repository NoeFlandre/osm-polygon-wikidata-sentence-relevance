from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.labeling import v2_engine
from osm_polygon_sentence_relevance.labeling.v2_engine import (
    V2EngineError,
    V2LogitEngine,
)


def _response(yes: float = -0.1, no: float = -1.1) -> dict[str, object]:
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "yes",
                            "logprob": yes,
                            "top_logprobs": [
                                {"token": "yes", "logprob": yes},
                                {"token": "no", "logprob": no},
                            ],
                        }
                    ]
                }
            }
        ]
    }


def test_v2_engine_requests_one_token_and_returns_binary_scores() -> None:
    seen: list[dict[str, object]] = []

    def transport(payload: dict[str, object], _timeout: float) -> dict[str, object]:
        seen.append(payload)
        return _response()

    engine = V2LogitEngine(
        endpoint="http://server/v1/chat/completions",
        model="ggml-org/Qwen3.6-27B-GGUF",
        concurrency=1,
        transport=transport,
    )
    results = engine.generate([[{"role": "user", "content": "classify"}]])

    assert results[0].place_relevance == "yes"
    assert results[0].yes_logprob == -0.1
    assert results[0].no_logprob == -1.1
    assert seen[0]["max_tokens"] == 1
    assert seen[0]["temperature"] == 0
    assert seen[0]["logprobs"] is True
    assert seen[0]["top_logprobs"] >= 2


def test_v2_engine_requires_explicit_single_token_contract() -> None:
    engine = V2LogitEngine(
        endpoint="unused",
        model="model",
        concurrency=1,
        transport=lambda _payload, _timeout: {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "yes please",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "yes please", "logprob": -0.1},
                                    {"token": "no", "logprob": -1.0},
                                ],
                            }
                        ]
                    }
                }
            ]
        },
    )
    with pytest.raises(V2EngineError, match="exact yes and no"):
        engine.generate([[{"role": "user", "content": "classify"}]])


def test_v2_engine_uses_no_when_no_score_is_higher() -> None:
    engine = V2LogitEngine(
        endpoint="unused",
        model="model",
        concurrency=1,
        transport=lambda _payload, _timeout: _response(yes=-1.2, no=-0.2),
    )
    assert (
        engine.generate([[{"role": "user", "content": "classify"}]])[0].place_relevance
        == "no"
    )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"logprobs": {"content": []}}]},
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "yes",
                                "logprob": -1.0,
                                "top_logprobs": [{"token": "yes", "logprob": -1.0}],
                            }
                        ]
                    }
                }
            ]
        },
    ],
)
def test_v2_engine_rejects_missing_exact_yes_no_scores(
    response: dict[str, object],
) -> None:
    engine = V2LogitEngine(
        endpoint="unused",
        model="model",
        concurrency=1,
        transport=lambda _payload, _timeout: response,
    )
    with pytest.raises(V2EngineError):
        engine.generate([[{"role": "user", "content": "classify"}]])


@pytest.mark.parametrize("concurrency", [0, -1, True, "1"])
def test_v2_engine_rejects_invalid_concurrency(concurrency: object) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        V2LogitEngine(endpoint="unused", model="model", concurrency=concurrency)  # type: ignore[arg-type]


def test_v2_engine_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        V2LogitEngine(endpoint="unused", model="model", timeout_seconds=0)


def test_v2_engine_rejects_mismatched_sentence_ids() -> None:
    engine = V2LogitEngine(
        endpoint="unused",
        model="model",
        transport=lambda _payload, _timeout: _response(),
    )
    with pytest.raises(ValueError, match="sentence_ids"):
        engine.generate([[{"role": "user", "content": "classify"}]], sentence_ids=[])


def test_v2_engine_wraps_transport_failures() -> None:
    def broken(_payload: object, _timeout: float) -> dict[str, object]:
        raise RuntimeError("transport failed")

    engine = V2LogitEngine(endpoint="unused", model="model", transport=broken)
    with pytest.raises(V2EngineError, match="request failed"):
        engine.generate([[{"role": "user", "content": "classify"}]])


@pytest.mark.parametrize(
    "top_logprobs",
    [
        "not-a-list",
        [None, {"token": "yes", "logprob": -0.1}],
        [{"token": "yes", "logprob": "bad"}, {"token": "no", "logprob": -0.1}],
        [{"token": "yes", "logprob": float("nan")}, {"token": "no", "logprob": -0.1}],
    ],
)
def test_v2_engine_rejects_malformed_token_alternatives(
    top_logprobs: object,
) -> None:
    response = _response()
    response["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top_logprobs  # type: ignore[index]
    with pytest.raises(V2EngineError):
        V2LogitEngine(
            endpoint="unused",
            model="model",
            concurrency=1,
            transport=lambda _payload, _timeout: response,
        ).generate([[{"role": "user", "content": "classify"}]])


def test_v2_http_transport_rejects_non_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr(
        v2_engine.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(V2EngineError, match="not an object"):
        v2_engine._http_transport("http://unused", {}, 1.0)


def test_v2_http_transport_wraps_network_and_json_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v2_engine.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("network")),
    )
    with pytest.raises(V2EngineError, match="request failed"):
        v2_engine._http_transport("http://unused", {}, 1.0)


def test_v2_http_transport_accepts_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ObjectResponse:
        def __enter__(self) -> ObjectResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    monkeypatch.setattr(
        v2_engine.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: ObjectResponse(),
    )
    assert v2_engine._http_transport("http://unused", {}, 1.0) == {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(
        v2_engine.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(V2EngineError, match="request failed"):
        v2_engine._http_transport("http://unused", {}, 1.0)


def test_v2_engine_preserves_engine_errors_from_transport() -> None:
    def broken(_payload: object, _timeout: float) -> dict[str, object]:
        raise V2EngineError("already classified")

    with pytest.raises(V2EngineError, match="already classified"):
        V2LogitEngine(endpoint="unused", model="model", transport=broken).generate(
            [[{"role": "user", "content": "classify"}]]
        )


def test_v2_engine_private_shape_helpers_reject_wrong_types() -> None:
    with pytest.raises(V2EngineError, match="non-object"):
        v2_engine._mapping([])
    with pytest.raises(V2EngineError, match="non-list"):
        v2_engine._sequence("not-a-list")


def test_v2_engine_reuses_worker_pool_until_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class _Executor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            self.shutdown_calls: list[bool] = []
            created.append(self)

        def map(self, function, values):
            return [function(value) for value in values]

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_calls.append(wait and cancel_futures)

    monkeypatch.setattr(v2_engine, "ThreadPoolExecutor", _Executor)
    engine = V2LogitEngine(
        endpoint="unused",
        model="model",
        concurrency=2,
        transport=lambda _payload, _timeout: _response(),
    )

    messages = [[{"role": "user", "content": "classify"}]]
    engine.generate(messages)
    engine.generate(messages)

    assert len(created) == 1
    engine.close()
    assert created[0].shutdown_calls == [True]
