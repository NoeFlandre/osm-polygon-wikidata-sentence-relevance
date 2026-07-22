from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from osm_polygon_sentence_relevance.labeling.engine import (
    EngineError,
    OpenAICompatibleEngine,
)
from osm_polygon_sentence_relevance.labeling.prompt import (
    LABEL_RESPONSE_JSON_SCHEMA,
)


def _messages(value: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": value}]


def test_sends_closed_json_schema_and_preserves_order() -> None:
    captured: list[Mapping[str, object]] = []

    def transport(
        payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]:
        captured.append(payload)
        content = payload["messages"][0]["content"]  # type: ignore[index]
        return {"choices": [{"message": {"content": json.dumps({"value": content})}}]}

    engine = OpenAICompatibleEngine(
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="pinned-model",
        concurrency=2,
        timeout_seconds=30,
        transport=transport,
    )
    outputs = engine.generate([_messages("first"), _messages("second")])

    assert [json.loads(output)["value"] for output in outputs] == ["first", "second"]
    assert all(payload["temperature"] == 0 for payload in captured)
    assert all(payload["max_tokens"] == 96 for payload in captured)
    schema = captured[0]["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert schema == LABEL_RESPONSE_JSON_SCHEMA


@pytest.mark.parametrize("concurrency", [0, -1, True])
def test_rejects_invalid_concurrency(concurrency: object) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        OpenAICompatibleEngine(
            endpoint="http://localhost",
            model="m",
            concurrency=concurrency,  # type: ignore[arg-type]
        )


def test_wraps_transport_failure_without_prompt_leak() -> None:
    def transport(
        payload: Mapping[str, object], timeout: float
    ) -> Mapping[str, object]:
        raise OSError("secret prompt path")

    engine = OpenAICompatibleEngine(
        endpoint="http://localhost", model="m", transport=transport
    )
    with pytest.raises(EngineError, match="request failed") as caught:
        engine.generate([_messages("sensitive sentence")])
    assert "sensitive" not in str(caught.value)


def test_rejects_malformed_server_response() -> None:
    engine = OpenAICompatibleEngine(
        endpoint="http://localhost",
        model="m",
        transport=lambda payload, timeout: {"choices": []},
    )
    with pytest.raises(EngineError, match="response"):
        engine.generate([_messages("x")])


def test_real_http_transport_round_trip() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            content = json.dumps({"echo": payload["messages"][0]["content"]})
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        output = OpenAICompatibleEngine(endpoint=endpoint, model="m").generate(
            [_messages("hello")]
        )
        assert json.loads(output[0]) == {"echo": "hello"}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleEngine(
            endpoint="http://localhost", model="m", timeout_seconds=0
        )
