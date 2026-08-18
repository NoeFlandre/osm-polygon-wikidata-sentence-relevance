from __future__ import annotations

import json

from osm_polygon_sentence_relevance.labeling.http_transport import _post_json


def test_post_json_builds_request_and_decodes_response() -> None:
    calls: list[tuple[object, float]] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def urlopen(request: object, *, timeout: float) -> _Response:
        calls.append((request, timeout))
        return _Response()

    result = _post_json(
        "https://example.test/v1/chat/completions",
        {"text": "café"},
        3.5,
        urlopen=urlopen,
    )

    assert result == {"ok": True}
    assert len(calls) == 1
    request, timeout = calls[0]
    assert timeout == 3.5
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode()) == {"text": "café"}
