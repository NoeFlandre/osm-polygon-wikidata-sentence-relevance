"""Contract tests for the extracted continuation state machine."""

from types import SimpleNamespace

from osm_polygon_sentence_relevance.operator import cli, continuation
from osm_polygon_sentence_relevance.operator.oar import ExitClass


def test_continuation_module_exposes_classifier() -> None:
    assert callable(continuation.classify_or_continue)


def test_cli_adapter_delegates_to_continuation_module(monkeypatch) -> None:
    seen: dict[str, object] = {}
    services = object()

    def fake_classifier(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ExitClass.CONTINUE

    monkeypatch.setattr(continuation, "classify_or_continue", fake_classifier)
    monkeypatch.setattr(cli, "_continuation_services", lambda: services)

    args = SimpleNamespace()
    store = object()
    config = object()
    result = cli._classify_or_continue(
        args,
        store,
        config,
        "nancy",
        42,
        destination_site="sophia",
    )

    assert result is ExitClass.CONTINUE
    assert seen == {
        "args": (args, store, config, "nancy", 42),
        "kwargs": {"destination_site": "sophia", "services": services},
    }
