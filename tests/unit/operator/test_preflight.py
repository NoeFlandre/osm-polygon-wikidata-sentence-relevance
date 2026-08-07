"""Contract tests for local and remote operator preflight checks."""

from __future__ import annotations

import sys
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import preflight


def test_git_head_requires_a_clean_full_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            SimpleNamespace(stdout="short\n"),
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout="dirty\n"),
        ]
    )
    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: next(responses))

    with pytest.raises(RuntimeError, match="immutable"):
        preflight.git_head()
    with pytest.raises(RuntimeError, match="clean"):
        preflight.git_head()


def test_git_head_returns_a_clean_full_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            SimpleNamespace(text="a" * 40 + "\n"),
            SimpleNamespace(stdout=""),
        ]
    )
    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: next(responses))

    assert preflight.git_head() == "a" * 40


def test_resolve_input_revision_uses_the_output_dataset_for_labeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert preflight.resolve_input_revision("a" * 40, "label") == "a" * 40

    class Api:
        def dataset_info(self, dataset_id: str, *, revision: str) -> SimpleNamespace:
            assert dataset_id == preflight.OUTPUT_DATASET_ID
            assert revision == "main"
            return SimpleNamespace(sha="b" * 40)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert preflight.resolve_input_revision(None, "label") == "b" * 40
    sys.modules.pop("huggingface_hub", None)


def test_resolve_input_revision_uses_input_dataset_for_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def dataset_info(self, dataset_id: str, *, revision: str) -> SimpleNamespace:
            assert dataset_id == preflight.INPUT_DATASET_ID
            assert revision == "main"
            return SimpleNamespace(sha="c" * 40)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert preflight.resolve_input_revision(None, "split") == "c" * 40
    sys.modules.pop("huggingface_hub", None)


def test_resolve_input_revision_rejects_missing_hub_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def dataset_info(self, _dataset_id: str, *, revision: str) -> SimpleNamespace:
            assert revision == "main"
            return SimpleNamespace(sha=None)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    with pytest.raises(RuntimeError, match="did not resolve"):
        preflight.resolve_input_revision(None, "split")
    sys.modules.pop("huggingface_hub", None)


def test_resolve_input_revision_reports_missing_hub_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def fail_hub(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub":
            raise ImportError("missing hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_hub)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    with pytest.raises(RuntimeError, match="hub extra"):
        preflight.resolve_input_revision(None, "split")


def test_remote_home_accepts_an_absolute_path_without_traversal() -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="/home/user\n")

    assert preflight.remote_home(FakeSsh()) == PurePosixPath("/home/user")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["relative\n", "/home/../escape\n", "/home/x\ny"])
def test_remote_home_rejects_unsafe_output(value: str) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout=value)

    with pytest.raises(RuntimeError, match="invalid"):
        preflight.remote_home(FakeSsh())  # type: ignore[arg-type]


def test_usage_policy_preflight_runs_both_live_checks() -> None:
    class FakeSsh:
        command = ""

        def run(self, command: str) -> SimpleNamespace:
            self.__class__.command = command
            return SimpleNamespace(stdout="")

    preflight.usage_policy_preflight(FakeSsh(), "nancy")  # type: ignore[arg-type]
    assert "usagepolicycheck -l --sites nancy" in FakeSsh.command
    assert "usagepolicycheck -t" in FakeSsh.command


@pytest.mark.parametrize("site", ["", "Nancy", "nancy;true", "nancy site"])
def test_usage_policy_preflight_rejects_unsafe_site(site: str) -> None:
    with pytest.raises(ValueError, match="site name"):
        preflight.usage_policy_preflight(SimpleNamespace(), site)  # type: ignore[arg-type]
