"""Tests for the CLI ``--device`` option and ``--input-source-dataset-id`` (the implementation).

Hardware independence: every device-availability assertion is driven by
an injected capability snapshot, not by the host's actual CUDA/MPS
presence. ``main`` is called with ``caps=...`` (when the production API
exposes it) or by patching the CLI module's ``default_caps`` factory.
The tests therefore produce the same result on any host.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from osm_polygon_sentence_relevance.cli import main


def _fake_acquire(dataset_id, requested_revision, **kwargs):
    raise AssertionError("acquisition must not run when args are invalid")


def _fake_model_factory(model_name, **kwargs):
    raise AssertionError("model factory must not run when args are invalid")


class _StaticCaps:
    def __init__(self, *, cuda: bool, mps: bool) -> None:
        self.cuda_available = cuda
        self.mps_available = mps


@pytest.fixture
def caps_cpu_only(monkeypatch):
    """Force ``default_caps()`` to return a CPU-only snapshot."""
    caps = _StaticCaps(cuda=False, mps=False)
    from osm_polygon_sentence_relevance.application import cli as cli_mod
    from osm_polygon_sentence_relevance.sentences import device as device_mod

    monkeypatch.setattr(cli_mod, "default_caps", lambda: caps)
    monkeypatch.setattr(device_mod, "default_caps", lambda: caps)
    return caps


@pytest.fixture
def caps_cpu_and_cuda(monkeypatch):
    caps = _StaticCaps(cuda=True, mps=False)
    from osm_polygon_sentence_relevance.application import cli as cli_mod
    from osm_polygon_sentence_relevance.sentences import device as device_mod

    monkeypatch.setattr(cli_mod, "default_caps", lambda: caps)
    monkeypatch.setattr(device_mod, "default_caps", lambda: caps)
    return caps


@pytest.fixture
def caps_cpu_and_mps(monkeypatch):
    caps = _StaticCaps(cuda=False, mps=True)
    from osm_polygon_sentence_relevance.application import cli as cli_mod
    from osm_polygon_sentence_relevance.sentences import device as device_mod

    monkeypatch.setattr(cli_mod, "default_caps", lambda: caps)
    monkeypatch.setattr(device_mod, "default_caps", lambda: caps)
    return caps


class TestCLIDeviceArgument:
    def test_help_lists_device_choices(self, capsys):
        code = main(["--help"])
        out = capsys.readouterr().out
        assert code == 0
        assert "--device" in out
        assert "{auto,cpu,cuda,mps}" in out

    def test_help_does_not_import_torch_or_wtpsplit(self):
        """In a fresh subprocess, ``--help`` must not import torch or
        wtpsplit. We verify by checking ``sys.modules`` after ``main``
        returns (rather than inferring from importtime text).
        """
        code = (
            "import sys, json\n"
            "sys.path.insert(0, 'src')\n"
            "from osm_polygon_sentence_relevance.cli import main\n"
            "try:\n"
            "    main(['--help'])\n"
            "except SystemExit:\n"
            "    pass\n"
            "names = sorted(n for n in sys.modules "
            "             if n == 'torch' or n == 'wtpsplit' "
            "             or n.startswith('torch.') "
            "             or n.startswith('wtpsplit.'))\n"
            "print(json.dumps(names))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd="/Users/noeflandre/osm-polygon-wikidata-sentence-relevance",
        )
        assert result.returncode == 0, (
            f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        import json

        leaked = json.loads(result.stdout.strip().splitlines()[-1])
        assert leaked == [], f"--help leaked optional deps into sys.modules: {leaked}"

    def test_segmenter_construction_does_not_import_torch(self):
        """Constructing ``SaTSentenceSegmenter`` with default settings
        must NOT import Torch. Torch is only required when a non-empty
        batch triggers capability resolution and model construction.
        """
        code = (
            "import sys, json\n"
            "sys.path.insert(0, 'src')\n"
            "from osm_polygon_sentence_relevance.sentences.sat import "
            "SaTSentenceSegmenter\n"
            "seg = SaTSentenceSegmenter()\n"
            "names = sorted(n for n in sys.modules "
            "             if n == 'torch' or n.startswith('torch.'))\n"
            "print(json.dumps(names))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd="/Users/noeflandre/osm-polygon-wikidata-sentence-relevance",
        )
        assert result.returncode == 0, (
            f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        import json

        leaked = json.loads(result.stdout.strip().splitlines()[-1])
        assert leaked == [], (
            f"segmenter construction leaked torch into sys.modules: {leaked}"
        )

    def test_empty_batch_does_not_import_torch_or_wtpsplit(self):
        """An empty batch through ``SaTSentenceSegmenter.split_batch``
        must NOT import torch or wtpsplit and must NOT construct a
        model. This proves the lazy-import guarantee at the inference
        boundary.
        """
        code = (
            "import sys, json\n"
            "sys.path.insert(0, 'src')\n"
            "from osm_polygon_sentence_relevance.sentences.sat import "
            "SaTSentenceSegmenter\n"
            "seg = SaTSentenceSegmenter()\n"
            "out = seg.split_batch([], [])\n"
            "assert out == (), out\n"
            "assert seg._model is None, 'model must not be constructed on empty batch'\n"
            "names = sorted(n for n in sys.modules "
            "             if n == 'torch' or n == 'wtpsplit' "
            "             or n.startswith('torch.') "
            "             or n.startswith('wtpsplit.'))\n"
            "print(json.dumps(names))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd="/Users/noeflandre/osm-polygon-wikidata-sentence-relevance",
        )
        assert result.returncode == 0, (
            f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        import json

        leaked = json.loads(result.stdout.strip().splitlines()[-1])
        assert leaked == [], (
            f"empty batch leaked torch/wtpsplit into sys.modules: {leaked}"
        )

    def test_invalid_device_value_rejected_before_acquisition(
        self, capsys, monkeypatch, caps_cpu_only
    ):
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--device",
                "gpu",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code in (1, 2)
        captured = capsys.readouterr()
        assert "device" in (captured.err + captured.out).lower()

    def test_explicit_cuda_rejected_when_unavailable(
        self, capsys, monkeypatch, caps_cpu_only
    ):
        # CPU-only caps; explicit --device cuda must fail before any
        # side effect (acquisition, model construction, pipeline).
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--device",
                "cuda",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "cuda" in captured.err.lower() or "device" in captured.err.lower()

    def test_explicit_mps_rejected_when_unavailable(
        self, capsys, monkeypatch, caps_cpu_only
    ):
        # CPU-only caps; explicit --device mps must fail before any
        # side effect (acquisition, model construction, pipeline).
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--device",
                "mps",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "mps" in captured.err.lower() or "device" in captured.err.lower()

    def test_explicit_cuda_accepted_when_available(
        self, capsys, monkeypatch, caps_cpu_and_cuda
    ):
        # When CUDA IS available, explicit --device cuda must pass
        # validation and reach the pipeline (we do not run it; we just
        # assert that validation did not short-circuit by returning 1
        # for device unavailability).
        from osm_polygon_sentence_relevance.application import cli as cli_mod

        def _explode(*a, **k):
            raise RuntimeError("pipeline reached")

        monkeypatch.setattr(cli_mod, "run_pipeline", _explode)
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--device",
                "cuda",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        # ``main`` returns 1 because the stub raised; the important
        # thing is that the error message reflects pipeline-side
        # failure, not device unavailability.
        assert code == 1
        captured = capsys.readouterr()
        assert "pipeline reached" in captured.err

    def test_blank_device_rejected(self, capsys, monkeypatch, caps_cpu_only):
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--device",
                "  ",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code in (1, 2)
        captured = capsys.readouterr()
        assert "device" in (captured.err + captured.out).lower()


class TestCLIInputSourceDatasetId:
    def test_local_source_id_rejected_in_hub_mode(
        self, capsys, monkeypatch, caps_cpu_only
    ):
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-dataset-id",
                "NoeFlandre/repo",
                "--input-source-dataset-id",
                "Local/Other",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert (
            "input-source-dataset-id" in captured.err
            or "source" in captured.err.lower()
        )

    def test_local_source_id_blank_rejected(self, capsys, monkeypatch, caps_cpu_only):
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: pytest.fail("run_pipeline must not run"),
        )
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--input-source-dataset-id",
                "  ",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "source" in captured.err.lower()

    def test_local_source_id_reaches_pipeline_dataset_id(
        self, capsys, monkeypatch, caps_cpu_only, tmp_path
    ):
        """Local mode + ``--input-source-dataset-id`` must thread the
        source ID into ``run_pipeline(input_dataset_id=...)``.
        """
        from osm_polygon_sentence_relevance.application import cli as cli_mod

        captured: dict[str, object] = {}

        def _fake_run_pipeline(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("pipeline stopped")

        monkeypatch.setattr(cli_mod, "run_pipeline", _fake_run_pipeline)
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--input-source-dataset-id",
                "Local/Source",
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            acquisition_fn=_fake_acquire,
            model_factory=_fake_model_factory,
        )
        assert code == 1
        assert captured.get("input_dataset_id") == "Local/Source"
