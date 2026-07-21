"""Characterization tests for the supported package surface."""

import subprocess
import sys
import textwrap
from inspect import signature

from osm_polygon_sentence_relevance.application.checkpoint import (
    load_shard_checkpoint,
    publish_shard_checkpoint,
    validate_source_commit,
    validate_work_dir,
)
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline
from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
    compute_parquet_statistics,
    render_dataset_card,
    render_dataset_card_from_profile,
)


def test_supported_callable_signatures_are_stable() -> None:
    """Structural cleanup must preserve documented entry points."""
    assert "work_dir" in signature(run_pipeline).parameters
    assert list(signature(validate_source_commit).parameters) == ["value"]
    assert list(signature(validate_work_dir).parameters) == ["work_dir"]
    assert "verified_manifest" in signature(publish_shard_checkpoint).parameters
    assert "shard_key" in signature(load_shard_checkpoint).parameters
    assert DatasetStatistics.__name__ == "DatasetStatistics"
    assert callable(compute_parquet_statistics)
    assert callable(render_dataset_card)
    assert callable(render_dataset_card_from_profile)


def test_cli_import_does_not_require_publication_plot_dependencies() -> None:
    code = textwrap.dedent(
        """
        import builtins
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.', 1)[0] in {'matplotlib', 'numpy', 'PIL'}:
                raise ImportError(f'blocked optional dependency: {name}')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        from osm_polygon_sentence_relevance.application.cli import main
        assert callable(main)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
