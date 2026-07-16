"""Tests for PipelineSettings and portable data-directory precedence.

No network access, no mounted external storage, no downloaded files, and no
probing of personal or platform-specific mount points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.application.settings import (
    PipelineSettings,
)
from osm_polygon_sentence_relevance.contracts.constants import (
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
)
from osm_polygon_sentence_relevance.contracts.errors import ConfigurationError

# Backwards-compatible legacy import path must keep working.
from osm_polygon_sentence_relevance.settings import PipelineSettings as LegacySettings


class TestDefaults:
    """Default settings use expected dataset IDs and revision."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Ensure no env var interference; cwd-based fallback.
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        s = PipelineSettings.create()
        assert s.input_dataset == INPUT_DATASET_ID
        assert s.input_revision == DEFAULT_INPUT_REVISION
        assert s.output_dataset == OUTPUT_DATASET_ID
        assert s.pipeline_version == PIPELINE_VERSION
        assert s.data_dir == tmp_path / "data"


class TestDataDirPrecedence:
    """Data directory resolves via explicit arg → OSM_DATA_DIR → cwd/data."""

    def test_cwd_data_fallback_is_dynamic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The fallback must be ``Path.cwd() / "data"`` at call time, not a
        fixed repository-relative path computed at import time."""
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        s = PipelineSettings.create()
        assert s.data_dir == tmp_path / "data"

    def test_env_var_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        custom = tmp_path / "custom"
        monkeypatch.setenv("OSM_DATA_DIR", str(custom))
        s = PipelineSettings.create()
        assert s.data_dir == custom

    def test_fallback_to_cwd_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        s = PipelineSettings.create()
        assert s.data_dir == tmp_path / "data"

    def test_explicit_data_dir_overrides_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("OSM_DATA_DIR", "/should/be/ignored")
        explicit = tmp_path / "explicit"
        s = PipelineSettings.create(data_dir=explicit)
        assert s.data_dir == explicit

    def test_whitespace_only_env_var_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Whitespace-only OSM_DATA_DIR is treated as unset (simplest defensible)."""
        monkeypatch.setenv("OSM_DATA_DIR", "   \t  ")
        monkeypatch.chdir(tmp_path)
        s = PipelineSettings.create()
        assert s.data_dir == tmp_path / "data"

    def test_expanduser_on_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A leading '~' in OSM_DATA_DIR is shell-expanded."""
        target = tmp_path / "home_data"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OSM_DATA_DIR", "~/home_data")
        s = PipelineSettings.create()
        assert s.data_dir == target


class TestValidation:
    """Empty dataset IDs and revision raise ConfigurationError."""

    def test_empty_input_dataset_raises(self):
        with pytest.raises(ConfigurationError, match="input_dataset"):
            PipelineSettings.create(input_dataset="")

    def test_empty_input_revision_raises(self):
        with pytest.raises(ConfigurationError, match="input_revision"):
            PipelineSettings.create(input_revision="")

    def test_empty_output_dataset_raises(self):
        with pytest.raises(ConfigurationError, match="output_dataset"):
            PipelineSettings.create(output_dataset="")

    @pytest.mark.parametrize("value", [" ", "  ", "\t", "\n", " \t\n "])
    def test_whitespace_only_input_dataset_raises(self, value: str):
        with pytest.raises(ConfigurationError, match="input_dataset"):
            PipelineSettings.create(input_dataset=value)

    @pytest.mark.parametrize("value", [" ", "  ", "\t", "\n", " \t\n "])
    def test_whitespace_only_input_revision_raises(self, value: str):
        with pytest.raises(ConfigurationError, match="input_revision"):
            PipelineSettings.create(input_revision=value)

    @pytest.mark.parametrize("value", [" ", "  ", "\t", "\n", " \t\n "])
    def test_whitespace_only_output_dataset_raises(self, value: str):
        with pytest.raises(ConfigurationError, match="output_dataset"):
            PipelineSettings.create(output_dataset=value)


class TestImmutability:
    """PipelineSettings is frozen."""

    def test_frozen(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        s = PipelineSettings.create()
        with pytest.raises(AttributeError):
            s.input_dataset = "something_else"  # type: ignore[misc]


class TestLegacyFacadeParity:
    """The legacy root ``settings`` module re-exports the canonical symbol."""

    def test_legacy_is_canonical(self):
        assert LegacySettings is PipelineSettings

    def test_legacy_facade_all_exposes_only_pipeline_settings(self):
        """The facade must not re-export private implementation constants."""
        import osm_polygon_sentence_relevance.settings as facade

        assert facade.__all__ == ["PipelineSettings"]


class TestNoSideEffects:
    """Construction must not create directories or access the network."""

    def test_no_directory_creation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        target = tmp_path / "should_not_exist"
        monkeypatch.setenv("OSM_DATA_DIR", str(target))
        s = PipelineSettings.create()
        assert s.data_dir == target
        assert not target.exists()
