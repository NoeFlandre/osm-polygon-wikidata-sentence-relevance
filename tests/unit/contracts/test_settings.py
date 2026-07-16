"""Tests for PipelineSettings and data-directory precedence.

No network access, no mounted external storage, no downloaded files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.constants import (
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
)
from osm_polygon_sentence_relevance.errors import ConfigurationError
from osm_polygon_sentence_relevance.settings import (
    _REPO_LOCAL_DATA_DIR,
    PipelineSettings,
)


class TestDefaults:
    """Default settings use expected dataset IDs and revision."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Ensure no env var interference and Seagate not detected.
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.settings._SEAGATE_DATA_DIR",
            tmp_path / "nonexistent",
        )
        s = PipelineSettings.create()
        assert s.input_dataset == INPUT_DATASET_ID
        assert s.input_revision == DEFAULT_INPUT_REVISION
        assert s.output_dataset == OUTPUT_DATASET_ID
        assert s.pipeline_version == PIPELINE_VERSION


class TestDataDirPrecedence:
    """Data directory resolves via env → Seagate → local fallback."""

    def test_env_var_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        custom = tmp_path / "custom"
        monkeypatch.setenv("OSM_DATA_DIR", str(custom))
        s = PipelineSettings.create()
        assert s.data_dir == custom

    def test_seagate_used_when_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        fake_seagate = tmp_path / "seagate"
        fake_seagate.mkdir()
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.settings._SEAGATE_DATA_DIR",
            fake_seagate,
        )
        s = PipelineSettings.create()
        assert s.data_dir == fake_seagate

    def test_fallback_to_repo_local(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("OSM_DATA_DIR", raising=False)
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.settings._SEAGATE_DATA_DIR",
            tmp_path / "nonexistent",
        )
        s = PipelineSettings.create()
        assert s.data_dir == _REPO_LOCAL_DATA_DIR

    def test_explicit_data_dir_overrides_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("OSM_DATA_DIR", "/should/be/ignored")
        explicit = tmp_path / "explicit"
        s = PipelineSettings.create(data_dir=explicit)
        assert s.data_dir == explicit


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
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.settings._SEAGATE_DATA_DIR",
            tmp_path / "nonexistent",
        )
        s = PipelineSettings.create()
        with pytest.raises(AttributeError):
            s.input_dataset = "something_else"  # type: ignore[misc]


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
