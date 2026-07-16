"""Characterization tests pinning the current public import surface.

These are intentionally import-only checks: they must remain green across
the Q3 package reorganization so that moving implementations into domain
packages never breaks a documented public import path.
"""

import importlib
import warnings

import osm_polygon_sentence_relevance as pkg


def _assert_same(obj, other):
    # Functions/modules/classes: identity (same object) after alias.
    assert obj is other, f"{obj!r} is not {other!r}"


def test_package_version_is_0_1_0():
    assert pkg.__version__ == "0.1.0"


def test_imports_are_side_effect_free():
    # Re-importing must not emit warnings or print output.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in [
            "osm_polygon_sentence_relevance.acquisition",
            "osm_polygon_sentence_relevance.cli",
            "osm_polygon_sentence_relevance.discovery",
            "osm_polygon_sentence_relevance.exporter",
            "osm_polygon_sentence_relevance.finalization",
            "osm_polygon_sentence_relevance.loading",
            "osm_polygon_sentence_relevance.pipeline",
            "osm_polygon_sentence_relevance.preprocessing",
            "osm_polygon_sentence_relevance.sat_adapter",
            "osm_polygon_sentence_relevance.segmentation",
            "osm_polygon_sentence_relevance.sentence_table",
            "osm_polygon_sentence_relevance.joins",
            "osm_polygon_sentence_relevance.constants",
            "osm_polygon_sentence_relevance.schemas",
            "osm_polygon_sentence_relevance.settings",
            "osm_polygon_sentence_relevance.errors",
        ]:
            importlib.import_module(name)


def test_acquisition_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.acquisition")
    assert hasattr(mod, "AcquisitionResult")
    assert callable(mod.acquire_dataset_snapshot)


def test_cli_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.cli")
    assert callable(mod.main)


def test_discovery_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.discovery")
    assert hasattr(mod, "RegionShardSet")
    assert callable(mod.discover_shards)


def test_exporter_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.exporter")
    assert hasattr(mod, "ExportResult")
    assert callable(mod.export_finalized_dataset)


def test_finalization_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.finalization")
    assert hasattr(mod, "FinalizationReport")
    assert hasattr(mod, "FinalizedDataset")
    assert callable(mod.sentence_content_hash)
    assert callable(mod.deterministic_sentence_id)
    assert callable(mod.finalize_sentence_dataset)


def test_loading_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.loading")
    assert callable(mod.load_validated_table)


def test_pipeline_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.pipeline")
    assert hasattr(mod, "PipelineResult")
    assert callable(mod.run_pipeline)


def test_preprocessing_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.preprocessing")
    assert callable(mod.parse_section_path)
    assert callable(mod.parse_osm_tags)
    assert callable(mod.normalize_sentence)


def test_sat_adapter_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.sat_adapter")
    assert hasattr(mod, "SaTSentenceSegmenter")


def test_segmentation_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.segmentation")
    assert hasattr(mod, "SentenceSegmenter")
    assert hasattr(mod, "SegmentationReport")
    assert callable(mod.split_validated_batch)


def test_sentence_table_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.sentence_table")
    assert hasattr(mod, "segment_joined_sections")
    assert hasattr(mod, "SegmentedTableResult")
    assert hasattr(mod, "validate_joined_sections_table")


def test_joins_public_api():
    mod = importlib.import_module("osm_polygon_sentence_relevance.joins")
    assert callable(mod.build_region_section_occurrences)
    assert callable(mod.join_wikipedia_sections)
    assert callable(mod.join_wikivoyage_sections)
    assert hasattr(mod, "JoinReport")
    assert hasattr(mod, "JoinedRegionSections")


def test_root_contract_modules_public_api():
    import osm_polygon_sentence_relevance.constants as constants
    import osm_polygon_sentence_relevance.errors as errors
    import osm_polygon_sentence_relevance.schemas as schemas
    import osm_polygon_sentence_relevance.settings as settings

    assert hasattr(constants, "INPUT_DATASET_ID")
    assert hasattr(schemas, "OUTPUT_SENTENCE_SCHEMA")
    assert hasattr(settings, "PipelineSettings")
    assert hasattr(errors, "ConfigurationError")
    assert hasattr(errors, "SegmentationError")
    assert hasattr(errors, "ExportError")
