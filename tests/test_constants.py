"""Tests for pipeline constants."""

from __future__ import annotations

from osm_polygon_sentence_relevance.constants import (
    ALLOWED_INPUT_PATHS,
    ALLOWED_SOURCES,
    SCHEMA_NAMES,
)


class TestAllowedInputPaths:
    """The input path allowlist must exclude obsolete articles/."""

    def test_articles_excluded(self):
        for path in ALLOWED_INPUT_PATHS:
            assert "articles" not in path.split("/"), (
                f"Obsolete articles/ must not appear in ALLOWED_INPUT_PATHS, "
                f"found: {path!r}"
            )

    def test_exact_six_paths(self):
        assert ALLOWED_INPUT_PATHS == (
            "polygons",
            "polygon_articles",
            "wikipedia/documents",
            "wikipedia/sections",
            "wikivoyage/documents",
            "wikivoyage/sections",
        )


class TestAllowedSources:
    def test_exactly_two_sources(self):
        assert ALLOWED_SOURCES == frozenset({"wikipedia", "wikivoyage"})


class TestSchemaNames:
    def test_exact_six_schema_names(self):
        assert SCHEMA_NAMES == (
            "polygons",
            "polygon_articles",
            "wikipedia_documents",
            "wikivoyage_documents",
            "wikipedia_sections",
            "wikivoyage_sections",
        )
