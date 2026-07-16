"""Phase 4 characterization: canonical ``contracts`` layout and identity.

These tests pin the *new* canonical module structure introduced by Q4:

- ``osm_polygon_sentence_relevance.contracts.constants``
- ``osm_polygon_sentence_relevance.contracts.errors``
- ``osm_polygon_sentence_relevance.contracts.schemas`` (stable public API)
  with ``input.py``, ``pipeline.py``, ``registry.py``

They assert that legacy root imports and the new canonical imports resolve
to the *same* objects (schema objects, constants, exception classes,
functions) so the move is purely structural. They also pin schema field
order, types, metadata, nullability, registry keys, validation behavior,
exception inheritance, and messages.

They are RED until the contracts domain exists.
"""

from __future__ import annotations

import importlib

import pyarrow as pa
import pytest

CONTRACT_NAMES = [
    "osm_polygon_sentence_relevance.contracts.constants",
    "osm_polygon_sentence_relevance.contracts.errors",
    "osm_polygon_sentence_relevance.contracts.schemas",
]

# (canonical module attr, legacy root module attr) pairs whose objects
# must be identical after the move.
SCHEMA_OBJECTS = [
    "POLYGONS_SCHEMA",
    "POLYGON_ARTICLES_SCHEMA",
    "WIKIPEDIA_DOCUMENTS_SCHEMA",
    "WIKIVOYAGE_DOCUMENTS_SCHEMA",
    "SECTIONS_SCHEMA",
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
]

CONSTANT_OBJECTS = [
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_INPUT_REVISION",
    "PIPELINE_VERSION",
    "ALLOWED_SOURCES",
    "SCHEMA_NAMES",
    "ALLOWED_INPUT_PATHS",
]

ERROR_CLASSES = [
    "ConfigurationError",
    "SchemaContractError",
    "UnknownTableError",
    "MissingColumnsError",
    "IncompatibleTypesError",
    "PreprocessingError",
    "SegmentationError",
    "ShardDiscoveryError",
    "JoinIntegrityError",
    "FinalizationError",
    "ExportError",
    "AcquisitionError",
]


def _canonical(attr: str):
    mod = importlib.import_module("osm_polygon_sentence_relevance.contracts.schemas")
    return getattr(mod, attr)


def _legacy(attr: str):
    mod = importlib.import_module("osm_polygon_sentence_relevance.schemas")
    return getattr(mod, attr)


class TestContractsModulesExist:
    """The canonical contracts subpackages must exist and import cleanly."""

    def test_contracts_package_imports(self):
        for name in CONTRACT_NAMES:
            assert importlib.import_module(name) is not None

    def test_schemas_submodules_exist(self):
        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.input"
        )
        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.pipeline"
        )
        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.registry"
        )


class TestSchemaIdentity:
    """Canonical and legacy schema objects are the same object (identity)."""

    @pytest.mark.parametrize("attr", SCHEMA_OBJECTS)
    def test_schema_object_identity(self, attr: str):
        assert _canonical(attr) is _legacy(attr)


class TestSchemaFieldContract:
    """Field order, name, type, and nullability are unchanged."""

    @pytest.mark.parametrize("attr", SCHEMA_OBJECTS)
    def test_field_order_names_types_nullability(self, attr: str):
        canonical = _canonical(attr)
        legacy = _legacy(attr)
        assert isinstance(canonical, pa.Schema)
        assert isinstance(legacy, pa.Schema)
        c_fields = [(f.name, f.type, f.nullable) for f in canonical]
        l_fields = [(f.name, f.type, f.nullable) for f in legacy]
        assert c_fields == l_fields


class TestConstantIdentity:
    """All public constants resolve identically across legacy and canonical."""

    @pytest.mark.parametrize("attr", CONSTANT_OBJECTS)
    def test_constant_identity(self, attr: str):
        c = getattr(
            importlib.import_module(
                "osm_polygon_sentence_relevance.contracts.constants"
            ),
            attr,
        )
        legacy = getattr(
            importlib.import_module("osm_polygon_sentence_relevance.constants"),
            attr,
        )
        assert c is legacy or c == legacy


class TestErrorIdentity:
    """Exception classes are identical across legacy and canonical."""

    @pytest.mark.parametrize("attr", ERROR_CLASSES)
    def test_error_identity(self, attr: str):
        c = getattr(
            importlib.import_module("osm_polygon_sentence_relevance.contracts.errors"),
            attr,
        )
        legacy = getattr(
            importlib.import_module("osm_polygon_sentence_relevance.errors"),
            attr,
        )
        assert c is legacy


class TestErrorInheritance:
    """Inheritance hierarchy and messages remain unchanged."""

    def test_hierarchy(self):
        import osm_polygon_sentence_relevance.contracts.errors as errs

        assert issubclass(errs.UnknownTableError, errs.SchemaContractError)
        assert issubclass(errs.MissingColumnsError, errs.SchemaContractError)
        assert issubclass(errs.IncompatibleTypesError, errs.SchemaContractError)
        assert issubclass(errs.FinalizationError, ValueError)
        assert issubclass(errs.ExportError, ValueError)
        assert issubclass(errs.AcquisitionError, ValueError)

    def test_unknown_table_message(self):
        import osm_polygon_sentence_relevance.contracts.errors as errs

        exc = errs.UnknownTableError("nope")
        assert exc.table_name == "nope"
        assert "Unknown table name" in str(exc)


class TestRegistry:
    """SCHEMA_REGISTRY keys and validation behavior are unchanged."""

    def test_registry_keys(self):
        import osm_polygon_sentence_relevance.contracts.schemas as sch

        assert set(sch.SCHEMA_REGISTRY) == {
            "polygons",
            "polygon_articles",
            "wikipedia_documents",
            "wikivoyage_documents",
            "wikipedia_sections",
            "wikivoyage_sections",
        }

    def test_registry_identity_with_legacy(self):
        import osm_polygon_sentence_relevance.contracts.schemas as csch
        import osm_polygon_sentence_relevance.schemas as lsch

        for key, value in csch.SCHEMA_REGISTRY.items():
            assert lsch.SCHEMA_REGISTRY[key] is value

    def test_validation_missing_columns(self):
        import osm_polygon_sentence_relevance.contracts.errors as errs
        import osm_polygon_sentence_relevance.contracts.schemas as sch

        reduced = pa.schema(
            [f for f in sch.POLYGONS_SCHEMA if f.name != "lat"],
            metadata=sch.POLYGONS_SCHEMA.metadata,
        )
        with pytest.raises(errs.MissingColumnsError):
            sch.validate_table_schema("polygons", reduced)

    def test_validation_unknown_table(self):
        import osm_polygon_sentence_relevance.contracts.errors as errs
        import osm_polygon_sentence_relevance.contracts.schemas as sch

        with pytest.raises(errs.UnknownTableError):
            sch.validate_table_schema("does_not_exist", sch.POLYGONS_SCHEMA)
