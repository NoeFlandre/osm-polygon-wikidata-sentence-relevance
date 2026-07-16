"""Structural guarantees for the Q4 contracts layout.

These are intentionally static-analysis checks (AST) so regressions in
production import direction or facade purity fail fast and clearly.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src" / "osm_polygon_sentence_relevance"

# Root modules that MUST remain thin compatibility facades.
FACADE_MODULES = [
    "constants",
    "errors",
    "schemas",
    "settings",
    "acquisition",
    "cli",
    "discovery",
    "exporter",
    "finalization",
    "loading",
    "pipeline",
    "preprocessing",
    "sat_adapter",
    "segmentation",
    "sentence_table",
]

# Canonical implementation is never behind a root facade in production code.
_CANONICAL_ROOT = "osm_polygon_sentence_relevance"


def _all_source_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*.py"):
        # exclude the facades themselves and tests; we check production imports.
        rel = p.relative_to(ROOT)
        if rel.parts[0] in ("tests",):
            continue
        files.append(p)
    return files


def _module_imports_from(node: ast.AST) -> list[str]:
    names: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.ImportFrom) and n.module:
            names.append(n.module)
        elif isinstance(n, ast.Import):
            for alias in n.names:
                names.append(alias.name)
    return names


class TestNoFacadeImportsInProduction:
    """Production code must import via canonical domain paths, never facades."""

    @pytest.mark.parametrize("forbidden", FACADE_MODULES)
    def test_production_never_imports_root_facade(self, forbidden: str) -> None:
        """No production module (anywhere under the package) may import any
        root compatibility facade. The facade files themselves are excluded
        from the scan so that their canonical-target imports are not
        mistaken for facade imports."""
        facade_path = (ROOT / f"{forbidden}.py").resolve()
        offenders: list[str] = []
        for f in _all_source_files():
            if f.resolve() == facade_path:
                continue
            tree = ast.parse(f.read_text())
            names = _module_imports_from(tree)
            for mod in names:
                if mod == f"osm_polygon_sentence_relevance.{forbidden}":
                    offenders.append(f"{f.relative_to(ROOT)} -> {mod}")
        assert not offenders, (
            f"production modules import root facade '{forbidden}': {offenders}"
        )


def _imported_names(tree: ast.Module) -> set[str]:
    """Return the set of public names brought into scope by import statements."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name.split(".")[0]
                names.add(bound)
    return names


class TestFacadePurity:
    """Each root compatibility facade is structurally pure.

    A facade may contain only:
    - exactly one leading module docstring;
    - import statements;
    - exactly one ``__all__`` assignment (a list/tuple literal of strings);
    and nothing else. ``__all__`` names must exactly match the imported
    public names, with no duplicates and no wildcard imports.
    """

    @pytest.mark.parametrize("name", FACADE_MODULES)
    def test_facade_structure(self, name: str) -> None:
        path = ROOT / f"{name}.py"
        assert path.is_file(), f"expected facade module {path}"
        source = path.read_text()
        tree = ast.parse(source)
        body = tree.body

        # --- exactly one leading module docstring ---
        assert body, f"{name}.py: module must have at least one statement"
        assert isinstance(body[0], ast.Expr), (
            f"{name}.py: first statement must be a module docstring"
        )
        assert isinstance(body[0].value, ast.Constant), (
            f"{name}.py: first statement must be a string constant (docstring)"
        )
        assert isinstance(body[0].value.value, str), (
            f"{name}.py: first statement must be a string constant (docstring)"
        )

        all_count = 0
        for node in body[1:]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                if targets and all(t.id == "__all__" for t in targets):
                    all_count += 1
                    assert isinstance(node.value, (ast.List, ast.Tuple)), (
                        f"{name}.py: __all__ must be a list/tuple literal"
                    )
                    continue
                raise AssertionError(
                    f"{name}.py: illegal assignment (only __all__ allowed): "
                    f"{ast.dump(node)}"
                )
            raise AssertionError(
                f"{name}.py: illegal statement '{type(node).__name__}' "
                f"(only docstring, imports, and __all__ allowed)"
            )

        assert all_count == 1, (
            f"{name}.py: expected exactly one __all__ assignment, found {all_count}"
        )

    @pytest.mark.parametrize("name", FACADE_MODULES)
    def test_no_wildcard_imports(self, name: str) -> None:
        path = ROOT / f"{name}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "*", (
                        f"{name}.py: wildcard import from {node.module}"
                    )

    @pytest.mark.parametrize("name", FACADE_MODULES)
    def test_all_matches_imported_names(self, name: str) -> None:
        """``__all__`` must contain exactly the names imported by the facade,
        with no duplicates, no extras, and no omissions."""
        path = ROOT / f"{name}.py"
        tree = ast.parse(path.read_text())

        imported = _imported_names(tree)

        # Extract __all__ literal.
        all_values: list[str] = []
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                for elt in node.value.elts:
                    assert isinstance(elt, ast.Constant), (
                        f"{name}.py: __all__ entries must be string literals"
                    )
                    assert isinstance(elt.value, str), (
                        f"{name}.py: __all__ entries must be string literals"
                    )
                    all_values.append(elt.value)

        assert len(all_values) == len(set(all_values)), (
            f"{name}.py: __all__ has duplicate entries: "
            f"{[v for v in all_values if all_values.count(v) > 1]}"
        )
        assert set(all_values) == imported, (
            f"{name}.py: __all__ does not match imported names.\n"
            f"  in __all__ but not imported: {set(all_values) - imported}\n"
            f"  imported but not in __all__: {imported - set(all_values)}"
        )

    def test_facade_has_explicit_all(self) -> None:
        for name in FACADE_MODULES:
            mod = importlib.import_module(f"osm_polygon_sentence_relevance.{name}")
            assert hasattr(mod, "__all__"), f"{name} facade missing __all__"
            assert isinstance(mod.__all__, (list, tuple))
            assert mod.__all__


class TestContractsLayout:
    """The canonical contracts package exists with the expected structure."""

    def test_contracts_package_present(self) -> None:
        import osm_polygon_sentence_relevance.contracts as contracts

        assert hasattr(contracts, "constants")
        assert hasattr(contracts, "errors")
        assert hasattr(contracts, "schemas")

    def test_schemas_submodules_present(self) -> None:
        import osm_polygon_sentence_relevance.contracts.schemas as sch

        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.input"
        )
        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.pipeline"
        )
        importlib.import_module(
            "osm_polygon_sentence_relevance.contracts.schemas.registry"
        )
        # public API re-exports each schema object once
        assert sch.SCHEMA_REGISTRY is not None

    def test_settings_canonical_location(self) -> None:
        path = ROOT / "application" / "settings.py"
        assert path.is_file(), "application/settings.py must hold the implementation"


class TestEntryPointsImportable:
    """The package and every legacy facade import without side effects."""

    def test_all_facades_importable(self) -> None:
        for name in FACADE_MODULES:
            importlib.import_module(f"osm_polygon_sentence_relevance.{name}")

    def test_pkg_import_no_optional_deps(self) -> None:
        # importing the base package must not pull in wtpsplit/huggingface_hub.
        import sys

        # ensure clean import
        import osm_polygon_sentence_relevance  # noqa: F401

        assert "wtpsplit" not in sys.modules
        assert "huggingface_hub" not in sys.modules
