"""Structural contract for the operator configuration boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from osm_polygon_sentence_relevance.operator import config

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "src" / "osm_polygon_sentence_relevance" / "operator"
INTERNAL = OPERATOR / "_config"

EXPECTED_EXPORTS = {
    "Scope",
    "Stage",
    "Grid5000Requirements",
    "RunIdentity",
    "OperatorConfig",
    "DATA_ROOT",
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_SPLIT_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LLAMA_PARALLEL",
    "DEFAULT_LLAMA_PER_SLOT_CONTEXT",
    "DEFAULT_ROW_LIMIT",
    "DEFAULT_SAMPLING_TARGET",
    "DEFAULT_SAMPLING_SEED",
    "DEFAULT_SAMPLING_H3_RESOLUTION",
    "SAMPLING_VERSION",
    "DEFAULT_LABEL_MODEL_REPO_ID",
    "DEFAULT_LABEL_MODEL_REVISION",
    "DEFAULT_LABEL_MODEL_FILE",
    "DEFAULT_LABEL_MODEL_FILE_SHA256",
    "DEFAULT_TOKENIZER_REPO_ID",
    "DEFAULT_TOKENIZER_REVISION",
    "SUPPORTED_LLAMA_PARALLEL",
}


def test_internal_configuration_modules_exist() -> None:
    assert (INTERNAL / "__init__.py").is_file()
    assert (INTERNAL / "validation.py").is_file()
    assert (INTERNAL / "requirements.py").is_file()
    assert (INTERNAL / "models.py").is_file()


def test_public_config_is_a_pure_explicit_facade() -> None:
    tree = ast.parse((OPERATOR / "config.py").read_text())
    assert ast.get_docstring(tree, clean=False) is not None, (
        "config facade requires a module docstring"
    )

    assignments = 0
    for node in tree.body[1:]:
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "*" for alias in node.names)
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            assignments += 1
            continue
        raise AssertionError(f"unexpected facade statement: {type(node).__name__}")
    assert assignments == 1


def test_public_config_exports_are_unchanged() -> None:
    assert set(config.__all__) == EXPECTED_EXPORTS
    assert len(config.__all__) == len(EXPECTED_EXPORTS)


def test_configuration_boundary_files_stay_focused() -> None:
    paths = [OPERATOR / "config.py", *INTERNAL.glob("*.py")]
    oversized = {
        path.name: len(path.read_text().splitlines())
        for path in paths
        if len(path.read_text().splitlines()) > 500
    }
    assert oversized == {}
