from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.analytics import (
    h3_sentence_distribution,
    render_h3_sentence_distribution,
)


def _table() -> pa.Table:
    return pa.table(
        {
            "lat": [48.85, 48.86, -34.6, None],
            "lon": [2.35, 2.36, -58.4, None],
        }
    )


def test_h3_distribution_counts_rows_and_missing_coordinates() -> None:
    pytest.importorskip("h3")
    cells, missing = h3_sentence_distribution(_table(), resolution=3)

    assert sum(cells.values()) == 3
    assert len(cells) == 2
    assert missing == 1


def test_h3_distribution_rejects_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="H3 resolution"):
        h3_sentence_distribution(_table(), resolution=16)


def test_h3_map_is_a_png_and_summary_matches_counts(tmp_path) -> None:
    pytest.importorskip("h3")
    path = tmp_path / "h3_sentence_distribution.png"
    summary = render_h3_sentence_distribution(
        _table(), path, resolution=3, scope_label="Worldwide"
    )

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert summary["sentence_count"] == 3
    assert summary["missing_coordinate_count"] == 1
    assert summary["resolution"] == 3
