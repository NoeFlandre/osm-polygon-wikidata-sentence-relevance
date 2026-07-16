"""Compatibility facade: ``osm_polygon_sentence_relevance.cli``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.application.cli`. Import from there in
new code; this module re-exports the stable public symbols so existing
imports keep working.
"""

from osm_polygon_sentence_relevance.application.cli import main
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline

__all__ = ["main", "run_pipeline"]
