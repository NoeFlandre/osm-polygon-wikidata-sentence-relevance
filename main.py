"""Minimal entry point — Phase 1 stub.

Prints the package and pipeline version.  A real CLI will be added in a
later phase.
"""

from osm_polygon_sentence_relevance import __version__
from osm_polygon_sentence_relevance.constants import PIPELINE_VERSION


def main() -> None:
    print(f"osm-polygon-sentence-relevance  package={__version__}  pipeline={PIPELINE_VERSION}")
    print("Phase 1: schema contracts and project foundation only.")


if __name__ == "__main__":
    main()
