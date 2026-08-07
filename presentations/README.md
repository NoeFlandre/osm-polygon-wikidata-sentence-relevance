# Project presentations

The decks are authored in Markdown with [Colloquium](https://github.com/natolambert/colloquium)
and built into static HTML for GitHub Pages:

The layout and authoring conventions follow the read-only
[`NoeFlandre/slides-colloquium`](https://github.com/NoeFlandre/slides-colloquium)
reference repository.

- [Dataset overview: V1 Afghanistan and V2 worldwide](afghanistan-dataset-overview/index.html)
- [Codebase overview](codebase-overview/index.html)

The source files are kept in [`source/`](source/). Build locally with
`colloquium build source/dataset-overview.md -o /tmp/dataset-deck` and
`colloquium build source/codebase-overview.md -o /tmp/codebase-deck`, then
copy each generated HTML file to its presentation directory as `index.html`.

The same decks are hosted on the project's
[GitHub Pages site](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/presentations/).
