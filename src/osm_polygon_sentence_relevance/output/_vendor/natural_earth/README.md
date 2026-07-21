# Natural Earth country outline data

This directory contains pinned, attributed geographic data used by the
publication pipeline to render the Afghanistan outline on the dataset
card's geographic-coverage asset.

## `afghanistan_outline.geojson`

Single-country subset of Natural Earth 1:110m Cultural Vectors -
Admin 0 - Countries. The full Natural Earth GeoJSON release is mirrored
on GitHub at <https://github.com/nvkelso/natural-earth-vector>; we
extract the single feature whose ``ADMIN`` property is
``"Afghanistan"`` and store it here so the publication pipeline does not
depend on outbound network access at render time.

| Field | Value |
| --- | --- |
| Source dataset | Natural Earth 1:110m Admin 0 Countries |
| Source URL | https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/ |
| Source mirror | https://github.com/nvkelso/natural-earth-vector |
| License | Public Domain (Creative Commons CC0 1.0) |
| License URL | https://creativecommons.org/publicdomain/zero/1.0/ |
| Scale | 1:110m |
| Original SHA-256 | `6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f` |
| Subset SHA-256 | `4fb163ae405f8be649f17e0d8ba83e0402f561268267512536d3f04cc4102feb` |
| Extraction date | 2026-07-21 |

The subset SHA-256 is checked at render time; the publication
validator must reject any modification of this file unless the SHA is
intentionally updated (which then requires updating the constant in
`src/osm_polygon_sentence_relevance/output/profile.py`).

## Attribution

Natural Earth is free for use in any type of project (commercial or
non-commercial) without restriction. The full set of Natural Earth
contributors is listed on the Natural Earth website. We reproduce
the attribution on the dataset card so downstream users can audit the
source.
