"""Ownership tests for the public dataset-publication surface."""


def test_dataset_card_facade_reexports_focused_implementations() -> None:
    from osm_polygon_sentence_relevance.output import dataset_card, plots
    from osm_polygon_sentence_relevance.output._card import rendering, statistics

    assert dataset_card.DatasetStatistics is statistics.DatasetStatistics
    assert dataset_card.compute_statistics is statistics.compute_statistics
    assert dataset_card.render_dataset_card is rendering.render_dataset_card
    assert (
        dataset_card.render_dataset_card_from_profile
        is rendering.render_dataset_card_from_profile
    )
    assert callable(plots.render_geographic_coverage_png)
    assert callable(plots.render_language_distribution_png)


def test_profile_keeps_plot_compatibility_exports() -> None:
    from osm_polygon_sentence_relevance.output import plots, profile

    assert (
        profile.render_geographic_coverage_png is plots.render_geographic_coverage_png
    )
    assert (
        profile.render_language_distribution_png
        is plots.render_language_distribution_png
    )
