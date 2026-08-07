"""How a single Gini value is chosen, and why the rule belongs to the dataset.

The catalogue used to offer ``gini: unu-wider-2025`` and `load_gini` would then
refuse it: the loader parsed a World Bank CSV and accepted only
``selection: latest-available``, while that source is UNU-WIDER's WIID
spreadsheet. The one gini dataset several runs actually use was one the loader
could not open.

What makes WIID different is not the file format, it is that it publishes
**many observations per country and year**, graded for quality. So it needs a
rule the World Bank data has no use for -- prefer quality, then recency -- and
that rule needs a tie-break, because the same country and year can carry half a
dozen values that disagree by twenty points.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fair_shares.api import inputs as inputs_module
from fair_shares.api.sources import Inputs, SourceSelection
from fair_shares.library.exceptions import ConfigurationError

SOURCES = SourceSelection(
    emissions="primap-202503",
    gdp="wdi-2025",
    population="un-owid-2025",
    gini="unu-wider-2025",
)

# Deliberately out of order, and deliberately with a duplicate country-year:
# AUS 2010 carries two high-quality values, and the rule has to pick one.
WIID_ROWS = [
    ("AUS", 2005, 30.0, "High"),
    ("AUS", 2010, 40.0, "High"),
    ("AUS", 2010, 55.0, "High"),
    ("AUS", 2020, 99.0, "Low"),
    ("NZL", 2001, 25.0, "Low"),
    ("NZL", 2015, 35.0, "Average"),
]


@pytest.fixture
def catalogue(tmp_path):
    path = tmp_path / "wiid.xlsx"
    pd.DataFrame(WIID_ROWS, columns=["c3", "year", "gini", "quality"]).to_excel(
        path, index=False
    )
    return {
        # `Inputs` validates every selected source against the catalogue, so
        # the three this test does not exercise still have to be declared.
        "emissions": {"primap-202503": {}},
        "gdp": {"wdi-2025": {}},
        "population": {"un-owid-2025": {}},
        "gini": {
            "unu-wider-2025": {
                "path": str(path),
                "data_parameters": {"selection": "latest-high-quality"},
            }
        },
    }


def _load(catalogue, tmp_path, countries={"AUS", "NZL"}):
    inputs = Inputs(sources=SOURCES, data_dir=tmp_path, catalogue=catalogue)
    return inputs_module.load_gini(inputs, countries)["gini"].droplevel("unit")


class TestTheWiidSelectionRule:
    def test_quality_beats_recency(self, catalogue, tmp_path):
        # AUS has a 2020 observation, but it is Low; the rule takes the latest
        # *High* one instead. Taking the most recent regardless of grade would
        # silently prefer weaker evidence.
        assert _load(catalogue, tmp_path)["AUS"] == pytest.approx(0.40)

    def test_it_falls_back_when_nothing_is_high_quality(self, catalogue, tmp_path):
        # NZL has no High row at all, so the rule takes its latest of any grade
        # rather than dropping the country.
        assert _load(catalogue, tmp_path)["NZL"] == pytest.approx(0.35)

    def test_a_tie_is_broken_by_file_order(self, catalogue, tmp_path):
        # AUS 2010 carries 40.0 and 55.0, both High. Nothing in the data ranks
        # them, so the rule takes the first as published -- arbitrary, but
        # deterministic, and the same row the notebook pipeline's `idxmax`
        # picks.
        assert _load(catalogue, tmp_path)["AUS"] == pytest.approx(0.40)

    def test_values_are_converted_to_a_fraction(self, catalogue, tmp_path):
        # Published on 0-100; the allocators expect 0-1.
        assert (_load(catalogue, tmp_path) <= 1.0).all()

    def test_only_the_countries_asked_for_come_back(self, catalogue, tmp_path):
        assert set(_load(catalogue, tmp_path, {"AUS"}).index) == {"AUS"}

    def test_a_year_window_still_applies(self, catalogue, tmp_path):
        catalogue["gini"]["unu-wider-2025"]["data_parameters"]["year_window"] = [
            2000,
            2008,
        ]
        assert _load(catalogue, tmp_path)["AUS"] == pytest.approx(0.30)


class TestTheRuleHasToMatchTheDataset:
    def test_an_unknown_selection_is_refused(self, catalogue, tmp_path):
        catalogue["gini"]["unu-wider-2025"]["data_parameters"]["selection"] = "newest"
        with pytest.raises(ConfigurationError, match="unknown gini selection"):
            _load(catalogue, tmp_path)

    def test_a_missing_selection_is_refused(self, catalogue, tmp_path):
        catalogue["gini"]["unu-wider-2025"]["data_parameters"] = {}
        with pytest.raises(ConfigurationError, match="unknown gini selection"):
            _load(catalogue, tmp_path)

    def test_a_file_that_is_not_wiid_is_named(self, catalogue, tmp_path):
        path = tmp_path / "not-wiid.xlsx"
        pd.DataFrame({"country": ["AUS"], "value": [30.0]}).to_excel(path, index=False)
        catalogue["gini"]["unu-wider-2025"]["path"] = str(path)
        with pytest.raises(Exception, match="does not look like a WIID extract"):
            _load(catalogue, tmp_path)
