"""The anchor filter, and why `allocate` applies it for you.

`run_all_allocations` iterates every remaining-budget row in the data it is
handed. A caller who hands it the whole table gets every climate target's rows
back under whatever single label it asked for -- no error, just a result that
means something other than what was requested. That is not a hypothetical: a
downstream consumer of this library hit it, diagnosed it, and had to carry its
own filter until this one existed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fair_shares.api.allocate import ClimateTarget, filter_to_anchor
from fair_shares.library.exceptions import ConfigurationError

# The eight rows a co2-ffi budget table actually carries, which is the reason
# the filter has to exist.
BUDGET_ROWS = [
    ("lamboll_2023", "1.5C", 0.50),
    ("lamboll_2023", "2C", 0.66),
    ("forster_2024", "1.5C", 0.50),
    ("forster_2024", "2C", 0.66),
    ("forster_2024", "2C", 0.83),
    ("ar6_2020", "1.5C", 0.50),
    ("ar6_2020", "2C", 0.66),
    ("ar6_2020", "2C", 0.83),
]


@pytest.fixture
def budgets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "climate-assessment": limit,
                "quantile": quantile,
                "emission-category": "co2-ffi",
                "rcb_2020_nghgi_mt": 100.0 * index,
            }
            for index, (source, limit, quantile) in enumerate(BUDGET_ROWS, start=1)
        ]
    )


@pytest.fixture
def loaded(budgets) -> dict:
    # `scenarios_data` is empty for a pure budget run.
    return {
        "rcbs_data": {"co2-ffi": budgets},
        "scenarios_data": {},
        "country_gdp_df": pd.DataFrame({"gdp": [1.0]}),
    }


def target(limit="1.5C", quantile=0.5, reference="ar6_2020") -> ClimateTarget:
    return ClimateTarget(limit=limit, quantile=quantile, reference=reference)


class TestFilterToAnchor:
    def test_keeps_exactly_one_row(self, loaded):
        kept = filter_to_anchor(loaded, target())["rcbs_data"]["co2-ffi"]
        assert len(kept) == 1
        assert kept.iloc[0]["source"] == "ar6_2020"
        assert kept.iloc[0]["climate-assessment"] == "1.5C"
        assert kept.iloc[0]["quantile"] == 0.5

    def test_distinguishes_references_sharing_a_limit(self, loaded):
        # 1.5C p50 exists under three assessments. Collapsing them would make
        # three different budgets indistinguishable in the output.
        values = {
            reference: filter_to_anchor(loaded, target(reference=reference))[
                "rcbs_data"
            ]["co2-ffi"].iloc[0]["rcb_2020_nghgi_mt"]
            for reference in ("lamboll_2023", "forster_2024", "ar6_2020")
        }
        assert len(set(values.values())) == 3

    def test_distinguishes_quantiles(self, loaded):
        kept = filter_to_anchor(loaded, target("2C", 0.83, "ar6_2020"))["rcbs_data"][
            "co2-ffi"
        ]
        assert len(kept) == 1
        assert kept.iloc[0]["quantile"] == 0.83

    def test_does_not_mutate_the_input(self, loaded):
        # One preprocessing serves every climate target in a grid, so filtering
        # in place would corrupt every request after the first.
        filter_to_anchor(loaded, target())
        assert len(loaded["rcbs_data"]["co2-ffi"]) == len(BUDGET_ROWS)

    def test_shares_the_frames_it_does_not_touch(self, loaded):
        filtered = filter_to_anchor(loaded, target())
        assert filtered["country_gdp_df"] is loaded["country_gdp_df"]

    def test_a_missing_anchor_says_what_was_available(self, loaded):
        with pytest.raises(ConfigurationError) as excinfo:
            filter_to_anchor(loaded, target(reference="nonesuch"))
        message = str(excinfo.value)
        assert "nonesuch" in message
        assert "1.5C p50 ar6_2020" in message

    def test_a_numeric_quantile_is_required_to_match(self, loaded):
        # `parse_rcb_scenario` returns the quantile as text, and every consumer
        # only ever saw a float because the table reached them through a CSV.
        # Building it in memory made the string visible and anchors stopped
        # matching, so the type is part of the contract now.
        assert loaded["rcbs_data"]["co2-ffi"]["quantile"].dtype.kind == "f"
