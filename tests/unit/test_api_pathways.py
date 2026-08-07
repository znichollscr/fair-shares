"""Choosing a scenario group, and refusing to fan out.

The pathway counterpart of `test_api_allocate.py`, and it pins the same two
ideas one layer over.

`select_world_pathway` is a *lookup*: a pathway allocation needs one world
pathway, and handing a whole ensemble to something that iterates whatever it is
given is what makes a filtering step necessary later.

`allocate_pathway` refuses a list. `run_parameter_grid`, the only way to run a
pathway allocation before this module existed, crosses every parameter that is
a list -- so a caller wanting one allocation had to wrap each value in a
one-element list and check the answer's length afterwards. Worse, the four
validators that decide whether a configuration is admissible are called by
`run_parameter_grid` and *not* by `run_allocation`, so going straight to the
single-allocation function to avoid the fan-out also silently switches off the
NGHGI year guard. Both are pinned here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fair_shares.api.pathways import (
    PathwayRequest,
    ScenarioGroup,
    _validate,
    allocate_pathway,
    select_world_pathway,
)
from fair_shares.api.pipeline import ProcessedData
from fair_shares.library.exceptions import AllocationError, ConfigurationError

YEARS = [str(year) for year in range(2015, 2026)]

# Two assessments at two quantiles: enough that a caller who filtered instead
# of looking up would get more rows than it asked for.
GROUPS = [("C1", 0.5), ("C1", 0.66), ("C3", 0.5), ("1.6", 0.5)]

# A land record covering 2000-2023, which is what the Melo NGHGI dataset
# actually carries. The year bound is read from this rather than assumed.
LULUCF_YEARS = (2000, 2023)


def processed(lulucf_years=None) -> ProcessedData:
    """A `ProcessedData` carrying only what validation reads.

    The frames are empty on purpose: every test below stops in validation, and
    a fixture that also carried plausible emissions would invite the reader to
    think the numbers mattered.
    """
    return ProcessedData(
        category="all-ghg",
        parts={},
        gdp=pd.DataFrame(),
        population=pd.DataFrame(),
        gini=pd.DataFrame(),
        world_emissions={},
        rcbs=pd.DataFrame(),
        coverage=pd.DataFrame(),
        analysis_countries=frozenset(),
        lulucf_years=lulucf_years,
    )


@pytest.fixture
def scenarios() -> pd.DataFrame:
    rows = []
    for index, (assessment, quantile) in enumerate(GROUPS, start=1):
        for iso3c in ("World", "AUS"):
            rows.append(
                {
                    "climate-assessment": assessment,
                    "quantile": quantile,
                    "source": "test",
                    "iso3c": iso3c,
                    "unit": "Mt * CO2e",
                    "emission-category": "all-ghg",
                    **{year: 1000.0 * index for year in YEARS},
                }
            )
    frame = pd.DataFrame(rows)
    return frame.set_index(
        [
            "climate-assessment",
            "quantile",
            "source",
            "iso3c",
            "unit",
            "emission-category",
        ]
    )


class TestSelectingOneScenarioGroup:
    def test_it_returns_one_world_pathway(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        assert len(world.emissions) == 1
        assert float(world.emissions[YEARS[0]].iloc[0]) == 1000.0

    def test_the_quantile_is_part_of_the_key(self, scenarios):
        # Same assessment, different quantile: a lookup on the assessment alone
        # would return both and leave the caller to pick.
        first = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        second = select_world_pathway(scenarios, ScenarioGroup("C1", 0.66))
        assert float(first.emissions[YEARS[0]].iloc[0]) != float(
            second.emissions[YEARS[0]].iloc[0]
        )

    def test_it_carries_the_ensembles_own_label(self, scenarios):
        # A config names a group in text; a peak-temperature ensemble read from
        # CSV may carry it as something else. Results are stamped with the
        # ensemble's value so two callers who spelled it differently still
        # produce matching metadata.
        world = select_world_pathway(scenarios, ScenarioGroup("1.6", 0.5))
        assert world.label == "1.6"

    def test_a_missing_group_says_what_there_is(self, scenarios):
        with pytest.raises(ConfigurationError, match="C1 q0.5"):
            select_world_pathway(scenarios, ScenarioGroup("C9", 0.5))

    def test_a_malformed_ensemble_is_named(self):
        with pytest.raises(ConfigurationError, match="missing index level"):
            select_world_pathway(
                pd.DataFrame({"2020": [1.0]}, index=pd.Index(["x"], name="iso3c")),
                ScenarioGroup("C1", 0.5),
            )


class TestItWillNotFanOut:
    """One request describes one allocation, and the call says so."""

    @staticmethod
    def _request(**parameters):
        return PathwayRequest(
            approach="equal-per-capita",
            parameters={"first_allocation_year": 2015, **parameters},
            scenario_group=ScenarioGroup("C1", 0.5),
        )

    def test_a_list_parameter_is_refused(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        with pytest.raises(ConfigurationError, match="are lists"):
            allocate_pathway(
                processed(LULUCF_YEARS),
                PathwayRequest(
                    approach="equal-per-capita",
                    parameters={"first_allocation_year": [2015, 2020]},
                    scenario_group=ScenarioGroup("C1", 0.5),
                ),
                world,
                category="all-ghg",
            )

    def test_a_budget_approach_is_refused(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        with pytest.raises(ConfigurationError, match="cumulative budget"):
            allocate_pathway(
                processed(LULUCF_YEARS),
                PathwayRequest(
                    approach="equal-per-capita-budget",
                    parameters={"allocation_year": 2020},
                    scenario_group=ScenarioGroup("C1", 0.5),
                ),
                world,
                category="all-ghg",
            )


class TestTheValidationSurvives:
    """The checks `run_allocation` does not do, done here.

    Each of these would pass straight through to the allocator if this module
    simply called `run_allocation` -- which is the trap that makes avoiding
    `run_parameter_grid` dangerous rather than merely tidy.
    """

    def test_a_year_before_the_land_record_is_refused(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        with pytest.raises(AllocationError, match="is before 2000"):
            allocate_pathway(
                processed(LULUCF_YEARS),
                PathwayRequest(
                    approach="per-capita-adjusted",
                    parameters={
                        "first_allocation_year": 2020,
                        "pre_allocation_responsibility_year": 1990,
                    },
                    scenario_group=ScenarioGroup("C1", 0.5),
                ),
                world,
                # Contains LULUCF, so the land record's range bounds it.
                category="all-ghg",
            )

    def test_the_bound_comes_from_the_data_not_a_constant(self):
        # 1990 is refused against a record starting in 2000 and accepted
        # against one starting in 1980 — the same request, the same category,
        # a different land source. The bound used to be the constant 2000
        # whatever the run had loaded.
        #
        # `_validate` rather than `allocate_pathway`, because the claim is
        # about what is admissible; going further would need real emissions
        # for an allocation whose numbers this test says nothing about.
        request = PathwayRequest(
            approach="per-capita-adjusted",
            parameters={
                "first_allocation_year": 2020,
                "pre_allocation_responsibility_year": 1990,
            },
            scenario_group=ScenarioGroup("C1", 0.5),
        )
        with pytest.raises(AllocationError, match="is before 2000"):
            _validate(request, "all-ghg", None, (2000, 2023))
        _validate(request, "all-ghg", None, (1980, 2023))

    def test_an_uncorrected_run_is_not_bounded(self):
        # No LULUCF source means no NGHGI correction, so there is no land
        # record for a year to be "before". This used to raise against a
        # hardcoded 2000, pointing at a parameter that was not the problem.
        _validate(
            PathwayRequest(
                approach="per-capita-adjusted",
                parameters={
                    "first_allocation_year": 2015,
                    "pre_allocation_responsibility_year": 1990,
                },
                scenario_group=ScenarioGroup("C1", 0.5),
            ),
            "all-ghg",
            None,
            None,
        )

    def test_a_category_without_lulucf_is_never_bounded(self):
        # `all-ghg-ex-co2-lulucf` contains no LULUCF CO2, so the question does
        # not arise whichever path the run took.
        _validate(
            PathwayRequest(
                approach="equal-per-capita",
                parameters={"first_allocation_year": 2015},
                scenario_group=ScenarioGroup("C1", 0.5),
            ),
            "all-ghg-ex-co2-lulucf",
            None,
            None,
        )

    def test_the_budget_year_parameter_is_refused(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        with pytest.raises(AllocationError, match="first_allocation_year"):
            allocate_pathway(
                processed(LULUCF_YEARS),
                PathwayRequest(
                    approach="equal-per-capita",
                    parameters={"allocation_year": 2020},
                    scenario_group=ScenarioGroup("C1", 0.5),
                ),
                world,
                category="all-ghg",
            )

    def test_an_allocation_year_after_harmonisation_is_refused(self, scenarios):
        world = select_world_pathway(scenarios, ScenarioGroup("C1", 0.5))
        with pytest.raises(AllocationError):
            allocate_pathway(
                processed(LULUCF_YEARS),
                PathwayRequest(
                    approach="equal-per-capita",
                    parameters={"first_allocation_year": 2025},
                    scenario_group=ScenarioGroup("C1", 0.5),
                ),
                world,
                category="all-ghg-ex-co2-lulucf",
                harmonisation_year=2020,
            )
