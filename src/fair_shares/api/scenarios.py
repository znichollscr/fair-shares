"""Scenario-derived corrections to a published remaining carbon budget.

Ported from the part of notebook 104 that budget runs actually consume. That
notebook is 1,200 lines, most of which harmonises and aggregates scenario
pathways for *pathway* targets; a budget run needs only two things from it, and
this module produces exactly those:

* per climate target, a handful of scalars -- the cumulative bookkeeping-model
  LULUCF flux to net zero, the inventory-vs-bookkeeping convention gap, and the
  net-zero year;
* per climate target, the year-by-year median land flux, used when a budget's
  baseline year is later than the 2020 the corrections are anchored to.

**Why a published budget needs correcting at all.** Remaining carbon budgets
are published for *total* CO2 against a bookkeeping definition of the land
sink. An allocation of fossil CO2 has to remove the land flux the scenarios
assume, and an allocation reported against national inventories has to be
re-expressed in that convention. Both corrections are integrals over the
scenario ensemble, and both are medians across scenarios rather than a single
representative pathway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fair_shares.api.sources import Inputs
from fair_shares.library.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

# AR6 scenario categories, and the climate target each stands for. The label
# encodes a temperature limit and an exceedance probability.
AR6_CATEGORY_TO_TARGET = {"C1": "1.5p50", "C2": "2p83", "C3": "2p66"}

# The scenario variables the corrections are computed from. `Direct` is the
# bookkeeping land flux; `Indirect` is the part of the land sink that
# inventories count and bookkeeping models do not.
SCENARIO_VARIABLES = {
    "CO2": "AR6 Reanalysis|OSCARv3.2|Emissions|CO2",
    "AFOLU_direct": "AR6 Reanalysis|OSCARv3.2|Emissions|CO2|AFOLU|Direct",
    "AFOLU_indirect": "AR6 Reanalysis|OSCARv3.2|Emissions|CO2|AFOLU|Indirect",
}

# Budgets are published against this baseline, so every integral starts here.
BUDGET_BASELINE_YEAR = 2020

# A scenario that has not crossed zero by the end of the horizon is treated as
# reaching net zero at the horizon rather than never, so it still contributes
# to the median instead of being dropped.
HORIZON_YEAR = 2100


@dataclass(frozen=True)
class TargetAdjustments:
    """The scenario-derived corrections for one climate target.

    Attributes
    ----------
    bm_lulucf_cumulative_median
        Median cumulative bookkeeping land flux from the baseline year to net
        zero, in Mt CO2. Negative is a sink.
    convention_gap_median
        Median cumulative difference between inventory and bookkeeping
        accounting over the same window, in Mt CO2. Zero when no inventory
        record was supplied, which is correct for fossil-only budgets.
    nz_year_median, nz_year_min, nz_year_max
        Net-zero year statistics across the ensemble.
    n_scenarios
        How many scenarios the medians are over.
    n_reaching_nz
        How many actually cross zero before the horizon.
    """

    bm_lulucf_cumulative_median: float
    convention_gap_median: float
    nz_year_median: int
    nz_year_min: int
    nz_year_max: int
    n_scenarios: int
    n_reaching_nz: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the adjustments as the mapping the budget step expects.

        Returns
        -------
        dict
            Rounded to 0.1 Mt, which is the precision the published budgets
            themselves carry.
        """
        return {
            "bm_lulucf_cumulative_median": round(self.bm_lulucf_cumulative_median, 1),
            "convention_gap_median": round(self.convention_gap_median, 1),
            "nz_year_median": self.nz_year_median,
            "nz_year_min": self.nz_year_min,
            "nz_year_max": self.nz_year_max,
            "n_scenarios": self.n_scenarios,
            "n_reaching_nz": self.n_reaching_nz,
        }


@dataclass(frozen=True)
class ScenarioAdjustments:
    """Everything the budget step needs from the scenario ensemble.

    Attributes
    ----------
    by_target
        Climate target label to its adjustments.
    lulucf_shift
        Climate target label to a single-row frame of the year-by-year median
        land flux, used to re-anchor a budget whose baseline is later than
        `BUDGET_BASELINE_YEAR`.
    """

    by_target: dict[str, TargetAdjustments]
    lulucf_shift: dict[str, pd.DataFrame]

    def scalars(self) -> dict[str, dict[str, float | int]]:
        """Return the plain mapping the budget step consumes.

        Returns
        -------
        dict
            Climate target label to its adjustment scalars.
        """
        return {target: values.as_dict() for target, values in self.by_target.items()}


def load_scenario_frames(inputs: Inputs) -> dict[str, pd.DataFrame]:
    """Read the world-level scenario variables the corrections are built from.

    Parameters
    ----------
    inputs
        The run's inputs.

    Returns
    -------
    dict
        Variable shorthand to its frame, each carrying ``Model``, ``Scenario``,
        ``climate-assessment`` and one string column per year.

    Raises
    ------
    DataProcessingError
        If a required variable is absent from the ensemble.
    """
    from fair_shares.library.utils.dataframes import process_iamc_zip

    path = inputs.path("scenarios")
    world_key = inputs.parameters("scenarios").get("world_key", "World")
    logger.info("reading scenario ensemble from %s", path)

    frame = process_iamc_zip(path)
    frame = frame.rename(columns={"Category": "climate-assessment"})
    frame = frame[frame["climate-assessment"].isin(AR6_CATEGORY_TO_TARGET)].copy()
    frame["climate-assessment"] = frame["climate-assessment"].map(
        AR6_CATEGORY_TO_TARGET
    )

    frames: dict[str, pd.DataFrame] = {}
    for shorthand, variable in SCENARIO_VARIABLES.items():
        selected = frame[
            (frame["Variable"] == variable) & (frame["Region"] == world_key)
        ].copy()
        if selected.empty:
            raise DataProcessingError(
                f"the scenario ensemble carries no rows for {variable!r} at "
                f"region {world_key!r}."
            )
        frames[shorthand] = selected
    return frames


def _year_columns(frame: pd.DataFrame) -> list[str]:
    return sorted((c for c in frame.columns if str(c).isdigit()), key=int)


def _net_zero_years(total_co2: pd.DataFrame, years: list[str]) -> pd.Series:
    """Return each scenario's first year of non-positive total CO2."""
    indexed = total_co2.set_index(["Model", "Scenario"])[years]
    # `idxmax` on the boolean picks the first True; a row that never crosses
    # gives False everywhere and idxmax returns the first column, so those are
    # replaced with the horizon.
    crossed = indexed.le(0)
    first = crossed.idxmax(axis=1).astype(int)
    return first.where(crossed.any(axis=1), HORIZON_YEAR).astype(int)


def compute_adjustments(
    frames: dict[str, pd.DataFrame],
    world_lulucf_inventory: pd.DataFrame | None = None,
) -> ScenarioAdjustments:
    """Compute the budget corrections for every climate target.

    Parameters
    ----------
    frames
        What `load_scenario_frames` returned.
    world_lulucf_inventory
        The world inventory-convention land flux, as
        :func:`fair_shares.api.lulucf.world_lulucf` returns. Required only for
        categories reported against inventories; when omitted the convention
        gap is zero, which is correct for a fossil-only budget.

    Returns
    -------
    ScenarioAdjustments
        Scalars and shift timeseries, keyed by climate target label.
    """
    from fair_shares.library.utils.data.nghgi import compute_cumulative_emissions

    direct_all = frames["AFOLU_direct"]
    total_all = frames["CO2"]
    indirect_all = frames["AFOLU_indirect"]
    years = _year_columns(direct_all)

    splice_year: int | None = None
    if world_lulucf_inventory is not None:
        splice_year = max(
            int(c) for c in world_lulucf_inventory.columns if str(c).isdigit()
        )

    by_target: dict[str, TargetAdjustments] = {}
    shifts: dict[str, pd.DataFrame] = {}

    for target in AR6_CATEGORY_TO_TARGET.values():
        direct = direct_all[direct_all["climate-assessment"] == target]
        total = total_all[total_all["climate-assessment"] == target]
        if direct.empty or total.empty:
            logger.debug("%s: no scenarios, skipping", target)
            continue

        net_zero = _net_zero_years(total, years)
        direct_indexed = direct.set_index(["Model", "Scenario"])[years]
        shared = direct_indexed.index.intersection(net_zero.index)

        cumulative = [
            float(
                direct_indexed.loc[
                    key,
                    [
                        str(y)
                        for y in range(BUDGET_BASELINE_YEAR, net_zero[key] + 1)
                        if str(y) in years
                    ],
                ].sum()
            )
            for key in shared
        ]
        bm_median = float(pd.Series(cumulative).median()) if cumulative else 0.0

        gap_median = 0.0
        if world_lulucf_inventory is not None and splice_year is not None:
            indirect = indirect_all[indirect_all["climate-assessment"] == target]
            indirect_indexed = indirect.set_index(["Model", "Scenario"])[years]
            gaps = []
            for key in shared.intersection(indirect_indexed.index):
                zero_year = int(net_zero[key])
                # Up to the end of the inventory record the gap is observed;
                # after it, only the indirect flux differs, because the direct
                # flux appears in both conventions and cancels.
                observed_end = min(splice_year, zero_year)
                inventory = compute_cumulative_emissions(
                    world_lulucf_inventory, BUDGET_BASELINE_YEAR, observed_end
                )
                observed_columns = [
                    str(y)
                    for y in range(BUDGET_BASELINE_YEAR, observed_end + 1)
                    if str(y) in years
                ]
                direct_observed = (
                    float(direct_indexed.loc[key, observed_columns].sum())
                    if observed_columns
                    else 0.0
                )
                future = 0.0
                if zero_year > splice_year:
                    future_columns = [
                        str(y)
                        for y in range(splice_year + 1, zero_year + 1)
                        if str(y) in years
                    ]
                    if future_columns:
                        future = float(indirect_indexed.loc[key, future_columns].sum())
                gaps.append(inventory - direct_observed + future)
            gap_median = float(pd.Series(gaps).median()) if gaps else 0.0

        by_target[target] = TargetAdjustments(
            bm_lulucf_cumulative_median=bm_median,
            convention_gap_median=gap_median,
            nz_year_median=int(net_zero.median()),
            nz_year_min=int(net_zero.min()),
            nz_year_max=int(net_zero.max()),
            n_scenarios=len(net_zero),
            n_reaching_nz=int((net_zero < HORIZON_YEAR).sum()),
        )

        shifts[target] = pd.DataFrame(
            [direct[years].median(axis=0).to_numpy()],
            columns=years,
            index=pd.Index([f"gidden_direct_{target}"], name="source"),
        )

        logger.info(
            "%s: n=%d, median net zero %d, BM LULUCF %.0f Mt, gap %.0f Mt",
            target,
            len(net_zero),
            int(net_zero.median()),
            bm_median,
            gap_median,
        )

    return ScenarioAdjustments(by_target=by_target, lulucf_shift=shifts)
