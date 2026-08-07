"""Running allocations against preprocessed data.

The half of the API a caller running a grid actually spends its time in.
Three things are deliberate here:

**Results know what produced them.** Every result carries the request that
made it. The notebook pipeline wrote allocations into a directory named after
a hash of the run's parameters and left the caller to reconstruct the mapping
from the path, which is fine until two runs differ in something the path does
not record.

**A budget is chosen, not filtered for.** An allocation needs one number --
the remaining budget for one climate target. `select_budget` looks that up by
key and raises if it is not there. Handing a *table* of budgets to something
that iterates whatever it is given is what made a filtering step necessary in
the first place, and filtering to avoid an unwanted loop is a workaround for a
signature that should not have taken a table.

**Shares and budgets are separate steps, because they are separate ideas.** A
method's country shares depend on population, GDP, inequality and history --
never on the size of the budget. Computing them is the expensive part;
attaching a budget is a multiplication. Splitting them means a grid of
14 methods x 5 climate targets does 14 share computations rather than 70.

**Nothing is written.** Results come back as frames for the caller to serialise
however it likes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from fair_shares.api.pathways import PathwayRequest

from fair_shares.api.pipeline import ProcessedData
from fair_shares.api.sources import SourceSelection
from fair_shares.library.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Categories whose responsibility is measured on fossil CO2 rather than on the
# category itself, because the land flux is not something a country is held
# responsible for in the same way.
_LULUCF_INCLUSIVE = frozenset({"co2", "all-ghg"})

# The column carrying the rebased budget in the table `preprocess` produces.
BUDGET_COLUMN = "rcb_2020_nghgi_mt"


@dataclass(frozen=True)
class ClimateTarget:
    """Which remaining-budget row an allocation is computed against.

    Attributes
    ----------
    limit
        The warming limit, e.g. ``"1.5C"``.
    quantile
        The exceedance probability, e.g. ``0.5``.
    reference
        Which assessment the budget comes from, e.g. ``"ar6_2020"``.
    """

    limit: str
    quantile: float
    reference: str

    def __str__(self) -> str:
        """Render the target the way the budget tables label it."""
        return f"{self.limit} p{self.quantile * 100:g} {self.reference}"


@dataclass(frozen=True)
class AllocationRequest:
    """One allocation to run: a method, its inputs, and a climate target.

    Attributes
    ----------
    approach
        An allocation registry name, e.g. ``"equal-per-capita-budget"``.
    parameters
        The approach's own keyword inputs.
    climate_target
        Which budget row to allocate. Required for budget targets; a run that
        does not allocate a budget leaves it unset.
    label
        Free-form, for the caller's own bookkeeping. Never affects the result.
    """

    approach: str
    parameters: dict[str, Any] = field(default_factory=dict)
    climate_target: ClimateTarget | None = None
    label: str | None = None


@dataclass(frozen=True)
class AllocationResult:
    """One allocation's output, tagged with what produced it.

    Attributes
    ----------
    request
        The request this answers -- an `AllocationRequest` for a budget, a
        `PathwayRequest` for a scenario pathway. Both kinds of run produce the
        same kind of result, so they share one container rather than two that
        would have to be kept in step.
    category
        The emission category allocated.
    sources
        Which input vintages it was computed from.
    allocations
        Absolute allocations, one row per (country, emission-category part).
    """

    request: AllocationRequest | PathwayRequest
    category: str
    sources: SourceSelection
    allocations: pd.DataFrame

    def countries(self) -> list[str]:
        """Return the countries this allocation covers.

        Returns
        -------
        list of str
            Sorted ISO3 codes.
        """
        return sorted(set(self.allocations["iso3c"]))


@dataclass(frozen=True)
class Budget:
    """One remaining carbon budget, and where it came from.

    Attributes
    ----------
    value_mt
        The budget in Mt CO2, rebased to 2020.
    climate_target
        The target it was selected for.
    """

    value_mt: float
    climate_target: ClimateTarget


def select_budget(rcbs: pd.DataFrame, climate_target: ClimateTarget) -> Budget:
    """Look up the remaining budget for one climate target.

    A lookup, not a filter: it returns one number, and says so if there is no
    such target rather than quietly returning fewer or more rows than the
    caller meant.

    Parameters
    ----------
    rcbs
        Rebased budgets, as `preprocess` returns them.
    climate_target
        Which one to take.

    Returns
    -------
    Budget
        The budget, tagged with the target it answers.

    Raises
    ------
    ConfigurationError
        If the table has no such target, or more than one row for it -- both
        mean the caller cannot be given the single number it asked for.
    """
    matches = rcbs[
        (rcbs["source"] == climate_target.reference)
        & (rcbs["climate-assessment"] == climate_target.limit)
        & (rcbs["quantile"] == climate_target.quantile)
    ]
    if matches.empty:
        available = sorted(
            {
                f"{row['climate-assessment']} p{row['quantile'] * 100:g} "
                f"{row['source']}"
                for _, row in rcbs.iterrows()
            }
        )
        raise ConfigurationError(
            f"no remaining carbon budget for {climate_target}. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"{len(matches)} remaining-budget rows match {climate_target}, so "
            "there is no single budget to allocate. The budget table should "
            "carry one row per (source, climate-assessment, quantile)."
        )
    return Budget(
        value_mt=float(matches.iloc[0][BUDGET_COLUMN]),
        climate_target=climate_target,
    )


def _world_frame(data: ProcessedData, category: str) -> pd.DataFrame:
    """Return a category's world total, labelled the way allocators expect.

    The allocators look for the literal ``"World"``; the emissions source uses
    its own key (PRIMAP says ``EARTH``).
    """
    frame = data.world_emissions[category]
    relabelled = frame.copy()
    relabelled.index = pd.MultiIndex.from_tuples(
        [("World", *rest) for _, *rest in frame.index], names=frame.index.names
    )
    return relabelled


def compute_shares(
    data: ProcessedData,
    approach: str,
    parameters: dict[str, Any],
    *,
    category: str,
    target: str = "rcbs",
    harmonisation_year: int | None = None,
):
    """Compute one method's country shares, before any budget is attached.

    The expensive half, and the half that does not depend on the climate
    target -- so a caller running several targets through one method computes
    this once.

    Parameters
    ----------
    data
        What `preprocess` returned.
    approach
        An allocation registry name.
    parameters
        The approach's keyword inputs.
    category
        Which emission-category part to allocate.
    target
        The target type.
    harmonisation_year
        Passed through to the allocators.

    Returns
    -------
    :
        The library's allocation result, carrying the shares.

    Raises
    ------
    ConfigurationError
        If the approach produced no result, or more than one.
    """
    from fair_shares.library.allocations.manager import run_parameter_grid

    emissions = data.parts[category]
    responsibility = (
        data.parts["co2-ffi"]
        if category in _LULUCF_INCLUSIVE and "co2-ffi" in data.parts
        else emissions
    )

    results = run_parameter_grid(
        allocations_config={approach: [dict(parameters)]},
        population_ts=data.population,
        gdp_ts=data.gdp,
        gini_s=data.gini,
        country_actual_emissions_ts=emissions,
        responsibility_emissions_ts=responsibility,
        emission_category=category,
        target_source=target,
        harmonisation_year=harmonisation_year,
        nghgi_years=data.nghgi_years,
    )
    if len(results) != 1:
        raise ConfigurationError(
            f"{approach!r} with {parameters} expanded to {len(results)} "
            "allocations; one request must describe exactly one. Parameter "
            "lists belong in separate requests."
        )
    return results[0]


def apply_budget(
    shares,
    budget: Budget,
    data: ProcessedData,
    *,
    category: str,
    context: dict[str, str],
) -> pd.DataFrame:
    """Attach a budget to already-computed shares.

    The cheap half: the budget is rebased to the allocation year and
    distributed by the shares.

    Parameters
    ----------
    shares
        What `compute_shares` returned.
    budget
        The budget to allocate.
    data
        The preprocessed data the shares came from.
    category
        The emission-category part being allocated.
    context
        Provenance stamped onto the output.

    Returns
    -------
    pandas.DataFrame
        Absolute allocations, one row per country.
    """
    from fair_shares.library.allocations.results.serializers import prepare_dataframe
    from fair_shares.library.utils.data.rcb import calculate_budget_from_rcb

    world = _world_frame(data, category)
    allocation_year = shares.parameters.get("allocation_year")

    # The published budget is stated from 2020; the allocation may start
    # elsewhere, so it is rebased before being shared out.
    total = calculate_budget_from_rcb(
        rcb_value=budget.value_mt,
        allocation_year=allocation_year,
        world_scenario_emissions_ts=world,
        verbose=False,
    )
    cumulative = pd.DataFrame({str(allocation_year): [total]}, index=world.index)
    absolute = shares.get_absolute_budgets(cumulative)

    return prepare_dataframe(
        data=absolute,
        result=shares,
        climate_assessment=budget.climate_target.limit,
        quantile=budget.climate_target.quantile,
        data_context={
            **context,
            "emission-category": category,
            "source": budget.climate_target.reference,
            "missing-net-negative-mtco2e": None,
        },
        is_budget=True,
        **{"total-budget": total},
    )


def allocate(
    data: ProcessedData,
    requests: list[AllocationRequest],
    *,
    sources: SourceSelection,
    target: str = "rcbs",
    harmonisation_year: int | None = None,
    source_id: str = "in-memory",
    share_cache: dict[tuple[str, str], Any] | None = None,
) -> list[AllocationResult]:
    """Run several allocations against one set of preprocessed data.

    Shares are computed once per distinct method and reused across every
    climate target that method is asked for, because they do not depend on the
    budget. Preprocessing is the expensive half overall, so a caller running a
    grid does that once and calls this many times.

    Parameters
    ----------
    data
        What `preprocess` returned.
    requests
        The allocations to run, each describing exactly one.
    sources
        Recorded on every result, so a result read later says what data it
        came from.
    target
        The target type.
    harmonisation_year
        Passed through to the allocators.
    source_id
        Provenance stamped into the allocator's metadata columns.
    share_cache
        A mapping the caller owns, so shares survive *between* calls as well
        as within one. A caller that runs each allocation separately -- to keep
        failures attributable to a single request, say -- would otherwise
        recompute the same shares for every climate target. Pass the same dict
        for as long as `data` is the same object; it is keyed on the method,
        not on the data.

    Returns
    -------
    list of AllocationResult
        One per request, in the order given.

    Raises
    ------
    ConfigurationError
        If a request names no climate target, or one the data does not carry.
    """
    if data.budget_category is None or data.rcbs.empty:
        raise ConfigurationError(
            f"this run carries no remaining carbon budget for {data.category!r}, "
            "so there is nothing to allocate. Preprocess with an "
            "`rcb_definition` to compute one."
        )

    category = data.budget_category
    context = _data_context(data, sources, source_id, target)
    cache = {} if share_cache is None else share_cache
    results: list[AllocationResult] = []

    for request in requests:
        if request.climate_target is None:
            raise ConfigurationError(
                f"{request.approach!r} allocates a remaining carbon budget, so "
                "it needs a climate_target saying which one."
            )
        budget = select_budget(data.rcbs, request.climate_target)

        # Keyed on the method alone: the shares do not depend on the budget,
        # which is the whole reason the two are separate steps.
        key = (request.approach, repr(sorted(request.parameters.items())))
        if key not in cache:
            logger.info("computing shares for %s", request.approach)
            cache[key] = compute_shares(
                data,
                request.approach,
                request.parameters,
                category=category,
                target=target,
                harmonisation_year=harmonisation_year,
            )

        logger.info("allocating %s at %s", request.approach, request.climate_target)
        allocations = apply_budget(
            cache[key], budget, data, category=category, context=context
        )
        results.append(
            AllocationResult(
                request=request,
                category=data.category,
                sources=sources,
                allocations=allocations,
            )
        )

    return results


def _data_context(
    data: ProcessedData, sources: SourceSelection, source_id: str, target: str
) -> dict[str, str]:
    """Build the provenance the allocator stamps onto its output."""
    return {
        "source-id": source_id,
        "allocation-folder": source_id,
        "emission-category": data.category,
        "target-source": target,
        "emissions-source": sources.emissions,
        "gdp-source": sources.gdp,
        "population-source": sources.population,
        "gini-source": sources.gini,
        "lulucf-source": sources.lulucf or "none",
    }
