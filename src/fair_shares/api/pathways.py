"""Allocating a scenario pathway, as a call rather than a grid.

The pathway counterpart of :mod:`fair_shares.api.allocation`, and it exists for
the same reason that module does: the only way to run an allocation was to hand
a *table* to something that iterated whatever it was given.

For budgets that was ``select_budget`` -- a caller wanting one number passed a
frame of budgets and filtered afterwards. For pathways it is
:func:`~fair_shares.library.allocations.manager.run_parameter_grid`, which is
worse in two specific ways:

**It fans out.** Every value may be a list, and the function crosses them. A
caller that wants exactly one allocation has to wrap each of its parameters in a
single-element list and then assert that the answer came back with length one --
which is what every caller of it currently does. The expansion is a feature for
exploring a grid interactively and a hazard for anything that needs one run to
mean one thing: a parameter that is a list *by accident* silently becomes two
allocations, and the second one quietly wins or is discarded depending on how
the caller indexes the result.

**The validation lives in the fan-out, not in the allocation.** The four checks
that decide whether a configuration is scientifically admissible --
``validate_target_source_compatibility``,
``validate_allocation_years_against_harmonisation``,
``validate_allocation_year_for_co2`` and ``validate_allocation_parameters`` --
are all called by ``run_parameter_grid``. ``run_allocation``, the function its
own documentation recommends "for single allocations", calls none of them. So
the obvious way to avoid the fan-out is also a way to silently switch off the
NGHGI year guard.

This module resolves both: one request in, one result out, with every check
``run_parameter_grid`` would have applied applied explicitly and in the same
order. Nothing here writes to disk -- serialisation is the caller's business,
as everywhere else in this package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fair_shares.api.allocation import AllocationResult
from fair_shares.api.pipeline import ProcessedData
from fair_shares.api.sources import SourceSelection
from fair_shares.library.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# What the library calls a scenario-pathway target internally. It decides how a
# composite category is decomposed -- ``all-ghg`` stays whole for a pathway and
# splits into ``("co2", "non-co2")`` for a budget -- so it is not interchangeable
# with the name of the ensemble a caller's config happens to use.
PATHWAY_TARGET = "pathway"

# The index the world scenario ensemble is expected to carry.
SCENARIO_INDEX = (
    "climate-assessment",
    "quantile",
    "source",
    "iso3c",
    "unit",
    "emission-category",
)


@dataclass(frozen=True)
class ScenarioGroup:
    """Which single group of the world scenario ensemble to allocate.

    The pathway counterpart of :class:`~fair_shares.api.allocation.ClimateTarget`,
    and the same idea: it names *one* row of the target data, so the allocator
    is handed a world pathway rather than a table it would iterate.

    Attributes
    ----------
    limit
        The ``climate-assessment`` label, e.g. ``"C1"`` for an AR6 category or
        ``"1.6"`` for a peak-temperature group.
    quantile
        The ensemble quantile, e.g. ``0.5``.
    """

    limit: str
    quantile: float

    def __str__(self) -> str:
        """Render the group the way the ensemble labels it."""
        return f"{self.limit} q{self.quantile:g}"


@dataclass(frozen=True)
class PathwayRequest:
    """One pathway allocation to run: a method, its inputs, and a scenario group.

    Attributes
    ----------
    approach
        An allocation registry name, e.g. ``"equal-per-capita"``. Must be a
        pathway approach; the budget ones cannot allocate a scenario.
    parameters
        The approach's own keyword inputs. Plain values, **not** lists: one
        request describes exactly one allocation.
    scenario_group
        Which world pathway to share out.
    label
        Free-form, for the caller's own bookkeeping. Never affects the result.
    """

    approach: str
    parameters: dict[str, Any] = field(default_factory=dict)
    scenario_group: ScenarioGroup | None = None
    label: str | None = None


@dataclass(frozen=True)
class WorldPathway:
    """One scenario group's world total, and the ensemble's own name for it.

    Attributes
    ----------
    emissions
        The world pathway, indexed as the ensemble was.
    label
        The ensemble's own ``climate-assessment`` value. Not necessarily the
        string the caller asked with: a config names a group in text, and a
        peak-temperature ensemble read from CSV carries ``1.6`` as a float.
        Results are stamped with this rather than with the request, so metadata
        written by two callers who spelled the same group differently still
        matches.
    group
        The group that was asked for.
    """

    emissions: pd.DataFrame
    label: Any
    group: ScenarioGroup


def select_world_pathway(
    scenarios: pd.DataFrame, group: ScenarioGroup, world_key: str = "World"
) -> WorldPathway:
    """Look up one scenario group's world pathway.

    A lookup, not a filter -- the same contract as
    :func:`~fair_shares.api.allocation.select_budget`. It returns one pathway,
    and says so if there is no such group rather than quietly returning several
    for something downstream to iterate.

    Parameters
    ----------
    scenarios
        The harmonised world scenario ensemble, indexed by `SCENARIO_INDEX`
        with one column per year.
    group
        Which group to take.
    world_key
        The ``iso3c`` value carrying the world total.

    Returns
    -------
    WorldPathway
        The pathway, tagged with the group it answers and the ensemble's own
        label for it.

    Raises
    ------
    ConfigurationError
        If the ensemble has no such group, listing what it does have.
    """
    from fair_shares.library.utils.data.completeness import get_world_totals_timeseries

    missing = [name for name in SCENARIO_INDEX if name not in scenarios.index.names]
    if missing:
        raise ConfigurationError(
            f"the scenario ensemble is missing index level(s) {missing}. "
            f"Expected {list(SCENARIO_INDEX)}."
        )

    labels = scenarios.index.get_level_values("climate-assessment")
    quantiles = scenarios.index.get_level_values("quantile").astype(float)
    matched = (labels.astype(str) == str(group.limit)) & (
        quantiles == float(group.quantile)
    )
    selected = scenarios[matched]
    if selected.empty:
        available = sorted(
            {f"{a} q{q:g}" for a, q in zip(labels.astype(str), quantiles, strict=True)}
        )
        raise ConfigurationError(
            f"no scenario group for {group}. Available: {available}"
        )

    world = get_world_totals_timeseries(
        selected, world_key, expected_index_names=list(SCENARIO_INDEX)
    )
    return WorldPathway(emissions=world, label=labels[matched][0], group=group)


def _validate(
    request: PathwayRequest,
    category: str,
    harmonisation_year: int | None,
    nghgi_years: tuple[int, int] | None,
) -> None:
    """Apply every check ``run_parameter_grid`` would have applied.

    Called explicitly because `run_allocation` applies none of them, so a
    caller that avoids the fan-out by going straight to it also loses the
    validation. The checks are the library's own -- restating any of them here
    would be a second opinion that drifts from the first.

    Parameters
    ----------
    request
        The allocation to check.
    category
        The emission category being allocated.
    harmonisation_year
        The year scenarios are anchored to observed emissions.
    nghgi_years
        The run's NGHGI-consistent LULUCF record, or None when it has none.

    Raises
    ------
    AllocationError
        If the configuration is not admissible, with the library's own message.
    """
    from fair_shares.library.allocations import is_budget_approach
    from fair_shares.library.validation.config import (
        validate_allocation_parameters,
        validate_allocation_year_for_co2,
        validate_allocation_years_against_harmonisation,
        validate_target_source_compatibility,
    )

    # The validators take the grid's own shape, so the request is presented in
    # it. This is the *only* place a single request is widened into that form,
    # and nothing is run from it.
    as_grid = {request.approach: [dict(request.parameters)]}

    validate_target_source_compatibility(as_grid, PATHWAY_TARGET)
    validate_allocation_years_against_harmonisation(
        as_grid, harmonisation_year, PATHWAY_TARGET
    )
    validate_allocation_year_for_co2(as_grid, category, nghgi_years)
    validate_allocation_parameters(
        request.approach,
        {key.replace("-", "_"): value for key, value in request.parameters.items()},
        is_budget_approach(request.approach),
    )


def allocate_pathway(
    data: ProcessedData,
    request: PathwayRequest,
    world: WorldPathway,
    *,
    category: str,
    harmonisation_year: int | None = None,
):
    """Run exactly one pathway allocation.

    Parameters
    ----------
    data
        What `preprocess` returned.
    request
        The allocation to run. Its parameters are plain values.
    world
        The world pathway to share out, from `select_world_pathway`.
    category
        The emission-category part to allocate. For a pathway this is the
        category itself -- a scenario pathway is not decomposed the way a
        budget is.
    harmonisation_year
        Passed through to the allocators.

    Returns
    -------
    :
        The library's `PathwayAllocationResult`.

    Raises
    ------
    ConfigurationError
        If the approach is not a pathway approach, or the request carries a
        list where a single value belongs.
    """
    from fair_shares.library.allocations import is_pathway_approach, run_allocation

    if not is_pathway_approach(request.approach):
        raise ConfigurationError(
            f"{request.approach!r} allocates a cumulative budget, so it cannot "
            "be run against a scenario pathway. Pathway approaches share out "
            "the scenario's annual emissions."
        )

    listed = sorted(
        key
        for key, value in request.parameters.items()
        if isinstance(value, list | tuple)
    )
    if listed:
        # The fan-out this module exists to avoid. Refused rather than
        # expanded: one request has to mean one allocation, or a caller keying
        # results by request has no way to tell which of them it got.
        raise ConfigurationError(
            f"{request.approach!r}: parameters {listed} are lists. One request "
            "describes exactly one allocation; put each combination in its own "
            "request."
        )

    _validate(request, category, harmonisation_year, data.nghgi_years)

    emissions = data.parts[category]
    # `harmonisation_year` is deliberately **not** forwarded. It constrains
    # which allocation years are admissible against a scenario that was
    # harmonised in a given year, and `_validate` above is the whole of its
    # effect; no allocator takes it. `run_allocation` passes anything it does
    # not recognise straight through to the allocator, which then rejects it,
    # so forwarding it would fail every pathway allocation in the library.
    return run_allocation(
        approach=request.approach,
        population_ts=data.population,
        gdp_ts=data.gdp,
        gini_s=data.gini,
        country_actual_emissions_ts=emissions,
        # A pathway approach measures responsibility on the category it is
        # allocating. The budget path substitutes fossil CO2 for the all-GHG
        # categories, which is a different question and belongs to that path.
        responsibility_emissions_ts=emissions,
        world_scenario_emissions_ts=world.emissions,
        emission_category=category,
        **request.parameters,
    )


def allocate_pathways(
    data: ProcessedData,
    requests: list[PathwayRequest],
    *,
    scenarios: pd.DataFrame,
    sources: SourceSelection,
    category: str,
    harmonisation_year: int | None = None,
    source_id: str = "in-memory",
    world_key: str = "World",
) -> list[AllocationResult]:
    """Run several pathway allocations against one set of preprocessed data.

    The pathway counterpart of :func:`~fair_shares.api.allocation.allocate`,
    and the same contract: requests in, results out, one for one, in order,
    each tagged with what produced it. Nothing is written.

    Unlike the budget path there is no share cache, and that is not an
    oversight: a pathway allocation's shares depend on the world pathway it is
    being run against -- the convergence approaches solve against it -- so two
    scenario groups do not share a share computation the way two budgets do.

    Parameters
    ----------
    data
        What `preprocess` returned.
    requests
        The allocations to run, each describing exactly one.
    scenarios
        The harmonised world scenario ensemble.
    sources
        Recorded on every result, so a result read later says what data it came
        from.
    category
        The emission category to allocate.
    harmonisation_year
        Passed through to the allocators.
    source_id
        Provenance stamped into the allocator's metadata columns.
    world_key
        The ``iso3c`` value carrying the world total in `scenarios`.

    Returns
    -------
    list of AllocationResult
        One per request, in the order given. `allocations` carries the absolute
        emissions with the library's own metadata columns.

    Raises
    ------
    ConfigurationError
        If a request names no scenario group, or one the ensemble does not
        carry, or the category is not one this run prepared.
    """
    from fair_shares.library.allocations.results.serializers import prepare_dataframe

    if category not in data.parts:
        raise ConfigurationError(
            f"this run carries no emissions for {category!r}. It has: "
            f"{sorted(data.parts)}."
        )

    context = {
        "source-id": source_id,
        "allocation-folder": source_id,
        "emission-category": category,
        "target-source": PATHWAY_TARGET,
        "emissions-source": sources.emissions,
        "gdp-source": sources.gdp,
        "population-source": sources.population,
        "gini-source": sources.gini,
    }

    results: list[AllocationResult] = []
    for request in requests:
        if request.scenario_group is None:
            raise ConfigurationError(
                f"{request.approach!r} allocates a scenario pathway, so it "
                "needs a scenario_group saying which one."
            )
        world = select_world_pathway(scenarios, request.scenario_group, world_key)

        logger.info("allocating %s at %s", request.approach, request.scenario_group)
        allocation = allocate_pathway(
            data,
            request,
            world,
            category=category,
            harmonisation_year=harmonisation_year,
        )
        absolute = allocation.get_absolute_emissions(world.emissions)

        frame = prepare_dataframe(
            data=absolute,
            result=allocation,
            climate_assessment=world.label,
            quantile=request.scenario_group.quantile,
            data_context={**context, "missing-net-negative-mtco2e": None},
            is_budget=False,
        )
        results.append(
            AllocationResult(
                request=request,  # type: ignore[arg-type]
                category=category,
                sources=sources,
                allocations=frame,
            )
        )
    return results
