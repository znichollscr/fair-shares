"""Running allocations against preprocessed data.

The half of the API a caller running a grid actually spends its time in.
Three things are deliberate here:

**Results know what produced them.** Every result carries the request that
made it. The notebook pipeline wrote allocations into a directory named after
a hash of the run's parameters and left the caller to reconstruct the mapping
from the path, which is fine until two runs differ in something the path does
not record.

**One allocation at a time, by construction.** ``run_all_allocations`` iterates
every remaining-budget row in the data it is handed, so handing it an unfiltered
table quietly produces eight climate targets' worth of rows per country under a
single label. :func:`allocate` narrows the data to the requested anchor first,
so "one request, one result" is true rather than merely intended.

**Nothing is written.** Results come back as frames for the caller to serialise
however it likes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fair_shares.api.preprocess import ProcessedData
from fair_shares.api.sources import SourceSelection
from fair_shares.library.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Categories whose responsibility is measured on fossil CO2 rather than on the
# category itself, because the land flux is not something a country is held
# responsible for in the same way.
_LULUCF_INCLUSIVE = frozenset({"co2", "all-ghg"})


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
        The request this answers.
    category
        The emission category allocated.
    sources
        Which input vintages it was computed from.
    allocations
        Absolute allocations, one row per (country, emission-category part).
    """

    request: AllocationRequest
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


def build_loaded_data(data: ProcessedData, *, target: str = "rcbs") -> dict[str, Any]:
    """Assemble what the allocators expect, from preprocessed frames.

    The in-memory equivalent of ``load_allocation_data``, which reads the same
    structure back off disk.

    Parameters
    ----------
    data
        What `preprocess` returned.
    target
        The target type, e.g. ``"rcbs"``.

    Returns
    -------
    dict
        Keyed as the allocators expect.
    """
    emissions = dict(data.parts)
    rcbs = {}
    if not data.rcbs.empty and data.budget_category is not None:
        rcbs[data.budget_category] = data.rcbs

    # Responsibility is measured on fossil CO2 for the LULUCF-inclusive
    # categories: a country is not held responsible for its land flux the way
    # it is for its fossil emissions.
    responsibility = {}
    for category, frame in emissions.items():
        if category in _LULUCF_INCLUSIVE and "co2-ffi" in emissions:
            responsibility[category] = emissions["co2-ffi"]
        else:
            responsibility[category] = frame

    return {
        "emissions_data": emissions,
        "responsibility_emissions_data": responsibility,
        "scenarios_data": {},
        "rcbs_data": rcbs,
        "world_emissions_data": {
            name: _label_world(frame) for name, frame in data.world_emissions.items()
        },
        "country_gdp_df": data.gdp,
        "country_population_df": data.population,
        "country_gini_df": data.gini,
        "net_negative_metadata": {},
        "target": target,
    }


def _label_world(frame: pd.DataFrame) -> pd.DataFrame:
    """Relabel a world total's country code to the literal ``"World"``.

    The allocators look for that exact string, while the emissions source uses
    its own world key (PRIMAP says ``EARTH``). The notebook pipeline did the
    same rename on its way to a CSV, so this is where it has to happen now.

    Parameters
    ----------
    frame
        A world total indexed by ``(iso3c, unit, emission-category)``.

    Returns
    -------
    pandas.DataFrame
        The same frame with its country level set to ``"World"``.
    """
    relabelled = frame.copy()
    relabelled.index = pd.MultiIndex.from_tuples(
        [("World", *rest) for _, *rest in frame.index],
        names=frame.index.names,
    )
    return relabelled


def filter_to_anchor(
    loaded_data: dict[str, Any], climate_target: ClimateTarget
) -> dict[str, Any]:
    """Narrow loaded data to a single climate target, without mutating it.

    This is what makes "one request, one result" true. The allocator loops over
    every budget row it can see, so an unfiltered table yields every climate
    target's rows under whatever label the caller gave the request.

    Parameters
    ----------
    loaded_data
        What `build_loaded_data` returned.
    climate_target
        The anchor to keep.

    Returns
    -------
    dict
        A new mapping whose budget and scenario tables carry only the anchor's
        rows. Every other entry is shared with the input, which is safe
        because they are read and never written.

    Raises
    ------
    ConfigurationError
        If the anchor matches no rows, naming what was on offer instead.
    """
    filtered = dict(loaded_data)

    budgets = {}
    for category, frame in loaded_data["rcbs_data"].items():
        keep = frame[
            (frame["source"] == climate_target.reference)
            & (frame["climate-assessment"] == climate_target.limit)
            & (frame["quantile"] == climate_target.quantile)
        ].reset_index(drop=True)
        if keep.empty:
            available = sorted(
                {
                    f"{row['climate-assessment']} p{row['quantile'] * 100:g} "
                    f"{row['source']}"
                    for _, row in frame.iterrows()
                }
            )
            raise ConfigurationError(
                f"no remaining-budget row for {climate_target} in category "
                f"{category!r}. Available: {available}"
            )
        budgets[category] = keep
    filtered["rcbs_data"] = budgets

    scenarios = {}
    for category, frame in loaded_data["scenarios_data"].items():
        levels = frame.index
        mask = (
            levels.get_level_values("climate-assessment") == climate_target.limit
        ) & (levels.get_level_values("quantile") == climate_target.quantile)
        keep = frame[mask]
        if keep.empty:
            raise ConfigurationError(
                f"no scenario rows for {climate_target} in category {category!r}"
            )
        scenarios[category] = keep
    filtered["scenarios_data"] = scenarios

    return filtered


def allocate(
    data: ProcessedData,
    requests: list[AllocationRequest],
    *,
    sources: SourceSelection,
    target: str = "rcbs",
    harmonisation_year: int | None = None,
    source_id: str = "in-memory",
) -> list[AllocationResult]:
    """Run several allocations against one set of preprocessed data.

    Preprocessing is the expensive half, so a caller running a grid does it
    once and calls this many times.

    Parameters
    ----------
    data
        What `preprocess` returned.
    requests
        The allocations to run. Each is run in isolation.
    sources
        Recorded on every result, so a result read later says what data it
        came from.
    target
        The target type, e.g. ``"rcbs"``.
    harmonisation_year
        Passed through to the allocators.
    source_id
        Provenance stamped into the allocator's own metadata columns. Defaults
        to something honest rather than a path that does not exist.

    Returns
    -------
    list of AllocationResult
        One per request, in the order given.

    Raises
    ------
    ConfigurationError
        If a budget target request carries no climate target, or names one the
        data does not hold.
    """
    from fair_shares.library.notebook_helpers import run_all_allocations

    loaded = build_loaded_data(data, target=target)
    final_categories = sorted(data.parts)
    results: list[AllocationResult] = []

    for request in requests:
        if loaded["rcbs_data"] and request.climate_target is None:
            raise ConfigurationError(
                f"{request.approach!r} allocates a remaining carbon budget, so "
                "it needs a climate_target saying which one."
            )

        anchored = (
            filter_to_anchor(loaded, request.climate_target)
            if request.climate_target is not None
            else loaded
        )

        logger.info(
            "allocating %s%s",
            request.approach,
            f" at {request.climate_target}" if request.climate_target else "",
        )

        _rows, absolute = run_all_allocations(
            allocations={request.approach: [dict(request.parameters)]},
            loaded_data=anchored,
            output_dir=None,
            data_context=_data_context(data, sources, source_id, target),
            target=target,
            final_categories=final_categories,
            harmonisation_year=harmonisation_year,
            write=False,
            return_allocations=True,
        )

        results.append(
            AllocationResult(
                request=request,
                category=data.category,
                sources=sources,
                allocations=absolute,
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
