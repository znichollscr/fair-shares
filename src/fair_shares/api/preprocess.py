"""Raw inputs to allocation-ready data, entirely in memory.

Ported from notebook 100. This is the step that decides *which countries are
in the analysis at all* and completes the rest of the world, so it is where the
pipeline's coverage rules live.

The whole of it is one function call returning one object. The notebook version
wrote nine CSVs into a directory named after a hash of its parameters and left
the next stage to find them again; the CSVs were never a product, only a way
for one notebook to talk to the next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fair_shares.api import emissions as emissions_module
from fair_shares.api import inputs as inputs_module
from fair_shares.api import lulucf as lulucf_module
from fair_shares.api import scenarios as scenarios_module
from fair_shares.api.sources import Inputs
from fair_shares.library.exceptions import DataProcessingError
from fair_shares.library.preprocessing.coverage import (
    compute_analysis_countries,
    create_coverage_summary,
)
from fair_shares.library.preprocessing.gini import complete_gini
from fair_shares.library.utils import ensure_string_year_columns
from fair_shares.library.utils.data.completeness import (
    add_row_timeseries,
    get_world_totals_timeseries,
)

logger = logging.getLogger(__name__)

# Coverage is judged from this year forward: a country with no 1990s record is
# not one the allocation approaches can use, whatever it reports later.
COVERAGE_START_YEAR = 1990

_EMISSIONS_INDEX = ["iso3c", "unit", "emission-category"]
_COUNTRY_INDEX = ["iso3c", "unit"]


@dataclass(frozen=True)
class ProcessedData:
    """Everything an allocation needs, for one emission category.

    Attributes
    ----------
    category
        The emission category these frames describe.
    gdp, population
        Country timeseries indexed by ``(iso3c, unit)``, rest-of-world
        included.
    gini
        One value per country, imputed where a country has none.
    world_emissions
        World totals per category, used to derive the rest of the world.
    rcbs
        Rebased remaining carbon budgets, one row per (source, scenario).
        Empty for a run that computes no budgets.
    coverage
        Per-country flags: which datasets carried it, whether it made the
        analysis, whether its Gini was imputed.
    analysis_countries
        The countries carried individually. Everything else is in the
        rest-of-world aggregate.
    """

    category: str
    parts: dict[str, pd.DataFrame]
    """The categories this one is actually allocated as, rest-of-world
    included. A composite is allocated as its parts and summed afterwards --
    ``all-ghg-ex-co2-lulucf`` is ``co2-ffi`` plus ``non-co2`` -- so the parts,
    not the whole, are what an allocation consumes."""
    gdp: pd.DataFrame
    population: pd.DataFrame
    gini: pd.DataFrame
    world_emissions: dict[str, pd.DataFrame]
    rcbs: pd.DataFrame
    coverage: pd.DataFrame
    analysis_countries: frozenset[str]
    budget_category: str | None = None
    """Which part the remaining carbon budget was rebased for. A budget is
    defined for CO2, so a composite category's budget belongs to its CO2 part
    rather than to the composite."""

    @property
    def emissions(self) -> pd.DataFrame:
        """Return this category's own emissions frame.

        Returns
        -------
        pandas.DataFrame
            The frame for `category`.

        Raises
        ------
        KeyError
            If the category is a composite, which has no single frame -- ask
            for `parts` instead.
        """
        if self.category not in self.parts:
            raise KeyError(
                f"{self.category!r} is allocated as its parts "
                f"({sorted(self.parts)}), so it has no single emissions frame. "
                "Use `parts`."
            )
        return self.parts[self.category]


def _region_countries(inputs: Inputs) -> set[str]:
    """Return every country the region mapping knows about."""
    mapping = inputs.catalogue["general"]["region_mapping"]["path"]
    path = inputs._resolve(str(mapping))
    return set(pd.read_csv(path)["iso3c"].unique())


def preprocess(
    inputs: Inputs,
    category: str,
    *,
    rcb_definition: dict | Path | None = None,
    gini_missing_policy: str = "fallback-mean",
    precautionary_lulucf: bool = True,
) -> ProcessedData:
    """Turn raw inputs into allocation-ready data for one category.

    Parameters
    ----------
    inputs
        Which vintages to read and where they live.
    category
        The emission category to prepare.
    rcb_definition
        The remaining-carbon-budget definition, as a parsed mapping or a path
        to its YAML. Omit for a run that allocates no budget.
    gini_missing_policy
        What to do with a country that has no Gini value: ``"fallback-mean"``
        gives it the analysis-country mean and records the imputation;
        ``"strict"`` stops the run.
    precautionary_lulucf
        When true, a projected land sink cannot enlarge a fossil budget.

    Returns
    -------
    ProcessedData
        Frames, not files.

    Raises
    ------
    DataProcessingError
        If a rest-of-world total cannot be formed because a dataset carries no
        world row.
    """
    logger.info("preprocessing %s", category)

    world_key = inputs.parameters("emissions").get("world_key")
    gdp_world_key = inputs.parameters("gdp").get("world_key")
    population_world_key = inputs.parameters("population").get("historical_world_key")

    # A composite is allocated as its parts, so those are what the run needs.
    final_categories = _final_categories(category)
    wanted = _required_parts(category)
    parts = emissions_module.load_emissions(inputs, wanted)

    if inputs.sources.lulucf is not None:
        land = lulucf_module.load_lulucf(inputs, world_key)
        parts = lulucf_module.apply_nghgi_corrections(parts, land)
    else:
        land = None

    gdp = inputs_module.load_gdp(inputs)
    population = inputs_module.load_population(inputs)
    region_countries = _region_countries(inputs)
    gini = inputs_module.load_gini(inputs, region_countries)

    world_emissions = {
        name: get_world_totals_timeseries(
            frame, world_key, expected_index_names=_EMISSIONS_INDEX
        )
        for name, frame in parts.items()
    }
    world_gdp = get_world_totals_timeseries(
        gdp, gdp_world_key, expected_index_names=_COUNTRY_INDEX
    )
    world_population = get_world_totals_timeseries(
        population, population_world_key, expected_index_names=_COUNTRY_INDEX
    )

    # A country is carried individually only if *every* dataset covers it for
    # the whole window. Gini is deliberately not part of that test: a missing
    # Gini is imputed rather than being grounds for exclusion. Both rules live
    # in the library already, so this calls them rather than restating them.
    # Only the categories actually allocated decide coverage. `parts` also
    # holds the ingredients they were built from (`co2-ffi` and the all-GHG
    # total behind `non-co2`), and intersecting over those too would exclude a
    # country for missing a series the run never allocates.
    #
    # This differs from the notebook pipeline for *composite* categories, and
    # deliberately. The master notebook ran once per decomposition pass and
    # each pass overwrote one `country_data_coverage_summary.csv`, so a
    # composite's `has_emissions` column records whichever pass happened to run
    # last -- which is why the all-ghg tree reports the same 205 countries as
    # the co2-ffi tree despite allocating a category only 166 of them have.
    # `in_analysis`, which is what everything downstream reads, is unaffected
    # and does still match.
    allocated = {name: parts[name] for name in final_categories if name in parts}
    analysis = compute_analysis_countries(allocated, gdp, population)
    coverage = create_coverage_summary(
        analysis_countries=analysis,
        emissions_data=allocated,
        gdp=gdp,
        population=population,
        gini=gini,
        region_mapping=pd.DataFrame({"iso3c": sorted(region_countries)}),
        output_dir=None,
        gdp_variant=inputs.parameters("gdp").get("gdp_variant"),
    )

    completed = {
        name: add_row_timeseries(
            frame,
            analysis,
            world_emissions[name],
            expected_index_names=_EMISSIONS_INDEX,
        )
        for name, frame in parts.items()
    }
    gdp_complete = add_row_timeseries(
        gdp, analysis, world_gdp, expected_index_names=_COUNTRY_INDEX
    )
    population_complete = add_row_timeseries(
        population, analysis, world_population, expected_index_names=_COUNTRY_INDEX
    )
    gini_complete, imputed = complete_gini(gini, analysis, policy=gini_missing_policy)
    if imputed:
        logger.info(
            "Gini imputed (analysis-country mean) for %d countries", len(imputed)
        )

    budget_category = _budget_category(category)
    rcbs = pd.DataFrame()
    if rcb_definition is not None and budget_category is not None:
        rcbs = _build_rcbs(
            inputs=inputs,
            category=budget_category,
            rcb_definition=rcb_definition,
            world_emissions=world_emissions,
            land=land,
            world_key=world_key,
            precautionary_lulucf=precautionary_lulucf,
        )

    logger.info(
        "%s: %d analysis countries, %d in rest of world",
        category,
        len(analysis),
        len(region_countries - analysis),
    )

    return ProcessedData(
        category=category,
        parts={
            name: ensure_string_year_columns(frame)
            for name, frame in completed.items()
            if name in final_categories or name == category
        },
        budget_category=budget_category,
        gdp=ensure_string_year_columns(gdp_complete),
        population=ensure_string_year_columns(population_complete),
        gini=gini_complete,
        world_emissions=world_emissions,
        rcbs=rcbs,
        coverage=coverage,
        analysis_countries=frozenset(analysis),
    )


def _final_categories(category: str) -> tuple[str, ...]:
    """Return the categories `category` is actually allocated as.

    A composite is decomposed: ``all-ghg`` is allocated as ``co2`` plus
    ``non-co2``, because a remaining carbon budget is defined for CO2 and the
    non-CO2 part is handled separately. Read from upstream's own rule rather
    than restated.
    """
    from fair_shares.library.utils.data.config import get_final_categories

    return tuple(get_final_categories("rcbs", category))


def _budget_category(category: str) -> str | None:
    """Return the part a remaining carbon budget is rebased for.

    Budgets are published for CO2, so a composite's budget belongs to its CO2
    part. Returns None for a category with no CO2 part at all.
    """
    for part in _final_categories(category):
        if part in ("co2-ffi", "co2"):
            return part
    return None


def _required_parts(category: str) -> list[str]:
    """Return every emission category that has to be read or derived.

    The parts the category is allocated as, plus the ingredients those parts
    are built from -- `co2` needs fossil CO2 and land, `non-co2` needs the
    all-GHG total. Reading them costs nothing extra in a single pass.
    """
    parts = list(_final_categories(category))
    if category not in parts:
        parts.append(category)
    if any(part in ("co2", "all-ghg", "non-co2") for part in parts):
        for ingredient in ("co2-ffi", "all-ghg-ex-co2-lulucf"):
            if ingredient not in parts:
                parts.append(ingredient)
    return parts


def _coverage_summary(
    region_countries: set[str],
    emissions_complete: dict[str, set[str]],
    gdp_countries: set[str],
    population_countries: set[str],
    gini_countries: set[str],
    analysis: set[str],
) -> pd.DataFrame:
    """Build the per-country coverage table.

    Recorded rather than merely used, because "this country is missing" and
    "this country was aggregated into the rest of the world" look identical in
    the output otherwise.
    """
    summary = pd.DataFrame({"iso3c": sorted(region_countries)})
    has_emissions = pd.Series(True, index=summary.index)
    for covered in emissions_complete.values():
        has_emissions &= summary["iso3c"].isin(covered)
    summary["has_emissions"] = has_emissions
    summary["has_gdp"] = summary["iso3c"].isin(gdp_countries)
    summary["has_population"] = summary["iso3c"].isin(population_countries)
    summary["has_gini"] = summary["iso3c"].isin(gini_countries)
    summary["in_analysis"] = summary["iso3c"].isin(analysis)
    summary["gini_imputed"] = summary["in_analysis"] & ~summary["has_gini"]
    summary["in_row"] = ~summary["in_analysis"]
    return summary


def _build_rcbs(
    *,
    inputs: Inputs,
    category: str,
    rcb_definition: dict | Path,
    world_emissions: dict[str, pd.DataFrame],
    land: pd.DataFrame | None,
    world_key: str,
    precautionary_lulucf: bool,
) -> pd.DataFrame:
    """Rebase the published budgets, using in-memory adjustments."""
    import yaml

    from fair_shares.library.preprocessing.rcbs import process_rcbs

    if not isinstance(rcb_definition, dict):
        rcb_definition = yaml.safe_load(Path(rcb_definition).read_text())

    if "co2-ffi" not in world_emissions:
        raise DataProcessingError(
            "rebasing a remaining carbon budget needs world fossil CO2, which "
            f"this run did not read (it has {sorted(world_emissions)})."
        )

    inventory_land = None
    if land is not None and category == "co2":
        inventory_land = lulucf_module.world_lulucf(land, world_key)

    frames = scenarios_module.load_scenario_frames(inputs)
    adjustments = scenarios_module.compute_adjustments(frames, inventory_land)

    bunkers = inputs_module.load_bunkers(inputs)
    actual_land = (
        ensure_string_year_columns(world_emissions["co2-lulucf"])
        if "co2-lulucf" in world_emissions
        else None
    )

    return process_rcbs(
        rcb_data=rcb_definition,
        world_fossil_emissions=ensure_string_year_columns(world_emissions["co2-ffi"]),
        emission_category=category,
        bunker_timeseries=bunkers,
        rcb_adjustments=adjustments.scalars(),
        lulucf_shift=adjustments.lulucf_shift,
        precautionary_lulucf=precautionary_lulucf,
        actual_bm_lulucf_emissions=actual_land,
        verbose=False,
    )
