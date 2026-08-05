"""Observed emissions by category, from the PRIMAP-hist record.

Ported from notebook 101. The notebook ran once per category, driven by a
Snakemake loop, and wrote one CSV per pass; this reads the netCDF **once** and
returns every category the caller asked for, which is both faster and the
reason the "which categories does this tree contain?" question stops depending
on which target built it.

The category definitions are the scientific content of this module: which
PRIMAP variable and which IPCC 2006 sectors add up to each of our categories.
They are stated once, here, rather than in a notebook that gets copied.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from fair_shares.api.sources import Inputs
from fair_shares.library.exceptions import (
    ConfigurationError,
    DataLoadingError,
    DataProcessingError,
)
from fair_shares.library.utils import ensure_string_year_columns
from fair_shares.library.utils.units import (
    convert_unit_robust,
    get_default_unit_registry,
    set_single_unit,
)

logger = logging.getLogger(__name__)

# How each of our emission categories is assembled from the PRIMAP record: a
# gas variable plus the IPCC 2006 sectors that make it up. Categories with
# several sectors are summed.
#
# `non-co2` is deliberately absent. It is never read directly from PRIMAP --
# it is derived by subtraction (`all-ghg-ex-co2-lulucf` minus `co2-ffi`), so
# that the parts of a decomposition are guaranteed to add back up to the whole.
CATEGORY_DEFINITIONS: dict[str, dict[str, object]] = {
    "co2-ffi": {"variable": "CO2", "sectors": ["1", "2"]},
    "co2": {"variable": "CO2", "sectors": ["1", "2", "M.LULUCF"]},
    "co2-lulucf": {"variable": "CO2", "sectors": ["M.LULUCF"]},
    "all-ghg": {"variable": "KYOTOGHG (AR6GWP100)", "sectors": ["0"]},
    "all-ghg-ex-co2-lulucf": {
        "variable": "KYOTOGHG (AR6GWP100)",
        "sectors": ["M.0.EL"],
    },
}

# The category derived by subtraction rather than read, and what it is made of.
DERIVED_CATEGORY = "non-co2"
DERIVED_FROM = ("all-ghg-ex-co2-lulucf", "co2-ffi")

# Everything the pipeline works in.
WORKING_UNIT = "Mt * CO2e"

# The emissions record starts well before this, but nothing downstream reads
# the earlier years and they are sparse enough to be misleading.
FIRST_YEAR = 1850

_SECTOR_DIMENSION = "category (IPCC2006_PRIMAP)"
_SCENARIO_COLUMN = "scenario (PRIMAP-hist)"
_AREA_COLUMN = "area (ISO3)"


def load_emissions(
    inputs: Inputs, categories: Iterable[str]
) -> dict[str, pd.DataFrame]:
    """Load observed emissions for several categories in one pass.

    Parameters
    ----------
    inputs
        The run's inputs.
    categories
        The categories to produce. `non-co2` is derived by subtraction; the
        categories it is derived from are read automatically whether or not
        they were asked for.

    Returns
    -------
    dict
        Category to a frame indexed by ``(iso3c, unit, emission-category)``
        with one string column per year.

    Raises
    ------
    ConfigurationError
        If a category is unknown, or the source does not offer it.
    DataLoadingError
        If the file is missing, or a category selects no rows.
    """
    import xarray as xr

    wanted = list(dict.fromkeys(categories))
    unknown = [
        category
        for category in wanted
        if category not in CATEGORY_DEFINITIONS and category != DERIVED_CATEGORY
    ]
    if unknown:
        raise ConfigurationError(
            f"unknown emission categories {unknown}. Known: "
            f"{sorted([*CATEGORY_DEFINITIONS, DERIVED_CATEGORY])}"
        )

    parameters = inputs.parameters("emissions")
    available = parameters.get("available_categories", [])
    missing = [category for category in wanted if category not in available]
    if missing:
        raise ConfigurationError(
            f"the {inputs.sources.emissions!r} emissions source does not offer "
            f"{missing}. It offers: {sorted(available)}"
        )

    # Read whatever the derivation needs, even when the caller did not ask for
    # its parts -- the alternative is a second pass over the netCDF.
    to_read = [category for category in wanted if category != DERIVED_CATEGORY]
    if DERIVED_CATEGORY in wanted:
        to_read.extend(part for part in DERIVED_FROM if part not in to_read)

    path = inputs.path("emissions")
    if not path.exists():
        raise DataLoadingError(
            f"the emissions source points at {path}, which does not exist."
        )

    scenario = parameters.get("scenario")
    logger.info("reading %s (%s) for %d categories", path.name, scenario, len(to_read))

    frames: dict[str, pd.DataFrame] = {}
    with xr.open_dataset(path) as dataset:
        for category in to_read:
            frames[category] = _extract(dataset, category, scenario)

    if DERIVED_CATEGORY in wanted:
        frames[DERIVED_CATEGORY] = derive_non_co2(*(frames[p] for p in DERIVED_FROM))

    return {category: frames[category] for category in wanted}


def _extract(dataset, category: str, scenario: str | None) -> pd.DataFrame:
    """Pull one category out of the open netCDF and convert its units."""
    definition = CATEGORY_DEFINITIONS[category]
    variable = str(definition["variable"])
    sectors: list[str] = list(definition["sectors"])  # type: ignore[arg-type]

    registry = get_default_unit_registry()
    source_units = dataset[variable].attrs.get("units")

    pieces = []
    for sector in sectors:
        try:
            selected = (
                dataset[variable]
                .sel({_SECTOR_DIMENSION: sector})
                .to_dataframe()
                .reset_index()
            )
        except Exception as error:
            raise DataProcessingError(
                f"cannot select {variable!r} sector {sector!r}: {error}"
            ) from error

        selected = selected[selected[_SCENARIO_COLUMN] == scenario]
        if selected.empty:
            raise DataLoadingError(
                f"no rows for variable {variable!r} sector {sector!r} in "
                f"scenario {scenario!r}"
            )

        selected["year"] = selected["time"].dt.year.astype(int)
        selected = selected.rename(columns={_AREA_COLUMN: "iso3c"})
        selected = selected[selected["year"] >= FIRST_YEAR]
        selected = selected[["iso3c", "year", variable]].copy()

        selected = selected.set_index(["iso3c", "year"])
        selected["unit"] = source_units
        selected = selected.set_index("unit", append=True)
        selected = set_single_unit(df=selected, unit_level="unit", ur=registry)
        converted = convert_unit_robust(
            selected, WORKING_UNIT, unit_level="unit", ur=registry
        )
        pieces.append(converted.reset_index().rename(columns={variable: category}))

    # Several sectors make one category, so they are summed per country-year.
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.groupby(["iso3c", "year"])[category].sum().reset_index()

    wide = combined.pivot_table(index=["iso3c"], columns="year", values=category)
    wide = ensure_string_year_columns(wide)
    wide.index = pd.MultiIndex.from_tuples(
        [(iso3c, WORKING_UNIT, category) for iso3c in wide.index],
        names=["iso3c", "unit", "emission-category"],
    )
    logger.debug("%s: %d countries, %d years", category, len(wide), wide.shape[1])
    return wide


def derive_non_co2(all_ghg_ex_lulucf: pd.DataFrame, co2_ffi: pd.DataFrame):
    """Derive non-CO2 emissions by subtraction.

    Never read directly from the source: taking the difference guarantees that
    the parts of a decomposition add back up to the whole, which reading two
    independently-rounded series would not.

    Parameters
    ----------
    all_ghg_ex_lulucf, co2_ffi
        Frames as `load_emissions` returns them.

    Returns
    -------
    pandas.DataFrame
        The difference, indexed like its inputs but labelled ``non-co2``. A
        year present in only one input yields NaN rather than the untouched
        other value.
    """
    left = all_ghg_ex_lulucf.reset_index(level=["unit", "emission-category"], drop=True)
    right = co2_ffi.reset_index(level=["unit", "emission-category"], drop=True)

    countries = left.index.intersection(right.index)
    years = [column for column in left.columns if column in right.columns]
    difference = left.loc[countries, years] - right.loc[countries, years]

    difference.index = pd.MultiIndex.from_tuples(
        [(iso3c, WORKING_UNIT, DERIVED_CATEGORY) for iso3c in difference.index],
        names=["iso3c", "unit", "emission-category"],
    )
    logger.debug("non-co2 derived for %d countries", len(difference))
    return difference
