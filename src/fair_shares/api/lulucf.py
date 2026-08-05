"""NGHGI-convention LULUCF, and the categories that depend on it.

Ported from notebook 107. Two different conventions describe the same land
carbon flux: bookkeeping models (what PRIMAP carries) and national greenhouse
gas inventories (what countries report). They disagree by a large and
systematic margin, so any category containing LULUCF CO2 has to say which
convention it is in.

Only three categories depend on the choice -- ``co2-lulucf``, ``co2`` and
``all-ghg``. ``co2-ffi``, ``non-co2`` and ``all-ghg-ex-co2-lulucf`` contain no
LULUCF CO2 and are the same either way. The notebook pipeline expressed that
distinction as a filename suffix (``_nghgi``) and left every consumer to
re-derive which variant it wanted; here the corrected categories simply replace
the uncorrected ones in the mapping that is returned, so there is nothing to
re-derive and no way to read the wrong file.
"""

from __future__ import annotations

import logging

import pandas as pd

from fair_shares.api.emissions import WORKING_UNIT
from fair_shares.api.sources import Inputs
from fair_shares.library.exceptions import DataLoadingError, DataProcessingError
from fair_shares.library.preprocessing.paths import NGHGI_CORRECTED_CATEGORIES
from fair_shares.library.utils import ensure_string_year_columns

logger = logging.getLogger(__name__)

# The row the source uses for its world total, and the label the world series
# carries once extracted.
_SOURCE_WORLD_KEY = "WRD"
WORLD_SERIES_LABEL = "nghgi_lulucf"

__all__ = [
    "NGHGI_CORRECTED_CATEGORIES",
    "WORLD_SERIES_LABEL",
    "apply_nghgi_corrections",
    "load_lulucf",
    "world_lulucf",
]


def load_lulucf(inputs: Inputs, world_key: str) -> pd.DataFrame:
    """Load NGHGI-convention LULUCF CO2 fluxes.

    Parameters
    ----------
    inputs
        The run's inputs.
    world_key
        The ISO3-ish code the emissions source uses for the world. The LULUCF
        source has its own (``WRD``), and the two have to agree before the
        series can be combined -- so the world row is relabelled here rather
        than everywhere it is used.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(iso3c, unit, emission-category)`` with
        ``emission-category`` of ``co2-lulucf``, one string column per year.
        Negative values are net sinks.

    Raises
    ------
    DataLoadingError
        If the file is missing.
    DataProcessingError
        If the source carries no world total.
    """
    parameters = inputs.parameters("lulucf")
    path = inputs.path("lulucf")
    if not path.exists():
        raise DataLoadingError(
            f"the lulucf source points at {path}, which does not exist."
        )

    logger.debug("reading LULUCF from %s", path)
    raw = pd.read_csv(path)

    # Regional aggregates would double-count against their members.
    excluded = parameters.get("exclude_regions", [])
    selected = raw[
        (
            raw[parameters.get("category_column", "Category")]
            == parameters["category_filter"]
        )
        & (raw[parameters.get("gas_column", "Gas")] == parameters["gas_filter"])
        & (~raw[parameters["iso3_column"]].isin(excluded))
    ].copy()

    wide = selected.pivot_table(
        index=parameters["iso3_column"],
        columns=parameters["year_column"],
        values=parameters["value_column"],
        aggfunc="sum",
    )
    wide.index.name = "iso3c"
    wide = ensure_string_year_columns(wide)

    world_rows = wide.index == _SOURCE_WORLD_KEY
    countries = wide.loc[~world_rows].copy()
    world = wide.loc[world_rows].copy()
    if world.empty:
        raise DataProcessingError(
            f"the LULUCF source carries no {_SOURCE_WORLD_KEY!r} world total, "
            "which the remaining-budget corrections need."
        )
    world.index = pd.Index([world_key], name="iso3c")

    combined = pd.concat([countries, world])
    combined.index = pd.MultiIndex.from_tuples(
        [(iso3c, WORKING_UNIT, "co2-lulucf") for iso3c in combined.index],
        names=["iso3c", "unit", "emission-category"],
    )
    years = sorted(int(column) for column in combined.columns)
    logger.info(
        "LULUCF (NGHGI): %d countries, %d-%d", len(countries), years[0], years[-1]
    )
    return combined


def world_lulucf(lulucf: pd.DataFrame, world_key: str) -> pd.DataFrame:
    """Extract the world LULUCF series the budget corrections consume.

    Parameters
    ----------
    lulucf
        What `load_lulucf` returned.
    world_key
        The world code used in that frame.

    Returns
    -------
    pandas.DataFrame
        A single row indexed `WORLD_SERIES_LABEL`, one string column per year.
    """
    world = lulucf[lulucf.index.get_level_values("iso3c") == world_key]
    values = world.reset_index(level=["unit", "emission-category"], drop=True)
    return pd.DataFrame(
        values.to_numpy(),
        columns=values.columns,
        index=pd.Index([WORLD_SERIES_LABEL], name="source"),
    )


def _combine(
    left: pd.DataFrame, right: pd.DataFrame, operation: str, category: str
) -> pd.DataFrame:
    """Add or subtract two category frames over what they have in common.

    The intersection is deliberate on both axes. The LULUCF record is much
    shorter than the emissions record, so a result that contains LULUCF must
    not claim to extend beyond it -- filling the missing years would invent a
    land flux of zero, which is not a neutral assumption.
    """
    a = left.reset_index(level="emission-category", drop=True)
    b = right.reset_index(level="emission-category", drop=True)

    shared_years = sorted(set(a.columns) & set(b.columns))
    a, b = a[shared_years], b[shared_years]
    shared_index = a.index.intersection(b.index)
    a, b = a.loc[shared_index], b.loc[shared_index]

    if operation == "add":
        result = a.add(b, fill_value=0)
    elif operation == "subtract":
        result = a.subtract(b, fill_value=0)
    else:
        raise ValueError(f"unknown operation {operation!r}")

    result.index = pd.MultiIndex.from_tuples(
        [(iso3c, unit, category) for iso3c, unit in result.index],
        names=["iso3c", "unit", "emission-category"],
    )
    return ensure_string_year_columns(result)


def apply_nghgi_corrections(
    categories: dict[str, pd.DataFrame], lulucf: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Re-express the LULUCF-dependent categories in the NGHGI convention.

    ``co2`` becomes fossil CO2 plus inventory LULUCF rather than PRIMAP's
    bookkeeping total, and ``all-ghg`` follows from it. ``co2-lulucf`` is
    replaced outright. Everything else is passed through untouched, because
    nothing else contains LULUCF CO2.

    Parameters
    ----------
    categories
        Category to frame, as `load_emissions` returns. Must contain whatever
        the corrected categories are built from: ``co2-ffi`` for ``co2``, and
        additionally ``non-co2`` (or the parts to derive it) for ``all-ghg``.
    lulucf
        What `load_lulucf` returned.

    Returns
    -------
    dict
        The same keys, with the LULUCF-dependent ones recomputed.

    Raises
    ------
    DataProcessingError
        If a correction is asked for without the inputs it needs.
    """
    corrected = dict(categories)
    wanted = NGHGI_CORRECTED_CATEGORIES & set(categories)
    if not wanted:
        return corrected

    if "co2-ffi" not in categories:
        raise DataProcessingError(
            f"cannot apply NGHGI corrections to {sorted(wanted)} without "
            "co2-ffi, which every corrected category is built from."
        )

    if "co2-lulucf" in wanted:
        corrected["co2-lulucf"] = lulucf

    # `co2` is needed for `all-ghg` whether or not the caller asked for it.
    co2 = _combine(categories["co2-ffi"], lulucf, "add", "co2")
    if "co2" in wanted:
        corrected["co2"] = co2

    if "all-ghg" in wanted:
        non_co2 = categories.get("non-co2")
        if non_co2 is None:
            if "all-ghg-ex-co2-lulucf" not in categories:
                raise DataProcessingError(
                    "cannot apply the NGHGI correction to all-ghg without "
                    "non-co2, or the all-ghg-ex-co2-lulucf it derives from."
                )
            non_co2 = _combine(
                categories["all-ghg-ex-co2-lulucf"],
                categories["co2-ffi"],
                "subtract",
                "non-co2",
            )
        corrected["all-ghg"] = _combine(co2, non_co2, "add", "all-ghg")

    logger.info("NGHGI corrections applied to %s", sorted(wanted))
    return corrected
