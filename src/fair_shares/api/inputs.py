"""Reading raw inputs into frames.

One function per source kind. Each takes an :class:`~fair_shares.api.sources.
Inputs`, opens exactly the files that source declares, and returns a frame --
no writing, no directory creation, no global state.

The transformations here were lifted from notebooks 102, 103, 105 and 108,
where they sat under a hundred-odd lines of papermill parameter handling and
progress printing. The arithmetic is unchanged; what has gone is everything
that was only there to let one notebook hand data to the next.

Conventions, applied uniformly so the pipeline can stop special-casing:

* year columns are **strings**, sorted ascending;
* the index is ``(iso3c, unit)``, so a frame always says what its numbers are;
* magnitudes are the pipeline's working units (Mt CO2 / yr, million people,
  billion USD), converted here rather than at the point of use.
"""

from __future__ import annotations

import logging

import pandas as pd

from fair_shares.api.sources import Inputs
from fair_shares.library.exceptions import ConfigurationError, DataLoadingError
from fair_shares.library.utils import ensure_string_year_columns
from fair_shares.library.utils.units import get_default_unit_registry

logger = logging.getLogger(__name__)

# GDP and Gini are published as wide World Bank sheets whose first four rows
# are provenance notes rather than data.
_WORLD_BANK_HEADER_ROWS = 4

# Historical population is trimmed to the era the emissions record covers;
# earlier values exist but nothing consumes them.
_FIRST_POPULATION_YEAR = 1850

# GDP before this is too sparse to allocate on.
_FIRST_GDP_YEAR = 1990


def _require(path, kind: str):
    """Return `path`, or raise naming the source that pointed at it."""
    if not path.exists():
        raise DataLoadingError(
            f"the {kind} source points at {path}, which does not exist. "
            "Check `Inputs.data_dir`, or pass an explicit override."
        )
    return path


def _with_unit(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Attach a unit level to an ``iso3c``-indexed frame."""
    frame = frame.copy()
    frame.index = pd.MultiIndex.from_tuples(
        [(iso3c, unit) for iso3c in frame.index], names=["iso3c", "unit"]
    )
    return frame


def load_gdp(inputs: Inputs) -> pd.DataFrame:
    """Load GDP as a timeseries, in billions.

    Parameters
    ----------
    inputs
        The run's inputs. Reads the ``gdp`` source's PPP or MER file according
        to its ``gdp_variant`` parameter.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(iso3c, unit)``, one string column per year from 1990.

    Raises
    ------
    ConfigurationError
        If ``gdp_variant`` is neither ``"PPP"`` nor ``"MER"``.
    DataLoadingError
        If the file is missing.
    """
    parameters = inputs.parameters("gdp")
    variant = parameters.get("gdp_variant", "PPP")
    if variant not in ("PPP", "MER"):
        raise ConfigurationError(
            f"invalid GDP variant {variant!r}. Must be 'PPP' or 'MER'."
        )

    path = _require(inputs.path("gdp", f"path_{variant.lower()}"), "gdp")
    logger.debug("reading %s GDP from %s", variant, path)
    raw = pd.read_csv(path, skiprows=_WORLD_BANK_HEADER_ROWS)

    identifiers = ["Country Name", "Country Code"]
    years = [column for column in raw.columns if column.isdigit()]
    long = raw[identifiers + years].melt(
        id_vars=identifiers, value_vars=years, var_name="year", value_name="gdp"
    )
    long = long.rename(columns={"Country Code": "iso3c"})
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["gdp"])
    long = long[long["year"] >= _FIRST_GDP_YEAR]

    wide = long.pivot_table(index=["iso3c"], columns="year", values="gdp")
    # Billions, so downstream arithmetic is not carried out on 1e12-scale
    # floats alongside Mt-scale emissions.
    wide = ensure_string_year_columns(wide) * 1e-9
    logger.info("GDP: %d countries, %d years", len(wide), wide.shape[1])
    return _with_unit(wide, "billion")


def load_population(inputs: Inputs) -> pd.DataFrame:
    """Load population as a timeseries, in millions.

    Historical observations are used wherever they exist; the projection
    supplies the years after the last observed one, so the two never overlap
    and the join cannot double-count.

    Parameters
    ----------
    inputs
        The run's inputs.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(iso3c, unit)``, one string column per year.

    Raises
    ------
    DataLoadingError
        If either file is missing.
    """
    parameters = inputs.parameters("population")
    historical_world = parameters.get("historical_world_key")
    projected_world = parameters.get("projected_world_key")

    historical_path = _require(
        inputs.path("population", "path_historical"), "population"
    )
    logger.debug("reading historical population from %s", historical_path)
    historical = pd.read_csv(historical_path)
    historical = historical[historical["Code"].notna()]
    historical = historical[historical["Year"] >= _FIRST_POPULATION_YEAR]
    historical = historical.rename(
        columns={
            "Code": "iso3c",
            "Year": "year",
            "Population (historical estimates)": "population",
        }
    )
    # Millions.
    historical["population"] = historical["population"] * 1e-6
    historical = historical[["iso3c", "year", "population"]].dropna()

    projected_path = _require(inputs.path("population", "path_projected"), "population")
    logger.debug("reading projected population from %s", projected_path)
    projected = pd.read_excel(projected_path, sheet_name="Median", header=16)

    year_columns = [
        column
        for column in projected.columns
        if isinstance(column, int | float) and column >= _FIRST_POPULATION_YEAR
    ]
    region_column = "Region, subregion, country or area *"
    identifiers = ["ISO3 Alpha-code", region_column]
    projected = projected.melt(
        id_vars=identifiers,
        value_vars=year_columns,
        var_name="year",
        value_name="population",
    )
    # The world row carries no ISO3 code, so it is identified by name and then
    # given one -- otherwise the world total is dropped with the aggregates.
    world_rows = (projected["ISO3 Alpha-code"].isna()) & (
        projected[region_column] == projected_world
    )
    projected.loc[world_rows, "ISO3 Alpha-code"] = projected_world
    projected = projected.drop(columns=[region_column])
    projected = projected[projected["ISO3 Alpha-code"].notna()]
    projected["year"] = projected["year"].astype(int)
    # Thousands in the source; millions here.
    projected["population"] = projected["population"] * 1e-3
    projected = projected.dropna(subset=["population"])
    projected = projected.rename(columns={"ISO3 Alpha-code": "iso3c"})
    projected.loc[projected["iso3c"] == projected_world, "iso3c"] = historical_world

    # Observations win where both exist, so the projection contributes only
    # the years after the last observed one and the join cannot double-count.
    last_observed = int(historical["year"].max())
    projected = projected[projected["year"] > last_observed]
    logger.info(
        "population: observed through %d, projected %d onwards",
        last_observed,
        last_observed + 1,
    )

    combined = pd.concat([historical, projected], ignore_index=True)
    combined["year"] = combined["year"].astype(int)
    combined = combined.sort_values(["iso3c", "year"])
    combined = combined.drop_duplicates(["iso3c", "year"], keep="first")

    combined = _fill_year_gaps(combined, "population")

    variant = parameters.get("projected_variant")
    combined = _apply_projection_variant(combined, variant)

    wide = combined.pivot_table(index=["iso3c"], columns="year", values="population")
    return _with_unit(ensure_string_year_columns(wide), "million")


def _fill_year_gaps(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Reindex each country onto a gap-free year range and interpolate.

    A country whose observations stop before the global last-observed year
    would otherwise carry a hole between its own final observation and the
    first projected year -- the projection only contributes years after the
    *global* last observation, not each country's.

    Parameters
    ----------
    frame
        Long, with `iso3c`, `year` and `column`.
    column
        The value column to interpolate.

    Returns
    -------
    pandas.DataFrame
        Long, gap-free within each country's own span.
    """
    index = (
        frame.groupby("iso3c")["year"]
        .apply(lambda years: pd.Series(range(years.min(), years.max() + 1)))
        .reset_index(level=1, drop=True)
        .rename("year")
    )
    filled = (
        frame.set_index(["iso3c", "year"]).reindex(index.reset_index()).reset_index()
    )
    filled[column] = pd.to_numeric(filled[column], errors="coerce")
    filled[column] = filled.groupby("iso3c")[column].transform(
        lambda group: group.interpolate()
    )
    # Six decimals of a million is single people; more is noise from the
    # interpolation rather than information from the source.
    filled[column] = filled[column].round(6)
    return filled.sort_values(["iso3c", "year"])


def _apply_projection_variant(frame: pd.DataFrame, variant: str | None) -> pd.DataFrame:
    """Apply the source's rule for how far the projection is trusted.

    Parameters
    ----------
    frame
        Long population, gap-free.
    variant
        ``"un-median-to-2100"`` uses the projection as published;
        ``"un-median-to-2050"`` holds each country flat after 2050;
        ``"hist-frozen"`` holds each country flat after the current year.

    Returns
    -------
    pandas.DataFrame
        The frame, adjusted.

    Raises
    ------
    ConfigurationError
        If the variant is not one of the three.
    """
    if variant in (None, "un-median-to-2100"):
        return frame

    if variant == "un-median-to-2050":
        freeze_year = 2050
    elif variant == "hist-frozen":
        from datetime import datetime

        freeze_year = datetime.now().year
    else:
        raise ConfigurationError(
            f"unknown projected_variant {variant!r}. Expected one of "
            "'un-median-to-2100', 'un-median-to-2050', 'hist-frozen'."
        )

    # Vectorised where the notebook looped per country: same result, and it
    # does not depend on every country having the freeze year.
    held = (
        frame[frame["year"] == freeze_year]
        .set_index("iso3c")["population"]
        .reindex(frame["iso3c"].unique())
    )
    future = frame["year"] > freeze_year
    replacement = frame.loc[future, "iso3c"].map(held)
    frame.loc[future, "population"] = replacement.where(
        replacement.notna(), frame.loc[future, "population"]
    )
    return frame


def load_gini(inputs: Inputs, countries: set[str]) -> pd.DataFrame:
    """Load one Gini coefficient per country.

    Survey Gini is sparse in any single year, so the selection rule takes the
    latest observation within a window rather than a fixed year -- a
    single-year rule would more than halve coverage.

    Parameters
    ----------
    inputs
        The run's inputs.
    countries
        ISO3 codes to keep. The published file mixes countries and regional
        aggregates under the same column, and only the caller knows which
        countries are in play.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(iso3c, unit)`` with a single ``gini`` column, on 0-1.

    Raises
    ------
    ConfigurationError
        If the source asks for a selection rule this loader does not implement.
    ValueError
        If any value falls outside 0-1 after conversion.
    """
    parameters = inputs.parameters("gini")
    selection = parameters.get("selection")
    if selection != "latest-available":
        raise ConfigurationError(
            f"this loader implements selection 'latest-available', got "
            f"{selection!r}. WDI publishes no quality flag, so a "
            "quality-preferring rule cannot be applied to it."
        )

    path = _require(inputs.path("gini"), "gini")
    logger.debug("reading Gini from %s", path)
    raw = pd.read_csv(path, skiprows=_WORLD_BANK_HEADER_ROWS)

    identifiers = ["Country Name", "Country Code"]
    years = [column for column in raw.columns if column.isdigit()]
    long = raw[identifiers + years].melt(
        id_vars=identifiers, value_vars=years, var_name="year", value_name="gini"
    )
    long = long.rename(columns={"Country Code": "iso3c"})
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["gini"])
    long = long[long["iso3c"].isin(countries)]

    window = parameters.get("year_window")
    if window:
        first, last = window
        long = long[(long["year"] >= first) & (long["year"] <= last)]

    latest = long.sort_values(["iso3c", "year"]).groupby("iso3c").tail(1)
    values = latest[["iso3c", "gini"]].copy()
    # Published as a percentage; the allocators expect a fraction.
    values["gini"] = values["gini"] / 100.0

    outside = values[(values["gini"] < 0) | (values["gini"] > 1)]
    if not outside.empty:
        raise ValueError(f"Gini values outside 0-1 after conversion:\n{outside}")

    logger.info("Gini: %d countries with an observation", len(values))
    return _with_unit(values.set_index("iso3c")[["gini"]], "unitless")


def load_bunkers(inputs: Inputs) -> pd.DataFrame:
    """Load international bunker CO2 as a single timeseries.

    Aviation and shipping emissions are not attributable to a territory, so
    they are deducted from the global remaining carbon budget before any
    country is allocated a share -- which makes this source move every
    allocated total in a budget run.

    Parameters
    ----------
    inputs
        The run's inputs.

    Returns
    -------
    pandas.DataFrame
        A single row indexed ``"bunkers"``, one string column per year, in
        Mt CO2 / yr.

    Raises
    ------
    DataLoadingError
        If the file is missing.
    """
    parameters = inputs.parameters("bunkers")
    path = _require(inputs.path("bunkers"), "bunkers")
    logger.debug("reading bunkers from %s", path)

    sheet = pd.read_excel(
        path,
        sheet_name=parameters["sheet_name"],
        header=parameters["header_row"],
        index_col=0,
    )
    column = sheet[parameters["bunker_column"]].dropna()

    def _is_year(value: object) -> bool:
        try:
            return 1800 <= int(value) <= 2200  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return False

    column = column[column.index.map(_is_year)]
    column.index = column.index.astype(int)

    # Published as carbon; the budget is carbon dioxide.
    registry = get_default_unit_registry()
    column = column * (1 * registry("Mt C")).to("Mt CO2").magnitude

    years = sorted(column.index)
    logger.info("bunkers: %d-%d", years[0], years[-1])
    return pd.DataFrame(
        [column[years].to_numpy()],
        columns=[str(int(year)) for year in years],
        index=pd.Index(["bunkers"], name="source"),
    )
