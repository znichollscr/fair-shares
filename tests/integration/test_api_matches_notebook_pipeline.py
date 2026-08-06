"""The two paths must agree: `fair_shares.api` versus the notebook pipeline.

`fair_shares.api` reimplements, in Python, the preprocessing that notebooks
100-108 perform through Snakemake and papermill. Reimplementation is only
defensible if it is *checked*, and checked against the thing it replaced rather
than against its own expectations -- so this builds a processed tree the old
way, preprocesses the same inputs the new way, and compares what comes out.

**Where the inputs come from.** Third-party raw data is no longer committed, and
re-fetching it does not reproduce it -- publishers reissue files under the same
name (the GCB workbook moved v1.0 -> v1.01 and shifted the bunker column by
~1.8%, which lands in every allocated total). So this test takes the pinned
vintages out of **git history**, where they still are, and writes them into the
repository's own `data/` tree, which is gitignored. Files already present are
left alone.

    Consequence, stated plainly: **this test breaks if history is rewritten.**
    A filter-branch, a squash of the early commits, or a fresh shallow clone
    will all take the pinned inputs with them. The commits it reads are listed
    in `PINNED_INPUTS` below; if history is rewritten, those hashes have to be
    updated to wherever the files ended up, or the test retired.

**It is slow** -- Snakemake and papermill, minutes -- and it moves ~180 MB
around, so it is opt-in:

    FAIR_SHARES_RUN_PIPELINE_COMPARISON=1 pytest tests/integration/test_api_matches_notebook_pipeline.py

Everything runs repository-relative, because the notebook pipeline's Snakemake
build resolves its own paths from the working directory and does not honour an
explicitly passed data or output directory.

Two differences are expected, asserted loosely, and explained where they are
checked: the rest-of-world aggregate's last few digits, and -- for composite
categories only -- the `has_emissions` coverage column.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_shares.api import Inputs, SourceSelection, preprocess

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RUN_ENV = "FAIR_SHARES_RUN_PIPELINE_COMPARISON"

# The pinned input vintages, and the last commit that still carries each. Read
# out of git history rather than fetched, because re-fetching does not
# reproduce them. **Rewriting history invalidates these.**
PINNED_INPUTS = {
    "data/emissions/primap-202503/"
    "Guetschow_et_al_2025-PRIMAP-hist_v2.6.1_final_no_rounding_13-Mar-2025.nc": (
        "5a66bdd"
    ),
    "data/gdp/wdi-2025/API_NY.GDP.MKTP.PP.KD_DS2_en_csv_v2_1004.csv": "e1c9263",
    "data/gdp/wdi-2025/API_NY.GDP.MKTP.KD_DS2_en_csv_v2_213435.csv": "e1c9263",
    "data/population/un-owid-2025/population.csv": "5a66bdd",
    "data/population/un-owid-2025/UN_PPP2024_Output_PopTot.xlsx": "5a66bdd",
    "data/gini/wdi-2025/API_SI.POV.GINI_DS2_en_csv_v2.csv": "df98729",
    "data/bunkers/gcb-2024/National_Fossil_Carbon_Emissions_2024v1.0.xlsx": "35bc5fa",
    "data/lulucf/melo-2026/timeseries_NGHGI_v3.1.csv": "6cd930b",
    "data/scenarios/ipcc_ar6_gidden/ar6_gidden.zip": "5a66bdd",
    "data/regions/iso3c_region_mapping_20240319.csv": "5a66bdd",
    "data/rcbs/rcbs.yaml": "a5084f0",
}

# One configuration the notebook path supports, chosen because it is the one
# with the most consumers: a fossil-CO2 budget run against the packaged
# sources. It exercises every preprocessing step the API implements.
CATEGORY = "co2-ffi"
TARGET = "rcbs"
SOURCES = SourceSelection(
    emissions="primap-202503",
    gdp="wdi-2025",
    population="un-owid-2025",
    gini="wdi-2025",
    bunkers="gcb-2024",
    scenarios="ar6",
)

# The rest-of-world row is what the world total leaves after every analysis
# country is subtracted from it -- a residual of near-total cancellation. Its
# last bits therefore depend on the order ~200 countries are summed in, and the
# two paths sum in different orders (one from a CSV, one from a pivot). The
# absolute size of the disagreement is femtograms; the relative size, on a
# residual, is not a meaningful number to hold to 1e-12.
ROW = "ROW"
COUNTRY_RTOL = 1e-12
ROW_RTOL = 1e-9


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _restore_pinned_inputs(root: Path) -> list[str]:
    """Write the pinned inputs out of git history, skipping any already there.

    Returns
    -------
    list of str
        What had to be restored, for the failure message when git cannot.
    """
    restored = []
    for relative, commit in PINNED_INPUTS.items():
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{relative}"],
                capture_output=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            pytest.skip(
                f"cannot read {relative} from git history at {commit}: {error}. "
                "History may have been rewritten -- see this module's docstring."
            )
        target.write_bytes(blob)
        restored.append(relative)
    return restored


@pytest.fixture(scope="module")
def data_dir() -> Path:
    if not os.environ.get(RUN_ENV):
        pytest.skip(f"set {RUN_ENV}=1 to run this; it takes minutes.")
    root = _repository_root()
    _restore_pinned_inputs(root)
    return root / "data"


@pytest.fixture(scope="module")
def notebook_processed_dir(data_dir) -> Path:
    """Build a processed tree the old way: Snakemake, papermill, CSVs.

    Repository-relative, and run from the repository root, because that build
    locates its own Snakefile and resolves its own output paths from the
    working directory rather than from what it was passed.
    """
    from fair_shares.library.utils.data.setup import setup_data

    active = {
        key: value
        for key, value in SOURCES.as_dict().items()
        if value is not None and key != "scenarios"
    }
    root = _repository_root()
    previous = Path.cwd()
    os.chdir(root)
    try:
        setup_info = setup_data(
            emission_category=CATEGORY,
            active_sources={**active, "target": TARGET},
            verbose=False,
        )
    finally:
        os.chdir(previous)
    return Path(setup_info["paths"]["processed_dir"])


@pytest.fixture(scope="module")
def api_result(data_dir):
    """Preprocess the same inputs the new way: functions and frames."""
    inputs = Inputs(sources=SOURCES, data_dir=data_dir)
    return preprocess(
        inputs,
        CATEGORY,
        rcb_definition=data_dir / "rcbs" / "rcbs.yaml",
    )


def _from_csv(path: Path, index: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.set_index(index).sort_index()


def _as_frame(frame: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    return frame.reset_index().set_index(index).sort_index()


def _assert_agrees(expected: pd.DataFrame, actual: pd.DataFrame, what: str) -> None:
    """Assert two frames agree, allowing the rest-of-world its last digits."""
    assert list(expected.index) == list(actual.index), f"{what}: different countries"
    shared = [c for c in expected.columns if c in actual.columns]
    assert shared, f"{what}: no columns in common"

    is_row = np.array(
        [ROW in (idx if isinstance(idx, tuple) else (idx,)) for idx in expected.index]
    )
    left = expected[shared].to_numpy(dtype=float)
    right = actual[shared].to_numpy(dtype=float)

    tolerance = np.where(is_row[:, None], ROW_RTOL, COUNTRY_RTOL)
    difference = np.abs(left - right)
    allowed = tolerance * np.abs(left)
    disagrees = (difference > allowed) & ~(np.isnan(left) & np.isnan(right))

    if disagrees.any():
        row, column = np.argwhere(disagrees)[0]
        raise AssertionError(
            f"{what}: {disagrees.sum()} of {disagrees.size} values disagree, "
            f"first at {expected.index[row]} / {shared[column]}: "
            f"notebook={left[row, column]!r} api={right[row, column]!r}"
        )


class TestPreprocessingAgrees:
    """Every product the notebook pipeline writes, against the API's frames."""

    def test_country_emissions(self, notebook_processed_dir, api_result):
        index = ["iso3c", "unit", "emission-category"]
        _assert_agrees(
            _from_csv(
                notebook_processed_dir / f"country_emissions_{CATEGORY}_timeseries.csv",
                index,
            ),
            _as_frame(api_result.parts[CATEGORY], index),
            "country emissions",
        )

    def test_country_gdp(self, notebook_processed_dir, api_result):
        index = ["iso3c", "unit"]
        _assert_agrees(
            _from_csv(notebook_processed_dir / "country_gdp_timeseries.csv", index),
            _as_frame(api_result.gdp, index),
            "country GDP",
        )

    def test_country_population(self, notebook_processed_dir, api_result):
        index = ["iso3c", "unit"]
        _assert_agrees(
            _from_csv(
                notebook_processed_dir / "country_population_timeseries.csv", index
            ),
            _as_frame(api_result.population, index),
            "country population",
        )

    def test_country_gini(self, notebook_processed_dir, api_result):
        index = ["iso3c", "unit"]
        _assert_agrees(
            _from_csv(notebook_processed_dir / "country_gini_stationary.csv", index),
            _as_frame(api_result.gini, index),
            "country Gini",
        )

    def test_remaining_carbon_budgets(self, notebook_processed_dir, api_result):
        index = ["source", "scenario"]
        expected = _from_csv(notebook_processed_dir / f"rcbs_{CATEGORY}.csv", index)
        actual = api_result.rcbs.set_index(index).sort_index()
        numeric = [
            column
            for column in expected.columns
            if pd.api.types.is_numeric_dtype(expected[column])
            and column in actual.columns
        ]
        _assert_agrees(expected[numeric], actual[numeric], "remaining budgets")

    def test_the_same_countries_are_in_the_analysis(
        self, notebook_processed_dir, api_result
    ):
        # `in_analysis` is what everything downstream reads. `has_emissions` is
        # deliberately not compared: for composite categories the notebook
        # pipeline's copy records whichever decomposition pass overwrote the
        # file last, because each pass wrote the same filename.
        expected = pd.read_csv(
            notebook_processed_dir / "country_data_coverage_summary.csv"
        ).set_index("iso3c")["in_analysis"]
        actual = api_result.coverage.set_index("iso3c")["in_analysis"]
        pd.testing.assert_series_equal(
            expected.sort_index(), actual.sort_index(), check_names=False
        )


class TestAllocationsAgree:
    """The allocations themselves, which is what any of this is for."""

    def test_equal_per_capita_budget(self, notebook_processed_dir, api_result):
        from fair_shares.api import AllocationRequest, ClimateTarget, allocate
        from fair_shares.library.notebook_helpers import (
            load_allocation_data,
            run_all_allocations,
        )

        target = ClimateTarget("1.5C", 0.5, "ar6_2020")
        parameters = {"allocation_year": 1990, "preserve_allocation_year_shares": True}

        # The old way: load the CSVs back off disk, then allocate.
        loaded = load_allocation_data(
            processed_dir=notebook_processed_dir,
            target=TARGET,
            final_categories=[CATEGORY],
            emission_category=CATEGORY,
        )
        from fair_shares.library.python_api import _filter_to_anchor

        _filter_to_anchor(loaded, target.reference, target.limit, target.quantile)
        _rows, expected = run_all_allocations(
            allocations={"equal-per-capita-budget": [dict(parameters)]},
            loaded_data=loaded,
            output_dir=None,
            data_context={"source-id": "notebook"},
            target=TARGET,
            final_categories=[CATEGORY],
            harmonisation_year=None,
            write=False,
            return_allocations=True,
        )

        # The new way: allocate the frames we already hold.
        actual = allocate(
            api_result,
            [
                AllocationRequest(
                    approach="equal-per-capita-budget",
                    parameters=parameters,
                    climate_target=target,
                )
            ],
            sources=SOURCES,
            target=TARGET,
        )[0].allocations

        year = str(parameters["allocation_year"])
        _assert_agrees(
            expected.set_index("iso3c")[[year]].sort_index(),
            actual.set_index("iso3c")[[year]].sort_index(),
            "equal-per-capita allocation",
        )
