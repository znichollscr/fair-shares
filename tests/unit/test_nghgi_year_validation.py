"""
Tests for NGHGI validation in LULUCF-containing allocations.

There are two ways to allocate a category containing LULUCF CO2, and only one
of them is bounded here:

* **with an NGHGI correction** -- the run names a LULUCF source, and can only
  be allocated over the years that source covers;
* **without one** -- the run allocates the emission source's own totals as
  published, and there is no land record to be bounded by.

The bound is **read from the data**. It used to be the constant 2000, which
happened to be right for the Melo NGHGI dataset and would be wrong for any
other -- and it was applied to uncorrected runs too, which reported a year
problem against a record they had never loaded.
"""

from __future__ import annotations

import pytest

from fair_shares.library.exceptions import AllocationError
from fair_shares.library.validation.config import validate_allocation_year_for_co2

# What the Melo NGHGI dataset actually covers. Every test in the first class
# describes a run that loaded it.
NGHGI_YEARS = (2000, 2023)


class TestValidateAllocationYearForCo2:
    """Test NGHGI minimum year enforcement for co2 emission category."""

    # --- allocation_year (budget approaches) ---

    def test_budget_allocation_year_below_2000_raises(self):
        """allocation_year < 2000 with co2 should raise AllocationError."""
        config = {
            "equal-per-capita-budget": [{"allocation_year": 1850}],
        }
        with pytest.raises(
            AllocationError, match="allocation_year = 1850 is before 2000"
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_budget_allocation_year_at_2000_passes(self):
        """allocation_year = 2000 with co2 should pass."""
        config = {
            "equal-per-capita-budget": [{"allocation_year": 2000}],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_budget_allocation_year_above_1990_passes(self):
        """allocation_year > 2000 with co2 should pass."""
        config = {
            "equal-per-capita-budget": [{"allocation_year": 2020}],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- first_allocation_year (pathway approaches) ---

    def test_pathway_first_allocation_year_below_2000_raises(self):
        """first_allocation_year < 2000 with co2 should raise AllocationError."""
        config = {
            "equal-per-capita": [{"first_allocation_year": 1980}],
        }
        with pytest.raises(
            AllocationError, match="first_allocation_year = 1980 is before 2000"
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_pathway_first_allocation_year_at_2000_passes(self):
        """first_allocation_year = 2000 with co2 should pass."""
        config = {
            "equal-per-capita": [{"first_allocation_year": 2000}],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- pre_allocation_responsibility_year ---

    def test_pre_allocation_responsibility_year_below_2000_raises(self):
        """pre_allocation_responsibility_year < 2000 with co2 should raise AllocationError."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        with pytest.raises(
            AllocationError,
            match="pre_allocation_responsibility_year = 1850 is before 2000",
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_pre_allocation_responsibility_year_at_2000_passes(self):
        """pre_allocation_responsibility_year = 2000 with co2 should pass."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 2000,
                }
            ],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_pre_allocation_responsibility_year_above_1990_passes(self):
        """pre_allocation_responsibility_year > 2000 with co2 should pass."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 2000,
                }
            ],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_pre_allocation_responsibility_year_absent_passes(self):
        """Missing pre_allocation_responsibility_year should pass (default is 2000)."""
        config = {
            "per-capita-adjusted": [{"first_allocation_year": 2020}],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- kebab-case parameter names ---

    def test_pre_allocation_responsibility_year_kebab_case_below_2000_raises(self):
        """Kebab-case pre-allocation-responsibility-year < 2000 with co2 should raise."""
        config = {
            "per-capita-adjusted": [
                {
                    "first-allocation-year": 2020,
                    "pre-allocation-responsibility-year": 1850,
                }
            ],
        }
        with pytest.raises(
            AllocationError,
            match="pre_allocation_responsibility_year = 1850 is before 2000",
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_pre_allocation_responsibility_year_kebab_case_at_2000_passes(self):
        """Kebab-case pre-allocation-responsibility-year = 2000 with co2 should pass."""
        config = {
            "per-capita-adjusted": [
                {
                    "first-allocation-year": 2020,
                    "pre-allocation-responsibility-year": 2000,
                }
            ],
        }
        validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- co2-ffi (no NGHGI constraint) ---

    def test_co2_ffi_allows_pre_allocation_responsibility_year_before_1990(self):
        """co2-ffi has no NGHGI constraint — early years should pass."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        validate_allocation_year_for_co2(config, "co2-ffi", NGHGI_YEARS)

    def test_co2_ffi_allows_allocation_year_before_1990(self):
        """co2-ffi has no NGHGI constraint — early allocation_year should pass."""
        config = {
            "equal-per-capita-budget": [{"allocation_year": 1850}],
        }
        validate_allocation_year_for_co2(config, "co2-ffi", NGHGI_YEARS)

    # --- all-ghg-ex-co2-lulucf (no NGHGI constraint) ---

    def test_all_ghg_ex_co2_lulucf_allows_early_years(self):
        """all-ghg-ex-co2-lulucf has no NGHGI constraint."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        validate_allocation_year_for_co2(config, "all-ghg-ex-co2-lulucf", NGHGI_YEARS)

    # --- budget approaches with pre_allocation_responsibility_year ---

    def test_budget_pre_allocation_responsibility_year_below_2000_raises(self):
        """Budget approach with pre_allocation_responsibility_year < 2000 and co2 should raise."""
        config = {
            "per-capita-adjusted-budget": [
                {
                    "allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        with pytest.raises(
            AllocationError,
            match="pre_allocation_responsibility_year = 1850 is before 2000",
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- multiple param sets ---

    def test_second_param_set_with_early_historical_year_raises(self):
        """Second param set with pre_allocation_responsibility_year < 2000 should raise."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 2000,
                },
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                },
            ],
        }
        with pytest.raises(
            AllocationError,
            match="pre_allocation_responsibility_year = 1850 is before 2000",
        ):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    # --- error message content ---

    def test_error_message_mentions_nghgi(self):
        """Error message should explain the NGHGI rationale."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        with pytest.raises(AllocationError, match="NGHGI-consistent"):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)

    def test_error_message_suggests_co2_ffi_alternative(self):
        """Error message should suggest co2-ffi as an alternative."""
        config = {
            "per-capita-adjusted": [
                {
                    "first_allocation_year": 2020,
                    "pre_allocation_responsibility_year": 1850,
                }
            ],
        }
        with pytest.raises(AllocationError, match="co2-ffi"):
            validate_allocation_year_for_co2(config, "co2", NGHGI_YEARS)


class TestTheTwoPaths:
    """Corrected runs are bounded by their land record; uncorrected ones are not."""

    def test_an_uncorrected_run_is_not_bounded(self):
        # No LULUCF source means no NGHGI correction was applied, so there is
        # no land record for a year to be "before". This used to raise against
        # a hardcoded 2000, pointing the caller at a parameter that was not the
        # problem.
        config = {"equal-per-capita": [{"first_allocation_year": 1850}]}
        validate_allocation_year_for_co2(config, "all-ghg", None)
        validate_allocation_year_for_co2(config, "co2", None)

    def test_a_corrected_run_is_bounded_by_its_record(self):
        config = {"equal-per-capita": [{"first_allocation_year": 1990}]}
        with pytest.raises(AllocationError, match="is before 2000"):
            validate_allocation_year_for_co2(config, "all-ghg", (2000, 2023))

    def test_the_bound_is_the_records_own_first_year(self):
        # Same config, same category, two land sources: refused against one
        # that starts in 2000, allowed against one that starts in 1980.
        config = {"equal-per-capita": [{"first_allocation_year": 1990}]}
        with pytest.raises(AllocationError, match="is before 2000"):
            validate_allocation_year_for_co2(config, "all-ghg", (2000, 2023))
        validate_allocation_year_for_co2(config, "all-ghg", (1980, 2023))

    def test_the_message_quotes_the_range_it_has(self):
        config = {"equal-per-capita": [{"first_allocation_year": 1990}]}
        with pytest.raises(AllocationError, match=r"covers 2000-2023"):
            validate_allocation_year_for_co2(config, "all-ghg", (2000, 2023))

    def test_a_category_without_lulucf_is_never_bounded(self):
        config = {"equal-per-capita": [{"first_allocation_year": 1850}]}
        validate_allocation_year_for_co2(config, "co2-ffi", (2000, 2023))
        validate_allocation_year_for_co2(config, "all-ghg-ex-co2-lulucf", (2000, 2023))
