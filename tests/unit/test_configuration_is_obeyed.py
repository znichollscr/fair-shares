"""Every configurable input must actually change the run.

The failure this guards against is not a crash: it is a knob that looks
settable, is accepted without complaint, and is then ignored — so a caller
believes they changed the science when they changed nothing. Two such knobs
existed before this file (the bunkers source, hardcoded in notebook 108; the
data and output directories, ignored by the Snakemake build), and neither
announced itself. Both were found by inspection rather than by any test
failing.

So the rule this file encodes is: for each input a caller can set, either

- it changes the **source id**, meaning it changes the processed data and so
  earns its own output tree; or
- it changes the **resolved config**, meaning downstream code reads it; and
- an unrecognised value is a loud error rather than a silent fallback.

An input that satisfies none of those is either dead or a bug.
"""

from __future__ import annotations

import pytest

from fair_shares.library.exceptions import ConfigurationError
from fair_shares.library.utils.data.config import build_data_config

BASE_SOURCES = {
    "emissions": "primap-202503",
    "gdp": "wdi-2025",
    "population": "un-owid-2025",
    "gini": "wdi-2025",
    "bunkers": "gcb-2024",
    "target": "rcbs",
}

# (source key, resolved-config attribute, in the source id?). Everything a
# caller can put in `active_sources` appears here — if you add a source, add a
# row, and the test below will tell you whether it is actually wired up.
SOURCE_KNOBS = [
    ("emissions", "active_emissions_source", True),
    ("gdp", "active_gdp_source", True),
    ("population", "active_population_source", True),
    ("gini", "active_gini_source", True),
    ("bunkers", "active_bunkers_source", True),
    ("target", "active_target_source", True),
]


@pytest.fixture
def baseline():
    return build_data_config("co2-ffi", dict(BASE_SOURCES))


class TestEverySourceReachesTheResolvedConfig:
    """A named source must come back as the active one."""

    @pytest.mark.parametrize(("key", "attribute", "_in_id"), SOURCE_KNOBS)
    def test_named_source_is_what_resolves(self, key, attribute, _in_id):
        config, _ = build_data_config("co2-ffi", dict(BASE_SOURCES))
        assert getattr(config, attribute) == BASE_SOURCES[key], (
            f"active_sources[{key!r}] is accepted but does not resolve to "
            f"{attribute} — callers can set it and it does nothing"
        )

    @pytest.mark.parametrize(("key", "_attribute", "in_id"), SOURCE_KNOBS)
    def test_source_appears_in_the_source_id(self, key, _attribute, in_id, baseline):
        _config, source_id = baseline
        if not in_id:
            return
        assert BASE_SOURCES[key] in source_id, (
            f"{key!r} changes the processed data but is not in the source id, "
            "so two different runs would share one output tree"
        )


class TestUnknownValuesAreRejected:
    """A typo must be an error, never a silent fallback to a default."""

    @pytest.mark.parametrize("key", [k for k, _, _ in SOURCE_KNOBS])
    def test_unrecognised_source_raises(self, key):
        with pytest.raises(ConfigurationError):
            build_data_config(
                "co2-ffi", {**BASE_SOURCES, key: "definitely-not-a-real-source"}
            )


class TestHarmonisationYear:
    """Obeyed, but deliberately absent from the source id."""

    def test_is_reflected_in_the_resolved_config(self):
        config, _ = build_data_config(
            "co2-ffi", dict(BASE_SOURCES), harmonisation_year=2019
        )
        assert config.harmonisation_year == 2019

    def test_is_not_in_the_source_id(self):
        """Harmonisation anchors modelled scenarios to observations; it cannot
        change the observations. Two runs differing only in harmonisation read
        byte-identical processed files, so they must share an output tree —
        putting the year in the id would fragment the cache for nothing."""
        ids = {
            build_data_config(
                "co2-ffi", dict(BASE_SOURCES), harmonisation_year=year
            )[1]
            for year in (2019, 2020, 2023)
        }
        assert len(ids) == 1


class TestEmissionCategory:
    """Categories never share an output tree."""

    def test_changes_the_source_id(self):
        ids = {
            build_data_config(category, dict(BASE_SOURCES))[1]
            for category in ("co2-ffi", "all-ghg-ex-co2-lulucf")
        }
        assert len(ids) == 2


class TestBunkersOnlyWhereItApplies:
    """The one knob whose presence in the id is conditional."""

    def test_absent_for_pathway_targets(self):
        """Pathway targets never deduct bunkers, so naming a source must not
        split their output tree over an input they ignore."""
        _config, source_id = build_data_config(
            "co2-ffi", {**BASE_SOURCES, "target": "pathway"}
        )
        assert "gcb-2024" not in source_id

    def test_present_for_budget_targets(self, baseline):
        _config, source_id = baseline
        assert "gcb-2024" in source_id
