"""Bunker fuel CO2 is a selectable source, not a path baked into a notebook.

Budget targets subtract bunker CO2 from the global remaining carbon budget
before any country is allocated a share, so this choice moves *every* allocated
total. It used to be a literal path in notebook 108, which meant swapping GCB
vintages silently changed every number with nothing in the source id to record
it — the v1.0 -> v1.01 revision moves the bunker column by about 1.8%.
"""

from __future__ import annotations

import pytest

from fair_shares.library.exceptions import ConfigurationError
from fair_shares.library.utils.data.config import build_data_config, build_source_id

BASE_SOURCES = {
    "emissions": "primap-202503",
    "gdp": "wdi-2025",
    "population": "un-owid-2025",
    "gini": "wdi-2025",
}


class TestSourceId:
    """The id must record the bunkers choice wherever it can change numbers."""

    def test_budget_target_includes_bunkers(self):
        source_id = build_source_id(
            **BASE_SOURCES,
            bunkers="gcb-2024",
            target="rcbs",
            emission_category="co2-ffi",
        )
        assert "gcb-2024" in source_id

    def test_pathway_target_omits_bunkers(self):
        """A pathway run never deducts bunkers, so naming one must not split
        its output directory over an input it ignores."""
        source_id = build_source_id(
            **BASE_SOURCES,
            bunkers="gcb-2024",
            target="pathway",
            emission_category="co2-ffi",
        )
        assert "gcb-2024" not in source_id

    def test_changing_the_source_changes_the_id(self):
        ids = {
            build_source_id(
                **BASE_SOURCES,
                bunkers=bunkers,
                target="rcbs",
                emission_category="co2-ffi",
            )
            for bunkers in ("gcb-2024", "gcb-2099")
        }
        assert len(ids) == 2, "the bunkers source must be part of the identity"

    def test_absent_bunkers_is_backwards_compatible(self):
        """Callers that name no bunkers source get the id they always got."""
        source_id = build_source_id(
            **BASE_SOURCES, target="rcbs", emission_category="co2-ffi"
        )
        assert source_id == (
            "primap-202503_wdi-2025_un-owid-2025_wdi-2025_rcbs_co2-ffi"
        )


class TestConfigResolution:
    """What the config resolves to is what the run will actually use."""

    def test_named_source_is_obeyed(self):
        config, source_id = build_data_config(
            "co2-ffi", {**BASE_SOURCES, "bunkers": "gcb-2024", "target": "rcbs"}
        )
        assert config.active_bunkers_source == "gcb-2024"
        assert "gcb-2024" in source_id

    def test_unnamed_source_defaults_and_is_recorded(self):
        """Defaulting is fine; defaulting *silently* is not. The id has to say
        which source the run used even when the caller did not choose one."""
        config, source_id = build_data_config(
            "co2-ffi", {**BASE_SOURCES, "target": "rcbs"}
        )
        assert config.active_bunkers_source == "gcb-2024"
        assert "gcb-2024" in source_id

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ConfigurationError, match="Bunkers source"):
            build_data_config(
                "co2-ffi",
                {**BASE_SOURCES, "bunkers": "does-not-exist", "target": "rcbs"},
            )

    def test_config_carries_the_read_parameters(self):
        """Notebook 108 reads sheet/header/column from here rather than
        hardcoding them, so a differently-shaped workbook is a config change."""
        config, _ = build_data_config(
            "co2-ffi", {**BASE_SOURCES, "bunkers": "gcb-2024", "target": "rcbs"}
        )
        params = config.bunkers["gcb-2024"].data_parameters
        assert params.sheet_name == "Territorial Emissions"
        assert params.header_row == 11
        assert params.bunker_column == "Bunkers"
