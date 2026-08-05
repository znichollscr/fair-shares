"""Configuration loading for preprocessing notebooks."""

import logging
from pathlib import Path
from typing import Any

import yaml

from fair_shares.library.paths import output_dir as resolve_output_dir
from fair_shares.library.utils import build_source_id
from fair_shares.library.utils.data.config import build_data_config

logger = logging.getLogger(__name__)


def load_preprocessing_config(
    emission_category: str | None,
    active_target_source: str | None,
    active_emissions_source: str | None,
    active_gdp_source: str | None,
    active_population_source: str | None,
    active_gini_source: str | None,
    active_lulucf_source: str | None = None,
    active_bunkers_source: str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str, Path]:
    """Load preprocessing configuration from Papermill parameters or interactive defaults.

    Args:
        emission_category: Emission category (e.g., "co2-ffi", "all-ghg-ex-co2-lulucf")
        active_target_source: Target source (e.g., "rcbs", "pathway")
        active_emissions_source: Emissions source (e.g., "primap-202503")
        active_gdp_source: GDP source (e.g., "wdi-2025")
        active_population_source: Population source (e.g., "un-owid-2025")
        active_gini_source: Gini source (e.g., "wdi-2025")
        active_lulucf_source: LULUCF source (e.g., "melo-2026")
        active_bunkers_source: Bunker fuel CO2 source (e.g., "gcb-2024").
            Part of the source id for budget targets, which deduct it.
        output_dir: Directory holding pipeline products. Defaults to the
            resolved output directory (see :mod:`fair_shares.library.paths`).

    Returns
    -------
        Tuple of (config dict, source_id string, output directory Path)
    """
    resolved_output = resolve_output_dir(output_dir)

    if emission_category is not None:
        # Running via Papermill - load composed config
        logger.info("Running via Papermill")

        source_id = build_source_id(
            emissions=active_emissions_source,
            gdp=active_gdp_source,
            population=active_population_source,
            gini=active_gini_source,
            lulucf=active_lulucf_source,
            bunkers=active_bunkers_source,
            target=active_target_source,
            emission_category=emission_category,
        )

        config_path = resolved_output / source_id / "config.yaml"
        logger.info(f"Loading config from: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

    else:
        # Running interactively - build config programmatically
        logger.info("Running interactively - build desired config")

        # Default interactive configuration
        emission_category = "co2-ffi"
        active_sources = {
            "emissions": "primap-202503",
            "gdp": "wdi-2025",
            "population": "un-owid-2025",
            "gini": "wdi-2025",
            "target": "rcbs",
        }

        config, source_id = build_data_config(emission_category, active_sources)
        # Convert Pydantic model to dict for consistency with pipeline
        config = config.model_dump()

    return config, source_id, resolved_output
