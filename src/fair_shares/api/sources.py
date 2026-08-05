"""Which input vintages a run uses, and where they live.

Two objects, and the split between them is the point:

:class:`SourceSelection` is *scientific* -- which vintage of each dataset the
numbers came from. It belongs in a caller's config, and it is the thing worth
hashing into an identifier.

:class:`Inputs` adds *locations* -- which directory, or which exact file. Moving
a file must never change a result, so locations are deliberately kept out of the
selection rather than mixed in with it.

Nothing here reads an environment variable or a working directory. The old
pipeline resolved paths from ``pyprojroot.here()``, which meant the answer
depended on where the process happened to be started; a caller that wants a
different tree now says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fair_shares.library.exceptions import ConfigurationError
from fair_shares.library.paths import packaged_config

logger = logging.getLogger(__name__)

# The source kinds a run can select. Ordered as the pipeline consumes them.
SOURCE_KINDS = (
    "emissions",
    "gdp",
    "population",
    "gini",
    "lulucf",
    "bunkers",
    "scenarios",
)


@dataclass(frozen=True)
class SourceSelection:
    """Which vintage of each input dataset a run uses.

    Every field names a key in the data-source catalogue, e.g.
    ``emissions="primap-202503"``.

    Parameters
    ----------
    emissions, gdp, population, gini
        Required: every allocation reads all four.
    lulucf
        Required only for categories whose content depends on it (``co2``,
        ``co2-lulucf``, ``all-ghg``).
    bunkers
        Required only for budget targets, which deduct international bunker
        CO2 from the global remaining carbon budget before allocating it.
    scenarios
        Required only when scenario pathways are consulted.
    """

    emissions: str
    gdp: str
    population: str
    gini: str
    lulucf: str | None = None
    bunkers: str | None = None
    scenarios: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return the selection as a plain mapping.

        Returns
        -------
        dict
            Keyed by source kind, including the unset ones as ``None`` -- a
            caller building an identifier wants to see what was *not* chosen
            as much as what was.
        """
        return {kind: getattr(self, kind) for kind in SOURCE_KINDS}


def _load_catalogue() -> dict[str, Any]:
    """Read the packaged data-source catalogue."""
    import yaml

    with packaged_config("data_sources/data_sources_unified.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class Inputs:
    """A resolved description of where a run reads its data from.

    Parameters
    ----------
    sources
        Which vintage of each dataset to use.
    data_dir
        The directory the catalogue's relative paths resolve against. Required
        and explicit: a default here would be a silent choice about which data
        a result came from.
    catalogue
        The data-source catalogue. Defaults to the packaged one; pass your own
        to add a source without editing the package.
    overrides
        Exact paths for individual files, keyed by ``"<kind>"`` or
        ``"<kind>.<path-field>"`` for sources with more than one file (e.g.
        ``"population.path_projected"``). Takes precedence over `data_dir`.

    Raises
    ------
    ConfigurationError
        If a selected source is not in the catalogue, naming what is.
    """

    sources: SourceSelection
    data_dir: Path
    catalogue: dict[str, Any] = field(default_factory=_load_catalogue)
    overrides: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise `data_dir` and reject a source the catalogue lacks."""
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        for kind, name in self.sources.as_dict().items():
            if name is None:
                continue
            available = self.catalogue.get(kind, {})
            if name not in available:
                raise ConfigurationError(
                    f"unknown {kind} source {name!r}. "
                    f"The catalogue offers: {sorted(available)}"
                )

    def parameters(self, kind: str) -> dict[str, Any]:
        """Return one source's ``data_parameters`` block.

        Parameters
        ----------
        kind
            A source kind, e.g. ``"gdp"``.

        Returns
        -------
        dict
            The parameters, or an empty mapping when the source declares none.

        Raises
        ------
        ConfigurationError
            If nothing is selected for this kind.
        """
        return self._entry(kind).get("data_parameters", {})

    def path(self, kind: str, field_name: str = "path") -> Path:
        """Resolve one input file to an absolute path.

        Parameters
        ----------
        kind
            A source kind, e.g. ``"gdp"``.
        field_name
            Which path the source declares; sources with several name them
            (``path_ppp``, ``path_historical``, ...).

        Returns
        -------
        pathlib.Path
            The resolved path. Existence is not checked here -- the loader
            that opens it gives a better error than a generic one could.

        Raises
        ------
        ConfigurationError
            If nothing is selected for this kind, or the source declares no
            such path.
        """
        for key in (f"{kind}.{field_name}", kind):
            override = self.overrides.get(key)
            if override is not None:
                return Path(override)

        entry = self._entry(kind)
        relative = entry.get(field_name)
        if relative is None:
            declared = sorted(k for k in entry if str(k).startswith("path"))
            raise ConfigurationError(
                f"{kind} source {getattr(self.sources, kind)!r} declares no "
                f"{field_name!r}. It declares: {declared}"
            )
        return self._resolve(str(relative))

    def files(self) -> dict[str, Path]:
        """Return every input file this run would read.

        The point of this is provenance: a caller that caches results needs to
        know when the inputs behind them have changed, and with nothing written
        to an intermediate tree there is no built directory to inspect instead.
        Naming the raw files is also the more honest question -- an intermediate
        tree is a consequence of the inputs, not the thing a result depends on.

        Returns
        -------
        dict
            Keyed ``"<kind>"`` or ``"<kind>.<path-field>"`` for sources with
            several files, in a stable order. Only selected sources appear.
        """
        found: dict[str, Path] = {}
        for kind, name in self.sources.as_dict().items():
            if name is None:
                continue
            entry = self._entry(kind)
            fields = sorted(k for k in entry if str(k).startswith("path"))
            for field_name in fields:
                key = kind if fields == ["path"] else f"{kind}.{field_name}"
                found[key] = self.path(kind, field_name)

        # The region mapping is read by every run and belongs to none of the
        # selected sources, so it would otherwise go unrecorded.
        mapping = self.catalogue.get("general", {}).get("region_mapping", {})
        if mapping.get("path"):
            found["general.region_mapping"] = self._resolve(str(mapping["path"]))
        return dict(sorted(found.items()))

    def _entry(self, kind: str) -> dict[str, Any]:
        name = getattr(self.sources, kind, None)
        if name is None:
            raise ConfigurationError(
                f"this run selected no {kind} source, so it cannot be read. "
                f"Set SourceSelection.{kind} if the run needs one."
            )
        return self.catalogue[kind][name]

    def _resolve(self, relative: str) -> Path:
        """Resolve a catalogue path against `data_dir`.

        The catalogue stores repo-relative strings beginning ``data/``. That
        prefix is stripped, so `data_dir` names the data tree itself rather
        than something that happens to contain a directory called ``data``.
        """
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        parts = candidate.parts
        if parts and parts[0] == "data":
            parts = parts[1:]
        return self.data_dir.joinpath(*parts)
