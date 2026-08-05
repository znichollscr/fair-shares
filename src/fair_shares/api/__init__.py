"""A Python API for fair-share allocations: functions in, objects out.

This package exists because the notebook pipeline could not be *called*. It
could only be *driven* -- by writing parameters into a Snakemake config, which
wrote a YAML file, which papermill read back into notebook globals, which wrote
CSVs into a directory named after a hash of the parameters, which the next
notebook read again. Every one of those hops is a place for the run to disagree
with itself, and several of them did.

What this package does differently:

* **Functions, not notebooks.** Everything is an ordinary Python call with
  arguments and a return value. There is no papermill, no Snakemake, and no
  parameter cell.
* **Frames, not files.** Preprocessing hands data step to step in memory. The
  intermediate CSVs were never anything but a way for one notebook to talk to
  the next, and writing them was most of the runtime.
* **Explicit inputs.** :class:`~fair_shares.api.sources.Inputs` says which
  vintage of each source to use and where it lives. Nothing is resolved from a
  working directory, an environment variable, or a default that changes what
  the numbers mean.
* **Results that know what they are.** Every allocation comes back tagged with
  the request that produced it, so a caller running a grid can tell its results
  apart without reconstructing the mapping from a directory name.
* **Serialisation is the caller's business.** Nothing here writes to disk.

The entry points:

>>> from fair_shares.api import Inputs, SourceSelection, preprocess, allocate

See :mod:`fair_shares.api.sources` for how to describe inputs,
:mod:`fair_shares.api.preprocess` for what preprocessing produces, and
:mod:`fair_shares.api.allocate` for running allocations against it.
"""

from __future__ import annotations

from fair_shares.api.allocate import (
    AllocationRequest,
    AllocationResult,
    ClimateTarget,
    allocate,
)
from fair_shares.api.preprocess import ProcessedData, preprocess
from fair_shares.api.sources import Inputs, SourceSelection

__all__ = [
    "AllocationRequest",
    "AllocationResult",
    "ClimateTarget",
    "Inputs",
    "ProcessedData",
    "SourceSelection",
    "allocate",
    "preprocess",
]
