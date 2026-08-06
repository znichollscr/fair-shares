"""Choosing a budget, and keeping shares separate from budgets.

An allocation needs one number: the remaining budget for one climate target.
`select_budget` looks that up by key. The alternative -- handing a whole table
of budgets to something that iterates whatever it is given -- is what used to
make a filtering step necessary, and a filter that exists to suppress an
unwanted loop is a workaround for a signature that should not have taken a
table.

The other half of these tests pins the separation `allocate` depends on: a
method's shares do not depend on the budget, so they are computed once per
method and reused across every climate target that method is asked for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fair_shares.api.allocation import BUDGET_COLUMN, ClimateTarget, select_budget
from fair_shares.api.sources import SourceSelection
from fair_shares.library.exceptions import ConfigurationError

SOURCES = SourceSelection(
    emissions="primap-202503",
    gdp="wdi-2025",
    population="un-owid-2025",
    gini="wdi-2025",
)

# The eight rows a co2-ffi budget table actually carries. A caller wants one.
BUDGET_ROWS = [
    ("lamboll_2023", "1.5C", 0.50),
    ("lamboll_2023", "2C", 0.66),
    ("forster_2024", "1.5C", 0.50),
    ("forster_2024", "2C", 0.66),
    ("forster_2024", "2C", 0.83),
    ("ar6_2020", "1.5C", 0.50),
    ("ar6_2020", "2C", 0.66),
    ("ar6_2020", "2C", 0.83),
]


@pytest.fixture
def rcbs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source,
                "climate-assessment": limit,
                "quantile": quantile,
                "emission-category": "co2-ffi",
                BUDGET_COLUMN: 100.0 * index,
            }
            for index, (source, limit, quantile) in enumerate(BUDGET_ROWS, start=1)
        ]
    )


def target(limit="1.5C", quantile=0.5, reference="ar6_2020") -> ClimateTarget:
    return ClimateTarget(limit=limit, quantile=quantile, reference=reference)


class TestSelectBudget:
    def test_returns_one_number(self, rcbs):
        budget = select_budget(rcbs, target())
        assert budget.value_mt == 600.0
        assert budget.climate_target == target()

    def test_distinguishes_references_sharing_a_limit(self, rcbs):
        # 1.5C p50 exists under three assessments and they are three different
        # budgets. Anything that collapsed them would be indistinguishable in
        # the output.
        values = {
            reference: select_budget(rcbs, target(reference=reference)).value_mt
            for reference in ("lamboll_2023", "forster_2024", "ar6_2020")
        }
        assert len(set(values.values())) == 3

    def test_distinguishes_quantiles(self, rcbs):
        assert select_budget(rcbs, target("2C", 0.83)).value_mt == 800.0

    def test_a_missing_target_says_what_was_available(self, rcbs):
        with pytest.raises(ConfigurationError) as excinfo:
            select_budget(rcbs, target(reference="nonesuch"))
        message = str(excinfo.value)
        assert "nonesuch" in message
        assert "1.5C p50 ar6_2020" in message

    def test_an_ambiguous_target_is_refused(self, rcbs):
        # Two rows for one target means there is no single budget to allocate.
        # Returning either, or both, would be worse than saying so.
        duplicated = pd.concat([rcbs, rcbs.iloc[[5]]], ignore_index=True)
        with pytest.raises(ConfigurationError, match="no single budget"):
            select_budget(duplicated, target())

    def test_does_not_mutate_the_table(self, rcbs):
        select_budget(rcbs, target())
        assert len(rcbs) == len(BUDGET_ROWS)

    def test_a_numeric_quantile_is_required_to_match(self, rcbs):
        # `parse_rcb_scenario` returns the quantile as text, and every consumer
        # compares it as a float. They agreed only because the table reached
        # consumers through a CSV, which converted it on the way; built in
        # memory, the strings survive and every match silently fails.
        assert rcbs["quantile"].dtype.kind == "f"


class TestSharesAreIndependentOfTheBudget:
    """The separation `allocate` relies on to compute shares once per method."""

    def test_one_method_at_several_targets_computes_shares_once(
        self, rcbs, monkeypatch
    ):
        # The point of splitting shares from budgets: a grid of 14 methods x 5
        # climate targets should do 14 share computations, not 70.
        import fair_shares.api.allocation as module

        computed: list[str] = []
        monkeypatch.setattr(
            module,
            "compute_shares",
            lambda data, approach, parameters, **kwargs: computed.append(approach),
        )
        monkeypatch.setattr(
            module,
            "apply_budget",
            lambda shares, budget, data, **kwargs: pd.DataFrame({"iso3c": ["AUT"]}),
        )

        data = SimpleNamespace(
            budget_category="co2-ffi",
            category="co2-ffi",
            rcbs=rcbs,
            parts={},
            world_emissions={},
        )
        requests = [
            module.AllocationRequest(
                approach="equal-per-capita-budget",
                parameters={"allocation_year": 1990},
                climate_target=target(limit, quantile),
            )
            for limit, quantile in (("1.5C", 0.5), ("2C", 0.66), ("2C", 0.83))
        ]

        results = module.allocate(data, requests, sources=SOURCES)

        assert len(results) == 3
        assert computed == ["equal-per-capita-budget"], (
            "shares should be computed once per method, not once per target"
        )

    def test_differing_parameters_are_not_shared(self, rcbs, monkeypatch):
        # Same approach, different inputs, is a different method.
        import fair_shares.api.allocation as module

        computed: list[dict] = []
        monkeypatch.setattr(
            module,
            "compute_shares",
            lambda data, approach, parameters, **kwargs: computed.append(parameters),
        )
        monkeypatch.setattr(
            module,
            "apply_budget",
            lambda shares, budget, data, **kwargs: pd.DataFrame({"iso3c": ["AUT"]}),
        )

        data = SimpleNamespace(
            budget_category="co2-ffi",
            category="co2-ffi",
            rcbs=rcbs,
            parts={},
            world_emissions={},
        )
        requests = [
            module.AllocationRequest(
                approach="equal-per-capita-budget",
                parameters={"allocation_year": year},
                climate_target=target(),
            )
            for year in (1990, 2015)
        ]

        module.allocate(data, requests, sources=SOURCES)
        assert len(computed) == 2

    def test_attaching_a_budget_does_not_recompute_shares(self):
        # Stated against the call graph rather than a timing: shares come from
        # `run_parameter_grid`, and attaching a budget must not reach for it.
        import inspect

        from fair_shares.api.allocation import apply_budget

        body = inspect.getsource(apply_budget)
        assert "run_parameter_grid" not in body
        assert "calculate_budget_from_rcb" in body


class TestABudgetIsRequired:
    def test_preprocessing_without_one_says_so(self, rcbs):
        import fair_shares.api.allocation as module

        data = SimpleNamespace(
            budget_category=None,
            category="co2-ffi",
            rcbs=pd.DataFrame(),
            parts={},
            world_emissions={},
        )
        with pytest.raises(ConfigurationError, match="nothing to allocate"):
            module.allocate(data, [], sources=SOURCES)

    def test_a_request_without_a_climate_target_says_so(self, rcbs):
        import fair_shares.api.allocation as module

        data = SimpleNamespace(
            budget_category="co2-ffi",
            category="co2-ffi",
            rcbs=rcbs,
            parts={},
            world_emissions={},
        )
        request = module.AllocationRequest(approach="equal-per-capita-budget")
        with pytest.raises(ConfigurationError, match="needs a climate_target"):
            module.allocate(data, [request], sources=SOURCES)
