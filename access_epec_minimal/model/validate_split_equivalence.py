"""Phase-1 validation gate for the owner-split renewable input.

With every split unit submitting the same offer as its parent generator, the
exact competitive market cleared on ``market_data_strategic_generation.json``
must reproduce the aggregate benchmark on ``market_data.json``:

- equal total renewable dispatch by node and hour,
- equal curtailment by node and hour,
- equal LMPs within solver tolerance,
- equal system cost,
- equal combined I3+I4 generation rent.

Runs the check twice: truthful zero-cost renewables and the -25 EUR/MWh PV
true-cost experiment configuration.

Default run:
    python validate_split_equivalence.py
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import (
    MarketData,
    build_primal_market_clearing_model,
    load_market_data,
)


MODEL_DIR = Path(__file__).resolve().parent
BASELINE = MODEL_DIR / "input" / "market_data.json"
SPLIT = MODEL_DIR / "input" / "market_data_strategic_generation.json"

PARENT_TO_CHILDREN = {
    "RES_Wind_N1": ("RES_Wind_I3_N1", "RES_Wind_I4_N1"),
    "RES_PV_N6": ("RES_PV_I3_N6", "RES_PV_I4_N6"),
    "RES_PV_N8": ("RES_PV_I3_N8", "RES_PV_I4_N8"),
}

# The QP pins lambda = penalty * DemandAdjustment, so LMP noise is primal
# noise amplified by the 500 EUR/MW^2 penalty.  Across Ipopt tol 1e-9..1e-12
# the observed cross-parametrization dual noise ranges 4e-5..2e-3 EUR/MWh
# while dispatch matches to <1e-8 MW and system cost to <1e-6 EUR, so the
# dual gate sits at the demonstrated noise ceiling; it is ~4e-5 relative to
# the 50-80 EUR/MWh price level and economically meaningless.
DISPATCH_TOL_MW = 1.0e-4
LMP_TOL_EUR_PER_MWH = 2.0e-3
COST_TOL_EUR = 1.0e-2
RENT_TOL_EUR = 0.5


def _ipopt() -> pyo.SolverFactory:
    candidates = []
    if os.environ.get("IPOPT_EXECUTABLE"):
        candidates.append(Path(os.environ["IPOPT_EXECUTABLE"]))
    if shutil.which("ipopt"):
        candidates.append(Path(shutil.which("ipopt") or ""))
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "idaes" / "bin" / "ipopt.exe")
    executable = next((c for c in candidates if c.is_file()), None)
    kwargs = {"solver_io": "nl"}
    if executable is not None:
        kwargs["executable"] = str(executable)
    solver = pyo.SolverFactory("ipopt", **kwargs)
    solver.options.update(
        {
            "linear_solver": "ma57",
            "tol": 1.0e-12,
            "acceptable_tol": 1.0e-11,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "print_level": 0,
        }
    )
    return solver


def _solve(data: MarketData) -> pyo.ConcreteModel:
    model = build_primal_market_clearing_model(data, include_load_shed=True)
    result = _ipopt().solve(model, tee=False)
    termination = result.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Exact market clearing failed: {termination}")
    return model


def _renewables(data: MarketData) -> list[str]:
    return [g for g in data.generators if "RES_" in g]


def _generator_node(data: MarketData, generator: str) -> str:
    for node, units in data.generators_at_node.items():
        if generator in units:
            return node
    raise KeyError(generator)


def _node_hour_dispatch(model: pyo.ConcreteModel, data: MarketData) -> dict:
    dispatch: dict[tuple[str, int], float] = {}
    for g in _renewables(data):
        node = _generator_node(data, g)
        for t in data.times:
            dispatch[node, int(t)] = dispatch.get((node, int(t)), 0.0) + float(
                pyo.value(model.P_gen[g, t])
            )
    return dispatch


def _node_hour_availability(data: MarketData) -> dict:
    availability: dict[tuple[str, int], float] = {}
    for g in _renewables(data):
        node = _generator_node(data, g)
        for t in data.times:
            availability[node, int(t)] = availability.get((node, int(t)), 0.0) + float(
                data.generation_capacity[g, int(t)]
            )
    return availability


def _renewable_rent(model: pyo.ConcreteModel, data: MarketData) -> float:
    total = 0.0
    for g in _renewables(data):
        node = _generator_node(data, g)
        cost = data.generation_cost[g]
        for t in data.times:
            lmp = float(model.dual[model.nodal_balance[node, t]])
            total += (lmp - cost) * float(pyo.value(model.P_gen[g, t]))
    return total


def compare(label: str, base: MarketData, split: MarketData) -> list[str]:
    base_model = _solve(base)
    split_model = _solve(split)

    failures: list[str] = []

    base_dispatch = _node_hour_dispatch(base_model, base)
    split_dispatch = _node_hour_dispatch(split_model, split)
    dispatch_err = max(
        abs(split_dispatch[key] - base_dispatch[key]) for key in base_dispatch
    )

    base_avail = _node_hour_availability(base)
    split_avail = _node_hour_availability(split)
    avail_err = max(abs(split_avail[key] - base_avail[key]) for key in base_avail)
    curtail_err = max(
        abs(
            (split_avail[key] - split_dispatch[key])
            - (base_avail[key] - base_dispatch[key])
        )
        for key in base_dispatch
    )

    lmp_err = max(
        abs(
            float(split_model.dual[split_model.nodal_balance[n, t]])
            - float(base_model.dual[base_model.nodal_balance[n, t]])
        )
        for n in base.nodes
        for t in base.times
    )
    cost_err = abs(
        float(pyo.value(split_model.objective)) - float(pyo.value(base_model.objective))
    )
    rent_base = _renewable_rent(base_model, base)
    rent_split = _renewable_rent(split_model, split)
    rent_err = abs(rent_split - rent_base)

    def check(name: str, value: float, tol: float) -> None:
        status = "OK  " if value <= tol else "FAIL"
        print(f"  {status} {name}: {value:.3e} (tol {tol:.0e})")
        if value > tol:
            failures.append(f"{label}: {name} = {value:.3e} > {tol:.0e}")

    print(f"[{label}]")
    print(f"  system cost base:  {float(pyo.value(base_model.objective)):,.4f}")
    print(f"  system cost split: {float(pyo.value(split_model.objective)):,.4f}")
    print(f"  I3+I4 rent base:   {rent_base:,.4f}")
    print(f"  I3+I4 rent split:  {rent_split:,.4f}")
    check("availability_split_error_mw", avail_err, DISPATCH_TOL_MW)
    check("node_hour_dispatch_error_mw", dispatch_err, DISPATCH_TOL_MW)
    check("node_hour_curtailment_error_mw", curtail_err, DISPATCH_TOL_MW)
    check("lmp_error_eur_per_mwh", lmp_err, LMP_TOL_EUR_PER_MWH)
    check("system_cost_error_eur", cost_err, COST_TOL_EUR)
    check("combined_rent_error_eur", rent_err, RENT_TOL_EUR)
    return failures


def with_costs(data: MarketData, overrides: dict[str, float]) -> MarketData:
    costs = dict(data.generation_cost)
    costs.update(overrides)
    return replace(data, generation_cost=costs)


def main() -> int:
    base = load_market_data(BASELINE)
    split = load_market_data(SPLIT)

    failures = compare("truthful zero-cost renewables", base, split)

    base_pv = with_costs(base, {"RES_PV_N6": -25.0, "RES_PV_N8": -25.0})
    split_pv = with_costs(
        split,
        {
            "RES_PV_I3_N6": -25.0,
            "RES_PV_I4_N6": -25.0,
            "RES_PV_I3_N8": -25.0,
            "RES_PV_I4_N8": -25.0,
        },
    )
    failures += compare("pv true cost -25 EUR/MWh", base_pv, split_pv)

    if failures:
        print("\nGATE FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nGATE PASSED: split input reproduces the aggregate benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
