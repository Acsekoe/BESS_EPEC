"""Standalone Pyomo implementation of the DUAL of the spot-market LLP.

This is the explicit linear-programming dual of the deterministic lower-level
market-clearing primal in ``primal_market_clearing_model.py`` (which mirrors
Overleaf/model_extension.tex). By LP strong duality the optimal dual objective
must equal the optimal primal objective, so solving both is a direct numerical
check that the two formulations describe the same problem.

The dual is derived term by term from the primal below. Using the standard
convention for a MIN primal:

    * equality rows            -> free dual
    * ``<=`` rows              -> dual <= 0
    * ``>=`` rows              -> dual >= 0
    * primal var x >= 0        -> dual stationarity is an inequality  (A^T y)_x <= c_x
    * primal var x free        -> dual stationarity is an equality    (A^T y)_x  = c_x

and the dual objective is ``max b^T y``.

Dual-variable naming follows the LaTeX notation table where practical:

    lam[n,t]        lambda_{n,t}     nodal balance            (free)
    lam_sys[t]      lambda_{sys,t}   system balance           (free)
    nu_gen[g,t]     nu^+_{g,t}       generator upper cap      (<= 0)
    mu_up[l,t]      mu^+_{l,t}       line flow upper           (<= 0)
    mu_dn[l,t]      mu^-_{l,t}       line flow lower           (>= 0)
    rho_ch[i,n,t]   rho^+_{i,n,t}    charge power cap          (<= 0)
    sig_dis[i,n,t]  sigma^+_{i,n,t}  discharge power cap       (<= 0)
    gam[i,n,t]      gamma_{i,n,t}    SOC transition            (free)
    del_soc[i,n,τ]  delta^+_{i,n,τ}  SOC energy cap            (<= 0)
    rho_per[i,n]    -                SOC periodicity           (free)
    xi_shed[n,t]    xi^+_{n,t}       load shed upper bound     (<= 0)

Default run:
    python dual_market_clearing_model.py           # solve the dual alone
    python dual_market_clearing_model.py --compare  # solve primal + dual and compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pyomo.environ as pyo

from primal_market_clearing_model import (
    DEFAULT_DATA_PATH,
    MarketData,
    build_primal_market_clearing_model,
    get_solver,
    load_market_data,
    solve_model,
    value,
)

MODEL_NAME = "Dual Spot Market Clearing Model"


def _nodes_of_generator(data: MarketData) -> Dict[str, list[str]]:
    """Map each generator to the node(s) whose nodal balance it enters."""
    gen_nodes: Dict[str, list[str]] = {g: [] for g in data.generators}
    for n in data.nodes:
        for g in data.generators_at_node.get(n, []):
            gen_nodes.setdefault(g, []).append(n)
    return gen_nodes


def build_dual_market_clearing_model(data: MarketData) -> pyo.ConcreteModel:
    """Build the LP dual of the deterministic primal market-clearing problem."""

    m = pyo.ConcreteModel(name=MODEL_NAME)

    m.N = pyo.Set(initialize=data.nodes, ordered=True)
    m.G = pyo.Set(initialize=data.generators, ordered=True)
    m.I = pyo.Set(initialize=data.storage_units, ordered=True)
    m.T = pyo.Set(initialize=data.times, ordered=True)
    m.T_SOC = pyo.Set(initialize=data.soc_times, ordered=True)
    m.L = pyo.Set(initialize=data.lines, ordered=True)

    gen_nodes = _nodes_of_generator(data)
    last_t = max(data.times)
    eta = data.eta

    # ------------------------------------------------------------------ #
    # Dual variables (domains follow the min-primal convention above).   #
    # ------------------------------------------------------------------ #
    m.lam = pyo.Var(m.N, m.T, domain=pyo.Reals)            # nodal balance (LMP)
    m.lam_sys = pyo.Var(m.T, domain=pyo.Reals)             # system balance
    m.nu_gen = pyo.Var(m.G, m.T, domain=pyo.NonPositiveReals)   # P_gen <= cap
    m.mu_up = pyo.Var(m.L, m.T, domain=pyo.NonPositiveReals)    # flow <= +limit
    m.mu_dn = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)    # flow >= -limit
    m.rho_ch = pyo.Var(m.I, m.N, m.T, domain=pyo.NonPositiveReals)  # charge <= X_pow
    m.sig_dis = pyo.Var(m.I, m.N, m.T, domain=pyo.NonPositiveReals)  # dis <= X_pow
    m.gam = pyo.Var(m.I, m.N, m.T, domain=pyo.Reals)       # SOC transition
    m.del_soc = pyo.Var(m.I, m.N, m.T_SOC, domain=pyo.NonPositiveReals)  # SOC <= X_en
    m.rho_per = pyo.Var(m.I, m.N, domain=pyo.Reals)        # SOC periodicity
    m.xi_shed = pyo.Var(m.N, m.T, domain=pyo.NonPositiveReals)  # shed <= demand

    # ------------------------------------------------------------------ #
    # Dual objective:  max  b^T y                                        #
    # ------------------------------------------------------------------ #
    def dual_objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
        demand_terms = sum(
            data.demand_el[n, t] * (model.lam[n, t] + model.xi_shed[n, t])
            for n in model.N
            for t in model.T
        )
        gen_cap_terms = sum(
            data.generation_capacity[g, t] * model.nu_gen[g, t]
            for g in model.G
            for t in model.T
        )
        line_terms = sum(
            data.line_limit[l] * (model.mu_up[l, t] - model.mu_dn[l, t])
            for l in model.L
            for t in model.T
        )
        power_terms = sum(
            data.x_power[i, n] * (model.rho_ch[i, n, t] + model.sig_dis[i, n, t])
            for i in model.I
            for n in model.N
            for t in model.T
        )
        energy_terms = sum(
            data.x_energy[i, n] * model.del_soc[i, n, tau]
            for i in model.I
            for n in model.N
            for tau in model.T_SOC
        )
        return demand_terms + gen_cap_terms + line_terms + power_terms + energy_terms

    m.objective = pyo.Objective(rule=dual_objective_rule, sense=pyo.maximize)

    # ------------------------------------------------------------------ #
    # Dual (stationarity) constraints, one per primal variable.          #
    # ------------------------------------------------------------------ #

    # P_gen[g,t] >= 0 :  sum_{n : g in G_n} lam[n,t] + nu_gen[g,t] <= C_g
    def gen_stationarity_rule(model: pyo.ConcreteModel, g: str, t: int) -> pyo.Expression:
        lam_sum = sum(model.lam[n, t] for n in gen_nodes.get(g, []))
        return lam_sum + model.nu_gen[g, t] <= data.generation_cost[g]

    m.gen_stationarity = pyo.Constraint(m.G, m.T, rule=gen_stationarity_rule)

    # P_shed[n,t] >= 0 :  lam[n,t] + xi_shed[n,t] <= VOLL
    def shed_stationarity_rule(model: pyo.ConcreteModel, n: str, t: int) -> pyo.Expression:
        return model.lam[n, t] + model.xi_shed[n, t] <= data.voll

    m.shed_stationarity = pyo.Constraint(m.N, m.T, rule=shed_stationarity_rule)

    # P_charge[i,n,t] >= 0 :  -lam[n,t] + rho_ch[i,n,t] - eta*gam[i,n,t] <= 0
    def charge_stationarity_rule(model: pyo.ConcreteModel, i: str, n: str, t: int) -> pyo.Expression:
        return -model.lam[n, t] + model.rho_ch[i, n, t] - eta * model.gam[i, n, t] <= 0.0

    m.charge_stationarity = pyo.Constraint(m.I, m.N, m.T, rule=charge_stationarity_rule)

    # P_discharge[i,n,t] >= 0 :  lam[n,t] + sig_dis[i,n,t] + (1/eta)*gam[i,n,t] <= 0
    def discharge_stationarity_rule(model: pyo.ConcreteModel, i: str, n: str, t: int) -> pyo.Expression:
        return model.lam[n, t] + model.sig_dis[i, n, t] + model.gam[i, n, t] / eta <= 0.0

    m.discharge_stationarity = pyo.Constraint(m.I, m.N, m.T, rule=discharge_stationarity_rule)

    # NetInjection[n,t] free :
    #   -lam[n,t] + lam_sys[t] + sum_l PTDF[l,n]*(mu_up[l,t] + mu_dn[l,t]) = 0
    def netinjection_stationarity_rule(model: pyo.ConcreteModel, n: str, t: int) -> pyo.Expression:
        flow_terms = sum(
            data.ptdf[l, n] * (model.mu_up[l, t] + model.mu_dn[l, t]) for l in model.L
        )
        return -model.lam[n, t] + model.lam_sys[t] + flow_terms == 0.0

    m.netinjection_stationarity = pyo.Constraint(m.N, m.T, rule=netinjection_stationarity_rule)

    # SOC[i,n,tau] >= 0, tau in T_SOC (0..last_t). The transition
    #   SOC[t] - SOC[t-1] - eta*P_charge[t] + (1/eta)*P_discharge[t] = 0
    # gives SOC[tau] the coefficient +gam at t=tau and -gam at t=tau+1.
    # Periodicity SOC[0]-SOC[last_t]=0 adds +rho_per at tau=0 and -rho_per at last_t.
    def soc_stationarity_rule(model: pyo.ConcreteModel, i: str, n: str, tau: int) -> pyo.Expression:
        expr = model.del_soc[i, n, tau]
        if tau in model.T:                 # +gam[tau] when this SOC is the "new" state
            expr = expr + model.gam[i, n, tau]
        if (tau + 1) in model.T:           # -gam[tau+1] when this SOC is the "previous" state
            expr = expr - model.gam[i, n, tau + 1]
        if tau == 0:
            expr = expr + model.rho_per[i, n]
        if tau == last_t:
            expr = expr - model.rho_per[i, n]
        return expr <= 0.0

    m.soc_stationarity = pyo.Constraint(m.I, m.N, m.T_SOC, rule=soc_stationarity_rule)

    m._market_data = data
    return m


def dual_objective_value(model: pyo.ConcreteModel) -> float:
    return value(model.objective)


def recover_lmps(model: pyo.ConcreteModel) -> Dict[Tuple[str, int], float]:
    """Nodal prices lambda_{n,t} read directly off the dual variables."""
    return {
        (n, t): value(model.lam[n, t])
        for n in model.N
        for t in model.T
    }


def print_dual_summary(model: pyo.ConcreteModel) -> None:
    data: MarketData = model._market_data
    print(f"\nDual objective value: {dual_objective_value(model):,.4f}")
    print("Formulation: explicit LP dual of the spot-market clearing primal.")

    print("\nSystem-wide lambda_sys by time:")
    for t in model.T:
        print(f"  t={t}: lambda_sys={value(model.lam_sys[t]):10.4f}")

    print("\nNodal prices lambda_{n,t} (LMPs) by time:")
    for t in model.T:
        prices = ", ".join(f"{n}={value(model.lam[n, t]):10.4f}" for n in model.N)
        print(f"  t={t}: {prices}")


# ---------------------------------------------------------------------- #
# Comparison harness: solve primal and dual, confirm they agree.         #
# ---------------------------------------------------------------------- #
def _primal_lmps(primal: pyo.ConcreteModel) -> Dict[Tuple[str, int], float]:
    """LMPs from the primal via the nodal_balance constraint duals (if available)."""
    lmps: Dict[Tuple[str, int], float] = {}
    for n in primal.N:
        for t in primal.T:
            dual = primal.dual.get(primal.nodal_balance[n, t], None)
            if dual is not None:
                lmps[(n, t)] = float(dual)
    return lmps


def _best_sign_diff(
    a: Mapping[Tuple[str, int], float],
    b: Mapping[Tuple[str, int], float],
) -> Tuple[float, int]:
    """Max abs difference between two price maps, trying both global signs.

    Returns (max_abs_diff, sign) where sign in {+1, -1} is the orientation of
    the primal duals that best matches the dual model's lambda variables.
    Pyomo's dual sign for equality constraints is solver-dependent, so we align
    a single global sign before comparing.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 1
    diff_pos = max(abs(a[k] - b[k]) for k in keys)
    diff_neg = max(abs(-a[k] - b[k]) for k in keys)
    return (diff_pos, 1) if diff_pos <= diff_neg else (diff_neg, -1)


def compare_models(data: MarketData, solver_name: str | None = None) -> int:
    primal = build_primal_market_clearing_model(data)
    dual = build_dual_market_clearing_model(data)

    primal_results = solve_model(primal, solver_name)
    dual_results = solve_model(dual, solver_name)

    for label, results in (("primal", primal_results), ("dual", dual_results)):
        tc = results.solver.termination_condition
        if tc not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
            print(f"{label} solve did not return optimal/feasible (termination={tc}).")
            return 1

    primal_obj = value(primal.objective)
    dual_obj = value(dual.objective)
    gap_abs = abs(primal_obj - dual_obj)
    denom = max(1.0, abs(primal_obj))
    gap_rel = gap_abs / denom

    print("\n" + "=" * 60)
    print("PRIMAL vs DUAL comparison")
    print("=" * 60)
    print(f"  Primal objective (min): {primal_obj:,.6f}")
    print(f"  Dual   objective (max): {dual_obj:,.6f}")
    print(f"  Absolute duality gap  : {gap_abs:.6e}")
    print(f"  Relative duality gap  : {gap_rel:.6e}")

    obj_ok = gap_rel <= 1e-6
    print(f"  Strong duality holds  : {'YES' if obj_ok else 'NO'}")

    # Optional: cross-check LMPs (primal constraint duals vs dual variables).
    dual_lmps = recover_lmps(dual)
    primal_lmps = _primal_lmps(primal)
    lmp_ok = True
    if primal_lmps:
        max_diff, sign = _best_sign_diff(primal_lmps, dual_lmps)
        lmp_ok = max_diff <= 1e-4
        orient = "same" if sign == 1 else "flipped (solver dual-sign convention)"
        print(f"  Max |LMP| difference  : {max_diff:.6e}  [primal orientation: {orient}]")
        print(f"  LMPs match            : {'YES' if lmp_ok else 'NO'}")
    else:
        print("  LMPs match            : primal duals unavailable (skipped)")

    print("=" * 60)
    return 0 if obj_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--solver", default=None, help="Optional Pyomo solver name, e.g. appsi_highs, glpk, cbc.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Processed data JSON from prepare_data.py.")
    parser.add_argument("--compare", action="store_true", help="Solve primal and dual and check strong duality.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)

    if args.compare:
        return compare_models(data, args.solver)

    model = build_dual_market_clearing_model(data)
    results = solve_model(model, args.solver)
    termination = results.solver.termination_condition
    print(f"Solver termination: {termination}")
    if termination not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        print("Solve did not return an optimal or feasible solution.")
        return 1

    print_dual_summary(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
