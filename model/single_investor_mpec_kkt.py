"""Single-investor MPEC via explicit KKT + Big-M, solved as one MILP.

This is the same bilevel model as ``single_investor_mpec.py`` (one strategic BESS
investor over a deterministic spot-market lower level), but the lower-level
optimality is imposed through the *explicit KKT system with Big-M linearised
complementarity* instead of the strong-duality (Wolfe) reformulation. The result
is a single mixed-integer linear program that HiGHS can solve to global
optimality, which lets us cross-check the Ipopt strong-duality NLP.

Design notes
------------
* Lower level is run in **fixed-demand** mode (no load shedding). The quadratic
  demand curve of the NLP would put a convex ``P_shed^2`` term in the linearised
  revenue, breaking the MILP; in the high-headroom European scenario nothing is
  ever shed, so fixed demand and the demand curve give the same dispatch. For a
  clean, apples-to-apples comparison the NLP is *also* run in fixed-demand mode.
* Investment ``X_power``/``X_energy`` stay **continuous**. The bilinear investor
  revenue ``sum lam*(dis-ch)`` is linearised exactly via the nodal balance plus
  lower-level complementarity, and the capacity variables cancel out of it:

      R = sum_l F_l*(mu_up-mu_dn) - sum_g c_g*P_gen + sum_g cap*nu_gen + sum_n,t D*lam

  so no dual x capacity products appear and discretisation is unnecessary.
* Big-M complementarity is added for every lower-level inequality: four
  variable/reduced-cost pairs (gen, charge, discharge, SOC) and six
  constraint/dual pairs (gen cap, two line limits, charge cap, discharge cap,
  SOC cap). No rival unit is supported (the revenue linearisation assumes the
  investor is the only storage), so ``existing_power_mw`` must be 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_NODE_LIMIT_MW,
    EXPERIMENT_DATA_PATH,
    InvestorConfig,
    _nodes_of_generator,
    build_single_investor_mpec,
    default_quadratic_demand_curve,
    initialize_from_reference_dispatch,
)
from solver_utils import get_ipopt_solver

MODEL_NAME = "Single Investor KKT Big-M MILP"
DEFAULT_BIG_M_DUAL = 800.0
DEFAULT_TIME_LIMIT = 300.0
DEFAULT_MIP_GAP = 1e-4


def _flow(model: pyo.ConcreteModel, data: MarketData, l: str, t: int) -> pyo.Expression:
    return sum(data.ptdf[l, n] * model.NetInjection[n, t] for n in data.nodes)


def _add_complementarity(
    model: pyo.ConcreteModel,
    name: str,
    index: list[tuple],
    slack_rule,
    dual_rule,
    m_slack_rule,
    m_dual: float,
) -> None:
    """Big-M linearised complementarity ``slack >= 0  _|_  dual_mag >= 0``.

    ``slack_rule`` and ``dual_rule`` return non-negative expressions (feasibility
    and dual-feasibility are already enforced elsewhere). Binary ``z=1`` frees the
    slack and forces the dual magnitude to zero; ``z=0`` does the reverse.
    """

    z = pyo.Var(index, domain=pyo.Binary)
    model.add_component(f"{name}_z", z)

    def _slack_con(m, *idx):
        return slack_rule(m, *idx) <= m_slack_rule(*idx) * z[idx]

    def _dual_con(m, *idx):
        return dual_rule(m, *idx) <= m_dual * (1.0 - z[idx])

    model.add_component(f"{name}_slack_bigm", pyo.Constraint(index, rule=_slack_con))
    model.add_component(f"{name}_dual_bigm", pyo.Constraint(index, rule=_dual_con))


def build_single_investor_mpec_kkt(
    data: MarketData,
    *,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    big_m_dual: float = DEFAULT_BIG_M_DUAL,
    investor: InvestorConfig | None = None,
    initial_power_mw: float = 0.0,
    fixed_power_mw: float | None = None,
) -> pyo.ConcreteModel:
    """Build the KKT + Big-M MILP form of the single-investor MPEC."""

    inv = investor or InvestorConfig()
    quad = default_quadratic_demand_curve()

    # Reuse the strong-duality builder purely for its scaffolding: primal vars and
    # constraints, dual vars, stationarity (dual-feasibility) constraints, and the
    # investment / capex / degradation expressions. Fixed-demand mode fixes P_shed
    # and xi_shed to zero, so no shed complementarity is required.
    m = build_single_investor_mpec(
        data,
        node_limit_mw=node_limit_mw,
        initial_power_mw=initial_power_mw,
        fixed_power_mw=fixed_power_mw,
        quad_demand=quad,
        use_demand_curve=False,
        existing_power_mw=0.0,
        investor=inv,
    )
    if any(v > 1e-9 for v in m._rival_power_mw.values()):
        raise ValueError("KKT MILP form does not support a rival storage unit.")

    # KKT replaces strong duality; the bilinear objective is replaced below.
    m.strong_duality.deactivate()
    m.objective.deactivate()

    # Tighten the (previously huge) dual-variable bounds for Big-M numerics.
    B = big_m_dual
    for (g, t) in m.nu_gen:
        m.nu_gen[g, t].setlb(-B); m.nu_gen[g, t].setub(0.0)
    for (l, t) in m.mu_up:
        m.mu_up[l, t].setlb(-B); m.mu_up[l, t].setub(0.0)
        m.mu_dn[l, t].setlb(0.0); m.mu_dn[l, t].setub(B)
    for key in m.rho_ch:
        m.rho_ch[key].setlb(-B); m.rho_ch[key].setub(0.0)
        m.sig_dis[key].setlb(-B); m.sig_dis[key].setub(0.0)
    for key in m.del_soc:
        m.del_soc[key].setlb(-B); m.del_soc[key].setub(0.0)
    for (n, t) in m.lam:
        m.lam[n, t].setlb(-B); m.lam[n, t].setub(B)
    for t in m.lam_sys:
        m.lam_sys[t].setlb(-B); m.lam_sys[t].setub(B)
    for key in m.gam:
        m.gam[key].setlb(-B); m.gam[key].setub(B)
    for key in m.rho_per:
        m.rho_per[key].setlb(-B); m.rho_per[key].setub(B)

    gen_nodes = _nodes_of_generator(data)
    eta = data.eta
    times = set(data.times)
    last_t = max(data.times)
    inv_id = m._investor_id
    cap = data.generation_capacity
    F = data.line_limit

    GT = [(g, t) for g in data.generators for t in data.times]
    LT = [(l, t) for l in data.lines for t in data.times]
    INT = [(inv_id, n, t) for n in data.nodes for t in data.times]
    INS = [(inv_id, n, tau) for n in data.nodes for tau in data.soc_times]

    # ---- Reduced-cost (variable) expressions; each is >= 0 by stationarity ----
    def rc_gen(mm, g, t):
        return data.generation_cost[g] - sum(mm.lam[n, t] for n in gen_nodes.get(g, [])) - mm.nu_gen[g, t]

    def rc_ch(mm, i, n, t):
        return mm.lam[n, t] - mm.rho_ch[i, n, t] + eta * mm.gam[i, n, t]

    def rc_dis(mm, i, n, t):
        return -mm.lam[n, t] - mm.sig_dis[i, n, t] - mm.gam[i, n, t] / eta

    def rc_soc(mm, i, n, tau):
        expr = mm.del_soc[i, n, tau]
        if tau in times:
            expr = expr + mm.gam[i, n, tau]
        if (tau + 1) in times:
            expr = expr - mm.gam[i, n, tau + 1]
        if tau == 0:
            expr = expr + mm.rho_per[i, n]
        if tau == last_t:
            expr = expr - mm.rho_per[i, n]
        return -expr

    # ---- Variable / reduced-cost complementarity ----
    _add_complementarity(m, "comp_gen", GT,
                         lambda mm, g, t: mm.P_gen[g, t], rc_gen,
                         lambda g, t: cap[g, t], B)
    _add_complementarity(m, "comp_charge", INT,
                         lambda mm, i, n, t: mm.P_charge[i, n, t], rc_ch,
                         lambda i, n, t: node_limit_mw, B)
    _add_complementarity(m, "comp_discharge", INT,
                         lambda mm, i, n, t: mm.P_discharge[i, n, t], rc_dis,
                         lambda i, n, t: node_limit_mw, B)
    _add_complementarity(m, "comp_soc", INS,
                         lambda mm, i, n, tau: mm.SOC[i, n, tau], rc_soc,
                         lambda i, n, tau: inv.ratio_max * node_limit_mw, B)

    # ---- Constraint / dual complementarity ----
    _add_complementarity(m, "comp_gencap", GT,
                         lambda mm, g, t: cap[g, t] - mm.P_gen[g, t],
                         lambda mm, g, t: -mm.nu_gen[g, t],
                         lambda g, t: cap[g, t], B)
    _add_complementarity(m, "comp_lineup", LT,
                         lambda mm, l, t: F[l] - _flow(mm, data, l, t),
                         lambda mm, l, t: -mm.mu_up[l, t],
                         lambda l, t: 2.0 * F[l], B)
    _add_complementarity(m, "comp_linedn", LT,
                         lambda mm, l, t: _flow(mm, data, l, t) + F[l],
                         lambda mm, l, t: mm.mu_dn[l, t],
                         lambda l, t: 2.0 * F[l], B)
    _add_complementarity(m, "comp_chargecap", INT,
                         lambda mm, i, n, t: mm.X_power[n] - mm.P_charge[i, n, t],
                         lambda mm, i, n, t: -mm.rho_ch[i, n, t],
                         lambda i, n, t: node_limit_mw, B)
    _add_complementarity(m, "comp_dischargecap", INT,
                         lambda mm, i, n, t: mm.X_power[n] - mm.P_discharge[i, n, t],
                         lambda mm, i, n, t: -mm.sig_dis[i, n, t],
                         lambda i, n, t: node_limit_mw, B)
    _add_complementarity(m, "comp_soccap", INS,
                         lambda mm, i, n, tau: mm.X_energy[n] - mm.SOC[i, n, tau],
                         lambda mm, i, n, tau: -mm.del_soc[i, n, tau],
                         lambda i, n, tau: inv.ratio_max * node_limit_mw, B)

    # ---- Exact linear form of the investor spot revenue (see module docstring) ----
    m.spot_revenue_linear_expr = pyo.Expression(
        expr=sum(F[l] * (m.mu_up[l, t] - m.mu_dn[l, t]) for l in m.L for t in m.T)
        - sum(data.generation_cost[g] * m.P_gen[g, t] for g in m.G for t in m.T)
        + sum(cap[g, t] * m.nu_gen[g, t] for g in m.G for t in m.T)
        + sum(data.demand_el[n, t] * m.lam[n, t] for n in m.N for t in m.T)
    )
    m.kkt_profit_expr = pyo.Expression(
        expr=m.spot_revenue_linear_expr - m.degradation_cost_expr - m.capex_daily_expr
    )
    m.kkt_objective = pyo.Objective(expr=m.kkt_profit_expr, sense=pyo.maximize)

    m._big_m_dual = B
    return m


def get_milp_solver(
    time_limit: float = DEFAULT_TIME_LIMIT,
    mip_gap: float = DEFAULT_MIP_GAP,
    load_solution: bool = True,
    warmstart: bool = False,
):
    from pyomo.contrib.appsi.solvers import Highs

    opt = Highs()
    if not opt.available():
        raise RuntimeError("appsi Highs is not available for the MILP.")
    opt.config.time_limit = time_limit
    opt.config.mip_gap = mip_gap
    opt.config.load_solution = load_solution
    opt.config.stream_solver = True
    opt.config.warmstart = warmstart
    return opt


def _binary_vars(model: pyo.ConcreteModel) -> list:
    return [v for v in model.component_data_objects(pyo.Var) if v.is_binary()]


def warm_start_milp_from_nlp(milp: pyo.ConcreteModel, nlp: pyo.ConcreteModel, data: MarketData) -> None:
    """Seed the MILP with the (KKT-feasible) strong-duality NLP solution as an incumbent."""

    def cp(vm, vn):
        for idx in vm:
            vm[idx].set_value(value(vn[idx]), skip_validation=True)

    for name in ("P_gen", "P_shed", "P_charge", "P_discharge", "SOC", "NetInjection",
                 "lam", "lam_sys", "nu_gen", "mu_up", "mu_dn", "rho_ch", "sig_dis",
                 "gam", "del_soc", "rho_per", "X_power", "X_energy"):
        cp(getattr(milp, name), getattr(nlp, name))

    tol = 1e-3
    v = value
    cap = data.generation_capacity

    def flow(l, t):
        return sum(data.ptdf[l, n] * v(milp.NetInjection[n, t]) for n in data.nodes)

    for key in milp.comp_gen_z:
        milp.comp_gen_z[key].set_value(1 if v(milp.P_gen[key]) > tol else 0)
    for key in milp.comp_charge_z:
        milp.comp_charge_z[key].set_value(1 if v(milp.P_charge[key]) > tol else 0)
    for key in milp.comp_discharge_z:
        milp.comp_discharge_z[key].set_value(1 if v(milp.P_discharge[key]) > tol else 0)
    for key in milp.comp_soc_z:
        milp.comp_soc_z[key].set_value(1 if v(milp.SOC[key]) > tol else 0)
    for (g, t) in milp.comp_gencap_z:
        milp.comp_gencap_z[g, t].set_value(1 if cap[g, t] - v(milp.P_gen[g, t]) > tol else 0)
    for (l, t) in milp.comp_lineup_z:
        milp.comp_lineup_z[l, t].set_value(1 if data.line_limit[l] - flow(l, t) > tol else 0)
    for (l, t) in milp.comp_linedn_z:
        milp.comp_linedn_z[l, t].set_value(1 if flow(l, t) + data.line_limit[l] > tol else 0)
    for (i, n, t) in milp.comp_chargecap_z:
        milp.comp_chargecap_z[i, n, t].set_value(1 if v(milp.X_power[n]) - v(milp.P_charge[i, n, t]) > tol else 0)
    for (i, n, t) in milp.comp_dischargecap_z:
        milp.comp_dischargecap_z[i, n, t].set_value(1 if v(milp.X_power[n]) - v(milp.P_discharge[i, n, t]) > tol else 0)
    for (i, n, tau) in milp.comp_soccap_z:
        milp.comp_soccap_z[i, n, tau].set_value(1 if v(milp.X_energy[n]) - v(milp.SOC[i, n, tau]) > tol else 0)


def _bilinear_spot_revenue(model: pyo.ConcreteModel) -> float:
    inv_id = model._investor_id
    return sum(
        value(model.lam[n, t]) * (value(model.P_discharge[inv_id, n, t]) - value(model.P_charge[inv_id, n, t]))
        for n in model.N
        for t in model.T
    )


def _dual_bound_report(model: pyo.ConcreteModel) -> str:
    """Per-family max |dual|. nu_gen (capacity shadow price) is dual-degenerate in
    fixed-demand mode, so it may sit at the Big-M bound without affecting the unique
    prices/investment/profit; the price-like duals are what must stay well below B."""

    B = model._big_m_dual

    def mx(comp):
        return max((abs(value(comp[key])) for key in comp), default=0.0)

    families = {
        "lam": model.lam, "mu_dn": model.mu_dn, "mu_up": model.mu_up,
        "gam": model.gam, "del_soc": model.del_soc, "nu_gen": model.nu_gen,
    }
    parts = [f"{name}={mx(comp):.0f}" for name, comp in families.items()]
    price_like = max(mx(model.lam), mx(model.mu_dn), mx(model.mu_up), mx(model.gam), mx(model.del_soc))
    return f"max|dual| [{', '.join(parts)}]  (big_m={B:.0f}; price-like/big_m={price_like / B:.2f})"


def solve_and_compare(
    data: MarketData,
    *,
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW,
    big_m_dual: float = DEFAULT_BIG_M_DUAL,
    time_limit: float = DEFAULT_TIME_LIMIT,
    tee: bool = False,
) -> None:
    inv = InvestorConfig()

    print("=" * 78)
    print("Strong-duality NLP (Ipopt), fixed-demand mode  [reference / warm start]")
    print("=" * 78)
    nlp = build_single_investor_mpec(
        data,
        node_limit_mw=node_limit_mw,
        initial_power_mw=10.0,
        quad_demand=default_quadratic_demand_curve(),
        use_demand_curve=False,
        existing_power_mw=0.0,
        investor=inv,
    )
    initialize_from_reference_dispatch(nlp, data, inv.ratio_min)
    nlp_res = get_ipopt_solver({"max_cpu_time": 120.0}).solve(nlp, tee=tee)
    print(f"status: {nlp_res.solver.termination_condition}")
    nlp_profit = value(nlp.investor_profit_expr)
    print(f"profit (NLP obj):             {nlp_profit:12.2f} EUR/day")

    print("\n" + "=" * 78)
    print("KKT + Big-M MILP (HiGHS), warm-started from the NLP incumbent")
    print("=" * 78)
    milp = build_single_investor_mpec_kkt(
        data, node_limit_mw=node_limit_mw, big_m_dual=big_m_dual, investor=inv
    )
    binaries = _binary_vars(milp)
    print(f"binaries: {len(binaries)}   big_m_dual: {big_m_dual}")
    warm_start_milp_from_nlp(milp, nlp, data)

    # (1) Validation: fix the complementarity pattern to the NLP's active set and
    # solve the resulting LP. If the KKT+Big-M encoding and the exact revenue
    # linearisation are correct, this reproduces the NLP profit exactly.
    for z in binaries:
        z.fix(round(z.value))
    get_milp_solver(time_limit=time_limit, load_solution=True).solve(milp)
    lp_profit = value(milp.kkt_profit_expr)
    lin_rev = value(milp.spot_revenue_linear_expr)
    bil_rev = _bilinear_spot_revenue(milp)
    print(f"[fixed-pattern LP]  profit {lp_profit:12.2f}   linrev {lin_rev:12.2f}"
          f"   bilinear-check {bil_rev:12.2f}  |diff|={abs(lin_rev - bil_rev):.3g}")
    print(f"                    {_dual_bound_report(milp)}")
    for z in binaries:
        z.unfix()

    # (2) Full MILP: let HiGHS search all complementarity patterns for a better
    # investor solution, warm-started from the NLP incumbent.
    opt = get_milp_solver(time_limit=time_limit, load_solution=False, warmstart=True)
    milp_res = opt.solve(milp)
    tc = milp_res.termination_condition
    best = getattr(milp_res, "best_feasible_objective", None)
    bound = getattr(milp_res, "best_objective_bound", None)
    print(f"[full MILP] termination: {tc}   best_obj: {best}   bound: {bound}")
    have_incumbent = best is not None
    if have_incumbent:
        try:
            milp_res.solution_loader.load_vars()
        except Exception:
            have_incumbent = False
    if not have_incumbent:
        print("  no incumbent loaded from full MILP; using fixed-pattern LP solution for comparison.")
        for z in binaries:
            z.fix(round(z.value))
        get_milp_solver(time_limit=time_limit, load_solution=True).solve(milp)
        for z in binaries:
            z.unfix()
    milp_profit = value(milp.kkt_profit_expr)

    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"{'node':>6} | {'X_power MW (MILP/NLP)':>26} | {'X_energy MWh (MILP/NLP)':>26}")
    for n in data.nodes:
        xp_m, xp_n = value(milp.X_power[n]), value(nlp.X_power[n])
        xe_m, xe_n = value(milp.X_energy[n]), value(nlp.X_energy[n])
        print(f"{n:>6} | {xp_m:11.2f} / {xp_n:11.2f} | {xe_m:11.2f} / {xe_n:11.2f}")
    print(f"\n{'':>6}   total X_power: MILP {sum(value(milp.X_power[n]) for n in data.nodes):8.2f}"
          f"   NLP {sum(value(nlp.X_power[n]) for n in data.nodes):8.2f} MW")
    print(f"{'':>6}   profit:        MILP {milp_profit:8.2f}   NLP {nlp_profit:8.2f} EUR/day"
          f"   diff {milp_profit - nlp_profit:+.2f}")

    print("\nNodal prices lambda (MILP vs NLP), selected hours:")
    for t in (9, 13, 14, 20):
        if t not in data.times:
            continue
        milp_row = "  ".join(f"{value(milp.lam[n, t]):7.1f}" for n in data.nodes)
        nlp_row = "  ".join(f"{value(nlp.lam[n, t]):7.1f}" for n in data.nodes)
        print(f"  t={t:2d} MILP: {milp_row}")
        print(f"       NLP : {nlp_row}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--node-limit-mw", type=float, default=DEFAULT_NODE_LIMIT_MW)
    parser.add_argument("--big-m-dual", type=float, default=DEFAULT_BIG_M_DUAL)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.data)
    solve_and_compare(
        data,
        node_limit_mw=args.node_limit_mw,
        big_m_dual=args.big_m_dual,
        time_limit=args.time_limit,
        tee=args.tee,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
