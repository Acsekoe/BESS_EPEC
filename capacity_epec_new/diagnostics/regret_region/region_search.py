"""Structured unilateral search for a low-regret region.

For each candidate profile and each investor, deviations are generated from a
local basis of FEASIBLE unilateral directions and every deviation is valued by
the maintained exact reclear:

  scale_node      duration-preserving scaling of one node, including node exit
  power_only      power change at fixed energy, kept inside the 2-8 h cone
  energy_only     energy change at fixed power, kept inside the 2-8 h cone
  transfer        move a fraction of one node's (P, E) to another node
  entry           new position at an empty node, 1 or 10 MW at 2, 4 or 8 h
  portfolio       scaling of the whole portfolio, including complete exit
  duration_all    every node set to one duration at unchanged power
  gradient        steps along the finite-difference profit gradient
                  (active-set guided), retracted onto the feasible set

Feasibility: every generated vector is retracted node by node onto
{0 <= P <= residual shared cap, 2P <= E <= 8P} and re-checked with the
maintained guard before it is cleared.  This covers expansion, contraction,
relocation, duration and exit decisions.

Also computed per candidate:
  * incumbent-start and best-probe-start MPEC refinements (run settings);
  * for I1 (no generation) the merchant-entry upper bound
      B = V(0) - min_x [V(x) + k(x)]
    which is a valid upper bound on I1's profit from ANY capacity because the
    ISO value function V is convex in capacity at rho >= 0;
  * at the final profile, joint random perturbations of all investors to
    measure payoff flatness (this is not a regret measure).

Every finite search here is a LOWER bound on regret; only the I1 bound is an
upper bound.  Outputs: output/region/*.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyomo.environ as pyo

import regret_common as rc
import capacity_game
import iso_market
import solvers
from investors import daily_investment_cost

OUT = rc.OUTPUT_ROOT / "region"
NODES = [f"N{k}" for k in range(1, 10)]
RMIN, RMAX = 2.0, 8.0


# --------------------------------------------------------------------------
# Candidates


def sweep_state(sweep: int):
    power, energy = {}, {}
    with (rc.RUN_DIR / "capacity_by_investor_node_by_sweep.csv").open(newline="", encoding="utf-8") as h:
        for row in csv.DictReader(h):
            if int(row["sweep"]) == sweep:
                power[row["investor"], row["node"]] = float(row["power_mw"])
                energy[row["investor"], row["node"]] = float(row["energy_mwh"])
    if len(power) != 27:
        raise RuntimeError(f"sweep {sweep} state incomplete")
    return power, energy


def candidates() -> dict:
    power, energy = rc.incumbent()
    out = {"final_sweep100": (power, energy)}
    i3_8h = rc.selected_record("I3")
    out["I3_adopts_duration_8h"] = rc.with_candidate(power, energy, "I3", i3_8h["power"], i3_8h["energy"])
    i3_loc = rc.audit_record("I3", "incumbent", "refine")
    out["I3_adopts_incumbent_start_response"] = rc.with_candidate(power, energy, "I3", i3_loc["power"], i3_loc["energy"])
    p, e = dict(power), dict(energy)
    for inv in rc.INVESTORS:
        sel = rc.selected_record(inv)
        p, e = rc.with_candidate(p, e, inv, sel["power"], sel["energy"])
    out["all_adopt_selected_responses"] = (p, e)
    # Points on I3's duration_8h path: just past the midday decongestion boundary
    # found by the path scan (L78 hours 12-13 unpriced from s ~ 0.021) and inside
    # the decongested regime.  Feasible by convexity of the segment.
    inc_p, inc_e = rc.investor_vector(power, energy, "I3", NODES)
    for tag, s in (("I3_8h_path_s0.021_decongestion_boundary", 0.02125), ("I3_8h_path_s0.5_decongested", 0.5)):
        cp = {n: inc_p[n] + s * (i3_8h["power"][n] - inc_p[n]) for n in NODES}
        ce = {n: inc_e[n] + s * (i3_8h["energy"][n] - inc_e[n]) for n in NODES}
        out[tag] = rc.with_candidate(power, energy, "I3", cp, ce)
    out["sweep43_state"] = sweep_state(43)
    out["sweep10_state"] = sweep_state(10)
    return out


# --------------------------------------------------------------------------
# Direction basis


def residual_caps(power, investor, limits):
    return {n: limits[n] - sum(power[j, n] for j in rc.INVESTORS if j != investor) for n in NODES}


def retract(p, e, caps):
    rp, re_ = {}, {}
    for n in NODES:
        P = min(max(p[n], 0.0), caps[n])
        if P <= 0.0:
            rp[n], re_[n] = 0.0, 0.0
            continue
        rp[n] = P
        re_[n] = min(max(e[n], RMIN * P), RMAX * P)
    return rp, re_


FULL_GRID = dict(scale=(-1.0, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 0.5, 1.0), power=(0.05, 0.25, 1.0),
                 energy=(0.02, 0.1, 0.5, 1.0), transfer=(0.05, 0.25, 1.0), entry_mw=(1.0, 10.0),
                 entry_h=(2.0, 4.0, 8.0), portfolio=(-1.0, -0.5, -0.1, -0.02, 0.02, 0.1, 0.5),
                 duration=(2.0, 3.0, 4.0, 6.0, 8.0), gradient=(0.01, 0.1, 0.5, 2.0, 5.0))
# Same direction families, fewer step sizes (every family still includes exit/entry extremes).
LEAN_GRID = dict(scale=(-1.0, -0.5, -0.1, -0.02, 0.02, 0.1, 0.5), power=(0.25, 1.0), energy=(0.1, 1.0),
                 transfer=(0.25, 1.0), entry_mw=(10.0,), entry_h=(2.0, 8.0), portfolio=(-1.0, -0.5, -0.1, 0.1, 0.5),
                 duration=(2.0, 4.0, 8.0), gradient=(0.1, 1.0, 5.0))
GRID = FULL_GRID


def direction_probes(p0: dict, e0: dict, caps: dict):
    probes = []
    material = [n for n in NODES if p0[n] > 1e-3]
    empty = [n for n in NODES if p0[n] <= 1e-3]

    def add(family, label, p, e, step):
        rp, re_ = retract(p, e, caps)
        probes.append((family, label, step, rp, re_))

    for n in material:
        for lam in GRID["scale"]:
            p, e = dict(p0), dict(e0)
            p[n], e[n] = (1 + lam) * p0[n], (1 + lam) * e0[n]
            add("scale_node", f"scale_{n}_{lam:+g}", p, e, lam)
        lo, hi = e0[n] / RMAX - p0[n], e0[n] / RMIN - p0[n]
        for frac in GRID["power"]:
            for side, bound in (("down", lo), ("up", hi)):
                p = dict(p0)
                p[n] = p0[n] + frac * bound
                add("power_only", f"power_{n}_{side}_{frac:g}", p, e0, frac * bound)
        lo, hi = RMIN * p0[n] - e0[n], RMAX * p0[n] - e0[n]
        for frac in GRID["energy"]:
            for side, bound in (("down", lo), ("up", hi)):
                e = dict(e0)
                e[n] = e0[n] + frac * bound
                add("energy_only", f"energy_{n}_{side}_{frac:g}", p0, e, frac * bound)
        for m in NODES:
            if m == n:
                continue
            for frac in GRID["transfer"]:
                p, e = dict(p0), dict(e0)
                p[n], e[n] = (1 - frac) * p0[n], (1 - frac) * e0[n]
                p[m], e[m] = p0[m] + frac * p0[n], e0[m] + frac * e0[n]
                add("transfer", f"transfer_{n}_to_{m}_{frac:g}", p, e, frac)
    for m in empty:
        for mw in GRID["entry_mw"]:
            for hours in GRID["entry_h"]:
                p, e = dict(p0), dict(e0)
                p[m], e[m] = p0[m] + mw, e0[m] + mw * hours
                add("entry", f"entry_{m}_{mw:g}mw_{hours:g}h", p, e, mw)
    for lam in GRID["portfolio"]:
        add("portfolio", f"portfolio_{lam:+g}", {n: (1 + lam) * p0[n] for n in NODES},
            {n: (1 + lam) * e0[n] for n in NODES}, lam)
    for hours in GRID["duration"]:
        add("duration_all", f"duration_all_{hours:g}h", p0, {n: hours * p0[n] for n in NODES}, hours)
    return probes


def gradient_probes(p0, e0, caps, gradient, scale_mw):
    """Steps along the finite-difference gradient (P in MW, E in MWh/2)."""

    probes = []
    norm = max(abs(v) for v in gradient.values()) or 1.0
    for step in GRID["gradient"]:
        for sign in (+1, -1):
            p, e = dict(p0), dict(e0)
            for (kind, n), g in gradient.items():
                if kind == "P":
                    p[n] += sign * step * g / norm
                else:
                    e[n] += sign * 2.0 * step * g / norm
            rp, re_ = retract(p, e, caps)
            probes.append(("gradient", f"gradient_{'ascent' if sign > 0 else 'descent'}_{step:g}", sign * step, rp, re_))
    return probes


# --------------------------------------------------------------------------
# Merchant-entry upper bound for I1


def merchant_bound(rho: float, power: dict, energy: dict, merchant: str = "I1") -> dict:
    data = rc.market_data(rho)
    config = rc.game_config(data)
    investor = next(i for i in config.investors if i.investor_id == merchant)
    if investor.owned_generation_shares:
        raise ValueError("the bound only holds for an investor without generation")
    limits = config.node_limits(data)
    zero_p, zero_e = rc.with_candidate(power, energy, merchant, {n: 0.0 for n in NODES}, {n: 0.0 for n in NODES})
    out = {}
    for solver_name in ("ma57", "ma27"):
        base = rc.resolve_market(data, config, zero_p, zero_e, solver_name)
        v0 = float(pyo.value(base.objective))
        m = pyo.ConcreteModel()
        m.MN = pyo.Set(initialize=NODES, ordered=True)
        m.XP = pyo.Var(m.MN, domain=pyo.NonNegativeReals, initialize=lambda mm, n: power[merchant, n])
        m.XE = pyo.Var(m.MN, domain=pyo.NonNegativeReals, initialize=lambda mm, n: energy[merchant, n])
        mixed_p, mixed_e = dict(power), dict(energy)
        for n in NODES:
            mixed_p[merchant, n] = m.XP[n]
            mixed_e[merchant, n] = m.XE[n]
        iso_market.build_market(data, mixed_p, mixed_e, config.degradation(), model=m)
        m.dur_lo = pyo.Constraint(m.MN, rule=lambda mm, n: mm.XE[n] >= investor.ratio_min * mm.XP[n])
        m.dur_hi = pyo.Constraint(m.MN, rule=lambda mm, n: mm.XE[n] <= investor.ratio_max * mm.XP[n])
        m.cap = pyo.Constraint(m.MN, rule=lambda mm, n: mm.XP[n] + sum(
            power[j, n] for j in rc.INVESTORS if j != merchant) <= limits[n])
        capex = daily_investment_cost(investor, (m.XP[n] for n in NODES), (m.XE[n] for n in NODES), NODES)
        m.objective.deactivate()
        m.bound_objective = pyo.Objective(expr=m.objective.expr + capex, sense=pyo.minimize)
        solver = solvers._ipopt(config.solver, {
            "linear_solver": solver_name, "max_iter": 3000, "max_cpu_time": 600.0, "tol": 1e-11,
            "nlp_scaling_method": "none", "acceptable_tol": 1e-10, "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes", "print_level": 0})
        result = solver.solve(m, tee=False)
        w = float(pyo.value(m.bound_objective))
        out[f"{solver_name}_status"] = str(result.solver.termination_condition)
        out[f"{solver_name}_V0"] = v0
        out[f"{solver_name}_min_V_plus_capex"] = w
        out[f"{solver_name}_upper_bound"] = v0 - w
        if solver_name == "ma57":
            out["argmin_power"] = {n: float(pyo.value(m.XP[n])) for n in NODES}
            out["argmin_energy"] = {n: float(pyo.value(m.XE[n])) for n in NODES}
    out["upper_bound"] = max(out["ma57_upper_bound"], out["ma27_upper_bound"])
    out["upper_bound_solver_spread"] = abs(out["ma57_upper_bound"] - out["ma27_upper_bound"])
    return out


# --------------------------------------------------------------------------
# MPEC refinement task


def mpec_task(args):
    label, rho, investor_id, power, energy, start_label, start_power, start_energy = args
    data = rc.market_data(rho)
    config = rc.game_config(data)
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    tick = time.perf_counter()
    response = capacity_game.solve_from_start(data, config, investor, power, energy, start_label,
                                              start_power, start_energy)
    response = capacity_game.attach_recleared_profit(data, config, investor, power, energy, response)
    row = dict(candidate=label, rho=rho, investor=investor_id, start=start_label,
               termination=response.outcome.termination, optimal=response.outcome.optimal,
               embedded_profit=response.embedded_profit_eur_per_day,
               recleared_profit=response.recleared_profit_eur_per_day,
               embedded_minus_reclear=response.embedded_profit_eur_per_day - response.recleared_profit_eur_per_day,
               complementarity_max_product=response.max_complementarity_product,
               primal_dual_gap=response.primal_dual_gap_eur_per_day, seconds=time.perf_counter() - tick)
    for n in NODES:
        row[f"P_{n}"] = response.power[n]
    for n in NODES:
        row[f"E_{n}"] = response.energy[n]
    return row


def merchant_task(args):
    label, rho, power, energy = args
    tick = time.perf_counter()
    try:
        out = merchant_bound(rho, power, energy)
    except Exception as exc:
        return dict(candidate=label, rho=rho, error=str(exc))
    row = dict(candidate=label, rho=rho, seconds=time.perf_counter() - tick)
    for k, v in out.items():
        if isinstance(v, dict):
            for n, x in v.items():
                row[f"{k}_{n}"] = x
        else:
            row[k] = v
    return row


# --------------------------------------------------------------------------


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--lean", action="store_true",
                        help="Reduced step grid and the five boundary-relevant candidates.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rho = args.rho
    data = rc.market_data(rho)
    config = rc.game_config(data)
    limits = config.node_limits(data)
    cands = candidates()
    if args.lean:
        global GRID
        GRID = LEAN_GRID
        keep = ("final_sweep100", "I3_adopts_duration_8h", "I3_8h_path_s0.021_decongestion_boundary",
                "I3_8h_path_s0.5_decongested", "all_adopt_selected_responses")
        cands = {k: v for k, v in cands.items() if k in keep}
    t0 = time.perf_counter()
    probe_rows, base_rows = [], []
    base_results = {}

    with rc.make_pool(args.workers) as pool:
        base_results = rc.evaluate_many(pool, [((label,), rho, p, e) for label, (p, e) in cands.items()])
        static = rc.static_arrays(data)
        midday = [static["times"].index(t) for t in static["times"] if 11 <= t <= 16]
        for (label,), res in base_results.items():
            sets = rc.active_sets(res, static)
            row = dict(candidate=label, rho=rho, status=res["status"], pd_gap=res.get("pd_gap"),
                       verify_max_abs_profit_diff=res.get("verify_max_abs_profit_diff", 0.0),
                       total_power_mw=sum(cands[label][0].values()),
                       lines_priced=";".join(sorted(sets["lines_priced"])),
                       midday_mean_lmp_N3=float(np.mean(res["lmp"][NODES.index("N3"), midday])),
                       midday_mean_lmp_N6=float(np.mean(res["lmp"][NODES.index("N6"), midday])),
                       midday_mean_lmp_N8=float(np.mean(res["lmp"][NODES.index("N8"), midday])),
                       total_abs_da_mwh=res["total_abs_da_mwh"])
            for inv in rc.INVESTORS:
                row[f"profit_{inv}"] = res["settle"][inv]["profit"]
                row[f"total_power_{inv}"] = sum(cands[label][0][inv, n] for n in NODES)
            base_rows.append(row)
        print(f"bases done {time.perf_counter()-t0:.0f}s", flush=True)

        # finite-difference gradients (one-sided, h = 1e-3 MW / 2e-3 MWh) at material nodes
        fd_tasks, fd_meta = [], {}
        for label, (p, e) in cands.items():
            for inv in rc.INVESTORS:
                p0, e0 = rc.investor_vector(p, e, inv, NODES)
                for n in NODES:
                    if p0[n] <= 1e-3:
                        continue
                    for kind, h in (("P", 1e-3), ("E", 2e-3)):
                        cp, ce = dict(p0), dict(e0)
                        if kind == "P":
                            cp[n] += h if 2 * (p0[n] + h) <= e0[n] else -h
                        else:
                            ce[n] += h if e0[n] + h <= 8 * p0[n] else -h
                        full_p, full_e = rc.with_candidate(p, e, inv, cp, ce)
                        key = (label, inv, kind, n)
                        fd_meta[key] = (cp[n] - p0[n]) if kind == "P" else (ce[n] - e0[n])
                        fd_tasks.append((key, rho, full_p, full_e))
        fd = rc.evaluate_many(pool, fd_tasks)
        gradients = {}
        for (label, inv, kind, n), res in fd.items():
            base = base_results[(label,)]["settle"][inv]["profit"]
            gradients.setdefault((label, inv), {})[(kind, n)] = (res["settle"][inv]["profit"] - base) / fd_meta[(label, inv, kind, n)]
        print(f"gradients done {time.perf_counter()-t0:.0f}s", flush=True)
        gradient_rows = [dict(candidate=l, investor=i, coordinate=f"{k}_{n}", derivative_eur_per_day_per_unit=v)
                         for (l, i), g in gradients.items() for (k, n), v in g.items()]
        write_csv(OUT / "fd_gradients.csv", gradient_rows)

        tasks, meta = [], {}
        for label, (p, e) in cands.items():
            for inv in rc.INVESTORS:
                p0, e0 = rc.investor_vector(p, e, inv, NODES)
                caps = residual_caps(p, inv, limits)
                probes = direction_probes(p0, e0, caps) + gradient_probes(p0, e0, caps, gradients.get((label, inv), {}), 1.0)
                for k, (family, plabel, step, cp, ce) in enumerate(probes):
                    full_p, full_e = rc.with_candidate(p, e, inv, cp, ce)
                    try:
                        rc.check_feasible(data, config, inv, cp, ce, full_p)
                        feasible = True
                    except ValueError as exc:
                        feasible = str(exc)
                    key = (label, inv, k)
                    meta[key] = (family, plabel, step, cp, ce, feasible)
                    if feasible is True:
                        tasks.append((key, rho, full_p, full_e))
        print(f"evaluating {len(tasks)} probes", flush=True)
        results = rc.evaluate_many(pool, tasks)
        for key, (family, plabel, step, cp, ce, feasible) in meta.items():
            label, inv, k = key
            base = base_results[(label,)]
            row = dict(candidate=label, rho=rho, investor=inv, family=family, probe=plabel, step=step, feasible=feasible)
            for n in NODES:
                row[f"P_{n}"] = cp[n]
            for n in NODES:
                row[f"E_{n}"] = ce[n]
            res = results.get(key)
            if res is not None and res.get("status") == "optimal":
                st = res["settle"][inv]
                cur = base["settle"][inv]["profit"]
                row.update(status="optimal", profit=st["profit"], gain=st["profit"] - cur,
                           relative_gain=(st["profit"] - cur) / max(1.0, abs(cur)),
                           storage_operating_surplus=st["storage_operating_surplus"],
                           owned_generation_rent=st["owned_generation_rent"], capex=st["capex"],
                           max_abs_lmp_change=float(np.max(np.abs(res["lmp"] - base["lmp"]))),
                           pd_gap=res["pd_gap"], verify_max_abs_profit_diff=res.get("verify_max_abs_profit_diff", 0.0),
                           total_abs_da_mwh=res["total_abs_da_mwh"])
                for other in rc.INVESTORS:
                    row[f"profit_{other}"] = res["settle"][other]["profit"]
            elif res is not None:
                row["status"] = res.get("status")
            probe_rows.append(row)
        print(f"probes done {time.perf_counter()-t0:.0f}s", flush=True)

        # joint perturbations at the final profile (flatness, not regret)
        rng = np.random.default_rng(20260911)
        p, e = cands["final_sweep100"]
        joint_tasks, joint_meta = [], {}
        for radius in (0.01, 0.1, 1.0):
            for draw in range(16):
                jp, je = dict(p), dict(e)
                for inv in rc.INVESTORS:
                    p0, e0 = rc.investor_vector(p, e, inv, NODES)
                    material = [n for n in NODES if p0[n] > 1e-3]
                    z = rng.normal(size=len(material))
                    z = z / max(np.max(np.abs(z)), 1e-12) * radius
                    cp, ce = dict(p0), dict(e0)
                    for n, dz in zip(material, z):
                        dur = e0[n] / p0[n]
                        cp[n] = p0[n] + dz
                        ce[n] = e0[n] + dz * dur
                    caps = residual_caps(jp, inv, limits)
                    cp, ce = retract(cp, ce, caps)
                    jp, je = rc.with_candidate(jp, je, inv, cp, ce)
                key = ("joint", radius, draw)
                joint_meta[key] = (jp, je)
                joint_tasks.append((key, rho, jp, je))
        joint = rc.evaluate_many(pool, joint_tasks)
        joint_rows = []
        base = base_results[("final_sweep100",)]
        for (tag, radius, draw), res in joint.items():
            row = dict(radius_mw=radius, draw=draw, status=res.get("status"))
            if res.get("status") == "optimal":
                for inv in rc.INVESTORS:
                    d = res["settle"][inv]["profit"] - base["settle"][inv]["profit"]
                    row[f"dprofit_{inv}"] = d
                    row[f"relative_dprofit_{inv}"] = d / max(1.0, abs(base["settle"][inv]["profit"]))
                row["max_abs_lmp_change"] = float(np.max(np.abs(res["lmp"] - base["lmp"])))
            joint_rows.append(row)
        write_csv(OUT / "joint_perturbations_final.csv", joint_rows)
        print(f"joint perturbations done {time.perf_counter()-t0:.0f}s", flush=True)

    write_csv(OUT / "candidate_bases.csv", base_rows)
    write_csv(OUT / "probes.csv", probe_rows)

    # MPEC refinements and I1 bound
    mpec_tasks, merchant_tasks = [], []
    for label, (p, e) in cands.items():
        merchant_tasks.append((label, rho, p, e))
        for inv in rc.INVESTORS:
            p0, e0 = rc.investor_vector(p, e, inv, NODES)
            mpec_tasks.append((label, rho, inv, p, e, "incumbent", p0, e0))
            ok = [r for r in probe_rows if r["candidate"] == label and r["investor"] == inv and r.get("status") == "optimal"]
            if ok:
                best = max(ok, key=lambda r: r["gain"])
                mpec_tasks.append((label, rho, inv, p, e, f"best_probe:{best['probe']}",
                                   {n: best[f"P_{n}"] for n in NODES}, {n: best[f"E_{n}"] for n in NODES}))
    mpec_rows, merchant_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(mpec_task, t) for t in mpec_tasks] + [pool.submit(merchant_task, t) for t in merchant_tasks]
        for fut in as_completed(futures):
            row = fut.result()
            (merchant_rows if "upper_bound" in row or "error" in row and "investor" not in row else mpec_rows).append(row)
            print(f"[{time.perf_counter()-t0:.0f}s]", {k: row[k] for k in list(row)[:8]}, flush=True)
    write_csv(OUT / "mpec_refinements.csv", mpec_rows)
    write_csv(OUT / "merchant_bound_I1.csv", merchant_rows)

    # summary per candidate
    summary = []
    for label in cands:
        base = base_results[(label,)]
        for inv in rc.INVESTORS:
            cur = base["settle"][inv]["profit"]
            pr = [r for r in probe_rows if r["candidate"] == label and r["investor"] == inv and r.get("status") == "optimal"]
            mr = [r for r in mpec_rows if r["candidate"] == label and r["investor"] == inv and finite_num(r.get("recleared_profit"))]
            best_probe = max(pr, key=lambda r: r["gain"]) if pr else None
            best_mpec = max(mr, key=lambda r: r["recleared_profit"]) if mr else None
            gains = [best_probe["gain"] if best_probe else -math.inf, (best_mpec["recleared_profit"] - cur) if best_mpec else -math.inf]
            lower = max(0.0, max(gains))
            row = dict(candidate=label, investor=inv, current_profit=cur,
                       best_probe=best_probe["probe"] if best_probe else "", best_probe_gain=best_probe["gain"] if best_probe else math.nan,
                       best_mpec_start=best_mpec["start"] if best_mpec else "",
                       best_mpec_gain=(best_mpec["recleared_profit"] - cur) if best_mpec else math.nan,
                       regret_lower_bound=lower, relative_regret_lower_bound=lower / max(1.0, abs(cur)))
            if inv == "I1":
                mb = next((r for r in merchant_rows if r["candidate"] == label and "upper_bound" in r), None)
                if mb:
                    row["I1_profit_upper_bound"] = mb["upper_bound"]
                    row["regret_upper_bound"] = max(0.0, mb["upper_bound"] - cur)
                    row["relative_regret_upper_bound"] = row["regret_upper_bound"] / max(1.0, abs(cur))
            summary.append(row)
    write_csv(OUT / "candidate_regret_summary.csv", summary)
    print(f"all done {time.perf_counter()-t0:.0f}s")


def finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


if __name__ == "__main__":
    main()
