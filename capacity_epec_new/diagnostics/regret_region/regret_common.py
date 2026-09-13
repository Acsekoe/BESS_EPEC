"""Shared, read-only helpers for the regret-region diagnostics.

Nothing here edits the maintained model.  Every market clear, settlement and
MPEC is built by the modules in ``model/`` and only called and recorded here:

* ``capacity_game.clear`` / ``iso_market.settle`` for the exact procedural
  reclear and settlement, exactly as the final audit uses them;
* ``capacity_game.build_best_response_model`` / ``solvers.solve_mpec`` for
  best responses.

On top of that this module extracts the complete cleared state (prices,
dispatch, duals, flows), evaluates the ISO's KKT residuals from the returned
duals, and classifies active sets, so that every sampled point can be saved and
compared.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL_DIR = PROJECT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import pyomo.environ as pyo  # noqa: E402

import capacity_game  # noqa: E402
import iso_market  # noqa: E402
import mpec  # noqa: E402
import solvers  # noqa: E402
from capacity_game import GameConfig  # noqa: E402
from investors import three_investors  # noqa: E402
from market_data import load_market_data  # noqa: E402

RUN_DIR = MODEL_DIR / "output" / "init5mw15mwh_jacobi_rho100_d025_s100"
DATA_PATH = MODEL_DIR / "input" / "market_data_smoothed.json"
OUTPUT_ROOT = HERE / "output"
INVESTORS = ("I1", "I2", "I3")

# Active-set classification thresholds (MW for slacks, EUR/MWh for duals).
SLACK_TOL = 1.0e-6
DUAL_TOL = 1.0e-6


# --------------------------------------------------------------------------
# Inputs


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_config() -> dict:
    return json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))


def market_data(rho: float):
    """The maintained input with the effective demand-adjustment penalty."""

    data = load_market_data(DATA_PATH)
    return replace(data, demand_adjustment_penalty_eur_per_mw2=float(rho))


def game_config(
    data,
    *,
    epsilon: float | None = None,
    tolerance: float | None = None,
    lower_level: str | None = None,
    max_seconds: float | None = None,
) -> GameConfig:
    """The run's GameConfig, optionally with one numerical setting changed."""

    rc = run_config()
    s = rc["solver"]
    solver = solvers.SolverSettings(
        linear_solver=s["linear_solver"],
        max_iterations=int(s["max_iterations"]),
        max_seconds=float(s["max_seconds"] if max_seconds is None else max_seconds),
        tolerance=float(s["tolerance"] if tolerance is None else tolerance),
    )
    config = GameConfig(
        investors=three_investors(data),
        lower_level=rc["lower_level"] if lower_level is None else lower_level,
        complementarity_epsilon=float(
            rc["complementarity_epsilon"] if epsilon is None else epsilon
        ),
        market_design=rc["market_design"],
        cleanup_tolerance=float(rc["cleanup_tolerance"]),
        proximal_penalty=0.0,
        uniform_node_connection_limit_mw=rc["uniform_node_connection_limit_mw"],
        initial_ratio_hours=float(rc["initial_ratio_hours"]),
        price_bound=float(rc["price_bound"]),
        dual_bound=float(rc["dual_bound"]),
        sparse_capacity_tol=float(rc["sparse_capacity_tol"]),
        parallel_workers=1,
        solver=solver,
    )
    expected = rc["investors"]
    actual = [
        {**asdict(i), "owned_generation_shares": dict(i.owned_generation_shares)}
        for i in config.investors
    ]
    if expected != actual:
        raise RuntimeError("Investor population differs from the run configuration.")
    return config


def read_capacities(path: Path) -> tuple[dict, dict]:
    power: dict[tuple[str, str], float] = {}
    energy: dict[tuple[str, str], float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            power[row["investor"], row["node"]] = float(row["power_mw"])
            energy[row["investor"], row["node"]] = float(row["energy_mwh"])
    return power, energy


def incumbent() -> tuple[dict, dict]:
    """Final profile of the run, cross-checked against its full-precision checkpoint."""

    power, energy = read_capacities(RUN_DIR / "final_capacities.csv")
    checkpoint = json.loads((RUN_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    for row in checkpoint["power"]:
        if power[row["investor"], row["node"]] != float(row["mw"]):
            raise RuntimeError("final_capacities.csv and checkpoint.json disagree (power).")
    for row in checkpoint["energy"]:
        if energy[row["investor"], row["node"]] != float(row["mwh"]):
            raise RuntimeError("final_capacities.csv and checkpoint.json disagree (energy).")
    return power, energy


def audit_records() -> dict:
    return json.loads((RUN_DIR / "starts_final_audit.json").read_text(encoding="utf-8"))


def audit_record(investor: str, start: str, phase: str) -> dict:
    for record in audit_records()[investor]:
        if record["start"] == start and record["phase"] == phase:
            return record
    raise KeyError((investor, start, phase))


def selected_record(investor: str) -> dict:
    return next(r for r in audit_records()[investor] if r.get("selected"))


def with_candidate(power, energy, investor, cand_power, cand_energy):
    merged_p, merged_e = dict(power), dict(energy)
    for node, value in cand_power.items():
        merged_p[investor, node] = float(value)
    for node, value in cand_energy.items():
        merged_e[investor, node] = float(value)
    return merged_p, merged_e


def investor_vector(power, energy, investor, nodes) -> tuple[dict, dict]:
    return (
        {n: float(power[investor, n]) for n in nodes},
        {n: float(energy[investor, n]) for n in nodes},
    )


def check_feasible(data, config, investor_id, cand_power, cand_energy, full_power) -> None:
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    capacity_game._check_capacity_feasible(
        data, config, investor, cand_power, cand_energy, full_power
    )


# --------------------------------------------------------------------------
# Exact market: clear, extract, residuals


INEXACT_PD_GAP = 1.0e-3
INEXACT_COMPLEMENTARITY = 1.0e-5


def resolve_market(data, config, power, energy, linear_solver: str = "ma27"):
    """The maintained QP and clearing options with only the linear solver changed.

    Used purely as a verification of the procedural reclear, never as the
    payoff: sub-microwatt 'ghost' capacities left by interior-point best
    responses make the reference solve accept points whose unscaled KKT
    residual is large although IPOPT's scaled error is below tolerance.
    """

    market = iso_market.build_market(data, power, energy, config.degradation())
    solver = solvers._ipopt(
        config.solver,
        {
            "linear_solver": linear_solver,
            "max_iter": config.solver.max_iterations,
            "max_cpu_time": config.solver.max_seconds,
            "tol": min(config.solver.tolerance, 1.0e-11),
            "nlp_scaling_method": "none",
            "acceptable_tol": 1.0e-10,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "warm_start_init_point": "no",
            "print_level": 0,
        },
    )
    result = solver.solve(market, tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"verification clear failed: {result.solver.termination_condition}")
    return market


def evaluate_profile(
    data, config, power, energy, *, repeat_check: bool = False, verify: bool = True
) -> dict:
    """Clear and settle one full profile with the maintained procedural rule.

    Returns scalars plus the full cleared state as numpy arrays.  An exception
    in clearing is recorded as a status, never swallowed into a number.  When
    the reference clear's own KKT residuals show it is inexact, the identical
    QP is re-solved with MA27 and the resulting price and profit differences
    are recorded; the reported payoff remains the maintained reclear.
    """

    tick = time.perf_counter()
    try:
        market = capacity_game.clear(data, config, power, energy)
    except Exception as exc:  # recorded, not hidden
        return {"status": f"error: {exc}", "seconds": time.perf_counter() - tick}
    result = extract_market(market, data, config, power, energy)
    result["status"] = "optimal"
    result["seconds"] = time.perf_counter() - tick
    inexact = (
        abs(result["pd_gap"]) > INEXACT_PD_GAP
        or result["max_complementarity"] > INEXACT_COMPLEMENTARITY
    )
    result["reference_inexact"] = bool(inexact)
    result["verify_status"] = "not_needed"
    if verify and inexact:
        try:
            alt = resolve_market(data, config, power, energy)
            second = extract_market(alt, data, config, power, energy)
            result["verify_status"] = "ma27_optimal"
            result["verify_pd_gap"] = second["pd_gap"]
            result["verify_max_complementarity"] = second["max_complementarity"]
            result["verify_max_abs_lmp_diff"] = float(np.max(np.abs(second["lmp"] - result["lmp"])))
            diffs = {
                i: second["settle"][i]["profit"] - result["settle"][i]["profit"]
                for i in result["settle"]
            }
            result["verify_profit_diff"] = diffs
            result["verify_max_abs_profit_diff"] = max(abs(v) for v in diffs.values())
        except Exception as exc:
            result["verify_status"] = f"failed: {exc}"
    if repeat_check:
        again = capacity_game.clear(data, config, power, energy)
        second = extract_market(again, data, config, power, energy)
        result["repeat_max_abs_lmp_diff"] = float(np.max(np.abs(second["lmp"] - result["lmp"])))
        result["repeat_max_abs_profit_diff"] = max(
            abs(second["settle"][i]["profit"] - result["settle"][i]["profit"])
            for i in result["settle"]
        )
    return result


def extract_market(market, data, config, power, energy) -> dict:
    N = list(data.nodes)
    T = list(data.times)
    S = list(data.soc_times)
    L = list(data.lines)
    G = list(data.generators)
    units = list(config.investor_ids)
    dual = market.dual
    rho = float(data.demand_adjustment_penalty_eur_per_mw2)
    eta = float(data.eta)
    deg = config.degradation()

    def value(component) -> float:
        return float(pyo.value(component))

    lmp = np.array([[float(dual[market.nodal_balance[n, t]]) for t in T] for n in N])
    lam_sys = np.array([float(dual[market.system_balance[t]]) for t in T])
    gen = np.array([[value(market.P_gen[g, t]) for t in T] for g in G])
    nu = np.array([[float(dual[market.generation_capacity_bound[g, t]]) for t in T] for g in G])
    cap = np.array([[float(data.generation_capacity[g, t]) for t in T] for g in G])
    ch = np.array([[[value(market.P_charge[i, n, t]) for t in T] for n in N] for i in units])
    dis = np.array([[[value(market.P_discharge[i, n, t]) for t in T] for n in N] for i in units])
    soc = np.array([[[value(market.SOC[i, n, s]) for s in S] for n in N] for i in units])
    rho_ch = np.array(
        [[[float(dual[market.charge_power_bound[i, n, t]]) for t in T] for n in N] for i in units]
    )
    sig = np.array(
        [[[float(dual[market.discharge_power_bound[i, n, t]]) for t in T] for n in N] for i in units]
    )
    gam = np.array(
        [[[float(dual[market.soc_transition[i, n, t]]) for t in T] for n in N] for i in units]
    )
    dsoc = np.array(
        [[[float(dual[market.soc_capacity_bound[i, n, s]]) for s in S] for n in N] for i in units]
    )
    rper = np.array([[float(dual[market.soc_periodicity[i, n]]) for n in N] for i in units])
    da = np.array([[value(market.DemandAdjustment[n, t]) for t in T] for n in N])
    ni = np.array([[value(market.NetInjection[n, t]) for t in T] for n in N])
    mu_up = np.array([[float(dual[market.line_upper_bound[l, t]]) for t in T] for l in L])
    mu_dn = np.array([[float(dual[market.line_lower_bound[l, t]]) for t in T] for l in L])
    ptdf = np.array([[float(data.ptdf[l, n]) for n in N] for l in L])
    flow = ptdf @ ni
    limit = np.array([float(data.line_limit[l]) for l in L])
    demand = np.array([[float(data.demand_el[n, t]) for t in T] for n in N])
    P = np.array([[float(power[i, n]) for n in N] for i in units])
    E = np.array([[float(energy[i, n]) for n in N] for i in units])
    offer = np.array([data.offer(g) for g in G])
    gen_node = {g: nodes[0] for g, nodes in data.nodes_by_generator().items()}
    gnode = np.array([N.index(gen_node[g]) for g in G])
    degv = np.array([float(deg[i]) for i in units])

    objective = value(market.objective)
    dual_objective = float(
        np.sum(demand * lmp)
        + np.sum(cap * nu)
        + np.sum(limit[:, None] * (mu_up - mu_dn))
        + np.sum(P[:, :, None] * (rho_ch + sig))
        + np.sum(E[:, :, None] * dsoc)
        - 0.5 * rho * np.sum(da**2)
    )

    # Reduced costs of the ISO QP, using the maintained MPEC's sign convention.
    rc_gen = offer[:, None] - lmp[gnode, :] - nu
    rc_ch = 0.5 * degv[:, None, None] + lmp[None, :, :] - rho_ch + eta * gam
    rc_dis = 0.5 * degv[:, None, None] - lmp[None, :, :] - sig - gam / eta
    stat_soc = dsoc.copy()
    stat_soc[:, :, 1:] += gam  # tau in T
    stat_soc[:, :, :-1] -= gam  # tau + 1 in T
    stat_soc[:, :, 0] += rper
    stat_soc[:, :, -1] -= rper
    rc_soc = -stat_soc
    stat_ni = -lmp + lam_sys[None, :] + ptdf.T @ (mu_up + mu_dn)
    adjustable = np.array(
        [[data.demand_is_adjustable(n, t) for t in T] for n in N], dtype=bool
    )
    stat_da = np.where(adjustable, rho * da - lmp, 0.0)

    comps = [
        gen * rc_gen,
        (cap - gen) * (-nu),
        (limit[:, None] - flow) * (-mu_up),
        (flow + limit[:, None]) * mu_dn,
        ch * rc_ch,
        (P[:, :, None] - ch) * (-rho_ch),
        dis * rc_dis,
        (P[:, :, None] - dis) * (-sig),
        soc * rc_soc,
        (E[:, :, None] - soc) * (-dsoc),
    ]
    dual_infeasibility = max(
        float(np.max(-rc_gen)),
        float(np.max(-rc_ch)),
        float(np.max(-rc_dis)),
        float(np.max(-rc_soc)),
        float(np.max(nu)),
        float(np.max(mu_up)),
        float(np.max(-mu_dn)),
        float(np.max(rho_ch)),
        float(np.max(sig)),
        float(np.max(dsoc)),
        0.0,
    )
    settle = {
        inv.investor_id: asdict(iso_market.settle(market, data, inv, power, energy))
        for inv in config.investors
    }
    return {
        "objective": objective,
        "dual_objective": dual_objective,
        "pd_gap": objective - dual_objective,
        "max_primal_violation": float(solvers.maximum_bound_violation(market)),
        "max_dual_infeasibility": dual_infeasibility,
        "max_stationarity_residual": max(
            float(np.max(np.abs(stat_ni))), float(np.max(np.abs(stat_da)))
        ),
        "max_complementarity": max(float(np.max(np.abs(c))) for c in comps),
        "sum_abs_complementarity": float(sum(np.sum(np.abs(c)) for c in comps)),
        "settle": settle,
        "lmp": lmp,
        "lam_sys": lam_sys,
        "gen": gen,
        "nu": nu,
        "charge": ch,
        "discharge": dis,
        "soc": soc,
        "rho_ch": rho_ch,
        "sig_dis": sig,
        "del_soc": dsoc,
        "da": da,
        "ni": ni,
        "flow": flow,
        "mu_up": mu_up,
        "mu_dn": mu_dn,
        "power": P,
        "energy": E,
        "total_abs_da_mwh": float(np.sum(np.abs(da))),
        "max_abs_da_mw": float(np.max(np.abs(da))),
    }


# --------------------------------------------------------------------------
# Active sets


def static_arrays(data) -> dict:
    T = list(data.times)
    G = list(data.generators)
    L = list(data.lines)
    return {
        "cap": np.array([[float(data.generation_capacity[g, t]) for t in T] for g in G]),
        "limit": np.array([float(data.line_limit[l]) for l in L]),
        "nodes": list(data.nodes),
        "times": T,
        "lines": L,
        "gens": G,
        "demand": np.array([[float(data.demand_el[n, t]) for t in T] for n in data.nodes]),
    }


def active_sets(res: dict, static: dict, units=INVESTORS) -> dict[str, frozenset]:
    """Regime descriptors of one cleared market.

    ``lines_priced`` (nonzero congestion multiplier) is the price-forming set;
    ``lines_binding`` adds physically binding lines whose multiplier is zero.
    """

    N, T, L, G = static["nodes"], static["times"], static["lines"], static["gens"]
    limit, cap = static["limit"], static["cap"]
    flow, mu_up, mu_dn = res["flow"], res["mu_up"], res["mu_dn"]
    lines_binding, lines_priced = set(), set()
    for a, l in enumerate(L):
        for b, t in enumerate(T):
            if limit[a] - flow[a, b] <= SLACK_TOL:
                lines_binding.add(f"{l}+@{t}")
            if flow[a, b] + limit[a] <= SLACK_TOL:
                lines_binding.add(f"{l}-@{t}")
            if -mu_up[a, b] > DUAL_TOL:
                lines_priced.add(f"{l}+@{t}")
            if mu_dn[a, b] > DUAL_TOL:
                lines_priced.add(f"{l}-@{t}")
    gen_state = set()
    gen = res["gen"]
    for a, g in enumerate(G):
        for b, t in enumerate(T):
            if cap[a, b] <= SLACK_TOL:
                continue
            if cap[a, b] - gen[a, b] <= SLACK_TOL:
                gen_state.add(f"{g}:cap@{t}")
            elif gen[a, b] <= SLACK_TOL:
                gen_state.add(f"{g}:off@{t}")
            else:
                gen_state.add(f"{g}:marg@{t}")
    storage_state = set()
    P, E = res["power"], res["energy"]
    ch, dis, soc = res["charge"], res["discharge"], res["soc"]
    for a, i in enumerate(units):
        for k, n in enumerate(N):
            if P[a, k] <= 1.0e-8:
                continue
            for b, t in enumerate(T):
                if P[a, k] - ch[a, k, b] <= SLACK_TOL:
                    storage_state.add(f"{i}.{n}:chmax@{t}")
                elif ch[a, k, b] > SLACK_TOL:
                    storage_state.add(f"{i}.{n}:ch@{t}")
                if P[a, k] - dis[a, k, b] <= SLACK_TOL:
                    storage_state.add(f"{i}.{n}:dismax@{t}")
                elif dis[a, k, b] > SLACK_TOL:
                    storage_state.add(f"{i}.{n}:dis@{t}")
            for b in range(soc.shape[2]):
                if E[a, k] - soc[a, k, b] <= SLACK_TOL:
                    storage_state.add(f"{i}.{n}:socfull@{b}")
                elif soc[a, k, b] <= SLACK_TOL:
                    storage_state.add(f"{i}.{n}:socempty@{b}")
    return {
        "lines_binding": frozenset(lines_binding),
        "lines_priced": frozenset(lines_priced),
        "generation": frozenset(gen_state),
        "storage": frozenset(storage_state),
    }


def signature_hash(items: frozenset) -> str:
    return hashlib.sha1("|".join(sorted(items)).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# Parallel evaluation


_WORKER: dict = {}


def _init_worker() -> None:
    _WORKER["base"] = load_market_data(DATA_PATH)
    _WORKER["by_rho"] = {}


def _worker_state(rho: float):
    cache = _WORKER["by_rho"]
    key = float(rho)
    if key not in cache:
        data = replace(_WORKER["base"], demand_adjustment_penalty_eur_per_mw2=key)
        cache[key] = (data, game_config(data))
    return cache[key]


def _evaluate_task(task):
    key, rho, power, energy, repeat_check = task
    data, config = _worker_state(rho)
    return key, evaluate_profile(data, config, power, energy, repeat_check=repeat_check)


def make_pool(workers: int = 12) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)


def evaluate_many(pool, tasks) -> dict:
    """tasks: iterable of (key, rho, power, energy[, repeat_check])."""

    normalised = [t if len(t) == 5 else (*t, False) for t in tasks]
    return dict(pool.map(_evaluate_task, normalised, chunksize=1))


def json_default(item):
    if isinstance(item, (np.floating,)):
        return float(item)
    if isinstance(item, (np.integer,)):
        return int(item)
    if isinstance(item, frozenset):
        return sorted(item)
    raise TypeError(type(item))


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
