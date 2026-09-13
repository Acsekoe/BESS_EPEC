"""Undamped Gauss-Seidel best-response dynamics with exact-reclear selection.

Starting from the final profile of init5mw15mwh_jacobi_rho100_d025_s100, the
investors move one at a time (order I3 -> I1 -> I2) to their best found
response, with no damping and no proximal term.  Each response is chosen by
EXACT reclear profit among
  * the incumbent (no move),
  * every structured feasible probe of region_search.direction_probes
    (includes node exit, whole-portfolio exit, entry, transfers, durations),
  * MPEC refinements under strong duality (IPOPT tol 1e-6) and relaxed KKT
    (epsilon 1e-4, tol 1e-4), started from the incumbent, from the best probe
    and, for I1/I2, from a small positive profile near exit.

Each mover's gain at its turn is a finite-search lower bound on its regret
at that profile.  After the last round every investor's response is searched
again at the final dynamics profile without moving (an unrestricted audit of
that profile with the same search).  For I1 the merchant-entry upper bound is
evaluated at every profile.

This is a search for a low-regret region or a best-response cycle, not a
convergence certificate.  Outputs: output/br_dynamics/*.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import regret_common as rc
import capacity_game
import region_search as rs

OUT = rc.OUTPUT_ROOT / "br_dynamics"
NODES = rs.NODES
ORDER = ("I3", "I1", "I2")
SETTINGS = {
    "strong_duality_tol1e-6": dict(lower_level="strong-duality", epsilon=0.0, tolerance=1e-6),
    "relaxed_eps1e-4_tol1e-4": dict(lower_level="relaxed-kkt", epsilon=1e-4, tolerance=1e-4),
}


def mpec_task(args):
    tag, rho, investor_id, power, energy, start_label, sp, se, setting = args
    data = rc.market_data(rho)
    spec = SETTINGS[setting]
    config = rc.game_config(data, epsilon=spec["epsilon"], tolerance=spec["tolerance"],
                            lower_level=spec["lower_level"], max_seconds=240.0)
    reference = rc.game_config(data)
    investor = next(i for i in config.investors if i.investor_id == investor_id)
    tick = time.perf_counter()
    try:
        response = capacity_game.solve_from_start(data, config, investor, power, energy, start_label, sp, se)
    except Exception as exc:
        return dict(tag=tag, investor=investor_id, start=start_label, setting=setting, termination=f"error: {exc}")
    row = dict(tag=tag, investor=investor_id, start=start_label, setting=setting,
               termination=response.outcome.termination, optimal=response.outcome.optimal,
               embedded_profit=response.embedded_profit_eur_per_day, seconds=time.perf_counter() - tick)
    if not response.outcome.has_solution:
        return row
    cp, ce = response.power, response.energy
    full_p, full_e = rc.with_candidate(power, energy, investor_id, cp, ce)
    try:
        rc.check_feasible(data, reference, investor_id, cp, ce, full_p)
    except ValueError as exc:
        row["termination"] += f"; infeasible after clipping: {exc}"
        return row
    res = rc.evaluate_profile(data, reference, full_p, full_e)
    if res.get("status") == "optimal":
        row["recleared_profit"] = res["settle"][investor_id]["profit"]
        row["embedded_minus_reclear"] = row["embedded_profit"] - row["recleared_profit"]
        row["power"] = cp
        row["energy"] = ce
    return row


def search_response(pool, rho, profile, investor_id, tag, limits, data, config):
    power, energy = profile
    p0, e0 = rc.investor_vector(power, energy, investor_id, NODES)
    caps = rs.residual_caps(power, investor_id, limits)
    base = rc.evaluate_profile(data, config, power, energy)
    current = base["settle"][investor_id]["profit"]
    probes = rs.direction_probes(p0, e0, caps)
    tasks, meta = [], {}
    for k, (family, label, step, cp, ce) in enumerate(probes):
        full_p, full_e = rc.with_candidate(power, energy, investor_id, cp, ce)
        try:
            rc.check_feasible(data, config, investor_id, cp, ce, full_p)
        except ValueError:
            continue
        meta[k] = (family, label, cp, ce)
        tasks.append((k, rho, full_p, full_e))
    results = rc.evaluate_many(pool, tasks)
    candidates = [dict(source="incumbent", label="incumbent", profit=current, power=p0, energy=e0)]
    for k, res in results.items():
        if res.get("status") == "optimal":
            family, label, cp, ce = meta[k]
            candidates.append(dict(source=f"probe:{family}", label=label, profit=res["settle"][investor_id]["profit"],
                                   power=cp, energy=ce))
    best_probe = max(candidates, key=lambda c: c["profit"])
    starts = [("incumbent", p0, e0), (f"best_probe:{best_probe['label']}", best_probe["power"], best_probe["energy"])]
    if investor_id in ("I1", "I2"):
        starts.append(("near_exit_1mw", {n: (1.0 if p0[n] > 1e-3 else 0.0) for n in NODES},
                       {n: (3.0 if p0[n] > 1e-3 else 0.0) for n in NODES}))
    futures = [pool.submit(mpec_task, (tag, rho, investor_id, power, energy, label, sp, se, setting))
               for label, sp, se in starts for setting in SETTINGS]
    mpec_rows = [f.result() for f in futures]
    for row in mpec_rows:
        if "recleared_profit" in row:
            candidates.append(dict(source=f"mpec:{row['setting']}", label=row["start"], profit=row["recleared_profit"],
                                   power=row["power"], energy=row["energy"], embedded_minus_reclear=row["embedded_minus_reclear"]))
    best = max(candidates, key=lambda c: c["profit"])
    return dict(current=current, best=best, best_probe=best_probe, candidates=candidates, mpec_rows=mpec_rows, base=base)


def profile_row(tag, profile, res, data):
    static = rc.static_arrays(data)
    sets = rc.active_sets(res, static)
    midday = [static["times"].index(t) for t in static["times"] if 11 <= t <= 16]
    row = dict(tag=tag, lines_priced=";".join(sorted(sets["lines_priced"])),
               midday_mean_lmp_N6=float(np.mean(res["lmp"][NODES.index("N6"), midday])),
               midday_mean_lmp_N8=float(np.mean(res["lmp"][NODES.index("N8"), midday])),
               total_abs_da_mwh=res["total_abs_da_mwh"])
    for inv in rc.INVESTORS:
        row[f"profit_{inv}"] = res["settle"][inv]["profit"]
        row[f"total_power_{inv}"] = sum(profile[0][inv, n] for n in NODES)
        row[f"total_energy_{inv}"] = sum(profile[1][inv, n] for n in NODES)
        for n in NODES:
            row[f"P_{inv}_{n}"] = profile[0][inv, n]
        for n in NODES:
            row[f"E_{inv}_{n}"] = profile[1][inv, n]
    return row


def write(name, rows):
    if rows:
        with (OUT / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rho", type=float, default=100.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rho = args.rho
    data = rc.market_data(rho)
    config = rc.game_config(data)
    limits = config.node_limits(data)
    profile = rc.incumbent()
    step_rows, profile_rows, candidate_rows, mpec_log, bound_rows = [], [], [], [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=rc._init_worker) as pool:
        start_res = rc.evaluate_profile(data, config, *profile)
        profile_rows.append(profile_row("step0_final_sweep100", profile, start_res, data))
        step = 0
        for rnd in range(1, args.rounds + 1):
            for inv in ORDER:
                step += 1
                tag = f"step{step}_round{rnd}_{inv}"
                out = search_response(pool, rho, profile, inv, tag, limits, data, config)
                best = out["best"]
                gain = best["profit"] - out["current"]
                bound = rs.merchant_bound(rho, *profile) if inv == "I1" else None
                step_rows.append(dict(step=step, round=rnd, investor=inv, current_profit=out["current"],
                                      best_profit=best["profit"], gain=gain,
                                      relative_gain=gain / max(1.0, abs(out["current"])),
                                      best_source=best["source"], best_label=best["label"],
                                      best_probe_label=out["best_probe"]["label"],
                                      best_probe_gain=out["best_probe"]["profit"] - out["current"],
                                      best_total_power=sum(best["power"].values()),
                                      best_total_energy=sum(best["energy"].values()),
                                      I1_profit_upper_bound=bound["upper_bound"] if bound else "",
                                      seconds=time.perf_counter() - t0))
                for c in out["candidates"]:
                    if c["source"] != "incumbent" and not c["source"].startswith("probe"):
                        candidate_rows.append(dict(tag=tag, investor=inv, source=c["source"], label=c["label"],
                                                   profit=c["profit"], gain=c["profit"] - out["current"],
                                                   embedded_minus_reclear=c.get("embedded_minus_reclear", "")))
                for r in out["mpec_rows"]:
                    mpec_log.append({k: v for k, v in r.items() if k not in ("power", "energy")})
                if bound:
                    bound_rows.append(dict(tag=tag, **{k: v for k, v in bound.items() if not isinstance(v, dict)}))
                print(f"[{time.perf_counter()-t0:6.0f}s] {tag}: current={out['current']:.1f} best={best['profit']:.1f}"
                      f" gain={gain:+.1f} via {best['source']}:{best['label']}"
                      + (f" I1-bound={bound['upper_bound']:.1f}" if bound else ""), flush=True)
                if gain > 1e-6:
                    profile = rc.with_candidate(*profile, inv, best["power"], best["energy"])
                res = rc.evaluate_profile(data, config, *profile)
                profile_rows.append(profile_row(tag, profile, res, data))
                write("steps.csv", step_rows)
                write("profiles.csv", profile_rows)
                write("response_candidates.csv", candidate_rows)
                write("mpec_log.csv", mpec_log)
                write("I1_merchant_bounds.csv", bound_rows)
        # unrestricted audit at the final dynamics profile (no moves)
        audit_rows = []
        for inv in rc.INVESTORS:
            out = search_response(pool, rho, profile, inv, f"final_audit_{inv}", limits, data, config)
            gain = out["best"]["profit"] - out["current"]
            bound = rs.merchant_bound(rho, *profile) if inv == "I1" else None
            audit_rows.append(dict(investor=inv, current_profit=out["current"], best_profit=out["best"]["profit"],
                                   regret_lower_bound=max(0.0, gain), relative_regret_lower_bound=max(0.0, gain) / max(1.0, abs(out["current"])),
                                   best_source=out["best"]["source"], best_label=out["best"]["label"],
                                   I1_profit_upper_bound=bound["upper_bound"] if bound else "",
                                   I1_regret_upper_bound=(max(0.0, bound["upper_bound"] - out["current"]) if bound else "")))
            for r in out["mpec_rows"]:
                mpec_log.append({k: v for k, v in r.items() if k not in ("power", "energy")})
            print(f"[{time.perf_counter()-t0:6.0f}s] final audit {inv}: current={out['current']:.1f} best={out['best']['profit']:.1f}"
                  f" gain={gain:+.1f} via {out['best']['source']}:{out['best']['label']}"
                  + (f" I1-bound={bound['upper_bound']:.1f}" if bound else ""), flush=True)
        write("final_profile_audit.csv", audit_rows)
        write("mpec_log.csv", mpec_log)
    print(f"done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
