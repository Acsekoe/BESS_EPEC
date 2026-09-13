"""Is the procedural reclear an exact ISO optimum at the audited deviations?

For each audited response profile this script
  * re-evaluates the maintained reclear and decomposes its KKT residuals by
    constraint block,
  * re-solves the identical QP with alternative IPOPT settings and with HiGHS,
  * compares objective, LMPs and every investor's settlement.

Output: output/reclear_accuracy/reclear_accuracy.csv and .json
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

import regret_common as rc
import capacity_game
import iso_market
import solvers

OUT = rc.OUTPUT_ROOT / "reclear_accuracy"


def block_residuals(res: dict, data, config) -> dict:
    """Largest complementarity product per constraint block, with its index."""

    N, T, L, G = list(data.nodes), list(data.times), list(data.lines), list(data.generators)
    units = list(config.investor_ids)
    eta = data.eta
    deg = np.array([config.degradation()[i] for i in units])
    lmp, nu, gen = res["lmp"], res["nu"], res["gen"]
    cap = np.array([[data.generation_capacity[g, t] for t in T] for g in G])
    offer = np.array([data.offer(g) for g in G])
    gnode = np.array([N.index(data.nodes_by_generator()[g][0]) for g in G])
    limit = np.array([data.line_limit[l] for l in L])
    P, E = res["power"], res["energy"]
    ch, dis, soc = res["charge"], res["discharge"], res["soc"]
    rho_ch, sig, dsoc = res["rho_ch"], res["sig_dis"], res["del_soc"]
    # gam is not stored; recover from charge stationarity is ambiguous, so use products that do not need it
    blocks = {
        "gen_upper": (cap - gen) * (-nu),
        "line_upper": (limit[:, None] - res["flow"]) * (-res["mu_up"]),
        "line_lower": (res["flow"] + limit[:, None]) * res["mu_dn"],
        "charge_upper": (P[:, :, None] - ch) * (-rho_ch),
        "discharge_upper": (P[:, :, None] - dis) * (-sig),
        "soc_upper": (E[:, :, None] - soc) * (-dsoc),
    }
    out = {}
    for name, arr in blocks.items():
        idx = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
        out[name] = {"max_abs": float(np.abs(arr).max()), "sum_abs": float(np.abs(arr).sum()), "argmax": [int(i) for i in idx]}
    # attach interpretable details for the storage blocks
    for name, dual_arr in (("charge_upper", rho_ch), ("discharge_upper", sig)):
        i, n, t = out[name]["argmax"]
        out[name]["where"] = {"unit": units[i], "node": N[n], "hour": T[t], "power": float(P[i, n]),
                              "flow": float((ch if name == "charge_upper" else dis)[i, n, t]),
                              "dual": float(dual_arr[i, n, t])}
    i, n, s = out["soc_upper"]["argmax"]
    out["soc_upper"]["where"] = {"unit": units[i], "node": N[n], "soc_time": int(s), "energy": float(E[i, n]),
                                 "soc": float(soc[i, n, s]), "dual": float(dsoc[i, n, s])}
    return out


def solve_variant(data, config, power, energy, variant: str, logdir: Path) -> dict:
    market = iso_market.build_market(data, power, energy, config.degradation())
    tick = time.perf_counter()
    if variant.startswith("ipopt"):
        options = {
            "linear_solver": config.solver.linear_solver,
            "max_iter": config.solver.max_iterations,
            "max_cpu_time": config.solver.max_seconds,
            "tol": min(config.solver.tolerance, 1e-11),
            "nlp_scaling_method": "none",
            "acceptable_tol": 1e-10,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",
            "warm_start_init_point": "no",
            "print_level": 5,
        }
        if variant == "ipopt_ma27":
            options["linear_solver"] = "ma27"
        elif variant == "ipopt_mumps":
            options["linear_solver"] = "mumps"
        elif variant == "ipopt_no_acceptable":
            options["acceptable_iter"] = 0
            options["tol"] = 1e-12
        elif variant == "ipopt_gradient_scaling":
            options["nlp_scaling_method"] = "gradient-based"
        elif variant == "ipopt_bound_push":
            options["bound_push"] = 1e-10
            options["bound_frac"] = 1e-10
            options["mu_init"] = 1e-6
        solver = solvers._ipopt(config.solver, options)
        log = logdir / f"{variant}.log"
        result = solver.solve(market, tee=False, logfile=str(log))
        status = str(result.solver.termination_condition)
    elif variant == "highs":
        solver = pyo.SolverFactory("appsi_highs")
        solver.config.load_solution = True
        result = solver.solve(market)
        status = str(result.solver.termination_condition)
    else:
        raise ValueError(variant)
    seconds = time.perf_counter() - tick
    try:
        res = rc.extract_market(market, data, config, power, energy)
    except Exception as exc:
        return {"variant": variant, "status": f"{status}; extract failed: {exc}", "seconds": seconds}
    res["status"] = status
    res["seconds"] = seconds
    res["variant"] = variant
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = rc.market_data(100.0)
    config = rc.game_config(data)
    power, energy = rc.incumbent()
    profiles = {"incumbent": (power, energy)}
    for inv in rc.INVESTORS:
        for rec in rc.audit_records()[inv]:
            if rec["phase"] == "refine":
                label = f"{inv}:{rec['start']}"
                profiles[label] = rc.with_candidate(power, energy, inv, rec["power"], rec["energy"])

    variants = ["ipopt_reference", "ipopt_ma27", "ipopt_mumps", "ipopt_no_acceptable",
                "ipopt_gradient_scaling", "ipopt_bound_push", "highs"]
    rows, blocks = [], {}
    for label, (p, e) in profiles.items():
        reference = None
        for variant in variants:
            logdir = OUT / "logs" / label.replace(":", "_")
            logdir.mkdir(parents=True, exist_ok=True)
            try:
                res = solve_variant(data, config, p, e, variant, logdir)
            except Exception as exc:
                rows.append({"profile": label, "variant": variant, "status": f"error: {exc}"})
                continue
            if "lmp" not in res:
                rows.append({"profile": label, "variant": variant, "status": res["status"]})
                continue
            if variant == "ipopt_reference":
                reference = res
                blocks[label] = block_residuals(res, data, config)
            row = {
                "profile": label,
                "variant": variant,
                "status": res["status"],
                "seconds": round(res["seconds"], 3),
                "objective": res["objective"],
                "pd_gap": res["pd_gap"],
                "max_primal_violation": res["max_primal_violation"],
                "max_dual_infeasibility": res["max_dual_infeasibility"],
                "max_stationarity_residual": res["max_stationarity_residual"],
                "max_complementarity": res["max_complementarity"],
                "sum_abs_complementarity": res["sum_abs_complementarity"],
            }
            for inv in rc.INVESTORS:
                s = res["settle"][inv]
                row[f"profit_{inv}"] = s["profit"]
                row[f"storage_surplus_{inv}"] = s["storage_operating_surplus"]
                row[f"gen_rent_{inv}"] = s["owned_generation_rent"]
            if reference is not None:
                row["objective_minus_reference"] = res["objective"] - reference["objective"]
                row["max_abs_lmp_diff_vs_reference"] = float(np.max(np.abs(res["lmp"] - reference["lmp"])))
                for inv in rc.INVESTORS:
                    row[f"profit_{inv}_minus_reference"] = res["settle"][inv]["profit"] - reference["settle"][inv]["profit"]
            rows.append(row)
            print(label, variant, row.get("status"), f"obj={row.get('objective', float('nan')):.6f}",
                  f"pd_gap={row.get('pd_gap', float('nan')):.3e}", f"comp={row.get('max_complementarity', float('nan')):.2e}",
                  f"dLMP={row.get('max_abs_lmp_diff_vs_reference', 0):.3e}",
                  " ".join(f"{inv}:{row.get(f'profit_{inv}_minus_reference', 0):+.3f}" for inv in rc.INVESTORS), flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("profile", "variant", "status"), k))
    with (OUT / "reclear_accuracy.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "reference_block_residuals.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
    for label, b in blocks.items():
        worst = max(b.items(), key=lambda kv: kv[1]["max_abs"])
        print("BLOCK", label, worst[0], json.dumps(worst[1]))


if __name__ == "__main__":
    main()
