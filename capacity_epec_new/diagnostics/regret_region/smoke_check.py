"""Reproduce the audited profits and validate residual sign conventions."""

from __future__ import annotations

import json
import time

import numpy as np

import regret_common as rc


def main() -> None:
    cfg = rc.run_config()
    assert rc.sha256(rc.DATA_PATH) == cfg["data_sha256"], "input hash differs from the run"
    data = rc.market_data(100.0)
    config = rc.game_config(data)
    power, energy = rc.incumbent()
    audit = {row: None for row in rc.INVESTORS}

    t0 = time.perf_counter()
    base = rc.evaluate_profile(data, config, power, energy, repeat_check=True)
    print("incumbent clear seconds", round(time.perf_counter() - t0, 2), base["status"])
    for key in ("objective", "dual_objective", "pd_gap", "max_primal_violation",
                "max_dual_infeasibility", "max_stationarity_residual", "max_complementarity",
                "sum_abs_complementarity", "repeat_max_abs_lmp_diff", "repeat_max_abs_profit_diff",
                "total_abs_da_mwh", "max_abs_da_mw"):
        print(f"  {key}: {base[key]:.6g}")
    expected_current = {"I1": 1920.9181763696738, "I2": 73564.62737151793, "I3": 90099.78858565052}
    for inv, exp in expected_current.items():
        print(f"  {inv} profit {base['settle'][inv]['profit']:.6f} audit {exp:.6f} diff {base['settle'][inv]['profit']-exp:.3e}")

    expected_dev = {"I1": 3572.6818051264654, "I2": 75764.69305746083, "I3": 125941.72525686698}
    for inv in rc.INVESTORS:
        rec = rc.selected_record(inv)
        p, e = rc.with_candidate(power, energy, inv, rec["power"], rec["energy"])
        res = rc.evaluate_profile(data, config, p, e)
        print(f"  {inv} selected deviation {res['settle'][inv]['profit']:.6f} audit {expected_dev[inv]:.6f}"
              f" diff {res['settle'][inv]['profit']-expected_dev[inv]:.3e}; pd_gap {res['pd_gap']:.3e};"
              f" dual_inf {res['max_dual_infeasibility']:.2e}; comp {res['max_complementarity']:.2e};"
              f" max dLMP vs incumbent {np.max(np.abs(res['lmp']-base['lmp'])):.4f}")

    static = rc.static_arrays(data)
    sets = rc.active_sets(base, static)
    print({k: len(v) for k, v in sets.items()})
    print("priced lines:", sorted(sets["lines_priced"])[:40])
    loaded = [n for n in data.nodes if any(data.demand_el[n, t] > 0 for t in data.times)]
    print("loaded nodes:", loaded)

    try:
        import pyomo.environ as pyo
        print("appsi_highs available:", pyo.SolverFactory("appsi_highs").available(exception_flag=False))
    except Exception as exc:
        print("appsi_highs check failed", exc)


if __name__ == "__main__":
    main()
