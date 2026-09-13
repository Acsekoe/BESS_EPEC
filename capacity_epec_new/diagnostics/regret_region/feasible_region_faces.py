"""Active and near-active faces of the investment feasible region.

For every investor i and node n the maintained code imposes (mpec.py,
capacity_game._check_capacity_feasible / _check_profile_node_limits):

    P_in >= 0,  E_in >= 0,  r_min P_in <= E_in <= r_max P_in,
    sum_i P_in <= L_n.

With rivals fixed, investor i's set is the product over nodes of the triangles
with vertices (0, 0), (R_n, r_min R_n), (R_n, r_max R_n), where R_n is the
residual shared cap.  This script reports, at each candidate profile, how far
each coordinate is from every face.  Output: output/feasible_region/faces_*.csv
"""

from __future__ import annotations

import csv

import regret_common as rc
import region_search as rs

OUT = rc.OUTPUT_ROOT / "feasible_region"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = rc.market_data(100.0)
    config = rc.game_config(data)
    limits = config.node_limits(data)
    rows, node_rows = [], []
    for label, (power, energy) in rs.candidates().items():
        for n in rs.NODES:
            total = sum(power[i, n] for i in rc.INVESTORS)
            node_rows.append(dict(candidate=label, node=n, total_power_mw=total, limit_mw=limits[n],
                                  utilization=total / limits[n], shared_cap_slack_mw=limits[n] - total))
        for inv in rc.INVESTORS:
            investor = next(x for x in config.investors if x.investor_id == inv)
            residual = rs.residual_caps(power, inv, limits)
            for n in rs.NODES:
                P, E = power[inv, n], energy[inv, n]
                if P == 0.0 and E == 0.0:
                    face = "origin_vertex(exact zero)"
                elif P < 1e-3:
                    face = "ghost(<1e-3 MW, numerically interior near origin)"
                else:
                    face = "material"
                dur = E / P if P > 0 else float("nan")
                rows.append(dict(
                    candidate=label, investor=inv, node=n, power_mw=P, energy_mwh=E, duration_h=dur,
                    classification=face,
                    slack_min_duration_mwh=E - investor.ratio_min * P,
                    slack_max_duration_mwh=investor.ratio_max * P - E,
                    relative_distance_to_2h=(dur - investor.ratio_min) / investor.ratio_min if P > 0 else float("nan"),
                    relative_distance_to_8h=(investor.ratio_max - dur) / investor.ratio_max if P > 0 else float("nan"),
                    near_min_duration_face=(P >= 1e-3 and (dur - investor.ratio_min) / investor.ratio_min < 0.02),
                    near_max_duration_face=(P >= 1e-3 and (investor.ratio_max - dur) / investor.ratio_max < 0.02),
                    residual_shared_cap_mw=residual[n], shared_cap_slack_mw=residual[n] - P,
                ))
    for name, data_rows in (("faces_by_investor_node.csv", rows), ("shared_cap_by_node.csv", node_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(data_rows[0]))
            w.writeheader()
            w.writerows(data_rows)
    final = [r for r in rows if r["candidate"] == "final_sweep100"]
    for r in final:
        if r["classification"] != "origin_vertex(exact zero)":
            print(f"{r['investor']} {r['node']} P={r['power_mw']:.6g} E={r['energy_mwh']:.6g} dur={r['duration_h']:.4f}"
                  f" {r['classification'][:8]} near2h={r['near_min_duration_face']} near8h={r['near_max_duration_face']}"
                  f" capslack={r['shared_cap_slack_mw']:.1f}")
    print("max utilization (final):", max(x["utilization"] for x in node_rows if x["candidate"] == "final_sweep100"))


if __name__ == "__main__":
    main()
