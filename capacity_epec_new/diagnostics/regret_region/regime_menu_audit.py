"""Continuous-search audit of the best restricted regime-menu candidates.

`regime_menu_game.py` can only measure regret against its finite strategy menu.
This script reconstructs chosen menu profiles and applies the same structured
feasible probes and MPEC refinements used by `br_dynamics.py`.  The resulting
gains remain lower bounds for I2/I3, while I1 also receives the merchant upper
bound.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import br_dynamics as bd
import regret_common as rc
import region_search as rs


MENU_OUT = rc.OUTPUT_ROOT / "regime_menu"
OUT = MENU_OUT / "continuous_audit"
NODES = tuple(rc.market_data(100.0).nodes)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
        )
        writer.writeheader()
        writer.writerows(rows)


def load_strategies() -> dict[tuple[str, str], tuple[dict, dict]]:
    strategies = {}
    for row in read_rows(MENU_OUT / "strategies.csv"):
        investor, label = row["investor"], row["label"]
        power = {node: float(row[f"P_{node}"]) for node in NODES}
        energy = {node: float(row[f"E_{node}"]) for node in NODES}
        strategies[investor, label] = power, energy
    return strategies


def profile_from_row(row: dict[str, str], strategies) -> tuple[dict, dict]:
    power, energy = {}, {}
    for investor in rc.INVESTORS:
        p, e = strategies[investor, row[f"strategy_{investor}"]]
        for node in NODES:
            power[investor, node] = p[node]
            energy[investor, node] = e[node]
    return power, energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-ids", default="33")
    parser.add_argument(
        "--menu-name",
        default="regime_menu",
        help="Output folder produced by regime_menu_game.py.",
    )
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--grid", choices=("lean", "full"), default="lean")
    parser.add_argument(
        "--investors",
        default=",".join(rc.INVESTORS),
        help="Comma-separated subset of I1,I2,I3 to audit.",
    )
    args = parser.parse_args()

    global MENU_OUT, OUT
    MENU_OUT = rc.OUTPUT_ROOT / args.menu_name
    OUT = MENU_OUT / "continuous_audit"
    OUT.mkdir(parents=True, exist_ok=True)
    rs.GRID = rs.LEAN_GRID if args.grid == "lean" else rs.FULL_GRID
    wanted = {int(value.strip()) for value in args.profile_ids.split(",") if value.strip()}
    selected_investors = tuple(
        value.strip() for value in args.investors.split(",") if value.strip()
    )
    unknown_investors = set(selected_investors) - set(rc.INVESTORS)
    if unknown_investors:
        raise ValueError(f"Unknown investors: {sorted(unknown_investors)}")
    menu_rows = {
        int(row["profile_id"]): row
        for row in read_rows(MENU_OUT / "payoffs_and_menu_regret.csv")
        if row.get("status") == "optimal"
    }
    missing = wanted - set(menu_rows)
    if missing:
        raise KeyError(f"Unknown feasible menu profile ids: {sorted(missing)}")

    strategies = load_strategies()
    rho = float(args.rho)
    data = rc.market_data(rho)
    config = rc.game_config(data)
    limits = config.node_limits(data)
    static = rc.static_arrays(data)
    # Preserve completed audits from earlier invocations while replacing any
    # rows for profiles requested in this invocation.  This lets expensive
    # candidate audits be resumed one profile at a time without overwriting
    # prior evidence.
    def preserved(name: str) -> list[dict]:
        path = OUT / name
        if not path.exists():
            return []
        return [
            row
            for row in read_rows(path)
            if int(row["profile_id"]) not in wanted
            or row.get("investor") not in selected_investors
        ]

    audit_rows = preserved("candidate_audit.csv")
    response_rows = preserved("response_capacities.csv")
    mpec_rows = preserved("mpec_log.csv")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=rc._init_worker) as pool:
        for profile_id in sorted(wanted):
            menu_row = menu_rows[profile_id]
            profile = profile_from_row(menu_row, strategies)
            base = rc.evaluate_profile(data, config, *profile)
            if base.get("status") != "optimal":
                raise RuntimeError(f"Profile {profile_id} failed to clear: {base.get('status')}")
            for investor in selected_investors:
                tag = f"profile{profile_id}_{investor}"
                result = bd.search_response(
                    pool, rho, profile, investor, tag, limits, data, config
                )
                best = result["best"]
                gain = max(0.0, best["profit"] - result["current"])
                bound = rs.merchant_bound(rho, *profile) if investor == "I1" else None
                response_profile = rc.with_candidate(
                    *profile, investor, best["power"], best["energy"]
                )
                response_result = rc.evaluate_profile(data, config, *response_profile)
                response_sets = rc.active_sets(response_result, static)
                audit_rows.append(
                    {
                        "profile_id": profile_id,
                        "investor": investor,
                        "current_strategy": menu_row[f"strategy_{investor}"],
                        "current_profit": result["current"],
                        "best_found_profit": best["profit"],
                        "regret_lower_bound": gain,
                        "relative_regret_lower_bound": gain
                        / max(1.0, abs(result["current"])),
                        "best_source": best["source"],
                        "best_label": best["label"],
                        "best_probe_label": result["best_probe"]["label"],
                        "best_probe_gain": max(
                            0.0, result["best_probe"]["profit"] - result["current"]
                        ),
                        "I1_profit_upper_bound": bound["upper_bound"] if bound else "",
                        "I1_regret_upper_bound": (
                            max(0.0, bound["upper_bound"] - result["current"])
                            if bound
                            else ""
                        ),
                        "response_lines_priced": ";".join(
                            sorted(response_sets["lines_priced"])
                        ),
                        "response_reference_inexact": response_result["reference_inexact"],
                        "response_verify_max_abs_profit_diff": response_result.get(
                            "verify_max_abs_profit_diff", ""
                        ),
                        "grid": args.grid,
                    }
                )
                for node in NODES:
                    response_rows.append(
                        {
                            "profile_id": profile_id,
                            "investor": investor,
                            "node": node,
                            "power_mw": best["power"][node],
                            "energy_mwh": best["energy"][node],
                            "source": best["source"],
                            "label": best["label"],
                        }
                    )
                for row in result["mpec_rows"]:
                    mpec_rows.append(
                        {
                            "profile_id": profile_id,
                            **{k: v for k, v in row.items() if k not in ("power", "energy")},
                        }
                    )
                print(
                    f"profile {profile_id} {investor}: current={result['current']:.3f}, "
                    f"best={best['profit']:.3f}, gain={gain:.3f}, "
                    f"via {best['source']}:{best['label']}"
                    + (
                        f", I1 upper={bound['upper_bound']:.3f}"
                        if bound
                        else ""
                    ),
                    flush=True,
                )
                write_rows(OUT / "candidate_audit.csv", audit_rows)
                write_rows(OUT / "response_capacities.csv", response_rows)
                write_rows(OUT / "mpec_log.csv", mpec_rows)


if __name__ == "__main__":
    main()
