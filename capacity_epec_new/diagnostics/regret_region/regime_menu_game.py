"""Exact finite regime-menu game built from observed response strategies.

The prior diagnostics produced several distinct capacity strategies for each
investor while the market alternated between congested and decongested
midday-corridor regimes.  This script crosses those strategies into a finite
normal-form approximation, re-clears every feasible profile with the
maintained ISO procedure, and computes each player's regret *within the menu*.

The calculation is deliberately modest in what it claims.  A restricted-menu
equilibrium is not an equilibrium of the continuous 54-dimensional game, and
absence of a restricted pure equilibrium is not a nonexistence proof.  The
menu is useful because it tests whether the observed regime alternation is an
artefact of the order in which the earlier dynamics visited strategies, or is
already present when all observed strategies can be combined freely.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import regret_common as rc


OUT = rc.OUTPUT_ROOT / "regime_menu"
DYNAMICS_PROFILES = rc.OUTPUT_ROOT / "br_dynamics" / "profiles.csv"
NODES = tuple(rc.market_data(100.0).nodes)

# Step 0 is the original sweep-100 profile.  Later entries are the strategies
# immediately after that investor moved in each of the three undamped rounds.
OBSERVED_TAGS = {
    "I1": (
        "step0_final_sweep100",
        "step2_round1_I1",
        "step5_round2_I1",
        "step8_round3_I1",
    ),
    "I2": (
        "step0_final_sweep100",
        "step3_round1_I2",
        "step6_round2_I2",
        "step9_round3_I2",
    ),
    "I3": (
        "step0_final_sweep100",
        "step1_round1_I3",
        "step4_round2_I3",
        "step7_round3_I3",
    ),
}


@dataclass(frozen=True)
class Strategy:
    investor: str
    label: str
    source: str
    power: dict[str, float]
    energy: dict[str, float]

    @property
    def total_power(self) -> float:
        return sum(self.power.values())

    @property
    def total_energy(self) -> float:
        return sum(self.energy.values())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def from_profile_row(row: dict[str, str], investor: str, label: str) -> Strategy:
    return Strategy(
        investor=investor,
        label=label,
        source=row["tag"],
        power={node: float(row[f"P_{investor}_{node}"]) for node in NODES},
        energy={node: float(row[f"E_{investor}_{node}"]) for node in NODES},
    )


def audit_strategy(investor: str) -> Strategy:
    record = rc.selected_record(investor)
    return Strategy(
        investor=investor,
        label="audit_selected",
        source=f"final audit selected: {record['start']}",
        power={node: float(record["power"][node]) for node in NODES},
        energy={node: float(record["energy"][node]) for node in NODES},
    )


def strategy_key(strategy: Strategy) -> tuple[float, ...]:
    return tuple(
        round(value, 8)
        for node in NODES
        for value in (strategy.power[node], strategy.energy[node])
    )


def oracle_response_paths(oracle_round: int) -> tuple[Path, ...]:
    """Return all completed audit-response files available before a round."""
    if oracle_round <= 0:
        return ()
    paths = [
        rc.OUTPUT_ROOT / "regime_menu" / "continuous_audit" / "response_capacities.csv"
    ]
    paths.extend(
        rc.OUTPUT_ROOT
        / f"regime_menu_oracle{completed_round}"
        / "continuous_audit"
        / "response_capacities.csv"
        for completed_round in range(1, oracle_round)
    )
    return tuple(paths)


def oracle_strategies(response_paths: tuple[Path, ...]) -> dict[str, list[Strategy]]:
    found = {investor: [] for investor in rc.INVESTORS}
    for response_path in response_paths:
        if not response_path.exists():
            raise FileNotFoundError(
                f"Oracle round requires completed response file: {response_path}"
            )
        menu_name = response_path.parent.parent.name
        label_prefix = (
            "oracle"
            if menu_name == "regime_menu"
            else menu_name.replace("regime_menu_", "")
        )
        grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_rows(response_path):
            key = (row["profile_id"], row["investor"], row["source"], row["label"])
            grouped[key].append(row)
        for (profile_id, investor, source, label), rows in grouped.items():
            by_node = {row["node"]: row for row in rows}
            if set(by_node) != set(NODES):
                raise ValueError(
                    f"Incomplete oracle response for profile {profile_id}, {investor}"
                )
            found[investor].append(
                Strategy(
                    investor=investor,
                    label=f"{label_prefix}_p{profile_id}",
                    source=f"continuous audit {menu_name} {source}:{label}",
                    power={node: float(by_node[node]["power_mw"]) for node in NODES},
                    energy={node: float(by_node[node]["energy_mwh"]) for node in NODES},
                )
            )
    return found


def strategy_menus(
    response_paths: tuple[Path, ...] = (),
) -> dict[str, tuple[Strategy, ...]]:
    rows = read_rows(DYNAMICS_PROFILES)
    by_tag = {row["tag"]: row for row in rows}
    oracle = oracle_strategies(response_paths) if response_paths else {
        investor: [] for investor in rc.INVESTORS
    }
    menus: dict[str, tuple[Strategy, ...]] = {}
    for investor in rc.INVESTORS:
        candidates: list[Strategy] = []
        for tag in OBSERVED_TAGS[investor]:
            if tag not in by_tag:
                raise KeyError(f"Missing dynamics profile {tag!r}")
            short = "sweep100" if tag.startswith("step0_") else tag.split("_")[0]
            candidates.append(from_profile_row(by_tag[tag], investor, short))
        candidates.append(audit_strategy(investor))
        candidates.append(
            Strategy(
                investor=investor,
                label="exit",
                source="constructed zero-capacity strategy",
                power={node: 0.0 for node in NODES},
                energy={node: 0.0 for node in NODES},
            )
        )
        candidates.extend(oracle[investor])
        unique: list[Strategy] = []
        seen: set[tuple[float, ...]] = set()
        for candidate in candidates:
            key = strategy_key(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        menus[investor] = tuple(unique)
    return menus


def full_profile(combo: tuple[Strategy, ...]) -> tuple[dict, dict]:
    power, energy = {}, {}
    for strategy in combo:
        for node in NODES:
            power[strategy.investor, node] = strategy.power[node]
            energy[strategy.investor, node] = strategy.energy[node]
    return power, energy


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--include-oracle-responses",
        action="store_true",
        help="Compatibility alias for --oracle-round 1.",
    )
    parser.add_argument(
        "--oracle-round",
        type=int,
        default=0,
        help=(
            "Adaptive menu round. Round 1 adds base-menu audit responses; round N "
            "adds responses from the base and completed rounds 1..N-1."
        ),
    )
    args = parser.parse_args()

    oracle_round = args.oracle_round
    if args.include_oracle_responses:
        if oracle_round not in (0, 1):
            parser.error("--include-oracle-responses conflicts with --oracle-round > 1")
        oracle_round = 1
    if oracle_round < 0:
        parser.error("--oracle-round must be nonnegative")
    response_paths = oracle_response_paths(oracle_round)

    global OUT
    OUT = rc.OUTPUT_ROOT / (
        f"regime_menu_oracle{oracle_round}" if oracle_round else "regime_menu"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rho = float(args.rho)
    data = rc.market_data(rho)
    config = rc.game_config(data)
    static = rc.static_arrays(data)
    menus = strategy_menus(response_paths)

    strategy_rows: list[dict] = []
    for investor, strategies in menus.items():
        for strategy in strategies:
            row = {
                "investor": investor,
                "label": strategy.label,
                "source": strategy.source,
                "total_power_mw": strategy.total_power,
                "total_energy_mwh": strategy.total_energy,
            }
            row.update({f"P_{node}": strategy.power[node] for node in NODES})
            row.update({f"E_{node}": strategy.energy[node] for node in NODES})
            strategy_rows.append(row)
    write_csv(OUT / "strategies.csv", strategy_rows)

    combos = list(itertools.product(*(menus[i] for i in rc.INVESTORS)))
    tasks = []
    combo_by_key = {}
    for key, combo in enumerate(combos):
        profile = full_profile(combo)
        combo_by_key[key] = (combo, profile)
        tasks.append((key, rho, *profile))

    with rc.make_pool(args.workers) as pool:
        results = rc.evaluate_many(pool, tasks)

    payoff_rows: list[dict] = []
    result_by_labels: dict[tuple[str, ...], dict] = {}
    for key, result in results.items():
        combo, profile = combo_by_key[key]
        labels = tuple(strategy.label for strategy in combo)
        row = {
            "profile_id": key,
            **{f"strategy_{i}": labels[j] for j, i in enumerate(rc.INVESTORS)},
            "status": result.get("status", "missing"),
        }
        if result.get("status") != "optimal":
            payoff_rows.append(row)
            continue
        sets = rc.active_sets(result, static)
        row.update(
            lines_priced=";".join(sorted(sets["lines_priced"])),
            lines_priced_hash=rc.signature_hash(sets["lines_priced"]),
            iso_objective_eur_per_day=result["objective"],
            reference_inexact=result["reference_inexact"],
            verify_status=result["verify_status"],
            verify_max_abs_profit_diff=result.get("verify_max_abs_profit_diff", ""),
        )
        for investor in rc.INVESTORS:
            row[f"profit_{investor}"] = result["settle"][investor]["profit"]
        payoff_rows.append(row)
        result_by_labels[labels] = row

    # Restricted best responses respect the shared cap automatically: only
    # feasible, successfully cleared cross-combinations enter the comparison.
    for labels, row in result_by_labels.items():
        max_abs = 0.0
        max_relative = 0.0
        for idx, investor in enumerate(rc.INVESTORS):
            rivals = labels[:idx] + labels[idx + 1 :]
            alternatives = [
                candidate
                for candidate_labels, candidate in result_by_labels.items()
                if candidate_labels[:idx] + candidate_labels[idx + 1 :] == rivals
            ]
            best = max(alternatives, key=lambda candidate: float(candidate[f"profit_{investor}"]))
            current_profit = float(row[f"profit_{investor}"])
            gain = max(0.0, float(best[f"profit_{investor}"]) - current_profit)
            relative = gain / max(1.0, abs(current_profit))
            row[f"menu_best_strategy_{investor}"] = best[f"strategy_{investor}"]
            row[f"menu_best_profit_{investor}"] = best[f"profit_{investor}"]
            row[f"menu_regret_{investor}"] = gain
            row[f"menu_relative_regret_{investor}"] = relative
            row[f"menu_response_lines_{investor}"] = best["lines_priced"]
            max_abs = max(max_abs, gain)
            max_relative = max(max_relative, relative)
        row["max_menu_regret"] = max_abs
        row["max_menu_relative_regret"] = max_relative
        row["restricted_pure_equilibrium"] = max_abs <= 1.0e-4

    payoff_rows = sorted(
        payoff_rows,
        key=lambda row: (
            row.get("status") != "optimal",
            float(row.get("max_menu_relative_regret", "inf")),
            float(row.get("max_menu_regret", "inf")),
        ),
    )
    write_csv(OUT / "payoffs_and_menu_regret.csv", payoff_rows)

    feasible = [row for row in payoff_rows if row.get("status") == "optimal"]
    candidates = feasible[: min(30, len(feasible))]
    write_csv(OUT / "ranked_candidates.csv", candidates)

    regime_groups: dict[str, list[dict]] = defaultdict(list)
    for row in feasible:
        regime_groups[row["lines_priced_hash"]].append(row)
    regime_rows = []
    for regime, rows in regime_groups.items():
        best = min(rows, key=lambda row: float(row["max_menu_relative_regret"]))
        regime_rows.append(
            {
                "lines_priced_hash": regime,
                "lines_priced": rows[0]["lines_priced"],
                "profile_count": len(rows),
                "best_profile_id": best["profile_id"],
                "best_max_menu_regret": best["max_menu_regret"],
                "best_max_menu_relative_regret": best["max_menu_relative_regret"],
            }
        )
    regime_rows.sort(key=lambda row: float(row["best_max_menu_relative_regret"]))
    write_csv(OUT / "regimes.csv", regime_rows)

    transition_rows = []
    for row in feasible:
        for investor in rc.INVESTORS:
            transition_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "investor": investor,
                    "from_regime": row["lines_priced_hash"],
                    "from_lines_priced": row["lines_priced"],
                    "current_strategy": row[f"strategy_{investor}"],
                    "best_menu_strategy": row[f"menu_best_strategy_{investor}"],
                    "menu_regret": row[f"menu_regret_{investor}"],
                    "menu_relative_regret": row[f"menu_relative_regret_{investor}"],
                    "response_lines_priced": row[f"menu_response_lines_{investor}"],
                }
            )
    write_csv(OUT / "menu_transitions.csv", transition_rows)

    equilibria = [row for row in feasible if row["restricted_pure_equilibrium"]]
    best = feasible[0]
    manifest = {
        "rho": rho,
        "data": str(rc.DATA_PATH),
        "data_sha256": rc.sha256(rc.DATA_PATH),
        "dynamics_profiles": str(DYNAMICS_PROFILES),
        "strategy_counts": {investor: len(menu) for investor, menu in menus.items()},
        "cross_product_profiles": len(combos),
        "feasible_cleared_profiles": len(feasible),
        "regime_count": len(regime_rows),
        "restricted_pure_equilibrium_count": len(equilibria),
        "oracle_round": oracle_round,
        "includes_continuous_oracle_responses": bool(response_paths),
        "oracle_response_sources": [str(path) for path in response_paths],
        "best_restricted_profile": {
            key: best[key]
            for key in (
                "profile_id",
                "strategy_I1",
                "strategy_I2",
                "strategy_I3",
                "lines_priced",
                "profit_I1",
                "profit_I2",
                "profit_I3",
                "menu_regret_I1",
                "menu_regret_I2",
                "menu_regret_I3",
                "max_menu_regret",
                "max_menu_relative_regret",
            )
        },
        "qualification": (
            "All payoffs use exact procedural reclearing. Regrets are restricted to the "
            "finite observed-strategy menu and are lower bounds on continuous-game regret."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
