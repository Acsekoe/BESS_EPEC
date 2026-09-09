"""Certify a capacity equilibrium from an exact response surface.

The iterative schemes in this directory (damped Jacobi, fixed- and
alternating-order Gauss--Seidel, the oracle-efficient Nikaido--Isoda descent)
all assume the best-response map has a fixed point they can walk to.  In this
model it does not: each investor's payoff is piecewise linear in its own
capacity and its optimum sits *on a kink*, so the best-response map is a step
function and the equilibria form a continuum rather than an isolated point.
No amount of damping, ordering or regret-guided line search terminates on that.

This runner therefore never iterates.  It evaluates an exact response surface,
reads the best responses straight off it, and certifies the candidates with the
same zero-proximal multistart MPEC audit used everywhere else in this package.

    Stage 0  screen   -- exact unilateral scale scan per investor; investors
                         that cannot earn a positive profit anywhere are pinned
                         at zero and re-checked by the stage-4 audit
    Stage 1  surface  -- one exact market clear per grid point over the active
                         investors' portfolio SCALES (nodal shape held at the
                         start profile, so each active investor is 1-D)
    Stage 2  read off -- best response = argmax along each grid line; a pure
                         Nash equilibrium is a mutual argmax.  Exact on the
                         grid: no tolerance, no convergence criterion
    Stage 3  refine   -- fine unilateral scans around each candidate give the
                         off-grid deviation gain within the scale-only space
    Stage 4  certify  -- audit_equilibrium lets every investor re-optimise
                         freely over all nodes and durations.  Its maximum
                         profitable deviation is the epsilon that is reported

Cost is fixed and known before the run starts: one market clear per grid point
plus one MPEC audit per audited candidate.  Nothing depends on a starting point
being close, on a step size, or on an update order.

    python model/run_grid_certified_equilibrium.py \
        --initial-capacities model/output/audited_fixed_gs_from_sweep254/best_capacities.csv \
        --grid I2=0:50:26 --grid I3=140:230:31 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyomo

import capacity_game as game
import reporting
from investors import three_investors
from market_data import load_market_data
from solvers import SolverSettings, ipopt_path


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = MODEL_DIR / "input" / "market_data_smoothed.json"
DEFAULT_START = MODEL_DIR / "output" / "audited_fixed_gs_from_sweep254" / "best_capacities.csv"
DEFAULT_OUTPUT = MODEL_DIR / "output" / "grid_certified_equilibrium"


# --------------------------------------------------------------------------
# Inputs


def read_capacities(path: Path) -> tuple[game.Capacities, game.Capacities]:
    power: game.Capacities = {}
    energy: game.Capacities = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["investor"], row["node"]
            power[key] = float(row["power_mw"])
            energy[key] = float(row["energy_mwh"])
    if not power:
        raise ValueError(f"No capacities found in {path}.")
    return power, energy


def parse_grid_spec(text: str) -> tuple[str, tuple[float, ...]]:
    """``I3=140:230:31`` becomes ("I3", (140.0, 143.0, ..., 230.0)) in total MW."""

    unit, _, rest = text.partition("=")
    parts = rest.split(":")
    if not unit or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Grid spec {text!r} must look like INVESTOR=min:max:count."
        )
    low, high, count = float(parts[0]), float(parts[1]), int(parts[2])
    if count < 2 or high <= low:
        raise argparse.ArgumentTypeError(f"Grid spec {text!r} is empty or reversed.")
    step = (high - low) / (count - 1)
    return unit, tuple(low + step * i for i in range(count))


# --------------------------------------------------------------------------
# Worker.  Windows spawns rather than forks, so the child rebuilds the model
# once in an initializer instead of pickling it with every task.

_WORKER: dict[str, object] = {}


def _init_worker(data_path: str, start_path: str) -> None:
    data = load_market_data(Path(data_path))
    investors = three_investors(data)
    _WORKER["data"] = data
    _WORKER["investors"] = investors
    _WORKER["config"] = game.GameConfig(investors=investors, solver=SolverSettings())
    _WORKER["base"] = read_capacities(Path(start_path))


def scaled_profile(data, base, scales: dict[str, float]):
    """Scale each investor's whole portfolio, preserving nodal shape and duration.

    Scaling power and energy by the same factor keeps every node's duration
    inside ``[ratio_min, ratio_max]`` whenever the start profile was feasible.
    """

    base_power, base_energy = base
    power, energy = dict(base_power), dict(base_energy)
    for unit, scale in scales.items():
        for node in data.nodes:
            power[unit, node] = base_power[unit, node] * scale
            energy[unit, node] = base_energy[unit, node] * scale
    return power, energy


def _profile_fits(data, config, power: game.Capacities) -> bool:
    limits = config.node_limits(data)
    return all(
        sum(power[unit, node] for unit in config.investor_ids) <= limits[node] + 1e-9
        for node in data.nodes
    )


def _evaluate(scales: dict[str, float]) -> dict[str, object] | None:
    """One grid point: exact clear plus settlement for every investor."""

    data = _WORKER["data"]
    config = _WORKER["config"]
    power, energy = scaled_profile(data, _WORKER["base"], scales)
    if not _profile_fits(data, config, power):
        return None
    row: dict[str, object] = {f"scale_{u}": s for u, s in scales.items()}
    for unit in config.investor_ids:
        row[f"power_{unit}_mw"] = sum(power[unit, node] for node in data.nodes)
    try:
        market = game.clear(data, config, power, energy)
    except Exception as exc:  # a failed clear is a hole in the surface, not a stop
        row["error"] = str(exc)
        return row
    for investor in _WORKER["investors"]:
        row[f"profit_{investor.investor_id}"] = game.recleared_profit(
            market, data, config, investor, power, energy
        )
    return row


def _map_profiles(tasks: list[dict[str, float]], workers: int, args) -> list:
    if workers <= 1:
        if not _WORKER:
            _init_worker(str(args.data), str(args.initial_capacities))
        return [_evaluate(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(args.data), str(args.initial_capacities)),
    ) as pool:
        return list(pool.map(_evaluate, tasks, chunksize=4))


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """``reporting.write_rows`` with a union schema.

    Rows here are genuinely ragged -- a grid point whose clear failed carries
    ``error`` instead of the profit columns, and an investor that could not be
    screened carries only its id -- so taking the first row's keys as the
    header, as ``reporting.write_rows`` does, would drop or reject fields.
    """

    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    reporting.write_rows(path, [{key: row.get(key, "") for key in fields} for row in rows])


# --------------------------------------------------------------------------
# Stage 0: which investors are actually in the game?


def screen_investors(config, scales, workers: int, args) -> list[dict[str, object]]:
    """Exact unilateral scale scan per investor, rivals held at the start profile."""

    tasks = [
        {unit: (scale if unit == target else 1.0) for unit in config.investor_ids}
        for target in config.investor_ids
        for scale in scales
    ]
    results = _map_profiles(tasks, workers, args)
    rows: list[dict[str, object]] = []
    for target, chunk in zip(config.investor_ids, _chunks(results, len(scales))):
        scored = [r for r in chunk if r and f"profit_{target}" in r]
        if not scored:
            rows.append({"investor": target, "screened": False})
            continue
        best = max(scored, key=lambda r: r[f"profit_{target}"])
        worst = min(scored, key=lambda r: r[f"profit_{target}"])
        rows.append(
            {
                "investor": target,
                "screened": True,
                "best_profit_eur_per_day": best[f"profit_{target}"],
                "best_power_mw": best[f"power_{target}_mw"],
                "worst_profit_eur_per_day": worst[f"profit_{target}"],
                "profit_span_eur_per_day": (
                    best[f"profit_{target}"] - worst[f"profit_{target}"]
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Stage 2: best responses and pure Nash equilibria, read off the surface


def best_responses(rows: list[dict[str, object]], active: list[str]):
    """For each investor, its argmax along its own axis at every rival combination."""

    table: dict[str, dict[tuple, tuple[float, float]]] = {}
    for unit in active:
        others = [u for u in active if u != unit]
        grouped: dict[tuple, list[dict]] = {}
        for row in rows:
            if f"profit_{unit}" not in row:
                continue
            key = tuple(round(float(row[f"scale_{u}"]), 9) for u in others)
            grouped.setdefault(key, []).append(row)
        table[unit] = {
            key: max(
                ((float(r[f"scale_{unit}"]), float(r[f"profit_{unit}"])) for r in group),
                key=lambda pair: pair[1],
            )
            for key, group in grouped.items()
        }
    return table


def grid_equilibria(rows, active: list[str], table) -> list[dict[str, object]]:
    """A pure NE is a profile that is simultaneously every active investor's argmax."""

    found: list[dict[str, object]] = []
    for row in rows:
        if any(f"profit_{u}" not in row for u in active):
            continue
        regrets = {}
        for unit in active:
            others = [u for u in active if u != unit]
            key = tuple(round(float(row[f"scale_{u}"]), 9) for u in others)
            _, best_profit = table[unit][key]
            regrets[unit] = best_profit - float(row[f"profit_{unit}"])
        if all(value <= 1e-9 for value in regrets.values()):
            entry: dict[str, object] = {}
            for unit in active:
                entry[f"scale_{unit}"] = float(row[f"scale_{unit}"])
                entry[f"power_{unit}_mw"] = float(row[f"power_{unit}_mw"])
                entry[f"profit_{unit}"] = float(row[f"profit_{unit}"])
            found.append(entry)
    return found


# --------------------------------------------------------------------------
# Stage 3: off-grid refinement inside the scale-only strategy space


def refine_candidate(candidate, active: list[str], steps, workers: int, args) -> float:
    """Largest unilateral gain available off the grid, in EUR/day."""

    base_scales = {u: float(candidate[f"scale_{u}"]) for u in active}
    tasks: list[dict[str, float]] = []
    owners: list[str] = []
    for unit in active:
        for delta in steps:
            trial = dict(base_scales)
            trial[unit] = base_scales[unit] + delta
            if trial[unit] < 0.0:
                continue
            tasks.append(trial)
            owners.append(unit)
    results = _map_profiles(tasks, workers, args)
    gain = 0.0
    for unit, row in zip(owners, results):
        if row and f"profit_{unit}" in row:
            gain = max(gain, row[f"profit_{unit}"] - float(candidate[f"profit_{unit}"]))
    return gain


def audit_numerically_valid(report: game.AuditReport) -> bool:
    """The same gate the Gauss--Seidel runner applies, reused for the log line."""

    if not report.valid or not all(row.optimal for row in report.rows):
        return False
    limits = {
        "embedded_reclear_profit_gap_eur_per_day": (
            report.config.audit_reclear_gap_tolerance_eur_per_day
        ),
        "complementarity_max_violation": report.config.solver.tolerance,
        "max_bound_violation": report.config.solver.tolerance,
    }
    return all(
        (value := report.worst(field)) is not None and value <= limit
        for field, limit in limits.items()
    )


# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = load_market_data(args.data)
    investors = three_investors(data)
    config = game.GameConfig(
        investors=investors,
        solver=SolverSettings(),
        parallel_workers=args.workers,
        refinement_starts=args.refinement_starts,
        audit_profit_tolerance_eur_per_day=args.epsilon,
    )
    base = read_capacities(args.initial_capacities)
    grids = dict(args.grid)
    unknown = set(grids) - set(config.investor_ids)
    if unknown:
        raise SystemExit(f"Unknown investors in --grid: {sorted(unknown)}")

    reporting.write_json(
        out / "run_config.json",
        {
            **reporting.run_config(
                config, args.data, hashlib.sha256(args.data.read_bytes()).hexdigest()
            ),
            "algorithm": "grid-certified-equilibrium",
            "iterative": False,
            "capacity_distance_is_stopping_criterion": False,
            "start_profile": {
                "file": str(args.initial_capacities.resolve()),
                "sha256": hashlib.sha256(
                    args.initial_capacities.read_bytes()
                ).hexdigest(),
            },
            "grid_total_power_mw": {u: list(v) for u, v in grids.items()},
            "screen_scales": list(args.screen_scales),
            "refine_scale_steps": list(args.refine_steps),
            "python_version": platform.python_version(),
            "pyomo_version": pyomo.version.version,
            "ipopt_executable": str(ipopt_path(config.solver)),
        },
    )

    # ---- Stage 0: screen -------------------------------------------------
    print("stage 0: screening investors", flush=True)
    screen = screen_investors(config, args.screen_scales, args.workers, args)
    write_rows(out / "screen.csv", screen)
    base_power = base[0]
    base_totals = {
        u: sum(base_power[u, n] for n in data.nodes) for u in config.investor_ids
    }
    pinned = [
        str(r["investor"])
        for r in screen
        if r.get("screened")
        and float(r["best_profit_eur_per_day"]) <= args.inactive_profit_eur_per_day
    ]
    for row in screen:
        tag = "   -> PINNED AT ZERO" if row["investor"] in pinned else ""
        print(
            f"   {row['investor']}: best "
            f"{float(row.get('best_profit_eur_per_day', math.nan)):.2f} EUR/day "
            f"at {float(row.get('best_power_mw', math.nan)):.2f} MW{tag}",
            flush=True,
        )
    active = [u for u in config.investor_ids if u in grids and u not in pinned]
    if not active:
        raise SystemExit("No active investors left to grid over.")

    # ---- Stage 1: response surface ---------------------------------------
    axes = []
    for unit in active:
        if base_totals[unit] <= 0.0:
            raise SystemExit(f"{unit} has zero capacity in the start profile.")
        axes.append([mw / base_totals[unit] for mw in grids[unit]])
    combos = list(itertools.product(*axes))
    if len(combos) > args.max_grid_points:
        raise SystemExit(
            f"{len(combos)} grid points exceeds --max-grid-points "
            f"({args.max_grid_points}); coarsen the grid or raise the guard."
        )
    tasks = []
    for combo in combos:
        scales = {u: (0.0 if u in pinned else 1.0) for u in config.investor_ids}
        scales.update(dict(zip(active, combo)))
        tasks.append(scales)
    print(
        f"stage 1: {len(tasks)} grid points over {active} "
        f"({' x '.join(str(len(a)) for a in axes)})",
        flush=True,
    )
    surface_started = time.perf_counter()
    rows = [r for r in _map_profiles(tasks, args.workers, args) if r is not None]
    surface_seconds = time.perf_counter() - surface_started
    write_rows(out / "grid.csv", rows)
    evaluated = [r for r in rows if "error" not in r]
    print(
        f"   {len(evaluated)}/{len(tasks)} points cleared in {surface_seconds:.0f}s",
        flush=True,
    )

    # ---- Stage 2: read off best responses --------------------------------
    table = best_responses(evaluated, active)
    # One uniform schema across investors, so the file loads as a single frame:
    # ``rival_power_mw`` is the plotting axis in the two-investor case and is
    # left empty when an investor faces more than one rival.
    br_rows = []
    for unit in active:
        others = [u for u in active if u != unit]
        for key, (scale, profit) in sorted(table[unit].items()):
            rival_mw = [value * base_totals[other] for other, value in zip(others, key)]
            br_rows.append(
                {
                    "investor": unit,
                    "rivals": "|".join(others),
                    "rival_profile_mw": "|".join(f"{mw:.6f}" for mw in rival_mw),
                    "rival_power_mw": rival_mw[0] if len(rival_mw) == 1 else "",
                    "best_response_scale": scale,
                    "best_response_power_mw": scale * base_totals[unit],
                    "best_response_profit_eur_per_day": profit,
                }
            )
    write_rows(out / "best_responses.csv", br_rows)

    candidates = grid_equilibria(evaluated, active, table)
    print(f"stage 2: {len(candidates)} pure Nash equilibria on the grid", flush=True)
    for candidate in candidates:
        total = sum(float(candidate[f"power_{u}_mw"]) for u in active)
        print(
            "   "
            + "  ".join(f"{u}={float(candidate[f'power_{u}_mw']):7.2f} MW" for u in active)
            + f"   total={total:7.2f} MW",
            flush=True,
        )
    if not candidates:
        print("   the best-response curves do not intersect on this grid.")

    # ---- Stage 3: off-grid refinement ------------------------------------
    print("stage 3: off-grid refinement", flush=True)
    for candidate in candidates:
        candidate["offgrid_epsilon_eur_per_day"] = refine_candidate(
            candidate, active, args.refine_steps, args.workers, args
        )
        print(
            "   "
            + "  ".join(f"{u}={float(candidate[f'power_{u}_mw']):7.2f}" for u in active)
            + f"   off-grid eps = "
            f"{float(candidate['offgrid_epsilon_eur_per_day']):8.2f} EUR/day",
            flush=True,
        )
    candidates.sort(key=lambda c: float(c["offgrid_epsilon_eur_per_day"]))
    write_rows(out / "equilibria.csv", candidates)

    # ---- Stage 4: MPEC certificate ---------------------------------------
    audited: list[dict[str, object]] = []
    certified: dict[str, object] | None = None
    to_audit = candidates if args.audit_all else candidates[: args.audit_candidates]
    for index, candidate in enumerate(to_audit):
        scales = {u: (0.0 if u in pinned else 1.0) for u in config.investor_ids}
        scales.update({u: float(candidate[f"scale_{u}"]) for u in active})
        power, energy = scaled_profile(data, base, scales)
        state = game.GameState(power=power, energy=energy)
        print(
            f"stage 4: auditing candidate {index + 1}/{len(to_audit)}  "
            + "  ".join(f"{u}={float(candidate[f'power_{u}_mw']):.2f} MW" for u in active),
            flush=True,
        )
        audit_started = time.perf_counter()
        report = game.audit_equilibrium(data, config, state)
        audit_seconds = time.perf_counter() - audit_started
        regret = report.worst("profitable_deviation_eur_per_day")
        for row in report.rows:
            audited.append(
                {
                    "candidate": index,
                    **{f"power_{u}_mw": candidate[f"power_{u}_mw"] for u in active},
                    "investor": row.investor,
                    "audit_valid": row.audit_valid,
                    "optimal": row.optimal,
                    "termination": row.termination,
                    "profitable_deviation_eur_per_day": (
                        row.profitable_deviation_eur_per_day
                    ),
                    "current_recleared_profit_eur_per_day": (
                        row.current_recleared_profit_eur_per_day
                    ),
                    "max_power_deviation_mw": row.max_power_deviation_mw,
                    "max_energy_deviation_mwh": row.max_energy_deviation_mwh,
                    "embedded_reclear_profit_gap_eur_per_day": (
                        row.embedded_reclear_profit_gap_eur_per_day
                    ),
                    "complementarity_max_violation": row.complementarity_max_violation,
                    "max_bound_violation": row.max_bound_violation,
                    "audit_error": row.audit_error,
                }
            )
        print(
            f"   valid={report.valid}  numerically_valid={audit_numerically_valid(report)}"
            f"  max regret={regret if regret is None else round(regret, 3)} EUR/day"
            f"  ({audit_seconds:.0f}s)",
            flush=True,
        )
        record = {
            **{f"power_{u}_mw": candidate[f"power_{u}_mw"] for u in active},
            "offgrid_epsilon_eur_per_day": candidate["offgrid_epsilon_eur_per_day"],
            "audited_epsilon_eur_per_day": regret,
            "audit_valid": report.valid,
            "audit_numerically_valid": audit_numerically_valid(report),
            "audit_seconds": audit_seconds,
        }
        better = (
            certified is None
            or (
                regret is not None
                and math.isfinite(regret)
                and (
                    certified["audited_epsilon_eur_per_day"] is None
                    or regret < float(certified["audited_epsilon_eur_per_day"])
                )
            )
        )
        if better:
            certified = record
            reporting.write_rows(
                out / "certified_capacities.csv",
                reporting.capacity_rows(data, config.investors, power, energy, sweep=0),
            )
    write_rows(out / "audit.csv", audited)

    totals = [sum(float(c[f"power_{u}_mw"]) for u in active) for c in candidates]
    reporting.write_json(
        out / "summary.json",
        {
            "algorithm": "grid-certified-equilibrium",
            "iterative": False,
            "active_investors": active,
            "pinned_investors": pinned,
            "grid_points_requested": len(tasks),
            "grid_points_cleared": len(evaluated),
            "grid_equilibria_found": len(candidates),
            "equilibrium_is_a_set": len(candidates) > 1,
            "system_total_power_mw": totals or None,
            "system_total_power_span_mw": (max(totals) - min(totals)) if totals else None,
            "candidates_audited": len(to_audit),
            "certified": certified,
            "epsilon_target_eur_per_day": args.epsilon,
            "surface_seconds": surface_seconds,
            "runtime_seconds": time.perf_counter() - started,
        },
    )

    if certified is None:
        print("\nNo candidate was audited.")
        return 1
    eps = certified["audited_epsilon_eur_per_day"]
    print(
        f"\ncertified epsilon-equilibrium: eps = "
        f"{eps if eps is None else round(eps, 3)} EUR/day at "
        + "  ".join(f"{u}={float(certified[f'power_{u}_mw']):.2f} MW" for u in active)
    )
    ok = (
        eps is not None
        and math.isfinite(eps)
        and eps <= args.epsilon
        and bool(certified["audit_numerically_valid"])
    )
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--initial-capacities",
        type=Path,
        default=DEFAULT_START,
        help="Profile whose nodal shape defines each investor's scale axis.",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid_spec,
        action="append",
        default=[],
        metavar="INVESTOR=min:max:count",
        help="Total-portfolio-MW grid for one investor; repeatable.",
    )
    parser.add_argument(
        "--screen-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
        help="Scale multiples used by the stage-0 screen.",
    )
    parser.add_argument(
        "--inactive-profit-eur-per-day",
        type=float,
        default=0.0,
        help="An investor whose best screened profit is at or below this is pinned "
        "at zero; the stage-4 audit re-checks that it really wants to stay out.",
    )
    parser.add_argument(
        "--refine-steps",
        type=float,
        nargs="+",
        default=(-0.02, -0.01, -0.005, 0.005, 0.01, 0.02),
        help="Scale offsets for the stage-3 off-grid check.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=100.0,
        help="Audited regret at or below which the run exits 0.",
    )
    parser.add_argument(
        "--audit-candidates",
        type=int,
        default=1,
        help="How many candidates to audit, best off-grid epsilon first.",
    )
    parser.add_argument(
        "--audit-all",
        action="store_true",
        help="Audit every grid equilibrium (one MPEC audit each).",
    )
    parser.add_argument("--refinement-starts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-grid-points", type=int, default=5000)
    args = parser.parse_args()
    if not args.grid:
        raise SystemExit("At least one --grid INVESTOR=min:max:count is required.")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
