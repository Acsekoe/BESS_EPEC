"""Multi-investor spot-market EPEC solved by diagonalization.

Each strategic BESS investor solves the single-investor MPEC while all rivals
are frozen at their current-iterate capacities as one aggregated non-strategic
storage unit inside the lower-level clearing (exact for the single shared
round-trip efficiency). The shared nodal connection limit couples the
investors, so the solution concept is a generalized Nash equilibrium and the
outcome may depend on the update rule: Gauss-Jacobi (all investors respond to
the same previous iterate) versus Gauss-Seidel (sequential, later investors
see earlier same-iteration updates - the potential first-mover artifact).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyomo.environ as pyo

from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_INITIAL_POWER_MW,
    DEFAULT_INITIAL_RATIO_HOURS,
    DEFAULT_NODE_LIMIT_MW,
    EXPERIMENT_DATA_PATH,
    InvestorConfig,
    QuadraticDemandCurve,
    build_single_investor_mpec,
    default_quadratic_demand_curve,
    initialize_from_reference_dispatch,
)
from solver_utils import get_ipopt_solver

RIVAL_ID = "RIV"
DEFAULT_DAMPING = 0.7
DEFAULT_TOL_REL = 0.01
DEFAULT_FLOOR_MW = 1.0
DEFAULT_FLOOR_MWH = 2.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "epec"


@dataclass(frozen=True)
class EpecConfig:
    investors: tuple[InvestorConfig, ...]
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW
    update_rule: str = "jacobi"  # "jacobi" | "seidel" (solve order = investors order)
    damping: float = DEFAULT_DAMPING  # x' = (1-a)*x_old + a*x_best_response
    max_iters: int = 60
    tol_rel: float = DEFAULT_TOL_REL
    floor_mw: float = DEFAULT_FLOOR_MW
    floor_mwh: float = DEFAULT_FLOOR_MWH
    seed_power_mw: float = DEFAULT_INITIAL_POWER_MW
    seed_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS
    max_cpu_time: float = 120.0
    dual_bound_scale: float = 10.0
    max_consecutive_failures: int = 3


@dataclass
class BestResponse:
    investor_id: str
    termination: str
    solve_seconds: float
    proposed_power: dict[str, float]  # node -> MW
    proposed_energy: dict[str, float]  # node -> MWh
    profit_belief_eur_per_day: float
    strong_duality_gap: float
    model: pyo.ConcreteModel | None

    @property
    def ok(self) -> bool:
        return self.termination == "optimal"


@dataclass
class EpecState:
    x_power: dict[tuple[str, str], float]  # (investor_id, node) -> MW, damped iterate
    x_energy: dict[tuple[str, str], float]  # (investor_id, node) -> MWh
    iteration: int = 0
    converged: bool = False
    stop_reason: str = ""
    history: list[dict] = field(default_factory=list)  # one row per (iteration, investor)
    trajectory: list[dict] = field(default_factory=list)  # one row per (iteration, investor, node)
    projection_events: list[dict] = field(default_factory=list)
    final_models: dict[str, pyo.ConcreteModel] = field(default_factory=dict)


def aggregate_rival_capacity(
    state: EpecState, cfg: EpecConfig, nodes: list[str], active_id: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Sum all other investors' capacities into one fixed rival fleet per node."""

    rival_power = {n: 0.0 for n in nodes}
    rival_energy = {n: 0.0 for n in nodes}
    for inv in cfg.investors:
        if inv.investor_id == active_id:
            continue
        for n in nodes:
            rival_power[n] += state.x_power[inv.investor_id, n]
            rival_energy[n] += state.x_energy[inv.investor_id, n]
    # Numerical guard: post-projection sums can sit epsilon above the limit.
    for n in nodes:
        rival_power[n] = min(rival_power[n], cfg.node_limit_mw)
    return rival_power, rival_energy


def solve_best_response(
    data: MarketData,
    quad: QuadraticDemandCurve,
    cfg: EpecConfig,
    investor: InvestorConfig,
    rival_power: dict[str, float],
    rival_energy: dict[str, float],
    x_prev_power: dict[str, float],
    x_prev_energy: dict[str, float],
    tee: bool = False,
) -> BestResponse:
    """One investor's MPEC against the fixed rival fleet, warm-started from its previous iterate."""

    def attempt(shrink: float) -> tuple[pyo.ConcreteModel, str, float]:
        model = build_single_investor_mpec(
            data,
            quad_demand=quad,
            investor=investor,
            rival_id=RIVAL_ID,
            rival_power_mw=rival_power,
            rival_energy_mwh=rival_energy,
            node_limit_mw=cfg.node_limit_mw,
            dual_bound_scale=cfg.dual_bound_scale,
            initial_power_mw=cfg.seed_power_mw,
            initial_ratio_hours=cfg.seed_ratio_hours,
        )
        for n in model.N:
            headroom = cfg.node_limit_mw - rival_power[n]
            power = min(max(0.0, shrink * x_prev_power[n]), headroom)
            energy = min(
                max(investor.ratio_min * power, shrink * x_prev_energy[n]),
                investor.ratio_max * headroom,
            )
            model.X_power[n].set_value(power)
            model.X_energy[n].set_value(energy)
        initialize_from_reference_dispatch(model, data, cfg.seed_ratio_hours)
        start = time.perf_counter()
        try:
            results = get_ipopt_solver({"max_cpu_time": cfg.max_cpu_time}).solve(model, tee=tee)
            termination = str(results.solver.termination_condition)
        except (ValueError, RuntimeError) as exc:
            # Pyomo raises instead of returning when Ipopt exits with status
            # "error" (e.g. restoration failure); treat it as a failed attempt.
            termination = f"solver_exception: {type(exc).__name__}"
        seconds = time.perf_counter() - start
        return model, termination, seconds

    model, termination, seconds = attempt(shrink=1.0)
    if termination != "optimal":
        model, termination, retry_seconds = attempt(shrink=0.9)
        seconds += retry_seconds
    if termination != "optimal":
        return BestResponse(
            investor_id=investor.investor_id,
            termination=termination,
            solve_seconds=seconds,
            proposed_power=dict(x_prev_power),
            proposed_energy=dict(x_prev_energy),
            profit_belief_eur_per_day=float("nan"),
            strong_duality_gap=float("nan"),
            model=None,
        )
    return BestResponse(
        investor_id=investor.investor_id,
        termination=termination,
        solve_seconds=seconds,
        proposed_power={n: max(0.0, value(model.X_power[n])) for n in model.N},
        proposed_energy={n: max(0.0, value(model.X_energy[n])) for n in model.N},
        profit_belief_eur_per_day=value(model.investor_profit_expr),
        strong_duality_gap=abs(value(model.primal_objective_expr) - value(model.dual_objective_expr)),
        model=model,
    )


def apply_damped_update(
    state: EpecState, cfg: EpecConfig, nodes: list[str], response: BestResponse
) -> None:
    a = cfg.damping
    inv_id = response.investor_id
    for n in nodes:
        state.x_power[inv_id, n] = (1.0 - a) * state.x_power[inv_id, n] + a * response.proposed_power[n]
        state.x_energy[inv_id, n] = (1.0 - a) * state.x_energy[inv_id, n] + a * response.proposed_energy[n]


def project_joint_limit(state: EpecState, cfg: EpecConfig, nodes: list[str]) -> None:
    """Scale capacities down where the joint nodal sum exceeds the connection limit.

    Power and energy are scaled by the same factor, preserving each investor's
    E/P ratio. Every activation is recorded: projection frequency measures how
    contested a node is under the chosen update rule and damping.
    """

    for n in nodes:
        total = sum(state.x_power[inv.investor_id, n] for inv in cfg.investors)
        if total <= cfg.node_limit_mw + 1e-6:
            continue
        scale = cfg.node_limit_mw / total
        for inv in cfg.investors:
            state.x_power[inv.investor_id, n] *= scale
            state.x_energy[inv.investor_id, n] *= scale
        state.projection_events.append(
            {"iteration": state.iteration, "node": n, "total_before_mw": total, "scale": scale}
        )
        print(f"  [projection] iter {state.iteration}, node {n}: {total:.3f} MW -> {cfg.node_limit_mw:.1f} MW")


def relative_delta(new: float, old: float, floor: float) -> float:
    return abs(new - old) / max(abs(old), floor)


def run_epec(data: MarketData, quad: QuadraticDemandCurve, cfg: EpecConfig, tee: bool = False) -> EpecState:
    nodes = list(data.nodes)
    n_inv = len(cfg.investors)
    seed = min(cfg.seed_power_mw, cfg.node_limit_mw / n_inv)
    state = EpecState(
        x_power={(inv.investor_id, n): seed for inv in cfg.investors for n in nodes},
        x_energy={(inv.investor_id, n): seed * cfg.seed_ratio_hours for inv in cfg.investors for n in nodes},
    )
    consecutive_failures = {inv.investor_id: 0 for inv in cfg.investors}

    for iteration in range(1, cfg.max_iters + 1):
        state.iteration = iteration
        x_power_start = dict(state.x_power)
        x_energy_start = dict(state.x_energy)
        responses: list[BestResponse] = []

        if cfg.update_rule == "jacobi":
            snapshot = EpecState(x_power=dict(state.x_power), x_energy=dict(state.x_energy))
            for inv in cfg.investors:
                rival_power, rival_energy = aggregate_rival_capacity(snapshot, cfg, nodes, inv.investor_id)
                responses.append(
                    solve_best_response(
                        data, quad, cfg, inv, rival_power, rival_energy,
                        {n: snapshot.x_power[inv.investor_id, n] for n in nodes},
                        {n: snapshot.x_energy[inv.investor_id, n] for n in nodes},
                        tee=tee,
                    )
                )
            for response in responses:
                if response.ok:
                    apply_damped_update(state, cfg, nodes, response)
        elif cfg.update_rule == "seidel":
            for inv in cfg.investors:
                rival_power, rival_energy = aggregate_rival_capacity(state, cfg, nodes, inv.investor_id)
                response = solve_best_response(
                    data, quad, cfg, inv, rival_power, rival_energy,
                    {n: state.x_power[inv.investor_id, n] for n in nodes},
                    {n: state.x_energy[inv.investor_id, n] for n in nodes},
                    tee=tee,
                )
                responses.append(response)
                if response.ok:
                    apply_damped_update(state, cfg, nodes, response)
        else:
            raise ValueError(f"Unknown update rule: {cfg.update_rule}")

        project_joint_limit(state, cfg, nodes)

        all_ok = all(r.ok for r in responses)
        max_rel_power = 0.0
        max_rel_energy = 0.0
        for response in responses:
            inv_id = response.investor_id
            if response.ok:
                consecutive_failures[inv_id] = 0
            else:
                consecutive_failures[inv_id] += 1
            rel_power = max(
                relative_delta(state.x_power[inv_id, n], x_power_start[inv_id, n], cfg.floor_mw) for n in nodes
            )
            rel_energy = max(
                relative_delta(state.x_energy[inv_id, n], x_energy_start[inv_id, n], cfg.floor_mwh) for n in nodes
            )
            undamped_power = max(abs(response.proposed_power[n] - x_power_start[inv_id, n]) for n in nodes)
            max_rel_power = max(max_rel_power, rel_power)
            max_rel_energy = max(max_rel_energy, rel_energy)
            state.history.append(
                {
                    "iteration": iteration,
                    "investor": inv_id,
                    "termination": response.termination,
                    "solve_seconds": response.solve_seconds,
                    "profit_belief_eur_per_day": response.profit_belief_eur_per_day,
                    "strong_duality_gap": response.strong_duality_gap,
                    "total_power_mw": sum(state.x_power[inv_id, n] for n in nodes),
                    "total_energy_mwh": sum(state.x_energy[inv_id, n] for n in nodes),
                    "max_rel_delta_power": rel_power,
                    "max_rel_delta_energy": rel_energy,
                    "max_undamped_delta_power_mw": undamped_power,
                }
            )
            for n in nodes:
                state.trajectory.append(
                    {
                        "iteration": iteration,
                        "investor": inv_id,
                        "node": n,
                        "x_power_mw": state.x_power[inv_id, n],
                        "x_energy_mwh": state.x_energy[inv_id, n],
                        "proposed_x_power_mw": response.proposed_power[n],
                        "headroom_mw": cfg.node_limit_mw - sum(state.x_power[j.investor_id, n] for j in cfg.investors),
                    }
                )

        beliefs = ", ".join(
            f"{r.investor_id}={r.profit_belief_eur_per_day:,.0f}" if r.ok else f"{r.investor_id}=FAILED"
            for r in responses
        )
        print(
            f"iter {iteration:2d} [{cfg.update_rule}] max_rel dP={max_rel_power:.4f} dE={max_rel_energy:.4f}"
            f"  beliefs [EUR/day]: {beliefs}"
        )

        if any(count >= cfg.max_consecutive_failures for count in consecutive_failures.values()):
            state.stop_reason = "aborted: repeated MPEC solve failures"
            break
        if all_ok and max_rel_power < cfg.tol_rel and max_rel_energy < cfg.tol_rel:
            state.converged = True
            state.stop_reason = f"converged in {iteration} iterations"
            state.final_models = {r.investor_id: r.model for r in responses if r.model is not None}
            break
    else:
        state.stop_reason = f"max iterations ({cfg.max_iters}) reached without convergence"

    if not state.final_models:
        state.final_models = {r.investor_id: r.model for r in responses if r.model is not None}
    print(state.stop_reason)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-investor BESS EPEC via diagonalization")
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--update-rule", choices=["jacobi", "seidel"], default="jacobi")
    parser.add_argument(
        "--wacc",
        type=float,
        nargs="+",
        default=[0.08, 0.12],
        help="One WACC per investor; investors are named I1, I2, ... in this (Seidel solve) order.",
    )
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    parser.add_argument("--floor-mw", type=float, default=DEFAULT_FLOOR_MW)
    parser.add_argument("--floor-mwh", type=float, default=DEFAULT_FLOOR_MWH)
    parser.add_argument("--seed-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--seed-ratio-hours", type=float, default=DEFAULT_INITIAL_RATIO_HOURS)
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument("--dual-bound-scale", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Optional label appended to the output folder name.")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    investors = tuple(
        InvestorConfig(investor_id=f"I{k + 1}", wacc=wacc) for k, wacc in enumerate(args.wacc)
    )
    cfg = EpecConfig(
        investors=investors,
        update_rule=args.update_rule,
        damping=args.damping,
        max_iters=args.max_iters,
        tol_rel=args.tol_rel,
        floor_mw=args.floor_mw,
        floor_mwh=args.floor_mwh,
        seed_power_mw=args.seed_power_mw,
        seed_ratio_hours=args.seed_ratio_hours,
        max_cpu_time=args.max_cpu_time,
        dual_bound_scale=args.dual_bound_scale,
    )
    data = load_market_data(args.data)
    quad = default_quadratic_demand_curve()
    print(
        f"EPEC diagonalization: {len(investors)} investors "
        f"(WACC {', '.join(f'{i.wacc:.1%}' for i in investors)}), "
        f"rule={cfg.update_rule}, damping={cfg.damping}, tol_rel={cfg.tol_rel}"
    )
    print(
        "Quadratic demand curve: "
        f"marginal WTP = {quad.alpha:,.2f} + {quad.beta:,.2f} * curtailed_share EUR/MWh"
    )

    state = run_epec(data, quad, cfg, tee=args.tee)

    from epec_results import compute_joint_settlement, export_epec_results, print_epec_summary

    settlement = compute_joint_settlement(data, quad, state, cfg)
    print_epec_summary(state, cfg, settlement)
    if not args.no_export:
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            name = cfg.update_rule + (f"_{args.tag}" if args.tag else "")
            output_dir = DEFAULT_OUTPUT_ROOT / name
        export_epec_results(output_dir, data, state, cfg, settlement, args.data)
        print(f"\nWrote EPEC outputs to {output_dir}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
