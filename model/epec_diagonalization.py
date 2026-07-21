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
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

# Wind-vs-solar tilt for the two renewable-portfolio investors: the dominant
# technology's rent share, the minor technology gets 1 - this. Shares sum to
# 1.0 per generator across the two portfolios, so all existing RES rent is
# allocated and none is double-counted.
PORTFOLIO_MAJORITY_SHARE = 0.8


def four_investor_portfolio_profiles(data: MarketData) -> tuple[InvestorConfig, ...]:
    """Four heterogeneous investors for the portfolio EPEC on 9-bus-style data.

    I1, I2: stand-alone merchant BESS (no generation), 8% and 12% WACC.
    I3, I4: 8% WACC renewable-portfolio BESS investors that differ only by a
    wind-vs-solar ownership tilt. I3 is wind-heavy, I4 is solar-heavy; each also
    earns the inframarginal spot rent of its owned share of the existing wind/PV
    fleet, so the two same-WACC portfolios face genuinely different economics.
    """

    wind = [g for g in data.generators if "Wind" in g]
    solar = [g for g in data.generators if "PV" in g]
    if not wind or not solar:
        raise SystemExit(
            "portfolio4 investor set needs both wind and PV generators in the data "
            f"(found wind={wind}, PV={solar})."
        )
    major = PORTFOLIO_MAJORITY_SHARE
    minor = 1.0 - major
    wind_heavy = {**{g: major for g in wind}, **{g: minor for g in solar}}
    solar_heavy = {**{g: minor for g in wind}, **{g: major for g in solar}}
    return (
        InvestorConfig(investor_id="I1", wacc=0.08),
        InvestorConfig(investor_id="I2", wacc=0.12),
        InvestorConfig(investor_id="I3", wacc=0.08, owned_generation_shares=wind_heavy),
        InvestorConfig(investor_id="I4", wacc=0.08, owned_generation_shares=solar_heavy),
    )

# Settlement price basis for investor revenue (drives BOTH the MPEC objective
# and the final settlement, so it changes siting, not just reported profit):
#   False -> nodal LMP: each investor is paid the locational price lam[n,t].
#   True  -> uniform system price: investors optimize and settle at lam_sys[t],
#            i.e. a single bidding-zone / zonal market that ignores congestion.
# Flip this for a zonal-pricing run, or override per-run with
# --settlement-price {nodal,system} on the CLI.
SYSTEM_PRICE_SETTLEMENT = False

DEFAULT_DAMPING = 0.7
DEFAULT_TOL_REL = 0.02
DEFAULT_FLOOR_MW = 1.0
DEFAULT_FLOOR_MWH = 2.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "epec"


@dataclass(frozen=True)
class EpecConfig:
    investors: tuple[InvestorConfig, ...]
    node_limit_mw: float = DEFAULT_NODE_LIMIT_MW
    update_rule: str = "seidel"  # "jacobi" | "seidel" (solve order = investors order)
    damping: float = DEFAULT_DAMPING  # x' = (1-a)*x_old + a*x_best_response
    max_iters: int = 60
    tol_rel: float = DEFAULT_TOL_REL
    floor_mw: float = DEFAULT_FLOOR_MW
    floor_mwh: float = DEFAULT_FLOOR_MWH
    seed_power_mw: float = DEFAULT_INITIAL_POWER_MW
    seed_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS
    max_cpu_time: float = 500.0
    dual_bound_scale: float = 10.0
    max_consecutive_failures: int = 3
    print_mpec_lambdas: bool = False
    system_price_settlement: bool = SYSTEM_PRICE_SETTLEMENT
    # "projection": today's mechanism -- private per-investor headroom bound,
    # joint sum clipped back onto node_limit_mw after each pass.
    # "price": nodal access pricing -- the bound is dropped, investors pay
    # capacity_price[n] per MW, and one shared price per node is found by an
    # projected dual-price update so that aggregate demand respects
    # node_limit_mw and complementary slackness holds at every node.
    allocation_mechanism: str = "projection"  # "projection" | "price"
    price_step_eur_per_mw: float = 0.05  # EUR/MW/day added per MW of node oversubscription; needs empirical tuning
    price_tol_mw: float = 1.0  # tolerance on the projected complementarity residual, expressed in MW
    starting_iteration: int = 0
    resume_from: str | None = None


@dataclass
class BestResponse:
    investor_id: str
    termination: str
    solve_seconds: float
    proposed_power: dict[str, float]  # node -> MW
    proposed_energy: dict[str, float]  # node -> MWh
    optimistic_mpec_profit_eur_per_day: float
    strong_duality_gap: float
    model: pyo.ConcreteModel | None

    @property
    def ok(self) -> bool:
        return self.termination == "optimal"


@dataclass
class EpecState:
    x_power: dict[tuple[str, str], float]  # (investor_id, node) -> MW, damped iterate
    x_energy: dict[tuple[str, str], float]  # (investor_id, node) -> MWh
    capacity_price: dict[str, float] = field(default_factory=dict)  # node -> EUR/MW/day, "price" mechanism only
    iteration: int = 0
    converged: bool = False
    stop_reason: str = ""
    history: list[dict] = field(default_factory=list)  # one row per (iteration, investor)
    price_history: list[dict] = field(default_factory=list)  # one row per (iteration, node), "price" mechanism only
    trajectory: list[dict] = field(default_factory=list)  # one row per (iteration, investor, node)
    projection_events: list[dict] = field(default_factory=list)
    final_models: dict[str, pyo.ConcreteModel] = field(default_factory=dict)


def load_checkpoint_state(
    checkpoint_or_directory: Path,
    data: MarketData,
    cfg: EpecConfig,
) -> tuple[EpecState, Path]:
    """Restore capacities and nodal access prices from a completed-iteration checkpoint."""

    checkpoint_path = Path(checkpoint_or_directory)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint_path}")

    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if raw.get("allocation_mechanism") != cfg.allocation_mechanism:
        raise ValueError(
            "Resume checkpoint allocation mechanism does not match this run "
            f"({raw.get('allocation_mechanism')!r} != {cfg.allocation_mechanism!r})."
        )
    if abs(float(raw.get("node_limit_mw", float("nan"))) - cfg.node_limit_mw) > 1e-9:
        raise ValueError("Resume checkpoint node limit does not match this run.")

    investor_ids = {inv.investor_id for inv in cfg.investors}
    nodes = set(data.nodes)
    expected = {(investor_id, node) for investor_id in investor_ids for node in nodes}

    def read_capacity(field_name: str) -> dict[tuple[str, str], float]:
        restored: dict[tuple[str, str], float] = {}
        for compound_key, value_ in raw[field_name].items():
            investor_id, separator, node = compound_key.partition("|")
            if not separator:
                raise ValueError(f"Invalid checkpoint capacity key: {compound_key!r}")
            restored[investor_id, node] = float(value_)
        if set(restored) != expected:
            missing = sorted(expected - set(restored))
            extra = sorted(set(restored) - expected)
            raise ValueError(f"Checkpoint capacity keys do not match this run (missing={missing}, extra={extra}).")
        return restored

    capacity_price = {node: float(value_) for node, value_ in raw["capacity_price_eur_per_mw"].items()}
    if set(capacity_price) != nodes:
        raise ValueError("Checkpoint nodal capacity-price keys do not match the dataset nodes.")

    state = EpecState(
        x_power=read_capacity("x_power_mw"),
        x_energy=read_capacity("x_energy_mwh"),
        capacity_price=capacity_price,
        iteration=int(raw["iteration"]),
    )
    return state, checkpoint_path.resolve()


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
    if cfg.allocation_mechanism == "projection":
        # Numerical guard: post-projection sums can sit epsilon above the limit.
        for n in nodes:
            rival_power[n] = min(rival_power[n], cfg.node_limit_mw)
    # Under "price", nodes can transiently sit above node_limit_mw while the
    # price tâtonnement is still converging; rival_power must reflect the real
    # (uncapped) installed capacity for the lower-level dispatch physics.
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
    capacity_price: dict[str, float] | None = None,
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
            system_price_settlement=cfg.system_price_settlement,
            capacity_price_eur_per_mw_day=capacity_price,
        )
        for n in model.N:
            # Under "price" the technical cap is node_limit_mw itself (no rival
            # deduction); under "projection" it is the private headroom. Either
            # way this only seeds Ipopt's starting point, never the real bound.
            cap = cfg.node_limit_mw if capacity_price is not None else max(0.0, cfg.node_limit_mw - rival_power[n])
            power = min(max(0.0, shrink * x_prev_power[n]), cap)
            energy = min(
                max(investor.ratio_min * power, shrink * x_prev_energy[n]),
                investor.ratio_max * cap,
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
            optimistic_mpec_profit_eur_per_day=float("nan"),
            strong_duality_gap=float("nan"),
            model=None,
        )
    return BestResponse(
        investor_id=investor.investor_id,
        termination=termination,
        solve_seconds=seconds,
        proposed_power={n: max(0.0, value(model.X_power[n])) for n in model.N},
        proposed_energy={n: max(0.0, value(model.X_energy[n])) for n in model.N},
        optimistic_mpec_profit_eur_per_day=value(model.investor_profit_expr),
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


def update_capacity_price(
    state: EpecState, cfg: EpecConfig, nodes: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """One subgradient step on the shared nodal capacity price.

    Raises the price where aggregate installed power exceeds node_limit_mw,
    lowers it (never below zero) where the node has slack. This is the price
    counterpart of ``project_joint_limit``: instead of clipping capacities
    back onto the limit after the fact, it nudges the common cost signal every
    investor's MPEC sees, so the limit is approached from the demand side.
    Returns both signed nodal excess (installed minus limit, MW) and the
    projected fixed-point residual

        |lambda - projection_+(lambda + step * excess)| / step.

    This residual is zero both at a binding node with a positive clearing
    price and at a slack node whose clearing price is zero. Unlike
    ``abs(excess)``, it therefore represents the intended complementary-
    slackness condition rather than incorrectly requiring every node to fill.
    """

    excess: dict[str, float] = {}
    projected_residual: dict[str, float] = {}
    for n in nodes:
        total = sum(state.x_power[inv.investor_id, n] for inv in cfg.investors)
        excess[n] = total - cfg.node_limit_mw
        old_price = state.capacity_price[n]
        new_price = max(0.0, old_price + cfg.price_step_eur_per_mw * excess[n])
        projected_residual[n] = abs(new_price - old_price) / cfg.price_step_eur_per_mw
        state.price_history.append(
            {
                "iteration": state.iteration,
                "node": n,
                "previous_capacity_price_eur_per_mw": old_price,
                "capacity_price_eur_per_mw": new_price,
                "capacity_price_change_eur_per_mw": new_price - old_price,
                "total_power_mw": total,
                "excess_mw": excess[n],
                "overload_mw": max(0.0, excess[n]),
                "headroom_mw": max(0.0, -excess[n]),
                "projected_complementarity_residual_mw": projected_residual[n],
            }
        )
        state.capacity_price[n] = new_price
    return excess, projected_residual


def relative_delta(new: float, old: float, floor: float) -> float:
    return abs(new - old) / max(abs(old), floor)


def print_mpec_lambdas(iteration: int, response: BestResponse) -> None:
    """Print embedded MPEC nodal prices for one solved best response."""

    if response.model is None:
        print(f"\niter {iteration}, {response.investor_id}: no MPEC lambdas ({response.termination})")
        return

    model = response.model
    print(f"\niter {iteration}, {response.investor_id}: embedded MPEC lambdas [EUR/MWh]")
    for t in model.T:
        parts = ", ".join(f"{n}={value(model.lam[n, t]):10.4f}" for n in model.N)
        print(f"  hour={int(t):2d}: {parts}")


def run_epec(
    data: MarketData,
    quad: QuadraticDemandCurve,
    cfg: EpecConfig,
    tee: bool = False,
    checkpoint_callback: Callable[[EpecState], None] | None = None,
    initial_state: EpecState | None = None,
) -> EpecState:
    if cfg.allocation_mechanism not in {"projection", "price"}:
        raise ValueError(f"Unknown allocation mechanism: {cfg.allocation_mechanism}")
    if cfg.allocation_mechanism == "price" and cfg.price_step_eur_per_mw <= 0.0:
        raise ValueError("price_step_eur_per_mw must be positive in price mode.")
    if cfg.price_tol_mw < 0.0:
        raise ValueError("price_tol_mw must be nonnegative.")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    nodes = list(data.nodes)
    n_inv = len(cfg.investors)
    if initial_state is None:
        seed = min(cfg.seed_power_mw, cfg.node_limit_mw / n_inv)
        state = EpecState(
            x_power={(inv.investor_id, n): seed for inv in cfg.investors for n in nodes},
            x_energy={(inv.investor_id, n): seed * cfg.seed_ratio_hours for inv in cfg.investors for n in nodes},
            capacity_price={n: 0.0 for n in nodes},
        )
    else:
        state = initial_state
    consecutive_failures = {inv.investor_id: 0 for inv in cfg.investors}
    final_iteration = state.iteration + cfg.max_iters

    for iteration in range(state.iteration + 1, final_iteration + 1):
        state.iteration = iteration
        x_power_start = dict(state.x_power)
        x_energy_start = dict(state.x_energy)
        responses: list[BestResponse] = []

        capacity_price = state.capacity_price if cfg.allocation_mechanism == "price" else None

        if cfg.update_rule == "jacobi":
            snapshot = EpecState(x_power=dict(state.x_power), x_energy=dict(state.x_energy))
            for inv in cfg.investors:
                rival_power, rival_energy = aggregate_rival_capacity(snapshot, cfg, nodes, inv.investor_id)
                responses.append(
                    solve_best_response(
                        data, quad, cfg, inv, rival_power, rival_energy,
                        {n: snapshot.x_power[inv.investor_id, n] for n in nodes},
                        {n: snapshot.x_energy[inv.investor_id, n] for n in nodes},
                        capacity_price=capacity_price,
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
                    capacity_price=capacity_price,
                    tee=tee,
                )
                responses.append(response)
                if response.ok:
                    apply_damped_update(state, cfg, nodes, response)
        else:
            raise ValueError(f"Unknown update rule: {cfg.update_rule}")

        if cfg.print_mpec_lambdas:
            for response in responses:
                print_mpec_lambdas(iteration, response)

        if cfg.allocation_mechanism == "projection":
            excess = None
            projected_price_residual = None
            project_joint_limit(state, cfg, nodes)
        else:
            excess, projected_price_residual = update_capacity_price(state, cfg, nodes)

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
                    "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
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

        optimistic = ", ".join(
            f"{r.investor_id}={r.optimistic_mpec_profit_eur_per_day:,.0f}" if r.ok else f"{r.investor_id}=FAILED"
            for r in responses
        )
        price_ok = True
        price_note = ""
        if excess is not None:
            max_overload = max(max(0.0, v) for v in excess.values())
            max_price_residual = max(projected_price_residual.values())
            price_ok = max_price_residual <= cfg.price_tol_mw
            price_note = (
                f"  max overload={max_overload:.3f} MW"
                f"  max price-KKT residual={max_price_residual:.3f} MW"
            )
        print(
            f"iter {iteration:2d} [{cfg.update_rule}] max_rel dP={max_rel_power:.4f} dE={max_rel_energy:.4f}"
            f"{price_note}  optimistic MPEC profit [EUR/day]: {optimistic}"
        )

        should_stop = False
        if any(count >= cfg.max_consecutive_failures for count in consecutive_failures.values()):
            state.stop_reason = "aborted: repeated MPEC solve failures"
            should_stop = True
        elif all_ok and max_rel_power < cfg.tol_rel and max_rel_energy < cfg.tol_rel and price_ok:
            state.converged = True
            state.stop_reason = f"converged in {iteration} iterations"
            state.final_models = {r.investor_id: r.model for r in responses if r.model is not None}
            should_stop = True

        if checkpoint_callback is not None:
            checkpoint_callback(state)
        if should_stop:
            break
    else:
        state.stop_reason = f"max iterations ({final_iteration}) reached without convergence"

    if not state.final_models:
        state.final_models = {r.investor_id: r.model for r in responses if r.model is not None}
    print(state.stop_reason)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-investor BESS EPEC via diagonalization")
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--update-rule", choices=["jacobi", "seidel"], default="jacobi")
    parser.add_argument(
        "--investor-set",
        choices=["wacc", "portfolio4"],
        default="wacc",
        help="'wacc' (default): homogeneous investors from --wacc. 'portfolio4': four "
        "heterogeneous investors (two merchants + two same-WACC wind/solar-tilted RES portfolios).",
    )
    parser.add_argument(
        "--wacc",
        type=float,
        nargs="+",
        default=[0.08, 0.12],
        help="One WACC per investor (only used when --investor-set wacc); investors are "
        "named I1, I2, ... in this (Seidel solve) order.",
    )
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument(
        "--node-limit-mw",
        type=float,
        default=DEFAULT_NODE_LIMIT_MW,
        help="Shared BESS power connection limit per node (sum over investors).",
    )
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    parser.add_argument("--floor-mw", type=float, default=DEFAULT_FLOOR_MW)
    parser.add_argument("--floor-mwh", type=float, default=DEFAULT_FLOOR_MWH)
    parser.add_argument("--seed-power-mw", type=float, default=DEFAULT_INITIAL_POWER_MW)
    parser.add_argument("--seed-ratio-hours", type=float, default=DEFAULT_INITIAL_RATIO_HOURS)
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument("--dual-bound-scale", type=float, default=10.0)
    parser.add_argument(
        "--allocation-mechanism",
        choices=["projection", "price"],
        default="projection",
        help="'projection' (default): private per-investor headroom bound, joint sum clipped "
        "onto node_limit_mw after each pass. 'price': nodal access pricing -- the bound is "
        "dropped and one shared EUR/MW/day price per node is found by an outer subgradient "
        "search so aggregate demand clears node_limit_mw.",
    )
    parser.add_argument(
        "--price-step-eur-per-mw",
        type=float,
        default=0.05,
        help="Subgradient step size for --allocation-mechanism price: EUR/MW/day added to a "
        "node's price per MW of oversubscription. Needs empirical tuning, same as --damping.",
    )
    parser.add_argument(
        "--price-tol-mw",
        type=float,
        default=1.0,
        help="Convergence tolerance for --allocation-mechanism price: maximum projected "
        "complementarity residual per node, MW. A slack node passes when its price is zero.",
    )
    parser.add_argument(
        "--print-mpec-lambdas",
        action="store_true",
        help="Print embedded MPEC nodal prices for every solved investor best response.",
    )
    parser.add_argument(
        "--settlement-price",
        choices=["nodal", "system"],
        default=None,
        help="Price basis for investor revenue (MPEC objective + settlement). "
        f"Default follows the SYSTEM_PRICE_SETTLEMENT toggle "
        f"({'system' if SYSTEM_PRICE_SETTLEMENT else 'nodal'}).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Optional label appended to the output folder name.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume MW/MWh capacities and nodal access prices from a checkpoint.json file "
        "or a directory containing it. --max-iters then means additional iterations.",
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    if args.allocation_mechanism == "price" and args.price_step_eur_per_mw <= 0.0:
        raise SystemExit("--price-step-eur-per-mw must be positive in price mode.")
    if args.price_tol_mw < 0.0:
        raise SystemExit("--price-tol-mw must be nonnegative.")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive.")
    data = load_market_data(args.data)
    if args.investor_set == "portfolio4":
        investors = four_investor_portfolio_profiles(data)
    else:
        investors = tuple(
            InvestorConfig(investor_id=f"I{k + 1}", wacc=wacc) for k, wacc in enumerate(args.wacc)
        )
    if args.settlement_price is None:
        system_price_settlement = SYSTEM_PRICE_SETTLEMENT
    else:
        system_price_settlement = args.settlement_price == "system"
    cfg = EpecConfig(
        investors=investors,
        node_limit_mw=args.node_limit_mw,
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
        print_mpec_lambdas=args.print_mpec_lambdas,
        system_price_settlement=system_price_settlement,
        allocation_mechanism=args.allocation_mechanism,
        price_step_eur_per_mw=args.price_step_eur_per_mw,
        price_tol_mw=args.price_tol_mw,
    )
    initial_state = None
    if args.resume_from is not None:
        try:
            initial_state, checkpoint_path = load_checkpoint_state(args.resume_from, data, cfg)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot resume EPEC run: {exc}") from exc
        cfg = replace(
            cfg,
            starting_iteration=initial_state.iteration,
            resume_from=str(checkpoint_path),
        )
    quad = default_quadratic_demand_curve()
    print(
        f"EPEC diagonalization: {len(investors)} investors "
        f"(WACC {', '.join(f'{i.wacc:.1%}' for i in investors)}), "
        f"rule={cfg.update_rule}, damping={cfg.damping}, tol_rel={cfg.tol_rel}, "
        f"settlement price={'system (zonal)' if cfg.system_price_settlement else 'nodal (LMP)'}, "
        f"allocation={cfg.allocation_mechanism}, "
        "dual_selection=optimistic"
    )
    for inv in investors:
        if inv.owned_generation_shares:
            owned = ", ".join(f"{g}={s:.2f}" for g, s in inv.owned_generation_shares.items())
            print(f"  {inv.investor_id}: portfolio-backed, generation shares [{owned}]")
        else:
            print(f"  {inv.investor_id}: stand-alone merchant BESS")
    print(
        "Quadratic demand curve: "
        f"marginal WTP = {quad.alpha:,.2f} + {quad.beta:,.2f} * curtailed_share EUR/MWh"
    )
    if initial_state is not None:
        print(
            f"Resuming from iteration {initial_state.iteration}; "
            f"running up to {cfg.max_iters} additional iterations from {cfg.resume_from}"
        )

    output_dir = None
    checkpoint_callback = None
    if not args.no_export:
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            name = cfg.update_rule + (f"_{args.tag}" if args.tag else "")
            output_dir = DEFAULT_OUTPUT_ROOT / name
        from epec_results import export_epec_checkpoint

        checkpoint_callback = lambda current_state: export_epec_checkpoint(output_dir, current_state, cfg)
        print(f"Iteration checkpoints will be written to {output_dir}")

    state = run_epec(
        data,
        quad,
        cfg,
        tee=args.tee,
        checkpoint_callback=checkpoint_callback,
        initial_state=initial_state,
    )

    from epec_results import compute_joint_settlement, export_epec_results, print_epec_summary

    settlement = compute_joint_settlement(data, quad, state, cfg)
    print_epec_summary(state, cfg, settlement)
    if output_dir is not None:
        export_epec_results(output_dir, data, state, cfg, settlement, args.data)
        print(f"\nWrote EPEC outputs to {output_dir}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
