"""Multi-investor spot-market EPEC solved by diagonalization.

Each strategic BESS investor solves the single-investor MPEC while every rival
is frozen as a separate non-strategic storage unit with its own nodal MW/MWh
capacities inside the lower-level clearing. The shared nodal connection limit couples the
investors, so the solution concept is a generalized Nash equilibrium and the
outcome may depend on the update rule: Gauss-Jacobi (all investors respond to
the same previous iterate) versus Gauss-Seidel (sequential, later investors
see earlier same-iteration updates - the potential first-mover artifact).

By default a fresh Gauss-Seidel run first performs one common-snapshot Jacobi
best-response sweep and proportionally projects only overloaded nodes. That
feasible projected fleet is iteration 0 of the subsequent Seidel loop. Resumed
runs continue directly from their checkpoint without repeating initialization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyomo.environ as pyo

# The maintained primal/dual market modules currently live in this subfolder.
_MODEL_DIR = Path(__file__).resolve().parent
_PRIMAL_DUAL_DIR = _MODEL_DIR / "Primal and dual problems"
if _PRIMAL_DUAL_DIR.is_dir() and str(_PRIMAL_DUAL_DIR) not in sys.path:
    sys.path.append(str(_PRIMAL_DUAL_DIR))

from primal_market_clearing_model import MarketData, load_market_data, value
from single_investor_mpec import (
    DEFAULT_DUAL_BOUND_EUR_PER_MWH,
    DEFAULT_INITIAL_POWER_MW,
    DEFAULT_INITIAL_RATIO_HOURS,
    DEFAULT_NODE_LIMIT_MW,
    DEFAULT_PRICE_BOUND_EUR_PER_MWH,
    DEFAULT_SOLVER_TOL,
    EXPERIMENT_DATA_PATH,
    InvestorConfig,
    QuadraticDemandCurve,
    build_single_investor_mpec,
    default_quadratic_demand_curve,
    initialize_from_reference_dispatch,
    investment_headroom_shadow_price,
)
from solver_utils import get_ipopt_solver


#### Constants and investor profiles
# -----------------------------------------------------------------------------

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


def order_investors(
    investors: tuple[InvestorConfig, ...], requested_order: list[str] | None
) -> tuple[InvestorConfig, ...]:
    """Return investors in an explicitly requested Gauss-Seidel solve order."""

    if requested_order is None:
        return investors
    configured_ids = [investor.investor_id for investor in investors]
    if (
        len(requested_order) != len(configured_ids)
        or len(set(requested_order)) != len(requested_order)
        or set(requested_order) != set(configured_ids)
    ):
        raise ValueError(
            "--investor-order must list every configured investor exactly once; "
            f"expected {configured_ids}, received {requested_order}."
        )
    by_id = {investor.investor_id: investor for investor in investors}
    return tuple(by_id[investor_id] for investor_id in requested_order)

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
DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH = 1.0e-4
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "epec"


#### Configuration and state
# -----------------------------------------------------------------------------

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
    price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH
    price_lower_bound_eur_per_mwh: float | None = None
    price_upper_bound_eur_per_mwh: float | None = None
    dual_bound_eur_per_mwh: float = DEFAULT_DUAL_BOUND_EUR_PER_MWH
    max_consecutive_failures: int = 3
    print_mpec_lambdas: bool = False
    system_price_settlement: bool = SYSTEM_PRICE_SETTLEMENT
    use_demand_curve: bool = False
    dispatch_regularization_eur_per_mw2h: float = 0.0
    solver_tol: float = DEFAULT_SOLVER_TOL
    capacity_cleanup_tol_mw_mwh: float = DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH
    automatic_jacobi_initializer: bool = True
    jacobi_initializer_snapshot_power_mw: float = 0.0
    jacobi_initializer_snapshot_ratio_hours: float = DEFAULT_INITIAL_RATIO_HOURS
    strategic_proximal_penalty_eur_per_mw2_day: float = 0.0
    strategic_proximal_energy_scale_hours: float = DEFAULT_INITIAL_RATIO_HOURS
    strategic_proximal_price_scale_eur_per_mwh: float = 10.0
    strategic_proximal_penalty_step_eur_per_mw2_day: float = 0.0
    strategic_proximal_penalty_step_iterations: int = 5
    strategic_bid_prices: bool = False
    strategic_bid_price_bound_eur_per_mwh: float = DEFAULT_PRICE_BOUND_EUR_PER_MWH
    strategic_price_floor_eur_per_mwh: float = 1.0
    strategic_epsilon_penalty: float = 0.0
    strategic_tol_abs_capacity_mw: float = 0.5
    strategic_tol_abs_offer_mw: float = 0.25
    strategic_tol_abs_price_eur_per_mwh: float = 0.5
    strategic_consecutive_converged_sweeps: int = 3
    strategic_parallel_workers: int = 1
    starting_iteration: int = 0
    resume_from: str | None = None


@dataclass
class BestResponse:
    investor_id: str
    termination: str
    solve_seconds: float
    proposed_power: dict[str, float]  # node -> MW
    proposed_energy: dict[str, float]  # node -> MWh
    private_headroom_limit_mw: dict[str, float]
    optimistic_mpec_profit_eur_per_day: float
    access_shadow_price_eur_per_mw_day: dict[str, float]
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
    initialization_method: str = "uniform_seed"
    initializer_summary: dict = field(default_factory=dict)


#### Checkpoint and resume
# -----------------------------------------------------------------------------

def load_checkpoint_state(
    checkpoint_or_directory: Path,
    data: MarketData,
    cfg: EpecConfig,
) -> tuple[EpecState, Path]:
    """Restore MW and MWh strategies from a completed-iteration checkpoint."""

    checkpoint_path = Path(checkpoint_or_directory)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint_path}")

    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
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

    state = EpecState(
        x_power=read_capacity("x_power_mw"),
        x_energy=read_capacity("x_energy_mwh"),
        iteration=int(raw["iteration"]),
        initialization_method=str(raw.get("initialization_method", "checkpoint_resume")),
        initializer_summary=dict(raw.get("initializer_summary", {})),
    )
    return state, checkpoint_path.resolve()


#### Rival aggregation and investor best responses
# -----------------------------------------------------------------------------

def separate_rival_capacities(
    state: EpecState, cfg: EpecConfig, nodes: list[str], active_id: str
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return each other investor as a distinct fixed rival battery."""

    rival_power: dict[str, dict[str, float]] = {}
    rival_energy: dict[str, dict[str, float]] = {}
    for inv in cfg.investors:
        if inv.investor_id == active_id:
            continue
        rival_power[inv.investor_id] = {
            n: max(0.0, state.x_power[inv.investor_id, n]) for n in nodes
        }
        rival_energy[inv.investor_id] = {
            n: max(0.0, state.x_energy[inv.investor_id, n]) for n in nodes
        }
    return rival_power, rival_energy


def solve_best_response(
    data: MarketData,
    quad: QuadraticDemandCurve,
    cfg: EpecConfig,
    investor: InvestorConfig,
    rival_power: dict[str, dict[str, float]],
    rival_energy: dict[str, dict[str, float]],
    x_prev_power: dict[str, float],
    x_prev_energy: dict[str, float],
    initial_guess_power: dict[str, float] | None = None,
    initial_guess_energy: dict[str, float] | None = None,
    tee: bool = False,
) -> BestResponse:
    """One investor's MPEC against the fixed rival fleet, warm-started from its previous iterate."""

    guess_power = initial_guess_power if initial_guess_power is not None else x_prev_power
    guess_energy = initial_guess_energy if initial_guess_energy is not None else x_prev_energy
    if set(guess_power) != set(data.nodes) or set(guess_energy) != set(data.nodes):
        raise ValueError("Best-response initial-guess mappings must contain every market node.")

    def attempt(shrink: float) -> tuple[pyo.ConcreteModel, str, float]:
        model = build_single_investor_mpec(
            data,
            quad_demand=quad,
            investor=investor,
            rival_power_mw_by_unit=rival_power,
            rival_energy_mwh_by_unit=rival_energy,
            rival_degradation_eur_per_mwh_by_unit={
                inv.investor_id: inv.degradation_eur_per_mwh
                for inv in cfg.investors
                if inv.investor_id != investor.investor_id
            },
            node_limit_mw=cfg.node_limit_mw,
            price_bound_eur_per_mwh=cfg.price_bound_eur_per_mwh,
            price_lower_bound_eur_per_mwh=cfg.price_lower_bound_eur_per_mwh,
            price_upper_bound_eur_per_mwh=cfg.price_upper_bound_eur_per_mwh,
            dual_bound_eur_per_mwh=cfg.dual_bound_eur_per_mwh,
            initial_power_mw=cfg.seed_power_mw,
            initial_ratio_hours=cfg.seed_ratio_hours,
            system_price_settlement=cfg.system_price_settlement,
            use_demand_curve=cfg.use_demand_curve,
            dispatch_regularization_eur_per_mw2h=cfg.dispatch_regularization_eur_per_mw2h,
            solver_tol=cfg.solver_tol,
        )
        for n in model.N:
            # Seed Ipopt inside the investor's private rival-headroom bound.
            # The actual bound remains part of the MPEC.
            cap = max(0.0, cfg.node_limit_mw - sum(unit[n] for unit in rival_power.values()))
            power = min(max(0.0, shrink * guess_power[n]), cap)
            energy = min(
                max(investor.ratio_min * power, shrink * guess_energy[n]),
                investor.ratio_max * cap,
            )
            model.X_power[n].set_value(power)
            model.X_energy[n].set_value(energy)
        initialize_from_reference_dispatch(model, data, cfg.seed_ratio_hours)
        start = time.perf_counter()
        try:
            results = get_ipopt_solver(
                {
                    "max_cpu_time": cfg.max_cpu_time,
                    "tol": cfg.solver_tol,
                    "acceptable_tol": cfg.solver_tol,
                }
            ).solve(model, tee=tee)
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
            private_headroom_limit_mw={
                n: cfg.node_limit_mw - sum(unit[n] for unit in rival_power.values())
                for n in data.nodes
            },
            optimistic_mpec_profit_eur_per_day=float("nan"),
            access_shadow_price_eur_per_mw_day={n: float("nan") for n in data.nodes},
            strong_duality_gap=float("nan"),
            model=None,
        )
    return BestResponse(
        investor_id=investor.investor_id,
        termination=termination,
        solve_seconds=seconds,
        proposed_power={n: max(0.0, value(model.X_power[n])) for n in model.N},
        proposed_energy={n: max(0.0, value(model.X_energy[n])) for n in model.N},
        private_headroom_limit_mw={
            n: cfg.node_limit_mw - sum(unit[n] for unit in rival_power.values())
            for n in data.nodes
        },
        optimistic_mpec_profit_eur_per_day=value(model.investor_profit_expr),
        access_shadow_price_eur_per_mw_day={
            n: investment_headroom_shadow_price(model, n) for n in model.N
        },
        strong_duality_gap=abs(value(model.primal_objective_expr) - value(model.dual_objective_expr)),
        model=model,
    )


def apply_damped_update(
    state: EpecState, cfg: EpecConfig, nodes: list[str], response: BestResponse
) -> None:
    a = cfg.damping
    inv_id = response.investor_id
    for n in nodes:
        power = (1.0 - a) * state.x_power[inv_id, n] + a * response.proposed_power[n]
        energy = (1.0 - a) * state.x_energy[inv_id, n] + a * response.proposed_energy[n]
        state.x_power[inv_id, n], state.x_energy[inv_id, n] = clean_capacity_pair(
            power, energy, cfg.capacity_cleanup_tol_mw_mwh
        )


def clean_capacity_pair(power_mw: float, energy_mwh: float, tolerance: float) -> tuple[float, float]:
    """Normalize solver-scale capacity dust to one physically absent battery."""

    if power_mw <= tolerance and energy_mwh <= tolerance:
        return 0.0, 0.0
    return power_mw, energy_mwh


def clean_capacity_state(state: EpecState, cfg: EpecConfig, nodes: list[str]) -> None:
    for inv in cfg.investors:
        for node in nodes:
            key = (inv.investor_id, node)
            state.x_power[key], state.x_energy[key] = clean_capacity_pair(
                state.x_power[key], state.x_energy[key], cfg.capacity_cleanup_tol_mw_mwh
            )


#### Shared nodal limit
# -----------------------------------------------------------------------------

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
    clean_capacity_state(state, cfg, nodes)


def projected_jacobi_initial_state(
    data: MarketData,
    quad: QuadraticDemandCurve,
    cfg: EpecConfig,
    *,
    tee: bool = False,
) -> EpecState:
    """Create a feasible iteration-0 state from one common-snapshot Jacobi sweep.

    Every investor responds to the same economic rival-capacity snapshot. Ipopt
    is initialized separately from ``cfg.seed_power_mw`` and
    ``cfg.seed_ratio_hours``. Raw desired capacities are then proportionally
    scaled only at overloaded nodes, preserving every investor's E/P ratio.
    """

    nodes = list(data.nodes)
    snapshot_power = cfg.jacobi_initializer_snapshot_power_mw
    snapshot_ratio = cfg.jacobi_initializer_snapshot_ratio_hours
    if snapshot_power < 0.0:
        raise ValueError("Jacobi initializer snapshot power must be nonnegative.")
    if snapshot_power * len(cfg.investors) > cfg.node_limit_mw + 1e-9:
        raise ValueError("Jacobi initializer snapshot violates the shared nodal limit.")
    for investor in cfg.investors:
        if not investor.ratio_min <= snapshot_ratio <= investor.ratio_max:
            raise ValueError(
                "Jacobi initializer snapshot ratio violates an investor E/P envelope."
            )
        if not investor.ratio_min <= cfg.seed_ratio_hours <= investor.ratio_max:
            raise ValueError("Numerical seed ratio violates an investor E/P envelope.")

    snapshot = EpecState(
        x_power={
            (investor.investor_id, node): snapshot_power
            for investor in cfg.investors
            for node in nodes
        },
        x_energy={
            (investor.investor_id, node): snapshot_power * snapshot_ratio
            for investor in cfg.investors
            for node in nodes
        },
        initialization_method="jacobi_common_snapshot_projected",
    )
    guess_power = {node: cfg.seed_power_mw for node in nodes}
    guess_energy = {
        node: cfg.seed_power_mw * cfg.seed_ratio_hours for node in nodes
    }
    responses: list[BestResponse] = []
    print(
        "Automatic Jacobi initializer: common snapshot "
        f"{snapshot_power:g} MW/node, numerical guess "
        f"{cfg.seed_power_mw:g} MW/node"
    )
    for investor in cfg.investors:
        rival_power, rival_energy = separate_rival_capacities(
            snapshot, cfg, nodes, investor.investor_id
        )
        response = solve_best_response(
            data,
            quad,
            cfg,
            investor,
            rival_power,
            rival_energy,
            {node: snapshot.x_power[investor.investor_id, node] for node in nodes},
            {node: snapshot.x_energy[investor.investor_id, node] for node in nodes},
            initial_guess_power=guess_power,
            initial_guess_energy=guess_energy,
            tee=tee,
        )
        responses.append(response)
        print(
            f"  {investor.investor_id}: {response.termination}, desired "
            f"{sum(response.proposed_power.values()):.3f} MW / "
            f"{sum(response.proposed_energy.values()):.3f} MWh"
        )
    failed = [response.investor_id for response in responses if not response.ok]
    if failed:
        raise RuntimeError(
            "Automatic Jacobi initializer failed for investor(s): " + ", ".join(failed)
        )

    projected_power: dict[tuple[str, str], float] = {}
    projected_energy: dict[tuple[str, str], float] = {}
    node_summary: dict[str, dict] = {}
    for node in nodes:
        desired_total = sum(response.proposed_power[node] for response in responses)
        scale = min(1.0, cfg.node_limit_mw / desired_total) if desired_total > 0.0 else 1.0
        for response in responses:
            power, energy = clean_capacity_pair(
                scale * response.proposed_power[node],
                scale * response.proposed_energy[node],
                cfg.capacity_cleanup_tol_mw_mwh,
            )
            projected_power[response.investor_id, node] = power
            projected_energy[response.investor_id, node] = energy
        node_summary[node] = {
            "desired_total_power_mw": desired_total,
            "projection_scale": scale,
            "projected_total_power_mw": sum(
                projected_power[investor.investor_id, node]
                for investor in cfg.investors
            ),
            "limit_mw": cfg.node_limit_mw,
        }
        if scale < 1.0:
            print(
                f"  [initializer projection] {node}: {desired_total:.3f} MW "
                f"-> {cfg.node_limit_mw:.1f} MW"
            )

    response_summary = {
        response.investor_id: {
            "termination": response.termination,
            "solve_seconds": response.solve_seconds,
            "desired_power_mw": sum(response.proposed_power.values()),
            "desired_energy_mwh": sum(response.proposed_energy.values()),
            "optimistic_mpec_profit_eur_per_day": (
                response.optimistic_mpec_profit_eur_per_day
            ),
            "strong_duality_gap": response.strong_duality_gap,
        }
        for response in responses
    }
    state = EpecState(
        x_power=projected_power,
        x_energy=projected_energy,
        iteration=0,
        initialization_method="jacobi_common_snapshot_projected",
        initializer_summary={
            "interpretation": "feasible initialization heuristic, not an equilibrium",
            "snapshot_power_mw_per_investor_node": snapshot_power,
            "snapshot_ratio_hours": snapshot_ratio,
            "numerical_guess_power_mw_per_node": cfg.seed_power_mw,
            "numerical_guess_ratio_hours": cfg.seed_ratio_hours,
            "responses": response_summary,
            "nodes": node_summary,
        },
    )
    clean_capacity_state(state, cfg, nodes)
    return state


#### Diagnostics
# -----------------------------------------------------------------------------

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


#### Jacobi and Gauss-Seidel diagonalization
# -----------------------------------------------------------------------------

def run_epec(
    data: MarketData,
    quad: QuadraticDemandCurve,
    cfg: EpecConfig,
    tee: bool = False,
    checkpoint_callback: Callable[[EpecState], None] | None = None,
    initial_state: EpecState | None = None,
) -> EpecState:
    if cfg.update_rule not in {"jacobi", "seidel"}:
        raise ValueError(f"Unknown update rule: {cfg.update_rule}")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    nodes = list(data.nodes)
    n_inv = len(cfg.investors)
    if (
        initial_state is None
        and cfg.update_rule == "seidel"
        and cfg.automatic_jacobi_initializer
    ):
        state = projected_jacobi_initial_state(data, quad, cfg, tee=tee)
        state.stop_reason = "automatic Jacobi initializer complete; ready for Gauss-Seidel"
        if checkpoint_callback is not None:
            checkpoint_callback(state)
        state.stop_reason = ""
        print(
            "Automatic Jacobi initializer complete; starting Gauss-Seidel "
            "from projected iteration-0 capacities."
        )
    elif initial_state is None:
        seed = min(cfg.seed_power_mw, cfg.node_limit_mw / n_inv)
        state = EpecState(
            x_power={(inv.investor_id, n): seed for inv in cfg.investors for n in nodes},
            x_energy={(inv.investor_id, n): seed * cfg.seed_ratio_hours for inv in cfg.investors for n in nodes},
        )
    else:
        state = initial_state
    clean_capacity_state(state, cfg, nodes)
    consecutive_failures = {inv.investor_id: 0 for inv in cfg.investors}
    final_iteration = state.iteration + cfg.max_iters

    for iteration in range(state.iteration + 1, final_iteration + 1):
        state.iteration = iteration
        x_power_start = dict(state.x_power)
        x_energy_start = dict(state.x_energy)
        responses: list[BestResponse] = []

        if cfg.update_rule == "jacobi":
            snapshot = EpecState(x_power=dict(state.x_power), x_energy=dict(state.x_energy))
            for inv in cfg.investors:
                rival_power, rival_energy = separate_rival_capacities(snapshot, cfg, nodes, inv.investor_id)
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
                rival_power, rival_energy = separate_rival_capacities(state, cfg, nodes, inv.investor_id)
                response = solve_best_response(
                    data, quad, cfg, inv, rival_power, rival_energy,
                    {n: state.x_power[inv.investor_id, n] for n in nodes},
                    {n: state.x_energy[inv.investor_id, n] for n in nodes},
                    tee=tee,
                )
                responses.append(response)
                if response.ok:
                    apply_damped_update(state, cfg, nodes, response)

        if cfg.print_mpec_lambdas:
            for response in responses:
                print_mpec_lambdas(iteration, response)

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
                    "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                    "max_access_shadow_price_eur_per_mw_day": max(
                        response.access_shadow_price_eur_per_mw_day.values()
                    ),
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
                        "private_headroom_limit_mw": response.private_headroom_limit_mw[n],
                        "private_headroom_slack_mw": max(
                            0.0,
                            response.private_headroom_limit_mw[n] - response.proposed_power[n],
                        ),
                        "access_shadow_price_eur_per_mw_day": response.access_shadow_price_eur_per_mw_day[n],
                        "headroom_complementarity_residual_eur_per_day": (
                            response.access_shadow_price_eur_per_mw_day[n]
                            * max(0.0, response.private_headroom_limit_mw[n] - response.proposed_power[n])
                        ),
                        "headroom_mw": cfg.node_limit_mw - sum(state.x_power[j.investor_id, n] for j in cfg.investors),
                    }
                )

        optimistic = ", ".join(
            f"{r.investor_id}={r.optimistic_mpec_profit_eur_per_day:,.0f}" if r.ok else f"{r.investor_id}=FAILED"
            for r in responses
        )
        print(
            f"iter {iteration:2d} [{cfg.update_rule}] max_rel dP={max_rel_power:.4f} dE={max_rel_energy:.4f}"
            f"  optimistic MPEC profit [EUR/day]: {optimistic}"
        )

        should_stop = False
        if any(count >= cfg.max_consecutive_failures for count in consecutive_failures.values()):
            state.stop_reason = "aborted: repeated MPEC solve failures"
            should_stop = True
        elif all_ok and max_rel_power < cfg.tol_rel and max_rel_energy < cfg.tol_rel:
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


#### Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-investor BESS EPEC via diagonalization")
    parser.add_argument("--data", type=Path, default=EXPERIMENT_DATA_PATH)
    parser.add_argument("--update-rule", choices=["jacobi", "seidel"], default="seidel")
    parser.add_argument(
        "--investor-set",
        choices=["wacc", "portfolio4"],
        default="portfolio4",
        help="'portfolio4' (default): four heterogeneous investors. 'wacc': homogeneous investors "
        "from --wacc. The portfolio set contains two "
        "merchants + two same-WACC wind/solar-tilted RES portfolios.",
    )
    parser.add_argument(
        "--investor-order",
        nargs="+",
        default=None,
        help=(
            "Explicit Gauss-Seidel solve order using configured investor IDs, "
            "for example: --investor-order I3 I1 I4 I2."
        ),
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
    parser.add_argument(
        "--initializer-snapshot-power-mw",
        type=float,
        default=0.0,
        help=(
            "Fresh Seidel runs only: common economic MW per investor-node in the "
            "automatic one-sweep Jacobi initializer. The numerical Ipopt guess "
            "continues to use --seed-power-mw."
        ),
    )
    parser.add_argument(
        "--initializer-snapshot-ratio-hours",
        type=float,
        default=DEFAULT_INITIAL_RATIO_HOURS,
        help="E/P ratio of the automatic Jacobi initializer's common snapshot.",
    )
    parser.add_argument(
        "--skip-jacobi-initializer",
        action="store_true",
        help=(
            "Start a fresh run from the legacy uniform seed instead of the automatic "
            "projected one-sweep Jacobi initializer. Ignored when resuming a checkpoint."
        ),
    )
    parser.add_argument("--max-cpu-time", type=float, default=120.0)
    parser.add_argument(
        "--price-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_PRICE_BOUND_EUR_PER_MWH,
        help="Absolute bound for lambda and lambda_system (default: 500 EUR/MWh).",
    )
    parser.add_argument(
        "--dual-bound-eur-per-mwh",
        type=float,
        default=DEFAULT_DUAL_BOUND_EUR_PER_MWH,
        help="Absolute bound for all non-price lower-level duals (default: 10000).",
    )
    parser.add_argument(
        "--capacity-cleanup-tol",
        type=float,
        default=DEFAULT_CAPACITY_CLEANUP_TOL_MW_MWH,
        help="Set a capacity pair to zero when both MW and MWh do not exceed this tolerance.",
    )
    parser.add_argument(
        "--demand-model",
        choices=["fixed", "quadratic"],
        default="fixed",
        help="Lower-level demand representation. The maintained base model uses fixed demand.",
    )
    parser.add_argument(
        "--dispatch-regularization",
        type=float,
        default=0.0,
        help="Optional lower-level quadratic tie-break coefficient in EUR/(MW^2 h).",
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=DEFAULT_SOLVER_TOL,
        help="Ipopt tol and acceptable_tol for best responses and joint settlement.",
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
        help="Resume MW/MWh strategies from a checkpoint.json file "
        "or a directory containing it. --max-iters then means additional iterations.",
    )
    parser.add_argument(
        "--conventional-capacity-adder-mw",
        type=float,
        default=0.0,
        help="Optional MW added to every conventional generator in every hour.",
    )
    parser.add_argument(
        "--peaker-node",
        type=str,
        default=None,
        help="Node for an optional linear-cost peaker (for example N5).",
    )
    parser.add_argument("--peaker-capacity-mw", type=float, default=0.0)
    parser.add_argument("--peaker-cost-eur-per-mwh", type=float, default=95.0)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.damping <= 1.0:
        raise SystemExit("--damping must be in (0, 1].")
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be positive.")
    if args.dispatch_regularization < 0.0:
        raise SystemExit("--dispatch-regularization must be non-negative.")
    if args.solver_tol <= 0.0:
        raise SystemExit("--solver-tol must be positive.")
    if args.dual_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--dual-bound-eur-per-mwh must be positive.")
    if args.price_bound_eur_per_mwh <= 0.0:
        raise SystemExit("--price-bound-eur-per-mwh must be positive.")
    if args.capacity_cleanup_tol < 0.0:
        raise SystemExit("--capacity-cleanup-tol must be non-negative.")
    if args.seed_power_mw < 0.0:
        raise SystemExit("--seed-power-mw must be non-negative.")
    if args.initializer_snapshot_power_mw < 0.0:
        raise SystemExit("--initializer-snapshot-power-mw must be non-negative.")
    base_data = load_market_data(args.data)
    # Keep the no-withholding driver independently runnable while allowing an
    # apples-to-apples comparison with the calibrated strategic experiment.
    from ieee9_strategic_operation_mpec import apply_generator_calibration

    data, generator_calibration = apply_generator_calibration(
        base_data,
        conventional_capacity_adder_mw=args.conventional_capacity_adder_mw,
        peaker_node=args.peaker_node,
        peaker_capacity_mw=args.peaker_capacity_mw,
        peaker_cost_eur_per_mwh=args.peaker_cost_eur_per_mwh,
    )
    if args.investor_set == "portfolio4":
        investors = four_investor_portfolio_profiles(data)
    else:
        investors = tuple(
            InvestorConfig(investor_id=f"I{k + 1}", wacc=wacc) for k, wacc in enumerate(args.wacc)
        )
    try:
        investors = order_investors(investors, args.investor_order)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
        price_bound_eur_per_mwh=args.price_bound_eur_per_mwh,
        dual_bound_eur_per_mwh=args.dual_bound_eur_per_mwh,
        print_mpec_lambdas=args.print_mpec_lambdas,
        system_price_settlement=system_price_settlement,
        use_demand_curve=args.demand_model == "quadratic",
        dispatch_regularization_eur_per_mw2h=args.dispatch_regularization,
        solver_tol=args.solver_tol,
        capacity_cleanup_tol_mw_mwh=args.capacity_cleanup_tol,
        automatic_jacobi_initializer=not args.skip_jacobi_initializer,
        jacobi_initializer_snapshot_power_mw=args.initializer_snapshot_power_mw,
        jacobi_initializer_snapshot_ratio_hours=args.initializer_snapshot_ratio_hours,
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
        f"solve_order={','.join(i.investor_id for i in investors)}, "
        f"rule={cfg.update_rule}, damping={cfg.damping}, tol_rel={cfg.tol_rel}, "
        f"settlement price={'system (zonal)' if cfg.system_price_settlement else 'nodal (LMP)'}, "
        f"demand={args.demand_model}, dispatch_regularization={cfg.dispatch_regularization_eur_per_mw2h:.3e}, "
        f"solver_tol={cfg.solver_tol:.1e}, dual_selection=optimistic"
    )
    if args.conventional_capacity_adder_mw > 0.0 or args.peaker_capacity_mw > 0.0:
        print(f"Generator calibration: {generator_calibration}")
    for inv in investors:
        if inv.owned_generation_shares:
            owned = ", ".join(f"{g}={s:.2f}" for g, s in inv.owned_generation_shares.items())
            print(f"  {inv.investor_id}: portfolio-backed, generation shares [{owned}]")
        else:
            print(f"  {inv.investor_id}: stand-alone merchant BESS")
    if cfg.use_demand_curve:
        print(
            "Quadratic demand curve: "
            f"marginal WTP = {quad.alpha:,.2f} + {quad.beta:,.2f} * curtailed_share EUR/MWh"
        )
    else:
        print("Fixed demand: lower-level load-shedding primal and dual blocks are omitted.")
    if cfg.resume_from is not None:
        print(
            f"Resuming from iteration {initial_state.iteration}; "
            f"running up to {cfg.max_iters} additional iterations from {cfg.resume_from}"
        )
    elif cfg.update_rule == "seidel" and cfg.automatic_jacobi_initializer:
        print(
            "Fresh Gauss-Seidel run: one projected common-snapshot Jacobi "
            "initializer will run first."
        )
    elif cfg.update_rule == "seidel":
        print("Automatic Jacobi initializer skipped; using the legacy uniform seed.")

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
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["generator_calibration"] = generator_calibration
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run_config_path = output_dir / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["generator_calibration"] = generator_calibration
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(f"\nWrote EPEC outputs to {output_dir}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
