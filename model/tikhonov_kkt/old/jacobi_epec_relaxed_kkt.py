"""Four-investor Gauss-Jacobi EPEC with directly regularized lower-level KKT.

Every investor best-responds to the same frozen capacity snapshot. Successful
proposals are damped and applied simultaneously, then the shared nodal access
limit is enforced by proportional projection. The exact, unregularized market
is re-cleared only after the capacity loop as an economic and physical audit.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

try:
    from ..common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case
except ImportError:
    from tikhonov_kkt.common import DEFAULT_DATA_PATH, MODEL_DIR, load_calibrated_case

from epec_diagonalization import (
    EpecConfig,
    EpecState,
    apply_damped_update,
    best_response_task,
    four_investor_portfolio_profiles,
    load_checkpoint_state,
    project_joint_limit,
    relative_delta,
    solve_best_response_tasks,
)
from epec_results import (
    compute_joint_settlement,
    export_epec_checkpoint,
    export_epec_results,
    print_epec_summary,
)
from single_investor_mpec import QuadraticDemandCurve


# USER CONTROLS ---------------------------------------------------------------
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = MODEL_DIR / "output" / "tikhonov_jacobi_epec_gamma1e-3_eps1e-3"
RESUME_FROM = None  # Path to this driver's checkpoint.json, or None.
RESUME_STAGE_NUMBER = 1

# Each entry is (dual Tikhonov gamma, complementarity epsilon, max Jacobi sweeps).
# Start with one known-solvable approximation. A continuation can be enabled by
# replacing it with, for example:
# ((1e-2, 1e-3, 10), (3e-3, 3e-4, 10), (1e-3, 1e-4, 20)).
REGULARIZATION_STAGES = ((1.0e-3, 1.0e-3, 20),)
COMPLEMENTARITY_FORMULATION = "scholtes"  # "scholtes" or "central-path"
COMPLEMENTARITY_SHIFT = 1.0e-8
REQUIRE_EACH_STAGE_TO_CONVERGE = False

NODE_LIMIT_MW = 200.0
INITIAL_POWER_MW_PER_INVESTOR_NODE = 0.0
INITIAL_DURATION_HOURS = 4.0
MPEC_INITIAL_GUESS_POWER_MW_PER_NODE = 10.0
MPEC_INITIAL_GUESS_DURATION_HOURS = 4.0
DAMPING = 0.25
CONVERGENCE_TOL_REL = 0.02
CONVERGENCE_FLOOR_MW = 1.0
CONVERGENCE_FLOOR_MWH = 2.0
CAPACITY_CLEANUP_TOL = 1.0e-4
RIVAL_SPARSITY_TOL_MW = 0.01
MAX_CONSECUTIVE_FAILURES = 3

PRICE_BOUND_EUR_PER_MWH = 500.0
OTHER_DUAL_BOUND = 10_000.0
DISPATCH_REGULARIZATION_EUR_PER_MW2H = 0.0
SOLVER_TOL = 1.0e-4
MAX_CPU_TIME_SECONDS_PER_INVESTOR = 180.0
PARALLEL_WORKERS = 2
TEE = False  # Must remain False when PARALLEL_WORKERS > 1.
# -----------------------------------------------------------------------------


def build_epec_config(data, gamma: float, epsilon: float, max_sweeps: int) -> EpecConfig:
    """Create the shared settings used by all four investor MPECs in one stage."""

    return EpecConfig(
        investors=four_investor_portfolio_profiles(data),
        node_limit_mw=NODE_LIMIT_MW,
        update_rule="jacobi",
        damping=DAMPING,
        max_iters=max_sweeps,
        tol_rel=CONVERGENCE_TOL_REL,
        floor_mw=CONVERGENCE_FLOOR_MW,
        floor_mwh=CONVERGENCE_FLOOR_MWH,
        seed_power_mw=MPEC_INITIAL_GUESS_POWER_MW_PER_NODE,
        seed_ratio_hours=MPEC_INITIAL_GUESS_DURATION_HOURS,
        max_cpu_time=MAX_CPU_TIME_SECONDS_PER_INVESTOR,
        price_bound_eur_per_mwh=PRICE_BOUND_EUR_PER_MWH,
        dual_bound_eur_per_mwh=OTHER_DUAL_BOUND,
        lower_level_optimality="relaxed-kkt",
        complementarity_epsilon=epsilon,
        complementarity_formulation=COMPLEMENTARITY_FORMULATION,
        complementarity_shift=COMPLEMENTARITY_SHIFT,
        dual_tikhonov_gamma=gamma,
        use_demand_curve=False,
        dispatch_regularization_eur_per_mw2h=DISPATCH_REGULARIZATION_EUR_PER_MW2H,
        solver_tol=SOLVER_TOL,
        capacity_cleanup_tol_mw_mwh=CAPACITY_CLEANUP_TOL,
        rival_sparsity_tol_mw=RIVAL_SPARSITY_TOL_MW,
        max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
        strategic_parallel_workers=PARALLEL_WORKERS,
        automatic_jacobi_initializer=False,
    )


def initial_state(data, cfg: EpecConfig) -> EpecState:
    """Feasible common capacity snapshot used before the first Jacobi sweep."""

    nodes = list(data.nodes)
    seed = min(
        INITIAL_POWER_MW_PER_INVESTOR_NODE,
        NODE_LIMIT_MW / len(cfg.investors),
    )
    return EpecState(
        x_power={
            (investor.investor_id, node): seed
            for investor in cfg.investors
            for node in nodes
        },
        x_energy={
            (investor.investor_id, node): seed * INITIAL_DURATION_HOURS
            for investor in cfg.investors
            for node in nodes
        },
        initialization_method="uniform_feasible_common_snapshot",
        initializer_summary={
            "power_mw_per_investor_node": seed,
            "duration_hours": INITIAL_DURATION_HOURS,
            "mpec_initial_guess_power_mw_per_node": MPEC_INITIAL_GUESS_POWER_MW_PER_NODE,
            "mpec_initial_guess_duration_hours": MPEC_INITIAL_GUESS_DURATION_HOURS,
        },
    )


def run_jacobi_stage(
    data,
    quad: QuadraticDemandCurve,
    state: EpecState,
    cfg: EpecConfig,
    *,
    stage_number: int,
) -> tuple[bool, list]:
    """Run simultaneous best-response sweeps for one (gamma, epsilon) pair."""

    nodes = list(data.nodes)
    failures = {investor.investor_id: 0 for investor in cfg.investors}
    last_responses = []

    def print_completed_response(response) -> None:
        proposed_power = sum(response.proposed_power.values())
        proposed_energy = sum(response.proposed_energy.values())
        if response.ok:
            diagnostics = (
                f"profit={response.optimistic_mpec_profit_eur_per_day:,.2f} EUR/day, "
                f"max_comp={response.max_complementarity_product:.3e}, "
                f"system_imbalance={response.max_hourly_system_balance_residual_mw:.6f} MW"
            )
        else:
            diagnostics = "proposal held at the previous capacity"
        print(
            f"  completed {response.investor_id}: {response.termination}, "
            f"solve={response.solve_seconds:.1f}s, preparation={response.preparation_seconds:.1f}s, "
            f"attempts={response.attempt_count} ({response.retry_status}), "
            f"proposed={proposed_power:.3f} MW / {proposed_energy:.3f} MWh, "
            f"{diagnostics}",
            flush=True,
        )

    for _ in range(cfg.max_iters):
        state.iteration += 1
        old_power = dict(state.x_power)
        old_energy = dict(state.x_energy)

        # This immutable snapshot is the defining Gauss-Jacobi step: none of the
        # four MPECs sees another investor's current-sweep proposal.
        snapshot = EpecState(
            x_power=dict(old_power),
            x_energy=dict(old_energy),
            iteration=state.iteration - 1,
        )
        tasks = [
            best_response_task(data, quad, cfg, investor, snapshot, nodes)
            for investor in cfg.investors
        ]
        print(
            f"stage {stage_number} iter {state.iteration}: starting "
            f"{len(tasks)} best responses with {cfg.strategic_parallel_workers} worker(s) "
            f"[gamma={cfg.dual_tikhonov_gamma:.3e}, "
            f"epsilon={cfg.complementarity_epsilon:.3e}]",
            flush=True,
        )
        responses = solve_best_response_tasks(
            tasks,
            parallel_workers=cfg.strategic_parallel_workers,
            tee=TEE,
            progress_callback=print_completed_response,
        )
        last_responses = responses

        # All proposals are now known, so the state may be changed simultaneously.
        for response in responses:
            if response.ok:
                apply_damped_update(state, cfg, nodes, response)
        project_joint_limit(state, cfg, nodes)

        max_rel_power = 0.0
        max_rel_energy = 0.0
        for response in responses:
            investor_id = response.investor_id
            failures[investor_id] = 0 if response.ok else failures[investor_id] + 1
            rel_power = max(
                relative_delta(
                    state.x_power[investor_id, node],
                    old_power[investor_id, node],
                    cfg.floor_mw,
                )
                for node in nodes
            )
            rel_energy = max(
                relative_delta(
                    state.x_energy[investor_id, node],
                    old_energy[investor_id, node],
                    cfg.floor_mwh,
                )
                for node in nodes
            )
            max_rel_power = max(max_rel_power, rel_power)
            max_rel_energy = max(max_rel_energy, rel_energy)
            undamped_power = max(
                abs(response.proposed_power[node] - old_power[investor_id, node])
                for node in nodes
            )

            state.history.append(
                {
                    "iteration": state.iteration,
                    "regularization_stage": stage_number,
                    "dual_tikhonov_gamma": cfg.dual_tikhonov_gamma,
                    "complementarity_epsilon": cfg.complementarity_epsilon,
                    "investor": investor_id,
                    "termination": response.termination,
                    "solve_seconds": response.solve_seconds,
                    "preparation_seconds": response.preparation_seconds,
                    "attempt_count": response.attempt_count,
                    "retry_status": response.retry_status,
                    "optimistic_mpec_profit_eur_per_day": response.optimistic_mpec_profit_eur_per_day,
                    "max_access_shadow_price_eur_per_mw_day": max(
                        response.access_shadow_price_eur_per_mw_day.values()
                    ),
                    "strong_duality_gap": response.strong_duality_gap,
                    "max_complementarity_product": response.max_complementarity_product,
                    "sum_complementarity_products": response.sum_complementarity_products,
                    "max_original_nodal_balance_residual_mw": response.max_original_nodal_balance_residual_mw,
                    "total_absolute_original_nodal_balance_residual_mwh": response.total_absolute_original_nodal_balance_residual_mwh,
                    "max_hourly_system_balance_residual_mw": response.max_hourly_system_balance_residual_mw,
                    "signed_primal_dual_gap_eur_per_day": response.signed_primal_dual_gap_eur_per_day,
                    "total_power_mw": sum(state.x_power[investor_id, node] for node in nodes),
                    "total_energy_mwh": sum(state.x_energy[investor_id, node] for node in nodes),
                    "max_rel_delta_power": rel_power,
                    "max_rel_delta_energy": rel_energy,
                    "max_undamped_delta_power_mw": undamped_power,
                }
            )
            for node in nodes:
                private_slack = max(
                    0.0,
                    response.private_headroom_limit_mw[node]
                    - response.proposed_power[node],
                )
                state.trajectory.append(
                    {
                        "iteration": state.iteration,
                        "investor": investor_id,
                        "node": node,
                        "x_power_mw": state.x_power[investor_id, node],
                        "x_energy_mwh": state.x_energy[investor_id, node],
                        "proposed_x_power_mw": response.proposed_power[node],
                        "private_headroom_limit_mw": response.private_headroom_limit_mw[node],
                        "private_headroom_slack_mw": private_slack,
                        "access_shadow_price_eur_per_mw_day": response.access_shadow_price_eur_per_mw_day[node],
                        "headroom_complementarity_residual_eur_per_day": (
                            response.access_shadow_price_eur_per_mw_day[node]
                            * private_slack
                        ),
                        "headroom_mw": cfg.node_limit_mw
                        - sum(state.x_power[i.investor_id, node] for i in cfg.investors),
                    }
                )

            if response.ok:
                state.final_selected_prices[investor_id] = dict(
                    response.selected_prices_eur_per_mwh
                )
                state.final_access_shadow_prices[investor_id] = dict(
                    response.access_shadow_price_eur_per_mw_day
                )

        all_ok = all(response.ok for response in responses)
        max_balance = max(
            (response.max_hourly_system_balance_residual_mw for response in responses if response.ok),
            default=float("nan"),
        )
        print(
            f"stage {stage_number} iter {state.iteration}: "
            f"max_rel dP={max_rel_power:.4f}, dE={max_rel_energy:.4f}, "
            f"max system imbalance={max_balance:.6f} MW; "
            f"capacity convergence requires all_optimal={all_ok} and "
            f"dP,dE < {cfg.tol_rel:.4f}",
            flush=True,
        )

        state.stop_reason = f"in progress after iteration {state.iteration}"
        export_epec_checkpoint(OUTPUT_DIR, state, cfg)

        if any(count >= cfg.max_consecutive_failures for count in failures.values()):
            state.stop_reason = "aborted: repeated MPEC solve failures"
            return False, last_responses
        if all_ok and max_rel_power < cfg.tol_rel and max_rel_energy < cfg.tol_rel:
            return True, last_responses

    return False, last_responses


def validate_controls() -> None:
    if not REGULARIZATION_STAGES:
        raise ValueError("REGULARIZATION_STAGES must contain at least one stage.")
    if PARALLEL_WORKERS < 1:
        raise ValueError("PARALLEL_WORKERS must be at least one.")
    if PARALLEL_WORKERS > 1 and TEE:
        raise ValueError("TEE must be False with parallel worker processes.")
    if not 1 <= RESUME_STAGE_NUMBER <= len(REGULARIZATION_STAGES):
        raise ValueError("RESUME_STAGE_NUMBER must identify a configured stage.")
    for gamma, epsilon, sweeps in REGULARIZATION_STAGES:
        if gamma < 0.0 or epsilon < 0.0 or sweeps <= 0:
            raise ValueError(
                "Every regularization stage needs gamma>=0, epsilon>=0, and sweeps>0."
            )


def main() -> int:
    validate_controls()
    data_path = Path(DATA_PATH)
    data = load_calibrated_case(data_path)
    quad = QuadraticDemandCurve(alpha=100.0, beta=0.1)

    first_stage_index = RESUME_STAGE_NUMBER - 1 if RESUME_FROM is not None else 0
    first_gamma, first_epsilon, first_sweeps = REGULARIZATION_STAGES[first_stage_index]
    cfg = build_epec_config(data, first_gamma, first_epsilon, first_sweeps)
    if RESUME_FROM is None:
        state = initial_state(data, cfg)
    else:
        state, checkpoint_path = load_checkpoint_state(Path(RESUME_FROM), data, cfg)
        cfg = replace(
            cfg,
            starting_iteration=state.iteration,
            resume_from=str(checkpoint_path),
        )
    last_responses = []

    print(
        f"Tikhonov Jacobi EPEC: investors={len(cfg.investors)}, "
        f"workers={PARALLEL_WORKERS}, damping={DAMPING}, stages={REGULARIZATION_STAGES}"
    )
    for stage_number, (gamma, epsilon, max_sweeps) in enumerate(
        REGULARIZATION_STAGES[first_stage_index:], start=first_stage_index + 1
    ):
        cfg = replace(
            cfg,
            dual_tikhonov_gamma=gamma,
            complementarity_epsilon=epsilon,
            max_iters=max_sweeps,
        )
        print(
            f"Starting stage {stage_number}: gamma={gamma:.3e}, "
            f"epsilon={epsilon:.3e}, max_sweeps={max_sweeps}"
        )
        stage_converged, last_responses = run_jacobi_stage(
            data,
            quad,
            state,
            cfg,
            stage_number=stage_number,
        )
        final_stage = stage_number == len(REGULARIZATION_STAGES)
        if final_stage:
            state.converged = stage_converged
        if state.stop_reason.startswith("aborted"):
            break
        if REQUIRE_EACH_STAGE_TO_CONVERGE and not stage_converged:
            state.stop_reason = f"stage {stage_number} did not converge"
            break
        if stage_converged:
            print(f"Stage {stage_number} capacity iteration converged.")

    if last_responses:
        state.final_models = {
            response.investor_id: response.model
            for response in last_responses
            if response.model is not None
        }
    if state.converged:
        state.stop_reason = f"converged at final regularization stage in {state.iteration} sweeps"
    elif not state.stop_reason.startswith(("aborted", "stage")):
        state.stop_reason = f"maximum configured sweeps reached ({state.iteration})"

    # This settlement intentionally removes gamma and epsilon. It audits the
    # final capacities in the original physical market rather than reporting
    # the regularized imbalance as actual settled energy.
    settlement = compute_joint_settlement(data, quad, state, cfg)
    print_epec_summary(state, cfg, settlement)
    export_epec_results(OUTPUT_DIR, data, state, cfg, settlement, data_path)
    run_config_path = OUTPUT_DIR / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["regularization_stages"] = [
        {
            "dual_tikhonov_gamma": gamma,
            "complementarity_epsilon": epsilon,
            "max_jacobi_sweeps": sweeps,
        }
        for gamma, epsilon, sweeps in REGULARIZATION_STAGES
    ]
    run_config["exact_unregularized_final_market_reclear"] = True
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Wrote Jacobi EPEC outputs to {OUTPUT_DIR}")
    return 0 if state.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
