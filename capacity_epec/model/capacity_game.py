"""The capacity game: three investors, one market, iterated best responses.

Every investor solves the MPEC of `mpec` against the others' current
capacities, and the profile is updated by a *damped Jacobi* step — all three
respond to the same frozen profile, then move part of the way towards their
answers.  Damping is what keeps the iteration from oscillating between the
symmetric branches of this game.

Two habits run through this module and are the reason its results can be
believed at all:

* every capacity a solver proposes is re-cleared in the exact market of
  `iso_market` before it is compared to anything, because the MPEC's own
  prices are only as good as its relaxation; and
* the proximal penalty that stabilises the iteration is always subtracted
  explicitly, never folded into a reported profit.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Callable

import pyomo.environ as pyo

import iso_market
import mpec
import mpec_afrr
from investors import InvestorConfig
from market_data import MarketData
from solvers import SolveOutcome, SolverSettings, maximum_bound_violation, solve_mpec


Capacities = dict[tuple[str, str], float]
NodalMap = dict[str, float]

# Capacity profile used by --asymmetric-initialization, kept only as a
# path-dependence diagnostic against the symmetric default.
HISTORICAL_INITIAL_POWER_MW = {"I1": 5.0, "I2": 2.0, "I3": 15.0}

# See GameConfig.convergence_metric.
CONVERGENCE_METRICS = ("capacity", "profit", "regret")


@dataclass(frozen=True)
class GameConfig:
    """Everything that defines one run of the capacity game."""

    investors: tuple[InvestorConfig, ...]

    # Lower-level embedding.
    lower_level: str = "relaxed-kkt"
    complementarity_epsilon: float = 1.0e-4
    market_design: str = "energy-only"

    # Damped Jacobi iteration.
    max_sweeps: int = 60
    damping: float = 0.25
    consecutive_sweeps: int = 2
    tolerance_mw: float = 0.5
    tolerance_mwh: float = 1.0
    cleanup_tolerance: float = 1.0e-6

    # What "converged" means.  ``capacity`` stops when the raw best-response
    # capacity deviations fall inside tolerance_mw/tolerance_mwh.  That is the
    # wrong question to ask of a game whose equilibria form a set: the iterate
    # slides along the set at roughly constant payoff and the capacity residual
    # never settles.
    #
    # ``regret`` stops on the Nash condition itself — the largest profitable
    # unilateral deviation, relative to the largest profit.  It is the same
    # quantity the final audit reports as profitable_deviation_eur_per_day, so
    # the per-sweep monitor and the certificate are one number.
    #
    # ``profit`` stops when incumbent payoffs stop moving between sweeps.  It is
    # the cheapest of the three but it is not a Nash gap: payoffs are also flat
    # in the dead zone between best-response branches, where an investor can be
    # losing money and no deviation has been tested.  Use it as a diagnostic.
    convergence_metric: str = "capacity"
    tolerance_relative_regret: float = 1.0e-3
    tolerance_profit_eur_per_day: float = 1.0
    # A stop with an investor below this profit is refused whatever the metric
    # says.  The symmetric damped runs parked at ~250 MW with I1 at
    # -1819 EUR/day; that is not an equilibrium, it is an unfinished run.
    require_individual_rationality: bool = True
    individual_rationality_tolerance_eur_per_day: float = 1.0e-6

    # Best-response search.
    multistart_every_sweeps: int = 5
    refinement_starts: int = 3
    always_refine_incumbent: bool = True
    # Exact market re-clears can differ by a few micro-EUR/day for numerically
    # identical capacities.  Within this deliberately tiny band, prefer a
    # candidate for which the NLP solver returned a local optimum over a raw
    # screened point.  This keeps zero-investment best responses auditable
    # without hiding an economically meaningful profit difference.
    candidate_selection_tolerance_eur_per_day: float = 1.0e-4

    # Moving proximal regularizer (algorithmic, never economic).  Off by
    # default: it biases every best response toward the incumbent, which
    # understates regret and so hides the very deviation a payoff-based
    # convergence test is looking for.  Switch it on only to damp an
    # iteration that will not settle otherwise.
    proximal_penalty: float = 0.0
    proximal_energy_scale: float = 2.0

    # Shared nodal connection capacity.  None uses the node-specific limits
    # from the market input; ``math.inf`` switches the shared cap off
    # everywhere, in the MPEC and in every exact screen and reclear alike.
    uniform_node_connection_limit_mw: float | None = None

    # Starting profile.
    symmetric_initialization: bool = True
    initial_power_mw: float = 5.0
    initial_ratio_hours: float = 3.0

    # Final zero-proximal audit.
    audit_profit_tolerance_eur_per_day: float = 1.0
    audit_reclear_gap_tolerance_eur_per_day: float = 10.0

    # Numerics.
    price_bound: float = 500.0
    dual_bound: float = 10_000.0
    sparse_capacity_tol: float = 1.0e-8
    parallel_workers: int = 4
    solver: SolverSettings = field(default_factory=SolverSettings)

    @property
    def investor_ids(self) -> tuple[str, ...]:
        return tuple(investor.investor_id for investor in self.investors)

    def degradation(self) -> dict[str, float]:
        return {i.investor_id: i.degradation_eur_per_mwh for i in self.investors}

    @property
    def shared_node_cap_enforced(self) -> bool:
        """Whether a finite shared cap couples the investors at all.

        ``math.inf`` and ``None`` both serialise to JSON ``null`` but mean
        opposite things -- cap switched off, and cap taken from the input --
        so the reports state which one it was rather than implying it.
        """

        limit = self.uniform_node_connection_limit_mw
        return limit is None or math.isfinite(limit)

    def node_limits(self, data: MarketData) -> dict[str, float]:
        if self.uniform_node_connection_limit_mw is None:
            return {node: float(data.node_connection_limit[node]) for node in data.nodes}
        return {node: float(self.uniform_node_connection_limit_mw) for node in data.nodes}


@dataclass(frozen=True)
class BestResponse:
    """One investor's answer to a frozen rival profile.

    ``embedded_profit`` is what the MPEC believed at its own relaxed prices;
    ``recleared_profit`` is the same capacity valued by the exact market.  A
    large gap between them means the relaxation, not the economics, is talking.
    """

    investor_id: str
    outcome: SolveOutcome
    power: NodalMap
    energy: NodalMap
    embedded_profit_eur_per_day: float
    max_complementarity_product: float
    max_complementarity_violation: float
    primal_dual_gap_eur_per_day: float
    recleared_profit_eur_per_day: float = math.nan
    start_label: str = "incumbent"
    start_records: tuple[dict[str, object], ...] = ()
    reclear_error: str = ""
    max_bound_violation: float = math.nan


@dataclass
class GameState:
    """The profile carried from sweep to sweep, plus its running diary."""

    power: Capacities
    energy: Capacities
    history: list[dict[str, object]] = field(default_factory=list)
    sweep: int = 0
    converged: bool = False
    stable_sweeps: int = 0
    stop_reason: str = ""
    responses: dict[str, BestResponse] = field(default_factory=dict)
    # Incumbent profit at the profile the last sweep's responses answered, and
    # the sweep's profitable deviations.  Empty until a sweep has measured them.
    profit: dict[str, float] = field(default_factory=dict)
    regret: dict[str, float] = field(default_factory=dict)

    def total_power_mw(self) -> float:
        return sum(self.power.values())

    def total_energy_mwh(self) -> float:
        return sum(self.energy.values())


SweepCallback = Callable[[GameState], None]


def validate(data: MarketData, config: GameConfig) -> None:
    if not config.investors:
        raise ValueError("At least one investor is required.")
    ids = config.investor_ids
    if len(ids) != len(set(ids)):
        raise ValueError("Investor identifiers must be unique.")
    if config.lower_level not in mpec.LOWER_LEVEL_FORMULATIONS:
        raise ValueError(f"Unknown lower-level formulation: {config.lower_level}")
    if not 0.0 < config.damping <= 1.0:
        raise ValueError("Damping must lie in (0, 1].")
    if config.max_sweeps <= 0 or config.consecutive_sweeps <= 0:
        raise ValueError("Sweep counts must be positive.")
    if config.multistart_every_sweeps < 0 or config.refinement_starts < 0:
        raise ValueError("Search counts cannot be negative.")
    if config.candidate_selection_tolerance_eur_per_day < 0.0:
        raise ValueError("Candidate selection tolerance cannot be negative.")
    if config.convergence_metric not in CONVERGENCE_METRICS:
        raise ValueError(
            f"Unknown convergence metric: {config.convergence_metric}. "
            f"Expected one of {sorted(CONVERGENCE_METRICS)}."
        )
    if config.convergence_metric == "regret" and config.proximal_penalty > 0.0:
        # The proximal term pulls the best response back toward the incumbent,
        # so the measured deviation understates the true one.  Stopping on a
        # regularized regret would certify the regularizer, not the game.
        raise ValueError(
            "convergence_metric='regret' requires proximal_penalty=0.0; "
            f"got {config.proximal_penalty}. A proximal term biases every "
            "best response toward the incumbent, which understates regret."
        )
    if min(config.tolerance_relative_regret, config.tolerance_profit_eur_per_day) < 0.0:
        raise ValueError("Convergence tolerances cannot be negative.")
    if config.complementarity_epsilon < 0.0 or config.proximal_penalty < 0.0:
        raise ValueError("Relaxation and penalty parameters cannot be negative.")
    if config.uniform_node_connection_limit_mw is not None and not (
        config.uniform_node_connection_limit_mw > 0.0
    ):
        raise ValueError("The uniform node connection limit must be positive.")
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive.")
    if min(
        config.tolerance_mw,
        config.tolerance_mwh,
        config.audit_profit_tolerance_eur_per_day,
        config.audit_reclear_gap_tolerance_eur_per_day,
    ) < 0.0:
        raise ValueError("Convergence tolerances cannot be negative.")
    unknown = {
        generator
        for investor in config.investors
        for generator in investor.owned_generation_shares
        if generator not in data.generators
    }
    if unknown:
        raise ValueError(f"Unknown owned generators: {sorted(unknown)}")


def initial_state(
    data: MarketData,
    config: GameConfig,
    capacities: tuple[Capacities, Capacities] | None = None,
) -> GameState:
    """Symmetric start by default; the historical or a stated profile on request.

    ``capacities`` is an explicit ``(power, energy)`` pair. The best-response
    surface of this game has a congestion-relief threshold with a different
    equilibrium on each side, so the starting profile is a real modelling
    choice and is stated rather than implied.
    """

    validate(data, config)
    if capacities is not None:
        power, energy = ({k: float(v) for k, v in side.items()} for side in capacities)
        missing = [
            (unit, node)
            for unit in config.investor_ids
            for node in data.nodes
            if (unit, node) not in power or (unit, node) not in energy
        ]
        if missing:
            raise ValueError(f"Initial capacities are missing entries for {missing[:5]}.")
        for investor in config.investors:
            _check_capacity_feasible(
                data,
                config,
                investor,
                {n: power[investor.investor_id, n] for n in data.nodes},
                {n: energy[investor.investor_id, n] for n in data.nodes},
            )
        _check_profile_node_limits(data, config, power)
        return GameState(power=power, energy=energy)
    power_by_investor = {
        investor.investor_id: (
            config.initial_power_mw
            if config.symmetric_initialization
            else HISTORICAL_INITIAL_POWER_MW.get(
                investor.investor_id, config.initial_power_mw
            )
        )
        for investor in config.investors
    }
    power = {
        (investor.investor_id, node): power_by_investor[investor.investor_id]
        for investor in config.investors
        for node in data.nodes
    }
    energy = {
        (investor.investor_id, node): power_by_investor[investor.investor_id]
        * investor.clip_duration(config.initial_ratio_hours)
        for investor in config.investors
        for node in data.nodes
    }
    _check_profile_node_limits(data, config, power)
    return GameState(power=power, energy=energy)


# --------------------------------------------------------------------------
# Exact market


def clear(
    data: MarketData, config: GameConfig, power: Capacities, energy: Capacities
) -> pyo.ConcreteModel:
    """Clear the competitive market for one full capacity profile."""

    _check_profile_node_limits(data, config, power)
    return iso_market.clear_market(
        data, config.solver, power, energy, config.degradation()
    )


def recleared_profit(
    market: pyo.ConcreteModel,
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
) -> float:
    return iso_market.settle(market, data, investor, power, energy).profit


def _with_candidate(
    data: MarketData,
    power: Capacities,
    energy: Capacities,
    investor_id: str,
    candidate_power: NodalMap,
    candidate_energy: NodalMap,
) -> tuple[Capacities, Capacities]:
    """The full profile with one investor's capacities swapped out."""

    merged_power = dict(power)
    merged_energy = dict(energy)
    for node, value in candidate_power.items():
        merged_power[investor_id, node] = value
    for node, value in candidate_energy.items():
        merged_energy[investor_id, node] = value
    return merged_power, merged_energy


# --------------------------------------------------------------------------
# One best response


def build_best_response_model(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
    start_power: NodalMap | None = None,
    start_energy: NodalMap | None = None,
) -> pyo.ConcreteModel:
    """Build one investor's MPEC and warm-start its embedded market."""

    active = investor.investor_id
    rivals = [item for item in config.investors if item.investor_id != active]
    formulation_module = mpec_afrr if data.afrr_enabled else mpec
    model = formulation_module.build_capacity_mpec(
        data,
        investor=investor,
        rival_power={
            rival.investor_id: {n: power[rival.investor_id, n] for n in data.nodes}
            for rival in rivals
        },
        rival_energy={
            rival.investor_id: {n: energy[rival.investor_id, n] for n in data.nodes}
            for rival in rivals
        },
        rival_degradation={rival.investor_id: rival.degradation_eur_per_mwh for rival in rivals},
        node_connection_limit=config.node_limits(data),
        lower_level=config.lower_level,
        complementarity_epsilon=config.complementarity_epsilon,
        initial_ratio_hours=config.initial_ratio_hours,
        price_bound=config.price_bound,
        dual_bound=config.dual_bound,
        sparse_capacity_tol=config.sparse_capacity_tol,
        proximal_power={n: power[active, n] for n in data.nodes},
        proximal_energy={n: energy[active, n] for n in data.nodes},
        proximal_penalty=config.proximal_penalty,
        proximal_energy_scale=config.proximal_energy_scale,
    )
    # IPOPT reads Var.value as its primal start, so the capacity start is set
    # first and the market state is then seeded consistently with it.
    for node in data.nodes:
        model.X_power[node].set_value(
            float((start_power or {}).get(node, power[active, node]))
        )
        model.X_energy[node].set_value(
            float((start_energy or {}).get(node, energy[active, node]))
        )
    seed_lower_level(model, config)
    return model


def seed_lower_level(model: pyo.ConcreteModel, config: GameConfig) -> None:
    """Warm-start the embedded KKT system from an exactly cleared market.

    Cold starts of this MPEC do not converge in practice: IPOPT needs a market
    solution that is already consistent with the capacities it starts from.
    """

    ctx = mpec.context(model)
    data = ctx.data
    units = [ctx.active_id, *ctx.rival_ids]
    power = {}
    energy = {}
    for unit in units:
        for node in data.nodes:
            if unit == ctx.active_id:
                power[unit, node] = float(pyo.value(model.X_power[node]))
                energy[unit, node] = float(pyo.value(model.X_energy[node]))
            else:
                power[unit, node] = ctx.rival_power[unit][node]
                energy[unit, node] = ctx.rival_energy[unit][node]
    lower = iso_market.clear_market(
        data, config.solver, power, energy, {u: ctx.degradation[u] for u in units}
    )

    def seed(variable: pyo.Var, value: float) -> None:
        clipped = float(value)
        if variable.lb is not None:
            clipped = max(clipped, float(variable.lb))
        if variable.ub is not None:
            clipped = min(clipped, float(variable.ub))
        variable.set_value(clipped)

    for generator, time in model.GT:
        seed(model.P_gen[generator, time], pyo.value(lower.P_gen[generator, time]))
        seed(
            model.nu_gen[generator, time],
            lower.dual[lower.generation_capacity_bound[generator, time]],
        )
    for node in model.N:
        for time in model.T:
            seed(model.NetInjection[node, time], pyo.value(lower.NetInjection[node, time]))
            seed(
                model.DemandAdjustment[node, time],
                pyo.value(lower.DemandAdjustment[node, time]),
            )
            seed(model.lam[node, time], lower.dual[lower.nodal_balance[node, time]])
    for time in model.T:
        seed(model.lam_sys[time], lower.dual[lower.system_balance[time]])
    for line in model.L:
        for time in model.T:
            seed(model.mu_up[line, time], lower.dual[lower.line_upper_bound[line, time]])
            seed(model.mu_dn[line, time], lower.dual[lower.line_lower_bound[line, time]])
    for unit, node in model.IN:
        for time in model.T:
            seed(model.P_charge[unit, node, time], pyo.value(lower.P_charge[unit, node, time]))
            seed(
                model.P_discharge[unit, node, time],
                pyo.value(lower.P_discharge[unit, node, time]),
            )
            seed(
                model.rho_ch[unit, node, time],
                lower.dual[lower.charge_power_bound[unit, node, time]],
            )
            seed(
                model.sig_dis[unit, node, time],
                lower.dual[lower.discharge_power_bound[unit, node, time]],
            )
            seed(model.gam[unit, node, time], lower.dual[lower.soc_transition[unit, node, time]])
        for soc_time in model.T_SOC:
            seed(model.SOC[unit, node, soc_time], pyo.value(lower.SOC[unit, node, soc_time]))
            seed(
                model.del_soc[unit, node, soc_time],
                lower.dual[lower.soc_capacity_bound[unit, node, soc_time]],
            )
        seed(model.rho_per[unit, node], lower.dual[lower.soc_periodicity[unit, node]])
    if data.afrr_enabled:
        mpec_afrr.seed_afrr(model, lower)


def _failed_response(
    investor: InvestorConfig,
    data: MarketData,
    power: Capacities,
    energy: Capacities,
    outcome: SolveOutcome,
    **extra: object,
) -> BestResponse:
    """Stand still: a failed solve keeps the investor at its current profile."""

    unit = investor.investor_id
    return BestResponse(
        investor_id=unit,
        outcome=outcome,
        power={node: power[unit, node] for node in data.nodes},
        energy={node: energy[unit, node] for node in data.nodes},
        embedded_profit_eur_per_day=math.nan,
        max_complementarity_product=math.nan,
        max_complementarity_violation=math.nan,
        primal_dual_gap_eur_per_day=math.nan,
        **extra,
    )


def solve_from_start(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
    start_label: str,
    start_power: NodalMap,
    start_energy: NodalMap,
) -> BestResponse:
    """Solve the MPEC from one capacity start and check the point returned."""

    model = build_best_response_model(
        data, config, investor, power, energy, start_power, start_energy
    )
    outcome = solve_mpec(model, config.solver)
    violation = maximum_bound_violation(model)
    feasible = math.isfinite(violation) and violation <= config.solver.tolerance
    # A time limit is not a certificate of local optimality, but a checked
    # feasible final iterate is still an admissible candidate.
    if not outcome.has_solution and feasible:
        outcome = replace(outcome, has_solution=True)
    if not outcome.has_solution:
        return _failed_response(
            investor, data, power, energy, outcome, start_label=start_label
        )
    if not feasible:
        return _failed_response(
            investor,
            data,
            power,
            energy,
            replace(
                outcome,
                termination=f"invalid_solution:{violation}",
                has_solution=False,
                optimal=False,
            ),
            start_label=start_label,
            max_bound_violation=violation,
        )
    diagnostics = mpec.complementarity_diagnostics(model)
    return BestResponse(
        investor_id=investor.investor_id,
        outcome=outcome,
        power={n: max(0.0, float(pyo.value(model.X_power[n]))) for n in data.nodes},
        energy={n: max(0.0, float(pyo.value(model.X_energy[n]))) for n in data.nodes},
        embedded_profit_eur_per_day=float(pyo.value(model.unregularized_profit)),
        max_complementarity_product=float(diagnostics["maximum_product"]),
        max_complementarity_violation=max(
            float(diagnostics["maximum_upper_bound_violation"]),
            float(diagnostics["maximum_nonnegativity_violation"]),
        ),
        primal_dual_gap_eur_per_day=float(diagnostics["primal_dual_gap_eur_per_day"]),
        start_label=start_label,
        max_bound_violation=violation,
    )


def capacity_starts(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
) -> tuple[tuple[str, NodalMap, NodalMap], ...]:
    """Capacity profiles to try, spanning exit, expansion, duration and location.

    The best-response surface here is flat in places and has genuinely separate
    branches, so a single local solve from the incumbent is not evidence of a
    best response.
    """

    unit = investor.investor_id
    incumbent_power = {node: float(power[unit, node]) for node in data.nodes}
    incumbent_energy = {node: float(energy[unit, node]) for node in data.nodes}
    total = sum(incumbent_power.values())
    zeros = {node: 0.0 for node in data.nodes}

    profiles: list[tuple[str, NodalMap, NodalMap]] = [
        ("incumbent", incumbent_power, incumbent_energy),
        ("zero", dict(zeros), dict(zeros)),
    ]
    if total > 0.0:
        profiles.append(
            (
                "scaled_up",
                {n: 2.0 * incumbent_power[n] for n in data.nodes},
                {n: 2.0 * incumbent_energy[n] for n in data.nodes},
            )
        )
        for duration in (investor.ratio_min, 4.0, investor.ratio_max):
            hours = investor.clip_duration(duration)
            profiles.append(
                (
                    f"duration_{hours:g}h",
                    dict(incumbent_power),
                    {n: hours * incumbent_power[n] for n in data.nodes},
                )
            )
    profiles.append(
        (
            "uniform_10mw_4h",
            {node: 10.0 for node in data.nodes},
            {node: 40.0 for node in data.nodes},
        )
    )
    profiles.append(
        (
            "shrunken",
            {n: 0.5 * incumbent_power[n] for n in data.nodes},
            {n: 0.5 * incumbent_energy[n] for n in data.nodes},
        )
    )
    # Every node is probed both as a replacement and as a sole location, so no
    # coordinate of the capacity vector is left unexplored.
    for node in data.nodes:
        for megawatts in (10.0, 40.0):
            for duration in (2.0, 4.0, 8.0):
                replaced_power = dict(incumbent_power)
                replaced_energy = dict(incumbent_energy)
                replaced_power[node] = megawatts
                replaced_energy[node] = megawatts * duration
                profiles.append(
                    (
                        f"replace_{node}_{megawatts:g}mw_{duration:g}h",
                        replaced_power,
                        replaced_energy,
                    )
                )
        relocated_power = dict(zeros)
        relocated_energy = dict(zeros)
        relocated_power[node] = max(10.0, total)
        relocated_energy[node] = 4.0 * max(10.0, total)
        profiles.append((f"relocate_{node}", relocated_power, relocated_energy))

    unique: list[tuple[str, NodalMap, NodalMap]] = []
    seen: set[tuple[float, ...]] = set()
    for label, start_power, start_energy in profiles:
        key = tuple(
            round(value, 9)
            for node in data.nodes
            for value in (start_power[node], start_energy[node])
        )
        if key not in seen:
            seen.add(key)
            unique.append((label, start_power, start_energy))
    return tuple(unique)


def best_response(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
    use_multistart: bool = False,
) -> BestResponse:
    """Search for one investor's best response and return the retained winner.

    Two phases.  *Screening* prices every candidate start in the exact market,
    which is cheap and cannot fail for a feasible capacity.  *Refinement* runs
    the MPEC from the most promising of them.  A screened candidate survives a
    failed refinement, because a feasible profitable deviation disproves an
    equilibrium whether or not IPOPT converged on it.
    """

    started = time.perf_counter()
    unit = investor.investor_id
    starts = capacity_starts(data, config, investor, power, energy)
    if not use_multistart:
        starts = starts[:1]

    records: list[dict[str, object]] = []
    candidates: list[tuple[BestResponse, dict[str, object]]] = []

    def proximal_cost(candidate_power: NodalMap, candidate_energy: NodalMap) -> float:
        return 0.5 * config.proximal_penalty * sum(
            (candidate_power[n] - power[unit, n]) ** 2
            + ((candidate_energy[n] - energy[unit, n]) / config.proximal_energy_scale) ** 2
            for n in data.nodes
        )

    def record(response: BestResponse, phase: str, error: str = "") -> None:
        cost = proximal_cost(response.power, response.energy)
        exact = response.recleared_profit_eur_per_day
        row = {
            "start": response.start_label,
            "phase": phase,
            "termination": response.outcome.termination,
            "has_solution": response.outcome.has_solution,
            "local_optimal": response.outcome.optimal,
            "power": dict(response.power),
            "energy": dict(response.energy),
            "exact_profit": exact,
            "embedded_profit": response.embedded_profit_eur_per_day,
            "proximal_cost": cost,
            "selection_objective": exact - cost,
            "seconds": response.outcome.seconds,
            "error": error or response.reclear_error,
            "max_bound_violation": response.max_bound_violation,
            "selected": False,
        }
        records.append(row)
        if response.outcome.has_solution and math.isfinite(exact):
            candidates.append((response, row))

    for label, start_power, start_energy in starts:
        tick = time.perf_counter()
        error = ""
        try:
            _check_capacity_feasible(
                data, config, investor, start_power, start_energy, power
            )
            candidate_power, candidate_energy = _with_candidate(
                data, power, energy, unit, start_power, start_energy
            )
            market = clear(data, config, candidate_power, candidate_energy)
            exact = recleared_profit(
                market, data, config, investor, candidate_power, candidate_energy
            )
            if not math.isfinite(exact):
                raise ValueError("Nonfinite exact profit")
            screened = True
        except Exception as exc:
            error, exact, screened = str(exc), math.nan, False
        record(
            BestResponse(
                investor_id=unit,
                outcome=SolveOutcome(
                    "screened_feasible" if screened else "screen_failed",
                    screened,
                    False,
                    time.perf_counter() - tick,
                ),
                power=start_power,
                energy=start_energy,
                embedded_profit_eur_per_day=math.nan,
                max_complementarity_product=math.nan,
                max_complementarity_violation=math.nan,
                primal_dual_gap_eur_per_day=math.nan,
                recleared_profit_eur_per_day=exact,
                start_label=label,
            ),
            "screen",
            error,
        )

    ranked = sorted(candidates, key=lambda pair: pair[1]["selection_objective"], reverse=True)
    to_refine = ranked[: config.refinement_starts]
    incumbent = next((pair for pair in ranked if pair[0].start_label == "incumbent"), None)
    if (
        config.always_refine_incumbent
        and config.refinement_starts
        and incumbent is not None
        and all(pair[0].start_label != "incumbent" for pair in to_refine)
    ):
        to_refine.append(incumbent)

    for base, _ in to_refine:
        try:
            response = solve_from_start(
                data, config, investor, power, energy, base.start_label, base.power, base.energy
            )
            record(attach_recleared_profit(data, config, investor, power, energy, response), "refine")
        except Exception as exc:
            record(
                replace(
                    base,
                    outcome=SolveOutcome("refinement_error", False, False, 0.0),
                    recleared_profit_eur_per_day=math.nan,
                ),
                "refine",
                str(exc),
            )

    if not candidates:
        chosen = _failed_response(
            investor,
            data,
            power,
            energy,
            SolveOutcome("no_valid_candidate", False, False, 0.0),
        )
    else:
        best_objective = max(row["selection_objective"] for _, row in candidates)
        tied = [
            pair
            for pair in candidates
            if pair[1]["selection_objective"]
            >= best_objective - config.candidate_selection_tolerance_eur_per_day
        ]
        # Among equally profitable candidates, prefer one IPOPT actually
        # converged on.
        chosen, row = max(tied, key=lambda pair: (pair[0].outcome.optimal, pair[1]["selection_objective"]))
        row["selected"] = True
        row["selection_reason"] = (
            "highest exact profit minus stated proximal cost; "
            "feasible screened candidates retained"
        )
    return replace(
        chosen,
        outcome=replace(chosen.outcome, seconds=time.perf_counter() - started),
        start_records=tuple(records),
    )


def _check_capacity_feasible(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: NodalMap,
    energy: NodalMap,
    full_power: Capacities | None = None,
) -> None:
    if any(not math.isfinite(v) for v in (*power.values(), *energy.values())):
        raise ValueError("Nonfinite capacities")
    for node in data.nodes:
        if (
            power[node] < 0.0
            or energy[node] < investor.ratio_min * power[node] - 1e-7
            or energy[node] > investor.ratio_max * power[node] + 1e-7
        ):
            raise ValueError("Capacity/duration infeasible")
    if full_power is not None:
        candidate = dict(full_power)
        for node in data.nodes:
            candidate[investor.investor_id, node] = power[node]
        _check_profile_node_limits(data, config, candidate)


def _check_profile_node_limits(
    data: MarketData, config: GameConfig, power: Capacities
) -> None:
    """Reject a profile that oversubscribes a node.

    This guard, not the MPEC constraint, is what the exact screen and reclear
    enforce, so an infinite limit is the only way to switch the shared cap off
    without leaving the two disagreeing.
    """

    limits = config.node_limits(data)
    for node in data.nodes:
        total = sum(power.get((unit, node), 0.0) for unit in config.investor_ids)
        if total > limits[node] + 1.0e-7:
            raise ValueError(
                f"Nodal BESS connection limit exceeded at {node}: "
                f"{total:.9g} MW > {limits[node]:.9g} MW"
            )


def attach_recleared_profit(
    data: MarketData,
    config: GameConfig,
    investor: InvestorConfig,
    power: Capacities,
    energy: Capacities,
    response: BestResponse,
) -> BestResponse:
    """Value a proposed capacity in the exact market rather than the MPEC's."""

    if not response.outcome.has_solution:
        return response
    candidate_power, candidate_energy = _with_candidate(
        data, power, energy, investor.investor_id, response.power, response.energy
    )
    try:
        market = clear(data, config, candidate_power, candidate_energy)
        profit = recleared_profit(
            market, data, config, investor, candidate_power, candidate_energy
        )
    except Exception as exc:
        return replace(
            response, recleared_profit_eur_per_day=math.nan, reclear_error=str(exc)
        )
    return replace(response, recleared_profit_eur_per_day=profit)


# --------------------------------------------------------------------------
# Simultaneous responses and the sweep


def solve_all(
    data: MarketData,
    config: GameConfig,
    power: Capacities,
    energy: Capacities,
    use_multistart: bool,
) -> dict[str, BestResponse]:
    """Every investor responds to the same frozen profile."""

    if config.parallel_workers == 1:
        return {
            investor.investor_id: best_response(
                data, config, investor, power, energy, use_multistart
            )
            for investor in config.investors
        }
    responses: dict[str, BestResponse] = {}
    with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {
            executor.submit(
                best_response, data, config, investor, power, energy, use_multistart
            ): investor
            for investor in config.investors
        }
        for future in as_completed(futures):
            investor = futures[future]
            try:
                responses[investor.investor_id] = future.result()
            except Exception as exc:
                responses[investor.investor_id] = _failed_response(
                    investor,
                    data,
                    power,
                    energy,
                    SolveOutcome(f"error: {exc}", False, False, 0.0),
                )
    return responses


def audit_state(
    data: MarketData, config: GameConfig, state: GameState
) -> dict[str, BestResponse]:
    """Re-solve every best response with no proximal penalty and full multistart.

    This is the only comparison that can support an equilibrium claim: the
    regularizer that guided the iteration is switched off entirely.
    """

    return solve_all(
        data, replace(config, proximal_penalty=0.0), dict(state.power), dict(state.energy), True
    )


@dataclass(frozen=True)
class SweepRegret:
    """The Nash gap at one sweep, measured the way the final audit measures it.

    ``profit`` is each investor's exact profit at the profile its rivals were
    frozen at; ``regret`` is what it would have gained by taking the best
    response the sweep just solved, clamped at zero exactly as
    ``audit_equilibrium`` clamps ``profitable_deviation_eur_per_day``.  The two
    are therefore the same quantity, and a run's per-sweep monitor can be read
    on the same axis as its final certificate.

    The marginal cost is one market clear per sweep: 0.65 s measured on the
    smoothed input, against a sweep of roughly 48 s.  The deviation profits are
    not an extra charge at all -- ``best_response`` already re-clears every
    candidate it refines, because it selects on exact profit rather than on the
    MPEC's own.  Regret is therefore about as cheap to watch as the capacity
    residual, and it answers the question the capacity residual cannot.
    """

    profit: dict[str, float]
    regret: dict[str, float]
    max_regret_eur_per_day: float
    relative_regret: float
    measured: bool
    individually_rational: bool
    error: str = ""


_UNMEASURED_REGRET = SweepRegret({}, {}, math.nan, math.nan, False, False)


def sweep_regret(
    data: MarketData,
    config: GameConfig,
    power: Capacities,
    energy: Capacities,
    responses: dict[str, BestResponse],
) -> SweepRegret:
    """Profitable deviation per investor at ``power``/``energy``.

    ``responses`` must be the answers to exactly this profile.  That is what a
    Jacobi sweep produces and what Gauss-Seidel does not: there, each investor
    answered a different, partially updated profile, so no single clear values
    them all.
    """

    try:
        market = clear(data, config, power, energy)
    except Exception as exc:
        return replace(_UNMEASURED_REGRET, error=f"incumbent reclear failed: {exc}")

    profit: dict[str, float] = {}
    regret: dict[str, float] = {}
    for investor in config.investors:
        unit = investor.investor_id
        try:
            profit[unit] = recleared_profit(market, data, config, investor, power, energy)
        except Exception as exc:
            return replace(_UNMEASURED_REGRET, error=f"{unit} incumbent profit: {exc}")
        response = responses.get(unit)
        deviation = response.recleared_profit_eur_per_day if response is not None else math.nan
        # A response that never solved has no deviation to report.  Leave it
        # unmeasured; a missing number must not read as zero regret.  A missing
        # response is recorded rather than raised: killing a several-hour run
        # at its last sweep is a worse failure than carrying one blank sweep.
        regret[unit] = (
            deviation - profit[unit]
            if response is not None
            and response.outcome.has_solution
            and math.isfinite(deviation)
            else math.nan
        )

    if not all(math.isfinite(value) for value in (*profit.values(), *regret.values())):
        return replace(
            _UNMEASURED_REGRET,
            profit=profit,
            regret=regret,
            error="at least one best response had no finite exact profit",
        )

    worst = max(0.0, max(regret.values()))
    scale = max(profit.values())
    return SweepRegret(
        profit=profit,
        regret=regret,
        max_regret_eur_per_day=worst,
        # Relative, so the criterion survives a change of investor portfolios:
        # no MW thresholds, no node names, no absolute EUR figure to retune.
        relative_regret=worst / scale if scale > 0.0 else math.inf,
        measured=True,
        individually_rational=(
            min(profit.values()) >= -config.individual_rationality_tolerance_eur_per_day
        ),
    )


def _profit_change(previous: dict[str, float], current: dict[str, float]) -> float:
    """Largest incumbent profit move between two sweeps, NaN before the second."""

    if not previous or set(previous) != set(current):
        return math.nan
    return max(abs(current[unit] - previous[unit]) for unit in current)


def run_jacobi(
    data: MarketData,
    config: GameConfig,
    *,
    initial: GameState | None = None,
    on_sweep: SweepCallback | None = None,
) -> GameState:
    """Iterate damped simultaneous best responses to a fixed point."""

    validate(data, config)
    state = initial if initial is not None else initial_state(data, config)

    for sweep in range(state.sweep + 1, config.max_sweeps + 1):
        old_power = dict(state.power)
        old_energy = dict(state.energy)
        use_multistart = config.multistart_every_sweeps > 0 and (
            sweep == 1 or sweep % config.multistart_every_sweeps == 0
        )
        responses = solve_all(data, config, old_power, old_energy, use_multistart)

        max_power_deviation = 0.0
        max_energy_deviation = 0.0
        max_product = 0.0
        max_violation = 0.0
        max_gap = 0.0
        for investor in config.investors:
            unit = investor.investor_id
            response = responses[unit]
            for node in data.nodes:
                key = unit, node
                answered = response.outcome.has_solution
                proposed_power = response.power[node] if answered else old_power[key]
                proposed_energy = response.energy[node] if answered else old_energy[key]
                # Convergence is judged on the raw deviation, never on the
                # damped step that is actually taken.
                max_power_deviation = max(
                    max_power_deviation, abs(proposed_power - old_power[key])
                )
                max_energy_deviation = max(
                    max_energy_deviation, abs(proposed_energy - old_energy[key])
                )
                state.power[key] = _damped(old_power[key], proposed_power, config.damping)
                state.energy[key] = _damped(old_energy[key], proposed_energy, config.damping)
                if state.power[key] < config.cleanup_tolerance:
                    state.power[key] = 0.0
                    state.energy[key] = 0.0
            if response.outcome.has_solution:
                max_product = max(max_product, response.max_complementarity_product)
                max_violation = max(max_violation, response.max_complementarity_violation)
                max_gap = max(max_gap, abs(response.primal_dual_gap_eur_per_day))

        all_optimal = all(response.outcome.optimal for response in responses.values())
        # Measured at the profile the responses actually answered, not at the
        # damped profile the sweep is about to move to.
        gap = (
            sweep_regret(data, config, old_power, old_energy, responses)
            if config.convergence_metric in ("profit", "regret")
            else _UNMEASURED_REGRET
        )
        profit_change = _profit_change(state.profit, gap.profit) if gap.measured else math.nan
        numerics_ok = all_optimal and max_violation <= config.solver.tolerance
        # ``hold`` marks a sweep that is evidence neither for convergence nor
        # against it, so it neither counts toward the stop nor resets the count.
        hold = False
        if config.convergence_metric == "regret":
            quiet = (
                numerics_ok
                and gap.measured
                and gap.relative_regret <= config.tolerance_relative_regret
            )
            # Only a multistart sweep searched widely enough for a small regret
            # to mean anything: without it the sweep solved one local NLP from
            # the incumbent and cannot have left the branch it started on.  A
            # large regret is evidence against convergence however it was found,
            # so it still resets the count.
            stable = quiet and use_multistart
            hold = quiet and not use_multistart
        elif config.convergence_metric == "profit":
            stable = (
                numerics_ok
                and gap.measured
                and math.isfinite(profit_change)
                and profit_change <= config.tolerance_profit_eur_per_day
            )
        else:
            stable = (
                numerics_ok
                and max_power_deviation <= config.tolerance_mw
                and max_energy_deviation <= config.tolerance_mwh
            )
        # An investor below zero means the run is unfinished, whatever the
        # metric says: it has parked between best-response branches rather than
        # arrived anywhere an investor would accept.
        if stable and config.require_individual_rationality and gap.measured:
            stable = gap.individually_rational
        if stable:
            state.stable_sweeps += 1
        elif not hold:
            state.stable_sweeps = 0
        state.sweep = sweep
        state.responses = responses
        if gap.measured:
            state.profit = dict(gap.profit)
            state.regret = dict(gap.regret)
        state.history.append(
            _history_row(
                config,
                state,
                responses,
                all_optimal=all_optimal,
                use_multistart=use_multistart,
                max_power_deviation=max_power_deviation,
                max_energy_deviation=max_energy_deviation,
                max_product=max_product,
                max_violation=max_violation,
                max_gap=max_gap,
                gap=gap,
                profit_change=profit_change,
            )
        )
        if on_sweep is not None:
            on_sweep(state)
        if state.stable_sweeps >= config.consecutive_sweeps:
            state.converged = True
            state.stop_reason = _stop_reason(config, state.stable_sweeps, gap, profit_change)
            break

    if not state.converged:
        state.stop_reason = "maximum sweeps reached"
        if config.convergence_metric == "regret" and config.multistart_every_sweeps <= 0:
            # Not a misconfiguration: measuring regret every sweep and stopping
            # on it are different jobs.  With no multistart the run is a monitor
            # -- every sweep records its regret, no sweep can certify a stop, and
            # the final audit remains the certificate.
            state.stop_reason += (
                "; no sweep used multistart, so none could certify a stop on "
                "regret. The per-sweep regret is a lower bound and the "
                "zero-proximal final audit is the certificate."
            )
    return state


def _stop_reason(
    config: GameConfig, stable_sweeps: int, gap: SweepRegret, profit_change: float
) -> str:
    """Name the criterion that fired, with the number that satisfied it."""

    tail = f"for {stable_sweeps} consecutive sweeps"
    if config.convergence_metric == "regret":
        return (
            f"relative regret {gap.relative_regret:.3e} "
            f"(max {gap.max_regret_eur_per_day:.2f} EUR/day) within "
            f"{config.tolerance_relative_regret:.3e} {tail}"
        )
    if config.convergence_metric == "profit":
        return (
            f"incumbent profit moved at most {profit_change:.4f} EUR/day, within "
            f"{config.tolerance_profit_eur_per_day} {tail}"
        )
    return f"best-response deviations within tolerance {tail}"


def run_gauss_seidel(
    data: MarketData,
    config: GameConfig,
    *,
    initial: GameState | None = None,
    on_sweep: SweepCallback | None = None,
) -> GameState:
    """Iterate best responses sequentially in configured investor order.

    Unlike :func:`run_jacobi`, each investor observes the updates already made
    by earlier investors in the same sweep.  The convergence residual is the
    largest raw deviation from the profile visible to that investor when its
    best response was solved.
    """

    validate(data, config)
    if config.convergence_metric != "capacity":
        # Each investor answered a different, partially updated profile, so no
        # single market clear values them all and there is no honest sweep-wide
        # Nash gap to stop on.  Run Jacobi for that, or audit the endpoint.
        raise ValueError(
            f"convergence_metric={config.convergence_metric!r} is only defined for "
            "run_jacobi, where every investor answers the same frozen profile. "
            "Gauss-Seidel supports 'capacity'; use audit_equilibrium on its result."
        )
    state = initial if initial is not None else initial_state(data, config)

    for sweep in range(state.sweep + 1, config.max_sweeps + 1):
        use_multistart = config.multistart_every_sweeps > 0 and (
            sweep == 1 or sweep % config.multistart_every_sweeps == 0
        )
        responses: dict[str, BestResponse] = {}
        max_power_deviation = 0.0
        max_energy_deviation = 0.0
        max_product = 0.0
        max_violation = 0.0
        max_gap = 0.0

        for investor in config.investors:
            unit = investor.investor_id
            response = best_response(
                data,
                config,
                investor,
                dict(state.power),
                dict(state.energy),
                use_multistart,
            )
            responses[unit] = response
            for node in data.nodes:
                key = unit, node
                old_power = state.power[key]
                old_energy = state.energy[key]
                answered = response.outcome.has_solution
                proposed_power = response.power[node] if answered else old_power
                proposed_energy = response.energy[node] if answered else old_energy
                max_power_deviation = max(
                    max_power_deviation, abs(proposed_power - old_power)
                )
                max_energy_deviation = max(
                    max_energy_deviation, abs(proposed_energy - old_energy)
                )
                state.power[key] = _damped(old_power, proposed_power, config.damping)
                state.energy[key] = _damped(old_energy, proposed_energy, config.damping)
                if state.power[key] < config.cleanup_tolerance:
                    state.power[key] = 0.0
                    state.energy[key] = 0.0
            if response.outcome.has_solution:
                max_product = max(max_product, response.max_complementarity_product)
                max_violation = max(max_violation, response.max_complementarity_violation)
                max_gap = max(max_gap, abs(response.primal_dual_gap_eur_per_day))

        state.responses = responses
        all_optimal = all(response.outcome.optimal for response in responses.values())
        stable = (
            all_optimal
            and max_power_deviation <= config.tolerance_mw
            and max_energy_deviation <= config.tolerance_mwh
            and max_violation <= config.solver.tolerance
        )
        state.stable_sweeps = state.stable_sweeps + 1 if stable else 0
        state.sweep = sweep
        state.history.append(
            _history_row(
                config,
                state,
                responses,
                all_optimal=all_optimal,
                use_multistart=use_multistart,
                max_power_deviation=max_power_deviation,
                max_energy_deviation=max_energy_deviation,
                max_product=max_product,
                max_violation=max_violation,
                max_gap=max_gap,
            )
        )
        if on_sweep is not None:
            on_sweep(state)
        if state.stable_sweeps >= config.consecutive_sweeps:
            state.converged = True
            state.stop_reason = (
                f"sequential best-response deviations within tolerance for "
                f"{state.stable_sweeps} consecutive sweeps"
            )
            break

    if not state.converged:
        state.stop_reason = "maximum sweeps reached"
    return state


def _damped(old: float, proposed: float, damping: float) -> float:
    return (1.0 - damping) * old + damping * proposed


def _history_row(
    config: GameConfig,
    state: GameState,
    responses: dict[str, BestResponse],
    *,
    all_optimal: bool,
    use_multistart: bool,
    max_power_deviation: float,
    max_energy_deviation: float,
    max_product: float,
    max_violation: float,
    max_gap: float,
    gap: SweepRegret | None = None,
    profit_change: float = math.nan,
) -> dict[str, object]:
    gap = gap if gap is not None else _UNMEASURED_REGRET
    row: dict[str, object] = {
        "sweep": state.sweep,
        "all_best_responses_optimal": all_optimal,
        "multistart_used": use_multistart,
        "max_raw_power_deviation_mw": max_power_deviation,
        "max_raw_energy_deviation_mwh": max_energy_deviation,
        "max_complementarity_product": max_product,
        "max_complementarity_violation": max_violation,
        "max_absolute_primal_dual_gap_eur_per_day": max_gap,
        "stable_sweeps": state.stable_sweeps,
        "total_power_mw": state.total_power_mw(),
        "total_energy_mwh": state.total_energy_mwh(),
        "solve_seconds": sum(r.outcome.seconds for r in responses.values()),
        # The convergence diagnostics.  max_regret is on the same axis as the
        # final audit's profitable_deviation_eur_per_day, so plotting it against
        # sweep number and marking the audited value is a like-for-like curve.
        "convergence_metric": config.convergence_metric,
        "max_regret_eur_per_day": gap.max_regret_eur_per_day,
        "relative_regret": gap.relative_regret,
        "max_incumbent_profit_change_eur_per_day": profit_change,
        "individually_rational": gap.individually_rational if gap.measured else "",
        "regret_error": gap.error,
    }
    for unit in config.investor_ids:
        row[f"termination_{unit}"] = responses[unit].outcome.termination
        row[f"embedded_profit_{unit}_eur_per_day"] = responses[unit].embedded_profit_eur_per_day
        row[f"recleared_profit_{unit}_eur_per_day"] = responses[unit].recleared_profit_eur_per_day
        row[f"start_{unit}"] = responses[unit].start_label
        # Signed: a negative value means the MPEC returned something worse than
        # the incumbent, which is a solver report, not an economic one.
        row[f"regret_{unit}_eur_per_day"] = gap.regret.get(unit, math.nan)
        row[f"incumbent_profit_{unit}_eur_per_day"] = gap.profit.get(unit, math.nan)
    return row


# --------------------------------------------------------------------------
# Final audit


@dataclass(frozen=True)
class InvestorAudit:
    """One row of the zero-proximal equilibrium audit."""

    investor: str
    audit_valid: bool
    optimal: bool
    termination: str
    max_power_deviation_mw: float
    max_energy_deviation_mwh: float
    embedded_profit_eur_per_day: float
    recleared_profit_eur_per_day: float
    embedded_reclear_profit_gap_eur_per_day: float
    current_recleared_profit_eur_per_day: float
    profitable_deviation_eur_per_day: float
    complementarity_max_product: float
    complementarity_max_violation: float
    max_bound_violation: float
    absolute_primal_dual_gap_eur_per_day: float
    audit_error: str = ""


@dataclass(frozen=True)
class AuditReport:
    """Whether the final profile survives an unregularized deviation check."""

    config: GameConfig
    rows: tuple[InvestorAudit, ...]
    market: pyo.ConcreteModel | None

    @property
    def valid(self) -> bool:
        return all(row.audit_valid for row in self.rows)

    def worst(self, field_name: str) -> float | None:
        """Worst value of one field, or None if any row is not finite.

        A missing number is never treated as a passing one.
        """

        values = [float(getattr(row, field_name)) for row in self.rows]
        return max(values) if values and all(math.isfinite(v) for v in values) else None

    @property
    def passed(self) -> bool:
        """True only if every investor is verifiably at a local best response."""

        config = self.config
        if not self.valid or not all(row.optimal for row in self.rows):
            return False
        profit_tolerance = config.audit_profit_tolerance_eur_per_day
        limits = {
            "max_power_deviation_mw": config.tolerance_mw,
            "max_energy_deviation_mwh": config.tolerance_mwh,
            "profitable_deviation_eur_per_day": profit_tolerance,
            "embedded_reclear_profit_gap_eur_per_day": (
                config.audit_reclear_gap_tolerance_eur_per_day
            ),
            "complementarity_max_violation": config.solver.tolerance,
            "max_bound_violation": config.solver.tolerance,
        }
        for name, limit in limits.items():
            worst = self.worst(name)
            if worst is None or worst > limit:
                return False
        return True


def audit_equilibrium(
    data: MarketData, config: GameConfig, state: GameState
) -> AuditReport:
    """Re-audit the final profile with the regularizer switched off.

    Incumbents and proposed deviations are re-cleared with IPOPT. There is
    no cross-solver gate; the local NLP status, residuals, embedded/recleared
    profit gap and unregularized profitable deviations determine acceptance.
    """

    audit_config = replace(config, proximal_penalty=0.0)

    try:
        market = clear(data, audit_config, state.power, state.energy)
    except Exception as exc:
        state.responses = {}
        return AuditReport(
            config=config,
            rows=tuple(
                _invalid_audit(investor.investor_id, "common_reclear_failed", str(exc))
                for investor in config.investors
            ),
            market=None,
        )

    def profit_of(cfg: GameConfig, investor, power: Capacities, energy: Capacities) -> float:
        try:
            return recleared_profit(
                clear(data, cfg, power, energy), data, cfg, investor, power, energy
            )
        except Exception:
            return math.nan

    current = {
        i.investor_id: recleared_profit(
            market, data, config, i, state.power, state.energy
        )
        for i in config.investors
    }

    responses = audit_state(data, audit_config, state)
    state.responses = responses

    rows = []
    for investor in config.investors:
        unit = investor.investor_id
        response = responses[unit]
        candidate_power, candidate_energy = _with_candidate(
            data, state.power, state.energy, unit, response.power, response.energy
        )
        deviation = math.nan
        if response.outcome.has_solution:
            deviation = profit_of(audit_config, investor, candidate_power, candidate_energy)

        audit_valid = response.outcome.has_solution and all(
            math.isfinite(x)
            for x in (
                deviation,
                current[unit],
                *response.power.values(),
                *response.energy.values(),
            )
        )
        embedded = response.embedded_profit_eur_per_day
        rows.append(
            InvestorAudit(
                investor=unit,
                audit_valid=audit_valid,
                optimal=response.outcome.optimal,
                termination=response.outcome.termination,
                max_power_deviation_mw=max(
                    abs(response.power[n] - state.power[unit, n]) for n in data.nodes
                ),
                max_energy_deviation_mwh=max(
                    abs(response.energy[n] - state.energy[unit, n]) for n in data.nodes
                ),
                embedded_profit_eur_per_day=embedded,
                recleared_profit_eur_per_day=deviation,
                embedded_reclear_profit_gap_eur_per_day=(
                    abs(embedded - deviation)
                    if audit_valid and math.isfinite(embedded)
                    else math.nan
                ),
                current_recleared_profit_eur_per_day=current[unit],
                profitable_deviation_eur_per_day=(
                    max(0.0, deviation - current[unit]) if audit_valid else math.nan
                ),
                complementarity_max_product=response.max_complementarity_product,
                complementarity_max_violation=response.max_complementarity_violation,
                max_bound_violation=response.max_bound_violation,
                absolute_primal_dual_gap_eur_per_day=abs(response.primal_dual_gap_eur_per_day),
            )
        )
    return AuditReport(config=config, rows=tuple(rows), market=market)


def _invalid_audit(investor_id: str, termination: str, error: str) -> InvestorAudit:
    """A row that can never pass, used when the common market fails to clear."""

    return InvestorAudit(
        investor=investor_id,
        audit_valid=False,
        optimal=False,
        termination=termination,
        max_power_deviation_mw=math.nan,
        max_energy_deviation_mwh=math.nan,
        embedded_profit_eur_per_day=math.nan,
        recleared_profit_eur_per_day=math.nan,
        embedded_reclear_profit_gap_eur_per_day=math.nan,
        current_recleared_profit_eur_per_day=math.nan,
        profitable_deviation_eur_per_day=math.nan,
        complementarity_max_product=math.nan,
        complementarity_max_violation=math.nan,
        max_bound_violation=math.nan,
        absolute_primal_dual_gap_eur_per_day=math.nan,
        audit_error=error,
    )
