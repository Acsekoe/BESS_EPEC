# Capacity-only BESS EPEC

The optional deterministic aFRR extension is available with `--market afrr`.
The default `--market energy-only` still uses the **unchanged** `model/mpec.py`;
the extension lives in `model/mpec_afrr.py`. A complete pre-extension source
backup is in `backups/energy_only_20260907_153308.zip`.
See [aFRR formulation, KKT derivation, units and commands](afrr_formulation.md).
Both quadratic investment coefficients can remain exactly zero. The aFRR
model uses six four-hour products, 50 MW per direction, generator headroom,
30-minute SOC buffers at all block endpoints, and expected activation wear.
It is a marginal-pricing capacity benchmark, not the Austrian pay-as-bid
auction or a stochastic balancing-energy model. The description below
documents the preserved energy-only baseline.

Three investors simultaneously choose nodal battery power `X_power[i,n]` and
energy `X_energy[i,n]` on the IEEE-9 network. Conditional on those capacities
the ISO clears a 24-hour market at minimum system cost. Each investor
maximises daily profit *through* that clearing, so the game is an EPEC: an
equilibrium problem whose every player solves an MPEC.

There is no quantity or price bidding. Capacity is the only strategic
variable. Investors share an optional physical nodal connection limit.

## Formulation

**Upper level** — one per investor, maximise daily profit:

```text
max   spot_revenue + generation_rent - degradation - daily_capex
s.t.  X_power[n] >= 0
      sum_i X_power[i,n] <= node_connection_limit[n]
      2 * X_power[n] <= X_energy[n] <= 8 * X_power[n]
      lower-level optimality conditions
```

`daily_capex` is the CRF-annualised overnight cost (6 600 EUR/MW,
18 800 EUR/MWh, 15-year life) divided by 365.25. `generation_rent` is the
owned renewable share times (LMP − true cost) times dispatch, which is what
makes I2 and I3 care about prices they do not set. Optional common quadratic
capacity curvature `qP`, `qE` defaults to zero.

`--node-limit-mw` overrides the cap uniformly for sensitivity cases; when it
is omitted the node-specific values in the input are used.

The affine nodal access tariff `alpha + beta * Q[n] / node_limit[n]` that this
model used to charge on each investor's own MW has been **removed**, for the
same reason the system-wide access auction before it was removed: it was a
symmetric cost that every investor faced identically, so its only effect on
the capacity split was to compress it towards equal shares. At `beta = 100`
and `L = 40` it supplied an own-curvature of `2*beta/L = 5` EUR/MW^2/day and
a marginal charge near 100 EUR/MW/day, and it produced a ~33/33/33 split at
every node. `tests/` asserts that neither the tariff nor the auction can be
reintroduced without the test failing.

The cap is a *shared* constraint, so it makes the game a GNEP: the investors'
feasible regions are coupled, the multiplier on the cap is not shared, and a
Jacobi iteration selects one arbitrary point from the continuum rather than a
unique equilibrium. `--node-limit-mw inf` switches it off. Switching it off
has to happen in one place for all three of the MPEC constraint, the candidate
screen and the exact reclear — deactivating only the MPEC constraint leaves
the screen and reclear still rejecting every oversubscribed profile, so a
deviation the MPEC is now free to propose could never be priced. An infinite
limit therefore builds no constraint *and* disarms the profile check.

The cap also censors the search whether or not it binds: with a 40 MW cap and
rivals already holding ~20 MW at a node, every `*_40mw_*` and `relocate_*`
multistart candidate is rejected before it is ever priced (37 of 75 candidates
in the `access_L40_A0_B100_20sweeps` run), so the large-capacity branch is
never seen.

`qP` and `qE` are **overnight portfolio-wide** coefficients. The quadratic
terms are `0.5*qP*(sum_n X_power[n])^2` and
`0.5*qE*(sum_n X_energy[n])^2`, then the CRF annualisation divides them by
365.25. Consequently the friction controls each investor's total scale but is
neutral to how that total is distributed across nodes.

**Lower level** — the ISO minimises daily system cost:

```text
min   sum_g,t offer[g] * P_gen[g,t]
    + sum_i,n,t 0.5 * degradation[i] * (P_charge + P_discharge)
    + 0.5 * penalty * sum_n,t DemandAdjustment[n,t]^2
s.t.  nodal balance, system balance, PTDF line limits (both directions),
      generation capacity, P_charge <= X_power, P_discharge <= X_power,
      SOC transition with efficiency eta, SOC <= X_energy, cyclic SOC.
```

Charge and discharge have separate power bounds; there is no shared-inverter
constraint. The problem is a convex QP, so its KKT conditions and its strong
duality are both exact, and the dual of nodal balance is the LMP.

### The demand-adjustment penalty rho

The quadratic demand-adjustment term exists to select a unique LMP, and it is
not a strategic quantity anyone withholds. It is **not** free, though, and it is
not a rounding term. At the ISO's optimum its stationarity condition is exactly

```text
DemandAdjustment[n,t] = LMP[n,t] / rho
```

(verified to 1.9e-12). So rho is an *inverse elasticity*: the term behaves as a
demand response with marginal willingness-to-pay `rho * x`, and a small rho
makes it a large one.

**It acts only where there is demand to adjust.** Six of the nine IEEE-9 nodes
carry no load at any hour. Left free at those nodes the term is not adjusting
anything — it is generation conjured from nothing, and at `rho = 25` it was
supplying 213.6 MWh/day there, 1.6 % of system load. `MarketData.demand_is_adjustable`
now holds it at zero wherever `demand_el == 0`, in the exact market and in both
MPEC builders alike, and the MPEC *skips its stationarity condition* at those
nodes — imposing `rho * 0 == lam` would pin the LMP to zero rather than leave it
to `net_injection_stationarity`, which is what actually defines it.

Cleared prices are unchanged by this: the LMP at an unloaded node is set by the
system price and the line duals either way. What changes is that the model no
longer creates energy.

### rho selects the congestion regime at the threshold

Away from the corridor limit the clearing is invariant to rho. **On** the limit
it is not: the term supplies LMP/rho MW at each demand node, and near the
threshold that trickle decides whether the corridor clears. Scaling the storage
profile through the threshold, the N8 price at hour 14:

| total MW | rho 100 | 500 | 2500 | 5000 | 20 000 | 100 000 |
|---|---|---|---|---|---|---|
| 169.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 178.4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **187.8** | **0.00** | **8.33** | **21.80** | 21.80 | 21.80 | 21.80 |
| 197.2 | 23.00 | 23.00 | 23.00 | 23.00 | 23.00 | 23.00 |
| 225.4 | 23.00 | 23.00 | 23.00 | 23.00 | 23.00 | 23.00 |

Every rho agrees below 178 MW and above 197 MW. Only at 187.8 — exactly where
the iteration lands — does rho pick the answer, and it is stable from
**rho = 2500** upward. An earlier version of this section claimed the clearing
was rho-invariant outright; that was measured at a 214 MW profile which sits
clear of the threshold, and it does not generalise.

Load served is the other thing rho controls: 99.19 % at rho = 25, 99.96 % at
500, 99.996 % at 5000.

### Recommended setting

Merit-order smoothing is the job of the nine-step thermal ladder (23–68
EUR/MWh), not of rho. Use **`--demand-adjustment-penalty-eur-per-mw2 5000`
together with `--complementarity-epsilon 1e-6`.**

rho = 5000 is clear of the regime-selection band and conjures 0.53 MWh/day;
solver behaviour is flat out to rho = 100 000 (3.6 s to 4.3 s, duals unchanged).

The tighter epsilon is not optional at that rho. Near the threshold the lower
level is nearly degenerate and the elastic term had been *cushioning* the
Scholtes relaxation; remove the cushion and the relaxation error grows. It is
linear in epsilon, so tightening recovers it:

| rho | eps | fixed-capacity price error | profit error EUR/day |
|---|---|---|---|
| 500 | 1e-4 | 0.024968 | 50.88 |
| 500 | 1e-6 | 0.000249 | 0.51 |
| 5000 | 1e-4 | 0.131125 | 205.20 |
| **5000** | **1e-6** | **0.001263** | **1.98** |

1.98 EUR/day is well inside the audit's 10.0 gap tolerance. The cost is roughly
a 2x slower MPEC solve. rho = 500 with eps = 1e-6 has better numerics still
(0.51) but solves a market that is wrong at the threshold, pricing N8 at 8.33
where the inelastic-demand answer is 21.80.

The effective rho is recorded in `run_config.json`, because once it can be
overridden the input hash alone no longer identifies the market.

**Embedding** — `--formulation relaxed-kkt` (default) bounds every
complementarity product by `epsilon = 1e-4` (Scholtes); `--formulation
strong-duality` imposes the exact `primal == dual` equality instead. Both are
generated from the same list of products, so they cannot drift apart.

**Solution** — damped Jacobi: all three investors respond to the same frozen
profile, then move `--damping` of the way towards their answers. A moving
quadratic proximal term centred on the previous sweep stabilises the
iteration; it is always subtracted explicitly and never folded into a reported
profit. The final audit re-solves everything with the penalty at zero.

## Modules

| File | Role |
| --- | --- |
| [model/run_model.py](model/run_model.py) | CLI: build the config, run the game, write the outputs |
| [model/capacity_game.py](model/capacity_game.py) | Best responses, multistart search, damped Jacobi sweep, final audit |
| [model/regret_guided_search.py](model/regret_guided_search.py) | Line-searched simultaneous relaxation on the Nikaido--Isoda regret sum, fallbacks, breakpoint refinement and certification |
| [model/mpec.py](model/mpec.py) | One investor's capacity MPEC — primal, duals, complementarity, profit |
| [model/iso_market.py](model/iso_market.py) | The ISO cost-minimisation QP and the settlement it implies |
| [model/investors.py](model/investors.py) | Investor parameters, capital recovery, the three-investor population |
| [model/market_data.py](model/market_data.py) | `MarketData` and the validated loader |
| [model/solvers.py](model/solvers.py) | IPOPT solves and bound-violation check |
| [model/reporting.py](model/reporting.py) | CSV and JSON serialisation, nothing else |
| [model/input/market_data.json](model/input/market_data.json) | IEEE-9 network, demand, generation, PTDF |
| [tests/](tests/) | Regression tests for the failures that would otherwise be silent |

Dependencies run one way: `run_model` → `reporting`, `capacity_game` →
`mpec`, `iso_market` → `investors`, `market_data`, `solvers`.

Investors: `I1` merchant, `I2` 100% wind owner, and `I3` 100% PV owner. All
three use an 8% WACC; I2 owns every wind unit and I3 owns every PV unit.

The input JSON contains the default `node_connection_limit` for every node.
The cap is enforced against the active investor plus all frozen rival
capacities in every best response, exact candidate screen, and final audit.

## Environment

- Python 3.13, Pyomo 6.10.1 — see `requirements.txt`.
- IPOPT 3.13.2 with MA57. On this machine it is at
  `%LOCALAPPDATA%\idaes\bin\ipopt.exe`; put that directory on `PATH`, or set
  `IPOPT_EXECUTABLE`.
- All market re-clears and MPECs use IPOPT; no second solver is required.

## Run

```powershell
$env:PATH = "$env:PATH;$env:LOCALAPPDATA\idaes\bin"
python model/run_model.py `
  --max-sweeps 60 `
  --damping 0.15 `
  --parallel-workers 4 `
  --output-dir model/output/capacity_baseline
```

The reproducible regret-guided search is a separate one-command driver.  It
starts at 5 MW / 15 MWh per investor and node unless an explicit diagnostic
warm start is supplied:

```powershell
$env:PATH = "$env:PATH;$env:LOCALAPPDATA\idaes\bin"
python model/regret_guided_search.py `
  --epsilon-eur-per-day 20 `
  --parallel-workers 4 `
  --output-dir model/output/regret_guided
```

For computationally tractable searches from the same declared symmetric
profile, use the two-loop oracle-efficient driver:

```powershell
python model/oracle_efficient_regret_search.py `
  --epsilon-eur-per-day 20 `
  --parallel-workers 3 `
  --targeted-multistart-interval 8 `
  --targeted-refinement-starts 1 `
  --targeted-max-solve-seconds 120 `
  --full-audit-interval 0 `
  --output-dir model/output/oracle_efficient_regret
```

Each inner iteration obtains one local response per investor in parallel and
uses a bounded archive of old response branches to score damped simultaneous,
one-investor and two-investor moves by exact market re-clearing.  This
restricted Nikaido--Isoda regret is explicitly reported as a lower bound.  A
targeted one-investor multistart is triggered periodically or after a rejected
step.  It refines only the requested number of best screened branches and does
not automatically add an incumbent refinement; an all-investor multistart audit
is triggered near the requested epsilon,
at an optional `--full-audit-interval`, and for final certification.  Only the
fresh all-investor audit can set `converged=true`.  `cost_ledger.json` separates
local best responses, targeted multistarts, full audits, MPEC refinements and
market re-clears so computational savings are auditable rather than inferred.

Its primary descent merit is the sum of exact-recleared regrets.  It first
line-searches the simultaneous best-response direction, then tries a
single-player and a two-player fallback.  Maximum regret is used only for the
reported epsilon-equilibrium test.  Active-set changes trigger local alpha
bisection, retained response branches accumulate across iterations, and a
stalled run records a short best-response cycle diagnostic plus an empirical
small-perturbation regret-resolution check.

`python model/run_model.py --help` groups the options by what they affect:
formulation, iteration, best-response search, audit, solver.

`--initial-capacities <csv>` starts the iteration from a stated profile in the
`final_capacities.csv` format (`investor,node,power_mw,energy_mwh`). This is
not a convenience: see *The congestion-relief threshold* below, which is why
the starting profile is a modelling choice rather than an implementation
detail.

Exit status is `0` only for an equilibrium candidate — the iteration converged
*and* the zero-proximal audit passed. Anything else exits `1`.

The output directory holds `run_config.json` (including the SHA-256 of the
input), `checkpoint.json`, `history.csv`, `current_capacities.csv`,
`final_capacities.csv`, `final_market.csv`, `profit_decomposition.csv`,
`final_audit.csv`, the per-sweep `starts_*.json` search records, `summary.json`,
and the two trajectories `capacity_by_investor_node_by_sweep.csv` and
`capacity_totals_by_investor_by_sweep.csv` (sweep 0 is the starting profile).

## Tests

```powershell
$env:PATH = "$env:PATH;$env:LOCALAPPDATA\idaes\bin"
python -m pytest -q tests/
```

## What the numbers do and do not show

Convergence is judged on raw best-response quantities, never on the damped step
actually taken. Any candidate equilibrium is re-audited with the proximal
penalty at zero, and every capacity a solver proposes is re-cleared in the
exact market before being compared to anything.

`--convergence-metric` chooses *which* quantity has to settle:

| metric | stops when | cost | is it a Nash gap? |
|---|---|---|---|
| `capacity` (default) | raw MW/MWh best-response deviations fall inside `--tolerance-mw` / `--tolerance-mwh` | free | no — see "The congestion-relief threshold" |
| `regret` | `max_i r_i / max_i profit_i` falls inside `--tolerance-relative-regret` | 0.65 s/sweep | yes |
| `profit` | incumbent payoffs move less than `--tolerance-profit-eur-per-day` between sweeps | 0.65 s/sweep | no |

`regret` is the Nash condition itself, and it is the *same quantity* the final
audit reports as `profitable_deviation_eur_per_day` — so the per-sweep column
`max_regret_eur_per_day` and the endpoint certificate plot on one axis. It
costs one extra market clear per sweep (0.65 s measured, against a sweep of
roughly 48 s): the deviation profits themselves are already paid for, because
`best_response` re-clears every candidate it refines in order to select on
exact profit. It requires `--proximal-penalty 0.0`, since a proximal term pulls
each best response back toward the incumbent and would understate the very
deviation being certified.

Measuring regret and *stopping* on it are different jobs, and the flags separate
them. Every sweep records its regret either way — that is the convergence curve,
and it costs 0.65 s. Only a **multistart** sweep can certify a stop: without
multistart a sweep solves a single local NLP from the incumbent, so a small
deviation says only that it did not leave the branch it started on, which is a
lower bound and not a Nash gap. A quiet single-start sweep therefore holds the
stability counter rather than advancing or clearing it; a large regret resets it
however it was found.

With `--multistart-every-sweeps 0` the regret metric is a pure monitor: no sweep
can certify a stop, the run goes to `--max-sweeps`, and the zero-proximal final
audit remains the certificate. That is the right setting when the trajectory is
the deliverable — a run that halts early cannot draw a regret-versus-sweep
curve — and it is what the 174/254-sweep chain used. Turn multistart on during
sweeps only if you want the run to stop by itself, and note that it also lets
the iterate jump between branches mid-run, which is what damping is there to
suppress.

`profit` is not a Nash gap. Payoffs are also flat in the dead zone *between*
best-response branches, where nobody is at a best response and no deviation has
been tested — the damped symmetric runs parked exactly there, at ~250 MW with
I1 losing 1 819 EUR/day. Both payoff metrics therefore refuse to stop while any
investor is below zero profit; pass `--allow-negative-profit-stop` to override
that, which you should not need.

Payoff metrics need `--update-scheme jacobi`. Under Gauss-Seidel each investor
answered a different, partially updated profile, so no single market clear
values them all and there is no honest sweep-wide gap to stop on.

IPOPT certifies local NLP solutions only. A converged Jacobi fixed point is
therefore not a proof of global best-response optimality; the multistart
search is a necessary check, not a sufficient one. `summary.json` says so
explicitly with `global_best_response_certified: false`.

The audit re-clears the incumbent and proposed deviations with IPOPT and
compares the embedded MPEC profit with the re-cleared profit. Market QPs
use tight tolerances with objective scaling disabled so residuals are small
in original EUR units. Final acceptance also requires optimal NLP status,
primal/KKT feasibility and no material unregularized profitable deviation.


## The congestion-relief threshold

On this input the profit surface is not smooth in capacity, and that dominates
everything else about the game.

Midday PV (hours 11-16) congests the corridor into the N3-N6-N8-N9 pocket and
pins the LMP there at ~0 EUR/MWh in hours 13-15, while N1, N2, N4, N5 and N7
stay at 52-60. Storage in the pocket absorbs the surplus and releases the
price. That release is a *threshold*, not a gradient: over roughly 1-2 MW of
aggregate pocket capacity the mean pocket LMP steps from ~46 to ~54 EUR/MWh,
and because I3 earns `LMP * PV_dispatch` on ~4 500 MWh/day of PV, its daily
rent steps by more than 100 000 EUR. Inside that step the exact marginal
profit of a megawatt is of order 1e5 EUR/MW/day.

The two sides of the threshold are opposed interests, not a coordination
problem:

* I3 (PV owner) wants the pocket above the threshold — the step more than
  doubles its generation rent;
* I1 (merchant) wants it below — relieving the congestion collapses the very
  price spread its arbitrage lives on. Crossing takes I1's N8 position from
  +5 447 to -451 EUR/day.

A damped Jacobi iteration parks the profile *inside* the step, where no
investor is at a best response, and then reports convergence because the
capacity residual is small. It is small: in the `decoupled_B0_q50_20sweeps`
run the iteration stopped at a 0.26 MW residual while a 0.14 MW move was worth
67 840 EUR/day to I3. The zero-proximal audit is what catches this, and it is
the only reason that run is correctly reported as `converged: false`.

Two consequences for how this model is used:

1. **The capacity-norm stopping rule is not sufficient here.** `--tolerance-mw`
   and `--tolerance-mwh` cannot see a cliff: a 0.14 MW move worth 67 840 EUR/day
   looks like convergence to them. Use `--convergence-metric regret`, which
   measures the cliff directly and costs 0.65 s per sweep, and read
   `max_regret_eur_per_day` in `history.csv` rather than the capacity residual.
   `final_audit.csv` and `summary.json`'s
   `final_profitable_deviation_eur_per_day` remain the certificate; the regret
   column is the same quantity watched continuously rather than once at the end.
2. **The starting profile selects the basin.** Use `--initial-capacities` to
   enter each side deliberately and report both, rather than letting the
   symmetric default decide.

## Relation to `access_epec_minimal`

This folder remains distinct from the former system-wide access auction in
`access_epec_minimal/`: there is no global supply quantity `K`, projected price
update, or independent access-allocation variable. The present extension is a
physical nodal connection cap on installed MW. The nodal tariff that briefly
replaced that auction has since been removed as well; see *Formulation*.

Also removed as dead weight in a capacity-only model: load shedding and VOLL
(never reachable, since the demand-adjustment term already makes the market
feasible), the fixed storage capacities in the input file (capacities are
always supplied by the game), and the standalone market-clearing CLI.
