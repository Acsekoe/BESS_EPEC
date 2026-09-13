# README long-form archive (cut 2026-09-10)

`README.md` was reduced from 527 lines to a current-state document on
2026-09-10. Nothing below was deleted because it was wrong — it was moved
because it is background, history, or a superseded workaround rather than the
current state of the model. The README now carries the current state; this file
and the `summary_*.md` notes carry the history.

---

## 1. Why the nodal access tariff was removed

The affine nodal access tariff `alpha + beta * Q[n] / node_limit[n]` that this
model used to charge on each investor's own MW was removed, for the same reason
the system-wide access auction before it was removed: it was a symmetric cost
that every investor faced identically, so its only effect on the capacity split
was to compress it towards equal shares. At `beta = 100` and `L = 40` it
supplied an own-curvature of `2*beta/L = 5` EUR/MW^2/day and a marginal charge
near 100 EUR/MW/day, and it produced a ~33/33/33 split at every node. `tests/`
asserts that neither the tariff nor the auction can be reintroduced without the
test failing.

Also removed as dead weight in a capacity-only model: load shedding and VOLL
(never reachable, since the demand-adjustment term already made the market
feasible), the fixed storage capacities in the input file (capacities are always
supplied by the game), and the standalone market-clearing CLI.

## 2. The shared node cap as a GNEP, and how it censors the search

The cap is a *shared* constraint, so it makes the game a GNEP: the investors'
feasible regions are coupled and the multiplier on the cap is not shared. This
can admit multiple equilibria, but neither a continuum nor uniqueness follows
from the cap alone; a Jacobi path merely selects a candidate.

`--node-limit-mw inf` switches it off. Switching it off has to happen in one
place for all three of the MPEC constraint, the candidate screen and the exact
reclear — deactivating only the MPEC constraint leaves the screen and reclear
still rejecting every oversubscribed profile, so a deviation the MPEC is now
free to propose could never be priced. An infinite limit therefore builds no
constraint *and* disarms the profile check.

The cap also censors the search whether or not it binds: with a 40 MW cap and
rivals already holding ~20 MW at a node, every `*_40mw_*` and `relocate_*`
multistart candidate is rejected before it is ever priced (37 of 75 candidates
in the `access_L40_A0_B100_20sweeps` run), so the large-capacity branch is never
seen.

## 3. Demand adjustment at unloaded nodes (the conjured-generation fix)

For positive rho the quadratic demand-adjustment term adds strict curvature in
that variable and can help select an LMP. At the ISO's optimum its stationarity
condition is exactly

```text
DemandAdjustment[n,t] = LMP[n,t] / rho
```

(verified to 1.9e-12). So rho is an *inverse elasticity*: the term behaves as a
demand response with marginal willingness-to-pay `rho * x`, and a small rho
makes it a large one.

**It acts only where there is demand to adjust.** Six of the nine IEEE-9 nodes
carry no load at any hour. Left free at those nodes the term is not adjusting
anything — it is generation conjured from nothing, and at `rho = 25` it was
supplying 213.6 MWh/day there, 1.6 % of system load.
`MarketData.demand_is_adjustable` holds it at zero wherever `demand_el == 0`, in
the exact market and in both MPEC builders alike, and the MPEC *skips its
stationarity condition* at those nodes — imposing `rho * 0 == lam` would pin the
LMP to zero rather than leave it to `net_injection_stationarity`, which is what
actually defines it.

Cleared prices are unchanged by this: the LMP at an unloaded node is set by the
system price and the line duals either way. What changes is that the model no
longer creates energy.

## 4. rho selects the congestion regime at the threshold

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

## 5. Historical positive-rho workaround (superseded by rho = 0)

Retained only to explain archived runs. New work uses rho zero as the primary
case and a positive rho only as a separately reported elasticity sensitivity.

Merit-order smoothing is the job of the nine-step thermal ladder (23–68
EUR/MWh), not of rho. The archived workaround was
`--demand-adjustment-penalty-eur-per-mw2 5000` together with
`--complementarity-epsilon 1e-6`.

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

## 6. Convergence-metric detail

`regret` is the Nash condition itself, and it is the *same quantity* the final
audit reports as `profitable_deviation_eur_per_day` — so the per-sweep column
`max_regret_eur_per_day` and the endpoint certificate plot on one axis. It costs
one extra market clear per sweep (0.65 s measured, against a sweep of roughly
48 s): the deviation profits themselves are already paid for, because
`best_response` re-clears every candidate it refines in order to select on exact
profit. It requires `--proximal-penalty 0.0`, since a proximal term pulls each
best response back toward the incumbent and would understate the very deviation
being certified.

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

## 7. Oracle-efficient regret search — mechanism detail

Each inner iteration obtains one local response per investor in parallel and
uses a bounded archive of old response branches to score damped simultaneous,
one-investor and two-investor moves by exact market re-clearing. This restricted
Nikaido–Isoda regret is explicitly reported as a lower bound. A targeted
one-investor multistart is triggered periodically or after a rejected step. It
refines only the requested number of best screened branches and does not
automatically add an incumbent refinement; an all-investor multistart audit is
triggered near the requested epsilon, at an optional `--full-audit-interval`,
and for final certification. Only the fresh all-investor audit can set
`converged=true`. `cost_ledger.json` separates local best responses, targeted
multistarts, full audits, MPEC refinements and market re-clears so computational
savings are auditable rather than inferred.

Its primary descent merit is the sum of exact-recleared regrets. It first
line-searches the simultaneous best-response direction, then tries a
single-player and a two-player fallback. Maximum regret is used only for the
reported epsilon-equilibrium test. Active-set changes trigger local alpha
bisection, retained response branches accumulate across iterations, and a
stalled run records a short best-response cycle diagnostic plus an empirical
small-perturbation regret-resolution check.

## 8. Congestion-relief threshold — the worked example

In the `decoupled_B0_q50_20sweeps` run the iteration stopped at a 0.26 MW
capacity residual while a 0.14 MW move was worth 67 840 EUR/day to I3. The
zero-proximal audit is what catches this, and it is the only reason that run is
correctly reported as `converged: false`. Crossing the threshold takes I1's N8
position from +5 447 to -451 EUR/day.

## 9. Relation to `access_epec_minimal`

This folder remains distinct from the former system-wide access auction in
`old/access_epec_minimal/`: there is no global supply quantity `K`, projected
price update, or independent access-allocation variable. The present extension
is a physical nodal connection cap on installed MW.
