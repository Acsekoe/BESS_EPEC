# Capacity-only restart

Updated 2026-10-07. This replaces the previous strategic-operation workflow.

## Objective and maintained code

Understand the fixed-capacity lower-level market's optimal solution set before
building another multi-investor equilibrium algorithm. `model/` is the sole
maintained implementation. At the user's request, `access_epec_minimal/`,
`repro_strategic_operation/`, and `strategic_power_3bus_toy/` were deleted.
Historical source remains in Git. Pre-pull uncommitted work remains in the stash
named `Local work preserved before GitHub pull 2026-10-07`; it is not the baseline.
The existing Overleaf documents are historical and have not yet been rewritten.

## Explicit baseline assumptions

- Deterministic one-hour periods, fixed demand, truthful generation marginal costs,
  DC PTDF transmission constraints, and optional physical load shedding at VOLL
  (the current core always includes this feasibility mechanism).
- The ISO controls storage dispatch. Shared inverter limit `charge + discharge <=
  power`; efficiency applies in each direction. The continuous LP can allow
  simultaneous charge/discharge; there is no operating-mode binary.
- SOC is bounded at every state including time zero; initial SOC is free and
  cyclic. This represents a repeated day, not a prescribed starting charge.
- Storage degradation costs `0.5 * degradation * (charge + discharge)` in both
  the market objective and investor profit. Each investor has its own parameter.
  With eta = 0.936 this is 1.002 * d per MWh of a full cycle.
- Since 2026-10-12 `capacities_capex120_curtail_d3.json` uses d = 3 EUR/MWh
  instead of the inherited 15. Reason: capex already pays for the whole system
  over a fixed 15-year life with constant usable MWh, and about 5 500 cycles
  (one per day) lie within LFP cycle life. A per-MWh charge of cell replacement
  cost (Xu et al. 2018; He et al. 2021: 7-17 $/MWh) would pay for the cells
  twice. d prices only the capacity a cycle wears out, replaced by
  augmentation: about 62 000 EUR/MWh (BNEF 2025 stationary pack, $70/kWh at
  1.13 USD/EUR) * 30 % fade / 6 000 cycles = 3.1 EUR/MWh. Cycle counts are
  datasheet values, not verified warranties. NREL ATB uses no variable O&M
  (augmentation in fixed O&M), so d = 0 and d = 10 (replacement cost) are the
  sensitivity bounds.
- Optional `dispatch_cost_slope_eur_per_mwh_per_mw` (default 0) adds
  `0.5 * slope * (charge^2 + discharge^2)` per storage and hour, kept separate
  from d. It is a selection device that makes dispatch unique (like eps for
  prices); physically it resembles current-dependent losses. Battery aging is
  convex in cycle depth, not in power (Xu et al. 2018 neglect current rate for
  grid-scale batteries), so it is not called degradation. The slope is stylized.
  Chosen value 1e-4 EUR/MWh per MW (`capacities_capex120_curtail_d3_slope1e-4.json`):
  about 4 EUR/day at a 173 MW battery (0.06 % of I1's profit); 1e-3 costs about
  10x more, mostly from full-power charging where no tie exists. It does not
  unstick the four-node MPEC bound (see `model/README.md`).
- No demand adjustment, quadratic price selection, access auction, proximal
  penalty, or equilibrium iteration.
- Storage bids truthfully by default: hourly charge bid -0.5 d and discharge
  offer 0.5 d per node. Since 2026-10-13 `--strategic-bids` (both MPECs) lets
  the active investor choose its bids in [-VOLL, VOLL] with eta^2 * offer >= bid;
  rivals' bids are fixed profile inputs. The investor's profit uses its true
  degradation cost, not its bids. Bids need `--balancing-eps > 0`. Bid levels
  are not unique (any bid beyond the price range supports the same dispatch);
  since 2026-10-13 `mpec` selects the bids closest to truthful at the optimal
  capacity, own dispatch and prices (`select_bids`), an assumption about how
  rivals' fixed bids react. With bids, capacity is no longer the only strategy.
- Strategic decisions are nodal MW and MWh with continuous 2--8-hour duration in
  the IEEE-9 profile. Rivals' capacities are fixed during a best response.
- The 1000 MW default shared nodal investment bound is retained as an explicit
  strategy bound; it is not assumed nonbinding without checking the solution.
- I1/I2 are merchants at 8%/12% WACC. I3 owns 80% of wind and 20% of PV; I4 owns
  the remainder. Lifetime 15 years, power cost 6600 EUR/MW, energy cost 18800
  EUR/MWh, and degradation 15 EUR/MWh are carried over from the previous core.
  Capital recovery is divided by 365.25.
- The inherited costs have no source and are about 5-7x too low (about
  21 EUR/kWh all-in). Since 2026-10-11 the sourced cases are
  `capacities_capex120.json` / `capacities_capex155.json`: the energy cost
  carries the BESS system (120 000 or 155 000 EUR/MWh, IRENA 2026 converted at
  the ECB 2025 rate), and the power cost is the grid-access fee (1 EUR/MW
  placeholder). Inverter costs (13-17% of installed cost) stay in the energy
  cost. WACC and lifetime remain unsourced; for degradation see above. The benchmark day is treated as
  representative; the two-hour toy is a mathematical test, not an annual study.
- Since 2026-10-12 an owner may carry a quadratic PV curtailment cost
  (`curtailment_slope_eur_per_mwh_per_mw`, default 0): `share * 0.5 * slope * u^2`
  for `u` MW curtailed, offered truthfully into the market as `-slope * u`.
  Justification: per-MWh support or pay-as-produced contracts lost when
  curtailing, heterogeneous across a portfolio so the cheapest plants are
  curtailed first. The value 0.1 EUR/MWh per MW (I3/I4, both PV plants,
  `capacities_capex120_curtail.json`) is a stylized choice, not sourced.
- Capacity profiles are explicit JSON inputs. Missing capacity nodes mean zero.
  Old `x_power`/`x_energy` entries in the inherited market JSON are not used.

## Readable model structure and interpretation

The implementation follows the urbs convention of named Pyomo variables and
constraints, with explicit equation rule functions. `prepare_input.py` handles
input validation, sets and parameters; `primal_llp.py` defines physical market
equations; `dual_llp.py` writes each stationarity equation and the dual objective;
`mpec.py` writes investment and every SOS1 complementarity pair explicitly.
`solve.py` contains the shared solver interface. The former matrix-assembly
module `market.py` has been removed. CSV coordinate labels remain compatible;
direct Python builders return a model, not a `(model, rows)` pair.

Physical quantities have nonnegative Pyomo domains; net injections are free.
The Lagrangian subtracts `price * nodal_balance_residual` and
`system_price * system_balance_residual`, adds free multipliers times the SOC
transition/cyclic residuals, adds nonnegative upper-bound multipliers times
`quantity - capacity`, and subtracts lower-bound multipliers times `quantity`.
`m.price[n,t]` is directly the LMP. The standalone primal solver's imported
nodal-balance dual also directly equals the LMP. The sign convention and the
SOC boundary terms are written out in `dual_llp.py`.

The MPEC reuses the same primal rule functions, replacing only the active
investor's capacity expressions by investment variables. It is organized as
investor constraints + LLP primal feasibility + LLP stationarity + SOS1
complementarity + physical bounds + investor objective. Since 2026-10-12 each
pair {slack, multiplier} is an SOS1 set (Gurobi branches on slack = 0 or
mu = 0): no binaries and no multiplier bound, so no Big-M assumption. Big-M,
bilinear products, relaxed products, a strong-duality constraint, and the
Ipopt relaxed MPEC (`mpec_relaxed.py`) were compared that day and removed
(evidence in `workflow/complementarity_and_degradation_2026-10-12.md`; code in
git history). All share the same root relaxation; SOS1 solved I1/N9 fastest.
The relaxation is weak: with four candidate nodes the bound stays at its root.
On 2026-10-13 the Ipopt relaxed MPEC returned as `mpec_relaxed.py` (same
blocks, Scholtes products `slack * mu <= epsilon`, epsilon continuation), as a
fast local method that is not a proof. On V6 it reaches the proven merchant
optima and the owners' best Gurobi incumbents in under a minute (`model/README.md`).

Without multiplier bounds the prices are unbounded, so the objective is the
strong-duality profit `profit_linear`: demand payments less market costs and
all other assets' scarcity rents (with the quadratic costs: also the balancing,
curtailment, and rival dispatch-rent terms; concave, so the MPEC is a MIQP with
SOS1 sets). Owned truthful generation earns its capacity scarcity rent. It
equals the direct nodal settlement `profit` (price times quantity) at every
KKT point; the difference is a weighted sum of other assets' complementarity
products. The runner reports both values and their difference, primal/dual
and complementarity residuals, and independently reclears the selected capacity.

Since 2026-10-09 `--balancing-eps eps` optionally adds a price-responsive
balancing injection `b = eps * price` (cost `b^2/(2 eps)`) at every node and
hour (option C of the 2026-10-08 summary). Prices, and hence each investor's
profit, are then unique at every capacity and continuous in capacity; the
market becomes a convex QP. Default `eps = 0` keeps the baseline LP, so the
baseline assumption "no quadratic price selection" still holds by default.
At eps = 1e-4 prices are only weakly pinned numerically, so reclears use
`price = b/eps` from a tight primal QP (Ipopt), not the dual QP.

The single-investor MPEC is **optimistic**: it can choose the market optimum
and supporting dual prices most favorable to that investor. It is not a unique
market selection rule, a pessimistic solution, or a multi-investor equilibrium.

At fixed capacity, `ranges` separately minimizes/maximizes quantities on the
primal optimal face and prices/profits on the dual optimal face. The default
1e-6 EUR objective tolerance includes near-optimal points; use zero to target
the exact face subject to solver tolerances. Ranges are coordinate projections,
not a Cartesian product of jointly attainable endpoints. Unbounded/unresolved
objectives are reported, not clipped to artificial price bounds.

## Next research steps

1. Reproduce the supplied no-storage, symmetric-storage, and analytical toy cases.
2. Sweep a small number of capacity/location/duration axes; identify active-set
   transitions and price/profit intervals, particularly at supply breakpoints.
3. Separate primal multiplicity, dual price multiplicity, and economically
   relevant payoff multiplicity. Free cyclic SOC can add harmless multiplicity.
4. Compare optimistic and pessimistic profit at each fixed profile. A pessimistic
   investment model requires a further optimization layer and is not implemented.
5. Only then specify market selection semantics and study best-response sets.
   Independently optimistic responses may select different supporting prices;
   a common market outcome must be checked when defining an equilibrium.
   Reintroduce multi-investor equilibrium search with independent deviation
   checks; do not interpret a converged iteration or a capacity grid as proof
   of the complete equilibrium spectrum.

See `model/README.md` for runnable commands and `workflow/` for verification.
