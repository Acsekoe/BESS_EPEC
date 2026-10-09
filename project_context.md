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
- No demand adjustment, quadratic price selection, strategic bidding, access
  auction, proximal penalty, or equilibrium iteration.
- Strategic decisions are nodal MW and MWh with continuous 2--8-hour duration in
  the IEEE-9 profile. Rivals' capacities are fixed during a best response.
- The 1000 MW default shared nodal investment bound is retained as an explicit
  strategy bound; it is not assumed nonbinding without checking the solution.
- I1/I2 are merchants at 8%/12% WACC. I3 owns 80% of wind and 20% of PV; I4 owns
  the remainder. Lifetime 15 years, power cost 6600 EUR/MW, energy cost 18800
  EUR/MWh, and degradation 15 EUR/MWh are carried over from the previous core.
  Capital recovery is divided by 365.25. The benchmark day is treated as
  representative; the two-hour toy is a mathematical test, not an annual study.
- Capacity profiles are explicit JSON inputs. Missing capacity nodes mean zero.
  Old `x_power`/`x_energy` entries in the inherited market JSON are not used.

## Readable model structure and interpretation

The implementation follows the urbs convention of named Pyomo variables and
constraints, with explicit equation rule functions. `prepare_input.py` handles
input validation, sets and parameters; `primal_llp.py` defines physical market
equations; `dual_llp.py` writes each stationarity equation and the dual objective;
`mpec.py` writes investment and every Big-M complementarity pair explicitly.
`solve.py` contains the shared solver interface. The former matrix-assembly
module `market.py` has been removed. CLI flags and CSV coordinate labels remain
compatible; direct Python builders now return a model, not a `(model, rows)` pair.

Physical quantities have nonnegative Pyomo domains; net injections are free.
The Lagrangian subtracts `price * nodal_balance_residual` and
`system_price * system_balance_residual`, adds free multipliers times the SOC
transition/cyclic residuals, adds nonnegative upper-bound multipliers times
`quantity - capacity`, and subtracts lower-bound multipliers times `quantity`.
`m.price[n,t]` is directly the LMP. The standalone primal solver's imported
nodal-balance dual also directly equals the LMP. The sign convention and the
SOC boundary terms are written out in `dual_llp.py`.

The MPEC reuses the same primal rule functions, replacing only the active
investor's capacity expressions by investment variables. Binary Big-M disjunctions enforce
slack/multiplier complementarity. Primal slack bounds follow from physical
capacity limits; the user-specified multiplier bound is a numerical assumption,
not an economic price cap. Global optimality of this bounded MPEC does not prove
that the chosen multiplier bound preserves every economically relevant solution.
Check bound sensitivity; no saturated multiplier is not itself a proof.

Since 2026-10-08 the MPEC is organized as investor objective + investor
constraints + LLP primal feasibility + LLP stationarity + Big-M complementarity
+ variable bounds. The default objective is the direct nodal settlement
(price times quantity, bilinear), solved to global optimality by Gurobi with
`NonConvex=2`. Price bounds are derived from load-shed and generator
stationarity with multipliers in `[0, dual_m]`, so they add no assumption
beyond the Big-M.

Strong duality gives the same profit as a linear expression (`profit_linear`):
demand payments less market costs and all other assets' scarcity rents. Owned
truthful generation earns its capacity scarcity rent. The two forms are equal at
every KKT point, so both objectives have the same optimum; `--objective linear`
solves the MILP form. Gurobi solves every model (LPs, MILPs, bilinear MPECs).
The bilinear model also contains
`profit == profit_linear` as a valid cut: the difference is a weighted sum of
other assets' complementarity products, so the cut removes no feasible point but
makes Gurobi's bound usable (I1/N6: global optimum in under 1 s instead of no
usable bound after 15 minutes). The runner reports both values and their
difference, primal/dual and complementarity residuals, and independently
reclears the selected capacity.

Since 2026-10-09 `mpec_relaxed.py` is a second MPEC version for machines
without a full Gurobi licence. It keeps all blocks of `mpec.py` and the same
`mu <= dual_m` and price bounds, but replaces Big-M by `0 <= slack * mu <=
epsilon` (Scholtes) and is solved by Ipopt over a decreasing epsilon sequence,
starting from an exact fixed-capacity market optimum. It gives a **local**
optimum with approximate complementarity: no global optimality proof. Its
selected capacity is recleared exactly and its exact profit range reported.
The equality `slack * mu = epsilon` is infeasible for pairs with structurally
zero slack and is not used. An August Scholtes/Ipopt variant
(`mpec_relaxed_kkt.py`, commit `45ceb8a`) was removed in the reset; this is a
new implementation on the current model.

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
