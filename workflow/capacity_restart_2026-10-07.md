# Capacity-only restart: implementation and first evidence

## Objective and decisions

Restart from the fixed-demand spot-market LLP; run market clearing and a single
capacity MPEC independently, then examine multiplicity before equilibrium work.
GitHub was pulled to `2d4811c`; earlier local work remains in the named pre-pull
stash. The user explicitly requested deletion of the three old model trees,
not archiving. Their tracked files are deleted; nothing is committed or pushed.

`model/` now contains four maintained Python modules, analytical tests, and
explicit market/capacity inputs. The primal, dual, and MPEC share the same LP
rows. Removed features include demand-adjustment price selection, strategic
bidding, access auctions, proximal penalties, and Jacobi iteration. A shared
inverter constraint, truthful generation, bounded load shedding, cyclic free
initial SOC, and degradation in both clearing and settlement are explicit.
Read `project_context.md` and `model/README.md` for equations and commands.

## Verification

- Eight analytical unit tests pass: price intervals at a supply kink, dispatch
  multiplicity with unique price, load shedding and price sign, optimistic MPEC
  and independent reclearing, rival degradation/owned generation settlement,
  shared inverter/cyclic SOC, congestion prices, and unbounded dual prices.
- No-storage IEEE-9 market cost: **394587.85188079 EUR/day**, zero shedding.
  Primal/dual gap zero; maximum primal residual below 8e-14.
- Two identical 10 MW / 30 MWh batteries (I1/I2, N6): cost
  **391977.22111156 EUR/day**, zero shedding, primal/dual gap zero.
  80 of 146 storage coordinates vary by more than 1e-3 MW/MWh across the
  1e-6-EUR near-optimal face. All 216 price ranges and four profit ranges have
  width below 1e-3 in their respective units. This distinguishes dispatch
  ambiguity from price/payoff ambiguity.
- Analytical two-hour toy: optimal investment **5 MW / 5 MWh**; optimistic
  profit **140 EUR**, but fixed-capacity profit interval **[-10, 140] EUR**.
  Primal/dual and reclear gaps zero. Also verified through the CLI.
- IEEE-9, I1 investment restricted to N6, all rivals at zero:
  **78.75960903 MW / 260.53146197 MWh**, optimistic profit
  **8981.97503610 EUR/day**. Reclear cost gap below 5e-10 EUR; direct settlement
  identity error below 5e-9 EUR; constraint violation below 6e-8.
- Repeated the N6 MILP with multiplier bounds 100000 and 20000: same economic
  result; largest multiplier about 10000, no bound saturation. This is a useful
  sensitivity check, not a proof of globally valid Big-M bounds.
- At the N6 optimum, independent ranges with **zero objective tolerance** find
  **28 nonunique node-hour prices** and 32 varying storage coordinates. I1's
  profit interval is **[2519.05404036, 8981.97503603] EUR/day**. I3 and I4 profits
  also vary. Range endpoints are marginal, not necessarily jointly attainable.
- The unrestricted nine-node MPEC hit the **60-second time limit**. Reported
  MILP profit bounds: approximately **[9836.95, 19006.67] EUR/day**. It is not
  solved to optimality; the runner exports no claimed optimal capacity profile.
- `git diff --check` passed. Generated outputs and Python/LaTeX caches are ignored.
  Existing Overleaf source and intentional PDFs were not edited.

Results are in ignored `model/output/`: `no_storage`, `no_storage_ranges`,
`fixed_storage`, `fixed_storage_ranges`, `toy_mpec`, `toy_mpec_verified`,
`toy_ranges`, `mpec_i1`, `mpec_i1_n6`, `mpec_i1_n6_m20000`, and
`mpec_i1_n6_ranges`. The later CLI verification gate was checked with
`toy_mpec_verified`; earlier run summaries still contain the underlying
numerical verification fields but not the new boolean flag.

## Next steps

1. Inspect the N6 chosen-capacity price/profit ranges. Separate cyclic-SOC freedom
   and identical-storage allocation from economically relevant dual ambiguity.
2. Sweep MW/MWh in a neighborhood of that capacity, including points on both
   sides of the breakpoint. Save market value, price ranges, and profit bounds;
   distinguish sharp transitions from solver tolerance effects.
3. Compare locations and duration slices, then nonzero rival profiles.
4. Choose explicit market selection semantics before EPEC iteration. Separately
   optimized optimistic responses can choose different supporting prices, so a
   common market outcome must be checked when defining an equilibrium.
5. Update the Overleaf formulation after agreeing on this baseline and the
   selection interpretation. No equilibrium spectrum has yet been enumerated.
