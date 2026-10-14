# Capacity-only model

One shared fixed-demand market supports three independent commands:

- `market`: clear an explicit fixed-capacity profile and verify its dual.
- `ranges`: find dispatch, SOC, price, and investor-profit ranges at that profile.
- `mpec`: optimize one investor's MW/MWh against the profile's fixed rivals
  (SOS1 complementarity, Gurobi, proven optimum).
- `mpec_relaxed`: the same MPEC with relaxed complementarity (Ipopt, local).
- `clearing`: share scarce hosting capacity by a uniform price (section
  "Scarce hosting capacity").
- `verify`: Gurobi check of a candidate equilibrium, each investor against
  fixed rivals (proven, refuted, or open).

Exploration order: Ipopt candidates, then Gurobi (section "Ipopt candidates,
Gurobi check").

Both MPECs take `--strategic-bids`: the investor then also chooses hourly
charge bids and discharge offers at each node (section "Strategic bids").

## Read the equations

The model follows the readable structure of
[urbs](https://github.com/tum-ens/urbs/blob/master/urbs/model.py): named sets,
parameters, variables, and constraint declarations followed by explicit rule
functions. The equations are written in the source, not assembled from a
dictionary of matrix coefficients. This is a structural refactor of our model;
it does not adopt urbs' economic or physical formulation.

Read the files in this order:

1. `prepare_input.py`: JSON loading, validation, indexed input parameters and sets.
2. `primal_llp.py`: dispatch variables, physical constraints, and market objective.
3. `dual_llp.py`: named multipliers, each stationarity equation, and dual objective.
4. `mpec.py`: the MPEC in named blocks, in this order: investor constraints,
   LLP primal feasibility, LLP stationarity, SOS1 complementarity, physical
   variable bounds, and the investor objective.
5. `mpec_relaxed.py`: the same blocks, with relaxed products instead of SOS1.

For example, the shared-inverter equation is literally:

```python
def storage_power_limit_rule(m, i, n, t):
    return m.charge[i, n, t] + m.discharge[i, n, t] <= m.power_capacity[i, n]
```

`build_primal()` uses fixed capacities. `build_mpec()` reuses the same physical
rule functions with active-investor capacity expressions and fixed rival
capacities. Its `slack_power_rule()` and `sos_power_rule()` explicitly enforce
complementarity for that inverter constraint. Nonnegativity is declared on each
physical variable; its dual multipliers and complementarity pairs are explicit
in `dual_llp.py` and `mpec.py`.

## Complementarity: SOS1

Each market inequality and its multiplier form one pair, and each pair
`{slack, mu}` is a special ordered set of type 1: at most one member is
nonzero, and Gurobi branches on the set (slack = 0 or mu = 0). Lower-bound
pairs use the dispatch variable itself (`{generation, mu_gen_lower}`);
upper-bound and line pairs get an explicit slack variable with a defining
equation (`slack_*_rule`), since SOS members must be variables. Equalities
have free multipliers and no pair. On IEEE-9 with I1 at four nodes there are
1 640 sets. There are no binaries and no bound on the multipliers, so the
result does not depend on a Big-M. `solve()` sets `PreSOS1BigM=0`; otherwise
Gurobi presolve may rewrite the sets as Big-M binaries.

Without multiplier bounds the prices are unbounded too, so the MPEC maximizes
`profit_linear`, the strong-duality form of the investor's profit, which has no
price * quantity products (MIQP with the quadratic costs, MILP without). The
direct nodal settlement
`price * (discharge - charge) - degradation - dispatch cost + owned generation margin - capex`
is kept as `profit` and reported; at every KKT point the two are equal, and
their difference is a weighted sum of slack*multiplier products of all other
assets and lines.

On 2026-10-12 SOS1 replaced Big-M binaries, bilinear products `slack * mu == 0`,
relaxed products, a single strong-duality constraint, and the Ipopt relaxed
MPEC (all removed; see git history and
`workflow/complementarity_and_degradation_2026-10-12.md`). On I1/N9 every
method has the same root relaxation; SOS1 proved the optimum fastest (7 s)
without a multiplier bound. The relaxation stays weak: on four candidate nodes
the bound does not leave its root value. Pyomo reports the incumbent as both
bounds for models without integer variables, so the runner reads both from
Gurobi (`objective_bounds()` in `solve.py`). The Ipopt relaxed MPEC returned
on 2026-10-13 as `mpec_relaxed.py`, as a fast local method next to SOS1.

## Run

Run from the repository root, using Python with `pyomo` and `gurobipy`
(`python -m pip install -r requirements.txt` if needed). Gurobi solves every
model: market LPs/QPs, ranges, and MPECs. The IEEE-9 MPEC needs a full Gurobi
licence; the pip licence is too small. Ipopt (on `PATH`) solves the
unique-price reclear when `--balancing-eps` > 0.

`model/input/` holds only the 5-bus test system (next section), the default
`--data` / `--profile`. The earlier inputs (IEEE-9 case and its cost,
degradation, and slope profiles, the toy case, the 5-bus variants) and
`scripts/data/build_ieee5.py` were removed on 2026-10-14. Every command below
that names one of them reproduces an earlier result after restoring them:

```powershell
git checkout 82690b2 -- model/input scripts/data/build_ieee5.py
```

IEEE-9 case:

```powershell
python model/primal_llp.py --data model/input/market_data.json --profile model/input/capacities.json --output model/output/no_storage
python model/run.py ranges --data model/input/market_data.json --profile model/input/capacities.json --dispatch all --output model/output/no_storage_ranges

python model/primal_llp.py --data model/input/market_data.json --profile model/input/capacities_storage.json --output model/output/fixed_storage
python model/run.py ranges --data model/input/market_data.json --profile model/input/capacities_storage.json --output model/output/fixed_storage_ranges

python model/mpec.py --data model/input/market_data.json --investor I1 --nodes N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
python model/run.py mpec --data model/input/market_data.json --investor I1 --nodes N3 N6 N8 N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4 --mip-gap 0.01 --seconds 600 --output model/output/<new_dir>
```

`capacities.json` still carries the inherited, unsourced costs (6 600 EUR/MW,
18 800 EUR/MWh). The sourced cost cases are `capacities_capex120.json` and
`capacities_capex155.json`: `cost_energy_eur_per_mwh` 120 000 or 155 000
(BESS system incl. power conversion; IRENA 2026, ECB 2025 USD/EUR rate) and
`cost_power_eur_per_mw` 1 (placeholder grid-access fee). Derivation and
daily values are in `Overleaf/model_extension.tex`, section "Investment-cost data".
The current working case is `capacities_capex120_curtail_d3_slope1e-4.json`
(curtailment cost, d = 3 EUR/MWh, dispatch slope 1e-4; see below and
`project_context.md`); `_d0`, `_d10`, `_d15` vary d, `_d3` and `_d3_slope1e-3`
vary the slope.

The IEEE-9 profile `capacities.json` starts all four investors at zero capacity. The
storage example installs 10 MW / 30 MWh each for I1 and I2 at N6. Only the
active investor's capacities are replaced in `mpec`; all others remain fixed.
Use an exported `capacities.json` as `--profile` to independently clear or
analyze a chosen investment.

For a small, fully solved analytical example:

```powershell
python model/run.py mpec --data model/input/toy_market.json --profile model/input/toy_capacities.json --investor I --node-limit 10 --output model/output/toy_mpec
python model/run.py ranges --data model/input/toy_market.json --profile model/output/toy_mpec/capacities.json --dispatch all --face-tolerance 0 --output model/output/toy_ranges
```

This toy has one node, two one-hour periods, lossless storage, and one-hour
duration. Cheap energy costs 10 EUR/MWh and is available only in hour one;
peaker energy costs 40 EUR/MWh. Its optimistic optimum is 5 MW / 5 MWh and
140 EUR profit. At exactly that capacity, alternative supporting prices allow
profit from -10 to 140 EUR. This is a deliberate illustration of price
selection, not a calibrated investment example.

`--nodes N6` restricts the active investor to N6 and forces its investment
elsewhere to zero. It changes the strategy set. `--node-limit` bounds total
installed MW per node. `--seconds` limits each main market/MPEC solve. Range
subproblems use a 60-second limit individually. `--mip-gap` controls the MPEC
proof tolerance. `run_config.json` records the Gurobi version.

## 5-bus test system (final market data)

Since 2026-10-14 all further work uses the 5-bus test system,
`five_bus_market_data.json` / `five_bus_capacities.json`, the default
`--data` / `--profile` of `run.py`. Built by `scripts/data/build_five_bus.py`;
every number is in the JSON. It is V6 (below) with one thermal unit whose
marginal cost rises with output, in place of G1 600 MW at 52 and G3 at 67.

- Network: two radial pockets. L12 140 MW (wind, N2), L45 120 MW (PV, N5),
  L13/L14/L34 300 MW (never binding); all reactances from the IEEE 5-bus table.
- Generation: G1 600 MW (N1) with marginal cost 41 + 0.075 * g EUR/MWh
  (section "Rising generation cost"), wind 200 MW peak (N2), PV 300 MW peak (N5).
- Load: peaks N1 120, N3 180, N4 150, N5 40 MW, none at N2; IEEE-9 hourly shapes.
- Flexible demand (since 2026-10-14, section "Flexible demand in the pockets"):
  N2 10 MW and N5 25 MW, willingness to pay 60 EUR/MWh at 0 MW falling to 0.
- Investors: I1 merchant (WACC 8 %), I3 80 % wind / 20 % PV, I4 the rest;
  120 000 EUR/MWh, 1 EUR/MW, d = 3 EUR/MWh, dispatch slope 1e-4, curtailment
  slope 0.1 on both plants. The second merchant I2 (as I1 at 12 % WACC) was
  dropped on 2026-10-14: it never entered and only cost a best response per
  round. Results before that date include it at zero capacity.
- Without storage (eps 1e-4): N1/N3/N4 follow residual load, from 52.16 EUR/MWh
  (hour 16) to 66.65 (hour 20). N2 down to -5.0 at night (370 MWh wind
  curtailed), N5 down to -12.2 at midday (482 MWh PV curtailed); the flexible
  demand takes 315 MWh. Market cost less the flexible demand's benefit
  250 315 EUR/day. Pricing stays nodal.
- Without the flexible demand (`--flexible-demand-scale 0`): hub 52.06-66.65,
  N2 down to -6.0 (505 MWh curtailed), N5 down to -14.7 (632 MWh), market cost
  260 857 EUR/day. Each pocket then has a price step at one storage size: its
  line is congested or not, and nothing in the pocket pays between the
  curtailment price and the storage value. The hub has no step.
- Results labelled V6 and V7 below were computed on those variants; the
  best-response table directly below was computed without the flexible demand.

Best responses against zero rivals without the flexible demand (reproduce with
`--flexible-demand-scale 0`), eps 1e-4, all buses (2026-10-14, Ipopt
3.13.2, Gurobi 13.0.3; `model/output/five_bus_best_responses/`, untracked;
capacities in `workflow/flexible_demand_2026-10-14/best_responses_zero_rivals.json`).
Relaxed: exact recleared profit in EUR/day from starts of 20, 50, and 150 MW,
all with `verification_passed`. Gurobi: MIP start from the best relaxed run,
180 s, `--mip-gap 0.01`; MPEC objective.

| investor | relaxed (Ipopt), best | N2 / N5 MW / MWh | other local optimum | Gurobi from Ipopt start |
|---|---|---|---|---|
| I1 | 10 209.84 (150 MW start) | 52.5 / 380.9, 100.8 / 489.0 | 10 205.05, N5 97.1 / 475.4 (20, 50) | 10 209.95, within 0.97 % in 11 s |
| I2 | 3 869.62 (all starts) | 35.0 / 125.0, 68.1 / 327.6 | - | 3 869.65, within 0.42 % in 3 s |
| I3 | 53 978.50 (50, 150) | 70.0 / 413.6, 134.9 / 577.7 | 53 658.94, N2 35.0 / 280.0 (20) | better point 54 037.64, bound 115 134 |
| I4 | 71 647.32 (20) | 35.0 / 280.0, 146.8 / 591.1 | 71 527.67, N2 65.0 / 391.9 (50, 150) | 71 647.56, bound 75 634 (5.6 %) |

- Against V7 the merchants got easy: I1 within 1 % in 11 s (V7: 7.3 % open
  after 180 s) and I2 needs no particular start. I1's N2 battery is interior
  (52.5 MW) instead of at the pocket edge (V7: 70 MW / 456.7 MWh).
- The owners still have several local optima, at the pocket surpluses: N2
  35 MW is the hour-22 wind surplus over L12, N5 146.8 MW the hour-14 PV
  surplus over L45. Gurobi found a better I3 point (+59 EUR/day) that the
  time-limited run does not export; I3's bound stays loose.

```powershell
python scripts/data/build_five_bus.py .
python model/run.py market --balancing-eps 1e-4 --output model/output/<new_dir>
python scripts/diagnostics/plot_prices.py <dir>/prices.png <dir> "5-bus test system, no storage"
```

## Earlier 5-bus variants

V7 (G1 300 MW at 52 EUR/MWh, G3 300 MW at 67 at N3, otherwise as above) was
the test system on 2026-10-14 until the rising-cost G1 replaced it. Its JSON is
`build_five_bus.py` with those two generators and no slope (copies in the
untracked `model/output/v7_step_best_responses/`). Without storage its hub
price is 52, and 67 in hours 19-21; market cost 274 313 EUR/day.

V7 best responses against zero rivals, eps 1e-4, all buses (2026-10-14,
Ipopt 3.13.2, Gurobi 13.0.3). Relaxed: exact recleared profit in EUR/day, best
of the 50 MW, 150 MW, and step starts (I2 also 20, 30, 42, 100 MW). Gurobi:
MPEC objective (strong-duality profit); it differs from the exact profit by
up to 2.5 EUR/day.

| investor | relaxed (Ipopt) | N2 / N5 MW / MWh | Gurobi cold, 600 s | Gurobi from Ipopt start, 180 s, gap 0.01 |
|---|---|---|---|---|
| I1 | 8040.86 (all starts) | 70.0 / 456.7, 98.0 / 478.8 | 8042.43, bound 8547 | 8042.43, bound 8629 (N2 N5 only: 8844) |
| I2 | 2238.15 (starts 20-50 MW) | 16.3 / 92.7, 42.0 / 94.1 | 2238.91, proven in 17 s (same point) | - |
| I3 | 53 473.49 (all starts) | 70.0 / 456.7, 133.7 / 575.5 | no incumbent, bound 129 849 | N2 N5 only: 53 478.6, bound 129 006 |
| I4 | 66 192.47 (all starts) | 70.0 / 456.7, 146.8 / 591.1 | 29 749, bound 73 985 | N2 N5 only: 66 195, bound 72 401 |

- Ipopt finds every best point Gurobi knows, in seconds. Gurobi improves none
  of them, cold or from the Ipopt start, and proves only I2.
- I2 needs several starts: 100-150 MW starts end at 1265.09 and the step start
  at 1855.27. Its best sits on the edge of the G3 block (evening price 66.99),
  held there by eps. This second edge besides the pocket steps is why V7 was
  replaced.
- The MIP start gives Gurobi the incumbent at once (I3 had none cold after
  600 s) but does not move the bound. Fewer nodes did not tighten it in 180 s.
- Every relaxed run has `verification_passed` false, narrowly: constraint
  violation 1.5-2.9e-6 (limit 1e-6) and |market cost - dual value| 2.4-3.0e-3
  EUR (limit 1282 pairs * 1e-6). The cause was Ipopt's default bound
  relaxation (`bound_relax_factor` 1e-8), also in the exact reclear (market
  cost 2.6e-3 EUR below Gurobi's). Every Ipopt solve now uses 0 (`solve.py`);
  V7 was not rerun.

Older variants: inputs and build script at commit 82690b2 (restore as in "Run").
`ieee5_market_data.json` / `ieee5_capacities.json` (built by
`scripts/data/build_ieee5.py --variant current`; every number is in the JSON): lines 1-2, 1-4,
2-3, 2-4, 3-5, 4-5; G1 thermal 250 MW at 52 EUR/MWh (N1), wind 250 MW (N2),
G3 thermal 250 MW at 67 EUR/MWh (N3), PV 350 MW (N5); peak loads N1 96,
N3 144, N4 120, N5 40 MW with the IEEE-9 hourly shapes. Reactances of 1-2, 2-3,
2-4, 4-5 are the IEEE 5-bus per-unit values; 1-4 and 3-5 are assumed. Loads,
generators, costs, and limits are design choices for two pockets: the PV
corridor (L35 120 MW, L45 100 MW) binds at midday (N5 down to -12 EUR/MWh,
406 MWh PV curtailed), the wind corridor (L12 80 MW, L23 60 MW) at night
(N2 down to -5, 340 MWh wind curtailed). No price exceeds 67, no load is shed.
Investors as in the IEEE-9 working case (I3 80 % wind / 20 % PV, I4 the rest,
curtailment slope 0.1 on both plants).

Best responses against zero rivals, eps 1e-4, all five buses (2026-10-12):
I2 is proven optimal in 53 s with 28.62 MW / 57.24 MWh at N2 and
93.53 MW / 325.54 MWh at N5 (5553.16 EUR/day). I1 finds no feasible point in
120 s; with the warm-start prototype `scripts/diagnostics/active_set_start.py`
its incumbent after 120 s is 9352.15 (N2 22.9 / 74.0, N5 118.8 / 380.9),
bound 10507. I3/I4 (owned generation) need that warm start too and are not
proven (single-pocket design: gaps 5 % and 3 % after 120 s).

```powershell
python scripts/data/build_ieee5.py . --variant current
python model/run.py market --data model/input/ieee5_market_data.json --profile model/input/ieee5_capacities.json --output model/output/<new_dir>
python model/run.py mpec --data model/input/ieee5_market_data.json --profile model/input/ieee5_capacities.json --investor I1 --nodes N1 N2 N3 N4 N5 --balancing-eps 1e-4 --seconds 600 --output model/output/<new_dir>
```

## Unique prices: nodal balancing (`--balancing-eps`)

`--balancing-eps eps` (MW per EUR/MWh, default 0) adds a free balancing
injection `b[n,t]` to every nodal balance with cost `b^2/(2 eps)`:

```
min  market_cost + sum b^2/(2 eps)    s.t.  ... + b[n,t] - injection = demand   (price)
stationarity:  b[n,t] = eps * price[n,t]
dual:          ... - (eps/2) * sum price^2
```

The dual is strictly concave in price, so every nodal price is unique and the
investor's profit is unique at every capacity: optimistic and pessimistic
selection coincide, and profit is continuous in capacity. Supply steps become
ramps about `eps * price jump * nodes` MW wide. As eps -> 0 the prices tend to
the minimum-norm LP prices. `eps = 0` omits `b` and reproduces the LP exactly.
The MPEC adds only the linear stationarity equation; `profit_linear` gains
`-(eps/2) sum price^2` (concave). `b` is exported as `balancing|n|t`.

Numerics: small eps pins prices only weakly. Moving a price by d changes the
dual objective by only `(eps/2) d^2`, so the dual QP with ordinary tolerances
can be far off (900 EUR/MWh on IEEE-9 at eps = 1e-4). MPEC reclears with
eps > 0 therefore take `price = b/eps` from the primal QP solved by Ipopt with
tolerance 1e-12 (`unique_price_reclear`) and report the investor's settlement
at those prices as a one-point profit range. Gurobi's size-limited licence
cannot solve QPs with more than about 200 variables. `ranges` with eps > 0
reflects solver tolerance, not price multiplicity.

```powershell
python model/run.py mpec --data model/input/toy_market.json --profile model/input/toy_capacities.json --investor I --node-limit 10 --balancing-eps 1e-4 --output model/output/toy_mpec_balancing
```

## Curtailment cost of owned generation

An investor profile may give each owned plant a curtailment cost slope
`curtailment_slope_eur_per_mwh_per_mw` (default none = 0). Curtailing
`u = capacity - generation` MW costs the owner `share * 0.5 * slope * u^2`
(for example support paid per MWh produced, with the cheapest plants of a
portfolio curtailed first). The plant's slope is the share-weighted sum over
its owners, so it offers its last MW at `generation_cost - slope * u`:

```
primal:        market_cost + sum 0.5 * slope * curtailment^2
stationarity:  generation_cost - slope * curtailment - price + mu_upper - mu_lower = 0
dual (Dorn):   ... + sum 0.5 * slope * (capacity^2 - generation^2)
```

While a plant is curtailed its price is `-slope * u`: negative and continuous
in storage capacity. Once curtailment is zero the cost no longer acts, so a
step from a curtailed price of 0 to a storage break-even price stays a step.
The settlement margin subtracts the owner's curtailment cost; `profit_linear`
stays concave (MIQP). The standalone dual keeps generation as a free Dorn
variable. `capacities_capex120_curtail.json` is `capacities_capex120.json`
with slope 0.1 EUR/MWh per MW on both PV plants for I3 and I4.

## Rising generation cost

Market data key `generation_cost_slope` (EUR/MWh per MW, per plant, default 0):
a plant costs cost * g + 0.5 * slope * g^2, so its marginal cost is cost +
slope * g. One such unit can stand for a thermal merit order (the 5-bus G1):
the price then rises continuously with residual load instead of jumping from
plant to plant, and the market becomes a QP. Generator stationarity gains
slope * generation; the dual keeps generation as a Dorn variable and
subtracts `generation_dispatch_rent` = 0.5 * slope * g^2, the plant's rent from
its rising cost. The strong-duality profit subtracts it as well and stays
concave. An owned plant must have slope 0 (the owned margin is linear in its
cost); `prepare_input` rejects an owned plant with a slope. Checked by
`test_rising_generation_cost`.

`ranges` handles this poorly: the optimal face becomes a quadratic constraint,
and Gurobi's barrier stops with SUBOPTIMAL (status 13) on most coordinates,
which `ranges` reports as unresolved. On the 5-bus system without storage at
eps 1e-4, 121 of 124 ranges are unresolved (V7: 22); at eps 0, 46 of 48
hub-price solves fail. While G1 is between its bounds the hub price 41 + 0.075 g
is unique anyway. Not fixed: `BarQCPConvTol` 1e-4 resolves 47 of 48 at eps 1e-4
but loosens the endpoints. A linear face would avoid the quadratic constraint:
each separable quadratic variable has one value on the optimal face
(Mangasarian 1988), so it can be fixed there.

## Flexible demand in the pockets

Market data key `flexible_demand` ({node: {max_mw, max_price_eur_per_mwh}},
default none): a price-responsive load q, e.g. power-to-heat, whose
willingness to pay falls linearly from max_price at 0 MW to 0 at max_mw. It is
nobody's asset and bids truthfully. The ISO maximizes welfare: `market_cost`
subtracts its benefit max_price * q - 0.5 * slope * q^2 (slope = max_price /
max_mw), and the nodal balance withdraws q. Stationarity gives price =
max_price - slope * q inside the bounds. The dual keeps q as a Dorn variable
and subtracts `flexible_demand_surplus` = max_mw * mu_upper + 0.5 * slope * q^2,
the load's consumer surplus at a KKT point; the strong-duality profit does the
same and stays concave. Two SOS1 pairs per node and hour (relaxed products in
`mpec_relaxed`). `run.py --flexible-demand-scale s` multiplies every max_mw by s
at the same max_price; 0 removes the load. Checked by `test_flexible_demand`:
storage that displaces the load raises the price along its curve (15 + 3X),
and both MPECs find the interior optimum X = 23/6.

Why: in a congested pocket nothing had a willingness to pay between the
curtailment price (about 0) and the storage value (about 47 EUR/MWh at N5), so
the pocket price, and every owner's profit, jumped at one storage size; only
the balancing eps set the width (workflow/summary_2026-10-13_15-01.md). With
the load, storage first displaces it and the price rises along its curve.
max_price must exceed the storage value, or a vertical piece remains. The load
is the demand-side counterpart of the rising G1 cost at the hub.

Size (2026-10-14, `workflow/flexible_demand_2026-10-14.md`): one battery grown
in each pocket, exact reclears at 0, 0.1, 0.25, 0.5, 1 and 2 times N2 20 /
N5 50 MW. The pocket price plateaus at the battery's break-even (47.7 EUR/MWh
at N5, 41-42 at N2) at every size; the size sets how early and how gradually
it gets there, and how much cheap surplus the load takes from storage.

| size (N2 / N5 MW) | N5 price ramp | N2 price ramp | merchant best, N5 / N2 (EUR/day) |
|---|---|---|---|
| none | 146.8 -> 147.0 MW (step) | none; day price collapses to 3.1 beyond 77 MW | 7 414 / 2 715 |
| 5 / 12.5 | 130 -> 145 | 72 -> 75 | 6 745 / 2 553 |
| **10 / 25 (chosen)** | **115 -> 140** | **60 -> 72** | **6 065 / 2 410** |
| 20 / 50 | 80 -> 130 | 32 -> 64 | 4 167 / 2 025 |
| 40 / 100 | price > 5 already at 0 MW | same | 1 384 / 490 |

Ramp: battery MW where the mean surplus-hour price passes 5 EUR/MWh and 90 % of
its plateau (N5 4 h, N2 6 h duration, no other storage). Without the load, N2
has a second pathology: once a battery absorbs the whole night surplus, the
line is congested in the day too, and the day price falls to the battery's
break-even 3.1, selected by eps alone. Hub prices without storage move by at
most 0.3 EUR/MWh across all sizes. 10 / 25 MW keeps ramps of 16-35 MW while
merchants keep 82-89 % of their arbitrage; 20 / 50 MW is the sensitivity
(`--flexible-demand-scale 2`).

## Storage dispatch cost

Next to the linear degradation `0.5 * d * (charge + discharge)`, an investor
profile may set `dispatch_cost_slope_eur_per_mwh_per_mw` (default 0):
charging or discharging `p` MW for one hour costs `0.5 * slope * p^2`
(losses rising with current). Its purpose is selection: the market objective
becomes strictly convex in charge and discharge, so the storage dispatch is
unique, as `--balancing-eps` makes prices unique. The slope is stylized.

```
primal:        market_cost + sum 0.5 * slope * (charge^2 + discharge^2)
stationarity:  0.5 * d + slope * charge + price - eta * soc_value + mu_power - mu_lower = 0
dual (Dorn):   ... - sum 0.5 * slope * (charge^2 + discharge^2)
```

The Dorn term is also each storage's rent from its rising marginal cost:
settlement less degradation and dispatch cost equals scarcity rent plus this
rent. `profit_linear` therefore subtracts the rivals' dispatch rents and stays
concave. The chosen slope is 1e-4 EUR/MWh per MW: about 4 EUR/day for a
173 MW battery, against about 40 EUR/day at 1e-3, which mostly penalizes
full-power charging where no tie exists. The slope does not unstick the
four-node MPEC bound. Evidence: `workflow/complementarity_and_degradation_2026-10-12.md`.

```powershell
python scripts/diagnostics/storage_dispatch_ranges.py . --profile <capacities.json> --degradation 3 --dispatch-slope 1e-4
```

## Relaxed MPEC (`mpec_relaxed`, Ipopt)

`mpec_relaxed.py` builds the MPEC of `mpec.py` from the same blocks but
replaces each SOS1 set by the Scholtes relaxation `slack * mu <= epsilon`. Ipopt
solves it for decreasing epsilon (`--epsilons`, default 1 to 1e-6), each step
warm-started from the previous one. The start is a fixed-capacity market KKT
point: `--start-power` MW at minimum duration at every investment node
(default 50), or the investor's capacity and bids in `--start-profile` (for
example a step-search result). Ipopt has no presolve, so shedding at zero demand
and plants at zero capacity are written as equalities with one free multiplier.

The result is a local optimum from one start, not a proven best response. As
epsilon goes to 0 the limit is in general only C-stationary (Scholtes 2001), and
at a supply step the relaxed objective can exceed the true profit. The runner
therefore reports `exact_profit_eur_per_day`, the investor's settlement in the
exactly recleared market at the chosen capacity and bids, next to the relaxed
objective. `verification_passed` requires every product to be at most the
final epsilon, constraint violations of at most 1e-6, and
|market cost - dual value| <= pairs * epsilon. The products are checked
signed: a negative product is a 1e-8 primal tolerance times a large multiplier.

V6 5-bus case (`ieee5_v6_*.json`), 2026-10-13: eps 1e-4, all buses, rivals at
zero; exact profit in EUR/day, Ipopt 3.14.20, Gurobi 13.0.3.

| investor | relaxed, 50 MW start | relaxed, step start | Gurobi SOS1 (12 Oct) |
|---|---|---|---|
| I1 | 8042.37 (33 s) | 8042.37 (22 s) | 8042.38, proven |
| I2 | 1264.95 (18 s) | 1264.70 (15 s) | 1264.95, proven |
| I3 | 53 478.54 (27 s) | 53 478.54 (20 s) | 53 478.55 incumbent, bound 108 213 after 600 s |
| I4 | 66 194.69 (33 s) | 66 194.71 (17 s) | 66 192.86 exact at the incumbent, bound 68 312 |

The relaxed MPEC reaches the proven merchant optima and the owners' best
incumbents within a minute. It does not prove them: the owners' Gurobi bounds
stay loose. Step starts are `scripts/diagnostics/step_candidates.py` outputs.

## Ipopt candidates, Gurobi check

Ipopt finds candidates in seconds; Gurobi improves, proves, or refutes them.

1. Run `mpec_relaxed` from several starts (`--start-power` 20, 50, 150, and a
   step-search `--start-profile`) and keep the best exact profit. One start is
   not enough (V7 I2: 2238 from 20-50 MW, 1265 from 100-150 MW).
2. Run `mpec --start-profile <relaxed run>/capacities.json`. The candidate
   becomes a Gurobi MIP start: `complete_sos1_start` sets each slack from its
   definition, fills the duals of storage absent from the start market, sets
   the smaller member of each SOS1 pair to zero, and polishes the point (dual
   simplex, FeasibilityTol 1e-9, zero members fixed, capacity free), because
   Gurobi rejects a start that violates a constraint by more than 1e-6.
   `gurobi.log` shows "Loaded user MIP start"; the run writes
   `start_profile.json` and `mip_start` in `summary.json`. The start gives an
   incumbent at once; it does not tighten the bound.
3. `verify` checks a candidate equilibrium. Each investor's MPEC gets Gurobi
   Cutoff = current profit + tolerance, so Gurobi searches only for gains;
   a cutoff status proves that no deviation gains more than the tolerance. A
   MIP start adds nothing there.
4. `--nodes` restricts the active investor in `mpec` and `verify`. Dropping
   nodes where Ipopt never invests saves time, but the result is then a proof
   against deviations at those nodes only; `verify` records `deviation_nodes`
   and refuses a profile with capacity outside them.
5. `--mip-gap 0.01` is enough while exploring; tighten it for final results.

```powershell
python model/run.py mpec_relaxed --investor I2 --balancing-eps 1e-4 --start-power 20 --seconds 300 --output model/output/<relaxed_dir>
python model/run.py mpec --investor I2 --balancing-eps 1e-4 --seconds 180 --mip-gap 0.01 --start-profile model/output/<relaxed_dir>/capacities.json --output model/output/<new_dir>
python model/run.py verify --profile <candidate profile> --balancing-eps 1e-4 --nodes N2 N5 --seconds 600 --output model/output/<new_dir>
```

## Strategic bids (`--strategic-bids`)

Every storage enters the market with an hourly charge bid (price to buy) and
discharge offer (price to sell) per node, and the ISO's storage cost is
`offer * discharge - bid * charge`. Truthful bids are `-0.5 d` and `0.5 d`,
which is the degradation cost, so every earlier result is unchanged. A profile
can fix bids as `charge_bid_eur_per_mwh` and `discharge_offer_eur_per_mwh`,
`{node: {"hour": EUR/MWh}}`; missing entries are truthful. The MPEC exports the
active investor's chosen bids in that format in `capacities.json`, so the
reclear and later rival profiles use them.

```
stationarity:  -charge_bid + slope * charge + price - eta * soc_value + mu_power - mu_lower = 0
               discharge_offer + slope * discharge - price + soc_value / eta + mu_power - mu_lower = 0
investor:      -VOLL <= bid, offer <= VOLL;   eta^2 * offer >= bid   (no negative-cost same-hour loop)
profit_linear: market cost with the active storage at its true degradation cost instead of its bids
```

The bids enter only the stationarity equations, linearly. `profit_linear`
counts the active investor's storage at its true cost, so the bid margin is
part of the profit and no bid * quantity product appears (Ruiz and Conejo
2009). The MPEC therefore keeps its structure, and both `mpec` and
`mpec_relaxed` accept the flag. `--strategic-bids` needs `--balancing-eps > 0`.
With LP prices the bid margin depends on a non-unique dispatch, so
`dual_profit` refuses non-truthful bids.

Bids are not unique. Any bid beyond the price range supports the same dispatch:
an offer of 6 729 EUR/MWh means "never discharge here". `mpec` therefore solves
twice (`solve_mpec` in `mpec.py`). After the optimum, `select_bids` keeps the
capacity, holds the investor's own charge/discharge and every nodal price
within 1e-6 of the optimum, and minimizes the squared distance of its bids from
truthful over the same KKT conditions and SOS1 pairs (linear constraints, a
convex objective). If Gurobi calls that hold infeasible (the optimum meets it
only to the first solve's tolerances; V6 I4 behind I3), it retries at 1e-5 and
1e-4 (`bid_selection.attempts`). The exported bids, the reclear, and later rival profiles
use the selected bids; `summary.json` reports them under `bid_selection`.
Because the ISO's SOC value is free, the rule spreads a markup over the hours
that set it (four-hour test: +15 on the offer, -15 on the charge bid, instead
of +30 on one). The rule is an assumption about rivals: minimal markups make a
held-back battery react as soon as a price moves (extreme bids would freeze
it). `mpec_relaxed` exports its bids unselected.

V6 selection, 2026-10-13 13:12 (each selection proven at the root in 0.02 s):

| investor | squared markup, first solve -> selected | largest markup [EUR/MWh] | exact - linear profit [EUR] |
|---|---|---|---|
| I1 | 1.5e10 -> 10.6 | 1.8 | -0.013 |
| I2 | 1.4e10 -> 15.1 | 2.2 | 0.000 |
| I3 | 1.5e10 -> 3 030 | 48.4 | -0.003 |
| I4 | 1.9e10 -> 13 500 | 28.9 | -0.041 |

Another Gurobi seed gives the same bids (I3 within 3e-3 EUR/MWh, after a
1e-3 MW capacity difference; I4 within 2e-9).

V6, 2026-10-13, settings as above (exact profit, EUR/day):

| investor | capacity only | with bids, relaxed (50 MW / step start) | with bids, Gurobi SOS1 |
|---|---|---|---|
| I1 | 8042.38 | 8094.93 / 8094.93 | 8094.93, proven in 1 s |
| I2 | 1264.95 | 1294.66 / 1294.66 | 1294.66, proven in 1 s |
| I3 | 53 478.54 | 124 420.44 / 124 554.76 | 124 554.73, proven in 3 s |
| I4 | 66 194.71 | 75 431.30 / 75 431.30 | 75 431.43, proven in 1 s |

- With bids Gurobi proves every investor within seconds. For the merchants the
  optimum equals the capacity MPEC's root relaxation (8094.93 and 1294.66 in
  `workflow/summary_2026-10-12_20-53.md`). The bids free the investor's own
  storage conditions, the ones that relaxation drops.
- I3 (80 % of the N2 wind) more than doubles its profit. Its N2 battery charges
  about 1 MW more per night hour, just enough to absorb the curtailed wind, and
  its charge bid then sets the night price in the wind pocket: 52 instead of 0.
  The owned wind earns that price; the storage's own arbitrage loses a little.
- Before the selection, I2 and I4 failed the then absolute 1e-4 EUR
  thresholds (5e-4 EUR errors on an as-bid market cost of about -4e5 EUR).
  The SOS1 check now allows 1e-4 EUR + 1e-8 of |market cost| + |profit|
  (`verification_tolerance_eur`, about 3e-3 EUR on V6); all four pass.

```powershell
python model/run.py mpec_relaxed --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I1 --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
python model/run.py mpec_relaxed --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I3 --balancing-eps 1e-4 --seconds 300 --start-profile model/output/step_starts/ieee5_v6_I3.json --output model/output/<new_dir>
python model/run.py mpec_relaxed --strategic-bids --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I3 --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
python model/run.py mpec --strategic-bids --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --investor I3 --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
```

## Scarce hosting capacity (`clearing`)

A node can host a limited storage power in total, `sum_i X_power[i, n] <= H_n`.
By default (`--node-limit`) this is first come, first served: rivals' MW reduce
the active investor's headroom, so the move order decides who gets the node.
`clearing` instead shares the node by one uniform price p_n [EUR/MW/day]:

```
investor i:  max profit_linear_i - sum_n p_n * X_power[i, n]    s.t. X_power[i, n] <= H_n, LLP KKT (unchanged)
clearing:    0 <= p_n  complementary to  H_n - sum_i X_power[i, n] >= 0
```

With one price for all investors this is the variational equilibrium of the
shared constraint (Rosen 1965, Harker 1991): a competitive, uniform-price
auction of hosting capacity in which investors are price takers. The energy
market (LLP) is unchanged; the price is a linear fee in each MPEC
(`capacity_price` in `prepare_input.py`, `investment_cost_rule` in `mpec.py`).

Two other sharing rules use the same Gauss-Seidel loop (since 2026-10-14):

- `clearing --first-come --hosting N=H`: first come, first served. Rivals' MW
  reduce each investor's headroom, so whoever holds capacity keeps it; the
  outcome can depend on `--order`. No price.
- `clearing --allocation I3:N=50 I4:N=50` (also `verify`): hosting rights,
  e.g. lots of an auction. At every node named there each investor builds at
  most its own right, 0 without one (`own_limit` in `prepare_input.py`,
  `own_limits` in `capacity_clearing.py`); `I3:N=0` closes N. Running every
  assignment of the lots gives each investor's profit when it wins and when a
  rival wins, so willingness to pay and auction outcomes follow without a
  price search. Checked by `test_hosting_rights`.

Removed on 2026-10-14 (code and results at commit b0b74b8,
`workflow/scarce_capacity_2026-10-14/`): the quadratic access cost
`--access-slope` (no slope reached the limit; the same jump in owners'
demand as the uniform price) and the soft-linked auction
`auction_softlink.py` (Jacobi rounds cycled; bid curves at fixed rivals
include the owners' price uplift, which they also get when a rival fills the
node).

`capacity_clearing.py`: at fixed prices, Gauss-Seidel best responses until a
full round has no move worth more than `--gain-tolerance` in exact profit;
bisection on the price of the one node that is over-full at zero prices. If the
node's demand jumps across H_n (lumpy investment), the bracket narrows while MW
stay unused at the upper price: then no uniform price clears the node.

- Capacity only (default, truthful bids): relaxed MPEC from the investor's
  current capacity and from 50 MW, plus building nothing. Local: "converged"
  means no improvement found. One start was not enough on V6 (moves worth
  48 000 EUR/day missed).
- `--strategic-bids`: bid MPEC (`solve_mpec`, proven). On V6 two batteries in
  one pocket cycle in their bids: at the step the pocket price is anywhere
  between 0 and the storage value in the LP, and bid differences of 4e-4
  EUR/MWh decide it through the balancing slope.

V6 capacity only, no hosting limit (2026-10-13): two candidates pass the
two-start check, both with merchants out and the steps filled (N2 456.7 MWh,
N5 146.8 MW): A, I3 N2 + N5 130.5 MW with I4 16.3 MW (profit 40 898 / 84 249),
and B, I3 N2 only with I4 N5 146.8 MW (62 457 / 62 812). The N5 battery lifts
the price for both PV owners, so each prefers the other to build it; the move
order selects. With H_N5 = 100 MW (below the step) I3 and I4 cycle at the step.

`verify` (600 s) proves merchant exit at both but no owner's side. A is not an
equilibrium: I3 gains 12 333 EUR/day by shrinking N5 to 129.1 MW (N5 total
below the step, midday N5 price to about 0). B is open; small moves
(`scripts/diagnostics/equilibrium_probe.py`) find no gain, but at B a 0.01 MW
battery at N2 drops the midday N2 price from 52 to 14.6 EUR/MWh (to 3.1 at
1 MW), a difference of 0.0037 MW balancing at eps = 1e-4. At eps = 1e-4 the
balancing term, not the economics, sets these payoffs; smoothing comes first.

```powershell
python model/run.py clearing --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --balancing-eps 1e-4 --order I3 I4 I1 I2 --seconds 120 --output model/output/<new_dir>
python model/run.py clearing --data model/input/ieee5_v6_market_data.json --profile model/input/ieee5_v6_capacities.json --balancing-eps 1e-4 --hosting N5=100 --order I3 I4 I1 I2 --seconds 120 --output model/output/<new_dir>
python model/run.py verify --data model/input/ieee5_v6_market_data.json --profile workflow/equilibria_2026-10-13/ieee5_v6_equilibrium_B.json --balancing-eps 1e-4 --seconds 600 --threads 6 --output model/output/<new_dir>
```

## Outputs and limits

Every command writes `run_config.json`, `input_profile.json`, and `summary.json`.
Use an empty output directory for each run; existing results are not overwritten.
Market runs also export dispatch and nodal prices. Range runs export `ranges.csv`;
default quantity ranges cover storage, `--dispatch all` includes every market
variable, and `--dispatch none` covers prices and profits only.

MPEC runs with proven optimal status export the chosen `capacities.json`,
embedded dispatch/prices, independently recleared dispatch/prices, settlement
and strong-duality profit with their identity error, optimality residuals, the
solver and its version, and the fixed-capacity profit interval.
Time-limited runs report status and objective bounds and return exit code 2.
Since 2026-10-14 they also export Gurobi's incumbent (capacities, embedded and
recleared market, exact profit) when there is one, labelled in
`interpretation` as an incumbent, not a proven best response: its exact profit
bounds the best response from below, and its capacities can serve as a start
or a deviation. With strategic bids the incumbent's bids are not selected.

The dual face has no artificial price or multiplier bounds, and neither has the
MPEC. It is **optimistic**: it may choose the market optimum and supporting
prices most favorable to the investor. Degenerate pairs (both shedding bounds
at zero demand, both generator bounds at zero capacity) leave their
multipliers undetermined; `largest_dual_multiplier` shows how large they got.
An arbitrary reclear need not reproduce the selected profit without
`--balancing-eps`. Neither a single best response nor coordinate ranges
enumerate market equilibria.

See `../project_context.md` for all physical/economic assumptions. For the LP
duality and complementary-slackness basis, see
[Vandenberghe's duality notes](https://seas.ucla.edu/~vandenbe/ee236a/lectures/duality.pdf).

## Code and verification

- `prepare_input.py`: input loading/validation, sets, parameters and annualized costs.
- `primal_llp.py`: named physical variables, explicit market equations and objective.
- `dual_llp.py`: explicit stationarity, dual objective and fixed-capacity profit.
- `mpec.py`: capacity investment and optional strategic bids, SOS1 complementarity, physical bounds, settlement/strong-duality profit, and the exact MIP start from a market KKT point.
- `mpec_relaxed.py`: the same MPEC with relaxed complementarity, its start point, and the Ipopt epsilon continuation.
- `solve.py`: Gurobi and Ipopt interfaces and numerical feasibility checks.
- `solution_space.py`: optimal-face coordinate ranges; unique-price reclear and settlement with balancing.
- `capacity_clearing.py`: hosting capacity shared by a uniform clearing price (Gauss-Seidel best responses, price bisection) and the Gurobi equilibrium check with Cutoff.
- `run.py`: commands (`market`, `ranges`, `mpec`, `mpec_relaxed`, `clearing`, `verify`) and results export.
- `../scripts/tests/test_core.py`: analytical cases that check signs, degeneracy, settlement,
  physical storage constraints, the quadratic costs (including a rising
  generation cost), flexible demand, independent reclearing, the 5-bus market
  value with and without flexible demand, the MIP
  start and the `verify` cutoff, and
  that the settlement profit matches the strong-duality form, strategic bids
  (withholding at a step: 300 capacity-only vs 340 with bids), and that the
  relaxed MPEC reaches the SOS1 optima (skipped without Ipopt on `PATH`), the
  bid selection, and the hosting capacity price.

```powershell
python -m unittest discover -s scripts/tests -v
```
