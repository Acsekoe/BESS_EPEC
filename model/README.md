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

```powershell
python model/primal_llp.py --output model/output/no_storage
python model/run.py ranges --dispatch all --output model/output/no_storage_ranges

python model/primal_llp.py --profile model/input/capacities_storage.json --output model/output/fixed_storage
python model/run.py ranges --profile model/input/capacities_storage.json --output model/output/fixed_storage_ranges

python model/mpec.py --investor I1 --nodes N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
python model/run.py mpec --investor I1 --nodes N3 N6 N8 N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4 --mip-gap 0.01 --seconds 600 --output model/output/<new_dir>
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

The default IEEE-9 profile starts all four investors at zero capacity. The
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

## Modified IEEE 5-bus case

`ieee5_market_data.json` / `ieee5_capacities.json` (built by
`scripts/data/build_ieee5.py`; every number is in the JSON): lines 1-2, 1-4,
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
python scripts/data/build_ieee5.py .
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
Time-limited runs report status and objective bounds and return a nonzero exit
code; no unproven incumbent is presented as an optimal best response.

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
- `mpec.py`: capacity investment and optional strategic bids, SOS1 complementarity, physical bounds, and settlement/strong-duality profit.
- `mpec_relaxed.py`: the same MPEC with relaxed complementarity, its start point, and the Ipopt epsilon continuation.
- `solve.py`: Gurobi and Ipopt interfaces and numerical feasibility checks.
- `solution_space.py`: optimal-face coordinate ranges; unique-price reclear and settlement with balancing.
- `capacity_clearing.py`: hosting capacity shared by a uniform clearing price (Gauss-Seidel best responses, price bisection).
- `run.py`: commands (`market`, `ranges`, `mpec`, `mpec_relaxed`, `clearing`) and results export.
- `../scripts/tests/test_core.py`: analytical cases that check signs, degeneracy, settlement,
  physical storage constraints, the quadratic costs, independent reclearing, and
  that the settlement profit matches the strong-duality form, strategic bids
  (withholding at a step: 300 capacity-only vs 340 with bids), and that the
  relaxed MPEC reaches the SOS1 optima (skipped without Ipopt on `PATH`), the
  bid selection, and the hosting capacity price.

```powershell
python -m unittest discover -s scripts/tests -v
```
