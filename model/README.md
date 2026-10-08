# Capacity-only model

One shared fixed-demand market LP supports three independent commands:

- `market`: clear an explicit fixed-capacity profile and verify its dual.
- `ranges`: find dispatch, SOC, price, and investor-profit ranges at that profile.
- `mpec`: optimize one investor's MW/MWh against the profile's fixed rivals.

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
4. `mpec.py`: investment variables, duration limits, explicit Big-M pairs, and profit.

For example, the shared-inverter equation is literally:

```python
def storage_power_limit_rule(m, i, n, t):
    return m.charge[i, n, t] + m.discharge[i, n, t] <= m.power_capacity[i, n]
```

`build_primal()` uses fixed capacities. `build_mpec()` reuses the same physical
rule functions with active-investor capacity expressions and fixed rival
capacities. Its `comp_power_slack_rule()` and `comp_power_dual_rule()` explicitly
enforce complementarity for that inverter constraint. Nonnegativity is declared
on each physical variable; its dual multipliers and complementarity pairs are
explicit in `dual_llp.py` and `mpec.py`.

## Run

Run from the repository root, using Python with `pyomo` and `highspy`
(`python -m pip install -r requirements.txt` if needed):

```powershell
python model/primal_llp.py --output model/output/no_storage
python model/run.py ranges --dispatch all --output model/output/no_storage_ranges

python model/primal_llp.py --profile model/input/capacities_storage.json --output model/output/fixed_storage
python model/run.py ranges --profile model/input/capacities_storage.json --output model/output/fixed_storage_ranges

python model/mpec.py --investor I1 --seconds 60 --output model/output/mpec_i1
```

The default IEEE-9 profile starts all four investors at zero capacity. The
storage example installs 10 MW / 30 MWh each for I1 and I2 at N6. Only the
active investor's capacities are replaced in `mpec`; all others remain fixed.
Use an exported `capacities.json` as `--profile` to independently clear or
analyze a chosen investment.

The previous `python model/run.py market ...` and `python model/run.py mpec ...`
commands also remain valid. Both entry points use the same runner and exports.

For a small, fully solved analytical example:

```powershell
python model/run.py mpec --data model/input/toy_market.json --profile model/input/toy_capacities.json --investor I --node-limit 10 --dual-m 10000 --output model/output/toy_mpec
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
installed MW per node. `--dual-m` bounds inequality multipliers only in the
MPEC. `--seconds` limits each main market/MPEC solve. Range subproblems use a
60-second limit individually. `--mip-gap` controls MILP proof tolerance.

## Outputs and limits

Every command writes `run_config.json`, `input_profile.json`, and `summary.json`.
Use an empty output directory for each run; existing results are not overwritten.
Market runs also export dispatch and nodal prices. Range runs export `ranges.csv`;
default quantity ranges cover storage, `--dispatch all` includes every market
variable, and `--dispatch none` covers prices and profits only.

MPEC runs with proven optimal status export the chosen `capacities.json`,
embedded dispatch/prices, independently recleared dispatch/prices, direct-profit
identity error, optimality residuals, and the fixed-capacity profit interval.
Time-limited runs report status and objective bounds and return a nonzero exit
code; no unproven incumbent is presented as an optimal best response.

The dual face has no artificial price or multiplier bounds. The MPEC is a
bounded KKT/Big-M MILP and is **optimistic**, subject to the validity of its
multiplier bound. An arbitrary reclear need not reproduce its selected profit.
The full IEEE-9 MPEC may require substantially more than 60 seconds. Neither a
single best response nor coordinate ranges enumerate market equilibria.

See `../project_context.md` for all physical/economic assumptions. For the LP
duality and complementary-slackness basis, see
[Vandenberghe's duality notes](https://seas.ucla.edu/~vandenbe/ee236a/lectures/duality.pdf).

## Code and verification

- `prepare_input.py`: input loading/validation, sets, parameters and annualized costs.
- `primal_llp.py`: named physical variables, explicit market equations and objective.
- `dual_llp.py`: explicit stationarity, dual objective and fixed-capacity profit.
- `mpec.py`: explicit capacity investment and binary complementarity; settlement check.
- `solve.py`: HiGHS interface and numerical feasibility checks.
- `solution_space.py`: optimal-face coordinate ranges.
- `run.py`: three commands and results export.
- `../scripts/tests/test_core.py`: analytical cases that check signs, degeneracy, settlement,
  physical storage constraints, and independent reclearing.

```powershell
python -m unittest discover -s scripts/tests -v
```
