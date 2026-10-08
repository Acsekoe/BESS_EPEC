# Explicit model equations: readability refactor

## Request and implementation

The user requested an urbs-like structure so the physical model and MPEC can be
read directly. Inspected the official urbs `model.py`, `input.py`, and storage
module. Adopted the pattern of named Pyomo components and separate `*_rule`
functions; retained this project's existing mathematics and JSON inputs.

- `model/prepare_input.py`: input loading/validation, sets, parameters, financing.
- `model/primal_llp.py`: explicit dispatch variables and physical equations.
- `model/dual_llp.py`: explicit multiplier signs, stationarity and dual objective.
- `model/mpec.py`: explicit investment constraints and all eleven complementarity
  families, each with its own binary, slack constraint and multiplier constraint.
- `model/solve.py`: shared solver interface and residual checks.
- `run.py` and `solution_space.py` now use the named components. The old
  matrix-assembly `market.py` is removed. Tests remain in `scripts/tests/`.

Both `python model/primal_llp.py --output ...` and
`python model/mpec.py --investor I1 --nodes N6 --output ...` run independently.
Existing `run.py market/ranges/mpec` commands and exported coordinate labels
remain valid. Python builders now return a model alone, not a model/row tuple.
The README, model README, project context and agent instructions are updated.

## Verification against the previous implementation

- Nine analytical/regression tests pass, including IEEE-9 primal/dual costs,
  congestion, price/dispatch ambiguity, storage, settlement and unbounded prices.
- No-storage market cost unchanged: 394587.85188079276 EUR/day.
- Fixed-storage market cost difference: -5.82e-11 EUR/day.
- N6-only I1 MPEC at dual Big-M 20000 solved to optimal status with independent
  reclearing and verification passed. Profit differs by 2.39e-9 EUR/day from
  the previous result; power/energy capacity changes are below 9e-12 MW/MWh.
- Toy MPEC still returns 5 MW / 5 MWh, 140 EUR profit and the same payoff range.
- Compared every minimum/maximum endpoint of the toy and N6 fixed-capacity
  range exports against previous outputs: maximum differences 5.69e-14 and
  9.46e-7 respectively. The N6 case still has 28 nonunique node-hour prices.
- New runs are saved under ignored `model/output/readable_*` directories.
  No output was overwritten; nothing was committed or pushed.

## Next steps

Read `primal_llp.py` first, then `dual_llp.py` and `mpec.py`. The prior research
plan remains: map capacity neighborhoods and price/profit selection before
reintroducing an equilibrium iteration. Big-M validity and optimistic selection
limitations remain unchanged by this refactor.

Style reference: https://github.com/tum-ens/urbs/blob/master/urbs/model.py
