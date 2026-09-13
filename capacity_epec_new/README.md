# Capacity EPEC — minimal Jacobi model

This folder contains only the runtime files needed to solve the maintained
three-investor BESS capacity game with Jacobi best-response sweeps.

## Included

- `model/run_model.py`: command-line entry point.
- The direct energy-only Python import closure used by that entry point.
- `model/input/market_data_smoothed.json`: maintained IEEE-9 market input.
- `model/input/planner_equal_split_capacities.csv`: planner benchmark divided
  equally among the three investors, for initialization.
- `workflow/`: the existing workflow notes and investigation summaries.

There are no prior model outputs, tests, plotting scripts, investigation
scripts, alternative search drivers, backups, or thesis documents in this
copy. The aFRR implementation is intentionally excluded; this project supports
only the energy-only market.

## Environment

Install the Python dependency:

```powershell
python -m pip install -r requirements.txt
```

The model also requires an IPOPT executable. The current setup uses MA57 as
IPOPT's linear solver.

## Jacobi run from the planner benchmark

Run this command from `capacity_epec_new`:

```powershell
python model/run_model.py --data model/input/market_data_smoothed.json --market energy-only --update-scheme jacobi --max-sweeps 200 --damping 0.25 --proximal-penalty 0.0 --demand-adjustment-penalty-eur-per-mw2 100 --complementarity-epsilon 1e-3 --solver-tolerance 1e-4 --multistart-every-sweeps 0 --initial-capacities model/input/planner_equal_split_capacities.csv --output-dir model/output/planner_jacobi_rho100
```

`--multistart-every-sweeps 0` disables multistart during Jacobi sweeps. The
runner still performs its full zero-proximal final audit after the iteration.
Generated files are written below `model/output/`, which is git-ignored.

## Four Gauss-Seidel order experiments

Run the reproducible matrix from this directory:

```powershell
& .\run_gauss_seidel_matrix.ps1
```

The wrapper creates one common Jacobi seed by applying exactly one damped
simultaneous sweep to the symmetric 5 MW / 15 MWh-per-node start. It then runs
exactly 50 Gauss-Seidel sweeps for the fixed cyclic orders `I1,I2,I3`,
`I2,I3,I1`, and `I3,I1,I2`, plus a fourth run that rotates among those orders
after every sweep. Every endpoint receives the existing zero-proximal,
multistart, exact-reclear audit.

The model's nonzero exit status means that its strict local equilibrium test
did not pass. The wrapper records that status and continues with every remaining
experiment, then exits successfully once all runs have been attempted. A custom
output root or Python executable can be supplied, for example:

```powershell
& .\run_gauss_seidel_matrix.ps1 -PythonExe py -OutputRoot model/output/my_gs_matrix
```

`summary.json` reports `final_relative_regret_by_investor` and
`final_max_relative_regret`, where each deviation gain is divided by that
investor's own absolute incumbent payoff (with a 1 EUR/day denominator floor).
`best_found_epsilon_stationary_candidate` uses a default 1% threshold and
requires valid local audit numerics. It remains explicitly separate from the
strict `converged` flag because IPOPT multistart is not a global best-response
certificate.

## Interpretation

The runner selects candidate responses using exact market reclearing. IPOPT
still supplies local MPEC solutions, so a completed trajectory or a stable
capacity total is not by itself an equilibrium certificate. Only the final
audit fields in `summary.json` determine whether the runner labels the endpoint
an equilibrium candidate.

## Lower-level formulations

`--formulation` selects how ISO optimality is embedded in each investor's MPEC:

- `relaxed-kkt` (default): every complementarity product is at most
  `--complementarity-epsilon`.
- `strong-duality`: primal objective equals dual objective (exact).
- `relaxed-strong-duality`: primal objective minus dual objective is at most
  `--complementarity-epsilon`, in EUR/day.

The relaxed variants let the optimistic MPEC choose favourable prices within
the allowed slack. At the rho = 100 endpoint studied in
`workflow/summary_2026-09-11_16-49.md` this inflated embedded profits by
9-225 EUR/day at epsilon = 1e-3 and by roughly 170 times the allowed gap for the
merchant under relaxed strong duality. Use strong duality, or relaxed strong
duality at 1e-3 EUR/day or below, when embedded and reclear payoffs must agree.
Diagnostic scripts for that study are in `diagnostics/regret_region/`.
