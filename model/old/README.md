# Archived model work

This directory preserves all model branches and results that are not part of
the minimal capacity-only baseline.

- `data/`: superseded and sensitivity input datasets, plus the original input workbook.
- `output/`: all model outputs produced before the cleanup.
- `scripts/`: superseded root-level scripts, diagnostics, safety copies, KKT
  alternatives, and strategic-operation code.
- `variants/auction/`: nodal-access auction formulation and results.
- `variants/stochastic/`: stochastic planner and MPEC experiment.
- `variants/tikhonov_kkt/`: finite-gamma Tikhonov formulations and recovery runs.
- `legacy_capacity_baseline/`: superseded monolithic capacity MPEC, planner,
  export modules, and solver helper from immediately before the minimal rewrite.

These files are retained for traceability. Because they were moved as an
archive, their historical relative imports and default paths are not guaranteed
to run from the new location without adjustment.
