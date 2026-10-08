# BESS capacity-only research core

The maintained code is in [`model/`](model/README.md). Read
[`project_context.md`](project_context.md) for assumptions and the research plan.

For the equations, start with [`primal_llp.py`](model/primal_llp.py): it declares
named Pyomo variables and constraints, with one explicit `*_rule` function per
equation. [`prepare_input.py`](model/prepare_input.py) reads and prepares the
inputs, [`dual_llp.py`](model/dual_llp.py) contains the stationarity equations,
and [`mpec.py`](model/mpec.py) adds investment and Big-M complementarity.

```powershell
python model/primal_llp.py --output model/output/market_run
python model/run.py ranges --output model/output/market_ranges
python model/mpec.py --investor I1 --nodes N6 --seconds 60 --output model/output/n6_best_response
```

These run fixed-capacity market clearing, lower-level solution-space analysis,
and one investor's capacity best response independently. The MPEC command above
restricts investment to N6. There is currently no multi-investor equilibrium loop.

The October 2026 restart deletes the previous model trees. `Overleaf/` retains
historical thesis/presentation material; it has not yet been updated to this core.
Current verification and research findings are recorded in `workflow/`.

## Workflow conventions

When I ask to summarize something, create or update `workflow/summary.md`.
Include the date and time in the document, using the Europe/Vienna timezone.
Briefly describe the latest findings, important decisions, changed files,
verification results or limitations, and next steps. Write the summary to the
file rather than providing it only in chat.

Sensitivity-study drivers, tests, diagnostics, and other auxiliary scripts belong
in the root `scripts/` folder. This folder must be safe to delete without losing
the ability to reproduce results. Keep the maintained model and results-export
code in `model/`; it must not import or depend on `scripts/`.

Store results, required inputs, effective run settings, scenario definitions,
random seeds where applicable, software/solver versions, and exact rerun commands
outside `scripts/`. Record findings and experiment instructions in `workflow/`.
If reproducing a result requires logic found only in a helper script, move that
logic into the maintained model or capture it in a reproducible run specification
before treating the script as disposable.
