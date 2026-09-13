---
name: summarize
description: Write a session/work summary as a timestamped file in capacity_epec/workflow/. Use whenever the user asks to "summarize", "write a summary", "summarise what we did", "document this session", "write it up", or asks for a recap of work, findings, experiments, or decisions in this repo. Also use when the user asks to summarize a run, a result, or an analysis.
---

# Summarize

Every summary in this repository is written to a **file**, never only to the
terminal.

## Where

```
capacity_epec/workflow/summary_YYYY-MM-DD_HH-MM.md
```

- The folder is `capacity_epec/workflow/` (singular `workflow`). Do not create
  a second `workflows/` folder.
- `YYYY-MM-DD_HH-MM` is the **current local date and time** at the moment of
  writing. Get it — do not guess:
  ```powershell
  Get-Date -Format "yyyy-MM-dd_HH-mm"
  ```
- Never overwrite an existing summary. If the filename already exists, bump the
  minute until it is free.

## What goes in it

Start with a header block:

```markdown
# <Short title of what this session did>

**Written:** YYYY-MM-DD HH:MM <timezone>
**Project:** `capacity_epec`
**Reference data:** `model/input/market_data_smoothed.json`
**Reference market setting:** energy-only, `DemandAdjustment = 0` (`rho = 0`)
**Reference MPEC relaxation:** relaxed KKT, complementarity epsilon `1e-6`

## Executive status
```

Then the substance. Follow the conventions the existing summaries in
`capacity_epec/workflow/` already use:

- **Numbered sections** by topic, not chronological narration.
- **Every number is measured**, by exact market clear + settle or by the
  zero-proximal multistart audit. Say which. Mark anything extrapolated,
  conditional, or inherited from an older run as such.
- **State what was *not* done** as explicitly as what was. If no equilibrium
  was certified, no strong-duality audit was run, or a result is a local
  single-solver result, write that down.
- Close with a **Verification and artifacts** section listing the test result,
  the files touched, and the output directories produced, and a
  **Recommended next step** section.
- Deviations, negative results, and disproven assumptions belong in the
  summary. They are the most valuable part of it.

## After writing

1. Tell the user the path to the new file.
2. If the summary changes the current state of the model, update the status
   banner at the top of `capacity_epec/README.md` to match. The README carries
   the current state; the workflow summaries carry the history.
