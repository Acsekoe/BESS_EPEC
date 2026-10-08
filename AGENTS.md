# Agent instructions

Read `project_context.md` before modeling or thesis-related changes.
Say clearly when a result, assumption, or solver conclusion is uncertain.

Keep model equations readable in the style of urbs: named Pyomo variables and
constraints with explicit `*_rule` functions. Keep input preparation separate
in `model/prepare_input.py`; the primal market model is `model/primal_llp.py`.
Do not replace the explicit primal, dual, or MPEC equations with generic matrix
assembly or dynamically generated constraint families.

Keep maintained model code small and direct. Put sensitivity-study drivers,
tests, diagnostics, and other auxiliary scripts in root `scripts/`. The folder
must be deletable without losing result reproducibility: core code in `model/`
must not depend on it. Preserve required inputs, effective settings, scenarios,
seeds where applicable, software/solver versions, results, and exact rerun
commands outside `scripts/`. Essential reproduction logic belongs in the
maintained model or a saved run specification, not solely in a helper script.

Use `workflow/` for ongoing notes. Whenever the user asks to summarize something,
write `workflow/summary_YYYY-MM-DD_HH-mm.md`, always stamped with the date and
time in Europe/Vienna, both in the file name and inside the document. Later
updates to the same summary also note their time inside the document. Keep it
brief and actionable: latest findings, important decisions, changed files,
verification or limitations, and next steps. A chat response alone does not
fulfill a summary request.

Do not commit LaTeX auxiliary build files; clean them after local compilation.
Keep intentional PDFs, figures, bibliography, and sources. Do not create
`.overleaf_alex_push/`. The Overleaf remote is
`https://git@git.overleaf.com/6a4b5a0bb65d67631b338431`, branch `main`.
