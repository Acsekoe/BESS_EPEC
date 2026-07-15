# Agent Instructions

Read `project_context.md` before making modeling or thesis-related changes.

If you are uncertain about anything, say so clearly. Failing is also a success when it exposes a real issue or missing assumption.

## Workflow Notes

- Use `workflow/` for ongoing workflow notes and handoff summaries.
- When asked to summarize for a new chat, continue in a new chat, create a handoff, or similar, write the chat findings and current project status as a Markdown file in `workflow/` named `summary_YYYY-MM-DD_HH-mm.md`.
- Keep workflow summaries concise but actionable: include current objective, important decisions, changed files, verification results, and next steps.

## Code Minimalism

- Keep the maintained codebase as small and direct as practical.
- Do not add permanent helper, smoke-test, diagnostic, migration, or one-off scripts unless they are clearly needed for the ongoing workflow.
- Prefer temporary throwaway checks for local debugging; delete them after use when they are not part of the model, data preparation, results export, or documented workflow.
- If a helper script is useful only during development, either remove it before finishing or clearly explain why it should remain.

## Overleaf / LaTeX Workflow

- Do not commit or push LaTeX auxiliary build files.
- Before pushing Overleaf changes, remove generated files such as `*.aux`, `*.bbl`, `*.bcf`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.log`, `*.nav`, `*.out`, `*.run.xml`, `*.snm`, `*.synctex.gz`, `*.toc`, `*.vrb`, and `*-blx.bib`.
- Do not use or create `.overleaf_alex_push/` inside the workspace.
- For `Overleaf_Alex`, push source changes to `https://git@git.overleaf.com/6a4b5a0bb65d67631b338431`.
- The Overleaf remote branch is `main`, not `master`.
- If compiling locally, clean auxiliary files afterwards.
- Keep source files, figures, bibliography files, and intentional PDFs; remove only generated build artifacts.
