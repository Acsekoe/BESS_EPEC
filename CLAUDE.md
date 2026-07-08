read @project_context.md
if you are uncertain about anything tell me - failing is also a success!

Workflow notes:
- Use workflow/ for ongoing workflow notes and handoff summaries.
- When asked to summarize for a new chat, continue in a new chat, create a handoff, or similar, write the chat findings and current project status as a Markdown file in workflow/ named summary_YYYY-MM-DD_HH-mm.md.
- Keep workflow summaries concise but actionable: include current objective, important decisions, changed files, verification results, and next steps.

Overleaf / LaTeX workflow:
- Do not commit or push LaTeX auxiliary build files.
- Before pushing Overleaf changes, remove generated files such as *.aux, *.bbl, *.bcf, *.blg, *.fdb_latexmk, *.fls, *.log, *.nav, *.out, *.run.xml, *.snm, *.synctex.gz, *.toc, *.vrb, and *-blx.bib.
- Do not use or create .overleaf_alex_push/ inside the workspace.
- For Overleaf_Alex, push source changes to https://git@git.overleaf.com/6a4b5a0bb65d67631b338431.
- The Overleaf remote branch is main, not master.
- If compiling locally, clean auxiliary files afterwards.
- Keep source files, figures, bibliography files, and intentional PDFs; remove only generated build artifacts.

