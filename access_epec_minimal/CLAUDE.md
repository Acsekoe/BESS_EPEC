read @project_context.md
if you are uncertain about anything tell me - failing is also a success!

Workflow notes:
- Use workflow/ for ongoing workflow notes and handoff summaries.
- When asked to summarize for a new chat, continue in a new chat, create a
  handoff, or similar, write the chat findings and current project status as a
  Markdown file in workflow/ named summary_YYYY-MM-DD_HH-mm.md.
- Keep workflow summaries concise but actionable: include current objective,
  important decisions, changed files, verification results, and next steps.

Code minimalism:
- Keep the maintained codebase as small and direct as practical.
- Do not add permanent helper, smoke-test, diagnostic, or one-off scripts
  unless they are clearly needed for the ongoing workflow.
- Prefer temporary throwaway checks for local debugging; delete them after use.
- This repository deliberately contains one formulation. Do not add a second
  one without an explicit decision recorded in workflow/.

Verification discipline:
- Convergence is judged only by raw best-response deviations.
- Any candidate equilibrium must be re-audited with proximal penalty zero.
- Report what was actually run, with numbers. If something was not verified,
  say so.
