---
name: push_with_overleaf
description: Triggered when the user asks to push changes to the repository or git remote. This skill automates cleaning LaTeX auxiliary build files from the Overleaf subdirectory, staging and committing changes, pushing the main repo to origin/main, and pushing the Overleaf subtree to Overleaf_Alex/main.
---

# Push with Overleaf Subtree Integration

This skill triggers when the user requests a git push, push to repository, or push changes. It ensures that the main repository changes are pushed to `origin main` (GitHub) and any changes to the embedded `Overleaf/` directory are clean, committed, and pushed to the `Overleaf_Alex` remote via git subtree.

## Instructions

Whenever the user asks to "push to repo", "git push", "push changes", "push", or similar commands, follow these steps exactly:

### 1. Clean Overleaf Build/Auxiliary Files
Before committing or pushing any Overleaf changes, delete all LaTeX build/auxiliary files in the `Overleaf/` directory to prevent committing build clutter. Run the following PowerShell command in the workspace root:
```powershell
Get-ChildItem -Path Overleaf -Include *.aux, *.bbl, *.bcf, *.blg, *.fdb_latexmk, *.fls, *.log, *.nav, *.out, *.run.xml, *.snm, *.synctex.gz, *.toc, *.vrb, *-blx.bib -Recurse | Remove-Item -Force
```

### 2. Check for Uncommitted Changes
Check the status of the repository:
```bash
git status --porcelain
```
- If there are uncommitted changes in `Overleaf/` (such as edits to `.tex` files or figures), stage them and commit them.
  - **Stage**: `git add Overleaf/`
  - **Commit**: Create a commit with a descriptive message (e.g., `git commit -m "Update Overleaf: <details>"`). You may combine this with other changes if appropriate.
  - *Note*: Ensure no LaTeX build files (e.g. `*.aux`, `*.log`, etc.) are committed.

### 3. Push Main Repository to Origin
Push the current local commits (including any new Overleaf commits) to the main GitHub repository:
```bash
git push origin main
```

### 4. Push Overleaf Subtree to Overleaf Remote
If there were changes/commits to the `Overleaf/` directory since the last sync, push the subtree to the `Overleaf_Alex` remote on the `main` branch:
```bash
git subtree push --prefix=Overleaf Overleaf_Alex main
```
- Do NOT use or create `.overleaf_alex_push/` inside the workspace.
- The remote branch is `main`.
