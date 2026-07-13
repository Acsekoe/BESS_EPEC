"""Solver setup helpers for nonlinear Pyomo models."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

import pyomo.environ as pyo


DEFAULT_IPOPT_OPTIONS: dict[str, float | int | str] = {
    "linear_solver": "mumps",
    "max_iter": 1500,
    "tol": 1e-6,
    "acceptable_tol": 1e-5,
    "print_level": 0,
}


def _candidate_ipopt_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("IPOPT_EXECUTABLE")
    if env_path:
        candidates.append(Path(env_path))

    path_hit = shutil.which("ipopt")
    if path_hit:
        candidates.append(Path(path_hit))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "idaes" / "bin" / "ipopt.exe")

    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        miniconda = Path(userprofile) / "miniconda3"
        candidates.extend(
            [
                miniconda / "envs" / "bilevel-ipopt" / "Library" / "bin" / "ipopt.exe",
                miniconda / "Library" / "bin" / "ipopt.exe",
            ]
        )

    return candidates


def find_ipopt_executable() -> Path | None:
    """Return the first usable Ipopt executable path, or None if unavailable."""

    seen: set[Path] = set()
    for candidate in _candidate_ipopt_paths():
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def get_ipopt_solver(
    options: Mapping[str, float | int | str] | None = None,
) -> pyo.SolverFactory:
    """Create a Pyomo Ipopt solver with project-standard defaults."""

    executable = find_ipopt_executable()
    solver_kwargs = {"solver_io": "nl"}
    if executable is not None:
        solver_kwargs["executable"] = str(executable)

    solver = pyo.SolverFactory("ipopt", **solver_kwargs)
    if not solver.available(exception_flag=False):
        searched = "\n".join(f"  - {path}" for path in _candidate_ipopt_paths())
        raise RuntimeError(
            "Ipopt is not available to Pyomo. Set IPOPT_EXECUTABLE to ipopt.exe "
            "or add Ipopt to PATH. Searched:\n" + searched
        )

    merged_options = dict(DEFAULT_IPOPT_OPTIONS)
    if options:
        merged_options.update(options)
    for key, value in merged_options.items():
        solver.options[key] = value

    return solver
