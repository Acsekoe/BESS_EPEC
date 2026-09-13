"""Selection-face test at the finest endpoints of every traced bracket.

If a profit jump across a bracket were a change in the ISO's *selected*
price/dispatch on a non-singleton optimal face, the face ranges at the bracket
endpoints would be of the order of the jump.  If the ranges vanish with delta
while the exact (MA27) settlement still differs between the endpoints, the
change is a genuine change of the unique market outcome.

Output: output/selection_faces/faces_bracket_endpoints.csv
"""

from __future__ import annotations

import argparse
import csv

import regret_common as rc
import path_scan
import selection_faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rho", type=float, nargs="+", default=[100.0])
    args = parser.parse_args()
    rows = list(csv.DictReader((rc.OUTPUT_ROOT / "paths" / "bracket_classification.csv").open(newline="", encoding="utf-8")))
    data = rc.market_data(100.0)
    paths, power, energy = path_scan.path_definitions(list(data.nodes))
    tasks = []
    for r in rows:
        if float(r["rho"]) not in args.rho or r["classification"] in ("not_traced", "change_resolves_to_zero"):
            continue
        path = paths[r["path"]]
        for side in ("a", "b"):
            s = float(r[f"final_{side}"])
            full_p, full_e, _, _ = path_scan.point_capacities(path, s, power, energy)
            label = f"{r['path']}|rho{float(r['rho']):g}|{r['bracket']}|{side}|s={s!r}"
            tasks.append((label, float(r["rho"]), path["investor"], full_p, full_e))
    print(f"{len(tasks)} endpoint face evaluations", flush=True)
    selection_faces.run_tasks(tasks, args.workers, "faces_bracket_endpoints.csv")


if __name__ == "__main__":
    main()
