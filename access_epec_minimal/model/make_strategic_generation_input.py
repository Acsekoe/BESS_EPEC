"""Generate the owner-split renewable experiment input.

Derives ``input/market_data_strategic_generation.json`` from the maintained
baseline ``input/market_data.json`` by splitting each aggregate renewable
generator into owner-specific physical units at the same node:

    RES_Wind_N1 -> RES_Wind_I3_N1 (80%), RES_Wind_I4_N1 (20%)
    RES_PV_N6   -> RES_PV_I3_N6   (20%), RES_PV_I4_N6   (80%)
    RES_PV_N8   -> RES_PV_I3_N8   (20%), RES_PV_I4_N8   (80%)

Hourly availability is the share of the parent availability; true economic
costs are inherited unchanged, so the split file must reproduce the aggregate
competitive benchmark exactly (Phase 1 validation gate). Ownership becomes
100% per unit in the experimental population (see investors.py), replacing
the fractional shares of one aggregate unit.

Default run:
    python make_strategic_generation_input.py
"""

from __future__ import annotations

import json
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
BASELINE = MODEL_DIR / "input" / "market_data.json"
TARGET = MODEL_DIR / "input" / "market_data_strategic_generation.json"

# parent -> ordered (child, availability share) pairs
SPLITS: dict[str, tuple[tuple[str, float], ...]] = {
    "RES_Wind_N1": (("RES_Wind_I3_N1", 0.8), ("RES_Wind_I4_N1", 0.2)),
    "RES_PV_N6": (("RES_PV_I3_N6", 0.2), ("RES_PV_I4_N6", 0.8)),
    "RES_PV_N8": (("RES_PV_I3_N8", 0.2), ("RES_PV_I4_N8", 0.8)),
}


def main() -> int:
    raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    for parent, children in SPLITS.items():
        if parent not in raw["generators"]:
            raise ValueError(f"Baseline is missing parent generator {parent}.")
        if abs(sum(share for _, share in children) - 1.0) > 1e-12:
            raise ValueError(f"Availability shares for {parent} must sum to 1.")

    generators: list[str] = []
    for generator in raw["generators"]:
        if generator in SPLITS:
            generators.extend(child for child, _ in SPLITS[generator])
        else:
            generators.append(generator)
    raw["generators"] = generators

    raw["generators_at_node"] = {
        node: [
            child
            for generator in units
            for child, _ in (
                SPLITS[generator] if generator in SPLITS else ((generator, 1.0),)
            )
        ]
        for node, units in raw["generators_at_node"].items()
    }

    raw["generation_cost"] = {
        child: cost
        for generator, cost in raw["generation_cost"].items()
        for child, _ in (
            SPLITS[generator] if generator in SPLITS else ((generator, 1.0),)
        )
    }

    capacity_records = []
    for record in raw["generation_capacity"]:
        generator = record["generator"]
        if generator in SPLITS:
            for child, share in SPLITS[generator]:
                capacity_records.append(
                    {
                        "generator": child,
                        "hour": record["hour"],
                        "capacity_mw": share * float(record["capacity_mw"]),
                    }
                )
        else:
            capacity_records.append(record)
    raw["generation_capacity"] = capacity_records

    raw["metadata"] = {
        **raw.get("metadata", {}),
        "derived_from": BASELINE.name,
        "derivation_script": Path(__file__).name,
        "split_description": (
            "Aggregate renewables split into owner-specific units: wind N1 "
            "80% I3 / 20% I4; PV N6 and N8 each 20% I3 / 80% I4. True costs "
            "inherited; submitted offers default to true costs."
        ),
    }

    TARGET.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"Wrote {TARGET}")
    print(f"generators: {generators}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
