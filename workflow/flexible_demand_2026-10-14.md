# Flexible demand in the pockets (2026-10-14, Europe/Vienna)

Smoothing the pocket price step before studying scarce nodal capacity.
Python 3.13.7, Pyomo 6.10.1, Gurobi 13.0.3, Ipopt 3.13.2 (IDAES), eps 1e-4.

## Decision

- Each pocket node gets a price-responsive flexible demand (e.g. power-to-heat),
  willingness to pay falling linearly from 60 EUR/MWh at 0 MW to 0 at full size:
  **N2 10 MW, N5 25 MW** (`flexible_demand` in the market data;
  `run.py --flexible-demand-scale s` scales it, 0 removes it).
- Size and price are stylized, not sourced. 20 / 50 MW is the sensitivity
  (scale 2 of the current data).

## Why not eps

Step scan, I4 alone at N5 (4.027 h), no flexible demand: the profit jump
(about +29 000 EUR/day) spans 0.005 MW at eps 1e-4, 0.7 MW at 1e-2 and about
8 MW at 1e-1. At 1e-1 the balancing term is about 28 MW of fictitious hub supply
at 55 EUR/MWh (evening hub price -1.5 EUR/MWh, owners' base profits -6 to
-18 %). Best responses against zero rivals barely move with eps up to 1e-2
(`best_responses_zero_rivals.json`, sets `noflex`).

## Size scan

`flex_size_scan.py`, one battery per pocket, no other storage, exact reclears.
**Scales in these files are relative to N2 20 / N5 50 MW** (the data before the
switch); file `*_s0.5_*` is the chosen size. Results in this folder:
`N5_s*_prices.csv`, `N5_s*_profits.csv`, `N2_s*` likewise, figures
`flex_size_vs_battery.png`, `flex_size_no_storage_prices.png`,
`flex_scale0.5_prices.png`, `flex_scale1_prices.png`. Table in
`model/README.md`, section "Flexible demand in the pockets".

- The plateau is the battery's break-even (N5 47.7, N2 41-42 EUR/MWh) at every
  size; the size sets the ramp width and how much cheap surplus the load takes.
- Without the load, N2 also fails on the discharge side: beyond about 77 MW
  (6 h) the day price falls to 3.1, selected by eps.
- Owners value a pocket battery far above merchants (best gain 45-64k vs
  2.4-7k EUR/day), mostly from the pocket price lift on their RES output.

## Other changes the same day

- `mpec` exports Gurobi's incumbent after a time limit (exit code 2, labelled).
- `unique_price_reclear` retries at Ipopt tolerance 1e-10 when 1e-12 stalls
  (5-bus without storage and flexible demand).

## Rerun (from the repository root)

```powershell
python scripts/data/build_five_bus.py .
python -m unittest discover -s scripts/tests -v
python model/run.py market --balancing-eps 1e-4 --output model/output/<dir>
# size scan; scales relative to the CURRENT data (1 = N2 10 / N5 25 MW):
foreach ($s in 0, 0.2, 0.5, 1, 2, 4) {
  python scripts/diagnostics/flex_size_scan.py . --node N5 --duration 4.027 --scale $s --powers (0..40 | % { $_ * 5 }) --output model/output/<dir>/N5_s$s
  python scripts/diagnostics/flex_size_scan.py . --node N2 --duration 6 --scale $s --powers (0..30 | % { $_ * 4 }) --output model/output/<dir>/N2_s$s
}
# eps step scan without flexible demand (I4 at N5):
python scripts/diagnostics/equilibrium_probe.py . --profile model/input/five_bus_capacities.json --flexible-demand-scale 0 --balancing-eps 1e-2 --case "I4.N5=147/591.905" --prices N5
```

The archived scan used scales 0, 0.1, 0.25, 0.5, 1, 2 of the 20 / 50 MW data, so
its file names differ by a factor of 2 from a rerun with the current data.
