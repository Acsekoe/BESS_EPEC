# Complementarity methods, degradation cost, dispatch slope: evidence

Written 2026-10-12 14:55 (Europe/Vienna). Records the runs behind three
decisions of 2026-10-12. The run outputs were deleted afterwards (fresh start),
and Big-M, product, strong-duality, and relaxed MPEC code was removed; it is
in git history (base commit `4bd4b47` plus that day's uncommitted work).

All runs: I1, rivals at zero storage, one benchmark day, eps = 1e-4, linear
(strong-duality) objective, Gurobi 13.0.3, unless stated.

## 1. SOS1 is the only complementarity method

I1 at N9, `capacities_capex120_curtail_d3_slope1e-4.json`; true optimum
6902.13 EUR/day at 172.73 MW / 450.62 MWh. Root relaxation 7742.93 for every method.

| method | result | time | verification |
|---|---|---|---|
| SOS1 | optimal 6902.13 | 7 s | passes (products 3e-12) |
| Big-M, dual_m 1e5 | optimal 6902.13 | 19 s | passes (products 1.5e-11) |
| product `slack*mu == 0`, mu <= 1e5 | optimal 6902.13 | 142 s | fails: primal-dual gap 5.9e-4 (tolerance x 97 multipliers at 1e5) |
| product, mu <= 1e3 | infeasible (`mu_shed_lower = VOLL - price` needs about 1e4) | - | - |
| product `slack*mu <= 1e-2` | no proof: best 6902.42, bound 7619.6 | 300 s | - |
| product `slack*mu <= 1e-4` | no feasible point, bound 7711.5 | 300 s | - |
| strong duality `market_cost <= dual_value`, mu bound 1e2 to 1e5 | no proof, bound 7535-7743 | 300 s | - |

- Product: on `[0,S] x [0,M]` the McCormick envelope of `slack*mu = 0` is
  `slack/S + mu/M <= 1`, the Big-M LP relaxation; the bounds act as Big-M.
- Relaxed products branch worse than exact: `slack >= c` forces `mu = 0` only
  with `== 0`; with `<= e` it gives `mu <= e/c`, so no pair is settled.
- Strong duality: Gurobi turns the convex quadratic terms into 508 quadratic
  constraints, so every node is a barrier QCP (1 127 nodes in 300 s).
  `MIQCPMethod=1` (bound 7707) and `--node-limit 300` (bound 7716) did not help.
- Four candidate nodes (N3 N6 N8 N9), Big-M, 600 s, 1 % target: stopped at
  2.3 % gap (incumbent about 8080, bound about 8268, root bound unchanged).
  Every formulation shares the root relaxation, so the bound problem is the
  relaxation, not the complementarity method.

## 2. Degradation d = 3 EUR/MWh (was 15)

Reason and sources in `project_context.md`. Sensitivity, I1 at N9, slope 1e-4,
Big-M, all proven optimal, profit from the independent unique-price reclear:

| d [EUR/MWh] | MW | MWh | profit [EUR/day] | cycles/day |
|---|---|---|---|---|
| 0 | 172.73 | 456.72 | 8220.19 | 1.00 |
| 3 | 172.73 | 450.62 | 6902.13 | 1.00 |
| 10 | 159.24 | 429.14 | 3828.09 | 1.00 |
| 15 | 108.83 | 344.12 | 1893.98 | 1.00 |

Each EUR/MWh of d costs about 440 EUR/day (about 450 MWh/day throughput).
Per MWh of capacity and day: margin about 56, capex 38, so d = 15 leaves about
3 and d = 3 about 15. Verification failed at d = 0, 10, 15 only by Big-M
tolerance (products up to 7.5e-3; 98 multipliers at dual_m).

## 3. Dispatch slope 1e-4 EUR/MWh per MW

| slope | N9 reclear profit, time | four nodes, 600 s: incumbent / bound |
|---|---|---|
| 0 | 6906.43, 4 s | 8082.0 / 8269.4 |
| 1e-4 | 6902.13, 19 s | 8080.3 / 8267.8 |
| 1e-3 | 6863.18, 15 s | 8064.8 / 8253.5 |

(Big-M.) Any slope > 0 makes dispatch unique: Gurobi and Ipopt dispatches
agree to 0.06 MW or better; the SOC curve cannot shift (full cycles). At 1e-3
about 33 of 39 EUR/day come from full-power midday charging at 173 MW, where
no tie exists. The slope does not move the four-node bound.

## Rerun (SOS1, current code)

```powershell
python model/run.py mpec --investor I1 --nodes N9 --profile model/input/capacities_capex120_curtail_d3_slope1e-4.json --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
python model/run.py mpec --investor I1 --nodes N9 --profile model/input/capacities_capex120_curtail_d<d>_slope1e-4.json --balancing-eps 1e-4 --seconds 300 --output model/output/<new_dir>
```
