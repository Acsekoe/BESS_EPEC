# Exact Tikhonov strong-duality MPEC and EPEC

This folder is an experimental layer. The maintained outer entry point,
`model/epec_diagonalization.py`, is locked to the clean fixed-demand exact-
balance strong-duality EPEC. The experimental drivers here reuse its generic
diagonalization machinery programmatically but do not alter its command-line
baseline.

This folder now uses one active lower-level formulation: the matched
finite-gamma soft-balance primal and Tikhonov-regularized dual, enforced by
exact strong duality. It has no Scholtes complementarity products and no
complementarity epsilon.

The superseded relaxed-KKT experiment is retained under `old/` only for
reproducibility. New experiments should not import it.

## Active files

- `primal_llp.py`: fixed-demand physical market with fixed BESS capacity.
- `dual_llp.py`: explicit unregularized dual.
- `dual_tikhonov_llp.py`: dual with
  `-(gamma/2) * sum(lambda[n,t]^2)`.
- `soft_balance_llp.py`: matched quadratic soft-balance primal and regularized
  dual, including standalone verification.
- `strong_duality_formulation.py`: exact finite-gamma single-investor MPEC
  builder, soft-market initialization, diagnostics, and same-fleet audit.
- `mpec_strong_duality.py`: standalone I1 exact MPEC.
- `mpec_strategic_operation_strong_duality.py`: standalone strategic-operation
  MPEC and reusable builder. The investor submits hourly charge/discharge
  quantities and optional two-sided bid prices; its initialization and audit
  solve the same fixed-fleet, fixed-bid soft primal/dual pair independently.
- `compare_four_investor_strong_duality.py`: four standalone MPECs and full
  anticipated-LMP comparison.
- `jacobi_epec.py`: four-investor simultaneous Gauss--Jacobi EPEC using the
  exact strong-duality builder.
- `epec.py`: unified command-line runner for capacity-only and strategic-
  operation EPECs. Both modes use finite-gamma strong duality and support
  parallel Jacobi best responses.
- `common.py`: data calibration and Ipopt helpers.

## Mathematical formulation

For `gamma > 0`, the matched lower-level pair is

```text
primal:  minimize C(x) + sum(h[n,t]^2)/(2*gamma)
dual:    maximize D(y) - (gamma/2)*sum(lambda[n,t]^2)
```

The MPEC imposes primal feasibility, dual feasibility, and equality of these
two objective values. The explicit equation

```text
h[n,t] + gamma*lambda[n,t] == 0
```

is retained to avoid numerical cancellation in the global strong-duality
equality. This selects one price vector for each fixed finite-gamma market, but
it also permits physical imbalance. At `gamma=1e-3`, a 60 EUR/MWh price implies
a 0.06 MW nodal residual. The final exact unregularized re-clear is therefore
reported separately and must not be confused with the MPEC's soft market.

## Jacobi EPEC

Every sweep uses one frozen common fleet. I1--I4 solve independent exact
strong-duality best responses against that same rival snapshot. After all four
solves finish, proposals are applied simultaneously:

```text
X_next = (1 - damping)*X_current + damping*X_best_response.
```

The shared nodal limit is then enforced by proportional projection where
needed. Fresh runs start from zero economic MW/MWh everywhere, while Ipopt gets
a separate positive numerical seed of 10 MW/node and four hours.

The default damping is `0.25`. It is conservative but justified by the smoke
test: I1 chose 60.55 MW against the quarter-applied first-sweep fleet but nearly
zero against the half-applied fleet, revealing a sharp response regime between
those states. For a fixed best response, 0.25 needs about eight sweeps to apply
90% of a move, compared with about four sweeps at 0.50. Once a stable trend is
observed, `0.35` or `0.50` is a useful faster sensitivity.

Convergence cannot be manufactured by lowering damping. The wrapper requires:

1. all four MPECs terminate optimally;
2. damped MW and MWh relative changes are below `CONVERGENCE_TOL_REL`;
3. raw undamped best-response MW and MWh residuals are below the same tolerance.

Both sets of residuals are printed and exported after every completed sweep.

## Running

From the repository root:

```powershell
python model/tikhonov_kkt/soft_balance_llp.py
python model/tikhonov_kkt/mpec_strong_duality.py
python model/tikhonov_kkt/mpec_strategic_operation_strong_duality.py
python model/tikhonov_kkt/epec.py --mode capacity --parallel-workers 2
python model/tikhonov_kkt/epec.py --mode strategic-operation --parallel-workers 2
python model/tikhonov_kkt/compare_four_investor_strong_duality.py
python model/tikhonov_kkt/jacobi_epec.py
```

Strategic-operation mode submits hourly charge/discharge quantities and, by
default, charging buy-bid and discharging sell-offer prices. Use
`--no-strategic-bid-prices` for quantity-only strategic operation. The unified
runner always uses Gauss--Jacobi because independent same-snapshot best
responses are what make `--parallel-workers` valid.

Fresh strategic-operation runs start economically from exactly zero MW/MWh
and zero charge/discharge offers at every investor-node-hour. Inactive prices
start at the truthful degradation values. Ipopt receives a separate positive
10 MW/node, four-hour numerical capacity guess; this guess is not part of the
economic iteration state. Use `--use-jacobi-initializer` only when the extra
projected best-response initialization sweep is desired.

Both modes use the normalized proximal strategy selector. Its staircase
coefficient is zero for iterations 1--10, 1 EUR/MW^2/day for iterations
11--15, 2 for iterations 16--20, and increases by 1 every five iterations
thereafter. Capacity mode penalizes normalized MW/MWh changes. Strategic mode
also includes withheld quantities and normalized bid-price changes; its direct
strategic epsilon penalty remains zero. The Jacobi initializer does not receive
the proximal penalty.

The default EPEC run uses `gamma=1e-3`, damping `0.25`, at most 40 Jacobi
sweeps, two parallel workers, and the 200 MW shared nodal limit. It checkpoints
after every completed sweep under `model/output/`. Set `RESUME_FROM` and
`RESUME_STAGE_NUMBER` in `jacobi_epec.py` to continue from a checkpoint.

The output distinguishes the last investor-specific anticipated MPEC prices
and profits from the common final unregularized market re-clear. A run that
reaches its sweep limit is diagnostic and must not be reported as an
equilibrium.
