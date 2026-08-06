# Exact Tikhonov strong-duality EPEC handoff

## Decision

The active Tikhonov approach is now the matched finite-gamma primal/dual MPEC
with exact strong duality. The relaxed-KKT/Scholtes implementation is archived
under `model/tikhonov_kkt/old/` and is not used by the new EPEC runner.

## Changes

- Added `tikhonov-strong-duality` as an EPEC lower-level-optimality mode.
- Replaced `model/tikhonov_kkt/jacobi_epec.py` with an exact strong-duality
  Gauss--Jacobi driver starting economically from zero MW/MWh.
- Kept the 10 MW/node, four-hour Ipopt seed separate from the economic state.
- Set default damping to 0.25 and 40 maximum sweeps at `gamma=1e-3`.
- Convergence now requires both damped iterate changes and raw undamped
  best-response MW/MWh residuals below 0.02, with all four MPECs optimal.
- Checkpoints and run configuration identify the exact method and gamma.
- Final capacities are still re-cleared once in the original unregularized
  physical market as a separate settlement/audit.

## Verification

A serial one-sweep smoke test from the zero fleet solved all four MPECs
optimally and reproduced the standalone first-sweep proposals:

| Investor | Proposed MW | Proposed MWh | Matched gap (EUR/day) |
|---|---:|---:|---:|
| I1 | 187.182 | 547.213 | `5.01e-8` |
| I2 | 187.182 | 547.213 | `4.56e-8` |
| I3 | 132.012 | 465.243 | `3.70e-7` |
| I4 | 187.336 | 547.932 | `5.56e-8` |

The integration smoke used damping 0.50, so its first applied state contains
half of each proposal. The
raw residuals are large, so the smoke run correctly reports nonconvergence.
The finite-gamma maximum hourly system residual is about 0.54 MW in each MPEC.
A follow-up I1 solve against that nonzero damped fleet also terminated optimal,
with a `2.97e-9` EUR/day matched gap, validating the rival-capacity path used
from sweep 2 onward. Its near-zero best response indicates that substantial
competitive movement should be expected and confirms the need for damping;
it is not itself an equilibrium result.

The smoke test exposed and fixed a wrapper bug: without explicit task-level
initial guesses, the shared solver overwrote the intended 10 MW numerical seed
with the zero economic state and sent I4 to a materially worse local solution.

## Damping recommendation

Start the full run at 0.25. I1 chose 60.55 MW against the quarter-applied
first-sweep fleet (current I1 capacity about 46.80 MW) but nearly zero against
the half-applied fleet. This sharp response change makes 0.25 prudently
conservative rather than unnecessarily low. For a fixed best response it takes
about eight sweeps to apply 90% of a move versus about four at 0.50. The
raw-residual convergence test makes the stopping decision independent of this
mechanical slowdown. Test 0.35 or 0.50 only after the raw response trajectory
shows a stable trend.

## Run command

```powershell
python model/tikhonov_kkt/jacobi_epec.py
```
