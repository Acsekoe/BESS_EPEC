# Maintained BESS EPEC formulations

The maintained model uses fixed demand, DC market clearing, and endogenous
BESS MW/MWh capacity. It supports price-taking operation, strategic operating
prices, and strategic hourly quantity availability.

## Active files

- `primal_market_clearing_model.py` — primal lower-level market LP.
- `dual_market_clearing_model.py` — explicit dual LP and primal/dual check.
- `mpec_strong_duality.py` — single-investor MPEC using primal feasibility,
  dual feasibility, and strong duality.
- `mpec_relaxed_kkt.py` — capacity-only smooth Scholtes relaxation of every
  lower-level KKT complementarity product for Ipopt.
- `mpec_kkt_bigm.py` — the same MPEC using explicit KKT complementarity and
  binary Big-M linearisation.
- `mpec_strategic_operation.py` — capacity-and-price MPEC with hourly charging
  bids and discharge offers, but no quantity withholding.
- `mpec_strategic_price_relaxed_kkt.py` — price-only MPEC with full
  availability, one physical shared-inverter limit, and Scholtes-relaxed
  lower-level KKT conditions.
- `mpec_strategic_quantity_relaxed_kkt.py` — relaxed-KKT MPEC supporting
  quantity-only bids or combined hourly charging/discharging quantity-price
  pairs. The complete equations are in
  `strategic_price_quantity_formulation.md`.
- `mpec_strategic_access_relaxed_kkt.py` — co-optimised nodal connection
  allocation and physical market clearing. Investors submit requested MW,
  non-negative pay-as-bid EUR/MW-day bids, and independently choose MWh; the
  equations are in `strategic_access_formulation.md`.
- `jacobi_diagonalization.py` — constructs fixed-rival best responses and
  applies simultaneous Gauss-Jacobi capacity, strategic-price, or
  strategic-quantity updates.
- `run_model.py` — the executable and sole home of solver, Big-M,
  regularisation, damping, and convergence settings.
- `input/market_data.json` — the only active input.

All superseded models, scripts, datasets, and results are under `old/`.

## Run

```powershell
python model/primal_market_clearing_model.py
python model/dual_market_clearing_model.py --compare
python model/run_model.py --formulation strong-duality
python model/run_model.py --formulation relaxed-kkt --complementarity-epsilon 1e-3
python model/run_model.py --formulation strong-duality --parallel-workers 4
python model/run_model.py --formulation kkt-bigm --big-m-dual 800
python model/run_model.py --formulation strategic-operation --bid-price-bound 500
python model/run_model.py --formulation strategic-price-relaxed-kkt --complementarity-epsilon 1e-3 --proximal-penalty 0.01
python model/run_model.py --formulation strategic-quantity --complementarity-epsilon 1e-3
python model/run_model.py --formulation strategic-price-quantity --complementarity-epsilon 1e-3 --proximal-penalty 0.01
python model/run_model.py --formulation strategic-access --node-limit-mw 40 --access-request-limit-mw 200 --complementarity-epsilon 1e-3 --proximal-penalty 0.01
python model/run_model.py --formulation strategic-quantity --complementarity-epsilon 1e-3 --damping 1 --max-sweeps 5 --run-to-max-sweeps --parallel-workers 4 --max-solve-seconds 600 --output-dir model/output/strategic_quantity_undamped_5sweeps
python model/run_model.py --investor-config model/input/investors_merchant_wind_pv.json --formulation strategic-operation --node-limit-mw 200 --initial-power-mw 0 --damping 0.25 --max-sweeps 20 --parallel-workers 3
python model/run_model.py --resume-from model/output/<run>/checkpoint.json --max-sweeps 60
python model/run_model.py --investor-config model/input/investors_merchant_wind_pv.json --parallel-workers 3
```

The `--formulation` selector exposes eight maintained MPECs:

1. `strong-duality`: capacity-only nonlinear MPEC;
2. `relaxed-kkt`: capacity-only nonlinear MPEC with each nonnegative
   complementarity product bounded by `--complementarity-epsilon` (default
   `1e-3`), solved by Ipopt;
3. `kkt-bigm`: capacity-only KKT/Big-M MILP;
4. `strategic-operation`: capacity plus hourly two-sided storage prices,
   embedded by strong duality;
5. `strategic-price-relaxed-kkt`: capacity plus hourly two-sided prices,
   full physical MW availability, and relaxed-KKT lower-level optimality;
6. `strategic-quantity`: capacity plus hourly maximum charging and
   discharging MW, embedded through relaxed KKT conditions;
7. `strategic-price-quantity`: capacity plus two hourly quantity-price pairs,
   one for charging and one for discharging, embedded through the same relaxed
   KKT conditions;
8. `strategic-access`: investors submit nodal requested MW, non-negative
   pay-as-bid access prices, and independent energy capacities. The lower
   level jointly awards only scarce nodal MW and clears physical storage
   operation through relaxed KKT conditions.

In `strategic-access`, `--node-limit-mw` is the common per-node connection
limit and `--access-request-limit-mw` caps each investor's total request over
all nodes. `--access-bid-bound` bounds willingness to pay in EUR/MW-day;
negative bids and access subsidies are excluded. Awarded MW is a lower-level
outcome, while each investor independently chooses MWh subject to the linear
technical bounds `2 * awarded MW <= MWh <= 8 * awarded MW`. Pay-as-bid access
payments are subtracted from investor profit. All SOC, shared-inverter,
dispatch, and network equations remain in the lower level.
After each simultaneous Jacobi strategy update, one exact common HiGHS
clearing determines the awarded fleet. Access runs write
`final_access_bids.csv`, `final_nodal_access.csv`, and format-v7 checkpoints.
The corrected relaxed KKT contains no cubic constraints: SOC complementarity
uses `(X_energy - SOC) * (-delta)`, and awarded-MW stationarity contains no
duration multiplier.

Its active investor chooses separate hourly maximum charging and discharging
quantities. The ISO pays and charges only for realised dispatch at the nodal
LMP. The complete one-hour bids are constrained by installed MW and by the
anticipated beginning-of-hour SOC; realised operation is bounded by the bids,
the shared inverter MW, cyclic SOC, and installed MWh. The ISO objective
contains generation cost only. Physical degradation remains in the investor's
profit and is assessed on realised throughput.
Quantity-strategic convergence includes the raw hourly MW-bid residual set by
`--quantity-bid-tolerance-mw`. These runs write
`final_strategic_quantities.csv` and use format-v4 checkpoints.
Combined price-quantity runs additionally require the price residual set by
`--bid-tolerance`, append both price columns to the same CSV, and use format-v5
checkpoints. Their ISO objective is generation cost plus discharge offer cost
minus charging willingness-to-pay. Settlement remains realized dispatch at
the nodal LMP.

Relaxed-KKT runs export the maximum complementarity product, its numerical
bound violation, and the primal-dual objective gap for every best response.
The formulation is an epsilon approximation and must not be described as an
exact KKT solve.

In `strategic-operation`, a charging bid is the maximum willingness to pay and
enters the ISO objective with a negative sign; a discharge offer enters with a
positive sign. Both are bounded by `--bid-price-bound`. Both initialize at zero
by default and can be changed with `--initial-bid-charge` and
`--initial-offer-discharge`. The restriction
`offer_discharge >= bid_charge / eta^2` excludes negative-cost same-hour
charge/discharge loops.

There are no bid-quantity variables. The ISO may dispatch charging and
discharging up to the investor's full installed MW in every node-hour. The
investor is settled at LMP and pays physical degradation; submitted prices are
not pay-as-bid settlement prices. Strategic runs write
`final_strategic_bids.csv`, and convergence also requires the raw bid-price
residual to satisfy `--bid-tolerance`.

`strategic-price-relaxed-kkt` uses the same price convention and LMP
settlement, but replaces strong duality by relaxed KKT. It contains no bid-MW
variables. The ISO can use the complete inverter in either direction through
`P_charge + P_discharge <= X_power`. SOC transition, energy capacity, and
periodicity remain lower-level physical dispatch constraints. Its moving L1
penalty covers capacities and scaled price changes.

Use `python model/run_model.py --help` for every numerical flag. The default
baseline has no proximal regularizer; `--proximal-penalty` enables the moving
L1 best-response regularizer. In `strategic-price-quantity`, it covers
capacity, hourly quantities, and scaled hourly prices; set
`--proximal-penalty 0.01` for the proposed stabilization stage and adjust the
price normalization with `--proximal-price-scale` if needed.
The default Jacobi damping factor is `0.25`.
Ipopt formulations use HSL MA57 by default; select MUMPS explicitly with
`--ipopt-linear-solver mumps` for a solver comparison.
For strategic-quantity sweeps after the first, the NLP starts from the active
investor's preceding damped capacity and hourly quantities. A fixed-bid HiGHS
clearing supplies an exact lower-level primal/dual KKT point; quantities used
only as initial values are clipped to the SOC trajectory's one-hour
deliverability limits before Ipopt starts. This is a warm start only and does
not alter any MPEC constraint, objective term, or final tolerance. Disable the
LP/KKT initialization with `--no-warm-start` when comparing starts.

`--parallel-workers 4` solves the four best responses concurrently in separate
processes. All four still use the same frozen capacity snapshot, so this changes
runtime only—not the Gauss-Jacobi update rule. Four workers is the default;
select one explicitly for sequential execution.

Each run writes a configuration, compact history, checkpoint, final capacities,
and summary under `model/output/`; strategic runs additionally write final
two-sided bids. Use `--no-output` for a smoke test. Checkpoints retain the full
Jacobi state, convergence streak, projection count, and compact history. On
resume, `--max-sweeps` is the total target sweep number. The runner rejects
changed game settings or input data, while solver limits and the number of
parallel workers may be changed.
To use a completed stage as the initial state for a different L1 proximal
penalty, pass `--allow-proximal-penalty-change`. Only the penalty itself may
change, and the convergence streak is reset. Use a new `--output-dir` to retain
the zero-penalty stage as a separate result.
Use `--run-to-max-sweeps` when a diagnostic continuation should complete its
full requested sweep horizon even if the formal convergence rule is met early.

`--investor-config` accepts a JSON investor population without changing the
market-clearing or MPEC equations. The supplied
`input/investors_merchant_wind_pv.json` sensitivity has three 8% WACC investors:
one merchant, one 100% wind owner, and one 100% PV owner. Use
`--parallel-workers 3` so their three best responses solve concurrently.
`input/investors_original_without_i2.json` instead retains the original I1,
I3, and I4 definitions and removes only the 12% WACC merchant I2.
