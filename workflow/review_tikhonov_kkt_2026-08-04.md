# Mathematical and numerical review: `model/tikhonov_kkt/`

Date: 2026-08-04. Scope: primal_llp.py, dual_llp.py, dual_tikhonov_llp.py,
kkt_formulation.py, mpec_relaxed_kkt.py, jacobi_epec.py, common.py, plus the
reused parent modules single_investor_mpec.py, epec_diagonalization.py,
epec_results.py.

Verification performed: line-by-line independent dual derivation; numerical
primal/dual cross-check; unlocked Tikhonov dual solve; full single-investor
relaxed-KKT MPEC solve with diagnostics; direct demand-perturbation LMP audit.

---

## Verdict

**Conditionally valid as a regularization/homotopy experiment; experimental
only for any equilibrium claim; not yet valid for reporting results.**

- The embedded primal LLP and explicit dual are **algebraically correct**
  (independently re-derived, every stationarity condition and dual-objective
  term matches; numerically: primal 379,084.571 vs dual 379,084.580 EUR/day,
  relative gap 2.3e-8).
- The Tikhonov construction and Scholtes relaxation are internally consistent
  and mean exactly what the README claims.
- However, finite-(γ, ε) solutions are optimistic selections in a **physically
  imbalanced approximate market with an unfunded phantom-energy subsidy of
  γ·Σλ² ≈ 662 EUR/day** (measured), the Jacobi convergence test can certify
  points that are far from any equilibrium, and there is no exact-market
  best-response verification.

---

## 1. Primal–dual derivation — correct, with caveats

Independent derivation (min problem, Lagrangian convention `L = C − Σ λ·h`,
h = gen + storage_net + shed − demand − NetInj):

| Variable | Derived condition | Implemented | Match |
|---|---|---|---|
| P_gen ≥ 0 | Σ_n λ + ν ≤ c_g (+ reg·P_gen) | gen_stationarity | ✓ |
| P_charge ≥ 0 | −λ + ρ_ch − ηγ_soc ≤ d/2 (+reg·P_ch) | charge_stationarity | ✓ |
| P_discharge ≥ 0 | λ + σ + γ_soc/η ≤ d/2 (+reg·P_dis) | discharge_stationarity | ✓ |
| NetInjection free | −λ + λ_sys + PTDF'(μ_up+μ_dn) = 0 (+reg·NetInj) | netinjection_stationarity | ✓ |
| SOC[τ] ≥ 0 | δ + γ_τ − γ_{τ+1} + ρ_per·1{τ=0} − ρ_per·1{τ=last} ≤ 0 (+reg·SOC) | soc_stationarity | ✓ |

Dual objective `Σ dλ + Σ cap·ν + Σ F(μ_up − μ_dn) + Σ X(ρ_ch+σ) + Σ E·δ` is
the correct `b'y + h'μ` for the stated sign conventions (ν, μ_up, ρ, σ, δ ≤ 0;
μ_dn ≥ 0; λ, λ_sys, γ_soc, ρ_per free). Zero-RHS equalities (system balance,
SOC transition, periodicity) correctly contribute nothing. Cyclic terminal SOC
is handled correctly by ρ_per entering SOC[0] with +1 and SOC[last] with −1.
Nothing is missing or double-counted. λ is the economic LMP (∂cost/∂demand).

Caveats:

1. **Box bounds truncate the dual polyhedron** (λ ∈ [−500,500], others
   ±10,000). If any bound is active at a solution, strong duality / the KKT
   system is distorted or infeasible. Must be verified inactive at every
   accepted solution (currently not gated).
2. **Sparse-balance dual-orientation trap (found, verified).** In
   `primal_llp.py` storage terms are built sparsely, so at nodes with no
   generator and no storage (N4, N5, N7, N9) the nodal balance degenerates to
   `constant == NetInjection`; Pyomo normalizes it with opposite body
   orientation and the imported solver dual flips sign *per node*. Measured:
   raw duals at hour 1 were `[60,60,60,−60,−60,60,−60,60,−60]` while direct
   demand perturbation proves the true LMP is **+60 at both N1 and N4** (and
   +35.30 at N9 h12 where raw = −35.299). The single global sign flip in
   `fixed_demand_reference_lambda` cannot repair a per-node flip. The
   maintained settlement path (`build_primal_market_clearing_model`, dense
   I×N storage) is *not* affected; the trap bites this package's demo output
   and any future consumer of raw duals from sparse-balance primals. Fix:
   write the balance as `expr − NetInjection == 0` (variable on the body side
   for every node), or take prices from the explicit dual model.
3. Elimination of zero-capacity generator-hours and zero-capacity storage
   nodes is legitimate (variables fixed at 0 removed with their conditions).

## 2. Tikhonov construction

**The implemented equation is correct.** For `max D(y) − (γ/2)Σλ²` over the
dual-feasible set, stationarity in λ under this sign convention gives exactly
`h[n,t] + γλ[n,t] = 0`. Verified numerically: max |h| = 0.0600 MW = γ·λ_max =
1e-3·60.

**What it is.** The correct classification is: *quadratic-penalty
(Moreau–Yosida) smoothing of the primal nodal balance*, equivalently *Tikhonov
regularization of the dual restricted to the λ-block*, used as a **penalty
homotopy**. It is not a "harmless dual selection" at finite γ — the README
already says this correctly. Economic reading: at every node-hour a virtual
price-elastic agent with inverse supply curve p = q/γ (slope 1/γ = 1000
EUR/MWh/MW at γ=1e-3) absorbs/supplies the residual. This is structurally the
same trick as the project's quadratic demand curve, but two-sided,
zero-intercept, and always active.

**What finite γ solves (exact result).** With ε = 0 the KKT system is exactly
the KKT of the convex QP

    min_x  C(x) + (1/2γ)·Σ_{n,t} h(x)²   s.t. all other primal constraints,

whose solution has **unique h and hence unique λ = −h/γ** (strict convexity of
the penalty in h; a two-minimizer midpoint argument rules out multiplicity).
So at ε=0, finite γ *removes* nodal-price multiplicity entirely — λ becomes a
function of the fleet. Other duals (ν, μ, γ_soc, δ) and the primal dispatch
split can remain nonunique.

**γ_k → 0 (exact, LLP level).** Classical quadratic-penalty convergence for a
feasible bounded LP: penalized minimizers converge to the exact primal optimal
set, C(x_γ) → C*, and the multiplier estimate λ_γ = −h_γ/γ converges to the
**minimum-Euclidean-norm λ among dual-optimal solutions** (in the λ-block).
Required assumptions: LP feasibility and boundedness (hold here), nonempty
dual (LP duality), and the artificial box bounds slack along the trajectory.

**Minimum-norm selection.**
- Unlocked, finite γ: λ_γ is *not* dual-optimal at all; min-norm only in the
  limit γ→0.
- Locked (`D = primal optimum`): exact min-norm λ on the face for **any**
  γ > 0 — γ is irrelevant under the lock (the objective is constant on the
  locked face, so only the penalty matters). The README's description is
  correct; note the γ-independence.
- Inside the implemented MPEC (ε = 1e-3 > 0): the guarantee is lost. The
  ε-inflation re-opens leader price freedom of order δλ ≲ ε/(γ·r), where r is
  the reduced cost of the cheapest dispatch adjustment used to manufacture
  imbalance δh = −γ·δλ. At ε = γ = 1e-3 this permits tens of EUR/MWh at
  near-degenerate nodes. Selection is again optimistic.

**Phantom energy (interpretation risk, quantified).** Hourly system imbalance
Σ_n h = −γΣ_n λ (measured max 0.54 MW/h; 11.39 MWh/day total absolute nodal
residual). Its market value γΣλ² ≈ 662 EUR/day is paid by no one — an
unfunded subsidy comparable to several percent of merchant profit. Finite-γ
"profit" is an objective in that subsidized market.

## 3. Relaxed complementarity

**Pair inventory: complete and correct.** All ten product families (gen
lower/upper, line upper/lower, charge lower/upper, discharge lower/upper, SOC
lower/upper; plus shed lower/upper in demand-curve mode) match the dual
feasibility inequalities exactly; slack ≥ 0 comes from primal feasibility and
multiplier-sign ≥ 0 from the dual variable bounds, so `product ≤ ε` with the
existing constraints is a valid Scholtes NLP relaxation; ε=0 recovers the
exact MPCC. Equality constraints correctly carry no pairs.

**Theory available.** Scholtes (2001): limit points of stationary points of
the relaxed NLPs as ε→0 are C-stationary for the MPCC; M-/B-stationarity
needs extra conditions (MPEC-LICQ/MFCQ, second-order). Combined with the γ
homotopy this remains plausible but is *not* covered by an off-the-shelf
theorem; state it as an assumption.

**Scaling.** With Δt = 1 h all products are commensurable in EUR, so the
README's "different physical units" is slightly off — the real problem is
**scale heterogeneity**: slacks span 0.01–300 MW and duals 0–500 EUR/MWh, so
one ε = 1e-3 forces reduced costs below 5e-6 EUR/MWh on 200-MW variables while
tolerating 0.1 EUR/MWh mispricing on 0.01-MW slacks. Tightness varies by 4–5
orders of magnitude. Recommended: normalized products
`(s/s_ref)·(μ/μ_ref) ≤ ε_rel` with s_ref = per-family capacity scale, μ_ref =
50 EUR/MWh, or at least per-family ε.

**Solver-tolerance interaction (measured).** tol = 1e-4 vs ε = 1e-3 is only
one order of magnitude; the solved MPEC exhibits `min product = −8.5e-5`,
i.e. feasibility noise at ~10 % of ε. Require **tol ≤ ε/100** and gate
`min product ≥ −10·tol`.

**Central-path variant.** `(s + shift)·μ = ε` is a consistent smoothed
central-path *diagnostic*; with the shift it is not the exact log-barrier
path, and it forces strictly positive multipliers on every constraint,
injecting a systematic O(ε/slack) dual bias. Keep it labeled as a diagnostic.

**Exact identity (derived and numerically confirmed).** At any point that is
feasible for the relaxed system,

    C(x) − D(y) = Σ_{n,t} λ·h + Σ products = −γ·Σλ² + Σ products.

Observed: −661.64 ≈ −662.5 + 0.87. Consequences: (i) the signed primal-dual
gap has an *expected offset* of −γΣλ² and must not be compared to 0; (ii) the
proper KKT-consistency residual is `|gap + γΣλ² − Σproducts|`; (iii) at γ=0
the gap equals the product sum, so bounding products by ε bounds the duality
gap by N·ε (N ≈ 2000 here → 2 EUR/day; fine).

**Continuation relationship.** From δλ ≲ ε/(γ·r): ε must shrink *faster* than
γ. Recommend ε_k = c·γ_k^{1+θ} with θ ∈ [0.5, 1], tol_k ≤ ε_k/100, e.g.
(γ, ε, tol): (1e-2, 1e-3, 1e-5) → (1e-3, 1e-4, 1e-6) → (3e-4, 1e-5, 1e-7).

## 4. Single-investor MPEC

**Bookkeeping verified correct**: spot revenue Σ λ(P_dis − P_ch) of the
investor's units; portfolio rent share·(λ_g − c_g)·P_gen at the generator's
node; degradation 0.5·d·throughput consistent with the ISO objective on both
sides; CAPEX via CRF(wacc, 15y)/365.25; private headroom
X_i,n ≤ K_n − Σ_j≠i X_j,n with sparse-dropped rival blocks still consuming
headroom; 2–8 h envelope on both sides. Projection preserves E/P, so
projected points remain envelope-feasible.

**Exploitation channels (real, bounded, must be measured):**
1. ε-inflation: classic optimistic dual selection over an inflated KKT set.
2. γ-imbalance: δλ = −δh/γ. At γ=1e-3, a 0.05 MW manufactured nodal residual
   moves the local price 50 EUR/MWh; the complementarity products price this
   channel at δλ ≲ ε/(γ·r). Nodes with an interior marginal generator are
   pinned (the ε-inflated version of "the marginal unit sets the price");
   congested/storage-only nodes are exposed.
3. Dispatch attribution: own and rival storage at one node with equal
   degradation cost are interchangeable in the ISO problem; the optimistic
   MPEC claims the profitable share. γ does **not** remove this (it fixes h,
   not the dispatch split). The settlement's capacity-proportional
   `alt_profit` is the right counterweight — keep reporting both.

**Reported "profit" is the objective in the γ-penalized, ε-relaxed,
phantom-subsidized market under an optimistic selection.** It is not
realizable profit in the original market and the two already differ materially
in this project's strong-duality experience.

**Empirical flag.** The γ=1e-3, ε=1e-3 solve installs 187.2 MW / 547.2 MWh
(profit 18,820 EUR/day) where the planner smoke test at the same calibration
and 200-MW limit installs ~119 MW. A monopolist investing ~57 % more than the
planner is atypical and must be audited in the exact market before any
interpretation.

## 5. Four-investor Jacobi EPEC

- The shared nodal limit **is a GNEP coupling constraint**. The
  private-headroom representation is the correct per-player section of it
  when rivals are frozen: for best responses the two are equivalent. It
  corresponds to *player-specific multipliers* — the full GNEP equilibrium
  manifold — not the variational/normalized equilibrium with a common
  multiplier. The exported investor-specific shadow prices are consistent
  with that reading (WTP diagnostics, not access prices).
- A **clean fixed point** of the *undamped, unprojected* map (all four solves
  optimal, proposals = state, no projection) is a GNE **of the
  finite-(γ, ε) optimistic game** — with the caveat that each investor
  optimizes against its own selected prices, so it is an equilibrium of the
  diagonalization game, not of a single common market, unless the lower-level
  solution is unique (which ε > 0 breaks).
- **Proportional projection does not preserve best-response optimality.** A
  state kept stationary by persistent projection is a projected fixed point,
  not a GNE. Damping alone maps fixed points to fixed points, but the damped
  2 % criterion allows an undamped best-response residual up to
  tol/damping = 0.02/0.25 = **8 %**; with projection the undamped residual is
  unbounded by the criterion. False convergence is possible and must be
  excluded by construction (see protocol).
- **No convergence theory applies** to this map (nonconvex MPEC best
  responses, nonuniqueness, coupling constraint). Jacobi best-response
  cycling is already documented in this project. Treat the loop as a
  fixed-point heuristic with a posteriori verification.
- More defensible alternatives, in order of cost: (1) keep diagonalization,
  fix the acceptance tests (cheapest, recommended now); (2) proximal
  regularized Gauss–Seidel (already prototyped in the strategic driver);
  (3) monolithic stacked-KKT "EPEC stationarity" NLP with shared Scholtes
  relaxation; (4) PATH/MCP on a variational-equilibrium reformulation —
  changes the solution concept to a common-multiplier equilibrium and must be
  labeled as such.

## 6. Convergence criterion — inadequate as implemented

Problems: damped post-projection capacities only; 2 % relative with 1 MW/2 MWh
floors; single sweep; no complementarity/gap/imbalance/undamped gating;
`REQUIRE_EACH_STAGE_TO_CONVERGE = False` means the "continuation" default
never enforces stage convergence; default `REGULARIZATION_STAGES` is a single
stage, so no continuation actually runs.

### Proposed acceptance protocol

**(a) Approximate best response** (investor i, frozen rivals, given γ, ε):
- Ipopt optimal at tol ≤ min(1e-6, ε/100);
- max product ≤ ε + 10·tol and min product ≥ −10·tol;
- |h + γλ| ≤ 10·tol at every node-hour;
- KKT-consistency: |(C − D) + γΣλ² − Σproducts| ≤ max(1 EUR/day, 1e-4·|D|);
- no λ or dual within (say) 1 % of its box bound;
- headroom and envelope feasible.

**(b) Converged (γ, ε) stage:**
- all four responses pass (a);
- **undamped, pre-projection** residual max_n |BR_i,n − x_i,n| ≤ 0.5 MW and
  ≤ 1 % relative (2 MWh / 1 % for energy);
- zero projection events in the passing sweep;
- 2–3 consecutive passing sweeps (Jacobi period-2 cycling guard);
- optimistic profits stationary to 0.1 %/sweep.

**(c) Successful continuation:** every stage passes (b) (set
REQUIRE_EACH_STAGE_TO_CONVERGE = True); capacities Cauchy across stages
(‖x*(k+1) − x*(k)‖ decreasing); γΣλ², Σproducts, and total |h| all
monotonically → 0; final-stage capacities within the stage-(b) tolerance of
the previous stage's.

**(d) Approximate equilibrium of the original market:** fix final capacities;
exact joint re-clear; for each investor solve the exact strong-duality MPEC
(γ = ε = 0) against frozen rivals and report the deviation gap
π_dev − π_current under (i) optimistic prices and (ii) joint-settlement
prices; declare a "δ-equilibrium under the optimistic convention" only if the
gap ≤ δ (e.g. 1 % of daily CAPEX) and report the dual-face profit interval per
the thesis plan. Without (d), no equilibrium claim about the original market
is admissible.

## 7. Final exact re-clear

As implemented it is **an audit, not a validation**: it establishes physical
feasibility and settled cash flows at the final capacities, nothing about
optimality or equilibrium. Disagreement between regularized MPEC profit and
settled profit does not by itself invalidate the fixed point (they are
different objects), but it bounds the economic relevance of the optimistic
numbers and must be reported side by side. The missing piece is exactly
protocol step (d): exact-market unilateral best responses in capacities (the
full MPEC — not merely re-dispatch), one per investor.

---

## Ranked findings

**Critical**
- C1. Finite-(γ,ε) profits are objectives in a phantom-subsidized approximate
  market (subsidy γΣλ² ≈ 662 EUR/day measured); reporting them as profits or
  the fixed point as an equilibrium would be wrong. (Interpretation, not
  algebra.)
- C2. Convergence certification can pass non-equilibria: damped
  post-projection 2 % ⇒ up to 8 % undamped residual, unbounded with active
  projection; single sweep; no KKT gating.
- C3. No exact-market best-response verification exists; the re-clear does
  not provide it.

**High**
- H1. ε/tol separation only 10×; measured negative products at 10 % of ε.
- H2. Single shared ε across 4–5 orders of magnitude of product scales.
- H3. ε re-opens the price freedom that γ closes (δλ ≲ ε/(γr)); the
  implemented ε = γ = 1e-3 is in the regime where manipulation is cheap; the
  γ–ε coupling is undocumented and unmanaged.
- H4. Dual/price box bounds may truncate the dual set; inactivity not gated.

**Medium**
- M1. Signed primal-dual gap lacks its expected offset −γΣλ² in diagnostics
  (identity above); should be exported as a consistency residual.
- M2. Sparse-balance dual-orientation flip (constant == variable) makes raw
  solver duals per-node sign-inconsistent in `primal_llp.py`; global
  sign-flip heuristics cannot repair it. Maintained dense settlement is
  unaffected; fix the constraint orientation anyway.
- M3. Default config runs a single stage — no actual continuation; and
  REQUIRE_EACH_STAGE_TO_CONVERGE=False weakens multi-stage claims.
- M4. Central-path variant biases all duals positive; diagnostic only.
- M5. Damping 0.25 / tol 2 % differ from the project's stated maintained
  preferences (0.7 / 1 %); fine for an experiment, but flag in run configs.

**Low**
- L1. Global sign-flip heuristic in `fixed_demand_reference_lambda` is
  fragile when average prices ≈ 0 (dense case).
- L2. `initial_state` seed and envelope checks are fine; no issue found.

## Minimal code changes before results can be reported

1. `jacobi_epec.py`: gate convergence on undamped pre-projection residuals,
   zero projection events, and ≥ 2 consecutive passing sweeps; log the gate.
2. `kkt_formulation.relaxed_kkt_diagnostics`: export Σλ², γΣλ² (phantom
   value), and the KKT-consistency residual `gap + γΣλ² − Σproducts`.
3. Enforce tol ≤ ε/100 in configs; add a min-product ≥ −10·tol check.
4. Add a post-run exact best-response verification pass (strong-duality MPEC
   per investor at final capacities) and export deviation gaps.
5. `primal_llp.py`: write nodal balance as `expr − NetInjection == 0` (or
   drop the raw-dual printout); never use raw solver duals of sparse-balance
   models as prices.
6. Configure a real continuation (≥ 3 stages, ε_k = γ_k^{1.5}-ish) with
   REQUIRE_EACH_STAGE_TO_CONVERGE = True for non-final stages.
7. Add a bound-activity check (λ and duals vs boxes) to diagnostics.

## Falsification / support experiments

1. **LLP γ-sweep vs locked min-norm**: γ ∈ {1e-1…1e-5}; expect
   ‖λ_γ − λ_locked‖ = O(γ) and D* − D(y_γ) = O(γ). Falsifies/validates the
   homotopy story at the market level.
2. **Unique-price audit at fixed capacities**: solve the soft-balance QP
   (penalized primal) at the MPEC's final X; its λ is unique. Any gap to the
   MPEC's λ measures leader price manipulation through ε. This is the single
   most informative cheap test.
3. **ε-sweep of MPEC profit at fixed γ**: steeply increasing profit in ε ⇒
   the leader is exploiting the relaxation, not the market.
4. **Continuation Cauchy test** on capacities and residuals (protocol (c)).
5. **Exact-market deviation test** (protocol (d)); also settles whether the
   187 MW single-investor result survives exact settlement.
6. **Update-rule/damping/order sensitivity** at fixed (γ, ε): path dependence
   of the fixed point = equilibrium-selection evidence.
7. **Planner comparison** at identical calibration and limit.

## Safe thesis terminology

Safe: "quadratically penalized (soft-balance) market approximation";
"Tikhonov-regularized dual price selection, exact only in the γ→0 limit";
"Scholtes-relaxed KKT reformulation"; "damped, projected simultaneous
best-response iteration"; "fixed-point candidate under the optimistic
convention"; "C-stationary candidate (under stated CQs)"; "regularized-market
objective value" (for MPEC profit).

Not safe (without the corresponding verification): "equilibrium" for the
Jacobi output; "market-clearing prices" for finite-γ λ; "profit" unqualified;
"the homotopy converges"; "minimum-norm prices" for the unlocked finite-γ
formulation; any statement implying the lower level is a cleared market at
finite γ.

## Epistemic classification

- **Exact mathematical results**: dual correctness; equivalence of the
  unlocked Tikhonov dual with the quadratic-penalty primal and
  h + γλ = 0; uniqueness of (h, λ) at ε = 0, γ > 0; the gap identity
  C − D = −γΣλ² + Σproducts; locked variant = exact min-norm for any γ > 0;
  quadratic-penalty convergence and least-norm multiplier limit for the LLP.
- **Claims requiring assumptions**: C-stationarity of (γ, ε) → 0 limits
  (MPEC-MFCQ etc.); GNE interpretation of clean fixed points; box bounds
  inactive; least-norm limit under the λ-only penalty.
- **Numerical heuristics**: damping, proportional projection, warm starts,
  single shared ε, box bounds, retry-with-shrink, sign-flip price recovery.
- **Empirical observations (this test system)**: primal/dual agreement;
  λ-face near-singleton for the 50/50 MW case; measured 0.06 MW = γλ_max
  residual, 0.54 MW hourly system imbalance, −661.6 EUR/day gap, 0.87 sum of
  products, min product −8.5e-5; 187 MW vs planner ~119 MW anomaly; the
  N4/N5/N7/N9 dual-sign flip.
