# Deterministic spot + aFRR capacity benchmark

Implemented 2026-09-07. `model/mpec.py` is the unchanged energy-only MPEC.
`model/mpec_afrr.py` contains the reserve extension. The original runnable
source, inputs, tests and README were archived in
`backups/energy_only_20260907_153308.zip` before editing. Original mpec SHA-256:
`65386cbfd4cb7b7986573188f83a4e6d4369edf47f3f60a9ba0c8806d76a6a4c`.

## Scope and units

The strategic decisions remain installed nodal BESS MW and MWh. All owners'
operations are reoptimized by the ISO on every deviation. Reserve awards
are lower-level decisions, not strategic offers. This is a continuous,
marginal-priced joint clearing benchmark, not a replication of Austria's
sequential pay-as-bid capacity auction.

Six blocks cover hours 1-4, 5-8, ..., 21-24 (clock time 00:00-04:00, etc.).
Every MW commitment applies throughout its block, H=4 hours. Upward and
downward reserve requirements are each 50 MW in every block. These are
test-system assumptions, not Austrian procurement quantities.

All capacity offers and shortage penalties are **EUR/MW/h**. Conventional
generator availability cost is 25 EUR/MW/h; the shortage penalty is 3000
EUR/MW/h. Renewable offers are recorded as 1000 EUR/MW/h, but renewables
are explicitly excluded by the eligible-generator list, rather than assumed
to be priced out. Conventional units are assumed eligible and fast enough;
no unit commitment, minimum stable output or ramp constraints are added.

The unscaled MW procurement balances have duals Gamma in EUR/MW/**block**.
Reported hourly prices are gamma=Gamma/H. Payment is Gamma*R=H*gamma*R,
never H*Gamma*R. Generator offers are assumed true availability costs, so
owned generators, if enabled for reserves, earn (Gamma-H*offer)*r in
addition to their spot generation rent. A conventional offer is not a hard
price cap: energy opportunity costs can raise the clearing price.

## Primal extension

For direction d in {up,dn}, block b, battery j=(i,n), and eligible generator g:

```
R[j,b,d], r[g,b,d], s[b,d] >= 0
sum_j R[j,b,d] + sum_g r[g,b,d] + s[b,d] >= D[b,d]

Pdis[j,t] + R[j,b(t),up] <= P[j]
Pch[j,t]  + R[j,b(t),dn] <= P[j]
Pg[g,t] + r[g,b(t),up] <= Gmax[g,t]
Pg[g,t] - r[g,b(t),dn] >= 0

SOC[j,k] - (tau/eta)*R[j,b,up] >= 0
SOC[j,k] + tau*eta*R[j,b,dn] <= E[j]
```

For each block b, k includes ALL its hourly endpoints including its start
and end. Adjacent blocks both constrain the common boundary. Interior
endpoints are included only once per block. tau=0.5 hours. A generator with
zero availability in any block hour is omitted from that entire block's
reserve set (also consistent with sparse zero-generation indices in the MPEC).

Existing hourly SOC transitions, cyclic SOC, network equations and power/
energy investment constraints remain. eta is the existing **one-way**
efficiency, 0.936; round-trip efficiency is therefore 0.876096. No cost or
efficiency calibration has been silently changed.

Extra daily ISO cost:

```
sum_g,b H[b]*offer[g]*(r_up[g,b]+r_dn[g,b])
+ sum_b H[b]*penalty*(s_up[b]+s_dn[b])
+ sum_j,b 0.5*Cdeg[j]*H[b]*(alpha_up*R_up[j,b]+alpha_dn*R_dn[j,b])
```

alpha_up=alpha_dn=0.10 **per direction**. The final term is charged in both
the ISO objective and investor profit. It is expected wear only: there is
no activation-energy market, expected recharge expenditure, activation SOC
drift, intraday trading, or stochastic dispatch in this version. The
30-minute buffer is a deterministic physical backing proxy, not a
certificate of sustained four-hour activation or network deliverability
under activation. Separate charge/discharge headroom follows the requested
conservative formulation and does not credit reserve from reversing charging.

## Signed KKT derivation

The implementation follows the original convention:

```
L = f - sum_constraints dual*(body-rhs)
<= inequality: dual <= 0
>= inequality: dual >= 0
equality: free dual
nonnegative variable x: reduced_cost=dL/dx >= 0, x*reduced_cost=0
```

Let nu<=0 price generator upper headroom, kappa>=0 price generator downward
headroom, sig<=0 price battery discharge headroom, rho<=0 price charge
headroom, xi>=0 price the upward SOC buffer, zeta<=0 price the downward
SOC buffer. Define a=tau/eta and z=tau*eta. For endpoints K_b of block b:

```
rc(R_up[j,b]) = H*0.5*Cdeg[j]*alpha_up - Gamma_up[b]
               - sum_t_in_b sig[j,t] + a*sum_k_in_Kb xi[j,b,k]
rc(R_dn[j,b]) = H*0.5*Cdeg[j]*alpha_dn - Gamma_dn[b]
               - sum_t_in_b rho[j,t] - z*sum_k_in_Kb zeta[j,b,k]
rc(r_up[g,b]) = H*offer[g] - Gamma_up[b] - sum_t_in_b nu[g,t]
rc(r_dn[g,b]) = H*offer[g] - Gamma_dn[b] + sum_t_in_b kappa[g,t]
rc(s[b,d])    = H*penalty - Gamma[b,d]
```

Existing generation reduced cost acquires `-kappa[g,t]`. Existing SOC
reduced cost acquires `-sum_b_containing_k (xi[j,b,k]+zeta[j,b,k])`.
Charge/discharge reduced costs retain their algebra, with their multipliers
now pricing the modified reserve headroom inequalities.

In addition to every variable-lower-bound product, include:

```
Gamma[b,d]*(sum R + sum r + s - D)
kappa[g,t]*(Pg[g,t]-r_dn[g,b(t)])
xi[j,b,k]*(SOC[j,k]-a*R_up[j,b])
(-zeta[j,b,k])*(E[j]-SOC[j,k]-z*R_dn[j,b])
```

Existing generation/charge/discharge upper complementarity products use
the **new** reserve-reduced slacks. All products are bounded above by
epsilon in Scholtes mode; primal and dual feasibility enforce their
nonnegativity. The shortage reduced cost is enforced by the analytic
bound `0 <= Gamma <= H*penalty` (12000 EUR/MW/block by default).

The original dual objective acquires
`sum_b,d D[b,d]*Gamma[b,d] + sum_j,b,k E[j]*zeta[j,b,k]`.
Original capacity terms remain because reserve headroom uses the same
power/generation upper bounds. Strong duality equates the full extended
primal and dual objectives; it and Scholtes use the same product catalogue.

Physical multipliers and LMPs in the aFRR MPEC retain their required signs
but not the old artificial magnitude caps. Those caps could exclude valid
reserve opportunity costs. Only the economically derived reserve-price cap
is imposed. The energy-only MPEC retains its original bounds unchanged.

## Running and rollback

From the repository root (not from capacity_epec/model):

```powershell
python capacity_epec/model/run_model.py --market afrr --symmetric-initialization --quadratic-cost-power-eur-per-mw2 0 --quadratic-cost-energy-eur-per-mwh2 0 --output-dir capacity_epec/model/output/afrr_linear
python capacity_epec/model/run_model.py --market energy-only --symmetric-initialization --quadratic-cost-power-eur-per-mw2 0 --quadratic-cost-energy-eur-per-mwh2 0 --output-dir capacity_epec/model/output/energy_only_linear
```

`--market energy-only` is the default and selects the untouched mpec.py.
The obsolete `--access-beta` option is not reintroduced. Existing physical
nodal caps remain; `--node-limit-mw inf` disables them consistently.

A zero-demand ablation within the same extension is available through
`--market afrr --afrr-demand-up-mw 0 --afrr-demand-down-mw 0`.
Changing capex curvature to zero leaves the algorithmic proximal term in
iteration; final audits remove it. The ISO demand-adjustment quadratic is
also retained. Thus "linear costs" means strictly linear investment costs.

Extra outputs: final_afrr_blocks.csv (requirements, awards, shortages, both
price units), final_afrr_awards.csv (provider-level awards/payments), and
final_afrr_soc.csv (endpoint buffers and margins). Profit decomposition
separates reserve revenue, expected reserve wear, and owned-generator reserve
surplus. Run configuration records all effective aFRR parameters and overrides.

## Verification and interpretation

Tests independently differentiate the lower-level Lagrangian and compare
all changed reduced costs, verify exact-QP KKT seeds in both embeddings,
verify IPOPT prices and the analytic shortage price,
check zero-energy reserve rejection, and check block/payment scaling.
All pre-existing energy-only regression tests remain applicable.

Fixed-capacity IEEE-9 validation at the symmetric starting profile produced
6698 complementarity products, maximum product 3.37e-12 and primal-dual
gap 6.12e-9 EUR/day with IPOPT seeding (objective scaling disabled). This verifies the embedding at
that point; it does not certify equilibrium or unique reserve allocation.

Reserve prices/allocations can be nonunique under linear offers. Keep the
IPOPT fixed-capacity reclear, embedded/recleared profit comparison, and
zero-proximal final audit. All solves use IPOPT; there is no second-solver
dependency or acceptance gate. Report
failed audits as failed audits. Different capacities after a finite number
of sweeps do not establish an equilibrium investor split.

Market reference: https://markt.apg.at/en/power-grid/balancing/secondary-control/
