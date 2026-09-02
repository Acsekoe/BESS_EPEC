# Mathematical formulation

The toy isolates operational market power. Installed inverter power `K_i` and
energy `E_i` are fixed. By default investor `i` strategically submits hourly
discharging availability ceilings while charging availability remains at
installed power. The optional two-sided mode makes both ceilings strategic:

\[
0\le \bar p^{ch}_{it}\le K_i,\qquad
0\le \bar p^{dis}_{it}\le K_i.
\]

These are availability offers, not dispatch schedules. For a fixed offer
profile, the ISO solves

\[
\min \sum_{g,t}\left(c_g p_{gt}+\frac12a_gp_{gt}^2\right)
+\frac12\sum_{i,t}\left[d_i(p^{ch}_{it}+p^{dis}_{it})
+\gamma_i((p^{ch}_{it})^2+(p^{dis}_{it})^2)\right]
\]

subject to nodal and system balance, PTDF line limits, generator capacity,

\[
p^{ch}_{it}\le\bar p^{ch}_{it},\qquad
p^{dis}_{it}\le\bar p^{dis}_{it},
\]

\[
p^{ch}_{it}+p^{dis}_{it}\le K_i,
\]

and the cyclic storage equations

\[
e_{it}=e_{i,t-1}+\eta p^{ch}_{it}-p^{dis}_{it}/\eta,
\qquad 0\le e_{it}\le E_i,
\qquad e_{i0}=e_{iT}.
\]

Investor `i` maximizes uniform-price profit

\[
\Pi_i=\sum_t\lambda_{n(i)t}(p^{dis}_{it}-p^{ch}_{it})
+\sum_{g,t}\theta_{ig}(\lambda_{n(g)t}-c_g)p_{gt}
-\frac12d_i\sum_t(p^{ch}_{it}+p^{dis}_{it})
-\frac12\gamma_i\sum_t((p^{ch}_{it})^2+(p^{dis}_{it})^2)
-C_i^{fixed}.
\]

The ISO LP is embedded using primal feasibility, dual
feasibility/stationarity, and Scholtes products

\[
0\le uv\le\epsilon
\]

for every lower-level complementarity pair. The resulting best response is a
nonconvex NLP solved by IPOPT.

The small positive quadratic slopes are part of the toy data. They make the
dispatch response smoother and reduce LP dual-price ambiguity. Every MPEC
candidate is nevertheless independently recleared, and convergence also
requires embedded versus recleared LMP and profit agreement.

To make the LMP single-valued at merit-order kinks, each node also has a tiny
two-sided demand adjustment `z_nt` with

\[
\frac12(500)z_{nt}^2
\]

in the ISO objective. Its stationarity condition is

\[
500z_{nt}=\lambda_{nt}.
\]

At ordinary prices this changes demand by only about 0.05--0.25 MW. It is an
explicit price-selection regularization, not unreported load shedding.

Availability above realised dispatch can otherwise be non-identifiable. The
implemented objective therefore adds the documented lexicographic term

\[
10^{-3}\sum_t(\bar p^{ch}_{it}+\bar p^{dis}_{it}).
\]

This is at most EUR 0.06/day for a 10 MW battery over six hours and selects
full availability among economically equivalent offers. Reported profit
excludes this tie-breaker.

The equilibrium driver uses damped Gauss-Seidel updates. After every complete
sweep, all four best responses are solved again against the same final profile.
Convergence is declared only from this simultaneous Nash audit, not from the
within-sweep update size.

## One-stage joint investment and availability variant

The separate `run_joint_investment.py` experiment makes installed power and
energy continuous upper-level variables in the same game as availability:

\[
0\le K_i\le30\text{ MW},\qquad 2K_i\le E_i\le8K_i.
\]

The active investor simultaneously selects \(K_i\), \(E_i\), and its hourly
discharge availability. Charging availability equals \(K_i\) in the default
mode. The ISO constraints become

\[
p^{ch}_{it}+p^{dis}_{it}\le K_i,\qquad e_{it}\le E_i,
\qquad \bar p^{dis}_{it}\le K_i.
\]

Daily annualised capacity cost is no longer constant in the best response:

\[
C_i^{inv}=\frac{\operatorname{CRF}(w_i,L_i)}{365.25}
\left(c_i^K K_i+c_i^E E_i\right).
\]

This formulation intentionally confounds investment and operational
withholding. It is useful as a diagnostic comparison, but it does not
represent capacity being committed before the operational game.
