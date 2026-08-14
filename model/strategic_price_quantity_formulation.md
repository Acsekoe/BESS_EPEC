# One-investor strategic price–quantity MPEC

This is the mathematical formulation implemented by
**mpec_strategic_quantity_relaxed_kkt.py** when **strategic_prices=True**. In
one best response, investor \(i\) chooses capacity and two hourly
quantity–price pairs, while every rival's capacity and bids are fixed at the
current Jacobi iterate.

## Upper-level strategy and objective

For every node \(n\) and hour \(t\), investor \(i\) chooses

\[
(X^P_{in},X^E_{in}),\qquad
(Q^{ch}_{int},b^{ch}_{int}),\qquad
(Q^{dis}_{int},o^{dis}_{int}),
\]

where \(Q^{ch}\) and \(Q^{dis}\) are maximum available MW, \(b^{ch}\) is the
maximum willingness to pay for charging, and \(o^{dis}\) is the minimum
discharge offer. The investor maximizes

\[
\begin{aligned}
\Pi_i={}&
\sum_{n,t}\lambda_{nt}(d_{int}-c_{int})
+\sum_{g,t}\alpha_{ig}(\lambda_{n(g)t}-C_g)g_{gt}\\
&-\frac{C_i^{deg}}{2}\sum_{n,t}(c_{int}+d_{int})
-\mathrm{CRF}^{day}_i\sum_n(C_i^P X^P_{in}+C_i^E X^E_{in})
-R_i^k .
\end{aligned}
\]

Thus, storage is settled at the nodal LMP on realized dispatch. Submitted
prices influence ISO dispatch but are not pay-as-bid revenues or costs.

Capacity and bid admissibility are

\[
\begin{gathered}
0\le X^P_{in}\le \bar X_n^P-\sum_{j\ne i}X^P_{jn},\qquad
r_i^{min}X^P_{in}\le X^E_{in}\le r_i^{max}X^P_{in},\\
0\le Q^{ch}_{int}\le X^P_{in},\qquad
0\le Q^{dis}_{int}\le X^P_{in},\\
\eta Q^{ch}_{int}\le X^E_{in}-s_{in,t-1},\qquad
Q^{dis}_{int}/\eta\le s_{in,t-1},\\
-\bar b\le b^{ch}_{int},o^{dis}_{int}\le\bar b,\qquad
o^{dis}_{int}\ge b^{ch}_{int}/\eta^2.
\end{gathered}
\]

The last inequality prevents a negative-cost same-hour charge/discharge loop.
The two SOC inequalities require the complete one-hour quantity bids to be
deliverable from the anticipated beginning-of-hour SOC.

## ISO lower-level problem

For all investors \(j\), including the active investor, let \(Q,b,o,X\) denote
the active decisions or fixed rival values. The ISO solves

\[
\min_{g,c,d,s,y}\quad
\sum_{g,t}C_g g_{gt}
+\sum_{j,n,t}\left(o^{dis}_{jnt}d_{jnt}-b^{ch}_{jnt}c_{jnt}\right)
\]

subject to

\[
\begin{aligned}
&\sum_{g\in G_n}g_{gt}+\sum_j(d_{jnt}-c_{jnt})-D_{nt}=y_{nt},
&&\forall n,t,\\
&\sum_n y_{nt}=0,
&&\forall t,\\
&-F_\ell\le\sum_n H_{\ell n}y_{nt}\le F_\ell,
&&\forall \ell,t,\\
&0\le g_{gt}\le\bar G_{gt},
&&\forall g,t,\\
&0\le c_{jnt}\le Q^{ch}_{jnt},\quad
0\le d_{jnt}\le Q^{dis}_{jnt},
&&\forall j,n,t,\\
&c_{jnt}+d_{jnt}\le X^P_{jn},
&&\forall j,n,t,\\
&s_{jnt}=s_{jn,t-1}+\eta c_{jnt}-d_{jnt}/\eta,
&&\forall j,n,t,\\
&0\le s_{jn\tau}\le X^E_{jn},
&&\forall j,n,\tau,\\
&s_{jn0}=s_{jnT},
&&\forall j,n.
\end{aligned}
\]

The shared-inverter constraint prevents charging and discharging from each
using the full installed MW simultaneously.

## Relaxed KKT embedding

The code uses duals \(\lambda,\lambda^{sys},\gamma,\pi\) free,
\(\nu,\mu^+,\rho,\sigma,\kappa,\delta\le0\), and \(\mu^-\ge0\). With the
implemented sign convention, the important reduced costs are

\[
\begin{aligned}
r^g_{gt}&=C_g-\sum_{n\in N(g)}\lambda_{nt}-\nu_{gt}\ge0,\\
r^{ch}_{jnt}&=\lambda_{nt}-\rho_{jnt}-\kappa_{jnt}
+\eta\gamma_{jnt}-b^{ch}_{jnt}\ge0,\\
r^{dis}_{jnt}&=o^{dis}_{jnt}-\lambda_{nt}-\sigma_{jnt}-\kappa_{jnt}
-\gamma_{jnt}/\eta\ge0.
\end{aligned}
\]

Net-injection stationarity is

\[
-\lambda_{nt}+\lambda_t^{sys}
+\sum_\ell H_{\ell n}(\mu^+_{\ell t}+\mu^-_{\ell t})=0,
\]

and SOC stationarity is the corresponding intertemporal condition containing
\(\delta,\gamma\), and the periodicity dual \(\pi\). Instead of exact
complementarity, every nonnegative primal-slack/dual or
variable/reduced-cost pair satisfies the Scholtes relaxation

\[
0\le z_m w_m\le\varepsilon.
\]

The products cover generator lower and upper bounds, charging and discharging
lower and bid bounds, the shared inverter, SOC lower and upper bounds, and both
line limits. Primal and dual feasibility plus these products form a smooth,
nonconvex approximate MPEC; no strong-duality equality is imposed.

## Moving L1 proximal term

At Jacobi sweep \(k\), the same moving L1 capacity penalty already used by the
model is extended directly to quantities and prices:

\[
\begin{aligned}
R_i^k=\rho\Bigg[&\sum_n\left(
|X^P_{in}-X^{P,k}_{in}|+
\frac{|X^E_{in}-X^{E,k}_{in}|}{h_E}\right)\\
&+\frac{1}{2|T|}\sum_{n,t}\left(
|Q^{ch}_{int}-Q^{ch,k}_{int}|+
|Q^{dis}_{int}-Q^{dis,k}_{int}|\\
&\hspace{39mm}+
\frac{|b^{ch}_{int}-b^{ch,k}_{int}|+
|o^{dis}_{int}-o^{dis,k}_{int}|}{h_b}
\right)\Bigg].
\end{aligned}
\]

The maintained defaults are \(\rho=0\), \(h_E=2\) hours, and
\(h_b=10\) EUR/MWh. The intended stabilization run sets \(\rho=0.01\).
The absolute values are implemented exactly with positive/negative deviation
variables. Setting all prices to zero recovers the existing quantity-only
MPEC.
