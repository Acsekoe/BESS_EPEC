# Co-optimised strategic nodal access formulation

For one best response, investor \(i\) chooses a nodal access request, a
pay-as-bid willingness to pay, and independent energy capacity while rival
strategies are fixed at the current Jacobi iterate:

\[
0\le q^A_{in}\le \bar Q_n^A,\qquad
0\le b^A_{in}\le \bar b^A,\qquad
X^E_{in}\ge0.
\]

The active investor also satisfies the portfolio request limit

\[
\sum_n q^A_{in}\le \bar Q_i^A.
\]

Access bids are measured in EUR/MW-day because the operational model and
annualised investment costs are represented per day.

## Co-optimised ISO lower level

The ISO chooses awarded connection power \(x^A\), generation \(g\), storage
charge/discharge \((c,d)\), state of charge \(s\), and nodal net injections
\(y\):

\[
\min\quad
\sum_{g,t} C_g g_{gt}
+\sum_{j,n,t}\frac{C_j^{deg}}{2}(c_{jnt}+d_{jnt})
-\sum_{j,n} b^A_{jn}x^A_{jn}.
\]

Only connection power is allocated by the ISO:

\[
0\le x^A_{jn}\le q^A_{jn},\qquad
\sum_jx^A_{jn}\le \bar Q_n^A.
\]

The awarded MW is the physical shared-inverter capacity. Energy capacity is
chosen independently by the investor and enters the LLP as fixed MWh capacity:

\[
c_{jnt}+d_{jnt}\le x^A_{jn},
\]

\[
s_{jnt}=s_{jn,t-1}+\eta c_{jnt}-d_{jnt}/\eta,
\]

\[
0\le s_{jn\tau}\le X^E_{jn},\qquad s_{jn0}=s_{jnT}.
\]

The technical duration limits remain upper-level feasibility constraints for
the active investor:

\[
r_i^{min}x^A_{in}\le X^E_{in}\le r_i^{max}x^A_{in}.
\]

The maintained nodal balances, system balance, generator bounds, and PTDF line
limits complete the lower-level LP. Thus, for fixed \((q^A,b^A,X^E)\), the ISO
jointly selects only the access-power allocation and its physically feasible
daily use.

## Investor objective and settlement

The active investor pays its own accepted bid:

\[
A_i=\sum_n b^A_{in}x^A_{in}.
\]

It maximises

\[
\begin{aligned}
\Pi_i={}&
\sum_{n,t}\lambda_{nt}(d_{int}-c_{int})
+\sum_{g,t}\alpha_{ig}(\lambda_{n(g)t}-C_g)g_{gt}\\
&-\frac{C_i^{deg}}{2}\sum_{n,t}(c_{int}+d_{int})
-\mathrm{CRF}^{day}_i\sum_n
  \left(C_i^P x^A_{in}+C_i^E X^E_{in}\right)\\
&-\sum_n b^A_{in}x^A_{in}-R_i^k.
\end{aligned}
\]

The dual of the nodal access constraint is reported as a scarcity value. It is
not the settlement price: settlement is pay-as-bid.

## Relaxed KKT embedding

In addition to the dispatch KKT conditions, the access allocation has duals
\(\alpha^{req}_{jn}\le0\) for \(x^A_{jn}\le q^A_{jn}\) and
\(\alpha^{node}_n\le0\) for the nodal access limit. With
\(\kappa_{jnt}\le0\) denoting the shared-inverter dual, the awarded-MW reduced
cost is

\[
r^A_{jn}=-b^A_{jn}
+\sum_t\kappa_{jnt}
-\alpha^{req}_{jn}-\alpha^{node}_n\ge0.
\]

The access complementarity pairs are

\[
x^A_{jn}r^A_{jn}=0,
\]

\[
(q^A_{jn}-x^A_{jn})(-\alpha^{req}_{jn})=0,
\]

\[
(\bar Q_n^A-\sum_jx^A_{jn})(-\alpha^{node}_n)=0.
\]

The SOC-capacity pair is now

\[
(X^E_{jn}-s_{jn\tau})(-\delta_{jn\tau})=0.
\]

Every lower-level complementarity product is replaced by
\(0\le zw\le\varepsilon\). All resulting nonlinear constraints are at most
quadratic; the former cubic
\(h_{jn}x^A_{jn}\delta_{jn\tau}\) terms no longer exist. A fixed-strategy
HiGHS solve supplies an exact lower-level primal/dual starting point before
Ipopt solves the nonconvex relaxed MPEC.

## Jacobi state

The damped upper strategies are \((q^A,b^A,X^E)\). After all best responses
are updated simultaneously, the runner clears one common exact ISO LP to
derive awarded MW without overwriting the independent MWh choices. During a
nonconverged Jacobi iterate, a simultaneous change in all bids can make a
player's common award differ from the award in its own counterfactual best
response, so the stored common state may temporarily violate that player's
2--8 hour bounds. These violations are exported explicitly and must vanish at
a claimed equilibrium. Format-v7 checkpoints retain requests, bids,
independent energy capacities, and common power awards.
