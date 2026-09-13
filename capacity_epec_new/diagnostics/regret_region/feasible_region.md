# Feasible investment region of the capacity game

**Project:** `capacity_epec_new` (maintained model files unchanged)
**Derived from:** `model/mpec.py`, `model/capacity_game.py`, `model/iso_market.py`, `model/investors.py`, `model/market_data.py`, `model/solvers.py`, the run configuration of `init5mw15mwh_jacobi_rho100_d025_s100`
**Measured faces:** `output/feasible_region/faces_by_investor_node.csv`, `output/feasible_region/shared_cap_by_node.csv`

## 1. Decision variables

Nodes $\mathcal N=\{N1,\dots,N9\}$, investors $\mathcal I=\{I1,I2,I3\}$. Investor $i$ chooses

$$x_i=(P_{in},E_{in})_{n\in\mathcal N}\in\mathbb R^{18},\qquad x=(x_{I1},x_{I2},x_{I3})\in\mathbb R^{54}.$$

Nothing else is strategic: `strategic_variables = [nodal_power_capacity_mw, nodal_energy_capacity_mwh]`, `operational_bidding = false`.

## 2. Every constraint on capacities in the maintained code

| # | Constraint | Where it is imposed | Tolerance |
|---|---|---|---|
| C1 | $P_{in}\ge 0,\;E_{in}\ge 0$ | `mpec.py` `X_power`, `X_energy` are `NonNegativeReals`; `capacity_game._check_capacity_feasible` rejects `power < 0`; returned responses are clipped `max(0, ·)` | exact / clipping |
| C2 | $r^{\min}_i P_{in}\le E_{in}\le r^{\max}_i P_{in}$ with $r^{\min}=2$, $r^{\max}=8$ h for all investors | `mpec.py` `energy_ratio_min`, `energy_ratio_max`; `_check_capacity_feasible` | $10^{-7}$ in the exact screen; IPOPT `constr_viol_tol` = solver tolerance in the MPEC |
| C3 | $\sum_{i}P_{in}\le L_n=200$ MW (shared nodal connection limit, from the market input) | `mpec.py` `node_connection_limit` (rivals as constants); `_check_profile_node_limits` before every exact clear | $10^{-7}$ MW |

There is **no** budget, per-investor maximum, minimum project size, integrality, or coupling of energy capacities across investors. The JSON fields `x_power`/`x_energy` are not read by `load_market_data`.

**Implicit feasibility from the market.** At $\rho>0$ `DemandAdjustment` is a free variable at the loaded nodes N5, N6, N8, storage may idle, and generators may be dispatched down, so zero net injection everywhere is ISO-feasible. The ISO QP therefore has an optimum for every capacity vector: the market adds **no** implicit restriction on $x$ in the run's setting. (At $\rho=0$ the LP would additionally need load-serving feasibility.)

**Artificial or algorithmic bounds.** None of these defines the economic strategy set, but each shapes what a solver can return:

| Item | Value in the run | Effect |
|---|---|---|
| MPEC price bounds $\lambda,\lambda^{sys}\in[-500,500]$ | `price_bound = 500` | the MPEC can only represent lower-level solutions whose LMPs fit; audit utilisation 0.17 |
| MPEC dual bounds (all other multipliers within $\pm10^{4}$) | `dual_bound = 10000` | same, for congestion, capacity and SOC multipliers |
| Relaxed complementarity $s\cdot\mu\le\varepsilon$ | $\varepsilon=10^{-3}$ | enlarges the lower-level solution set (an $\varepsilon$-optimal market) |
| MPEC acceptance | bound violation $\le$ solver tolerance $10^{-4}$ | returned capacities satisfy C1–C3 to $10^{-4}$ before clipping |
| Iterate cleanup | `cleanup_tolerance = 1e-6` MW | Jacobi iterates below it are set to $(0,0)$; **not** applied to best responses |
| Sparse rivals | `sparse_capacity_tol = 1e-8` MW | rivals below it are omitted from the MPEC lower level but included in the exact reclear |

## 3. Structure of the region

Fix rivals and define the residual shared capacity $R_n(x_{-i})=L_n-\sum_{j\ne i}P_{jn}$. Inequalities C2 imply C1 ($E\ge 2P$ and $E\le 8P$ give $6P\ge0$), so at each node the feasible pair is the triangle

$$T_n(R_n)=\operatorname{conv}\{(0,0),\,(R_n,2R_n),\,(R_n,8R_n)\}\quad(R_n>0),\qquad T_n(0)=\{(0,0)\}.$$

Investor $i$'s strategy set is the **product of nine triangles**

$$K_i(x_{-i})=\prod_{n\in\mathcal N}T_n\big(R_n(x_{-i})\big)\subset\mathbb R^{18}.$$

It is polyhedral (27 half-spaces), convex, closed and bounded ($0\le P_{in}\le200$, $0\le E_{in}\le1600$), hence a nonempty compact polytope. When every $R_n>0$ it has dimension 18 and $3^9=19{,}683$ vertices, the products of triangle corners.

The joint feasible set is

$$K=\{x\in\mathbb R^{54}:\;(P_{in},E_{in})\in C\;\forall i,n,\;\; \textstyle\sum_iP_{in}\le L_n\;\forall n\},\qquad C=\{(P,E):2P\le E\le 8P\},$$

a compact convex polytope of dimension 54. It factorises by node, $K=\prod_n K_n$, with each $K_n\subset\mathbb R^6$ three 2–8 h wedges cut by one shared power half-space. The investors are coupled **only** through the nine shared power inequalities.

**Local feasible directions** at a point with $P_{in}>0$ strictly inside $T_n$:
- all directions in $(P_{in},E_{in})$ are feasible;
- power-only moves stay feasible while $E/8\le P\le E/2$;
- energy-only moves stay feasible while $2P\le E\le 8P$;
- duration-preserving scaling is feasible down to exit.

At the origin vertex only entry directions $\{dP\ge0,\,2dP\le dE\le 8dP\}$ are feasible (a one-sided tangent cone). On the 2 h face, $dE\ge 2dP$ is required.

## 4. Active and near-active faces at the final profile (sweep 100)

| Face | Status at the final profile |
|---|---|
| Shared cap C3 | inactive everywhere; highest utilisation 26.6 % at N8, slack $\ge146.8$ MW for every investor |
| 8 h face | no material position within 2 % |
| 2 h face | I2 at N3 (2.0020 h, 0.10 % above), I1 at N9 (2.0293 h, 1.5 %), I3 at N8 (2.0302 h, 1.5 %) |
| Origin vertex, exactly | I1 and I2 at N2 and N7 |
| Near the origin ("ghost" positions, $10^{-6}$–$10^{-3}$ MW) | I1 at N1, N4, N5; I2 at N1, N4, N5; I3 at N1, N2, N4, N5, N7, N9 |
| Material interior positions | I1: N3 (0.29 MW), N6, N8, N9. I2: N3, N6, N8, N9. I3: N3, N6, N8 |

So the effective local dimension is 8 coordinates for I1 and I2 and 6 for I3. Entry directions at the remaining nodes are one-sided, and three material positions sit on (or within 1.5 % of) the 2 h face, which makes duration-shortening moves one-sided too.

The ghost positions matter numerically. Their power-bound multipliers in the reference IPOPT reclear reach $-4\times10^5$ EUR/MW while the batteries are idle, producing primal-dual gaps of 11–30 EUR/day that IPOPT accepts on its scaled error (see the summary).

Because the shared cap is slack by more than 146 MW, the **local** game around the final profile is an ordinary Nash problem on a product of triangles. Its generalized (shared-constraint) nature matters only for moves that bring a node near 200 MW.

## 5. Why the complete model is a generalized Nash complementarity problem

The polyhedral capacity region does not make the equilibrium problem convex:

1. **Generalized Nash.** $K_i$ depends on $x_{-i}$ through C3, so an equilibrium is a GNE of a shared-constraint game. Variational (common-multiplier) and general GNE differ whenever a cap binds.
2. **Payoffs through a parametric market.** $\pi_i(x)=\sum_{n,t}\lambda_{nt}(x)\,q_{int}(x)-\mathrm{deg}_i(x)+\sum_{g}s_{ig}(\lambda_{n(g)t}(x)-c_g)\,p_{gt}(x)-k_i(x_i)$, where $(\lambda,q,p)$ is an optimal primal-dual solution of the ISO QP with $x$ in its right-hand sides.
   - At $\rho>0$ the loaded-node LMPs are unique and Lipschitz in $x$ ($\lambda_{nt}=\rho\,\mathrm{DA}_{nt}$ with $\mathrm{DA}$ unique).
   - Their slope grows with $\rho$, and $\pi_i$ is nonsmooth and nonconcave across ISO active-set boundaries.
   - Owned-generation rent makes I2's and I3's payoffs depend on prices at their generators' nodes, not only on their own storage dispatch.
3. **Each best response is an MPEC.** The lower-level optimality conditions enter as complementarity $0\le s\perp\mu\ge0$ or as strong duality. Constraint qualifications fail at every feasible point, so IPOPT returns local points of a relaxation.
4. **The three MPECs share one market**, so the equilibrium conditions form an EPEC. The stacked stationarity conditions are a nonconvex complementarity system at two levels, lower-level complementarity plus the upper-level multipliers of C1–C3, and no monotonicity follows from polyhedrality of $K$.

Existence therefore cannot be read off compactness and convexity of $K$ alone. Standard GNE theorems also need payoff quasi-concavity, which an earlier inelastic-demand witness shows fails, and continuity.
