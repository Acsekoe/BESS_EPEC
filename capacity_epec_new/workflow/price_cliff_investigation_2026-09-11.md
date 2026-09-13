# Capacity–profit discontinuities in the storage investment game

The principal observed difficulty is a change in marginal electricity prices that revalues existing portfolios when storage capacities change by tiny amounts. The new experiments confirm this mechanism, distinguish it from arbitrary price selection at a fixed capacity, and show that the proposed remedies have substantially different effects. Demand response suppresses the saved jumps most effectively. A consistent price rule makes payoffs reproducible but leaves important jumps intact. Curved operating costs are insufficient. Averaging over uncertain operating conditions is promising, but a finite scenario approximation can retain or miss the very thresholds that matter to investment incentives.

No full-game equilibrium is established. Neither the existence nor the multiplicity of pure equilibria follows from the observed response branches. The experiments below investigate the payoff mechanism and possible remedies; they are not a new unrestricted equilibrium search.

## Findings at a glance

| Question | Finding | Evidence and limit |
|---|---|---|
| Do tiny capacity changes create material profit changes? | Yes, in all three saved counterexamples. | Fresh reference IPOPT clearing reproduces the earlier numbers. |
| What generates the largest gains? | Repricing existing renewable generation. | At the planner point, I3 gains €37,024.79/day on generation prices and loses €2,676.04/day on storage prices. |
| Will a consistent price-selection rule remove the jumps? | No, although it is needed to define consistent payoffs. | A minimum-norm LMP selection retains approximately €33,397.73/day of the planner jump and €626.60/day of the final N5 jump. |
| Does positive demand-response rho help? | Strongly on the tested paths. | At rho = 100, the saved planner perturbation gains €0.000337/day; the saved N5 expansion loses €0.01560/day. |
| Does adding operating-cost curvature solve the problem? | No. | Even combined generation and storage curvature leaves an €8,966.53/day planner jump and a new verified non-quasiconcavity example. |
| Does scenario averaging help? | It spreads operating thresholds, but finite scenarios preserve individual jumps. | Under ±1% demand variation, the planner tiny-step gain is €1,079.76/day with 33 scenarios and €548.19/day with 65, under the stated HiGHS convention. |
| Is a smooth-looking curve or a stable capacity profile enough? | No. | An even scenario grid can miss the threshold; small capacity residuals can still accompany large profitable deviations. |

## 1. Scope and experimental design

The reference is the energy-only capacity game in `market_data_smoothed.json`, with inelastic demand (`rho = 0`), the maintained investor costs and ownership, nodal power and energy decisions, the 2–8 hour duration band, and shared connection limits. I1 owns no generation, I2 owns the wind portfolio, and I3 owns the solar portfolio. Each comparison changes one investor's capacities while keeping its rivals fixed. The saved nodal power and energy allocation is retained; the planner fleet is not redistributed uniformly.

Three cases anchor the investigation:

1. **Planner path:** interpolate I3's capacities from the equal planner split toward its saved first single-start response. The original perturbation goes from interpolation parameter 0 to 2.384185791015625 × 10⁻⁷; its largest nodal power change is 8.336078543677642 × 10⁻⁷ MW.
2. **Final N5 candidate:** increase I2's N5 power by 0.001 MW and energy by 0.003 MWh, starting from the saved 227.007691 MW fleet.
3. **I2/I3 payoff witness:** hold I3's N6 capacity at 51.92654866758656 MW and vary I2's N5 power at three hours. The two nearby points are 45.13671975 and 45.136721480529644 MW; all other entries follow the saved first corrected coordinate-sweep profile.

The investigation uses fresh IPOPT reference solves for the headline before/after comparisons. Large deterministic screens use a new compiler that reads the maintained Pyomo ISO expressions: HiGHS for LPs and Clarabel for QPs. These backends have their own dual-selection behaviour. Their results are labelled as screens and are compared with independently rebuilt IPOPT models at important points. Clarabel's documented quadratic/conic problem form supplies the numerical interface, not a new economic formulation.[^1]

The screens cover the planner interpolation interval, ±2 MW around the final N5 position, and the full feasible scalar I2–N5 interval for the payoff-shape test. Local refinement locates candidate transitions. Separate reference-only refinements test whether changes decay as the bracket shrinks. These are targeted deterministic probes; there are no MPEC multistarts.

The completed study includes 39,239 LP/QP screening evaluations across the geometry, payoff-shape, scenario and scenario-resolution experiments, additional dual-face and price-selection diagnostics, and 189 saved fresh IPOPT reference evaluations. Smoke checks and automated tests add further independent solves. The reference count covers market clearing, not 189 best-response or equilibrium searches.

New experiments and data are in [price_cliffs_20260911](../model/output/price_cliffs_20260911/). The source profiles and their hashes are recorded in [manifest.json](../model/output/price_cliffs_20260911/manifest.json). The earlier evidence is documented in [the September 11, 09:55 summary](summary_2026-09-11_09-55.md).

## 2. The profit jumps are mainly portfolio repricing

The before/after decomposition separates changes in prices from changes in dispatched quantities. For any settlement term λq, the exact symmetric decomposition is

    Δ(λq) = Δλ × (q_before + q_after)/2
            + Δq × (λ_before + λ_after)/2.

For generation rent, the quantity term uses the average price minus true generation cost. Changes in degradation, experimental curvature costs, and capital cost are subtracted separately. This avoids assigning the entire price–quantity interaction to whichever variable is changed first in the accounting. Numerical reconstruction errors are below 3 × 10⁻¹¹ EUR/day in the saved comparisons.

| Component, EUR/day | Planner perturbation: I3 | Final N5 expansion: I2 | Nearby pair-witness points: I2 |
|---|---:|---:|---:|
| Generation price effect | +37,024.78570 | +959.21558 | +959.21092 |
| Storage price effect | −2,676.03968 | −332.60189 | −396.25991 |
| Storage quantity effect | +0.000012 | +0.020363 | +0.000039 |
| Generation quantity effect | Approximately zero | Approximately zero | Approximately zero |
| Additional degradation | 0.000007 | 0.015786 | 0.000030 |
| Additional capex | 0.000003 | 0.020151 | 0.000035 |
| **Net profit change** | **+34,348.74602** | **+626.59811** | **+562.95099** |

![Decomposition of the saved profit changes](../model/output/price_cliffs_20260911/figures/01_profit_decomposition.png)

*The small quantity, degradation and capital-cost terms are omitted from the bars but included in the net profit and the table. Source: fresh IPOPT comparisons in [decomposition.csv](../model/output/price_cliffs_20260911/baseline/decomposition.csv).*

At the planner comparison, ISO operating cost changes by only −€0.00000214/day. The largest generation-output change is approximately 2.98 × 10⁻⁷ MW, while the largest LMP change is €22.37682/MWh. I3's benefit is overwhelmingly a transfer through prices on its existing solar output, rather than additional physical production or a material reduction in system cost.

The effect concentrates in solar hours and locations. At N8 in hour 14, the reference price rises from €0.62297 to €22.99979/MWh, creating approximately €9,765.80/day of I3 generation-price benefit in that one node-hour. N8 hours 12, 13 and 15 and N6 midday hours account for most of the remainder. See [hourly_price_effects.csv](../model/output/price_cliffs_20260911/baseline/hourly_price_effects.csv).

The final N5 expansion similarly supports I2's wind generation at N1. Its N1 price rises from €23 to approximately €27.10579/MWh in hours 12–16. Meanwhile I3 loses about €8,391.28/day in generation rent and recovers about €2,297.43/day through storage settlement, for a net loss of €6,093.85/day. The investors' opposing incentives therefore involve the prices received by their existing portfolios, even when they change only a small battery position.

The fixed-profile accounting control also answers what happens if generation rent is excluded from the measured gain: both headline moves become unattractive on their storage economics alone. That control isolates a mechanism; it is not a solution of a different game in which ownership has been removed and every investor is allowed to re-optimise.

### Congestion prices can change while a line remains at its limit

The LP shadow-price diagnostics identify L46 as important for the N5 transition. Its lower-bound multiplier changes from approximately zero to €11.36270 per MW in hours 12–16. The relevant lines need not move from visibly uncongested to visibly congested in every affected hour: several are already at their limits before the move but have zero congestion multipliers. The small extra capacity changes the marginal value of the constraint.

In the HiGHS before/after dispatch, L46's hour-15 slack falls from 5.23 × 10⁻⁶ MW to zero at the final N5 case, and from 8.91 × 10⁻⁸ MW to zero in the pair witness. A congestion flag using a 10⁻⁵ MW threshold would classify both sides as binding and miss this distinction. At the planner point, L46 and L78 congestion multipliers disappear across the perturbation even though several physical flows remain at their bounds.

These diagnostics use the explicitly labelled HiGHS primal-dual solution, whose dispatch can differ from IPOPT on economically indifferent operating choices. They explain the marginal-price mechanism without asserting that every difference between the solvers' dispatches is economically material. The underlying values are in [line_shadow_changes.csv](../model/output/price_cliffs_20260911/baseline/line_shadow_changes.csv).

## 3. A common price rule resolves ambiguity, not the underlying cliff

Optimal ISO cost and optimal prices are different objects. Convex clearing can have a unique optimal value while permitting several dual solutions. Duality and sensitivity analysis link prices to marginal cost changes, so a change in the relevant marginal constraint can matter much more for settlements than for total operating cost.[^2]

The new dual-face diagnostics impose the LP stationarity equations and nonnegative inequality multipliers, with no artificial bounds on prices or multipliers. They then maximise and minimise each investor's settlement over a near-optimal dual-cost band. The band is tightened from €0.001/day to zero as entered numerically.

At the planner endpoint, the I3 price-face range remains about €35,632/day as the band tightens. At the final N5 incumbent, the I2 range instead falls from approximately €626.72 at a €0.001 cost band to €106.25 at €0.00001 and €1.85 at €0.0000001. A residual range of about €0.79 remains when the requested band is zero, but the actual numerical dual-cost residual is still about 2.6 × 10⁻⁸ EUR/day. That residual is amplified by the proximity to the threshold. It must not be presented as a certificate of genuine €0.79 price multiplicity.

Thus, the final N5 jump cannot be explained simply as two arbitrary prices at the same incumbent capacity. Price uncertainty from a loose optimality band and a genuine payoff change across nearby capacities are distinct.

### Minimum-norm LMP experiment

The tested convention selects the LMP vector with minimum squared Euclidean norm over the optimal dual face and pairs it with a deterministic HiGHS optimal dispatch. The numerical implementation eliminates dual variables attached to strictly slack primal inequalities, repeats the calculation at several slack cutoffs, and checks stationarity, nonnegativity and the primal–dual gap in original units. The accepted values below use a 10⁻¹⁰ MW slack cutoff and agree with the 10⁻⁸ cutoff at the important before/after points.

| Case | Reference IPOPT profit gain | Minimum-norm LMP profit gain |
|---|---:|---:|
| Planner perturbation, I3 | €34,348.75/day | €33,397.73/day |
| Final N5 expansion, I2 | €626.59811/day | €626.59816/day |
| Nearby pair-witness points, I2 | €562.95099/day | €562.95376/day |

At the planner endpoint, the rule changes I3's selected profit to approximately €89,481.63/day; immediately beyond it the selected profit is approximately €122,879.36/day. A reproducible rule changes the endpoint valuation but leaves a substantial jump. At the final N5 case, the rule essentially reproduces the prices already used by the reference clearing.

The minimum-norm rule is a numerical prototype for evaluating fixed profiles. It has **not** been embedded as an additional lower-level selection condition in the production MPEC. Merely applying a rule after an optimistic MPEC solve would still leave the MPEC optimising a different payoff. Either the rule must be represented in the best-response formulation or the best-response search must evaluate candidate capacities through the same complete clearing rule.

Minimum norm is also a convention, not an empirically established market practice. Its usefulness here is diagnostic: even a consistent, reproducible convention does not remove these cliffs. Data: [selections/summary.csv](../model/output/price_cliffs_20260911/selections/summary.csv) and [faces/ranges.csv](../model/output/price_cliffs_20260911/faces/ranges.csv).

## 4. Demand response is the strongest tested local remedy

For positive rho, the ISO adds a quadratic demand-adjustment cost. At a loaded node with a freely adjustable quantity, the stationarity relation is

    DemandAdjustment = LMP / rho.

Larger positive rho means more expensive adjustment and a steeper price response. In this code, `rho = 0` is a separate inelastic-demand switch that fixes adjustment to zero. It is not the limit obtained by reducing a positive penalty to zero. Positive rho changes the economic model; it is not a neutral price tie-break.

All entries below compare the **same original pair of capacities** under each market specification, with fresh IPOPT clearing.

| Market | I3 gain at saved planner perturbation | I2 gain from saved 0.001 MW N5 expansion |
|---|---:|---:|
| Inelastic demand | +€34,348.74602/day | +€626.59811/day |
| rho = 25 | +€0.000088/day | −€0.020151/day |
| rho = 100 | +€0.000337/day | −€0.015604/day |
| rho = 500 | +€0.001661/day | −€0.015604/day |

The original N5 profitable deviation becomes a small loss in all three positive-rho cases. The planner tiny-step gain falls by many orders of magnitude. However, the wider planner path can still offer substantial gains: eliminating the infinitesimal jump does not make the original fleet an equilibrium.

The economic adjustment at the original, unperturbed profiles is:

| rho | Planner adjustment | Planner daily demand share | Final-candidate adjustment | Final daily demand share | Largest node-hour adjustment share, both profiles |
|---|---:|---:|---:|---:|---:|
| 25 | 102.243 MWh/day | 0.77545% | 104.196 MWh/day | 0.79026% | 5.06173% |
| 100 | 25.720 MWh/day | 0.19507% | 26.857 MWh/day | 0.20369% | 1.26543% |
| 500 | 5.144 MWh/day | 0.03901% | 5.371 MWh/day | 0.04074% | 0.25309% |

A small daily average can conceal a larger adjustment in a low-demand node-hour. Calibration should therefore use both measures. These percentages quantify the synthetic sensitivity; they do not establish that actual consumers have these elasticities. The present quadratic response remains unbounded at loaded nodes. A physically bounded, calibrated response curve is a separate specification to validate.

![Profit paths under different demand-response penalties](../model/output/price_cliffs_20260911/figures/02_demand_response_curves.png)

*Curves use the explicitly labelled LP/QP screening backends. Important endpoints and transitions are independently checked with IPOPT. The horizontal coordinate is an I3 interpolation in the left panel and an I2 N5 capacity change at three hours in the right panel; rivals remain fixed. Sources: [rho/curves.csv](../model/output/price_cliffs_20260911/rho/curves.csv), [rho/reference_checks.csv](../model/output/price_cliffs_20260911/rho/reference_checks.csv).*

### Steepness remains relevant

At adaptively selected planner intervals of approximately 8.89 × 10⁻⁶ MW maximum nodal width, the reference profit changes are €0.16805/day at rho = 100 and €0.84041/day at rho = 500. These correspond to local finite-difference slopes of roughly €18,900 and €94,500 per MW per day. A larger penalty reduces load adjustment but can make the transition harder to resolve.

Further reference-only halving at rho = 500 reduces the change to €0.000821/day over 8.68 × 10⁻⁹ MW, consistent with a steep continuous transition. At rho = 100, the tested sequence approaches a roughly €0.004/day numerical floor, accompanied by solver sensitivity in indifferent dispatch. This is a limit of the computation, not a proof of a residual economic discontinuity. There is no global theorem of price uniqueness or smoothness established by these experiments.

### The saved payoff counterexample disappears on this slice

For each rho, an 83-point deterministic screen covers I2's feasible N5 interval from 0 to approximately 200 MW, including the old counterexample points. The inelastic case reproduces a quasiconcavity violation of €315.58025/day with fresh IPOPT clearing. No positive violation is found in the three positive-rho screens, and the old three-point witness is independently revalued without a violation.

The sampled best point on this particular slice changes from approximately 45.13672148 MW in the inelastic case to exit at all three positive rho values. That is evidence of a material change in investment incentives, not merely a numerical improvement. It does not prove global quasiconcavity, prove a best response in all nodal power/energy directions, or establish an equilibrium with other investors reacting. Data: [shape/summary.csv](../model/output/price_cliffs_20260911/shape/summary.csv).

## 5. Curved operating costs do not provide a general cure

Four separate sensitivities add curvature inside ISO operations while retaining inelastic demand:

- **Generation +1:** each generator's marginal cost rises linearly by €1/MWh from zero output to its maximum available output in the reference day.
- **Generation +5:** the corresponding marginal-cost increment is €5/MWh.
- **Storage curvature:** add 0.5 × 0.1 × (charge² + discharge²) to each battery's operating cost, using MW variables and the model's hourly convention.
- **Combined:** generation +5 and storage curvature together.

These are explicit sensitivity assumptions, not fitted generator offers or degradation curves. The added terms are treated as true operating costs: owned generation curvature is subtracted from generation rent, and storage curvature is subtracted from storage surplus. Otherwise the comparison would accidentally count added operating costs as investor profit. The production ISO and MPEC formulations are not changed by these experiments.

| Experimental operating costs | Planner tiny-step gain, I3 | Final N5 0.001 MW gain, I2 |
|---|---:|---:|
| Original linear costs | +€34,348.75/day | +€626.598/day |
| Generation +1 | +€27,582.87/day | +€412.980/day |
| Generation +5 | +€9,920.77/day | −€0.02990/day |
| Storage curvature | +€31,929.12/day | −€0.02015/day |
| Combined | +€8,966.53/day | −€0.02015/day |

Some modifications remove the specific final N5 incentive, but every curvature variant retains a large planner-profile jump. The combined case also has an interior transition: fresh reference refinement finds an approximately €45.62/day profit change over a maximum nodal interval of 8.68 × 10⁻⁹ MW. Generation +1 retains a roughly €214.74/day N5 change across a 9.54 × 10⁻¹⁰ MW interval. Such extremely small intervals approach numerical limits, so the robust conclusion rests additionally on the larger saved perturbations in the table; these are not exact mathematical discontinuity proofs.

![Comparison of the same perturbations across market modifications](../model/output/price_cliffs_20260911/figures/03_reference_perturbation_comparison.png)

*All bars use fresh IPOPT values for the same capacities. The symmetric log scale makes both very large gains and small losses visible. Sources: [rho/summary.csv](../model/output/price_cliffs_20260911/rho/summary.csv), [curvature/summary.csv](../model/output/price_cliffs_20260911/curvature/summary.csv).*

The combined case also produces a new, independently verified violation of quasiconcavity on the saved I2 unilateral line:

| I2 N5 power at 3 hours | I2 profit under combined curvature |
|---|---:|
| 0 MW | €72,160.66580/day |
| 5 MW | €72,150.68607/day |
| 12.5 MW | €72,259.29719/day |

The middle strategy is a convex combination of the two endpoints, yet earns €9.97973/day less than the worse endpoint. Therefore this modified payoff is not quasiconcave on that line to the measured numerical accuracy. Curvature in operating costs has not made the upper-level investment game concave.

![Payoff-shape comparisons and the new curvature counterexample](../model/output/price_cliffs_20260911/figures/06_payoff_shape.png)

*Lines show deterministic screened values; the three marked points in the right panel are fresh IPOPT evaluations. The dashed line is the lower endpoint profit required by quasiconcavity. The left panel displays the first 70 MW of a screen extending to the full approximately 200 MW feasible scalar limit. Source: [shape](../model/output/price_cliffs_20260911/shape/).*

### Why strict primal curvature is insufficient

A simple independent example illustrates the distinction. Minimise

    0.5 q1² + 10 q2 + 0.5 q2²
    subject to q1 + q2 = D,  0 ≤ q1 ≤ 1,  q2 ≥ 0.

The objective is strictly convex and the dispatch is unique. Just below D = 1, the marginal price tends to 1. Just above D = 1, it tends to 10. At D = 1, any balance price between 1 and 10 can satisfy optimality with suitable bound multipliers. The marginal-cost gap between the exhausted cheap unit and the next unit survives the quadratic terms.

This example does not assert that the storage model reduces to two generators. It demonstrates why unique optimal dispatch and continuous marginal prices require separate analysis. Curving existing supply blocks or storage costs need not fill the relevant marginal gap or remove constraint degeneracy.

Quadratic **investment** costs are still less direct: if the original operating-profit jump is J and the added investment cost k(x) is continuous, subtracting k(x) leaves the limiting jump J. Such costs can change which branch is attractive, but cannot make an existing revenue discontinuity continuous. Likewise, damping changes the iteration path rather than the payoff function. Neither is an appropriate substitute for diagnosing the price mechanism.

## 6. Scenario averaging helps, but naive discretisation can mislead

The scenario experiments hold investment capacities fixed across operating scenarios. They vary one source of uncertainty at a time: demand by ±0.1%, ±1% or ±5%, or wind or solar availability by ±1%. Each daily shape is scaled coherently; chronology within the day is retained. The distributions are stylised uniform stresses, not calibrated joint demand/weather distributions or a representative-year dataset.

For each stress and each of the two main paths, expected profits are evaluated using 9, 33 and 65 equally weighted midpoint scenarios. The deterministic HiGHS clearing convention is used for these LP screens, with its basis reset for each solve. Selected low, zero and high shocks are separately revalued with fresh IPOPT. Consequently the scenario screen's central planner jump is €35,632.18/day, rather than the IPOPT convention's €34,348.75/day. This convention difference is material and is not hidden by averaging.

For ±1% demand variation, the gain from the original tiny perturbation is:

| Number of scenarios | Planner gain, I3 | Final N5 gain, I2 |
|---|---:|---:|
| 9 | €3,959.13/day | €69.60981/day |
| 33 | €1,079.76/day | €18.97423/day |
| 65 | €548.19/day | €9.62620/day |

The reduction is real for this discrete expected-payoff model, but it is largely the dilution of the central scenario's discontinuity by its weight. One equally weighted scenario still sits at the original problematic operating condition. Other scenarios place their thresholds elsewhere along the investment path.

![Expected profit curves across scenario counts](../model/output/price_cliffs_20260911/figures/04_scenario_expectations.png)

*Stylised ±1% demand variation, fixed investment across scenarios, deterministic HiGHS settlement. Connecting sampled points aids comparison and is not a claim of continuity between them. Source: [scenarios/curves.csv](../model/output/price_cliffs_20260911/scenarios/curves.csv).*

Wind and solar perturbations also reduce the original single-day sensitivity as scenario count increases, but do not eliminate the central atom. At 65 scenarios, the final N5 gain remains approximately €9.63/day in these cases. Increasing the uncertainty amplitude while retaining the same scenario count does not remove the central planner contribution: at 65 scenarios it remains roughly €548/day across the demand amplitudes tested.

### Even versus odd scenario counts reveal a sampling trap

An additional check changes the midpoint count so that the grid alternately includes or excludes a zero shock. For ±1% demand uncertainty:

| Scenarios | Zero shock included? | Planner tiny-step gain | Final N5 tiny-step gain |
|---|---|---:|---:|
| 32 | No | About €0.000003/day | −€0.01396/day |
| 33 | Yes | +€1,079.76/day | +€18.97423/day |
| 64 | No | About €0.000003/day | −€0.01396/day |
| 65 | Yes | +€548.19/day | +€9.62620/day |
| 128 | No | About €0.000003/day | −€0.01393/day |
| 129 | Yes | +€276.22/day | +€4.84349/day |

The even-grid result is not proof that the model's cliffs have disappeared. It shows that a particular tiny capacity interval contains none of those scenarios' thresholds. A strategic optimiser can seek the thresholds at other capacities. The scenario law, scenario weights and integration resolution must be distinguished from a favourable result at one chosen point.

![Sensitivity to whether the scenario grid includes the central threshold](../model/output/price_cliffs_20260911/figures/05_scenario_parity.png)

*The tested capacity pair is held fixed. Coloured bars are odd counts containing zero shock; grey bars are even counts excluding it. This is a quadrature sensitivity test, not a comparison of different investment equilibria. Source: [scenarios/parity.csv](../model/output/price_cliffs_20260911/scenarios/parity.csv).*

### Continuous uncertainty can smooth a moving threshold

An elementary example makes the mechanism precise. If a payoff includes a jump J when capacity x exceeds a random threshold a + ξ, with ξ uniform on [−h, h], the expected contribution is

    0,                              x < a − h
    J (x − a + h)/(2h),             a − h ≤ x ≤ a + h
    J,                              x > a + h.

The discontinuity becomes a continuous ramp because the random threshold has no point masses. A discrete approximation with scenario weights w_s instead retains jumps of size w_s J at its individual thresholds. This reasoning explains the approximately 1/N behaviour in the experiments; it is not a proof that every threshold in the full market moves under every source of uncertainty.

Convolution-based smoothing is an established optimisation idea, but the convex stochastic-optimisation convergence results in the cited literature do not directly apply to this nonconcave equilibrium problem.[^6] For investment modelling, the economically relevant uncertainty should enter demand, availability or other operating fundamentals, rather than being chosen solely to make the graph look smooth. Preserving chronological storage constraints remains necessary when extending beyond the single daily profile.

The practical implication is to validate expected **deviation gains** under scenario refinement and shifted quadrature grids, and to refine integration near moving thresholds. A small number of weighted representative days may be useful economically, but each high-weight day can still contribute a substantial payoff cliff. The current results support further work on continuous operating uncertainty or an economically calibrated combination of uncertainty and demand response; they do not certify that 65 scenarios solve the problem.

## 7. Consequences for equilibrium existence and the next search

The evidence does not support a probability statement that there are multiple equilibria. The original game could have no pure equilibrium, one, or several under a fully specified clearing convention. Several local maxima, discontinuous best-response branches, different algorithmic trajectories, and interchangeable-looking locations do not establish multiple equilibria.

Standard existence results for generalised Nash problems require conditions on both payoffs and feasible-set correspondences; continuity alone is insufficient. In particular, the reference payoff's verified failure of quasiconcavity prevents directly invoking the standard sufficient theorem on that basis.[^3] Failure of a sufficient condition does not establish nonexistence. Results for discontinuous games allow equilibria under additional conditions, but those conditions have not been verified for this capacity game.[^4]

The shared nodal connection limits also mean the model is a generalised Nash game. A finite independent action grid or an off-the-shelf mixed-equilibrium argument cannot silently ignore joint feasibility. Even where finite approximating games are well defined, convergence of their equilibria to an equilibrium of a discontinuous continuous-strategy game requires additional care.[^5]

The most defensible sequence is therefore:

1. **Choose and state the economic specification.** Keep the inelastic model as a benchmark. For the leading practical sensitivity, use rho = 100 as an experimental comparison, while calibrating and bounding actual demand response before treating it as a maintained market assumption. The measured tradeoff is about 0.2% daily adjustment and up to 1.27% in a node-hour at the two anchor profiles.
2. **Use one complete clearing convention for candidate valuation and response evaluation.** A minimum-norm rule is one prototype, but an optimistic MPEC needs additional modelling to respect it. A consistent direct market-evaluation search is an alternative. History-dependent price selection would change the static game and should not be introduced implicitly.
3. **Add economically motivated operating scenarios and check integration error.** Keep investment common across scenarios. Refine scenario weights and threshold regions, and compare shifted grids. A fixed arbitrary scenario count is not an accuracy guarantee.
4. **Search I2 and I3 using verified deviation gains.** Use broad deterministic scans to locate branches and targeted one-sided evaluations near boundaries. Continue to allow power, energy, siting and duration deviations; a scalar slice cannot certify the full response. Single-start local MPEC solutions remain candidates, not global best-response bounds.
5. **Recompute the merchant-entry bound when rivals or the market specification change.** The earlier zero-entry LP certificate is valuable at its stated inelastic rival profile. It cannot simply be carried over after demand response, curved costs, uncertainty or rival capacities change.
6. **Separate economic regret from numerical and integration error.** A claimed maximum deviation gain must be assessed against clearing/settlement error and expected-value approximation error. Stability of capacities, an incumbent winning a failed search, or zero gain on a finite grid is insufficient.

A concrete next computational experiment would compare the reference inelastic model, a calibrated bounded-demand-response model near the rho = 100 sensitivity, and that same response model with refined operating uncertainty. Each should start from the documented planner allocation, use the same unrestricted deviation checks, and report both profit regret and economic distortion. Curved investment costs or stronger damping should not be the primary mechanism for suppressing the measured cliff.

## 8. Numerical reliability and limits

**Reference versus screening selection.** The LP/QP compiler is generated from the maintained ISO expressions and verified against independently rebuilt reference models. Equal optimal operating costs do not imply equal selected prices or dispatch. At the planner profile, several curvature variants have materially different Clarabel and IPOPT prices at nearly identical operating cost. Those differences are precisely why the report's perturbation tables use fresh reference settlements rather than the screening values.

**False confidence from solver status.** A direct minimum-norm dual QP over a nearly empty optimal-cost band sometimes returned a `Solved` status while its residuals in original units were unacceptable. For example, one saved planner-side result had a stationarity residual around 0.00235 and negative inequality multipliers around −0.00183. Those selections are rejected as evidence. The residual-audited active-face reduction produces the accepted selection table; the earlier direct-slab results remain in `faces/` for traceability.

**Active-set cutoff sensitivity.** At the perturbed planner point, a loose 10⁻⁶ MW cutoff incorrectly admits the old price branch and produces a roughly €6.65 × 10⁻⁶/day primal–dual gap, above the acceptance gate. The tighter 10⁻⁸ and 10⁻¹⁰ cutoffs agree and pass. A cutoff that is small in engineering units can still be large relative to the strategically important perturbation.

**Refinement below reliable numerical resolution.** The broad screens sometimes refine a bracket down to about 5 × 10⁻¹¹ MW around the planner endpoint. Those brackets are useful warnings about conditioning, not reliable estimates of exact discontinuity locations. A small reference change at that numerical scale does not overturn the verified large change across the larger original bracket. All refinement traces are retained, including the rho = 100 numerical floor.

**Finite evidence.** No finite grid proves global smoothness, quasiconcavity or absence of profitable deviations. The new curvature counterexample proves a local failure of quasiconcavity to numerical accuracy; the absence of a counterexample under positive rho proves no global property. The scenario stresses do not constitute a calibrated weather model or a full stochastic equilibrium.

**Testing.** The full suite passes: **70 tests in 15.70 seconds**, with pytest's cache provider disabled because of the pre-existing cache permissions. The four new tests check numerical settlement against the maintained accounting, demand-RHS changes and return to the original profile, curved-QP results against fresh IPOPT, survival of the N5 cliff under accepted minimum-norm selection, and exact reconstruction of the price/quantity decomposition. No multistart search is performed.

## Reproduction and data

From the repository root, run the following commands with the investigation dependencies installed:

```powershell
python capacity_epec/model/investigate_price_cliffs.py baseline
python capacity_epec/model/investigate_price_cliffs.py rho
python capacity_epec/model/investigate_price_cliffs.py curvature
python capacity_epec/model/investigate_price_cliffs.py faces
python capacity_epec/model/investigate_price_cliffs.py selections
python capacity_epec/model/investigate_price_cliffs.py scenarios
python capacity_epec/model/investigate_price_cliffs.py scenario-resolution
python capacity_epec/model/investigate_price_cliffs.py shape
python capacity_epec/model/investigate_price_cliffs.py refinement
python capacity_epec/model/report_price_cliffs.py
python -m pytest capacity_epec/tests -q -p no:cacheprovider
```

The reference-evaluation cache stores the actual independent IPOPT results. For fresh reruns after changing model code or input data, use a new output directory with `--output-dir`; do not reuse a cache created from a different specification. The plotting script defaults to the saved investigation directory. Input and code hashes and saved solve counts are in [manifest.json](../model/output/price_cliffs_20260911/manifest.json).

| Artifact | Contents |
|---|---|
| [baseline/decomposition.csv](../model/output/price_cliffs_20260911/baseline/decomposition.csv) | Exact accounting split for all investors at three saved transitions |
| [baseline/hourly_price_effects.csv](../model/output/price_cliffs_20260911/baseline/hourly_price_effects.csv) | Node-hour contributions to portfolio repricing |
| [faces/ranges.csv](../model/output/price_cliffs_20260911/faces/ranges.csv) | Profit ranges over progressively tighter numerical dual-cost bands |
| [selections/summary.csv](../model/output/price_cliffs_20260911/selections/summary.csv) | Minimum-norm selections, cutoff sensitivity and raw residual gates |
| [rho/summary.csv](../model/output/price_cliffs_20260911/rho/summary.csv) | Demand-response comparisons and selected transitions |
| [curvature/summary.csv](../model/output/price_cliffs_20260911/curvature/summary.csv) | Operating-cost curvature comparisons |
| [scenarios/summary.csv](../model/output/price_cliffs_20260911/scenarios/summary.csv) | Scenario counts, stress sizes and expected tiny-step gains |
| [scenarios/parity.csv](../model/output/price_cliffs_20260911/scenarios/parity.csv) | Even/odd grid sensitivity |
| [shape/summary.csv](../model/output/price_cliffs_20260911/shape/summary.csv) | Verified payoff-shape witnesses and limits of the scalar screens |
| [refinement/brackets.csv](../model/output/price_cliffs_20260911/refinement/brackets.csv) | Reference-only shrinking-bracket results |
| [figures](../model/output/price_cliffs_20260911/figures/) | PNG and SVG versions of all six figures; their data are in the linked CSVs |
| [investigate_price_cliffs.py](../model/investigate_price_cliffs.py) | Reproducible experimental market probes |

## Sources

The numerical findings are original calculations on the local model and saved profiles, documented by the linked artifacts above. External sources support the optimisation framework and interpretation, not the numerical results or a claimed equilibrium.

[^1]: Clarabel documentation, “[Getting Started: Python](https://clarabel.org/stable/python/getting_started_py/),” accessed September 11, 2026. Canonical quadratic/conic formulation and solver interface. Installed version recorded in the manifest: 0.11.1.
[^2]: Stephen Boyd and Lieven Vandenberghe, *[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)*, Cambridge University Press, 2004, Chapter 5, especially sensitivity analysis. The [authors' slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf) provide the corresponding dual-sensitivity statements.
[^3]: Francisco Facchinei and Christian Kanzow, “[Generalized Nash Equilibrium Problems](https://www.mathematik.uni-wuerzburg.de/fileadmin/10040700/paper/Facchinei_Kanzow_AORP.pdf),” updated preprint, June 4, 2009, Section 4.1 and Theorem 4.1. This is an updated version of the 2007 *4OR* survey; objectives in that statement are expressed as minimisation costs.
[^4]: Philip J. Reny, “[Nash Equilibrium in Discontinuous Games](https://bfi.uchicago.edu/wp-content/uploads/BFI_2013-004.pdf),” BFI Working Paper 2013-004, September 3, 2013. Additional conditions for equilibrium existence in discontinuous games; none is assumed verified here.
[^5]: Philip J. Reny, “[Strategic Approximations of Discontinuous Games](https://bfi.uchicago.edu/wp-content/uploads/BFI_2009-010.pdf),” MFI Working Paper 2009-010, October 2009. Conditions for finite approximations to preserve equilibrium conclusions.
[^6]: John C. Duchi, Peter L. Bartlett and Martin J. Wainwright, “[Randomized Smoothing for (Parallel) Stochastic Optimization](https://stanford.edu/~jduchi/projects/DuchiBaWa12_icml.pdf),” 2012. Convolution-based smoothing in convex stochastic optimisation. Its convergence theorems are not asserted for this nonconcave generalised Nash problem.
