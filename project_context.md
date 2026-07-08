# Project Context: Extending Daniel Horvath's BESS EPEC Thesis

Source project reviewed: `Overleaf_Daniel/6a4b5e807a222023e352c4e8`.

Thesis: *Strategic Allocation of Battery Energy Storage Systems in Multi-Service Markets: An EPEC Modeling Approach* by Daniel Horvath, BSc, TU Wien, April 2026.

Repository inventory reviewed:

| Category | Files / contents |
|---|---|
| Main thesis source | `main.tex`, `Abstract.tex`, `Introduction.tex`, `State-of-the-art and progress beyond.tex`, `Methodology.tex`, `Results and Discussion.tex`, `Synthesis of the results.tex`, `Conclusions.tex`, `settings.tex` |
| Bibliography | `references.bib`, `references-2.bib` |
| Figures | Grid topology, EPEC/MPEC schematic, load/renewable profiles, convergence history, investment capacities, line congestion, system balance, node-level balances |
| Template | TU Wien title/preamble/settings files under `template/` |
| Implementation code/data | No Pyomo, Python, CSV, Excel, or other model implementation/data files were present in the reviewed repository. The thesis states that the model was implemented in Python/Pyomo, but that implementation is not included here. |

## Current Supervisor Direction: Spot-Market BESS Competition Only

Status as of 2026-07-08: the follow-up project should completely ignore aFRR markets. Reserve capacity, activation, reserve prices, and stochastic activation scenarios are no longer part of the active model.

The current research target is a spot-market EPEC for strategic BESS investment. The core question is how different investors compete for scarce nodal BESS power access and then operate their batteries in the spot market.

The model should focus on three elements:

- **Nodal power-access competition:** each node has a shared BESS connection limit, e.g.

$$
\sum_i X^{power}_{i,n} \le Limit^{node}_n
\qquad \forall n .
$$

- **Spot-market operation:** the lower level remains a deterministic DC-OPF / spot-market clearing problem with BESS charging, discharging, SOC dynamics, generator dispatch, nodal balance, and PTDF line limits.

- **Investor heterogeneity:** investors should differ not only by WACC, but also by operational constraints and business model. Examples are a stand-alone battery investor, a portfolio-backed investor with generation/load positions, or an investor with stricter cycling, SOC, or availability constraints.

The main methodological concern is the treatment of competition for \(Limit^{node}_n\). A naive Gauss-Seidel diagonalization may create an artificial first-mover advantage: the investor solved first could buy all 100 MW at every node, leaving little or no connection capacity for later investors. This would be an artifact of the solution algorithm rather than a meaningful market outcome.

Therefore, the next modeling step is to test allocation mechanisms that avoid pure sequential first-mover capture. Candidate approaches:

- use parallel Gauss-Jacobi updates so investors respond to the same previous-iteration market state;
- use damping/projection after each iteration so total nodal power remains feasible;
- represent nodal access as a shared equilibrium constraint with shadow prices rather than a simple first-come-first-served residual limit;
- compare sequential and parallel update rules explicitly to show whether allocation results are algorithm-dependent.

Working upper-level objective for investor \(i\):

$$
\max NPV_i
=
\sum_{t,n}
\lambda^{LMP}_{n,t}
\left(P^{discharge}_{i,n,t}-P^{charge}_{i,n,t}\right)
-
Cost^{degrad}_i
-
CAPEX^{daily}_i .
$$

Working lower-level model: deterministic spot-market clearing with
\(P_{g,t}\), \(P^{charge}_{i,n,t}\), \(P^{discharge}_{i,n,t}\),
\(SOC_{i,n,t}\), \(NetInjection_{n,t}\), system balance, and PTDF line constraints.

Current code/documentation status:

- `Overleaf_Alex/model_extension.tex` has been reduced to a deterministic spot-market formulation.
- `prepare_data.py` now reads only spot-market input sheets, computes PTDF data, and writes spot-only processed JSON.
- `data/input/bess_epec_inputs.xlsx` has been cleaned to remove reserve-market sheets, scenario sheets, the storage reserve-offer column, and the reserve-deficit penalty setting.
- `data/processed/market_data.json` contains only deterministic spot-market fields.
- `primal_market_clearing_model.py` has been reduced to a deterministic spot-market primal LP and loads benchmark data from processed JSON instead of hard-coded model data.

Equation numbering note: the thesis source uses unlabelled `equation` environments in Chapter 3. The equation numbers below are inferred from their order in `Methodology.tex`, matching the review references supplied for Eq. 3.5, Eq. 3.7, Eq. 3.8, Eq. 3.10, Eq. 3.13, and Eq. 3.14.

## 1. Thesis Summary

### Core Research Question and Objectives

The thesis asks:

> What insights into strategic investor behavior can be gained by shifting from a traditional central-planner cost-minimization approach to a decentralized, game-theoretic EPEC framework when determining optimal BESS capacities?

The stated objective is to determine optimal BESS siting and sizing in both MW and MWh at nodal grid level while combining economic profitability with technical grid constraints. The key methodological departure is to replace central-planner storage allocation with strategic, profit-maximizing investors interacting through market prices and shared grid capacity.

The model is positioned as a proof-of-concept for multi-service BESS allocation, co-optimizing day-ahead arbitrage and aFRR reserve provision under a DC-OPF market clearing representation.

### Original EPEC/MPEC Bilevel Structure

The thesis formulates an Equilibrium Problem with Equilibrium Constraints (EPEC) as a set of coupled Mathematical Programs with Equilibrium Constraints (MPECs). Each strategic investor solves one MPEC while anticipating the ISO market-clearing response.

Upper Level: strategic BESS investor $i$

- Decision variables: installed power capacity $X_{power,i,n}$ in MW and energy capacity $X_{energy,i,n}$ in MWh at each node $n$.
- Objective: maximize investor-specific daily NPV from spot arbitrage and, in the expanded model, aFRR capacity revenue, net of degradation and annualized daily CAPEX.
- Heterogeneity: investors differ by WACC / discount rate $r_i \in \{8\%,12\%,15\%,20\%\}$.

Lower Level: ISO / market clearing

- Objective: minimize conventional generation dispatch cost using a DC-OPF-style formulation.
- Outputs: dispatch, storage operation, nodal prices / LMPs $\lambda_{n,t}$, and network congestion signals.
- In the thesis text, the lower-level primal problem is replaced by its KKT optimality conditions and embedded in the upper-level MPEC.

### Key Mathematical Formulation in the Original Thesis

The thesis equations in Chapter 3 can be summarized as follows.

Upper-level spot-only investor objective, Eq. 3.1:

$$
\max_{X_{power},X_{energy}} NPV_i =
\sum_{t,n} \lambda_{n,t}\left(P_{discharge,i,n,t}-P_{charge,i,n,t}\right)
- CAPEX_{daily,i}
$$

Capital recovery factor, Eq. 3.2:

$$
CRF_i = \frac{r_i(1+r_i)^{Lifetime}}{(1+r_i)^{Lifetime}-1}
$$

Daily annualized CAPEX, Eq. 3.3:

$$
CAPEX_{daily,i}
= \frac{CRF_i}{365.25}
\sum_n
\left(
C_{power}X_{power,i,n}+C_{energy}X_{energy,i,n}
\right)
$$

Shared nodal connection limit, Eq. 3.4:

$$
X_{power,i,n}+\sum_{j\ne i}X^{fixed}_{power,j,n} \le Limit_n
$$

Minimum energy-to-power ratio, Eq. 3.5:

$$
X_{energy,i,n} \ge Ratio_{min}X_{power,i,n}
$$

Lower-level ISO objective, Eq. 3.6:

$$
\min \sum_{t,g} MC_g P_{gen,g,t}
$$

Nodal balance, Eq. 3.7:

$$
\sum_{g\in G_n}P_{gen,g,n,t}
+P_{wind/pv,n,t}
+\sum_i(P_{discharge,n,t}-P_{charge,n,t})
-D_{n,t}
-NetExport_{n,t}
=0
$$

PTDF / transmission formulation, Eq. 3.8:

$$
\begin{aligned}
NetInjection_{n,t}
&=
\sum_{g\in G_n}P_{gen,g,n,t}
+P_{wind/pv,n,t}
+\sum_i(P_{discharge,i,n,t}-P_{charge,i,n,t})
-D_{n,t} \\
Flow_{l,t}
&=
\sum_{n\in N}PTDF_{l,n}NetInjection_{n,t} \\
|Flow_{l,t}| &\le Limit_l
\end{aligned}
$$

Generator limits, Eq. 3.9:

$$
0 \le P_{gen,g,t} \le P_{max,g}
$$

Storage technical constraints and SOC balance, Eq. 3.10:

$$
\begin{aligned}
0 &\le P_{charge} \le X_{power} \\
0 &\le P_{discharge} \le X_{power} \\
SOC_t &= SOC_{t-1}+\eta P_{charge,t}-\frac{1}{\eta}P_{discharge,t} \\
0 &\le SOC_t \le X_{energy} \\
SOC_{t=0} &= SOC_{t=24}
\end{aligned}
$$

Linear degradation cost, Eq. 3.11:

$$
Cost_{degrad}
=
\sum_{t\in T}\sum_{n\in N}
0.5 C_{degrad}
\left(P_{charge,i,n,t}+P_{discharge,i,n,t}\right)
$$

Expanded multi-service upper-level objective, Eq. 3.12:

$$
\max_{X_{power},X_{energy}} NPV_i
=
Revenue_{spot}+Revenue_{afrr}-Cost_{degrad}-CAPEX_{daily}
$$

aFRR revenue, Eq. 3.13:

$$
Revenue_{afrr}
=
\sum_{b,n}H_{block}
\left(
\lambda_{afrr,up,b}R_{up,i,n,b}
+\lambda_{afrr,down,b}R_{down,i,n,b}
\right)
$$

Cournot inverse-demand aFRR prices, Eq. 3.14:

$$
\begin{aligned}
\lambda_{afrr,up,b}
&=
Price_{cap}
\left(
1-
\frac{
\sum_nR_{up,i,n,b}+\sum_{j\ne i,n}R^{fixed}_{up,j,n,b}
}{
Demand_{up,b}
}
\right) \\
\lambda_{afrr,down,b}
&=
Price_{cap}
\left(
1-
\frac{
\sum_nR_{down,i,n,b}+\sum_{j\ne i,n}R^{fixed}_{down,j,n,b}
}{
Demand_{down,b}
}
\right)
\end{aligned}
$$

Reserve/spot power-capacity coupling, Eq. 3.15:

$$
\begin{aligned}
P_{charge,t}+R_{down,b} &\le X_{power} \\
P_{discharge,t}+R_{up,b} &\le X_{power}
\end{aligned}
$$

Damped diagonalization update, Eq. 3.16:

$$
X_{new}=(1-\alpha)X_{old}+\alpha X_{calculated}
$$

Residual demand formulation, Eq. 3.17:

$$
D_{adj,n,t}
=
D_{n,t}
+\sum_{j\ne i}
\left(P^{fixed}_{charge,j,n,t}-P^{fixed}_{discharge,j,n,t}\right)
$$

KKT/MPEC transformation:

- The thesis states that the ISO lower-level problem is replaced by its KKT optimality conditions and embedded in each investor's upper-level problem.
- The thesis source does not explicitly list the full stationarity, dual feasibility, and complementarity equations in TeX.
- Complementarity is relaxed using a tolerance $\epsilon$, stated as $\epsilon = 10.0$ in the synthesis and methodology discussion.
- Earlier KKT-based aFRR deficit clearing was replaced by the continuous Cournot inverse demand curve because the solver exploited complementarity slack to create a reserve price-dictator outcome.

### Diagonalization Algorithm

The EPEC is solved iteratively through diagonalization. In each iteration, investor $i$ solves its MPEC while other investors' decisions are fixed. The thesis discusses both:

- Gauss-Seidel: sequential updates, where later investors see earlier investors' current-iteration decisions.
- Gauss-Jacobi: parallel updates, where all investors solve against the previous global market state.

The implemented approach is described as parallel Gauss-Jacobi using Python `ProcessPoolExecutor`. Damping with $\alpha=0.25$ is used to avoid oscillatory over- and under-investment. Convergence is measured by the maximum change in installed power capacity, `Max Delta`, with a threshold of 0.5 MW.

Solver details from the thesis:

- Pyomo used for model construction.
- Ipopt used as NLP solver.
- CoinHSL / MA97 used for linear algebra acceleration.
- `OMP_NUM_THREADS = 16` used for multithreading.
- Four co-optimized MPECs including aFRR reportedly solved in about 5 to 15 seconds per iteration, with the overall EPEC converging in roughly 3 minutes.

### Test System Description

The model uses a modified 5-bus network over one representative 24-hour day with hourly resolution.

Network:

| Node | Connection limit | Resources shown in grid figure |
|---|---:|---|
| N1 | 100 MW | Base-load generator, wind |
| N2 | 100 MW | Load / network node |
| N3 | 100 MW | PV |
| N4 | 100 MW | Load / network node |
| N5 | 100 MW | Peak-load generator |

Transmission line limits visible in Figure `Figure_GridTopology.pdf`:

| Line | Limit |
|---|---:|
| N1-N2 | 400 MW |
| N2-N3 | 400 MW |
| N1-N4 | 400 MW |
| N1-N5 | 400 MW |
| N3-N4 | 240 MW |
| N4-N5 | 240 MW |

Input profiles and generation:

- Representative day: 2025-09-11 from APG generation/load data.
- System-wide peak load: 1200 MW.
- Load distribution: 90% in central nodes N2, N3, N4 and 10% in peripheral nodes N1, N5.
- Wind: peak capacity 370 MW at N1, strongest at night and early morning.
- PV: peak capacity 430 MW at N3, daytime bell curve.
- Conventional generation costs in calibrated model: base-load 40 EUR/MWh, peak-load 80 EUR/MWh.
- Literature comparison noted in text: Devine and Siddiqui parameters with base-load 48.87 EUR/MWh and peak-load 63.38 EUR/MWh.

BESS / investor parameters:

| Parameter | Value in thesis |
|---|---:|
| Investors | 4 |
| WACC / discount rates | 8%, 12%, 15%, 20% |
| Project lifetime | 15 years |
| $C_{power}$ | 6,600 EUR/MW, annualized daily parameter as written in thesis |
| $C_{energy}$ | 18,800 EUR/MWh, annualized daily parameter as written in thesis |
| Round-trip efficiency parameter $\eta$ | 93.6% |
| Degradation cost | 15 EUR/MWh throughput |
| Minimum E/P ratio | 2 hours |
| aFRR block duration | 4 hours |
| aFRR price cap | 3,000 EUR/MW |
| Nodal BESS connection limit | 100 MW per node |

### Key Results

Algorithmic convergence:

- Energy-only scenario converged in 3 iterations.
- Multi-service scenario also converged in 3 iterations.
- Both used damping factor $\alpha=0.25$ and convergence tolerance 0.5 MW.

Table 4.1, aggregated BESS investment metrics:

| Investor | WACC | Energy-only MW | Energy-only MWh | Energy-only E/P | Multi-service MW | Multi-service MWh | Multi-service E/P |
|---|---:|---:|---:|---:|---:|---:|---:|
| Investor 1 | 8.0% | 122.22 | 484.32 | 3.96 | 124.90 | 2648.58 | 21.21 |
| Investor 2 | 12.0% | 126.57 | 2392.83 | 18.91 | 123.62 | 291.13 | 2.35 |
| Investor 3 | 15.0% | 124.42 | 758.93 | 6.10 | 123.99 | 716.09 | 5.78 |
| Investor 4 | 20.0% | 123.90 | 595.77 | 4.81 | 124.57 | 625.89 | 5.02 |

Table 4.2, daily revenue breakdown:

| Investor | WACC | Energy-only arbitrage | Energy-only aFRR | Multi-service arbitrage | Multi-service aFRR |
|---|---:|---:|---:|---:|---:|
| Investor 1 | 8.0% | 64,660.66 EUR | 0.00 EUR | -2,574.26 EUR | 332,852.44 EUR |
| Investor 2 | 12.0% | 68,297.04 EUR | 0.00 EUR | 4,521.00 EUR | 292,986.68 EUR |
| Investor 3 | 15.0% | 52,641.06 EUR | 0.00 EUR | 3,759.11 EUR | 293,034.14 EUR |
| Investor 4 | 20.0% | 52,686.17 EUR | 0.00 EUR | 3,532.13 EUR | 293,041.04 EUR |

Revenue interpretation:

- Energy-only operation yields positive arbitrage revenue for all investors.
- Multi-service operation makes aFRR capacity revenue dominant.
- Investor 1 accepts negative spot arbitrage in the multi-service case because reserve revenue is much larger.
- For Investors 2-4, multi-service aFRR revenue is roughly 65-83 times larger than remaining arbitrage revenue. Investor 1 has negative arbitrage and 332,852 EUR/day of aFRR revenue.

Nodal price and grid behavior:

- Energy-only case creates strong locational price differences.
- Node 3 exhibits scarcity pricing during morning/evening peaks, with prices above 600 EUR/MWh in the thesis discussion.
- Node 1 can fall to 40 EUR/MWh during system peaks because congestion traps cheap base-load/wind locally.
- Multi-service operation synchronizes BESS behavior across nodes because aFRR obligations dominate local arbitrage signals.

### Main Conclusions

The thesis concludes that strategic investors:

- aggressively distribute BESS capacity across all nodes;
- exhaust the 100 MW/node connection limits;
- share power capacity in a balanced oligopoly under the Cournot reserve formulation;
- strongly prioritize aFRR capacity revenue over spot arbitrage when reserve prices are lucrative;
- oversize energy capacity to preserve some arbitrage ability while inverter capacity is reserved for aFRR.

The central economic conclusion is that optimal BESS sizing and operation are driven less by renewable availability alone and more by the regulatory/market design of the services in which storage participates.

The most important empirical anomaly for the follow-up project is the extreme E/P range: 2.35 to 21.21 hours in the multi-service case, with Investor 1 reaching 21.21 hours and Investor 2 reaching 18.91 hours in the energy-only case.

## 2. Critical Review of the Original Methodology

### Weakness 1: aFRR Market Not Cleared in the Lower Level

The aFRR clearing price is not derived from a genuine reserve-capacity market-clearing constraint in the ISO lower-level problem. The lower-level problem in the thesis explicitly contains:

- ISO dispatch objective, Eq. 3.6;
- nodal balance, Eq. 3.7;
- PTDF / line limits, Eq. 3.8;
- generator limits, Eq. 3.9;
- storage constraints, Eq. 3.10.

It does not contain an aFRR capacity balance such as:

$$
\sum_{i,n}R_{i,n,b} \ge Demand_b
$$

Instead, aFRR prices are imposed through the upper-level Cournot inverse-demand equation, Eq. 3.14:

$$
\lambda_{afrr,up,b}
=
Price_{cap}
\left(
1-
\frac{
\sum_nR_{up,i,n,b}+\sum_{j\ne i,n}R^{fixed}_{up,j,n,b}
}{
Demand_{up,b}
}
\right)
$$

and analogously for $\lambda_{afrr,down,b}$.

This conflicts with the statement immediately before Eq. 3.14 that $\lambda_{afrr,up,b}$ and $\lambda_{afrr,down,b}$ are "derived from the Lower Level problem as dual variables." In the documented formulation, no lower-level aFRR balance constraint exists, so $\lambda_{afrr}$ cannot emerge as a KKT dual variable of a reserve-capacity clearing problem.

Implication: the reserve market is not truly co-cleared by the ISO. It is a closed-form strategic price function bolted onto the investor objective. This weakens the economic interpretation of reserve prices and prevents analysis of congestion, scarcity, or activation risk within the reserve market clearing itself.

### Weakness 2: Two Independent Lambdas for Up/Down Regulation

The thesis models positive and negative reserve separately:

- $R_{up,i,n,b}$ and $R_{down,i,n,b}$ in aFRR revenue, Eq. 3.13.
- $\lambda_{afrr,up,b}$ and $\lambda_{afrr,down,b}$ as separate Cournot prices, Eq. 3.14.
- $Demand_{up,b}$ and $Demand_{down,b}$ as separate denominators in Eq. 3.14.
- Separate capacity coupling in Eq. 3.15:

$$
P_{charge,t}+R_{down,b} \le X_{power}
$$

$$
P_{discharge,t}+R_{up,b} \le X_{power}
$$

This is inconsistent with the real Austrian APG aFRR capacity product if the relevant procurement is a single symmetric capacity band at one clearing price. The independent up/down curves let the model earn two separate reserve capacity revenues without a single symmetric commitment variable forcing the physical trade-off.

Implication: the model can double-count reserve revenue from simultaneous up/down provision. Because $R_{up}$ and $R_{down}$ are priced independently, there is no single reserve-band scarcity condition binding the battery's symmetric service commitment.

The follow-up model should replace $R_{up}$ and $R_{down}$ with a single first-stage symmetric capacity commitment $R_{i,n,b}$ and a single reserve capacity clearing price $\lambda_{aFRR,b}$.

### Weakness 3: No aFRR Activation in the SOC Dynamics

This is the core physical flaw. The SOC balance in Eq. 3.10 contains only spot charging and discharging:

$$
SOC_t
=
SOC_{t-1}
+\eta P_{charge,t}
-\frac{1}{\eta}P_{discharge,t}
$$

There are no activation terms such as upward activation $A_{up}$ or downward activation $A_{down}$.

As a result, committing capacity to aFRR does not change the battery state of charge. The only aFRR-related physical constraints are the power headroom constraints in Eq. 3.15. Energy feasibility under activation is not modeled.

Implication: a battery can reserve nearly its full inverter capacity for aFRR and still execute spot-market energy cycles without the SOC consequences of reserve activation. The model text later says batteries must maintain energy reservoirs for potential reserve activation, but Eq. 3.10 does not implement activation in the SOC state equation. This is the likely root cause of the unrealistic E/P ratios and the apparent ability to stack large aFRR capacity payments with shallow physical energy consequences.

### Weakness 4: No Upper Bound on the Energy-to-Power Ratio

The thesis enforces only a minimum E/P ratio in Eq. 3.5:

$$
X_{energy,i,n} \ge Ratio_{min}X_{power,i,n}
$$

with $Ratio_{min}=2$ hours. There is no upper bound:

$$
X_{energy,i,n} \le Ratio_{max}X_{power,i,n}
$$

Real utility-scale Li-ion, especially LFP systems, allow independent sizing of battery cells and PCS more than household BESS products do. However, independence is still bounded by physical C-rate limits, PCS sizing, thermal design, project bankability, grid connection approval, warranty constraints, and commercial product envelopes. A practical utility-scale range is roughly 2-8 hours for this research context.

Implication: the optimization can exploit the low energy-capacity cost parameter and build physically implausible reservoirs. Investor 1's multi-service E/P ratio of 21.21 hours and Investor 2's energy-only E/P ratio of 18.91 hours are not realistic utility-scale Li-ion BESS configurations.

The follow-up model should enforce both:

$$
Ratio_{min}X_{power,i,n} \le X_{energy,i,n}
$$

and:

$$
X_{energy,i,n} \le Ratio_{max}X_{power,i,n}
$$

with $Ratio_{max}=8$ hours as a defensible initial bound.

### Weakness 5: Perfect Foresight / Deterministic Activation

The thesis explicitly discards stochasticity and balancing energy in Section 3.4.4 ("Discarded model features"), stating that perfect foresight is assumed and stochastic optimization for forecast errors and balancing energy activation was excluded for tractability. The synthesis repeats this limitation under "Perfect Foresight (Deterministic Approach)."

This matters specifically for aFRR because reserve capacity commitment is valuable precisely because it may be activated under uncertain system imbalance. In reality, a BESS must preserve safe SOC margins against uncertain upward/downward activation. Activation uncertainty affects:

- feasible reserve capacity;
- SOC trajectory;
- degradation cost;
- opportunity cost against spot arbitrage;
- expected activation-energy revenue/cost;
- optimal power/energy sizing.

Implication: the deterministic model does not price activation risk. It treats reserve capacity as a capacity payment with power headroom but without stochastic energy obligations. The follow-up should model aFRR activation as uncertainty in a two-stage stochastic EPEC.

### Weakness 6: Review Needed: PTDF / DC-OPF Formulation

The thesis Eq. 3.7 nodal balance is written as:

$$
\sum_{g\in G_n}P_{gen,g,n,t}
+P_{wind/pv,n,t}
+\sum_i(P_{discharge,n,t}-P_{charge,n,t})
-D_{n,t}
-NetExport_{n,t}
=0
$$

Eq. 3.8 then defines:

$$
NetInjection_{n,t}
=
\sum_{g\in G_n}P_{gen,g,n,t}
+P_{wind/pv,n,t}
+\sum_i(P_{discharge,i,n,t}-P_{charge,i,n,t})
-D_{n,t}
$$

and:

$$
Flow_{l,t}
=
\sum_{n\in N}PTDF_{l,n}NetInjection_{n,t}
$$

The issue is not conclusively provable from the LaTeX alone. However, the write-up is ambiguous. If Eq. 3.7 is implemented as a nodal equality that forces each node's generation plus renewables plus BESS net discharge minus demand minus exports to zero, and Eq. 3.8 uses effectively the same nodal expression, then $NetInjection_{n,t}$ could be collapsed through a separately defined $NetExport_{n,t}$ in a way that risks trivial or inconsistent flow formation.

The standard DC-OPF/PTDF structure should instead be:

$$
NetInjection_{n,t,s}
=
\sum_g P_{gen,g,n,t,s}
+P_{wind/pv,n,t}
+\sum_i
\left(
P_{discharge,i,n,t,s}
-P_{charge,i,n,t,s}
-A_{up,i,n,t,s}
+A_{down,i,n,t,s}
\right)
-D_{n,t}
$$

with nodal injections not forced to zero individually. Then:

$$
\sum_n NetInjection_{n,t,s}=0
$$

and:

$$
Flow_{l,t,s}=\sum_nPTDF_{l,n}NetInjection_{n,t,s}
$$

Nodal prices can then be recovered as:

$$
\lambda_{n,t,s}
=
\lambda_{sys,t,s}
+\sum_l PTDF_{l,n}\mu_{l,t,s}
$$

where $\mu_{l,t,s}$ is the congestion dual contribution.

Important repository limitation: no Pyomo implementation is present in `Overleaf_Daniel`, so the actual code-level DC-OPF cannot be verified from this project. The thesis results show non-trivial nodal price dynamics and line loading figures, which suggests the implementation may have produced non-zero flows and LMP differences. However, before extending the model, the actual implementation must be inspected to confirm whether system balance, nodal injection, PTDF flow, and nodal price recovery are coded correctly.

### Cost Calibration Note

The thesis uses $C_{energy}=18{,}800$ EUR/MWh and $C_{power}=6{,}600$ EUR/MW, described as daily annualized cost parameters derived from utility-scale Li-ion projections. The absolute magnitude and the relationship between energy and power costs should be rechecked before extension.

Relative to typical utility-scale Li-ion/LFP capital costs around 2025, rough non-annualized benchmarks are closer to:

- cells / battery energy capacity: 200,000-300,000 EUR/MWh;
- PCS / power capacity: 50,000-100,000 EUR/MW.

If the thesis parameters are already annualized daily equivalents, the documentation should make the conversion explicit and ensure unit consistency. If they are being interpreted as CAPEX inputs directly, they are too low. In either case, the low effective energy cost relative to modeled reserve revenue likely contributes to oversized MWh capacity and extreme E/P ratios.

## 3. Superseded Previous Extension Plan: Stochastic Two-Stage EPEC

Important: this section is retained only as historical context from the earlier project direction. After the supervisor meeting on 2026-07-08, this stochastic aFRR extension is no longer the active implementation plan. The active plan is the spot-market BESS competition formulation documented at the top of this file.

### Modeling Approach

The proposed extension is a two-stage stochastic MPEC/EPEC.

First-stage variables are decided before activation uncertainty resolves and have no scenario index:

$$
X_{power,i,n},\quad X_{energy,i,n},\quad R_{i,n,b}
$$

where $R_{i,n,b}$ is a single symmetric aFRR capacity commitment replacing $R_{up,i,n,b}$ and $R_{down,i,n,b}$.

Second-stage variables are scenario-indexed and decided after uncertainty resolves:

$$
P_{charge,i,n,t,s},\quad
P_{discharge,i,n,t,s},\quad
A_{up,i,n,t,s},\quad
A_{down,i,n,t,s},\quad
SOC_{i,n,t,s}
$$

Non-anticipativity is enforced implicitly through indexing:

- first-stage variables appear once and are shared across all scenarios;
- second-stage variables are replicated for each scenario $s\in S$;
- first-stage and second-stage decisions are linked through power-capacity, SOC, activation, and E/P constraints.

No explicit non-anticipativity constraints are needed unless future implementation creates scenario-indexed copies of first-stage variables.

### Illustrative Scenario Set

The scenario probabilities and activation fractions below are placeholders. They must be calibrated from historical APG aFRR activation data.

| Scenario | Description | Probability $\pi_s$ | $A_{up}$ | $A_{down}$ |
|---|---|---:|---:|---:|
| $s_1$ | No activation | 0.40 | 0 | 0 |
| $s_2$ | Light upward | 0.25 | $0.3R_b$ | 0 |
| $s_3$ | Heavy upward | 0.20 | $R_b$ | 0 |
| $s_4$ | Downward | 0.15 | 0 | $0.5R_b$ |

In implementation, scenario activation should be represented either as exogenous activation factors:

$$
A_{up,i,n,t,s}=\rho^{up}_{t,s}R_{i,n,b(t)}
$$

and:

$$
A_{down,i,n,t,s}=\rho^{down}_{t,s}R_{i,n,b(t)}
$$

or as bounded recourse variables with scenario-specific realized activation requirements. The first option is simpler and avoids giving the BESS control over whether it is activated.

### Full Extended Lower-Level Problem: ISO Clearing

For each time $t$, scenario $s$, node $n$, generator $g$, line $l$, and aFRR block $b$, the ISO minimizes expected generation dispatch cost:

$$
\min
\sum_{s\in S}\pi_s
\sum_{t\in T}\sum_{g\in G}
MC_gP_{gen,g,t,s}
$$

Subject to the following constraints.

1. Nodal surplus/deficit definition, not forced to zero individually:

$$
NetInjection_{n,t,s}
=
\sum_{g\in G_n}P_{gen,g,n,t,s}
+P_{wind/pv,n,t}
+\sum_i
\left(
P_{discharge,i,n,t,s}
-P_{charge,i,n,t,s}
-A_{up,i,n,t,s}
+A_{down,i,n,t,s}
\right)
-D_{n,t}
$$

Interpretation: upward activation means the BESS injects additional energy into the grid and depletes SOC. If $A_{up}$ is represented as extra delivered upward balancing energy, the sign in net injection can be written as $+A_{up}$. The sign convention above follows the user-supplied target formulation; it must be aligned consistently with the SOC equation and market settlement convention during implementation.

2. System-wide power balance:

$$
\sum_n NetInjection_{n,t,s}=0
\qquad \forall t,s
$$

Dual:

$$
\lambda_{sys,t,s}
$$

3. Line flow via PTDF:

$$
Flow_{l,t,s}
=
\sum_nPTDF_{l,n}NetInjection_{n,t,s}
\qquad \forall l,t,s
$$

4. Thermal limits:

$$
-Limit_l \le Flow_{l,t,s} \le Limit_l
\qquad \forall l,t,s
$$

Nodal price recovery:

$$
\lambda_{n,t,s}
=
\lambda_{sys,t,s}
+\sum_lPTDF_{l,n}\mu_{l,t,s}
$$

where $\mu_{l,t,s}$ is the congestion dual contribution. In code, use separate upper/lower flow duals and combine them consistently.

5. Global aFRR capacity balance:

$$
\sum_i\sum_nR_{i,n,b} \ge Demand_b
\qquad \forall b
$$

Dual:

$$
\lambda_{aFRR,b}
$$

This is the key replacement for the exogenous Cournot price in Eq. 3.14. If strategic withholding is still desired, it should enter through offer-curve/bid formulation or residual demand, not through an externally imposed price curve masquerading as a dual.

6. SOC dynamics with activation terms:

$$
SOC_{i,n,t,s}
=
SOC_{i,n,t-1,s}
+\eta P_{charge,i,n,t,s}
-\frac{1}{\eta}P_{discharge,i,n,t,s}
-\frac{1}{\eta}A_{up,i,n,t,s}
+\eta A_{down,i,n,t,s}
\qquad \forall i,n,t,s
$$

This is the core physical fix. Upward activation depletes SOC; downward activation charges the battery.

7. SOC bounds:

$$
0 \le SOC_{i,n,t,s} \le X_{energy,i,n}
\qquad \forall i,n,t,s
$$

8. Power capacity with one symmetric reserve band:

$$
P_{discharge,i,n,t,s}+R_{i,n,b(t)}
\le
X_{power,i,n}
\qquad \forall i,n,t,s
$$

$$
P_{charge,i,n,t,s}+R_{i,n,b(t)}
\le
X_{power,i,n}
\qquad \forall i,n,t,s
$$

9. Activation bounds:

$$
0 \le A_{up,i,n,t,s} \le R_{i,n,b(t)}
\qquad \forall i,n,t,s
$$

$$
0 \le A_{down,i,n,t,s} \le R_{i,n,b(t)}
\qquad \forall i,n,t,s
$$

If activation is exogenous, replace these with equalities to realized activation factors:

$$
A_{up,i,n,t,s}=\rho^{up}_{t,s}R_{i,n,b(t)}
$$

$$
A_{down,i,n,t,s}=\rho^{down}_{t,s}R_{i,n,b(t)}
$$

10. Generator limits:

$$
0 \le P_{gen,g,t,s} \le P_{max,g}
\qquad \forall g,t,s
$$

11. SOC periodicity:

$$
SOC_{i,n,0,s}=SOC_{i,n,24,s}
\qquad \forall i,n,s
$$

12. E/P bounds:

$$
Ratio_{min}X_{power,i,n}
\le
X_{energy,i,n}
\le
Ratio_{max}X_{power,i,n}
\qquad \forall i,n
$$

Suggested initial values:

$$
Ratio_{min}=2,\qquad Ratio_{max}=8
$$

based on a practical utility-scale LFP technology envelope.

### Upper-Level Objective: Expected NPV Maximization

For investor $i$, replace the deterministic objective with expected NPV:

$$
\begin{aligned}
\max NPV_i
=&
\sum_{b,n}
H_{block}\lambda_{aFRR,b}R_{i,n,b}
\\
&+
\sum_{s\in S}\pi_s
\left[
\sum_{t,n}
\lambda_{n,t,s}
\left(
P_{discharge,i,n,t,s}
-P_{charge,i,n,t,s}
\right)
\right.
\\
&\left.
\quad+
\sum_{b,n}
\lambda_{E,b,s}A_{up,i,n,b,s}
\right]
\\
&-
CAPEX_{daily,i}
-
\sum_{s\in S}\pi_sCost_{degrad,i,s}
\end{aligned}
$$

Capacity payment and CAPEX remain outside the expectation because $R_{i,n,b}$, $X_{power,i,n}$, and $X_{energy,i,n}$ are first-stage decisions. Spot revenue, activation-energy revenue, and degradation are scenario-weighted second-stage outcomes.

Scenario-specific degradation:

$$
Cost_{degrad,i,s}
=
\sum_{t,n}
0.5C_{degrad}
\left(
P_{charge,i,n,t,s}
+P_{discharge,i,n,t,s}
+A_{up,i,n,t,s}
+A_{down,i,n,t,s}
\right)
$$

This extends Eq. 3.11 so activation throughput also causes degradation.

Activation energy price $\lambda_{E,b,s}$ can initially be modeled as an exogenous scenario parameter for tractability. A later version can endogenize activation energy price if computational performance allows.

### MPEC/KKT Form of the Extension

For each investor $i$, the lower-level stochastic ISO problem should again be transformed into KKT conditions and embedded in the upper-level problem:

- primal feasibility: all lower-level constraints above;
- dual feasibility: sign restrictions for inequality duals;
- stationarity: derivative of lower-level Lagrangian with respect to $P_{gen}$, flows/injections if represented as variables, BESS dispatch variables, activation variables, and SOC variables;
- complementarity: each inequality slack times its dual equals zero, or is relaxed as in the original thesis if Ipopt remains the solver.

Because stochastic scenarios replicate most lower-level variables and constraints, the KKT system grows roughly linearly with $|S|$. With four scenarios, expect approximately a 4x increase in dispatch/SOC/activation variables and their associated KKT equations, while first-stage investment variables remain unreplicated.

### Solution Algorithm

Retain the original thesis algorithmic backbone:

- parallel Gauss-Jacobi diagonalization;
- damping with $\alpha \approx 0.25$;
- convergence based on maximum change in installed power capacity and, preferably, reserve commitment;
- residual demand formulation.

Extension-specific changes:

- compute residual demand per scenario:

$$
D_{adj,n,t,s}
=
D_{n,t}
+\sum_{j\ne i}
\left(
P^{fixed}_{charge,j,n,t,s}
-P^{fixed}_{discharge,j,n,t,s}
+A^{fixed}_{up,j,n,t,s}
-A^{fixed}_{down,j,n,t,s}
\right)
$$

with sign convention checked against the final net-injection equation;

- carry competitor reserve commitments $R^{fixed}_{j,n,b}$ as first-stage fixed variables in each investor's MPEC;
- update first-stage capacities and reserve commitments with damping;
- compare convergence across both $X_{power}$ and $R$.

Computational expectations:

- Four activation scenarios imply roughly 4x problem size for time-indexed recourse and KKT blocks.
- CoinHSL/MA97 and Ipopt tuning may be required.
- If the extended NLP becomes unstable, consider decomposition: Benders-like decomposition, progressive hedging for scenario structure, or a simplified lower-level stochastic dispatch approximation before full MPEC embedding.

### Expected Outcome Versus Original Thesis

| Metric | Original thesis | Expected after extension |
|---|---|---|
| E/P ratio range | 2.35-21.21 hours in multi-service comparison context | Approximately 2-8 hours due to explicit bound and activation feasibility |
| aFRR vs arbitrage revenue | aFRR dominates, often by roughly two orders of magnitude | More balanced because reserve commitment consumes stochastic SOC headroom |
| $\lambda_{aFRR}$ origin | Exogenous Cournot inverse-demand formula, Eq. 3.14 | Endogenous dual of lower-level aFRR capacity balance |
| Reserve product | Separate $R_{up}$ and $R_{down}$ with separate prices | One symmetric band $R$ with one price |
| SOC realism | No reserve activation terms in Eq. 3.10 | Scenario-dependent SOC with upward/downward activation |
| Physical sizing realism | No E/P upper bound | E/P constrained by utility-scale technology envelope |
| Main investment effect | Energy oversizing to preserve arbitrage while reserving inverter capacity | Lower reserve commitments and/or larger but bounded energy capacity, with explicit activation-risk trade-off |

## 4. Superseded aFRR/Stochastic Open Questions

Important: these open questions belong to the previous aFRR/stochastic extension and are not the immediate implementation target after the 2026-07-08 supervisor meeting. Keep them only as historical notes unless the project later returns to reserve-market modeling.

1. Confirm actual DC-OPF implementation.

The Overleaf repository does not include the Pyomo code. Before extending the model, inspect the implementation and verify that:

- $NetInjection_{n,t,s}$ is defined as nodal surplus/deficit and not forced to zero node-by-node;
- a separate system-wide balance $\sum_nNetInjection_{n,t,s}=0$ exists;
- line flows use $Flow_{l,t,s}=\sum_nPTDF_{l,n}NetInjection_{n,t,s}$;
- line constraints produce non-trivial congestion duals;
- nodal prices are recovered consistently from system energy balance and congestion duals.

2. Calibrate realistic aFRR activation scenarios from APG transparency data.

The illustrative probabilities in this document are placeholders. Historical APG activation data from `https://markt.apg.at` should be used to estimate:

- probability of no/light/heavy upward activation;
- probability and magnitude of downward activation;
- temporal clustering within 4-hour reserve blocks;
- correlation between activation events, load, renewable output, and spot prices;
- duration and energy content of activation calls.

3. Decide how to model activation energy settlement.

Open modeling choice:

- treat $\lambda_{E,b,s}$ as exogenous scenario data for tractability; or
- include activation energy clearing endogenously in the lower-level problem.

The first option is likely preferable for the first extension because the core methodological fix is activation-dependent SOC feasibility, not full balancing-energy price formation.

4. Decide whether aFRR capacity price should be fully endogenous.

A clean formulation makes $\lambda_{aFRR,b}$ the dual of:

$$
\sum_{i,n}R_{i,n,b}\ge Demand_b
$$

However, if strategic withholding must be preserved, a pure competitive reserve balance may collapse strategic price effects. Options include:

- retain EPEC strategic behavior through each investor's effect on the lower-level reserve-clearing KKT system;
- model reserve offers explicitly;
- use residual demand curves carefully, but avoid calling the result a lower-level dual if it is not one;
- start with competitive reserve clearing as a physically correct benchmark before reintroducing strategic market power.

5. Recalibrate CAPEX and E/P technology envelope.

Clarify whether $C_{power}=6{,}600$ EUR/MW and $C_{energy}=18{,}800$ EUR/MWh are annualized daily equivalents or raw CAPEX. Rebuild the CAPEX calculation from raw technology assumptions:

$$
CAPEX_{daily,i}
=
\frac{CRF_i}{365.25}
\sum_n
\left(
C^{raw}_{power}X_{power,i,n}
+C^{raw}_{energy}X_{energy,i,n}
\right)
$$

Then test sensitivity over plausible utility-scale LFP cost ranges and $Ratio_{max}\in\{6,8,10\}$.

6. Check sign convention for activation in net injection.

The SOC equation should unambiguously satisfy:

- upward activation depletes SOC;
- downward activation increases SOC.

The grid injection sign must be aligned with the market convention. If $A_{up}$ represents additional energy injected by BESS, it should increase net injection. If it represents energy withdrawn from the BESS reserve margin or a demand-side activation convention, signs may differ. Resolve this before coding.

7. Assess computational tractability on the existing 5-bus / 4-investor / 24-hour setup.

Before scaling to larger systems:

- implement 2 scenarios first, then 4;
- compare solve time, convergence iterations, complementarity residuals, and stability;
- track whether damping $\alpha=0.25$ remains sufficient;
- include convergence checks for $R_{i,n,b}$ as well as $X_{power,i,n}$;
- test whether KKT relaxation $\epsilon=10.0$ remains acceptable or needs tightening.

8. Preserve comparability with the original thesis.

For the follow-up research, run at least four cases:

| Case | Purpose |
|---|---|
| Original deterministic formulation | Baseline replication |
| Add E/P upper bound only | Isolate physical sizing correction |
| Add SOC activation terms with deterministic activation | Isolate activation-energy feasibility |
| Full stochastic two-stage EPEC | Measure combined uncertainty and market-clearing correction |

This decomposition will make it clear whether changes in E/P ratios and revenue mix are caused by technology bounds, reserve activation physics, stochasticity, or endogenous reserve price formation.
