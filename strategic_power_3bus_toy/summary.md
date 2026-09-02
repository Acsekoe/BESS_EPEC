# Strategic-power three-bus toy and proposed two-stage investment game

Date: 2026-09-01

## 1. Purpose

This toy removes the strategic access auction and studies market power through
the operation of already-installed batteries. It is intended to answer a
smaller and cleaner question before returning to the IEEE-9 model:

> Can battery investors profitably exercise market power by withholding hourly
> charging or discharging capability, and how does generation ownership change
> that incentive?

The present model is an operational game with fixed battery investment. The
proposed extension adds an earlier investment game and solves the operational
game as its continuation equilibrium.

## 2. Test system

The deterministic test case contains three buses, a triangular DC network,
six one-hour periods, five generators, and four batteries at the constrained
bus N3.

| Line | Limit |
|---|---:|
| N1--N2 | 40 MW |
| N2--N3 | 25 MW |
| N1--N3 | 30 MW |

| Generator | Bus | Linear cost | Capacity or profile |
|---|---|---:|---|
| G1_BASE | N1 | 25 EUR/MWh | 100 MW |
| G2_MID | N2 | 55 EUR/MWh | 65 MW |
| G3_PEAK | N3 | 110 EUR/MWh | 100 MW |
| RES_WIND_N1 | N1 | 0 EUR/MWh | (18, 15, 12, 10, 14, 20) MW |
| RES_PV_N3 | N3 | 0 EUR/MWh | (0, 15, 45, 20, 0, 0) MW |

Small positive quadratic generation slopes smooth dispatch and reduce price
degeneracy. Demand is concentrated increasingly at N3 during the peak, with
the largest N3 demand of 90 MW in hour 5.

## 3. Investors and portfolios

There are four investors matching the maintained real-model population. The
original aggregate toy fleet of 40 MW / 100 MWh is divided equally among them
so the ownership experiment does not also increase total storage capacity.

| Investor | Location | Installed power | Installed energy | Generation ownership |
|---|---|---:|---:|---|
| I1 | N3 | 10 MW | 25 MWh | None; merchant, 8% WACC |
| I2 | N3 | 10 MW | 25 MWh | None; merchant, 12% WACC |
| I3 | N3 | 10 MW | 25 MWh | 80% of wind and 20% of PV; wind-heavy |
| I4 | N3 | 10 MW | 25 MWh | 20% of wind and 80% of PV; solar-heavy |

All batteries have 92% efficiency, a 2.5-hour duration, a degradation cost
of 15 EUR/MWh, and the same technology-cost assumptions. I2 differs from I1
only through its 12% rather than 8% WACC. I3 and I4 receive renewable
generation rent according to the same 80/20 wind/PV split used in the real
model.

## 4. Present operational game

### 4.1 Fixed versus strategic quantities

Installed inverter power \(K_i=10\) MW and energy \(E_i=25\) MWh are fixed.
The investors do not currently choose location, MW, MWh, or duration.
Annualised battery capital cost is deducted from reported profit, but it is a
constant in the operational best response.

By default, each investor strategically selects an hourly discharge
availability ceiling:

\[
0\le a^{dis}_{it}\le K_i,
\qquad a^{ch}_{it}=K_i.
\]

With the optional `--two-sided` mode, charging availability is also strategic:

\[
0\le a^{ch}_{it}\le K_i,
\qquad 0\le a^{dis}_{it}\le K_i.
\]

Availability is an offer of physical capability, not a dispatch schedule. The
ISO chooses actual charging and discharging.

### 4.2 ISO clearing

For fixed availability offers, the ISO solves a convex quadratic market
clearing problem:

\[
\min_y C^{ISO}(y;a,K,E)
\]

subject to nodal balance, PTDF line limits, generation capacity, and

\[
0\le p^{ch}_{it}\le a^{ch}_{it},
\qquad
0\le p^{dis}_{it}\le a^{dis}_{it},
\]

\[
p^{ch}_{it}+p^{dis}_{it}\le K_i,
\]

\[
e_{it}=e_{i,t-1}+\eta p^{ch}_{it}-p^{dis}_{it}/\eta,
\qquad 0\le e_{it}\le E_i,
\qquad e_{i0}=e_{iT}.
\]

The shared-inverter inequality limits combined charging and discharging but
does not strictly impose \(p^{ch}_{it}p^{dis}_{it}=0\). Simultaneous operation
is economically discouraged by efficiency losses, degradation, and positive
quadratic dispatch costs. It did not occur above 1e-6 MW in the converged
default run. A strict prohibition would make the lower level mixed-integer or
nonconvex and would invalidate the present convex-KKT construction.

A strongly penalised two-sided demand adjustment pins the LMP selection at
merit-order kinks:

\[
\frac{1}{2}(500)z_{nt}^2,
\qquad 500z_{nt}=\lambda_{nt}.
\]

The adjustment is explicitly exported and is normally only a few hundredths
of a MW.

### 4.3 Investor objective

Given the ISO outcome and LMPs, investor \(i\) maximises

\[
\begin{aligned}
\Pi_i={}&\sum_t\lambda_{n(i)t}
  (p^{dis}_{it}-p^{ch}_{it})\\
&+\sum_{g,t}\theta_{ig}(\lambda_{n(g)t}-c_g)p_{gt}\\
&-C_i^{deg}-C_i^{fixed}.
\end{aligned}
\]

For I1 and I2, every ownership share is zero. I3 has shares 0.8 in
RES_WIND_N1 and 0.2 in RES_PV_N3; I4 has the reverse 0.2/0.8 shares.

A documented reward of 0.001 EUR/MW of offered availability selects full
availability when an offer is slack and economically irrelevant. It is only a
lexicographic tie-breaker and is excluded from reported profit.

### 4.4 MPEC and equilibrium algorithm

Each investor's operational best response embeds the ISO using primal
feasibility, stationarity, and Scholtes-relaxed complementarity:

\[
0\le uv\le\epsilon,
\qquad \epsilon=10^{-5}.
\]

The nonconvex best-response NLP is solved with IPOPT. The equilibrium driver
uses rotating, damped Gauss--Seidel updates with damping 0.35. After every
complete sweep, both investors are solved again against the same frozen final
profile. Convergence is declared from this simultaneous Nash audit rather
than from within-sweep movement alone.

Every MPEC candidate is independently cleared by the exact convex ISO model.
The audit checks:

- best-response termination;
- strategy deviation;
- complementarity residuals and primal--dual gap;
- embedded versus recleared LMPs;
- embedded versus recleared profit; and
- exact-recleared profitable unilateral deviation.

## 5. Four-investor transition status

The previously reported two-investor equilibrium is superseded by the new
four-investor data and must not be interpreted as a result for this case. The
market-clearing and individual best-response unit tests pass.

A preliminary 40-sweep discharge-only run reached a very small Nash strategy
residual and profitable-deviation residual, but it did not pass the strict
embedded-versus-recleared price and profit audit. It is therefore not yet a
verified operational equilibrium.

| Diagnostic | Final value |
|---|---:|
| Maximum Nash strategy deviation | 0.000492 MW |
| Maximum recleared profitable deviation | 0.000185 EUR/day |
| Maximum embedded/recleared LMP gap | 0.063547 EUR/MWh |
| Maximum embedded/recleared profit gap | 2.143451 EUR/day |

The diagnostic outputs are stored in `output/four_investor/`. The remaining
audit mismatch needs to be resolved before reporting four-investor profits or
withholding behavior as results.

## 6. Proposed two-stage investment--operation game

### 6.1 Timing

The extension separates investment commitment from operational withholding:

\[
\boxed{
\text{investment Nash game}
\;\longrightarrow\;
\text{operational-offer Nash game}
\;\longrightarrow\;
\text{ISO clearing}
}
\]

The first two levels are strategic decisions by the same investors at
different dates. The operational investors are simultaneous Nash players,
not followers of one another. The ISO is the physical market-clearing
follower inside the operational subgame.

### 6.2 Stage 1: investment

Investor \(i\) chooses installed power and energy:

\[
K_i\ge0,\qquad E_i\ge0,
\]

subject initially to individual bounds and duration restrictions:

\[
0\le K_i\le\overline K_i,
\qquad
\underline h_iK_i\le E_i\le\overline h_iK_i.
\]

Its annualised daily investment cost is

\[
C_i^{inv}(K_i,E_i)=c_i^K K_i+c_i^E E_i.
\]

A shared constraint such as

\[
\sum_iK_i\le\overline K_{N3}
\]

would create a generalized Nash investment game because one investor's
feasible investment depends on the rival's investment. The first extension
should therefore use individual upper bounds and let the transmission network
create scarcity. A common connection cap should only be added together with
an explicit allocation or generalized-equilibrium rule.

### 6.3 Stage 2: operational equilibrium conditional on investment

For fixed \((K,E)\), the operational availability game is the current toy
with the constants \(K_i,E_i\) replaced by the Stage-1 choices. Let

\[
\mathcal{O}(K,E)
\]

denote its equilibrium correspondence. An operational equilibrium satisfies,
for every investor \(i\),

\[
a_i^*(K,E)\in
\arg\max_{a_i}
\pi_i^{op}
\left(a_i,a_{-i}^*(K,E);K,E\right),
\]

where every candidate offer profile is cleared by the ISO. Capital cost is
sunk in this subgame and therefore does not affect hourly offers.

If a documented selection rule gives a single continuation equilibrium
\(S(K,E)\in\mathcal O(K,E)\), define the resulting operating value as

\[
V_i(K,E)=\pi_i^{op}\bigl(S(K,E);K,E\bigr).
\]

### 6.4 Investment equilibrium

The Stage-1 objective becomes

\[
\max_{K_i,E_i}
\left[
V_i(K_i,E_i,K_{-i},E_{-i})
-C_i^{inv}(K_i,E_i)
\right].
\]

An investment Nash equilibrium \((K^*,E^*)\) requires, for every feasible
unilateral deviation \((\widetilde K_i,\widetilde E_i)\),

\[
\begin{aligned}
&V_i(K^*,E^*)-C_i^{inv}(K_i^*,E_i^*)\\
&\quad\ge
V_i(\widetilde K_i,\widetilde E_i,K_{-i}^*,E_{-i}^*)
-C_i^{inv}(\widetilde K_i,\widetilde E_i).
\end{aligned}
\]

Every investment deviation therefore requires a new solution and audit of
the complete operational EPEC.

## 7. Why the stages must remain separate

If investment and hourly availability are chosen simultaneously in one MPEC,
unused installed capacity is normally dominated by lower investment because
it incurs capital cost without affecting the market. Strategic
underinvestment and operational withholding then become observationally
confounded.

With sequential timing, installed capacity is fixed and sunk when the hourly
market is played. An investor that previously installed 20 MW can rationally
offer only 15 MW in a peak hour if the price effect is profitable. This is
genuine physical withholding. The earlier investment decision anticipates
the entire continuation equilibrium but cannot be revised during operation.

In a deterministic single-day model with positive capital cost, the investor
will still not build capacity that is never useful in any continuation hour.
Installed power can nevertheless exceed offers in particular hours because
the same inverter supports charging and discharging opportunities across the
whole horizon.

## 8. Equilibrium-selection issue

The operational game can, in principle, possess multiple equilibria. In that
case \(V_i(K,E)\) is not automatically single-valued. Warm-start path
selection must not silently define investment payoffs.

Before claiming an investment equilibrium, the model must do at least one of
the following:

- establish a unique operational equilibrium over the tested capacity range;
- state and justify a deterministic continuation-equilibrium selection rule;
- report optimistic and pessimistic investment payoffs over all identified
  operational equilibria; or
- analyse separate operational-equilibrium branches.

Operational best responses should retain independent exact reclearing and
should be multistarted when used to evaluate investment payoffs.

## 9. Recommended implementation path

For the four-investor toy, use backward induction rather than immediately
constructing one large nested MPEC.

1. Keep all four batteries at N3.
2. Start with a discrete power grid, for example
   \(K_i\in\{0,5,10,15,20,25,30\}\) MW.
3. Initially fix duration, for example \(E_i=2.5K_i\); introduce alternative
   durations only after the power-investment game is understood.
4. For every investment vector \((K_1,K_2,K_3,K_4)\), solve and audit the
   operational EPEC.
5. Store recleared operating profits and subtract investment costs.
6. Construct the four investment-payoff arrays.
7. Identify cells in which neither investor has a profitable unilateral
   investment deviation.
8. Repeat selected cells with multistart operational best responses and a
   tighter complementarity tolerance.
9. Only then consider continuous outer optimisation or endogenous duration.

This grid method is an exact Nash calculation for the discretised investment
game, exposes discontinuities and multiple branches, and avoids taking KKT
conditions of an operational EPEC. A fully continuous single-level
reformulation would contain the equilibrium conditions of every operational
investor, each already containing the ISO KKT system, and would be
substantially harder to solve and validate.

## 10. Immediate modelling decisions still required

Before implementing Stage 1, decide:

1. Whether only power is endogenous initially or both power and energy.
2. Whether duration is fixed or selected from a small set.
3. Whether investments have individual upper bounds or a shared nodal cap.
4. How operational equilibria are selected if more than one is found.
5. Whether the horizon is one repeated representative day or a weighted set
   of representative operating conditions.
6. Whether discharge-only availability remains the baseline before testing
   two-sided strategic availability.

The recommended minimal extension is endogenous power with fixed 2.5-hour
duration, individual investment bounds, discharge-only operational
withholding, and a discrete backward-induction payoff table.

## 11. Reproduction

From this folder, run the fixed-capacity operational game with:

```powershell
python run_toy.py
```

Run the two-sided availability experiment with:

```powershell
python run_toy.py --two-sided
```

The legacy two-investor outputs remain in `output/default/`; the preliminary
four-investor fixed-capacity diagnostic is in `output/four_investor/`. The
main equations are in `formulation.md`, and the implementations are in
`model.py`, `run_toy.py`, and `run_joint_investment.py`.

## 12. Implemented one-stage alternative

The toy now also contains `run_joint_investment.py`, which tests a different
timing assumption from Section 6. Each of the four investors simultaneously
chooses continuous power, continuous energy, and hourly availability in one
best-response MPEC:

\[
0\le K_i\le30\text{ MW},\qquad 2K_i\le E_i\le8K_i,
\qquad 0\le a^{dis}_{it}\le K_i.
\]

This is a one-stage investment-and-operation Nash game, not backward
induction. The default keeps charging availability equal to installed power.

A 30-sweep diagnostic run did not converge to an investment Nash equilibrium.
At its final iterate, however, all investors offered essentially all installed
discharge capacity: the largest power-minus-offer gap was 0.00140 MW. All four
selected durations extremely close to the two-hour lower bound. The final
iterate was

| Investor | Power | Energy | Duration | Maximum discharge withholding |
|---|---:|---:|---:|---:|
| I1 | 5.869 MW | 11.738 MWh | 2.0000 h | 0.00003 MW |
| I2 | 6.818 MW | 13.637 MWh | 2.0000 h | 0.00015 MW |
| I3 | 7.891 MW | 15.783 MWh | 2.0001 h | 0.00140 MW |
| I4 | 9.793 MW | 19.587 MWh | 2.0000 h | 0.00071 MW |

These capacities are not equilibrium results. The simultaneous final audit
still found a maximum 0.116 MW capacity deviation and a 515.99 EUR/day
profitable deviation, concentrated in the solar-heavy investor. Small
capacity changes cross price regimes and materially change renewable
portfolio rent. IPOPT also reached its iteration limit for several I3/I4
best responses. The exact market-reclear checks themselves remained tight:
the final maximum LMP gap was 0.00090 EUR/MWh and the maximum embedded-profit
gap was 0.00706 EUR/day.

The diagnostic therefore supports the narrow hypothesis that simultaneous
capacity choice removes material quantity withholding in this run, but it
does not yet establish a one-stage investment equilibrium. Outputs are in
`output/joint_investment/`.
