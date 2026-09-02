# Portfolio bidding game status

Last updated: 2026-09-02

## Current formulation

The experimental owner-level renewable-curtailment penalties have been
removed. Neither the investor objective nor the exact profit decomposition
contains a linear or quadratic curtailment cost. Renewable curtailment remains
available as the physical diagnostic `renewable_curtailment_mwh` in
`exact_final_profit_decomposition.csv`.

Investor profit therefore contains storage LMP settlement, owned-generation
rent evaluated against true generation cost, degradation cost, and annualised
power/energy investment cost. Curtailment affects profit only through the
generation revenue that is not earned when available renewable energy is not
dispatched.

The portfolio game still supports:

- endogenous or fixed storage capacity;
- strategic hourly renewable-generation offers;
- strategic or non-strategic hourly storage prices;
- optional inclusion of merchant investor I2, including in seeded runs;
- exact final reclearing and truthful-generation counterfactuals.

## Verification guidance

Judge convergence from the raw best-response residuals rather than damped
state changes. Before treating a candidate as an equilibrium, run
zero-proximal best responses and independently reclear the proposed deviations.
IPOPT provides local NLP solutions, so multistart checks remain necessary for
strong claims.
