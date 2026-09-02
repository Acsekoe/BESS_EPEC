# diagnostics/portfolio

Audit and seeding scripts for the portfolio bidding game (2026-09-02):

- `audit_zero_proximal.py`: re-solve every investor's BR at a run's final
  state with proximal_generation_penalty = 0; validates each proposal by
  exact reclear. (Paths point at `portfolio_gen_only_40_sweeps`; edit RUN.)
- `multistart_offers.py`: same audit from multiple offer initializations
  (truthful vs run-A-seeded) to detect plateau/local-optimum artifacts.
- `seed_offer_state.py <source_run_dir> <seeded_dir>`: copy a finished run
  and overwrite I3/I4 offers with their zero-proximal BR proposals so a
  resumed Jacobi starts with the offer channel active. Needed because a
  truthful offer has zero gradient below the local LMP: unaided warm-started
  BRs in the joint game never discover manipulation.

Findings and numbers: `workflow/summary_2026-09-02_11-08.md`.
