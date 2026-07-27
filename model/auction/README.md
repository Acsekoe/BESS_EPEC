# Isolated auction-MPEC experiment

This folder contains the two-follower access-auction experiment. All code, case
data, and generated results for this branch stay below `model/auction/`; the
maintained thesis workflow remains `model/epec_diagonalization.py`.

For each best response, one BESS investor is the leader and the model embeds:

1. a nodal access-allocation auction; and
2. the electricity spot-market dispatch.

Sequentially solving the investor best responses therefore approximates a
multi-leader/two-follower EPEC by Gauss--Seidel diagonalization. It is a local,
optimistic MPEC calculation rather than a proof that a global equilibrium
exists.

Access settlement is pay-as-bid: awarded MW are charged at the active
investor's submitted price. `--tie-break-epsilon` adds a fixed lexicographic
merit offset so equal submitted prices allocate reproducibly; it changes only
auction ranking, not the submitted-price payment.

For a fully continuous pay-as-bid diagnostic, set the merit perturbation to
zero. Both the active bid quantity and submitted price then remain continuous
upper-level decisions, and awarded access is charged at the active investor's
own bid:

```powershell
python model\auction\single_investor_auction_mpec.py `
  --active-investor I1 `
  --active-node N8 `
  --rival-bids model\auction\data\auction_mpec_cases\two_investor_n8_uniform.json `
  --tie-break-epsilon 0 `
  --initial-bid-quantity 70 `
  --initial-bid-price 30 `
  --initial-duration 4 `
  --max-cpu-time 180 `
  --output model\auction\output\single_investor_auction_mpec\two_investor_n8_uniform\I1_pay_as_bid_continuous.json
```

This diagnostic deliberately permits exact and near bid ties. Its output is
acceptable only if the embedded award is reproduced by the independent auction
reclear; a mismatch identifies optimistic allocation on a nonunique tie face.

For the maintained pay-as-bid tick experiment, add a positive grid size and a
smaller deterministic merit offset for exact raw-price ties:

```powershell
python model\auction\single_investor_auction_mpec.py `
  --active-investor I1 `
  --active-node N8 `
  --rival-bids model\auction\data\auction_mpec_cases\two_investor_n8_uniform.json `
  --bid-price-tick 0.01 `
  --tie-break-epsilon 0.001 `
  --initial-bid-quantity 70 `
  --initial-bid-price 30 `
  --initial-duration 4 `
  --max-cpu-time 180 `
  --output model\auction\output\single_investor_auction_mpec\two_investor_n8_uniform\I1_pay_as_bid_tick001.json
```

Grid mode currently requires one active node. It evaluates the minimum grid
price, every fixed rival price at that node, and one tick above each rival, with
continuous bid quantity and energy in every MPEC. A separate zero-quantity
outside option is mandatory. Candidates with a non-optimal solve, an auction
award reclear difference above `1e-4` MW, an excessive strong-duality gap, or a
payment mismatch are rejected before the highest-profit valid response is
selected. Rival bids must lie on the same grid, and the maximum priority offset
must remain smaller than one price tick.

## Two-investor N8 diagnostic

`data/auction_mpec_cases/two_investor_n8_uniform.json` starts I1 and I2 with
equal 70 MW bids at N8, where the auction limit is 100 MW. The deliberate tie
tests the deterministic priority rule.

Solve I1's tick-constrained best response against I2:

```powershell
python model\auction\single_investor_auction_mpec.py `
  --active-investor I1 `
  --active-node N8 `
  --rival-bids model\auction\data\auction_mpec_cases\two_investor_n8_i2_100mw_30.json `
  --bid-price-tick 0.01 `
  --tie-break-epsilon 0.001 `
  --initial-bid-quantity 100 `
  --initial-bid-price 30 `
  --initial-duration 4 `
  --max-cpu-time 180 `
  --output model\auction\output\single_investor_auction_mpec\two_investor_n8_i2_100mw_30\I1.json
```

The Gauss--Seidel driver uses the same tick-price enumeration and candidate
validity checks when `--bid-price-tick` is positive. Grid mode currently
requires one active node and damping equal to one:

```powershell
python model\auction\gauss_seidel.py `
  --initial-bids model\auction\data\auction_mpec_cases\two_investor_n8_uniform.json `
  --investor-order I1 I2 `
  --active-nodes N8 `
  --bid-price-tick 0.01 `
  --tie-break-epsilon 0.001 `
  --max-iterations 1 `
  --max-cpu-time 180 `
  --output-dir model\auction\output\gauss_seidel\two_investor_n8_uniform_iter1
```

The sweep compares each strategic response with an explicit zero-bid outside
option. In this two-investor case, each response evaluates three grid prices
plus the outside option, so one sweep requires eight MPEC solves. For a faster
solver smoke test only, add `--skip-outside-option`.

Inspect `iteration_history.csv`, each investor JSON below `iterations/`, and
`final_state.json`. In addition to termination and strong-duality gaps, the
investor JSON reports independently recleared awards. A material embedded-
versus-reclear award difference identifies an optimistic or tied allocation
that the standalone auction does not reproduce.

## Archived four-investor cases

Each JSON file contains the fixed bid vectors for all four thesis investors at
all nine IEEE-9 nodes. When one investor is selected as strategic, its records
are automatically removed from the fixed input and the other three investors
remain as rivals.

The profiles are:

- `low_competition.json`: substantial residual access capacity;
- `balanced_competition.json`: moderate competition and congestion at the main
  storage nodes;
- `high_competition.json`: every node is oversubscribed by the three rivals.

Run one case from the repository root:

```powershell
python model\auction\single_investor_auction_mpec.py `
  --active-investor I3 `
  --active-node N8 `
  --rival-bids model\auction\data\auction_mpec_cases\balanced_competition.json `
  --output model\auction\output\single_investor_auction_mpec\tests\balanced_I3_N8.json
```

Run the balanced profile for every investor and every node:

```powershell
$investors = @("I1", "I2", "I3", "I4")
$nodes = 1..9 | ForEach-Object { "N$_" }
$case = "balanced_competition"

foreach ($investor in $investors) {
  foreach ($node in $nodes) {
    python model\auction\single_investor_auction_mpec.py `
      --active-investor $investor `
      --active-node $node `
      --rival-bids "model\auction\data\auction_mpec_cases\$case.json" `
      --output "model\auction\output\single_investor_auction_mpec\tests\$case\${investor}_${node}.json"
  }
}
```

To run all three profiles, wrap the same loop in:

```powershell
foreach ($case in @("low_competition", "balanced_competition", "high_competition")) {
  # investor/node loop from above
}
```

Every output records solver termination, active bid and award, profit,
strong-duality residuals, and the maximum difference between the embedded and
independently recleared auction awards. Treat only `optimal` runs as usable
local optimistic MPEC candidates.

### Gauss--Seidel diagonalization

Run the four investors sequentially from the balanced starting bids:

```powershell
python model\auction\gauss_seidel.py `
  --initial-bids model\auction\data\auction_mpec_cases\balanced_competition.json `
  --output-dir model\auction\output\gauss_seidel\balanced
```

The default CPU limit is 60 seconds per MPEC solve. Each investor response is
also compared with its explicit zero-bid outside option, so a complete sweep
normally solves eight MPECs. Continue an interrupted run from its last completed
sweep with:

```powershell
python model\auction\gauss_seidel.py `
  --resume model\auction\output\gauss_seidel\balanced\checkpoint.json `
  --output-dir model\auction\output\gauss_seidel\balanced
```
