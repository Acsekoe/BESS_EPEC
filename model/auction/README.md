# Archived auction-MPEC experiment

This folder contains the deferred two-follower access-auction experiment. The
maintained thesis workflow uses `model/epec_diagonalization.py`; these commands
are retained only to reproduce the auction diagnostics.

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

## Gauss--Seidel diagonalization

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
