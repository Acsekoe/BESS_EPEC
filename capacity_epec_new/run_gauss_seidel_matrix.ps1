param(
    [string]$PythonExe = "python",
    [string]$OutputRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
# PowerShell 7 can turn a nonzero native exit into a PowerShell error.  These
# experiments deliberately use exit 1 for a completed, non-equilibrium run.
$PSNativeCommandUseErrorActionPreference = $false

$projectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $projectRoot "model/output/gs_matrix_$stamp"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot $OutputRoot
}
[void][System.IO.Directory]::CreateDirectory($OutputRoot)

$statuses = [System.Collections.Generic.List[object]]::new()

function Invoke-CapacityRun {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    $exitCode = -999
    try {
        & $PythonExe "model/run_model.py" @Arguments
        $exitCode = $LASTEXITCODE
    }
    catch {
        Write-Warning "$Label could not complete: $($_.Exception.Message)"
    }
    $script:statuses.Add([pscustomobject]@{ Run = $Label; ExitCode = $exitCode })
    Write-Host "=== $Label finished with model exit code $exitCode; continuing ==="
}

$common = @(
    "--data", "model/input/market_data_smoothed.json",
    "--market", "energy-only",
    "--formulation", "relaxed-kkt",
    "--demand-adjustment-penalty-eur-per-mw2", "100",
    "--complementarity-epsilon", "1e-3",
    "--solver-tolerance", "1e-4",
    "--damping", "0.25",
    "--proximal-penalty", "0",
    "--multistart-every-sweeps", "0",
    "--refinement-starts", "3",
    "--audit-relative-regret-tolerance", "0.01",
    "--parallel-workers", "4",
    "--max-solver-iterations", "3000",
    "--max-solve-seconds", "600"
)

$seedDirectory = Join-Path $OutputRoot "jacobi_seed"
$seedCapacities = Join-Path $seedDirectory "final_capacities.csv"

Push-Location $projectRoot
try {
    # One common simultaneous sweep gives all four sequential experiments the
    # exact same 5 MW / 15 MWh-per-node seed and avoids four redundant audits.
    Invoke-CapacityRun -Label "one-sweep Jacobi seed" -Arguments ($common + @(
        "--update-scheme", "jacobi",
        "--max-sweeps", "1",
        "--consecutive-sweeps", "2",
        "--convergence-metric", "capacity",
        "--initial-power-mw", "5",
        "--initial-ratio-hours", "3",
        "--output-dir", $seedDirectory
    ))

    $experiments = @(
        [pscustomobject]@{ Name = "gs_fixed_i1_first"; Order = @("I1", "I2", "I3"); Rotate = $false },
        [pscustomobject]@{ Name = "gs_fixed_i2_first"; Order = @("I2", "I3", "I1"); Rotate = $false },
        [pscustomobject]@{ Name = "gs_fixed_i3_first"; Order = @("I3", "I1", "I2"); Rotate = $false },
        [pscustomobject]@{ Name = "gs_rotating_first_mover"; Order = @("I1", "I2", "I3"); Rotate = $true }
    )

    foreach ($experiment in $experiments) {
        $arguments = $common + @(
            "--update-scheme", "gauss-seidel",
            "--gauss-seidel-order"
        ) + $experiment.Order + @(
            "--max-sweeps", "50",
            # This deliberately prevents the capacity-distance rule from ending
            # a run early.  The endpoint gets a separate multistart payoff audit.
            "--consecutive-sweeps", "51",
            "--convergence-metric", "capacity",
            "--initial-capacities", $seedCapacities,
            "--output-dir", (Join-Path $OutputRoot $experiment.Name)
        )
        if ($experiment.Rotate) {
            $arguments += "--gauss-seidel-rotate-first-mover"
        }
        Invoke-CapacityRun -Label $experiment.Name -Arguments $arguments
    }
}
finally {
    Pop-Location
}

Write-Host "`nAll requested runs were attempted. Output root: $OutputRoot" -ForegroundColor Green
$statuses | Format-Table -AutoSize
exit 0
