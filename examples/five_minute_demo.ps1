$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked "Install locked dependencies and optional Jev SDK" "uv" @("sync", "--frozen", "--extra", "live")
Invoke-Checked "Run the complete four-turn offline/mock demonstration" "uv" @("run", "jevloop", "--mode", "offline", "--generator-mode", "mock", "--transcript", "transcripts/run.jsonl")
Invoke-Checked "Run unit and orchestration tests (live tests are opt-in)" "uv" @("run", "pytest", "-q")
Invoke-Checked "Evaluate the deterministic mock (plumbing only)" "uv" @("run", "jevloop-eval", "--mode", "offline")
Invoke-Checked "Evaluate the separate keyword baseline" "uv" @("run", "jevloop-eval", "--mode", "baseline")

Write-Host "Done. Live Jev verification requires TYPESAFE_API_KEY and is not performed by this offline script."
