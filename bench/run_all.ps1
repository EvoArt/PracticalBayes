# Runs the Turing-side reference benchmark first, then the PracticalBayes
# corpus benchmark, sequentially, from PowerShell. Meant to be run directly
# by a person (not VSCode's integrated terminal, to avoid its RAM overhead) -
# e.g.:
#
#   cd c:\Users\arn203\.julia\dev\PracticalBayes
#   .\bench\run_all.ps1
#
# Both stages append to bench/results/history_corpus.jsonl as they go, so
# Ctrl+C at any point still leaves completed models' results on disk.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== [1/2] Turing-side corpus benchmark ===" -ForegroundColor Cyan
Push-Location (Join-Path $repoRoot "test\comparison_env")
try {
    & julia --project=. corpus_bench_turing.jl
    if ($LASTEXITCODE -ne 0) {
        throw "Turing-side benchmark exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "=== [2/2] PracticalBayes-side corpus benchmark ===" -ForegroundColor Cyan
Push-Location $repoRoot
try {
    & julia --project=bench/bench_env bench/corpus_bench.jl
    if ($LASTEXITCODE -ne 0) {
        throw "PB-side benchmark exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "=== Done. Results in bench/results/history_corpus.jsonl ===" -ForegroundColor Green
