param(
  [ValidateSet("dir", "nsis")]
  [string]$Target = "nsis",
  [string]$OutputDir = "release"
)

$ErrorActionPreference = "Stop"
$Frontend = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Frontend

Write-Host "[prepare-dist] build portable python runtime"
& (Join-Path $PSScriptRoot "build-portable-python.ps1") -TorchWheel cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:ELECTRON_RUN_AS_NODE = $null
if (-not $env:ELECTRON_BUILDER_BINARIES_MIRROR) {
  $env:ELECTRON_BUILDER_BINARIES_MIRROR = "https://npmmirror.com/mirrors/electron-builder-binaries/"
}

Set-Location $Frontend
Write-Host "[prepare-dist] build frontend"
npm run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[prepare-dist] package electron with portable-python (about 5GB)"
$builderArgs = @("--config.directories.output=$OutputDir")
if ($Target -eq "dir") {
  $builderArgs += @("--win", "dir", "--x64")
} else {
  $builderArgs += @("--win", "nsis", "--x64")
}
npx electron-builder @builderArgs
exit $LASTEXITCODE
