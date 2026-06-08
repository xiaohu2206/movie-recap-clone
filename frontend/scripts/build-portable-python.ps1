param(
  [string]$PythonSource = "",
  [ValidateSet("cpu", "cu121", "cu124", "cu128")]
  [string]$TorchWheel = "cu128"
)

$ErrorActionPreference = "Stop"
$Frontend = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Frontend
$Portable = Join-Path $Root "portable-python"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Marker = Join-Path $Portable ".runtime-ready"

if (-not (Test-Path $VenvPy)) {
  Write-Host "[portable-python] creating .venv..."
  & (Join-Path $Root "setup_venv.ps1") -TorchWheel $TorchWheel
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $PythonSource) {
  $cfgPath = Join-Path $Root ".venv\pyvenv.cfg"
  if (-not (Test-Path $cfgPath)) {
    throw "missing .venv\pyvenv.cfg"
  }
  $homeLine = Get-Content $cfgPath | Where-Object { $_ -match "^home = " } | Select-Object -First 1
  if (-not $homeLine) {
    throw "pyvenv.cfg missing home"
  }
  $PythonSource = ($homeLine -replace "^home = ", "").Trim()
}

if (-not (Test-Path (Join-Path $PythonSource "python.exe"))) {
  throw "python.exe not found in $PythonSource"
}

if (Test-Path $Marker) {
  Write-Host "[portable-python] reuse existing runtime at $Portable"
} else {
  if (Test-Path $Portable) {
    Remove-Item $Portable -Recurse -Force
  }
  New-Item -ItemType Directory -Path $Portable | Out-Null

  Write-Host "[portable-python] copy python runtime"
  Write-Host "  from: $PythonSource"
  Write-Host "  to:   $Portable"
  robocopy $PythonSource $Portable /E /XD "__pycache__" "Lib\site-packages" /XF "*.pyc" /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  if ($LASTEXITCODE -ge 8) {
    throw "robocopy python runtime failed with exit code $LASTEXITCODE"
  }

  $SiteDest = Join-Path $Portable "Lib\site-packages"
  $SiteSrc = Join-Path $Root ".venv\Lib\site-packages"
  if (-not (Test-Path $SiteSrc)) {
    throw "missing venv site-packages at $SiteSrc"
  }

  Write-Host "[portable-python] sync site-packages"
  New-Item -ItemType Directory -Path $SiteDest -Force | Out-Null
  robocopy $SiteSrc $SiteDest /E /XD "__pycache__" /XF "*.pyc" /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
  if ($LASTEXITCODE -ge 8) {
    throw "robocopy site-packages failed with exit code $LASTEXITCODE"
  }

  Set-Content -Path $Marker -Value ("built-at=" + (Get-Date -Format "o")) -Encoding utf8
}

Write-Host "[portable-python] verify imports"
& (Join-Path $Portable "python.exe") -c "import cv2, torch, numpy; print('portable ok', cv2.__version__, torch.__version__)"
if ($LASTEXITCODE -ne 0) {
  Remove-Item $Marker -Force -ErrorAction SilentlyContinue
  throw "portable-python verification failed"
}

Write-Host "[portable-python] ready at $Portable"
