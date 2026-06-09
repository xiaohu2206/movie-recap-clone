param(
  [string]$PythonSource = "",
  [ValidateSet("cpu", "cu121", "cu124", "cu128")]
  [string]$TorchWheel = "cu128",
  [ValidateSet("", "cu121", "cu124")]
  [string]$TorchFallback = "cu124"
)

$ErrorActionPreference = "Stop"
$Frontend = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Frontend
$Portable = Join-Path $Root "portable-python"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Marker = Join-Path $Portable ".runtime-ready"
$MarkerTorch = if ($TorchFallback) { "torch=$TorchWheel+fallback=$TorchFallback" } else { "torch=$TorchWheel" }
$VenvPip = Join-Path $Root ".venv\Scripts\pip.exe"

function Install-TorchWheel {
  param(
    [string]$PythonExe,
    [string]$PipExe,
    [string]$Wheel
  )
  if ($Wheel -eq "cpu") {
    & $PipExe install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
  } else {
    & $PipExe install --force-reinstall torch --index-url "https://download.pytorch.org/whl/$Wheel"
  }
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $VenvPy)) {
  Write-Host "[portable-python] creating .venv..."
  & (Join-Path $Root "setup_venv.ps1") -TorchWheel $TorchWheel
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  $venvTorch = ""
  try {
    $venvTorch = & $VenvPy -c "import torch; print(torch.__version__)" 2>$null
  } catch {
    $venvTorch = ""
  }
  if (-not $venvTorch -or $venvTorch -notmatch "\+$TorchWheel(\b|$)") {
    Write-Host "[portable-python] ensuring venv primary torch=$TorchWheel (was: $venvTorch)"
    Install-TorchWheel -PythonExe $VenvPy -PipExe $VenvPip -Wheel $TorchWheel
  }
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

$ReuseRuntime = $false
if (Test-Path $Marker) {
  $markerText = Get-Content $Marker -Raw
  if ($markerText -match [regex]::Escape($MarkerTorch)) {
    $ReuseRuntime = $true
    Write-Host "[portable-python] reuse existing runtime at $Portable ($MarkerTorch)"
  } else {
    Write-Host "[portable-python] torch wheel changed, rebuilding portable runtime"
    Remove-Item $Marker -Force -ErrorAction SilentlyContinue
  }
}

if (-not $ReuseRuntime) {
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

  if ($TorchFallback) {
    $FallbackSite = Join-Path $Portable ("torch_fallbacks\{0}\Lib\site-packages" -f $TorchFallback)
    New-Item -ItemType Directory -Path $FallbackSite -Force | Out-Null
    Write-Host "[portable-python] install torch fallback $TorchFallback"
    & (Join-Path $Portable "python.exe") -m pip install --upgrade --target $FallbackSite torch --index-url "https://download.pytorch.org/whl/$TorchFallback"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  Set-Content -Path $Marker -Value ("built-at=" + (Get-Date -Format "o") + "`n" + $MarkerTorch) -Encoding utf8
}

Write-Host "[portable-python] verify imports"
& (Join-Path $Portable "python.exe") -c "import cv2, torch, numpy; print('portable ok', cv2.__version__, torch.__version__)"
if ($LASTEXITCODE -ne 0) {
  Remove-Item $Marker -Force -ErrorAction SilentlyContinue
  throw "portable-python verification failed"
}

Write-Host "[portable-python] ready at $Portable"
