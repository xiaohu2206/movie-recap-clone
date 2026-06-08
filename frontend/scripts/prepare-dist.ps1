param(
  [ValidateSet("dir", "nsis")]
  [string]$Target = "nsis",
  [string]$OutputDir = "release"
)

$ErrorActionPreference = "Stop"
$Frontend = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Frontend
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
  Write-Host "[prepare-dist] 未找到 .venv，正在创建（首次较慢）..."
  & (Join-Path $Root "setup_venv.ps1") -TorchWheel cu128
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[prepare-dist] 检查 Python 依赖..."
& $VenvPy -c "import cv2, torch, numpy; print('[prepare-dist] cv2', cv2.__version__, '| torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
  Write-Error "Python 依赖不完整，请在项目根目录运行 .\setup_venv.ps1 -TorchWheel cu128"
}

$env:ELECTRON_RUN_AS_NODE = $null
if (-not $env:ELECTRON_BUILDER_BINARIES_MIRROR) {
  $env:ELECTRON_BUILDER_BINARIES_MIRROR = "https://npmmirror.com/mirrors/electron-builder-binaries/"
}

Set-Location $Frontend
Write-Host "[prepare-dist] 构建前端..."
npm run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[prepare-dist] 打包 Electron（含 .venv，体积约 5GB，请耐心等待）..."
$builderArgs = @("--config.directories.output=$OutputDir")
if ($Target -eq "dir") {
  $builderArgs += @("--win", "dir", "--x64")
} else {
  $builderArgs += @("--win", "nsis", "--x64")
}
npx electron-builder @builderArgs
exit $LASTEXITCODE
