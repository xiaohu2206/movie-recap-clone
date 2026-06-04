param(
  [string]$Python = "python",
  [string]$VenvPath = ".\.venv",
  [ValidateSet("default", "cpu", "cu121", "cu124", "cu128")]
  [string]$TorchWheel = "default"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root $VenvPath

& $Python -m venv $Venv
$Pip = Join-Path $Venv "Scripts\pip.exe"
$Py = Join-Path $Venv "Scripts\python.exe"

& $Py -m pip install --upgrade pip
& $Pip install -r (Join-Path $Root "requirements.txt")

if ($TorchWheel -eq "cpu") {
  & $Pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
} elseif ($TorchWheel -eq "cu121") {
  & $Pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
} elseif ($TorchWheel -eq "cu124") {
  & $Pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
} elseif ($TorchWheel -eq "cu128") {
  & $Pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
}

Write-Host "虚拟环境已准备好: $Venv"
Write-Host "激活命令: $Venv\Scripts\Activate.ps1"
Write-Host "检查 GPU: $Py -c `"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')`""
