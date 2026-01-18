# env_setter.ps1
$venvPath = "c:\py_venv\AWS-CapacityForecaster"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated: $venvPath" -ForegroundColor Green
    Write-Host "Python executable: $(Get-Command python).Path" -ForegroundColor Cyan
}
else {
    Write-Host "Activation script not found! Check the path:" $activateScript -ForegroundColor Red
}
