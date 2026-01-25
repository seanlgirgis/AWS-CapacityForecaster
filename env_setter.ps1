# env_setter.ps1
$venvPath = "c:\py_venv\AWS-CapacityForecaster"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    
    # Set PROJECT_ROOT to the current script's directory
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $env:PROJECT_ROOT = $scriptDir
    
    # Add project root to PYTHONPATH so imports work from anywhere
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$scriptDir;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $scriptDir
    }

    Write-Host "Virtual environment activated: $venvPath" -ForegroundColor Green
    Write-Host "PROJECT_ROOT set to: $env:PROJECT_ROOT" -ForegroundColor Green
    Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Green
    Write-Host "Python executable: $(Get-Command python).Path" -ForegroundColor Cyan
}
else {
    Write-Host "Activation script not found! Check the path:" $activateScript -ForegroundColor Red
}
