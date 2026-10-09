param(
    [string]$TaskName = "TB Nightly Alarm Risk Chain",
    [string]$RunTime = "04:30"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ProjectRoot "run_nightly_risk.bat"

if (-not (Test-Path $Runner)) {
    throw "Missing runner: $Runner"
}

$Action = New-ScheduledTaskAction -Execute $Runner -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Nightly 10-day telemetry pull -> feature build -> ALARM-EVENT risk scoring (24h, any non-camera alarm type) -> audit_reports CSV (self-healing venv)." `
    -Force | Out-Null

Write-Host "Scheduled '$TaskName' daily at $RunTime."
