param(
    [string]$TaskName = "ThingsBoard Current State Snapshot",
    [string]$RunTime = "02:00"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ProjectRoot "run_current_state_snapshot.bat"

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
    -Description "Pull ThingsBoard active/lastDisconnectTime/lastConnectTime snapshots nightly." `
    -Force | Out-Null

Write-Host "Scheduled '$TaskName' daily at $RunTime."
