@echo off
setlocal

cd /d "C:\workspace\Data_scrapping_thgingsboard_ml-intern"

set SNAPSHOT_DIR=current_state_snapshots
set LOG_FILE=%SNAPSHOT_DIR%\nightly_current_state.log

if not exist "%SNAPSHOT_DIR%" mkdir "%SNAPSHOT_DIR%"

echo ==================================================== >> "%LOG_FILE%"
echo Starting current-state snapshot: %date% %time% >> "%LOG_FILE%"
echo ==================================================== >> "%LOG_FILE%"

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Missing .venv\Scripts\python.exe >> "%LOG_FILE%"
    exit /b 1
)

.\.venv\Scripts\python.exe pull_current_state_snapshots.py ^
    --output-dir "%SNAPSHOT_DIR%" ^
    --recovery-output "%SNAPSHOT_DIR%\offline_recoveries.csv" ^
    >> "%LOG_FILE%" 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] Current-state snapshot failed: %date% %time% >> "%LOG_FILE%"
    exit /b %errorlevel%
)

echo Current-state snapshot completed: %date% %time% >> "%LOG_FILE%"
endlocal
