@echo off
setlocal

echo ====================================================
echo Starting Automated Data Harvest: %date% %time%
echo ====================================================

cd /d "C:\workspace\Data_scrapping_thgingsboard_ml-intern"

rem Self-healing environment: create/repair .venv from requirements.txt
rem (idempotent; a healthy env costs ~1s). Fails the run early if unrepairable.
python provision_venv.py
if %errorlevel% neq 0 (
    echo [ERROR] .venv could not be provisioned - see output above
    exit /b 1
)

echo Running Jupyter Notebook...
rem Execute to a SEPARATE output notebook: papermill in-place execution was the
rem cause of the half-executed-notebook corruption seen on 2026-09-16.
rem (papermill does not create the output folder itself.)
if not exist "executed" mkdir "executed"
.\.venv\Scripts\python.exe -m papermill TB_Full_Harvest_v11.ipynb executed\TB_Full_Harvest_v11_executed.ipynb
if %errorlevel% neq 0 (
    echo [ERROR] Notebook execution failed!
    exit /b %errorlevel%
)

echo Running current-state snapshot...
call run_current_state_snapshot.bat

if %errorlevel% neq 0 (
    echo [ERROR] Current-state snapshot failed!
    exit /b %errorlevel%
)

echo Staging files for commit...
git add .

echo Committing files...
git commit -m "Automated data harvest: %date% %time%"

if %errorlevel% neq 0 (
    echo [WARN] Git commit failed or there were no changes to commit.
)

echo Pushing to GitHub...
git push origin main

if %errorlevel% neq 0 (
    echo [ERROR] Git push failed!
    exit /b %errorlevel%
)

echo ====================================================
echo Harvest and Sync Completed: %date% %time%
echo ====================================================
endlocal
