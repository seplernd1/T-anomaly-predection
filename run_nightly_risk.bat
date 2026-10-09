@echo off
setlocal

cd /d "C:\workspace\Data_scrapping_thgingsboard_ml-intern"

set LOG_FILE=audit_reports\nightly_risk_chain.log
if not exist "audit_reports" mkdir "audit_reports"

echo ==================================================== >> "%LOG_FILE%"
Starting nightly risk chain: %date% %time% >> "%LOG_FILE%"
echo ==================================================== >> "%LOG_FILE%"

rem Self-healing environment: create/repair .venv from requirements.txt
python provision_venv.py >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] .venv could not be provisioned >> "%LOG_FILE%"
    exit /b 1
)

.\.venv\Scripts\python.exe -X utf8 nightly_risk_chain.py >> "%LOG_FILE%" 2>&1
set CHAIN_RC=%errorlevel%

echo Chain exit code: %CHAIN_RC% >> "%LOG_FILE%"
if %CHAIN_RC% neq 0 (
    echo [ERROR] nightly risk chain failed >> "%LOG_FILE%"
    exit /b %CHAIN_RC%
)
echo [OK] nightly risk chain completed >> "%LOG_FILE%"
exit /b 0
