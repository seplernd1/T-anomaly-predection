@echo off
cd /d "C:\workspace\Data_scrapping_thgingsboard_ml-intern"
call automate_harvest.bat
echo.
echo Exit code: %errorlevel%
pause
