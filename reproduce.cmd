@echo off
setlocal
cd /d "%~dp0"
if defined VIRTUAL_ENV (
    "%VIRTUAL_ENV%\Scripts\python.exe" reproduce.py %*
    goto finished
)
where py >nul 2>nul
if errorlevel 1 (
    python reproduce.py %*
) else (
    py -3 reproduce.py %*
)
:finished
set "result=%errorlevel%"
echo.
if not "%result%"=="0" echo Reproduction failed. See the log path above.
pause
exit /b %result%
