@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "BOT_PY=py -3"
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto python_ready
set "BOT_PY=python"
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto python_ready
echo Python 3.10 or newer was not found.
echo Install Python from https://www.python.org/downloads/windows/
echo Enable "Add python.exe to PATH", then open this file again.
goto failed

:python_ready
%BOT_PY% run_windows.py
if errorlevel 1 goto failed
echo Bot stopped.
pause
exit /b 0

:failed
echo.
echo The bot could not start. Keep this window open to read the error.
pause
exit /b 1
