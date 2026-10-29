@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "AI_PY=py -3.12"
py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto python_ready
set "AI_PY=python"
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto python_ready
echo This AI version needs Python 3.12, 64-bit.
echo Keep your existing Python version installed.
echo Download Windows installer (64-bit) here:
echo https://www.python.org/downloads/release/python-31210/
echo Install Python 3.12, then open start_ai.bat again.
goto failed

:python_ready
%AI_PY% run_ai_windows.py
if errorlevel 1 goto failed
echo AI bot stopped.
pause
exit /b 0

:failed
echo.
echo The AI bot could not start. Keep this window open to read the error.
pause
exit /b 1
