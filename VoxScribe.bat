@echo off
REM ============================================================
REM  VoxScribe - double-click launcher for the GUI
REM ============================================================
REM  Put this on your Desktop as a shortcut:
REM    - run  Create-Desktop-Shortcut.ps1  once (right-click > Run with PowerShell),
REM      or
REM    - right-click this file > Send to > Desktop (create shortcut)
REM
REM  First time on a fresh machine: run  setup.bat  once to install
REM  dependencies, then use this launcher from then on.
REM ============================================================

setlocal
REM Always run from the repo folder, regardless of where the shortcut lives
cd /d "%~dp0"

REM Locate uv: the official installer's path first, then anything on PATH
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" set "UV=uv"

"%UV%" run python gui_qt.py
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ------------------------------------------------------------
    echo  VoxScribe exited with an error ^(code %RC%^).
    echo  If this is the first start on this machine, run  setup.bat
    echo  once to install dependencies, then try again.
    echo ------------------------------------------------------------
    echo.
    pause
)

endlocal
exit /b %RC%
