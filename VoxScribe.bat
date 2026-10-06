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

REM Only sync the venv when it's missing or uv.lock changed since the last
REM successful sync - not on every start. A plain `uv run` re-syncs every
REM time, which on an Intel Arc machine reverts torch to the CUDA build pinned
REM in pyproject.toml and makes voxscribe/gpu_setup.py re-download the XPU
REM build on every launch. The stamp is a copy of the uv.lock last synced.
set "STAMP=.venv\.voxscribe-synced-uv.lock"
set "NEED_SYNC=0"
if not exist ".venv\Scripts\python.exe" set "NEED_SYNC=1"
if not exist "%STAMP%" set "NEED_SYNC=1"
if "%NEED_SYNC%"=="0" (
    fc /b "uv.lock" "%STAMP%" >nul 2>nul || set "NEED_SYNC=1"
)

if "%NEED_SYNC%"=="1" (
    echo Dependencies changed - updating the environment...
    "%UV%" sync
    if errorlevel 1 (
        set "RC=1"
        goto :done
    )
    copy /y "uv.lock" "%STAMP%" >nul
)

"%UV%" run --no-sync python gui_qt.py
set "RC=%ERRORLEVEL%"

:done

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
