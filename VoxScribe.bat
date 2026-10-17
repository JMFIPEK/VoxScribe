@echo off
REM ============================================================
REM  VoxScribe - launcher for the GUI
REM ============================================================
REM  Run  Create-Shortcuts.ps1  once (right-click > Run with PowerShell) to get
REM  a Desktop and Start menu entry; pin the Start menu entry to the taskbar
REM  from there. Also works on a fresh machine: the first start creates the
REM  environment (needs uv, see README).
REM ============================================================

setlocal
REM Always run from the repo folder, regardless of where the shortcut lives
cd /d "%~dp0"

REM Locate uv: the official installer's path first, then anything on PATH
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" set "UV=uv"
"%UV%" --version >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it once in PowerShell with:
    echo.
    echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo then start VoxScribe again.
    set "RC=1"
    goto :done
)

REM Only sync the venv when it's missing or uv.lock changed since the last
REM successful sync - not on every start. A plain `uv run` re-syncs every
REM time, which on an Intel Arc machine reverts torch to the CUDA build pinned
REM in pyproject.toml and makes voxscribe/gpu_setup.py re-download the XPU
REM build on every launch. The stamp is a copy of the uv.lock last synced.
set "STAMP=.venv\.voxscribe-synced-uv.lock"
set "NEED_SYNC=0"
if not exist ".venv\Scripts\pythonw.exe" set "NEED_SYNC=1"
if not exist "%STAMP%" set "NEED_SYNC=1"
if "%NEED_SYNC%"=="0" (
    fc /b "uv.lock" "%STAMP%" >nul 2>nul || set "NEED_SYNC=1"
)

REM If voxscribe/gpu_setup.py has already switched torch to the Intel Arc
REM (XPU) build, keep it: a plain sync would swap it back to the CUDA build
REM from uv.lock, and gpu_setup would then swap it to XPU again right after -
REM two multi-GB reinstalls for nothing. --inexact keeps the XPU-only extras
REM (triton-xpu, ...) that aren't in uv.lock.
set "SYNC_ARGS="
dir /b /ad ".venv\Lib\site-packages\torch-*+xpu.dist-info" >nul 2>nul && set "SYNC_ARGS=--inexact --no-install-package torch --no-install-package torchaudio --no-install-package torchvision"

if "%NEED_SYNC%"=="1" (
    echo Dependencies changed - updating the environment...
    "%UV%" sync %SYNC_ARGS%
    if errorlevel 1 (
        set "RC=1"
        goto :done
    )
    copy /y "uv.lock" "%STAMP%" >nul
)

REM Start the GUI without a console and return right away, so only the app's
REM own taskbar button stays. Its output goes to %USERPROFILE%\.voxscribe\
REM voxscribe.log (see gui_qt.py). VIRTUAL_ENV lets gpu_setup's `uv pip
REM install` find this venv, as it would under `uv run`.
set "VIRTUAL_ENV=%~dp0.venv"
start "" ".venv\Scripts\pythonw.exe" gui_qt.py
set "RC=0"

:done

if not "%RC%"=="0" (
    echo.
    echo ------------------------------------------------------------
    echo  VoxScribe could not be started ^(see the message above^).
    echo ------------------------------------------------------------
    echo.
    pause
)

endlocal
exit /b %RC%
