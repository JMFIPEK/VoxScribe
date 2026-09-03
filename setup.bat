@echo off
REM VoxScribe - Setup script using uv
REM Creates a virtual environment and installs all dependencies

setlocal enabledelayedexpansion

echo ============================================================
echo   VoxScribe - Installation
echo ============================================================
echo.

REM 1. Check whether uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [1/5] uv not found - installing...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if !errorlevel! neq 0 (
        echo ERROR: uv installation failed.
        echo Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/
        exit /b 1
    )
    REM Add uv to PATH (for this session) - use the full path
    set "UV_PATH=%USERPROFILE%\.local\bin\uv.exe"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    echo.
    echo   uv was installed to: %UV_PATH%
    echo.
) else (
    echo [1/5] uv found
    set "UV_PATH=uv"
)

REM 2. Create virtual environment (overwrite existing)
echo [2/5] Creating virtual environment...
%UV_PATH% venv --python 3.11 --force
if !errorlevel! neq 0 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.11 is installed.
    exit /b 1
)
echo   Activating virtual environment...
call .venv\Scripts\activate.bat

REM 3. Install PyTorch with CUDA
echo [3/5] Installing PyTorch with CUDA 12.8 (takes a few minutes)...
%UV_PATH% pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
if !errorlevel! neq 0 (
    echo ERROR: PyTorch installation failed.
    exit /b 1
)
echo   PyTorch installation successful.

REM 4. Install remaining dependencies
echo [4/5] Installing remaining dependencies...
%UV_PATH% pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)

REM 5. Make sure the CUDA build of PyTorch is used
echo [5/5] Ensuring PyTorch CUDA build...
%UV_PATH% pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
if !errorlevel! neq 0 (
    echo ERROR: CUDA build update failed.
    exit /b 1
)

echo.
echo ============================================================
echo   Installation completed successfully!
echo ============================================================
echo.
echo Next steps:
echo   1. Create a .env file and add HF_TOKEN
echo   2. Download models: uv run python download_models.py
echo   3. Start the GUI: uv run python gui_qt.py
echo.
echo Or activate the virtual environment and run commands directly:
echo   call .venv\Scripts\activate
echo   python gui_qt.py
echo.

endlocal
