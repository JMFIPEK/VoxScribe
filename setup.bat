@echo off
REM VoxScribe - Setup Script mit uv
REM Erstellt virtuelle Umgebung und installiert alle Abhaengigkeiten

setlocal enabledelayedexpansion

echo ============================================================
echo   VoxScribe - Installation
echo ============================================================
echo.

REM 1. Pruefen ob uv installiert ist
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [1/5] uv nicht gefunden - wird installiert...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if !errorlevel! neq 0 (
        echo FEHLER: uv Installation fehlgeschlagen.
        echo Bitte installieren Sie uv manuell: https://docs.astral.sh/uv/getting-started/installation/
        exit /b 1
    )
    REM uv zur PATH hinzufuegen (fuer diese Session) - benutze vollstaendigen Pfad
    set "UV_PATH=%USERPROFILE%\.local\bin\uv.exe"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    echo.
    echo   uv wurde installiert unter: %UV_PATH%
    echo.
) else (
    echo [1/5] uv gefunden
    set "UV_PATH=uv"
)

REM 2. Virtuelle Umgebung erstellen (bestehende ueberschreiben)
echo [2/5] Virtuelle Umgebung wird erstellt...
%UV_PATH% venv --python 3.11 --force
if !errorlevel! neq 0 (
    echo FEHLER: Virtuelle Umgebung konnte nicht erstellt werden.
    echo Stellen Sie sicher, dass Python 3.11 installiert ist.
    exit /b 1
)
echo   Aktiviere virtuelle Umgebung...
call .venv\Scripts\activate.bat

REM 3. PyTorch mit CUDA installieren
echo [3/5] PyTorch mit CUDA 12.8 wird installiert (dauert einige Minuten)...
%UV_PATH% pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
if !errorlevel! neq 0 (
    echo FEHLER: PyTorch Installation fehlgeschlagen.
    exit /b 1
)
echo   PyTorch Installation erfolgreich.

REM 4. Weitere Abhaengigkeiten installieren
echo [4/5] Weitere Abhaengigkeiten werden installiert...
%UV_PATH% pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo FEHLER: Installation der Abhaengigkeiten fehlgeschlagen.
    exit /b 1
)

REM 5. PyTorch CUDA-Version sicherstellen
echo [5/5] Sichere CUDA-Version von PyTorch...
%UV_PATH% pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
if !errorlevel! neq 0 (
    echo FEHLER: CUDA-Version Update fehlgeschlagen.
    exit /b 1
)

echo.
echo ============================================================
echo   Installation erfolgreich abgeschlossen!
echo ============================================================
echo.
echo Naechste Schritte:
echo   1. .env Datei erstellen und HF_TOKEN eintragen
echo   2. Modelle herunterladen: uv run python download_models.py
echo   3. GUI starten: uv run python gui_qt.py
echo.
echo Oder virtuelle Umgebung aktivieren und Befehle direkt ausfuehren:
echo   call .venv\Scripts\activate
echo   python gui_qt.py
echo.

endlocal
