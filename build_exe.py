"""Build-Skript: Erstellt die WhisperX .exe mit PyInstaller.

Verwendung:
    1. Zuerst Modelle herunterladen:  python download_models.py
    2. Dann builden:                  python build_exe.py

Das Ergebnis liegt in dist/WhisperX/WhisperX.exe
"""

import os
import sys
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "bundled_models")
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")


def check_prerequisites():
    """Prueft ob alle Voraussetzungen erfuellt sind."""
    # PyInstaller installiert?
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  [!] PyInstaller nicht installiert. Installiere mit: pip install pyinstaller")
        sys.exit(1)

    # Modelle vorhanden?
    if not os.path.isdir(MODELS_DIR):
        print(f"  [!] Modelle nicht gefunden: {MODELS_DIR}")
        print("  Fuehre zuerst aus: python download_models.py")
        sys.exit(1)

    whisper_dir = os.path.join(MODELS_DIR, "whisper", "medium")
    if not os.path.isdir(whisper_dir):
        print(f"  [!] Whisper medium Modell nicht gefunden: {whisper_dir}")
        print("  Fuehre zuerst aus: python download_models.py")
        sys.exit(1)

    print(f"  [OK] Bundled Models: {MODELS_DIR}")


def get_hidden_imports():
    """Liste der Hidden Imports die PyInstaller nicht automatisch erkennt."""
    return [
        "whisperx",
        "faster_whisper",
        "ctranslate2",
        "pyannote.audio",
        "pyannote.core",
        "pyannote.pipeline",
        "speechbrain",
        "torch",
        "torchaudio",
        "soundfile",
        "scipy",
        "scipy.signal",
        "numpy",
        "PySide6",
        "dotenv",
        "psutil",
        "PIL",
        "sklearn",
        "sklearn.cluster",
        "huggingface_hub",
        "transformers",
    ]


def build():
    """Fuehrt den PyInstaller-Build aus."""
    print("=" * 60)
    print("  WhisperX — .exe Build")
    print("=" * 60)
    print()

    print("[1/3] Voraussetzungen pruefen...")
    check_prerequisites()
    print()

    print("[2/3] PyInstaller-Konfiguration erstellen...")

    # Icon
    icon_path = os.path.join(BASE_DIR, "Logo.png")
    icon_arg = []
    if os.path.exists(icon_path):
        # Convert PNG to ICO for PyInstaller
        try:
            from PIL import Image
            ico_path = os.path.join(BASE_DIR, "Logo.ico")
            img = Image.open(icon_path)
            img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
            icon_arg = [f"--icon={ico_path}"]
            print(f"  [OK] Icon: {ico_path}")
        except Exception as e:
            print(f"  [!] Icon-Konvertierung fehlgeschlagen: {e}")

    # Hidden imports
    hidden_imports = get_hidden_imports()
    hidden_import_args = []
    for hi in hidden_imports:
        hidden_import_args.extend(["--hidden-import", hi])

    # Data files
    data_args = [
        f"--add-data={MODELS_DIR};bundled_models",
    ]
    # Logo mitliefern
    if os.path.exists(icon_path):
        data_args.append(f"--add-data={icon_path};.")

    print(f"  [OK] {len(hidden_imports)} Hidden Imports konfiguriert")
    print(f"  [OK] bundled_models wird eingebunden")
    print()

    print("[3/3] Build starten (das dauert einige Minuten)...")
    print()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=WhisperX",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        *icon_arg,
        *hidden_import_args,
        *data_args,
        # Collect all packages for torch/torchaudio
        "--collect-all=torch",
        "--collect-all=torchaudio",
        "--collect-all=whisperx",
        "--collect-all=faster_whisper",
        "--collect-all=ctranslate2",
        "--collect-all=PySide6",
        "--collect-all=pyannote.audio",
        "--collect-all=speechbrain",
        os.path.join(BASE_DIR, "gui_qt.py"),
    ]

    print("  Befehl:", " ".join(cmd[:10]), "...")
    print()

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        print("\n  [FEHLER] Build fehlgeschlagen!")
        sys.exit(1)

    # bundled_models in dist-Ordner kopieren (fuer --onedir)
    dist_app_dir = os.path.join(DIST_DIR, "WhisperX")
    dist_models = os.path.join(dist_app_dir, "bundled_models")

    if not os.path.isdir(dist_models):
        print("\n  Kopiere bundled_models in dist/...")
        shutil.copytree(MODELS_DIR, dist_models)

    print()
    print("=" * 60)
    print("  Build erfolgreich!")
    print(f"  Ergebnis: {dist_app_dir}\\WhisperX.exe")
    print()
    print("  Den gesamten Ordner dist/WhisperX/ weitergeben.")
    print("  Die .exe startet direkt ohne Installation.")
    print("=" * 60)


if __name__ == "__main__":
    build()
