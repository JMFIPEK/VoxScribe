"""Build-Skript: Erstellt die VoxScribe .exe mit PyInstaller.

Verwendung:
    python build_exe.py

Das Ergebnis liegt in dist/VoxScribe/VoxScribe.exe

Baut bewusst "online-first" - ohne bundled_models/ vorab einzubetten. Die App
laedt lokale Modelle ohnehin nur bei Bedarf herunter (KIT ToolBox (Server) ist
der Default, siehe transcriber.default_model_size()), und die Modelle machen
mit ~4-9 GB den grossen Teil der Groesse einer eingebetteten .exe aus. Wer
zusaetzlich eine volloffline-faehige .exe will, kann nach dem Build manuell
`python download_models.py` ausfuehren und den Ergebnisordner
(bundled_models/) in dist/VoxScribe/ hineinkopieren - transcriber.py erkennt
ihn automatisch (siehe _get_bundled_models_dir()), ganz ohne Code-Aenderung.
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")


def check_prerequisites():
    """Prueft ob alle Voraussetzungen erfuellt sind."""
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  [!] PyInstaller nicht installiert. Installiere mit: pip install pyinstaller")
        sys.exit(1)


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
        # Intel Arc GPU transcription backend (transcriber.transcribe_openvino()) -
        # imported lazily inside that function, not at module load, so PyInstaller's
        # static import scan needs the explicit hint.
        "optimum",
        "optimum.intel",
        "optimum.intel.openvino",
        "openvino",
    ]


def build():
    """Fuehrt den PyInstaller-Build aus."""
    print("=" * 60)
    print("  VoxScribe — .exe Build")
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

    # Data files - bundled_models is deliberately not embedded, see module
    # docstring ("online-first" build).
    data_args = []
    # Logo mitliefern - gui_qt.py looks this up via __file__, which for the
    # frozen entry script resolves under _internal/, so this one DOES need
    # --add-data (unlike bundled_models above).
    if os.path.exists(icon_path):
        data_args.append(f"--add-data={icon_path};.")

    print(f"  [OK] {len(hidden_imports)} Hidden Imports konfiguriert")
    print()

    print("[3/3] Build starten (das dauert einige Minuten)...")
    print()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=VoxScribe",
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
        # Intel Arc GPU backend: openvino ships its inference-device backend
        # plugins (CPU/GPU/NPU) as native DLLs PyInstaller's default import scan
        # won't discover on its own - needs the explicit collect-all, same as
        # torch/ctranslate2 above. Optional at runtime (see hardware_detect.py's
        # guarded import) so this doesn't break the build on non-Windows.
        "--collect-all=openvino",
        "--collect-all=optimum",
        os.path.join(BASE_DIR, "gui_qt.py"),
    ]

    print("  Befehl:", " ".join(cmd[:10]), "...")
    print()

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        print("\n  [FEHLER] Build fehlgeschlagen!")
        sys.exit(1)

    dist_app_dir = os.path.join(DIST_DIR, "VoxScribe")

    print()
    print("=" * 60)
    print("  Build erfolgreich!")
    print(f"  Ergebnis: {dist_app_dir}\\VoxScribe.exe")
    print()
    print("  Den gesamten Ordner dist/VoxScribe/ weitergeben.")
    print("  Die .exe startet direkt ohne Installation. Lokale Modelle werden")
    print("  automatisch heruntergeladen, sobald sie tatsaechlich gebraucht")
    print("  werden (KIT ToolBox (Server) ist der Standard und braucht keine).")
    print("=" * 60)


if __name__ == "__main__":
    build()
