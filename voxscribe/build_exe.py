"""Build script: creates the VoxScribe .exe with PyInstaller.

Usage:
    python build_exe.py

The result is in dist/VoxScribe/VoxScribe.exe

Deliberately builds "online-first" - without pre-embedding bundled_models/.
The app only downloads local models when actually needed (a configured
provider is the default, see transcriber.default_model_size()), and models
make up ~4-9 GB, the bulk of an embedded .exe's size. Anyone who also wants a
fully-offline-capable .exe can manually run `python download_models.py` after
the build and copy the resulting folder (bundled_models/) into
dist/VoxScribe/ - transcriber.py picks it up automatically (see
_get_bundled_models_dir()), no code change needed.
"""

import os
import sys
import subprocess

# This file lives in voxscribe/, the project root (gui_qt.py, Logo.png,
# dist/, build/) is one level up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")


def check_prerequisites():
    """Checks that all prerequisites are met."""
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  [!] PyInstaller not installed. Install with: pip install pyinstaller")
        sys.exit(1)


def get_hidden_imports():
    """List of hidden imports PyInstaller doesn't detect automatically."""
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
    """Runs the PyInstaller build."""
    print("=" * 60)
    print("  VoxScribe — .exe build")
    print("=" * 60)
    print()

    print("[1/3] Checking prerequisites...")
    check_prerequisites()
    print()

    print("[2/3] Building PyInstaller configuration...")

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
            print(f"  [!] Icon conversion failed: {e}")

    # Hidden imports
    hidden_imports = get_hidden_imports()
    hidden_import_args = []
    for hi in hidden_imports:
        hidden_import_args.extend(["--hidden-import", hi])

    # Data files - bundled_models is deliberately not embedded, see module
    # docstring ("online-first" build).
    data_args = []
    # Include the logo - gui_qt.py looks this up via __file__, which for the
    # frozen entry script resolves under _internal/, so this one DOES need
    # --add-data (unlike bundled_models above).
    if os.path.exists(icon_path):
        data_args.append(f"--add-data={icon_path};.")

    print(f"  [OK] {len(hidden_imports)} hidden imports configured")
    print()

    print("[3/3] Starting build (this takes a few minutes)...")
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

    print("  Command:", " ".join(cmd[:10]), "...")
    print()

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        print("\n  [ERROR] Build failed!")
        sys.exit(1)

    dist_app_dir = os.path.join(DIST_DIR, "VoxScribe")

    print()
    print("=" * 60)
    print("  Build successful!")
    print(f"  Result: {dist_app_dir}\\VoxScribe.exe")
    print()
    print("  Distribute the whole dist/VoxScribe/ folder.")
    print("  The .exe runs directly, no installation needed. Local models are")
    print("  downloaded automatically only when actually needed (a configured")
    print("  provider is the default and doesn't need any).")
    print("=" * 60)


if __name__ == "__main__":
    build()
