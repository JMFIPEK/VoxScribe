#!/bin/bash
# Creates VoxScribe.app: a double-clickable launcher (Dock, Spotlight,
# Launchpad) that starts this repo's GUI in place - the macOS counterpart of
# VoxScribe.bat + Create-Shortcuts.ps1.
#
# Usage:
#   macos/make-app.sh [target-folder]      (default: ~/Applications)
#
# The repo path is baked into the launcher - re-run this script if you move
# the repo. Safe to run again anytime; it replaces the existing app.
# Afterwards: drag VoxScribe from ~/Applications to the Dock (or open it once,
# right-click its Dock icon > Options > Keep in Dock).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${1:-$HOME/Applications}"
APP="$TARGET_DIR/VoxScribe.app"
BUNDLE_ID="app.voxscribe.VoxScribe"

UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
    echo "uv was not found. Install it once with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$REPO"
if [ ! -x .venv/bin/python ]; then
    echo "Creating the environment (uv sync)..."
    "$UV" sync
fi

echo "Rendering the icon..."
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ICONSET="$WORK/Logo.iconset"
mkdir -p "$ICONSET"
QT_QPA_PLATFORM=offscreen .venv/bin/python - "$ICONSET" <<'PY'
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from qt_app.app_icon import build_icon_pixmap

app = QGuiApplication([])
master = build_icon_pixmap("Logo.png", 1024)
for base in (16, 32, 128, 256, 512):
    for scale in (1, 2):
        px = base * scale
        name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
        master.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(
            f"{sys.argv[1]}/{name}")
PY
iconutil -c icns "$ICONSET" -o "$WORK/Logo.icns"

echo "Building $APP ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$WORK/Logo.icns" "$APP/Contents/Resources/Logo.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>VoxScribe</string>
    <key>CFBundleDisplayName</key><string>VoxScribe</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key><string>VoxScribe</string>
    <key>CFBundleIconFile</key><string>Logo</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>26.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>VoxScribe records your microphone to transcribe meetings.</string>
</dict>
</plist>
PLIST

# The launcher script itself is macos/launcher.sh (tests/test_macos_launcher.py
# runs it), with this repo and uv filled in.
LAUNCHER="$(<"$REPO/macos/launcher.sh")"
LAUNCHER="${LAUNCHER//@REPO@/$REPO}"
printf '%s\n' "${LAUNCHER//@UV@/$UV}" > "$APP/Contents/MacOS/VoxScribe"
chmod +x "$APP/Contents/MacOS/VoxScribe"

# Ad-hoc signature: gives the bundle a stable identity so macOS keeps its
# microphone / screen-recording permissions between launches.
codesign --force --deep --sign - "$APP"

echo
echo "Done: $APP"
echo "Open it from Spotlight/Launchpad, then keep it in the Dock."
