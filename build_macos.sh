#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "==> Local Private Assistant: macOS build"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_OK="$(python3 -c 'import sys; print(int(sys.version_info >= (3,10)))')"
if [[ "$PY_OK" != "1" ]]; then
  echo "ERROR: Python 3.10+ required. Found: ${PY_VER}" >&2
  exit 1
fi

if [[ ! -d "venv" ]]; then
  echo "==> Creating venv/"
  python3 -m venv venv
fi

echo "==> Activating venv"
# shellcheck disable=SC1091
source "venv/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Downloading model (if missing)"
python download_model.py

echo "==> Building .app via PyInstaller"
pyinstaller --noconfirm build_macos.spec

APP_PATH="dist/PersonalRAG.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: Build succeeded but app not found at $APP_PATH" >&2
  exit 1
fi

DMG_PATH="dist/PersonalRAG.dmg"
echo "==> Creating DMG at $DMG_PATH"
rm -f "$DMG_PATH"

VOL_NAME="PersonalRAG"
TMP_DMG="dist/.tmp_personalrag.dmg"
rm -f "$TMP_DMG"

hdiutil create -volname "$VOL_NAME" -srcfolder "$APP_PATH" -ov -format UDZO "$TMP_DMG" >/dev/null
mv "$TMP_DMG" "$DMG_PATH"

echo
echo "✅ Build complete"
echo "App: $APP_PATH"
echo "DMG: $DMG_PATH"

