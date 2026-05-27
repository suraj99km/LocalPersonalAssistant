# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(__file__).resolve().parent

datas = []
hiddenimports = []
binaries = []

for pkg in [
    "streamlit",
    "chromadb",
    "sentence_transformers",
    "llama_cpp",
    "watchdog",
    "fitz",  # PyMuPDF
    "docx",  # python-docx
]:
    collected = collect_all(pkg)
    datas += collected.datas
    binaries += collected.binaries
    hiddenimports += collected.hiddenimports

# Bundle Streamlit config (model is downloaded on first run)
datas += [
    (str(project_root / ".streamlit" / "config.toml"), ".streamlit"),
]


a = Analysis(
    ["run_app.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PersonalRAG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

app = BUNDLE(
    exe,
    name="PersonalRAG.app",
    bundle_identifier="com.personalrag.app",
)

