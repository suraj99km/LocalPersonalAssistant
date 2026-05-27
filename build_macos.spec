# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(globals().get("SPECPATH") or Path.cwd()).resolve()

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
    collected_datas, collected_binaries, collected_hidden = collect_all(pkg)
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hidden

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

