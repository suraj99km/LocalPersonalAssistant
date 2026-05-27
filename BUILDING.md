## Building for macOS (PyInstaller + DMG)

### Prerequisites
- macOS 12+
- Python **3.10+**
- Xcode Command Line Tools

Install Xcode CLT if needed:

```bash
xcode-select --install
```

### One-command build

From the repo root:

```bash
./build_macos.sh
```

Outputs:
- `dist/PersonalRAG.app`
- `dist/PersonalRAG.dmg`

### What the build script does
- Creates `venv/` (if missing)
- Installs dependencies
- Downloads the GGUF model via `download_model.py`
- Runs PyInstaller using `build_macos.spec`
- Wraps the `.app` into a `.dmg` using `hdiutil`

### Troubleshooting

#### GPU acceleration not active
GPU offload depends on how `llama-cpp-python` was built/installed. If the wheel you installed doesn’t include Metal support, it will fall back to CPU automatically.

#### PyInstaller fails to import a package
If you see missing module errors at runtime, add the package to `build_macos.spec`’s `collect_all(...)` list or add an explicit hidden import.

