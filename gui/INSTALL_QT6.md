# Installing Qt6 for nvCOMP GUI

## Automatic Installation (Recommended - NEW!)

The CMakeLists.txt now supports **automatic Qt6 installation** during CMake configuration!

### Prerequisites for Automatic Install

- Python 3.6+ installed and in PATH
- Internet connection (~1-2 GB download)
- ~5 GB free disk space

### Just Run CMake!

```powershell
# Windows - CMake will auto-install Qt6 if not found
cmake -B build_gui -DBUILD_GUI=ON

# It will:
# 1. Check for existing Qt6
# 2. If not found, install Python package 'aqtinstall'
# 3. Download Qt 6.7.3 to build_gui/qt6/
# 4. Configure automatically
```

**That's it!** CMake handles everything. First run takes 5-10 minutes for Qt download.

### What Happens During Auto-Install

```
-- Searching for Qt6...
-- Qt6 not found. Attempting automatic installation...
-- Checking for aqtinstall...
-- Installing aqtinstall (Qt installer tool)...
-- Downloading Qt 6.7.3 for windows win64_msvc2019_64...
-- This may take several minutes (Qt is ~1-2 GB)...
-- Installation directory: C:/Git/nvCOMP_CLI/build_gui/qt6
-- Qt6 installed successfully
```

Qt is installed to: `build_gui/qt6/6.7.3/<arch>/`

---

## Manual Installation (If Auto-Install Fails)

### Option 1: Qt Online Installer

1. **Download Qt Online Installer**
   - Visit: https://www.qt.io/download-open-source
   - Click "Download the Qt Online Installer"
   - Choose Windows installer

2. **Run Installer**
   - Create a Qt account (free for open source)
   - Choose installation directory (e.g., `C:\Qt`)
   - Select components:
     - ✅ Qt 6.7.3 or 6.8.0 (any 6.x version)
       - ✅ MSVC 2019 64-bit (or MSVC 2022 64-bit)
       - ✅ Desktop development
     - ✅ Qt Creator (optional, helpful for UI editing)

3. **Configure CMake with Qt Path**
   
   ```powershell
   cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.7.3/msvc2019_64"
   ```

### Option 2: vcpkg Package Manager

```powershell
# Install vcpkg (if not already installed)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install Qt6
.\vcpkg install qt6-base:x64-windows
.\vcpkg install qt6-widgets:x64-windows

# Integrate with CMake
.\vcpkg integrate install

# Build with vcpkg toolchain
cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_TOOLCHAIN_FILE="[vcpkg-root]/scripts/buildsystems/vcpkg.cmake"
```

### Option 3: Manual aqtinstall (Advanced)

```powershell
# Install aqtinstall
pip install aqtinstall

# Download Qt 6.7.3 for Windows MSVC 2019
python -m aqt install-qt windows desktop 6.7.3 win64_msvc2019_64 -O C:\Qt

# Configure CMake
cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.7.3/win64_msvc2019_64"
```

---

## Troubleshooting

### "Python3 not found"

Install Python 3.6+:
- Download from: https://www.python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation
- Verify: `python --version`

### "aqtinstall installation failed"

Try manual pip install:

```powershell
pip install aqtinstall
# Or with Python module syntax:
python -m pip install aqtinstall
```

### "Failed to install Qt6 automatically"

Fall back to manual Qt Online Installer (Option 1 above) or:

```powershell
# Skip GUI for now
cmake -B build -DBUILD_GUI=OFF
```

### Qt DLLs Not Found at Runtime

After building, deploy Qt dependencies:

```powershell
cd build_gui\gui\Release
C:\Qt\6.7.3\msvc2019_64\bin\windeployqt.exe nvcomp_gui.exe

# Or if auto-installed:
build_gui\qt6\6.7.3\msvc2019_64\bin\windeployqt.exe nvcomp_gui.exe
```

### Wrong Compiler Version

Auto-install uses MSVC 2019 binaries (compatible with VS 2022).
For manual install, ensure Qt compiler matches Visual Studio:
- VS 2019 → use `msvc2019_64`
- VS 2022 → use `msvc2019_64` (2022 uses 2019 binaries)

---

## Verifying Qt Installation

```powershell
# Check if qmake is available
qmake --version

# Or find Qt6Config.cmake
dir "C:\Qt\6.7.3\msvc2019_64\lib\cmake\Qt6" -Recurse -Filter "Qt6Config.cmake"

# For auto-installed Qt:
dir "build_gui\qt6\6.7.3\*\lib\cmake\Qt6" -Recurse -Filter "Qt6Config.cmake"
```

---

## Common Installation Paths

**Auto-installed (CMake):**
- `<build_dir>/qt6/6.7.3/<arch>/`
- Example: `build_gui/qt6/6.7.3/win64_msvc2019_64/`

**Manual Qt Installer:**
- Windows: `C:\Qt\<version>\<compiler>_64\`
- Linux: `~/Qt/<version>/gcc_64/`

---

## Building After Qt Installation

```powershell
# Configure (auto-installs Qt if needed)
cmake -B build_gui -DBUILD_GUI=ON

# Build
cmake --build build_gui --config Release

# Run GUI
.\build_gui\gui\Release\nvcomp_gui.exe

# Run tests
.\build_gui\gui\Release\test_mainwindow.exe
```

---

## Requirements

- **Version**: Qt 6.2 or later (Qt 6.7.3 recommended)
- **Modules**: Core, Widgets, Gui, Test
- **Compiler**: 
  - Windows: MSVC 2019/2022
  - Linux: GCC 9+ or Clang 10+
- **C++ Standard**: C++17
- **Disk Space**: ~2 GB for Qt, ~5 GB total with build artifacts

---

## Skip GUI Build

If you want to build CLI only without Qt:

```powershell
cmake -B build -DBUILD_GUI=OFF
cmake --build build --config Release
```

---

## Technical Details: Automatic Installation

The CMakeLists.txt uses `aqtinstall` (Another Qt Installer) which:

1. **Lightweight**: Pure Python, no GUI installer needed
2. **Fast**: Downloads only required components (~1 GB vs 4+ GB)
3. **Automated**: Perfect for CI/CD and build systems
4. **Reliable**: Official mirrors from Qt servers

More info: https://github.com/miurahr/aqtinstall

---

## Next Steps

Once Qt is installed and CMake succeeds:

1. **Build**: `cmake --build build_gui --config Release`
2. **Run GUI**: `.\build_gui\gui\Release\nvcomp_gui.exe`
3. **Run Tests**: `.\build_gui\gui\Release\test_mainwindow.exe`
4. **Edit UI**: Use Qt Designer on `.ui` files

Task 2.1 is complete - ready for Task 2.2!
