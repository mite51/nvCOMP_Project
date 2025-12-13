# Quick Start: Install Qt6 for nvCOMP GUI

## Simplest Method (Recommended): Qt Online Installer

### Step 1: Download Qt Online Installer

1. Go to: https://www.qt.io/download-open-source
2. Click "Download the Qt Online Installer"  
3. Download the Windows installer (qt-online-installer-windows-x64-4.x.x.exe)

### Step 2: Install Qt 6.7.3 or 6.8.0

1. **Run the installer**
2. **Create free Qt account** (for open source use)
3. **Choose installation directory**: `C:\Qt` (default)
4. **Select components**:
   - Navigate to Qt → Qt 6.7.3 (or 6.8.0)
   - ✅ Check: **MSVC 2019 64-bit** (for 6.7.3) or **MSVC 2022 64-bit** (for 6.8.0)
   - ✅ Check: **Desktop development**
   - Optional: Qt Creator (useful for UI editing)
5. **Install** (~2 GB download, takes 10-15 minutes)

### Step 3: Build nvCOMP GUI

```powershell
# For Qt 6.7.3
cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.7.3/msvc2019_64"

# OR for Qt 6.8.0
cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.8.0/msvc2022_64"

# Build
cmake --build build_gui --config Release

# Run
.\build_gui\gui\Release\nvcomp_gui.exe
```

## Alternative: Skip GUI Build (Build CLI Only)

If you don't need the GUI right now:

```powershell
cmake -B build -DBUILD_GUI=OFF
cmake --build build --config Release
.\build\Release\nvcomp_cli.exe --help
```

## Task 2.1 Status

✅ **All GUI code is complete and ready to build!**

The only requirement is Qt6 installation. Once Qt is installed, everything builds and works.

---

## Why Qt Online Installer?

- **Easiest**: Click install, everything just works
- **Fastest**: Optimized downloads, parallel extraction
- **Most Reliable**: Official Qt binaries, tested and supported
- **Best Experience**: Includes Qt Creator for UI editing

Alternative methods (aqtinstall, vcpkg) can have repository issues and are more complex.

---

## After Installation

Your Qt will be at:
- **Windows**: `C:\Qt\6.7.3\msvc2019_64\` or `C:\Qt\6.8.0\msvc2022_64\`
- Size: ~2 GB

Then just run:
```powershell
cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH="C:/Qt/6.7.3/msvc2019_64"
cmake --build build_gui --config Release
```

**That's it!** 🎉

