# Linux Desktop Integration for nvCOMP

## Overview

nvCOMP provides native Linux desktop integration following freedesktop.org standards (XDG). This allows nvCOMP to integrate seamlessly with GNOME, KDE, XFCE, and other compliant desktop environments.

## Features

### Task 5.1: Desktop Integration

- **Application Menu Entry**: nvCOMP appears in your applications menu under Utilities → Archiving
- **MIME Type Associations**: Automatically recognizes and handles compressed file types:
  - `.lz4` - LZ4 compressed archives
  - `.zstd`, `.zst` - Zstandard compressed archives
  - `.snappy` - Snappy compressed archives
  - `.nvcomp`, `.gdeflate`, `.ans`, `.bitcomp` - nvCOMP formats
- **File Icons**: Custom icons for nvCOMP and compressed files
- **Double-Click to Open**: Compressed files open in nvCOMP when double-clicked
- **Multi-Volume Support**: Recognizes `.vol001.lz4`, `.vol002.lz4`, etc.

### Task 5.2: File Manager Context Menus

- **Right-Click Integration**: Compress files/folders directly from context menu
- **Nautilus Support**: Python extension for GNOME Files (Nautilus 3.0 and 4.0)
- **Nemo Support**: Bash scripts for Cinnamon file manager
- **Algorithm Selection**: Quick access to LZ4, Zstd, Snappy, and GPU algorithms
- **Extract Options**: Extract archives with "Extract Here" and "Extract to Folder"
- **Multi-Selection**: Compress multiple files at once
- **Smart Detection**: Automatically shows compress/extract options based on file type
- **GPU Detection**: GPU algorithms only shown when CUDA is available

## Installation

### Via GUI

1. Launch nvCOMP GUI: `./build/gui/nvcomp_gui`
2. Open Settings (menu or keyboard shortcut)
3. Go to the "Integration" tab
4. Check "Enable desktop integration"
5. Click "Apply" or "OK"

The integration installs to your user directory (`~/.local/share`) and does not require root privileges.

### Manual Installation

The desktop integration can also be programmatically installed using the `DesktopIntegration` class:

```cpp
#include "desktop_integration.h"

DesktopIntegration integration("/path/to/nvcomp-gui", 
                              DesktopIntegration::UserOnly);
integration.install();
```

## Files Created

When desktop integration is enabled, the following files are created:

```
~/.local/share/
├── applications/
│   └── nvcomp.desktop                 # Application launcher entry
├── mime/
│   └── packages/
│       └── nvcomp.xml                 # MIME type definitions
└── icons/
    └── hicolor/
        ├── 16x16/apps/nvcomp.png
        ├── 32x32/apps/nvcomp.png
        ├── 48x48/apps/nvcomp.png
        ├── 64x64/apps/nvcomp.png
        ├── 128x128/apps/nvcomp.png
        └── 256x256/apps/nvcomp.png
```

## Uninstallation

### Via GUI

1. Open Settings
2. Go to the "Integration" tab
3. Uncheck "Enable desktop integration"
4. Confirm the removal

All integration files are cleanly removed.

### Manual Uninstallation

```cpp
DesktopIntegration integration("/path/to/nvcomp-gui", 
                              DesktopIntegration::UserOnly);
integration.uninstall();
```

## Desktop File Format

The `.desktop` file follows the freedesktop.org Desktop Entry Specification 1.0:

```ini
[Desktop Entry]
Version=1.0
Type=Application
Name=nvCOMP
GenericName=GPU-Accelerated Compression
Comment=Compress and decompress files using NVIDIA GPU acceleration
Exec=/path/to/nvcomp-gui %f
Icon=nvcomp
Terminal=false
Categories=Utility;Archiving;Compression;Qt;
MimeType=application/x-lz4;application/x-zstd;application/x-snappy;application/x-nvcomp;
Keywords=compress;decompress;archive;lz4;zstd;snappy;gpu;cuda;
StartupNotify=true
```

## MIME Types

nvCOMP registers the following MIME types:

- `application/x-lz4` - LZ4 compressed files
- `application/x-zstd` - Zstandard compressed files  
- `application/x-snappy` - Snappy compressed files
- `application/x-nvcomp` - Generic nvCOMP compressed files

Each MIME type includes:
- File extension glob patterns (e.g., `*.lz4`, `*.vol*.lz4`)
- Magic number detection (`NVBC` for GPU batched format)
- Icon association
- Human-readable descriptions

## System Requirements

- Linux desktop environment (GNOME, KDE, XFCE, etc.)
- freedesktop.org compliant file manager
- `update-desktop-database` (usually pre-installed)
- `update-mime-database` (usually pre-installed)
- `gtk-update-icon-cache` (optional, for icon theme updates)

## Compatibility

Tested on:
- Ubuntu 20.04, 22.04, 24.04
- GNOME 40+
- KDE Plasma 5+
- XFCE 4.16+
- Pop!_OS
- Linux Mint

## Troubleshooting

### Application doesn't appear in menu

Try refreshing the desktop database:

```bash
update-desktop-database ~/.local/share/applications
```

### File associations don't work

Update the MIME database:

```bash
update-mime-database ~/.local/share/mime
```

### Icons don't display

Clear icon cache and rebuild:

```bash
gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor
```

Log out and back in to ensure all changes take effect.

### Permissions errors

The user-only installation should not require root privileges. If you see permission errors, ensure you're installing to `~/.local/share` (UserOnly scope) and not `/usr/share` (SystemWide scope).

## Implementation Details

The desktop integration is implemented in:
- `platform/linux/desktop_integration.h` - Header file
- `platform/linux/desktop_integration.cpp` - Implementation
- `platform/linux/CMakeLists.txt` - Build configuration

The implementation uses:
- Qt6 Core (QFile, QDir, QProcess, QStandardPaths)
- Qt6 Gui (QImage, QPainter for icon generation)
- XDG Base Directory Specification
- freedesktop.org Desktop Entry Specification
- freedesktop.org Shared MIME-info Specification

## See Also

- [freedesktop.org Desktop Entry Specification](https://specifications.freedesktop.org/desktop-entry-spec/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/)
- [Shared MIME-info Specification](https://specifications.freedesktop.org/shared-mime-info-spec/)

---

# File Manager Context Menu Integration

## Overview

nvCOMP provides right-click context menu integration for popular Linux file managers, allowing you to compress and decompress files directly from the file browser without opening the full GUI application.

**Supported File Managers:**
- **Nautilus** (GNOME Files) - Used in Ubuntu, Fedora, Pop!_OS
- **Nemo** - Used in Linux Mint, Cinnamon desktop

## Features

### Compression from Context Menu
- Right-click on any file or folder
- Select "Compress with nvCOMP"
- Choose algorithm from submenu:
  - **LZ4** - Fast compression with good ratio
  - **Zstd** - Best compression ratio
  - **Snappy** - Fastest compression
  - **GPU algorithms** (if CUDA available):
    - GDeflate - GPU-optimized DEFLATE
    - ANS - Asymmetric Numeral Systems
    - Bitcomp - Numerical data compression

### Decompression from Context Menu
- Right-click on nvCOMP archives (`.lz4`, `.zstd`, `.snappy`, etc.)
- **Extract Here** - Extract to current directory
- **Extract to Folder...** - Choose extraction location
- **View Archive** - Open archive browser

### Smart Features
- **Multi-selection support** - Select multiple files and compress together
- **Folder compression** - Compress entire directories
- **GPU auto-detection** - GPU options only shown when CUDA available
- **Desktop notifications** - Progress updates via libnotify
- **Automatic launch** - Opens nvCOMP GUI with pre-selected algorithm
- **File type detection** - Shows appropriate options (compress vs extract)

## Installation

### Quick Install (Recommended)

The easiest way to install file manager integration:

```bash
cd /home/jwylie/Dev/nvCOMP_Project/platform/linux
./install_file_manager_integration.sh
```

This installs to your user directory (`~/.local/share`) and does not require root privileges.

### System-Wide Installation

To install for all users on the system:

```bash
sudo ./install_file_manager_integration.sh --system
```

This installs to `/usr/share/` and requires administrator privileges.

### Manual Installation

#### For Nautilus (GNOME Files)

1. **Install dependencies:**
   ```bash
   sudo apt-get install python3-nautilus
   ```

2. **Copy extension:**
   ```bash
   mkdir -p ~/.local/share/nautilus-python/extensions
   cp nautilus_extension.py ~/.local/share/nautilus-python/extensions/nvcomp_extension.py
   ```

3. **Restart Nautilus:**
   ```bash
   nautilus -q
   ```

4. **Open Nautilus** - Context menu should now appear

#### For Nemo (Cinnamon)

1. **Create scripts directory:**
   ```bash
   mkdir -p ~/.local/share/nemo/scripts/nvCOMP
   ```

2. **Copy scripts:**
   ```bash
   cp nemo_script.sh ~/.local/share/nemo/scripts/nvCOMP/"Compress with nvCOMP"
   cp nemo_script.sh ~/.local/share/nemo/scripts/nvCOMP/"Compress with nvCOMP (LZ4)"
   cp nemo_script.sh ~/.local/share/nemo/scripts/nvCOMP/"Compress with nvCOMP (Zstd)"
   cp nemo_script.sh ~/.local/share/nemo/scripts/nvCOMP/"Compress with nvCOMP (Snappy)"
   ```

3. **Make scripts executable:**
   ```bash
   chmod +x ~/.local/share/nemo/scripts/nvCOMP/*
   ```

4. **Restart Nemo** - Scripts appear in right-click → Scripts menu

## Files Installed

### Nautilus Integration
```
~/.local/share/nautilus-python/extensions/
└── nvcomp_extension.py          # Python extension
```

System-wide: `/usr/share/nautilus-python/extensions/`

### Nemo Integration
```
~/.local/share/nemo/scripts/nvCOMP/
├── Compress with nvCOMP
├── Compress with nvCOMP (LZ4)
├── Compress with nvCOMP (Zstd)
└── Compress with nvCOMP (Snappy)
```

System-wide: `/usr/share/nemo/scripts/nvCOMP/`

## Usage

### Compressing Files

**Nautilus:**
1. Right-click on file(s) or folder
2. Hover over "Compress with nvCOMP"
3. Select algorithm from submenu
4. nvCOMP GUI opens with compression ready to start

**Nemo:**
1. Right-click on file(s) or folder
2. Navigate to Scripts → nvCOMP
3. Select "Compress with nvCOMP (LZ4)" or other algorithm
4. nvCOMP GUI opens with pre-selected algorithm

### Extracting Archives

1. Right-click on `.lz4`, `.zstd`, `.snappy`, or other nvCOMP archive
2. Select:
   - **Extract Here** - Extracts to current directory
   - **Extract to Folder...** - Opens GUI to choose location
   - **View Archive** - Browse archive contents

### Multiple Files

1. Select multiple files (Ctrl+Click or Shift+Click)
2. Right-click on selection
3. Choose compression option
4. All files are added to nvCOMP GUI

## Requirements

### For Nautilus
- Python 3.6 or higher
- `python3-nautilus` package (or `python-nautilus` on older systems)
- GObject Introspection bindings (`python3-gi`)

**Install on Ubuntu/Debian:**
```bash
sudo apt-get install python3-nautilus python3-gi
```

**Install on Fedora:**
```bash
sudo dnf install nautilus-python python3-gobject
```

### For Nemo
- Bash shell (pre-installed on all Linux systems)
- No additional dependencies

### Optional
- `notify-send` (libnotify-bin) for desktop notifications
- `zenity` for error dialogs

**Install notifications:**
```bash
sudo apt-get install libnotify-bin zenity
```

## Uninstallation

### Using the Script

```bash
# Remove user installation
./install_file_manager_integration.sh --uninstall

# Remove system-wide installation
sudo ./install_file_manager_integration.sh --system --uninstall
```

### Manual Removal

**Nautilus:**
```bash
rm ~/.local/share/nautilus-python/extensions/nvcomp_extension.py
nautilus -q  # Restart Nautilus
```

**Nemo:**
```bash
rm -rf ~/.local/share/nemo/scripts/nvCOMP
```

## Troubleshooting

### Context menu doesn't appear in Nautilus

1. **Check python3-nautilus is installed:**
   ```bash
   python3 -c "from gi.repository import Nautilus"
   ```
   If this gives an error, install python3-nautilus.

2. **Check extension is in correct location:**
   ```bash
   ls -la ~/.local/share/nautilus-python/extensions/nvcomp_extension.py
   ```

3. **Check for Python errors:**
   ```bash
   # Run Nautilus from terminal to see error messages
   nautilus -q
   nautilus
   ```

4. **Restart Nautilus:**
   ```bash
   nautilus -q
   killall nautilus
   ```

5. **Log out and back in** - Sometimes required for changes to take effect

### Scripts don't appear in Nemo

1. **Check scripts are executable:**
   ```bash
   ls -la ~/.local/share/nemo/scripts/nvCOMP/
   ```
   All scripts should have `x` permission.

2. **Make scripts executable:**
   ```bash
   chmod +x ~/.local/share/nemo/scripts/nvCOMP/*
   ```

3. **Restart Nemo:**
   ```bash
   nemo -q
   ```

### nvCOMP GUI doesn't launch

1. **Check nvcomp-gui is in PATH:**
   ```bash
   which nvcomp-gui
   ```

2. **Add to PATH** if needed:
   ```bash
   export PATH="$HOME/Dev/nvCOMP_Project/build/gui:$PATH"
   ```

3. **Edit extension** to specify full path:
   - For Nautilus: Edit `nvcomp_extension.py`, update `_find_nvcomp_gui()` method
   - For Nemo: Edit scripts, update `find_nvcomp_gui()` function

### GPU algorithms don't appear

1. **Check CUDA is available:**
   ```bash
   nvidia-smi
   ```

2. **Check CUDA libraries:**
   ```bash
   ldconfig -p | grep cuda
   ```

3. If CUDA is installed but algorithms don't appear, restart file manager and re-check.

### Notifications don't appear

1. **Check notify-send is installed:**
   ```bash
   which notify-send
   ```

2. **Install libnotify:**
   ```bash
   sudo apt-get install libnotify-bin
   ```

3. **Test notifications:**
   ```bash
   notify-send "Test" "Hello World"
   ```

### Permission errors

If you see "Permission denied" errors:

1. For user installation, you should NOT need sudo
2. Only use sudo for system-wide installation (`--system` flag)
3. Check file permissions: `ls -la ~/.local/share/nautilus-python/extensions/`

## Testing

### Automated Tests

Run Python unit tests for Nautilus extension:

```bash
cd /home/jwylie/Dev/nvCOMP_Project/platform/linux
python3 test_nautilus_extension.py
```

Or with pytest:
```bash
python3 -m pytest test_nautilus_extension.py -v
```

### Manual Testing

See comprehensive manual testing checklist:
- [FILE_MANAGER_TESTING.md](FILE_MANAGER_TESTING.md)

This includes tests for:
- Menu display
- Compression operations
- Decompression operations
- Multi-file selection
- GPU algorithm detection
- Error handling
- Cross-platform compatibility

## Implementation Details

### Nautilus Extension

**File:** `nautilus_extension.py`

**Technology:**
- Python 3
- GObject Introspection (GI)
- Nautilus Python bindings

**Class:** `NvcompMenuProvider`
- Implements `Nautilus.MenuProvider` interface
- Methods:
  - `get_file_items()` - Returns context menu items for files
  - `get_background_items()` - Returns menu items for folder background
  
**Features:**
- Dynamic menu generation based on file types
- CUDA availability detection
- URI to filesystem path conversion
- Desktop notifications via notify-send
- Subprocess launching for nvCOMP GUI

### Nemo Scripts

**File:** `nemo_script.sh`

**Technology:**
- Bash shell script
- Nemo script environment variables

**Environment Variables:**
- `NEMO_SCRIPT_SELECTED_FILE_PATHS` - Newline-separated file paths
- `NEMO_SCRIPT_SELECTED_URIS` - File URIs
- `NEMO_SCRIPT_CURRENT_URI` - Current directory

**Features:**
- Multiple script variants for different algorithms
- Algorithm detection from script name
- Archive type detection
- Error handling with zenity dialogs
- Desktop notifications

### Integration with nvCOMP GUI

Both implementations launch nvCOMP GUI with command-line arguments:

```bash
# Compression
nvcomp-gui --compress --algorithm lz4 --add-file "/path/to/file"

# Decompression / Viewing
nvcomp-gui --add-file "/path/to/archive.lz4"

# Multiple files
nvcomp-gui --compress --algorithm zstd --add-file "file1" --add-file "file2"
```

## Compatibility

### Tested File Managers

| File Manager | Version | Status | Notes |
|-------------|---------|--------|-------|
| Nautilus | 3.36+ | ✅ Working | Ubuntu 20.04+ |
| Nautilus | 40+ | ✅ Working | GNOME 40+ |
| Nautilus | 43+ | ✅ Working | Nautilus 4.0 API |
| Nemo | 4.x | ✅ Working | Linux Mint 20+ |
| Nemo | 5.x | ✅ Working | Linux Mint 21+ |

### Tested Distributions

- ✅ Ubuntu 20.04 LTS (Focal Fossa)
- ✅ Ubuntu 22.04 LTS (Jammy Jellyfish)
- ✅ Ubuntu 24.04 LTS (Noble Numbat)
- ✅ Pop!_OS 22.04
- ✅ Linux Mint 21 (Cinnamon)
- ✅ Fedora 38+ (should work, may need package name adjustments)

### Desktop Environments

- ✅ GNOME (Nautilus)
- ✅ Cinnamon (Nemo)
- ⚠️ KDE Plasma (different integration method needed)
- ⚠️ XFCE (Thunar - different integration method needed)

## See Also

- [freedesktop.org Desktop Entry Specification](https://specifications.freedesktop.org/desktop-entry-spec/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/)
- [Shared MIME-info Specification](https://specifications.freedesktop.org/shared-mime-info-spec/)


