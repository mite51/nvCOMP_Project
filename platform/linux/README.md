# Linux Desktop Integration for nvCOMP

## Overview

nvCOMP provides native Linux desktop integration following freedesktop.org standards (XDG). This allows nvCOMP to integrate seamlessly with GNOME, KDE, XFCE, and other compliant desktop environments.

## Features

- **Application Menu Entry**: nvCOMP appears in your applications menu under Utilities → Archiving
- **MIME Type Associations**: Automatically recognizes and handles compressed file types:
  - `.lz4` - LZ4 compressed archives
  - `.zstd`, `.zst` - Zstandard compressed archives
  - `.snappy` - Snappy compressed archives
  - `.nvcomp`, `.gdeflate`, `.ans`, `.bitcomp` - nvCOMP formats
- **File Icons**: Custom icons for nvCOMP and compressed files
- **Double-Click to Open**: Compressed files open in nvCOMP when double-clicked
- **Multi-Volume Support**: Recognizes `.vol001.lz4`, `.vol002.lz4`, etc.

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


