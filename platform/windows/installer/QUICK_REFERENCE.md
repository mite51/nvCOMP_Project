# WiX Installer Quick Reference

## Build the Installer

```bash
# Option 1: Using build script (recommended)
cd platform/windows/installer
build_installer.bat Release

# Option 2: Using CMake
cd build_gui
cmake --build . --target installer --config Release

# Option 3: Manual
cd platform/windows/installer
candle.exe -arch x64 -ext WixUIExtension installer.wxs components.wxs ui.wxs
light.exe -ext WixUIExtension -out nvCOMP-4.0.0-x64.msi installer.wixobj components.wixobj ui.wixobj -sval
```

## Install

```bash
# Interactive
nvCOMP-4.0.0-x64.msi

# Silent (all features)
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn /norestart

# Custom directory
msiexec /i nvCOMP-4.0.0-x64.msi /quiet INSTALLDIR="D:\nvCOMP" /norestart

# No optional features
msiexec /i nvCOMP-4.0.0-x64.msi /quiet INSTALL_CLI=0 INSTALL_CONTEXT_MENU=0 /norestart

# With logging
msiexec /i nvCOMP-4.0.0-x64.msi /l*v install.log
```

## Uninstall

```bash
# Interactive
msiexec /x nvCOMP-4.0.0-x64.msi

# Silent
msiexec /x nvCOMP-4.0.0-x64.msi /quiet /qn /norestart
```

## Installation Properties

| Property | Values | Default | Description |
|----------|--------|---------|-------------|
| `INSTALLDIR` | Path | `C:\Program Files\nvCOMP\nvCOMP` | Installation directory |
| `INSTALL_GUI` | 0/1 | 1 | Install GUI (required) |
| `INSTALL_CLI` | 0/1 | 1 | Install CLI tools |
| `INSTALL_CONTEXT_MENU` | 0/1 | 1 | Register context menu |
| `INSTALL_FILE_ASSOCIATIONS` | 0/1 | 1 | Register file associations |
| `INSTALL_START_MENU_SHORTCUTS` | 0/1 | 1 | Create Start Menu shortcuts |
| `INSTALL_DESKTOP_SHORTCUT` | 0/1 | 0 | Create desktop shortcut |

## Features

| Feature ID | Name | Description | Default |
|------------|------|-------------|---------|
| `ProductFeature` | nvCOMP GUI | Main application (required) | Yes |
| `CliFeature` | CLI Tools | Command-line tools | Yes |
| `ContextMenuFeature` | Context Menu | Explorer integration | Yes |
| `FileAssociationFeature` | File Associations | Associate file types | Yes |
| `StartMenuFeature` | Start Menu | Start Menu shortcuts | Yes |
| `DesktopShortcutFeature` | Desktop Shortcut | Desktop shortcut | No |

## Files Installed

### Executables
- `nvcomp-gui.exe` - Main GUI application
- `nvcomp-cli.exe` - Command-line tool (optional)

### Libraries
- `nvcomp_core.dll` - Core compression library
- `nvcomp64_5.dll` - NVIDIA GPU compression
- `nvcomp_cpu64_5.dll` - NVIDIA CPU fallback

### Qt Dependencies
- `Qt6Core.dll`, `Qt6Gui.dll`, `Qt6Widgets.dll`
- `platforms/qwindows.dll`
- `styles/qwindowsvistastyle.dll`
- `imageformats/*.dll`

## Registry Entries

### Product Information
```
HKEY_LOCAL_MACHINE\Software\nvCOMP\nvCOMP
├── InstallPath = "C:\Program Files\nvCOMP\nvCOMP"
├── Version = "4.0.0"
├── GuiExePath = "C:\Program Files\nvCOMP\nvCOMP\nvcomp-gui.exe"
└── CliExePath = "C:\Program Files\nvCOMP\nvCOMP\nvcomp-cli.exe"
```

### Uninstall Information
```
HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall\nvCOMP
└── (Auto-managed by Windows Installer)
```

## Shortcuts Created

### Start Menu
- `nvCOMP` → Launch GUI
- `Uninstall nvCOMP` → Uninstaller

### Desktop (Optional)
- `nvCOMP` → Launch GUI

## Prerequisites

### Required
- Windows 10 (build 17763) or later
- 64-bit (x86-64) architecture
- Visual C++ Redistributable 2019-2022 or later
- 4 GB RAM minimum, 8 GB recommended

### Optional (for GPU acceleration)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or later

## Troubleshooting

### Build Issues

**"WiX Toolset not found"**
```bash
# Install from: https://wixtoolset.org/releases/
# Or set WIX environment variable
set WIX=C:\Program Files (x86)\WiX Toolset v3.14
```

**"Files not found in source directory"**
```bash
# Build the application first
cd build_gui
cmake --build . --config Release
```

### Installation Issues

**"This installation is forbidden by system policy"**
- Run as Administrator
- Check Group Policy settings

**"Another version of this product is already installed"**
- Uninstall previous version first
- Or use upgrade: installer will auto-remove old version

**Custom actions fail**
```bash
# Test commands manually
nvcomp-gui.exe --register-context-menu
nvcomp-gui.exe --register-file-associations
```

### Debugging

Enable MSI logging:
```bash
msiexec /i nvCOMP-4.0.0-x64.msi /l*v install.log
```

Check log for errors:
```bash
findstr /C:"error" /C:"failed" /C:"returned 3" install.log
```

## Code Signing

```bash
# Set up environment
set SIGNTOOL_PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64
set CERT_THUMBPRINT=YOUR_CERTIFICATE_THUMBPRINT

# Sign (automatic in build_installer.bat if variables set)
signtool.exe sign /sha1 %CERT_THUMBPRINT% /t http://timestamp.digicert.com /fd SHA256 nvCOMP-4.0.0-x64.msi

# Verify
signtool.exe verify /pa nvCOMP-4.0.0-x64.msi
```

## Upgrade Behavior

1. User runs new version's MSI
2. Old version detected via UpgradeCode
3. Old version uninstalled automatically
4. New version installed
5. User settings preserved (`%APPDATA%\nvCOMP`)

## File Locations

### Installation (default)
```
C:\Program Files\nvCOMP\nvCOMP\
├── nvcomp-gui.exe
├── nvcomp-cli.exe
├── nvcomp_core.dll
├── nvcomp64_5.dll
├── nvcomp_cpu64_5.dll
├── Qt6Core.dll
├── Qt6Gui.dll
├── Qt6Widgets.dll
├── platforms\
│   └── qwindows.dll
├── styles\
│   └── qwindowsvistastyle.dll
└── imageformats\
    ├── qico.dll
    ├── qjpeg.dll
    └── qpng.dll
```

### User Settings
```
%APPDATA%\nvCOMP\
└── (preserved during upgrades)
```

### Start Menu
```
%ProgramData%\Microsoft\Windows\Start Menu\Programs\nvCOMP\
├── nvCOMP.lnk
└── Uninstall nvCOMP.lnk
```

## Testing Checklist

Before release:
- [ ] Build succeeds without warnings
- [ ] Install on clean Windows 10 VM
- [ ] Install on clean Windows 11 VM
- [ ] All features work correctly
- [ ] Context menu appears and works
- [ ] File associations work
- [ ] Start Menu shortcuts work
- [ ] Desktop shortcut works (if selected)
- [ ] GUI launches correctly
- [ ] CLI works correctly (if installed)
- [ ] Upgrade from previous version works
- [ ] Uninstall removes all files
- [ ] No registry keys left after uninstall
- [ ] No files left after uninstall

## Resources

- WiX Toolset: https://wixtoolset.org/
- Documentation: https://wixtoolset.org/documentation/
- Tutorial: https://www.firegiant.com/wix/tutorial/
- Windows Installer: https://docs.microsoft.com/en-us/windows/win32/msi/

---

**Quick Reference v1.0** - Task 4.3 Complete ✅

