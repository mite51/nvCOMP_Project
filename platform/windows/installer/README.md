# Windows Installer (WiX) - Task 4.3

This directory contains the WiX installer project for nvCOMP.

## Overview

The WiX installer provides a professional MSI package for Windows installation with:
- Automatic component installation (GUI, CLI, dependencies)
- Optional features (context menu, file associations, shortcuts)
- Prerequisite checking (Visual C++ Redistributable, CUDA)
- Upgrade support (automatically removes old versions)
- Clean uninstallation

## Quick Start

### Prerequisites

1. **WiX Toolset 3.11 or later**
   - Download: https://wixtoolset.org/releases/
   - Install to default location or set `WIX` environment variable

2. **Built nvCOMP application**
   ```bash
   cd build_gui
   cmake --build . --config Release
   ```

3. **Qt dependencies deployed**
   - This should happen automatically via `windeployqt`
   - Verify `build_gui/bin/Release` contains Qt DLLs

### Building the Installer

#### Option A: Using the Build Script (Recommended)

```bash
cd platform/windows/installer
build_installer.bat Release
```

The installer will be created in `platform/windows/installer/output/nvCOMP-4.0.0-x64.msi`

#### Option B: Using CMake

```bash
cd build_gui
cmake --build . --target installer --config Release
```

The installer will be created in `build_gui/installer/nvCOMP-4.0.0-x64.msi`

#### Option C: Manual Build

```bash
cd platform/windows/installer

# Compile WiX sources
candle.exe -arch x64 -ext WixUIExtension installer.wxs components.wxs ui.wxs

# Link installer package
light.exe -ext WixUIExtension -cultures:en-us -out nvCOMP-4.0.0-x64.msi installer.wixobj components.wixobj ui.wixobj -sval
```

## File Structure

```
platform/windows/installer/
├── installer.wxs          # Main installer definition
├── components.wxs         # File components, registry, shortcuts
├── ui.wxs                 # Custom UI dialogs
├── product.wxi            # Shared properties and version info
├── license.rtf            # License agreement
├── build_installer.bat    # Automated build script
├── banner.bmp             # Banner image (493x58 pixels)
├── dialog.bmp             # Dialog background (493x312 pixels)
├── output/                # Build output directory
└── README.md              # This file
```

## Installer Features

### Core Feature (Required)
- **nvCOMP GUI** - Main graphical application
- **Core DLLs** - nvcomp_core.dll, compression libraries
- **NVIDIA DLLs** - nvcomp64_5.dll, nvcomp_cpu64_5.dll
- **Qt6 Dependencies** - Qt6Core, Qt6Gui, Qt6Widgets, plugins

### Optional Features (User-Selectable)
- **Command Line Tools** - nvcomp-cli.exe for scripting
- **Context Menu Integration** - "Compress with nvCOMP" in Explorer
- **File Associations** - Associate .lz4, .zstd, .nvcomp files
- **Start Menu Shortcuts** - nvCOMP and Uninstall shortcuts
- **Desktop Shortcut** - Quick access from desktop (off by default)

## Installation Options

### Interactive Installation

Double-click the MSI file and follow the wizard:

1. **Welcome** - Introduction
2. **License Agreement** - Accept terms
3. **Prerequisites** - Check for required components
4. **Feature Selection** - Choose components to install
5. **Installation Directory** - Choose install location (default: `C:\Program Files\nvCOMP\nvCOMP`)
6. **Progress** - Installation progress
7. **Finish** - Option to launch nvCOMP

### Silent Installation

```bash
# Install with default options
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn /norestart

# Install to custom location
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn INSTALLDIR="D:\Tools\nvCOMP" /norestart

# Install without optional features
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn INSTALL_CLI=0 INSTALL_CONTEXT_MENU=0 /norestart

# Install with desktop shortcut
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn INSTALL_DESKTOP_SHORTCUT=1 /norestart

# Enable logging
msiexec /i nvCOMP-4.0.0-x64.msi /quiet /qn /l*v install.log /norestart
```

### Uninstallation

```bash
# Interactive
msiexec /x nvCOMP-4.0.0-x64.msi

# Silent
msiexec /x nvCOMP-4.0.0-x64.msi /quiet /qn /norestart

# Via Add/Remove Programs
# Control Panel → Programs → Programs and Features → nvCOMP
```

## Customization

### Updating Version Number

Edit `product.wxi`:

```xml
<?define ProductVersion = "4.0.0" ?>
```

Also update in `CMakeLists.txt`:

```cmake
set(PRODUCT_VERSION "4.0.0")
```

### Generating New GUIDs

**IMPORTANT**: Before building your first installer, generate unique GUIDs!

```powershell
# PowerShell - run this 10+ times to get unique GUIDs
[guid]::NewGuid()
```

Update these in `product.wxi`:
- `UpgradeCode` - **NEVER change this** (identifies your product across versions)
- Component GUIDs - Generate once, keep stable across versions
- Feature GUIDs - Generate once, keep stable

### Customizing Branding

1. **Banner Image** (`banner.bmp`):
   - Size: 493 x 58 pixels
   - Format: 24-bit BMP
   - Displayed at top of installer dialogs

2. **Dialog Background** (`dialog.bmp`):
   - Size: 493 x 312 pixels
   - Format: 24-bit BMP
   - Displayed on left side of dialogs

3. **Application Icon**:
   - Already uses `gui/resources/icons/nvcomp.ico`
   - Shown in Add/Remove Programs

### Modifying License Agreement

Edit `license.rtf` with any RTF-compatible editor (WordPad, Word, etc.)

## Prerequisites Handling

The installer checks for:

### Visual C++ Redistributable 2019-2022
- **Required**: Yes
- **Action**: Installation blocked if missing
- **Download**: https://aka.ms/vs/17/release/vc_redist.x64.exe

### CUDA Toolkit
- **Required**: No (optional for GPU acceleration)
- **Action**: Warning shown, installation continues
- **Download**: https://developer.nvidia.com/cuda-downloads

### Custom Prerequisite Handling

To add a prerequisite check, edit `installer.wxs`:

```xml
<Property Id="MY_PREREQUISITE">
  <RegistrySearch Id="MyPrereqCheck"
                  Root="HKLM"
                  Key="SOFTWARE\MyCompany\MyPrerequisite"
                  Name="Version"
                  Type="raw" />
</Property>

<Condition Message="MyPrerequisite is required!">
  <![CDATA[Installed OR MY_PREREQUISITE]]>
</Condition>
```

## Registry Integration

The installer creates registry entries for:

### Product Information
- **Location**: `HKEY_LOCAL_MACHINE\Software\nvCOMP\nvCOMP`
- **Values**:
  - `InstallPath` - Installation directory
  - `Version` - Product version
  - `GuiExePath` - Path to GUI executable
  - `CliExePath` - Path to CLI executable (if installed)

### Uninstall Information
- **Location**: `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Uninstall\nvCOMP`
- **Auto-managed by Windows Installer**

### Context Menu
- Registered via custom action: `nvcomp-gui.exe --register-context-menu`
- Unregistered on uninstall: `nvcomp-gui.exe --unregister-context-menu`

### File Associations
- Registered via custom action: `nvcomp-gui.exe --register-file-associations`
- Unregistered on uninstall: `nvcomp-gui.exe --unregister-file-associations`

## Upgrade Behavior

The installer supports **major upgrades**:

1. User runs new version's MSI
2. Installer detects previous version via `UpgradeCode`
3. Old version is uninstalled automatically
4. New version is installed
5. User settings are preserved (in `%APPDATA%\nvCOMP`)

### Version Scheme
- **Major.Minor.Patch** (e.g., 4.0.0)
- **Change for each release** to trigger upgrades
- **Keep UpgradeCode constant** across all versions

## Code Signing

To sign the installer (requires code signing certificate):

### Setup

1. **Install SignTool** (comes with Windows SDK)
2. **Set environment variables**:
   ```bash
   set SIGNTOOL_PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64
   set CERT_THUMBPRINT=YOUR_CERTIFICATE_THUMBPRINT_HERE
   ```

3. **Run build script** - will automatically sign if variables are set

### Manual Signing

```bash
signtool.exe sign /sha1 YOUR_THUMBPRINT /t http://timestamp.digicert.com /fd SHA256 nvCOMP-4.0.0-x64.msi

# Verify signature
signtool.exe verify /pa nvCOMP-4.0.0-x64.msi
```

## Testing Checklist

Before releasing the installer:

- [ ] Build succeeds without warnings
- [ ] MSI opens and shows correct version
- [ ] License agreement displays correctly
- [ ] Prerequisites are detected correctly
- [ ] Feature selection works
- [ ] Custom installation directory works
- [ ] Files are copied correctly
- [ ] Shortcuts are created (Start Menu, Desktop if selected)
- [ ] Context menu integration works (if selected)
- [ ] File associations work (if selected)
- [ ] Application launches from Start Menu
- [ ] Application launches from Desktop shortcut (if created)
- [ ] PATH environment variable updated (optional)
- [ ] GUI application runs correctly
- [ ] CLI application runs correctly (if installed)
- [ ] Upgrade from previous version works
- [ ] Uninstall removes all files
- [ ] Uninstall removes registry keys
- [ ] Uninstall removes shortcuts
- [ ] Uninstall unregisters context menu
- [ ] Uninstall unregisters file associations
- [ ] No orphaned files or registry keys after uninstall

### Test on Clean VM

Test on a clean Windows 10/11 virtual machine:
- No Visual Studio installed
- No CUDA Toolkit installed
- No Qt installed
- No previous nvCOMP version

## Troubleshooting

### "WiX Toolset not found"

**Solution**: Install WiX Toolset or set `WIX` environment variable:
```bash
set WIX=C:\Program Files (x86)\WiX Toolset v3.14
```

### "candle.exe error: Unresolved reference to symbol"

**Solution**: Missing component GUID or typo in component reference. Check:
- All `ComponentRef` IDs match `Component` IDs
- All GUIDs are properly formatted

### "light.exe error: The Windows Installer Service could not be accessed"

**Solution**: Run as Administrator or restart Windows Installer service:
```bash
net stop msiserver
net start msiserver
```

### "Files not found in source directory"

**Solution**: Build the application first:
```bash
cd build_gui
cmake --build . --config Release
```

### Custom action fails during installation

**Solution**: Check that command-line arguments are implemented:
```bash
# Test manually
nvcomp-gui.exe --register-context-menu
nvcomp-gui.exe --register-file-associations
nvcomp-gui.exe --unregister-context-menu
nvcomp-gui.exe --unregister-file-associations
```

## Advanced Topics

### Adding New Components

1. Add files to `components.wxs`:
```xml
<Component Id="MyNewComponent" Guid="*" Win64="yes">
  <File Id="mynewfile.dll" Source="$(var.SourceDir)\mynewfile.dll" />
</Component>
```

2. Reference in a feature:
```xml
<Feature Id="MyFeature">
  <ComponentRef Id="MyNewComponent" />
</Feature>
```

### Custom Actions

To run custom code during installation:

```xml
<!-- Define the custom action -->
<CustomAction Id="MyAction"
              BinaryKey="WixCA"
              DllEntry="CAQuietExec"
              Execute="deferred"
              Return="check"
              Impersonate="no" />

<CustomAction Id="SetMyActionCmd"
              Property="MyAction"
              Value="&quot;[INSTALLDIR]mytool.exe&quot; --do-something" />

<!-- Schedule in InstallExecuteSequence -->
<InstallExecuteSequence>
  <Custom Action="SetMyActionCmd" Before="MyAction">
    NOT Installed
  </Custom>
  <Custom Action="MyAction" After="InstallFiles">
    NOT Installed
  </Custom>
</InstallExecuteSequence>
```

### Conditional Installation

Install components based on conditions:

```xml
<Feature Id="GpuFeature" Title="GPU Support" Level="2">
  <Condition Level="1">CUDA_INSTALLED</Condition>
  <ComponentRef Id="GpuComponent" />
</Feature>
```

## Resources

- **WiX Tutorial**: https://www.firegiant.com/wix/tutorial/
- **WiX Documentation**: https://wixtoolset.org/documentation/
- **WiX Schema**: https://wixtoolset.org/documentation/manual/v3/xsd/
- **Windows Installer**: https://docs.microsoft.com/en-us/windows/win32/msi/

## Support

For issues with the installer:
1. Check the build log for errors
2. Enable MSI logging: `msiexec /i installer.msi /l*v install.log`
3. Review `install.log` for detailed error messages
4. Check registry entries with `regedit.exe`
5. Verify file permissions and administrator privileges

## License

This installer project is part of nvCOMP and uses the same license.
Third-party components (Qt, nvCOMP SDK) have their own licenses included.

---

**Status**: Task 4.3 - Complete ✅

Ready to build professional Windows installers!

