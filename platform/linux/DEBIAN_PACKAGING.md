# Debian/Ubuntu Package Guide for nvCOMP

This guide explains how to build, test, and distribute .deb packages for the nvCOMP compression application.

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Build Requirements](#build-requirements)
- [Building Packages](#building-packages)
- [Testing Packages](#testing-packages)
- [Installation](#installation)
- [Distribution](#distribution)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

The nvCOMP project provides three Debian packages:

1. **nvcomp-gui** - Full GUI application with desktop integration
2. **nvcomp-cli** - Command-line interface only (lighter, no Qt dependencies)
3. **nvcomp-dev** - Development headers and libraries

All packages are built for **amd64** architecture and support:
- Ubuntu 20.04 LTS (Focal)
- Ubuntu 22.04 LTS (Jammy)
- Ubuntu 24.04 LTS (Noble)
- Debian 11 (Bullseye)
- Debian 12 (Bookworm)

---

## Package Structure

### Debian Package Files

```
platform/linux/debian/
├── control              # Package metadata and dependencies
├── rules                # Build instructions (Makefile)
├── changelog            # Version history
├── copyright            # License information
├── compat               # Debhelper compatibility level (13)
├── source/
│   └── format           # Source package format (3.0 quilt)
│
├── nvcomp-gui.install   # File installation list for GUI package
├── nvcomp-cli.install   # File installation list for CLI package
├── nvcomp-dev.install   # File installation list for dev package
│
├── nvcomp-gui.postinst  # Post-installation script (GUI)
├── nvcomp-gui.prerm     # Pre-removal script (GUI)
├── nvcomp-gui.postrm    # Post-removal script (GUI)
│
├── nvcomp-gui.1         # Man page for GUI
├── nvcomp-cli.1         # Man page for CLI
├── nvcomp-gui.manpages  # Man page installation list (GUI)
├── nvcomp-cli.manpages  # Man page installation list (CLI)
│
├── nvcomp-gui.docs      # Documentation files (GUI)
├── nvcomp-cli.docs      # Documentation files (CLI)
│
├── nvcomp.desktop       # Desktop entry file
└── nvcomp-mime.xml      # MIME type definitions
```

### Installed File Layout

After installation, files are placed as follows:

```
/usr/
├── bin/
│   ├── nvcomp-gui              # GUI executable
│   └── nvcomp-cli              # CLI executable
│
├── lib/x86_64-linux-gnu/
│   └── libnvcomp_core.so       # Core compression library
│
├── share/
│   ├── applications/
│   │   └── nvcomp.desktop      # Desktop application entry
│   │
│   ├── mime/packages/
│   │   └── nvcomp-mime.xml     # MIME type definitions
│   │
│   ├── icons/hicolor/
│   │   ├── 16x16/apps/nvcomp.png
│   │   ├── 32x32/apps/nvcomp.png
│   │   ├── 48x48/apps/nvcomp.png
│   │   ├── 64x64/apps/nvcomp.png
│   │   ├── 128x128/apps/nvcomp.png
│   │   └── 256x256/apps/nvcomp.png
│   │
│   ├── nvcomp/
│   │   ├── nautilus/
│   │   │   └── nautilus_extension.py
│   │   ├── nemo/
│   │   │   └── nemo_script.sh
│   │   └── install_file_manager_integration.sh
│   │
│   ├── doc/nvcomp-gui/
│   │   ├── README.md
│   │   ├── CHANGELOG.md
│   │   ├── copyright
│   │   └── linux-integration.md
│   │
│   └── man/man1/
│       ├── nvcomp-gui.1.gz
│       └── nvcomp-cli.1.gz
```

---

## Build Requirements

### Required Tools

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    debhelper \
    devscripts \
    build-essential \
    cmake \
    g++-12 \
    qt6-base-dev \
    qt6-base-dev-tools \
    libqt6core6 \
    libqt6widgets6 \
    libqt6gui6 \
    lintian \
    fakeroot
```

### Optional (Recommended) - CUDA Support

For GPU acceleration:

```bash
sudo apt-get install -y nvidia-cuda-toolkit
```

**Note:** CUDA is optional. Without it, packages will build with CPU-only support.

### Platform-Specific Notes

#### Ubuntu 24.04

Ubuntu 24.04 uses GCC 13 by default, which has compatibility issues with CUDA 12.x. Install GCC-12:

```bash
sudo apt-get install -y gcc-12 g++-12
```

The build system automatically detects and uses GCC-12 when available.

#### Ubuntu 20.04

Qt6 may not be available in default repositories. Add PPA or build Qt6 manually:

```bash
sudo add-apt-repository ppa:okirby/qt6-backports
sudo apt-get update
sudo apt-get install qt6-base-dev
```

---

## Building Packages

### Quick Build

Use the provided build script:

```bash
cd platform/linux
./build_deb.sh --install-deps --no-sign
```

### Build Options

```bash
./build_deb.sh [options]

Options:
  --install-deps   Install build dependencies before building
  --clean          Clean build directory before building
  --no-sign        Skip GPG signing (for testing)
  --help           Show help message
```

### Manual Build (Advanced)

If you prefer manual control:

```bash
# Copy debian directory to project root
cp -r platform/linux/debian/ debian/

# Build packages
debuild -us -uc -b
```

### Build Output

Successful builds create the following files in the parent directory:

```
../
├── nvcomp-gui_1.0.0-1_amd64.deb        # GUI package
├── nvcomp-cli_1.0.0-1_amd64.deb        # CLI package
├── nvcomp-dev_1.0.0-1_amd64.deb        # Dev package
├── nvcomp-gui_1.0.0-1_amd64.buildinfo  # Build metadata
└── nvcomp-gui_1.0.0-1_amd64.changes    # Changes file
```

---

## Testing Packages

### Automated Testing

Run the test script:

```bash
cd platform/linux
./test_package.sh ../nvcomp-gui_1.0.0-1_amd64.deb
```

The test script validates:
- Package integrity
- File contents
- Metadata correctness
- Lintian compliance
- Desktop file validity
- MIME type definitions
- File permissions
- Maintainer scripts

### Manual Testing

#### 1. Install Package

```bash
sudo dpkg -i ../nvcomp-gui_1.0.0-1_amd64.deb
sudo apt-get install -f  # Install missing dependencies
```

#### 2. Verify Installation

```bash
# Check executables
which nvcomp-gui
which nvcomp-cli

# Check library
ldconfig -p | grep nvcomp

# Check desktop integration
ls -la /usr/share/applications/nvcomp.desktop
ls -la /usr/share/mime/packages/nvcomp-mime.xml
```

#### 3. Test Functionality

```bash
# Test CLI
nvcomp-cli --version
nvcomp-cli --help

# Test GUI
nvcomp-gui --version
nvcomp-gui &
```

#### 4. Test Desktop Integration

- Check Applications menu for "nvCOMP"
- Right-click on a file, look for nvCOMP in "Open With"
- Double-click a .lz4/.zstd/.snappy file

#### 5. Test File Manager Integration

```bash
# Install context menus
/usr/share/nvcomp/install_file_manager_integration.sh

# Restart Nautilus/Nemo
nautilus -q && nautilus &
```

Right-click on files to verify context menu entries.

#### 6. Uninstall Test

```bash
sudo apt-get remove nvcomp-gui
# Verify files are removed
which nvcomp-gui  # Should return nothing
```

---

## Installation

### End-User Installation

#### From Local .deb File

```bash
sudo dpkg -i nvcomp-gui_1.0.0-1_amd64.deb
sudo apt-get install -f
```

#### From PPA (Future)

```bash
sudo add-apt-repository ppa:nvcomp/stable
sudo apt-get update
sudo apt-get install nvcomp-gui
```

### Installation Options

#### Full GUI (Recommended)

```bash
sudo apt-get install nvcomp-gui
```

Includes: GUI, CLI, libraries, desktop integration

#### CLI Only

```bash
sudo apt-get install nvcomp-cli
```

Includes: CLI, libraries (no Qt dependencies)

#### Development

```bash
sudo apt-get install nvcomp-dev
```

Includes: Headers, libraries for development

---

## Distribution

### Local Repository

Create a simple local repository:

```bash
# Create repository directory
mkdir -p /var/www/html/debian/pool/main/n/nvcomp

# Copy packages
cp *.deb /var/www/html/debian/pool/main/n/nvcomp/

# Generate Packages file
cd /var/www/html/debian
dpkg-scanpackages pool/main /dev/null | gzip -9c > dists/stable/main/binary-amd64/Packages.gz

# Create Release file
apt-ftparchive release dists/stable > dists/stable/Release
```

### PPA (Ubuntu Personal Package Archive)

#### Prerequisites

1. Create Launchpad account: https://launchpad.net
2. Set up GPG key
3. Install packaging tools: `sudo apt-get install dput`

#### Upload to PPA

```bash
# Sign the package
debuild -S -sa

# Upload to Launchpad
dput ppa:your-username/nvcomp ../nvcomp-gui_1.0.0-1_source.changes
```

### GitHub Releases

Upload .deb files to GitHub releases:

```bash
gh release create v1.0.0 \
  ../nvcomp-gui_1.0.0-1_amd64.deb \
  ../nvcomp-cli_1.0.0-1_amd64.deb \
  ../nvcomp-dev_1.0.0-1_amd64.deb \
  --title "nvCOMP 1.0.0" \
  --notes "First stable release"
```

---

## Troubleshooting

### Build Errors

#### CMake Configuration Fails

**Error:** `CUDA not found`

**Solution:**
```bash
sudo apt-get install nvidia-cuda-toolkit
# Or build without CUDA
export CUDA_AVAILABLE=OFF
```

#### Qt6 Not Found

**Error:** `Qt6 not found`

**Solution (Ubuntu 22.04+):**
```bash
sudo apt-get install qt6-base-dev qt6-base-dev-tools
```

**Solution (Ubuntu 20.04):**
```bash
sudo add-apt-repository ppa:okirby/qt6-backports
sudo apt-get update
sudo apt-get install qt6-base-dev
```

#### GCC Version Incompatibility

**Error:** `CUDA requires GCC version <= 12`

**Solution:**
```bash
sudo apt-get install gcc-12 g++-12
export CC=gcc-12
export CXX=g++-12
```

The build system handles this automatically if GCC-12 is installed.

### Installation Errors

#### Dependency Issues

**Error:** `Depends: libqt6core6 but it is not installable`

**Solution:**
```bash
# Enable universe repository
sudo add-apt-repository universe
sudo apt-get update

# Install Qt6 manually
sudo apt-get install libqt6core6 libqt6widgets6 libqt6gui6
```

#### Conflicting Packages

**Error:** `conflicts with existing package`

**Solution:**
```bash
# Remove old version first
sudo apt-get remove nvcomp-gui
sudo apt-get autoremove

# Then install new version
sudo dpkg -i nvcomp-gui_1.0.0-1_amd64.deb
```

### Runtime Errors

#### GPU Not Detected

**Issue:** Application runs in CPU mode despite having NVIDIA GPU

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA runtime
sudo apt-get install nvidia-cuda-toolkit

# Verify LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Missing Desktop Integration

**Issue:** Application not showing in Applications menu

**Solution:**
```bash
# Update desktop database
update-desktop-database ~/.local/share/applications
update-desktop-database /usr/share/applications

# Clear cache
rm -rf ~/.cache/menus
```

---

## Advanced Topics

### Multi-Version Support

To support multiple Ubuntu versions, build separate packages:

```bash
# Build for Ubuntu 20.04
debuild -us -uc -b
mv ../nvcomp-gui_1.0.0-1_amd64.deb ../nvcomp-gui_1.0.0-1ubuntu20.04_amd64.deb

# Build for Ubuntu 22.04
debuild -us -uc -b
mv ../nvcomp-gui_1.0.0-1_amd64.deb ../nvcomp-gui_1.0.0-1ubuntu22.04_amd64.deb
```

### Modifying Package Metadata

Edit `debian/control` to change:
- Package name
- Version
- Dependencies
- Description
- Maintainer

Edit `debian/changelog` to add version history:

```bash
dch -v 1.0.1-1 "New upstream release"
```

### Creating Debug Packages

Enable debug symbol packages:

```bash
# Build with debug symbols
DEB_BUILD_OPTIONS="nostrip noopt" debuild -us -uc -b

# Automatic debug package creation
# debian/rules already includes dh_strip --automatic-dbgsym
```

### GPG Signing

To sign packages for distribution:

```bash
# Generate GPG key (if you don't have one)
gpg --gen-key

# Build and sign
debuild -k<YOUR_KEY_ID>

# Verify signature
dpkg-sig --verify ../nvcomp-gui_1.0.0-1_amd64.deb
```

### Lintian Override

If lintian reports false positives, create overrides:

```bash
# Create override file
cat > debian/nvcomp-gui.lintian-overrides << EOF
# CUDA libraries are dynamically loaded
nvcomp-gui: package-must-activate-ldconfig-trigger
EOF
```

---

## Package Maintenance Checklist

### Before Release

- [ ] Update `debian/changelog` with version and changes
- [ ] Verify `debian/control` dependencies are correct
- [ ] Test build on all supported Ubuntu/Debian versions
- [ ] Run lintian checks and fix errors
- [ ] Test installation and uninstallation
- [ ] Verify desktop integration works
- [ ] Test with and without CUDA
- [ ] Update documentation

### After Release

- [ ] Upload to GitHub releases
- [ ] Upload to PPA (if applicable)
- [ ] Update website/download page
- [ ] Announce release
- [ ] Monitor bug reports

---

## Resources

- **Debian Policy Manual:** https://www.debian.org/doc/debian-policy/
- **Debian New Maintainers' Guide:** https://www.debian.org/doc/manuals/maint-guide/
- **Ubuntu Packaging Guide:** https://packaging.ubuntu.com/html/
- **Lintian Tags:** https://lintian.debian.org/tags.html
- **Debhelper Manual:** https://manpages.debian.org/testing/debhelper/debhelper.7.en.html

---

## Support

For packaging questions or issues:
- GitHub Issues: https://github.com/nvcomp/nvcomp-gui/issues
- Email: support@nvcomp.example.com
- IRC: #nvcomp on irc.freenode.net

---

**Last Updated:** December 19, 2025  
**Package Version:** 1.0.0-1  
**Maintainer:** nvCOMP Project

