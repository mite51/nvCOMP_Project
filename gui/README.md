# nvCOMP GUI Application

Qt-based graphical interface for NVIDIA nvCOMP compression tool.

## Overview

This directory contains the Qt 6 GUI application for nvCOMP, providing a user-friendly interface for GPU-accelerated file compression and decompression.

## Structure

```
gui/
├── CMakeLists.txt          # Qt build configuration
├── src/                    # Source files
│   ├── main.cpp           # Application entry point
│   ├── mainwindow.cpp     # Main window implementation
│   └── mainwindow.h       # Main window header
├── ui/                     # Qt Designer UI files
│   └── mainwindow.ui      # Main window layout
├── resources/              # Application resources
│   ├── nvcomp.qrc         # Qt resource file
│   ├── generate_icons.py  # Icon generation script
│   └── icons/             # Application icons
└── tests/                  # Unit tests
    └── test_mainwindow.cpp # MainWindow tests
```

## Building

### Prerequisites

- Qt 6.2 or later (with Widgets module)
- CMake 3.18+
- C++17 compiler
- CUDA Toolkit (for core library)

### Build Steps

#### Windows

```powershell
# Configure with GUI enabled
cmake -B build -DBUILD_GUI=ON

# Build
cmake --build build --config Release

# The executable will be in: build/gui/Release/nvcomp_gui.exe
```

#### Linux

```bash
# Configure with GUI enabled
cmake -B build -DBUILD_GUI=ON

# Build
cmake --build build

# The executable will be in: build/gui/nvcomp_gui
```

### Qt Dependency Deployment

#### Windows

After building, deploy Qt dependencies:

```powershell
cd build\gui\Release
windeployqt nvcomp_gui.exe
```

#### Linux

Qt libraries should be found via RPATH. If issues occur, ensure Qt libraries are in your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/qt6/lib:$LD_LIBRARY_PATH
```

## Running Tests

The GUI includes a comprehensive test suite using Qt Test framework.

```bash
# Run all tests
ctest --test-dir build

# Run GUI tests specifically
ctest --test-dir build -R test_mainwindow

# Or run the test executable directly
./build/gui/test_mainwindow         # Linux
.\build\gui\Release\test_mainwindow.exe  # Windows
```

## Development

### Adding New Features

When adding GUI features:

1. **Update UI file**: Edit `ui/mainwindow.ui` in Qt Designer
2. **Implement logic**: Add code to `src/mainwindow.cpp`
3. **Add tests**: Create tests in `tests/` directory
4. **Update CMakeLists.txt**: Add new source files if needed

### Qt Designer

To edit UI files visually:

```bash
designer ui/mainwindow.ui
```

### Icons

Placeholder icons are generated automatically. For production:

1. Replace PNG files in `resources/icons/` with professional designs
2. Icon sizes: 16, 32, 48, 64, 128, 256 pixels
3. Update `resources/nvcomp.qrc` if adding new icons

## Current Status

**Task 2.1 Complete** ✅

- Qt project structure established
- Basic main window with menu and status bar
- Application icons and resources
- Comprehensive unit tests

**Next Steps (Future Tasks):**

- Task 2.2: Implement drag-and-drop file handling
- Task 2.3: Add compression/decompression workers
- Task 2.4: Implement progress display
- Task 2.5: Add GPU monitoring
- Task 2.6: Create settings dialog

## Architecture

The GUI is a thin layer over the `nvcomp_core` shared library:

```
GUI (Qt) → nvcomp_core (C API) → CUDA/CPU compression
```

This separation ensures:
- Clean separation of concerns
- Easy testability
- CLI and GUI share the same compression logic
- No code duplication

## License

Part of the nvCOMP project. See main repository LICENSE.

