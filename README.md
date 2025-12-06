# nvCOMP CLI Application

A command-line interface for NVIDIA's nvCOMP library, enabling high-performance GPU compression and decompression.
*Currenly building for Windows 11 CUDA 13.0

## Prerequisites

*   **NVIDIA GPU** (Compute Capability 6.0+)
*   **CUDA Toolkit** (tested with 13.0)
*   **CMake** (3.18 or newer)
*   **Visual Studio** (C++ compiler)
*   **nvCOMP Library**: make will download 

## Building

1.  Open a terminal (PowerShell or Command Prompt) in the project root.
2.  Configure the project using CMake:
    ```powershell
    cmake -B build -S .
    ```
3.  Build the Release configuration:
    ```powershell
    cmake --build build --config Release
    ```

The executable will be created at: `build\Release\nvcomp_cli.exe`.
The required DLLs (`nvcomp64_5.dll`, `nvcomp_cpu64_5.dll`) are automatically copied to the output directory.

## Usage

### Compression

```powershell
nvcomp_cli.exe -c <input_file> <output_file> [algorithm]
```

*   `input_file`: Path to the file to compress.
*   `output_file`: Path where the compressed data will be written.
*   `algorithm`: (Optional) The compression algorithm to use. Default is `lz4`.
    *   Available algorithms: `lz4`, `gdeflate`, `snappy`, `ans`, `cascaded`, `bitcomp`

### Decompression

```powershell
nvcomp_cli.exe -d <input_file> <output_file>
```

*   `input_file`: Path to the compressed file.
*   `output_file`: Path where the decompressed data will be written.
*   **Note**: The tool automatically detects the compression format from the input file header.

## Examples

**Compress a file using GDeflate:**
```powershell
.\build\Release\nvcomp_cli.exe -c large_data.bin compressed.bin gdeflate
```

**Decompress the file:**
```powershell
.\build\Release\nvcomp_cli.exe -d compressed.bin restored_data.bin
```

**Compress using default algorithm (LZ4):**
```powershell
.\build\Release\nvcomp_cli.exe -c data.txt data.lz4
```

## Project Structure

*   `main.cu`: Main source code implementing the CLI using nvCOMP High-Level C++ API.
*   `CMakeLists.txt`: Build configuration file.
*   `nvcomp-windows-x86_64-5.1.0.21_cuda13/`: nvCOMP library binaries and headers.

