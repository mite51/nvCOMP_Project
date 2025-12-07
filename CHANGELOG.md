# Changelog

All notable changes to the nvCOMP CLI project will be documented in this file.

## [2.0.0] - 2025-12-06

### Breaking Changes
- **Complete Architecture Rewrite**: Implemented dual-API approach with different handling for cross-compatible vs GPU-only algorithms
- **Cascaded Algorithm Removed**: Removed due to incompatibility with text data (caused failures in reference examples)
- **Command Interface Changed**: Updated to support `--cpu` flag for forcing CPU mode

### Added
- **CPU Fallback Support**
  - Automatic CPU fallback when CUDA is not available
  - CPU compression/decompression for LZ4, Snappy, and Zstd using native libraries
  - `--cpu` flag to force CPU mode even when GPU is available

- **Cross-Compatibility (GPU ↔ CPU)**
  - **Batched API Implementation** for LZ4, Snappy, Zstd:
    - Uses C header files (`nvcomp/lz4.h`, not `.hpp`)
    - Functions: `nvcompBatchedLZ4CompressAsync`, etc.
    - Produces **raw compressed data** compatible with CPU libraries
    - Enables GPU compress → CPU decompress and vice versa
  - Successfully tested cross-compatibility for all three algorithms

- **GPU-Only Manager API** for GDeflate, ANS, Bitcomp:
  - Uses C++ headers (`nvcomp.hpp`, `nvcomp/gdeflate.hpp`)
  - Classes: `GdeflateManager`, `ANSManager`, `BitcompManager`
  - Produces nvCOMP container format with metadata
  - Automatic format detection on decompression via `create_manager()`

- **Automatic Dependency Management**
  - CMake FetchContent integration for LZ4 (1.9.4)
  - CMake FetchContent integration for Snappy (1.2.1)
  - CMake FetchContent integration for Zstd (1.5.5)
  - Automatic CMake version patching for old dependencies
  - Cross-platform nvCOMP SDK download (Windows/Linux detection)

- **Comprehensive Testing**
  - Test suite with 15 comprehensive tests (reduced from initial plan, optimized):
    - GPU Batched ↔ GPU Batched (3 tests): LZ4, Snappy, Zstd
    - GPU Manager ↔ GPU Manager (3 tests): GDeflate, ANS, Bitcomp
    - CPU ↔ CPU (3 tests): LZ4, Snappy, Zstd
    - **GPU → CPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd
    - **CPU → GPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd
  - Test scripts (`test.bat`, `test.sh`) now located in `unit_test/` folder
  - Test output isolated to `unit_test/output/` directory
  - All tests passing on Windows with CUDA 13.0

- **Enhanced Documentation**
  - Comprehensive README.md with architecture explanation
  - Build instructions for Windows and Linux
  - Usage examples including cross-compatibility scenarios
  - Performance metrics and comparison table
  - Troubleshooting guide
  - API differences clearly documented

### Fixed
- **API Compatibility Issues** (nvCOMP 5.1.0):
  - Corrected function names: Added "Async" suffix (`nvcompBatchedLZ4CompressGetTempSizeAsync`)
  - Fixed option constants: Changed to `nvcompBatchedLZ4CompressDefaultOpts` (with "Compress")
  - Fixed Manager constructors: Added both compress and decompress options
  - Added missing header: `nvcomp/nvcompManagerFactory.hpp`
  - Fixed function signatures: Added `nullptr` status pointer parameter before stream
  - Added total input size parameter to `GetTempSizeAsync` functions

- **Build System**
  - Fixed CMake minimum version conflicts by patching fetched dependencies
  - Resolved CUDA architecture targeting (75, 80, 86, 89, 90)
  - Fixed linker warnings (LIBCMT conflicts - non-critical, common in CUDA projects)

### Changed
- **Chunk Management**: Implemented 64KB chunking for batched API (required for cross-compatibility)
- **File Format**: Cross-compatible algorithms produce raw format (concatenated chunks), GPU-only algorithms use nvCOMP container
- **Error Handling**: Enhanced error messages with line numbers and algorithm context
- **Performance Reporting**: Improved throughput calculation and formatting

### Technical Details
- **Two Different Implementations**:
  - **Implementation A (Batched API)**: For LZ4, Snappy, Zstd - produces standard format
  - **Implementation B (Manager API)**: For GDeflate, ANS, Bitcomp - produces nvCOMP format
- **Why Two Implementations?**: Batched API enables CPU interoperability at the cost of complexity; Manager API is simpler but GPU-only
- **Chunking**: Data split into 64KB chunks for batched processing, enabling better GPU utilization

### Known Limitations
- GPU batched decompression not implemented (requires chunk metadata storage)
- Files larger than GPU memory not supported (no streaming implementation yet)
- Default compression levels used (no level configuration exposed)
- Single GPU only (device 0)

### Dependencies
- nvCOMP 5.1.0.21 (CUDA 13)
- LZ4 1.9.4 (auto-fetched)
- Snappy 1.2.1 (auto-fetched)
- Zstd 1.5.5 (auto-fetched)
- CUDA Toolkit 11.0+ (user-provided)
- CMake 3.18+ (user-provided)

### Testing Environment
- Windows 10/11 with Visual Studio 2022
- CUDA 13.0
- NVIDIA GPU (tested with various architectures)
- All 15 tests passing

---

## [1.0.0] - 2025-12-06

### Added
- **Command Line Interface (`nvcomp_cli`)**
  - Implemented a C++ CLI application (`main.cu`) using the nvCOMP High-Level C++ API.
  - **Compression**:
    - Support for multiple GPU-accelerated algorithms: LZ4, GDeflate, Snappy, ANS, Cascaded, and Bitcomp.
    - Default algorithm set to LZ4.
    - Automatic resource management (GPU memory allocation, stream synchronization).
  - **Decompression**:
    - Automatic format detection using `nvcomp::create_manager`.
    - Support for decompressing any valid nvCOMP high-level stream.
  - **Performance Metrics**: Reports compressed size, compression ratio, and execution time (including throughput in GB/s).

- **Build System**
  - Added `CMakeLists.txt` for cross-platform build configuration.
  - Configured to link against local nvCOMP 5.1 library (`nvcomp-windows-x86_64-5.1.0.21_cuda13`).
  - Added post-build steps to automatically copy required DLLs (`nvcomp64_5.dll`, `nvcomp_cpu64_5.dll`) to the output directory.

- **Documentation**
  - Added `README.md` with build instructions, prerequisites, and usage examples.

### Fixed
- Resolved compilation errors related to missing default options for GDeflate, Snappy, ANS, Cascaded, and Bitcomp managers by updating constructor calls to match nvCOMP 5.1 headers.
- Fixed a runtime crash caused by improper destruction order of `nvcompManager` and `cudaStream_t` (ensured manager is destroyed before the stream).

### Notes
- The current implementation loads entire files into memory; files larger than available GPU memory are not yet supported (requires chunked streaming implementation).
- GPU-only implementation; requires NVIDIA GPU with Compute Capability 6.0+ and CUDA drivers.

