# Changelog

All notable changes to the nvCOMP CLI project will be documented in this file.

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

