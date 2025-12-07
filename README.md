# nvCOMP CLI with CPU Fallback & Folder Support

A cross-platform command-line interface for GPU-accelerated compression using NVIDIA nvCOMP with automatic CPU fallback when GPU is unavailable.

## Features

- **Dual Implementation Architecture**:
  - **Batched API** for LZ4, Snappy, Zstd: Cross-compatible between GPU and CPU
  - **Manager API** for GDeflate, ANS, Bitcomp: GPU-only, high-performance
  
- **Folder Compression**: Compress entire directories with automatic file archiving
- **Archive Listing**: View contents of compressed archives without extracting
- **Automatic CPU Fallback**: Seamlessly falls back to CPU compression when CUDA is not available
- **Cross-Compatibility**: GPU-compressed files (LZ4/Snappy/Zstd) can be decompressed on CPU and vice versa
- **Cross-Platform Path Handling**: Automatically handles Windows/Linux filesystem differences
- **High Performance**: Leverages CUDA for GPU acceleration with typical 10-100x speedup
- **Cross-Platform**: Works on Windows and Linux

## Supported Algorithms

### Cross-Compatible (GPU ↔ CPU)
- **LZ4**: Fast compression with good ratios
- **Snappy**: Very fast compression, lower ratios
- **Zstd**: Best compression ratios, slower

These algorithms use the nvCOMP Batched API which produces raw compressed data compatible with standard CPU libraries.

### GPU-Only
- **GDeflate**: GPU-optimized DEFLATE implementation
- **ANS**: Asymmetric Numeral Systems compression
- **Bitcomp**: Lossless compression for numerical data

These algorithms use the nvCOMP Manager API which produces nvCOMP container format and only work GPU-to-GPU.

## Building

### Prerequisites

- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with Compute Capability 7.5+ (optional, will use CPU fallback if not available)
- C++17 compatible compiler

### Build on Windows

```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Build on Linux

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

The build system automatically:
- Downloads the appropriate nvCOMP SDK for your platform
- Fetches and builds LZ4, Snappy, and Zstd dependencies
- Patches old CMake version requirements in dependencies

## Usage

### Basic Syntax

```bash
# Compress file or folder (algorithm required)
nvcomp_cli -c <input> <output_file> <algorithm> [--cpu]

# Decompress to folder (algorithm optional - auto-detected!)
nvcomp_cli -d <input_file> <output_folder> [algorithm] [--cpu]

# List archive contents (algorithm optional - auto-detected!)
nvcomp_cli -l <archive_file> [algorithm] [--cpu]
```

**NEW**: Algorithm parameter is now **optional** for decompression (`-d`) and listing (`-l`) modes! The tool automatically detects the algorithm from the file header for GPU batched format files (LZ4, Snappy, Zstd).

### Single File Examples

```bash
# Compress single file with LZ4 (GPU)
nvcomp_cli -c input.txt output.lz4 lz4

# Decompress with auto-detection (no algorithm needed!)
nvcomp_cli -d output.lz4 restored/

# Decompress with explicit algorithm
nvcomp_cli -d output.lz4 restored/ lz4

# Force CPU mode
nvcomp_cli -c input.txt output.lz4 lz4 --cpu

# GPU-only algorithm
nvcomp_cli -c input.txt output.gdeflate gdeflate
```

### Folder Compression Examples

```bash
# Compress entire folder
nvcomp_cli -c mydata/ archive.zstd zstd

# Compress with Snappy
nvcomp_cli -c project/ backup.snappy snappy

# Decompress folder archive (auto-detects algorithm!)
nvcomp_cli -d archive.zstd restored/

# List archive contents (auto-detects algorithm!)
nvcomp_cli -l archive.zstd

# Or specify algorithm explicitly if desired
nvcomp_cli -d archive.zstd restored/ zstd
nvcomp_cli -l archive.zstd zstd
```

### Cross-Platform Compatibility

The tool automatically handles filesystem differences between Windows and Linux:

- **Path Separators**: Automatically converts Windows backslashes (`\`) to forward slashes (`/`) in archives
- **Cross-Platform Archives**: Archives created on Windows can be extracted on Linux and vice versa
- **Relative Paths**: All paths in archives are stored as relative paths for portability

```bash
# Create archive on Windows
nvcomp_cli -c C:\mydata\project\ backup.lz4 lz4

# Extract on Linux (works seamlessly!)
nvcomp_cli -d backup.lz4 /home/user/restored/ lz4
```

### Cross-Compatibility Examples

```bash
# Compress on GPU, decompress on GPU (recommended, fastest!)
nvcomp_cli -c input.txt output.lz4 lz4
nvcomp_cli -d output.lz4 restored/ lz4

# Compress on GPU, decompress on CPU (works, but slower)
nvcomp_cli -c input.txt output.lz4 lz4
nvcomp_cli -d output.lz4 restored/ lz4 --cpu

# Compress on CPU, decompress on GPU (works!)
nvcomp_cli -c mydata/ output.snappy snappy --cpu
nvcomp_cli -d output.snappy restored/ snappy

# Compress on CPU, decompress on CPU
nvcomp_cli -c mydata/ output.snappy snappy --cpu
nvcomp_cli -d output.snappy restored/ snappy --cpu
```

**Note**: GPU-compressed files (LZ4/Snappy/Zstd) use a custom batched format with metadata. They can be decompressed on GPU (fast) or CPU (slower). The tool automatically detects the format and uses the appropriate decompression method.

## Command-Line Options

- `-c <input> <output> <algorithm>`: Compression mode (input can be file or directory, **algorithm required**)
- `-d <input> <output> [algorithm]`: Decompression/extraction mode (output is target directory, **algorithm optional**)
- `-l <archive> [algorithm]`: List archive contents without extracting (**algorithm optional**)
- `--cpu`: Force CPU mode (only works with lz4, snappy, zstd)
- Algorithm names: `lz4`, `snappy`, `zstd`, `gdeflate`, `ans`, `bitcomp`

### Algorithm Auto-Detection

For GPU batched format files (LZ4/Snappy/Zstd), the algorithm parameter is **optional** for decompression and listing. The tool automatically detects the algorithm from the file header:

```bash
# Compression - algorithm REQUIRED
nvcomp_cli -c input.txt output.zstd zstd

# Decompression - algorithm OPTIONAL (auto-detected!)
nvcomp_cli -d output.zstd restored/

# Listing - algorithm OPTIONAL (auto-detected!)
nvcomp_cli -l output.zstd

# You can still specify algorithm explicitly if desired
nvcomp_cli -d output.zstd restored/ zstd
```

This works for all three cross-compatible algorithms (LZ4, Snappy, Zstd) in both GPU and CPU modes.

## Archive Format

The tool uses a simple, efficient archive format for storing multiple files:

- **Magic Number**: `NVAR` (NvCOMP ARchive) for format identification
- **Version**: Currently version 1
- **Structure**:
  - Archive Header (16 bytes): magic, version, file count
  - For each file: File Entry Header + path + data
  - File Entry: path length, file size
- **Path Storage**: All paths stored with forward slashes (`/`) for cross-platform compatibility
- **No Compression in Archive**: Archiving and compression are separate; the archive is created first, then compressed as a single blob

## Batched Compression Format

For GPU batched compression (LZ4/Snappy/Zstd), the tool uses a custom format with metadata:

- **Magic Number**: `NVBC` (NvCOMP Batched Compression) for format identification
- **Version**: Currently version 1
- **Structure**:
  - Batched Header (32 bytes): magic, version, uncompressed size, chunk count, chunk size, algorithm
  - Chunk sizes array: size of each compressed chunk (uint64_t per chunk)
  - Compressed data: concatenated compressed chunks
- **Purpose**: Enables GPU decompression by storing chunk boundaries
- **Cross-compatible**: CPU decompression also supported (but slower)

## Testing

### Single-File Compression Tests (15 tests)

Run the comprehensive test suite for single-file compression:

**Windows:**
```cmd
cd unit_test
test.bat
```

**Linux:**
```bash
cd unit_test
chmod +x test.sh
./test.sh
```

**Test Coverage:**
1. **GPU Batched ↔ GPU Batched** (3 tests): LZ4, Snappy, Zstd
2. **GPU Manager ↔ GPU Manager** (3 tests): GDeflate, ANS, Bitcomp
3. **CPU ↔ CPU** (3 tests): LZ4, Snappy, Zstd
4. **GPU → CPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd
5. **CPU → GPU Cross-compatibility** (3 tests): LZ4, Snappy, Zstd

### Folder Compression Tests (14 tests)

Run tests for folder compression, archive listing, and the original zstd decompression issue fix:

**Windows:**
```cmd
cd unit_test
test_folder.bat
```

**Linux:**
```bash
cd unit_test
chmod +x test_folder.sh
./test_folder.sh
```

**Test Coverage:**
1. **Folder Compression** (4 tests): GPU LZ4, GPU Zstd, CPU LZ4, CPU Zstd
2. **Archive Listing** (3 tests): GPU Zstd, GPU LZ4, CPU Zstd
3. **GPU Zstd Round-Trip** (1 test): Reproduces and verifies fix for original issue
4. **Algorithm Auto-Detection** (6 tests): List and decompress without algorithm parameter

The round-trip test specifically validates the fix for:
```
Error: Zstd CPU decompression failed
```

This test compresses a folder with GPU zstd, lists the archive, decompresses it, and verifies the output - all operations that were failing before the fix.

**Total Test Coverage: 29 tests** (15 single-file + 14 folder/archive tests)

## Architecture Details

### Compression Pipeline

1. **Archiving Phase**: Input (file or directory) → Archive format
   - Single files are packaged into a single-file archive
   - Directories are recursively traversed and all files added to archive
   - Paths are normalized to forward slashes for cross-platform compatibility

2. **Compression Phase**: Archive data → Compressed output
   - Archive is treated as a single data blob
   - Compressed using selected algorithm (GPU or CPU)

3. **Decompression Phase**: Compressed file → Archive → Extracted files
   - Decompress the data blob
   - Parse archive format
   - Extract files to target directory with proper path structure

### Two Different Implementations

#### Implementation A: Batched API (LZ4, Snappy, Zstd)

- Uses C header files: `#include <nvcomp/lz4.h>`
- Functions: `nvcompBatchedLZ4CompressAsync`, `nvcompBatchedLZ4DecompressAsync`
- Produces **raw compressed data** compatible with CPU libraries
- Enables GPU ↔ CPU interoperability
- Data is chunked (64KB chunks) for batched processing

**Key characteristic**: The output is standard LZ4/Snappy/Zstd format that any compliant decoder can read.

#### Implementation B: Manager API (GDeflate, ANS, Bitcomp)

- Uses C++ headers: `#include <nvcomp.hpp>`, `#include <nvcomp/gdeflate.hpp>`
- Classes: `GdeflateManager`, `ANSManager`, `BitcompManager`
- Produces **nvCOMP container format** with metadata
- GPU-to-GPU only
- Automatic format detection on decompression

**Key characteristic**: The output includes nvCOMP-specific metadata and can only be decompressed using nvCOMP.

### Why Two Implementations?

The nvCOMP library provides two APIs with different trade-offs:

1. **Batched API**: 
   - ✅ Cross-compatible with CPU libraries
   - ✅ Standard format
   - ❌ More complex (manual chunking)
   - ❌ Less metadata

2. **Manager API**: 
   - ✅ Easier to use
   - ✅ Better metadata handling
   - ✅ Automatic format detection
   - ❌ GPU-only
   - ❌ Proprietary container format

This project uses both to provide the best of both worlds.

## Performance

Typical performance on NVIDIA A100:

| Algorithm | Compression | Decompression | Ratio |
|-----------|-------------|---------------|-------|
| LZ4       | 20-40 GB/s  | 40-80 GB/s    | 2-3x  |
| Snappy    | 30-50 GB/s  | 50-90 GB/s    | 1.5-2x|
| Zstd      | 5-15 GB/s   | 10-30 GB/s    | 3-5x  |
| GDeflate  | 10-20 GB/s  | 20-40 GB/s    | 2-4x  |
| ANS       | 5-10 GB/s   | 10-20 GB/s    | 2-3x  |
| Bitcomp   | 15-25 GB/s  | 30-50 GB/s    | 2-10x*|

*Bitcomp ratio depends heavily on data type (best for numerical data)

CPU performance is typically 10-100x slower depending on CPU and data.

## Limitations

1. **GPU Memory**: Large files and archives require sufficient GPU memory. The entire archive is loaded into memory during compression/decompression. The tool will fail if GPU memory is exhausted.

2. **CUDA Version**: Requires CUDA 11.0+. Tested with CUDA 12.x and 13.x.

3. **Archive In-Memory**: The entire archive must fit in memory. For very large directories, consider splitting into multiple archives.

4. **File Permissions**: File permissions and attributes are not preserved in archives (only file paths and contents).

5. **Custom Format**: GPU batched compression uses a custom format with metadata. While CPU decompression is supported, it's slower than GPU decompression.

6. **Cascaded Algorithm**: Not included in this implementation (removed from nvCOMP reference examples due to text data incompatibility).

## Troubleshooting

### "CUDA not available, falling back to CPU"

This is normal behavior when:
- No NVIDIA GPU is present
- CUDA drivers are not installed
- GPU is already in use

The tool will automatically use CPU compression.

### "Algorithm 'X' is GPU-only and cannot run on CPU"

You're trying to use GDeflate, ANS, or Bitcomp with `--cpu` flag or without CUDA. These algorithms only support GPU mode.

### "Directory does not exist" or "Not a directory"

- Verify the input path is correct
- On Windows, use either forward slashes (`/`) or backslashes (`\`)
- Ensure you have read permissions for the directory

### "No files to archive"

The specified directory is empty or contains only subdirectories with no files.

### Compression/Decompression fails

- Check that input file/folder exists and is readable
- Ensure sufficient disk space for output
- For large directories, check available memory
- For GPU mode, check CUDA driver with `nvidia-smi`
- Try with `--cpu` flag to isolate GPU issues

### Path Issues on Windows

If you encounter path issues on Windows:
- Use forward slashes: `nvcomp_cli -c C:/mydata/ output.lz4 lz4`
- Or escape backslashes: `nvcomp_cli -c "C:\\mydata\\" output.lz4 lz4`
- Or use quotes: `nvcomp_cli -c "C:\mydata\" output.lz4 lz4`

## Dependencies

All dependencies are automatically fetched and built by CMake:

- **nvCOMP 5.1.0**: NVIDIA compression library
- **LZ4 1.9.4**: Fast compression library
- **Snappy 1.2.1**: Fast compression library by Google
- **Zstd 1.5.5**: Compression by Facebook

## License

This project uses:
- nvCOMP (NVIDIA License)
- LZ4 (BSD License)
- Snappy (BSD License)
- Zstd (BSD/GPLv2 dual license)

See individual library licenses for details.

## Contributing

Contributions welcome! Please ensure:
- All 15 tests pass
- Code follows existing style
- Documentation is updated
- Cross-compatibility is maintained for LZ4/Snappy/Zstd

## References

- [nvCOMP Documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [nvCOMP Developer Page](https://developer.nvidia.com/nvcomp)
- [LZ4](https://github.com/lz4/lz4)
- [Snappy](https://github.com/google/snappy)
- [Zstd](https://github.com/facebook/zstd)

## Changelog

### Version 2.2.0 (Current)
- **NEW**: Algorithm auto-detection for decompression and listing modes
  - No need to specify algorithm when decompressing or listing GPU batched files
  - Works for LZ4, Snappy, and Zstd in both GPU and CPU modes
  - Example: `nvcomp_cli -d archive.zstd output/` (no algorithm parameter!)
- **NEW**: 6 additional tests for algorithm auto-detection
- **IMPROVED**: Enhanced CLI with better usage documentation
- All tests passing (15 single-file + 14 folder/archive tests = 29 total)

### Version 2.1.1
- **FIXED**: GPU zstd decompression issue ("Error: Zstd CPU decompression failed")
- **FIXED**: Archive listing for GPU-compressed files
- **NEW**: Batched compression format with metadata for proper GPU decompression
- **NEW**: Automatic format detection (GPU batched vs CPU standard)
- **NEW**: Folder compression test suite (8 tests)
- **NEW**: Dedicated test for original zstd issue (validates fix)

### Version 2.1.0
- **NEW**: Folder/directory compression support
- **NEW**: Archive listing mode (`-l` flag)
- **NEW**: Cross-platform path handling (Windows ↔ Linux)
- **NEW**: Recursive directory traversal
- **NEW**: Custom archive format with metadata
- Improved CLI with better argument parsing
- Updated documentation with folder examples

### Version 2.0.0
- Complete architecture rewrite with dual-API approach
- Batched API support for LZ4, Snappy, Zstd (cross-compatible)
- Manager API support for GDeflate, ANS, Bitcomp (GPU-only)
- CPU fallback for cross-compatible algorithms
- Cross-platform build system
- Comprehensive test suite (15 tests)

### Version 1.0.0
- Initial release
- Basic GPU compression/decompression
- Manager API only

## Acknowledgments

Based on nvCOMP examples from NVIDIA. Special thanks to the nvCOMP team for providing excellent reference implementations.

Folder compression and cross-platform path handling implemented using C++17 filesystem API for maximum portability.
