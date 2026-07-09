#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <memory>

// Windows DLL export/import macros
#ifdef _WIN32
    #ifdef NVCOMP_CORE_EXPORTS
        #define NVCOMP_CORE_API __declspec(dllexport)
    #else
        #define NVCOMP_CORE_API __declspec(dllimport)
    #endif
#else
    #define NVCOMP_CORE_API
#endif

namespace nvcomp_core {

// ============================================================================
// Constants
// ============================================================================

constexpr size_t CHUNK_SIZE = 1 << 16; // 64KB
constexpr uint32_t ARCHIVE_MAGIC = 0x4E564152; // "NVAR"
constexpr uint32_t ARCHIVE_VERSION = 1;
constexpr uint32_t BATCHED_MAGIC = 0x4E564243; // "NVBC"
constexpr uint32_t BATCHED_VERSION = 1;
constexpr uint32_t VOLUME_MAGIC = 0x4E56564D; // "NVVM"
constexpr uint32_t VOLUME_VERSION = 1;
constexpr uint64_t DEFAULT_VOLUME_SIZE = 2684354560ULL; // 2.5GB

// GPU pipeline: number of sub-batch slots in flight (disk read / compress /
// readback overlap). Each slot holds one sub-batch of CHUNK_SIZE chunks.
constexpr size_t PIPELINE_DEPTH = 3;

// ============================================================================
// Data Structures
// ============================================================================

struct BlockProgressInfo {
    int totalBlocks;
    int completedBlocks;
    int currentBlock;
    size_t currentBlockSize;
    float overallProgress;
    float currentBlockProgress;
    double throughputMBps;
    std::string stage;
};

using ProgressCallback = std::function<void(const BlockProgressInfo&)>;

/**
 * Phase-by-phase timing for one compression or decompression operation.
 * Populated by the core and returned via the optional outStats parameter.
 *
 * Phases:
 *   readSec    - reading input file(s) from disk and assembling the in-memory archive
 *   prepareSec - allocating GPU buffers, host->device copies, scratch setup
 *   computeSec - actual compression/decompression kernels (cudaStreamSynchronize wall time)
 *   writeSec   - writing the output file(s) to disk
 *   totalSec   - sum of the above
 *
 * Throughput is computed against the uncompressed (input) size.
 * For decompression, inputBytes is the uncompressed output size and outputBytes is the
 * compressed input size, so 'ratio' is consistent (uncompressed/compressed) for both
 * directions.
 */
struct CompressionStats {
    double readSec = 0.0;
    double prepareSec = 0.0;
    double computeSec = 0.0;
    double writeSec = 0.0;
    double totalSec = 0.0;
    uint64_t inputBytes = 0;       // uncompressed size (the "real" data)
    uint64_t outputBytes = 0;      // compressed size (on disk)
    double throughputMBps = 0.0;   // inputBytes / totalSec, in MB/s
    double throughputGBps = 0.0;   // inputBytes / totalSec, in GB/s
    double ratio = 0.0;            // uncompressed / compressed
};

struct ArchiveHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t fileCount;
    uint32_t reserved;
};

struct FileEntry {
    uint32_t pathLength;
    uint64_t fileSize;
};

struct BatchedHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t uncompressedSize;
    uint32_t chunkCount;
    uint32_t chunkSize;
    uint32_t algorithm;
    uint32_t reserved;
};

struct VolumeManifest {
    uint32_t magic;
    uint32_t version;
    uint32_t volumeCount;
    uint32_t algorithm;
    uint64_t volumeSize;
    uint64_t totalUncompressedSize;
    uint64_t reserved;
};

struct VolumeMetadata {
    uint64_t volumeIndex;
    uint64_t compressedSize;
    uint64_t uncompressedOffset;
    uint64_t uncompressedSize;
};

enum AlgoType {
    ALGO_LZ4,
    ALGO_SNAPPY,
    ALGO_ZSTD,
    ALGO_GDEFLATE,
    ALGO_ANS,
    ALGO_BITCOMP,
    ALGO_UNKNOWN
};

// ============================================================================
// Algorithm Utilities
// ============================================================================

NVCOMP_CORE_API AlgoType parseAlgorithm(const std::string& algo);
NVCOMP_CORE_API std::string algoToString(AlgoType algo);
NVCOMP_CORE_API bool isCrossCompatible(AlgoType algo);
NVCOMP_CORE_API bool isCudaAvailable();

// ============================================================================
// File I/O Utilities
// ============================================================================

// fs::path overloads for proper Unicode handling on Windows
NVCOMP_CORE_API std::vector<uint8_t> readFile(const std::filesystem::path& filepath);
NVCOMP_CORE_API void writeFile(const std::filesystem::path& filepath, const void* data, size_t size);
NVCOMP_CORE_API void writeFile(const std::filesystem::path& filepath, const void* data, size_t size, ProgressCallback callback);

/**
 * Read exactly `size` bytes from `filepath` directly into `dst`. Uses memory
 * mapping (Windows: CreateFileMappingW + MapViewOfFile + PrefetchVirtualMemory;
 * POSIX: mmap + MAP_POPULATE) for files >= 16 MB to skip the kernel-buffer ->
 * user-buffer copy that std::ifstream::read incurs. Falls back to ifstream for
 * small files and on any mmap failure. Throws on open / read failure.
 */
NVCOMP_CORE_API void readFileInto(const std::filesystem::path& filepath, uint8_t* dst, uint64_t size);

// String overloads for backward compatibility
NVCOMP_CORE_API std::vector<uint8_t> readFile(const std::string& filename);
NVCOMP_CORE_API void writeFile(const std::string& filename, const void* data, size_t size);
NVCOMP_CORE_API void writeFile(const std::string& filename, const void* data, size_t size, ProgressCallback callback);
NVCOMP_CORE_API std::string normalizePath(const std::string& path);
NVCOMP_CORE_API std::string normalizePath(const std::filesystem::path& path);
NVCOMP_CORE_API std::string getRelativePath(const std::filesystem::path& path, const std::filesystem::path& base);
NVCOMP_CORE_API std::string getRelativePath(const std::string& path, const std::string& base);
NVCOMP_CORE_API bool isDirectory(const std::string& path);
NVCOMP_CORE_API void createDirectories(const std::string& path);

// ============================================================================
// Volume Support
// ============================================================================

NVCOMP_CORE_API std::string generateVolumeFilename(const std::string& baseFile, size_t volumeIndex);
NVCOMP_CORE_API std::vector<std::string> detectVolumeFiles(const std::string& firstVolume);
NVCOMP_CORE_API bool isVolumeFile(const std::string& filename);
NVCOMP_CORE_API uint64_t parseVolumeSize(const std::string& sizeStr);
NVCOMP_CORE_API bool checkGPUMemoryForVolume(uint64_t volumeSize);

// ============================================================================
// Archive Operations
// ============================================================================

NVCOMP_CORE_API std::vector<uint8_t> createArchiveFromFolder(const std::string& folderPath, ProgressCallback callback = nullptr);
NVCOMP_CORE_API std::vector<uint8_t> createArchiveFromFile(const std::string& filePath, ProgressCallback callback = nullptr);
NVCOMP_CORE_API std::vector<uint8_t> createArchiveFromFileList(const std::vector<std::string>& filePaths, ProgressCallback callback = nullptr);
NVCOMP_CORE_API void extractArchive(const std::vector<uint8_t>& archiveData, const std::string& outputPath);
NVCOMP_CORE_API void listArchive(const std::vector<uint8_t>& archiveData);

/**
 * Incremental (streaming) NVAR archive extractor.
 *
 * feed() accepts the decompressed archive byte stream in arbitrary pieces --
 * headers, paths, and file data may be split across calls -- and writes files
 * as their bytes arrive. With writerThreads > 0, whole-in-one-feed files are
 * written by a worker pool; the optional per-feed releaseBuffer callback fires
 * only after every write referencing that feed's bytes has completed, so
 * callers can hand over reusable buffers (e.g. pinned GPU-download buffers)
 * without copying. Files spanning multiple feeds are written inline on the
 * feeding thread. finish() drains the pool, validates the stream ended
 * exactly at the expected file count, and rethrows the first worker error.
 * abandon() stops everything without validation (used before an external
 * fallback re-extracts from scratch).
 *
 * The legacy whole-buffer extractArchive() is implemented on top of this
 * class, so all archive-format parsing lives in one place.
 */
class NVCOMP_CORE_API ArchiveExtractor {
public:
    explicit ArchiveExtractor(const std::string& outputPath, size_t writerThreads = 0);
    ~ArchiveExtractor();
    ArchiveExtractor(const ArchiveExtractor&) = delete;
    ArchiveExtractor& operator=(const ArchiveExtractor&) = delete;

    void feed(const uint8_t* data, size_t n,
              std::function<void()> releaseBuffer = nullptr);
    void finish();
    void abandon();
    uint64_t bytesWritten() const;
    uint32_t filesWritten() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
NVCOMP_CORE_API void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable);

/**
 * One file entry that the streaming compression pipeline consumes. Held by
 * value so the streaming function can iterate without reopening directories.
 */
struct ArchiveEntry {
    std::filesystem::path filePath;     // absolute / OS-native path on disk
    std::string relativePath;           // path stored inside the archive
    uint64_t fileSize;                  // cached fs::file_size(filePath)
};

/**
 * Enumerate the files an archive built from `folderOrFile` would contain,
 * pre-computing relative paths and sizes. For a single regular file the
 * returned vector has exactly one entry whose relativePath is the filename.
 * For a directory the vector has one entry per regular file (recursive).
 */
NVCOMP_CORE_API std::vector<ArchiveEntry> collectArchiveEntries(const std::string& folderOrFile);

/**
 * Same as collectArchiveEntries, but takes an explicit list of files and/or
 * folders. Mirrors the behavior of createArchiveFromFileList: files use their
 * parent directory as the relative-path base; folders are expanded recursively
 * with the folder itself as the base.
 */
NVCOMP_CORE_API std::vector<ArchiveEntry> collectArchiveEntriesFromList(const std::vector<std::string>& filePaths);

// ============================================================================
// GPU Compression (Batched API)
// ============================================================================

NVCOMP_CORE_API void compressGPUBatched(AlgoType algo, const std::string& inputPath, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr,
                                         CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void compressGPUBatchedFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr,
                                         CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void decompressGPUBatched(AlgoType algo, const std::string& inputFile, 
                                           const std::string& outputPath,
                                           ProgressCallback callback = nullptr,
                                           CompressionStats* outStats = nullptr);

// ============================================================================
// GPU Compression (Manager API)
// ============================================================================

NVCOMP_CORE_API void compressGPUManager(AlgoType algo, const std::string& inputPath, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr,
                                         CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void compressGPUManagerFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr,
                                         CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void decompressGPUManager(const std::string& inputFile, const std::string& outputPath,
                                           ProgressCallback callback = nullptr,
                                           CompressionStats* outStats = nullptr);

// ============================================================================
// CPU Compression
// ============================================================================

NVCOMP_CORE_API void compressCPU(AlgoType algo, const std::string& inputPath, 
                                  const std::string& outputFile, uint64_t maxVolumeSize,
                                  ProgressCallback callback = nullptr,
                                  CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void compressCPUFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                  const std::string& outputFile, uint64_t maxVolumeSize,
                                  ProgressCallback callback = nullptr,
                                  CompressionStats* outStats = nullptr);
NVCOMP_CORE_API void decompressCPU(AlgoType algo, const std::string& inputFile, 
                                    const std::string& outputPath,
                                    ProgressCallback callback = nullptr,
                                    CompressionStats* outStats = nullptr);

// ============================================================================
// Stats Helpers
// ============================================================================

/**
 * Finalize a CompressionStats struct: compute totalSec from the per-phase fields
 * (if not already set) and derive throughputMBps, throughputGBps, ratio from
 * inputBytes/outputBytes.
 */
NVCOMP_CORE_API void finalizeStats(CompressionStats& stats);

/**
 * Format a CompressionStats struct as a single multi-line summary string.
 * The CLI and GUI both render this so their output matches exactly.
 */
NVCOMP_CORE_API std::string formatStatsSummary(const CompressionStats& stats, const std::string& opName);

/**
 * Wrap a progress callback so it fires at most maxRateHz times per second AND
 * only when overallProgress advances by >= 1 percent (or on stage change /
 * terminal 100%). Returns nullptr if `raw` is null.
 *
 * This is the single mechanism that protects the GUI from being flooded by
 * thousands of cross-thread Qt signals from the inside of GPU loops.
 */
NVCOMP_CORE_API ProgressCallback makeThrottledCallback(ProgressCallback raw, double maxRateHz = 30.0);

/**
 * Process-wide verbose flag. When false (the default), the core suppresses
 * per-file "Adding:", "Collecting files...", and per-volume progress chatter
 * during compression so std::cout doesn't dominate Read-phase throughput.
 *
 * Final result summaries (compression stats, "Multi-Volume Compression
 * SUCCESSFUL", extraction completion) are always printed regardless of this
 * flag - this only gates incremental progress output.
 *
 * The CLI flips it on with --verbose / -v at startup. The GUI never touches
 * it, so its compression operations stay silent.
 */
NVCOMP_CORE_API void setVerbose(bool verbose);
NVCOMP_CORE_API bool isVerbose();

// Helper function for decompressing batched format (used by GPU decompression too)
NVCOMP_CORE_API std::vector<uint8_t> decompressBatchedFormatCPU(AlgoType algo, 
                                                                  const std::vector<uint8_t>& compressedData);

// ============================================================================
// Algorithm Detection
// ============================================================================

NVCOMP_CORE_API AlgoType detectAlgorithmFromFile(const std::string& filename);

} // namespace nvcomp_core


