#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

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

NVCOMP_CORE_API std::vector<uint8_t> readFile(const std::string& filename);
NVCOMP_CORE_API void writeFile(const std::string& filename, const void* data, size_t size);
NVCOMP_CORE_API void writeFile(const std::string& filename, const void* data, size_t size, ProgressCallback callback);
NVCOMP_CORE_API std::string normalizePath(const std::string& path);
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
NVCOMP_CORE_API void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable);

// ============================================================================
// GPU Compression (Batched API)
// ============================================================================

NVCOMP_CORE_API void compressGPUBatched(AlgoType algo, const std::string& inputPath, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr);
NVCOMP_CORE_API void compressGPUBatchedFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr);
NVCOMP_CORE_API void decompressGPUBatched(AlgoType algo, const std::string& inputFile, 
                                           const std::string& outputPath,
                                           ProgressCallback callback = nullptr);

// ============================================================================
// GPU Compression (Manager API)
// ============================================================================

NVCOMP_CORE_API void compressGPUManager(AlgoType algo, const std::string& inputPath, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr);
NVCOMP_CORE_API void compressGPUManagerFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr);
NVCOMP_CORE_API void decompressGPUManager(const std::string& inputFile, const std::string& outputPath,
                                           ProgressCallback callback = nullptr);

// ============================================================================
// CPU Compression
// ============================================================================

NVCOMP_CORE_API void compressCPU(AlgoType algo, const std::string& inputPath, 
                                  const std::string& outputFile, uint64_t maxVolumeSize,
                                  ProgressCallback callback = nullptr);
NVCOMP_CORE_API void compressCPUFileList(AlgoType algo, const std::vector<std::string>& filePaths, 
                                  const std::string& outputFile, uint64_t maxVolumeSize,
                                  ProgressCallback callback = nullptr);
NVCOMP_CORE_API void decompressCPU(AlgoType algo, const std::string& inputFile, 
                                    const std::string& outputPath,
                                    ProgressCallback callback = nullptr);

// Helper function for decompressing batched format (used by GPU decompression too)
NVCOMP_CORE_API std::vector<uint8_t> decompressBatchedFormatCPU(AlgoType algo, 
                                                                  const std::vector<uint8_t>& compressedData);

// ============================================================================
// Algorithm Detection
// ============================================================================

NVCOMP_CORE_API AlgoType detectAlgorithmFromFile(const std::string& filename);

} // namespace nvcomp_core


