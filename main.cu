/*
 * nvCOMP CLI with CPU Fallback
 * 
 * Two implementations:
 * - Batched API (LZ4, Snappy, Zstd): Cross-compatible with CPU
 * - Manager API (GDeflate, ANS, Bitcomp): GPU-only
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <filesystem>
#include <algorithm>

#include <cuda_runtime.h>

namespace fs = std::filesystem;

// Batched API headers (for cross-compatible algorithms)
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

// Manager API headers (for GPU-only algorithms)
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

// CPU compression libraries
#include "lz4.h"
#include "lz4hc.h"
#include "snappy.h"
#include "zstd.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvCOMP Error at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Chunk size for batched API
constexpr size_t CHUNK_SIZE = 1 << 16; // 64KB

// Archive magic number for identification
constexpr uint32_t ARCHIVE_MAGIC = 0x4E564152; // "NVAR" (NvCOMP ARchive)
constexpr uint32_t ARCHIVE_VERSION = 1;

// Batched compression metadata magic
constexpr uint32_t BATCHED_MAGIC = 0x4E564243; // "NVBC" (NvCOMP Batched Compression)
constexpr uint32_t BATCHED_VERSION = 1;

// Archive header structure
struct ArchiveHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t fileCount;
    uint32_t reserved;
};

// File entry in archive
struct FileEntry {
    uint32_t pathLength;
    uint64_t fileSize;
    // Followed by: path (pathLength bytes), then file data (fileSize bytes)
};

// Batched compression header (for GPU batched API)
struct BatchedHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t uncompressedSize;
    uint32_t chunkCount;
    uint32_t chunkSize;
    uint32_t algorithm; // AlgoType
    uint32_t reserved;
    // Followed by: chunk sizes array (chunkCount * uint64_t), then compressed data
};

// Algorithm types
enum AlgoType {
    ALGO_LZ4,
    ALGO_SNAPPY,
    ALGO_ZSTD,
    ALGO_GDEFLATE,
    ALGO_ANS,
    ALGO_BITCOMP,
    ALGO_UNKNOWN
};

AlgoType parseAlgorithm(const std::string& algo) {
    if (algo == "lz4") return ALGO_LZ4;
    if (algo == "snappy") return ALGO_SNAPPY;
    if (algo == "zstd") return ALGO_ZSTD;
    if (algo == "gdeflate") return ALGO_GDEFLATE;
    if (algo == "ans") return ALGO_ANS;
    if (algo == "bitcomp") return ALGO_BITCOMP;
    return ALGO_UNKNOWN;
}

std::string algoToString(AlgoType algo) {
    switch(algo) {
        case ALGO_LZ4: return "lz4";
        case ALGO_SNAPPY: return "snappy";
        case ALGO_ZSTD: return "zstd";
        case ALGO_GDEFLATE: return "gdeflate";
        case ALGO_ANS: return "ans";
        case ALGO_BITCOMP: return "bitcomp";
        default: return "unknown";
    }
}

bool isCrossCompatible(AlgoType algo) {
    return algo == ALGO_LZ4 || algo == ALGO_SNAPPY || algo == ALGO_ZSTD;
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

// Forward declarations
AlgoType detectAlgorithmFromFile(const std::string& filename);
std::vector<uint8_t> decompressBatchedFormat(AlgoType algo, const std::vector<uint8_t>& compressedData);

std::vector<uint8_t> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    throw std::runtime_error("Failed to read file: " + filename);
}

void writeFile(const std::string& filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data), size);
}

// ============================================================================
// CROSS-PLATFORM PATH AND DIRECTORY UTILITIES
// ============================================================================

// Normalize path separators to forward slashes (cross-platform standard)
std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

// Get relative path from base directory
std::string getRelativePath(const fs::path& path, const fs::path& base) {
    fs::path relativePath = fs::relative(path, base);
    return normalizePath(relativePath.string());
}

// Check if path is a directory
bool isDirectory(const std::string& path) {
    try {
        return fs::is_directory(path);
    } catch (...) {
        return false;
    }
}

// Recursively collect all files in a directory
std::vector<fs::path> collectFiles(const fs::path& dirPath) {
    std::vector<fs::path> files;
    
    if (!fs::exists(dirPath)) {
        throw std::runtime_error("Directory does not exist: " + dirPath.string());
    }
    
    if (!fs::is_directory(dirPath)) {
        throw std::runtime_error("Not a directory: " + dirPath.string());
    }
    
    for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    
    return files;
}

// Create directories recursively
void createDirectories(const fs::path& path) {
    if (!path.empty() && path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
}

// ============================================================================
// ARCHIVE CREATION AND EXTRACTION
// ============================================================================

// Create an uncompressed archive from directory
std::vector<uint8_t> createArchive(const std::string& inputPath) {
    std::vector<uint8_t> archiveData;
    std::vector<fs::path> files;
    fs::path basePath;
    
    if (isDirectory(inputPath)) {
        basePath = fs::path(inputPath);
        files = collectFiles(basePath);
        std::cout << "Collecting files from directory: " << inputPath << std::endl;
        std::cout << "Found " << files.size() << " file(s)" << std::endl;
    } else {
        // Single file - create archive with just this file
        basePath = fs::path(inputPath).parent_path();
        files.push_back(fs::path(inputPath));
        std::cout << "Adding single file: " << inputPath << std::endl;
    }
    
    if (files.empty()) {
        throw std::runtime_error("No files to archive");
    }
    
    // Write header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = static_cast<uint32_t>(files.size());
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));
    
    // Write each file
    for (const auto& filePath : files) {
        std::string relativePath = getRelativePath(filePath, basePath);
        if (relativePath.empty() || relativePath == ".") {
            relativePath = filePath.filename().string();
        }
        
        std::cout << "  Adding: " << relativePath << std::flush;
        
        auto fileData = readFile(filePath.string());
        
        FileEntry entry;
        entry.pathLength = static_cast<uint32_t>(relativePath.length());
        entry.fileSize = fileData.size();
        
        // Write entry header
        const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
        archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
        
        // Write path
        archiveData.insert(archiveData.end(), relativePath.begin(), relativePath.end());
        
        // Write file data
        archiveData.insert(archiveData.end(), fileData.begin(), fileData.end());
        
        std::cout << " (" << fileData.size() << " bytes)" << std::endl;
    }
    
    return archiveData;
}

// Extract archive to directory
void extractArchive(const std::vector<uint8_t>& archiveData, const std::string& outputPath) {
    if (archiveData.size() < sizeof(ArchiveHeader)) {
        throw std::runtime_error("Invalid archive: too small");
    }
    
    size_t offset = 0;
    
    // Read header
    ArchiveHeader header;
    std::memcpy(&header, archiveData.data() + offset, sizeof(ArchiveHeader));
    offset += sizeof(ArchiveHeader);
    
    if (header.magic != ARCHIVE_MAGIC) {
        throw std::runtime_error("Invalid archive: bad magic number");
    }
    
    if (header.version != ARCHIVE_VERSION) {
        throw std::runtime_error("Unsupported archive version");
    }
    
    std::cout << "Extracting " << header.fileCount << " file(s) to: " << outputPath << std::endl;
    
    // Create output directory if it doesn't exist
    if (!outputPath.empty()) {
        fs::create_directories(outputPath);
    }
    
    // Extract each file
    for (uint32_t i = 0; i < header.fileCount; i++) {
        if (offset + sizeof(FileEntry) > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file entry");
        }
        
        FileEntry entry;
        std::memcpy(&entry, archiveData.data() + offset, sizeof(FileEntry));
        offset += sizeof(FileEntry);
        
        if (offset + entry.pathLength + entry.fileSize > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file data");
        }
        
        // Read path
        std::string filePath(
            reinterpret_cast<const char*>(archiveData.data() + offset),
            entry.pathLength
        );
        offset += entry.pathLength;
        
        std::cout << "  Extracting: " << filePath << " (" << entry.fileSize << " bytes)" << std::endl;
        
        // Construct full output path
        fs::path fullPath = fs::path(outputPath) / fs::path(filePath);
        
        // Create parent directories
        createDirectories(fullPath);
        
        // Write file
        writeFile(fullPath.string(), archiveData.data() + offset, entry.fileSize);
        offset += entry.fileSize;
    }
    
    std::cout << "Extraction complete." << std::endl;
}

// List archive contents
void listArchive(const std::vector<uint8_t>& archiveData) {
    if (archiveData.size() < sizeof(ArchiveHeader)) {
        throw std::runtime_error("Invalid archive: too small");
    }
    
    size_t offset = 0;
    
    // Read header
    ArchiveHeader header;
    std::memcpy(&header, archiveData.data() + offset, sizeof(ArchiveHeader));
    offset += sizeof(ArchiveHeader);
    
    if (header.magic != ARCHIVE_MAGIC) {
        throw std::runtime_error("Invalid archive: bad magic number");
    }
    
    if (header.version != ARCHIVE_VERSION) {
        throw std::runtime_error("Unsupported archive version");
    }
    
    std::cout << "Archive contains " << header.fileCount << " file(s):" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    uint64_t totalSize = 0;
    
    // List each file
    for (uint32_t i = 0; i < header.fileCount; i++) {
        if (offset + sizeof(FileEntry) > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file entry");
        }
        
        FileEntry entry;
        std::memcpy(&entry, archiveData.data() + offset, sizeof(FileEntry));
        offset += sizeof(FileEntry);
        
        if (offset + entry.pathLength + entry.fileSize > archiveData.size()) {
            throw std::runtime_error("Invalid archive: truncated file data");
        }
        
        // Read path
        std::string filePath(
            reinterpret_cast<const char*>(archiveData.data() + offset),
            entry.pathLength
        );
        offset += entry.pathLength;
        
        // Skip file data
        offset += entry.fileSize;
        totalSize += entry.fileSize;
        
        // Format size with appropriate unit
        double displaySize = static_cast<double>(entry.fileSize);
        std::string sizeUnit = "B";
        
        if (displaySize >= 1024 * 1024 * 1024) {
            displaySize /= (1024.0 * 1024.0 * 1024.0);
            sizeUnit = "GB";
        } else if (displaySize >= 1024 * 1024) {
            displaySize /= (1024.0 * 1024.0);
            sizeUnit = "MB";
        } else if (displaySize >= 1024) {
            displaySize /= 1024.0;
            sizeUnit = "KB";
        }
        
        std::cout << "  " << std::left << std::setw(50) << filePath
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2) 
                  << displaySize << " " << sizeUnit << std::endl;
    }
    
    std::cout << std::string(60, '-') << std::endl;
    
    // Total size
    double totalDisplaySize = static_cast<double>(totalSize);
    std::string totalUnit = "B";
    
    if (totalDisplaySize >= 1024 * 1024 * 1024) {
        totalDisplaySize /= (1024.0 * 1024.0 * 1024.0);
        totalUnit = "GB";
    } else if (totalDisplaySize >= 1024 * 1024) {
        totalDisplaySize /= (1024.0 * 1024.0);
        totalUnit = "MB";
    } else if (totalDisplaySize >= 1024) {
        totalDisplaySize /= 1024.0;
        totalUnit = "KB";
    }
    
    std::cout << "Total: " << std::fixed << std::setprecision(2) 
              << totalDisplaySize << " " << totalUnit << std::endl;
}

// ============================================================================
// CPU COMPRESSION/DECOMPRESSION (LZ4, Snappy, Zstd)
// ============================================================================

std::vector<uint8_t> compressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t compSize = 0;
    
    if (algo == ALGO_LZ4) {
        size_t maxSize = LZ4_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(inputData.data()),
            reinterpret_cast<char*>(outputData.data()),
            inputSize,
            maxSize,
            LZ4HC_CLEVEL_DEFAULT
        );
        if (compSize == 0) {
            throw std::runtime_error("LZ4 CPU compression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t maxSize = snappy::MaxCompressedLength(inputSize);
        outputData.resize(maxSize);
        snappy::RawCompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data()),
            &compSize
        );
        if (compSize == 0) {
            throw std::runtime_error("Snappy CPU compression failed");
        }
    } else if (algo == ALGO_ZSTD) {
        size_t maxSize = ZSTD_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = ZSTD_compress(
            outputData.data(),
            maxSize,
            inputData.data(),
            inputSize,
            ZSTD_CLEVEL_DEFAULT
        );
        if (ZSTD_isError(compSize)) {
            throw std::runtime_error("Zstd CPU compression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU compression");
    }
    
    outputData.resize(compSize);
    return outputData;
}

std::vector<uint8_t> decompressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t decompSize = 0;
    
    if (algo == ALGO_LZ4) {
        // Try different output sizes
        for (size_t multiplier = 10; multiplier <= 1000; multiplier *= 10) {
            outputData.resize(inputSize * multiplier);
            int result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(inputData.data()),
                reinterpret_cast<char*>(outputData.data()),
                inputSize,
                outputData.size()
            );
            if (result > 0) {
                decompSize = result;
                break;
            }
        }
        if (decompSize == 0) {
            throw std::runtime_error("LZ4 CPU decompression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t uncompressedLength;
        if (!snappy::GetUncompressedLength(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            &uncompressedLength
        )) {
            throw std::runtime_error("Snappy: Failed to get uncompressed length");
        }
        outputData.resize(uncompressedLength);
        if (!snappy::RawUncompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data())
        )) {
            throw std::runtime_error("Snappy CPU decompression failed");
        }
        decompSize = uncompressedLength;
    } else if (algo == ALGO_ZSTD) {
        unsigned long long uncompressedSize = ZSTD_getFrameContentSize(inputData.data(), inputSize);
        if (uncompressedSize == ZSTD_CONTENTSIZE_ERROR || uncompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Zstd: Failed to get uncompressed size");
        }
        outputData.resize(uncompressedSize);
        decompSize = ZSTD_decompress(
            outputData.data(),
            uncompressedSize,
            inputData.data(),
            inputSize
        );
        if (ZSTD_isError(decompSize)) {
            throw std::runtime_error("Zstd CPU decompression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU decompression");
    }
    
    outputData.resize(decompSize);
    return outputData;
}

void compressCPU(AlgoType algo, const std::string& inputPath, const std::string& outputFile) {
    std::cout << "Using CPU compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> inputData;
    if (isDirectory(inputPath)) {
        inputData = createArchive(inputPath);
    } else {
        inputData = createArchive(inputPath);
    }
    
    size_t inputSize = inputData.size();
    std::cout << "Archive size: " << inputSize << " bytes" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto outputData = compressDataCPU(algo, inputData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t compSize = outputData.size();
    std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x" << std::endl;
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    writeFile(outputFile, outputData.data(), outputData.size());
}

void decompressCPU(AlgoType algo, const std::string& inputFile, const std::string& outputPath) {
    // Try to auto-detect algorithm if not specified
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Using CPU decompression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use batched format handler which works for both batched and standard formats
    auto archiveData = decompressBatchedFormat(algo, inputData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t decompSize = archiveData.size();
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// GPU BATCHED API (LZ4, Snappy, Zstd) - Cross-compatible with CPU
// ============================================================================

void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile) {
    std::cout << "Using GPU batched compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> inputData;
    if (isDirectory(inputPath)) {
        inputData = createArchive(inputPath);
    } else {
        inputData = createArchive(inputPath);
    }
    
    size_t inputSize = inputData.size();
    std::cout << "Archive size: " << inputSize << " bytes" << std::endl;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Calculate chunks
    size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::cout << "Chunks: " << chunk_count << std::endl;
    
    // Prepare input chunks on host
    std::vector<void*> h_input_ptrs(chunk_count);
    std::vector<size_t> h_input_sizes(chunk_count);
    
    for (size_t i = 0; i < chunk_count; i++) {
        size_t offset = i * CHUNK_SIZE;
        h_input_sizes[i] = std::min(CHUNK_SIZE, inputSize - offset);
    }
    
    // Allocate device memory for input
    uint8_t* d_input_data;
    CUDA_CHECK(cudaMalloc(&d_input_data, inputSize));
    CUDA_CHECK(cudaMemcpy(d_input_data, inputData.data(), inputSize, cudaMemcpyHostToDevice));
    
    // Setup input pointers
    void** d_input_ptrs;
    size_t* d_input_sizes;
    CUDA_CHECK(cudaMalloc(&d_input_ptrs, sizeof(void*) * chunk_count));
    CUDA_CHECK(cudaMalloc(&d_input_sizes, sizeof(size_t) * chunk_count));
    
    for (size_t i = 0; i < chunk_count; i++) {
        h_input_ptrs[i] = d_input_data + i * CHUNK_SIZE;
    }
    CUDA_CHECK(cudaMemcpy(d_input_ptrs, h_input_ptrs.data(), sizeof(void*) * chunk_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_sizes, h_input_sizes.data(), sizeof(size_t) * chunk_count, cudaMemcpyHostToDevice));
    
    // Get temp size and max output size
    size_t temp_bytes;
    size_t max_out_bytes;
    
    if (algo == ALGO_LZ4) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &max_out_bytes));
    } else if (algo == ALGO_SNAPPY) {
        NVCOMP_CHECK(nvcompBatchedSnappyCompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &max_out_bytes));
    } else if (algo == ALGO_ZSTD) {
        NVCOMP_CHECK(nvcompBatchedZstdCompressGetTempSizeAsync(
            chunk_count, CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &temp_bytes, inputSize));
        NVCOMP_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &max_out_bytes));
    }
    
    // Allocate temp and output
    void* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    
    uint8_t* d_output_data;
    CUDA_CHECK(cudaMalloc(&d_output_data, max_out_bytes * chunk_count));
    
    void** d_output_ptrs;
    size_t* d_output_sizes;
    CUDA_CHECK(cudaMalloc(&d_output_ptrs, sizeof(void*) * chunk_count));
    CUDA_CHECK(cudaMalloc(&d_output_sizes, sizeof(size_t) * chunk_count));
    
    std::vector<void*> h_output_ptrs(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) {
        h_output_ptrs[i] = d_output_data + i * max_out_bytes;
    }
    CUDA_CHECK(cudaMemcpy(d_output_ptrs, h_output_ptrs.data(), sizeof(void*) * chunk_count, cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Compress
    if (algo == ALGO_LZ4) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedLZ4CompressDefaultOpts, nullptr, stream));
    } else if (algo == ALGO_SNAPPY) {
        NVCOMP_CHECK(nvcompBatchedSnappyCompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedSnappyCompressDefaultOpts, nullptr, stream));
    } else if (algo == ALGO_ZSTD) {
        NVCOMP_CHECK(nvcompBatchedZstdCompressAsync(
            d_input_ptrs, d_input_sizes, CHUNK_SIZE, chunk_count,
            d_temp, temp_bytes, d_output_ptrs, d_output_sizes,
            nvcompBatchedZstdCompressDefaultOpts, nullptr, stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get output sizes
    std::vector<size_t> h_output_sizes(chunk_count);
    CUDA_CHECK(cudaMemcpy(h_output_sizes.data(), d_output_sizes, sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost));
    
    // Calculate total size
    size_t totalCompSize = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        totalCompSize += h_output_sizes[i];
    }
    
    std::cout << "Compressed size: " << totalCompSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / totalCompSize << "x" << std::endl;
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Create output with metadata
    std::vector<uint8_t> outputData;
    
    // Write batched header
    BatchedHeader header;
    header.magic = BATCHED_MAGIC;
    header.version = BATCHED_VERSION;
    header.uncompressedSize = inputSize;
    header.chunkCount = static_cast<uint32_t>(chunk_count);
    header.chunkSize = CHUNK_SIZE;
    header.algorithm = static_cast<uint32_t>(algo);
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    outputData.insert(outputData.end(), headerBytes, headerBytes + sizeof(BatchedHeader));
    
    // Write chunk sizes
    std::vector<uint64_t> chunkSizes64(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) {
        chunkSizes64[i] = h_output_sizes[i];
    }
    const uint8_t* sizesBytes = reinterpret_cast<const uint8_t*>(chunkSizes64.data());
    outputData.insert(outputData.end(), sizesBytes, sizesBytes + sizeof(uint64_t) * chunk_count);
    
    // Copy compressed chunks
    size_t dataStart = outputData.size();
    outputData.resize(dataStart + totalCompSize);
    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        CUDA_CHECK(cudaMemcpy(outputData.data() + dataStart + offset, h_output_ptrs[i], h_output_sizes[i], cudaMemcpyDeviceToHost));
        offset += h_output_sizes[i];
    }
    
    size_t totalSize = outputData.size();
    std::cout << "Total size with metadata: " << totalSize << " bytes" << std::endl;
    
    writeFile(outputFile, outputData.data(), outputData.size());
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_input_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_data);
    cudaFree(d_output_ptrs);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);
}

// Detect algorithm from batched format file header
AlgoType detectAlgorithmFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return ALGO_UNKNOWN;
    }
    
    BatchedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BatchedHeader));
    
    if (file.gcount() < sizeof(BatchedHeader)) {
        return ALGO_UNKNOWN;
    }
    
    if (header.magic != BATCHED_MAGIC) {
        return ALGO_UNKNOWN;
    }
    
    return static_cast<AlgoType>(header.algorithm);
}

std::vector<uint8_t> decompressBatchedFormat(AlgoType algo, const std::vector<uint8_t>& compressedData) {
    // Check if it's batched format
    if (compressedData.size() < sizeof(BatchedHeader)) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    BatchedHeader header;
    std::memcpy(&header, compressedData.data(), sizeof(BatchedHeader));
    
    if (header.magic != BATCHED_MAGIC) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    // It's a batched format - extract the compressed chunks and decompress with CPU
    // Use algorithm from header (auto-detect)
    AlgoType actualAlgo = static_cast<AlgoType>(header.algorithm);
    std::cout << "Auto-detected algorithm: " << algoToString(actualAlgo) << std::endl;
    
    size_t chunk_count = header.chunkCount;
    size_t uncompressedSize = header.uncompressedSize;
    
    // Read chunk sizes
    size_t offset = sizeof(BatchedHeader);
    std::vector<uint64_t> chunkSizes64(chunk_count);
    std::memcpy(chunkSizes64.data(), compressedData.data() + offset, sizeof(uint64_t) * chunk_count);
    offset += sizeof(uint64_t) * chunk_count;
    
    // Decompress each chunk with CPU
    std::vector<uint8_t> result;
    result.reserve(uncompressedSize);
    
    for (size_t i = 0; i < chunk_count; i++) {
        size_t chunkCompSize = static_cast<size_t>(chunkSizes64[i]);
        
        // Extract this chunk's data
        std::vector<uint8_t> chunkCompressed(chunkCompSize);
        std::memcpy(chunkCompressed.data(), compressedData.data() + offset, chunkCompSize);
        offset += chunkCompSize;
        
        // Decompress this chunk (use detected algorithm)
        auto chunkDecompressed = decompressDataCPU(actualAlgo, chunkCompressed);
        
        // Append to result
        result.insert(result.end(), chunkDecompressed.begin(), chunkDecompressed.end());
    }
    
    if (result.size() != uncompressedSize) {
        std::cerr << "Warning: Decompressed size (" << result.size() << ") doesn't match expected size (" << uncompressedSize << ")" << std::endl;
    }
    
    return result;
}

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath) {
    // Try to auto-detect algorithm if not specified
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Decompressing (" << algoToString(algo) << ")..." << std::endl;
    
    auto compressedData = readFile(inputFile);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Decompress (handles both batched and standard formats)
    auto archiveData = decompressBatchedFormat(algo, compressedData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t decompSize = archiveData.size();
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// GPU MANAGER API (GDeflate, ANS, Bitcomp) - GPU-only
// ============================================================================

void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile) {
    std::cout << "Using GPU manager compression (" << algoToString(algo) << ")..." << std::endl;
    
    // Create archive (handles both files and directories)
    std::vector<uint8_t> inputData;
    if (isDirectory(inputPath)) {
        inputData = createArchive(inputPath);
    } else {
        inputData = createArchive(inputPath);
    }
    
    size_t inputSize = inputData.size();
    std::cout << "Archive size: " << inputSize << " bytes" << std::endl;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    std::shared_ptr<nvcompManagerBase> manager;
    
    if (algo == ALGO_GDEFLATE) {
        manager = std::make_shared<GdeflateManager>(
            CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    } else if (algo == ALGO_ANS) {
        manager = std::make_shared<ANSManager>(
            CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
    } else if (algo == ALGO_BITCOMP) {
        manager = std::make_shared<BitcompManager>(
            CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    }
    
    CompressionConfig comp_config = manager->configure_compression(inputSize);
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->compress(d_input, d_output, comp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    size_t compSize = manager->get_compressed_output_size(d_output);
    
    std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x" << std::endl;
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    std::vector<uint8_t> outputData(compSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
    
    writeFile(outputFile, outputData.data(), outputData.size());
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath) {
    std::cout << "Using GPU manager decompression (auto-detect)..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    auto manager = create_manager(d_input, stream);
    
    DecompressionConfig decomp_config = manager->configure_decompression(d_input);
    size_t outputSize = decomp_config.decomp_data_size;
    std::cout << "Detected original size: " << outputSize << " bytes" << std::endl;
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->decompress(d_output, d_input, decomp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    std::vector<uint8_t> archiveData(outputSize);
    CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// LIST MODE - Show archive contents
// ============================================================================

void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable) {
    // Try to auto-detect algorithm if not specified
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Listing archive contents..." << std::endl;
    
    // Read compressed file
    auto compressedData = readFile(inputFile);
    
    // Decompress
    std::vector<uint8_t> archiveData;
    
    if (isCrossCompatible(algo)) {
        // Use helper function that handles both batched and standard formats (auto-detects algorithm)
        archiveData = decompressBatchedFormat(algo, compressedData);
    } else {
        // GPU Manager decompression
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        uint8_t* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, compressedData.size()));
        CUDA_CHECK(cudaMemcpyAsync(d_input, compressedData.data(), compressedData.size(), cudaMemcpyHostToDevice, stream));
        
        auto manager = create_manager(d_input, stream);
        
        DecompressionConfig decomp_config = manager->configure_decompression(d_input);
        size_t outputSize = decomp_config.decomp_data_size;
        
        uint8_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, outputSize));
        
        manager->decompress(d_output, d_input, decomp_config);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        archiveData.resize(outputSize);
        CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    // List archive
    listArchive(archiveData);
}

// ============================================================================
// MAIN
// ============================================================================

void printUsage(const char* appName) {
    std::cerr << "nvCOMP CLI with CPU Fallback & Folder Support\n\n";
    std::cerr << "Usage:\n";
    std::cerr << "  Compress:   " << appName << " -c <input> <output> <algorithm> [--cpu]\n";
    std::cerr << "  Decompress: " << appName << " -d <input> <output> [algorithm] [--cpu]\n";
    std::cerr << "  List:       " << appName << " -l <archive> [algorithm] [--cpu]\n\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  <input>     Input file or directory (for compression)\n";
    std::cerr << "  <output>    Output file (compression) or directory (decompression)\n";
    std::cerr << "  <archive>   Compressed archive file to list\n";
    std::cerr << "  [algorithm] Optional for -d/-l (auto-detected), required for -c\n\n";
    std::cerr << "Algorithms:\n";
    std::cerr << "  Cross-compatible (GPU/CPU): lz4, snappy, zstd\n";
    std::cerr << "  GPU-only: gdeflate, ans, bitcomp\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --cpu    Force CPU mode\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  # Compress single file (algorithm required)\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4\n\n";
    std::cerr << "  # Compress entire folder\n";
    std::cerr << "  " << appName << " -c mydata/ output.zstd zstd\n\n";
    std::cerr << "  # Decompress with auto-detection (no algorithm needed!)\n";
    std::cerr << "  " << appName << " -d output.lz4 restored/\n\n";
    std::cerr << "  # Decompress with explicit algorithm\n";
    std::cerr << "  " << appName << " -d output.lz4 restored/ lz4\n\n";
    std::cerr << "  # List archive with auto-detection\n";
    std::cerr << "  " << appName << " -l output.zstd\n\n";
    std::cerr << "  # Force CPU mode\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4 --cpu\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string mode = argv[1];
        
        // List mode requires at least 3 args, compress/decompress require at least 4
        if (mode == "-l") {
            if (argc < 3) {
                printUsage(argv[0]);
                return 1;
            }
        } else if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string inputPath = argv[2];
        std::string outputPath = (argc >= 4) ? argv[3] : "";
        std::string algoStr = "";
        bool forceCPU = false;
        
        // Parse algorithm and flags
        for (int i = (mode == "-l" ? 3 : 4); i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cpu") {
                forceCPU = true;
            } else {
                // Assume it's an algorithm
                algoStr = arg;
            }
        }
        
        // Parse algorithm (default to ALGO_UNKNOWN for auto-detection)
        AlgoType algo = ALGO_UNKNOWN;
        if (!algoStr.empty()) {
            algo = parseAlgorithm(algoStr);
            if (algo == ALGO_UNKNOWN) {
                std::cerr << "Unknown algorithm: " << algoStr << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else {
            // No algorithm specified, try auto-detection (only for decompression/list modes)
            if (mode == "-c") {
                std::cerr << "Error: Algorithm required for compression mode" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            // For decompression and list modes, will auto-detect from file
            algo = ALGO_LZ4; // Default fallback if auto-detection fails
        }
        
        bool cudaAvailable = isCudaAvailable();
        bool useCPU = forceCPU || !cudaAvailable;
        
        if (!cudaAvailable && !forceCPU) {
            std::cout << "CUDA not available, falling back to CPU..." << std::endl;
            useCPU = true;
        }
        
        if (useCPU && !isCrossCompatible(algo)) {
            std::cerr << "Error: Algorithm '" << algoStr << "' is GPU-only and cannot run on CPU" << std::endl;
            return 1;
        }
        
        if (mode == "-c") {
            // Compression mode
            if (useCPU) {
                compressCPU(algo, inputPath, outputPath);
            } else {
                if (isCrossCompatible(algo)) {
                    compressGPUBatched(algo, inputPath, outputPath);
                } else {
                    compressGPUManager(algo, inputPath, outputPath);
                }
            }
        } else if (mode == "-d") {
            // Decompression mode
            if (useCPU) {
                decompressCPU(algo, inputPath, outputPath);
            } else {
                if (isCrossCompatible(algo)) {
                    // Use GPU batched decompression
                    decompressGPUBatched(algo, inputPath, outputPath);
                } else {
                    decompressGPUManager(inputPath, outputPath);
                }
            }
        } else if (mode == "-l") {
            // List mode
            listCompressedArchive(algo, inputPath, useCPU, cudaAvailable);
        } else {
            printUsage(argv[0]);
            return 1;
        }
        
        std::cout << "Done." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
