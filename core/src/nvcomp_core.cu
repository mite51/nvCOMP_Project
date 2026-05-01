#include "nvcomp_core.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

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

namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvCOMP Error at line " << __LINE__ << std::endl; \
            throw std::runtime_error("nvCOMP Error"); \
        } \
    } while (0)

namespace nvcomp_core {

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<std::vector<uint8_t>> splitIntoVolumes(
    const std::vector<uint8_t>& archiveData,
    uint64_t maxVolumeSize
) {
    std::vector<std::vector<uint8_t>> volumes;
    
    // If maxVolumeSize is 0 (disabled) or archive fits in single volume, return as-is
    if (maxVolumeSize == 0 || archiveData.size() <= maxVolumeSize) {
        volumes.push_back(archiveData);
        return volumes;
    }
    
    // Split into multiple volumes (mid-file splitting allowed)
    size_t remaining = archiveData.size();
    size_t offset = 0;
    size_t volumeIndex = 1;
    
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Splitting archive into volumes (max "
                  << (maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB each)...\n";
    }

    while (remaining > 0) {
        size_t volumeSize = std::min(static_cast<size_t>(maxVolumeSize), remaining);
        
        std::vector<uint8_t> volume(
            archiveData.begin() + offset,
            archiveData.begin() + offset + volumeSize
        );
        
        volumes.push_back(volume);

        if (verbose && (volumeIndex % 100 == 0 || remaining <= volumeSize)) {
            std::cout << "\r  Creating volumes... " << volumeIndex << " created" << std::flush;
        }
        
        offset += volumeSize;
        remaining -= volumeSize;
        volumeIndex++;
    }
    
    if (verbose) {
        std::cout << "\r  Created " << volumes.size() << " volume(s)" << std::string(20, ' ') << "\n";
    }
    
    return volumes;
}

// ============================================================================
// Algorithm Detection
// ============================================================================

AlgoType detectAlgorithmFromFile(const std::string& filename) {
    // Use filesystem::path to properly handle Unicode paths on Windows
    std::ifstream file(fs::path(filename), std::ios::binary);
    if (!file.is_open()) {
        return ALGO_UNKNOWN;
    }
    
    // Try to read BatchedHeader
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

// ============================================================================
// GPU Batched Compression
// ============================================================================

// Internal function that compresses in-memory archive data.
// Populates stats->prepareSec, computeSec, writeSec, outputBytes; leaves
// readSec/inputBytes/total/throughput/ratio for the caller to fill.
static void compressGPUBatchedFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback rawCallback = nullptr,
                                         CompressionStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    auto callback = makeThrottledCallback(rawCallback);
    const bool verbose = isVerbose();

    if (verbose) {
        std::cout << "Using GPU batched compression (" << algoToString(algo) << ")...\n";
    }
    
    size_t totalSize = archiveData.size();
    if (verbose) {
        std::cout << "Archive size: " << totalSize << " bytes\n";
    }
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior (continue with existing code)
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    auto prepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Calculate chunks
    size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (verbose) {
        std::cout << "Chunks: " << chunk_count << "\n";
    }
    
    // Report total blocks and preparing stage if callback provided
    // (Reading phase is complete at this point, reported by createArchive functions)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = 0;
        info.currentBlock = 0;
        info.currentBlockSize = 0;
        info.overallProgress = 0.25f;  // Reading complete (0-25%), now preparing (25%)
        info.currentBlockProgress = 0.0f;
        info.throughputMBps = 0.0;
        info.stage = "preparing";
        callback(info);
    }
    
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
    
    // Report compressing stage (25% allocated for reading/preparing)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = 0;
        info.currentBlock = 0;
        info.currentBlockSize = CHUNK_SIZE;
        info.overallProgress = 0.25f;  // 25% for reading/preparing
        info.currentBlockProgress = 0.0f;
        info.throughputMBps = 0.0;
        info.stage = "compressing";
        callback(info);
    }
    
    auto computeStart = clock::now();
    if (stats) {
        stats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    }

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
    auto computeEnd = clock::now();
    
    // Get output sizes
    std::vector<size_t> h_output_sizes(chunk_count);
    CUDA_CHECK(cudaMemcpy(h_output_sizes.data(), d_output_sizes, sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost));
    
    // Calculate total size
    size_t totalCompSize = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        totalCompSize += h_output_sizes[i];
    }
    
    // Calculate throughput and duration
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    double throughputMBps = (inputSize / (1024.0 * 1024.0)) / duration;
    if (stats) {
        stats->computeSec += duration;
    }
    
    if (verbose) {
        std::cout << "Compressed size: " << totalCompSize << " bytes\n";
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / totalCompSize << "x\n";
        std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    
    // Single "compression complete" callback before writing
    // (replaces the previous per-chunk loop that fired thousands of cross-thread signals).
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = static_cast<int>(chunk_count);
        info.currentBlock = static_cast<int>(chunk_count > 0 ? chunk_count - 1 : 0);
        info.currentBlockSize = chunk_count > 0 ? h_input_sizes.back() : 0;
        info.overallProgress = 0.75f;
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = throughputMBps;
        info.stage = "compressing";
        callback(info);
    }
    
    // Create output with metadata
    auto writeStart = clock::now();
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
    
    size_t totalSizeWithMeta = outputData.size();
    if (verbose) {
        std::cout << "Total size with metadata: " << totalSizeWithMeta << " bytes\n";
    }
    
    // Report writing stage (75% - start of writing phase)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = static_cast<int>(chunk_count);
        info.currentBlock = static_cast<int>(chunk_count - 1);
        info.currentBlockSize = 0;
        info.overallProgress = 0.75f;  // Writing starts at 75%
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = throughputMBps;
        info.stage = "writing";
        callback(info);
    }
    
    writeFile(outputFile, outputData.data(), outputData.size(), callback);
    auto writeEnd = clock::now();
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(writeEnd - writeStart).count();
        stats->outputBytes = outputData.size();
    }
    
    // Report completion (100%)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = static_cast<int>(chunk_count);
        info.currentBlock = static_cast<int>(chunk_count > 0 ? chunk_count - 1 : 0);
        info.currentBlockSize = 0;
        info.overallProgress = 1.0f;  // 100% complete
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = throughputMBps;
        info.stage = "complete";
        callback(info);
    }
    
    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_input_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_data);
    cudaFree(d_output_ptrs);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
    cudaStreamDestroy(stream);
        return;
    }
    
    // Multi-volume compression
    if (verbose) {
        std::cout << "\nCompressing " << volumes.size() << " volume(s)...\n";
    }
    
    // Create volume manifest
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumes.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalSize;
    manifest.reserved = 0;
    
    // Prepare metadata and compressed data storage
    std::vector<VolumeMetadata> volumeMetadata(volumes.size());
    std::vector<std::vector<uint8_t>> volumeCompressedData(volumes.size());
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    // Create CUDA stream for compression
    auto mvPrepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Compress each volume
    for (size_t vol_idx = 0; vol_idx < volumes.size(); vol_idx++) {
        if (verbose) {
            std::cout << "\r  Processing volume " << (vol_idx + 1) << "/" << volumes.size() << "..." << std::flush;
        }
        
        // Report compressing stage for this volume (scale to 25%-75% range)
        if (callback) {
            float volumeProgress = (float)vol_idx / volumes.size();
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(volumes.size());
            info.completedBlocks = static_cast<int>(vol_idx);
            info.currentBlock = static_cast<int>(vol_idx);
            info.currentBlockSize = volumes[vol_idx].size();
            info.overallProgress = 0.25f + (volumeProgress * 0.5f);  // 25% to 75%
            info.currentBlockProgress = 0.0f;
            info.throughputMBps = 0.0;
            info.stage = "compressing";
            callback(info);
        }
        
        size_t inputSize = volumes[vol_idx].size();
        std::vector<uint8_t>& inputData = volumes[vol_idx];
        
        // Calculate chunks for this volume
        size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
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
        
        auto computeStartV = clock::now();
        if (stats) {
            stats->prepareSec += std::chrono::duration<double>(computeStartV - mvPrepareStart).count();
        }

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
        auto computeEndV = clock::now();
        
        double duration = std::chrono::duration<double>(computeEndV - computeStartV).count();
        totalDuration += duration;
        if (stats) {
            stats->computeSec += duration;
        }
        // Reset prepare clock so the next volume's prepare phase is timed from here.
        mvPrepareStart = computeEndV;
        
        // Get output sizes
        std::vector<size_t> h_output_sizes(chunk_count);
        CUDA_CHECK(cudaMemcpy(h_output_sizes.data(), d_output_sizes, sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost));
        
        // Calculate total compressed size for this volume
        size_t volumeCompSize = 0;
        for (size_t i = 0; i < chunk_count; i++) {
            volumeCompSize += h_output_sizes[i];
        }
        
        // Create output with metadata for this volume
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
        outputData.resize(dataStart + volumeCompSize);
        size_t offset = 0;
        for (size_t i = 0; i < chunk_count; i++) {
            CUDA_CHECK(cudaMemcpy(outputData.data() + dataStart + offset, h_output_ptrs[i], h_output_sizes[i], cudaMemcpyDeviceToHost));
            offset += h_output_sizes[i];
        }
        
        // Store compressed data for this volume
        volumeCompressedData[vol_idx] = outputData;
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = vol_idx + 1;
        meta.compressedSize = outputData.size();
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = inputSize;
        volumeMetadata[vol_idx] = meta;
        
        uncompressedOffset += inputSize;
        totalCompressedSize += outputData.size();
        
        // Report progress after this volume completes (scale to 25%-75% range)
        // Throttled by makeThrottledCallback wrapper, so safe to call.
        if (callback) {
            float volumeProgress = (float)(vol_idx + 1) / volumes.size();  // Completed volumes
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(volumes.size());
            info.completedBlocks = static_cast<int>(vol_idx + 1);
            info.currentBlock = static_cast<int>(vol_idx);
            info.currentBlockSize = volumes[vol_idx].size();
            info.overallProgress = 0.25f + (volumeProgress * 0.5f);  // 25% to 75%
            info.currentBlockProgress = 1.0f;
            double throughputMBps = (inputSize / (1024.0 * 1024.0)) / duration;
            info.throughputMBps = throughputMBps;
            info.stage = "compressing";
            callback(info);
        }
        
        // Cleanup GPU memory for this volume
        cudaFree(d_input_data);
        cudaFree(d_input_ptrs);
        cudaFree(d_input_sizes);
        cudaFree(d_output_data);
        cudaFree(d_output_ptrs);
        cudaFree(d_output_sizes);
        cudaFree(d_temp);
    }
    
    if (verbose) {
        std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!\n";
    }
    
    // Destroy CUDA stream
    cudaStreamDestroy(stream);
    
    // Report writing stage starting (75%)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(volumes.size());
        info.completedBlocks = static_cast<int>(volumes.size());
        info.currentBlock = static_cast<int>(volumes.size() - 1);
        info.currentBlockSize = 0;
        info.overallProgress = 0.75f;  // Writing starts at 75%
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = 0.0;
        info.stage = "writing";
        callback(info);
    }
    
    auto writeStartMv = clock::now();
    // Write volume files
    // First volume gets manifest + metadata + compressed data
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    std::vector<uint8_t> firstVolumeOutput;
    
    // Add manifest header
    const uint8_t* manifestBytes = reinterpret_cast<const uint8_t*>(&manifest);
    firstVolumeOutput.insert(firstVolumeOutput.end(), manifestBytes, manifestBytes + sizeof(VolumeManifest));
    
    // Add volume metadata array
    const uint8_t* metadataBytes = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    firstVolumeOutput.insert(firstVolumeOutput.end(), metadataBytes, 
                            metadataBytes + sizeof(VolumeMetadata) * volumeMetadata.size());
    
    // Add first volume compressed data
    firstVolumeOutput.insert(firstVolumeOutput.end(), 
                            volumeCompressedData[0].begin(), volumeCompressedData[0].end());
    
    writeFile(firstVolumeFile, firstVolumeOutput.data(), firstVolumeOutput.size(), callback);
    
    // Update first volume metadata with actual size (including manifest and metadata)
    volumeMetadata[0].compressedSize = firstVolumeOutput.size();
    totalCompressedSize = totalCompressedSize - volumeCompressedData[0].size() + firstVolumeOutput.size();
    
    // Write remaining volumes (just compressed data)
    for (size_t i = 1; i < volumes.size(); i++) {
        std::string volumeFile = generateVolumeFilename(outputFile, i + 1);
        writeFile(volumeFile, volumeCompressedData[i].data(), volumeCompressedData[i].size(), callback);
    }
    auto writeEndMv = clock::now();
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(writeEndMv - writeStartMv).count();
        stats->outputBytes = totalCompressedSize;
    }
    
    // Report completion (100%)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(volumes.size());
        info.completedBlocks = static_cast<int>(volumes.size());
        info.currentBlock = static_cast<int>(volumes.size() - 1);
        info.currentBlockSize = 0;
        info.overallProgress = 1.0f;  // 100% complete
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = 0.0;
        info.stage = "complete";
        callback(info);
    }
    
    // Print summary
    std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
    std::cout << "Volumes created: " << volumes.size() << std::endl;
    std::cout << "Total uncompressed: " << (totalSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Total compressed: " << (totalCompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Overall ratio: " << std::fixed << std::setprecision(2) 
              << (double)totalSize / totalCompressedSize << "x" << std::endl;
    std::cout << "Total time: " << totalDuration << "s (" 
              << (totalSize / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
}

// ============================================================================
// Streaming volume compression (Phase 3)
// ============================================================================
//
// One reusable host fillBuffer of capacity = maxVolumeSize is filled directly
// from disk (mmap'd via readFileInto) one file at a time. When it reaches
// maxVolumeSize the volume is GPU-compressed and either:
//   - buffered in volume1Buffered (volume index 0) so we can prepend the
//     manifest+metadata at the end, or
//   - written straight to disk (volume index >= 1).
//
// This eliminates two full archive-sized memcpys (the realloc cascade and the
// splitIntoVolumes copy) and drops peak RAM from ~12 GB to ~3.5 GB on a
// 4.7 GB / 2 x 2.5 GB-volume workload. The on-disk volume layout is identical
// to the in-memory pipeline, so all decompression code is unchanged.

// Compress one volume's bytes on the GPU into a wrapped batched-format output.
// Allocates and frees device buffers per call; that's fine because volume
// sizes are large (>>cudaMalloc overhead). `stream` is reused across volumes
// by the caller. Updates stats->prepareSec and stats->computeSec if non-null.
static void compressVolumeBatched(AlgoType algo,
                                  const uint8_t* inputData,
                                  size_t inputSize,
                                  cudaStream_t stream,
                                  std::vector<uint8_t>& outputData,
                                  CompressionStats* stats) {
    using clock = std::chrono::steady_clock;
    auto prepareStart = clock::now();

    size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Per-chunk host arrays
    std::vector<size_t> h_input_sizes(chunk_count);
    std::vector<void*> h_input_ptrs(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) {
        h_input_sizes[i] = std::min(CHUNK_SIZE, inputSize - i * CHUNK_SIZE);
    }

    // Device input
    uint8_t* d_input_data;
    CUDA_CHECK(cudaMalloc(&d_input_data, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input_data, inputData, inputSize, cudaMemcpyHostToDevice, stream));

    void** d_input_ptrs;
    size_t* d_input_sizes;
    CUDA_CHECK(cudaMalloc(&d_input_ptrs, sizeof(void*) * chunk_count));
    CUDA_CHECK(cudaMalloc(&d_input_sizes, sizeof(size_t) * chunk_count));
    for (size_t i = 0; i < chunk_count; i++) {
        h_input_ptrs[i] = d_input_data + i * CHUNK_SIZE;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
                               sizeof(void*) * chunk_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
                               sizeof(size_t) * chunk_count, cudaMemcpyHostToDevice, stream));

    // Algo-specific temp + max output sizing
    size_t temp_bytes = 0;
    size_t max_out_bytes = 0;
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
    CUDA_CHECK(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
                               sizeof(void*) * chunk_count, cudaMemcpyHostToDevice, stream));

    auto computeStart = clock::now();
    if (stats) {
        stats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    }

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
    auto computeEnd = clock::now();
    if (stats) {
        stats->computeSec += std::chrono::duration<double>(computeEnd - computeStart).count();
    }

    // Read back per-chunk sizes and assemble outputData = BatchedHeader + chunkSizes64[] + chunks.
    std::vector<size_t> h_output_sizes(chunk_count);
    CUDA_CHECK(cudaMemcpy(h_output_sizes.data(), d_output_sizes,
                          sizeof(size_t) * chunk_count, cudaMemcpyDeviceToHost));
    size_t volumeCompSize = 0;
    for (size_t i = 0; i < chunk_count; i++) volumeCompSize += h_output_sizes[i];

    outputData.clear();
    outputData.reserve(sizeof(BatchedHeader) + sizeof(uint64_t) * chunk_count + volumeCompSize);

    BatchedHeader header;
    header.magic = BATCHED_MAGIC;
    header.version = BATCHED_VERSION;
    header.uncompressedSize = inputSize;
    header.chunkCount = static_cast<uint32_t>(chunk_count);
    header.chunkSize = CHUNK_SIZE;
    header.algorithm = static_cast<uint32_t>(algo);
    header.reserved = 0;
    const uint8_t* hb = reinterpret_cast<const uint8_t*>(&header);
    outputData.insert(outputData.end(), hb, hb + sizeof(BatchedHeader));

    std::vector<uint64_t> chunkSizes64(chunk_count);
    for (size_t i = 0; i < chunk_count; i++) chunkSizes64[i] = h_output_sizes[i];
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(chunkSizes64.data());
    outputData.insert(outputData.end(), sb, sb + sizeof(uint64_t) * chunk_count);

    size_t dataStart = outputData.size();
    outputData.resize(dataStart + volumeCompSize);
    size_t off = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        CUDA_CHECK(cudaMemcpy(outputData.data() + dataStart + off,
                              h_output_ptrs[i], h_output_sizes[i],
                              cudaMemcpyDeviceToHost));
        off += h_output_sizes[i];
    }

    cudaFree(d_input_data);
    cudaFree(d_input_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_data);
    cudaFree(d_output_ptrs);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
}

// True streaming compressor: walks `entries` once, fills a single host buffer
// of capacity maxVolumeSize, flushes (compress + write) when full. Volume 1's
// compressed bytes are kept in RAM until all volumes are done so we can
// prepend the volume manifest + metadata table (which we don't know until
// every volume's compressedSize is recorded). Volumes 2..N stream straight to
// disk with no intermediate buffering.
static void compressGPUBatchedStreaming(AlgoType algo,
                                        const std::vector<ArchiveEntry>& entries,
                                        const std::string& outputFile,
                                        uint64_t maxVolumeSize,
                                        ProgressCallback rawCallback,
                                        CompressionStats* stats) {
    using clock = std::chrono::steady_clock;
    auto callback = makeThrottledCallback(rawCallback);
    const bool verbose = isVerbose();
    auto opStart = clock::now();

    // Total uncompressed archive size as it would appear if we built the
    // whole thing in RAM. Needed for the VolumeManifest field and as the
    // "input size" reported by stats.
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    uint64_t totalFileBytes = 0;
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
        totalFileBytes += e.fileSize;
    }

    if (verbose) {
        std::cout << "Using GPU batched compression (" << algoToString(algo)
                  << ") [streaming]...\n";
        std::cout << "Archive size: " << totalArchiveSize << " bytes\n";
    }

    if (stats) stats->inputBytes = totalArchiveSize;

    // Reusable host buffer sized to one volume. capacity is set once; only
    // size() varies as we append/flush. resize(0) preserves capacity.
    std::vector<uint8_t> fillBuffer;
    fillBuffer.reserve(static_cast<size_t>(maxVolumeSize));

    std::vector<uint8_t> volume1Buffered;     // first volume waits for manifest prepend
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t totalCompressedBytes = 0;
    uint64_t uncompressedOffset = 0;
    size_t volumeIndex = 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto flushVolume = [&]() {
        std::vector<uint8_t> compressed;
        compressVolumeBatched(algo, fillBuffer.data(), fillBuffer.size(),
                              stream, compressed, stats);

        VolumeMetadata meta;
        meta.volumeIndex = volumeIndex + 1;
        meta.compressedSize = compressed.size();  // volume 1 patched after manifest prepend
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = fillBuffer.size();
        volumeMetadata.push_back(meta);

        uncompressedOffset += fillBuffer.size();
        totalCompressedBytes += compressed.size();

        if (volumeIndex == 0) {
            volume1Buffered = std::move(compressed);
        } else {
            auto writeStart = clock::now();
            std::string filename = generateVolumeFilename(outputFile, volumeIndex + 1);
            writeFile(filename, compressed.data(), compressed.size());
            if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        }

        if (verbose) {
            std::cout << "  Volume " << (volumeIndex + 1)
                      << " flushed (" << fillBuffer.size() << " B uncompressed -> "
                      << meta.compressedSize << " B compressed)\n";
        }

        fillBuffer.resize(0);
        volumeIndex++;
    };

    // Append `n` bytes from `src` to fillBuffer, flushing when full.
    auto appendBytes = [&](const uint8_t* src, uint64_t n) {
        while (n > 0) {
            uint64_t avail = maxVolumeSize - fillBuffer.size();
            if (avail == 0) {
                flushVolume();
                avail = maxVolumeSize;
            }
            uint64_t take = std::min(avail, n);
            size_t off = fillBuffer.size();
            fillBuffer.resize(off + static_cast<size_t>(take));
            std::memcpy(fillBuffer.data() + off, src, static_cast<size_t>(take));
            src += take;
            n -= take;
        }
    };

    // 1. ArchiveHeader at the start of the very first volume.
    ArchiveHeader hdr;
    hdr.magic = ARCHIVE_MAGIC;
    hdr.version = ARCHIVE_VERSION;
    hdr.fileCount = static_cast<uint32_t>(entries.size());
    hdr.reserved = 0;
    appendBytes(reinterpret_cast<const uint8_t*>(&hdr), sizeof(ArchiveHeader));

    // 2. For each entry: FileEntry + path bytes + file bytes.
    uint64_t processedFileBytes = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];

        FileEntry fe;
        fe.pathLength = static_cast<uint32_t>(e.relativePath.size());
        fe.fileSize = e.fileSize;
        appendBytes(reinterpret_cast<const uint8_t*>(&fe), sizeof(FileEntry));
        appendBytes(reinterpret_cast<const uint8_t*>(e.relativePath.data()),
                    e.relativePath.size());

        if (e.fileSize > 0) {
            uint64_t avail = maxVolumeSize - fillBuffer.size();
            if (e.fileSize <= avail) {
                // Common case: whole file fits in current volume - one mmap'd read.
                size_t writeOff = fillBuffer.size();
                fillBuffer.resize(writeOff + static_cast<size_t>(e.fileSize));
                readFileInto(e.filePath, fillBuffer.data() + writeOff, e.fileSize);
            } else {
                // File spans volumes (mid-file split allowed). Open once,
                // stream in chunks bounded by remaining-volume-capacity.
                std::ifstream f(e.filePath, std::ios::binary);
                if (!f.is_open()) {
                    throw std::runtime_error("Failed to open input file: " + e.filePath.string());
                }
                uint64_t remaining = e.fileSize;
                while (remaining > 0) {
                    avail = maxVolumeSize - fillBuffer.size();
                    if (avail == 0) {
                        flushVolume();
                        avail = maxVolumeSize;
                    }
                    uint64_t take = std::min(avail, remaining);
                    size_t writeOff = fillBuffer.size();
                    fillBuffer.resize(writeOff + static_cast<size_t>(take));
                    if (!f.read(reinterpret_cast<char*>(fillBuffer.data() + writeOff),
                                static_cast<std::streamsize>(take))) {
                        throw std::runtime_error("Failed to read file: " + e.filePath.string());
                    }
                    remaining -= take;
                }
            }
        }

        processedFileBytes += e.fileSize;
        if (verbose) {
            std::cout << "  Adding: " << e.relativePath
                      << " (" << e.fileSize << " bytes)\n";
        }

        // Throttled progress: scale to 0-75% range while we're filling+compressing.
        if (callback && totalFileBytes > 0) {
            float p = static_cast<float>(processedFileBytes) / totalFileBytes;
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(entries.size());
            info.completedBlocks = static_cast<int>(i + 1);
            info.currentBlock = static_cast<int>(i);
            info.currentBlockSize = e.fileSize;
            info.overallProgress = p * 0.75f;
            info.currentBlockProgress = 1.0f;
            info.throughputMBps = 0.0;
            info.stage = "compressing";
            callback(info);
        }
    }

    // 3. Flush final partial volume (if any).
    if (!fillBuffer.empty()) {
        flushVolume();
    }
    cudaStreamDestroy(stream);

    // 4. Build manifest, prepend it + the metadata array to volume 1's buffered
    //    compressed bytes, and write volume 1 to disk last.
    auto writeStart = clock::now();
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumeMetadata.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalArchiveSize;
    manifest.reserved = 0;

    std::vector<uint8_t> volume1OnDisk;
    volume1OnDisk.reserve(sizeof(VolumeManifest)
                          + sizeof(VolumeMetadata) * volumeMetadata.size()
                          + volume1Buffered.size());
    const uint8_t* mb = reinterpret_cast<const uint8_t*>(&manifest);
    volume1OnDisk.insert(volume1OnDisk.end(), mb, mb + sizeof(VolumeManifest));
    const uint8_t* vmb = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    volume1OnDisk.insert(volume1OnDisk.end(), vmb,
                         vmb + sizeof(VolumeMetadata) * volumeMetadata.size());
    volume1OnDisk.insert(volume1OnDisk.end(),
                         volume1Buffered.begin(), volume1Buffered.end());

    // Patch totalCompressedBytes for the larger volume 1.
    totalCompressedBytes = totalCompressedBytes - volume1Buffered.size() + volume1OnDisk.size();

    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    writeFile(firstVolumeFile, volume1OnDisk.data(), volume1OnDisk.size());
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        stats->outputBytes = totalCompressedBytes;
    }

    // Final 100% progress notification.
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(volumeMetadata.size());
        info.completedBlocks = static_cast<int>(volumeMetadata.size());
        info.currentBlock = static_cast<int>(volumeMetadata.size() > 0 ? volumeMetadata.size() - 1 : 0);
        info.currentBlockSize = 0;
        info.overallProgress = 1.0f;
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = 0.0;
        info.stage = "complete";
        callback(info);
    }

    // Always-on result summary (matches the in-memory pipeline's output).
    double totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
    std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
    std::cout << "Volumes created: " << volumeMetadata.size() << std::endl;
    std::cout << "Total uncompressed: " << (totalArchiveSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Total compressed: " << (totalCompressedBytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    if (totalCompressedBytes > 0) {
        std::cout << "Overall ratio: " << std::fixed << std::setprecision(2)
                  << (double)totalArchiveSize / totalCompressedBytes << "x" << std::endl;
    }
    std::cout << "Total time: " << totalSec << "s ("
              << (totalArchiveSize / (1024.0 * 1024.0 * 1024.0)) / std::max(totalSec, 1e-9) << " GB/s)" << std::endl;
}

// Should we route this compression through the streaming pipeline?
// Streaming is only worth it for true multi-volume work; for everything else
// the in-memory path (createArchive* + compressGPUBatchedFromBuffer) is fine
// and already benefits from Phase 1's reserve+direct-read changes.
static inline bool shouldStreamMultiVolume(uint64_t totalArchiveSize, uint64_t maxVolumeSize) {
    return maxVolumeSize > 0
        && maxVolumeSize != UINT64_MAX
        && totalArchiveSize > maxVolumeSize;
}

// Public wrapper for single file/folder compression
void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    // Walk inputs once to decide single-volume vs streaming. collectArchiveEntries
    // is cheap (stat + relative-path build per file), so it's fine to call even
    // if we end up using the in-memory path next.
    auto entries = collectArchiveEntries(inputPath);
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
    }

    if (shouldStreamMultiVolume(totalArchiveSize, maxVolumeSize)) {
        compressGPUBatchedStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);
    } else {
        auto readStart = clock::now();
        std::vector<uint8_t> archiveData;
        if (isDirectory(inputPath)) {
            archiveData = createArchiveFromFolder(inputPath, throttled);
        } else {
            archiveData = createArchiveFromFile(inputPath, throttled);
        }
        auto readEnd = clock::now();
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(readEnd - readStart).count();
            outStats->inputBytes = archiveData.size();
        }

        compressGPUBatchedFromBuffer(algo, archiveData, outputFile, maxVolumeSize, callback, outStats);
    }

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void compressGPUBatchedFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    if (isVerbose()) {
        std::cout << "Compressing file list (" << filePaths.size() << " files)...\n";
    }

    auto entries = collectArchiveEntriesFromList(filePaths);
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
    }

    if (shouldStreamMultiVolume(totalArchiveSize, maxVolumeSize)) {
        compressGPUBatchedStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);
    } else {
        auto readStart = clock::now();
        std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, throttled);
        auto readEnd = clock::now();
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(readEnd - readStart).count();
            outStats->inputBytes = archiveData.size();
        }

        compressGPUBatchedFromBuffer(algo, archiveData, outputFile, maxVolumeSize, callback, outStats);
    }

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

// ============================================================================
// GPU Batched Decompression
// ============================================================================

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    (void)makeThrottledCallback(callback); // throttle reserved for future per-block callbacks

    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto readStart = clock::now();
        auto firstVolumeData = readFile(volumeFiles[0]);
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        }
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        const bool verbose = isVerbose();
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
            if (detectedAlgo != ALGO_UNKNOWN) {
                algo = detectedAlgo;
                if (verbose) {
                    std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << "\n";
                }
            }
            
            if (verbose) {
                std::cout << "Decompressing (" << algoToString(algo) << ")...\n";
            }
            
            auto computeStart = clock::now();
            auto archiveData = decompressBatchedFormatCPU(algo, firstVolumeData);
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            size_t decompSize = archiveData.size();
            if (verbose) {
                std::cout << "Decompressed size: " << decompSize << " bytes\n";
                std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
            }

            if (outStats) {
                outStats->computeSec += duration;
                outStats->inputBytes = decompSize;        // uncompressed payload
                outStats->outputBytes = firstVolumeData.size();
            }
            
            auto writeStart = clock::now();
            extractArchive(archiveData, outputPath);
            if (outStats) {
                outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
                outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
                finalizeStats(*outStats);
                std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
            }
            return;
        }
        
        // Multi-volume archive
        if (verbose) {
            std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)\n";
        }
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            std::cout << "Falling back to CPU decompression..." << std::endl;
            decompressCPU(algo, inputFile, outputPath, callback, outStats);
            return;
        }
        
        if (verbose) {
            std::cout << "Using GPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")...\n";
        }
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes (using CPU for batched format since it's easier)
        std::vector<uint8_t> fullArchive;
        fullArchive.reserve(manifest.totalUncompressedSize);
        double totalDuration = 0;
        uint64_t totalCompressedRead = firstVolumeData.size();
        
        if (verbose) {
            std::cout << "Decompressing " << volumeFiles.size() << " volume(s)...\n";
        }
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            if (verbose && ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1)) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }
            
            auto volReadStart = clock::now();
            auto volumeData = readFile(volumeFiles[i]);
            if (outStats && i > 0) {
                outStats->readSec += std::chrono::duration<double>(clock::now() - volReadStart).count();
                totalCompressedRead += volumeData.size();
            }
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
            auto computeStart = clock::now();
            auto decompressed = decompressBatchedFormatCPU(static_cast<AlgoType>(manifest.algorithm), volumeData);
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            totalDuration += duration;
            if (outStats) outStats->computeSec += duration;
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
        }
        
        if (verbose) {
            std::cout << "\n";
            std::cout << "\n=== Decompression Summary ===\n";
            std::cout << "Total decompressed: " << fullArchive.size() << " bytes\n";
            std::cout << "Total time: " << totalDuration << "s ("
                      << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)\n";
        }
        
        if (outStats) {
            outStats->inputBytes = fullArchive.size();
            outStats->outputBytes = totalCompressedRead;
        }

        auto writeStart = clock::now();
        extractArchive(fullArchive, outputPath);
        if (outStats) {
            outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
            outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
            finalizeStats(*outStats);
            std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
        }
        return;
    }
    
    // Single file (non-volume)
    const bool verbose = isVerbose();
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        if (verbose) {
            std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << "\n";
        }
    }
    
    if (verbose) {
        std::cout << "Decompressing (" << algoToString(algo) << ")...\n";
    }
    
    auto readStart = clock::now();
    auto compressedData = readFile(inputFile);
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->outputBytes = compressedData.size();
    }
    
    auto computeStart = clock::now();
    
    // Decompress (handles both batched and standard formats)
    auto archiveData = decompressBatchedFormatCPU(algo, compressedData);
    
    auto computeEnd = clock::now();
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    
    size_t decompSize = archiveData.size();
    if (verbose) {
        std::cout << "Decompressed size: " << decompSize << " bytes\n";
        std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (outStats) {
        outStats->computeSec += duration;
        outStats->inputBytes = decompSize;
    }
    
    auto writeStart = clock::now();
    extractArchive(archiveData, outputPath);
    if (outStats) {
        outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
    }
}

// ============================================================================
// GPU Manager API Compression
// ============================================================================

// Internal function that compresses in-memory archive data
static void compressGPUManagerFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData,
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         CompressionStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Using GPU manager compression (" << algoToString(algo) << ")...\n";
    }
    
    size_t totalSize = archiveData.size();
    if (verbose) {
        std::cout << "Archive size: " << totalSize << " bytes\n";
    }
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    auto prepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    std::shared_ptr<nvcomp::nvcompManagerBase> manager;
    
    if (algo == ALGO_GDEFLATE) {
        manager = std::make_shared<nvcomp::GdeflateManager>(
            CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    } else if (algo == ALGO_ANS) {
        manager = std::make_shared<nvcomp::ANSManager>(
            CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
    } else if (algo == ALGO_BITCOMP) {
        manager = std::make_shared<nvcomp::BitcompManager>(
            CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    }
    
    nvcomp::CompressionConfig comp_config = manager->configure_compression(inputSize);
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
    
    auto computeStart = clock::now();
    if (stats) stats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    
    manager->compress(d_input, d_output, comp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto computeEnd = clock::now();
    
    size_t compSize = manager->get_compressed_output_size(d_output);
    
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    if (verbose) {
        std::cout << "Compressed size: " << compSize << " bytes\n";
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x\n";
        std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (stats) stats->computeSec += duration;
    
    std::vector<uint8_t> outputData(compSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
    
    auto writeStart = clock::now();
    writeFile(outputFile, outputData.data(), outputData.size());
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        stats->outputBytes = outputData.size();
    }
    
    // Cleanup: destroy manager before stream (manager references stream)
    manager.reset();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
        return;
    }
    
    // Multi-volume compression
    if (verbose) {
        std::cout << "\nCompressing " << volumes.size() << " volume(s)...\n";
    }
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    for (size_t volIdx = 0; volIdx < volumes.size(); volIdx++) {
        if (verbose) {
            std::cout << "\r  Processing volume " << (volIdx + 1) << "/" << volumes.size() << "..." << std::flush;
        }
        
        std::vector<uint8_t>& inputData = volumes[volIdx];
        size_t inputSize = inputData.size();
        
        auto volPrepareStart = clock::now();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        uint8_t* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, inputSize));
        CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
        
        std::shared_ptr<nvcomp::nvcompManagerBase> manager;
        
        if (algo == ALGO_GDEFLATE) {
            manager = std::make_shared<nvcomp::GdeflateManager>(
                CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
        } else if (algo == ALGO_ANS) {
            manager = std::make_shared<nvcomp::ANSManager>(
                CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
        } else if (algo == ALGO_BITCOMP) {
            manager = std::make_shared<nvcomp::BitcompManager>(
                CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
        }
        
        nvcomp::CompressionConfig comp_config = manager->configure_compression(inputSize);
        
        uint8_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
        
        auto computeStartV = clock::now();
        if (stats) stats->prepareSec += std::chrono::duration<double>(computeStartV - volPrepareStart).count();
        
        manager->compress(d_input, d_output, comp_config);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto computeEndV = clock::now();
        
        size_t compSize = manager->get_compressed_output_size(d_output);
        
        double duration = std::chrono::duration<double>(computeEndV - computeStartV).count();
        totalDuration += duration;
        if (stats) stats->computeSec += duration;
        
        std::vector<uint8_t> outputData(compSize);
        CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = volIdx + 1;
        meta.compressedSize = compSize;
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = inputSize;
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += inputSize;
        totalCompressedSize += compSize;
        
        // Write volume file
        auto volWriteStart = clock::now();
        std::string volumeFile = generateVolumeFilename(outputFile, volIdx + 1);
        writeFile(volumeFile, outputData.data(), outputData.size());
        if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - volWriteStart).count();
        
        // Cleanup: destroy manager before stream (manager references stream)
        manager.reset();
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    if (verbose) {
        std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!\n";
    }
    
    // Create and prepend manifest to first volume
    
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumes.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalSize;
    manifest.reserved = 0;
    
    // Read first volume
    auto fixupReadStart = clock::now();
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    auto firstVolumeData = readFile(firstVolumeFile);
    if (stats) stats->readSec += std::chrono::duration<double>(clock::now() - fixupReadStart).count();
    
    // Create new first volume with manifest
    std::vector<uint8_t> newFirstVolume;
    
    // Add manifest header
    const uint8_t* manifestBytes = reinterpret_cast<const uint8_t*>(&manifest);
    newFirstVolume.insert(newFirstVolume.end(), manifestBytes, manifestBytes + sizeof(VolumeManifest));
    
    // Add volume metadata array
    const uint8_t* metadataBytes = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    newFirstVolume.insert(newFirstVolume.end(), metadataBytes, 
                         metadataBytes + sizeof(VolumeMetadata) * volumeMetadata.size());
    
    // Add original compressed data
    newFirstVolume.insert(newFirstVolume.end(), firstVolumeData.begin(), firstVolumeData.end());
    
    // Write updated first volume
    auto fixupWriteStart = clock::now();
    writeFile(firstVolumeFile, newFirstVolume.data(), newFirstVolume.size());
    if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - fixupWriteStart).count();
    
    // Update metadata for first volume
    volumeMetadata[0].compressedSize = newFirstVolume.size();
    totalCompressedSize = totalCompressedSize - firstVolumeData.size() + newFirstVolume.size();
    
    if (stats) stats->outputBytes = totalCompressedSize;

    std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
    std::cout << "Volumes created: " << volumes.size() << std::endl;
    std::cout << "Total uncompressed: " << (totalSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Total compressed: " << (totalCompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Overall ratio: " << std::fixed << std::setprecision(2) 
              << (double)totalSize / totalCompressedSize << "x" << std::endl;
    std::cout << "Total time: " << totalDuration << "s (" 
              << (totalSize / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
}

// Public wrapper for single file/folder compression
void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    auto readStart = clock::now();
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath, throttled);
    } else {
        archiveData = createArchiveFromFile(inputPath, throttled);
    }
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->inputBytes = archiveData.size();
    }
    
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void compressGPUManagerFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    if (isVerbose()) {
        std::cout << "Compressing file list (" << filePaths.size() << " files)...\n";
    }
    
    auto readStart = clock::now();
    std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, throttled);
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->inputBytes = archiveData.size();
    }
    
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

// ============================================================================
// GPU Manager API Decompression
// ============================================================================

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    (void)callback;

    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto manifestReadStart = clock::now();
        auto firstVolumeData = readFile(volumeFiles[0]);
        if (outStats) outStats->readSec += std::chrono::duration<double>(clock::now() - manifestReadStart).count();
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        const bool verbose = isVerbose();
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            if (verbose) {
                std::cout << "Using GPU manager decompression (auto-detect)...\n";
            }
            
            auto prepareStart = clock::now();
            size_t inputSize = firstVolumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, firstVolumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = nvcomp::create_manager(d_input, stream);
            nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            if (verbose) {
                std::cout << "Detected original size: " << outputSize << " bytes\n";
            }
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto computeStart = clock::now();
            if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            if (verbose) {
                std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
            }
            if (outStats) {
                outStats->computeSec += duration;
                outStats->inputBytes = outputSize;
                outStats->outputBytes = inputSize;
            }
            
            std::vector<uint8_t> archiveData(outputSize);
            CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            // Cleanup: destroy manager before stream (manager references stream)
            manager.reset();
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
            
            auto writeStart = clock::now();
            extractArchive(archiveData, outputPath);
            if (outStats) {
                outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
                outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
                finalizeStats(*outStats);
                std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
            }
            return;
        }
        
        // Multi-volume archive
        if (verbose) {
            std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)\n";
        }
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            throw std::runtime_error("Insufficient GPU memory for GPU-only algorithm. Cannot fall back to CPU.");
        }
        
        if (verbose) {
            std::cout << "Using GPU manager decompression...\n";
        }
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes
        std::vector<uint8_t> fullArchive;
        fullArchive.reserve(manifest.totalUncompressedSize);
        double totalDuration = 0;
        uint64_t totalCompressedRead = firstVolumeData.size();
        
        if (verbose) {
            std::cout << "Decompressing " << volumeFiles.size() << " volume(s)...\n";
        }
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            if (verbose && ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1)) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }
            
            auto volReadStart = clock::now();
            auto volumeData = readFile(volumeFiles[i]);
            if (outStats && i > 0) {
                outStats->readSec += std::chrono::duration<double>(clock::now() - volReadStart).count();
                totalCompressedRead += volumeData.size();
            }
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
            auto prepareStart = clock::now();
            size_t inputSize = volumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, volumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = nvcomp::create_manager(d_input, stream);
            nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto computeStart = clock::now();
            if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            totalDuration += duration;
            if (outStats) outStats->computeSec += duration;
            
            std::vector<uint8_t> decompressed(outputSize);
            CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
            
            // Cleanup: destroy manager before stream (manager references stream)
            manager.reset();
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
        }
        
        if (verbose) {
            std::cout << "\n";
            std::cout << "\n=== Decompression Summary ===\n";
            std::cout << "Total decompressed: " << fullArchive.size() << " bytes\n";
            std::cout << "Total time: " << totalDuration << "s ("
                      << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)\n";
        }
        
        if (outStats) {
            outStats->inputBytes = fullArchive.size();
            outStats->outputBytes = totalCompressedRead;
        }

        auto writeStart = clock::now();
        extractArchive(fullArchive, outputPath);
        if (outStats) {
            outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
            outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
            finalizeStats(*outStats);
            std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
        }
        return;
    }
    
    // Single file (non-volume)
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Using GPU manager decompression (auto-detect)...\n";
    }
    
    auto readStart = clock::now();
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->outputBytes = inputSize;
    }
    
    auto prepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    auto manager = nvcomp::create_manager(d_input, stream);
    
    nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
    size_t outputSize = decomp_config.decomp_data_size;
    if (verbose) {
        std::cout << "Detected original size: " << outputSize << " bytes\n";
    }
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    
    auto computeStart = clock::now();
    if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    
    manager->decompress(d_output, d_input, decomp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto computeEnd = clock::now();
    
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    if (verbose) {
        std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (outStats) {
        outStats->computeSec += duration;
        outStats->inputBytes = outputSize;
    }
    
    std::vector<uint8_t> archiveData(outputSize);
    CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Cleanup: destroy manager before stream (manager references stream)
    manager.reset();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    auto writeStart = clock::now();
    extractArchive(archiveData, outputPath);
    if (outStats) {
        outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
    }
}

// ============================================================================
// List Compressed Archive
// ============================================================================

void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        std::cout << "Multi-volume archive detected: " << volumeFiles.size() << " volume(s)" << std::endl;
        
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            throw std::runtime_error("Invalid volume manifest");
        }
        
        std::cout << "Algorithm: " << algoToString(static_cast<AlgoType>(manifest.algorithm)) << std::endl;
        std::cout << "Volume size: " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "Total uncompressed: " << (manifest.totalUncompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        std::cout << "\nVolume breakdown:" << std::endl;
        for (const auto& meta : volumeMetadata) {
            std::cout << "  Volume " << meta.volumeIndex << ": " 
                      << (meta.compressedSize / (1024.0 * 1024.0)) << " MB compressed, "
                      << (meta.uncompressedSize / (1024.0 * 1024.0)) << " MB uncompressed" << std::endl;
        }
        
        std::cout << "\nListing archive contents requires decompression..." << std::endl;
        return;
    }
    
    // Single file
    auto inputData = readFile(inputFile);
    
    // Try to decompress and list
    std::cout << "Listing contents of single file archive..." << std::endl;
    std::cout << "Full listing requires decompression implementation" << std::endl;
}

} // namespace nvcomp_core


