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
    
    std::cout << "Splitting archive into volumes (max " 
              << (maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB each)..." << std::endl;
    
    while (remaining > 0) {
        size_t volumeSize = std::min(static_cast<size_t>(maxVolumeSize), remaining);
        
        std::vector<uint8_t> volume(
            archiveData.begin() + offset,
            archiveData.begin() + offset + volumeSize
        );
        
        volumes.push_back(volume);
        
        // Show progress every 100 volumes or for the last volume
        if (volumeIndex % 100 == 0 || remaining <= volumeSize) {
            std::cout << "\r  Creating volumes... " << volumeIndex << " created" << std::flush;
        }
        
        offset += volumeSize;
        remaining -= volumeSize;
        volumeIndex++;
    }
    
    std::cout << "\r  Created " << volumes.size() << " volume(s)" << std::string(20, ' ') << std::endl;
    
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

// Internal function that compresses in-memory archive data
static void compressGPUBatchedFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData, 
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         ProgressCallback callback = nullptr) {
    std::cout << "Using GPU batched compression (" << algoToString(algo) << ")..." << std::endl;
    
    size_t totalSize = archiveData.size();
    std::cout << "Archive size: " << totalSize << " bytes" << std::endl;
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior (continue with existing code)
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Calculate chunks
    size_t chunk_count = (inputSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::cout << "Chunks: " << chunk_count << std::endl;
    
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
    
    // Calculate throughput and duration
    double duration = std::chrono::duration<double>(end - start).count();
    double throughputMBps = (inputSize / (1024.0 * 1024.0)) / duration;
    
    std::cout << "Compressed size: " << totalCompSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / totalCompSize << "x" << std::endl;
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Report all blocks as completed (scale to 25%-75% range)
    if (callback) {
        for (size_t i = 0; i < chunk_count; i++) {
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(chunk_count);
            info.completedBlocks = static_cast<int>(i + 1);
            info.currentBlock = static_cast<int>(i);
            info.currentBlockSize = h_input_sizes[i];
            // Scale compression progress to 25%-75% range
            float compressProgress = (float)(i + 1) / chunk_count;
            info.overallProgress = 0.25f + (compressProgress * 0.5f);  // 25% to 75%
            info.currentBlockProgress = 1.0f;
            info.throughputMBps = throughputMBps;
            info.stage = "compressing";
            callback(info);
        }
    }
    
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
    
    size_t totalSizeWithMeta = outputData.size();
    std::cout << "Total size with metadata: " << totalSizeWithMeta << " bytes" << std::endl;
    
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
    
    // Report completion (100%)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(chunk_count);
        info.completedBlocks = static_cast<int>(chunk_count);
        info.currentBlock = static_cast<int>(chunk_count - 1);
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
    std::cout << "\nCompressing " << volumes.size() << " volume(s)..." << std::endl;
    
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
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Compress each volume
    for (size_t vol_idx = 0; vol_idx < volumes.size(); vol_idx++) {
        // Show progress on single line
        std::cout << "\r  Processing volume " << (vol_idx + 1) << "/" << volumes.size() << "..." << std::flush;
        
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
        
        double duration = std::chrono::duration<double>(end - start).count();
        totalDuration += duration;
        
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
    
    std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!" << std::endl;
    
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

// Public wrapper for single file/folder compression
void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback) {
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath, callback);
    } else {
        archiveData = createArchiveFromFile(inputPath, callback);
    }
    
    // Call internal function with archive data
    compressGPUBatchedFromBuffer(algo, archiveData, outputFile, maxVolumeSize, callback);
}

void compressGPUBatchedFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback) {
    std::cout << "Compressing file list (" << filePaths.size() << " files)..." << std::endl;
    
    // Create archive from file list (in memory)
    std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, callback);
    
    // Compress directly from buffer - no temporary file needed!
    compressGPUBatchedFromBuffer(algo, archiveData, outputFile, maxVolumeSize, callback);
}

// ============================================================================
// GPU Batched Decompression
// ============================================================================

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath, ProgressCallback callback) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
            if (detectedAlgo != ALGO_UNKNOWN) {
                algo = detectedAlgo;
                std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
            }
            
            std::cout << "Decompressing (" << algoToString(algo) << ")..." << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto archiveData = decompressBatchedFormatCPU(algo, firstVolumeData);
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            size_t decompSize = archiveData.size();
            std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
            std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
            
            extractArchive(archiveData, outputPath);
            return;
        }
        
        // Multi-volume archive
        std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)" << std::endl;
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            std::cout << "Falling back to CPU decompression..." << std::endl;
            decompressCPU(algo, inputFile, outputPath, callback);
            return;
        }
        
        std::cout << "Using GPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")..." << std::endl;
        
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
        
        std::cout << "Decompressing " << volumeFiles.size() << " volume(s)..." << std::endl;
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            // Show progress every 100 volumes or for the last volume
            if ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }
            
            auto volumeData = readFile(volumeFiles[i]);
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto decompressed = decompressBatchedFormatCPU(static_cast<AlgoType>(manifest.algorithm), volumeData);
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            totalDuration += duration;
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
        }
        
        std::cout << std::endl; // New line after progress
        
        std::cout << "\n=== Decompression Summary ===" << std::endl;
        std::cout << "Total decompressed: " << fullArchive.size() << " bytes" << std::endl;
        std::cout << "Total time: " << totalDuration << "s (" 
                  << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
        
        // Extract archive
        extractArchive(fullArchive, outputPath);
        return;
    }
    
    // Single file (non-volume)
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << std::endl;
    }
    
    std::cout << "Decompressing (" << algoToString(algo) << ")..." << std::endl;
    
    auto compressedData = readFile(inputFile);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Decompress (handles both batched and standard formats)
    auto archiveData = decompressBatchedFormatCPU(algo, compressedData);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    size_t decompSize = archiveData.size();
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    // Extract archive
    extractArchive(archiveData, outputPath);
}

// ============================================================================
// GPU Manager API Compression
// ============================================================================

// Internal function that compresses in-memory archive data
static void compressGPUManagerFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData,
                                         const std::string& outputFile, uint64_t maxVolumeSize) {
    std::cout << "Using GPU manager compression (" << algoToString(algo) << ")..." << std::endl;
    
    size_t totalSize = archiveData.size();
    std::cout << "Archive size: " << totalSize << " bytes" << std::endl;
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
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
        return;
    }
    
    // Multi-volume compression
    std::cout << "\nCompressing " << volumes.size() << " volume(s)..." << std::endl;
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    for (size_t volIdx = 0; volIdx < volumes.size(); volIdx++) {
        // Show progress on single line
        std::cout << "\r  Processing volume " << (volIdx + 1) << "/" << volumes.size() << "..." << std::flush;
        
        std::vector<uint8_t>& inputData = volumes[volIdx];
        size_t inputSize = inputData.size();
        
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
        
        auto start = std::chrono::high_resolution_clock::now();
        
        manager->compress(d_input, d_output, comp_config);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        size_t compSize = manager->get_compressed_output_size(d_output);
        
        double duration = std::chrono::duration<double>(end - start).count();
        totalDuration += duration;
        
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
        std::string volumeFile = generateVolumeFilename(outputFile, volIdx + 1);
        writeFile(volumeFile, outputData.data(), outputData.size());
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!" << std::endl;
    
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
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    auto firstVolumeData = readFile(firstVolumeFile);
    
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
    writeFile(firstVolumeFile, newFirstVolume.data(), newFirstVolume.size());
    
    // Update metadata for first volume
    volumeMetadata[0].compressedSize = newFirstVolume.size();
    totalCompressedSize = totalCompressedSize - firstVolumeData.size() + newFirstVolume.size();
    
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
void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback) {
    // Create archive (handles both files and directories)
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath, callback);
    } else {
        archiveData = createArchiveFromFile(inputPath, callback);
    }
    
    // Call internal function with archive data
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize);
}

void compressGPUManagerFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback) {
    std::cout << "Compressing file list (" << filePaths.size() << " files)..." << std::endl;
    
    // Create archive from file list (in memory)
    std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, callback);
    
    // Compress directly from buffer - no temporary file needed!
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize);
}

// ============================================================================
// GPU Manager API Decompression
// ============================================================================

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath, ProgressCallback callback) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            std::cout << "Using GPU manager decompression (auto-detect)..." << std::endl;
            
            size_t inputSize = firstVolumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, firstVolumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = nvcomp::create_manager(d_input, stream);
            nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
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
            
            extractArchive(archiveData, outputPath);
            return;
        }
        
        // Multi-volume archive
        std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)" << std::endl;
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            throw std::runtime_error("Insufficient GPU memory for GPU-only algorithm. Cannot fall back to CPU.");
        }
        
        std::cout << "Using GPU manager decompression..." << std::endl;
        
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
        
        std::cout << "Decompressing " << volumeFiles.size() << " volume(s)..." << std::endl;
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            // Show progress every 100 volumes or for the last volume
            if ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }
            
            auto volumeData = readFile(volumeFiles[i]);
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
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
            
            auto start = std::chrono::high_resolution_clock::now();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double>(end - start).count();
            totalDuration += duration;
            
            std::vector<uint8_t> decompressed(outputSize);
            CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
        }
        
        std::cout << std::endl; // New line after progress
        
        std::cout << "\n=== Decompression Summary ===" << std::endl;
        std::cout << "Total decompressed: " << fullArchive.size() << " bytes" << std::endl;
        std::cout << "Total time: " << totalDuration << "s (" 
                  << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
        
        // Extract archive
        extractArchive(fullArchive, outputPath);
        return;
    }
    
    // Single file (non-volume)
    std::cout << "Using GPU manager decompression (auto-detect)..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    auto manager = nvcomp::create_manager(d_input, stream);
    
    nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
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


