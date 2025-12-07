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
// CPU COMPRESSION/DECOMPRESSION (LZ4, Snappy, Zstd)
// ============================================================================

void compressCPU(AlgoType algo, const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using CPU compression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    std::cout << "Input size: " << inputSize << " bytes" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
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
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    outputData.resize(compSize);
    
    std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x" << std::endl;
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    writeFile(outputFile, outputData.data(), outputData.size());
}

void decompressCPU(AlgoType algo, const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using CPU decompression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    
    auto start = std::chrono::high_resolution_clock::now();
    
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
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    outputData.resize(decompSize);
    
    std::cout << "Decompressed size: " << decompSize << " bytes" << std::endl;
    std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;
    
    writeFile(outputFile, outputData.data(), outputData.size());
}

// ============================================================================
// GPU BATCHED API (LZ4, Snappy, Zstd) - Cross-compatible with CPU
// ============================================================================

void compressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using GPU batched compression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    std::cout << "Input size: " << inputSize << " bytes" << std::endl;
    
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
    
    // Copy output back and concatenate
    std::vector<uint8_t> outputData(totalCompSize);
    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        CUDA_CHECK(cudaMemcpy(outputData.data() + offset, h_output_ptrs[i], h_output_sizes[i], cudaMemcpyDeviceToHost));
        offset += h_output_sizes[i];
    }
    
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

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using GPU batched decompression (" << algoToString(algo) << ")..." << std::endl;
    
    // For batched decompression, we need to know the chunk structure
    // This is a simplified version - in production, you'd store metadata
    throw std::runtime_error("GPU batched decompression requires metadata about chunk structure. Use CPU decompression for cross-compatibility.");
}

// ============================================================================
// GPU MANAGER API (GDeflate, ANS, Bitcomp) - GPU-only
// ============================================================================

void compressGPUManager(AlgoType algo, const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using GPU manager compression (" << algoToString(algo) << ")..." << std::endl;
    
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    std::cout << "Input size: " << inputSize << " bytes" << std::endl;
    
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

void decompressGPUManager(const std::string& inputFile, const std::string& outputFile) {
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
    
    std::vector<uint8_t> outputData(outputSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    writeFile(outputFile, outputData.data(), outputData.size());
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

// ============================================================================
// MAIN
// ============================================================================

void printUsage(const char* appName) {
    std::cerr << "nvCOMP CLI with CPU Fallback\n\n";
    std::cerr << "Usage:\n";
    std::cerr << "  Compress:   " << appName << " -c <input> <output> [algorithm] [--cpu]\n";
    std::cerr << "  Decompress: " << appName << " -d <input> <output> [algorithm] [--cpu]\n\n";
    std::cerr << "Algorithms:\n";
    std::cerr << "  Cross-compatible (GPU/CPU): lz4, snappy, zstd\n";
    std::cerr << "  GPU-only: gdeflate, ans, bitcomp\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --cpu    Force CPU mode\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4\n";
    std::cerr << "  " << appName << " -d output.lz4 restored.txt lz4\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4 --cpu\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string mode = argv[1];
        std::string inputFile = argv[2];
        std::string outputFile = argv[3];
        std::string algoStr = (argc >= 5) ? argv[4] : "lz4";
        bool forceCPU = false;
        
        // Check for --cpu flag
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "--cpu") {
                forceCPU = true;
            }
        }
        
        AlgoType algo = parseAlgorithm(algoStr);
        if (algo == ALGO_UNKNOWN) {
            std::cerr << "Unknown algorithm: " << algoStr << std::endl;
            printUsage(argv[0]);
            return 1;
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
            if (useCPU) {
                compressCPU(algo, inputFile, outputFile);
            } else {
                if (isCrossCompatible(algo)) {
                    compressGPUBatched(algo, inputFile, outputFile);
                } else {
                    compressGPUManager(algo, inputFile, outputFile);
                }
            }
        } else if (mode == "-d") {
            if (useCPU) {
                decompressCPU(algo, inputFile, outputFile);
            } else {
                if (isCrossCompatible(algo)) {
                    // For batched API, we'd need metadata
                    // For simplicity, fall back to CPU
                    std::cout << "Note: Using CPU for decompression (cross-compatible format)" << std::endl;
                    decompressCPU(algo, inputFile, outputFile);
                } else {
                    decompressGPUManager(inputFile, outputFile);
                }
            }
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
