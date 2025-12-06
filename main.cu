/*
 * nvCOMP CLI Application
 * Usage:
 *   Compress:   nvcomp_cli -c <input_file> <output_file> [algorithm]
 *   Decompress: nvcomp_cli -d <input_file> <output_file>
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include <cuda_runtime.h>

// nvCOMP High-level API headers
#include <nvcomp.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Helper to read file into host vector
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

// Helper to write host vector to file
void writeFile(const std::string& filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data), size);
}

// Compression function
void compressFile(const std::string& inputFile, const std::string& outputFile, const std::string& algo) {
    std::cout << "Reading input file: " << inputFile << "..." << std::endl;
    std::vector<uint8_t> inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    std::cout << "Input size: " << inputSize << " bytes" << std::endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate input buffer on GPU
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));

    // Create Manager based on algorithm
    std::shared_ptr<nvcomp::nvcompManagerBase> manager;
    size_t chunkSize = 1 << 16; // 64KB default chunk size

    if (algo == "lz4") {
        manager = std::make_shared<nvcomp::LZ4Manager>(chunkSize, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, stream);
    } else if (algo == "gdeflate") {
        manager = std::make_shared<nvcomp::GdeflateManager>(chunkSize, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    } else if (algo == "snappy") {
        manager = std::make_shared<nvcomp::SnappyManager>(chunkSize, nvcompBatchedSnappyCompressDefaultOpts, nvcompBatchedSnappyDecompressDefaultOpts, stream);
    } else if (algo == "ans") {
        manager = std::make_shared<nvcomp::ANSManager>(chunkSize, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
    } else if (algo == "cascaded") {
        manager = std::make_shared<nvcomp::CascadedManager>(chunkSize, nvcompBatchedCascadedCompressDefaultOpts, nvcompBatchedCascadedDecompressDefaultOpts, stream);
    } else if (algo == "bitcomp") {
        manager = std::make_shared<nvcomp::BitcompManager>(chunkSize, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    } else {
        std::cerr << "Unknown algorithm: " << algo << ". Defaulting to LZ4." << std::endl;
        manager = std::make_shared<nvcomp::LZ4Manager>(chunkSize, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, stream);
    }

    // Configure Compression
    std::cout << "Configuring compression (" << algo << ")..." << std::endl;
    nvcomp::CompressionConfig config = manager->configure_compression(inputSize);

    // Allocate output buffer on GPU
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, config.max_compressed_buffer_size));
    
    // Allocate size output on GPU (optional for native format, but good practice)
    // For NVCOMP_NATIVE, the size is implicitly handled or stored in headers, 
    // but `compress` API usually takes a pointer if we want the exact result size immediately on device.
    // However, the high-level API usually manages this. Let's check the signature.
    // virtual void compress(const uint8_t* uncomp, uint8_t* comp, const CompressionConfig& config)
    
    std::cout << "Compressing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->compress(d_input, d_output, config);
    
    // Sync to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    // Get compressed size
    // With the high-level manager and native format, the compressed data includes headers.
    // The `compress` call doesn't return the exact size in a host variable directly in all overloads.
    // We need to retrieve the exact compressed size. 
    // Looking at the API, configure_compression gives us `max_compressed_buffer_size`.
    // The `compress` method writes the result. 
    // For NATIVE format, the size is tricky to get without a `comp_size` output parameter on device.
    // Let's allocate a device variable for size.
    
    size_t* d_comp_size;
    CUDA_CHECK(cudaMalloc(&d_comp_size, sizeof(size_t)));
    
    // Re-call compress with size pointer if supported by the specific override, 
    // or we assume the API signature: compress(in, out, config, out_size)
    manager->compress(d_input, d_output, config);
    
    // Actually, the high level API often hides the exact size in the header or we need to calculate it?
    // The docs say: virtual void compress(..., size_t* comp_size = nullptr)
    // So we should pass it.
    manager->compress(d_input, d_output, config, d_comp_size);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t compSize;
    CUDA_CHECK(cudaMemcpy(&compSize, d_comp_size, sizeof(size_t), cudaMemcpyDeviceToHost));

    std::cout << "Compressed size: " << compSize << " bytes" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x" << std::endl;
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;

    // Copy back to host
    std::vector<uint8_t> outputData(compSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));

    std::cout << "Writing output file: " << outputFile << "..." << std::endl;
    writeFile(outputFile, outputData.data(), outputData.size());

    // Cleanup
    manager.reset(); // Destroy manager before stream
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_comp_size);
    cudaStreamDestroy(stream);
}

// Decompression function
void decompressFile(const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Reading compressed file: " << inputFile << "..." << std::endl;
    std::vector<uint8_t> inputData = readFile(inputFile);
    size_t inputSize = inputData.size();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate input buffer on GPU
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));

    // Use Factory to create manager from compressed data
    // This automatically detects the format
    std::cout << "Detecting format and creating manager..." << std::endl;
    
    // Note: create_manager usually takes the compressed buffer on device
    auto manager = nvcomp::create_manager(d_input, stream);

    // Configure Decompression
    std::cout << "Configuring decompression..." << std::endl;
    nvcomp::DecompressionConfig config = manager->configure_decompression(d_input);
    
    size_t outputSize = config.decomp_data_size;
    std::cout << "Original size detected: " << outputSize << " bytes" << std::endl;

    // Allocate output buffer on GPU
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    std::cout << "Decompressing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    manager->decompress(d_output, d_input, config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)" << std::endl;

    // Copy back to host
    std::vector<uint8_t> outputData(outputSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

    std::cout << "Writing output file: " << outputFile << "..." << std::endl;
    writeFile(outputFile, outputData.data(), outputData.size());

    // Cleanup
    manager.reset(); // Destroy manager before stream
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void printUsage(const char* appName) {
    std::cerr << "Usage:\n"
              << "  Compress:   " << appName << " -c <input> <output> [lz4|gdeflate|snappy|bitcomp|ans|cascaded]\n"
              << "  Decompress: " << appName << " -d <input> <output>\n";
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

        if (mode == "-c") {
            std::string algo = "lz4"; // Default
            if (argc >= 5) {
                algo = argv[4];
            }
            compressFile(inputFile, outputFile, algo);
        } else if (mode == "-d") {
            decompressFile(inputFile, outputFile);
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

