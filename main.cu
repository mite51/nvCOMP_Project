/*
 * nvCOMP CLI - Thin wrapper around nvcomp_core library
 * 
 * This CLI provides command-line access to the nvcomp_core library's
 * compression and decompression functionality via the C API.
 */

#include <iostream>
#include <string>
#include <cstring>

// Include the C API from the core library
#include "nvcomp_c_api.h"

// ============================================================================
// Helper Functions
// ============================================================================

void printUsage(const char* appName) {
    std::cerr << "nvCOMP CLI with CPU Fallback & Multi-Volume Support\n\n";
    std::cerr << "Usage:\n";
    std::cerr << "  Compress:   " << appName << " -c <input> <output> <algorithm> [options]\n";
    std::cerr << "  Decompress: " << appName << " -d <input> <output> [algorithm] [options]\n";
    std::cerr << "  List:       " << appName << " -l <archive> [algorithm] [options]\n\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  <input>     Input file or directory (for compression)\n";
    std::cerr << "  <output>    Output file (compression) or directory (decompression)\n";
    std::cerr << "  <archive>   Compressed archive file to list (e.g., output.vol001.lz4 or output.lz4)\n";
    std::cerr << "  [algorithm] Optional for -d/-l (auto-detected), required for -c\n\n";
    std::cerr << "Algorithms:\n";
    std::cerr << "  Cross-compatible (GPU/CPU): lz4, snappy, zstd\n";
    std::cerr << "  GPU-only: gdeflate, ans, bitcomp\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --cpu              Force CPU mode\n";
    std::cerr << "  --volume-size <N>  Set max volume size (default: 2.5GB)\n";
    std::cerr << "                     Examples: 1GB, 500MB, 5GB\n";
    std::cerr << "  --no-volumes       Disable volume splitting (single file)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  # Compress with default 2.5GB volumes\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4\n\n";
    std::cerr << "  # Compress entire folder with custom volume size\n";
    std::cerr << "  " << appName << " -c mydata/ output.zstd zstd --volume-size 1GB\n\n";
    std::cerr << "  # Compress without volume splitting\n";
    std::cerr << "  " << appName << " -c mydata/ output.lz4 lz4 --no-volumes\n\n";
    std::cerr << "  # Decompress multi-volume archive (auto-detects volumes)\n";
    std::cerr << "  " << appName << " -d output.vol001.lz4 restored/\n\n";
    std::cerr << "  # Decompress single file with auto-detection\n";
    std::cerr << "  " << appName << " -d output.lz4 restored/\n\n";
    std::cerr << "  # List multi-volume archive\n";
    std::cerr << "  " << appName << " -l output.vol001.zstd\n\n";
    std::cerr << "  # Force CPU mode\n";
    std::cerr << "  " << appName << " -c input.txt output.lz4 lz4 --cpu\n";
}

void printError(const char* message) {
    std::cerr << "Error: " << message << std::endl;
    const char* lastError = nvcomp_get_last_error();
    if (lastError && strlen(lastError) > 0) {
        std::cerr << "Details: " << lastError << std::endl;
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

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
        // Default volume size: 2.5GB
        uint64_t maxVolumeSize = 2684354560ULL; // 2.5GB default
        bool noVolumes = false;
        
        // Parse algorithm and flags
        for (int i = (mode == "-l" ? 3 : 4); i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cpu") {
                forceCPU = true;
            } else if (arg == "--no-volumes") {
                noVolumes = true;
            } else if (arg == "--volume-size") {
                if (i + 1 < argc) {
                    i++;
                    maxVolumeSize = nvcomp_parse_volume_size(argv[i]);
                    if (maxVolumeSize == 0) {
                        printError("Invalid volume size");
                        return 1;
                    }
                } else {
                    std::cerr << "Error: --volume-size requires a size argument" << std::endl;
                    printUsage(argv[0]);
                    return 1;
                }
            } else {
                // Assume it's an algorithm
                algoStr = arg;
            }
        }
        
        // If --no-volumes is set, use maximum volume size
        if (noVolumes) {
            maxVolumeSize = UINT64_MAX;
        }
        
        // Parse algorithm (default to NVCOMP_ALGO_UNKNOWN for auto-detection)
        nvcomp_algorithm_t algo = NVCOMP_ALGO_UNKNOWN;
        if (!algoStr.empty()) {
            algo = nvcomp_parse_algorithm(algoStr.c_str());
            if (algo == NVCOMP_ALGO_UNKNOWN) {
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
            algo = NVCOMP_ALGO_LZ4; // Default fallback if auto-detection fails
        }
        
        bool cudaAvailable = nvcomp_is_cuda_available();
        bool useCPU = forceCPU || !cudaAvailable;
        
        if (!cudaAvailable && !forceCPU) {
            std::cout << "CUDA not available, falling back to CPU..." << std::endl;
            useCPU = true;
        }
        
        if (useCPU && !nvcomp_is_cross_compatible(algo)) {
            std::cerr << "Error: Algorithm '" << algoStr << "' is GPU-only and cannot run on CPU" << std::endl;
            return 1;
        }
        
        // Display volume size for compression mode
        if (mode == "-c") {
            if (maxVolumeSize == UINT64_MAX) {
                std::cout << "Volume size: unlimited" << std::endl;
            } else if (maxVolumeSize > 0) {
                std::cout << "Volume size: " << (maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
            } else {
                std::cout << "Volume size: 2.5 GB (default)" << std::endl;
            }
        }
        
        // Execute the requested operation using the C API
        nvcomp_error_t result = NVCOMP_SUCCESS;
        
        if (mode == "-c") {
            // Compression mode
            if (useCPU) {
                result = nvcomp_compress_cpu(NULL, algo, inputPath.c_str(), outputPath.c_str(), maxVolumeSize);
            } else {
                if (nvcomp_is_cross_compatible(algo)) {
                    result = nvcomp_compress_gpu_batched(NULL, algo, inputPath.c_str(), outputPath.c_str(), maxVolumeSize);
                } else {
                    result = nvcomp_compress_gpu_manager(NULL, algo, inputPath.c_str(), outputPath.c_str(), maxVolumeSize);
                }
            }
            
            if (result != NVCOMP_SUCCESS) {
                printError("Compression failed");
                return 1;
            }
            
        } else if (mode == "-d") {
            // Decompression mode
            if (useCPU) {
                result = nvcomp_decompress_cpu(NULL, algo, inputPath.c_str(), outputPath.c_str());
            } else {
                if (nvcomp_is_cross_compatible(algo)) {
                    result = nvcomp_decompress_gpu_batched(NULL, algo, inputPath.c_str(), outputPath.c_str());
                } else {
                    result = nvcomp_decompress_gpu_manager(NULL, inputPath.c_str(), outputPath.c_str());
                }
            }
            
            if (result != NVCOMP_SUCCESS) {
                printError("Decompression failed");
                return 1;
            }
            
        } else if (mode == "-l") {
            // List mode
            result = nvcomp_list_compressed_archive(algo, inputPath.c_str(), useCPU, cudaAvailable);
            
            if (result != NVCOMP_SUCCESS) {
                printError("Failed to list archive");
                return 1;
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
