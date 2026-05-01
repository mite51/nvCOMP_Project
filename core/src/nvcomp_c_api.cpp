/**
 * @file nvcomp_c_api.cpp
 * @brief C API wrapper implementation
 */

#include "nvcomp_c_api.h"
#include "nvcomp_core.hpp"

#include <string>
#include <cstring>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <fstream>


// ============================================================================
// Thread-local Error Storage
// ============================================================================

thread_local std::string g_last_error;

// ============================================================================
// Operation Handle Implementation
// ============================================================================

// Lock-free callback delivery: we use atomics for the callback pointers so
// reportBlockProgress (called from inside the GPU compression loop on the
// worker thread) does not have to take a mutex per chunk. The set_*_callback
// functions are called once at handle-setup time, before the worker thread
// is started, so a simple atomic store/load with acquire/release ordering is
// sufficient.
struct nvcomp_operation_t {
    std::atomic<nvcomp_progress_callback_t> callback;
    std::atomic<void*> user_data;
    std::atomic<nvcomp_block_progress_callback_t> block_callback;
    std::atomic<void*> block_user_data;

    nvcomp_operation_t()
        : callback(nullptr), user_data(nullptr),
          block_callback(nullptr), block_user_data(nullptr) {}

    void reportProgress(uint64_t current, uint64_t total) {
        auto cb = callback.load(std::memory_order_acquire);
        if (cb) {
            cb(current, total, user_data.load(std::memory_order_acquire));
        }
    }

    void reportBlockProgress(const nvcomp_progress_info_t* info) {
        auto cb = block_callback.load(std::memory_order_acquire);
        if (cb) {
            cb(this, info, block_user_data.load(std::memory_order_acquire));
        }
    }
};

// ============================================================================
// Stats conversion helper
// ============================================================================

static void copyStatsToC(const nvcomp_core::CompressionStats& src,
                        nvcomp_compression_stats_t* dst) {
    if (!dst) return;
    dst->read_sec = src.readSec;
    dst->prepare_sec = src.prepareSec;
    dst->compute_sec = src.computeSec;
    dst->write_sec = src.writeSec;
    dst->total_sec = src.totalSec;
    dst->input_bytes = src.inputBytes;
    dst->output_bytes = src.outputBytes;
    dst->throughput_mbps = src.throughputMBps;
    dst->throughput_gbps = src.throughputGBps;
    dst->ratio = src.ratio;
}

// ============================================================================
// Error Handling Helper
// ============================================================================

/**
 * @brief Helper template to execute C++ code and convert exceptions to error codes
 */
template<typename Func>
nvcomp_error_t executeSafely(Func&& func) {
    try {
        g_last_error.clear();
        func();
        return NVCOMP_SUCCESS;
    } catch (const std::invalid_argument& e) {
        g_last_error = std::string("Invalid argument: ") + e.what();
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        g_last_error = msg;
        
        // Categorize runtime errors
        if (msg.find("file") != std::string::npos || msg.find("File") != std::string::npos) {
            if (msg.find("not found") != std::string::npos || msg.find("does not exist") != std::string::npos) {
                return NVCOMP_ERROR_FILE_NOT_FOUND;
            }
            return NVCOMP_ERROR_FILE_IO;
        }
        if (msg.find("format") != std::string::npos || msg.find("invalid") != std::string::npos) {
            return NVCOMP_ERROR_INVALID_FORMAT;
        }
        if (msg.find("compress") != std::string::npos) {
            return NVCOMP_ERROR_COMPRESSION_FAILED;
        }
        if (msg.find("decompress") != std::string::npos) {
            return NVCOMP_ERROR_DECOMPRESSION_FAILED;
        }
        if (msg.find("memory") != std::string::npos || msg.find("allocation") != std::string::npos) {
            return NVCOMP_ERROR_OUT_OF_MEMORY;
        }
        if (msg.find("CUDA") != std::string::npos || msg.find("GPU") != std::string::npos) {
            return NVCOMP_ERROR_CUDA_ERROR;
        }
        if (msg.find("algorithm") != std::string::npos || msg.find("unsupported") != std::string::npos) {
            return NVCOMP_ERROR_UNSUPPORTED_ALGORITHM;
        }
        
        return NVCOMP_ERROR_UNKNOWN;
    } catch (const std::bad_alloc& e) {
        g_last_error = std::string("Out of memory: ") + e.what();
        return NVCOMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        g_last_error = std::string("Unknown error: ") + e.what();
        return NVCOMP_ERROR_UNKNOWN;
    } catch (...) {
        g_last_error = "Unknown error occurred";
        return NVCOMP_ERROR_UNKNOWN;
    }
}

// ============================================================================
// Algorithm Conversion Helpers
// ============================================================================

nvcomp_core::AlgoType toCorealgo(nvcomp_algorithm_t algo) {
    switch (algo) {
        case NVCOMP_ALGO_LZ4: return nvcomp_core::ALGO_LZ4;
        case NVCOMP_ALGO_SNAPPY: return nvcomp_core::ALGO_SNAPPY;
        case NVCOMP_ALGO_ZSTD: return nvcomp_core::ALGO_ZSTD;
        case NVCOMP_ALGO_GDEFLATE: return nvcomp_core::ALGO_GDEFLATE;
        case NVCOMP_ALGO_ANS: return nvcomp_core::ALGO_ANS;
        case NVCOMP_ALGO_BITCOMP: return nvcomp_core::ALGO_BITCOMP;
        default: return nvcomp_core::ALGO_UNKNOWN;
    }
}

nvcomp_algorithm_t fromCoreAlgo(nvcomp_core::AlgoType algo) {
    switch (algo) {
        case nvcomp_core::ALGO_LZ4: return NVCOMP_ALGO_LZ4;
        case nvcomp_core::ALGO_SNAPPY: return NVCOMP_ALGO_SNAPPY;
        case nvcomp_core::ALGO_ZSTD: return NVCOMP_ALGO_ZSTD;
        case nvcomp_core::ALGO_GDEFLATE: return NVCOMP_ALGO_GDEFLATE;
        case nvcomp_core::ALGO_ANS: return NVCOMP_ALGO_ANS;
        case nvcomp_core::ALGO_BITCOMP: return NVCOMP_ALGO_BITCOMP;
        default: return NVCOMP_ALGO_UNKNOWN;
    }
}

// ============================================================================
// Error Handling Functions
// ============================================================================

const char* nvcomp_get_last_error(void) {
    return g_last_error.c_str();
}

void nvcomp_clear_last_error(void) {
    g_last_error.clear();
}

// ============================================================================
// Algorithm Utility Functions
// ============================================================================

nvcomp_algorithm_t nvcomp_parse_algorithm(const char* algo_str) {
    if (!algo_str) return NVCOMP_ALGO_UNKNOWN;
    
    try {
        auto coreAlgo = nvcomp_core::parseAlgorithm(algo_str);
        return fromCoreAlgo(coreAlgo);
    } catch (...) {
        return NVCOMP_ALGO_UNKNOWN;
    }
}

const char* nvcomp_algorithm_to_string(nvcomp_algorithm_t algo) {
    thread_local std::string algo_str;
    try {
        algo_str = nvcomp_core::algoToString(toCorealgo(algo));
        return algo_str.c_str();
    } catch (...) {
        return "unknown";
    }
}

bool nvcomp_is_cross_compatible(nvcomp_algorithm_t algo) {
    try {
        return nvcomp_core::isCrossCompatible(toCorealgo(algo));
    } catch (...) {
        return false;
    }
}

bool nvcomp_is_cuda_available(void) {
    try {
        return nvcomp_core::isCudaAvailable();
    } catch (...) {
        return false;
    }
}

// ============================================================================
// File I/O Utilities
// ============================================================================

bool nvcomp_is_directory(const char* path) {
    if (!path) return false;
    
    try {
        return nvcomp_core::isDirectory(path);
    } catch (...) {
        return false;
    }
}

nvcomp_error_t nvcomp_create_directories(const char* path) {
    if (!path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    return executeSafely([&]() {
        nvcomp_core::createDirectories(path);
    });
}

// ============================================================================
// Volume Support Functions
// ============================================================================

bool nvcomp_is_volume_file(const char* filename) {
    if (!filename) return false;
    
    try {
        return nvcomp_core::isVolumeFile(filename);
    } catch (...) {
        return false;
    }
}

uint64_t nvcomp_parse_volume_size(const char* size_str) {
    if (!size_str) return 0;
    
    try {
        return nvcomp_core::parseVolumeSize(size_str);
    } catch (...) {
        return 0;
    }
}

bool nvcomp_check_gpu_memory_for_volume(uint64_t volume_size) {
    try {
        return nvcomp_core::checkGPUMemoryForVolume(volume_size);
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Operation Handle Functions
// ============================================================================

nvcomp_operation_handle nvcomp_create_operation_handle(void) {
    try {
        return new nvcomp_operation_t();
    } catch (...) {
        return nullptr;
    }
}

void nvcomp_destroy_operation_handle(nvcomp_operation_handle handle) {
    if (handle) {
        delete handle;
    }
}

nvcomp_error_t nvcomp_set_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_progress_callback_t callback,
    void* user_data
) {
    if (!handle) {
        g_last_error = "Null operation handle";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    handle->user_data.store(user_data, std::memory_order_release);
    handle->callback.store(callback, std::memory_order_release);
    return NVCOMP_SUCCESS;
}

nvcomp_error_t nvcomp_set_block_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_block_progress_callback_t callback,
    void* user_data
) {
    if (!handle) {
        g_last_error = "Null operation handle";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    handle->block_user_data.store(user_data, std::memory_order_release);
    handle->block_callback.store(callback, std::memory_order_release);
    return NVCOMP_SUCCESS;
}

// ============================================================================
// Compression Functions
// ============================================================================

// Builds a C++ progress callback that forwards to the handle's C callback.
// Returns nullptr if the handle has no block callback set.
static nvcomp_core::ProgressCallback makeForwardingCallback(nvcomp_operation_handle handle) {
    if (!handle) return nullptr;
    if (!handle->block_callback.load(std::memory_order_acquire)) return nullptr;
    return [handle](const nvcomp_core::BlockProgressInfo& info) {
        nvcomp_progress_info_t c_info;
        c_info.totalBlocks = info.totalBlocks;
        c_info.completedBlocks = info.completedBlocks;
        c_info.currentBlock = info.currentBlock;
        c_info.currentBlockSize = info.currentBlockSize;
        c_info.overallProgress = info.overallProgress;
        c_info.currentBlockProgress = info.currentBlockProgress;
        c_info.throughputMBps = info.throughputMBps;
        c_info.stage = info.stage.c_str();
        handle->reportBlockProgress(&c_info);
    };
}

nvcomp_error_t nvcomp_compress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::compressGPUBatched(
            toCorealgo(algo), input_path, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_compress_gpu_batched_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!file_paths || file_count == 0 || !output_file) {
        g_last_error = "Invalid file list or output path";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        std::vector<std::string> paths;
        paths.reserve(file_count);
        for (size_t i = 0; i < file_count; i++) {
            if (file_paths[i]) paths.emplace_back(file_paths[i]);
        }
        nvcomp_core::compressGPUBatchedFileList(
            toCorealgo(algo), paths, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_decompress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::decompressGPUBatched(
            toCorealgo(algo), input_file, output_path,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_compress_gpu_manager(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::compressGPUManager(
            toCorealgo(algo), input_path, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_compress_gpu_manager_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!file_paths || file_count == 0 || !output_file) {
        g_last_error = "Invalid file list or output path";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        std::vector<std::string> paths;
        paths.reserve(file_count);
        for (size_t i = 0; i < file_count; i++) {
            if (file_paths[i]) paths.emplace_back(file_paths[i]);
        }
        nvcomp_core::compressGPUManagerFileList(
            toCorealgo(algo), paths, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_decompress_gpu_manager(
    nvcomp_operation_handle handle,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::decompressGPUManager(
            input_file, output_path,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_compress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_path || !output_file) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::compressCPU(
            toCorealgo(algo), input_path, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_compress_cpu_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
) {
    if (!file_paths || file_count == 0 || !output_file) {
        g_last_error = "Invalid file list or output path";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        std::vector<std::string> paths;
        paths.reserve(file_count);
        for (size_t i = 0; i < file_count; i++) {
            if (file_paths[i]) paths.emplace_back(file_paths[i]);
        }
        nvcomp_core::compressCPUFileList(
            toCorealgo(algo), paths, output_file, max_volume_size,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

nvcomp_error_t nvcomp_decompress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
) {
    if (!input_file || !output_path) {
        g_last_error = "Null path provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }

    nvcomp_core::CompressionStats stats;
    auto result = executeSafely([&]() {
        nvcomp_core::decompressCPU(
            toCorealgo(algo), input_file, output_path,
            makeForwardingCallback(handle),
            out_stats ? &stats : nullptr);
    });

    if (result == NVCOMP_SUCCESS) {
        copyStatsToC(stats, out_stats);
        if (handle) handle->reportProgress(100, 100);
    }
    return result;
}

// ============================================================================
// Stats helpers
// ============================================================================

size_t nvcomp_format_stats_summary(
    const nvcomp_compression_stats_t* stats,
    const char* op_name,
    char* out_buffer,
    size_t buffer_size
) {
    if (!stats || !out_buffer || buffer_size == 0) return 0;

    nvcomp_core::CompressionStats s;
    s.readSec = stats->read_sec;
    s.prepareSec = stats->prepare_sec;
    s.computeSec = stats->compute_sec;
    s.writeSec = stats->write_sec;
    s.totalSec = stats->total_sec;
    s.inputBytes = stats->input_bytes;
    s.outputBytes = stats->output_bytes;
    s.throughputMBps = stats->throughput_mbps;
    s.throughputGBps = stats->throughput_gbps;
    s.ratio = stats->ratio;

    std::string summary = nvcomp_core::formatStatsSummary(s, op_name ? op_name : "Operation");
    size_t copyLen = std::min(summary.size(), buffer_size - 1);
    std::memcpy(out_buffer, summary.data(), copyLen);
    out_buffer[copyLen] = '\0';
    return copyLen;
}

// ============================================================================
// Verbose Flag
// ============================================================================

void nvcomp_set_verbose(int verbose) {
    nvcomp_core::setVerbose(verbose != 0);
}

int nvcomp_get_verbose(void) {
    return nvcomp_core::isVerbose() ? 1 : 0;
}

// ============================================================================
// Algorithm Detection
// ============================================================================

nvcomp_algorithm_t nvcomp_detect_algorithm_from_file(const char* filename) {
    if (!filename) return NVCOMP_ALGO_UNKNOWN;
    
    try {
        auto coreAlgo = nvcomp_core::detectAlgorithmFromFile(filename);
        return fromCoreAlgo(coreAlgo);
    } catch (...) {
        return NVCOMP_ALGO_UNKNOWN;
    }
}

// ============================================================================
// Archive Listing
// ============================================================================

nvcomp_error_t nvcomp_list_compressed_archive(
    nvcomp_algorithm_t algo,
    const char* input_file,
    bool use_cpu,
    bool cuda_available
) {
    if (!input_file) {
        g_last_error = "Null input file provided";
        return NVCOMP_ERROR_INVALID_ARGUMENT;
    }
    
    return executeSafely([&]() {
        nvcomp_core::listCompressedArchive(
            toCorealgo(algo),
            input_file,
            use_cpu,
            cuda_available
        );
    });
}



