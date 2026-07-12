/**
 * @file nvcomp_c_api.h
 * @brief C API wrapper for nvcomp_core library
 * 
 * Provides a C-compatible interface for cross-language compatibility
 * and potential future bindings (Python, C#, etc.)
 */

#ifndef NVCOMP_C_API_H
#define NVCOMP_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Windows DLL export/import macros
#ifdef _WIN32
    #ifdef NVCOMP_CORE_EXPORTS
        #define NVCOMP_C_API __declspec(dllexport)
    #else
        #define NVCOMP_C_API __declspec(dllimport)
    #endif
#else
    #define NVCOMP_C_API
#endif

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
    NVCOMP_SUCCESS = 0,
    NVCOMP_ERROR_INVALID_ARGUMENT,
    NVCOMP_ERROR_FILE_NOT_FOUND,
    NVCOMP_ERROR_FILE_IO,
    NVCOMP_ERROR_INVALID_FORMAT,
    NVCOMP_ERROR_COMPRESSION_FAILED,
    NVCOMP_ERROR_DECOMPRESSION_FAILED,
    NVCOMP_ERROR_OUT_OF_MEMORY,
    NVCOMP_ERROR_CUDA_ERROR,
    NVCOMP_ERROR_UNSUPPORTED_ALGORITHM,
    NVCOMP_ERROR_UNKNOWN,
    NVCOMP_ERROR_CANCELED
} nvcomp_error_t;

// ============================================================================
// Algorithm Types
// ============================================================================

typedef enum {
    NVCOMP_ALGO_LZ4 = 0,
    NVCOMP_ALGO_SNAPPY,
    NVCOMP_ALGO_ZSTD,
    NVCOMP_ALGO_GDEFLATE,
    NVCOMP_ALGO_ANS,
    NVCOMP_ALGO_BITCOMP,
    NVCOMP_ALGO_UNKNOWN
} nvcomp_algorithm_t;

// ============================================================================
// Operation Handle (Opaque Type for Progress Callbacks)
// ============================================================================

typedef struct nvcomp_operation_t* nvcomp_operation_handle;

// ============================================================================
// Compression Statistics
// ============================================================================

/**
 * @brief Phase-by-phase timing/metrics for a single compression or
 *        decompression operation.
 *
 * Pass a pointer to one of these structs as the trailing `out_stats`
 * argument of any nvcomp_compress_* / nvcomp_decompress_* call.
 * NULL is allowed if the caller does not want stats.
 *
 * For decompression, input_bytes is the uncompressed (output) size and
 * output_bytes is the compressed (input) size, so 'ratio' stays consistent
 * with compression results.
 */
typedef struct {
    double read_sec;          ///< Reading input + assembling in-memory archive
    double prepare_sec;       ///< GPU buffer setup, host->device copies, scratch
    double compute_sec;       ///< Actual compression/decompression kernels (sync wall time)
    double write_sec;         ///< Writing output file(s)
    double total_sec;         ///< Sum of the four phases
    uint64_t input_bytes;     ///< Uncompressed payload size (in bytes)
    uint64_t output_bytes;    ///< Compressed file size (in bytes)
    double throughput_mbps;   ///< input_bytes / total_sec, in MB/s
    double throughput_gbps;   ///< input_bytes / total_sec, in GB/s
    double ratio;             ///< uncompressed / compressed
} nvcomp_compression_stats_t;

// ============================================================================
// Progress Callback
// ============================================================================

/**
 * @brief Block-level progress information
 */
typedef struct {
    int totalBlocks;              ///< Total number of blocks
    int completedBlocks;          ///< Number of completed blocks
    int currentBlock;             ///< Index of current block being processed
    size_t currentBlockSize;      ///< Size of current block in bytes
    float overallProgress;        ///< Overall progress (0.0 to 1.0)
    float currentBlockProgress;   ///< Current block progress (0.0 to 1.0)
    double throughputMBps;        ///< Current throughput in MB/s
    const char* stage;            ///< Current stage (e.g., "preparing", "compressing", "writing")
} nvcomp_progress_info_t;

/**
 * @brief Progress callback function type (simple)
 * @param current Current progress value
 * @param total Total progress value
 * @param user_data User-provided context data
 */
typedef void (*nvcomp_progress_callback_t)(uint64_t current, uint64_t total, void* user_data);

/**
 * @brief Block-level progress callback function type
 * @param handle Operation handle
 * @param info Detailed progress information
 * @param user_data User-provided context data
 */
typedef void (*nvcomp_block_progress_callback_t)(
    nvcomp_operation_handle handle,
    const nvcomp_progress_info_t* info,
    void* user_data
);

// ============================================================================
// Error Handling Functions
// ============================================================================

/**
 * @brief Get the last error message for the current thread
 * @return Error message string (thread-local, do not free)
 */
NVCOMP_C_API const char* nvcomp_get_last_error(void);

/**
 * @brief Clear the last error for the current thread
 */
NVCOMP_C_API void nvcomp_clear_last_error(void);

// ============================================================================
// Algorithm Utility Functions
// ============================================================================

/**
 * @brief Parse algorithm string to enum
 * @param algo_str Algorithm name (e.g., "lz4", "snappy", "zstd")
 * @return Algorithm enum value
 */
NVCOMP_C_API nvcomp_algorithm_t nvcomp_parse_algorithm(const char* algo_str);

/**
 * @brief Convert algorithm enum to string
 * @param algo Algorithm enum value
 * @return Algorithm name string (do not free)
 */
NVCOMP_C_API const char* nvcomp_algorithm_to_string(nvcomp_algorithm_t algo);

/**
 * @brief Check if algorithm is cross-compatible (CPU/GPU)
 * @param algo Algorithm to check
 * @return true if cross-compatible, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_cross_compatible(nvcomp_algorithm_t algo);

/**
 * @brief Check if CUDA is available
 * @return true if CUDA is available, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_cuda_available(void);

// ============================================================================
// File I/O Utilities
// ============================================================================

/**
 * @brief Check if path is a directory
 * @param path Path to check
 * @return true if directory, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_directory(const char* path);

/**
 * @brief Create directories recursively
 * @param path Directory path to create
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_create_directories(const char* path);

// ============================================================================
// Volume Support Functions
// ============================================================================

/**
 * @brief Check if filename indicates a volume file
 * @param filename Filename to check
 * @return true if volume file, false otherwise
 */
NVCOMP_C_API bool nvcomp_is_volume_file(const char* filename);

/**
 * @brief Parse volume size string (e.g., "2.5GB", "100MB")
 * @param size_str Size string to parse
 * @return Size in bytes, or 0 on error
 */
NVCOMP_C_API uint64_t nvcomp_parse_volume_size(const char* size_str);

/**
 * @brief Check if GPU memory is sufficient for volume size
 * @param volume_size Volume size in bytes
 * @return true if sufficient, false otherwise
 */
NVCOMP_C_API bool nvcomp_check_gpu_memory_for_volume(uint64_t volume_size);

// ============================================================================
// Operation Handle Functions (for Progress Callbacks)
// ============================================================================

/**
 * @brief Create an operation handle for tracking progress
 * @return Operation handle, or NULL on error
 */
NVCOMP_C_API nvcomp_operation_handle nvcomp_create_operation_handle(void);

/**
 * @brief Destroy an operation handle
 * @param handle Handle to destroy
 */
NVCOMP_C_API void nvcomp_destroy_operation_handle(nvcomp_operation_handle handle);

/**
 * @brief Set progress callback for an operation
 * @param handle Operation handle
 * @param callback Callback function
 * @param user_data User data to pass to callback
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_set_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_progress_callback_t callback,
    void* user_data
);

/**
 * @brief Set block-level progress callback for an operation
 * @param handle Operation handle
 * @param callback Block progress callback function
 * @param user_data User data to pass to callback
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_set_block_progress_callback(
    nvcomp_operation_handle handle,
    nvcomp_block_progress_callback_t callback,
    void* user_data
);

// ============================================================================
// Compression Functions
// ============================================================================

/**
 * @brief Compress file or folder using GPU (batched API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Compress multiple files using GPU (batched API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param file_paths Array of file paths to compress
 * @param file_count Number of files in the array
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_batched_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Decompress file using GPU (batched API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_gpu_batched(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Compress file or folder using GPU (manager API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_manager(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Compress multiple files using GPU (manager API)
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param file_paths Array of file paths to compress
 * @param file_count Number of files in the array
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_gpu_manager_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Decompress file using GPU (manager API - auto-detects algorithm)
 * @param handle Operation handle (can be NULL)
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_gpu_manager(
    nvcomp_operation_handle handle,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Compress file or folder using CPU
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_path Input file or folder path
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_path,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Compress multiple files using CPU
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param file_paths Array of file paths to compress
 * @param file_count Number of files in the array
 * @param output_file Output compressed file path
 * @param max_volume_size Maximum volume size (0 for no splitting)
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_compress_cpu_file_list(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char** file_paths,
    size_t file_count,
    const char* output_file,
    uint64_t max_volume_size,
    nvcomp_compression_stats_t* out_stats
);

/**
 * @brief Decompress file using CPU
 * @param handle Operation handle (can be NULL)
 * @param algo Algorithm to use
 * @param input_file Input compressed file path
 * @param output_path Output file or folder path
 * @param out_stats Optional, populated with per-phase timing on success (can be NULL)
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_decompress_cpu(
    nvcomp_operation_handle handle,
    nvcomp_algorithm_t algo,
    const char* input_file,
    const char* output_path,
    nvcomp_compression_stats_t* out_stats
);

// ============================================================================
// Stats Helpers
// ============================================================================

/**
 * @brief Format a stats struct as a multi-line summary identical to the
 *        one printed by the core after every operation.
 * @param stats Stats to format (must not be NULL)
 * @param op_name Short operation label, e.g. "Compression" or "Decompression"
 * @param out_buffer Caller-provided buffer
 * @param buffer_size Size of out_buffer in bytes
 * @return Number of bytes written (excluding trailing NUL), or 0 on error.
 */
NVCOMP_C_API size_t nvcomp_format_stats_summary(
    const nvcomp_compression_stats_t* stats,
    const char* op_name,
    char* out_buffer,
    size_t buffer_size
);

// ============================================================================
// Verbose Flag
// ============================================================================

/**
 * @brief Enable or disable verbose stdout chatter from the core (per-file
 *        "Adding:" lines, per-volume progress messages, etc.).
 *
 * Default is off. The CLI flips it on for --verbose / -v. Final result
 * summaries (stats, "SUCCESSFUL", extraction completion) are always printed
 * regardless of this setting; this only gates incremental progress chatter
 * that otherwise dominates Read-phase throughput on large inputs.
 *
 * @param verbose Non-zero to enable, zero to disable.
 */
NVCOMP_C_API void nvcomp_set_verbose(int verbose);

/**
 * @brief Query the current verbose flag.
 * @return Non-zero if verbose output is enabled.
 */
NVCOMP_C_API int nvcomp_get_verbose(void);

// ============================================================================
// Algorithm Detection
// ============================================================================

/**
 * @brief Detect compression algorithm from file
 * @param filename Compressed file path
 * @return Detected algorithm, or NVCOMP_ALGO_UNKNOWN on error
 */
NVCOMP_C_API nvcomp_algorithm_t nvcomp_detect_algorithm_from_file(const char* filename);

// ============================================================================
// Archive Listing
// ============================================================================

/**
 * @brief Per-entry callback for nvcomp_list_archive_entries
 * @param path Entry path inside the archive (UTF-8, valid only during the call)
 * @param size Uncompressed file size in bytes
 * @param mode POSIX permission bits (0 = unknown, e.g. v1 archives)
 * @param mtime_ns Modification time, ns since the Unix epoch (0 = unknown)
 * @param user_data User-provided context data
 * @return 0 to continue, nonzero to cancel the listing
 */
typedef int (*nvcomp_entry_callback_t)(const char* path, uint64_t size,
                                       uint32_t mode, uint64_t mtime_ns,
                                       void* user_data);

/**
 * @brief Enumerate archive entries without extracting anything to disk.
 *
 * Handles uncompressed (NVAR), compressed single-file (NVBC), and
 * multi-volume (NVVM) archives; any volume path selects the whole set. The
 * algorithm is auto-detected. Compressed archives stream through the GPU
 * decompressor (CPU fallback) with sub-batch memory use; no temp files are
 * written.
 *
 * @param input_file Archive path
 * @param entry_callback Called once per entry, in archive order (required)
 * @param progress_callback Optional; receives (processed, total) uncompressed
 *                          byte counts
 * @param user_data Passed to both callbacks
 * @return NVCOMP_SUCCESS, NVCOMP_ERROR_CANCELED if the entry callback
 *         returned nonzero, or an error code (see nvcomp_get_last_error)
 */
NVCOMP_C_API nvcomp_error_t nvcomp_list_archive_entries(
    const char* input_file,
    nvcomp_entry_callback_t entry_callback,
    nvcomp_progress_callback_t progress_callback,
    void* user_data
);

/**
 * @brief List contents of compressed archive
 * @param algo Algorithm used for compression
 * @param input_file Compressed archive path
 * @param use_cpu Use CPU for decompression
 * @param cuda_available Whether CUDA is available
 * @return Error code
 */
NVCOMP_C_API nvcomp_error_t nvcomp_list_compressed_archive(
    nvcomp_algorithm_t algo,
    const char* input_file,
    bool use_cpu,
    bool cuda_available
);

#ifdef __cplusplus
}
#endif

#endif // NVCOMP_C_API_H



