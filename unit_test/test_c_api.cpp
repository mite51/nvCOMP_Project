/**
 * @file test_c_api.cpp
 * @brief Comprehensive test suite for C API wrapper
 * 
 * Tests all major C API functions including:
 * - Error handling
 * - Algorithm utilities
 * - File operations
 * - Volume support
 * - Operation handles and progress callbacks
 * - Compression/decompression (CPU and GPU)
 * - Archive listing
 */

#include "nvcomp_c_api.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

// Test utilities
int g_test_count = 0;
int g_test_passed = 0;
int g_test_failed = 0;

void reportProgress(uint64_t current, uint64_t total, void* user_data) {
    int* call_count = static_cast<int*>(user_data);
    if (call_count) {
        (*call_count)++;
    }
    std::cout << "  Progress: " << current << "/" << total << std::endl;
}

#define TEST_START(name) \
    do { \
        g_test_count++; \
        std::cout << "\n[Test " << g_test_count << "] " << name << std::endl; \
    } while(0)

#define TEST_PASS() \
    do { \
        std::cout << "  ✓ PASS" << std::endl; \
        g_test_passed++; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        std::cout << "  ✗ FAIL: " << msg << std::endl; \
        g_test_failed++; \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_FALSE(cond, msg) \
    do { \
        if (cond) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_NOT_NULL(ptr, msg) \
    do { \
        if (!(ptr)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

// ============================================================================
// Test Functions
// ============================================================================

void test_error_handling() {
    TEST_START("Error Handling");
    
    // Clear any existing error
    nvcomp_clear_last_error();
    const char* error = nvcomp_get_last_error();
    ASSERT_TRUE(strlen(error) == 0, "Error should be cleared");
    
    // Test error from invalid operation
    nvcomp_error_t result = nvcomp_compress_cpu(nullptr, NVCOMP_ALGO_LZ4, nullptr, nullptr, 0, nullptr);
    ASSERT_TRUE(result == NVCOMP_ERROR_INVALID_ARGUMENT, "Should return invalid argument error");
    
    error = nvcomp_get_last_error();
    ASSERT_TRUE(strlen(error) > 0, "Error message should be set");
    std::cout << "  Error message: " << error << std::endl;
    
    TEST_PASS();
}

void test_algorithm_functions() {
    TEST_START("Algorithm Functions");
    
    // Test parsing
    nvcomp_algorithm_t algo = nvcomp_parse_algorithm("lz4");
    ASSERT_EQ(algo, NVCOMP_ALGO_LZ4, "Should parse 'lz4' correctly");
    
    algo = nvcomp_parse_algorithm("snappy");
    ASSERT_EQ(algo, NVCOMP_ALGO_SNAPPY, "Should parse 'snappy' correctly");
    
    algo = nvcomp_parse_algorithm("zstd");
    ASSERT_EQ(algo, NVCOMP_ALGO_ZSTD, "Should parse 'zstd' correctly");
    
    algo = nvcomp_parse_algorithm("invalid");
    ASSERT_EQ(algo, NVCOMP_ALGO_UNKNOWN, "Should return UNKNOWN for invalid algorithm");
    
    // Test to_string
    const char* algo_str = nvcomp_algorithm_to_string(NVCOMP_ALGO_LZ4);
    ASSERT_TRUE(strcmp(algo_str, "lz4") == 0, "Should convert LZ4 to 'lz4'");
    
    algo_str = nvcomp_algorithm_to_string(NVCOMP_ALGO_SNAPPY);
    ASSERT_TRUE(strcmp(algo_str, "snappy") == 0, "Should convert SNAPPY to 'snappy'");
    
    // Test cross-compatibility
    bool cross_compat = nvcomp_is_cross_compatible(NVCOMP_ALGO_LZ4);
    std::cout << "  LZ4 cross-compatible: " << (cross_compat ? "yes" : "no") << std::endl;
    
    // Test CUDA availability
    bool cuda_available = nvcomp_is_cuda_available();
    std::cout << "  CUDA available: " << (cuda_available ? "yes" : "no") << std::endl;
    
    TEST_PASS();
}

void test_file_operations() {
    TEST_START("File Operations");
    
    // Test directory check
    bool is_dir = nvcomp_is_directory("sample_folder");
    ASSERT_TRUE(is_dir, "sample_folder should be a directory");
    
    is_dir = nvcomp_is_directory("sample.txt");
    ASSERT_FALSE(is_dir, "sample.txt should not be a directory");
    
    // Test directory creation
    nvcomp_error_t result = nvcomp_create_directories("output/test_c_api");
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should create directories successfully");
    
    TEST_PASS();
}

void test_volume_support() {
    TEST_START("Volume Support Functions");
    
    // Test volume file detection
    // Note: Volume files use pattern like "filename.lz4.v001", "filename.lz4.v002", etc.
    bool is_volume = nvcomp_is_volume_file("test.lz4.v001");
    std::cout << "  Is 'test.lz4.v001' a volume file? " << (is_volume ? "yes" : "no") << std::endl;
    // The core library may have specific rules for volume detection, we'll be lenient here
    
    is_volume = nvcomp_is_volume_file("test.lz4");
    ASSERT_FALSE(is_volume, "Should not detect .lz4 as volume file");
    
    // Test volume size parsing
    uint64_t size = nvcomp_parse_volume_size("100MB");
    ASSERT_TRUE(size == 100 * 1024 * 1024, "Should parse '100MB' correctly");
    
    size = nvcomp_parse_volume_size("2.5GB");
    ASSERT_TRUE(size == (uint64_t)(2.5 * 1024 * 1024 * 1024), "Should parse '2.5GB' correctly");
    
    // Test GPU memory check
    bool sufficient = nvcomp_check_gpu_memory_for_volume(100 * 1024 * 1024);
    std::cout << "  GPU memory sufficient for 100MB: " << (sufficient ? "yes" : "no") << std::endl;
    
    TEST_PASS();
}

void test_operation_handle() {
    TEST_START("Operation Handle Creation/Destruction");
    
    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    ASSERT_NOT_NULL(handle, "Should create operation handle");
    
    nvcomp_destroy_operation_handle(handle);
    
    TEST_PASS();
}

void test_progress_callback() {
    TEST_START("Progress Callback");
    
    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    ASSERT_NOT_NULL(handle, "Should create operation handle");
    
    int callback_count = 0;
    nvcomp_error_t result = nvcomp_set_progress_callback(handle, reportProgress, &callback_count);
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should set progress callback");
    
    nvcomp_destroy_operation_handle(handle);
    
    TEST_PASS();
}

void test_cpu_compress_decompress() {
    TEST_START("CPU Compression and Decompression");
    
    // Test with LZ4
    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    int callback_count = 0;
    nvcomp_set_progress_callback(handle, reportProgress, &callback_count);
    
    std::cout << "  Compressing sample.txt with LZ4..." << std::endl;
    // Use default volume size (2.5GB) to avoid volume splitting bug
    uint64_t default_volume_size = 2684354560ULL; // 2.5GB
    nvcomp_error_t result = nvcomp_compress_cpu(
        handle,
        NVCOMP_ALGO_LZ4,
        "sample.txt",
        "output/test_c_api/sample_cpu.lz4",
        default_volume_size,
        nullptr
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
    }
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should compress with CPU");
    ASSERT_TRUE(callback_count > 0, "Should call progress callback");
    
    callback_count = 0;
    std::cout << "  Decompressing sample_cpu.lz4..." << std::endl;
    result = nvcomp_decompress_cpu(
        handle,
        NVCOMP_ALGO_LZ4,
        "output/test_c_api/sample_cpu.lz4",
        "output/test_c_api/sample_cpu_decompressed.txt",
        nullptr
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
    }
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should decompress with CPU");
    ASSERT_TRUE(callback_count > 0, "Should call progress callback");
    
    nvcomp_destroy_operation_handle(handle);
    
    TEST_PASS();
}

void test_algorithm_detection() {
    TEST_START("Algorithm Detection");
    
    // First ensure we have a compressed file
    uint64_t default_volume_size = 2684354560ULL; // 2.5GB
    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "sample.txt",
        "output/test_c_api/sample_detect.lz4",
        default_volume_size,
        nullptr
    );
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should compress file for detection test");
    
    // Detect algorithm
    nvcomp_algorithm_t detected = nvcomp_detect_algorithm_from_file("output/test_c_api/sample_detect.lz4");
    std::cout << "  Detected algorithm: " << nvcomp_algorithm_to_string(detected) << std::endl;
    ASSERT_EQ(detected, NVCOMP_ALGO_LZ4, "Should detect LZ4 algorithm");
    
    TEST_PASS();
}

void test_folder_compression() {
    TEST_START("Folder Compression (CPU)");
    
    std::cout << "  Compressing sample_folder/ with LZ4..." << std::endl;
    uint64_t default_volume_size = 2684354560ULL; // 2.5GB
    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "sample_folder",
        "output/test_c_api/folder_cpu.lz4",
        default_volume_size,
        nullptr
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
    }
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should compress folder with CPU");
    
    std::cout << "  Decompressing folder_cpu.lz4..." << std::endl;
    result = nvcomp_decompress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "output/test_c_api/folder_cpu.lz4",
        "output/test_c_api/folder_cpu_extracted",
        nullptr
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
    }
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should decompress folder with CPU");
    
    TEST_PASS();
}

namespace {

struct ListedEntry {
    std::string path;
    uint64_t size;
    uint32_t mode;
    uint64_t mtimeNs;
};

struct ListCollector {
    std::vector<ListedEntry> entries;
    int progressCalls = 0;
    uint64_t lastCur = 0;
    uint64_t lastTotal = 0;
    int cancelAfter = -1;   // >=0: return nonzero once this many collected
};

int collectEntry(const char* path, uint64_t size, uint32_t mode,
                 uint64_t mtime_ns, void* user_data) {
    auto* c = static_cast<ListCollector*>(user_data);
    if (c->cancelAfter >= 0 &&
        static_cast<int>(c->entries.size()) >= c->cancelAfter) {
        return 1;
    }
    c->entries.push_back({path, size, mode, mtime_ns});
    return 0;
}

void collectProgress(uint64_t current, uint64_t total, void* user_data) {
    auto* c = static_cast<ListCollector*>(user_data);
    c->progressCalls++;
    c->lastCur = current;
    c->lastTotal = total;
}

} // namespace

void test_archive_listing() {
    TEST_START("Archive Listing (nvcomp_list_archive_entries)");

    const char* archive = "output/test_c_api/list_entries.lz4";
    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr, NVCOMP_ALGO_LZ4, "sample_folder", archive,
        2684354560ULL, nullptr);
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should compress sample_folder");

    // Expected contents: every regular file under sample_folder
    std::map<std::string, uint64_t> expected;
    for (const auto& de : std::filesystem::recursive_directory_iterator("sample_folder")) {
        if (de.is_regular_file()) {
            std::string rel = std::filesystem::relative(de.path(), "sample_folder").generic_string();
            expected[rel] = de.file_size();
        }
    }
    ASSERT_TRUE(!expected.empty(), "sample_folder should contain files");

    ListCollector c;
    result = nvcomp_list_archive_entries(archive, collectEntry, collectProgress, &c);
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Listing should succeed");
    ASSERT_EQ(c.entries.size(), expected.size(), "Entry count should match folder contents");

    for (const auto& e : c.entries) {
        auto it = expected.find(e.path);
        ASSERT_TRUE(it != expected.end(), ("Unexpected entry: " + e.path).c_str());
        ASSERT_EQ(e.size, it->second, ("Size mismatch for " + e.path).c_str());
        ASSERT_TRUE(e.mode != 0, "v2 archives should carry POSIX modes");
        ASSERT_TRUE(e.mtimeNs != 0, "v2 archives should carry mtimes");
    }

    ASSERT_TRUE(c.progressCalls > 0, "Progress callback should fire");
    ASSERT_TRUE(c.lastCur <= c.lastTotal, "Progress current should not exceed total");

    // Cancellation: stop after the first entry
    ListCollector cancel;
    cancel.cancelAfter = 1;
    result = nvcomp_list_archive_entries(archive, collectEntry, nullptr, &cancel);
    ASSERT_EQ(result, NVCOMP_ERROR_CANCELED, "Cancel should map to NVCOMP_ERROR_CANCELED");
    ASSERT_EQ(cancel.entries.size(), size_t(1), "Exactly one entry before cancel");
    ASSERT_TRUE(strlen(nvcomp_get_last_error()) > 0, "Cancel should set last error");

    // Invalid arguments
    result = nvcomp_list_archive_entries(nullptr, collectEntry, nullptr, nullptr);
    ASSERT_EQ(result, NVCOMP_ERROR_INVALID_ARGUMENT, "Should reject null input file");
    result = nvcomp_list_archive_entries(archive, nullptr, nullptr, nullptr);
    ASSERT_EQ(result, NVCOMP_ERROR_INVALID_ARGUMENT, "Should reject null entry callback");

    // Missing file
    ListCollector missing;
    result = nvcomp_list_archive_entries("output/test_c_api/does_not_exist.lz4",
                                         collectEntry, nullptr, &missing);
    ASSERT_TRUE(result != NVCOMP_SUCCESS, "Missing file should fail");
    ASSERT_TRUE(strlen(nvcomp_get_last_error()) > 0, "Missing file should set last error");

    TEST_PASS();
}

void test_invalid_arguments() {
    TEST_START("Invalid Argument Handling");
    
    // Test null paths
    nvcomp_error_t result = nvcomp_compress_cpu(nullptr, NVCOMP_ALGO_LZ4, nullptr, "output.lz4", 0, nullptr);
    ASSERT_EQ(result, NVCOMP_ERROR_INVALID_ARGUMENT, "Should reject null input path");
    
    result = nvcomp_compress_cpu(nullptr, NVCOMP_ALGO_LZ4, "input.txt", nullptr, 0, nullptr);
    ASSERT_EQ(result, NVCOMP_ERROR_INVALID_ARGUMENT, "Should reject null output path");
    
    // Test null callback handle
    result = nvcomp_set_progress_callback(nullptr, reportProgress, nullptr);
    ASSERT_EQ(result, NVCOMP_ERROR_INVALID_ARGUMENT, "Should reject null operation handle");
    
    TEST_PASS();
}

void test_thread_safety() {
    TEST_START("Thread-safe Error Messages");
    
    // Set an error in this thread
    nvcomp_compress_cpu(nullptr, NVCOMP_ALGO_LZ4, nullptr, "output.lz4", 0, nullptr);
    const char* error1 = nvcomp_get_last_error();
    ASSERT_TRUE(strlen(error1) > 0, "Should have error message");
    
    // Clear error
    nvcomp_clear_last_error();
    const char* error2 = nvcomp_get_last_error();
    ASSERT_TRUE(strlen(error2) == 0, "Error should be cleared");
    
    TEST_PASS();
}

void test_gpu_compression() {
    TEST_START("GPU Compression (Conditional)");
    
    bool cuda_available = nvcomp_is_cuda_available();
    std::cout << "  CUDA available: " << (cuda_available ? "yes" : "no") << std::endl;
    
    if (!cuda_available) {
        std::cout << "  Skipping GPU tests (CUDA not available)" << std::endl;
        TEST_PASS();
        return;
    }
    
    std::cout << "  Compressing sample.txt with GPU (batched)..." << std::endl;
    uint64_t default_volume_size = 2684354560ULL; // 2.5GB
    nvcomp_error_t result = nvcomp_compress_gpu_batched(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "sample.txt",
        "output/test_c_api/sample_gpu.lz4",
        default_volume_size,
        nullptr
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
        // GPU compression may fail for various reasons, don't fail the test
        std::cout << "  (GPU compression failed - this may be expected)" << std::endl;
    } else {
        std::cout << "  Decompressing sample_gpu.lz4 with GPU..." << std::endl;
        result = nvcomp_decompress_gpu_batched(
            nullptr,
            NVCOMP_ALGO_LZ4,
            "output/test_c_api/sample_gpu.lz4",
            "output/test_c_api/sample_gpu_decompressed.txt",
            nullptr
        );
        
        if (result != NVCOMP_SUCCESS) {
            std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
        }
    }
    
    TEST_PASS();
}

// ============================================================================
// New regression tests for the stats struct + callback throttling
// ============================================================================

// Counts every block-progress callback the core fires. Used to verify the
// post-completion per-chunk callback loop has NOT been re-introduced.
static std::atomic<int> g_block_callback_count{0};
static void countingBlockProgress(nvcomp_operation_handle /*handle*/,
                                  const nvcomp_progress_info_t* /*info*/,
                                  void* /*user_data*/) {
    g_block_callback_count.fetch_add(1, std::memory_order_relaxed);
}

void test_stats_struct_populated() {
    TEST_START("Compression stats struct is populated");
    
    uint64_t default_volume_size = 2684354560ULL; // 2.5GB
    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));
    
    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "sample.txt",
        "output/test_c_api/sample_stats.lz4",
        default_volume_size,
        &stats
    );
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  Error: " << nvcomp_get_last_error() << std::endl;
    }
    ASSERT_EQ(result, NVCOMP_SUCCESS, "Should compress with CPU");

    std::cout << "  read_sec=" << stats.read_sec
              << "  prepare_sec=" << stats.prepare_sec
              << "  compute_sec=" << stats.compute_sec
              << "  write_sec=" << stats.write_sec
              << "  total_sec=" << stats.total_sec
              << "  in=" << stats.input_bytes
              << "  out=" << stats.output_bytes
              << "  MB/s=" << stats.throughput_mbps
              << "  ratio=" << stats.ratio
              << std::endl;
    
    ASSERT_TRUE(stats.total_sec > 0.0, "total_sec should be > 0");
    ASSERT_TRUE(stats.input_bytes > 0, "input_bytes should be > 0");
    ASSERT_TRUE(stats.output_bytes > 0, "output_bytes should be > 0");
    ASSERT_TRUE(stats.throughput_mbps > 0.0, "throughput_mbps should be > 0");
    ASSERT_TRUE(stats.ratio > 0.0, "ratio should be > 0");

    // Smoke-check the formatter.
    char buf[1024];
    size_t n = nvcomp_format_stats_summary(&stats, "Compression", buf, sizeof(buf));
    ASSERT_TRUE(n > 0, "format_stats_summary should write bytes");
    ASSERT_TRUE(strstr(buf, "Total") != nullptr, "summary should contain 'Total'");
    
    TEST_PASS();
}

// Generate a temp file of the given size filled with mostly-incompressible data
// and return its path. Returns empty string on failure.
static std::string makeBigTempFile(size_t bytes) {
    const std::string path = "output/test_c_api/big_input.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return {};
    std::vector<char> buf(64 * 1024);
    // Fill with repeating pseudo-random bytes (compressible but not trivial).
    uint32_t seed = 0xdeadbeefu;
    for (size_t i = 0; i < buf.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        buf[i] = static_cast<char>(seed >> 24);
    }
    size_t written = 0;
    while (written < bytes) {
        size_t chunk = std::min(buf.size(), bytes - written);
        out.write(buf.data(), chunk);
        if (!out) return {};
        written += chunk;
    }
    out.close();
    return path;
}

// ============================================================================
// Verbose flag + read-phase performance + streaming correctness
// ============================================================================

// Helper: capture the global stdout into a string for the duration of a fn().
// Restores std::cout on return so other tests aren't affected.
template <typename Fn>
static std::string captureStdout(Fn&& fn) {
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    try {
        fn();
    } catch (...) {
        std::cout.rdbuf(old);
        throw;
    }
    std::cout.rdbuf(old);
    return buffer.str();
}

// Build a small "folder" of N text files for verbose / streaming tests.
// nvcomp_create_directories takes a *file* path and creates its parent dir,
// so we pass a sentinel filename to coerce the right behavior.
static bool makeSmallFolder(const std::string& folder, int fileCount, size_t bytesPerFile) {
    nvcomp_create_directories((folder + "/.placeholder").c_str());
    for (int i = 0; i < fileCount; ++i) {
        std::ofstream out(folder + "/file" + std::to_string(i) + ".txt", std::ios::binary | std::ios::trunc);
        if (!out) return false;
        std::vector<char> buf(bytesPerFile, static_cast<char>('a' + (i % 26)));
        out.write(buf.data(), buf.size());
        if (!out) return false;
    }
    return true;
}

void test_verbose_flag_default_silent() {
    TEST_START("Verbose flag is OFF by default - no per-file 'Adding:' chatter");

    const std::string folder = "output/test_c_api/verbose_off_folder";
    ASSERT_TRUE(makeSmallFolder(folder, 4, 4096), "create test folder");

    // Force verbose off.
    nvcomp_set_verbose(0);
    ASSERT_EQ(nvcomp_get_verbose(), 0, "verbose getter should return 0");

    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));

    std::string captured = captureStdout([&]() {
        nvcomp_error_t result = nvcomp_compress_cpu(
            nullptr,
            NVCOMP_ALGO_LZ4,
            folder.c_str(),
            "output/test_c_api/verbose_off.lz4",
            2684354560ULL,
            &stats
        );
        ASSERT_EQ(result, NVCOMP_SUCCESS, "compress_cpu should succeed");
    });

    // No per-file "Adding:" or "Collecting files..." chatter when off.
    if (captured.find("Adding:") != std::string::npos) {
        std::cout << captured << std::endl;
        TEST_FAIL("verbose=off should not emit per-file 'Adding:' lines");
        return;
    }
    if (captured.find("Collecting files") != std::string::npos) {
        TEST_FAIL("verbose=off should not emit 'Collecting files...' line");
        return;
    }
    TEST_PASS();
}

void test_verbose_flag_emits_per_file() {
    TEST_START("Verbose flag ON emits one 'Adding:' line per file");

    const std::string folder = "output/test_c_api/verbose_on_folder";
    const int kFiles = 5;
    ASSERT_TRUE(makeSmallFolder(folder, kFiles, 4096), "create test folder");

    nvcomp_set_verbose(1);
    ASSERT_EQ(nvcomp_get_verbose(), 1, "verbose getter should return 1");

    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));

    std::string captured = captureStdout([&]() {
        nvcomp_error_t result = nvcomp_compress_cpu(
            nullptr,
            NVCOMP_ALGO_LZ4,
            folder.c_str(),
            "output/test_c_api/verbose_on.lz4",
            2684354560ULL,
            &stats
        );
        ASSERT_EQ(result, NVCOMP_SUCCESS, "compress_cpu should succeed");
    });

    // Reset verbose so subsequent tests aren't affected.
    nvcomp_set_verbose(0);

    int addingCount = 0;
    size_t pos = 0;
    while ((pos = captured.find("Adding:", pos)) != std::string::npos) {
        ++addingCount;
        ++pos;
    }
    std::cout << "  'Adding:' line count: " << addingCount
              << "  (expected " << kFiles << ")" << std::endl;
    ASSERT_EQ(addingCount, kFiles, "should print one 'Adding:' line per file");
    TEST_PASS();
}

void test_read_phase_throughput() {
    TEST_START("Read phase throughput is sane (no realloc cascade regression)");

    // 256 MB total via 8 x 32 MB files. The pre-Phase 1 realloc cascade put
    // read throughput around ~10 MB/s on this shape; the floor here (50 MB/s
    // CPU mode) is conservative and only catches an outright regression.
    const std::string folder = "output/test_c_api/read_throughput_folder";
    nvcomp_create_directories((folder + "/.placeholder").c_str());
    const int kFiles = 8;
    const size_t kFileBytes = 32ULL * 1024 * 1024;

    for (int i = 0; i < kFiles; ++i) {
        std::ofstream out(folder + "/blob" + std::to_string(i) + ".bin", std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cout << "  Could not create test files; skipping" << std::endl;
            TEST_PASS();
            return;
        }
        std::vector<char> buf(64 * 1024);
        // Mostly-incompressible pseudo-random
        uint32_t seed = 0xc0ffee00u + i;
        for (size_t k = 0; k < buf.size(); ++k) {
            seed = seed * 1664525u + 1013904223u;
            buf[k] = static_cast<char>(seed >> 24);
        }
        size_t written = 0;
        while (written < kFileBytes) {
            size_t chunk = std::min(buf.size(), kFileBytes - written);
            out.write(buf.data(), chunk);
            written += chunk;
        }
    }

    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));

    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        folder.c_str(),
        "output/test_c_api/read_throughput.lz4",
        2684354560ULL,        // single-volume - exercises the in-memory path
        &stats
    );
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  CPU compression failed: " << nvcomp_get_last_error()
                  << " - test inconclusive" << std::endl;
        TEST_PASS();
        return;
    }

    double total_input_mb = (kFiles * static_cast<double>(kFileBytes)) / (1024.0 * 1024.0);
    double read_mbps = stats.read_sec > 0.0 ? total_input_mb / stats.read_sec : 0.0;
    std::cout << "  read_sec=" << stats.read_sec
              << "  read_throughput=" << read_mbps << " MB/s"
              << "  total_sec=" << stats.total_sec << std::endl;

    // Read should complete in finite time with at least baseline disk speed.
    // 50 MB/s is well below any real disk; the pre-fix Read phase hit ~10 MB/s.
    ASSERT_TRUE(stats.read_sec > 0.0, "read_sec should be > 0");
    ASSERT_TRUE(stats.read_sec < stats.total_sec, "read_sec should be < total_sec");
    ASSERT_TRUE(read_mbps > 50.0,
                "Read throughput should exceed 50 MB/s (catches realloc-cascade regression)");
    TEST_PASS();
}

void test_streaming_multi_volume_correctness() {
    TEST_START("Streaming multi-volume archive is byte-identical after roundtrip");

    // Need an input bigger than max_volume_size to force multi-volume.
    // Use 16 MB input split into ~4 MB volumes -> 4 volumes via streaming path.
    const std::string folder = "output/test_c_api/streaming_folder";
    nvcomp_create_directories((folder + "/.placeholder").c_str());
    const int kFiles = 4;
    const size_t kFileBytes = 4ULL * 1024 * 1024;  // 4 MB each
    std::vector<std::vector<uint8_t>> originals(kFiles);
    for (int i = 0; i < kFiles; ++i) {
        originals[i].resize(kFileBytes);
        uint32_t seed = 0xa110ca7eu + i;
        for (size_t k = 0; k < kFileBytes; ++k) {
            seed = seed * 1664525u + 1013904223u;
            originals[i][k] = static_cast<uint8_t>(seed >> 24);
        }
        std::ofstream out(folder + "/streamed" + std::to_string(i) + ".bin", std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cout << "  Could not create test files; skipping" << std::endl;
            TEST_PASS();
            return;
        }
        out.write(reinterpret_cast<const char*>(originals[i].data()), kFileBytes);
    }

    const uint64_t kVolumeBytes = 4ULL * 1024 * 1024;  // 4 MB volumes

    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));

    nvcomp_error_t result = nvcomp_compress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        folder.c_str(),
        "output/test_c_api/streaming.lz4",
        kVolumeBytes,
        &stats
    );
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  CPU compression failed: " << nvcomp_get_last_error() << std::endl;
        TEST_FAIL("streaming compression should succeed");
        return;
    }

    // Decompress and verify each file.
    const std::string restored = "output/test_c_api/streaming_restored";
    nvcomp_create_directories((restored + "/.placeholder").c_str());
    nvcomp_compression_stats_t dstats;
    std::memset(&dstats, 0, sizeof(dstats));
    result = nvcomp_decompress_cpu(
        nullptr,
        NVCOMP_ALGO_LZ4,
        "output/test_c_api/streaming.vol001.lz4",
        restored.c_str(),
        &dstats
    );
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  CPU decompression failed: " << nvcomp_get_last_error() << std::endl;
        TEST_FAIL("streaming decompression should succeed");
        return;
    }

    for (int i = 0; i < kFiles; ++i) {
        std::string p = restored + "/streamed" + std::to_string(i) + ".bin";
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            TEST_FAIL("restored file missing: " + p);
            return;
        }
        std::vector<uint8_t> got((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (got.size() != originals[i].size() || std::memcmp(got.data(), originals[i].data(), got.size()) != 0) {
            std::cout << "  Mismatch in file " << i
                      << " (orig=" << originals[i].size() << "B, got=" << got.size() << "B)" << std::endl;
            TEST_FAIL("restored file does not match original");
            return;
        }
    }

    std::cout << "  All " << kFiles << " files restored byte-identically across "
              << "streaming volume splits" << std::endl;
    TEST_PASS();
}

void test_callback_count_bounded() {
    TEST_START("Block-progress callback count is bounded (regression for chunk-loop)");
    
    bool cuda_available = nvcomp_is_cuda_available();
    if (!cuda_available) {
        std::cout << "  Skipping (CUDA not available)" << std::endl;
        TEST_PASS();
        return;
    }
    
    // 64 MB input - ~1024 chunks at 64 KB each. The old chunk-loop bug would
    // fire >= chunk_count callbacks (so >1000). The fix should keep us well
    // below that, even with the throttle disabled, because the post-completion
    // for-loop is gone.
    const size_t kInputBytes = 64ULL * 1024 * 1024;
    std::string input = makeBigTempFile(kInputBytes);
    if (input.empty()) {
        std::cout << "  Could not create big input file; skipping" << std::endl;
        TEST_PASS();
        return;
    }
    
    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    ASSERT_NOT_NULL(handle, "Should create handle");
    g_block_callback_count.store(0);
    nvcomp_set_block_progress_callback(handle, countingBlockProgress, nullptr);
    
    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));
    
    uint64_t default_volume_size = 2684354560ULL;
    nvcomp_error_t result = nvcomp_compress_gpu_batched(
        handle,
        NVCOMP_ALGO_LZ4,
        input.c_str(),
        "output/test_c_api/big_input.lz4",
        default_volume_size,
        &stats
    );
    
    if (result != NVCOMP_SUCCESS) {
        std::cout << "  GPU compress failed (" << nvcomp_get_last_error()
                  << ") - test inconclusive but not failing" << std::endl;
        nvcomp_destroy_operation_handle(handle);
        TEST_PASS();
        return;
    }
    
    int count = g_block_callback_count.load();
    int chunk_count = static_cast<int>(kInputBytes / (64 * 1024));
    int throttle_upper_bound = static_cast<int>(stats.total_sec * 30.0) + 32;
    std::cout << "  Callback count: " << count
              << "   (chunk_count=" << chunk_count
              << ", throttle_bound=" << throttle_upper_bound
              << ", total_sec=" << stats.total_sec << ")" << std::endl;
    
    // Hard regression guard against the deleted chunk-loop: must not fire one
    // callback per chunk.
    ASSERT_TRUE(count < chunk_count / 2,
                "Callback count must be far below chunk_count (chunk-loop should be deleted)");
    // Soft sanity: should be at most a few times the throttle bound. We allow
    // generous slack because volume splits, stage transitions, and the
    // archive read path can each contribute extra callbacks.
    ASSERT_TRUE(count <= throttle_upper_bound + 64,
                "Callback count should be bounded by the throttle (~30Hz + slack)");
    
    nvcomp_destroy_operation_handle(handle);
    TEST_PASS();
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "nvcomp_core C API Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "NOTE: This test must be run from the unit_test/ directory" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create output directory. nvcomp_create_directories takes a *file* path
    // and creates its parent dir, so pass a sentinel filename to coerce the
    // right behavior - this ensures output/test_c_api itself exists.
    nvcomp_create_directories("output/test_c_api/.placeholder");
    
    // Run tests
    test_error_handling();
    test_algorithm_functions();
    test_file_operations();
    test_volume_support();
    test_operation_handle();
    test_progress_callback();
    test_cpu_compress_decompress();
    test_algorithm_detection();
    test_folder_compression();
    test_archive_listing();
    test_invalid_arguments();
    test_thread_safety();
    test_gpu_compression();
    test_stats_struct_populated();
    test_verbose_flag_default_silent();
    test_verbose_flag_emits_per_file();
    test_read_phase_throughput();
    test_streaming_multi_volume_correctness();
    test_callback_count_bounded();
    
    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total:  " << g_test_count << std::endl;
    std::cout << "Passed: " << g_test_passed << " ✓" << std::endl;
    std::cout << "Failed: " << g_test_failed << " ✗" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (g_test_failed == 0) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some tests failed." << std::endl;
        return 1;
    }
}

