#include "nvcomp_core.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_set>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace nvcomp_core {

// ============================================================================
// Verbose Flag
// ============================================================================

namespace {
std::atomic<bool> g_verbose{false};
} // namespace

void setVerbose(bool verbose) {
    g_verbose.store(verbose, std::memory_order_relaxed);
}

bool isVerbose() {
    return g_verbose.load(std::memory_order_relaxed);
}

// ============================================================================
// Stats Helpers
// ============================================================================

void finalizeStats(CompressionStats& stats) {
    if (stats.totalSec <= 0.0) {
        stats.totalSec = stats.readSec + stats.prepareSec + stats.computeSec + stats.writeSec;
    }
    if (stats.totalSec > 0.0 && stats.inputBytes > 0) {
        double mb = static_cast<double>(stats.inputBytes) / (1024.0 * 1024.0);
        stats.throughputMBps = mb / stats.totalSec;
        stats.throughputGBps = stats.throughputMBps / 1024.0;
    } else {
        stats.throughputMBps = 0.0;
        stats.throughputGBps = 0.0;
    }
    if (stats.outputBytes > 0) {
        stats.ratio = static_cast<double>(stats.inputBytes) / static_cast<double>(stats.outputBytes);
    } else {
        stats.ratio = 0.0;
    }
}

static std::string formatBytesHuman(uint64_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (bytes >= (1ULL << 30)) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
    } else if (bytes >= (1ULL << 20)) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= (1ULL << 10)) {
        oss << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

std::string formatStatsSummary(const CompressionStats& stats, const std::string& opName) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "=== " << opName << " stats ===\n";
    oss << "  Read    : " << stats.readSec    << " s\n";
    oss << "  Prepare : " << stats.prepareSec << " s\n";
    oss << "  Compute : " << stats.computeSec << " s\n";
    oss << "  Write   : " << stats.writeSec   << " s\n";
    oss << "  Total   : " << stats.totalSec   << " s\n";
    oss << "  Input   : " << formatBytesHuman(stats.inputBytes)  << " (" << stats.inputBytes  << " B)\n";
    oss << "  Output  : " << formatBytesHuman(stats.outputBytes) << " (" << stats.outputBytes << " B)\n";
    oss << std::setprecision(2);
    oss << "  Speed   : " << stats.throughputMBps << " MB/s ("
        << stats.throughputGBps << " GB/s)\n";
    if (stats.ratio > 0.0) {
        oss << "  Ratio   : " << stats.ratio << "x";
    }
    return oss.str();
}

// ============================================================================
// Throttled Callback
// ============================================================================

namespace {
struct ThrottleState {
    std::chrono::steady_clock::time_point lastFire = std::chrono::steady_clock::time_point::min();
    int lastPct = -1;
    std::string lastStage;
};
} // namespace

ProgressCallback makeThrottledCallback(ProgressCallback raw, double maxRateHz) {
    if (!raw) return nullptr;
    auto state = std::make_shared<ThrottleState>();
    auto interval = std::chrono::nanoseconds(
        static_cast<int64_t>(1.0e9 / std::max(1.0, maxRateHz)));

    return [raw, state, interval](const BlockProgressInfo& info) {
        const auto now = std::chrono::steady_clock::now();
        const int pct = static_cast<int>(info.overallProgress * 100.0f);
        const bool isTerminal = info.overallProgress >= 1.0f;
        const bool stageChanged = info.stage != state->lastStage;
        const bool pctAdvanced = pct != state->lastPct;
        const bool timeElapsed = (now - state->lastFire) >= interval;

        // Fire when:
        //  - stage changed (always show stage transitions)
        //  - reached 100% (always show completion)
        //  - percent advanced AND enough time has passed
        if (isTerminal || stageChanged || (pctAdvanced && timeElapsed)) {
            state->lastFire = now;
            state->lastPct = pct;
            state->lastStage = info.stage;
            raw(info);
        }
    };
}

// ============================================================================
// File I/O Utilities
// ============================================================================

// Helper overload that accepts fs::path directly (avoids Unicode conversion issues)
std::vector<uint8_t> readFile(const fs::path& filepath) {
    // Use filesystem::path directly to properly handle Unicode paths on Windows
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filepath.string());
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    throw std::runtime_error("Failed to read file: " + filepath.string());
}

// String overload for backward compatibility
std::vector<uint8_t> readFile(const std::string& filename) {
    return readFile(fs::path(filename));
}

// ============================================================================
// Fast direct read into pre-allocated destination
// ============================================================================
//
// Reads exactly `size` bytes from `path` directly into `dst`. For files larger
// than MMAP_THRESHOLD this maps the file into our address space (Windows
// CreateFileMappingW + MapViewOfFile, POSIX mmap+MAP_POPULATE) and memcpy's
// into the caller's buffer. mmap eliminates the kernel-buffer -> user-buffer
// copy that std::ifstream::read incurs and (with MAP_POPULATE / pre-touch)
// turns the read into a single sequential I/O. For small files the syscall
// overhead of mmap dominates so we fall back to std::ifstream::read.
//
// Errors (open fail, mapping fail, partial read) cleanly fall through to the
// std::ifstream path before throwing - the caller never sees a partial buffer.
namespace {
constexpr uint64_t MMAP_THRESHOLD = 16 * 1024 * 1024; // 16 MB

// std::ifstream fallback used by readFileInto() and as last resort on mmap fail.
void readFileIntoStream(const fs::path& path, uint8_t* dst, uint64_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    if (size == 0) return;
    if (!file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Failed to read file: " + path.string());
    }
}
} // namespace

void readFileInto(const fs::path& path, uint8_t* dst, uint64_t size) {
    if (size == 0) return;
    if (size < MMAP_THRESHOLD) {
        readFileIntoStream(path, dst, size);
        return;
    }

#ifdef _WIN32
    // Use the wide-character API so Unicode paths (which fs::path already
    // holds as wide on Windows) survive intact.
    HANDLE hFile = CreateFileW(path.wstring().c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                               nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        readFileIntoStream(path, dst, size);
        return;
    }
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        CloseHandle(hFile);
        readFileIntoStream(path, dst, size);
        return;
    }
    void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, static_cast<SIZE_T>(size));
    if (!view) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        readFileIntoStream(path, dst, size);
        return;
    }
    // Hint the kernel to fault the whole range in eagerly so the memcpy below
    // is one sequential read rather than thousands of demand-paged faults.
    // Available since Windows 8 / Server 2012; ignore failure (best-effort).
    typedef BOOL (WINAPI *PrefetchVirtualMemoryFn)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static PrefetchVirtualMemoryFn pfnPrefetch = []() -> PrefetchVirtualMemoryFn {
        HMODULE hKernel = GetModuleHandleW(L"kernel32.dll");
        return hKernel ? reinterpret_cast<PrefetchVirtualMemoryFn>(
            GetProcAddress(hKernel, "PrefetchVirtualMemory")) : nullptr;
    }();
    if (pfnPrefetch) {
        WIN32_MEMORY_RANGE_ENTRY range{view, static_cast<SIZE_T>(size)};
        pfnPrefetch(GetCurrentProcess(), 1, &range, 0);
    }
    std::memcpy(dst, view, static_cast<size_t>(size));
    UnmapViewOfFile(view);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        readFileIntoStream(path, dst, size);
        return;
    }
    // MAP_POPULATE pre-faults pages so the memcpy is one sequential read.
    // On platforms that don't have it (older POSIX), the flag is just ignored.
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    void* view = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, flags, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        readFileIntoStream(path, dst, size);
        return;
    }
    std::memcpy(dst, view, static_cast<size_t>(size));
    ::munmap(view, static_cast<size_t>(size));
    ::close(fd);
#endif
}

// Helper overload that accepts fs::path directly (avoids Unicode conversion issues)
void writeFile(const fs::path& filepath, const void* data, size_t size, ProgressCallback callback) {
    // Use filesystem::path directly to properly handle Unicode paths on Windows
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filepath.string());
    }
    
    // If no callback or small file, write all at once
    if (!callback || size < 1024 * 1024) {  // Less than 1MB
        file.write(reinterpret_cast<const char*>(data), size);
        return;
    }
    
    // Write in chunks and report progress
    const size_t WRITE_CHUNK_SIZE = 64 * 1024 * 1024;  // 64MB chunks for writing
    const uint8_t* dataPtr = reinterpret_cast<const uint8_t*>(data);
    size_t remaining = size;
    size_t written = 0;
    
    while (remaining > 0) {
        size_t chunkSize = std::min(WRITE_CHUNK_SIZE, remaining);
        file.write(reinterpret_cast<const char*>(dataPtr + written), chunkSize);
        
        written += chunkSize;
        remaining -= chunkSize;
        
        // Report progress (map to 75%-100% range)
        float writeProgress = static_cast<float>(written) / size;
        BlockProgressInfo info;
        info.totalBlocks = 1;
        info.completedBlocks = (remaining == 0) ? 1 : 0;
        info.currentBlock = 0;
        info.currentBlockSize = size;
        info.overallProgress = 0.75f + (writeProgress * 0.25f);  // 75% to 100%
        info.currentBlockProgress = writeProgress;
        info.throughputMBps = 0.0;
        info.stage = "writing";
        callback(info);
    }
}

void writeFile(const fs::path& filepath, const void* data, size_t size) {
    writeFile(filepath, data, size, nullptr);
}

// String overloads for backward compatibility
void writeFile(const std::string& filename, const void* data, size_t size) {
    writeFile(fs::path(filename), data, size, nullptr);
}

void writeFile(const std::string& filename, const void* data, size_t size, ProgressCallback callback) {
    writeFile(fs::path(filename), data, size, callback);
}

std::string normalizePath(const std::string& path) {
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

// fs::path overload for proper Unicode handling
std::string normalizePath(const fs::path& path) {
    // Use u8string() to get UTF-8 encoding which preserves Unicode characters
    std::string normalized = path.generic_u8string();
    return normalized;
}

// fs::path overload for proper Unicode handling
std::string getRelativePath(const fs::path& path, const fs::path& base) {
    fs::path relativePath = fs::relative(path, base);
    // Use u8string() to properly preserve Unicode characters
    std::string u8str = relativePath.u8string();
    return normalizePath(u8str);
}

// String overload for backward compatibility
std::string getRelativePath(const std::string& path, const std::string& base) {
    return getRelativePath(fs::path(path), fs::path(base));
}

bool isDirectory(const std::string& path) {
    try {
        return fs::is_directory(path);
    } catch (...) {
        return false;
    }
}

void createDirectories(const std::string& path) {
    fs::path fsPath(path);
    if (!fsPath.empty() && fsPath.has_parent_path()) {
        fs::create_directories(fsPath.parent_path());
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<fs::path> collectFiles(const fs::path& dirPath) {
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

// ============================================================================
// Entry Enumeration (for streaming pipeline)
// ============================================================================

std::vector<ArchiveEntry> collectArchiveEntries(const std::string& folderOrFile) {
    std::vector<ArchiveEntry> entries;
    fs::path p(folderOrFile);

    if (!fs::exists(p)) {
        throw std::runtime_error("Path does not exist: " + folderOrFile);
    }

    if (fs::is_regular_file(p)) {
        ArchiveEntry e;
        e.filePath = p;
        e.relativePath = p.filename().string();
        e.fileSize = fs::file_size(p);
        entries.push_back(std::move(e));
        return entries;
    }

    if (!fs::is_directory(p)) {
        throw std::runtime_error("Not a regular file or directory: " + folderOrFile);
    }

    auto files = collectFiles(p);
    entries.reserve(files.size());
    for (const auto& f : files) {
        ArchiveEntry e;
        e.filePath = f;
        e.relativePath = getRelativePath(f, p);
        if (e.relativePath.empty() || e.relativePath == ".") {
            e.relativePath = f.filename().string();
        }
        e.fileSize = fs::file_size(f);
        entries.push_back(std::move(e));
    }
    return entries;
}

std::vector<ArchiveEntry> collectArchiveEntriesFromList(const std::vector<std::string>& filePaths) {
    std::vector<ArchiveEntry> entries;

    for (const auto& itemPath : filePaths) {
        fs::path p(itemPath);
        if (!fs::exists(p)) {
            std::cerr << "Warning: Skipping non-existent path: " << itemPath << std::endl;
            continue;
        }
        if (fs::is_regular_file(p)) {
            // Single file - use parent directory as base (matches createArchiveFromFileList).
            ArchiveEntry e;
            e.filePath = p;
            e.relativePath = getRelativePath(p, p.parent_path());
            if (e.relativePath.empty() || e.relativePath == ".") {
                e.relativePath = p.filename().string();
            }
            e.fileSize = fs::file_size(p);
            entries.push_back(std::move(e));
        } else if (fs::is_directory(p)) {
            auto files = collectFiles(p);
            for (const auto& f : files) {
                ArchiveEntry e;
                e.filePath = f;
                e.relativePath = getRelativePath(f, p);
                if (e.relativePath.empty() || e.relativePath == ".") {
                    e.relativePath = f.filename().string();
                }
                e.fileSize = fs::file_size(f);
                entries.push_back(std::move(e));
            }
        }
    }
    return entries;
}

// ============================================================================
// Archive Creation
// ============================================================================

std::vector<uint8_t> createArchiveFromFolder(const std::string& folderPath, ProgressCallback callback) {
    std::vector<uint8_t> archiveData;
    fs::path basePath(folderPath);
    
    if (!isDirectory(folderPath)) {
        throw std::runtime_error("Not a directory: " + folderPath);
    }
    
    std::vector<fs::path> files = collectFiles(basePath);
    if (isVerbose()) {
        std::cout << "Collecting files from directory: " << folderPath << "\n";
        std::cout << "Found " << files.size() << " file(s)\n";
    }
    
    if (files.empty()) {
        throw std::runtime_error("No files to archive");
    }
    
    // Pre-compute exact archive size and per-file metadata so we can:
    //   1. reserve archiveData once (no realloc cascade), and
    //   2. read each file directly into the live tail of archiveData (no
    //      intermediate temp vector / second memcpy).
    struct PreparedEntry {
        fs::path filePath;
        std::string relativePath;
        uint64_t fileSize;
    };
    std::vector<PreparedEntry> entries;
    entries.reserve(files.size());

    uint64_t totalSize = 0;
    size_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& filePath : files) {
        std::string relativePath = getRelativePath(filePath, basePath);
        if (relativePath.empty() || relativePath == ".") {
            relativePath = filePath.filename().string();
        }
        uint64_t fsize = fs::file_size(filePath);
        totalSize += fsize;
        totalArchiveSize += sizeof(FileEntry) + relativePath.size() + fsize;
        entries.push_back({filePath, std::move(relativePath), fsize});
    }
    archiveData.reserve(totalArchiveSize);

    // Write header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = static_cast<uint32_t>(entries.size());
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));
    
    // Write each file
    uint64_t processedSize = 0;
    const bool verbose = isVerbose();
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];

        FileEntry entry;
        entry.pathLength = static_cast<uint32_t>(e.relativePath.length());
        entry.fileSize = e.fileSize;
        
        const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
        archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
        archiveData.insert(archiveData.end(), e.relativePath.begin(), e.relativePath.end());

        // Read file data directly into archiveData's tail. Since we reserved
        // totalArchiveSize up front, this resize() does not reallocate. Use
        // mmap for large files (no kernel->user copy) and ifstream for small.
        size_t writeOffset = archiveData.size();
        archiveData.resize(writeOffset + e.fileSize);
        readFileInto(e.filePath, archiveData.data() + writeOffset, e.fileSize);
        
        processedSize += e.fileSize;
        if (verbose) {
            std::cout << "  Adding: " << e.relativePath
                      << " (" << e.fileSize << " bytes)\n";
        }
        
        // Report progress (0-25% range for reading)
        if (callback && totalSize > 0) {
            float readProgress = static_cast<float>(processedSize) / totalSize;
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(entries.size());
            info.completedBlocks = static_cast<int>(i + 1);
            info.currentBlock = static_cast<int>(i);
            info.currentBlockSize = e.fileSize;
            info.overallProgress = readProgress * 0.25f;  // Scale to 0-25%
            info.currentBlockProgress = 1.0f;
            info.throughputMBps = 0.0;
            info.stage = "reading";
            callback(info);
        }
    }
    
    return archiveData;
}

std::vector<uint8_t> createArchiveFromFile(const std::string& filePath, ProgressCallback callback) {
    std::vector<uint8_t> archiveData;
    
    fs::path p(filePath);
    if (!fs::exists(p) || !fs::is_regular_file(p)) {
        throw std::runtime_error("File does not exist or is not a regular file: " + filePath);
    }
    
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Adding single file: " << filePath << "\n";
    }

    std::string filename = p.filename().string();
    uint64_t fileSize = fs::file_size(p);

    // Reserve once for header + entry + path + file data. No realloc.
    size_t totalArchiveSize = sizeof(ArchiveHeader)
                              + sizeof(FileEntry)
                              + filename.size()
                              + fileSize;
    archiveData.reserve(totalArchiveSize);

    // Write archive header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = 1;
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));

    // Write file entry header + path
    FileEntry entry;
    entry.pathLength = static_cast<uint32_t>(filename.length());
    entry.fileSize = fileSize;
    
    const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
    archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
    archiveData.insert(archiveData.end(), filename.begin(), filename.end());

    // Read file bytes directly into archiveData (no temp vector / second copy)
    size_t writeOffset = archiveData.size();
    archiveData.resize(writeOffset + fileSize);
    readFileInto(p, archiveData.data() + writeOffset, fileSize);

    if (verbose) {
        std::cout << "  Added: " << filename << " (" << fileSize << " bytes)\n";
    }
    
    // Report progress (0-25% range for reading)
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = 1;
        info.completedBlocks = 1;
        info.currentBlock = 0;
        info.currentBlockSize = fileSize;
        info.overallProgress = 0.25f;  // Complete reading phase
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = 0.0;
        info.stage = "reading";
        callback(info);
    }
    
    return archiveData;
}

std::vector<uint8_t> createArchiveFromFileList(const std::vector<std::string>& filePaths, ProgressCallback callback) {
    std::vector<uint8_t> archiveData;
    
    if (filePaths.empty()) {
        throw std::runtime_error("No files to archive");
    }
    
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Creating archive from " << filePaths.size() << " item(s)\n";
    }
    
    // Collect all files, expanding directories recursively
    struct FileWithBase {
        fs::path filePath;
        fs::path basePath;  // For calculating relative paths
    };
    
    std::vector<FileWithBase> allFiles;
    
    for (const auto& itemPath : filePaths) {
        fs::path p(itemPath);
        
        if (!fs::exists(p)) {
            std::cerr << "Warning: Skipping non-existent path: " << itemPath << std::endl;
            continue;
        }
        
        if (fs::is_regular_file(p)) {
            // Single file - use parent directory as base
            allFiles.push_back({p, p.parent_path()});
        } else if (fs::is_directory(p)) {
            // Directory - collect all files recursively
            auto dirFiles = collectFiles(p);
            for (const auto& file : dirFiles) {
                allFiles.push_back({file, p});  // Use the directory itself as base
            }
        }
    }
    
    if (allFiles.empty()) {
        throw std::runtime_error("No files found to archive");
    }
    
    if (verbose) {
        std::cout << "Total files to archive: " << allFiles.size() << "\n";
    }

    // Pre-compute exact archive size and per-file metadata so we can:
    //   1. reserve archiveData once (no realloc cascade), and
    //   2. read each file directly into the live tail of archiveData.
    struct PreparedEntry {
        fs::path filePath;
        std::string relativePath;
        uint64_t fileSize;
    };
    std::vector<PreparedEntry> entries;
    entries.reserve(allFiles.size());

    uint64_t totalSize = 0;
    size_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& fileWithBase : allFiles) {
        std::string relativePath = getRelativePath(fileWithBase.filePath, fileWithBase.basePath);
        if (relativePath.empty() || relativePath == ".") {
            relativePath = fileWithBase.filePath.filename().string();
        }
        uint64_t fsize = fs::file_size(fileWithBase.filePath);
        totalSize += fsize;
        totalArchiveSize += sizeof(FileEntry) + relativePath.size() + fsize;
        entries.push_back({fileWithBase.filePath, std::move(relativePath), fsize});
    }
    archiveData.reserve(totalArchiveSize);

    // Write archive header
    ArchiveHeader header;
    header.magic = ARCHIVE_MAGIC;
    header.version = ARCHIVE_VERSION;
    header.fileCount = static_cast<uint32_t>(entries.size());
    header.reserved = 0;
    
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header);
    archiveData.insert(archiveData.end(), headerBytes, headerBytes + sizeof(ArchiveHeader));
    
    // Write each file
    uint64_t processedSize = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];

        FileEntry entry;
        entry.pathLength = static_cast<uint32_t>(e.relativePath.length());
        entry.fileSize = e.fileSize;
        
        const uint8_t* entryBytes = reinterpret_cast<const uint8_t*>(&entry);
        archiveData.insert(archiveData.end(), entryBytes, entryBytes + sizeof(FileEntry));
        archiveData.insert(archiveData.end(), e.relativePath.begin(), e.relativePath.end());

        // Read file bytes directly into archiveData's tail (no realloc, no temp copy)
        size_t writeOffset = archiveData.size();
        archiveData.resize(writeOffset + e.fileSize);
        readFileInto(e.filePath, archiveData.data() + writeOffset, e.fileSize);
        
        processedSize += e.fileSize;
        if (verbose) {
            std::cout << "  Adding: " << e.relativePath
                      << " (" << e.fileSize << " bytes)\n";
        }
        
        // Report progress (0-25% range for reading)
        if (callback && totalSize > 0) {
            float readProgress = static_cast<float>(processedSize) / totalSize;
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(entries.size());
            info.completedBlocks = static_cast<int>(i + 1);
            info.currentBlock = static_cast<int>(i);
            info.currentBlockSize = e.fileSize;
            info.overallProgress = readProgress * 0.25f;  // Scale to 0-25%
            info.currentBlockProgress = 1.0f;
            info.throughputMBps = 0.0;
            info.stage = "reading";
            callback(info);
        }
    }
    
    return archiveData;
}

// ============================================================================
// Archive Extraction
// ============================================================================

// ============================================================================
// Streaming archive extraction
// ============================================================================

// Lean single-shot file write for extraction workers. On POSIX this is plain
// open/write/close (measured ~equal to ofstream single-threaded but scales to
// ~4x with 8 workers); elsewhere it falls back to the existing writeFile.
static void writeFileFast(const fs::path& path, const uint8_t* data, size_t n) {
#ifdef _WIN32
    writeFile(path, data, n);
#else
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Failed to create output file: " + path.string());
    }
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::write(fd, data + off, n - off);
        if (w <= 0) {
            ::close(fd);
            throw std::runtime_error("Failed to write output file: " + path.string());
        }
        off += static_cast<size_t>(w);
    }
    ::close(fd);
#endif
}

struct ArchiveExtractor::Impl {
    enum class State { Header, Entry, Path, Data, Done };

    // Holds a feed's releaseBuffer callback; fires when the last reference
    // (the feed() call itself or any write task using the buffer) drops.
    struct FeedGuard {
        std::function<void()> release;
        ~FeedGuard() { if (release) release(); }
    };

    struct Task {
        fs::path path;
        const uint8_t* data;
        size_t n;
        std::shared_ptr<FeedGuard> guard;
    };

    explicit Impl(const std::string& outputPath, size_t writerThreads)
        : outputPath_(outputPath), verbose_(isVerbose()) {
        for (size_t i = 0; i < writerThreads; i++) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~Impl() { stopWorkers(); }

    // ---- parsing -----------------------------------------------------------

    // Accumulate exactly `need` contiguous bytes; returns nullptr until enough
    // input has arrived. Fast path: parse in place with no copy.
    const uint8_t* fill(size_t need, const uint8_t*& p, size_t& n) {
        if (carry_.empty() && n >= need) {
            const uint8_t* r = p;
            p += need;
            n -= need;
            return r;
        }
        size_t take = std::min(need - carry_.size(), n);
        carry_.insert(carry_.end(), p, p + take);
        p += take;
        n -= take;
        return carry_.size() == need ? carry_.data() : nullptr;
    }

    void feed(const uint8_t* data, size_t n, std::function<void()> releaseBuffer) {
        auto guard = std::make_shared<FeedGuard>();
        guard->release = std::move(releaseBuffer);
        checkWorkerError();

        const uint8_t* p = data;
        while (n > 0) {
            switch (state_) {
            case State::Header: {
                const uint8_t* b = fill(sizeof(ArchiveHeader), p, n);
                if (!b) break;
                std::memcpy(&header_, b, sizeof(ArchiveHeader));
                carry_.clear();
                if (header_.magic != ARCHIVE_MAGIC) {
                    throw std::runtime_error("Invalid archive: bad magic number");
                }
                if (header_.version != ARCHIVE_VERSION) {
                    throw std::runtime_error("Unsupported archive version");
                }
                if (verbose_) {
                    std::cout << "Extracting " << header_.fileCount
                              << " file(s) to: " << outputPath_ << "\n";
                }
                if (!outputPath_.empty()) {
                    fs::create_directories(outputPath_);
                }
                state_ = header_.fileCount == 0 ? State::Done : State::Entry;
                break;
            }
            case State::Entry: {
                const uint8_t* b = fill(sizeof(FileEntry), p, n);
                if (!b) break;
                std::memcpy(&entry_, b, sizeof(FileEntry));
                carry_.clear();
                if (entry_.pathLength == 0 || entry_.pathLength > (1u << 20)) {
                    throw std::runtime_error("Invalid archive: bad path length");
                }
                state_ = State::Path;
                break;
            }
            case State::Path: {
                const uint8_t* b = fill(entry_.pathLength, p, n);
                if (!b) break;
                std::string rel(reinterpret_cast<const char*>(b), entry_.pathLength);
                carry_.clear();
                if (verbose_) {
                    std::cout << "  Extracting: " << rel
                              << " (" << entry_.fileSize << " bytes)\n";
                }
                fullPath_ = fs::path(outputPath_) / fs::path(rel);
                makeParentDirs(fullPath_);
                if (entry_.fileSize == 0) {
                    dispatch(fullPath_, nullptr, 0, guard);
                    fileDone();
                } else {
                    dataRemaining_ = entry_.fileSize;
                    state_ = State::Data;
                }
                break;
            }
            case State::Data: {
                size_t take = static_cast<size_t>(
                    std::min<uint64_t>(dataRemaining_, n));
                if (!spanFile_.is_open() && take == dataRemaining_) {
                    // Whole (rest of) file inside this feed: pool-writable.
                    dispatch(fullPath_, p, take, guard);
                } else {
                    // File spans feeds: write inline through a kept-open
                    // stream (only large files hit this; no lifetime issues).
                    if (!spanFile_.is_open()) {
                        spanFile_.open(fullPath_, std::ios::binary | std::ios::trunc);
                        if (!spanFile_.is_open()) {
                            throw std::runtime_error(
                                "Failed to create output file: " + fullPath_.string());
                        }
                    }
                    spanFile_.write(reinterpret_cast<const char*>(p),
                                    static_cast<std::streamsize>(take));
                    if (!spanFile_) {
                        throw std::runtime_error(
                            "Failed to write output file: " + fullPath_.string());
                    }
                    bytesWritten_.fetch_add(take, std::memory_order_relaxed);
                }
                p += take;
                n -= take;
                dataRemaining_ -= take;
                if (dataRemaining_ == 0) {
                    if (spanFile_.is_open()) spanFile_.close();
                    fileDone();
                }
                break;
            }
            case State::Done:
                throw std::runtime_error(
                    "Invalid archive: data past the last file entry");
            }
            // Note: when fill() returns nullptr it has consumed all of n,
            // so the loop terminates naturally on partial fields.
        }
    }

    void fileDone() {
        filesDone_++;
        state_ = filesDone_ == header_.fileCount ? State::Done : State::Entry;
    }

    void makeParentDirs(const fs::path& fullPath) {
        if (!fullPath.has_parent_path()) return;
        std::string parent = fullPath.parent_path().string();
        if (parent == lastParent_) return;               // consecutive-file fast path
        if (dirsMade_.insert(parent).second) {
            fs::create_directories(fullPath.parent_path());
        }
        lastParent_ = std::move(parent);
    }

    // ---- writing -----------------------------------------------------------

    void dispatch(const fs::path& path, const uint8_t* data, size_t n,
                  const std::shared_ptr<FeedGuard>& guard) {
        if (workers_.empty()) {
            writeFileFast(path, data, n);
            bytesWritten_.fetch_add(n, std::memory_order_relaxed);
            filesWritten_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::unique_lock<std::mutex> lk(mtx_);
        cvSpace_.wait(lk, [&] {
            return tasks_.size() < kMaxQueuedTasks || abort_.load();
        });
        if (abort_.load()) {
            lk.unlock();
            checkWorkerError();
            return;
        }
        tasks_.push_back(Task{path, data, n, guard});
        outstanding_++;
        lk.unlock();
        cvWork_.notify_one();
    }

    void workerLoop() {
        for (;;) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cvWork_.wait(lk, [&] { return !tasks_.empty() || stop_.load(); });
                if (tasks_.empty()) return;
                t = std::move(tasks_.front());
                tasks_.pop_front();
            }
            cvSpace_.notify_one();
            if (!abort_.load()) {
                try {
                    writeFileFast(t.path, t.data, t.n);
                    bytesWritten_.fetch_add(t.n, std::memory_order_relaxed);
                    filesWritten_.fetch_add(1, std::memory_order_relaxed);
                } catch (...) {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (!workerError_) workerError_ = std::current_exception();
                    abort_.store(true);
                }
            }
            // t.guard drops here, releasing the feed buffer when last user.
            {
                std::lock_guard<std::mutex> lk(mtx_);
                outstanding_--;
            }
            cvIdle_.notify_all();
        }
    }

    void checkWorkerError() {
        if (!abort_.load()) return;
        std::lock_guard<std::mutex> lk(mtx_);
        if (workerError_) std::rethrow_exception(workerError_);
        throw std::runtime_error("Extraction aborted");
    }

    void drainWorkers() {
        std::unique_lock<std::mutex> lk(mtx_);
        cvIdle_.wait(lk, [&] { return outstanding_ == 0; });
    }

    void stopWorkers() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_.store(true);
        }
        cvWork_.notify_all();
        cvSpace_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    void finish() {
        drainWorkers();
        stopWorkers();
        checkWorkerError();
        if (state_ == State::Header) {
            throw std::runtime_error("Invalid archive: too small");
        }
        if (state_ != State::Done || !carry_.empty() || spanFile_.is_open()) {
            throw std::runtime_error("Invalid archive: truncated file data");
        }
        if (verbose_) {
            std::cout << "Extraction complete.\n";
        }
    }

    void abandon() {
        abort_.store(true);
        stopWorkers();
        if (spanFile_.is_open()) spanFile_.close();
    }

    // parsing state
    std::string outputPath_;
    bool verbose_;
    State state_ = State::Header;
    std::vector<uint8_t> carry_;
    ArchiveHeader header_{};
    FileEntry entry_{};
    fs::path fullPath_;
    uint64_t dataRemaining_ = 0;
    uint32_t filesDone_ = 0;
    std::ofstream spanFile_;
    std::unordered_set<std::string> dirsMade_;
    std::string lastParent_;

    // writer pool
    static constexpr size_t kMaxQueuedTasks = 8192;
    std::vector<std::thread> workers_;
    std::deque<Task> tasks_;
    std::mutex mtx_;
    std::condition_variable cvWork_, cvSpace_, cvIdle_;
    size_t outstanding_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> abort_{false};
    std::exception_ptr workerError_;

    std::atomic<uint64_t> bytesWritten_{0};
    std::atomic<uint32_t> filesWritten_{0};
};

ArchiveExtractor::ArchiveExtractor(const std::string& outputPath, size_t writerThreads)
    : impl_(std::make_unique<Impl>(outputPath, writerThreads)) {}
ArchiveExtractor::~ArchiveExtractor() = default;

void ArchiveExtractor::feed(const uint8_t* data, size_t n,
                            std::function<void()> releaseBuffer) {
    impl_->feed(data, n, std::move(releaseBuffer));
}
void ArchiveExtractor::finish() { impl_->finish(); }
void ArchiveExtractor::abandon() { impl_->abandon(); }
uint64_t ArchiveExtractor::bytesWritten() const {
    return impl_->bytesWritten_.load(std::memory_order_relaxed);
}
uint32_t ArchiveExtractor::filesWritten() const {
    return impl_->filesWritten_.load(std::memory_order_relaxed);
}

// Legacy whole-buffer extraction, now a thin wrapper over the streaming
// extractor so there is exactly one archive parser. Synchronous writes
// (writerThreads = 0) preserve the original behavior.
void extractArchive(const std::vector<uint8_t>& archiveData, const std::string& outputPath) {
    if (archiveData.size() < sizeof(ArchiveHeader)) {
        throw std::runtime_error("Invalid archive: too small");
    }
    ArchiveExtractor ex(outputPath, 0);
    ex.feed(archiveData.data(), archiveData.size());
    ex.finish();
}

// ============================================================================
// Archive Listing
// ============================================================================

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

} // namespace nvcomp_core



