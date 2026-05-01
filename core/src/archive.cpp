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
    
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Extracting " << header.fileCount << " file(s) to: " << outputPath << "\n";
    }
    
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
        
        if (verbose) {
            std::cout << "  Extracting: " << filePath << " (" << entry.fileSize << " bytes)\n";
        }
        
        // Construct full output path
        fs::path fullPath = fs::path(outputPath) / fs::path(filePath);
        
        // Create parent directories
        createDirectories(fullPath.string());
        
        // Write file
        writeFile(fullPath, archiveData.data() + offset, entry.fileSize);
        offset += entry.fileSize;
    }
    
    if (verbose) {
        std::cout << "Extraction complete.\n";
    }
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



