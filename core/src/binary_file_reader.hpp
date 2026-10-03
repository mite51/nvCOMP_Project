#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace nvcomp_core {
// Bulk binary reads avoid the 4095-byte fread loop in MSVC 19.38's
// basic_filebuf::xsgetn. Keep the native wide path on Windows.
class BinaryFileReader {
public:
    explicit BinaryFileReader(const std::filesystem::path& path) : path_(path) {
#ifdef _WIN32
        file_ = _wfopen(path.c_str(), L"rb");
#else
        file_ = std::fopen(path.c_str(), "rb");
#endif
        if (!file_) throw std::runtime_error("Failed to open input file: " + path.string());
    }
    ~BinaryFileReader() { if (file_) std::fclose(file_); }
    BinaryFileReader(const BinaryFileReader&) = delete;
    BinaryFileReader& operator=(const BinaryFileReader&) = delete;
    void read(uint8_t* destination, uint64_t bytes) {
        while (bytes) {
            const size_t take = static_cast<size_t>(std::min<uint64_t>(bytes, 64ull << 20));
            if (std::fread(destination, 1, take, file_) != take)
                throw std::runtime_error("Failed to read file: " + path_.string());
            destination += take;
            bytes -= take;
        }
    }
private:
    std::filesystem::path path_;
    FILE* file_ = nullptr;
};
}
