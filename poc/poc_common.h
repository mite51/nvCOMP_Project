// Shared helpers for the optimization-pass proof-of-concept micro-benchmarks.
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(_e));                        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define NVCOMP_CHECK(call)                                                    \
    do {                                                                      \
        nvcompStatus_t _s = (call);                                           \
        if (_s != nvcompSuccess) {                                            \
            fprintf(stderr, "nvCOMP error %s at %s:%d: status=%d\n", #call,   \
                    __FILE__, __LINE__, (int)_s);                             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

struct Timer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double sec() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

// Deterministic pseudo-random fill (xorshift), ~50% compressible pattern mix.
inline void fill_mixed(uint8_t* p, size_t n, uint64_t seed = 0x9e3779b97f4a7c15ULL) {
    uint64_t x = seed;
    for (size_t i = 0; i < n; i += 8) {
        // alternate 1MB stripes: random / repetitive
        if ((i >> 20) & 1) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            memcpy(p + i, &x, (n - i) < 8 ? (n - i) : 8);
        } else {
            uint64_t v = 0x4141414141414141ULL + (i >> 12);
            memcpy(p + i, &v, (n - i) < 8 ? (n - i) : 8);
        }
    }
}

inline std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return buf;
}
