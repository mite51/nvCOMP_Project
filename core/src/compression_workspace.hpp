#pragma once
#include "cuda_buffers.hpp"
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>

namespace nvcomp_core {

struct CompressionSlot {
    PinnedBuffer h_in, h_out, h_sizes, h_offsets;
    DeviceBuffer d_in, d_out, d_temp;
    DeviceBuffer d_in_ptrs, d_in_sizes, d_out_ptrs, d_out_sizes, d_offsets;
    std::unique_ptr<CudaStream> stream;
    std::unique_ptr<CudaEvent> computeDone;
    bool defaultSizes = true;
    size_t bytes = 0, volumeIdx = 0, packedBytes = 0;
    bool lastInVolume = false;
};

struct CompressionWorkspace {
    int device = 0, algorithm = 0;
    size_t chunks = 0, depth = 0, maxOutput = 0, scratch = 0;
    std::vector<CompressionSlot> slots;
    size_t deviceBytes() const {
        size_t n = 0;
        for (const auto& s : slots)
            n += s.d_in.size() + s.d_out.size() + s.d_temp.size()
               + s.d_in_ptrs.size() + s.d_in_sizes.size() + s.d_out_ptrs.size()
               + s.d_out_sizes.size() + s.d_offsets.size();
        return n;
    }
    size_t hostBytes() const {
        size_t n = 0;
        for (const auto& s : slots)
            n += s.h_in.size() + s.h_out.size() + s.h_sizes.size() + s.h_offsets.size();
        return n;
    }
    ~CompressionWorkspace() {
        // An idle workspace can be evicted by a caller on another CUDA device.
        int previous = device;
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
        slots.clear();
        if (previous != device) cudaSetDevice(previous);
    }
};

// Opt-in process cache, containing at most ONE idle workspace, not one per
// thread/codec/device. Active jobs exclusively own their leases. Large shapes
// still run but are not retained. Configure the environment before starting jobs.
struct CompressionWorkspaceCache {
    std::mutex mutex;
    std::unique_ptr<CompressionWorkspace> idle;
    uint64_t generation = 0;
};
inline CompressionWorkspaceCache& compressionWorkspaceCache() {
    static CompressionWorkspaceCache cache;
    return cache;
}

struct CompressionWorkspaceLease {
    std::unique_ptr<CompressionWorkspace> workspace;
    uint64_t generation = 0;
    bool enabled = false, hit = false;

    CompressionWorkspaceLease(int device, int algorithm, size_t chunks,
                              size_t depth, size_t maxOutput, size_t scratch) {
        const char* value = std::getenv("NVCOMP_REUSE_BUFFERS");
        enabled = value && std::strcmp(value, "1") == 0;
        auto& cache = compressionWorkspaceCache();
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            generation = cache.generation;
            workspace = std::move(cache.idle);
        }
        hit = enabled && workspace && workspace->device == device
            && workspace->algorithm == algorithm && workspace->chunks == chunks
            && workspace->depth == depth && workspace->maxOutput == maxOutput
            && workspace->scratch == scratch;
        // Release mismatched allocations BEFORE the free-VRAM sizing query.
        if (!hit) workspace.reset();
    }

    void retainSuccessful() {
        if (!enabled || !workspace || workspace->deviceBytes() > (4ull << 30)
            || workspace->hostBytes() > (1ull << 30)) return;
        auto& cache = compressionWorkspaceCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (generation == cache.generation && !cache.idle)
            cache.idle = std::move(workspace);
    }
    // Failed jobs never return their workspace to the cache. The pipeline
    // drains outstanding CUDA work before stack unwinding reaches this lease.
};
} // namespace nvcomp_core
