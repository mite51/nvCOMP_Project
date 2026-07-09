// RAII wrappers for CUDA resources used by the compression pipelines.
// These guarantee cleanup when exceptions propagate out of a pipeline stage
// (the previous code leaked every device buffer on a thrown CUDA_CHECK).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace nvcomp_core {

// Device memory. Allocate-on-construct or empty + reserve() for reuse.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { reserve(bytes); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { release(); ptr_ = o.ptr_; size_ = o.size_; o.ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }

    // Grows only; keeps existing allocation when already large enough.
    void reserve(size_t bytes) {
        if (bytes <= size_) return;
        release();
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                      << " (cudaMalloc " << bytes << " bytes)" << std::endl;
            throw std::runtime_error("CUDA Error");
        }
        size_ = bytes;
    }
    void release() {
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; size_ = 0; }
    }
    template <typename T = void> T* get() const { return static_cast<T*>(ptr_); }
    uint8_t* bytes() const { return static_cast<uint8_t*>(ptr_); }
    size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// Page-locked host memory with pageable fallback (warns once per process).
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes) { reserve(bytes); }
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), pinned_(o.pinned_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_; size_ = o.size_; pinned_ = o.pinned_;
            o.ptr_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void reserve(size_t bytes) {
        if (bytes <= size_) return;
        release();
        if (cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault) == cudaSuccess) {
            pinned_ = true;
        } else {
            cudaGetLastError(); // clear sticky error
            static bool warned = false;
            if (!warned) {
                std::cerr << "Warning: pinned allocation failed ("
                          << bytes << " bytes); using pageable memory\n";
                warned = true;
            }
            ptr_ = std::malloc(bytes);
            if (!ptr_) throw std::bad_alloc();
            pinned_ = false;
        }
        size_ = bytes;
    }
    void release() {
        if (ptr_) {
            if (pinned_) cudaFreeHost(ptr_); else std::free(ptr_);
            ptr_ = nullptr; size_ = 0;
        }
    }
    uint8_t* bytes() const { return static_cast<uint8_t*>(ptr_); }
    size_t size() const { return size_; }
    bool pinned() const { return pinned_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    bool pinned_ = false;
};

// CUDA stream handle.
class CudaStream {
public:
    CudaStream() {
        if (cudaStreamCreate(&s_) != cudaSuccess)
            throw std::runtime_error("CUDA Error: cudaStreamCreate failed");
    }
    ~CudaStream() { if (s_) cudaStreamDestroy(s_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return s_; }
    operator cudaStream_t() const { return s_; }

private:
    cudaStream_t s_ = nullptr;
};

// CUDA event (blocking-sync so waiting threads sleep instead of spinning).
class CudaEvent {
public:
    explicit CudaEvent(unsigned flags = cudaEventBlockingSync | cudaEventDisableTiming) {
        if (cudaEventCreateWithFlags(&e_, flags) != cudaSuccess)
            throw std::runtime_error("CUDA Error: cudaEventCreate failed");
    }
    ~CudaEvent() { if (e_) cudaEventDestroy(e_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return e_; }
    operator cudaEvent_t() const { return e_; }

private:
    cudaEvent_t e_ = nullptr;
};

} // namespace nvcomp_core
