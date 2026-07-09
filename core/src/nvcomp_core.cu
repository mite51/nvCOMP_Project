#include "nvcomp_core.hpp"
#include "cuda_buffers.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <memory>

#ifdef __linux__
#include <sys/resource.h>
#endif

#include <cuda_runtime.h>

// Batched API headers (for cross-compatible algorithms)
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

// Manager API headers (for GPU-only algorithms)
#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

#define NVCOMP_CHECK(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvCOMP Error at line " << __LINE__ << std::endl; \
            throw std::runtime_error("nvCOMP Error"); \
        } \
    } while (0)

namespace nvcomp_core {

// ============================================================================
// Device helpers
// ============================================================================

// Compacts variable-size compressed chunks from their worst-case-strided
// output slots into one contiguous buffer. One block per chunk. Replaces the
// previous per-chunk cudaMemcpy D2H loop (measured 7.3x faster readback).
__global__ static void packChunksKernel(const void* const* srcPtrs,
                                        const size_t* srcSizes,
                                        const size_t* dstOffsets,
                                        uint8_t* dst,
                                        size_t numChunks) {
    size_t c = blockIdx.x;
    if (c >= numChunks) return;
    const uint8_t* src = static_cast<const uint8_t*>(srcPtrs[c]);
    uint8_t* out = dst + dstOffsets[c];
    size_t sz = srcSizes[c];
    if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        size_t nv = sz / 16;
        const uint4* src4 = reinterpret_cast<const uint4*>(src);
        for (size_t i = threadIdx.x; i < nv; i += blockDim.x) {
            uint4 v = src4[i];
            memcpy(out + i * 16, &v, 16); // dst offset may be unaligned
        }
        for (size_t i = nv * 16 + threadIdx.x; i < sz; i += blockDim.x) {
            out[i] = src[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < sz; i += blockDim.x) {
            out[i] = src[i];
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

static std::vector<std::vector<uint8_t>> splitIntoVolumes(
    const std::vector<uint8_t>& archiveData,
    uint64_t maxVolumeSize
) {
    std::vector<std::vector<uint8_t>> volumes;
    
    // If maxVolumeSize is 0 (disabled) or archive fits in single volume, return as-is
    if (maxVolumeSize == 0 || archiveData.size() <= maxVolumeSize) {
        volumes.push_back(archiveData);
        return volumes;
    }
    
    // Split into multiple volumes (mid-file splitting allowed)
    size_t remaining = archiveData.size();
    size_t offset = 0;
    size_t volumeIndex = 1;
    
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Splitting archive into volumes (max "
                  << (maxVolumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB each)...\n";
    }

    while (remaining > 0) {
        size_t volumeSize = std::min(static_cast<size_t>(maxVolumeSize), remaining);
        
        std::vector<uint8_t> volume(
            archiveData.begin() + offset,
            archiveData.begin() + offset + volumeSize
        );
        
        volumes.push_back(volume);

        if (verbose && (volumeIndex % 100 == 0 || remaining <= volumeSize)) {
            std::cout << "\r  Creating volumes... " << volumeIndex << " created" << std::flush;
        }
        
        offset += volumeSize;
        remaining -= volumeSize;
        volumeIndex++;
    }
    
    if (verbose) {
        std::cout << "\r  Created " << volumes.size() << " volume(s)" << std::string(20, ' ') << "\n";
    }
    
    return volumes;
}

// ============================================================================
// Algorithm Detection
// ============================================================================

AlgoType detectAlgorithmFromFile(const std::string& filename) {
    // Use filesystem::path to properly handle Unicode paths on Windows
    std::ifstream file(fs::path(filename), std::ios::binary);
    if (!file.is_open()) {
        return ALGO_UNKNOWN;
    }
    
    // Try to read BatchedHeader
    BatchedHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BatchedHeader));
    
    if (file.gcount() < sizeof(BatchedHeader)) {
        return ALGO_UNKNOWN;
    }
    
    if (header.magic != BATCHED_MAGIC) {
        return ALGO_UNKNOWN;
    }
    
    return static_cast<AlgoType>(header.algorithm);
}

// ============================================================================
// GPU Batched Compression
// ============================================================================

// Sub-batch sizing (chunks per GPU batch). POC sweep on RTX 5090: LZ4/Snappy
// peak at 2048 chunks (128 MB), Zstd at 1024 (64 MB); the old whole-volume
// batches (~40k chunks) ran 2.5x below peak. Override: NVCOMP_SUBBATCH_MB.
static size_t subBatchChunksFor(AlgoType algo) {
    size_t chunks = (algo == ALGO_ZSTD) ? 1024 : 2048;
    if (const char* env = std::getenv("NVCOMP_SUBBATCH_MB")) {
        long mb = std::atol(env);
        if (mb > 0) {
            chunks = std::max<size_t>(1, (static_cast<size_t>(mb) << 20) / CHUNK_SIZE);
        }
    }
    return chunks;
}

// Pipelined streaming compressor.
//
// Three concurrent actors connected by slot queues (PIPELINE_DEPTH pinned
// sub-batch slots, each with its own CUDA stream):
//   reader thread : walks `entries`, reads archive bytes from disk straight
//                   into a free slot's pinned input buffer
//   GPU (async)   : per slot: H2D -> nvcompBatched*CompressAsync -> chunk
//                   sizes D2H (event signals completion)
//   main thread   : submits filled slots, retires the oldest (pack kernel
//                   compacts chunks, one pinned D2H), appends to the output
//                   volume, fires progress callbacks
// Disk reads, compression kernels, and PCIe transfers of different sub-batches
// overlap; peak VRAM is ~depth x sub-batch working set instead of ~2.1x volume.
//
// Volume layout on disk is byte-identical to the previous implementation:
// sub-batch boundaries are CHUNK_SIZE-aligned within a volume and chunk sizes
// are appended in order, so the NVBC header + size table + chunk stream are
// unchanged. Volumes 2..N stream to disk through a placeholder size table that
// is patched via seekp once the volume completes; volume 1 (multi-volume) is
// buffered in RAM for the manifest prepend, exactly as before.
static void compressGPUBatchedStreaming(AlgoType algo,
                                        const std::vector<ArchiveEntry>& entries,
                                        const std::string& outputFile,
                                        uint64_t maxVolumeSize,
                                        ProgressCallback rawCallback,
                                        CompressionStats* stats) {
    using clock = std::chrono::steady_clock;
    auto callback = makeThrottledCallback(rawCallback);
    const bool verbose = isVerbose();
    auto opStart = clock::now();

    // Total uncompressed archive size (VolumeManifest field + stats input).
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
    }

    const bool multiVolume = maxVolumeSize > 0 && maxVolumeSize != UINT64_MAX
                             && totalArchiveSize > maxVolumeSize;
    auto volBytes = [&](size_t v) -> uint64_t {
        if (!multiVolume) return v == 0 ? totalArchiveSize : 0;
        uint64_t off = static_cast<uint64_t>(v) * maxVolumeSize;
        if (off >= totalArchiveSize) return 0;
        return std::min<uint64_t>(maxVolumeSize, totalArchiveSize - off);
    };
    const size_t volumeCount = multiVolume
        ? static_cast<size_t>((totalArchiveSize + maxVolumeSize - 1) / maxVolumeSize)
        : 1;

    if (verbose) {
        std::cout << "Using GPU batched compression (" << algoToString(algo)
                  << ") [pipelined, " << volumeCount << " volume(s)]...\n";
        std::cout << "Archive size: " << totalArchiveSize << " bytes\n";
    }
    if (stats) stats->inputBytes = totalArchiveSize;

    // ---- per-algo query + dispatch -----------------------------------------
    // Never allocate slots bigger than the whole job (small archives would
    // otherwise pay pinned-allocation cost for buffers they can't fill).
    size_t chunksPerSub = std::min<size_t>(
        subBatchChunksFor(algo),
        (totalArchiveSize + CHUNK_SIZE - 1) / CHUNK_SIZE);
    size_t max_out_bytes = 0, temp_bytes = 0;
    if (algo == ALGO_LZ4) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &max_out_bytes));
    } else if (algo == ALGO_SNAPPY) {
        NVCOMP_CHECK(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &max_out_bytes));
    } else {
        NVCOMP_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &max_out_bytes));
    }
    auto queryTemp = [&](size_t chunks) {
        size_t t = 0;
        size_t bytes = chunks * CHUNK_SIZE;
        if (algo == ALGO_LZ4) {
            NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync(
                chunks, CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &t, bytes));
        } else if (algo == ALGO_SNAPPY) {
            NVCOMP_CHECK(nvcompBatchedSnappyCompressGetTempSizeAsync(
                chunks, CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &t, bytes));
        } else {
            NVCOMP_CHECK(nvcompBatchedZstdCompressGetTempSizeAsync(
                chunks, CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &t, bytes));
        }
        return t;
    };
    temp_bytes = queryTemp(chunksPerSub);

    size_t subBytes = chunksPerSub * CHUNK_SIZE;
    size_t depth = std::max<size_t>(1, std::min<uint64_t>(
        PIPELINE_DEPTH, (totalArchiveSize + subBytes - 1) / subBytes));

    // Fit the pipeline into free VRAM: shrink depth first, then sub-batch
    // size (temp requirements scale with chunk count). Actual footprint per
    // slot: input (sized for worst-case packed reuse) + strided output + temp.
    {
        size_t freeMem = 0, totalMem = 0;
        if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
            const uint64_t slack = 256ull << 20;
            auto pipelineNeed = [&]() -> uint64_t {
                uint64_t perSlot = 2ull * max_out_bytes * chunksPerSub  // d_in + d_out
                    + temp_bytes
                    + 5ull * chunksPerSub * sizeof(size_t);
                return depth * perSlot + slack;
            };
            while (pipelineNeed() > freeMem &&
                   (depth > 1 || chunksPerSub > 256)) {
                if (depth > 1) {
                    depth--;
                } else {
                    chunksPerSub /= 2;
                    temp_bytes = queryTemp(chunksPerSub);
                }
            }
            if (verbose && (depth < PIPELINE_DEPTH || chunksPerSub < subBatchChunksFor(algo))) {
                std::cout << "  VRAM fit: depth=" << depth
                          << ", sub-batch=" << (chunksPerSub * CHUNK_SIZE >> 20)
                          << " MB (" << (freeMem >> 20) << " MB free)\n";
            }
            subBytes = chunksPerSub * CHUNK_SIZE;
        } else {
            cudaGetLastError();
        }
    }
    const size_t maxPackedBytes = max_out_bytes * chunksPerSub;

    // ---- pipeline slots -----------------------------------------------------
    struct Slot {
        PinnedBuffer h_in;         // sub-batch input, filled by the reader
        PinnedBuffer h_out;        // packed compressed output
        PinnedBuffer h_sizes;      // per-chunk compressed sizes (D2H)
        PinnedBuffer h_offsets;    // per-chunk packed offsets (H2D)
        DeviceBuffer d_in;         // input; reused as pack destination
        DeviceBuffer d_out;        // worst-case strided compressed output
        DeviceBuffer d_temp;
        DeviceBuffer d_in_ptrs, d_in_sizes, d_out_ptrs, d_out_sizes, d_offsets;
        std::unique_ptr<CudaStream> stream;
        std::unique_ptr<CudaEvent> computeDone;
        bool defaultSizes = true;  // d_in_sizes holds all-CHUNK_SIZE values
        // descriptor of the filled sub-batch
        size_t bytes = 0;
        size_t volumeIdx = 0;
        bool lastInVolume = false;
    };
    std::vector<Slot> slots(depth);
    for (auto& s : slots) {
        // d_in doubles as the pack destination, so size it for the worst-case
        // packed output (slightly larger than the input for incompressible data).
        s.h_in.reserve(subBytes);
        s.h_out.reserve(maxPackedBytes);
        s.h_sizes.reserve(sizeof(size_t) * chunksPerSub);
        s.h_offsets.reserve(sizeof(size_t) * chunksPerSub);
        s.d_in.reserve(std::max(subBytes, maxPackedBytes));
        s.d_out.reserve(maxPackedBytes);
        s.d_temp.reserve(temp_bytes);
        s.d_in_ptrs.reserve(sizeof(void*) * chunksPerSub);
        s.d_in_sizes.reserve(sizeof(size_t) * chunksPerSub);
        s.d_out_ptrs.reserve(sizeof(void*) * chunksPerSub);
        s.d_out_sizes.reserve(sizeof(size_t) * chunksPerSub);
        s.d_offsets.reserve(sizeof(size_t) * chunksPerSub);
        s.stream = std::make_unique<CudaStream>();
        s.computeDone = std::make_unique<CudaEvent>();

        // Input/output pointer tables are position-invariant per slot: set once.
        std::vector<void*> in_ptrs(chunksPerSub), out_ptrs(chunksPerSub);
        std::vector<size_t> in_sizes(chunksPerSub, CHUNK_SIZE);
        for (size_t i = 0; i < chunksPerSub; i++) {
            in_ptrs[i] = s.d_in.bytes() + i * CHUNK_SIZE;
            out_ptrs[i] = s.d_out.bytes() + i * max_out_bytes;
        }
        CUDA_CHECK(cudaMemcpy(s.d_in_ptrs.get(), in_ptrs.data(),
                              sizeof(void*) * chunksPerSub, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_out_ptrs.get(), out_ptrs.data(),
                              sizeof(void*) * chunksPerSub, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_in_sizes.get(), in_sizes.data(),
                              sizeof(size_t) * chunksPerSub, cudaMemcpyHostToDevice));
    }

    // ---- reader thread <-> main thread queues -------------------------------
    std::mutex mtx;
    std::condition_variable cvFree, cvFilled;
    std::deque<size_t> freeSlots, filledSlots;
    for (size_t i = 0; i < depth; i++) freeSlots.push_back(i);
    bool readerDone = false;
    std::atomic<bool> abortFlag{false};
    std::exception_ptr readerError;

    auto readerFn = [&]() {
        try {
            size_t vol = 0;
            uint64_t volRemaining = volBytes(0);
            ptrdiff_t cur = -1;
            size_t curFill = 0;

            auto acquireSlot = [&]() {
                std::unique_lock<std::mutex> lk(mtx);
                cvFree.wait(lk, [&] { return !freeSlots.empty() || abortFlag.load(); });
                if (abortFlag.load()) throw std::runtime_error("compression aborted");
                cur = static_cast<ptrdiff_t>(freeSlots.front());
                freeSlots.pop_front();
                curFill = 0;
            };
            auto emitSlot = [&](bool volEnd) {
                Slot& s = slots[cur];
                s.bytes = curFill;
                s.volumeIdx = vol;
                s.lastInVolume = volEnd;
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    filledSlots.push_back(static_cast<size_t>(cur));
                }
                cvFilled.notify_one();
                cur = -1;
                curFill = 0;
            };
            // Reserve up to `want` bytes of contiguous slot space (bounded by
            // slot capacity and the current volume); returns the span.
            auto reserveSpan = [&](uint64_t want, uint8_t** dst) -> uint64_t {
                if (cur < 0) acquireSlot();
                uint64_t take = std::min<uint64_t>(
                    {want, subBytes - curFill, volRemaining});
                *dst = slots[cur].h_in.bytes() + curFill;
                return take;
            };
            auto commitSpan = [&](uint64_t took) {
                curFill += static_cast<size_t>(took);
                volRemaining -= took;
                bool volEnd = volRemaining == 0;
                if (volEnd || curFill == subBytes) {
                    emitSlot(volEnd);
                    if (volEnd) {
                        vol++;
                        volRemaining = volBytes(vol);
                    }
                }
            };
            auto putBytes = [&](const uint8_t* src, uint64_t n) {
                while (n > 0) {
                    uint8_t* dst = nullptr;
                    uint64_t take = reserveSpan(n, &dst);
                    std::memcpy(dst, src, static_cast<size_t>(take));
                    src += take;
                    n -= take;
                    commitSpan(take);
                }
            };
            auto putFile = [&](const ArchiveEntry& e) {
                if (e.fileSize == 0) return;
                uint8_t* dst = nullptr;
                uint64_t take = reserveSpan(e.fileSize, &dst);
                if (take == e.fileSize) {
                    // Whole file fits in the current span: one mmap'd read.
                    readFileInto(e.filePath, dst, e.fileSize);
                    commitSpan(take);
                    return;
                }
                // File spans slots/volumes: stream it.
                std::ifstream f(e.filePath, std::ios::binary);
                if (!f.is_open()) {
                    throw std::runtime_error("Failed to open input file: " + e.filePath.string());
                }
                uint64_t remaining = e.fileSize;
                while (remaining > 0) {
                    take = reserveSpan(remaining, &dst);
                    if (!f.read(reinterpret_cast<char*>(dst),
                                static_cast<std::streamsize>(take))) {
                        throw std::runtime_error("Failed to read file: " + e.filePath.string());
                    }
                    remaining -= take;
                    commitSpan(take);
                }
            };

            ArchiveHeader hdr;
            hdr.magic = ARCHIVE_MAGIC;
            hdr.version = ARCHIVE_VERSION;
            hdr.fileCount = static_cast<uint32_t>(entries.size());
            hdr.reserved = 0;
            putBytes(reinterpret_cast<const uint8_t*>(&hdr), sizeof(ArchiveHeader));

            for (const auto& e : entries) {
                if (abortFlag.load()) throw std::runtime_error("compression aborted");
                FileEntry fe;
                fe.pathLength = static_cast<uint32_t>(e.relativePath.size());
                fe.mode = e.mode;
                fe.fileSize = e.fileSize;
                fe.mtimeNs = e.mtimeNs;
                putBytes(reinterpret_cast<const uint8_t*>(&fe), sizeof(FileEntry));
                putBytes(reinterpret_cast<const uint8_t*>(e.relativePath.data()),
                         e.relativePath.size());
                putFile(e);
                if (verbose) {
                    std::cout << "  Adding: " << e.relativePath
                              << " (" << e.fileSize << " bytes)\n";
                }
            }
            // The archive stream ends exactly at the last volume boundary, so
            // the final emitSlot(volEnd=true) already fired inside commitSpan.
        } catch (...) {
            readerError = std::current_exception();
            abortFlag.store(true);
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            readerDone = true;
        }
        cvFilled.notify_all();
    };

    // ---- GPU submit / retire -------------------------------------------------
    auto submitSlot = [&](Slot& s) {
        size_t chunk_count = (s.bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
        cudaStream_t st = *s.stream;
        CUDA_CHECK(cudaMemcpyAsync(s.d_in.get(), s.h_in.bytes(), s.bytes,
                                   cudaMemcpyHostToDevice, st));
        if (s.bytes < subBytes) {
            // Partial sub-batch (volume tail): last chunk may be short.
            std::vector<size_t> sizes(chunk_count, CHUNK_SIZE);
            sizes[chunk_count - 1] = s.bytes - (chunk_count - 1) * CHUNK_SIZE;
            CUDA_CHECK(cudaMemcpyAsync(s.d_in_sizes.get(), sizes.data(),
                                       sizeof(size_t) * chunk_count,
                                       cudaMemcpyHostToDevice, st));
            s.defaultSizes = false;
        } else if (!s.defaultSizes) {
            std::vector<size_t> sizes(chunksPerSub, CHUNK_SIZE);
            CUDA_CHECK(cudaMemcpyAsync(s.d_in_sizes.get(), sizes.data(),
                                       sizeof(size_t) * chunksPerSub,
                                       cudaMemcpyHostToDevice, st));
            s.defaultSizes = true;
        }
        if (algo == ALGO_LZ4) {
            NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
                s.d_in_ptrs.get<void*>(), s.d_in_sizes.get<size_t>(), CHUNK_SIZE, chunk_count,
                s.d_temp.get(), temp_bytes, s.d_out_ptrs.get<void*>(), s.d_out_sizes.get<size_t>(),
                nvcompBatchedLZ4CompressDefaultOpts, nullptr, st));
        } else if (algo == ALGO_SNAPPY) {
            NVCOMP_CHECK(nvcompBatchedSnappyCompressAsync(
                s.d_in_ptrs.get<void*>(), s.d_in_sizes.get<size_t>(), CHUNK_SIZE, chunk_count,
                s.d_temp.get(), temp_bytes, s.d_out_ptrs.get<void*>(), s.d_out_sizes.get<size_t>(),
                nvcompBatchedSnappyCompressDefaultOpts, nullptr, st));
        } else {
            NVCOMP_CHECK(nvcompBatchedZstdCompressAsync(
                s.d_in_ptrs.get<void*>(), s.d_in_sizes.get<size_t>(), CHUNK_SIZE, chunk_count,
                s.d_temp.get(), temp_bytes, s.d_out_ptrs.get<void*>(), s.d_out_sizes.get<size_t>(),
                nvcompBatchedZstdCompressDefaultOpts, nullptr, st));
        }
        CUDA_CHECK(cudaMemcpyAsync(s.h_sizes.bytes(), s.d_out_sizes.get(),
                                   sizeof(size_t) * chunk_count,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(*s.computeDone, st));
    };

    // ---- writer state (main thread) -----------------------------------------
    std::vector<VolumeMetadata> volumeMetadata;
    std::vector<uint64_t> volume1Sizes;   // multi-volume: volume 1 buffered in RAM
    std::vector<uint8_t> volume1Data;
    std::ofstream volFile;
    std::vector<uint64_t> volSizeTable;
    size_t volExpectedChunks = 0;
    size_t curWriteVol = SIZE_MAX;
    uint64_t uncompressedOffset = 0;
    uint64_t totalCompressedBytes = 0;
    uint64_t bytesRetired = 0;
    double writeSec = 0.0;
    std::vector<std::string> createdFiles;

    auto makeHeader = [&](size_t vol) {
        BatchedHeader h;
        h.magic = BATCHED_MAGIC;
        h.version = BATCHED_VERSION;
        h.uncompressedSize = volBytes(vol);
        h.chunkCount = static_cast<uint32_t>((volBytes(vol) + CHUNK_SIZE - 1) / CHUNK_SIZE);
        h.chunkSize = CHUNK_SIZE;
        h.algorithm = static_cast<uint32_t>(algo);
        h.reserved = 0;
        return h;
    };

    auto beginVolume = [&](size_t vol) {
        curWriteVol = vol;
        volExpectedChunks = (volBytes(vol) + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (multiVolume && vol == 0) {
            volume1Sizes.clear();
            volume1Data.clear();
            return;
        }
        std::string name = multiVolume ? generateVolumeFilename(outputFile, vol + 1)
                                       : outputFile;
        volFile.open(fs::path(name), std::ios::binary | std::ios::trunc);
        if (!volFile.is_open()) {
            throw std::runtime_error("Failed to create output file: " + name);
        }
        createdFiles.push_back(name);
        BatchedHeader h = makeHeader(vol);
        volFile.write(reinterpret_cast<const char*>(&h), sizeof(h));
        // Placeholder size table, patched when the volume completes.
        std::vector<uint64_t> zeros(volExpectedChunks, 0);
        volFile.write(reinterpret_cast<const char*>(zeros.data()),
                      sizeof(uint64_t) * volExpectedChunks);
        volSizeTable.clear();
        volSizeTable.reserve(volExpectedChunks);
    };

    auto finishVolume = [&](size_t vol) {
        VolumeMetadata meta;
        meta.volumeIndex = vol + 1;
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = volBytes(vol);
        if (multiVolume && vol == 0) {
            meta.compressedSize = sizeof(BatchedHeader)
                + sizeof(uint64_t) * volume1Sizes.size() + volume1Data.size();
        } else {
            if (volSizeTable.size() != volExpectedChunks) {
                throw std::runtime_error("Internal error: volume chunk count mismatch");
            }
            volFile.seekp(sizeof(BatchedHeader), std::ios::beg);
            volFile.write(reinterpret_cast<const char*>(volSizeTable.data()),
                          sizeof(uint64_t) * volSizeTable.size());
            volFile.close();
            uint64_t dataBytes = 0;
            for (uint64_t sz : volSizeTable) dataBytes += sz;
            meta.compressedSize = sizeof(BatchedHeader)
                + sizeof(uint64_t) * volSizeTable.size() + dataBytes;
        }
        volumeMetadata.push_back(meta);
        uncompressedOffset += meta.uncompressedSize;
        totalCompressedBytes += meta.compressedSize;
        if (verbose) {
            std::cout << "  Volume " << (vol + 1) << " complete ("
                      << meta.uncompressedSize << " B -> "
                      << meta.compressedSize << " B)\n";
        }
        curWriteVol = SIZE_MAX;
    };

    auto retireSlot = [&](size_t idx) {
        Slot& s = slots[idx];
        CUDA_CHECK(cudaEventSynchronize(*s.computeDone));
        size_t chunk_count = (s.bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
        cudaStream_t st = *s.stream;

        // Host exclusive-scan of compressed sizes -> packed offsets.
        const size_t* csizes = reinterpret_cast<const size_t*>(s.h_sizes.bytes());
        size_t* offs = reinterpret_cast<size_t*>(s.h_offsets.bytes());
        size_t packed = 0;
        for (size_t i = 0; i < chunk_count; i++) {
            offs[i] = packed;
            packed += csizes[i];
        }
        CUDA_CHECK(cudaMemcpyAsync(s.d_offsets.get(), offs,
                                   sizeof(size_t) * chunk_count,
                                   cudaMemcpyHostToDevice, st));
        packChunksKernel<<<static_cast<unsigned>(chunk_count), 256, 0, st>>>(
            s.d_out_ptrs.get<const void* const>(), s.d_out_sizes.get<size_t>(),
            s.d_offsets.get<size_t>(), s.d_in.bytes(), chunk_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(s.h_out.bytes(), s.d_in.bytes(), packed,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        // Append to the current volume (writer role).
        if (s.volumeIdx != curWriteVol) beginVolume(s.volumeIdx);
        auto writeStart = clock::now();
        if (multiVolume && s.volumeIdx == 0) {
            volume1Sizes.insert(volume1Sizes.end(), csizes, csizes + chunk_count);
            volume1Data.insert(volume1Data.end(), s.h_out.bytes(), s.h_out.bytes() + packed);
        } else {
            volSizeTable.insert(volSizeTable.end(), csizes, csizes + chunk_count);
            volFile.write(reinterpret_cast<const char*>(s.h_out.bytes()),
                          static_cast<std::streamsize>(packed));
            if (!volFile) {
                throw std::runtime_error("Failed to write output volume");
            }
        }
        bytesRetired += s.bytes;
        bool volEnd = s.lastInVolume;
        size_t volIdx = s.volumeIdx;
        writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();

        // Slot is free again.
        {
            std::lock_guard<std::mutex> lk(mtx);
            freeSlots.push_back(idx);
        }
        cvFree.notify_one();

        if (volEnd) {
            auto ws = clock::now();
            finishVolume(volIdx);
            writeSec += std::chrono::duration<double>(clock::now() - ws).count();
        }

        if (callback && totalArchiveSize > 0) {
            float p = static_cast<float>(bytesRetired) / totalArchiveSize;
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(
                (totalArchiveSize + subBytes - 1) / subBytes);
            info.completedBlocks = static_cast<int>(
                (bytesRetired + subBytes - 1) / subBytes);
            info.currentBlock = info.completedBlocks > 0 ? info.completedBlocks - 1 : 0;
            info.currentBlockSize = s.bytes;
            info.overallProgress = p * 0.75f;
            info.currentBlockProgress = 1.0f;
            double elapsed = std::chrono::duration<double>(clock::now() - opStart).count();
            info.throughputMBps = elapsed > 0
                ? (bytesRetired / (1024.0 * 1024.0)) / elapsed : 0.0;
            info.stage = "compressing";
            callback(info);
        }
    };

    // ---- pump ----------------------------------------------------------------
    std::thread reader(readerFn);
    std::deque<size_t> inFlight;
    auto pumpStart = clock::now();
    try {
        while (true) {
            size_t idx = SIZE_MAX;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cvFilled.wait(lk, [&] {
                    return !filledSlots.empty() || readerDone || abortFlag.load();
                });
                if (abortFlag.load() && filledSlots.empty()) break;
                if (!filledSlots.empty()) {
                    idx = filledSlots.front();
                    filledSlots.pop_front();
                } else if (readerDone) {
                    break;
                }
            }
            if (idx != SIZE_MAX) {
                submitSlot(slots[idx]);
                inFlight.push_back(idx);
                if (inFlight.size() >= depth) {
                    retireSlot(inFlight.front());
                    inFlight.pop_front();
                }
            }
        }
        while (!inFlight.empty()) {
            retireSlot(inFlight.front());
            inFlight.pop_front();
        }
    } catch (...) {
        abortFlag.store(true);
        cvFree.notify_all();
        cvFilled.notify_all();
        reader.join();
        if (volFile.is_open()) volFile.close();
        for (const auto& f : createdFiles) {
            std::error_code ec;
            fs::remove(fs::path(f), ec);
        }
        throw;
    }
    reader.join();
    if (readerError) {
        if (volFile.is_open()) volFile.close();
        for (const auto& f : createdFiles) {
            std::error_code ec;
            fs::remove(fs::path(f), ec);
        }
        std::rethrow_exception(readerError);
    }
    if (stats) {
        double pumpSec = std::chrono::duration<double>(clock::now() - pumpStart).count();
        // Disk reads overlap compression in this pipeline, so "compute" covers
        // the whole overlapped fill+compress region; write time is separate.
        stats->computeSec += std::max(0.0, pumpSec - writeSec);
        stats->writeSec += writeSec;
    }

    // ---- multi-volume: write volume 1 (manifest + metadata + NVBC) last ------
    auto writeStart = clock::now();
    if (multiVolume) {
        VolumeManifest manifest;
        manifest.magic = VOLUME_MAGIC;
        manifest.version = VOLUME_VERSION;
        manifest.volumeCount = static_cast<uint32_t>(volumeMetadata.size());
        manifest.algorithm = static_cast<uint32_t>(algo);
        manifest.volumeSize = maxVolumeSize;
        manifest.totalUncompressedSize = totalArchiveSize;
        manifest.reserved = 0;

        std::vector<uint8_t> volume1OnDisk;
        BatchedHeader h1 = makeHeader(0);
        volume1OnDisk.reserve(sizeof(VolumeManifest)
                              + sizeof(VolumeMetadata) * volumeMetadata.size()
                              + sizeof(BatchedHeader)
                              + sizeof(uint64_t) * volume1Sizes.size()
                              + volume1Data.size());
        const uint8_t* mb = reinterpret_cast<const uint8_t*>(&manifest);
        volume1OnDisk.insert(volume1OnDisk.end(), mb, mb + sizeof(VolumeManifest));
        const uint8_t* vmb = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
        volume1OnDisk.insert(volume1OnDisk.end(), vmb,
                             vmb + sizeof(VolumeMetadata) * volumeMetadata.size());
        const uint8_t* hb = reinterpret_cast<const uint8_t*>(&h1);
        volume1OnDisk.insert(volume1OnDisk.end(), hb, hb + sizeof(BatchedHeader));
        const uint8_t* sb = reinterpret_cast<const uint8_t*>(volume1Sizes.data());
        volume1OnDisk.insert(volume1OnDisk.end(), sb,
                             sb + sizeof(uint64_t) * volume1Sizes.size());
        volume1OnDisk.insert(volume1OnDisk.end(),
                             volume1Data.begin(), volume1Data.end());

        // Volume 1's on-disk size includes the manifest + metadata prepend.
        totalCompressedBytes = totalCompressedBytes
            - volumeMetadata[0].compressedSize + volume1OnDisk.size();
        volumeMetadata[0].compressedSize = volume1OnDisk.size();

        std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
        writeFile(firstVolumeFile, volume1OnDisk.data(), volume1OnDisk.size());
    }
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        stats->outputBytes = totalCompressedBytes;
    }

    // Final 100% progress notification.
    if (callback) {
        BlockProgressInfo info;
        info.totalBlocks = static_cast<int>(volumeMetadata.size());
        info.completedBlocks = static_cast<int>(volumeMetadata.size());
        info.currentBlock = static_cast<int>(volumeMetadata.size() > 0 ? volumeMetadata.size() - 1 : 0);
        info.currentBlockSize = 0;
        info.overallProgress = 1.0f;
        info.currentBlockProgress = 1.0f;
        info.throughputMBps = 0.0;
        info.stage = "complete";
        callback(info);
    }

    if (multiVolume) {
        // Always-on result summary (matches the previous pipeline's output).
        double totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
        std::cout << "Volumes created: " << volumeMetadata.size() << std::endl;
        std::cout << "Total uncompressed: " << (totalArchiveSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Total compressed: " << (totalCompressedBytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        if (totalCompressedBytes > 0) {
            std::cout << "Overall ratio: " << std::fixed << std::setprecision(2)
                      << (double)totalArchiveSize / totalCompressedBytes << "x" << std::endl;
        }
        std::cout << "Total time: " << totalSec << "s ("
                  << (totalArchiveSize / (1024.0 * 1024.0 * 1024.0)) / std::max(totalSec, 1e-9) << " GB/s)" << std::endl;
    }
}

// Public wrapper for single file/folder compression. All GPU-batched
// compression (single-volume, --no-volumes, and multi-volume) routes through
// the pipelined streaming compressor: a single volume is simply one volume.
void compressGPUBatched(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    auto entries = collectArchiveEntries(inputPath);
    compressGPUBatchedStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void compressGPUBatchedFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    if (isVerbose()) {
        std::cout << "Compressing file list (" << filePaths.size() << " files)...\n";
    }

    auto entries = collectArchiveEntriesFromList(filePaths);
    compressGPUBatchedStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

// ============================================================================
// GPU Batched Decompression
// ============================================================================

// Pipelined GPU decompressor for NVBC files. Constructed once per operation
// and reused across volumes (buffers, streams, and temp space allocated once).
//
// Per volume, the chunk-size table gives the exact compressed byte range and
// the exact uncompressed size of every sub-batch up front, so each slot's
// whole chain (H2D -> nvcompBatched*DecompressAsync -> statuses/actual D2H ->
// output D2H -> event) is enqueued in one go. The main thread reads the next
// sub-batch from disk while previous slots decompress on their own streams --
// disk, PCIe, and decompression overlap with no extra threads.
//
// Any ineligibility (CPU-compressed single-chunk files, unexpected chunk size,
// oversized chunks) or GPU failure returns false; the caller falls back to
// decompressBatchedFormatCPU exactly as before, so old archives cannot break.
// nvCOMP 5.1 decompress alignment requirements are 1 byte for LZ4/Snappy/Zstd
// inputs/outputs (verified via GetRequiredAlignments), so compressed chunk
// pointers can point directly at NVBC's back-to-back chunk layout.
// Experiment instrumentation (NVCOMP_PHASE_DEBUG=1): fine-grained timing of
// the decompress pump to attribute wall time between GPU waits, disk reads,
// and host memcpy/page-fault costs.
static bool phaseDebug() {
    static const bool on = [] {
        const char* e = std::getenv("NVCOMP_PHASE_DEBUG");
        return e && *e && *e != '0';
    }();
    return on;
}

static long minorFaults() {
#ifdef __linux__
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_minflt;
#else
    return 0;
#endif
}

// Thrown when the downstream consumer (extraction) fails. Distinct from GPU
// errors on purpose: it must propagate out of the operation instead of
// triggering the CPU fallback (which would re-decompress and re-extract).
struct ExtractionAbort : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Consumer of decompressed bytes, called on the pipeline thread in stream
// order. The consumer must invoke `release` exactly once when it is done with
// [data, data+n) -- the memory is a reusable pinned download buffer.
struct DecompressSink {
    virtual ~DecompressSink() = default;
    virtual void consume(const uint8_t* data, size_t n,
                         std::function<void()> release) = 0;
    virtual uint64_t consumed() const = 0;   // bytes accepted so far
};

// Appends into a std::vector (the pre-streaming behavior).
struct VectorSink : DecompressSink {
    explicit VectorSink(std::vector<uint8_t>& out) : out_(out) {}
    void consume(const uint8_t* data, size_t n, std::function<void()> release) override {
        out_.insert(out_.end(), data, data + n);
        if (release) release();
    }
    uint64_t consumed() const override { return out_.size(); }
    std::vector<uint8_t>& out_;
};

// Feeds decompressed bytes to an ArchiveExtractor on a dedicated thread, so
// file writes overlap GPU decompression. Buffers are released back to the
// pinned pool only after every write referencing them completes (the
// extractor's per-feed guards handle that).
class StreamingExtractSink : public DecompressSink {
public:
    StreamingExtractSink(const std::string& outputPath, size_t writerThreads)
        : extractor_(outputPath, writerThreads) {
        thread_ = std::thread([this] { run(); });
    }
    ~StreamingExtractSink() override {
        {
            std::lock_guard<std::mutex> lk(m_);
            done_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void consume(const uint8_t* data, size_t n, std::function<void()> release) override {
        {
            std::unique_lock<std::mutex> lk(m_);
            if (error_) {
                lk.unlock();
                if (release) release();
                std::rethrow_exception(makeAbort());
            }
            queue_.push_back(Item{data, n, std::move(release)});
        }
        consumed_ += n;
        cv_.notify_one();
    }
    uint64_t consumed() const override { return consumed_; }

    // Drain the queue, join the extractor thread, validate the archive ended
    // cleanly. Throws ExtractionAbort on any extraction/write failure.
    void finish() {
        {
            std::lock_guard<std::mutex> lk(m_);
            done_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        if (error_) std::rethrow_exception(makeAbort());
        try {
            extractor_.finish();
        } catch (const std::exception& e) {
            throw ExtractionAbort(e.what());
        }
    }

    // Stop without validation (caller falls back to a from-scratch extract).
    void abandon() {
        {
            std::lock_guard<std::mutex> lk(m_);
            done_ = true;
            if (!error_) {
                error_ = std::make_exception_ptr(std::runtime_error("abandoned"));
            }
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        extractor_.abandon();
    }

    uint64_t bytesWritten() const { return extractor_.bytesWritten(); }

private:
    struct Item {
        const uint8_t* data;
        size_t n;
        std::function<void()> release;
    };

    std::exception_ptr makeAbort() {
        try {
            std::rethrow_exception(error_);
        } catch (const std::exception& e) {
            return std::make_exception_ptr(ExtractionAbort(e.what()));
        }
    }

    void run() {
        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !queue_.empty() || done_; });
                if (queue_.empty()) return;
                it = std::move(queue_.front());
                queue_.pop_front();
                if (error_) {
                    lk.unlock();
                    if (it.release) it.release();  // drain: free buffer, skip work
                    continue;
                }
            }
            try {
                extractor_.feed(it.data, it.n, std::move(it.release));
                // feed() wraps the release in a guard that fires even when
                // feed throws, so buffers are never leaked past this point.
            } catch (...) {
                std::lock_guard<std::mutex> lk(m_);
                if (!error_) error_ = std::current_exception();
            }
        }
    }

    ArchiveExtractor extractor_;
    std::thread thread_;
    std::deque<Item> queue_;
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
    std::exception_ptr error_;
    std::atomic<uint64_t> consumed_{0};
};

// Free-list pool of pinned download buffers shared by the pipeline slots and
// the streaming consumer. Sized depth+spare so the extractor can trail the
// GPU by a few sub-batches before backpressure (acquire blocks) kicks in.
class PinnedOutPool {
public:
    void init(size_t count, size_t bytes) {
        std::unique_lock<std::mutex> lk(m_);
        // Never realloc while a consumer still holds a buffer.
        cv_.wait(lk, [&] { return free_.size() == bufs_.size(); });
        if (bufs_.size() >= count && !bufs_.empty() && bufs_[0].size() >= bytes) return;
        bufs_.clear();
        free_.clear();
        bufs_.resize(count);
        for (size_t i = 0; i < count; i++) {
            bufs_[i].reserve(bytes);
            free_.push_back(i);
        }
    }
    size_t acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        size_t i = free_.front();
        free_.pop_front();
        return i;
    }
    uint8_t* ptr(size_t i) { return bufs_[i].bytes(); }
    void release(size_t i) {
        {
            std::lock_guard<std::mutex> lk(m_);
            free_.push_back(i);
        }
        cv_.notify_all();
    }

private:
    std::vector<PinnedBuffer> bufs_;
    std::deque<size_t> free_;
    std::mutex m_;
    std::condition_variable cv_;
};

class BatchedDecompressPipeline {
public:
    // Convenience wrapper: append the decompressed payload onto a vector.
    bool decompressFile(const std::string& path, uint64_t payloadOffset,
                        std::vector<uint8_t>& out) {
        VectorSink sink(out);
        return decompressFile(path, payloadOffset, sink);
    }

    // Streams the decompressed payload of the NVBC stream at `payloadOffset`
    // in `path` into `sink` in order. Returns false -> caller uses the CPU
    // path (resuming after sink.consumed() bytes when a mid-volume failure
    // left a partial stream). Consumer failures throw ExtractionAbort.
    bool decompressFile(const std::string& path, uint64_t payloadOffset,
                        DecompressSink& sink) {
        std::ifstream f(fs::path(path), std::ios::binary);
        if (!f.is_open()) return false;
        f.seekg(static_cast<std::streamoff>(payloadOffset));

        BatchedHeader header;
        if (!f.read(reinterpret_cast<char*>(&header), sizeof(header))) return false;
        if (header.magic != BATCHED_MAGIC) return false;
        AlgoType algo = static_cast<AlgoType>(header.algorithm);
        if (algo != ALGO_LZ4 && algo != ALGO_SNAPPY && algo != ALGO_ZSTD) return false;
        const size_t n = header.chunkCount;
        if (n <= 1 || header.chunkSize != CHUNK_SIZE) return false;
        if (header.uncompressedSize == 0 ||
            header.uncompressedSize > (uint64_t)n * CHUNK_SIZE) return false;
        // Small payloads decode faster on the CPU than the one-time CUDA
        // context/pipeline setup costs (~150 ms) -- unless this pipeline is
        // already warm from a previous volume of the same operation.
        if (!ready_ && header.uncompressedSize < (64ull << 20)) return false;

        std::vector<uint64_t> sizes(n);
        if (!f.read(reinterpret_cast<char*>(sizes.data()), sizeof(uint64_t) * n)) {
            return false;
        }

        try {
            if (!ensureResources(algo, n)) return false;
            for (uint64_t sz : sizes) {
                if (sz == 0 || sz > maxCompChunk_) return false; // corrupt/foreign
            }

            const bool dbg = phaseDebug();
            using tclk = std::chrono::steady_clock;
            dbgWait_ = dbgInsert_ = dbgRead_ = 0.0;
            long faults0 = dbg ? minorFaults() : 0;
            auto volStart = tclk::now();

            const size_t numSub = (n + chunksPerSub_ - 1) / chunksPerSub_;
            size_t retired = 0;   // count of retired sub-batches
            size_t submitted = 0;
            for (size_t sb = 0; sb < numSub; sb++) {
                Slot& s = slots_[sb % depth_];
                if (s.inFlight) {
                    retireSlot(s, sink);
                    retired++;
                }
                const size_t c0 = sb * chunksPerSub_;
                const size_t cn = std::min(chunksPerSub_, n - c0);
                uint64_t compBytes = 0;
                for (size_t i = 0; i < cn; i++) compBytes += sizes[c0 + i];
                // Disk read overlaps the other slots' in-flight GPU work.
                auto r0 = tclk::now();
                if (!f.read(reinterpret_cast<char*>(s.h_in.bytes()),
                            static_cast<std::streamsize>(compBytes))) {
                    throw std::runtime_error("short read in " + path);
                }
                dbgRead_ += std::chrono::duration<double>(tclk::now() - r0).count();
                submitSlot(s, algo, &sizes[c0], cn,
                           subUncompBytes(header, c0, cn), compBytes);
                submitted++;
            }
            for (size_t sb = retired; sb < submitted; sb++) {
                Slot& s = slots_[sb % depth_];
                if (s.inFlight) retireSlot(s, sink);
            }
            if (dbg) {
                static tclk::time_point epoch = volStart;
                double t0 = std::chrono::duration<double>(volStart - epoch).count();
                double t1 = std::chrono::duration<double>(tclk::now() - epoch).count();
                std::cout << "  [phase-debug] vol " << path.substr(path.size() > 24 ? path.size() - 24 : 0)
                          << " span=" << t0 << ".." << t1
                          << " wall=" << (t1 - t0)
                          << " read=" << dbgRead_
                          << " wait=" << dbgWait_
                          << " insert=" << dbgInsert_
                          << " minor-faults=" << (minorFaults() - faults0)
                          << "\n";
            }
            return true;
        } catch (const ExtractionAbort&) {
            // Consumer-side failure: not a GPU problem, so no CPU fallback.
            quiesce();
            throw;
        } catch (const std::exception& e) {
            std::cerr << "GPU decompression failed (" << e.what()
                      << "); falling back to CPU\n";
            quiesce();
            return false;
        }
    }

private:
    struct Slot {
        PinnedBuffer h_in;
        PinnedBuffer h_in_ptrs, h_in_sizes, h_out_sizes; // staging for tables
        PinnedBuffer h_statuses, h_actual;
        DeviceBuffer d_in, d_out, d_temp;
        DeviceBuffer d_in_ptrs, d_in_sizes, d_out_ptrs, d_out_sizes;
        DeviceBuffer d_statuses, d_actual;
        std::unique_ptr<CudaStream> stream;
        std::unique_ptr<CudaEvent> done;
        bool inFlight = false;
        size_t chunkCount = 0;
        size_t uncompBytes = 0;
        size_t poolIdx = SIZE_MAX;   // pinned download buffer for this flight
    };

    static uint64_t subUncompBytes(const BatchedHeader& h, size_t c0, size_t cn) {
        uint64_t end = std::min<uint64_t>((uint64_t)(c0 + cn) * CHUNK_SIZE,
                                          h.uncompressedSize);
        return end - (uint64_t)c0 * CHUNK_SIZE;
    }

    bool ensureResources(AlgoType algo, size_t totalChunks) {
        // Slots never need to exceed the volume's chunk count (small archives
        // shouldn't pay full-size pinned allocations). Grow-only across
        // volumes of one operation.
        size_t wantChunks = std::min(subBatchChunksFor(algo), totalChunks);
        if (ready_ && algo == algo_ && wantChunks <= chunksPerSub_) return true;
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
            cudaGetLastError();
            return false;
        }
        ready_ = false;
        algo_ = algo;
        chunksPerSub_ = wantChunks;
        if (algo == ALGO_LZ4) {
            NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                CHUNK_SIZE, nvcompBatchedLZ4CompressDefaultOpts, &maxCompChunk_));
        } else if (algo == ALGO_SNAPPY) {
            NVCOMP_CHECK(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
                CHUNK_SIZE, nvcompBatchedSnappyCompressDefaultOpts, &maxCompChunk_));
        } else {
            NVCOMP_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                CHUNK_SIZE, nvcompBatchedZstdCompressDefaultOpts, &maxCompChunk_));
        }
        auto queryTemp = [&](size_t chunks) {
            size_t t = 0;
            size_t bytes = chunks * CHUNK_SIZE;
            if (algo == ALGO_LZ4) {
                NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
                    chunks, CHUNK_SIZE, nvcompBatchedLZ4DecompressDefaultOpts, &t, bytes));
            } else if (algo == ALGO_SNAPPY) {
                NVCOMP_CHECK(nvcompBatchedSnappyDecompressGetTempSizeAsync(
                    chunks, CHUNK_SIZE, nvcompBatchedSnappyDecompressDefaultOpts, &t, bytes));
            } else {
                NVCOMP_CHECK(nvcompBatchedZstdDecompressGetTempSizeAsync(
                    chunks, CHUNK_SIZE, nvcompBatchedZstdDecompressDefaultOpts, &t, bytes));
            }
            return t;
        };
        tempBytes_ = queryTemp(chunksPerSub_);
        depth_ = PIPELINE_DEPTH;

        // Fit into free VRAM: shrink depth, then sub-batch size. Bail out to
        // the CPU path when even the minimum footprint doesn't fit.
        {
            size_t freeMem = 0, totalMem = 0;
            if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            const uint64_t slack = 256ull << 20;
            auto need = [&]() -> uint64_t {
                uint64_t perSlot = (uint64_t)maxCompChunk_ * chunksPerSub_   // d_in
                    + (uint64_t)chunksPerSub_ * CHUNK_SIZE                   // d_out
                    + tempBytes_
                    + 6ull * chunksPerSub_ * sizeof(size_t);
                return depth_ * perSlot + slack;
            };
            while (need() > freeMem && (depth_ > 1 || chunksPerSub_ > 256)) {
                if (depth_ > 1) {
                    depth_--;
                } else {
                    chunksPerSub_ /= 2;
                    tempBytes_ = queryTemp(chunksPerSub_);
                }
            }
            if (need() > freeMem) {
                if (isVerbose()) {
                    std::cout << "GPU decompression pipeline needs ~"
                              << (need() >> 20) << " MB VRAM, "
                              << (freeMem >> 20) << " MB free; using CPU\n";
                }
                return false;
            }
        }
        const size_t subBytes = chunksPerSub_ * CHUNK_SIZE;
        // Download buffers: depth + spare so the streaming consumer can trail
        // the GPU by a few sub-batches before backpressure blocks acquire().
        size_t spare = depth_;
        if (const char* e = std::getenv("NVCOMP_DECOMP_SPARE_BUFS")) {
            long v = std::atol(e);
            if (v >= 0) spare = static_cast<size_t>(v);
        }
        pool_.init(depth_ + spare, subBytes);
        slots_.clear();
        slots_.resize(depth_);
        for (auto& s : slots_) {
            s.h_in.reserve(maxCompChunk_ * chunksPerSub_);
            s.h_in_ptrs.reserve(sizeof(void*) * chunksPerSub_);
            s.h_in_sizes.reserve(sizeof(size_t) * chunksPerSub_);
            s.h_out_sizes.reserve(sizeof(size_t) * chunksPerSub_);
            s.h_statuses.reserve(sizeof(nvcompStatus_t) * chunksPerSub_);
            s.h_actual.reserve(sizeof(size_t) * chunksPerSub_);
            s.d_in.reserve(maxCompChunk_ * chunksPerSub_);
            s.d_out.reserve(subBytes);
            s.d_temp.reserve(tempBytes_);
            s.d_in_ptrs.reserve(sizeof(void*) * chunksPerSub_);
            s.d_in_sizes.reserve(sizeof(size_t) * chunksPerSub_);
            s.d_out_ptrs.reserve(sizeof(void*) * chunksPerSub_);
            s.d_out_sizes.reserve(sizeof(size_t) * chunksPerSub_);
            s.d_statuses.reserve(sizeof(nvcompStatus_t) * chunksPerSub_);
            s.d_actual.reserve(sizeof(size_t) * chunksPerSub_);
            s.stream = std::make_unique<CudaStream>();
            s.done = std::make_unique<CudaEvent>();
            // Output chunk pointers are position-invariant per slot.
            std::vector<void*> out_ptrs(chunksPerSub_);
            for (size_t i = 0; i < chunksPerSub_; i++) {
                out_ptrs[i] = s.d_out.bytes() + i * CHUNK_SIZE;
            }
            CUDA_CHECK(cudaMemcpy(s.d_out_ptrs.get(), out_ptrs.data(),
                                  sizeof(void*) * chunksPerSub_, cudaMemcpyHostToDevice));
        }
        ready_ = true;
        return true;
    }

    void submitSlot(Slot& s, AlgoType algo, const uint64_t* chunkSizes,
                    size_t cn, uint64_t uncompBytes, uint64_t compBytes) {
        cudaStream_t st = *s.stream;
        CUDA_CHECK(cudaMemcpyAsync(s.d_in.get(), s.h_in.bytes(), compBytes,
                                   cudaMemcpyHostToDevice, st));

        void** in_ptrs = reinterpret_cast<void**>(s.h_in_ptrs.bytes());
        size_t* in_sizes = reinterpret_cast<size_t*>(s.h_in_sizes.bytes());
        size_t* out_sizes = reinterpret_cast<size_t*>(s.h_out_sizes.bytes());
        uint64_t off = 0;
        for (size_t i = 0; i < cn; i++) {
            in_ptrs[i] = s.d_in.bytes() + off;
            in_sizes[i] = chunkSizes[i];
            off += chunkSizes[i];
            out_sizes[i] = CHUNK_SIZE;
        }
        out_sizes[cn - 1] = uncompBytes - (cn - 1) * (uint64_t)CHUNK_SIZE;
        CUDA_CHECK(cudaMemcpyAsync(s.d_in_ptrs.get(), in_ptrs,
                                   sizeof(void*) * cn, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(s.d_in_sizes.get(), in_sizes,
                                   sizeof(size_t) * cn, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(s.d_out_sizes.get(), out_sizes,
                                   sizeof(size_t) * cn, cudaMemcpyHostToDevice, st));

        if (algo == ALGO_LZ4) {
            NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
                s.d_in_ptrs.get<const void* const>(), s.d_in_sizes.get<size_t>(),
                s.d_out_sizes.get<size_t>(), s.d_actual.get<size_t>(), cn,
                s.d_temp.get(), tempBytes_, s.d_out_ptrs.get<void* const>(),
                nvcompBatchedLZ4DecompressDefaultOpts,
                s.d_statuses.get<nvcompStatus_t>(), st));
        } else if (algo == ALGO_SNAPPY) {
            NVCOMP_CHECK(nvcompBatchedSnappyDecompressAsync(
                s.d_in_ptrs.get<const void* const>(), s.d_in_sizes.get<size_t>(),
                s.d_out_sizes.get<size_t>(), s.d_actual.get<size_t>(), cn,
                s.d_temp.get(), tempBytes_, s.d_out_ptrs.get<void* const>(),
                nvcompBatchedSnappyDecompressDefaultOpts,
                s.d_statuses.get<nvcompStatus_t>(), st));
        } else {
            NVCOMP_CHECK(nvcompBatchedZstdDecompressAsync(
                s.d_in_ptrs.get<const void* const>(), s.d_in_sizes.get<size_t>(),
                s.d_out_sizes.get<size_t>(), s.d_actual.get<size_t>(), cn,
                s.d_temp.get(), tempBytes_, s.d_out_ptrs.get<void* const>(),
                nvcompBatchedZstdDecompressDefaultOpts,
                s.d_statuses.get<nvcompStatus_t>(), st));
        }
        CUDA_CHECK(cudaMemcpyAsync(s.h_statuses.bytes(), s.d_statuses.get(),
                                   sizeof(nvcompStatus_t) * cn, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(s.h_actual.bytes(), s.d_actual.get(),
                                   sizeof(size_t) * cn, cudaMemcpyDeviceToHost, st));
        // Download into a pool buffer (blocks when the consumer is behind).
        s.poolIdx = pool_.acquire();
        CUDA_CHECK(cudaMemcpyAsync(pool_.ptr(s.poolIdx), s.d_out.get(), uncompBytes,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(*s.done, st));
        s.inFlight = true;
        s.chunkCount = cn;
        s.uncompBytes = uncompBytes;
    }

    void retireSlot(Slot& s, DecompressSink& sink) {
        using tclk = std::chrono::steady_clock;
        auto w0 = tclk::now();
        CUDA_CHECK(cudaEventSynchronize(*s.done));
        auto w1 = tclk::now();
        s.inFlight = false;
        const nvcompStatus_t* st = reinterpret_cast<const nvcompStatus_t*>(s.h_statuses.bytes());
        const size_t* actual = reinterpret_cast<const size_t*>(s.h_actual.bytes());
        const size_t* expect = reinterpret_cast<const size_t*>(s.h_out_sizes.bytes());
        for (size_t i = 0; i < s.chunkCount; i++) {
            if (st[i] != nvcompSuccess || actual[i] != expect[i]) {
                throw std::runtime_error("chunk decompression failed");
            }
        }
        size_t idx = s.poolIdx;
        s.poolIdx = SIZE_MAX;
        PinnedOutPool* pool = &pool_;
        sink.consume(pool_.ptr(idx), s.uncompBytes,
                     [pool, idx] { pool->release(idx); });
        auto w2 = tclk::now();
        dbgWait_ += std::chrono::duration<double>(w1 - w0).count();
        dbgInsert_ += std::chrono::duration<double>(w2 - w1).count();
    }

    // After a failure with work possibly still in flight, wait everything out
    // so slot buffers are safe to reuse (or free), return unconsumed pool
    // buffers, then clear sticky errors.
    void quiesce() {
        for (auto& s : slots_) {
            if (s.stream) cudaStreamSynchronize(*s.stream);
            s.inFlight = false;
            if (s.poolIdx != SIZE_MAX) {
                pool_.release(s.poolIdx);
                s.poolIdx = SIZE_MAX;
            }
        }
        cudaGetLastError();
    }

    bool ready_ = false;
    AlgoType algo_ = ALGO_UNKNOWN;
    size_t chunksPerSub_ = 0;
    size_t maxCompChunk_ = 0;
    size_t tempBytes_ = 0;
    size_t depth_ = 0;
    std::vector<Slot> slots_;
    PinnedOutPool pool_;
    double dbgWait_ = 0.0, dbgInsert_ = 0.0, dbgRead_ = 0.0; // NVCOMP_PHASE_DEBUG
};

// Extraction writer-pool size: measured on NVMe, file creation scales ~4x up
// to 8 threads and regresses past that. Override with NVCOMP_EXTRACT_THREADS
// (0 = write inline on the extractor thread).
static size_t extractWriterThreads() {
    size_t n = std::min<size_t>(8, std::max<size_t>(1, std::thread::hardware_concurrency() / 2));
    if (const char* e = std::getenv("NVCOMP_EXTRACT_THREADS")) {
        long v = std::atol(e);
        if (v >= 0) n = static_cast<size_t>(v);
    }
    return n;
}

void decompressGPUBatched(AlgoType algo, const std::string& inputFile, const std::string& outputPath, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    (void)makeThrottledCallback(callback); // throttle reserved for future per-block callbacks

    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read only the manifest prefix (and metadata array when present).
        // Previously this readFile()'d the entire first volume -- up to
        // several GB -- just to parse ~48 bytes plus the metadata table.
        auto readStart = clock::now();
        std::error_code fsEc;
        uint64_t firstVolumeSize = fs::file_size(fs::path(volumeFiles[0]), fsEc);
        if (fsEc) {
            throw std::runtime_error("Cannot stat volume file: " + volumeFiles[0]);
        }
        if (firstVolumeSize < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }

        VolumeManifest manifest;
        std::vector<VolumeMetadata> volumeMetadata;
        {
            std::ifstream mf(fs::path(volumeFiles[0]), std::ios::binary);
            if (!mf.is_open()) {
                throw std::runtime_error("Cannot open volume file: " + volumeFiles[0]);
            }
            if (!mf.read(reinterpret_cast<char*>(&manifest), sizeof(VolumeManifest))) {
                throw std::runtime_error("Invalid volume file: failed reading manifest");
            }
            if (manifest.magic == VOLUME_MAGIC) {
                uint64_t prefix = sizeof(VolumeManifest)
                    + sizeof(VolumeMetadata) * (uint64_t)manifest.volumeCount;
                if (firstVolumeSize < prefix) {
                    throw std::runtime_error("Invalid volume file: truncated metadata");
                }
                volumeMetadata.resize(manifest.volumeCount);
                if (!mf.read(reinterpret_cast<char*>(volumeMetadata.data()),
                             sizeof(VolumeMetadata) * manifest.volumeCount)) {
                    throw std::runtime_error("Invalid volume file: failed reading metadata");
                }
            }
        }
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        }

        const bool verbose = isVerbose();
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
            if (detectedAlgo != ALGO_UNKNOWN) {
                algo = detectedAlgo;
                if (verbose) {
                    std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << "\n";
                }
            }
            
            if (verbose) {
                std::cout << "Decompressing (" << algoToString(algo) << ")...\n";
            }
            
            auto computeStart = clock::now();
            // Decompress and extract concurrently: the pipeline streams the
            // file from disk, the sink extracts on its own thread + writer
            // pool. CPU fallback resumes the same stream (skipping any bytes
            // the GPU already delivered before failing).
            StreamingExtractSink sink(outputPath, extractWriterThreads());
            BatchedDecompressPipeline gpuPipe;
            if (!gpuPipe.decompressFile(volumeFiles[0], 0, sink)) {
                uint64_t partial = sink.consumed();
                auto firstVolumeData = readFile(volumeFiles[0]);
                auto archiveData = decompressBatchedFormatCPU(algo, firstVolumeData);
                if (partial > archiveData.size()) {
                    throw std::runtime_error("Decompression resume mismatch");
                }
                auto keep = std::make_shared<std::vector<uint8_t>>(std::move(archiveData));
                sink.consume(keep->data() + partial, keep->size() - partial,
                             [keep] {});
            }
            sink.finish();
            auto computeEnd = clock::now();

            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            size_t decompSize = sink.consumed();
            if (verbose) {
                std::cout << "Decompressed size: " << decompSize << " bytes\n";
                std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
            }

            if (outStats) {
                outStats->computeSec += duration;   // extraction overlaps decompression
                outStats->inputBytes = decompSize;  // uncompressed payload
                outStats->outputBytes = firstVolumeSize;
                outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
                finalizeStats(*outStats);
                std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
            }
            return;
        }
        
        // Multi-volume archive
        if (verbose) {
            std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)\n";
        }
        
        // No up-front whole-volume VRAM gate here anymore: the pipelined GPU
        // decompressor sizes itself to free VRAM (shrinking depth/sub-batch)
        // and falls back to the CPU decoder per volume when it can't fit.
        if (verbose) {
            std::cout << "Using GPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")...\n";
        }
        
        // Volume metadata was parsed with the manifest above.
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes through the pipelined GPU engine (buffers and
        // streams reused across volumes) into a streaming extraction sink:
        // files are written while later sub-batches/volumes still decompress.
        // The NVAR stream continues seamlessly across volume boundaries, and
        // the per-volume CPU fallback resumes the same stream (skipping bytes
        // the GPU already delivered).
        double totalDuration = 0;
        uint64_t totalCompressedRead = firstVolumeSize;
        StreamingExtractSink sink(outputPath, extractWriterThreads());
        BatchedDecompressPipeline gpuPipe;

        if (verbose) {
            std::cout << "Decompressing " << volumeFiles.size() << " volume(s)...\n";
        }

        for (size_t i = 0; i < volumeFiles.size(); i++) {
            if (verbose && ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1)) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }

            // Manifest and metadata prefix exists only in the first volume.
            size_t dataOffset = (i == 0)
                ? sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount
                : 0;

            auto computeStart = clock::now();
            uint64_t volConsumedBefore = sink.consumed();
            if (!gpuPipe.decompressFile(volumeFiles[i], dataOffset, sink)) {
                uint64_t partial = sink.consumed() - volConsumedBefore;
                auto volumeData = readFile(volumeFiles[i]);
                std::vector<uint8_t> payload(volumeData.begin() + dataOffset, volumeData.end());
                auto decompressed = decompressBatchedFormatCPU(
                    static_cast<AlgoType>(manifest.algorithm), payload);
                if (partial > decompressed.size()) {
                    throw std::runtime_error("Decompression resume mismatch");
                }
                auto keep = std::make_shared<std::vector<uint8_t>>(std::move(decompressed));
                sink.consume(keep->data() + partial, keep->size() - partial,
                             [keep] {});
            }
            auto computeEnd = clock::now();
            if (i > 0) {
                std::error_code ec;
                totalCompressedRead += fs::file_size(fs::path(volumeFiles[i]), ec);
            }

            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            totalDuration += duration;
            if (outStats) outStats->computeSec += duration;
        }

        // Drain the extraction tail (most writes already overlapped).
        auto writeStart = clock::now();
        sink.finish();
        double drainSec = std::chrono::duration<double>(clock::now() - writeStart).count();

        if (verbose) {
            std::cout << "\n";
            std::cout << "\n=== Decompression Summary ===\n";
            std::cout << "Total decompressed: " << sink.consumed() << " bytes\n";
            std::cout << "Total time: " << totalDuration << "s ("
                      << (sink.consumed() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)\n";
        }

        if (outStats) {
            outStats->inputBytes = sink.consumed();
            outStats->outputBytes = totalCompressedRead;
            outStats->writeSec += drainSec;
            outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
            finalizeStats(*outStats);
            std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
        }
        return;
    }
    
    // Single file (non-volume)
    const bool verbose = isVerbose();
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        if (verbose) {
            std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << "\n";
        }
    }
    
    if (verbose) {
        std::cout << "Decompressing (" << algoToString(algo) << ")...\n";
    }
    
    if (outStats) {
        std::error_code ec;
        outStats->outputBytes = fs::file_size(fs::path(inputFile), ec);
    }

    auto computeStart = clock::now();

    // Decompress and extract concurrently: the pipeline streams the file from
    // disk, the sink extracts on its own thread + writer pool. The CPU
    // fallback (also handles CPU-compressed single-chunk and standard
    // formats) resumes the same stream past any bytes the GPU delivered.
    StreamingExtractSink sink(outputPath, extractWriterThreads());
    BatchedDecompressPipeline gpuPipe;
    if (!gpuPipe.decompressFile(inputFile, 0, sink)) {
        uint64_t partial = sink.consumed();
        auto readStart = clock::now();
        auto compressedData = readFile(inputFile);
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        }
        auto archiveData = decompressBatchedFormatCPU(algo, compressedData);
        if (partial > archiveData.size()) {
            throw std::runtime_error("Decompression resume mismatch");
        }
        auto keep = std::make_shared<std::vector<uint8_t>>(std::move(archiveData));
        sink.consume(keep->data() + partial, keep->size() - partial, [keep] {});
    }
    sink.finish();

    auto computeEnd = clock::now();
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();

    size_t decompSize = sink.consumed();
    if (verbose) {
        std::cout << "Decompressed size: " << decompSize << " bytes\n";
        std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (outStats) {
        outStats->computeSec += duration;   // extraction overlaps decompression
        outStats->inputBytes = decompSize;
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
    }
}

// ============================================================================
// GPU Manager API Compression
// ============================================================================

// Internal function that compresses in-memory archive data
static void compressGPUManagerFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData,
                                         const std::string& outputFile, uint64_t maxVolumeSize,
                                         CompressionStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Using GPU manager compression (" << algoToString(algo) << ")...\n";
    }
    
    size_t totalSize = archiveData.size();
    if (verbose) {
        std::cout << "Archive size: " << totalSize << " bytes\n";
    }
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        size_t inputSize = volumes[0].size();
        std::vector<uint8_t> inputData = volumes[0];
    
    auto prepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    std::shared_ptr<nvcomp::nvcompManagerBase> manager;
    
    if (algo == ALGO_GDEFLATE) {
        manager = std::make_shared<nvcomp::GdeflateManager>(
            CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
    } else if (algo == ALGO_ANS) {
        manager = std::make_shared<nvcomp::ANSManager>(
            CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
    } else if (algo == ALGO_BITCOMP) {
        manager = std::make_shared<nvcomp::BitcompManager>(
            CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
    }
    
    nvcomp::CompressionConfig comp_config = manager->configure_compression(inputSize);
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
    
    auto computeStart = clock::now();
    if (stats) stats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    
    manager->compress(d_input, d_output, comp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto computeEnd = clock::now();
    
    size_t compSize = manager->get_compressed_output_size(d_output);
    
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    if (verbose) {
        std::cout << "Compressed size: " << compSize << " bytes\n";
        std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)inputSize / compSize << "x\n";
        std::cout << "Time: " << duration << "s (" << (inputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (stats) stats->computeSec += duration;
    
    std::vector<uint8_t> outputData(compSize);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
    
    auto writeStart = clock::now();
    writeFile(outputFile, outputData.data(), outputData.size());
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        stats->outputBytes = outputData.size();
    }
    
    // Cleanup: destroy manager before stream (manager references stream)
    manager.reset();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
        return;
    }
    
    // Multi-volume compression
    if (verbose) {
        std::cout << "\nCompressing " << volumes.size() << " volume(s)...\n";
    }
    
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t uncompressedOffset = 0;
    double totalDuration = 0;
    size_t totalCompressedSize = 0;
    
    for (size_t volIdx = 0; volIdx < volumes.size(); volIdx++) {
        if (verbose) {
            std::cout << "\r  Processing volume " << (volIdx + 1) << "/" << volumes.size() << "..." << std::flush;
        }
        
        std::vector<uint8_t>& inputData = volumes[volIdx];
        size_t inputSize = inputData.size();
        
        auto volPrepareStart = clock::now();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        uint8_t* d_input;
        CUDA_CHECK(cudaMalloc(&d_input, inputSize));
        CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
        
        std::shared_ptr<nvcomp::nvcompManagerBase> manager;
        
        if (algo == ALGO_GDEFLATE) {
            manager = std::make_shared<nvcomp::GdeflateManager>(
                CHUNK_SIZE, nvcompBatchedGdeflateCompressDefaultOpts, nvcompBatchedGdeflateDecompressDefaultOpts, stream);
        } else if (algo == ALGO_ANS) {
            manager = std::make_shared<nvcomp::ANSManager>(
                CHUNK_SIZE, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream);
        } else if (algo == ALGO_BITCOMP) {
            manager = std::make_shared<nvcomp::BitcompManager>(
                CHUNK_SIZE, nvcompBatchedBitcompCompressDefaultOpts, nvcompBatchedBitcompDecompressDefaultOpts, stream);
        }
        
        nvcomp::CompressionConfig comp_config = manager->configure_compression(inputSize);
        
        uint8_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
        
        auto computeStartV = clock::now();
        if (stats) stats->prepareSec += std::chrono::duration<double>(computeStartV - volPrepareStart).count();
        
        manager->compress(d_input, d_output, comp_config);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto computeEndV = clock::now();
        
        size_t compSize = manager->get_compressed_output_size(d_output);
        
        double duration = std::chrono::duration<double>(computeEndV - computeStartV).count();
        totalDuration += duration;
        if (stats) stats->computeSec += duration;
        
        std::vector<uint8_t> outputData(compSize);
        CUDA_CHECK(cudaMemcpy(outputData.data(), d_output, compSize, cudaMemcpyDeviceToHost));
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = volIdx + 1;
        meta.compressedSize = compSize;
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = inputSize;
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += inputSize;
        totalCompressedSize += compSize;
        
        // Write volume file
        auto volWriteStart = clock::now();
        std::string volumeFile = generateVolumeFilename(outputFile, volIdx + 1);
        writeFile(volumeFile, outputData.data(), outputData.size());
        if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - volWriteStart).count();
        
        // Cleanup: destroy manager before stream (manager references stream)
        manager.reset();
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }
    
    if (verbose) {
        std::cout << "\r  Processing volume " << volumes.size() << "/" << volumes.size() << "... Done!\n";
    }
    
    // Create and prepend manifest to first volume
    
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumes.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalSize;
    manifest.reserved = 0;
    
    // Read first volume
    auto fixupReadStart = clock::now();
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    auto firstVolumeData = readFile(firstVolumeFile);
    if (stats) stats->readSec += std::chrono::duration<double>(clock::now() - fixupReadStart).count();
    
    // Create new first volume with manifest
    std::vector<uint8_t> newFirstVolume;
    
    // Add manifest header
    const uint8_t* manifestBytes = reinterpret_cast<const uint8_t*>(&manifest);
    newFirstVolume.insert(newFirstVolume.end(), manifestBytes, manifestBytes + sizeof(VolumeManifest));
    
    // Add volume metadata array
    const uint8_t* metadataBytes = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    newFirstVolume.insert(newFirstVolume.end(), metadataBytes, 
                         metadataBytes + sizeof(VolumeMetadata) * volumeMetadata.size());
    
    // Add original compressed data
    newFirstVolume.insert(newFirstVolume.end(), firstVolumeData.begin(), firstVolumeData.end());
    
    // Write updated first volume
    auto fixupWriteStart = clock::now();
    writeFile(firstVolumeFile, newFirstVolume.data(), newFirstVolume.size());
    if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - fixupWriteStart).count();
    
    // Update metadata for first volume
    volumeMetadata[0].compressedSize = newFirstVolume.size();
    totalCompressedSize = totalCompressedSize - firstVolumeData.size() + newFirstVolume.size();
    
    if (stats) stats->outputBytes = totalCompressedSize;

    std::cout << "\n=== Multi-Volume Compression SUCCESSFUL ===" << std::endl;
    std::cout << "Volumes created: " << volumes.size() << std::endl;
    std::cout << "Total uncompressed: " << (totalSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Total compressed: " << (totalCompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Overall ratio: " << std::fixed << std::setprecision(2) 
              << (double)totalSize / totalCompressedSize << "x" << std::endl;
    std::cout << "Total time: " << totalDuration << "s (" 
              << (totalSize / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)" << std::endl;
}

// Public wrapper for single file/folder compression
void compressGPUManager(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    auto readStart = clock::now();
    std::vector<uint8_t> archiveData;
    if (isDirectory(inputPath)) {
        archiveData = createArchiveFromFolder(inputPath, throttled);
    } else {
        archiveData = createArchiveFromFile(inputPath, throttled);
    }
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->inputBytes = archiveData.size();
    }
    
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void compressGPUManagerFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    if (isVerbose()) {
        std::cout << "Compressing file list (" << filePaths.size() << " files)...\n";
    }
    
    auto readStart = clock::now();
    std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, throttled);
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->inputBytes = archiveData.size();
    }
    
    compressGPUManagerFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

// ============================================================================
// GPU Manager API Decompression
// ============================================================================

void decompressGPUManager(const std::string& inputFile, const std::string& outputPath, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    (void)callback;

    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        // Read manifest from first volume
        auto manifestReadStart = clock::now();
        auto firstVolumeData = readFile(volumeFiles[0]);
        if (outStats) outStats->readSec += std::chrono::duration<double>(clock::now() - manifestReadStart).count();
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file: too small for manifest");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        const bool verbose = isVerbose();
        if (manifest.magic != VOLUME_MAGIC) {
            // Not a multi-volume archive, treat as single file
            if (verbose) {
                std::cout << "Using GPU manager decompression (auto-detect)...\n";
            }
            
            auto prepareStart = clock::now();
            size_t inputSize = firstVolumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, firstVolumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = nvcomp::create_manager(d_input, stream);
            nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            if (verbose) {
                std::cout << "Detected original size: " << outputSize << " bytes\n";
            }
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto computeStart = clock::now();
            if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            if (verbose) {
                std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
            }
            if (outStats) {
                outStats->computeSec += duration;
                outStats->inputBytes = outputSize;
                outStats->outputBytes = inputSize;
            }
            
            std::vector<uint8_t> archiveData(outputSize);
            CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            // Cleanup: destroy manager before stream (manager references stream)
            manager.reset();
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
            
            auto writeStart = clock::now();
            extractArchive(archiveData, outputPath);
            if (outStats) {
                outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
                outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
                finalizeStats(*outStats);
                std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
            }
            return;
        }
        
        // Multi-volume archive
        if (verbose) {
            std::cout << "Multi-volume archive detected: " << manifest.volumeCount << " volume(s)\n";
        }
        
        // Check GPU memory
        if (!checkGPUMemoryForVolume(manifest.volumeSize)) {
            std::cout << "Insufficient GPU memory for " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB volumes (need ~" << (manifest.volumeSize * 2.1 / (1024.0 * 1024.0 * 1024.0)) 
                      << " GB VRAM)." << std::endl;
            throw std::runtime_error("Insufficient GPU memory for GPU-only algorithm. Cannot fall back to CPU.");
        }
        
        if (verbose) {
            std::cout << "Using GPU manager decompression...\n";
        }
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            throw std::runtime_error("Missing volume files");
        }
        
        // Decompress all volumes
        std::vector<uint8_t> fullArchive;
        fullArchive.reserve(manifest.totalUncompressedSize);
        double totalDuration = 0;
        uint64_t totalCompressedRead = firstVolumeData.size();
        
        if (verbose) {
            std::cout << "Decompressing " << volumeFiles.size() << " volume(s)...\n";
        }
        
        for (size_t i = 0; i < volumeFiles.size(); i++) {
            if (verbose && ((i + 1) % 100 == 0 || i == volumeFiles.size() - 1)) {
                std::cout << "\r  Decompressing... " << (i + 1) << "/" << volumeFiles.size() << std::flush;
            }
            
            auto volReadStart = clock::now();
            auto volumeData = readFile(volumeFiles[i]);
            if (outStats && i > 0) {
                outStats->readSec += std::chrono::duration<double>(clock::now() - volReadStart).count();
                totalCompressedRead += volumeData.size();
            }
            
            // Skip manifest and metadata in first volume
            size_t dataOffset = 0;
            if (i == 0) {
                dataOffset = sizeof(VolumeManifest) + sizeof(VolumeMetadata) * manifest.volumeCount;
                volumeData = std::vector<uint8_t>(volumeData.begin() + dataOffset, volumeData.end());
            }
            
            auto prepareStart = clock::now();
            size_t inputSize = volumeData.size();
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            
            uint8_t* d_input;
            CUDA_CHECK(cudaMalloc(&d_input, inputSize));
            CUDA_CHECK(cudaMemcpyAsync(d_input, volumeData.data(), inputSize, cudaMemcpyHostToDevice, stream));
            
            auto manager = nvcomp::create_manager(d_input, stream);
            nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
            size_t outputSize = decomp_config.decomp_data_size;
            
            uint8_t* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, outputSize));
            
            auto computeStart = clock::now();
            if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
            manager->decompress(d_output, d_input, decomp_config);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            totalDuration += duration;
            if (outStats) outStats->computeSec += duration;
            
            std::vector<uint8_t> decompressed(outputSize);
            CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
            
            // Cleanup: destroy manager before stream (manager references stream)
            manager.reset();
            
            cudaFree(d_input);
            cudaFree(d_output);
            cudaStreamDestroy(stream);
        }
        
        if (verbose) {
            std::cout << "\n";
            std::cout << "\n=== Decompression Summary ===\n";
            std::cout << "Total decompressed: " << fullArchive.size() << " bytes\n";
            std::cout << "Total time: " << totalDuration << "s ("
                      << (fullArchive.size() / (1024.0 * 1024.0 * 1024.0)) / totalDuration << " GB/s)\n";
        }
        
        if (outStats) {
            outStats->inputBytes = fullArchive.size();
            outStats->outputBytes = totalCompressedRead;
        }

        auto writeStart = clock::now();
        extractArchive(fullArchive, outputPath);
        if (outStats) {
            outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
            outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
            finalizeStats(*outStats);
            std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
        }
        return;
    }
    
    // Single file (non-volume)
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Using GPU manager decompression (auto-detect)...\n";
    }
    
    auto readStart = clock::now();
    auto inputData = readFile(inputFile);
    size_t inputSize = inputData.size();
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->outputBytes = inputSize;
    }
    
    auto prepareStart = clock::now();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));
    
    auto manager = nvcomp::create_manager(d_input, stream);
    
    nvcomp::DecompressionConfig decomp_config = manager->configure_decompression(d_input);
    size_t outputSize = decomp_config.decomp_data_size;
    if (verbose) {
        std::cout << "Detected original size: " << outputSize << " bytes\n";
    }
    
    uint8_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    
    auto computeStart = clock::now();
    if (outStats) outStats->prepareSec += std::chrono::duration<double>(computeStart - prepareStart).count();
    
    manager->decompress(d_output, d_input, decomp_config);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto computeEnd = clock::now();
    
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    if (verbose) {
        std::cout << "Time: " << duration << "s (" << (outputSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (outStats) {
        outStats->computeSec += duration;
        outStats->inputBytes = outputSize;
    }
    
    std::vector<uint8_t> archiveData(outputSize);
    CUDA_CHECK(cudaMemcpy(archiveData.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Cleanup: destroy manager before stream (manager references stream)
    manager.reset();
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    auto writeStart = clock::now();
    extractArchive(archiveData, outputPath);
    if (outStats) {
        outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
    }
}

// ============================================================================
// List Compressed Archive
// ============================================================================

void listCompressedArchive(AlgoType algo, const std::string& inputFile, bool useCPU, bool cudaAvailable) {
    // Detect volume files
    auto volumeFiles = detectVolumeFiles(inputFile);
    
    // Check if multi-volume
    if (volumeFiles.size() > 1 || isVolumeFile(volumeFiles[0])) {
        std::cout << "Multi-volume archive detected: " << volumeFiles.size() << " volume(s)" << std::endl;
        
        // Read manifest from first volume
        auto firstVolumeData = readFile(volumeFiles[0]);
        
        if (firstVolumeData.size() < sizeof(VolumeManifest)) {
            throw std::runtime_error("Invalid volume file");
        }
        
        VolumeManifest manifest;
        std::memcpy(&manifest, firstVolumeData.data(), sizeof(VolumeManifest));
        
        if (manifest.magic != VOLUME_MAGIC) {
            throw std::runtime_error("Invalid volume manifest");
        }
        
        std::cout << "Algorithm: " << algoToString(static_cast<AlgoType>(manifest.algorithm)) << std::endl;
        std::cout << "Volume size: " << (manifest.volumeSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "Total uncompressed: " << (manifest.totalUncompressedSize / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        std::cout << "\nVolume breakdown:" << std::endl;
        for (const auto& meta : volumeMetadata) {
            std::cout << "  Volume " << meta.volumeIndex << ": " 
                      << (meta.compressedSize / (1024.0 * 1024.0)) << " MB compressed, "
                      << (meta.uncompressedSize / (1024.0 * 1024.0)) << " MB uncompressed" << std::endl;
        }
        
        std::cout << "\nListing archive contents requires decompression..." << std::endl;
        return;
    }
    
    // Single file
    auto inputData = readFile(inputFile);
    
    // Try to decompress and list
    std::cout << "Listing contents of single file archive..." << std::endl;
    std::cout << "Full listing requires decompression implementation" << std::endl;
}

} // namespace nvcomp_core


