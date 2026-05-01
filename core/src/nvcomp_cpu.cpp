#include "nvcomp_core.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

// CPU compression libraries
#include "lz4.h"
#include "lz4hc.h"
#include "snappy.h"
#include "zstd.h"

namespace nvcomp_core {

// ============================================================================
// Algorithm Utilities
// ============================================================================

AlgoType parseAlgorithm(const std::string& algo) {
    if (algo == "lz4") return ALGO_LZ4;
    if (algo == "snappy") return ALGO_SNAPPY;
    if (algo == "zstd") return ALGO_ZSTD;
    if (algo == "gdeflate") return ALGO_GDEFLATE;
    if (algo == "ans") return ALGO_ANS;
    if (algo == "bitcomp") return ALGO_BITCOMP;
    return ALGO_UNKNOWN;
}

std::string algoToString(AlgoType algo) {
    switch(algo) {
        case ALGO_LZ4: return "lz4";
        case ALGO_SNAPPY: return "snappy";
        case ALGO_ZSTD: return "zstd";
        case ALGO_GDEFLATE: return "gdeflate";
        case ALGO_ANS: return "ans";
        case ALGO_BITCOMP: return "bitcomp";
        default: return "unknown";
    }
}

bool isCrossCompatible(AlgoType algo) {
    return algo == ALGO_LZ4 || algo == ALGO_SNAPPY || algo == ALGO_ZSTD;
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
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

static std::vector<uint8_t> compressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t compSize = 0;
    
    if (algo == ALGO_LZ4) {
        size_t maxSize = LZ4_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(inputData.data()),
            reinterpret_cast<char*>(outputData.data()),
            inputSize,
            maxSize,
            LZ4HC_CLEVEL_DEFAULT
        );
        if (compSize == 0) {
            throw std::runtime_error("LZ4 CPU compression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t maxSize = snappy::MaxCompressedLength(inputSize);
        outputData.resize(maxSize);
        snappy::RawCompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data()),
            &compSize
        );
        if (compSize == 0) {
            throw std::runtime_error("Snappy CPU compression failed");
        }
    } else if (algo == ALGO_ZSTD) {
        size_t maxSize = ZSTD_compressBound(inputSize);
        outputData.resize(maxSize);
        compSize = ZSTD_compress(
            outputData.data(),
            maxSize,
            inputData.data(),
            inputSize,
            ZSTD_CLEVEL_DEFAULT
        );
        if (ZSTD_isError(compSize)) {
            throw std::runtime_error("Zstd CPU compression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU compression");
    }
    
    outputData.resize(compSize);
    return outputData;
}

static std::vector<uint8_t> decompressDataCPU(AlgoType algo, const std::vector<uint8_t>& inputData) {
    size_t inputSize = inputData.size();
    std::vector<uint8_t> outputData;
    size_t decompSize = 0;
    
    if (algo == ALGO_LZ4) {
        // Try different output sizes
        for (size_t multiplier = 10; multiplier <= 1000; multiplier *= 10) {
            outputData.resize(inputSize * multiplier);
            int result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(inputData.data()),
                reinterpret_cast<char*>(outputData.data()),
                inputSize,
                outputData.size()
            );
            if (result > 0) {
                decompSize = result;
                break;
            }
        }
        if (decompSize == 0) {
            throw std::runtime_error("LZ4 CPU decompression failed");
        }
    } else if (algo == ALGO_SNAPPY) {
        size_t uncompressedLength;
        if (!snappy::GetUncompressedLength(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            &uncompressedLength
        )) {
            throw std::runtime_error("Snappy: Failed to get uncompressed length");
        }
        outputData.resize(uncompressedLength);
        if (!snappy::RawUncompress(
            reinterpret_cast<const char*>(inputData.data()),
            inputSize,
            reinterpret_cast<char*>(outputData.data())
        )) {
            throw std::runtime_error("Snappy CPU decompression failed");
        }
        decompSize = uncompressedLength;
    } else if (algo == ALGO_ZSTD) {
        unsigned long long uncompressedSize = ZSTD_getFrameContentSize(inputData.data(), inputSize);
        if (uncompressedSize == ZSTD_CONTENTSIZE_ERROR || uncompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Zstd: Failed to get uncompressed size");
        }
        outputData.resize(uncompressedSize);
        decompSize = ZSTD_decompress(
            outputData.data(),
            uncompressedSize,
            inputData.data(),
            inputSize
        );
        if (ZSTD_isError(decompSize)) {
            throw std::runtime_error("Zstd CPU decompression failed");
        }
    } else {
        throw std::runtime_error("Algorithm not supported for CPU decompression");
    }
    
    outputData.resize(decompSize);
    return outputData;
}

std::vector<uint8_t> decompressBatchedFormatCPU(AlgoType algo, const std::vector<uint8_t>& compressedData) {
    // Check if it's batched format
    if (compressedData.size() < sizeof(BatchedHeader)) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    BatchedHeader header;
    std::memcpy(&header, compressedData.data(), sizeof(BatchedHeader));
    
    if (header.magic != BATCHED_MAGIC) {
        // Not batched format, use CPU decompression directly
        return decompressDataCPU(algo, compressedData);
    }
    
    // It's a batched format - extract the compressed chunks and decompress with CPU
    // Use algorithm from header (auto-detect)
    AlgoType actualAlgo = static_cast<AlgoType>(header.algorithm);
    
    size_t chunk_count = header.chunkCount;
    size_t uncompressedSize = header.uncompressedSize;
    
    // Read chunk sizes
    size_t offset = sizeof(BatchedHeader);
    std::vector<uint64_t> chunkSizes64(chunk_count);
    std::memcpy(chunkSizes64.data(), compressedData.data() + offset, sizeof(uint64_t) * chunk_count);
    offset += sizeof(uint64_t) * chunk_count;
    
    // Decompress each chunk with CPU
    std::vector<uint8_t> result;
    result.reserve(uncompressedSize);
    
    for (size_t i = 0; i < chunk_count; i++) {
        size_t chunkSize = static_cast<size_t>(chunkSizes64[i]);
        std::vector<uint8_t> chunk(compressedData.begin() + offset, 
                                   compressedData.begin() + offset + chunkSize);
        
        auto decompressed = decompressDataCPU(actualAlgo, chunk);
        result.insert(result.end(), decompressed.begin(), decompressed.end());
        
        offset += chunkSize;
    }
    
    return result;
}

// ============================================================================
// CPU Compression
// ============================================================================

// Internal function that compresses in-memory archive data.
// Populates stats->computeSec and stats->writeSec when stats != nullptr.
static void compressCPUFromBuffer(AlgoType algo, const std::vector<uint8_t>& archiveData,
                                  const std::string& outputFile, uint64_t maxVolumeSize,
                                  CompressionStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    const bool verbose = isVerbose();
    if (verbose) {
        std::cout << "Using CPU compression (" << algoToString(algo) << ")...\n";
    }
    
    size_t totalSize = archiveData.size();
    if (verbose) {
        std::cout << "Archive size: " << totalSize << " bytes\n";
    }
    
    // Split into volumes if needed
    auto volumes = splitIntoVolumes(archiveData, maxVolumeSize);
    
    // If single volume, use original behavior
    if (volumes.size() == 1) {
        auto computeStart = clock::now();
        auto compressedData = compressDataCPU(algo, volumes[0]);
        auto computeEnd = clock::now();
        
        double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
        size_t compSize = compressedData.size();
        if (stats) stats->computeSec += duration;
        
        if (verbose) {
            std::cout << "Compressed size: " << compSize << " bytes\n";
            std::cout << "Ratio: " << std::fixed << std::setprecision(2) << (double)totalSize / compSize << "x\n";
            std::cout << "Time: " << duration << "s (" << (totalSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
        }
        
        // Write with BatchedHeader for algorithm detection compatibility
        // Use a simple single-chunk format
        BatchedHeader header;
        header.magic = BATCHED_MAGIC;
        header.version = BATCHED_VERSION;
        header.uncompressedSize = totalSize;
        header.chunkCount = 1;
        header.chunkSize = totalSize;
        header.algorithm = static_cast<uint32_t>(algo);
        header.reserved = 0;
        
        // Build output: header + chunk size + compressed data
        std::vector<uint8_t> outputData;
        outputData.reserve(sizeof(BatchedHeader) + sizeof(uint64_t) + compSize);
        
        // Append header
        outputData.insert(outputData.end(), 
                         reinterpret_cast<uint8_t*>(&header),
                         reinterpret_cast<uint8_t*>(&header) + sizeof(BatchedHeader));
        
        // Append chunk size
        uint64_t chunkSize64 = compSize;
        outputData.insert(outputData.end(),
                         reinterpret_cast<uint8_t*>(&chunkSize64),
                         reinterpret_cast<uint8_t*>(&chunkSize64) + sizeof(uint64_t));
        
        // Append compressed data
        outputData.insert(outputData.end(), compressedData.begin(), compressedData.end());
        
        auto writeStart = clock::now();
        writeFile(outputFile, outputData.data(), outputData.size());
        if (stats) {
            stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
            stats->outputBytes = outputData.size();
        }
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
    
    for (size_t i = 0; i < volumes.size(); i++) {
        if (verbose) {
            std::cout << "\r  Processing volume " << (i + 1) << "/" << volumes.size() << "..." << std::flush;
        }
        
        auto computeStart = clock::now();
        auto compressed = compressDataCPU(algo, volumes[i]);
        auto computeEnd = clock::now();
        
        double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
        totalDuration += duration;
        if (stats) stats->computeSec += duration;
        
        // Create volume metadata
        VolumeMetadata meta;
        meta.volumeIndex = i + 1;
        meta.compressedSize = compressed.size();
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = volumes[i].size();
        volumeMetadata.push_back(meta);
        
        uncompressedOffset += volumes[i].size();
        totalCompressedSize += compressed.size();
        
        // Write volume file
        auto volWriteStart = clock::now();
        std::string volumeFile = generateVolumeFilename(outputFile, i + 1);
        writeFile(volumeFile, compressed.data(), compressed.size());
        if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - volWriteStart).count();
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
    
    // Read first volume (re-read to splice manifest)
    auto fixupReadStart = clock::now();
    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    auto firstVolumeData = readFile(firstVolumeFile);
    if (stats) {
        stats->readSec += std::chrono::duration<double>(clock::now() - fixupReadStart).count();
    }
    
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
// ============================================================================
// Streaming CPU compression (Phase 3, mirrors compressGPUBatchedStreaming)
// ============================================================================
//
// One reusable host fillBuffer of capacity = maxVolumeSize is filled directly
// from disk (mmap'd via readFileInto) one file at a time. When it reaches
// maxVolumeSize the volume is CPU-compressed (raw compressDataCPU output, no
// batched wrapper - matches the CPU multi-volume on-disk format) and either:
//   - buffered in volume1Buffered (volume index 0) so we can prepend the
//     manifest+metadata at the end, or
//   - written straight to disk (volume index >= 1).
//
// On-disk format is unchanged from the in-memory CPU pipeline so all
// decompression paths (decompressBatchedFormatCPU auto-detects raw vs wrapped)
// are reused as-is.

static inline bool shouldStreamMultiVolume(uint64_t totalArchiveSize, uint64_t maxVolumeSize) {
    return maxVolumeSize > 0
        && maxVolumeSize != UINT64_MAX
        && totalArchiveSize > maxVolumeSize;
}

static void compressCPUStreaming(AlgoType algo,
                                 const std::vector<ArchiveEntry>& entries,
                                 const std::string& outputFile,
                                 uint64_t maxVolumeSize,
                                 ProgressCallback rawCallback,
                                 CompressionStats* stats) {
    using clock = std::chrono::steady_clock;
    auto callback = makeThrottledCallback(rawCallback);
    const bool verbose = isVerbose();
    auto opStart = clock::now();

    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    uint64_t totalFileBytes = 0;
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
        totalFileBytes += e.fileSize;
    }

    if (verbose) {
        std::cout << "Using CPU compression (" << algoToString(algo)
                  << ") [streaming]...\n";
        std::cout << "Archive size: " << totalArchiveSize << " bytes\n";
    }

    if (stats) stats->inputBytes = totalArchiveSize;

    std::vector<uint8_t> fillBuffer;
    fillBuffer.reserve(static_cast<size_t>(maxVolumeSize));

    std::vector<uint8_t> volume1Buffered;
    std::vector<VolumeMetadata> volumeMetadata;
    uint64_t totalCompressedBytes = 0;
    uint64_t uncompressedOffset = 0;
    size_t volumeIndex = 0;

    auto flushVolume = [&]() {
        auto computeStart = clock::now();
        auto compressed = compressDataCPU(algo, fillBuffer);
        if (stats) stats->computeSec += std::chrono::duration<double>(clock::now() - computeStart).count();

        VolumeMetadata meta;
        meta.volumeIndex = volumeIndex + 1;
        meta.compressedSize = compressed.size();
        meta.uncompressedOffset = uncompressedOffset;
        meta.uncompressedSize = fillBuffer.size();
        volumeMetadata.push_back(meta);

        uncompressedOffset += fillBuffer.size();
        totalCompressedBytes += compressed.size();

        if (volumeIndex == 0) {
            volume1Buffered = std::move(compressed);
        } else {
            auto writeStart = clock::now();
            std::string filename = generateVolumeFilename(outputFile, volumeIndex + 1);
            writeFile(filename, compressed.data(), compressed.size());
            if (stats) stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        }

        if (verbose) {
            std::cout << "  Volume " << (volumeIndex + 1)
                      << " flushed (" << fillBuffer.size() << " B uncompressed -> "
                      << meta.compressedSize << " B compressed)\n";
        }

        fillBuffer.resize(0);
        volumeIndex++;
    };

    auto appendBytes = [&](const uint8_t* src, uint64_t n) {
        while (n > 0) {
            uint64_t avail = maxVolumeSize - fillBuffer.size();
            if (avail == 0) {
                flushVolume();
                avail = maxVolumeSize;
            }
            uint64_t take = std::min(avail, n);
            size_t off = fillBuffer.size();
            fillBuffer.resize(off + static_cast<size_t>(take));
            std::memcpy(fillBuffer.data() + off, src, static_cast<size_t>(take));
            src += take;
            n -= take;
        }
    };

    ArchiveHeader hdr;
    hdr.magic = ARCHIVE_MAGIC;
    hdr.version = ARCHIVE_VERSION;
    hdr.fileCount = static_cast<uint32_t>(entries.size());
    hdr.reserved = 0;
    appendBytes(reinterpret_cast<const uint8_t*>(&hdr), sizeof(ArchiveHeader));

    uint64_t processedFileBytes = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];

        FileEntry fe;
        fe.pathLength = static_cast<uint32_t>(e.relativePath.size());
        fe.fileSize = e.fileSize;
        appendBytes(reinterpret_cast<const uint8_t*>(&fe), sizeof(FileEntry));
        appendBytes(reinterpret_cast<const uint8_t*>(e.relativePath.data()),
                    e.relativePath.size());

        if (e.fileSize > 0) {
            uint64_t avail = maxVolumeSize - fillBuffer.size();
            if (e.fileSize <= avail) {
                size_t writeOff = fillBuffer.size();
                fillBuffer.resize(writeOff + static_cast<size_t>(e.fileSize));
                readFileInto(e.filePath, fillBuffer.data() + writeOff, e.fileSize);
            } else {
                std::ifstream f(e.filePath, std::ios::binary);
                if (!f.is_open()) {
                    throw std::runtime_error("Failed to open input file: " + e.filePath.string());
                }
                uint64_t remaining = e.fileSize;
                while (remaining > 0) {
                    avail = maxVolumeSize - fillBuffer.size();
                    if (avail == 0) {
                        flushVolume();
                        avail = maxVolumeSize;
                    }
                    uint64_t take = std::min(avail, remaining);
                    size_t writeOff = fillBuffer.size();
                    fillBuffer.resize(writeOff + static_cast<size_t>(take));
                    if (!f.read(reinterpret_cast<char*>(fillBuffer.data() + writeOff),
                                static_cast<std::streamsize>(take))) {
                        throw std::runtime_error("Failed to read file: " + e.filePath.string());
                    }
                    remaining -= take;
                }
            }
        }

        processedFileBytes += e.fileSize;
        if (verbose) {
            std::cout << "  Adding: " << e.relativePath
                      << " (" << e.fileSize << " bytes)\n";
        }

        if (callback && totalFileBytes > 0) {
            float p = static_cast<float>(processedFileBytes) / totalFileBytes;
            BlockProgressInfo info;
            info.totalBlocks = static_cast<int>(entries.size());
            info.completedBlocks = static_cast<int>(i + 1);
            info.currentBlock = static_cast<int>(i);
            info.currentBlockSize = e.fileSize;
            info.overallProgress = p * 0.75f;
            info.currentBlockProgress = 1.0f;
            info.throughputMBps = 0.0;
            info.stage = "compressing";
            callback(info);
        }
    }

    if (!fillBuffer.empty()) {
        flushVolume();
    }

    auto writeStart = clock::now();
    VolumeManifest manifest;
    manifest.magic = VOLUME_MAGIC;
    manifest.version = VOLUME_VERSION;
    manifest.volumeCount = static_cast<uint32_t>(volumeMetadata.size());
    manifest.algorithm = static_cast<uint32_t>(algo);
    manifest.volumeSize = maxVolumeSize;
    manifest.totalUncompressedSize = totalArchiveSize;
    manifest.reserved = 0;

    std::vector<uint8_t> volume1OnDisk;
    volume1OnDisk.reserve(sizeof(VolumeManifest)
                          + sizeof(VolumeMetadata) * volumeMetadata.size()
                          + volume1Buffered.size());
    const uint8_t* mb = reinterpret_cast<const uint8_t*>(&manifest);
    volume1OnDisk.insert(volume1OnDisk.end(), mb, mb + sizeof(VolumeManifest));
    const uint8_t* vmb = reinterpret_cast<const uint8_t*>(volumeMetadata.data());
    volume1OnDisk.insert(volume1OnDisk.end(), vmb,
                         vmb + sizeof(VolumeMetadata) * volumeMetadata.size());
    volume1OnDisk.insert(volume1OnDisk.end(),
                         volume1Buffered.begin(), volume1Buffered.end());

    totalCompressedBytes = totalCompressedBytes - volume1Buffered.size() + volume1OnDisk.size();

    std::string firstVolumeFile = generateVolumeFilename(outputFile, 1);
    writeFile(firstVolumeFile, volume1OnDisk.data(), volume1OnDisk.size());
    if (stats) {
        stats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        stats->outputBytes = totalCompressedBytes;
    }

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

void compressCPU(AlgoType algo, const std::string& inputPath, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    auto entries = collectArchiveEntries(inputPath);
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
    }

    if (shouldStreamMultiVolume(totalArchiveSize, maxVolumeSize)) {
        compressCPUStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);
    } else {
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

        compressCPUFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);
    }

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void compressCPUFileList(AlgoType algo, const std::vector<std::string>& filePaths, const std::string& outputFile, uint64_t maxVolumeSize, ProgressCallback callback, CompressionStats* outStats) {
    using clock = std::chrono::steady_clock;
    auto opStart = clock::now();
    auto throttled = makeThrottledCallback(callback);

    if (isVerbose()) {
        std::cout << "Compressing file list (" << filePaths.size() << " files)...\n";
    }

    auto entries = collectArchiveEntriesFromList(filePaths);
    uint64_t totalArchiveSize = sizeof(ArchiveHeader);
    for (const auto& e : entries) {
        totalArchiveSize += sizeof(FileEntry) + e.relativePath.size() + e.fileSize;
    }

    if (shouldStreamMultiVolume(totalArchiveSize, maxVolumeSize)) {
        compressCPUStreaming(algo, entries, outputFile, maxVolumeSize, throttled, outStats);
    } else {
        auto readStart = clock::now();
        std::vector<uint8_t> archiveData = createArchiveFromFileList(filePaths, throttled);
        if (outStats) {
            outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
            outStats->inputBytes = archiveData.size();
        }

        compressCPUFromBuffer(algo, archiveData, outputFile, maxVolumeSize, outStats);
    }

    if (outStats) {
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Compression") << std::endl;
    }
}

void decompressCPU(AlgoType algo, const std::string& inputFile, const std::string& outputPath, ProgressCallback callback, CompressionStats* outStats) {
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
                std::cout << "Using CPU decompression (" << algoToString(algo) << ")...\n";
            }
            
            auto computeStart = clock::now();
            auto archiveData = decompressBatchedFormatCPU(algo, firstVolumeData);
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            size_t decompSize = archiveData.size();
            if (verbose) {
                std::cout << "Decompressed size: " << decompSize << " bytes\n";
                std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
            }

            if (outStats) {
                outStats->computeSec += duration;
                outStats->inputBytes = decompSize;
                outStats->outputBytes = firstVolumeData.size();
            }
            
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
            std::cout << "Using CPU decompression (" << algoToString(static_cast<AlgoType>(manifest.algorithm)) << ")...\n";
        }
        
        // Read volume metadata
        size_t metadataOffset = sizeof(VolumeManifest);
        std::vector<VolumeMetadata> volumeMetadata(manifest.volumeCount);
        std::memcpy(volumeMetadata.data(), firstVolumeData.data() + metadataOffset, 
                   sizeof(VolumeMetadata) * manifest.volumeCount);
        
        // Check all volumes exist
        if (volumeFiles.size() != manifest.volumeCount) {
            std::cerr << "Error: Expected " << manifest.volumeCount << " volumes, found " << volumeFiles.size() << std::endl;
            std::cerr << "Missing volumes!" << std::endl;
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
            
            auto computeStart = clock::now();
            auto decompressed = decompressBatchedFormatCPU(static_cast<AlgoType>(manifest.algorithm), volumeData);
            auto computeEnd = clock::now();
            
            double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
            totalDuration += duration;
            if (outStats) outStats->computeSec += duration;
            
            fullArchive.insert(fullArchive.end(), decompressed.begin(), decompressed.end());
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
    AlgoType detectedAlgo = detectAlgorithmFromFile(inputFile);
    if (detectedAlgo != ALGO_UNKNOWN) {
        algo = detectedAlgo;
        if (verbose) {
            std::cout << "Auto-detected algorithm from file: " << algoToString(algo) << "\n";
        }
    }
    
    if (verbose) {
        std::cout << "Using CPU decompression (" << algoToString(algo) << ")...\n";
    }
    
    auto readStart = clock::now();
    auto inputData = readFile(inputFile);
    if (outStats) {
        outStats->readSec += std::chrono::duration<double>(clock::now() - readStart).count();
        outStats->outputBytes = inputData.size();
    }
    
    auto computeStart = clock::now();
    
    // Use batched format handler which works for both batched and standard formats
    auto archiveData = decompressBatchedFormatCPU(algo, inputData);
    
    auto computeEnd = clock::now();
    double duration = std::chrono::duration<double>(computeEnd - computeStart).count();
    
    size_t decompSize = archiveData.size();
    if (verbose) {
        std::cout << "Decompressed size: " << decompSize << " bytes\n";
        std::cout << "Time: " << duration << "s (" << (decompSize / (1024.0 * 1024.0 * 1024.0)) / duration << " GB/s)\n";
    }
    if (outStats) {
        outStats->computeSec += duration;
        outStats->inputBytes = decompSize;
    }
    
    auto writeStart = clock::now();
    extractArchive(archiveData, outputPath);
    if (outStats) {
        outStats->writeSec += std::chrono::duration<double>(clock::now() - writeStart).count();
        outStats->totalSec = std::chrono::duration<double>(clock::now() - opStart).count();
        finalizeStats(*outStats);
        std::cout << formatStatsSummary(*outStats, "Decompression") << std::endl;
    }
}

} // namespace nvcomp_core


