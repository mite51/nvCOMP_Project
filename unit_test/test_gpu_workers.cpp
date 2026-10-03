#include "nvcomp_core.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace nvcomp_core;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<uint8_t> payload(size_t bytes) {
    std::vector<uint8_t> result(bytes);
    uint32_t state = 20261003;
    for (auto& b : result) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        b = static_cast<uint8_t>(state);
    }
    return result;
}
static void save(const fs::path& path, const std::vector<uint8_t>& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
    out.close();
    require(bool(out), "Fixture write failed");
}
static std::vector<uint8_t> load(const fs::path& path) {
    std::vector<uint8_t> result(static_cast<size_t>(fs::file_size(path)));
    std::ifstream in(path, std::ios::binary);
    if (!result.empty()) in.read(reinterpret_cast<char*>(result.data()), result.size());
    require(bool(in), "Fixture read failed");
    return result;
}
static void verify(const fs::path& source, const fs::path& restored) {
    if (fs::is_regular_file(source)) {
        require(load(source) == load(restored/source.filename()), "File round trip differs");
        return;
    }
    size_t expected = 0, actual = 0;
    for (const auto& e : fs::recursive_directory_iterator(source)) {
        if (!e.is_regular_file()) continue;
        ++expected;
        require(load(e.path()) == load(restored/fs::relative(e.path(),source)), "Folder round trip differs");
    }
    for (const auto& e : fs::recursive_directory_iterator(restored)) if (e.is_regular_file()) ++actual;
    require(expected == actual, "Unexpected extracted files");
}
static void roundtrip(AlgoType algo, const fs::path& source, const fs::path& output, uint64_t volume) {
    const auto caller = std::this_thread::get_id();
    float last = -1;
    int callbacks = 0;
    auto callback = [&](const BlockProgressInfo& info) {
        require(std::this_thread::get_id() == caller, "Callback changed thread");
        require(info.overallProgress >= last, "Progress moved backwards");
        last = info.overallProgress;
        ++callbacks;
    };
    CompressionStats stats{};
    compressGPUBatched(algo, source.string(), output.string(), volume, callback, &stats);
    require(callbacks > 0 && last == 1.0f, "Missing completion callback");
    fs::path archive = fs::exists(output) ? output : fs::path(generateVolumeFilename(output.string(),1));
    if (!fs::exists(output)) {
        std::ifstream in(archive, std::ios::binary);
        VolumeManifest manifest{};
        in.read(reinterpret_cast<char*>(&manifest), sizeof(manifest));
        require(manifest.magic == VOLUME_MAGIC, "Missing volume manifest");
        uint64_t total = 0, offset = 0;
        for (uint32_t i = 0; i < manifest.volumeCount; ++i) {
            VolumeMetadata meta{};
            in.read(reinterpret_cast<char*>(&meta), sizeof(meta));
            require(bool(in) && meta.volumeIndex == i+1 && meta.uncompressedOffset == offset,
                    "Invalid volume metadata");
            require(meta.compressedSize == fs::file_size(generateVolumeFilename(output.string(),i+1)),
                    "Manifest must include the first volume prefix in its on-disk size");
            total += meta.compressedSize;
            offset += meta.uncompressedSize;
        }
        require(total == stats.outputBytes && offset == manifest.totalUncompressedSize,
                "Volume totals disagree with stats");
    }
    fs::path restored(output.string()+".restored");
    // The unchanged CPU decoder validates compatibility of the produced container.
    decompressCPU(algo, archive.string(), restored.string());
    verify(source, restored);
}

int main() {
    fs::path root;
    try {
        if (!isCudaAvailable()) { std::cout << "CUDA unavailable; skipping\n"; return 77; }
#ifdef _WIN32
        _putenv_s("NVCOMP_SUBBATCH_MB", "1");
        _putenv_s("NVCOMP_REUSE_BUFFERS", "1");
#else
        setenv("NVCOMP_SUBBATCH_MB", "1", 1);
        setenv("NVCOMP_REUSE_BUFFERS", "1", 1);
#endif
        // Create and later remove only this newly owned fixture directory.
        fs::create_directories("output");
        root = fs::absolute(fs::path("output")/("gpu_workers_"+std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count())));
        require(fs::create_directory(root), "Fixture directory already exists");
        const fs::path source = root/"source";
        save(source/"a.bin", payload((3 << 20)+131));
        save(source/"nested"/"b.bin", payload((2 << 20)+79));
        save(source/"binary-special.bin", {0,13,10,26,255});
        save(source/"empty", {});
        int passed = 0;
        for (AlgoType algo : {ALGO_LZ4, ALGO_SNAPPY, ALGO_ZSTD}) {
            for (uint64_t volume : {0ull, 700003ull, (2ull << 20)+3}) {
                roundtrip(algo, source, root/("case"+std::to_string(passed)+".archive"), volume);
                require(compressionBufferCacheDeviceBytes() > 0
                     && compressionBufferCacheDeviceBytes() <= (4ull << 30), "Cache missing or exceeds cap");
                ++passed;
            }
        }
        save(root/"tiny.bin", {0,13,10,26,255});
        roundtrip(ALGO_ZSTD, root/"tiny.bin", root/"tiny.archive", 0);
        ++passed;
        auto first = std::async(std::launch::async, [&] {
            roundtrip(ALGO_LZ4, source, root/"concurrent-lz4.archive", 700003);
        });
        auto second = std::async(std::launch::async, [&] {
            roundtrip(ALGO_ZSTD, source, root/"concurrent-zstd.archive", 0);
        });
        first.get(); second.get(); passed += 2;

        bool failed = false;
        const auto callbackOutput = root/"callback-fail.archive";
        try {
            compressGPUBatched(ALGO_ZSTD, source.string(), callbackOutput.string(), 0,
                [](const BlockProgressInfo&) { throw std::runtime_error("callback failure test"); });
        } catch (const std::exception& e) { failed = std::string(e.what()) == "callback failure test"; }
        require(failed && !fs::exists(callbackOutput), "Callback failure did not cleanly abort");
        require(compressionBufferCacheDeviceBytes() == 0, "Failed workspace was cached");
        roundtrip(ALGO_ZSTD, source, root/"after-callback.archive", 0); ++passed;

        const auto writerOutput = root/"writer-fail.archive";
        const fs::path blocked(generateVolumeFilename(writerOutput.string(),3));
        require(fs::create_directory(blocked), "Could not create output blocker");
        failed = false;
        try { compressGPUBatched(ALGO_ZSTD, source.string(), writerOutput.string(), 1 << 20); }
        catch (const std::exception& e) { failed = std::string(e.what()).find("Failed to create output") != std::string::npos; }
        require(failed && fs::is_directory(blocked), "Writer failure did not propagate");
        require(!fs::exists(generateVolumeFilename(writerOutput.string(),2)), "Partial volume was left behind");
        require(!fs::exists(generateVolumeFilename(writerOutput.string(),1)), "Partial first volume was left behind");
        roundtrip(ALGO_ZSTD, source, root/"after-writer.archive", 0); ++passed;

        const auto truncated = root/"truncate.bin";
        const auto readerOutput = root/"reader-fail.archive";
        save(truncated, payload((13 << 20)+7));
        bool didTruncate = false;
        failed = false;
        try {
            compressGPUBatched(ALGO_ZSTD, truncated.string(), readerOutput.string(), 0,
                [&](const BlockProgressInfo&) {
                    if (!didTruncate) { fs::resize_file(truncated, 0); didTruncate = true; }
                });
        } catch (const std::exception& e) { failed = std::string(e.what()).find("Failed to read file") != std::string::npos; }
        require(didTruncate && failed && !fs::exists(readerOutput), "Reader failure did not cleanly abort");
        roundtrip(ALGO_ZSTD, source, root/"after-reader.archive", 0); ++passed;

        // Clearing while a workspace is leased must prevent it from being
        // retained when that operation completes, without interrupting the job.
        const auto clearOutput = root/"clear-active.archive";
        compressGPUBatched(ALGO_ZSTD, source.string(), clearOutput.string(), 0,
            [](const BlockProgressInfo&) { clearCompressionBufferCache(); });
        require(compressionBufferCacheDeviceBytes() == 0, "Active job repopulated cleared cache");
        decompressCPU(ALGO_ZSTD, clearOutput.string(), (root/"clear-restored").string());
        verify(source, root/"clear-restored"); ++passed;

        const auto finalizeOutput = root/"finalize-fail.archive";
        bool moved = false;
        failed = false;
        try {
            compressGPUBatched(ALGO_ZSTD, source.string(), finalizeOutput.string(), 1 << 20,
                [&](const BlockProgressInfo& info) {
                    if (!moved && info.overallProgress > 0 && info.overallProgress < 1) {
                        fs::rename(generateVolumeFilename(finalizeOutput.string(),1), root/"moved-first.archive");
                        moved = true;
                    }
                });
        } catch (const std::exception& e) {
            failed = std::string(e.what()).find("Failed to reopen first") != std::string::npos;
        }
        require(moved && failed, "Manifest finalization failure did not propagate");
        require(!fs::exists(generateVolumeFilename(finalizeOutput.string(),2)), "Finalization left partial volume");
        require(compressionBufferCacheDeviceBytes() == 0, "Finalization failure retained buffers");
        roundtrip(ALGO_ZSTD, source, root/"after-finalize.archive", 0); ++passed;
        clearCompressionBufferCache();
        require(compressionBufferCacheDeviceBytes() == 0, "Explicit cache clear failed");
        std::cout << passed << " verified round trips; 4 failure/recovery scenarios passed; cache release verified\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << " (fixtures retained at " << root << ")\n";
        return 1;
    }
}
