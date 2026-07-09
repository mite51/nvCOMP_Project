// E4 micro-benchmark: parallel small-file creation/write scaling.
// Recreates a real size histogram (one size per line) with T writer threads,
// comparing ofstream vs open/write/close.
//
//   g++ -O2 -pthread -o microbench_writers microbench_writers.cpp
//   ./microbench_writers <sizes.txt> <target_dir> <threads> <ofstream|posix>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <sizes.txt> <dir> <threads> <ofstream|posix>\n", argv[0]);
        return 1;
    }
    std::vector<size_t> sizes;
    {
        std::ifstream f(argv[1]);
        size_t s;
        while (f >> s) sizes.push_back(s);
    }
    fs::path dir(argv[2]);
    int threads = atoi(argv[3]);
    bool posix = std::string(argv[4]) == "posix";

    size_t maxSize = 0, total = 0;
    for (size_t s : sizes) { maxSize = std::max(maxSize, s); total += s; }
    std::vector<uint8_t> src(maxSize, 0xAB);

    // 256 pre-created subdirs to spread directory contention realistically.
    for (int d = 0; d < 256; d++) fs::create_directories(dir / std::to_string(d));

    std::atomic<size_t> next{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            size_t i;
            while ((i = next.fetch_add(1)) < sizes.size()) {
                fs::path p = dir / std::to_string(i & 255) / ("f" + std::to_string(i));
                if (posix) {
                    int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                    if (fd < 0) { perror("open"); exit(1); }
                    size_t left = sizes[i], off = 0;
                    while (left > 0) {
                        ssize_t w = ::write(fd, src.data() + off, left);
                        if (w <= 0) { perror("write"); exit(1); }
                        left -= w; off += w;
                    }
                    ::close(fd);
                } else {
                    std::ofstream f(p, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(src.data()), sizes[i]);
                    if (!f) { fprintf(stderr, "write failed\n"); exit(1); }
                }
            }
        });
    }
    for (auto& th : pool) th.join();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%d threads (%s): %.2f s  %.0f files/s  %.2f GB/s\n",
           threads, posix ? "posix" : "ofstream", sec, sizes.size() / sec, total / 1e9 / sec);
    return 0;
}
