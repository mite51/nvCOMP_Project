// POC 4 (riskiest, gates Phase 2): GPU-decompress a real NVBC file produced by
// the current build, using nvcompBatched*DecompressAsync.
//   - prints each algo's DecompressGetRequiredAlignments
//   - tries chunk pointers at raw back-to-back offsets first; if the API/statuses
//     reject that, retries with an aligned repack
//   - validates: decompressed blob tail == original input file bytes
//
// usage: poc_gpu_decomp <file.nvbc> <original_input_file>
#include "poc_common.h"
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

struct BatchedHeader {
    uint32_t magic, version;
    uint64_t uncompressedSize;
    uint32_t chunkCount, chunkSize, algorithm, reserved;
};
static_assert(sizeof(BatchedHeader) == 32, "layout must match nvcomp_core.hpp");
const uint32_t NVBC = 0x4E564243;
enum { ALGO_LZ4 = 0, ALGO_SNAPPY = 1, ALGO_ZSTD = 2 }; // matches AlgoType order

static void print_alignments() {
    nvcompAlignmentRequirements_t a;
    NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetRequiredAlignments(nvcompBatchedLZ4DecompressDefaultOpts, &a));
    printf("alignments LZ4    in=%zu out=%zu temp=%zu\n", a.input, a.output, a.temp);
    NVCOMP_CHECK(nvcompBatchedSnappyDecompressGetRequiredAlignments(nvcompBatchedSnappyDecompressDefaultOpts, &a));
    printf("alignments Snappy in=%zu out=%zu temp=%zu\n", a.input, a.output, a.temp);
    NVCOMP_CHECK(nvcompBatchedZstdDecompressGetRequiredAlignments(nvcompBatchedZstdDecompressDefaultOpts, &a));
    printf("alignments Zstd   in=%zu out=%zu temp=%zu\n", a.input, a.output, a.temp);
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <file.nvbc> <original>\n", argv[0]); return 1; }
    print_alignments();

    auto file = read_file(argv[1]);
    auto orig = read_file(argv[2]);
    BatchedHeader h;
    memcpy(&h, file.data(), sizeof(h));
    if (h.magic != NVBC) { fprintf(stderr, "not an NVBC file (magic %08x)\n", h.magic); return 1; }
    printf("NVBC: algo=%u chunks=%u chunkSize=%u uncompressed=%zu\n",
           h.algorithm, h.chunkCount, h.chunkSize, (size_t)h.uncompressedSize);

    const uint64_t* sizes64 = reinterpret_cast<const uint64_t*>(file.data() + sizeof(h));
    const uint8_t* blob = file.data() + sizeof(h) + h.chunkCount * sizeof(uint64_t);
    size_t blobSize = file.size() - sizeof(h) - h.chunkCount * sizeof(uint64_t);
    size_t n = h.chunkCount;

    nvcompAlignmentRequirements_t align;
    if (h.algorithm == ALGO_LZ4)
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetRequiredAlignments(nvcompBatchedLZ4DecompressDefaultOpts, &align));
    else if (h.algorithm == ALGO_SNAPPY)
        NVCOMP_CHECK(nvcompBatchedSnappyDecompressGetRequiredAlignments(nvcompBatchedSnappyDecompressDefaultOpts, &align));
    else
        NVCOMP_CHECK(nvcompBatchedZstdDecompressGetRequiredAlignments(nvcompBatchedZstdDecompressDefaultOpts, &align));

    for (int attempt = 0; attempt < 2; attempt++) {
        bool aligned = attempt == 1;
        size_t inAlign = aligned ? align.input : 1;

        // layout compressed chunks (raw offsets, or padded to inAlign)
        std::vector<size_t> off(n);
        size_t upload = 0;
        for (size_t i = 0; i < n; i++) {
            upload = (upload + inAlign - 1) / inAlign * inAlign;
            off[i] = upload;
            upload += sizes64[i];
        }
        std::vector<uint8_t> staged;
        const uint8_t* src = blob;
        if (aligned) {
            staged.resize(upload);
            size_t roff = 0;
            for (size_t i = 0; i < n; i++) {
                memcpy(staged.data() + off[i], blob + roff, sizes64[i]);
                roff += sizes64[i];
            }
            src = staged.data();
        }

        uint8_t *d_comp, *d_out;
        void **d_in_ptrs, **d_out_ptrs;
        size_t *d_in_sizes, *d_out_sizes, *d_actual;
        nvcompStatus_t* d_status;
        CUDA_CHECK(cudaMalloc(&d_comp, upload));
        CUDA_CHECK(cudaMalloc(&d_out, h.uncompressedSize));
        CUDA_CHECK(cudaMalloc(&d_in_ptrs, n * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&d_out_ptrs, n * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&d_in_sizes, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_out_sizes, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_actual, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_status, n * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMemcpy(d_comp, src, upload, cudaMemcpyHostToDevice));

        std::vector<void*> in(n), out(n);
        std::vector<size_t> insz(n), outsz(n);
        for (size_t i = 0; i < n; i++) {
            in[i] = d_comp + off[i];
            insz[i] = sizes64[i];
            out[i] = d_out + i * (size_t)h.chunkSize;
            outsz[i] = (i == n - 1) ? h.uncompressedSize - (n - 1) * (size_t)h.chunkSize
                                    : h.chunkSize;
        }
        CUDA_CHECK(cudaMemcpy(d_in_ptrs, in.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_ptrs, out.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_sizes, insz.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_sizes, outsz.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t temp = 0;
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        nvcompStatus_t rc;
        Timer t; t.start();
        if (h.algorithm == ALGO_LZ4) {
            NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync(n, h.chunkSize,
                nvcompBatchedLZ4DecompressDefaultOpts, &temp, h.uncompressedSize));
            void* d_temp; CUDA_CHECK(cudaMalloc(&d_temp, temp));
            rc = nvcompBatchedLZ4DecompressAsync(d_in_ptrs, d_in_sizes, d_out_sizes,
                d_actual, n, d_temp, temp, d_out_ptrs,
                nvcompBatchedLZ4DecompressDefaultOpts, d_status, s);
        } else if (h.algorithm == ALGO_SNAPPY) {
            NVCOMP_CHECK(nvcompBatchedSnappyDecompressGetTempSizeAsync(n, h.chunkSize,
                nvcompBatchedSnappyDecompressDefaultOpts, &temp, h.uncompressedSize));
            void* d_temp; CUDA_CHECK(cudaMalloc(&d_temp, temp));
            rc = nvcompBatchedSnappyDecompressAsync(d_in_ptrs, d_in_sizes, d_out_sizes,
                d_actual, n, d_temp, temp, d_out_ptrs,
                nvcompBatchedSnappyDecompressDefaultOpts, d_status, s);
        } else {
            NVCOMP_CHECK(nvcompBatchedZstdDecompressGetTempSizeAsync(n, h.chunkSize,
                nvcompBatchedZstdDecompressDefaultOpts, &temp, h.uncompressedSize));
            void* d_temp; CUDA_CHECK(cudaMalloc(&d_temp, temp));
            rc = nvcompBatchedZstdDecompressAsync(d_in_ptrs, d_in_sizes, d_out_sizes,
                d_actual, n, d_temp, temp, d_out_ptrs,
                nvcompBatchedZstdDecompressDefaultOpts, d_status, s);
        }
        printf("[%s] DecompressAsync rc=%d\n", aligned ? "aligned" : "raw", (int)rc);
        bool ok = rc == nvcompSuccess;
        if (ok) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<nvcompStatus_t> st(n);
            CUDA_CHECK(cudaMemcpy(st.data(), d_status, n * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (auto v : st) if (v != nvcompSuccess) bad++;
            double dt = t.sec();
            printf("[%s] statuses: %zu/%zu ok, decompress %.3f s (%.2f GB/s)\n",
                   aligned ? "aligned" : "raw", n - bad, n, dt, h.uncompressedSize / 1e9 / dt);
            ok = bad == 0;
            if (ok) {
                std::vector<uint8_t> blob_out(h.uncompressedSize);
                CUDA_CHECK(cudaMemcpy(blob_out.data(), d_out, h.uncompressedSize, cudaMemcpyDeviceToHost));
                // archive blob tail = original file bytes
                bool match = h.uncompressedSize >= orig.size() &&
                             memcmp(blob_out.data() + h.uncompressedSize - orig.size(),
                                    orig.data(), orig.size()) == 0;
                printf("[%s] content check vs original: %s\n",
                       aligned ? "aligned" : "raw", match ? "MATCH" : "MISMATCH");
                ok = match;
            }
        } else {
            cudaGetLastError(); // clear
        }
        cudaFree(d_comp); cudaFree(d_out); cudaFree(d_in_ptrs); cudaFree(d_out_ptrs);
        cudaFree(d_in_sizes); cudaFree(d_out_sizes); cudaFree(d_actual); cudaFree(d_status);
        cudaStreamDestroy(s);
        if (ok) { printf("RESULT: %s offsets work\n", aligned ? "ALIGNED-REPACK" : "RAW"); return 0; }
    }
    printf("RESULT: FAILED both raw and aligned\n");
    return 1;
}
