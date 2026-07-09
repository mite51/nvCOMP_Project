// POC 3: sequential (H2D -> compress -> D2H, hard syncs) vs depth-3 pipelined
// sub-batches on rotating streams. Proves the read/compute overlap theory.
// 1 GB pinned-resident input, 64 MB sub-batches, LZ4.
#include "poc_common.h"
#include <nvcomp/lz4.h>

const size_t CHUNK = 64 * 1024;
const size_t TOTAL = 1ull << 30;
const size_t SUB = 64ull << 20;               // 64 MB
const size_t NCH = SUB / CHUNK;               // 1024 chunks / sub-batch
const int DEPTH = 3;

struct Slot {
    uint8_t *d_in, *d_out;
    void **d_in_ptrs, **d_out_ptrs;
    size_t *d_in_sizes, *d_out_sizes;
    void* d_temp;
    size_t temp;
    uint8_t* h_out;      // pinned
    cudaStream_t stream;
};

int main() {
    size_t max_out = 0;
    NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        CHUNK, nvcompBatchedLZ4CompressDefaultOpts, &max_out));

    uint8_t* h_in;
    CUDA_CHECK(cudaHostAlloc(&h_in, TOTAL, cudaHostAllocDefault));
    fill_mixed(h_in, TOTAL);

    Slot slots[DEPTH];
    for (auto& sl : slots) {
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync(
            NCH, CHUNK, nvcompBatchedLZ4CompressDefaultOpts, &sl.temp, SUB));
        CUDA_CHECK(cudaMalloc(&sl.d_in, SUB));
        CUDA_CHECK(cudaMalloc(&sl.d_out, max_out * NCH));
        CUDA_CHECK(cudaMalloc(&sl.d_in_ptrs, NCH * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&sl.d_out_ptrs, NCH * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&sl.d_in_sizes, NCH * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_out_sizes, NCH * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&sl.d_temp, sl.temp));
        CUDA_CHECK(cudaHostAlloc(&sl.h_out, max_out * NCH, cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreate(&sl.stream));

        std::vector<void*> in(NCH), out(NCH);
        std::vector<size_t> sz(NCH, CHUNK);
        for (size_t i = 0; i < NCH; i++) {
            in[i] = sl.d_in + i * CHUNK;
            out[i] = sl.d_out + i * max_out;
        }
        CUDA_CHECK(cudaMemcpy(sl.d_in_ptrs, in.data(), NCH * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sl.d_out_ptrs, out.data(), NCH * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sl.d_in_sizes, sz.data(), NCH * sizeof(size_t), cudaMemcpyHostToDevice));
    }
    const size_t NSUB = TOTAL / SUB;

    auto submit = [&](Slot& sl, size_t sub) {
        CUDA_CHECK(cudaMemcpyAsync(sl.d_in, h_in + sub * SUB, SUB,
                                   cudaMemcpyHostToDevice, sl.stream));
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            sl.d_in_ptrs, sl.d_in_sizes, CHUNK, NCH, sl.d_temp, sl.temp,
            sl.d_out_ptrs, sl.d_out_sizes, nvcompBatchedLZ4CompressDefaultOpts,
            nullptr, sl.stream));
        // download worst-case strided output (upper bound on D2H cost)
        CUDA_CHECK(cudaMemcpyAsync(sl.h_out, sl.d_out, max_out * NCH,
                                   cudaMemcpyDeviceToHost, sl.stream));
    };

    Timer t;
    for (int pass = 0; pass < 2; pass++) {   // pass 0 = warmup
        // (a) sequential, one slot, sync each sub-batch (mimics current code)
        t.start();
        for (size_t sub = 0; sub < NSUB; sub++) {
            submit(slots[0], sub);
            CUDA_CHECK(cudaStreamSynchronize(slots[0].stream));
        }
        double t_seq = t.sec();

        // (b) pipelined: rotate DEPTH slots, only sync a slot when reusing it
        t.start();
        for (size_t sub = 0; sub < NSUB; sub++) {
            Slot& sl = slots[sub % DEPTH];
            CUDA_CHECK(cudaStreamSynchronize(sl.stream)); // wait for slot's previous work
            submit(sl, sub);
        }
        for (auto& sl : slots) CUDA_CHECK(cudaStreamSynchronize(sl.stream));
        double t_pipe = t.sec();

        if (pass == 1) {
            printf("sequential (sync each 64MB): %.3f s  (%.2f GB/s)\n", t_seq, TOTAL / 1e9 / t_seq);
            printf("pipelined  depth-3         : %.3f s  (%.2f GB/s)  speedup %.2fx\n",
                   t_pipe, TOTAL / 1e9 / t_pipe, t_seq / t_pipe);
        }
    }
    return 0;
}
