// POC 2: batched compress throughput vs chunks-per-batch (sub-batch size).
// Gates: sub-batch default size for Phase 3. 1 GB input, 64KB chunks.
#include "poc_common.h"
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

const size_t CHUNK = 64 * 1024;
const size_t TOTAL = 1ull << 30;

template <typename GetTemp, typename GetMax, typename Compress, typename Opts>
void sweep(const char* name, uint8_t* d_data, GetTemp getTemp, GetMax getMax,
           Compress compress, Opts opts) {
    const size_t ALL = TOTAL / CHUNK; // 16384
    printf("%s: chunks/batch  ->  GB/s (compress kernel only, data resident)\n", name);
    for (size_t batch : {256ul, 512ul, 1024ul, 2048ul, 4096ul, 16384ul}) {
        size_t max_out = 0;
        NVCOMP_CHECK(getMax(CHUNK, opts, &max_out));
        size_t temp = 0;
        NVCOMP_CHECK(getTemp(batch, CHUNK, opts, &temp, batch * CHUNK));

        void *d_temp, **d_in_ptrs, **d_out_ptrs;
        size_t *d_in_sizes, *d_out_sizes;
        uint8_t* d_out;
        CUDA_CHECK(cudaMalloc(&d_temp, temp));
        CUDA_CHECK(cudaMalloc(&d_out, max_out * batch));
        CUDA_CHECK(cudaMalloc(&d_in_ptrs, batch * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&d_out_ptrs, batch * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&d_in_sizes, batch * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_out_sizes, batch * sizeof(size_t)));

        std::vector<void*> h_in(batch), h_out(batch);
        std::vector<size_t> h_sz(batch, CHUNK);
        CUDA_CHECK(cudaMemcpy(d_in_sizes, h_sz.data(), batch * sizeof(size_t), cudaMemcpyHostToDevice));

        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));

        // one warmup pass + timed pass over the full 1GB in `batch`-sized launches
        for (int pass = 0; pass < 2; pass++) {
            Timer t; t.start();
            for (size_t base = 0; base + batch <= ALL; base += batch) {
                for (size_t i = 0; i < batch; i++) {
                    h_in[i] = d_data + (base + i) * CHUNK;
                    h_out[i] = d_out + i * max_out;
                }
                CUDA_CHECK(cudaMemcpyAsync(d_in_ptrs, h_in.data(), batch * sizeof(void*), cudaMemcpyHostToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(d_out_ptrs, h_out.data(), batch * sizeof(void*), cudaMemcpyHostToDevice, s));
                NVCOMP_CHECK(compress(d_in_ptrs, d_in_sizes, CHUNK, batch, d_temp,
                                      temp, d_out_ptrs, d_out_sizes, opts, nullptr, s));
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (pass == 1)
                printf("  %6zu  ->  %7.2f GB/s\n", batch, TOTAL / 1e9 / t.sec());
        }
        cudaStreamDestroy(s);
        cudaFree(d_temp); cudaFree(d_out); cudaFree(d_in_ptrs);
        cudaFree(d_out_ptrs); cudaFree(d_in_sizes); cudaFree(d_out_sizes);
    }
}

int main() {
    std::vector<uint8_t> host(TOTAL);
    fill_mixed(host.data(), TOTAL);
    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, TOTAL));
    CUDA_CHECK(cudaMemcpy(d_data, host.data(), TOTAL, cudaMemcpyHostToDevice));

    sweep("LZ4", d_data,
          nvcompBatchedLZ4CompressGetTempSizeAsync,
          nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
          nvcompBatchedLZ4CompressAsync,
          nvcompBatchedLZ4CompressDefaultOpts);
    sweep("Zstd", d_data,
          nvcompBatchedZstdCompressGetTempSizeAsync,
          nvcompBatchedZstdCompressGetMaxOutputChunkSize,
          nvcompBatchedZstdCompressAsync,
          nvcompBatchedZstdCompressDefaultOpts);
    return 0;
}
