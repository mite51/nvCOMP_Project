// POC 5: per-chunk D2H gather loop (current code) vs pack-kernel + single D2H.
// Gates: Phase 1a. Simulates compressed chunks in worst-case strided output slots.
#include "poc_common.h"

__global__ void packChunksKernel(const uint8_t* src, size_t stride,
                                 const size_t* sizes, const size_t* offsets,
                                 uint8_t* dst, size_t n) {
    size_t c = blockIdx.x;
    if (c >= n) return;
    const uint8_t* s = src + c * stride;
    uint8_t* d = dst + offsets[c];
    size_t sz = sizes[c];
    // vectorized main body
    size_t nv = sz / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(s);
    for (size_t i = threadIdx.x; i < nv; i += blockDim.x) {
        uint4 v = s4[i];
        // dst may not be 16B aligned (packed offsets) -> byte store via memcpy
        uint8_t* dd = d + i * 16;
        memcpy(dd, &v, 16);
    }
    for (size_t i = nv * 16 + threadIdx.x; i < sz; i += blockDim.x) d[i] = s[i];
}

int main() {
    const size_t CHUNKS = 16384;          // = 1 GB of 64KB input chunks
    const size_t STRIDE = 65792;          // ~ max_out_bytes for 64KB LZ4
    std::vector<size_t> sizes(CHUNKS), offsets(CHUNKS);
    uint64_t x = 12345;
    size_t total = 0;
    for (size_t i = 0; i < CHUNKS; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        sizes[i] = 20000 + (x % 45000);   // "compressed" sizes 20-65KB
        offsets[i] = total;
        total += sizes[i];
    }
    printf("chunks=%zu packed=%.2f GB strided=%.2f GB\n", CHUNKS,
           total / 1e9, CHUNKS * STRIDE / 1e9);

    uint8_t* d_strided;
    CUDA_CHECK(cudaMalloc(&d_strided, CHUNKS * STRIDE));
    CUDA_CHECK(cudaMemset(d_strided, 7, CHUNKS * STRIDE));
    std::vector<uint8_t> host(total);
    Timer t;

    // (a) current: per-chunk sync D2H
    t.start();
    for (size_t i = 0; i < CHUNKS; i++)
        CUDA_CHECK(cudaMemcpy(host.data() + offsets[i], d_strided + i * STRIDE,
                              sizes[i], cudaMemcpyDeviceToHost));
    double t_loop = t.sec();
    printf("per-chunk D2H loop      : %.3f s  (%.2f GB/s)\n", t_loop, total / 1e9 / t_loop);

    // (b) pack kernel + single D2H (pinned)
    uint8_t* d_packed; size_t *d_sizes, *d_offsets;
    CUDA_CHECK(cudaMalloc(&d_packed, total));
    CUDA_CHECK(cudaMalloc(&d_sizes, CHUNKS * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, CHUNKS * sizeof(size_t)));
    uint8_t* pinned;
    CUDA_CHECK(cudaHostAlloc(&pinned, total, cudaHostAllocDefault));

    t.start();
    CUDA_CHECK(cudaMemcpy(d_sizes, sizes.data(), CHUNKS * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), CHUNKS * sizeof(size_t), cudaMemcpyHostToDevice));
    packChunksKernel<<<(unsigned)CHUNKS, 256>>>(d_strided, STRIDE, d_sizes, d_offsets, d_packed, CHUNKS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(pinned, d_packed, total, cudaMemcpyDeviceToHost));
    double t_pack = t.sec();
    printf("pack kernel + single D2H: %.3f s  (%.2f GB/s)  speedup %.1fx\n",
           t_pack, total / 1e9 / t_pack, t_loop / t_pack);
    return 0;
}
