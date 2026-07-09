// POC 1: pageable vs pinned host memory transfer bandwidth on this system.
// Gates: Phase 1b (pinned staging buffers).
#include "poc_common.h"

int main() {
    const size_t N = 256ull << 20; // 256 MB
    const int REPS = 10;

    uint8_t* d;
    CUDA_CHECK(cudaMalloc(&d, N));

    std::vector<uint8_t> pageable(N, 42);
    uint8_t* pinned;
    CUDA_CHECK(cudaHostAlloc(&pinned, N, cudaHostAllocDefault));
    memset(pinned, 42, N);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    Timer t;
    double gb = (double)N * REPS / (1ull << 30);

    // warmup
    CUDA_CHECK(cudaMemcpy(d, pinned, N, cudaMemcpyHostToDevice));

    t.start();
    for (int i = 0; i < REPS; i++)
        CUDA_CHECK(cudaMemcpy(d, pageable.data(), N, cudaMemcpyHostToDevice));
    printf("H2D pageable : %7.2f GB/s\n", gb / t.sec());

    t.start();
    for (int i = 0; i < REPS; i++)
        CUDA_CHECK(cudaMemcpyAsync(d, pinned, N, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    printf("H2D pinned   : %7.2f GB/s\n", gb / t.sec());

    t.start();
    for (int i = 0; i < REPS; i++)
        CUDA_CHECK(cudaMemcpy(pageable.data(), d, N, cudaMemcpyDeviceToHost));
    printf("D2H pageable : %7.2f GB/s\n", gb / t.sec());

    t.start();
    for (int i = 0; i < REPS; i++)
        CUDA_CHECK(cudaMemcpyAsync(pinned, d, N, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    printf("D2H pinned   : %7.2f GB/s\n", gb / t.sec());

    cudaFreeHost(pinned);
    cudaFree(d);
    return 0;
}
