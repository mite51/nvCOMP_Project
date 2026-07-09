#!/usr/bin/env bash
# Build the POC micro-benchmarks against the fetched nvCOMP SDK.
set -e
cd "$(dirname "$0")"
SDK=../build/_deps/nvcomp_archive-src
NVCC=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
mkdir -p bin
for f in poc_pinned_bw poc_gather poc_batch_sweep poc_overlap poc_gpu_decomp; do
    echo "building $f"
    $NVCC -O3 -std=c++17 -arch=sm_120 -I"$SDK/include" "$f.cu" \
          -L"$SDK/lib" -lnvcomp -Xlinker -rpath -Xlinker "$(readlink -f $SDK/lib)" \
          -o bin/$f
done
echo done
