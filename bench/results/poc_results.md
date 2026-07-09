# POC micro-benchmark results (2026-07-07, RTX 5090, CUDA 12.8, nvCOMP 5.1.0.21)

Evidence backing the optimization-pass design decisions. Sources: `poc/*.cu`.

## POC-1: pinned vs pageable transfers (`poc_pinned_bw`)
| Direction | Pageable | Pinned | Speedup |
|---|---|---|---|
| H2D | 13.77 GB/s | 45.81 GB/s | 3.3× |
| D2H | 12.29 GB/s | 36.33 GB/s | 3.0× |

**Decision: use pinned staging buffers for all GPU transfers (Phase 1b).**

## POC-2: compress throughput vs sub-batch size (`poc_batch_sweep`, data GPU-resident)
| chunks/batch (64KB each) | LZ4 | Zstd |
|---|---|---|
| 256 (16MB) | 6.29 GB/s | 4.14 GB/s |
| 512 (32MB) | 12.16 GB/s | 7.46 GB/s |
| 1024 (64MB) | 23.23 GB/s | 11.78 GB/s |
| 2048 (128MB) | **37.48 GB/s** | 9.46 GB/s |
| 4096 (256MB) | 28.22 GB/s | 8.22 GB/s |
| 16384 (1GB, ≈ current whole-volume batch) | 14.87 GB/s | 8.00 GB/s |

Surprise: the current one-giant-batch design is ~2.5× SLOWER than the sweet spot
even before overlap. **Decision: sub-batch default 2048 chunks (128MB) for
LZ4/Snappy, 1024 (64MB) for Zstd — keep tunable.**

## POC-3: sequential vs depth-3 pipelined sub-batches (`poc_overlap`, 1GB, LZ4, 64MB sub-batches)
- sequential (sync each sub-batch): 0.094 s (11.4 GB/s)
- pipelined depth-3: 0.045 s (24.0 GB/s) — **2.10× speedup**

**Decision: pipeline depth 3 with rotating streams (Phase 3/4).**

## POC-4: GPU batched decompression of real NVBC files (`poc_gpu_decomp`, 512MB med_mixed)
- Alignment requirements (nvcomp 5.1): LZ4 in=1/out=1/temp=1, Snappy 1/1/1, Zstd 1/1/8
  → **raw back-to-back chunk offsets are legal, no aligned repack needed**
- Results (all statuses success, content byte-exact vs original):
  - LZ4: 97.84 GB/s
  - Snappy: 172.23 GB/s
  - Zstd: 27.62 GB/s
- Baseline CPU decompress compute (from baseline CSV): ~1.2–1.6 GB/s single-threaded

**Decision: Phase 2 GPU decompress is viable as designed; expect 20–100× compute speedup.**

## POC-5: per-chunk D2H gather vs pack kernel (`poc_gather`, 16384 chunks, 0.70GB packed)
- per-chunk cudaMemcpy loop (current code): 0.163 s (4.28 GB/s)
- pack kernel + single D2H: 0.022 s (31.05 GB/s) — **7.3× speedup**

**Decision: replace all three gather loops with packChunksKernel + single D2H (Phase 1a).**

## Baseline observations (bench/results/baseline_20260707_223300.csv)
- Batched decompress peak VRAM = 0 → confirms decompression is CPU-only today.
- gdeflate (Manager) compress of 6GB peaked at **27.4GB VRAM** and 16.4GB RSS.
- Batched compress of 6GB (lz4): 6.9GB VRAM — matches the ~2.1× volume model.
- Wall throughput 300–750 MB/s across the board — I/O + sync pipeline dominated.
- Decompress of 6GB file: RSS ~12.3GB (~2× uncompressed) — whole-archive buffering.
