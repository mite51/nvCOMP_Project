# Optimization Pass Results — 2026-07-07/08 (RTX 5090, CUDA 12.8, nvCOMP 5.1.0.21)

Baseline: `baseline_20260707_223300.csv` (pre-change binary).
Final: `final_20260707_231831.csv` (all phases). POC evidence: `poc_results.md`.
All 24 final round-trips byte-exact; 29 unit tests + 19 C API tests + 13
backward-compat fixture checks pass; archives are byte-identical to the old
build in both directions.

## Wall clock, baseline → final (seconds; speedup ×)

| Algo | Asset | Compress | Decompress |
|---|---|---|---|
| lz4 | med_mixed (512MB) | 1.80 → 0.64 (**2.8×**) | 0.82 → 0.84 (1.0×)¹ |
| lz4 | large_mixed (6GB) | 10.19 → 5.28 (**1.9×**) | 14.34 → 5.69 (**2.5×**) |
| lz4 | tree_med (1.5GB, 2k files) | 5.71 → 1.01 (**5.7×**) | 2.32 → 1.49 (1.6×) |
| lz4 | tree_large (4.9GB, 8k files) | 12.27 → 4.87 (**2.5×**) | 11.22 → 5.17 (**2.2×**) |
| snappy | med_mixed | 1.34 → 0.61 (2.2×) | 0.93 → 0.79 (1.2×) |
| snappy | large_mixed | 8.35 → 5.28 (1.6×) | 14.78 → 5.79 (**2.6×**) |
| snappy | tree_med | 3.69 → 1.03 (**3.6×**) | 2.56 → 1.49 (1.7×) |
| zstd | med_mixed | 1.36 → 0.50 (**2.7×**) | 0.86 → 0.62 (1.4×) |
| zstd | large_mixed | 8.75 → 4.55 (1.9×) | 13.06 → 5.32 (**2.5×**) |
| zstd | tree_med | 3.77 → 0.90 (**4.2×**) | 2.35 → 1.37 (1.7×) |
| gdeflate (untouched Manager path) | all | ~1.0× | ~1.0× |

¹ single-volume medium decompress is disk/extract-bound; its compute phase is
now GPU (0.36s CPU → ~5ms kernel) but the wall is dominated by extraction I/O.

## Peak VRAM (MB, baseline → final)

| Case | Baseline | Final |
|---|---|---|
| lz4/snappy compress, any size | 1,790–6,914 (scales with volume) | **1,404–1,476 (constant)** |
| zstd compress | 3,548–10,972 | **4,764 (constant; zstd temp-heavy)** |
| batched decompress | 0 (was CPU-only!) | 1,278–1,546 (GPU, constant) |
| gdeflate compress 6GB (Manager, untouched) | 27,424 | 27,424 |

## Peak host RSS (MB, baseline → final)

| Case | Baseline | Final |
|---|---|---|
| 512MB single-volume compress | ~2,000–2,160 | **496–944** |
| tree_med compress | ~5,830–6,030 | **496–950** |
| 6GB multi-volume compress | 6,514–7,580 | 4,339–5,853 |
| 6GB decompress | 11,494–12,450 | 7,623–9,419 |

## Throughput highlights
- tree_med compress: 274 MB/s → **1,749 MB/s** (lz4)
- GPU decompress kernels: LZ4 ~98 GB/s, Snappy ~172 GB/s, Zstd ~28 GB/s
  (vs 0.5–1.6 GB/s single-thread CPU before); end-to-end now disk-bound.
- Large jobs are now bound by disk write (compress) / extract I/O (decompress),
  not by the GPU pipeline.
