# Decompression-pass experiments (E0–E5), 2026-07-08

Target: isaac-sim archive — 13.67 GiB, 87,994 files (median 2.1 KB, 53k files ≤4 KB,
81k ≤64 KB), Zstd, 3×5 GiB volumes. RTX 5090, 24 cores, 125 GB RAM, NVMe, warm cache.

## Instrumented baseline (NVCOMP_PHASE_DEBUG=1, commit 7016637 + probes)

Total wall 11.27 s, peak RSS 17.3 GB:

| Component | Time | Verdict |
|---|---|---|
| Read phase = whole-vol001 `readFile` for the 48 B manifest | 1.24 s | **T1 confirmed** |
| `fullArchive` insert memcpy + page faults (3.6 M minor faults) | **4.54 s** | **T2 confirmed — dominant** |
| Sub-batch compressed reads (inside compute window) | 0.71 s | — |
| Waiting on the GPU (`cudaEventSynchronize`) | **0.0015 s** | pump is 100 % host-bound |
| `extractArchive` (88 k files, single thread) | 4.59 s | **T3 confirmed** |
| Volume-boundary pipeline gaps | ~1 ms total | **T4 dead — cut** |

Per-volume detail: wall {1.99, 1.90, 1.37} s ≈ insert {1.75, 1.62, 1.17} + read
{0.24, 0.27, 0.20}; wait ≈ 0 everywhere.

## E4 — parallel writer scaling (`bench/microbench_writers`, real size histogram ÷2: 44k files / 6.8 GB, same fs)

| Threads | files/s | GB/s |
|---|---|---|
| 1 (posix) | 25,115 | 4.58 |
| 2 | 44,349 | 8.09 |
| 4 | 78,707 | 14.35 |
| 8 | **102,877** | **18.76** |
| 16 | 89,075 | 16.24 |
| 1 (ofstream) | 24,587 | 4.48 |

→ writer pool of 8; open/write/close ≈ ofstream at 1 thread (syscall-bound), posix
chosen for the pool. 4.1× scaling to 8 threads.

## Decisions
- Implement T1 (slim manifest read), T2 (streaming extraction via sink + pinned pool
  + extractor thread), T3 (8-thread writer pool in the extractor).
- Cut T4 (volume gaps are ~1 ms).
- T5 (reader thread): re-measure after T2 — insert vanishing may leave reads
  (0.7 s) exposed but they already overlap GPU work; likely unnecessary.
- Projection: 11.3 s → **~2–2.5 s** (floor ≈ compressed read ∥ GPU ∥ extraction).

## Results after implementation (T1 + T2 + T3)

isaac-sim, warm cache, 3 reps: **2.80 / 2.86 s wall** (rep1 6.4 s, dirty-writeback
interference from the prior output delete), vs 11.27 s instrumented baseline —
**~4× faster**. Peak RSS **725 MB vs 17.3 GB (24× lower)**. All 88k extracted
files verified byte-identical (`diff -rq` vs the original tree).

Post-change phase-debug (E6 re-run): insert ≈ 0.0005 s, minor faults ≈ 0,
GPU wait ≈ 0.01 s, sub-batch reads 1.27 s; the pump now throttles on
extraction backpressure (pinned-pool acquire), i.e. the filesystem write path
is the new floor. **T5 (reader thread) cut** — reads are not the limiter.

Standard bench (512 MB med_mixed): decompress 0.77→0.63 s (lz4),
0.65→**0.45 s** (zstd, 1.45 GB/s end-to-end incl. extraction). Small archives
(tree_small): 0.24→**0.03 s** (slim manifest read; CPU path, no whole-file
pre-read). No compression regressions.
