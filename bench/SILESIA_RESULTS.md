# nvCOMP 5.1 vs 5.3 on Silesia

Run date: 2026-09-30. Hardware: RTX 4090, driver 591.86, Windows, CUDA Toolkit
13.0. SDKs: 5.1.0.21 and 5.3.0.16, both CUDA 13 variants. The CLI builds use the
same application compression code and CPU codec dependencies. Input:
[Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia), all 12
original files, 211,938,580 bytes (202.12 MiB), verified against the published
sizes and MD5 checksums. See [the benchmark guide](SILESIA.md) for reproduction.

On this corpus, median application compression times were close between SDKs
and compression ratios were effectively unchanged. The earlier synthetic
workload's ANS compression improvement did not carry over. Stability was the
clear improvement: all upgraded processes completed cleanly, while many legacy
manager-codec decompressors crashed after writing correct extracted content.

## Complete corpus as one folder archive

Seconds below are medians of five measured runs after one warm-up. Lower is
better. Times include process startup, GPU setup, transfers, disk I/O and shutdown.
Sizes include application archive headers. The two builds alternated execution
order. All use the normal settings, 64 KiB chunks and `--no-volumes`.

| Algorithm | Compress 5.1 (s) | Compress 5.3 (s) | Decompress 5.1 (s) | Decompress 5.3 (s) | 5.3 size (MiB) | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| LZ4 | 0.414 | 0.415 | 0.431 | 0.423 | 102.69 | 1.97x |
| Snappy | 0.416 | 0.415 | 0.443 | 0.437 | 101.77 | 1.99x |
| Zstd | 0.424 | 0.425 | 0.328 | 0.330 | 70.42 | 2.87x |
| GDeflate | 0.543 | 0.552 | Failed 5/5 | 0.390 | 85.37 | 2.37x |
| ANS | 0.535 | 0.543 | Failed 5/5 | 0.450 | 127.07 | 1.59x |
| Bitcomp | 0.576 | 0.566 | Failed 4/5 | 0.481 | 167.69 | 1.21x |

Every compression invocation succeeded and its output decompressed to the
correct bytes. All 5.3 decompressions succeeded. Legacy GDeflate/ANS had no
successful measured folder-decompression process exits; Bitcomp had only one,
so its lone 0.472 s result is not presented as a comparable five-run median.

The folder archive's median byte count was identical between builds for every
algorithm except ANS, where 5.3 was 16 bytes smaller out of approximately 133 MB.
ANS, GDeflate and Zstd also showed tiny within-build size variations. There is
no material compression-ratio improvement.

All folder compression median changes were within approximately 2%. Timing
ranges overlap; occasional outliers were substantial (one 5.3 GDeflate run took
1.939 s to compress and 2.129 s to decompress). Medians, ranges and raw samples
are retained; the data do not establish a meaningful application speedup.

## The 12 files compressed separately

These totals sum each file's median time and compressed bytes. They are a
different workload from the folder test and include 12 separate CLI launches per
operation. They are not averages of per-file throughput or compression ratios.

| Algorithm | Compress 5.1 (s) | Compress 5.3 (s) | Decompress 5.1 (s) | Decompress 5.3 (s) | 5.3 total size (MiB) |
|---|---:|---:|---:|---:|---:|
| LZ4 | 1.885 | 1.920 | 0.819 | 0.845 | 102.630 |
| Snappy | 1.868 | 1.866 | 0.722 | 0.718 | 101.774 |
| Zstd | 2.101 | 2.106 | 1.376 | 1.333 | 70.406 |
| GDeflate | 2.095 | 2.135 | Incomplete | 1.882 | 85.314 |
| ANS | 2.010 | 2.027 | Incomplete | 1.909 | 131.808 |
| Bitcomp | 2.056 | 2.072 | Incomplete | 2.003 | 167.623 |

Across the 60 measured individual-file decompressions per manager codec, legacy
GDeflate succeeded 7 times, ANS 29 times and Bitcomp 17 times. Their incomplete
timings are not combined into a misleading total. All upgraded codecs succeeded
in all 60 runs. Every individual-file compression completed and verified.

## Correctness, failures and scope

- 78 cases: six algorithms times 12 individual files plus one folder workload.
- 936 round trips / 1,872 CLI invocations, including warm-ups.
- All 936 extracted filename sets and SHA-256 contents matched their sources.
- 5.1: 170 nonzero decompression exits out of 468 round trips, including warm-ups
  (141 failures in 390 measured round trips). These were Windows access
  violations, exit code 3221225477, after extraction. All were manager codecs.
- 5.3: zero process or verification failures in all 468 round trips.
- The benchmark command correctly returned nonzero for the legacy failures;
  correct extracted bytes do not turn a crashing process into a passing run.
- No CPU-fallback indications were found in the process logs.

This is a warm-cache application benchmark on historical real-world data. It
does not isolate GPU kernels, measure peak memory, control all desktop GPU/disk
activity, or prove performance on larger modern datasets or other GPUs.

Local raw results are under
`bench/results/silesia_20260930T182906_235966Z/`: `metadata.json`, `samples.jsonl`,
`summary.json`, `summary.csv` and per-process logs. That generated directory is
ignored by Git; this report and the reusable harness are retained in the source
tree. Both the author-verified corpus hashes and tested binary/runtime hashes
are recorded in the run metadata.
